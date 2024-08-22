from collections import defaultdict
from typing import Dict, Iterable, List, TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure
import numpy as np
import astropy.io.fits

import lsst.log

from lsst.pex.config import Field, ListField
from lsst.pipe.base import CmdLineTask, Struct, ArgumentParser, TaskRunner
from lsst.daf.persistence import ButlerDataRef

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.butlerQuantumContext import ButlerQuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from pfs.datamodel import CalibIdentity, PfsConfig

from .combineSpectra import combineSpectraSets
from .constructSpectralCalibs import setCalibHeader
from .datamodel import PfsArm, PfsFiberNorms
from .gen3 import DatasetRefList
from .utils.math import robustRms


class MeasureFiberNormsRunner(TaskRunner):
    """Runner for MeasureFiberNormsTask

    Gen2 middleware input parsing.
    """
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Produce list of targets for MeasureFiberNormsTask

        We operate on sets of arms.
        """
        groups = defaultdict(lambda: defaultdict(list))
        for ref in parsedCmd.id.refList:
            arm = ref.dataId["arm"]
            spectrograph = ref.dataId["spectrograph"]
            groups[arm][spectrograph].append(ref)

        if not parsedCmd.single:
            return [(groups[arm], kwargs) for arm in groups.keys()]

        # Want to split the groups by visit as well
        targets = []
        for spectrographRefs in groups.values():
            visitGroups = defaultdict(lambda: defaultdict(list))
            for spectrograph, refs in spectrographRefs.items():
                for ref in refs:
                    visit = ref.dataId["visit"]
                    visitGroups[visit][spectrograph].append(ref)
            targets += [(specRefs, kwargs) for specRefs in visitGroups.values()]
        return targets


class MeasureFiberNormsConnections(PipelineTaskConnections, dimensions=("instrument", "arm")):
    """Pipeline connections for MeasureFiberNormsTask

    Gen3 middleware pipeline input/output definitions.
    """
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra from arm",
        storageClass="PfsArm",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    pfsConfig = InputConnection(
        name="pfsConfig",
        doc="Top-end configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "exposure"),
    )
    fiberNorms = OutputConnection(
        name="fiberNorms_meas",
        doc="Measured fiber normalisations",
        storageClass="PfsFiberNorms",
        dimensions=("instrument", "arm"),
    )


class MeasureFiberNormsConfig(PipelineTaskConfig, pipelineConnections=MeasureFiberNormsConnections):
    """Configuration for MeasureFiberNormsTask"""
    mask = ListField(
        dtype=str,
        default=["BAD_FLAT", "CR", "SAT", "NO_DATA", "SUSPECT"],
        doc="Mask planes to exclude from fiberNorms measurement",
    )
    rejIter = Field(dtype=int, default=1, doc="Number of iterations for fiberNorms measurement")
    rejThresh = Field(dtype=float, default=4.0, doc="Threshold for rejection in fiberNorms measurement")
    insrotTol = Field(dtype=float, default=1.0, doc="Tolerance for INSROT values (degrees)")
    doCheckHash = Field(dtype=bool, default=True, doc="Check that fiberProfilesHashes are consistent?")
    doPlot = Field(dtype=bool, default=True, doc="Produce a plot of the fiber normalization values?")
    plotLower = Field(dtype=float, default=2.5, doc="Lower bound for plot (standard deviations from median)")
    plotUpper = Field(dtype=float, default=2.5, doc="Upper bound for plot (standard deviations from median)")


class MeasureFiberNormsTask(CmdLineTask, PipelineTask):
    """Task to measure fiber normalization values"""

    ConfigClass = MeasureFiberNormsConfig
    RunnerClass = MeasureFiberNormsRunner
    _DefaultName = "measureFiberNorms"

    @classmethod
    def _makeArgumentParser(cls):
        """Make an ArgumentParser"""
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="pfsArm", help="data IDs, e.g. --id exp=12345")
        parser.add_argument("--single", action="store_true", default=False, help="Run on a single visit")
        return parser

    def runDataRef(self, dataRefs: Dict[int, List[ButlerDataRef]]) -> Struct:
        """Run on a list of data references

        Gen2 middleware entry point.

        Parameters
        ----------
        dataRefs : `dict` mapping `int` to `list` of `lsst.daf.butler.DataRef`
            Data references for multiple visits, indexed by spectrograph number.

        Returns
        -------
        fiberNorms : `lsst.afw.table.SimpleCatalog`
            Fiber normalization values
        """
        pfsArms = {spec: [dataRef.get("pfsArm") for dataRef in dataRefList]
                   for spec, dataRefList in dataRefs.items()}
        pfsConfig = next(iter(dataRefs.values()))[0].get("pfsConfig")

        arm = set((dataRef.dataId["arm"] for dataRefList in dataRefs.values() for dataRef in dataRefList))
        assert len(arm) == 1, f"Multiple arms: {arm}"
        lsst.log.MDC("LABEL", arm.pop())

        result = self.run(pfsArms, pfsConfig)

        ref = dataRefs.popitem()[1][0]
        ref.put(result.fiberNorms, "fiberNorms_meas")
        if self.config.doPlot:
            ref.put(result.plot, "fiberNorms_plot")

        return result

    def runQuantum(
        self,
        butler: ButlerQuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Entry point with Gen3 butler I/O

        Parameters
        ----------
        butler : `ButlerQuantumContext`
            Data butler, specialised to operate in the context of a quantum.
        inputRefs : `InputQuantizedConnection`
            Container with attributes that are data references for the various
            input connections.
        outputRefs : `OutputQuantizedConnection`
            Container with attributes that are data references for the various
            output connections.
        """
        groups = defaultdict(list)
        for ref in DatasetRefList.fromList(inputRefs.pfsArm):
            spectrograph = ref.dataId["spectrograph"]
            groups[spectrograph].append(butler.get(ref))
        pfsConfig = butler.get(inputRefs[0].pfsConfig)

        outputs = self.run(groups, pfsConfig)
        butler.get(outputs.fiberNorms, outputRefs.fiberNorms)

    def run(self, armSpectra: Dict[int, List[PfsArm]], pfsConfig: PfsConfig) -> Struct:
        """Measure fiber normalization values

        Parameters
        ----------
        armSpectra : `dict` mapping `int` to `list` of `pfs.datamodel.PfsArm`
            pfsArm spectra, indexed by spectrograph.

        Returns
        -------
        fiberNorms : `pfs.datamodel.pfsFiberNorms.PfsFiberNorms`
            Fiber normalization values
        """
        if len(armSpectra) == 0:
            raise ValueError("No data provided")
        visits = {spec: set(aa.identity.visit for aa in armList) for spec, armList in armSpectra.items()}
        _, visitSet = visits.popitem()
        if not all(vv == visitSet for vv in visits.values()):
            raise ValueError(f"Inconsistent visit lists: {visits} vs {visitSet}")

        coadded = {spec: self.coaddSpectra(pfsArmList) for spec, pfsArmList in armSpectra.items()}
        fiberNorms = self.measureFiberNorms(coadded, visitSet)
        if self.config.doPlot:
            plot = self.plotFiberNorms(
                fiberNorms, pfsConfig, visitSet, next(iter(coadded.values())).identity.arm
            )
        else:
            plot = None
        return Struct(fiberNorms=fiberNorms, coadded=coadded, visits=visitSet, plot=plot)

    def checkSpectra(self, pfsArmList: Iterable[PfsArm], sameArm: bool = True) -> None:
        """Check that the spectra are compatible

        Parameters
        ----------
        pfsArmList : iterable of `pfs.datamodel.PfsArm`
            List of pfsArm.
        sameArm : `bool`, optional
            Spectra are supposed to be from the same arm?

        Raises
        ------
        ValueError
            If the spectra are not compatible.
        """
        archetype = pfsArmList[0]
        insrot = archetype.metadata["INSROT"]  # degrees
        fiberProfilesHash = archetype.metadata["PFS.HASH.FIBERPROFILES"]
        for pfsArm in pfsArmList[1:]:
            if sameArm:
                if not np.array_equal(archetype.fiberId, pfsArm.fiberId):
                    raise ValueError("Mismatched fiberIds")
                if self.config.doCheckHash and pfsArm.metadata["PFS.HASH.FIBERPROFILES"] != fiberProfilesHash:
                    raise ValueError(
                        f"Mismatched fiberProfilesHash for visit={pfsArm.identity.visit}: "
                        f"{pfsArm.metadata['PFS.HASH.FIBERPROFILES']} != {fiberProfilesHash}"
                    )
            insrotDiff = np.abs(pfsArm.metadata["INSROT"] - insrot) % 360  # degrees
            if min(insrotDiff, np.abs(360 - insrotDiff)) > self.config.insrotTol:
                raise ValueError(
                    f"INSROT values are not consistent: {pfsArm.metadata['INSROT']} != {insrot}"
                )

    def coaddSpectra(self, pfsArmList: List[PfsArm]) -> PfsArm:
        """Coadd spectra

        Parameters
        ----------
        pfsArmList : `list` of `pfs.datamodel.PfsArm`
            List of pfsArm.

        Returns
        -------
        spectra : `pfs.datamodel.PfsArm`
            Coadded spectra
        """
        archetype = pfsArmList[0]
        if len(pfsArmList) == 1:
            return archetype

        self.checkSpectra(pfsArmList)
        coadd = combineSpectraSets(pfsArmList, archetype.flags, self.config.mask)
        return PfsArm(
            archetype.identity,
            archetype.fiberId,
            coadd.wavelength,
            coadd.flux,
            coadd.mask,
            coadd.sky,
            coadd.norm,
            coadd.covar,
            archetype.flags,
            archetype.metadata.copy(),
        )

    def measureFiberNorms(self, spectra: Dict[int, PfsArm], visitList: Iterable[int]) -> PfsFiberNorms:
        """Measure fiber normalization values

        Parameters
        ----------
        spectra : `dict` mapping `int` to `pfs.datamodel.PfsArm`
            Spectra from which to measure fiber normalization values.
        visitList : iterable of `int`
            List of visit numbers.

        Returns
        -------
        fiberNorms : `lsst.afw.table.SimpleCatalog`
            Fiber normalization values
        """
        numFibers = sum(len(ss) for ss in spectra.values())
        heightSet = set(ss.length for ss in spectra.values())
        if len(heightSet) != 1:
            raise ValueError(f"Height mismatch: {heightSet}")
        height = heightSet.pop()
        fiberId = np.zeros(numFibers, dtype=int)
        wavelength = np.full((numFibers, height), np.nan, dtype=float)
        values = np.full((numFibers, height), np.nan, dtype=float)
        norms = np.full(numFibers, np.nan, dtype=float)
        index = 0

        self.checkSpectra(list(spectra.values()), sameArm=False)
        fiberProfilesHash = {
            ss.identity.spectrograph: ss.metadata["PFS.HASH.FIBERPROFILES"] for ss in spectra.values()
        }

        for spectrograph, ss in spectra.items():
            # Calculate the mean normalization for each fiber
            bad = (ss.mask & ss.flags.get(*self.config.mask)) != 0
            bad |= ~np.isfinite(ss.flux) | ~np.isfinite(ss.norm)
            bad |= ~np.isfinite(ss.variance) | (ss.variance == 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                flux = np.ma.masked_where(bad, ss.flux/ss.norm)
                weights = np.ma.masked_where(bad, ss.norm**2/ss.variance)
                error = np.sqrt(ss.variance)

            rejected = np.zeros_like(flux, dtype=bool)
            for _ in range(self.config.rejIter):
                median = np.ma.median(flux, axis=1)
                rejected |= np.abs(flux - median[..., np.newaxis]) > self.config.rejThresh*error
                flux.mask |= rejected

            weights.mask |= rejected
            with np.errstate(invalid="ignore"):
                average = np.ma.average(flux, weights=weights, axis=1).filled(np.nan)

            goodAverages = average[np.isfinite(average)]
            self.log.info(
                "Median normalization of spectrograph %d is %.2f +- %.2f (min %.2f, max %.2f)",
                spectrograph,
                np.median(goodAverages),
                robustRms(goodAverages),
                np.min(goodAverages),
                np.max(goodAverages),
            )

            select = slice(index, index + len(ss))
            fiberId[select] = ss.fiberId
            wavelength[select] = ss.wavelength
            with np.errstate(invalid="ignore"):
                values[select] = ss.flux/ss.norm
            norms[select] = average
            index += len(ss)

        # Ensure that the fiberIds are sorted
        indices = np.argsort(fiberId)
        fiberId = fiberId[indices]
        wavelength = wavelength[indices]
        values = values[indices]
        norms = norms[indices]

        good = np.isfinite(norms)
        goodNorms = norms[good]
        self.log.info(
            "Median normalization for all spectrographs is %.2f +- %.2f (min %.2f, max %.2f)",
            np.median(goodNorms),
            robustRms(goodNorms),
            np.min(goodNorms),
            np.max(goodNorms),
        )

        archetype = next(iter(spectra.values()))
        identity = CalibIdentity(
            obsDate=archetype.identity.obsTime,
            spectrograph=0,
            arm=archetype.identity.arm,
            visit0=min(visitList),
        )

        outputId = dict(
            arm=archetype.identity.arm,
            calibTime=archetype.identity.obsTime,
            calibDate=archetype.identity.obsTime.split("T")[0],
            visit0=min(visitList),
            # Below are required by Gen2 middleware
            filter=archetype.identity.arm,
            ccd=-1,
            spectrograph=0,
        )

        header = {}
        setCalibHeader(header, "fiberNorms", visitList, outputId)

        model = astropy.io.fits.BinTableHDU.from_columns(
            [
                astropy.io.fits.Column(name="FIBERID", format="J", array=fiberId),
                astropy.io.fits.Column(name="NORM", format="D", array=norms),
            ],
            header=astropy.io.fits.Header(cards=dict(MODELTYP="CONSTANT")),
        )

        return PfsFiberNorms(identity, fiberId, wavelength, values, fiberProfilesHash, model, header)

    def plotFiberNorms(
        self,
        fiberNorms: PfsFiberNorms,
        pfsConfig: PfsConfig,
        visitList: Iterable[int],
        arm: str,
    ) -> "Figure":
        """Plot fiber normalization values

        Parameters
        ----------
        fiberNorms : `pfs.drp.stella.datamodel.pfsFiberNorms.PfsFiberNorms`
            Fiber normalization values
        pfsConfig : `pfs.datamodel.PfsConfig`
            Configuration for the PFS system

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure containing the plot.
        """
        fig, axes = fiberNorms.plot(pfsConfig, lower=self.config.plotLower, upper=self.config.plotUpper)
        axes.set_title(f"Fiber normalization for arm={arm}\nvisits: {','.join(map(str, visitList))}")
        return fig

    def _getMetadataName(self):
        """Suppress output of task metadata"""
        return None
