from collections import defaultdict
from typing import Dict, Iterable, List

import numpy as np
import astropy.io.fits

from lsst.pex.config import Field, ListField
from lsst.pipe.base import Struct

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from pfs.datamodel import CalibIdentity

from .calibs import setCalibHeader
from .combineSpectra import combineSpectraSets
from .datamodel import PfsArm, PfsFiberNorms
from .gen3 import DatasetRefList
from .utils.math import robustRms

__all__ = ("MeasureFiberNormsTask", "ExposureFiberNormsTask")


class MeasureFiberNormsConnections(PipelineTaskConnections, dimensions=("instrument", "arm")):
    """Pipeline connections for MeasureFiberNormsTask

    Gen3 middleware pipeline input/output definitions.

    This version coadds the spectra from multiple exposures.
    """
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra from arm",
        storageClass="PfsArm",
        dimensions=("instrument", "exposure", "arm", "spectrograph"),
        multiple=True,
    )
    fiberNorms = OutputConnection(
        name="fiberNorms_calib",
        doc="Measured fiber normalisations",
        storageClass="PfsFiberNorms",
        dimensions=("instrument", "arm"),
        isCalibration=True,
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


class MeasureFiberNormsTask(PipelineTask):
    """Task to measure fiber normalization values"""

    ConfigClass = MeasureFiberNormsConfig
    _DefaultName = "measureFiberNorms"

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Entry point with Gen3 butler I/O

        Parameters
        ----------
        butler : `QuantumContext`
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

        outputs = self.run(groups)
        butler.put(outputs.fiberNorms, outputRefs.fiberNorms)

    def run(self, armSpectra: Dict[int, List[PfsArm]]) -> Struct:
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
        return Struct(fiberNorms=fiberNorms, coadded=coadded, visits=visitSet)

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
            select = slice(index, index + len(ss))
            fiberId[select] = ss.fiberId
            wavelength[select] = ss.wavelength
            values[select] = self.calculateFiberNormValue(ss)

            # Calculate the mean normalization for each fiber
            bad = (ss.mask & ss.flags.get(*self.config.mask)) != 0
            bad |= ~np.isfinite(ss.flux) | ~np.isfinite(ss.norm)
            bad |= ~np.isfinite(ss.variance) | (ss.variance == 0)
            flux = np.ma.masked_where(bad, values[select])
            with np.errstate(invalid="ignore", divide="ignore"):
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

    def calculateFiberNormValue(self, spectra: PfsArm) -> float:
        """Calculate the fiber normalization value

        Parameters
        ----------
        spectra : `pfs.datamodel.PfsArm`
            Spectra from which to measure fiber normalization values.

        Returns
        -------
        norm : `float`
            Fiber normalization value
        """
        return spectra.flux


class ExposureFiberNormsConnections(
    MeasureFiberNormsConnections, dimensions=("instrument", "arm", "exposure")
):
    """Pipeline connections for MeasureFiberNormsExposureTask

    Gen3 middleware pipeline input/output definitions.

    This version works on a single exposure.
    """
    fiberNorms = OutputConnection(
        name="fiberNorms",
        doc="Measured fiber normalisations",
        storageClass="PfsFiberNorms",
        dimensions=("instrument", "arm", "exposure"),
    )


class ExposureFiberNormsConfig(MeasureFiberNormsConfig, pipelineConnections=ExposureFiberNormsConnections):
    """Configuration for ExposureFiberNormsExposureTask"""
    requireQuartz = Field(dtype=bool, default=True, doc="Require quartz lamp data to measure fiberNorms?")


class ExposureFiberNormsTask(MeasureFiberNormsTask):
    """Measure fiberNorms for a single exposure"""
    ConfigClass = ExposureFiberNormsConfig

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Entry point with Gen3 butler I/O

        Parameters
        ----------
        butler : `QuantumContext`
            Data butler, specialised to operate in the context of a quantum.
        inputRefs : `InputQuantizedConnection`
            Container with attributes that are data references for the various
            input connections.
        outputRefs : `OutputQuantizedConnection`
            Container with attributes that are data references for the various
            output connections.
        """
        if self.config.requireQuartz:
            dataId = inputRefs.pfsArm[0].dataId
            if dataId.records["exposure"].lamps != "Quartz":
                self.log.info("Ignoring non-quartz exposure %s", dataId)
                return  # Nothing to do
        return super().runQuantum(butler, inputRefs, outputRefs)

    def calculateFiberNormValue(self, spectra: PfsArm) -> float:
        """Calculate the fiber normalization value

        Parameters
        ----------
        spectra : `pfs.datamodel.PfsArm`
            Spectra from which to measure fiber normalization values.

        Returns
        -------
        norm : `float`
            Fiber normalization value
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            return spectra.flux/spectra.norm

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
        assert len(pfsArmList) == 1
        return pfsArmList[0]
