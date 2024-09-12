from typing import Dict, Iterable, List, Mapping

import numpy as np
from collections import defaultdict, Counter

from lsst.pex.config import ConfigurableField, ListField, ConfigField
from lsst.pipe.base import Struct

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from lsst.geom import SpherePoint, averageSpherePoint, degrees

from pfs.datamodel import Target, Observations, PfsConfig, Identity
from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.pfsConfig import TargetType, FiberStatus
from pfs.drp.stella.datamodel.drp import PfsArm

from .datamodel import PfsObject, PfsSingle
from .fluxCalibrate import calibratePfsArm
from .mergeArms import WavelengthSamplingConfig
from .FluxTableTask import FluxTableTask
from .utils import getPfsVersions
from .lsf import Lsf, warpLsf, coaddLsf
from .gen3 import DatasetRefList, zipDatasetRefs

__all__ = ("CoaddSpectraConfig", "CoaddSpectraTask")


class SetWithNaN:
    """A subset of set which includes at most one NaN, despite the fact that NaN != NaN"""

    def __init__(self, iterable):
        """Construct a set from iterable with at most one NaN"""
        self.__set = set()
        sawNaN = False

        for x in iterable:
            if np.isnan(x):
                if not sawNaN:
                    self.__set.add(x)
                    sawNaN = True
            else:
                self.add(x)

    def add(self, val):
        if np.isnan(val) and np.isnan(list(self)).any():
            return

        self.__set.add(val)

    def __len__(self):
        return self.__set.__len__()

    def __repr__(self):
        return self.__set.__repr__()

    def pop(self):
        return self.__set.pop()


class CoaddSpectraConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "cat_id"),
):
    """Connections for CoaddSpectraTask"""

    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "exposure"),
        multiple=True,
    )
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra",
        storageClass="PfsArm",
        dimensions=("instrument", "exposure", "arm", "spectrograph"),
        multiple=True,
    )
    pfsArmLsf = InputConnection(
        name="pfsArmLsf",
        doc="1d line-spread function for extracted spectra",
        storageClass="LsfDict",
        dimensions=("instrument", "exposure", "arm", "spectrograph"),
        multiple=True,
    )
    sky1d = InputConnection(
        name="sky1d",
        doc="1d sky model",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "exposure", "arm", "spectrograph"),
        multiple=True,
    )
    fluxCal = InputConnection(
        name="fluxCal",
        doc="Flux calibration model",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "exposure"),
        multiple=True,
    )

    pfsObject = OutputConnection(
        name="pfsObject",
        doc="Flux-calibrated coadded object spectrum",
        storageClass="PfsObject",
        dimensions=("instrument", "cat_id", "obj_id"),
        multiple=True,
    )
    pfsObjectLsf = OutputConnection(
        name="pfsObjectLsf",
        doc="Line-spread function for pfsObject",
        storageClass="Lsf",
        dimensions=("instrument", "cat_id", "obj_id"),
        multiple=True,
    )


class CoaddSpectraConfig(PipelineTaskConfig, pipelineConnections=CoaddSpectraConnections):
    """Configuration for CoaddSpectraTask"""
    wavelength = ConfigField(dtype=WavelengthSamplingConfig, doc="Wavelength configuration")
    mask = ListField(dtype=str, default=["NO_DATA", "BAD_SKY", "BAD_FLUXCAL", "BAD_FIBERNORMS"],
                     doc="Mask values to reject when combining")
    fluxTable = ConfigurableField(target=FluxTableTask, doc="Flux table")


class CoaddSpectraTask(PipelineTask):
    """Coadd multiple observations"""
    _DefaultName = "coaddSpectra"
    ConfigClass = CoaddSpectraConfig

    fluxTable: FluxTableTask

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fluxTable")

    def run(self, data: Mapping[Identity, Struct]) -> Struct:
        """Coadd multiple observations

        data : `dict` mapping `Identity` to `Struct`
            Input data. Each element contains data for a single spectrograph
            arm, with attributes:
            - ``identity`` (`Identity`): identity of the data.
            - ``pfsArm`` (`PfsArm`): extracted spectra from spectrograph arm.
            - ``pfsArmLsf`` (`LsfDict`): line-spread function for ``pfsArm``.
            - ``sky1d`` (`FocalPlaneFunction`): 1d sky subtraction model.
            - ``fluxCal`` (`FocalPlaneFunction`): flux calibration solution.
            - ``pfsConfig`` (`PfsConfig`): PFS fiber configuration.

        Returns
        -------
        pfsObject : `dict` mapping `Target` to `PfsObject`
            Coadded spectra, indexed by target.
        pfsObjectLsf : `LsfDict`
            Line-spread functions for coadded spectra, indexed by target.
        """
        targetTypes = (TargetType.SCIENCE, TargetType.FLUXSTD, TargetType.SKY)
        targetSources = defaultdict(list)
        for identity, dd in data.items():
            for target in dd.pfsConfig.select(fiberStatus=FiberStatus.GOOD, targetType=targetTypes):
                targetSources[target].append(identity)

        pfsObject: Dict[Target, PfsObject] = {}
        pfsObjectLsf: Dict[Target, Lsf] = {}
        for target, sources in targetSources.items():
            result = self.process(target, {identity: data[identity] for identity in sources})
            pfsObject[target] = result.pfsObject
            pfsObjectLsf[target] = result.pfsObjectLsf

        return Struct(pfsObject=pfsObject, pfsObjectLsf=pfsObjectLsf)

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Entry point with butler I/O

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
        catId = set(ref.dataId["cat_id"] for ref in outputRefs.pfsObject)
        assert len(catId) == 1, "All pfsObject must have the same cat_id"
        catId = catId.pop()

        data: Dict[Identity, Struct] = {}
        for pfsConfigRef, pfsArmRef, pfsArmLsfRef, sky1dRef, fluxCalRef in zipDatasetRefs(
            DatasetRefList.fromList(inputRefs.pfsConfig),
            DatasetRefList.fromList(inputRefs.pfsArm),
            DatasetRefList.fromList(inputRefs.pfsArmLsf),
            DatasetRefList.fromList(inputRefs.sky1d),
            DatasetRefList.fromList(inputRefs.fluxCal),
        ):
            dataId = pfsArmRef.dataId.full
            expId = dataId["exposure"]
            arm = dataId["arm"]
            identity = Identity(
                visit=expId,
                arm=arm,
                spectrograph=dataId["spectrograph"],
                pfsDesignId=dataId["pfs_design_id"]
            )
            pfsConfig: PfsConfig = butler.get(pfsConfigRef)
            pfsArm: PfsArm = butler.get(pfsArmRef).select(pfsConfig, catId=catId)
            data[identity] = Struct(
                identity=identity,
                pfsArm=pfsArm,
                pfsArmLsf=butler.get(pfsArmLsfRef),
                sky1d=butler.get(sky1dRef),
                fluxCal=butler.get(fluxCalRef),
                pfsConfig=pfsConfig.select(fiberId=pfsArm.fiberId),
            )

        outputs = self.run(data)

        pfsObjectRef = {(ref.dataId["cat_id"], ref.dataId["obj_id"]): ref for ref in outputRefs.pfsObject}
        pfsObjectLsfRef = {
            (ref.dataId["cat_id"], ref.dataId["obj_id"]): ref for ref in outputRefs.pfsObjectLsf
        }

        for target in outputs.pfsObject:
            targetId = (target.catId, target.objId)
            if targetId not in pfsObjectRef:
                raise RuntimeError(f"Missing output data reference for {target}")

            butler.put(outputs.pfsObject[target], pfsObjectRef[targetId])
            butler.put(outputs.pfsObjectLsf[target], pfsObjectLsfRef[targetId])

    def getTarget(self, target: Target, pfsConfigList: List[PfsConfig]) -> Target:
        """Generate a fully-populated `Target` for this target

        We combine the various declarations about the target in the
        ``PfsConfig``s, ensuring the output `Target` has everything it needs
        (e.g., ``targetType``, ``fiberFlux``).

        Parameters
        ----------
        target : `Target`
            Basic identity of target (including ``catId`` and ``objId``).
        pfsConfigList : iterable of `PfsConfig`
            List of top-end configurations. This should include only the target
            of interest.

        Returns
        -------
        result : `Target`
            Fully-populated ``Target``.
        """
        if any(len(cfg) != 1 for cfg in pfsConfigList):
            raise RuntimeError("Multiple fibers included in pfsConfig")
        radec = averageSpherePoint([SpherePoint(cfg.ra[0]*degrees, cfg.dec[0]*degrees) for
                                    cfg in pfsConfigList])

        targetType = Counter([cfg.targetType[0] for cfg in pfsConfigList])
        if len(targetType) > 1:
            self.log.warn("Multiple targetType for target %s (%s); using most common" % (target, targetType))
        targetType = targetType.most_common(1)[0][0]

        fiberFlux = defaultdict(list)
        for pfsConfig in pfsConfigList:
            for ff, flux in zip(pfsConfig.filterNames[0], pfsConfig.fiberFlux[0]):
                fiberFlux[ff].append(flux)
        for ff in fiberFlux:
            flux = SetWithNaN(fiberFlux[ff])
            if len(flux) > 1:
                self.log.warn("Multiple %s flux for target %s (%s); using average" % (ff, target, flux))
                flux = np.average(np.array(fiberFlux[ff]))
            else:
                flux = flux.pop()
            fiberFlux[ff] = flux

        return Target(target.catId, target.tract, target.patch, target.objId,
                      radec.getRa().asDegrees(), radec.getDec().asDegrees(),
                      targetType, dict(**fiberFlux))

    def getObservations(self, dataIdList: Iterable[Identity], pfsConfigList: Iterable[PfsConfig]
                        ) -> Observations:
        """Construct a list of observations of the target

        Parameters
        ----------
        dataIdList : iterable of `Identity`
            List of structs that identify the observation, containing ``visit``,
            ``arm`` and ``spectrograph``.
        pfsConfigList : iterable of `pfs.datamodel.PfsConfig`
            List of top-end configurations. This should include only the target
            of interest.

        Returns
        -------
        observations : `Observations`
            Observations of the target.
        """
        if any(len(cfg) != 1 for cfg in pfsConfigList):
            raise RuntimeError("Multiple fibers included in pfsConfig")
        visit = np.array([dataId.visit for dataId in dataIdList])
        arm = [dataId.arm for dataId in dataIdList]
        spectrograph = np.array([dataId.spectrograph for dataId in dataIdList])
        pfsDesignId = np.array([pfsConfig.pfsDesignId for pfsConfig in pfsConfigList])
        fiberId = np.array([pfsConfig.fiberId[0] for pfsConfig in pfsConfigList])
        pfiNominal = np.array([pfsConfig.pfiNominal[0] for pfsConfig in pfsConfigList])
        pfiCenter = np.array([pfsConfig.pfiCenter[0] for pfsConfig in pfsConfigList])
        return Observations(visit, arm, spectrograph, pfsDesignId, fiberId, pfiNominal, pfiCenter)

    def getSpectrum(self, target: Target, data: Struct) -> PfsSingle:
        """Return a calibrated spectrum for the nominated target

        Parameters
        ----------
        target : `Target`
            Target of interest.
        data : `Struct`
            Data for a pfsArm containing the object of interest.

        Returns
        -------
        spectrum : `PfsSingle`
            Calibrated spectrum of the target.
        """
        spectrum = data.pfsArm.select(data.pfsConfig, catId=target.catId, objId=target.objId)
        spectrum = calibratePfsArm(spectrum, data.pfsConfig, data.sky1d, data.fluxCal)
        return spectrum.extractFiber(PfsSingle, data.pfsConfig, spectrum.fiberId[0])

    def process(self, target: Target, data: Dict[Identity, Struct]) -> Struct:
        """Generate coadded spectra for a single target

        Parameters
        ----------
        target : `Target`
            Target for which to generate coadded spectra.
        data : `dict` mapping `Identity` to `Struct`
            Data from which to generate coadded spectra. These are the results
            from the ``readData`` method.

        Returns
        -------
        pfsObject : `PfsObject`
            Coadded spectrum.
        pfsObjectLsf : `Lsf`
            Line-spread function for coadded spectrum.
        """
        pfsConfigList = [dd.pfsConfig.select(catId=target.catId, objId=target.objId) for dd in data.values()]
        target = self.getTarget(target, pfsConfigList)
        observations = self.getObservations(data.keys(), pfsConfigList)

        spectra = [self.getSpectrum(target, dd) for dd in data.values()]
        lsfList = [dd.pfsArmLsf for dd in data.values()]
        flags = MaskHelper.fromMerge([ss.flags for ss in spectra])
        combination = self.combine(spectra, lsfList, flags)
        fluxTable = self.fluxTable.run([dd.getDict() for dd in data.keys()], spectra)

        coadd = PfsObject(target, observations, combination.wavelength, combination.flux,
                          combination.mask, combination.sky, combination.covar, combination.covar2, flags,
                          getPfsVersions(), fluxTable)
        return Struct(pfsObject=coadd, pfsObjectLsf=combination.lsf)

    def combine(self, spectraList, lsfList, flags):
        """Combine spectra

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsFiberArray`
            List of spectra to combine for each visit.
        lsfList : iterable of LSF (type TBD)
            List of line-spread functions for each arm for each visit.
        flags : `pfs.datamodel.MaskHelper`
            Mask interpreter, for identifying bad pixels.

        Returns
        -------
        flux : `numpy.ndarray` of `float`
            Flux measurements for combined spectrum.
        sky : `numpy.ndarray` of `float`
            Sky measurements for combined spectrum.
        covar : `numpy.ndarray` of `float`
            Covariance matrix for combined spectrum.
        mask : `numpy.ndarray` of `int`
            Mask for combined spectrum.
        """
        # First, resample to a common wavelength sampling
        wavelength = self.config.wavelength.wavelength
        resampled = []
        resampledLsf = []
        for spectrum, lsf in zip(spectraList, lsfList):
            fiberId = spectrum.observations.fiberId[0]
            resampled.append(spectrum.resample(wavelength))
            resampledLsf.append(warpLsf(lsf[fiberId], spectrum.wavelength, wavelength))

        # Now do a weighted coaddition
        archetype = resampled[0]
        length = archetype.length
        mask = np.zeros(length, dtype=archetype.mask.dtype)
        flux = np.zeros(length, dtype=archetype.flux.dtype)
        sky = np.zeros(length, dtype=archetype.sky.dtype)
        covar = np.zeros((3, length), dtype=archetype.covar.dtype)
        sumWeights = np.zeros(length, dtype=archetype.flux.dtype)

        for ss in resampled:
            weight = np.zeros_like(flux)
            with np.errstate(invalid="ignore", divide="ignore"):
                good = ((ss.mask & ss.flags.get(*self.config.mask)) == 0) & (ss.covar[0] > 0)
                weight[good] = 1.0/ss.covar[0][good]
                flux[good] += ss.flux[good]*weight[good]
                sky[good] += ss.sky[good]*weight[good]
                mask[good] |= ss.mask[good]
                sumWeights += weight

        good = sumWeights > 0
        flux[good] /= sumWeights[good]
        sky[good] /= sumWeights[good]
        covar[0][good] = 1.0/sumWeights[good]
        covar[0][~good] = np.inf
        covar[1:2] = np.where(good, 0.0, np.inf)
        mask[~good] = flags["NO_DATA"]
        covar2 = np.zeros((1, 1), dtype=archetype.covar.dtype)
        lsf = coaddLsf(resampledLsf)

        return Struct(wavelength=archetype.wavelength, flux=flux, sky=sky, covar=covar,
                      mask=mask, covar2=covar2, lsf=lsf)

    def _getMetadataName(self):
        return None
