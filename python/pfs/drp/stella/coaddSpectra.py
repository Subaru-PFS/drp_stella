from typing import Dict, List

import numpy as np
from collections import defaultdict, Counter

from lsst.pex.config import Config, ConfigurableField, ListField, ConfigField
from lsst.daf.persistence import Butler
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, Struct
from lsst.geom import SpherePoint, averageSpherePoint, degrees

from pfs.datamodel import Target, Observations, PfsConfig, Identity
from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.pfsConfig import TargetType, FiberStatus

from .datamodel import PfsObject, PfsFiberArraySet, PfsFiberArray
from .fluxCalibrate import calibratePfsArm
from .mergeArms import WavelengthSamplingConfig
from .FluxTableTask import FluxTableTask
from .utils import getPfsVersions
from .lsf import warpLsf, coaddLsf

__all__ = ("CoaddSpectraConfig", "CoaddSpectraTask")


class CoaddSpectraConfig(Config):
    """Configuration for CoaddSpectraTask"""
    wavelength = ConfigField(dtype=WavelengthSamplingConfig, doc="Wavelength configuration")
    mask = ListField(dtype=str, default=["NO_DATA", "CR", "BAD_SKY", "BAD_FLUXCAL", "INTRP", "SAT"],
                     doc="Mask values to reject when combining")
    fluxTable = ConfigurableField(target=FluxTableTask, doc="Flux table")


class CoaddSpectraRunner(TaskRunner):
    """Runner for CoaddSpectraTask"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Produce list of targets for CoaddSpectraTask

        We want to operate on all objects within a list of exposures.
        """
        targets = defaultdict(list)  # target --> [exposures]
        if len(parsedCmd.id.refList) == 0:
            raise RuntimeError("No inputs found")
        keys = sorted(list(parsedCmd.id.refList[0].dataId.keys()))

        def dataRefToTuple(ref):
            return tuple([ref.dataId[kk] for kk in keys])

        dataRefs = {dataRefToTuple(ref): ref for ref in parsedCmd.id.refList}
        for ref in parsedCmd.id.refList:
            if not all(ref.datasetExists(dataset) for dataset in ("pfsArm", "sky1d", "fluxCal")):
                continue
            pfsConfig = ref.get("pfsConfig")
            pfsConfig = pfsConfig.select(spectrograph=ref.dataId["spectrograph"])
            for index, fiberStatus in enumerate(pfsConfig.fiberStatus):
                if fiberStatus != FiberStatus.GOOD:
                    continue
                targ = Target.fromPfsConfig(pfsConfig, index)
                if targ.targetType not in (TargetType.SCIENCE, TargetType.FLUXSTD, TargetType.SKY):
                    continue
                targets[targ].append(dataRefToTuple(ref))
        # Have target --> [exposures]; invert to [exposures] --> [targets]
        exposures = defaultdict(list)
        for targ, exps in targets.items():
            exps = tuple(exps)
            exposures[exps].append(targ)

        result = []
        for exps, targetList in exposures.items():
            refList = [dataRefs[ee] for ee in exps]
            result.append((refList, dict(targetList=targetList, **kwargs)))

        return result


class CoaddSpectraTask(CmdLineTask):
    """Coadd multiple observations"""
    _DefaultName = "coaddSpectra"
    ConfigClass = CoaddSpectraConfig
    RunnerClass = CoaddSpectraRunner

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="raw", help="data IDs, e.g. --id exp=12345..23456")
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fluxTable")

    def runDataRef(self, dataRefList, targetList):
        """Coadd multiple observations

        We base the coaddition on the ``pfsArm`` files because that data hasn't
        been interpolated. Of course, at the moment we immediately interpolate,
        but we may choose not to in the future.

        Parameters
        ----------
        dataRefList : iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for all exposures observing the targets.
        targetList : iterable of `types.SimpleNamespace`
            List of target identity structs (with ``catId``, ``tract``,
            ``patch`` and ``objId``).
        """
        dataRefDict = {Identity.fromDict(dataRef.dataId): dataRef for dataRef in dataRefList}
        data = {}
        targetSources = defaultdict(list)
        for dataId, dataRef in dataRefDict.items():
            data[dataId] = self.readArm(dataRef)
            for target in data[dataId].pfsConfig:
                targetSources[target].append(dataId)

        butler = dataRefList[0].getButler()
        for target in targetList:
            self.process(butler, target, {dataId: data[dataId] for dataId in targetSources[target]})

    def readArm(self, dataRef) -> Struct:
        """Read a single visit+spectrograph

        The input ``pfsArm`` and corresponding calibrations are read.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Butler data reference.

        Returns
        -------
        dataId : `Identity`
            Identifier for the arm.
        pfsArm : `PfsArm`
            Spectra from the arm.
        lsf : `pfs.drp.stella.Lsf`
            Line-spread function for pfsArm.
        sky1d : `pfs.drp.stella.FocalPlaneFunction`
            1D sky model.
        fluxCal : `pfs.drp.stella.FocalPlaneFunction`
            Flux calibration.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, including only the fibers in the pfsArm.
        """
        pfsArm = dataRef.get("pfsArm")
        return Struct(
            dataId=Identity.fromDict(dataRef.dataId),
            pfsArm=pfsArm,
            lsf=dataRef.get("pfsArmLsf"),
            sky1d=dataRef.get("sky1d"),
            fluxCal=dataRef.get("fluxCal"),
            pfsConfig=dataRef.get("pfsConfig").select(fiberId=pfsArm.fiberId),
        )

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
            flux = set(fiberFlux[ff])
            if len(flux) > 1:
                self.log.warn("Multiple %s flux for target %s (%s); using average" % (ff, target, flux))
                flux = np.average(np.array(fiberFlux[ff]))
            else:
                flux = flux.pop()
            fiberFlux[ff] = flux

        return Target(target.catId, target.tract, target.patch, target.objId,
                      radec.getRa().asDegrees(), radec.getDec().asDegrees(),
                      targetType, dict(**fiberFlux))

    def getObservations(self, dataIdList: List[Identity], pfsConfigList: List[PfsConfig]
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

    def getSpectrum(self, target: Target, data: Struct) -> PfsFiberArraySet:
        """Return a calibrated spectrum for the nominated target

        Parameters
        ----------
        target : `Target`
            Target of interest.
        data : `Struct`
            Data for a pfsArm containing the object of interest.

        Returns
        -------
        spectrum : `PfsFiberArray`
            Calibrated spectrum of the target.
        """
        spectrum = data.pfsArm.select(data.pfsConfig, catId=target.catId, objId=target.objId)
        spectrum = calibratePfsArm(spectrum, data.pfsConfig, data.sky1d, data.fluxCal)
        return spectrum.extractFiber(PfsFiberArray, data.pfsConfig, spectrum.fiberId[0])

    def process(self, butler: Butler, target: Target, data: Dict[Identity, Struct]):
        """Generate coadded spectra for a single target

        Parameters
        ----------
        butler : `Butler`
            Data butler.
        target : `Target`
            Target for which to generate coadded spectra.
        data : `dict` mapping `Identity` to `Struct`
            Data from which to generate coadded spectra. These are the results
            from the ``readData`` method.
        """
        pfsConfigList = [dd.pfsConfig.select(catId=target.catId, objId=target.objId) for dd in data.values()]
        target = self.getTarget(target, pfsConfigList)
        observations = self.getObservations(data.keys(), pfsConfigList)

        spectra = [self.getSpectrum(target, dd) for dd in data.values()]
        lsfList = [dd.lsf for dd in data.values()]
        flags = MaskHelper.fromMerge([ss.flags for ss in spectra])
        combination = self.combine(spectra, lsfList, flags)
        fluxTable = self.fluxTable.run([dd.getDict() for dd in data.keys()], spectra, flags)

        coadd = PfsObject(target, observations, combination.wavelength, combination.flux,
                          combination.mask, combination.sky, combination.covar, combination.covar2, flags,
                          getPfsVersions(), fluxTable)
        butler.put(coadd, "pfsObject", coadd.getIdentity())
        butler.put(combination.lsf, "pfsObjectLsf", coadd.getIdentity())

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
            resampled.append(spectrum.resample(wavelength, jacobian=False))  # Flux-calibrated --> no jacobian
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
