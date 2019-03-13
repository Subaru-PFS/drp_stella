from types import SimpleNamespace
import numpy as np
from collections import defaultdict, Counter

from lsst.pex.config import Config, ConfigurableField, ListField
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, Struct
from lsst.afw.geom import SpherePoint, averageSpherePoint, degrees

from pfs.datamodel.target import TargetData, TargetObservations
from pfs.datamodel.drp import PfsCoadd
from pfs.datamodel.masks import MaskHelper
from .mergeArms import MergeArmsTask
from .measureFluxCalibration import MeasureFluxCalibrationTask


class CoaddSpectraConfig(Config):
    """Configuration for CoaddSpectraTask"""
    mergeArms = ConfigurableField(target=MergeArmsTask, doc="Merge arms")
    measureFluxCalibration = ConfigurableField(target=MeasureFluxCalibrationTask, doc="Flux calibration")
    mask = ListField(dtype=str, default=["NO_DATA", "CR"], doc="Mask values to reject when combining")


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
            pfsConfig = ref.get("pfsConfig")
            for targ in zip(pfsConfig.catId, pfsConfig.tract, pfsConfig.patch, pfsConfig.objId):
                targets[targ].append(dataRefToTuple(ref))
        # Have target --> [exposures]; invert to [exposures] --> [targets]
        exposures = defaultdict(list)
        for targ, exps in targets.items():
            exps = tuple(exps)
            exposures[exps].append(targ)

        result = []
        for exps, targs in exposures.items():
            refList = [dataRefs[ee] for ee in exps]
            targetList = [SimpleNamespace(**dict(zip(("catId", "tract", "patch", "objId"), tt)))
                          for tt in targs]
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
        self.makeSubtask("mergeArms")
        self.makeSubtask("measureFluxCalibration")

    def run(self, dataRefList, targetList):
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
        # Split into visits, spectrographs
        visits = defaultdict(list)
        for dataRef in dataRefList:
            visits[dataRef.dataId["visit"]].append(dataRef)

        data = [self.readVisit(vv) for vv in visits.values()]

        spectra = defaultdict(list)
        for dd in data:
            for ss in dd.spectra:
                target = ss.target
                spectra[(target.catId, target.tract, target.patch, target.objId)].append(ss)

        pfsConfigList = [dataRef.get("pfsConfig") for dataRef in dataRefList]
        butler = dataRefList[0].getButler()
        for target in targetList:
            indices = [pfsConfig.selectTarget(target.catId, target.tract, target.patch, target.objId) for
                       pfsConfig in pfsConfigList]
            targetData = self.getTargetData(target, pfsConfigList, indices)
            identityList = [dataRef.dataId for dataRef in dataRefList]
            observations = self.getObservations(identityList, pfsConfigList, indices)
            spectrumList = spectra[(target.catId, target.tract, target.patch, target.objId)]
            flags = MaskHelper.fromMerge([ss.flags for ss in spectrumList])
            combination = self.combine(spectrumList, flags)
            coadd = PfsCoadd(targetData, observations, combination.wavelength, combination.flux,
                             combination.mask, combination.sky, combination.covar, combination.covar2, flags)
            butler.put(coadd, "pfsCoadd", coadd.getIdentity())

    def readVisit(self, dataRefList):
        """Read a single visit

        The input ``pfsArm`` files are read, and the 1D sky subtraction from
        ``sky1d`` is applied.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Butler data reference.

        Returns
        -------
        spectra : `list` of `pfs.datamodel.PfsObject`
            Sky-subtracted, flux-calibrated arm spectra.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration.
        """
        spectrographs = defaultdict(list)
        for dataRef in dataRefList:
            spectrographs[dataRef.dataId["spectrograph"]].append(dataRef)
        merged = self.mergeArms.run(list(spectrographs.values()))
        fluxCal = dataRefList[0].get("fluxCal")
        spectra = self.measureFluxCalibration.apply(merged.spectra, merged.pfsConfig, fluxCal)
        return Struct(spectra=spectra, pfsConfig=merged.pfsConfig)

    def getTargetData(self, target, pfsConfigList, indices):
        """Generate a ``TargetData`` for this target

        We combine the various declarations about the target in the
        ``PfsConfig``s.

        Parameters
        ----------
        target : `types.SimpleNamespace`
            Struct with target identity (with ``catId``, ``tract``, ``patch``
            and ``objId``).
        pfsConfigList : iterable of `pfs.datamodel.PfsConfig`
            List of top-end configurations.
        indices : `numpy.ndarray` of `int`
            Indices for the fiber of interest in each of the ``pfsConfigList``.

        Returns
        -------
        result : `pfs.datamodel.TargetData`
            ``TargetData`` for this target
        """
        radec = [SpherePoint(pfsConfig.ra[ii]*degrees, pfsConfig.dec[ii]*degrees) for
                 pfsConfig, ii in zip(pfsConfigList, indices)]
        radec = averageSpherePoint(radec)

        targetType = Counter([pfsConfig.targetType[ii] for pfsConfig, ii in zip(pfsConfigList, indices)])
        if len(targetType) > 1:
            self.log.warn("Multiple targetType for target %s (%s); using most common" % (target, targetType))
        targetType = targetType.most_common(1)[0][0]

        fiberMags = defaultdict(list)
        for pfsConfig, ii in zip(pfsConfigList, indices):
            for ff, mag in zip(pfsConfig.filterNames[ii], pfsConfig.fiberMag[ii]):
                fiberMags[ff].append(mag)
        for ff in fiberMags:
            mag = set(fiberMags[ff])
            if len(mag) > 1:
                self.log.warn("Multiple %s mag for target %s (%s); using average" % (ff, target, mag))
                mag = np.average(np.array(fiberMags[ff]))
            else:
                mag = mag.pop()
            fiberMags[ff] = mag

        return TargetData(target.catId, target.tract, target.patch, target.objId,
                          radec.getRa().asDegrees(), radec.getDec().asDegrees(),
                          targetType, dict(**fiberMags))

    def getObservations(self, identityList, pfsConfigList, indices):
        """Construct a list of observations of the target

        Parameters
        ----------
        identityList : iterable of `dict`
            List of sets of keyword-value pairs that identify the observation.
        pfsConfigList : iterable of `pfs.datamodel.PfsConfig`
            List of top-end configurations.
        indices : `numpy.ndarray` of `int`
            Indices for the fiber of interest in each of the ``pfsConfigList``.

        Returns
        -------
        observations : `pfs.datamodel.TargetObservations`
            Observations of the target.
        """
        fiberId = np.array([pfsConfig.fiberId[ii] for pfsConfig, ii in zip(pfsConfigList, indices)])
        pfiNominal = np.array([pfsConfig.pfiNominal[ii] for pfsConfig, ii in zip(pfsConfigList, indices)])
        pfiCenter = np.array([pfsConfig.pfiCenter[ii] for pfsConfig, ii in zip(pfsConfigList, indices)])
        return TargetObservations(identityList, fiberId, pfiNominal, pfiCenter)

    def combine(self, spectra, flags):
        """Combine spectra

        Parameters
        ----------
        spectra : iterable of `pfs.datamodel.PfsSpectrum`
            List of spectra to combine. These should already have been
            resampled to a common wavelength representation.
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
        archetype = spectra[0]
        length = archetype.length
        mask = np.zeros(length, dtype=archetype.mask.dtype)
        flux = np.zeros(length, dtype=archetype.flux.dtype)
        sky = np.zeros(length, dtype=archetype.sky.dtype)
        covar = np.zeros((3, length), dtype=archetype.covar.dtype)
        sumWeights = np.zeros(length, dtype=archetype.flux.dtype)

        for ss in spectra:
            good = ((ss.mask & ss.flags.get(*self.config.mask)) == 0) & (ss.covar[0] > 0)
            weight = np.zeros_like(flux)
            weight[good] = 1.0/ss.covar[0][good]
            flux += ss.flux*weight
            sky += ss.sky*weight
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

        return Struct(wavelength=archetype.wavelength, flux=flux, sky=sky, covar=covar,
                      mask=mask, covar2=covar2)

    def _getMetadataName(self):
        return None
