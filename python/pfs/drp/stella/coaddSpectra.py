from types import SimpleNamespace
import numpy as np
from collections import defaultdict, Counter

from lsst.pex.config import Config, ConfigurableField
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner
from lsst.afw.geom import SpherePoint, averageSpherePoint, degrees

from pfs.datamodel.target import TargetData, TargetObservations
from pfs.datamodel.drp import PfsCoadd
from .combine import CombineTask


class CoaddSpectraConfig(Config):
    """Configuration for CoaddSpectraTask"""
    combine = ConfigurableField(target=CombineTask, doc="Combine spectra")


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
        self.makeSubtask("combine")

    def run(self, dataRefList, targetList):
        """Coadd multiple observations

        Parameters
        ----------
        dataRefList : iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for all exposures observing the targets.
        targetList : iterable of `types.SimpleNamespace`
            List of target identity structs (with ``catId``, ``tract``,
            ``patch`` and ``objId``).
        """
        armList = [dataRef.get("pfsArm") for dataRef in dataRefList]
        pfsConfigList = [dataRef.get("pfsConfig") for dataRef in dataRefList]
        butler = dataRefList[0].getButler()
        for target in targetList:
            indices = [pfsConfig.selectTarget(target.catId, target.tract, target.patch, target.objId) for
                       pfsConfig in pfsConfigList]
            fiberList = [pfsConfig.fiberId[ii] for ii, pfsConfig in zip(indices, pfsConfigList)]
            targetData = self.getTargetData(target, pfsConfigList, indices)
            identityList = [dataRef.dataId for dataRef in dataRefList]
            observations = self.getObservations(identityList, pfsConfigList, indices)
            # XXX read and apply fluxCal and sky1d
            combined = self.combine.runSingle(armList, fiberList, PfsCoadd, targetData, observations)
            butler.put(combined, "pfsCoadd", combined.getIdentity())

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

    def _getMetadataName(self):
        return None
