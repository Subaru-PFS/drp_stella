from typing import List

import yaml
import numpy as np

from lsst.pipe.base import CmdLineTask, TaskRunner, ArgumentParser
from lsst.pex.config import Config, ConfigurableField, Field

from pfs.datamodel import TargetType
from .fiberProfileSet import FiberProfileSet
from .normalizeFiberProfiles import NormalizeFiberProfilesTask


class MergeFiberProfilesConfig(Config):
    """Configuration for MergeFiberProfilesTask"""
    fiberInfluence = Field(dtype=int, default=3,
                           doc="Number of fibers around an illuminated fiber whose profiles are influenced")
    replaceNearest = Field(dtype=int, default=3, doc="Number of nearest good fibers to use for replacement")
    normalize = ConfigurableField(target=NormalizeFiberProfilesTask, doc="Normalize fiber profiles")


class MergeFiberProfilesRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Produce list of targets for MergeFiberProfilesTask

        We want to operate on everything all at once.
        """
        if parsedCmd.profiles is None:
            return []
        numProfiles = len(parsedCmd.profiles)
        if parsedCmd.visits is None or len(parsedCmd.visits) != numProfiles:
            raise ValueError("Number of profiles and visits must match")
        if parsedCmd.darkVisits is None or len(parsedCmd.darkVisits) != numProfiles:
            raise ValueError("Number of profiles and darkVisits must match")

        kwargs.update(
            filenames=parsedCmd.profiles,
            visits=parsedCmd.visits,
            darkVisits=parsedCmd.darkVisits,
            badFibersFile=parsedCmd.badFibersFile,
        )
        return [(parsedCmd.id.refList, kwargs)]


class MergeFiberProfilesTask(CmdLineTask):
    """Merge fiber profiles"""
    ConfigClass = MergeFiberProfilesConfig
    _DefaultName = "mergeFiberProfiles"
    RunnerClass = MergeFiberProfilesRunner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("normalize")

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "raw", help="data ID for norm, e.g., visit=12345 arm=r spectrograph=1")
        parser.add_argument("--profiles", nargs="+", help="List of fiber profile filenames to merge")
        parser.add_argument("--visits", nargs="+", type=int,
                            help="Corresponding list of visit numbers (one per profile)")
        parser.add_argument("--darkVisits", nargs="+", type=int,
                            help="Corresponding dark visit numbers (one per profile)")
        parser.add_argument(
            "--badFibersFile",
            type=str,
            help=("YAML file containing list of bad fibers indexed by arm+spectrograph "
                  "(e.g., b1: [1, 2, 3])"),
        )
        return parser

    def runDataRef(
        self,
        dataRefList,
        filenames: List[str],
        visits: List[int],
        darkVisits: List[int],
        badFibersFile: str = None,
    ) -> FiberProfileSet:
        """Merge fiber profiles"""

        badFibers = None
        if badFibersFile is not None:
            badFibersYaml = yaml.safe_load(open(badFibersFile))
            camera = f"{dataRefList[0].dataId['arm']}{dataRefList[0].dataId['spectrograph']}"
            if camera in badFibersYaml:
                badFibers = badFibersYaml[camera]
                self.log.info("Will replace manually-designated bad fibers: %s", badFibers)

        spectrograph = dataRefList[0].dataId["spectrograph"]
        profiles = self.mergeFiberProfiles(
            dataRefList[0].getButler(), filenames, visits, darkVisits, spectrograph, badFibers
        )
        self.normalize.run(profiles, dataRefList, [])
        return profiles

    def mergeFiberProfiles(
        self,
        butler,
        filenames: List[str],
        visits: List[int],
        darkVisits: List[int],
        spectrograph: int,
        badFibers: List[int] = None,
    ) -> FiberProfileSet:
        """Merge fiber profiles

        Parameters
        ----------
        butler : `lsst.daf.persistence.Butler`
            Data butler.
        filenames : list of `str`
            List of fiber profile filenames to merge.
        visits : list of `int`
            Corresponding list of visit numbers (one per profile).
        darkVisits : list of `int`
            Corresponding dark visit numbers (one per profile).
        spectrograph : `int`
            Spectrograph number (1..4).

        Returns
        -------
        merged : `FiberProfileSet`
            Merged fiber profiles.
        """
        if badFibers is None:
            badFibers = []

        profiles = [FiberProfileSet.readFits(fn) for fn in filenames]
        pfsConfigs = [butler.get("pfsConfig", visit=vv).select(spectrograph=spectrograph) for vv in visits]
        darkConfigs = [butler.get("pfsConfig", visit=vv).select(spectrograph=spectrograph)
                       for vv in darkVisits]
        numProfiles = len(profiles)
        assert len(pfsConfigs) == numProfiles and len(darkConfigs) == numProfiles  # enforced in TaskRunner

        # Fibers that were intended to be exposed
        intended = [set(conf.fiberId[np.logical_and.reduce(np.isfinite(conf.pfiNominal), axis=1)])
                    for conf in pfsConfigs]

        # Fibers that were exposed that were not intended to be
        bad = [set(conf.fiberId[np.logical_and.reduce(np.isfinite(conf.pfiCenter), axis=1)])
               for conf in darkConfigs]

        # Fibers that were always exposed; probably these are broken cobras, though some might just have had
        # trouble in their one dot-roach of consequence
        alwaysBad = bad[0].copy()
        for bb in bad[1:]:
            alwaysBad &= set(bb)

        # Fibers that were bad when we expected them to be good
        badWhenWanted = [bb & ii for bb, ii in zip(bad, intended)]

        tooCloseNormal = set()  # Fibers that were too close to a fiber that was intended to be exposed
        tooCloseBad = set()  # Fibers that were measured too close to a bad fiber
        for pp in profiles:
            for ff in pp:
                for delta in list(range(-self.config.fiberInfluence, self.config.fiberInfluence + 1)):
                    if delta == 0:
                        continue
                    if ff + delta in pp:
                        if ff + delta in alwaysBad:
                            tooCloseBad.add(ff)
                        else:
                            tooCloseNormal.add(ff)
        tooCloseNormal.difference_update(tooCloseBad)

        self.log.info("%d fibers alwaysBad: %s", len(alwaysBad), sorted(alwaysBad))
        self.log.info("%d fibers tooCloseNormal: %s", len(tooCloseNormal), sorted(tooCloseNormal))
        self.log.info("%d fibers tooCloseBad: %s", len(tooCloseBad), sorted(tooCloseBad))

        good = set()  # Fibers for which we expect we have good profile measurements
        for ii, (pp, bb) in enumerate(zip(profiles, badWhenWanted)):
            thisGood = set()
            for ff in pp:
                if ff not in bb and ff not in tooCloseNormal and ff not in tooCloseBad:
                    thisGood.add(ff)
            self.log.info("%d good fibers in set %d: %s", len(thisGood), ii, sorted(thisGood))
            good |= thisGood
        self.log.info("%d good fibers: %s", len(good), sorted(good))

        bad = alwaysBad | tooCloseNormal | tooCloseBad

        # Get profiles for the fibers we don't have good profiles for
        merged = FiberProfileSet.makeEmpty(profiles[0].identity, profiles[0].visitInfo, profiles[0].metadata)
        for ff in bad:
            for pp in profiles:
                if ff in pp:
                    merged[ff] = pp[ff]
                    break

        for pp, bb in zip(profiles, badWhenWanted):
            for ff in pp:
                if ff in good or (ff in bb and ff not in merged):
                    merged[ff] = pp[ff]
        self.log.info("Merged %d fibers", len(merged))

        # Fill in missing fibers
        allFibers = set()
        engFibers = set()
        for conf in pfsConfigs:
            allFibers.update(conf.fiberId)
            engFibers.update(conf.select(targetType=TargetType.ENGINEERING).fiberId)
        missing = allFibers - set(merged) - engFibers
        average = merged.average()
        for ff in missing:
            merged[ff] = average.copy()

        # Replace bad fibers with nearest good fibers
        bad |= set(badFibers) | missing
        self.log.info("Replacing %d bad and missing fibers: %s", len(bad), sorted(bad))
        merged.replaceFibers(bad, self.config.replaceNearest)

        return merged

    def _getMetadataName(self):
        return None

    def _getConfigName(self):
        return None
