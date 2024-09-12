from typing import List

from lsst.pex.config import Field

from lsst.pipe.base import Struct
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from pfs.datamodel import TargetType, PfsConfig
from .fiberProfileSet import FiberProfileSet
from .reduceProfiles import getIlluminatedFibers


class MergeFiberProfilesConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "profiles_run", "arm", "spectrograph"),
):
    """Connections for MergeFiberProfilesTask"""
    profiles = InputConnection(
        name="fiberProfiles_group",
        doc="Fiber profiles for individual groups",
        dimensions=("instrument", "profiles_run", "profiles_group", "arm", "spectrograph"),
        storageClass="FiberProfileSet",
        multiple=True,
    )
    data = PrerequisiteConnection(
        name="profiles_exposures",
        doc="List of bright and dark exposures",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "profiles_group"),
        multiple=True,
    )
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="PFS fiber configuration",
        dimensions=("instrument", "exposure"),
        storageClass="PfsConfig",
        multiple=True,
    )

    merged = OutputConnection(
        name="fiberProfiles",
        doc="Merged fiber profiles",
        dimensions=("instrument", "arm", "spectrograph"),
        storageClass="FiberProfileSet",
        isCalibration=True,
    )


class MergeFiberProfilesConfig(PipelineTaskConfig, pipelineConnections=MergeFiberProfilesConnections):
    """Configuration for MergeFiberProfilesTask"""
    fiberInfluence = Field(dtype=int, default=3,
                           doc="Number of fibers around an illuminated fiber whose profiles are influenced")
    replaceNearest = Field(dtype=int, default=3, doc="Number of nearest good fibers to use for replacement")


class MergeFiberProfilesTask(PipelineTask):
    """Merge fiber profiles"""
    ConfigClass = MergeFiberProfilesConfig
    _DefaultName = "mergeFiberProfiles"

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
        spectrograph = outputRefs.merged.dataId["spectrograph"]

        profiles = {ref.dataId["profiles_group"]: butler.get(ref) for ref in inputRefs.profiles}
        data = {ref.dataId["profiles_group"]: butler.get(ref) for ref in inputRefs.data}
        pfsConfigs = {ref.dataId["exposure"]: butler.get(ref) for ref in inputRefs.pfsConfig}
        brightConfigs = [[pfsConfigs[expId] for expId in data[group]["bright"]] for group in profiles]
        darkConfigs = [[pfsConfigs[expId] for expId in data[group]["dark"]] for group in profiles]

        results = self.run(list(profiles.values()), brightConfigs, darkConfigs, spectrograph)
        butler.put(results.merged, outputRefs.merged)

    def run(
        self,
        profiles: List[FiberProfileSet],
        brightConfigs: List[List[PfsConfig]],
        darkConfigs: List[List[PfsConfig]],
        spectrograph: int,
        badFibers: List[int] = None,
    ) -> FiberProfileSet:
        """Merge fiber profiles

        Parameters
        ----------
        profiles : list of `FiberProfileSet`
            Fiber profiles to merge.
        brightConfigs : list of list of `PfsConfig`
            Bright fiber configurations for each profile.
        darkConfigs : list of `int`
            Dark fiber configurations for each profile.
        spectrograph : `int`
            Spectrograph number (1..4).

        Returns
        -------
        merged : `FiberProfileSet`
            Merged fiber profiles.
        """
        if badFibers is None:
            badFibers = []

        # Dump everything but the first
        brightConfigs = [groupConfigs[0].select(spectrograph=spectrograph) for groupConfigs in brightConfigs]
        darkConfigs = [
            groupConfigs[0].select(spectrograph=spectrograph) if groupConfigs else None
            for groupConfigs in darkConfigs
        ]
        numProfiles = len(profiles)
        assert len(brightConfigs) == numProfiles and len(darkConfigs) == numProfiles  # enforced in runQuantum

        # Fibers that were intended to be exposed
        intended = [set(getIlluminatedFibers(conf, False)) for conf in brightConfigs]

        # Fibers that were exposed that were not intended to be
        bad = [set(getIlluminatedFibers(conf, True) if conf else []) for conf in darkConfigs]

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
        for conf in brightConfigs:
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

        return Struct(merged=merged)
