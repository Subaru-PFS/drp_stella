from collections import defaultdict
from lsst.pex.config import Config, Field, ConfigurableField
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner

from pfs.datamodel.drp import PfsMerged
from .combine import CombineTask
from .subtractSky1d import SubtractSky1dTask


class MergeArmsConfig(Config):
    """Configuration for MergeArmsTask"""
    doSubtractSky1d = Field(dtype=bool, default=True, doc="Do 1D sky subtraction?")
    subtractSky1d = ConfigurableField(target=SubtractSky1dTask, doc="1d sky subtraction")
    combine = ConfigurableField(target=CombineTask, doc="Combine spectra")
    doBarycentricCorr = Field(dtype=bool, default=True, doc="Do barycentric correction?")


class MergeArmsRunner(TaskRunner):
    """Runner for MergeArmsTask"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Produce list of targets for MergeArmsTask

        We want to operate on all data within a single exposure at once.
        """
        exposures = defaultdict(lambda: defaultdict(list))
        for ref in parsedCmd.id.refList:
            expId = ref.dataId["expId"]
            spectrograph = ref.dataId["spectrograph"]
            exposures[expId][spectrograph].append(ref)
        return [(list(specs.values()), kwargs) for specs in exposures.values()]


class MergeArmsTask(CmdLineTask):
    """Merge all extracted spectra from a single exposure"""
    _DefaultName = "mergeArms"
    ConfigClass = MergeArmsConfig
    RunnerClass = MergeArmsRunner

    @classmethod
    def _makeArgumentParser(cls):
        """Make an ArgumentParser"""
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="pfsArm",
                               help="data IDs, e.g. --id exp=12345 spectrograph=1..3")
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("subtractSky1d")
        self.makeSubtask("combine")

    def run(self, expSpecRefList):
        """Merge all extracted spectra from a single exposure

        Parameters
        ----------
        expSpecRefList : iterable of iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for each sensor, grouped by spectrograph.

        Returns
        -------
        merged : `pfs.datamodel.PfsMerged`
            Merged spectra.
        """
        spectra = [[dataRef.get("pfsArm") for dataRef in specRefList] for
                   specRefList in expSpecRefList]
        # XXX fix when we have LSF implemented
        lsf = [[None for dataRef in specRefList] for specRefList in expSpecRefList]
        pfsConfig = expSpecRefList[0][0].get("pfsConfig")
        if self.config.doSubtractSky1d:
            self.subtractSky1d.run(sum(spectra, []), pfsConfig, sum(lsf, []))

        spectrographs = [self.runSpectrograph(ss) for ss in spectra]  # Merge in wavelength
        merged = self.mergeSpectrographs(spectrographs)  # Merge across spectrographs
        expSpecRefList[0][0].put(merged, "pfsMerged")
        return merged

    def runSpectrograph(self, spectraList):
        """Merge spectra from arms within the same spectrograph

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsArm`
            Spectra from the multiple arms of a single spectrograph.

        Returns
        -------
        result : `pfs.datamodel.PfsMerged`
            Merged spectra for spectrograph.
        """
        return self.combine.run(spectraList, ["expId", "spectrograph"], PfsMerged)

    def mergeSpectrographs(self, spectraList):
        """Merge spectra from multiple spectrographs

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsMerged`
            Spectra to merge.

        Returns
        -------
        merged : `pfs.datamodel.PfsMerged`
            Merged spectra.
        """
        return PfsMerged.fromMerge(["expId"], spectraList)

    def _getMetadataName(self):
        return None
