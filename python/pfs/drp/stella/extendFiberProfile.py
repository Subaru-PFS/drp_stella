import os
import pickle
import warnings

import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt

from lsst.utils import getPackageDir
from lsst.pipe.base import CmdLineTask, TaskRunner, Struct, ArgumentParser
from lsst.pex.config import Config, Field, ConfigurableField

from .buildFiberTraces import BuildFiberTracesConfig, BuildFiberTracesTask

import lsstDebug


class MeasureFiberProfileRunner(TaskRunner):
    """How to run MeasureFiberProfileTask"""
    @classmethod
    def getTargetList(cls, parsedCmd, **kwargs):
        return super().getTargetList(parsedCmd, filename=parsedCmd.filename)


class MeasureFiberProfileConfig(Config):
    """Configuration for MeasureFiberProfileTask"""
    buildFiberTraces = ConfigurableField(target=BuildFiberTracesTask, doc="Build fiber traces")
    rejIter = Field(dtype=int, default=3, doc="Number of rejection iterations for collapsing the profile")
    rejThresh = Field(dtype=float, default=3.0, doc="Rejection threshold (sigma) for collapsing the profile")


class MeasureFiberProfileTask(CmdLineTask):
    """Measure and persist the average fiber profile, for later use"""
    ConfigClass = MeasureFiberProfileConfig
    _DefaultName = "measureFiberProfile"
    RunnerClass = MeasureFiberProfileRunner

    def __init__(self, *args, **kwargs):
        CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("buildFiberTraces")
        self.debugInfo = lsstDebug.Info(__name__)

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="calexp",
                               help="data IDs, e.g. --id visit=12345 arm=b^r")
        parser.add_argument("--filename", default="profile-%(visit)d-%(arm)s%(spectrograph)d.pkl",
                            help=("Format for plot filename, e.g., "
                                  "profile-%%(visit)d-%%(arm)s%%(spectrograph)d.pkl"))
        return parser

    def runDataRef(self, dataRef, filename):
        """Measure and persist the average fiber profile

        Main entry point for `lsst.pipe.base.CmdLineTask`.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Butler data reference for a ``calexp``.
        filename : `str`
            Filename template.

        Returns
        -------
        profile : `lsst.pipe.base.Struct`
            Average profile data, with ``index`` (`numpy.ndarray` of `int`,
            shape ``(N,)``) and ``values`` (`numpy.ndarray` of `float`, shape
            ``(N,)``) elements.
        """
        exposure = dataRef.get("calexp")
        detMap = dataRef.get("detectorMap")
        pfsConfig = dataRef.get("pfsConfig")
        return self.run(exposure.maskedImage, detMap, pfsConfig, filename % dataRef.dataId)

    def run(self, maskedImage, detMap, pfsConfig, filename):
        """Measure and persist the average fiber profile

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which to measure the fiber profile.
        detMap : `pfs.drp.stella.DetectorMap`
            Mapping from x,y to fiberId,wavelength.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Fiber configuration.
        filename : `str`
            Filename template.

        Returns
        -------
        profile : `lsst.pipe.base.Struct`
            Average profile data, with ``index`` (`numpy.ndarray` of `int`,
            shape ``(N,)``) and ``values`` (`numpy.ndarray` of `float`, shape
            ``(N,)``) elements.
        """
        results = self.buildFiberTraces.buildFiberTraces(maskedImage, detMap, pfsConfig)
        collapsed = self.collapseProfile(results.profiles)
        self.write(collapsed, filename)
        return collapsed

    def collapseProfile(self, profiles):
        """Collapse the measured profiles into an average profile

        We take a clipped mean of the profile over all swaths and all fibers.

        Parameters
        ----------
        profiles : `list` of `pfs.drp.stella.fiberProfile.FiberProfile`
              Profiles of each fiber, in the same order as the fiber traces.

        Returns
        -------
        index : `numpy.ndarray` of `int`, shape ``(N,)``
            Positions in the spatial dimension.
        values : `numpy.ndarray` of `float`, shape ``(N,)``
            Corresponding values of the profile.
        """
        index = profiles[0].index
        assert all(np.all(prof.index == index) for prof in profiles)
        shape = index.shape[0]
        num = sum(len(prof.profiles) for prof in profiles)
        samples = np.ma.empty((num, shape), dtype=float)
        start = 0
        for prof in profiles:
            end = start + len(prof.profiles)
            samples[start:end] = prof.profiles
            start = end

        for ii in range(self.config.rejIter):
            with warnings.catch_warnings():
                # Suppress "RuntimeWarning: All-NaN slice encountered" from nanmedian/nanpercentile
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                median = np.nanmedian(samples.filled(np.nan), axis=0)
                lq, uq = np.nanpercentile(samples.filled(np.nan), (25.0, 75.0), axis=0)
            rms = 0.741*(uq - lq)
            residual = samples - median[np.newaxis]
            with np.errstate(invalid="ignore"):  # Ignore NANs
                samples.mask |= np.abs(residual) > self.config.rejThresh*rms[np.newaxis]
        collapsed = np.mean(samples, axis=0)

        if self.debugInfo.plotProfiles:
            numCols = int(np.ceil(np.sqrt(len(profiles))))
            numRows = int(np.ceil(len(profiles)/numCols))
            fig, axes = plt.subplots(numRows, numCols, sharex=True, sharey=True, squeeze=False)
            fig.subplots_adjust(hspace=0, wspace=0)

            for prof, ax in zip(profiles, sum(axes.tolist(), [])):
                colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(prof.profiles)))
                for pp, col in zip(prof.profiles, colors):
                    ax.scatter(prof.index, pp, marker=".", color=col)
                ax.scatter(prof.index, collapsed, marker=".", color="k")
            axes[0][0].set_yscale("symlog", linthreshy=1.0e-3)
            plt.show()

        return Struct(index=index, values=collapsed)

    def write(self, profile, filename):
        """Persist the average fiber profile

        Parameters
        ----------
        profile : `lsst.pipe.base.Struct`
            Average profile; output from ``collapseProfile``.
        filename : `str`
            Filename to which to write the profile.
        """
        with open(filename, "wb") as fd:
            pickle.dump(profile, fd)

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None


class BuildExtendedFiberTracesConfig(BuildFiberTracesConfig):
    """Configuration for BuildExtendedFiberTracesTask"""
    filename = Field(
        dtype=str,
        default=os.path.join(getPackageDir("drp_pfs_data"), "fiberTraces/extended-2019-12-17.pkl"),
        doc="Filename for extended fiber traces"
    )

    def setDefaults(self):
        """Adjust default settings for dense fibers"""
        BuildFiberTracesConfig.setDefaults(self)
        self.profileRadius = 2
        self.centroidRadius = 2


class BuildExtendedFiberTracesTask(BuildFiberTracesTask):
    ConfigClass = BuildExtendedFiberTracesConfig
    _DefaultName = "buildExtendedFiberTraces"

    def buildFiberTraces(self, maskedImage, detectorMap=None, pfsConfig=None):
        """Build a FiberTraceSet from an image and extend it with an average

        We find traces on the image, centroid those traces, measure the fiber
        profile, and construct the FiberTraces.

        This method provides more ouputs than the ``run`` method.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which to build FiberTraces.
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Mapping from x,y to fiberId,row.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end fiber configuration.

        Returns
        -------
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            The fiber traces.
        profiles : `list` of `pfs.drp.stella.fiberProfile.FiberProfile`
            Profiles of each fiber, in the same order as the fiber traces.
        centers : `list` of callable
            Callable for each fiber that provides the center of the trace as a
            function of row.
        """
        results = BuildFiberTracesTask.buildFiberTraces(self, maskedImage, detectorMap, pfsConfig)
        extended = self.readExtended()
        for profile in results.profiles:
            self.extendFiberProfile(profile, extended)

        # Update the FiberTraceSet using the extended profiles
        for ii, ft in enumerate(results.fiberTraces):
            new = profile.makeFiberTrace(maskedImage.getWidth(), maskedImage.getHeight(), results.centers[ii])
            new.fiberId = ft.fiberId
            results.fiberTraces[ii] = new

        return results

    def readExtended(self):
        """Read the extended profile

        Returns
        -------
        extended : `lsst.pipe.base.Struct`
            Extended profile data, with ``index`` (`numpy.ndarray` of `int`,
            shape ``(N,)``) and ``values`` (`numpy.ndarray` of `float`, shape
            ``(N,)``) elements.
        """
        with open(self.config.filename, "rb") as fd:
            return pickle.load(fd)

    def extendFiberProfile(self, profile, extended):
        """Hack the FiberProfile to extend the reach of the profile

        We supplement the profile in each swath with the extended profile.
        We only add elements from the extended profile that are outside the
        reach of the original profile.

        Parameters
        ----------
        profile : `pfs.drp.stella.FiberProfile`
            Fiber profile to extend; modified in-place.
        extended : `lsst.pipe.base.Struct`
            Extended profile data, with ``index`` (`numpy.ndarray` of `int`,
            shape ``(N,)``) and ``values`` (`numpy.ndarray` of `float`, shape
            ``(N,)``) elements.
        """
        index = extended.index
        values = extended.values/extended.values.sum()
        select = (index < profile.index.min()) | (index > profile.index.max())
        if not np.any(select):
            self.log.warn("Extended profile doesn't extend existing profile")
            return
        newSize = profile.index.shape[0] + select.sum()
        newIndex = np.empty(newSize, dtype=float)
        newIndex[select] = index[select]
        newIndex[~select] = profile.index

        fluxRatio = values[~select].sum()/values[select].sum()  # ratio inner:outer
        numSwaths = len(profile.profiles)
        replaced = np.ma.empty((numSwaths, newSize), dtype=float)
        replaced.mask = np.empty_like(replaced.data, dtype=bool)
        for ii in range(numSwaths):
            replaced[ii][select] = values[select]
            replaced.mask[ii][select] = values.mask[select]
            replaced[ii][~select] = profile.profiles[ii]*fluxRatio/profile.profiles[ii].sum()
            replaced.mask[ii][~select] = profile.profiles.mask[ii]
        profile.profiles = replaced
        profile.index = newIndex

        radius = index.max()
        profile.radius = int(radius)
        assert profile.radius == radius
