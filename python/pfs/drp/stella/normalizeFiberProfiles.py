from typing import Iterable, List

import numpy as np
from scipy.signal import medfilt  # noqa: E402

from lsst.pex.config import Config, ConfigurableField, Field, ListField
from lsst.pipe.base import Task, Struct
from lsst.daf.persistence import ButlerDataRef

from .reduceExposure import ReduceExposureTask
from .combineImages import CombineImagesTask
from .adjustDetectorMap import AdjustDetectorMapTask
from .blackSpotCorrection import BlackSpotCorrectionTask
from .fiberProfileSet import FiberProfileSet
from .centroidTraces import CentroidTracesTask, tracesToLines
from .constructSpectralCalibs import setCalibHeader
from .screen import ScreenResponseTask

__all__ = ("NormalizeFiberProfilesConfig", "NormalizeFiberProfilesTask")


class NormalizeFiberProfilesConfig(Config):
    """Configuration for normalizing fiber profiles"""
    reduceExposure = ConfigurableField(target=ReduceExposureTask, doc="Reduce single exposure")
    combine = ConfigurableField(target=CombineImagesTask, doc="CombineImages")
    doAdjustDetectorMap = Field(dtype=bool, default=False, doc="Adjust detectorMap using trace positions?")
    adjustDetectorMap = ConfigurableField(target=AdjustDetectorMapTask, doc="Adjust detectorMap")
    centroidTraces = ConfigurableField(target=CentroidTracesTask, doc="Centroid traces")
    traceSpectralError = Field(dtype=float, default=1.0,
                               doc="Error in the spectral dimension to give trace centroids (pixels)")
    mask = ListField(dtype=str, default=["BAD_FLAT", "CR", "SAT", "NO_DATA"],
                     doc="Mask planes to exclude from fiberTrace")
    doApplyScreenResponse = Field(dtype=bool, default=True, doc="Apply screen response correction?")
    screen = ConfigurableField(target=ScreenResponseTask, doc="Screen response correction")
    blackspots = ConfigurableField(target=BlackSpotCorrectionTask, doc="Black spot correction")

    def setDefaults(self):
        self.reduceExposure.doMeasureLines = False
        self.reduceExposure.doMeasurePsf = False
        self.reduceExposure.doSubtractSky2d = False
        self.reduceExposure.doExtractSpectra = False
        self.reduceExposure.doWriteArm = False
        self.adjustDetectorMap.minSignalToNoise = 0  # We don't measure S/N


class NormalizeFiberProfilesTask(Task):
    """Task to normalize fiber profiles"""
    ConfigClass = NormalizeFiberProfilesConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("reduceExposure")
        self.makeSubtask("combine")
        self.makeSubtask("centroidTraces")
        self.makeSubtask("adjustDetectorMap")
        self.makeSubtask("screen")
        self.makeSubtask("blackspots")

    def run(self, profiles: FiberProfileSet, normRefList: List[ButlerDataRef], visitList: List[int]):
        combined = self.makeCombinedExposure(normRefList)
        spectra = profiles.extractSpectra(
            combined.exposure.maskedImage,
            combined.detectorMap,
            combined.exposure.mask.getPlaneBitMask(self.config.mask),
        )
        if self.config.doApplyScreenResponse:
            self.screen.run(combined.exposure.getMetadata(), spectra, combined.pfsConfig)
        self.blackspots.run(combined.pfsConfig, spectra)

        for ss in spectra:
            good = (ss.mask.array[0] & ss.mask.getPlaneBitMask("NO_DATA")) == 0
            profiles[ss.fiberId].norm = np.where(good, ss.flux/ss.norm, np.nan)

        self.write(normRefList[0], profiles, visitList, [dataRef.dataId["visit"] for dataRef in normRefList])
        self.plotProfiles(normRefList[0], profiles)

    def processExposure(self, dataRef: ButlerDataRef) -> Struct:
        """Process an exposure

        We read existing data from the butler, if available. Otherwise, we
        construct it by running ``reduceExposure``.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.

        Returns
        -------
        data : `lsst.pipe.base.Struct`
            Struct with ``exposure``, ``detectorMap``, ``pfsConfig``.
        """
        require = ("calexp", "detectorMap_used")
        if all(dataRef.datasetExists(name) for name in require):
            self.log.info("Reading existing data for %s", dataRef.dataId)
            data = Struct(
                exposure=dataRef.get("calexp"),
                detectorMap=dataRef.get("detectorMap_used"),
                pfsConfig=dataRef.get("pfsConfig"),
            )
        else:
            data = self.reduceExposure.runDataRef(dataRef)
        return data

    def makeCombinedExposure(self, dataRefList: List[ButlerDataRef]) -> Struct:
        """Generate a combined exposure

        Combines the input exposures, and optionally adjusts the detectorMap.

        Parameters
        ----------
        dataRefList : iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for exposures.
        bgImage : `lsst.afw.image.Image`
            Background image.
        darkList : list of `lsst.pipe.base.Struct`
            List of dark processing results.

        Returns
        -------
        data : `lsst.pipe.base.Struct`
            Struct with ``exposure``, ``detectorMap``, ``pfsConfig``.
        """
        dataList = [self.processExposure(ref) for ref in dataRefList]
        combined = self.combine.run([data.exposure for data in dataList])

        if self.config.doAdjustDetectorMap:
            detectorMap = dataList[0].detectorMap
            arm = dataRefList[0].dataId["arm"]
            traces = self.centroidTraces.run(combined, detectorMap, dataList[0].pfsConfig)
            lines = tracesToLines(detectorMap, traces, self.config.traceSpectralError)
            detectorMap = self.adjustDetectorMap.run(
                detectorMap, lines, arm, combined.visitInfo.id
            ).detectorMap
        else:
            detectorMap = dataList[0].detectorMap

        return Struct(exposure=combined, detectorMap=detectorMap, pfsConfig=dataList[0].pfsConfig)

    def getOutputId(self, dataRef: ButlerDataRef, profiles: FiberProfileSet) -> dict:
        """Get the output data identifier

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Example data reference.
        profiles : `pfs.drp.stella.FiberProfileSet`
            Fiber profiles.

        Returns
        -------
        outputId : `dict`
            Output data identifier.
        """
        return dict(
            visit0=profiles.identity.visit0,
            calibDate=profiles.identity.obsDate.split("T")[0],
            calibTime=profiles.identity.obsDate,
            arm=profiles.identity.arm,
            spectrograph=profiles.identity.spectrograph,
            ccd=dataRef.dataId["ccd"],
            filter=profiles.identity.arm,
        )

    def write(
        self,
        dataRef: ButlerDataRef,
        profiles: FiberProfileSet,
        dataVisits: Iterable[int],
        normVisits: Iterable[int],
    ):
        """Write outputs

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for output.
        profiles : `pfs.drp.stella.FiberProfileSet`
            Fiber profiles.
        dataVisits : iterable of `int`
            List of visits used to construct the fiber profiles.
        normVisits : iterable of `int`
            List of visits used to measure the normalisation.
        """
        self.log.info("Writing output for %s", dataRef.dataId)
        outputId = self.getOutputId(dataRef, profiles)

        setCalibHeader(profiles.metadata, "fiberProfiles", dataVisits, outputId)
        for ii, vv in enumerate(sorted(set(normVisits))):
            profiles.metadata.set(f"CALIB_NORM_{ii}", vv)

        dataRef.put(profiles, "fiberProfiles", **outputId)

    def plotProfiles(self, dataRef: ButlerDataRef, profiles: FiberProfileSet):
        """Plot fiber profiles

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for output.
        profiles : `pfs.drp.stella.FiberProfileSet`
            Fiber profiles.
        """
        import matplotlib.pyplot as plt
        plt.switch_backend("agg")
        from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

        outputId = self.getOutputId(dataRef, profiles)
        description = f"{profiles.identity.arm}{profiles.identity.spectrograph}"

        filename = dataRef.get("fiberProfilesStats_filename", **outputId)[0]
        with PdfPages(filename) as pdf:
            figAxes = profiles.plotHistograms(show=False)
            for fig, axes in figAxes:
                pdf.savefig(fig)
                plt.close(fig)

        filename = dataRef.get("fiberProfilesPlots_filename", **outputId)[0]
        with PdfPages(filename) as pdf:
            figAxes = profiles.plot(show=False)
            for ff in profiles:
                fig, axes = figAxes[ff]
                fig.suptitle(f"Fiber profiles for {description}")
                axes.semilogy()
                axes.set_ylim(3e-4, 5e-1)

            figures = {id(fig): fig for fig, _ in figAxes.values()}
            for ff in profiles:
                fig, axes = figAxes[ff]
                if id(fig) in figures:
                    pdf.savefig(fig)
                    del figures[id(fig)]
                    plt.close(fig)

            haveNorm = False
            fig, axes = plt.subplots()
            for ff in profiles:
                if profiles[ff].norm is None:
                    continue
                axes.plot(profiles[ff].norm, label=str(ff))
                haveNorm = True
            if haveNorm:
                axes.set_xlabel("Row (pixels)")
                axes.set_ylabel("Flux (electrons)")
                axes.set_title(f"{description} normalization")
                axes.semilogy()
                top = np.max([np.max(medfilt(np.nan_to_num(profiles[ff].norm), 15)) for ff in profiles])
                axes.set_ylim(1, 2*top)
                pdf.savefig(fig)
            plt.close(fig)
