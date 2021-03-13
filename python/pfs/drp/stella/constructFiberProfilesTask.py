import numpy as np
from collections import defaultdict

import lsst.afw.display as afwDisplay
import lsst.afw.image as afwImage
from lsst.pex.config import Field, ConfigurableField, ConfigField, ListField
from lsst.pipe.drivers.constructCalibs import CalibTaskRunner
from .constructSpectralCalibs import SpectralCalibConfig, SpectralCalibTask
from .buildFiberProfiles import BuildFiberProfilesTask
from pfs.drp.stella import SlitOffsetsConfig
from pfs.drp.stella.fitContinuum import FitContinuumTask


class ConstructFiberProfilesTaskRunner(CalibTaskRunner):
    """Split values with different pfsDesignId"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        pfsDesignId = defaultdict(list)
        for dataRef in parsedCmd.id.refList:
            pfsDesignId[dataRef.dataId["pfsDesignId"]].append(dataRef)
        return [dict(expRefList=expRefList, butler=parsedCmd.butler, calibId=parsedCmd.calibId) for
                expRefList in pfsDesignId.values()]


class ConstructFiberProfilesConfig(SpectralCalibConfig):
    """Configuration for FiberTrace construction"""
    profiles = ConfigurableField(target=BuildFiberProfilesTask, doc="Build fiber profiles")
    requireZeroDither = Field(
        dtype=bool,
        default=True,
        doc="""Require a zero slit dither value?

        The fiber trace should have the same dither as the data, which is usually zero.
        """,
    )
    slitOffsets = ConfigField(dtype=SlitOffsetsConfig, doc="Manual slit offsets to apply to detectorMap")
    fitContinuum = ConfigurableField(target=FitContinuumTask, doc="Fit continuum")
    mask = ListField(dtype=str, default=["BAD_FLAT", "CR", "SAT", "NO_DATA"],
                     doc="Mask planes to exclude from fiberTrace")
    forceFiberIds = Field(dtype=bool, default=False, doc="Force identified fiberIds to match pfsConfig?")

    def setDefaults(self):
        super().setDefaults()
        self.doCameraImage = False  # We don't produce 2D images


class ConstructFiberProfilesTask(SpectralCalibTask):
    """Task to construct the fiber trace"""
    ConfigClass = ConstructFiberProfilesConfig
    _DefaultName = "fiberProfiles"
    calibName = "fiberProfiles"
    RunnerClass = ConstructFiberProfilesTaskRunner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("profiles")
        self.makeSubtask("fitContinuum")

    def run(self, expRefList, butler, calibId):
        """Construct the ``fiberProfiles`` calib

        Parameters
        ----------
        expRefList : iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for exposures.
        butler : `lsst.daf.persistence.Butler`
            Data butler.
        calibId : `dict`
            Data identifier keyword-value pairs to use for the calib.
        """
        pfsDesignId = set([ref.dataId["pfsDesignId"] for ref in expRefList])
        if len(pfsDesignId) != 1:
            raise RuntimeError("Multiple pfsDesignId values: %s", pfsDesignId)
        if self.config.requireZeroDither:
            # Only run for Flats with slitOffset == 0.0
            rejected = []
            newExpRefList = []
            for expRef in expRefList:
                dither = expRef.getButler().queryMetadata("raw", "dither", expRef.dataId)
                assert len(dither) == 1, "Expect a single answer for this single dataset"
                dither = dither.pop()
                if dither == 0.0:
                    newExpRefList.append(expRef)
                else:
                    rejected.append(expRef.dataId)
            if rejected:
                self.log.warn("Rejected the following exposures with non-zero dither: %s", rejected)
                self.log.warn("To overcome this, either set 'requireZeroDither=False' or select only "
                              "input exposures with zero dither")
            expRefList = newExpRefList

        if not expRefList:
            raise RuntimeError("No input exposures")

        return super().run(expRefList, butler, calibId)

    def combine(self, cache, struct, outputId):
        """Combine multiple exposures of a particular CCD and write the output

        Only the slave nodes execute this method.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        struct : `lsst.pipe.base.Struct`
            Parameters for the combination, which has the following components:

            - ``ccdName`` (`tuple`): Name tuple for CCD.
            - ``ccdIdList`` (`list`): List of data identifiers for combination.
            - ``scales``: Unused by this implementation.

        Returns
        -------
        outputId : `dict`
            Data identifier for combined image (exposure part only).
        """
        combineResults = super().combine(cache, struct, outputId)
        dataRefList = combineResults.dataRefList
        outputId = combineResults.outputId

        calib = self.combination.run(dataRefList, expScales=struct.scales.expScales,
                                     finalScale=struct.scales.ccdScale)
        exposure = afwImage.makeExposure(calib)

        visitInfo = dataRefList[0].get("postISRCCD_visitInfo")
        exposure.getInfo().setVisitInfo(visitInfo)

        self.interpolateNans(exposure)

        if self.debugInfo.display and self.debugInfo.combinedFrame >= 0:
            display = afwDisplay.Display(frame=self.debugInfo.combinedFrame)
            display.mtv(exposure, "Combined")

        detMap = dataRefList[0].get('detectorMap')
        pfsConfig = dataRefList[0].get("pfsConfig")
        self.config.slitOffsets.apply(detMap, self.log)

        results = self.profiles.run(exposure, detMap, pfsConfig)
        self.log.info('%d fiber profiles found on combined flat', len(results.profiles))

        # Set the normalisation of the FiberProfiles
        # The normalisation is the flat: we want extracted spectra to be relative to the flat.
        traces = results.profiles.makeFiberTracesFromDetectorMap(detMap)
        spectra = traces.extractSpectra(exposure.maskedImage)
        medianTransmission = np.empty(len(spectra))
        for i, ss in enumerate(spectra):
            results.profiles[ss.fiberId].norm = ss.flux
            medianTransmission[i] = np.nanmedian(ss.flux)
            self.log.debug("Median relative transmission of fiber %d is %f",
                           ss.fiberId, medianTransmission[i])

        self.log.info("Median relative transmission of fibers %.2f +- %.2f (min %.2f, max %.2f)",
                      np.mean(medianTransmission), np.std(medianTransmission, ddof=1),
                      np.min(medianTransmission), np.max(medianTransmission))

        results.profiles.metadata.set("OBSTYPE", "fiberProfiles")
        date = results.profiles.getVisitInfo().getDate()
        results.profiles.metadata.set("calibDate", date.toPython(date.UTC).strftime("%Y-%m-%d"))

        if self.debugInfo.display and self.debugInfo.combinedFrame >= 0:
            display = afwDisplay.Display(frame=self.debugInfo.combinedFrame)
            traces.applyToMask(exposure.getMaskedImage().getMask())
            display.setMaskTransparency(50)
            display.mtv(exposure, "Traces")
        #
        # And write it
        #
        self.recordCalibInputs(cache.butler, results.profiles, struct.ccdIdList, outputId)
        self.write(cache.butler, results.profiles, outputId)
