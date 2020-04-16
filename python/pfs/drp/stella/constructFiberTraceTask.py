import numpy as np
from collections import defaultdict

import lsst.daf.base as dafBase
import lsst.afw.display as afwDisplay
import lsst.afw.image as afwImage
from lsst.pex.config import Field, ConfigurableField, ConfigField
from lsst.pipe.drivers.constructCalibs import CalibTaskRunner
from .constructSpectralCalibs import SpectralCalibConfig, SpectralCalibTask
from .findAndTraceAperturesTask import FindAndTraceAperturesTask
from pfs.drp.stella import Spectrum, SlitOffsetsConfig
from pfs.drp.stella.fitContinuum import FitContinuumTask


class ConstructFiberTraceTaskRunner(CalibTaskRunner):
    """Split values with different pfsDesignId"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        pfsDesignId = defaultdict(list)
        for dataRef in parsedCmd.id.refList:
            pfsDesignId[dataRef.dataId["pfsDesignId"]].append(dataRef)
        return [dict(expRefList=expRefList, butler=parsedCmd.butler, calibId=parsedCmd.calibId) for
                expRefList in pfsDesignId.values()]


class ConstructFiberTraceConfig(SpectralCalibConfig):
    """Configuration for FiberTrace construction"""
    trace = ConfigurableField(
        target=FindAndTraceAperturesTask,
        doc="Task to trace apertures"
    )
    requireZeroDither = Field(
        dtype=bool,
        default=True,
        doc="""Require a zero slit dither value?

        The fiber trace should have the same dither as the data, which is usually zero.
        """,
    )
    slitOffsets = ConfigField(dtype=SlitOffsetsConfig, doc="Manual slit offsets to apply to detectorMap")
    fitContinuum = ConfigurableField(target=FitContinuumTask, doc="Fit continuum")

    def setDefaults(self):
        super().setDefaults()
        self.doCameraImage = False  # We don't produce 2D images


class ConstructFiberTraceTask(SpectralCalibTask):
    """Task to construct the fiber trace"""
    ConfigClass = ConstructFiberTraceConfig
    _DefaultName = "fiberTrace"
    calibName = "fiberTrace"
    RunnerClass = ConstructFiberTraceTaskRunner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("trace")
        self.makeSubtask("fitContinuum")

    def run(self, expRefList, butler, calibId):
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
        """!Combine multiple exposures of a particular CCD and write the output

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

        self.interpolateNans(exposure)

        if self.debugInfo.display and self.debugInfo.combinedFrame >= 0:
            display = afwDisplay.Display(frame=self.debugInfo.combinedFrame)
            display.mtv(exposure, "Combined")

        detMap = dataRefList[0].get('detectorMap')
        self.config.slitOffsets.apply(detMap, self.log)

        traces = self.trace.run(exposure.maskedImage, detMap)
        self.log.info('%d fiber traces found on combined flat' % (traces.size(),))

        # Set the normalisation of the FiberTraces
        # The normalisation is the flat: we want extracted spectra to be relative to the flat.
        spectra = traces.extractSpectra(exposure.maskedImage)
        for ss, tt in zip(spectra, traces):
            bbox = tt.trace.getBBox()
            select = slice(bbox.getMinY(), bbox.getMaxY() + 1)
            scale = ss.spectrum[select, np.newaxis]
            tt.trace.image.array *= scale
            tt.trace.variance.array *= scale**2
            norm = np.sum(tt.trace.image.array, axis=1)
            self.log.info("Median relative transmission of fiber %d is %f",
                          tt.fiberId, np.median(norm[np.isfinite(norm)]))

        if self.debugInfo.display and self.debugInfo.combinedFrame >= 0:
            display = afwDisplay.Display(frame=self.debugInfo.combinedFrame)
            traces.applyToMask(exposure.getMaskedImage().getMask())
            display.setMaskTransparency(50)
            display.mtv(exposure, "Traces")
        #
        # And write it
        #
        visitInfo = exposure.getInfo().getVisitInfo()
        if visitInfo is None:
            dateObs = dafBase.DateTime('%sT00:00:00Z' % dataRefList[0].dataId['dateObs'],
                                       dafBase.DateTime.UTC)
            visitInfo = afwImage.VisitInfo(date=dateObs)

        # Clear out metadata to avoid conflicts with existing keywords when we set the stuff we need
        for key in detMap.metadata.names():
            detMap.metadata.remove(key)
        detMap.setVisitInfo(visitInfo)
        self.recordCalibInputs(cache.butler, detMap, struct.ccdIdList, outputId)
        detMap.getMetadata().set("OBSTYPE", "detectorMap")  # Overwrite "fiberTrace"

        dataRefList[0].put(detMap, 'detectorMap', visit0=dataRefList[0].dataId['visit'])

        self.recordCalibInputs(cache.butler, traces, struct.ccdIdList, outputId)
        self.write(cache.butler, traces, outputId)

    def calculateAverage(self, spectra):
        """Calculate an average spectrum

        We calculate the median spectrum across rows. This is not the same as
        averaging over wavelength, and it matters: small features like
        absorption lines are a function of wavelength, so can show up on
        different rows. We'll work around this by fitting the continuum and
        using that instead of the average. That also avoids having sharp
        features in the average spectrum, which should mean that the flux
        calibration shouldn't have to include sharp features either (except for
        telluric lines).

        Parameters
        ----------
        spectra : `pfs.drp.stella.SpectrumSet`
            Spectra to average. Will be modified to remove non-finite values.

        Returns
        -------
        average : `pfs.drp.stella.Spectrum`
            Average spectrum.
        """
        for ss in spectra:
            ss.spectrum[:] = np.where(np.isfinite(ss.spectrum), ss.spectrum, 0.0)
        average = Spectrum(spectra.getLength())
        average.spectrum[:] = np.median([ss.spectrum for ss in spectra], axis=0)
        average.spectrum = self.fitContinuum.fitContinuum(average)
        return average
