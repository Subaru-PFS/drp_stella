import numpy as np

from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task

from lsst.afw.display import Display
from lsst.ip.isr.isrFunctions import createPsf

from pfs.datamodel import FiberStatus
from pfs.drp.stella.traces import findTracePeaks, centroidTrace
from pfs.drp.stella.images import convolveImage

import lsstDebug

__all__ = ("CentroidTracesConfig", "CentroidTracesTask")


class CentroidTracesConfig(Config):
    """Configuration for CentroidLinesTask"""
    fwhm = Field(dtype=float, default=1.5, doc="FWHM of PSF (pixels)")
    kernelSize = Field(dtype=float, default=4.0, doc="Size of convolution kernel (sigma)")
    mask = ListField(dtype=str, default=["CR", "BAD", "NO_DATA"], doc="Mask planes to ignore")
    threshold = Field(dtype=float, default=50.0, doc="Signal-to-noise threshold for trace")
    searchRadius = Field(dtype=float, default=3, doc="Radius about the expected peak to search")
    centroidRadius = Field(dtype=int, default=3, doc="Radius about the peak for centroiding")


class CentroidTracesTask(Task):
    """Centroid traces on an exposure"""
    ConfigClass = CentroidTracesConfig
    _DefaultName = "centroidTraces"

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, exposure, detectorMap, pfsConfig=None):
        """Centroid traces on an exposure

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Arc exposure on which to centroid lines.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Approximate mapping between fiberId,wavelength and x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end fiber configuration.

        Returns
        -------
        tracePeaks : `dict` [`int`: `list` of `pfs.drp.stella.TracePeak`]
            Peaks for each trace, indexed by fiberId.
        """
        psf = exposure.getPsf()
        if psf is None:
            psf = createPsf(self.config.fwhm)
        convolved = self.convolveImage(exposure, psf)
        convolved.image.array /= np.sqrt(convolved.variance.array)
        traces = self.findTracePeaks(convolved, detectorMap, pfsConfig)
        self.centroidTraces(exposure.maskedImage, traces)
        self.log.info("Measured %d centroids for %d traces",
                      sum((len(tt)) for tt in traces.values()), len(traces))
        return traces

    def convolveImage(self, exposure, psf):
        """Convolve image by Gaussian kernel

        If the PSF isn't provided in the ``exposure``, then we use the ``fwhm``
        from the config.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image to convolve. The PSF must be set.
        psf : `lsst.afw.detection.Psf`
            Two-dimensional point-spread function.

        Returns
        -------
        convolved : `lsst.afw.image.MaskedImage`
            Convolved image.
        """
        sigma = psf.computeShape().getTraceRadius()
        convolvedImage = convolveImage(exposure.maskedImage, sigma, sigma, sigmaNotFwhm=True)
        if self.debugInfo.displayConvolved:
            Display(frame=1).mtv(convolvedImage)

        return convolvedImage

    def findTracePeaks(self, maskedImage, detectorMap, pfsConfig=None):
        """Find peaks for each trace

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Signal-to-noise image.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Approximate mapping between fiberId,wavelength and x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end fiber configuration.

        Returns
        -------
        tracePeaks : `dict` [`int`: `list` of `pfs.drp.stella.TracePeak`]
            Peaks for each trace, indexed by fiberId.
        """
        fiberId = None
        if pfsConfig is not None:
            select = pfsConfig.getSelection(fiberId=detectorMap.fiberId, fiberStatus=FiberStatus.GOOD)
            fiberId = pfsConfig.fiberId[select]
        badBitMask = maskedImage.mask.getPlaneBitMask(self.config.mask)
        return findTracePeaks(maskedImage, detectorMap, self.config.threshold, self.config.searchRadius,
                              badBitMask, fiberId)

    def centroidTraces(self, maskedImage, tracePeaks):
        """Measure the centroids for peaks in each trace

        The centroid for each peak is updated with the measured value.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImageF`
            Image on which to measure centroids.
        tracePeaks : `dict` [`int`: `list` of `pfs.drp.stella.TracePeak`]
            Peaks for each trace, indexed by fiberId.
        """
        badBitmask = maskedImage.mask.getPlaneBitMask(self.config.mask)
        for ff in tracePeaks:
            centroidTrace(tracePeaks[ff], maskedImage, self.config.centroidRadius, badBitmask)
        if self.debugInfo.plotCentroids:
            import matplotlib.pyplot as plt
            for ff in tracePeaks:
                plt.plot([pp.peak for pp in tracePeaks[ff]], [pp.row for pp in tracePeaks[ff]], 'k.')
            plt.show()
