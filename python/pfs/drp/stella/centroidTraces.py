from typing import Dict, Iterable

import numpy as np

from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task

from lsst.afw.display import Display
from lsst.ip.isr.isrFunctions import createPsf

from pfs.datamodel import FiberStatus
from pfs.drp.stella.traces import findTracePeaks, centroidPeak, TracePeak, extractTraceData
from pfs.drp.stella.images import convolveImage
from .DetectorMapContinued import DetectorMap
from .arcLine import ArcLineSet
from .referenceLine import ReferenceLineSource, ReferenceLineStatus
from .utils.psf import fwhmToSigma

import lsstDebug

__all__ = ("CentroidTracesConfig", "CentroidTracesTask", "tracesToLines")


class CentroidTracesConfig(Config):
    """Configuration for CentroidLinesTask"""
    fwhm = Field(dtype=float, default=1.5, doc="FWHM of PSF (pixels)")
    kernelSize = Field(dtype=float, default=4.0, doc="Size of convolution kernel (sigma)")
    mask = ListField(dtype=str, default=["CR", "BAD", "NO_DATA"], doc="Mask planes to ignore")
    threshold = Field(dtype=float, default=20.0, doc="Signal-to-noise threshold for trace")
    searchRadius = Field(dtype=float, default=1, doc="Radius about the expected peak to search")


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
        with np.errstate(invalid="ignore", divide="ignore"):
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
        sigma = psf.computeShape(psf.getAveragePosition()).getTraceRadius()
        convolvedImage = convolveImage(exposure.maskedImage, sigma, 0.0, sigmaNotFwhm=True)
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
        tracePeaks = findTracePeaks(maskedImage, detectorMap, self.config.threshold, self.config.searchRadius,
                                    badBitMask, fiberId)
        if self.debugInfo.plotPeaks:
            display = Display(frame=1)
            display.mtv(maskedImage)
            for ff in tracePeaks:
                with display.Buffering():
                    for pp in tracePeaks[ff]:
                        display.dot("+", pp.peak, pp.row)

        return tracePeaks

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
        psfSigma = fwhmToSigma(self.config.fwhm)
        for ff in tracePeaks:
            for pp in tracePeaks[ff]:
                centroidPeak(pp, maskedImage, psfSigma, badBitmask)
        if self.debugInfo.plotCentroids:
            import matplotlib.pyplot as plt
            for ff in tracePeaks:
                plt.plot([pp.peak for pp in tracePeaks[ff]], [pp.row for pp in tracePeaks[ff]], 'k.')
            plt.show()


def tracesToLines(detectorMap: DetectorMap, traces: Dict[int, Iterable[TracePeak]],
                  spectralError: float) -> ArcLineSet:
    """Convert traces to lines

    Well, they're not really lines, but we have measurements on where the
    traces are in x, so that will allow us to fit some distortion
    parameters. If there aren't any lines, we won't be able to update the
    wavelength solution much, but we're probably working with a quartz so
    that doesn't matter.

    Trace measurements in the line list may be distinguished as having the
    ``description == "Trace"``.

    Parameters
    ----------
    detectorMap : `pfs.drp.stella.DetectorMap`
        Mapping of fiberId,wavelength to x,y.
    traces : `dict` mapping `int` to `list` of `pfs.drp.stella.TracePeak`
        Measured peak positions for each row, indexed by (identified)
        fiberId. These are only used if we don't have lines.
    spectralError : `float`
        Error in spectral dimension (pixels) to give lines.

    Returns
    -------
    lines : `pfs.drp.stella.ArcLineSet`
        Line measurements, treating every trace row with a centroid as a
        line.
    """
    lines = []
    data = extractTraceData(traces)
    for fiberId in data:
        dd = data[fiberId]
        num = len(dd)
        nan = np.full(num, np.nan, dtype=float)
        lines.append(ArcLineSet.fromColumns(
            fiberId=np.full(num, fiberId, dtype=int),
            wavelength=detectorMap.findWavelength(fiberId, dd[:, 0].astype(np.float32)),
            x=dd[:, 1],
            y=dd[:, 0],
            xErr=dd[:, 2],
            yErr=np.full(num, spectralError, dtype=float),
            xx=nan,
            yy=nan,
            xy=nan,
            flux=dd[:, 3],
            fluxErr=dd[:, 4],
            fluxNorm=nan,
            flag=np.full(num, False, dtype=bool),
            status=np.full(num, ReferenceLineStatus.GOOD, dtype=int),
            description=["Trace"]*num,
            transition=np.full(num, "UNKNOWN", dtype=str),
            source=np.full(num, ReferenceLineSource.NONE, dtype=int),
        ))
    return ArcLineSet.fromMultiple(*lines)
