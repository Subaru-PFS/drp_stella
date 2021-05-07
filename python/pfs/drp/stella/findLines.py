import numpy as np

from lsst.pex.config import Config, Field, ListField, ConfigurableField
from lsst.pipe.base import Task, Struct
from lsst.afw.geom import SpanSet

from .fitContinuum import FitContinuumTask
from .fitLine import fitLine
from . import Spectrum
from .utils.psf import sigmaToFwhm

import lsstDebug

__all__ = ["FittingError", "FindLinesConfig", "FindLinesTask"]


class FittingError(RuntimeError):
    """Exception indicating that line fitting failed"""
    pass


class FindLinesConfig(Config):
    """Configuration for FindLinesTask"""
    threshold = Field(dtype=float, default=5.0, doc="Threshold for line detection (sigma)")
    mask = ListField(dtype=str, default=["NO_DATA"], doc="Mask planes to ignore")
    width = Field(dtype=float, default=1.0, doc="Guess width of line (stdev, pixels)")
    kernelHalfSize = Field(dtype=float, default=4.0, doc="Half-size of kernel, in units of the width")
    fittingRadius = Field(dtype=float, default=10.0,
                          doc="Radius of fitting region for centroid as a multiple of 'width'")
    exclusionRadius = Field(dtype=float, default=3.0,
                            doc="Fit exclusion radius for pixels around other peaks, "
                                "as a multiple of 'width'")
    maskRadius = Field(dtype=float, default=1.0, doc="Mask grow radius, as a multiple of 'width'")
    doSubtractContinuum = Field(dtype=bool, default=True, doc="Subtract continuum before finding peaks?")
    fitContinuum = ConfigurableField(target=FitContinuumTask, doc="Fit continuum")


class FindLinesTask(Task):
    ConfigClass = FindLinesConfig
    _DefaultName = "findLines"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fitContinuum")

    def run(self, spectrum):
        """Find and fit emission lines in a spectrum

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum on which to find peaks.

        Returns
        -------
        lines : `list` of `lsst.pipe.base.Struct`
            List of fit parameters for each line, including ``center``,
            ``amplitude``, ``width``, ``fwhm``, ``flux``, ``backgroundSlope``,
            ``backgroundIntercept`` (see ``fitSingleLine`` method outputs).
        continuum : `numpy.ndarray`
            Array continuum fit.
        """
        continuum = self.fitContinuum.fitContinuum(spectrum) if self.config.doSubtractContinuum else None
        convolved = self.convolve(spectrum, continuum=continuum)
        peaks = self.findPeaks(convolved)
        lines = self.fitLines(spectrum, peaks)
        return Struct(lines=lines, continuum=continuum)

    def runCentroids(self, spectrum):
        """Find and fit centroids to lines in a spectrum

        This method is a convenience for obtaining just the centroids. It
        implements the interface for the ``run`` method from before the
        `FindLinesTask` was expanded to provide the full fit results for each
        line. It also includes a debug plot of the centroids on the spectrum.

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum on which to find peaks.

        Returns
        -------
        centroids : `list` of `float`
            Centroid for each line.
        errors : `list` of `float`
            Centroid error for each line.
        continuum : `numpy.ndarray`
            Array continuum fit.
        """
        result = self.run(spectrum)
        centroids = [ll.center for ll in result.lines]
        errors = [ll.centerErr for ll in result.lines]

        if lsstDebug.Info(__name__).plotCentroids:
            import matplotlib.pyplot as plt
            figure, axes = plt.subplots()
            indices = np.arange(len(spectrum))
            axes.plot(indices, spectrum.spectrum, 'k-')
            for cc in centroids:
                axes.axvline(cc, color="r", linestyle=":")
            plt.show()

        return Struct(centroids=centroids, errors=errors, continuum=result.continuum)

    def convolve(self, spectrum, continuum=None):
        """Convolve a spectrum by the estimated LSF

        We use a Gaussian approximation to the LSF for speed and convenience.
        The variance is convolved, but the co-variance is not.

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum on which to find peaks.
        continuum : `numpy.ndarray` of `float`, optional
            Continuum to subtract before finding peaks.

        Returns
        -------
        convolved : `pfs.drp.stella.Spectrum`
            Convolved spectrum.
        """
        halfSize = int(self.config.kernelHalfSize*self.config.width + 0.5)
        size = 2*halfSize + 1
        xx = np.arange(size, dtype=float) - halfSize
        sigma = self.config.width
        kernel = np.exp(-0.5*xx**2/sigma**2)/sigma/np.sqrt(2.0*np.pi)

        flux = np.convolve(spectrum.spectrum if continuum is None else spectrum.spectrum - continuum,
                           kernel, mode="same")
        covariance = np.zeros_like(spectrum.covariance)
        covariance[0, :] = np.convolve(spectrum.variance, kernel**2, mode="same")
        background = np.convolve(spectrum.background, kernel, mode="same")

        # Expand each mask plane
        grow = int(self.config.maskRadius*self.config.width + 0.5)
        mask = spectrum.mask.clone()
        for plane in mask.getMaskPlaneDict():
            value = mask.getPlaneBitMask(plane)
            SpanSet.fromMask(mask, value).dilated(grow).clippedTo(mask.getBBox()).setMask(mask, value)
        mask.array[0, :halfSize] |= mask.getPlaneBitMask("NO_DATA")
        mask.array[0, len(spectrum) - halfSize:] |= mask.getPlaneBitMask("NO_DATA")

        return Spectrum(flux, mask, background, covariance, spectrum.wavelength, spectrum.fiberId)

    def findPeaks(self, spectrum):
        """Find positive peaks in the spectrum

        Peak flux must exceed ``threshold`` config parameter.

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum on which to find peaks.

        Returns
        -------
        indices : `numpy.ndarray` of `int`
            Indices of peaks.
        """
        flux = spectrum.spectrum
        with np.errstate(invalid='ignore', divide="ignore"):
            stdev = np.sqrt(spectrum.variance)
            diff = flux[1:] - flux[:-1]  # flux[i + 1] - flux[i]
            select = (diff[:-1] > 0) & (diff[1:] < 0)  # A positive peak
            select &= flux[1:-1]/stdev[1:-1] > self.config.threshold  # Over threshold
        if self.config.mask:
            maskVal = spectrum.mask.getPlaneBitMask(self.config.mask)
            mask = spectrum.mask.array[0]
            select &= ((mask[:-2] | mask[1:-1] | mask[2:]) & maskVal) == 0  # Not masked either side

        indices = np.nonzero(select)[0] + 1  # +1 to account for the definition of diff
        self.log.debug("Found peaks: %s", indices)

        if lsstDebug.Info(__name__).plotPeaks:
            import matplotlib.pyplot as plt
            figure, axes = plt.subplots()
            axes.plot(np.arange(len(flux)), flux/stdev, 'k-')
            for xx in indices:
                axes.axvline(xx, color="r", linestyle=":")
            plt.show()

        return indices

    def fitLines(self, spectrum, peaks, ignoreFittingError=True):
        """Fit all lines in the spectrum

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum on which to fit the line.
        peaks : iterable of `int`, optional
            List of the pixel indices of all peaks.
        ignoreFittingErrors : `bool`, optional
            Ignore fitting errors? If ``True``, lines that generate a
            `FittingError` are dropped; otherwise, the exception will propagate.

        Raises
        ------
        FittingError
            If ``ignoreFittingErrors=False`` and a line fit fails.

        Returns
        -------
        lines : `list` of `lsst.pipe.base.Struct`
            List of fit parameters for each line, including ``center``,
            ``amplitude``, ``width``, ``fwhm``, ``flux``, ``backgroundSlope``,
            ``backgroundIntercept`` (see ``fitSingleLine`` method outputs).
        """
        lines = []
        for pp in peaks:
            try:
                fit = self.fitSingleLine(spectrum, pp, peaks)
            except FittingError as exc:
                if not ignoreFittingError:
                    raise
                self.log.debug(f"Ignoring line {pp}: {exc}")
                continue
            lines.append(fit)
        return lines

    def interloperPixels(self, peak, allPeaks, lowIndex, highIndex):
        """Identify pixels belonging to an interloping peak

        Since we fit only one line at a time, we need to mask pixels that belong
        to a peak other than the peak that we're fitting ("interlopers").
        Pixels within ``exclusionRadius`` (specified as multiples of the guess
        line ``width``) are flagged.

        Parameters
        ----------
        peak : `int`
            Pixel index of the peak of interest.
        allPeaks : iterable of `int`
            List of the pixel indices of all peaks.
        lowIndex : `int`
            Pixel index of lower bound of fit.
        highIndex : `int`
            Pixel index of upper bound of fit.

        Returns
        -------
        isInterloper : `numpy.ndarray` of `bool`
            Array indicating whether the pixel is affected by an interloper.
        """
        exclusionRadius = int(self.config.exclusionRadius*self.config.width + 0.5)
        interlopers = np.nonzero((allPeaks >= lowIndex - exclusionRadius) &
                                 (allPeaks < highIndex + exclusionRadius) &
                                 (allPeaks != peak))[0]
        isInterloper = np.zeros(highIndex - lowIndex, dtype=bool)
        for ii in allPeaks[interlopers]:
            lowBound = max(lowIndex, int(ii) - exclusionRadius) - lowIndex
            highBound = min(highIndex, int(ii) + exclusionRadius) - lowIndex
            isInterloper[lowBound:highBound] = True
        return isInterloper

    def fitSingleLine(self, spectrum, peak, allPeaks=None):
        """Fit a single line in the spectrum

        We fit a Gaussian plus a linear background to the ``centroidRadius``
        (specified as a mutliple of the guess line ``width``) pixels either side
        of the peak. If ``allPeaks`` are provided, the pixels belonging to other
        peaks are masked out of the fit.

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum on which to fit the line.
        peak : `int`
            Pixel index of the peak of the line of interest.
        allPeaks : iterable of `int`, optional
            List of the pixel indices of all peaks.

        Raises
        ------
        FittingError
            If the fit fails.

        Returns
        -------
        center : `float`
            Fit center of the line.
        amplitude : `float`
            Fit amplitude of the line.
        width : `float`
            Fit width (as a standard deviation) of the line.
        fwhm : `float`
            Derived FWHM of the line.
        flux : `float`
            Derived integrated flux of the line.
        backgroundSlope : `float`
            Fit slope of the background.
        backgroundIntercept : `float`
            Fit intercept of the background.
        rms : `float`
            RMS residual.
        num : `int`
            Number of values in fit.
        """
        fittingRadius = int(self.config.fittingRadius*self.config.width + 0.5)
        lowIndex = max(int(peak) - fittingRadius, 0)
        highIndex = min(int(peak) + fittingRadius, len(spectrum))

        mask = spectrum.mask.array[0]
        maskVal = spectrum.mask.getPlaneBitMask(self.config.mask)
        interloper = 1 << spectrum.mask.addMaskPlane("INTERLOPER")
        if allPeaks is not None:
            isInterloper = self.interloperPixels(peak, allPeaks, lowIndex, highIndex)
            if np.any(isInterloper):
                mask[lowIndex:highIndex][isInterloper] |= interloper

        try:
            result = fitLine(spectrum, peak, self.config.width, maskVal | interloper, fittingRadius)
        except Exception as exc:
            raise FittingError(f"Failure to fit line for peak at {peak}") from exc
        finally:
            mask[lowIndex:highIndex] &= ~interloper
            spectrum.mask.removeAndClearMaskPlane("INTERLOPER", True)
        if not result.isValid:
            raise FittingError(f"Invalid fit result for peak at {peak}")

        if lsstDebug.Info(__name__).plotCentroidLines:
            def fit(xx):
                """Calculate the fit values as a function of pixel index"""
                gaussian = result.amplitude*np.exp(-0.5*((xx - result.center)/result.width)**2)
                background = result.bg0 + result.bg1*(xx - result.center)
                return gaussian + background

            import matplotlib.pyplot as plt
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)
            indices = np.arange(lowIndex, highIndex)
            axes.plot(indices, spectrum.spectrum[lowIndex:highIndex], "k-")
            good = (mask[lowIndex:highIndex] & maskVal) == 0
            if good.sum() != len(good):
                axes.plot(indices[~good], spectrum.spectrum[lowIndex:highIndex][~good], "rx")
            if allPeaks is not None and np.any(isInterloper):
                axes.plot(indices[isInterloper], spectrum.spectrum[lowIndex:highIndex][isInterloper], "bx")
            xx = np.arange(lowIndex, highIndex, 0.01)
            axes.plot(xx, fit(xx), "b--")
            axes.axvline(result.center, color="b", linestyle=":")
            axes.set_xlabel("Index")
            axes.set_ylabel("Flux")
            plt.show()

        if result.center < lowIndex or result.center > highIndex:
            raise FittingError(f"Fit center ({result.center}) is out of range ({lowIndex}, {highIndex})")

        return Struct(
            center=result.center,
            centerErr=result.centerErr,
            amplitude=result.amplitude,
            amplitudeErr=result.amplitudeErr,
            width=result.rmsSize,
            fwhm=sigmaToFwhm(result.rmsSize),
            flux=result.amplitude*result.rmsSize*np.sqrt(2*np.pi),
            backgroundIntercept=result.bg0,
            backgroundSlope=result.bg1,
            rms=result.rms,
            num=result.num,
        )
