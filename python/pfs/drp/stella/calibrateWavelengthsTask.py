from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import astropy.io.fits

import lsstDebug
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella
from pfs.drp.stella.utils import plotReferenceLines

__all__ = ["CalibrateWavelengthsConfig", "CalibrateWavelengthsTask"]


class LineData(SimpleNamespace):
    """Data for a single reference line

    Parameters
    ----------
    fiberId : `int`
        Fiber identifier.
    measuredPosition : `float`
        Pixel position of line measured on the spectrum.
    measuredPositionErr : `float`
        Error in pixel position of line measured on the spectrum.
    xCenter : `float`
        Trace center from detectorMap.
    refWavelength : `float`
        Reference line wavelength (nm).
    fitWavelength : `float`
        Wavelength determined by the wavelength fit.
    correction : `float`
        Correction applied in wavelength fit.
    status : `pfs.drp.stella.ReferenceLine.Status`
        Flags whether the lines are fitted, clipped or reserved etc.
    """
    def __init__(self, fiberId, measuredPosition, measuredPositionErr, xCenter, refWavelength,
                 fitWavelength, correction, status):
        return super().__init__(fiberId=fiberId, measuredPosition=measuredPosition,
                                measuredPositionErr=measuredPositionErr, xCenter=xCenter,
                                refWavelength=refWavelength, fitWavelength=fitWavelength,
                                correction=correction, status=status)

    @classmethod
    def fromSpectrum(cls, spectrum, detectorMap, correction):
        """Construct a list of `LineData` from a spectrum and fit solution

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Extracted spectra, with lines identified.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        correction : function taking `float` returning `float`
            Non-linear correction applied in wavelength fit.

        Returns
        -------
        data : `list` of `LineData`
            List of line data.
        """
        fiberId = spectrum.getFiberId()
        refLines = (rl for rl in spectrum.getReferenceLines() if
                    (rl.status & drpStella.ReferenceLine.Status.FIT) != 0 and
                    (rl.status & drpStella.ReferenceLine.Status.INTERPOLATED) == 0)
        return [cls(fiberId, rl.fitPosition, rl.fitPositionErr,
                    detectorMap.findPoint(fiberId, rl.wavelength)[0], rl.wavelength,
                    detectorMap.findWavelength(fiberId, rl.fitPosition), correction(rl.fitPosition),
                    rl.status) for rl in refLines]


class WavelengthFitData:
    """Data characterising the quality of the wavelength fit for an exposure

    Parameters
    ----------
    lines : `list` of `LineData`
        List of lines in the spectra.
    """
    fitsExtName = "WLFITDATA"

    def __init__(self, lines):
        self.lines = lines

    @property
    def fiberId(self):
        """Array of fiber identifiers (`numpy.ndarray` of `int`)"""
        return np.array([ll.fiberId for ll in self.lines])

    @property
    def measuredPosition(self):
        """Array of measured position (`numpy.ndarray` of `float`)"""
        return np.array([ll.measuredPosition for ll in self.lines])

    @property
    def measuredPositionErr(self):
        """Array of error in measured position (`numpy.ndarray` of `float`)"""
        return np.array([ll.measuredPositionErr for ll in self.lines])

    @property
    def xCenter(self):
        """Array of trace center (`numpy.ndarray` of `float`)"""
        return np.array([ll.xCenter for ll in self.lines])

    @property
    def refWavelength(self):
        """Array of reference wavelength, nm (`numpy.ndarray` of `float`)"""
        return np.array([ll.refWavelength for ll in self.lines])

    @property
    def fitWavelength(self):
        """Array of fit wavelength, nm (`numpy.ndarray` of `float`)"""
        return np.array([ll.fitWavelength for ll in self.lines])

    @property
    def correction(self):
        """Array of non-linear correction (`numpy.ndarray` of `float`)"""
        return np.array([ll.correction for ll in self.lines])

    @property
    def status(self):
        """Array of status flags (`numpy.ndarray` of `int`)"""
        return np.array([ll.status for ll in self.lines])

    def __len__(self):
        """Number of lines"""
        return len(self.lines)

    def __iter__(self):
        """Iterator"""
        return iter(self.lines)

    @classmethod
    def fromSpectrum(cls, spectrum, detectorMap, correction):
        """Construct from a spectrum and fitting solution

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Extracted spectra, with lines identified.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        correction : function taking `float` returning `float`
            Non-linear correction applied in wavelength fit.

        Returns
        -------
        self : cls
            Wavelength fit data.
        """
        return cls(LineData.fromSpectrum(spectrum, detectorMap, correction))

    @classmethod
    def fromSpectrumSet(cls, spectra, detectorMap, corrections):
        """Construct from spectra and fitting solutions

        Parameters
        ----------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra, with lines identified.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        correction : iterable of function taking `float` returning `float`
            Non-linear correction applied in wavelength fit.

        Returns
        -------
        self : cls
            Wavelength fit data.
        """
        self = None
        for ss, corr in zip(spectra, corrections):
            if self is None:
                self = cls.fromSpectrum(ss, detectorMap, corr)
            else:
                self.extendFromSpectrum(ss, detectorMap, corr)
        return self

    def extendFromSpectrum(self, spectrum, detectorMap, correction):
        """Extend from a spectrum and fitting solution

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Extracted spectra, with lines identified.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        correction : function taking `float` returning `float`
            Non-linear correction applied in wavelength fit.
        """
        self.lines.extend(LineData.fromSpectrum(spectrum, detectorMap, correction))

    def residuals(self, fiberId=None):
        """Return wavelength residuals (nm)

        Parameters
        ----------
        fiberId : `int`, optional
            Fiber identifier to select.

        Returns
        -------
        residuals : `numpy.ndarray` of `float`
            Wavelength residuals (nm): measured - actual.
        """
        return np.array([ll.fitWavelength - ll.refWavelength for ll in self.lines if
                         (fiberId is None or ll.fiberId == fiberId) and
                         (ll.status & drpStella.ReferenceLine.Status.CLIPPED) == 0])

    def mean(self, fiberId=None):
        """Return the mean of wavelength residuals (nm)

        Parameters
        ----------
        fiberId : `int`, optional
            Fiber identifier to select.

        Returns
        -------
        mean : `float`
            Mean wavelength residual (nm).
        """
        return self.residuals(fiberId).mean()

    def stdev(self, fiberId=None):
        """Return the standard deviation of wavelength residuals (nm).

        Parameters
        ----------
        fiberId : `int`, optional
            Fiber identifier to select.

        Returns
        -------
        stdev : `float`
            Standard deviation of wavelength residuals (nm).
        """
        return self.residuals(fiberId).std()

    @classmethod
    def readFits(cls, filename):
        """Read from file

        Parameters
        ----------
        filename : `str`
            Name of file from which to read.

        Returns
        -------
        self : cls
            Constructed object from reading file.
        """
        with astropy.io.fits.open(filename) as fits:
            hdu = fits[cls.fitsExtName]
            fiberId = hdu.data["fiberId"]
            measuredPosition = hdu.data["measuredPosition"]
            measuredPositionErr = hdu.data["measuredPositionErr"]
            xCenter = hdu.data["xCenter"]
            refWavelength = hdu.data["refWavelength"]
            fitWavelength = hdu.data["fitWavelength"]
            correction = hdu.data["correction"]
            status = hdu.data["status"]

        return cls([LineData(*args) for args in zip(fiberId, measuredPosition, measuredPositionErr, xCenter,
                                                    refWavelength, fitWavelength, correction, status)])

    def writeFits(self, filename):
        """Write to file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        hdu = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="fiberId", format="J", array=self.fiberId),
            astropy.io.fits.Column(name="measuredPosition", format="E", array=self.measuredPosition),
            astropy.io.fits.Column(name="measuredPositionErr", format="E", array=self.measuredPositionErr),
            astropy.io.fits.Column(name="xCenter", format="E", array=self.xCenter),
            astropy.io.fits.Column(name="refWavelength", format="E", array=self.refWavelength),
            astropy.io.fits.Column(name="fitWavelength", format="E", array=self.fitWavelength),
            astropy.io.fits.Column(name="correction", format="E", array=self.correction),
            astropy.io.fits.Column(name="status", format="J", array=self.status),
        ], name=self.fitsExtName)
        hdu.header["INHERIT"] = True

        fits = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(), hdu])
        with open(filename, "wb") as fd:
            fits.writeto(fd)


class CalibrateWavelengthsConfig(pexConfig.Config):
    order = pexConfig.Field(doc="Fitting function order", dtype=int, default=3)
    nLinesKeptBack = pexConfig.Field(doc="Number of lines to withhold from line fitting to estimate errors",
                                     dtype=int, default=10)
    nSigmaClip = pexConfig.ListField(doc="Number of sigma to clip points in the initial wavelength fit",
                                     dtype=float, default=[10, 5, 4, 3])
    pixelPosErrorFloor = pexConfig.Field(doc="Floor on pixel positional errors, "
                                         "added in quadrature to quoted errors",
                                         dtype=float, default=0.05)
    resetSlitDy = pexConfig.Field(doc="Reset the slitOffset values in the DetectorMap to 0",
                                  dtype=bool, default=False)


class CalibrateWavelengthsTask(pipeBase.Task):
    ConfigClass = CalibrateWavelengthsConfig
    _DefaultName = "calibrateWavelengths"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def fitWavelengthSolution(self, spec, detectorMap, rng=np.random):
        """Fit wavelength solution for a spectrum

        Parameters
        ----------
        spec : `pfs.drp.stella.Spectrum`
            Spectrum to fit; updated with solution.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position; updated with solution.
        rng : `numpy.random.RandomState`
            Random number generator, for reserving some lines from the fit.

        Returns
        -------
        wavelengthCorr : `numpy.polynomial.chebyshev.Chebyshev`
            Non-linear fit solution.
        """
        rows = np.arange(len(spec.wavelength), dtype='float32')
        fiberId = spec.getFiberId()
        lam = detectorMap.getWavelength(fiberId)
        nmPerPix = (lam[-1] - lam[0])/(rows[-1] - rows[0])
        self.log.trace("FiberId %d, dispersion (nm/pixel) = %.3f" % (fiberId, nmPerPix))

        refLines = [rl for rl in spec.getReferenceLines() if
                    (rl.status & drpStella.ReferenceLine.Status.FIT) != 0 and
                    (rl.status & drpStella.ReferenceLine.Status.INTERPOLATED) == 0]

        #
        # Unpack reference lines
        #
        wavelength = np.array([rl.wavelength for rl in refLines])
        status = np.array([rl.status for rl in refLines])
        nominalPixelPos = np.array([(rl.wavelength - lam[0])/nmPerPix for rl in refLines])
        fitWavelength = np.array([detectorMap.findWavelength(fiberId, rl.fitPosition) for rl in refLines])
        fitWavelengthErr = np.array([rl.fitPositionErr*nmPerPix for rl in refLines])

        nSigmaClip = self.config.nSigmaClip[:]
        nSigmaClip.append(None)         # None => don't clip on the last pass, but do reserve some values

        good = np.ones_like(status, dtype=bool)
        reserved = np.zeros_like(status, dtype=bool)
        for nSigma in nSigmaClip:
            if nSigma is None:      # i.e. the last pass
                #
                # Reserve some lines to estimate the quality of the fit
                #
                goodIndices = np.where(good)[0]
                if self.config.nLinesKeptBack >= len(good):
                    self.log.warn("Number of good points %d <= nLinesKeptBack == %d; not reserving points" %
                                  (len(goodIndices), self.config.nLinesKeptBack))
                else:
                    for i in rng.choice(len(goodIndices), self.config.nLinesKeptBack, replace=False):
                        reserved[goodIndices[i]] = True
                    assert sum(reserved) == self.config.nLinesKeptBack
            #
            # Fit the residuals
            #
            x = nominalPixelPos
            y = wavelength - fitWavelength
            yerr = np.hypot(fitWavelengthErr, self.config.pixelPosErrorFloor*nmPerPix)
            use = good & ~reserved

            wavelengthCorr = np.polynomial.chebyshev.Chebyshev.fit(
                x[use], y[use], self.config.order, domain=[0, len(spec.wavelength) - 1], w=1/yerr[use])
            yfit = wavelengthCorr(x)

            if nSigma is not None:
                resid = y - yfit
                stdev = robustStdev(resid)
                good &= (np.fabs(resid) < nSigma*np.where(yerr > stdev, yerr, stdev))

                if good.sum() == 0:
                    self.log.warn("All points were clipped for fiberId %d; disabled clipping" % fiberId)
                    good[:] = True
        #
        # Update the status flags
        #
        for i, rl in enumerate(refLines):
            if not good[i]:
                rl.status |= rl.Status.CLIPPED
            if reserved[i]:
                rl.status |= rl.Status.RESERVED
        #
        # Correct the initial wavelength solution
        #
        spec.wavelength = wavelengthCorr(rows).astype('float32') + detectorMap.getWavelength(fiberId)

        rmsFit = np.sqrt(np.sum(((y - yfit)**2)[use])/(use.sum() - self.config.order))
        rmsReserved = (y[reserved] - yfit[reserved]).std()
        self.log.info("FiberId %d, rms %f nm (%.3f pix) from %d/%d "
                      "(%f nm = %.3f pix for %d reserved points), %.2f-%.2f nm",
                      fiberId, rmsFit, rmsFit/nmPerPix, good.sum(), len(refLines),
                      rmsReserved, rmsReserved/nmPerPix, reserved.sum(),
                      wavelength[good].min(), wavelength[good].max())
        #
        # Update the DetectorMap
        #
        if self.config.resetSlitDy:
            offsets = detectorMap.getSlitOffsets(fiberId)
            dy = offsets[detectorMap.FIBER_DY]
            offsets[detectorMap.FIBER_DY] = 0
            detectorMap.setSlitOffsets(fiberId, offsets)

            if dy > 0:
                dy = int(dy)
                spec.wavelength[:-dy] = spec.wavelength[dy:]
            elif dy == 0:
                pass
            else:
                dy = -int(-dy)
                spec.wavelength[dy:] = spec.wavelength[:-dy]

        diff = detectorMap.getWavelength(fiberId) - spec.wavelength
        self.log.info("Fiber %d: wavelength correction %f +/- %f nm" % (fiberId, diff.mean(), diff.std()))
        detectorMap.setWavelength(fiberId, rows, spec.wavelength)

        return wavelengthCorr

    def plot(self, spec, detectorMap, wavelengthCorr):
        """Plot fit results

        Parameters
        ----------
        spec : `pfs.drp.stella.Spectrum`
            Spectrum to fit; updated with solution.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position; updated with solution.
        wavelengthCorr : `np.polynomial.chebyshev.Chebyshev`
            Wavelength solution.
        """
        rows = np.arange(len(spec.wavelength), dtype='float32')
        refLines = spec.getReferenceLines()
        fiberId = spec.getFiberId()

        if self.debugInfo.display and self.debugInfo.showFibers is not None:
            import matplotlib.pyplot as plt

            if self.debugInfo.showFibers and fiberId not in self.debugInfo.showFibers:
                return

            if self.debugInfo.plotWavelengthResiduals:
                #
                # x is a nominal position which we used as an index for the Chebyshev fit.
                # This makes the plot confusing, so update it
                #
                x = np.array([rl.fitPosition for rl in refLines])  # Pixel position
                yTrue = np.array([rl.wavelength for rl in refLines])  # True position of line
                yLinearResid = wavelengthCorr(x)
                yFit = np.array([detectorMap.findWavelength(fiberId, rl.fitPosition) for rl in refLines])
                yResid = yFit - yTrue
                status = np.array([rl.status for rl in refLines])

                # things we're going to plot: logical, marker, colour, label
                dataItems = [((status & drpStella.ReferenceLine.FIT) > 0, 'o', 'green', 'used'),
                             ((status & drpStella.ReferenceLine.RESERVED) > 0, 'o', 'blue', 'reserved'),
                             ((status & drpStella.ReferenceLine.CLIPPED) > 0, '+', 'red', 'clipped'),
                             ]

                plt.figure().subplots_adjust(hspace=0)

                axes = []
                axes.append(plt.subplot2grid((4, 1), (0, 0)))
                axes.append(plt.subplot2grid((4, 1), (1, 0), sharex=axes[-1]))
                axes.append(plt.subplot2grid((4, 1), (2, 0), rowspan=2, sharex=axes[-1]))

                ax = axes[0]
                for l, marker, color, label in dataItems:
                    ax.errorbar(x[l], yLinearResid[l] + yResid[l], marker=marker, ls='none', color=color)
                ax.plot(rows, wavelengthCorr(rows))

                ax.axhline(0, ls=':', color='black')
                ax.set_ylabel('Linear fit residuals (nm)')

                ax.set_title("FiberId %d" % fiberId)  # applies to the whole plot

                ax = axes[1]
                for l, marker, color, label in dataItems:
                    if l.sum() > 0:  # no points confuses plt.legend()
                        ax.errorbar(x[l], yResid[l], marker=marker, ls='none', color=color, label=label)
                ax.axhline(0, ls=':', color='black')
                ax.set_ylabel("Fit residuals (nm)")

                ax = axes[2]
                for l, marker, color, label in dataItems:
                    if l.sum() > 0:  # no points confuses plt.legend()
                        ax.errorbar(x[l], yTrue[l], marker=marker, ls='none', color=color, label=label)
                ax.plot(rows, spec.wavelength)

                ax.legend(loc='best')
                ax.set_xlabel('pixel')  # applies to the whole plot
                ax.set_ylabel('wavelength (nm)')

                plt.show()

            if self.debugInfo.plotArcLinesRow:
                plt.plot(rows, spec.spectrum)
                xlim = plt.xlim()
                plotReferenceLines(spec.getReferenceLines(), "guessedPosition", alpha=0.1,
                                   labelLines=True, labelStatus=False)
                plotReferenceLines(spec.getReferenceLines(), "fitPosition", ls='-', alpha=0.5,
                                   labelLines=True, labelStatus=True)

                plt.xlim(xlim)
                plt.legend(loc='best')
                plt.xlabel('row')
                plt.title("FiberId %d" % fiberId)
                plt.show()

            if self.debugInfo.plotArcLinesLambda:
                plt.plot(spec.wavelength, spec.spectrum)
                xlim = plt.xlim()
                plotReferenceLines(spec.getReferenceLines(), "wavelength", ls='-', alpha=0.5,
                                   labelLines=True, wavelength=spec.wavelength, spectrum=spec.spectrum)
                plt.xlim(xlim)
                plt.legend(loc='best')
                plt.xlabel("Wavelength (vacuum nm)")
                plt.title("FiberId %d" % fiberId)
                plt.show()

    def measureStatistics(self, wlFitData):
        """Measure some statistics about the solution

        Parameters
        ----------
        wlFitDta : `WavelengthFitData`
            Lines used in wavelength fits.
        """
        fiberId = wlFitData.fiberId
        refWavelength = wlFitData.refWavelength
        fitWavelength = wlFitData.fitWavelength

        lines = defaultdict(list)
        for ff in sorted(set(wlFitData.fiberId)):
            select = fiberId == ff
            for ref, meas in zip(refWavelength[select], fitWavelength[select]):
                lines[ref].append(meas)
        for actualWl in sorted(lines.keys()):
            fitWl = np.array(lines[actualWl]) - actualWl
            self.log.debug("Line %f: %f +/- %f from %d" % (actualWl, fitWl.mean(), fitWl.std(), len(fitWl)))

    def run(self, spectrumSet, detectorMap, seed=1):
        """Run the wavelength calibration

        Assumes that line identification has been done already.

        Parameters
        ----------
        spectrumSet : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        seed : `int`
            Seed for random number generator.

        Returns
        -------
        wlFitData : `WavelengthFitData`
            Data on quality of the wavelength fit.
        """
        rng = np.random.RandomState(seed)  # Used for random selection of lines to reserve from the fit
        if self.debugInfo.display and self.debugInfo.showArcLines:
            display = afwDisplay.Display(self.debugInfo.arc_frame)
            display.erase()

        corrections = []
        for spec in spectrumSet:
            wavelengthCorr = self.fitWavelengthSolution(spec, detectorMap, rng)
            corrections.append(wavelengthCorr)
            if self.debugInfo.display:
                self.plot(spec, detectorMap, wavelengthCorr)

        wlFitData = WavelengthFitData.fromSpectrumSet(spectrumSet, detectorMap, corrections)
        self.measureStatistics(wlFitData)

        return wlFitData

    def runDataRef(self, dataRef, spectrumSet, detectorMap, seed=1):
        """Run the wavelength calibration

        Assumes that line identification has been done already.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Butler data reference.
        spectrumSet : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        seed : `int`
            Seed for random number generator.

        Returns
        -------
        wlFitData : `WavelengthFitData`
            Data on quality of the wavelength fit.
        """
        wlFitData = self.run(spectrumSet, detectorMap, seed=seed)
        dataRef.put(wlFitData, "wlFitData")
        return wlFitData


def robustStdev(array):
    """Calculate a robust standard deviation

    From the inter-quartile range.

    Parameters
    ----------
    array : array-like
        Array of values to use in calculation.

    Returns
    -------
    stdev : `float`
        Robust standard deviation.
    """
    lq, uq = np.percentile(array, (25.0, 75.0))
    return 0.741*(uq - lq)
