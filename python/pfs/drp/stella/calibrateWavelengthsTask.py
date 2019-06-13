from collections import defaultdict
import numpy as np
import lsstDebug
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella
from pfs.drp.stella.utils import plotReferenceLines

__all__ = ["CalibrateWavelengthsConfig", "CalibrateWavelengthsTask"]


class CalibrateWavelengthsConfig(pexConfig.Config):
    order = pexConfig.Field(doc="Fitting function order", dtype=int, default=6)
    nLinesKeptBack = pexConfig.Field(doc="Number of lines to withhold from line fitting to estimate errors",
                                     dtype=int, default=4)
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
        wavelengthCorr : `np.polynomial.chebyshev.Chebyshev`
            Wavelength solution.
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
        fitPosition = np.array([rl.fitPosition for rl in refLines])
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
                      "(%f nm = %.3f pix for %d reserved points)",
                      fiberId, rmsFit, rmsFit/nmPerPix, good.sum(), len(refLines),
                      rmsReserved, rmsReserved/nmPerPix, reserved.sum())
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

    def measureStatistics(self, spectrumSet, detectorMap):
        """Measure some statistics about the solution

        Parameters
        ----------
        spectrumSet : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra, with lines identified.
        """
        lines = defaultdict(list)
        for spec in spectrumSet:
            fiberId = spec.fiberId
            for rl in spec.getReferenceLines():
                if (rl.status & drpStella.ReferenceLine.Status.FIT) == 0:
                    continue
                wl = detectorMap.findWavelength(fiberId, rl.fitPosition)
                lines[rl.wavelength].append(wl)
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
        solutions : `list` of `np.polynomial.chebyshev.Chebyshev`
            Wavelength solutions.
        """
        rng = np.random.RandomState(seed)  # Used for random selection of lines to reserve from the fit
        if self.debugInfo.display and self.debugInfo.showArcLines:
            display = afwDisplay.Display(self.debugInfo.arc_frame)
            display.erase()

        solutions = []
        for spec in spectrumSet:
            wavelengthCorr = self.fitWavelengthSolution(spec, detectorMap, rng)
            if self.debugInfo.display:
                self.plot(spec, detectorMap, wavelengthCorr)
            solutions.append(wavelengthCorr)

        self.measureStatistics(spectrumSet, detectorMap)

        return pipeBase.Struct(solutions=solutions)


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
