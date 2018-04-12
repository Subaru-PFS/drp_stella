#!/usr/bin/env python
import numpy as np
import lsstDebug
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella
from pfs.drp.stella.utils import plotReferenceLines

__all__ = ["IdentifyConfig", "CalibrateWavelengthsConfig", "CalibrateWavelengthsTask"]

IdentifyConfig = pexConfig.makeConfigClass(drpStella.DispCorControl, "IdentifyConfig")


class CalibrateWavelengthsConfig(pexConfig.Config):
    identify = pexConfig.ConfigField(dtype=IdentifyConfig, doc="Configuration for line identification")
    order = pexConfig.Field(doc="Fitting function order", dtype=int, default=4);
    searchRadius = pexConfig.Field(
        doc="Radius in pixels relative to line list to search for emission line peak",
        dtype=int,
        default=5
    )
    fwhm = pexConfig.Field(doc="FWHM of emission lines", dtype=float, default=2.6);
    maxDistance = pexConfig.Field(
        doc="Reject lines with center more than maxDistance from predicted position",
        dtype=float,
        default=2.5
    )
    nLinesKeptBack = pexConfig.Field(doc="Number of lines to withhold from line fitting to estimate errors",
                                     dtype=int, default=4);
    nSigmaClip = pexConfig.ListField(doc="Number of sigma to clip points in the initial wavelength fit",
                                     dtype=float, default=[10, 5, 4, 3])
    pixelPosErrorFloor = pexConfig.Field(doc="Floor on pixel positional errors, " +
                                         "added in quadrature to quoted errors",
                                         dtype=float, default=0.05)
    resetSlitDy = pexConfig.Field(doc="Reset the slitOffset values in the DetectorMap to 0",
                                  dtype=bool, default=False)


class CalibrateWavelengthsTask(pipeBase.Task):
    ConfigClass = CalibrateWavelengthsConfig
    _DefaultName = "CalibrateWavelengthsTask"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def identifyArcLines(self, spectrumSet, detectorMap, arcLines):
        """Identify arc lines on the extracted spectra

        Parameters
        ----------
        spectrumSet : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra.
        detectorMap : `pfs.drp.stella.utils.DetectorMapIO`
            Mapping of wl,fiber to detector position.
        arcLines : `list` of `pfs.drp.stella.ReferenceLine`
            Reference arc lines.
        """
        for spec in spectrumSet:
            fiberId = spec.getFiberId()

            # Lookup the pixel positions of those lines
            for rl in arcLines:
                rl.guessedPixelPos = detectorMap.findPoint(fiberId, rl.wavelength)[1]

            # Identify emission lines and fit dispersion
            try:
                spec.identify(arcLines, self.config.identify.makeControl())
            except Exception as exc:
                self.log.warn("Failed to identify lines for fiberId %d: %s" % (fiberId, exc))
                continue

    def fitWavelengthSolution(self, spec, detectorMap, rng=np.random):
        """Fit wavelength solution for a spectrum

        Parameters
        ----------
        spec : `pfs.drp.stella.Spectrum`
            Spectrum to fit; updated with solution.
        detectorMap : `pfs.drp.stella.utils.DetectorMapIO`
            Mapping of wl,fiber to detector position; updated with solution.
        rng : `numpy.random.RandomState`
            Random number generator, for reserving some lines from the fit.

        Returns
        -------
        wavelengthCorr : `np.polynomial.chebyshev.Chebyshev`
            Wavelength solution.
        """
        rows = np.arange(len(spec.wavelength), dtype='float32')
        refLines = spec.getReferenceLines()

        wavelength = np.array([rl.wavelength for rl in refLines])
        status = np.empty_like(wavelength, dtype=int)
        nominalPixelPos = np.empty_like(wavelength)
        fitWavelength = np.empty_like(wavelength)
        fitWavelengthErr = np.empty_like(wavelength)

        fiberId = spec.getFiberId()
        refLines = spec.getReferenceLines()
        lam = detectorMap.getWavelength(fiberId)
        nmPerPix = (lam[-1] - lam[0])/(rows[-1] - rows[0])
        self.log.trace("FiberId %d, dispersion (nm/pixel) = %.3f" % (fiberId, nmPerPix))
        #
        # Unpack reference lines
        #
        for i, rl in enumerate(refLines):
            nominalPixelPos[i] = (rl.wavelength - lam[0])/nmPerPix
            fitWavelength[i] = detectorMap.findWavelength(fiberId, rl.fitPixelPos)
            fitWavelengthErr[i] = rl.fitPixelPosErr*nmPerPix
            status[i] = rl.status

        fitted = (status & drpStella.ReferenceLine.Status.FIT) != 0
        fitted = fitted & ((status & drpStella.ReferenceLine.Status.INTERPOLATED) == 0)

        nSigma = self.config.nSigmaClip[:]
        try:
            nSigma[0]
        except TypeError:
            nSigma = [nSigma]
        nSigma.append(None)         # None => don't clip on the last pass, but do reserve some values

        used = fitted.copy()        # the lines that we use in the fit
        clipped = np.zeros_like(fitted, dtype=bool)
        reserved = np.zeros_like(fitted, dtype=bool)
        for nSigma in nSigma:
            if nSigma is None:      # i.e. the last pass
                #
                # Reserve some lines to estimate the quality of the fit
                #
                good = np.where(used)[0]
                oldUsed = used.copy()

                if self.config.nLinesKeptBack >= len(good):
                    self.log.warn("No. good points %d <= nLinesKeptBack == %d; not reserving points" %
                              (len(good), self.config.nLinesKeptBack))
                else:
                    for i in rng.choice(len(good), self.config.nLinesKeptBack, replace=False):
                        used[good[i]] = False
                    
                    reserved = (fitted & ~clipped) & ~used
                    assert sum(reserved) == self.config.nLinesKeptBack
            #
            # Fit the residuals
            #
            x = nominalPixelPos
            y = wavelength - fitWavelength
            yerr = np.hypot(fitWavelengthErr, self.config.pixelPosErrorFloor*nmPerPix)

            wavelengthCorr = np.polynomial.chebyshev.Chebyshev.fit(
                x[used], y[used], self.config.order, domain=[0, len(spec.wavelength) - 1], w=1/yerr[used])
            yfit = wavelengthCorr(x)

            if nSigma is not None:
                clipped |= fitted & (np.fabs(y - yfit) > nSigma*yerr)
                used = used & ~clipped

                if used.sum() == 0:
                    self.log.warn("All points were clipped for fiberId %d; disabled clipping" % fiberId)
                    clipped[:] = False
                    used = fitted.copy()
        #
        # Update the status flags
        #
        for i, rl in enumerate(refLines):
            if clipped[i]:
                rl.status |= rl.Status.CLIPPED
            if reserved[i]:
                rl.status |= rl.Status.RESERVED
        #
        # Correct the initial wavelength solution
        #
        spec.wavelength = detectorMap.getWavelength(fiberId) + wavelengthCorr(rows).astype('float32')

        self.log.info("FiberId %4d, rms %.3fpix (%.3fpix for reserved points)" %
                      (fiberId,
                       np.sqrt(np.sum(((y - yfit)**2)[used]))/(used.sum() - self.config.order)/nmPerPix,
                       np.sqrt(np.sum(((y - yfit)**2)[reserved]))/reserved.sum()/nmPerPix,
                       ))
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
                
        detectorMap.setWavelength(fiberId, spec.wavelength)

        return wavelengthCorr


    def plot(self, spec, detectorMap, wavelengthCorr):
        """Plot fit results

        Parameters
        ----------
        spec : `pfs.drp.stella.Spectrum`
            Spectrum to fit; updated with solution.
        detectorMap : `pfs.drp.stella.utils.DetectorMapIO`
            Mapping of wl,fiber to detector position; updated with solution.
        wavelengthCorr : `np.polynomial.chebyshev.Chebyshev`
            Wavelength solution.
        """
        rows = np.arange(len(spec.wavelength), dtype='float32')
        refLines = spec.getReferenceLines()
        if self.debugInfo.display and self.debugInfo.showArcLines:
            display.dot(str(fiberId),
                        detectorMap.findPoint(fiberId, arcLines[0].wavelength)[0],
                        0.5*len(detectorMap.getXCenter(fiberId)) + 10*(fiberId%2), ctype='blue')
            
            for rl in refLines:
                xc, wl = detectorMap.findPoint(fiberId, rl.wavelength)

                if not (rl.status & rl.Status.FIT):
                    ctype = afwDisplay.BLACK
                elif (rl.status & rl.Status.RESERVED):
                    ctype = afwDisplay.BLUE
                elif (rl.status & rl.Status.SATURATED):
                    ctype = afwDisplay.MAGENTA
                elif (rl.status & rl.Status.CR):
                    ctype = afwDisplay.CYAN
                elif (rl.status & rl.Status.MISIDENTIFIED):
                    ctype = "brown"
                elif (rl.status & rl.Status.CLIPPED):
                    ctype = afwDisplay.RED
                else:
                    ctype = afwDisplay.GREEN

                display.dot('+', xc, wl, ctype=ctype)
                display.dot('x', xc, rl.fitPixelPos, ctype=ctype)

        if self.debugInfo.display and self.debugInfo.showFibers is not None:
            import matplotlib.pyplot as plt

            if self.debugInfo.showFibers and fiberId not in self.debugInfo.showFibers:
                return

            if self.debugInfo.plotWavelengthResiduals:
                # things we're going to plot
                dataItems = [(used, 'o', 'green', 'used'), #          logical, marker, colour, label
                             (reserved, 'o', 'blue', 'reserved'),
                             (clipped, '+', 'red', 'clipped'),
                ]
                #
                # x is a nominal position which we used as an index for the Chebyshev fit.
                # This makes the plot confusing, so update it
                #
                if False:
                    for i, rl in enumerate(refLines):
                        x[i] = detectorMap.findPoint(fiberId, rl.wavelength)[1]
                    yfit = wavelengthCorr(x)

                plt.figure().subplots_adjust(hspace=0)

                axes = []
                axes.append(plt.subplot2grid((3, 1), (0, 0)))
                axes.append(plt.subplot2grid((3, 1), (1, 0), rowspan=2, sharex=axes[-1]))

                ax = axes[0]
                for l, marker, color, label in dataItems:
                    ax.errorbar(x[l], (y - yfit)[l], yerr=yerr[l],
                                marker=marker, ls='none', color=color)

                ax.set_ylim(0.1*np.array([-1, 1]))
                ax.axhline(0, ls=':', color='black')
                ax.set_ylabel('residuals (nm)')

                ax.set_title("FiberId %d" % fiberId) # applies to the whole plot
                
                ax = axes[1]
                for l, marker, color, label in dataItems:
                    if l.sum() > 0: # no points confuses plt.legend()
                        ax.errorbar(x[l], y[l], yerr=yerr[l],
                                    marker=marker, ls='none', color=color, label=label)
                ax.plot(rows, wavelengthCorr(rows))

                ax.legend(loc='best')
                ax.set_xlabel('pixel') # applies to the whole plot
                ax.set_ylabel('lambda - fit (nm)')

                plt.show()

            if self.debugInfo.plotArcLinesRow:
                plt.plot(rows, spec.spectrum)
                xlim = plt.xlim()
                plotReferenceLines(spec.getReferenceLines(), "guessedPixelPos", alpha=0.1,
                                   labelLines=True, labelStatus=False)
                plotReferenceLines(spec.getReferenceLines(), "fitPixelPos", ls='-', alpha=0.5,
                                   labelLines=True, labelStatus=True)

                plt.xlim(xlim)
                plt.legend(loc='best')
                plt.xlabel('row')
                plt.title("FiberId %d" % fiberId);
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

    def run(self, spectrumSet, detectorMap, seed=1):
        """Run the wavelength calibration

        Assumes that line identification has been done already
        (i.e., call ``identifyArcLines`` before this).

        Parameters
        ----------
        spectrumSet : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra.
        detectorMap : `pfs.drp.stella.utils.DetectorMapIO`
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

        return solutions
