#!/usr/bin/env python
import numpy as np
import scipy.interpolate
import lsstDebug
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella
from pfs.drp.stella.utils import plotReferenceLines

@pexConfig.wrap(drpStella.DispCorControl) # should wrap IdentifyLinesTaskConfig when it's written
class CalibrateWavelengthsConfig(pexConfig.Config):
    order=pexConfig.Field(doc="Fitting function order", dtype=int, default=4);
    searchRadius=pexConfig.Field(doc="Radius in pixels relative to line list to search for emission line peak",
                                 dtype=int, default=5);
    fwhm=pexConfig.Field(doc="FWHM of emission lines", dtype=float, default=2.6);
    maxDistance=pexConfig.Field(doc="Reject lines with center more than maxDistance from predicted position",
                                dtype=float, default=2.5);
    nLinesKeptBack=pexConfig.Field(doc="Number of lines to withhold from line fitting to estimate errors",
                                   dtype=int, default=4);
    nSigmaClip = pexConfig.ListField(doc="Number of sigma to clip points in the initial wavelength fit",
                                     dtype=float, default=[10, 5, 4, 3])
    pixelPosErrorFloor = pexConfig.Field(doc="Floor on pixel positional errors, " +
                                         "added in quadrature to quoted errors",
                                         dtype=float, default=0.05)
    resetSlitDy = pexConfig.Field(doc="Reset the slitOffset values in the DetectorMap to 0",
                                  dtype=bool, default=False);

class CalibrateWavelengthsTask(pipeBase.Task):
    ConfigClass = CalibrateWavelengthsConfig
    _DefaultName = "CalibrateWavelengthsTask"

    def __init__(self, *args, **kwargs):
        super(CalibrateWavelengthsTask, self).__init__(*args, **kwargs)

        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, detectorMap, spectrumSet, arcLines):
        """Calibrate the SpectrumSet's wavelengths

        @param spectrumSet:  a set of spectra extracted from an image (usually an arc or sky spectrum)

        @return pipe_base Struct containing these fields:
         - spectrumSet: set of extracted spectra
        """

        if self.debugInfo.display and self.debugInfo.showArcLines:
            display = afwDisplay.Display(self.debugInfo.arc_frame)
            display.erase()
            
        # Fit the wavelength solution
        dispCorControl = self.config.makeControl()

        for spec in spectrumSet:
            fiberId = spec.getFiberId()

            # Lookup the pixel positions of those lines
            for rl in arcLines:
                rl.guessedPixelPos = detectorMap.findPoint(fiberId, rl.wavelength)[1]

            # Identify emission lines and fit dispersion
            try:
                spec.identify(arcLines, dispCorControl, 8)
            except Exception as e:
                self.log.info("FiberId %d: %s" % (fiberId), e)
                continue
        #
        # Fit the wavelength solutions
        #
        # N.b. we do this after fitting all the fibres, even though we currently do this fibre by fibre
        #
        rows = np.arange(len(spec.wavelength), dtype='float32')

        if spectrumSet.getNtrace() > 0:
            refLines = spectrumSet.getSpectrum(0).getReferenceLines()

            wavelength = np.empty(len(refLines))
            status = np.empty_like(wavelength, dtype=int)
            nominalPixelPos = np.empty_like(wavelength)
            fitWavelength = np.empty_like(wavelength)
            fitWavelengthErr = np.empty_like(wavelength)

            for i, rl in enumerate(refLines):
                wavelength[i] = rl.wavelength

        for spec in spectrumSet:
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

            fitted = (status & arcLines[0].Status.FIT) != 0
            fitted = fitted & ((status & arcLines[0].Status.INTERPOLATED) == 0)

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
                        for i in np.random.choice(len(good), self.config.nLinesKeptBack, replace=False):
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
            #
            # Debug outputs
            #
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
                    continue

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

        return pipeBase.Struct(
            spectrumSet=spectrumSet,
        )
