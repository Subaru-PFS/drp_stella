#!/usr/bin/env python
import numpy as np
import scipy.interpolate
import lsstDebug
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import pfs.drp.stella as drpStella
from pfs.drp.stella.utils import plotReferenceLines

@pexConfig.wrap(drpStella.DispCorControl) # should wrap IdentifyLinesTaskConfig when it's written
class CalibrateWavelengthsConfig(pexConfig.Config):
    fittingFunction=pexConfig.Field(doc="Function for fitting the dispersion", dtype=str, default="POLYNOMIAL");
    order=pexConfig.Field(doc="Fitting function order", dtype=int, default=4);
    searchRadius=pexConfig.Field(doc="Radius in pixels relative to line list to search for emission line peak",
                                 dtype=int, default=2);
    fwhm=pexConfig.Field(doc="FWHM of emission lines", dtype=float, default=2.6);
    maxDistance=pexConfig.Field(doc="Reject lines with center more than maxDistance from predicted position",
                                dtype=float, default=2.5);
    nLinesKeptBack=pexConfig.Field(doc="Number of lines to withhold from line fitting to estimate errors",
                                   dtype=int, default=4);
    nSigmaClip = pexConfig.ListField(doc="Number of sigma to clip points in the initial wavelength fit",
                                     dtype=float, default=[10, 5, 4])
    errorFloor = pexConfig.Field(doc="Floor on positional errors, added in quadrature to quoted errors",
                                 dtype=float, default=5e-2)

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
            guessedIntensity = np.empty_like(wavelength)
            guessedPixelPos = np.empty_like(wavelength)
            fitIntensity = np.empty_like(wavelength)
            fitPixelPos = np.empty_like(wavelength)
            fitPixelPosErr = np.empty_like(wavelength)

            for i, rl in enumerate(refLines):
                wavelength[i] = rl.wavelength
                guessedIntensity[i] = rl.guessedIntensity

        for spec in spectrumSet:
            fiberId = spec.getFiberId()
            refLines = spec.getReferenceLines()
            #
            # Unpack reference lines
            for i, rl in enumerate(refLines):
                guessedPixelPos[i] = rl.guessedPixelPos

                fitIntensity[i] = rl.fitIntensity
                fitPixelPos[i] = rl.fitPixelPos
                fitPixelPosErr[i] = rl.fitPixelPosErr

                status[i] = rl.status

            fitted = (status & arcLines[0].Status.FIT != 0)

            nSigma = self.config.nSigmaClip[:]
            try:
                nSigma[0]
            except TypeError:
                nSigma = [nSigma]
            nSigma.append(None)         # None => don't clip on the last pass, but do reserve some values

            used = fitted.copy()        # the lines that we use in the fit
            clipped = np.zeros_like(fitted, dtype=bool)
            reserved = np.ones_like(fitted, dtype=bool)
            for nSigma in nSigma:
                if nSigma is None:      # i.e. the last pass
                    #
                    # Reserve some lines to estimate the quality of the fit
                    #
                    good = np.where(used)[0]
                    for i in np.random.choice(len(good), self.config.nLinesKeptBack, replace=False):
                        used[good[i]] = False
                        
                    reserved = (fitted & ~clipped) & ~used
                    assert sum(reserved) == self.config.nLinesKeptBack
                #
                # Fit the residuals
                #
                x = guessedPixelPos
                y = fitPixelPos - guessedPixelPos
                yerr = np.hypot(fitPixelPosErr, self.config.errorFloor)

                wavelengthFit = np.polynomial.chebyshev.Chebyshev.fit(
                    x[used], y[used], self.config.order, domain=[0, len(spec.wavelength) - 1], w=1/yerr[used])
                yfit = wavelengthFit(x)

                if nSigma is not None:
                    nclipped = fitted & (np.fabs(y - yfit) > nSigma*yerr) # newly clipped
                    if nclipped.sum() == 0:
                        break

                    clipped |= nclipped

                    if clipped.sum() == len(clipped):
                        self.info.warn("All points were clipped for fiberId %d; disabled clipping" % fiberId)
                        clipped[:] = False

                    used = np.logical_and(used, np.logical_not(clipped))
            #
            # spec.wavelength is the wavelengths at the positions rows, and we now know the correction
            # from the nominal model in the detectorMap based on the lines.
            #
            # We use a spline to correct the wavelengths in the DetectorMap
            #
            # N.b. we could/should use this to update the DetectorMap (which I happen to know used
            # a spline internally...)
            #
            nominalWavelength = detectorMap.getWavelength(fiberId)
            correctedRows = rows - wavelengthFit(rows)

            splineFit = scipy.interpolate.UnivariateSpline(rows, nominalWavelength) # not-a-knot spline
            spec.wavelength = splineFit(correctedRows).astype('float32')

            self.log.info("FiberId %4d, rms %.3fpix (%.3fpix for reserved points)" %
                          (fiberId,
                           np.sqrt(np.sum(((y - yfit)**2)[used]))/(used.sum() - self.config.order),
                           np.sqrt(np.sum(((y - yfit)**2)[reserved]))/reserved.sum(),
                           ))

            if self.debugInfo.display and self.debugInfo.showFibers is not None:
                import matplotlib.pyplot as plt

                if self.debugInfo.showFibers and fiberId not in self.debugInfo.showFibers:
                    continue

                if self.debugInfo.plotArcLinesRow:
                    plt.plot(spec.getSpectrum())
                    plotReferenceLines(spec.getReferenceLines(), "guessedPixelPos", alpha=0.1)
                    plotReferenceLines(spec.getReferenceLines(), "fitPixelPos", ls='-', alpha=0.5)
                    plt.xlabel('row')
                    plt.title("FiberId %d" % fiberId);
                    plt.show()

                if self.debugInfo.plotArcLinesLambda:
                    plt.plot(spec.wavelength, spec.spectrum)
                    plotReferenceLines(spec.getReferenceLines(), "wavelength", ls='-', alpha=0.5)
                    plt.xlabel("Wavelength (vacuum nm)")
                    plt.title("FiberId %d" % fiberId)
                    plt.show()

                if self.debugInfo.plotPositionResiduals:
                    # things we're going to plot
                    dataItems = [(used, 'o', 'used'), #                     logical, marker, label
                                 (reserved, 'o', 'reserved'),
                                 (clipped, '+', 'clipped'),
                    ]

                    plt.figure().subplots_adjust(hspace=0)

                    axes = []
                    axes.append(plt.subplot2grid((3, 1), (0, 0)))
                    axes.append(plt.subplot2grid((3, 1), (1, 0), rowspan=2, sharex=axes[-1]))

                    ax = axes[0]
                    for l, marker, label in dataItems:
                        ax.errorbar(x[l], (y - yfit)[l], yerr=yerr[l], marker=marker, ls='none')

                    ax.set_ylim(0.5*np.array([-1, 1]))
                    ax.axhline(0, ls=':', color='black')
                    ax.set_ylabel('residuals')

                    ax.set_title("FiberId %d" % fiberId) # applies to the whole plot
                    
                    ax = axes[1]
                    for l, marker, label in dataItems:
                        if l.sum() > 0: # no points confuses plt.legend()
                            ax.errorbar(x[l], y[l], yerr=yerr[l], marker=marker, ls='none', label=label)
                    ax.plot(rows, wavelengthFit(rows))

                    ax.legend(loc='best')
                    ax.set_xlabel('pixel') # applies to the whole plot
                    ax.set_ylabel('pixel - fit')

                    plt.show()

        return pipeBase.Struct(
            spectrumSet=spectrumSet,
        )
