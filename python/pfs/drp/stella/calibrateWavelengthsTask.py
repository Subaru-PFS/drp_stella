#!/usr/bin/env python
import numpy as np
import lsstDebug
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import pfs.drp.stella as drpStella
from pfs.drp.stella.utils import plotReferenceLines

@pexConfig.wrap(drpStella.DispCorControl) # should wrap IdentifyLinesTaskConfig when it's written
class CalibrateWavelengthsConfig(pexConfig.Config):
    fittingFunction=pexConfig.Field(doc="Function for fitting the dispersion", dtype=str, default="POLYNOMIAL");
    order=pexConfig.Field(doc="Fitting function order", dtype=int, default=5);
    searchRadius=pexConfig.Field(doc="Radius in pixels relative to line list to search for emission line peak",
                                 dtype=int, default=2);
    fwhm=pexConfig.Field(doc="FWHM of emission lines", dtype=float, default=2.6);
    maxDistance=pexConfig.Field(doc="Reject arc lines with center more than maxDistance from predicted position",
                                dtype=float, default=2.5);
    nLinesKeptBack=pexConfig.Field(doc="Number of lines to withhold from line fitting to estimate errors",
                                   dtype=int, default=4);

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

        for i in range(spectrumSet.getNtrace()):
            spec = spectrumSet.getSpectrum(i)
            fiberId = spec.getITrace()

            # Lookup the pixel positions of those lines
            for rl in arcLines:
                rl.guessedPixelPos = detectorMap.findPoint(fiberId, rl.wavelength)[1]

            # Identify emission lines and fit dispersion
            try:
                spec.identify(arcLines, dispCorControl, 8)
                self.log.info("FiberId %d: spec.getDispRms() = %f" % (fiberId, spec.getDispRms()))
            except Exception as e:
                print(e)
                continue

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
                    fiberId = spec.getITrace()
                    refLines = spec.getReferenceLines()

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
                        guessedPixelPos[i] = rl.guessedPixelPos

                        status[i] = rl.status
                        fitIntensity[i] = rl.fitIntensity
                        fitPixelPos[i] = rl.fitPixelPos
                        fitPixelPosErr[i] = rl.fitPixelPosErr

                    fitted = (status & arcLines[0].Status.FIT != 0)
                    plt.errorbar(fitPixelPos[fitted], (fitPixelPos - guessedPixelPos)[fitted],
                                 xerr=fitPixelPosErr[fitted], marker='o', ls='none')
                    plt.xlabel("Wavelength (vacuum nm)")
                    plt.ylabel("(fitted - input) positions")
                    plt.title("FiberId %d" % fiberId)
                    plt.show()

        return pipeBase.Struct(
            spectrumSet=spectrumSet,
        )
