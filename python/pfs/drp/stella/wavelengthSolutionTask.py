from collections import defaultdict
import numpy as np
import lsstDebug
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella
from pfs.drp.stella.utils import plotReferenceLines

__all__ = ["WavelengthSolutionConfig", "WavelengthSolutionTask"]


class WavelengthSolutionConfig(pexConfig.Config):
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


class WavelengthSolutionTask(pipeBase.Task):
    ConfigClass = WavelengthSolutionConfig
    _DefaultName = "wavelengthSolution"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def WavelengthSolutionInfo(self, spec, detectorMap):
        """Fit wavelength solution for a spectrum        
        """

        rows = np.arange(len(spec.wavelength), dtype='float32')
        refLines = spec.getReferenceLines()
        wavelength = np.array([rl.wavelength for rl in refLines])
        nominalPixelPos = np.empty_like(wavelength)
        PixelPos = np.empty_like(wavelength)
        fiberId = spec.getFiberId()
 
        fitWavelength = np.empty_like(wavelength)
        fitWavelengthErr = np.empty_like(wavelength)

        status = np.empty_like(wavelength, dtype=int)
        lam = detectorMap.getWavelength(fiberId)
        nmPerPix = (lam[-1] - lam[0])/(rows[-1] - rows[0])

        self.log.trace("FiberId %d, dispersion (nm/pixel) = %.3f" % (fiberId, nmPerPix))
        #
        # Unpack reference lines
        #
        for i, rl in enumerate(refLines):
        
            nominalPixelPos[i] = (rl.wavelength - lam[0])/nmPerPix
            wavelength[i] = rl.wavelength 
            PixelPos = rl.fitPosition
            fitWavelength[i] = detectorMap.findWavelength(fiberId, rl.fitPosition)
            fitWavelengthErr[i] = rl.fitPositionErr*nmPerPix
            
            status[i] = rl.status

        return pipeBase.Struct(
            fiberId = fiberId, 
        wavelength=wavelength,
        PixelPos=PixelPos
        )

    def plot(self, spec, detectorMap):
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
                yLinearResid = 0
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
                ax.plot(rows, wavelengthErr)

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

    def run(self, spectrumSet, detectorMap):
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
        if self.debugInfo.display and self.debugInfo.showArcLines:
            display = afwDisplay.Display(self.debugInfo.arc_frame)
            display.erase()

        solutions = []
        for spec in spectrumSet:
            if self.debugInfo.display:
                self.plot(spec, detectorMap)
            WLInfo = self.WavelengthSolutionInfo(spec, detectorMap)
            solutions.append(WLInfo)

        #self.measureStatistics(spectrumSet, detectorMap)

        return pipeBase.Struct(solutions=solutions)
