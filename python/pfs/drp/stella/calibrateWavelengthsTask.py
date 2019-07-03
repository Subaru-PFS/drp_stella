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
    reflines : 'float'
        reference line wavelength (nm).         
    nominalPixelPos : `float`
        Pixel position calculated with detectormap.
    fitPixelPos : `float`
        Pixel position of reference line determined by the fit
    fitWavelength : 'float'
        wavelength determined by the fit position.
    fitPixelPosErr : `float`
        Error of fit fitPixelPos 
    fitWavelengthErr : 
        Error of fit Wavelength
    wavelengthCorr : 'float'
        Chevichev fit result 
    status : int
        Flags whether the lines are fitted, clipped or reserved etc.
    """

    def __init__(self, fiberId, reflines, nominalPixelPos, fitPixelPos, fitWavelength, fitPixelPosErr, fitWavelengthErr, wavelengthCorr,status):
        return super().__init__(fiberId=fiberId, reflines=reflines, nominalPixelPos=nominalPixelPos, fitPixelPos=fitPixelPos, fitWavelength=fitWavelength, fitPixelPosErr=fitPixelPosErr, fitWavelengthErr=fitWavelengthErr, wavelengthCorr=wavelengthCorr,status=status)

class WavelengthFitData:
    """Data characterising the quality of the wavelength fit for an image

    Parameters
    ----------
    lines : `list` of `LineData`
        List of lines in the spectra.
    """
    FitsExtName = "WLFITDATA"

    def __init__(self, lines):
        self.lines = lines

    @property
    def fiberId(self):
        """Array of fiber identifiers (`numpy.ndarray` of `int`)"""
        return np.array([ll.fiberId for ll in self.lines])

    @property
    def reflines(self):
        """Array of reference lines (`numpy.ndarray` of `float`)"""
        return np.array([ll.reflines for ll in self.lines])

    @property
    def nominalPixelPos(self):
        """Array of nominalPixelPos (`numpy.ndarray` of `float`)"""
        return np.array([ll.nominalPixelPos for ll in self.lines])

    @property
    def fitPixelPos(self):
        """Array of fitted pixel positions (`numpy.ndarray` of `float`)"""
        return np.array([ll.fitPixelPos for ll in self.lines])

    @property
    def fitWavelength(self):
        """Array of fit wavelengths in nm (`numpy.ndarray` of `float`)"""
        return np.array([ll.fitWavelength for ll in self.lines])
    @property
    def fitPixelPosErr(self):
        """Array of measured position Error in nm (`numpy.ndarray` of `float`)"""
        return np.array([ll.fitPixelPosErr for ll in self.lines])

    @property
    def fitWavelengthErr(self):
        """Array of measured wavelengths Err in nm (`numpy.ndarray` of `float`)"""
        return np.array([ll.fitWavelengthErr for ll in self.lines])

    @property
    def wavelengthCorr(self):
        """Array of Chebichev fit(`numpy.ndarray` of `float`)"""
        return np.array([ll.wavelengthCorr for ll in self.lines])

    @property
    def status(self):
        """flags for status """
        return np.array([ll.status for ll in self.lines])

    def __len__(self):
        """Number of lines"""
        return len(self.lines)

    def __iter__(self):
        """Iterator"""
        return iter(self.lines)

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

        return np.array([ll.fitWavelength - ll.reflines for ll in self.lines if
                         fiberId is None or ll.fiberId == fiberId])

 
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
    def fromSpectrumSet(cls,Datalines):
        """Measure some statistics about the wavelength solution

        Parameters
        ----------
        spectrumSet : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra, with lines identified.
        """
        return cls(Datalines)

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
            hdu = fits[cls.FitsExtName]
            fiberId = hdu.data["fiberId"]
            reflines = hdu.data["reflines"]
            nominalPixelPos = hdu.data["nominalPixelPos"]
            fitPixelPos = hdu.data["fitPixelPos"]
            fitWavelength = hdu.data["fitWavelength"]
            fitPixelPosErr = hdu.data["fitPixelPosErr"]
            fitWavelengthErr = hdu.data["fitWavelengthErr"]        
            wavelengthCorr = hdu.data["wavelengthCorr"]
            status =hdu.data["status"] 
        
        return cls([LineData(*args) for args in zip(fiberId, reflines, nominalPixelPos, fitPixelPos, fitWavelength, fitPixelPosErr, fitWavelengthErr, wavelengthCorr, status)])

    def writeFits(self, filename):
        """Write to file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        hdu = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="fiberId", format="J", array=self.fiberId),
            astropy.io.fits.Column(name="reflines", format="D", array=self.reflines),
            astropy.io.fits.Column(name="nominalPixelPos", format="D", array=self.nominalPixelPos),
            astropy.io.fits.Column(name="fitPixelPos", format="D", array=self.fitPixelPos),
            astropy.io.fits.Column(name="fitWavelength", format="D", array=self.fitWavelength),
            astropy.io.fits.Column(name="fitPixelPosErr", format="D", array=self.fitPixelPosErr),
            astropy.io.fits.Column(name="fitWavelengthErr", format="D", array=self.fitWavelengthErr),
            astropy.io.fits.Column(name="wavelengthCorr", format="D", array=self.wavelengthCorr),
            astropy.io.fits.Column(name="status", format="J", array=self.status),

        ], name=self.FitsExtName)
        hdu.header["INHERIT"] = True

        fits = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(), hdu])
        with open(filename, "wb") as fd:
            fits.writeto(fd)


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

    def fitWavelengthSolution(self, spec, detectorMap,rng=np.random):
    
        """
        Fit wavelength solution for a spectrum
        
        Parameters
        ----------
        spectrumSet : `pfs.drp.stella.SpectrumSet`
            SpectrumSet to fit; updated with solution.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position; updated with solution.
        rng : `numpy.random.RandomState`
            Random number generator, for reserving some lines from the fit.

        Returns
        -------
        : wlFitData
        """

        rows = np.arange(len(spec.wavelength), dtype='float32')
        refLines = spec.getReferenceLines()

        wavelength = np.array([rl.wavelength for rl in refLines])
        status = np.empty_like(wavelength, dtype=int)
        flag = np.empty_like(wavelength, dtype=int)
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
            fitWavelength[i] = detectorMap.findWavelength(fiberId, rl.fitPosition)
            fitWavelengthErr[i] = rl.fitPositionErr*nmPerPix
            status[i] = rl.status

        # NB: "fitted" here refers to the position of the line, not whether the line was used in the
        # wavelength fit.
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
                    
                if self.config.nLinesKeptBack >= len(good):
                    self.log.warn("Number of good points %d <= nLinesKeptBack == %d; not reserving points" %
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
            resid = y - yfit
            lq, uq = np.percentile(resid[fitted], (25.0, 75.0))
            stdev = 0.741*(uq - lq)
            clipped |= fitted & (np.fabs(resid) > nSigma*np.where(yerr > stdev, yerr, stdev))
            used = used & ~clipped
                
            if used.sum() == 0:
                self.log.warn("All points were clipped for fiberId %d; disabled clipping" % fiberId)
                clipped[:] = False
                used = fitted.copy()
        #
        # Update the status flags
        #

        for i, rl in enumerate(refLines):
            flag[i] = used[i]
            if clipped[i]:
                rl.status |= rl.Status.CLIPPED                           
                flag[i]=3
            if reserved[i]:
                rl.status |= rl.Status.RESERVED
                flag[i]=5
        #
        # Correct the initial wavelength solution
        #
        spec.wavelength = detectorMap.getWavelength(fiberId) + wavelengthCorr(rows).astype('float32')
        
        rmsUsed = np.sqrt(np.sum(((y - yfit)**2)[used]))/(used.sum() - self.config.order)
        rmsReserved = np.sqrt(np.sum(((y - yfit)**2)[reserved])/reserved.sum())
        self.log.info("FiberId %4d, rms %f nm (%.3f pix) from %d (%f nm = %.3f pix for %d reserved points)" %
                      (fiberId,
                       rmsUsed,
                       rmsUsed/nmPerPix,
                       used.sum(),
                       rmsReserved,
                       rmsReserved/nmPerPix,
                       reserved.sum(),
                   ))
        
        #
        # Update the DetectorMap
        #
        if self.config.resetSlitDy:
            offsets = detectorMap.getSlitOffsets(wlData.fiberId)
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
                
        detectorMap.setWavelength(fiberId, rows, spec.wavelength)
        diff = detectorMap.getWavelength(fiberId) - spec.wavelength
        self.log.info("Fiber %d: wavelength correction %f +/- %f nm" % (fiberId, diff.mean(), diff.std()))
        
        Lines=[]
        for i, rl in enumerate(refLines):
            if (rl.status & drpStella.ReferenceLine.Status.FIT) == 0:
                continue    
            line=LineData(fiberId, wavelength[i], nominalPixelPos[i], rl.fitPosition, fitWavelength[i], rl.fitPositionErr, fitWavelengthErr[i], wavelengthCorr(nominalPixelPos[i]),flag[i])
            Lines.append(line)

        return [Lines, wavelengthCorr]



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
        wlFitData : `WavelengthFitData`
            Data on quality of the wavelength fit.
        """
        rng = np.random.RandomState(seed)  # Used for random selection of lines to reserve from the fit
        if self.debugInfo.display and self.debugInfo.showArcLines:
            display = afwDisplay.Display(self.debugInfo.arc_frame)
            display.erase()

        LineDatas = []
        solutions = []
        
        for spec in spectrumSet:
            results = self.fitWavelengthSolution(spec, detectorMap, rng)
            wavelengthCorr =results[1]
            Lines =results[0]
            if self.debugInfo.display:
                self.plot(spec, detectorMap, wavelengthCorr)
            solutions.append(wavelengthCorr)
            LineDatas.extend(Lines)
            
        wlFitData = WavelengthFitData.fromSpectrumSet(LineDatas)    
        self.measureStatistics(spectrumSet, detectorMap)        
        
        return pipeBase.Struct(solutions=solutions, wlFitData=wlFitData)

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
        solutions : `list` of `np.polynomial.chebyshev.Chebyshev`
            Wavelength solutions.
        wlFitData : `WavelengthFitData`
            Data on quality of the wavelength fit.
        """

        results = self.run(spectrumSet, detectorMap, seed=seed)
        dataRef.put(results.wlFitData, "wlFitData")
        return results
