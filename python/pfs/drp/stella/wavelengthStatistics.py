import types

import lsst.pex.config as pexConfig
from lsst.pipe.base import Task
import lsstDebug

import numpy as np
import lsst.pipe.base as pipeBase
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella
from pfs.drp.stella.utils import plotReferenceLines

class WavelengthSolutionInfo(types.SimpleNamespace):

    def __init__(self, spec, detectorMap):
        print("init")
        super().__init__(spec=spec)
        super().__init__(detectorMap=detectorMap)
        self.debugInfo = lsstDebug.Info(__name__)

    def __call__(self, spec, detectorMap):
        print("call")
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

        fiberId = spec.getFiberId()            
        lam = detectorMap.getWavelength(fiberId)
        nmPerPix = (lam[-1] - lam[0])/(rows[-1] - rows[0])
  
        #
        # Unpack reference lines
        #  
        results = []

        for i, rl in enumerate(refLines):
            nominalPixelPos[i] = (rl.wavelength - lam[0])/nmPerPix
            wavelength[i] = rl.wavelength 
            PixelPos[i] = rl.fitPosition
            fitWavelength[i] = detectorMap.findWavelength(fiberId, rl.fitPosition)
            fitWavelengthErr[i] = rl.fitPositionErr*nmPerPix
        
        #results.append([fiberId, wavelength,fitWavelength,fitWavelengthErr,PixelPos])
        results.append([wavelength,fitWavelength,fitWavelengthErr,PixelPos])    

        return results

        """
        return pipeBase.Struct(
            fiberId = fiberId, 
        wavelength=wavelength,
        PixelPos=PixelPos
        )
        """
        #return [fiberId, wavelength,fitWavelength]

    @classmethod
    def readFits(cls, filename):
        """Read from FITS file

        Parameters
        ----------
        filename : `str`
            Filename to read.

        Returns
        -------
        self : `FocalPlaneFunction`
            Function read from FITS file.
        """

    def writeFits(self, filename):
        """Write to FITS file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        from astropy.table import Table 
        import astropy.io.fits
        fits = astropy.io.fits.HDUList()
        fits.append(astropy.io.fits.PrimaryHDU(header=astropyHeaderFromDict(header)))
        for attr in ("fiberId", "wavelength", "pos", "pos2", "pos3", "pos4"):
            hduName = attr.upper()
            data = getattr(self, attr)
            fits.append(astropy.io.fits.ImageHDU(data, name=hduName))
        with open(filename, "w") as fd:
            fits.writeto(fd)
                          
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

    def run(self, spec, detectorMap):
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

        wavelengthSolutionInfo=WavelengthSolutionInfo(spec, detectorMap)
        
        return wavelengthSolutionInfo(spec, detectorMap)

        #solutions.append(wavelengthSolutionInfo(spec, detectorMap))
 
