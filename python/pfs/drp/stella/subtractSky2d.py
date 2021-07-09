from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import astropy.io.fits

from lsst.pex.config import Config, Field
from lsst.pipe.base import Task, Struct
from lsst.afw.image import MaskedImageF

from pfs.datamodel.pfsConfig import TargetType


class SkyModel(SimpleNamespace):
    """Model of the sky flux

    This implementation is a placeholder, as it uses the average flux for each
    line, and no continuum is involved.

    Parameters
    ----------
    wavelength : `numpy.ndarray` of `float`, size ``N``
        Wavelengths of sky lines.
    flux : `numpy.ndarray` of `float`, size ``N``
        Corresponding fluxes of sky lines.
    """
    def __init__(self, wavelength, flux):
        super().__init__(wavelength=wavelength, flux=flux)

    def __call__(self, positions):
        """Evaluate the function at the provided positions

        Parameters
        ----------
        positions : `numpy.ndarray` of shape ``(M, 2)``
            Positions at which to evaluate.

        Returns
        -------
        wavelength : `numpy.ndarray` of `float`, size ``N``
            Wavelengths of sky lines.
        flux : `numpy.ndarray` of `float`, shape ``(N,M)``
            Corresponding fluxes of sky lines.
        """
        return Struct(wavelength=self.wavelength, flux=np.array([self.flux for _ in positions]))

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
        with astropy.io.fits.open(filename) as fits:
            wavelength = fits["WAVELENGTH"].data
            flux = fits["FLUX"].data
        return cls(wavelength, flux)

    def writeFits(self, filename):
        """Write to FITS file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        fits = astropy.io.fits.HDUList()
        fits.append(astropy.io.fits.ImageHDU(self.wavelength, name="WAVELENGTH"))
        fits.append(astropy.io.fits.ImageHDU(self.flux, name="FLUX"))
        with open(filename, "wb") as fd:
            fits.writeto(fd)


def computePsfImage(psf, fiberTrace, wavelength, bbox):
    """Return an image of the PSF for a fiber,wavelength

    Parameters
    ----------
    psf : subclass of `pfs.drp.stella.SpectralPsf`
        Point-spread function model.
    fiberTrace : `pfs.drp.stella.FiberTrace`
        Profile of the fiber. Also contains the fiberId and flux scaling.
    wavelength : `float`
        Wavelength (nm) at which to realise the PSF.
    bbox : `lsst.geom.Box2I`
        Bounding box of the image on which we'll realise the PSF.

    Returns
    -------
    psfImage : `lsst.afw.image.Image`
        Image of the point-spread function, with suitable ``xy0``.
    """
    psfImage = psf.computeImage(fiberTrace.fiberId, wavelength)
    psfBox = psfImage.getBBox()
    psfBox.clip(bbox)
    psfImage = psfImage[psfBox].convertF()

    throughput = np.sum(fiberTrace.trace.image.array[psfBox.getMinY():psfBox.getMaxY() + 1], axis=1)
    psfImage.array *= throughput[:, np.newaxis]

    return psfImage


class SubtractSky2dConfig(Config):
    """Configuration for SubtractSky2dTask"""
    useAllFibers = Field(dtype=bool, default=False, doc="Use all fibers to measure sky?")


class SubtractSky2dTask(Task):
    """Subtract sky from 2D spectra image"""
    ConfigClass = SubtractSky2dConfig
    _DefaultName = "subtractSky2d"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, exposureList, pfsConfig, psfList, fiberTraceList, detectorMapList, linesList):
        """Measure and subtract sky from 2D spectra image

        Parameters
        ----------
        exposureList : iterable of `lsst.afw.image.Exposure`
            Images from which to subtract sky.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for identifying sky fibers.
        psfList : iterable of PSFs (type TBD)
            Point-spread functions.
        fiberTraceList : iterable of `pfs.drp.stella.FiberTraceSet`
            Fiber traces.
        detectorMapList : iterable of `pfs.drp.stella.DetectorMap`
            Mapping of fiber,wavelength to x,y.
        linesList : iterable of `pfs.drp.stella.ArcLineSet`
            Measured sky lines.

        Returns
        -------
        sky2d : `pfs.drp.stella.fitFocalPlane.FocalPlaneFunction`
            2D sky subtraction solution.
        imageList : `list` of `lsst.afw.image.MaskedImage`
            List of images of the sky model.
        """
        sky2d = self.measureSky(exposureList, pfsConfig, psfList, fiberTraceList, detectorMapList, linesList)
        imageList = []
        for exposure, psf, fiberTrace, detectorMap in zip(exposureList, psfList,
                                                          fiberTraceList, detectorMapList):
            image = self.subtractSky(exposure, psf, fiberTrace, detectorMap, pfsConfig, sky2d)
            imageList.append(image)
        return Struct(sky2d=sky2d, imageList=imageList)

    def measureSky(self, exposureList, pfsConfig, psfList, fiberTraceList, detectorMapList, linesList):
        """Measure the 2D sky model

        Parameters
        ----------
        exposureList : iterable of `lsst.afw.image.Exposure`
            Images from which to subtract sky.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for identifying sky fibers.
        psfList : iterable of PSFs (type TBD)
            Point-spread functions.
        fiberTraceList : iterable of `pfs.drp.stella.FiberTraceSet`
            Fiber traces.
        detectorMapList : iterable of `pfs.drp.stella.DetectorMap`
            Mapping of fiber,wavelength to x,y.
        linesList : iterable of `pfs.drp.stella.ArcLineSet`
            Measured sky lines.

        Returns
        -------
        sky2d : pfs.drp.stella.fitFocalPlane.FocalPlaneFunction`
            2D sky subtraction solution.
        """
        skyFibers = set(pfsConfig.fiberId[pfsConfig.targetType == int(TargetType.SKY)])

        intensities = defaultdict(list)  # List of flux for each line, by wavelength
        # Fit lines one by one for now
        # Might do a simultaneous fit later
        for exp, psf, fiberTraces, detMap, lines in zip(exposureList, psfList, fiberTraceList,
                                                        detectorMapList, linesList):
            if psf is None:
                raise RuntimeError("Unable to measure 2D sky model: PSF is None")
            exp.setPsf(psf)
            if self.config.useAllFibers:
                select = np.ones(len(lines), dtype=bool)
            else:
                select = np.zeros(len(lines), dtype=bool)
                for ff in skyFibers:
                    select |= lines.fiberId == ff

            for wl in set(lines.wavelength[select]):
                choose = select & (lines.wavelength == wl) & ~lines.flag
                inten = lines.intensity[choose]
                if np.isfinite(inten).any():
                    intensities[wl] = inten

        self.log.debug("Line flux intensities: %s", intensities)

        # Combine the line intensities into a model
        model = {}
        for wl, inten in intensities.items():
            model[wl] = np.nanmean(inten)  # Ignore flux errors to avoid bias

        self.log.debug("Line flux model: %s", model)

        return SkyModel(np.array(list(model.keys())), np.array(list(model.values())))

    def makeSkyImage(self, bbox, psf, fiberTraces, pfsConfig, sky2d):
        """Construct a 2D image of the sky model

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box of image.
        psf : PSF (type TBD)
            Point-spread function.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Fiber traces.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for getting location of fibers.
        sky2d : `pfs.drp.stella.subtractSky2d.SkyModel`
            2D sky subtraction solution.

        Returns
        -------
        result : `lsst.afw.image.MaskedImage`
            Image of the sky model.
        """
        if psf is None:
            raise RuntimeError("Unable to construct sky image: PSF is None")
        result = MaskedImageF(bbox)
        fiberId = np.array([ft.fiberId for ft in fiberTraces])
        centers = pfsConfig.extractCenters(fiberId)
        model = sky2d(centers)
        for ii, ft in enumerate(fiberTraces):
            for wl, flux in zip(model.wavelength, model.flux[ii]):
                try:
                    psfImage = computePsfImage(psf, ft, wl, bbox)
                except Exception as exc:
                    self.log.debug("Unable to subtract line %f on fiber %d: %s", wl, ft.fiberId, exc)
                    continue

                psfImage *= flux
                result[psfImage.getBBox()] += psfImage
        return result

    def subtractSky(self, exposure, psf, fiberTraces, detectorMap, pfsConfig, sky2d):
        """Subtract the 2D sky model from the images

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image from which to subtract sky.
        psf : PSF (type TBD)
            Point-spread function.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Fiber traces.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiber,wavelength to x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for getting location of fibers.
        sky2d : `pfs.drp.stella.subtractSky2d.SkyModel`
            2D sky subtraction solution.

        Returns
        -------
        image : `lsst.afw.image.MaskedImage`
            Image of the sky model.
        """
        image = self.makeSkyImage(exposure.getBBox(), psf, fiberTraces, pfsConfig, sky2d)
        exposure.maskedImage -= image
        return image
