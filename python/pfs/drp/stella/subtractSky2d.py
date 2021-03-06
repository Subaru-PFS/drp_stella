from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import astropy.io.fits

from lsst.pex.config import Config, Field, ListField, ConfigurableField
from lsst.pipe.base import Task, Struct
from lsst.afw.image import MaskedImageF

from pfs.datamodel.pfsConfig import TargetType

from .readLineList import ReadLineListTask


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
    readLineList = ConfigurableField(target=ReadLineListTask, doc="Read line list")
    mask = ListField(dtype=str, default=["BAD", "SAT", "NO_DATA"],
                     doc="Mask planes to ignore when measuring line fluxes")
    useAllFibers = Field(dtype=bool, default=False, doc="Use all fibers to measure sky?")


class SubtractSky2dTask(Task):
    """Subtract sky from 2D spectra image"""
    ConfigClass = SubtractSky2dConfig
    _DefaultName = "subtractSky2d"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("readLineList")

    def run(self, exposureList, pfsConfig, psfList, fiberTraceList, detectorMapList):
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

        Returns
        -------
        sky2d : `pfs.drp.stella.fitFocalPlane.FocalPlaneFunction`
            2D sky subtraction solution.
        imageList : `list` of `lsst.afw.image.MaskedImage`
            List of images of the sky model.
        """
        sky2d = self.measureSky(exposureList, pfsConfig, psfList, fiberTraceList, detectorMapList)
        imageList = []
        for exposure, psf, fiberTrace, detectorMap in zip(exposureList, psfList,
                                                          fiberTraceList, detectorMapList):
            image = self.subtractSky(exposure, psf, fiberTrace, detectorMap, pfsConfig, sky2d)
            imageList.append(image)
        return Struct(sky2d=sky2d, imageList=imageList)

    def measureSky(self, exposureList, pfsConfig, psfList, fiberTraceList, detectorMapList):
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

        Returns
        -------
        sky2d : pfs.drp.stella.fitFocalPlane.FocalPlaneFunction`
            2D sky subtraction solution.
        """
        skyFibers = set(pfsConfig.fiberId[pfsConfig.targetType == int(TargetType.SKY)])
        refLines = self.readLineList(exposureList[0].getMetadata())
        lines = [ref.wavelength for ref in refLines]
        measurements = defaultdict(list)  # List of fit results for each line, by wavelength
        # Fit lines one by one for now
        # Might do a simultaneous fit later
        for exp, psf, fiberTraces, detMap in zip(exposureList, psfList, fiberTraceList, detectorMapList):
            if psf is None:
                raise RuntimeError("Unable to measure 2D sky model: PSF is None")
            for ft in fiberTraces:
                fiberId = ft.fiberId
                if not self.config.useAllFibers and fiberId not in skyFibers:
                    continue

                wavelength = detMap.getWavelength(fiberId)
                wlMin = wavelength.min()
                wlMax = wavelength.max()
                for wl in lines:
                    if wl < wlMin or wl > wlMax:
                        self.log.debug("Skipping line %f for fiber %d (min=%f, max=%f)",
                                       wl, fiberId, wlMin, wlMax)
                        continue
                    try:
                        meas = self.measureLineFlux(exp, psf, ft, wl)
                    except Exception as exc:
                        self.log.debug("Failed to measure fiber %d at %f: %s", fiberId, wl, exc)
                        continue
                    measurements[wl].append(meas)

        self.log.debug("Line flux measurements: %s", measurements)

        # Combine the line measurements into a model
        model = {}
        for wl, meas in measurements.items():
            flux = np.array([mm.flux for mm in meas])
            select = np.isfinite(flux)
            if not np.any(select):
                continue
            model[wl] = flux[select].mean()  # Ignore flux errors to avoid bias

        self.log.debug("Line flux model: %s", model)

        return SkyModel(np.array(list(model.keys())), np.array(list(model.values())))

    def measureLineFlux(self, exposure, psf, fiberTrace, wavelength):
        """Measure the flux of a single line

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image containing the line to fit.
        psf : `pfs.drp.stella.SpectralPsf`
            Point-spread function model.
        fiberTrace : `pfs.drp.stella.FiberTrace`
            Profile of the fiber.
        wavelength : `float`
            Wavelength (nm) of line to fit.
        """
        psfImage = computePsfImage(psf, fiberTrace, wavelength, exposure.getBBox())
        image = exposure[psfImage.getBBox()]

        select = (image.mask.array & image.mask.getPlaneBitMask(self.config.mask)) == 0
        modelDotModel = np.sum(psfImage.array[select]**2)
        modelDotData = np.sum(psfImage.array[select]*image.image.array[select])
        modelDotModelVariance = np.sum(psfImage.array[select]**2*image.variance.array[select])
        flux = modelDotData/modelDotModel
        fluxErr = np.sqrt(modelDotModelVariance)/modelDotModel

        return Struct(flux=flux, fluxErr=fluxErr)

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
