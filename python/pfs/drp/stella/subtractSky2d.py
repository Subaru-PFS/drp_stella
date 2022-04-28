from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import astropy.io.fits

from lsst.pex.config import Config, ListField, ConfigurableField
from lsst.pipe.base import Task, Struct
from lsst.afw.image import MaskedImageF, makeMaskedImage

from pfs.datamodel.pfsConfig import PfsConfig
from pfs.drp.stella.referenceLine import ReferenceLineStatus

from .fitFocalPlane import FitBlockedOversampledSplineTask
from .apertureCorrections import calculateApertureCorrection
from .selectFibers import SelectFibersTask


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
    selectFibers = ConfigurableField(target=SelectFibersTask, doc="Select sky fibers")

    def setDefaults(self):
        super().setDefaults()
        self.selectFibers.targetType = ("SKY", "SUNSS_DIFFUSE")


class SubtractSky2dTask(Task):
    """Subtract sky from 2D spectra image"""
    ConfigClass = SubtractSky2dConfig
    _DefaultName = "subtractSky2d"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("selectFibers")

    def run(self, exposureList, pfsConfig, psfList, fiberTraceList, detectorMapList, linesList, apCorrList):
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
            Measured sky lines. The ``status`` members will be updated to
            indicate which lines were used.
        apCorrList : iterable of `pfs.drp.stella.FocalPlaneFunction`
            Aperture corrections.

        Returns
        -------
        sky2d : `pfs.drp.stella.fitFocalPlane.FocalPlaneFunction`
            2D sky subtraction solution.
        imageList : `list` of `lsst.afw.image.MaskedImage`
            List of images of the sky model.
        """
        sky2d = self.measureSky(pfsConfig, linesList)
        imageList = []
        for exposure, psf, fiberTrace, detectorMap, apCorr in zip(exposureList, psfList, fiberTraceList,
                                                                  detectorMapList, apCorrList):
            image = self.subtractSky(exposure, psf, fiberTrace, detectorMap, pfsConfig, sky2d, apCorr)
            imageList.append(image)
        return Struct(sky2d=sky2d, imageList=imageList)

    def measureSky(self, pfsConfig, linesList):
        """Measure the 2D sky model

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for identifying sky fibers.
        linesList : iterable of `pfs.drp.stella.ArcLineSet`
            Measured sky lines. The ``status`` members will be updated to
            indicate which lines were used.

        Returns
        -------
        sky2d : pfs.drp.stella.fitFocalPlane.FocalPlaneFunction`
            2D sky subtraction solution.

        Raises
        ------
        RuntimeError
            If there are no sky fibers found.
        """
        skyFibers = self.selectFibers.run(pfsConfig).fiberId
        if skyFibers is None or len(skyFibers) == 0:
            raise RuntimeError('No sky fibers found.')
        intensities = defaultdict(list)  # List of flux for each line, by wavelength
        # Fit lines one by one for now
        # Might do a simultaneous fit later
        for lines in linesList:
            select = np.isin(lines.fiberId, skyFibers)

            for wl in set(lines.wavelength[select]):
                choose = select & (lines.wavelength == wl) & ~lines.flag
                lines.status[choose] |= ReferenceLineStatus.SKYSUB_USED
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

    def makeSkyImage(self, bbox, psf, fiberTraces, pfsConfig, sky2d, apCorr):
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
        apCorr : `pfs.drp.stella.FocalPlaneFunction`
            Aperture correction.

        Returns
        -------
        result : `lsst.afw.image.MaskedImage`
            Image of the sky model.
        """
        if psf is None:
            raise RuntimeError("Unable to construct sky image: PSF is None")
        result = MaskedImageF(bbox)

        # Compute essentially the intersection of fiberIds and traces from the 3 input sources:
        # 1. the pfsConfig
        # 2. the aperture correction object
        # 3. the input fiberTraceSet.
        fiberIdsFromConfigApCorr = set(pfsConfig.fiberId).intersection(apCorr.fiberId)
        fiberTracesMatched = [ft for ft in fiberTraces if ft.fiberId in fiberIdsFromConfigApCorr]
        fiberIdMatched = np.array([ft.fiberId for ft in fiberTracesMatched])

        excludedFiberIds = list(set(pfsConfig.fiberId) - set(fiberIdMatched))
        self.log.debugf('pfsConfig fiberIds {}, with corresponding fiberStatus {} '
                        'are excluded from sky model image generation.',
                        excludedFiberIds,
                        pfsConfig.fiberStatus[np.isin(pfsConfig.fiberId, excludedFiberIds)].tolist())

        centers = pfsConfig.extractCenters(fiberIdMatched)
        model = sky2d(centers)
        for ii, ft in enumerate(fiberTracesMatched):

            # Current fluxes are aperture-corrected, but the flux we put down will be a PSF flux,
            # so we need to remove the aperture correction.
            psfFlux = calculateApertureCorrection(apCorr, ft.fiberId, model.wavelength, pfsConfig,
                                                  model.flux[ii], invert=True)
            for wl, flux in zip(model.wavelength, psfFlux.flux):
                try:
                    psfImage = computePsfImage(psf, ft, wl, bbox)
                except Exception as exc:
                    self.log.debug("Unable to subtract line %f on fiber %d: %s", wl, ft.fiberId, exc)
                    continue
                psfImage *= flux
                result[psfImage.getBBox()] += psfImage
        return result

    def subtractSky(self, exposure, psf, fiberTraces, detectorMap, pfsConfig, sky2d, apCorr):
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
        apCorr : `pfs.drp.stella.FocalPlaneFunction`
            Aperture correction.

        Returns
        -------
        image : `lsst.afw.image.MaskedImage`
            Image of the sky model.
        """
        image = self.makeSkyImage(exposure.getBBox(), psf, fiberTraces, pfsConfig, sky2d, apCorr)
        exposure.maskedImage -= image
        return image


class DummySubtractSky2dConfig(Config):
    """Configuration for DummySubtractSky2dTask"""
    selectFibers = ConfigurableField(target=SelectFibersTask, doc="Select fibers")
    fitSkyModel = ConfigurableField(target=FitBlockedOversampledSplineTask, doc="1D sky subtraction")
    mask = ListField(dtype=str, default=["NO_DATA", "BAD", "SAT", "CR", "BAD_FLAT"],
                     doc="Mask pixels to ignore in extracting spectra")

    def setDefaults(self):
        super().setDefaults()
        self.selectFibers.targetType = ("SKY", "SUNSS_DIFFUSE", "SUNSS_IMAGING")


class DummySubtractSky2dTask(Task):
    """Subtract sky from 2D spectra image using 1D sky subtraction methods"""
    ConfigClass = DummySubtractSky2dConfig
    _DefaultName = "subtractSky2d"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("selectFibers")
        self.makeSubtask("fitSkyModel")

    def run(self, exposureList, pfsConfig: PfsConfig, psfList, fiberTraceList, detectorMapList, linesList):
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
        sky2d : `SkyModel`
            Dummy sky model, containing no data.
        imageList : `list` of `lsst.afw.image.MaskedImage`
            List of images of the sky model.
        """
        skyFibers = self.selectFibers.run(pfsConfig).fiberId
        imageList = []
        for exp, ft, detMap in zip(exposureList, fiberTraceList, detectorMapList):
            imageList.append(self.runSingle(exp, pfsConfig, skyFibers, ft, detMap))
        dummySky2d = SkyModel(wavelength=[], flux=[])
        return Struct(sky2d=dummySky2d, imageList=imageList)

    def runSingle(self, exposure, pfsConfig: PfsConfig, skyFibers, fiberTraces, detectorMap):
        """Run dummy 2d sky subtraction on a single image

        We don't attempt to combine the information from different detectors.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure of a spectrograph arm.
        pfsConfig : `PfsConfig`
            Top-end configuration.
        skyFibers : iterable of `int`
            Fiber identifiers to use for sky model.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Fiber traces.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y.

        Returns
        -------
        skyImage : `lsst.afw.image.Image`
            Sky flux subtracted from the exposure.
        """
        # Extract spectra
        badBitMask = exposure.mask.getPlaneBitMask(self.config.mask)
        spectra = fiberTraces.extractSpectra(exposure.maskedImage, badBitMask)
        for spectrum in spectra:
            spectrum.setWavelength(detectorMap.getWavelength(spectrum.fiberId))
        dataId = dict(visit=0, arm="x", spectrograph=0)  # We need something...

        # Measure 1D sky model
        sky1d = self.fitSkyModel.run(spectra.toPfsArm(dataId).select(pfsConfig, fiberId=skyFibers),
                                     pfsConfig.select(fiberId=skyFibers))

        # Evaluate and subtract 1D sky model for all fibers
        for spectrum in spectra:
            sky = sky1d(spectrum.wavelength, pfsConfig.select(fiberId=spectrum.fiberId))
            spectrum.flux[:] = sky.values
        skyImage = makeMaskedImage(spectra.makeImage(exposure.getBBox(), fiberTraces))
        exposure.maskedImage -= skyImage

        return skyImage
