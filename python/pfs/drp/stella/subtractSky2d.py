import numpy as np
from lsst.pex.config import Config, ConfigurableField
from lsst.pipe.base import Task

from .extractSpectraTask import ExtractSpectraTask
from .subtractSky1d import SubtractSky1dTask
from . import SpectrumSet


class SubtractSky2dConfig(Config):
    """Configuration for SubtractSky2dTask"""
    extractSpectra = ConfigurableField(target=ExtractSpectraTask, doc="Extract spectra from image")
    subtractSky1d = ConfigurableField(target=SubtractSky1dTask, doc="Subtract 1D sky")


class SubtractSky2dTask(Task):
    """Subtract sky from 2D spectra image"""
    ConfigClass = SubtractSky2dConfig
    _DefaultName = "subtractSky2d"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("extractSpectra")
        self.makeSubtask("subtractSky1d")

    def run(self, exposureList, pfsConfig, psfList, fiberTraceList, detectorMapList):
        """Measure and subtract sky from 2D spectra image

        This is a placeholder implementation that extracts the spectra,
        measures the average 1D sky spectrum, and then subtracts it in 2D.

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
        sky2d = self.measureSky(exposureList, pfsConfig, psfList, fiberTraceList, detectorMapList)
        for exposure, psf, fiberTrace, detectorMap in zip(exposureList, psfList,
                                                          fiberTraceList, detectorMapList):
            self.subtractSky(exposure, psf, fiberTrace, detectorMap, pfsConfig, sky2d)
        return sky2d

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
        spectraList = [self.extractSpectra.run(exposure.maskedImage, fiberTrace, detectorMap).spectra for
                       exposure, fiberTrace, detectorMap in
                       zip(exposureList, fiberTraceList, detectorMapList)]

        spectraList = [ss.toPfsArm({}) for ss in spectraList]
        resampledList = self.subtractSky1d.resampleSpectra(spectraList)
        return self.subtractSky1d.measureSky(resampledList, pfsConfig, [None]*len(spectraList))

    def subtractSky(self, exposure, psf, fiberTrace, detectorMap, pfsConfig, sky2d):
        """Subtract the 2D sky model from the images

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image from which to subtract sky.
        psf : PSF (type TBD)
            Point-spread function.
        fiberTrace : `pfs.drp.stella.FiberTraceSet`
            Fiber trace.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiber,wavelength to x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for getting location of fibers.
        sky2d : pfs.drp.stella.fitFocalPlane.FocalPlaneFunction`
            2D sky subtraction solution.
        """
        spectra = SpectrumSet(len(fiberTrace), exposure.getHeight())
        centers = pfsConfig.extractCenters([ft.fiberId for ft in fiberTrace])
        for spectrum, ft in zip(spectra, fiberTrace):
            spectrum.fiberId = ft.fiberId
            spectrum.setWavelength(detectorMap.getWavelength(ft.fiberId))
        fluxes = sky2d(spectra.getAllWavelengths(), centers)
        for ss, flux in zip(spectra, fluxes):
            ss.spectrum = flux.astype(np.float32)

        skyImage = spectra.makeImage(exposure.getBBox(), fiberTrace)
        exposure.maskedImage -= skyImage
