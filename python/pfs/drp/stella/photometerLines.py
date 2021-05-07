import numpy as np

from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.pipe.base import Task

from pfs.datamodel import FiberStatus
from .arcLine import ArcLine, ArcLineSet
from .fitContinuum import FitContinuumTask
from .photometry import photometer
from .utils.psf import checkPsf

import lsstDebug

__all__ = ("PhotometerLinesConfig", "PhotometerLinesTask")


class PhotometerLinesConfig(Config):
    """Configuration for CentroidLinesTask"""
    doSubtractContinuum = Field(dtype=bool, default=True, doc="Subtract continuum before centroiding lines?")
    continuum = ConfigurableField(target=FitContinuumTask, doc="Continuum subtraction")
    mask = ListField(dtype=str, default=["BAD", "SAT", "CR", "NO_DATA"], doc="Mask planes for bad pixels")
    fwhm = Field(dtype=float, default=1.5, doc="FWHM of PSF (pixels)")


class PhotometerLinesTask(Task):
    """Centroid lines on an arc"""
    ConfigClass = PhotometerLinesConfig

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)
        self.makeSubtask("continuum")

    def run(self, exposure, referenceLines, detectorMap, pfsConfig=None, fiberTraces=None):
        """Photometer lines on an arc

        We perform a simultaneous fit of PSFs to each of the lines.

        This method optionally performs continuum subtraction before handing
        off to the ``photometerLines`` method to do the actual photometry.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Arc exposure on which to centroid lines.
        referenceLines : `pfs.drp.stella.ReferenceLineSet`
            List of reference lines.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Approximate mapping between fiberId,wavelength and x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration, for specifying good fibers. If not provided,
            will use all fibers in the detectorMap.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`, optional
            Position and profile of fiber traces. Required only for continuum
            subtraction.

        Returns
        -------
        lines : `pfs.drp.stella.ArcLineSet`
            Centroided lines.
        """
        if self.config.doSubtractContinuum:
            if fiberTraces is None:
                raise RuntimeError("No fiberTraces provided for continuum subtraction")
            with self.continuum.subtractionContext(exposure.maskedImage, fiberTraces, referenceLines):
                return self.photometerLines(exposure, referenceLines, detectorMap, pfsConfig)
        return self.photometerLines(exposure, referenceLines, detectorMap, pfsConfig)

    def photometerLines(self, exposure, referenceLines, detectorMap, pfsConfig=None):
        """Photometer lines on an image

        We perform a simultaneous fit of PSFs to each of the lines.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Arc exposure on which to centroid lines.
        referenceLines : `pfs.drp.stella.ReferenceLineSet`
            List of reference lines.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Approximate mapping between fiberId,wavelength and x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration, for specifying good fibers. If not provided,
            will use all fibers in the detectorMap.

        Returns
        -------
        lines : `pfs.drp.stella.ArcLineSet`
            Photometered lines.
        """
        psf = checkPsf(exposure, detectorMap, self.config.fwhm)
        fiberId = detectorMap.fiberId
        if pfsConfig is not None:
            indices = pfsConfig.selectByFiberStatus(FiberStatus.GOOD, fiberId)
            fiberId = fiberId[indices]
        badBitMask = exposure.mask.getPlaneBitMask(self.config.mask)
        catalog = photometer(exposure.maskedImage, fiberId, referenceLines.wavelength, psf, badBitMask)
        fiberId = catalog["fiberId"]
        wavelength = catalog["wavelength"]
        flag = catalog["flag"]
        flux = catalog["flux"]
        fluxErr = catalog["fluxErr"]
        lookup = dict(zip(referenceLines.wavelength, referenceLines))
        points = detectorMap.findPoint(fiberId.copy(), wavelength.copy())  # copy required for pybind
        return ArcLineSet([ArcLine(ff, wl, xx, yy, np.nan, np.nan, fx, fxErr, bad,
                                   lookup[wl].status, lookup[wl].description) for
                           ff, wl, xx, yy, fx, fxErr, bad in
                           zip(fiberId, wavelength, points[:, 0], points[:, 1], flux, fluxErr, flag)])
