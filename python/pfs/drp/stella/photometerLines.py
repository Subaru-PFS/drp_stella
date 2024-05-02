import itertools
import numpy as np
from scipy.interpolate import interp1d

from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.pipe.base import Task, Struct

from pfs.datamodel import FiberStatus
from .arcLine import ArcLineSet
from .referenceLine import ReferenceLineSet, ReferenceLineStatus
from .fitContinuum import FitContinuumTask
from .photometry import photometer
from .utils.psf import checkPsf
from pfs.drp.stella import FiberProfileSet, FiberTraceSet
from .apertureCorrections import MeasureApertureCorrectionsTask, calculateApertureCorrection

import lsstDebug

__all__ = ("PhotometerLinesConfig", "PhotometerLinesTask")


def cartesianProduct(array1, array2):
    """Return the cartesian product of two arrays.

    Provides all combinations of the two arrays.

    Parameters
    ----------
    array1, array2 : array_like
        Arrays, which may have different lengths.

    Returns
    -------
    cartProd1, cartProd2 : array_like
        Combinations of values from ``array1`` and ``array2``.
    """
    return np.tile(array1, len(array2)), np.repeat(array2, len(array1))


class PhotometerLinesConfig(Config):
    """Configuration for CentroidLinesTask"""
    doSubtractContinuum = Field(dtype=bool, default=True, doc="Subtract continuum before centroiding lines?")
    continuum = ConfigurableField(target=FitContinuumTask, doc="Continuum subtraction")
    doForced = Field(dtype=bool, default=True, doc="Use forced positions to measure lines?")
    mask = ListField(dtype=str, default=["BAD", "SAT", "CR", "NO_DATA"], doc="Mask planes for bad pixels")
    excludeStatus = ListField(dtype=str, default=[],
                              doc="Reference line status flags indicating that line should be excluded")
    fwhm = Field(dtype=float, default=1.5, doc="FWHM of PSF (pixels)")
    doSubtractLines = Field(dtype=bool, default=False, doc="Subtract lines after measurement?")
    doApertureCorrection = Field(dtype=bool, default=True, doc="Perform aperture correction?")
    apertureCorrection = ConfigurableField(target=MeasureApertureCorrectionsTask, doc="Aperture correction")


class PhotometerLinesTask(Task):
    """Centroid lines on an arc"""
    ConfigClass = PhotometerLinesConfig
    _DefaultName = "photometerLines"

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)
        self.makeSubtask("continuum")
        self.makeSubtask("apertureCorrection")

    def run(
        self, exposure, referenceLines, detectorMap, pfsConfig, fiberTraces=None, fiberNorms=None
    ) -> Struct:
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
            Mapping between fiberId,wavelength and x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration, for specifying good fibers.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`, optional
            Position and profile of fiber traces. Required for continuum
            subtraction and/or flux normalisation.
        fiberNorms : `pfs.drp.stella.datamodel.PfsFiberNorms`, optional
            Fiber normalizations.

        Returns
        -------
        lines : `pfs.drp.stella.ArcLineSet`
            Centroided lines.
        apCorr : `pfs.drp.stella.FocalPlaneFunction`
            Aperture correction.
        """
        if self.config.doSubtractContinuum:
            if fiberTraces is None:
                raise RuntimeError("No fiberTraces provided for continuum subtraction")
            with self.continuum.subtractionContext(exposure.maskedImage, fiberTraces, detectorMap,
                                                   referenceLines):
                phot = self.photometerLines(exposure, referenceLines, detectorMap, pfsConfig, fiberTraces)
        else:
            phot = self.photometerLines(exposure, referenceLines, detectorMap, pfsConfig, fiberTraces)

        if fiberTraces is not None:
            self.addFluxNormalizations(phot.lines, fiberTraces, fiberNorms)
        else:
            self.log.warning("Not normalizing measured line fluxes")
        self.log.info("Photometered %d lines", len(phot.lines))
        return phot

    def photometerLines(self, exposure, lines, detectorMap, pfsConfig, fiberTraces=None):
        """Photometer lines on an image

        We perform a simultaneous fit of PSFs to each of the lines.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Arc exposure on which to centroid lines.
        lines : `pfs.drp.stella.ReferenceLineSet` or `ArcListSet`
            List of reference lines.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Approximate mapping between fiberId,wavelength and x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for specifying good fibers.

        Returns
        -------
        lines : `pfs.drp.stella.ArcLineSet`
            Photometered lines.
        apCorr : `pfs.drp.stella.FocalPlaneFunction`
            Aperture correction that was applied, or ``None``.
        """
        psf = checkPsf(exposure, detectorMap, self.config.fwhm)

        positions = None
        if isinstance(lines, ReferenceLineSet):
            fiberId = pfsConfig.select(fiberStatus=FiberStatus.GOOD, fiberId=detectorMap.fiberId).fiberId
            fiberId, wavelength = cartesianProduct(fiberId, lines.wavelength)
            if not self.config.doForced:
                self.log.warning("Unable to perform unforced photometry without centroided positions "
                                 "provided; performing forced photometry instead")
            xx, yy = detectorMap.findPoint(fiberId.copy(), wavelength.copy()).T  # copy required for pybind
            nan = np.full_like(wavelength, np.nan)
            flags = itertools.repeat(False)
            lookup = {rl.wavelength: rl for rl in lines}
            status = [lookup[wl].status for wl in wavelength]
            description = [lookup[wl].description for wl in wavelength]
            transition = [lookup[wl].transition for wl in wavelength]
            source = [lookup[wl].status for wl in wavelength]

            lines = ArcLineSet.fromColumns(fiberId=fiberId, wavelength=wavelength, x=xx, y=yy,
                                           xErr=nan, yErr=nan, flux=nan, fluxErr=nan, fluxNorm=nan,
                                           xx=nan, yy=nan, xy=nan,
                                           flag=flags, status=status, description=description,
                                           transition=transition,
                                           source=source)
        else:
            fiberId = lines.fiberId
            wavelength = lines.wavelength
            if not self.config.doForced:
                positions = np.array((lines.x.T, lines.y.T))

        badBitMask = exposure.mask.getPlaneBitMask(self.config.mask)
        select = (lines.status & ReferenceLineStatus.fromNames(*self.config.excludeStatus)) == 0
        catalog = photometer(exposure.maskedImage, fiberId[select], wavelength[select], psf, badBitMask,
                             positions[select] if positions is not None else None)

        assert np.all(catalog["fiberId"] == lines.fiberId[select])
        assert np.all(catalog["wavelength"] == lines.wavelength[select])
        lines.flux[select] = catalog["flux"]
        lines.fluxErr[select] = catalog["fluxErr"]
        lines.flag[~select] = True
        lines.flag[select] = catalog["flag"] & np.isfinite(catalog["flux"]) & np.isfinite(catalog["fluxErr"])

        apCorr = None
        if self.config.doApertureCorrection:
            apCorr = self.apertureCorrection.run(exposure, pfsConfig, detectorMap, lines)

        if self.config.doSubtractLines:
            self.subtractLines(exposure, lines, apCorr, pfsConfig)

        return Struct(lines=lines, apCorr=apCorr)

    def getNormalizations(self, tracesOrProfiles):
        """Get an object that can be used for normalization

        Parameters
        ----------
        tracesOrProfiles : `FiberTraceSet` or `FiberProfilesSet`
            Fiber traces or fiber profiles, which contain the normalization.

        Returns
        -------
        norms : `dict` mapping `int` to callable
            Functions for each fiberId that will return the normalization given
            the row.
        """
        def getInterpolator(array):
            """Return interpolator for an array of values"""
            return interp1d(np.arange(len(array)), array, bounds_error=False, fill_value=np.nan)

        if isinstance(tracesOrProfiles, FiberTraceSet):
            norm = {ft.fiberId: ft.trace.image.array.sum(axis=1) for ft in tracesOrProfiles}
        elif isinstance(tracesOrProfiles, FiberProfileSet):
            norm = {ff: tracesOrProfiles[ff].norm for ff in tracesOrProfiles}
        else:
            raise RuntimeError(f"Unrecognised traces/profiles object: {tracesOrProfiles}")
        return {ff: getInterpolator(np.where(np.isfinite(norm[ff]), norm[ff], 0.0)) for ff in norm}

    def addFluxNormalizations(self, lines, tracesOrProfiles, fiberNorms):
        """Add the trace normalization

        We provide the normalization, which is typically the extracted flux of
        a quartz, so the flux can be corrected to be relative to the quartz.

        Parameters
        ----------
        lines : `pfs.drp.stella.ArcLineSet`
            Measured lines. This will be modified with the normalization
            applied.
        tracesOrProfiles : `FiberTraceSet` or `FiberProfilesSet`
            Fiber traces or fiber profiles, which contain the normalization.
        fiberNorms : `pfs.drp.stella.datamodel.PfsFiberNorms`
            Fiber normalizations.
        """
        interpolators = self.getNormalizations(tracesOrProfiles)
        lines.fluxNorm[:] = [
            interpolators[ff](yy) if ff in interpolators else np.nan for ff, yy in zip(lines.fiberId, lines.y)
        ]
        if fiberNorms is not None:
            lines.fluxNorm[:] *= [fiberNorms.calculate(ff, yy) for ff, yy in zip(lines.fiberId, lines.y)]

    def subtractLines(self, exposure, lines, apCorr, pfsConfig):
        """Subtract lines from the image

        This can be used as a check of the quality of the measurement.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure containing the lines to subtract; modified.
        lines : `pfs.drp.stella.ArcLineSet`
            Measured lines.
        apCorr : `pfs.drp.stella.FocalPlaneFunction`
            Aperture corrections.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration, for specifying fiber positions.
        """
        if apCorr is not None:
            psfFlux = np.full_like(lines.flux, np.nan, dtype=float)
            for fiberId in set(lines.fiberId):
                select = lines.fiberId == fiberId
                wavelength = lines.wavelength[select]
                psfFlux[select] = calculateApertureCorrection(apCorr, fiberId, wavelength, pfsConfig,
                                                              lines.flux[select], invert=True).flux
        imageBox = exposure.getBBox()
        psf = exposure.getPsf()
        for ii, ll in enumerate(lines):
            if ll.flag:
                continue
            flux = ll.flux if apCorr is None else psfFlux[ii]
            if not np.isfinite(flux):
                continue

            psfImage = psf.computeImage(ll.fiberId, ll.wavelength)
            psfBox = psfImage.getBBox()
            psfBox.clip(imageBox)
            exposure.image[psfBox].scaledMinus(flux, psfImage[psfBox].convertF())
