import itertools
import numpy as np
from scipy.interpolate import interp1d

from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.pipe.base import Task

from pfs.datamodel import FiberStatus
from .arcLine import ArcLine, ArcLineSet
from .fitContinuum import FitContinuumTask
from .photometry import photometer
from .utils.psf import checkPsf
from pfs.drp.stella import FiberProfileSet, FiberTraceSet

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
    fwhm = Field(dtype=float, default=1.5, doc="FWHM of PSF (pixels)")
    doSubtractLines = Field(dtype=bool, default=False, doc="Subtract lines after measurement?")


class PhotometerLinesTask(Task):
    """Centroid lines on an arc"""
    ConfigClass = PhotometerLinesConfig
    _DefaultName = "photometerLines"

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
            Mapping between fiberId,wavelength and x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration, for specifying good fibers. If not provided,
            will use all fibers in the detectorMap.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`, optional
            Position and profile of fiber traces. Required for continuum
            subtraction and/or flux normalisation.

        Returns
        -------
        lines : `pfs.drp.stella.ArcLineSet`
            Centroided lines.
        """
        if self.config.doSubtractContinuum:
            if fiberTraces is None:
                raise RuntimeError("No fiberTraces provided for continuum subtraction")
            with self.continuum.subtractionContext(exposure.maskedImage, fiberTraces, detectorMap,
                                                   referenceLines):
                lines = self.photometerLines(exposure, referenceLines, detectorMap, pfsConfig, fiberTraces)
                if self.config.doSubtractLines:
                    self.subtractLines(exposure, lines, detectorMap)
        else:
            lines = self.photometerLines(exposure, referenceLines, detectorMap, pfsConfig, fiberTraces)
            if self.config.doSubtractLines:
                self.subtractLines(exposure, lines, detectorMap)

        if fiberTraces is not None:
            self.correctFluxNormalizations(lines, fiberTraces)
        else:
            self.log.warn("Not normalizing measured line fluxes")
        self.log.info("Photometered %d lines", len(lines))
        return lines

    def photometerLines(self, exposure, lines, detectorMap, pfsConfig=None, fiberTraces=None):
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
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration, for specifying good fibers. If not provided,
            will use all fibers in the detectorMap.

        Returns
        -------
        lines : `pfs.drp.stella.ArcLineSet`
            Photometered lines.
        """
        psf = checkPsf(exposure, detectorMap, self.config.fwhm)

        positions = None
        if isinstance(lines, ReferenceLineSet):
            fiberId = detectorMap.fiberId
            if pfsConfig is not None:
                indices = pfsConfig.selectByFiberStatus(FiberStatus.GOOD, fiberId)
                fiberId = fiberId[indices]
            fiberId, wavelength = cartesianProduct(fiberId, lines.wavelength)
            if self.config.doForced:
                self.log.warn("Unable to perform unforced photometry without centroided positions provided; "
                              "performing forced photometry instead")
            xx, yy = detectorMap.findPoint(fiberId.copy(), wavelength.copy()).T  # copy required for pybind
            nan = itertools.repeat(np.nan)
            flags = itertools.repeat(False)
            lookup = {rl.wavelength: rl for rl in lines}
            status = [lookup[wl].status for wl in wavelength]
            description = [lookup[wl].description for wl in wavelength]
            lines = ArcLineSet([ArcLine(*args) for args in
                                zip(fiberId, wavelength, xx, yy, nan, nan, nan, nan, flags,
                                    status, description)])
        else:
            fiberId = lines.fiberId
            wavelength = lines.wavelength
            if not self.config.doForced:
                positions = np.array((lines.x.T, lines.y.T))

        badBitMask = exposure.mask.getPlaneBitMask(self.config.mask)
        catalog = photometer(exposure.maskedImage, fiberId, wavelength, psf, badBitMask,
                             positions if positions is not None else None)

        cat = iter(catalog)
        for ii, rl in enumerate(lines):
            if select[ii]:
                row = next(cat)
                assert row["fiberId"] == rl.fiberId and row["wavelength"] == rl.wavelength
                rl.intensity = row["flux"]
                rl.intensityErr = row["fluxErr"]
                rl.flag |= row["flag"]
            else:
                rl.flag = True

        return lines

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

        from scipy.signal import medfilt

        if isinstance(tracesOrProfiles, FiberTraceSet):
            norm = {ft.fiberId: ft.trace.image.array.sum(axis=1) for ft in tracesOrProfiles}
        elif isinstance(tracesOrProfiles, FiberProfileSet):
            norm = {ff: tracesOrProfiles[ff].norm for ff in tracesOrProfiles}
        else:
            raise RuntimeError(f"Unrecognised traces/profiles object: {tracesOrProfiles}")
        return {ff: getInterpolator(medfilt(norm[ff], 9)) for ff in norm}

    def correctFluxNormalizations(self, lines, tracesOrProfiles):
        """Correct the raw flux measurements for the trace normalization

        We divide by the normalization, which is typically the extracted flux of
        a quartz, so the flux is relative to the quartz.

        Parameters
        ----------
        lines : `pfs.drp.stella.ArcLineSet`
            Measured lines. This will be modified with the normalization
            applied.
        tracesOrProfiles : `FiberTraceSet` or `FiberProfilesSet`
            Fiber traces or fiber profiles, which contain the normalization.
        """
        interpolators = self.getNormalizations(tracesOrProfiles)
        for ll in lines:
            norm = interpolators[ll.fiberId](ll.y)
            with np.errstate(divide="ignore", invalid="ignore"):
                ll.intensity /= norm
                ll.intensityErr /= norm

    def subtractLines(self, exposure, lines, detectorMap):
        """Subtract lines from the image

        This can be used as a check of the quality of the measurement.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure containing the lines to subtract; modified.
        lines : `pfs.drp.stella.ArcLineSet`
            Measured lines.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        """
        psf = exposure.getPsf()
        for ll in lines:
            if ll.flag:
                continue
            psfImage = psf.computeImage(ll.fiberId, ll.wavelength)
            exposure.image[psfImage.getBBox()].scaledMinus(ll.intensity, psfImage.convertF())
