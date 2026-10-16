from typing import Optional, Tuple, Union

import numpy as np
import scipy.signal

import lsst.afw.math as afwMath
from lsst.afw.detection import Psf
from lsst.afw.image import ExposureF, ImageD, MaskedImageF, PARENT
from lsst.afw.table import SourceCatalog, SourceTable
from lsst.geom import Box2I, Extent2I, Point2D, Point2I
from lsst.meas.base import SdssCentroidAlgorithm, SdssCentroidControl

from .AlardLupton import AlardLuptonResult
from .images import calculateCentroid
from .makeFootprint import makeFootprint

__all__ = ("calculatePeakToCentroidBias",)


def calculatePeakToCentroidBias(
    psf: Psf,
    x: np.ndarray,
    y: np.ndarray,
    kernel: Optional[Union[AlardLuptonResult, afwMath.Kernel]] = None,
    footprintHeight: int = 11,
    footprintWidth: float = 3.0,
    centroidControl: Optional[SdssCentroidControl] = None,
    allowSubPixels: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the peak-to-centroid bias of a PSF at nominated positions

    ``SdssCentroidAlgorithm`` (as used throughout this package, e.g. by
    `pfs.drp.stella.centroidLines.CentroidLinesTask`) estimates a source's
    position from a local quadratic/quartic fit to the *peak* of its (PSF-
    smoothed) image. That estimate is largely insensitive to broad,
    asymmetric structure in the PSF's wings, unlike the image's true
    flux-weighted first moment (its "centroid" in the photometric sense).
    For a realistic, non-Gaussian, spatially-varying PSF the two can differ
    by a non-negligible amount; this function quantifies that difference
    ("bias") at a set of nominated positions, for diagnostic use (e.g.
    deciding whether it needs to be corrected for in astrometric fitting).

    Parameters
    ----------
    psf : `lsst.afw.detection.Psf`
        Point-spread function to evaluate.
    x, y : `numpy.ndarray`
        Nominated positions (pixels) at which to calculate the bias.
    kernel : `pfs.drp.stella.AlardLupton.AlardLuptonResult` or \
            `lsst.afw.math.Kernel`, optional
        PSF-matching convolution kernel to apply to the realized PSF image
        before measuring the bias, for characterizing the bias as it would
        appear after convolution (e.g. in difference imaging) rather than
        for the bare PSF. Two kinds are accepted: an `AlardLuptonResult`
        (e.g. from `fitAlardLuptonKernel`), which is piecewise-constant
        over its own fitted regions and looked up by position; or a
        spatially varying `lsst.afw.math.Kernel` (e.g. `PsfMatchTask.run`'s
        ``psfMatchingKernel``), which is instead realized directly at each
        nominated position -- a simpler alternative that needs no region
        lookup. Either way, only the kernel itself is applied: for an
        `AlardLuptonResult`, ``solution.background`` (a differential
        sky/bias offset between two real images) is not meaningful for a
        synthetic PSF realization, so it is ignored. The kernel need not be
        flux-normalized: a uniform flux rescaling cannot shift a centroid or
        peak position, so it does not affect the bias.
    footprintHeight : `int`
        Half-height (pixels) of the footprint built around each position's
        peak; see `pfs.drp.stella.makeFootprint.makeFootprint`. Matches
        `pfs.drp.stella.centroidLines.CentroidLinesConfig`'s default, for
        consistency with the real measurement pipeline.
    footprintWidth : `float`
        Half-width (pixels) of the footprint; see ``footprintHeight``.
    centroidControl : `lsst.meas.base.SdssCentroidControl`, optional
        Configuration for the centroid algorithm. Defaults to
        ``SdssCentroidControl()`` with ``binmax=1``, matching
        `pfs.drp.stella.centroidLines.CentroidLinesConfig`'s default.
    allowSubPixels : `bool`
        If ``True``, the input ``x`` and ``y`` positions are allowed to
        have sub-pixel values. If ``False``, they are rounded to the nearest
        integer, to avoid following the artificial sub-pixel variation in the
        PSF's realized image (which can be magnified by the kernel convolution)
        rather than the intended bias.

    Returns
    -------
    xBias, yBias : `numpy.ndarray`
        Bias (peak-based position minus flux-weighted centroid position),
        one pair of values per nominated position, in pixels.

    Notes
    -----
    The peak-based position is measured with the real
    `lsst.meas.base.SdssCentroidAlgorithm`, via a real
    `~pfs.drp.stella.makeFootprint.makeFootprint`-built footprint (*not*
    `pfs.drp.stella.centroidImage.centroidExposure`, which builds a bare,
    zero-area ``Footprint`` that always trips ``doFootprintCheck`` and
    silently resets the result back to the peak guess).

    `makeFootprint` assumes its input image's ``XY0`` is ``(0, 0)`` -- it
    indexes rows by the *absolute* pixel coordinate, with no offset
    correction. Separately, `SdssCentroidAlgorithm.measure` queries ``psf``
    internally (for its local smoothing kernel) using whatever absolute
    position it is given, so for a spatially-varying ``psf`` the working
    image must use the same global coordinate system ``psf`` is defined
    against, or the wrong part of the model would be evaluated. Both
    constraints are satisfied at once by using a single working image with
    ``XY0=(0, 0)`` that is large enough to directly contain the nominated
    positions, rather than, say, a small image re-centered on each position
    in turn.

    A nominated position close to ``x=0`` or ``y=0`` may have its PSF
    realization extend past the working image's edge; the overlapping part
    is used consistently for both the peak and centroid measurements (the
    same truncation a real source near a detector edge would suffer), so
    the bias is still meaningful, if based on less information.
    """
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length; got {len(x)} and {len(y)}")
    if len(x) == 0:
        return np.array([]), np.array([])
    good = np.isfinite(x) & np.isfinite(y)

    if centroidControl is None:
        centroidControl = SdssCentroidControl()
        centroidControl.binmax = 1

    schema = SourceTable.makeMinimalSchema()
    centroidAlgorithm = SdssCentroidAlgorithm(centroidControl, "centroid", schema)
    schema.getAliasMap().set("slot_Centroid", "centroid")

    isAlardLupton = isinstance(kernel, AlardLuptonResult)
    isSpatialKernel = isinstance(kernel, afwMath.Kernel)
    if kernel is not None and not (isAlardLupton or isSpatialKernel):
        raise TypeError(f"kernel must be an AlardLuptonResult or lsst.afw.math.Kernel; got {type(kernel)}")

    # Validate (and look up) the kernel solution for every position up front, before allocating the
    # (potentially large) working image below: an invalid position should fail fast and cleanly,
    # rather than only after attempting to size that image from it. A spatially varying
    # lsst.afw.math.Kernel needs no such lookup: it is realized directly at each position below.
    solutions = [None] * len(x)
    if isAlardLupton:
        for index, (xPos, yPos) in enumerate(zip(x, y)):
            xPeak, yPeak = int(round(float(xPos))), int(round(float(yPos)))
            solution = kernel.getSolutionAt(xPeak, yPeak)
            if solution is None:
                raise ValueError(f"Position ({xPos}, {yPos}) is outside kernel's fitted image")
            if not solution.success:
                raise ValueError(f"Kernel fit failed in the region containing ({xPos}, {yPos})")
            solutions[index] = solution

    # Margin (beyond the PSF's own kernel footprint) that SdssCentroidAlgorithm's internal
    # smoothing needs around each position; estimated once, from the first position, on the
    # assumption that the PSF's kernel footprint size does not vary drastically across the field.
    stampBBox = psf.computeImage(Point2D(float(x[0]), float(y[0]))).getBBox()
    margin = max(stampBBox.getWidth(), stampBBox.getHeight())  # one more full kernel-width of padding

    workingBBox = Box2I(
        Point2I(0, 0),
        Extent2I(int(np.ceil(np.max(x[good]))) + margin + 1, int(np.ceil(np.max(y[good]))) + margin + 1),
    )
    working = MaskedImageF(workingBBox)
    working.variance.array[:] = 1.0  # arbitrary, finite: feeds only SdssCentroid's (unused) error estimate
    exposure = ExposureF(working)
    exposure.setPsf(psf)

    xBias = np.full(len(x), np.nan)
    yBias = np.full(len(y), np.nan)
    for index, (xPos, yPos, gg) in enumerate(zip(x, y, good)):
        if not gg:
            continue
        xPos, yPos = (float(xPos), float(yPos)) if allowSubPixels else (int(xPos), int(yPos))
        model = psf.computeImage(Point2D(xPos, yPos))

        kernelArray = None
        if isAlardLupton:
            kernelArray = solutions[index].kernel
        elif isSpatialKernel:
            kernelImage = ImageD(kernel.getDimensions())
            kernel.computeImage(kernelImage, False, xPos, yPos)
            kernelArray = kernelImage.array

        if kernelArray is not None:
            convolved = ImageD(model.getBBox())
            convolved.array[:] = scipy.signal.convolve2d(model.array, kernelArray, mode="same")
            model = convolved

        # A position close to the working image's edge may have its PSF realization extend past
        # it; crop to the overlap so both measurements below see consistent (if incomplete) data.
        modelBBox = model.getBBox()
        overlap = Box2I(modelBBox)
        overlap.clip(workingBBox)
        croppedModel = model[overlap, PARENT] if overlap != modelBBox else model

        centroid = calculateCentroid(croppedModel)

        paddedBBox = Box2I(modelBBox)
        paddedBBox.grow(margin)
        paddedBBox.clip(workingBBox)
        working.image[paddedBBox, PARENT].array[:] = 0.0
        working.image[overlap, PARENT].array[:] = croppedModel.array

        catalog = SourceCatalog(schema)
        source = catalog.addNew()
        xPeak, yPeak = int(round(xPos)), int(round(yPos))
        source.setFootprint(
            makeFootprint(working.image, Point2I(xPeak, yPeak), footprintHeight, footprintWidth)
        )
        centroidAlgorithm.measure(source, exposure)

        xBias[index] = source["centroid_x"] - centroid.x
        yBias[index] = source["centroid_y"] - centroid.y

    return xBias, yBias
