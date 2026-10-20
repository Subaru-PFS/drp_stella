from typing import Optional, Tuple, Union

import numpy as np
import scipy.signal

import lsst.afw.math as afwMath
from lsst.afw.detection import GaussianPsf, Psf
from lsst.afw.image import ExposureF, ImageD, MaskedImageF, PARENT
from lsst.afw.table import SourceCatalog, SourceTable
from lsst.geom import Box2I, Extent2I, Point2D, Point2I
from lsst.meas.base import SdssCentroidAlgorithm, SdssCentroidControl

from .AlardLupton import AlardLuptonResult
from .images import calculateCentroid
from .makeFootprint import makeFootprint
from .utils.psf import fwhmToSigma

__all__ = ("calculatePeakToCentroidBias",)


def calculatePeakToCentroidBias(
    psf: Psf,
    x: np.ndarray,
    y: np.ndarray,
    kernel: Optional[Union[AlardLuptonResult, afwMath.Kernel]] = None,
    footprintHeight: int = 11,
    footprintWidth: float = 3.0,
    centroidControl: Optional[SdssCentroidControl] = None,
    allowSubPixels: bool = True,
    centroidPsf: Optional[Psf] = None,
    peakSearch: int = 2,
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

        The kernel must map ``psf`` onto the PSF of interest, and each kind
        is applied with its own convention: an `AlardLuptonResult` by true
        convolution (as used by `fitAlardLuptonKernel`), and an
        `lsst.afw.math.Kernel` as `lsst.afw.math.convolve` applies it
        (which is a correlation: a kernel with a first moment of ``+dx``
        shifts the image by ``-dx``). So, for a `PsfMatchTask` kernel, the
        exposure whose PSF is ``psf`` must have been the ``source`` and the
        exposure of interest the ``target``.
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
        If ``True``, the PSF is realized at the input ``x`` and ``y``
        positions, including their sub-pixel phase. ``SdssCentroid``'s
        peak estimate depends on the sub-pixel phase, so this is required
        to reproduce the measurement of real lines. If ``False``, the
        positions are rounded to the nearest integer.
    centroidPsf : `lsst.afw.detection.Psf`, optional
        PSF to attach to the working exposure, which ``SdssCentroid`` uses
        as its smoothing (matched-filter) kernel. This should be the PSF
        used when measuring the real lines, which is not necessarily
        ``psf``: `pfs.drp.stella.centroidLines.CentroidLinesTask` measures
        exposures without a PSF, and so attaches a Gaussian PSF with a FWHM
        of `CentroidLinesConfig.fwhm`. Defaults to that Gaussian (FWHM of
        1.5 pixels).
    peakSearch : `int`
        Half-size (pixels) of the box, centered on each nominated position,
        within which to find the peak pixel of the realized (and possibly
        convolved) PSF, used as ``SdssCentroid``'s starting point. The
        kernel can shift the realized PSF so that its peak pixel is no
        longer the one nearest the nominated position.

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
    if isSpatialKernel:
        margin += max(kernel.getWidth(), kernel.getHeight()) // 2
    elif isAlardLupton:
        margin += max(max(solution.kernel.shape) for solution in solutions) // 2

    if centroidPsf is None:
        # Matches CentroidLinesTask, via pfs.drp.stella.utils.psf.checkPsf
        sigma = fwhmToSigma(1.5)
        size = 2 * int(4.0 * sigma + 0.5) + 1
        centroidPsf = GaussianPsf(size, size, sigma)

    workingBBox = Box2I(
        Point2I(0, 0),
        Extent2I(int(np.ceil(np.max(x[good]))) + margin + 1, int(np.ceil(np.max(y[good]))) + margin + 1),
    )
    working = MaskedImageF(workingBBox)
    working.variance.array[:] = 1.0  # arbitrary, finite: feeds only SdssCentroid's (unused) error estimate
    exposure = ExposureF(working)
    exposure.setPsf(centroidPsf)

    xBias = np.full(len(x), np.nan)
    yBias = np.full(len(y), np.nan)
    for index, (xPos, yPos, gg) in enumerate(zip(x, y, good)):
        if not gg:
            continue
        xPos, yPos = (float(xPos), float(yPos)) if allowSubPixels else (round(xPos), round(yPos))
        model = psf.computeImage(Point2D(xPos, yPos))

        if isAlardLupton or isSpatialKernel:
            # Use the "full" mode, so the convolution doesn't truncate the model, which would bias its
            # centroid (the kernel's first moment is calculated over the full kernel).
            if isAlardLupton:
                kernelArray = solutions[index].kernel
                convolve = scipy.signal.convolve2d
            else:
                kernelImage = ImageD(kernel.getDimensions())
                kernel.computeImage(kernelImage, False, xPos, yPos)
                kernelArray = kernelImage.array
                convolve = scipy.signal.correlate2d  # lsst.afw.math.convolve is actually a correlation
            if any(size % 2 == 0 for size in kernelArray.shape):
                raise ValueError(f"Kernel dimensions must be odd; got {kernelArray.shape}")
            convolvedBBox = model.getBBox()
            convolvedBBox.grow(Extent2I(kernelArray.shape[1] // 2, kernelArray.shape[0] // 2))
            convolved = ImageD(convolvedBBox)
            convolved.array[:] = convolve(model.array, kernelArray, mode="full")
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

        # Start SdssCentroid from the peak pixel (as CentroidLinesTask does, via findPeak), which the
        # kernel may have shifted away from the pixel nearest the nominated position.
        searchBBox = Box2I(
            Point2I(int(round(xPos)) - peakSearch, int(round(yPos)) - peakSearch),
            Extent2I(2 * peakSearch + 1, 2 * peakSearch + 1),
        )
        searchBBox.clip(workingBBox)
        searchArray = working.image[searchBBox, PARENT].array
        yIndex, xIndex = np.unravel_index(np.argmax(searchArray), searchArray.shape)
        xPeak, yPeak = searchBBox.getMinX() + int(xIndex), searchBBox.getMinY() + int(yIndex)

        catalog = SourceCatalog(schema)
        source = catalog.addNew()
        source.setFootprint(
            makeFootprint(working.image, Point2I(xPeak, yPeak), footprintHeight, footprintWidth)
        )
        centroidAlgorithm.measure(source, exposure)

        xBias[index] = source["centroid_x"] - centroid.x
        yBias[index] = source["centroid_y"] - centroid.y

    return xBias, yBias
