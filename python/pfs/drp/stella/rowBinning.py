"""Support for extracting spectra from data binned in the parallel direction

The PFS CCDs may be binned in the parallel (row, i.e. spectral) direction.
The raw data are full-sized, with each binned row written ``W_CDRBIN`` times;
the ISR bins the calibrations to match (see `lsst.obs.pfs.rowBinning`), so the
post-ISR image is full-sized with groups of identical rows.

For extraction we collapse the image to one row per binned row, and bin the
detectorMap, fiberTraces and fiberNorms to match. The calibrations are not
regenerated: we bin them in memory.

Conventions:

- The binned image has the value of each binned pixel: the charge summed over
  the (up to) ``factor`` physical rows, with the variance of a single read.
- The binned detectorMap gives, for each binned row, the mean ``xCenter`` and
  wavelength of the physical rows it contains, so the dispersion (nm/pixel)
  is ``factor`` times larger than unbinned.
- The extracted ``flux`` is the number of electrons in the binned pixel and
  the ``norm`` is ``factor`` times the unbinned norm, so that ``flux/norm``
  and ``norm/dispersion`` (used when merging arms) match unbinned data.
"""

from typing import TYPE_CHECKING

import numpy as np

import lsst.geom
from lsst.afw.image import MaskedImageF
from lsst.obs.pfs.rowBinning import RowBinning, getRowBinning

from .FiberTrace import FiberTrace
from .FiberTraceSet import FiberTraceSet
from .SplinedDetectorMapContinued import SplinedDetectorMap

if TYPE_CHECKING:
    from pfs.datamodel.pfsFiberNorms import PfsFiberNorms
    from .arcLine import ArcLineSet
    from .DetectorMapContinued import DetectorMap
    from .SpectrumSetContinued import SpectrumSet

__all__ = (
    "RowBinning",
    "getRowBinning",
    "binDetectorMap",
    "binFiberTraces",
    "binFiberNorms",
    "unbinLines",
    "scaleSpectraNorm",
)


def binDetectorMap(
    detectorMap: "DetectorMap",
    binning: RowBinning,
    wavelengthDetectorMap: "DetectorMap | None" = None,
) -> SplinedDetectorMap:
    """Bin a detectorMap in the parallel direction

    The binned detectorMap is a `SplinedDetectorMap` with knots at each binned
    row; the values are the mean ``xCenter`` and wavelength of the unbinned
    rows that make up each binned row. The slit offsets of the input
    detectorMap are incorporated into the values, so the binned detectorMap
    has zero slit offsets.

    Parameters
    ----------
    detectorMap : `pfs.drp.stella.DetectorMap`
        Unbinned detectorMap.
    binning : `RowBinning`
        Row binning.
    wavelengthDetectorMap : `pfs.drp.stella.DetectorMap`, optional
        Unbinned detectorMap from which to take the wavelengths as a function
        of row (e.g., because the line positions in the spectral direction
        are poorly measured in binned data, so we don't trust the adjustment
        of ``detectorMap`` in that direction). If not provided, we use
        ``detectorMap``.

    Returns
    -------
    binned : `pfs.drp.stella.SplinedDetectorMap`
        Binned detectorMap.
    """
    bbox = detectorMap.getBBox()
    if bbox.getHeight() != binning.height:
        raise RuntimeError(f"DetectorMap height ({bbox.getHeight()}) doesn't match binning ({binning})")
    if wavelengthDetectorMap is None:
        wavelengthDetectorMap = detectorMap
    elif wavelengthDetectorMap.getBBox() != bbox:
        raise RuntimeError(f"Bounding boxes of detectorMaps don't match: {wavelengthDetectorMap.getBBox()} "
                           f"vs {bbox}")
    binnedBox = lsst.geom.Box2I(bbox.getMin(), lsst.geom.Extent2I(bbox.getWidth(), binning.numBinned))
    fiberId = detectorMap.fiberId
    knots = np.arange(binning.numBinned, dtype=float)

    xCenter = []
    wavelength = []
    for ff in fiberId:
        xCenter.append(binning.mean(detectorMap.getXCenter(ff)))
        wavelength.append(binning.mean(wavelengthDetectorMap.getWavelength(ff)))

    metadata = detectorMap.getMetadata().deepCopy()
    metadata.set("PFS.BINNING.ROWS", binning.factor)

    return SplinedDetectorMap(
        binnedBox,
        fiberId,
        [knots]*len(fiberId),
        xCenter,
        [knots]*len(fiberId),
        wavelength,
        np.zeros(len(fiberId), dtype=float),
        np.zeros(len(fiberId), dtype=float),
        detectorMap.getVisitInfo(),
        metadata,
    )


def binFiberTrace(fiberTrace: FiberTrace, binning: RowBinning) -> FiberTrace:
    """Bin a fiberTrace in the parallel direction

    Each binned row of the trace is the mean of the unbinned rows; this is
    the profile appropriate for a binned row if the spectrum is smooth over
    the rows binned together. The fiber trace mask is ORed.

    Parameters
    ----------
    fiberTrace : `pfs.drp.stella.FiberTrace`
        Unbinned fiber trace.
    binning : `RowBinning`
        Row binning.

    Returns
    -------
    binned : `pfs.drp.stella.FiberTrace`
        Binned fiber trace.
    """
    trace = fiberTrace.trace
    box = trace.getBBox()
    yMin = box.getMinY()
    yMax = box.getMaxY()
    if yMin < 0 or yMax >= binning.height:
        raise RuntimeError(f"FiberTrace {fiberTrace.fiberId} bbox {box} extends beyond the image")

    # Embed in a full-height array so that rows are assigned to the correct groups
    width = box.getWidth()
    image = np.zeros((binning.height, width), dtype=float)
    mask = np.zeros((binning.height, width), dtype=trace.mask.array.dtype)
    image[yMin:yMax + 1] = trace.image.array
    mask[yMin:yMax + 1] = trace.mask.array

    jMin = binning.index[yMin]
    jMax = binning.index[yMax]
    binnedBox = lsst.geom.Box2I(lsst.geom.Point2I(box.getMinX(), jMin),
                                lsst.geom.Point2I(box.getMaxX(), jMax))
    binned = MaskedImageF(binnedBox)
    binned.mask.conformMaskPlanes(trace.mask.getMaskPlaneDict())
    binned.image.array[:] = binning.mean(image)[jMin:jMax + 1]
    binned.mask.array[:] = binning.bitwiseOr(mask)[jMin:jMax + 1]
    binned.variance.array[:] = 0.0

    return FiberTrace(binned, fiberTrace.fiberId)


def binFiberTraces(fiberTraces: FiberTraceSet, binning: RowBinning) -> FiberTraceSet:
    """Bin a set of fiberTraces in the parallel direction

    Parameters
    ----------
    fiberTraces : `pfs.drp.stella.FiberTraceSet`
        Unbinned fiber traces.
    binning : `RowBinning`
        Row binning.

    Returns
    -------
    binned : `pfs.drp.stella.FiberTraceSet`
        Binned fiber traces.
    """
    binned = FiberTraceSet(len(fiberTraces), fiberTraces.metadata)
    for ft in fiberTraces:
        binned.add(binFiberTrace(ft, binning))
    return binned


def scaleSpectraNorm(spectra: "SpectrumSet", binning: RowBinning) -> None:
    """Scale the norm of extracted spectra by the number of rows binned

    The binned fiberTraces are the mean of the unbinned rows, so the norm of
    the extracted spectra is that of a single unbinned row. Multiplying by the
    number of rows in each binned row makes ``flux/norm`` the same as for
    unbinned data.

    Parameters
    ----------
    spectra : `pfs.drp.stella.SpectrumSet`
        Spectra extracted from binned data; modified in place.
    binning : `RowBinning`
        Row binning.
    """
    for spectrum in spectra:
        if len(spectrum.norm) != binning.numBinned:
            raise RuntimeError(f"Spectrum length ({len(spectrum.norm)}) doesn't match binning ({binning})")
        spectrum.norm[:] *= binning.counts


def binFiberNorms(fiberNorms: "PfsFiberNorms", binning: RowBinning) -> "PfsFiberNorms":
    """Bin fiberNorms in the parallel direction

    The values are averaged over the unbinned rows (ignoring bad values);
    combined with `scaleSpectraNorm`, the norm of the binned spectra is the
    sum of the unbinned norms.

    Parameters
    ----------
    fiberNorms : `pfs.datamodel.PfsFiberNorms`
        Unbinned fiber normalisations.
    binning : `RowBinning`
        Row binning.

    Returns
    -------
    binned : `pfs.datamodel.PfsFiberNorms`
        Binned fiber normalisations.
    """
    if fiberNorms.height != binning.height:
        raise RuntimeError(f"FiberNorms height ({fiberNorms.height}) doesn't match binning ({binning})")
    values = np.array(fiberNorms.values, dtype=float)
    good = np.isfinite(values) & (values != 0.0)
    values[~good] = 0.0
    numGood = binning.sum(good.astype(float), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        binnedValues = np.where(numGood > 0, binning.sum(values, axis=1)/numGood, np.nan)

    return type(fiberNorms)(
        identity=fiberNorms.identity,
        fiberId=fiberNorms.fiberId,
        wavelength=binning.mean(fiberNorms.wavelength, axis=1),
        values=binnedValues,
        fiberProfilesHash=fiberNorms.fiberProfilesHash,
        model=fiberNorms.model,
        metadata=dict(fiberNorms.metadata, **{"PFS.BINNING.ROWS": binning.factor}),
    )


def unbinLines(lines: "ArcLineSet", binning: RowBinning) -> "ArcLineSet":
    """Convert line measurements from binned to unbinned coordinates

    Parameters
    ----------
    lines : `pfs.drp.stella.ArcLineSet`
        Line measurements made on binned data.
    binning : `RowBinning`
        Row binning.

    Returns
    -------
    unbinned : `pfs.drp.stella.ArcLineSet`
        Copy of the line measurements, in unbinned coordinates.
    """
    unbinned = lines.copy()
    factor = binning.factor
    data = unbinned.data
    data["y"] = binning.toUnbinned(data["y"].values).astype(data["y"].dtype)
    data["yErr"] = data["yErr"]*factor
    data["yy"] = data["yy"]*factor**2
    data["xy"] = data["xy"]*factor
    return unbinned
