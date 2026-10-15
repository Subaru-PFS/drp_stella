"""Smooth 2-D (Chebyshev) surfaces over a detector, with per-group offsets"""
from dataclasses import dataclass

import numpy as np

__all__ = ("DetectorCoords", "SmoothSurface", "fitSmoothSurface", "maxGap", "supportedDegree")


class DetectorCoords:
    """Detector pixels -> the physical detector frame, allowing for two CCDs

    The b, r (and m) detectors are two 2k x 4k CCDs side by side in x with
    contiguous pixel numbering; the right-hand CCD (larger x) is displaced
    (mostly a ~70 pixel gap in x) and slightly rotated relative to the left one.
    This applies drp_stella's model of that (``DetectorModel.pixelsToDetector``,
    an affine transform; the detector distortions are not included), giving
    positions in the left CCD's pixel frame. The n detector is monolithic: the
    identity.

    Parameters
    ----------
    xCenter : `float`
        Pixels with ``x >= xCenter`` are on the right-hand CCD.
    affine : `numpy.ndarray`, shape ``(3, 2)``, optional
        ``[x, y, 1] @ affine -> (u, v)`` on the right-hand CCD; `None` for a
        monolithic detector.
    maxError : `float`
        Largest error of the affine fit to ``pixelsToDetector`` (pixels).
    """
    def __init__(self, xCenter, affine=None, maxError=0.0):
        self.xCenter = float(xCenter)
        self.affine = None if affine is None else np.asarray(affine, dtype=float)
        self.maxError = float(maxError)

    @classmethod
    def fromDetectorMap(cls, detectorMap, nGrid=9):
        """Construct from a detectorMap's `DetectorModel`

        The affine is fit to ``pixelsToDetector`` on a grid of points on the
        right CCD (it is exact; drp_stella's iterative inverse can fail at the
        CCD's edge), so it's fast and defined at every pixel. DetectorMaps with
        no `DetectorModel` (i.e. not an `OpticalModelDetectorMap`) give the
        identity.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId, wavelength to x, y.
        nGrid : `int`
            Grid points per axis for fitting the affine.

        Returns
        -------
        self : `DetectorCoords`
        """
        bbox = detectorMap.getBBox()
        xCenter = 0.5*(bbox.getMinX() + bbox.getMaxX() + 1)
        if not hasattr(detectorMap, "getDetectorModel"):
            return cls(xCenter)
        detectorModel = detectorMap.getDetectorModel()
        if not detectorModel.getIsDivided():
            return cls(xCenter)
        pg, qg = np.meshgrid(np.linspace(xCenter + 50, bbox.getMaxX(), nGrid),
                             np.linspace(bbox.getMinY(), bbox.getMaxY(), nGrid))
        pg, qg = pg.ravel(), qg.ravel()
        xy = np.array(detectorModel.pixelsToDetector(pg, qg), dtype=float).reshape(-1, 2)
        good = np.all(np.isfinite(xy), axis=1)
        design = np.column_stack([pg, qg, np.ones_like(pg)])[good]
        affine, *_ = np.linalg.lstsq(design, xy[good], rcond=None)
        maxError = float(np.max(np.abs(design @ affine - xy[good])))
        return cls(xCenter, affine, maxError)

    @property
    def isDivided(self):
        """Is the detector two CCDs?"""
        return self.affine is not None

    def __call__(self, x, y):
        """Convert detector pixels ``(x, y)`` to the frame ``(u, v)``"""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.affine is None:
            return np.broadcast_arrays(x, y)
        x, y = np.broadcast_arrays(x, y)
        right = x >= self.xCenter
        u = np.where(right, self.affine[0, 0]*x + self.affine[1, 0]*y + self.affine[2, 0], x)
        v = np.where(right, self.affine[0, 1]*x + self.affine[1, 1]*y + self.affine[2, 1], y)
        return u, v

    def __reduce__(self):
        return self.__class__, (self.xCenter, self.affine, self.maxError)

    def __repr__(self):
        if self.affine is None:
            return "DetectorCoords(monolithic)"
        return (f"DetectorCoords(right CCD (x >= {self.xCenter:g}): dx/dx {self.affine[0, 0] - 1:+.1e},"
                f" dx/dy {self.affine[1, 0]:+.2e}, dy/dx {self.affine[0, 1]:+.2e},"
                f" offset ({self.affine[2, 0]:+.2f}, {self.affine[2, 1]:+.2f}) pix;"
                f" fit error {self.maxError:.1e} pix)")


def _chebBasis(u, v, degree, domain):
    """Tensor-product Chebyshev basis, shape (N, (du+1)*(dv+1)), with ``domain`` mapped to [-1, 1]^2"""
    (u0, u1), (v0, v1) = domain
    su = (2*np.asarray(u, dtype=float) - (u0 + u1))/(u1 - u0)
    sv = (2*np.asarray(v, dtype=float) - (v0 + v1))/(v1 - v0)
    return np.polynomial.chebyshev.chebvander2d(su, sv, degree)


@dataclass
class SmoothSurface:
    """A parameter as a smooth 2-D Chebyshev series over the detector, with group offsets

    Parameters
    ----------
    name : `str`
        Parameter name.
    degree : (`int`, `int`)
        Chebyshev degree in (u, v).
    domain : ((`float`, `float`), (`float`, `float`))
        Ranges of (u, v) mapped onto [-1, 1].
    coeffs : `numpy.ndarray`
        Flattened coefficients, matching `numpy.polynomial.chebyshev.chebvander2d`.
    offsets : `dict` [`str`, `dict` [`str`, (`float`, `float`)]]
        For each grouping factor (e.g. ``element``, ``fiberId``), the constant
        offset of each of its groups (labels are `str`), with its 1-sigma error
        scaled by ``sqrt(chi2/ndof)``. For an ordinary factor the reference group
        has offset 0 and the bare surface describes it; for the trace factor
        (see `fitSmoothSurface`) every group has an offset and the surface has no
        v-independent part.
    reference : `dict` [`str`, `str`]
        Each factor's reference group (`None` for the trace factor).
    rms : `float`
        RMS residual of the points used (parameter units).
    chi2 : `float`
        chi^2 of the points used, with their (statistical) errors.
    ndof : `int`
        Degrees of freedom (points used minus coefficients).
    nUsed : `int`
        Number of points used.
    """
    name: str
    degree: tuple
    domain: tuple
    coeffs: np.ndarray
    offsets: dict
    reference: dict
    rms: float = np.nan
    chi2: float = np.nan
    ndof: int = 0
    nUsed: int = 0

    def __call__(self, u, v):
        """Evaluate the surface (no group offsets) at ``(u, v)``"""
        u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
        return (_chebBasis(u.ravel(), v.ravel(), self.degree, self.domain) @ self.coeffs).reshape(u.shape)

    def getOffset(self, factor, label):
        """Offset of group ``label`` of ``factor`` (0 if unknown)

        Parameters
        ----------
        factor : `str`
            Grouping factor, e.g. ``element`` or ``fiberId``.
        label : scalar or array-like
            Group label(s); compared as `str`.

        Returns
        -------
        offset : `float` or `numpy.ndarray`
            The offset(s).
        """
        table = self.offsets.get(factor, {})
        if np.ndim(label) == 0:
            return table.get(str(label), (0.0, 0.0))[0]
        offsets = [table.get(str(ll), (0.0, 0.0))[0] for ll in np.ravel(label)]
        return np.array(offsets).reshape(np.shape(label))

    def evaluate(self, u, v, **labels):
        """Surface plus the offsets of the given groups

        E.g. ``evaluate(u, v, fiberId=84, element="Kr")``; labels may be arrays
        matching ``u, v``.
        """
        value = self(u, v)
        for factor, label in labels.items():
            if label is not None:
                value = value + self.getOffset(factor, label)
        return value


def maxGap(values, domain):
    """Largest interval of ``domain`` with no ``values`` in it (including the intervals at the ends)

    Parameters
    ----------
    values : array-like
        Positions (non-finite values are ignored).
    domain : (`float`, `float`)
        Range of interest.

    Returns
    -------
    gap : `float`
        Length of the largest empty interval.
    """
    values = np.asarray(values, dtype=float)
    values = np.sort(values[np.isfinite(values) & (values >= domain[0]) & (values <= domain[1])])
    return float(np.max(np.diff(np.concatenate([[domain[0]], values, [domain[1]]]))))


def supportedDegree(values, domain, degree, factor=1.0):
    """The highest Chebyshev degree, up to ``degree``, that the coverage of ``domain`` supports

    A degree-``d`` polynomial can have features on scales of ``span/(d + 1)``
    (``span`` the length of ``domain``); where there are no data over a longer
    interval it is unconstrained, and it can wander there. The degree is
    therefore reduced until the largest empty interval (`maxGap`, including the
    ends of the domain) is at most ``factor*span/(d + 1)``.

    Parameters
    ----------
    values : array-like
        Positions of the data.
    domain : (`float`, `float`)
        Range over which the polynomial will be evaluated.
    degree : `int`
        Requested degree.
    factor : `float`
        Tolerance: larger values allow larger gaps.

    Returns
    -------
    degree : `int`
        The supported degree (at least 0).
    """
    span = domain[1] - domain[0]
    gap = maxGap(values, domain)
    while degree > 0 and gap > factor*span/(degree + 1):
        degree -= 1
    return degree


def fitSmoothSurface(u, v, value, error, degree, *, domain=None, name="", errFloor=0.0, nIter=3, nSigma=4.0,
                     groups=None, reference=None, traceFactor="fiberId"):
    """Weighted, clipped least-squares fit of a 2-D Chebyshev series with group offsets

    The per-point errors are statistical only and the scatter about any smooth
    model is dominated by systematics, so (a) an ``errFloor`` can be added in
    quadrature to the weights, and (b) clipping uses a robust (1.4826*MAD) scale
    of the normalised residuals rather than assuming they are unit-variance.

    Parameters
    ----------
    u, v, value, error : array-like
        Positions, values and their 1-sigma errors. Non-finite points, and those
        with ``error <= 0``, are ignored.
    degree : `int` or (`int`, `int`)
        Chebyshev degree in (u, v).
    domain : ((`float`, `float`), (`float`, `float`)), optional
        Ranges mapped onto [-1, 1]; e.g. the detector's extent, so the fit is
        stable to evaluate anywhere on it. Defaults to the range of the data.
    name : `str`
        Stored in the result.
    errFloor : `float`
        Added in quadrature to ``error`` for the weights.
    nIter : `int`
        Maximum number of clip-and-refit iterations.
    nSigma : `float`
        Clip points with ``|normalised residual| > nSigma*robust scale``.
    groups : `dict` [`str`, array-like], optional
        Grouping factors: ``{factor: label per point}``. Each group gets a
        constant offset, fit simultaneously with the surface. For an ordinary
        factor (e.g. the lamp, i.e. the exposure) one ``reference`` group has no
        offset. The ``traceFactor`` (a per-fibre offset, e.g. a wavelength
        zero-point error) can't be separated from the surface's v-independent
        terms, so for it every group gets an offset and those terms are dropped
        from the surface: each fibre's offset is then its whole along-trace
        constant.
    reference : `dict` [`str`, `str`], optional
        Reference group per ordinary factor; defaults to the most populous.
    traceFactor : `str`, optional
        Name of the per-trace factor, see ``groups``.

    Returns
    -------
    surface : `SmoothSurface`
        The fit.
    used : `numpy.ndarray` of `bool`
        Points used (finite, and not clipped).
    residual : `numpy.ndarray`
        ``value - model`` at every point, including the group offsets.
    """
    u, v, value, error = (np.asarray(a, dtype=float) for a in (u, v, value, error))
    if np.ndim(degree) == 0:
        degree = (int(degree), int(degree))
    degree = tuple(int(d) for d in degree)
    good = np.isfinite(u) & np.isfinite(v) & np.isfinite(value) & np.isfinite(error) & (error > 0)
    if domain is None:
        domain = ((np.min(u[good]), np.max(u[good])), (np.min(v[good]), np.max(v[good])))
    domain = tuple(tuple(float(d) for d in dd) for dd in domain)

    A = _chebBasis(np.where(good, u, 0.0), np.where(good, v, 0.0), degree, domain)
    keepPoly = np.ones(A.shape[1], dtype=bool)
    groups = {f: np.asarray(g).astype(str) for f, g in (groups or {}).items()}
    reference = reference or {}
    columns = []                       # (factor, label) of each offset column
    refs = {}
    for factor, labels in groups.items():
        values, counts = np.unique(labels[good], return_counts=True)
        if factor == traceFactor:
            refs[factor] = None
            keepPoly &= (np.arange(A.shape[1]) % (degree[1] + 1)) != 0      # drop the v-independent terms
            columns += [(factor, str(g)) for g in values]
        else:
            ref = reference.get(factor) or str(values[np.argmax(counts)])
            refs[factor] = ref
            columns += [(factor, str(g)) for g in values if g != ref]
    nPoly = int(keepPoly.sum())
    A = np.hstack([A[:, keepPoly]] + [(groups[f] == g).astype(float)[:, None] for f, g in columns])
    nCoeff = A.shape[1]

    weight = np.where(good, 1.0/np.sqrt(np.where(good, error, 1.0)**2 + errFloor**2), 0.0)
    used = good.copy()
    coeffs = np.full(nCoeff, np.nan)
    for _ in range(max(1, nIter) + 1):
        if used.sum() <= nCoeff:
            raise RuntimeError(f"{name}: only {used.sum()} points for {nCoeff} coefficients")
        coeffs = np.linalg.lstsq(A[used]*weight[used, None], value[used]*weight[used], rcond=None)[0]
        resid = np.where(good, value - A @ coeffs, np.nan)
        norm = resid*weight
        scale = 1.4826*np.nanmedian(np.abs(norm[used] - np.nanmedian(norm[used])))
        newUsed = good & (np.abs(norm) <= nSigma*scale) if scale > 0 else used
        if np.array_equal(newUsed, used):
            break
        used = newUsed

    residual = value - A @ coeffs
    r = residual[used]
    chi2 = float(np.sum((r/error[used])**2))
    ndof = int(used.sum() - nCoeff)
    offsets = {f: ({} if refs[f] is None else {refs[f]: (0.0, 0.0)}) for f in groups}
    if columns:
        Aw = A[used]*weight[used, None]
        try:
            err = np.sqrt(np.clip(np.diag(np.linalg.pinv(Aw.T @ Aw)), 0, np.inf))
            err *= np.sqrt(max(1.0, chi2/max(1, ndof)))
        except np.linalg.LinAlgError:
            err = np.full(nCoeff, np.nan)
        for j, (f, g) in enumerate(columns):
            offsets[f][g] = (float(coeffs[nPoly + j]), float(err[nPoly + j]))
    full = np.zeros(keepPoly.size)
    full[keepPoly] = coeffs[:nPoly]
    surface = SmoothSurface(name=name, degree=degree, domain=domain, coeffs=full, offsets=offsets,
                            reference=refs, rms=float(np.sqrt(np.mean(r**2))), chi2=chi2, ndof=ndof,
                            nUsed=int(used.sum()))
    return surface, used, residual
