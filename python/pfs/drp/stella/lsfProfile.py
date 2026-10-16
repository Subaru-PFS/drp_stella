"""Disk (x) Edgeworth line-spread function profiles

The LSF of an extracted fibre spectrum is modelled as the image of the fibre
face -- a uniform disk of radius ``radius`` (pixels) -- convolved with a rms
kernel (optics, charge diffusion) of rms ``rms`` and integrated over each
pixel. The blur is a Gaussian modulated by an Edgeworth expansion in its
standardised cumulants ``[S, K, c5, c6]`` (skewness, excess kurtosis, ...),
truncated consistently by order::

    g(z) = phi(z) [ 1
                  + (S/6) He3                                        # order 1 (n=3)
                  + (K/24) He4 + (S**2/72) He6                       # order 2 (n=4)
                  + (c5/120) He5 + (S K/144) He7 + (S**3/1296) He9   # order 3 (n=5)
                  + (c6/720) He6 + (K**2/1152 + S c5/720) He8
                      + (S**2 K/1728) He10 + (S**4/31104) He12       # order 4 (n=6)
                  ]

with ``He`` the probabilists' Hermite polynomials. The Edgeworth terms have
zero mean, so the profile's first moment is its centre ``mu``.

The profile may optionally be convolved with a Lorentzian of FWHM ``lorentz``
(pixels): an extended, symmetric wing.
"""
import numpy as np
from numpy.polynomial.hermite_e import hermeval
import scipy.special

__all__ = ("MAX_BLUR_ORDER", "edgeworthCdf", "diskProfile", "diskSigma")

MAX_BLUR_ORDER = 6   # standardised cumulants lambda_3 .. lambda_6 may be used

_LORENTZ_STEP = 0.1                  # pixels: grid for the Lorentzian convolution in `diskProfile`
_LORENTZ_HALF = 25.0                 # pixels: Lorentzian wings beyond this are dropped

_GC_NODES = 48                                                        # Gauss-Chebyshev (2nd kind) quadrature
_gcK = np.arange(1, _GC_NODES + 1)
_GC_U = np.cos(_gcK*np.pi/(_GC_NODES + 1))                            # nodes on (-1, 1)
_GC_W = np.sin(_gcK*np.pi/(_GC_NODES + 1))**2*(2.0/(_GC_NODES + 1))   # sqrt(1-u^2)*(2/pi) weights; sum 1


def _edgeworthCoeffs(cumulants):
    """Coefficients ``c[n]`` of ``He_n`` in the Edgeworth bracket ``1 + sum_n c[n] He_n``

    Parameters
    ----------
    cumulants : array-like of `float`
        Standardised cumulants ``[S, K, c5, c6]`` (any leading subset).

    Returns
    -------
    coeffs : `numpy.ndarray`
        Coefficients of ``He_0 .. He_12`` (``c[0] = 1``).
    """
    c = np.zeros(13)
    c[0] = 1.0
    n = len(cumulants)
    if n >= 1:
        S = cumulants[0]
        c[3] += S/6
    if n >= 2:
        K = cumulants[1]
        c[4] += K/24
        c[6] += S**2/72
    if n >= 3:
        c5 = cumulants[2]
        c[5] += c5/120
        c[7] += S*K/144
        c[9] += S**3/1296
    if n >= 4:
        c6 = cumulants[3]
        c[6] += c6/720
        c[8] += K**2/1152 + S*c5/720
        c[10] += S**2*K/1728
        c[12] += S**4/31104
    if n >= 5:
        raise NotImplementedError(f"Edgeworth order {n + 2} not implemented (max {MAX_BLUR_ORDER})")
    return c


def edgeworthCdf(z, cumulants=None):
    """CDF of the unit Edgeworth density with standardised ``cumulants``

    ``F(z) = Phi(z) - phi(z) sum_{n>=1} c[n] He_{n-1}(z)``, as
    ``d/dz [-phi He_{n-1}] = phi He_n``.

    Parameters
    ----------
    z : array-like
        Standardised positions.
    cumulants : array-like of `float`, optional
        Standardised cumulants ``[S, K, ...]``; `None` or empty for a Gaussian.

    Returns
    -------
    cdf : `numpy.ndarray`
        The CDF at ``z``.
    """
    cdf = scipy.special.ndtr(z)
    if cumulants is None or len(cumulants) == 0:
        return cdf
    c = _edgeworthCoeffs(cumulants)
    phi = np.exp(-0.5*np.square(z))/np.sqrt(2*np.pi)
    return cdf - phi*hermeval(z, c[1:])


def _diskWeights(radius, rms, extraction):
    """Quadrature weights over the disk, along the dispersion, for `diskProfile`

    The disk is sampled at ``t = radius*_GC_U`` along the dispersion direction;
    ``_GC_W`` already contains the semicircle ``sqrt(1 - u**2)``, so for a boxcar
    extraction -- a plain projection of the 2-D image across the trace -- the
    weights are just ``_GC_W``.

    For an optimal extraction each row is the profile-weighted sum
    ``sum_x P(x) I(x, y)``, with ``P`` the fibre's spatial profile (the disk
    projected along the trace, ``semicircle (x) G_rms``) and ``I`` the 2-D image
    (``disk (x) G_rms``, isotropic). Moving the Gaussians onto ``P``, the
    extracted profile is ``G_rms (x)_y h``, with

        h(y') = integral_{-a}^{a} Q(x') dx',   a = sqrt(R**2 - y'**2),
        Q = semicircle (x) G_{sqrt(2) rms},

    i.e. the disk's chord at ``y'`` weighted by the (blurred) spatial profile
    rather than by 1. ``h = sqrt(1 - u**2)*H(u)`` with ``H`` smooth, so the same
    Gauss-Chebyshev rule applies, with weights ``_GC_W*H``.

    Parameters
    ----------
    radius : `float`
        Disk radius (pixels).
    rms : `float`
        Blur rms (pixels).
    extraction : `str`
        ``"optimal"`` or ``"boxcar"``.

    Returns
    -------
    weights : `numpy.ndarray`
        Weights for the nodes ``radius*_GC_U``, summing to 1.
    """
    if extraction == "boxcar":
        return _GC_W
    if extraction != "optimal":
        raise ValueError(f"extraction must be 'optimal' or 'boxcar', not {extraction!r}")
    chord = radius*np.sqrt(1.0 - _GC_U**2)                          # half-chord a at each y' node
    s2 = np.sqrt(2.0)*rms
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inside = (scipy.special.ndtr((chord[:, None] - radius*_GC_U[None, :])/s2)
                  - scipy.special.ndtr((-chord[:, None] - radius*_GC_U[None, :])/s2)) @ _GC_W
        weight = _GC_W*inside/np.sqrt(1.0 - _GC_U**2)
        return weight/np.sum(weight)        # NaN if an optimiser probes extreme radius/rms: step rejected


def diskProfile(pix, mu, radius, rms, cumulants=None, extraction="optimal", lorentz=0.0):
    """Unit-area, pixel-integrated, extracted disk (x) Edgeworth profile

    The fibre image is a uniform disk of radius ``radius`` (pixels) convolved
    with an isotropic blur of rms ``rms`` (an Edgeworth expansion about a
    Gaussian, with standardised ``cumulants``), extracted (``"optimal"``:
    weighted by the fibre's spatial profile; ``"boxcar"``: summed across the
    trace; see `_diskWeights`) and integrated over each pixel: the flux in
    pixel ``p`` is ``F(p + 1/2) - F(p - 1/2)``, with ``F`` the CDF, a weighted
    sum of Edgeworth CDFs over Gauss-Chebyshev nodes on the disk. The
    optimal-extraction weights assume a Gaussian rms.

    If ``lorentz > 0`` the profile is further convolved with a Lorentzian of
    that FWHM (pixels), numerically, on a ``_LORENTZ_STEP``-pixel grid, with
    the Lorentzian's cell masses exact and its wings beyond ``_LORENTZ_HALF``
    pixels dropped (a fraction ~``lorentz/(pi*_LORENTZ_HALF)`` of the light).
    Its second moment is infinite, but it is symmetric, so the first moment is
    still ``mu``.

    Parameters
    ----------
    pix : array-like
        Contiguous, increasing pixel positions (centres).
    mu : `float`
        Centre (= first moment) of the profile (pixels).
    radius : `float`
        Disk radius (pixels).
    rms : `float`
        Blur rms (pixels).
    cumulants : array-like of `float`, optional
        The blur's standardised cumulants ``[S, K, ...]``.
    extraction : `str`
        ``"optimal"`` or ``"boxcar"``.
    lorentz : `float`
        FWHM (pixels) of a Lorentzian to convolve with; 0 for none.

    Returns
    -------
    profile : `numpy.ndarray`
        Flux in each pixel; sums to 1 over an infinite range of pixels.
    """
    pix = np.asarray(pix, dtype=float)
    if pix.size == 0:
        return np.zeros(0)
    edges = np.append(pix - 0.5, pix[-1] + 0.5) - mu
    weight = _diskWeights(radius, rms, extraction)
    if lorentz > 0:                    # the CDF on a fine grid: the pixels +/- the Lorentzian's extent
        nStep = int(round(1/_LORENTZ_STEP))
        nHalf = int(round(_LORENTZ_HALF/_LORENTZ_STEP))
        pixEdges = edges
        nFine = (len(pixEdges) - 1)*nStep + 2*nHalf + 1
        edges = edges[0] - _LORENTZ_HALF + _LORENTZ_STEP*np.arange(nFine)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = (edges[:, None] - radius*_GC_U[None, :])/rms
        cdf = edgeworthCdf(z, cumulants) @ weight
    if lorentz > 0:
        cell = np.diff(cdf)                                          # mass in each fine cell
        offset = _LORENTZ_STEP*np.arange(-nHalf, nHalf + 1)
        cellEdges = np.append(offset - _LORENTZ_STEP/2, offset[-1] + _LORENTZ_STEP/2)
        kernel = np.diff(np.arctan(cellEdges/(0.5*lorentz)))/np.pi
        cumulative = np.concatenate([[0.0], np.cumsum(np.convolve(cell, kernel, mode="same"))])
        cdf = cumulative[nHalf + nStep*np.arange(len(pixEdges))]  # at the pixel edges
    return np.diff(cdf)


def diskSigma(radius, rms):
    """RMS width of a disk (x) blur profile, before pixel integration

    The disk projected along the dispersion is a semicircle, of variance
    ``radius**2/4``.

    Parameters
    ----------
    radius, rms : array-like
        Disk radius and blur kernel rms (pixels).

    Returns
    -------
    sigma : `numpy.ndarray`
        RMS width (pixels).
    """
    return np.sqrt(0.25*np.square(radius) + np.square(rms))
