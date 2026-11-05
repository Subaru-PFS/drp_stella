"""Models of the line-spread function over a spectrograph's fibres

An `LsfModel` returns, for a fibre and a (floating-point) position along its
spectrum, a short `LsfKernel`: the LSF's flux in each pixel of a slice of the
spectrum's row, with methods to extract the matching slice of e.g. ``flux``.

`ArmLsfModel` is the model for one detector (arm, spectrograph), in pixels:
an extracted, pixel-integrated image of the fibre face (a disk) blurred by an
Edgeworth-expanded Gaussian, whose parameters vary smoothly over the detector.
It is fit from arc lines by `pfs.drp.stella.pipelines.FitLsfTask`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import scipy.interpolate
import scipy.ndimage

from .lsfProfile import diskProfile, diskSigma
from .smoothSurface import DetectorCoords, SmoothSurface  # noqa: F401 (docs)

__all__ = ("LsfKernel", "LsfKernelArray", "LsfModel", "ArmLsfModel", "FiberSpline", "gaussianEquivalentSigma")


@dataclass
class LsfKernel:
    """The LSF of one fibre at one position: a short slice of a spectrum's row

    Parameters
    ----------
    fiberId : `int`
        Fibre identifier.
    center : `float`
        Position of the LSF's first moment, in the model's native coordinate
        (pixels for an arm; the row index of the spectrum).
    start : `int`
        Index of ``values[0]`` in the spectrum's row.
    values : `numpy.ndarray`
        LSF flux in each pixel, times the window (normalised to sum 1 unless
        the kernel was computed with ``normalize=False``). The window is the
        fraction of each pixel within ``halfWidth`` of ``center``: 1 except for
        the end pixels, so that it is symmetric about ``center`` and the
        truncation of the LSF's wings doesn't shift the kernel's first moment
        (to first order).
    wavelength : `numpy.ndarray`, optional
        Wavelength (nm) of each pixel, if known.
    norm : `float`
        Fraction of the LSF's flux within the (windowed) kernel before
        normalisation (< 1 from the truncated wings, and if clipped at the ends
        of the row).
    """
    fiberId: int
    center: float
    start: int
    values: np.ndarray
    wavelength: np.ndarray = None
    norm: float = 1.0

    @property
    def stop(self):
        """Index one beyond the kernel's last pixel in the spectrum's row"""
        return self.start + len(self.values)

    @property
    def slice(self):
        """The `slice` of the spectrum's row covered by the kernel"""
        return slice(self.start, self.stop)

    @property
    def pixels(self):
        """Indices of the kernel's pixels in the spectrum's row"""
        return np.arange(self.start, self.stop)

    @property
    def offsets(self):
        """Pixel positions relative to ``center``"""
        return self.pixels - self.center

    def __len__(self):
        return len(self.values)

    def extract(self, array):
        """The slice of ``array`` matching the kernel

        Parameters
        ----------
        array : `numpy.ndarray`
            A spectrum's row (or an array whose last axis is the row).

        Returns
        -------
        sliced : `numpy.ndarray`
            ``array[..., self.slice]``.
        """
        return np.asarray(array)[..., self.slice]

    def extractFrom(self, spectra, name="flux"):
        """The slice of this fibre's row of an attribute of a set of spectra

        Parameters
        ----------
        spectra : `pfs.datamodel.PfsFiberArraySet`
            Spectra, e.g. a `PfsArm`.
        name : `str`
            Attribute, e.g. ``flux``, ``mask``, ``variance`` or ``wavelength``.

        Returns
        -------
        sliced : `numpy.ndarray`
            ``getattr(spectra, name)[index, self.slice]``, with ``index`` the
            row for ``fiberId``.
        """
        index = np.flatnonzero(np.asarray(spectra.fiberId) == self.fiberId)
        if index.size != 1:
            raise KeyError(f"fiberId {self.fiberId} not present exactly once in spectra")
        return np.asarray(getattr(spectra, name))[index[0], self.slice]


@dataclass
class LsfKernelArray:
    """LSFs at many positions, all of the same length

    Row ``i`` of ``values`` covers pixels ``start[i] .. start[i] + width - 1``
    of fibre ``fiberId[i]``'s row; pixels beyond the ends of the row are 0.

    Parameters
    ----------
    fiberId : `numpy.ndarray` of `int`
        Fibre of each kernel.
    center : `numpy.ndarray` of `float`
        First moment of each kernel (native coordinate).
    start : `numpy.ndarray` of `int`
        Index of each kernel's first pixel.
    values : `numpy.ndarray`, shape ``(N, width)``
        The kernels.
    """
    fiberId: np.ndarray
    center: np.ndarray
    start: np.ndarray
    values: np.ndarray

    def __len__(self):
        return len(self.center)

    def __getitem__(self, index):
        """The ``index``-th kernel, as an `LsfKernel` (not trimmed at the row's ends)"""
        return LsfKernel(int(self.fiberId[index]), float(self.center[index]), int(self.start[index]),
                         self.values[index])


class LsfModel(ABC):
    """The line-spread function of a set of fibres

    Positions are in the model's native coordinate, given by ``units``:
    ``"pixel"`` (the row index of an arm's spectrum) or ``"nm"`` (a merged
    spectrum).

    Parameters
    ----------
    fiberId : array-like of `int`
        Fibres covered by the model.
    length : `int`
        Length of each fibre's spectrum (pixels).
    """
    units = None
    defaultHalfWidth = 12.0
    defaultNodeSpacing = 64.0   # spacing (native units) of the nodes for computeGaussianSigma
    defaultSampling = 1/15      # sampling (native units) of the profiles for computeGaussianSigma (odd 1/n)

    def __init__(self, fiberId, length):
        self.fiberId = np.asarray(fiberId, dtype=int)
        self.length = int(length)

    def __contains__(self, fiberId):
        return int(fiberId) in set(self.fiberId.tolist())

    def _checkFiberId(self, fiberId):
        """Raise if any of ``fiberId`` is not in the model"""
        missing = np.setdiff1d(np.unique(np.asarray(fiberId, dtype=int)), self.fiberId)
        if missing.size > 0:
            raise KeyError(f"fiberId(s) {missing.tolist()} not in the LSF model")

    @abstractmethod
    def computeKernel(self, fiberId, center, halfWidth=None, normalize=True):
        """The LSF of a fibre at a position

        Parameters
        ----------
        fiberId : `int`
            Fibre identifier.
        center : `float`
            Position of the LSF's first moment (native units).
        halfWidth : `float`, optional
            The kernel covers ``center +/- halfWidth`` (pixels), with the end
            pixels weighted by the fraction of them inside (see `LsfKernel`);
            defaults to ``defaultHalfWidth``.
        normalize : `bool`
            Normalise the kernel to sum 1?

        Returns
        -------
        kernel : `LsfKernel`
            The LSF, clipped to the spectrum's row.
        """
        raise NotImplementedError("Subclasses must implement")

    @abstractmethod
    def computeKernelAtWavelength(self, fiberId, wavelength, halfWidth=None, normalize=True, *,
                                  wavelengthArray=None):
        """The LSF of a fibre at a wavelength

        Parameters
        ----------
        fiberId : `int`
            Fibre identifier.
        wavelength : `float`
            Wavelength of the LSF's first moment (nm).
        halfWidth : `float`, optional
            The kernel covers pixels within ``halfWidth`` (pixels) of ``center``;
            defaults to ``defaultHalfWidth``.
        normalize : `bool`
            Normalise the kernel to sum 1?
        wavelengthArray : `numpy.ndarray`, optional
            Wavelength of each pixel of the fibre's spectrum (e.g. the row of a
            `PfsArm`'s ``wavelength``); defaults to the model's own.

        Returns
        -------
        kernel : `LsfKernel`
            The LSF, with ``wavelength`` set.
        """
        raise NotImplementedError("Subclasses must implement")

    @abstractmethod
    def getParameters(self, fiberId, center):
        """The LSF's parameters at fibre(s) and position(s)

        Parameters
        ----------
        fiberId : `int` or array-like of `int`
            Fibre identifier(s).
        center : `float` or array-like of `float`
            Position(s) (native units); broadcast against ``fiberId``.

        Returns
        -------
        params : `dict` [`str`, `numpy.ndarray`]
            The parameters, by name.
        """
        raise NotImplementedError("Subclasses must implement")

    def computeKernels(self, fiberId, center, halfWidth=None):
        """LSFs at many positions, all the same length (normalised to sum 1)

        Parameters
        ----------
        fiberId : `int` or array-like of `int`
            Fibre identifier(s).
        center : array-like of `float`
            Positions (native units); broadcast against ``fiberId``.
        halfWidth : `float`, optional
            Each kernel is as from `computeKernel`, padded with zeros to
            ``2*ceil(halfWidth) + 2`` pixels starting at ``ceil(center -
            halfWidth - 0.5)``.

        Returns
        -------
        kernels : `LsfKernelArray`
            The kernels.
        """
        halfWidth = self.defaultHalfWidth if halfWidth is None else halfWidth
        fiberId, center = np.broadcast_arrays(np.asarray(fiberId, dtype=int), np.asarray(center, dtype=float))
        fiberId = fiberId.ravel()
        center = center.ravel()
        width = 2*int(np.ceil(halfWidth)) + 2
        start = np.ceil(center - halfWidth - 0.5).astype(int)
        values = np.zeros((center.size, width))
        for i, (ff, cc, ss) in enumerate(zip(fiberId, center, start)):
            kernel = self.computeKernel(int(ff), float(cc), halfWidth=halfWidth)
            values[i, kernel.start - ss:kernel.stop - ss] = kernel.values
        return LsfKernelArray(fiberId, center, start, values)

    def _sampleProfiles(self, fiberId, center, step, halfWidth, integrated=True):
        """The LSF densities on a fine grid about their first moments

        Parameters
        ----------
        fiberId, center : `numpy.ndarray`
            Fibre identifiers and positions (native units), 1-D, same length.
        step : `float`
            Sampling of the grid (native units).
        halfWidth : `float`
            The grid covers ``+/- halfWidth`` (native units).
        integrated : `bool`
            Include the sampling of the spectrum (e.g. the pixel integration)?

        Returns
        -------
        profiles : `numpy.ndarray`, shape ``(len(center), M)``
            Densities (per native unit) at offsets
            ``(arange(M) - (M - 1)/2)*step`` from the first moment.
        """
        raise NotImplementedError("Subclasses must implement")

    def computeGaussianSigma(self, fiberId, center, method="fit", intrinsicWidth=0.0, *, integrated=True,
                             halfWidth=None, nodeSpacing=None, returnAmplitude=False):
        """The rms of the Gaussian equivalent to the LSF

        Claude: the LSF isn't Gaussian (it's flat-topped: the disk is wider
        than the blur), so its "Gaussian equivalent" depends on what it's used
        for. A Gaussian ``G`` used in place of the LSF ``L`` to convolve a
        spectrum ``T`` makes an error ``int |T(k)|^2 |L(k) - G(k)|^2 dk``, so
        the right ``G`` depends on the spectrum's power spectrum. The limits:

        - ``"fit"``: unresolved features (a white spectrum): the least-squares
          fit of a Gaussian, with a free amplitude and constant weights, to
          the LSF (which, by Parseval, minimises the error); this is also the
          maximum-likelihood fit of a Gaussian to a faint, background-limited
          line.
        - ``"moment"``: features much wider than the LSF: only
          ``L(k) ~ 1 - mu_2 k^2/2`` matters, so the Gaussian with the LSF's
          second moment (within ``+/- halfWidth``, halo included).

        With ``method="fit"`` and ``intrinsicWidth > 0``, the Gaussian is fit
        to the LSF convolved with a Gaussian of rms ``intrinsicWidth`` (a
        feature of that width), and ``intrinsicWidth`` is then removed in
        quadrature; this goes from the first limit to the second as
        ``intrinsicWidth`` increases.

        Parameters
        ----------
        fiberId : `int` or array-like of `int`
            Fibre identifier(s).
        center : `float` or array-like of `float`
            Position(s) (native units); broadcast against ``fiberId``.
        method : `str`
            ``"fit"`` or ``"moment"``.
        intrinsicWidth : `float`
            RMS (native units) of the features (``method="fit"`` only).
        integrated : `bool`
            The LSF as sampled in the spectrum (pixel-integrated: the Gaussian
            to sample at pixel centres, as `pfs.drp.stella.GaussianLsf`
            does), or before integration?
        halfWidth : `float`, optional
            The LSF is used out to ``+/- halfWidth`` (native units); defaults
            to ``defaultHalfWidth``.
        nodeSpacing : `float`, optional
            The result is computed at nodes this far apart (native units)
            along each fibre and interpolated (linearly); 0 to compute it at
            every position. Defaults to ``defaultNodeSpacing``.
        returnAmplitude : `bool`
            Also return the fitted Gaussian's area (the LSF has unit area): the
            factor by which a flux measured by fitting the Gaussian is biased
            (1 for ``"moment"``).

        Returns
        -------
        sigma : `numpy.ndarray`
            RMS of the equivalent Gaussian (native units).
        amplitude : `numpy.ndarray`
            Area of the fitted Gaussian; only if ``returnAmplitude``.
        """
        if method not in ("fit", "moment"):
            raise ValueError(f"Unknown method {method!r}: use 'fit' or 'moment'")
        halfWidth = self.defaultHalfWidth if halfWidth is None else halfWidth
        nodeSpacing = self.defaultNodeSpacing if nodeSpacing is None else nodeSpacing
        self._checkFiberId(fiberId)
        fiberId, center = np.broadcast_arrays(np.asarray(fiberId, dtype=int), np.asarray(center, dtype=float))
        shape = center.shape
        fiberId, center = fiberId.ravel(), center.ravel()

        # The nodes at which to compute it
        nodeFiber, nodeCenter, which = [], [], []
        for ff in np.unique(fiberId):
            select = np.flatnonzero(fiberId == ff)
            cc = center[select]
            if nodeSpacing > 0:
                lo = np.floor(np.nanmin(cc)/nodeSpacing)
                hi = np.ceil(np.nanmax(cc)/nodeSpacing)
                nodes = np.arange(lo, hi + 1)*nodeSpacing
            else:
                nodes = cc
            nodeFiber.append(np.full(len(nodes), ff))
            nodeCenter.append(nodes)
            which.append((select, len(nodes)))
        nodeFiber = np.concatenate(nodeFiber)
        nodeCenter = np.concatenate(nodeCenter)

        step = self.defaultSampling
        nodeSigma = np.empty(nodeCenter.size)
        nodeAmplitude = np.empty(nodeCenter.size)
        batch = 1000         # limits the memory used by the profiles
        for start in range(0, nodeCenter.size, batch):
            nodes = slice(start, start + batch)
            profiles = self._sampleProfiles(nodeFiber[nodes], nodeCenter[nodes], step, halfWidth, integrated)
            nodeSigma[nodes], nodeAmplitude[nodes] = gaussianEquivalentSigma(profiles, step, method,
                                                                             intrinsicWidth)

        sigma = np.empty(center.size)
        amplitude = np.empty(center.size)
        start = 0
        for select, num in which:
            nodes = slice(start, start + num)
            if nodeSpacing > 0:
                sigma[select] = np.interp(center[select], nodeCenter[nodes], nodeSigma[nodes])
                amplitude[select] = np.interp(center[select], nodeCenter[nodes], nodeAmplitude[nodes])
            else:
                sigma[select] = nodeSigma[nodes]
                amplitude[select] = nodeAmplitude[nodes]
            start += num
        sigma = sigma.reshape(shape)
        amplitude = amplitude.reshape(shape)
        return (sigma, amplitude) if returnAmplitude else sigma

    def computeGaussianSigmaAtWavelength(self, fiberId, wavelength, method="fit", intrinsicWidth=0.0, *,
                                         wavelengthArray=None, **kwargs):
        """The rms (nm) of the Gaussian equivalent to the LSF at wavelength(s)

        As `computeGaussianSigma`, but with positions and ``intrinsicWidth``
        (and the result) in nm.

        Parameters
        ----------
        fiberId : `int` or array-like of `int`
            Fibre identifier(s).
        wavelength : `float` or array-like of `float`
            Wavelength(s) (nm); broadcast against ``fiberId``.
        method : `str`
            ``"fit"`` or ``"moment"``; see `computeGaussianSigma`.
        intrinsicWidth : `float`
            RMS (nm) of the features (``method="fit"`` only; converted to
            pixels with the median dispersion of the positions requested).
        wavelengthArray : `numpy.ndarray`, optional
            Wavelength of each pixel of the fibre's spectrum (a single
            ``fiberId`` only); defaults to the model's own.
        **kwargs
            As for `computeGaussianSigma`.

        Returns
        -------
        sigma : `numpy.ndarray`
            RMS of the equivalent Gaussian (nm).
        amplitude : `numpy.ndarray`
            Area of the fitted Gaussian; only if ``returnAmplitude``.
        """
        raise NotImplementedError("Subclasses must implement")

    def _window(self, center, halfWidth):
        """The pixels of a kernel and its window

        Parameters
        ----------
        center : `float`
            Centre of the kernel (pixels).
        halfWidth : `float`
            Half-width of the window (pixels).

        Returns
        -------
        start : `int`
            Index of the first pixel.
        window : `numpy.ndarray`
            Fraction of each pixel, from ``start``, within ``center +/- halfWidth``
            (clipped to the spectrum's row).
        """
        start = max(0, int(np.ceil(center - halfWidth - 0.5)))
        stop = min(self.length, int(np.floor(center + halfWidth + 0.5)) + 1)
        if stop <= start:
            raise ValueError(f"LSF at {center} with halfWidth {halfWidth} doesn't overlap the spectrum")
        window = np.clip(halfWidth + 0.5 - np.abs(np.arange(start, stop) - center), 0.0, 1.0)
        return start, window


class FiberSpline:
    """A function of row along each fibre, as a cubic spline per fibre

    Used to carry a detectorMap's x(row) and wavelength(row) in an LSF model,
    so that it is self-contained.

    Parameters
    ----------
    fiberId : array-like of `int`
        Fibre identifiers.
    knots : array-like of `float`
        Rows of the spline knots (shared by all fibres; increasing).
    values : `numpy.ndarray`, shape ``(len(fiberId), len(knots))``
        Values at the knots for each fibre.
    """
    def __init__(self, fiberId, knots, values):
        self.fiberId = np.asarray(fiberId, dtype=int)
        self.knots = np.asarray(knots, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self._index = {int(ff): ii for ii, ff in enumerate(self.fiberId)}
        self._splines = {}

    @classmethod
    def fromFunction(cls, fiberId, length, step, func):
        """Construct by sampling ``func(fiberId, rows)`` every ``step`` rows of ``0 .. length - 1``"""
        knots = np.unique(np.append(np.arange(0, length, step), length - 1)).astype(float)
        return cls(fiberId, knots, [func(int(ff), knots) for ff in fiberId])

    def _spline(self, fiberId):
        """The spline for ``fiberId``"""
        spline = self._splines.get(fiberId)
        if spline is None:
            spline = scipy.interpolate.CubicSpline(self.knots, self.values[self._index[fiberId]])
            self._splines[fiberId] = spline
        return spline

    def __call__(self, fiberId, row):
        """Evaluate at fibre(s) ``fiberId`` and row(s) ``row`` (broadcast)"""
        fiberId, row = np.broadcast_arrays(np.asarray(fiberId, dtype=int), np.asarray(row, dtype=float))
        result = np.empty(row.shape)
        for ff in np.unique(fiberId):
            select = fiberId == ff
            result[select] = self._spline(int(ff))(row[select])
        return result

    def __reduce__(self):
        return self.__class__, (self.fiberId, self.knots, self.values)


CUMULANT_NAMES = ("S", "K", "lam5", "lam6")          # the blur's standardised cumulants, in order


def addBlur(rms, cumulants, extraVariance, extraKappa4=0.0):
    """Convolve a blur kernel with a further symmetric kernel

    Cumulants add under convolution, so the variances add and so do the
    fourth cumulants; the odd cumulants are unchanged (the extra kernel is
    symmetric), but all the standardised cumulants change with the rms.
    The extra variance may be negative (a narrower kernel), as long as the
    result's is positive.

    Parameters
    ----------
    rms : `float` or `numpy.ndarray`
        The kernel's rms (pixels).
    cumulants : sequence of `float` or `numpy.ndarray`
        The kernel's standardised cumulants ``[S, K, lam5, lam6]`` (any
        leading subset).
    extraVariance : `float`
        The extra kernel's variance (pixels^2; may be negative).
    extraKappa4 : `float`
        The extra kernel's fourth cumulant (pixels^4; not standardised).

    Returns
    -------
    rms : `float` or `numpy.ndarray`
        The convolved kernel's rms.
    cumulants : `list`
        The convolved kernel's standardised cumulants.
    """
    rms = np.asarray(rms, dtype=float)
    newRms = np.sqrt(np.maximum(rms**2 + extraVariance, 1.0e-6))
    newCumulants = []
    for order, value in enumerate(cumulants, start=3):
        kappa = np.asarray(value, dtype=float)*rms**order
        if order == 4:
            kappa = kappa + extraKappa4
        newCumulants.append(kappa/newRms**order)
    return newRms, newCumulants


SHAPE_NAMES = ("radius", "rms") + CUMULANT_NAMES + ("lorentz", "haloFrac", "haloWidth")   # profile parameters


def gaussianEquivalentSigma(profiles, step, method="fit", intrinsicWidth=0.0):
    """Claude: the rms of the Gaussian equivalent to sampled profiles

    See `LsfModel.computeGaussianSigma` for the methods.

    Parameters
    ----------
    profiles : `numpy.ndarray`, shape ``(N, M)``
        Profiles, sampled at offsets ``(arange(M) - (M - 1)/2)*step`` from
        their first moments (any normalisation).
    step : `float`
        Sampling of the profiles.
    method : `str`
        ``"fit"`` (least squares, free amplitude, constant weights) or
        ``"moment"`` (second moment).
    intrinsicWidth : `float`
        For ``"fit"``, convolve the profiles with a Gaussian of this rms before
        fitting, and remove it in quadrature afterwards.

    Returns
    -------
    sigma : `numpy.ndarray`, shape ``(N,)``
        RMS of the equivalent Gaussian (units of ``step``).
    amplitude : `numpy.ndarray`, shape ``(N,)``
        Area of the fitted Gaussian relative to the profile's (1 for
        ``"moment"``).
    """
    profiles = np.atleast_2d(np.asarray(profiles, dtype=float))
    num, size = profiles.shape
    offsets = (np.arange(size) - 0.5*(size - 1))*step
    area = profiles.sum(axis=1)*step
    if method == "moment":
        mean = (profiles*offsets).sum(axis=1)*step/area
        variance = (profiles*(offsets - mean[:, None])**2).sum(axis=1)*step/area
        return np.sqrt(variance), np.ones(num)
    if method != "fit":
        raise ValueError(f"Unknown method {method!r}: use 'fit' or 'moment'")
    profiles = profiles/area[:, None]
    if intrinsicWidth > 0:
        pad = int(np.ceil(5*intrinsicWidth/step))
        profiles = np.pad(profiles, ((0, 0), (pad, pad)))
        offsets = (np.arange(size + 2*pad) - 0.5*(size + 2*pad - 1))*step
        profiles = scipy.ndimage.gaussian_filter1d(profiles, intrinsicWidth/step, axis=1, mode="constant")

    def score(width):
        """Matched-filter statistics of unit-area Gaussians of rms ``width`` (shape ``(K,)`` or ``(N,)``)"""
        width = np.asarray(width, dtype=float)
        gauss = np.exp(-0.5*(offsets/width[..., None])**2)/(np.sqrt(2*np.pi)*width[..., None])
        norm2 = (gauss**2).sum(axis=-1)*step
        return gauss, norm2

    # Minimising |p - A g|^2 over A is maximising (p.g)^2/|g|^2: scan a grid of widths, then refine
    widths = np.geomspace(0.05, 20.0, 600)*max(np.sqrt(np.median(
        (profiles*offsets**2).sum(axis=1)*step)), step)
    gauss, norm2 = score(widths)
    stat = (profiles @ gauss.T*step)**2/norm2
    best = np.clip(np.argmax(stat, axis=1), 1, len(widths) - 2)
    rows = np.arange(num)
    lnW = np.log(widths)
    y0, y1, y2 = stat[rows, best - 1], stat[rows, best], stat[rows, best + 1]
    denom = y0 - 2*y1 + y2
    shift = np.where(denom < 0, 0.5*(y0 - y2)/np.where(denom < 0, denom, -1.0), 0.0)
    width = np.exp(lnW[best] + np.clip(shift, -1, 1)*(lnW[1] - lnW[0]))
    gauss, norm2 = score(width)
    amplitude = (profiles*gauss).sum(axis=1)*step/norm2
    sigma = np.sqrt(np.clip(width**2 - intrinsicWidth**2, 0.0, None))
    return sigma, amplitude


@dataclass
class ArmLsfModel(LsfModel):
    """The LSF of one detector (arm, spectrograph), in pixels

    The LSF is the extracted, pixel-integrated image of the fibre face, a disk
    of radius ``radius``, convolved with a blur of rms ``rms`` whose shape is
    an Edgeworth series in the standardised cumulants ``S, K, ...``,
    optionally with a halo (a fraction ``haloFrac`` of the light blurred by a
    Gaussian of rms ``haloWidth``) and convolved with a Lorentzian of FWHM
    ``lorentz`` (pixels; see `pfs.drp.stella.lsfProfile.diskProfile`). Each parameter is a smooth
    function (`SmoothSurface`) of position on the detector, in the frame given
    by ``coords`` (which allows for the offset between the two CCDs of a b, r or
    m detector). The kernel's first moment is at the requested centre: the
    per-line fits from which the model was built were pinned so that their
    first moments match the data's.

    The surfaces were fit with per-group offsets (e.g. per lamp, which absorb
    differences in focus or brightness between exposures); the kernels use the
    bare surfaces, which describe the reference groups (``surfaces[name].reference``).

    ``centroid`` (if present) is not part of the profile: it is the measured
    offset (pixels) of the arc lines from the detectorMap used in the fit, with
    a constant per fibre (`getCentroidCorrection`).

    Parameters
    ----------
    arm : `str`
        Arm (``b``, ``r``, ``m``, ``n``).
    spectrograph : `int`
        Spectrograph module.
    fiberId : array-like of `int`
        Fibres covered by the model.
    length : `int`
        Number of rows (pixels) in each spectrum.
    surfaces : `dict` [`str`, `SmoothSurface`]
        The fitted parameters: ``radius``, ``rms``, the blur's cumulants
        ``S, K, ...``, ``lorentz``, ``haloFrac`` and ``haloWidth`` (if fit), and
        ``centroid``.
    coords : `DetectorCoords`
        Detector pixels to the frame of the surfaces.
    xCenter : `FiberSpline`
        Each fibre's trace position x (pixels) as a function of row.
    wavelength : `FiberSpline`
        Each fibre's wavelength (nm) as a function of row, from the
        detectorMap used in the fit.
    extraction : `str`
        ``"optimal"`` or ``"boxcar"``: the extraction that the profile models.
    centroidZeroPoint : `float`
        Offset (pixels) between the data's and the free-fit models' first
        moments (removed when pinning the fits).
    metadata : `dict`
        Provenance: visits, lamps, configuration.
    engineeringFiberId : array-like of `int`
        The engineering (IIS) fibres. Their cores are smaller than the science
        (PFI) fibres', so their disk radius is the surface's times
        ``iisRadiusScale``; everything else is shared.
    iisRadiusScale : `SmoothSurface`, optional
        Ratio of the engineering fibres' disk radius to the science fibres',
        as a function of position (fit from IIS arcs, with the PFI blur kernel
        held fixed).
    iisHaloScale : `SmoothSurface`, optional
        Ratio of the engineering fibres' halo fraction to the science fibres'.
    iisExtraBlur : (`float`, `float`), optional
        The engineering fibres' blur kernel is the science fibres' convolved
        with a further symmetric kernel of this variance (pixels^2; may be
        negative) and fourth cumulant (pixels^4); see `addBlur`.
    fiberGroups : `dict` [`str`, `dict` [`int`, `str`]]
        Per-fibre grouping factors whose offsets are part of the model, e.g.
        ``{"mtp": {fiberId: "U3-1"}}`` (the fibres' MTP sub-bundle, whose
        focal-ratio degradation blurs their images): each surface's offset
        for a fibre's group is added to the surface.
    """
    arm: str
    spectrograph: int
    fiberId: np.ndarray
    length: int
    surfaces: dict
    coords: DetectorCoords
    xCenter: FiberSpline
    wavelength: FiberSpline
    extraction: str = "optimal"
    centroidZeroPoint: float = 0.0
    metadata: dict = field(default_factory=dict)
    engineeringFiberId: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    iisRadiusScale: SmoothSurface = None
    iisHaloScale: SmoothSurface = None
    iisExtraBlur: tuple = None
    fiberGroups: dict = field(default_factory=dict)

    units = "pixel"

    def __post_init__(self):
        LsfModel.__init__(self, self.fiberId, self.length)

    @property
    def cumulantNames(self):
        """Names of the blur's cumulants in the model, in order"""
        names = []
        for nn in CUMULANT_NAMES:
            if nn not in self.surfaces:
                break
            names.append(nn)
        return names

    @property
    def parameterNames(self):
        """Names of the model's parameters"""
        return [nn for nn in SHAPE_NAMES + ("centroid",) if nn in self.surfaces]

    @property
    def blurOrder(self):
        """Edgeworth order of the blur (2: Gaussian)"""
        return 2 + len(self.cumulantNames)

    def getDetectorPosition(self, fiberId, row):
        """Detector position (x, y) (pixels) of fibre(s) ``fiberId`` at row(s) ``row``"""
        fiberId, row = np.broadcast_arrays(np.asarray(fiberId, dtype=int), np.asarray(row, dtype=float))
        return self.xCenter(fiberId, row), row.astype(float)

    def evaluateParameter(self, name, x, y, **labels):
        """A parameter at detector position(s) ``(x, y)`` (pixels)

        Parameters
        ----------
        name : `str`
            Parameter name (see ``parameterNames``).
        x, y : array-like
            Detector position(s) (pixels).
        **labels
            Group labels whose offsets to include, e.g. ``element="Ne"`` or
            ``fiberId=...`` (scalar or arrays matching ``x``).

        Returns
        -------
        value : `numpy.ndarray`
            The parameter.
        """
        u, v = self.coords(x, y)
        return self.surfaces[name].evaluate(u, v, **labels)

    def getParameters(self, fiberId, center):
        """The LSF's parameters at fibre(s) and row(s)

        Parameters
        ----------
        fiberId : `int` or array-like of `int`
            Fibre identifier(s).
        center : `float` or array-like of `float`
            Row(s) (pixels); broadcast against ``fiberId``.

        Returns
        -------
        params : `dict` [`str`, `numpy.ndarray`]
            ``x``, ``y`` (detector position, pixels), the profile parameters
            (``radius``, ``rms``, the cumulants, and ``lorentz``, ``haloFrac``
            and ``haloWidth`` if fit), and ``sigma``, the rms width of the
            profile's core before pixel integration (pixels; no halo or
            Lorentzian).
        """
        self._checkFiberId(fiberId)
        x, y = self.getDetectorPosition(fiberId, center)
        u, v = self.coords(x, y)
        params = dict(x=x, y=y)
        groups = getattr(self, "fiberGroups", {})      # older models lack these
        if groups:
            fibers = np.broadcast_to(np.asarray(fiberId, dtype=int), np.shape(x))
            groupLabels = {}
            for factor, table in groups.items():
                labels = [table.get(int(ff), "") for ff in fibers.ravel()]
                groupLabels[factor] = np.array(labels, dtype=str).reshape(fibers.shape)
        for name in ("radius", "rms") + tuple(self.cumulantNames):
            params[name] = self.surfaces[name](u, v)
            for factor, labels in (groupLabels.items() if groups else ()):
                params[name] = params[name] + self.surfaces[name].getOffset(factor, labels)
        if "lorentz" in self.surfaces:
            params["lorentz"] = np.clip(self.surfaces["lorentz"](u, v), 0.0, None)
        if "haloFrac" in self.surfaces:
            params["haloFrac"] = np.clip(self.surfaces["haloFrac"](u, v), 0.0, 1.0)
            params["haloWidth"] = np.clip(self.surfaces["haloWidth"](u, v), 1.0e-3, None)
        # The engineering (IIS) fibres: their own disk radius and halo fraction (older models lack these)
        engineering = getattr(self, "engineeringFiberId", np.zeros(0, dtype=int))
        radiusScale = getattr(self, "iisRadiusScale", None)
        haloScale = getattr(self, "iisHaloScale", None)
        extraBlur = getattr(self, "iisExtraBlur", None)
        if len(engineering) > 0 and any(xx is not None for xx in (radiusScale, haloScale, extraBlur)):
            isEngineering = np.isin(np.broadcast_to(np.asarray(fiberId, dtype=int), np.shape(x)), engineering)
            if radiusScale is not None:
                params["radius"] = np.where(isEngineering, params["radius"]*radiusScale(u, v),
                                            params["radius"])
            if haloScale is not None and "haloFrac" in params:
                scaled = np.clip(params["haloFrac"]*haloScale(u, v), 0.0, 1.0)
                params["haloFrac"] = np.where(isEngineering, scaled, params["haloFrac"])
            if extraBlur is not None:
                names = [nn for nn in CUMULANT_NAMES if nn in params]
                rms, cumulants = addBlur(params["rms"], [params[nn] for nn in names], *extraBlur)
                params["rms"] = np.where(isEngineering, rms, params["rms"])
                for nn, cc in zip(names, cumulants):
                    params[nn] = np.where(isEngineering, cc, params[nn])
        params["sigma"] = diskSigma(params["radius"], params["rms"])
        return params

    def getCentroidCorrection(self, fiberId, row):
        """Measured offset of the arc lines from the detectorMap used in the fit

        Parameters
        ----------
        fiberId : `int` or array-like of `int`
            Fibre identifier(s).
        row : `float` or array-like of `float`
            Row(s) (pixels).

        Returns
        -------
        offset : `numpy.ndarray`
            Data minus detectorMap position along the dispersion (pixels), with
            each fibre's constant offset (for the reference lamp).
        """
        self._checkFiberId(fiberId)
        x, y = self.getDetectorPosition(fiberId, row)
        return self.evaluateParameter("centroid", x, y, fiberId=np.broadcast_to(fiberId, np.shape(x)))

    def _profile(self, fiberId, center, pixels):
        """The LSF profile at ``pixels`` for ``fiberId`` with first moment at row ``center``"""
        params = self.getParameters(int(fiberId), float(center))
        cumulants = [float(params[nn]) for nn in self.cumulantNames]
        return diskProfile(pixels, float(center), float(params["radius"]), float(params["rms"]),
                           cumulants, self.extraction, float(params.get("lorentz", 0.0)),
                           float(params.get("haloFrac", 0.0)), float(params.get("haloWidth", np.nan)))

    def computeKernel(self, fiberId, center, halfWidth=None, normalize=True):
        halfWidth = self.defaultHalfWidth if halfWidth is None else halfWidth
        start, window = self._window(center, halfWidth)
        values = self._profile(fiberId, center, np.arange(start, start + len(window)))*window
        norm = float(np.sum(values))
        if normalize and norm > 0:
            values = values/norm
        return LsfKernel(int(fiberId), float(center), start, values, norm=norm)

    def computeKernelAtWavelength(self, fiberId, wavelength, halfWidth=None, normalize=True, *,
                                  wavelengthArray=None):
        rows = np.arange(self.length, dtype=float)
        if wavelengthArray is None:
            wavelengthArray = self.wavelength(int(fiberId), rows)
        wavelengthArray = np.asarray(wavelengthArray, dtype=float)
        good = np.isfinite(wavelengthArray)
        if wavelengthArray[good][-1] < wavelengthArray[good][0]:
            raise ValueError("wavelengthArray must increase with row")
        center = float(np.interp(wavelength, wavelengthArray[good], rows[good], left=np.nan, right=np.nan))
        if not np.isfinite(center):
            raise ValueError(f"wavelength {wavelength} is outside the spectrum of fiberId {fiberId}")
        kernel = self.computeKernel(fiberId, center, halfWidth, normalize)
        kernel.wavelength = wavelengthArray[kernel.slice]
        return kernel

    def _sampleProfiles(self, fiberId, center, step, halfWidth, integrated=True):
        fiberId = np.asarray(fiberId, dtype=int)
        center = np.asarray(center, dtype=float)
        half = int(np.round(halfWidth/step))
        offsets = np.arange(-half, half + 1)*step
        params = self.getParameters(fiberId, center)
        names = self.cumulantNames
        box = np.ones(max(1, int(np.round(1/step))))
        box /= box.size
        profiles = np.empty((center.size, offsets.size))
        for ii in range(center.size):
            # pixel integration over a tiny "pixel" of width `step` approximates the density
            density = diskProfile(offsets/step, 0.0, float(params["radius"][ii])/step,
                                  float(params["rms"][ii])/step, [float(params[nn][ii]) for nn in names],
                                  self.extraction, 0.0, float(params["haloFrac"][ii]) if "haloFrac" in params
                                  else 0.0, float(params["haloWidth"][ii])/step if "haloWidth" in params
                                  else np.nan)/step
            lorentz = float(params["lorentz"][ii]) if "lorentz" in params else 0.0
            if lorentz > 0:                # convolve with the Lorentzian's mass in each cell
                cellEdges = np.append(offsets - step/2, offsets[-1] + step/2)
                density = np.convolve(density, np.diff(np.arctan(cellEdges/(0.5*lorentz)))/np.pi, mode="same")
            if integrated:                 # the flux in a pixel centred at each offset
                density = np.convolve(density, box, mode="same")
            profiles[ii] = density
        return profiles

    def computeGaussianSigmaAtWavelength(self, fiberId, wavelength, method="fit", intrinsicWidth=0.0, *,
                                         wavelengthArray=None, **kwargs):
        fiberId, wavelength = np.broadcast_arrays(np.asarray(fiberId, dtype=int),
                                                  np.asarray(wavelength, dtype=float))
        rows = np.arange(self.length, dtype=float)
        center = np.empty(wavelength.shape)
        dispersion = np.empty(wavelength.shape)
        for ff in np.unique(fiberId):
            select = fiberId == ff
            lam = (self.wavelength(int(ff), rows) if wavelengthArray is None
                   else np.asarray(wavelengthArray, dtype=float))
            good = np.isfinite(lam)
            if lam[good][-1] < lam[good][0]:
                raise ValueError("wavelengthArray must increase with row")
            center[select] = np.interp(wavelength[select], lam[good], rows[good], left=np.nan, right=np.nan)
            dispersion[select] = np.interp(center[select], rows[good], np.gradient(lam[good], rows[good]))
        if not np.all(np.isfinite(center)):
            raise ValueError("Some wavelengths are outside the spectra")
        intrinsicPixels = intrinsicWidth/np.median(dispersion) if intrinsicWidth > 0 else 0.0
        result = self.computeGaussianSigma(fiberId, center, method, intrinsicPixels, **kwargs)
        if kwargs.get("returnAmplitude", False):
            return result[0]*dispersion, result[1]
        return result*dispersion

    def computeFwhm(self, fiberId, center, step=0.01, halfWidth=None):
        """FWHM (pixels) of the LSF before pixel integration

        Parameters
        ----------
        fiberId : `int`
            Fibre identifier.
        center : `float`
            Row (pixels).
        step : `float`
            Sampling (pixels) for the measurement.
        halfWidth : `float`, optional
            Half-width (pixels) of the region searched.

        Returns
        -------
        fwhm : `float`
            Full width at half maximum (pixels).
        """
        halfWidth = self.defaultHalfWidth if halfWidth is None else halfWidth
        density = self._sampleProfiles(np.array([fiberId]), np.array([center]), step, halfWidth,
                                       integrated=False)[0]
        centers = (np.arange(density.size) - 0.5*(density.size - 1))*step
        above = np.flatnonzero(density >= 0.5*density.max())
        lo, hi = above[0], above[-1]
        half = 0.5*density.max()
        xLo = np.interp(half, density[lo - 1:lo + 1], centers[lo - 1:lo + 1])
        xHi = np.interp(half, density[hi:hi + 2][::-1], centers[hi:hi + 2][::-1])
        return float(xHi - xLo)

    def __repr__(self):
        engineering = len(getattr(self, "engineeringFiberId", []))
        return (f"ArmLsfModel({self.arm}{self.spectrograph}: {len(self.fiberId)} fibres"
                + (f" ({engineering} engineering)" if engineering else "")
                + f", blur order {self.blurOrder}, {self.extraction} extraction;"
                + f" {', '.join(self.parameterNames)})")
