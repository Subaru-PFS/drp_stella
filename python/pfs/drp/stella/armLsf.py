"""Models of the line-spread function over a spectrograph's fibres

An `LsfModel` returns, for a fibre and a (floating-point) position along its
spectrum, a short `LsfKernel`: the LSF's flux in each pixel of a slice of the
spectrum's row, with methods to extract the matching slice of e.g. ``flux``.

`ArmLsf` is the model for one detector (arm, spectrograph), in pixels:
an extracted, pixel-integrated image of the fibre face (a disk) blurred by an
Edgeworth-expanded Gaussian, whose parameters vary smoothly over the detector.
It is fit from arc lines by `pfs.drp.stella.pipelines.FitLsfTask`.
"""
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import astropy.units as u
from astropy.table import QTable
import numpy as np
import scipy.interpolate
import scipy.ndimage

from .lsfProfile import diskProfile, diskSigma
from pfs.datamodel.pfsLsf import PfsArmLsfData, PfsLsf, PfsLsfSurface

from .smoothSurface import DetectorCoords, SmoothSurface

__all__ = ("LsfKernel", "LsfParameter", "LsfModel", "ArmLsf", "GaussianArmLsf", "FiberSpline",
           "gaussianEquivalentSigma")


@dataclass
class LsfKernel:
    """The LSF of a spectrum at a position: a short slice of the spectrum

    It is both the convolution kernel and the model of a line at a sub-pixel
    position. It may hold several kernels at once (computed for an array of
    positions), all of the same width: then ``center`` and ``start`` are
    arrays, and ``values`` is two-dimensional.

    Parameters
    ----------
    key : `int` or `pfs.datamodel.Target`
        The spectrum: its ``fiberId``, or its target (coadds).
    center : `float` or `numpy.ndarray`
        Position (pixels) of the LSF's first moment in the spectrum.
    start : `int` or `numpy.ndarray` of `int`
        Index of ``values[..., 0]`` in the spectrum.
    values : `numpy.ndarray`
        LSF flux in each pixel, times the window (normalised to sum 1 unless
        the kernel was computed with ``normalize=False``). The window is the
        fraction of each pixel within ``halfWidth`` of ``center``: 1 except for
        the end pixels, so that it is symmetric about ``center`` and the
        truncation of the LSF's wings doesn't shift the kernel's first moment
        (to first order).
    wavelength : `astropy.units.Quantity`, optional
        Vacuum wavelength of each pixel, if known.
    norm : `float` or `numpy.ndarray`
        Fraction of the LSF's flux within the (windowed) kernel before
        normalisation (< 1 from the truncated wings, and if clipped at the ends
        of the spectrum).
    """
    key: object
    center: float
    start: int
    values: np.ndarray
    wavelength: u.Quantity = None
    norm: float = 1.0

    @property
    def fiberId(self):
        """The spectrum's fiberId (same as ``key``, for spectra identified by fiberId)"""
        return self.key

    @property
    def isBatch(self):
        """Does this hold several kernels?"""
        return np.ndim(self.values) == 2

    @property
    def numKernels(self):
        """Number of kernels held"""
        return len(self.values) if self.isBatch else 1

    def __getitem__(self, index):
        """The ``index``-th of several kernels"""
        if not self.isBatch:
            raise TypeError("Not a batch of kernels")
        return LsfKernel(self.key, float(self.center[index]), int(self.start[index]), self.values[index],
                         None if self.wavelength is None else self.wavelength[index],
                         float(np.broadcast_to(self.norm, (self.numKernels,))[index]))

    def __repr__(self):
        if self.isBatch:
            return f"LsfKernel(key={self.key}, {self.numKernels} kernels of {len(self)} pixels)"
        where = f"center={self.center:.3f} pixels"
        if self.wavelength is not None and len(self.wavelength) == len(self.values):
            lam = np.interp(self.center, self.pixels, self.wavelength.to_value(u.nm))
            where += f" ({lam:.3f} nm, vacuum)"
        return (f"LsfKernel(key={self.key}, {where}, pixels {self.start}:{self.stop}, "
                f"norm={self.norm:.4f})")

    @property
    def stop(self):
        """Index one beyond the kernel's last pixel in the spectrum"""
        return np.asarray(self.start) + len(self) if self.isBatch else self.start + len(self)

    @property
    def slice(self):
        """The `slice` of the spectrum covered by the kernel"""
        if self.isBatch:
            raise TypeError("A batch of kernels has no single slice")
        return slice(self.start, self.stop)

    @property
    def pixels(self):
        """Indices of the kernel's pixels in the spectrum"""
        return np.asarray(self.start)[..., np.newaxis] + np.arange(len(self)) if self.isBatch else \
            np.arange(self.start, self.stop)

    @property
    def offsets(self):
        """Pixel positions relative to ``center``"""
        center = np.asarray(self.center)
        return self.pixels - (center[..., np.newaxis] if self.isBatch else center)

    def __len__(self):
        """Number of pixels in the kernel"""
        return np.shape(self.values)[-1]

    def extract(self, array):
        """The slice of ``array`` matching the kernel

        Parameters
        ----------
        array : `numpy.ndarray`
            A spectrum (or an array whose last axis is the spectrum).

        Returns
        -------
        sliced : `numpy.ndarray`
            ``array[..., self.slice]``; for a batch of kernels, one slice per
            kernel (zero beyond the ends of ``array``).
        """
        array = np.asarray(array)
        if not self.isBatch:
            return array[..., self.slice]
        pixels = self.pixels
        inside = (pixels >= 0) & (pixels < array.shape[-1])
        return np.where(inside, np.take(array, np.clip(pixels, 0, array.shape[-1] - 1), axis=-1), 0)

    def extractFrom(self, spectra, name="flux"):
        """The slice of this spectrum of an attribute of a set of spectra

        Parameters
        ----------
        spectra : `pfs.datamodel.PfsFiberArraySet`
            Spectra, e.g. a `PfsArm`, containing ``fiberId == key``.
        name : `str`
            Attribute, e.g. ``flux``, ``mask``, ``variance`` or ``wavelength``.

        Returns
        -------
        sliced : `numpy.ndarray`
            The matching part of ``getattr(spectra, name)`` for this spectrum.
        """
        index = np.flatnonzero(np.asarray(spectra.fiberId) == self.key)
        if index.size != 1:
            raise KeyError(f"fiberId {self.key} not present exactly once in spectra")
        return self.extract(np.asarray(getattr(spectra, name))[index[0]])


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
    def computeKernel(self, fiberId, center, halfWidth=None, normalize=True, clip=True):
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
        clip : `bool`
            Clip the kernel to the spectrum's row? If `False`, a kernel near the
            end of the row extends beyond it: the LSF belongs to the instrument,
            and the end of the row only means that there's no data there.

        Returns
        -------
        kernel : `LsfKernel`
            The LSF.
        """
        raise NotImplementedError("Subclasses must implement")

    @abstractmethod
    def computeKernelAtWavelength(self, fiberId, wavelength, halfWidth=None, normalize=True):
        """The LSF of a fibre at a wavelength

        Parameters
        ----------
        fiberId : `int`
            Fibre identifier.
        wavelength : `float`
            Vacuum wavelength (nm) of the LSF's first moment.
        halfWidth : `float`, optional
            The kernel covers pixels within ``halfWidth`` (pixels) of ``center``;
            defaults to ``defaultHalfWidth``.
        normalize : `bool`
            Normalise the kernel to sum 1?

        The wavelengths are those of the model (from the fit). For a particular
        set of spectra, use the LSF bound to them (``forSpectra``), which uses
        their wavelengths.

        Returns
        -------
        kernel : `LsfKernel`
            The LSF, with ``wavelength`` set.
        """
        raise NotImplementedError("Subclasses must implement")

    @abstractmethod
    def getParameters(self, fiberId, center):
        """The LSF's parameters at fibre(s) and position(s), as an `astropy.table.QTable`

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
        """The LSFs of a fibre at many positions, all the same length (normalised to sum 1)

        Parameters
        ----------
        fiberId : `int`
            Fibre identifier.
        center : array-like of `float`
            Positions (native units).
        halfWidth : `float`, optional
            Each kernel is as from `computeKernel`, padded with zeros to
            ``2*ceil(halfWidth) + 2`` pixels starting at ``ceil(center -
            halfWidth - 0.5)``.

        Returns
        -------
        kernels : `LsfKernel`
            The kernels, as a batch.
        """
        halfWidth = self.defaultHalfWidth if halfWidth is None else halfWidth
        center = np.atleast_1d(np.asarray(center, dtype=float)).ravel()
        width = 2*int(np.ceil(halfWidth)) + 2
        start = np.ceil(center - halfWidth - 0.5).astype(int)
        values = np.zeros((center.size, width))
        norm = np.zeros(center.size)
        for i, (cc, ss) in enumerate(zip(center, start)):
            kernel = self.computeKernel(int(fiberId), float(cc), halfWidth=halfWidth)
            values[i, kernel.start - ss:kernel.stop - ss] = kernel.values
            norm[i] = kernel.norm
        return LsfKernel(int(fiberId), center, start, values, norm=norm)

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

    def computeGaussianSigmaAtWavelength(self, fiberId, wavelength, method="fit", intrinsicWidth=0.0,
                                         **kwargs):
        """The rms (nm) of the Gaussian equivalent to the LSF at vacuum wavelength(s)

        As `computeGaussianSigma`, but with positions and ``intrinsicWidth``
        (and the result) in nm.

        Parameters
        ----------
        fiberId : `int` or array-like of `int`
            Fibre identifier(s).
        wavelength : `float` or array-like of `float`
            Vacuum wavelength(s) (nm); broadcast against ``fiberId``.
        method : `str`
            ``"fit"`` or ``"moment"``; see `computeGaussianSigma`.
        intrinsicWidth : `float`
            RMS (nm) of the features (``method="fit"`` only; converted to
            pixels with the median dispersion of the positions requested).
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

    def _window(self, center, halfWidth, clip=True):
        """The pixels of a kernel and its window

        Parameters
        ----------
        center : `float`
            Centre of the kernel (pixels).
        halfWidth : `float`
            Half-width of the window (pixels).
        clip : `bool`
            Clip the window to the spectrum's row?

        Returns
        -------
        start : `int`
            Index of the first pixel.
        window : `numpy.ndarray`
            Fraction of each pixel, from ``start``, within ``center +/- halfWidth``.
        """
        start = int(np.ceil(center - halfWidth - 0.5))
        stop = int(np.floor(center + halfWidth + 0.5)) + 1
        if clip:
            start = max(0, start)
            stop = min(self.length, stop)
        if stop <= start:
            raise ValueError(f"LSF at {center} with halfWidth {halfWidth} doesn't overlap the spectrum")
        window = np.clip(halfWidth + 0.5 - np.abs(np.arange(start, stop) - center), 0.0, 1.0)
        return start, window


def rowAtWavelength(wavelengthArray, wavelength):
    """Row(s) of a spectrum at vacuum wavelength(s)

    Parameters
    ----------
    wavelengthArray : `numpy.ndarray`
        Vacuum wavelength (nm) of each pixel of the spectrum (increasing).
    wavelength : `float` or `numpy.ndarray`
        Vacuum wavelength(s) (nm).

    Returns
    -------
    row : `float` or `numpy.ndarray`
        Row(s), NaN outside the spectrum.
    """
    wavelengthArray = np.asarray(wavelengthArray, dtype=float)
    good = np.isfinite(wavelengthArray)
    rows = np.arange(len(wavelengthArray), dtype=float)[good]
    lam = wavelengthArray[good]
    if lam[-1] < lam[0]:
        raise ValueError("The wavelengths must increase with row")
    result = np.interp(wavelength, lam, rows, left=np.nan, right=np.nan)
    return float(result) if np.ndim(result) == 0 else result


def dispersionAtRow(wavelengthArray, row):
    """Dispersion (nm/pixel) of a spectrum at row(s)"""
    wavelengthArray = np.asarray(wavelengthArray, dtype=float)
    good = np.isfinite(wavelengthArray)
    rows = np.arange(len(wavelengthArray), dtype=float)[good]
    return np.interp(row, rows, np.gradient(wavelengthArray[good], rows))


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


@dataclass(frozen=True)
class LsfParameter:
    """A parameter of an LSF model: its unit and meaning"""
    unit: u.UnitBase
    doc: str


PARAMETERS = dict(
    x=LsfParameter(u.pix, "Detector position across the dispersion"),
    y=LsfParameter(u.pix, "Detector position along the dispersion (the row)"),
    radius=LsfParameter(u.pix, "Radius of the fibre's disk"),
    rms=LsfParameter(u.pix, "RMS of the blur kernel"),
    S=LsfParameter(u.dimensionless_unscaled, "Skewness of the blur kernel"),
    K=LsfParameter(u.dimensionless_unscaled, "Excess kurtosis of the blur kernel"),
    lam5=LsfParameter(u.dimensionless_unscaled, "Fifth standardised cumulant of the blur kernel"),
    lam6=LsfParameter(u.dimensionless_unscaled, "Sixth standardised cumulant of the blur kernel"),
    lorentz=LsfParameter(u.pix, "FWHM of a convolving Lorentzian"),
    haloFrac=LsfParameter(u.dimensionless_unscaled, "Fraction of the light in the halo"),
    haloWidth=LsfParameter(u.pix, "RMS of the halo's Gaussian"),
    centroid=LsfParameter(u.pix, "Measured offset of the arc lines from the detectorMap (a diagnostic)"),
    sigma=LsfParameter(u.pix, "RMS width of the profile's core before pixel integration"),
)
"""Known parameters of LSF models"""


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
class ArmLsf(LsfModel):
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
        Each fibre's vacuum wavelength (nm) as a function of row, from the
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
    def parameters(self):
        """The model's parameters: `dict` [`str`, `LsfParameter`]

        Different models have different parameters (the blur's Edgeworth order
        decides whether there are ``lam5`` and ``lam6``, and the halo and the
        Lorentzian are optional), so code that loops over them needs to know
        which are present.
        """
        return {nn: PARAMETERS[nn] for nn in SHAPE_NAMES + ("centroid",) if nn in self.surfaces}

    def getParameters(self, fiberId, row):
        """The LSF's parameters at fibre(s) and row(s)

        Parameters
        ----------
        fiberId : `int` or array-like of `int`
            Fibre identifier(s).
        row : `float` or array-like of `float`, or `astropy.units.Quantity`
            Row(s) (pixels); broadcast against ``fiberId``.

        Returns
        -------
        params : `astropy.table.QTable`
            One row per position: ``fiberId``, ``row``, the detector position
            ``x`` and ``y``, each profile parameter (see ``parameters``), and
            ``sigma``, the rms width of the profile's core before pixel
            integration (no halo or Lorentzian); with units.
        """
        if isinstance(row, u.Quantity):
            row = row.to_value(u.pix)
        fiberId, row = np.broadcast_arrays(np.asarray(fiberId, dtype=int), np.asarray(row, dtype=float))
        values = self._parameterValues(fiberId.ravel(), row.ravel())
        table = QTable()
        table["fiberId"] = fiberId.ravel()
        table["row"] = row.ravel()*u.pix
        for name, value in values.items():
            unit = PARAMETERS[name].unit if name in PARAMETERS else u.pix
            table[name] = np.ravel(value)*unit
        if "centroid" in self.surfaces:
            table["centroid"] = np.ravel(self.getCentroidCorrection(fiberId.ravel(), row.ravel()))*u.pix
        return table

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
            Parameter name (see ``parameters``).
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

    def _parameterValues(self, fiberId, center):
        """The LSF's parameters at fibre(s) and row(s), as arrays (see `getParameters`)

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
        params = self._parameterValues(int(fiberId), float(center))
        cumulants = [float(params[nn]) for nn in self.cumulantNames]
        return diskProfile(pixels, float(center), float(params["radius"]), float(params["rms"]),
                           cumulants, self.extraction, float(params.get("lorentz", 0.0)),
                           float(params.get("haloFrac", 0.0)), float(params.get("haloWidth", np.nan)))

    def computeKernel(self, fiberId, center, halfWidth=None, normalize=True, clip=True):
        halfWidth = self.defaultHalfWidth if halfWidth is None else halfWidth
        start, window = self._window(center, halfWidth, clip)
        values = self._profile(fiberId, center, np.arange(start, start + len(window)))*window
        norm = float(np.sum(values))
        if normalize and norm > 0:
            values = values/norm
        return LsfKernel(int(fiberId), float(center), start, values, norm=norm)

    def computeKernelAtWavelength(self, fiberId, wavelength, halfWidth=None, normalize=True):
        if self.wavelength is None:
            raise RuntimeError("This model has no wavelengths: use the LSF bound to spectra")
        wavelengthArray = self.wavelength(int(fiberId), np.arange(self.length, dtype=float))
        center = rowAtWavelength(wavelengthArray, wavelength)
        if not np.isfinite(center):
            raise ValueError(f"wavelength {wavelength} is outside the spectrum of fiberId {fiberId}")
        kernel = self.computeKernel(fiberId, center, halfWidth, normalize)
        kernel.wavelength = wavelengthArray[kernel.slice]*u.nm
        return kernel

    def _sampleProfiles(self, fiberId, center, step, halfWidth, integrated=True):
        fiberId = np.asarray(fiberId, dtype=int)
        center = np.asarray(center, dtype=float)
        half = int(np.round(halfWidth/step))
        offsets = np.arange(-half, half + 1)*step
        params = self._parameterValues(fiberId, center)
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

    def computeGaussianSigmaAtWavelength(self, fiberId, wavelength, method="fit", intrinsicWidth=0.0,
                                         **kwargs):
        if self.wavelength is None:
            raise RuntimeError("This model has no wavelengths: use the LSF bound to spectra")
        fiberId, wavelength = np.broadcast_arrays(np.asarray(fiberId, dtype=int),
                                                  np.asarray(wavelength, dtype=float))
        rows = np.arange(self.length, dtype=float)
        center = np.empty(wavelength.shape)
        dispersion = np.empty(wavelength.shape)
        for ff in np.unique(fiberId):
            select = fiberId == ff
            lam = self.wavelength(int(ff), rows)
            center[select] = rowAtWavelength(lam, wavelength[select])
            dispersion[select] = dispersionAtRow(lam, center[select])
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

    def toDatamodel(self, fiberId=None, includeWavelength=True):
        """The datamodel representation, for writing to FITS

        Parameters
        ----------
        fiberId : array-like of `int`, optional
            Fibres to include (default: all).
        includeWavelength : `bool`
            Include the fibres' wavelengths (as for an ``armLsf`` file), or not
            (as when stored with spectra, which have their own)?

        Returns
        -------
        data : `pfs.datamodel.PfsArmLsfData`
            Datamodel representation.
        """
        fiberId = self.fiberId if fiberId is None else np.asarray(fiberId, dtype=int)
        self._checkFiberId(fiberId)
        surfaces = dict(self.surfaces)
        for name in ("iisRadiusScale", "iisHaloScale"):
            surface = getattr(self, name, None)
            if surface is not None:
                surfaces[name] = surface
        haveWavelength = includeWavelength and self.wavelength is not None
        engineering = np.asarray(getattr(self, "engineeringFiberId", []), dtype=int)
        groups = getattr(self, "fiberGroups", {})
        return PfsArmLsfData(
            arm=self.arm,
            spectrograph=int(self.spectrograph),
            length=int(self.length),
            kind="disk",
            fiberId=fiberId,
            surfaces={name: _surfaceToDatamodel(name, ss) for name, ss in surfaces.items()},
            extraction=self.extraction,
            blurOrder=int(self.blurOrder),
            centroidZeroPoint=float(self.centroidZeroPoint),
            xSplit=float(self.coords.xCenter),
            affine=self.coords.affine,
            affineMaxError=float(self.coords.maxError),
            isEngineering=np.isin(fiberId, engineering),
            fiberGroups={factor: {int(ff): table[int(ff)] for ff in fiberId if int(ff) in table}
                         for factor, table in groups.items()},
            iisExtraBlur=getattr(self, "iisExtraBlur", None),
            traceKnots=self.xCenter.knots,
            traceValues=self.xCenter.values[[self.xCenter._index[int(ff)] for ff in fiberId]],
            wavelengthKnots=self.wavelength.knots if haveWavelength else None,
            wavelengthValues=(self.wavelength.values[[self.wavelength._index[int(ff)] for ff in fiberId]]
                              if haveWavelength else None),
            metadata=self.metadata,
        )

    @classmethod
    def fromDatamodel(cls, data):
        """Construct from the datamodel representation

        Parameters
        ----------
        data : `pfs.datamodel.PfsArmLsfData`
            Datamodel representation, of kind ``disk``.

        Returns
        -------
        self : `ArmLsf`
            The LSF.
        """
        if data.kind != "disk":
            raise ValueError(f"Not a disk LSF: {data.kind}")
        surfaces = {name: _surfaceFromDatamodel(ss) for name, ss in data.surfaces.items()}
        iisRadiusScale = surfaces.pop("iisRadiusScale", None)
        iisHaloScale = surfaces.pop("iisHaloScale", None)
        wavelength = (FiberSpline(data.fiberId, data.wavelengthKnots, data.wavelengthValues)
                      if data.wavelengthKnots is not None else None)
        return cls(
            arm=data.arm,
            spectrograph=data.spectrograph,
            fiberId=data.fiberId,
            length=data.length,
            surfaces=surfaces,
            coords=DetectorCoords(data.xSplit, data.affine, data.affineMaxError),
            xCenter=FiberSpline(data.fiberId, data.traceKnots, data.traceValues),
            wavelength=wavelength,
            extraction=data.extraction,
            centroidZeroPoint=data.centroidZeroPoint,
            metadata=dict(data.metadata),
            engineeringFiberId=data.fiberId[data.isEngineering],
            iisRadiusScale=iisRadiusScale,
            iisHaloScale=iisHaloScale,
            iisExtraBlur=data.iisExtraBlur,
            fiberGroups=data.fiberGroups,
        )

    def writeFits(self, filename):
        """Write to a FITS file (the ``armLsf`` dataset)"""
        PfsLsf([self.toDatamodel()]).writeFits(filename)

    @classmethod
    def readFits(cls, filename):
        """Read from a FITS file (the ``armLsf`` dataset)"""
        return detectorLsfFromDatamodel(PfsLsf.readFits(filename).arms[0])

    def forSpectra(self, spectra, warnShift=1.0, maxShift=20.0):
        """The LSF of a set of spectra from this detector

        The LSF is bound to the spectra's wavelengths. Raises if the spectra
        aren't from this detector (arm, spectrograph, length, fibres), and
        checks that their wavelength solution is close to the one with which
        the LSF was measured: a shift by more than ``warnShift`` pixels gives a
        warning, and by more than ``maxShift`` pixels is an error.

        Parameters
        ----------
        spectra : `pfs.datamodel.PfsFiberArraySet`
            Spectra (e.g. a `PfsArm`).
        warnShift, maxShift : `float`
            Thresholds (pixels) for the shift of the wavelength solution.

        Returns
        -------
        lsf : `pfs.drp.stella.spectraLsf.BoundArmLsf`
            The LSF of the spectra.
        """
        from .spectraLsf import BoundArmLsf
        checkSpectra(self, spectra)
        if self.wavelength is not None:
            rows = np.arange(self.length, dtype=float)
            worst = 0.0
            for ff, wl in zip(spectra.fiberId, spectra.wavelength):
                model = self.wavelength(int(ff), rows)
                shift = (np.asarray(wl, dtype=float) - model)/dispersionAtRow(model, rows)
                worst = max(worst, float(np.nanmax(np.abs(shift))))
            if worst > maxShift:
                raise RuntimeError(f"The wavelengths of the spectra differ from the LSF's by up to "
                                   f"{worst:.1f} pixels (> {maxShift}): the wrong LSF?")
            if worst > warnShift:
                warnings.warn(f"The wavelengths of the spectra differ from the LSF's by up to {worst:.2f} "
                              "pixels", stacklevel=2)
        return BoundArmLsf(self, spectra.fiberId, spectra.wavelength)

    def _wavelengthRange(self):
        """Summary of the vacuum wavelength range, for ``__repr__``"""
        try:
            middle = self.fiberId[len(self.fiberId)//2]
            ends = self.wavelength(np.full(2, middle), np.array([0, self.length - 1]))
            return f", {np.min(ends):.1f}-{np.max(ends):.1f} nm (vacuum)"
        except Exception:
            return ""

    def __repr__(self):
        engineering = len(getattr(self, "engineeringFiberId", []))
        return (f"ArmLsf({self.arm}{self.spectrograph}: {len(self.fiberId)} fibres"
                + (f" ({engineering} engineering)" if engineering else "")
                + f", blur order {self.blurOrder}, {self.extraction} extraction"
                + self._wavelengthRange()
                + ";"
                + f" {', '.join(self.parameters)})")


class GaussianArmLsf(LsfModel):
    """A Gaussian LSF of fixed width for each fibre of a detector

    The placeholder used when there's no `ArmLsf`: the same as the
    `pfs.drp.stella.GaussianLsf` that ``reduceExposure`` writes into
    ``pfsArmLsf``, evaluated at the pixel centres.

    Parameters
    ----------
    arm : `str`
        Arm (``b``, ``r``, ``m``, ``n``).
    spectrograph : `int`
        Spectrograph module.
    fiberId : array-like of `int`
        Fibres covered.
    length : `int`
        Number of rows (pixels) in each spectrum.
    width : array-like of `float`
        Gaussian rms (pixels) for each fibre.
    """
    units = "pixel"

    def __init__(self, arm, spectrograph, fiberId, length, width):
        super().__init__(fiberId, length)
        self.arm = arm
        self.spectrograph = int(spectrograph)
        self.width = np.asarray(width, dtype=float)
        self._index = {int(ff): ii for ii, ff in enumerate(self.fiberId)}

    def getWidth(self, fiberId):
        """The rms (pixels) of a fibre's Gaussian"""
        return float(self.width[self._index[int(fiberId)]])

    def legacyLsf(self, fiberId):
        """The `pfs.drp.stella.GaussianLsf` of a fibre"""
        from .lsf import GaussianLsf
        return GaussianLsf(self.length, self.getWidth(fiberId))

    @property
    def parameters(self):
        """The model's parameters: `dict` [`str`, `LsfParameter`]"""
        return dict(sigma=PARAMETERS["sigma"])

    def computeKernel(self, fiberId, center, halfWidth=None, normalize=True, clip=True):
        start, values = self.legacyLsf(fiberId).computeWindow(float(center), clip=clip)
        norm = float(np.sum(values))
        if normalize and norm > 0:
            values = values/norm
        return LsfKernel(int(fiberId), float(center), int(start), np.asarray(values, dtype=float), norm=norm)

    def computeKernelAtWavelength(self, fiberId, wavelength, halfWidth=None, normalize=True):
        raise RuntimeError("A GaussianArmLsf has no wavelengths: use the LSF bound to spectra")

    def getParameters(self, fiberId, center):
        if isinstance(center, u.Quantity):
            center = center.to_value(u.pix)
        fiberId, center = np.broadcast_arrays(np.asarray(fiberId, dtype=int), np.asarray(center, dtype=float))
        table = QTable()
        table["fiberId"] = fiberId.ravel()
        table["row"] = center.ravel()*u.pix
        table["sigma"] = np.array([self.getWidth(ff) for ff in fiberId.ravel()])*u.pix
        return table

    def computeGaussianSigma(self, fiberId, center, method="fit", intrinsicWidth=0.0, **kwargs):
        fiberId, center = np.broadcast_arrays(np.asarray(fiberId, dtype=int), np.asarray(center, dtype=float))
        sigma = np.array([self.getWidth(ff) for ff in fiberId.ravel()]).reshape(fiberId.shape)
        return sigma if np.ndim(sigma) else float(sigma)

    def toDatamodel(self, fiberId=None, includeWavelength=True):
        """The datamodel representation, for writing to FITS

        Parameters are as for `ArmLsf.toDatamodel` (there are no wavelengths).
        """
        fiberId = self.fiberId if fiberId is None else np.asarray(fiberId, dtype=int)
        self._checkFiberId(fiberId)
        return PfsArmLsfData(arm=self.arm, spectrograph=self.spectrograph, length=self.length,
                             kind="gaussian", fiberId=fiberId,
                             gaussianWidth=np.array([self.getWidth(ff) for ff in fiberId]))

    @classmethod
    def fromDatamodel(cls, data):
        """Construct from the datamodel representation (of kind ``gaussian``)"""
        if data.kind != "gaussian":
            raise ValueError(f"Not a Gaussian LSF: {data.kind}")
        return cls(data.arm, data.spectrograph, data.fiberId, data.length, data.gaussianWidth)

    @classmethod
    def fromLegacy(cls, arm, spectrograph, lsfDict):
        """Construct from the deprecated `pfsArmLsf` (a `dict` of `GaussianLsf` by fiberId)

        Returns `None` if the LSFs aren't all `GaussianLsf`.
        """
        from .lsf import GaussianLsf
        if not lsfDict or not all(type(ll) is GaussianLsf for ll in lsfDict.values()):
            return None
        fiberId = np.array(sorted(lsfDict), dtype=int)
        lengths = {lsfDict[ff].length for ff in fiberId}
        if len(lengths) != 1:
            return None
        return cls(arm, spectrograph, fiberId, lengths.pop(), [lsfDict[ff].width for ff in fiberId])

    def forSpectra(self, spectra, warnShift=1.0, maxShift=20.0):
        """The LSF of a set of spectra from this detector (see `ArmLsf.forSpectra`)"""
        from .spectraLsf import BoundArmLsf
        checkSpectra(self, spectra)
        return BoundArmLsf(self, spectra.fiberId, spectra.wavelength)

    def __reduce__(self):
        return self.__class__, (self.arm, self.spectrograph, self.fiberId, self.length, self.width)

    def __repr__(self):
        return (f"GaussianArmLsf({self.arm}{self.spectrograph}: {len(self.fiberId)} fibres, "
                f"rms {np.median(self.width):.3f} pixels (median))")


def checkSpectra(detectorLsf, spectra):
    """Raise if spectra aren't from the detector of an LSF

    Parameters
    ----------
    detectorLsf : `ArmLsf` or `GaussianArmLsf`
        LSF of a detector.
    spectra : `pfs.datamodel.PfsFiberArraySet`
        Spectra (e.g. a `PfsArm`).
    """
    identity = getattr(spectra, "identity", None)
    if identity is not None:
        arm = getattr(identity, "arm", None)
        spectrograph = getattr(identity, "spectrograph", None)
        if arm is not None and arm != detectorLsf.arm:
            raise RuntimeError(f"Spectra are from arm {arm}, but the LSF is for {detectorLsf.arm}")
        if spectrograph is not None and int(spectrograph) != int(detectorLsf.spectrograph):
            raise RuntimeError(f"Spectra are from spectrograph {spectrograph}, but the LSF is for "
                               f"{detectorLsf.spectrograph}")
    if spectra.length != detectorLsf.length:
        raise RuntimeError(f"Spectra have {spectra.length} pixels, but the LSF {detectorLsf.length}")
    missing = np.setdiff1d(np.asarray(spectra.fiberId, dtype=int), detectorLsf.fiberId)
    if missing.size > 0:
        raise RuntimeError(f"The LSF lacks fibres {missing.tolist()}")


def detectorLsfFromDatamodel(data):
    """The LSF of a detector from its datamodel representation

    Parameters
    ----------
    data : `pfs.datamodel.PfsArmLsfData`
        Datamodel representation.

    Returns
    -------
    lsf : `ArmLsf` or `GaussianArmLsf`
        LSF of the detector.
    """
    if data.kind == "disk":
        return ArmLsf.fromDatamodel(data)
    if data.kind == "gaussian":
        return GaussianArmLsf.fromDatamodel(data)
    raise ValueError(f"Unknown kind of detector LSF: {data.kind}")


def _surfaceToDatamodel(name, surface):
    """A `SmoothSurface` as a `pfs.datamodel.PfsLsfSurface`"""
    domain = np.ravel(np.asarray(surface.domain, dtype=float))
    return PfsLsfSurface(
        name=name,
        degree=tuple(int(dd) for dd in surface.degree),
        domain=tuple(float(dd) for dd in domain),
        coeffs=np.asarray(surface.coeffs, dtype=float),
        offsets={factor: {str(label): (float(vv[0]), float(vv[1])) for label, vv in table.items()}
                 for factor, table in surface.offsets.items()},
        reference={factor: (None if label is None else str(label))
                   for factor, label in surface.reference.items()},
        rms=float(surface.rms),
        chi2=float(surface.chi2),
        ndof=int(surface.ndof),
        nUsed=int(surface.nUsed),
    )


def _surfaceFromDatamodel(data):
    """A `pfs.datamodel.PfsLsfSurface` as a `SmoothSurface`"""
    domain = tuple(float(dd) for dd in data.domain)
    return SmoothSurface(
        name=data.name,
        degree=tuple(data.degree),
        domain=(domain[:2], domain[2:]),
        coeffs=np.asarray(data.coeffs, dtype=float),
        offsets={factor: dict(table) for factor, table in data.offsets.items()},
        reference=dict(data.reference),
        rms=data.rms,
        chi2=data.chi2,
        ndof=data.ndof,
        nUsed=data.nUsed,
    )
