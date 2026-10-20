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

from .lsfProfile import diskProfile, diskSigma
from .smoothSurface import DetectorCoords, SmoothSurface  # noqa: F401 (docs)

__all__ = ("LsfKernel", "LsfKernelArray", "LsfModel", "ArmLsfModel", "FiberSpline")


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
SHAPE_NAMES = ("radius", "rms") + CUMULANT_NAMES + ("lorentz", "haloFrac", "haloWidth")   # profile parameters


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
        for name in ("radius", "rms") + tuple(self.cumulantNames):
            params[name] = self.surfaces[name](u, v)
        engineering = getattr(self, "engineeringFiberId", np.zeros(0, dtype=int))   # older models lack it
        scale = getattr(self, "iisRadiusScale", None)
        if scale is not None and len(engineering) > 0:
            isEngineering = np.isin(np.broadcast_to(np.asarray(fiberId, dtype=int), np.shape(x)), engineering)
            params["radius"] = np.where(isEngineering, params["radius"]*scale(u, v), params["radius"])
        if "lorentz" in self.surfaces:
            params["lorentz"] = np.clip(self.surfaces["lorentz"](u, v), 0.0, None)
        if "haloFrac" in self.surfaces:
            params["haloFrac"] = np.clip(self.surfaces["haloFrac"](u, v), 0.0, 1.0)
            params["haloWidth"] = np.clip(self.surfaces["haloWidth"](u, v), 1.0e-3, None)
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
        params = self.getParameters(int(fiberId), float(center))
        cumulants = [float(params[nn]) for nn in self.cumulantNames]
        edges = np.arange(-halfWidth, halfWidth + step/2, step)
        # pixel integration over a tiny "pixel" of width `step` approximates the density
        density = diskProfile(edges[:-1]/step + 0.5, 0.0, float(params["radius"])/step,
                              float(params["rms"])/step, cumulants, self.extraction, 0.0,
                              float(params.get("haloFrac", 0.0)), float(params.get("haloWidth", np.nan))/step)
        centers = 0.5*(edges[:-1] + edges[1:])
        lorentz = float(params.get("lorentz", 0.0))
        if lorentz > 0:                # convolve with the Lorentzian's mass in each cell
            cellEdges = np.append(centers - step/2, centers[-1] + step/2)
            kernel = np.diff(np.arctan(cellEdges/(0.5*lorentz)))/np.pi
            density = np.convolve(density, kernel, mode="same")
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
