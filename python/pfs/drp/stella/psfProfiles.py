from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from scipy.special import erf

from lsst.geom import Box2D, Point2D
from pfs.drp.stella.math import NormalizedPolynomial2D

__all__ = [
    "gaussian1D",
    "tophatConvGaussian1D",
    "moffat2D",
    "WingParams",
    "PsfParams",
    "evaluateParametricPsf",
    "ChebyshevSurface2D",
    "ParametricPsfModel",
]

_SQRT_2 = np.sqrt(2.0)


def gaussian1D(xx: np.ndarray, sigma: float) -> np.ndarray:
    """Evaluate a unit-integral 1D Gaussian

    Parameters
    ----------
    xx : `numpy.ndarray`
        Positions at which to evaluate, relative to the center.
    sigma : `float`
        Gaussian sigma.

    Returns
    -------
    values : `numpy.ndarray`
        Gaussian evaluated at ``xx``, normalized to unit integral.
    """
    return np.exp(-0.5 * (xx / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def tophatConvGaussian1D(xx: np.ndarray, width: float, sigma: float) -> np.ndarray:
    """Evaluate a unit-integral top-hat convolved with a Gaussian, in 1D

    This is the analytic convolution of a top-hat of the given ``width``
    (representing, e.g., the finite width of a fiber image in the spatial
    direction) with a Gaussian of the given ``sigma`` (representing, e.g.,
    optics blur and charge diffusion). It reduces to a pure Gaussian when
    ``width == 0``.

    Parameters
    ----------
    xx : `numpy.ndarray`
        Positions at which to evaluate, relative to the center.
    width : `float`
        Full width of the top-hat. May be zero.
    sigma : `float`
        Gaussian sigma of the blurring kernel.

    Returns
    -------
    values : `numpy.ndarray`
        Values evaluated at ``xx``, normalized to unit integral.
    """
    if width <= 0:
        return gaussian1D(xx, sigma)
    half = 0.5 * width
    upper = erf((xx + half) / (sigma * _SQRT_2))
    lower = erf((xx - half) / (sigma * _SQRT_2))
    return (upper - lower) / (2 * width)


def moffat2D(dx: np.ndarray, dy: np.ndarray, scaleX: float, scaleY: float, beta: float) -> np.ndarray:
    """Evaluate a unit-flux elliptical Moffat profile

    The Moffat profile provides a Gaussian-like core (at small radius) that
    transitions to a power-law tail (``beta`` sets the tail slope) suitable
    for representing PSF wings; ``scaleX``, ``scaleY`` set independent scale
    lengths along the two axes.

    Parameters
    ----------
    dx, dy : `numpy.ndarray`
        Positions at which to evaluate, relative to the center.
    scaleX, scaleY : `float`
        Scale lengths of the Moffat profile along x and y.
    beta : `float`
        Moffat exponent; must be greater than 1 for the profile to have a
        finite (unit) integral.

    Returns
    -------
    values : `numpy.ndarray`
        Values evaluated at ``(dx, dy)``, normalized to unit integral.
    """
    if beta <= 1:
        raise ValueError(f"Moffat beta must be > 1 for a finite integral (got {beta})")
    norm = (beta - 1) / (np.pi * scaleX * scaleY)
    base = 1.0 + (dx / scaleX) ** 2 + (dy / scaleY) ** 2
    return norm * base ** (-beta)


@dataclass
class WingParams:
    """Parameters for a single Moffat wing component

    Parameters
    ----------
    scaleX, scaleY : `float`
        Scale lengths of the Moffat profile along x (spatial) and y
        (dispersion).
    beta : `float`
        Moffat exponent.
    fraction : `float`
        Fraction of the total PSF flux contained in this wing component.
    """

    scaleX: float
    scaleY: float
    beta: float
    fraction: float


@dataclass
class PsfParams:
    """Concrete parametric-PSF parameters at a single detector position

    Parameters
    ----------
    sigmaX, sigmaY : `float`
        Gaussian blur sigma along x (spatial) and y (dispersion) for the
        core, applied after convolution with the fiber top-hat (in x only).
    tophatWidth : `float`
        Full width of the fiber top-hat in the spatial (x) direction. A
        monochromatic line has no analogous extent in the dispersion (y)
        direction, so no top-hat is applied there.
    wings : `list` of `WingParams`
        One or more Moffat wing components. The fractions of all wings must
        sum to less than 1; the remainder is the core fraction.
    """

    sigmaX: float
    sigmaY: float
    tophatWidth: float
    wings: List[WingParams] = field(default_factory=list)

    @property
    def coreFraction(self) -> float:
        """Fraction of the total PSF flux in the core"""
        return 1.0 - sum(ww.fraction for ww in self.wings)


def evaluateParametricPsf(dx: np.ndarray, dy: np.ndarray, params: PsfParams) -> np.ndarray:
    """Evaluate the unit-flux parametric PSF backbone (``P_param``)

    Parameters
    ----------
    dx, dy : `numpy.ndarray`
        Positions at which to evaluate, relative to the PSF center.
    params : `PsfParams`
        Parametric PSF parameters.

    Returns
    -------
    values : `numpy.ndarray`
        ``P_param`` evaluated at ``(dx, dy)``, normalized to unit integral.
    """
    core = tophatConvGaussian1D(dx, params.tophatWidth, params.sigmaX) * gaussian1D(dy, params.sigmaY)
    result = params.coreFraction * core
    for wing in params.wings:
        result = result + wing.fraction * moffat2D(dx, dy, wing.scaleX, wing.scaleY, wing.beta)
    return result


class ChebyshevSurface2D:
    """A single scalar quantity varying smoothly over the detector

    The quantity is represented as a low-order 2D polynomial
    (`pfs.drp.stella.math.NormalizedPolynomial2D`, total-degree truncation)
    in ``(fiberIndex, row)``, so it can be evaluated for any fiber/row and
    its coefficients can be packed into (and unpacked from) a flat vector
    for use with a non-linear optimizer.

    Parameters
    ----------
    order : `int`
        Total polynomial order in fiberIndex and row (``(order + 1)(order +
        2) / 2`` coefficients).
    fiberDomain, rowDomain : `tuple` of `float`, size 2
        Minimum and maximum values of fiberIndex and row, used to normalize
        the polynomial domain to [-1, 1].
    coeffs : `numpy.ndarray`, optional
        Initial coefficients. If not provided, all coefficients are zero
        except the constant term, which may be set via ``constant``.
    constant : `float`, optional
        Initial value for the constant coefficient.
    """

    def __init__(
        self,
        order: int,
        fiberDomain: Tuple[float, float],
        rowDomain: Tuple[float, float],
        coeffs: np.ndarray = None,
        constant: float = 0.0,
    ):
        self.order = order
        box = Box2D(Point2D(fiberDomain[0], rowDomain[0]), Point2D(fiberDomain[1], rowDomain[1]))
        self.model = NormalizedPolynomial2D(order, box)
        if coeffs is not None:
            self.setCoefficients(coeffs)
        elif constant != 0.0:
            self.setConstant(constant)

    def __call__(self, fiberIndex: np.ndarray, row: np.ndarray) -> np.ndarray:
        """Evaluate the surface at the given (fiberIndex, row) positions"""
        return self.model(fiberIndex, row)

    def getCoefficients(self) -> np.ndarray:
        """Return the coefficients as a flat array, in a fixed order"""
        return np.array(self.model.getParameters(), dtype=float)

    def setCoefficients(self, coeffs: np.ndarray) -> None:
        """Set the coefficients from a flat array, in the same order as ``getCoefficients``"""
        coeffs = np.asarray(coeffs, dtype=float)
        if len(coeffs) != self.numCoefficients:
            raise ValueError(f"Expected {self.numCoefficients} coefficients, got {len(coeffs)}")
        self.model.setParameters(coeffs)

    def setConstant(self, value: float) -> None:
        """Set the constant (position-independent) coefficient

        The constant term is always at index 0, regardless of order.
        """
        params = self.getCoefficients()
        params[0] = value
        self.setCoefficients(params)

    @property
    def numCoefficients(self) -> int:
        """Number of coefficients in this surface"""
        return self.model.getNParameters()


class ParametricPsfModel:
    """The parametric PSF backbone (``P_param``), varying over the detector

    Holds a `ChebyshevSurface2D` for each named scalar parameter of
    `PsfParams`, so that the full parametric PSF can be evaluated at any
    ``(fiberIndex, row)`` and its parameters can be fit as a single flat
    vector with a non-linear least-squares optimizer.

    Parameters
    ----------
    order : `int`
        Total polynomial order (in fiberIndex and row) for the core
        parameters (``sigmaX``, ``sigmaY``, ``tophatWidth``).
    numWings : `int`
        Number of Moffat wing components.
    fiberDomain, rowDomain : `tuple` of `float`, size 2
        Domain of fiberIndex and row, for normalizing the polynomial domain.
    wingOrder : `int`, optional
        Polynomial order for the wing parameters, if different from the core
        (defaults to the core order).
    """

    _CORE_PARAMS = ("sigmaX", "sigmaY", "tophatWidth")
    _WING_PARAMS = ("scaleX", "scaleY", "beta", "fraction")
    _CORE_BOUNDS = {"sigmaX": (1.0e-3, 20.0), "sigmaY": (1.0e-3, 20.0), "tophatWidth": (0.0, 20.0)}
    _WING_BOUNDS = {
        # Upper bounds keep the optimizer out of unphysical territory where a
        # very broad, high-beta Moffat becomes numerically degenerate with a
        # constant background over any finite stamp.
        "scaleX": (1.0e-3, 20.0),
        "scaleY": (1.0e-3, 20.0),
        "beta": (1.001, 10.0),
        "fraction": (0.0, 1.0),
    }

    def __init__(
        self,
        order: int,
        numWings: int,
        fiberDomain: Tuple[float, float],
        rowDomain: Tuple[float, float],
        wingOrder: int = None,
    ):
        self.numWings = numWings
        self.fiberDomain = fiberDomain
        self.rowDomain = rowDomain
        wingOrder = order if wingOrder is None else wingOrder
        self.order = order
        self.wingOrder = wingOrder

        self.surfaces = {}
        for name in self._CORE_PARAMS:
            self.surfaces[name] = ChebyshevSurface2D(order, fiberDomain, rowDomain)
        self.wingSurfaces: List[dict] = []
        for _ in range(numWings):
            wing = {
                name: ChebyshevSurface2D(wingOrder, fiberDomain, rowDomain) for name in self._WING_PARAMS
            }
            self.wingSurfaces.append(wing)

    def setInitialGuess(
        self,
        sigmaX: float,
        sigmaY: float,
        tophatWidth: float,
        wings: Sequence[WingParams] = (),
    ) -> None:
        """Set constant (position-independent) initial values for all parameters

        Parameters
        ----------
        sigmaX, sigmaY, tophatWidth : `float`
            Initial core parameters.
        wings : sequence of `WingParams`
            Initial wing parameters; must have length ``self.numWings``.
        """
        if len(wings) != self.numWings:
            raise ValueError(f"Expected {self.numWings} wings, got {len(wings)}")
        self.surfaces["sigmaX"].setConstant(sigmaX)
        self.surfaces["sigmaY"].setConstant(sigmaY)
        self.surfaces["tophatWidth"].setConstant(tophatWidth)
        for wing, params in zip(self.wingSurfaces, wings):
            wing["scaleX"].setConstant(params.scaleX)
            wing["scaleY"].setConstant(params.scaleY)
            wing["beta"].setConstant(params.beta)
            wing["fraction"].setConstant(params.fraction)

    def getParamsAt(self, fiberIndex: float, row: float) -> PsfParams:
        """Evaluate the concrete `PsfParams` at a single detector position

        Parameters
        ----------
        fiberIndex : `float`
            Index of the fiber (not fiberId) within the detector.
        row : `float`
            Row (dispersion-direction pixel) on the detector.

        Returns
        -------
        params : `PsfParams`
            Concrete parameters at this position.
        """
        core = {name: float(self.surfaces[name](fiberIndex, row)) for name in self._CORE_PARAMS}
        wings = [
            WingParams(**{name: float(wing[name](fiberIndex, row)) for name in self._WING_PARAMS})
            for wing in self.wingSurfaces
        ]
        return PsfParams(
            sigmaX=core["sigmaX"], sigmaY=core["sigmaY"], tophatWidth=core["tophatWidth"], wings=wings
        )

    def evaluate(self, dx: np.ndarray, dy: np.ndarray, fiberIndex: float, row: float) -> np.ndarray:
        """Evaluate ``P_param`` at a detector position

        Parameters
        ----------
        dx, dy : `numpy.ndarray`
            Positions at which to evaluate, relative to the PSF center.
        fiberIndex : `float`
            Index of the fiber (not fiberId) within the detector.
        row : `float`
            Row (dispersion-direction pixel) on the detector.

        Returns
        -------
        values : `numpy.ndarray`
            ``P_param`` evaluated at ``(dx, dy)``, normalized to unit
            integral.
        """
        return evaluateParametricPsf(dx, dy, self.getParamsAt(fiberIndex, row))

    def _allSurfaces(self) -> List[ChebyshevSurface2D]:
        """Return all surfaces, in the fixed order used for (un)packing"""
        surfaces = [self.surfaces[name] for name in self._CORE_PARAMS]
        for wing in self.wingSurfaces:
            surfaces.extend(wing[name] for name in self._WING_PARAMS)
        return surfaces

    def _allBounds(self) -> List[Tuple[float, float]]:
        """Return the physical (lower, upper) bound for each surface in `_allSurfaces` order"""
        bounds = [self._CORE_BOUNDS[name] for name in self._CORE_PARAMS]
        for _ in self.wingSurfaces:
            bounds.extend(self._WING_BOUNDS[name] for name in self._WING_PARAMS)
        return bounds

    def getParameterBounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) bound arrays matching `getParameterVector`'s layout

        A physical bound (e.g. sigma > 0) is meaningful only for a surface's
        constant (detector-averaged) term; its higher-order coefficients
        describe *variation* across the detector and are left unbounded.
        The constant term is always at index 0 (see
        `ChebyshevSurface2D.setConstant`), so the physical bound is applied
        there regardless of the surface's polynomial order.

        Returns
        -------
        lower, upper : `numpy.ndarray`
            Bound arrays, suitable for `scipy.optimize.least_squares`'s
            ``bounds`` argument.
        """
        lower = []
        upper = []
        for surface, (lo, hi) in zip(self._allSurfaces(), self._allBounds()):
            numCoefficients = surface.numCoefficients
            lower.append(-np.inf if lo is None else lo)
            upper.append(np.inf if hi is None else hi)
            lower.extend([-np.inf] * (numCoefficients - 1))
            upper.extend([np.inf] * (numCoefficients - 1))
        return np.array(lower), np.array(upper)

    def getParameterVector(self) -> np.ndarray:
        """Pack all Chebyshev coefficients into a single flat vector"""
        return np.concatenate([ss.getCoefficients() for ss in self._allSurfaces()])

    def setParameterVector(self, vector: np.ndarray) -> None:
        """Unpack a flat vector (as produced by `getParameterVector`) into the surfaces"""
        vector = np.asarray(vector, dtype=float)
        start = 0
        for ss in self._allSurfaces():
            stop = start + ss.numCoefficients
            ss.setCoefficients(vector[start:stop])
            start = stop
        if start != len(vector):
            raise ValueError(f"Parameter vector length {len(vector)} does not match expected {start}")
