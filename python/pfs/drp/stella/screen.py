import os

import numpy as np

from astropy.io import fits

from lsst.pex.config import Config, Field
from lsst.pipe.base import Task
from lsst.daf.base import PropertyList
from lsst.utils import getPackageDir

from pfs.datamodel import PfsConfig, TargetType, PfsFiberArraySet
from lsst.obs.pfs.utils import getLamps


__all__ = ("ScreenResponseConfig", "ScreenResponseTask", "ScreenResponseModel",
           "rotateCoordinatesAroundCenter")


DEFAULT_MODEL = "screenResponse/pfsScreenResponse-1.1.fits"
"""Screen response model within ``drp_pfs_data``."""

FORMAT_VERSION = 2
"""Version of the screen response file format."""


class ScreenResponseModel:
    """How the flat-field screen and the telescope illuminate the fibers.

    The model describes the smooth part of ``100 ln(twilight / quartz)`` per fiber as

    ``T(R(theta) p) - G(R(phi) p)``

    where ``p`` is the fiber's PFI position, ``R(angle)`` the rotation by the instrument
    rotator angle into the telescope frame (in which the telescope and the screen stay
    fixed), ``theta`` and ``phi`` the rotator angles of the twilight and of the quartz,
    and, per wavelength bin:

    * ``T``, the telescope: a polynomial in the scaled position ``R(theta) p / radius``,
      on the monomials ``x^(n - m) y^m`` of total order ``n`` from ``lowestOrder`` to
      ``degree`` in the order ``n = lowestOrder, ..., m = 0, ..., n``;
    * ``G``, the screen: bilinear interpolation on a square grid of nodes, ``spacing`` mm
      apart and ordered row-major from ``(-halfSize, -halfSize)``, less its mean over the
      telescope-frame rotations ``rotations``.

    Each is evaluated at the bin centres and interpolated linearly in wavelength between
    them, holding the end values beyond them.

    Parameters
    ----------
    wavelengths : `numpy.ndarray`
        Bin centres in nm, of shape ``(nBin,)``, ascending.
    telescope : `numpy.ndarray`
        Coefficients of ``T`` in units of ``100 ln``, of shape ``(nBin, nTerm)``.
    degree, lowestOrder : `int`
        Highest and lowest total order of the monomials of ``T``.
    radius : `float`
        Scale of the positions in ``T``, in mm.
    screen : `numpy.ndarray`
        Grid node values of ``G`` in units of ``100 ln``, of shape ``(nBin, nSide**2)``.
    spacing, halfSize : `float`
        Grid spacing and half-size of ``G``, in mm.
    rotations : `numpy.ndarray`
        Angles in degrees over which the mean is taken out of ``G``.
    """

    def __init__(self, wavelengths, telescope, degree, lowestOrder, radius, screen, spacing, halfSize,
                 rotations):
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.telescope = np.asarray(telescope, dtype=float)
        self.degree = int(degree)
        self.lowestOrder = int(lowestOrder)
        self.radius = float(radius)
        self.screen = np.asarray(screen, dtype=float)
        self.spacing = float(spacing)
        self.halfSize = float(halfSize)
        self.rotations = np.asarray(rotations, dtype=float)

        numBins = self.wavelengths.size
        if np.any(np.diff(self.wavelengths) <= 0):
            raise RuntimeError("Wavelengths are not ascending")
        numTerms = len(self._orders())
        if self.telescope.shape != (numBins, numTerms):
            raise RuntimeError(f"Telescope coefficients {self.telescope.shape} do not match {numBins} "
                               f"wavelengths and orders {self.lowestOrder} to {self.degree}")
        if self.screen.shape != (numBins, self.numSide**2):
            raise RuntimeError(f"Screen nodes {self.screen.shape} do not match {numBins} wavelengths "
                               f"and a grid of {self.numSide}x{self.numSide}")

    @property
    def numSide(self):
        """Number of grid nodes along each side of ``G``."""
        return int(np.ceil(2*self.halfSize/self.spacing)) + 1

    def _orders(self):
        """(order, power of y) of each monomial of ``T``."""
        return [(order, power) for order in range(self.lowestOrder, self.degree + 1)
                for power in range(order + 1)]

    def writeFits(self, path, metadata=None):
        """Write the model.

        Parameters
        ----------
        path : `str`
            FITS file to write.
        metadata : `dict`, optional
            Additional header keywords recording where the model came from.
        """
        primary = fits.PrimaryHDU()
        primary.header["SCRNVER"] = (FORMAT_VERSION, "screen model format version")
        primary.header["DEGREE"] = (self.degree, "highest total order of the telescope surface")
        primary.header["LOWORDER"] = (self.lowestOrder, "lowest total order of the telescope surface")
        primary.header["RADIUS"] = (self.radius, "mm, scale of the positions")
        primary.header["SPACING"] = (self.spacing, "mm, screen grid spacing")
        primary.header["HALFSIZE"] = (self.halfSize, "mm, screen grid half-size")
        primary.header["NWAVE"] = (self.wavelengths.size, "number of wavelength bins")
        for key, value in (metadata or {}).items():
            primary.header[key] = value

        fits.HDUList([
            primary,
            fits.BinTableHDU.from_columns([
                fits.Column(name="WAVELENGTH", format="E", unit="nm",
                            array=self.wavelengths.astype("float32")),
                fits.Column(name="TELESCOPE", format="%dD" % self.telescope.shape[1], array=self.telescope),
                fits.Column(name="SCREEN", format="%dD" % self.screen.shape[1], array=self.screen),
            ], name="BINS"),
            fits.BinTableHDU.from_columns([
                fits.Column(name="ANGLE", format="D", unit="deg", array=self.rotations),
            ], name="ROTATIONS"),
        ]).writeto(path, overwrite=True)

    @classmethod
    def readFits(cls, path):
        """Read a screen response model.

        Parameters
        ----------
        path : `str`
            FITS file to read.

        Returns
        -------
        self : `ScreenResponseModel`
            The model.
        """
        with fits.open(path) as hdus:
            header = hdus[0].header
            if header.get("SCRNVER") != FORMAT_VERSION:
                raise RuntimeError(f"Unsupported screen response format {header.get('SCRNVER')} "
                                   f"in {path}")
            bins = hdus["BINS"].data
            numBins = len(bins)
            return cls(
                wavelengths=np.asarray(bins["WAVELENGTH"], dtype=float),
                telescope=np.asarray(bins["TELESCOPE"], dtype=float).reshape(numBins, -1),
                degree=header["DEGREE"],
                lowestOrder=header["LOWORDER"],
                radius=header["RADIUS"],
                screen=np.asarray(bins["SCREEN"], dtype=float).reshape(numBins, -1),
                spacing=header["SPACING"],
                halfSize=header["HALFSIZE"],
                rotations=np.asarray(hdus["ROTATIONS"].data["ANGLE"], dtype=float),
            )

    def _interpolate(self, values, wavelength):
        """Per-bin values of shape ``(nFiber, nBin)`` at the wavelengths of shape
        ``(nFiber, nWavelength)``."""
        wavelength = np.asarray(wavelength, dtype=float)
        # np.interp holds the end values beyond the bin centres
        return np.array([np.interp(ww, self.wavelengths, vv) for ww, vv in zip(wavelength, values)])

    def _telescopeBins(self, x, y, insrot):
        """``T`` at the bin centres, of shape ``(nFiber, nBin)``."""
        u, v = rotateCoordinatesAroundCenter(np.vstack((x, y)), 0.0, 0.0, np.deg2rad(insrot))/self.radius
        terms = np.array([u**(order - power)*v**power for order, power in self._orders()])
        return terms.T @ self.telescope.T

    def _gridWeights(self, u, v):
        """Bilinear weights of telescope-frame positions on the screen grid, of shape
        ``(nPoint, nSide**2)``."""
        numSide = self.numSide
        gu, gv = (u + self.halfSize)/self.spacing, (v + self.halfSize)/self.spacing
        i0 = np.clip(np.floor(gu).astype(int), 0, numSide - 2)
        j0 = np.clip(np.floor(gv).astype(int), 0, numSide - 2)
        fu, fv = gu - i0, gv - j0
        weights = np.zeros((u.size, numSide**2))
        rows = np.arange(u.size)
        for di, dj, ww in ((0, 0, (1 - fu)*(1 - fv)), (1, 0, fu*(1 - fv)),
                           (0, 1, (1 - fu)*fv), (1, 1, fu*fv)):
            np.add.at(weights, (rows, (j0 + dj)*numSide + i0 + di), ww)
        return weights

    def _screenBins(self, x, y, insrot):
        """``G`` at the bin centres, of shape ``(nFiber, nBin)``."""
        u, v = rotateCoordinatesAroundCenter(np.vstack((x, y)), 0.0, 0.0, np.deg2rad(insrot))
        here = self._gridWeights(u, v) @ self.screen.T
        mean = np.mean([self._gridWeights(*rotateCoordinatesAroundCenter(np.vstack((u, v)), 0.0, 0.0,
                                                                        np.deg2rad(angle))) @ self.screen.T
                        for angle in self.rotations], axis=0)
        return here - mean

    def telescopeResponse(self, x, y, insrot, wavelength):
        """The telescope term ``T`` of an exposure through the telescope.

        Parameters
        ----------
        x, y : `numpy.ndarray`
            PFI coordinates in mm, of shape ``(nFiber,)``.
        insrot : `float`
            The instrument rotator angle in degrees of the exposure.
        wavelength : `numpy.ndarray`
            Wavelengths in nm, of shape ``(nFiber, nWavelength)``.

        Returns
        -------
        values : `numpy.ndarray`
            ``T`` in units of ``100 ln``, of shape ``(nFiber, nWavelength)``.
        """
        return self._interpolate(self._telescopeBins(x, y, insrot), wavelength)

    def quartzResponse(self, x, y, insrot, wavelength):
        """The quartz's term ``-G`` of a quartz exposure.

        Parameters
        ----------
        x, y : `numpy.ndarray`
            PFI coordinates in mm, of shape ``(nFiber,)``.
        insrot : `float`
            The instrument rotator angle in degrees of the exposure.
        wavelength : `numpy.ndarray`
            Wavelengths in nm, of shape ``(nFiber, nWavelength)``.

        Returns
        -------
        values : `numpy.ndarray`
            ``-G`` in units of ``100 ln``, of shape ``(nFiber, nWavelength)``.
        """
        return self._interpolate(-self._screenBins(x, y, insrot), wavelength)


class ScreenResponseConfig(Config):
    modelFile = Field(
        dtype=str,
        default=DEFAULT_MODEL,
        doc="Screen response model, as a path within the drp_pfs_data package.",
    )


class ScreenResponseTask(Task):
    """Correct the fiber normalisations for how the screen and the telescope light the fibers.

    A quartz's norm takes the quartz's term of the `ScreenResponseModel` at its rotator
    angle, so that the fiber normalisations measured from it no longer carry the screen; an
    exposure through the telescope (no lamp) takes the telescope term at its own rotator
    angle. Other lamp exposures are left alone.
    """
    ConfigClass = ScreenResponseConfig
    _DefaultName = "screenResponse"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = None

    @property
    def model(self):
        """The screen response model (`ScreenResponseModel`), read on first use."""
        if self._model is None:
            path = os.path.join(getPackageDir("drp_pfs_data"), self.config.modelFile)
            self.log.info("Reading screen response model from %s", path)
            self._model = ScreenResponseModel.readFits(path)
        return self._model

    def run(self, metadata: PropertyList, spectra: PfsFiberArraySet, pfsConfig: PfsConfig):
        """Correct the spectra for the screen response.

        Parameters
        ----------
        metadata : `PropertyList`
            Metadata for the exposure.
        spectra : `PfsFiberArraySet`
            The spectra to be corrected.
        pfsConfig : `PfsConfig`
            Fiber configuration.
        """
        lamps = getLamps(metadata)
        if lamps and not self.isQuartz(metadata):
            self.log.debug("Not applying screen response correction to a lamp exposure other than quartz")
            return
        if np.sum(pfsConfig.getSelection(targetType=~TargetType.ENGINEERING)) == 0:
            return
        insrot = metadata["INSROT"]
        if lamps:
            self.log.info("Applying screen response correction to quartz lamp spectra, INSROT=%f", insrot)
            self.applyQuartz(spectra, pfsConfig, insrot)
        else:
            self.log.info("Applying telescope response correction to spectra, INSROT=%f", insrot)
            self.applyTelescope(spectra, pfsConfig, insrot)

    def isQuartz(self, metadata: PropertyList) -> bool:
        """Return whether the exposure is a quartz lamp exposure

        Parameters
        ----------
        metadata : `PropertyList`
            Metadata for the exposure.

        Returns
        -------
        isQuartz : `bool`
            Whether the exposure is a quartz lamp exposure.
        """
        lamps = getLamps(metadata)
        return bool(lamps & set(("Quartz", "Quartz_eng")))

    def _select(self, spectra: PfsFiberArraySet, pfsConfig: PfsConfig, insrot: float):
        """The non-engineering fibers with a position, and the configuration matching the
        spectra."""
        pfsConfig = pfsConfig.select(fiberId=spectra.fiberId)
        if not np.array_equal(pfsConfig.fiberId, spectra.fiberId):
            raise RuntimeError("FiberId mismatch")
        if not np.isfinite(insrot):
            raise RuntimeError("Rotator angle is not finite")
        select = pfsConfig.getSelection(targetType=~TargetType.ENGINEERING)
        select &= ~np.isnan(pfsConfig.pfiCenter).all(axis=1)
        return select, pfsConfig

    def applyQuartz(self, spectra: PfsFiberArraySet, pfsConfig: PfsConfig, insrot: float):
        """Take the quartz's term into the norm of quartz spectra.

        Parameters
        ----------
        spectra : `PfsFiberArraySet`
            The quartz spectra to be corrected.
        pfsConfig : `PfsConfig`
            Fiber configuration.
        insrot : `float`
            The instrument rotator angle (degrees) of the exposure.
        """
        select, pfsConfig = self._select(spectra, pfsConfig, insrot)
        if not np.any(select):
            return
        terms = self.model.quartzResponse(
            pfsConfig.pfiCenter[select, 0],
            pfsConfig.pfiCenter[select, 1],
            insrot,
            spectra.wavelength[select],
        )
        # measureFiberNorms divides the quartz by the norm, so this carries the quartz's term,
        # with the sign it has in ln(twilight / quartz), into the fiber normalisations.
        spectra.norm[select] *= np.exp(-terms/100.0)

    def applyTelescope(self, spectra: PfsFiberArraySet, pfsConfig: PfsConfig, insrot: float):
        """Take the telescope term into the norm of spectra taken through the telescope.

        Parameters
        ----------
        spectra : `PfsFiberArraySet`
            The spectra to be corrected.
        pfsConfig : `PfsConfig`
            Fiber configuration.
        insrot : `float`
            The instrument rotator angle (degrees) of the exposure.
        """
        select, pfsConfig = self._select(spectra, pfsConfig, insrot)
        if not np.any(select):
            return
        terms = self.model.telescopeResponse(
            pfsConfig.pfiCenter[select, 0],
            pfsConfig.pfiCenter[select, 1],
            insrot,
            spectra.wavelength[select],
        )
        spectra.norm[select] *= np.exp(terms/100.0)


def rotationMatrix(theta: float) -> np.ndarray:
    """Compute a 2D rotation matrix for a given angle.

    Parameters
    ----------
    theta : `float`
        Rotation angle in radians.

    Returns
    -------
    matrix : `numpy.ndarray`
        A 2x2 rotation matrix.
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rotateCoordinatesAroundCenter(x: np.ndarray, x0: float, y0: float, theta: float) -> np.ndarray:
    """Rotate the given coordinates around the specified center.

    Parameters
    ----------
    x : `numpy.ndarray`
        The coordinates to be rotated, in the format
        ``[[x1, x2, ..., xn], [y1, y2, ..., yn]]``.
    x0, y0 : `float`
        The x and y coordinates of the center around which the rotation is
        performed.
    theta : `float`
        The rotation angle in radians.

    Returns
    -------
    xRot : `numpy.ndarray`
        The rotated coordinates in the same format as the input `x`.
    """
    rotation = rotationMatrix(theta)
    center = np.array(([x0], [y0]))
    xCentered = x - center
    xRot = np.matmul(rotation, xCentered)
    xRot += center
    return xRot
