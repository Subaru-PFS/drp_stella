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


DEFAULT_MODEL = "screenResponse/pfsScreenResponse-2026-09-16.fits"
"""Screen response model within ``drp_pfs_data``."""

FORMAT_VERSION = 1
"""Version of the screen response file format."""


class ScreenResponseModel:
    """How the flat-field screen illuminates the fibers, relative to the sky.

    The model is ``S = 100 ln(twilight / quartz)`` as a polynomial surface over the focal
    plane, in the frame the screen sits in, with one set of coefficients per wavelength
    bin. The terms are the monomials of ``(x/radius, y/radius)`` up to ``degree``, constant
    excluded, in the order ``x, y, x^2, xy, y^2, ...``. Evaluating it interpolates the
    coefficients linearly in wavelength between the bin centres and holds the end values
    beyond them, and returns the factor ``exp(-S/100)``, the quartz over the twilight.

    The screen does not turn with the instrument rotator, so positions are carried into
    the frame the screen sits in before the surface is evaluated.

    Parameters
    ----------
    wavelengths : `numpy.ndarray`
        Bin centres in nm, of shape ``(nWavelength,)``, ascending.
    coefficients : `numpy.ndarray`
        Surface coefficients in units of ``100 ln``, of shape ``(nWavelength, nTerm)``.
    degree : `int`
        Highest total power of the surface; ``nTerm`` is ``(degree + 1)(degree + 2)/2 - 1``.
    radius : `float`
        Scale of the positions in mm.
    """

    def __init__(self, wavelengths, coefficients, degree, radius):
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.degree = int(degree)
        self.radius = float(radius)

        numTerms = (self.degree + 1)*(self.degree + 2)//2 - 1
        if self.coefficients.shape != (self.wavelengths.size, numTerms):
            raise RuntimeError(f"Coefficients {self.coefficients.shape} do not match "
                               f"{self.wavelengths.size} wavelengths and degree {self.degree}")
        if np.any(np.diff(self.wavelengths) <= 0):
            raise RuntimeError("Wavelengths are not ascending")

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
        primary.header["DEGREE"] = (self.degree, "polynomial degree of the surface")
        primary.header["RADIUS"] = (self.radius, "mm, scale of the positions")
        primary.header["NWAVE"] = (self.wavelengths.size, "number of wavelength bins")
        for key, value in (metadata or {}).items():
            primary.header[key] = value

        fits.HDUList([
            primary,
            fits.BinTableHDU.from_columns([
                fits.Column(name="WAVELENGTH", format="E", unit="nm",
                            array=self.wavelengths.astype("float32")),
                fits.Column(name="COEFFICIENTS", format="%dD" % self.coefficients.shape[1],
                            array=self.coefficients),
            ], name="SURFACE"),
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
            table = hdus["SURFACE"].data
            wavelengths = np.asarray(table["WAVELENGTH"], dtype=float)
            coefficients = np.asarray(table["COEFFICIENTS"], dtype=float)
            degree, radius = header["DEGREE"], header["RADIUS"]
        return cls(wavelengths, coefficients.reshape(wavelengths.size, -1), degree, radius)

    def __call__(self, x, y, insrot, wavelength):
        """Evaluate the screen response.

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
            The quartz over the twilight, of shape ``(nFiber, nWavelength)``.
        """
        rotated = rotateCoordinatesAroundCenter(np.vstack((x, y)), 0.0, 0.0,
                                                np.deg2rad(insrot))
        scaledX, scaledY = rotated/self.radius
        terms = np.array([scaledX**(order - power)*scaledY**power
                          for order in range(1, self.degree + 1)
                          for power in range(order + 1)])

        wavelength = np.asarray(wavelength, dtype=float)
        # np.interp holds the end values beyond the bin centres
        coefficients = np.stack([np.interp(wavelength, self.wavelengths, self.coefficients[:, index])
                                 for index in range(terms.shape[0])], axis=-1)
        surface = np.einsum("fwt,tf->fw", coefficients, terms)
        return np.exp(-surface/100.0)


class ScreenResponseConfig(Config):
    modelFile = Field(
        dtype=str,
        default=DEFAULT_MODEL,
        doc="Screen response model, as a path within the drp_pfs_data package.",
    )


class ScreenResponseTask(Task):
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
        if not self.isQuartz(metadata):
            self.log.debug("Not applying screen response correction since not a quartz lamp exposure")
            return
        insrot = metadata["INSROT"]
        if np.sum(pfsConfig.getSelection(targetType=~TargetType.ENGINEERING)) > 0:
            self.log.info("Applying screen response correction to quartz lamp spectra, INSROT=%f", insrot)
            self.apply(spectra, pfsConfig, insrot)

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

    def apply(self, spectra: PfsFiberArraySet, pfsConfig: PfsConfig, insrot: float):
        """Correct the spectra for the screen response.

        Parameters
        ----------
        spectra : `PfsFiberArraySet`
            The spectra to be corrected.
        pfsConfig : `PfsConfig`
            Fiber configuration.
        insrot : `float`
            The instrument rotator angle (degrees) of the exposure.
        """
        pfsConfig = pfsConfig.select(fiberId=spectra.fiberId)
        if not np.array_equal(pfsConfig.fiberId, spectra.fiberId):
            raise RuntimeError("FiberId mismatch")
        if not np.isfinite(insrot):
            raise RuntimeError("Rotator angle is not finite")

        select = pfsConfig.getSelection(targetType=~TargetType.ENGINEERING)
        select &= ~np.isnan(pfsConfig.pfiCenter).all(axis=1)
        if not np.any(select):
            return

        screen = self.model(
            pfsConfig.pfiCenter[select, 0],
            pfsConfig.pfiCenter[select, 1],
            insrot,
            spectra.wavelength[select],
        )

        # The screen imprints this pattern on the quartz, and measureFiberNorms divides the
        # quartz by the norm, so multiplying here is what carries the screen into the fiber
        # normalisations and cancels it out of the science spectra.
        spectra.norm[select] *= screen


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
