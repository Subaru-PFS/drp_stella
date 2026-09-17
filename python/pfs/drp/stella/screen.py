import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

from astropy.io import fits

from lsst.pex.config import Config, Field
from lsst.pipe.base import Task
from lsst.daf.base import PropertyList
from lsst.utils import getPackageDir

from pfs.datamodel import PfsConfig, TargetType, PfsFiberArraySet
from lsst.obs.pfs.utils import getLamps


__all__ = ("ScreenResponseConfig", "ScreenResponseTask", "ScreenResponseModel",
           "rotateCoordinatesAroundCenter")


DEFAULT_MODEL = "screenResponse/pfsScreenResponse-2025-05-22.fits"
"""Screen response model within ``drp_pfs_data``."""

FORMAT_VERSION = 1
"""Version of the screen response file format."""


class ScreenResponseModel:
    """The multiplicative pattern the flat-field screen imprints on a quartz exposure.

    The pattern is held as a mean field over wavelength plus a few principal components,
    all on a regular grid over the focal plane, with one score per component per
    wavelength. Evaluating it interpolates the components bilinearly in position and the
    scores cubically in wavelength.

    The screen does not turn with the instrument rotator, so positions are carried into
    the frame the screen sits in before the grid is sampled.

    Parameters
    ----------
    xGrid, yGrid : `numpy.ndarray`
        Grid axes in mm, of shape ``(nx,)`` and ``(ny,)``, uniformly spaced and ascending.
    mean : `numpy.ndarray`
        Mean field, of shape ``(ny, nx)``.
    components : `numpy.ndarray`
        Principal components, of shape ``(nComponent, ny, nx)``.
    wavelengths : `numpy.ndarray`
        Wavelengths in nm at which the scores are tabulated, of shape ``(nWavelength,)``.
    scores : `numpy.ndarray`
        Component amplitudes, of shape ``(nWavelength, nComponent)``.
    """

    def __init__(self, xGrid, yGrid, mean, components, wavelengths, scores):
        self.xGrid = np.asarray(xGrid, dtype=float)
        self.yGrid = np.asarray(yGrid, dtype=float)
        self.mean = np.asarray(mean, dtype=float)
        self.components = np.asarray(components, dtype=float)
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        self.scores = np.asarray(scores, dtype=float)

        if self.mean.shape != (self.yGrid.size, self.xGrid.size):
            raise RuntimeError(f"Mean field {self.mean.shape} does not match the grid "
                               f"{(self.yGrid.size, self.xGrid.size)}")
        if self.components.shape[1:] != self.mean.shape:
            raise RuntimeError("Components do not match the mean field")
        if self.scores.shape != (self.wavelengths.size, self.components.shape[0]):
            raise RuntimeError("Scores do not match the wavelengths and components")

        # the scores are held at the end values outside the range they were measured over,
        # where a cubic would diverge
        self._interpolators = [
            interp1d(self.wavelengths, self.scores[:, index], kind="cubic",
                     fill_value=(self.scores[0, index], self.scores[-1, index]),
                     bounds_error=False)
            for index in range(self.components.shape[0])
        ]

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
        primary.header["NCOMP"] = (self.components.shape[0], "number of principal components")
        primary.header["NWAVE"] = (self.wavelengths.size, "number of wavelength samples")
        primary.header["WAVEMIN"] = (float(self.wavelengths.min()), "nm, lowest wavelength")
        primary.header["WAVEMAX"] = (float(self.wavelengths.max()), "nm, highest wavelength")
        for key, value in (metadata or {}).items():
            primary.header[key] = value

        def addGrid(hdu):
            """Describe the regular grid, so the coordinates need not be stored."""
            for axis, values in ((1, self.xGrid), (2, self.yGrid)):
                hdu.header[f"CTYPE{axis}"] = "X" if axis == 1 else "Y"
                hdu.header[f"CUNIT{axis}"] = "mm"
                hdu.header[f"CRPIX{axis}"] = 1.0
                hdu.header[f"CRVAL{axis}"] = float(values[0])
                hdu.header[f"CDELT{axis}"] = float(values[1] - values[0])
            return hdu

        fits.HDUList([
            primary,
            addGrid(fits.ImageHDU(self.mean.astype("float32"), name="MEAN")),
            addGrid(fits.ImageHDU(self.components.astype("float32"), name="COMPONENTS")),
            fits.BinTableHDU.from_columns([
                fits.Column(name="WAVELENGTH", format="E", unit="nm",
                            array=self.wavelengths.astype("float32")),
                fits.Column(name="SCORE", format="%dE" % self.components.shape[0],
                            array=self.scores.astype("float32")),
            ], name="SCORES"),
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
            mean = hdus["MEAN"].data
            components = hdus["COMPONENTS"].data
            header = hdus["MEAN"].header
            table = hdus["SCORES"].data
            wavelengths = np.asarray(table["WAVELENGTH"], dtype=float)
            scores = np.asarray(table["SCORE"], dtype=float)

        numY, numX = mean.shape
        xGrid = header["CRVAL1"] + header["CDELT1"] * np.arange(numX)
        yGrid = header["CRVAL2"] + header["CDELT2"] * np.arange(numY)
        return cls(xGrid, yGrid, mean, components, wavelengths, scores)

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
            The screen response, of shape ``(nFiber, nWavelength)``.
        """
        rotated = rotateCoordinatesAroundCenter(np.vstack((x, y)), 0.0, 0.0,
                                                np.deg2rad(insrot))
        indexX = (rotated[0] - self.xGrid[0]) / (self.xGrid[1] - self.xGrid[0])
        indexY = (rotated[1] - self.yGrid[0]) / (self.yGrid[1] - self.yGrid[0])
        coordinates = np.vstack([indexY, indexX])

        # 'nearest' clamps fibers that fall outside the grid to its edge
        fields = np.concatenate([self.components, self.mean[np.newaxis]], axis=0)
        sampled = np.array([map_coordinates(field, coordinates, order=1, mode="nearest",
                                            prefilter=False) for field in fields])
        components, mean = sampled[:-1], sampled[-1]

        wavelength = np.asarray(wavelength, dtype=float)
        scores = np.stack([one(wavelength) for one in self._interpolators], axis=-1)
        return np.einsum("fwc,cf->fw", scores, components) + mean[:, np.newaxis]


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
