from typing import Iterable

import numpy as np

from lsst.pex.config import Config, ListField
from lsst.pipe.base import Task
from lsst.daf.base import PropertyList

from pfs.datamodel import PfsConfig
from lsst.obs.pfs.utils import getLamps
from .SpectrumSetContinued import SpectrumSet


__all__ = ("ScreenResponseConfig", "ScreenResponseTask", "screenResponse")


class ScreenResponseConfig(Config):
    screenParams = ListField(
        dtype=float,
        default=[0, 0, -1.62131294e-07, -7.96517605e-05, -1.13541195e-04],
        doc="Flat-field screen response parameters",
    )


class ScreenResponseTask(Task):
    ConfigClass = ScreenResponseConfig
    _DefaultName = "screenResponse"

    def run(self, metadata: PropertyList, spectra: SpectrumSet, pfsConfig: PfsConfig):
        """Correct the spectra for the screen response.

        Parameters
        ----------
        metadata : `PropertyList`
            Metadata for the exposure.
        spectra : `SpectrumSet`
            The spectra to be corrected.
        pfsConfig : `PfsConfig`
            Fiber configuration.
        """
        if not self.isQuartz(metadata):
            self.log.debug("Not applying screen response correction since not a quartz lamp exposure")
            return
        insrot = metadata["INSROT"]
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

    def apply(self, spectra: SpectrumSet, pfsConfig: PfsConfig, insrot: float):
        """Correct the spectra for the screen response.

        Parameters
        ----------
        spectra : `SpectrumSet`
            The spectra to be corrected.
        pfsConfig : `PfsConfig`
            Fiber configuration.
        insrot : `float`
            The instrument rotator angle (degrees) of the exposure.
        """
        # Apply screen response correction
        pfsConfig = pfsConfig.select(fiberId=spectra.fiberId)
        if not np.array_equal(pfsConfig.fiberId, spectra.fiberId):
            raise RuntimeError("FiberId mismatch")
        if not np.isfinite(insrot):
            raise RuntimeError("Rotator angle is not finite")
        screen = screenResponse(
            pfsConfig.pfiCenter[:, 0],
            pfsConfig.pfiCenter[:, 1],
            insrot,
            self.config.screenParams,
        )
        # The "screen response" is the quartz flux divided by the twilight flux.
        # To get the twilight flux, we need to divide our quartz flux by the screen response.
        # We have the quartz flux as the "norm" of the pfsMerged.
        # By dividing the "norm" by the screen response, we get the twilight flux in the "norm".
        for spectrum, value in zip(spectra, screen):
            spectrum.norm /= value


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


def poly2dScreen(coords: Iterable[np.ndarray], a: float, b: float, c: float) -> np.ndarray:
    """Define a 2D polynomial function with cross terms.

    Parameters
    ----------
    coords : `tuple` of `numpy.ndarray`
        The x and y coordinates.
    a, b, c : `float`
        The coefficients of the 2D polynomial.

    Returns
    -------
    values : `numpy.ndarray`
        The values of the 2D polynomial at the given coordinates.
    """
    x, y = coords
    return a * x * y + b * x + c * y + 1


def screenResponse(x: np.ndarray, y: np.ndarray, insrot: float, params: np.ndarray) -> np.ndarray:
    """Model of the screen response

    Provides a model of the screen response as a function of position on the
    focal plane.

    Parameters
    ----------
    x, y : `numpy.ndarray`
        PFI coordinates.
    insrot : `float`
        The instrument rotator angle in degrees of the quartz exposure.
    params : `numpy.ndarray`
        The screen response parameters. The first two parameters are the
        coordinates of the center of the screen, and the remaining three
        parameters are the coefficients of the 2D polynomial.

    Returns
    -------
    values : `numpy.ndarray`
        The simulated screen response values.
    """
    if len(params) != 5:
        raise ValueError("The screen response model requires 5 parameters")

    x0 = params[0]
    y0 = params[1]

    coords = np.vstack((x, y))
    rotated = rotateCoordinatesAroundCenter(coords, x0, y0, np.deg2rad(insrot))
    # You have to remember that we computed this model by dividing twilight by the quartzes.
    # Providing the twilight is uniform, what you have left is actually the inverse of the screen response.
    return 1 / poly2dScreen(rotated, *params[2:])
