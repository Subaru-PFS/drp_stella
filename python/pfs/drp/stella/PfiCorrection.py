from __future__ import annotations

import os
from typing import TYPE_CHECKING

import astropy.io.fits
import numpy as np

from lsst.pex.config import Config, Field
from lsst.pipe.base import Task, Struct
from lsst.utils import getPackageDir

from pfs.datamodel import PfsFiberPolynomials

if TYPE_CHECKING:
    from pfs.datamodel import PfsConfig


class EdgeVignetting:
    """Edge vignetting correction model

    Parameters are written as a 1D image in the primary HDU.

    Parameters
    ----------
    parameters : `numpy.ndarray`
        Array of parameters.
    """
    def __init__(self, parameters: np.ndarray) -> None:
        if len(parameters) != 4:
            raise ValueError("Expected 4 parameters for edge vignetting model")
        self.parameters = parameters
        self.pfiRadius = parameters[0]  # mm
        self.powerLaw = parameters[1]  # Power-law index
        self.rScale = parameters[2]  # Scale for radial distance
        self.sScale = parameters[3]  # Scale for tangential distance

    @classmethod
    def readFits(cls, path: str) -> EdgeVignetting:
        """Read parameters from FITS file"""
        with astropy.io.fits.open(path) as fits:
            return cls(fits[0].data.astype(float))

    def writeFits(self, path: str) -> None:
        """Write parameters to FITS file"""
        hdu = astropy.io.fits.PrimaryHDU(data=self.parameters)
        hdu.writeto(path, overwrite=True)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate edge vignetting correction at positions (x, y)

        The model is a power-law fall-off from the center of the closest edge of
        the hexagonal PFI footprint.

        Parameters
        ----------
        x : `numpy.ndarray`
            X coordinates in mm.
        y : `numpy.ndarray`
            Y coordinates in mm.

        Returns
        -------
        correction : `numpy.ndarray`
            Edge vignetting correction factors at the input positions.
        """
        # Radial distance from the closest edge of the hexagon
        rr = self.pfiRadius - np.maximum.reduce([
            np.abs(x),
            np.abs(0.5*(np.sqrt(3)*y + x)),
            np.abs(0.5*(np.sqrt(3)*y - x)),
        ])

        # Tangential distance from the center of the closest edge of the hexagon
        ss = np.minimum.reduce([
            np.abs(y),
            np.abs(0.5*np.sqrt(3)*x - 0.5*y),
            np.abs(0.5*np.sqrt(3)*x + 0.5*y),
        ])

        rPart = np.minimum(rr/self.rScale, 0.0)
        sPart = np.minimum(ss/self.sScale, 0.0)
        distance = np.hypot(rPart, sPart)
        return 1.0 - np.clip(distance**-self.powerLaw, 0.0, 1.0)


class BlobVignetting:
    """Blob vignetting correction model

    Parameters are written as a binary table, with one row per blob.

    Parameters
    ----------
    parameters : `list` of `numpy.ndarray`
        List of parameter arrays, one per blob.
    """
    def __init__(self, parameters: list[np.ndarray]) -> None:
        for param in parameters:
            if len(param) != 6:
                raise ValueError("Each blob parameter array must have 6 elements")
        self.parameters = parameters

    @classmethod
    def readFits(cls, path: str) -> BlobVignetting:
        """Read parameters from FITS file"""
        with astropy.io.fits.open(path) as fits:
            hdu = fits[1]
            parameters = [row.astype(float) for row in hdu.data["parameters"]]
            return cls(parameters)

    def writeFits(self, path: str) -> None:
        """Write parameters to FITS file"""
        cols = [astropy.io.fits.Column(name="parameters", format=f"PD()", array=self.parameters)]
        hdu = astropy.io.fits.BinTableHDU.from_columns(cols)
        hdu.writeto(path, overwrite=True)

    def __len__(self) -> int:
        """Number of blobs in the model"""
        return len(self.parameters)

    def evaluate(self, index: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate a single blob vignetting correction at positions (x, y)

        Parameters
        ----------
        index : `int`
            Index of the blob to evaluate.
        x : `numpy.ndarray`
            X coordinates in mm.
        y : `numpy.ndarray`
            Y coordinates in mm.

        Returns
        -------
        correction : `numpy.ndarray`
            Blob vignetting correction factors at the input positions.
        """
        if index < 0 or index >= len(self.parameters):
            raise IndexError(f"Blob index {index} out of range")
        params = self.parameters[index]
        x0 = params[0]  # Center x position (mm)
        y0 = params[1]  # Center y position (mm)
        sigma1 = params[2]  # Gaussian sigma along major axis (mm)
        sigma2 = params[3]  # Gaussian sigma along minor axis (mm)
        theta = params[4]  # Rotation angle of major axis (radians)
        amplitude = params[5]  # Amplitude of the Gaussian

        dx = x - x0
        dy = y - y0

        aa = 0.5*(np.cos(theta)/sigma1)**2 + 0.5*(np.sin(theta)/sigma2)**2
        bb = -np.sin(2*theta)/(4*sigma1**2) + np.sin(2*theta)/(4*sigma2**2)
        cc = 0.5*(np.sin(theta)/sigma1)**2 + 0.5*(np.cos(theta)/sigma2)**2
        return amplitude * np.exp(-aa*dx**2 - 2*bb*dx*dy - cc*dy**2)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate blob vignetting correction at positions (x, y)

        The model is a 2D Gaussian.

        Parameters
        ----------
        x : `numpy.ndarray`
            X coordinates in mm.
        y : `numpy.ndarray`
            Y coordinates in mm.

        Returns
        -------
        correction : `numpy.ndarray`
            Blob vignetting correction factors at the input positions.
        """
        result = np.zeros_like(x)
        for ii in range(len(self)):
            result += self.evaluate(ii, x, y)
        return result


class PfiCorrectionConfig(Config):
    """Configuration for PfiCorrection"""

    edge = Field(
        dtype=str,
        default="pfi_edge.fits",
        optional=True,
        doc="Filename for edge vignetting model parameters"
    )
    blob = Field(
        dtype=float,
        default="pfi_blob.fits",
        optional=True,
        doc="Filename for for blob vignetting model parameters"
    )
    fiber = Field(
        dtype=str,
        default="pfi_fiber.fits",
        optional=True,
        doc="Filename for fiber throughput model parameters",
    )


class PfiCorrectionTask(Task):
    """Flux corrections for the Prime Focus Instrument (PFI).

    This class includes vignetting corrections and position-dependent throughput
    variations for each fiber.
    """
    ConfigClass = PfiCorrectionConfig
    _DefaultName = "pfiCorrection"

    config: PfiCorrectionConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._readData()

    def _readData(self) -> None:
        """Read correction models from files"""
        drpPfsData = getPackageDir("drp_pfs_data")

        def getAbsPath(path: str) -> str:
            """Return an absolute path, interpreting relative paths as relative to drp_pfs_data"""
            if not os.path.isabs(path):
                path = os.path.join(drpPfsData, "pfi", path)
            return path

        self.edge = EdgeVignetting.readFits(getAbsPath(self.config.edge))
        self.blob = BlobVignetting.readFits(getAbsPath(self.config.blob))
        self.fibers = PfsFiberPolynomials.readFits(getAbsPath(self.config.fiber))

    def run(self, pfsConfig: PfsConfig) -> Struct:
        """Compute PFI corrections

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`
            Configuration of fibers on the focal plane.

        Returns
        -------
        correction : `numpy.ndarray`
            Total correction factors for each fiber.
        edge : `numpy.ndarray`
            Edge vignetting correction factors for each fiber.
        blob : `numpy.ndarray`
            Blob vignetting correction factors for each fiber.
        fibers : `numpy.ndarray`
            Fiber throughput correction factors for each fiber.
        """
        xx = pfsConfig.pfiCenter[:, 0]
        yy = pfsConfig.pfiCenter[:, 1]
        edge = self.edge(xx, yy)
        blob = self.blob(xx, yy)
        fibers = np.array([self.fibers.evaluateSingle(*args) for args in zip(pfsConfig.fiberId, xx, yy)])
        correction = edge * blob * fibers
        return Struct(
            correction=correction,
            edge=edge,
            blob=blob,
            fibers=fibers,
        )
