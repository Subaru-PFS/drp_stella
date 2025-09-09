from __future__ import annotations

import os
from typing import TYPE_CHECKING

import astropy.io.fits
import numpy as np

from lsst.pex.config import Config, Field
from lsst.pipe.base import Task, Struct
from lsst.utils import getPackageDir

from .focalPlaneFunction import ConstantPerFiber, FiberPolynomials

if TYPE_CHECKING:
    from pfs.datamodel import PfsConfig
    from .datamodel.drp import PfsArm


class EdgeVignettingConfig(Config):
    """Configuration for EdgeVignetting"""

    pfiRadius = Field(
        dtype=float,
        default=200.0,
        doc="Radius of the PFI hexagon (mm)",
    )
    powerLaw = Field(
        dtype=float,
        default=2.08321709,
        doc="Power-law index for edge vignetting fall-off",
    )
    rScale = Field(
        dtype=float,
        default=0.54491177,
        doc="Scale for radial distance from edge (mm)",
    )
    sScale = Field(
        dtype=float,
        default=15.74269352,
        doc="Scale for tangential distance from edge (mm)",
    )


class EdgeVignetting:
    """Edge vignetting correction model

    Parameters are written as a 1D image in the primary HDU.

    Parameters
    ----------
    parameters : `numpy.ndarray`
        Array of parameters.
    pfiRadius : `float`, optional
        Radius of the PFI hexagon in mm.
    """
    def __init__(self, parameters: np.ndarray, pfiRadius: float = 200.0) -> None:
        if len(parameters) != 3:
            raise ValueError("Expected 3 parameters for edge vignetting model")
        self.parameters = parameters
        self.pfiRadius = pfiRadius

        self.powerLaw = parameters[0]  # Power-law index
        self.rScale = parameters[1]  # Scale for radial distance
        self.sScale = parameters[2]  # Scale for tangential distance

    @classmethod
    def fromConfig(cls, config: EdgeVignettingConfig) -> EdgeVignetting:
        """Create from configuration"""
        parameters = np.array([config.powerLaw, config.rScale, config.sScale], dtype=float)
        return cls(parameters, config.pfiRadius, )

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
            np.abs(0.5*(np.sqrt(3)*x - y)),
            np.abs(0.5*(np.sqrt(3)*x + y)),
        ])

        rPart = np.clip(rr/self.rScale, 0.0, None)
        sPart = np.clip(ss/self.sScale, 0.0, None)
        distance = np.hypot(rPart, sPart)
        value = -np.clip(distance**-self.powerLaw, 0.0, 1.0)
        return value


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
        cols = [astropy.io.fits.Column(name="parameters", format="PD()", array=self.parameters)]
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
        default=None,
        optional=True,
        doc="Filename for edge vignetting model parameters"
    )
    blob = Field(
        dtype=str,
        default="pfi_blob.fits",
        optional=True,
        doc="Filename for for blob vignetting model parameters"
    )
    fibers = Field(
        dtype=str,
        default="pfi_fibers.fits",
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

        if self.config.edge is None:
            edgeConfig = EdgeVignettingConfig()
            self.edge = EdgeVignetting.fromConfig(edgeConfig)
        else:
            self.edge = EdgeVignetting.readFits(getAbsPath(self.config.edge))

        self.blob = BlobVignetting.readFits(getAbsPath(self.config.blob))
        self.fibers = FiberPolynomials.readFits(getAbsPath(self.config.fibers))

    def calculate(self, fiberId: np.ndarray, x: np.ndarray, y: np.ndarray) -> Struct:
        """Calculate PFI corrections

        Parameters
        ----------
        fiberId : `numpy.ndarray`
            Fiber IDs.
        x, y : `numpy.ndarray`
            X and Y coordinates of fibers in mm.

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
        edge = self.edge(x, y)
        blob = self.blob(x, y)
        fibers = np.array([self.fibers.evaluateSingle(*args) for args in zip(fiberId, x, y)])
        correction = edge + blob + fibers
        return Struct(
            correction=correction,
            edge=edge,
            blob=blob,
            fibers=fibers,
        )

    def run(self, pfsArm: PfsArm, pfsConfig: PfsConfig, skyNorms: ConstantPerFiber | None = None) -> Struct:
        """Calculate and apply the PFI corrections

        Parameters
        ----------
        pfsArm : `pfs.datamodel.drp.PfsArm`
            Arm data to be corrected.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Configuration of fibers on the focal plane.
        skyNorms : `ConstantPerFiber`, optional
            Sky normalization corrections to apply (if any).

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
        skyNorms : `numpy.ndarray` or `None`
            Sky normalization factors for each fiber (if `skyNorms` was provided).
        """
        pfsConfig = pfsConfig.select(fiberId=pfsArm.fiberId)
        result = self.calculate(pfsConfig.fiberId, pfsConfig.pfiCenter[:, 0], pfsConfig.pfiCenter[:, 1])

        self.log.info(
            "Applying PFI correction: %.3f +/- %.3f",
            np.nanmean(result.correction),
            np.nanstd(result.correction),
        )

        result.skyNorms = None
        if skyNorms is not None:
            result.skyNorms = skyNorms.eval(pfsArm.fiberId).values.reshape(len(pfsArm))
            self.log.info(
                "Applying sky norms: %.3f +/- %.3f", np.nanmean(result.skyNorms), np.nanstd(result.skyNorms)
            )
            result.correction += result.skyNorms

        good = np.isfinite(result.correction)
        pfsArm.norm[good] *= (1.0 + result.correction[good])[:, None]
        pfsArm.mask[~good] |= pfsArm.flags.add("BAD_PFI_CORRECTION")
        pfsArm.notes.pfiCorrection[:] = result.correction

        return result
