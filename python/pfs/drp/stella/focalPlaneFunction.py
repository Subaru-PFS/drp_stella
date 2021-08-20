from abc import ABC, abstractmethod
from typing import List, Dict

import numpy as np
from scipy.interpolate import LSQUnivariateSpline, BSpline, interp1d
from scipy.stats import binned_statistic
import astropy.io.fits

from lsst.pipe.base import Struct

from pfs.datamodel import PfsConfig
from pfs.drp.stella.datamodel import PfsFiberArraySet
from pfs.drp.stella.datamodel.interpolate import interpolateFlux, interpolateMask

__all__ = ("FocalPlaneFunction", "ConstantFocalPlaneFunction", "OversampledSpline",
           "BlockedOversampledSpline")


class FocalPlaneFunction(ABC):
    """Spectral function on the focal plane

    A model of some spectrum as a function of position on the focal plane.

    This is an abstract base class. Subclasses need to override:
    * ``fitArrays`` class method: fit the function
    * ``evaluate`` method: evaluate the spectrum at a position.
    * ``toFits`` method: write to a FITS file.
    * ``fromFits`` class method: construct from a FITS file.
    """
    @classmethod
    def fit(cls, spectra: PfsFiberArraySet, pfsConfig: PfsConfig, maskFlags: List[str],
            *, rejected=None, robust: bool = False, **kwargs) -> "FocalPlaneFunction":
        """Fit a spectral function on the focal plane to spectra

        This method is intended as a convenience for users.

        Parameters
        ----------
        spectra : `PfsFiberArraySet`
            Spectra to fit. This should contain only the fibers to be fit.
        pfsConfig : `PfsConfig`
            Top-end configuration. This should contain only the fibers to be
            fit.
        maskFlags : iterable of `str`
            Mask flags to exclude from fit.
        rejected : `numpy.ndarray` of `bool`, shape ``(Nspectra, length)``
            Which pixels should be rejected?
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.
        **kwargs : `dict`
            Fitting parameters.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Fit function.
        """
        if len(spectra) != len(pfsConfig):
            raise RuntimeError(
                f"Length mismatch between spectra ({len(spectra)}) and pfsConfig ({len(pfsConfig)})")
        if np.all(spectra.fiberId != pfsConfig.fiberId):
            raise RuntimeError("fiberId mismatch between spectra and pfsConfig")

        values = spectra.flux/spectra.norm
        variance = spectra.variance/spectra.norm**2
        masks = spectra.mask & spectra.flags.get(*maskFlags) != 0
        if rejected is not None:
            masks |= rejected
        positions = pfsConfig.extractCenters(pfsConfig.fiberId)
        return cls.fitArrays(spectra.fiberId, spectra.wavelength, values, masks, variance,
                             positions, robust=robust, **kwargs)

    @classmethod
    @abstractmethod
    def fitArrays(cls, fiberId, wavelengths, values, masks, variances, positions, robust: bool = False,
                  **kwargs) -> "FocalPlaneFunction":
        """Fit a spectral function on the focal plane to arrays

        This method is intended as a convenience for developers.

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            Fiber identifiers.
        wavelengths : `numpy.ndarray` of `float`, shape ``(N, M)``
            Wavelength array.
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Values to fit.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Boolean array indicating values to ignore from the fit.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variance values to use in fit.
        positions : `numpy.ndarray` of `float`, shape ``(2, N)``
            Focal-plane positions of fibers.
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.
        **kwargs : `dict`
            Fitting parameters.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Function fit to input arrays.
        """
        raise NotImplementedError("Subclasses must override")

    def __call__(self, wavelengths, pfsConfig: PfsConfig):
        """Evaluate the function for the provided fiberIds

        This method is intended as a convenience for users.

        For convenience, if the ``pfsConfig`` contains only a single fiber,
        we'll return a single array for each output.

        Parameters
        ----------
        wavelengths : iterable (length ``N``) of `numpy.ndarray` of shape ``(M,)``
            Wavelength arrays.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration. This should contain only the fibers of
            interest.

        Returns
        -------
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Vector function evaluated at each position.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Indicates whether the value at each position is valid.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variances for each position.
        """
        isSingle = False  # Only a single fiber?
        if len(wavelengths.shape) == 1:
            isSingle = True
            wavelengths = np.array([wavelengths]*len(pfsConfig))
        positions = pfsConfig.extractCenters(pfsConfig.fiberId)
        result = self.evaluate(wavelengths, pfsConfig.fiberId, positions)
        if isSingle:
            result.values = result.values[0]
            result.masks = result.masks[0]
            result.variances = result.variances[0]
        return result

    @abstractmethod
    def evaluate(self, wavelengths, fiberIds, positions):
        """Evaluate the function at the provided positions

        This method is intended as a convenience for developers.

        This abstract method must be overridden by subclasses.

        Parameters
        ----------
        wavelengths : `numpy.ndarray` of shape ``(N, M)``
            Wavelength arrays.
        fiberIds : `numpy.ndarray` of `int` of shape ``(N,)``
            Fiber identifiers.
        positions : `numpy.ndarray` of shape ``(N, 2)``
            Focal-plane positions at which to evaluate.

        Returns
        -------
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Vector function evaluated at each position.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Indicates whether the value at each position is valid.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variances for each position.
        """
        raise NotImplementedError("Subclasses must override")

    @classmethod
    def readFits(cls, filename: str) -> "FocalPlaneFunction":
        """Read from FITS file

        Parameters
        ----------
        filename : `str`
            Filename to read.

        Returns
        -------
        self : subclass of `FocalPlaneFunction`
            Function read from FITS file.
        """
        subclasses = {ss.__name__: ss for ss in cls.__subclasses__()}  # ignores sub-sub-classes
        with astropy.io.fits.open(filename) as fits:
            name = fits[0].header["pfs_focalPlaneFunction_class"]
            if name not in subclasses:
                raise RuntimeError(f"Unrecognised pfs_focalPlaneFunction_class value: {name}")
            return subclasses[name].fromFits(fits)

    def writeFits(self, filename: str) -> None:
        """Write to FITS file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        header = astropy.io.fits.Header()
        header["HIERARCH pfs_focalPlaneFunction_class"] = type(self).__name__
        fits = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(header=header)])
        self.toFits(fits)
        with open(filename, "wb") as fd:
            fits.writeto(fd)

    @classmethod
    @abstractmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "FocalPlaneFunction":
        """Construct from FITS file

        This abstract class method must be overridden by subclasses.

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        self : cls
            Constructed focal plane function.
        """
        raise NotImplementedError("Subclasses must override")

    @abstractmethod
    def toFits(self, fits: astropy.io.fits.HDUList) -> None:
        """Write to FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file to which to write.
        """
        raise NotImplementedError("Subclasses must override")


class ConstantFocalPlaneFunction(FocalPlaneFunction):
    """Constant function over the focal plane

    This implementation is something of a placeholder, as it simply returns a
    constant vector as a function of wavelength. No attention is paid to the
    position of the fibers on the focal plane.

    Parameters
    ----------
    wavelength : `numpy.ndarray` of `float`, shape ``(N,)``
        Wavelengths for each value.
    value : `numpy.ndarray` of `float`, shape ``(N,)``
        Value at each wavelength.
    mask : `numpy.ndarray` of `bool`, shape ``(N,)``
        Indicate whether values should be ignored.
    variance : `numpy.ndarray` of `float`, shape ``(N,)``
        Variance in value at each wavelength.
    """
    def __init__(self, wavelength, value, mask, variance):
        self.wavelength = wavelength
        self.value = value
        self.mask = mask
        self.variance = variance

    @classmethod
    def fitArrays(cls, fiberId, wavelength, values, masks, variances, positions, robust: bool = False,
                  **kwargs) -> "ConstantFocalPlaneFunction":
        """Fit a spectral function on the focal plane to arrays

        This implementation is something of a placeholder, as no attention is
        paid to the position of the fibers on the focal plane. Essentially, this
        is a coadd.

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            Fiber identifiers.
        wavelength : `numpy.ndarray` of `float`, shape ``(N, M)``
            Wavelength array. The wavelength array for all the inputs must be
            identical.
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Values to fit.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Boolean array indicating values to ignore from the fit.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variance values to use in fit.
        positions : `numpy.ndarray` of `float`, shape ``(2, N)``
            Focal-plane positions of fibers.
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.
        **kwargs : `dict`
            Fitting parameters.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Function fit to input arrays.
        """
        if not np.all(np.equal.reduce(wavelength, axis=0)):
            raise RuntimeError("Wavelength arrays not identical")

        if robust:
            return ConstantFocalPlaneFunction(wavelength[0], np.median(values, axis=0),
                                              np.logical_and.reduce(masks, axis=0),
                                              np.median(variances, axis=0))

        weight = 1.0/variances
        good = ~masks & (variances > 0)
        weight[~good] = 0.0

        sumWeights = np.sum(weight, axis=0)
        coaddValues = np.sum(values*weight, axis=0)/sumWeights
        coaddVariance = np.where(sumWeights > 0, 1.0/sumWeights, np.inf)
        coaddMask = sumWeights <= 0
        return ConstantFocalPlaneFunction(wavelength[0], coaddValues, coaddMask, coaddVariance)

    def evaluate(self, wavelengths, fiberIds, positions) -> Struct:
        """Evaluate the function at the provided positions

        We interpolate the variance without doing any of the usual error
        propagation. Since there is likely some amount of unknown covariance
        (if only from resampling to a common wavelength scale), following the
        usual error propagation formulae as if there is no covariance would
        artificially suppress the noise estimates.

        Parameters
        ----------
        wavelengths : `numpy.ndarray` of shape ``(N, M)``
            Wavelength arrays.
        fiberIds : `numpy.ndarray` of `int` of shape ``(N,)``
            Fiber identifiers.
        positions : `numpy.ndarray` of shape ``(N, 2)``
            Focal-plane positions at which to evaluate.

        Returns
        -------
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Vector function evaluated at each position.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Indicates whether the value at each position is valid.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variances for each position.
        """
        assert len(wavelengths) == len(positions)

        doResample = [wl.shape != self.wavelength.shape or not np.all(wl == self.wavelength) for
                      wl in wavelengths]

        values = [interpolateFlux(self.wavelength, self.value, wl) if resamp else self.value for
                  wl, resamp in zip(wavelengths, doResample)]
        masks = [interpolateMask(self.wavelength, self.mask, wl).astype(bool) if resamp else self.mask for
                 wl, resamp in zip(wavelengths, doResample)]
        variances = [interpolateFlux(self.wavelength, self.variance, wl) if resamp else self.variance for
                     wl, resamp in zip(wavelengths, doResample)]
        return Struct(values=np.array(values), masks=np.array(masks), variances=np.array(variances))

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "ConstantFocalPlaneFunction":
        """Construct from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        self : cls
            Constructed focal plane function.
        """
        wavelength = fits["WAVELENGTH"].data
        value = fits["VALUE"].data
        mask = fits["MASK"].data.astype(bool)
        variance = fits["VARIANCE"].data
        return cls(wavelength, value, mask, variance)

    def toFits(self, fits: astropy.io.fits.HDUList) -> None:
        """Write to FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file to which to write.
        """
        fits.append(astropy.io.fits.ImageHDU(self.wavelength, name="WAVELENGTH"))
        fits.append(astropy.io.fits.ImageHDU(self.value, name="VALUE"))
        fits.append(astropy.io.fits.ImageHDU(self.mask.astype(np.uint8), name="MASK"))
        fits.append(astropy.io.fits.ImageHDU(self.variance, name="VARIANCE"))

