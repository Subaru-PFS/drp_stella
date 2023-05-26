from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.target import Target
from pfs.datamodel.wavelengthArray import WavelengthArray
from pfs.drp.stella.datamodel.pfsFiberArray import PfsSimpleSpectrum

import astropy.io.fits
from deprecated import deprecated
import numpy as np
import scipy.interpolate

import json
import os
import pickle

__all__ = ("FluxModelInterpolator",)


class PCACompositor:
    """Given ``mean`` and basis vectors, compute ``mean`` plus
    linear combinations of the basis vectors.

    Parameters
    ----------
    basis : `np.ndarray`
        Basis vectors. Shape (M, N), M being the number of basis vectors
        and N being the dimension of the vectors.

    mean : `np.ndarray`
        Mean. Shape (N,). This vector is added to linear combinations
        of basis vectors.

    header : `astropy.io.fits.Header`
        FITS header that contains WCS cards
        for ``mean`` and for vectors output from ``__call__()``.
    """

    def __init__(self, basis: np.ndarray, mean: np.ndarray, header: astropy.io.fits.Header) -> None:
        self.basis = np.asarray(basis)
        self.mean = np.asarray(mean)
        self.header = header

    def __call__(self, coeff: np.ndarray) -> np.ndarray:
        r"""Compute a linear combination (+ ``self.mean``).

        This method returns ``mean + \sum_i coeff[..., i] basis[i]``

        Parameters
        ----------
        coeff : `np.ndarray`
            Coefficients of the linear combination.
            Shape (..., M), M being the number of basis vectors.

        Returns
        -------
        sum : `np.ndarray`
            The sum. Shape (..., N), N being the dimension of a basis vector.
        """
        coeff = np.asarray(coeff)
        lenShape = len(coeff.shape)
        if lenShape > 1:
            ones = (1,) * (lenShape - 1)
            mean = self.mean.reshape(ones + self.mean.shape)
        else:
            mean = self.mean

        return mean + coeff @ self.basis

    def getLength(self) -> int:
        """Get the length of a vector output from __call__()."""
        return len(self.mean)

    @classmethod
    def fromFluxModelData(cls, path: str) -> "PCACompositor":
        """Read the PCA basis file in ``fluxmodeldata`` package.

        Parameters
        ----------
        path : `str`
            Path to ``fluxmodeldata`` package.

        Returns
        -------
        pcaCompositor : `PCACompositor`
            Instance of this class.
        """
        with astropy.io.fits.open(os.path.join(path, "pca.fits"), memmap=False) as pca:
            return cls.fromHDUList(pca)

    @classmethod
    def fromHDUList(cls, hduList: astropy.io.fits.HDUList) -> "PCACompositor":
        """Create an instance from a PCA basis file
        in ``fluxmodeldata`` package.

        Parameters
        ----------
        hduList : `astropy.io.fits.HDUList`
            "pca.fits" in `fluxmodeldata`` package.

        Returns
        -------
        pcaCompositor : `PCACompositor`
            Instance of this class.
        """
        basis = np.array(hduList["BASIS"].data, dtype=float)  # force conversion '>f8' => 'f8'
        mean = np.array(hduList["MEAN"].data, dtype=float)  # force conversion '>f8' => 'f8'
        wcs = astropy.io.fits.Header.fromstring(hduList["WCS"].data.tobytes())

        return cls(basis, mean, wcs)


class FluxModelInterpolator:
    """Model spectrum interpolator.

    This is a mere base class, having class method ``fromFluxModelData()``,
    which is the factory of subclasses.
    """

    def interpolate(self, teff: float, logg: float, m: float, alpha: float) -> PfsSimpleSpectrum:
        """Generate an interpolated spectrum at a given parameter point.

        Parameters
        ----------
        teff : `float`
            Effective temperature in K for interpolation.
        logg : `float`
            Surface gravity in log(/(cm/s^2)) for interpolation.
        m : `float`
            Metallicity [Fe/H] for interpolation.
        alpha : `float`
            Alpha element index [alpha/Fe] for interpolation.

        Returns
        -------
        spectrum : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
            Interpolated spectrum.
        """
        raise NotImplementedError()

    @classmethod
    def fromFluxModelData(cls, path: str) -> "FluxModelInterpolator":
        """Read the flux model in ``fluxmodeldata`` package.

        Parameters
        ----------
        path : `str`
            Path to ``fluxmodeldata`` package.

        Returns
        -------
        fluxModelInterpolator : `FluxModelInterpolator`
            The model.
        """
        if os.path.exists(os.path.join(path, "pca.fits")):
            # The fluxmodeldata package is compatible with PIPE2D-1231
            return PCAFluxModelInterpolator.fromFluxModelData(path)
        else:
            # The fluxmodeldata package is an old one for NaiveFluxModelInterpolator
            return NaiveFluxModelInterpolator.fromFluxModelData(path)


@deprecated(
    reason="NaiveFluxModelInterpolator has been replaced by PCAFluxModelInterpolator,"
    " which requires fluxmodeldata >= ambre-20230608. See PIPE2D-1231."
)
class NaiveFluxModelInterpolator(FluxModelInterpolator):
    """Model spectrum interpolator.

    This class interpolates a spectrum at a point in the parameter space.
    (It is not an interpolation in the wavelength space.)

    This interpolator is deprecated (See tickets/PIPE2D-1231).
    To use this interpolator, fluxmodeldata <= ambre-20230428 is required.

    Parameters
    ----------
    interpolator : `scipy.interpolate.RBFInterpolator`
        Instance of ``RBFInterpolator``.
    teffScale : `float`
        Constant by which ``teff`` (see ``self.interpolator()``)
        is multiplied before being passed to ``interpolator``.
    loggScale : `float`
        Constant by which ``logg`` (see ``self.interpolator()``)
        is multiplied before being passed to ``interpolator``.
    mScale : `float`
        Constant by which ``m`` (see ``self.interpolator()``)
        is multiplied before being passed to ``interpolator``.
    alphaScale : `float`
        Constant by which ``alpha`` (see ``self.interpolator()``)
        is multiplied before being passed to ``interpolator``.
    fluxScale : `float`
        Constant by which outputs of ``interpolator`` are multiplied.
    lenWavelength : `int`
        Length of wavelength array.
    wcs : `astropy.io.fits.Header`
        FITS header with WCS specifying wavelength array.
    """

    def __init__(
        self,
        interpolator: scipy.interpolate.RBFInterpolator,
        teffScale: float,
        loggScale: float,
        mScale: float,
        alphaScale: float,
        fluxScale: float,
        lenWavelength: int,
        wcs: astropy.io.fits.Header,
    ) -> None:
        self.interpolator = interpolator
        self.teffScale = teffScale
        self.loggScale = loggScale
        self.mScale = mScale
        self.alphaScale = alphaScale
        self.fluxScale = fluxScale
        self.wavelength = WavelengthArray.fromFitsHeader(wcs, lenWavelength, dtype=float)

    @classmethod
    def fromFluxModelData(cls, path: str) -> "FluxModelInterpolator":
        """Read the RBF model in ``fluxmodeldata`` package.

        The RBF model must be generated in advance
        by `makeFluxModelInterpolator.py`.

        Parameters
        ----------
        path : `str`
            Path to ``fluxmodeldata`` package.

        Returns
        -------
        fluxModelInterpolator : `FluxModelInterpolator`
            The model.
        """
        filePath = os.path.join(path, "interpolator.pickle")
        if not os.path.exists(filePath):
            raise RuntimeError(f"'{filePath}' not found. Run `makeFluxModelInterpolator.py` to generate it.")
        return cls.fromPickle(filePath)

    @classmethod
    def fromPickle(cls, path: str) -> "FluxModelInterpolator":
        """Read an RBF model from a pickle file.

        Parameters
        ----------
        path : `str`
            File name of an RBF model
            generated by `makeFluxModelInterpolator.py`.

        Returns
        -------
        fluxModelInterpolator : `FluxModelInterpolator`
            The model.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
            return cls(
                obj["interpolator"],
                obj["teffScale"],
                obj["loggScale"],
                obj["mScale"],
                obj["alphaScale"],
                obj["fluxScale"],
                obj["lenWavelength"],
                astropy.io.fits.Header.fromstring(obj["wcs"]),
            )

    def interpolate(self, teff: float, logg: float, m: float, alpha: float) -> PfsSimpleSpectrum:
        """Generate an interpolated spectrum at a given parameter point.

        Parameters
        ----------
        teff : `float`
            Effective temperature in K for interpolation.
        logg : `float`
            Surface gravity in log(/(cm/s^2)) for interpolation.
        m : `float`
            Metallicity [Fe/H] for interpolation.
        alpha : `float`
            Alpha element index [alpha/Fe] for interpolation.

        Returns
        -------
        spectrum : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
            Interpolated spectrum.
        """
        x = np.empty(shape=(1, 4), dtype=float)
        x[0, 0] = self.teffScale * teff
        x[0, 1] = self.loggScale * logg
        x[0, 2] = self.mScale * m
        x[0, 3] = self.alphaScale * alpha

        flux = self.interpolator(x)[0, :]
        flux *= self.fluxScale

        target = Target(0, 0, "0,0", 0)
        mask = np.zeros(shape=flux.shape, dtype=int)
        flags = MaskHelper()
        mask[:] = np.where(np.isfinite(flux), 0, flags.add("BAD"))

        return PfsSimpleSpectrum(target, self.wavelength, flux, mask, flags)


class PCAFluxModelInterpolator(FluxModelInterpolator):
    """Model spectrum interpolator.

    This class interpolates a spectrum at a point in the parameter space.
    (It is not an interpolation in the wavelength space.)

    This interpolator is different from `NaiveFluxModelInterpolator`
    in that the `RBFInterpolator` used by the former outputs expansion
    coefficients, which is fewer than the sampling points of a spectrum.
    The actual output spectrum is then made from (mean plus) the linear
    combination of PCA basis vectors with the interpolated expansion
    coefficients.

    Parameters
    ----------
    interpolator : `scipy.interpolate.RBFInterpolator`
        Instance of ``RBFInterpolator``.
        It interpolates expansion coefficients at given (teff, logg, m, alpha),
        the expansion coefficients being to be passed to ``pcaCompositor``.
    teffScale : `float`
        Constant by which ``teff`` (see ``self.interpolator()``)
        is multiplied before being passed to ``interpolator``.
    loggScale : `float`
        Constant by which ``logg`` (see ``self.interpolator()``)
        is multiplied before being passed to ``interpolator``.
    mScale : `float`
        Constant by which ``m`` (see ``self.interpolator()``)
        is multiplied before being passed to ``interpolator``.
    alphaScale : `float`
        Constant by which ``alpha`` (see ``self.interpolator()``)
        is multiplied before being passed to ``interpolator``.
    pcaCompositor : `PCACompositor`
        Converter that converts an output of ``interpolator``
        to a spectrum.
    """

    def __init__(
        self,
        interpolator: scipy.interpolate.RBFInterpolator,
        teffScale: float,
        loggScale: float,
        mScale: float,
        alphaScale: float,
        pcaCompositor: PCACompositor,
    ) -> None:
        self.interpolator = interpolator
        self.teffScale = teffScale
        self.loggScale = loggScale
        self.mScale = mScale
        self.alphaScale = alphaScale
        self.pcaCompositor = pcaCompositor
        self.wavelength = WavelengthArray.fromFitsHeader(
            pcaCompositor.header,
            pcaCompositor.getLength(),
            dtype=float,
        )

    @classmethod
    def fromFluxModelData(cls, path: str) -> "FluxModelInterpolator":
        """Read the RBF model in ``fluxmodeldata`` package.

        Parameters
        ----------
        path : `str`
            Path to ``fluxmodeldata`` package.

        Returns
        -------
        fluxModelInterpolator : `FluxModelInterpolator`
            The model.
        """
        with open(os.path.join(path, "interpolator-hyperparams.json")) as f:
            hyperparams = json.load(f)
        kernel = hyperparams["kernel"]
        teffScale = hyperparams["teffScale"]
        loggScale = hyperparams["loggScale"]
        mScale = hyperparams["mScale"]
        alphaScale = hyperparams["alphaScale"]
        epsilon = hyperparams["epsilon"]

        with astropy.io.fits.open(os.path.join(path, "pca.fits"), memmap=False) as pca:
            coeff = np.array(pca["COEFF"].data, dtype=float)  # force conversion '>f8' => 'f8'
            params = np.array(pca["PARAMS"].data)
            pcaCompositor = PCACompositor.fromHDUList(pca)

        xList = np.lib.recfunctions.structured_to_unstructured(params).astype(float)
        xList[:, 0] *= teffScale
        xList[:, 1] *= loggScale
        xList[:, 2] *= mScale
        xList[:, 3] *= alphaScale

        interpolator = scipy.interpolate.RBFInterpolator(xList, coeff, kernel=kernel, epsilon=epsilon)

        return cls(
            interpolator,
            teffScale,
            loggScale,
            mScale,
            alphaScale,
            pcaCompositor,
        )

    def interpolate(self, teff: float, logg: float, m: float, alpha: float) -> PfsSimpleSpectrum:
        """Generate an interpolated spectrum at a given parameter point.

        Parameters
        ----------
        teff : `float`
            Effective temperature in K for interpolation.
        logg : `float`
            Surface gravity in log(/(cm/s^2)) for interpolation.
        m : `float`
            Metallicity [Fe/H] for interpolation.
        alpha : `float`
            Alpha element index [alpha/Fe] for interpolation.

        Returns
        -------
        spectrum : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
            Interpolated spectrum.
        """
        x = np.empty(shape=(1, 4), dtype=float)
        x[0, 0] = self.teffScale * teff
        x[0, 1] = self.loggScale * logg
        x[0, 2] = self.mScale * m
        x[0, 3] = self.alphaScale * alpha

        coeff = self.interpolator(x)[0, :]
        flux = self.pcaCompositor(coeff)

        target = Target(0, 0, "0,0", 0)
        mask = np.zeros(shape=flux.shape, dtype=int)
        flags = MaskHelper()
        mask[:] = np.where(np.isfinite(flux), 0, flags.add("BAD"))

        return PfsSimpleSpectrum(target, self.wavelength, flux, mask, flags)
