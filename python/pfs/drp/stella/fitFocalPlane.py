import types

import numpy as np
import scipy.interpolate
import astropy.io.fits

from lsst.pex.config import Config
from lsst.pipe.base import Task


class FocalPlaneFunction(types.SimpleNamespace):
    """Vector function on the focal plane

    This implementation is a placeholder, as it simply returns a constant
    vector as a function of wavelength.

    Parameters
    ----------
    array : `numpy.ndarray`
        Constant vector to use.
    """
    def __init__(self, wavelength, vector):
        interpolator = scipy.interpolate.interp1d(wavelength, vector, kind='linear', bounds_error=False,
                                                  fill_value=0, copy=True, assume_sorted=True)
        super().__init__(wavelength=wavelength, vector=vector, interpolator=interpolator)

    def __call__(self, wavelengths, positions):
        """Evaluate the function at the provided positions

        Parameters
        ----------
        wavelengths : iterable (length ``N``) of `numpy.ndarray` of shape ``(M)``
            Wavelength arrays
        positions : `numpy.ndarray` of shape ``(N, 2)``
            Positions at which to evaluate.

        Returns
        -------
        result : list (length ``N``) of `numpy.ndarray` of shape ``(M)``
            Vector function evaluated at each position.
        """
        assert len(wavelengths) == len(positions)
        doResample = (wl.shape != self.wavelength.shape or not np.all(wl == self.wavelength) for
                      wl in wavelengths)
        return [self.interpolator(wl) if resamp else self.vector for
                wl, resamp in zip(wavelengths, doResample)]

    @classmethod
    def readFits(cls, filename):
        """Read from FITS file

        Parameters
        ----------
        filename : `str`
            Filename to read.

        Returns
        -------
        self : `FocalPlaneFunction`
            Function read from FITS file.
        """
        with astropy.io.fits.open(filename) as fits:
            wavelength = fits["WAVELENGTH"].data
            vector = fits["VECTOR"].data
        return cls(wavelength, vector)

    def writeFits(self, filename):
        """Write to FITS file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        fits = astropy.io.fits.HDUList()
        fits.append(astropy.io.fits.ImageHDU(self.wavelength, name="WAVELENGTH"))
        fits.append(astropy.io.fits.ImageHDU(self.vector, name="VECTOR"))
        with open(filename, "wb") as fd:
            fits.writeto(fd)


class FitFocalPlaneConfig(Config):
    """Configuration for FitFocalPlaneTask"""
    pass


class FitFocalPlaneTask(Task):
    """Fit a vector function over the focal plane

    This implementation is a placeholder, as it simply averages the input
    vectors instead of doing any real fitting or rejection of outliers, and
    no attention is paid to the position of the fibers on the focal plane.
    """
    ConfigClass = FitFocalPlaneConfig

    def run(self, wavelength, vectors, errors, masks, fiberIdList, pfsConfig):
        """Fit a vector function as a function of wavelength over the focal plane

        Note that this requires that all the input vectors have the same
        wavelength array.

        Parameters
        ----------
        wavelength : `numpy.ndarray` of length ``N``
            Wavelength values, of length ``N``.
        vectors : `numpy.ndarray` of shape ``(M, N)``
            Measured vectors of length ``N`` for ``M`` positions.
        errors : `numpy.ndarray` of shape ``(M, N)``
            Errors in the measured vectors.
        masks : `numpy.ndarray` of shape ``(M, N)``
            Non-zero entries should be masked in the fit. If a boolean array,
            we'll mask entries where this is ``True``.
        fiberIdList : iterable of `int` of length ``M``
            Fibers being fit.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for getting the fiber centers.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Function fit to the data.
        """
        centers = pfsConfig.extractCenters(fiberIdList)
        return self.fit(wavelength, vectors, errors, masks, centers)

    def fit(self, wavelength, vectors, errors, masks, centers):
        """Fit a vector function over the focal plane

        This implementation is a placeholder, as it simply averages the input
        vectors instead of doing any real fitting or rejection of outliers, and
        no attention is paid to the position of the fibers on the focal plane.

        We assume that all interpolation and resampling has already been done,
        so all the inputs have a common wavelength scale.

        Parameters
        ----------
        wavelength : `numpy.ndarray` of length ``N``
            Wavelength values, of length ``N``.
        vectors : `numpy.ndarray` of shape ``(M, N)``
            Measured vectors of length ``N`` for ``M`` positions.
        errors : `numpy.ndarray` of shape ``(M, N)``
            Errors in the measured vectors.
        masks : `numpy.ndarray` of shape ``(M, N)``
            Non-zero entries should be masked in the fit. If a boolean array,
            we'll mask entries where this is ``True``.
        centers : `numpy.ndarray` of shape ``(M, 2)``
            Fiber centers.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Function fit to the data.
        """
        wavelength = np.array(wavelength)
        vectors = np.array(vectors)
        masks = np.array(masks)
        assert vectors.shape == masks.shape
        assert len(vectors) == len(centers)
        numSamples, numElements = vectors.shape
        assert centers.shape == (numSamples, 2)
        good = (masks == 0)
        average = np.zeros(numElements, dtype=vectors.dtype)
        for ii in range(numElements):
            vv = vectors[:, ii]
            gg = good[:, ii]
            average[ii] = np.average(vv[gg]) if np.any(gg) else 0.0
        return FocalPlaneFunction(wavelength, average)

    def apply(self, func, wavelength, fiberIdList, pfsConfig):
        """Apply the fit to fibers

        Parameters
        ----------
        func : `FocalPlaneFunction`
            Function fit to the data.
        wavelength : `numpy.ndarray` of length ``N`` or an iterable of the same
            Wavelength values. This may be a single array (same for all fibers)
            or an iterable of arrays (of length ``M``).
        fiberIdList : iterable of `int` of length ``M``
            Fibers being fit.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for getting the fiber centers.

        Returns
        -------
        result : `numpy.ndarray` of shape ``(M, N)``
            Function fit to the data.
        """
        if len(wavelength.shape) == 1:
            wavelength = np.array([wavelength]*len(fiberIdList))
        centers = pfsConfig.extractCenters(fiberIdList)
        return func(wavelength, centers)
