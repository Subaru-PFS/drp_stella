import types

import numpy as np
import astropy.io.fits

from lsst.pex.config import Config
from lsst.pipe.base import Task


class FocalPlaneFunction(types.SimpleNamespace):
    """Vector function on the focal plane

    This implementation is a placeholder, as it simply returns a constant
    vector.

    Parameters
    ----------
    vector : `numpy.ndarray`
        Constant vector to use.
    """
    def __init__(self, vector):
        super().__init__(vector=vector)

    def __call__(self, positions):
        """Evaluate the function at the provided positions

        Parameters
        ----------
        positions : `numpy.ndarray` of shape ``(N, 2)``
            Positions at which to evaluate.

        Returns
        -------
        result : `numpy.ndarray` of shape ``(N, M)``
            Vector function evaluated at each position.
        """
        numPositions = len(positions)
        result = np.empty((numPositions, len(self.vector)), dtype=self.vector.dtype)
        for ii in range(numPositions):
            result[ii] = self.vector
        return result

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
            vector = fits[0].data
        return cls(vector)

    def writeFits(self, filename):
        """Write to FITS file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        fits = astropy.io.fits.HDUList()
        fits.append(astropy.io.fits.ImageHDU(self.vector))
        with open(filename, "w") as fd:
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

    def run(self, vectors, errors, masks, fiberIdList, pfsConfig):
        """Fit a vector function over the focal plane

        Parameters
        ----------
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
        return self.fit(vectors, errors, masks, centers)

    def fit(self, vectors, errors, masks, centers):
        """Fit a vector function over the focal plane

        This implementation is a placeholder, as it simply averages the input
        vectors instead of doing any real fitting or rejection of outliers, and
        no attention is paid to the position of the fibers on the focal plane.

        Parameters
        ----------
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
        return FocalPlaneFunction(average)

    def apply(self, func, fiberIdList, pfsConfig):
        """Apply the fit to fibers

        Parameters
        ----------
        func : `FocalPlaneFunction`
            Function fit to the data.
        fiberIdList : iterable of `int` of length ``M``
            Fibers being fit.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for getting the fiber centers.

        Returns
        -------
        result : `numpy.ndarray` of shape ``(M, N)``
            Function fit to the data.
        """
        centers = pfsConfig.extractCenters(fiberIdList)
        return func(centers)
