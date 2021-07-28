import types

import numpy as np
import astropy.io.fits

from lsst.pex.config import Config, Field
from lsst.pipe.base import Task, Struct

from pfs.drp.stella.datamodel.interpolate import interpolateFlux, interpolateMask

import lsstDebug


class FocalPlaneFunction(types.SimpleNamespace):
    """Vector function on the focal plane

    This implementation is a placeholder, as it simply returns a constant
    vector as a function of wavelength.

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
        super().__init__(wavelength=wavelength, value=value, mask=mask, variance=variance)

    def __call__(self, wavelengths, positions):
        """Evaluate the function at the provided positions

        We interpolate the variance without doing any of the usual error
        propagation. Since there is likely some amount of unknown covariance
        (if only from resampling to a common wavelength scale), following the
        usual error propagation formulae as if there is no covariance would
        artificially suppress the noise estimates.

        Parameters
        ----------
        wavelengths : iterable (length ``N``) of `numpy.ndarray` of shape ``(M)``
            Wavelength arrays
        positions : `numpy.ndarray` of shape ``(N, 2)``
            Positions at which to evaluate.

        Returns
        -------
        values : list (length ``N``) of `numpy.ndarray` of `float`, shape ``(M)``
            Vector function evaluated at each position.
        masks : list (length ``N``) of `numpy.ndarray` of `bool`, shape ``(M)``
            Indicates whether the value at each position is valid.
        variances : list (length ``N``) of `numpy.ndarray` of `float`, shape ``(M)``
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
        return Struct(values=values, masks=masks, variances=variances)

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
            value = fits["VALUE"].data
            mask = fits["MASK"].data.astype(bool)
            variance = fits["VARIANCE"].data
        return cls(wavelength, value, mask, variance)

    def writeFits(self, filename):
        """Write to FITS file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        fits = astropy.io.fits.HDUList()
        fits.append(astropy.io.fits.ImageHDU(self.wavelength, name="WAVELENGTH"))
        fits.append(astropy.io.fits.ImageHDU(self.value, name="VALUE"))
        fits.append(astropy.io.fits.ImageHDU(self.mask.astype(np.uint8), name="MASK"))
        fits.append(astropy.io.fits.ImageHDU(self.variance, name="VARIANCE"))
        with open(filename, "wb") as fd:
            fits.writeto(fd)


class FitFocalPlaneConfig(Config):
    """Configuration for FitFocalPlaneTask"""
    rejIterations = Field(dtype=int, default=3, doc="Number of rejection iterations")
    rejThreshold = Field(dtype=float, default=3.0, doc="Rejection threshold (sigma)")


class FitFocalPlaneTask(Task):
    """Fit a vector function over the focal plane

    This implementation is a placeholder, as no attention is paid to the
    position of the fibers on the focal plane.
    """
    ConfigClass = FitFocalPlaneConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)
        self._DefaultName = "fitFocalPlane"

    def run(self, wavelength, vectors, variances, masks, fiberIdList, pfsConfig):
        """Fit a vector function as a function of wavelength over the focal plane

        Note that this requires that all the input vectors have the same
        wavelength array.

        Parameters
        ----------
        wavelength : `numpy.ndarray` of length ``N``
            Wavelength values, of length ``N``.
        vectors : `numpy.ndarray` of shape ``(M, N)``
            Measured vectors of length ``N`` for ``M`` positions.
        variances : `numpy.ndarray` of shape ``(M, N)``
            Variances in the measured vectors.
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
        return self.fit(wavelength, vectors, variances, masks, centers)

    def fitImpl(self, wavelength, values, masks, variances):
        """Implementation to fit a vector function over the focal plane

        This implementation is a placeholder, as no attention is paid to the
        position of the fibers on the focal plane. Essentially, this is a coadd.

        We assume that all interpolation and resampling has already been done,
        so all the inputs have a common wavelength scale.

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`, length ``N``
            Wavelength values, of length ``N``.
        values : `numpy.ndarray` of `float`, shape ``(M, N)``
            Measured values, of length ``N`` for ``M`` positions.
        masks : `numpy.ndarray` of `bool`, shape ``(M, N)``
            Array indicating entries that should be masked.
        variances : `numpy.ndarray` of `float`, shape ``(M, N)``
            Errors in the measured vectors.
        centers : `numpy.ndarray` of shape ``(M, 2)``
            Fiber centers.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Function fit to the data.
        """
        numSamples, length = values.shape
        coaddValues = np.zeros(length, dtype=float)
        coaddMask = np.ones(length, dtype=bool)
        coaddVariance = np.zeros(length, dtype=float)
        sumWeights = np.zeros(length, dtype=float)

        for ii in range(numSamples):
            weight = np.zeros(length, dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                good = ~masks[ii] & (variances[ii] > 0)
                weight[good] = 1.0/variances[ii][good]
                coaddValues[good] += values[ii][good]*weight[good]
                coaddMask[good] = False
                sumWeights += weight

        good = sumWeights > 0
        coaddValues[good] /= sumWeights[good]
        coaddVariance[good] = 1.0/sumWeights[good]
        coaddVariance[~good] = np.inf
        coaddMask = ~good

        return FocalPlaneFunction(wavelength, coaddValues, coaddMask, coaddVariance)

    def fit(self, wavelength, values, masks, variances, centers):
        """Fit a vector function over the focal plane

        This implementation is a placeholder, as no attention is paid to the
        position of the fibers on the focal plane. Essentially, this is a coadd.

        We assume that all interpolation and resampling has already been done,
        so all the inputs have a common wavelength scale.

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`, length ``N``
            Wavelength values, of length ``N``.
        values : `numpy.ndarray` of `float`, shape ``(M, N)``
            Measured values, of length ``N`` for ``M`` positions.
        masks : `numpy.ndarray` of `bool`, shape ``(M, N)``
            Array indicating entries that should be masked.
        variances : `numpy.ndarray` of `float`, shape ``(M, N)``
            Errors in the measured vectors.
        centers : `numpy.ndarray` of shape ``(M, 2)``
            Fiber centers.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Function fit to the data.
        """
        wavelength = np.array(wavelength)
        values = np.array(values)
        masks = np.array(masks)
        variances = np.array(variances)
        assert values.shape == masks.shape
        assert len(values) == len(centers)
        numSamples, length = values.shape
        assert variances.shape == values.shape
        assert wavelength.shape == (length,)
        assert centers.shape == (numSamples, 2)

        rejected = np.zeros((numSamples, length), dtype=bool)
        for ii in range(self.config.rejIterations):
            func = self.fitImpl(wavelength, values, masks | rejected, variances)
            with np.errstate(invalid="ignore", divide="ignore"):
                resid = (values - func.value[np.newaxis, :])/np.sqrt(variances)
                newRejected = ~rejected & ~masks & ~func.mask & (np.abs(resid) > self.config.rejThreshold)
            if self.debugInfo.plot:
                self.plot(wavelength, values, masks | rejected, func, f"Iteration {ii}", newRejected)
            if not np.any(newRejected):
                break
            rejected |= newRejected
        else:
            func = self.fitImpl(wavelength, values, masks | rejected, variances)
            resid = (values - func.value[np.newaxis, :])/np.sqrt(variances)
            if self.debugInfo.plot:
                self.plot(wavelength, values, masks | rejected, func, f"Final")

        good = ~(rejected | masks)
        chi2 = np.sum(resid[good]**2)
        self.log.info("Fit focal plane function: "
                      "chi^2=%f length=%d/%d numSamples=%d numGood=%d numRejected=%d",
                      chi2, (~func.mask).sum(), length, numSamples, good.sum(), rejected.sum())
        return func

    def plot(self, wavelength, values, masks, func, title, rejected=None):
        """Plot the input and fit values

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`, shape ``(M,)``
            Common wavelength array.
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Measured values for each sample+wavelength.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Boolean flag indicating whether a point should be masked.
        func : `FocalPlaneFunction`
            Fit result.
        title : `str`
            Title to use for the plot.
        rejected : `numpy.ndarray` of `bool`, shape ``(N, M)``; optional
            Boolean flag indicating whether a point has been rejected.
        """
        import matplotlib.pyplot as plt
        num = len(values)
        for ii in range(num):
            plt.plot(wavelength, values[ii], "k-")
            mm = masks[ii]
            if np.any(mm):
                plt.plot(wavelength[mm], values[ii][mm], "kx")
            if rejected is not None:
                rej = rejected[ii]
                if np.any(rej):
                    plt.plot(wavelength[rej], values[ii][rej], "rx")
        plt.plot(func.wavelength, func.value, "b-")
        if np.any(func.mask):
            plt.plot(func.wavelength[func.mask], func.value[func.mask], "bx")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Value")
        plt.title(title)
        plt.show()

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
        values : list (length ``N``) of `numpy.ndarray` of shape ``(M)``
            Vector function evaluated at each position.
        variances : list (length ``N``) of `numpy.ndarray` of shape ``(M)``
            Variances for each position.
        """
        if len(wavelength.shape) == 1:
            wavelength = np.array([wavelength]*len(fiberIdList))
        centers = pfsConfig.extractCenters(fiberIdList)
        return func(wavelength, centers)
