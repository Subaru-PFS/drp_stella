import numpy as np
import scipy.optimize

import lsst.utils.tests

from pfs.drp.stella.interpolate import interpolateFlux, interpolateCovariance
from pfs.drp.stella.tests import runTests, methodParametersProduct
from pfs.drp.stella.utils.math import robustRms

display = None


def makeCovarianceMatrix(covar: np.ndarray) -> np.ndarray:
    """Make full covariance matrix from compact representation

    Parameters
    ----------
    covar : `numpy.ndarray` of `float`
        Compact covariance representation, shape ``(numCovar, length)``.

    Returns
    -------
    sigma : `numpy.ndarray` of `float`
        Full covariance matrix, shape ``(length, length)``.
    """
    numCovar = covar.shape[0]
    length = covar.shape[1]
    sigma = np.zeros((length, length), dtype=float)
    sigma += np.diag(covar[0], k=0)
    for offset in range(1, numCovar):
        diag = covar[offset, :length - offset]
        sigma += np.diag(diag, k=offset)
        sigma += np.diag(diag, k=-offset)
    return sigma


class CovarianceTestCase(lsst.utils.tests.TestCase):
    @methodParametersProduct(
        order=(1, 3, 7),
        mode=("NONE", "VARIANCE", "COVAR_DIAG", "COVAR_OFFDIAG", "COVAR_PROJECT", "MONTE_CARLO"),
    )
    def testFitting(self, order: int, mode: str):
        """Test covariance fitting on synthetic data"""
        length = 50
        center = length//2 + 1.234
        width = 2.34
        fluxValue = 123.45
        sigNoise = 23.456  # signal-to-noise of the peak (not of the total flux)
        varianceValue = (fluxValue/sigNoise)**2
        scale = 2.468
        rng = np.random.RandomState(12345)
        numCovar = order + 2

        outLength = int(length*scale)
        inWavelength = np.linspace(500.0, 900.0, length)
        outWavelength = np.linspace(500.0, 900.0, outLength)
        inIndices = np.arange(length, dtype=float)
        outIndices = np.arange(outWavelength.size, dtype=float)

        inFlux = fluxValue*np.exp(-0.5*(inIndices - center)**2/width**2)

        def model(xx: np.ndarray, amplitude: float, center: float, width: float) -> np.ndarray:
            return amplitude*np.exp(-0.5*(xx - center)**2/width**2)

        xx = np.arange(outLength, dtype=float)
        thisVariance = np.full(length, varianceValue)
        thisFlux = inFlux + rng.normal(scale=np.sqrt(thisVariance), size=length)

        outFlux = interpolateFlux(inWavelength, thisFlux, outWavelength, order=order)
        outCovar = interpolateCovariance(
            inWavelength, thisVariance, outWavelength, order=order, numCovar=numCovar
        )

        guessAmplitude = np.max(outFlux)
        guessCenter = xx[np.argmax(outFlux)]
        guessWidth = 4.0

        if mode == "COVAR_DIAG":  # Increase the diagonal
            regularize = 0.25
            sigma = makeCovarianceMatrix(outCovar)
            sigma += np.diag(np.full(outLength, regularize*np.max(outCovar[0])))
        elif mode == "COVAR_OFFDIAG":  # Shrink the off-diagonal elements
            shrink = 0.3
            sigma = makeCovarianceMatrix(outCovar)
            numCovar = outCovar.shape[0]
            for offset in range(1, numCovar):
                diag = np.diag(sigma, k=offset)*shrink
                sigma -= np.diag(diag, k=offset)
                sigma -= np.diag(diag, k=-offset)
        elif mode == "COVAR_PROJECT":  # Project onto positive eigenvalues
            # Larger values may result in more sensible fits
            threshold = 1.0e-13
            sigma = makeCovarianceMatrix(outCovar)
            eigvals, eigvecs = np.linalg.eigh(sigma)
            eigvals = np.maximum(eigvals, threshold)
            sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
        elif mode == "VARIANCE":
            sigma = np.sqrt(outCovar[0])
        elif mode in ("NONE", "MONTE_CARLO"):
            sigma = None
        else:
            raise RuntimeError(f"Unknown mode: {mode}")

        if mode.startswith("COVAR_"):
            # Check that the covariance matrix is positive definite
            eigvals = np.linalg.eigvals(sigma)
            self.assertTrue(np.all(np.abs(np.imag(eigvals) < 1.0e-9)))
            self.assertTrue(np.all(np.real(eigvals) > -1.0e-9))

        guess = np.array([guessAmplitude, guessCenter, guessWidth], dtype=float)
        actual = np.array([
            fluxValue,
            np.interp(np.interp(center, inIndices, inWavelength), outWavelength, outIndices),
            width*scale
        ], dtype=float)

        if mode == "MONTE_CARLO":
            # Use Monte Carlo to estimate uncertainties
            samples = 1000
            fits = np.full((samples, 3), np.nan, dtype=float)
            for ii in range(samples):
                thisVariance = np.full(length, varianceValue)
                thisFlux = inFlux + rng.normal(scale=np.sqrt(thisVariance), size=length)
                outFlux = interpolateFlux(inWavelength, thisFlux, outWavelength, order=order)
                fits[ii] = scipy.optimize.curve_fit(model, xx, outFlux, guess)[0]

            bias = np.nanmedian(fits, axis=0) - actual
            stdev = [robustRms(ff, True) for ff in fits.T]
        else:
            fit = scipy.optimize.curve_fit(
                model,
                xx,
                outFlux,
                p0=guess,
                sigma=sigma,
                absolute_sigma=True,
            )
            bias = fit[0] - actual
            stdev = np.sqrt(np.diag(fit[1]))

        print(
            f"order={order} mode={mode} --> "
            f"amplitude = {bias[0]:.3f} +/- {stdev[0]:.3f}, "
            f"center = {bias[1]:.3f} +/- {stdev[1]:.3f}, "
            f"width = {bias[2]:.3f} +/- {stdev[2]:.3f}"
        )


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
