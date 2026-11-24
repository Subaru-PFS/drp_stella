import numpy as np

import lsst.utils.tests
from lsst.pipe.base import Struct
from lsst.afw.math import offsetImage
from lsst.afw.image import ImageD

from pfs.drp.stella.interpolate import (
    calculateDispersion,
    interpolateFlux,
    interpolateVariance,
    interpolate,
    interpolateCovariance,
    interpolateMaskedVariance,
    Resampler,
)
from pfs.drp.stella.tests import runTests, classParameters, methodParameters, methodParametersProduct

display = None


@classParameters(order=(1, 3))
class InterpolateTestCase(lsst.utils.tests.TestCase):
    order: int  # Interpolation order; set by classParameters

    def setUp(self):
        self.min = 650
        self.max = 950

    def makeSpectrum(self, dispersion, flux=1.0):
        """Create a spectrum

        Parameters
        ----------
        dispersion : `float`
            Dispersion, in nm/pixel.
        flux : `float`
            Flux value

        Returns
        -------
        wavelength : `numpy.ndarray` of `float`
            Array of wavelengths.
        flux : `numpy.ndarray` of `float`
            Array of flux.
        """
        num = int((self.max - self.min)/dispersion)
        wavelength = np.linspace(self.min, self.max, num + 1, dtype=float)
        fluxArray = np.full_like(wavelength, flux)
        return Struct(wavelength=wavelength, flux=fluxArray)

    @methodParameters(dispersion=(0.1, 0.3, 0.5))
    def testCalculateDispersion(self, dispersion):
        """Test calculateDispersion function

        We create a spectrum with a particular dispersion. The function should
        measure the correct dispersion.

        Parameters
        ----------
        dispersion : `float`
            Dispersion, in nm/pixel.
        """
        spectrum = self.makeSpectrum(dispersion)
        self.assertFloatsAlmostEqual(calculateDispersion(spectrum.wavelength), dispersion, atol=1.0e-13)

    @methodParametersProduct(inDispersion=(0.1, 0.3, 0.5),
                             outDispersion=(0.01, 0.3, 1.0))
    def testFlux(self, inDispersion, outDispersion):
        """Test interpolateFlux function

        Since all we're doing is interpolating, the flux should stay the same.

        Parameters
        ----------
        inDispersion : `float`
            Dispersion of input spectrum.
        outDispersion : `float`
            Dispersion of output spectrum.
        """
        flux = 123.45
        inSpectrum = self.makeSpectrum(inDispersion, flux)
        expected = self.makeSpectrum(outDispersion, flux)
        outSpectrum = interpolateFlux(
            inSpectrum.wavelength, inSpectrum.flux, expected.wavelength, order=self.order
        )
        self.assertFloatsAlmostEqual(outSpectrum, flux, atol=1.0e-12)

    @methodParameters(inDispersion=(3.0, 5.0, 10.0),
                      outDispersion=(1.23, 4.321, 7.654)
                      )
    def testVariance(self, inDispersion, outDispersion):
        """Test interpolateVariance"""
        flux = 10000.0
        noise = 30.0
        inSpectrum = self.makeSpectrum(inDispersion, flux)
        expected = self.makeSpectrum(outDispersion, flux)
        rng = np.random.RandomState(12345)
        inSpectrum.flux += rng.normal(scale=noise, size=inSpectrum.flux.shape)
        inVariance = np.full_like(inSpectrum.flux, noise**2)

        outFlux = interpolateFlux(
            inSpectrum.wavelength, inSpectrum.flux, expected.wavelength, order=self.order
        )
        outVariance = interpolateVariance(
            inSpectrum.wavelength, inVariance, expected.wavelength, order=self.order
        )

        mean = np.average(outFlux)
        stdChi = np.std((outFlux - flux)/np.sqrt(outVariance))
        self.assertFloatsAlmostEqual(mean, flux, rtol=1.5e-3)
        self.assertFloatsAlmostEqual(stdChi, 1.0, rtol=0.15)

    @methodParameters(
        inDispersion=(3.0, 5.0, 10.0),
        outDispersion=(1.23, 4.321, 7.654)
    )
    def testInterpolate(self, inDispersion: float, outDispersion: float):
        """Test general-purpose interpolate function"""
        flux = 10000.0
        noise = 30.0
        inSpectrum = self.makeSpectrum(inDispersion, flux)
        expected = self.makeSpectrum(outDispersion, flux)
        rng = np.random.RandomState(12345)
        inSpectrum.flux += rng.normal(scale=noise, size=inSpectrum.flux.shape)
        inVariance = np.full_like(inSpectrum.flux, noise**2)

        result = interpolate(
            inSpectrum.wavelength,
            inSpectrum.flux,
            np.zeros_like(inSpectrum.flux, dtype=int),
            inVariance,
            expected.wavelength,
            order=self.order,
        )
        outFlux = result.flux
        outVariance = result.variance

        mean = np.average(outFlux)
        stdChi = np.std((outFlux - flux)/np.sqrt(outVariance))
        self.assertFloatsAlmostEqual(mean, flux, rtol=1.5e-3)
        self.assertFloatsAlmostEqual(stdChi, 1.0, rtol=0.15)

    @methodParameters(dtype=("float32", "float64"))  # strings, so pytest-xdist can serialize subtests
    def testMaskedVariance(self, dtype: str):
        """Test that interpolateMaskedVariance matches interpolate"""
        rng = np.random.default_rng(12345)
        inSpectrum = self.makeSpectrum(5.0)
        expected = self.makeSpectrum(4.321)
        size = inSpectrum.wavelength.size
        flux = rng.normal(size=size).astype(dtype)
        variance = rng.uniform(1.0, 2.0, size=size).astype(dtype)
        mask = np.where(rng.uniform(size=size) < 0.05, 1, 0).astype(np.int32)
        mask[size//2:size//2 + 3] = 1  # failed interpolation, flagged with fillMask
        kwargs = dict(order=self.order, minWeight=0.5, badMask=1, fillMask=2, numCovar=self.order + 1)

        full = interpolate(inSpectrum.wavelength, flux, mask, variance, expected.wavelength, **kwargs)
        result = interpolateMaskedVariance(
            inSpectrum.wavelength, mask, variance, expected.wavelength, **kwargs
        )
        self.assertTrue(np.any(result.mask & 2))
        self.assertFloatsEqual(result.mask, full.mask)
        self.assertFloatsEqual(result.covariance, full.covariance)
        self.assertFloatsEqual(result.variance, full.variance)
        self.assertEqual(result.covariance.dtype, np.dtype(dtype))

    def testLsst(self):
        """Test that we get the same result as lsst.afw.math.offsetImage"""
        algorithm = "bilinear" if self.order <= 1 else f"lanczos{self.order}"

        sigma = 3.21
        size = 101
        center = (size // 2) + 0.321
        shift = 1.25

        xx = np.arange(size, dtype=float)
        yy = np.exp(-0.5*((xx - center)/sigma)**2)

        # Make an image containing the spectrum, and use LSST's offsetImage to shift it
        array = np.zeros((2*self.order + 1, size), dtype=float)
        array[self.order] = yy
        image = ImageD(array)
        offset = offsetImage(image, shift, 0, algorithm)
        x0 = offset.getX0()
        lsst = np.zeros_like(yy)
        lsst[x0:] = offset.array[self.order][:-x0]

        pfs = interpolateFlux(xx, yy, xx - shift, order=self.order)

        select = slice(self.order, -self.order)
        self.assertFloatsAlmostEqual(pfs[select], lsst[select], atol=1.0e-6)

    def testGaussian(self):
        """Test interpolateFlux with a Gaussian"""
        sigma = 3.21
        size = 101
        center = (size // 2) + 0.321
        shift = 1.25

        xx = np.arange(size, dtype=float)
        yy = np.exp(-0.5*((xx - center)/sigma)**2)

        result = interpolateFlux(xx, yy, xx - shift, order=self.order)

        result[0:self.order] = 0.0  # Ignore edge effects
        result[-self.order:] = 0.0
        expected = np.exp(-0.5*((xx - center - shift)/sigma)**2)

        if False:
            import matplotlib.pyplot as plt
            plt.plot(xx, yy, "r-", label="input")
            plt.plot(xx, expected, "b-", label="expected")
            plt.plot(xx, result, "k-", label="result")
            plt.axvline(center, color="k", ls=":")
            plt.legend()
            plt.ylim(0, 1)
            plt.show()

        moment1 = np.sum(result*xx)/np.sum(result) - center
        moment2 = np.sqrt(np.sum(result*(xx - center - moment1)**2)/np.sum(result))

        if self.order <= 1:
            # Bilinear gets the centroid right, but grows the width
            atol1 = 1.0e-6
            atol2 = 3.0e-2
        else:
            # Lanczos isn't as good on centroid, but the width is much better
            atol1 = 2.0e-2
            atol2 = 1.0e-4

        self.assertFloatsAlmostEqual(result, expected, atol=1.0e-2)
        self.assertFloatsAlmostEqual(moment1, shift, atol=atol1)
        self.assertFloatsAlmostEqual(moment2, sigma, atol=atol2)

    @methodParameters(
        inDispersion=(3.0, 5.0, 10.0),
        outDispersion=(1.23, 4.321, 7.654)
    )
    def testCovariance(self, inDispersion: float, outDispersion: float):
        """Test interpolateCovariance function"""
        if inDispersion == 3:
            return
        fluxValue = 100.0
        varianceValue = 54.321
        numSamples = 10000
        numCovar = 2*(self.order + int(inDispersion/outDispersion + 0.5))

        inSpectrum = self.makeSpectrum(inDispersion, fluxValue)
        variance = np.full_like(inSpectrum.flux, varianceValue)
        outSpectrum = self.makeSpectrum(outDispersion, fluxValue)

        covar = interpolateCovariance(
            inSpectrum.wavelength, variance, outSpectrum.wavelength, order=self.order, numCovar=numCovar
        )
        self.assertEqual(covar.shape, (numCovar, outSpectrum.wavelength.size))

        rng = np.random.RandomState(12345)
        samples = np.zeros((numSamples, outSpectrum.wavelength.size), dtype=float)
        for ii in range(numSamples):
            flux = rng.normal(fluxValue, np.sqrt(varianceValue), size=inSpectrum.wavelength.size)
            samples[ii] = interpolateFlux(
                inSpectrum.wavelength, flux, outSpectrum.wavelength, order=self.order
            )
        residuals = samples - fluxValue

        outVar = interpolateVariance(
            inSpectrum.wavelength, variance, outSpectrum.wavelength, order=self.order
        )
        measVar = np.nanvar(residuals, axis=0)
        self.assertFloatsAlmostEqual(covar[0], outVar, rtol=1.0e-6)
        self.assertFloatsAlmostEqual(measVar, outVar, rtol=5.0e-2, atol=1.0e-2*varianceValue)

        for offset in range(numCovar):
            calculated = covar[offset]
            measured = np.mean(residuals[:, :-offset or None]*residuals[:, offset or None:], axis=0)
            self.assertFloatsAlmostEqual(
                measured, calculated[:-offset or None], rtol=0.1, atol=0.05*varianceValue
            )

        # Check eigenvalues of covariance matrix
        length = outSpectrum.wavelength.size
        matrix = np.zeros((length, length), dtype=float)
        for offset in range(numCovar):
            matrix += np.diag(covar[offset, :length - offset], k=offset)
            if offset != 0:
                matrix += np.diag(covar[offset, :length - offset], k=-offset)
        eigvals = np.linalg.eigvals(matrix)

        # Eigenvalues may be zero, because there are more output pixels than input pixels.
        self.assertTrue(np.all(np.imag(eigvals) < 1.0e-12))

        # Eigenvalues should be non-negative
        # They may be negative if numCovar is too small.
        self.assertTrue(np.all(np.real(eigvals) > -1.0e-12))


class ResamplerTestCase(lsst.utils.tests.TestCase):
    """Claude: Resampler applies the spectra's resampling to other rows (PIPE2D-1993)"""

    def setUp(self):
        self.rng = np.random.default_rng(54321)
        numIn = 500
        # a non-uniform source grid, resampled onto a finer one that extends beyond it at both ends
        self.fromWavelength = 600.0 + 0.1*np.arange(numIn) + 1.0e-5*np.arange(numIn)**2
        self.toWavelength = np.linspace(599.5, self.fromWavelength[-1] + 0.5, 700)

    def testSameAsFunctions(self):
        """Resampler gives the same results as interpolate and interpolateFlux"""
        numIn = len(self.fromWavelength)
        flux = self.rng.normal(size=numIn)
        variance = self.rng.uniform(1.0, 2.0, size=numIn)
        mask = (self.rng.random(numIn) < 0.05).astype(np.int32)
        for order in (1, 3):
            resampler = Resampler(self.fromWavelength, self.toWavelength, order=order)
            expect = interpolate(self.fromWavelength, flux, mask, variance, self.toWavelength, order=order,
                                 minWeight=0.5, badMask=1, fillMask=2, numCovar=order + 1)
            result = resampler(flux, mask, variance, minWeight=0.5, badMask=1, fillMask=2, numCovar=order + 1)
            for name in ("flux", "mask", "covariance"):
                np.testing.assert_array_equal(getattr(result, name), getattr(expect, name))
            np.testing.assert_array_equal(
                resampler.resampleFlux(flux),
                interpolateFlux(self.fromWavelength, flux, self.toWavelength, order=order),
            )

    def testFloatPrecision(self):
        """Claude: float32 values are resampled at double-precision positions

        The source index used to be cast to the type of the values, which for
        float32 rounds it by up to ~1e-4 pixels at indices of a few thousand.
        """
        numIn = 4096
        fromWavelength = 600.0 + 0.0731*np.arange(numIn)
        toWavelength = np.linspace(601.0, fromWavelength[-1] - 1.0, 3000)
        flux = np.sin(np.arange(numIn)/2.0)  # Large gradient, so position errors matter
        variance = np.ones(numIn)
        mask = np.zeros(numIn, dtype=np.int32)
        for order in (1, 3):
            resampler = Resampler(fromWavelength, toWavelength, order=order)
            expect = resampler.resampleFlux(flux)  # double precision
            result = resampler(flux.astype(np.float32), mask, variance.astype(np.float32))
            np.testing.assert_allclose(result.flux, expect, rtol=0, atol=1.0e-6)

    def testWindow(self):
        """resampleWindow is resampleFlux of a row that is zero outside the window"""
        numIn = len(self.fromWavelength)
        for order in (1, 3, 5):
            resampler = Resampler(self.fromWavelength, self.toWavelength, order=order)
            for start, size in ((200, 11), (0, 7), (numIn - 9, 9), (3, 1), (numIn - 1, 1)):
                values = self.rng.uniform(0.5, 1.5, size=size)
                row = np.zeros(numIn)
                row[start:start + size] = values
                expect = resampler.resampleFlux(row)
                outStart, result = resampler.resampleWindow(values, start)
                full = np.zeros_like(expect)
                full[outStart:outStart + len(result)] = result
                np.testing.assert_array_equal(full, expect, err_msg=f"order={order} start={start}")
                self.assertGreater(len(result), 0)


class ResamplerCovarianceTestCase(lsst.utils.tests.TestCase):
    """Claude: the resampling as an operator, and the propagation of correlated noise"""

    def setUp(self):
        self.rng = np.random.default_rng(24680)
        numIn = 400
        self.fromWavelength = 600.0 + 0.067*np.arange(numIn)
        # ~0.07 nm: once resampled; then ~0.088 nm: a scale change
        self.midWavelength = np.linspace(600.5, self.fromWavelength[-1] - 0.5, 380)
        self.toWavelength = np.linspace(601.0, self.fromWavelength[-1] - 1.0, 300)

    def testMatrix(self):
        """The operator reproduces the C++ resampling: values and covariance"""
        numIn = len(self.fromWavelength)
        flux = self.rng.normal(size=numIn)
        variance = self.rng.uniform(1.0, 2.0, size=numIn)
        mask = (self.rng.random(numIn) < 0.05).astype(np.int32)
        for order in (1, 3, 5):
            numCovar = 2*order + 2
            resampler = Resampler(self.fromWavelength, self.toWavelength, order=order)
            expect = resampler(flux, mask, variance, minWeight=0.0, badMask=1, numCovar=numCovar)
            matrix = resampler.matrix(mask, 1)
            np.testing.assert_allclose(matrix @ flux, expect.flux, rtol=0, atol=1.0e-12,
                                       err_msg=f"order={order}")
            covar = resampler.propagateCovariance(variance[np.newaxis, :], mask, 1, numCovar)
            # The C++ gives garbage off-diagonals for pairs including an output pixel whose kernel sums to
            # <= 0 after masking (it is masked, but its unnormalised kernel is still used); skip those pairs
            empty = np.diff(matrix.indptr) == 0
            numOut = len(self.toWavelength)
            for kk in range(numCovar):
                select = ~(empty[:numOut - kk] | empty[kk:])
                np.testing.assert_allclose(
                    covar[kk, :numOut - kk][select], expect.covariance[kk, :numOut - kk][select],
                    rtol=1.0e-10, atol=1.0e-12, err_msg=f"order={order} offset={kk}",
                )

    def testCorrelated(self):
        """Correlated input (resampled twice, with a scale change): K C K^T, against Monte Carlo"""
        numIn = len(self.fromWavelength)
        numSamples = 20000
        for order in (1, 3):
            first = Resampler(self.fromWavelength, self.midWavelength, order=order)
            second = Resampler(self.midWavelength, self.toWavelength, order=order)
            numMid = len(self.midWavelength)
            numDiag = 2*order + 2
            # Covariance after the first resampling (unit-variance, independent input)
            midCovar = first.propagateCovariance(np.ones((1, numIn)), numCovar=numDiag)
            self.assertGreater(np.max(np.abs(midCovar[1])), 0.05)  # correlated
            # Exact, against a dense calculation
            numCovar = 2*order + 2
            result = second.propagateCovariance(midCovar, numCovar=numCovar)
            dense = np.zeros((numMid, numMid))
            for kk in range(numDiag):
                dense += np.diag(midCovar[kk, :numMid - kk], k=kk)
                if kk > 0:
                    dense += np.diag(midCovar[kk, :numMid - kk], k=-kk)
            operator = second.matrix().toarray()
            expect = operator @ dense @ operator.T
            for kk in range(numCovar):
                np.testing.assert_allclose(result[kk, :len(self.toWavelength) - kk], np.diagonal(expect, kk),
                                           rtol=0, atol=1.0e-12, err_msg=f"order={order} offset={kk}")
            # Through __call__
            called = second(np.zeros(numMid), np.zeros(numMid, dtype=np.int32), midCovar[0],
                            numCovar=numCovar, covariance=midCovar)
            np.testing.assert_allclose(called.covariance, result, rtol=0, atol=1.0e-12)
            # Without the off-diagonals, the C++ path (uncorrelated) is used, and differs
            uncorrelated = second(np.zeros(numMid), np.zeros(numMid, dtype=np.int32), midCovar[0],
                                  numCovar=numCovar)
            self.assertGreater(np.max(np.abs(uncorrelated.covariance - result)), 0.05)
            # Monte Carlo:  noise resampled twice
            noise = self.rng.normal(size=(numSamples, numIn))
            mid = (first.matrix() @ noise.T).T
            out = (second.matrix() @ mid.T).T
            interior = slice(2*order + 2, len(self.toWavelength) - 3*order - 3)
            for kk in range(numCovar):
                measured = np.mean(out[:, :len(self.toWavelength) - kk]*out[:, kk:], axis=0)
                np.testing.assert_allclose(
                    measured[interior], result[kk, :len(self.toWavelength) - kk][interior],
                    rtol=0, atol=0.05, err_msg=f"order={order} offset={kk}",
                )

    def testGuard(self):
        """Resampling a model variance for correlated spectra is refused"""
        from pfs.datamodel import Identity, MaskHelper
        from pfs.drp.stella.datamodel import PfsFiberArraySet
        numIn = len(self.fromWavelength)
        flags = MaskHelper(NO_DATA=0)
        covar = np.zeros((2, 3, numIn), dtype=np.float32)
        covar[:, 0] = 1.0
        spectra = PfsFiberArraySet(
            Identity(visit=1, arm="r", spectrograph=1), np.array([1, 2]),
            np.array([self.fromWavelength]*2), np.ones((2, numIn), dtype=np.float32),
            np.zeros((2, numIn), dtype=np.int32), np.zeros((2, numIn), dtype=np.float32),
            np.ones((2, numIn), dtype=np.float32), covar, flags, {},
        )
        variance = np.ones((2, numIn))
        spectra.resampleVariance(self.toWavelength, variance)  # uncorrelated: fine
        covar[:, 1] = 0.2  # correlated
        with self.assertRaises(NotImplementedError):
            spectra.resampleVariance(self.toWavelength, variance)
        # The spectra themselves are resampled with their correlations
        resampled = spectra.resample(self.toWavelength, order=1, numCovar=2)
        resampler = Resampler(self.fromWavelength, self.toWavelength, order=1)
        expect = resampler.propagateCovariance(covar[0, :2], numCovar=2)
        np.testing.assert_allclose(resampled.covar[0], expect, rtol=1.0e-6, atol=1.0e-6)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    runTests(globals())
