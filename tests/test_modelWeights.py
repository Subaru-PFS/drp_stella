import sys
import unittest
from types import SimpleNamespace

import numpy as np

import lsst.utils.tests

from pfs.datamodel import CalibIdentity, Observations, Target
from pfs.datamodel.masks import MaskHelper
from pfs.drp.stella.datamodel.drp import PfsArm, PfsSingle
from pfs.drp.stella.modelWeights import ModelWeightsTask, resampleWeights


class ModelWeightsTestCase(lsst.utils.tests.TestCase):
    """Tests for ModelWeightsTask"""

    def setUp(self):
        self.makeSpectra(10)
        self.task = ModelWeightsTask()

    def makeSpectra(self, numSpectra: int):
        """Construct spectra for testing

        Parameters
        ----------
        numSpectra : `int`
            Number of spectra (repeat observations of the same source).
        """
        self.rng = np.random.default_rng(12345)
        self.numSpectra = numSpectra
        self.length = 4000
        self.readNoise = 6.0  # Read noise per spectral pixel (counts)
        self.wavelength = np.linspace(600, 900, self.length)

        # Expected counts: sky continuum plus sky lines, plus a faint object continuum
        skyCounts = 30.0 + 10.0 * np.sin((self.wavelength - 600) / 50)
        for center in self.rng.uniform(610, 890, size=40):
            skyCounts += self.rng.uniform(20, 2000) * np.exp(-0.5 * ((self.wavelength - center) / 0.15) ** 2)
        objectCounts = 5.0 + 2.0 * (self.wavelength - 600) / 300
        self.skyCounts = skyCounts
        self.objectCounts = objectCounts
        self.expected = skyCounts + objectCounts
        self.trueVariance = self.expected + self.readNoise**2

        # Units: smoothly-varying scale from counts
        self.scale = 0.5 + 0.5 * (self.wavelength - 600) / 300

        shape = (self.numSpectra, self.length)
        counts = self.rng.poisson(self.expected, size=shape) + self.rng.normal(0, self.readNoise, size=shape)
        self.noise = counts - self.expected
        measuredVariance = np.clip(counts, 0, None) + self.readNoise**2  # variance from the data
        self.trueFlux = objectCounts * self.scale
        flux = (counts - skyCounts) * self.scale  # perfect sky subtraction
        sky = np.tile(skyCounts * self.scale, (self.numSpectra, 1))
        covar = np.zeros((self.numSpectra, 3, self.length))
        covar[:, 0, :] = measuredVariance * self.scale**2

        self.flags = MaskHelper(NO_DATA=0, CR=1, BAD_SKY=2)
        mask = np.zeros(shape, dtype=np.int32)
        self.badIndices = self.rng.choice(self.length, size=50, replace=False)
        mask[:, self.badIndices] |= self.flags["CR"]
        flux[:, self.badIndices] += 1.0e4  # these must not affect the fit

        self.spectra = PfsArm(
            CalibIdentity(visit0=12345, arm="r", spectrograph=1, obsDate="2026-11-16"),
            np.arange(1, self.numSpectra + 1, dtype=int),
            np.tile(self.wavelength, (self.numSpectra, 1)),
            flux,
            mask,
            sky,
            np.ones(shape),  # norm
            covar,
            self.flags,
            {},  # metadata
        )

    def coadd(self, weights: np.ndarray) -> np.ndarray:
        """Return the weighted average of the spectra"""
        return np.sum(self.spectra.flux * weights, axis=0) / np.sum(weights, axis=0)

    def testInputUnmodified(self):
        """Test that the input spectra are not modified"""
        copies = {
            name: getattr(self.spectra, name).copy() for name in ("flux", "sky", "covar", "mask", "norm")
        }
        self.task.run(self.spectra)
        for name, array in copies.items():
            self.assertFloatsEqual(getattr(self.spectra, name), array, msg=name)

    def testModelVariance(self):
        """Test that the model variance matches the true variance"""
        weights = self.task.run(self.spectra)
        self.assertEqual(weights.shape, self.spectra.flux.shape)
        self.assertTrue(np.all(weights > 0))
        good = np.ones(self.length, dtype=bool)
        good[self.badIndices] = False
        ratio = (1.0 / weights[:, good]) / (self.trueVariance * self.scale**2)[good]
        self.assertFloatsAlmostEqual(np.median(ratio), 1.0, atol=0.02)
        # Away from the sharp sky lines, the model should be good everywhere
        smooth = good & (self.skyCounts < 45.0)
        ratio = (1.0 / weights[:, smooth]) / (self.trueVariance * self.scale**2)[smooth]
        self.assertFloatsAlmostEqual(ratio, 1.0, atol=0.1)

    def testBias(self):
        """Test that weighting doesn't bias the weighted average

        The measured variance is correlated with the noise, and weighting with
        it should produce a coadd that is biased low. The model weights should
        not produce such a bias.
        """
        good = np.ones(self.length, dtype=bool)
        good[self.badIndices] = False
        normalized = self.scale[good]  # Convert from flux units to counts

        with np.errstate(divide="ignore"):
            measuredWeights = 1.0 / self.spectra.variance
        measuredBias = np.mean((self.coadd(measuredWeights) - self.trueFlux)[good] / normalized)
        self.assertLess(measuredBias, -0.5)  # Of order one count; check we can see the problem

        modelWeights = self.task.run(self.spectra)
        modelBias = np.mean((self.coadd(modelWeights) - self.trueFlux)[good] / normalized)
        self.assertLess(abs(modelBias), 0.1)

    def testEdges(self):
        """Test that the weights don't introduce a bias near edges

        The object model is a median filter, which can be dominated by a single
        pixel near the edges (of the array, or of masked regions) if care is
        not taken, making the weights correlated with that pixel's noise.
        """
        self.makeSpectra(200)
        # Masked regions at the ends, as for a single arm resampled onto a wider wavelength grid
        noData = np.zeros(self.length, dtype=bool)
        noData[:500] = True
        noData[-500:] = True
        self.spectra.mask[:, noData] |= self.flags["NO_DATA"]
        self.spectra.flux[:, noData] = 0.0
        self.spectra.covar[:, :, noData] = 0.0

        edges = [index for index in (500, 501, 502, 3497, 3498, 3499) if index not in self.badIndices]
        self.assertGreater(len(edges), 0)

        def calculateBias(weights, index):
            """Bias (counts) in the weighted mean at a pixel, from covariance of weights and noise"""
            ww = weights[:, index]
            return np.mean((ww - ww.mean()) * self.noise[:, index]) / ww.mean()

        with np.errstate(divide="ignore"):
            measuredWeights = 1.0 / self.spectra.variance
        modelWeights = self.task.run(self.spectra)
        for index in edges:
            # Check we can see the problem: of order one count
            self.assertLess(calculateBias(measuredWeights, index), -0.3, msg=f"index={index}")
            # Expect of order 1/smoothWidth
            self.assertLess(abs(calculateBias(modelWeights, index)), 0.1, msg=f"index={index}")

    def testSteepScale(self):
        """Test spectra whose units vary steeply relative to counts

        This is the case for flux-calibrated spectra near a dichroic, where the
        throughput falls steeply at the end of the wavelength range. The
        variance model can't follow that in the units of the spectra, but it
        can if we provide the scale, so it can work in units of counts.
        """
        throughput = np.clip(1.0 / (1.0 + np.exp((self.wavelength - 870.0) / 4.0)), 0.02, None)
        scale = 1.0 / throughput  # counts --> spectrum units
        expected = self.expected * throughput
        shape = (self.numSpectra, self.length)
        counts = self.rng.poisson(expected, size=shape) + self.rng.normal(0, self.readNoise, size=shape)
        self.spectra.flux[:] = (counts - self.skyCounts * throughput) * scale
        self.spectra.sky[:] = self.skyCounts * throughput * scale
        self.spectra.covar[:, 0, :] = (np.clip(counts, 0, None) + self.readNoise**2) * scale**2
        self.spectra.mask[:] = 0
        trueVariance = (expected + self.readNoise**2) * scale**2
        dichroic = self.wavelength > 850

        def getRatio(weights):
            """Return the median ratio of the model to true variance in the dichroic"""
            return np.median((1.0 / weights[:, dichroic]) / trueVariance[dichroic], axis=0)

        # Without the scale, the model fails badly in the dichroic: check we can see the problem
        ratio = getRatio(self.task.run(self.spectra))
        self.assertLess(ratio.min(), 0.5)

        # With the scale, the model is good
        scaleArray = np.tile(scale, (self.numSpectra, 1))
        ratio = getRatio(self.task.run(self.spectra, scaleArray))
        self.assertFloatsAlmostEqual(ratio, 1.0, atol=0.1)

        # Bad scale values get zero weight
        scaleArray[:, :10] = np.nan
        scaleArray[:, 10:20] = 0.0
        weights = self.task.run(self.spectra, scaleArray)
        self.assertFloatsEqual(weights[:, :20], 0.0)
        self.assertTrue(np.all(weights[:, 20:] > 0))

        with self.assertRaises(RuntimeError):
            self.task.run(self.spectra, scale)  # wrong shape

    def testSingleSpectrum(self):
        """Test operation on a single spectrum, as for a PfsFiberArray"""
        weights = self.task.run(self.spectra)
        spectrum = SimpleNamespace(
            wavelength=self.spectra.wavelength[0],
            flux=self.spectra.flux[0],
            sky=self.spectra.sky[0],
            variance=self.spectra.variance[0],
            mask=self.spectra.mask[0],
            flags=self.flags,
        )
        single = self.task.run(spectrum)
        self.assertEqual(single.shape, (self.length,))
        self.assertFloatsEqual(single, weights[0])

    def testBadSpectra(self):
        """Test handling of spectra with little or no good data"""
        self.spectra.mask[0, :] |= self.flags["NO_DATA"]  # No good pixels
        self.spectra.mask[1, 10:] |= self.flags["NO_DATA"]  # Too few good pixels for a model
        self.spectra.flux[2, :] = np.nan  # Non-finite data
        weights = self.task.run(self.spectra)
        self.assertFloatsEqual(weights[0], 0.0)
        self.assertFloatsEqual(weights[2], 0.0)
        # Too few pixels: use a constant, to avoid correlation with the noise
        self.assertFloatsEqual(weights[1], 1.0 / np.median(self.spectra.variance[1, :10]))
        self.assertTrue(np.all(weights[3:] > 0))

    def testResampleWeights(self):
        """Test resampling of weights

        Resampling the inverse of the measured variance should produce the
        inverse of the variance of the resampled spectra.
        """
        wavelength = np.linspace(610, 890, 3000)  # different sampling, with pixels at different phases
        kwargs = dict(order=3, minWeight=0.1, bad=["CR"])
        resampled = self.spectra.resample(wavelength, **kwargs)
        weights = 1.0 / self.spectra.variance
        covar = self.spectra.covar.copy()
        result = resampleWeights(self.spectra, weights, wavelength, **kwargs)
        self.assertEqual(result.shape, resampled.flux.shape)
        # Zero where interpolation failed or was affected by bad pixels
        valid = (resampled.variance > 0) & ((resampled.mask & self.flags.get("NO_DATA")) == 0)
        self.assertGreater(valid.sum(), 0.99 * valid.size)
        self.assertFloatsAlmostEqual(result[valid], 1.0 / resampled.variance[valid], rtol=1.0e-5)
        self.assertFloatsEqual(result[~valid], 0.0)
        self.assertFloatsEqual(self.spectra.covar, covar)  # input unmodified

        # Variance changes with the position of the new pixels relative to the old: the weights should too
        modelWeights = resampleWeights(self.spectra, self.task.run(self.spectra), wavelength, **kwargs)
        self.assertTrue(np.all(modelWeights[valid] > 0))
        good = valid & ((resampled.mask & self.flags.get("CR")) == 0)
        self.assertGreater(np.std(np.diff(np.log(modelWeights[good]))), 0.01)

        # Zero weight: the resampled weights are zero, not contaminated with zeros
        maxWeight = result.max()
        weights[:, 1000:1100] = 0.0
        result = resampleWeights(self.spectra, weights, wavelength, **kwargs)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(result >= 0))
        self.assertLessEqual(result.max(), maxWeight)  # no wild values at the edges of the bad region
        index = np.searchsorted(wavelength, self.wavelength[1050])
        self.assertFloatsEqual(result[:, index], 0.0)
        self.assertGreater(result.max(), 0)

        # Single spectrum: same as for the corresponding fiber in the set
        single = PfsSingle(
            Target(1, 0, "0,0", 1),
            Observations(
                np.array([1]), ["r"], np.array([1]), np.array([1]), self.spectra.fiberId[:1],
                np.zeros((1, 2)), np.zeros((1, 2)), ["2026-11-18T00:00:00"], np.array([900.0]),
            ),
            self.spectra.wavelength[0],
            self.spectra.flux[0],
            self.spectra.mask[0],
            self.spectra.sky[0],
            self.spectra.covar[0],
            np.zeros((1, 1)),
            self.flags,
        )
        # PfsFiberArraySet.resample interpolates in single precision
        self.assertFloatsAlmostEqual(
            resampleWeights(single, weights[0], wavelength, **kwargs), result[0], rtol=1.0e-3
        )
        singleResampled = single.resample(wavelength, **kwargs)
        singleResult = resampleWeights(single, single.variance**-1, wavelength, **kwargs)
        valid = (singleResampled.variance > 0) & ((singleResampled.mask & self.flags.get("NO_DATA")) == 0)
        self.assertFloatsAlmostEqual(singleResult[valid], 1.0 / singleResampled.variance[valid], rtol=1.0e-12)
        self.assertFloatsEqual(singleResult[~valid], 0.0)

        # Single precision, as read from file; and both linear and Lanczos interpolation
        single.flux = single.flux.astype(np.float32)
        single.sky = single.sky.astype(np.float32)
        single.covar = single.covar.astype(np.float32)
        for order in (1, 3):
            orderKwargs = {**kwargs, "order": order}
            singleResult = resampleWeights(single, weights[0], wavelength, **orderKwargs)
            self.assertEqual(singleResult.shape, wavelength.shape)
            self.assertTrue(np.all(np.isfinite(singleResult)))
            self.assertGreater(np.sum(singleResult > 0), 0.9 * wavelength.size)
            expected = resampleWeights(self.spectra, weights, wavelength, **orderKwargs)[0]
            self.assertFloatsAlmostEqual(singleResult, expected, rtol=1.0e-3)

        with self.assertRaises(RuntimeError):
            resampleWeights(self.spectra, weights[0], wavelength)  # wrong shape


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main()
