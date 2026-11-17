import numpy as np

import lsst.utils.tests

from pfs.drp.stella import Spectrum, SpectrumSet
from pfs.drp.stella.fitContinuum import FitContinuumConfig, FitContinuumTask, binData, binDataByKnots
from pfs.drp.stella.fitContinuum import wavelengthToPixels
from pfs.drp.stella.tests.utils import runTests

display = None


class FitContinuumTestCase(lsst.utils.tests.TestCase):
    def makeSpectrum(self) -> Spectrum:
        """Make a flat spectrum with deep telluric A and B bands"""
        length = 2000
        spectrum = Spectrum(length)
        spectrum.setWavelength(np.linspace(650.0, 950.0, length))
        spectrum.flux[:] = 1000.0
        spectrum.mask[:] = 0
        spectrum.variance[:] = 1.0
        for low, high in ((686.5, 697.5), (758.2, 773.0)):
            select = (spectrum.wavelength >= low) & (spectrum.wavelength <= high)
            spectrum.flux[select] *= 0.2
        return spectrum

    def fit(self, spectrum: Spectrum, config: FitContinuumConfig) -> np.ndarray:
        """Fit the continuum of a spectrum with the provided config"""
        config.doMaskLines = False
        config.validate()
        return FitContinuumTask(config=config).fitContinuum(spectrum)

    def testTelluric(self):
        """Test that the telluric bands are masked when fitting the continuum"""
        spectrum = self.makeSpectrum()
        continuum = self.fit(spectrum, FitContinuumConfig())
        self.assertFloatsAlmostEqual(continuum, 1000.0, rtol=1.0e-3)

        # Without masking the telluric bands, the continuum is dragged down by the absorption
        config = FitContinuumConfig()
        config.telluricCenter = []
        config.telluricWidth = []
        continuum = self.fit(spectrum, config)
        self.assertLess(np.min(continuum), 900.0)

    def testBinData(self):
        """Test that binData calculates medians as np.nanmedian does"""
        rng = np.random.default_rng(12345)
        for length, numBins in ((1000, 50), (1003, 50), (11251, 50), (100, 30)):
            for dtype in (np.float32, np.float64):
                xx = np.arange(length, dtype=dtype)
                yy = rng.normal(size=length).astype(dtype)
                good = rng.uniform(size=length) > 0.3
                good[: length // 5] = False  # Some bins with no good values
                xBinned, yBinned = binData(xx, yy, good, numBins)
                self.assertEqual(xBinned.dtype, dtype)
                self.assertEqual(yBinned.dtype, dtype)

                # Reference implementation
                lenWide = (length + numBins - 1) // numBins * numBins
                numExtra = lenWide - length
                select = np.ones(lenWide, dtype=bool)
                if numExtra > 0:
                    indexExtra = np.around(
                        (lenWide - 1) / numExtra * (np.arange(numExtra).astype(np.float32) + 0.5)
                    ).astype(int)
                    select[indexExtra] = False
                for values, binned in ((xx, xBinned), (yy, yBinned)):
                    wide = np.full(lenWide, np.nan, dtype=dtype)
                    wide[select] = np.where(good, values, np.nan)
                    with np.testing.suppress_warnings() as sup:
                        sup.filter(RuntimeWarning, "All-NaN slice encountered")
                        expected = np.nanmedian(wide.reshape(numBins, -1), axis=1)
                    np.testing.assert_array_equal(binned, expected)

    def testBinDataByKnots(self):
        """Test binning around nominated knots"""
        rng = np.random.default_rng(54321)
        length = 1000
        xx = np.arange(length, dtype=float)
        yy = rng.normal(size=length)
        good = rng.uniform(size=length) > 0.3
        good[400:500] = False  # A bin with no good values
        knots = np.array([30.0, 200.0, 450.0, 470.0, 800.0])
        edges = [-np.inf, 115.0, 325.0, 460.0, 635.0, np.inf]
        for fixed in (False, True):
            xBinned, yBinned = binDataByKnots(xx, yy, good, knots, fixed)
            for ii in range(len(knots)):
                select = good & (xx >= edges[ii]) & (xx < edges[ii + 1])
                if not np.any(select):
                    self.assertTrue(np.isnan(xBinned[ii]))
                    self.assertTrue(np.isnan(yBinned[ii]))
                    continue
                self.assertEqual(yBinned[ii], np.median(yy[select]))
                self.assertEqual(xBinned[ii], knots[ii] if fixed else np.median(xx[select]))

    def testKnots(self):
        """Test fitting with nominated knots"""
        spectrum = self.makeSpectrum()
        length = spectrum.getNumPixels()
        expected = 1000.0 + 0.1*np.arange(length)
        # Noise is needed for the rejection to work (the scatter must not be zero)
        spectrum.flux[:] = expected + np.random.default_rng(12345).normal(0.0, 0.1, length)
        for knotPosition in ("median", "fixed"):
            config = FitContinuumConfig()
            config.knotPosition = knotPosition
            config.doMaskLines = False
            config.telluricCenter = []
            config.telluricWidth = []
            task = FitContinuumTask(config=config)
            for knots in (np.linspace(50, 1950, 20), None):  # None means bins of equal size
                continuum = task.fitContinuum(spectrum, knots=knots)
                # Only test between the outermost knots: the interpolation doesn't extrapolate the slope
                self.assertFloatsAlmostEqual(continuum[100:-100], expected[100:-100], atol=0.3)

    def testCommonKnots(self):
        """Test that run places knots at the same wavelengths for all spectra"""
        length = 2000
        spectra = SpectrumSet(length)
        for start in (650.0, 655.5, 661.0):
            spectrum = Spectrum(length)
            spectrum.setWavelength(np.linspace(start, start + 300.0, length))
            spectrum.flux[:] = 1000.0
            spectrum.mask[:] = 0
            spectrum.variance[:] = 1.0
            spectra.add(spectrum)

        task = FitContinuumTask()
        knotWavelengths = task.getKnotWavelengths(spectra)
        self.assertEqual(len(knotWavelengths), task.config.numKnots)
        binWidth = (961.0 - 650.0) / task.config.numKnots
        self.assertFloatsAlmostEqual(knotWavelengths[0], 650.0 + 0.5*binWidth, atol=1.0e-9)
        self.assertFloatsAlmostEqual(knotWavelengths[-1], 961.0 - 0.5*binWidth, atol=1.0e-9)

        for spectrum in spectra:
            knots = wavelengthToPixels(spectrum.wavelength, knotWavelengths)
            # Knots outside the range of the spectrum are dropped
            wavelength = spectrum.wavelength
            inside = (knotWavelengths >= wavelength[0]) & (knotWavelengths <= wavelength[-1])
            self.assertEqual(len(knots), inside.sum())
            backToWavelength = np.interp(knots, np.arange(length), spectrum.wavelength)
            self.assertFloatsAlmostEqual(backToWavelength, knotWavelengths[inside], atol=1.0e-9)

        continua = task.run(spectra)
        self.assertFloatsAlmostEqual(continua, 1000.0, rtol=1.0e-6)

    def testGoodPixels(self):
        """Test reporting and supplying the pixels used in the fit"""
        rng = np.random.default_rng(13579)
        spectrum = self.makeSpectrum()
        length = spectrum.getNumPixels()
        spectrum.flux[:] += rng.normal(0.0, 1.0, length)
        spectrum.flux[1000:1010] += 500.0  # An emission line, which should be rejected
        config = FitContinuumConfig()
        config.doMaskLines = False
        task = FitContinuumTask(config=config)

        result = task.measureContinuum(spectrum)
        self.assertFloatsEqual(result.continuum, task.fitContinuum(spectrum))
        self.assertFalse(np.any(result.good[1000:1010]))  # Line rejected
        self.assertFalse(np.all(result.good))  # Telluric bands excluded

        # Supplying the pixels reproduces the fit exactly
        again = task.measureContinuum(spectrum, good=result.good)
        self.assertFloatsEqual(again.continuum, result.continuum)
        np.testing.assert_array_equal(again.good, result.good)

        # With the pixels supplied, there's no rejection, so the continuum is
        # a continuous function of the flux: a small change in the flux makes
        # a small change in the continuum.
        perturbed = self.makeSpectrum()
        perturbed.flux[:] = spectrum.flux + 1.0e-3*rng.normal(size=length)
        fixedMask = task.measureContinuum(perturbed, good=result.good)
        self.assertFloatsAlmostEqual(fixedMask.continuum, result.continuum, atol=1.0e-2)

        # A set of spectra
        spectra = SpectrumSet(length)
        spectra.add(spectrum)
        spectra.add(perturbed)
        measured = task.measureContinua(spectra)
        self.assertFloatsEqual(measured.continua, task.run(spectra))
        self.assertEqual(measured.good.shape, (2, length))
        again = task.measureContinua(spectra, good=measured.good)
        self.assertFloatsEqual(again.continua, measured.continua)

    def testValidate(self):
        """Test that mismatched telluric lists are rejected"""
        config = FitContinuumConfig()
        config.telluricWidth = [1.0]
        with self.assertRaises(RuntimeError):
            config.validate()


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
