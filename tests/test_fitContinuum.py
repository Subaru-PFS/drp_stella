import numpy as np

import lsst.utils.tests

from pfs.drp.stella import Spectrum
from pfs.drp.stella.fitContinuum import FitContinuumConfig, FitContinuumTask, binData
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
