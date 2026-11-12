import numpy as np

import lsst.utils.tests

from pfs.drp.stella import Spectrum
from pfs.drp.stella.fitContinuum import FitContinuumConfig, FitContinuumTask
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
