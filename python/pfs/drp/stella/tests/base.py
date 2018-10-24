"""
``BaseTestCase`` is a base class for testing the ``ReferenceLine``,
``Spectrum`` and ``SpectrumSet`` classes.
"""

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.daf.base
import lsst.afw.image
import lsst.afw.image.testUtils

from pfs.drp.stella import ReferenceLine, Spectrum

__all__  = ["BaseTestCase"]


class BaseTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(54321)

        # Spectrum
        self.length = 123
        self.fiberId = 456
        self.image = self.rng.uniform(size=self.length).astype(np.float32)
        self.mask = lsst.afw.image.Mask(self.length, 1)
        self.mask.array[:] = self.rng.randint(0, 2**30, self.length).astype(np.int32)
        self.background = self.rng.uniform(size=self.length).astype(np.float32)
        self.covariance = self.rng.uniform(size=(3, self.length)).astype(np.float32)
        self.wavelengthArray = np.arange(1, self.length + 1, dtype=np.float32)

        # ReferenceLine
        self.description = "line"
        self.status = ReferenceLine.Status.NOWT
        self.wavelength = 1234.5678
        self.guessedIntensity = 5.4321
        self.guessedPosition = 123.45678
        self.fitIntensity = 6.54321
        self.fitPosition = 234.56789
        self.fitPositionErr = 0.12345
        self.line = self.makeReferenceLine()

    def makeReferenceLine(self):
        """Make a ``ReferenceLine`` with the provided attributes"""
        line = ReferenceLine(self.description, self.status, self.wavelength, self.guessedIntensity)
        for name in ("guessedPosition", "fitIntensity", "fitPosition", "fitPositionErr"):
            setattr(line, name, getattr(self, name))
        return line

    def assertReferenceLine(self, line):
        """Assert that the values are as expected"""
        for name in ("description", "status", "wavelength", "guessedIntensity", "guessedPosition",
                     "fitIntensity", "fitPosition", "fitPositionErr"):
            self.assertEqual(getattr(line, name), getattr(self, name), name)

    def makeSpectrum(self):
        """Make a ``Spectrum`` for testing"""
        return Spectrum(self.image, self.mask, self.background, self.covariance,
                        self.wavelengthArray, [self.line], self.fiberId)

    def assertSpectrum(self, spectrum):
        """Assert that the ``Spectrum`` has the expected contents"""
        self.assertEqual(len(spectrum), self.length)
        self.assertEqual(spectrum.fiberId, self.fiberId)
        self.assertFloatsEqual(spectrum.spectrum, self.image)
        self.assertImagesEqual(spectrum.mask, self.mask)
        self.assertFloatsEqual(spectrum.variance, self.covariance[0])
        self.assertFloatsEqual(spectrum.background, self.background)
        self.assertFloatsEqual(spectrum.covariance, self.covariance)
        self.assertFloatsEqual(spectrum.wavelength, self.wavelengthArray)
        self.assertEqual(len(spectrum.referenceLines), 1)
        self.assertReferenceLine(spectrum.referenceLines[0])
