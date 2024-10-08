import sys
import unittest
import pickle

import matplotlib
matplotlib.use("Agg")  # noqa

import numpy as np  # noqa E402: import after code

import lsst.utils.tests  # noqa E402: import after code
import lsst.afw.image  # noqa E402: import after code
import lsst.afw.image.testUtils  # noqa E402: import after code
from lsst.daf.base import PropertySet  # noqa E402: import after code

from pfs.drp.stella import Spectrum  # noqa E402: import after code

display = None


class SpectrumTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(54321)

        # Spectrum
        self.length = 123
        self.fiberId = 456
        self.image = self.rng.uniform(size=self.length).astype(np.float32)
        self.mask = lsst.afw.image.Mask(self.length, 1)
        self.mask.array[:] = self.rng.randint(0, 2**30, self.length).astype(np.int32)
        self.norm = self.rng.uniform(size=self.length).astype(np.float32)
        self.variance = self.rng.uniform(size=self.length).astype(np.float32)
        self.wavelengthArray = np.arange(1, self.length + 1, dtype=float)
        self.notes = {"foo": 123, "bar": 4.56}

    def makeSpectrum(self):
        """Make a ``Spectrum`` for testing"""
        return Spectrum(self.image, self.mask, self.norm, self.variance,
                        self.wavelengthArray, self.fiberId, PropertySet.from_mapping(self.notes))

    def assertSpectrum(self, spectrum):
        """Assert that the ``Spectrum`` has the expected contents"""
        self.assertEqual(len(spectrum), self.length)
        self.assertEqual(spectrum.fiberId, self.fiberId)
        self.assertFloatsEqual(spectrum.flux, self.image)
        self.assertImagesEqual(spectrum.mask, self.mask)
        self.assertFloatsEqual(spectrum.variance, self.variance)
        self.assertFloatsEqual(spectrum.norm, self.norm)
        self.assertFloatsEqual(spectrum.wavelength, self.wavelengthArray)
        self.assertDictEqual(spectrum.notes.toDict(), self.notes)

    def testCreateEmpty(self):
        """Test creation of an empty ``Spectrum``

        Since it's empty, we also use the opportunity to test the setters.
        """
        spectrum = Spectrum(self.length, 0)
        self.assertEqual(len(spectrum), self.length)
        self.assertEqual(spectrum.fiberId, 0)
        self.assertEqual(spectrum.flux.shape, (self.length,))
        self.assertEqual(spectrum.mask.getHeight(), 1)
        self.assertEqual(spectrum.mask.getWidth(), self.length)
        self.assertEqual(spectrum.variance.shape, (self.length,))
        self.assertEqual(spectrum.norm.shape, (self.length,))
        self.assertEqual(spectrum.wavelength.shape, (self.length,))
        self.assertEqual(len(spectrum.notes), 0)

        # Set elements via setters
        spectrum.setFiberId(self.fiberId)
        spectrum.setFlux(self.image)
        spectrum.setMask(self.mask)
        spectrum.setNorm(self.norm)
        spectrum.setVariance(self.variance)
        spectrum.setWavelength(self.wavelengthArray)
        spectrum.getNotes().update(self.notes)
        self.assertSpectrum(spectrum)

        # Reset, and set elements via properties
        spectrum = Spectrum(self.length, 0)
        spectrum.fiberId = self.fiberId
        spectrum.flux = self.image
        spectrum.mask = self.mask
        spectrum.norm = self.norm
        spectrum.variance = self.variance
        spectrum.wavelength = self.wavelengthArray
        spectrum.notes.update(self.notes)
        self.assertSpectrum(spectrum)

        # Change versions in self and ensure the values in spectrum DO NOT change (setters do a copy)
        self.image[:] = self.rng.uniform(size=self.length)
        self.mask[:] = lsst.afw.image.Mask(self.length, 1)
        self.mask.array[:] = self.rng.randint(0, 2**30, self.length).astype(np.int32)
        self.norm[:] = self.rng.uniform(size=self.length)
        self.variance[:] = self.rng.uniform(size=self.length)
        self.wavelengthArray[:] = self.rng.uniform(size=self.length)
        self.assertFloatsNotEqual(spectrum.flux, self.image)
        self.assertFloatsNotEqual(spectrum.mask.array, self.mask.array)
        self.assertFloatsNotEqual(spectrum.variance, self.variance)
        self.assertFloatsNotEqual(spectrum.norm, self.norm)
        self.assertFloatsNotEqual(spectrum.wavelength, self.wavelengthArray)

    def testBasics(self):
        """Test the basics of ``Spectrum``

        This includes the constructor (through ``makeSpectrum``), the
        getters (through ``assertSpectrum`` and here).
        """
        spectrum = self.makeSpectrum()
        self.assertSpectrum(spectrum)

        # Getters instead of properties
        self.assertEqual(spectrum.getFiberId(), self.fiberId)
        self.assertFloatsEqual(spectrum.getFlux(), self.image)
        self.assertImagesEqual(spectrum.getMask(), self.mask)
        self.assertFloatsEqual(spectrum.getVariance(), self.variance)
        self.assertFloatsEqual(spectrum.getNorm(), self.norm)
        self.assertFloatsEqual(spectrum.getVariance(), self.variance)
        self.assertFloatsEqual(spectrum.getWavelength(), self.wavelengthArray)
        self.assertFloatsEqual(spectrum.getNormFlux(), self.image/self.norm)

        # Change versions in self and ensure the values in spectrum change (pointers)
        self.image[:] = self.rng.uniform(size=self.length)
        self.mask[:] = lsst.afw.image.Mask(self.length, 1)
        self.mask.array[:] = self.rng.randint(0, 2**31, self.length).astype(np.int32)
        self.norm[:] = self.rng.uniform(size=self.length)
        self.variance[:] = self.rng.uniform(size=self.length)
        self.wavelengthArray[:] = self.rng.uniform(size=self.length)
        self.assertFloatsEqual(spectrum.flux, self.image)
        self.assertFloatsEqual(spectrum.mask.array, self.mask.array)
        self.assertFloatsEqual(spectrum.variance, self.variance)
        self.assertFloatsEqual(spectrum.wavelength, self.wavelengthArray)
        self.assertFloatsEqual(spectrum.normFlux, self.image/self.norm)

    def testPickle(self):
        """Test pickling of ``Spectrum``"""
        spectrum = self.makeSpectrum()
        self.assertSpectrum(spectrum)
        copy = pickle.loads(pickle.dumps(spectrum))
        self.assertSpectrum(copy)

    def testPlot(self):
        """Test plotting of spectrum

        Not easy to test the actual result, but we can test that the API hasn't
        been broken.
        """
        import matplotlib.pyplot as plt
        plt.switch_backend("agg")  # In case someone has loaded a different backend that will cause trouble
        ext = ".png"  # Extension to use for plot filenames
        spectrum = self.makeSpectrum()
        # Write directly to file
        with lsst.utils.tests.getTempFilePath(ext) as filename:
            spectrum.plot(numRows=4, filename=filename)
        # Check return values
        with lsst.utils.tests.getTempFilePath(ext) as filename:
            numRows = 4  # Must be > 1 for len(axes) to work
            figure, axes = spectrum.plot(numRows=numRows)
            self.assertEqual(len(axes), numRows)
            figure.savefig(filename)
        # Test one row, write directly to file
        with lsst.utils.tests.getTempFilePath(ext) as filename:
            figure, axes = spectrum.plot(numRows=1, filename=filename)
        # Test one row, check return values
        with lsst.utils.tests.getTempFilePath(ext) as filename:
            figure, axes = spectrum.plot(numRows=1)
            with self.assertRaises(TypeError):
                axes[0]
            figure.savefig(filename)

    def testWavelength(self):
        """Test conversion pixels <--> wavelength"""
        spectrum = Spectrum(self.length, 0)
        indices = np.arange(self.length, dtype=float)
        wl0 = 500.0
        wlSlope = 3.21
        spectrum.setWavelength(indices*wlSlope + wl0)

        num = 50
        rng = np.random.RandomState(12345)
        pixels = rng.uniform(size=num)*(self.length - 1)
        wavelength = wlSlope*pixels + wl0
        self.assertFloatsAlmostEqual(spectrum.wavelengthToPixels(wavelength), pixels, atol=1.0e-5)
        self.assertFloatsAlmostEqual(spectrum.pixelsToWavelength(pixels), wavelength, atol=1.0e-4)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    from argparse import ArgumentParser
    parser = ArgumentParser(__file__)
    parser.add_argument("--display", help="Display backend")
    args, argv = parser.parse_known_args()
    display = args.display
    unittest.main(failfast=True, argv=[__file__] + argv)
