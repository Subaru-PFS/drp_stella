import sys
import unittest
import pickle

import matplotlib
matplotlib.use("Agg")  # noqa

import numpy as np

import lsst.utils.tests
import lsst.afw.image
import lsst.afw.image.testUtils

import pfs.drp.stella as drpStella
from pfs.drp.stella.tests import BaseTestCase

display = None


class SpectrumTestCase(BaseTestCase):
    def testCreateEmpty(self):
        """Test creation of an empty ``Spectrum``

        Since it's empty, we also use the opportunity to test the setters.
        """
        spectrum = drpStella.Spectrum(self.length, 0)
        self.assertEqual(len(spectrum), self.length)
        self.assertEqual(spectrum.fiberId, 0)
        self.assertEqual(spectrum.spectrum.shape, (self.length,))
        self.assertEqual(spectrum.mask.getHeight(), 1)
        self.assertEqual(spectrum.mask.getWidth(), self.length)
        self.assertEqual(spectrum.variance.shape, (self.length,))
        self.assertEqual(spectrum.background.shape, (self.length,))
        self.assertEqual(spectrum.covariance.shape, (3, self.length))
        self.assertEqual(spectrum.wavelength.shape, (self.length,))
        self.assertListEqual(spectrum.referenceLines, [])

        # Set elements via setters
        spectrum.setFiberId(self.fiberId)
        spectrum.setSpectrum(self.image)
        spectrum.setMask(self.mask)
        spectrum.setBackground(self.background)
        spectrum.setCovariance(self.covariance)
        spectrum.setWavelength(self.wavelengthArray)
        spectrum.setReferenceLines([self.line])
        self.assertSpectrum(spectrum)

        # Reset, and set elements via properties
        spectrum = drpStella.Spectrum(self.length, 0)
        spectrum.fiberId = self.fiberId
        spectrum.spectrum = self.image
        spectrum.mask = self.mask
        spectrum.background = self.background
        spectrum.covariance = self.covariance
        spectrum.wavelength = self.wavelengthArray
        spectrum.referenceLines = [self.line]
        self.assertSpectrum(spectrum)

        # Change versions in self and ensure the values in spectrum DO NOT change (setters do a copy)
        self.image[:] = self.rng.uniform(size=self.length).astype(np.float32)
        self.mask[:] = lsst.afw.image.Mask(self.length, 1)
        self.mask.array[:] = self.rng.randint(0, 2**30, self.length).astype(np.int32)
        self.background[:] = self.rng.uniform(size=self.length).astype(np.float32)
        self.covariance[:] = self.rng.uniform(size=(3, self.length)).astype(np.float32)
        self.wavelengthArray[:] = self.rng.uniform(size=self.length).astype(np.float32)
        self.assertFloatsNotEqual(spectrum.spectrum, self.image)
        self.assertFloatsNotEqual(spectrum.mask.array, self.mask.array)
        self.assertFloatsNotEqual(spectrum.variance, self.covariance[0])
        self.assertFloatsNotEqual(spectrum.background, self.background)
        self.assertFloatsNotEqual(spectrum.covariance, self.covariance)
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
        self.assertFloatsEqual(spectrum.getSpectrum(), self.image)
        self.assertImagesEqual(spectrum.getMask(), self.mask)
        self.assertFloatsEqual(spectrum.getVariance(), self.covariance[0])
        self.assertFloatsEqual(spectrum.getBackground(), self.background)
        self.assertFloatsEqual(spectrum.getCovariance(), self.covariance)
        self.assertFloatsEqual(spectrum.getWavelength(), self.wavelengthArray)
        self.assertEqual(len(spectrum.getReferenceLines()), 1)
        self.assertReferenceLine(spectrum.getReferenceLines()[0])

        # Change versions in self and ensure the values in spectrum change (pointers)
        self.image[:] = self.rng.uniform(size=self.length).astype(np.float32)
        self.mask[:] = lsst.afw.image.Mask(self.length, 1)
        self.mask.array[:] = self.rng.randint(0, 2**31, self.length).astype(np.int32)
        self.background[:] = self.rng.uniform(size=self.length).astype(np.float32)
        self.covariance[:] = self.rng.uniform(size=(3, self.length)).astype(np.float32)
        self.wavelengthArray[:] = self.rng.uniform(size=self.length).astype(np.float32)
        self.assertFloatsEqual(spectrum.spectrum, self.image)
        self.assertFloatsEqual(spectrum.mask.array, self.mask.array)
        self.assertFloatsEqual(spectrum.variance, self.covariance[0])
        self.assertFloatsEqual(spectrum.background, self.background)
        self.assertFloatsEqual(spectrum.covariance, self.covariance)
        self.assertFloatsEqual(spectrum.wavelength, self.wavelengthArray)

        # Set the variance and ensure it changes the covariance
        spectrum.covariance[:] = self.covariance
        self.assertFloatsEqual(spectrum.covariance, self.covariance)
        variance = self.rng.uniform(size=self.length).astype(np.float32)
        spectrum.variance = variance
        self.covariance[0] = variance
        self.assertFloatsEqual(spectrum.covariance, self.covariance)

    def testPickle(self):
        """Test pickling of ``Spectrum``"""
        spectrum = self.makeSpectrum()
        self.assertSpectrum(spectrum)
        copy = pickle.loads(pickle.dumps(spectrum))
        self.assertSpectrum(copy)

    def testIdentifyLines(self):
        """Test ``Spectrum.identifyLines``"""
        center = 0.5*self.length
        sigma = 1.2345
        peak = 1000.0
        spectrum = drpStella.Spectrum(self.length, self.fiberId)
        spectrum.wavelength = np.arange(self.length, dtype=np.float32)
        spectrum.spectrum[:] = peak*np.exp(-0.5*((spectrum.wavelength - center)/sigma)**2)
        spectrum.mask[:] = 0
        spectrum.variance[:] = 0.1
        spectrum.referenceLines = []

        referenceLines = [self.line]
        referenceLines[0].guessedPosition = center

        dispCtrl = drpStella.DispersionCorrectionControl()
        spectrum.identify(referenceLines, dispCtrl, 0)
        self.assertEqual(len(spectrum.referenceLines), 1)
        line = spectrum.referenceLines[0]
        self.assertEqual(line.guessedPosition, center)
        self.assertEqual(line.status, drpStella.ReferenceLine.Status.FIT)
        self.assertAlmostEqual(line.fitPosition, center)
        self.assertAlmostEqual(line.fitIntensity, peak)
        self.assertLess(line.fitPositionErr, sigma/5)

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
            spectrum.plot(numRows=4, doBackground=True, doReferenceLines=True, filename=filename)
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
        spectrum = drpStella.Spectrum(self.length, 0)
        indices = np.arange(self.length, dtype=float)
        wl0 = 500.0
        wlSlope = 3.21
        spectrum.setWavelength((indices*wlSlope + wl0).astype(np.float32))

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
