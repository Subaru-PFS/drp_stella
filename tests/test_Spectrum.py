import os
import sys
import unittest
import pickle

import matplotlib
matplotlib.use("Agg")  # noqa

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.daf.base
import lsst.afw.geom
import lsst.afw.image
import lsst.afw.image.testUtils

import pfs.drp.stella as drpStella

display = None


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
        self.status = drpStella.ReferenceLine.Status.NOWT
        self.wavelength = 1234.5678
        self.guessedIntensity = 5.4321
        self.guessedPosition = 123.45678
        self.fitIntensity = 6.54321
        self.fitPosition = 234.56789
        self.fitPositionErr = 0.12345
        self.line = self.makeReferenceLine()

    def makeReferenceLine(self):
        """Make a ``ReferenceLine`` with the provided attributes"""
        line = drpStella.ReferenceLine(self.description, self.status, self.wavelength, self.guessedIntensity)
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
        return drpStella.Spectrum(self.image, self.mask, self.background, self.covariance,
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


class ReferenceLineTestCase(BaseTestCase):
    def testBasics(self):
        """Test construction, getters, setters and arithmetic with ``Status``"""
        self.assertReferenceLine(self.line)

        # Check that enums can be OR-ed together
        self.status = 0
        for name in ("NOWT", "FIT", "RESERVED", "MISIDENTIFIED", "CLIPPED", "SATURATED",
                     "INTERPOLATED", "CR"):
            self.status |= getattr(drpStella.ReferenceLine.Status, name)

        # Check that attributes set at construction time can also be set directly
        self.line.description = self.description = "reference"
        self.line.status = self.status
        self.line.wavelength = self.wavelength = 9876.54321
        self.line.guessedIntensity = self.guessedIntensity = 0.98765
        self.assertReferenceLine(self.line)

    def testPickle(self):
        """Test that ``ReferenceLine`` can be pickled"""
        copy = pickle.loads(pickle.dumps(self.line))
        self.assertReferenceLine(copy)


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
        spectrum = self.makeSpectrum()
        with lsst.utils.tests.getTempFilePath(".png") as filename:
            spectrum.plot(numRows=4, plotBackground=True, plotReferenceLines=True, filename=filename)


class SpectrumSetTestCase(BaseTestCase):
    def testBasics(self):
        """Test basic APIs.

        Includes construction, length, iteration.
        """
        spectra = drpStella.SpectrumSet(self.length)
        self.assertEqual(spectra.getLength(), self.length)
        self.assertEqual(spectra.size(), 0)
        self.assertEqual(len(spectra), 0)
        spectra.reserve(123)
        self.assertEqual(spectra.size(), 0)
        self.assertEqual(len(spectra), 0)

        spectra.add(self.makeSpectrum())
        self.assertEqual(spectra.size(), 1)
        self.assertEqual(len(spectra), 1)
        self.assertSpectrum(spectra[0])

        num = 3
        spectra = drpStella.SpectrumSet(num, self.length)
        self.assertEqual(spectra.getLength(), self.length)
        self.assertEqual(spectra.size(), num)
        self.assertEqual(len(spectra), num)
        for ii in range(num):
            self.assertEqual(spectra[ii].fiberId, ii)
        for ii, spectrum in enumerate(spectra):
            self.assertEqual(spectrum.fiberId, ii)

    def makeSpectrumSet(self, num):
        """Make a ``SpectrumSet`` for testing

        Parameters
        ----------
        num : `int`
            Number of spectra to put in the ``SpectrumSet``.

        Returns
        -------
        spectra : `pfs.drp.stella.SpectrumSet`
            ``SpectrumSet`` for testing.
        """
        spectra = drpStella.SpectrumSet(num, self.length)
        for ii, ss in enumerate(spectra):
            multiplier = ii + 1
            ss.spectrum = self.image*multiplier
            ss.wavelength = self.wavelengthArray*multiplier
            ss.mask.array[0, :] = ii
            ss.covariance = self.covariance*multiplier
            ss.background = self.background*multiplier
            spectra[ii] = ss
        self.assertSpectrumSet(spectra, num)
        return spectra

    def assertSpectrumSet(self, spectra, num):
        """Assert that the ``SpectrumSet`` matches what's expected

        Parameters
        ----------
        spectra : `pfs.drp.stella.SpectrumSet`
            Spectra to validate.
        num : `int`
            Number of spectra to expect.
        """
        self.assertEqual(len(spectra), num)
        for ii, spectrum in enumerate(spectra):
            multiplier = ii + 1
            self.assertEqual(spectrum.fiberId, ii)
            self.assertFloatsEqual(spectrum.spectrum, self.image*multiplier)
            self.assertFloatsEqual(spectrum.wavelength, self.wavelengthArray*multiplier)
            self.assertFloatsEqual(spectrum.mask.array[0, :], ii)
            self.assertFloatsEqual(spectrum.covariance, self.covariance*multiplier)
            self.assertFloatsEqual(spectrum.background, self.background*multiplier)

    def testGetAll(self):
        """Test the ``getAll*`` methods"""
        num = 5
        spectra = self.makeSpectrumSet(num)

        image = np.concatenate([self.image[np.newaxis, :]*(ii + 1) for ii in range(num)])
        wavelength = np.concatenate([self.wavelengthArray[np.newaxis, :]*(ii + 1) for ii in range(num)])
        mask = np.concatenate([np.ones((1, self.length), dtype=np.int32)*ii for ii in range(num)])
        background = np.concatenate([self.background[np.newaxis, :]*(ii + 1) for ii in range(num)])
        covariance = np.concatenate([self.covariance[np.newaxis, :, :]*(ii + 1) for ii in range(num)])

        self.assertFloatsEqual(spectra.getAllFluxes(), image)
        self.assertFloatsEqual(spectra.getAllWavelengths(), wavelength)
        self.assertFloatsEqual(spectra.getAllMasks(), mask)
        self.assertFloatsEqual(spectra.getAllCovariances(), covariance)
        self.assertFloatsEqual(spectra.getAllBackgrounds(), background)

    def testPickle(self):
        """Test pickling of ``SpectrumSet``"""
        num = 5
        spectra = self.makeSpectrumSet(num)
        copy = pickle.loads(pickle.dumps(spectra))
        self.assertSpectrumSet(copy, num)

    def testDatamodel(self):
        """Test conversion to `pfs.datamodel.PfsArm`"""
        num = 5
        spectra = self.makeSpectrumSet(num)

        spectrograph = 3
        arm = "r"
        visit = 12345
        dataId = dict(spectrograph=spectrograph, arm=arm, visit=visit)

        converted = drpStella.SpectrumSet.fromPfsArm(spectra.toPfsArm(dataId))

        self.assertSpectrumSet(converted, num)

    @lsst.utils.tests.debugger(Exception)
    def testReadWriteFits(self):
        """Test reading and writing to/from FITS"""
        num = 5
        spectra = self.makeSpectrumSet(num)

        dirName = os.path.splitext(__file__)[0]
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        filename = os.path.join(dirName, "pfsArm-123456-r3.fits")

        if os.path.exists(filename):
            os.unlink(filename)

        try:
            spectra.writeFits(filename)
            copy = drpStella.SpectrumSet.readFits(filename)

            # writeFits by itself cannot preserve the fiberId values
            # (they're written as a pfsConfig, separately)
            # so put in the correct values so the assertion can pass
            for ii, ss in enumerate(copy):
                ss.fiberId = ii

            self.assertSpectrumSet(copy, num)
        except Exception:
            raise  # Leave file for manual inspection
        else:
            os.unlink(filename)


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
