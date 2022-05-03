import os
import sys
import pickle
import unittest

import matplotlib
matplotlib.use("Agg")  # noqa

import numpy as np  # noqa E402: import after code

import lsst.utils.tests  # noqa E402: import after code
from pfs.drp.stella import Spectrum, SpectrumSet  # noqa E402: import after code

display = None


class SpectrumSetTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(54321)

        # Spectrum
        self.length = 123
        self.fiberId = 456
        self.image = self.rng.uniform(size=self.length).astype(np.float32)
        self.mask = lsst.afw.image.Mask(self.length, 1)
        self.mask.array[:] = self.rng.randint(0, 2**30, self.length).astype(np.int32)
        self.background = self.rng.uniform(size=self.length).astype(np.float32)
        self.norm = self.rng.uniform(size=self.length).astype(np.float32)
        self.covariance = self.rng.uniform(size=(3, self.length)).astype(np.float32)
        self.wavelengthArray = np.arange(1, self.length + 1, dtype=float)

    def makeSpectrum(self):
        """Make a ``Spectrum`` for testing"""
        return Spectrum(self.image, self.mask, self.background, self.norm, self.covariance,
                        self.wavelengthArray, self.fiberId)

    def assertSpectrum(self, spectrum):
        """Assert that the ``Spectrum`` has the expected contents"""
        self.assertEqual(len(spectrum), self.length)
        self.assertEqual(spectrum.fiberId, self.fiberId)
        self.assertFloatsEqual(spectrum.spectrum, self.image)
        self.assertImagesEqual(spectrum.mask, self.mask)
        self.assertFloatsEqual(spectrum.variance, self.covariance[0])
        self.assertFloatsEqual(spectrum.background, self.background)
        self.assertFloatsEqual(spectrum.norm, self.norm)
        self.assertFloatsEqual(spectrum.covariance, self.covariance)
        self.assertFloatsEqual(spectrum.wavelength, self.wavelengthArray)

    def testBasics(self):
        """Test basic APIs.

        Includes construction, length, iteration.
        """
        spectra = SpectrumSet(self.length)
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
        spectra = SpectrumSet(num, self.length)
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
        spectra = SpectrumSet(num, self.length)
        for ii, ss in enumerate(spectra):
            multiplier = ii + 1
            ss.spectrum = self.image*multiplier
            ss.wavelength = self.wavelengthArray*multiplier
            ss.mask.array[0, :] = ii
            ss.covariance = self.covariance*multiplier
            ss.background = self.background*multiplier
            ss.norm = self.norm*multiplier
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
            self.assertFloatsEqual(spectrum.norm, self.norm*multiplier)

    def testGetAll(self):
        """Test the ``getAll*`` methods"""
        num = 5
        spectra = self.makeSpectrumSet(num)

        image = np.concatenate([self.image[np.newaxis, :]*(ii + 1) for ii in range(num)])
        wavelength = np.concatenate([self.wavelengthArray[np.newaxis, :]*(ii + 1) for ii in range(num)])
        mask = np.concatenate([np.ones((1, self.length), dtype=np.int32)*ii for ii in range(num)])
        background = np.concatenate([self.background[np.newaxis, :]*(ii + 1) for ii in range(num)])
        norm = np.concatenate([self.norm[np.newaxis, :]*(ii + 1) for ii in range(num)])
        covariance = np.concatenate([self.covariance[np.newaxis, :, :]*(ii + 1) for ii in range(num)])

        self.assertFloatsEqual(spectra.getAllFluxes(), image)
        self.assertFloatsEqual(spectra.getAllWavelengths(), wavelength)
        self.assertFloatsEqual(spectra.getAllMasks(), mask)
        self.assertFloatsEqual(spectra.getAllCovariances(), covariance)
        self.assertFloatsEqual(spectra.getAllBackgrounds(), background)
        self.assertFloatsEqual(spectra.getAllNormalizations(), norm)

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

        converted = SpectrumSet.fromPfsArm(spectra.toPfsArm(dataId))

        # datamodel currently does not preserve the fiberId values
        # so put in the correct values so the assertion can pass
        for ii, ss in enumerate(converted):
            ss.fiberId = ii

        self.assertSpectrumSet(converted, num)

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
            copy = SpectrumSet.readFits(filename)

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

    def testPlot(self):
        """Test plotting of spectra

        Not easy to test the actual result, but we can test that the API hasn't
        been broken.
        """
        import matplotlib.pyplot as plt
        plt.switch_backend("agg")  # In case someone has loaded a different backend that will cause trouble
        ext = ".png"  # Extension to use for plot filenames
        spectra = self.makeSpectrumSet(5)
        # Write directly to file
        with lsst.utils.tests.getTempFilePath(ext) as filename:
            spectra.plot(numRows=4, filename=filename)
        # Check return values
        with lsst.utils.tests.getTempFilePath(ext) as filename:
            numRows = 4  # Must be > 1 for len(axes) to work
            figure, axes = spectra.plot(numRows=numRows)
            figure.savefig(filename)
            self.assertEqual(len(axes), numRows)
        # Test one row, write directly to file
        with lsst.utils.tests.getTempFilePath(ext) as filename:
            spectra.plot(numRows=1, filename=filename)
        # Test one row, check return values
        with lsst.utils.tests.getTempFilePath(ext) as filename:
            figure, axes = spectra.plot(numRows=1)
            with self.assertRaises(TypeError):
                axes[0]
            figure.savefig(filename)


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
