import sys
import unittest
import numpy as np
import lsst.utils.tests
from pfs.drp.stella.fitLine import fitLine
from pfs.drp.stella import Spectrum

display = None


def gaussian(xx, center, amplitude, rmsSize):
    """Generate a Gaussian

    Parameters
    ----------
    xx : `numpy.ndarray` of `int`, shape ``(N)``
        Array of pixel indices.
    center : `float`
        Mean of Gaussian.
    amplitude : `float`
        Amplitude of Gaussian.
    rmsSize : `float`
        Gaussian RMS (stdev),.

    Returns
    -------
    values : `numpy.ndarray` of `float`, shape ``(N)``
        Array of Gaussian values.
    """
    return amplitude*np.exp(-0.5*(xx - center)**2/rmsSize**2)


def makeSpectrum(length, center, amplitude, rmsSize, bgConst=0.0, bgSlope=0.0):
    """Construct an artificial spectrum of Gaussians

    Parameters
    ----------
    length : `int`
        Length of spectrum.
    center : iterable of `float`
        Center of Gaussian.
    amplitude : iterable or scalar of `float`
        Amplitude of Gaussian.
    rmsSize : iterable or scalar of `float`
        Gaussian RMS (stdev).
    bgConst : `float`, optional
        Background constant value.
    bgSlope : `float`, optional
        Background slope value.

    Returns
    -------
    spectrum : `pfs.drp.stella.Spectrum`
        Spectrum of Gaussians.
    """
    indices = np.arange(length)
    spectrum = Spectrum(length)
    background = bgConst + bgSlope*(indices - center)
    spectrum.flux[:] = gaussian(indices, center, amplitude, rmsSize) + background
    spectrum.mask[:] = 0
    return spectrum


class FitLinesTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.length = 256
        self.center = 123.45
        self.amplitude = 100.0
        self.rmsSize = 2.345
        self.bgConst = 54.321
        self.bgSlope = 1.23
        self.fittingRadius = 50
        self.spectrum = makeSpectrum(self.length, self.center, self.amplitude, self.rmsSize,
                                     self.bgConst, self.bgSlope)

    def tearDown(self):
        del self.spectrum

    def assertFitLineResult(self, result, numMasked=0):
        self.assertIsNotNone(result)
        self.assertTrue(np.isfinite(result.rms))
        self.assertGreater(result.rms, 0)
        self.assertTrue(result.isValid)
        self.assertEqual(result.num, 2*self.fittingRadius + 1 - numMasked)
        self.assertFloatsAlmostEqual(result.amplitude, self.amplitude, rtol=1.0e-5)
        self.assertFloatsAlmostEqual(result.center, self.center, atol=1.0e-4)
        self.assertFloatsAlmostEqual(result.rmsSize, self.rmsSize, atol=1.0e-4)
        self.assertFloatsAlmostEqual(result.bg0, self.bgConst, atol=2.0e-4)
        self.assertFloatsAlmostEqual(result.bg1, self.bgSlope, atol=1.0e-5)
        self.assertGreater(result.amplitudeErr, 0.0)
        self.assertGreater(result.centerErr, 0.0)
        self.assertGreater(result.rmsSizeErr, 0.0)
        self.assertGreater(result.bg0Err, 0.0)
        self.assertGreater(result.bg1Err, 0.0)

    def testBasic(self):
        """Test basic functionality of fitLine"""
        result = fitLine(self.spectrum, int(self.center), int(self.rmsSize), 0, self.fittingRadius)
        self.assertFitLineResult(result)

    def testArray(self):
        """Test array-based API"""
        flux = self.spectrum.flux
        mask = self.spectrum.mask.array[0]
        result = fitLine(flux, mask, int(self.center), int(self.rmsSize), 0, self.fittingRadius)
        self.assertFitLineResult(result)

    def testMasking(self):
        """Test that masked pixels are ignored"""
        pixel = int(self.center) + 1  # +1 so the guess amplitude isn't NAN
        maskVal = 4
        self.spectrum.flux[pixel] = np.nan
        self.spectrum.mask.array[0, pixel] = maskVal

        # Not respecting the bad pixel means the NAN contaminates everything
        result = fitLine(self.spectrum, int(self.center), int(self.rmsSize), 0, 50)
        self.assertFalse(result.isValid)
        self.assertTrue(np.isnan(result.rms))

        # Respecting the bad pixel causes no problems
        result = fitLine(self.spectrum, int(self.center), int(self.rmsSize), maskVal, 50)
        self.assertFitLineResult(result, 1)

    def testErrors(self):
        """Test that the errors are set, and they scale as expected"""
        factor = 100
        before = fitLine(self.spectrum, int(self.center), int(self.rmsSize), 0, self.fittingRadius)

        spectrum = makeSpectrum(self.length, self.center, factor*self.amplitude, self.rmsSize,
                                self.bgConst, self.bgSlope)
        after = fitLine(spectrum, int(self.center), int(self.rmsSize), 0, self.fittingRadius)

        # The amplitude and background errors doesn't change, because they are related to the variance,
        # which we aren't using.
        self.assertFloatsAlmostEqual(after.amplitudeErr, before.amplitudeErr, atol=1.0e-5)
        self.assertFloatsAlmostEqual(after.bg0Err, before.bg0Err, atol=1.0e-2)
        self.assertFloatsAlmostEqual(after.bg1Err, before.bg1Err, atol=1.0e-5)
        # The center and rmsSize errors decrease proportionally with the line flux
        # (they scale inversely with the S/N, and the noise is constant)
        self.assertFloatsAlmostEqual(after.centerErr, before.centerErr/factor, atol=1.0e-5)
        self.assertFloatsAlmostEqual(after.rmsSizeErr, before.rmsSizeErr/factor, atol=1.0e-5)


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
