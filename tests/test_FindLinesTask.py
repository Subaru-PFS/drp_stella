import sys
import unittest

import numpy as np

import lsst.utils.tests

from pfs.drp.stella.findLines import FindLinesTask, FindLinesConfig, FittingError
from pfs.drp.stella import Spectrum
from pfs.drp.stella.utils.psf import sigmaToFwhm

display = None


def gaussian(xx, center, amplitude, width):
    """Generate a Gaussian

    Parameters
    ----------
    xx : `numpy.ndarray` of `int`, shape ``(N)``
        Array of pixel indices.
    center : `float`
        Mean of Gaussian.
    amplitude : `float`
        Amplitude of Gaussian.
    width : `float`
        Width (stdev) of Gaussian.

    Returns
    -------
    values : `numpy.ndarray` of `float`, shape ``(N)``
        Array of Gaussian values.
    """
    return amplitude*np.exp(-0.5*(xx - center)**2/width**2)


def makeIterable(value, num=1):
    """Convert value to an iterable if it isn't already

    Parameters
    ----------
    value : iterable or scalar
        Value to convert to iterable.
    num : `int`
        Length of desired iterable for conversion.

    Returns
    -------
    iterable : iterable
        Something you can iterate on.
    """
    try:
        iter(value)
    except TypeError:
        return [value]*num
    return value


def makeSpectrum(length, centers, amplitudes, widths, noise, bgIntercept=0.0, bgSlope=0.0):
    """Construct an artificial spectrum of Gaussians

    Parameters
    ----------
    length : `int`
        Length of spectrum.
    centers : iterable of `float`
        Centers of Gaussians.
    amplitudes : iterable or scalar of `float`
        Amplitudes of Gaussians.
    widths : iterable or scalar of `float`
        Widths (stdev) of Gaussians.
    noise : `float`
        Noise (stdev) to report in the variance array.
    bgIntercept : `float`, optional
        Background intercept value.
    bgSlope : `float`, optional
        Background slope value.

    Returns
    -------
    spectrum : `pfs.drp.stella.Spectrum`
        Spectrum of Gaussians.
    """
    indices = np.arange(length)
    spectrum = Spectrum(length)
    spectrum.spectrum[:] = bgIntercept + bgSlope*indices
    spectrum.mask[:] = 0
    spectrum.variance[:] = noise**2

    for cc, aa, ww in zip(makeIterable(centers),
                          makeIterable(amplitudes, len(centers)),
                          makeIterable(widths, len(centers))):
        spectrum.spectrum += gaussian(indices, cc, aa, ww)

    return spectrum


class FindLinesTestCase(lsst.utils.tests.TestCase):
    def testBasic(self):
        """Test basic functionality of FindLinesTask"""
        length = 512
        centers = np.array([123.45, 234.56, 345.67, 456.78])  # Lines spread out
        amplitudes = 100.0
        widths = 2.345
        noise = 3.21
        background = 54.321
        spectrum = makeSpectrum(length, centers, amplitudes, widths, noise, background)

        config = FindLinesConfig()
        config.fitContinuum.numKnots = 5  # There's not a lot of spectrum...
        config.width = 2
        task = FindLinesTask(config=config)
        result = task.run(spectrum)

        self.assertEqual(result.continuum.shape, spectrum.spectrum.shape)

        lines = result.lines
        self.assertEqual(len(lines), len(centers))
        self.assertFloatsAlmostEqual(np.array([ll.center for ll in lines]), centers, atol=1.0e-5)
        self.assertFloatsAlmostEqual(np.array([ll.amplitude for ll in lines]), amplitudes, rtol=1.0e-4)
        self.assertFloatsAlmostEqual(np.array([ll.width for ll in lines]), widths, atol=1.0e-5)
        self.assertFloatsAlmostEqual(np.array([ll.flux for ll in lines]), widths*amplitudes*np.sqrt(2*np.pi),
                                     rtol=1.0e-5)
        self.assertFloatsAlmostEqual(np.array([ll.fwhm for ll in lines]), sigmaToFwhm(widths),
                                     atol=1.0e-4)
        self.assertFloatsAlmostEqual(np.array([ll.backgroundSlope for ll in lines]), 0.0, atol=1.0e-4)
        self.assertFloatsAlmostEqual(np.array([ll.backgroundIntercept/ll.center for ll in lines]), 0.0,
                                     atol=3.0)  # The distance to x=0 can be substantial

    def testInterlopers(self):
        """Test that interlopers can be masked out effectively

        The tolerances have been loosened compared to ``testBasic``, since the
        accuracy of the measurements are slightly affected by the near
        neighbours, but we're still getting 1% flux measurements.
        """
        length = 512
        centers = np.array([234.56, 245.67])  # Lines close together
        amplitudes = 100.0
        widths = 2.345
        noise = 3.21
        background = 54.321
        spectrum = makeSpectrum(length, centers, amplitudes, widths, noise, background)

        config = FindLinesConfig()
        config.fitContinuum.numKnots = 5  # There's not a lot of spectrum...
        config.width = 2
        task = FindLinesTask(config=config)
        result = task.run(spectrum)

        self.assertEqual(result.continuum.shape, spectrum.spectrum.shape)

        lines = result.lines
        self.assertEqual(len(lines), len(centers))
        self.assertFloatsAlmostEqual(np.array([ll.center for ll in lines]), centers, atol=1.0e-2)
        self.assertFloatsAlmostEqual(np.array([ll.amplitude for ll in lines]), amplitudes, rtol=1.0e-2)
        self.assertFloatsAlmostEqual(np.array([ll.width for ll in lines]), widths, atol=1.0e-2)
        self.assertFloatsAlmostEqual(np.array([ll.flux for ll in lines]), widths*amplitudes*np.sqrt(2*np.pi),
                                     rtol=1.0e-2)
        self.assertFloatsAlmostEqual(np.array([ll.fwhm for ll in lines]), sigmaToFwhm(widths),
                                     atol=2.0e-2)
        self.assertFloatsAlmostEqual(np.array([ll.backgroundSlope for ll in lines]), 0.0, atol=0.1)
        self.assertFloatsAlmostEqual(np.array([ll.backgroundIntercept/ll.center for ll in lines]), 0.0,
                                     atol=3.0)  # The distance to x=0 can be substantial

    def testCentroids(self):
        """Test the ``runCentroids`` method"""
        length = 512
        centers = np.array([123.45, 234.56, 345.67, 456.78])  # Lines spread out
        amplitudes = 100.0
        widths = 2.345
        noise = 3.21
        background = 54.321
        spectrum = makeSpectrum(length, centers, amplitudes, widths, noise, background)

        config = FindLinesConfig()
        config.fitContinuum.numKnots = 5  # There's not a lot of spectrum...
        config.width = 2
        task = FindLinesTask(config=config)
        result = task.runCentroids(spectrum)

        self.assertEqual(result.continuum.shape, spectrum.spectrum.shape)

        centroids = result.centroids
        self.assertEqual(len(centroids), len(centers))
        self.assertFloatsAlmostEqual(centroids, centers, atol=1.0e-5)

    def testBadFit(self):
        """Test that bad fits are ignored"""
        length = 512
        centers = np.array([345.67])
        spectrum = makeSpectrum(length, [], 0.0, 0.0, 0.0)  # No lines!
        bad = "BAD"
        maskVal = spectrum.mask.getPlaneBitMask(bad)
        spectrum.mask.array[0] |= maskVal

        task = FindLinesTask()
        task.config.mask = [bad]
        lines = task.fitLines(spectrum, centers)

        self.assertEqual(len(lines), 0)  # All failed

        with self.assertRaises(FittingError):
            task.fitLines(spectrum, centers, False)


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
