import numpy as np
import scipy.interpolate
import lsst.utils.tests
from pfs.drp.stella.tests import runTests, methodParameters
from pfs.drp.stella.maskLines import maskLines

display = None


class MaskLinesTestCase(lsst.utils.tests.TestCase):
    """Testing pfs.drp.stella.maskLines"""
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.wlMin = 4_000  # Minimum wavelength (nm)
        self.wlMax = 10_000  # Maximum wavelength (nm)
        self.length = 6001  # Length to wavelength array
        self.wavelength = np.linspace(self.wlMin, self.wlMax, self.length, dtype=float)

    def pyImpl(self, wavelength, lines, maskRadius):
        """Python implementation of maskLines

        We used this before pushing it down to C++ for better performance.

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            Array of spectrum wavelengths.
        lines : `numpy.ndarray` of `float`
            Wavelengths of lines to be masked.
        maskRadius : `int`
            Radius around lines to mask.

        Returns
        -------
        mask : `numpy.ndarray` of `bool`
            Whether the spectral element is near a line.
        """
        delta = np.diff(wavelength)
        assert np.all(delta >= 0) or np.all(delta <= 0), "Monotonic"
        del delta
        num = len(wavelength)
        mask = np.zeros(num, dtype=bool)

        indices = scipy.interpolate.interp1d(wavelength, np.arange(num, dtype=int), kind="linear",
                                             bounds_error=False, fill_value="extrapolate",
                                             assume_sorted=True)(lines)
        for ii in indices:
            if np.isnan(ii):
                continue
            index = int(np.floor(ii))
            if index < -maskRadius or index > num + maskRadius:
                continue
            low = max(0, index - maskRadius + 1)
            high = min(num, index + maskRadius + 1)
            mask[low:high] = True
        return mask

    def check(self, wavelength, lines, maskRadius, sortedLines=False):
        """Check that the C++ maskLines works like the python version

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            Wavelength array of spectrum.
        lines : `numpy.ndarray` of `float`
            List of wavelengths of lines to mask.
        maskRadius : `int`
            Number of pixels either side of line to mask.
        sortedLines : `bool`, optional
            Is the ``lines`` array sorted in increasing wavelength?
        """
        mask = maskLines(wavelength, lines, maskRadius, sortedLines)
        expect = self.pyImpl(wavelength, lines, maskRadius)
        np.testing.assert_array_equal(mask, expect)
        return mask

    @methodParameters(
        numLines=(5, 25, 100, 1000),
        maskRadius=(5, 3, 2, 5),
    )
    def testBasic(self, numLines, maskRadius):
        """Test basic functionality

        Parameters
        ----------
        numLines : `int`
            Number of lines to create.
        maskRadius : `int`
            Number of pixels either side of line to mask.
        """
        lines = self.rng.uniform(self.wlMin, self.wlMax, numLines)
        self.check(self.wavelength, lines, maskRadius)

    @methodParameters(maskRadius=(1, 5, 20))
    def testLinesOffEnds(self, maskRadius):
        """Test that lines off the end get masked too

        Parameters
        ----------
        maskRadius : `int`
            Number of pixels either side of line to mask.
        """
        lines = np.array([self.wavelength[0] - 12.345, self.wavelength[0] - 0.5, self.wavelength[-1] + 0.5,
                          self.wavelength[-1] + 12.345])
        self.check(self.wavelength, lines, maskRadius)

    def testNoLines(self):
        """Test that it works with an empty list of lines"""
        lines = np.empty(0, dtype=float)
        mask = self.check(self.wavelength, lines, 100)
        self.assertTrue(np.all(~mask))

    @methodParameters(
        numLines=(5, 25, 100, 1000),
        maskRadius=(5, 3, 2, 5),
    )
    def testReversedWavelength(self, numLines, maskRadius):
        """Test that it works with a backwards wavelength array

        Parameters
        ----------
        numLines : `int`
            Number of lines to create.
        maskRadius : `int`
            Number of pixels either side of line to mask.
        """
        self.wavelength = self.wavelength[::-1].copy()
        lines = self.rng.uniform(self.wlMin, self.wlMax, numLines)
        self.check(self.wavelength, lines, maskRadius)

    def testNonMonotonicWavelength(self):
        """Test that maskLines fails appropriate with a bad wavelength array"""
        self.rng.shuffle(self.wavelength)
        lines = self.rng.uniform(self.wlMin, self.wlMax, 100)
        with self.assertRaises(RuntimeError):
            self.check(self.wavelength, lines, 5)

    def testSortedLines(self):
        """Test that maskLines works OK when we pre-sort the lines"""
        lines = np.sort(self.rng.uniform(self.wlMin, self.wlMax, 100))
        self.check(self.wavelength, lines, 5, True)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
