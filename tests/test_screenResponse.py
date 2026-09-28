import numpy as np

import lsst.utils.tests

from pfs.drp.stella.screen import ScreenResponseModel
from pfs.drp.stella.tests.utils import runTests


class ScreenResponseModelTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.radius = 224.0
        self.wavelengths = np.array([450.0, 650.0, 850.0])
        # terms x, y, x^2, xy, y^2 in units of 100 ln: a tilt along x growing with wavelength
        # and a quadratic term along y
        self.coefficients = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.5],
            [2.0, 0.0, 0.0, 0.0, 0.5],
            [3.0, 0.0, 0.0, 0.0, 0.5],
        ])

    def makeModel(self):
        return ScreenResponseModel(self.wavelengths, self.coefficients, 2, self.radius)

    def expected(self, x, y, tilt):
        """The quartz over the twilight for a given tilt coefficient"""
        surface = tilt*x/self.radius + 0.5*(y/self.radius)**2
        return np.exp(-surface/100.0)

    def testReadWriteFits(self):
        """Test reading and writing to/from FITS"""
        model = self.makeModel()
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            model.writeFits(filename, metadata=dict(VISITS="1,2,3"))
            copy = ScreenResponseModel.readFits(filename)
            self.assertFloatsAlmostEqual(copy.wavelengths, self.wavelengths, atol=1.0e-4)
            self.assertFloatsAlmostEqual(copy.coefficients, self.coefficients, atol=1.0e-12)
            self.assertEqual(copy.degree, 2)
            self.assertFloatsAlmostEqual(copy.radius, self.radius, atol=1.0e-12)

            x = np.array([-150.0, 0.0, 150.0])
            y = np.array([100.0, 0.0, -100.0])
            wavelength = np.full((x.size, 1), 650.0)
            self.assertFloatsAlmostEqual(copy(x, y, 30.0, wavelength),
                                         model(x, y, 30.0, wavelength), atol=1.0e-12)

    def testEvaluate(self):
        """The model reproduces the surface it holds, as quartz over twilight"""
        model = self.makeModel()
        x = np.array([-100.0, 0.0, 100.0, 50.0])
        y = np.array([0.0, 0.0, 0.0, -70.0])
        wavelength = np.full((x.size, 1), 650.0)
        self.assertFloatsAlmostEqual(model(x, y, 0.0, wavelength)[:, 0],
                                     self.expected(x, y, 2.0), atol=1.0e-12)

    def testWavelengthInterpolation(self):
        """Between bin centres the coefficients are interpolated linearly"""
        model = self.makeModel()
        x, y = np.array([150.0]), np.array([0.0])
        self.assertFloatsAlmostEqual(model(x, y, 0.0, np.array([[550.0]]))[0, 0],
                                     self.expected(x, y, 1.5)[0], atol=1.0e-12)

    def testRotation(self):
        """A rotated exposure samples the screen at rotated positions"""
        model = self.makeModel()
        x = np.array([100.0, 0.0, -100.0])
        y = np.array([0.0, 100.0, 50.0])
        wavelength = np.full((x.size, 1), 500.0)

        # rotating the exposure by 90 degrees is the same as rotating the fibers by -90
        rotated = model(x, y, 90.0, wavelength)
        direct = model(-y, x, 0.0, wavelength)
        self.assertFloatsAlmostEqual(rotated, direct, atol=1.0e-12)

    def testWavelengthClamp(self):
        """Outside the bin centres, the end values are held"""
        model = self.makeModel()
        x, y = np.array([150.0]), np.array([0.0])

        below = model(x, y, 0.0, np.array([[300.0]]))
        atLow = model(x, y, 0.0, np.array([[self.wavelengths[0]]]))
        above = model(x, y, 0.0, np.array([[1300.0]]))
        atHigh = model(x, y, 0.0, np.array([[self.wavelengths[-1]]]))

        self.assertFloatsAlmostEqual(below, atLow, atol=1.0e-12)
        self.assertFloatsAlmostEqual(above, atHigh, atol=1.0e-12)

    def testShape(self):
        """The result has one row per fiber and one column per wavelength"""
        model = self.makeModel()
        x = self.rng.uniform(-200.0, 200.0, 17)
        y = self.rng.uniform(-200.0, 200.0, 17)
        wavelength = np.broadcast_to(np.linspace(400.0, 900.0, 23), (17, 23))
        self.assertEqual(model(x, y, 15.0, wavelength).shape, (17, 23))

    def testBadCtor(self):
        """Mismatched arrays are rejected"""
        with self.assertRaises(RuntimeError):
            ScreenResponseModel(self.wavelengths[:-1], self.coefficients, 2, self.radius)
        with self.assertRaises(RuntimeError):
            ScreenResponseModel(self.wavelengths, self.coefficients[:, :-1], 2, self.radius)
        with self.assertRaises(RuntimeError):
            ScreenResponseModel(self.wavelengths[::-1], self.coefficients, 2, self.radius)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
