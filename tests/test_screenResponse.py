import numpy as np

import lsst.utils.tests

from pfs.drp.stella.screen import ScreenResponseModel
from pfs.drp.stella.tests.utils import runTests


class ScreenResponseModelTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.xGrid = np.linspace(-200.0, 200.0, 41)
        self.yGrid = np.linspace(-200.0, 200.0, 41)
        gridX, gridY = np.meshgrid(self.xGrid, self.yGrid, indexing="xy")

        # a tilt and a quadrupole, so the two components behave differently under rotation
        self.mean = np.ones_like(gridX)
        self.components = np.array([gridX / 200.0, (gridX**2 - gridY**2) / 200.0**2])
        self.wavelengths = np.linspace(400.0, 900.0, 6)
        self.scores = np.stack([
            np.linspace(0.01, 0.03, self.wavelengths.size),
            np.linspace(0.005, -0.005, self.wavelengths.size),
        ], axis=-1)

    def makeModel(self):
        return ScreenResponseModel(self.xGrid, self.yGrid, self.mean, self.components,
                                   self.wavelengths, self.scores)

    def assertScreenResponseModel(self, model):
        """Assert that a ``ScreenResponseModel`` matches expectations"""
        self.assertFloatsAlmostEqual(model.xGrid, self.xGrid, atol=1.0e-5)
        self.assertFloatsAlmostEqual(model.yGrid, self.yGrid, atol=1.0e-5)
        self.assertFloatsAlmostEqual(model.mean, self.mean, atol=1.0e-6)
        self.assertFloatsAlmostEqual(model.components, self.components, atol=1.0e-6)
        self.assertFloatsAlmostEqual(model.wavelengths, self.wavelengths, atol=1.0e-4)
        self.assertFloatsAlmostEqual(model.scores, self.scores, atol=1.0e-8)

    def testReadWriteFits(self):
        """Test reading and writing to/from FITS"""
        model = self.makeModel()
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            model.writeFits(filename, metadata=dict(VISITS="1,2,3"))
            copy = ScreenResponseModel.readFits(filename)
            self.assertScreenResponseModel(copy)

            x = np.array([-150.0, 0.0, 150.0])
            y = np.array([100.0, 0.0, -100.0])
            wavelength = np.full((x.size, 1), 650.0)
            self.assertFloatsAlmostEqual(copy(x, y, 30.0, wavelength),
                                         model(x, y, 30.0, wavelength), atol=1.0e-6)

    def testEvaluate(self):
        """The model reproduces the field it was built from

        The positions sit on grid nodes: bilinear interpolation is exact for a linear
        field but not for the quadratic component, so off-node points differ from the
        analytic field by the interpolation error rather than by a fault.
        """
        model = self.makeModel()
        x = np.array([-100.0, 0.0, 100.0, 50.0])
        y = np.array([0.0, 0.0, 0.0, -70.0])
        wavelength = np.full((x.size, 1), self.wavelengths[0])

        expected = 1.0 + (self.scores[0, 0] * x / 200.0
                          + self.scores[0, 1] * (x**2 - y**2) / 200.0**2)
        self.assertFloatsAlmostEqual(model(x, y, 0.0, wavelength)[:, 0], expected, atol=1.0e-6)

    def testRotation(self):
        """A rotated exposure samples the screen at rotated positions"""
        model = self.makeModel()
        x = np.array([100.0, 0.0, -100.0])
        y = np.array([0.0, 100.0, 50.0])
        wavelength = np.full((x.size, 1), 500.0)

        # rotating the exposure by 90 degrees is the same as rotating the fibers by -90
        rotated = model(x, y, 90.0, wavelength)
        direct = model(-y, x, 0.0, wavelength)
        self.assertFloatsAlmostEqual(rotated, direct, atol=1.0e-6)

    def testWavelengthClamp(self):
        """Outside the range the scores were measured over, the end values are held"""
        model = self.makeModel()
        x = np.array([150.0])
        y = np.array([0.0])

        below = model(x, y, 0.0, np.array([[300.0]]))
        atLow = model(x, y, 0.0, np.array([[self.wavelengths[0]]]))
        above = model(x, y, 0.0, np.array([[1300.0]]))
        atHigh = model(x, y, 0.0, np.array([[self.wavelengths[-1]]]))

        self.assertFloatsAlmostEqual(below, atLow, atol=1.0e-9)
        self.assertFloatsAlmostEqual(above, atHigh, atol=1.0e-9)

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
            ScreenResponseModel(self.xGrid, self.yGrid[:-1], self.mean, self.components,
                                self.wavelengths, self.scores)
        with self.assertRaises(RuntimeError):
            ScreenResponseModel(self.xGrid, self.yGrid, self.mean, self.components[:, :-1],
                                self.wavelengths, self.scores)
        with self.assertRaises(RuntimeError):
            ScreenResponseModel(self.xGrid, self.yGrid, self.mean, self.components,
                                self.wavelengths[:-1], self.scores)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
