import numpy as np

import lsst.utils.tests

from pfs.drp.stella.screen import ScreenResponseModel
from pfs.drp.stella.tests.utils import runTests


class ScreenResponseModelTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.radius = 224.0
        self.wavelengths = np.array([450.0, 650.0, 850.0])
        # telescope terms x^2, xy, y^2 in units of 100 ln: x^2 growing with wavelength, y^2 fixed
        self.telescope = np.array([
            [1.0, 0.0, 0.5],
            [2.0, 0.0, 0.5],
            [3.0, 0.0, 0.5],
        ])
        # screen grid 5x5, 100 mm apart, from -200 mm: a gradient along u, which bilinear
        # interpolation holds exactly and whose mean over the rotations is zero
        self.spacing = 100.0
        self.halfSize = 200.0
        self.rotations = np.array([0.0, 90.0, 180.0, 270.0])
        nodes = np.arange(-self.halfSize, self.halfSize + 1, self.spacing)
        uu, _ = np.meshgrid(nodes, nodes)
        self.gradients = np.array([0.01, 0.02, 0.03])
        self.screen = self.gradients[:, None]*uu.ravel()[None, :]

    def makeModel(self):
        return ScreenResponseModel(self.wavelengths, self.telescope, 2, 2, self.radius, self.screen,
                                   self.spacing, self.halfSize, self.rotations)

    def testReadWriteFits(self):
        """Test reading and writing to/from FITS"""
        model = self.makeModel()
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            model.writeFits(filename, metadata=dict(DECOMP="test"))
            copy = ScreenResponseModel.readFits(filename)
            self.assertFloatsAlmostEqual(copy.wavelengths, self.wavelengths, atol=1.0e-4)
            self.assertFloatsAlmostEqual(copy.telescope, self.telescope, atol=1.0e-12)
            self.assertFloatsAlmostEqual(copy.screen, self.screen, atol=1.0e-12)
            self.assertFloatsAlmostEqual(copy.rotations, self.rotations, atol=1.0e-12)
            self.assertEqual((copy.degree, copy.lowestOrder), (2, 2))

            x = np.array([-150.0, 0.0, 150.0])
            y = np.array([100.0, 0.0, -100.0])
            wavelength = np.full((x.size, 1), 650.0)
            self.assertFloatsAlmostEqual(copy.telescopeResponse(x, y, 30.0, wavelength),
                                         model.telescopeResponse(x, y, 30.0, wavelength), atol=1.0e-12)
            self.assertFloatsAlmostEqual(copy.quartzResponse(x, y, 30.0, wavelength),
                                         model.quartzResponse(x, y, 30.0, wavelength), atol=1.0e-12)

    def testTelescope(self):
        """The telescope term is the polynomial it holds, in the scaled position"""
        model = self.makeModel()
        x = np.array([-100.0, 0.0, 100.0, 50.0])
        y = np.array([0.0, 0.0, 0.0, -70.0])
        wavelength = np.full((x.size, 1), 650.0)
        expected = 2.0*(x/self.radius)**2 + 0.5*(y/self.radius)**2
        self.assertFloatsAlmostEqual(model.telescopeResponse(x, y, 0.0, wavelength)[:, 0], expected,
                                     atol=1.0e-12)

    def testTelescopeRotation(self):
        """A rotated exposure samples the telescope at rotated positions"""
        model = self.makeModel()
        x = np.array([100.0, 0.0, -100.0])
        y = np.array([0.0, 100.0, 50.0])
        wavelength = np.full((x.size, 1), 500.0)
        # rotating the exposure by 90 degrees carries (x, y) to (-y, x) in the telescope frame
        self.assertFloatsAlmostEqual(model.telescopeResponse(x, y, 90.0, wavelength),
                                     model.telescopeResponse(-y, x, 0.0, wavelength), atol=1.0e-12)

    def testScreen(self):
        """The quartz term is minus the screen at the rotated positions"""
        model = self.makeModel()
        x = np.array([-120.0, 30.0, 170.0])
        y = np.array([40.0, -90.0, 10.0])
        phi = 90.0
        wavelength = np.full((x.size, 1), 650.0)
        u = -y  # the rotation by 90 degrees carries (x, y) to (-y, x)
        values = model.quartzResponse(x, y, phi, wavelength)[:, 0]
        self.assertFloatsAlmostEqual(values, -0.02*u, atol=1.0e-12)

    def testScreenMeanRemoved(self):
        """The screen's mean over the rotations is taken out"""
        nodes = np.arange(-self.halfSize, self.halfSize + 1, self.spacing)
        uu, vv = np.meshgrid(nodes, nodes)
        screen = np.repeat((uu**2 + vv**2).ravel()[None, :], 3, axis=0)  # rotationally symmetric
        model = ScreenResponseModel(self.wavelengths, self.telescope, 2, 2, self.radius, screen,
                                    self.spacing, self.halfSize, self.rotations)
        x = np.array([0.0, 100.0, -100.0, 0.0])
        y = np.array([0.0, 0.0, 100.0, -200.0])
        values = model.quartzResponse(x, y, 0.0, np.full((x.size, 1), 650.0))
        self.assertFloatsAlmostEqual(values, 0.0, atol=1.0e-12)

    def testWavelengthInterpolation(self):
        """Between bin centres the terms are interpolated linearly"""
        model = self.makeModel()
        x, y = np.array([150.0]), np.array([0.0])
        self.assertFloatsAlmostEqual(model.telescopeResponse(x, y, 0.0, np.array([[550.0]]))[0, 0],
                                     1.5*(150.0/self.radius)**2, atol=1.0e-12)

    def testWavelengthClamp(self):
        """Outside the bin centres, the end values are held"""
        model = self.makeModel()
        x, y = np.array([150.0]), np.array([0.0])
        for method, args in ((model.telescopeResponse, (x, y, 0.0)),
                             (model.quartzResponse, (x, y, 0.0))):
            below = method(*args, np.array([[300.0]]))
            atLow = method(*args, np.array([[self.wavelengths[0]]]))
            above = method(*args, np.array([[1300.0]]))
            atHigh = method(*args, np.array([[self.wavelengths[-1]]]))
            self.assertFloatsAlmostEqual(below, atLow, atol=1.0e-12)
            self.assertFloatsAlmostEqual(above, atHigh, atol=1.0e-12)

    def testShape(self):
        """The result has one row per fiber and one column per wavelength"""
        model = self.makeModel()
        x = self.rng.uniform(-200.0, 200.0, 17)
        y = self.rng.uniform(-200.0, 200.0, 17)
        wavelength = np.broadcast_to(np.linspace(400.0, 900.0, 23), (17, 23))
        self.assertEqual(model.telescopeResponse(x, y, 15.0, wavelength).shape, (17, 23))
        self.assertEqual(model.quartzResponse(x, y, 15.0, wavelength).shape, (17, 23))

    def testBadCtor(self):
        """Mismatched arrays are rejected"""
        def make(**kwargs):
            args = dict(wavelengths=self.wavelengths, telescope=self.telescope, degree=2, lowestOrder=2,
                        radius=self.radius, screen=self.screen, spacing=self.spacing, halfSize=self.halfSize,
                        rotations=self.rotations)
            args.update(kwargs)
            return ScreenResponseModel(**args)

        make()
        for kwargs in (dict(wavelengths=self.wavelengths[::-1]),
                       dict(telescope=self.telescope[:, :-1]),
                       dict(screen=self.screen[:, :-1])):
            with self.assertRaises(RuntimeError):
                make(**kwargs)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
