import numpy as np

import lsst.utils.tests

from pfs.drp.stella.atmosphere import AtmosphericTransmission
from pfs.drp.stella.tests import runTests


class AtmosphericTransmissionTestCase(lsst.utils.tests.TestCase):
    """Test `AtmosphericTransmission` interpolation"""

    def setUp(self):
        self.wavelength = np.linspace(400.0, 600.0, 21)
        self.zd = [0.0, 90.0]
        self.pwv = [0.0, 10.0, 20.0]
        # Transmission depends on both zd and pwv, so the ZD-blend and PWV-interpolation
        # branches can each be checked against a known linear construction.
        self.transmission = {
            (zz, pp): np.full_like(self.wavelength, 1.0 - 0.01 * zz - 0.02 * pp)
            for zz in self.zd
            for pp in self.pwv
        }
        self.model = AtmosphericTransmission(
            wavelength=self.wavelength, zd=self.zd, pwv=self.pwv, transmission=self.transmission
        )

    def testExactZenithDistance(self):
        """Exact ZD grid match should resample onto the requested wavelength grid"""
        wavelength = np.linspace(450.0, 550.0, 11)
        interpolator = self.model.makeInterpolator(0.0, wavelength)
        for pwv in self.pwv:
            result = interpolator(pwv)
            expected = np.full_like(wavelength, 1.0 - 0.02 * pwv)
            self.assertFloatsAlmostEqual(result, expected, atol=1.0e-10)

    def testZenithDistanceBlend(self):
        """ZD between grid points should linearly blend the two neighboring ZD models"""
        wavelength = np.linspace(450.0, 550.0, 11)
        zd = 30.0  # 1/3 of the way from 0 to 90
        interpolator = self.model.makeInterpolator(zd, wavelength)
        for pwv in self.pwv:
            result = interpolator(pwv)
            expected = np.full_like(wavelength, 1.0 - 0.01 * zd - 0.02 * pwv)
            self.assertFloatsAlmostEqual(result, expected, atol=1.0e-10)

    def testPwvInterpolation(self):
        """PWV between grid points should linearly interpolate"""
        wavelength = np.linspace(450.0, 550.0, 11)
        interpolator = self.model.makeInterpolator(0.0, wavelength)
        pwv = 5.0  # Halfway between grid points 0 and 10
        result = interpolator(pwv)
        expected = np.full_like(wavelength, 1.0 - 0.02 * pwv)
        self.assertFloatsAlmostEqual(result, expected, atol=1.0e-10)

    def testPwvBelowGrid(self):
        """PWV below the grid minimum should return NaN"""
        wavelength = np.linspace(450.0, 550.0, 11)
        interpolator = self.model.makeInterpolator(0.0, wavelength)
        result = interpolator(-1.0)
        self.assertTrue(np.all(np.isnan(result)))

    def testZenithDistanceOutOfRange(self):
        """ZD below the grid minimum should raise"""
        wavelength = np.linspace(450.0, 550.0, 11)
        with self.assertRaises(RuntimeError):
            self.model.makeInterpolator(-1.0, wavelength)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
