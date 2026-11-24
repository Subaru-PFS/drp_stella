import sys
import unittest

import numpy as np

import lsst.utils.tests

from pfs.datamodel import CalibIdentity
from pfs.datamodel.masks import MaskHelper
from pfs.drp.stella.datamodel.drp import PfsArm
from pfs.drp.stella.combineSpectra import combineSpectraSets


class CombineSpectraSetsTestCase(lsst.utils.tests.TestCase):
    """Tests for combineSpectraSets"""

    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.flags = MaskHelper(NO_DATA=0, SUSPECT=1, CR=2)
        self.numFibers = 5
        self.length = 300
        self.badMasks = ["NO_DATA", "CR"]
        self.suspectMasks = ["SUSPECT"]
        self.spectra = [self.makeSpectra() for _ in range(3)]

    def makeSpectra(self) -> PfsArm:
        """Make spectra with random values"""
        shape = (self.numFibers, self.length)
        flux = self.rng.normal(100, 10, shape)
        mask = np.zeros(shape, dtype=np.int32)
        mask[self.rng.uniform(size=shape) < 0.1] |= self.flags["CR"]
        mask[self.rng.uniform(size=shape) < 0.05] |= self.flags["SUSPECT"]
        covar = np.zeros((self.numFibers, 3, self.length))
        covar[:, 0] = self.rng.uniform(50, 150, shape)
        return PfsArm(
            CalibIdentity(visit0=12345, arm="r", spectrograph=1, obsDate="2026-11-16"),
            np.arange(1, self.numFibers + 1, dtype=int),
            np.tile(np.linspace(600, 900, self.length), (self.numFibers, 1)),
            flux,
            mask,
            self.rng.uniform(0, 50, shape),  # sky
            self.rng.uniform(0.5, 2, shape),  # norm
            covar,
            self.flags,
            {},  # metadata
        )

    def combine(self, weights=None):
        """Combine the spectra"""
        return combineSpectraSets(self.spectra, self.flags, self.badMasks, self.suspectMasks, weights)

    def testInverseVarianceWeights(self):
        """Test that explicit inverse-variance weights reproduce the default"""
        default = self.combine()
        explicit = self.combine([1.0 / ss.variance for ss in self.spectra])
        for name in ("flux", "sky", "norm", "covar", "mask"):
            self.assertFloatsAlmostEqual(getattr(explicit, name), getattr(default, name), rtol=1.0e-6)
        # Variance of an inverse-variance weighted mean is the inverse of the sum of the weights
        good = np.all(
            [(ss.mask & self.flags.get(*self.badMasks, *self.suspectMasks)) == 0 for ss in self.spectra],
            axis=0,
        )
        sumWeights = np.sum([ss.norm**2 / ss.variance for ss in self.spectra], axis=0)
        self.assertFloatsAlmostEqual(
            default.covar[:, 0][good], (default.norm**2 / sumWeights)[good], rtol=1.0e-6
        )

    def testUniformWeights(self):
        """Test that the weights are applied, and the variance propagated"""
        weights = [1.0 / ss.norm**2 for ss in self.spectra]  # uniform weights for the normalised flux
        combination = self.combine(weights)
        good = np.all(
            [(ss.mask & self.flags.get(*self.badMasks, *self.suspectMasks)) == 0 for ss in self.spectra],
            axis=0,
        )
        num = len(self.spectra)
        expectFlux = np.mean([ss.flux / ss.norm for ss in self.spectra], axis=0) * combination.norm
        expectVariance = (
            np.sum([ss.variance / ss.norm**2 for ss in self.spectra], axis=0) / num**2 * combination.norm**2
        )
        self.assertFloatsAlmostEqual(combination.flux[good], expectFlux[good], rtol=1.0e-6)
        self.assertFloatsAlmostEqual(combination.covar[:, 0][good], expectVariance[good], rtol=1.0e-6)

    def testLengthMismatch(self):
        """Test that a mismatch between spectra and weights is caught"""
        with self.assertRaises(RuntimeError):
            self.combine([1.0 / self.spectra[0].variance])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main()
