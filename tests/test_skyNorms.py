import sys
import unittest

import numpy as np

import lsst.utils.tests

from pfs.datamodel.identity import Identity
from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.pfsConfig import PfsConfig, FiberStatus, TargetType
from pfs.drp.stella.datamodel.drp import PfsArm
from pfs.drp.stella.focalPlaneFunction import ConstantPerFiber
from pfs.drp.stella.PfiCorrection import PfiCorrectionTask, removePfiCorrection
from pfs.drp.stella.skyNorms import MeasureSkyNormsTask


class RemovePfiCorrectionTestCase(lsst.utils.tests.TestCase):
    """Test removal of the PFI correction (and sky norms) from a pfsArm"""
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.length = 100
        self.fiberId = np.arange(1, 600, 7, dtype=int)
        self.numFibers = self.fiberId.size

        flux = self.rng.uniform(size=(self.numFibers, self.length))
        self.pfsArm = PfsArm(
            Identity(visit=12345, arm="r", spectrograph=1),
            self.fiberId,
            np.tile(np.linspace(600, 900, self.length), (self.numFibers, 1)),  # wavelength
            flux,
            np.zeros_like(flux, dtype=np.int32),  # mask
            np.zeros_like(flux),  # sky
            self.rng.uniform(1, 2, size=flux.shape),  # norm
            self.rng.uniform(size=(self.numFibers, 3, self.length)),  # covar
            MaskHelper(BAD=0, SAT=1, NO_DATA=2),  # flags
            {},  # metadata
        )

        angle = self.rng.uniform(0, 2*np.pi, size=self.numFibers)
        radius = 200*np.sqrt(self.rng.uniform(size=self.numFibers))
        pfiCenter = np.array([radius*np.cos(angle), radius*np.sin(angle)]).T
        n = self.numFibers
        flux = [np.zeros(0, dtype=float)]*n
        radec = np.zeros(n, dtype=float)
        self.pfsConfig = PfsConfig(
            123456789, 12345, 0.0, 0.0, 0.0, "r", self.fiberId,
            np.full(n, 9812, dtype=int),  # tract
            ["5,5"]*n,  # patch
            radec, radec,
            np.full(n, 1, dtype=int),  # catId
            np.arange(n, dtype=int),  # objId
            np.full(n, TargetType.SKY, dtype=int),
            np.full(n, FiberStatus.GOOD, dtype=int),
            np.full(n, "J2000.0"),  # epoch
            np.zeros(n, dtype=np.float32),  # pmRa
            np.zeros(n, dtype=np.float32),  # pmDec
            np.full(n, 1e-5, dtype=np.float32),  # parallax
            np.full(n, "S24B-001QN"),  # proposalId
            np.array([f"obcode_{ii:04d}" for ii in range(n)]),  # obCode
            flux, flux, flux, flux, flux, flux,
            [[]]*n,  # filterNames
            pfiCenter, pfiCenter, None,
        )

    def testNothingApplied(self):
        """No correction was applied:  nothing should change"""
        norm = self.pfsArm.norm.copy()
        self.assertTrue(np.all(np.isnan(self.pfsArm.notes.pfiCorrection)))
        removed = removePfiCorrection(self.pfsArm)
        self.assertFalse(np.any(removed))
        self.assertFloatsEqual(self.pfsArm.norm, norm)

    def testRemove(self):
        """Removing the PFI correction and sky norms restores the norm"""
        norm = self.pfsArm.norm.copy()
        skyNorms = ConstantPerFiber(
            self.fiberId, self.rng.normal(0, 0.02, size=self.numFibers), np.zeros(self.numFibers)
        )
        result = PfiCorrectionTask().run(self.pfsArm, self.pfsConfig, skyNorms)
        good = np.isfinite(result.correction)
        self.assertGreater(good.sum(), 0)
        self.assertFalse(np.allclose(self.pfsArm.norm[good], norm[good]))

        removed = removePfiCorrection(self.pfsArm)
        np.testing.assert_array_equal(removed, good)
        rtol = 1.0e-6  # notes.pfiCorrection is float32
        self.assertFloatsAlmostEqual(self.pfsArm.norm, norm, rtol=rtol)
        self.assertTrue(np.all(np.isnan(self.pfsArm.notes.pfiCorrection)))

        # Removing again is harmless
        removed = removePfiCorrection(self.pfsArm)
        self.assertFalse(np.any(removed))
        self.assertFloatsAlmostEqual(self.pfsArm.norm, norm, rtol=rtol)

    def testConfig(self):
        """The default is to remove the correction"""
        config = MeasureSkyNormsTask.ConfigClass()
        self.assertTrue(config.doRemovePfiCorrection)
        self.assertTrue(config.requireSky)


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
    unittest.main(failfast=True, argv=[__file__] + argv)
