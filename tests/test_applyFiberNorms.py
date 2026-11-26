import sys
import unittest

import numpy as np

import lsst.utils.tests

from pfs.datamodel.identity import CalibIdentity, Identity
from pfs.datamodel.masks import MaskHelper
from pfs.drp.stella.applyFluxCal import applyFiberNorms
from pfs.drp.stella.datamodel.drp import PfsArm
from pfs.drp.stella.datamodel.pfsFiberNorms import PfsFiberNorms


class ApplyFiberNormsTestCase(lsst.utils.tests.TestCase):
    """Test applyFiberNorms"""
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.length = 50
        self.fiberId = np.arange(1, 30, 2, dtype=int)
        self.numFibers = self.fiberId.size
        self.norm = self.rng.uniform(1, 2, size=(self.numFibers, self.length))

        flux = self.rng.uniform(size=(self.numFibers, self.length))
        self.pfsArm = PfsArm(
            Identity(visit=12345, arm="r", spectrograph=1),
            self.fiberId,
            np.tile(np.linspace(600, 900, self.length), (self.numFibers, 1)),  # wavelength
            flux,
            np.zeros_like(flux, dtype=np.int32),  # mask
            np.zeros_like(flux),  # sky
            self.norm.copy(),
            self.rng.uniform(size=(self.numFibers, 3, self.length)),  # covar
            MaskHelper(BAD=0, SAT=1, NO_DATA=2),  # flags
            {"PFS.HASH.FIBERPROFILES": 0x12345678},  # metadata
        )

    def makeFiberNorms(self, fiberId):
        """Make fiberNorms for the given fiberIds, with values = 1 + fiberId/100"""
        fiberId = np.asarray(fiberId, dtype=int)
        values = np.tile(1.0 + fiberId[:, None]/100, (1, self.length))
        return PfsFiberNorms(
            CalibIdentity(visit0=12345, arm="r", spectrograph=1, obsDate="2026-01-01"),
            fiberId,
            np.tile(np.linspace(600, 900, self.length), (fiberId.size, 1)),  # wavelength
            values,
            {1: 0x12345678},  # fiberProfilesHash
            None,  # model
            {},  # metadata
        )

    def expectedValues(self, fiberId):
        return np.tile(1.0 + np.asarray(fiberId)[:, None]/100, (1, self.length))

    def testAllPresent(self):
        """All fibers have fiberNorms (in a different order, plus extra fibers)"""
        fiberNorms = self.makeFiberNorms(np.concatenate([self.fiberId[::-1], [100, 200]]))
        missing = applyFiberNorms(self.pfsArm, fiberNorms)
        self.assertEqual(missing, set())
        expected = self.norm*self.expectedValues(self.fiberId)
        self.assertFloatsAlmostEqual(self.pfsArm.norm, expected, rtol=1.0e-14)
        self.assertTrue(np.all(self.pfsArm.mask == 0))

    def testMissing(self):
        """Some fibers lack fiberNorms:  the others must still be normalised, and the missing ones masked

        This used to apply no fiberNorms at all.
        """
        have = np.ones(self.numFibers, dtype=bool)
        have[[0, 5, 6]] = False
        fiberNorms = self.makeFiberNorms(self.fiberId[have])
        missing = applyFiberNorms(self.pfsArm, fiberNorms)
        self.assertEqual(missing, set(self.fiberId[~have].tolist()))

        bad = self.pfsArm.flags["BAD_FIBERNORMS"]
        expected = self.norm[have]*self.expectedValues(self.fiberId[have])
        self.assertFloatsAlmostEqual(self.pfsArm.norm[have], expected, rtol=1.0e-14)
        self.assertFloatsEqual(self.pfsArm.norm[~have], self.norm[~have])
        self.assertTrue(np.all(self.pfsArm.mask[~have] & bad))
        self.assertTrue(np.all(self.pfsArm.mask[have] == 0))

    def testBadValues(self):
        """Zero or non-finite fiberNorms are replaced by unity and masked"""
        fiberNorms = self.makeFiberNorms(self.fiberId)
        fiberNorms.values[2, 10] = 0.0
        fiberNorms.values[3, 20] = np.nan
        original = fiberNorms.values.copy()
        applyFiberNorms(self.pfsArm, fiberNorms)

        bad = self.pfsArm.flags["BAD_FIBERNORMS"]
        isBad = np.zeros_like(self.pfsArm.mask, dtype=bool)
        isBad[2, 10] = isBad[3, 20] = True
        expected = np.where(isBad, 1.0, self.expectedValues(self.fiberId))
        self.assertFloatsAlmostEqual(self.pfsArm.norm, self.norm*expected, rtol=1.0e-14)
        np.testing.assert_array_equal((self.pfsArm.mask & bad) != 0, isBad)
        np.testing.assert_array_equal(fiberNorms.values, original)  # the input fiberNorms aren't modified

    def testHashMismatch(self):
        """A fiberProfiles hash mismatch is an error"""
        self.pfsArm.metadata["PFS.HASH.FIBERPROFILES"] = 0x87654321
        with self.assertRaises(RuntimeError):
            applyFiberNorms(self.pfsArm, self.makeFiberNorms(self.fiberId))
        applyFiberNorms(self.pfsArm, self.makeFiberNorms(self.fiberId), doCheckHashes=False)


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
