import sys
import unittest

import numpy as np

import lsst.pipe.base
import lsst.utils.tests

from pfs.datamodel.identity import Identity
from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.pfsConfig import PfsConfig, FiberStatus, TargetType
from pfs.drp.stella.datamodel.drp import PfsArm
from pfs.drp.stella.focalPlaneFunction import ConstantPerFiber
from pfs.drp.stella.PfiCorrection import PfiCorrectionTask, removeBlackSpotCorrection, removePfiCorrection
from pfs.drp.stella.skyNorms import MeasureSkyNormsTask, NoSkyDataError


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

    def testRemoveBlackSpot(self):
        """Removing the black spot correction (and the PFI correction) restores the norm"""
        norm = self.pfsArm.norm.copy()
        self.assertFalse(np.any(removeBlackSpotCorrection(self.pfsArm)))   # nothing applied
        self.assertFloatsEqual(self.pfsArm.norm, norm)

        # As BlackSpotCorrectionTask:  most fibers are far from a spot (correction 1), some are dimmed
        corr = np.ones(self.numFibers, dtype=np.float32)
        near = self.rng.uniform(size=self.numFibers) < 0.2
        corr[near] = self.rng.uniform(0.3, 1.0, size=near.sum())
        corr[0] = 0.0   # can't be undone
        self.pfsArm.norm *= corr[:, None]
        self.pfsArm.notes.blackSpotCorrection[:] = corr
        PfiCorrectionTask().run(self.pfsArm, self.pfsConfig, None)

        rtol = 1.0e-6  # the notes are float32
        self.assertFalse(np.any(np.isnan(self.pfsArm.notes.blackSpotCorrection[1:])))
        removePfiCorrection(self.pfsArm)
        removed = removeBlackSpotCorrection(self.pfsArm)
        self.assertFalse(removed[0])
        self.assertTrue(np.all(removed[1:]))
        self.assertFloatsAlmostEqual(self.pfsArm.norm[1:], norm[1:], rtol=rtol)
        self.assertTrue(np.all(np.isnan(self.pfsArm.notes.blackSpotCorrection[1:])))

        # Removing again is harmless
        self.assertFalse(np.any(removeBlackSpotCorrection(self.pfsArm)))
        self.assertFloatsAlmostEqual(self.pfsArm.norm[1:], norm[1:], rtol=rtol)

    def testConfig(self):
        """The default is to remove the corrections"""
        config = MeasureSkyNormsTask.ConfigClass()
        self.assertTrue(config.doRemovePfiCorrection)
        self.assertTrue(config.doRemoveBlackSpotCorrection)
        self.assertTrue(config.requireSky)

    def testDeviations(self):
        """skyNorms are deviations from unity, whether or not the focal plane fit is removed"""
        x, y = self.pfsConfig.pfiCenter.T
        deviation = self.rng.normal(0, 0.01, size=self.numFibers)
        ratio = 1.0 + 0.02*x/200 + deviation   # data/sky:  a gradient over the focal plane, plus noise
        fiberId = self.fiberId

        class Task(MeasureSkyNormsTask):
            def runSpectrograph(self, pfsArmList, pfsConfig):
                return lsst.pipe.base.Struct(skyNorms=ConstantPerFiber(fiberId, ratio.copy(),
                                                                       np.full(len(fiberId), 0.01)))

        for doFitFocalPlane in (False, True):
            config = Task.ConfigClass()
            config.doFitFocalPlane = doFitFocalPlane
            result = Task(config=config).run({1: [self.pfsArm]}, self.pfsConfig)
            values = result.skyNorms.value
            if doFitFocalPlane:
                self.assertLess(np.abs(np.median(values)), 0.003)   # the gradient (and unity) are removed
                self.assertLess(np.std(values - deviation), 0.003)
            else:
                self.assertFloatsAlmostEqual(values, ratio - 1, rtol=0, atol=1e-12)
                self.assertIsNone(result.focalPlane)

    def testNoSkyData(self):
        """No usable sky data is NoWorkFound, not an error (the task is run in reduceExposure.yaml)"""
        task = MeasureSkyNormsTask()
        noSky = self.pfsConfig.copy()
        noSky.targetType[:] = TargetType.SCIENCE
        with self.assertRaises(NoSkyDataError):
            task.measureSky(self.pfsArm, noSky)
        with self.assertRaises(lsst.pipe.base.NoWorkFound):
            task.run({1: [self.pfsArm]}, noSky)

        self.pfsArm.mask[:] = self.pfsArm.flags["NO_DATA"]
        with self.assertRaises(NoSkyDataError):
            task.measureSky(self.pfsArm, self.pfsConfig)

    def testSkipSpectrograph(self):
        """A spectrograph without usable sky data is skipped;  the others are measured"""
        fiberId = self.fiberId
        ratio = 1.0 + self.rng.normal(0, 0.01, size=self.numFibers)

        class Task(MeasureSkyNormsTask):
            def runSpectrograph(self, pfsArmList, pfsConfig):
                if pfsArmList[0].identity.spectrograph == 1:
                    raise NoSkyDataError("no sky")
                return lsst.pipe.base.Struct(skyNorms=ConstantPerFiber(fiberId, ratio.copy(),
                                                                       np.full(len(fiberId), 0.01)))

        other = self.pfsArm.select(self.pfsConfig, fiberId=self.fiberId)
        other.identity = Identity(visit=12345, arm="r", spectrograph=2)
        config = Task.ConfigClass()
        config.doFitFocalPlane = False
        result = Task(config=config).run({1: [self.pfsArm], 2: [other]}, self.pfsConfig)
        self.assertEqual(list(result.perSpectrograph), [2])
        self.assertFloatsAlmostEqual(result.skyNorms.value, ratio - 1, rtol=0, atol=1e-12)

    def testRunRemovesCorrections(self):
        """run() removes the PFI and black spot corrections before measuring (unless configured not to)"""
        norm = self.pfsArm.norm.copy()
        corr = np.ones(self.numFibers, dtype=np.float32)
        corr[::3] = 0.5
        skyNorms = ConstantPerFiber(
            self.fiberId, self.rng.normal(0, 0.02, size=self.numFibers), np.zeros(self.numFibers)
        )
        seen = {}
        fiberId = self.fiberId

        class Task(MeasureSkyNormsTask):
            def runSpectrograph(self, pfsArmList, pfsConfig):
                seen["norm"] = pfsArmList[0].norm.copy()
                return lsst.pipe.base.Struct(skyNorms=ConstantPerFiber(fiberId, np.ones(len(fiberId)),
                                                                       np.full(len(fiberId), 0.01)))

        for doRemove in (True, False):
            pfsArm = self.pfsArm.select(self.pfsConfig, fiberId=self.fiberId)
            pfsArm.norm *= corr[:, None]
            pfsArm.notes.blackSpotCorrection[:] = corr
            PfiCorrectionTask().run(pfsArm, self.pfsConfig, skyNorms)
            applied = pfsArm.norm.copy()
            config = Task.ConfigClass()
            config.doFitFocalPlane = False
            config.doRemovePfiCorrection = doRemove
            config.doRemoveBlackSpotCorrection = doRemove
            Task(config=config).run({1: [pfsArm]}, self.pfsConfig)
            if doRemove:
                self.assertFloatsAlmostEqual(seen["norm"], norm, rtol=1.0e-6)
            else:
                self.assertFloatsEqual(seen["norm"], applied)


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
