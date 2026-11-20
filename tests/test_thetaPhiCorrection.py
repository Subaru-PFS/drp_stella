import sys
import unittest

import numpy as np

import lsst.utils.tests

from pfs.datamodel import Identity, MaskHelper
from pfs.datamodel.pfsConfig import CobraCommand, FiberStatus, PfsConfig, TargetType
from pfs.datamodel.pfsThetaPhiFine import PfsThetaPhiFine
from pfs.datamodel.pfsThetaPhiModel import PfsThetaPhiModel, makeThetaPhiBasis
from pfs.drp.stella.datamodel.drp import PfsArm
from pfs.drp.stella.PfiCorrection import PfiCorrectionTask
from pfs.drp.stella.thetaPhiCorrection import ThetaPhiCorrectionTask, isCobraHome


class ThetaPhiCorrectionTestCase(lsst.utils.tests.TestCase):
    """Test the cobra-position correction of each fiber's throughput"""
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.length = 100
        self.fiberId = np.arange(1, 41, dtype=int)
        self.numFibers = len(self.fiberId)
        num = self.numFibers

        flux = self.rng.uniform(size=(num, self.length))
        self.norm = self.rng.uniform(1, 2, size=flux.shape)
        self.pfsArm = PfsArm(
            Identity(visit=12345, arm="r", spectrograph=1), self.fiberId,
            np.tile(np.linspace(600, 900, self.length), (num, 1)), flux, np.zeros_like(flux, dtype=np.int32),
            np.zeros_like(flux), self.norm.copy(), self.rng.uniform(size=(num, 3, self.length)),
            MaskHelper(BAD=0, SAT=1, NO_DATA=2), {},
        )

        self.homeTheta = 360 + self.rng.uniform(5, 40, num)
        self.theta = self.rng.uniform(0, 380, num)
        self.phi = self.rng.uniform(20, 170, num)
        command = np.full(num, CobraCommand.CONVERGE, dtype=np.int32)
        command[:3] = CobraCommand.HOME            # cobraHome:  both arms at their stops
        self.theta[:3] = self.homeTheta[:3]
        self.phi[:3] = 0.0
        command[3] = CobraCommand.HOME             # phi sent home, theta converged:  not cobraHome
        self.phi[3] = 0.0
        self.command = command

        angle = self.rng.uniform(0, 2*np.pi, num)
        radius = 200*np.sqrt(self.rng.uniform(size=num))
        pfiCenter = np.array([radius*np.cos(angle), radius*np.sin(angle)]).T
        empty = [np.zeros(0, dtype=float)]*num
        radec = np.zeros(num, dtype=float)
        self.pfsConfig = PfsConfig(
            123456789, 12345, 0.0, 0.0, 0.0, "r", self.fiberId,
            np.full(num, 9812, dtype=int), ["5,5"]*num, radec, radec,
            np.full(num, 1, dtype=int), np.arange(num, dtype=int),
            np.full(num, TargetType.SKY, dtype=int), np.full(num, FiberStatus.GOOD, dtype=int),
            np.full(num, "J2000.0"), np.zeros(num, dtype=np.float32), np.zeros(num, dtype=np.float32),
            np.full(num, 1e-5, dtype=np.float32), np.full(num, "S24B-001QN"),
            np.array([f"obcode_{ii:04d}" for ii in range(num)]),
            empty, empty, empty, empty, empty, empty, [[]]*num, pfiCenter, pfiCenter, None,
            cobraTheta=self.theta.astype(np.float32), cobraPhi=self.phi.astype(np.float32),
            cobraCommand=command,
        )

        basis = makeThetaPhiBasis(3, 2)
        self.model = PfsThetaPhiModel(
            self.fiberId, basis, self.rng.normal(0, 0.005, size=(num, len(basis))), self.homeTheta,
            np.zeros(num), np.zeros(num, dtype=int), np.full(num, 300), np.zeros(num), np.full(num, 180.0),
            np.ones(num), np.full(num, 0.002), np.zeros(num), ["Run 30 (2026-09)"], ["2026-09-01"],
            self.rng.normal(0, 0.006, size=(num, 1)), dict(r=0.93, b=0.95), {},
            dict(ORDER=3, PURE_HARMONICS=2),
        )
        numPoints = np.full(num, 20)
        total = numPoints.sum()
        self.fine = PfsThetaPhiFine(
            self.fiberId, np.r_[0, np.cumsum(numPoints)[:-1]], numPoints, np.tile([0.0015, 0.0015], (num, 1)),
            np.tile([2.1, 21.0], (num, 1)), np.tile([2.2, 15.0], (num, 1)), np.zeros(num, dtype=int),
            np.full(num, 0.5), np.full(num, 0.25), self.rng.uniform(0, 400, total),
            self.rng.uniform(0, 180, total), self.rng.normal(0, 100, total), [0.0, 5.0], [0.4, 1.0],
            dict(ARM="r", SMOOTH_ID=self.model.modelId),
        )

    def testIsCobraHome(self):
        """HOME at the home angles is cobraHome;  HOME elsewhere (phi only) isn't"""
        isHome = isCobraHome(self.pfsConfig, self.homeTheta, np.zeros(self.numFibers))
        np.testing.assert_array_equal(np.where(isHome)[0], [0, 1, 2])

    def testCorrection(self):
        """The factors, and their application and recording by PfiCorrectionTask"""
        task = ThetaPhiCorrectionTask()
        result = task.run(self.pfsArm, self.pfsConfig, self.model, self.fine, "r", "2026-11-19")
        isHome = result.isCobraHome
        smooth = self.model.getCorrection(self.fiberId, self.theta, self.phi, "r", "2026-11-19", isHome)
        fine = self.fine.getCorrection(self.fiberId, self.theta, self.phi, isHome)
        self.assertFloatsAlmostEqual(result.factor, smooth*fine, rtol=1e-6)
        self.assertFloatsEqual(result.factor[:3], 1.0)
        self.assertNotEqual(result.factor[3], 1.0)                 # phi home only:  corrected

        config = PfiCorrectionTask.ConfigClass()
        config.doBlob = False                                      # only the theta/phi factor
        PfiCorrectionTask(config=config).run(self.pfsArm, self.pfsConfig, None, result.factor)
        self.assertFloatsAlmostEqual(self.pfsArm.norm, self.norm*result.factor[:, None], rtol=1e-6)
        self.assertFloatsAlmostEqual(1 + self.pfsArm.notes.pfiCorrection, result.factor, rtol=1e-6)

    def testChecks(self):
        """A fine layer for another arm, or another smooth model, is refused"""
        task = ThetaPhiCorrectionTask()
        with self.assertRaises(RuntimeError):
            task.run(self.pfsArm, self.pfsConfig, self.model, self.fine, "b", "2026-11-19")
        self.fine.metadata["SMOOTH_ID"] = "another model"
        with self.assertRaises(RuntimeError):
            task.run(self.pfsArm, self.pfsConfig, self.model, self.fine, "r", "2026-11-19")
        result = task.run(self.pfsArm, self.pfsConfig, self.model, None, "r", "2026-11-19")   # no fine layer
        self.assertFloatsEqual(result.corrFine, 1.0)

    def testPfiCorrectionTerms(self):
        """The edge and per-fiber terms are off by default"""
        config = PfiCorrectionTask.ConfigClass()
        self.assertFalse(config.doEdge)
        self.assertFalse(config.doFibers)
        self.assertTrue(config.doBlob)
        result = PfiCorrectionTask(config=config).calculate(
            self.fiberId, self.pfsConfig.pfiCenter[:, 0], self.pfsConfig.pfiCenter[:, 1]
        )
        self.assertFloatsEqual(result.edge, 0.0)
        self.assertFloatsEqual(result.fibers, 0.0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main(failfast=True)
