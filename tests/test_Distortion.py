"""Tests of persistence of Distortion subclasses

Distortions can be persisted on their own (FITS file, bytes, pickle), in
addition to being persisted as part of a detectorMap. These tests check that
the standalone persistence round-trips for each registered subclass.
"""

import os
import pickle

import numpy as np

import lsst.utils.tests
from lsst.geom import Box2D, Box2I, Extent2I, Point2I

from pfs.drp.stella import Distortion, MosaicPolynomialDistortion, PolynomialDistortion
from pfs.drp.stella.tests.utils import runTests


class DistortionTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.order = 3
        self.bbox = Box2I(Point2I(0, 0), Extent2I(4096, 4176))
        self.range = Box2D(self.bbox)
        self.rng = np.random.default_rng(12345)
        numPoints = 100
        self.xx = self.rng.uniform(self.range.getMinX(), self.range.getMaxX(), numPoints)
        self.yy = self.rng.uniform(self.range.getMinY(), self.range.getMaxY(), numPoints)

    def makePolynomialDistortion(self) -> PolynomialDistortion:
        """Construct a PolynomialDistortion with random coefficients"""
        numCoeff = PolynomialDistortion.getNumDistortionForOrder(self.order)
        xCoeff = self.rng.normal(size=numCoeff)
        yCoeff = self.rng.normal(size=numCoeff)
        return PolynomialDistortion(self.order, self.range, xCoeff, yCoeff)

    def makeMosaicPolynomialDistortion(self) -> MosaicPolynomialDistortion:
        """Construct a MosaicPolynomialDistortion with random coefficients"""
        numParams = MosaicPolynomialDistortion.getNumParametersForOrder(self.order)
        return MosaicPolynomialDistortion(self.order, self.range, self.rng.normal(size=numParams))

    def assertDistortionsEqual(self, left: Distortion, right: Distortion):
        """Assert that two distortions are identical"""
        self.assertIs(type(left), type(right))
        self.assertEqual(left.getOrder(), right.getOrder())
        self.assertEqual(left.getRange(), right.getRange())
        np.testing.assert_array_equal(left.getCoefficients(), right.getCoefficients())
        np.testing.assert_array_equal(left(self.xx, self.yy), right(self.xx, self.yy))

    def checkPersistence(self, distortion: Distortion):
        """Check that persistence round-trips

        Parameters
        ----------
        distortion : `Distortion`
            Distortion to check.
        """
        self.assertDistortionsEqual(pickle.loads(pickle.dumps(distortion)), distortion)

        data = distortion.toBytes()
        self.assertDistortionsEqual(Distortion.fromBytes(data), distortion)
        self.assertDistortionsEqual(type(distortion).fromBytes(data), distortion)

        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            distortion.writeFits(filename)
            self.assertTrue(os.path.exists(filename))
            self.assertDistortionsEqual(Distortion.readFits(filename), distortion)
            self.assertDistortionsEqual(type(distortion).readFits(filename), distortion)

    def testPolynomialDistortion(self):
        """Test persistence of PolynomialDistortion"""
        self.checkPersistence(self.makePolynomialDistortion())

    def testMosaicPolynomialDistortion(self):
        """Test persistence of MosaicPolynomialDistortion"""
        self.checkPersistence(self.makeMosaicPolynomialDistortion())


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
