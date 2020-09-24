import sys
import unittest
import numpy as np

import lsst.utils.tests
import lsst.afw.geom

from pfs.drp.stella.calibrateWavelengthsTask import WavelengthFitData, LineData
from pfs.drp.stella import ReferenceLine, SplinedDetectorMap

display = None


class WavelengthFitDataTestCase(lsst.utils.tests.TestCase):
    """Test functionality of WavelengthFitData"""
    def setUp(self):
        self.length = 128
        self.fiberId = np.arange(123, 456, 78, dtype=np.int32)

        self.wlMin = 600.0  # nm
        self.wlMax = 900.0  # nm
        scale = self.wlMax - self.wlMin

        float32 = np.float32
        self.measuredPosition = np.random.uniform(size=self.fiberId.shape).astype(float32)*(self.length - 1)
        self.measuredPositionErr = 0.01*np.random.uniform(size=self.fiberId.shape).astype(float32)
        self.xCenter = np.random.uniform(size=self.fiberId.shape).astype(float32)*(self.length - 1)
        self.refWavelength = np.random.uniform(size=self.fiberId.shape).astype(float32)*scale + self.wlMin
        self.fitWavelength = np.random.uniform(size=self.fiberId.shape).astype(float32)*scale + self.wlMin
        self.correction = 0.1*np.random.uniform(size=self.fiberId.shape).astype(float32)
        self.status = int(ReferenceLine.Status.FIT)*np.ones_like(self.fiberId, dtype=int)
        self.description = np.array([chr(ord("A") + ii) for ii in range(len(self.fiberId))])

        self.lines = [LineData(*args) for args in zip(self.fiberId, self.measuredPosition,
                                                      self.measuredPositionErr, self.xCenter,
                                                      self.refWavelength, self.fitWavelength,
                                                      self.correction, self.status, self.description)]

    def assertWavelengthFitData(self, wlFitData, atol=0.0):
        """Check that the WavelengthFitData is what we expect

        Parameters
        ----------
        wlFitData : `pfs.drp.stella.calibrateWavelengthsTask.WavelengthFitData`
            Wavelength fit data.
        atol : `float`, optional
            Absolute tolerance.
        """
        self.assertEqual(len(wlFitData), len(self.fiberId))
        self.assertFloatsEqual(wlFitData.fiberId, self.fiberId)
        self.assertFloatsAlmostEqual(wlFitData.measuredPosition, self.measuredPosition, atol=atol)
        self.assertFloatsAlmostEqual(wlFitData.measuredPositionErr, self.measuredPositionErr, atol=atol)
        self.assertFloatsAlmostEqual(wlFitData.xCenter, self.xCenter, atol=atol)
        self.assertFloatsAlmostEqual(wlFitData.refWavelength, self.refWavelength, atol=atol)
        self.assertFloatsAlmostEqual(wlFitData.fitWavelength, self.fitWavelength, atol=atol)
        self.assertFloatsAlmostEqual(wlFitData.correction, self.correction, atol=atol)
        self.assertFloatsEqual(wlFitData.status, self.status)
        self.assertListEqual(wlFitData.description.tolist(), self.description.tolist())

    def testBasic(self):
        """Test basic functionality"""
        wlFitData = WavelengthFitData(self.lines)
        self.assertWavelengthFitData(wlFitData)
        residuals = self.fitWavelength - self.refWavelength

        self.assertFloatsEqual(wlFitData.residuals(), residuals)
        self.assertEqual(wlFitData.mean(), residuals.mean())
        self.assertEqual(wlFitData.stdev(), residuals.std())

        index = 3
        fiberId = self.fiberId[index]
        self.assertFloatsEqual(wlFitData.residuals(fiberId), residuals[index])
        self.assertEqual(wlFitData.mean(fiberId), residuals[index].mean())
        self.assertEqual(wlFitData.stdev(fiberId), residuals[index].std())

    def testFromReferenceLines(self):
        """Test creation from ReferenceLines"""
        num = len(self.fiberId)
        refLines = {}
        centerKnots = []
        centerValues = []
        wavelengthKnots = []
        wavelengthValues = []
        for ii in range(num):
            line = ReferenceLine(self.description[ii])
            line.status = self.status[ii]
            line.wavelength = self.refWavelength[ii]
            line.fitPosition = self.measuredPosition[ii]
            line.fitPositionErr = self.measuredPositionErr[ii]
            refLines[self.fiberId[ii]] = [line]

            knots = np.array([-1, 0, self.measuredPosition[ii], self.length, self.length + 1],
                             dtype=np.float32)
            centerKnots.append(knots)
            wavelengthKnots.append(knots)
            centerValues.append(np.array([self.xCenter[ii]]*len(knots), dtype=np.float32))
            wavelengthValues.append(np.array([self.wlMin - 1, self.wlMin, self.fitWavelength[ii],
                                              self.wlMax, self.wlMax + 1],
                                             dtype=np.float32))

        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                               lsst.geom.Extent2I(self.length, self.length))
        detMap = SplinedDetectorMap(bbox, self.fiberId, centerKnots, centerValues,
                                    wavelengthKnots, wavelengthValues)

        corrections = {fiberId: lambda xx: self.correction[self.measuredPosition == xx][0] for
                       fiberId in self.fiberId}
        wlFitData = WavelengthFitData.fromReferenceLines(refLines, detMap, corrections)
        self.assertWavelengthFitData(wlFitData)

    def testPersistence(self):
        """Test persistence"""
        wlFitData = WavelengthFitData(self.lines)
        self.assertWavelengthFitData(wlFitData)
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            wlFitData.writeFits(filename)
            copy = WavelengthFitData.readFits(filename)
            self.assertWavelengthFitData(copy)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()
    np.random.seed(12345)


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])

    from argparse import ArgumentParser
    parser = ArgumentParser(__file__)
    parser.add_argument("--display", help="Display backend")
    args, argv = parser.parse_known_args()
    display = args.display
    unittest.main(failfast=True, argv=[__file__] + argv)
