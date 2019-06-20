import sys
import unittest
import numpy as np


import lsst.utils.tests
import lsst.afw.geom

from pfs.drp.stella.calibrateWavelengthsTask import WavelengthFitData, LineData
from pfs.drp.stella import SpectrumSet, ReferenceLine, DetectorMap


display = None


class WavelengthFitDataTestCase(lsst.utils.tests.TestCase):
    """Test functionality of WavelengthFitData"""
    def setUp(self):
        self.length = 128
        self.fiberId = np.arange(123, 456, 78, dtype=np.int32)
<<<<<<< HEAD
        self.pixels = np.random.uniform(size=self.fiberId.shape)*self.length
        self.pixelsErr = np.random.uniform(size=self.fiberId.shape)*self.length
        self.refpixels = np.random.uniform(size=self.fiberId.shape)*self.length
        self.wlMin = 600.0  # nm
        self.wlMax = 900.0  # nm
        scale = self.wlMax - self.wlMin
        self.measuredWavelength = np.random.uniform(size=self.fiberId.shape)*scale + self.wlMin
        self.measuredWavelengthErr = np.random.uniform(size=self.fiberId.shape)*scale + self.wlMin
        self.refWavelength = np.random.uniform(size=self.fiberId.shape)*scale + self.wlMin
        self.status = np.random.randint(2,size=self.fiberId.shape)

        self.actualWavelength = np.random.uniform(size=self.fiberId.shape)*scale + self.wlMin 
        self.lines = [LineData(*args) for args in zip(self.fiberId, self.pixels, self.pixelsErr, self.refpixels, self.measuredWavelength, self.measuredWavelengthErr, self.refWavelength, self.status)]
=======
        self.fitPixelPos = np.random.uniform(size=self.fiberId.shape)*self.length
        self.nominalPixelPos = np.random.uniform(size=self.fiberId.shape)*self.length
        self.fitPixelPosErr = np.random.uniform(size=self.fiberId.shape)*self.length*0.01
        self.wlMin = 600.0  # nm
        self.wlMax = 900.0  # nm
        scale = self.wlMax - self.wlMin
        self.fitWavelength = np.random.uniform(size=self.fiberId.shape)*scale + self.wlMin
        self.actualWavelength = np.random.uniform(size=self.fiberId.shape)*scale + self.wlMin
        self.fitWavelengthErr = (np.random.uniform(size=self.fiberId.shape)*scale + self.wlMin)*0.0
        self.wavelengthCorr = (np.random.uniform(size=self.fiberId.shape)*scale + self.wlMin)*0.01
        self.reflines = np.random.uniform(size=self.fiberId.shape)*scale + self.wlMin
        self.status = np.random.uniform(size=self.fiberId.shape)
        self.lines = [LineData(*args) for args in zip(self.fiberId, self.reflines,self.nominalPixelPos, self.fitPixelPos, self.fitWavelength, self.fitPixelPosErr, self.fitWavelengthErr,self.wavelengthCorr, self.status)]
>>>>>>> 991f511d1a4942743f13332aafbe6245a16a989b

    def assertWavelengthFitData(self, wlFitData, atol=0.0):
        """Check that the WavelengthFitData is what we expect

        Parameters
        ----------
<<<<<<< HEAD
        wlFitData : `pfs.drp.stella.calibrateWavelengthsTask.WavelengthFitData`
            Object to check.
        atol : `float`
            Absolute tolerance for ``measuredWavelength`` (might not be exact
=======
        wlFitData : `pfs.drp.stella.calibrateWavelengthsTask.WavelengFitData`
            Object to check.
        atol : `float`
            Absolute tolerance for ``fitWavelength`` (might not be exact
>>>>>>> 991f511d1a4942743f13332aafbe6245a16a989b
            if calculated).
        """
        self.assertEqual(len(wlFitData), len(self.fiberId))
        self.assertFloatsEqual(wlFitData.fiberId, self.fiberId)
<<<<<<< HEAD
        self.assertFloatsEqual(wlFitData.pixels, self.pixels)
        self.assertFloatsAlmostEqual(wlFitData.measuredWavelength, self.measuredWavelength, atol=atol)
=======
        self.assertFloatsEqual(wlFitData.nominalPixelPos, self.nominalPixelPos)
        self.assertFloatsAlmostEqual(wlFitData.fitWavelength, self.fitWavelength, atol=atol)
>>>>>>> 991f511d1a4942743f13332aafbe6245a16a989b

    def testBasic(self):
        """Test basic functionality"""
        wlFitData = WavelengthFitData(self.lines)
        self.assertWavelengthFitData(wlFitData)
<<<<<<< HEAD
        residuals = self.measuredWavelength - self.refWavelength
        pixelresiduals = self.pixels - self.refpixels
=======
        residuals = self.fitWavelength - self.reflines
>>>>>>> 991f511d1a4942743f13332aafbe6245a16a989b

        self.assertFloatsEqual(wlFitData.residuals(), residuals)
        self.assertEqual(wlFitData.mean(), residuals.mean())
        self.assertEqual(wlFitData.stdev(), residuals.std())
<<<<<<< HEAD

        self.assertFloatsEqual(wlFitData.pixelresiduals(), pixelresiduals)
        self.assertEqual(wlFitData.pixelmean(), pixelresiduals.mean())

=======
>>>>>>> 991f511d1a4942743f13332aafbe6245a16a989b
        index = 3
        fiberId = self.fiberId[index]
        self.assertFloatsEqual(wlFitData.residuals(fiberId), residuals[index])
        self.assertEqual(wlFitData.mean(fiberId), residuals[index].mean())
        self.assertEqual(wlFitData.stdev(fiberId), residuals[index].std())

    def testFromSpectrumSet(self):
        """Test creation from SpectrumSet"""
        num = len(self.fiberId)
        spectra = SpectrumSet(num, self.length)
        centerKnots = []
        centerValues = []
        wavelengthKnots = []
        wavelengthValues = []
        for ii, ss in enumerate(spectra):
            ss.fiberId = self.fiberId[ii]
            line = ReferenceLine("fake")
            line.status = ReferenceLine.Status.FIT
            line.wavelength = self.actualWavelength[ii]
<<<<<<< HEAD
            line.fitPosition = self.pixels[ii]
            ss.setReferenceLines([line])
            knots = np.array([-1, 0, self.pixels[ii], self.length, self.length + 1], dtype=np.float32)
            centerKnots.append(knots)
            wavelengthKnots.append(knots)
            centerValues.append(np.array([12.34, 123.45, 234.56, 345.67, 456.78], dtype=np.float32))
            wavelengthValues.append(np.array([self.wlMin - 1, self.wlMin, self.measuredWavelength[ii],
=======
            line.fitPosition = self.nominalPixelPos[ii]
            ss.setReferenceLines([line])
            knots = np.array([-1, 0, self.nominalPixelPos[ii], self.length, self.length + 1], dtype=np.float32)
            centerKnots.append(knots)
            wavelengthKnots.append(knots)
            centerValues.append(np.array([12.34, 123.45, 234.56, 345.67, 456.78], dtype=np.float32))
            wavelengthValues.append(np.array([self.wlMin - 1, self.wlMin, self.fitWavelength[ii],
>>>>>>> 991f511d1a4942743f13332aafbe6245a16a989b
                                              self.wlMax, self.wlMax + 1],
                                             dtype=np.float32))
        bbox = lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(0, 0),
                                   lsst.afw.geom.Extent2I(self.length, self.length))
        detMap = DetectorMap(bbox, self.fiberId, centerKnots, centerValues, wavelengthKnots, wavelengthValues)

<<<<<<< HEAD
        wlFitData = WavelengthFitData.fromSpectrumSet()
=======
        wlFitData = WavelengthFitData.fromSpectrumSet(lines)
>>>>>>> 991f511d1a4942743f13332aafbe6245a16a989b
        self.assertWavelengthFitData(wlFitData, atol=1.0e-4)

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
<<<<<<< HEAD
    from argparse import ArgumetParser
=======
    from argparse import ArgumentParser
>>>>>>> 991f511d1a4942743f13332aafbe6245a16a989b
    parser = ArgumentParser(__file__)
    parser.add_argument("--display", help="Display backend")
    args, argv = parser.parse_known_args()
    display = args.display
    unittest.main(failfast=True, argv=[__file__] + argv)
