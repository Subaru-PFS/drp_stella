from contextlib import contextmanager

import numpy as np

import lsst.utils.tests
import lsst.afw.image
import lsst.afw.image.testUtils

import pfs.drp.stella.synthetic as synthetic
from pfs.drp.stella import GlobalDetectorMap
from pfs.drp.stella.measureSlitOffsets import MeasureSlitOffsetsTask
from pfs.drp.stella.tests.utils import runTests, methodParameters

display = None


class MeasureSlitOffsetsTestCase(lsst.utils.tests.TestCase):
    """Test MeasureSlitOffsetsTask"""
    def setUp(self):
        """Produce synthetic exposure"""
        self.rng = np.random.RandomState(12345)
        self.synth = synthetic.SyntheticConfig()
        self.synth.gain = 0.0  # Makes variance image creation easier
        self.synth.readnoise = 10.0
        self.fwhm = 4.321

        self.arc = synthetic.makeSyntheticArc(self.synth, fwhm=self.fwhm, rng=self.rng)
        self.exposure = lsst.afw.image.makeExposure(lsst.afw.image.makeMaskedImage(self.arc.image))
        self.exposure.variance.set(self.synth.readnoise)
        psfSigma = self.fwhm/(2*np.sqrt(2*np.log(2)))
        psfSize = 2*int(psfSigma + 0.5) + 1
        self.exposure.setPsf(lsst.afw.detection.GaussianPsf(psfSize, psfSize, psfSigma))

        self.splinedDetectorMap = synthetic.makeSyntheticDetectorMap(self.synth)
        self.globalDetectorMap = synthetic.makeSyntheticGlobalDetectorMap(self.synth)
        self.pfsConfig = synthetic.makeSyntheticPfsConfig(self.synth, 12345, 6789, rng=self.rng)

        self.config = MeasureSlitOffsetsTask.ConfigClass()
        self.config.readLineList.restrictByLamps = False  # We're not bothering with lamps
        self.config.rejIterations = 0  # So the number of points in the fit is known

    @contextmanager
    def makeLineList(self):
        """Generate a linelist that we can read

        The configuration is updated to point to the linelist.

        Yields
        ------
        filename : `str`
            Filename of linelist.
        """
        with lsst.utils.tests.getTempFilePath(".txt") as filename:
            with open(filename, "w") as fd:
                fd.write("# This is a fake line list\n")
                fiberId = self.synth.fiberId[self.synth.numFibers//2]  # We need any fiber; choose the middle
                for yy in self.arc.lines:
                    wl = self.splinedDetectorMap.findWavelength(fiberId, yy)
                    fd.write(f"{wl} 1000.0 ArI 0\n")
            self.config.readLineList.lineList = filename
            yield filename

    def assertOffsets(self, offsets, dx=0.0, dy=0.0, numRejected=0, numParams=2):
        """Check that the results from MeasureSlitOffsetsTask are as expected

        Parameters
        ----------
        offsets : `lsst.pipe.base.Struct`
            Results from `MeasureSlitOffsetsTask`.
        dx, dy : `float`, optional
            Expected x and y offsets (pixels).
        numRejected : `int`
            Expected number of points to be rejected.
        """
        self.assertFloatsAlmostEqual(offsets.spatial, dx, atol=5.0e-2)
        self.assertFloatsAlmostEqual(offsets.spectral, dy, atol=5.0e-2)
        numLines = len(self.arc.lines)*self.synth.numFibers
        numGood = numLines - numRejected
        self.assertEqual(offsets.num, numGood)
        self.assertEqual(offsets.dof, 2*numGood - numParams)
        self.assertGreater(offsets.chi2, 0.0)
        self.assertLess(offsets.soften, 3.0e-2)
        self.assertEqual(len(offsets.select), numLines)
        self.assertEqual(offsets.select.sum(), numGood)

    @methodParameters(dx=(0.0, -0.3456, 1.2432, 0.0, 0.0),
                      dy=(0.0, 0.0, 0.0, 0.3210, -0.4321))
    def testOffset(self, dx, dy):
        """Test that we can retrieve the expected offset

        We offset the detectorMap in the opposite direction, so there's an
        apparent shift between the exposure and the detectorMap. Then we test
        that we can recover that shift.

        Parameters
        ----------
        dx, dy : `float`
            Offset to apply and retrieve.
        """
        for detMap in (self.splinedDetectorMap, self.globalDetectorMap):
            with self.makeLineList():
                detMap.applySlitOffset(-dx, -dy)
                task = MeasureSlitOffsetsTask(name="measureSlitOffsets", config=self.config)
                offsets = task.run(self.exposure, detMap, self.pfsConfig)
                numParams = 2 if isinstance(detMap, GlobalDetectorMap) else 2*detMap.getNumFibers()
                self.assertOffsets(offsets, dx=dx, dy=dy, numParams=numParams)

    def testRejection(self):
        """Test that we can reject bad points

        We put bad pixels near a line from each fiber that will cause bad
        centroids. We expect these to be rejected as part of the offset
        measurement.
        """
        for fiberId in self.synth.fiberId:
            yy = int(self.rng.choice(self.arc.lines, 1) + 0.5)
            xx = int(self.splinedDetectorMap.getXCenter(fiberId, yy) + 0.5)
            value = self.exposure.image.array[yy, xx]
            self.exposure.image.array[yy:yy + 2, xx:xx + 2] += value

        self.config.rejIterations = 3
        for detMap in (self.splinedDetectorMap, self.globalDetectorMap):
            with self.makeLineList():
                task = MeasureSlitOffsetsTask(name="measureSlitOffsets", config=self.config)
                task.log.setLevel(task.log.DEBUG)
                offsets = task.run(self.exposure, detMap, self.pfsConfig)
                numParams = 2 if isinstance(detMap, GlobalDetectorMap) else 2*detMap.getNumFibers()
                self.assertOffsets(offsets, numRejected=self.synth.numFibers, numParams=numParams)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
