import numpy as np

import lsst.utils.tests
from lsst.afw.image import ExposureF

from pfs.drp.stella.synthetic import SyntheticPsfConfig, makeSyntheticDetectorMap, makeSyntheticPsfArc
from pfs.drp.stella.arcLine import ArcLineSet
from pfs.drp.stella.pfsPsf import PfsPsf
from pfs.drp.stella.fitPsfTask import FitPsfConfig, FitPsfTask
from pfs.drp.stella.psfSpline import RegularizationConfig
from pfs.drp.stella.tests import runTests


class FitPsfTaskTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 200
        self.config.width = 80
        self.config.blockSize = 20
        self.config.psfHalfSize = 10
        self.config.oversampling = 2
        self.config.bumpFraction = 0.0
        self.truth = self.config.makeGroundTruthPsf()

        self.detectorMap = makeSyntheticDetectorMap(self.config)
        self.rng = np.random.RandomState(12345)
        self.result = makeSyntheticPsfArc(self.config, numLines=12, addNoise=True, rng=self.rng)

        wavelength = self.detectorMap.findWavelength(self.result.fiberId, self.result.row)
        self.arcLines = ArcLineSet.fromColumns(
            fiberId=self.result.fiberId,
            wavelength=wavelength,
            x=self.result.xCenter.astype(np.float32),
            y=self.result.row.astype(np.float32),
            xErr=np.full(len(self.result.row), 0.01, dtype=np.float32),
            yErr=np.full(len(self.result.row), 0.01, dtype=np.float32),
            xx=np.zeros(len(self.result.row), dtype=np.float32),
            yy=np.zeros(len(self.result.row), dtype=np.float32),
            xy=np.zeros(len(self.result.row), dtype=np.float32),
            flux=self.result.amplitude.astype(np.float32),
            fluxErr=np.full(len(self.result.row), 0.01, dtype=np.float32),
            fluxNorm=np.ones(len(self.result.row), dtype=np.float32),
            flag=np.zeros(len(self.result.row), dtype=bool),
            status=np.zeros(len(self.result.row), dtype=np.int32),
            description=np.full(len(self.result.row), "Line", dtype=object),
            transition=np.full(len(self.result.row), "", dtype=object),
            source=np.zeros(len(self.result.row), dtype=np.int32),
        )

        self.exposure = ExposureF(self.result.image.getBBox())
        self.exposure.image.array[:] = self.result.image.array

        taskConfig = FitPsfConfig()
        taskConfig.order = 0
        taskConfig.numWings = len(self.truth.wings)
        taskConfig.halfSize = self.config.psfHalfSize
        taskConfig.oversampling = self.config.oversampling
        taskConfig.gain = self.config.gain
        taskConfig.readnoise = self.config.readnoise
        taskConfig.maxOuterIter = 4
        taskConfig.splineExtent = 6.0
        taskConfig.splineFineRadius = 1.0
        taskConfig.splineMediumRadius = 2.0
        taskConfig.splineCoarseSpacing = 2.0
        taskConfig.regularizationSmoothness = 1.0
        taskConfig.regularizationRidge = 10.0
        taskConfig.initialSigmaX = self.truth.sigmaX
        taskConfig.initialSigmaY = self.truth.sigmaY
        taskConfig.initialTophatWidth = self.truth.tophatWidth
        wing = self.truth.wings[0]
        taskConfig.initialWingScaleX = wing.scaleX
        taskConfig.initialWingScaleY = wing.scaleY
        taskConfig.initialWingBeta = wing.beta
        taskConfig.initialWingFraction = wing.fraction
        self.taskConfig = taskConfig

    def testRun(self):
        """FitPsfTask.run recovers a working PfsPsf from synthetic data"""
        task = FitPsfTask(config=self.taskConfig)
        result = task.run(self.exposure, self.detectorMap, self.arcLines)

        self.assertIsInstance(result.psf, PfsPsf)

        fitted = result.fit.model.parametricModel.getParamsAt(0, 0)
        self.assertFloatsAlmostEqual(fitted.sigmaX, self.truth.sigmaX, rtol=0.2)
        self.assertFloatsAlmostEqual(fitted.sigmaY, self.truth.sigmaY, rtol=0.2)

        fiberId = int(self.detectorMap.fiberId[0])
        row = 0.5 * self.config.height
        wavelength = self.detectorMap.findWavelength(fiberId, row)
        position = result.psf.getPosition(fiberId, wavelength)
        image = result.psf.computeKernelImage(position)
        self.assertFloatsAlmostEqual(np.sum(image.array), 1.0, atol=1.0e-8)

    def testPersistence(self):
        """The PfsPsf returned by run() round-trips through FITS persistence"""
        task = FitPsfTask(config=self.taskConfig)
        result = task.run(self.exposure, self.detectorMap, self.arcLines)
        position = result.psf.getAveragePosition()
        original = result.psf.computeKernelImage(position)

        exposure = ExposureF(self.detectorMap.bbox)
        exposure.setPsf(result.psf)
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            exposure.writeFits(filename)
            restored = ExposureF(filename)
        recovered = restored.getPsf().computeKernelImage(position)
        self.assertFloatsAlmostEqual(original.array, recovered.array, atol=1.0e-10)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
