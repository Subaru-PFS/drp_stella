import numpy as np

import lsst.utils.tests
import lsst.afw.image
import lsst.afw.image.testUtils

import pfs.drp.stella.synthetic
from pfs.drp.stella.centroidLines import CentroidLinesTask, CentroidLinesConfig
from pfs.drp.stella import ReferenceLine
from pfs.drp.stella.tests.utils import runTests, methodParameters

display = None


class CentroidLinesTestCase(lsst.utils.tests.TestCase):
    @methodParameters(fwhm=(3.21, 1.987))
    def testCentroiding(self, fwhm):
        """Test centroiding on an arc"""
        description = "Simulated"
        centroidErr = 0.05
        synthConfig = pfs.drp.stella.synthetic.SyntheticConfig()
        rng = np.random.RandomState(12345)
        arc = pfs.drp.stella.synthetic.makeSyntheticArc(synthConfig, fwhm=fwhm, rng=rng)
        detMap = pfs.drp.stella.synthetic.makeSyntheticDetectorMap(synthConfig)

        referenceLines = {}
        for fiberId in synthConfig.fiberId:
            fiberLines = []
            for yy in arc.lines:
                wavelength = detMap.getWavelength(fiberId, yy)
                line = ReferenceLine(description, status=ReferenceLine.FIT, wavelength=wavelength)
                line.fitPosition = yy + rng.normal(0.0, centroidErr)
                line.fitPositionErr = centroidErr
                fiberLines.append(line)
            referenceLines[fiberId] = fiberLines

        exposure = lsst.afw.image.makeExposure(lsst.afw.image.makeMaskedImage(arc.image))
        exposure.mask.set(0)
        exposure.variance.set(synthConfig.readnoise)  # Ignore Poisson noise; not relevant

        config = CentroidLinesConfig()
        config.fwhm = fwhm
        task = CentroidLinesTask(name="centroiding", config=config)
        lines = task.run(exposure, referenceLines, detMap)

        self.assertEqual(len(lines), sum(len(fiberLines) for fiberLines in referenceLines.values()))
        xyExpect = np.array([detMap.findPoint(ff, wl) for ff, wl in zip(lines.fiberId, lines.wavelength)])
        self.assertFloatsAlmostEqual(lines.x, xyExpect[:, 0], atol=2.0e-2)
        self.assertFloatsAlmostEqual(lines.y, xyExpect[:, 1], atol=2.0e-2)
        self.assertTrue(np.all((lines.xErr > 0) & (lines.xErr < 0.1)))
        self.assertTrue(np.all((lines.yErr > 0) & (lines.yErr < 0.1)))
        self.assertFloatsEqual(lines.flag, 0)
        self.assertFloatsEqual(lines.status, int(ReferenceLine.FIT))
        self.assertListEqual(lines.description.tolist(), [description]*len(lines))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
