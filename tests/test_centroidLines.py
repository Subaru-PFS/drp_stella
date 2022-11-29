import numpy as np

import lsst.utils.tests
import lsst.afw.image
import lsst.afw.image.testUtils

import pfs.drp.stella.synthetic
from pfs.drp.stella.centroidLines import CentroidLinesTask, CentroidLinesConfig
from pfs.drp.stella import ReferenceLine, ReferenceLineSet, ReferenceLineStatus
from pfs.drp.stella import ReferenceLineSource
from pfs.drp.stella.tests.utils import runTests, methodParametersProduct

display = None


class CentroidLinesTestCase(lsst.utils.tests.TestCase):
    @methodParametersProduct(fwhm=(3.21, 1.987), dense=(False, True))
    def testCentroiding(self, fwhm, dense):
        """Test centroiding on an arc"""
        if dense:
            numLines = 300  # Lots of lines, to simulate blending
            doSubtractTraces = False  # Lines are so dense that we can't subtract traces
        else:
            numLines = 50  # Fewer lines, to test trace subtraction
            doSubtractTraces = True
        description = "Simulated"
        flux = 123456.789
        synthConfig = pfs.drp.stella.synthetic.SyntheticConfig()
        synthConfig.fwhm = fwhm
        rng = np.random.RandomState(12345)
        arc = pfs.drp.stella.synthetic.makeSyntheticArc(synthConfig, numLines=numLines, fwhm=fwhm,
                                                        flux=flux, rng=rng, addNoise=False)
        detMap = pfs.drp.stella.synthetic.makeSyntheticDetectorMap(synthConfig)

        referenceLines = []
        fiberId = detMap.fiberId[detMap.getNumFibers()//2]
        for yy in arc.lines:
            wavelength = detMap.getWavelength(fiberId, yy)
            referenceLines.append(
                ReferenceLine(
                    description=description,
                    wavelength=wavelength,
                    intensity=flux,
                    status=ReferenceLineStatus.GOOD,
                    transition="UNKNOWN",
                    source=ReferenceLineSource.NONE,
                )
            )
        referenceLines = ReferenceLineSet.fromRows(referenceLines)

        exposure = lsst.afw.image.makeExposure(lsst.afw.image.makeMaskedImage(arc.image))
        exposure.mask.set(0)
        exposure.variance.set(synthConfig.readnoise)  # Ignore Poisson noise; not relevant

        config = CentroidLinesConfig()
        config.fwhm = fwhm
        config.doSubtractContinuum = False  # No continuum, don't want to bother with fiberProfile creation
        config.doSubtractTraces = doSubtractTraces
        task = CentroidLinesTask(name="centroiding", config=config)
        lines = task.run(exposure, referenceLines, detMap, seed=12345)

        if display is not None:
            from lsst.afw.display import Display
            disp = Display(backend=display, frame=1)
            disp.mtv(exposure)
            with disp.Buffering():
                for ll in lines:
                    disp.dot("o", ll.x, ll.y)

        self.assertEqual(len(lines), detMap.getNumFibers()*len(referenceLines))
        xyExpect = np.array([detMap.findPoint(ff, wl) for ff, wl in zip(lines.fiberId, lines.wavelength)])
        self.assertFloatsAlmostEqual(lines.x, xyExpect[:, 0], atol=2.0e-2)
        self.assertFloatsAlmostEqual(lines.y, xyExpect[:, 1], atol=2.0e-2)
        self.assertTrue(np.all((lines.xErr > 0) & (lines.xErr < 0.1)))
        self.assertTrue(np.all((lines.yErr > 0) & (lines.yErr < 0.1)))
        self.assertFloatsAlmostEqual(lines.flux, flux, rtol=1.0e-2)
        self.assertTrue(np.all(lines.fluxErr > 0))
        self.assertFloatsEqual(lines.flag, 0)
        self.assertFloatsEqual(lines.status, int(ReferenceLineStatus.GOOD))
        self.assertListEqual(lines.description.tolist(), [description]*len(lines))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
