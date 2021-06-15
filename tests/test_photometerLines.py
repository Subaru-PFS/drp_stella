import numpy as np

import lsst.utils.tests
import lsst.afw.image
import lsst.afw.image.testUtils

import pfs.drp.stella.synthetic
from pfs.drp.stella.photometerLines import PhotometerLinesTask, PhotometerLinesConfig
from pfs.drp.stella import ReferenceLineSet, ReferenceLineStatus
from pfs.drp.stella.tests.utils import runTests, methodParameters

display = None


class PhotometerLinesTestCase(lsst.utils.tests.TestCase):
    @methodParameters(numLines=(10, 100, 200),
                      rtol=(3.0e-3, 4.0e-3, 5.0e-3))
    def testPhotometry(self, numLines, rtol):
        """Test photometry on an arc

        Parameters
        ----------
        numLines : `int`
            Number of emission lines per fiber. Since the size is fixed, this
            affects the density of lines.
        rtol : `float`
            Relative tolerance for flux measurements.
        """
        description = "Simulated"
        intensity = 123456.789
        fwhm = 3.21
        synthConfig = pfs.drp.stella.synthetic.SyntheticConfig()
        synthConfig.fwhm = fwhm
        synthConfig.height = 432
        rng = np.random.RandomState(12345)
        arc = pfs.drp.stella.synthetic.makeSyntheticArc(synthConfig, numLines=numLines,
                                                        fwhm=fwhm, flux=intensity, rng=rng,
                                                        addNoise=False)
        detMap = pfs.drp.stella.synthetic.makeSyntheticDetectorMap(synthConfig)

        referenceLines = ReferenceLineSet.empty()
        fiberId = detMap.fiberId[detMap.getNumFibers()//2]
        rng.shuffle(arc.lines)  # Ensure we get some cases of multiple blends
        for yy in arc.lines:
            wavelength = detMap.getWavelength(fiberId, yy)
            referenceLines.append(description, wavelength, intensity, ReferenceLineStatus.GOOD)

        exposure = lsst.afw.image.makeExposure(lsst.afw.image.makeMaskedImage(arc.image))
        exposure.mask.set(0)
        exposure.variance.set(synthConfig.readnoise)  # Ignore Poisson noise; not relevant

        config = PhotometerLinesConfig()
        config.fwhm = fwhm
        config.doSubtractContinuum = False  # No continuum, don't want to bother with fiberProfile creation
        task = PhotometerLinesTask(name="photometer", config=config)
        lines = task.run(exposure, referenceLines, detMap)

        self.assertEqual(len(lines), detMap.getNumFibers()*len(referenceLines))
        expectFiberId = np.concatenate([synthConfig.fiberId for _ in referenceLines.wavelength])
        self.assertFloatsEqual(lines.fiberId, expectFiberId)
        expectWavelength = sum(([wl]*synthConfig.numFibers for wl in referenceLines.wavelength), [])
        self.assertFloatsEqual(lines.wavelength, expectWavelength)
        xyExpect = detMap.findPoint(lines.fiberId, lines.wavelength)
        self.assertFloatsEqual(lines.x, xyExpect[:, 0])
        self.assertFloatsEqual(lines.y, xyExpect[:, 1])
        self.assertTrue(np.all(np.isnan(lines.xErr)))
        self.assertTrue(np.all(np.isnan(lines.yErr)))
        self.assertFloatsAlmostEqual(lines.intensity, intensity, rtol=rtol)
        self.assertTrue(np.all(lines.intensityErr > 0))
        self.assertFloatsEqual(lines.flag, 0)
        self.assertFloatsEqual(lines.status, int(ReferenceLineStatus.GOOD))
        self.assertListEqual(lines.description.tolist(), [description]*len(lines))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
