import numpy as np

import lsst.utils.tests
import lsst.afw.image
import lsst.afw.image.testUtils
from lsst.afw.display import Display

from pfs.datamodel import CalibIdentity
from pfs.drp.stella.synthetic import makeSpectrumImage, SyntheticConfig
from pfs.drp.stella.synthetic import makeSyntheticDetectorMap, makeSyntheticPfsConfig
from pfs.drp.stella.buildFiberProfiles import BuildFiberProfilesTask
from pfs.drp.stella.tests.utils import runTests

display = None


class FiberProfilesTestCase(lsst.utils.tests.TestCase):
    def testProfiles(self):
        rng = np.random.RandomState(12345)

        synth = SyntheticConfig()
        synth.height = 256
        synth.width = 128
        synth.separation = 19.876
        synth.fwhm = 3.45
        synth.slope = 0.05

        flux = 1.0e5

        image = makeSpectrumImage(flux, synth.dims, synth.traceCenters, synth.traceOffset, synth.fwhm)
        mask = lsst.afw.image.Mask(image.getBBox())
        mask.set(0)
        variance = lsst.afw.image.ImageF(image.getBBox())
        variance.array[:] = image.array/synth.gain + synth.readnoise**2
        image = lsst.afw.image.makeMaskedImage(image, mask, variance)

        darkTime = 123.456
        metadata = ("FOO", "BAR")
        visitInfo = lsst.afw.image.VisitInfo(darkTime=darkTime)
        exposure = lsst.afw.image.makeExposure(image)
        exposure.getMetadata().set(*metadata)
        exposure.getInfo().setVisitInfo(visitInfo)

        detMap = makeSyntheticDetectorMap(synth)
        pfsConfig = makeSyntheticPfsConfig(synth, 123456789, 54321, rng, fracSky=0.0, fracFluxStd=0.0)
        identity = CalibIdentity("2020-01-01", 5, "x", 12345)

        config = BuildFiberProfilesTask.ConfigClass()
        config.pruneMinLength = int(0.9*synth.height)
        config.profileSwath = 128
        config.profileRadius = 2
        config.profileOversample = 5
        config.rowFwhm = synth.fwhm
        config.columnFwhm = synth.fwhm
        config.doBlindFind = False
        profiles = BuildFiberProfilesTask(config=config).run(
            exposure, identity, detectorMap=detMap, pfsConfig=pfsConfig
        ).profiles

        pp = profiles[21]
        print(pp.index)
        print(pp.profiles[0])
        pp.plot()
        import matplotlib.pyplot as plt
        plt.show()

        traces = profiles.makeFiberTracesFromDetectorMap(detMap)
        spectra = traces.extractSpectra(exposure.maskedImage)
        extracted = spectra.makeImage(exposure.getBBox(), traces)

        Display(frame=1).mtv(exposure, title="Exposure")
        Display(frame=2).mtv(extracted, title="Traces")

        exposure.maskedImage -= extracted

        Display(frame=3).mtv(exposure, title="Residuals")


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
