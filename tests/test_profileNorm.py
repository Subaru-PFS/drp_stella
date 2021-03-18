import numpy as np

import lsst.utils.tests
import lsst.afw.image.testUtils

from pfs.drp.stella.synthetic import makeSyntheticFlat, SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella.buildFiberProfiles import BuildFiberProfilesTask
from pfs.drp.stella.tests import runTests

display = None


class ProfileNormTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(12345)  # I have the same combination on my luggage

    def testProfileNorm(self):
        """Test that the profiles get normalised correctly

        This was created for PIPE2D-782 ("Spectra extracted from data used to
        construct fiberTraces are not flat"), wherein it was noticed that
        extracting a spectrum with a normalisation got a different result than
        extracting a spectrum without a normalisation and multiplying by the
        original spectrum.
        """
        flux = 1.0e5

        synthConfig = SyntheticConfig()
        synthConfig.height = 256
        synthConfig.width = 128
        synthConfig.fwhm = 3.21
        synthConfig.separation = 10
        synthConfig.slope = 0.05

        profConfig = BuildFiberProfilesTask.ConfigClass()
        profConfig.pruneMinLength = synthConfig.height//2
        profConfig.centroidRadius = 3
        profConfig.profileRadius = 3
        profConfig.doBlindFind = False

        detMap = makeSyntheticDetectorMap(synthConfig)
        image = makeSyntheticFlat(synthConfig, flux=flux, addNoise=False, rng=self.rng)
        image = lsst.afw.image.makeMaskedImage(image)
        image.mask.array[:] = 0x0
        image.variance.array[:] = synthConfig.readnoise**2

        if display:
            from lsst.afw.display import Display
            Display(backend=display, frame=1).mtv(image, title="Synthetic image")

        task = BuildFiberProfilesTask(config=profConfig)
        fiberProfiles = task.run(lsst.afw.image.makeExposure(image), detMap).profiles
        fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detMap)

        spectra = fiberTraces.extractSpectra(image)

        for ss in spectra:
            fiberProfiles[ss.fiberId].norm = ss.flux
        traces = fiberProfiles.makeFiberTracesFromDetectorMap(detMap)
        spectra = traces.extractSpectra(image)
        for ss in spectra:
            self.assertFloatsAlmostEqual(ss.flux, 1.0, atol=5.0e-8)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
