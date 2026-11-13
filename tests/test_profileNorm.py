import numpy as np

import lsst.utils.tests
import lsst.afw.image.testUtils

from pfs.datamodel import CalibIdentity
from pfs.drp.stella.synthetic import makeSyntheticFlat, SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella.buildFiberProfiles import BuildFiberProfilesTask
from pfs.drp.stella.fiberProfileSet import FiberProfileSet
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
        synthConfig.separation = 15
        synthConfig.slope = 0.05

        profConfig = BuildFiberProfilesTask.ConfigClass()
        profConfig.pruneMinLength = synthConfig.height//2
        profConfig.profileRadius = 7
        profConfig.doBlindFind = False

        detMap = makeSyntheticDetectorMap(synthConfig)
        image = makeSyntheticFlat(synthConfig, flux=flux, addNoise=False, rng=self.rng)
        image = lsst.afw.image.makeMaskedImage(image)
        image.mask.array[:] = 0x0
        image.variance.array[:] = synthConfig.readnoise**2

        identity = CalibIdentity("2020-01-01", 5, "x", 12345)

        if display:
            from lsst.afw.display import Display
            Display(backend=display, frame=1).mtv(image, title="Synthetic image")

        task = BuildFiberProfilesTask(config=profConfig)
        fiberProfiles = task.run(lsst.afw.image.makeExposure(image), identity, detectorMap=detMap).profiles
        fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detMap)

        spectra = fiberTraces.extractSpectra(image)

        for ss in spectra:
            fiberProfiles[ss.fiberId].norm = ss.normFlux
        traces = fiberProfiles.makeFiberTracesFromDetectorMap(detMap)
        newSpectra = traces.extractSpectra(image)
        for new, old in zip(newSpectra, spectra):
            self.assertFloatsAlmostEqual(new.norm, old.normFlux, rtol=1.0e-6)
            self.assertFloatsAlmostEqual(new.normFlux, 1.0, atol=1.0e-6)

    def testIncompleteSampling(self):
        """Test profile normalisation when the oversampled profile is not
        completely sampled

        When the trace moves by less than a pixel over a swath, many of the
        oversampled profile nodes are not constrained and are masked. The
        normalisation must account for these gaps, else the profile is inflated
        and the extracted ``flux/norm`` is biased low by the fraction of nodes
        that are masked.
        """
        flux = 1.0e5
        oversample = 10

        synthConfig = SyntheticConfig()
        synthConfig.height = 1024
        synthConfig.width = 128
        synthConfig.fwhm = 3.21
        synthConfig.separation = 15
        synthConfig.slope = 0.0005  # Trace moves by 0.15 pixels over a swath

        detMap = makeSyntheticDetectorMap(synthConfig)
        image = makeSyntheticFlat(synthConfig, flux=flux, addNoise=False, rng=self.rng)
        image = lsst.afw.image.makeMaskedImage(image)
        image.mask.array[:] = 0x0
        image.variance.array[:] = synthConfig.readnoise**2

        fiberId = detMap.fiberId
        centers = np.array([detMap.getXCenter(ff) for ff in fiberId])
        norm = np.full_like(centers, flux, dtype=np.float32)
        identity = CalibIdentity("2020-01-01", 5, "x", 12345)
        fiberProfiles = FiberProfileSet.fromImages(
            identity, [image], fiberId, [centers], [norm], radius=7, oversample=oversample, swathSize=300
        )
        masked = np.mean([np.ma.getmaskarray(fiberProfiles[ff].profiles).mean() for ff in fiberId])
        self.assertGreater(masked, 0.5)  # Otherwise we're not testing incomplete sampling

        spectra = fiberProfiles.makeFiberTracesFromDetectorMap(detMap).extractSpectra(image)
        for ss in spectra:
            self.assertFloatsAlmostEqual(ss.normFlux/flux, 1.0, atol=1.0e-2)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
