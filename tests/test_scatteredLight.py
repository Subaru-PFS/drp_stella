import numpy as np

import lsst.utils.tests
import lsst.afw.image

from pfs.datamodel import Identity
from pfs.drp.stella import SpectrumSet
from pfs.drp.stella.scatteredLight import ScatteredLightTask
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella.tests.utils import runTests


class ScatteredLightTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.synthConfig = SyntheticConfig()
        self.detectorMap = makeSyntheticDetectorMap(self.synthConfig)
        height = self.synthConfig.height

        spectra = SpectrumSet(len(self.detectorMap), height)
        for ii, fiberId in enumerate(self.detectorMap.fiberId):
            spectrum = spectra[ii]
            spectrum.fiberId = fiberId
            spectrum.flux = np.full(height, 1000.0, dtype=np.float32)
            spectrum.variance = np.full(height, 1000.0, dtype=np.float32)
            spectrum.norm = np.ones(height, dtype=np.float32)
            spectrum.wavelength = self.detectorMap.getWavelength(fiberId)
            spectra[ii] = spectrum
        self.pfsArm = spectra.toPfsArm(Identity(visit=12345, arm="r", spectrograph=1))

        self.image = lsst.afw.image.MaskedImageF(self.synthConfig.width, height)
        self.image.image.array[:] = 0.0

    def makeTask(self, top=None, bottom=None):
        """Make a ScatteredLightTask with a small kernel

        Parameters
        ----------
        top, bottom : `float`, optional
            Scale factors for camera ``r1``; the defaults if not given.
        """
        config = ScatteredLightTask.ConfigClass()
        config.halfSize = 64
        if top is not None:
            config.top["r1"] = top
        if bottom is not None:
            config.bottom["r1"] = bottom
        return ScatteredLightTask(config=config)

    def testMetadata(self):
        """The applied scale factors and the subtracted model are in the
        task metadata"""
        task = self.makeTask()
        model = task.run(self.image, self.pfsArm, self.detectorMap).model

        metadata = task.metadata
        self.assertEqual(metadata["top"], task.config.top["r1"])
        self.assertEqual(metadata["bottom"], task.config.bottom["r1"])
        self.assertGreater(metadata["modelMean"], 0.0)
        self.assertFloatsAlmostEqual(metadata["modelMean"], np.mean(model.array), rtol=1.0e-6)
        self.assertFloatsAlmostEqual(metadata["modelMax"], np.max(model.array), rtol=1.0e-6)
        self.assertGreater(metadata["modelMax"], metadata["modelMean"])
        # The model is what was subtracted from the image
        self.assertFloatsAlmostEqual(np.mean(self.image.image.array), -metadata["modelMean"], rtol=1.0e-5)

    def testMetadataZeroScale(self):
        """Nothing subtracted: the metadata says so"""
        task = self.makeTask(top=0.0, bottom=0.0)
        self.assertIsNone(task.run(self.image, self.pfsArm, self.detectorMap).model)

        metadata = task.metadata
        self.assertEqual(metadata["top"], 0.0)
        self.assertEqual(metadata["bottom"], 0.0)
        self.assertEqual(metadata["modelMean"], 0.0)
        self.assertEqual(metadata["modelMax"], 0.0)
        self.assertFloatsEqual(self.image.image.array, 0.0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
