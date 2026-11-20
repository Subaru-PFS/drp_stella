import unittest

import numpy as np

import lsst.utils.tests
from lsst.utils import getPackageDir

from pfs.datamodel import Identity
from pfs.drp.stella import SpectrumSet
from pfs.drp.stella.focalPlaneFunction import ConstantPerFiber
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticPfsConfig
from pfs.drp.stella.tests.utils import runTests

try:
    getPackageDir("drp_pfs_data")
    haveDrpPfsData = True
except LookupError:
    haveDrpPfsData = False


@unittest.skipUnless(haveDrpPfsData, "drp_pfs_data is not set up (it holds the PFI correction models)")
class PfiCorrectionMetadataTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        from pfs.drp.stella.PfiCorrection import PfiCorrectionTask

        self.synthConfig = SyntheticConfig()
        rng = np.random.RandomState(1)
        self.pfsConfig = makeSyntheticPfsConfig(self.synthConfig, 123456789, 12345, rng=rng)
        length = 100
        spectra = SpectrumSet(len(self.pfsConfig), length)
        for ii, fiberId in enumerate(self.pfsConfig.fiberId):
            spectrum = spectra[ii]
            spectrum.fiberId = fiberId
            spectrum.flux = np.full(length, 1000.0, dtype=np.float32)
            spectrum.variance = np.full(length, 1000.0, dtype=np.float32)
            spectrum.norm = np.ones(length, dtype=np.float32)
            spectrum.wavelength = np.linspace(600.0, 900.0, length)
            spectra[ii] = spectrum
        self.pfsArm = spectra.toPfsArm(Identity(visit=12345, arm="r", spectrograph=1))
        self.task = PfiCorrectionTask()

    def assertCorrectionMetadata(self, result):
        """The PFI correction statistics and bad count are in the metadata"""
        metadata = self.task.metadata
        pfiOnly = result.edge + result.blob + result.fibers  # Without the sky norms
        self.assertFloatsAlmostEqual(metadata["correctionMean"], np.nanmean(pfiOnly), rtol=1.0e-12)
        self.assertFloatsAlmostEqual(metadata["correctionStd"], np.nanstd(pfiOnly), rtol=1.0e-12)
        numBad = (~np.isfinite(result.correction)).sum()
        self.assertEqual(metadata["numBad"], numBad)
        bad = (self.pfsArm.mask & self.pfsArm.flags.get("BAD_PFI_CORRECTION")) != 0
        self.assertEqual(np.all(bad, axis=1).sum(), numBad)

    def testMetadata(self):
        """No sky norms: their keys are NaN"""
        result = self.task.run(self.pfsArm, self.pfsConfig)
        self.assertCorrectionMetadata(result)
        self.assertTrue(np.isnan(self.task.metadata["skyNormsMean"]))
        self.assertTrue(np.isnan(self.task.metadata["skyNormsStd"]))

    def testMetadataSkyNorms(self):
        """Sky norms: their statistics are recorded, apart from the PFI correction"""
        fiberId = self.pfsArm.fiberId
        values = np.linspace(-0.02, 0.02, len(fiberId))
        skyNorms = ConstantPerFiber(fiberId, values, np.zeros_like(values))
        result = self.task.run(self.pfsArm, self.pfsConfig, skyNorms)
        self.assertCorrectionMetadata(result)
        self.assertFloatsAlmostEqual(self.task.metadata["skyNormsMean"], np.mean(values), atol=1.0e-12)
        self.assertFloatsAlmostEqual(self.task.metadata["skyNormsStd"], np.std(values), rtol=1.0e-12)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
