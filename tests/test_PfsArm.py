import os
import numpy as np

import lsst.utils.tests
import lsst.afw.image.testUtils

from pfs.datamodel import Identity, MaskHelper
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticPfsConfig
from pfs.drp.stella.datamodel import PfsArm, PfsMerged
from pfs.drp.stella.tests import runTests, classParameters

display = None


@classParameters(SpectraClass=(PfsArm, PfsMerged))  # PfsMerged should be just like PfsArm
class PfsArmTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(12345)  # I have the same combination on my luggage
        self.synthConfig = SyntheticConfig()
        self.synthConfig.separation = 5.31
        self.pfsDesignId = 987654321
        self.visit = 24680
        self.arm = "r"
        self.spectrograph = 3

        self.pfsConfig = makeSyntheticPfsConfig(self.synthConfig, self.pfsDesignId, self.visit, rng=self.rng)
        num = len(self.pfsConfig)
        shape = (num, self.synthConfig.height)

        self.identity = Identity(self.visit, self.arm, self.spectrograph, self.pfsDesignId)
        self.fiberId = self.synthConfig.fiberId
        self.wavelength = np.vstack([np.linspace(600, 900, self.synthConfig.height)]*num)
        self.flux = self.rng.uniform(size=shape)
        self.mask = self.rng.uniform(high=2**31, size=shape).astype(np.int32)
        self.sky = self.rng.uniform(size=shape)
        self.norm = self.rng.uniform(size=shape)
        self.covar = self.rng.uniform(size=(num, 3, self.synthConfig.height))
        self.flags = MaskHelper()
        self.metadata = dict(FOO=12345, BAR=0.9876)  # FITS keywords get capitalised

    def makeSpectra(self):
        """Construct an instance of the spectral class we're testing"""
        kwargs = {name: getattr(self, name) for name in
                  ("identity", "fiberId", "wavelength", "flux", "mask", "sky", "norm",
                   "covar", "flags", "metadata")}
        return self.SpectraClass(**kwargs)

    def assertSpectra(self, spectra):
        """Check that the spectra match what's expected"""
        for name in ("visit", "arm", "spectrograph", "pfsDesignId"):
            self.assertEqual(getattr(spectra.identity, name), getattr(self, name))
        for name in ("fiberId", "wavelength", "flux", "mask", "sky", "norm", "covar"):
            self.assertFloatsEqual(getattr(spectra, name), getattr(self, name))
        self.assertDictEqual(spectra.flags.flags, self.flags.flags)
        self.assertDictEqual({**self.metadata, **spectra.metadata}, spectra.metadata)

    def testBasic(self):
        """Test basic functionality"""
        spectra = self.makeSpectra()
        self.assertSpectra(spectra)
        self.assertEqual(len(spectra), self.synthConfig.numFibers)

    def testGetitem(self):
        """Test __getitem__"""
        spectra = self.makeSpectra()
        select = self.fiberId % 2 == 0
        sub = spectra[select]
        self.assertEqual(len(sub), select.sum())
        for name in ("fiberId", "wavelength", "flux", "mask", "sky", "norm", "covar"):
            setattr(self, name, getattr(self, name)[select])
        self.assertSpectra(spectra)

    def testSelect(self):
        """Test select method"""
        spectra = self.makeSpectra()
        index = 7
        fiberId = self.fiberId[index]
        sub = spectra.select(self.pfsConfig, fiberId=fiberId)
        self.assertEqual(len(sub), 1)
        self.assertEqual(sub.fiberId[0], fiberId)
        self.assertFloatsEqual(sub.flux[0], self.flux[index])

    def testIo(self):
        """Test I/O"""
        spectra = self.makeSpectra()
        dirName = os.path.splitext(__file__)[0]
        if not os.path.exists(dirName):
            os.makedirs(dirName)

        filename = os.path.join(dirName, spectra.filename)
        if os.path.exists(filename):
            os.remove(filename)
        spectra.write(dirName=dirName)
        self.assertTrue(os.path.exists(filename))
        copy = self.SpectraClass.read(self.identity, dirName=dirName)
        self.assertSpectra(copy)
        os.remove(filename)

    def testPlot(self):
        """Test plotting"""
        spectra = self.makeSpectra()

        import matplotlib.pyplot as plt
        plt.switch_backend("agg")  # In case someone has loaded a different backend that will cause trouble

        figure, axes = spectra.plot(show=False)
        self.assertIsNotNone(figure)
        self.assertIsNotNone(axes)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
