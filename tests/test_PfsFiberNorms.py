import sys
import unittest

import matplotlib
matplotlib.use("Agg")

import astropy.io.fits  # noqa:E402
import numpy as np  # noqa:E402

import lsst.utils.tests  # noqa:E402

from pfs.datamodel.identity import CalibIdentity  # noqa:E402
from pfs.datamodel.pfsConfig import PfsConfig, FiberStatus, TargetType  # noqa:E402
from pfs.datamodel.masks import MaskHelper  # noqa:E402
from pfs.drp.stella.datamodel.pfsFiberNorms import PfsFiberNorms  # noqa:E402
from pfs.drp.stella.datamodel.drp import PfsArm  # noqa:E402
from pfs.drp.stella.utils.display import showAllSpectraAsImage  # noqa:E402


class PfsFiberNormsTestCase(lsst.utils.tests.TestCase):
    """Test for PfsFiberNorms

    This differs from the tests in the datamodel package in that we're only
    testing the features added in drp_stella.
    """
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.length = 123

        self.identity = CalibIdentity(visit0=12345, arm="r", spectrograph=1, obsDate="2024-04-15")
        self.fiberId = np.arange(1, 100, 3, dtype=int)
        self.numFibers = self.fiberId.size
        self.wavelength = self.rng.uniform(size=(self.numFibers, self.length))
        self.values = self.rng.uniform(1, 2, size=(self.numFibers, self.length))
        self.fiberProfilesHash = {1: 0x12345678, 2: 0x23456789}
        self.model = astropy.io.fits.ImageHDU(
            self.rng.uniform(size=1234),
            astropy.io.fits.Header(cards=dict(KEY="VALUE")),
        )
        self.metadata = {"KEY": "VALUE", "NUMBER": 67890}

        self.fiberNorms = PfsFiberNorms(
            self.identity,
            self.fiberId,
            self.wavelength,
            self.values,
            self.fiberProfilesHash,
            self.model,
            self.metadata,
        )

    def testMath(self):
        """Test math operations"""
        values = self.fiberNorms.values.copy()
        self.fiberNorms *= 2
        self.assertTrue(np.array_equal(self.fiberNorms.values, values*2))

        values = self.fiberNorms.values.copy()
        self.fiberNorms /= 4
        self.assertTrue(np.array_equal(self.fiberNorms.values, values/4))

        values = self.fiberNorms.values.copy()
        self.fiberNorms *= self.fiberNorms
        self.assertTrue(np.array_equal(self.fiberNorms.values, values**2))

        values = self.fiberNorms.values.copy()
        self.fiberNorms /= self.fiberNorms
        self.assertTrue(np.array_equal(self.fiberNorms.values, np.ones_like(values)))

        flux = self.rng.uniform(size=(self.numFibers, self.length))
        pfsArm = PfsArm(
            self.identity,
            self.fiberId,
            self.wavelength,
            flux,
            np.zeros_like(flux, dtype=np.int32),  # mask
            self.rng.uniform(size=flux.shape),  # sky
            self.rng.uniform(size=flux.shape),  # norm
            self.rng.uniform(size=(self.numFibers, 3, self.length)),  # covar
            MaskHelper(BAD=0, SAT=1, FOOBAR=2, NO_DATA=3),  # flags
            {},  # metadata
        )

        values = flux.copy()
        pfsArm /= self.fiberNorms
        self.assertTrue(np.array_equal(pfsArm.flux, values/self.fiberNorms.values))

        values = flux.copy()
        pfsArm *= self.fiberNorms
        self.assertTrue(np.array_equal(pfsArm.flux, values*self.fiberNorms.values))

    def testPlot(self):
        """Test plotting"""
        import matplotlib.pyplot as plt
        plt.switch_backend("agg")  # In case someone has loaded a different backend that will cause trouble

        num = 2*self.numFibers
        fiberId = np.concatenate(
            [self.fiberId, np.arange(self.fiberId.max() + 1, self.fiberId.max() + 1 + self.numFibers)]
        )
        rng = np.random.RandomState(12345)

        pfsConfig = PfsConfig(
            0xfeedfacedeadbeef,  # pfsDesignId
            12345,  # visit
            123.456789,  # raBoresight
            -0.123456789,  # decBoresight
            12.3456789,  # posAng
            "brn",  # arms
            fiberId,  # fiberId
            np.zeros(num, dtype=int),  # tract
            ["123,456"]*num,  # patch
            np.zeros(num, dtype=float),  # ra
            np.zeros(num, dtype=float),  # dec
            np.zeros(num, dtype=int),  # catId
            np.arange(num, dtype=int),  # objId
            np.full(num, TargetType.SCIENCE),  # targetType
            np.full(num, FiberStatus.GOOD),  # fiberStatus
            np.full(num, 2000.0),  # epoch
            np.zeros(num, dtype=float),  # pmRa
            np.zeros(num, dtype=float),  # pmDec
            np.zeros(num, dtype=float),  # parallax
            np.zeros(num, dtype=int),  # proposalId
            np.zeros(num, dtype=int),  # obCode
            np.zeros((num, 1), dtype=float),  # fiberFlux
            np.zeros((num, 1), dtype=float),  # psfFlux
            np.zeros((num, 1), dtype=float),  # totalFlux
            np.zeros((num, 1), dtype=float),  # fiberFluxErr
            np.zeros((num, 1), dtype=float),  # psfFluxErr
            np.zeros((num, 1), dtype=float),  # totalFluxErr
            [["V"]]*num,  # filterNames
            rng.uniform(-1, 1, size=(num, 2)),  # pfiCenter
            rng.uniform(-1, 1, size=(num, 2)),  # pfiNominal
            None,  # guideStars
        )
        self.fiberNorms.plot(pfsConfig)
        showAllSpectraAsImage(self.fiberNorms.toPfsFiberArraySet())


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main()
