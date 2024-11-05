import lsst.utils.tests
from pfs.drp.stella.tests import runTests

from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.observations import Observations
from pfs.datamodel.target import Target
from pfs.drp.stella.datamodel import PfsSingle
from pfs.drp.stella.fitReference import _trapezoidal, FilterCurve

import numpy as np


class FitReferenceTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        try:
            self.np_random = np.random.default_rng(0x981808A8FA8A744C)
        except AttributeError:
            self.np_random = np.random
            self.np_random.seed(0xF6503311)

    def testTrapezoidal(self):
        """Test ``_trapezoidal()`` method."""
        x = np.array([2, 3, 5, 7, 11, 13, 17], dtype=float)
        y = w = np.ones_like(x)
        # assert _exact_ equality
        self.assertEqual(_trapezoidal(x, y, w), x[-1] - x[0])

    def testPhotometer(self):
        """Test ``TransmissionCurve.photometer()`` method"""
        nSamples = 50
        wavelength = np.linspace(400, 1000, num=nSamples)
        expectedFlux = (wavelength - wavelength[0]) * (3 / (wavelength[-1] - wavelength[0]))
        variance = 2 + np.sin((wavelength - wavelength[0]) * (4 * np.pi / (wavelength[-1] - wavelength[0])))
        stddev = np.sqrt(variance)

        covar = np.zeros(shape=(3, nSamples))
        covar[0, :] = variance

        filterCurve = FilterCurve("i2_hsc")
        target = Target(0, 0, "0,0", 0)
        observations = Observations(
            visit=np.zeros(shape=1),
            arm=["b"],
            spectrograph=np.ones(shape=1),
            pfsDesignId=np.zeros(shape=1),
            fiberId=np.zeros(shape=1),
            pfiNominal=np.zeros(shape=(1, 2)),
            pfiCenter=np.zeros(shape=(1, 2)),
        )
        maskHelper = MaskHelper()

        photometries = []
        photoVars = []

        for i in range(1000):
            flux = expectedFlux + stddev * self.np_random.normal(size=nSamples)

            spectrum = PfsSingle(
                target=target,
                observations=observations,
                wavelength=wavelength,
                flux=flux,
                mask=np.zeros(shape=nSamples, dtype=int),
                sky=np.zeros(shape=nSamples, dtype=int),
                covar=covar,
                covar2=np.zeros(shape=(1, 1), dtype=int),
                flags=maskHelper,
            )

            photo, error = filterCurve.photometer(spectrum, doComputeError=True)
            photometries.append(photo)
            photoVars.append(error**2)

        measuredPhotoVar = np.var(photometries, ddof=1)
        estimatedPhotoVar = np.mean(photoVars)
        self.assertAlmostEqual(measuredPhotoVar, estimatedPhotoVar, places=1)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
