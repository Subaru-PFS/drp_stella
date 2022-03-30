import lsst.utils.tests
from pfs.drp.stella.extinctionCurve import F99ExtinctionCurve
from pfs.drp.stella.tests import runTests

import numpy as np


class F99ExtinctionCurveTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.extinctionCurve = F99ExtinctionCurve()

    def testAttenuation(self):
        """Test attenuation()
        """
        # Table 3 in Fitzpatrick (1999)
        reference = np.array([
            (260, 6.591),
            (270, 6.265),
            (411, 4.315),
            (467, 3.806),
            (547, 3.055),
            (600, 2.688),
            (1220, 0.829),
            (2650, 0.265),
        ], dtype=[("lambda", float), ("A(lambda)/E(B-V)", float)])

        ebv = 0.2
        att = self.extinctionCurve.attenuation(reference["lambda"], ebv)
        aOverE = (-2.5 * np.log10(att)) / ebv

        self.assertFloatsAlmostEqual(aOverE, reference["A(lambda)/E(B-V)"], atol=2e-3)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
