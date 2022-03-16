from pfs.drp.stella.makeFluxModelInterpolator import makeFluxModelInterpolator
import lsst.utils.tests


class MakeFluxModelInterpolatorTestCase(lsst.utils.tests.TestCase):
    def test(self):
        # TODO: Tests must be done.
        self.assertIsNotNone(makeFluxModelInterpolator)


def setup_module(module):
    lsst.utils.tests.init()
