import lsst.utils
import lsst.utils.tests
from pfs.datamodel.pfsSimpleSpectrum import PfsSimpleSpectrum
from pfs.datamodel.wavelengthArray import WavelengthArray
from pfs.drp.stella.fluxModelSet import FluxModelSet
from pfs.drp.stella.tests import runTests

import unittest

try:
    dataDir = lsst.utils.getPackageDir("fluxmodeldata")
except LookupError:
    dataDir = None


@unittest.skipIf(dataDir is None, "fluxmodeldata not setup")
class FluxModelSetTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.fluxModelSet = FluxModelSet(dataDir)

    def testGetSpectrum(self):
        """Test getSpectrum()
        """
        parameters = self.fluxModelSet.parameters
        param = parameters[len(parameters) // 2]
        spectrum = self.fluxModelSet.getSpectrum(
            teff=param["teff"], logg=param["logg"], m=param["m"], alpha=param["alpha"],
        )
        self.assertIsInstance(spectrum, PfsSimpleSpectrum)
        self.assertIsInstance(spectrum.wavelength, WavelengthArray)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
