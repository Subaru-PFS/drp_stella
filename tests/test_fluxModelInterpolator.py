from pfs.drp.stella.fluxModelInterpolator import FluxModelInterpolator
from pfs.datamodel.wavelengthArray import WavelengthArray
import lsst.utils
import lsst.utils.tests

import os


class FluxModelInterpolatorTestCase(lsst.utils.tests.TestCase):
    def test(self):
        try:
            dataDir = lsst.utils.getPackageDir("fluxmodeldata")
        except LookupError:
            self.skipTest("fluxmodeldata not setup")

        if not os.path.exists(os.path.join(dataDir, "interpolator.pickle")):
            self.skipTest("makeFluxModelInterpolator.py has not been run")

        model = FluxModelInterpolator.fromFluxModelData(dataDir)
        spectrum = model.interpolate(teff=7777, logg=3.333, metal=0.555, alpha=0.222)
        self.assertIsInstance(spectrum.wavelength, WavelengthArray)


def setup_module(module):
    lsst.utils.tests.init()
