from pfs.drp.stella.fluxModelInterpolator import FluxModelInterpolator
from pfs.drp.stella.fluxModelInterpolator import PCACompositor, PCAFluxModelInterpolator
from pfs.datamodel.wavelengthArray import WavelengthArray
import lsst.utils
import lsst.utils.tests

import astropy.io.fits
import numpy as np
import scipy.interpolate

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
        spectrum = model.interpolate(teff=7777, logg=3.333, m=0.555, alpha=0.222)
        self.assertIsInstance(spectrum.wavelength, WavelengthArray)


class PCAFluxModelInterpolatorTestCase(lsst.utils.tests.TestCase):
    """Test PCAFluxModelInterpolator with a small synthetic model"""

    def setUp(self):
        rng = np.random.default_rng(12345)
        numModels = 50
        numComponents = 20
        length = 100
        params = np.column_stack(
            [
                rng.uniform(5000, 8000, numModels),
                rng.uniform(3, 5, numModels),
                rng.uniform(-2, 0.5, numModels),
                rng.uniform(0, 0.4, numModels),
            ]
        )
        scales = np.array([1.0e-3, 1.0, 1.0, 1.0])
        coeff = rng.normal(size=(numModels, numComponents))
        rbf = scipy.interpolate.RBFInterpolator(params * scales, coeff, kernel="thin_plate_spline")
        header = astropy.io.fits.Header()
        header["CTYPE1"] = "WAVE"
        header["CUNIT1"] = "nm"
        header["CRPIX1"] = 1.0
        header["CRVAL1"] = 400.0
        header["CDELT1"] = 1.0
        compositor = PCACompositor(rng.normal(size=(numComponents, length)), rng.normal(size=length), header)
        self.model = PCAFluxModelInterpolator(rbf, *scales, compositor)
        self.param = dict(teff=6543.0, logg=4.321, m=-0.432, alpha=0.123)

    def testSmallerInterpolator(self):
        """Test that getSmallerInterpolator matches truncating the full model"""
        numComponents = 7
        smaller = self.model.getSmallerInterpolator(numComponents)
        # The RBF should have been truncated, rather than wrapped
        self.assertIsInstance(smaller.interpolator, scipy.interpolate.RBFInterpolator)

        xx = np.array(
            [
                [
                    self.param["teff"] * self.model.teffScale,
                    self.param["logg"] * self.model.loggScale,
                    self.param["m"] * self.model.mScale,
                    self.param["alpha"] * self.model.alphaScale,
                ]
            ]
        )
        fullCoeff = self.model.interpolator(xx)
        self.assertFloatsAlmostEqual(smaller.interpolator(xx), fullCoeff[:, :numComponents], atol=1.0e-12)

        expected = self.model.pcaCompositor.mean + fullCoeff[0, :numComponents] @ (
            self.model.pcaCompositor.basis[:numComponents]
        )
        spectrum = smaller.interpolate(**self.param)
        self.assertFloatsAlmostEqual(spectrum.flux, expected, atol=1.0e-12)
        self.assertFloatsEqual(spectrum.wavelength, self.model.wavelength)

        # The original should be unaffected
        self.assertEqual(self.model.interpolator(xx).shape, fullCoeff.shape)


def setup_module(module):
    lsst.utils.tests.init()
