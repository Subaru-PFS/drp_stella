import numpy as np
import pandas as pd

import lsst.utils.tests

from pfs.datamodel import Identity, MaskHelper, PfsMerged
from pfs.drp.stella.lsf import GaussianLsf
from pfs.drp.stella.referenceLine import ReferenceLineSet, ReferenceLineSource
from pfs.drp.stella.lineLists.intensities import (fitLineIntensities, updateLineIntensities, combineFibres,
                                                  integratedFluxes, scaleUnmeasured)
from pfs.drp.stella.tests.utils import runTests

display = None
EMPIRICAL = int(ReferenceLineSource.EMPIRICAL_INTENSITY)


def makeLineList(wavelengths, intensity=1.0, description="NeI", source=2):
    """Write and read back a minimal line list"""
    with lsst.utils.tests.getTempFilePath(".txt") as filename:
        with open(filename, "w") as fd:
            for wl in wavelengths:
                fd.write(f"{wl:.5f} {intensity} {description} 0 UNKNOWN {source}\n")
        return ReferenceLineSet.fromLineList(filename)


class LineIntensitiesTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def testFitLineIntensities(self):
        """Recover the fluxes of synthetic lines on a continuum, with correct errors"""
        length = 3000
        wavelength = np.linspace(600.0, 900.0, length)
        lsf = GaussianLsf(length, 1.2)
        lam = np.round(np.sort(self.rng.uniform(605, 895, 40)), 5)   # as stored in a line list
        lines = makeLineList(lam)
        truth = 10**self.rng.uniform(2, 4, size=lam.size)
        pixel = np.interp(lam, wavelength, np.arange(length))
        flux = 50 + 20*np.sin(wavelength/30.0)
        for pp, ff in zip(pixel, truth):
            arr = lsf.computeArray(pp)
            flux += ff*arr/arr.sum()
        variance = flux.copy()
        flux += self.rng.normal(size=length)*np.sqrt(variance)
        flux[1500:1505] = np.nan

        newLines, result, model, continuum = fitLineIntensities(wavelength, flux, lsf, lines, variance,
                                                                rescale=False)
        index = {ll: ii for ii, ll in enumerate(lam)}
        tt = truth[[index[ll] for ll in result.wavelength]]
        iso = (result.fitted & (result.group < 0)).to_numpy()
        self.assertGreater(iso.sum(), 25)
        chi = (result.flux.to_numpy()[iso] - tt[iso])/result.fluxErr.to_numpy()[iso]
        self.assertLess(abs(np.mean(chi)), 0.6)
        self.assertLess(abs(np.std(chi) - 1), 0.4)

        # EMPIRICAL_INTENSITY is set exactly on the fitted lines
        fitted = result.fitted.to_numpy()
        source = newLines.data.source.to_numpy()[[index[ll] for ll in result.wavelength]]
        self.assertTrue(np.all(source[fitted] & EMPIRICAL))
        self.assertFalse(np.any(source[~fitted] & EMPIRICAL))

        # lamMax: lines beyond it keep their catalogue intensity and source
        newCut, _, _, _ = fitLineIntensities(wavelength, flux, lsf, lines, variance, lamMax=800.0,
                                             rescale=False)
        red = lam > 800
        np.testing.assert_array_equal(newCut.data.intensity.to_numpy()[red],
                                      lines.data.intensity.to_numpy()[red])
        np.testing.assert_array_equal(newCut.data.source.to_numpy()[red], lines.data.source.to_numpy()[red])

    def testUpdateLineIntensities(self):
        """Inverse-variance combination across arms, global scale, and source flags"""
        lines = makeLineList([400.0, 650.0, 900.0, 950.0])
        columns = ["wavelength", "description", "flux", "fluxErr", "fitted"]
        armB = pd.DataFrame([[400.0, "NeI", 100.0, 1.0, True],
                             [650.0, "NeI", 200.0, 4.0, True]], columns=columns)
        armR = pd.DataFrame([[650.0, "NeI", 260.0, 3.0, True],
                             [900.0, "NeI", 50.0, 2.0, True]], columns=columns)
        newLines, combined = updateLineIntensities(lines, [armB, armR], scale=2.0)
        w = np.array([1/16, 1/9])
        expect650 = np.sum(w*np.array([200.0, 260.0]))/np.sum(w)
        byLam = {round(ll.wavelength, 3): ll for ll in newLines.rows}
        self.assertFloatsAlmostEqual(byLam[400.0].intensity, 200.0, rtol=1e-9)
        self.assertFloatsAlmostEqual(byLam[650.0].intensity, 2*expect650, rtol=1e-9)
        self.assertEqual(int(combined.loc[combined.wavelength == 650.0, "nArm"].iloc[0]), 2)
        self.assertEqual(byLam[950.0].intensity, 1.0)   # never fitted: keeps the catalogue value
        self.assertTrue(byLam[400.0].source & EMPIRICAL)
        self.assertFalse(byLam[950.0].source & EMPIRICAL)

    def testCombineFibres(self):
        """Quartz normalisation removes per-fibre throughput, and outlying fibres are clipped"""
        nFiber, length = 20, 100
        wavelength = np.tile(np.linspace(600, 610, length), (nFiber, 1))
        throughput = self.rng.uniform(0.5, 1.5, nFiber)[:, np.newaxis]
        intrinsic = 1000 + 100*np.sin(np.arange(length)/5.0)
        flux = throughput*intrinsic*(1 + 1e-4*self.rng.normal(size=(nFiber, length)))
        flux[3] *= 10                            # an outlier fibre
        norm = throughput*np.ones((1, length))
        variance = np.ones_like(flux)
        flags = MaskHelper(SAT=1)
        spec = PfsMerged(Identity(visit=1), np.arange(nFiber) + 1, wavelength, flux,
                         np.zeros_like(flux, dtype=np.int32), np.zeros_like(flux), norm,
                         np.stack([variance, np.zeros_like(flux), np.zeros_like(flux)], axis=1), flags, {})
        _, combined, _ = combineFibres(spec)
        normBar = np.median(norm)
        self.assertFloatsAlmostEqual(combined, intrinsic*normBar, rtol=2e-4)   # the outlier fibre is clipped

    def testIntegratedFluxes(self):
        """Bright groups get the integrated flux (split by fitted ratio); faint lines keep the fit"""
        wavelength = np.arange(600.0, 620.0, 0.01)
        continuum = np.full_like(wavelength, 10.0)
        flux = continuum.copy()
        sigma = 0.05
        truth = {605.0: 1000.0, 610.0: 300.0, 610.2: 100.0, 615.0: 2.0}
        for wl, f in truth.items():
            flux += f*np.exp(-0.5*((wavelength - wl)/sigma)**2)/np.sqrt(2*np.pi)/sigma*0.01
        variance = np.full_like(wavelength, 0.01)
        # The fit misses 20% of the flux (a profile mismatch), but gets the blend's ratio right
        result = pd.DataFrame(dict(wavelength=list(truth), description="HeI", fitted=True,
                                   flux=[0.8*f for f in truth.values()], fluxErr=[1.0, 1.0, 1.0, 1.0]))
        res = integratedFluxes(result, wavelength, flux, variance, continuum, minSN=30)
        got = dict(zip(res.wavelength, res.flux))
        self.assertFloatsAlmostEqual(got[605.0], 1000.0, rtol=1e-3)
        self.assertFloatsAlmostEqual(got[610.0], 300.0, rtol=1e-3)
        self.assertFloatsAlmostEqual(got[610.2], 100.0, rtol=1e-3)
        self.assertFloatsAlmostEqual(got[615.0], 0.8*2.0, rtol=1e-9)   # S/N < 30: unchanged
        self.assertEqual(dict(zip(res.wavelength, res.method))[610.0], "integrated, split 2")

    def testScaleUnmeasured(self):
        """Unmeasured lines get the previous intensity times the median measured/previous ratio"""
        lam = [600.0, 610.0, 620.0, 630.0, 640.0]
        previous = makeLineList(lam, intensity=100.0)
        data = previous.data.copy()
        data["intensity"] = [50.0, 50.0, 50.0, 1.0, 1.0]
        data["intensityErr"] = [0.1, 0.1, 0.1, 0.1, 0.1]
        data["source"] = data.source | EMPIRICAL
        lines = ReferenceLineSet(data)
        newLines, scales = scaleUnmeasured(lines, previous, measured=[600.0, 610.0, 620.0])
        self.assertFloatsAlmostEqual(scales["NeI"], 0.5, rtol=1e-12)
        byLam = {round(ll.wavelength, 3): ll for ll in newLines.rows}
        self.assertFloatsAlmostEqual(byLam[630.0].intensity, 50.0, rtol=1e-12)
        self.assertTrue(np.isnan(byLam[630.0].intensityErr))
        self.assertFalse(byLam[630.0].source & EMPIRICAL)
        self.assertTrue(byLam[600.0].source & EMPIRICAL)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
