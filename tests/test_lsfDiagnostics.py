"""Claude: tests of pfs.drp.stella.lsfDiagnostics on a synthetic ArmLsf"""
import types
import unittest

import numpy as np
import pandas as pd

import lsst.utils.tests
from pfs.drp.stella.armLsf import ArmLsf, DetectorCoords, FiberSpline
from pfs.drp.stella.lsfDiagnostics import (chi2Breakdown, checkSpectraLsf, gaussianEquivalents, lineResiduals,
                                           parameterBreakdown, robustStats, selectUsedLines,
                                           summarizeResiduals, withGroupOffsets)
from pfs.drp.stella.lsfLine import LsfLineFlag
from pfs.drp.stella.smoothSurface import SmoothSurface
from pfs.drp.stella.spectraLsf import BoundArmLsf

LENGTH = 1000
FIBER_ID = np.array([10, 20, 30])


def constantSurface(name, value):
    """A `SmoothSurface` with a constant value"""
    return SmoothSurface(name, (0, 0), ((0.0, 4096.0), (0.0, 4176.0)), np.array([value]), {}, {})


def makeModel(halo=False):
    """An `ArmLsf` with constant parameters"""
    values = dict(radius=1.8, rms=0.65, S=0.1, K=0.5, centroid=0.0)
    if halo:
        values.update(haloFrac=0.05, haloWidth=3.0)
    surfaces = {name: constantSurface(name, value) for name, value in values.items()}
    xCenter = FiberSpline.fromFunction(FIBER_ID, LENGTH, 32, lambda ff, rows: 100.0*ff + 0.01*rows)
    wavelength = FiberSpline.fromFunction(FIBER_ID, LENGTH, 32, lambda ff, rows: 600 + 0.05*rows)
    return ArmLsf("r", 1, FIBER_ID, LENGTH, surfaces, DetectorCoords(2048.0), xCenter, wavelength)


def makeSpectra(model, lines, background=10.0, amplitude=1000.0):
    """Noiseless spectra made of the model's kernels at the lines"""
    flux = np.full((len(FIBER_ID), LENGTH), background)
    for row in lines.itertuples():
        ii = int(np.flatnonzero(FIBER_ID == row.fiberId)[0])
        kernel = model.computeKernel(row.fiberId, row.pixel, halfWidth=20)
        flux[ii, kernel.start:kernel.start + len(kernel.values)] += amplitude*kernel.values
    wavelength = np.array([model.wavelength(ff, np.arange(LENGTH, dtype=float)) for ff in FIBER_ID])
    return types.SimpleNamespace(fiberId=FIBER_ID, length=LENGTH, flux=flux, variance=flux.copy(),
                                 mask=np.zeros_like(flux, dtype=int), wavelength=wavelength)


def makeLines():
    """A few lines in each fibre, from two visits"""
    rows = []
    for visit in (1, 2):
        for ff in FIBER_ID:
            for pixel in (200.3, 500.0, 750.7):
                rows.append(dict(visit=visit, fiberId=ff, pixel=pixel, description="Ne I"))
    return pd.DataFrame(rows)


class LsfDiagnosticsTestCase(lsst.utils.tests.TestCase):
    def testResiduals(self):
        """Lines made from the model's kernels have no residuals; a visit's offsets change the kernels"""
        model = makeModel()
        lines = makeLines()
        spectra = makeSpectra(model, lines)
        residuals = lineResiduals(model, lines, lambda visit: spectra)
        self.assertEqual(len(residuals), len(lines))
        for rr in residuals:
            self.assertFloatsAlmostEqual(rr.residuals, 0.0, atol=1e-9)
            self.assertGreater(rr.noise, 0.0)
        summary = summarizeResiduals(residuals)
        self.assertEqual(list(summary), ["Ne"])
        self.assertEqual(summary["Ne"]["num"], len(lines))
        self.assertFloatsAlmostEqual(summary["Ne"]["maxAbs"], 0.0, atol=1e-9)

        # Give visit 2 a wider blur: its lines are misfit by the average model, but not by its own
        model.surfaces["rms"].offsets["visit"] = {"1": (0.0, 0.0), "2": (0.1, 0.0)}
        wide = withGroupOffsets(model, "visit", "2")
        self.assertFloatsAlmostEqual(wide.getParameters(10, 500.0)["rms"].value,
                                     model.getParameters(10, 500.0)["rms"].value + 0.1, rtol=1e-12)
        spectra2 = makeSpectra(wide, lines[lines.visit == 2])
        self.assertGreater(np.max(np.abs(lineResiduals(model, lines[lines.visit == 2],
                                                       lambda visit: spectra2)[0].residuals)), 1e-3)
        for rr in lineResiduals(model, lines[lines.visit == 2], lambda visit: spectra2, perVisit=True):
            self.assertFloatsAlmostEqual(rr.residuals, 0.0, atol=1e-9)

    def testHaloOffsets(self):
        """withGroupOffsets includes the halo surfaces' offsets"""
        model = makeModel(halo=True)
        model.surfaces["haloFrac"].offsets["visit"] = {"2": (0.01, 0.0)}
        params = withGroupOffsets(model, "visit", "2").getParameters(20, 300.0)
        self.assertFloatsAlmostEqual(params["haloFrac"].value, 0.06, rtol=1e-12)
        self.assertFloatsAlmostEqual(model.getParameters(20, 300.0)["haloFrac"].value, 0.05, rtol=1e-12)

    def testGaussianEquivalents(self):
        """The widths are ordered as for the measured LSFs, and fit agrees with computeGaussianSigma"""
        model = makeModel()
        widths = gaussianEquivalents(model, 20, 500.0)
        self.assertFloatsAlmostEqual(widths["fit"], float(model.computeGaussianSigma(20, 500.0, "fit")),
                                     rtol=1e-12)
        self.assertEqual(set(widths), {"fwhm", "fit", "fit(1)", "fit(2)", "fit(4)", "moment", "core"})
        self.assertGreater(widths["fwhm"], widths["fit"])
        self.assertLess(widths["core"], widths["moment"])

    def testCheckSpectraLsf(self):
        """A pfsArm made of its own LSF's kernels: no position, width or skewness errors"""
        model = makeModel()
        lines = makeLines()
        lines = lines[lines.visit == 1]
        spectra = makeSpectra(model, lines, amplitude=1e6)
        lsf = BoundArmLsf(model, FIBER_ID, spectra.wavelength)
        wavelength = np.array([model.wavelength(rr.fiberId, rr.pixel) for rr in lines.itertuples()])
        arcLines = types.SimpleNamespace(fiberId=lines.fiberId.to_numpy(), y=lines.pixel.to_numpy(),
                                         yErr=np.full(len(lines), 0.01), wavelength=wavelength,
                                         description=np.full(len(lines), "Ne I"),
                                         flag=np.zeros(len(lines), dtype=bool))
        result = checkSpectraLsf(spectra, lsf, arcLines, fiberStep=1)
        self.assertEqual(len(result.position), len(lines))
        self.assertFloatsAlmostEqual(result.position, 0.0, atol=1e-3)
        self.assertEqual(len(result.lines), len(lines))
        self.assertFloatsAlmostEqual(result.width, 1.0, atol=1e-6)
        self.assertFloatsAlmostEqual(result.skew, 0.0, atol=1e-6)

    def testSelectAndBreakdown(self):
        """Unused and rejected lines are dropped; the breakdowns bin the rest"""
        rng = np.random.default_rng(12345)
        num = 400
        df = pd.DataFrame(dict(x=rng.uniform(0, 4096, num), y=rng.uniform(0, 4176, num),
                               chi2=rng.uniform(0.5, 1.5, num)*10, ndof=np.full(num, 10),
                               flux=rng.uniform(100, 1e5, num), fluxErr=np.full(num, 10.0),
                               visit=rng.choice([1, 2], num), description="Ne I",
                               fiberId=rng.choice(FIBER_ID, num), rms=np.full(num, 0.65),
                               flag=np.zeros(num, dtype=int)))
        df.loc[:9, "flag"] = int(LsfLineFlag.SATURATED)
        df.loc[10:19, "flag"] = int(LsfLineFlag.REJECTED_RMS)
        used = selectUsedLines(df)
        self.assertEqual(len(used), num - 20)
        bright = selectUsedLines(df, minSn=300)
        self.assertEqual(len(bright), np.sum((df.flag == 0) & (df.flux/df.fluxErr > 300)))
        breakdown = chi2Breakdown(used, length=4176)
        self.assertEqual(breakdown["num"], len(used))
        self.assertFloatsAlmostEqual(breakdown["median"], float(np.median(used.chi2/used.ndof)), rtol=1e-12)
        self.assertEqual(sum(nn for _, _, nn in breakdown["x"]), len(used))
        self.assertEqual({vv for vv, _, _, _ in breakdown["visit"]}, {1, 2})
        table = parameterBreakdown(used, "y", 4176)
        self.assertEqual(table.num.sum(), len(used))
        self.assertFloatsAlmostEqual(table.rms.to_numpy(), 0.65, rtol=1e-12)

    def testRobustStats(self):
        median, scatter, num = robustStats([np.nan])
        self.assertTrue(np.isnan(median) and np.isnan(scatter))
        self.assertEqual(num, 0)
        median, scatter, num = robustStats([1.0, 2.0, 3.0, np.nan])
        self.assertEqual((median, num), (2.0, 3))
        self.assertFloatsAlmostEqual(scatter, 1.4826, rtol=1e-12)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
