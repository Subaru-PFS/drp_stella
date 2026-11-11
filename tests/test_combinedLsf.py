"""Claude: the line-spread function of merged and coadded spectra (PIPE2D-1993)

The acceptance test of `CombinedLsf` is that a spectrum made of the input LSFs
(delta functions, convolved with each input's LSF) and combined like a real
spectrum, is the combined LSF.

- Where a single input contributes, the combination is the input resampled, so
  the merged/coadded flux must match the LSF. It does so exactly when resampled
  in double precision; the spectra are resampled in float32, so the flux itself
  matches to ``FLOAT_TOL``.
- Where several inputs overlap, the LSF describes the combination with smooth
  weights. The flux combination currently uses per-pixel weights (which
  oscillate with the phase of the resampling, as the resampled variance does),
  so there we compare with an explicit smooth-weight combination exactly, and
  bound the difference from the actual flux.
"""
import pickle
import unittest

import numpy as np

import lsst.utils.tests
from pfs.datamodel import Identity, MaskHelper, Observations, Target
from pfs.drp.stella.coaddSpectra import CoaddSpectraConfig, CoaddSpectraTask
from pfs.drp.stella.combinedLsf import CombinedLsf, CombinedLsfInput, IndexMap, SmoothWeight
from pfs.drp.stella.datamodel import PfsArm, PfsSingle
from pfs.drp.stella.interpolate import Resampler, interpolateIndices
from pfs.drp.stella.lsf import GaussianLsf
from pfs.drp.stella.mergeArms import MergeArmsConfig, MergeArmsTask

# Arm: wavelength range (nm), length, LSF sigma (pixels), variance (powers of two, for exact float32 weights)
ARMS = dict(
    b=(380.0, 650.0, 4096, 1.5, 4.0),
    r=(630.0, 970.0, 4096, 2.0, 1.0),
    n=(940.0, 1260.0, 4096, 2.5, 2.0),
)
SINGLE = (450.0, 800.0, 1100.0)  # Line wavelengths covered by a single arm
OVERLAP = (640.0, 955.0)  # Line wavelengths in the overlaps between arms
HALF = 40  # Half-size of the region compared around each line (output pixels)
FLOAT_TOL = 1.0e-6  # Tolerance (relative to peak) for spectra resampled in float32


def makeFlags():
    flags = MaskHelper()
    for name in ("NO_DATA", "SUSPECT", "BAD_SKY", "BAD_FLUXCAL", "BAD_FIBERNORMS"):
        flags.add(name)
    return flags


def armWavelength(arm, offset=0.0):
    lo, hi, length, _, _ = ARMS[arm]
    return np.linspace(lo, hi, length) + offset


def lineFlux(lsf, inWavelength, outWavelength, lines, order):
    """Flux of lines, each the input LSF scaled so its resampled sum is unity

    The scaling (the input's gain for that line) is the one degree of freedom
    that the combined LSF doesn't model: it's normalised away.
    """
    resampler = Resampler(inWavelength, outWavelength, order=order)
    flux = np.zeros(len(inWavelength))
    for wl in lines:
        if not inWavelength[0] + 5 < wl < inWavelength[-1] - 5:
            continue
        center = np.interp(wl, inWavelength, np.arange(len(inWavelength)))
        start, values = lsf.computeWindow(center)
        _, resampled = resampler.resampleWindow(values, start)
        flux[start:start + len(values)] += values/np.sum(resampled)
    return flux


class CombinedLsfTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.flags = makeFlags()
        self.wavelength = np.linspace(380.0, 1260.0, 11001)
        self.fiberId = np.array([3, 7])

    def outPixel(self, wavelength):
        """Output (fractional) pixel of a wavelength"""
        return np.interp(wavelength, self.wavelength, np.arange(len(self.wavelength), dtype=float))

    def compareLines(self, data, lsf, lines, rtol, normalize=False):
        """Compare data around lines with the LSF's model

        If ``normalize``, the data around each line is normalised to unit sum
        before comparing: where weights vary across a line, the combination
        doesn't preserve its sum exactly, while the LSF has unit sum.
        """
        for wl in lines:
            center = self.outPixel(wl)
            model = lsf.computeArray(center)
            select = slice(int(center) - HALF, int(center) + HALF)
            values = data[select]/np.sum(data[select]) if normalize else data[select]
            peak = np.max(model[select])
            self.assertGreater(peak, 0.05)
            np.testing.assert_allclose(values, model[select], rtol=0, atol=rtol*peak,
                                       err_msg=f"line at {wl}")
            self.assertFloatsAlmostEqual(np.sum(model), 1.0, atol=1.0e-12)
            # Nothing outside the compared region
            self.assertFloatsAlmostEqual(np.sum(model[select]), 1.0, atol=1.0e-12)

    def makeArms(self, order):
        """PfsArms whose fluxes are lines convolved by their LSFs"""
        arms = []
        lsfs = []
        for arm, (lo, hi, length, sigma, variance) in ARMS.items():
            wavelength = np.array([armWavelength(arm, offset) for offset in (0.0, 0.013)])
            lsf = GaussianLsf(length, sigma)
            lines = SINGLE + OVERLAP
            flux = np.array([lineFlux(lsf, wl, self.wavelength, lines, order) for wl in wavelength])
            num = len(self.fiberId)
            covar = np.zeros((num, 3, length))
            covar[:, 0] = variance
            identity = Identity(visit=1, arm=arm, spectrograph=1, pfsDesignId=0)
            metadata = {"PFS.HASH.FIBERPROFILES": 0}
            mask = np.zeros((num, length), dtype=np.int32)
            arms.append(PfsArm(identity, self.fiberId, wavelength, flux, mask, np.zeros_like(flux),
                               np.ones_like(flux), covar, self.flags, metadata))
            lsfs.append({ff: lsf for ff in self.fiberId})
        return arms, lsfs

    def smoothCombination(self, lsf, fluxes, inWavelengths, order):
        """Combine fluxes with the LSF's smooth weights (Σ W r / Σ W)"""
        pixels = np.arange(len(self.wavelength), dtype=float)
        numerator = np.zeros(len(self.wavelength))
        denominator = np.zeros(len(self.wavelength))
        for inp, flux, inWavelength in zip(lsf.inputs, fluxes, inWavelengths):
            resampled = Resampler(inWavelength, self.wavelength, order=order).resampleFlux(flux)
            weight = inp.weight(pixels)
            numerator += weight*resampled
            denominator += weight
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(denominator > 0, numerator/denominator, 0.0)

    def testMergeArms(self):
        """Merged spectra of the arm LSFs are the merged LSF"""
        for order in (1, 3):
            config = MergeArmsConfig()
            config.resampleOrder = order
            task = MergeArmsTask(config=config)
            arms, lsfs = self.makeArms(order)
            merged = task.mergeSpectra(arms, self.wavelength)
            lsfDict = task.mergeLsfs(lsfs, arms, self.wavelength, merged.weights)
            self.assertEqual(set(lsfDict), set(self.fiberId))
            for ii, ff in enumerate(self.fiberId):
                lsf = lsfDict[ff]
                self.assertIsInstance(lsf, CombinedLsf)
                with np.errstate(invalid="ignore", divide="ignore"):
                    data = merged.pfsMerged.flux[ii]/merged.pfsMerged.norm[ii]
                # Single arm: exact (up to the float32 resampling)
                self.compareLines(data, lsf, SINGLE, FLOAT_TOL)
                # Smooth weights: exact everywhere
                smooth = self.smoothCombination(lsf, [aa.flux[ii] for aa in arms],
                                                [aa.wavelength[ii] for aa in arms], order)
                self.compareLines(smooth, lsf, SINGLE, 1.0e-12)
                self.compareLines(smooth, lsf, OVERLAP, 1.0e-12, normalize=True)
                # The arm weights are 1/variance (the arm norms are unity), increased by the resampling,
                # which reduces the variance by a factor depending on the phase (up to 2 for linear)
                for inp, (_, _, _, _, variance) in zip(lsf.inputs, ARMS.values()):
                    middle = 0.5*(inp.indexMap.minIndex + inp.indexMap.maxIndex)
                    self.assertTrue(1.0 < inp.weight(np.array([middle]))[0]*variance < 2.0)
                # Actual (per-pixel weights): approximate. In this test the differences are up to 2% of the
                # peak, ~0.02 pixels in centroid and 1% in width (linear resampling; half that for Lanczos)
                self.compareLines(data, lsf, OVERLAP, 0.05, normalize=True)

    def testCoadd(self):
        """Coadds of the visit LSFs are the coadd LSF"""
        fiberId = 7
        for order in (1, 3):
            config = CoaddSpectraConfig()
            config.resampleOrder = order
            task = CoaddSpectraTask(config=config)
            singles = []
            lsfList = []
            for visit, offset, varScale in ((1, 0.0, 1.0), (2, 0.031, 2.0), (3, -0.017, 0.5)):
                for arm, (lo, hi, length, sigma, variance) in ARMS.items():
                    wavelength = armWavelength(arm, offset)
                    lsf = GaussianLsf(length, sigma*(1 + 0.1*visit))
                    flux = lineFlux(lsf, wavelength, self.wavelength, SINGLE + OVERLAP, order)
                    covar = np.zeros((3, length))
                    covar[0] = variance*varScale
                    observations = Observations(
                        visit=np.array([visit]), arm=[arm], spectrograph=np.array([1]),
                        pfsDesignId=np.array([0]), fiberId=np.array([fiberId]),
                        pfiNominal=np.zeros((1, 2)), pfiCenter=np.zeros((1, 2)),
                    )
                    singles.append(PfsSingle(Target(0, 0, "0,0", 0), observations, wavelength, flux,
                                             np.zeros(length, dtype=np.int32), np.zeros(length), covar,
                                             np.zeros((1, 1)), self.flags))
                    lsfList.append({fiberId: lsf})

            # A single visit: one input everywhere but the overlaps, so exact (up to the float32 resampling)
            first = task.combine(singles[:3], lsfList[:3], self.flags, self.wavelength)
            self.assertIsInstance(first.lsf, CombinedLsf)
            self.compareLines(first.flux, first.lsf, SINGLE, FLOAT_TOL)

            # Several visits: exact for smooth weights; approximate for the actual (per-pixel) weights
            combination = task.combine(singles, lsfList, self.flags, self.wavelength)
            lsf = combination.lsf
            self.assertEqual(len(lsf.inputs), len(singles))
            smooth = self.smoothCombination(lsf, [ss.flux for ss in singles],
                                            [ss.wavelength for ss in singles], order)
            self.compareLines(smooth, lsf, SINGLE + OVERLAP, 1.0e-12, normalize=True)
            self.compareLines(combination.flux, lsf, SINGLE + OVERLAP, 0.05, normalize=True)

            # No LSF requested
            self.assertIsNone(task.combine(singles, None, self.flags, self.wavelength).lsf)

    def testWarp(self):
        """Warping the LSF is resampling it (linearly)"""
        task = MergeArmsTask(config=MergeArmsConfig())
        arms, lsfs = self.makeArms(1)
        merged = task.mergeSpectra(arms, self.wavelength)
        lsf = task.mergeLsfs(lsfs, arms, self.wavelength, merged.weights)[self.fiberId[0]]
        for fine in (np.linspace(600.0, 1000.0, 20001), np.linspace(370.0, 1300.0, 5001)):  # finer, coarser
            warped = lsf.warp(self.wavelength, fine)
            self.assertEqual(warped.length, len(fine))
            resampler = Resampler(self.wavelength, fine, order=1)
            for wl in (640.0, 800.0, 955.0):
                center = np.interp(wl, fine, np.arange(len(fine), dtype=float))
                expect = resampler.resampleFlux(lsf.computeArray(self.outPixel(wl)))
                expect /= np.sum(expect)
                np.testing.assert_allclose(warped.computeArray(center), expect, rtol=0, atol=1.0e-12)
            # Convolution works, and conserves flux (a delta function in the middle of a block of the
            # convolution, so it sees a single kernel)
            array = np.zeros(len(fine))
            array[100*(len(fine)//200) + 50] = 1.0
            self.assertFloatsAlmostEqual(np.sum(warped.convolve(array)), 1.0, atol=1.0e-12)

    def testKernel(self):
        """computeKernel is the LSF for a line on a pixel center"""
        task = MergeArmsTask(config=MergeArmsConfig())
        arms, lsfs = self.makeArms(3)
        merged = task.mergeSpectra(arms, self.wavelength)
        lsf = task.mergeLsfs(lsfs, arms, self.wavelength, merged.weights)[self.fiberId[0]]
        for wl in SINGLE + OVERLAP:
            center = self.outPixel(wl)
            kernel = lsf.computeKernel(center)
            pixel = int(np.floor(center + 0.5))
            array = lsf.computeArray(pixel)
            np.testing.assert_array_equal(kernel.values, array[pixel + kernel.min:pixel + kernel.max + 1])
            self.assertGreater(kernel.computeStdev(), 1.0)
            self.assertEqual(lsf.computeShape1D(center), kernel.computeStdev())

    def testPickle(self):
        """Pickling preserves the LSF"""
        task = MergeArmsTask(config=MergeArmsConfig())
        arms, lsfs = self.makeArms(1)
        merged = task.mergeSpectra(arms, self.wavelength)
        lsfDict = task.mergeLsfs(lsfs, arms, self.wavelength, merged.weights)
        copy = pickle.loads(pickle.dumps(lsfDict))
        for ff in lsfDict:
            for wl in SINGLE + OVERLAP:
                center = self.outPixel(wl)
                np.testing.assert_array_equal(copy[ff].computeArray(center), lsfDict[ff].computeArray(center))

    def testArmEnd(self):
        """Claude: a line at the end of an arm has the full LSF, as if the detector continued

        The dichroic acts before the LSF is formed, so the end of the data
        doesn't truncate the LSF.
        """
        lo, hi, length, sigma, _ = ARMS["r"]
        step = (hi - lo)/(length - 1)
        extra = 100  # Pixels by which the extended arm continues beyond each end
        inWavelength = armWavelength("r")
        extended = np.linspace(lo - extra*step, hi + extra*step, length + 2*extra)
        lsf = GaussianLsf(length, sigma)
        extendedLsf = GaussianLsf(length + 2*extra, sigma)

        # The input LSF itself
        for center in (0.3, length - 1.3):
            start, values = lsf.computeWindow(center, clip=False)
            expectStart, expect = extendedLsf.computeWindow(center + extra)
            self.assertEqual(start, expectStart - extra)
            np.testing.assert_allclose(values, expect, rtol=0, atol=1.0e-15)
            clipped = lsf.computeWindow(center)
            self.assertGreaterEqual(clipped[0], 0)
            self.assertLessEqual(clipped[0] + len(clipped[1]), length)

        for order in (1, 3):
            indexMap = IndexMap.fromWavelength(inWavelength, self.wavelength)
            weight = SmoothWeight.constant(indexMap.minIndex, indexMap.maxIndex)
            combined = CombinedLsf(len(self.wavelength), [CombinedLsfInput(lsf, indexMap, weight, order)])
            resampler = Resampler(extended, self.wavelength, order=order)
            middle = combined.computeShape1D(0.5*(indexMap.minIndex + indexMap.maxIndex))
            for center in (indexMap.minIndex + 0.3, indexMap.maxIndex - 0.3):
                # The arm extended, resampled: the LSF we want
                wl = np.interp(center, np.arange(len(self.wavelength)), self.wavelength)
                inCenter = np.interp(wl, extended, np.arange(len(extended)))
                inStart, inValues = extendedLsf.computeWindow(inCenter)
                outStart, expect = resampler.resampleWindow(inValues, inStart)
                expect /= np.sum(expect)
                start, values = combined.computeWindow(center, clip=False)
                array = np.zeros(len(self.wavelength))
                array[start:start + len(values)] = values
                select = slice(outStart, outStart + len(expect))
                np.testing.assert_allclose(array[select], expect, rtol=0, atol=1.0e-9)
                self.assertFloatsAlmostEqual(np.sum(array), 1.0, atol=1.0e-12)
                self.assertFloatsAlmostEqual(np.sum(combined.computeArray(center)), 1.0, atol=1.0e-12)
                # Same width as in the middle of the arm (the dispersion is constant)
                self.assertFloatsAlmostEqual(combined.computeShape1D(center), middle, rtol=0.03)

    def testIndexMap(self):
        """The Chebyshev approximation to the resampling indices is accurate"""
        pixels = np.arange(4096, dtype=float)
        inWavelength = 630.0 + 0.083*pixels + 3.0e-7*pixels**2 + 0.01*np.sin(pixels/700.0)
        indices = interpolateIndices(inWavelength, self.wavelength)
        for tolerance in (1.0e-3, 1.0e-5):
            indexMap = IndexMap.fromWavelength(inWavelength, self.wavelength, tolerance=tolerance)
            covered = np.isfinite(indices)
            self.assertEqual(indexMap.minIndex, np.flatnonzero(covered)[0])
            self.assertEqual(indexMap.maxIndex, np.flatnonzero(covered)[-1])
            self.assertLess(indexMap.maxError, tolerance)
            out = np.arange(len(self.wavelength), dtype=float)
            values = indexMap(out)
            np.testing.assert_array_equal(np.isfinite(values), covered)
            self.assertLess(np.max(np.abs(values[covered] - indices[covered])), tolerance)
            self.assertTrue(np.isnan(indexMap(indexMap.minIndex - 1.0)))
        self.assertIsNone(IndexMap.fromWavelength(inWavelength, np.linspace(100, 200, 10)))

    def testSmoothWeight(self):
        """Smooth weights ignore lines and unselected pixels"""
        rng = np.random.default_rng(54321)
        weight = np.full(3000, 2.0) + rng.normal(0.0, 0.01, 3000)
        select = np.ones(3000, dtype=bool)
        weight[rng.choice(3000, 100, replace=False)] *= 0.1  # Lines lower the weight
        select[1000:1300] = False  # A gap
        weight[1000:1300] = 0.0
        smooth = SmoothWeight.fit(weight, select, 100, 2899, binSize=50)
        pixels = np.arange(3000, dtype=float)
        values = smooth(pixels)
        np.testing.assert_allclose(values[100:2900], 2.0, atol=0.01)
        self.assertTrue(np.all(values[:100] == 0) and np.all(values[2900:] == 0))
        self.assertEqual(len(smooth.values), 56)
        constant = SmoothWeight.constant(10, 20, 3.0)
        np.testing.assert_array_equal(constant(np.array([9.0, 10.0, 15.5, 20.0, 21.0])), [0, 3, 3, 3, 0])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
