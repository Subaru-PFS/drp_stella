import numpy as np

import lsst.utils.tests

from pfs.drp.stella.synthetic import SyntheticPsfConfig, makeSyntheticPsfArc
from pfs.drp.stella.psfProfiles import ParametricPsfModel, WingParams
from pfs.drp.stella.psfSpline import PsfSplineBasis, RegularizationConfig, SplineAxisConfig
from pfs.drp.stella.hybridPsfModel import HybridPsfModel
from pfs.drp.stella.fitPsfModel import fitHybridPsf
from pfs.drp.stella.psfValidation import (
    compareModelVariants,
    computeHeldOutChi2,
    crossValidateByFiber,
    dpCoherenceCheck,
    energyConservationCheck,
    radialProfileWithBootstrap,
    stackedResidualMap,
    subPixelPhaseHistogram,
)
from pfs.drp.stella.tests import runTests


def _makeSplineBasis() -> PsfSplineBasis:
    axisConfig = SplineAxisConfig(extent=6.0, fineRadius=1.0, mediumRadius=2.0, coarseSpacing=2.0)
    return PsfSplineBasis(axisConfig, axisConfig)


class ComputeHeldOutChi2TestCase(lsst.utils.tests.TestCase):
    """The core mechanism behind the cross-validation/model-comparison
    diagnostics: solve amplitudes for a fixed model and report chi2"""

    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 90
        self.config.width = 45
        self.config.psfHalfSize = 10
        self.config.oversampling = 2
        self.config.bumpFraction = 0.0
        self.config.wingFraction = 0.0
        self.truth = self.config.makeGroundTruthPsf()
        self.result = makeSyntheticPsfArc(
            self.config, numLines=3, addNoise=False, rng=np.random.RandomState(42)
        )
        self.fiberIndex = self.result.fiberIndex.astype(float)

    def _makeModel(self, sigmaScale: float = 1.0) -> ParametricPsfModel:
        model = ParametricPsfModel(0, 0, (0, self.config.numFibers - 1), (0, self.config.height - 1))
        model.setInitialGuess(
            sigmaX=self.truth.sigmaX * sigmaScale,
            sigmaY=self.truth.sigmaY * sigmaScale,
            tophatWidth=self.truth.tophatWidth,
        )
        return model

    def _heldOutMaskAllLines(self) -> np.ndarray:
        # No training lines at all: every line's amplitude is solved from
        # scratch, exercising computeHeldOutChi2's mechanism without also
        # needing a separate (fixed) training fit for this simple check.
        return np.ones(len(self.fiberIndex), dtype=bool)

    def testGoodModelGivesLowChi2(self):
        """The true (noiseless) model should reproduce the data almost exactly"""
        result = computeHeldOutChi2(
            self.result.image,
            self.fiberIndex,
            self.result.row,
            self.result.xCenter,
            self.result.row,
            self._heldOutMaskAllLines(),
            self._makeModel(),
            np.array([]),
            self.config.psfHalfSize,
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
        )
        self.assertLess(result.reducedChi2, 0.01)

    def testWrongModelGivesHighChi2(self):
        """A grossly wrong PSF shape should fit the held-out data much worse"""
        good = computeHeldOutChi2(
            self.result.image,
            self.fiberIndex,
            self.result.row,
            self.result.xCenter,
            self.result.row,
            self._heldOutMaskAllLines(),
            self._makeModel(),
            np.array([]),
            self.config.psfHalfSize,
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
        )
        bad = computeHeldOutChi2(
            self.result.image,
            self.fiberIndex,
            self.result.row,
            self.result.xCenter,
            self.result.row,
            self._heldOutMaskAllLines(),
            self._makeModel(sigmaScale=3.0),
            np.array([]),
            self.config.psfHalfSize,
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
        )
        self.assertGreater(bad.chi2, 100 * max(good.chi2, 1.0e-12))


class CrossValidateByFiberTestCase(lsst.utils.tests.TestCase):
    """Synthetic test 3 of PIPE2D-1823-psf.md: cross-validation by
    withholding edge/gap fibers

    ``psfHalfSize`` is kept modest (well under the synthetic harness'
    buffer margin, so edge fibers still render lines) but still close
    enough to the 6.5 px fiber pitch that held-out edge fibers' stamps
    reach into their training neighbors -- exactly the regime
    `computeHeldOutChi2` has to handle correctly (see its docstring).
    """

    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 150
        self.config.width = 90
        self.config.psfHalfSize = 6
        self.config.oversampling = 2
        self.config.bumpFraction = 0.0
        self.config.wingFraction = 0.0
        self.truth = self.config.makeGroundTruthPsf()
        self.result = makeSyntheticPsfArc(
            self.config, numLines=6, addNoise=True, rng=np.random.RandomState(7)
        )
        self.fiberIndex = self.result.fiberIndex.astype(float)
        self.edgeIndices = set(int(ii) for ii in self.config.blockEdgeIndices)

    def _makeHybridModel(self) -> HybridPsfModel:
        axisConfig = SplineAxisConfig(extent=4.0, fineRadius=1.0, mediumRadius=2.0, coarseSpacing=1.0)
        model = HybridPsfModel(
            ParametricPsfModel(0, 0, (0, self.config.numFibers - 1), (0, self.config.height - 1)),
            PsfSplineBasis(axisConfig, axisConfig),
        )
        model.parametricModel.setInitialGuess(
            sigmaX=self.truth.sigmaX, sigmaY=self.truth.sigmaY, tophatWidth=self.truth.tophatWidth
        )
        return model

    def _fit(self, heldOutMask):
        return crossValidateByFiber(
            self.result.image,
            self.fiberIndex,
            self.result.row,
            self.result.xCenter,
            self.result.row,
            heldOutMask,
            self._makeHybridModel,
            self.config.psfHalfSize,
            regularizationConfig=RegularizationConfig(smoothness=1.0, ridge=10.0),
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
            maxOuterIter=6,
            fitCenters=True,
        )

    def testEdgeHeldOutPredictsWell(self):
        """Training on the (many) interior fibers should predict the
        held-out edge fibers' wings reasonably well"""
        heldOutMask = np.array([ii in self.edgeIndices for ii in self.result.fiberIndex])
        cv = self._fit(heldOutMask)
        self.assertGreater(cv.numHeldOut, 0)
        self.assertGreater(cv.numTrain, cv.numHeldOut)
        self.assertLess(cv.heldOut.reducedChi2, 200.0)

    def testTrainingOnFewFibersDegradesPrediction(self):
        """Training on only the two edge fibers (few lines) and predicting
        the interior should do noticeably worse than the reverse, which
        trains on many more, better-sampled interior fibers"""
        heldOutEdge = np.array([ii in self.edgeIndices for ii in self.result.fiberIndex])
        trainOnInteriorCv = self._fit(heldOutEdge)

        heldOutInterior = np.array([ii not in self.edgeIndices for ii in self.result.fiberIndex])
        trainOnEdgeCv = self._fit(heldOutInterior)

        self.assertGreater(trainOnEdgeCv.heldOut.reducedChi2, 5 * trainOnInteriorCv.heldOut.reducedChi2)


class CompareModelVariantsTestCase(lsst.utils.tests.TestCase):
    """The hybrid model should out-predict the P_param-only model on
    held-out data when the true PSF has structure P_param can't capture"""

    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 150
        self.config.width = 90
        self.config.psfHalfSize = 6
        self.config.oversampling = 2
        self.config.wingFraction = 0.0
        self.config.bumpFraction = 0.15
        self.config.bumpOffsetX = 2.0
        self.config.bumpOffsetY = 0.0
        self.config.bumpWidth = 1.0
        self.truth = self.config.makeGroundTruthPsf()
        self.result = makeSyntheticPsfArc(
            self.config, numLines=6, addNoise=True, rng=np.random.RandomState(99)
        )
        self.fiberIndex = self.result.fiberIndex.astype(float)

    def _makeHybridModel(self) -> HybridPsfModel:
        axisConfig = SplineAxisConfig(extent=4.0, fineRadius=1.0, mediumRadius=2.0, coarseSpacing=1.0)
        model = HybridPsfModel(
            ParametricPsfModel(0, 0, (0, self.config.numFibers - 1), (0, self.config.height - 1)),
            PsfSplineBasis(axisConfig, axisConfig),
        )
        model.parametricModel.setInitialGuess(
            sigmaX=self.truth.sigmaX, sigmaY=self.truth.sigmaY, tophatWidth=self.truth.tophatWidth
        )
        return model

    def testHybridBeatsParamOnly(self):
        rng = np.random.RandomState(13)
        heldOutMask = rng.uniform(size=len(self.fiberIndex)) < 0.3
        comparison = compareModelVariants(
            self.result.image,
            self.fiberIndex,
            self.result.row,
            self.result.xCenter,
            self.result.row,
            heldOutMask,
            self._makeHybridModel,
            self.config.psfHalfSize,
            regularizationConfig=RegularizationConfig(smoothness=1.0, ridge=10.0),
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
            maxOuterIter=6,
            fitCenters=True,
        )
        self.assertLess(comparison.hybrid.heldOut.chi2, comparison.paramOnly.heldOut.chi2)


class SubPixelPhaseHistogramTestCase(lsst.utils.tests.TestCase):
    def testUniformPhasesNotFlagged(self):
        rng = np.random.RandomState(0)
        xCenter = rng.uniform(0, 100, size=2000)
        yCenter = rng.uniform(0, 100, size=2000)
        result = subPixelPhaseHistogram(xCenter, yCenter, show=False)
        self.assertFalse(result.xClustered)
        self.assertFalse(result.yClustered)

    def testClusteredPhasesFlagged(self):
        rng = np.random.RandomState(0)
        xCenter = rng.uniform(0, 100, size=2000).astype(int) + rng.normal(0, 0.02, size=2000)
        yCenter = rng.uniform(0, 100, size=2000)
        result = subPixelPhaseHistogram(xCenter, yCenter, show=False)
        self.assertTrue(result.xClustered)


class StackedResidualMapTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 90
        self.config.width = 45
        self.config.psfHalfSize = 10
        self.config.oversampling = 2
        self.config.bumpFraction = 0.0
        self.config.wingFraction = 0.0
        self.truth = self.config.makeGroundTruthPsf()
        self.result = makeSyntheticPsfArc(
            self.config, numLines=3, addNoise=False, rng=np.random.RandomState(21)
        )
        self.fiberIndex = self.result.fiberIndex.astype(float)

    def testGoodModelResidualsAreSmall(self):
        model = ParametricPsfModel(0, 0, (0, self.config.numFibers - 1), (0, self.config.height - 1))
        model.setInitialGuess(
            sigmaX=self.truth.sigmaX, sigmaY=self.truth.sigmaY, tophatWidth=self.truth.tophatWidth
        )
        result = stackedResidualMap(
            self.result.image,
            self.fiberIndex,
            self.result.row,
            self.result.xCenter,
            self.result.row,
            model,
            self.config.psfHalfSize,
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
            show=False,
        )
        valid = np.isfinite(result.stackedResidual)
        self.assertTrue(np.any(valid))
        self.assertFloatsAlmostEqual(result.stackedResidual[valid], 0.0, atol=0.1)


class RadialProfileWithBootstrapTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 90
        self.config.width = 45
        self.config.psfHalfSize = 8
        self.config.oversampling = 2
        self.config.bumpFraction = 0.0
        self.config.wingFraction = 0.0
        self.truth = self.config.makeGroundTruthPsf()
        self.result = makeSyntheticPsfArc(
            self.config, numLines=2, addNoise=True, rng=np.random.RandomState(3)
        )
        self.fiberIndex = self.result.fiberIndex.astype(float)

    def _makeHybridModel(self) -> HybridPsfModel:
        model = HybridPsfModel(
            ParametricPsfModel(0, 0, (0, self.config.numFibers - 1), (0, self.config.height - 1)),
            _makeSplineBasis(),
        )
        model.parametricModel.setInitialGuess(
            sigmaX=self.truth.sigmaX, sigmaY=self.truth.sigmaY, tophatWidth=self.truth.tophatWidth
        )
        return model

    def testReturnsSensibleProfiles(self):
        result = radialProfileWithBootstrap(
            self.result.image,
            self.fiberIndex,
            self.result.row,
            self.result.xCenter,
            self.result.row,
            self._makeHybridModel,
            self.config.psfHalfSize,
            regularizationConfig=RegularizationConfig(smoothness=1.0, ridge=10.0),
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
            maxOuterIter=3,
            fitCenters=False,
            numBootstrap=3,
            numRadialPoints=6,
            numAngles=8,
            rng=np.random.RandomState(5),
            show=False,
        )
        self.assertEqual(len(result.radii), 6)
        for name in ("radialTotal", "xTotal", "yTotal", "radialLower", "radialUpper"):
            values = getattr(result, name)
            self.assertEqual(len(values), 6)
            self.assertTrue(np.all(np.isfinite(values)))
        # The profile should decrease away from the center (r=0).
        self.assertGreater(result.radialTotal[0], result.radialTotal[-1])


class EnergyConservationCheckTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        parametricModel = ParametricPsfModel(0, 1, (0, 1), (0, 1))
        parametricModel.setInitialGuess(
            sigmaX=1.0,
            sigmaY=1.0,
            tophatWidth=1.0,
            wings=[WingParams(scaleX=4.0, scaleY=4.0, beta=2.5, fraction=0.05)],
        )
        self.model = HybridPsfModel(parametricModel, _makeSplineBasis())

    def testMostFluxWithinGenerousExtent(self):
        result = energyConservationCheck(self.model, 0.0, 0.0, halfExtent=20)
        self.assertFloatsAlmostEqual(result.totalFlux, 1.0, atol=0.02)
        self.assertLess(abs(result.fractionOutside), 0.02)

    def testTruncationShowsUpWithSmallExtent(self):
        result = energyConservationCheck(self.model, 0.0, 0.0, halfExtent=2)
        self.assertGreater(result.fractionOutside, 0.03)


class DpCoherenceCheckTestCase(lsst.utils.tests.TestCase):
    """``psfHalfSize`` is kept small relative to the fiber pitch here: each
    even/odd half's stamps must not reach into the *other* half's (in this
    test, excluded-from-the-half-being-solved) fibers, or their real flux
    would be misattributed the same way `computeHeldOutChi2` had to guard
    against (see its docstring) -- but `dpCoherenceCheck` itself does not
    (its per-half solves are independent, smaller-scale analyses by
    design), so the mitigation here is simply to keep the stamps tight.
    """

    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 150
        self.config.width = 90
        self.config.psfHalfSize = 5
        self.config.oversampling = 2
        self.config.wingFraction = 0.0
        self.config.bumpFraction = 0.2
        self.config.bumpOffsetX = 2.0
        self.config.bumpOffsetY = 0.0
        self.config.bumpWidth = 1.0
        self.truth = self.config.makeGroundTruthPsf()
        self.result = makeSyntheticPsfArc(
            self.config, numLines=8, addNoise=True, rng=np.random.RandomState(11)
        )
        self.fiberIndex = self.result.fiberIndex.astype(float)

    def testCoherentBumpIsCorrelatedAcrossHalves(self):
        axisConfig = SplineAxisConfig(extent=3.0, fineRadius=1.0, mediumRadius=2.0, coarseSpacing=1.0)
        model = HybridPsfModel(
            ParametricPsfModel(0, 0, (0, self.config.numFibers - 1), (0, self.config.height - 1)),
            PsfSplineBasis(axisConfig, axisConfig),
        )
        model.parametricModel.setInitialGuess(
            sigmaX=self.truth.sigmaX, sigmaY=self.truth.sigmaY, tophatWidth=self.truth.tophatWidth
        )
        regularizationConfig = RegularizationConfig(smoothness=0.1, ridge=0.1)
        fullFit = fitHybridPsf(
            self.result.image,
            self.fiberIndex,
            self.result.row,
            self.result.xCenter,
            self.result.row,
            model,
            self.config.psfHalfSize,
            regularizationConfig=regularizationConfig,
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
            maxOuterIter=6,
            fitCenters=True,
        )
        result = dpCoherenceCheck(
            self.result.image,
            self.fiberIndex,
            self.result.row,
            fullFit,
            self.config.psfHalfSize,
            regularizationConfig=regularizationConfig,
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
        )
        self.assertGreater(result.correlation, 0.3)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
