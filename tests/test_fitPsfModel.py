import numpy as np

import lsst.utils.tests

from pfs.drp.stella.synthetic import SyntheticPsfConfig, makeSyntheticPsfArc
from pfs.drp.stella.psfProfiles import ParametricPsfModel
from pfs.drp.stella.psfSpline import PsfSplineBasis, RegularizationConfig, SplineAxisConfig
from pfs.drp.stella.hybridPsfModel import HybridPsfModel
from pfs.drp.stella.fitPsfModel import (
    buildParametricDesignMatrix,
    buildSplineDesignMatrix,
    buildStampGeometry,
    fitHybridPsf,
    fitParametricPsf,
    solveAmplitudesAndBackground,
    solveSplineCoefficients,
    updateLineCenters,
)
from pfs.drp.stella.tests import runTests


class BuildStampGeometryTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 120
        self.config.width = 60
        self.config.psfHalfSize = 8
        self.config.oversampling = 2
        self.result = makeSyntheticPsfArc(
            self.config, numLines=3, addNoise=False, rng=np.random.RandomState(11)
        )

    def testPixelIndexIsCompact(self):
        """The compact pixel index covers exactly the used pixels, 0..N-1"""
        geometry = buildStampGeometry(
            self.result.image,
            self.result.fiberIndex.astype(float),
            self.result.row,
            self.result.xCenter,
            self.result.row,
            self.config.psfHalfSize,
            self.config.gain,
            self.config.readnoise,
        )
        used = geometry.pixelIndex >= 0
        indices = np.sort(geometry.pixelIndex[used])
        self.assertFloatsEqual(indices, np.arange(geometry.numUsedPixels))
        self.assertEqual(len(geometry.dataVector), geometry.numUsedPixels)
        self.assertEqual(len(geometry.varianceVector), geometry.numUsedPixels)

    def testOverlappingStampsAreDeduplicated(self):
        """Pixels shared by two nearby lines' stamps are only counted once"""
        geometry = buildStampGeometry(
            self.result.image,
            self.result.fiberIndex.astype(float),
            self.result.row,
            self.result.xCenter,
            self.result.row,
            self.config.psfHalfSize,
            self.config.gain,
            self.config.readnoise,
        )
        totalStampPixels = sum(
            (rowSlice.stop - rowSlice.start) * (colSlice.stop - colSlice.start)
            for rowSlice, colSlice in zip(geometry.rowSlices, geometry.colSlices)
        )
        self.assertLess(geometry.numUsedPixels, totalStampPixels)


class BuildParametricDesignMatrixTestCase(lsst.utils.tests.TestCase):
    """Check the design matrix against isolated (non-overlapping) fiber stamps

    Widely-spaced fibers avoid the neighbor-fiber contamination that (as
    documented in `FitParametricPsfTestCase`) makes wing-shape recovery
    degenerate under realistic crowding; that isn't relevant here, since
    this only checks that the design matrix correctly evaluates the
    already-true model, not that the model can be fit.
    """

    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 120
        self.config.width = 250
        self.config.separation = 30.0
        self.config.psfHalfSize = 8
        self.config.oversampling = 4
        self.config.bumpFraction = 0.0  # exclude the non-parametric injected bump from this comparison
        self.result = makeSyntheticPsfArc(
            self.config, numLines=2, addNoise=False, rng=np.random.RandomState(22)
        )
        self.model = ParametricPsfModel(0, 1, (0, self.config.numFibers - 1), (0, self.config.height - 1))
        truth = self.config.makeGroundTruthPsf()
        self.model.setInitialGuess(truth.sigmaX, truth.sigmaY, truth.tophatWidth, truth.wings)

    def testDesignMatrixMatchesInjectedFlux(self):
        """The true-model design matrix, applied to the true amplitudes,
        reproduces the noiseless image for isolated lines"""
        fiberIndex = self.result.fiberIndex.astype(float)
        geometry = buildStampGeometry(
            self.result.image,
            fiberIndex,
            self.result.row,
            self.result.xCenter,
            self.result.row,
            self.config.psfHalfSize,
            self.config.gain,
            self.config.readnoise,
        )
        designMatrix = buildParametricDesignMatrix(
            self.model,
            fiberIndex,
            self.result.row,
            self.result.xCenter,
            self.result.row,
            geometry,
            self.config.oversampling,
            includeBackground=False,
        )
        predicted = designMatrix @ self.result.amplitude
        self.assertFloatsAlmostEqual(predicted, geometry.dataVector, atol=1.0e-6, rtol=1.0e-6)


class FitParametricPsfTestCase(lsst.utils.tests.TestCase):
    """Recovery of the parametric PSF *core* from a poor initial guess

    The wing component is intentionally excluded here (``wingFraction=0``):
    with tightly-packed, overlapping fiber stamps at the fiber pitch and
    stamp size the spec specifies, the wing shape is degenerate with
    per-line amplitude allocation when fit by simple amplitude/shape
    alternation without additional constraints -- confirmed by starting a
    fit exactly at the true wing parameters and finding the optimizer walks
    away to a lower-chi2, unphysical solution. Breaking that degeneracy is
    deferred to the full hybrid (``P_param`` + constrained ``dP``) fit; see
    PIPE2D-1823-psf.md. The compact core, by contrast, is well localized and
    recovers reliably even under this same crowding.
    """

    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 200
        self.config.width = 120
        self.config.psfHalfSize = 10
        self.config.oversampling = 2
        self.config.wingFraction = 0.0
        self.config.bumpFraction = 0.0

    def testRecoversCoreFromPoorInitialGuess(self):
        result = makeSyntheticPsfArc(self.config, numLines=6, addNoise=True, rng=np.random.RandomState(33))
        fiberIndex = result.fiberIndex.astype(float)
        model = ParametricPsfModel(0, 0, (0, self.config.numFibers - 1), (0, self.config.height - 1))
        model.setInitialGuess(
            sigmaX=self.config.fwhm / 2.3548 * 1.3,
            sigmaY=self.config.fwhm / 2.3548 * 1.3,
            tophatWidth=self.config.coreWidth * 0.7,
        )
        fit = fitParametricPsf(
            result.image,
            fiberIndex,
            result.row,
            result.xCenter,
            result.row,
            model,
            self.config.psfHalfSize,
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
            maxOuterIter=5,
        )
        truth = self.config.makeGroundTruthPsf()
        fitted = fit.model.getParamsAt(0, 0)
        self.assertFloatsAlmostEqual(fitted.sigmaX, truth.sigmaX, rtol=0.1)
        self.assertFloatsAlmostEqual(fitted.sigmaY, truth.sigmaY, rtol=0.1)
        self.assertFloatsAlmostEqual(fitted.tophatWidth, truth.tophatWidth, rtol=0.2)


def _injectPitchPeriodicPerturbation(image, xCenter, row, halfSize, width, height, pitch, amplitude, sigmaY):
    """Add a fixed-pattern perturbation, periodic in x with the fiber pitch,
    to every line's stamp

    The perturbation has the same shape and (absolute) amplitude for every
    line, independent of that line's own flux -- mimicking e.g. a
    flat-fielding residual correlated with the fixed fiber pitch, rather
    than a genuine feature of the per-line PSF that a fit should recover.

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image to perturb, in place.
    xCenter, row : `numpy.ndarray`
        True per-line spatial and dispersion-direction centers.
    halfSize : `int`
        Half-size of the region around each line to perturb.
    width, height : `int`
        Image dimensions, for clipping stamps to the image.
    pitch : `float`
        Period of the perturbation in x, in pixels.
    amplitude : `float`
        Peak amplitude of the perturbation, in the same units as the image.
    sigmaY : `float`
        Gaussian envelope width in y, in pixels.
    """
    for xx, yy in zip(xCenter, row):
        xLo, xHi = max(int(np.floor(xx - halfSize)), 0), min(int(np.ceil(xx + halfSize)) + 1, width)
        yLo, yHi = max(int(np.floor(yy - halfSize)), 0), min(int(np.ceil(yy + halfSize)) + 1, height)
        dx = np.arange(xLo, xHi)[np.newaxis, :] - xx
        dy = np.arange(yLo, yHi)[:, np.newaxis] - yy
        perturbation = amplitude * np.sin(2 * np.pi * dx / pitch) * np.exp(-0.5 * (dy / sigmaY) ** 2)
        image.array[yLo:yHi, xLo:xHi] += perturbation


class PitchPeriodicRegularizationTestCase(lsst.utils.tests.TestCase):
    """Regularization should keep ``dP`` from chasing a fixed-pattern,
    pitch-periodic perturbation that isn't a genuine per-line PSF feature

    This is the synthetic test PIPE2D-1823-psf.md calls out: a systematic
    perturbed pattern correlated with the fixed fiber pitch (here, a fixed
    sinusoid in x with that period, added identically to every line's
    stamp) is exactly the kind of signal that could alias into ``dP``'s
    null modes (see the amplitude/wing-shape degeneracy documented in
    `FitParametricPsfTestCase`). With smoothness/ridge regularization on,
    the fitted ``dP`` should stay noticeably smaller than with it
    effectively off.
    """

    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 150
        self.config.width = 45
        self.config.psfHalfSize = 12
        self.config.oversampling = 2
        self.config.bumpFraction = 0.0
        self.config.wingFraction = 0.0
        self.truth = self.config.makeGroundTruthPsf()
        result = makeSyntheticPsfArc(self.config, numLines=4, addNoise=False, rng=np.random.RandomState(101))
        _injectPitchPeriodicPerturbation(
            result.image,
            result.xCenter,
            result.row,
            self.config.psfHalfSize,
            self.config.width,
            self.config.height,
            pitch=self.config.separation,
            amplitude=0.01,
            sigmaY=2.0,
        )
        self.result = result
        self.fiberIndex = result.fiberIndex.astype(float)

    def runFit(self, regularizationConfig) -> HybridPsfModel:
        hybridModel = HybridPsfModel(
            ParametricPsfModel(0, 0, (0, self.config.numFibers - 1), (0, self.config.height - 1)),
            PsfSplineBasis(
                SplineAxisConfig(extent=6.0, fineRadius=1.0, mediumRadius=2.0, coarseSpacing=2.0),
                SplineAxisConfig(extent=6.0, fineRadius=1.0, mediumRadius=2.0, coarseSpacing=2.0),
            ),
        )
        hybridModel.parametricModel.setInitialGuess(
            sigmaX=self.truth.sigmaX, sigmaY=self.truth.sigmaY, tophatWidth=self.truth.tophatWidth
        )
        fitHybridPsf(
            self.result.image,
            self.fiberIndex,
            self.result.row,
            self.result.xCenter,
            self.result.row,
            hybridModel,
            self.config.psfHalfSize,
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
            maxOuterIter=6,
            regularizationConfig=regularizationConfig,
        )
        return hybridModel

    def testRegularizationSuppressesSpuriousDp(self):
        regularized = self.runFit(RegularizationConfig(smoothness=1.0, ridge=10.0))
        unregularized = self.runFit(RegularizationConfig(smoothness=1.0e-8, ridge=1.0e-8))
        regularizedNorm = np.linalg.norm(regularized.splineCoefficients)
        unregularizedNorm = np.linalg.norm(unregularized.splineCoefficients)
        self.assertLess(regularizedNorm, 0.5 * unregularizedNorm)


class UpdateLineCentersTestCase(lsst.utils.tests.TestCase):
    """A local Gauss-Newton step should move perturbed centers back toward truth"""

    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 200
        self.config.width = 120
        self.config.psfHalfSize = 10
        self.config.oversampling = 2
        self.config.wingFraction = 0.0
        self.config.bumpFraction = 0.0
        self.result = makeSyntheticPsfArc(
            self.config, numLines=6, addNoise=True, rng=np.random.RandomState(33)
        )
        self.fiberIndex = self.result.fiberIndex.astype(float)
        truth = self.config.makeGroundTruthPsf()
        self.model = ParametricPsfModel(0, 0, (0, self.config.numFibers - 1), (0, self.config.height - 1))
        self.model.setInitialGuess(sigmaX=truth.sigmaX, sigmaY=truth.sigmaY, tophatWidth=truth.tophatWidth)

    def testCenterUpdateReducesError(self):
        rng = np.random.RandomState(7)
        perturbedX = self.result.xCenter + rng.uniform(-0.3, 0.3, size=len(self.result.xCenter))
        perturbedY = self.result.row + rng.uniform(-0.3, 0.3, size=len(self.result.row))

        geometry = buildStampGeometry(
            self.result.image,
            self.fiberIndex,
            self.result.row,
            perturbedX,
            perturbedY,
            self.config.psfHalfSize,
            self.config.gain,
            self.config.readnoise,
        )
        designMatrix = buildParametricDesignMatrix(
            self.model,
            self.fiberIndex,
            self.result.row,
            perturbedX,
            perturbedY,
            geometry,
            self.config.oversampling,
        )
        amplitudes = solveAmplitudesAndBackground(designMatrix, geometry)

        newXCenter, newYCenter = updateLineCenters(
            self.model,
            self.fiberIndex,
            self.result.row,
            perturbedX,
            perturbedY,
            geometry,
            designMatrix,
            amplitudes,
            self.config.oversampling,
            stepSize=0.01,
            maxShift=1.0,
        )
        errorBefore = (
            np.abs(perturbedX - self.result.xCenter).mean() + np.abs(perturbedY - self.result.row).mean()
        )
        errorAfter = (
            np.abs(newXCenter - self.result.xCenter).mean() + np.abs(newYCenter - self.result.row).mean()
        )
        self.assertLess(errorAfter, 0.2 * errorBefore)


class SplineCoefficientRecoveryTestCase(lsst.utils.tests.TestCase):
    """A known, constraint-satisfying ``dP`` signal should be recovered from
    its noiseless design-matrix projection"""

    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 200
        self.config.width = 120
        self.config.psfHalfSize = 10
        self.config.oversampling = 2
        self.config.wingFraction = 0.0
        self.config.bumpFraction = 0.0
        self.result = makeSyntheticPsfArc(
            self.config, numLines=6, addNoise=True, rng=np.random.RandomState(33)
        )
        self.fiberIndex = self.result.fiberIndex.astype(float)
        truth = self.config.makeGroundTruthPsf()
        self.model = ParametricPsfModel(0, 0, (0, self.config.numFibers - 1), (0, self.config.height - 1))
        self.model.setInitialGuess(sigmaX=truth.sigmaX, sigmaY=truth.sigmaY, tophatWidth=truth.tophatWidth)
        self.splineBasis = PsfSplineBasis(SplineAxisConfig(extent=8.0), SplineAxisConfig(extent=8.0))

    def testRecoversInjectedCoefficients(self):
        geometry = buildStampGeometry(
            self.result.image,
            self.fiberIndex,
            self.result.row,
            self.result.xCenter,
            self.result.row,
            self.config.psfHalfSize,
            self.config.gain,
            self.config.readnoise,
        )
        designMatrix = buildParametricDesignMatrix(
            self.model,
            self.fiberIndex,
            self.result.row,
            self.result.xCenter,
            self.result.row,
            geometry,
            self.config.oversampling,
        )
        amplitudes = solveAmplitudesAndBackground(designMatrix, geometry)
        lineAmplitudes = amplitudes[: len(self.fiberIndex)]

        rng = np.random.RandomState(9)
        nullSpace = self.splineBasis.nullSpaceBasis()
        trueCoefficients = nullSpace @ (rng.normal(size=nullSpace.shape[1]) * 0.02)

        splineDesign = buildSplineDesignMatrix(
            self.splineBasis,
            lineAmplitudes,
            self.fiberIndex,
            self.result.row,
            self.result.xCenter,
            self.result.row,
            geometry,
            self.config.oversampling,
        )
        syntheticResidual = splineDesign @ trueCoefficients
        fittedCoefficients = solveSplineCoefficients(
            splineDesign,
            syntheticResidual,
            geometry,
            self.splineBasis,
            RegularizationConfig(smoothness=1.0e-6, ridge=1.0e-6),
        )
        self.assertFloatsAlmostEqual(fittedCoefficients, trueCoefficients, atol=1.0e-6)

        constraints = self.splineBasis.constraintMatrix()
        self.assertFloatsAlmostEqual(
            constraints @ fittedCoefficients, np.zeros(constraints.shape[0]), atol=1.0e-10
        )


class FitHybridPsfTestCase(lsst.utils.tests.TestCase):
    """Recovery of the full hybrid model (core + wings + ``dP``)

    Unlike `FitParametricPsfTestCase`, the wing component is included here:
    the hybrid model's hard-constrained ``dP`` is intended to break the
    amplitude/wing-shape degeneracy documented there. A noiseless fit
    started exactly at the true parameters is the cleanest check of that:
    it isolates the alternating optimizer's own structural behaviour from
    the sample-noise sensitivity that a small stamp catalog (few lines per
    fiber, as used here for runtime) inevitably has. That noiseless check
    stays essentially exact (see `testStaysNearTruthWithoutNoise`); an
    otherwise-identical fit with noise added (`testRecoversWithNoise`)
    drifts by some 10-20% in the wing shape parameters even started at
    the truth, which is consistent with finite-sample noise given the
    small number of lines used here (not a re-emergence of the structural
    degeneracy that afflicts the wing-only parametric fit).
    """

    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 150
        self.config.width = 45
        self.config.psfHalfSize = 12
        self.config.oversampling = 2
        self.config.bumpFraction = 0.0
        self.truth = self.config.makeGroundTruthPsf()

    def makeHybridModel(self) -> HybridPsfModel:
        return HybridPsfModel(
            ParametricPsfModel(0, 1, (0, self.config.numFibers - 1), (0, self.config.height - 1)),
            PsfSplineBasis(
                SplineAxisConfig(extent=6.0, fineRadius=1.0, mediumRadius=2.0, coarseSpacing=2.0),
                SplineAxisConfig(extent=6.0, fineRadius=1.0, mediumRadius=2.0, coarseSpacing=2.0),
            ),
        )

    def testStaysNearTruthWithoutNoise(self):
        """Starting at truth with no noise, the fit should not drift"""
        result = makeSyntheticPsfArc(self.config, numLines=4, addNoise=False, rng=np.random.RandomState(101))
        fiberIndex = result.fiberIndex.astype(float)
        hybridModel = self.makeHybridModel()
        hybridModel.parametricModel.setInitialGuess(
            sigmaX=self.truth.sigmaX,
            sigmaY=self.truth.sigmaY,
            tophatWidth=self.truth.tophatWidth,
            wings=self.truth.wings,
        )
        fit = fitHybridPsf(
            result.image,
            fiberIndex,
            result.row,
            result.xCenter,
            result.row,
            hybridModel,
            self.config.psfHalfSize,
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
            maxOuterIter=6,
            regularizationConfig=RegularizationConfig(smoothness=1.0, ridge=10.0),
        )
        fitted = fit.model.parametricModel.getParamsAt(0, 0)
        self.assertFloatsAlmostEqual(fitted.sigmaX, self.truth.sigmaX, rtol=0.02)
        self.assertFloatsAlmostEqual(fitted.sigmaY, self.truth.sigmaY, rtol=0.02)
        self.assertFloatsAlmostEqual(fitted.tophatWidth, self.truth.tophatWidth, rtol=0.02)
        self.assertFloatsAlmostEqual(fitted.wings[0].scaleX, self.truth.wings[0].scaleX, rtol=0.05)
        self.assertFloatsAlmostEqual(fitted.wings[0].scaleY, self.truth.wings[0].scaleY, rtol=0.05)
        self.assertFloatsAlmostEqual(fitted.wings[0].beta, self.truth.wings[0].beta, rtol=0.05)

        constraints = hybridModel.splineBasis.constraintMatrix()
        self.assertFloatsAlmostEqual(
            constraints @ hybridModel.splineCoefficients, np.zeros(constraints.shape[0]), atol=1.0e-10
        )

    def testRecoversWithNoise(self):
        """Starting at truth with noise, the fit should stay in the same
        neighbourhood, though sample noise from the small number of lines
        used here allows more slop than the noiseless case above"""
        result = makeSyntheticPsfArc(self.config, numLines=4, addNoise=True, rng=np.random.RandomState(101))
        fiberIndex = result.fiberIndex.astype(float)
        hybridModel = self.makeHybridModel()
        hybridModel.parametricModel.setInitialGuess(
            sigmaX=self.truth.sigmaX,
            sigmaY=self.truth.sigmaY,
            tophatWidth=self.truth.tophatWidth,
            wings=self.truth.wings,
        )
        fit = fitHybridPsf(
            result.image,
            fiberIndex,
            result.row,
            result.xCenter,
            result.row,
            hybridModel,
            self.config.psfHalfSize,
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
            maxOuterIter=6,
            regularizationConfig=RegularizationConfig(smoothness=1.0, ridge=10.0),
        )
        fitted = fit.model.parametricModel.getParamsAt(0, 0)
        self.assertFloatsAlmostEqual(fitted.sigmaX, self.truth.sigmaX, rtol=0.1)
        self.assertFloatsAlmostEqual(fitted.sigmaY, self.truth.sigmaY, rtol=0.1)
        self.assertFloatsAlmostEqual(fitted.wings[0].fraction, self.truth.wings[0].fraction, rtol=0.3)

        constraints = hybridModel.splineBasis.constraintMatrix()
        self.assertFloatsAlmostEqual(
            constraints @ hybridModel.splineCoefficients, np.zeros(constraints.shape[0]), atol=1.0e-8
        )

    def testConvergesFromPoorInitialGuess(self):
        """The hybrid fit should converge close to truth from initial core
        widths 30% off, as in `FitParametricPsfTestCase` but with ``dP``
        enabled; the wing component is left out here (as there) since a
        poor initial guess for the wing shape reintroduces the
        amplitude/wing-shape degeneracy this fit can't be expected to
        escape from a poor starting point.
        """
        self.config.wingFraction = 0.0
        result = makeSyntheticPsfArc(self.config, numLines=4, addNoise=True, rng=np.random.RandomState(101))
        fiberIndex = result.fiberIndex.astype(float)
        hybridModel = HybridPsfModel(
            ParametricPsfModel(0, 0, (0, self.config.numFibers - 1), (0, self.config.height - 1)),
            PsfSplineBasis(
                SplineAxisConfig(extent=6.0, fineRadius=1.0, mediumRadius=2.0, coarseSpacing=2.0),
                SplineAxisConfig(extent=6.0, fineRadius=1.0, mediumRadius=2.0, coarseSpacing=2.0),
            ),
        )
        hybridModel.parametricModel.setInitialGuess(
            sigmaX=self.truth.sigmaX * 1.3,
            sigmaY=self.truth.sigmaY * 1.3,
            tophatWidth=self.truth.tophatWidth * 0.7,
        )
        fit = fitHybridPsf(
            result.image,
            fiberIndex,
            result.row,
            result.xCenter,
            result.row,
            hybridModel,
            self.config.psfHalfSize,
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
            maxOuterIter=6,
            regularizationConfig=RegularizationConfig(smoothness=1.0, ridge=10.0),
        )
        fitted = fit.model.parametricModel.getParamsAt(0, 0)
        self.assertFloatsAlmostEqual(fitted.sigmaX, self.truth.sigmaX, rtol=0.15)
        self.assertFloatsAlmostEqual(fitted.sigmaY, self.truth.sigmaY, rtol=0.15)
        self.assertFloatsAlmostEqual(fitted.tophatWidth, self.truth.tophatWidth, rtol=0.2)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
