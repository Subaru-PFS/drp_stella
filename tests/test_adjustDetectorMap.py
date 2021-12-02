import numpy as np

import lsst.utils.tests
from lsst.geom import Box2D
import lsst.afw.image
from lsst.afw.display import Display

from pfs.drp.stella.adjustDetectorMap import AdjustDetectorMapConfig, AdjustDetectorMapTask
from pfs.drp.stella.buildFiberProfiles import BuildFiberProfilesTask
from pfs.drp.stella.referenceLine import ReferenceLine, ReferenceLineSet, ReferenceLineStatus
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap, makeSyntheticPfsConfig
from pfs.drp.stella.synthetic import makeSyntheticArc, makeSyntheticFlat
from pfs.drp.stella.centroidLines import CentroidLinesTask
from pfs.drp.stella.centroidTraces import CentroidTracesTask
from pfs.drp.stella import DistortedDetectorMap, DetectorDistortion
from pfs.drp.stella.tests.utils import runTests, methodParameters


display = None


class AdjustDetectorMapTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        """Construct a ``SplinedDetectorMap`` to play with"""
        self.synthConfig = SyntheticConfig()
        self.minWl = 650.0
        self.maxWl = 950.0
        self.base = makeSyntheticDetectorMap(self.synthConfig, self.minWl, self.maxWl)
        self.metadata = 123456
        self.darkTime = 12345.6

        distortionOrder = 5
        numCoeffs = DetectorDistortion.getNumDistortionForOrder(distortionOrder)
        xDistortion = np.zeros(numCoeffs, dtype=float)
        yDistortion = np.zeros(numCoeffs, dtype=float)
        rightCcd = np.zeros(6, dtype=float)

        # Introduce a low-order inaccuracy that we will correct with AdjustDetectorMapTask
        xDistortion[0] = 1.5
        xDistortion[1] = 1.23
        xDistortion[2] = -1.23
        yDistortion[0] = -1.5
        yDistortion[1] = 1.23
        yDistortion[2] = -1.23

        distortion = DetectorDistortion(distortionOrder, Box2D(self.base.bbox), xDistortion, yDistortion,
                                        rightCcd)

        visitInfo = lsst.afw.image.VisitInfo(darkTime=self.darkTime)
        metadata = lsst.daf.base.PropertyList()
        metadata.set("METADATA", self.metadata)
        self.distorted = DistortedDetectorMap(self.base, distortion, visitInfo, metadata)

    def assertExpected(self, detMap, checkWavelengths=True):
        """Assert that the ``DistortedDetectorMap``s is as expected

        Parameters
        ----------
        detMap : `pfs.drp.stella.DistortedDetectorMap`
            DetectorMap to check.
        checkWavelengths : `bool`
            Check that wavelengths are correct? If there are no lines, the
            wavelength may be inaccurate, since the straight traces we have in
            the synthetic data don't give any handle on the wavelength fit.
        """
        # Positions as for the original detectorMap
        for fiberId in self.base.getFiberId():
            self.assertFloatsAlmostEqual(detMap.getXCenter(fiberId), self.base.getXCenter(fiberId),
                                         atol=5.0e-3)
            if checkWavelengths:
                self.assertFloatsAlmostEqual(detMap.getWavelength(fiberId), self.base.getWavelength(fiberId),
                                             atol=1.0e-3)

        # Metadata: we only care that what we planted is there;
        # there may be other stuff that we don't care about.
        self.assertTrue(detMap.getMetadata() is not None)
        self.assertTrue(detMap.metadata is not None)
        self.assertIn("METADATA", detMap.metadata.names())
        self.assertEqual(detMap.metadata.get("METADATA"), self.metadata)
        # VisitInfo: only checking one element, assuming the rest are protected by afw unit tests
        self.assertTrue(detMap.visitInfo is not None)
        self.assertTrue(detMap.getVisitInfo() is not None)
        self.assertEqual(detMap.visitInfo.getDarkTime(), self.darkTime)

        # Base hasn't been modified
        self.assertFloatsEqual(detMap.fiberId, self.base.fiberId)
        for ff in detMap.fiberId:
            self.assertFloatsEqual(detMap.base.getXCenterSpline(ff).getX(),
                                   self.base.getXCenterSpline(ff).getX())
            self.assertFloatsEqual(detMap.base.getXCenterSpline(ff).getY(),
                                   self.base.getXCenterSpline(ff).getY())
            self.assertFloatsEqual(detMap.base.getWavelengthSpline(ff).getX(),
                                   self.base.getWavelengthSpline(ff).getX())
            self.assertFloatsEqual(detMap.base.getWavelengthSpline(ff).getY(),
                                   self.base.getWavelengthSpline(ff).getY())
            self.assertEqual(detMap.base.getSpatialOffset(ff), self.base.getSpatialOffset(ff))
            self.assertEqual(detMap.base.getSpectralOffset(ff), self.base.getSpectralOffset(ff))

        # Distortion should be zero after adjustment
        if checkWavelengths:  # If the wavelength is unconstrained, some coefficients may be non-zero
            self.assertFloatsAlmostEqual(detMap.distortion.getXCoefficients(), 0.0, atol=5.0e-3)
            self.assertFloatsAlmostEqual(detMap.distortion.getYCoefficients(), 0.0, atol=5.0e-3)
        self.assertFloatsEqual(detMap.distortion.getRightCcdCoefficients(), 0.0)

    @methodParameters(flatFlux=(10, 1000, 1000),
                      numLines=(50, 0, 50))
    def testAdjustment(self, flatFlux=1000, numLines=0, arcFlux=10000):
        """Test adjustment of a detectorMap

        We construct a synthetic image with both trace and lines, and check that
        we can adjust the detectorMap back to what it should be.

        Parameters
        ----------
        flatFlux : `float`
            Flux per row in the trace.
        numLines : `int`
            Number of lines per fiber.
        arcFlux : `float`
            Flux in each line.
        """
        rng = np.random.RandomState(12345)
        pfsConfig = makeSyntheticPfsConfig(self.synthConfig, 1234567890, 12345)
        image = makeSyntheticFlat(self.synthConfig, flux=flatFlux, addNoise=True, rng=rng)
        arc = makeSyntheticArc(self.synthConfig, numLines=numLines, flux=arcFlux, fwhm=self.synthConfig.fwhm,
                               addNoise=True, rng=rng)
        image += arc.image
        exposure = lsst.afw.image.makeExposure(lsst.afw.image.makeMaskedImage(image))
        exposure.mask.set(0)
        readnoise = self.synthConfig.readnoise
        gain = self.synthConfig.gain
        exposure.variance.array = readnoise**2/gain + image.array*gain

        refLines = []
        fiberId = self.synthConfig.fiberId[self.synthConfig.numFibers//2]
        for yy in arc.lines:
            refLines.append(ReferenceLine("Fake", self.base.findWavelength(fiberId, yy), arcFlux,
                            ReferenceLineStatus.GOOD))
        refLines = ReferenceLineSet.fromRows(refLines)

        profilesConfig = BuildFiberProfilesTask.ConfigClass()
        profilesConfig.pruneMinLength = self.synthConfig.height//2
        profilesConfig.profileRadius = 3
        profilesConfig.doBlindFind = False

        fiberProfiles = BuildFiberProfilesTask(config=profilesConfig).run(exposure, self.base).profiles
        fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(self.base)

        centroidLines = CentroidLinesTask()
        centroidTraces = CentroidTracesTask()
        centroidLines.config.fwhm = self.synthConfig.fwhm
        centroidLines.config.doSubtractContinuum = False  # Can bias the measurements if we're off by a lot
        centroidTraces.config.fwhm = self.synthConfig.fwhm
        if numLines > 0:
            lines = centroidLines.run(exposure, refLines, self.distorted, pfsConfig, fiberTraces)
        else:
            lines = None
        traces = centroidTraces.run(exposure, self.distorted, pfsConfig)

        config = AdjustDetectorMapConfig()
        config.order = 1
        task = AdjustDetectorMapTask(config=config)
        task.log.setLevel(task.log.DEBUG)

        if display is not None:
            disp = Display(frame=1, backend=display)
            disp.mtv(exposure)
            self.distorted.display(disp, ctype="red", wavelengths=refLines.wavelength)
            with disp.Buffering():
                for ll in lines:
                    disp.dot("o", ll.x, ll.y, ctype="blue")
                for ff in traces:
                    for tt in traces[ff]:
                        disp.dot("+", tt.peak, tt.row, ctype="blue")

        adjusted = task.run(self.distorted, lines, traces=traces)
        self.assertExpected(adjusted.detectorMap, checkWavelengths=(numLines > 0))

        if display is not None:
            adjusted.detectorMap.display(disp, ctype="green", wavelengths=lines.wavelength)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
