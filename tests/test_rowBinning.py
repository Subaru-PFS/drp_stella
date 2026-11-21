import numpy as np

import lsst.utils.tests
import lsst.afw.image
import lsst.afw.image.testUtils

from pfs.datamodel import CalibIdentity
from pfs.datamodel.pfsFiberNorms import PfsFiberNorms
from pfs.drp.stella.arcLine import ArcLineSet
from pfs.drp.stella.buildFiberProfiles import BuildFiberProfilesTask
from pfs.drp.stella.rowBinning import (
    RowBinning, binDetectorMap, binFiberNorms, binFiberTraces, scaleSpectraNorm, unbinLines
)
from pfs.drp.stella.synthetic import makeSyntheticFlat, SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella.tests import runTests

display = None


class RowBinningTestCase(lsst.utils.tests.TestCase):
    """Test extraction of spectra from data binned in the parallel direction"""
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.synthConfig = SyntheticConfig()
        self.synthConfig.height = 256  # Not a multiple of the binning, so the last group is partial
        self.synthConfig.width = 128
        self.synthConfig.fwhm = 3.21
        self.synthConfig.separation = 12.3
        self.synthConfig.slope = 0.01

        self.binning = RowBinning(20, self.synthConfig.height)
        self.assertEqual(self.binning.numBinned, 13)
        self.assertEqual(self.binning.counts[-1], 16)

        self.detMap = makeSyntheticDetectorMap(self.synthConfig)
        self.flux = 1.0e5
        image = makeSyntheticFlat(self.synthConfig, flux=self.flux, addNoise=False, rng=self.rng)
        self.image = lsst.afw.image.makeMaskedImage(image)
        self.image.mask.array[:] = 0x0
        self.image.variance.array[:] = self.synthConfig.readnoise**2
        self.identity = CalibIdentity("2020-01-01", 5, "x", 12345)

        config = BuildFiberProfilesTask.ConfigClass()
        config.pruneMinLength = self.synthConfig.height//2
        task = BuildFiberProfilesTask(config=config)
        self.fiberProfiles = task.run(
            lsst.afw.image.makeExposure(self.image), self.identity, detectorMap=self.detMap
        ).profiles
        for fiberId in self.fiberProfiles:
            self.fiberProfiles[fiberId].norm = np.full(self.synthConfig.height, self.flux, dtype=np.float32)
        self.fiberTraces = self.fiberProfiles.makeFiberTracesFromDetectorMap(self.detMap)
        self.assertEqual(len(self.fiberTraces), self.synthConfig.numFibers)

    def makeBinnedImage(self):
        """Make a binned image, as read out by the binned CCD

        Each binned pixel contains the sum of the charge in the unbinned
        pixels, with the variance of a single read.
        """
        binning = self.binning
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                               lsst.geom.Extent2I(self.synthConfig.width, binning.numBinned))
        binned = lsst.afw.image.MaskedImageF(bbox)
        binned.image.array[:] = binning.sum(self.image.image.array)
        binned.mask.array[:] = 0
        binned.variance.array[:] = self.synthConfig.readnoise**2
        return binned

    def testDetectorMap(self):
        """Test binning of the detectorMap"""
        binning = self.binning
        binned = binDetectorMap(self.detMap, binning)
        self.assertEqual(binned.getBBox().getHeight(), binning.numBinned)
        self.assertEqual(binned.getBBox().getWidth(), self.detMap.getBBox().getWidth())
        self.assertFloatsEqual(binned.fiberId, self.detMap.fiberId)
        for fiberId in self.detMap.fiberId:
            self.assertFloatsAlmostEqual(
                binned.getXCenter(fiberId), binning.mean(self.detMap.getXCenter(fiberId)), atol=1.0e-5
            )
            self.assertFloatsAlmostEqual(
                binned.getWavelength(fiberId), binning.mean(self.detMap.getWavelength(fiberId)), atol=1.0e-5
            )
            # Positions of full groups map to the centers of the groups
            row = np.arange(binning.numBinned - 1, dtype=float)
            wavelength = np.array([binned.getWavelength(fiberId, yy) for yy in row])
            unbinnedRow = np.array([self.detMap.findPoint(fiberId, wl).getY() for wl in wavelength])
            self.assertFloatsAlmostEqual(unbinnedRow, binning.toUnbinned(row), atol=1.0e-2)
        # Dispersion is larger by the binning factor
        fiberId = self.detMap.fiberId[0]
        self.assertFloatsAlmostEqual(
            binned.getDispersionAtCenter(fiberId),
            binning.factor*self.detMap.getDispersionAtCenter(fiberId),
            rtol=1.0e-3,
        )

    def testDetectorMapWavelength(self):
        """Test binning of the detectorMap with wavelengths from another"""
        binning = self.binning
        other = self.detMap.clone()
        other.applySlitOffset(0.0, 1.0)  # Shift by a pixel in the spectral direction
        binned = binDetectorMap(self.detMap, binning, other)
        for fiberId in self.detMap.fiberId:
            self.assertFloatsAlmostEqual(
                binned.getXCenter(fiberId), binning.mean(self.detMap.getXCenter(fiberId)), atol=1.0e-5
            )
            self.assertFloatsAlmostEqual(
                binned.getWavelength(fiberId), binning.mean(other.getWavelength(fiberId)), atol=1.0e-5
            )

    def testExtraction(self):
        """Test extraction of binned data with binned fiberTraces"""
        binning = self.binning
        image = self.makeBinnedImage()
        fiberTraces = binFiberTraces(self.fiberTraces, binning)
        self.assertEqual(len(fiberTraces), len(self.fiberTraces))
        for ft in fiberTraces:
            self.assertEqual(ft.trace.getHeight(), binning.numBinned)

        spectra = fiberTraces.extractSpectra(image)
        scaleSpectraNorm(spectra, binning)
        self.assertEqual(spectra.getLength(), binning.numBinned)
        for ss in spectra:
            # flux is the number of electrons in the binned pixel
            self.assertFloatsAlmostEqual(ss.flux, self.flux*binning.counts, rtol=2.0e-3)
            # norm is the sum of the unbinned norms
            self.assertFloatsAlmostEqual(ss.norm, self.flux*binning.counts, rtol=2.0e-3)
            # variance is consistent with a single read per binned pixel
            self.assertTrue(np.all(ss.variance > 0))
            self.assertTrue(np.all(ss.mask.array == 0))

    def testExtractionReplicated(self):
        """Test that collapsing a full-sized image with replicated rows works

        This is what the post-ISR image of binned data looks like.
        """
        binning = self.binning
        binned = self.makeBinnedImage()
        replicated = lsst.afw.image.MaskedImageF(self.image.getBBox())
        replicated.image.array[:] = binning.expand(binned.image.array)
        replicated.mask.array[:] = binning.expand(binned.mask.array)
        replicated.variance.array[:] = binning.expand(binned.variance.array)
        replicated.mask.array[100, 50] |= replicated.mask.getPlaneBitMask("BAD")

        collapsed = binning.binMaskedImage(replicated)
        self.assertImagesEqual(collapsed.image, binned.image)
        self.assertImagesEqual(collapsed.variance, binned.variance)
        self.assertEqual(collapsed.mask.array[binning.index[100], 50],
                         collapsed.mask.getPlaneBitMask("BAD"))
        self.assertEqual(np.sum(collapsed.mask.array != 0), 1)

    def testBoxcar(self):
        """Test boxcar extraction of binned data"""
        binning = self.binning
        image = self.makeBinnedImage()
        boxcarWidth = 5
        traces = self.fiberProfiles.makeFiberTracesFromDetectorMap(self.detMap, boxcarWidth)
        unbinned = [ft.extractAperture(self.image, 0) for ft in traces]
        fiberTraces = binFiberTraces(traces, binning)
        for ft, uu in zip(fiberTraces, unbinned):
            ss = ft.extractAperture(image, 0)
            norm = ss.norm.copy()
            norm *= binning.counts
            self.assertFloatsAlmostEqual(ss.flux, binning.sum(uu.flux), rtol=1.0e-2)
            self.assertFloatsAlmostEqual(norm, binning.sum(uu.norm), rtol=1.0e-2)

    def testFiberNorms(self):
        """Test binning of fiberNorms"""
        binning = self.binning
        numFibers = len(self.detMap)
        height = binning.height
        wavelength = np.array([self.detMap.getWavelength(ff) for ff in self.detMap.fiberId])
        values = 1.0 + 0.1*self.rng.normal(size=(numFibers, height))
        values[0, 3] = np.nan  # Bad value that should be ignored
        values[1, :binning.factor] = 0.0  # Entire group is bad
        fiberNorms = PfsFiberNorms(self.identity, self.detMap.fiberId, wavelength, values, {}, None)
        binned = binFiberNorms(fiberNorms, binning)
        self.assertEqual(binned.values.shape, (numFibers, binning.numBinned))
        self.assertFloatsAlmostEqual(binned.wavelength, binning.mean(wavelength, axis=1), atol=1.0e-8)
        self.assertFloatsAlmostEqual(binned.values[2:], binning.mean(values[2:], axis=1), atol=1.0e-8)
        self.assertFloatsAlmostEqual(binned.values[0, 0], np.mean(np.delete(values[0, :binning.factor], 3)),
                                     atol=1.0e-8)
        self.assertTrue(np.isnan(binned.values[1, 0]))

    def testLines(self):
        """Test conversion of line measurements to unbinned coordinates"""
        binning = self.binning
        num = 10
        yBinned = self.rng.uniform(0, binning.numBinned - 1, size=num)
        lines = ArcLineSet.fromColumns(
            fiberId=np.full(num, 1, dtype=int),
            wavelength=np.linspace(500, 600, num),
            x=self.rng.uniform(0, 100, size=num),
            y=yBinned,
            xErr=np.full(num, 0.1),
            yErr=np.full(num, 0.2),
            xx=np.full(num, 1.0),
            yy=np.full(num, 0.5),
            xy=np.full(num, 0.1),
            flux=np.full(num, 1000.0),
            fluxErr=np.full(num, 10.0),
            fluxNorm=np.full(num, 1.0),
            flag=np.zeros(num, dtype=bool),
            status=np.zeros(num, dtype=int),
            description=["Trace"]*num,
            transition=["UNKNOWN"]*num,
            source=np.zeros(num, dtype=int),
        )
        unbinned = unbinLines(lines, binning)
        self.assertFloatsAlmostEqual(unbinned.y, binning.toUnbinned(yBinned), atol=1.0e-3)
        self.assertFloatsAlmostEqual(binning.toBinned(unbinned.y), yBinned, atol=1.0e-4)
        self.assertFloatsAlmostEqual(unbinned.yErr, 0.2*binning.factor, atol=1.0e-5)
        self.assertFloatsAlmostEqual(unbinned.yy, 0.5*binning.factor**2, atol=1.0e-4)
        self.assertFloatsAlmostEqual(unbinned.xy, 0.1*binning.factor, atol=1.0e-5)
        self.assertFloatsEqual(unbinned.x, lines.x)
        self.assertFloatsEqual(lines.y, yBinned.astype(np.float32))  # input unchanged


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
