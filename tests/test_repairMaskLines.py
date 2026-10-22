import numpy as np

import lsst.utils.tests
import lsst.afw.image as afwImage

from pfs.drp.stella.referenceLine import ReferenceLineSet, ReferenceLineStatus
from pfs.drp.stella.repair import maskLines
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella.tests.utils import runTests

display = None


def makeReferenceLines(wavelength):
    """Construct a `ReferenceLineSet` with the given wavelengths

    Parameters
    ----------
    wavelength : `numpy.ndarray` of `float`
        Line wavelengths (nm).

    Returns
    -------
    refLines : `ReferenceLineSet`
        Line list.
    """
    num = len(wavelength)
    return ReferenceLineSet.fromColumns(
        description=["Fake"] * num,
        wavelength=np.asarray(wavelength, dtype=float),
        intensity=np.ones(num, dtype=float),
        status=np.full(num, ReferenceLineStatus.GOOD, dtype=np.int32),
        transition=["UNKNOWN"] * num,
        source=np.zeros(num, dtype=np.int32),
        intensityErr=np.full(num, np.nan),
    )


class RepairMaskLinesTestCase(lsst.utils.tests.TestCase):
    """Test pfs.drp.stella.repair.maskLines

    This sets a mask plane in a disk around the position of each reference
    line in each fiber.
    """

    def setUp(self):
        synth = SyntheticConfig()
        synth.width = 200
        synth.height = 300
        synth.separation = 30.0
        synth.slope = 0.05  # So the traces aren't vertical
        self.synth = synth
        self.detectorMap = makeSyntheticDetectorMap(synth, minWl=500.0, maxWl=800.0)
        self.bbox = self.detectorMap.bbox
        self.wavelength = np.array([550.123, 640.0, 712.345])
        self.refLines = makeReferenceLines(self.wavelength)

    def tearDown(self):
        del self.detectorMap
        del self.refLines

    def makeMask(self):
        """Return an empty mask covering the detector"""
        mask = afwImage.Mask(self.bbox)
        mask.set(0)
        return mask

    def calculateExpected(self, wavelength, radius):
        """Calculate the expected masked pixels

        This is an independent implementation of the masking: integer
        pixels within ``radius`` of the line position, rounded to the
        nearest pixel.

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            Line wavelengths (nm).
        radius : `int`
            Masking radius (pixels).

        Returns
        -------
        expected : `numpy.ndarray` of `bool`
            Pixels expected to be masked.
        """
        height, width = self.bbox.getHeight(), self.bbox.getWidth()
        yy, xx = np.mgrid[:height, :width]
        expected = np.zeros((height, width), dtype=bool)
        for fiberId in self.detectorMap.fiberId:
            for wl in wavelength:
                point = self.detectorMap.findPoint(fiberId, wl)
                if not np.all(np.isfinite(point)):
                    continue
                xCenter = int(np.floor(point.getX() + 0.5))
                yCenter = int(np.floor(point.getY() + 0.5))
                expected |= (xx - xCenter) ** 2 + (yy - yCenter) ** 2 <= radius**2
        return expected

    def testBasic(self):
        """Masked pixels are disks around each line in each fiber"""
        for radius in (0, 1, 3, 7):
            with self.subTest(radius=radius):
                mask = self.makeMask()
                maskLines(mask, self.detectorMap, self.refLines, radius)
                bitmask = mask.getPlaneBitMask("REFLINE")  # Default mask plane
                self.assertTrue(np.all((mask.array & ~bitmask) == 0))  # Nothing else set
                masked = (mask.array & bitmask) != 0
                expected = self.calculateExpected(self.wavelength, radius)
                self.assertTrue(np.array_equal(masked, expected))

                # Disks don't overlap or touch the edge with this geometry
                numPoints = len(self.detectorMap) * len(self.wavelength)
                offsets = np.arange(-radius, radius + 1)
                diskArea = np.sum(offsets[:, np.newaxis] ** 2 + offsets[np.newaxis, :] ** 2 <= radius**2)
                self.assertEqual(masked.sum(), numPoints * diskArea)

    def testMaskPlane(self):
        """A specified mask plane is set, and other bits are preserved"""
        mask = self.makeMask()
        bad = mask.getPlaneBitMask("BAD")
        mask.array[:10, :] = bad
        maskLines(mask, self.detectorMap, self.refLines, 3, maskPlane="CR")
        bitmask = mask.getPlaneBitMask("CR")
        self.assertTrue(np.all((mask.array & ~(bitmask | bad)) == 0))
        self.assertTrue(np.all((mask.array[:10, :] & bad) != 0))
        self.assertTrue(np.all((mask.array[10:, :] & bad) == 0))
        masked = (mask.array & bitmask) != 0
        self.assertTrue(np.array_equal(masked, self.calculateExpected(self.wavelength, 3)))

    def testEdge(self):
        """Disks overlapping the edge of the image are clipped"""
        # Wavelength of the first row of the detector, so half of each disk falls off the image
        fiberId = self.detectorMap.fiberId[0]
        wavelength = [self.detectorMap.findWavelength(fiberId, 0.0)]
        refLines = makeReferenceLines(wavelength)
        mask = self.makeMask()
        maskLines(mask, self.detectorMap, refLines, 5)
        masked = (mask.array & mask.getPlaneBitMask("REFLINE")) != 0
        expected = self.calculateExpected(wavelength, 5)
        self.assertTrue(np.array_equal(masked, expected))
        self.assertGreater(masked[0].sum(), 0)

    def testOutOfRange(self):
        """Lines that don't land on the detector are ignored"""
        mask = self.makeMask()
        refLines = makeReferenceLines([300.0, 1000.0])  # Outside the detectorMap's wavelength range
        maskLines(mask, self.detectorMap, refLines, 5)
        self.assertTrue(np.all(mask.array == 0))

    def testNoLines(self):
        """No lines, or no detectorMap: nothing is masked"""
        for refLines, detectorMap in (
            (None, self.detectorMap),
            (makeReferenceLines([]), self.detectorMap),
            (self.refLines, None),
        ):
            mask = self.makeMask()
            maskLines(mask, detectorMap, refLines, 5)
            self.assertTrue(np.all(mask.array == 0))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
