import os
import tempfile

import numpy as np

import lsst.utils.tests
import lsst.daf.base
from lsst.afw.image import VisitInfo

from pfs.drp.stella.arcLine import ArcLine, ArcLineSet
from pfs.drp.stella.fitDetectorMap import FittingError
from pfs.drp.stella.pipelines.fitDetectorMapCombined import FitDetectorMapCombinedTask
from pfs.drp.stella.referenceLine import ReferenceLineSource, ReferenceLineStatus
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella.tests import runTests


def makeArcLines(detectorMap, fiberId, description, rowStep=50, xOffset=0.0, yOffset=0.0):
    """Construct a noiseless synthetic ArcLineSet from a detectorMap's truth

    Parameters
    ----------
    detectorMap : `pfs.drp.stella.DetectorMap`
        DetectorMap providing the true fiberId,wavelength <--> x,y mapping.
    fiberId : iterable of `int`
        Fiber identifiers to generate lines for.
    description : `str`
        Line description to record (e.g. ``"Fake"`` or ``"solar"``).
    rowStep : `int`
        Spacing (pixels) between synthetic line rows.
    xOffset, yOffset : `float`
        Constant offset (pixels) to apply to the line positions, simulating a
        systematic (e.g. slit) shift relative to the detectorMap truth.

    Returns
    -------
    lines : `pfs.drp.stella.ArcLineSet`
        Synthetic line measurements.
    """
    bbox = detectorMap.bbox
    rows = np.arange(bbox.getMinY() + 1, bbox.getMaxY(), rowStep, dtype=float)
    rows = list(rows)
    lines = []
    for ff in fiberId:
        for yy in rows:
            wavelength = detectorMap.findWavelength(ff, yy)
            xCenter = detectorMap.getXCenter(ff, yy)
            lines.append(
                ArcLine(
                    ff,
                    wavelength,
                    xCenter + xOffset,
                    yy + yOffset,
                    0.01,
                    0.01,
                    np.nan,
                    np.nan,
                    np.nan,
                    1000.0,
                    1.0,
                    np.nan,
                    False,
                    ReferenceLineStatus.GOOD,
                    description,
                    None,
                    ReferenceLineSource.NONE,
                )
            )
    return ArcLineSet.fromRows(lines)


class FitDetectorMapCombinedTestCase(lsst.utils.tests.TestCase):
    """Test FitDetectorMapCombinedTask.run() with purely synthetic data

    We avoid image rendering/centroiding entirely: line positions are read
    directly from a synthetic ``SplinedDetectorMap``'s truth, following the
    pattern used by ``test_OpticalModelDetectorMap.testFit``.
    """

    def setUp(self):
        self.synthConfig = SyntheticConfig()
        self.synthConfig.separation = 100  # Avoid having a fiber go down the middle
        self.minWl = 650.0
        self.maxWl = 950.0
        self.base = makeSyntheticDetectorMap(self.synthConfig, self.minWl, self.maxWl)
        self.arm = "n"  # Avoids the brm chip-gap logic in getBaseDetectorMap
        self.spectrograph = 1
        self.dataId = dict(arm=self.arm, spectrograph=self.spectrograph)
        self.visitInfo = VisitInfo(id=12345)
        self.metadata = lsst.daf.base.PropertyList()

        self.tmpDir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpDir.cleanup)
        self.base.writeFits(os.path.join(self.tmpDir.name, "detectorMap-sim-n1.fits"))
        self.basePathTemplate = os.path.join(self.tmpDir.name, "detectorMap-sim-%(arm)s%(spectrograph)s.fits")

    def makeTask(self, maxIterations=5, convergenceTolerance=0.01, requireConvergence=False):
        config = FitDetectorMapCombinedTask.ConfigClass()
        config.fitDetectorMap.order = 1
        config.fitDetectorMap.doSlitOffsets = False
        config.fitDetectorMap.exclusionRadius = 1.0
        config.fitDetectorMap.base = self.basePathTemplate
        config.adjustDetectorMap.order = 1
        config.adjustDetectorMap.doSlitOffsets = False
        config.adjustDetectorMap.exclusionRadius = 1.0
        config.adjustDetectorMap.base = self.basePathTemplate
        config.maxIterations = maxIterations
        config.convergenceTolerance = convergenceTolerance
        config.requireConvergence = requireConvergence
        return FitDetectorMapCombinedTask(config=config)

    def testArcOnly(self):
        """With no twilight lines, the loop should fit once and stop"""
        task = self.makeTask()
        lines = makeArcLines(self.base, self.synthConfig.fiberId, "Fake")
        result = task.run(self.dataId, [lines], self.visitInfo, self.metadata, self.base.bbox)

        self.assertFloatsAlmostEqual(result.detectorMap.getXCenter(), self.base.getXCenter(), atol=1.0e-3)
        self.assertFloatsAlmostEqual(
            result.detectorMap.getWavelength(), self.base.getWavelength(), rtol=1.0e-5
        )
        self.assertEqual(len(result.lines), len(lines))
        self.assertTrue(np.all(result.lines.description == "Fake"))

    def testConverges(self):
        """Twilight lines consistent with the arc truth converge immediately"""
        task = self.makeTask(requireConvergence=True)
        arcLines = makeArcLines(self.base, self.synthConfig.fiberId, "Fake")
        twilightLines = makeArcLines(self.base, self.synthConfig.fiberId, "solar")

        result = task.run(
            self.dataId, [arcLines, twilightLines], self.visitInfo, self.metadata, self.base.bbox
        )

        self.assertFloatsAlmostEqual(result.detectorMap.getXCenter(), self.base.getXCenter(), atol=1.0e-3)
        self.assertEqual(len(result.lines), len(arcLines) + len(twilightLines))
        self.assertTrue(np.any(result.lines.description == "solar"))

    def testRequireConvergenceRaises(self):
        """A twilight offset that can't settle within one iteration raises"""
        task = self.makeTask(maxIterations=1, convergenceTolerance=1.0e-9, requireConvergence=True)
        arcLines = makeArcLines(self.base, self.synthConfig.fiberId, "Fake")
        twilightLines = makeArcLines(self.base, self.synthConfig.fiberId, "solar", xOffset=0.5, yOffset=-0.3)

        with self.assertRaises(FittingError):
            task.run(self.dataId, [arcLines, twilightLines], self.visitInfo, self.metadata, self.base.bbox)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
