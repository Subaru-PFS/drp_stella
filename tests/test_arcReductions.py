from types import SimpleNamespace

import numpy as np

import lsst.utils.tests

from pfs.drp.stella.referenceLine import ReferenceLineSet, ReferenceLineStatus, ReferenceLineSource
from pfs.drp.stella.lineLists.reductions import crMaskedFraction, compareReductions, compareLineLists
from pfs.drp.stella.tests.utils import runTests

display = None


def makeLines(wavelength, intensity, intensityErr=1.0):
    return ReferenceLineSet.fromColumns(wavelength=np.asarray(wavelength, dtype=float),
                                        intensity=np.asarray(intensity, dtype=float),
                                        intensityErr=intensityErr,
                                        status=ReferenceLineStatus.GOOD, description="Fake", transition=None,
                                        source=ReferenceLineSource.NONE)


class FakeDetectorMap:
    """Line at x = 10*fiberId, y = 100*(wavelength - 600)"""
    def __init__(self, numFibers):
        self.fiberId = np.arange(1, numFibers + 1)

    def findPoint(self, fiberId, wavelength):
        return 10.0*fiberId, 100.0*(wavelength - 600)


class FakeButler:
    """pfsArm, crMask and detectorMap for one visit, r arm, spectrograph 1"""
    def __init__(self, lineFluxes, numFibers=30, scramble=False, crLines=()):
        wavelength = np.tile(np.arange(600.0, 700.0, 0.02), (numFibers, 1))
        flux = np.full(wavelength.shape, 3.0)
        sigma = 0.1
        for wl, total in lineFluxes.items():
            profile = total*0.02/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*((wavelength - wl)/sigma)**2)
            if scramble:   # move flux between neighbouring fibres: the median changes, the mean doesn't
                profile *= np.where(np.arange(numFibers) % 3 == 0, 2.0, 0.5)[:, np.newaxis]
            flux += profile
        self.pfsArm = FakeSpectra(wavelength, flux)
        self.detectorMap = FakeDetectorMap(numFibers)
        mask = np.zeros((10000, 10*numFibers + 20), dtype=np.int32)
        for wl in crLines:
            mask[int(100*(wl - 600)), :] = 1
        self.crMask = SimpleNamespace(array=mask)

    def get(self, datasetType, dataId):
        if dataId["arm"] != "r" or dataId["spectrograph"] != 1:
            raise LookupError(datasetType)
        return dict(pfsArm=self.pfsArm, crMask=self.crMask, detectorMap=self.detectorMap)[datasetType]


class FakeSpectra:
    def __init__(self, wavelength, flux):
        self.wavelength = wavelength
        self.flux = flux
        self.mask = np.zeros(flux.shape, dtype=np.int32)

    def __len__(self):
        return len(self.flux)


class ArcReductionsTestCase(lsst.utils.tests.TestCase):
    def testCrMaskedFraction(self):
        """Fraction of the fibres used whose line core is flagged"""
        butler = FakeButler({}, numFibers=40, crLines=(650.0,))
        mask = butler.crMask.array
        self.assertEqual(crMaskedFraction(butler.crMask, butler.detectorMap, 650.0, fiberStep=1), 1.0)
        self.assertEqual(crMaskedFraction(butler.crMask, butler.detectorMap, 640.0, fiberStep=1), 0.0)
        mask[:, :10*20 + 5] = 0   # unflag fibres 1-20
        self.assertEqual(crMaskedFraction(butler.crMask, butler.detectorMap, 650.0, fiberStep=1), 0.5)
        self.assertTrue(np.isnan(crMaskedFraction(butler.crMask, butler.detectorMap, 750.0)))

    def testCompareReductions(self):
        """Old/new flux ratios of the brightest lines; the mean is robust to scrambling between fibres"""
        lines = makeLines([640.0, 650.0, 680.0, 1000.0], [1000.0, 100.0, 10.0, 1e6])
        new = FakeButler({640.0: 1000.0, 650.0: 100.0, 680.0: 10.0})
        old = FakeButler({640.0: 1000.0, 650.0: 50.0, 680.0: 10.0}, scramble=True, crLines=(650.0,))
        table = compareReductions(old, new, 1, lines, numLines=2).set_index("wavelength")
        self.assertEqual(list(table.index), [640.0, 650.0])   # the brightest two in r; 1000 isn't in r
        self.assertFloatsAlmostEqual(table.ratio.to_numpy(), [1.0, 0.5], rtol=1e-3)
        self.assertFloatsAlmostEqual(table.loc[640.0, "ratioMedian"], 0.5, rtol=1e-3)
        self.assertFloatsAlmostEqual(table.crFrac.to_numpy(), [0.0, 1.0])

    def testCompareLineLists(self):
        """Ratios, S/N and flags of lines measured in both lists"""
        old = makeLines([600.0, 610.0, 620.0, 630.0], [100.0, 50.0, 10.0, 5.0],
                        intensityErr=[1.0, 1.0, 1.0, 0.0])
        new = makeLines([600.0, 610.0, 620.0, 630.0], [100.0, 40.0, 10.5, 5.0])
        table = compareLineLists(old, new).set_index("wavelength")
        self.assertEqual(list(table.index), [600.0, 610.0, 620.0])   # 630 has no error: unmeasured
        self.assertFloatsAlmostEqual(table.ratio.to_numpy(), [1.0, 1.25, 10/10.5])
        np.testing.assert_array_equal(table.bright, [True, True, False])
        np.testing.assert_array_equal(table.changed, [False, True, True])
        with self.assertRaises(ValueError):
            compareLineLists(old, makeLines([600.0, 610.0, 620.0], [1.0, 1.0, 1.0]))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
