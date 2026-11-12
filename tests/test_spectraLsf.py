"""Claude: the LSF of a set of spectra (the lsf component of pfsArm, pfsMerged, pfsCoadd)"""
import os
import pickle
import tempfile
import unittest
import warnings

import astropy.units as u
import numpy as np

import lsst.utils.tests
from pfs.datamodel import Identity, MaskHelper
from pfs.drp.stella import GaussianLsf, LsfDict
from pfs.drp.stella.armLsf import ArmLsf, FiberSpline, GaussianArmLsf
from pfs.drp.stella.combinedLsf import CombinedLsfConfig, makeCombinedLsf
from pfs.drp.stella.datamodel import PfsArm, PfsMerged
from pfs.drp.stella.smoothSurface import DetectorCoords, SmoothSurface
from pfs.drp.stella.spectraLsf import ArmFiberLsf, BoundArmLsf, CombinedSpectraLsf

LENGTH = 4176
FIBERS = np.array([10, 20, 30])


def makeArmLsf(arm="r", lo=600.0, step=0.05):
    """An `ArmLsf` with smoothly varying parameters"""
    def surface(name, value, slope=0.0):
        coeffs = np.zeros(4)
        coeffs[0] = value
        coeffs[1] = slope  # Varies in v (the row)
        return SmoothSurface(name, (1, 1), ((0.0, 4096.0), (0.0, LENGTH)), coeffs,
                             {"mtp": {"U3-1": (0.01, 0.001)}} if name == "rms" else {},
                             {"mtp": None} if name == "rms" else {})

    surfaces = dict(radius=surface("radius", 1.8, 0.05), rms=surface("rms", 0.65, -0.03),
                    S=surface("S", 0.1), K=surface("K", 0.5))
    xCenter = FiberSpline.fromFunction(FIBERS, LENGTH, 32, lambda ff, rows: 100.0*ff + 0.01*rows)
    wavelength = FiberSpline.fromFunction(FIBERS, LENGTH, 32,
                                          lambda ff, rows: lo + step*rows + 1e-6*rows**2)
    return ArmLsf(arm, 1, FIBERS, LENGTH, surfaces, DetectorCoords(2048.0), xCenter, wavelength,
                  metadata=dict(visits=[1, 2]), engineeringFiberId=np.array([30]),
                  fiberGroups=dict(mtp={10: "U3-1", 20: "U3-2"}))


def makeArm(armLsf, shift=0.0, arm="r", lsf=None):
    """A `PfsArm` from the detector of an LSF, with its wavelengths shifted by ``shift`` nm"""
    rows = np.arange(LENGTH, dtype=float)
    wavelength = np.array([armLsf.wavelength(ff, rows) for ff in FIBERS]) + shift
    num = len(FIBERS)
    flags = MaskHelper()
    flags.add("NO_DATA")
    return PfsArm(Identity(visit=1, arm=arm, spectrograph=1, pfsDesignId=0), FIBERS, wavelength,
                  np.ones((num, LENGTH), dtype=np.float32), np.zeros((num, LENGTH), dtype=np.int32),
                  np.zeros((num, LENGTH), dtype=np.float32), np.ones((num, LENGTH), dtype=np.float32),
                  np.ones((num, 3, LENGTH), dtype=np.float32), flags, {}, lsf=lsf)


class SpectraLsfTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.armLsf = makeArmLsf()

    def assertKernelsEqual(self, left, right, atol=0.0):
        self.assertEqual(left.start, right.start)
        self.assertFloatsAlmostEqual(left.values, right.values, rtol=0, atol=atol)

    def testFits(self):
        """The armLsf dataset round-trips through FITS"""
        with tempfile.TemporaryDirectory() as tempDir:
            filename = os.path.join(tempDir, "armLsf.fits")
            self.armLsf.writeFits(filename)
            copy = ArmLsf.readFits(filename)
        self.assertEqual(copy.metadata, self.armLsf.metadata)
        self.assertEqual(copy.fiberGroups, self.armLsf.fiberGroups)
        self.assertFloatsEqual(copy.engineeringFiberId, [30])
        for ff in FIBERS:
            for row in (3.0, 1234.5, LENGTH - 2.0):
                self.assertKernelsEqual(copy.computeKernel(ff, row), self.armLsf.computeKernel(ff, row))
        rows = np.arange(10.0)
        self.assertFloatsEqual(copy.wavelength(20, rows), self.armLsf.wavelength(20, rows))

    def testBind(self):
        """Binding checks that the spectra and the LSF belong together"""
        lsf = self.armLsf.forSpectra(makeArm(self.armLsf))
        self.assertIsInstance(lsf, BoundArmLsf)
        self.assertEqual(lsf.keys, list(FIBERS))
        with self.assertRaises(RuntimeError):
            self.armLsf.forSpectra(makeArm(self.armLsf, arm="b"))  # Wrong arm
        with self.assertWarns(UserWarning):
            self.armLsf.forSpectra(makeArm(self.armLsf, shift=0.1))  # 2 pixels
        with self.assertRaises(RuntimeError):
            self.armLsf.forSpectra(makeArm(self.armLsf, shift=2.0))  # 40 pixels
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.armLsf.forSpectra(makeArm(self.armLsf, shift=0.01))  # 0.2 pixels: fine

    def testBoundKernels(self):
        """The bound LSF uses the spectra's wavelengths"""
        spectra = makeArm(self.armLsf, shift=0.03)
        lsf = self.armLsf.forSpectra(spectra, warnShift=1.0)
        kernel = lsf.computeKernel(20, 1234.5*u.pix)
        self.assertKernelsEqual(kernel, self.armLsf.computeKernel(20, 1234.5))
        self.assertFloatsEqual(kernel.wavelength.to_value(u.nm), spectra.wavelength[1, kernel.slice])

        # At a wavelength, the kernel is centred on that wavelength in the spectra's solution
        kernel = lsf.computeKernel(20, 700.0*u.nm)
        rows = np.arange(LENGTH, dtype=float)
        self.assertFloatsAlmostEqual(np.interp(kernel.center, rows, spectra.wavelength[1]), 700.0, atol=1e-9)

        # Several positions at once
        batch = lsf.computeKernel(20, [500.2, 1500.7]*u.pix, halfWidth=8*u.pix)
        self.assertEqual(batch.numKernels, 2)
        for ii, cc in enumerate((500.2, 1500.7)):
            single = lsf.computeKernel(20, cc*u.pix, halfWidth=8*u.pix)
            self.assertFloatsEqual(batch[ii].values[single.start - batch[ii].start:][:len(single)],
                                   single.values)

        # Widths, in the units of the position
        width = lsf.computeWidth(20, 1234.5*u.pix)
        self.assertEqual(width.unit, u.pix)
        self.assertFloatsAlmostEqual(width.value, self.armLsf.computeGaussianSigma(20, 1234.5), rtol=1e-12)
        widthNm = lsf.computeWidth(20, 700.0*u.nm)
        self.assertEqual(widthNm.unit, u.nm)
        self.assertGreater(widthNm.value, 0.0)

    def testPfsArm(self):
        """A pfsArm writes and reads its LSF"""
        spectra = makeArm(self.armLsf)
        spectra.lsf = self.armLsf.forSpectra(spectra)
        with tempfile.TemporaryDirectory() as tempDir:
            filename = os.path.join(tempDir, "pfsArm.fits")
            spectra.writeFits(filename)
            copy = PfsArm.readFits(filename)
        self.assertIsInstance(copy.lsf, BoundArmLsf)
        self.assertIsNone(copy.lsf.detectorLsf.wavelength)  # The spectra have their own
        for position in (3.0*u.pix, 1234.5*u.pix, 700.0*u.nm):
            self.assertKernelsEqual(copy.lsf.computeKernel(20, position),
                                    spectra.lsf.computeKernel(20, position))

    def testGaussian(self):
        """The Gaussian placeholder is the old GaussianLsf"""
        legacy = LsfDict({ff: GaussianLsf(LENGTH, 1.1 + 0.01*ff) for ff in FIBERS})
        gaussian = GaussianArmLsf.fromLegacy("r", 1, legacy)
        lsf = gaussian.forSpectra(makeArm(self.armLsf))
        for ff in FIBERS:
            start, values = legacy[ff].computeWindow(1234.5)
            kernel = lsf.computeKernel(ff, 1234.5*u.pix, halfWidth=100*u.pix)
            self.assertEqual(kernel.start, start)
            self.assertFloatsAlmostEqual(kernel.values, values/values.sum(), atol=1e-15)
            self.assertFloatsEqual(lsf.toLegacy()[ff].computeArray(1234.5), legacy[ff].computeArray(1234.5))
        self.assertIsNone(GaussianArmLsf.fromLegacy("r", 1, {10: ArmFiberLsf(self.armLsf, 10)}))

    def testArmFiberLsf(self):
        """One fibre of a detector's LSF, with the Lsf API, is complete at the ends of the row"""
        fiberLsf = ArmFiberLsf(self.armLsf, 20)
        start, values = fiberLsf.computeWindow(0.3, clip=False)
        self.assertLess(start, 0)
        self.assertFloatsAlmostEqual(np.sum(values), 1.0, atol=1e-12)
        start, values = fiberLsf.computeWindow(0.3)
        self.assertEqual(start, 0)
        self.assertEqual(fiberLsf.computeKernel(1234.4).computeStdev(), fiberLsf.computeShape1D(1234.4))
        copy = pickle.loads(pickle.dumps(fiberLsf))
        self.assertFloatsEqual(copy.computeArray(1234.5), fiberLsf.computeArray(1234.5))

    def testCombined(self):
        """Combined spectra (a pfsMerged) write and read their LSF"""
        red = makeArmLsf("r", 620.0, 0.08)
        blue = GaussianArmLsf("b", 1, FIBERS, LENGTH, [1.2, 1.3, 1.4])
        blueArm = makeArm(red, shift=-260.0, arm="b")  # 360-694 nm
        redArm = makeArm(red)  # 620-967 nm
        outWavelength = np.linspace(380.0, 960.0, 6000)
        config = CombinedLsfConfig()
        lsfs = {}
        for ii, ff in enumerate(FIBERS):
            inputs = [ArmFiberLsf(blue, ff), ArmFiberLsf(red, ff)]
            weights = [np.ones(len(outWavelength)), 2*np.ones(len(outWavelength))]
            selects = [np.ones(len(outWavelength), dtype=bool)]*2
            lsfs[int(ff)] = makeCombinedLsf(inputs, [blueArm.wavelength[ii], redArm.wavelength[ii]],
                                            outWavelength, weights, selects, 3, config)
        combined = CombinedSpectraLsf(lsfs, {int(ff): outWavelength for ff in FIBERS})

        num = len(FIBERS)
        flags = MaskHelper()
        merged = PfsMerged(Identity(visit=1, pfsDesignId=0), FIBERS, np.tile(outWavelength, (num, 1)),
                           np.ones((num, 6000), dtype=np.float32), np.zeros((num, 6000), dtype=np.int32),
                           np.zeros((num, 6000), dtype=np.float32), np.ones((num, 6000), dtype=np.float32),
                           np.ones((num, 3, 6000), dtype=np.float32), flags, {}, lsf=combined)
        with tempfile.TemporaryDirectory() as tempDir:
            filename = os.path.join(tempDir, "pfsMerged.fits")
            merged.writeFits(filename)
            copy = PfsMerged.readFits(filename)
        self.assertIsInstance(copy.lsf, CombinedSpectraLsf)
        self.assertEqual(copy.lsf.keys, list(FIBERS))
        for ff in FIBERS:
            for position in (640.0*u.nm, 680.0*u.nm, 800.0*u.nm, 1234.5*u.pix):
                self.assertKernelsEqual(copy.lsf.computeKernel(ff, position),
                                        combined.computeKernel(ff, position), atol=1e-12)
            self.assertFloatsAlmostEqual(copy.lsf.toLegacy()[ff].computeArray(3000.0),
                                         lsfs[int(ff)].computeArray(3000.0), atol=1e-12)
        # The arm LSFs are stored once each, with only the fibres used
        datamodel = combined.toDatamodel()
        self.assertEqual(len(datamodel.arms), 2)
        self.assertEqual(len(datamodel.inputs), 2*num)


class ConvertersTestCase(lsst.utils.tests.TestCase):
    def testConverters(self):
        """The converters to the deprecated storage classes keep the LSF"""
        from pfs.datamodel.drp import PfsCalibrated, PfsCoadd
        from pfs.drp.stella.datamodel.pfsTargetSpectra import PfsCalibratedSpectra, PfsObjectSpectra
        lsf = object()
        self.assertIs(PfsObjectSpectra.fromPfsCoadd(PfsCoadd([], {}, lsf=lsf)).lsf, lsf)
        self.assertIs(PfsCalibratedSpectra.fromPfsCalibrated(PfsCalibrated([], {}, lsf=lsf)).lsf, lsf)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
