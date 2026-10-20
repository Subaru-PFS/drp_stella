import os
import pickle
import sys
import tempfile
import types
import unittest

import numpy as np

import lsst.utils.tests

from pfs.drp.stella.lsfProfile import diskProfile, diskSigma
from pfs.drp.stella.smoothSurface import DetectorCoords, SmoothSurface, fitSmoothSurface, supportedDegree
from pfs.drp.stella.lsfModel import ArmLsfModel, FiberSpline
from pfs.drp.stella.fitLsfLines import fitDiskLine
from pfs.drp.stella.lsfLine import LsfLineSet, LsfLineFlag

display = None


def moments(pix, values):
    """Mean, variance and skewness of ``values`` sampled at ``pix``"""
    norm = np.sum(values)
    mean = np.sum(pix*values)/norm
    var = np.sum((pix - mean)**2*values)/norm
    skew = np.sum((pix - mean)**3*values)/norm/var**1.5
    return mean, var, skew


def constantSurface(name, value):
    """A `SmoothSurface` with a constant value"""
    return SmoothSurface(name, (0, 0), ((0.0, 4096.0), (0.0, 4176.0)), np.array([value]), {}, {})


def makeModel(radius=1.8, rms=0.65, S=0.1, K=0.5, length=4176, fiberId=(10, 20, 30)):
    """Make an `ArmLsfModel` with constant parameters"""
    fiberId = np.array(fiberId)
    surfaces = {name: constantSurface(name, value) for name, value in
                dict(radius=radius, rms=rms, S=S, K=K, centroid=0.0).items()}
    xCenter = FiberSpline.fromFunction(fiberId, length, 32, lambda ff, rows: 100.0*ff + 0.01*rows)
    wavelength = FiberSpline.fromFunction(fiberId, length, 32,
                                          lambda ff, rows: 600 + 0.05*rows + 1e-6*rows**2)
    return ArmLsfModel("r", 1, fiberId, length, surfaces, DetectorCoords(2048.0), xCenter, wavelength)


class LsfProfileTestCase(lsst.utils.tests.TestCase):
    """Tests for pfs.drp.stella.lsfProfile"""
    def testMoments(self):
        """The profile has unit area, first moment mu, and the expected variance"""
        pix = np.arange(-30, 31, dtype=float)
        radius, rms = 1.7, 0.6
        for mu in (0.0, 0.37, -0.81):
            values = diskProfile(pix, mu, radius, rms, extraction="boxcar")
            self.assertFloatsAlmostEqual(values.sum(), 1.0, atol=1.0e-10)
            mean, var, _ = moments(pix, values)
            self.assertFloatsAlmostEqual(mean, mu, atol=3.0e-5)   # quadrature over the disk
            # disk projected to a semicircle (r^2/4) + rms^2 + pixel integration (1/12)
            self.assertFloatsAlmostEqual(var, diskSigma(radius, rms)**2 + 1/12, rtol=2.0e-3)

    def testEdgeworth(self):
        """The Edgeworth terms keep the area and first moment, and S gives the skewness's sign"""
        pix = np.arange(-30, 31, dtype=float)
        for S in (-0.3, 0.3):
            values = diskProfile(pix, 0.2, 1.7, 0.6, [S, 0.4])
            self.assertFloatsAlmostEqual(values.sum(), 1.0, atol=1.0e-10)
            mean, _, skew = moments(pix, values)
            self.assertFloatsAlmostEqual(mean, 0.2, atol=3.0e-5)   # quadrature over the disk
            self.assertEqual(np.sign(skew), np.sign(S))

    def testLorentz(self):
        """Convolving with a Lorentzian keeps the area (to the dropped wings) and the first moment"""
        pix = np.arange(-40, 41, dtype=float)
        lorentz = 0.12
        core = diskProfile(pix, 0.3, 1.7, 0.6, [0.2, 0.5])
        values = diskProfile(pix, 0.3, 1.7, 0.6, [0.2, 0.5], lorentz=lorentz)
        # The Lorentzian's wings beyond 25 pixels are dropped
        self.assertFloatsAlmostEqual(values.sum(), 1.0 - 2*np.arctan(0.5*lorentz/25.0)/np.pi, atol=1.0e-4)
        mean, _, _ = moments(pix, values)
        self.assertFloatsAlmostEqual(mean, 0.3, atol=3.0e-5)
        # The wings: the Lorentzian adds ~ (lorentz/2/pi)/dx^2 per pixel far from the core
        far = (np.abs(pix - 0.3) > 8) & (np.abs(pix - 0.3) < 20)
        self.assertFloatsAlmostEqual((values - core)[far], 0.5*lorentz/np.pi/(pix[far] - 0.3)**2, rtol=0.1)
        self.assertFloatsAlmostEqual(diskProfile(pix, 0.3, 1.7, 0.6, [0.2, 0.5], lorentz=0.0), core, atol=0.0)

    def testHalo(self):
        """A halo keeps the area and first moment, and adds light in the wings"""
        pix = np.arange(-40, 41, dtype=float)
        core = diskProfile(pix, 0.3, 1.7, 0.6, [0.2, 0.5])
        values = diskProfile(pix, 0.3, 1.7, 0.6, [0.2, 0.5], haloFrac=0.03, haloWidth=2.2)
        self.assertFloatsAlmostEqual(values.sum(), 1.0, atol=1.0e-8)
        mean, var, _ = moments(pix, values)
        self.assertFloatsAlmostEqual(mean, 0.3, atol=5.0e-5)   # quadrature over the disk
        meanCore, varCore, _ = moments(pix, core)
        # variance adds linearly
        self.assertFloatsAlmostEqual(var - varCore, 0.03*(2.2**2 - 0.6**2), rtol=0.02)
        far = np.abs(pix - 0.3) > 5
        self.assertTrue(np.all(values[far] >= core[far]))


class LsfModelTestCase(lsst.utils.tests.TestCase):
    """Tests for pfs.drp.stella.lsfModel"""
    def setUp(self):
        self.model = makeModel()

    def testKernel(self):
        """A kernel is short, normalised, and has its first moment at the centre"""
        kernel = self.model.computeKernel(20, 1000.3, halfWidth=10)
        self.assertEqual(kernel.start, 990)
        self.assertEqual(len(kernel), 21)
        self.assertEqual(kernel.slice, slice(990, 1011))
        self.assertFloatsAlmostEqual(kernel.values.sum(), 1.0, atol=1.0e-12)
        self.assertGreater(kernel.norm, 0.999)
        self.assertFloatsAlmostEqual(np.sum(kernel.pixels*kernel.values), 1000.3, atol=1.0e-4)
        self.assertFloatsAlmostEqual(kernel.offsets, kernel.pixels - 1000.3, atol=0.0)

    def testWindow(self):
        """The window is symmetric about the centre: the fraction of each pixel within halfWidth"""
        unwindowed = self.model.computeKernel(20, 1000.3, halfWidth=20, normalize=False)
        kernel = self.model.computeKernel(20, 1000.3, halfWidth=10, normalize=False)
        expect = np.ones(21)
        expect[0] = 0.2
        expect[-1] = 0.8
        core = unwindowed.values[kernel.start - unwindowed.start:kernel.stop - unwindowed.start]
        np.testing.assert_allclose(kernel.values, core*expect, rtol=1.0e-12, atol=0.0)
        # An integer centre with an integer halfWidth has half-weighted end pixels
        kernel = self.model.computeKernel(20, 1000.0, halfWidth=10, normalize=False)
        unwindowed = self.model.computeKernel(20, 1000.0, halfWidth=20, normalize=False)
        self.assertEqual(kernel.slice, slice(990, 1011))
        expect = np.ones(21)
        expect[[0, -1]] = 0.5
        np.testing.assert_allclose(kernel.values, unwindowed.values[10:31]*expect, rtol=1.0e-12, atol=0.0)

    def testEdge(self):
        """A kernel at the end of the spectrum is clipped"""
        kernel = self.model.computeKernel(20, 2.0, halfWidth=10)
        self.assertEqual(kernel.start, 0)
        self.assertEqual(kernel.stop, 13)
        unwindowed = self.model.computeKernel(20, 2.0, halfWidth=20, normalize=False)
        expect = unwindowed.values[:13]*np.append(np.ones(12), 0.5)   # the end pixel is half-weighted
        np.testing.assert_allclose(kernel.values*kernel.norm, expect, rtol=1.0e-12, atol=0.0)
        self.assertLess(kernel.norm, 1.0)
        kernel = self.model.computeKernel(20, self.model.length - 1.5, halfWidth=10)
        self.assertEqual(kernel.stop, self.model.length)
        with self.assertRaises(ValueError):
            self.model.computeKernel(20, -50.0, halfWidth=10)
        with self.assertRaises(KeyError):
            self.model.computeKernel(21, 1000.0)

    def testExtract(self):
        """The kernel extracts the matching slice of a spectrum"""
        length = self.model.length
        flux = np.arange(3*length, dtype=float).reshape(3, length)
        spectra = types.SimpleNamespace(fiberId=self.model.fiberId, flux=flux)
        kernel = self.model.computeKernel(20, 1234.5)
        self.assertFloatsEqual(kernel.extract(flux[1]), flux[1, kernel.start:kernel.stop])
        self.assertFloatsEqual(kernel.extractFrom(spectra, "flux"), flux[1, kernel.start:kernel.stop])

    def testWavelength(self):
        """A kernel at a wavelength has its centre there, and its pixels' wavelengths"""
        rows = np.arange(self.model.length, dtype=float)
        wavelength = 600 + 0.05*rows + 1e-6*rows**2
        kernel = self.model.computeKernelAtWavelength(20, 700.0)
        self.assertFloatsAlmostEqual(np.interp(kernel.center, rows, wavelength), 700.0, atol=1.0e-6)
        self.assertFloatsAlmostEqual(kernel.wavelength, wavelength[kernel.slice], atol=1.0e-6)
        shifted = wavelength + 0.01
        kernel = self.model.computeKernelAtWavelength(20, 700.0, wavelengthArray=shifted)
        self.assertFloatsAlmostEqual(np.interp(kernel.center, rows, shifted), 700.0, atol=1.0e-9)
        self.assertFloatsEqual(kernel.wavelength, shifted[kernel.slice])

    def testKernels(self):
        """The stacked kernels match the individual ones"""
        centers = np.array([500.2, 1500.7, 3.0])
        kernels = self.model.computeKernels(30, centers, halfWidth=8)
        self.assertEqual(kernels.values.shape, (3, 18))
        self.assertFloatsAlmostEqual(kernels.values.sum(axis=1), 1.0, atol=1.0e-12)
        for ii, cc in enumerate(centers):
            single = self.model.computeKernel(30, cc, halfWidth=8)
            stacked = kernels[ii]
            self.assertFloatsEqual(stacked.values[single.start - stacked.start:single.stop - stacked.start],
                                   single.values)
            self.assertEqual(stacked.values.sum(), single.values.sum())
        self.assertTrue(np.all(kernels.values[2, :5] == 0))   # off the start of the row

    def testParameters(self):
        """getParameters returns the surfaces' values"""
        params = self.model.getParameters([10, 20], [100.0, 200.0])
        self.assertFloatsAlmostEqual(params["radius"], 1.8, atol=1.0e-12)
        self.assertFloatsAlmostEqual(params["K"], 0.5, atol=1.0e-12)
        self.assertFloatsAlmostEqual(params["x"], [1001.0, 2002.0], atol=1.0e-6)
        self.assertFloatsAlmostEqual(params["sigma"], diskSigma(1.8, 0.65), atol=1.0e-12)
        self.assertGreater(self.model.computeFwhm(20, 1000.0), 2*diskSigma(1.8, 0.65))

    def testLorentz(self):
        """A model with a Lorentzian has broader wings"""
        model = makeModel()
        model.surfaces["lorentz"] = constantSurface("lorentz", 0.12)
        self.assertIn("lorentz", model.parameterNames)
        self.assertFloatsAlmostEqual(model.getParameters(20, 1000.0)["lorentz"], 0.12, atol=1.0e-12)
        kernel = model.computeKernel(20, 1000.3, halfWidth=10)
        core = self.model.computeKernel(20, 1000.3, halfWidth=10)
        self.assertLess(kernel.norm, core.norm)
        # The window is symmetric about the centre, so the truncated wings hardly shift the first moment
        self.assertFloatsAlmostEqual(np.sum(kernel.pixels*kernel.values), 1000.3, atol=2.0e-4)
        self.assertGreater(kernel.values[0], core.values[0])
        self.assertGreater(model.computeFwhm(20, 1000.0), self.model.computeFwhm(20, 1000.0))

    def testEngineeringFibres(self):
        """Engineering fibres get the science fibres' radius times iisRadiusScale; nothing else changes"""
        model = makeModel()
        model.engineeringFiberId = np.array([20])
        model.iisRadiusScale = constantSurface("iisRadiusScale", 0.5)
        eng = model.getParameters(20, 1000.0)
        sci = model.getParameters(10, 1000.0)
        self.assertFloatsAlmostEqual(eng["radius"], 0.5*sci["radius"], rtol=1.0e-12)
        self.assertFloatsAlmostEqual(eng["rms"], sci["rms"], rtol=1.0e-12)
        both = model.getParameters([10, 20], [1000.0, 1000.0])
        self.assertFloatsAlmostEqual(both["radius"], [sci["radius"], eng["radius"]], rtol=1.0e-12)
        self.assertLess(model.computeFwhm(20, 1000.0), model.computeFwhm(10, 1000.0))
        copy = pickle.loads(pickle.dumps(model))
        self.assertFloatsEqual(copy.computeKernel(20, 1000.3).values, model.computeKernel(20, 1000.3).values)

    def testModelHalo(self):
        """A model with a halo puts more light in the kernel's wings, keeping its first moment"""
        model = makeModel()
        model.surfaces["haloFrac"] = constantSurface("haloFrac", 0.03)
        model.surfaces["haloWidth"] = constantSurface("haloWidth", 2.2)
        params = model.getParameters(20, 1000.0)
        self.assertFloatsAlmostEqual(params["haloFrac"], 0.03, atol=1.0e-12)
        kernel = model.computeKernel(20, 1000.3, halfWidth=10)
        core = self.model.computeKernel(20, 1000.3, halfWidth=10)
        self.assertFloatsAlmostEqual(np.sum(kernel.pixels*kernel.values), 1000.3, atol=1.0e-4)
        self.assertGreater(kernel.values[2], core.values[2])
        copy = pickle.loads(pickle.dumps(model))
        self.assertFloatsEqual(copy.computeKernel(20, 1000.3).values, model.computeKernel(20, 1000.3).values)

    def testPickle(self):
        """The model can be pickled"""
        copy = pickle.loads(pickle.dumps(self.model))
        self.assertFloatsEqual(copy.computeKernel(20, 1000.3).values,
                               self.model.computeKernel(20, 1000.3).values)
        self.assertFloatsEqual(copy.wavelength(20, [1.0, 2.5]), self.model.wavelength(20, [1.0, 2.5]))


class FitLsfLinesTestCase(lsst.utils.tests.TestCase):
    """Tests for pfs.drp.stella.fitLsfLines"""
    def testFitDiskLine(self):
        """We recover the parameters of a simulated line"""
        rng = np.random.default_rng(12345)
        length = 200
        pix = np.arange(length, dtype=float)
        wavelength = 700 + 0.07*pix
        radius, rms, S, K = 1.8, 0.65, 0.3, 0.8
        # The lines are 0.3 pixels from their catalogue positions (an error in the wavelength solution)
        center, flux = 100.3, 1.0e5
        companion = 105.1                                  # catalogue position
        model = (flux*diskProfile(pix, center, radius, rms, [S, K])
                 + 0.1*flux*diskProfile(pix, companion + 0.3, radius, rms, [S, K]) + 50.0 + 0.1*pix)
        variance = model + 10.0**2
        data = model + rng.normal(size=length)*np.sqrt(variance)
        fit = fitDiskLine(wavelength, data, variance, 700 + 0.07*100.0, blurOrder=4,
                          companions=[700 + 0.07*companion])
        self.assertTrue(fit.success)
        for value, error, truth in [(fit.pixel, fit.pixelErr, center), (fit.radius, fit.radiusErr, radius),
                                    (fit.rms, fit.rmsErr, rms), (fit.cumulants[0], fit.cumulantsErr[0], S),
                                    (fit.cumulants[1], fit.cumulantsErr[1], K),
                                    (fit.amplitude, fit.amplitudeErr, flux)]:
            self.assertLess(abs(value - truth), 5*error)
        self.assertLess(fit.chi2/fit.ndof, 2.0)

        pinned = fitDiskLine(wavelength, data, variance, 700 + 0.07*100.0, blurOrder=4,
                             companions=[700 + 0.07*companion], fixCentre=center + 0.01)
        self.assertTrue(pinned.pinned)
        self.assertEqual(fit.lorentz, 0.0)
        self.assertEqual(pinned.pixel, center + 0.01)
        self.assertEqual(pinned.pixelErr, 0.0)

    def testFitLorentz(self):
        """We recover the Lorentzian of a simulated line"""
        rng = np.random.default_rng(2468)
        length = 200
        pix = np.arange(length, dtype=float)
        wavelength = 700 + 0.07*pix
        radius, rms, S, K, lorentz = 1.8, 0.65, 0.3, 0.8, 0.15
        center, flux = 100.3, 3.0e5
        model = flux*diskProfile(pix, center, radius, rms, [S, K], lorentz=lorentz) + 50.0
        variance = model + 10.0**2
        data = model + rng.normal(size=length)*np.sqrt(variance)
        fit = fitDiskLine(wavelength, data, variance, 700 + 0.07*100.0, blurOrder=4, fitLorentz=True)
        self.assertTrue(fit.success)
        self.assertLess(abs(fit.lorentz - lorentz), 5*fit.lorentzErr)
        self.assertLess(fit.lorentzErr, 0.03)
        self.assertLess(abs(fit.pixel - center), 5*fit.pixelErr)

    def testFitHalo(self):
        """We recover the halo of a simulated line"""
        rng = np.random.default_rng(97531)
        length = 200
        pix = np.arange(length, dtype=float)
        wavelength = 700 + 0.07*pix
        radius, rms, S, K, haloFrac, haloWidth = 1.8, 0.65, 0.3, 0.8, 0.04, 2.3
        center, flux = 100.3, 1.0e6
        model = 50.0 + flux*diskProfile(pix, center, radius, rms, [S, K], haloFrac=haloFrac,
                                        haloWidth=haloWidth)
        variance = model + 10.0**2
        data = model + rng.normal(size=length)*np.sqrt(variance)
        fit = fitDiskLine(wavelength, data, variance, 700 + 0.07*100.0, blurOrder=4, fitHalo=True,
                          halfWindow=16)
        self.assertTrue(fit.success)
        self.assertLess(abs(fit.haloFrac - haloFrac), 5*fit.haloFracErr)
        self.assertLess(abs(fit.haloWidth - haloWidth), 5*fit.haloWidthErr)
        self.assertLess(fit.chi2/fit.ndof, 2.0)

    def testFixShape(self):
        """With the blur kernel held fixed, the radius of a narrower disk is recovered"""
        rng = np.random.default_rng(8642)
        length = 200
        pix = np.arange(length, dtype=float)
        wavelength = 700 + 0.07*pix
        radius, rms, S, K = 0.95, 0.65, 0.3, 0.8                # a small (IIS-like) fibre image
        center, flux = 100.3, 3.0e5
        model = 50.0 + flux*diskProfile(pix, center, radius, rms, [S, K])
        variance = model + 10.0**2
        data = model + rng.normal(size=length)*np.sqrt(variance)
        fit = fitDiskLine(wavelength, data, variance, 700 + 0.07*100.0, blurOrder=4, fixShape=(rms, [S, K]))
        self.assertTrue(fit.success)
        self.assertEqual(fit.rms, rms)
        self.assertFloatsAlmostEqual(fit.cumulants, [S, K], atol=1.0e-12)
        self.assertEqual(fit.rmsErr, 0.0)
        self.assertLess(abs(fit.radius - radius), 5*fit.radiusErr)
        self.assertLess(abs(fit.pixel - center), 5*fit.pixelErr)

    def testFixCentre(self):
        """A fixed centre is reported with zero error, and the other parameters are still fit"""
        rng = np.random.default_rng(1357)
        pix = np.arange(200, dtype=float)
        wavelength = 700 + 0.07*pix
        model = 50.0 + 1.0e5*diskProfile(pix, 100.3, 1.8, 0.65, [0.3, 0.8])
        data = model + rng.normal(size=200)*np.sqrt(model + 100.0)
        fit = fitDiskLine(wavelength, data, model + 100.0, 707.0, blurOrder=4, fixCentre=100.3)
        self.assertTrue(fit.success and fit.pinned)
        self.assertEqual(fit.pixel, 100.3)
        self.assertEqual(fit.pixelErr, 0.0)
        self.assertLess(abs(fit.radius - 1.8), 5*fit.radiusErr)


class SmoothSurfaceTestCase(lsst.utils.tests.TestCase):
    """Tests for pfs.drp.stella.smoothSurface"""
    def testFit(self):
        """We recover a surface and per-group offsets"""
        rng = np.random.default_rng(54321)
        num = 2000
        u = rng.uniform(0, 4000, num)
        v = rng.uniform(0, 4000, num)
        group = np.where(u < 2000, "a", "b")
        truth = 1.0 + 1.0e-4*u - 2.0e-5*v + 0.05*(group == "b")
        value = truth + rng.normal(0, 0.01, num)
        value[:5] += 1.0                                   # outliers
        surface, used, residual = fitSmoothSurface(u, v, value, np.full(num, 0.01), 1,
                                                   domain=((0, 4000), (0, 4000)), groups=dict(element=group))
        self.assertFalse(np.any(used[:5]))
        self.assertEqual(surface.reference["element"], "b" if np.sum(group == "b") > num/2 else "a")
        model = surface.evaluate(u, v, element=group)
        self.assertLess(np.std(model - truth), 0.002)
        self.assertFloatsAlmostEqual(residual, value - model, atol=1.0e-10)
        self.assertFloatsAlmostEqual(surface.chi2/surface.ndof, 1.0, atol=0.15)

    def testSupportedDegree(self):
        """The coverage guard lowers the degree where there are large gaps"""
        domain = (0.0, 4000.0)
        even = np.linspace(100, 3900, 40)
        self.assertEqual(supportedDegree(even, domain, 4), 4)
        clustered = np.concatenate([np.linspace(3000, 3900, 30), [800, 1400, 1900]])   # a gap of 1100
        self.assertEqual(supportedDegree(clustered, domain, 4), 2)            # 4000/(2 + 1) > 1100 > 4000/4
        self.assertEqual(supportedDegree(clustered, domain, 4, factor=1.5), 4)
        self.assertEqual(supportedDegree([3000.0], domain, 3), 0)

    def testCoords(self):
        """DetectorCoords: identity for a monolithic detector; affine on the right CCD"""
        identity = DetectorCoords(2048.0)
        self.assertFalse(identity.isDivided)
        self.assertFloatsEqual(np.array(identity([10.0, 3000.0], [5.0, 6.0])), [[10.0, 3000.0], [5.0, 6.0]])
        affine = np.array([[1.0, 0.0], [0.0, 1.0], [70.0, 1.0]])
        coords = DetectorCoords(2048.0, affine)
        self.assertTrue(coords.isDivided)
        self.assertFloatsEqual(np.array(coords([10.0, 3000.0], [5.0, 6.0])), [[10.0, 3070.0], [5.0, 7.0]])
        copy = pickle.loads(pickle.dumps(coords))
        self.assertFloatsEqual(copy.affine, affine)


class LsfLineSetTestCase(lsst.utils.tests.TestCase):
    """Tests for pfs.drp.stella.LsfLineSet"""
    def testFits(self):
        """FITS I/O round trip"""
        num = 5
        columns = {name: np.zeros(num, dtype=column.dtype if column.dtype is not str else object)
                   for name, column in LsfLineSet.DamdClass.getSchemaDict().items()}
        columns["fiberId"] = np.arange(num, dtype=np.int32)
        columns["radius"] = np.linspace(1.5, 2.0, num).astype(np.float32)
        columns["description"] = np.array(["NeI"]*num, dtype=object)
        columns["transition"] = np.array(["2p->1s"]*num, dtype=object)
        columns["flag"] = np.array([0, LsfLineFlag.BROAD, LsfLineFlag.PINNED, 0, LsfLineFlag.SATURATED],
                                   dtype=np.int32)
        lines = LsfLineSet.fromColumns(**columns)
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "lsfLines.fits")
            lines.writeFits(filename)
            copy = LsfLineSet.readFits(filename)
        self.assertEqual(len(copy), num)
        self.assertFloatsEqual(copy.radius, lines.radius)
        self.assertFloatsEqual(copy.flag, lines.flag)
        self.assertEqual(list(copy.description), ["NeI"]*num)
        self.assertEqual(LsfLineFlag.rejected("rms"), LsfLineFlag.REJECTED_RMS)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    from argparse import ArgumentParser
    parser = ArgumentParser(__file__)
    parser.add_argument("--display", help="Display to use")
    args = parser.parse_args()
    display = args.display
    unittest.main(failfast=True, argv=[__file__])
