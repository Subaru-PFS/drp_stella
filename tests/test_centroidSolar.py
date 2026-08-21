import copy
import os
import tempfile

import astropy.io.fits
import numpy as np

import lsst.utils.tests
from lsst.afw.image import VisitInfo
from lsst.geom import SpherePoint, degrees

from pfs.datamodel import Identity, MaskHelper, Target, TargetType
from pfs.datamodel.pfsSimpleSpectrum import PfsSimpleSpectrum
from pfs.drp.stella.centroidSolar import CentroidSolarConfig, CentroidSolarTask
from pfs.drp.stella.datamodel import PfsArm
from pfs.drp.stella.lsf import GaussianLsf, LsfDict
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap, makeSyntheticPfsConfig
from pfs.drp.stella.tests import runTests

display = None


def makeAbsorptionFlux(wavelength, centers, depth, sigma):
    """Continuum of unity with narrow Gaussian absorption dips

    Parameters
    ----------
    wavelength : `numpy.ndarray` of `float`
        Wavelength array (nm).
    centers : iterable of `float`
        Central wavelengths of the absorption lines (nm).
    depth : `float`
        Depth of each absorption line, as a fraction of the continuum.
    sigma : `float`
        Width of each absorption line (nm).

    Returns
    -------
    flux : `numpy.ndarray` of `float`
        Flux array, with the same shape as ``wavelength``.
    """
    flux = np.ones_like(wavelength)
    for center in centers:
        flux *= 1.0 - depth*np.exp(-0.5*((wavelength - center)/sigma)**2)
    return flux


def makeEmissionFlux(wavelength, centers, depth, sigma):
    """Continuum of unity with narrow Gaussian emission bumps

    Parameters
    ----------
    wavelength : `numpy.ndarray` of `float`
        Wavelength array (nm).
    centers : iterable of `float`
        Central wavelengths of the emission lines (nm).
    depth : `float`
        Height of each emission line, as a fraction of the continuum.
    sigma : `float`
        Width of each emission line (nm).

    Returns
    -------
    flux : `numpy.ndarray` of `float`
        Flux array, with the same shape as ``wavelength``.
    """
    flux = np.ones_like(wavelength)
    for center in centers:
        flux += depth*np.exp(-0.5*((wavelength - center)/sigma)**2)
    return flux


class CentroidSolarTestCase(lsst.utils.tests.TestCase):
    """Test that `CentroidSolarTask` recovers injected wavelength shifts"""

    def setUp(self):
        self.rng = np.random.RandomState(12345)

        lineSigma = 4.321  # nm
        density = 0.5  # Average number of lines per lineSigma
        depth = 0.4  # Maximum depth of the absorption lines (fraction of continuum)
        minWavelength = 400.0  # nm
        maxWavelength = 600.0  # nm
        buffer = 10.0  # nm
        templateResolution = 0.01  # nm

        self.synthConfig = SyntheticConfig()
        self.synthConfig.height = 512  # Fewer rows than the default: keeps the test fast
        self.pfsDesignId = 0x1234
        self.visit = 98765
        self.arm = "r"
        self.spectrograph = 1

        self.pfsConfig = makeSyntheticPfsConfig(
            self.synthConfig, self.pfsDesignId, self.visit, rng=self.rng, fracSky=0.5, fracFluxStd=0.0
        )
        self.detectorMap = makeSyntheticDetectorMap(
            self.synthConfig, minWl=minWavelength, maxWl=maxWavelength
        )
        # Dispersion is about 0.39 nm/pixel

        self.centers = [450.0, 500.0, 550.0]
        self.halfWidth = 25.0
        self.shiftsMinMax = 0.321  # nm

        self.centroidSolarConfig = CentroidSolarConfig()
        self.centroidSolarConfig.wavelengths = self.centers
        self.centroidSolarConfig.halfWidth = [self.halfWidth]*len(self.centers)
        self.centroidSolarConfig.maxOffset = 0.7
        self.centroidSolarConfig.priorOffset = 0.5
        # Atmosphere fitting is exercised separately, with its own visitInfo, in testFitAtmosphere.
        self.centroidSolarConfig.doFitAtmosphere = False

        # Ensure mask names are set up
        maskFlags = MaskHelper()
        for name in self.centroidSolarConfig.mask:
            maskFlags.add(name)
        for name in self.centroidSolarConfig.targetMask:
            maskFlags.add(name)
        for name in self.centroidSolarConfig.skyMask:
            maskFlags.add(name)

        self.skyFiberId = self.pfsConfig.fiberId[self.pfsConfig.targetType == int(TargetType.SKY)]
        self.assertGreater(len(self.skyFiberId), 1)

        # A finely-sampled template with absorption dips at the line centers, well outside the
        # range we'll fit so the shifted template never runs off the end during interpolation.
        templateWavelength = np.arange(minWavelength - buffer, maxWavelength + buffer, templateResolution)
        numLines = (maxWavelength - minWavelength)/lineSigma*density
        centers = self.rng.uniform(minWavelength - buffer, maxWavelength + buffer, size=int(numLines))
        # Saved for reuse by tests (e.g. testFitAtmosphere) that need to synthesize an observed
        # spectrum whose solar-absorption component actually matches this template's lines --
        # as opposed to self.centers, which is the (unrelated) list of fitting-window centers.
        self.templateCenters = centers
        self.templateDepth = depth
        self.templateLineSigma = lineSigma
        templateFlux = makeAbsorptionFlux(templateWavelength, centers, depth, lineSigma)
        templateMask = np.zeros(templateWavelength.shape, dtype=np.int32)
        template = PfsSimpleSpectrum(
            Target(0, 0, "0,0", 0, ra=0.0, dec=0.0),
            templateWavelength,
            templateFlux,
            templateMask,
            maskFlags,
        )
        templateDir = tempfile.mkdtemp()
        self.templatePath = os.path.join(templateDir, "solar_spectrum.fits")
        template.writeFits(self.templatePath)
        self.addCleanup(os.remove, self.templatePath)
        self.centroidSolarConfig.targetTemplate = self.templatePath

        # A sky template with narrow emission bumps (unlike the target template's absorption dips,
        # so the two templates' flux-ratio bases aren't degenerate/collinear). This test doesn't
        # inject any sky emission lines into the observed spectrum, so the sky component should
        # just fit to ~zero amplitude and not affect the recovered offset.
        skyLineSigma = 1.234  # nm: narrower than the target's lines, so the shapes are distinct
        skyDepth = 0.4
        skyCenters = self.rng.uniform(minWavelength - buffer, maxWavelength + buffer, size=int(numLines))
        skyTemplateFlux = makeEmissionFlux(templateWavelength, skyCenters, skyDepth, skyLineSigma)
        skyMask = np.zeros(templateWavelength.shape, dtype=np.int32)
        skyTemplate = PfsSimpleSpectrum(
            Target(0, 0, "0,0", 0, ra=0.0, dec=0.0),
            templateWavelength,
            skyTemplateFlux,
            skyMask,
            maskFlags,
        )
        self.skyTemplatePath = os.path.join(templateDir, "sky_spectrum.fits")
        skyTemplate.writeFits(self.skyTemplatePath)
        self.addCleanup(os.remove, self.skyTemplatePath)
        self.centroidSolarConfig.skyTemplate = self.skyTemplatePath

        # A distinct injected wavelength shift for each sky fiber (nm).
        self.shifts = dict(zip(
            self.skyFiberId, np.linspace(-self.shiftsMinMax, self.shiftsMinMax, len(self.skyFiberId))
        ))

        numFibers = len(self.pfsConfig)
        length = self.synthConfig.height
        identity = Identity(self.visit, self.arm, self.spectrograph, self.pfsDesignId)
        wavelength = np.vstack([self.detectorMap.getWavelength(ff) for ff in self.pfsConfig.fiberId])
        variance = np.full((numFibers, length), 1.0e-4, dtype=np.float32)
        flux = np.empty((numFibers, length), dtype=np.float32)
        for ii, ff in enumerate(self.pfsConfig.fiberId):
            wl = wavelength[ii]
            shift = self.shifts.get(ff, 0.0)
            spectrumFlux = makeAbsorptionFlux(wl - shift, centers, depth, lineSigma)

            # Perturb the continuum with a smooth polynomial (e.g., as from imperfect flux calibration).
            # Keep the perturbation modest so the continuum stays well clear of zero everywhere in the
            # fitting range: a large coefficient on the quadratic term can otherwise drive chebval to
            # ~zero (or negative) in the middle of the range, destroying the SNR of the affected windows.
            norm = 2*(wl - wl.min())/(wl.max() - wl.min()) - 1
            coeff = np.array([1.0, 0.0, 0.0]) + self.rng.uniform(-0.1, 0.1, size=3)
            spectrumFlux = spectrumFlux*np.polynomial.chebyshev.chebval(norm, coeff)

            spectrumFlux += self.rng.normal(0.0, np.sqrt(variance[ii]), size=length)
            flux[ii] = spectrumFlux

        mask = np.zeros((numFibers, length), dtype=np.int32)
        sky = np.zeros((numFibers, length), dtype=np.float32)
        norm = np.ones((numFibers, length), dtype=np.float32)
        covar = np.zeros((numFibers, 3, length), dtype=np.float32)
        covar[:, 0, :] = variance

        self.pfsArm = PfsArm(
            identity, self.pfsConfig.fiberId, wavelength, flux, mask, sky, norm, covar, maskFlags, {}
        )

        # An LSF so narrow (in detector pixels) that warping it onto the template's much finer
        # native wavelength grid still yields (to within floating-point precision) an identity
        # convolution kernel: the runtime LSF-convolution of the (now raw) target template should
        # not perturb this test's recovered offsets relative to the pre-runtime-convolution behaviour.
        self.lsf = LsfDict(
            {ff: GaussianLsf(self.synthConfig.height, 1.0e-6) for ff in self.pfsConfig.fiberId}
        )

    def testRecoverShifts(self):
        """Run CentroidSolarTask and check that the injected shifts are recovered"""
        task = CentroidSolarTask(config=self.centroidSolarConfig)
        results = task.run(self.pfsArm, self.pfsConfig, self.detectorMap, self.lsf)

        lines = results.lines
        fiberId = np.asarray(lines.fiberId)
        flag = np.asarray(lines.flag)
        offset = results.offset
        offsetErr = results.offsetErr

        template = task.loadTargetTemplate()

        for ff in self.skyFiberId:
            select = fiberId == ff
            self.assertEqual(select.sum(), len(self.centers))

            if False:
                # NOTE: enabling this block causes a test failure due to open file descriptors
                import matplotlib.pyplot as plt
                thisArm = self.pfsArm.select(fiberId=ff)
                plt.plot(thisArm.wavelength[0], thisArm.flux[0], label="Observed")
                plt.plot(template.wavelength, template.flux, label="Template")
                plt.plot(template.wavelength - self.shifts[ff], template.flux, label="Template shifted")
                for ii, (wl, color) in enumerate(zip(self.centers, ["C0", "C1", "C2"])):
                    plt.axvspan(
                        wl - self.halfWidth,
                        wl + self.halfWidth,
                        alpha=0.2,
                        color=color,
                        label=f"Region {ii}",
                    )
                plt.xlabel("Wavelength (nm)")
                plt.ylabel("Flux (arbitrary units)")
                plt.title(f"fiberId={ff}")
                plt.legend()
                print(
                    f"Showing fiberId={ff}, expected offset={self.shifts[ff]} nm, "
                    f"measured offset={offset[select]} +/- {offsetErr[select]} nm"
                )
                plt.show()

            self.assertTrue(np.all(~flag[select]), msg=f"fiberId={ff}")
            self.assertTrue(np.all(np.isfinite(offset[select])), msg=f"fiberId={ff}")
            self.assertTrue(np.all(np.isfinite(offsetErr[select])), msg=f"fiberId={ff}")
            diff = np.abs(offset[select] - self.shifts[ff])
            self.assertTrue(
                np.all(diff <= 3*offsetErr[select]),
                msg=f"fiberId={ff}: diff={diff}, offsetErr={offsetErr[select]}",
            )

    def testFitAtmosphere(self):
        """Run CentroidSolarTask with doFitAtmosphere and check PWV and shifts are recovered"""
        featureCenter = 500.0  # nm: within the halfWidth=25nm window centered there
        featureSigma = 3.0  # nm
        maxDepth = 0.5  # depth of the absorption feature at pwvGridMax
        pwvGridMax = 20.0  # mm
        pwvTrue = 8.0  # mm: the value we're trying to recover

        transmissionWavelength = np.arange(390.0, 610.0, 0.05)

        def makeTransmission(pwv):
            depth = maxDepth*pwv/pwvGridMax
            return 1.0 - depth*np.exp(-0.5*((transmissionWavelength - featureCenter)/featureSigma)**2)

        zdGrid = [0.0, 90.0]
        pwvGrid = [0.0, 5.0, 10.0, 15.0, 20.0]
        zdList = []
        pwvList = []
        transmissionList = []
        for zd in zdGrid:
            for pwv in pwvGrid:
                zdList.append(zd)
                pwvList.append(pwv)
                transmissionList.append(makeTransmission(pwv))

        atmosphereDir = tempfile.mkdtemp()
        atmospherePath = os.path.join(atmosphereDir, "pfs_atmosphere.fits")
        waveHdu = astropy.io.fits.ImageHDU(
            transmissionWavelength.astype(np.float64), name="WAVELENGTH"
        )
        columns = [
            astropy.io.fits.Column(name="zd", format="D", array=np.array(zdList, dtype=float)),
            astropy.io.fits.Column(name="pwv", format="D", array=np.array(pwvList, dtype=float)),
            astropy.io.fits.Column(
                name="transmission",
                format=f"{transmissionWavelength.size}D",
                array=np.array(transmissionList, dtype=float),
            ),
        ]
        transHdu = astropy.io.fits.BinTableHDU.from_columns(columns, name="TRANSMISSION")
        hduList = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(), waveHdu, transHdu])
        hduList.writeto(atmospherePath, overwrite=True)
        self.addCleanup(os.remove, atmospherePath)

        config = copy.deepcopy(self.centroidSolarConfig)
        config.doFitAtmosphere = True
        config.atmosphereTransmission = atmospherePath
        config.pwvMin = pwvGrid[0]
        config.pwvMax = pwvGrid[-1]

        # zd=0 exactly matches a grid point (exercises the exact-ZD-match interpolation branch).
        visitInfo = VisitInfo(boresightAzAlt=SpherePoint(0.0, 90.0, degrees))

        transmissionTrue = makeTransmission(pwvTrue)
        numFibers = len(self.pfsConfig)
        length = self.synthConfig.height
        flux = np.empty((numFibers, length), dtype=np.float32)
        wavelength = self.pfsArm.wavelength
        for ii, ff in enumerate(self.pfsConfig.fiberId):
            wl = wavelength[ii]
            shift = self.shifts.get(ff, 0.0)
            # Use the same line centers/depth/sigma that built the actual solar template file
            # (self.centers is instead the unrelated list of fitting-window centers).
            spectrumFlux = makeAbsorptionFlux(
                wl - shift, self.templateCenters, self.templateDepth, self.templateLineSigma
            )
            spectrumFlux *= np.interp(wl, transmissionWavelength, transmissionTrue)

            norm = 2*(wl - wl.min())/(wl.max() - wl.min()) - 1
            coeff = np.array([1.0, 0.0, 0.0]) + self.rng.uniform(-0.1, 0.1, size=3)
            spectrumFlux = spectrumFlux*np.polynomial.chebyshev.chebval(norm, coeff)

            variance = self.pfsArm.covar[ii, 0, :]
            spectrumFlux += self.rng.normal(0.0, np.sqrt(variance), size=length)
            flux[ii] = spectrumFlux

        pfsArm = PfsArm(
            self.pfsArm.identity,
            self.pfsConfig.fiberId,
            wavelength,
            flux,
            self.pfsArm.mask,
            self.pfsArm.sky,
            self.pfsArm.norm,
            self.pfsArm.covar,
            self.pfsArm.flags,
            {},
        )

        task = CentroidSolarTask(config=config)
        results = task.run(pfsArm, self.pfsConfig, self.detectorMap, self.lsf, visitInfo=visitInfo)

        self.assertTrue(np.isfinite(results.pwv))
        self.assertLess(abs(results.pwv - pwvTrue), 2.0, msg=f"pwv={results.pwv}, expected {pwvTrue}")

        lines = results.lines
        fiberId = np.asarray(lines.fiberId)
        flag = np.asarray(lines.flag)
        offset = results.offset
        offsetErr = results.offsetErr
        for ff in self.skyFiberId:
            select = fiberId == ff
            self.assertTrue(np.all(~flag[select]), msg=f"fiberId={ff}")
            diff = np.abs(offset[select] - self.shifts[ff])
            self.assertTrue(
                np.all(diff <= 5*offsetErr[select]),
                msg=f"fiberId={ff}: diff={diff}, offsetErr={offsetErr[select]}",
            )


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
