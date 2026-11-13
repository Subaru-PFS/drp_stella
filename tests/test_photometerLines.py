import numpy as np

import lsst.utils.tests
import lsst.afw.image
import lsst.afw.image.testUtils
import lsst.pex.exceptions
from lsst.afw.detection import GaussianPsf
from pfs.drp.stella.focalPlaneFunction import FocalPlaneFunction

import pfs.drp.stella.synthetic
from pfs.drp.stella.photometerLines import PhotometerLinesTask, PhotometerLinesConfig
from pfs.drp.stella.photometry import photometer
from pfs.drp.stella import ReferenceLine, ReferenceLineSet, ReferenceLineStatus
from pfs.drp.stella import ReferenceLineSource
from pfs.drp.stella.tests.utils import runTests, methodParameters
from pfs.drp.stella.utils.psf import fwhmToSigma

display = None


class PhotometerLinesTestCase(lsst.utils.tests.TestCase):
    @methodParameters(numLines=(10, 100, 200),
                      rtol=(1.0e-2, 4.0e-3, 5.0e-3),
                      doApCorr=(True, False, False))
    def testPhotometry(self, numLines, rtol, doApCorr):
        """Test photometry on an arc

        Parameters
        ----------
        numLines : `int`
            Number of emission lines per fiber. Since the size is fixed, this
            affects the density of lines.
        rtol : `float`
            Relative tolerance for flux measurements.
        doApCorr : `bool`
            Do aperture correction?
        """
        description = "Simulated"
        flux = 123456.789
        fwhm = 3.21
        synthConfig = pfs.drp.stella.synthetic.SyntheticConfig()
        synthConfig.fwhm = fwhm
        synthConfig.height = 432
        rng = np.random.RandomState(12345)
        arc = pfs.drp.stella.synthetic.makeSyntheticArc(synthConfig, numLines=numLines,
                                                        fwhm=fwhm, flux=flux, rng=rng,
                                                        addNoise=False)
        detMap = pfs.drp.stella.synthetic.makeSyntheticDetectorMap(synthConfig)
        pfsConfig = pfs.drp.stella.synthetic.makeSyntheticPfsConfig(synthConfig, 12345, 67890, rng)

        referenceLines = []
        fiberId = detMap.fiberId[detMap.getNumFibers()//2]
        rng.shuffle(arc.lines)  # Ensure we get some cases of multiple blends
        for yy in arc.lines:
            wavelength = detMap.getWavelength(fiberId, yy)
            referenceLines.append(
                ReferenceLine(
                    description=description,
                    wavelength=wavelength,
                    intensity=flux,
                    status=ReferenceLineStatus.GOOD,
                    transition="UNKNOWN",
                    source=ReferenceLineSource.NONE,
                )
            )
        referenceLines = ReferenceLineSet.fromRows(referenceLines)

        exposure = lsst.afw.image.makeExposure(lsst.afw.image.makeMaskedImage(arc.image))
        exposure.mask.set(0)
        exposure.variance.set(synthConfig.readnoise)  # Ignore Poisson noise; not relevant

        config = PhotometerLinesConfig()
        config.fwhm = fwhm
        config.doSubtractContinuum = False  # No continuum, don't want to bother with fiberProfile creation
        radius = config.apertureCorrection.apertureFlux.radii[0]
        canApCorr = synthConfig.height/numLines > 2*radius
        assert doApCorr == canApCorr
        config.doApertureCorrection = doApCorr
        task = PhotometerLinesTask(name="photometer", config=config)
        task.log.setLevel(task.log.DEBUG)
        phot = task.run(exposure, referenceLines, detMap, pfsConfig)

        apCorr = phot.apCorr
        if doApCorr:
            self.assertIsNotNone(apCorr)
            self.assertFloatsEqual(np.sort(apCorr.fiberId), np.sort(synthConfig.fiberId))
            with lsst.utils.tests.getTempFilePath(".fits") as filename:
                apCorr.writeFits(filename)
                copy = FocalPlaneFunction.readFits(filename)
            self.assertFloatsEqual(np.sort(copy.fiberId), np.sort(synthConfig.fiberId))
            for ff in synthConfig.fiberId:
                apCorrEval = apCorr(referenceLines.wavelength, pfsConfig.select(fiberId=ff))
                copyEval = copy(referenceLines.wavelength, pfsConfig.select(fiberId=ff))
                self.assertFloatsEqual(apCorrEval.values, copyEval.values)
                self.assertFloatsEqual(apCorrEval.variances, copyEval.variances)
                np.testing.assert_array_equal(apCorrEval.masks, copyEval.masks)
        else:
            self.assertIsNone(apCorr)

        lines = phot.lines
        self.assertEqual(len(lines), detMap.getNumFibers()*len(referenceLines))
        expectFiberId = np.concatenate([synthConfig.fiberId for _ in referenceLines.wavelength])
        self.assertFloatsEqual(lines.fiberId, expectFiberId)
        expectWavelength = sum(([wl]*synthConfig.numFibers for wl in referenceLines.wavelength), [])
        self.assertFloatsEqual(lines.wavelength, expectWavelength)
        xyExpect = detMap.findPoint(lines.fiberId, lines.wavelength)
        self.assertFloatsAlmostEqual(lines.x, xyExpect[:, 0], rtol=1.0e-6)
        self.assertFloatsAlmostEqual(lines.y, xyExpect[:, 1], rtol=1.0e-6)
        self.assertTrue(np.all(np.isnan(lines.xErr)))
        self.assertTrue(np.all(np.isnan(lines.yErr)))
        sigma = fwhmToSigma(fwhm)
        expectFlux = flux*(1 - np.exp(-0.5*radius**2/sigma**2)) if doApCorr else flux
        meanFlux = lines.flux.mean()
        # Not sure why I'm not getting exactly the expected flux in the case of aperture corrections.
        # Maybe my math is wrong, or there's some other effect I'm not considering.
        self.assertFloatsAlmostEqual(meanFlux, expectFlux, rtol=rtol)
        self.assertFloatsAlmostEqual(lines.flux, meanFlux, rtol=2.0e-3)
        self.assertTrue(np.all(lines.fluxErr > 0))
        self.assertFloatsEqual(lines.flag, 0)
        self.assertFloatsEqual(lines.status, int(ReferenceLineStatus.GOOD))
        self.assertListEqual(lines.description.tolist(), [description]*len(lines))

    def testSubtractLines(self):
        """Test that subtracting lines skips lines that are off the detector"""
        flux = 123456.789
        fwhm = 3.21
        synthConfig = pfs.drp.stella.synthetic.SyntheticConfig()
        synthConfig.fwhm = fwhm
        synthConfig.height = 432
        rng = np.random.RandomState(12345)
        arc = pfs.drp.stella.synthetic.makeSyntheticArc(synthConfig, numLines=10, fwhm=fwhm, flux=flux,
                                                        rng=rng, addNoise=False)
        detMap = pfs.drp.stella.synthetic.makeSyntheticDetectorMap(synthConfig)
        pfsConfig = pfs.drp.stella.synthetic.makeSyntheticPfsConfig(synthConfig, 12345, 67890, rng)
        fiberId = detMap.fiberId[detMap.getNumFibers()//2]
        referenceLines = ReferenceLineSet.fromRows([
            ReferenceLine(
                description="Simulated",
                wavelength=detMap.getWavelength(fiberId, yy),
                intensity=flux,
                status=ReferenceLineStatus.GOOD,
                transition="UNKNOWN",
                source=ReferenceLineSource.NONE,
            ) for yy in arc.lines
        ])

        exposure = lsst.afw.image.makeExposure(lsst.afw.image.makeMaskedImage(arc.image))
        exposure.mask.set(0)
        exposure.variance.set(synthConfig.readnoise)

        config = PhotometerLinesConfig()
        config.fwhm = fwhm
        config.doSubtractContinuum = False
        config.doApertureCorrection = False
        task = PhotometerLinesTask(name="photometer", config=config)
        lines = task.run(exposure, referenceLines, detMap, pfsConfig).lines
        self.assertFloatsEqual(lines.flag, 0)

        # Move one line off the detector: it should be skipped, rather than causing an error
        lines.wavelength[0] = 1.0e4
        self.assertFalse(np.all(np.isfinite(detMap.findPoint(lines.fiberId[0], lines.wavelength[0]))))
        peak = np.max(exposure.image.array)
        numBright = np.sum(exposure.image.array > 0.5*peak)
        task.subtractLines(exposure, lines, None, pfsConfig, detMap)
        # Everything is subtracted apart from the line we moved, which remains
        remaining = np.sum(np.abs(exposure.image.array) > 0.5*peak)
        self.assertGreater(remaining, 0)
        self.assertLess(remaining, numBright/5)

    def testPhotometerRequiresDetectorMap(self):
        """Test that photometer requires a detectorMap if there are no positions"""
        image = lsst.afw.image.MaskedImageF(10, 10)
        psf = GaussianPsf(11, 11, 1.5)
        fiberId = np.array([1], dtype=np.int32)
        wavelength = np.array([500.0])
        with self.assertRaises(lsst.pex.exceptions.InvalidParameterError):
            photometer(image, fiberId, wavelength, psf, None)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
