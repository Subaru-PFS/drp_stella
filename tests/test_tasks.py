import sys
import unittest

import numpy as np

import lsst.utils.tests
import lsst.afw.image
import lsst.afw.image.testUtils
import lsst.afw.display
from lsst.ip.isr import IsrTask
from lsst.pipe.base import Struct

import pfs.drp.stella.synthetic

from pfs.drp.stella.findAndTraceAperturesTask import FindAndTraceAperturesTask
from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask
from pfs.drp.stella.calibrateWavelengthsTask import CalibrateWavelengthsTask
from pfs.drp.stella.reduceExposure import ReduceExposureTask

display = None


class DummyDataRef:
    """Quacks like a ButlerDataRef

    The outputs aren't available as inputs, but that doesn't matter for our
    purposes.

    Parameters
    ----------
    **kwargs
        Datasets to serve as inputs.
    """
    def __init__(self, **kwargs):
        self._inputs = kwargs
        self._outputs = {}
        self.dataId = dict(name="dummy")

    def get(self, name, **kwargs):
        """Get dataset from storage"""
        if name in self._inputs:
            return self._inputs[name]
        raise RuntimeError("Unknown dataset: {}".format(name))

    def put(self, obj, name, **kwargs):
        """Put dataset to storage"""
        self._outputs[name] = obj

    def getPutObjects(self):
        """Return a `dict` of objects that have been ``put``"""
        return self._outputs


class DummyIsrTask(IsrTask):
    """``IsrTask`` that doesn't really do ISR"""
    def runDataRef(self, dataRef):
        return Struct(exposure=dataRef.get("raw"))


class TasksTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        """Create a synthetic flat, arc and detector map"""
        self.rng = np.random.RandomState(98765)
        self.synthConfig = pfs.drp.stella.synthetic.SyntheticConfig()
        self.detMap = pfs.drp.stella.synthetic.makeSyntheticDetectorMap(self.synthConfig)
        self.flux = 1.0e5
        flat = pfs.drp.stella.synthetic.makeSyntheticFlat(self.synthConfig, flux=self.flux,
                                                          addNoise=False, rng=self.rng)
        self.flat = lsst.afw.image.makeMaskedImage(flat)
        self.flat.mask.array[:] = 0x0
        self.flat.variance.array[:] = self.synthConfig.readnoise**2

        self.numLines = 50
        self.lineFwhm = 4.321
        self.arcData = pfs.drp.stella.synthetic.makeSyntheticArc(self.synthConfig, flux=self.flux,
                                                                 fwhm=self.lineFwhm, addNoise=False,
                                                                 rng=self.rng)
        self.continuumFactor = 0.01
        self.arc = lsst.afw.image.makeMaskedImage(self.arcData.image)
        self.arc.image.array += self.flat.image.array*self.continuumFactor
        self.arc.mask.array[:] = 0x0
        self.arc.variance.array[:] = self.synthConfig.readnoise**2

        self.lines = []
        middle = self.detMap.fiberIds[self.synthConfig.numFibers//2]
        for center in self.arcData.lines:
            wavelength = self.detMap.findWavelength(middle, center)
            ll = pfs.drp.stella.ReferenceLine("arc", wavelength=wavelength, guessedIntensity=self.flux)
            ll.guessedPosition = center + (2*self.rng.uniform() - 1)*2.0
            self.lines.append(ll)

    def tearDown(self):
        del self.flat

    def makeFiberTraces(self):
        """Construct fiber traces

        We do this using ``FindAndTraceAperturesTask`` on the flat.
        """
        config = FindAndTraceAperturesTask.ConfigClass()
        config.finding.minLength = self.synthConfig.height//2
        task = FindAndTraceAperturesTask(config=config)
        return task.run(self.flat, self.detMap)

    def assertSpectra(self, spectra, hasContinuum=True):
        """Assert that the extracted arc spectra are as expected"""
        self.assertEqual(len(spectra), self.synthConfig.numFibers)
        continuum = self.flux*self.continuumFactor
        expectSpectrum = self.arcData.spectrum + continuum
        for spectrum in spectra:
            self.assertEqual(len(spectrum.referenceLines), self.numLines)
            self.assertFloatsEqual(np.array([int(ll.status) for ll in spectrum.referenceLines]),
                                   np.ones(self.numLines, dtype=int)*int(pfs.drp.stella.ReferenceLine.FIT))
            self.assertFloatsAlmostEqual(np.array([ll.fitPosition for ll in spectrum.referenceLines]),
                                         self.arcData.lines, atol=2.0e-4)
            sigma = self.lineFwhm/(2*np.sqrt(2*np.log(2)))
            norm = 1.0/(sigma*np.sqrt(2*np.pi))
            expectIntensity = norm*self.flux
            self.assertFloatsAlmostEqual(np.array([ll.fitIntensity for ll in spectrum.referenceLines]),
                                         np.ones(self.numLines)*expectIntensity, rtol=3.0e-3)
            # Add continuum back in, so the rtol can be relative to something non-zero
            values = spectrum.spectrum if hasContinuum else spectrum.spectrum + continuum
            self.assertFloatsAlmostEqual(values, expectSpectrum, rtol=3.0e-3)

    def testBasic(self):
        """Test tasks

        We test, in series:
        * FindAndTraceAperturesTask
        * ExtractSpectraTask (which also runs IdentifyLinesTask)
        * CalibrateWavelengthsTask
        """
        if display:
            lsst.afw.display.Display(1, display).mtv(self.flat)
            lsst.afw.display.Display(2, display).mtv(self.arc)

        traces = self.makeFiberTraces()
        self.assertEqual(len(traces), self.synthConfig.numFibers)
        self.assertFloatsEqual([tt.fiberId for tt in traces], self.detMap.fiberIds)

        # Extract traces on flat
        config = ExtractSpectraTask.ConfigClass()
        task = ExtractSpectraTask(config=config)
        spectra = task.run(self.flat, traces, self.detMap).spectra
        for spectrum in spectra:
            self.assertFloatsAlmostEqual(spectrum.spectrum, self.flux, rtol=3.0e-3)

        # Extract arc and identify lines
        spectra = task.run(self.arc, traces, self.detMap, self.lines).spectra
        self.assertSpectra(spectra)

        # Wavelength calibration
        config = CalibrateWavelengthsTask.ConfigClass()
        task = CalibrateWavelengthsTask(config=config)
        solutions = task.run(spectra, self.detMap, seed=12345).solutions
        for tt, ss, sol in zip(traces, spectra, solutions):
            self.assertFloatsAlmostEqual(sol(np.arange(len(ss))), np.zeros(len(ss)), atol=3.0e-5)
            self.assertFloatsAlmostEqual(ss.wavelength, self.detMap.getWavelength(tt.fiberId), atol=1.0e-4)

    def testReduceExposure(self):
        """Test ReduceExposureTask"""
        traces = self.makeFiberTraces()

        config = ReduceExposureTask.ConfigClass()
        config.isr.retarget(DummyIsrTask)
        config.doRepair = False
        config.doSubtractContinuum = True
        config.doWriteCalexp = True
        task = ReduceExposureTask(config=config)

        raw = lsst.afw.image.makeExposure(self.arc)
        dataRef = DummyDataRef(raw=raw, fibertrace=traces, detectormap=self.detMap)
        results = task.run(dataRef, lines=self.lines)
        self.assertMaskedImagesEqual(results.exposure.maskedImage, self.arc)
        self.assertSpectra(results.original)
        self.assertSpectra(results.spectra, False)
        self.assertTrue(results.detectorMap is not None)

        putted = dataRef.getPutObjects()
        self.assertMaskedImagesEqual(putted["calexp"].maskedImage, self.arc)
        self.assertSpectra(putted["pfsArm"], False)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    from argparse import ArgumentParser
    parser = ArgumentParser(__file__)
    parser.add_argument("--display", help="Display backend")
    parser.add_argument("--log", action="store_true", help="Activate verbose logging")
    args, argv = parser.parse_known_args()
    display = args.display
    if args.log:
        import lsst.log  # noqa
        lsst.log.setLevel("pfs.drp.stella.FiberTrace", lsst.log.TRACE)
    unittest.main(failfast=True, argv=[__file__] + argv)
