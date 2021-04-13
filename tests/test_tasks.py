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

from pfs.drp.stella.buildFiberProfiles import BuildFiberProfilesTask
from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask
from pfs.drp.stella.reduceExposure import ReduceExposureTask
from pfs.drp.stella.identifyLines import IdentifyLinesConfig, IdentifyLinesTask
from pfs.drp.stella.referenceLine import ReferenceLineSet, ReferenceLineStatus
from pfs.drp.stella import SpectrumSet

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
        self.dataId = dict(visit=12345, arm="r")

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
        self.pfsConfig = pfs.drp.stella.synthetic.makeSyntheticPfsConfig(self.synthConfig, 123, 456,
                                                                         rng=self.rng)

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
        middle = self.detMap.fiberId[self.synthConfig.numFibers//2]
        self.referenceLines = ReferenceLineSet.empty()
        for center in self.arcData.lines:
            wavelength = self.detMap.findWavelength(middle, center)
            self.referenceLines.append("arc", wavelength, self.flux, ReferenceLineStatus.GOOD)

    def tearDown(self):
        del self.flat

    def makeFiberProfiles(self):
        """Construct fiber profiles

        We do this using ``BuildFiberProfilesTask`` on the flat.
        """
        config = BuildFiberProfilesTask.ConfigClass()
        config.pruneMinLength = self.synthConfig.height//2
        task = BuildFiberProfilesTask(config=config)
        return task.run(lsst.afw.image.makeExposure(self.flat), self.detMap).profiles

    def assertSpectra(self, spectra, hasContinuum=True):
        """Assert that the extracted arc spectra are as expected"""
        self.assertEqual(len(spectra), self.synthConfig.numFibers)
        continuum = self.flux*self.continuumFactor
        expectSpectrum = self.arcData.spectrum + continuum
        for spectrum in spectra:
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

        profiles = self.makeFiberProfiles()
        self.assertEqual(len(profiles), self.synthConfig.numFibers)
        self.assertFloatsEqual(profiles.fiberId, sorted(self.detMap.fiberId))
        traces = profiles.makeFiberTracesFromDetectorMap(self.detMap)
        self.assertEqual(len(traces), self.synthConfig.numFibers)
        self.assertFloatsEqual(np.array(sorted([tt.fiberId for tt in traces])), sorted(self.detMap.fiberId))

        # Extract traces on flat
        config = ExtractSpectraTask.ConfigClass()
        task = ExtractSpectraTask(config=config)
        spectra = task.run(self.flat, traces, self.detMap).spectra
        for spectrum in spectra:
            self.assertFloatsAlmostEqual(spectrum.spectrum, self.flux, rtol=3.0e-3)

        # Extract arc
        spectra = task.run(self.arc, traces, self.detMap).spectra
        self.assertSpectra(spectra)

        # Identify lines
        config = IdentifyLinesConfig()
        task = IdentifyLinesTask(config=config)
        task.run(spectra, self.detMap, self.lines)
        self.assertSpectra(spectra)

    def testReduceExposure(self):
        """Test ReduceExposureTask"""
        profiles = self.makeFiberProfiles()

        config = ReduceExposureTask.ConfigClass()
        config.isr.retarget(DummyIsrTask)
        config.doRepair = False
        config.doSubtractContinuum = True
        config.doSubtractSky2d = False
        config.doWriteCalexp = True
        task = ReduceExposureTask(config=config)

        raw = lsst.afw.image.makeExposure(self.arc)
        dataRef = DummyDataRef(raw=raw, fiberProfiles=profiles, detectorMap=self.detMap,
                               pfsConfig=self.pfsConfig)
        results = task.runDataRef([dataRef])
        self.assertEqual(len(results.exposureList), 1)
        self.assertEqual(len(results.originalList), 1)
        self.assertEqual(len(results.spectraList), 1)
        self.assertEqual(len(results.fiberTraceList), 1)
        self.assertEqual(len(results.detectorMapList), 1)
        self.assertMaskedImagesEqual(results.exposureList[0].maskedImage, self.arc)
        self.assertSpectra(results.originalList[0])
        self.assertSpectra(results.spectraList[0], False)
        self.assertTrue(results.fiberTraceList[0] is not None)
        self.assertTrue(results.detectorMapList[0] is not None)

        putted = dataRef.getPutObjects()
        self.assertMaskedImagesEqual(putted["calexp"].maskedImage, self.arc)
        self.assertSpectra(SpectrumSet.fromPfsArm(putted["pfsArm"]), False)

    def testExtractSpectraFiberInclusion(self):
        """Test the feature of specifying fibers to include

        Check that fibers that weren't specified aren't included, and that
        fibers that were specified but weren't extracted are included with NANs.
        """
        numFibers = self.synthConfig.numFibers
        index = numFibers//2  # Index with the new fiberId that's not in the profiles
        newFiberId = 12345
        fiberId = np.flip(self.synthConfig.fiberId)
        fiberId[index] = newFiberId

        profiles = self.makeFiberProfiles()
        traces = profiles.makeFiberTracesFromDetectorMap(self.detMap)
        task = ExtractSpectraTask()
        spectra = task.run(self.flat, traces, self.detMap, fiberId).spectra
        self.assertEqual(len(spectra), numFibers)
        for ii, (ff, ss) in enumerate(zip(fiberId, spectra)):
            self.assertEqual(ss.fiberId, ff)
        self.assertTrue(np.all(np.isnan(spectra[index].flux)))
        self.assertTrue(np.all(np.isnan(spectra[index].background)))
        self.assertTrue(np.all(np.isnan(spectra[index].covariance)))
        self.assertTrue(np.all(np.isnan(spectra[index].wavelength)))
        self.assertFloatsEqual(spectra[index].mask.array, spectra[index].mask.getPlaneBitMask("NO_DATA"))


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
