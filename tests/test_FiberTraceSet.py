import os
import sys
import unittest
import pickle

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.daf.base
import lsst.geom
import lsst.afw.image
import lsst.afw.image.testUtils

import pfs.drp.stella as drpStella
from pfs.drp.stella.synthetic import makeSpectrumImage, SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella.findAndTraceAperturesTask import FindAndTraceAperturesTask

display = None


class FiberTraceSetTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        """Set parameters for artificial objects"""
        self.width = 11  # Width of trace
        self.height = 111  # Height of trace
        self.xy0 = lsst.geom.Point2I(123, 123)  # Origin for trace
        self.bbox = lsst.geom.Box2I(
            self.xy0, lsst.geom.Extent2I(self.width, self.height)
        )  # Bounding box for trace
        self.metadata = lsst.daf.base.PropertyList()
        self.metadata.add("METADATA", 12345)

    def assertMetadata(self, traces):
        """Assert that the metadata element of the FiberTraceSet is correct

        It's compared with the ``self.metadata``.
        """
        self.assertTrue(traces.getMetadata() is not None)
        self.assertTrue(traces.metadata is not None)
        self.assertListEqual(sorted(traces.metadata.names()), sorted(self.metadata.names()))
        for name in self.metadata.names():
            self.assertEqual(traces.metadata.get(name), self.metadata.get(name))

    def makeFiberTrace(self, fiberId):
        """Build a ``FiberTrace``

        Parameters
        ----------
        fiberId : `int`
            Fiber identifier for the trace.

        Returns
        -------
        trace : `pfs.drp.stella.FiberTrace`
            Artificial trace of a fiber.
        """
        trace = lsst.afw.image.MaskedImageF(self.width, self.height)
        trace.setXY0(self.xy0)
        rng = np.random.RandomState(12345)
        trace.image.array[:] = rng.uniform(size=trace.image.array.shape)
        trace.mask.addMaskPlane(drpStella.fiberMaskPlane)
        maskVal = trace.mask.getPlaneBitMask(drpStella.fiberMaskPlane)
        trace.mask.array[:] = maskVal
        return drpStella.FiberTrace(trace, fiberId)

    def makeFiberTraceSet(self, num):
        """Build a ``FiberTraceSet``

        Parameters
        ----------
        num : int
            Number of component ``FiberTrace``s.

        Returns
        -------
        traces : `pfs.drp.stella.FiberTraceSet`
            Set of fiber traces.
        """
        traces = drpStella.FiberTraceSet(123, self.metadata)
        for ii in range(num):
            traces.add(self.makeFiberTrace(ii))
        self.assertEqual(len(traces), num)
        return traces

    def assertFiberTraceSet(self, traces, num):
        """Assert that the ``FiberTraceSet`` is as expected

        We check the number of elements, that they're in the expected order,
        and that the metadata is as expected.

        Parameters
        ----------
        traces : `pfs.drp.stella.FiberTraceSet`
            Set of fiber traces.
        num : `int`
            Number of component fiber traces to expect.
        """
        self.assertEqual(len(traces), num)
        for ii, tt in enumerate(traces):
            self.assertEqual(tt.fiberId, ii)
        self.assertMetadata(traces)

    def testBasics(self):
        """Test construction, getters, setters, iteration"""
        traces = drpStella.FiberTraceSet(123, self.metadata)
        # Have reserved space, but empty
        self.assertEqual(len(traces), 0)
        self.assertEqual(traces.size(), 0)
        self.assertMetadata(traces)

        # Populate
        num = 5
        for ii in range(num):
            traces.add(self.makeFiberTrace(ii))
            self.assertEqual(len(traces), ii + 1)
            self.assertEqual(traces.size(), ii + 1)
        self.assertEqual(len(traces), num)
        for ii in range(num):
            self.assertEqual(traces.get(ii).fiberId, ii)
            self.assertEqual(traces[ii].fiberId, ii)
        self.assertEqual(len(traces), num)
        # Iteration
        for ii, tt in enumerate(traces):
            self.assertEqual(tt.fiberId, ii)

        # Can't get with index out of limits
        with self.assertRaises(IndexError):
            traces.get(num)
        with self.assertRaises(IndexError):
            traces[num]
        with self.assertRaises(IndexError):
            traces.get(-1)
        with self.assertRaises(IndexError):
            traces[-1]

        # Set different values
        for ii, jj in zip(range(num), reversed(range(num))):
            traces.set(ii, self.makeFiberTrace(jj))
            self.assertEqual(traces.get(ii).fiberId, jj)
            traces.set(ii, self.makeFiberTrace(ii))
            self.assertEqual(traces[ii].fiberId, ii)
        self.assertEqual(len(traces), num)

        # Can't set with index out of limits
        with self.assertRaises(IndexError):
            traces.set(num, self.makeFiberTrace(num))
        with self.assertRaises(IndexError):
            traces[num] = self.makeFiberTrace(num)
        with self.assertRaises(IndexError):
            traces.set(-1, self.makeFiberTrace(num))
        with self.assertRaises(IndexError):
            traces[-1] = self.makeFiberTrace(num)

        self.assertEqual(len(traces), num)

    def testPickle(self):
        """Test that a ``FiberTraceSet`` can be pickled"""
        num = 5
        traces = self.makeFiberTraceSet(num)
        copy = pickle.loads(pickle.dumps(traces))
        self.assertFiberTraceSet(copy, num)

    def testApplyMask(self):
        """Test application of trace mask to image"""
        traces = drpStella.FiberTraceSet(123, self.metadata)
        num = 5
        space = 10  # Space between traces
        fullWidth = num*self.width + (num - 1)*space
        fullHeight = self.height + self.xy0.getY()
        expect = lsst.afw.image.Mask(fullWidth, fullHeight)
        expect.set(0)
        for ii in range(num):
            ft = self.makeFiberTrace(ii)
            x0 = ii*(self.width + space)
            y0 = ft.trace.getY0()
            ft.trace.setXY0(lsst.geom.Point2I(x0, y0))
            traces.add(ft)
            subExpect = expect[ft.trace.getBBox(), lsst.afw.image.PARENT]
            subExpect.set(ft.trace.mask.getPlaneBitMask(drpStella.fiberMaskPlane))
        self.assertEqual(len(traces), num)

        mask = lsst.afw.image.Mask(fullWidth, fullHeight)
        traces.applyToMask(mask)
        self.assertImagesEqual(mask, expect)

    def testDatamodel(self):
        """Test conversion to `pfs.datamodel.PfsFiberTrace`"""
        num = 5
        traces = self.makeFiberTraceSet(num)

        calibDate = "2018-08-20"
        spectrograph = 3
        arm = "r"
        visit0 = 12345
        dataId = dict(calibDate=calibDate, spectrograph=spectrograph, arm=arm, visit0=visit0)

        converted = drpStella.FiberTraceSet.fromPfsFiberTrace(traces.toPfsFiberTrace(dataId))

        self.metadata.add("ARM", arm)
        self.metadata.add("SPECTROGRAPH", spectrograph)
        self.assertFiberTraceSet(converted, num)

    def testReadWriteFits(self):
        """Test reading and writing to/from FITS"""
        num = 5
        traces = self.makeFiberTraceSet(num)

        dirName = os.path.splitext(__file__)[0]
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        filename = os.path.join(dirName, "pfsFiberTrace-2018-08-20-123456-r3.fits")

        if os.path.exists(filename):
            os.unlink(filename)

        try:
            traces.writeFits(filename)
            copy = drpStella.FiberTraceSet.readFits(filename)
            self.assertFiberTraceSet(copy, num)
        except Exception:
            raise  # Leave file for manual inspection
        else:
            os.unlink(filename)

    def testExtractSpectra(self):
        """Test that simultaneous extraction of many spectra works"""
        config = SyntheticConfig()
        config.height = 256
        config.width = 128
        config.separation = 6.54321
        config.fwhm = 3.21
        config.slope = 0.0

        flux = 1.0e5
        rtol = 2.0e-3

        evenFlat = makeSpectrumImage(flux, config.dims, config.traceCenters[0::2],
                                     config.traceOffset, config.fwhm)
        oddFlat = makeSpectrumImage(flux, config.dims, config.traceCenters[1::2],
                                    config.traceOffset, config.fwhm)
        allFlat = makeSpectrumImage(flux, config.dims, config.traceCenters, config.traceOffset, config.fwhm)
        detMap = makeSyntheticDetectorMap(config)

        mask = lsst.afw.image.Mask(allFlat.getBBox())
        mask.set(0)
        variance = lsst.afw.image.ImageF(allFlat.getBBox())
        variance.set(10000.0)

        task = FindAndTraceAperturesTask()
        task.config.finding.minLength = 200
        task.config.finding.signalThreshold = 10
        evenTraces = task.run(lsst.afw.image.makeMaskedImage(evenFlat, mask, variance), detMap)
        oddTraces = task.run(lsst.afw.image.makeMaskedImage(oddFlat, mask, variance), detMap)

        self.assertEqual(len(evenTraces), len(config.traceCenters[0::2]))
        self.assertEqual(len(oddTraces), len(config.traceCenters[1::2]))

        allTraces = drpStella.FiberTraceSet.fromCombination(evenTraces, oddTraces)
        self.assertEqual(len(allTraces), config.numFibers)
        for ft in allTraces:
            ft.trace.image.array /= ft.trace.image.array.sum(axis=1)[:, np.newaxis]

        # Vanilla extraction
        spectra = allTraces.extractSpectra(lsst.afw.image.makeMaskedImage(allFlat, mask, variance))
        self.assertEqual(len(spectra), config.numFibers)
        for ii, ss in enumerate(spectra):
            self.assertFloatsAlmostEqual(ss.spectrum, flux, rtol=rtol)
            self.assertFloatsEqual(ss.mask.array, 0)
            self.assertTrue(np.all(ss.variance > 0))
            self.assertTrue(np.all(np.isfinite(ss.variance)))
            self.assertTrue(np.all(np.isfinite(ss.covariance)))
            if ii == 0:
                self.assertTrue(np.all(ss.covariance[-1] == 0))
                self.assertTrue(np.all(ss.covariance[0] != 0))
                self.assertTrue(np.all(ss.covariance[1] != 0))
            elif ii == config.numFibers - 1:
                self.assertTrue(np.all(ss.covariance[-1] != 0))
                self.assertTrue(np.all(ss.covariance[0] != 0))
                self.assertTrue(np.all(ss.covariance[1] == 0))
            else:
                self.assertTrue(np.all(ss.covariance != 0))

        # Ditto, with a row entirely bad
        # Flux should be zero in the bad row, and it should be masked
        bad = mask.getPlaneBitMask("BAD")
        noData = mask.getPlaneBitMask("NO_DATA")
        badRow = 123
        isGood = np.arange(config.height) != badRow
        mask.array[badRow, :] |= bad
        spectra = allTraces.extractSpectra(lsst.afw.image.makeMaskedImage(allFlat, mask, variance), bad)
        self.assertEqual(len(spectra), config.numFibers)
        for ii, ss in enumerate(spectra):
            self.assertFloatsAlmostEqual(ss.spectrum, np.where(np.arange(config.height) == badRow, 0.0, flux),
                                         rtol=rtol)
            self.assertFloatsEqual(ss.mask.array[0],
                                   np.where(np.arange(config.height) == badRow, bad | noData, 0))
            self.assertTrue(np.all(ss.variance[isGood] > 0))
            if ii == 0:
                self.assertTrue(np.all(ss.covariance[-1][isGood] == 0))
                self.assertTrue(np.all(ss.covariance[0][isGood] != 0))
                self.assertTrue(np.all(ss.covariance[1][isGood] != 0))
            elif ii == config.numFibers - 1:
                self.assertTrue(np.all(ss.covariance[-1][isGood] != 0))
                self.assertTrue(np.all(ss.covariance[0][isGood] != 0))
                self.assertTrue(np.all(ss.covariance[1][isGood] == 0))
            else:
                self.assertTrue(np.all(ss.covariance[:, isGood] != 0))

        # Bad column on every trace
        # Flux should still be reasonable, but everything should be masked
        mask.array[:] = 0
        for col in config.traceCenters:
            mask.array[:, int(col)] = bad
            allFlat.array[:, int(col)] = np.nan

        spectra = allTraces.extractSpectra(lsst.afw.image.makeMaskedImage(allFlat, mask, variance), bad)
        self.assertEqual(len(spectra), config.numFibers)
        for ii, ss in enumerate(spectra):
            self.assertFloatsAlmostEqual(ss.spectrum, flux, rtol=2*rtol)
            self.assertFloatsEqual(ss.mask.array, bad)
            self.assertTrue(np.all(ss.variance > 0))
            self.assertTrue(np.all(np.isfinite(ss.variance)))
            self.assertTrue(np.all(np.isfinite(ss.covariance)))
            if ii == 0:
                self.assertTrue(np.all(ss.covariance[-1] == 0))
                self.assertTrue(np.all(ss.covariance[0] != 0))
                self.assertTrue(np.all(ss.covariance[1] != 0))
            elif ii == config.numFibers - 1:
                self.assertTrue(np.all(ss.covariance[-1] != 0))
                self.assertTrue(np.all(ss.covariance[0] != 0))
                self.assertTrue(np.all(ss.covariance[1] == 0))
            else:
                self.assertTrue(np.all(ss.covariance != 0))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    from argparse import ArgumentParser
    parser = ArgumentParser(__file__)
    parser.add_argument("--display", help="Display backend")
    args, argv = parser.parse_known_args()
    display = args.display
    unittest.main(failfast=True, argv=[__file__] + argv)
