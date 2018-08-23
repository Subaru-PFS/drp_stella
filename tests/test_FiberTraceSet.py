import os
import sys
import unittest
import pickle

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.daf.base
import lsst.afw.geom
import lsst.afw.image
import lsst.afw.image.testUtils

import pfs.drp.stella as drpStella

display = None


class FiberTraceSetTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        """Set parameters for artificial objects"""
        self.width = 11  # Width of trace
        self.height = 111  # Height of trace
        self.xy0 = lsst.afw.geom.Point2I(123, 123)  # Origin for trace
        self.bbox = lsst.afw.geom.Box2I(
            self.xy0, lsst.afw.geom.Extent2I(self.width, self.height)
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
            ft.trace.setXY0(lsst.afw.geom.Point2I(x0, y0))
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
