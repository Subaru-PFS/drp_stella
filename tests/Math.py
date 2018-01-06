#!/usr/bin/env python
"""
Tests for Stella math functions

Run with:
   python Math.py
or
   python
   >>> import Math; Math.run()
"""
from builtins import zip
from builtins import range
import unittest

import numpy as np

import lsst.utils.tests as tests
import pfs.drp.stella as drpStella

class MathTestCase(tests.TestCase):
    """A test case for Stella math functions"""

    @unittest.skip("'where' is strongly deprecated and is not bound to python")
    def testWhere(self):
        # Test that we can pass a numpy array as ndArray
        int1DArr = np.ndarray(shape=(10), dtype=np.int32)
        int1DArr[:] = 0;
        int1DArr[5] = 1;
        int1DArrRes = drpStella.where(int1DArr, '>', 0, 2, 0)
        self.assertEqual(int1DArrRes[5], 2)
        for i in range(10):
            if i != 5:
                self.assertEqual(int1DArrRes[i], 0)

    # @unittest.skip("Not bound to python; will use boost::unittest if we need tests")
    def testPolyFit(self):
        # We are running a C++ test function here because PolyFit functions with
        # keywords are not exposed to Python as I couldn't find a way to make
        # SWIG handle a vector of void pointers to different objects.
        drpStella.testPolyFit()

    @unittest.skip("'sortIndices' is strongly deprecated and is not bound to python")
    def testSortIndices(self):
        """Test drpStella.sortIndices"""

        # ascending list
        int1DArr = np.ndarray(shape=(10), dtype=np.int32)
        int1DArr[:] = range(10)[:]
        sortedIndices = drpStella.sortIndices(int1DArr)
        for ind, val in zip(sortedIndices, int1DArr):
            self.assertEqual(ind, val)

        # descending list
        int1DArr[:] = np.arange(9,-1,-1)[:]
        sortedIndices = drpStella.sortIndices(int1DArr)
        for ind, val in zip(sortedIndices, int1DArr):
            self.assertEqual(ind, val)

        # unordered list
        int1DArr[0] = 5
        int1DArr[1] = 8
        int1DArr[2] = 0
        int1DArr[3] = 9
        int1DArr[4] = 1
        int1DArr[5] = 6
        int1DArr[6] = 4
        int1DArr[7] = 2
        int1DArr[8] = 3
        int1DArr[9] = 7
        sortedIndices = drpStella.sortIndices(int1DArr)
        for ind, val in zip(sortedIndices, [2,4,7,8,6,0,5,9,1,3]):
            self.assertEqual(ind, val)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(MathTestCase)
    return unittest.TestSuite(suites)

def run(exit = False):
    """Run the tests"""
    tests.run(suite(), exit)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--verbose", '-v', type=int, default=0, help="Verbosity level")
    args = parser.parse_args()
    verbose = args.verbose
    run(True)
