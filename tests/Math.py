#!/usr/bin/env python
"""
Tests for Stella math functions

Run with:
   python Math.py
or
   python
   >>> import Math; Math.run()
"""
import unittest
import numpy as np
import lsst.utils.tests as tests
import pfs.drp.stella as drpStella

class MathTestCase(tests.TestCase):
    """A test case for Stella math functions"""

    def testWhere(self):
        """Test that we can pass a numpy array as ndArray"""
        int1DArr = np.ndarray(shape=(10), dtype=np.int32)
        int1DArr[:] = 0;
        int1DArr[5] = 1;
        int1DArrRes = drpStella.where(int1DArr, '>', 0, 2, 0)
        self.assertEqual(int1DArrRes[5], 2)
        for i in range(10):
            if i != 5:
                self.assertEqual(int1DArrRes[i], 0)
                
    def testPolyFit(self):
        # We are running a C++ test function here because PolyFit functions with
        # keywords are not exposed to Python as I couldn't find a way to make
        # SWIG handle a vector of void pointers to different objects.
        drpStella.testPolyFit()

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
