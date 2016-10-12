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

    def setUp(self):
        self.int1DArr = np.ndarray(shape=(10), dtype=np.int32)
        self.float1DArr = np.ndarray(shape=(10), dtype='float32')
        
    def tearDown(self):
        del self.int1DArr
        del self.float1DArr

    def testWhere(self):
        """Test that we can pass a numpy array as ndArray"""
        self.int1DArr[:] = 0;
        self.int1DArr[5] = 1;
        int1DArrRes = drpStella.where( self.int1DArr, '>', 0, 2, 0 )
        self.assertEqual(int1DArrRes[5], 2)
        for i in range(10):
            if i != 5:
                self.assertEqual(int1DArrRes[i], 0)
            
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
