#!/usr/bin/env python
"""
Tests for Stella math functions

Run with:
   python Controls.py
or
   python
   >>> import Controls; Controls.run()
"""
import unittest

import numpy as np

import lsst.utils.tests as tests
import pfs.drp.stella as drpStella

class ControlsTestCase(tests.TestCase):
    """A test case for Stella controls"""


    def testSetCoefficients(self):
        fiberTraceFunction = drpStella.FiberTraceFunction()
        fiberTraceFunction.fiberTraceFunctionControl.order = 1
        fiberTraceFunction.xCenter = 10.0
        fiberTraceFunction.yCenter = 10
        fiberTraceFunction.yLow = -10
        fiberTraceFunction.yHigh = 10

        newCoeffs = np.ndarray(shape=(fiberTraceFunction.fiberTraceFunctionControl.order+2), dtype=np.float32)
        for i in range(newCoeffs.shape[0]):
            newCoeffs[i] = float(i)
        self.assertFalse(fiberTraceFunction.setCoefficients(newCoeffs))

        newCoeffs = np.ndarray(shape=(fiberTraceFunction.fiberTraceFunctionControl.order+1), dtype=np.float32)
        for i in range(newCoeffs.shape[0]):
            newCoeffs[i] = float(i)
        self.assertTrue(fiberTraceFunction.setCoefficients(newCoeffs))

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(ControlsTestCase)
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
