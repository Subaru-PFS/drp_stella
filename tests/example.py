#!/usr/bin/env python
"""
Tests for measuring things

Run with:
   python example.py
or
   python
   >>> import example; example.run()
"""
import unittest

import numpy as np

import lsst.afw.image as afwImage
import lsst.utils.tests as tests

class ExampleTestCase(tests.TestCase):
    """A test case for measuring Example quantities"""

    def setUp(self):
        self.val1, self.val2 = 10, 90
        bim1 = afwImage.ImageD(10, 20)
        self.im1 = bim1[2:4, 3:6];        self.im1[:] = self.val1 # i.e. memory is not contiguous
        self.im1.setXY0(0, 0)
        self.im2 = self.im1.clone();      self.im2[:] = self.val2

    def tearDown(self):
        del self.im1
        del self.im2

    def checkAddImages(self, result, name):
        print "starting checkAddImages"
        for im, val in [(self.im1, self.val1), (self.im2, self.val2), (result, self.val1 + self.val2)]:
            ima = im.getArray()
            self.assertEqual(np.min(ima), val, name)
            self.assertEqual(np.max(ima), val, name)

        self.assertEqual(result.get(0,0), self.val1 + self.val2, name)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(ExampleTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit = False):
    """Run the tests"""
    tests.run(suite(), exit)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--display", default=False, action="store_true", help="Activate display?")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level")
    args = parser.parse_args()
    display = args.display
    verbose = args.verbose
    run(True)
