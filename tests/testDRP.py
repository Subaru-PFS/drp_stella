#!/usr/bin/env python
"""
Integration test for reduceArc.py and reduceArcRefSpec.py

Run with:
   python testDRP.py
"""
import os
import subprocess
import unittest

import lsst.utils
import lsst.utils.tests as tests

class testDRPTestCase(tests.TestCase):
    """A test case for trying out the PFS DRP"""

    def setUp(self):
        self.testDataDir = os.path.join(lsst.utils.getPackageDir("drp_stella_data"),'tests/data/PFS')
        self.testCalibDir = os.path.join(self.testDataDir,'CALIB/')
        self.arcVisit = 103
        self.wLenFile = os.path.join(lsst.utils.getPackageDir('obs_pfs'), 'pfs/RedFiberPixels.fits.gz')
        self.refSpec = os.path.join(lsst.utils.getPackageDir('obs_pfs'), 'pfs/arcSpectra/refSpec_CdHgKrNeXe_red.fits')
        self.lineList = os.path.join(lsst.utils.getPackageDir('obs_pfs'), 'pfs/lineLists/CdHgKrNeXe_red.fits')

    def tearDown(self):
        del self.testDataDir
        del self.testCalibDir
        del self.arcVisit
        del self.refSpec
        del self.wLenFile
        del self.lineList

    def testDRP(self):
        logLevel = {0: "WARN", 1: "INFO", 2: "DEBUG"}[verbose]
        args = ["bin/reduceArcRefSpec.py",
                self.testDataDir,
                "--id",
                "visit=%d" % (self.arcVisit,),
                "--refSpec",
                "%s" % (self.refSpec),
                "--lineList", "%s" % (self.lineList),
                "--loglevel",
                "%s" % (logLevel),
                "--calib",
                "%s" % (self.testCalibDir),
                "--output",
                "%s" % (self.testDataDir),
                "--clobber-config",
                "--clobber-versions",
                "--doraise"
        ]
        print " ".join(args)
        proc = subprocess.Popen(args)
        proc.communicate()
        self.assertEqual(proc.returncode, 0)

        args = ["bin/reduceArc.py",
                self.testDataDir,
                "--id",
                "visit=%d" % (self.arcVisit),
                "--wLenFile",
                "%s" % (self.wLenFile),
                "--lineList",
                "%s" % (self.lineList),
                "--loglevel",
                "%s" % (logLevel),
                "--calib",
                "%s" % (self.testCalibDir),
                "--output",
                "%s" % (self.testDataDir),
                "--clobber-config",
                "--clobber-versions",
                "--doraise"
        ]
        print " ".join(args)
        proc = subprocess.Popen(args)
        proc.communicate()
        self.assertEqual(proc.returncode, 0)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(testDRPTestCase)
    return unittest.TestSuite(suites)

def run(exit = False):
    """Run the tests"""
    tests.run(suite(), exit)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--verbose", '-v', type=int, default=0, help="Verbosity level. 0: WARN, 1: INFO, 2: DEBUG")
    args = parser.parse_args()
    verbose = args.verbose
    run(True)
