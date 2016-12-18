#!/usr/bin/env python
"""
Tests for measuring things

Run with:
   python testDRP.py
"""

import unittest
import os
import lsst.utils
import lsst.utils.tests as tests
import subprocess

try:
    type(display)
except NameError:
    display = False

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
        print 'self.testDataDir = <',self.testDataDir,'>'
        print 'self.testCalibDir = <',self.testCalibDir,'>'
        print 'self.arcVisit = <',self.arcVisit,'>'
        print 'self.refSpec = <',self.refSpec,'>'
        print 'self.lineList = <',self.lineList,'>'
        print "os.environ['OBS_PFS_DIR'] = <",os.environ['OBS_PFS_DIR'],">"
        subprocess.Popen("bin.src/reduceArcRefSpec.py %s --id visit=%d --refSpec %s --lineList %s --loglevel 'info' --calib %s --output %s --clobber-config --clobber-versions" % (self.testDataDir, self.arcVisit, self.refSpec, self.lineList, self.testCalibDir, self.testDataDir), shell=True)
        subprocess.Popen("bin.src/reduceArc.py %s --id visit=%d --wLenFile %s --lineList %s --loglevel 'info' --calib %s --output %s --clobber-config --clobber-versions" % (self.testDataDir, self.arcVisit, self.wLenFile, self.lineList, self.testCalibDir, self.testDataDir), shell=True)
            
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
    parser.add_argument("--display", '-d', default=False, action="store_true", help="Activate display?")
    parser.add_argument("--verbose", '-v', type=int, default=0, help="Verbosity level")
    args = parser.parse_args()
    display = args.display
    verbose = args.verbose
    run(True)
