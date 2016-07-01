#!/usr/bin/env python
"""
Tests for measuring things

Run with:
   python testDRP.py
"""

import unittest
import sys
import os
import numpy as np
import lsst.utils.tests as tests
import pfs.drp.stella as drpStella
import subprocess

try:
    type(display)
except NameError:
    display = False

class testDRPTestCase(tests.TestCase):
    """A test case for trying out the PFS DRP"""

# IGNORE: (DELETE when finished)
# reduceArcRefSpec.py '/Users/azuri/spectra/pfs/PFS' --id visit=4 
#  --refSpec '/Users/azuri/stella-git/obs_subaru/pfs/lineLists/refCdHgKrNeXe_red.fits' 
#  --lineList '/Users/azuri/stella-git/obs_subaru/pfs/lineLists/CdHgKrNeXe_red.fits' --loglevel 'info' --calib '/Users/azuri/spectra/pfs/PFS/CALIB/' 
#  --output '/Users/azuri/spectra/pfs/PFS'
    def setUp(self):
        self.testDataDir = os.environ.get('DRP_STELLA_DATA_DIR')+'tests/data/'
        self.testCalibDir = self.testDataDir+'CALIB/'
        self.arcVisit = 4
        self.refSpec = 'pfs/arcSpectra/refSpec_CdHgKrNeXe_red.fits'
        self.wLenFile = 'pfs/RedFiberPixels.fits.gz'
        self.lineList = 'pfs/lineLists/CdHgKrNeXe_red.fits'
        
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
        subprocess.Popen("bin.src/reduceArcRefSpec.py %s --id visit=%d --refSpec %s%s --lineList %s%s --loglevel 'info' --calib %s --output %s --clobber-config" % (self.testDataDir, self.arcVisit, os.environ['OBS_PFS_DIR'], self.refSpec, os.environ['OBS_PFS_DIR'], self.lineList, self.testCalibDir, self.testDataDir), shell=True)#, stderr=subprocess.STDOUT, stdout=subprocess.STDOUT)
        subprocess.Popen("bin.src/reduceArc.py %s --id visit=%d --wLenFile %s%s --lineList %s%s --loglevel 'info' --calib %s --output %s --clobber-config" % (self.testDataDir, self.arcVisit, os.environ['OBS_PFS_DIR'], self.wLenFile, os.environ['OBS_PFS_DIR'], self.lineList, self.testCalibDir, self.testDataDir), shell=True)#, stderr=subprocess.STDOUT, stdout=subprocess.STDOUT)
#        for line in p.stdout.readlines():
#            print line,
#        retval = p.wait()
            
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(testDRPTestCase)
#    suites += unittest.makeSuite(tests.MemoryTestCase)
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
