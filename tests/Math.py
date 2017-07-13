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

import lsst.log as log
import lsst.utils.tests as tests
import pfs.drp.stella as drpStella
from pfs.drp.stella.math import gauss, makeArtificialSpectrum
from pfs.drp.stella.utils import measureLinesInPixelSpace, removeBadLines

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

    def testGaussFit(self):
        guess = drpStella.GaussCoeffs()
        guess.strength, guess.mu, guess.sigma = [10.0, 3.0, 1.0]
        x = np.arange(7)
        y = gauss(x,guess)
        self.assertAlmostEqual(guess.strength, np.max(y))
        gaussFitResult = drpStella.math.gaussFit(x, y, guess)
        coeffs = drpStella.GaussCoeffs()
        coeffs.strength, coeffs.mu, coeffs.sigma = [gaussFitResult[0],
                                                    gaussFitResult[1],
                                                    gaussFitResult[2]]
        eCoeffs = drpStella.GaussCoeffs()
        eCoeffs.strength, eCoeffs.mu, eCoeffs.sigma = [gaussFitResult[3],
                                                       gaussFitResult[4],
                                                       gaussFitResult[5]]
        self.assertAlmostEqual(guess.strength, coeffs.strength)
        self.assertAlmostEqual(guess.mu, coeffs.mu)
        self.assertAlmostEqual(guess.sigma, coeffs.sigma)
        self.assertAlmostEqual(0.0, eCoeffs.strength)
        self.assertAlmostEqual(0.0, eCoeffs.mu)
        self.assertAlmostEqual(0.0, eCoeffs.sigma)

    def testMakeArtificialSpectrum(self):
        lines = []
        line = drpStella.NistLine()
        line.element='Hg'
        line.flags='g'
        line.id=0
        line.ion='I'
        line.sources='abc'
        line.predictedStrength = 10.0
        line.laboratoryWavelength=30.0
        lines.append(line)
        lambdaPix = np.arange(100)
        fwhm = 1.1774 # sigma = 1
        lambdaMin = 40.0 # line outside range
        lambdaMax = 80.0
        spec = makeArtificialSpectrum(lambdaPix, lines, lambdaMin, lambdaMax, fwhm)
        self.assertAlmostEqual(np.max(spec),0.0)

        lambdaMin = 0.0 # line outside range
        lambdaMax = 20.0
        spec = makeArtificialSpectrum(lambdaPix, lines, lambdaMin, lambdaMax, fwhm)
        self.assertAlmostEqual(np.max(spec),0.0)

        lambdaMin = 10.0 # line outside range
        lambdaMax = 50.0
        spec = makeArtificialSpectrum(lambdaPix, lines, lambdaMin, lambdaMax, fwhm)
        specMax = np.max(spec)
        self.assertAlmostEqual(specMax,10.0)
        self.assertAlmostEqual(specMax,spec[30])

        lambdaMin = 0.0 # line outside range
        lambdaMax = 0.0
        spec = makeArtificialSpectrum(lambdaPix, lines, lambdaMin, lambdaMax, fwhm)
        specMax = np.max(spec)
        self.assertAlmostEqual(specMax,10.0)
        self.assertAlmostEqual(specMax,spec[30])

    def testMeasureLinesInPixelSpace(self):
        lines = []
        lineA = drpStella.NistLine()
        lineA.element='Hg'
        lineA.flags=''
        lineA.id=0
        lineA.ion='I'
        lineA.sources='a'
        lineA.predictedStrength = 10.0
        lineA.laboratoryWavelength=10.0
        lines.append(lineA)

        lineB = drpStella.NistLine()
        lineB.element='Hg'
        lineB.flags=''
        lineB.id=1
        lineB.ion='II'
        lineB.sources='a'
        lineB.predictedStrength = 10.0
        lineB.laboratoryWavelength=30.0
        lines.append(lineB)

        lineC = drpStella.NistLine()
        lineC.element='Hg'
        lineC.flags=''
        lineC.id=2
        lineC.ion='II'
        lineC.sources='a'
        lineC.predictedStrength = 20.0
        lineC.laboratoryWavelength=31.0
        lines.append(lineC)

        lineD = drpStella.NistLine()
        lineD.element='Hg'
        lineD.flags='a'
        lineD.id=3
        lineD.ion='I'
        lineD.sources='a'
        lineD.predictedStrength = 10.0
        lineD.laboratoryWavelength=40.0
        lines.append(lineD)

        lambdaPix = np.arange(50.0)
        fwhm = 2.0
        fluxPix = makeArtificialSpectrum(lambdaPix, lines, fwhm=fwhm)
        linesMeas = measureLinesInPixelSpace(lines, lambdaPix, fluxPix, fwhm)
        for i in np.arange(0,len(linesMeas),3):
            self.assertAlmostEqual(linesMeas[i].pixelPosPredicted,
                                   linesMeas[i].nistLine.laboratoryWavelength)
            self.assertAlmostEqual(linesMeas[i].gaussCoeffsPixel.mu,
                                   linesMeas[i].nistLine.laboratoryWavelength)
            self.assertAlmostEqual(linesMeas[i].gaussCoeffsPixel.strength,
                                   linesMeas[i].nistLine.predictedStrength)
            self.assertAlmostEqual(linesMeas[i].gaussCoeffsPixel.sigma,
                                   fwhm / 1.1774)

        logger = log.Log.getLogger("removeBadLines")
        logger.setLevel(log.WARN)
        goodLines = removeBadLines(linesMeas,
                                   fluxPix,
                                   False,
                                   fwhm,
                                   2.0,
                                   1.0,
                                   8.0,
                                   100.0)
        self.assertEqual(len(goodLines), 1)
        self.assertEqual(linesMeas[0].flags, 'g')
        self.assertEqual(linesMeas[1].flags, 'b')
        self.assertEqual(linesMeas[2].flags, 'b')
        self.assertEqual(linesMeas[3].flags, 'n')

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
