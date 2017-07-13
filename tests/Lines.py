#!/usr/bin/env python
"""
Tests for measuring things

Run with:
   python Lines.py
or
   python
   >>> import Lines; Lines.run()
"""
import lsst.log as log
import lsst.utils.tests as tests
import numpy as np
import pfs.drp.stella as drpStella
from pfs.drp.stella.math import gauss
from pfs.drp.stella.utils import getLinesInWavelengthRange
from pfs.drp.stella.utils import measureLinesInPixelSpace
from pfs.drp.stella.utils import measureLinesInWavelengthSpace
from pfs.drp.stella.utils import removeBadLines
import unittest

class LinesTestCase(tests.TestCase):
    """A test case for measuring Lines quantities"""

    def setUp(self):
        self.lineList = drpStella.getNistLineMeasVec()

    def tearDown(self):
        pass

    def testNistLineStandardConstructor(self):
        """Test that we can create a NistLine with the standard constructor"""
        nistLine = drpStella.NistLine()
        self.assertEqual(nistLine.element, "")
        self.assertEqual(nistLine.flags, "")
        self.assertEqual(nistLine.id, 0)
        self.assertEqual(nistLine.ion, "")
        self.assertEqual(nistLine.laboratoryWavelength, 0.0)
        self.assertEqual(nistLine.predictedStrength, 0.0)
        self.assertEqual(nistLine.sources, "")

        nistLineA = drpStella.NistLine()
        self.assertFalse(nistLine is nistLineA)
        element = 'Hg'
        nistLine.element = element
        self.assertEqual(nistLine.element, element)
        self.assertFalse(nistLine.element is nistLineA.element)

        flags = 'g'
        nistLine.flags = flags
        self.assertEqual(nistLine.flags, flags)
        self.assertFalse(nistLine.flags is nistLineA.flags)

        id = 1
        nistLine.id = id
        self.assertEqual(nistLine.id, id)
        self.assertFalse(nistLine.id is nistLineA.id)

        ion = 'I'
        nistLine.ion = ion
        self.assertEqual(nistLine.ion, ion)
        self.assertFalse(nistLine.ion is nistLineA.ion)

        wavelength = 333.3393
        nistLine.laboratoryWavelength = wavelength
        self.assertAlmostEqual(nistLine.laboratoryWavelength, wavelength, 4)
        self.assertFalse(nistLine.laboratoryWavelength
                         is nistLineA.laboratoryWavelength)

        strength = 333.3393
        nistLine.predictedStrength = strength
        self.assertAlmostEqual(nistLine.predictedStrength, strength, 4)
        self.assertFalse(nistLine.predictedStrength
                         is nistLineA.predictedStrength)

        sources = 'L1'
        nistLine.sources = sources
        self.assertEqual(nistLine.sources, sources)
        self.assertFalse(nistLine.sources is nistLineA.sources)

    def testNistLineCopyConstructor(self):
        """Test that we can create a NistLine with the copy constructor"""
        nistLine = drpStella.NistLine()

        element = 'Hg'
        nistLine.element = element

        flags = 'g'
        nistLine.flags = flags

        id = 1
        nistLine.id = id

        ion = 'I'
        nistLine.ion = ion

        wavelength = 333.3393
        nistLine.laboratoryWavelength = wavelength

        strength = 333.3393
        nistLine.predictedStrength = strength

        sources = 'L1'
        nistLine.sources = sources

        nistLineA = drpStella.NistLine(nistLine)
        self.assertFalse(nistLine is nistLineA)

        self.assertEqual(nistLine.element, nistLineA.element)
        nistLine.element = 'Xe'
        self.assertFalse(nistLine.element is nistLineA.element)

        self.assertEqual(nistLine.flags, nistLineA.flags)
        nistLine.flags = 'h'
        self.assertFalse(nistLine.flags is nistLineA.flags)

        self.assertEqual(nistLine.id, nistLineA.id)
        nistLine.id = 7
        self.assertFalse(nistLine.id is nistLineA.id)

        self.assertEqual(nistLine.ion, nistLineA.ion)
        nistLine.ion = 'II'
        self.assertFalse(nistLine.ion is nistLineA.ion)

        self.assertAlmostEqual(nistLine.laboratoryWavelength,
                              nistLineA.laboratoryWavelength)
        nistLine.laboratoryWavelength = 123.456
        self.assertFalse(nistLine.laboratoryWavelength
                         is nistLineA.laboratoryWavelength)

        self.assertAlmostEqual(nistLine.predictedStrength, nistLineA.predictedStrength)
        nistLine.predictedStrength = 3.9
        self.assertFalse(nistLine.predictedStrength is nistLineA.predictedStrength)

    def testNistLineMeas(self):
        """Test that we can create a NistLineMeas with the standard constructor"""
        nistLineMeas = drpStella.NistLineMeas()
        coeffs = drpStella.GaussCoeffs()
        self.assertAlmostEqual(nistLineMeas.eGaussCoeffsLambda.mu,
                               coeffs.mu)
        self.assertAlmostEqual(nistLineMeas.eGaussCoeffsLambda.strength,
                               coeffs.strength)
        self.assertAlmostEqual(nistLineMeas.eGaussCoeffsLambda.sigma,
                               coeffs.sigma)
        self.assertAlmostEqual(nistLineMeas.eGaussCoeffsPixel.mu,
                               coeffs.mu)
        self.assertAlmostEqual(nistLineMeas.eGaussCoeffsPixel.strength,
                               coeffs.strength)
        self.assertAlmostEqual(nistLineMeas.eGaussCoeffsPixel.sigma,
                               coeffs.sigma)
        self.assertAlmostEqual(nistLineMeas.gaussCoeffsLambda.mu,
                               coeffs.mu)
        self.assertAlmostEqual(nistLineMeas.gaussCoeffsLambda.strength,
                               coeffs.strength)
        self.assertAlmostEqual(nistLineMeas.gaussCoeffsLambda.sigma,
                               coeffs.sigma)
        self.assertAlmostEqual(nistLineMeas.gaussCoeffsPixel.mu,
                               coeffs.mu)
        self.assertAlmostEqual(nistLineMeas.gaussCoeffsPixel.strength,
                               coeffs.strength)
        self.assertAlmostEqual(nistLineMeas.gaussCoeffsPixel.sigma,
                               coeffs.sigma)

        self.assertEqual(nistLineMeas.flags, '')

        nistLine = drpStella.NistLine()
        self.assertEqual(nistLineMeas.nistLine.element, nistLine.element)

        self.assertAlmostEqual(nistLineMeas.pixelPosPredicted, 0.0)

        self.assertAlmostEqual(nistLineMeas.wavelengthFromPixelPosAndPoly, 0.0)

        nistLineMeasA = drpStella.NistLineMeas()
        self.assertFalse(nistLineMeas is nistLineMeasA)

        mu = 1.0
        nistLineMeasA.eGaussCoeffsLambda.mu = mu
        self.assertAlmostEqual(nistLineMeasA.eGaussCoeffsLambda.mu, mu)
        self.assertFalse(nistLineMeasA.eGaussCoeffsLambda.mu
                         is nistLineMeas.eGaussCoeffsLambda.mu)
        nistLineMeasA.eGaussCoeffsPixel.mu = mu
        self.assertAlmostEqual(nistLineMeasA.eGaussCoeffsPixel.mu, mu)
        self.assertFalse(nistLineMeasA.eGaussCoeffsPixel.mu
                         is nistLineMeas.eGaussCoeffsPixel.mu)
        nistLineMeasA.gaussCoeffsLambda.mu = mu
        self.assertAlmostEqual(nistLineMeasA.gaussCoeffsLambda.mu, mu)
        self.assertFalse(nistLineMeasA.gaussCoeffsLambda.mu
                         is nistLineMeas.gaussCoeffsLambda.mu)
        nistLineMeasA.gaussCoeffsPixel.mu = mu
        self.assertAlmostEqual(nistLineMeasA.gaussCoeffsPixel.mu, mu)
        self.assertFalse(nistLineMeasA.gaussCoeffsPixel.mu
                         is nistLineMeas.gaussCoeffsPixel.mu)

        sigma = 1.0
        nistLineMeasA.eGaussCoeffsLambda.sigma = sigma
        self.assertAlmostEqual(nistLineMeasA.eGaussCoeffsLambda.sigma, sigma)
        self.assertFalse(nistLineMeasA.eGaussCoeffsLambda.sigma
                         is nistLineMeas.eGaussCoeffsLambda.sigma)
        nistLineMeasA.eGaussCoeffsPixel.sigma = sigma
        self.assertAlmostEqual(nistLineMeasA.eGaussCoeffsPixel.sigma, sigma)
        self.assertFalse(nistLineMeasA.eGaussCoeffsPixel.sigma
                         is nistLineMeas.eGaussCoeffsPixel.sigma)
        nistLineMeasA.gaussCoeffsLambda.sigma = sigma
        self.assertAlmostEqual(nistLineMeasA.gaussCoeffsLambda.sigma, sigma)
        self.assertFalse(nistLineMeasA.gaussCoeffsLambda.sigma
                         is nistLineMeas.gaussCoeffsLambda.sigma)
        nistLineMeasA.gaussCoeffsPixel.sigma = sigma
        self.assertAlmostEqual(nistLineMeasA.gaussCoeffsPixel.sigma, sigma)
        self.assertFalse(nistLineMeasA.gaussCoeffsPixel.sigma
                         is nistLineMeas.gaussCoeffsPixel.sigma)

        strength = 1.0
        nistLineMeasA.eGaussCoeffsLambda.strength = strength
        self.assertAlmostEqual(nistLineMeasA.eGaussCoeffsLambda.strength, strength)
        self.assertFalse(nistLineMeasA.eGaussCoeffsLambda.strength
                         is nistLineMeas.eGaussCoeffsLambda.strength)
        nistLineMeasA.eGaussCoeffsPixel.strength = strength
        self.assertAlmostEqual(nistLineMeasA.eGaussCoeffsPixel.strength, strength)
        self.assertFalse(nistLineMeasA.eGaussCoeffsPixel.strength
                         is nistLineMeas.eGaussCoeffsPixel.strength)

        flags = 'i'
        nistLineMeasA.flags = flags
        self.assertEqual(nistLineMeasA.flags, flags)
        self.assertFalse(nistLineMeasA.flags is nistLineMeas.flags)

        pix = 1.0
        nistLineMeasA.pixelPosPredicted = pix
        self.assertAlmostEqual(nistLineMeasA.pixelPosPredicted, pix)
        self.assertFalse(nistLineMeasA.pixelPosPredicted
                         is nistLineMeas.pixelPosPredicted)

        wave = 628.1113
        nistLineMeasA.wavelengthFromPixelPosAndPoly = wave
        self.assertAlmostEqual(nistLineMeasA.wavelengthFromPixelPosAndPoly, wave, 4)
        self.assertFalse(nistLineMeasA.wavelengthFromPixelPosAndPoly
                         is nistLineMeas.wavelengthFromPixelPosAndPoly)

    def testNistLineMeasCopyConstructor(self):
        """Test that we can create a NistLineMeas with the copy constructor"""
        nistLineMeas = drpStella.NistLineMeas()

        element = 'Hg'
        nistLineMeas.nistLine.element = element

        nlFlags = 'g'
        nistLineMeas.nistLine.flags = nlFlags

        id = 1
        nistLineMeas.nistLine.id = id

        ion = 'I'
        nistLineMeas.nistLine.ion = ion

        wavelength = 333.3393
        nistLineMeas.nistLine.laboratoryWavelength = wavelength

        strength = 333.3393
        nistLineMeas.nistLine.predictedStrength = strength

        sources = 'L1'
        nistLineMeas.nistLine.sources = sources

        coeffsLam = drpStella.GaussCoeffs()
        coeffsLam.mu = 1.0
        coeffsLam.sigma = 2.0
        coeffsLam.strength = 0.1
        nistLineMeas.gaussCoeffsLambda = coeffsLam

        coeffsPix = drpStella.GaussCoeffs()
        coeffsPix.mu = 1.0
        coeffsPix.sigma = 2.0
        coeffsPix.strength = 0.1
        nistLineMeas.gaussCoeffsPixel = coeffsPix

        coeffsELam = drpStella.GaussCoeffs()
        coeffsELam.mu = 1.0
        coeffsELam.sigma = 2.0
        coeffsELam.strength = 0.1
        nistLineMeas.eGaussCoeffsLambda = coeffsELam

        coeffsEPix = drpStella.GaussCoeffs()
        coeffsEPix.mu = 1.0
        coeffsEPix.sigma = 2.0
        coeffsEPix.strength = 0.1
        nistLineMeas.eGaussCoeffsPixel = coeffsEPix

        flags = "g"
        nistLineMeas.flags = flags

        pixelPosPredicted = 1923.45
        nistLineMeas.pixelPosPredicted = pixelPosPredicted

        wavelengthFromPixelPosAndPoly = 923.546
        nistLineMeas.wavelengthFromPixelPosAndPoly = wavelengthFromPixelPosAndPoly

        # create copy of nistLineMeas
        nistLineMeasA = drpStella.NistLineMeas(nistLineMeas)
        self.assertFalse(nistLineMeas is nistLineMeasA)

        self.assertEqual(nistLineMeas.nistLine.element,
                         element)
        nistLineMeas.nistLine.element = 'Xe'
        self.assertFalse(nistLineMeas.nistLine.element
                         is nistLineMeasA.nistLine.element)

        self.assertEqual(nistLineMeas.nistLine.flags,
                         nlFlags)
        nistLineMeas.nistLine.flags = 'h'
        self.assertFalse(nistLineMeas.nistLine.flags
                         is nistLineMeasA.nistLine.flags)

        self.assertEqual(nistLineMeas.nistLine.id,
                         id)
        nistLineMeas.nistLine.id = 4
        self.assertFalse(nistLineMeas.nistLine.id is nistLineMeasA.nistLine.id)

        self.assertEqual(nistLineMeas.nistLine.ion,
                         ion)
        nistLineMeas.nistLine.ion = "IV"
        self.assertFalse(nistLineMeas.nistLine.ion is nistLineMeasA.nistLine.ion)

        self.assertAlmostEqual(nistLineMeas.nistLine.laboratoryWavelength,
                               wavelength,
                               4)
        nistLineMeas.nistLine.laboratoryWavelength = 777.777
        self.assertFalse(nistLineMeas.nistLine.laboratoryWavelength
                         is nistLineMeasA.nistLine.laboratoryWavelength)

        self.assertAlmostEqual(nistLineMeas.nistLine.predictedStrength,
                               strength,
                               4)
        nistLineMeas.nistLine.predictedStrength = 0.7
        self.assertFalse(nistLineMeas.nistLine.predictedStrength
                         is nistLineMeasA.nistLine.predictedStrength)

        self.assertEqual(nistLineMeas.nistLine.sources,
                         sources)
        nistLineMeas.nistLine.sources = 'R3252'
        self.assertFalse(nistLineMeas.nistLine.sources
                         is nistLineMeasA.nistLine.sources)

        self.assertAlmostEqual(nistLineMeas.eGaussCoeffsLambda.mu,
                               coeffsLam.mu)
        nistLineMeas.eGaussCoeffsLambda.mu = 7.8
        self.assertFalse(nistLineMeas.eGaussCoeffsLambda.mu
                         is nistLineMeasA.eGaussCoeffsLambda.mu)

        self.assertAlmostEqual(nistLineMeas.eGaussCoeffsLambda.sigma,
                               coeffsLam.sigma)
        nistLineMeas.eGaussCoeffsLambda.sigma = 7.8
        self.assertFalse(nistLineMeas.eGaussCoeffsLambda.sigma
                         is nistLineMeasA.eGaussCoeffsLambda.sigma)

        self.assertAlmostEqual(nistLineMeas.eGaussCoeffsLambda.strength,
                               coeffsLam.strength)
        nistLineMeas.eGaussCoeffsLambda.strength = 7.8
        self.assertFalse(nistLineMeas.eGaussCoeffsLambda.strength
                         is nistLineMeasA.eGaussCoeffsLambda.strength)


        self.assertAlmostEqual(nistLineMeas.eGaussCoeffsPixel.mu,
                               coeffsPix.mu)
        nistLineMeas.eGaussCoeffsPixel.mu = 7.8
        self.assertFalse(nistLineMeas.eGaussCoeffsPixel.mu
                         is nistLineMeasA.eGaussCoeffsPixel.mu)

        self.assertAlmostEqual(nistLineMeas.eGaussCoeffsPixel.sigma,
                               coeffsPix.sigma)
        nistLineMeas.eGaussCoeffsPixel.sigma = 7.8
        self.assertFalse(nistLineMeas.eGaussCoeffsPixel.sigma
                         is nistLineMeasA.eGaussCoeffsPixel.sigma)

        self.assertAlmostEqual(nistLineMeas.eGaussCoeffsPixel.strength,
                               coeffsPix.strength)
        nistLineMeas.eGaussCoeffsPixel.strength = 7.8
        self.assertFalse(nistLineMeas.eGaussCoeffsPixel.strength
                         is nistLineMeasA.eGaussCoeffsPixel.strength)


        self.assertAlmostEqual(nistLineMeas.gaussCoeffsLambda.mu,
                               coeffsELam.mu)
        nistLineMeas.gaussCoeffsLambda.mu = 7.8
        self.assertFalse(nistLineMeas.gaussCoeffsLambda.mu
                         is nistLineMeasA.gaussCoeffsLambda.mu)

        self.assertAlmostEqual(nistLineMeas.gaussCoeffsLambda.sigma,
                               coeffsELam.sigma)
        nistLineMeas.gaussCoeffsLambda.sigma = 7.8
        self.assertFalse(nistLineMeas.gaussCoeffsLambda.sigma
                         is nistLineMeasA.gaussCoeffsLambda.sigma)

        self.assertAlmostEqual(nistLineMeas.gaussCoeffsLambda.strength,
                               coeffsELam.strength)
        nistLineMeas.gaussCoeffsLambda.strength = 7.8
        self.assertFalse(nistLineMeas.gaussCoeffsLambda.strength
                         is nistLineMeasA.gaussCoeffsLambda.strength)


        self.assertAlmostEqual(nistLineMeas.gaussCoeffsPixel.mu,
                               coeffsEPix.mu)
        nistLineMeas.gaussCoeffsPixel.mu = 7.8
        self.assertFalse(nistLineMeas.gaussCoeffsPixel.mu
                         is nistLineMeasA.gaussCoeffsPixel.mu)

        self.assertAlmostEqual(nistLineMeas.gaussCoeffsPixel.sigma,
                               coeffsEPix.sigma)
        nistLineMeas.gaussCoeffsPixel.sigma = 7.8
        self.assertFalse(nistLineMeas.gaussCoeffsPixel.sigma
                         is nistLineMeasA.gaussCoeffsPixel.sigma)

        self.assertAlmostEqual(nistLineMeas.gaussCoeffsPixel.strength,
                               coeffsEPix.strength)
        nistLineMeas.gaussCoeffsPixel.strength = 7.8
        self.assertFalse(nistLineMeas.gaussCoeffsPixel.strength
                         is nistLineMeasA.gaussCoeffsPixel.strength)


        self.assertEqual(nistLineMeas.flags, flags)
        nistLineMeas.flags = 'f'
        self.assertFalse(nistLineMeas.flags
                         is nistLineMeasA.flags)

        self.assertAlmostEqual(nistLineMeas.pixelPosPredicted,
                               pixelPosPredicted,
                               4)
        nistLineMeas.pixelPosPredicted = 7.8
        self.assertFalse(nistLineMeas.pixelPosPredicted
                         is nistLineMeasA.pixelPosPredicted)

        self.assertAlmostEqual(nistLineMeas.wavelengthFromPixelPosAndPoly,
                               wavelengthFromPixelPosAndPoly,
                               4)
        nistLineMeas.wavelengthFromPixelPosAndPoly = 7.8
        self.assertFalse(nistLineMeas.wavelengthFromPixelPosAndPoly
                         is nistLineMeasA.wavelengthFromPixelPosAndPoly)

    def testGetPointer(self):
        nistLine = drpStella.NistLine()
        nistLine.ion = 'V'
        nistLineA = nistLine.getPointer()
        self.assertEqual(nistLine.ion, nistLineA.ion)

    def testGetLinesWithFlags(self):
        lines = drpStella.getNistLineMeasVec()
        lineA = drpStella.NistLineMeas()
        lineA.flags = 'g'
        lines.append(lineA)

        lineB = drpStella.NistLineMeas()
        lineB.flags = 'gi'
        lines.append(lineB)

        lineC = drpStella.NistLineMeas()
        lineC.flags = 'gf'
        lines.append(lineC)

        lineD = drpStella.NistLineMeas()
        lineD.flags = 'g'
        lines.append(lineD)

        returnedLines = drpStella.getLinesWithFlags(lines,
                                                    'g')
        self.assertEqual(len(returnedLines), 4)
        self.assertEqual(lineA.flags, returnedLines[0].flags)
        self.assertEqual(lineB.flags, returnedLines[1].flags)
        self.assertEqual(lineC.flags, returnedLines[2].flags)
        self.assertEqual(lineD.flags, returnedLines[3].flags)

        returnedLines = drpStella.getLinesWithFlags(lines,
                                                    'g',
                                                    'i')
        self.assertEqual(len(returnedLines), 3)
        self.assertEqual(lineA.flags, returnedLines[0].flags)
        self.assertEqual(lineC.flags, returnedLines[1].flags)
        self.assertEqual(lineD.flags, returnedLines[2].flags)

        returnedLines = drpStella.getLinesWithFlags(lines,
                                                    'gf')
        self.assertEqual(len(returnedLines), 1)
        self.assertEqual(lineC.flags, returnedLines[0].flags)

        returnedLines = drpStella.getLinesWithFlags(lines,
                                                    'gf',
                                                    'f')
        self.assertEqual(len(returnedLines), 0)

    def testGetLinesWithID(self):
        lines = drpStella.getNistLineMeasVec()
        lineA = drpStella.NistLineMeas()
        lineA.nistLine.id = 0
        lines.append(lineA)

        lineB = drpStella.NistLineMeas()
        lineB.nistLine.id = 1
        lines.append(lineB)

        lineC = drpStella.NistLineMeas()
        lineC.nistLine.id = 0
        lines.append(lineC)

        lineD = drpStella.NistLineMeas()
        lineD.nistLine.id = 2
        lines.append(lineD)

        returnedLines = drpStella.getLinesWithID(lines,
                                                 0)
        self.assertEqual(len(returnedLines), 2)
        self.assertEqual(lineA.nistLine.id, returnedLines[0].nistLine.id)
        self.assertEqual(lineC.nistLine.id, returnedLines[1].nistLine.id)

        returnedLines = drpStella.getLinesWithID(lines,
                                                 1)
        self.assertEqual(len(returnedLines), 1)
        self.assertEqual(lineB.nistLine.id, returnedLines[0].nistLine.id)

        returnedLines = drpStella.getLinesWithID(lines,
                                                 2)
        self.assertEqual(len(returnedLines), 1)
        self.assertEqual(lineD.nistLine.id, returnedLines[0].nistLine.id)

        returnedLines = drpStella.getLinesWithID(lines,
                                                 3)
        self.assertEqual(len(returnedLines), 0)

    def testGetIndexOfLineWithID(self):
        lines = drpStella.getNistLineMeasVec()
        lineA = drpStella.NistLineMeas()
        lineA.nistLine.id = 0
        lines.append(lineA)

        lineB = drpStella.NistLineMeas()
        lineB.nistLine.id = 1
        lines.append(lineB)

        lineC = drpStella.NistLineMeas()
        lineC.nistLine.id = 0
        lines.append(lineC)

        lineD = drpStella.NistLineMeas()
        lineD.nistLine.id = 2
        lines.append(lineD)

        returnedLines = drpStella.getIndexOfLineWithID(lines,
                                                       0)
        self.assertEqual(len(returnedLines), 2)
        self.assertEqual(returnedLines[0], 0)
        self.assertEqual(returnedLines[1], 2)
        self.assertEqual(lineA.nistLine.id, lines[returnedLines[0]].nistLine.id)
        self.assertEqual(lineC.nistLine.id, lines[returnedLines[1]].nistLine.id)

        returnedLines = drpStella.getIndexOfLineWithID(lines,
                                                       1)
        self.assertEqual(len(returnedLines), 1)
        self.assertEqual(returnedLines[0], 1)
        self.assertEqual(lineB.nistLine.id, lines[returnedLines[0]].nistLine.id)

        returnedLines = drpStella.getIndexOfLineWithID(lines,
                                                       2)
        self.assertEqual(len(returnedLines), 1)
        self.assertEqual(returnedLines[0], 3)
        self.assertEqual(lineD.nistLine.id, lines[returnedLines[0]].nistLine.id)

        returnedLines = drpStella.getIndexOfLineWithID(lines,
                                                       3)
        self.assertEqual(len(returnedLines), 0)

    def testCreateLineListFromWLenPix(self):
        lineListArr = np.ndarray(shape=(4, 2), dtype=np.float32)#drpStella.ndarrayF(4,2)#
        lineListArr[0, 0] = 1.0
        lineListArr[0, 1] = 2.0
        lineListArr[1, 0] = 3.0
        lineListArr[1, 1] = 4.0
        lineListArr[2, 0] = 5.0
        lineListArr[2, 1] = 6.0
        lineListArr[3, 0] = 7.0
        lineListArr[3, 1] = 8.0

        lines = drpStella.createLineListFromWLenPix(lineListArr)
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0].nistLine.laboratoryWavelength, lineListArr[0, 0])
        self.assertEqual(lines[0].pixelPosPredicted, lineListArr[0, 1])
        self.assertEqual(lines[1].nistLine.laboratoryWavelength, lineListArr[1, 0])
        self.assertEqual(lines[1].pixelPosPredicted, lineListArr[1, 1])
        self.assertEqual(lines[2].nistLine.laboratoryWavelength, lineListArr[2, 0])
        self.assertEqual(lines[2].pixelPosPredicted, lineListArr[2, 1])
        self.assertEqual(lines[3].nistLine.laboratoryWavelength, lineListArr[3, 0])
        self.assertEqual(lines[3].pixelPosPredicted, lineListArr[3, 1])

        lineListArrA = np.ndarray(shape=(4, 3), dtype=np.float32)
        lineListArrA[:,0:2] = lineListArr
        lineListArrA[0,2] = 9
        lineListArrA[1,2] = 10
        lineListArrA[2,2] = 11
        lineListArrA[3,2] = 12

        lines = drpStella.createLineListFromWLenPix(lineListArrA)
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0].nistLine.predictedStrength, lineListArrA[0, 2])
        self.assertEqual(lines[1].nistLine.predictedStrength, lineListArrA[1, 2])
        self.assertEqual(lines[2].nistLine.predictedStrength, lineListArrA[2, 2])
        self.assertEqual(lines[3].nistLine.predictedStrength, lineListArrA[3, 2])

    def testGetLinesInWavelengthRange(self):
        linesIn = []
        lineA = drpStella.NistLine()
        lineA.laboratoryWavelength = 400.0
        linesIn.append(lineA)
        lineB = drpStella.NistLine()
        lineB.laboratoryWavelength = 300.0
        linesIn.append(lineB)
        self.assertNotAlmostEqual(linesIn[0].laboratoryWavelength,
                                  linesIn[1].laboratoryWavelength)
        lineC = drpStella.NistLine()
        lineC.laboratoryWavelength = 200.0
        linesIn.append(lineC)
        lineD = drpStella.NistLine()
        lineD.laboratoryWavelength = 100.0
        linesIn.append(lineD)

        linesOut = getLinesInWavelengthRange(linesIn, 200.0, 300.0)
        self.assertEqual(len(linesOut), 0)

        linesOut = getLinesInWavelengthRange(linesIn, 199.0, 301.0)
        self.assertEqual(len(linesOut), 2)
        self.assertEqual(linesOut[0].laboratoryWavelength, lineC.laboratoryWavelength)
        self.assertEqual(linesOut[1].laboratoryWavelength, lineB.laboratoryWavelength)

        linesOut = getLinesInWavelengthRange(linesIn, 99.0, 401.0)
        self.assertEqual(len(linesOut), 4)
        self.assertEqual(linesOut[0].laboratoryWavelength, lineD.laboratoryWavelength)
        self.assertEqual(linesOut[1].laboratoryWavelength, lineC.laboratoryWavelength)
        self.assertEqual(linesOut[2].laboratoryWavelength, lineB.laboratoryWavelength)
        self.assertEqual(linesOut[3].laboratoryWavelength, lineA.laboratoryWavelength)

        linesOut = getLinesInWavelengthRange(linesIn, 0.0, 99.0)
        self.assertEqual(len(linesOut), 0)

    def testMeasureLines(self):
        logger = log.Log.getLogger("measureLinesInPixelSpace")
        logger.setLevel(log.WARN)

        lines = []

        lineA = drpStella.NistLine()
        lineA.laboratoryWavelength = 10.0
        lineA.predictedStrength = 10.0
        lines.append(lineA)
        self.assertAlmostEqual(lineA.laboratoryWavelength, lines[0].laboratoryWavelength)
        self.assertAlmostEqual(lineA.predictedStrength, lines[0].predictedStrength)

        lineB = drpStella.NistLine()
        lineB.laboratoryWavelength = 20.0
        lineB.predictedStrength = 20.0
        lines.append(lineB)
        self.assertAlmostEqual(lineB.laboratoryWavelength, lines[1].laboratoryWavelength)
        self.assertAlmostEqual(lineB.predictedStrength, lines[1].predictedStrength)
        self.assertNotAlmostEqual(lines[0].laboratoryWavelength, lines[1].laboratoryWavelength)

        lambdaPix = range(30)
        fluxPix = np.ndarray(shape=(30), dtype=np.float32)
        fluxPix[:] = 0.0

        sigma = 1.0
        fluxPix += self.getGaussian(lineA, sigma, lambdaPix)
        fluxPix += self.getGaussian(lineB, sigma, lambdaPix)

        # fit all lines by not setting minStrength
        measuredLines = measureLinesInPixelSpace(lines,
                                                 lambdaPix,
                                                 fluxPix,
                                                 2.34)
        self.assertEqual(len(lines), len(measuredLines))

        self.assertAlmostEqual(measuredLines[0].pixelPosPredicted,
                               lineA.laboratoryWavelength)
        self.assertAlmostEqual(measuredLines[0].gaussCoeffsPixel.mu,
                               lineA.laboratoryWavelength)
        self.assertAlmostEqual(measuredLines[0].eGaussCoeffsPixel.mu, 0.0)
        self.assertAlmostEqual(measuredLines[0].gaussCoeffsPixel.sigma,
                               sigma)
        self.assertAlmostEqual(measuredLines[0].eGaussCoeffsPixel.sigma, 0.0)
        self.assertAlmostEqual(measuredLines[0].gaussCoeffsPixel.strength,
                               lineA.predictedStrength)
        self.assertAlmostEqual(measuredLines[0].eGaussCoeffsPixel.strength, 0.0)

        self.assertAlmostEqual(measuredLines[1].pixelPosPredicted,
                               lineB.laboratoryWavelength)
        self.assertAlmostEqual(measuredLines[1].gaussCoeffsPixel.mu,
                               lineB.laboratoryWavelength)
        self.assertAlmostEqual(measuredLines[1].eGaussCoeffsPixel.mu, 0.0)
        self.assertAlmostEqual(measuredLines[1].gaussCoeffsPixel.sigma,
                               sigma)
        self.assertAlmostEqual(measuredLines[1].eGaussCoeffsPixel.sigma, 0.0)
        self.assertAlmostEqual(measuredLines[1].gaussCoeffsPixel.strength,
                               lineB.predictedStrength)
        self.assertAlmostEqual(measuredLines[1].eGaussCoeffsPixel.strength, 0.0)

        # run again with minStrength set to 11.0, so that the 1st line won't get
        # fitted
        measuredLines = measureLinesInPixelSpace(lines,
                                                 lambdaPix,
                                                 fluxPix,
                                                 2.34,
                                                 11.0)
        self.assertEqual(len(measuredLines), len(measuredLines))
        self.assertAlmostEqual(measuredLines[0].gaussCoeffsPixel.mu,
                               0.0)
        self.assertAlmostEqual(measuredLines[0].eGaussCoeffsPixel.mu, 0.0)
        self.assertAlmostEqual(measuredLines[0].gaussCoeffsPixel.sigma,
                               0.0)
        self.assertAlmostEqual(measuredLines[0].eGaussCoeffsPixel.sigma, 0.0)
        self.assertAlmostEqual(measuredLines[0].gaussCoeffsPixel.strength,
                               0.0)
        self.assertAlmostEqual(measuredLines[0].eGaussCoeffsPixel.strength, 0.0)

        self.assertAlmostEqual(measuredLines[1].pixelPosPredicted,
                               lineB.laboratoryWavelength)
        self.assertAlmostEqual(measuredLines[1].gaussCoeffsPixel.mu,
                               lineB.laboratoryWavelength)
        self.assertAlmostEqual(measuredLines[1].eGaussCoeffsPixel.mu, 0.0)
        self.assertAlmostEqual(measuredLines[1].gaussCoeffsPixel.sigma,
                               sigma)
        self.assertAlmostEqual(measuredLines[1].eGaussCoeffsPixel.sigma, 0.0)
        self.assertAlmostEqual(measuredLines[1].gaussCoeffsPixel.strength,
                               lineB.predictedStrength)
        self.assertAlmostEqual(measuredLines[1].eGaussCoeffsPixel.strength, 0.0)

    def testRemoveBadLines(self):
        logger = log.Log.getLogger("measureLinesInPixelSpace")
        logger.setLevel(log.WARN)
        logger = log.Log.getLogger("removeBadLines")
        logger.setLevel(log.WARN)

        nistLines = []

        lineA = drpStella.NistLine()
        lineA.laboratoryWavelength = 5.0
        lineA.predictedStrength = 5.0
        nistLines.append(lineA)

        lineB = drpStella.NistLine()
        lineB.laboratoryWavelength = 20.0
        lineB.predictedStrength = 20.0
        nistLines.append(lineB)

        lineC = drpStella.NistLine()
        lineC.laboratoryWavelength = 6.0
        lineC.predictedStrength = 1.0
        nistLines.append(lineC)

        lineD = drpStella.NistLine()
        lineD.laboratoryWavelength = 25.0
        lineD.predictedStrength = 20.0
        nistLines.append(lineD)

        lineE = drpStella.NistLine()
        lineE.laboratoryWavelength = 20.2
        lineE.predictedStrength = .1
        nistLines.append(lineE)

        lambdaPix = range(30)
        fluxPix = np.ndarray(shape=(30), dtype=np.float32)
        fluxPix[:] = 0.0

        sigma = 1.0
        fluxPix += self.getGaussian(lineA, sigma, lambdaPix)
        fluxPix += self.getGaussian(lineB, sigma, lambdaPix)
        fluxPix += self.getGaussian(lineC, sigma, lambdaPix)
        fluxPix += self.getGaussian(lineD, sigma, lambdaPix)
        fluxPix += self.getGaussian(lineE, sigma, lambdaPix)

        # fit all lines by not setting minStrength
        nistLinesMeas = measureLinesInPixelSpace(nistLines,
                                                 lambdaPix,
                                                 fluxPix,
                                                 2.34)
        self.assertEqual(len(nistLines), len(nistLinesMeas))

        goodLines = removeBadLines(lines = nistLinesMeas,
                                   fluxPix = fluxPix,
                                   plot = False,
                                   fwhm = 2.355,
                                   minDistance = 2.0,
                                   maxDistance = 1.0,
                                   minStrength = 1.0,
                                   minRatio = 100.0)

        self.assertEqual(len(goodLines), 2)
        self.assertAlmostEqual(goodLines[0].nistLine.laboratoryWavelength,
                               lineB.laboratoryWavelength)
        self.assertAlmostEqual(goodLines[1].nistLine.laboratoryWavelength,
                               lineD.laboratoryWavelength)
        self.assertEqual(nistLinesMeas[0].flags, 'b')
        self.assertEqual(nistLinesMeas[1].flags, 'g')
        self.assertEqual(nistLinesMeas[2].flags, 'b')
        self.assertEqual(nistLinesMeas[3].flags, 'g')
        self.assertEqual(nistLinesMeas[4].flags, 'b')

    def testMeasureWavelengths(self):
        lines = []

        nistLineA = drpStella.NistLine()
        nistLineA.laboratoryWavelength = 10.0
        nistLineA.predictedStrength = 10.0
        lineA = drpStella.NistLineMeas()
        lineA.nistLine = nistLineA
        lines.append(lineA)
        self.assertAlmostEqual(nistLineA.laboratoryWavelength,
                               lines[0].nistLine.laboratoryWavelength)
        self.assertAlmostEqual(nistLineA.predictedStrength,
                               lines[0].nistLine.predictedStrength)

        nistLineB = drpStella.NistLine()
        nistLineB.laboratoryWavelength = 20.0
        nistLineB.predictedStrength = 20.0
        lineB = drpStella.NistLineMeas()
        lineB.nistLine = nistLineB
        lines.append(lineB)
        self.assertAlmostEqual(nistLineB.laboratoryWavelength,
                               lines[1].nistLine.laboratoryWavelength)
        self.assertAlmostEqual(nistLineB.predictedStrength,
                               lines[1].nistLine.predictedStrength)
        self.assertNotAlmostEqual(lines[0].nistLine.laboratoryWavelength,
                                  lines[1].nistLine.laboratoryWavelength)

        lambdaPix = range(30)
        fluxPix = np.ndarray(shape=(30), dtype=np.float32)
        fluxPix[:] = 0.0

        sigma = 1.0
        fluxPix += self.getGaussian(nistLineA, sigma, lambdaPix)
        fluxPix += self.getGaussian(nistLineB, sigma, lambdaPix)

        measureLinesInWavelengthSpace(lines, fluxPix, lambdaPix, sigma)
        self.assertAlmostEqual(lines[0].gaussCoeffsLambda.mu,
                               nistLineA.laboratoryWavelength)
        self.assertAlmostEqual(lines[0].gaussCoeffsLambda.strength,
                               nistLineA.predictedStrength)
        self.assertAlmostEqual(lines[0].gaussCoeffsLambda.sigma,
                               sigma)
        self.assertAlmostEqual(lines[0].eGaussCoeffsLambda.mu,
                               0.0)
        self.assertAlmostEqual(lines[0].eGaussCoeffsLambda.strength,
                               0.0)
        self.assertAlmostEqual(lines[0].eGaussCoeffsLambda.sigma,
                               0.0)

        self.assertAlmostEqual(lines[1].gaussCoeffsLambda.mu,
                               nistLineB.laboratoryWavelength)
        self.assertAlmostEqual(lines[1].gaussCoeffsLambda.strength,
                               nistLineB.predictedStrength)
        self.assertAlmostEqual(lines[1].gaussCoeffsLambda.sigma,
                               sigma)
        self.assertAlmostEqual(lines[1].eGaussCoeffsLambda.mu,
                               0.0)
        self.assertAlmostEqual(lines[1].eGaussCoeffsLambda.strength,
                               0.0)
        self.assertAlmostEqual(lines[1].eGaussCoeffsLambda.sigma,
                               0.0)

    def getGaussian(self, line, sigma, x):
        """
        Calculate the Gaussian according to line.laboratoryWavelength,
        line.predictedStrength, and sigma for x
        @param line : NistLine with laboratoryWavelength and predictedStrength
        @param sigma : sigma for the Gaussian
        @param x : x values for which to calculate the Gaussian
        """
        gaussCoeffs = drpStella.GaussCoeffs()
        gaussCoeffs.mu = line.laboratoryWavelength
        gaussCoeffs.sigma = sigma
        gaussCoeffs.strength = line.predictedStrength
        gaussian = gauss(x, gaussCoeffs)
        return gaussian

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(LinesTestCase)
    return unittest.TestSuite(suites)

def run(exit = False):
    """Run the tests"""
    tests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
