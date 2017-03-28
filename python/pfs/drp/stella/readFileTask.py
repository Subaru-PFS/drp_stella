#!/usr/bin/env python

#USAGE: exp = lsst.afw.image.ExposureF("/home/azuri/spectra/pfs/2014-10-14/IR-23-0-sampledFlatx2-nonoise.fits")
#       myFindTask = findAndTraceAperturesTask.FindAndTraceAperturesTask()
#       fts = myFindTask.run(exp)
#       myExtractTask = createFlatFiberTraceProfileTask.CreateFlatFiberTraceProfileTask()
#       myExtractTask.run(fts)

import numpy as np
from lsst.pipe.base import Task
import lsst.pex.config as pexConfig

class ReadFileConfig(pexConfig.Config):
        dType = pexConfig.Field(
            doc = "data type to be read",
            dtype = str,
            default = "S120")

class ReadFileTask(Task):
    ConfigClass = ReadFileConfig
    _DefaultName = "readFileTask"

    def __init__(self, *args, **kwargs):
        super(ReadFileTask, self).__init__(*args, **kwargs)

    def countLines(self, fileToRead):
        """ count lines """
        nLines = 0
        nDataLines = 0
        with open(fileToRead, 'r') as inFile:
            while inFile.readline() != '':
                nLines = nLines + 1
        inFile.closed
        nDataLines = nLines

        """ remove comment lines from nDataLines """
        with open(fileToRead, 'r') as inFile:
            for iLine in np.arange(0,nLines,1):
                line = inFile.readline()
                if len(line) == 0:
                    nDataLines = nDataLines - 1
                else:
                    if line[0] == '#':
                        nDataLines = nDataLines - 1
        return nDataLines

    def countCols(self, fileToRead, nLines):
        nCols = 0
        with open( fileToRead, 'r' ) as inFile:
            for iLine in np.arange( 0, nLines, 1 ):
                line = inFile.readline()
                if len(line) > 0:
                    if line[0] != '#':
                        words = line.split()
                        if len(words) > nCols:
                            nCols = len(words)
        inFile.closed
        return nCols

    def readFile(self, fileToRead):
        nDataLines = self.countLines(fileToRead)
        print 'readFile: nDataLines = ',nDataLines

        nCols = self.countCols(fileToRead, nDataLines)
        print 'readFile: nCols = ',nCols

        """ read file to dataArr """
        print 'readFile: dType = <',self.config.dType,'>'
        print 'readFile: type(nDataLines) = ',type(nDataLines)
        print 'readFile: type(nCols) = ',type(nCols)
        dataArr = np.ndarray(shape=(int(nDataLines), int(nCols)), dtype=self.config.dType)
        print 'readFile: dataArr = ',dataArr
        iDataLine = 0
        with open(fileToRead, 'r') as inFile:
            for iLine in np.arange(0,nDataLines,1):
                line = inFile.readline()
                if len(line) > 0:
                    if line[0] != '#':
                        words = line.split()
                        for iWord in range(len(words)):
                            if self.config.dType == "float":
                                dataArr[iDataLine, iWord] = float(words[iWord])
                            else: # dType == "string"
                                dataArr[iDataLine, iWord] = words[iWord]
        #                    print 'readFile: dataArr[',iDataLine,', ',iWord,'] = <',dataArr[iDataLine, iWord],'>'
                        iDataLine = iDataLine + 1
        inFile.closed

        return dataArr

    def run(self, fileToRead):

        dataArr = self.readFile(fileToRead)

        return dataArr
