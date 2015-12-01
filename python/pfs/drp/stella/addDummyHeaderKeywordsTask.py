#!/usr/bin/env python

#USAGE: exp = lsst.afw.image.ExposureF("/home/azuri/spectra/pfs/2014-10-14/IR-23-0-sampledFlatx2-nonoise.fits")
#       myFindTask = findAndTraceAperturesTask.FindAndTraceAperturesTask()
#       fts = myFindTask.run(exp)
#       myExtractTask = createFlatFiberTraceProfileTask.CreateFlatFiberTraceProfileTask()
#       myExtractTask.run(fts)

#import os
#import math
#import numpy

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab

from pfs.drp.stella.readFileTask import ReadFileTask
from astropy.io import fits as pyfits
from lsst.pipe.base import Task
import lsst.pex.config as pexConfig

class AddDummyHeaderKeywordsConfig(pexConfig.Config):
        editHeader = pexConfig.Field(
            doc = "Do edit Header?",
            dtype = bool,
            default = False)

class AddDummyHeaderKeywordsTask(Task):
    ConfigClass = AddDummyHeaderKeywordsConfig
    _DefaultName = "addDummyHeaderKeywordsTask"

    def __init__(self, *args, **kwargs):
        super(AddDummyHeaderKeywordsTask, self).__init__(*args, **kwargs)

    def addDummyHeaderKeywords(self, fileList):

        """ read fileList """
        readFileTask = ReadFileTask()
        fitsList = readFileTask.run(fileList)
        for fitsFile in fitsList:
            fitsFile = fitsFile[0]
            dateObs = fitsFile[fitsFile.find('2015'):fitsFile.find('/PFSA')]
            print 'dateObs = <',dateObs,'>'
            frameID = fitsFile[fitsFile.find('PFSA'):fitsFile.find('.fits')]
            spectrograph = 0
            dewar = 1
            print 'frameID = <',frameID,'>'
            print 'fitsFile = <',fitsFile,'>'
            hduList = pyfits.open(fitsFile, mode='update')
            #print 'dir(hduList) = ',dir(hduList)
            #print 'len(hduList) = ',len(hduList)
            header = hduList[0].header
            #print 'header = ',header

            if 'EXP-ID' in header:
                print 'EXP-ID = <',header['EXP-ID'],'>'
            else:
                if self.config.editHeader:
                    header['EXP-ID'] = 'EXP-ID'

            if 'IMAGETYP' in header:
                imageType = header['IMAGETYP']
                print 'imageType = <',imageType,'>'
                if self.config.editHeader:
                    header['DATA-TYP'] = imageType

            if 'INST-PA' in header:
                print 'INST-PA = <',header['INST-PA'],'>'
        #    else:
            if self.config.editHeader:
                header['INST-PA'] = '0.0'

            if 'FRAMEID' in header:
                print 'FRAMEID = <',header['FRAMEID'],'>'
            if self.config.editHeader:
                header['FRAMEID'] = frameID

            if 'DET-ID' in header:
                print 'DET-ID = <',header['DET-ID'],'>'
            if self.config.editHeader:
                header['DET-ID'] = '0'

            if 'PROP-ID' in header:
                print 'PROP-ID = <',header['PROP-ID'],'>'
        #    else:
            if self.config.editHeader:
                header['PROP-ID'] = dateObs
                header['VISIT'] = dateObs

            if 'T_CFGFIL' in header:
                print 'T_CFGFIL = <',header['T_CFGFIL'],'>'
            else:
                if self.config.editHeader:
                    header['T_CFGFIL'] = 'T_CFGFIL'

            if 'T_AG' in header:
                print 'T_AG = <',header['T_AG'],'>'
        #    else:
            if self.config.editHeader:
                header['T_AG'] = 'FALSE'

            hduList.close()

        return None

    def run(self, fileList):
        self.addDummyHeaderKeywords(fileList)
        
        return None
