#!/usr/bin/env python
from astropy.io import fits as pyfits

import lsst.pex.config as pexConfig
from lsst.pipe.base import Task
from pfs.drp.stella.readFileTask import ReadFileTask

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
        # read fileList
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
            header = hduList[0].header

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
            if self.config.editHeader:
                header['T_AG'] = 'FALSE'

            hduList.close()

        return None

    def run(self, fileList):
        self.addDummyHeaderKeywords(fileList)

        return None
