#!/usr/bin/env python

import os
import sys
import math
import numpy
import argparse
import traceback
import time
import shutil
import glob

from lsst.pex.config import Config, ConfigField, ConfigurableField, Field, ListField
from lsst.pipe.base import Task, Struct, TaskRunner, ArgumentParser
import lsst.daf.base as dafBase
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDet
import lsst.afw.image as afwImage
import lsst.afw.cameraGeom as cameraGeom
import lsst.meas.algorithms as measAlg
import lsst.afw.geom.ellipses as afwEll
from lsst.pipe.tasks.repair import RepairTask

import lsst.obs.subaru.isr as lsstIsr

#import hsc.pipe.base.butler as hscButler
import lsst.daf.persistence.butler as lsstButler

#from hsc.pipe.base.parallel import BatchPoolTask
from pfs.drp.stella.parallel import BatchPoolTask
#from lsst.pipe.base import Task

from pfs.drp.stella.pool import Pool, NODE

from . import checksum

class DetrendStatsConfig(Config):
    """Parameters controlling background statistics"""
    stat = Field(doc="Statistic to use to estimate background (from lsst.afw.math)", dtype=int,
                   default=afwMath.MEANCLIP)
    clip = Field(doc="Clipping threshold for background", dtype=float, default=3.0)
    iter = Field(doc="Clipping iterations for background", dtype=int, default=3)

class DetrendStatsTask(Task):
    """Measure statistic of the background"""
    ConfigClass = DetrendStatsConfig

    def run(self, exposureOrImage):
        """Measure a particular statistic on an image (of some sort).

        @param exposureOrImage    Exposure, MaskedImage or Image.
        @return Value of desired statistic
        """
        stats = afwMath.StatisticsControl(self.config.clip, self.config.iter,
                                          afwImage.MaskU.getPlaneBitMask("DETECTED"))
        try:
            image = exposureOrImage.getMaskedImage()
        except:
            try:
                image = exposureOrImage.getImage()
            except:
                image = exposureOrImage

        return afwMath.makeStatistics(image, self.config.stat, stats).getValue()


class DetrendCombineConfig(Config):
    """Configuration for combining detrend images"""
    rows = Field(doc="Number of rows to read at a time", dtype=int, default=512)
    mask = ListField(doc="Mask planes to respect", dtype=str, default=["SAT", "DETECTED", "INTRP"])
    combine = Field(doc="Statistic to use for combination (from lsst.afw.math)", dtype=int,
                    default=afwMath.MEANCLIP)
    clip = Field(doc="Clipping threshold for combination", dtype=float, default=3.0)
    iter = Field(doc="Clipping iterations for combination", dtype=int, default=3)
    stats = ConfigurableField(target=DetrendStatsTask, doc="Background statistics configuration")

class DetrendCombineTask(Task):
    """Task to combine detrend images"""
    ConfigClass = DetrendCombineConfig

    def __init__(self, *args, **kwargs):
        super(DetrendCombineTask, self).__init__(*args, **kwargs)
        self.makeSubtask("stats")

    def run(self, sensorRefList, expScales=None, finalScale=None, inputName="postISRCCD"):
        """Combine detrend images for a single sensor

        @param sensorRefList   List of data references to combine (for a single sensor)
        @param expScales       List of scales to apply for each exposure
        @param finalScale      Desired scale for final combined image
        @param inputName       Data set name for inputs
        @return combined image
        """
        width, height = self.getDimensions(sensorRefList)
        maskVal = 0
        for mask in self.config.mask:
            maskVal |= afwImage.MaskU.getPlaneBitMask(mask)
        stats = afwMath.StatisticsControl(self.config.clip, self.config.iter, maskVal)

        # Combine images
        combined = afwImage.ImageF(width, height)
        numImages = len(sensorRefList)
        imageList = [None]*numImages
        for start in range(0, height, self.config.rows):
            rows = min(self.config.rows, height - start)
            box = afwGeom.Box2I(afwGeom.Point2I(0, start), afwGeom.Extent2I(width, rows))
            subCombined = combined.Factory(combined, box)

            for i, sensorRef in enumerate(sensorRefList):
                if sensorRef is None:
                    imageList[i] = None
                    continue
                exposure = sensorRef.get(inputName + "_sub", bbox=box)
                if expScales is not None:
                    self.applyScale(exposure, expScales[i])
                imageList[i] = exposure.getMaskedImage()

            self.combine(subCombined, imageList, stats)

        if finalScale is not None:
            background = self.stats.run(combined)
            self.log.info("%s: Measured background of stack is %f; adjusting to %f" %
                         (NODE, background, finalScale))
            combined *= finalScale / background

        return afwImage.DecoratedImageF(combined)

    def getDimensions(self, sensorRefList, inputName="postISRCCD"):
        """Get dimensions of the inputs"""
        print 'DetrendCombineTask: sensorRefList = ',sensorRefList
        print 'DetrendCombineTask: dir(sensorRefList) = ',dir(sensorRefList)
        print 'DetrendCombineTask: len(sensorRefList) = ',len(sensorRefList)
        dimList = []
        for sensorRef in sensorRefList:
            if sensorRef is None:
                continue
            md = sensorRef.get(inputName + "_md")
            print 'DetrendCombineTask: md = ',md
#            dimList.append(afwGeom.Extent2I(md.get("NAXIS1"), md.get("NAXIS2")))
            return [md.get("NAXIS1"), md.get("NAXIS2")]
        print 'DetrendCombineTask: type(dimList) = ',type(dimList)
        print 'DetrendCombineTask: dir(dimList) = ',dir(dimList)
        print 'DetrendCombineTask: dimList = ',dimList
        return getSize(dimList)#[height, width]

    def applyScale(self, exposure, scale=None):
        """Apply scale to input exposure"""
        if scale is not None:
            mi = exposure.getMaskedImage()
            mi /= scale

    def combine(self, target, imageList, stats):
        """Combine multiple images

        @param target      Target image to receive the combined pixels
        @param imageList   List of input images
        @param stats       Statistics control
        """
        imageList = afwImage.vectorMaskedImageF([image for image in imageList if image is not None])
        if False:
#        if True:
            # In-place stacks are now supported on LSST's afw, but not yet on HSC
            afwMath.statisticsStack(target, imageList, self.config.combine, stats)
        else:
            stack = afwMath.statisticsStack(imageList, self.config.combine, stats)
            target <<= stack.getImage()


def getCcdName(ccdId, ccdKeys):
    """Return the 'CCD name' from the data identifier

    The 'CCD name' is a tuple of the values in the data identifier
    that identify the CCD.  The names in the data identifier that
    identify the CCD is provided as 'ccdKeys'.

    @param ccdId    Data identifier for CCD
    @param ccdKeys  Data identifier keys for the 'sensor' level
    @return ccd name
    """
    return tuple(ccdId[k] for k in ccdKeys)

def getCcdIdListFromExposures(expRefList, level="sensor"):
    """Determine a list of CCDs from exposure references

    This essentially inverts the exposure-level references (which
    provides a list of CCDs for each exposure), by providing
    a set of keywords that identify a CCD in the dataId, and a
    dataId list for each CCD.  Consider an input list of exposures
    [e1, e2, e3], and each exposure has CCDs c1 and c2.  Then this
    function returns:

        set(['ccd']),
        {(c1,): [e1c1, e2c1, e3c1], (c2,): [e1c2, e2c2, e3c2]}

    The latter part is a dict whose keys are tuples of the identifying
    values of a CCD (usually just the CCD number) and the values are
    lists of dataIds for that CCD in each exposure.  A missing dataId
    is given the value None.

    @param expRefList   List of data references for exposures
    @param level        Level for the butler to generate CCDs
    @return CCD keywords, dict of data identifier lists for each CCD
    """
    expIdList = [[ccdRef.dataId for ccdRef in expRef.subItems(level)] for expRef in expRefList]

    # Determine what additional keys make a CCD from an exposure
    ccdKeys = set() # Set of keywords in the dataId that identify a CCD
    ccdNames = set() # Set of tuples which are values for each of the CCDs in an exposure
    for ccdIdList, expRef in zip(expIdList, expRefList):
        expKeys = set(expRef.dataId.keys())
        for ccdId in ccdIdList:
            keys = set(ccdId.keys()).difference(expKeys)
            if len(ccdKeys) == 0:
                ccdKeys = keys
            elif keys != ccdKeys:
                raise RuntimeError("Keys for CCD differ: %s vs %s" % (keys, ccdList.keys()))
            name = getCcdName(ccdId, ccdKeys)
            ccdNames.add(name)

    # if ccdKeys is an empty set, then ccdKeys == expKeys and expRefList was split at the 'sensor' level,
    # so we'll set ccdKeys ourselves to avoid later confusion
    isExpList = True
    if len(ccdKeys) == 0:
        ccdKeys = set(['ccd'])
        isExpList = False

    # Turn the list of CCDs for each exposure into a list of exposures for each CCD
    ccdLists = {}
    for n, ccdIdList in enumerate(expIdList):
        for ccdId in ccdIdList:
            name = getCcdName(ccdId, ccdKeys)
            if name not in ccdLists:
                ccdLists[name] = []
            ccdLists[name].append(ccdId)
        # 'None' padding only makes sense if it's really an exposureId list.
        if isExpList:
            for idList in ccdLists.values():
                if len(idList) == n:
                    idList.append(None)

    return ccdKeys, ccdLists

class DetrendIdAction(argparse.Action):
    """Split name=value pairs and put the result in a dict"""
    def __call__(self, parser, namespace, values, option_string):
        output = getattr(namespace, self.dest, {})
        for nameValue in values:
            name, sep, valueStr = nameValue.partition("=")
            if not valueStr:
                parser.error("%s value %s must be in form name=value" % (option_string, nameValue))
            output[name] = valueStr
        setattr(namespace, self.dest, output)

class DetrendArgumentParser(ArgumentParser):
    """Add a --detrendId argument to the argument parser"""
    def __init__(self, calibName, *args, **kwargs):
        super(DetrendArgumentParser, self).__init__(*args, **kwargs)
        self.calibName = calibName
        self.add_id_argument("--id", datasetType="raw",
                             help="input identifiers, e.g., --id visit=123 ccd=4")
        self.add_argument("--detrendId", nargs="*", action=DetrendIdAction, default={},
                          help="identifiers for detrend, e.g., --detrendId version=1",
                          metavar="KEY=VALUE1[^VALUE2[^VALUE3...]")
    def parse_args(self, *args, **kwargs):
        namespace = super(DetrendArgumentParser, self).parse_args(*args, **kwargs)
#        self.log.info('DetrendArgumentParser.parse_args: namespace = %s'%(namespace))
        keys = namespace.butler.getKeys(self.calibName)
#        self.log.info('DetrendArgumentParser.parse_args: keys = %s'%(keys))
        parsed = {}
        for name, value in namespace.detrendId.items():
            if not name in keys:
                self.error("%s is not a relevant detrend identifier key (%s)" % (name, keys))
            parsed[name] = keys[name](value)
#        parsed['category'] = keys['category'](value)
#        parsed['site'] = keys['category'](value)
#        parsed['filter'] = keys['filter'](value)
#        self.log.info('DetrendArgumentParser.parse_args: parsed = %s'%(parsed))
        namespace.detrendId = parsed

        return namespace

class DetrendConfig(Config):
    """Configuration for constructing detrends"""
    clobber = Field(dtype=bool, default=True, doc="Clobber existing processed images?")
    isr = ConfigurableField(target=lsstIsr.SubaruIsrTask, doc="ISR configuration")
    dateObs = Field(dtype=str, default="dateObs", doc="Key for observation date in exposure registry")
    dateCalib = Field(dtype=str, default="calibDate", doc="Key for detrend date in calib registry")
    filter = Field(dtype=str, default="filter", doc="Key for filter name in exposure/calib registries")
    combination = ConfigurableField(target=DetrendCombineTask, doc="Detrend combination configuration")
    def setDefaults(self):
        self.isr.doWrite = False

class DetrendTaskRunner(TaskRunner):
    """Get parsed values into the DetrendTask.run"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [dict(expRefList=parsedCmd.id.refList, butler=parsedCmd.butler, detrendId=parsedCmd.detrendId)]

    def __call__(self, args):
        task = self.TaskClass(config=self.config, log=self.log)
        if self.doRaise:
            result = task.run(**args)
        else:
            try:
                result = task.run(**args)
            except Exception, e:
                task.log.fatal("Failed: %s" % e)
                traceback.print_exc(file=sys.stderr)

        if self.doReturnResults:
            return Struct(
                args = args,
                metadata = task.metadata,
                result = result,
            )

class DetrendTask(BatchPoolTask):
    """Base class for constructing detrends.

    This should be subclassed for each of the required detrend types.
    The subclass should be sure to define the following class variables:
    * _DefaultName: default name of the task, used by CmdLineTask
    * calibName: name of the calibration data set in the butler
    * overrides: a list of functions for setting a configuration, used by CmdLineTask
    """
    ConfigClass = DetrendConfig
    RunnerClass = DetrendTaskRunner
    FilterName = None

    def __init__(self, *args, **kwargs):
        """Constructor.

        All nodes execute this method.
        """
        super(DetrendTask, self).__init__(*args, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("combination")

    @classmethod
    def batchWallTime(cls, time, parsedCmd, numNodes, numProcs):
        numCcds = 1#sum(1 for raft in parsedCmd.butler.get("camera") for ccd in cameraGeom.cast_Raft(raft))
        numExps = len(cls.RunnerClass.getTargetList(parsedCmd)[0]['expRefList'])
        numCycles = int(numCcds/float(numNodes*numProcs) + 0.5)
        return time*numExps*numCycles

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        doBatch = kwargs.pop("doBatch", False)
        return DetrendArgumentParser(calibName=cls.calibName, name=cls._DefaultName, *args, **kwargs)

    def run(self, expRefList, butler, detrendId):
        """Construct a detrend from a list of exposure references

        This is the entry point, called by the TaskRunner.__call__

        All nodes execute this method.

        @param expRefList  List of data references at the exposure level
        @param butler      Data butler
        @param detrendId   Identifier dict for detrend
        """
        
        self.log.info('DetrendTask.run: expRefList = %s'%(expRefList))
        self.log.info('DetrendTask.run: detrendId = %s'%(detrendId))
        outputId = self.getOutputId(expRefList, detrendId)
        ccdKeys, ccdIdLists = getCcdIdListFromExposures(expRefList, level="sensor")
        self.log.info('DetrendTask.run: outputId = %s'%(outputId))
        self.log.info('DetrendTask.run: ccdKeys = %s'%(ccdKeys))
        self.log.info('DetrendTask.run: ccdIdLists = %s'%(ccdIdLists))

        # Ensure we can generate filenames for each output
        for ccdName in ccdIdLists:
            self.log.info('ccdName = %s' %(ccdName))
            dataId = dict(outputId.items() + [(k, ccdName[i]) for i, k in enumerate(ccdKeys)])
            self.log.info('DetrendTask.run: dataId = %s'%(dataId))
            try:
                filename = butler.get(self.calibName + "_filename", dataId)
                self.log.info('DetrendTask.run: filename = %s'%(filename))
            except Exception, e:
                raise RuntimeError("Unable to determine output filename from %s: %s" % (dataId, e))

        self.copyConfig(butler, dataId)

        pool = Pool()
        pool.storeSet(butler=butler)

        # Scatter: process CCDs independently
        data = self.scatterProcess(pool, ccdKeys, ccdIdLists)

        # Gather: determine scalings
        scales = self.scale(ccdKeys, ccdIdLists, data)

        # Scatter: combine
        self.scatterCombine(pool, outputId, ccdKeys, ccdIdLists, scales)

    def getOutputId(self, expRefList, detrendId):
        """Generate the data identifier for the output detrend

        The mean date and the common filter are included, using keywords
        from the configuration.  The CCD-specific part is not included
        in the data identifier.

        Only the root node executes this method (it will share the results with the slaves).

        @param expRefList  List of data references at exposure level
        """
        self.log.info('DetrendTask.getOutputId: expRefList = %s' % expRefList)
        self.log.info('DetrendTask.getOutputId: detrendId = %s' % detrendId)
        expIdList = [expRef.dataId for expRef in expRefList]
        self.log.info('DetrendTask.getOutputId: expIdList = %s' % expIdList)
        midTime = 0
        filterName = None
        self.log.info('DetrendTask.getOutputId: filterName set to %s' % filterName)
        for expId in expIdList:
            self.log.info('DetrendTask.getOutputId: expId = %s' % expId)
            midTime += self.getMjd(expId)
            self.log.info('DetrendTask.getOutputId: midTime = %g' % midTime)
            self.log.info('DetrendTask.getOutputId: self.FilterName = %s' % self.FilterName)
            self.log.info('DetrendTask.getOutputId: self.getFilter(expId) = %s' % self.getFilter(expId))
            if self.FilterName is None:
                self.log.info('DetrendTask.getOutputId: self.FilterName = %s is None' % self.FilterName)
            else:
                self.log.info('DetrendTask.getOutputId: self.FilterName = %s is not None' % self.FilterName)
            thisFilter = self.getFilter(expId) if self.FilterName is None else self.FilterName
            self.log.info('DetrendTask.getOutputId: thisFilter = %s' % thisFilter)
            if filterName is None:
                filterName = thisFilter
            elif filterName != thisFilter:
                raise RuntimeError("Filter mismatch for %s: %s vs %s" % (expId, thisFilter, filterName))
            self.log.info('DetrendTask.getOutputId: filterName = %s' % filterName)

        self.log.info('DetrendTask.getOutputId: len(expRefList) = %d' % len(expRefList))
        midTime /= len(expRefList)
        self.log.info('DetrendTask.getOutputId: midTime = %g' % midTime)
        date = str(dafBase.DateTime(midTime, dafBase.DateTime.MJD).toPython().date())
        self.log.info('DetrendTask.getOutputId: date = %s' % date)

        outputId = {self.config.filter: filterName, self.config.dateCalib: date}
        self.log.info('DetrendTask.getOutputId: outputId = %s' % outputId)
        outputId.update(detrendId)
        self.log.info('DetrendTask.getOutputId: returning outputId = %s' % outputId)
        return outputId

    def getMjd(self, dataId):
        """Determine the Modified Julian Date (MJD) from a data identifier"""
        dateObs = dataId[self.config.dateObs]
        try:
            dt = dafBase.DateTime(dateObs)
        except:
            dt = dafBase.DateTime(dateObs + "T12:00:00.0Z")
        return dt.get(dafBase.DateTime.MJD)

    def getFilter(self, dataId):
        """Determine the filter from a data identifier"""
        return dataId[self.config.filter]

    def scatterProcess(self, pool, ccdKeys, ccdIdLists):
        """Scatter the processing among the nodes

        We scatter the data wider than the just the number of CCDs, to make
        full use of all available processors.  This necessitates piecing
        everything back together in the same format as ccdIdLists afterwards.

        All nodes execute this method (though with different behaviour).

        @param ccdKeys     Keywords that identify a CCD in the data identifier
        @param ccdIdLists  Dict of data identifier lists for each CCD name
        @return Dict of lists of returned data for each CCD name
        """
        dataIdList = sum(ccdIdLists.values(), [])
        self.log.info("Scatter processing")

        resultList = pool.map(self.process, dataIdList)

        # Piece everything back together
        data = dict((ccdName, [None] * len(expList)) for ccdName, expList in ccdIdLists.items())
        indices = dict(sum([[(tuple(dataId.values()) if dataId is not None else None, (ccdName, expNum))
                             for expNum, dataId in enumerate(expList)]
                            for ccdName, expList in ccdIdLists.items()], []))
        for dataId, result in zip(dataIdList, resultList):
            if dataId is None:
                continue
            ccdName, expNum = indices[tuple(dataId.values())]
            data[ccdName][expNum] = result

        return data

    def process(self, cache, ccdId, outputName="postISRCCD"):
        """Process a CCD, specified by a data identifier

        Only slave nodes execute this method.
        """
        self.log.info("ccdId = %s" % ccdId)
        self.log.info("type(ccdId) = %s" % type(ccdId))
        self.log.info("dir(ccdId) = %s" % dir(ccdId))
        if ccdId is None:
            self.log.warn("Null identifier received on %s" % NODE)
            return None
        self.log.info("Processing %s on %s" % (ccdId, NODE))
#        sensorRef = lsstButler.getDataRef(cache.butler, ccdId)
        sensorRef = getDataRef(cache.butler, ccdId)
        self.log.info('process: sensorRef = %s' % sensorRef)
        self.log.info('process: dir(sensorRef) = %s' % dir(sensorRef))
        self.log.info('process: sensorRef.dataId = %s' % sensorRef.dataId)
        self.log.info('process: sensorRef.get() = %s' % sensorRef.get())
        if self.config.clobber or not sensorRef.datasetExists(outputName):
            try:
                exposure = self.processSingle(sensorRef)
            except Exception as e:
                print 'exception e = <',e,'>'
                self.log.warn("Unable to process %s: %s" % (ccdId, e))
                return None
            self.processWrite(sensorRef, exposure)
        else:
            exposure = sensorRef.get(outputName, immediate=True)
        return self.processResult(exposure)

    def processSingle(self, dataRef):
        """Process a single CCD, specified by a data reference

        Only slave nodes execute this method.
        """
        return self.isr.run(dataRef.get("raw")).exposure

    def processWrite(self, dataRef, exposure, outputName="postISRCCD"):
        """Write the processed CCD

        Only slave nodes execute this method.

        @param dataRef     Data reference
        @param exposure    CCD exposure to write
        @param outputName  Data type name for butler.
        """
        dataRef.put(exposure, outputName)

    def processResult(self, exposure):
        """Extract processing results from a processed exposure

        Only slave nodes execute this method.  This method generates
        what is gathered by the master node --- it must be picklable!
        """
        return None

    def scale(self, ccdKeys, ccdIdLists, data):
        """Determine scaling across CCDs and exposures

        This is necessary mainly for flats, so as to determine a
        consistent scaling across the entire focal plane.

        Only the master node executes this method.

        @param ccdKeys     Keywords that identify a CCD in the data identifier
        @param ccdIdLists  Dict of data identifier lists for each CCD name
        @param data        Dict of lists of returned data for each CCD name
        @return dict of Struct(ccdScale: scaling for CCD,
                               expScales: scaling for each exposure
                               ) for each CCD name
        """
        self.log.info("Scale on %s" % NODE)
        return dict((name, Struct(ccdScale=None, expScales=[None] * len(ccdIdLists[name])))
                    for name in ccdIdLists.keys())

    def scatterCombine(self, pool, outputId, ccdKeys, ccdIdLists, scales):
        """Scatter the combination across multiple nodes

        In this case, we can only scatter across as many nodes as
        there are CCDs.

        All nodes execute this method (though with different behaviour).

        @param outputId    Output identifier (exposure part only)
        @param ccdKeys     Keywords that identify a CCD in the data identifier
        @param ccdIdLists  Dict of data identifier lists for each CCD name
        @param scales      Dict of structs with scales, for each CCD name
        """
        self.log.info("Scatter combination")
        data = [Struct(ccdIdList=ccdIdLists[ccdName], scales=scales[ccdName],
                       outputId=dict(outputId.items() + [(k,ccdName[i]) for i, k in enumerate(ccdKeys)]))
                for ccdName in ccdIdLists.keys()]
        pool.map(self.combine, data)

    def combine(self, cache, struct):
        """Combine multiple exposures of a particular CCD and write the output

        Only the slave nodes execute this method.

        The input is a struct consisting of the following components:
        @param ccdIdList   List of data identifiers for combination
        @param scales      Scales to apply (expScales are scalings for each exposure,
                           ccdScale is final scale for combined image)
        @param outputId    Data identifier for combined image (fully qualified for this CCD)
        """
#        dataRefList = [lsstButler.getDataRef(cache.butler, dataId) if dataId is not None else None for
        dataRefList = [getDataRef(cache.butler, dataId) if dataId is not None else None for
                       dataId in struct.ccdIdList]
        self.log.info("Combining %s on %s" % (struct.outputId, NODE))
        detrend = self.combination.run(dataRefList, expScales=struct.scales.expScales,
                                       finalScale=struct.scales.ccdScale)

        self.recordCalibInputs(cache.butler, detrend, struct.ccdIdList, struct.outputId)

        self.maskNans(detrend)
        self.log.info("writing detrend=<%s>, struct.outputId=<%s>"%(detrend, struct.outputId))
        self.write(cache.butler, detrend, struct.outputId)


    def recordCalibInputs(self, butler, detrend, dataIdList, outputId):
        """Record metadata (FITS header) for input visits, date, time, host, and directory.

        @param detrend     The combined detrend exposure.
        @param dataIdList  List of data identifiers for calibration inputs
        """

        md = detrend.getMetadata()

        # date, time, host, and root
        now = time.localtime()
        md.add("COMBINE_DATE", time.strftime("%Y-%m-%d", now))
        md.add("COMBINE_TIME", time.strftime("%X %Z", now))
        md.add("COMBINE_ROOT", butler.mapper.root)

        # visits
        visits = [dataId['visit'] for dataId in dataIdList if dataId is not None and 'visit' in dataId]
        for i, v in enumerate(sorted(set(visits))):
            md.add("CALIB_INPUT_%04d" % (i), v)


        for calibName in 'bias', 'dark', 'flat', 'fringe':

            # default to 'not_available'
            calibPath, calibVersion, calibDate = "not_available", "not_available", "not_available"

            # If they ran with do<detrend> = False, be more specific ... say 'not_applied'
            # But note that doFoo=False for reduceFoo.py.
            doDetrend = getattr(self.config.isr, "do"+calibName.capitalize())
            if not doDetrend and calibName != self.calibName:
                calibPath, calibVersion, calibDate = "not_applied", "not_applied", "not_applied"

            else:

                # for the detrend we're running, we need the outputId ... dataIdList contains inputs
                dataId = dataIdList[0]
                if calibName == self.calibName:
                    dataId = outputId

                try:
                    calibPath      = butler.get(calibName+"_filename", dataId)
                    additionalData = butler.mapper.map(calibName, dataId).getAdditionalData()
                    calibVersion   = additionalData.get('calibVersion')
                    calibDate      = additionalData.get('calibDate')
                except:
                    pass

            md.add(calibName.upper()+"_VERSION", calibVersion)
            md.add(calibName.upper()+"_DATE", calibDate)
            md.add(calibName.upper()+"_PATH", calibPath)

        sums = checksum.CheckSum()(detrend)
        for k,v in sums.iteritems():
            md.add(k, v)
            

    def maskNans(self, image):
        """Mask NANs in the combined image"""
        if hasattr(image, "getMaskedImage"): # Deal with Exposure vs Image
            self.maskNans(image.getMaskedImage().getVariance())
            image = image.getMaskedImage().getImage()
        if hasattr(image, "getImage"): # Deal with DecoratedImage or MaskedImage vs Image
            image = image.getImage()
        array = image.getArray()
        bad = numpy.isnan(array)
        array[bad] = numpy.median(array[numpy.logical_not(bad)])


    def copyConfig(self, butler, dataId):
        """Copy the persisted config files to the same output directory as the detrends.
        """

        # we want to make sure we get all the foo~1, foo~2 old copies, not just the current one.
        configDir, _configFile   = os.path.split(butler.get('processCcd_config_filename', dataId)[0])
        configFiles              = glob.glob(os.path.join(configDir, "*"))

        detrendDir, _detrendFile = os.path.split(butler.get(self.calibName+"_filename", dataId)[0])
        detrendConfigDir         = os.path.join(detrendDir, "config")

        if not os.path.exists(detrendConfigDir):
            # handle possible race condition on makedirs
            try:
                os.makedirs(detrendConfigDir)
            except OSError, e:
                pass

        for conFile in configFiles:
            _configDir, conFileBase = os.path.split(conFile)
            dst = os.path.join(detrendConfigDir, conFileBase)
            shutil.copy(conFile, dst)
            shutil.copystat(conFile, dst)

    def write(self, butler, exposure, dataId):
        """Write the final combined detrend

        Only the slave nodes execute this method

        @param exposure  CCD exposure to write
        @param dataId    Data identifier
        """
        self.log.info("Writing %s on %s" % (dataId, NODE))
        butler.put(exposure, self.calibName, dataId)


class BiasConfig(DetrendConfig):
    """Configuration for bias construction.

    No changes required compared to the base class, but
    subclassed for distinction.
    """
    pass


class BiasTask(DetrendTask):
    """Bias construction"""
    ConfigClass = BiasConfig
    _DefaultName = "bias"
    calibName = "bias"
    FilterName = "NONE"

    @classmethod
    def applyOverrides(cls, config):
        """Overrides to apply for bias construction"""
        config.isr.doBias = False
        config.isr.doDark = False
        config.isr.doFlat = False
        config.isr.doFringe = False


class DarkConfig(DetrendConfig):
    """Configuration for dark construction"""
    doRepair = Field(dtype=bool, default=True, doc="Repair artifacts?")
    psfFwhm = Field(dtype=float, default=3.0, doc="Repair PSF FWHM (pixels)")
    psfSize = Field(dtype=int, default=21, doc="Repair PSF size (pixels)")
    crGrow = Field(dtype=int, default=2, doc="Grow radius for CR (pixels)")
    repair = ConfigurableField(target=RepairTask, doc="Task to repair artifacts")
    darkTime = Field(dtype=str, default="DARKTIME", doc="Header keyword for time since last CCD wipe, or None",
                     optional=True)

    def setDefaults(self):
        super(DarkConfig, self).setDefaults()
        self.combination.mask.append("CR")

class DarkTask(DetrendTask):
    """Dark construction

    The only major difference from the base class is dividing
    each image by the dark time to generate images of the
    dark rate.
    """
    ConfigClass = DarkConfig
    _DefaultName = "dark"
    calibName = "dark"
#    FilterName = "NONE"

    def __init__(self, *args, **kwargs):
        super(DarkTask, self).__init__(*args, **kwargs)
        self.makeSubtask("repair")

    @classmethod
    def applyOverrides(cls, config):
        """Overrides to apply for dark construction"""
        config.isr.doDark = False
        config.isr.doFlat = False
        config.isr.doFringe = False

    def processSingle(self, sensorRef):
        """Divide each processed image by the dark time to generate images of the dark rate"""
        exposure = super(DarkTask, self).processSingle(sensorRef)

        if self.config.doRepair:
            psf = measAlg.DoubleGaussianPsf(self.config.psfSize, self.config.psfSize,
                                            self.config.psfFwhm/(2*math.sqrt(2*math.log(2))))
            exposure.setPsf(psf)
            #import pdb; pdb.set_trace()
            self.repair.run(exposure, keepCRs=False)
            if self.config.crGrow > 0:
                mask = exposure.getMaskedImage().getMask().clone()
                mask &= mask.getPlaneBitMask("CR")
                fpSet = afwDet.FootprintSet(mask.convertU(), afwDet.Threshold(0.5))
                fpSet = afwDet.FootprintSet(fpSet, self.config.crGrow, True)
                fpSet.setMask(exposure.getMaskedImage().getMask(), "CR")

        if True:                # RHL
            import lsst.afw.display as afwDisplay
            disp = afwDisplay.Display(frame=1)
            disp.mtv(exposure, title="CR")        

        mi = exposure.getMaskedImage()
        mi /= self.getDarkTime(exposure)
        return exposure

    def getDarkTime(self, exposure):
        """Retrieve the dark time"""
        if self.config.darkTime is not None:
            return exposure.getMetadata().get(self.config.darkTime)
        return exposure.getCalib().getExptime()


class FlatCombineConfig(DetrendCombineConfig):
    """Configuration for flat construction"""
    doJacobian = Field(dtype=bool, default=False, doc="Apply Jacobian to flat-field?")


class FlatCombineTask(DetrendCombineTask):
    """Combination for flat-fields

    We allow the flat-field to be corrected for the Jacobian.

    The observed flat-field has a constant exposure per unit area.
    However, distortion in the camera makes the pixel area (the angle
    on the sky subtended per pixel) larger as one moves away from the
    optical axis, so that the flux in the observed flat-field drops.
    But this drop does not mean the detector is less sensitive.  The
    Jacobian is a rough estimate of the relative area of each pixel.
    The correction is therefore achieved by multiplying the observed
    flat-field by the Jacobian to create a "photometric flat" which
    has constant exposure per pixel --- a true measure of the
    point-source sensitivity of the camera as a function of pixel
    (modulo contributions from scattered light, which require much
    more work to account for).

    Note, however, that application of this correction means that
    images flattened with this (Jacobian-corrected) "photometric flat"
    will not have a flat sky, potentially making sky subtraction more
    difficult.  Furthermore, care must be taken to ensure this
    correction is not applied more than once (e.g., in warping).
    """
    ConfigClass = FlatCombineConfig

    def run(self, sensorRefList, expScales=None, finalScale=None):
        """Multiply the combined flat-field by the Jacobian"""
        combined = super(FlatCombineTask, self).run(sensorRefList, expScales=expScales, finalScale=finalScale)
        if self.config.doJacobian:
            dataRef = next((dataRef for dataref in sensorRefList if dataRef is not None), None)
            if dataRef is None:
                raise RuntimeError("No non-None data references: %s" % sensorRefList)
            jacobian = self.getJacobian(dataRef, combined.getDimensions())
            combined *= jacobian
        return combined

    def getJacobian(self, sensorRef, dimensions, inputName="postISRCCD"):
        """Calculate the Jacobian as a function of position

        @param sensorRef    Data reference for a representative CCD (to get the Detector)
        @param dimensions   Dimensions of the flat-field
        @param inputName    Data set name for inputs
        @return Jacobian image
        """
        # Retrieve the detector and distortion
        # XXX It's unfortunate that we have to read an entire image to get the detector, but there's no
        # public API in the butler to get the same.
        image = sensorRef.get(inputName)
        detector = image.getDetector()
        distortion = detector.getDistortion()
        del image

        # Calculate the Jacobian for each pixel
        # XXX This would be faster in C++, but it's not awfully slow.
        jacobian = afwImage.ImageF(dimensions)
        array = jacobian.getArray()
        width, height = dimensions
        circle = afwEll.Quadrupole(1.0, 1.0, 0.0)
        for y in xrange(height):
            for x in xrange(width):
                array[y,x] = distortion.distort(afwGeom.Point2D(x, y), circle, detector).getDeterminant()

        return jacobian


class FlatConfig(DetrendConfig):
    """Configuration for flat construction"""
    iterations = Field(dtype=int, default=10, doc="Number of iterations for scale determination")
    stats = ConfigurableField(target=DetrendStatsTask, doc="Background statistics configuration")
    def setDefaults(self):
        self.combination.retarget(FlatCombineTask)

class FlatTask(DetrendTask):
    """Flat construction

    The principal change involves gathering the background values from each
    image and using them to determine the scalings for the final combination.
    """
    ConfigClass = FlatConfig
    _DefaultName = "flat"
    calibName = "flat"

    @classmethod
    def applyOverrides(cls, config):
        """Overrides for flat construction"""
        config.isr.doFlat = False
        config.isr.doFringe = False


    def __init__(self, *args, **kwargs):
        super(FlatTask, self).__init__(*args, **kwargs)
        self.makeSubtask("stats")

    def processResult(self, exposure):
        return self.stats.run(exposure)

    def scale(self, ccdKeys, ccdIdLists, data):
        """Determine the scalings for the final combination

        We have a matrix B_ij = C_i E_j, where C_i is the relative scaling
        of one CCD to all the others in an exposure, and E_j is the scaling
        of the exposure.  We determine the C_i and E_j from B_ij by iteration,
        under the additional constraint that the average CCD scale is unity.
        We convert everything to logarithms so we can work linearly.  This
        algorithm comes from Eugene Magnier and Pan-STARRS.
        """
        # Format background measurements into a matrix
        indices = dict((name, i) for i, name in enumerate(ccdIdLists.keys()))
        bgMatrix = numpy.array([[0.0] * len(expList) for expList in ccdIdLists.values()])
        for name in ccdIdLists.keys():
            i = indices[name]
            bgMatrix[i] = [d if d is not None else numpy.nan for d in data[name]]

        numpyPrint = numpy.get_printoptions()
        numpy.set_printoptions(threshold='nan')
        self.log.info("Input backgrounds: %s" % bgMatrix)

        # Flat-field scaling
        numCcds = len(ccdIdLists)
        numExps = bgMatrix.shape[1]
        bgMatrix = numpy.log(bgMatrix)      # log(Background) for each exposure/component
        bgMatrix = numpy.ma.masked_array(bgMatrix, numpy.isnan(bgMatrix))
        compScales = numpy.zeros(numCcds) # Initial guess at log(scale) for each component
        expScales = numpy.array([(bgMatrix[:,i] - compScales).mean() for i in range(numExps)])

        for iterate in range(self.config.iterations):
            compScales = numpy.array([(bgMatrix[i,:] - expScales).mean() for i in range(numCcds)])
            expScales = numpy.array([(bgMatrix[:,i] - compScales).mean() for i in range(numExps)])

            avgScale = numpy.average(numpy.exp(compScales))
            compScales -= numpy.log(avgScale)
            self.log.logdebug("Iteration %d exposure scales: %s" % (iterate, numpy.exp(expScales)))
            self.log.logdebug("Iteration %d component scales: %s" % (iterate, numpy.exp(compScales)))

        expScales = numpy.array([(bgMatrix[:,i] - compScales).mean() for i in range(numExps)])

        self.log.info("expScales = %g" % expScales)
        if numpy.any(numpy.isnan(expScales)):
            raise RuntimeError("Bad exposure scales: %s --> %s" % (bgMatrix, expScales))

        expScales = numpy.exp(expScales)
        compScales = numpy.exp(compScales)

        self.log.info("Exposure scales: %s" % expScales)
        self.log.info("Component relative scaling: %s" % compScales)

        return dict((ccdName, Struct(ccdScale=compScales[indices[ccdName]], expScales=expScales))
                    for ccdName in ccdIdLists.keys())

def getDataRef(butler, dataId, datasetType="raw"):
    """Construct a dataRef from a butler and data identifier"""
    dataRefList = [ref for ref in butler.subset(datasetType, **dataId)]
    print 'getDataRef: dataId = ',dataId
    camera = dataRefList[0].get("camera")
    dataRef = dataRefList[0]
    assert len(dataRefList) == 1
    return dataRefList[0]
