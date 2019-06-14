import os
import re
from collections import defaultdict
import lsstDebug
import lsst.pex.config as pexConfig
from lsst.utils import getPackageDir
from lsst.pipe.base import TaskRunner, ArgumentParser, CmdLineTask, Struct
import lsst.afw.display as afwDisplay
from .wavelengthStatistics import WavelengthSolutionTask
from .reduceExposure import ReduceExposureTask
from .identifyLines import IdentifyLinesTask
from .utils import readLineListFile
from lsst.obs.pfs.utils import getLampElements
from pfs.drp.stella import Spectrum, SpectrumSet


__all__ = ["CalcResolutionConfig", "CalcResolutionTask"]


class CalcResolutionConfig(pexConfig.Config):
    """Configuration for reducing arc images"""
    reduceExposure = pexConfig.ConfigurableField(target=ReduceExposureTask,
                                                 doc="Extract spectra from exposure")
    identifyLines = pexConfig.ConfigurableField(target=IdentifyLinesTask, doc="Identify arc lines")
    wavelengthSolution = pexConfig.ConfigurableField(target=WavelengthSolutionTask,
                                                    doc="Calibrate a SpectrumSet's wavelengths")

    minArcLineIntensity = pexConfig.Field(doc="Minimum 'NIST' intensity to use emission lines",
                                          dtype=float, default=0)
    

    def setDefaults(self):
        super().setDefaults()
        self.reduceExposure.doSubtractSky2d = False


class CalcResolutionRunner(TaskRunner):
    """TaskRunner that does scatter/gather for ReduceArcTask"""
    def __init__(self, *args, **kwargs):
        kwargs["doReturnResults"] = True
        super().__init__(*args, **kwargs)

    @classmethod
    def getTargetList(cls, parsedCmd, **kwargs):
        return super().getTargetList(parsedCmd, lineListFilename=parsedCmd.lineList)

    def run(self, parsedCmd):
        """Scatter-gather"""
        if not self.precall(parsedCmd):
            exit(1)
        task = self.makeTask(parsedCmd=parsedCmd)
        scatterResults = super().run(parsedCmd)

        # Group inputs by spectrograph+arm
        groupedResults = defaultdict(list)
        for rr, dataRef in zip(scatterResults, parsedCmd.id.refList):
            spectrograph = dataRef.dataId["spectrograph"]
            arm = dataRef.dataId["arm"]
            groupedResults[(spectrograph, arm)].append(Struct(dataRef=dataRef, result=rr))

        gatherResults = []
        for results in groupedResults.values():
            dataRefList = [rr.dataRef for rr in results]
            task.verify(dataRefList)
            dataRef = task.reduceDataRefs(dataRefList)
            final = None
            exitStatus = 0
            if len(results) > 0:
                exitStatus = max(rr.result.exitStatus for rr in results if rr is not None)
            if len(results) == 0:
                task.log.fatal("No results for %s." % (dataRef.dataId,))
            elif exitStatus > 0:
                task.log.fatal("Failed to process at least one of the components for %s" % (dataRef.dataId,))
            else:
                final = task.gather(dataRef, [rr.result.result for rr in results],
                                    lineListFilename=parsedCmd.lineList)
            gatherResults.append(Struct(result=final, exitStatus=exitStatus))
        return gatherResults

    def __call__(self, args):
        """Run the Task on a single target.

        This implementation strips out the ``dataRef`` from the
        original implementation, because that can cause problems
        from SQLite databases (used by the butler registries)
        being destroyed from threads other than the one in which
        they were created.
        """
        result = super().__call__(args)
        return Struct(**{key: value for key, value in result.getDict().items() if key is not "dataRef"})


class CalcResolutionTask(CmdLineTask):
    """Task to reduce Arc images"""
    ConfigClass = CalcResolutionConfig
    _DefaultName = "calcResolution"
    RunnerClass = CalcResolutionRunner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("reduceExposure")
        self.makeSubtask("identifyLines")
        self.makeSubtask("wavelengthSolution")
        
        self.debugInfo = lsstDebug.Info(__name__)

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", datasetType="raw",
                               help="input identifiers, e.g., --id visit=123 ccd=4")
        parser.add_argument("--lineList", help="Reference line list",
                            default=os.path.join(getPackageDir("obs_pfs"),
                                                 "pfs", "lineLists", "ArCdHgKrNeXe.txt"))
        return parser

    def verify(self, dataRefList):
        """Verify inputs

        Ensure that all inputs are from the same CCD.

        Parameters
        ----------
        dataRefList : `list` of `lsst.daf.persistence.ButlerDataRef`
            List of data references.

        Raises
        ------
        `RuntimeError`
            If the inputs are from different CCDs.
        """
        for prop in ("arm", "spectrograph"):
            values = set([ref.dataId["arm"] for ref in dataRefList])
            if len(values) > 1:
                raise RuntimeError("%s varies for inputs: %s" % (prop, [ref.dataId for ref in dataRefList]))

    def run(self, dataRef, lineListFilename):
        """Entry point for scatter stage

        Extracts spectra from the exposure pointed to by the ``dataRef``.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.
        lineListFilename : `str`
            Filename of arc line list.

        Returns
        -------
        spectrumSet : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        visitInfo : `lsst.afw.image.VisitInfo`
            Visit information for exposure.
        metadata : `lsst.daf.base.PropertyList`
            Exposure metadata (FITS header)
        lamps : `list` of `str`
            List of arc species.
        """

        metadata = dataRef.get("raw_md")
        lamps = getLampElements(metadata)
        lines = self.readLineList(lamps, lineListFilename)
        results = self.reduceExposure.run([dataRef])
        self.identifyLines.run(results.spectraList[0], results.detectorMapList[0], lines)

        return Struct(
            spectra=results.spectraList[0],
            detectorMap=results.detectorMapList[0],
            visitInfo=results.exposureList[0].getInfo().getVisitInfo(),
            metadata=metadata,
            lamps=lamps,
        )

    def gather(self, dataRef, results, lineListFilename):
        """Entry point for gather stage

        Combines the input spectra and fits a wavelength calibration.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.
        results : `list` of `lsst.pipe.base.Struct`
            List of results from the ``run`` method.
        lineListFilename : `str`
            Filename of arc line list.
        """
        if len(results) == 0:
            raise RuntimeError("No input spectra")
        spectra = [rr.spectra for rr in results if rr is not None]
        detectorMap = next(rr.detectorMap for rr in results if rr is not None)  # All identical
        visitInfo = next(rr.visitInfo for rr in results if rr is not None)  # More or less identical
        metadata = next(rr.metadata for rr in results if rr is not None)  # More or less identical
        lamps = set(sum((rr.lamps for rr in results if rr is not None), []))
        spectrumSet = self.coaddSpectra(spectra)
        arcLines = self.readLineList(lamps, lineListFilename)
        self.identifyLines.run(spectrumSet, detectorMap, arcLines)
        
        self.write(dataRef, spectrumSet, detectorMap, metadata, visitInfo)

        for spec in spectrumSet:
            WLInfo = self.wavelengthSolution.run(spec, detectorMap)
            dataRef.put(WLInfo,"WLInfo")

        if self.debugInfo.display:
            self.plot(spectrumSet, detectorMap, arcLines)


    def coaddSpectra(self, spectra):
        """Coadd multiple SpectrumSets

        Adds the wavelength, spectrum, background, covar and reference lines
        of the input `SpectrumSet`s for each aperture.

        XXX need to make a set of lines, not a list

        Parameters
        ----------
        spectra : `list` of `pfs.drp.stella.SpectrumSet`
            List of extracted spectra for different exposures.

        Returns
        -------
        result : `pfs.drp.stella.SpectrumSet`
            Coadded spectra.
        """
        numApertures = set(len(ss) for ss in spectra)
        assert len(numApertures) == 1, "Same number of apertures in each SpectrumSet"
        numApertures = numApertures.pop()
        length = set(ss.getLength() for ss in spectra)
        assert len(length) == 1, "Same length in each SpectrumSet"
        length = length.pop()

        result = SpectrumSet(numApertures, length)
        for ap in range(numApertures):
            inputs = [ss[ap] for ss in spectra]
            fiberId = set([ss.getFiberId() for ss in inputs])
            if len(fiberId) > 1:
                raise RuntimeError("Multiple fiber IDs (%s) for aperture %d" % (fiberId, ap))
            coadd = Spectrum(inputs[0].getNumPixels(), fiberId.pop())
            # Coadd the various elements of the spectrum
            for elem in ("Wavelength", "Spectrum", "Background", "Covariance"):
                element = getattr(inputs[0], "get" + elem)()
                for ss in inputs[1:]:
                    element += getattr(ss, "get" + elem)()
                getattr(coadd, "set" + elem)(element)
            result[ap] = coadd
        return result

    def readLineList(self, lamps, lineListFilename):
        """Read the arc line list from file

        Parameters
        ----------
        lamps : iterable of `str`
            Lamp species.
        lineListFilename : `str`
            Filename of line list.

        Returns
        -------
        arcLines : `list` of `pfs.drp.stella.ReferenceLine`
            List of reference lines.
        """
        self.log.info("Arc lamp elements are: %s", " ".join(lamps))
        return readLineListFile(lineListFilename, lamps, minIntensity=self.config.minArcLineIntensity)

    def write(self, dataRef, spectrumSet, detectorMap, metadata, visitInfo):
        """Write outputs

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference.
        spectrumSet : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        metadata : `lsst.daf.base.PropertySet`
            Exposure header.
        visitInfo : `lsst.afw.image.VisitInfo`
            Visit information for exposure.
        """

        self.log.info("Writing output for %s", dataRef.dataId)
        # XXX set metadata in spectrumSet
        dataRef.put(spectrumSet, "pfsArm")
        detectorMap.setVisitInfo(visitInfo)
        visit0 = dataRef.dataId["expId"]
        calibId = detectorMap.metadata.get("CALIB_ID")
        detectorMap.metadata.set("CALIB_ID", re.sub("visit0=\d+", "visit0=%d" % (visit0,), calibId))
        dataRef.put(detectorMap, 'detectormap', visit0=visit0)
        

    def reduceDataRefs(self, dataRefList):
        """Reduce a list of data references

        Produces a single data reference to use for the 'gather' stage.

        This implementation simply returns the data reference with the
        lowest ``visit``.

        Parameters
        ----------
        dataRefList : `list` of `lsst.daf.peristence.ButlerDatRef`
            List of data references.

        Returns
        -------
        dataRef : `lsst.daf.peristence.ButlerDataRef`
            Data reference.
        """
        return sorted(dataRefList, key=lambda dataRef: dataRef.dataId["visit"])[0]

    def plot(self, spectrumSet, detectorMap, arcLines):
        """Plot results

        Parameters
        ----------
        spectrumSet : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        arcLines : `list` of `pfs.drp.stella.ReferenceLine`
            List of reference lines.
        """
        display = afwDisplay.Display(self.debugInfo.arc_frame)
        if self.debugInfo.display and self.debugInfo.arc_frame >= 0 and self.debugInfo.showArcLines:
            for spec in spectrumSet:
                fiberId = spec.getFiberId()

                x = detectorMap.findPoint(fiberId, arcLines[0].wavelength)[0]
                y = 0.5*len(spec)
                display.dot(str(fiberId), x, y + 10*(fiberId%2), ctype='blue')

                for rl in arcLines:
                    x, y = detectorMap.findPoint(fiberId, rl.wavelength)
                    display.dot('o', x, y, ctype='blue')

    def _getMetadataName(self):
        """Disable writing metadata (doesn't work with lists of dataRefs anyway)"""
        return None
