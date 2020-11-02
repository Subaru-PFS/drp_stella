import re
from collections import defaultdict
import numpy as np
import lsstDebug
import lsst.pex.config as pexConfig
from lsst.pipe.base import TaskRunner, ArgumentParser, CmdLineTask, Struct
import lsst.afw.display as afwDisplay
from .reduceExposure import ReduceExposureTask
from .identifyLines import IdentifyLinesTask
from pfs.drp.stella.utils import plotReferenceLines
from pfs.drp.stella.fitGlobalDetectorMap import FitGlobalDetectorMapTask
from .centroidLines import CentroidLinesTask
from .arcLine import ArcLineSet
from .readLineList import ReadLineListTask

__all__ = ["ReduceArcConfig", "ReduceArcTask"]


class ReduceArcConfig(pexConfig.Config):
    """Configuration for reducing arc images"""
    reduceExposure = pexConfig.ConfigurableField(target=ReduceExposureTask,
                                                 doc="Extract spectra from exposure")
    readLineList = pexConfig.ConfigurableField(target=ReadLineListTask, doc="Read linelist")
    identifyLines = pexConfig.ConfigurableField(target=IdentifyLinesTask, doc="Identify arc lines")
    centroidLines = pexConfig.ConfigurableField(target=CentroidLinesTask, doc="Centroid lines")
    fitDetectorMap = pexConfig.ConfigurableField(target=FitGlobalDetectorMapTask, doc="Fit detectorMap")

    def setDefaults(self):
        super().setDefaults()
        self.reduceExposure.doSubtractSky2d = False
        self.reduceExposure.doWriteArm = False  # We'll do this ourselves, after wavelength calibration
        self.readLineList.restrictByLamps = True


class ReduceArcRunner(TaskRunner):
    """TaskRunner that does scatter/gather for ReduceArcTask"""
    def __init__(self, *args, **kwargs):
        kwargs["doReturnResults"] = True
        super().__init__(*args, **kwargs)

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
                final = task.gather(dataRefList, [rr.result.result for rr in results])
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
        return Struct(**{key: value for key, value in result.getDict().items() if key != "dataRef"})


class ReduceArcTask(CmdLineTask):
    """Task to reduce Arc images"""
    ConfigClass = ReduceArcConfig
    _DefaultName = "reduceArc"
    RunnerClass = ReduceArcRunner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("reduceExposure")
        self.makeSubtask("readLineList")
        self.makeSubtask("identifyLines")
        self.makeSubtask("fitDetectorMap")
        self.makeSubtask("centroidLines")
        self.debugInfo = lsstDebug.Info(__name__)

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", datasetType="raw",
                               help="input identifiers, e.g., --id visit=123 ccd=4")
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

    def runDataRef(self, dataRef):
        """Entry point for scatter stage

        Extracts spectra from the exposure pointed to by the ``dataRef``,
        centroids and identifies lines.

        Debug parameters include:

        - ``display`` (`bool`): Activate displays and plotting (master switch)?
        - ``frame`` (`dict` mapping `int` to `int`): Display frame to use as a
            function of visit.
        - ``backend`` (`str`): Display backend name.
        - ``displayIdentifications`` (`bool`): Display image with lines
            identified?
        - ``displayCalibrations`` (`bool`): Plot calibration results? See the
            ``plotCalibrations`` method for additional debug parameters
            controlling these plots.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.

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
        """
        metadata = dataRef.get("raw_md")
        lines = self.readLineList.run(metadata=metadata)
        results = self.reduceExposure.runDataRef([dataRef])
        assert len(results.spectraList) == 1, "Single in, single out"
        spectra = results.spectraList[0]
        exposure = results.exposureList[0]
        detectorMap = results.detectorMapList[0]

        self.identifyLines.run(spectra, detectorMap, lines)
        if self.debugInfo.display and self.debugInfo.displayIdentifications:
            frame = self.debugInfo.frame[dataRef.dataId["visit"]] if self.debugInfo.frame is not None else 1
            self.plotIdentifications(self.debugInfo.backend or "ds9", exposure, spectra, detectorMap, frame)
        referenceLines = self.centroidLines.getReferenceLines(spectra)
        lines = self.centroidLines.run(exposure, referenceLines, detectorMap)
        dataRef.put(lines, "arcLines")

        return Struct(
            spectra=spectra,
            lines=lines,
            detectorMap=detectorMap,
            visitInfo=exposure.getInfo().getVisitInfo(),
            metadata=metadata,
        )

    def gather(self, dataRefList, results):
        """Entry point for gather stage

        Combines the input spectra and fits a wavelength calibration.

        Parameters
        ----------
        dataRefList : iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for arms.
        results : `list` of `lsst.pipe.base.Struct`
            List of results from the ``run`` method.
        """
        if len(results) == 0:
            raise RuntimeError("No input spectra")

        dataRef = self.reduceDataRefs(dataRefList)
        lines = sum((rr.lines for rr in results), ArcLineSet.empty())
        archetype = next(rr for rr in results if rr is not None)
        visitInfo = archetype.visitInfo  # All more or less identical
        metadata = archetype.metadata  # All more or less identical
        oldDetMap = archetype.detectorMap  # All more or less identical

        detectorMap = self.fitDetectorMap.run(dataRef.dataId["arm"], oldDetMap.bbox, lines,
                                              visitInfo, oldDetMap.metadata)

        self.write(dataRef, detectorMap, metadata, visitInfo)
        if self.debugInfo.display and self.debugInfo.displayCalibrations:
            for rr in results:
                self.plotCalibrations(rr.spectra, detectorMap)

    def write(self, dataRef, detectorMap, metadata, visitInfo):
        """Write outputs

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        metadata : `lsst.daf.base.PropertySet`
            Exposure header.
        visitInfo : `lsst.afw.image.VisitInfo`
            Visit information for exposure.
        """
        self.log.info("Writing output for %s", dataRef.dataId)
        detectorMap.setVisitInfo(visitInfo)
        visit0 = dataRef.dataId["visit"]
        dateObs = dataRef.dataId["dateObs"]
        taiObs = dataRef.dataId["taiObs"]
        calibId = detectorMap.metadata.get("CALIB_ID")
        calibId = re.sub(r"visit0=\d+", "visit0=%d" % (visit0,), calibId)
        calibId = re.sub(r"calibDate=\d\d\d\d-\d\d-\d\d", "calibDate=%s" % (dateObs,), calibId)
        calibId = re.sub(r"calibTime=\S+", "calibDate=%s" % (taiObs,), calibId)
        detectorMap.metadata.set("CALIB_ID", calibId)
        dataRef.put(detectorMap, 'detectorMap', visit0=visit0)

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

    def plotIdentifications(self, backend, exposure, spectrumSet, detectorMap, frame=1):
        """Plot line identifications

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Arc image.
        spectrum : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        """
        display = afwDisplay.Display(frame, backend=backend)
        display.mtv(exposure)
        for spec in spectrumSet:
            fiberId = spec.getFiberId()
            refLines = spec.getGoodReferenceLines()
            with display.Buffering():
                # Label fibers
                y = 0.5*len(spec)
                x = detectorMap.getXCenter(fiberId, y)
                display.dot(str(fiberId), x, y, ctype='green')

                # Plot arc lines
                for rl in refLines:
                    yActual = rl.fitPosition
                    xActual = detectorMap.getXCenter(fiberId, yActual)
                    display.dot('o', xActual, yActual, ctype='blue')
                    xExpect, yExpect = detectorMap.findPoint(fiberId, rl.wavelength)
                    display.dot('x', xExpect, yExpect, ctype='red')

    def plotCalibrations(self, spectrumSet, detectorMap):
        """Plot wavelength calibration results

        Important debug parameters:

        - ``fiberId`` (iterable of `int`): fibers to plot, if set.
        - ``plotCalibrationsRows`` (`bool`): plot spectrum as a function of
          rows?
        - ``plotCalibrationsWavelength`` (`bool`): plot spectrum as a function
          of wavelength?

        Parameters
        ----------
        spectrumSet : `pfs.drp.stella.SpectrumSet`
            Spectra that have been wavelength-calibrated.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of wl,fiber to detector position.
        """
        import matplotlib.pyplot as plt
        for spectrum in spectrumSet:
            if self.debugInfo.fiberId and spectrum.fiberId not in self.debugInfo.fiberId:
                continue
            if self.debugInfo.plotCalibrationsRows:
                rows = np.arange(detectorMap.bbox.getHeight(), dtype=float)
                plt.plot(rows, spectrum.spectrum)
                xlim = plt.xlim()
                plotReferenceLines(spectrum.referenceLines, "guessedPosition", alpha=0.1,
                                   labelLines=True, labelStatus=False)
                plotReferenceLines(spectrum.referenceLines, "fitPosition", ls='-', alpha=0.5,
                                   labelLines=True, labelStatus=True)
                plt.xlim(xlim)
                plt.legend(loc='best')
                plt.xlabel('row')
                plt.title(f"FiberId {spectrum.fiberId}")
                plt.show()

            if self.debugInfo.plotCalibrationsWavelength:
                plt.plot(spectrum.wavelength, spectrum.spectrum)
                xlim = plt.xlim()
                plotReferenceLines(spectrum.referenceLines, "wavelength", ls='-', alpha=0.5,
                                   labelLines=True, wavelength=spectrum.wavelength,
                                   spectrum=spectrum.spectrum)
                plt.xlim(xlim)
                plt.legend(loc='best')
                plt.xlabel("Wavelength (vacuum nm)")
                plt.title(f"FiberId {spectrum.fiberId}")
                plt.show()

    def _getMetadataName(self):
        """Disable writing metadata (doesn't work with lists of dataRefs anyway)"""
        return None
