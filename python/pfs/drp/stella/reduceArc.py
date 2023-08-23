from collections import defaultdict
from datetime import datetime

import numpy as np
import lsstDebug
import lsst.pex.config as pexConfig
from lsst.pipe.base import TaskRunner, ArgumentParser, CmdLineTask, Struct
from .reduceExposure import ReduceExposureTask
from pfs.drp.stella.fitDistortedDetectorMap import FitDistortedDetectorMapTask
from .arcLine import ArcLineSet
from .utils import addPfsCursor
from .referenceLine import ReferenceLineStatus
from .constructSpectralCalibs import setCalibHeader
from .SpectrumSetContinued import SpectrumSet


__all__ = ["ReduceArcConfig", "ReduceArcTask"]


class ReduceArcConfig(pexConfig.Config):
    """Configuration for reducing arc images"""
    reduceExposure = pexConfig.ConfigurableField(target=ReduceExposureTask,
                                                 doc="Extract spectra from exposure")
    fitDetectorMap = pexConfig.ConfigurableField(target=FitDistortedDetectorMapTask, doc="Fit detectorMap")
    doUpdateDetectorMap = pexConfig.Field(dtype=bool, default=True,
                                          doc="Write an updated detectorMap?")

    def setDefaults(self):
        super().setDefaults()
        self.reduceExposure.doAdjustDetectorMap = False  # We'll do a full-order fit
        self.reduceExposure.doSubtractSky2d = False
        self.reduceExposure.doWriteArm = False  # We'll do this ourselves, after wavelength calibration
        self.reduceExposure.photometerLines.doApertureCorrection = False  # Unnecessary for centroiding
        self.reduceExposure.readLineList.exclusionRadius = 0.3  # Eliminate confusion


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
        if self.config.doUpdateDetectorMap:
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
                    task.log.fatal("Failed to process at least one of the components for %s" %
                                   (dataRef.dataId,))
                else:
                    final = task.gather(dataRefList, [rr.result.result for rr in results])
                gatherResults.append(Struct(result=final, exitStatus=exitStatus))
        else:
            task.log.info("Not writing an updated DetectorMap")

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
        self.makeSubtask("fitDetectorMap")
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
        - ``displayExposure`` (`bool`) Display the image before marking lines
        - ``fiberIds`` (iterable of `int`) Only show these fibres
        - ``displayCalibrations`` (`bool`): Plot calibration results? See also:
        - ``plotCalibrationsRows`` (`bool`): plot spectrum as a function of rows?
        - ``plotCalibrationsWavelength`` (`bool`): plot spectrum as a function of wavelength?

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

        try:
            spectra = SpectrumSet.fromPfsArm(dataRef.get("pfsArm"))
            lines = dataRef.get("arcLines")
            detectorMap = dataRef.get("detectorMap_used")
            exposure = dataRef.get("calexp")  # Reading this just to return the visitInfo: inefficient, but...
            self.log.info("Read existing data for %s", dataRef.dataId)
        except Exception:
            results = self.reduceExposure.runDataRef(dataRef)
            spectra = results.spectra
            exposure = results.exposure
            detectorMap = results.detectorMap
            lines = results.lines

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

        if not self.config.doUpdateDetectorMap:
            self.log.fatal("you can't get here")

        dataRef = self.reduceDataRefs(dataRefList)
        lines = sum((rr.lines for rr in results), ArcLineSet.empty())
        archetype = next(rr for rr in results if rr is not None)
        visitInfo = archetype.visitInfo  # All more or less identical
        metadata = archetype.metadata  # All more or less identical
        oldDetMap = archetype.detectorMap  # All more or less identical
        spatialOffsets = oldDetMap.getSpatialOffsets()
        spectralOffsets = oldDetMap.getSpectralOffsets()

        dataRef.put(lines, "arcLines")

        detectorMap = self.fitDetectorMap.run(dataRef.dataId, oldDetMap.bbox, lines, visitInfo,
                                              oldDetMap.metadata, spatialOffsets, spectralOffsets).detectorMap

        self.write(dataRef, detectorMap, metadata, visitInfo, [ref.dataId["visit"] for ref in dataRefList])

        # Update wavelength calibrations on extracted spectra, and write
        for dataRef, rr in zip(dataRefList, results):
            for ss in rr.spectra:
                ss.setWavelength(detectorMap.getWavelength(ss.fiberId))
            dataRef.put(rr.spectra.toPfsArm(dataRef.dataId), "pfsArm")

            if self.debugInfo.display and self.debugInfo.displayCalibrations:
                if self.debugInfo.fiberId and not self.debugInfo.fiberIds:
                    self.debugInfo.fiberIds = self.debugInfo.fiberId

                fiberIds = self.debugInfo.fiberIds
                if fiberIds is False:
                    fiberIds = None

                self.plotCalibrations(rr.spectra, rr.lines, detectorMap, fiberIds=fiberIds,
                                      plotCalibrationsRows=self.debugInfo.plotCalibrationsRows,
                                      plotCalibrationsWavelength=self.debugInfo.plotCalibrationsWavelength)

    def write(self, dataRef, detectorMap, metadata, visitInfo, visits):
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
        visits : iterable of `int`
            List of visits used to construct the detectorMap.
        """
        self.log.info("Writing output for %s", dataRef.dataId)
        detectorMap.setVisitInfo(visitInfo)
        visit0 = dataRef.dataId["visit"]

        outputId = dict(
            visit0=visit0,
            calibDate=dataRef.dataId["dateObs"],
            calibTime=dataRef.dataId["taiObs"],
            arm=dataRef.dataId["arm"],
            spectrograph=dataRef.dataId["spectrograph"],
            ccd=dataRef.dataId["ccd"],
            filter=dataRef.dataId["filter"],
        )

        setCalibHeader(detectorMap.metadata, "detectorMap", visits, outputId)
        date = datetime.now().isoformat()
        history = f"reduceArc on {date} with visit={','.join(str(vv) for vv in sorted(visits))}"
        detectorMap.metadata.add("HISTORY", history)

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

    def plotIdentifications(self, display, exposure, lines, detectorMap, displayExposure=True,
                            showAllCandidates=False, fiberIds=False):
        """Plot line identifications

        Parameters
        ----------
        display : `lsst.afw.display.Display`
            Display to use
        exposure : `lsst.afw.image.Exposure`
            Arc image.
        lines : `pfs.drp.stella.ArcLineSet`
            Set of reference lines matched to the data
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        displayExposure : `bool`
            Use display.mtv to show the exposure; if False
            the caller is responsible for the mtv and maybe
            an erase too
        showAllCandidates : `bool`
            Show all candidates from the line list, not just
            the ones that are matched
        fiberIds : `list` of `int`
            Show detected lines for these fiberIds; ignore if None or False
        """

        labelFibers = False             # write fiberId to the display?
        if displayExposure:
            display.mtv(exposure)
            try:
                addPfsCursor(display, detectorMap)  # mouseover provides fiberIds
            except NameError:               # only supported by matplotlib at present
                labelFibers = True

        for fiberId in lines:
            # N.b. "fiberIds not in (False, None)" fails with ndarray
            if fiberIds is not False and fiberIds is not None or fiberId not in self.fiberIds:
                continue

            with display.Buffering():
                if labelFibers:         # Label fibers if addPfsCursor failed
                    y = 0.5*exposure.getHeight()
                    x = detectorMap.getXCenter(fiberId, y)
                    display.dot(str(fiberId), x, y, ctype='green')

                # Plot arc lines
                for rl in lines[fiberId]:
                    yActual = rl.fitPosition
                    xActual = detectorMap.getXCenter(fiberId, yActual)
                    if (rl.status & ReferenceLineStatus.GOOD) == 0:
                        ctype = 'cyan'
                    elif (rl.status & ReferenceLineStatus.NOT_VISIBLE) != 0:
                        ctype = 'magenta'
                    elif (rl.status & (ReferenceLineStatus.BLEND |
                                       ReferenceLineStatus.SUSPECT |
                                       ReferenceLineStatus.REJECTED)) != 0:
                        ctype = 'green'
                    else:
                        ctype = 'blue'

                    display.dot('o', xActual, yActual, ctype=ctype)

                    if showAllCandidates or (rl.status & ReferenceLineStatus.GOOD) == 0:
                        xExpect, yExpect = detectorMap.findPoint(fiberId, rl.wavelength)
                        display.dot('x', xExpect, yExpect, ctype='red', size=0.5)

    def plotCalibrations(self, spectrumSet, lines, detectorMap, fiberIds=None, plotCalibrationsRows=False,
                         plotCalibrationsWavelength=True):
        """Plot wavelength calibration results

        Parameters
        ----------
        spectrumSet : `pfs.drp.stella.SpectrumSet`
            Spectra that have been wavelength-calibrated.
        lines : `pfs.drp.stella.ArcLineSet`
            Measured arc lines.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of wl,fiber to detector position.
        fiberIds : (iterable of `int`)
            fibers to plot if not None
        plotCalibrationsRows : (`bool`)
            plot spectrum as a function of rows?
        plotCalibrationsWavelength (`bool`)
            plot spectrum as a function of wavelength?
        """
        import matplotlib.pyplot as plt

        for spectrum in spectrumSet:
            if not spectrum.isWavelengthSet() or np.all(np.isnan(spectrum.spectrum)):
                continue
            refLines = lines.extractReferenceLines(spectrum.fiberId)
            if fiberIds is not None and spectrum.fiberId not in fiberIds:
                continue
            if plotCalibrationsRows:
                rows = np.arange(detectorMap.bbox.getHeight(), dtype=float)
                plt.plot(rows, spectrum.spectrum)
                xlim = plt.xlim()
                refLines.plot(plt.gca(), alpha=0.5, ls="-", labelLines=True, labelStatus=True, pixels=True,
                              wavelength=spectrum.wavelength, spectrum=spectrum.spectrum)
                plt.xlim(xlim)
                if len(plt.gca().get_legend_handles_labels()[1]) > 0:
                    plt.legend(loc='best')
                plt.xlabel('row')
                plt.title(f"FiberId {spectrum.fiberId}")
                plt.show()

            if plotCalibrationsWavelength:
                plt.plot(spectrum.wavelength, spectrum.spectrum,
                         label=None if fiberIds is None else spectrum.fiberId)
                xlim = plt.xlim()
                refLines.plot(plt.gca(), alpha=0.5, ls='-', labelLines=True, labelStatus=True,
                              wavelength=spectrum.wavelength, spectrum=spectrum.spectrum)
                plt.xlim(xlim)
                if len(plt.gca().get_legend_handles_labels()[1]) > 0:
                    plt.legend(loc='best')
                plt.xlabel("Wavelength (vacuum nm)")
                if fiberIds is not None and len(fiberIds) == 1:
                    plt.title(f"FiberId {spectrum.fiberId}")
                plt.show()

    def _getMetadataName(self):
        """Disable writing metadata (doesn't work with lists of dataRefs anyway)"""
        return None
