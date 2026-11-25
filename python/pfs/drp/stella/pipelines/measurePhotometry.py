import lsstDebug
from lsst.afw.image import Exposure
from lsst.daf.butler import DataCoordinate
from lsst.pex.config import ConfigurableField, DictField, Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from pfs.datamodel import FiberStatus, PfsConfig, TargetType

from ..DetectorMapContinued import DetectorMap
from ..fiberProfileSet import FiberProfileSet
from ..photometerLines import PhotometerLinesTask
from ..readLineList import ReadLineListTask

__all__ = ("MeasurePhotometryTask",)


class MeasurePhotometryConnections(
    PipelineTaskConnections, dimensions=("instrument", "visit", "arm", "spectrograph")
):
    """Connections for MeasurePhotometryTask"""

    exposure = InputConnection(
        name="calexp",
        doc="Input ISR-corrected exposure",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
    )
    detectorMap = InputConnection(
        name="detectorMap",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    fiberProfiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc="Profile of fibers",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )

    photometry = OutputConnection(
        name="photometry",
        doc="Emission line measurements",
        storageClass="ArcLineSet",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    apCorr = OutputConnection(
        name="apCorr",
        doc="Aperture correction for line photometry",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config:
            return
        if not config.doMeasureLines:
            self.outputs.remove("photometry")
            self.outputs.remove("apCorr")


class MeasurePhotometryConfig(PipelineTaskConfig, pipelineConnections=MeasurePhotometryConnections):
    """Configuration for MeasurePhotometryTask"""

    targetType = ListField(
        dtype=str,
        default=["SCIENCE", "SKY", "FLUXSTD", "SUNSS_IMAGING", "SUNSS_DIFFUSE"],
        doc="Target type for which to extract spectra",
    )
    doMeasureLines = Field(dtype=bool, default=True, doc="Measure emission lines (sky, arc)?")
    readLineList = ConfigurableField(target=ReadLineListTask, doc="Read line lists for photometry")
    photometerLines = ConfigurableField(target=PhotometerLinesTask, doc="Photometer lines")
    gaussianLsfWidth = DictField(
        keytype=str,
        itemtype=float,
        doc="Unused (this task no longer writes a placeholder pfsArmLsf); kept for old configs",
        default=dict(b=0.21, r=0.27, m=0.16, n=0.24),
    )


class MeasurePhotometryTask(PipelineTask):
    """Measure the LSF and line fluxes"""

    ConfigClass = MeasurePhotometryConfig
    _DefaultName = "measurePhotometry"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)
        self.makeSubtask("readLineList")
        self.makeSubtask("photometerLines")

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Entry point with butler I/O

        Parameters
        ----------
        butler : `QuantumContext`
            Data butler, specialised to operate in the context of a quantum.
        inputRefs : `InputQuantizedConnection`
            Container with attributes that are data references for the various
            input connections.
        outputRefs : `OutputQuantizedConnection`
            Container with attributes that are data references for the various
            output connections.
        """
        inputs = butler.get(inputRefs)
        dataId: DataCoordinate = inputRefs.exposure.dataId
        arm = dataId.arm.name
        spectrograph = dataId.spectrograph.num
        assert arm in "brnm"
        assert spectrograph in (1, 2, 3, 4)

        outputs = self.run(**inputs, arm=arm, spectrograph=spectrograph)
        if self.config.doMeasureLines:
            butler.put(outputs.photometry, outputRefs.photometry)
            butler.put(outputs.apCorr, outputRefs.apCorr)

    def run(
        self,
        exposure: Exposure,
        pfsConfig: PfsConfig,
        detectorMap: DetectorMap,
        fiberProfiles: FiberProfileSet,
        arm: str,
        spectrograph: int,
    ) -> Struct:
        """Measure the line fluxes

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image of spectra. Required for measuring a slit offset.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration. Required for measuring a slit offset.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
            Profile for each fiber.
        arm : `str`
            Spectrograph arm (``b``, ``r``, ``n``, ``m``).
        spectrograph : `int`
            Spectrograph module number (1-4).

        Returns
        -------
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Trace for each fiber.
        refLines : `pfs.drp.stella.ReferenceLineSet`
            Reference lines.
        photometry : `pfs.drp.stella.ArcLineSet`
            Measured lines.
        apCorr : `pfs.drp.stella.FocalPlaneFunction`
            Aperture correction.
        """
        lines = None
        apCorr = None

        self.checkFibers(pfsConfig, detectorMap, fiberProfiles, spectrograph)
        fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap)

        apCorr = None
        if self.config.doMeasureLines:
            refLines = self.readLineList.run(detectorMap, exposure.getMetadata(), arm=arm)
            phot = self.photometerLines.run(exposure, refLines, detectorMap, pfsConfig, fiberTraces)
            apCorr = phot.apCorr
            lines = phot.lines

        return Struct(
            fiberTraces=fiberTraces,
            refLines=refLines,
            photometry=lines,
            apCorr=apCorr,
        )

    def checkFibers(
        self,
        pfsConfig: PfsConfig,
        detectorMap: DetectorMap,
        fiberProfiles: FiberProfileSet,
        spectrograph: int,
    ) -> None:
        """Check that the calibs have the expected number of fibers

        Parameters
        ----------
        pfsConfig : `PfsConfig`
            Top-end fiber configuration.
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        fiberProfiles : `FiberProfileSet`
            Profile of each fiber.
        spectrograph : `int`
            Spectrograph module number (1-4).

        Raises
        ------
        RuntimeError
            If there is a mismatch between set of fibers required and the set
            of fibers that we have available in the detectorMap or
            fiberProfiles.
        """
        select = pfsConfig.getSelection(
            fiberStatus=FiberStatus.GOOD,
            targetType=[TargetType.fromString(tt) for tt in self.config.targetType],
            spectrograph=spectrograph,
        )
        fiberId = pfsConfig.fiberId[select]
        need = set(fiberId)
        haveDetMap = set(detectorMap.fiberId)
        haveProfiles = set(fiberProfiles.fiberId)
        missingDetMap = need - haveDetMap
        missingProfiles = need - haveProfiles
        if missingDetMap:
            raise RuntimeError(f"detectorMap does not include fibers: {list(sorted(missingDetMap))}")
        if need - haveProfiles:
            raise RuntimeError(f"fiberProfiles does not include fibers: {list(sorted(missingProfiles))}")
