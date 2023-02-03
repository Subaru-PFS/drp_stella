from typing import Optional

import lsstDebug
from lsst.afw.image import ExposureF
from lsst.pex.config import ConfigurableField, Field
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.butlerQuantumContext import ButlerQuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from pfs.datamodel import Identity, PfsConfig

from ..blackSpotCorrection import BlackSpotCorrectionTask
from ..DetectorMapContinued import DetectorMap
from ..extractSpectraTask import ExtractSpectraTask as ExtractionTask
from ..fiberProfileSet import FiberProfileSet
from ..fitContinuum import FitContinuumTask
from ..focalPlaneFunction import FocalPlaneFunction
from ..NevenPsfContinued import NevenPsf
from ..readLineList import ReadLineListTask
from ..repair import PfsRepairTask, maskLines
from ..subtractSky2d import SkyModel, SubtractSky2dTask

__all__ = ("ExtractSpectraTask",)


class ExtractSpectraConnections(PipelineTaskConnections, dimensions=("instrument", "exposure", "detector")):
    """Connections for ExtractSpectraTask"""

    exposure = InputConnection(
        name="postISRCCD",
        doc="Input ISR-corrected exposure",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "exposure"),
    )
    detectorMap = InputConnection(
        name="detectorMap_used",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "exposure", "detector"),
    )
    fiberProfiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc="Position and shape of fibers",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )
    sky2d = InputConnection(
        name="sky2d",
        doc="2D sky subtraction model",
        storageClass="SkyModel",
        dimensions=("instrument", "exposure", "arm"),
    )
    psf = InputConnection(
        name="psf",
        doc="2D point-spread function",
        storageClass="NevenPsf",
        dimensions=("instrument", "exposure", "detector"),
    )
    apCorr = InputConnection(
        name="apCorr",
        doc="Aperture correction for line photometry",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "exposure", "detector"),
    )

    calexp = OutputConnection(
        name="calexp",
        doc="Calibrated exposure, optionally sky-subtracted",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )
    pfsArm = OutputConnection(
        name="pfsArm",
        doc="Extracted spectra from arm",
        storageClass="PfsArm",
        dimensions=("instrument", "exposure", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.doSubtractSky2d:
            self.inputs.remove("sky2d")
            self.inputs.remove("psf")
            self.inputs.remove("apCorr")


class ExtractSpectraConfig(PipelineTaskConfig, pipelineConnections=ExtractSpectraConnections):
    """Configuration for ExtractSpectraTask"""

    doMaskLines = Field(dtype=bool, default=True, doc="Mask reference lines on image?")
    readLineList = ConfigurableField(target=ReadLineListTask, doc="Read reference lines")
    linesRadius = Field(dtype=int, default=2, doc="Radius around reference lines to mask (pixels)")
    doRepair = Field(dtype=bool, default=True, doc="Repair artifacts?")
    repair = ConfigurableField(target=PfsRepairTask, doc="Task to repair artifacts")
    doSubtractSky2d = Field(dtype=bool, default=False, doc="Subtract sky on 2D image?")
    subtractSky2d = ConfigurableField(target=SubtractSky2dTask, doc="2D sky subtraction")
    doSubtractContinuum = Field(dtype=bool, default=False, doc="Subtract continuum as part of extraction?")
    fitContinuum = ConfigurableField(target=FitContinuumTask, doc="Fit continuum for subtraction")
    doExtractSpectra = Field(dtype=bool, default=True, doc="Extract spectra from exposure?")
    extractSpectra = ConfigurableField(target=ExtractionTask, doc="Extract spectra from exposure")
    doBlackSpotCorrection = Field(dtype=bool, default=True, doc="Correct for black spot penumbra?")
    blackSpotCorrection = ConfigurableField(target=BlackSpotCorrectionTask, doc="Black spot correction")


class ExtractSpectraTask(PipelineTask):
    """Extract spectra from an exposure, optionally subtracting sky or continuum"""

    ConfigClass = ExtractSpectraConfig
    _DefaultName = "subtractSky2d"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)
        self.makeSubtask("readLineList")
        self.makeSubtask("repair")
        self.makeSubtask("subtractSky2d")
        self.makeSubtask("fitContinuum")
        self.makeSubtask("extractSpectra")
        self.makeSubtask("blackSpotCorrection")

    def runQuantum(
        self,
        butler: ButlerQuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Entry point with butler I/O

        Parameters
        ----------
        butler : `ButlerQuantumContext`
            Data butler, specialised to operate in the context of a quantum.
        inputRefs : `InputQuantizedConnection`
            Container with attributes that are data references for the various
            input connections.
        outputRefs : `OutputQuantizedConnection`
            Container with attributes that are data references for the various
            output connections.
        """
        inputs = butler.get(inputRefs)
        dataId = inputRefs.exposure.dataId.full
        inputs["identity"] = Identity(
            visit=dataId["exposure"],
            arm=dataId["arm"],
            spectrograph=dataId["spectrograph"],
            pfsDesignId=dataId["pfs_design_id"],
        )
        outputs = self.run(**inputs)
        butler.put(outputs, outputRefs)
        return outputs

    def run(
        self,
        exposure: ExposureF,
        identity: Identity,
        pfsConfig: PfsConfig,
        detectorMap: DetectorMap,
        fiberProfiles: FiberProfileSet,
        sky2d: Optional[SkyModel] = None,
        psf: Optional[NevenPsf] = None,
        apCorr: Optional[FocalPlaneFunction] = None,
    ) -> Struct:
        """Extract spectra from an exposure, optionally subtracting sky or
        continuum

        Parameters
        ----------
        exposure : `ExposureF`
            Exposure from which to extract spectra.
        identity : `Identity`
            Identity of the extracted spectra.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        detectorMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        fiberProfiles : `FiberProfileSet`
            Profile of fibers.
        sky2d : `SkyModel`, optional
            Model of sky line fluxes.
        psf : `NevenPsf`, optional
            Two-dimensional point-spread function, used for 2d sky subtraction.
        apCorr : `FocalPlaneFunction`, optional
            Aperture corrections, used for 2d sky subtraction.

        Returns
        -------
        calexp : `ExposureF`
            Exposure with any sky and continuum subtraction applied.
        pfsArm : `pfs.datamodel.PfsArm`
            Extracted spectra.
        original : `pfs.drp.stella.SpectrumSet`
            Extracted spectra before any continuum subtraction.
        spectra : pfs.drp.stella.SpectrumSet`
            Extracted spectra (different format as ``pfsArm``).
        continuum : `numpy.ndarray`
            Array of continuum fit.
        """
        maskedImage = exposure.maskedImage
        pfsConfig = pfsConfig.select(fiberId=detectorMap.fiberId)
        fiberId = pfsConfig.fiberId

        fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap)

        if self.config.doMaskLines:
            refLines = self.readLineList.run(detectorMap, exposure.getMetadata())
            maskLines(exposure.mask, detectorMap, refLines, self.config.linesRadius)

        if self.config.doRepair:
            if psf is None:
                psf = self.config.repair.interp.modelPsf.apply()
            exposure.setPsf(psf)
            self.repair.run(exposure)

        skyImage = None
        if self.config.doSubtractSky2d:
            if sky2d is None or psf is None or apCorr is None:
                raise RuntimeError("Can't do 2d sky subtraction without sky2d, psf and apCorr")
            skyImage = self.subtractSky(exposure, psf, fiberTraces, detectorMap, pfsConfig, sky2d, apCorr)

        original = self.extractSpectra.run(maskedImage, fiberTraces, detectorMap, fiberId).spectra

        continuum = None
        spectra = original
        if self.config.doSubtractContinuum:
            continuum = self.fitContinuum.run(original)
            maskedImage -= continuum.makeImage(exposure.getBBox(), fiberTraces)
            spectra = self.extractSpectra.run(maskedImage, fiberTraces, detectorMap, fiberId).spectra
            # Set sky flux from continuum
            for ss, cc in zip(spectra, continuum):
                ss.background += cc.spectrum

        if self.config.doBlackSpotCorrection:
            self.blackSpotCorrection.run(pfsConfig, original)

        # Set sky flux from realised 2d sky model
        if skyImage is not None:
            skySpectra = self.extractSpectra.run(skyImage, fiberTraces, detectorMap, fiberId).spectra
            for spec, skySpec in zip(spectra, skySpectra):
                spec.background += skySpec.spectrum

        return Struct(
            calexp=exposure,
            pfsArm=spectra.toPfsArm(identity.getDict()),
            original=original,
            spectra=spectra,
            continuum=continuum,
        )
