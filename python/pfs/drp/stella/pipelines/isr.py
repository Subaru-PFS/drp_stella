from lsst.pipe.base import Struct, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.ip.isr.defects import Defects

from lsst.obs.pfs.isrTask import PfsIsrTask, PfsIsrTaskConfig
import lsst.afw.image as afwImage

__all__ = ("IsrTask",)


def lookupDefects(datasetType, registry, dataId, collections):
    """Look up defects

    Parameters
    ----------
    datasetType : `str`
        The dataset type to look up.
    registry : `lsst.daf.butler.Registry`
        The butler registry.
    dataId : `lsst.daf.butler.DataCoordinate`
        The data identifier.
    collections : `list` of `str`
        The collections to search.

    Returns
    -------
    refs : `list` of `lsst.daf.butler.Reference`
        The references to the bias or dark frame.
    """
    results = list(registry.queryDimensionRecords("detector", dataId=dataId))
    if len(results) != 1:
        raise RuntimeError(f"Unable to find detector for {dataId}: {results}")
    detector = results[0].id

    return [registry.findDataset(
        datasetType, collections=collections, dataId=dataId, timespan=dataId.timespan, detector=detector,
    )]


def lookupCrosstalkSources(*args, **kwargs):
    return []


class IsrConnections(PipelineTaskConnections, dimensions=("instrument", "visit", "arm", "spectrograph")):
    """Connections for IsrTask"""

    ccdExposure = InputConnection(
        name="raw.exposure",
        doc="Input exposure to process.",
        storageClass="Exposure",
        dimensions=["instrument", "visit", "arm", "spectrograph"],
    )
    camera = PrerequisiteConnection(
        name="camera",
        storageClass="Camera",
        doc="Input camera to construct complete exposures.",
        dimensions=["instrument"],
        isCalibration=True,
    )

    crosstalk = PrerequisiteConnection(
        name="crosstalk",
        doc="Input crosstalk object",
        storageClass="CrosstalkCalib",
        dimensions=["instrument", "arm", "spectrograph"],
        isCalibration=True,
        minimum=0,  # can fall back to cameraGeom
    )
    crosstalkSources = PrerequisiteConnection(
        name="isrOverscanCorrected",
        doc="Overscan corrected input images.",
        storageClass="Exposure",
        dimensions=["instrument", "visit", "arm", "spectrograph"],
        deferLoad=True,
        multiple=True,
        lookupFunction=lookupCrosstalkSources,
        minimum=0,  # not needed for all instruments, no config to control this
    )
    bias = PrerequisiteConnection(
        name="bias",
        doc="Input bias calibration.",
        storageClass="ExposureF",
        dimensions=["instrument", "arm", "spectrograph"],
        isCalibration=True,
    )
    dark = PrerequisiteConnection(
        name='dark',
        doc="Input dark calibration.",
        storageClass="ExposureF",
        dimensions=["instrument", "arm", "spectrograph"],
        isCalibration=True,
    )
    flat = PrerequisiteConnection(
        name="fiberFlat",
        doc="Combined flat",
        storageClass="ExposureF",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    ptc = PrerequisiteConnection(
        name="ptc",
        doc="Input Photon Transfer Curve dataset",
        storageClass="PhotonTransferCurveDataset",
        dimensions=["instrument", "arm", "spectrograph"],
        isCalibration=True,
    )
    fringes = PrerequisiteConnection(
        name="fringe",
        doc="Input fringe calibration.",
        storageClass="ExposureF",
        dimensions=["instrument", "physical_filter", "arm", "spectrograph"],
        isCalibration=True,
        minimum=0,  # only needed for some bands, even when enabled
    )
    strayLightData = PrerequisiteConnection(
        name='yBackground',
        doc="Input stray light calibration.",
        storageClass="StrayLightData",
        dimensions=["instrument", "physical_filter", "arm", "spectrograph"],
        deferLoad=True,
        isCalibration=True,
        minimum=0,  # only needed for some bands, even when enabled
    )
    bfKernel = PrerequisiteConnection(
        name='bfKernel',
        doc="Input brighter-fatter kernel.",
        storageClass="NumpyArray",
        dimensions=["instrument"],
        isCalibration=True,
        minimum=0,  # can use either bfKernel or newBFKernel
    )
    newBFKernel = PrerequisiteConnection(
        name='brighterFatterKernel',
        doc="Newer complete kernel + gain solutions.",
        storageClass="BrighterFatterKernel",
        dimensions=["instrument", "arm", "spectrograph"],
        isCalibration=True,
        minimum=0,  # can use either bfKernel or newBFKernel
    )
    defects = PrerequisiteConnection(
        name='defects',
        doc="Input defect tables.",
        storageClass="Defects",
        dimensions=["instrument", "detector", "arm", "spectrograph"],
        isCalibration=True,
        lookupFunction=lookupDefects,
    )
    linearizer = PrerequisiteConnection(
        name='linearizer',
        storageClass="Linearizer",
        doc="Linearity correction calibration.",
        dimensions=["instrument", "arm", "spectrograph"],
        isCalibration=True,
        minimum=0,  # can fall back to cameraGeom
    )
    opticsTransmission = PrerequisiteConnection(
        name="transmission_optics",
        storageClass="TransmissionCurve",
        doc="Transmission curve due to the optics.",
        dimensions=["instrument"],
        isCalibration=True,
    )
    filterTransmission = PrerequisiteConnection(
        name="transmission_filter",
        storageClass="TransmissionCurve",
        doc="Transmission curve due to the filter.",
        dimensions=["instrument", "physical_filter"],
        isCalibration=True,
    )
    sensorTransmission = PrerequisiteConnection(
        name="transmission_sensor",
        storageClass="TransmissionCurve",
        doc="Transmission curve due to the sensor.",
        dimensions=["instrument", "arm", "spectrograph"],
        isCalibration=True,
    )
    atmosphereTransmission = PrerequisiteConnection(
        name="transmission_atmosphere",
        storageClass="TransmissionCurve",
        doc="Transmission curve due to the atmosphere.",
        dimensions=["instrument"],
        isCalibration=True,
    )
    illumMaskedImage = PrerequisiteConnection(
        name="illum",
        doc="Input illumination correction.",
        storageClass="MaskedImageF",
        dimensions=["instrument", "physical_filter", "arm", "spectrograph"],
        isCalibration=True,
    )
    deferredChargeCalib = PrerequisiteConnection(
        name="cpCtiCalib",
        doc="Deferred charge/CTI correction dataset.",
        storageClass="IsrCalib",
        dimensions=["instrument", "arm", "spectrograph"],
        isCalibration=True,
    )

    outputExposure = OutputConnection(
        name='postISRCCD',
        doc="Output ISR processed exposure.",
        storageClass="Exposure",
        dimensions=["instrument", "visit", "arm", "spectrograph"],
    )
    preInterpExposure = OutputConnection(
        name='preInterpISRCCD',
        doc="Output ISR processed exposure, with pixels left uninterpolated.",
        storageClass="ExposureF",
        dimensions=["instrument", "visit", "arm", "spectrograph"],
    )
    outputOssThumbnail = OutputConnection(
        name="OssThumb",
        doc="Output Overscan-subtracted thumbnail image.",
        storageClass="Thumbnail",
        dimensions=["instrument", "visit", "arm", "spectrograph"],
    )
    outputFlattenedThumbnail = OutputConnection(
        name="FlattenedThumb",
        doc="Output flat-corrected thumbnail image.",
        storageClass="Thumbnail",
        dimensions=["instrument", "visit", "arm", "spectrograph"],
    )
    outputStatistics = OutputConnection(
        name="isrStatistics",
        doc="Output of additional statistics table.",
        storageClass="StructuredDataDict",
        dimensions=["instrument", "visit", "arm", "spectrograph"],
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config.doBias is not True:
            self.prerequisiteInputs.remove("bias")
        if config.doLinearize is not True:
            self.prerequisiteInputs.remove("linearizer")
        if config.doCrosstalk is not True:
            self.prerequisiteInputs.remove("crosstalkSources")
            self.prerequisiteInputs.remove("crosstalk")
        if config.doBrighterFatter is not True:
            self.prerequisiteInputs.remove("bfKernel")
            self.prerequisiteInputs.remove("newBFKernel")
        if config.doDefect is not True:
            self.prerequisiteInputs.remove("defects")
        if config.doDark is not True:
            self.prerequisiteInputs.remove("dark")
        if config.doFlat is not True:
            self.prerequisiteInputs.remove("flat")
        if config.doFringe is not True:
            self.prerequisiteInputs.remove("fringes")
        if config.doStrayLight is not True:
            self.prerequisiteInputs.remove("strayLightData")
        if config.usePtcGains is not True and config.usePtcReadNoise is not True:
            self.prerequisiteInputs.remove("ptc")
        if config.doAttachTransmissionCurve is not True:
            self.prerequisiteInputs.remove("opticsTransmission")
            self.prerequisiteInputs.remove("filterTransmission")
            self.prerequisiteInputs.remove("sensorTransmission")
            self.prerequisiteInputs.remove("atmosphereTransmission")
        else:
            if config.doUseOpticsTransmission is not True:
                self.prerequisiteInputs.remove("opticsTransmission")
            if config.doUseFilterTransmission is not True:
                self.prerequisiteInputs.remove("filterTransmission")
            if config.doUseSensorTransmission is not True:
                self.prerequisiteInputs.remove("sensorTransmission")
            if config.doUseAtmosphereTransmission is not True:
                self.prerequisiteInputs.remove("atmosphereTransmission")
        if config.doIlluminationCorrection is not True:
            self.prerequisiteInputs.remove("illumMaskedImage")
        if config.doDeferredCharge is not True:
            self.prerequisiteInputs.remove("deferredChargeCalib")

        if config.doWrite is not True:
            self.outputs.remove("outputExposure")
            self.outputs.remove("preInterpExposure")
            self.outputs.remove("outputFlattenedThumbnail")
            self.outputs.remove("outputOssThumbnail")
            self.outputs.remove("outputStatistics")

        if config.doSaveInterpPixels is not True:
            self.outputs.remove("preInterpExposure")
        if config.qa.doThumbnailOss is not True:
            self.outputs.remove("outputOssThumbnail")
        if config.qa.doThumbnailFlattened is not True:
            self.outputs.remove("outputFlattenedThumbnail")
        if config.doCalculateStatistics is not True:
            self.outputs.remove("outputStatistics")


class IsrConfig(PfsIsrTaskConfig, pipelineConnections=IsrConnections):
    """Configuration for IsrTask"""

    def setDefaults(self):
        super().setDefaults()
        self.doLinearize = False
        self.doCrosstalk = False
        self.doBrighterFatter = False
        self.doFringe = False
        self.doStrayLight = False


class IsrTask(PfsIsrTask):
    """Perform instrumental signature removal and repair artifacts"""

    ConfigClass = IsrConfig

    def ensureExposure(self, inputExp, camera=None, detectorNum=None):
        if isinstance(inputExp, afwImage.DecoratedImageU):
            inputExp = afwImage.makeExposure(afwImage.makeMaskedImage(inputExp))
        elif isinstance(inputExp, afwImage.ImageF):
            inputExp = afwImage.makeExposure(afwImage.makeMaskedImage(inputExp))
        elif isinstance(inputExp, afwImage.MaskedImageF):
            inputExp = afwImage.makeExposure(inputExp)
        elif isinstance(inputExp, afwImage.Exposure):
            pass
        elif inputExp is None:
            # Assume this will be caught by the setup if it is a problem.
            return inputExp
        else:
            raise TypeError("Input Exposure is not known type in isrTask.ensureExposure: %s." %
                            (type(inputExp), ))
        return inputExp

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        assert self.config.doCrosstalk is False
        assert self.config.doLinearize is False

        if self.config.doDefect is True:
            if "defects" in inputs and inputs['defects'] is not None:
                # defects is loaded as a BaseCatalog with columns
                # x0, y0, width, height. Masking expects a list of defects
                # defined by their bounding box
                if not isinstance(inputs["defects"], Defects):
                    inputs["defects"] = Defects.fromTable(inputs["defects"])

        assert self.config.doBrighterFatter is False

        assert self.config.doFringe is False
        inputs['fringes'] = Struct(fringes=None)

        assert self.config.doStrayLight is False

        if self.config.doHeaderProvenance:
            # Add calibration provenanace info to header.
            exposureMetadata = inputs['ccdExposure'].getMetadata()
            for inputName in sorted(inputs.keys()):
                reference = getattr(inputRefs, inputName, None)
                if reference is not None and hasattr(reference, "run"):
                    runKey = f"LSST CALIB RUN {inputName.upper()}"
                    runValue = reference.run
                    idKey = f"LSST CALIB UUID {inputName.upper()}"
                    idValue = str(reference.id)

                    exposureMetadata[runKey] = runValue
                    exposureMetadata[idKey] = idValue

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)
