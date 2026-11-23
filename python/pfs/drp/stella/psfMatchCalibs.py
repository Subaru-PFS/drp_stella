import dataclasses
from functools import partial
from typing import Optional

import numpy as np
import scipy.ndimage

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.afw.detection import Psf
from lsst.afw.image import Exposure, VisitInfo
from lsst.daf.base import PropertyList
from lsst.geom import Box2D
from lsst.pex.config import ChoiceField, ConfigurableField, Field
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

from pfs.datamodel import CalibIdentity

from .adjustDetectorMap import AdjustDetectorMapTask
from .arcLine import ArcLineSet
from .calibs import setCalibHeader
from .centroidLines import CentroidLinesTask
from .centroidTraces import CentroidTracesTask
from .datamodel import PfsConfig
from .DetectorMapContinued import DetectorMap
from .fiberProfile import FiberProfile
from .fiberProfileSet import FiberProfileSet
from .OpticalModel import OpticsModel
from .OpticalModelDetectorMapContinued import OpticalModelDetectorMap
from .pipelines.lookups import lookupDatasetsForArm
from .PolynomialDistortionContinued import PolynomialDistortion
from .psfBias import calculatePeakToCentroidBias
from .psfMatch import PsfMatchTask, calculateFirstMoments, calculateWarpShifts
from .readLineList import ReadLineListTask
from .referenceLine import ReferenceLineStatus
from .utils.math import robustRms
from .utils.psf import makeGaussianPsf

__all__ = (
    "PsfMatchCalibsTask",
    "calculatePeakShifts",
    "addOpticsDistortion",
    "convolveFiberProfile",
)


def calculatePeakShifts(
    kernel: afwMath.Kernel,
    psf: Psf,
    xx: np.ndarray,
    yy: np.ndarray,
    **kwargs,
) -> Struct:
    """Calculate the shift in peak position due to a PSF-matching kernel

    The kernel matches a reference exposure (with PSF ``psf``) to a target
    exposure. The centroid shifts by the (negative) first moment of the kernel
    (because `lsst.afw.math.convolve` is a correlation), and the peak shifts by
    that plus the change in the peak-to-centroid bias between the reference
    PSF and the reference PSF convolved by the kernel.

    Parameters
    ----------
    kernel : `lsst.afw.math.Kernel`
        PSF-matching kernel that takes the reference to the target.
    psf : `lsst.afw.detection.Psf`
        PSF of the reference exposure.
    xx, yy : `numpy.ndarray` of `float`
        Positions (in the reference) at which to calculate the shifts.
    **kwargs
        Additional arguments for `calculatePeakToCentroidBias`.

    Returns
    -------
    dx, dy : `numpy.ndarray` of `float`
        Shift in peak position from the reference to the target.
    dxCentroid, dyCentroid : `numpy.ndarray` of `float`
        Shift in centroid position from the reference to the target.
    xBiasRef, yBiasRef : `numpy.ndarray` of `float`
        Peak minus centroid for the reference.
    xBiasTarget, yBiasTarget : `numpy.ndarray` of `float`
        Peak minus centroid for the target.
    kernelSum : `numpy.ndarray` of `float`
        Sum of the kernel at each position.
    """
    moments = calculateFirstMoments(kernel, xx, yy)
    xBiasRef, yBiasRef = calculatePeakToCentroidBias(psf, xx, yy, **kwargs)
    xBiasTarget, yBiasTarget = calculatePeakToCentroidBias(psf, xx, yy, kernel, **kwargs)
    dxCentroid = -moments.x
    dyCentroid = -moments.y
    return Struct(
        dx=dxCentroid + xBiasTarget - xBiasRef,
        dy=dyCentroid + yBiasTarget - yBiasRef,
        dxCentroid=dxCentroid,
        dyCentroid=dyCentroid,
        xBiasRef=xBiasRef,
        yBiasRef=yBiasRef,
        xBiasTarget=xBiasTarget,
        yBiasTarget=yBiasTarget,
        kernelSum=moments.kernelSum,
    )


def addOpticsDistortion(
    detectorMap: DetectorMap,
    distortion: PolynomialDistortion,
    visitInfo: Optional[VisitInfo] = None,
    metadata: Optional[PropertyList] = None,
) -> OpticalModelDetectorMap:
    """Add a distortion to the optics model of a detectorMap

    The distortions of the optics model are applied in sequence on top of the
    optical model grid (in detector coordinates), and are preserved by
    `AdjustDetectorMapTask` (which replaces only the slit distortion).

    Parameters
    ----------
    detectorMap : `pfs.drp.stella.DetectorMap`
        Mapping from fiberId,wavelength to x,y. This will be converted to an
        `OpticalModelDetectorMap` if necessary.
    distortion : `pfs.drp.stella.PolynomialDistortion`
        Distortion (in detector coordinates) to add.
    visitInfo : `lsst.afw.image.VisitInfo`, optional
        Visit information for the new detectorMap. If not provided, we use the
        ``visitInfo`` of ``detectorMap``.
    metadata : `lsst.daf.base.PropertyList`, optional
        Metadata for the new detectorMap. If not provided, we use the
        ``metadata`` of ``detectorMap``.

    Returns
    -------
    detectorMap : `pfs.drp.stella.OpticalModelDetectorMap`
        Mapping from fiberId,wavelength to x,y, with the distortion applied.
    """
    detectorMap = OpticalModelDetectorMap.fromDetectorMap(detectorMap)
    optics = detectorMap.opticsModel
    newOptics = OpticsModel(
        optics.getSpatial(),
        optics.getSpectral(),
        optics.getX(),
        optics.getY(),
        [dd.clone() for dd in optics.getDistortions()] + [distortion],
    )
    return OpticalModelDetectorMap(
        detectorMap.slitModel.copy(),
        newOptics,
        detectorMap.detectorModel.copy(),
        visitInfo if visitInfo is not None else detectorMap.visitInfo,
        metadata if metadata is not None else detectorMap.metadata,
    )


def convolveFiberProfile(
    profile: FiberProfile,
    kernel: afwMath.Kernel,
    xOld: np.ndarray,
    xNew: np.ndarray,
    interpOrder: int = 3,
) -> FiberProfile:
    """Convolve a fiber profile with a PSF-matching kernel, and shift it

    For each swath, the kernel is realised at the position of the fiber
    (``xOld``, ``profile.rows``) and collapsed in y to produce a 1D kernel in
    x, normalised to unit sum. The profile is correlated with this 1D kernel
    (matching the `lsst.afw.math.convolve` convention); because the kernel is
    pixel-sampled and the profile is oversampled by an integer factor, this
    requires no interpolation. The convolved profile is then shifted so that it
    is relative to the new trace position (``xNew``), so that the modeled image
    of the fiber is the original model convolved by the kernel.

    The normalisation (``profile.norm``) is copied unchanged: the shift in the
    spectral direction is small, and the spectrum (typically from a quartz) is
    smooth.

    Parameters
    ----------
    profile : `pfs.drp.stella.FiberProfile`
        Fiber profile to convolve.
    kernel : `lsst.afw.math.Kernel`
        PSF-matching kernel.
    xOld : `numpy.ndarray` of `float`
        Trace center for each swath (``profile.rows``) in the original
        detectorMap.
    xNew : `numpy.ndarray` of `float`
        Trace center for each swath in the new detectorMap.
    interpOrder : `int`, optional
        Spline order for the sub-pixel shift.

    Returns
    -------
    convolved : `pfs.drp.stella.FiberProfile`
        Convolved and shifted fiber profile.
    """
    oversample = profile.oversample
    data = profile.profiles.filled(0.0)
    mask = np.ma.getmaskarray(profile.profiles)
    numSwaths, profileSize = data.shape
    convolved = np.zeros_like(data, dtype=float)
    newMask = np.zeros_like(mask, dtype=bool)

    image = afwImage.ImageD(kernel.getDimensions())
    center = kernel.getCtr().getX()
    for ii, (xx, yy) in enumerate(zip(xOld, profile.rows)):
        if not np.isfinite(xx) or not np.isfinite(xNew[ii]):
            newMask[ii] = True
            continue
        kernel.computeImage(image, False, float(xx), float(yy))
        kernel1d = image.array.sum(axis=0)
        kernel1d /= kernel1d.sum()

        # Correlation: convolved(u) = sum_k kernel1d[k] profile(u + k - center)
        values = np.zeros(profileSize, dtype=float)
        masked = np.zeros(profileSize, dtype=bool)
        for kk, kernelValue in enumerate(kernel1d):
            offset = (kk - center) * oversample
            if abs(offset) >= profileSize:
                continue
            if offset >= 0:
                target = slice(0, profileSize - offset)
                source = slice(offset, profileSize)
            else:
                target = slice(-offset, profileSize)
                source = slice(0, profileSize + offset)
            values[target] += kernelValue * data[ii, source]
            if kernelValue != 0:
                masked[target] |= mask[ii, source]

        # Shift so the profile is relative to the new trace: shifted(u) = convolved(u + xNew - xOld)
        traceShift = xNew[ii] - xx
        shift = -traceShift * oversample
        convolved[ii] = scipy.ndimage.shift(values, shift, order=interpOrder, mode="constant", cval=0.0)
        newMask[ii] = scipy.ndimage.shift(masked.astype(float), shift, order=1, mode="constant", cval=0.0) > 0

    convolved[newMask] = np.nan
    return FiberProfile(
        profile.radius,
        profile.oversample,
        np.array(profile.rows, copy=True),
        convolved,
        None if profile.norm is None else np.array(profile.norm, copy=True),
    )


def selectReferenceVisit(visits: list[int], referenceIsLater: bool) -> tuple[int, int]:
    """Identify the reference and target visits

    Parameters
    ----------
    visits : `list` of `int`
        Visits in the quantum; there should be two distinct visits.
    referenceIsLater : `bool`
        Is the reference the visit with the larger visit number?

    Returns
    -------
    referenceVisit, targetVisit : `int`
        Reference and target visits.
    """
    unique = sorted(set(visits))
    if len(unique) != 2:
        raise RuntimeError(f"Expected two visits (reference and target), found {unique}")
    return (unique[1], unique[0]) if referenceIsLater else (unique[0], unique[1])


class PsfMatchCalibsConnections(PipelineTaskConnections, dimensions=("instrument", "arm", "spectrograph")):
    """Connections for PsfMatchCalibsTask"""

    exposures = InputConnection(
        name="calexp",
        doc="Reference and target exposures (arcs or sky)",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    pfsConfigs = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration, one per visit",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
        multiple=True,
    )
    detectorMaps = InputConnection(
        name="detectorMap_forVisit",
        doc="Calib mapping from fiberId,wavelength to x,y valid for each visit (from CalibsForVisitTask). "
        "That for the reference is adjusted to make the output; that for the target is used for the warp "
        "(``config.doWarp``).",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    fiberProfiles = InputConnection(
        name="fiberProfiles_forVisit",
        doc="Calib profiles of fibers valid for each visit (from CalibsForVisitTask); only that for the "
        "reference is used",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
        deferGraphConstraint=True,  # Only needed for the reference
    )
    psfs = InputConnection(
        name="imagePsf_forVisit",
        doc="Calib PSF model valid for each visit (from CalibsForVisitTask); only that for the reference is "
        "used",
        storageClass="Psf",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
        deferGraphConstraint=True,  # Only needed for the reference
    )
    otherArmKernel = PrerequisiteConnection(
        name="psfMatchKernel",
        doc="PSF-matching kernel measured for another arm (``config.kernelArm``), used instead of measuring "
        "one",
        storageClass="MatchingKernel",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,  # One per visit; runQuantum selects the target visit
    )
    outputDetectorMap = OutputConnection(
        name="detectorMap_candidate",
        doc="Mapping from fiberId,wavelength to x,y, adjusted for the target",
        storageClass="DetectorMap",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    outputFiberProfiles = OutputConnection(
        name="fiberProfiles_candidate",
        doc="Profiles of fibers, adjusted for the target",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    outputKernel = OutputConnection(
        name="psfMatchKernel",
        doc="PSF-matching kernel from the reference to the target",
        storageClass="MatchingKernel",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,  # The quantum has no visit dimension; adjustQuantum selects the target visit
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if config is not None:
            if config.kernelArm is None:
                self.prerequisiteInputs.remove("otherArmKernel")
            else:
                lookup = partial(lookupDatasetsForArm, arm=config.kernelArm)
                self.otherArmKernel = dataclasses.replace(self.otherArmKernel, lookupFunction=lookup)
                # The kernel is the input from the other arm, so there's no need to write it again (and a task
                # can't both read and write the same dataset type).
                self.outputs.remove("outputKernel")

    def adjustQuantum(self, inputs, outputs, label, dataId):
        """Remove the reference visit from the kernel outputs

        The kernel output has a ``visit`` dimension, which isn't one of the
        quantum dimensions, so the quantum graph builder generates a kernel
        output for each visit. We only want the kernel for the target visit
        (i.e., not the reference visit).
        """
        if "outputKernel" not in outputs:
            return super().adjustQuantum(inputs, outputs, label, dataId)
        _, exposureRefs = inputs["exposures"]
        referenceVisit, _ = selectReferenceVisit(
            [ref.dataId["visit"] for ref in exposureRefs], self.config.referenceIsLater
        )
        connection, oldRefs = outputs["outputKernel"]
        newRefs = [ref for ref in oldRefs if ref.dataId["visit"] != referenceVisit]
        adjusted = {"outputKernel": (connection, newRefs)}
        outputs.update(adjusted)
        super().adjustQuantum(inputs, outputs, label, dataId)
        return {}, adjusted


class PsfMatchCalibsConfig(PipelineTaskConfig, pipelineConnections=PsfMatchCalibsConnections):
    """Configuration for PsfMatchCalibsTask"""

    referenceIsLater = Field(
        dtype=bool,
        optional=True,
        default=None,
        doc="Is the reference the visit with the larger visit number (the other is the target)? Required.",
    )
    lineMode = ChoiceField(
        dtype=str,
        default="auto",
        allowed={
            "auto": "Use the line list if the target has no lamps active (i.e., sky)",
            "arc": "Don't use the line list",
            "sky": "Use the line list",
        },
        doc="Whether to use the line list when measuring the PSF-matching kernel",
    )
    kernelArm = ChoiceField(
        dtype=str,
        optional=True,
        default=None,
        allowed={"b": "blue", "r": "red", "n": "near-infrared", "m": "medium-resolution red"},
        doc="Arm whose PSF-matching kernel (measured by a previous run of this task, for the same "
        "spectrograph, reference and target) to use instead of measuring one for this arm; e.g., 'r' when "
        "processing 'b', which has few lines in sky exposures. If None, measure the kernel for this arm.",
    )
    doAdjustReference = Field(
        dtype=bool,
        default=True,
        doc="Adjust the detectorMap to the reference exposure before applying the kernel?",
    )
    doWarp = Field(
        dtype=bool,
        default=True,
        doc="Warp the target onto the reference before measuring the PSF-matching kernel, using the "
        "calib detectorMap valid for the target and the (adjusted) calib detectorMap for the reference? "
        "The kernel then measures the change since the target's calib detectorMap was made, and the shift "
        "of the warp is added to the kernel's shift. This allows large and variable shifts. A kernel for "
        "another arm (``kernelArm``) must have been measured with the same setting.",
    )
    doCheckTarget = Field(
        dtype=bool,
        default=True,
        doc="Measure lines and traces on the target, and compare them with the new detectorMap?",
    )
    readLineList = ConfigurableField(target=ReadLineListTask, doc="Read line list")
    centroidLines = ConfigurableField(target=CentroidLinesTask, doc="Centroid lines on reference")
    doForceTraces = Field(dtype=bool, default=True, doc="Centroid traces on reference, even for arcs?")
    centroidTraces = ConfigurableField(target=CentroidTracesTask, doc="Centroid traces on reference")
    adjustReference = ConfigurableField(
        target=AdjustDetectorMapTask, doc="Adjust the detectorMap to the reference exposure"
    )
    psfMatch = ConfigurableField(target=PsfMatchTask, doc="Measure PSF-matching kernel")
    numFibers = Field(dtype=int, default=40, doc="Number of fibers in grid for measuring peak shifts")
    numRows = Field(dtype=int, default=40, doc="Number of rows in grid for measuring peak shifts")
    gridMargin = Field(dtype=int, default=20, doc="Margin from top and bottom of detector for grid (pixels)")
    distortionOrder = Field(dtype=int, default=3, doc="Polynomial order for distortion fit to peak shifts")
    rejection = Field(dtype=float, default=4.0, doc="Rejection threshold for distortion fit (stdev)")
    iterations = Field(dtype=int, default=3, doc="Number of rejection iterations for distortion fit")
    profileInterpOrder = Field(dtype=int, default=3, doc="Spline order for shifting fiber profiles")

    def setDefaults(self):
        super().setDefaults()
        # A linear spatial kernel can't follow the curvature in the change of position with focus,
        # especially at the edges of the detector
        self.psfMatch.dfKernel.spatialKernelOrder = 3

    def validate(self):
        super().validate()
        if self.referenceIsLater is None:
            raise ValueError("referenceIsLater must be set")


class PsfMatchCalibsTask(PipelineTask):
    """Adjust the detectorMap and fiberProfiles for a change in focus

    When the focus changes, both the positions and the PSF change. The arc
    line centroider measures peak positions, and the difference between the
    peak and the centroid depends on the PSF, so the change in peak positions
    may be of high order even when the change in centroid positions is low
    order.

    We measure the PSF-matching kernel from the reference exposure (with a
    known PSF model, and the same focus as the input detectorMap and
    fiberProfiles) to the target exposure. The detectorMap is first adjusted
    to the reference exposure (a low-order distortion), so that any drift
    between the input detectorMap and the reference is not inherited by the
    output. The first moment of the kernel provides the change in centroid
    positions, and the PSF model convolved with the kernel provides the change
    in the peak-to-centroid bias. The resulting change in peak positions is
    fit with a polynomial distortion that is added to the optics model of the
    detectorMap (which is referenced to peak positions). The fiberProfiles are
    convolved with the kernel, and shifted to be relative to the new (peak)
    trace positions.

    The shift between the reference and the target can be large (several
    pixels) and vary over the detector, which a kernel of reasonable size
    can't represent. With ``config.doWarp`` (the default), the target is
    warped onto the reference before measuring the kernel. The warp uses the
    calib detectorMap valid for the target (not adjusted to the target, so
    the kernel measures the change since that detectorMap was made) and the
    adjusted calib detectorMap for the reference. The kernel then takes the
    reference to the warped target, and the inverse of the warp takes the
    warped target to the target: the shift of the warp is added to the shift
    of the kernel when measuring the change in peak positions, and removed
    when shifting the convolved fiberProfiles (since the kernel doesn't
    include it). The output is the reference's detectorMap with the
    distortion fit to these combined shifts; errors in the target's calib
    detectorMap are measured by the kernel, and so cancel.

    The blue arm has too few lines in sky exposures to measure a good kernel.
    Instead, ``config.kernelArm`` can specify another arm (e.g., ``r``) whose
    kernel (from a previous run of this task, with the same reference and
    target visits) is used in place of measuring one; it is combined with the
    PSF model, detectorMaps and fiberProfiles for this arm. The kernel is
    evaluated at the same pixel position: the same fiber is only a few tens of
    pixels from the same position in the other arm, and the kernel varies
    slowly. This assumes that the change since the calib detectorMaps for the
    target were made (focus and position) is the same in both arms, and that
    those detectorMaps are equally good in both arms. The kernel is not
    written for this arm in that case.

    With ``config.doCheckTarget``, lines and traces are measured on the
    target using the new detectorMap and fiberProfiles, and their residuals
    from the new detectorMap are logged: this is an independent check of the
    outputs against the data.

    In a pipeline, the data query should select two visits: the reference
    and the target; ``config.referenceIsLater`` says which is which. The
    calibs valid for each visit are provided by `CalibsForVisitTask`: the
    detectorMap, fiberProfiles and PSF model for the reference, and the
    detectorMap for the target. The outputs are written as candidates
    (``detectorMap_candidate``, ``fiberProfiles_candidate``), which can be
    certified using ``certifyDetectorMaps.py`` and ``certifyFiberProfiles.py``.
    """

    ConfigClass = PsfMatchCalibsConfig
    _DefaultName = "psfMatchCalibs"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("readLineList")
        self.makeSubtask("centroidLines")
        self.makeSubtask("centroidTraces")
        self.makeSubtask("adjustReference")
        self.makeSubtask("psfMatch")

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
        exposureRefs = inputRefs.exposures
        if len(exposureRefs) != 2:
            raise RuntimeError(f"Expected 2 exposures, found {len(exposureRefs)}")
        referenceVisit, targetVisit = selectReferenceVisit(
            [ref.dataId["visit"] for ref in exposureRefs], self.config.referenceIsLater
        )
        self.log.info("Reference visit %d, target visit %d", referenceVisit, targetVisit)
        exposureRefs = {ref.dataId["visit"]: ref for ref in exposureRefs}
        referenceRef = exposureRefs[referenceVisit]
        targetRef = exposureRefs[targetVisit]

        def getForVisit(refs, visit: int, name: str):
            """Get the dataset for a visit from a list of references"""
            selected = [ref for ref in refs if ref.dataId["visit"] == visit]
            if len(selected) != 1:
                raise RuntimeError(f"Expected one {name} for visit {visit}, found {len(selected)}")
            return butler.get(selected[0])

        pfsConfigRefs = {ref.dataId["visit"]: ref for ref in inputRefs.pfsConfigs}

        kernel = None
        if self.config.kernelArm is not None:
            kernelRefs = [ref for ref in inputRefs.otherArmKernel if ref.dataId["visit"] == targetVisit]
            if len(kernelRefs) != 1:
                raise RuntimeError(
                    f"Expected one PSF-matching kernel for arm {self.config.kernelArm} and visit "
                    f"{targetVisit}, found {len(kernelRefs)}"
                )
            kernel = butler.get(kernelRefs[0])

        if kernel is not None:
            useLineList = False  # We're not measuring the kernel
        elif self.config.lineMode == "auto":
            useLineList = not targetRef.dataId.records["visit"].lamps
        elif self.config.lineMode == "sky":
            useLineList = True
        elif self.config.lineMode == "arc":
            useLineList = False
        else:
            raise ValueError(f"Unrecognised lineMode: {self.config.lineMode}")

        # pfsConfigs are required with the line list (for excluding fibers), but are otherwise optional
        missingPfsConfig = {referenceVisit, targetVisit} - set(pfsConfigRefs)
        if useLineList and missingPfsConfig:
            raise RuntimeError(f"Missing pfsConfig for visits {sorted(missingPfsConfig)}")
        pfsConfigs = {
            visit: butler.get(pfsConfigRefs[visit]) if visit in pfsConfigRefs else None
            for visit in (referenceVisit, targetVisit)
        }

        targetDetectorMap = None
        if self.config.doWarp:
            targetDetectorMap = getForVisit(inputRefs.detectorMaps, targetVisit, "detectorMap")

        dataId = targetRef.dataId
        identity = CalibIdentity(
            obsDate=dataId.timespan.begin.to_datetime().date().isoformat(),
            spectrograph=dataId["spectrograph"],
            arm=dataId["arm"],
            visit0=targetVisit,
        )

        outputs = self.run(
            reference=butler.get(referenceRef),
            target=butler.get(targetRef),
            psf=getForVisit(inputRefs.psfs, referenceVisit, "imagePsf"),
            detectorMap=getForVisit(inputRefs.detectorMaps, referenceVisit, "detectorMap"),
            fiberProfiles=getForVisit(inputRefs.fiberProfiles, referenceVisit, "fiberProfiles"),
            identity=identity,
            referencePfsConfig=pfsConfigs[referenceVisit],
            targetPfsConfig=pfsConfigs[targetVisit],
            useLineList=useLineList,
            kernel=kernel,
            targetDetectorMap=targetDetectorMap,
        )
        butler.put(outputs.detectorMap, outputRefs.outputDetectorMap)
        butler.put(outputs.fiberProfiles, outputRefs.outputFiberProfiles)
        if self.config.kernelArm is not None:
            return  # No kernel output
        kernelRefs = [ref for ref in outputRefs.outputKernel if ref.dataId["visit"] == targetVisit]
        if len(kernelRefs) != 1:
            raise RuntimeError(f"Expected one kernel output for visit {targetVisit}, found {len(kernelRefs)}")
        butler.put(outputs.kernel, kernelRefs[0])

    def run(
        self,
        reference: Exposure,
        target: Exposure,
        psf: Psf,
        detectorMap: DetectorMap,
        fiberProfiles: FiberProfileSet,
        identity: CalibIdentity,
        referencePfsConfig: Optional[PfsConfig] = None,
        targetPfsConfig: Optional[PfsConfig] = None,
        useLineList: bool = False,
        kernel: Optional[afwMath.Kernel] = None,
        targetDetectorMap: Optional[DetectorMap] = None,
    ) -> Struct:
        """Adjust the detectorMap and fiberProfiles for the target

        Parameters
        ----------
        reference : `lsst.afw.image.Exposure`
            Reference exposure, with the same focus as ``psf``,
            ``detectorMap`` and ``fiberProfiles``.
        target : `lsst.afw.image.Exposure`
            Target exposure, for which to adjust the calibs.
        psf : `lsst.afw.detection.Psf`
            PSF model for the reference.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y for the reference.
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
            Profiles of fibers for the reference.
        identity : `pfs.datamodel.CalibIdentity`
            Identity of the outputs.
        referencePfsConfig, targetPfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configurations for the reference and target. These are
            used for excluding fibers from the kernel fit and identifying
            breaks in the comb of fibers (see `PsfMatchTask`), and the
            ``referencePfsConfig`` is used for selecting fibers when adjusting
            the detectorMap to the reference.
        useLineList : `bool`, optional
            Use the line list when measuring the PSF-matching kernel? This is
            appropriate for sky exposures.
        kernel : `lsst.afw.math.Kernel`, optional
            PSF-matching kernel from the reference to the target, to use
            instead of measuring one (e.g., measured for another arm). If
            provided, ``useLineList`` is not used. If ``config.doWarp``, this
            must have been measured with warping.
        targetDetectorMap : `pfs.drp.stella.DetectorMap`, optional
            Calib mapping from fiberId,wavelength to x,y valid for the target
            (not adjusted to the target); required if ``config.doWarp``. The
            target is warped onto the reference using this and the (adjusted)
            reference detectorMap before measuring the kernel, and the shift
            of the warp is added to that of the kernel.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y for the target.
        referenceDetectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y for the reference.
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
            Profiles of fibers for the target.
        kernel : `lsst.afw.math.Kernel`
            PSF-matching kernel from the reference to the target.
        psfMatch : `PsfMatchResult` or `None`
            Results from PSF-matching, or `None` if the ``kernel`` was
            provided.
        shifts : `lsst.pipe.base.Struct`
            Peak shifts measured on the grid (see `measurePeakShifts`).
        fit : `lsst.pipe.base.Struct`
            Results of fitting the distortion (see `fitDistortion`).
        targetLines : `pfs.drp.stella.ArcLineSet` or `None`
            Lines and traces measured on the target with the new
            detectorMap (see `checkTarget`), or `None` if not
            ``config.doCheckTarget``.
        """
        if self.config.doWarp and targetDetectorMap is None:
            raise ValueError("targetDetectorMap is required for doWarp")

        referenceDetectorMap = detectorMap
        if self.config.doAdjustReference:
            referenceDetectorMap = self.adjustReferenceDetectorMap(
                reference, detectorMap, fiberProfiles, referencePfsConfig, identity.arm
            )
        warpDetectorMaps = (referenceDetectorMap, targetDetectorMap) if self.config.doWarp else None

        if kernel is not None:
            self.log.info("Using the provided PSF-matching kernel instead of measuring one")
            psfMatch = None
        else:
            refLines = None
            if useLineList:
                refLines = self.readLineList.run(referenceDetectorMap, target.getMetadata(), arm=identity.arm)
            psfMatch = self.psfMatch.run(
                reference,
                target,
                refLines=refLines,
                detectorMap=referenceDetectorMap,
                sourcePfsConfig=referencePfsConfig,
                targetPfsConfig=targetPfsConfig,
                sourceDetectorMap=referenceDetectorMap if self.config.doWarp else None,
                targetDetectorMap=targetDetectorMap if self.config.doWarp else None,
            )
        if psfMatch is not None:
            kernel = psfMatch.psfMatchingKernel

        shifts = self.measurePeakShifts(kernel, psf, referenceDetectorMap, warpDetectorMaps)
        fit = self.fitDistortion(referenceDetectorMap, shifts)

        metadata = referenceDetectorMap.metadata.deepCopy()
        setCalibHeader(metadata, "detectorMap", [identity.visit0], identity.toDict())
        newDetectorMap = addOpticsDistortion(referenceDetectorMap, fit.distortion, target.visitInfo, metadata)
        self.checkDetectorMap(referenceDetectorMap, newDetectorMap, shifts, fit)

        # The fiberProfiles are relative to the input detectorMap, so the shift is relative to that
        newFiberProfiles = self.adjustFiberProfiles(
            fiberProfiles,
            kernel,
            detectorMap,
            newDetectorMap,
            identity,
            target.visitInfo,
            warpDetectorMaps,
        )

        targetLines = None
        if self.config.doCheckTarget:
            targetLines = self.checkTarget(
                target, newDetectorMap, newFiberProfiles, targetPfsConfig, identity.arm
            )

        return Struct(
            detectorMap=newDetectorMap,
            referenceDetectorMap=referenceDetectorMap,
            fiberProfiles=newFiberProfiles,
            kernel=kernel,
            psfMatch=psfMatch,
            shifts=shifts,
            fit=fit,
            targetLines=targetLines,
        )

    def adjustReferenceDetectorMap(
        self,
        reference: Exposure,
        detectorMap: DetectorMap,
        fiberProfiles: FiberProfileSet,
        pfsConfig: Optional[PfsConfig],
        arm: str,
    ) -> DetectorMap:
        """Adjust the detectorMap to the reference exposure

        We measure lines (and, optionally, traces) on the reference, and fit
        a low-order distortion of the detectorMap to them, so that drift
        between the input detectorMap and the reference is not inherited by
        the output.

        Parameters
        ----------
        reference : `lsst.afw.image.Exposure`
            Reference exposure.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y, approximately correct for
            the reference.
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
            Profiles of fibers, for continuum subtraction when centroiding
            lines.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration for the reference, for selecting fibers.
        arm : `str`
            Spectrograph arm in use (``b``, ``r``, ``n``, ``m``).

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y for the reference.
        """
        fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap)
        refLines = self.readLineList.run(detectorMap, reference.getMetadata(), arm=arm)
        seed = reference.visitInfo.id if reference.visitInfo is not None else 0
        lines = ArcLineSet.empty()
        if len(refLines) > 0:
            lines = self.centroidLines.run(
                reference, refLines, detectorMap, pfsConfig, fiberTraces, seed=seed
            )
        if self.config.doForceTraces or not lines:
            lines.extend(self.centroidTraces.run(reference, detectorMap, pfsConfig))
        return self.adjustReference.run(detectorMap, lines, arm, reference.visitInfo, seed=seed).detectorMap

    def measurePeakShifts(
        self,
        kernel: afwMath.Kernel,
        psf: Psf,
        detectorMap: DetectorMap,
        warpDetectorMaps: Optional[tuple[DetectorMap, DetectorMap]] = None,
    ) -> Struct:
        """Measure the shifts in peak position on a grid

        Parameters
        ----------
        kernel : `lsst.afw.math.Kernel`
            PSF-matching kernel from the reference to the target.
        psf : `lsst.afw.detection.Psf`
            PSF model for the reference.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y for the reference.
        warpDetectorMaps : `tuple` of `pfs.drp.stella.DetectorMap`, optional
            Mappings from fiberId,wavelength to x,y for the reference and
            target exposures, if the target was warped onto the reference
            when measuring the kernel. The shift of the warp is added to the
            shift of the kernel.

        Returns
        -------
        fiberId : `numpy.ndarray` of `int`
            Fiber identifiers for grid points.
        row : `numpy.ndarray` of `float`
            Rows for grid points.
        xx, yy : `numpy.ndarray` of `float`
            Positions of grid points.
        dx, dy : `numpy.ndarray` of `float`
            Shift in peak position from the reference to the target.
        dxCentroid, dyCentroid : `numpy.ndarray` of `float`
            Shift in centroid position from the reference to the target.
        xBiasRef, yBiasRef : `numpy.ndarray` of `float`
            Peak minus centroid for the reference.
        xBiasTarget, yBiasTarget : `numpy.ndarray` of `float`
            Peak minus centroid for the target.
        kernelSum : `numpy.ndarray` of `float`
            Sum of the kernel at each position.
        dxWarp, dyWarp : `numpy.ndarray` of `float`
            Shift of the warp (included in ``dx``, ``dy``, ``dxCentroid``
            and ``dyCentroid``); zero if not warped.
        """
        allFiberId = detectorMap.fiberId
        numFibers = min(self.config.numFibers, len(allFiberId))
        fiberIndices = np.unique(np.linspace(0, len(allFiberId) - 1, numFibers).round().astype(int))
        bbox = detectorMap.bbox
        rows = np.linspace(
            bbox.getMinY() + self.config.gridMargin,
            bbox.getMaxY() - self.config.gridMargin,
            self.config.numRows,
        )
        fiberId, row = np.meshgrid(allFiberId[fiberIndices], rows, indexing="ij")
        fiberId = fiberId.flatten()
        row = row.flatten()
        xx = detectorMap.getXCenter(fiberId, row)
        select = np.isfinite(xx) & (xx >= bbox.getMinX()) & (xx <= bbox.getMaxX())
        fiberId = fiberId[select]
        row = row[select]
        xx = xx[select]
        yy = row.copy()

        # Measure the peak as CentroidLinesTask does, for consistency with the line measurements
        centroidConfig = self.config.centroidLines
        shifts = calculatePeakShifts(
            kernel,
            psf,
            xx,
            yy,
            footprintHeight=centroidConfig.footprintHeight,
            footprintWidth=centroidConfig.footprintWidth,
            centroidControl=centroidConfig.centroider.makeControl(),
            centroidPsf=makeGaussianPsf(centroidConfig.fwhm, centroidConfig.kernelSize),
            peakSearch=centroidConfig.peakSearch,
        )
        shifts.fiberId = fiberId
        shifts.row = row
        shifts.xx = xx
        shifts.yy = yy

        shifts.dxWarp = np.zeros_like(xx)
        shifts.dyWarp = np.zeros_like(xx)
        if warpDetectorMaps is not None:
            shifts.dxWarp, shifts.dyWarp = calculateWarpShifts(*warpDetectorMaps, fiberId, row)
            self.log.info(
                "Kernel peak shifts: dx = %.3f +/- %.3f, dy = %.3f +/- %.3f; adding warp shifts: "
                "dx = %.3f +/- %.3f, dy = %.3f +/- %.3f",
                np.nanmedian(shifts.dx),
                robustRms(shifts.dx[np.isfinite(shifts.dx)]),
                np.nanmedian(shifts.dy),
                robustRms(shifts.dy[np.isfinite(shifts.dy)]),
                np.nanmedian(shifts.dxWarp),
                robustRms(shifts.dxWarp[np.isfinite(shifts.dxWarp)]),
                np.nanmedian(shifts.dyWarp),
                robustRms(shifts.dyWarp[np.isfinite(shifts.dyWarp)]),
            )
            shifts.dx = shifts.dx + shifts.dxWarp
            shifts.dy = shifts.dy + shifts.dyWarp
            shifts.dxCentroid = shifts.dxCentroid + shifts.dxWarp
            shifts.dyCentroid = shifts.dyCentroid + shifts.dyWarp

        good = np.isfinite(shifts.dx) & np.isfinite(shifts.dy)
        self.log.info(
            "Measured peak shifts at %d/%d grid points: dx = %.3f +/- %.3f, dy = %.3f +/- %.3f "
            "(centroid: dx = %.3f +/- %.3f, dy = %.3f +/- %.3f)",
            good.sum(),
            len(good),
            np.median(shifts.dx[good]),
            robustRms(shifts.dx[good]),
            np.median(shifts.dy[good]),
            robustRms(shifts.dy[good]),
            np.median(shifts.dxCentroid[good]),
            robustRms(shifts.dxCentroid[good]),
            np.median(shifts.dyCentroid[good]),
            robustRms(shifts.dyCentroid[good]),
        )
        self.metadata["numGrid"] = int(good.sum())
        self.metadata["kernelSum"] = float(np.median(shifts.kernelSum[good]))
        self.metadata["dxMedian"] = float(np.median(shifts.dx[good]))
        self.metadata["dyMedian"] = float(np.median(shifts.dy[good]))
        return shifts

    def fitDistortion(self, detectorMap: DetectorMap, shifts: Struct) -> Struct:
        """Fit a polynomial distortion to the peak shifts

        The distortion is fit in detector coordinates (i.e., before the
        mapping to pixels that accounts for the chip gap in the optical arms),
        since it will be added to the optics model.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y for the reference.
        shifts : `lsst.pipe.base.Struct`
            Peak shifts measured on the grid (see `measurePeakShifts`).

        Returns
        -------
        distortion : `pfs.drp.stella.PolynomialDistortion`
            Fitted distortion.
        xDetector, yDetector : `numpy.ndarray` of `float`
            Grid positions in detector coordinates.
        xResid, yResid : `numpy.ndarray` of `float`
            Residuals of the shifts from the fit.
        select : `numpy.ndarray` of `bool`
            Points used in the final fit.
        """
        detectorMap = OpticalModelDetectorMap.fromDetectorMap(detectorMap)
        detector = detectorMap.detectorModel.pixelsToDetector(shifts.xx, shifts.yy)
        xDetector = detector[:, 0]
        yDetector = detector[:, 1]

        dx = shifts.dx
        dy = shifts.dy
        good = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(xDetector) & np.isfinite(yDetector)
        select = good.copy()
        numPoints = len(dx)
        errors = np.ones(numPoints, dtype=float)
        isLine = np.ones(numPoints, dtype=bool)
        bbox = Box2D(detectorMap.bbox)
        for ii in range(self.config.iterations + 1):
            distortion = PolynomialDistortion.fit(
                self.config.distortionOrder,
                bbox,
                xDetector[select],
                yDetector[select],
                dx[select],
                dy[select],
                errors[select],
                errors[select],
                isLine[select],
            )
            model = np.full((numPoints, 2), np.nan, dtype=float)
            model[good] = distortion(xDetector[good], yDetector[good])
            xResid = dx - model[:, 0]
            yResid = dy - model[:, 1]
            xRms = robustRms(xResid[select])
            yRms = robustRms(yResid[select])
            self.log.debug(
                "Distortion fit iteration %d: %d points, xRms = %.4f, yRms = %.4f",
                ii,
                select.sum(),
                xRms,
                yRms,
            )
            if ii == self.config.iterations:
                break
            with np.errstate(invalid="ignore"):
                newSelect = (
                    good
                    & (np.abs(xResid) < self.config.rejection * xRms)
                    & (np.abs(yResid) < self.config.rejection * yRms)
                )
            if np.all(newSelect == select):
                break
            select = newSelect

        self.log.info(
            "Fit order %d distortion to peak shifts with %d/%d points: xRms = %.4f, yRms = %.4f",
            self.config.distortionOrder,
            select.sum(),
            numPoints,
            xRms,
            yRms,
        )
        self.metadata["fitNumPoints"] = int(select.sum())
        self.metadata["fitXRms"] = float(xRms)
        self.metadata["fitYRms"] = float(yRms)
        return Struct(
            distortion=distortion,
            xDetector=xDetector,
            yDetector=yDetector,
            xResid=xResid,
            yResid=yResid,
            select=select,
        )

    def checkDetectorMap(
        self, oldDetectorMap: DetectorMap, newDetectorMap: DetectorMap, shifts: Struct, fit: Struct
    ) -> None:
        """Check that the new detectorMap reproduces the peak shifts

        We compare the change in position of points of fixed fiberId and
        wavelength with the measured peak shifts, and log the result. This
        checks the fit of the distortion and its application to the
        detectorMap, not the shifts themselves (see `checkTarget` for a
        comparison with the data).

        Parameters
        ----------
        oldDetectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y for the reference.
        newDetectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y for the target.
        shifts : `lsst.pipe.base.Struct`
            Peak shifts measured on the grid (see `measurePeakShifts`).
        fit : `lsst.pipe.base.Struct`
            Results of fitting the distortion (see `fitDistortion`).
        """
        select = fit.select
        fiberId = shifts.fiberId[select]
        wavelength = oldDetectorMap.findWavelength(fiberId, shifts.row[select]).astype(float)
        oldPoints = oldDetectorMap.findPoint(fiberId, wavelength)
        newPoints = newDetectorMap.findPoint(fiberId, wavelength)
        xDiff = newPoints[:, 0] - oldPoints[:, 0] - shifts.dx[select]
        yDiff = newPoints[:, 1] - oldPoints[:, 1] - shifts.dy[select]
        good = np.isfinite(xDiff) & np.isfinite(yDiff)
        xRms = robustRms(xDiff[good])
        yRms = robustRms(yDiff[good])
        self.log.info(
            "New detectorMap reproduces peak shifts to: x = %.4f +/- %.4f, y = %.4f +/- %.4f",
            np.median(xDiff[good]),
            xRms,
            np.median(yDiff[good]),
            yRms,
        )
        self.metadata["checkXRms"] = float(xRms)
        self.metadata["checkYRms"] = float(yRms)

    def adjustFiberProfiles(
        self,
        fiberProfiles: FiberProfileSet,
        kernel: afwMath.Kernel,
        oldDetectorMap: DetectorMap,
        newDetectorMap: DetectorMap,
        identity: CalibIdentity,
        visitInfo: VisitInfo,
        warpDetectorMaps: Optional[tuple[DetectorMap, DetectorMap]] = None,
    ) -> FiberProfileSet:
        """Convolve the fiber profiles with the kernel, and shift them

        Parameters
        ----------
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
            Profiles of fibers for the reference.
        kernel : `lsst.afw.math.Kernel`
            PSF-matching kernel from the reference to the target.
        oldDetectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y for the reference.
        newDetectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y for the target.
        identity : `pfs.datamodel.CalibIdentity`
            Identity of the output.
        visitInfo : `lsst.afw.image.VisitInfo`
            Visit information for the target.
        warpDetectorMaps : `tuple` of `pfs.drp.stella.DetectorMap`, optional
            Mappings from fiberId,wavelength to x,y for the reference and
            target exposures, if the target was warped onto the reference
            when measuring the kernel. The kernel doesn't include the shift of
            the warp, so the shift of the trace between these is removed from
            the shift of the trace when shifting the convolved profiles.

        Returns
        -------
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
            Profiles of fibers for the target.
        """
        metadata = fiberProfiles.metadata.deepCopy()
        setCalibHeader(metadata, "fiberProfiles", [identity.visit0], identity.toDict())
        result = FiberProfileSet.makeEmpty(identity, visitInfo, metadata)
        missing = []
        haveFiberId = set(oldDetectorMap.fiberId)
        for fiberId in fiberProfiles:
            profile = fiberProfiles[fiberId]
            if fiberId not in haveFiberId:
                missing.append(fiberId)
                result[fiberId] = profile.copy()
                continue
            rows = np.asarray(profile.rows, dtype=float)
            xOld = oldDetectorMap.getXCenter(fiberId, rows)
            xNew = newDetectorMap.getXCenter(fiberId, rows)
            if warpDetectorMaps is not None:
                # The convolved profile is displaced by the kernel's shift only, not the warp's. The
                # profile is at a fixed row, so we need the shift of the trace at fixed row (not at fixed
                # wavelength, as for the peak shifts).
                warpReference, warpTarget = warpDetectorMaps
                xNew = xNew - (warpTarget.getXCenter(fiberId, rows) - warpReference.getXCenter(fiberId, rows))
            result[fiberId] = convolveFiberProfile(
                profile, kernel, xOld, xNew, self.config.profileInterpOrder
            )
        if missing:
            self.log.warning("Fibers not in detectorMap; profiles not adjusted: %s", missing)
        return result

    def checkTarget(
        self,
        target: Exposure,
        detectorMap: DetectorMap,
        fiberProfiles: FiberProfileSet,
        pfsConfig: Optional[PfsConfig],
        arm: str,
    ) -> ArcLineSet:
        """Compare lines and traces measured on the target with the new
        detectorMap

        This is an independent check of the outputs against the data, as when
        fitting a detectorMap. We log the median and robust rms of the
        residuals (measured minus model) of the lines (in x and y) and traces
        (in x).

        Parameters
        ----------
        target : `lsst.afw.image.Exposure`
            Target exposure.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y for the target.
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
            Profiles of fibers for the target.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration for the target, for selecting fibers.
        arm : `str`
            Spectrograph arm in use (``b``, ``r``, ``n``, ``m``).

        Returns
        -------
        lines : `pfs.drp.stella.ArcLineSet`
            Lines and traces measured on the target.
        """
        fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap)
        refLines = self.readLineList.run(detectorMap, target.getMetadata(), arm=arm)
        seed = target.visitInfo.id if target.visitInfo is not None else 0
        lines = ArcLineSet.empty()
        if len(refLines) > 0:
            lines = self.centroidLines.run(target, refLines, detectorMap, pfsConfig, fiberTraces, seed=seed)
        lines.extend(self.centroidTraces.run(target, detectorMap, pfsConfig))

        isTrace = lines.description == "Trace"
        numLinesMeasured = int((~isTrace).sum())
        good = ~lines.flag & (lines.status == ReferenceLineStatus.GOOD)
        good &= np.isfinite(lines.x) & np.isfinite(lines.y)
        isLine = good & ~isTrace
        isTrace &= good

        model = detectorMap.findPoint(lines.fiberId[isLine], lines.wavelength[isLine].astype(float))
        xLine = lines.x[isLine] - model[:, 0]
        yLine = lines.y[isLine] - model[:, 1]
        xTrace = lines.x[isTrace] - detectorMap.getXCenter(lines.fiberId[isTrace], lines.y[isTrace])

        def stats(values: np.ndarray) -> tuple[int, float, float]:
            values = values[np.isfinite(values)]
            if len(values) == 0:
                return 0, np.nan, np.nan
            return len(values), float(np.median(values)), float(robustRms(values))

        numLines, xLineMedian, xLineRms = stats(xLine)
        _, yLineMedian, yLineRms = stats(yLine)
        numTrace, xTraceMedian, xTraceRms = stats(xTrace)
        self.log.info(
            "Target measurements about new detectorMap: %d/%d lines: x = %.4f +/- %.4f, y = %.4f +/- %.4f; "
            "%d traces: x = %.4f +/- %.4f",
            numLines,
            numLinesMeasured,
            xLineMedian,
            xLineRms,
            yLineMedian,
            yLineRms,
            numTrace,
            xTraceMedian,
            xTraceRms,
        )
        self.metadata["targetNumLines"] = numLines
        self.metadata["targetLineXMedian"] = xLineMedian
        self.metadata["targetLineXRms"] = xLineRms
        self.metadata["targetLineYMedian"] = yLineMedian
        self.metadata["targetLineYRms"] = yLineRms
        self.metadata["targetNumTrace"] = numTrace
        self.metadata["targetTraceXMedian"] = xTraceMedian
        self.metadata["targetTraceXRms"] = xTraceRms
        return lines
