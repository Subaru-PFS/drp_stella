from typing import Iterable, Optional

import numpy as np

from lsst.geom import Box2I
from lsst.afw.image import VisitInfo
from lsst.daf.base import PropertyList
from lsst.daf.butler import DataCoordinate
from lsst.pex.config import ConfigurableField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from pfs.drp.stella.gen3 import readDatasetRefs

from ..arcLine import ArcLineSet
from ..fitDistortedDetectorMap import FitDistortedDetectorMapTask
from .lookups import lookupDetectorMap

__all__ = ("FitDetectorMapTask",)


class FitDetectorMapConnections(
    PipelineTaskConnections, dimensions=("instrument", "arm", "spectrograph")
):
    """Connections for FitDetectorMapTask"""

    arcLines = InputConnection(
        name="centroids",
        doc="Emission line measurements",
        storageClass="ArcLineSet",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )

    slitOffsets = PrerequisiteConnection(
        name="detectorMap_calib.slitOffsets",
        doc="Mapping from fiberId,wavelength to x,y: measured from real data",
        storageClass="NumpyArray",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
        lookupFunction=lookupDetectorMap,
    )

    bbox = InputConnection(
        name="postISRCCD.bbox",
        doc="Bounding box for detector",
        storageClass="Box2I",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )

    visitInfo = InputConnection(
        name="postISRCCD.visitInfo",
        doc="Visit information",
        storageClass="VisitInfo",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    metadata = InputConnection(
        name="postISRCCD.metadata",
        doc="Exposure header",
        storageClass="PropertyList",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )

    detectorMap = OutputConnection(
        name="detectorMap_candidate",
        doc="Mapping between fiberId,wavelength and x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config:
            return
        if config.fitDetectorMap.doSlitOffsets:
            self.prerequisiteInputs.remove("slitOffsets")


class FitDetectorMapConfig(PipelineTaskConfig, pipelineConnections=FitDetectorMapConnections):
    """Configuration for FitDetectorMapTask"""

    fitDetectorMap = ConfigurableField(target=FitDistortedDetectorMapTask, doc="Fit detectorMap")


class FitDetectorMapTask(PipelineTask):
    """Fit a detectorMap based on centroids measured on multiple exposures

    This differs from the ``MeasureDetectorMapTask``, which adjusts a
    detectorMap with a low-order correction based on centroids measured on a
    single exposure.

    The contents could be moved into the `FitDistortedDetectorMapTask` at a
    later date.
    """

    ConfigClass = FitDetectorMapConfig
    _DefaultName = "fitDetectorMap"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fitDetectorMap")

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
        # Determine what spectrograph+arm we're dealing with
        arm = inputRefs.arcLines[0].dataId.arm.name
        spectrograph = inputRefs.arcLines[0].dataId.spectrograph.num
        assert arm in "brnm"
        assert spectrograph in (1, 2, 3, 4)
        dataId = dict(arm=arm, spectrograph=spectrograph)

        # Get only the first detectorMap, visitInfo and metadata
        data = readDatasetRefs(butler, inputRefs, "arcLines", "visitInfo", "metadata", "bbox")
        first = min(range(len(data.visitInfo)), key=lambda ii: data.visitInfo[ii].id)

        outputs = self.run(
            dataId,
            data.arcLines,
            data.visitInfo[first],
            data.metadata[first],
            data.bbox[first],
            butler.get(inputRefs.slitOffsets) if not self.config.fitDetectorMap.doSlitOffsets else None,
        )
        butler.put(outputs, outputRefs)

    def run(
        self,
        dataId: DataCoordinate,
        arcLines: Iterable[ArcLineSet],
        visitInfo: VisitInfo,
        metadata: PropertyList,
        bbox: Box2I,
        slitOffsets: Optional[np.ndarray] = None,
    ):
        """Fit a detectorMap based on centroids measured on multiple exposures

        Parameters
        ----------
        dataId : `DataCoordinate`
            Keyword-value pairs that identify the data, containing at least
            ``"arm"`` and ``"spectrograph"`` keys.
        arcLines : iterable of `ArcLineSet`
            List of centroid measurements from different exposures.
        visitInfo : `VisitInfo`
            Visit information to apply to the detectorMap.
        metadata : `PropertyList`
            Metadata (header) to apply to the detectorMap.
        bbox : `Box2I`
            Bounding box for the detector.
        slitOffsets : `numpy.ndarray` of `float`, optional
            Slit offsets to apply to the detectorMap.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        xResid, yResid : `numpy.ndarray` of `float`
            Fit residual in x,y for each of the ``lines`` (pixels).
        xRms, yRms : `float`
            Weighted RMS residual in x,y (pixels).
        xRobustRms, yRobustRms : `float`
            Robust RMS (from IQR) residual in x,y (pixels).
        chi2 : `float`
            Fit chi^2.
        dof : `float`
            Degrees of freedom.
        num : `int`
            Number of points selected.
        numParameters : `int`
            Number of parameters in fit.
        selection : `numpy.ndarray` of `bool`
            Selection used in calculating statistics.
        soften : `tuple` (`float`, `float`), optional
            Systematic error in x and y that was applied to measured errors
            (pixels) in chi^2 calculation.
        xSoften, ySoften : `float`
            Calculated systematic errors required to soften errors to attain
            chi^2/dof = 1.
        reserved : `numpy.ndarray` of `bool`
            Array indicating which lines were reserved from the fit.
        """
        return self.fitDetectorMap.run(
            dataId,
            bbox,
            sum(arcLines, ArcLineSet.empty()),
            visitInfo,
            metadata,
            slitOffsets[0] if slitOffsets is not None else None,
            slitOffsets[1] if slitOffsets is not None else None,
        )
