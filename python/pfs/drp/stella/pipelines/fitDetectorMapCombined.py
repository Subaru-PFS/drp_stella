from typing import Iterable, Optional

import numpy as np

from lsst.geom import Box2I
from lsst.afw.image import VisitInfo
from lsst.daf.base import PropertyList
from lsst.pex.config import ConfigurableField, Field
from lsst.pipe.base import PipelineTask
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from ..adjustDetectorMap import AdjustDetectorMapTask
from ..arcLine import ArcLineSet
from ..fitDetectorMap import FittingError
from .fitDetectorMap import FitDetectorMapConfig, FitDetectorMapConnections, gatherFitDetectorMapInputs

__all__ = ("FitDetectorMapCombinedTask",)


class FitDetectorMapCombinedConnections(FitDetectorMapConnections):
    """Connections for FitDetectorMapCombinedTask"""


class FitDetectorMapCombinedConfig(
    FitDetectorMapConfig, pipelineConnections=FitDetectorMapCombinedConnections
):
    """Configuration for FitDetectorMapCombinedTask"""

    adjustDetectorMap = ConfigurableField(
        target=AdjustDetectorMapTask, doc="Adjust detectorMap to twilight lines"
    )
    maxIterations = Field(
        dtype=int, default=5, doc="Maximum number of arc+twilight fit / twilight adjustment iterations"
    )
    convergenceTolerance = Field(
        dtype=float,
        default=0.01,
        doc="Convergence tolerance for twilight adjustment RMS position shift (pixels)",
    )
    requireConvergence = Field(
        dtype=bool, default=False, doc="Require the twilight adjustment iteration to converge?"
    )


class FitDetectorMapCombinedTask(PipelineTask):
    """Fit a detectorMap from arc and twilight lines simultaneously

    Arc and twilight exposures are taken at different times, so the slit and
    optical state can differ slightly between them. We fit a detectorMap from
    the combined arc+twilight line set and iteratively correct for this
    difference: fit the combined detectorMap, fit a low-order correction to
    just the twilight lines relative to that detectorMap, apply the
    correction to the twilight line positions, and repeat until the
    correction becomes negligible.
    """

    ConfigClass = FitDetectorMapCombinedConfig
    _DefaultName = "fitDetectorMapCombined"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fitDetectorMap")
        self.makeSubtask("adjustDetectorMap")

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
        dataId, arcLines, visitInfo, metadata, bbox, slitOffsets = gatherFitDetectorMapInputs(
            butler, inputRefs, self.config.fitDetectorMap.doSlitOffsets
        )
        outputs = self.run(dataId, arcLines, visitInfo, metadata, bbox, slitOffsets)
        butler.put(outputs, outputRefs)

    def run(
        self,
        dataId: dict,
        arcLines: Iterable[ArcLineSet],
        visitInfo: VisitInfo,
        metadata: PropertyList,
        bbox: Box2I,
        slitOffsets: Optional[np.ndarray] = None,
    ):
        """Fit a detectorMap from arc and twilight lines simultaneously

        Parameters
        ----------
        dataId : `dict`
            Keyword-value pairs that identify the data, containing at least
            ``"arm"`` and ``"spectrograph"`` keys.
        arcLines : iterable of `ArcLineSet`
            List of centroid measurements from different exposures. Rows with
            ``description == "solar"`` are twilight (solar-absorption)
            measurements; all others are treated as arc/trace measurements.
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
            Mapping of fiberId,wavelength to x,y, fit from the combined
            arc+twilight line set.
        lines : `ArcLineSet`
            Combined line measurements with status updated to reflect the
            final detectorMap fit.
        (other fields as returned by the low-level `FitDetectorMapTask.run`)

        Raises
        ------
        pfs.drp.stella.fitDetectorMap.FittingError
            If ``requireConvergence`` is set and the twilight adjustment
            iteration does not converge within ``maxIterations``.
        """
        lines = sum(arcLines, ArcLineSet.empty())
        isTwilight = lines.description == "solar"
        twilight = lines[isTwilight].copy()
        rest = lines[~isTwilight].copy()

        result = None
        combined = rest
        for iteration in range(self.config.maxIterations):
            combined = rest.copy()
            combined.extend(twilight)
            result = self.fitDetectorMap.run(
                dataId,
                bbox,
                combined,
                visitInfo,
                metadata,
                slitOffsets[0] if slitOffsets is not None else None,
                slitOffsets[1] if slitOffsets is not None else None,
            )
            if not len(twilight):
                break  # arc-only: nothing to adjust/iterate

            try:
                adjusted = self.adjustDetectorMap.run(
                    result.detectorMap,
                    twilight,
                    dataId["arm"],
                    visitInfo,
                    metadata,
                    seed=visitInfo.id,
                ).detectorMap
            except FittingError as exc:
                self.log.warn("Twilight adjustment failed on iteration %d: %s", iteration, exc)
                break

            preX, preY = result.detectorMap.findPoint(twilight.fiberId, twilight.wavelength).T
            postX, postY = adjusted.findPoint(twilight.fiberId, twilight.wavelength).T
            dx, dy = postX - preX, postY - preY
            shiftRms = np.sqrt(np.mean(dx**2 + dy**2))
            self.log.info("Iteration %d: twilight adjustment RMS shift = %f pixels", iteration, shiftRms)
            if shiftRms < self.config.convergenceTolerance:
                break

            twilight.x[:] += dx
            twilight.y[:] += dy
        else:
            if self.config.requireConvergence:
                raise FittingError(f"Did not converge within {self.config.maxIterations} iterations")

        result.lines = combined
        return result
