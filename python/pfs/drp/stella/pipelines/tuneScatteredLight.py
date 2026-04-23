"""PipelineTask wrapper around ``utils.scatteredLightTuning.tuneScatteredLight``.

Dimensions ``(instrument, arm, spectrograph)`` — one quantum per camera, so
``pipetask run --processes 8 -d "arm IN ('b','r') AND spectrograph IN (1,2,3,4)"``
fans out to all 8 visible cameras in parallel.

Inputs:
  postISRCCD     multiple, per visit
  detectorMap    calibration
  fiberProfiles  calibration

Output:
  scatteredLightTune  StructuredDataDict (serialisable plain-Python dict)

Validation is deliberately left out of this task; run ``diagnoseScatteredLight``
from a notebook on a held-out visit set to inspect.
"""

from __future__ import annotations

from typing import List

import numpy as np

from lsst.pex.config import Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base import QuantumContext, Struct
from lsst.pipe.base.connections import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
)
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

from ..scatteredLight import ScatteredLightConfig
from ..utils.scatteredLightTuning import (
    DEFAULT_BIN_EDGES,
    tuneScatteredLight,
)

__all__ = (
    "TuneScatteredLightConnections",
    "TuneScatteredLightConfig",
    "TuneScatteredLightTask",
)


class TuneScatteredLightConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "arm", "spectrograph"),
):
    """Connections for ``TuneScatteredLightTask``."""

    postISRCCD = InputConnection(
        name="postISRCCD",
        doc="Post-ISR exposures (pre-scatter-correction) used for tuning.",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    detectorMap = PrerequisiteConnection(
        name="detectorMap_calib",
        doc="Detector map (used for fiber x-centers).",
        storageClass="DetectorMap",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    fiberProfiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc="Fiber profiles (used to identify illuminated fibers).",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    tuneResult = OutputConnection(
        name="scatteredLightTune",
        doc="Tuned scattered-light parameters and grid diagnostics.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "arm", "spectrograph"),
    )


class TuneScatteredLightConfig(
    PipelineTaskConfig, pipelineConnections=TuneScatteredLightConnections,
):
    """Configuration for ``TuneScatteredLightTask``.

    Leave any ``*Grid`` field empty (default) to keep that parameter at the
    ``ScatteredLightConfig`` value for this camera; set a non-empty list to
    scan it.
    """

    frac1Grid = ListField(
        dtype=float,
        default=list(np.arange(0.010, 0.101, 0.005).round(4)),
        doc="Grid of frac1 values to scan (empty = keep fixed at config).",
    )
    frac2Grid = ListField(
        dtype=float,
        default=list(np.arange(0.005, 0.101, 0.005).round(4)),
        doc="Grid of frac2 values to scan (empty = keep fixed at config).",
    )
    powerLaw1Grid = ListField(
        dtype=float, default=[], optional=True,
        doc="Grid of powerLaw1 values (empty = keep fixed at config).",
    )
    powerLaw2Grid = ListField(
        dtype=float, default=[], optional=True,
        doc="Grid of powerLaw2 values (empty = keep fixed at config).",
    )
    soften1Grid = ListField(
        dtype=float, default=[], optional=True,
        doc="Grid of soften1 values (empty = keep fixed at config).",
    )
    soften2Grid = ListField(
        dtype=float, default=[], optional=True,
        doc="Grid of soften2 values (empty = keep fixed at config).",
    )
    halfIllum = Field(
        dtype=int, default=5,
        doc="Half-width of the illuminated-pixel mask around each fiber (columns).",
    )
    brightPercentile = Field(
        dtype=float, default=60.0,
        doc="Per-frame: rows whose mean on-fiber flux exceeds this percentile "
            "of positive per-row values are used in the metric.",
    )
    binEdges = ListField(
        dtype=int,
        default=list(DEFAULT_BIN_EDGES),
        doc="Distance bin edges (px from nearest illuminated fiber).",
    )


class TuneScatteredLightTask(PipelineTask):
    """Tune the ``ScatteredLightModel`` ``frac1``/``frac2`` (and optionally
    kernel-shape parameters) against a set of post-ISR frames for one camera.

    Quantum dimensions are ``(instrument, arm, spectrograph)``. One quantum
    per camera, so the pipeline framework parallelises across cameras by
    itself.

    Delegates the actual fitting to
    :func:`pfs.drp.stella.utils.scatteredLightTuning.tuneScatteredLight`.
    """

    ConfigClass = TuneScatteredLightConfig
    _DefaultName = "tuneScatteredLight"

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        arm = outputRefs.tuneResult.dataId["arm"]
        spectrograph = outputRefs.tuneResult.dataId["spectrograph"]
        camera = f"{arm}{spectrograph}"

        postISRCCDs = [butler.get(ref) for ref in inputRefs.postISRCCD]
        detectorMap = butler.get(inputRefs.detectorMap)
        fiberProfiles = butler.get(inputRefs.fiberProfiles)
        visits = [int(ref.dataId["visit"]) for ref in inputRefs.postISRCCD]

        result = self.run(
            postISRCCDs=postISRCCDs,
            detectorMap=detectorMap,
            fiberProfiles=fiberProfiles,
            camera=camera,
            visits=visits,
        )
        butler.put(result.tuneResult, outputRefs.tuneResult)

    def run(
        self,
        postISRCCDs,
        detectorMap,
        fiberProfiles,
        camera: str,
        visits: List[int] | None = None,
    ) -> Struct:
        """Tune scattered-light parameters for one camera.

        Parameters
        ----------
        postISRCCDs : sequence of `lsst.afw.image.Exposure`
        detectorMap : `pfs.drp.stella.DetectorMap`
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
        camera : `str`
            e.g. ``"b1"``. Used to look up current defaults from
            ``ScatteredLightConfig`` and for logging.
        visits : list[int], optional
            Visits the frames correspond to. Recorded in the output for
            provenance. If not given, frames are labelled by index.

        Returns
        -------
        struct : `Struct`
            ``tuneResult`` : plain-Python dict (butler ``StructuredDataDict``)
            with keys ``best``, ``best_rms``, ``best_bins``, ``grids``,
            ``rms_grid``, ``bin_grid``, ``binLabels``, ``raw_bins``, ``meta``.
            Numpy arrays are converted to lists so the dict is JSON-safe.
        """
        tuneGrids = {}
        for paramName, field in (
            ("frac1", self.config.frac1Grid),
            ("frac2", self.config.frac2Grid),
            ("powerLaw1", self.config.powerLaw1Grid),
            ("powerLaw2", self.config.powerLaw2Grid),
            ("soften1", self.config.soften1Grid),
            ("soften2", self.config.soften2Grid),
        ):
            vals = list(field) if field else []
            if vals:
                tuneGrids[paramName] = np.asarray(vals, dtype=np.float64)

        if not tuneGrids:
            raise ValueError(
                "At least one of frac1Grid/frac2Grid/... must be non-empty",
            )

        self.log.info(
            "[%s] tuning %s on %d frames",
            camera,
            ",".join(f"{k}({len(v)})" for k, v in tuneGrids.items()),
            len(postISRCCDs),
        )

        result = tuneScatteredLight(
            postISRCCDs=postISRCCDs,
            detectorMap=detectorMap,
            fiberProfiles=fiberProfiles,
            camera=camera,
            scatConfig=ScatteredLightConfig(),
            tuneGrids=tuneGrids,
            halfIllum=self.config.halfIllum,
            brightPercentile=self.config.brightPercentile,
            binEdges=tuple(self.config.binEdges),
        )

        best = result["best"]
        self.log.info(
            "[%s] best frac1=%.4f frac2=%.4f  RMS=%.5f",
            camera, best["frac1"], best["frac2"], result["best_rms"],
        )

        payload = _toJsonSafe(result)
        payload["visits"] = list(visits) if visits else list(range(len(postISRCCDs)))
        payload["camera"] = camera

        return Struct(tuneResult=payload)


def _toJsonSafe(obj):
    """Recursively convert numpy arrays / scalars to plain Python.

    Required for ``StructuredDataDict`` storage.
    """
    if isinstance(obj, dict):
        return {k: _toJsonSafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_toJsonSafe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
