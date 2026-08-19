from collections.abc import Collection, Mapping
import typing

import astropy.time
import numpy as np
from scipy.ndimage import median_filter

from lsst.pipe.base import Struct
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import BaseInput as BaseInputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection, NoWorkFound

from lsst.daf.butler import DataCoordinate, DatasetRef

from pfs.datamodel.h4Persistence import H4PersistenceModel, H4Persistence

from pfs.drp.stella.datamodel.drp import PfsArm

__all__ = ("InferPersistenceConfig", "InferPersistenceTask", "H4PersistenceInputConnectionMixIn")


class InferPersistenceConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "arm", "spectrograph"),
):
    """Connections for InferPersistenceTask"""

    h4PersistenceModel = PrerequisiteConnection(
        name="h4PersistenceModel",
        doc="H4RG detector persistence model",
        storageClass="H4PersistenceModel",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra",
        storageClass="PfsArm",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    h4Persistence = OutputConnection(
        name="h4Persistence",
        doc="H4RG detector persistence",
        storageClass="H4Persistence",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )

    @typing.override
    def adjustQuantum(
        self,
        inputs: dict[str, tuple[BaseInputConnection, Collection[DatasetRef]]],
        outputs: dict[str, tuple[OutputConnection, Collection[DatasetRef]]],
        label: str,
        data_id: DataCoordinate,
    ) -> tuple[
        Mapping[str, tuple[BaseInputConnection, Collection[DatasetRef]]],
        Mapping[str, tuple[OutputConnection, Collection[DatasetRef]]],
    ]:
        """Make adjustments to `lsst.daf.butler.DatasetRef` objects
        in the `lsst.daf.butler.Quantum` during the graph generation stage
        of the activator.

        Parameters
        ----------
        inputs : `dict`
            Dictionary whose keys are an input (regular or prerequisite)
            connection name and whose values are a tuple of the connection
            instance and a collection of associated
            `~lsst.daf.butler.DatasetRef` objects.
        outputs : `~collections.abc.Mapping`
            Mapping of output datasets, with the same structure as ``inputs``.
        label : `str`
            Label for this task in the pipeline (should be used in all
            diagnostic messages).
        data_id : `lsst.daf.butler.DataCoordinate`
            Data ID for this quantum in the pipeline (should be used in all
            diagnostic messages).

        Returns
        -------
        adjusted_inputs : `~collections.abc.Mapping`
            Mapping of the same form as ``inputs`` with updated containers of
            input `~lsst.daf.butler.DatasetRef` objects.
        adjusted_outputs : `~collections.abc.Mapping`
            Mapping of updated output datasets, with the same structure and
            interpretation as ``adjusted_inputs``.
        """
        if data_id["arm"] != "n":
            raise NoWorkFound()
        return super().adjustQuantum(inputs, outputs, label, data_id)


class InferPersistenceConfig(PipelineTaskConfig, pipelineConnections=InferPersistenceConnections):
    """Configuration for InferPersistenceTask"""

    pass


class InferPersistenceTask(PipelineTask):
    """Infer persistent electrons released during each exposure."""

    _DefaultName = "inferPersistence"
    ConfigClass = InferPersistenceConfig

    def run(
        self,
        h4PersistenceModel: H4PersistenceModel,
        pfsArm: PfsArm,
        pfsArmNext: PfsArm,
        qCurrent: np.ndarray[tuple[int, int, int], np.dtype[np.floating]],
    ) -> Struct:
        """Infer persistent electrons released during an exposure.

        Parameters
        ----------
        h4PersistenceModel : `H4PersistenceModel`
            H4RG detector persistence model.
        pfsArm : `PfsArm`
            Extracted spectra.
        pfsArmNext : `PfsArm`
            Next ``pfsArm``. ``pfsArmNext.identity.visit`` must be
            1 plus ``pfsArm.identity.visit`` except for the last visit.
            If ``pfsArm`` is the last visit, ``pfsArmNext`` can be anything.
        qCurrent : `np.ndarray`
            Current state. Shape (``nFibers``, ``nWavelengths``, ``nComponents``)
            where ``nComponents`` is ``h4PersistenceModel.n_components``.

        Returns
        -------
        h4Persistence : `H4Persistence`
            Persistent electrons released during the exposure
        qNext : `np.ndarray`
            Next state. This array must be passed as ``qCurrent`` on the next call.
        """
        expTime = pfsArm.identity.expTime
        # `identity.obsTime` is the average of exposure's start and stop.
        # `obsDate` computed here is the start time.
        obsDate = astropy.time.Time(
            pfsArm.identity.obsTime, format="isot", scale="tai"
        ) - astropy.time.TimeDelta(expTime / 2, format="sec")

        expTimeNext = pfsArmNext.identity.expTime
        obsDateNext = astropy.time.Time(
            pfsArmNext.identity.obsTime, format="isot", scale="tai"
        ) - astropy.time.TimeDelta(expTimeNext / 2, format="sec")

        dtGap = (obsDateNext - obsDate).to_value("sec") - expTime

        # Smooth the observed spectrum in the spectral direction.
        fluxesFiltered = median_filter(pfsArm.flux, size=5, axes=(1,), mode="nearest")

        taus = np.asarray(h4PersistenceModel.taus, dtype=float)
        decayFull = np.exp(-expTime / taus)

        persistenceReleased = qCurrent @ (1.0 - decayFull)

        # Estimate the true incident flux by first removing persistence.
        fluxesCorrected = fluxesFiltered - persistenceReleased

        fluxRate = np.nan_to_num(fluxesCorrected / expTime, nan=0.0, posinf=0.0, neginf=0.0).astype(
            float, copy=False
        )

        qEnd, qNext, persistenceReleased = h4PersistenceModel.step(
            qCurrent,
            fluxRate,
            expTime,
            dt_gap=dtGap,
            new_shape=h4PersistenceModel.spatialProfile.select(pfsArm.fiberId),
        )

        persistence = H4Persistence(
            fiberId=pfsArm.fiberId,
            wavelength=pfsArm.wavelength,
            flux=persistenceReleased,
            identity=pfsArm.identity,
        )
        return Struct(
            h4Persistence=persistence,
            qNext=qNext,
        )

    @typing.override
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
        pfsArmRefs = sorted(
            inputRefs.pfsArm,
            key=lambda ref: ref.dataId["visit"],
        )

        persistenceRefs = {ref.dataId["visit"]: ref for ref in outputRefs.h4Persistence}

        h4PersistenceModel = butler.get(inputRefs.h4PersistenceModel)
        pfsArmNext = butler.get(pfsArmRefs[0])

        qCurrent = np.zeros(
            shape=(*pfsArmNext.flux.shape, h4PersistenceModel.n_components),
            dtype=float,
        )

        for i, pfsArmRef in enumerate(pfsArmRefs):
            visit = pfsArmRef.dataId["visit"]
            pfsArm = pfsArmNext

            if i + 1 < len(pfsArmRefs):
                pfsArmNext = butler.get(pfsArmRefs[i + 1])
            else:
                pfsArmNext = pfsArm

            result = self.run(
                h4PersistenceModel=h4PersistenceModel,
                pfsArm=pfsArm,
                pfsArmNext=pfsArmNext,
                qCurrent=qCurrent,
            )

            butler.put(result.h4Persistence, persistenceRefs[visit])

            qCurrent = result.qNext
            del result


class H4PersistenceInputConnectionMixIn:
    """Mix-in for subclasses of `PipelineConnections`
    that have ``h4Persistence`` as input (not output.)

    This class overrides ``adjustQuantum()`` method
    such that ``h4Persistence`` won't be required by arms other than ``n``.
    This class also drops ``h4Persistence`` entirely
    if ``config.doSubtractPersistence = False``.

    This class expects:
      - that the name of the input connection is ``h4Persistence``, and
      - that the config class has ``doSubtractPersistence`` member.

    This class doesn't define ``h4Persistence`` class member.
    Subclasses must define it.
    """

    def __init__(self, *, config=None) -> None:
        super().__init__(config=config)

        if config is not None and not getattr(config, "doSubtractPersistence", True):
            del self.h4Persistence

    def adjustQuantum(
        self,
        inputs: dict[str, tuple[BaseInputConnection, Collection[DatasetRef]]],
        outputs: dict[str, tuple[OutputConnection, Collection[DatasetRef]]],
        label: str,
        data_id: DataCoordinate,
    ) -> tuple[
        Mapping[str, tuple[BaseInputConnection, Collection[DatasetRef]]],
        Mapping[str, tuple[OutputConnection, Collection[DatasetRef]]],
    ]:
        """Make adjustments to `lsst.daf.butler.DatasetRef` objects
        in the `lsst.daf.butler.Quantum` during the graph generation stage
        of the activator.

        Parameters
        ----------
        inputs : `dict`
            Dictionary whose keys are an input (regular or prerequisite)
            connection name and whose values are a tuple of the connection
            instance and a collection of associated
            `~lsst.daf.butler.DatasetRef` objects.
        outputs : `~collections.abc.Mapping`
            Mapping of output datasets, with the same structure as ``inputs``.
        label : `str`
            Label for this task in the pipeline (should be used in all
            diagnostic messages).
        data_id : `lsst.daf.butler.DataCoordinate`
            Data ID for this quantum in the pipeline (should be used in all
            diagnostic messages).

        Returns
        -------
        adjusted_inputs : `~collections.abc.Mapping`
            Mapping of the same form as ``inputs`` with updated containers of
            input `~lsst.daf.butler.DatasetRef` objects.
        adjusted_outputs : `~collections.abc.Mapping`
            Mapping of updated output datasets, with the same structure and
            interpretation as ``adjusted_inputs``.
        """
        if "h4Persistence" not in inputs:
            return super().adjustQuantum(
                inputs,
                outputs,
                label,
                data_id,
            )

        connection, refs = inputs["h4Persistence"]

        refs = [ref for ref in refs if ref.dataId["arm"] == "n"]

        adjustedInputs = {
            "h4Persistence": (connection, refs),
        }
        inputs.update(adjustedInputs)

        baseAdjustedInputs, baseAdjustedOutputs = super().adjustQuantum(inputs, outputs, label, data_id)

        baseAdjustedInputs.update(adjustedInputs)
        return baseAdjustedInputs, baseAdjustedOutputs
