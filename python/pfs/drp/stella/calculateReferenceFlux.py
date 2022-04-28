import numpy as np

from lsst.pex.config import ConfigurableField
from lsst.pipe.base import ArgumentParser

from lsst.pipe.base import CmdLineTask, PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base.butlerQuantumContext import ButlerQuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from pfs.datamodel.pfsConfig import PfsConfig
from pfs.datamodel.pfsFluxReference import PfsFluxReference
from pfs.datamodel import Identity, MaskHelper
from pfs.datamodel.wavelengthArray import WavelengthArray
from .datamodel import PfsSingle, PfsMerged
from .fitReference import FitReferenceTask
from .selectFibers import SelectFibersTask


class CalculateReferenceFluxConnections(PipelineTaskConnections, dimensions=("instrument", "exposure")):
    """Connections for CalculateReferenceFluxTask"""

    pfsMerged = InputConnection(
        name="pfsMerged",
        doc="Merged spectra from exposure",
        storageClass="PfsMerged",
        dimensions=("instrument", "exposure"),
    )
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "exposure"),
    )

    references = OutputConnection(
        name="pfsFluxReference",
        doc="Fit reference spectrum of flux standards",
        storageClass="PfsFluxReference",
        dimensions=("instrument", "exposure"),
    )


class CalculateReferenceFluxConfig(PipelineTaskConfig, pipelineConnections=CalculateReferenceFluxConnections):
    """Configuration for CalculateReferenceFluxTask"""

    selectFibers = ConfigurableField(target=SelectFibersTask, doc="Select fibers to fit")
    fitReference = ConfigurableField(target=FitReferenceTask, doc="Fit reference spectrum")

    def setDefaults(self):
        super().setDefaults()
        self.selectFibers.targetType = ["FLUXSTD"]


class CalculateReferenceFluxTask(CmdLineTask, PipelineTask):
    """Calculate the physical reference flux for flux standards

    The heavy lifting is done by the ``fitReference`` sub-task.
    """
    ConfigClass = CalculateReferenceFluxConfig
    _DefaultName = "calculateReferenceFlux"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("selectFibers")
        self.makeSubtask("fitReference")

    def run(self, pfsMerged: PfsMerged, pfsConfig: PfsConfig) -> Struct:
        """Calculate reference spectra for flux standards in merged spectra

        Parameters
        ----------
        pfsMerged : `PfsMerged`
            Merged spectra for an exposure.
        pfsConfig : `PfsConfig`
            Top-end fiber configuration.

        Returns
        -------
        references : `dict` [`pfs.datamodel.Target`: `PfsReference`]
            Reference spectra, indexed by target.
        pfsConfig : `PfsConfig`
            Top-end fiber configuration, containing only the selected fibers.
        """
        pfsConfig = self.selectFibers.run(pfsConfig.select(fiberId=pfsMerged.fiberId))
        references = {}
        for fiberId in pfsConfig.fiberId:
            single = pfsMerged.extractFiber(PfsSingle, pfsConfig, fiberId)
            references[single.target] = self.fitReference.run(single)
        return Struct(pfsConfig=pfsConfig, references=references)

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
        results = self.run(**inputs)

        dataId = inputRefs.pfsConfig.dataId.full
        spectra = results.references
        first = spectra[next(iter(results.pfsConfig))]
        wavelength = WavelengthArray(first.wavelength[0], first.wavelength[-1], first.wavelength.size)
        for ss in spectra.values():
            assert np.all(ss.wavelength == wavelength)

        references = PfsFluxReference(
            Identity(visit=dataId["exposure"], pfsDesignId=dataId["pfs_design_id"]),
            results.pfsConfig.fiberId,
            wavelength,
            np.array([spectra[target].flux for target in results.pfsConfig]),
            first.metadata,
            np.zeros(len(results.pfsConfig), dtype=np.int32),
            MaskHelper(),
            np.full(
                shape=(len(spectra),),
                fill_value=np.nan,
                dtype=[
                    ("teff", np.float32),
                    ("logg", np.float32),
                    ("m", np.float32),
                    ("alpha", np.float32),
                    ("radial_velocity", np.float32),
                    ("radial_velocity_err", np.float32),
                ],
            ),
        )
        butler.put(references, outputRefs.references)

    @classmethod
    def _makeArgumentParser(cls):
        """Make ArgumentParser"""
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="pfsMerged",
                               help="data IDs, e.g. --id exp=12345")
        return parser

    def runDataRef(self, dataRef):
        """Run on an exposure

        This is the entry point for the Gen2 middleware.

        We write the reference spectra to individual files. This is in contrast
        to the Gen3 middleware, which writes the reference spectra to a single
        file.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.
        """
        merged = dataRef.get("pfsMerged")
        pfsConfig = dataRef.get("pfsConfig")
        butler = dataRef.getButler()
        spectra = self.run(merged, pfsConfig).references
        for reference in spectra.values():
            butler.put(reference, "pfsReference", reference.getIdentity())

    def _getMetadataName(self):
        return None
