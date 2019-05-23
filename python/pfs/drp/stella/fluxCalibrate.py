from lsst.pex.config import Config, Field, ConfigurableField
from lsst.pipe.base import CmdLineTask, ArgumentParser, Struct

from pfs.datamodel.pfsConfig import TargetType

from .measureFluxCalibration import MeasureFluxCalibrationTask


class FluxCalibrateConfig(Config):
    """Configuration for FluxCalibrateTask"""
    measureFluxCalibration = ConfigurableField(target=MeasureFluxCalibrationTask, doc="Measure flux calibn")
    doWrite = Field(dtype=bool, default=True, doc="Write outputs?")


class FluxCalibrateTask(CmdLineTask):
    """Measure and apply the flux calibration"""
    ConfigClass = FluxCalibrateConfig
    _DefaultName = "fluxCalibrate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("measureFluxCalibration")

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="pfsMerged",
                               help="data IDs, e.g. --id exp=12345")
        return parser

    def run(self, dataRef):
        """Measure and apply the flux calibration

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for merged spectrum.

        Returns
        -------
        calib : `pfs.drp.stella.FocalPlaneFunction`
            Flux calibration.
        spectra : `list` of `pfs.datamodel.PfsSpectrum`
            Calibrated spectra for each fiber.
        """
        merged = dataRef.get("pfsMerged")
        pfsConfig = dataRef.get("pfsConfig")
        butler = dataRef.getButler()
        references = self.readReferences(butler, pfsConfig)
        calib = self.measureFluxCalibration.run(merged, references, pfsConfig)
        spectra = self.measureFluxCalibration.apply(merged, pfsConfig, calib)

        if self.config.doWrite:
            dataRef.put(calib, "fluxCal")
            for spectrum in spectra:
                dataId = spectrum.getIdentity().copy()
                dataId.update(dataRef.dataId)
                butler.put(spectrum, "pfsSingle", dataId)
        return Struct(calib=calib, spectra=spectra)

    def readReferences(self, butler, pfsConfig):
        """Read the physical reference fluxes

        If you get a read error here, it's likely because you haven't got a
        physical reference flux; try running ``calibrateReferenceFlux``.

        Parameters
        ----------
        butler : `lsst.daf.persistence.Butler`
            Data butler.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for identifying flux standards.

        Returns
        -------
        references : `dict` mapping `int` to `pfs.datamodel.PfsSimpleSpectrum`
            Reference spectra, indexed by fiber identifier.
        """
        indices = pfsConfig.selectByTargetType(TargetType.FLUXSTD)
        return {pfsConfig.fiberId[ii]: butler.get("pfsReference", pfsConfig.getIdentityFromIndex(ii)) for
                ii in indices}

    def _getMetadataName(self):
        return None
