from collections import defaultdict
from lsst.pex.config import Config, Field, ConfigurableField
from lsst.pipe.base import CmdLineTask, ArgumentParser, Struct

from pfs.datamodel.pfsConfig import TargetType
from pfs.datamodel import MaskHelper

from .datamodel import PfsSingle
from .measureFluxCalibration import MeasureFluxCalibrationTask
from .subtractSky1d import SubtractSky1dTask
from .FluxTableTask import FluxTableTask
from .utils import getPfsVersions


class FluxCalibrateConfig(Config):
    """Configuration for FluxCalibrateTask"""
    measureFluxCalibration = ConfigurableField(target=MeasureFluxCalibrationTask, doc="Measure flux calibn")
    subtractSky1d = ConfigurableField(target=SubtractSky1dTask, doc="1D sky subtraction")
    fluxTable = ConfigurableField(target=FluxTableTask, doc="Flux table")
    doWrite = Field(dtype=bool, default=True, doc="Write outputs?")


class FluxCalibrateTask(CmdLineTask):
    """Measure and apply the flux calibration"""
    ConfigClass = FluxCalibrateConfig
    _DefaultName = "fluxCalibrate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("measureFluxCalibration")
        self.makeSubtask("subtractSky1d")
        self.makeSubtask("fluxTable")

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="pfsMerged", level="Visit",
                               help="data IDs, e.g. --id exp=12345")
        return parser

    def runDataRef(self, dataRef):
        """Measure and apply the flux calibration

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for merged spectrum.

        Returns
        -------
        calib : `pfs.drp.stella.FocalPlaneFunction`
            Flux calibration.
        spectra : `list` of `pfs.datamodel.PfsSingle`
            Calibrated spectra for each fiber.
        """
        merged = dataRef.get("pfsMerged")
        pfsConfig = dataRef.get("pfsConfig")
        butler = dataRef.getButler()

        references = self.readReferences(butler, pfsConfig)
        calib = self.measureFluxCalibration.run(merged, references, pfsConfig)
        self.measureFluxCalibration.applySpectra(merged, pfsConfig, calib)
        spectra = [merged.extractFiber(PfsSingle, pfsConfig, fiberId) for fiberId in merged.fiberId]

        armRefList = list(butler.subset("raw", dataId=dataRef.dataId))
        armList = [ref.get("pfsArm") for ref in armRefList]
        sky1d = dataRef.get("sky1d")
        fiberToArm = defaultdict(list)
        for ii, arm in enumerate(armList):
            lsf = None
            self.subtractSky1d.subtractSkySpectra(arm, lsf, pfsConfig, sky1d)
            self.measureFluxCalibration.applySpectra(arm, pfsConfig, calib)
            for ff in arm.fiberId:
                fiberToArm[ff].append(ii)

        # Add the fluxTable
        for ss, ff in zip(spectra, merged.fiberId):
            ss.fluxTable = self.fluxTable.run([ref.dataId for ref in armRefList],
                                              [armList[ii].extractFiber(PfsSingle, pfsConfig, ff) for
                                               ii in fiberToArm[ff]],
                                              MaskHelper.fromMerge([armList[ii].flags]))
            ss.metadata = getPfsVersions()

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
