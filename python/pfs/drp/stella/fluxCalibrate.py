from typing import Union, Dict
from collections import defaultdict

import numpy as np

from lsst.pex.config import Config, Field, ConfigurableField
from lsst.pipe.base import CmdLineTask, ArgumentParser, Struct
from lsst.daf.persistence import Butler

from pfs.datamodel.pfsConfig import PfsConfig, TargetType
from pfs.datamodel import MaskHelper, FiberStatus

from .focalPlaneFunction import FocalPlaneFunction
from .datamodel import PfsArm, PfsSingle, PfsMerged, PfsReference, PfsFiberArray, PfsFiberArraySet
from .fitFocalPlane import FitFocalPlaneTask
from .subtractSky1d import subtractSky1d
from .FluxTableTask import FluxTableTask
from .utils import getPfsVersions


__all__ = ("PfsReferenceSet", "fluxCalibrate", "FluxCalibrateConfig", "FluxCalibrateTask")

PfsReferenceSet = Dict[int, PfsReference]


def fluxCalibrate(spectra: Union[PfsFiberArray, PfsFiberArraySet], pfsConfig: PfsConfig,
                  fluxCal: FocalPlaneFunction) -> None:
    """Apply flux calibration to spectra

    Parameters
    ----------
    spectra : subclass of `PfsFiberArray` or `PfsFiberArraySet`
        Spectra (or spectrum) to flux-calibrate.
    pfsConfig : `PfsConfig`
        Top-end configuration.
    fluxCal : subclass of `FocalPlaneFunction`
        Flux calibration model.
    """
    cal = fluxCal(spectra.wavelength, pfsConfig.select(fiberId=spectra.fiberId))
    with np.errstate(divide="ignore", invalid="ignore"):
        spectra /= spectra.norm
        spectra /= cal.values  # includes spectrum.variance /= cal.values**2
        spectra.variance[:] += cal.variances*spectra.flux**2/cal.values**2
    spectra.norm[:] = 1.0  # We've deliberately changed the normalisation
    bad = np.array(cal.masks) | (np.array(cal.values) == 0.0)
    bad |= ~np.isfinite(cal.values) | ~np.isfinite(cal.variances)
    spectra.mask[bad] |= spectra.flags.add("BAD_FLUXCAL")


def calibratePfsArm(spectra: PfsArm, pfsConfig: PfsConfig, sky1d: FocalPlaneFunction,
                    fluxCal: FocalPlaneFunction, wavelength=None) -> PfsArm:
    """Calibrate a PfsArm

    Parameters
    ----------
    spectra : `PfsArm`
        PfsArm spectra, modified.
    sky1d : `FocalPlaneFunction`
        1d sky model.
    fluxCal : `FocalPlaneFunction`
        Flux calibration model.
    wavelength : `numpy.ndarray` of `float`, optional
        Wavelength array for optional resampling.

    Returns
    -------
    spectra : `PfsArm`
        Calibrated PfsArm spectra.
    """
    pfsConfig = pfsConfig.select(fiberId=spectra.fiberId)
    subtractSky1d(spectra, pfsConfig, sky1d)
    fluxCalibrate(spectra, pfsConfig, fluxCal)
    if wavelength is not None:
        spectra = spectra.resample(wavelength, jacobian=True)  # sampling of pfsArm related to the flux values
    return spectra


class FluxCalibrateConfig(Config):
    """Configuration for FluxCalibrateTask"""
    sysErr = Field(dtype=float, default=1.0e-4,
                   doc=("Fraction of value to add to variance before fitting. This attempts to offset the "
                        "loss of variance as covariance when we resample, the result of which is "
                        "underestimated errors and excess rejection."))
    fitFluxCal = ConfigurableField(target=FitFocalPlaneTask, doc="Fit flux calibration model")
    fluxTable = ConfigurableField(target=FluxTableTask, doc="Flux table")
    doWrite = Field(dtype=bool, default=True, doc="Write outputs?")


class FluxCalibrateTask(CmdLineTask):
    """Measure and apply the flux calibration"""
    ConfigClass = FluxCalibrateConfig
    _DefaultName = "fluxCalibrate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fitFluxCal")
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
        fluxCal : `pfs.drp.stella.FocalPlaneFunction`
            Flux calibration.
        spectra : `list` of `pfs.datamodel.PfsSingle`
            Calibrated spectra for each fiber.
        """
        merged = dataRef.get("pfsMerged")
        mergedLsf = dataRef.get("pfsMergedLsf")
        pfsConfig = dataRef.get("pfsConfig")
        butler = dataRef.getButler()

        mergedFluxCal = merged.select(pfsConfig, targetType=TargetType.FLUXSTD)
        pfsConfigFluxCal = pfsConfig.select(fiberId=mergedFluxCal.fiberId)

        references = self.readReferences(butler, pfsConfigFluxCal)
        self.calculateCalibrations(mergedFluxCal, references)
        fluxCal = self.fitFluxCal.run(mergedFluxCal, pfsConfigFluxCal)
        fluxCalibrate(merged, pfsConfig, fluxCal)

        indices = pfsConfig.selectByFiberStatus(FiberStatus.GOOD, merged.fiberId)
        fiberId = merged.fiberId[indices]
        spectra = [merged.extractFiber(PfsSingle, pfsConfig, ff) for ff in fiberId]

        armRefList = list(butler.subset("raw", dataId=dataRef.dataId))
        armList = []
        fiberToArm = defaultdict(list)
        for ii, ref in enumerate(armRefList):
            arm = ref.get("pfsArm")
            sky1d = ref.get("sky1d")
            subtractSky1d(arm, pfsConfig, sky1d)
            fluxCalibrate(arm, pfsConfig, fluxCal)
            for ff in arm.fiberId:
                fiberToArm[ff].append(ii)
            armList.append(arm)

        # Add the fluxTable
        for ss, ff in zip(spectra, fiberId):
            ss.fluxTable = self.fluxTable.run([ref.dataId for ref in armRefList],
                                              [armList[ii].extractFiber(PfsSingle, pfsConfig, ff) for
                                               ii in fiberToArm[ff]],
                                              MaskHelper.fromMerge([armList[ii].flags]))
            ss.metadata = getPfsVersions()

        if self.config.doWrite:
            dataRef.put(fluxCal, "fluxCal")
            for ff, spectrum in zip(fiberId, spectra):
                dataId = spectrum.getIdentity().copy()
                dataId.update(dataRef.dataId)
                butler.put(spectrum, "pfsSingle", dataId)
                butler.put(mergedLsf[ff], "pfsSingleLsf", dataId)
        return Struct(fluxCal=fluxCal, spectra=spectra)

    def readReferences(self, butler: Butler, pfsConfig: PfsConfig) -> PfsReferenceSet:
        """Read the physical reference fluxes

        If you get a read error here, it's likely because you haven't got a
        physical reference flux; try running ``calibrateReferenceFlux``.

        Parameters
        ----------
        butler : `lsst.daf.persistence.Butler`
            Data butler.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for identifying flux standards. This should
            contain only the fibers of interest.

        Returns
        -------
        references : `dict` mapping `int` to `pfs.datamodel.PfsSimpleSpectrum`
            Reference spectra, indexed by fiber identifier.
        """
        return {ff: butler.get("pfsReference", pfsConfig.getIdentity(ff)) for ff in pfsConfig.fiberId}

    def calculateCalibrations(self, merged: PfsMerged, references: PfsReferenceSet) -> None:
        """Calculate the flux calibration vector for each fluxCal fiber

        Parameters
        ----------
        merged : `PfsMerged`
            Merged spectra. Contains only fluxCal fibers. These will be
            modified.
        references : `dict` mapping `int` to `PfsReference`
            Reference spectra, indexed by fiber identifier.
        """
        ref = np.zeros_like(merged.flux)
        for ii, fiberId in enumerate(merged.fiberId):
            ref[ii] = references[fiberId].flux

        merged.covar[:, 0] += self.config.sysErr*merged.flux  # add systematic error
        merged /= merged.norm
        merged /= ref
        merged.norm[:] = 1.0  # We're deliberately changing the normalisation

    def _getMetadataName(self):
        return None
