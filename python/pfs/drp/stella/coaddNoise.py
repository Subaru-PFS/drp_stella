from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import hashlib
import numpy as np

from lsst.pex.config import Field
from lsst.pipe.base import Struct

from lsst.pipe.base import PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

from pfs.datamodel import Identity, PfsConfig, Target, TargetType, FiberStatus, MaskHelper
from pfs.datamodel.drp import PfsCoadd, PfsObject

from .barycentricCorrection import applyBarycentricCorrection
from .coaddSpectra import CoaddSpectraTask, CoaddSpectraConfig
from .datamodel.drp import PfsArm, PfsSingle
from .fitFluxCal import calibratePfsArm
from .focalPlaneFunction import FocalPlaneFunction
from .interpolate import calculateDispersion
from .utils import getPfsVersions
from .wavelengthSampling import WavelengthSamplingTask

__all__ = ("CoaddNoiseConfig", "CoaddNoiseTask")


class CoaddNoiseConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "combination", "cat_id", "obj_group"),
):
    """Connections for CoaddNoiseTask

    This looks a lot like CoaddSpectraConnections, but the output is noise
    statistics rather than spectra, and we don't care about some things that
    CoaddSpectra does (like LSF).
    """
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
        multiple=True,
    )
    objectGroupMap = PrerequisiteConnection(
        name="objectGroupMap",
        doc="Object group map",
        storageClass="ObjectGroupMap",
        dimensions=("instrument", "combination", "cat_id"),
    )
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra",
        storageClass="PfsArm",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    pfsArmLsf = InputConnection(
        name="pfsArmLsf",
        doc="1d line-spread function for extracted spectra",
        storageClass="LsfDict",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    sky1d = InputConnection(
        name="sky1d",
        doc="1d sky model",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    fluxCal = InputConnection(
        name="fluxCal",
        doc="Flux calibration model",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "visit"),
        multiple=True,
    )

    pfsCoaddNoise = OutputConnection(
        name="pfsCoaddNoise",
        doc="Noise statistics of coadded object spectra",
        storageClass="PfsCoadd",
        dimensions=("instrument", "combination", "cat_id", "obj_group"),
    )


class CoaddNoiseConfig(CoaddSpectraConfig, pipelineConnections=CoaddNoiseConnections):
    """Configuration for CoaddNoiseTask"""
    fluxTable = None  # Disable flux table generation
    flux = Field(dtype=float, default=1.0e6, doc="Flux level to use (nJy)")
    samples = Field(dtype=int, default=100, doc="Number of samples to use")


class CoaddNoiseTask(CoaddSpectraTask):
    """Measure coadd noise statistics for PFS object spectra

    This is similar to CoaddSpectraTask, but instead of coadding spectra it
    coadds noise statistics.
    """

    ConfigClass = CoaddNoiseConfig
    _DefaultName = "coaddNoise"

    config: CoaddNoiseConfig  # type: ignore
    wavelength: WavelengthSamplingTask

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("wavelength")

    def run(self, data: Mapping[Identity, Struct]) -> Struct:
        """Coadd multiple observations

        data : `dict` mapping `Identity` to `Struct`
            Input data. Each element contains data for a single spectrograph
            arm, with attributes:
            - ``identity`` (`Identity`): identity of the data.
            - ``pfsArm`` (`PfsArm`): extracted spectra from spectrograph arm.
            - ``sky1d`` (`FocalPlaneFunction`): 1d sky subtraction model.
            - ``fluxCal`` (`FocalPlaneFunction`): flux calibration solution.
            - ``pfsConfig`` (`PfsConfig`): PFS fiber configuration.

        Returns
        -------
        pfsCoaddNoise : `PfsCoadd`
            Results of coadding noise spectra, indexed by target.
        """
        targetTypes = (TargetType.SCIENCE, TargetType.FLUXSTD, TargetType.SKY)
        targetSources = defaultdict(list)
        for identity, dd in data.items():
            for target in dd.pfsConfig.select(fiberStatus=FiberStatus.GOOD, targetType=targetTypes):
                targetSources[target].append(identity)
            self.resetSpectra(dd.pfsArm, dd.pfsConfig, dd.sky1d, dd.fluxCal)

        pfsCoaddNoise: dict[Target, PfsObject] = {}
        for target, sources in targetSources.items():
            result = self.process(target, {identity: data[identity] for identity in sources})
            pfsCoaddNoise[target] = result.pfsObject

        return Struct(pfsCoaddNoise=PfsCoadd(pfsCoaddNoise.values(), getPfsVersions()))

    def resetSpectra(
        self,
        pfsArm: PfsArm,
        pfsConfig: PfsConfig,
        sky1d: FocalPlaneFunction,
        fluxCal: FocalPlaneFunction,
    ) -> None:
        """Reset spectra to a fixed flux level

        This is like a reverse version of "calibratePfsArm": given a target
        calibrated flux level (in nJy), set the correspond flux in the pfsArm
        (in electrons).

        We'll add noise later.
        """
        pfsConfig = pfsConfig.select(fiberId=pfsArm.fiberId)
        flux = np.full_like(pfsArm.flux, self.config.flux)  # nJy

        # Flux calibration is applied in the barycentric frame
        applyBarycentricCorrection(pfsArm, inverse=False)  # Flux calibration is in barycentric frame
        try:
            cal = fluxCal(pfsArm.wavelength, pfsConfig)  # Flux calibration, in normalized electrons/nm/nJy
            dispersion = calculateDispersion(pfsArm.wavelength)  # barycentric nm per pixel
            norm = pfsArm.norm/dispersion  # normalization, in electrons/nm
            flux *= cal.values*norm  # convert to electrons/nm
        finally:
            applyBarycentricCorrection(pfsArm, inverse=True)

        # Sky is in observed frame
        sky = sky1d(pfsArm.wavelength, pfsConfig)  # Sky model, in normalized electrons/nm
        dispersion = calculateDispersion(pfsArm.wavelength)  # observed nm per pixel
        norm = pfsArm.norm/dispersion  # normalization, in electrons/nm
        flux += sky.values*norm  # add sky electrons/nm
        flux *= dispersion  # convert to electrons

        pfsArm.flux[:] = flux

    def addNoise(self, pfsArm: PfsArm, number: int) -> PfsArm:
        """Add noise to a pfsArm spectrum

        We add Gaussian noise according to the variance in the pfsArm.

        We return a new PfsArm with the noisy data, to avoid modifying the
        input.
        """
        seed = int.from_bytes(hashlib.sha256((repr(pfsArm.identity) + repr(number)).encode()).digest())
        rng = np.random.default_rng(seed)

        noisy = PfsArm.fromMerge([pfsArm])  # Deep copy
        noisy.flux += rng.normal(size=pfsArm.flux.shape)*np.sqrt(pfsArm.variance)
        return noisy

    def getSpectrum(self, target: Target, data: Struct, number: int | None = None) -> PfsSingle:
        """Return a calibrated spectrum for the nominated target

        Parameters
        ----------
        target : `Target`
            Target of interest.
        data : `Struct`
            Data for a pfsArm containing the object of interest.
        number : `int`, optional
            Sample number (for random number seeding), if non-`None`.

        Returns
        -------
        spectrum : `PfsSingle`
            Calibrated spectrum of the target.
        """
        spectrum = data.pfsArm.select(data.pfsConfig, catId=target.catId, objId=target.objId)
        if number is not None:
            spectrum = self.addNoise(spectrum, number)
        spectrum = calibratePfsArm(spectrum, data.pfsConfig, data.sky1d, data.fluxCal)
        return spectrum.extractFiber(PfsSingle, data.pfsConfig, spectrum.fiberId[0])

    def process(self, target: Target, data: dict[Identity, Struct]) -> Struct:
        """Generate coadd noise spectra for a single target

        We stuff statistics into a PfsObject as follows:
        * flux: mean bias
        * mask: bitwise-AND of all the masks
        * variance: variance of fluxes from samples
        * covariance: mean covariance of fluxes from samples
        * sky: no noise combined spectrum

        Parameters
        ----------
        target : `Target`
            Target for which to generate coadded spectra.
        data : `dict` mapping `Identity` to `Struct`
            Data from which to generate coadded spectra. These are the results
            from the ``readData`` method.

        Returns
        -------
        pfsObject : `PfsObject`
            Coadded noise spectrum.
        """
        pfsConfigList = [dd.pfsConfig.select(catId=target.catId, objId=target.objId) for dd in data.values()]
        target = self.getTarget(target, pfsConfigList)
        observations = self.getObservations(data.keys(), pfsConfigList)
        wavelength = self.wavelength.run(any(ident.arm == "m" for ident in data))

        lsfList = [dd.pfsArmLsf for dd in data.values()]
        flags = MaskHelper.fromMerge([dd.pfsArm.flags for dd in data.values()])
        combineArgs = (lsfList, flags, wavelength)
        original = self.combine([self.getSpectrum(target, dd) for dd in data.values()], *combineArgs)

        combinations = [
            self.combine([self.getSpectrum(target, dd, ii) for dd in data.values()], *combineArgs)
            for ii in range(self.config.samples)
        ]

        residuals = np.array([combo.flux - original.flux for combo in combinations])

        flux = np.mean(residuals, axis=0)
        covariance = np.zeros((3, original.flux.size), dtype=original.covar.dtype)
        covariance[0] = np.var(residuals, axis=0, ddof=1)  # Variance
        covariance[1, :-1] = np.mean(residuals[:, :-1]*residuals[:, 1:], axis=0)  # lag-1 covariance
        covariance[2, :-2] = np.mean(residuals[:, :-2]*residuals[:, 2:], axis=0)  # lag-2 covariance

        mask = np.bitwise_and.accumulate([combo.mask for combo in combinations])[0]

        coadd = PfsObject(
            target,
            observations,
            original.wavelength,
            flux,
            mask,
            original.flux,
            covariance,
            original.covar2,
            flags,
            getPfsVersions(),
        )
        return Struct(pfsObject=coadd)
