from pfs.datamodel.pfsConfig import PfsConfig, TargetType

import lsstDebug
import lsst.daf.persistence
from lsst.pex.config import Config, ChoiceField, DictField
from lsst.pipe.base import Task
from lsst.utils import getPackageDir

from .fluxModelSet import FluxModelSet
from .utils.math import ChisqList
from .utils import debugging

import numpy
import numpy.lib.recfunctions

from typing import List, Union
from typing import Sequence

try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = Sequence


class FitBroadbandSEDConfig(Config):
    """Configuration for FitBroadbandSEDTask"""

    broadbandFluxType = ChoiceField(
        doc="Type of broadband fluxes to use.",
        dtype=str,
        allowed={
            "fiber": "Use `psfConfig.fiberFlux`.",
            "psf": "Use `psfConfig.psfFlux`.",
            "total": "Use `psfConfig.totalFlux`.",
        },
        default="psf",
        optional=False,
    )

    filterMappings = DictField(
        keytype=str,
        itemtype=str,
        default={
            "g_hsc": "HSCg",
            "r_old_hsc": "HSCr",
            "r2_hsc": "HSCr2",
            "i_old_hsc": "HSCi",
            "i2_hsc": "HSCi2",
            "z_hsc": "HSCz",
            "y_hsc": "HSCy",
            "g_ps1": "PS1g",
            "r_ps1": "PS1r",
            "i_ps1": "PS1i",
            "z_ps1": "PS1z",
            "y_ps1": "PS1y",
            "bp_gaia": "GaiaBp",
            "rp_gaia": "GaiaRp",
            "g_gaia": "GaiaG",
            "u_sdss": "SDSSu",
            "g_sdss": "SDSSg",
            "r_sdss": "SDSSr",
            "i_sdss": "SDSSi",
            "z_sdss": "SDSSz",
        },
        doc="Conversion table from pfsConfig's filter names to those used by `fluxLibrary`",
    )


class FitBroadbandSEDTask(Task):
    """Fit an observed SED with synthetic model SEDs"""

    ConfigClass = FitBroadbandSEDConfig
    _DefaultName = "fitBroadbandSED"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fluxLibrary = FluxModelSet(getPackageDir("fluxmodeldata")).parameters
        self.debugInfo = lsstDebug.Info(__name__)

    def runDataRef(
        self, dataRef: lsst.daf.persistence.ButlerDataRef
    ) -> List[Union[NDArray[numpy.float64], None]]:
        """For each spectrum,
        calculate probabilities of model SEDs matching the spectrum.

        Only spectra for which ``targetType == TargetType.FLUXSTD``
        are processed.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for merged spectrum.

        Returns
        -------
        broadbandPDF : `list` of `numpy.array` of `float`
            ``broadbandPDF[iSpectrum][iSED]`` is the normalized probability
            of the SED ``iSED`` matching the spectrum ``iSpectrum``.
            If the targetType of ``iSpectrum`` is not ``TargetType.FLUXSTD``,
            ``broadbandPDF[iSpectrum]`` is None.
        """
        pfsConfig = dataRef.get("pfsConfig")
        return self.run(pfsConfig)

    def run(self, pfsConfig: PfsConfig) -> List[Union[NDArray[numpy.float64], None]]:
        """For each spectrum,
        calculate probabilities of model SEDs matching the spectrum.

        Only spectra for which ``targetType == TargetType.FLUXSTD``
        are processed, and included in the returned list.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            Top-end configuration, for identifying sky fibers.

        Returns
        -------
        broadbandPDF : `list` of `numpy.array` of `float`
            ``broadbandPDF[iSpectrum][iSED]`` is the normalized probability
            of the SED ``iSED`` matching the spectrum ``iSpectrum``.
            If the targetType of ``iSpectrum`` is not ``TargetType.FLUXSTD``,
            ``broadbandPDF[iSpectrum]`` is None.
        """
        if self.config.broadbandFluxType == "fiber":
            broadbandFlux = pfsConfig.fiberFlux
            broadbandFluxErr = pfsConfig.fiberFluxErr
        elif self.config.broadbandFluxType == "psf":
            broadbandFlux = pfsConfig.psfFlux
            broadbandFluxErr = pfsConfig.psfFluxErr
        elif self.config.broadbandFluxType == "total":
            broadbandFlux = pfsConfig.totalFlux
            broadbandFluxErr = pfsConfig.totalFluxErr
        else:
            raise ValueError(
                f"config.broadbandFluxType must be one of fiber|psf|total."
                f" ('{self.config.broadbandFluxType}')"
            )

        chisqs: List[Union[ChisqList, None]] = []
        for targetType, filterNames, bbFlux, bbFluxErr in zip(
            pfsConfig.targetType, pfsConfig.filterNames, broadbandFlux, broadbandFluxErr
        ):
            if targetType == TargetType.FLUXSTD:
                chisqs.append(self.getChisq(filterNames, bbFlux, bbFluxErr))
            else:
                chisqs.append(None)

        if self.debugInfo.doWriteChisq:
            debugging.writeExtraData(
                f"fitBroadbandSED-output/chisq-{pfsConfig.filename}.pickle",
                fiberId=pfsConfig.fiberId,
                chisq=chisqs,
            )

        return [(chisq.toProbability() if chisq is not None else None) for chisq in chisqs]

    def getChisq(
        self, filterNames: Sequence[str], bbFlux: Sequence[float], bbFluxErr: Sequence[float]
    ) -> ChisqList:
        """Calculate chi square for various model SEDs compared to ``bbFlux``.

        Parameters
        ----------
        filterNames : `list` of `str`
            List of filters used to measure the broadband fluxes for each filter.
            e.g. ``["g_hsc", "i2_hsc"]``.
        bbFlux : `list` of `float`
            Array of broadband fluxes for each fiber, in [nJy].
        bbFluxErr : `list` of `float`
            Array of broadband flux errors for each fiber in [nJy].

        Returns
        -------
        chisq : `ChisqList`
            Chi square for various model SEDs compared to ``bbFlux``.
        """
        if not (len(filterNames) == len(bbFlux) == len(bbFluxErr)):
            raise ValueError("Lengths of arguments must be equal.")

        observedFluxes = numpy.asarray(bbFlux, dtype=float)
        observedNoises = numpy.asarray(bbFluxErr, dtype=float)

        isgood = numpy.isfinite(observedFluxes) & (observedNoises > 0)
        if not numpy.any(isgood):
            nSEDs = len(self.fluxLibrary)
            return ChisqList(numpy.zeros(shape=(nSEDs,), dtype=float), 0)

        observedFluxes = observedFluxes[isgood]
        observedNoises = observedNoises[isgood]

        # Convert filter names.
        filterNames = [self.config.filterMappings.get(f, f) for f, good in zip(filterNames, isgood) if good]

        # Note: fluxLibrary.shape == (nSEDs, nBands)
        fluxLibrary = numpy.lib.recfunctions.structured_to_unstructured(
            self.fluxLibrary[filterNames], dtype=float
        )

        # Note: observedFluxes.shape == (1, nBands)
        observedFluxes = observedFluxes.reshape(1, -1)
        # Note: observedNoises.shape == (1, nBands)
        observedNoises = observedNoises.reshape(1, -1)
        # Note: numer.shape == (nSEDs, 1)
        numer = numpy.sum(fluxLibrary * (observedFluxes / (observedNoises**2)), axis=1, keepdims=True)
        # Note: denom.shape == (nSEDs, 1)
        denom = numpy.sum((fluxLibrary / observedNoises) ** 2, axis=1, keepdims=True)
        # Note: alpha.shape == (nSEDs, 1)
        alpha = numer / denom

        # Compute chi square
        # Note: chisq.shape == (nSEDs,)
        chisq = numpy.sum(((observedFluxes - alpha * fluxLibrary) / observedNoises) ** 2, axis=1)

        degree_of_freedom = len(filterNames) - 1
        return ChisqList(chisq, degree_of_freedom)
