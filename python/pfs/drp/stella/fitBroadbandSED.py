from pfs.datamodel.pfsConfig import PfsConfig, TargetType

import lsstDebug
from lsst.pex.config import Config, ChoiceField, Field
from lsst.pipe.base import Task
from lsst.utils import getPackageDir

from .fluxModelSet import FluxModelSet
from .utils.math import ChisqList
from .utils import debugging

import numpy
import numpy.lib.recfunctions

from typing import Dict
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

    soften = Field(
        doc="Soften flux errors: err**2 -> err**2 + (soften*flux)**2",
        dtype=float,
        default=0.1,
        optional=False,
    )


class FitBroadbandSEDTask(Task):
    """Fit an observed SED with synthetic model SEDs"""

    ConfigClass = FitBroadbandSEDConfig
    _DefaultName = "fitBroadbandSED"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fluxLibrary = FluxModelSet(getPackageDir("fluxmodeldata")).parameters
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, pfsConfig: PfsConfig) -> Dict[int, NDArray[numpy.float64]]:
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
        broadbandPDF : `Dict[int, NDArray[numpy.float64]]`
            ``broadbandPDF[fiberId][iSED]`` is the normalized probability
            of the SED ``iSED`` matching the spectrum ``fiberId``.
            If the targetType of ``fiberId`` is not ``TargetType.FLUXSTD``,
            ``broadbandPDF[fiberId]`` does not exist.
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

        chisqs: Dict[ChisqList] = {}
        for fiberId, targetType, filterNames, bbFlux, bbFluxErr in zip(
            pfsConfig.fiberId, pfsConfig.targetType, pfsConfig.filterNames, broadbandFlux, broadbandFluxErr
        ):
            if targetType == TargetType.FLUXSTD:
                chisqs[fiberId] = self.getChisq(filterNames, bbFlux, bbFluxErr)

        if self.debugInfo.doWriteChisq:
            debugging.writeExtraData(
                f"fitBroadbandSED-output/chisq-{pfsConfig.filename}.pickle",
                chisq=chisqs,
            )

        return {fiberId: chisq.toProbability() for fiberId, chisq in chisqs.items()}

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
        # Soften noises
        observedNoises = numpy.hypot(observedNoises[isgood], self.config.soften * observedFluxes)

        filterNames = [f for f, good in zip(filterNames, isgood) if good]

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
