from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct, Task

from pfs.datamodel.pfsConfig import CobraCommand

if TYPE_CHECKING:
    from pfs.datamodel import PfsConfig, PfsThetaPhiFine, PfsThetaPhiModel
    from .datamodel.drp import PfsArm

__all__ = ("ThetaPhiCorrectionConfig", "ThetaPhiCorrectionTask", "isCobraHome")


def isCobraHome(
    pfsConfig: PfsConfig, homeTheta: np.ndarray, homePhi: np.ndarray, tolerance: float = 0.5
) -> np.ndarray:
    """Is each fiber's cobra in the cobraHome state (both arms against their hard stops)?

    The state is recognised by what the FPS did (the pfsConfig's ``cobraCommand``), not by the angles
    alone:  in the cobraHome state the fiber sits 60-70 microns off where its angles would put it.
    ``HOME_BOTH`` (DAMD-200) says so directly;  ``HOME`` doesn't say which arm went home, so for it we also
    require theta and phi to be at their home angles.

    Parameters
    ----------
    pfsConfig : `pfs.datamodel.PfsConfig`
        Configuration of the fibers, in the order of interest.
    homeTheta, homePhi : `numpy.ndarray` of `float`
        Each fiber's home angles (degrees), in the same order.
    tolerance : `float`, optional
        How close (degrees) the angles must be to home for ``HOME``.

    Returns
    -------
    isHome : `numpy.ndarray` of `bool`
        Is each fiber's cobra in the cobraHome state?
    """
    command = np.asarray(pfsConfig.cobraCommand)
    isHome = np.zeros(len(command), dtype=bool)
    homeBoth = getattr(CobraCommand, "HOME_BOTH", None)
    if homeBoth is not None:
        isHome |= command == homeBoth
    with np.errstate(invalid="ignore"):
        dTheta = np.abs((np.asarray(pfsConfig.cobraTheta) - homeTheta + 180) % 360 - 180)
        dPhi = np.abs(np.asarray(pfsConfig.cobraPhi) - homePhi)
        isHome |= (command == CobraCommand.HOME) & (dTheta < tolerance) & (dPhi < tolerance)
    return isHome


class ThetaPhiCorrectionConfig(Config):
    """Configuration for ThetaPhiCorrectionTask"""

    homeTolerance = Field(
        dtype=float,
        default=0.5,
        doc="How close (degrees) a cobra commanded HOME must be to its home angles to be cobraHome",
    )
    doFine = Field(dtype=bool, default=True, doc="Apply the fine layer (if provided)?")


class ThetaPhiCorrectionTask(Task):
    """Correct each fiber's throughput for the position of its cobra

    The fiberNorms are measured with the cobras at home, so a fiber anywhere else is corrected by
    ``corrThetaPhi`` (the smooth model, `pfs.datamodel.PfsThetaPhiModel`) times ``corrFine`` (each fiber's
    fine structure, `pfs.datamodel.PfsThetaPhiFine`);  both are 1 in the cobraHome state.  This task
    calculates the factors;  `PfiCorrectionTask` applies them along with its own terms, and records the
    total.
    """

    ConfigClass = ThetaPhiCorrectionConfig
    _DefaultName = "thetaPhiCorrection"

    config: ThetaPhiCorrectionConfig

    def run(
        self,
        pfsArm: PfsArm,
        pfsConfig: PfsConfig,
        thetaPhiModel: PfsThetaPhiModel,
        thetaPhiFine: PfsThetaPhiFine | None,
        arm: str,
        date: str,
    ) -> Struct:
        """Calculate the corrections for the fibers of a pfsArm

        Parameters
        ----------
        pfsArm : `pfs.datamodel.drp.PfsArm`
            Spectra to correct (used for their fiberIds;  not modified).
        pfsConfig : `pfs.datamodel.PfsConfig`
            Configuration of the fibers on the focal plane.
        thetaPhiModel : `pfs.datamodel.PfsThetaPhiModel`
            The smooth model.
        thetaPhiFine : `pfs.datamodel.PfsThetaPhiFine`, optional
            Each fiber's fine structure, for this arm;  must have been fitted against ``thetaPhiModel``.
        arm : `str`
            Arm of the spectra (selects alpha).
        date : `str`
            Date of the observation, ISO-8601 (selects delta).

        Returns
        -------
        factor : `numpy.ndarray` of `float`
            The product of the factors for each fiber of ``pfsArm``;  NaN where there's no usable smooth
            model.
        corrThetaPhi, corrFine : `numpy.ndarray` of `float`
            The two factors.
        isCobraHome : `numpy.ndarray` of `bool`
            Was each fiber's cobra in the cobraHome state?
        """
        pfsConfig = pfsConfig.select(fiberId=pfsArm.fiberId)
        fiberId = pfsConfig.fiberId
        homeTheta, homePhi = thetaPhiModel.getHomeAngles(fiberId)
        isHome = isCobraHome(pfsConfig, homeTheta, homePhi, self.config.homeTolerance)

        theta = np.asarray(pfsConfig.cobraTheta, dtype=float)
        phi = np.asarray(pfsConfig.cobraPhi, dtype=float)
        corrThetaPhi = thetaPhiModel.getCorrection(fiberId, theta, phi, arm, date, isHome)
        corrFine = np.ones(len(fiberId))
        if self.config.doFine and thetaPhiFine is not None:
            thetaPhiFine.checkSmoothModel(thetaPhiModel)
            if thetaPhiFine.arm is not None and thetaPhiFine.arm != arm:
                raise RuntimeError(f"thetaPhiFine is for arm {thetaPhiFine.arm}, not {arm}")
            corrFine = thetaPhiFine.getCorrection(fiberId, theta, phi, isHome)

        factor = corrThetaPhi*corrFine
        self.log.info(
            "theta/phi correction: %.4f +/- %.4f (%d fibers;  %d cobraHome;  %d without a model)",
            np.nanmean(factor), np.nanstd(factor), len(factor), isHome.sum(), np.sum(~np.isfinite(factor)),
        )
        return Struct(factor=factor, corrThetaPhi=corrThetaPhi, corrFine=corrFine, isCobraHome=isHome)
