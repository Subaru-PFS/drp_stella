from typing import Union
import numpy as np

from lsst.pex.config import Config, Field, ConfigField
from lsst.pipe.base import Task, Struct

from lsst.obs.pfs.blackSpots import BlackSpotsConfig
from pfs.datamodel import PfsConfig, PfsFiberArraySet

from .SpectrumSetContinued import SpectrumSet


class BlackSpotCorrectionConfig(Config):
    data = ConfigField(dtype=BlackSpotsConfig, doc="Black spot source")
    radius = Field(dtype=float, default=1.108, doc="Radius of influence of black spots (mm)")
    slope = Field(dtype=float, default=1.232, doc="Slope of black spot correction inside 'radius' (mm^-1)")


class BlackSpotCorrectionTask(Task):
    ConfigClass = BlackSpotCorrectionConfig
    _DefaultName = "blackSpot"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blackSpots = self.config.data.read()

    def run(self, pfsConfig: PfsConfig, spectra: Union[PfsFiberArraySet, SpectrumSet]) -> Struct:
        """Correct spectra for black spots

        Black spots have a radius of influence (the penumbra of the black spot),
        inside which the transmission is reduced, apparently linearly with
        distance from the spot. We calculate the correction, and apply it to the
        spectral normalisation.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`
            Fiber configuration.
        spectra : SpectrumSet or subclass of `pfs.datamodel.PfsFiberArraySet`
            Spectra to correct.

        Returns
        -------
        backspot : `lsst.pipe.base.Struct`
            Black spot data, including the correction to apply. Elements are:

            - ``distance``: distance to nearest black spot (mm).
            - ``spotId``: spotId of nearest black spot; -1 if no spot was found.
            - ``x``: x-coordinate of nearest black spot (mm).
            - ``y``: y-coordinate of nearest black spot (mm).
            - ``r``: radius of nearest black spot (mm).
            - ``correction``: correction to apply.
        """
        pfsConfig = pfsConfig.select(fiberId=spectra.fiberId)
        bs = self.calculate(pfsConfig)
        self.apply(spectra, bs)
        return bs

    def calculate(self, pfsConfig: PfsConfig) -> Struct:
        """Calculate the black spot correction

        Parameters
        ----------
        pfsConfig : `PfsConfig`
            Fiber configuration.

        Returns
        -------
        backspot : `lsst.pipe.base.Struct`
            Black spot data, including the correction to apply. Elements are:

            - ``distance``: distance to nearest black spot (mm).
            - ``spotId``: spotId of nearest black spot; -1 if no spot was found.
            - ``x``: x-coordinate of nearest black spot (mm).
            - ``y``: y-coordinate of nearest black spot (mm).
            - ``r``: radius of nearest black spot (mm).
            - ``correction``: correction to apply.
        """
        bs = self.blackSpots.findNearest(pfsConfig.pfiCenter)
        bs.correction = np.ones_like(bs.distance)
        with np.errstate(invalid="ignore"):
            select = np.isfinite(bs.distance) & (bs.distance < self.config.radius)
        bs.correction[select] = 1.0 - self.config.slope*(self.config.radius - bs.distance[select])
        return bs

    def apply(self, spectra: Union[PfsFiberArraySet, SpectrumSet], bs: Struct):
        """Apply black spot correction to spectra

        We apply the correction to the normalisation of the spectra, so that
        the measured values are unchanged.

        Parameters
        ----------
        spectra : `SpectrumSet` or subclass of `pfs.datamodel.PfsFiberArraySet`
            Spectra to correct; modified.
        bs : `lsst.pipe.base.Struct`
            Black spot data, including the correction to apply. Elements are:

            - ``distance``: distance to nearest black spot (mm).
            - ``spotId``: spotId of nearest black spot; -1 if no spot was found.
            - ``x``: x-coordinate of nearest black spot (mm).
            - ``y``: y-coordinate of nearest black spot (mm).
            - ``r``: radius of nearest black spot (mm).
            - ``correction``: correction to apply.
        """
        self.log.info("Applying black spot correction")

        # We want the normalised flux to be increased by a correction that is less than unity.
        # Multiplying the normalisation by the correction results in a lower normalisation, and hence a
        # larger normalised flux.
        for ss, spotId, distance, corr in zip(spectra, bs.spotId, bs.distance, bs.correction):
            ss.norm *= corr
            ss.notes["blackSpotId"] = spotId
            ss.notes["blackSpotDistance"] = distance
            ss.notes["blackSpotCorrection"] = corr
