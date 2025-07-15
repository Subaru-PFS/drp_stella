from types import SimpleNamespace

from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u

from lsst.pex.config import Config, Field
from lsst.pipe.base import Task

try:
    from pfs.gaia import GaiaCatalog
except ImportError:
    GaiaCatalog = None

__all__ = ("GaiaConfig", "GaiaTask")


class GaiaConfig(Config):
    """Configuration for Gaia data access."""
    path = Field(dtype=str, default=None, optional=True, doc="Path to the local Gaia catalog directory")


class GaiaTask(Task):
    """Task to search for a source in the Gaia catalog"""
    ConfigClass = GaiaConfig
    _DefaultName = "gaia"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if GaiaCatalog is not None:
            if self.config.path is not None:
                self.catalog = GaiaCatalog.from_path(self.config.path)
            else:
                self.catalog = GaiaCatalog.from_package()
        else:
            self.catalog = None

    def run(self, ra: float, dec: float, radius: float = 1.0) -> SimpleNamespace | None:
        """Find the nearest source

        Parameters
        ----------
        ra : `float`
            Right Ascension in degrees.
        dec : `float`
            Declination in degrees.
        radius : `float`, optional
            Search radius in arcseconds.

        Returns
        -------
        result : `SimpleNamespace` or `None`
            An object containing the source information and distance, or
            `None` if no source is found.
        """
        if self.catalog:
            return self.findLocal(ra, dec, radius)
        return self.findRemote(ra, dec, radius)

    def findLocal(self, ra: float, dec: float, radius: float = 1.0) -> SimpleNamespace | None:
        """Find the nearest source in the local Gaia catalog

        Parameters
        ----------
        ra : `float`
            Right Ascension in degrees.
        dec : `float`
            Declination in degrees.
        radius : `float`, optional
            Search radius in arcseconds.

        Returns
        -------
        result : `SimpleNamespace` or `None`
            An object containing the source information and distance, or
            `None` if no source is found.
        """
        assert self.catalog is not None
        return self.catalog.find_nearest(ra, dec, radius)

    def findRemote(self, ra: float, dec: float, radius: float = 1.0) -> SimpleNamespace | None:
        """Find the nearest source in the remote Gaia catalog

        Parameters
        ----------
        ra : `float`
            Right Ascension in degrees.
        dec : `float`
            Declination in degrees.
        radius : `float`, optional
            Search radius in arcseconds.

        Returns
        -------
        result : `SimpleNamespace` or `None`
            An object containing the source information and distance, or
            `None` if no source is found.
        """
        coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        results = Gaia.cone_search(coord, radius=radius*u.arcsec).get_results()
        if results is None:
            raise RuntimeError("Gaia query failed; check network connection or Gaia service status.")
        if len(results) == 0:
            self.log.debug("No Gaia source found for ra=%f dec=%f radius=%f", ra, dec, radius)
            return None
        if len(results) > 1:
            self.log.warn(
                "Multiple (%d) Gaia sources found for %s within radius=%f; taking the closest",
                len(results),
                coord,
                radius
            )
        return SimpleNamespace(**results[0])
