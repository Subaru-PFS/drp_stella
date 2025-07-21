from types import SimpleNamespace

from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
from sqlalchemy import create_engine, text

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
    dbUrl = Field(dtype=str, default=None, optional=True, doc="URL for the remote Gaia catalog database")
    dbSelectSql = Field(
        dtype=str,
        default=None,
        optional=True,
        doc=("SQL query to select data from the remote Gaia catalog; "
             "this should include placehoders for 'ra' and 'dec' (degrees) and 'radius' (arcseconds)"),
    )


class GaiaTask(Task):
    """Task to search for a source in the Gaia catalog"""
    ConfigClass = GaiaConfig
    config: GaiaConfig
    _DefaultName = "gaia"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dbEngine = None
        self.catalog = None
        if self.config.dbUrl is not None and self.config.dbSelectSql is not None:
            self.dbEngine = create_engine(self.config.dbUrl)
        elif GaiaCatalog is not None:
            if self.config.path is not None:
                self.catalog = GaiaCatalog.from_path(self.config.path)
            else:
                self.catalog = GaiaCatalog.from_package()

    def run(self, ra: float, dec: float, radius: float = 1.0) -> SimpleNamespace | None:
        """Find the nearest source

        Note that none of these methods account for proper motion or parallax,
        so ensure that the ``radius`` is large enough to include the source
        at the epoch of interest.

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
        if self.dbEngine is not None and self.config.dbSelectSql is not None:
            return self.findDatabase(ra, dec, radius)
        elif self.catalog:
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

    def findDatabase(self, ra: float, dec: float, radius: float = 1.0) -> SimpleNamespace | None:
        """Find the nearest Gaia source in a SQL database

        The config parameters ``dbUrl`` and ``dbSelectSql`` must be set to
        specify the database connection and query. For example:

            config.dbUrl = "postgresql://user:password@hostname:5432/gaia"
            config.dbSelectSql = (
                "SELECT * FROM gaia3 "
                "WHERE q3c_radial_query(ra, dec, :ra, :dec, :radius/3600) "
                "ORDER BY q3c_dist(ra, dec, :ra, :dec) "
                "LIMIT 1"
            )

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
        assert self.dbEngine is not None and self.config.dbSelectSql is not None
        with self.dbEngine.connect() as conn:
            result = conn.execute(text(self.config.dbSelectSql), dict(ra=ra, dec=dec, radius=radius))
            rows = result.mappings().all()
        if not rows:
            self.log.debug("No Gaia source found for ra=%f dec=%f radius=%f", ra, dec, radius)
            return None
        if len(rows) > 1:
            self.log.warn(
                "Multiple (%d) Gaia sources found for ra=%f dec=%f radius=%f; taking the first",
                len(rows), ra, dec, radius
            )
        return SimpleNamespace(**rows[0])

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
