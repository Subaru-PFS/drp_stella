from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task

from .makeFluxTable import makeFluxTable

__all__ = ["FluxTableConfig", "FluxTableTask"]


class FluxTableConfig(Config):
    """Configuration for FluxTableTask"""
    ignoreFlags = ListField(dtype=str, default=[], doc="Flags to ignore in coadd")
    rejIterations = Field(dtype=int, default=3, doc="Number of rejection iterations")
    rejThreshold = Field(dtype=float, default=3.0, doc="Rejection threshold (standard deviations)")


class FluxTableTask(Task):
    """Build a FluxTable"""
    ConfigClass = FluxTableConfig
    _DefaultName = "fluxTable"

    def run(self, identities, spectra, flags):
        """Create a FluxTable from multiple spectra

        Parameters
        ----------
        identities : iterable of `dict`
            Key-value pairs describing the identity of each spectrum. Requires at
            least the ``visit`` and ``arm`` keywords.
        spectra : iterable of `pfs.datamodel.PfsFiberArray`
            Spectra to coadd.
        flags : `pfs.datamodel.MaskHelper`
            Helper for dealing with symbolic names for mask values.

        Returns
        -------
        fluxTable : `pfs.datamodel.FluxTable`
            Fluxes at near the native resolution.
        """
        return makeFluxTable(identities, spectra, flags, self.config.ignoreFlags,
                             self.config.rejIterations, self.config.rejThreshold)
