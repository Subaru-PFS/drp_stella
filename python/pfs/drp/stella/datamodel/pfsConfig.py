import os

from pfs.utils.fibers import spectrographFromFiberId, fiberHoleFromFiberId
import pfs.datamodel.pfsConfig

__all__ = ("PfsConfig",)


class PfsConfig(pfs.datamodel.PfsConfig):
    """A class to perform butler-based I/O for pfsConfig

    Provides the necessary ``readFits`` and ``writeFits`` methods for I/O
    with the LSST data butler as the ``FitsCatalogStorage`` storage type.
    Because both the butler and the parent class PfsConfig's I/O methods
    determine the path, we extract values from the butler's completed pathname
    using the ``fileNameRegex`` class variable and hand the values to the
    PfsConfig's I/O methods to re-determine the path. This dance is unfortunate
    but necessary.
    """
    fileNameRegex = r"^pfsConfig-(0x[0-9a-f]+)-([0-9]+)\.fits.*"

    @property
    def spectrograph(self):
        """Return spectrograph number"""
        return spectrographFromFiberId(self.fiberId)

    @property
    def fiberHole(self):
        """Return fiber hole number"""
        return fiberHoleFromFiberId(self.fiberId)

    def writeFits(self, path: str):
        """Write as FITS

        This is the output API for the ``FitsCatalogStorage`` storage type used
        by the LSST data butler.

        Parameters
        ----------
        path : `str`
            Path name from the LSST data butler.
        """
        dirName, fileName = os.path.split(path)
        self.write(dirName, fileName)

    @classmethod
    def readFits(cls, path):
        """Read from FITS

        This is the input API for the ``FitsCatalogStorage`` storage type used
        by the LSST data butler.

        Parameters
        ----------
        path : `str`
            Path name from the LSST data butler. Besides the usual directory and
            filename with extension, this may include a suffix with additional
            characters added by the butler.

        Returns
        -------
        self : `pfs.drp.stella.PfsConfig`
            Configuration read from FITS.

        Raises
        ------
        NotImplementedError
            If ``hdu`` or ``flags`` arguments are provided.
        """
        try:
            # Read file that contains the pfsDesignId and visit in the headers
            return cls._readImpl(path)
        except Exception:
            # Need to get the pfsDesignId and visit from the filename
            parsed = pfs.datamodel.pfsConfig.parsePfsConfigFilename(path)
            return cls._readImpl(path, pfsDesignId=parsed.pfsDesignId, visit=parsed.visit)
