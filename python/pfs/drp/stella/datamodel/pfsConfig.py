import os
import re

from lsst.pipe.base import Struct
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

    @classmethod
    def _parsePath(cls, path, hdu=None, flags=None):
        """Parse path from the data butler

        We need to determine the ``pfsConfigId`` to pass to the
        `pfs.datamodel.PfsConfig` I/O methods.

        Parameters
        ----------
        path : `str`
            Path name from the LSST data butler. Besides the usual directory and
            filename with extension, this may include a suffix with additional
            characters added by the butler.
        hdu : `int`
            Part of the ``FitsCatalogStorage`` API, but not utilised.
        flags : `int`
            Part of the ``FitsCatalogStorage`` API, but not utilised.

        Raises
        ------
        NotImplementedError
            If ``hdu`` or ``flags`` arguments are provided.
        """
        if hdu is not None:
            raise NotImplementedError("%s read/write doesn't use the 'hdu' argument" % (cls.__name__,))
        if flags is not None:
            raise NotImplementedError("%s read/write doesn't use the 'flags' argument" % (cls.__name__,))
        dirName, fileName = os.path.split(path)
        matches = re.search(cls.fileNameRegex, fileName)
        if not matches:
            raise RuntimeError("Unable to parse filename: %s" % (fileName,))
        pfsDesignId = int(matches.group(1), 16)
        visit = int(matches.group(2))
        return Struct(dirName=dirName, fileName=fileName, pfsDesignId=pfsDesignId, visit=visit)

    def writeFits(self, *args, **kwargs):
        """Write as FITS

        This is the output API for the ``FitsCatalogStorage`` storage type used
        by the LSST data butler.

        Parameters
        ----------
        path : `str`
            Path name from the LSST data butler. Besides the usual directory and
            filename with extension, this may include a suffix with additional
            characters added by the butler.
        flags : `int`
            Part of the ``FitsCatalogStorage`` API, but not utilised.

        Raises
        ------
        NotImplementedError
            If ``hdu`` or ``flags`` arguments are provided.
        """
        parsed = self._parsePath(*args, **kwargs)
        self.write(parsed.dirName, parsed.fileName)

    @classmethod
    def readFits(cls, *args, **kwargs):
        """Read from FITS

        This is the input API for the ``FitsCatalogStorage`` storage type used
        by the LSST data butler.

        Parameters
        ----------
        path : `str`
            Path name from the LSST data butler. Besides the usual directory and
            filename with extension, this may include a suffix with additional
            characters added by the butler.
        hdu : `int`
            Part of the ``FitsCatalogStorage`` API, but not utilised.
        flags : `int`
            Part of the ``FitsCatalogStorage`` API, but not utilised.

        Returns
        -------
        self : `pfs.drp.stella.PfsConfig`
            Configuration read from FITS.

        Raises
        ------
        NotImplementedError
            If ``hdu`` or ``flags`` arguments are provided.
        """
        parsed = cls._parsePath(*args, **kwargs)
        return cls.read(parsed.pfsDesignId, parsed.visit, dirName=parsed.dirName)
