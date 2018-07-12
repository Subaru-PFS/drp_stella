import os
import re

from lsst.daf.base import PropertyList
from lsst.utils import continueClass
from lsst.pipe.base import Struct
from lsst.afw.image import PARENT

from pfs.datamodel.pfsFiberTrace import PfsFiberTrace

from .fiberTraces import FiberTraceSet, FiberTrace

__all__ = ["FiberTrace", "FiberTraceSet"]


@continueClass
class FiberTrace:
    def applyToMask(self, mask):
        """Apply the trace mask to the provided mask

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Mask to which to apply the trace mask.
        """
        traceMask = self.getTrace().mask
        mask.Factory(mask, traceMask.getBBox(), PARENT)[:] |= traceMask


@continueClass
class FiberTraceSet:
    """Collection of FiberTrace

    Persistence is via the `pfs.datamodel.PfsFiberTrace` class, and we provide
    methods for interpreting between the two (``toPfsFiberTrace``,
    ``fromPfsFiberTrace``). We also provide the necessary ``readFits`` and
    ``writeFits`` methods for I/O with the LSST data butler as the
    ``FitsCatalogStorage`` storage type. Because both the butler and the
    PfsFiberTrace's I/O methods determine the path, we extract values from the
    butler's completed pathname using the ``fileNameRegex`` class variable and
    hand the values to the PfsFiberTrace's I/O methods to re-determine the path.
    This dance is unfortunate but necessary.
    """
    fileNameRegex = r"^pfsFiberTrace-(\d{4}-\d{2}-\d{2})-0-([brmn])([1-4])\.fits.*"

    def toPfsFiberTrace(self, dataId):
        """Convert to a `pfs.datamodel.PfsFiberTrace`

        Parameters
        ----------
        dataId : `dict`
            Data identifier, which is expected to contain:

            - ``calibDate`` or ``dateObs`` (`str`: "YYYY-MM-DD"): date of
                observation
            - ``spectrograph`` (`int`): spectrograph number
            - ``arm`` (`str`: "b", "r", "m" or "n"): spectrograph arm

        Returns
        -------
        out : `pfs.datamodel.PfsFiberTrace`
            Traces in standard PFS datamodel form.
        """
        obsDate = dataId['calibDate'] if 'calibDate' in dataId else dataId['dateObs']
        spectrograph = dataId['spectrograph']
        arm = dataId['arm']
        metadata = self.getMetadata()
        metadata.set("arm", arm)
        metadata.set("spectrograph", spectrograph)

        out = PfsFiberTrace(obsDate, spectrograph, arm, metadata)
        for ft in self:
            out.fiberId.append(ft.getFiberId())
            out.traces.append(ft.getTrace())

        return out

    @classmethod
    def fromPfsFiberTrace(cls, fiberTrace):
        """Generate from a `pfs.datamodel.PfsFiberTrace`

        Parameters
        ----------
        fiberTrace : `pfs.datamodel.PfsFiberTrace`
            Traces in standard PFS datamodel form.

        Returns
        -------
        out : `pfs.drp.stella.FiberTraceSet`
            Traces in drp_stella form.
        """
        num = len(fiberTrace.traces)
        out = cls(num, fiberTrace.metadata)
        for ii in range(num):
            out.add(FiberTrace(fiberTrace.traces[ii], fiberTrace.fiberId[ii]))
        return out

    @classmethod
    def _parsePath(cls, path, hdu=None, flags=None):
        """Parse path from the data butler

        We need to determine the ``dateObs``, ``spectrograph`` and ``arm`` to
        pass to the `pfs.datamodel.PfsFiberTrace` I/O methods.

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
        dateObs, arm, spectrograph = matches.groups()
        spectrograph = int(spectrograph)
        dataId = dict(dateObs=dateObs, arm=arm, spectrograph=spectrograph)
        return Struct(dirName=dirName, fileName=fileName, dateObs=dateObs, arm=arm, spectrograph=spectrograph,
                      dataId=dataId)

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
        fiberTrace = self.toPfsFiberTrace(parsed.dataId)
        fiberTrace.write(parsed.dirName, parsed.fileName, metadata=fiberTrace.metadata)

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
        out : `pfs.drp.stella.FiberTraceSet`
            Traces read from FITS.

        Raises
        ------
        NotImplementedError
            If ``hdu`` or ``flags`` arguments are provided.
        """
        parsed = cls._parsePath(*args, **kwargs)
        fiberTrace = PfsFiberTrace(parsed.dateObs, parsed.spectrograph, parsed.arm)
        fiberTrace.read(dirName=parsed.dirName)
        return cls.fromPfsFiberTrace(fiberTrace)

    def applyToMask(self, mask):
        """Apply the trace masks to the provided mask

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Mask to which to apply the trace mask.
        """
        for trace in self:
            trace.applyToMask(mask)
