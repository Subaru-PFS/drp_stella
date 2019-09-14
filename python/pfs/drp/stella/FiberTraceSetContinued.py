import os
import re

from lsst.utils import continueClass
from lsst.pipe.base import Struct

from pfs.datamodel.pfsFiberTrace import PfsFiberTrace

from .FiberTraceSet import FiberTraceSet
from .FiberTraceContinued import FiberTrace
from .SpectrumSetContinued import SpectrumSet

__all__ = ["FiberTraceSet"]


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
    fileNameRegex = r"^pfsFiberTrace-(\d{4}-\d{2}-\d{2})-(\d{6})-([brmn])([1-4])\.fits.*"

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
        visit0 = dataId['visit0']
        metadata = self.getMetadata()
        metadata.set("ARM", arm)
        metadata.set("SPECTROGRAPH", spectrograph)

        out = PfsFiberTrace(obsDate, spectrograph, arm, visit0, metadata)
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

    @property
    def fiberId(self):
        """Return the fiberIds of the component fiberTraces"""
        return [ft.fiberId for ft in self]

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
        dateObs, visit0, arm, spectrograph = matches.groups()
        spectrograph = int(spectrograph)
        visit0 = int(visit0)
        dataId = dict(dateObs=dateObs, visit0=visit0, arm=arm, spectrograph=spectrograph)
        return Struct(dirName=dirName, fileName=fileName, dateObs=dateObs, arm=arm, spectrograph=spectrograph,
                      visit0=visit0, dataId=dataId)

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
        fiberTrace = PfsFiberTrace(parsed.dateObs, parsed.spectrograph, parsed.arm, parsed.visit0)
        fiberTrace.read(dirName=parsed.dirName)
        return cls.fromPfsFiberTrace(fiberTrace)

    def extractSpectra(self, maskedImage, detectorMap=None, optimal=True):
        """Extract spectra from an image

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which to extract spectra.
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Map of expected detector coordinates to fiber, wavelength.
            If provided, they will be used to provide a rough wavelength
            calibration.
        optimal : `bool`, optional
            Use optimal extraction? Otherwise, use a simple sum of pixels
            within the trace.

        Returns
        -------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        """
        spectra = SpectrumSet(maskedImage.getHeight())
        for ft in self:
            spectrum = ft.extractSpectrum(maskedImage, optimal)
            if detectorMap is not None:
                spectrum.setWavelength(detectorMap.getWavelength(ft.getFiberId()))
            spectra.add(spectrum)
        return spectra

    def applyToMask(self, mask):
        """Apply the trace masks to the provided mask

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Mask to which to apply the trace mask.
        """
        for trace in self:
            trace.applyToMask(mask)
