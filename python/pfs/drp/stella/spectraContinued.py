import os
import re

import numpy as np

from lsst.utils import continueClass
from lsst.daf.base import PropertySet
from lsst.pipe.base import Struct

import lsst.afw.image as afwImage

from pfs.datamodel.pfsArm import PfsArm

from .spectra import SpectrumSet, Spectrum

__all__ = ["SpectrumSet"]


@continueClass
class SpectrumSet:
    """Collection of `Spectrum`s

    Persistence is via the `pfs.datamodel.PfsArm` class, and we provide
    methods for interpreting between the two (``toPfsArm``, ``fromPfsArm``).
    We also provide the necessary ``readFits`` and ``writeFits`` methods for I/O
    with the LSST data butler as the ``FitsCatalogStorage`` storage type.
    Because both the butler and the PfsArm's I/O methods determine the path, we
    extract values from the butler's completed pathname using the
    ``fileNameRegex`` class variable and hand the values to the PfsArm's I/O
    methods to re-determine the path. This dance is unfortunate but necessary.
    """
    fileNameRegex = r"^pfsArm-(\d{6})-([brnm])(\d)\.fits.*$"

    def toPfsArm(self, dataId):
        """Convert to a `pfs.datamodel.PfsArm`

        Parameters
        ----------
        dataId : `dict`
            Data identifier, which is expected to contain:

            - ``visit`` (`int`): visit number
            - ``spectrograph`` (`int`): spectrograph number
            - ``arm`` (`str`: "b", "r", "m" or "n"): spectrograph arm
            - ``pfsConfigId`` (`int`, optional): instrument configuration ID

        Returns
        -------
        pfsArm : `pfs.datamodel.PfsArm`
            Spectra in standard PFS datamodel form.
        """
        visit = dataId["visit"]
        spectrograph = dataId["spectrograph"]
        arm = dataId["arm"]
        pfsConfigId = dataId["pfsConfigId"] if "pfsConfigId" in dataId else 0
        pfsArm = PfsArm(visit, spectrograph, arm, pfsConfigId=pfsConfigId)
        pfsArm.flux = self.getAllFluxes()
        pfsArm.covar = self.getAllCovariances()
        pfsArm.mask = self.getAllMasks()
        pfsArm.lam = self.getAllWavelengths()
        pfsArm.lam[pfsArm.lam == 0] = np.nan
        pfsArm.sky = self.getAllBackgrounds()

        if len(self) > 0:
            md = PropertySet()
            self[0].mask.addMaskPlanesToMetadata(md)
            for k in md.names():
                pfsArm._metadata[k] = md.get(k)

        return pfsArm

    @classmethod
    def fromPfsArm(cls, pfsArm):
        """Generate from a `pfs.datamodel.PfsArm`

        Parameters
        ----------
        pfsArm : `pfs.datamodel.PfsArm`
            Spectra in standard PFS datamodel form.

        Returns
        -------
        out : `pfs.drp.stella.SpectrumSet`
            Spectra in drp_stella form.
        """
        numFibers = len(pfsArm.flux)
        if numFibers == 0:
            raise RuntimeError("Unable to construct SpectrumSet from empty PfsArm")
        length = len(pfsArm.flux[0])
        self = cls(length)
        for ii in range(numFibers):
            self.add(Spectrum(pfsArm.flux[ii], pfsArm.mask[ii], pfsArm.sky[ii], pfsArm.covar[ii],
                              pfsArm.lam[ii]))
        return self

    @classmethod
    def _parsePath(cls, path, hdu=None, flags=None):
        """Parse path from the data butler

        We need to determine the ``visit``, ``spectrograph`` and ``arm`` to pass
        to the `pfs.datamodel.PfsArm` I/O methods.

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
        visit, arm, spectrograph = matches.groups()
        visit = int(visit)
        spectrograph = int(spectrograph)
        dataId = dict(visit=visit, spectrograph=spectrograph, arm=arm)
        return Struct(dirName=dirName, fileName=fileName, visit=visit, arm=arm, spectrograph=spectrograph,
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
        pfsArm = self.toPfsArm(parsed.dataId)
        pfsArm.write(parsed.dirName, parsed.fileName)

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
        out : `pfs.drp.stella.SpectrumSet`
            SpectrumSet read from FITS.

        Raises
        ------
        NotImplementedError
            If ``hdu`` or ``flags`` arguments are provided.
        """
        parsed = cls._parsePath(*args, **kwargs)
        pfsArm = PfsArm(parsed.visit, parsed.spectrograph, parsed.arm)
        pfsArm.read(dirName=parsed.dirName, setPfsConfig=False)
        return cls.fromPfsArm(pfsArm)

    def makeImage(self, box, fiberTraces):
        """Make a 2D image of the spectra

        Parameters
        ----------
        box : `lsst.afw.geom.Box2I`
            Bounding box for image.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Fiber traces indicating where on the image the spectra go.

        Returns
        -------
        image : `lsst.afw.image.Image`
            2D image of the spectra.
        """
        image = afwImage.ImageF(box)
        image.set(0.0)
        assert len(self) == len(self), "Number of spectra and fiberTraces don't match"
        for spec, ft in zip(self, fiberTraces):
            fiberImage = ft.constructImage(spec)
            image[fiberImage.getBBox(), afwImage.PARENT] += fiberImage
        return image
