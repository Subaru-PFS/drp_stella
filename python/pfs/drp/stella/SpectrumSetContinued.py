import os
import re
import numpy as np

from lsst.utils import continueClass
from lsst.daf.base import PropertySet
from lsst.pipe.base import Struct
import lsst.afw.image as afwImage

from pfs.datamodel.pfsArm import PfsArm, PfsConfig
from .SpectrumContinued import Spectrum
from .SpectrumSet import SpectrumSet

__all__ = ["SpectrumSet"]


@continueClass  # noqa: F811 (redefinition)
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

    def toPfsArm(self, dataId, pfsConfig=None):
        """Convert to a `pfs.datamodel.PfsArm`

        Parameters
        ----------
        dataId : `dict`
            Data identifier, which is expected to contain:

            - ``visit`` (`int`): visit number
            - ``spectrograph`` (`int`): spectrograph number
            - ``arm`` (`str`: "b", "r", "m" or "n"): spectrograph arm

        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Configuration of the PFS top-end.

        Returns
        -------
        pfsArm : `pfs.datamodel.PfsArm`
            Spectra in standard PFS datamodel form.
        """
        visit = dataId["visit"]
        spectrograph = dataId["spectrograph"]
        arm = dataId["arm"]
        pfsConfigId = 0 if pfsConfig is None else pfsConfig.pfiDesignId
        pfsArm = PfsArm(visit, spectrograph, arm, pfsConfig=pfsConfig, pfsConfigId=pfsConfigId)
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
            spectrum = Spectrum(length)
            spectrum.spectrum[:] = pfsArm.flux[ii]
            spectrum.mask.array[:] = pfsArm.mask[ii]
            spectrum.background[:] = pfsArm.sky[ii]
            spectrum.covariance[:] = pfsArm.covar[ii]
            spectrum.wavelength[:] = pfsArm.lam[ii]
            if pfsArm.pfsConfig is not None and pfsArm.pfsConfig.fiberId is not None:
                spectrum.fiberId = pfsArm.pfsConfig.fiberId[ii]
            self.add(spectrum)

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

        Note that the fiberIds cannot be preserved without also persisting the
        associated ``pfsConfig``, and the reference lines cannot be persisted
        at all.

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
        box : `lsst.geom.Box2I`
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

    def plot(self, numRows=3, filename=None):
        """Plot the spectra

        Parameters
        ----------
        numRows : `int`
            Number of row panels over which to plot the spectra.
        filename : `str`, optional
            Name of file to which to write the plot. If a ``filename`` is
            specified, the matplotlib `figure` will be closed.

        Returns
        -------
        figure : `matplotlib.figure`
            Figure on which we plotted.
        axes : `list` of `matplotlib.Axes`
            Axes on which we plotted.
        """
        import matplotlib.cm
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(numRows)

        minWavelength = np.min([ss.wavelength for ss in self])
        maxWavelength = np.max([ss.wavelength for ss in self])
        if minWavelength == 0.0 and maxWavelength == 0.0:
            # No wavelength calibration; plot by pixel row
            minWavelength = 0.0
            maxWavelength = self.getLength()

        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(self)))

        for spectrum, cc in zip(self, colors):
            useWavelength = spectrum.getWavelength()
            if np.all(useWavelength == 0.0):
                useWavelength = np.arange(len(spectrum), dtype=np.float32)
            division = np.searchsorted(useWavelength,
                                       np.linspace(minWavelength, maxWavelength, numRows + 1)[1:-1])
            spectrum.plotDivided(axes, division, doBackground=False, doReferenceLines=False,
                                 fluxStyle=dict(ls="solid", color=cc))

        if filename is not None:
            figure.savefig(filename, bbox_inches='tight')
            plt.close(figure)
        return figure, axes
