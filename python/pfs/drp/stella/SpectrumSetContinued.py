import os
import re
import numpy as np

from lsst.utils import continueClass
from lsst.pipe.base import Struct
import lsst.afw.image as afwImage

from .datamodel import PfsArm
from pfs.datamodel.masks import MaskHelper
from pfs.datamodel import Identity
from .SpectrumContinued import Spectrum
from .SpectrumSet import SpectrumSet
from .utils import getPfsVersions

__all__ = ["SpectrumSet"]


@continueClass  # noqa: F811 (redefinition)
class SpectrumSet:  # noqa: F811 (redefinition)
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

    def toPfsArm(self, identity: Identity):
        """Convert to a `pfs.datamodel.PfsArm`

        Parameters
        ----------
        identity : `pfs.datamodel.Identity`
            Identity of the spectra.

        Returns
        -------
        spectra : ``pfs.datamodel.PfsArm``
            Spectra in standard PFS datamodel form.
        """
        fiberIds = np.array(self.getAllFiberIds())
        wavelength = self.getAllWavelengths()
        wavelength[wavelength == 0] = np.nan
        flags = MaskHelper(**self[0].mask.getMaskPlaneDict())
        numSpectra = len(self)
        covar = self.getAllVariances().reshape(numSpectra, 1, -1)
        metadata = getPfsVersions()
        notes = PfsArm.NotesClass(**{
            col.name: np.array([self[ii].notes.get(col.name, col.default) for ii in range(numSpectra)])
            for col in PfsArm.NotesClass.schema
        })
        flux = self.getAllFluxes()
        background = np.zeros_like(flux)
        return PfsArm(identity, fiberIds, wavelength, flux, self.getAllMasks(),
                      background, self.getAllNormalizations(), covar, flags, metadata, notes)

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
        pfsArm.validate()
        numFibers = len(pfsArm.flux)
        if numFibers == 0:
            raise RuntimeError("Unable to construct SpectrumSet from empty PfsArm")
        length = len(pfsArm.flux[0])
        self = cls(length)
        for ii in range(numFibers):
            spectrum = Spectrum(length)
            spectrum.fiberId = pfsArm.fiberId[ii]
            spectrum.flux[:] = pfsArm.flux[ii]
            spectrum.norm[:] = pfsArm.norm[ii]
            spectrum.variance[:] = pfsArm.variance[ii]
            spectrum.wavelength[:] = pfsArm.wavelength[ii]

            # We need to take care that the mask planes are copied across, not the values
            spectrum.mask.array[:] = 0
            for name in pfsArm.flags.flags:
                source = pfsArm.flags.get(name)
                target = 2**spectrum.mask.addMaskPlane(name)
                select = (pfsArm.mask[ii] & source) != 0
                spectrum.mask.array[0, select] |= target

            # Need to convert notes to pure-python types so PropertySet will recognise them
            notesTypes = {
                str: str,
                bool: bool,
                int: int,
                float: float,
                np.int16: int,
                np.int32: int,
                np.int64: int,
                np.float32: float,
                np.float64: float,
            }
            columns = {
                col.name: getattr(pfsArm.notes, col.name).astype(notesTypes[col.dtype])
                for col in pfsArm.notes.schema
            }
            spectrum.notes.update({col.name: columns[col.name][ii] for col in pfsArm.notes.schema})
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

    def writeFits(self, filename):
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
        parsed = self._parsePath(filename)
        pfsArm = self.toPfsArm(Identity.fromDict(parsed.dataId))
        pfsArm.writeFits(filename)

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
        identity = Identity.fromDict(parsed.dataId)
        pfsArm = PfsArm.read(identity, dirName=parsed.dirName)
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

        specDict = {fiberId: spec for fiberId, spec in zip(self.getAllFiberIds(), self)}
        for ft in fiberTraces:
            spec = specDict.get(ft.fiberId, None)
            if spec is None:
                continue
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
                useWavelength = np.arange(len(spectrum), dtype=float)
            division = np.searchsorted(useWavelength,
                                       np.linspace(minWavelength, maxWavelength, numRows + 1)[1:-1])
            spectrum.plotDivided(axes, division, fluxStyle=dict(ls="solid", color=cc))

        if filename is not None:
            figure.savefig(filename, bbox_inches='tight')
            plt.close(figure)
        return figure, axes
