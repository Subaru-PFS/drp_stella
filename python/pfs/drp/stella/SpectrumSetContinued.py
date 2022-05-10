import os
import re
from typing import Callable, Dict, Iterator, Optional, Tuple, Union, TYPE_CHECKING
from types import SimpleNamespace
import numpy as np

from lsst.utils import continueClass
from lsst.geom import Box2I
from lsst.afw.image import ImageF, PARENT

from .datamodel import PfsArm
from pfs.datamodel.masks import MaskHelper
from pfs.datamodel import Identity
from .SpectrumContinued import Spectrum
from .SpectrumSet import SpectrumSet
from .utils import getPfsVersions

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes, Figure
    from .FiberTraceSetContinued import FiberTraceSet


__all__ = ["SpectrumSet"]


@continueClass  # noqa: F811 (redefinition)
class SpectrumSet:  # type: ignore  # noqa: F811 (redefinition)
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

    # Types for interfaces defined in C++ pybind layer; useful for typing in this file
    # Better types provided in stub file.
    __init__: Callable[["SpectrumSet", int, Optional[int]], None]  # type: ignore
    size: Callable[["SpectrumSet"], int]
    reserve: Callable[["SpectrumSet", int], None]
    add: Callable[["SpectrumSet", "Spectrum"], None]
    getLength: Callable[["SpectrumSet"], int]
    getAllFiberIds: Callable[["SpectrumSet"], np.ndarray]
    getAllFluxes: Callable[["SpectrumSet"], np.ndarray]
    getAllWavelengths: Callable[["SpectrumSet"], np.ndarray]
    getAllMasks: Callable[["SpectrumSet"], np.ndarray]
    getAllCovariances: Callable[["SpectrumSet"], np.ndarray]
    getAllBackgrounds: Callable[["SpectrumSet"], np.ndarray]
    getAllNormalizations: Callable[["SpectrumSet"], np.ndarray]
    __len__: Callable[["SpectrumSet"], int]
    __getitem__: Callable[["SpectrumSet", int], "Spectrum"]
    __setitem__: Callable[["SpectrumSet", int, "Spectrum"], None]
    __iter__: Callable[["SpectrumSet"], Iterator["SpectrumSet"]]

    def toPfsArm(self, dataId: Dict[str, Union[str, int, float]]) -> PfsArm:
        """Convert to a `pfs.datamodel.PfsArm`

        Parameters
        ----------
        dataId : `dict`
            Data identifier, which is expected to contain:

            - ``visit`` (`int`): exposure number
            - ``spectrograph`` (`int`): spectrograph number
            - ``arm`` (`str`: "b", "r", "m" or "n"): spectrograph arm

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
        covar = np.zeros((numSpectra, 3, self.getLength()))
        for ii, ss in enumerate(self):
            covar[ii] = ss.getCovariance()
        metadata = getPfsVersions()
        identity = Identity.fromDict(dataId)
        return PfsArm(identity, fiberIds, wavelength, self.getAllFluxes(), self.getAllMasks(),
                      self.getAllBackgrounds(), self.getAllNormalizations(), covar, flags, metadata)

    @classmethod
    def fromPfsArm(cls, pfsArm: PfsArm) -> "SpectrumSet":
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
            spectrum.spectrum[:] = pfsArm.flux[ii]
            spectrum.mask.array[:] = pfsArm.mask[ii]
            spectrum.background[:] = pfsArm.sky[ii]
            spectrum.norm[:] = pfsArm.norm[ii]
            spectrum.covariance[:] = pfsArm.covar[ii]
            spectrum.wavelength[:] = pfsArm.wavelength[ii]
            self.add(spectrum)

        return self

    @classmethod
    def _parsePath(cls, path: str, hdu: Optional[int] = None, flags: Optional[int] = None) -> SimpleNamespace:
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
        return SimpleNamespace(
            dirName=dirName, fileName=fileName, visit=visit, arm=arm, spectrograph=spectrograph, dataId=dataId
        )

    def writeFits(self, filename: str):
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
        pfsArm = self.toPfsArm(parsed.dataId)
        pfsArm.writeFits(filename)

    @classmethod
    def readFits(cls, *args, **kwargs) -> "SpectrumSet":
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

    def makeImage(self, box: Box2I, fiberTraces: "FiberTraceSet") -> ImageF:
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
        image = ImageF(box)
        image.set(0.0)

        specDict = {fiberId: spec for fiberId, spec in zip(self.getAllFiberIds(), self)}
        for ft in fiberTraces:
            spec = specDict.get(ft.fiberId, None)
            if spec is None:
                continue
            fiberImage = ft.constructImage(spec)
            image[fiberImage.getBBox(), PARENT] += fiberImage

        return image

    def plot(self, numRows: int = 3, filename: str = None) -> Tuple["Figure", "Axes"]:
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
            spectrum.plotDivided(axes, division, doBackground=False, fluxStyle=dict(ls="solid", color=cc))

        if filename is not None:
            figure.savefig(filename, bbox_inches='tight')
            plt.close(figure)
        return figure, axes
