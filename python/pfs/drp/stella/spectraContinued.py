import os
import re

import numpy as np

from lsst.utils import continueClass
from lsst.daf.base import PropertySet
from lsst.pipe.base import Struct

import lsst.afw.image as afwImage

from pfs.datamodel.pfsArm import PfsArm, PfsConfig
from pfs.drp.stella import ReferenceLine

from .spectra import SpectrumSet, Spectrum

__all__ = ["Spectrum", "SpectrumSet"]

BAD_REFERENCE = (ReferenceLine.MISIDENTIFIED | ReferenceLine.CLIPPED | ReferenceLine.SATURATED |
                 ReferenceLine.INTERPOLATED | ReferenceLine.CR)


def plotReferenceLines(ax, referenceLines, flux, wavelength, badReference=BAD_REFERENCE):
    """Plot reference lines on a spectrum

    Parameters
    ----------
    ax : `matplotlib.Axes`
        Axes on which to plot.
    referenceLines : `list` of `pfs.drp.stella.ReferenceLine`
        Reference lines to plot.
    flux : `numpy.ndarray`
        Flux vector, for setting height of reference line indicators.
    wavelength : `numpy.ndarray`
        Wavelength vector, for setting height of reference line indicators.
    badReference : `pfs.drp.stella.ReferenceLine.Status`
        Bit mask for identifying bad reference lines. Bad reference lines
        are plotted in red instead of black.
    """
    minWavelength = wavelength[0]
    maxWavelength = wavelength[-1]
    isGood = np.isfinite(flux)
    ff = flux[isGood]
    wl = wavelength[isGood]
    vertical = np.max(ff)
    for rl in referenceLines:
        xx = rl.wavelength
        if xx < minWavelength or xx > maxWavelength:
            continue
        style = "dotted" if rl.status & ReferenceLine.RESERVED > 0 else "solid"
        color = "red" if rl.status & badReference > 0 else "black"

        index = int(np.searchsorted(wl, xx))
        yy = np.max(ff[max(0, index - 2):min(len(ff) - 1, index + 2 + 1)])
        ax.plot((xx, xx), (yy + 0.10*vertical, yy + 0.20*vertical), ls=style, color=color)
        ax.text(xx, yy + 0.25*vertical, rl.description, color=color, ha='center')


@continueClass  # noqa: F811 (redefinition)
class Spectrum:
    """Flux as a function of wavelength"""
    def plot(self, numRows=3, doBackground=False, doReferenceLines=False, badReference=BAD_REFERENCE,
             filename=None):
        """Plot spectrum

        Parameters
        ----------
        numRows : `int`
            Number of row panels over which to plot the spectrum.
        doBackground : `bool`, optional
            Plot the background values in addition to the flux values?
        doReferenceLines : `bool`, optional
            Plot the reference lines?
        badReference : `pfs.drp.stella.ReferenceLine.Status`
            Bit mask for identifying bad reference lines. Bad reference lines
            are plotted in red instead of black.
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
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(numRows)

        division = np.linspace(0, len(self), numRows + 1, dtype=int)[1:-1]
        self.plotDivided(axes, division, doBackground=doBackground, doReferenceLines=doReferenceLines,
                         badReference=badReference)

        if filename is not None:
            figure.savefig(filename, bbox_inches='tight')
            plt.close(figure)
        return figure, axes

    def plotDivided(self, axes, division, doBackground=False, doReferenceLines=False,
                    badReference=BAD_REFERENCE, fluxStyle=None, backgroundStyle=None):
        """Plot spectrum that has been divided into parts

        This is intended as the implementation of the guts of the ``plot``
        method, but it may be of use elsewhere.

        Parameters
        ----------
        axes : `list` of `matplotlib.Axes`
            Axes on which to plot.
        division : `numpy.ndarray`
            Array of indices at which to divide the spectrum. This should be one
            element shorter than the list of ``axes``.
        doBackground : `bool`, optional
            Plot the background values in addition to the flux values?
        doReferenceLines : `bool`, optional
            Plot the reference lines?
        badReference : `pfs.drp.stella.ReferenceLine.Status`
            Bit mask for identifying bad reference lines. Bad reference lines
            are plotted in red instead of black.
        fluxStyle : `dict`
            Arguments for the ``matplotlib.Axes.plot`` method when plotting the
            flux vector.
        backgroundStyle : `dict`
            Arguments for the ``matplotlib.Axes.plot`` method when plotting the
            background vector.
        """
        # subplots(1) returns an Axes not embedded in a list
        try:
            axes[0]
        except TypeError:
            axes = [axes]

        assert len(axes) == len(division) + 1, ("Axes/division length mismatch: %d vs %d" %
                                                (len(axes), len(division)))
        if fluxStyle is None:
            fluxStyle = dict(ls="solid", color="black")
        if backgroundStyle is None:
            backgroundStyle = dict(ls="solid", color="blue")

        useWavelength = self.getWavelength()
        if np.all(useWavelength == 0.0):
            useWavelength = np.arange(len(self), dtype=np.float32)
        wavelength = np.split(useWavelength, division)
        flux = np.split(self.getSpectrum(), division)
        if doBackground:
            background = np.split(self.getBackground(), division)

        for ii, ax in enumerate(axes):
            ax.plot(wavelength[ii], flux[ii], **fluxStyle)
            if doBackground:
                ax.plot(wavelength[ii], background[ii], **backgroundStyle)
            ax.set_ylim(bottom=0.0)
            if doReferenceLines:
                plotReferenceLines(ax, self.getReferenceLines(), flux[ii], wavelength[ii], badReference)


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
        pfsConfig = PfsConfig(fiberId=np.array([ss.fiberId for ss in self]),
                              ra=np.zeros(len(self)), dec=np.zeros(len(self)))
        pfsArm = PfsArm(visit, spectrograph, arm, pfsConfigId=pfsConfigId, pfsConfig=pfsConfig)
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
