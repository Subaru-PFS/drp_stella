import os
import io
import numpy as np
import astropy.io.fits

from lsst.utils import continueClass, getPackageDir
from lsst.afw.display import Display
from lsst.pex.config import Config, Field
from lsst.pipe.base import CmdLineTask, ArgumentParser

from pfs.datamodel import FiberStatus
from lsst.obs.pfs.utils import getLampElements
from .utils import readLineListFile
from pfs.drp.stella.DetectorMap import DetectorMap


__all__ = ["DetectorMap", "DisplayDetectorMapTask", "DisplayDetectorMapConfig"]


@continueClass  # noqa F811: redefinition
class DetectorMap:
    """A pseudo-class, containing factory methods and derived methods

    This class cannot be instantiated. It exists to provide factory classes the
    ability to construct a ``DetectorMap`` based on the content of the source
    data. It also provides some "inherited" methods for the "derived" classes.
    We take this approach to avoid stepping on the inheritance hierarchy of a
    pybind11 class.

    To support this interface, classes should implement the ``canReadFits``
    and ``fromFits`` methods, and be registered via
    ``DetectorMap.register``.
    """
    _subclasses = []  # List of registered subclasses

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This is a pure-virtual base class")

    @classmethod
    def register(cls, SubClass):
        """Register a SubClass

        Parameters
        ----------
        SubClass : `type`
            Subclass of ``pfs::drp::stella::DetectorMap``.
        """
        assert SubClass not in cls._subclasses
        cls._subclasses.append(SubClass)
        for method in ("readFits", "fromBytes", "writeFits", "display", "toBytes", "__reduce__"):
            setattr(SubClass, "method", getattr(cls, method))

    @classmethod
    def fromFits(cls, fits):
        """Construct from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file.

        Returns
        -------
        self : subclass of ``pfs::drp::stella::DetectorMap``
            Instantiated subclass, read from the file.
        """
        for SubClass in cls._subclasses:
            if SubClass.canReadFits(fits):
                return SubClass.fromFits(fits)
        else:
            raise RuntimeError("Unrecognised file format")

    @classmethod
    def readFits(cls, pathName, hdu=None, flags=None):
        """Read DetectorMap from FITS

        Parameters
        ----------
        pathName : `str`
            Path to file from which to read.
        hdu : `int`, optional
            HDU from which to read; unused in this implementation.
        flags : `int`, optional
            Flags for reading; unused in this implementation.

        Raises
        ------
        NotImplementedError
            If ``hdu`` or ``flags`` are requested.

        Returns
        -------
        out : `pfs.drp.stella.DetectorMap`
            DetectorMap read from FITS file.
        """
        if hdu is not None:
            raise NotImplementedError("hdu is not used")
        if flags is not None:
            raise NotImplementedError("flags is not used")
        with astropy.io.fits.open(pathName) as fits:
            return cls.fromFits(fits)

    @classmethod
    def fromBytes(cls, string):
        """Construct from bytes

        Parameters
        ----------
        string : `bytes`
            String of bytes.

        Returns
        -------
        self : cls
            Constructed object.
        """
        with astropy.io.fits.open(io.BytesIO(string)) as fd:
            return cls.fromFits(fd)

    def writeFits(self, pathName, flags=None):
        """Write DetectorMap to FITS

        Parameters
        ----------
        pathName : `str`
            Path of file to which to write.
        flags : `int`, optional
            Flags for writing; unused in this implementation.

        Raises
        ------
        NotImplementedError
            If ``flags`` are requested.
        """
        if flags is not None:
            raise NotImplementedError("flags is not used")
        hdus = self.toFits()
        # clobber=True in writeto prints a message, so use open instead
        with open(pathName, "wb") as fd:
            hdus.writeto(fd)

    def display(self, display, fiberId=None, wavelengths=None, ctype="green"):
        """Plot wavelengths on an image

        Useful for visually inspecting the detectorMap on an arc image.

        Parameters
        ----------
        display : `lsst.afw.display.Display`
            Display on which to plot.
        fiberId : iterable of `int`, optional
            Fiber identifiers to plot.
        wavelengths : iterable of `float`, optional
            Wavelengths to plot.
        ctype : `str`
            Color for `lsst.afw.display.Display` commands.
        """
        if fiberId is None:
            fiberId = self.fiberId
        if wavelengths:
            minWl = min(self.getWavelength(ff).min() for ff in fiberId)
            maxWl = max(self.getWavelength(ff).max() for ff in fiberId)
            wavelengths = sorted([wl for wl in wavelengths if wl > minWl and wl < maxWl])

        with display.Buffering():
            for fiberId in fiberId:
                xCenter = self.getXCenter(fiberId)
                points = list(zip(xCenter, np.arange(len(xCenter))))

                # Work around extremely long ds9 commands from display.line getting truncated
                for p1, p2 in zip(points[:-1], points[1:]):
                    display.line((p1, p2), ctype=ctype)

                if wavelengths:
                    points = [self.findPoint(fiberId, wl) for wl in wavelengths]
                    for xx, yy in points:
                        display.dot("x", xx, yy, size=5, ctype=ctype)

    def toBytes(self):
        """Convert to bytes

        Returns
        -------
        string : `bytes`
            String of bytes.
        """
        fits = self.toFits()
        buffer = io.BytesIO()
        fits.writeto(buffer)
        fits.close()
        return buffer.getvalue()

    def __reduce__(self):
        """How to pickle"""
        return self.__class__.fromBytes, (self.toBytes(),)


class DisplayDetectorMapConfig(Config):
    """Configuration for DisplayDetectorMapTask"""
    backend = Field(dtype=str, default="ds9", doc="Display backend to use")
    frame = Field(dtype=int, default=1, doc="Frame to use for display")
    lineList = Field(dtype=str, doc="Line list to use for marking wavelengths",
                     default=os.path.join(getPackageDir("obs_pfs"), "pfs", "lineLists", "ArCdHgKrNeXe.txt"))
    minArcLineIntensity = Field(doc="Minimum 'NIST' intensity to use for arc lines",
                                dtype=float, default=0)


class DisplayDetectorMapTask(CmdLineTask):
    """Display an image with the detectorMap superimposed"""
    ConfigClass = DisplayDetectorMapConfig
    _DefaultName = "displayDetectorMap"

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", datasetType="calexp",
                               help="input identifiers, e.g., --id visit=123 ccd=4")
        return parser

    def runDataRef(self, dataRef):
        """Display an image with the detectorMap superimposed

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Butler data reference.
        """
        exposure = dataRef.get("calexp")
        detectorMap = dataRef.get("detectorMap")
        pfsConfig = dataRef.get("pfsConfig")
        self.run(exposure, detectorMap, pfsConfig)

    def run(self, exposure, detectorMap, pfsConfig):
        """Display an image with the detectorMap superimposed

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image to display.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping between fiberId,wavelength <--> x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Fiber configuration.
        """
        lamps = getLampElements(exposure.getMetadata())
        if not lamps:
            raise RuntimeError("No lamps found from header")
        lines = readLineListFile(self.config.lineList, lamps, minIntensity=self.config.minArcLineIntensity)

        display = Display(backend=self.config.backend, frame=self.config.frame)
        display.mtv(exposure)
        indices = pfsConfig.selectByFiberStatus(FiberStatus.GOOD)
        detectorMap.display(display, pfsConfig.fiberId[indices], [rl.wavelength for rl in lines])

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None

    def _getEupsVersionsName(self):
        return None
