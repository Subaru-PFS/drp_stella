import io
import numpy as np
import astropy.io.fits

from lsst.utils import continueClass
from lsst.afw.display import Display
from lsst.pex.config import Config, Field, ConfigurableField
from lsst.pipe.base import CmdLineTask, InputOnlyArgumentParser

import pfs.datamodel
from .readLineList import ReadLineListTask
from pfs.drp.stella.DetectorMap import DetectorMap


__all__ = ["DetectorMap", "DisplayDetectorMapTask", "DisplayDetectorMapConfig"]


@continueClass  # noqa F811: redefinition
class DetectorMap:  # noqa F811: redefinition
    """A pseudo-class, containing factory methods and derived methods

    This class cannot be instantiated. It exists to provide factory classes the
    ability to construct a ``DetectorMap`` based on the content of the source
    data. It also provides some "inherited" methods for the "derived" classes.
    We take this approach to avoid stepping on the inheritance hierarchy of a
    pybind11 class.

    To support this interface, classes should implement the ``fromDatamodel``
    and ``toDatamodel`` methods to convert to/from the representation in
    pfs.datamodel, and be registered via ``DetectorMap.register``.
    """
    _subclasses = {}  # Registered subclasses (mapping name to class)

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
        cls._subclasses[SubClass.__name__] = SubClass
        for method in ("readFits", "fromBytes", "fromFits", "writeFits", "toBytes", "toFits",
                       "display", "__reduce__"):
            setattr(SubClass, "method", getattr(cls, method))

    @classmethod
    def fromDatamodel(cls, detectorMap):
        """Construct from subclass of pfs.datamodel.DetectorMap

        Converts from the datamodel representation to the drp_stella
        representation.

        Parameters
        ----------
        detectorMap : subclass of `pfs.datamodel.DetectorMap`
            DetectorMap to convert.

        Returns
        -------
        self : subclass of `pfs.drp.stella.DetectorMap`
            Converted detectorMap.
        """
        subclass = type(detectorMap).__name__
        return cls._subclasses[subclass].fromDatamodel(detectorMap)

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
        detMap = pfs.datamodel.PfsDetectorMap._readImpl(fits, None)
        return cls.fromDatamodel(detMap)

    def toFits(self):
        """Generate a FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file.
        """
        return self.toDatamodel()._writeImpl()

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

    def display(self, display, fiberId=None, wavelengths=None, ctype="green", plotTraces=True):
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
        plotTraces : `bool`
            Plot the traces? This is slow, but shows the position of xCenter
            as a function of row for all fibers.
        """
        if fiberId is None:
            fiberId = self.fiberId
        if wavelengths is not None:
            minWl = min(self.getWavelength(ff).min() for ff in fiberId)
            maxWl = max(self.getWavelength(ff).max() for ff in fiberId)
            wavelengths = np.array([wl for wl in set(wavelengths) if wl > minWl and wl < maxWl],
                                   dtype=float)

        with display.Buffering():
            for fiberId in set(fiberId):
                if plotTraces:
                    xCenter = self.getXCenter(fiberId)
                    points = list(zip(xCenter, np.arange(len(xCenter))))

                    # Work around extremely long ds9 commands from display.line getting truncated
                    for p1, p2 in zip(points[:-1], points[1:]):
                        display.line((p1, p2), ctype=ctype)

                if wavelengths is not None:
                    points = self.findPoint(fiberId, wavelengths)
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
    frame = Field(dtype=int, default=1, doc="Frame to use for display")
    backend = Field(dtype=str, doc="Display backend to use")
    doPlotLines = Field(dtype=bool, default=True, doc="Plot the location of lines from line list?")
    readLineList = ConfigurableField(target=ReadLineListTask, doc="Read line list")


class DisplayDetectorMapTask(CmdLineTask):
    """Display an image with the detectorMap superimposed"""
    ConfigClass = DisplayDetectorMapConfig
    _DefaultName = "displayDetectorMap"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("readLineList")

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = InputOnlyArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", datasetType="postISRCCD",
                               help="input identifiers, e.g., --id visit=123 ccd=4")
        return parser

    def runDataRef(self, dataRef):
        """Display an image with the detectorMap superimposed

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Butler data reference.
        """
        exposure = dataRef.get("postISRCCD")
        detectorMap = dataRef.get("detectorMap")
        pfsConfig = dataRef.get("pfsConfig")
        self.log.info("Displaying %s", dataRef.dataId)
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
        wavelengths = None
        if self.config.doPlotLines:
            lines = self.readLineList.run(metadata=exposure.getMetadata())
            wavelengths = [rl.wavelength for rl in lines]

        display = Display(frame=self.config.frame, backend=self.config.backend)
        display.mtv(exposure)

        goodFibers = detectorMap.fiberId[pfsConfig.selectByFiberStatus(pfs.datamodel.FiberStatus.GOOD,
                                         detectorMap.fiberId)]
        for targetType, color in ((pfs.datamodel.TargetType.SCIENCE, "GREEN"),
                                  (pfs.datamodel.TargetType.SKY, "BLUE"),
                                  (pfs.datamodel.TargetType.FLUXSTD, "YELLOW"),
                                  (pfs.datamodel.TargetType.UNASSIGNED, "MAGENTA"),
                                  (pfs.datamodel.TargetType.ENGINEERING, "CYAN"),
                                  (pfs.datamodel.TargetType.SUNSS_DIFFUSE, "BLUE"),
                                  (pfs.datamodel.TargetType.SUNSS_IMAGING, "GREEN"),
                                  (pfs.datamodel.TargetType.DCB, "GREEN"),
                                  ):
            indices = pfsConfig.selectByTargetType(targetType, goodFibers)
            if indices.size == 0:
                self.log.info("No %s fibers found", targetType)
                continue
            fiberId = goodFibers[indices]
            detectorMap.display(display, fiberId, wavelengths, color)
            self.log.info("%s fibers (%d) are shown in %s", targetType, indices.size, color)

        for fiberStatus, color in ((pfs.datamodel.FiberStatus.BROKENFIBER, "BLACK"),
                                   (pfs.datamodel.FiberStatus.BLOCKED, "BLACK"),
                                   (pfs.datamodel.FiberStatus.BLACKSPOT, "RED"),
                                   (pfs.datamodel.FiberStatus.UNILLUMINATED, "BLACK")
                                   ):
            indices = pfsConfig.selectByFiberStatus(fiberStatus, detectorMap.fiberId)
            if indices.size == 0:
                self.log.info("No %s fibers found", fiberStatus)
                continue
            self.log.info("%s fibers (%d) are shown in %s", fiberStatus, indices.size, color)
            detectorMap.display(display, detectorMap.fiberId[indices], wavelengths, color)

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None

    def writePackageVersions(self, *args, **kwargs):
        return
