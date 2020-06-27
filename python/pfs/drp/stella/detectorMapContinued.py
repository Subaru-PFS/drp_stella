import os
import io
import numpy as np
import astropy.io.fits

import lsst.afw.fits
import lsst.geom

from lsst.afw.display import Display
from lsst.utils import continueClass, getPackageDir
from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import CmdLineTask, ArgumentParser

from pfs.datamodel import FiberStatus
from lsst.obs.pfs.utils import getLampElements
from .utils import readLineListFile

from .detectorMap import DetectorMap

__all__ = ["DetectorMap", "SlitOffsetsConfig"]


@continueClass  # noqa: F811 (redefinition)
class DetectorMap:
    @classmethod
    def fromFits(cls, fits):
        """Read DetectorMap from open FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file.

        Returns
        -------
        out : `pfs.drp.stella.DetectorMap`
            DetectorMap read from FITS file.
        """
        pdu = fits[0]
        minX = pdu.header['MINX']
        minY = pdu.header['MINY']
        maxX = pdu.header['MAXX']
        maxY = pdu.header['MAXY']

        bbox = lsst.geom.BoxI(lsst.geom.PointI(minX, minY), lsst.geom.PointI(maxX, maxY))

        hdu = fits["FIBERID"]
        fiberId = hdu.data
        fiberId = fiberId.astype(np.int32)   # astype() forces machine-native byte order

        hdu = fits["SLITOFF"]
        slitOffsets = hdu.data.astype(np.float32)

        # array.astype() required to force byte swapping (dtype('>f4') --> np.float32)
        # otherwise pybind doesn't recognise them as the proper type.
        numFibers = len(fiberId)
        centerIndexData = fits["CENTER"].data["index"]
        centerKnotsData = fits["CENTER"].data["knot"].astype(np.float32)
        centerValuesData = fits["CENTER"].data["value"].astype(np.float32)
        centerKnots = []
        centerValues = []
        for ii in range(numFibers):
            select = centerIndexData == ii
            centerKnots.append(centerKnotsData[select])
            centerValues.append(centerValuesData[select])

        wavelengthIndexData = fits["WAVELENGTH"].data["index"]
        wavelengthKnotsData = fits["WAVELENGTH"].data["knot"].astype(np.float32)
        wavelengthValuesData = fits["WAVELENGTH"].data["value"].astype(np.float32)
        wavelengthKnots = []
        wavelengthValues = []
        for ii in range(numFibers):
            select = wavelengthIndexData == ii
            wavelengthKnots.append(wavelengthKnotsData[select])
            wavelengthValues.append(wavelengthValuesData[select])

        # Read the primary header with lsst.afw.fits
        # This requires writing the FITS file into memory and reading it from there
        buffer = io.BytesIO()
        fits.writeto(buffer)
        ss = buffer.getvalue()
        size = len(ss)
        ff = lsst.afw.fits.MemFileManager(size)
        ff.setData(ss, size)
        metadata = ff.readMetadata(0)

        visitInfo = lsst.afw.image.VisitInfo(metadata)
        lsst.afw.image.stripVisitInfoKeywords(metadata)

        return cls(bbox, fiberId, centerKnots, centerValues, wavelengthKnots, wavelengthValues,
                   slitOffsets, visitInfo, metadata)

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

    def toFits(self):
        """Write DetectorMap to FITS

        Returns
        -------
        hdus : `astropy.io.fits.HDUList`
            FITS file.
        """
        #
        # Unpack detectorMap into python objects
        #
        bbox = self.getBBox()
        fiberId = np.array(self.getFiberId(), dtype=np.int32)
        slitOffsets = self.getSlitOffsets()

        numFibers = len(fiberId)
        centerKnots = [self.getCenterSpline(ii).getX() for ii in range(numFibers)]
        centerValues = [self.getCenterSpline(ii).getY() for ii in range(numFibers)]
        wavelengthKnots = [self.getWavelengthSpline(ii).getX() for ii in range(numFibers)]
        wavelengthValues = [self.getWavelengthSpline(ii).getY() for ii in range(numFibers)]

        centerIndex = np.array(sum(([ii]*len(vv) for ii, vv in enumerate(centerKnots)), []))
        centerKnots = np.concatenate(centerKnots)
        centerValues = np.concatenate(centerValues)
        wavelengthIndex = np.array(sum(([ii]*len(vv) for ii, vv in enumerate(wavelengthKnots)), []))
        wavelengthKnots = np.concatenate(wavelengthKnots)
        wavelengthValues = np.concatenate(wavelengthValues)

        #
        # OK, we've unpacked the DetectorMap; time to write the contents to disk
        #
        hdus = astropy.io.fits.HDUList()

        hdr = astropy.io.fits.Header()
        hdr["MINX"] = bbox.getMinX()
        hdr["MINY"] = bbox.getMinY()
        hdr["MAXX"] = bbox.getMaxX()
        hdr["MAXY"] = bbox.getMaxY()
        hdr["OBSTYPE"] = 'detectorMap'
        date = self.getVisitInfo().getDate()
        hdr["HIERARCH calibDate"] = date.toPython(date.UTC).strftime("%Y-%m-%d")
        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)
        for key in metadata.names():
            hdr[key] = metadata.get(key)

        phu = astropy.io.fits.PrimaryHDU(header=hdr)
        hdus.append(phu)

        hdu = astropy.io.fits.ImageHDU(fiberId, name="FIBERID")
        hdu.header["INHERIT"] = True
        hdus.append(hdu)

        hdu = astropy.io.fits.ImageHDU(slitOffsets, name="SLITOFF")
        hdu.header["INHERIT"] = True
        hdus.append(hdu)

        hdu = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="index", format="K", array=centerIndex),
            astropy.io.fits.Column(name="knot", format="E", array=centerKnots),
            astropy.io.fits.Column(name="value", format="E", array=centerValues),
        ], name="CENTER")
        hdu.header["INHERIT"] = True
        hdus.append(hdu)

        hdu = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="index", format="K", array=wavelengthIndex),
            astropy.io.fits.Column(name="knot", format="E", array=wavelengthKnots),
            astropy.io.fits.Column(name="value", format="E", array=wavelengthValues),
        ], name="WAVELENGTH")
        hdu.header["INHERIT"] = True
        hdus.append(hdu)
        return hdus

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

    def display(self, display, fiberId=None, wavelengths=None):
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
        """
        if wavelengths:
            minWl = min(array.min() for array in self.getWavelength())
            maxWl = max(array.max() for array in self.getWavelength())
            wavelengths = sorted([wl for wl in wavelengths if wl > minWl and wl < maxWl])
        if fiberId is None:
            fiberId = self.fiberId

        with display.Buffering():
            for fiberId in fiberId:
                xCenter = self.getXCenter(fiberId)
                points = list(zip(xCenter, np.arange(len(xCenter))))

                # Work around extremely long ds9 commands from display.line getting truncated
                for p1, p2 in zip(points[:-1], points[1:]):
                    display.line((p1, p2), ctype="green")

                if wavelengths:
                    points = [self.findPoint(fiberId, wl) for wl in wavelengths]
                    for xx, yy in points:
                        display.dot("x", xx, yy, size=5)

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

    def __eq__(self, other):
        """Test for numerical/functional equality

        We do not test that the ``metadata`` and ``visitInfo`` members match,
        as they are more difficult to compare, and don't govern the important
        mappings.
        """
        equal = self.bbox == other.bbox
        equal &= np.all(self.fiberId == other.fiberId)
        equal &= all(np.all(this == that) for this, that in zip(self.xCenter, other.xCenter))
        equal &= all(np.all(this == that) for this, that in zip(self.wavelength, other.wavelength))
        equal &= np.all(self.slitOffsets == other.slitOffsets)
        return equal


class SlitOffsetsConfig(Config):
    """Configuration of slit offsets for DetectorMap"""
    x = ListField(dtype=float, default=[], doc="Slit offsets in x for each fiber; or empty")
    y = ListField(dtype=float, default=[], doc="Slit offsets in y for each fiber; or empty")
    focus = ListField(dtype=float, default=[], doc="Slit offsets in focus for each fiber; or empty")

    @property
    def numOffsets(self):
        """The number of slit offsets"""
        if self.x:
            return len(self.x)
        if self.y:
            return len(self.y)
        if self.focus:
            return len(self.focus)
        return 0

    def validate(self):
        super().validate()
        numOffsets = self.numOffsets
        if self.x and len(self.x) != numOffsets:
            raise ValueError("Inconsistent number of slit offsets in x: %d vs %d" %
                             (len(self.x), numOffsets))
        if self.y and len(self.y) != numOffsets:
            raise ValueError("Inconsistent number of slit offsets in y: %d vs %d" %
                             (len(self.y), numOffsets))
        if self.focus and len(self.focus) != numOffsets:
            raise ValueError("Inconsistent number of slit offsets in focus: %d vs %d" %
                             (len(self.focus), numOffsets))

    def apply(self, detectorMap, log=None):
        """Apply slit offsets to detectorMap

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            DetectorMap to which to apply slit offsets.
        log : `lsst.log.Log`, or ``None``
            Optional logger for reporting the application of slit offsets.
        """
        numOffsets = self.numOffsets
        if numOffsets == 0:
            return  # Nothing to do
        if len(detectorMap) != self.numOffsets:
            raise ValueError("Number of offsets (%d) doesn't match number of fibers (%d)" %
                             (numOffsets, len(detectorMap)))
        slitOffsets = detectorMap.getSlitOffsets()
        if log is not None:
            which = []
            if self.x:
                which += ["X"]
            if self.y:
                which += ["Y"]
            if self.focus:
                which += ["FOCUS"]
            log.info("Applying slit offsets in %s to detectorMap" %
                     (",".join(which),))
        if self.x:
            slitOffsets[DetectorMap.DX][:] += np.array(self.x)
        if self.y:
            slitOffsets[DetectorMap.DY][:] += np.array(self.y)
        if self.focus:
            slitOffsets[DetectorMap.DFOCUS][:] += np.array(self.focus)
        detectorMap.setSlitOffsets(slitOffsets)


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
