import numpy as np
import astropy.io.fits as pyfits

import lsst.afw.fits
import lsst.geom

from lsst.utils import continueClass
from lsst.pex.config import Config, ListField

from .detectorMap import DetectorMap

__all__ = ["DetectorMap", "SlitOffsetsConfig"]


@continueClass  # noqa: F811 (redefinition)
class DetectorMap:

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
        with pyfits.open(pathName) as fd:
            pdu = fd[0]
            minX = pdu.header['MINX']
            minY = pdu.header['MINY']
            maxX = pdu.header['MAXX']
            maxY = pdu.header['MAXY']

            bbox = lsst.geom.BoxI(lsst.geom.PointI(minX, minY), lsst.geom.PointI(maxX, maxY))

            hdu = fd["FIBERID"]
            fiberIds = hdu.data
            fiberIds = fiberIds.astype(np.int32)   # why is this astype needed? BITPIX=32, no BZERO/BSCALE

            hdu = fd["SLITOFF"]
            slitOffsets = hdu.data.astype(np.float32)

            centerData = fd["CENTER"].data
            wavelengthData = fd["WAVELENGTH"].data

        # array.astype() required to force byte swapping (dtype('>f4') --> np.float32)
        # otherwise pybind doesn't recognise them as the proper type.
        numFibers = len(fiberIds)
        centerKnots = [centerData["knot"][centerData["index"] == ii].astype(np.float32) for
                       ii in range(numFibers)]
        centerValues = [centerData["value"][centerData["index"] == ii].astype(np.float32) for
                        ii in range(numFibers)]
        wavelengthKnots = [wavelengthData["knot"][wavelengthData["index"] == ii].astype(np.float32)
                           for ii in range(numFibers)]
        wavelengthValues = [wavelengthData["value"][wavelengthData["index"] == ii].astype(np.float32)
                            for ii in range(numFibers)]

        metadata = lsst.afw.fits.readMetadata(pathName, hdu=0, strip=True)
        visitInfo = lsst.afw.image.VisitInfo(metadata)
        lsst.afw.image.stripVisitInfoKeywords(metadata)

        return cls(bbox, fiberIds, centerKnots, centerValues, wavelengthKnots, wavelengthValues,
                   slitOffsets, visitInfo, metadata)

    def writeFits(self, pathName, flags=None):
        """Read DetectorMap from FITS

        Parameters
        ----------
        pathName : `str`
            Path to file from which to read.
        flags : `int`, optional
            Flags for reading; unused in this implementation.

        Raises
        ------
        NotImplementedError
            If ``flags`` are requested.
        """
        if flags is not None:
            raise NotImplementedError("flags is not used")
        #
        # Unpack detectorMap into python objects
        #
        bbox = self.getBBox()
        fiberIds = np.array(self.getFiberIds(), dtype=np.int32)
        slitOffsets = self.getSlitOffsets()

        numFibers = len(fiberIds)
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
        hdus = pyfits.HDUList()

        hdr = pyfits.Header()
        hdr["MINX"] = bbox.getMinX()
        hdr["MINY"] = bbox.getMinY()
        hdr["MAXX"] = bbox.getMaxX()
        hdr["MAXY"] = bbox.getMaxY()
        hdr["OBSTYPE"] = 'detectormap'
        date = self.getVisitInfo().getDate()
        hdr["HIERARCH calibDate"] = date.toPython(date.UTC).strftime("%Y-%m-%d")
        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)
        for key in metadata.names():
            hdr[key] = metadata.get(key)

        phu = pyfits.PrimaryHDU(header=hdr)
        hdus.append(phu)

        hdu = pyfits.ImageHDU(fiberIds, name="FIBERID")
        hdu.header["INHERIT"] = True
        hdus.append(hdu)

        hdu = pyfits.ImageHDU(slitOffsets, name="SLITOFF")
        hdu.header["INHERIT"] = True
        hdus.append(hdu)

        hdu = pyfits.BinTableHDU.from_columns([
            pyfits.Column(name="index", format="K", array=centerIndex),
            pyfits.Column(name="knot", format="E", array=centerKnots),
            pyfits.Column(name="value", format="E", array=centerValues),
        ], name="CENTER")
        hdu.header["INHERIT"] = True
        hdus.append(hdu)

        hdu = pyfits.BinTableHDU.from_columns([
            pyfits.Column(name="index", format="K", array=wavelengthIndex),
            pyfits.Column(name="knot", format="E", array=wavelengthKnots),
            pyfits.Column(name="value", format="E", array=wavelengthValues),
        ], name="WAVELENGTH")
        hdu.header["INHERIT"] = True
        hdus.append(hdu)

        # clobber=True in writeto prints a message, so use open instead
        with open(pathName, "wb") as fd:
            hdus.writeto(fd)

    def display(self, display, fiberId, wavelengths):
        """Plot wavelengths on an image

        Useful for visually inspecting the detectorMap on an arc image.

        Parameters
        ----------
        display : `lsst.afw.display.Display`
            Display on which to plot.
        fiberId : iterable of `int`
            Fiber identifiers to plot.
        wavelengths : iterable of `float`
            Wavelengths to plot.
        """
        minWl = min(array.min() for array in self.getWavelength())
        maxWl = max(array.max() for array in self.getWavelength())
        wavelengths = sorted([wl for wl in wavelengths if wl > minWl and wl < maxWl])

        with display.Buffering():
            for fiberId in fiberId:
                points = [self.findPoint(fiberId, wl) for wl in wavelengths]
                for xx, yy in points:
                    display.dot("o", xx, yy, size=5)
                display.line(points)


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
