import io
import numpy as np
import astropy.io.fits

import lsst.afw.fits
import lsst.geom

from lsst.utils import continueClass

from .GlobalDetectorMap import GlobalDetectorMap, GlobalDetectorModel
from .DetectorMapContinued import DetectorMap


__all__ = ["GlobalDetectorMap", "GlobalDetectorModel"]


@continueClass  # noqa: F811 (redefinition)
class GlobalDetectorMap:
    @classmethod
    def canReadFits(cls, fits):
        """Return whether this class can read the FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file.

        Returns
        -------
        canRead : `bool`
            Whether we can read the file.
        """
        keyword = "HIERARCH pfs_detectorMap_class"
        return keyword in fits[0].header and fits[0].header[keyword] == "GlobalDetectorMap"

    @classmethod
    def fromFits(cls, fits):
        """Read GlobalDetectorMap from open FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file.

        Returns
        -------
        out : `pfs.drp.stella.GlobalDetectorMap`
            DetectorMap read from FITS file.
        """
        header = fits["FIBERS"].header
        table = fits["FIBERS"].data

        bbox = lsst.geom.BoxI(lsst.geom.PointI(header['MINX'], header['MINY']),
                              lsst.geom.PointI(header['MAXX'], header['MAXY']))

        # array.astype() required to force byte swapping (e.g., dtype('>f4') --> np.float32)
        # otherwise pybind doesn't recognise them as the proper type.
        fiberId = table["fiberId"].astype(np.int32)
        distortionOrder = header["ORDER"]
        dualDetector = header["DUALDET"]
        spatialOffsets = table["spatialOffsets"].astype(np.float32)
        spectralOffsets = table["spectralOffsets"].astype(np.float32)
        parameters = fits["PARAMETERS"].data["parameters"].astype(float)

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

        return cls(bbox, fiberId, distortionOrder, dualDetector, parameters,
                   spatialOffsets, spectralOffsets, visitInfo, metadata)

    def toFits(self):
        """Write DetectorMap to FITS

        Returns
        -------
        hdus : `astropy.io.fits.HDUList`
            FITS file.
        """
        bbox = self.getBBox()
        fiberId = self.getFiberId()

        fits = astropy.io.fits.HDUList()
        header = astropy.io.fits.Header()
        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)
        for key in metadata.names():
            header[key if len(key) <= 8 else "HIERARCH " + key] = metadata.get(key)

        header["HIERARCH pfs_detectorMap_class"] = "GlobalDetectorMap"
        header["OBSTYPE"] = 'detectorMap'
        date = self.getVisitInfo().getDate()
        header["HIERARCH calibDate"] = date.toPython(date.UTC).strftime("%Y-%m-%d")

        phu = astropy.io.fits.PrimaryHDU(header=header)
        fits.append(phu)

        header = astropy.io.fits.Header()
        header["MINX"] = bbox.getMinX()
        header["MINY"] = bbox.getMinY()
        header["MAXX"] = bbox.getMaxX()
        header["MAXY"] = bbox.getMaxY()
        header["ORDER"] = self.getDistortionOrder()
        header["DUALDET"] = self.getDualDetector()
        header["INHERIT"] = True

        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="fiberId", format="J", array=fiberId),
            astropy.io.fits.Column(name="spatialOffsets", format="E", array=self.getSpatialOffsets()),
            astropy.io.fits.Column(name="spectralOffsets", format="E", array=self.getSpectralOffsets()),
        ], header=header, name="FIBERS")
        fits.append(table)

        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="parameters", format="E", array=self.getParameters()),
        ], header=header, name="PARAMETERS")
        fits.append(table)

        return fits


DetectorMap.register(GlobalDetectorMap)