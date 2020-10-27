import io
import numpy as np
import astropy.io.fits

import lsst.afw.fits
import lsst.geom

from lsst.utils import continueClass

from .GlobalDetectorMap import GlobalDetectorMap, GlobalDetectorModel, GlobalDetectorModelScaling, FiberMap
from .DetectorMapContinued import DetectorMap


__all__ = ["GlobalDetectorMap", "GlobalDetectorModel", "GlobalDetectorModelScaling", "FiberMap"]


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

        scaling = GlobalDetectorModelScaling(
            fiberPitch=header["scaling.fiberPitch"],
            dispersion=header["scaling.dispersion"],
            wavelengthCenter=header["scaling.wavelengthCenter"],
            minFiberId=header["scaling.minFiberId"],
            maxFiberId=header["scaling.maxFiberId"],
            height=header["scaling.height"],
            buffer=header["scaling.buffer"],
        )

        spatialOffsets = table["spatialOffsets"].astype(np.float32)
        spectralOffsets = table["spectralOffsets"].astype(np.float32)
        xCoeff = fits["COEFFICIENTS"].data["x"].astype(float)
        yCoeff = fits["COEFFICIENTS"].data["y"].astype(float)
        rightCcd = fits["RIGHTCCD"].data["coeff"].astype(float)

        model = GlobalDetectorModel(bbox, distortionOrder, fiberId, scaling, xCoeff, yCoeff, rightCcd,
                                    spatialOffsets, spectralOffsets)

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

        return cls(bbox, model, visitInfo, metadata)

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
        header["INHERIT"] = True

        model = self.getModel()
        scaling = model.getScaling()
        header["HIERARCH scaling.fiberPitch"] = scaling.fiberPitch
        header["HIERARCH scaling.dispersion"] = scaling.dispersion
        header["HIERARCH scaling.wavelengthCenter"] = scaling.wavelengthCenter
        header["HIERARCH scaling.minFiberId"] = scaling.minFiberId
        header["HIERARCH scaling.maxFiberId"] = scaling.maxFiberId
        header["HIERARCH scaling.height"] = scaling.height
        header["HIERARCH scaling.buffer"] = scaling.buffer

        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="fiberId", format="J", array=fiberId),
            astropy.io.fits.Column(name="spatialOffsets", format="E", array=self.getSpatialOffsets()),
            astropy.io.fits.Column(name="spectralOffsets", format="E", array=self.getSpectralOffsets()),
        ], header=header, name="FIBERS")
        fits.append(table)

        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="x", format="E", array=model.getXCoefficients()),
            astropy.io.fits.Column(name="y", format="E", array=model.getYCoefficients()),
        ], header=header, name="COEFFICIENTS")
        fits.append(table)

        table = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="coeff", format="E", array=model.getRightCcdCoefficients()),
        ], header=header, name="RIGHTCCD")
        fits.append(table)

        return fits


DetectorMap.register(GlobalDetectorMap)
