import io
import numpy as np
import astropy.io.fits

import lsst.afw.fits
import lsst.geom

from lsst.utils import continueClass

from .SplinedDetectorMap import SplinedDetectorMap
from .DetectorMapContinued import DetectorMap


__all__ = ["SplinedDetectorMap"]


@continueClass  # noqa: F811 (redefinition)
class SplinedDetectorMap:
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
        if keyword not in fits[0].header:
            # Backward compatibility with the old DetectorMap
            return True
        return fits[0].header[keyword] == "SplinedDetectorMap"

    @classmethod
    def fromFits(cls, fits):
        """Read SplinedDetectorMap from open FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file.

        Returns
        -------
        out : `pfs.drp.stella.SplinedDetectorMap`
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
                   slitOffsets[0], slitOffsets[1], visitInfo, metadata)

    def toFits(self):
        """Write SplinedDetectorMap to FITS

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
        numFibers = len(fiberId)

        slitOffsets = np.zeros((3, numFibers))
        slitOffsets[0] = self.getSpatialOffsets()
        slitOffsets[1] = self.getSpectralOffsets()
        # slitOffsets[2] is focus (backward compatibility), but we're not using that

        centerKnots = [self.getXCenterSpline(ff).getX() for ff in fiberId]
        centerValues = [self.getXCenterSpline(ff).getY() for ff in fiberId]
        wavelengthKnots = [self.getWavelengthSpline(ff).getX() for ff in fiberId]
        wavelengthValues = [self.getWavelengthSpline(ff).getY() for ff in fiberId]

        centerIndex = np.array(sum(([ii]*len(vv) for ii, vv in enumerate(centerKnots)), []))
        centerKnots = np.concatenate(centerKnots)
        centerValues = np.concatenate(centerValues)
        wavelengthIndex = np.array(sum(([ii]*len(vv) for ii, vv in enumerate(wavelengthKnots)), []))
        wavelengthKnots = np.concatenate(wavelengthKnots)
        wavelengthValues = np.concatenate(wavelengthValues)

        #
        # OK, we've unpacked the DetectorMap; time to write the contents to disk
        #
        fits = astropy.io.fits.HDUList()

        header = astropy.io.fits.Header()
        header["MINX"] = bbox.getMinX()
        header["MINY"] = bbox.getMinY()
        header["MAXX"] = bbox.getMaxX()
        header["MAXY"] = bbox.getMaxY()
        header["HIERARCH pfs_detectorMap_class"] = "SplinedDetectorMap"
        header["OBSTYPE"] = 'detectorMap'
        date = self.getVisitInfo().getDate()
        header["HIERARCH calibDate"] = date.toPython(date.UTC).strftime("%Y-%m-%d")
        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)
        for key in metadata.names():
            header[key] = metadata.get(key)

        phu = astropy.io.fits.PrimaryHDU(header=header)
        fits.append(phu)

        hdu = astropy.io.fits.ImageHDU(fiberId, name="FIBERID")
        hdu.header["INHERIT"] = True
        fits.append(hdu)

        hdu = astropy.io.fits.ImageHDU(slitOffsets, name="SLITOFF")
        hdu.header["INHERIT"] = True
        fits.append(hdu)

        hdu = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="index", format="K", array=centerIndex),
            astropy.io.fits.Column(name="knot", format="E", array=centerKnots),
            astropy.io.fits.Column(name="value", format="E", array=centerValues),
        ], name="CENTER")
        hdu.header["INHERIT"] = True
        fits.append(hdu)

        hdu = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="index", format="K", array=wavelengthIndex),
            astropy.io.fits.Column(name="knot", format="E", array=wavelengthKnots),
            astropy.io.fits.Column(name="value", format="E", array=wavelengthValues),
        ], name="WAVELENGTH")
        hdu.header["INHERIT"] = True
        fits.append(hdu)
        return fits


DetectorMap.register(SplinedDetectorMap)
