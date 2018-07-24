import numpy as np
import pyfits

import lsst.afw.geom as afwGeom

from lsst.utils import continueClass
from lsst.daf.base import PropertyList

from .detectorMap import DetectorMap

__all__ = ["DetectorMap"]


@continueClass
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

            bbox = afwGeom.BoxI(afwGeom.PointI(minX, minY), afwGeom.PointI(maxX, maxY))

            hdu = fd["FIBERID"]
            fiberIds = hdu.data
            fiberIds = fiberIds.astype(np.int32)   # why is this astype needed? BITPIX=32, no BZERO/BSCALE

            hdu = fd["SLITOFF"]
            slitOffsets = hdu.data.astype(np.float32)

            hdu = fd["SPLINE"]
            splineDataArr = hdu.data.astype(np.float32)

            try:
                hdu = fd["THROUGHPUT"]
                throughputs = hdu.data.astype(np.float32)
            except KeyError:
                throughputs = np.ones_like(slitOffsets[0])

        centerKnots = splineDataArr[:, 0, :]
        centerValues = splineDataArr[:, 1, :]
        wavelengthKnots = splineDataArr[:, 2, :]
        wavelengthValues = splineDataArr[:, 3, :]

        return cls(bbox, fiberIds, centerKnots, centerValues, wavelengthKnots, wavelengthValues,
                   slitOffsets, throughputs)

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

        nKnot = self.getNKnot()
        numFibers = len(fiberIds)
        splineDataArr = np.empty((len(fiberIds), 4, nKnot))
        for ii in range(numFibers):
            splineDataArr[ii][0] = self.getCenterSpline(ii).getX()
            splineDataArr[ii][1] = self.getCenterSpline(ii).getY()
            splineDataArr[ii][2] = self.getWavelengthSpline(ii).getX()
            splineDataArr[ii][3] = self.getWavelengthSpline(ii).getY()

        throughputs = np.empty_like(slitOffsets[0])
        for i, fiberId in enumerate(fiberIds):
            throughputs[i] = self.getThroughput(fiberId)
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
        metadata = self.getMetadata()
        for key in metadata.names():
            hdr[key] = metadata.get(key)

        phu = pyfits.PrimaryHDU(header=hdr)
        hdus.append(phu)

        hdu = pyfits.ImageHDU(fiberIds)
        hdu.name = "FIBERID"
        hdu.header["INHERIT"] = True
        hdus.append(hdu)

        hdu = pyfits.ImageHDU(slitOffsets)
        hdu.name = "SLITOFF"
        hdu.header["INHERIT"] = True
        hdus.append(hdu)

        hdu = pyfits.ImageHDU(splineDataArr)
        hdu.name = "SPLINE"
        hdu.header["INHERIT"] = True
        hdus.append(hdu)

        hdu = pyfits.ImageHDU(throughputs)
        hdu.name = "THROUGHPUT"
        hdu.header["INHERIT"] = True
        hdus.append(hdu)

        # clobber=True in writeto prints a message, so use open instead
        with open(pathName, "wb") as fd:
            hdus.writeto(fd)
