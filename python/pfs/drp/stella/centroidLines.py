import numpy as np

from lsst.pex.config import Config, Field, ConfigurableField, ConfigField, makeConfigClass
from lsst.pipe.base import Task

from lsst.geom import Point2D, Point2I, Box2I, Extent2I
from lsst.afw.geom.ellipses import Ellipse, Axes
from lsst.afw.geom import SpanSet
from lsst.afw.detection import Footprint
from lsst.afw.table import SourceCatalog, SourceTable
from lsst.meas.base.exceptions import FatalAlgorithmError, MeasurementError
from lsst.meas.base.sdssCentroid import SdssCentroidAlgorithm, SdssCentroidControl
from lsst.meas.base.psfFlux import PsfFluxAlgorithm, PsfFluxControl
from lsst.ip.isr.isrFunctions import createPsf

from .arcLine import ArcLine, ArcLineSet
from .images import convolveImage

import lsstDebug

__all__ = ("CentroidLinesConfig", "CentroidLinesTask")


# Exceptions that the measurement tasks should always propagate up to their callers
FATAL_EXCEPTIONS = (MemoryError, FatalAlgorithmError)


CentroidConfig = makeConfigClass(SdssCentroidControl)
PhotometryConfig = makeConfigClass(PsfFluxControl)


class CentroidLinesConfig(Config):
    """Configuration for CentroidLinesTask"""
    centroider = ConfigField(dtype=CentroidConfig, doc="Centroider")
    photometer = ConfigField(dtype=PhotometryConfig, doc="Photometer")
    footprintSize = Field(dtype=float, default=3, doc="Radius of footprint (pixels)")
    fwhm = Field(dtype=float, default=1.5, doc="FWHM of PSF (pixels)")
    kernelSize = Field(dtype=float, default=4.0, doc="Size of convolution kernel (sigma)")
    threshold = Field(dtype=float, default=5.0, doc="Signal-to-noise threshold for lines")


class CentroidLinesTask(Task):
    """Centroid lines on an arc"""
    ConfigClass = CentroidLinesConfig

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.centroidName = "centroid"
        self.photometryName = "flux"
        self.schema = SourceTable.makeMinimalSchema()
        self.fiberId = self.schema.addField("fiberId", type=np.int32, doc="Fiber identifier")
        self.wavelength = self.schema.addField("wavelength", type=float, doc="Line wavelength")
        self.description = self.schema.addField("description", type=str, size=128, doc="Line description")
        self.ignore = self.schema.addField("ignore", type="Flag", doc="Ignore line?")
        self.status = self.schema.addField("status", type=np.int32, doc="Line status flags")
        self.centroider = SdssCentroidAlgorithm(self.config.centroider.makeControl(), self.centroidName,
                                                self.schema)
        self.schema.getAliasMap().set("slot_Centroid", self.centroidName)
        self.photometer = PsfFluxAlgorithm(self.config.photometer.makeControl(), self.photometryName,
                                           self.schema)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, exposure, referenceLines, detectorMap):
        """Centroid lines on an arc

        We use the LSST stack's ``SdssCentroid`` measurement at the position
        of known arc lines.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Arc exposure on which to centroid lines.
        referenceLines : `dict` (`int`: `pfs.drp.stella.ReferenceLineSet`)
            List of reference lines for each fiberId.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Approximate mapping between fiberId,wavelength and x,y.

        Returns
        -------
        lines : `pfs.drp.stella.ArcLineSet`
            Centroided lines.
        """
        self.checkPsf(exposure)
        convolved = self.convolveImage(exposure)
        catalog = self.makeCatalog(referenceLines, detectorMap, convolved)
        self.measure(exposure, catalog)
        self.display(exposure, catalog)
        return self.translate(catalog)

    def checkPsf(self, exposure):
        """Check that the PSF is present in the ``exposure``

        If the PSF isn't present in the ``exposure``, then we use the ``fwhm``
        from the config.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image to convolve. The PSF must be set.

        Returns
        -------
        psf : `lsst.afw.detection.Psf`
            Two-dimensional point-spread function.
        """
        psf = exposure.getPsf()
        if psf is not None:
            return
        exposure.setPsf(createPsf(self.config.fwhm))

    def convolveImage(self, exposure):
        """Convolve image by Gaussian kernel

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image to convolve. The PSF must be set.

        Returns
        -------
        convolved : `lsst.afw.image.MaskedImage`
            Convolved image.
        """
        psf = exposure.getPsf()
        sigma = psf.computeShape().getTraceRadius()
        convolvedImage = convolveImage(exposure.maskedImage, sigma, sigma, sigmaNotFwhm=True)
        if self.debugInfo.displayConvolved:
            from lsst.afw.display import Display
            Display(frame=1).mtv(convolvedImage)

        return convolvedImage

    def makeCatalog(self, referenceLines, detectorMap, convolved):
        """Make a catalog of arc lines

        We plug in the rough position of all the arc lines, from the identified
        reference lines and the ``detectorMap``, to serve as input to the
        ``SdssCentroid`` measurement plugin.

        Parameters
        ----------
        referenceLines : `dict` (`int`: `pfs.drp.stella.ReferenceLineSet`)
            List of reference lines for each fiberId.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Approximate mapping between fiberId,wavelength and x,y.
        convolved : `lsst.afw.image.MaskedImage`
            PSF-convolved image.

        Returns
        -------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of arc lines.
        """
        num = sum(len(rl) for rl in referenceLines.values())
        catalog = SourceCatalog(self.schema)
        catalog.reserve(num)
        bbox = convolved.getBBox()

        for fiberId in referenceLines:
            for rl in referenceLines[fiberId]:
                xx, yy = detectorMap.findPoint(fiberId, rl.wavelength)
                if not np.isfinite(xx) or not np.isfinite(yy):
                    continue
                point = Point2D(xx, yy)
                spans = SpanSet.fromShape(Ellipse(Axes(self.config.footprintSize, self.config.footprintSize),
                                                  point))
                peak = self.findPeak(convolved.image, point)

                source = catalog.addNew()
                source.set(self.fiberId, fiberId)
                source.set(self.wavelength, rl.wavelength)
                source.set(self.description, rl.description)
                source.set(self.status, rl.status)

                if bbox.contains(peak):
                    sn = convolved.image[peak]/np.sqrt(convolved.variance[peak])
                    ignore = sn < self.config.threshold
                else:
                    ignore = True

                source.set(self.ignore, ignore)

                footprint = Footprint(spans, detectorMap.bbox)
                fpPeak = footprint.getPeaks().addNew()
                fpPeak.setFx(peak.getX())
                fpPeak.setFy(peak.getY())
                source.setFootprint(footprint)
        return catalog

    def findPeak(self, image, center):
        """Find a peak in the footprint around the expected peak

        Parameters
        ----------
        image : `lsst.afw.image.Image`
            Image on which to find peak.
        center : `lsst.geom.Point2D`
            Expected center of peak.

        Returns
        -------
        peak : `lsst.geom.Point2I`
            Coordinates of the peak.
        """
        x0 = int(center.getX() + 0.5) - self.config.footprintSize
        y0 = int(center.getY() + 0.5) - self.config.footprintSize
        size = 2*self.config.footprintSize + 1
        box = Box2I(Point2I(x0, y0), Extent2I(size, size))
        box.clip(image.getBBox())
        subImage = image[box]
        yy, xx = np.unravel_index(np.argmax(subImage.array), subImage.array.shape)
        return Point2I(xx + x0, yy + y0)

    def measure(self, exposure, catalog):
        """Measure the centroids

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Arc exposure on which to centroid lines.
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of arc lines; modified with the measured positions.
        """
        for source in catalog:
            if not source.get(self.ignore):
                self.measureLine(exposure, source)

    def measureLine(self, exposure, source):
        """Measure a single line

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Arc exposure on which to centroid the line.
        source : `lsst.afw.table.SourceRecord`
            Row from the catalog of arc lines; modified with the measured
            position.
        """
        for measurement in (self.centroider, self.photometer):
            try:
                measurement.measure(source, exposure)
            except FATAL_EXCEPTIONS:
                raise
            except MeasurementError as error:
                self.log.debug("MeasurementError on source %d at (%f,%f): %s",
                               source.getId(), source.get("centroid_x"), source.get("centroid_y"), error)
                measurement.fail(source, error.cpp)
            except Exception as error:
                self.log.debug("Exception for source %s at (%f,%f): %s",
                               source.getId(), source.get("centroid_x"), source.get("centroid_y"), error)
                measurement.fail(source)

    def translate(self, catalog):
        """Translate the catalog of measured centroids to a simpler format

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of arc lines with measured centroids.

        Returns
        -------
        lines : `pfs.drp.stella.ArcLineSet`
            List of arc lines.
        """
        return ArcLineSet([
            ArcLine(source[self.fiberId], source[self.wavelength], source[self.centroidName + "_x"],
                    source[self.centroidName + "_y"], source[self.centroidName + "_xErr"],
                    source[self.centroidName + "_yErr"], source[self.photometryName + "_instFlux"],
                    source[self.photometryName + "_instFluxErr"],
                    (source[self.centroidName + "_flag"] | source[self.photometryName + "_flag"] |
                     source[self.ignore]),
                    source[self.status], source[self.description]) for source in catalog])

    def display(self, exposure, catalog):
        """Display centroids

        Displays the exposure, initial positions with a red ``+``, and final
        positions with a ``x`` that is green if the measurement is clean, and
        yellow otherwise.

        The display is controlled by debug parameters:
        - ``display`` (`bool`): Enable display?
        - ``frame`` (`int` or  afwDisplay.Display, optional): Frame to use for display (defaults to 1).
        - ``displayExposure'' : `bool`
            Use display.mtv to show the exposure; if False
            the caller is responsible for the mtv and maybe
            an erase too
        - ```fiberIds`` (list of int): only show these fiberIds

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to display.
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog with measurements.
        """
        if not self.debugInfo.display:
            return
        from lsst.afw.display import Display
        if isinstance(self.debugInfo.frame, Display):
            disp = self.debugInfo.frame
        else:
            disp = Display(frame=self.debugInfo.frame or 1)

        if self.debugInfo.displayExposure:
            disp.mtv(exposure)

        with disp.Buffering():
            if self.debugInfo.fiberIds:
                showPeak = np.zeros(len(catalog), dtype=bool)
                fiberId = catalog["fiberId"]
                for fid in self.debugInfo.fiberIds:
                    showPeak = np.logical_or(showPeak, fiberId == fid)

                catalog = catalog[showPeak]

            for row in catalog:
                peak = row.getFootprint().getPeaks()[0]
                disp.dot("+", peak.getFx(), peak.getFy(), size=2, ctype="red")
                ctype = "yellow" if row.get("centroid_flag") else "green"
                disp.dot("x", row.get("centroid_x"), row.get("centroid_y"), size=5, ctype=ctype)
