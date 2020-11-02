import numpy as np

from lsst.pex.config import Config, Field, ConfigField, makeConfigClass
from lsst.pipe.base import Task

from lsst.geom import Point2D, Point2I, Box2I, Extent2I
from lsst.afw.geom.ellipses import Ellipse, Axes
from lsst.afw.geom import SpanSet
from lsst.afw.detection import Footprint
from lsst.afw.table import SourceCatalog, SourceTable
from lsst.afw.math import GaussianFunction1D, SeparableKernel, convolve, ConvolutionControl
from lsst.meas.base.exceptions import FatalAlgorithmError, MeasurementError
from lsst.meas.base.sdssCentroid import SdssCentroidAlgorithm, SdssCentroidControl
from lsst.ip.isr.isrFunctions import createPsf

from .arcLine import ArcLine, ArcLineSet

import lsstDebug

__all__ = ("CentroidLinesConfig", "CentroidLinesTask")


# Exceptions that the measurement tasks should always propagate up to their callers
FATAL_EXCEPTIONS = (MemoryError, FatalAlgorithmError)


CentroidConfig = makeConfigClass(SdssCentroidControl)


class CentroidLinesConfig(Config):
    """Configuration for CentroidLinesTask"""
    centroider = ConfigField(dtype=CentroidConfig, doc="Centroider")
    footprintSize = Field(dtype=float, default=3, doc="Radius of footprint (pixels)")
    fwhm = Field(dtype=float, default=1.5, doc="FWHM of PSF (pixels)")
    kernelSize = Field(dtype=float, default=4.0, doc="Size of convolution kernel (sigma)")


class CentroidLinesTask(Task):
    """Centroid lines on an arc"""
    ConfigClass = CentroidLinesConfig

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.centroidName = "centroid"
        self.schema = SourceTable.makeMinimalSchema()
        self.fiberId = self.schema.addField("fiberId", type=np.int32, doc="Fiber identifier")
        self.wavelength = self.schema.addField("wavelength", type=float, doc="Line wavelength")
        self.description = self.schema.addField("description", type=str, size=128, doc="Line description")
        self.status = self.schema.addField("status", type=np.int32, doc="Line status flags")
        self.centroider = SdssCentroidAlgorithm(self.config.centroider.makeControl(), self.centroidName,
                                                self.schema)
        self.schema.getAliasMap().set("slot_Centroid", self.centroidName)
        self.debugInfo = lsstDebug.Info(__name__)

    def getReferenceLines(self, spectra):
        """Get reference lines from spectra

        This is a convenience method for generating the ``referenceLines`` input
        for the ``run`` method.

        Parameters
        ----------
        spectra: : `pfs.drp.stella.SpectrumSet`
            Extracted spectra, with reference lines identified.

        Returns
        -------
        referenceLines : `dict` (`int`: `list` of `pfs.drp.stella.ReferenceLine`)
            List of reference lines for each fiberId.
        """
        return {ss.fiberId: [rl for rl in ss.referenceLines if (rl.status & rl.Status.FIT) != 0]
                for ss in spectra}

    def run(self, exposure, referenceLines, detectorMap):
        """Centroid lines on an arc

        We use the LSST stack's ``SdssCentroid`` measurement at the position
        of known arc lines.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Arc exposure on which to centroid lines.
        referenceLines : `dict` (`int`: `list` of `pfs.drp.stella.ReferenceLine`)
            List of reference lines for each fiberId.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Approximate mapping between fiberId,wavelength and x,y.

        Returns
        -------
        lines : `pfs.drp.stella.ArcLineSet`
            Centroided lines.
        """
        if not exposure.getPsf():
            exposure.setPsf(createPsf(self.config.fwhm))
        convolved = self.convolveImage(exposure)
        catalog = self.makeCatalog(referenceLines, detectorMap, convolved)
        self.measure(exposure, catalog)
        self.display(exposure, catalog)
        return self.translate(catalog)

    def convolveImage(self, exposure):
        """Convolve image by Gaussian kernel set by the PSF

        The boundary is unconvolved, and is set to ``NaN``.

        We don't convolve the mask or variance plane, since we don't use those
        for finding the peak.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image to convolve. The PSF must be set.

        Returns
        -------
        convolved : `lsst.afw.image.Image`
            Convolved image.
        """
        psfSigma = exposure.getPsf().computeShape().getTraceRadius()
        psfSize = 2*int(self.config.kernelSize*psfSigma) + 1
        kernel = SeparableKernel(psfSize, psfSize, GaussianFunction1D(psfSigma), GaussianFunction1D(psfSigma))

        convolvedImage = exposure.image.Factory(exposure.image.getBBox())
        convolve(convolvedImage, exposure.image, kernel, ConvolutionControl())

        if self.debugInfo.displayConvolved:
            from lsst.afw.display import Display
            Display(backend=self.debugInfo.backend or "ds9", frame=1).mtv(convolvedImage)

        return convolvedImage

    def makeCatalog(self, referenceLines, detectorMap, convolved):
        """Make a catalog of arc lines

        We plug in the rough position of all the arc lines, from the identified
        reference lines and the ``detectorMap``, to serve as input to the
        ``SdssCentroid`` measurement plugin.

        Parameters
        ----------
        referenceLines : `dict` (`int`: `list` of `pfs.drp.stella.ReferenceLine`)
            List of reference lines for each fiberId.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Approximate mapping between fiberId,wavelength and x,y.
        convolved : `lsst.afw.image.Image`
            PSF-convolved image.

        Returns
        -------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of arc lines.
        """
        num = sum(len(rl) for rl in referenceLines.values())
        catalog = SourceCatalog(self.schema)
        catalog.reserve(num)

        for fiberId in referenceLines:
            for rl in referenceLines[fiberId]:
                source = catalog.addNew()
                xx, yy = detectorMap.findPoint(fiberId, rl.wavelength)
                point = Point2D(xx, yy)
                source.set(self.fiberId, fiberId)
                source.set(self.wavelength, rl.wavelength)
                source.set(self.description, rl.description)
                source.set(self.status, rl.status)

                spans = SpanSet.fromShape(Ellipse(Axes(self.config.footprintSize, self.config.footprintSize),
                                                  point))
                rough = self.findPeak(convolved, point)
                footprint = Footprint(spans, detectorMap.bbox)
                peak = footprint.getPeaks().addNew()
                peak.setFx(rough.getX())
                peak.setFy(rough.getY())
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
        try:
            self.centroider.measure(source, exposure)
        except FATAL_EXCEPTIONS:
            raise
        except MeasurementError as error:
            self.log.debug("MeasurementError on source %d at (%f,%f): %s",
                           source.getId(), source.get("centroid_x"), source.get("centroid_y"), error)
            self.centroider.fail(source, error.cpp)
        except Exception as error:
            self.log.debug("Exception for source %s at (%f,%f): %s",
                           source.getId(), source.get("centroid_x"), source.get("centroid_y"), error)
            self.centroider.fail(source)

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
                    source[self.centroidName + "_yErr"], source["centroid_flag"], source[self.status],
                    source[self.description]) for source in catalog])

    def display(self, exposure, catalog):
        """Display centroids

        Displays the exposure, initial positions with a red ``+``, and final
        positions with a ``x`` that is green if the measurement is clean, and
        yellow otherwise.

        The display is controlled by debug parameters:
        - ``display`` (`bool`): Enable display?
        - ``frame`` (`int`, optional): Frame to use for display (defaults to 1).
        - ``backend`` (`str`, optional): Backend to use for display (defaults to
          ``ds9``).

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
        disp = Display(frame=self.debugInfo.frame or 1, backend=self.debugInfo.backend or "ds9")
        disp.mtv(exposure)
        with disp.Buffering():
            for row in catalog:
                peak = row.getFootprint().getPeaks()[0]
                disp.dot("+", peak.getFx(), peak.getFy(), size=5, ctype="red")
                ctype = "yellow" if row.get("centroid_flag") else "green"
                disp.dot("x", row.get("centroid_x"), row.get("centroid_y"), size=5, ctype=ctype)
