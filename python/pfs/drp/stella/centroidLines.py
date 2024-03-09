import numpy as np

from lsst.pex.config import Config, Field, ConfigurableField, ConfigField, ListField, makeConfigClass
from lsst.pipe.base import Task

from lsst.geom import Point2D, Point2I, Box2I, Extent2I
from lsst.afw.table import SourceCatalog, SourceTable
from lsst.meas.base.exceptions import FatalAlgorithmError, MeasurementError
from lsst.meas.base.sdssCentroid import SdssCentroidAlgorithm, SdssCentroidControl
from lsst.meas.base.sdssShape import SdssShapeAlgorithm, SdssShapeControl
from lsst.meas.base.psfFlux import PsfFluxAlgorithm, PsfFluxControl

from pfs.datamodel import FiberStatus
from .arcLine import ArcLineSet
from .images import convolveImage
from .fitContinuum import FitContinuumTask
from .utils.psf import checkPsf
from .makeFootprint import makeFootprint
from .traces import medianFilterColumns
from .referenceLine import ReferenceLineStatus

import lsstDebug

__all__ = ("CentroidLinesConfig", "CentroidLinesTask")


# Exceptions that the measurement tasks should always propagate up to their callers
FATAL_EXCEPTIONS = (MemoryError, FatalAlgorithmError)


CentroidConfig = makeConfigClass(SdssCentroidControl)
ShapeConfig = makeConfigClass(SdssShapeControl)
PhotometryConfig = makeConfigClass(PsfFluxControl)


class CentroidLinesConfig(Config):
    """Configuration for CentroidLinesTask"""
    doSubtractContinuum = Field(dtype=bool, default=False, doc="Subtract continuum before centroiding lines?")
    continuum = ConfigurableField(target=FitContinuumTask, doc="Continuum subtraction")
    doSubtractTraces = Field(dtype=bool, default=True, doc="Subtract traces before centroiding lines?")
    halfHeight = Field(dtype=int, default=35, doc="Half-height for column trace determination")
    mask = ListField(
        dtype=str,
        default=["BAD", "SAT", "CR", "REFLINE", "NO_DATA"],
        doc="Mask planes to ignore in trace removal and peak finding",
    )
    centroider = ConfigField(dtype=CentroidConfig, doc="Centroider")
    shapes = ConfigField(dtype=ShapeConfig, doc="Shape measurement")
    peakSearch = Field(dtype=float, default=3, doc="Radius of peak search (pixels)")
    footprintHeight = Field(dtype=int, default=11, doc="Height of footprint (pixels)")
    footprintWidth = Field(dtype=float, default=3, doc="Width of footprint (pixels)")
    photometer = ConfigField(dtype=PhotometryConfig, doc="Photometer")
    fwhm = Field(dtype=float, default=1.5, doc="FWHM of PSF (pixels)")
    kernelSize = Field(dtype=float, default=4.0, doc="Size of convolution kernel (sigma)")
    threshold = Field(dtype=float, default=5.0, doc="Signal-to-noise threshold for lines")

    def setDefaults(self):
        super().setDefaults()
        self.centroider.binmax = 1
        self.photometer.badMaskPlanes = ["BAD", "SAT", "CR", "NO_DATA"]


class CentroidLinesTask(Task):
    """Centroid lines on an arc"""
    ConfigClass = CentroidLinesConfig
    _DefaultName = "centroidLines"

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.centroidName = "centroid"
        self.shapeName = "shape"
        self.photometryName = "flux"
        self.schema = SourceTable.makeMinimalSchema()
        self.fiberId = self.schema.addField("fiberId", type=np.int32, doc="Fiber identifier")
        self.wavelength = self.schema.addField("wavelength", type=float, doc="Line wavelength")
        self.description = self.schema.addField("description", type=str, size=128, doc="Line description")
        self.ignore = self.schema.addField("ignore", type="Flag", doc="Ignore line?")
        self.status = self.schema.addField("status", type=np.int32, doc="Line status flags")
        self.transition = self.schema.addField("transition", type=str, size=128, doc="Line transition")
        self.source = self.schema.addField("source", type=np.int32, doc="Source of line information")
        self.centroider = SdssCentroidAlgorithm(self.config.centroider.makeControl(), self.centroidName,
                                                self.schema)
        self.schema.getAliasMap().set("slot_Centroid", self.centroidName)
        self.shapes = SdssShapeAlgorithm(self.config.shapes.makeControl(), self.shapeName, self.schema)
        self.photometer = PsfFluxAlgorithm(self.config.photometer.makeControl(), self.photometryName,
                                           self.schema)
        self.debugInfo = lsstDebug.Info(__name__)
        self.makeSubtask("continuum")

    def run(self, exposure, referenceLines, detectorMap, pfsConfig=None, fiberTraces=None, seed=0):
        """Centroid lines on an arc

        We use the LSST stack's ``SdssCentroid`` measurement at the position
        of known arc lines.

        This method optionally performs continuum subtraction before handing
        off to the ``centroidLines`` method to do the actual centroiding.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Arc exposure on which to centroid lines.
        referenceLines : `pfs.drp.stella.ReferenceLineSet`
            List of reference lines.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Approximate mapping between fiberId,wavelength and x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration, for specifying good fibers. If not provided,
            will use all fibers in the detectorMap.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`, optional
            Position and profile of fiber traces. Required only for continuum
            subtraction.
        seed : `int`
            Seed for random number generator.

        Returns
        -------
        lines : `pfs.drp.stella.ArcLineSet`
            Centroided lines.
        """
        if self.config.doSubtractContinuum:
            if fiberTraces is None:
                raise RuntimeError("No fiberTraces provided for continuum subtraction")
            with self.continuum.subtractionContext(exposure.maskedImage, fiberTraces, detectorMap,
                                                   referenceLines):
                return self.centroidLines(exposure, referenceLines, detectorMap, pfsConfig, seed)
        return self.centroidLines(exposure, referenceLines, detectorMap, pfsConfig, seed)

    def centroidLines(self, exposure, referenceLines, detectorMap, pfsConfig=None, seed=0):
        """Centroid lines on an arc

        We use the LSST stack's ``SdssCentroid`` measurement at the position
        of known arc lines.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Arc exposure on which to centroid lines.
        referenceLines : `pfs.drp.stella.ReferenceLineSet`
            List of reference lines.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Approximate mapping between fiberId,wavelength and x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration, for specifying good fibers. If not provided,
            will use all fibers in the detectorMap.
        seed : `int`
            Seed for random number generator.

        Returns
        -------
        lines : `pfs.drp.stella.ArcLineSet`
            Centroided lines.
        """
        checkPsf(exposure, fwhm=self.config.fwhm)

        traces = None
        if self.config.doSubtractTraces:
            # Median filter on columns
            bad = (exposure.mask.array & exposure.mask.getPlaneBitMask(self.config.mask)) != 0
            traces = medianFilterColumns(exposure.image.array, bad, self.config.halfHeight)

        # Measure centroids on traces-subtracted image
        if traces is not None:
            exposure.image.array -= traces
        try:
            convolved = self.convolveImage(exposure)
            catalog = self.makeCatalog(referenceLines, detectorMap, exposure.image, convolved, pfsConfig)
            self.measure(exposure, catalog, seed)
            self.display(exposure, catalog, detectorMap)  # arguably convolved would be more useful
        finally:
            if traces is not None:
                exposure.image.array += traces

        lines = self.translate(catalog)
        self.log.info("Measured %d line centroids", len(lines))
        return lines

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
        sigma = psf.computeShape(psf.getAveragePosition()).getTraceRadius()
        convolvedImage = convolveImage(exposure.maskedImage, sigma, sigma, sigmaNotFwhm=True)
        if self.debugInfo.displayConvolved:
            from lsst.afw.display import Display
            Display(frame=1).mtv(convolvedImage)

        return convolvedImage

    def makeCatalog(self, referenceLines, detectorMap, image, convolved, pfsConfig=None):
        """Make a catalog of arc lines

        We plug in the rough position of all the arc lines, from the identified
        reference lines and the ``detectorMap``, to serve as input to the
        ``SdssCentroid`` measurement plugin.

        Parameters
        ----------
        referenceLines : `pfs.drp.stella.ReferenceLineSet`
            List of reference lines.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Approximate mapping between fiberId,wavelength and x,y.
        image : `lsst.afw.image.Image`
            Image (not convolved).
        convolved : `lsst.afw.image.MaskedImage`
            PSF-convolved image.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configuration, for specifying good fibers. If not provided,
            will use all fibers in the detectorMap.

        Returns
        -------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of arc lines.
        """
        num = detectorMap.getNumFibers()*len(referenceLines)
        catalog = SourceCatalog(self.schema)
        catalog.reserve(num)
        bbox = convolved.getBBox()

        if pfsConfig is not None:
            indices = pfsConfig.selectByFiberStatus(FiberStatus.GOOD, detectorMap.fiberId)
            fiberId = detectorMap.fiberId[indices]
        else:
            fiberId = detectorMap.fiberId

        for ff in fiberId:
            for rl in referenceLines:
                xx, yy = detectorMap.findPoint(ff, rl.wavelength)
                if not np.isfinite(xx) or not np.isfinite(yy):
                    continue
                expected = Point2D(xx, yy)
                peak = self.findPeak(convolved, expected)
                footprint = makeFootprint(image, peak, self.config.footprintHeight,
                                          self.config.footprintWidth)
                assert image.getBBox().contains(footprint.getSpans().getBBox())

                source = catalog.addNew()
                source.set(self.fiberId, ff)
                source.set(self.wavelength, rl.wavelength)
                source.set(self.description, rl.description)
                source.set(self.status, rl.status)
                source.set(self.transition, rl.transition)
                source.set(self.source, rl.source)
                source.setFootprint(footprint)

                if bbox.contains(peak):
                    sn = convolved.image[peak]/np.sqrt(convolved.variance[peak])
                    ignore = sn < self.config.threshold
                else:
                    ignore = True

                source.set(self.ignore, ignore)
        return catalog

    def findPeak(self, image, center):
        """Find a peak in the footprint around the expected peak

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Image on which to find peak.
        center : `lsst.geom.Point2D`
            Expected center of peak.

        Returns
        -------
        peak : `lsst.geom.Point2I`
            Coordinates of the peak.
        """
        x0 = int(center.getX() + 0.5) - self.config.peakSearch
        y0 = int(center.getY() + 0.5) - self.config.peakSearch
        size = 2*self.config.peakSearch + 1
        box = Box2I(Point2I(x0, y0), Extent2I(size, size))
        box.clip(image.getBBox())
        subImage = image[box]
        badBitmask = subImage.mask.getPlaneBitMask(self.config.mask)
        good = np.isfinite(subImage.image.array) & ((subImage.mask.array & badBitmask) == 0)
        yy, xx = np.unravel_index(np.argmax(np.where(good, subImage.image.array, 0.0)),
                                  subImage.image.array.shape)
        return Point2I(xx + x0, yy + y0)

    def measure(self, exposure, catalog, seed=0):
        """Measure the centroids

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Arc exposure on which to centroid lines.
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of arc lines; modified with the measured positions.
        seed : `int`
            Seed for random number generator.
        """
        with DeblendContext(exposure.maskedImage, catalog, seed) as deblend:
            for source in catalog:
                if source.get(self.ignore):
                    continue
                deblend.insert(source)
                self.measureLine(exposure, source)
                deblend.remove(source)

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
        for measurement in (self.centroider, self.shapes, self.photometer):
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
                measurement.fail(source, None)

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
        return ArcLineSet.fromColumns(
            fiberId=catalog[self.fiberId],
            wavelength=catalog[self.wavelength],
            x=catalog[self.centroidName + "_x"],
            y=catalog[self.centroidName + "_y"],
            xErr=catalog[self.centroidName + "_xErr"],
            yErr=catalog[self.centroidName + "_yErr"],
            xx=catalog[self.shapeName + "_xx"],
            yy=catalog[self.shapeName + "_yy"],
            xy=catalog[self.shapeName + "_xy"],
            flux=catalog[self.photometryName + "_instFlux"],
            fluxErr=catalog[self.photometryName + "_instFluxErr"],
            fluxNorm=np.nan,  # Don't have a value yet
            flag=(catalog[self.centroidName + "_flag"] | catalog[self.photometryName + "_flag"] |
                  catalog[self.ignore]),
            status=catalog[self.status],
            description=[row[self.description] for row in catalog],
            transition=[row[self.transition] for row in catalog],
            source=[row[self.source] for row in catalog],
        )

    def display(self, exposure, catalog, detectorMap):
        """Display centroids

        Displays the exposure, detectorMap positions with a ``o`` in cyan (good)
        or blue (bad), initial peaks with a red ``+``, and final positions with
        a ``x`` that is green if the measurement is clean, and yellow otherwise.

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
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        """
        if not self.debugInfo.display:
            return
        from lsst.afw.display import Display
        if isinstance(self.debugInfo.frame, Display):
            disp = self.debugInfo.frame
        else:
            disp = Display(frame=self.debugInfo.frame or 1)

        if self.debugInfo.displayExposure:
            disp.scale('asinh', 'zscale', Q=8)
            disp.mtv(exposure)

        badLine = ReferenceLineStatus.fromNames("NOT_VISIBLE", "BLEND", "SUSPECT", "REJECTED")

        with disp.Buffering():
            # N.b. "not fiberIds" and "fiberIds not in (False, None)" fail with ndarray
            if self.debugInfo.fiberIds is not False and self.debugInfo.fiberIds is not None:
                showPeak = np.zeros(len(catalog), dtype=bool)
                fiberId = catalog["fiberId"]
                for fid in self.debugInfo.fiberIds:
                    showPeak = np.logical_or(showPeak, fiberId == fid)

                catalog = catalog[showPeak]

            for row in catalog:
                point = detectorMap.findPoint(row["fiberId"], row["wavelength"])
                ctype = "blue" if (row["status"] & badLine) != 0 else "cyan"
                disp.dot("o", point.getX(), point.getY(), ctype=ctype)
                peak = row.getFootprint().getPeaks()[0]
                disp.dot("+", peak.getFx(), peak.getFy(), size=2, ctype="red")
                ctype = "yellow" if row.get("centroid_flag") else "green"
                disp.dot("x", row.get("centroid_x"), row.get("centroid_y"), size=5, ctype=ctype)


class DeblendContext:
    """Context manager that removes all sources from the image

    Allows insertion and removal of sources one by one, and puts them all back
    when done.

    Parameters
    ----------
    image : `lsst.afw.image.MaskedImage`
        Arc image on which to centroid lines.
    catalog : `lsst.afw.table.SourceCatalog`
        Catalog of arc lines; modified with the measured positions.
    seed : `int`, optional
        Seed for random number generator.
    """
    def __init__(self, image, catalog, seed=0):
        self.image = image
        self.catalog = catalog
        self.xy0 = image.getXY0()

        array = image.image.array
        self.original = array.copy()
        rng = np.random.RandomState(seed)
        with np.errstate(invalid="ignore"):
            sigma = np.where(image.variance.array > 0, np.sqrt(image.variance.array), 0.0)
        self.noise = rng.normal(0.0, sigma, array.shape).astype(array.dtype)

    def __enter__(self):
        """Start the context management

        We replace the entire image with noise: we don't want to leave any wings
        outside the source footprints. Individual sources will be added and
        removed in turn via the ``insert`` and ``remove`` methods.

        Returns
        -------
        self : `DeblendContext`
            Deblender that can ``insert`` and ``remove`` sources from the image.
        """
        self.image.image.array[:] = self.noise
        return self

    def insert(self, source):
        """Insert a source"""
        self._apply(source, self.original)

    def remove(self, source):
        """Remove a source"""
        self._apply(source, self.noise)

    def _apply(self, source, array):
        """Put pixels of a source from an array into the image

        Parameters
        ----------
        source : `lsst.afw.table.SourceRecord`
            Source whose pixels will be modified.
        array : `numpy.ndarray`
            Array with pixels to put into the image.
        """
        spans = source.getFootprint().getSpans()
        spans.unflatten(self.image.image.array, spans.flatten(array, self.xy0), self.xy0)

    def __exit__(self, *args):
        """Finish the context management

        We restore the original image completely. No exceptions are suppressed.
        """
        self.image.image.array[:] = self.original
