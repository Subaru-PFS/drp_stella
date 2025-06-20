import numpy as np

from lsst.pex.config import Config, Field, ListField, ConfigField, ConfigurableField
from lsst.pex.config import makeConfigClass, FieldValidationError
from lsst.pipe.base import Task, Struct
from lsst.meas.base import CircularApertureFluxAlgorithm, ApertureFluxControl
from lsst.meas.base import FatalAlgorithmError, MeasurementError
from lsst.afw.image import ExposureF, abMagFromFlux, abMagErrFromFluxErr
from lsst.afw.table import SourceCatalog, SourceTable, Point2DKey
from lsst.geom import Point2D
from lsst.daf.base import PropertyList

from pfs.datamodel import FiberStatus
from pfs.drp.stella.fitFocalPlane import FitPolynomialPerFiberTask
from pfs.drp.stella.focalPlaneFunction import FocalPlaneFunction
from .referenceLine import ReferenceLineStatus
from .arcLine import ArcLineSet
from .DetectorMapContinued import DetectorMap
from .datamodel.pfsConfig import PfsConfig

import lsstDebug

__all__ = ("MeasureApertureCorrectionsConfig", "MeasureApertureCorrectionsTask",
           "calculateApertureCorrection")

# Exceptions that the measurement tasks should always propagate up to their callers
FATAL_EXCEPTIONS = (MemoryError, FatalAlgorithmError)

ApertureFluxConfig = makeConfigClass(ApertureFluxControl)


class MeasureApertureCorrectionsConfig(Config):
    """Configuration for MeasureApertureCorrectionsTask"""
    apertureFlux = ConfigField(dtype=ApertureFluxConfig, doc="Aperture flux")
    exclusionFactor = Field(dtype=float, default=2.0,
                            doc="Exclusion zone to apply, as a multiple of the aperture")
    status = ListField(dtype=str, default=["BAD"], doc="Reference line status flags to reject")
    minSignalToNoise = Field(dtype=float, default=10.0, doc="Minumum signal-to-noise ratio for aperture flux")
    fit = ConfigurableField(target=FitPolynomialPerFiberTask, doc="Fit polynomial to each fiber")

    def setDefaults(self):
        super().setDefaults()
        self.apertureFlux.radii = [3.0]
        self.fit.mask = ["NOT_VISIBLE", "BLEND", "SUSPECT", "REJECTED", "BROAD"]
        self.fit.order = 3

    def validate(self):
        super().validate()
        if len(self.apertureFlux.radii) != 1:
            raise FieldValidationError(self.apertureFlux.radii, self.apertureFlux,
                                       "Only one radius is supported (despite the plural name, sorry!)")


class MeasureApertureCorrectionsTask(Task):
    ConfigClass = MeasureApertureCorrectionsConfig
    _DefaultName = "measureApCorr"

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.fluxName = "apFlux"
        self.centerName = "center"
        self.schema = SourceTable.makeMinimalSchema()
        self.index = self.schema.addField("index", type=np.int32, doc="Index to ArcLineSet")
        self.fiberId = self.schema.addField("fiberId", type=np.int32, doc="Fiber identifier")
        self.wavelength = self.schema.addField("wavelength", type=float, doc="Line wavelength")
        self.description = self.schema.addField("description", type=str, size=128, doc="Line description")
        self.status = self.schema.addField("status", type=np.int32, doc="Line status flags")
        self.transition = self.schema.addField("transition", type=str, size=128, doc="Line transition")
        self.source = self.schema.addField("source", type=np.int32, doc="Line source information")
        self.center = Point2DKey.addFields(self.schema, self.centerName, "forced center from detectorMap",
                                           "pixel")
        self.schema.getAliasMap().set("slot_Centroid", self.centerName)
        self.psfFlux = self.schema.addField("psfFlux", type=float, doc="PSF flux")
        self.psfFluxErr = self.schema.addField("psfFluxErr", type=float, doc="Error in PSF flux")
        self.apertureFlux = CircularApertureFluxAlgorithm(self.config.apertureFlux.makeControl(),
                                                          self.fluxName, self.schema, PropertyList())
        self.prefix = self.apertureFlux.makeFieldPrefix(self.fluxName, self.config.apertureFlux.radii[0])
        self.apFlux = self.schema[self.prefix + "_instFlux"].asKey()
        self.apFluxErr = self.schema[self.prefix + "_instFluxErr"].asKey()

        self.makeSubtask("fit")
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, exposure: ExposureF, pfsConfig: PfsConfig, detectorMap: DetectorMap,
            lines: ArcLineSet) -> FocalPlaneFunction:
        """Measure and apply aperture correction

        Parameters
        ----------
        exposure : `ExposureF`
            Exposure containing spectral data.
        pfsConfig : `PfsConfig`
            Top-end configuration.
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        lines : `ArcLineSet`
            Line measurements. Modified in-place to apply the aperture
            correction.

        Returns
        -------
        apCorr : `FocalPlaneFunction`
            Aperture correction model.
        """
        if not lines:
            self.log.warn("Unable to measure aperture correction: no lines")
            return None
        pfsConfig = pfsConfig.select(fiberStatus=FiberStatus.GOOD, fiberId=list(set(lines.fiberId)))

        # Flag lines that are too close; don't even bother measuring them
        linesCopy = lines.copy()
        dispersion = np.array([detectorMap.getDispersionAtCenter(ff) for ff in pfsConfig.fiberId])
        exclusionRadius = self.config.exclusionFactor*self.config.apertureFlux.radii[0]*np.median(dispersion)
        linesCopy.applyExclusionZone(exclusionRadius)

        # Measure aperture photometry
        catalog = self.makeCatalog(linesCopy, detectorMap, pfsConfig)
        self.measure(exposure, catalog)

        # Calculate and fit aperture corrections
        corrections = self.calculate(catalog)
        wavelength = detectorMap.getWavelength()
        minWavelength = wavelength.min()
        maxWavelength = wavelength.max()
        try:
            apCorr = self.fit.run(corrections, pfsConfig, minWavelength=minWavelength,
                                  maxWavelength=maxWavelength)
        except RuntimeError as e:
            self.log.warn("Unable to measure aperture correction: %s", e)
            apCorr = None               # will set all fluxes and flux errors to NaN
            self.log.info("Setting all line fluxes to NaN")

        # Apply aperture corrections.
        self.apply(lines, apCorr, pfsConfig)
        return apCorr

    def makeCatalog(self, lines, detectorMap, pfsConfig):
        """Make a catalog of arc lines in preparation for measurement

        We plug in the detectorMap position of all the arc lines.

        Parameters
        ----------
        lines : `pfs.drp.stella.ArcLineSet`
            List of lines.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Approximate mapping between fiberId,wavelength and x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for specifying good fibers. If not provided,
            will use all fibers in the detectorMap.

        Returns
        -------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of arc lines.
        """
        fiberId = pfsConfig.fiberId[pfsConfig.getSelection(fiberStatus=FiberStatus.GOOD,
                                                           fiberId=detectorMap.fiberId)]
        select = (lines.status & ReferenceLineStatus.fromNames(self.config.status)) == 0
        select &= np.isin(lines.fiberId, fiberId)

        num = select.sum()
        catalog = SourceCatalog(self.schema)
        catalog.reserve(num)

        lines = lines[select]
        points = detectorMap.findPoint(lines.fiberId, lines.wavelength)
        for ii, (xx, yy) in enumerate(points):
            if not np.isfinite(xx) or not np.isfinite(yy):
                continue
            source = catalog.addNew()
            source.set(self.index, ii)
            source.set(self.fiberId, lines.fiberId[ii])
            source.set(self.wavelength, lines.wavelength[ii])
            source.set(self.description, lines.description[ii])
            source.set(self.status, lines.status[ii])
            source.set(self.center, Point2D(xx, yy))
            source.set(self.psfFlux, lines.flux[ii])
            source.set(self.psfFluxErr, lines.fluxErr[ii])
            source.set(self.transition, lines.transition[ii])
            source.set(self.source, lines.source[ii])

        return catalog

    def measure(self, exposure, catalog):
        """Measure aperture photometry

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
        """Measure aperture photometry of a single line

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Arc exposure on which to centroid the line.
        source : `lsst.afw.table.SourceRecord`
            Row from the catalog of arc lines.
        """
        try:
            self.apertureFlux.measure(source, exposure)
        except FATAL_EXCEPTIONS:
            raise
        except MeasurementError as error:
            self.log.debug("MeasurementError on source %d at %s: %s",
                           source.getId(), source.get(self.center), error)
            self.apertureFlux.fail(source, error.cpp)
        except Exception as error:
            self.log.debug("Exception for source %s at %s: %s",
                           source.getId(), source.get(self.center), error)
            self.apertureFlux.fail(source, None)

    def calculate(self, catalog):
        """Calculate aperture corrections

        Parameters
        ----------
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of arc line measurements.

        Returns
        -------
        corrections : `PfsFiberArraySet`
            Aperture corrections for each fiberId,wavelength. This might seem
            like a strange format, but it's exactly what's wanted for input to
            the `FitPolynomialPerFiberTask`.
        """
        apFlux = catalog[self.apFlux]
        apFluxErr = catalog[self.apFluxErr]

        psfFlux = catalog[self.psfFlux]
        psfFluxErr = catalog[self.psfFluxErr]

        # Individual magnitudes will be "AB" (as if our fluxes are in Jy), but the normalisation divides out
        # when we subtract the magnitudes for the aperture correction.
        apMag = abMagFromFlux(apFlux)
        apMagErr = abMagErrFromFluxErr(apFluxErr, apFlux)
        psfMag = abMagFromFlux(psfFlux)
        psfMagErr = abMagErrFromFluxErr(psfFluxErr, psfFlux)

        apCorr = apMag - psfMag
        apCorrErr = np.hypot(apMagErr, psfMagErr)
        fiberId = catalog[self.fiberId]
        wavelength = catalog[self.wavelength]
        flag = catalog[self.prefix + "_flag"]
        status = catalog[self.status]
        empty = np.full_like(wavelength, np.nan)

        reject = ~np.isfinite(apCorr) | ~np.isfinite(apCorrErr)
        with np.errstate(invalid="ignore", divide="ignore"):
            reject |= (apFlux/apFluxErr < self.config.minSignalToNoise)

        status[reject] = ReferenceLineStatus.REJECTED.value
        description = [row[self.description] for row in catalog]
        transition = [row[self.transition] for row in catalog]
        source = [row[self.source] for row in catalog]

        lines = ArcLineSet.fromColumns(fiberId=fiberId, wavelength=wavelength, x=empty, y=empty,
                                       xErr=empty, yErr=empty, xx=empty, yy=empty, xy=empty,
                                       flux=apCorr, fluxErr=apCorrErr, fluxNorm=np.nan,
                                       flag=flag, status=status, description=description,
                                       transition=transition, source=source)
        return lines.asPfsFiberArraySet()

    def apply(self, lines: ArcLineSet, apCorr: FocalPlaneFunction, pfsConfig: PfsConfig):
        """Apply aperture correction to line measurements

        Parameters
        ----------
        lines : `ArcLineSet`
            Line measurements. Modified in-place.
        apCorr : `FocalPlaneFunction`
            Aperture correction model.
        pfsConfig : `PfsConfig`
            Top-end configuration.
        """
        lookup = {}
        for fiberId in set(lines.fiberId):
            select = lines.fiberId == fiberId
            wavelength = lines.wavelength[select]
            if apCorr is None:
                lookup[fiberId] = {wl: (np.nan, np.nan) for wl in wavelength}
            else:
                try:
                    result = calculateApertureCorrection(
                        apCorr, fiberId, wavelength, pfsConfig, lines.flux[select], lines.fluxErr[select]
                    )
                except Exception as exc:
                    self.log.warn("Failed to apply aperture correction for fiber %d: %s", fiberId, exc)
                    lookup[fiberId] = {wl: (np.nan, np.nan) for wl in wavelength}
                    continue
                lookup[fiberId] = {wl: (flux, fluxErr) for
                                   wl, flux, fluxErr in zip(wavelength, result.flux, result.fluxErr)}

        values = np.array([lookup[ff][wl] for ff, wl in zip(lines.fiberId, lines.wavelength)])
        lines.flux[:] = values[:, 0]
        lines.fluxErr[:] = values[:, 1]


def calculateApertureCorrection(apCorr: FocalPlaneFunction, fiberId: int, wavelength, pfsConfig: PfsConfig,
                                flux, fluxErr=None, invert: bool = False) -> Struct:
    """Calculate aperture corrections

    Parameters
    ----------
    apCorr : `FocalPlaneFunction`
        Aperture correction model.
    fiberId : `int`
        Fiber identified. Note that this can't be an array.
    wavelength : `ndarray.Array` of `float`, shape ``(N,)``
        Wavelengths at which to evaluate the aperture correction.
    pfsConfig : `PfsConfig`
        Top-end configuration.
    flux : `ndarray.Array` of `float`, shape ``(N,)``
        Flux measurements.
    fluxErr : `ndarray.Array` of `float`, shape ``(N,)``, optional
        Errors in flux measurements.
    invert : `bool`, optional
        Invert the aperture correction?

    Returns
    -------
    flux : `ndarray.Array` of `float`, shape ``(N,)``
        Aperture-corrected flux measurements.
    fluxErr : `ndarray.Array` of `float`, shape ``(N,)`` or ``None``
        Errors in aperture-corrected flux measurements, or ``None`` if
        ``fluxErr`` is not provided.
    """
    if not np.issubdtype(fiberId, np.integer):
        raise RuntimeError("fiberId must be an integer")
    result = apCorr(wavelength, pfsConfig.select(fiberId=fiberId))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        ratio = 10.0**(-0.4*result.values)
        newFlux = flux/ratio if invert else flux*ratio
        if fluxErr is not None:
            ratioErr = np.abs(-0.4*np.sqrt(result.variances)*ratio*np.log(10))
            newFluxErr = newFlux*np.hypot(fluxErr/flux, ratioErr/ratio) if fluxErr is not None else None
        else:
            newFluxErr = None
    return Struct(flux=newFlux, fluxErr=newFluxErr)
