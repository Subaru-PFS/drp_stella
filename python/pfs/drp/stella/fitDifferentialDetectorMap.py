import os
from types import SimpleNamespace

import numpy as np

from lsst.utils import getPackageDir
from lsst.pex.config import Field

from pfs.drp.stella.fitGlobalDetectorMap import FitGlobalDetectorMapTask, fitStraightLine
from pfs.drp.stella import DetectorMap, DifferentialDetectorMap, GlobalDetectorModelScaling
from pfs.drp.stella.arcLine import ArcLineSet


class ArcLineResiduals(SimpleNamespace):
    """Residuals in arc line positions

    Analagous to `ArcLine`, this stores the position measurement of a single
    arc line, but the ``x,y`` positions are relative to a detectorMap. The
    original ``x,y`` positions are stored as ``xOrig,yOrig``.

    Parameters
    ----------
    fiberId : `int`
        Fiber identifier.
    wavelength : `float`
        Reference line wavelength (nm).
    x, y : `float`
        Differential position relative to an external detectorMap.
    xOrig, yOrig : `float`
        Measured position.
    xErr, yErr : `float`
        Error in measured position.
    flag : `bool`
        Measurement flag (``True`` indicates an error in measurement).
    status : `pfs.drp.stella.ReferenceLine.Status`
        Flags whether the lines are fitted, clipped or reserved etc.
    description : `str`
        Line description (e.g., ionic species)
    """
    def __init__(self, fiberId, wavelength, x, y, xOrig, yOrig, xErr, yErr, flag, status, description):
        return super().__init__(fiberId=fiberId, wavelength=wavelength, x=x, y=y, xOrig=xOrig, yOrig=yOrig,
                                xErr=xErr, yErr=yErr, flag=flag, status=status, description=description)


class ArcLineResidualsSet(ArcLineSet):
    """A list of `ArcLineResiduals`

    Analagous to `ArcLineSet`, this stores the position measurement of a list
    of arc lines, but the ``x,y`` positions are relative to a detectorMap. The
    original ``x,y`` positions are stored as ``xOrig,yOrig``.

    Parameters
    ----------
    lines : `list` of `ArcLineResiduals`
        List of lines in the spectra.
    """
    def append(self, fiberId, wavelength, x, y, xOrig, yOrig, xErr, yErr, flag, status, description):
        """Append to the list of lines

        Parameters
        ----------
        fiberId : `int`
            Fiber identifier.
        wavelength : `float`
            Reference line wavelength (nm).
        x, y : `float`
            Differential position relative to an external detectorMap.
        xOrig, yOrig : `float`
            Measured position.
        xErr, yErr : `float`
            Error in measured position.
        flag : `bool`
            Measurement flag (``True`` indicates an error in measurement).
        status : `pfs.drp.stella.ReferenceLine.Status`
            Flags whether the lines are fitted, clipped or reserved etc.
        description : `str`
            Line description (e.g., ionic species)
        """
        self.lines.append(ArcLineResiduals(fiberId, wavelength, x, y, xOrig, yOrig, xErr, yErr,
                                           flag, status, description))

    @property
    def xOrig(self):
        """Array of original x position (`numpy.ndarray` of `float`)"""
        return np.array([ll.xOrig for ll in self.lines])

    @property
    def yOrig(self):
        """Array of original y position (`numpy.ndarray` of `float`)"""
        return np.array([ll.yOrig for ll in self.lines])

    @classmethod
    def readFits(cls, filename):
        """Read from FITS file

        Not implemented, because we don't expect to write this.
        """
        raise NotImplementedError("Not implemented")

    def writeFits(self, filename):
        """Write to FITS file

        Not implemented, because we don't expect to write this.
        """
        raise NotImplementedError("Not implemented")


class FitDifferentialDetectorMapConfig(FitGlobalDetectorMapTask.ConfigClass):
    """Configuration for FitDifferentialDetectorMapTask"""
    base = Field(dtype=str,
                 doc="Template for base detectorMap; should include '%%(arm)s' and '%%(spectrograph)s'",
                 default=os.path.join(getPackageDir("drp_pfs_data"), "detectorMap",
                                      "detectorMap-sim-%(arm)s%(spectrograph)s.fits")
                 )

    def setDefaults(self):
        super().setDefaults()
        self.doSlitOffsets = False  # Slit offsets should be part of the base, not the global model
        self.order = 4  # Can use a lower order because we're fitting residuals


class FitDifferentialDetectorMapTask(FitGlobalDetectorMapTask):
    ConfigClass = FitDifferentialDetectorMapConfig

    def run(self, dataId, bbox, lines, visitInfo, metadata=None, base=None):
        """Fit a DifferentialDetectorMap to arc line measurements

        Parameters
        ----------
        dataId : `dict`
            Data identifier. Should contain at least ``arm`` (`str`; one of
            ``b``, ``r``, ``n``, ``m``) and ``spectrograph`` (`int`).
        bbox : `lsst.geom.Box2I`
            Bounding box for detector.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        visitInfo : `lsst.afw.image.VisitInfo`
            Visit information for exposure.
        metadata : `lsst.daf.base.PropertyList`, optional
            DetectorMap metadata (FITS header).
        base : `pfs.drp.stella.SplinedDetectorMap`, optional
            Base detectorMap. If not provided, one pointed to in the config will
            be read in.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DifferentialDetectorMap`
            Mapping of fiberId,wavelength to x,y.
        model : `pfs.drp.stella.GlobalDetectorModel`
            Model that was fit to the data.
        xResid, yResid : `numpy.ndarray` of `float`
            Fit residual in x,y for each of the ``lines`` (pixels).
        xRms, yRms : `float`
            Residual RMS in x,y (pixels)
        chi2 : `float`
            Fit chi^2.
        soften : `float`
            Systematic error that was applied to measured errors (pixels).
        used : `numpy.ndarray` of `bool`
            Array indicating which lines were used in the fit.
        reserved : `numpy.ndarray` of `bool`
            Array indicating which lines were reserved from the fit.
        """
        arm = dataId["arm"]
        spectrograph = dataId["spectrograph"]
        doFitHighCcd = (arm != "n") and not self.config.forceSingleCcd
        fiberCenter = self.config.fiberCenter[spectrograph] if doFitHighCcd else 0
        if base is None:
            base = self.getBaseDetectorMap(dataId)
        if self.config.doSlitOffsets:
            self.measureSlitOffsets(base, lines)
        residuals = self.calculateBaseResiduals(base, lines)
        results = self.fitGlobalDetectorModel(bbox, residuals, doFitHighCcd, fiberCenter,
                                              seed=visitInfo.getExposureId())
        results.detectorMap = DifferentialDetectorMap(base, results.model, visitInfo, metadata)

        if self.debugInfo.lineQa:
            self.lineQa(lines, results.detectorMap)
        if self.debugInfo.wlResid:
            self.plotWavelengthResiduals(results.detectorMap, lines, results.used, results.reserved)
        return results

    def getBaseDetectorMap(self, dataId):
        """Provide the detectorMap on which this will be based

        We retrieve the detectorMap by filename (through the config). This
        might be upgraded later to use the butler.

        Parameters
        ----------
        dataId : `dict`
            Data identifier. Should include ``arm`` (`str`) and ``spectrograph``
            (`int`) in order to allow selection of different detectorMaps for
            different gratings.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Base detectorMap.
        """
        filename = self.config.base % dataId
        return DetectorMap.readFits(filename)

    def measureSlitOffsets(self, detectorMap, lines):
        """Measure slit offsets for base detectorMap

        The detectorMap is modified in-place.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Base detectorMap.
        lines : `ArcLineSet`
            Original line measurements (NOT the residuals).
        """
        good = (lines.flag == 0)
        sysErr = self.config.soften
        detectorMap.measureSlitOffsets(
            lines.fiberId[good], lines.wavelength[good],
            lines.x[good], lines.y[good],
            np.hypot(lines.xErr[good], sysErr), np.hypot(lines.yErr[good], sysErr)
        )

    def calculateBaseResiduals(self, detectorMap, lines):
        """Calculate position residuals w.r.t. base detectorMap

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Base detectorMap.
        lines : `ArcLineSet`
            Original line measurements (NOT the residuals).

        Returns
        -------
        residuals : `ArcLineResidualsSet`
            Arc line position residuals.
        """
        points = detectorMap.findPoint(lines.fiberId, lines.wavelength)
        residuals = ArcLineResidualsSet.empty()
        for ll, pp in zip(lines, points):
            residuals.append(ll.fiberId, ll.wavelength, ll.x - pp[0], ll.y - pp[1], ll.x, ll.y,
                             ll.xErr, ll.yErr, ll.flag, ll.status, ll.description)
        return residuals

    def fitScaling(self, bbox, lines, select):
        """Determine scaling for GlobalDetectorModel

        This overrides the `FitGlobalDetectorMapTask` implementation, because
        we want to get the values from ``xOrig,yOrig`` instead of ``x,y``.

        These are not fit parameters, but merely provides convenient scaling
        factors so that the fit parameters, and especially the slit offsets, are
        in units roughly approximating pixels.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box for detector.
        lines : `pfs.drp.stella.ArcLineResidualsSet`
            Arc line measurements.
        select : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` are to be fit.

        Returns
        -------
        scaling : `pfs.drp.stella.GlobalDetectorModelScaling`
            Scaling for model.
        """
        fiberFit = fitStraightLine(lines.fiberId[select], lines.xOrig[select])
        wlFit = fitStraightLine(lines.yOrig[select], lines.wavelength[select])
        return GlobalDetectorModelScaling(
            fiberPitch=np.abs(fiberFit.slope),  # pixels per fiber
            dispersion=wlFit.slope,  # nm per pixel,
            wavelengthCenter=wlFit.slope*bbox.getHeight()/2 + wlFit.intercept,
            minFiberId=lines.fiberId.min(),
            maxFiberId=lines.fiberId.max(),
            height=bbox.getHeight(),
            buffer=self.config.buffer,
        )
