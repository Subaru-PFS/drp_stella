from lsst.pex.config import Field

from . import SplinedDetectorMap
from . import ReferenceLineStatus
from .DoubleDetectorMapContinued import DoubleDetectorMap
from .DistortedDetectorMapContinued import DistortedDetectorMap
from .MultipleDistortionsDetectorMapContinued import MultipleDistortionsDetectorMap
from .fitDistortedDetectorMap import FitDistortedDetectorMapTask, FitDistortedDetectorMapConfig

__all__ = ("AdjustDetectorMapConfig", "AdjustDetectorMapTask")


class AdjustDetectorMapConfig(FitDistortedDetectorMapConfig):
    """Configuration for AdjustDetectorMapTask"""
    fitStatic = Field(dtype=bool, default=False, doc="Fit static terms?")

    def setDefaults(self):
        super().setDefaults()
        self.order = 2  # Fit low-order coefficients only


class AdjustDetectorMapTask(FitDistortedDetectorMapTask):
    ConfigClass = AdjustDetectorMapConfig
    _DefaultName = "adjustDetectorMap"

    def run(self, detectorMap, lines, arm: str, seed=0):
        """Adjust a DistortedDetectorMap to fit arc line measurements

        We fit only the lowest order terms.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength --> x,y.
        lines : `pfs.drp.stella.ArcLineSet`
            Measured line positions. The ``status`` member will be updated to
            indicate which lines were used and reserved.
        arm : `str`
            Spectrograph arm in use (``b``, ``r``, ``n``, ``m``).
        seed : `int`
            Seed for random number generator used for selecting reserved lines.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Adjusted detectorMap.
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

        Raises
        ------
        pfs.drp.stella.fitDistortedDetectorMap.FittingError
            If the data is not of sufficient quality to fit.
        """
        base = self.getBaseDetectorMap(detectorMap, arm)  # NB: not SplinedDetectorMap
        Distortion = type(base.distortion)
        needNumLines = Distortion.getNumParametersForOrder(self.config.order)
        dispersion = base.getDispersionAtCenter(base.fiberId[len(base)//2])
        numGoodLines = self.getGoodLines(lines, detectorMap.getDispersionAtCenter()).sum()
        if numGoodLines < needNumLines:
            raise RuntimeError(f"Insufficient good lines: {numGoodLines} vs {needNumLines}")
        for ii in range(self.config.traceIterations):
            self.log.debug("Commencing trace iteration %d", ii)
            residuals = self.calculateBaseResiduals(base, lines)
            weights = self.calculateWeights(lines)
            fit = self.fitDistortion(
                detectorMap.bbox, residuals, weights, dispersion, seed=seed, fitStatic=self.config.fitStatic
            )
            detectorMap = self.constructAdjustedDetectorMap(base, fit.distortion)
            if not self.updateTraceWavelengths(lines, detectorMap):
                break

        results = self.measureQuality(lines, detectorMap, fit.selection, fit.numParameters)
        results.detectorMap = detectorMap

        lines.status[fit.selection] |= ReferenceLineStatus.DETECTORMAP_USED
        lines.status[fit.reserved] |= ReferenceLineStatus.DETECTORMAP_RESERVED

        if self.debugInfo.finalResiduals:
            self.plotResiduals(residuals, fit.xResid, fit.yResid, fit.selection, fit.reserved, dispersion)
        if self.debugInfo.lineQa:
            self.lineQa(lines, results.detectorMap)
        if self.debugInfo.wlResid:
            self.plotWavelengthResiduals(results.detectorMap, lines, fit.selection, fit.reserved)
        return results

    def getBaseDetectorMap(self, detectorMap, arm: str):
        """Get detectorMap to use as a base

        We need to ensure that the detectorMap is indeed a
        ``DistortionBasedDetectorMap`` with a sufficient polynomial order.
        We zero out the low-order coefficients to make it easy to construct the
        adjusted version.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength --> x,y.
        arm : `str`
            Spectrograph arm in use.

        Returns
        -------
        base : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength --> x,y.
        """
        visitInfo = detectorMap.visitInfo
        metadata = detectorMap.metadata
        # Promote other detectorMap classes to MultipleDistortionsDetectorMap
        if isinstance(detectorMap, SplinedDetectorMap):
            distortions = []
            base = detectorMap
        elif isinstance(detectorMap, (DoubleDetectorMap, DistortedDetectorMap)):
            distortions = [detectorMap.distortion.clone()]
            base = detectorMap.base
        elif isinstance(detectorMap, MultipleDistortionsDetectorMap):
            distortions = detectorMap.distortions
            base = detectorMap.base
        else:
            distortions = [dd.clone() for dd in detectorMap.distortions]
            base = detectorMap.base
        return MultipleDistortionsDetectorMap(base, distortions, visitInfo, metadata)

    def constructAdjustedDetectorMap(self, base, distortion):
        """Construct an adjusted detectorMap

        The adjusted detectorMap uses the low-order coefficients that we've just
        fit, and the high-order coefficients from the original detectorMap.

        Parameters
        ----------
        base : `pfs.drp.stella.MultipleDistortionsDetectorMap`
            Mapping from fiberId,wavelength --> x,y with low-order coefficients
            zeroed out.
        distortion : `pfs.drp.stella.Distortion`
            Low-order distortion to apply.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength --> x,y with both low- and
            high-order coefficients set.
        """
        return type(base)(base.base, [distortion] + base.distortions, base.visitInfo, base.metadata)
