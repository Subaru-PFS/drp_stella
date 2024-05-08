import numpy as np
from lsst.daf.base import PropertyList
from lsst.afw.image import makeExposure, setVisitInfoMetadata, RotType
from lsst.pex.config import ConfigurableField, Field, ListField
from pfs.datamodel import CalibIdentity
from pfs.drp.stella.adjustDetectorMap import AdjustDetectorMapTask
from pfs.drp.stella.centroidTraces import CentroidTracesTask, tracesToLines

from .blackSpotCorrection import BlackSpotCorrectionTask
from .constructSpectralCalibs import SpectralCalibConfig, SpectralCalibTask
from .datamodel.drp import PfsFiberNorms
from .extractSpectraTask import ExtractSpectraTask


def rotationMatrix(theta):
    """Compute a 2D rotation matrix for a given angle.

    Parameters
    ----------
    theta : `float`
        Rotation angle in radians.

    Returns
    -------
    matrix : `numpy.ndarray`
        A 2x2 rotation matrix.
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rotateCoordinatesAroundCenter(x, x0, y0, theta):
    """Rotate the given coordinates around the specified center.

    Parameters
    ----------
    x : `numpy.ndarray`
        The coordinates to be rotated, in the format
        ``[[x1, x2, ..., xn], [y1, y2, ..., yn]]``.
    x0, y0 : `float`
        The x and y coordinates of the center around which the rotation is
        performed.
    theta : `float`
        The rotation angle in radians.

    Returns
    -------
    xRot : `numpy.ndarray`
        The rotated coordinates in the same format as the input `x`.
    """
    rotation = rotationMatrix(theta)
    center = np.array(([x0], [y0]))
    xCentered = x - center
    xRot = np.matmul(rotation, xCentered)
    xRot += center
    return xRot


def poly2dScreen(coords, a, b, c):
    """Define a 2D polynomial function with cross terms.

    Parameters
    ----------
    coords : `tuple` of `numpy.ndarray`
        The x and y coordinates.
    a, b, c : `float`
        The coefficients of the 2D polynomial.

    Returns
    -------
    values : `numpy.ndarray`
        The values of the 2D polynomial at the given coordinates.
    """
    x, y = coords
    return a * x * y + b * x + c * y + 1


def simScreenResponse(x, y, insrot, params):
    """Simulate the screen response

    Parameters
    ----------
    x, y : `numpy.ndarray`
        PFI coordinates.
    insrot : `float`
        The instrument rotator angle in degrees.
    params : `numpy.ndarray`
        The screen response parameters.

    Returns
    -------
    values : `numpy.ndarray`
        The simulated screen response values.
    """
    x0 = params[0]
    y0 = params[1]

    coords = np.vstack((x, y))
    rotated = rotateCoordinatesAroundCenter(coords, x0, y0, np.deg2rad(insrot))
    # You have to remember that we computed this model by dividing twilight by the quartzes.
    # Providing the twilight is uniform, what you have left is actually the inverse of the screen response.
    return 1 / poly2dScreen(rotated, *params[2:])


class ConstructFiberNormsConfig(SpectralCalibConfig):
    """Configuration for fiberNorms construction"""

    centroidTraces = ConfigurableField(target=CentroidTracesTask, doc="Centroid traces")
    doAdjustDetectorMap = Field(dtype=bool, default=True, doc="Adjust detectorMap using trace positions?")
    adjustDetectorMap = ConfigurableField(target=AdjustDetectorMapTask, doc="Adjust detectorMap")
    traceSpectralError = Field(
        dtype=float, default=1.0, doc="Error in the spectral dimension to give trace centroids (pixels)"
    )
    extractSpectra = ConfigurableField(target=ExtractSpectraTask, doc="Extract spectra")
    blackspots = ConfigurableField(target=BlackSpotCorrectionTask, doc="Black spot correction")
    doScreen = Field(dtype=bool, default=True, doc="Apply flat-field screen response?")
    screenParams = ListField(
        dtype=float,
        default=[-1.25426776e-04, 1.04442944e-04, 2.42251468e-07, 8.29359712e-05, 1.28199568e-04],
        doc="Flat-field screen response parameters",
    )
    rejIter = Field(dtype=int, default=1, doc="Number of iterations for fiberNorms measurement")
    rejThresh = Field(dtype=float, default=4.0, doc="Threshold for rejection in fiberNorms measurement")

    def setDefaults(self):
        super().setDefaults()
        self.doCameraImage = False  # We don't produce 2D images
        self.adjustDetectorMap.minSignalToNoise = 0  # We don't measure S/N
        self.extractSpectra.minFracMask = 0.3


class ConstructFiberNormsTask(SpectralCalibTask):
    """Task to construct the fiberNorms"""

    ConfigClass = ConstructFiberNormsConfig
    _DefaultName = "fiberNorms"
    calibName = "fiberNorms"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("centroidTraces")
        self.makeSubtask("adjustDetectorMap")
        self.makeSubtask("extractSpectra")
        self.makeSubtask("blackspots")

    def combine(self, cache, struct, outputId):
        """Combine multiple exposures of a particular CCD and write the output

        Only the slave nodes execute this method.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        struct : `lsst.pipe.base.Struct`
            Parameters for the combination, which has the following components:

            - ``ccdName`` (`tuple`): Name tuple for CCD.
            - ``ccdIdList`` (`list`): List of data identifiers for combination.
            - ``scales``: Unused by this implementation.

        Returns
        -------
        outputId : `dict`
            Data identifier for combined image (exposure part only).
        """
        combineResults = super().combine(cache, struct, outputId)
        dataRefList = combineResults.dataRefList
        outputId = combineResults.outputId

        calib = self.combination.run(
            dataRefList, expScales=struct.scales.expScales, finalScale=struct.scales.ccdScale
        )
        exposure = makeExposure(calib)
        self.interpolateNans(exposure)

        detMap = dataRefList[0].get("detectorMap")
        pfsConfig = dataRefList[0].get("pfsConfig")
        arm = dataRefList[0].dataId["arm"]
        visitInfo = dataRefList[0].get("postISRCCD_visitInfo")

        if self.config.doAdjustDetectorMap:
            traces = self.centroidTraces.run(exposure, detMap, pfsConfig)
            lines = tracesToLines(detMap, traces, self.config.traceSpectralError)
            detMap = self.adjustDetectorMap.run(detMap, lines, arm, visitInfo.id).detectorMap
            dataRefList[0].put(detMap, "detectorMap_used")

        profiles = dataRefList[0].get("fiberProfiles")
        traces = profiles.makeFiberTracesFromDetectorMap(detMap)
        spectra = self.extractSpectra.run(exposure.maskedImage, traces, detMap).spectra
        self.blackspots.run(pfsConfig, spectra)

        fiberId = spectra.getAllFiberIds()
        pfsConfig = pfsConfig.select(fiberId=fiberId)
        assert np.all(pfsConfig.fiberId == fiberId)

        assert visitInfo.getRotType() == RotType.UNKNOWN  # We want the rotator value, not the position angle
        rotAngle = visitInfo.getBoresightRotAngle().asDegrees()
        if not np.isfinite(rotAngle):
            raise RuntimeError("Rotator angle is not finite")
        if self.config.doScreen:
            screen = simScreenResponse(
                pfsConfig.pfiCenter[:, 0],
                pfsConfig.pfiCenter[:, 1],
                rotAngle,
                self.config.screenParams,
            )
        else:
            screen = np.ones_like(pfsConfig.fiberId, dtype=float)

        # The "screen response" is the quartz flux divided by the twilight flux.
        # To get the twilight flux, we need to divide our quartz flux by the screen response.
        # Since the "norm" will be used to divide the quartz flux, we can multiply it by the screen response.

        # Calculate the mean normalization for each fiber
        # PfsFiberNorms allows a polynomial for each fiber, but we're using only a single value per fiber
        norms = np.ones((len(spectra), 1), dtype=float)
        for ii, ss in enumerate(spectra):
            bad = (ss.mask.array[0] & ss.mask.getPlaneBitMask("NO_DATA")) != 0
            bad |= ~np.isfinite(ss.flux) | ~np.isfinite(ss.norm)
            bad |= ~np.isfinite(ss.variance) | (ss.variance == 0)
            nn = ss.norm*screen[ii]
            flux = np.ma.masked_where(bad, ss.flux/nn)
            weights = np.ma.masked_where(bad, nn**2/ss.variance)
            error = np.sqrt(ss.variance)

            rejected = np.zeros_like(flux, dtype=bool)
            for _ in range(self.config.rejIter):
                median = np.ma.median(flux)
                rejected |= np.abs(flux - median) > self.config.rejThresh*error
                flux.mask |= rejected

            weights.mask |= rejected
            norms[ii, 0] = np.ma.average(flux, weights=weights)
            self.log.debug("Normalization of fiber %d is %f", ss.fiberId, norms[ii])

        self.log.info(
            "Median normalization is %.2f +- %.2f (min %.2f, max %.2f)",
            np.mean(norms),
            np.std(norms, ddof=1),
            np.min(norms),
            np.max(norms),
        )

        date = visitInfo.getDate()
        identity = CalibIdentity(
            obsDate=date.toPython().isoformat(),
            spectrograph=dataRefList[0].dataId["spectrograph"],
            arm=arm,
            visit0=dataRefList[0].dataId["visit"],
        )
        metadata = PropertyList()
        metadata["OBSTYPE"] = "fiberNorms"
        metadata["calibDate"] = date.toPython(date.UTC).strftime("%Y-%m-%d")
        setVisitInfoMetadata(metadata, visitInfo)

        fiberNorms = PfsFiberNorms(identity, spectra.fiberId, exposure.getHeight(), norms, metadata)
        self.recordCalibInputs(cache.butler, fiberNorms, struct.ccdIdList, outputId)
        cache.butler.put(fiberNorms, "fiberNorms", outputId)
        self.log.info("Wrote fiberNorms for %s", outputId)
