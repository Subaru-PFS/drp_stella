import math
import collections

from lsst.pex.config import Field, ConfigurableField
import lsst.daf.base as dafBase
from lsst.daf.persistence import NoResults
from lsst.afw.detection import FootprintSet, Threshold
from lsst.afw.display import Display
from lsst.meas.algorithms import DoubleGaussianPsf
from lsst.pipe.base import Struct
from lsst.ctrl.pool.pool import NODE
import lsst.pipe.drivers.constructCalibs
from lsst.pipe.drivers.utils import getDataRef
from lsst.pipe.tasks.repair import RepairTask

__all__ = ["SpectralCalibConfig", "SpectralCalibTask"]


# Monkey-patching:
# * CalibTask.getOutputId to include visit0
# * CalibTask.recordCalibOutputs to include SPECTROGRAPH, ARM header keywords
_originalGetOutputId = lsst.pipe.drivers.constructCalibs.CalibTask.getOutputId
_originalRecordCalibInputs = lsst.pipe.drivers.constructCalibs.CalibTask.recordCalibInputs
_originalHeaderFromRaws = lsst.pipe.drivers.constructCalibs.CalibTask.calculateOutputHeaderFromRaws


def getOutputId(self, expRefList, calibId):
    """Generate the data identifier for the output calib

    This override implementation uses sub-day resolution for calibDate,
    downgrades multiple filters to a warning instead of an error, and
    adds ``visit0`` to the output identifier.

    Parameters
    ----------
    expRefList : `list` of `lsst.daf.persistence.ButlerDataRef`
        List of data references for input exposures.
    calibId : `dict`
        Data identifier elements for the calib, provided by the user.

    Returns
    -------
    outputId : `dict`
        Data identifier for output.
    """
    midTime = 0
    filterNames = collections.Counter()
    for expRef in expRefList:
        butler = expRef.getButler()
        dataId = expRef.dataId

        midTime += self.getMjd(butler, dataId)
        thisFilter = self.getFilter(butler, dataId) if self.filterName is None else self.filterName
        filterNames[thisFilter] += 1

    if len(filterNames) != 1:
        self.log.warn("Multiple filters specified for %s: %s" % (dataId, filterNames))

    midTime /= len(expRefList)
    date = dafBase.DateTime(midTime, dafBase.DateTime.MJD).toString(dafBase.DateTime.TAI)

    outputId = {self.config.filter: filterNames.most_common()[0][0],
                self.config.dateCalib: date}
    outputId.update(calibId)

    outputId["visit0"] = min(ref.dataId["visit"] for ref in expRefList)
    outputId["calibDate"] = date[:date.find("T")]  # Date only
    return outputId


def recordCalibInputs(self, butler, calib, dataIdList, outputId):
    """Record metadata including the inputs and creation details

    This metadata will go into the FITS header.

    Parameters
    ----------
    butler : `lsst.daf.persistence.Butler`
        Data butler.
    calib : `lsst.afw.image.Exposure`
        Combined calib exposure.
    dataIdList : iterable of `dict`
        List of data identifiers for calibration inputs.
    outputId : `dict`
        Data identifier for output.
    """
    _originalRecordCalibInputs(self, butler, calib, dataIdList, outputId)
    header = calib.getMetadata()
    header.set("SPECTROGRAPH", outputId["spectrograph"])
    header.set("ARM", outputId["arm"])


def getFilter(self, butler, dataId):
    """Determine the filter from a data identifier

    Querying the butler based on the visit doesn't yield the filter, because it
    can return multiple arms, and then we choose one essentially at random
    (whichever one happened to be ingested first), which results in filter
    mismatches. This is because we're working on a spectrograph rather than an
    imager with one filter per visit.

    Parameters
    ----------
    butler : `lsst.daf.persistence.Butler`
        Data butler.
    dataId: `dict`
        Data identifier for calibration input.

    Returns
    -------
    filterName : `NoneType`
        Name of filter, for which we use ``None``.
    """
    return None


def calculateOutputHeaderFromRaws(self, butler, calib, dataIdList, outputId):
    """Calculate the output header from the raw headers.

    This metadata will go into the output FITS header. It will include all
    headers that are identical in all inputs.

    This version removes the extraneous T00:00:00 from the end of DATE-OBS
    (there should already be a time).

    Parameters
    ----------
    butler : `lsst.daf.persistence.Butler`
        Data butler.
    calib : `lsst.afw.image.Exposure`
        Combined calib exposure.
    dataIdList : iterable of `dict` (`str`: POD)
        List of data identifiers for calibration inputs.
    outputId : `dict`
        Data identifier for output.
    """
    _originalHeaderFromRaws(self, butler, calib, dataIdList, outputId)
    header = calib.getMetadata()
    dateObs = header.get("DATE-OBS")
    if dateObs.endswith("T00:00:00.00"):
        fixed = dateObs[:dateObs.rfind("T")]
        if "T" in fixed:  # Make sure we haven't broken a good DATE-OBS
            header.set("DATE-OBS", fixed, comment="Start date of earliest input observation")


lsst.pipe.drivers.constructCalibs.CalibTask.getOutputId = getOutputId
lsst.pipe.drivers.constructCalibs.CalibTask.recordCalibInputs = recordCalibInputs
lsst.pipe.drivers.constructCalibs.CalibTask.getFilter = getFilter
lsst.pipe.drivers.constructCalibs.CalibTask.calculateOutputHeaderFromRaws = calculateOutputHeaderFromRaws


class PfsBiasTask(lsst.pipe.drivers.constructCalibs.BiasTask):
    """PFS-specialised bias construction

    Includes the above monkey-patched fixes. This is defined here to ensure
    the monkey-patching gets done (otherwise the ctrl_pool pickling will
    re-instantiate a BiasTask without the monkey-patching).
    """
    pass


class PfsDarkTask(lsst.pipe.drivers.constructCalibs.DarkTask):
    """PFS-specialised dark construction

    Includes the above monkey-patched fixes. This is defined here to ensure
    the monkey-patching gets done (otherwise the ctrl_pool pickling will
    re-instantiate a BiasTask without the monkey-patching).
    """
    pass


class SpectralCalibConfig(lsst.pipe.drivers.constructCalibs.CalibConfig):
    """Base configuration for constructing spectral calibs"""
    rerunISR = Field(
        dtype=bool,
        default=True,
        doc="Rerun ISR even if postISRCCD is available (may be e.g. not flat fielded)"
    )
    crGrow = Field(
        dtype=int,
        default=2,
        doc="Grow radius for CR (pixels)"
    )
    doRepair = Field(
        dtype=bool,
        default=True,
        doc="Repair artifacts?"
    )
    psfFwhm = Field(
        dtype=float,
        default=1.5,
        doc="Repair PSF FWHM (pixels)"
    )
    psfSize = Field(
        dtype=int,
        default=21,
        doc="Repair PSF size (pixels)"
    )
    repair = ConfigurableField(
        target=RepairTask,
        doc="Task to repair artifacts"
    )


class SpectralCalibTask(lsst.pipe.drivers.constructCalibs.CalibTask):
    """Base Task to construct a spectral calib

    The user still needs to set at least the following class variables:
    * ``_DefaultName``
    * ``calibName``
    """
    ConfigClass = SpectralCalibConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("repair")
        import lsstDebug
        self.debugInfo = lsstDebug.Info(__name__)

    @classmethod
    def applyOverrides(cls, config):
        """Config overrides to apply"""
        config.isr.doFringe = False

    def processSingle(self, sensorRef):
        """Process a single CCD

        Besides the regular ISR, also masks cosmic-rays.
        """
        if not self.config.rerunISR:
            try:
                exposure = sensorRef.get('postISRCCD')
                self.log.debug("Obtained postISRCCD from butler for %s" % sensorRef.dataId)
                return exposure
            except NoResults:
                pass  # ah well.  We'll have to run the ISR

        exposure = super().processSingle(sensorRef)

        if self.config.doRepair:
            psf = DoubleGaussianPsf(self.config.psfSize, self.config.psfSize,
                                    self.config.psfFwhm/(2*math.sqrt(2*math.log(2))))
            exposure.setPsf(psf)
            self.repair.run(exposure, keepCRs=False)
            if self.config.crGrow > 0:
                mask = exposure.getMaskedImage().getMask().clone()
                mask &= mask.getPlaneBitMask("CR")
                fpSet = FootprintSet(mask, Threshold(0.5))
                fpSet = FootprintSet(fpSet, self.config.crGrow, True)
                fpSet.setMask(exposure.getMaskedImage().getMask(), "CR")

        if self.debugInfo.display and self.debugInfo.inputsFrame >= 0:
            display = Display(frame=self.debugInfo.inputsFrame)
            display.mtv(exposure, "raw %(visit)d" % sensorRef.dataId)

        return exposure

    def combine(self, cache, struct, outputId):
        """!Combine multiple exposures of a particular CCD and write the output

        Only the slave nodes execute this method.

        This is a helper routine, containing just the start of what's needed to
        actually combine the inputs as an aid to subclasses. Note that
        thismethod does not return what the ``scatterCombine`` method expects;
        the user should call this method, and then return the combined exposure.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache. Contains a data butler (``butler``).
        struct : `lsst.pipe.base.Struct`
            Parameters for the combination, which has the following components:

            - ``ccdName`` (`tuple`): Name tuple for CCD.
            - ``ccdIdList`` (`list`): List of data identifiers for combination.
            - ``scales``: Unused by this implementation.

        Returns
        -------
        dataRefList : `dict`
            Data identifier for combined image (exposure part only).
        outputId : `dict`
            Fully-qualified data identifier for the output.
        """
        # Check if we need to look up any keys that aren't in the output dataId
        fullOutputId = {k: struct.ccdName[i] for i, k in enumerate(self.config.ccdKeys)}
        self.addMissingKeys(fullOutputId, cache.butler)
        fullOutputId.update(outputId)  # must be after the call to queryMetadata

        dataRefList = [getDataRef(cache.butler, dataId) if dataId is not None else None for
                       dataId in struct.ccdIdList]

        self.log.info("Combining %d inputs %s on %s" % (len(dataRefList), fullOutputId, NODE))
        return Struct(dataRefList=dataRefList, outputId=fullOutputId)
