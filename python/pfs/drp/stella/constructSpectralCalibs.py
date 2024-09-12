import time
import getpass
import platform
import collections

from typing import Iterable, Dict, Any, Union

from lsst.pex.config import Field, ConfigurableField
import lsst.daf.base as dafBase
from lsst.daf.persistence import NoResults
from lsst.afw.detection import FootprintSet, Threshold
from lsst.afw.display import Display
from lsst.meas.algorithms import DoubleGaussianPsf
from lsst.pipe.base import Struct
import lsst.pipe.drivers.constructCalibs
from lsst.pipe.drivers.utils import getDataRef
from .repair import PfsRepairTask
from pfs.drp.stella.utils.psf import fwhmToSigma

__all__ = ["SpectralCalibConfig", "SpectralCalibTask"]


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
    setCalibHeader(calib.getMetadata(), self.calibName, [dataId["visit"] for dataId in dataIdList], outputId)


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


def setCalibHeader(header: Union[dafBase.PropertyList, Dict], calibName: str, visitList: Iterable[int],
                   outputId: Dict[str, Any]) -> None:
    """Set header keys for calibs

    We record the type, the time, the inputs, and the output.

    Parameters
    ----------
    header : `lsst.daf.base.PropertyList` or `dict`
        Header/metadata for calibration; modified.
    visitList : iterable of `int`
        List of visits for data that went into the calib.
    outputId : `dict` [`str`: POD]
        Data identifier for output. Should include at least ``spectrograph`` and
        ``arm``.
    """
    header["OBSTYPE"] = calibName  # Used by ingestCalibs.py

    now = time.localtime()
    header["CALIB_CREATION_DATE"] = time.strftime("%Y-%m-%d", now)
    header["CALIB_CREATION_TIME"] = time.strftime("%X %Z", now)
    try:
        hostname = platform.node()
    except Exception:
        hostname = None
    header["CALIB_CREATION_HOST"] = hostname if hostname else "unknown host"
    try:
        username = getpass.getuser()
    except Exception:
        username = None
    header["CALIB_CREATION_USER"] = username if username else "unknown user"

    # Clobber any existing CALIB_INPUT_*
    names = list(header.keys())
    for key in names:
        if key.startswith("CALIB_INPUT_"):
            header.remove(key)
    # Set new CALIB_INPUT_*
    for ii, vv in enumerate(sorted(set(visitList))):
        header[f"CALIB_INPUT_{ii}"] = vv

    header["CALIB_ID"] = " ".join(f"{key}={value}" for key, value in outputId.items())
    header["SPECTROGRAPH"] = outputId["spectrograph"]
    header["ARM"] = outputId["arm"]


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
        target=PfsRepairTask,
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
                                    fwhmToSigma(self.config.psfFwhm))
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

        self.log.info("Combining %d inputs %s" % (len(dataRefList), fullOutputId))
        return Struct(dataRefList=dataRefList, outputId=fullOutputId)
