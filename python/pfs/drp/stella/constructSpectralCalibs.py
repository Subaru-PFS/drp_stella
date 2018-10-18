import math

from lsst.pex.config import Field, ConfigurableField
from lsst.daf.persistence import NoResults
from lsst.afw.detection import FootprintSet, Threshold
from lsst.afw.display import Display
from lsst.meas.algorithms import DoubleGaussianPsf
from lsst.pipe.base import Struct
from lsst.ctrl.pool.pool import NODE
from lsst.pipe.drivers.constructCalibs import CalibConfig, CalibTask
from lsst.pipe.drivers.utils import getDataRef
from lsst.pipe.tasks.repair import RepairTask

__all__ = ["SpectralCalibConfig", "SpectralCalibTask"]


class SpectralCalibConfig(CalibConfig):
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
        default=3.0,
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


class SpectralCalibTask(CalibTask):
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

    def getOutputId(self, expRefList, calibId):
        """Generate the data identifier for the output calib

        The mean date and the common filter are included, using keywords
        from the configuration.  The CCD-specific part is not included
        in the data identifier.

        This override implementation adds ``visit0`` to the output identifier.

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
        outputId = super().getOutputId(expRefList, calibId)
        outputId["visit0"] = expRefList[0].dataId['visit']
        return outputId

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
