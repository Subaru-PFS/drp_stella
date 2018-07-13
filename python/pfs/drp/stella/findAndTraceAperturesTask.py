import lsst.pex.config as pexConfig
from lsst.pipe.base import Task
import pfs.drp.stella as drpStella


FiberTraceFunctionConfig = pexConfig.makeConfigClass(drpStella.FiberTraceFunctionControl)
FiberTraceFindingConfig = pexConfig.makeConfigClass(drpStella.FiberTraceFindingControl)
FiberTraceProfileFittingConfig = pexConfig.makeConfigClass(drpStella.FiberTraceProfileFittingControl)


class FindAndTraceAperturesConfig(pexConfig.Config):
    finding = pexConfig.ConfigField(dtype=FiberTraceFindingConfig, doc="Trace finding")
    function = pexConfig.ConfigField(dtype=FiberTraceFunctionConfig, doc="Interpolation function")
    fitting = pexConfig.ConfigField(dtype=FiberTraceProfileFittingConfig, doc="Profile fitting")


class FindAndTraceAperturesTask(Task):
    ConfigClass = FindAndTraceAperturesConfig
    _DefaultName = "findAndTraceApertures"

    def run(self, maskedImage, detectorMap):
        """Find and trace fibers on the image

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image on which to find and trace fibers.
        detectorMap : `pfs.drp.stella.detectorMap`
            Map of the canonical positions of the fibers.

        Returns
        -------
        traces : `pfs.drp.stella.FiberTraceSet`
            The fiber traces.
        """
        finding = self.config.finding.makeControl()
        function = self.config.function.makeControl()
        fitting = self.config.fitting.makeControl()
        return drpStella.findAndTraceApertures(maskedImage, detectorMap, finding, function, fitting)
