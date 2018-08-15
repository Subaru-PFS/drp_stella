from lsst.pex.config import makeConfigClass
from lsst.pipe.base import Task
from pfs.drp.stella import DispersionCorrectionControl

__all__ = ["IdentifyLinesConfig", "IdentifyLinesTask"]


IdentifyLinesConfig = makeConfigClass(DispersionCorrectionControl, "IdentifyConfig")


class IdentifyLinesTask(Task):
    ConfigClass = IdentifyLinesConfig
    _DefaultName = "identifyLines"

    def run(self, spectra, detectorMap, lines):
        """Identify arc lines on the extracted spectra

        Parameters
        ----------
        spectra : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        lines : `list` of `pfs.drp.stella.ReferenceLine`
            Reference lines.
        """
        for spec in spectra:
            try:
                self.identifyLines(spec, detectorMap, lines)
            except Exception as exc:
                self.log.warn("Failed to identify lines for fiberId %d: %s" % (spec.getFiberId(), exc))
                continue

    def identifyLines(self, spectrum, detectorMap, lines):
        """Identify lines on the spectrum

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum on which to identify lines.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        lines : `list` of `pfs.drp.stella.ReferenceLine`
            Reference lines.
        """
        for rl in lines:
            rl.guessedPosition = detectorMap.findPoint(spectrum.getFiberId(), rl.wavelength)[1]
        spectrum.identify(lines, self.config.makeControl())
