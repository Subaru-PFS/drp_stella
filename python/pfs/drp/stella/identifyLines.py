import numpy as np

import lsstDebug
from lsst.pex.config import Config, Field, ConfigurableField, DictField
from lsst.pipe.base import Task
from pfs.drp.stella import ReferenceLine
from .findLines import FindLinesTask

__all__ = ["IdentifyLinesConfig", "IdentifyLinesTask"]


class IdentifyLinesConfig(Config):
    """Configuration for IdentifyLinesTask"""
    findLines = ConfigurableField(target=FindLinesTask, doc="Find arc lines")
    matchRadius = Field(dtype=float, default=0.1, doc="Line matching radius (nm)")
    refExclusionRadius = Field(dtype=float, default=0.1,
                               doc="Minimum allowed wavelength difference between reference lines (nm)")
    refThreshold = DictField(keytype=str, itemtype=float,
                             doc="Lower limit to guessedIntensity for reference lines, by their description",
                             default={"HgI": 5.0})


class IdentifyLinesTask(Task):
    ConfigClass = IdentifyLinesConfig
    _DefaultName = "identifyLines"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("findLines")
        self.debugInfo = lsstDebug.Info(__name__)

        self.debugInfo = lsstDebug.Info(__name__)

        if self.debugInfo.display:
            from lsst.afw.display import Display
            if isinstance(self.debugInfo.frame, Display):
                self.debugInfo_display = self.debugInfo.frame
            else:
                self.debugInfo_display = Display(frame=self.debugInfo.frame or 1)
        else:
            self.debugInfo_display = None

        self.debug_fiberIds = self.debugInfo.fiberIds  # don't load it in an inner loop
        if not self.debug_fiberIds:
            self.debug_fiberIds = None

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
        nMatched = []
        nUnmatched = []
        for spec in spectra:
            if spec.isWavelengthSet():
                self.identifyLines(spec, detectorMap, lines)

                nMatched.append(sum((s.status & 0x1) for s in spec.referenceLines))
                nUnmatched.append(sum((s.status & 0x1) == 0 for s in spec.referenceLines))

        nMatched = np.array(nMatched, dtype=int)
        nReference = nMatched + nUnmatched

        # Different fibres see slightly slightly different numbers of reference lines,
        # so the number of reference lines reported isn't an int
        self.log.info("Matched %.1f +- %.1f (min %d, max %d) lines"
                      " out of %.1f reference lines for %d fibers",
                      np.mean(nMatched), np.std(nMatched, ddof=1),
                      min(nMatched), max(nMatched), np.mean(nReference),
                      len(nMatched))

    def identifyLines(self, spectrum, detectorMap, lines):
        """Identify lines on the spectrum

        This is done by first finding lines on the spectrum and then matching
        them against the reference lines (using the detectorMap's
        pixel->wavelength solution).

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum on which to identify lines.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        lines : `list` of `pfs.drp.stella.ReferenceLine`
            Reference lines.
        """
        minWl = spectrum.wavelength.min()
        maxWl = spectrum.wavelength.max()
        lines = [rl for rl in lines if rl.wavelength > minWl and rl.wavelength < maxWl]

        if self.config.refExclusionRadius > 0.0:
            lines = sorted(lines, key=lambda rl: rl.wavelength)
            wavelengths = np.array([rl.wavelength for rl in lines])
            dWl = wavelengths[1:] - wavelengths[:-1]
            keep = np.ones_like(wavelengths, dtype=bool)
            keep[:-1] &= dWl > self.config.refExclusionRadius
            keep[1:] &= dWl > self.config.refExclusionRadius
            lines = [rl for rl, kk in zip(lines, keep) if kk]

        for descr, threshold in self.config.refThreshold.items():
            lines = [rl for rl in lines if
                     not rl.description.startswith(descr) or rl.guessedIntensity > threshold]

        obsLines = self.findLines.runCentroids(spectrum)
        indices = sorted(list(range(len(obsLines.centroids))),
                         key=lambda ii: spectrum.spectrum[int(obsLines.centroids[ii] + 0.5)], reverse=True)
        candidates = [rl for rl in lines]

        if self.debugInfo_display:
            if self.debug_fiberIds is None or spectrum.fiberId in self.debug_fiberIds:
                for yc in obsLines.centroids:
                    xc = detectorMap.getXCenter(spectrum.fiberId, yc)
                    self.debugInfo_display.dot('+', xc, yc)

        matches = []
        for ii in indices:
            if not candidates:
                break
            obs = obsLines.centroids[ii]
            obsErr = obsLines.errors[ii]
            wl = detectorMap.findWavelength(spectrum.fiberId, obs)
            ref = min(candidates, key=lambda rl: np.abs(rl.wavelength - wl))
            if np.abs(ref.wavelength - wl) > self.config.matchRadius:
                continue
            candidates.remove(ref)
            new = ReferenceLine(ref.description)
            new.wavelength = ref.wavelength
            new.guessedIntensity = ref.guessedIntensity
            new.guessedPosition = detectorMap.findPoint(spectrum.fiberId, ref.wavelength)[1]
            new.fitPosition = obs
            new.fitPositionErr = obsErr
            new.fitIntensity = spectrum.spectrum[int(obs + 0.5)]
            new.status = ReferenceLine.Status.FIT
            matches.append(new)

        spectrum.setReferenceLines(matches + candidates)
