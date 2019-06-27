import os
from types import SimpleNamespace
import pickle
import numpy as np

from lsst.pex.config import Config, Field, ConfigField, makeConfigClass, ConfigurableField, DictField
from lsst.pipe.base import Task
from pfs.drp.stella import DispersionCorrectionControl, ReferenceLine
from .findLines import FindLinesTask

__all__ = ["IdentifyConfig", "IdentifyLinesConfig", "IdentifyLinesTask",
           "OldIdentifyLinesConfig", "OldIdentifyLinesTask"]

IdentifyConfig = makeConfigClass(DispersionCorrectionControl, "IdentifyConfig")


class IdentifyLinesConfig(Config):
    """Configuration for IdentifyLinesTask"""
    findLines = ConfigurableField(target=FindLinesTask, doc="Find arc lines")
    matchRadius = Field(dtype=float, default=0.5, doc="Line matching radius (nm)")
    refExclusionRadius = Field(dtype=float, default=0.0,
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
            self.identifyLines(spec, detectorMap, lines)

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

        obsLines = self.findLines.run(spectrum).lines
        obsLines = sorted(obsLines, key=lambda xx: spectrum.spectrum[int(xx + 0.5)], reverse=True)
        used = set()
        matches = []
        for obs in obsLines:
            wl = detectorMap.findWavelength(spectrum.fiberId, obs)
            ref = min([rl for rl in lines if rl.wavelength not in used],
                      key=lambda rl: np.abs(rl.wavelength - wl))
            if np.abs(ref.wavelength - wl) > self.config.matchRadius:
                continue
            used.add(ref.wavelength)
            new = ReferenceLine(ref.description)
            new.wavelength = ref.wavelength
            new.guessedIntensity = ref.guessedIntensity
            new.guessedPosition = detectorMap.findPoint(spectrum.fiberId, ref.wavelength)[1]
            new.fitPosition = obs
            new.fitIntensity = spectrum.spectrum[int(obs + 0.5)]
            new.status = ReferenceLine.Status.FIT
            matches.append(new)
        self.log.info("Matched %d from %d observed and %d reference lines",
                      len(matches), len(obsLines), len(lines))
        spectrum.setReferenceLines(matches + [rl for rl in lines if rl.wavelength not in used])


class OldIdentifyLinesConfig(Config):
    doInteractive = Field(dtype=bool, default=False, doc="Identify lines interactively?")
    identify = ConfigField(dtype=IdentifyConfig, doc="Automated line identification")


class OldIdentifyLinesTask(IdentifyLinesTask):
    """Version of IdentifyLinesTask that preserves the pre-June 2019 behavior"""
    ConfigClass = OldIdentifyLinesConfig
    _DefaultName = "identifyLines"

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)

    def identifyLines(self, spectrum, detectorMap, lines):
        """Identify lines on the spectrum

        This uses ``Spectrum.identify`` to fit lines on the spectrum at the
        expected positions of reference lines (using the detectorMap). However,
        if there is no actual line in the spectrum at the position of the
        reference line (e.g., the line is fainter than the detection limit),
        this will centroid on noise peaks, compromising the wavelength solution.
        It is possible to specify a non-zero ``minIntensity`` when reading the
        line list with ``pfs.drp.stella.utils.readLineListFile``, but this
        requires changing the configuration according to the data (exposure
        time, lamp setup, etc.).

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

        if self.config.doInteractive:
            filename = "identification-%d.pickle" % (spectrum.fiberId,)
            InteractiveIdentification.run(filename, spectrum, detectorMap, lines)

        spectrum.identify(lines, self.config.identify.makeControl())


class InteractiveIdentification:
    """Interactive identification of spectral lines

    If the wavelength solution in the ``DetectorMap`` is significantly
    inaccurate, the automated identification of lines (``Spectrum.identify``)
    may fail or produce non-optimal results. This module allows an interactive
    (i.e., manual) identification of lines that will be fit and used to
    modify the ``DetectorMap``, allowing the subsequent automated line
    identification and wavelength calibration to succeed.

    The user is presented with a plot of the spectrum, with the expected
    location of lines marked. The user should alternately mark the actual
    position of the line with an ``x`` (mark the middle of the feature: no
    centroiding is performed) and the estimated position (the vertical line)
    with an ``r`` (the vertical line will turn red). Once sufficient lines have
    been marked, hit ``g`` to fit the line associations and update the
    ``DetectorMap``. Further lines can then be marked if necessary. The fitting
    order may be changed with ``o``. ``h`` brings up a help message. When done,
    ``ENTER`` quits.

    The main entry point is the ``run`` classmethod, which saves the
    line associations and automatically reloads them on future invocations.
    An alternative is to instantiate the class and run the ``interactive``
    method (which will not save/load the associations).

    Parameters
    ----------
    spectrum : `pfs.drp.stella.Spectrum`
        The spectrum on which lines are being identified.
    detectorMap : `pfs.drp.stella.DetectorMap`
        Mapping between x,y and fiber,wavelength.
    lines : `list` of `pfs.drp.stella.ReferenceLine`
        Reference lines.
    """
    def __init__(self, spectrum, detectorMap, lines):
        self.spectrum = spectrum
        self.detectorMap = detectorMap
        self.lines = [rl for rl in lines if np.isfinite(rl.guessedPosition)]
        self.num = len(self.spectrum.wavelength)
        self.order = 2

        self.calculatePositions()
        self.linePositions = np.array([rl.guessedPosition for rl in self.lines])
        self.wavelengths = np.array([rl.wavelength for rl in self.lines])

        self.fig = None
        self.axes = None
        self.linesDrawn = None

        self.selectedLine = None
        self.selectedPosition = None
        self.associations = []

    @classmethod
    def run(cls, filename, spectrum, detectorMap, lines):
        """Main entry point

        Attempts to use associations that were previously saved to file, but
        if that doesn't exist then the user is prompted to make associations
        interactively.

        Parameters
        ----------
        filename : `str`
            File for associations.
        spectrum : `pfs.drp.stella.Spectrum`
            The spectrum on which lines are being identified.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping between x,y and fiber,wavelength.
        lines : `list` of `pfs.drp.stella.ReferenceLine`
            Reference lines.
        """
        self = cls(spectrum, detectorMap, lines)
        if os.path.exists(filename):
            self.loadAssociations(filename)
        else:
            self.interactive()
            self.saveAssociations(filename)
        return self

    def saveAssociations(self, filename):
        """Save associations to file

        Parameters
        ----------
        filename : `str`
            File to which to save associations.
        """
        position = [assoc.position for assoc in self.associations]
        wavelength = [assoc.line.wavelength for assoc in self.associations]
        with open(filename, "wb") as fd:
            pickle.dump(SimpleNamespace(position=position, wavelength=wavelength), fd)

    def loadAssociations(self, filename):
        """Load associations from file

        Parameters
        ----------
        filename : `str`
            File to which to save associations.
        """
        with open(filename, "rb") as fd:
            data = pickle.load(fd)
        lineMap = {rl.wavelength: rl.guessedPosition for rl in self.lines}
        for position, wavelength in zip(data.position, data.wavelength):
            self.selectPosition(position)
            self.selectLine(lineMap[wavelength])
            self.checkAssociated()
        self.fit()

    def interactive(self):
        """Start interactive line identification"""
        import matplotlib.pyplot as plt
        self.fig = plt.figure()
        conxn = self.fig.canvas.mpl_connect('key_press_event', self.keyPressed)
        self.plot()
        print("Fiber %d, order %d; use rxgo; 'h' for help, ENTER when done" %
              (self.spectrum.fiberId, self.order))
        self.fig.show()
        self.fig.ginput(-1, 0, False)
        self.fig.canvas.mpl_disconnect(conxn)

    def keyPressed(self, event):
        """Matplotlib callback for pressing a key"""
        if event.key == "r":  # Marking a (r)eference line
            self.selectLine(event.xdata)
        elif event.key == "x":  # Marking where the line should be (x marks the spot)
            self.selectPosition(event.xdata)
        elif event.key == "g":  # (G)o fit the lines I've marked
            self.fit()
            self.calculatePositions()
            self.plot()
        elif event.key == "o":  # Change fitting (o)rder
            try:
                self.order = int(input("Old order = %d; New order: " % (self.order,)))
            except Exception:
                print("Ignoring Exception while reading order; current order is %d" % (self.order,))
        elif event.key == "h":  # (h)elp!
            print("r : select reference line nearest cursor")
            print("x : select position at cursor")
            print("o : change order")
            print("g : go fit it")
            print("ENTER : quit")

    def plot(self):
        """Plot the spectrum and line identifications"""
        self.fig.clear()
        self.axes = self.fig.add_subplot(1, 1, 1)
        self.axes.plot(np.arange(self.num), self.spectrum.spectrum, "b-")
        self.linesDrawn = []
        associated = set([assoc.line.wavelength for assoc in self.associations])
        for rl in self.lines:
            x = rl.guessedPosition
            xLow = max(0, int(x + 0.5) - 5)
            xHigh = min(self.num - 1, int(x + 0.5) + 5)
            y = 1.05*self.spectrum.spectrum[xLow:xHigh].max()
            if rl.wavelength in associated:
                color = "r"
            else:
                color = "k"
            vline = self.axes.axvline(x, ls=':', color=color)
            text = self.axes.text(x, y, rl.description, ha='center', color=color)
            self.linesDrawn.append((vline, text))

    def selectLine(self, position):
        """Select the line nearest to the position

        Parameters
        ----------
        position : `float`
            Pixel position close to a line identification.
        """
        index = None
        distance = np.inf
        for ii, rl in enumerate(self.lines):
            dd = abs(rl.guessedPosition - position)
            if dd < distance:
                index = ii
                distance = dd
        self.selectedLine = self.lines[index]
        if self.linesDrawn:
            vline, text = self.linesDrawn[index]
            vline.set_color("r")
            text.set_color("r")
        print("Selected line at %f" % (self.selectedLine.guessedPosition,))
        self.checkAssociated()

    def selectPosition(self, position):
        """Select the position the line should be

        Parameters
        ----------
        position : `float`
            Pixel position line should be.
        """
        self.selectedPosition = position
        print("Selected position %f" % (self.selectedPosition,))
        self.checkAssociated()

    def checkAssociated(self):
        """Check if an association has been made"""
        if self.selectedLine is not None and self.selectedPosition is not None:
            self.associations.append(SimpleNamespace(line=self.selectedLine, position=self.selectedPosition))
            print("Associated: %f --> %f" % (self.selectedLine.guessedPosition, self.selectedPosition))
            self.selectedLine = None
            self.selectedPosition = None

    def fit(self):
        """Fit associations and update the ``DetectorMap``

        We fit a Chebyshev polynomial using the associations the user has
        identified. We fit the residuals in the positions, and add the resultant
        fit into the ``DetectorMap``.
        """
        if len(self.associations) < self.order + 1:
            print("Insufficient associations (%d) for order (%d)" % (len(self.associations), self.order))
            return

        xx = np.array([assoc.position for assoc in self.associations])
        yObs = np.array([self.detectorMap.findWavelength(self.spectrum.fiberId, assoc.position) for
                         assoc in self.associations])
        yTrue = np.array([assoc.line.wavelength for assoc in self.associations])
        yy = yTrue - yObs

        corr = np.polynomial.chebyshev.Chebyshev.fit(xx, yy, self.order, domain=[0.0, float(self.num)])
        yFit = corr(xx)
        resid = yy - yFit
        print("Order %d fit RMS: %f from %d associations" % (self.order, resid.std(), len(self.associations)))
        newWavelength = self.detectorMap.getWavelength(self.spectrum.fiberId)
        newWavelength += corr(np.arange(self.num, dtype=np.float32))
        self.detectorMap.setWavelength(self.spectrum.fiberId, newWavelength)

    def calculatePositions(self):
        """Calculate positions of reference lines from the ``DetectorMap``"""
        for rl in self.lines:
            rl.guessedPosition = self.detectorMap.findPoint(self.spectrum.fiberId, rl.wavelength)[1]
