import os
from types import SimpleNamespace
import pickle
import collections
import numpy as np
import astropy.modeling

import lsstDebug
from lsst.pex.config import Config, Field, ConfigField, makeConfigClass
from lsst.pipe.base import Task
from pfs.drp.stella import DispersionCorrectionControl, ReferenceLine

__all__ = ["IdentifyConfig", "IdentifyLinesConfig", "IdentifyLinesTask",
           "PairMatchIdentifyLinesConfig", "PairMatchIdentifyLinesTask"]

IdentifyConfig = makeConfigClass(DispersionCorrectionControl, "IdentifyConfig")


class IdentifyLinesConfig(Config):
    """Configuration for IdentifyLinesTask"""
    doInteractive = Field(dtype=bool, default=False, doc="Identify lines interactively?")
    identify = ConfigField(dtype=IdentifyConfig, doc="Automated line identification")


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
            self.identifyLines(spec, detectorMap, lines)
            self.plotIdentifications(spec)

    def plotIdentifications(self, spectrum):
        """Plot line identifications for spectrum

        If debugging option ``plotIdentifications`` is set.

        Parameters
        ----------
        spectrum : `pfs.drp.stella.SpectrumSet`
            Extracted spectrum. Includes the reference lines as an attribute.
        """

        if lsstDebug.Info(__name__).plotIdentifications:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)
            axes.plot(np.arange(len(spectrum)), spectrum.spectrum, "b-")
            for rl in spectrum.referenceLines:
                if rl.status & ReferenceLine.FIT:
                    x = rl.fitPosition
                    color = "r"
                else:
                    x = np.interp(rl.wavelength, spectrum.wavelength, np.arange(len(spectrum)))
                    color = "c"
                xLow = max(0, int(x + 0.5) - 5)
                xHigh = min(len(spectrum) - 1, int(x + 0.5) + 5)
                y = 1.05*spectrum.spectrum[xLow:xHigh].max()
                axes.axvline(x, ls='--', color=color)
                axes.text(x, y, rl.description, ha='center', color=color)
            for x in obsPixels:
                xLow = max(0, int(x + 0.5) - 5)
                xHigh = min(len(spectrum) - 1, int(x + 0.5) + 5)
                y = 1.05*spectrum.spectrum[xLow:xHigh].max()
                axes.axvline(x, ls=':', color="k")
            plt.show()

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

        if self.config.doInteractive:
            filename = "identification-%d.pickle" % (spectrum.fiberId,)
            InteractiveIdentification.run(filename, spectrum, detectorMap, lines)

        spectrum.identify(lines, self.config.identify.makeControl())


class PairMatchIdentifyLinesConfig(Config):
    minPeakFlux = Field(dtype=float, default=500.0, doc="Minimum flux for peak")
    width = Field(dtype=float, default=1.0, doc="Width of line (stdev, pixels)")
    centroidRadius = Field(dtype=int, default=10, doc="Radius of fitting region for centroid (pixels)")
    searchMinSep = Field(dtype=float, default=0.2,
                         doc="Minimum search radius for line pairs (nm). "
                             "This allows ignoring close pairs, which might get confused.")
    searchMaxSep = Field(dtype=float, default=100.0,
                         doc="Maximum search radius for line pairs (nm). "
                             "This prevents searching over a range where there are nonlinearities.")
    matchMeanTol = Field(dtype=float, default=5.0,
                         doc="Maximum difference of the means for a match (nm). "
                             "This is the maximum shift that is allowed from the original detectorMap.")
    matchDiffTol = Field(dtype=float, default=0.5,
                         doc="Maximum difference of the differences for a match (nm). "
                             "This is the maximum nonlinearity over the 'searchMaxSep' pixels.")
    voteFraction = Field(dtype=float, default=0.8,
                         doc="Fraction of votes required for selecting an association. "
                             "Too high, and you'll be affected by chance associations. "
                             "Too low, and you'll be overwhelmed by chance associations.")
    fitPositionErr = Field(dtype=float, default=0.01, doc="Typical centroid error (pixels)")


class PairMatchIdentifyLinesTask(IdentifyLinesTask):
    ConfigClass = PairMatchIdentifyLinesConfig

    def findPeaks(self, flux):
        """Find positive peaks in the flux

        Peak flux must exceed ``minPeakFlux``.

        Parameters
        ----------
        flux : `numpy.ndarray` of `float`
            Array of fluxes.

        Returns
        -------
        indices : `numpy.ndarray` of `int`
            Indices of peaks.
        """
        diff = flux[1:] - flux[:-1]  # flux[i + 1] - flux[i]
        select = (diff[:-1] > 0) & (diff[1:] < 0) & (flux[1:-1] > self.config.minPeakFlux)
        indices = np.nonzero(select)[0] + 1  # +1 to account for the definition of diff
        self.log.debug("Found peaks: %s", indices)

        if lsstDebug.Info(__name__).plotPeaks:
            import matplotlib.pyplot as plt
            figure, axes = plt.subplots()
            axes.plot(np.arange(len(flux)), flux, 'k-')
            for xx in indices:
                axes.axvline(xx, color="r", linestyle=":")
            plt.show()

        return indices

    def centroidLines(self, flux, peaks):
        """Centroid lines in a spectrum

        We fit a Gaussian plus a linear background to the ``centroidRadius``
        pixels either side of the peak.

        Parameters
        ----------
        flux : `numpy.ndarray` of `float`
            Array of fluxes.
        peaks : `numpy.ndarray` of `int`
            Indices of peaks.

        Returns
        -------
        centroids : `list` of `float`
            Centroid for each line.
        """
        interloperWidth = int(3*self.config.width + 0.5)
        centroids = []
        for pp in peaks:
            amplitude = flux[pp]
            lowIndex = max(pp - self.config.centroidRadius, 0)
            highIndex = min(pp + self.config.centroidRadius, len(flux))
            indices = np.arange(lowIndex, highIndex)
            interlopers = np.nonzero((peaks >= lowIndex - interloperWidth) &
                                     (peaks < highIndex + interloperWidth) &
                                     (peaks != pp))[0]
            good = np.ones_like(indices, dtype=bool)
            for ii in peaks[interlopers]:
                lowBound = max(lowIndex, ii - interloperWidth) - lowIndex
                highBound = min(highIndex, ii + interloperWidth) - lowIndex
                good[lowBound:highBound] = False
            if good.sum() < 5:
                continue

            lineModel = astropy.modeling.models.Gaussian1D(amplitude, pp, self.config.width,
                                                           bounds={"mean": (lowIndex, highIndex)},
                                                           name="line")
            bgModel = astropy.modeling.models.Linear1D(0.0, 0.0, name="bg")
            fitter = astropy.modeling.fitting.LevMarLSQFitter()
            fit = fitter(lineModel + bgModel, indices[good], flux[lowIndex:highIndex][good])
            center = fit["line"].mean.value

            if lsstDebug.Info(__name__).plotCentroidLines:
                import matplotlib.pyplot as plt
                fig = plt.figure()
                axes = fig.add_subplot(1, 1, 1)
                axes.plot(indices, flux[lowIndex:highIndex], "k-")
                if good.sum() != len(good):
                    axes.plot(indices[~good], flux[lowIndex:highIndex][~good], "rx")
                xx = np.arange(lowIndex, highIndex, 0.01)
                axes.plot(xx, fit(xx), "b--")
                axes.axvline(center, color="b", linestyle=":")
                axes.set_xlabel("Index")
                axes.set_ylabel("Flux")
                plt.show()

            if fitter.fit_info["ierr"] not in (1, 2, 3, 4):
                # Bad fit
                continue
            centroids.append(center)

        if lsstDebug.Info(__name__).plotCentroids:
            import matplotlib.pyplot as plt
            figure, axes = plt.subplots()
            indices = np.arange(len(flux))
            axes.plot(indices, flux, 'k-')
            for cc in centroids:
                axes.axvline(cc, color="r", linestyle=":")
            plt.show()

        return centroids

    def identifyLines(self, spectrum, detectorMap, lines):
        """Identify lines on the spectrum

        We match pairs of lines between the observed spectrum and the line list.
        The separation of lines in a pair is limited (between ``searchMinSep``
        and ``searchMaxSep``): if the separation is too small, lines can get
        confused due to centroiding noise; if the separation is too large, the
        wavelength solution may not be roughly linear. Pairs between the two
        lists are matched based on their wavelength difference
        (``matchDiffTol``; this is a threshold on the centroiding noise) and
        their wavelength mean (``matchMeanTol``; the maximum shift allowed from
        the original wavelength solution). Each matched pair then votes for each
        of the line associations in the pair. Finally, each line in the
        observed spectrum has a list of votes for the associated reference line
        (and vice-versa); if a specific association obtains a minimum fraction
        of votes (``voteFraction``), the association is considered real.

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum on which to identify lines.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        lines : `list` of `pfs.drp.stella.ReferenceLine`
            Reference lines.
        """
        # Find and centroid peaks on the observed spectrum
        peaks = self.findPeaks(spectrum.spectrum)
        obsPixels = self.centroidLines(spectrum.spectrum, peaks)
        self.log.info("Centroided %d lines", len(obsPixels))

        # Convert centroids to wavelength using guess detectorMap
        fiberId = spectrum.fiberId
        obsWavelength = [detectorMap.findWavelength(fiberId, xx) for xx in obsPixels]
        obsLines = dict(zip(obsWavelength, obsPixels))

        refLines = {rl.wavelength: rl for rl in lines}
        refWavelength = list(refLines.keys())

        def makePairs(wavelengths):
            """Make pairs of lines

            Pairs must have ``searchMinSep < separation < searchMaxSep``.

            Parameters
            ----------
            wavelengths : iterable of `float`
                Wavelengths of lines.

            Returns
            -------
            pairs : `list` of `types.SimpleNamespace`
                List of pairs; each has elements:
                - ``first`` (`float`): lower wavelength.
                - ``second`` (`float`): upper wavelength.
                - ``mean`` (`float`): mean wavelength.
                - ``diff`` (`float`): difference in wavelength.
            """
            wavelengths = sorted(wavelengths)
            num = len(wavelengths)
            pairs = []
            for ii, wl1 in enumerate(wavelengths):
                jj = ii + 1
                while jj < num and wavelengths[jj] < wl1 + self.config.searchMinSep:
                    jj += 1
                while jj < num and wavelengths[jj] < wl1 + self.config.searchMaxSep:
                    wl2 = wavelengths[jj]
                    pairs.append(SimpleNamespace(first=wl1, second=wl2, mean=0.5*(wl1 + wl2), diff=wl2 - wl1))
                    jj += 1
            return pairs

        # Construct pairs of observed lines and pairs of reference lines
        obsPairs = makePairs(obsWavelength)
        self.log.debug("Made %d pairs from observed wavelength", len(obsPairs))
        refPairs = makePairs(refWavelength)
        self.log.debug("Made %d pairs from reference wavelength", len(refPairs))
        refDiffs = np.array([pp.diff for pp in refPairs])
        refMeans = np.array([pp.mean for pp in refPairs])

        if lsstDebug.Info(__name__).plotPairs:
            obsDiffs = np.array([pp.diff for pp in obsPairs])
            obsMeans = np.array([pp.mean for pp in obsPairs])
            import matplotlib.pyplot as plt
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)
            axes.plot(obsMeans, obsDiffs, "bx")
            axes.plot(refMeans, refDiffs, "r+")
            axes.set_xlim(spectrum.wavelength.min(), spectrum.wavelength.max())
            axes.set_xlabel("Mean")
            axes.set_ylabel("Diff")
            plt.show()

        # Match pairs based on difference and mean
        obsAssociations = collections.defaultdict(collections.Counter)
        refAssociations = collections.defaultdict(collections.Counter)
        for obs in obsPairs:
            selectDiff = np.abs(refDiffs - obs.diff) < self.config.matchDiffTol
            selectMean = np.abs(refMeans - obs.mean) < self.config.matchMeanTol
            indices = np.nonzero(selectDiff & selectMean)[0]
            for ii in indices:
                ref = refPairs[ii]
                obsAssociations[obs.first][ref.first] += 1
                obsAssociations[obs.second][ref.second] += 1
                refAssociations[ref.first][obs.first] += 1
                refAssociations[ref.second][obs.second] += 1
        self.log.debug("Built associations for %d lines", len(obsAssociations))

        # Vote for associations: ref --> obs
        approvedRefs = {}
        for ref, counts in refAssociations.items():
            total = len(counts)
            bestObs, num = counts.most_common(1)[0]
            if num/total < self.config.voteFraction:
                continue
            approvedRefs[ref] = bestObs

        # Vote for associations: obs --> ref
        result = []
        for obs, counts in obsAssociations.items():
            total = len(counts)
            bestRef, num = counts.most_common(1)[0]
            if (bestRef not in approvedRefs or approvedRefs[bestRef] != obs or
                    num/total < self.config.voteFraction):
                continue
            self.log.debug("Identified line: %f --> %f", obs, bestRef)
            refLines[bestRef].fitPosition = obsLines[obs]
            refLines[bestRef].guessedPosition = obsLines[obs]
            refLines[bestRef].fitPositionErr = self.config.fitPositionErr
            refLines[bestRef].status |= ReferenceLine.FIT
            result.append(refLines[bestRef])
        spectrum.setReferenceLines(result)
        self.log.info("Identified %d lines", len(spectrum.referenceLines))


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
