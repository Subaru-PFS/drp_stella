import warnings
import numpy as np
import astropy.modeling

from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task

import lsstDebug


class FindLinesConfig(Config):
    """Configuration for FindLinesTask"""
    threshold = Field(dtype=float, default=5.0, doc="Threshold for line detection (sigma)")
    mask = ListField(dtype=str, default=[], doc="Mask planes to ignore")
    width = Field(dtype=float, default=1.0, doc="Guess width of line (stdev, pixels)")
    fittingRadius = Field(dtype=float, default=10.0,
                          doc="Radius of fitting region for centroid as a multiple of 'width'")
    exclusionRadius = Field(dtype=float, default=2.0,
                            doc="Fit exclusion radius for pixels around other peaks, "
                                "as a multiple of 'width'")


class FindLinesTask(Task):
    ConfigClass = FindLinesConfig

    def run(self, spectrum):
        """Find and centroid peaks in a spectrum

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum on which to find peaks.

        Returns
        -------
        centroids : `list` of `float`
            Centroid for each line.
        """
        peaks = self.findPeaks(spectrum)
        return self.centroidLines(spectrum, peaks)

    def findPeaks(self, spectrum):
        """Find positive peaks in the spectrum

        Peak flux must exceed ``threshold`` config parameter.

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum on which to find peaks.

        Returns
        -------
        indices : `numpy.ndarray` of `int`
            Indices of peaks.
        """
        flux = spectrum.spectrum
        with np.errstate(invalid='ignore'):
            stdev = np.sqrt(spectrum.variance)
            diff = flux[1:] - flux[:-1]  # flux[i + 1] - flux[i]
            select = (diff[:-1] > 0) & (diff[1:] < 0)  # A positive peak
            select &= flux[1:-1]/stdev[1:-1] > self.config.threshold  # Over threshold
        if self.config.mask:
            maskVal = spectrum.mask.getPlaneBitMask(self.config.mask)
            mask = spectrum.mask.array[0]
            select &= ((mask[:-2] | mask[1:-1] | mask[2:]) & maskVal) == 0  # Not masked either side

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

    def centroidLines(self, spectrum, peaks):
        """Centroid lines in a spectrum

        We fit a Gaussian plus a linear background to the ``centroidRadius``
        pixels either side of the peak.

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum on which to fit the peak.
        peaks : `numpy.ndarray` of `int`
            Indices of peaks.

        Returns
        -------
        centroids : `list` of `float`
            Centroid for each line.
        """
        exclusionRadius = int(self.config.exclusionRadius*self.config.width + 0.5)
        fittingRadius = int(self.config.fittingRadius*self.config.width + 0.5)
        flux = spectrum.spectrum
        mask = spectrum.mask.array[0]
        maskVal = spectrum.mask.getPlaneBitMask(self.config.mask)
        centroids = []
        for pp in peaks:
            amplitude = flux[pp]
            lowIndex = max(pp - fittingRadius, 0)
            highIndex = min(pp + fittingRadius, len(spectrum))
            indices = np.arange(lowIndex, highIndex)
            interlopers = np.nonzero((peaks >= lowIndex - exclusionRadius) &
                                     (peaks < highIndex + exclusionRadius) &
                                     (peaks != pp))[0]
            good = np.ones_like(indices, dtype=bool)
            for ii in peaks[interlopers]:
                lowBound = max(lowIndex, ii - exclusionRadius) - lowIndex
                highBound = min(highIndex, ii + exclusionRadius) - lowIndex
                good[lowBound:highBound] = False
            good &= (mask[lowIndex:highIndex] & maskVal) == 0
            if good.sum() < 5:
                continue

            lineModel = astropy.modeling.models.Gaussian1D(amplitude, pp, self.config.width,
                                                           bounds={"mean": (lowIndex, highIndex)},
                                                           name="line")
            bgModel = astropy.modeling.models.Linear1D(0.0, 0.0, name="bg")
            fitter = astropy.modeling.fitting.LevMarLSQFitter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
