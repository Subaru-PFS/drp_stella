import numpy as np

from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task, Struct

from pfs.datamodel.masks import MaskHelper


class CombineConfig(Config):
    """Configuration for CombineTask"""
    minWavelength = Field(dtype=float, default=350, doc="Minimum wavelength (nm)")
    maxWavelength = Field(dtype=float, default=1260, doc="Maximum wavelength (nm)")
    dWavelength = Field(dtype=float, default=0.1, doc="Spacing in wavelength (nm)")
    mask = ListField(dtype=str, default=["NO_DATA", "CR"], doc="Mask values to reject when combining")


class CombineTask(Task):
    """Combine spectra

    This can be done in order to merge arms (where different sensors have
    recorded measurements for the same wavelength, e.g., due to the use of a
    dichroic), or to combine spectra from different exposures. Both currently
    use the same simplistic placeholder algorithm.

    Note
    ----
    This involves resampling in wavelength.
    """
    ConfigClass = CombineConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        minWl = self.config.minWavelength
        maxWl = self.config.maxWavelength
        dWl = self.config.dWavelength
        self.wavelength = minWl + dWl*np.arange(int((maxWl - minWl)/dWl), dtype=float)

    def run(self, spectraList, identityKeys, SpectraClass):
        """Combine all spectra from the same exposure

        All spectra should have the same fibers, so we simply iterate over the
        fibers, combining each spectrum from that fiber.

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsSpectra`
            List of spectra to coadd.
        identityKeys : iterable of `str`
            Keys to select from the input spectra's ``identity`` for the
            combined spectra's ``identity``.
        SpectraClass : `type`, subclass of `pfs.datamodel.PfsSpectra`
            Class to use to hold result.

        Returns
        -------
        result : ``SpectraClass``
            Combined spectra.
        """
        archetype = spectraList[0]
        identity = {key: archetype.identity[key] for key in identityKeys}
        fiberId = archetype.fiberId
        if any(np.any(ss.fiberId != fiberId) for ss in spectraList):
            raise RuntimeError("Selection of fibers differs")

        flags = MaskHelper.fromMerge([ss.flags for ss in spectraList])

        wavelength = []
        flux = []
        sky = []
        covar = []
        mask = []
        for ii, ff in enumerate(fiberId):
            combination = self.combine(spectraList, [ff]*len(spectraList), flags)
            wavelength.append(self.wavelength.astype(archetype.wavelength.dtype))
            flux.append(combination.flux)
            sky.append(combination.sky)
            covar.append(combination.covar)
            mask.append(combination.mask)

        return SpectraClass(identity, fiberId, np.array(wavelength), np.array(flux), np.array(mask),
                            np.array(sky), np.array(covar), flags, archetype.metadata)

    def combine(self, spectraList, fiberId, flags):
        """Combine a single spectrum

        The spectrum to combine is specified by a list of (potentially
        different) fiberIds.

        This implementation is a placeholder, as the algorithm is overly
        simplistic: we blindly coadd the spectra, without any rejection or care
        with the covariances.

        Note
        ----
        This involves resampling in wavelength.

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsSpectra`
            List of spectra containers that include the spectrum to be combined.
        fiberId : iterable of `int`
            The fiber identifier for each of the spectra that specifies which
            spectrum is to be combined.
        flags : `pfs.datamodel.MaskHelper`
            Mask interpreter, for identifying bad pixels.

        Returns
        -------
        flux : `numpy.ndarray` of `float`
            Flux measurements for combined spectrum.
        sky : `numpy.ndarray` of `float`
            Sky measurements for combined spectrum.
        covar : `numpy.ndarray` of `float`
            Covariance matrix for combined spectrum.
        mask : `numpy.ndarray` of `int`
            Mask for combined spectrum.
        """
        wavelength = self.wavelength
        resampled = [ss.resample(wavelength, [ff]) for ss, ff in zip(spectraList, fiberId)]
        archetype = resampled[0]
        length = archetype.length
        mask = np.zeros(length, dtype=archetype.mask.dtype)
        flux = np.zeros(length, dtype=archetype.flux.dtype)
        sky = np.zeros(length, dtype=archetype.sky.dtype)
        covar = np.zeros((3, length), dtype=archetype.covar.dtype)
        numInputs = np.zeros(length, dtype=int)

        for ss in resampled:
            good = (ss.mask[0] & ss.flags.get(*self.config.mask)) == 0
            flux[good] += ss.flux[0, good]
            sky[good] += ss.sky[0, good]
            for ii in range(3):
                covar[ii][good] += ss.covar[0, ii][good]
            mask[good] |= ss.mask[0, good]
            numInputs[good] += 1

        good = numInputs > 0
        flux[good] /= numInputs[good]
        sky[good] /= numInputs[good]
        for ii in range(3):
            covar[ii][good] /= numInputs[good]
        mask[~good] = flags["NO_DATA"]

        return Struct(flux=flux, sky=sky, covar=covar, mask=mask)

    def runSingle(self, spectraList, fiberId, SpectrumClass, target, observations):
        """Combine a single spectrum from a list of spectra

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsSpectra`
            List of spectra that each contains the spectrum to coadd.
        fiberId : iterable of `int`
            The fiber identifier for each of the spectra that specifies which
            spectrum is to be combined.
        SpectrumClass : `type`, subclass of `pfs.datamodel.PfsSpectrum`
            Class to use to hold result.
        target : `pfs.datamodel.TargetData`
            Target of the observations.
        observations : iterable of `pfs.datamodel.TargetObservations`
            List of observations of the target.

        Returns
        -------
        result : ``SpectrumClass``
            Combined spectrum.
        """

        flags = MaskHelper.fromMerge([ss.flags for ss in spectraList])
        combination = self.combine(spectraList, fiberId, flags)
        covar2 = np.zeros((1, 1), dtype=combination.covar.dtype)
        return SpectrumClass(target, observations, self.wavelength, combination.flux, combination.mask,
                             combination.sky, combination.covar, covar2, flags)
