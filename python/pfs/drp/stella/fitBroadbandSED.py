from pfs.datamodel.pfsConfig import TargetType

from lsst.pex.config import Config, Field, DictField
from lsst.pipe.base import Task
from lsst.utils import getPackageDir

import numpy
import numpy.lib.recfunctions
import astropy.io.fits

import os


class FitBroadbandSEDConfig(Config):
    """Configuration for FitBroadbandSEDTask
    """

    fluxLibraryPath = Field(
        dtype=str,
        doc="Synthetic photometry table"
    )

    filterMappings = DictField(
        keytype=str, itemtype=str,
        default={
            "g": "HSCg", "r": "HSCr", "r2": "HSCr2", "i": "HSCi", "i2": "HSCi2",
            "z": "HSCz", "y": "HSCy"
        },
        doc="Conversion table from pfsConfig's filter names to those used by `fluxLibrary`"
    )

    def setDefaults(self):
        super().setDefaults()

        try:
            dataDir = getPackageDir("fluxmodeldata")
        except LookupError:
            # We don't make this an exception because this method is called
            # even when `fluxLibraryPath` is specified in a call to the
            # constructor.
            dataDir = None

        if dataDir is not None:
            self.fluxLibraryPath = os.path.join(
                dataDir, "broadband", "photometries.fits"
            )


class FitBroadbandSEDTask(Task):
    """Fit an observed SED with synthetic model SEDs
    """

    ConfigClass = FitBroadbandSEDConfig
    _DefaultName = "fitBroadbandSED"

    def runDataRef(self, dataRef):
        """For each spectrum,
        calculate probabilities of model SEDs matching the spectrum.

        Only spectra for which ``targetType == TargetType.FLUXSTD``
        are processed.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for merged spectrum.

        Returns
        -------
        broadbandPDF : `list` of `numpy.array` of `float`
            ``broadbandPDF[iSpectrum][iSED]`` is the normalized probability
            of the SED ``iSED`` matching the spectrum ``iSpectrum``.
            If the targetType of ``iSpectrum`` is not ``TargetType.FLUXSTD``,
            ``broadbandPDF[iSpectrum]`` is None.
        """
        pfsConfig = dataRef.get("pfsConfig")
        return self.run(pfsConfig)

    def run(self, pfsConfig):
        """For each spectrum,
        calculate probabilities of model SEDs matching the spectrum.

        Only spectra for which ``targetType == TargetType.FLUXSTD``
        are processed, and included in the returned list.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            Top-end configuration, for identifying sky fibers.

        Returns
        -------
        broadbandPDF : `list` of `numpy.array` of `float`
            ``broadbandPDF[iSpectrum][iSED]`` is the normalized probability
            of the SED ``iSED`` matching the spectrum ``iSpectrum``.
            If the targetType of ``iSpectrum`` is not ``TargetType.FLUXSTD``,
            ``broadbandPDF[iSpectrum]`` is None.
        """
        pdfs = []
        for targetType, filterNames, fiberFlux, fiberFluxErr \
                in zip(pfsConfig.targetType, pfsConfig.filterNames,
                       pfsConfig.fiberFlux, pfsConfig.fiberFluxErr):
            if targetType == TargetType.FLUXSTD:
                pdfs.append(self.getProbabilities(filterNames, fiberFlux, fiberFluxErr))
            else:
                pdfs.append(None)

        return pdfs

    def getProbabilities(self, filterNames, fiberFlux, fiberFluxErr):
        """Calculate a chi square and a probability for each model SED.

        Parameters
        ----------
        filterNames : `list` of `str`
            List of filters used to measure the fiber fluxes for each filter.
            e.g. ``["g", "i"]``.
        fiberFlux : `list` of `float`
            Array of fiber fluxes for each fiber, in [nJy].
        fiberFluxErr : `list` of `float`
            Array of fiber flux errors for each fiber in [nJy].

        Returns
        -------
        broadbandPDF : `numpy.array` of `float`
            Array of normalized probabilities of the SED fitting.
        """
        if not (len(filterNames) == len(fiberFlux) == len(fiberFluxErr)):
            raise ValueError("Lengths of arguments must be equal.")
        if not filterNames:
            nSEDs = len(self.fluxLibrary)
            return numpy.full(shape=(nSEDs,), fill_value=1.0/nSEDs, dtype=float)

        # Convert filter names.
        filterNames = [self.config.filterMappings.get(f, f) for f in filterNames]

        # Note: fluxLibrary.shape == (nSEDs, nBands)
        fluxLibrary = numpy.lib.recfunctions.structured_to_unstructured(
            self.fluxLibrary[filterNames], dtype=float
        )

        # Note: observedFluxes.shape == (1, nBands)
        observedFluxes = numpy.asarray(fiberFlux, dtype=float).reshape(1, -1)
        # Note: observedNoises.shape == (1, nBands)
        observedNoises = numpy.asarray(fiberFluxErr, dtype=float).reshape(1, -1)
        # Note: numer.shape == (nSEDs, 1)
        numer = numpy.sum(fluxLibrary * (observedFluxes / (observedNoises**2)), axis=1, keepdims=True)
        # Note: denom.shape == (nSEDs, 1)
        denom = numpy.sum((fluxLibrary / observedNoises)**2, axis=1, keepdims=True)
        # Note: alpha.shape == (nSEDs, 1)
        alpha = numer / denom

        # Compute chi square
        # Note: chisq.shape == (nSEDs,)
        chisq = numpy.sum(((observedFluxes - alpha*fluxLibrary)/observedNoises)**2, axis=1)

        # Convert a list of chi squares to a probability distribution
        delta_chisq = chisq - numpy.min(chisq)
        prob = numpy.exp(delta_chisq / (-2.))
        prob_norm = prob / numpy.sum(prob)

        return prob_norm

    @property
    def fluxLibrary(self):
        """Table of the synthetic model SEDs

        Returns
        -------
        fluxLibrary : `numpy.array`
            A structured array,
            whose columns are SED parameters and various fluxes.
        """
        table = getattr(self, "_fluxLibrary", None)
        if table is None:
            # We convert FITS_rec back to numpy's structured array
            # because FITS_rec cannot be indexed with multiple column names.
            with astropy.io.fits.open(self.config.fluxLibraryPath) as fits:
                table = self._fluxLibrary = numpy.asarray(fits[1].data)
        return table
