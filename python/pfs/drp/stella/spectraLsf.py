"""The line-spread function of a set of spectra (the ``lsf`` of a pfsArm, pfsMerged or pfsCoadd)

`SpectraLsf` is the interface: the LSF of each spectrum in a set, in that
set's pixels, at positions given in pixels (``astropy.units.pix``) or as
vacuum wavelengths. It has two implementations:

- `BoundArmLsf`: the LSF of a detector (`pfs.drp.stella.armLsf.ArmLsf`, or the
  `pfs.drp.stella.armLsf.GaussianArmLsf` placeholder) bound to the wavelengths
  of a set of spectra from it (a pfsArm).
- `CombinedSpectraLsf`: the LSFs of combined spectra (a pfsMerged or pfsCoadd):
  a `pfs.drp.stella.CombinedLsf` for each, whose inputs are the LSFs of the
  inputs' detectors (`ArmFiberLsf`).

`ArmFiberLsf` adapts one fibre of a detector's LSF to the older
`pfs.drp.stella.Lsf` API, which `CombinedLsf` and the deprecated ``pfsArmLsf``,
``pfsMergedLsf`` and ``pfsCoaddLsf`` products use.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import astropy.units as u
import numpy as np
import scipy.optimize
import scipy.special

from pfs.datamodel.pfsLsf import PfsLsf, PfsLsfInput

from .armLsf import GaussianArmLsf, LsfKernel, detectorLsfFromDatamodel, dispersionAtRow, rowAtWavelength
from .combinedLsf import CombinedLsf, CombinedLsfInput, IndexMap, SmoothWeight
from .lsf import Kernel1D, Lsf, LsfDict, convolveInBlocks

__all__ = ("SpectraLsf", "BoundArmLsf", "CombinedSpectraLsf", "ArmFiberLsf", "spectraLsfFromDatamodel",
           "armFiberLsfs", "combinedSpectraLsf")

DEFAULT_HALF_WIDTH = 12*u.pix  # Default half-width of kernels


class SpectraLsf(ABC):
    """The LSF of each spectrum in a set of spectra

    Spectra are identified by ``key``: their ``fiberId`` (pfsArm, pfsMerged)
    or their `pfs.datamodel.Target` (pfsCoadd). Positions along a spectrum are
    pixels (``u.pix``) or vacuum wavelengths (e.g. ``u.nm``); a plain number is
    taken to be pixels.
    """

    @property
    @abstractmethod
    def keys(self) -> list:
        """The spectra covered"""
        raise NotImplementedError("Subclasses must implement")

    @property
    @abstractmethod
    def length(self) -> int:
        """Number of pixels in each spectrum"""
        raise NotImplementedError("Subclasses must implement")

    def __contains__(self, key) -> bool:
        return key in set(self.keys)

    @abstractmethod
    def getWavelength(self, key) -> np.ndarray:
        """Vacuum wavelength (nm) of each pixel of a spectrum"""
        raise NotImplementedError("Subclasses must implement")

    def toPixels(self, key, position) -> np.ndarray:
        """Position(s) in a spectrum's pixels

        Parameters
        ----------
        key : `int` or `pfs.datamodel.Target`
            The spectrum.
        position : `float`, array-like, or `astropy.units.Quantity`
            Pixels, or vacuum wavelength(s).

        Returns
        -------
        pixels : `float` or `numpy.ndarray`
            Position(s) in pixels (NaN outside the spectrum).
        """
        if isinstance(position, u.Quantity):
            if position.unit.is_equivalent(u.pix):
                return position.to_value(u.pix)
            return rowAtWavelength(self.getWavelength(key), position.to_value(u.nm))
        return np.asarray(position, dtype=float) if np.ndim(position) else float(position)

    def computeKernel(self, key, position, halfWidth=DEFAULT_HALF_WIDTH) -> LsfKernel:
        """The LSF of a spectrum for a line at a position

        Parameters
        ----------
        key : `int` or `pfs.datamodel.Target`
            The spectrum.
        position : `float`, array-like, or `astropy.units.Quantity`
            Position(s) of the line's first moment: pixels, or vacuum
            wavelength(s). An array gives a batch of kernels.
        halfWidth : `astropy.units.Quantity`
            The kernel covers ``position +/- halfWidth`` (pixels).

        Returns
        -------
        kernel : `LsfKernel`
            The LSF, normalised to unit sum, with the vacuum wavelength of each
            pixel.
        """
        center = self.toPixels(key, position)
        half = halfWidth.to_value(u.pix) if isinstance(halfWidth, u.Quantity) else float(halfWidth)
        if np.ndim(center) == 0:
            if not np.isfinite(center):
                raise ValueError(f"Position {position} is outside spectrum {key}")
            kernel = self._computeKernel(key, float(center), half)
        else:
            kernels = [self._computeKernel(key, float(cc), half) for cc in np.ravel(center)]
            width = 2*int(np.ceil(half)) + 2
            start = np.ceil(np.ravel(center) - half - 0.5).astype(int)
            values = np.zeros((len(kernels), width))
            for ii, (kk, ss) in enumerate(zip(kernels, start)):
                lo = max(kk.start - ss, 0)
                hi = min(kk.stop - ss, width)
                values[ii, lo:hi] = kk.values[lo - (kk.start - ss):hi - (kk.start - ss)]
            norm = np.array([kk.norm for kk in kernels])
            kernel = LsfKernel(key, np.ravel(center), start, values, norm=norm)
        wavelength = self.getWavelength(key)
        kernel.wavelength = kernel.extract(np.append(wavelength, np.nan))*u.nm if kernel.isBatch else \
            wavelength[kernel.slice]*u.nm
        return kernel

    @abstractmethod
    def _computeKernel(self, key, center: float, halfWidth: float) -> LsfKernel:
        """The LSF of a spectrum at a position (pixels), clipped to the spectrum"""
        raise NotImplementedError("Subclasses must implement")

    @abstractmethod
    def getLegacyLsf(self, key) -> Lsf:
        """The LSF of a spectrum, with the older `pfs.drp.stella.Lsf` API"""
        raise NotImplementedError("Subclasses must implement")

    def convolve(self, key, flux: np.ndarray) -> np.ndarray:
        """A spectrum (e.g. a model, on the same pixels) convolved with the LSF

        Parameters
        ----------
        key : `int` or `pfs.datamodel.Target`
            The spectrum.
        flux : `numpy.ndarray`
            Array of the spectrum's length.

        Returns
        -------
        convolved : `numpy.ndarray`
            ``flux`` convolved with the (varying) LSF.
        """
        return convolveInBlocks(self.getLegacyLsf(key), np.asarray(flux, dtype=float))

    def computeWidth(self, key, position, method="fit", intrinsicWidth=0*u.pix) -> u.Quantity:
        """The width of the Gaussian equivalent to the LSF

        Parameters
        ----------
        key : `int` or `pfs.datamodel.Target`
            The spectrum.
        position : `float`, array-like, or `astropy.units.Quantity`
            Position(s): pixels, or vacuum wavelength(s).
        method : `str`
            ``"fit"`` (the least-squares Gaussian: for unresolved features) or
            ``"moment"`` (the second moment: for features much wider than the
            LSF).
        intrinsicWidth : `astropy.units.Quantity`
            For ``"fit"``, the rms of the features (pixels or wavelength).

        Returns
        -------
        width : `astropy.units.Quantity`
            RMS of the equivalent Gaussian, in pixels, or in nm if ``position``
            is a wavelength.
        """
        center = np.atleast_1d(self.toPixels(key, position))
        wavelength = self.getWavelength(key)
        dispersion = dispersionAtRow(wavelength, center)
        if isinstance(intrinsicWidth, u.Quantity) and not intrinsicWidth.unit.is_equivalent(u.pix):
            intrinsic = intrinsicWidth.to_value(u.nm)/np.median(dispersion)
        else:
            intrinsic = u.Quantity(intrinsicWidth, u.pix).to_value(u.pix)
        sigma = self._computeWidth(key, center, method, intrinsic)
        inWavelength = isinstance(position, u.Quantity) and not position.unit.is_equivalent(u.pix)
        result = sigma*dispersion*u.nm if inWavelength else sigma*u.pix
        return result if np.ndim(position) else result[0]

    def _computeWidth(self, key, center: np.ndarray, method: str, intrinsicWidth: float) -> np.ndarray:
        """The width (pixels) of the Gaussian equivalent to the LSF at positions (pixels)

        This implementation measures the pixel-integrated kernels.
        """
        return np.array([kernelWidth(self._computeKernel(key, float(cc), DEFAULT_HALF_WIDTH.value),
                                     method, intrinsicWidth) for cc in center])

    def toLegacy(self) -> LsfDict:
        """The deprecated product (``pfsArmLsf``, ``pfsMergedLsf``, ``pfsCoaddLsf``)"""
        return LsfDict({key: self.getLegacyLsf(key) for key in self.keys})

    @abstractmethod
    def toDatamodel(self, keys: Optional[list] = None) -> PfsLsf:
        """The datamodel representation, for writing to FITS with the spectra

        Parameters
        ----------
        keys : `list`, optional
            The spectra in the order in which they are written, if they are
            identified by their index (the targets of a pfsCoadd).
        """
        raise NotImplementedError("Subclasses must implement")


class ArmFiberLsf(Lsf):
    """One fibre of a detector's LSF, with the older `pfs.drp.stella.Lsf` API

    Parameters
    ----------
    detectorLsf : `pfs.drp.stella.armLsf.ArmLsf` or `pfs.drp.stella.armLsf.GaussianArmLsf`
        LSF of the detector.
    fiberId : `int`
        The fibre.
    """

    def __init__(self, detectorLsf, fiberId):
        super().__init__(detectorLsf.length)
        self.detectorLsf = detectorLsf
        self.fiberId = int(fiberId)
        legacy = getattr(detectorLsf, "legacyLsf", None)
        self._legacy = legacy(self.fiberId) if legacy is not None else None  # For the Gaussian placeholder

    def computeKernel(self, center):
        if self._legacy is not None:
            return self._legacy.computeKernel(center)
        pixel = int(np.floor(center + 0.5))
        kernel = self.detectorLsf.computeKernel(self.fiberId, float(pixel), clip=False)
        return Kernel1D(kernel.values, pixel - kernel.start, normalize=False)  # Already normalised

    def computeWindow(self, center, clip=True):
        if self._legacy is not None:
            return self._legacy.computeWindow(center, clip=clip)
        kernel = self.detectorLsf.computeKernel(self.fiberId, float(center), clip=clip)
        return kernel.start, kernel.values

    def computeArray(self, center):
        if self._legacy is not None:
            return self._legacy.computeArray(center)
        start, values = self.computeWindow(center)
        array = np.zeros(self.length)
        array[start:start + len(values)] = values
        return array

    def convolve(self, array):
        if self._legacy is not None:
            return self._legacy.convolve(array)
        return convolveInBlocks(self, array)

    def warp(self, inWavelength, outWavelength):
        if self._legacy is not None:
            return self._legacy.warp(inWavelength, outWavelength)
        indexMap = IndexMap.fromWavelength(np.asarray(inWavelength), np.asarray(outWavelength))
        if indexMap is None:
            return CombinedLsf(len(outWavelength), [])
        weight = SmoothWeight.constant(indexMap.minIndex, indexMap.maxIndex)
        return CombinedLsf(len(outWavelength), [CombinedLsfInput(self, indexMap, weight, 1)])

    def __reduce__(self):
        return self.__class__, (self.detectorLsf, self.fiberId)


class BoundArmLsf(SpectraLsf):
    """The LSF of a detector, bound to the wavelengths of a set of its spectra

    Usually made with ``ArmLsf.forSpectra``, which checks that the spectra
    and the LSF belong together.

    Parameters
    ----------
    detectorLsf : `pfs.drp.stella.armLsf.ArmLsf` or `pfs.drp.stella.armLsf.GaussianArmLsf`
        LSF of the detector.
    fiberId : array-like of `int`
        The spectra's fibres.
    wavelength : `numpy.ndarray`, shape ``(len(fiberId), length)``
        The spectra's vacuum wavelengths (nm).
    """

    def __init__(self, detectorLsf, fiberId, wavelength):
        self.detectorLsf = detectorLsf
        self.fiberId = np.asarray(fiberId, dtype=int)
        self.wavelength = np.asarray(wavelength, dtype=float)
        self._index = {int(ff): ii for ii, ff in enumerate(self.fiberId)}

    @property
    def keys(self):
        return self.fiberId.tolist()

    @property
    def length(self):
        return self.detectorLsf.length

    def getWavelength(self, key):
        return self.wavelength[self._index[int(key)]]

    def _computeKernel(self, key, center, halfWidth):
        return self.detectorLsf.computeKernel(int(key), center, halfWidth)

    def getLegacyLsf(self, key):
        if int(key) not in self._index:
            raise KeyError(f"fiberId {key} not in the spectra")
        return ArmFiberLsf(self.detectorLsf, int(key))

    def _computeWidth(self, key, center, method, intrinsicWidth):
        sigma = self.detectorLsf.computeGaussianSigma(np.full(len(center), int(key)), center, method,
                                                      intrinsicWidth)
        return np.atleast_1d(sigma)

    def toDatamodel(self, keys=None):
        return PfsLsf([self.detectorLsf.toDatamodel(self.fiberId, includeWavelength=False)])

    def __reduce__(self):
        return self.__class__, (self.detectorLsf, self.fiberId, self.wavelength)

    def __repr__(self):
        return f"BoundArmLsf({self.detectorLsf!r}, {len(self.fiberId)} spectra)"


class CombinedSpectraLsf(SpectraLsf):
    """The LSFs of combined spectra (a pfsMerged or pfsCoadd)

    Parameters
    ----------
    lsfs : `dict` [`int` or `pfs.datamodel.Target`, `pfs.drp.stella.CombinedLsf`]
        LSF of each spectrum, by ``fiberId`` or target. The inputs of each
        `CombinedLsf` are `ArmFiberLsf`.
    wavelength : `dict` [`int` or `pfs.datamodel.Target`, `numpy.ndarray`]
        Vacuum wavelength (nm) of each spectrum's pixels.
    """

    def __init__(self, lsfs: Dict[object, CombinedLsf], wavelength: Dict[object, np.ndarray]):
        self.lsfs = dict(lsfs)
        self.wavelength = {key: np.asarray(wl, dtype=float) for key, wl in wavelength.items()}

    @property
    def keys(self):
        return list(self.lsfs)

    @property
    def length(self):
        return next(iter(self.lsfs.values())).length if self.lsfs else 0

    def getWavelength(self, key):
        return self.wavelength[key]

    def _computeKernel(self, key, center, halfWidth):
        start, values = self.lsfs[key].computeWindow(center, clip=True)
        pixels = np.arange(start, start + len(values))
        window = np.clip(halfWidth + 0.5 - np.abs(pixels - center), 0.0, 1.0)
        values = np.asarray(values, dtype=float)*window
        keep = np.flatnonzero(window > 0)
        if keep.size == 0:
            raise ValueError(f"No LSF at {center} in spectrum {key}")
        values = values[keep[0]:keep[-1] + 1]
        norm = float(np.sum(values))
        values = values/norm if norm > 0 else values
        return LsfKernel(key, float(center), int(start + keep[0]), values, norm=norm)

    def getLegacyLsf(self, key):
        return self.lsfs[key]

    def toDatamodel(self, keys=None):
        index = {key: ii for ii, key in enumerate(keys)} if keys is not None else None
        arms: List = []
        armIndex: Dict[int, int] = {}
        armFibers: Dict[int, set] = {}
        inputs = []
        for key, lsf in self.lsfs.items():
            spectrum = index[key] if index is not None else int(key)
            for inp in lsf.inputs:
                fiberLsf = inp.lsf
                if not isinstance(fiberLsf, ArmFiberLsf):
                    raise RuntimeError(f"Can't write an LSF whose input is a {type(fiberLsf).__name__}")
                detector = fiberLsf.detectorLsf
                if id(detector) not in armIndex:
                    armIndex[id(detector)] = len(arms)
                    arms.append(detector)
                    armFibers[id(detector)] = set()
                armFibers[id(detector)].add(fiberLsf.fiberId)
                inputs.append(PfsLsfInput(
                    spectrum=spectrum,
                    armIndex=armIndex[id(detector)],
                    fiberId=fiberLsf.fiberId,
                    order=int(inp.order),
                    indexPositions=inp.indexMap.positions,
                    indexSamples=inp.indexMap.samples,
                    indexMaxError=float(inp.indexMap.maxError),
                    weightMinIndex=inp.weight.minIndex,
                    weightMaxIndex=inp.weight.maxIndex,
                    weightBinSize=inp.weight.binSize,
                    weightValues=inp.weight.values,
                ))
        data = [detector.toDatamodel(np.array(sorted(armFibers[id(detector)]), dtype=int),
                                     includeWavelength=False) for detector in arms]
        return PfsLsf(data, inputs)

    def __reduce__(self):
        return self.__class__, (self.lsfs, self.wavelength)

    def __repr__(self):
        return f"CombinedSpectraLsf({len(self.lsfs)} spectra)"


def spectraLsfFromDatamodel(data: PfsLsf, keys: list, wavelength: Dict[object, np.ndarray],
                            length: int) -> SpectraLsf:
    """The LSF of a set of spectra from its datamodel representation

    Parameters
    ----------
    data : `pfs.datamodel.PfsLsf`
        Datamodel representation.
    keys : `list`
        The spectra, in the order in which they are written (their ``fiberId``,
        or their targets for a pfsCoadd).
    wavelength : `dict` [`int` or `pfs.datamodel.Target`, `numpy.ndarray`]
        Vacuum wavelength (nm) of each spectrum's pixels.
    length : `int`
        Number of pixels in each spectrum.

    Returns
    -------
    lsf : `SpectraLsf`
        The LSF of the spectra.
    """
    detectors = [detectorLsfFromDatamodel(arm) for arm in data.arms]
    if not data.isCombined:
        detector = detectors[0]
        fiberId = np.asarray(keys, dtype=int)
        return BoundArmLsf(detector, fiberId, np.array([wavelength[ff] for ff in keys]))

    # Identify the spectra: by fiberId, or by index for targets
    byIndex = not all(isinstance(key, (int, np.integer)) for key in keys)
    inputs: Dict[object, list] = {key: [] for key in keys}
    fiberLsfs: Dict[tuple, ArmFiberLsf] = {}
    for inp in data.inputs:
        key = keys[inp.spectrum] if byIndex else inp.spectrum
        if key not in inputs:
            continue
        fiberKey = (inp.armIndex, inp.fiberId)
        if fiberKey not in fiberLsfs:
            fiberLsfs[fiberKey] = ArmFiberLsf(detectors[inp.armIndex], inp.fiberId)
        indexMap = IndexMap(inp.indexPositions, inp.indexSamples, inp.indexMaxError)
        weight = SmoothWeight(inp.weightMinIndex, inp.weightMaxIndex, inp.weightBinSize, inp.weightValues)
        inputs[key].append(CombinedLsfInput(fiberLsfs[fiberKey], indexMap, weight, inp.order))
    lsfs = {key: CombinedLsf(length, ii) for key, ii in inputs.items() if ii}
    return CombinedSpectraLsf(lsfs, {key: wavelength[key] for key in lsfs})


def kernelWidth(kernel: LsfKernel, method: str = "fit", intrinsicWidth: float = 0.0) -> float:
    """The rms (pixels) of the Gaussian equivalent to a pixel-integrated kernel

    Parameters
    ----------
    kernel : `LsfKernel`
        The kernel (a single one).
    method : `str`
        ``"fit"`` (least-squares fit of a pixel-integrated Gaussian, with free
        centre and amplitude) or ``"moment"`` (second moment).
    intrinsicWidth : `float`
        For ``"fit"``, convolve with a Gaussian of this rms (pixels) before
        fitting, and remove it in quadrature afterwards.

    Returns
    -------
    sigma : `float`
        RMS (pixels).
    """
    values = np.asarray(kernel.values, dtype=float)
    offsets = kernel.pixels - kernel.center
    total = values.sum()
    if method == "moment":
        mean = np.sum(values*offsets)/total
        return float(np.sqrt(np.sum(values*(offsets - mean)**2)/total))
    if method != "fit":
        raise ValueError(f"Unknown method {method!r}: use 'fit' or 'moment'")
    if intrinsicWidth > 0:
        pad = int(np.ceil(5*intrinsicWidth))
        grid = np.arange(-pad, pad + 1)
        smooth = np.diff(scipy.special.ndtr((np.append(grid - 0.5, pad + 0.5))/intrinsicWidth))
        values = np.convolve(np.pad(values, pad), smooth, mode="same")
        offsets = np.arange(len(values)) - pad + kernel.start - kernel.center

    def model(params):
        amplitude, mean, sigma = params
        edges = np.append(offsets - 0.5, offsets[-1] + 0.5)
        return amplitude*np.diff(scipy.special.ndtr((edges - mean)/abs(sigma)))

    guess = np.sqrt(max(np.sum(values*offsets**2)/values.sum(), 0.1))
    result = scipy.optimize.least_squares(lambda pp: model(pp) - values, [values.sum(), 0.0, guess])
    sigma = abs(float(result.x[2]))
    return float(np.sqrt(max(sigma**2 - intrinsicWidth**2, 0.0)))


def armFiberLsfs(spectra, legacy=None) -> Dict[int, Lsf]:
    """The LSF of each fibre of a pfsArm, as inputs to a combination

    Parameters
    ----------
    spectra : `pfs.drp.stella.datamodel.PfsArm`
        Spectra of one arm, with their ``lsf`` if known.
    legacy : `dict` [`int`, `pfs.drp.stella.Lsf`], optional
        The deprecated ``pfsArmLsf``, used if the spectra have no ``lsf``
        (they were written before it existed).

    Returns
    -------
    lsfs : `dict` [`int`, `pfs.drp.stella.Lsf`]
        LSF of each fibre: `ArmFiberLsf` if the detector's LSF is known (so
        that the combination's LSF can be written with the spectra);
        otherwise the legacy LSFs.
    """
    lsf = getattr(spectra, "lsf", None)
    if isinstance(lsf, BoundArmLsf):
        return {int(ff): ArmFiberLsf(lsf.detectorLsf, ff) for ff in spectra.fiberId}
    if legacy is None:
        return {}
    identity = spectra.identity
    gaussian = GaussianArmLsf.fromLegacy(identity.arm, identity.spectrograph,
                                         {ff: ll for ff, ll in legacy.items() if ll is not None})
    if gaussian is None:
        return dict(legacy)
    return {int(ff): (ArmFiberLsf(gaussian, ff) if ll is not None else None) for ff, ll in legacy.items()}


def combinedSpectraLsf(lsfs: Dict[object, Lsf], wavelength: Dict[object, np.ndarray]
                       ) -> Optional[CombinedSpectraLsf]:
    """The LSF of combined spectra, if it can be written with them

    Parameters
    ----------
    lsfs : `dict` [`int` or `pfs.datamodel.Target`, `pfs.drp.stella.Lsf`]
        LSF of each combined spectrum (`pfs.drp.stella.CombinedLsf`).
    wavelength : `dict` [`int` or `pfs.datamodel.Target`, `numpy.ndarray`]
        Vacuum wavelength (nm) of each spectrum's pixels.

    Returns
    -------
    lsf : `CombinedSpectraLsf` or `None`
        The LSF of the spectra, or `None` if any of their inputs' LSFs isn't a
        detector's (an `ArmFiberLsf`), as for spectra reduced before the LSF
        was written with them.
    """
    if not lsfs:
        return None
    for lsf in lsfs.values():
        if not isinstance(lsf, CombinedLsf):
            return None
        if not all(isinstance(inp.lsf, ArmFiberLsf) for inp in lsf.inputs):
            return None
    return CombinedSpectraLsf(lsfs, wavelength)
