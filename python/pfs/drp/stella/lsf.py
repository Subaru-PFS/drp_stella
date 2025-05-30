from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING
from collections.abc import Callable, Iterable, Iterator
import numbers
import pickle
import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from numpy.typing import NDArray, ArrayLike

from lsst.geom import Point2D, Point2I, Extent2I, Box2I
from lsst.afw.image import ImageF, ImageD, MaskedImageF
from lsst.afw.math import FixedKernel
from lsst.afw.geom.ellipses import Quadrupole, BaseCore

from .FiberTraceSetContinued import FiberTraceSet

__all__ = ("Kernel1D", "Lsf", "GaussianKernel1D", "GaussianLsf", "FixedEmpiricalLsf", "ExtractionLsf",
           "CoaddLsf", "LsfDict")

# Default interpolator factory
DEFAULT_INTERPOLATOR = partial(interp1d, kind="linear", bounds_error=False, fill_value=0, copy=True,
                               assume_sorted=True)

# Floating-point types
FloatingPoint = np.float32 | np.float64

# Interpolator type
Interpolator = Callable[[ArrayLike], ArrayLike]

# Interpolator factory type
InterpolatorFactory = Callable[[NDArray[FloatingPoint], NDArray[FloatingPoint]], Interpolator]

if TYPE_CHECKING:
    from pfs.drp.stella import SpectralPsf, FiberTrace


class Kernel1D:
    """A one-dimensional kernel

    A ``Kernel1D`` is essentially a one-dimensional array with an indexing
    offset, so that index 0 refers to the center of the kernel. Main operations
    are ``convolve`` to convolve an array by a kernel, and ``toArray`` which
    inserts the kernel at any point in an array.

    Parameters
    ----------
    values : `numpy.ndarray`, floating-point, shape ``(N,)``
        Array of kernel values.
    center : `int`
        Index of center of kernel in the ``values``. It needn't be the middle.
    normalize : `bool`, optional
        Normalize the kernel to unity?
    """
    def __init__(self, values: NDArray[FloatingPoint], center: int, normalize: bool = True):
        self.length = len(values)
        self.center = center
        self.values = values
        self.indices = np.arange(self.length) - center
        self.min = self.indices[0]
        self.max = self.indices[-1]
        if normalize:
            self.values /= self.normalization()

    def copy(self) -> "Kernel1D":
        """Return a copy of the kernel"""
        return self.__class__(self.values.copy(), self.center, normalize=False)

    @classmethod
    def makeEmpty(cls, halfSize: int) -> "Kernel1D":
        """Create an empty ``Kernel1D``

        Parameters
        ----------
        halfSize : `int`
            Half-size of the kernel.

        Returns
        -------
        self : ``cls``
            Empty kernel.
        """
        size = 2*halfSize + 1
        return cls(np.zeros(size, dtype=float), halfSize, normalize=False)

    def __len__(self) -> int:
        """Number of elements in kernel array"""
        return self.length

    def __getitem__(self, index: int) -> float:
        """Get value(s) from the kernel

        Supports integer or array indexing.

        Values off the end are zero.
        """
        if isinstance(index, numbers.Integral):
            return self.values[index + self.center]
        ii, vv = np.broadcast_arrays(index, self.values)
        return vv[ii + self.center]

    def __setitem__(self, index: int, value: FloatingPoint):
        """Set value(s) in the kernel.

        Supports integer or array indexing.

        Values off the end raise an `IndexError`.
        """
        ii, vv = np.broadcast_arrays(index, value)
        self.values[ii + self.center] = vv

    def __iter__(self) -> Iterator[tuple[int, FloatingPoint]]:
        """Iterator

        Produces ``index,value`` pairs.
        """
        return zip(iter(self.indices), iter(self.values))

    def _getOtherValues(self, other: "Kernel1D | NDArray[FloatingPoint]") -> NDArray[FloatingPoint]:
        """Get values from another Kernel1D IFF dimensions match"""
        if isinstance(other, Kernel1D):
            if self.min != other.min or self.max != other.max:
                raise IndexError("Kernel1D ranges don't match: %d..%d vs %d..%d" %
                                 (self.min, self.max, other.min, other.max))
            other = other.values
        return other

    def __eq__(self, other: "Kernel1D") -> bool:
        """Test for equality"""
        return (self.__class__ == other.__class__ and
                self.min == other.min and
                self.max == other.max and
                np.all(self.values == other.values))

    def __imul__(self, other: "Kernel1D | FloatingPoint") -> "Kernel1D":
        """Multiplication in-place

        Supports multiplication by a scalar or an array.
        """
        self.values *= self._getOtherValues(other)
        return self

    def __itruediv__(self, other: "Kernel1D | FloatingPoint") -> "Kernel1D":
        """Division in-place

        Supports division by a scalar or an array.
        """
        self.values /= self._getOtherValues(other)
        return self

    def __iadd__(self, other: "Kernel1D") -> "Kernel1D":
        """Addition in-place"""
        if self.min == other.min and self.max == other.max:
            self.values += self._getOtherValues(other)
        elif self.min <= other.min and self.max >= other.max:
            self.values[other.min - self.min:other.max - self.min + 1] += other.values
        else:
            newMin = min(self.min, other.min)
            newMax = max(self.max, other.max)
            length = newMax - newMin + 1
            center = -newMin
            indices = np.arange(length) - center
            values = np.zeros(length, dtype=self.values.dtype)

            selfSlice = slice(self.min - newMin, self.max - newMin + 1)
            assert np.all(indices[selfSlice] == self.indices)
            values[selfSlice] += self.values

            otherSlice = slice(other.min - newMin, other.max - newMin + 1)
            assert np.all(indices[otherSlice] == other.indices)
            values[otherSlice] += other.values

            self.min = newMin
            self.max = newMax
            self.center = center
            self.values = values
            self.indices = indices
            self.length = length

        return self

    def __add__(self, other: "Kernel1D") -> "Kernel1D":
        """Addition"""
        result = self.copy()
        result += other
        return result

    def __reduce__(self):
        """Pickle prescription"""
        return self.__class__, (self.values, self.center, False)

    def __repr__(self) -> str:
        """Representation"""
        return "%s(%s, %f)" % (self.__class__.__name__, self.values, self.center)

    def normalization(self) -> FloatingPoint:
        """Return the normalization"""
        return np.sum(self.values)

    def convolve(self, array: NDArray[FloatingPoint]) -> NDArray[FloatingPoint]:
        """Convolve an array by the kernel

        The convolved array has the same length as the input array in the usual
        case that the array is longer than the kernel.

        Note that the ends of the array, the kernel does not overlap completely,
        and there will be boundary effects (implementation is ``numpy.convolve``
        with ``mode="same"``).

        Parameters
        ----------
        array : `numpy.ndarray`, floating-point, shape ``(M,)``
            Array to convolve.

        Returns
        -------
        convolved : `numpy.ndarray`, floating-point, shape ``(M,)``
            Convolved array.

        Raises
        ------
        RuntimeError
            If the array is shorter than the kernel.
        """
        buffer = 0
        if len(array) < self.length:
            buffer = (self.length - len(array) + 1)//2
            array = np.pad(array, buffer, mode="constant")
        if self.center != (self.length - 1)//2:
            # Pad to put center of kernel in the middle, so we don't introduce a shift
            left = self.center
            right = self.length - self.center - 1
            middle = max(left, right)
            length = 2*middle + 1
            kernel = np.zeros(length, dtype=self.values.dtype)
            kernel[middle - left:middle + right + 1] = self.values
        else:
            kernel = self.values
        convolved = np.convolve(array, kernel, "same")
        if buffer != 0:
            convolved = convolved[buffer:-buffer]
        return convolved

    def toArray(
        self, length: int, center: float, interpolator: InterpolatorFactory = DEFAULT_INTERPOLATOR
    ) -> NDArray[FloatingPoint]:
        """Convert to an array at a nominated position

        Parameters
        ----------
        length : `int`
            Desired length of array.
        center : `float`
            Index position at which to insert the kernel.
        interpolator : callable, optional
            Interpolator factory. Calling this with arrays of ``x`` and ``y``
            should produce an interpolator that can be called with an array of
            ``x`` values at which to interpolate.

        Returns
        -------
        array : `numpy.ndarray`, floating-point, shape ``(length,)``
            Array containing inserted kernel.
        """
        return interpolator(self.indices + center, self.values)(np.arange(length))

    def computeStdev(self) -> float:
        """Compute standard deviation of kernel"""
        values = self.values.astype(np.float64)
        norm = np.sum(values)
        centroid = np.sum(self.indices*values)/norm
        return np.sqrt(np.sum((values*(self.indices - centroid)**2))/norm)

    def interpolate(self, indices: ArrayLike) -> NDArray[FloatingPoint]:
        """Interpolate the kernel at provided indices

        Parameters
        ----------
        indices : array_like, floating-point
            Indices at which to interpolate.

        Returns
        -------
        values : array_like, floating-point
            Values of kernel at the provided indices.
        """
        return np.interp(indices, self.indices, self.values, left=0.0, right=0.0)


class Lsf(ABC):
    """Abstract base class for a Line-Spread Function

    A line-spread function (LSF) is the one-dimensional version of a
    point-spread function (PSF). It models the response of the spectrograph to a
    single infinitely thin line, which may be a function of position.

    This class knows nothing about wavelengths: all positions are specified in
    pixels.

    Subclasses should implement the abstract ``computeKernel`` method; other
    methods have default implementations that rely on that.

    This class includes two main styles of methods:

    - methods taking a ``center`` position of type `float`. These methods are
      intended for interacting with a numpy array.
    - methods taking a ``point`` of type `lsst.geom.Point2D` (optionally; if not
      provided, the ``point`` defaults to the result of ``getAveragePosition``).
      These methods are  intended for emulating (duck-typing) an
      `lsst.afw.detection.Psf` for when interacting with a
      `lsst.afw.image.Image`.

    Parameters
    ----------
    length : `int`
        Array length.
    interpolator : callable, optional
        Interpolator factory for kernel. Calling this with arrays of ``x`` and
        ``y`` should produce an interpolator that can be called with an array of
        ``x`` values at which to interpolate.
    """
    def __init__(self, length: int, interpolator: InterpolatorFactory = DEFAULT_INTERPOLATOR):
        self.length = length
        self.interpolator = interpolator

    def computeImage(self, point: Point2D | None = None) -> ImageF:
        """Return an image with the LSF inserted at the nominated position

        This method does the same as the ``computeArray`` method, but packages
        the result as an ``Image``.

        Parameters
        ----------
        point : `lsst.geom.Point2D`, optional
            Position at which to evaluate the LSF.

        Returns
        -------
        image : `lsst.afw.image.Image`
            Image containing the LSF inserted.
        """
        if point is None:
            point = self.getAveragePosition()
        return ImageF(self.computeArray(point.getX())[np.newaxis, :].astype(np.float32))

    def computeArray(self, center: float) -> NDArray[FloatingPoint]:
        """Return an array with the LSF inserted at the nominated position

        Besides the difference in return types from the ``computeKernel``
        method (this returns an array, while ``computeKernel`` returns a
        `Kernel1D`), this method allows positioning of the LSF at sub-pixel
        positions. This is the method you want to use if you want a model for
        a particular line in a spectrum, e.g., to measure the flux and/or
        subtract the line.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to insert the LSF.

        Returns
        -------
        array : `numpy.ndarray`, floating-point, shape ``(self.length,)``
            Array containing the LSF inserted.
        """
        return self.computeKernel(center).toArray(self.length, center, interpolator=self.interpolator)

    def computeKernelImage(self, point: Point2D | None = None) -> ImageF:
        """Return an image of the kernel for the nominated position

        This method does the same as the ``computeKernel`` method, but packages
        the result as an ``Image``.

        Parameters
        ----------
        point : `lsst.geom.Point2D`, optional
            Position at which to evaluate the LSF.

        Returns
        -------
        kernelImage : `lsst.afw.image.Image`
            Image with a model of the LSF at the nominated position.
        """
        if point is None:
            point = self.getAveragePosition()
        kernel = self.computeKernel(point.getX())
        return ImageF(kernel.values[np.newaxis, :].astype(np.float32), True, Point2I(-kernel.center, 0))

    @abstractmethod
    def computeKernel(self, center: float) -> Kernel1D:
        """Return a kernel for the nominated position

        Besides the difference in return types from the ``computeArray``
        method (this returns a `Kernel1D`, while ``computeArray`` returns a
        an array), this method provides a model for the LSF centered at zero.
        This serves as the foundation for the other methods in the class.

        This abstract methods requires definition by subclasses.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to evaluate the LSF.

        Returns
        -------
        kernel : `Kernel1D`
            Kernel with a model of the LSF at the nominated position.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def computePeak(self, point: Point2D | None = None) -> float:
        """Return the value of the peak pixel in the LSF image

        Parameters
        ----------
        point : `lsst.geom.Point2D`, optional
            Position at which to evaluate the LSF.

        Returns
        -------
        value : `float`
            Value of the peak pixel in the LSF image.
        """
        if point is None:
            point = self.getAveragePosition()
        return self.computeKernel(point.getX())[0]

    def computeShape(self, point: Point2D | None = None) -> BaseCore:
        """Return the shape of the Lsf as if it's a ``Psf``

        This method is included for compatibility with `lsst.afw.detection.Psf`.

        Parameters
        ----------
        point : `lsst.geom.Point2D`, optional
            Position at which to evaluate the LSF.

        Returns
        -------
        shape : `lsst.afw.geom.ellipses.Quadrupole`
            The second moments of the Lsf.
        """
        if point is None:
            point = self.getAveragePosition()
        sigma = self.computeShape1D(point.getX())
        return Quadrupole(sigma**2, 0, 0)

    def computeShape1D(self, center: float) -> float:
        """Return the standard deviation at the nominated position

        The standard deviation is a measure of the width of the LSF.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to evaluate the LSF.

        Returns
        -------
        stdev : `float`
            Standard deviation of the LSF at the nominated position.
        """
        return self.computeKernel(center).computeStdev()

    def computeApertureFlux(self, radius: float, point=None) -> float:
        """Return the flux within the nominated radius

        Parameters
        ----------
        radius : `float`
            (1 dimensional) radius within which to measure the flux.
        point : `lsst.geom.Point2D`, optional
            Position at which to evaluate the LSF.

        Returns
        -------
        flux : `float`
            Flux within the nominated radius.
        """
        if point is None:
            point = self.getAveragePosition()
        kernel = self.computeKernel(point.getX())
        intRadius = int(radius)

        flux = trapezoid([kernel[ii] for ii in range(-intRadius, intRadius + 1)])
        if radius != intRadius:
            residual = radius - intRadius
            left, right = kernel.interpolate([-radius, radius])
            flux += 0.5*residual*(kernel[-intRadius - 1] + kernel[-intRadius])
            flux += 0.5*residual*(kernel[intRadius + 1] + kernel[intRadius])
        return flux

    def getLocalKernel(self, point: Point2D | None = None) -> FixedKernel:
        """Return a 2D ``FixedKernel`` representing the Lsf at a point

        Parameters
        ----------
        point : `lsst.geom.Point2D`, optional
            Position at which to evaluate the LSF.

        Returns
        -------
        kernel : `lsst.afw.math.FixedKernel`
            Kernel representing the Lsf at the nominated point.
        """
        if point is None:
            point = self.getAveragePosition()
        kernel = self.computeKernel(point.getX())
        image = ImageD(kernel.values[np.newaxis, :], True, Point2I(-kernel.center, 0))
        return FixedKernel(image)

    def getAveragePosition(self) -> Point2D:
        """Return the average position for which the Lsf is defined

        Returns
        -------
        position : `lsst.geom.Point2D`, optional
            Average position for which the Lsf is defined.
        """
        return Point2D(0.5*self.length, 0)

    def computeBBox(self, point: Point2D | None = None) -> Box2I:
        """Return the bounding box of the kernel image

        Parameters
        ----------
        point : `lsst.geom.Point2D`, optional
            Position at which to evaluate the LSF.

        Returns
        -------
        bbox : `lsst.afw.geom.Box2I`
            Bounding box for the kernel image.
        """
        if point is None:
            point = self.getAveragePosition()
        return self.computeKernelImage(point.getX()).getBBox()

    @classmethod
    def readFits(cls, filename: str) -> "Lsf":
        """Read from file

        We don't yet know exactly what datamodel we want to use for sharing
        these with users, so this is a placeholder implementation. For now,
        though the method name says "FITS", we read a pickle file, because
        pickling takes care of subclasses for us.

        Parameters
        ----------
        filename : `str`
            Filename from which to read.

        Returns
        -------
        self : ``cls``
            `Lsf` object read from file.
        """
        with open(filename, "rb") as fd:
            return pickle.load(fd)

    def writeFits(self, filename: str) -> None:
        """Write to file

        We don't yet know exactly what datamodel we want to use for sharing
        these with users, so this is a placeholder implementation. For now,
        though the method name says "FITS", we write a pickle file, because
        pickling takes care of subclasses for us.

        Parameters
        ----------
        filename : `str`
            Filename to which to write.
        """
        with open(filename, "wb") as fd:
            pickle.dump(self, fd)

    def convolve(self, array: NDArray[FloatingPoint]) -> NDArray[FloatingPoint]:
        """Convolve an array by the LSF

        Parameters
        ----------
        array : `numpy.ndarray`, floating-point, shape ``(M,)``
            Array to convolve.

        Returns
        -------
        convolved : `numpy.ndarray`, floating-point, shape ``(M,)``
            Convolved array.
        """
        return self.computeKernel(0.5*self.length).convolve(array)

    def warp(self, inWavelength: NDArray[FloatingPoint], outWavelength: NDArray[FloatingPoint]) -> "Lsf":
        """Warp the LSF to a new wavelength frame

        Parameters
        ----------
        inWavelength : `numpy.ndarray` of `float`
            Wavelength array in the same frame as for ``lsf``.
        outWavelength : `numpy.ndarray` of `float`
            Wavelength array in the target frame.

        Returns
        -------
        warpedLsf : `pfs.drp.stella.Lsf`
            LSF in the warped frame.
        """
        raise NotImplementedError("Subclasses must implement this method")


def gaussian(indices: ArrayLike, width: float) -> ArrayLike:
    """Evaluate an un-normalized Gaussian

    Parameters
    ----------
    indices : array-like
        Positions at which to evaluate.
    width : `float`
        Gaussian RMS width.

    Returns
    -------
    values : array-like
        Gaussian evaluations.
    """
    return np.exp(-0.5*(indices/width)**2)


class GaussianKernel1D(Kernel1D):
    """A Gaussian kernel

    Parameters
    ----------
    width : `float`
        Gaussian RMS width.
    nWidth : `float`, optional
        Multiple of ``width`` for the width of the kernel.
    """
    def __init__(self, width: float, nWidth: float = 4.0):
        halfSize = int(width*nWidth + 0.5)
        size = 2*halfSize + 1
        indices = np.arange(size)
        super().__init__(gaussian(indices - halfSize, width), halfSize)
        self.width = width
        self.nWidth = nWidth

    def copy(self) -> "GaussianKernel1D":
        """Return a copy of the kernel"""
        return self.__class__(self.width, self.nWidth)

    def toKernel1D(self) -> Kernel1D:
        """Convert to a generic Kernel1D (without the Gaussian-specific
        attributes)
        """
        return Kernel1D(self.values.copy(), self.center, normalize=False)

    def computeShape1D(self) -> float:
        """Compute standard deviation of kernel

        We know it, so we don't have to calculate it.
        """
        return self.width

    def __setitem__(self, index: int, value: FloatingPoint):
        """Prevent setting values"""
        raise NotImplementedError("If you modify the values directly, this won't be a Gaussian any more!")

    def __imul__(self, other: "Kernel1D | FloatingPoint") -> "GaussianKernel1D":
        """Multiplication in-place

        This changes the type, which is a bit unexpected for an in-place
        operation, but we can't compute the product otherwise.
        """
        new = self.toKernel1D()
        new *= other
        return new

    def __itruediv__(self, other: "Kernel1D | FloatingPoint") -> "GaussianKernel1D":
        """Division in-place

        This changes the type, which is a bit unexpected for an in-place
        operation, but we can't compute the quotient otherwise.
        """
        new = self.toKernel1D()
        new /= other
        return new

    def __iadd__(self, other: "Kernel1D") -> "GaussianKernel1D":
        """Addition in-place

        This changes the type, which is a bit unexpected for an in-place
        operation, but we can't compute the sum otherwise.
        """
        new = self.toKernel1D()
        new += other
        return new

    def __add__(self, other: "Kernel1D") -> "Kernel1D":
        """Addition"""
        return self.toKernel1D() + other

    def __reduce__(self):
        """Pickle prescription"""
        return self.__class__, (self.width, self.nWidth)


class GaussianLsf(Lsf):
    """A Gaussian LSF of fixed width

    Parameters
    ----------
    length : `int`
        Array length.
    width : `float`
        Gaussian RMS width.
    """
    def __init__(self, length: int, width: float):
        super().__init__(length)
        self.width = width
        self.kernel = GaussianKernel1D(width)
        self.norm = 1.0/(width*np.sqrt(2*np.pi))

    def computeArray(self, center: float) -> NDArray[FloatingPoint]:
        """Return an array with the LSF inserted at the nominated position

        Besides the difference in return types from the ``computeKernel``
        method (this returns an array, while ``computeKernel`` returns a
        `Kernel1D`), this method allows positioning of the LSF at sub-pixel
        positions. This is the method you want to use if you want a model for
        a particular line in a spectrum, e.g., to measure the flux and/or
        subtract the line.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to insert the LSF.

        Returns
        -------
        array : `numpy.ndarray`, floating-point, shape ``(self.length,)``
            Array containing the LSF inserted.
        """
        xx = np.arange(self.length)
        return self.norm*gaussian(xx - center, self.width)

    def computeKernel(self, center: float) -> Kernel1D:
        """Return a kernel for the nominated position

        Besides the difference in return types from the ``computeArray``
        method (this returns a `Kernel1D`, while ``computeArray`` returns a
        an array), this method provides a model for the LSF centered at zero.
        This serves as the foundation for the other methods in the class.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to evaluate the LSF. Not that it matters, as it's
            the same everywhere.

        Returns
        -------
        kernel : `Kernel1D`
            Kernel with a model of the LSF at the nominated position.
        """
        return self.kernel

    def computeApertureFlux(self, radius: float, point: Point2D | None = None) -> float:
        """Return the flux within the nominated radius

        Parameters
        ----------
        point : `lsst.geom.Point2D`, optional
            Position at which to evaluate the LSF.

        Returns
        -------
        flux : `float`
            Flux within the nominated radius.
        """
        return scipy.special.erf(np.sqrt(0.5)*radius/self.width)

    def computeShape1D(self, center: float) -> float:
        """Return the standard deviation at the nominated position

        The standard deviation is a measure of the width of the LSF.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to evaluate the LSF. Not that it matters, as it's
            the same everywhere.

        Returns
        -------
        stdev : `float`
            Standard deviation of the LSF at the nominated position.
        """
        return self.width

    def warp(
        self, inWavelength: NDArray[FloatingPoint], outWavelength: NDArray[FloatingPoint]
    ) -> "GaussianLsf":
        """Warp the LSF to a new wavelength frame

        Parameters
        ----------
        inWavelength : `numpy.ndarray` of `float`
            Wavelength array in the same frame as for ``lsf``.
        outWavelength : `numpy.ndarray` of `float`
            Wavelength array in the target frame.

        Returns
        -------
        warpedLsf : `pfs.drp.stella.GaussianLsf`
            LSF in the warped frame.
        """
        inLength = len(inWavelength)
        if self.length != inLength:
            raise RuntimeError(f"Length mismatch between LSF ({self.length}) and wavelength ({inLength})")
        outLength = len(outWavelength)
        inPixels = np.arange(inLength, dtype=inWavelength.dtype)
        outPixels = np.arange(outLength, dtype=outWavelength.dtype)

        inToWavelength = interp1d(inPixels, inWavelength, kind="linear", assume_sorted=True)
        wavelengthToOut = interp1d(outWavelength, outPixels, kind="linear", assume_sorted=True)

        def transform(inRow: float) -> float:
            """Transform input row value to output row value

            Parameters
            ----------
            inRow : `float`
                Input row value.

            Returns
            -------
            outRow : `float`
                Output row value.
            """
            return wavelengthToOut(inToWavelength(inRow))

        inMiddle = 0.5*inLength
        inWidth = self.computeShape1D(inMiddle)
        outWidth = abs(transform(inMiddle + 0.5*inWidth) - transform(inMiddle - 0.5*inWidth))
        return GaussianLsf(outLength, outWidth)

    def __reduce__(self):
        """Pickling"""
        return self.__class__, (self.length, self.width)


class FixedEmpiricalLsf(Lsf):
    """A constant empirical LSF

    Parameters
    ----------
    kernel : `Kernel1D`
        The kernel to use.
    length : `int`
        Array length.
    interpolator : callable, optional
        Interpolator factory for kernel. Calling this with arrays of ``x`` and
        ``y`` should produce an interpolator that can be called with an array of
        ``x`` values at which to interpolate.
    """
    def __init__(
        self, kernel: Kernel1D, length: int, interpolator: InterpolatorFactory = DEFAULT_INTERPOLATOR
    ):
        super().__init__(length, interpolator=interpolator)
        self.kernel = kernel

    def computeKernel(self, center: float) -> Kernel1D:
        """Return a kernel for the nominated position

        Besides the difference in return types from the ``computeArray``
        method (this returns a `Kernel1D`, while ``computeArray`` returns a
        an array), this method provides a model for the LSF centered at zero.
        This serves as the foundation for the other methods in the class.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to evaluate the LSF. Not that it matters, as it's
            the same everywhere.

        Returns
        -------
        kernel : `Kernel1D`
            Kernel with a model of the LSF at the nominated position.
        """
        return self.kernel

    def __reduce__(self):
        """Pickling"""
        return self.__class__, (self.kernel, self.length, self.interpolator)


class ExtractionLsf(Lsf):
    """LSF from extracting a PSF

    Parameters
    ----------
    psf : `pfs.drp.stella.SpectralPsf`
        Point-spread function for spectral data.
    fiberTrace : `pfs.drp.stella.FiberTrace`
        Fiber profile.
    length : `int`
        Array length.
    """
    def __init__(self, psf: "SpectralPsf", fiberTrace: "FiberTrace", length: int):
        self.psf = psf
        self.fiberTrace = fiberTrace
        self.fiberId = fiberTrace.fiberId
        super().__init__(length)

    def computeKernel(self, center: float) -> Kernel1D:
        """Return a kernel for the nominated position

        Besides the difference in return types from the ``computeArray``
        method (this returns a `Kernel1D`, while ``computeArray`` returns a
        an array), this method provides a model for the LSF centered at zero.

        This implementation realises the PSF and extracts it with the
        fiberTrace. The difference from ``computeArray` is that this method
        puts the center on the middle of a pixel.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to evaluate the LSF. Not that it matters, as it's
            the same everywhere.

        Returns
        -------
        kernel : `Kernel1D`
            Kernel with a model of the LSF at the nominated position.
        """
        detMap = self.psf.getDetectorMap()
        xx = detMap.getXCenter(self.fiberId, center)
        psfImage = MaskedImageF(self.psf.computeKernelImage(Point2D(xx, center)).convertF())
        psfImage.variance.set(1.0)
        xy0 = psfImage.getXY0()
        psfImage.setXY0(xy0 + Extent2I(int(xx + 0.5), int(center + 0.5)))  # So extraction works properly
        traces = FiberTraceSet(1)
        traces.add(self.fiberTrace)
        spectra = traces.extractSpectra(psfImage)
        assert len(spectra) == 1
        return Kernel1D(spectra[0].flux, -xy0.getY())

    def computeArray(self, center: float) -> NDArray[FloatingPoint]:
        """Return an array with the LSF inserted at the nominated position

        Besides the difference in return types from the ``computeKernel``
        method (this returns an array, while ``computeKernel`` returns a
        `Kernel1D`), this method allows positioning of the LSF at sub-pixel
        positions. This is the method you want to use if you want a model for
        a particular line in a spectrum, e.g., to measure the flux and/or
        subtract the line.

        This implementation realises the PSF and extracts it with the
        fiberTrace. The difference from ``computeKernel` is that this method
        puts the center at the nominated position instead on the center of a
        pixel.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to insert the LSF.

        Returns
        -------
        array : `numpy.ndarray`, floating-point, shape ``(self.length,)``
            Array containing the LSF inserted.
        """
        detMap = self.psf.getDetectorMap()
        xx = detMap.getXCenter(self.fiberId, center)
        psfImage = MaskedImageF(self.psf.computeImage(Point2D(xx, center)).convertF())
        psfImage.variance.set(1.0)
        psfBox = psfImage.getBBox()
        traces = FiberTraceSet(1)
        traces.add(self.fiberTrace)
        spectra = traces.extractSpectra(psfImage)
        assert len(spectra) == 1
        array = np.zeros(self.length)
        yMin = max(psfBox.getMinY(), 0)
        yMax = min(psfBox.getMaxY() + 1, self.length)
        spectrum = spectra[0].flux[yMin - psfBox.getMinY():yMax - psfBox.getMinY()]
        spectrum /= spectrum.sum()
        array[yMin:yMax] = spectrum
        return array

    def __reduce__(self):
        """Pickling"""
        return self.__class__, (self.psf, self.fiberTrace, self.length)


class NoLsfsError(RuntimeError):
    """Exception for no LSFs at a position"""
    pass


class CoaddLsf(Lsf):
    """Coadded line-spread functions

    Parameters
    ----------
    lsfList : iterable of `pfs.drp.stella.Lsf`
        Line-spread functions to coadd.
    minIndex, maxIndex : array-like of `int`
        Minimum (inclusive) and maximum (inclusive) indices for each LSF.
    weights : array-like of `float`
        Weights for each LSF.
    """

    def __init__(
        self,
        lsfList: Iterable[Lsf],
        minIndex: ArrayLike,
        maxIndex: ArrayLike,
        weights: ArrayLike | None = None,
    ):
        self.lsfList = [lsf for lsf in lsfList]
        self.minIndex = np.asarray(minIndex, dtype=int)
        self.maxIndex = np.asarray(maxIndex, dtype=int)
        self.weights = np.asarray(weights, dtype=float) if weights is not None else np.ones(len(self.lsfList))
        length = set(lsf.length for lsf in self.lsfList)
        if len(length) != 1:
            raise ValueError("LSFs have different lengths")
        super().__init__(length.pop())

    def _iterateLsfs(self, center: float) -> Iterator[tuple[Lsf, float]]:
        """Iterate over LSFs and weights"""
        for ii in range(len(self.lsfList)):
            lsf = self.lsfList[ii]
            if lsf is not None and center >= self.minIndex[ii] and center <= self.maxIndex[ii]:
                yield lsf, self.weights[ii]

    def computeArray(self, center: float) -> NDArray[FloatingPoint]:
        """Return an array with the LSF inserted at the nominated position

        Besides the difference in return types from the ``computeKernel``
        method (this returns an array, while ``computeKernel`` returns a
        `Kernel1D`), this method allows positioning of the LSF at sub-pixel
        positions. This is the method you want to use if you want a model for
        a particular line in a spectrum, e.g., to measure the flux and/or
        subtract the line.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to insert the LSF.

        Returns
        -------
        array : `numpy.ndarray`, floating-point, shape ``(length,)``
            Array containing the LSF inserted.
        """
        array = np.zeros(self.length)
        for lsf, weight in self._iterateLsfs(center):
            array += weight*lsf.computeArray(center)
        return array

    def computeKernel(self, center: float) -> Kernel1D:
        """Return a kernel for the nominated position

        Besides the difference in return types from the ``computeArray``
        method (this returns a `Kernel1D`, while ``computeArray`` returns a
        an array), this method provides a model for the LSF centered at zero.
        This serves as the foundation for the other methods in the class.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to evaluate the LSF.

        Returns
        -------
        kernel : `Kernel1D`
            Kernel with a model of the LSF at the nominated position.

        Raises
        ------
        NoLsfsError
            If there are no LSFs overlapping the ``center``.
        """
        lsfList = [(lsf, weight) for lsf, weight in self._iterateLsfs(center)]
        if not lsfList:
            raise NoLsfsError("No LSFs at this position")
        sumWeight = 0.0
        lsf, weight = lsfList.pop(0)
        kernel = lsf.computeKernel(center).copy()
        kernel *= weight
        sumWeight += weight
        for lsf, weight in lsfList:
            kk = lsf.computeKernel(center)
            kk *= weight
            kernel += kk
            sumWeight += weight
        kernel /= sumWeight
        return kernel

    def convolve(self, array: NDArray[FloatingPoint], sampling: int = 100) -> NDArray[FloatingPoint]:
        """Convolve an array by the LSF

        Parameters
        ----------
        array : `numpy.ndarray`, floating-point, shape ``(M,)``
            Array to convolve.
        sampling : `int`
            Sampling length for convolution, pixels.

        Returns
        -------
        convolved : `numpy.ndarray`, floating-point, shape ``(M,)``
            Convolved array.
        """
        length = len(array)
        convolved = np.zeros_like(array)
        for start in range(0, length, sampling):
            stop = min(start + sampling, length)  # exclusive
            center = 0.5*(start + stop)
            try:
                kernel = self.computeKernel(center)
            except NoLsfsError:
                convolved[start:stop] = 0.0
                continue

            # We're going to convolve a section of the array from start to stop,
            # with a buffer on either side to allow for boundary effects.

            # Range of pixels on the full array that we will convolve
            beginFull = max(0, start + kernel.min)
            endFull = min(length, stop + kernel.max + 1)

            # Range of pixels on the convolved subarray
            beginSub = start - beginFull
            endSub = beginSub + stop - start

            conv = kernel.convolve(array[beginFull:endFull])
            convolved[start:stop] = conv[beginSub:endSub]

        return convolved

    def warp(
        self, inWavelength: NDArray[FloatingPoint], outWavelength: NDArray[FloatingPoint]
    ) -> "CoaddLsf":
        """Warp the LSF to a new wavelength frame

        Parameters
        ----------
        inWavelength : `numpy.ndarray` of `float`
            Wavelength array in the same frame as for ``lsf``.
        outWavelength : `numpy.ndarray` of `float`
            Wavelength array in the target frame.

        Returns
        -------
        warpedLsf : `pfs.drp.stella.CoaddLsf`
            LSF in the warped frame.
        """
        lsfList = [lsf.warp(inWavelength, outWavelength) for lsf in self.lsfList]
        minIndex = [np.searchsorted(outWavelength, inWavelength[minIndex]) for minIndex in self.minIndex]
        maxIndex = [np.searchsorted(outWavelength, inWavelength[maxIndex]) for maxIndex in self.maxIndex]
        return self.__class__(lsfList, minIndex, maxIndex, self.weights)

    def __reduce__(self):
        """Pickling"""
        return self.__class__, (self.lsfList, self.minIndex, self.maxIndex, self.weights)


class LsfDict(dict):
    """A mapping from some index to Lsf

    Depending upon the dataset type, the index might be fiberId or spectral
    target.

    This exists just so that we can have a specific python class that will hold
    the appropriate data.
    """
    pass
