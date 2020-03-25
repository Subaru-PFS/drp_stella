from abc import ABC, abstractmethod
from functools import partial
import pickle
import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.integrate import trapz

from lsst.geom import Point2D, Point2I
from lsst.afw.image import ImageF, ImageD
from lsst.afw.math import FixedKernel
from lsst.afw.geom.ellipses import Quadrupole

__all__ = ("Kernel1D", "Lsf", "GaussianKernel1D", "GaussianLsf", "FixedEmpiricalLsf")

# Default interpolator factory
DEFAULT_INTERPOLATOR = partial(interp1d, kind="linear", bounds_error=False, fill_value=0, copy=True,
                               assume_sorted=True)


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
    def __init__(self, values, center, normalize=True):
        self.length = len(values)
        self.center = center
        self.values = values
        self.indices = np.arange(self.length) - center
        self.min = self.indices[0]
        self.max = self.indices[-1]
        if normalize:
            self.values /= self.normalization()

    @classmethod
    def makeEmpty(cls, halfSize):
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

    def __len__(self):
        """Number of elements in kernel array"""
        return self.length

    def __getitem__(self, index):
        """Get value(s) from the kernel

        Supports integer or array indexing.

        Values off the end are zero.
        """
        if isinstance(index, int):
            return self.values[index + self.center]
        result = np.empty_like(index)
        rr, ii, vv = np.broadcast_arrays(result, index, self.values)
        rr[:] = vv[ii + self.center]
        return result

    def __setitem__(self, index, value):
        """Set value(s) in the kernel.

        Supports integer or array indexing.

        Values off the end raise an `IndexError`.
        """
        ii, vv = np.broadcast_arrays(index, value)
        self.values[ii + self.center] = vv

    def __iter__(self):
        """Iterator

        Produces ``index,value`` pairs.
        """
        return zip(iter(self.indices), iter(self.values))

    def _getOtherValues(self, other):
        """Get values from another Kernel1D IFF dimensions match"""
        if isinstance(other, Kernel1D):
            if self.min != other.min or self.max != other.max:
                raise IndexError("Kernel1D ranges don't match: %d..%d vs %d..%d" %
                                 (self.min, self.max, other.min, other.max))
            other = other.values
        return other

    def __eq__(self, other):
        """Test for equality"""
        return (self.__class__ == other.__class__ and
                self.min == other.min and
                self.max == other.max and
                np.all(self.values == other.values))

    def __imul__(self, other):
        """Multiplication in-place

        Supports multiplication by a scalar or an array.
        """
        self.values *= self._getOtherValues(other)
        return self

    def __itruediv__(self, other):
        """Division in-place

        Supports division by a scalar or an array.
        """
        self.values /= self._getOtherValues(other)
        return self

    def __reduce__(self):
        """Pickle prescription"""
        return self.__class__, (self.values, self.center, False)

    def __repr__(self):
        """Representation"""
        return "%s(%s, %f)" % (self.__class__.__name__, self.values, self.center)

    def normalization(self):
        """Return the normalization"""
        return np.sum(self.values)

    def convolve(self, array):
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

        Raises
        ------
        RuntimeError
            If the array is shorter than the kernel.
        """
        if len(array) < self.length:
            raise RuntimeError("Array too short for convolution by this kernel")
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
        return np.convolve(array, kernel, "same")

    def toArray(self, length, center, interpolator=DEFAULT_INTERPOLATOR):
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

    def computeStdev(self):
        """Compute standard deviation of kernel"""
        centroid = np.sum((self.indices*self.values).astype(np.float64))
        return np.sqrt(np.sum((self.values*(self.indices - centroid)**2).astype(np.float64)))

    def interpolate(self, indices):
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
    def __init__(self, length, interpolator=DEFAULT_INTERPOLATOR):
        self.length = length
        self.interpolator = interpolator

    def computeImage(self, point=None):
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

    def computeArray(self, center):
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

    def computeKernelImage(self, point=None):
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
    def computeKernel(self, center):
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

    def computePeak(self, point=None):
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

    def computeShape(self, point=None):
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

    def computeShape1D(self, center):
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

    def computeApertureFlux(self, radius, point=None):
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
        if point is None:
            point = self.getAveragePosition()
        kernel = self.computeKernel(point.getX())
        intRadius = int(radius)

        flux = trapz([kernel[ii] for ii in range(-intRadius, intRadius + 1)])
        if radius != intRadius:
            residual = radius - intRadius
            left, right = kernel.interpolate([-radius, radius])
            flux += 0.5*residual*(kernel[-intRadius - 1] + kernel[-intRadius])
            flux += 0.5*residual*(kernel[intRadius + 1] + kernel[intRadius])
        return flux

    def getLocalKernel(self, point=None):
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

    def getAveragePosition(self):
        """Return the average position for which the Lsf is defined

        Returns
        -------
        position : `lsst.geom.Point2D`, optional
            Average position for which the Lsf is defined.
        """
        return Point2D(0.5*self.length, 0)

    def computeBBox(self, point=None):
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
    def readFits(cls, filename):
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

    def writeFits(self, filename):
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


def gaussian(indices, width):
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
    def __init__(self, width, nWidth=4.0):
        halfSize = int(width*nWidth + 0.5)
        size = 2*halfSize + 1
        indices = np.arange(size)
        super().__init__(gaussian(indices - halfSize, width), halfSize)
        self.width = width

    def computeShape1D(self):
        """Compute standard deviation of kernel

        We know it, so we don't have to calculate it.
        """
        return self.width

    def __setitem__(self, index, value):
        """Prevent setting values"""
        raise NotImplementedError("If you modify the values directly, this won't be a Gaussian any more!")


class GaussianLsf(Lsf):
    """A Gaussian LSF of fixed width

    Parameters
    ----------
    length : `int`
        Array length.
    width : `float`
        Gaussian RMS width.
    """
    def __init__(self, length, width):
        super().__init__(length)
        self.width = width
        self.kernel = GaussianKernel1D(width)
        self.norm = 1.0/(width*np.sqrt(2*np.pi))

    def computeArray(self, center):
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

    def computeKernel(self, center):
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

    def computeApertureFlux(self, radius, point=None):
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

    def computeShape1D(self, center):
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
    def __init__(self, kernel, length, interpolator=DEFAULT_INTERPOLATOR):
        super().__init__(length, interpolator=interpolator)
        self.kernel = kernel

    def computeKernel(self, center):
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
