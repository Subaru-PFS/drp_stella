from abc import ABC, abstractmethod
from functools import partial
import pickle
import numpy as np
from scipy.interpolate import interp1d

__all__ = ["Kernel", "Lsf", "GaussianKernel", "GaussianLsf", "FixedEmpiricalLsf"]

# Default interpolator factory
DEFAULT_INTERPOLATOR = partial(interp1d, kind="linear", bounds_error=False, fill_value=0, copy=True,
                               assume_sorted=True)


class Kernel:
    """A one-dimensional kernel

    A Kernel is a one-dimensional version of an `lsst.afw.image.Image` that
    has an ``xy0`` set so that ``0,0`` is at the center of the image.
    Main operations are ``convolve`` to convolve an array by a kernel, and
    ``toArray`` which inserts the kernel at any point in an array.

    Parameters
    ----------
    values : `numpy.ndarray`, floating-point
        Kernel values.
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
        """Create an empty ``Kernel``

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

    def _indexOutOfRange(self, index):
        """Is the index out of range?"""
        return index < self.min or index > self.max

    def _getitem(self, index):
        """Get a single value from the kernel

        Values off the end are zero.
        """
        return self.values[index + self.center] if not self._indexOutOfRange(index) else 0.0

    def _setitem(self, index, value):
        """Set a single value in the kernel

        Values off the end raise an `IndexError`.
        """
        if self._indexOutOfRange(index):
            raise IndexError("Index out of range: %d vs %d..%d" % (index, self.min, self.max))
        self.values[index + self.center] = value

    def __getitem__(self, index):
        """Get value(s) from the kernel

        Supports integer or array indexing.

        Values off the end are zero.
        """
        if isinstance(index, int):
            return self._getitem(index)
        return np.vectorize(self._getitem)(index)

    def __setitem__(self, index, value):
        """Set value(s) in the kernel.

        Supports integer or array indexing.

        Values off the end raise an `IndexError`.
        """
        return np.vectorize(self._setitem)(index, value)

    def __iter__(self):
        """Iterator

        Produces ``index,value`` pairs.
        """
        return zip(iter(self.indices), iter(self.values))

    def _getOtherValues(self, other):
        """Get values from another Kernel IFF dimensions match"""
        if isinstance(other, Kernel):
            if self.min != other.min or self.max != other.max:
                raise IndexError("Kernel ranges don't match: %d..%d vs %d..%d" %
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

        Note that the ends of the array will not be fully convolved.

        Parameters
        ----------
        array : `numpy.ndarray`, floating-point
            Array to convolve.

        Raises
        ------
        RuntimeError
            If the array is shorter than the kernel.
        """
        if len(array) < self.length:
            raise RuntimeError("Array to short for convolution by this kernel")
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
        array : `numpy.ndarray`, floating-point
            Array containing inserted kernel.
        """
        return interpolator(self.indices + center, self.values)(np.arange(length))

    def computeStdev(self):
        """Compute standard deviation of kernel"""
        centroid = np.sum((self.indices*self.values).astype(np.float64))
        return np.sqrt(np.sum((self.values*(self.indices - centroid)**2).astype(np.float64)))


class Lsf(ABC):
    """Abstract base class for a Line-Spread Function

    A "line-spread function (LSF) is the one-dimensional version of a
    point-spread function (PSF). It models the response of the spectrograph to a
    single infinitely thin line, which may be a function of position.

    This class knows nothing about wavelengths: all positions are specified in
    pixels.

    Subclasses should implement the abstract ``computeKernel`` method; other
    methods have default implementations that rely on that.

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

    def computeArray(self, center):
        """Return an array with the LSF inserted at the nominated position

        Besides the difference in return types from the ``computeKernel``
        method (this returns an array, while ``computeKernel`` returns a
        `Kernel`), this method allows positioning of the LSF at sub-pixel
        positions. This is the method you want to use if you want a model for
        a particular line in a spectrum, e.g., to measure the flux and/or
        subtract the line.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to insert the LSF.

        Returns
        -------
        array : `numpy.ndarray`, floating-point
            Array containing the LSF inserted.
        """
        return self.computeKernel(center).toArray(self.length, center, interpolator=self.interpolator)

    @abstractmethod
    def computeKernel(self, center):
        """Return a kernel for the nominated position

        Besides the difference in return types from the ``computeArray``
        method (this returns a `Kernel`, while ``computeArray`` returns a
        an array), this method provides a model for the LSF centered at zero.
        This serves as the foundation for the other methods in the class.

        This abstract methods requires definition by subclasses.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to evaluate the LSF.

        Returns
        -------
        kernel : `Kernel`
            Kernel with a model of the LSF at the nominated position.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def computeStdev(self, center):
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

    @classmethod
    def readFits(cls, filename):
        """Read from file

        Though the method name says "FITS", we read a pickle file, because
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

        Though the method name says "FITS", we read a pickle file, because
        pickling takes care of subclasses for us.

        Parameters
        ----------
        filename : `str`
            Filename to which to write.
        """
        with open(filename, "wb") as fd:
            pickle.dump(self, fd)


def gaussian(indices, sigma):
    """Evaluate an un-normalized Gaussian

    Parameters
    ----------
    indices : array-like
        Positions at which to evaluate.
    sigma : `float`
        Gaussian sigma.

    Returns
    -------
    values : array-like
        Gaussian evaluations.
    """
    return np.exp(-0.5*(indices/sigma)**2)


class GaussianKernel(Kernel):
    """A Gaussian kernel

    Parameters
    ----------
    sigma : `float`
        Gaussian sigma.
    nSigma : `float`, optional
        Multiple of sigma for the width of the kernel.
    """
    def __init__(self, sigma, nSigma=4.0):
        halfSize = int(sigma*nSigma + 0.5)
        size = 2*halfSize + 1
        indices = np.arange(size)
        super().__init__(gaussian(indices - halfSize, sigma), halfSize)
        self.sigma = sigma

    def computeStdev(self):
        """Compute standard deviation of kernel

        We know it, so we don't have to calculate it.
        """
        return self.sigma

    def __setitem__(self, index, value):
        """Prevent setting values"""
        raise NotImplementedError("If you modify the values directly, this won't be a Gaussian any more!")


class GaussianLsf(Lsf):
    """A Gaussian LSF of fixed width

    Parameters
    ----------
    length : `int`
        Array length.
    sigma : `float`
        Gaussian sigma.
    """
    def __init__(self, length, sigma):
        super().__init__(length)
        self.sigma = sigma
        self.kernel = GaussianKernel(sigma)
        self.norm = 1.0/(sigma*np.sqrt(2*np.pi))

    def computeArray(self, center):
        """Return an array with the LSF inserted at the nominated position

        Besides the difference in return types from the ``computeKernel``
        method (this returns an array, while ``computeKernel`` returns a
        `Kernel`), this method allows positioning of the LSF at sub-pixel
        positions. This is the method you want to use if you want a model for
        a particular line in a spectrum, e.g., to measure the flux and/or
        subtract the line.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to insert the LSF.

        Returns
        -------
        array : `numpy.ndarray`, floating-point
            Array containing the LSF inserted.
        """
        xx = np.arange(self.length)
        return self.norm*gaussian(xx - center, self.sigma)

    def computeKernel(self, center):
        """Return a kernel for the nominated position

        Besides the difference in return types from the ``computeArray``
        method (this returns a `Kernel`, while ``computeArray`` returns a
        an array), this method provides a model for the LSF centered at zero.
        This serves as the foundation for the other methods in the class.

        This abstract methods requires definition by subclasses.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to evaluate the LSF. Not that it matters, as it's
            the same everywhere.

        Returns
        -------
        kernel : `Kernel`
            Kernel with a model of the LSF at the nominated position.
        """
        return self.kernel

    def computeShape(self, center):
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
        return self.sigma

    def __reduce__(self):
        """Pickling"""
        return self.__class__, (self.length, self.sigma)


class FixedEmpiricalLsf(Lsf):
    """A constant empirical LSF

    Parameters
    ----------
    kernel : `Kernel`
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
        method (this returns a `Kernel`, while ``computeArray`` returns a
        an array), this method provides a model for the LSF centered at zero.
        This serves as the foundation for the other methods in the class.

        This abstract methods requires definition by subclasses.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to evaluate the LSF. Not that it matters, as it's
            the same everywhere.

        Returns
        -------
        kernel : `Kernel`
            Kernel with a model of the LSF at the nominated position.
        """
        return self.kernel

    def __reduce__(self):
        """Pickling"""
        return self.__class__, (self.kernel, self.length, self.interpolator)
