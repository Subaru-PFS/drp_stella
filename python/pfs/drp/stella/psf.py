from abc import ABC, abstractmethod

from lsst.afw.image import positionToIndex
from lsst.afw.math import offsetImage
from lsst.afw.fits import MemFileManager
from lsst.afw.detection import Psf

__all__ = ["SpectralPsf", "ImagePsf"]


class SpectralPsf(ABC):
    """Base class for PSF on a spectral image

    This is based on the lsst::afw::detection::Psf class; we could inherit from
    that class, but reimplementation in python is easy, and it allows us to
    side-step all the hassles of C++. Also, the main calculation methods
    (``computeImage``, ``computeKernelImage`` and ``computeShape``) take
    ``fiberId,wavelength`` arguments; there are corresponding methods
    (``computeImageXY``, ``computeKernelImageXY`` and ``computeShapeXY``) that
    work in terms of position on the image.

    Parameters
    ----------
    detectorMap : `pfs.drp.stella.DetectorMap`
        Mapping from fiberId,wavelength <--> x,y.
    warpAlgorithm : `str`
        Warp kernel to use in ``computeImageXY``.
    warpBuffer : `int`
        Buffer around image to use ``computeImageXY``.
    """
    def __init__(self, detectorMap, warpAlgorithm="lanczos5", warpBuffer=5):
        self.detectorMap = detectorMap
        self.warpAlgorithm = warpAlgorithm
        self.warpBuffer = warpBuffer

    def computeImage(self, fiberId, wavelength):
        """Return an image in the same coordinate system as the pixelated image

        This is appropriate for fitting or subtracting a PSF.

        Parameters
        ----------
        fiberId : `int`
            Fiber identifier.
        wavelength : `float`
            Wavelength.

        Returns
        -------
        image : `lsst.afw.image.Image`
            Image of the PSF.
        """
        return self.computeImageXY(self.detectorMap.findPoint(fiberId, wavelength))

    def computeImageXY(self, point):
        """Return an image in the same coordinate system as the pixelated image

        This is appropriate for fitting or subtracting a PSF.

        Parameters
        ----------
        point : `lsst.geom.Point2D`
            Image position at which to calculate.

        Returns
        -------
        image : `lsst.afw.image.Image`
            Image of the PSF.
        """

        kernel = self.computeKernelImageXY(point)
        xWhole, xFrac = positionToIndex(point.getX(), True)
        yWhole, yFrac = positionToIndex(point.getY(), True)
        if xFrac != 0.0 or yFrac != 0.0:
            image = offsetImage(kernel, xFrac, yFrac, self.warpAlgorithm, self.warpBuffer)
        image.setXY0(xWhole + image.getX0(), yWhole + image.getY0())
        return image

    def computeKernelImage(self, fiberId, wavelength):
        """Return an image with the PSF centered at 0,0

        The returned image will have ``xy0`` (the offset between the physical
        image coordinate frame and the parent coordinate frame) set so that the
        PSF is centered at `0,0` in the parent coordinate frame.

        This is appropriate for visualizing the PSF. It is also used in this
        base class implementation to provide an image of the PSF that can be
        shifted (for ``computeImage``).

        Parameters
        ----------
        fiberId : `int`
            Fiber identifier.
        wavelength : `float`
            Wavelength.

        Returns
        -------
        image : `lsst.afw.image.Image`
            Image of the PSF.
        """
        return self.computeKernelImageXY(self.detectorMap.findPoint(fiberId, wavelength))

    @abstractmethod
    def computeKernelImageXY(self, point):
        """Return an image with the PSF centered at 0,0 (when xy0 is respected)

        This is appropriate for visualizing the PSF. It is also used in this
        base class implementation to provide an image of the PSF that can be
        shifted (for ``computeImage``).

        This is an abstract method, to be implemented by subclasses.

        Parameters
        ----------
        point : `lsst.geom.Point2D`
            Image position at which to calculate.

        Returns
        -------
        image : `lsst.afw.image.Image`
            Image of the PSF.
        """
        raise NotImplementedError("Subclass must define")

    def computeShape(self, fiberId, wavelength):
        """Return the shape of the PSF

        Parameters
        ----------
        fiberId : `int`
            Fiber identifier.
        wavelength : `float`
            Wavelength.

        Returns
        -------
        shape : `lsst.afw.geom.ellipses.Axes`
            Shape of the PSF.
        """
        return self.computeShapeXY(self.detectorMap.findPoint(fiberId, wavelength))

    @abstractmethod
    def computeShapeXY(self, point):
        """Return the shape of the PSF

        This is an abstract method, to be implemented by subclasses.

        Parameters
        ----------
        point : `lsst.geom.Point2D`
            Image position at which to calculate.

        Returns
        -------
        shape : `lsst.afw.geom.ellipses.Axes`
            Shape of the PSF.
        """
        raise NotImplementedError("Subclass must define")

    @abstractmethod
    def clone(self):
        """Polymorphic deep copy"""
        raise NotImplementedError("Subclass must define")


class ImagePsf(SpectralPsf):
    """ A SpectralPsf based on an imaging PSF

    This allows us to use a traditional imaging PSF from LSST as a SpectralPsf.

    Parameters
    ----------
    imagePsf : `lsst.afw.detection.Psf`
        LSST imaging PSF.
    detectorMap : `pfs.drp.stella.DetectorMap`
        Mapping from fiberId,wavelength <--> x,y.
    warpAlgorithm : `str`
        Warp kernel to use in ``computeImageXY``.
    warpBuffer : `int`
        Buffer around image to use ``computeImageXY``.
    """
    def __init__(self, imagePsf, *args, **kwargs):
        self.imagePsf = imagePsf
        super().__init__(*args, **kwargs)

    def computeKernelImageXY(self, point):
        """Return an image with the PSF centered at 0,0 (when xy0 is respected)

        This is appropriate for visualizing the PSF. It is also used in this
        base class implementation to provide an image of the PSF that can be
        shifted (for ``computeImage``).

        This implementation defers to the LSST imaging PSF.

        Parameters
        ----------
        point : `lsst.geom.Point2D`
            Image position at which to calculate.

        Returns
        -------
        image : `lsst.afw.image.Image`
            Image of the PSF.
        """
        return self.imagePsf.computeKernelImage(point)

    def computeShapeXY(self, point):
        """Return the shape of the PSF

        This implementation defers to the LSST imaging PSF.

        Parameters
        ----------
        point : `lsst.geom.Point2D`
            Image position at which to calculate.

        Returns
        -------
        shape : `lsst.afw.geom.ellipses.Axes`
            Shape of the PSF.
        """
        return self.imagePsf.computeShape(point)

    def clone(self):
        """Polymorphic deep copy"""
        return type(self)(self.imagePsf.clone(), self.detectorMap, self.warpAlgorithm, self.warpBuffer)

    @classmethod
    def _fromPickle(cls, psfData, detectorMap):
        """Construct from pickle

        Parameters
        ----------
        psfData : `bytes`
            Data stream describing the PSF.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength <--> x,y.

        Returns
        -------
        self : cls
            Constructed object.
        """
        size = len(psfData)
        manager = MemFileManager(size)
        manager.setData(psfData, size)
        psf = Psf.readFits(manager)
        return cls(psf, detectorMap)

    def __reduce__(self):
        """Pickling support"""
        manager = MemFileManager()
        self.imagePsf.writeFits(manager)
        return ImagePsf._fromPickle, (manager.getData(), self.detectorMap)
