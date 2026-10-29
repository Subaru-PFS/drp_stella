import io
import astropy.io.fits

from lsst.utils import continueClass

import pfs.datamodel
from .Distortion import Distortion

__all__ = ("Distortion",)


@continueClass  # noqa: F811 (redefinition)
class Distortion:  # noqa: F811 (redefinition)
    """A pseudo-class, containing factory methods and derived methods

    This class cannot be instantiated. It exists to provide factory classes the
    ability to construct a ``Distortion`` based on the content of the source
    data. It also provides some "inherited" methods for the "derived" classes.
    We take this approach to avoid stepping on the inheritance hierarchy of a
    pybind11 class.

    To support this interface, classes should implement the ``fromDatamodel``
    and ``toDatamodel`` methods to convert to/from the representation in
    pfs.datamodel, and be registered via ``Distortion.register``.
    """
    _subclasses = {}  # Registered subclasses (mapping name to class)

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This is a pure-virtual base class")

    def __reduce__(self):
        """How to pickle"""
        return self.__class__.fromBytes, (self.toBytes(),)

    @classmethod
    def register(cls, SubClass: type):
        """Register a SubClass

        Parameters
        ----------
        SubClass : `type`
            Subclass of ``pfs::drp::stella::Distortion``.
        """
        assert SubClass not in cls._subclasses
        cls._subclasses[SubClass.__name__] = SubClass

    @classmethod
    def fromDatamodel(cls, distortion: pfs.datamodel.PfsDistortion) -> "Distortion":
        """Construct from subclass of pfs.datamodel.Distortion

        Converts from the datamodel representation to the drp_stella
        representation.

        Parameters
        ----------
        distortion : subclass of `pfs.datamodel.Distortion`
            Distortion to convert.

        Returns
        -------
        self : subclass of `pfs.drp.stella.Distortion`
            Converted distortion.
        """
        subclass = type(distortion).__name__
        return cls._subclasses[subclass].fromDatamodel(distortion)

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "Distortion":
        """Construct from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file.

        Returns
        -------
        self : subclass of ``pfs::drp::stella::Distortion``
            Instantiated subclass, read from the file.
        """
        name = fits[0].header["HIERARCH pfs_distortion_class"]
        distortion = pfs.datamodel.PfsDistortion.getDistortion(name).readFits(fits, 0)
        return cls.fromDatamodel(distortion)

    def toFits(self):
        """Generate a FITS file

        Returns
        -------
        fits : `astropy.io.fits.HDUList`
            FITS file.
        """
        distortion = self.toDatamodel()
        header = astropy.io.fits.Header()
        header["HIERARCH pfs_distortion_class"] = distortion.getName()
        fits = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(header=header)])
        distortion.writeFits(fits, 0)
        return fits

    @classmethod
    def readFits(cls, pathName: str) -> "Distortion":
        """Read Distortion from FITS

        Parameters
        ----------
        pathName : `str`
            Path to file from which to read.
        hdu : `int`, optional
            HDU from which to read; unused in this implementation.
        flags : `int`, optional
            Flags for reading; unused in this implementation.

        Raises
        ------
        NotImplementedError
            If ``hdu`` or ``flags`` are requested.

        Returns
        -------
        out : `pfs.drp.stella.Distortion`
            Distortion read from FITS file.
        """
        with astropy.io.fits.open(pathName) as fits:
            return cls.fromFits(fits)

    def writeFits(self, pathName: str):
        """Write Distortion to FITS

        Parameters
        ----------
        pathName : `str`
            Path of file to which to write.

        Raises
        ------
        NotImplementedError
            If ``flags`` are requested.
        """
        hdus = self.toFits()
        # clobber=True in writeto prints a message, so use open instead
        with open(pathName, "wb") as fd:
            hdus.writeto(fd)

    @classmethod
    def fromBytes(cls, string: bytes) -> "Distortion":
        """Construct from bytes

        Parameters
        ----------
        string : `bytes`
            String of bytes.

        Returns
        -------
        self : cls
            Constructed object.
        """
        with astropy.io.fits.open(io.BytesIO(string)) as fd:
            return cls.fromFits(fd)

    def toBytes(self):
        """Convert to bytes

        Returns
        -------
        string : `bytes`
            String of bytes.
        """
        fits = self.toFits()
        buffer = io.BytesIO()
        fits.writeto(buffer)
        fits.close()
        return buffer.getvalue()
