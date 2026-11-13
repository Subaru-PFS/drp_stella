import io

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
    def readFits(cls, pathName: str) -> "Distortion":
        """Read Distortion from FITS

        Parameters
        ----------
        pathName : `str` or file-like object
            Path to file (or file-like object) from which to read.

        Returns
        -------
        out : `pfs.drp.stella.Distortion`
            Distortion read from FITS file.
        """
        return cls.fromDatamodel(pfs.datamodel.PfsDistortion.readFits(pathName))

    def writeFits(self, pathName: str):
        """Write Distortion to FITS

        Parameters
        ----------
        pathName : `str` or file-like object
            Path of file (or file-like object) to which to write.
        """
        self.toDatamodel().writeFits(pathName)

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
        return cls.readFits(io.BytesIO(string))

    def toBytes(self) -> bytes:
        """Convert to bytes

        Returns
        -------
        string : `bytes`
            String of bytes.
        """
        buffer = io.BytesIO()
        self.writeFits(buffer)
        return buffer.getvalue()
