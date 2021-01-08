from types import SimpleNamespace

import numpy as np
import astropy.io.fits

__all__ = ("ArcLine", "ArcLineSet")


class ArcLine(SimpleNamespace):
    """Data for a single reference line

    Parameters
    ----------
    fiberId : `int`
        Fiber identifier.
    wavelength : `float`
        Reference line wavelength (nm).
    x, y : `float`
        Measured position.
    xErr, yErr : `float`
        Error in measured position.
    flag : `bool`
        Measurement flag (``True`` indicates an error in measurement).
    status : `pfs.drp.stella.ReferenceLine.Status`
        Flags whether the lines are fitted, clipped or reserved etc.
    description : `str`
        Line description (e.g., ionic species)
    """
    def __init__(self, fiberId, wavelength, x, y, xErr, yErr, flag, status, description):
        return super().__init__(fiberId=fiberId, wavelength=wavelength, x=x, y=y, xErr=xErr, yErr=yErr,
                                flag=flag, status=status, description=description)

    def __reduce__(self):
        """Pickling"""
        return type(self), (self.fiberId, self.wavelength, self.x, self.y, self.xErr, self.yErr,
                            self.flag, self.status, self.description)


class ArcLineSet:
    """A list of `ArcLine`s.

    Parameters
    ----------
    lines : `list` of `ArcLine`
        List of lines in the spectra.
    """
    fitsExtName = "ARCLINES"

    def __init__(self, lines):
        self.lines = lines

    @property
    def fiberId(self):
        """Array of fiber identifiers (`numpy.ndarray` of `int`)"""
        return np.array([ll.fiberId for ll in self.lines])

    @property
    def wavelength(self):
        """Array of reference wavelength, nm (`numpy.ndarray` of `float`)"""
        return np.array([ll.wavelength for ll in self.lines])

    @property
    def x(self):
        """Array of x position (`numpy.ndarray` of `float`)"""
        return np.array([ll.x for ll in self.lines])

    @property
    def y(self):
        """Array of y position (`numpy.ndarray` of `float`)"""
        return np.array([ll.y for ll in self.lines])

    @property
    def xErr(self):
        """Array of error in x position (`numpy.ndarray` of `float`)"""
        return np.array([ll.xErr for ll in self.lines])

    @property
    def yErr(self):
        """Array of error in y position (`numpy.ndarray` of `float`)"""
        return np.array([ll.yErr for ll in self.lines])

    @property
    def flag(self):
        """Array of measurement status flags (`numpy.ndarray` of `int`)"""
        return np.array([ll.flag for ll in self.lines])

    @property
    def status(self):
        """Array of `ReferenceLine` status flags (`numpy.ndarray` of `int`)"""
        return np.array([ll.status for ll in self.lines])

    @property
    def description(self):
        """Array of description (`numpy.ndarray` of `str`)"""
        return np.array([ll.description for ll in self.lines])

    def __len__(self):
        """Number of lines"""
        return len(self.lines)

    def __iter__(self):
        """Iterator"""
        return iter(self.lines)

    def extend(self, lines):
        """Extend the list of lines

        Parameters
        ----------
        lines : iterable of `ArcLine`
            List of lines to add.
        """
        self.lines.extend(lines)

    def __add__(self, rhs):
        """Addition"""
        return type(self)(self.lines + rhs.lines)

    def __iadd__(self, rhs):
        """In-place addition"""
        self.extend(rhs.lines)
        return self

    def append(self, fiberId, wavelength, x, y, xErr, yErr, flag, status, description):
        """Append to the list of lines

        Parameters
        ----------
        fiberId : `int`
            Fiber identifier.
        wavelength : `float`
            Reference line wavelength (nm).
        x, y : `float`
            Measured position.
        xErr, yErr : `float`
            Error in measured position.
        flag : `bool`
            Measurement flag (``True`` indicates an error in measurement).
        status : `pfs.drp.stella.ReferenceLine.Status`
            Flags whether the lines are fitted, clipped or reserved etc.
        description : `str`
            Line description (e.g., ionic species)
        """
        self.lines.append(ArcLine(fiberId, wavelength, x, y, xErr, yErr, flag, status, description))

    @classmethod
    def empty(cls):
        """Construct an empty ArcLineSet"""
        return cls([])

    @classmethod
    def readFits(cls, filename):
        """Read from file

        Parameters
        ----------
        filename : `str`
            Name of file from which to read.

        Returns
        -------
        self : cls
            Constructed object from reading file.
        """
        with astropy.io.fits.open(filename) as fits:
            hdu = fits[cls.fitsExtName]
            fiberId = hdu.data["fiberId"].astype(np.int32)
            wavelength = hdu.data["wavelength"].astype(float)
            x = hdu.data["x"].astype(float)
            y = hdu.data["y"].astype(float)
            xErr = hdu.data["xErr"].astype(float)
            yErr = hdu.data["yErr"].astype(float)
            flag = hdu.data["flag"].astype(np.int32)
            status = hdu.data["status"].astype(np.int32)
            description = hdu.data["description"]

        return cls([ArcLine(*args) for args in zip(fiberId, wavelength, x, y, xErr, yErr, flag, status,
                                                   description)])

    def writeFits(self, filename):
        """Write to file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        hdu = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="fiberId", format="J", array=self.fiberId),
            astropy.io.fits.Column(name="wavelength", format="D", array=self.wavelength),
            astropy.io.fits.Column(name="x", format="D", array=self.x),
            astropy.io.fits.Column(name="y", format="D", array=self.y),
            astropy.io.fits.Column(name="xErr", format="D", array=self.xErr),
            astropy.io.fits.Column(name="yErr", format="D", array=self.yErr),
            astropy.io.fits.Column(name="flag", format="J", array=self.flag),
            astropy.io.fits.Column(name="status", format="J", array=self.status),
            astropy.io.fits.Column(name="description", format="A", array=self.description),
        ], name=self.fitsExtName)
        hdu.header["INHERIT"] = True

        fits = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(), hdu])
        with open(filename, "wb") as fd:
            fits.writeto(fd)
