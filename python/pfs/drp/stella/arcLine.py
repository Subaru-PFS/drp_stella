from types import SimpleNamespace

import numpy as np
import astropy.io.fits

from .referenceLine import ReferenceLineSet

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
    intensity : `float`
        Measured intensity (arbitrary units).
    intensityErr : `float`
        Error in measured intensity (arbitrary units).
    flag : `bool`
        Measurement flag (``True`` indicates an error in measurement).
    status : `int`
        Bitmask indicating the quality of the reference line.
    description : `str`
        Line description (e.g., ionic species)
    """
    def __init__(self, fiberId, wavelength, x, y, xErr, yErr, intensity, intensityErr,
                 flag, status, description):
        return super().__init__(fiberId=fiberId, wavelength=wavelength, x=x, y=y, xErr=xErr, yErr=yErr,
                                intensity=intensity, intensityErr=intensityErr, flag=flag,
                                status=status, description=description)

    def __reduce__(self):
        """Pickling"""
        return type(self), (self.fiberId, self.wavelength, self.x, self.y, self.xErr, self.yErr,
                            self.intensity, self.intensityErr, self.flag, self.status, self.description)


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
        return np.array([ll.fiberId for ll in self.lines], dtype=np.int32)

    @property
    def wavelength(self):
        """Array of reference wavelength, nm (`numpy.ndarray` of `float`)"""
        return np.array([ll.wavelength for ll in self.lines], dtype=float)

    @property
    def x(self):
        """Array of x position (`numpy.ndarray` of `float`)"""
        return np.array([ll.x for ll in self.lines], dtype=float)

    @property
    def y(self):
        """Array of y position (`numpy.ndarray` of `float`)"""
        return np.array([ll.y for ll in self.lines], dtype=float)

    @property
    def xErr(self):
        """Array of error in x position (`numpy.ndarray` of `float`)"""
        return np.array([ll.xErr for ll in self.lines], dtype=float)

    @property
    def yErr(self):
        """Array of error in y position (`numpy.ndarray` of `float`)"""
        return np.array([ll.yErr for ll in self.lines], dtype=float)

    @property
    def intensity(self):
        """Array of intensity (`numpy.ndarray` of `float`)"""
        return np.array([ll.intensity for ll in self.lines], dtype=float)

    @property
    def intensityErr(self):
        """Array of intensity error (`numpy.ndarray` of `float`)"""
        return np.array([ll.intensityErr for ll in self.lines], dtype=float)

    @property
    def flag(self):
        """Array of measurement status flags (`numpy.ndarray` of `bool`)"""
        return np.array([ll.flag for ll in self.lines], dtype=bool)

    @property
    def status(self):
        """Array of `ReferenceLine` status flags (`numpy.ndarray` of `int`)"""
        return np.array([ll.status for ll in self.lines], dtype=np.int32)

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

    def append(self, fiberId, wavelength, x, y, xErr, yErr, intensity, intensityErr,
               flag, status, description):
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
        intensity : `float`
            Measured intensity (arbitrary units).
        intensityErr : `float`
            Error in measured intensity (arbitrary units).
        flag : `bool`
            Measurement flag (``True`` indicates an error in measurement).
        status : `pfs.drp.stella.ReferenceLine.Status`
            Flags whether the lines are fitted, clipped or reserved etc.
        description : `str`
            Line description (e.g., ionic species)
        """
        self.lines.append(ArcLine(fiberId, wavelength, x, y, xErr, yErr, intensity, intensityErr,
                                  flag, status, description))

    @classmethod
    def empty(cls):
        """Construct an empty ArcLineSet"""
        return cls([])

    def extractReferenceLines(self, fiberId=None):
        """Generate a list of reference lines

        Parameters
        ----------
        fiberId : `int`, optional
            Use lines from this fiber exclusively. Otherwise, we'll average the
            intensities of lines with the same wavelength and description.

        Returns
        -------
        refLines : `pfs.drp.stella.ReferenceLineSet`
            Reference lines.
        """
        refLines = ReferenceLineSet.empty()
        if fiberId is not None:
            select = self.fiberId == fiberId
            for args in zip(self.description[select], self.wavelength[select], self.intensity[select],
                            self.status[select]):
                refLines.append(*args)
        else:
            unique = set(zip(self.wavelength, self.description, self.status))
            for wavelength, description, status in sorted(unique):
                select = ((self.description == description) & (self.wavelength == wavelength) &
                          (self.status == status) & np.isfinite(self.intensity))

                intensity = np.average(self.intensity[select]) if np.any(select) else np.nan
                refLines.append(description, wavelength, intensity, status)
        return refLines

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
            if "DAMD_VER" in hdu.header and hdu.header["DAMD_VER"] >= 1:
                intensity = hdu.data["intensity"].astype(float)
                intensityErr = hdu.data["intensityErr"].astype(float)
            else:
                intensity = np.full(len)(fiberId, np.nan, dtype=float)
                intensityErr = np.full(len)(fiberId, np.nan, dtype=float)
            flag = hdu.data["flag"].astype(np.int32)
            status = hdu.data["status"].astype(np.int32)
            description = hdu.data["description"]

        return cls([ArcLine(*args) for args in zip(fiberId, wavelength, x, y, xErr, yErr,
                                                   intensity, intensityErr, flag, status, description)])

    def writeFits(self, filename):
        """Write to file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        # NOTE: When making any changes to this method that modify the output
        # format, increment the DAMD_VER header value.

        lengths = [len(dd) for dd in self.description]
        longest = 1 if len(lengths) == 0 else max(lengths)  # 1 not 0 to make astropy happy
        hdu = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column(name="fiberId", format="J", array=self.fiberId),
            astropy.io.fits.Column(name="wavelength", format="D", array=self.wavelength),
            astropy.io.fits.Column(name="x", format="D", array=self.x),
            astropy.io.fits.Column(name="y", format="D", array=self.y),
            astropy.io.fits.Column(name="xErr", format="D", array=self.xErr),
            astropy.io.fits.Column(name="yErr", format="D", array=self.yErr),
            astropy.io.fits.Column(name="intensity", format="D", array=self.intensity),
            astropy.io.fits.Column(name="intensityErr", format="D", array=self.intensityErr),
            astropy.io.fits.Column(name="flag", format="J", array=self.flag),
            astropy.io.fits.Column(name="status", format="J", array=self.status),
            astropy.io.fits.Column(name="description", format=f"{longest}A", array=self.description),
        ], name=self.fitsExtName)
        hdu.header["INHERIT"] = True
        hdu.header["DAMD_VER"] = (1, "ArcLineSet datamodel version")

        fits = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(), hdu])
        with open(filename, "wb") as fd:
            fits.writeto(fd)
