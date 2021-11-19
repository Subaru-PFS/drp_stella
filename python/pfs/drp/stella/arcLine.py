from dataclasses import dataclass
from typing import Iterable

import numpy as np
from pandas import DataFrame
import astropy.io.fits

from pfs.datamodel import Identity

from .referenceLine import ReferenceLineSet, ReferenceLineStatus
from .datamodel.pfsFiberArraySet import PfsFiberArraySet

__all__ = ("ArcLine", "ArcLineSet")


@dataclass
class ArcLine:
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
    fiberId: int
    wavelength: float
    x: float
    y: float
    xErr: float
    yErr: float
    intensity: float
    intensityErr: float
    flag: bool
    status: int
    description: str


class ArcLineSet:
    """A list of `ArcLine`s.

    Parameters
    ----------
    data : `pandas.DataFrame`
        Arc line data.
    """
    fitsExtName = "ARCLINES"
    RowClass = ArcLine
    schema = (("fiberId", np.int32),
              ("wavelength", float),
              ("x", float),
              ("y", float),
              ("xErr", float),
              ("yErr", float),
              ("intensity", float),
              ("intensityErr", float),
              ("flag", bool),
              ("status", np.int32),
              ("description", str),
              )

    def __init__(self, data: DataFrame):
        self.data = data

    @classmethod
    def fromArcLines(cls, lines: Iterable[ArcLine]):
        return cls.fromArrays(**{name: np.array([getattr(ll, name) for ll in lines], dtype=dtype) for
                                 name, dtype in cls.schema})

    @classmethod
    def fromArrays(cls, **kwargs):
        return cls(DataFrame({name: np.array(kwargs[name], dtype=dtype) for name, dtype in cls.schema}))

    @property
    def lines(self):
        """Return list of `ArcLine`"""
        return [self.RowClass(**row[1].to_dict()) for row in self.data.iterrows()]

    @property
    def fiberId(self):
        """Array of fiber identifiers (`numpy.ndarray` of `int`)"""
        return self.data["fiberId"].values

    @property
    def wavelength(self):
        """Array of reference wavelength, nm (`numpy.ndarray` of `float`)"""
        return self.data["wavelength"].values

    @property
    def x(self):
        """Array of x position (`numpy.ndarray` of `float`)"""
        return self.data["x"].values

    @property
    def y(self):
        """Array of y position (`numpy.ndarray` of `float`)"""
        return self.data["y"].values

    @property
    def xErr(self):
        """Array of error in x position (`numpy.ndarray` of `float`)"""
        return self.data["xErr"].values

    @property
    def yErr(self):
        """Array of error in y position (`numpy.ndarray` of `float`)"""
        return self.data["yErr"].values

    @property
    def intensity(self):
        """Array of intensity (`numpy.ndarray` of `float`)"""
        return self.data["intensity"].values

    @property
    def intensityErr(self):
        """Array of intensity error (`numpy.ndarray` of `float`)"""
        return self.data["intensityErr"].values

    @property
    def flag(self):
        """Array of measurement status flags (`numpy.ndarray` of `bool`)"""
        return self.data["flag"].values

    @property
    def status(self):
        """Array of `ReferenceLine` status flags (`numpy.ndarray` of `int`)"""
        return self.data["status"].values

    @property
    def description(self):
        """Array of description (`numpy.ndarray` of `str`)"""
        return self.data["description"].values

    def __len__(self) -> int:
        """Number of lines"""
        return len(self.data)

    def __iter__(self):
        """Iterator"""
        return iter(self.lines)

    def __getitem__(self, index: int) -> ArcLine:
        """Retrieve by index"""
        return ArcLine(**self.data.iloc[index].to_dict())

    def extend(self, other: "ArcLineSet"):
        """Extend the list of lines

        This is an inefficient way of populating an `ArcLineSet`.

        Parameters
        ----------
        lines : `ArcLineSet`
            List of lines to add.
        """
        self.data = self.data.append(other.data)

    def __add__(self, rhs: "ArcLineSet") -> "ArcLineSet":
        """Addition

        This is an inefficient way of populating an `ArcLineSet`.
        """
        return type(self)(self.data.append(rhs.data))

    def __iadd__(self, rhs: "ArcLineSet") -> "ArcLineSet":
        """In-place addition

        This is an inefficient way of populating an `ArcLineSet`.
        """
        self.extend(rhs)
        return self

    def copy(self) -> "ArcLineSet":
        """Return a deep copy"""
        return type(self)(self.data.copy())

    @classmethod
    def empty(cls) -> "ArcLineSet":
        """Construct an empty ArcLineSet"""
        return cls.fromArrays(**{name: [] for name, _ in cls.schema})

    def extractReferenceLines(self, fiberId: int = None) -> ReferenceLineSet:
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

    def applyExclusionZone(self, exclusionRadius: float,
                           status: ReferenceLineStatus = ReferenceLineStatus.BLEND
                           ):
        """Apply an exclusion zone around each line

        A line cannot have another line within ``exclusionRadius``.

        The line list is modified in-place.

        Parameters
        ----------
        exclusionRadius : `float`
            Radius in wavelength (nm) to apply around lines.
        status : `ReferenceLineStatus`
            Status to apply to lines that fall within the exclusion zone.
        """
        from .applyExclusionZone import applyExclusionZone
        return applyExclusionZone(self, exclusionRadius, status)

    @classmethod
    def readFits(cls, filename: str) -> "ArcLineSet":
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
        data = {}
        with astropy.io.fits.open(filename) as fits:
            hdu = fits[cls.fitsExtName]
            data["fiberId"] = hdu.data["fiberId"].astype(np.int32)
            data["wavelength"] = hdu.data["wavelength"].astype(float)
            data["x"] = hdu.data["x"].astype(float)
            data["y"] = hdu.data["y"].astype(float)
            data["xErr"] = hdu.data["xErr"].astype(float)
            data["yErr"] = hdu.data["yErr"].astype(float)
            if "DAMD_VER" in hdu.header and hdu.header["DAMD_VER"] >= 1:
                data["intensity"] = hdu.data["intensity"].astype(float)
                data["intensityErr"] = hdu.data["intensityErr"].astype(float)
            else:
                data["intensity"] = np.full(len(hdu), np.nan, dtype=float)
                data["intensityErr"] = np.full(len(hdu), np.nan, dtype=float)
            data["flag"] = hdu.data["flag"].astype(np.int32)
            data["status"] = hdu.data["status"].astype(np.int32)
            data["description"] = hdu.data["description"]

        return cls(DataFrame(data))

    def writeFits(self, filename: str):
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

    def toDataFrame(self) -> DataFrame:
        """Convert to a `pandas.DataFrame`"""
        return self.data

    def asPfsFiberArraySet(self, identity: Identity = None) -> PfsFiberArraySet:
        """Represent as a PfsFiberArraySet

        This can be useful when fitting models of line intensities.

        Parameters
        ----------
        identity : `Identity`
            Identity to give the output `PfsFiberArraySet`.

        Returns
        -------
        spectra : `PfsFiberArraySet`
            Lines represented as a `PfsFiberArraySet`.
        """
        fiberId = np.array(sorted(set(self.fiberId)), dtype=int)
        numFibers = fiberId.size
        wlSet = np.array(sorted(set(self.wavelength)), dtype=float)
        numWavelength = wlSet.size
        if identity is None:
            identity = Identity(-1)
        flags = ReferenceLineStatus.getMasks()
        metadata = {}

        wavelength = np.vstack([wlSet]*numFibers)
        flux = np.full((numFibers, numWavelength), np.nan, dtype=float)
        mask = np.full((numFibers, numWavelength), flags["REJECTED"], dtype=np.int32)
        covar = np.full((numFibers, 3, numWavelength), np.nan, dtype=float)
        variance = covar[:, 0, :]
        sky = np.zeros_like(flux)
        norm = np.ones_like(flux)

        wlLookup = {wl: ii for ii, wl in enumerate(wlSet)}
        fiberLookup = {ff: ii for ii, ff in enumerate(fiberId)}
        wlIndices = np.array([wlLookup[ll.wavelength] for ii, ll in enumerate(self)])
        fiberIndices = np.array([fiberLookup[ll.fiberId] for ii, ll in enumerate(self)])
        flux[fiberIndices, wlIndices] = self.intensity
        variance[fiberIndices, wlIndices] = self.intensityErr**2
        mask[fiberIndices, wlIndices] = self.status

        return PfsFiberArraySet(identity, fiberId, wavelength, flux, mask, sky, norm, covar, flags, metadata)
