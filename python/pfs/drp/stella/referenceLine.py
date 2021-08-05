import os
import re
from types import SimpleNamespace

import numpy as np

from lsst.utils import getPackageDir

from .utils.bitmask import Bitmask

__all__ = ("ReferenceLineStatus", "ReferenceLine", "ReferenceLineSet")


class ReferenceLineStatus(Bitmask):
    """Bitmasks for quality of reference lines"""
    GOOD = 0x00, "Line is good"
    NOT_VISIBLE = 0x01, "Line is not typically visible in PFS (maybe too faint)"
    BLEND = 0x02, "Line is blended with other line(s)"
    SUSPECT = 0x04, "Line is of suspect quality"
    REJECTED = 0x08, "Line has been rejected from use in PFS"
    BROAD = 0x10, "Line is broader than normal"
    BAD = 0xFFFF, "Line is bad for any reason"


class ReferenceLine(SimpleNamespace):
    """A reference arc line

    Parameters
    ----------
    description : `str`
        Description of the line; usually the atomic or molecular identification.
    wavelength : `float`
        Reference line wavelength (nm).
    intensity : `float`
        Estimated intensity (arbitrary units).
    status : `int`
        Bitmask indicating the quality of the line.
    """
    def __init__(self, description, wavelength, intensity, status):
        super().__init__(description=description, wavelength=wavelength, intensity=intensity, status=status)

    def __reduce__(self):
        """Pickling"""
        return type(self), (self.description, self.wavelength, self.intensity, self.status)


class ReferenceLineSet:
    """A list of `ReferenceLine`s.

    Parameters
    ----------
    lines : iterable of `ReferenceLine`
        List of reference lines.
    """
    def __init__(self, lines):
        self.lines = list(lines)

    @property
    def description(self):
        """Array of line description (`numpy.ndarray` of `str`)"""
        return np.array([ll.description for ll in self.lines])

    @property
    def wavelength(self):
        """Array of line wavelength, nm (`numpy.ndarray` of `float`)"""
        return np.array([ll.wavelength for ll in self.lines], dtype=float)

    @property
    def intensity(self):
        """Array of line intensity (`numpy.ndarray` of `float`)"""
        return np.array([ll.intensity for ll in self.lines], dtype=float)

    @property
    def status(self):
        """Array of line status (`numpy.ndarray` of `int`)"""
        return np.array([ll.status for ll in self.lines], dtype=np.int32)

    def __len__(self):
        """Number of lines"""
        return len(self.lines)

    def __iter__(self):
        """Iterator"""
        return iter(self.lines)

    def __getitem__(self, index):
        """Retrieve by index"""
        return self.lines[index]

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

    def append(self, description, wavelength, intensity, status):
        """Append to the list of lines

        Parameters
        ----------
        description : `str`
            Description of the line; usually the atomic or molecular identification.
        wavelength : `float`
            Reference line wavelength (nm).
        intensity : `float`
            Estimated intensity (arbitrary units).
        status : `int`
            Bitmask indicating the quality of the line.
        """
        self.lines.append(ReferenceLine(description, wavelength, intensity, status))

    @classmethod
    def empty(cls):
        """Construct an empty `ReferenceLineSet`"""
        return cls([])

    @classmethod
    def fromLineList(cls, filename):
        """Construct from a line list text file

        The line list is a text file with entry on each line, with the
        wavelength (nm), intensity (arbitrary units), description and status
        separated by whitespace. The hash character (``#``) causes the rest of
        the line to be ignored.

        Parameters
        ----------
        filename : `str`
            Filename of line list. If this is not an absolute path, we attempt
            to read a file of that name from ``obs_pfs/pfs/lineLists/``.

        Returns
        -------
        self : `ReferenceLineSet`
            Reference line list.
        """
        if not os.path.isabs(filename):
            absFilename = os.path.join(getPackageDir("obs_pfs"), "pfs", "lineLists", filename)
            if os.path.exists(absFilename):
                filename = absFilename

        self = cls.empty()
        with open(filename) as fd:
            for ii, line in enumerate(fd):
                line = re.sub(r"\s*#.*$", "", line).rstrip()  # strip comments
                if not line:
                    continue

                fields = line.split()
                try:
                    wavelength, intensity, description, status = fields
                except Exception as ex:
                    raise RuntimeError(f"Unable to parse line {ii} of {filename}: {ex}")

                try:
                    intensity = float(intensity)
                except ValueError:
                    intensity = np.nan

                self.append(description, float(wavelength), intensity, ReferenceLineStatus(int(status)))

        return self

    def sort(self):
        """Sort the line list, in-place

        Lines are sorted by species and then by wavelength.
        """
        self.lines = sorted(self.lines, key=lambda ll: (ll.description, ll.wavelength, ll.status))

    def writeLineList(self, filename):
        """Write a line list text file

        This is intended for programmatically updating a line list file, e.g.,
        to correct intensity values using measurements.

        No comments are propagated from former versions, but we provide a
        helpful header.

        Parameters
        ----------
        filename : `str`
            Filename of line list.
        """
        with open(filename, "w") as fd:
            print("# Columns:", file=fd)
            print("# 1: wavelength (nm)", file=fd)
            print("# 2: intensity (arbitrary units)", file=fd)
            print("# 3: description (ionic species)", file=fd)
            print("# 4: status (bitmask)", file=fd)
            print("#", file=fd)
            print("# Status bitmask elements:", file=fd)
            for flag in ReferenceLineStatus:
                if flag != ReferenceLineStatus.BAD:
                    print(f"# {flag.name}={flag.value}: {flag.__doc__}", file=fd)
            print("#", file=fd)
            for line in self.lines:
                print(f"{line.wavelength:<12.5f} {line.intensity:12.2f}    "
                      f"{line.description:7s} {line.status:6d}",
                      file=fd)

    def plot(self, axes, ls='-', alpha=1, color=None, label=None,
             labelStatus=True, labelLines=False, pixels=False, wavelength=None, spectrum=None):
        """Plot a set of reference lines using axvline

        Parameters
        ----------
        axes : `matplotlib.Axes`
            Plot axes.
        ls : `str`, optional
            Line style for plotting.
        alpha : `float`, optional
            Transparency.
        color : various, optional
            Matplotlib color to use (``None`` means let matplotlib choose).
        label : `str`, optional
            Label for lines (``None`` means use ``what`` or ``what status``).
        labelStatus : `bool`, optional
            Include status in labels?
        labelLines : `bool`, optional
            Label lines? Lines will be labelled at the top of the plot unless a
            ``spectrum`` is provided, in which case the labels will appear near
            the peaks of the lines.
        pixels : `bool`, optional
            Plot is in pixels? If so, you need to provide a ``wavelength`` array
            as well.
        wavelength : array_like, optional
            Wavelength array for underlying plot; for setting the horizontal
            position when the plot is in pixels, and finding the intensity
            when labeling lines.
        spectrum : array_like, optional
            Intensity array for underlying plot; for setting the vertical
            position of labels.
        """
        if label == "":
            label = None
        if pixels:
            if wavelength is None:
                raise RuntimeError("No wavelength array provided")
            from scipy.interpolate import interp1d
            interpolate = interp1d(wavelength, np.arange(len(wavelength)), bounds_error=False)

        if len(axes.get_lines()) > 0:  # they've plotted something already
            xlim = axes.get_xlim()
        else:
            xlim = None

        usedLabels = set()  # labels that we've used if labelLines is True
        for rl in self:
            if (rl.status & ReferenceLineStatus.NOT_VISIBLE):
                color = 'black'
                label = "not visible"
            if (rl.status & ReferenceLineStatus.BLEND):
                color = 'blue'
                label = "blended"
            elif (rl.status & ReferenceLineStatus.SUSPECT):
                color = 'magenta'
                label = "suspect"
            elif (rl.status & ReferenceLineStatus.REJECTED):
                color = 'red'
                label = "rejected"
            elif (rl.status & ReferenceLineStatus.BROAD):
                color = 'brown'
                label = "broad"
            else:
                color = 'green'
                label = "good"

            label = label if label not in usedLabels else None
            usedLabels.add(label)

            if pixels:
                x = interpolate(rl.wavelength)
            else:
                x = rl.wavelength
            if not np.isfinite(x):
                continue

            axes.axvline(x, ls=ls, color=color, alpha=alpha, label=label)

            if labelLines:
                if xlim is not None and not (xlim[0] < x < xlim[1]):
                    continue

                if wavelength is None or spectrum is None:
                    y = 0.95*axes.get_ylim()[1]
                else:
                    ix = np.searchsorted(wavelength, rl.wavelength)

                    if ix <= 0 or ix >= len(wavelength):
                        continue

                    i0 = max(0, int(ix) - 2)
                    i1 = min(len(spectrum), int(ix) + 2 + 1)
                    y = 1.05*spectrum[i0:i1].max()

                axes.text(x, y, rl.description, ha='center')

        axes.set_xlim(xlim)
