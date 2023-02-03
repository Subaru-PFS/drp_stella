import os
import re
from math import log2

import numpy as np

from lsst.utils import getPackageDir

from pfs.datamodel import MaskHelper
from pfs.datamodel.pfsTable import PfsTable, Column
from .utils.bitmask import Bitmask
from .table import Table

__all__ = ("ReferenceLineStatus", "ReferenceLine", "ReferenceLineSet", "ReferenceLineSource")


class ReferenceLineStatus(Bitmask):
    """Bitmasks for quality of reference lines"""
    GOOD = 0x00, "Line is good"
    NOT_VISIBLE = 0x01, "Line is not typically visible in PFS (maybe too faint)"
    BLEND = 0x02, "Line is blended with other line(s)"
    SUSPECT = 0x04, "Line is of suspect quality"
    REJECTED = 0x08, "Line has been rejected from use in PFS"
    BROAD = 0x10, "Line is broader than normal"
    DETECTORMAP_USED = 0x20, "Used for fitting detectorMap"
    DETECTORMAP_RESERVED = 0x40, "Reserved during fitting detectorMap"
    SKYSUB_USED = 0x80, "Used for 2d sky subtraction"
    MERGED = 0x100, "Line has been merged into another"
    COMBINED = 0x200, "Line created from MERGED lines"
    PROTECTED = 0x400, "Line is protected from discard by exclusion zone"
    BAD = 0x01F, "Line is bad for any reason"

    @classmethod
    def getMasks(cls):
        """Return the corresponding `MaskHelper`

        for representing the `ReferenceLineStatus` as a bitmask in a
        `pfs.datamodel.PfsFiberArraySet` or similar.
        """
        def findBitSet(bitmask: ReferenceLineStatus) -> int:
            """Return the position of the only single bit set, or zero

            https://stackoverflow.com/a/51094793/834250
            """
            value = bitmask.value
            if value == 0 or (value & (value - 1)) != 0:
                return 0
            return int(log2(value)) + 1

        bits = {name: findBitSet(value) for name, value in cls.__members__.items()}
        return MaskHelper(**{name: number - 1 for name, number in bits.items() if number != 0})


class ReferenceLineTable(PfsTable):
    """Table of reference lines

    These are usually arc or sky line catalogs.

    Parameters
    ----------
    description : `numpy.ndarray` of `str`
        Description of each line; usually the atomic or molecular
        identification.
    wavelength : `numpy.ndarray` of `float`
        Reference line wavelengths (nm).
    intensity : `numpy.ndarray` of `float`
        Estimated intensities (arbitrary units).
    status : `numpy.ndarray` of `int`
        Bitmask indicating the quality of each line.
    transition: `str`
        Line transitions, or UNKNOWN if not known
    source: `numpy.ndarray` of `int`
        Bitmask indicating the reference source information.
    """
    schema = [
        Column("description", str, "Description of line; usually the atomic or molecular identification", ""),
        Column("wavelength", float, "Line wavelength (nm)", np.nan),
        Column("intensity", float, "Estimated intensity (arbitrary units)", np.nan),
        Column("status", np.int32, "Line status bitmask", -1),
        Column("transition", str, "Line transition", "UNKNOWN"),
        Column("source", np.int32, "Source information bitmask", -1),
    ]
    fitsExtName = "REFLINES"


class ReferenceLineSource(Bitmask):
    """Source information for reference line"""
    NONE = 0x00, 'No source information available'
    MANUAL = 0x01, 'Lines added or modified by hand'
    NIST = 0x02, 'Lines taken from NIST'
    OSTERBROCK97 = 0x04, 'Osterbrock+1996 (1996PASP..108..277O) and Osterbrock+1997 (1997PASP..109..614O)'
    ROUSSELOT2000 = 0x08, 'Rousselot+2000 (2000A&A...354.1134R)'
    SDSS = 0x10, 'Lines taken from SDSS (https://www.sdss.org/dr14/spectro/spectro_basics)'


class ReferenceLineSet(Table):
    """A list of `ReferenceLine`s.

    Parameters
    ----------
    data : `pandas.DataFrame`
        Table data.
    """
    DamdClass = ReferenceLineTable

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

        lines = []
        with open(filename) as fd:
            for ii, line in enumerate(fd):
                line = re.sub(r"\s*#.*$", "", line).rstrip()  # strip comments
                if not line:
                    continue

                fields = line.split()
                try:
                    # Support linelists that do not have transition or source information
                    if len(fields) == 4:
                        wavelength, intensity, description, status = fields
                        transition = "UNKNOWN"
                        source = ReferenceLineSource.NONE
                    else:
                        wavelength, intensity, description, status, transition, source = fields
                except Exception as ex:
                    raise RuntimeError(f"Unable to parse line {ii} of {filename}: {ex}")

                try:
                    intensity = float(intensity)
                except ValueError:
                    intensity = np.nan

                lines.append(
                    ReferenceLine(
                        description=description,
                        wavelength=float(wavelength),
                        intensity=intensity,
                        status=ReferenceLineStatus(int(status)),
                        transition=transition,
                        source=source,
                    )
                )

        return cls.fromRows(lines)

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

    def sort(self):
        """Sort the line list, in-place

        Lines are sorted by species and then by wavelength.
        """
        self.data.sort_values(by=["description", "wavelength", "status"], inplace=True)

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
            print("# 5: transition", file=fd)
            print("# 6: source reference", file=fd)
            print("#", file=fd)
            print("# Status bitmask elements:", file=fd)
            for flag in ReferenceLineStatus:
                if flag != ReferenceLineStatus.BAD:
                    print(f"# {flag.name}={flag.value}: {flag.__doc__}", file=fd)
            print("#", file=fd)
            print("# Source bitmask elements:", file=fd)
            for flag in ReferenceLineSource:
                print(f"# {flag.name}={flag.value}: {flag.__doc__}", file=fd)
            print("#", file=fd)
            for line in self.rows:
                print(f"{line.wavelength:<12.5f} {line.intensity:12.2f}    "
                      f"{line.description:7s} {line.status:6d}",
                      f"{line.transition:7s} {line.source:3d}",
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


ReferenceLine = ReferenceLineSet.RowClass
