from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task

from lsst.obs.pfs.utils import getLampElements
from .referenceLine import ReferenceLineSet
import re


__all__ = ("ReadLineListConfig", "ReadLineListTask")


class ReadLineListConfig(Config):
    """Configuration for ReadLineListTask"""
    lineListFiles = ListField(dtype=str, doc="list of names of linelist files",
                              default=["HgCd.txt", "Ar.txt", "Hg.txt", "Kr.txt",
                                       "Ne.txt", "Xe.txt", "skyLines.txt"])
    restrictByLamps = Field(dtype=bool, default=True,
                            doc="Restrict linelist by the list of active lamps? True is appropriate for arcs")
    lampElementList = ListField(dtype=str, doc="list of lamp elements; overrides the header", default=[])
    assumeSkyIfNoLamps = Field(dtype=bool, default=True,
                               doc="Assume that we're looking at sky if no lamps are active?")
    minIntensity = Field(dtype=float, default=0.0, doc="Minimum linelist intensity; <= 0 means no filtering")
    exclusionRadius = Field(dtype=float, default=0.0, doc="Exclusion radius around lines (nm)")


class ReadLineListTask(Task):
    """Read a linelist"""
    ConfigClass = ReadLineListConfig
    _DefaultName = "readLineList"

    def run(self, detectorMap=None, metadata=None):
        """Read a linelist

        This serves as a wrapper around the common operation of looking up the
        lamps, reading the linelist and formatting it.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Mapping from x,y to fiberId,wavelength. This is required in order to
            filter the linelist by wavelength.
        metadata : `lsst.daf.base.PropertyList`, optional
            FITS header, containing the lamp indicators. This is required in
            order to restrict the linelist by the list of active lamps.

        Returns
        -------
        lines : `pfs.drp.stella.ReferenceLineSet`
            Lines from the linelist.
        """
        lines = ReferenceLineSet.empty()
        for filename in self.config.lineListFiles:
            lines.extend(ReferenceLineSet.fromLineList(filename))
        if self.config.restrictByLamps:
            lampElements = self.getLampElements(metadata)
            lines = self.filterByLampElements(lines, lampElements)
        lines = self.filterByIntensity(lines)
        lines = self.filterByWavelength(lines, detectorMap)
        lines.applyExclusionZone(self.config.exclusionRadius)
        return lines

    def filterByLampElements(self, lines, lampElements):
        """Filter the line list by the elements in the active lamps

        Parameters
        ----------
        lines : `pfs.drp.stella.ReferenceLineSet`
            List of reference lines.
        elements : ``list` of `str`
            The list of elements in the active lamps.

        Returns
        -------
        filtered : `pfs.drp.stella.ReferenceLineSet`
            Filtered list of reference lines.
        """
        keep = []
        for element in lampElements:
            keep += [ll for ll in lines if re.match(f'^{element}[IVX]*$', ll.description)]
        self.log.info(f"Filtered line lists for elements: {sorted(lampElements)}, "
                      f"keeping species {[ll.description for ll in keep]}.")
        return ReferenceLineSet(keep)

    def filterByIntensity(self, lines):
        """Filter the line list by intensity level

        No filtering is applied if the configuration parameter
        ``minIntensity <= 0``.

        Parameters
        ----------
        lines : `pfs.drp.stella.ReferenceLineSet`
            List of reference lines.

        Returns
        -------
        filtered : `pfs.drp.stella.ReferenceLineSet`
            Filtered list of reference lines.
        """
        if self.config.minIntensity <= 0:
            return lines
        keep = [ll for ll in lines if ll.intensity >= self.config.minIntensity]
        return ReferenceLineSet(keep)

    def getLampElements(self, metadata):
        """Determine which lamps are active and return the lamp elements

        Parameters
        ----------
        metadata : `lsst.daf.base.PropertyList`, optional
            FITS header, containing the lamp indicators.

        Returns
        -------
        elements : `list` of `str`, or `None`
            The list of lamp elements, if the ``restrictByLamps`` configuration option
            is ``True`` or ``lampElementList`` is set; otherwise, ``None``.

        Raises
        ------
        RuntimeError
            If ``metadata`` is ``None`` and ``restrictByLamps`` is ``True``, and
            lampElementList is not set.
        """
        if len(self.config.lampElementList) > 0:
            return self.config.lampElementList
        if metadata is None:
            raise RuntimeError("Cannot determine lamp elements because metadata was not provided")
        lampElements = getLampElements(metadata)
        if not lampElements and self.config.assumeSkyIfNoLamps:
            self.log.info("No lamps on; assuming sky.")
            lampElements = ["OI", "NaI", "OH"]
        return lampElements

    def filterByWavelength(self, lines, detectorMap=None):
        """Filter the line list by wavelength

        Parameters
        ----------
        lines : `pfs.drp.stella.ReferenceLineSet`
            Lines from the linelist.
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Mapping from x,y to fiberId,wavelength. This is required in order to
            filter the linelist by wavelength.

        Returns
        -------
        filtered : `pfs.drp.stella.ReferenceLineSet`
            Filtered list of reference lines.
        """
        if detectorMap is None:
            return lines
        wavelength = detectorMap.getWavelength()
        minWl = wavelength.min()
        maxWl = wavelength.max()
        return ReferenceLineSet([rl for rl in lines if rl.wavelength >= minWl and
                                 rl.wavelength <= maxWl])
