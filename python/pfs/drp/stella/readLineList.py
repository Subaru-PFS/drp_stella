import numpy as np

from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Struct, Task

from lsst.obs.pfs.utils import getLamps, getLampElements
from .referenceLine import ReferenceLineSet
import re

__all__ = ("ReadLineListConfig", "ReadLineListTask")


class ReadLineListConfig(Config):
    """Configuration for ReadLineListTask"""
    lineListFiles = ListField(dtype=str,
                              doc="list of names of linelist files (overrides the OBS_PFS linelists)",
                              default=[])
    lampElementList = ListField(dtype=str,
                                doc="list of lamp elements or species to filter by", default=[])
    assumeSkyIfNoLamps = Field(dtype=bool, default=True,
                               doc="Assume that we're looking at sky if no lamps are active?")
    minIntensity = Field(dtype=float, default=0.0, doc="Minimum linelist intensity; <= 0 means no filtering")
    exclusionRadius = Field(dtype=float, default=0.0, doc="Exclusion radius around lines (nm)")


class ReadLineListTask(Task):
    """Read a linelist"""
    ConfigClass = ReadLineListConfig
    _DefaultName = "readLineList"
    _isSpeciesPattern = re.compile(r'^[A-Z][A-Za-z]*[IVX]+')

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
        lineListFiles = []
        lampElementList = []
        lampInfo = None

        if metadata:
            lampInfo = self.getLampInfo(metadata)

        if self.config.lampElementList:
            lampElementList = self.config.lampElementList
        else:
            if lampInfo:
                lampElementList = lampInfo.lampElementList

        if self.config.lineListFiles:
            lineListFiles = self.config.lineListFiles
        else:
            if lampInfo:
                lamps = lampInfo.lamps
                if lamps:
                    lineListFiles = [f'{lamp}.txt' for lamp in lamps]

        lines = ReferenceLineSet.empty()
        for filename in lineListFiles:
            lines.extend(ReferenceLineSet.fromLineList(filename))

        lines = self.filterByLampElements(lines, lampElementList)
        lines = self.filterByIntensity(lines)
        lines = self.filterByWavelength(lines, detectorMap)
        lines.applyExclusionZone(self.config.exclusionRadius)
        return lines

    def filterByLampElements(self, lines, lampElements):
        """Filter the line list by the elements or species in the active lamps

        Parameters
        ----------
        lines : `pfs.drp.stella.ReferenceLineSet`
            List of reference lines.
        elements : ``list` of `str`
            The list of elements or species in the active lamps.

        Returns
        -------
        filtered : `pfs.drp.stella.ReferenceLineSet`
            Filtered list of reference lines, or input lines if ``elements`` is Falsey.
        """
        if not lampElements:
            return lines
        keep = []
        for component in lampElements:
            if self._isSpeciesPattern.match(component):
                # Component is a species. Perform a search for lines matching only this
                keep += [ll for ll in lines if component == ll.description]
            else:
                # Component is a general element. Match for all available species
                elementPattern = re.compile(f'^{component}[IVX]*$')
                keep += [ll for ll in lines if elementPattern.match(ll.description)]
        speciesKept = {ll.description for ll in keep}
        self.log.info(f"Filtered line lists for elements/species: {sorted(lampElements)}, "
                      f"keeping species {speciesKept}.")
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
        select = lines.intensity >= self.config.minIntensity
        return lines[select]

    def getLampInfo(self, metadata):
        """Determine which lamps are active and return the lamp names

        Parameters
        ----------
        metadata : `lsst.daf.base.PropertyList`, optional
            FITS header, containing the lamp indicators.

        Returns
        -------
        lampInfo : `lsst.pipe.base.Struct`
           Resultant struct with components:
            lamps : `list` of `str`, or `None`
                The list of active lamps from the metadata.
                If no active lamps can be retrieved from the metadata, and ``assumeSkyIfNoLamps`` is set,
                then ``[skyLines]`` is returned, otherwise, ``None``.
            lampElementList:
                The list of lamp elements across all active lamps.

        Raises
        ------
        RuntimeError
            If ``metadata`` is ``None``.
        """
        if metadata is None:
            raise RuntimeError("Cannot determine lamp information because metadata was not provided")
        lamps = getLamps(metadata)
        lampElementList = getLampElements(metadata)
        if not lamps:
            if self.config.assumeSkyIfNoLamps:
                self.log.info("No lamps on; assuming sky.")
                lamps = ["skyLines"]
                lampElementList = ["OI", "NaI", "OH"]
            else:
                self.log.warning('No lamp information can be found in the metadata')
        return Struct(lamps=lamps, lampElementList=lampElementList)

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
        select = (lines.wavelength >= minWl) & (lines.wavelength <= maxWl)
        return lines[select]
