from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task

from lsst.obs.pfs.utils import getLampElements
from .referenceLine import ReferenceLineSet

__all__ = ("ReadLineListConfig", "ReadLineListTask")


class ReadLineListConfig(Config):
    """Configuration for ReadLineListTask"""
    lineList = Field(dtype=str, doc="Filename of linelist", default="ArCdHgKrNeXe.txt")
    lineListFiles = ListField(dtype=str, doc="list of names of linelist files", default=["ArCdHgKrNeXe.txt"])
    restrictByLamps = Field(dtype=bool, default=True,
                            doc="Restrict linelist by the list of active lamps? True is appropriate for arcs")
    lampList = ListField(dtype=str, doc="list of species in lamps", default=[])
    minIntensity = Field(dtype=float, default=0.0, doc="Minimum linelist intensity")

    def validate(self):
        super().validate()
        if len(self.lineListFiles) == 0:  # should check if both are set.  Hard with defaults in one place
            self.lineListFiles = [self.lineList]

        if len(self.lampList) > 0 and self.restrictByLamps:
            raise ValueError("You may not specify both lampList and restrictByLamps")


class ReadLineListTask(Task):
    """Read a linelist"""
    ConfigClass = ReadLineListConfig
    _DefaultName = "ReadLineListTask"

    def run(self, detectorMap=None, fiberId=None, metadata=None):
        """Read a linelist

        This serves as a wrapper around the common operation of looking up the
        lamps, reading the linelist and formatting it.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Mapping from x,y to fiberId,wavelength. This is required in order to
            reformat the linelist into a `dict` of lines for each fiber.
        fiberId : `numpy.ndarray` of `int`, optional
            Fibers for which retrieve the linelist. If ``None``, all fibers in
            the ``detectorMap`` will be used.
        metadata : `lsst.daf.base.PropertyList`, optional
            FITS header, containing the lamp indicators. This is required in
            order to restrict the linelist by the list of active lamps.

        Returns
        -------
        lines : `pfs.drp.stella.ReferenceLineSet` or `dict` of `pfs.drp.stella.ReferenceLineSet`
            Lines from the linelist. If ``detectorMap`` was provided, this is a
            `dict` mapping ``fiberId`` to the `list` of reference lines;
            otherwise this is a list of reference lines.
        """
        lamps = self.getLamps(metadata)
        lines = ReferenceLineSet.empty()
        for filename in self.config.lineListFiles:
            lines.extend(ReferenceLineSet.fromLineList(filename))
        lines = self.filterByLamps(lines, lamps)
        lines = self.filterByIntensity(lines)
        if detectorMap is None:
            return lines
        return self.getFiberLines(lines, detectorMap, fiberId=fiberId)

    def filterByLamps(self, lines, lamps):
        """Filter the line list by which lamps are active

        Parameters
        ----------
        lines : `pfs.drp.stella.ReferenceLineSet`
            List of reference lines.
        lamps : ``list` of `str`
            The list of lamps.

        Returns
        -------
        filtered : `pfs.drp.stella.ReferenceLineSet`
            Filtered list of reference lines.
        """
        if not self.config.restrictByLamps:
            return lines
        keep = []
        for desc in lamps:
            keep += [ll for ll in lines if ll.description.startswith(desc)]
        return ReferenceLineSet(keep)

    def filterByIntensity(self, lines):
        """Filter the line list by intensity level

        Parameters
        ----------
        lines : `pfs.drp.stella.ReferenceLineSet`
            List of reference lines.

        Returns
        -------
        filtered : `pfs.drp.stella.ReferenceLineSet`
            Filtered list of reference lines.
        """
        keep = [ll for ll in lines if ll.intensity >= self.config.minIntensity]
        return ReferenceLineSet(keep)

    def getLamps(self, metadata):
        """Determine which lamps are active

        Parameters
        ----------
        metadata : `lsst.daf.base.PropertyList`, optional
            FITS header, containing the lamp indicators.

        Returns
        -------
        lamps : `list` of `str`, or `None`
            The list of lamps, if the ``restrictByLamps`` configuration option
            is ``True`` or ``lampList`` is set; otherwise, ``None``.

        Raises
        ------
        RuntimeError
            If ``metadata`` is ``None`` and ``restrictByLamps`` is ``True``, and
            lampList is not set.
        """
        if len(self.config.lampList) > 0:
            return self.config.lampList

        if not self.config.restrictByLamps:
            return None
        if metadata is None:
            raise RuntimeError("Cannot determine lamps because metadata was not provided")
        return getLampElements(metadata)

    def getFiberLines(self, lines, detectorMap, fiberId=None):
        """Reformat line list into a list of lines for each fiber

        Parameters
        ----------
        lines : `pfs.drp.stella.ReferenceLineSet`
            Lines from the linelist.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from x,y to fiberId,wavelength.
        fiberId : `numpy.ndarray` of `int`, optional
            Fibers for which retrieve the linelist. If ``None``, all fibers in
            the ``detectorMap`` will be used.

        Returns
        -------
        refLines : `dict` of `pfs.drp.stella.ReferenceLineSet`
            List of lines for each fiber, indexed by ``fiberId``.
        """
        if fiberId is None:
            fiberId = detectorMap.fiberId
        refLines = {}
        for ff in fiberId:
            wavelength = detectorMap.getWavelength(ff)
            minWl = wavelength.min()
            maxWl = wavelength.max()
            refLines[ff] = ReferenceLineSet([rl for rl in lines if rl.wavelength >= minWl and
                                             rl.wavelength <= maxWl])
        return refLines
