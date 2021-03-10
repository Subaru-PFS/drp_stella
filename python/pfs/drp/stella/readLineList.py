from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task

from lsst.obs.pfs.utils import getLampElements

from .utils import readLineListFile

__all__ = ("ReadLineListConfig", "ReadLineListTask")


class ReadLineListConfig(Config):
    """Configuration for ReadLineListTask"""
    lineList = Field(dtype=str, doc="Filename of linelist", default="ArCdHgKrNeXe.txt")
    lineListFiles = ListField(dtype=str, doc="list of names of linelist files", default=["ArCdHgKrNeXe.txt"])
    restrictByLamps = Field(dtype=bool, default=True,
                            doc="Restrict linelist by the list of active lamps? True is appropriate for arcs")
    minIntensity = Field(dtype=float, default=0.0, doc="Minimum linelist intensity")

    def validate(self):
        super().validate()
        if len(self.lineListFiles) == 0: # should check if both are set.  Hard
            self.lineListFiles = [self.lineList]

class ReadLineListTask(Task):
    """Read a linelist"""
    ConfigClass = ReadLineListConfig

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
        lines : `list` or `dict` of `list` of `pfs.drp.stella.ReferenceLine`
            Lines from the linelist. If ``detectorMap`` was provided, this is a
            `dict` mapping ``fiberId`` to the `list` of reference lines;
            otherwise this is a `list` of reference lines.
        """
        lamps = self.getLamps(metadata)
        lines = []
        for lineListFile in self.config.lineListFiles:
            lines += readLineListFile(lineListFile, lamps, minIntensity=self.config.minIntensity)
        if detectorMap is None:
            return lines
        return self.getFiberLines(lines, detectorMap, fiberId=fiberId)

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
            is ``True``; otherwise, ``None``.

        Raises
        ------
        RuntimeError
            If ``metadata`` is ``None`` and ``restrictByLamps`` is ``True``.
        """
        if not self.config.restrictByLamps:
            return None
        if metadata is None:
            raise RuntimeError("Cannot determine lamps because metadata was not provided")
        return getLampElements(metadata)

    def getFiberLines(self, lines, detectorMap, fiberId=None):
        """Reformat line list into a list of lines for each fiber

        Parameters
        ----------
        lines : `list` of `pfs.drp.stella.ReferenceLine`
            Lines from the linelist.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from x,y to fiberId,wavelength.
        fiberId : `numpy.ndarray` of `int`, optional
            Fibers for which retrieve the linelist. If ``None``, all fibers in
            the ``detectorMap`` will be used.

        Returns
        -------
        refLines : `dict` of `list` of `pfs.drp.stella.ReferenceLine`
            List of lines for each fiber, indexed by ``fiberId``.
        """
        if fiberId is None:
            fiberId = detectorMap.fiberId
        refLines = {}
        for ff in fiberId:
            wavelength = detectorMap.getWavelength(ff)
            minWl = wavelength.min()
            maxWl = wavelength.max()
            refLines[ff] = [rl for rl in lines if rl.wavelength >= minWl and rl.wavelength <= maxWl]
        return refLines
