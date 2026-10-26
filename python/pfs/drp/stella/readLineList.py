from lsst.pex.config import Config, Field, ListField, DictField, FieldValidationError
from lsst.pipe.base import Struct, Task

from lsst.obs.pfs.utils import getLamps, getLampElements
from .referenceLine import ReferenceLineSet, ReferenceLineStatus
import re
from collections import Counter
import numpy as np

__all__ = ("ReadLineListConfig", "ReadLineListTask")

# Arms lacking their own per-arm linelist fall back to another arm's file.
_ARM_ALIASES = {"m": "r"}


def isEngineering(lightSource):
    """Is ``lightSource`` an engineering-fibre (IIS) lamp, i.e. named ``<lamp>_eng``?"""
    return lightSource.endswith("_eng")


def lightSourceKey(lightSource, lightSourceMap):
    """Return the ``lightSourceMap`` key to use for ``lightSource``, or `None`

    An engineering-fibre (IIS) lamp ``<lamp>_eng`` uses its own entry if there
    is one, and otherwise the entry for ``<lamp>``.
    """
    if lightSource in lightSourceMap:
        return lightSource
    if isEngineering(lightSource) and lightSource[:-len("_eng")] in lightSourceMap:
        return lightSource[:-len("_eng")]
    return None


class ReadLineListConfig(Config):
    """Configuration for ReadLineListTask"""
    lightSources = ListField(dtype=str,
                             doc=("list of unique light sources that provide "
                                  "the emission lines (overrides header). "
                                  "These must be keys in the lightSourceMap. "),
                             default=[])
    lampElements = ListField(dtype=str,
                             doc=("list of unique lamp elements or species to filter by. "),
                             default=[])
    lightSourceMap = DictField(keytype=str, itemtype=str,
                               doc=("Mapping of lamp or other light source to linelist file."
                                    "A value of ``None`` means no line list file "
                                    "is available for that source. "
                                    "If the value is not an absolute path, "
                                    "it is assumed that the path is relative to ``obs_pfs/pfs/lineLists/``. "
                                    "Engineering-fibre (IIS) lamps are named ``<lamp>_eng``; if there is no "
                                    "entry for one, the entry for ``<lamp>`` is used."
                                    " If the value contains the literal substring ``{arm}``, it is "
                                    "substituted with the ``arm`` supplied to ``ReadLineListTask.run`` "
                                    "(e.g. ``'skyLines-ragnar-{arm}.txt'``); this requires that ``arm`` "
                                    "be provided to ``run``."),
                               default={'Ar': 'Ar.txt',
                                        'Ne': 'Ne.txt',
                                        'Kr': 'Kr.txt',
                                        'OI': 'contamination.txt',
                                        'Xe': 'Xe.txt',
                                        'HgAr': 'HgAr.txt',
                                        'HgCd': 'HgCd.txt',
                                        'Quartz': None,
                                        'sky': 'skyLines-ragnar.txt',
                                        # IIS lamps (engineering fibres)
                                        'Ar_eng': 'Ar_IIS.txt',
                                        'H_eng': 'H_IIS.txt',
                                        'He_eng': 'He_IIS.txt',
                                        'HgAr_eng': 'HgAr_IIS.txt',
                                        'Kr_eng': 'Kr_IIS.txt',
                                        'Ne_eng': 'Ne_IIS.txt',
                                        'Quartz_eng': None,
                                        })
    assumeSky = Field(dtype=bool, default=False,
                      doc="Assume that we're looking at sky even if lamps claim to be active?")
    assumeSkyIfNoLamps = Field(dtype=bool, default=True,
                               doc="Assume that we're looking at sky if no lamps are active?")
    minIntensity = Field(dtype=float, default=0.0, doc="Minimum linelist intensity; <= 0 means no filtering")
    exclusionRadius = Field(dtype=float, default=0.0, doc="Exclusion radius around lines (nm)")

    def validate(self):
        """Validate input config parameters"""

        def getDuplicateItems(iterable):
            """Return a list of duplicates"""
            return [k for k, v in Counter(iterable).items() if v > 1]

        super().validate()

        duplicateLightSources = getDuplicateItems(self.lightSources)
        if duplicateLightSources:
            raise RuntimeError(f'There are duplicate light sources {duplicateLightSources}')

        invalidSources = {ll for ll in self.lightSources if lightSourceKey(ll, self.lightSourceMap) is None}
        if invalidSources:
            raise RuntimeError('The following light sources '
                               f'are not in the lightSourceMap: {invalidSources}')
        duplicateLampElements = getDuplicateItems(self.lampElements)
        if duplicateLampElements:
            raise FieldValidationError(f'There are duplicate lamp elements {duplicateLightSources}')


class ReadLineListTask(Task):
    """Read a linelist"""
    ConfigClass = ReadLineListConfig
    _DefaultName = "readLineList"
    _isSpeciesPattern = re.compile(r'^[A-Z][A-Za-z]*[IVX]+')

    def run(self, detectorMap=None, metadata=None, exclusionRadius=None, arm: str | None = None):
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
        exclusionRadius : `float`, optional
            Exclusion radius (nm) to apply instead of ``config.exclusionRadius``
            for this call only. Useful for callers that need a one-off override
            (e.g. disabling exclusion for a bootstrap pass) without modifying
            ``self.config``, which may be frozen (e.g. when run as a subtask
            under ``pipetask``).
        arm : `str`, optional
            Spectrograph arm (``b``, ``r``, ``n``, ``m``). Only required if a
            selected ``lightSourceMap`` entry contains an ``{arm}`` placeholder.

        Returns
        -------
        lines : `pfs.drp.stella.ReferenceLineSet`
            Lines from the linelist.
        """
        lightSources = {}
        lampElements = {}
        lampInfoFromMetaData = None

        if len(self.config.lightSources) > 0:
            lightSources = set(self.config.lightSources)
        else:
            lampInfoFromMetaData = self.getLampInfo(metadata)
            lightSources = lampInfoFromMetaData.lamps

        if len(self.config.lampElements) > 0:
            lampElements = set(self.config.lampElements)
        elif lampInfoFromMetaData:
            lampElements = lampInfoFromMetaData.lampElements

        lines = ReferenceLineSet.empty()
        engineeringIlluminated = set()
        scienceIlluminated = set()
        # Read the PFI lamps' lists before the IIS (engineering-fibre) ones, so that when the same line
        # appears in both, filterDuplicates keeps the PFI copy.
        for ll in sorted(lightSources, key=lambda name: (isEngineering(name), name)):
            key = lightSourceKey(ll, self.config.lightSourceMap)
            if key is None:
                raise KeyError(
                    f"Lamp {ll} is unknown; valid options [{', '.join(self.config.lightSourceMap)}]")

            if isEngineering(ll):
                engineeringIlluminated.add(ll[:-len("_eng")])
            else:
                scienceIlluminated.add(ll)

            filename = self.config.lightSourceMap[key]
            if filename is not None:
                if "{arm}" in filename:
                    if arm is None:
                        raise RuntimeError(
                            f"lightSourceMap entry for {ll!r} ({filename!r}) requires an arm, "
                            "but none was provided to ReadLineListTask.run"
                        )
                    filename = filename.format(arm=_ARM_ALIASES.get(arm, arm))
                lines.extend(ReferenceLineSet.fromLineList(filename))

        if engineeringIlluminated and scienceIlluminated:
            diff = engineeringIlluminated.symmetric_difference(scienceIlluminated)
            if diff:
                self.log.warn("Both Engineering and Science fibres are illuminated but with different lamps: "
                              "%s v. %s", engineeringIlluminated, scienceIlluminated)

        if lampElements:
            lines = self.filterByLampElements(lines, lampElements)
        lines = self.filterByIntensity(lines)
        lines = self.filterByWavelength(lines, detectorMap)
        self.filterDuplicates(lines)
        radius = self.config.exclusionRadius if exclusionRadius is None else exclusionRadius
        lines.applyExclusionZone(radius)
        return lines

    def filterDuplicates(self, lines: ReferenceLineSet) -> None:
        """Reject duplicate lines by wavelength from the line list.

        This is performed in-place.

        This may occur if multiple input sources may share
        lines with the same wavelength.

        Parameters
        ----------
        lines : `pfs.drp.stella.ReferenceLineSet`
            List of reference lines.
        """
        _, indices = np.unique(lines.wavelength, return_index=True)
        reject = np.full(len(lines), True, dtype=bool)
        reject[indices] = False
        lines.status[reject] |= ReferenceLineStatus.BLEND

    def filterByLampElements(self, lines, lampElements):
        """Filter the line list by the elements or species in the active lamps

        Parameters
        ----------
        lines : `pfs.drp.stella.ReferenceLineSet`
            List of reference lines.
        lampElements : `set` of `str`
            The set of elements or species in the active lamps.

        Returns
        -------
        filtered : `pfs.drp.stella.ReferenceLineSet`
            Filtered list of reference lines, or input lines if ``lampElements``
            is either ``None`` or empty.
        """
        if lampElements is None or len(lampElements) == 0:
            return lines
        keep = []
        for component in sorted(lampElements):
            mat = re.search(r"^(.*)_eng$", component)
            if mat:
                component = mat.group(1)

            if self._isSpeciesPattern.match(component):
                # Component is a species. Perform a search for lines matching only this
                keep += [ll for ll in lines if component == ll.description]
            else:
                # Component is a general element. Match for all available species
                elementPattern = re.compile(f'^{component}[IVX]*$')
                keep += [ll for ll in lines if elementPattern.match(ll.description)]
        speciesKept = {ll.description for ll in keep}
        self.log.info("Filtered line lists %s by %s, keeping species %s.",
                      sorted({ll.description for ll in lines}), lampElements, speciesKept)
        return ReferenceLineSet.fromRows(keep)

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
        """Determine from the metadata which lamps are active and return the lamp names

        Parameters
        ----------
        metadata : `lsst.daf.base.PropertyList`, optional
            FITS header, containing the lamp indicators.

        Returns
        -------

        lampInfo : `lsst.pipe.base.Struct`
           Lamp information as a struct with components:

           ``lamps``
                List of active lamps from the metadata (`set` of `str`]).

                If no active lamps can be retrieved from the metadata, and ``assumeSkyIfNoLamps`` is set,
                then ``[sky]`` is returned, otherwise, ``None``.

            ``lampElements``
                Set of active lamp elements across all active lamps (`set` of `str`).

        Raises
        ------
        RuntimeError
            If ``metadata`` is ``None``.
        """
        if metadata is None:
            raise RuntimeError("Cannot determine lamp information because metadata was not provided")

        lamps = getLamps(metadata)
        lampElements = getLampElements(metadata)
        if self.config.assumeSky or not lamps:
            if self.config.assumeSky or self.config.assumeSkyIfNoLamps:
                if self.config.assumeSky:
                    self.log.info("Ignoring lamp status and assuming sky.")
                else:
                    self.log.info("No lamps on; assuming sky.")
                lamps = {"sky"}
                lampElements = None
            else:
                self.log.warning('No lamp information can be found in the metadata')
        return Struct(lamps=lamps, lampElements=lampElements)

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
