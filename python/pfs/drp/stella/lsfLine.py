from enum import IntFlag

from pfs.datamodel.drp import LsfLineMeasurements

from .table import Table

__all__ = ("LsfLineFlag", "LsfLineSet")


class LsfLineFlag(IntFlag):
    """How an arc line was used in fitting the LSF model (`LsfLineSet.flag`)"""

    SATURATED = 0x1
    """The line's core is masked in this fibre (saturated): not fit"""
    SATURATED_ELSEWHERE = 0x2
    """The line is saturated in too many fibres: not used in any"""
    EXCLUDED_STATUS = 0x4
    """The line's catalogue status is excluded (e.g. BLEND): not used"""
    BROAD = 0x8
    """The line is broader than the LSF (measured, or flagged BROAD in the catalogue): not used"""
    PINNED = 0x10
    """The line's centre was held fixed (at the free fit's + the zero point)"""
    REJECTED_CENTROID = 0x20
    """Rejected (sigma-clipped) by the smooth fit of the centroid"""
    REJECTED_RADIUS = 0x40
    """Rejected (sigma-clipped) by the smooth fit of the disk radius"""
    REJECTED_RMS = 0x80
    """Rejected (sigma-clipped) by the smooth fit of the rms"""
    REJECTED_S = 0x100
    """Rejected (sigma-clipped) by the smooth fit of the skewness"""
    REJECTED_K = 0x200
    """Rejected (sigma-clipped) by the smooth fit of the excess kurtosis"""
    REJECTED_LAM5 = 0x400
    """Rejected (sigma-clipped) by the smooth fit of lam5"""
    REJECTED_LAM6 = 0x800
    """Rejected (sigma-clipped) by the smooth fit of lam6"""
    REJECTED_LORENTZ = 0x1000
    """Rejected (sigma-clipped) by the smooth fit of the Lorentzian's FWHM"""
    LOW_SN = 0x2000
    """The line's median S/N over the fibres is too low: not used"""

    @classmethod
    def rejected(cls, name):
        """The REJECTED_ flag for parameter ``name``"""
        return cls[f"REJECTED_{name.upper()}"]

    @classmethod
    def notUsed(cls):
        """Flags of lines that didn't enter the smooth fits"""
        return cls.SATURATED | cls.SATURATED_ELSEWHERE | cls.EXCLUDED_STATUS | cls.BROAD | cls.LOW_SN


class LsfLineSet(Table):
    """Per-line fits of the line-spread function to arc lines

    One row per arc line per fibre per exposure; see
    `pfs.datamodel.drp.LsfLineMeasurements` for the columns and `LsfLineFlag`
    for the ``flag`` bits. ``wavelength`` is the reference (catalogue)
    wavelength, and ``refPixel`` its position in the spectrum according to the
    exposure's wavelength solution; ``pixel - refPixel`` is the measured offset
    of the line from the detectorMap.
    """
    DamdClass = LsfLineMeasurements
