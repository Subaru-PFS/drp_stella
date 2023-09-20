import pfs.datamodel.drp
from .pfsFiberArraySet import PfsFiberArraySet
from .pfsFiberArray import PfsFiberArray, PfsSimpleSpectrum

__all__ = ("PfsArm", "PfsMerged", "PfsReference", "PfsSingle", "PfsObject")


class PfsArm(pfs.datamodel.drp.PfsArm, PfsFiberArraySet):
    _ylabel = "electrons/spectral pixel"
    pass


class PfsMerged(pfs.datamodel.drp.PfsMerged, PfsFiberArraySet):
    _ylabel = "electrons/nm"
    pass


class PfsReference(pfs.datamodel.drp.PfsReference, PfsSimpleSpectrum):
    _ylabel = "nJy"
    pass


class PfsSingle(pfs.datamodel.drp.PfsSingle, PfsFiberArray):
    _ylabel = "nJy"
    pass


class PfsObject(pfs.datamodel.drp.PfsObject, PfsFiberArray):
    _ylabel = "nJy"
    pass
