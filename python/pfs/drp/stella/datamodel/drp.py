import pfs.datamodel.drp
from .pfsFiberArraySet import PfsFiberArraySet
from .pfsFiberArray import PfsFiberArray, PfsSimpleSpectrum

__all__ = ("PfsArm", "PfsMerged", "PfsReference", "PfsSingle", "PfsObject")


class PfsArm(pfs.datamodel.drp.PfsArm, PfsFiberArraySet):
    pass


class PfsMerged(pfs.datamodel.drp.PfsMerged, PfsFiberArraySet):
    pass


class PfsReference(pfs.datamodel.drp.PfsReference, PfsSimpleSpectrum):
    pass


class PfsSingle(pfs.datamodel.drp.PfsSingle, PfsFiberArray):
    pass


class PfsObject(pfs.datamodel.drp.PfsObject, PfsFiberArray):
    pass
