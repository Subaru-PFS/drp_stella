import pfs.datamodel.drp
from .pfsSpectra import PfsSpectra
from .pfsSpectrum import PfsSpectrum, PfsSimpleSpectrum

__all__ = ("PfsArm", "PfsMerged", "PfsReference", "PfsSingle", "PfsObject")


class PfsArm(pfs.datamodel.drp.PfsArm, PfsSpectra):
    pass


class PfsMerged(pfs.datamodel.drp.PfsMerged, PfsSpectra):
    pass


class PfsReference(pfs.datamodel.drp.PfsReference, PfsSimpleSpectrum):
    pass


class PfsSingle(pfs.datamodel.drp.PfsSingle, PfsSpectrum):
    pass


class PfsObject(pfs.datamodel.drp.PfsObject, PfsSpectrum):
    pass
