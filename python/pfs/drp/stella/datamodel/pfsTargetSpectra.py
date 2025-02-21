import pfs.datamodel.pfsTargetSpectra

__all__ = ["PfsTargetSpectra", "PfsCalibratedSpectra", "PfsObjectSpectra"]


"""The contents of this file have been moved to the 'datamodel' package with
some slightly different names. This file exists to maintain backwards
compatibility because the names have been baked into the registry database."""


class PfsTargetSpectra(pfs.datamodel.pfsTargetSpectra.PfsTargetSpectra):
    pass


class PfsCalibratedSpectra(pfs.datamodel.drp.PfsCalibrated):
    @classmethod
    def fromPfsCalibrated(cls, pfsCalibrated):
        """Construct from a PfsCalibrated

        This supports the (long-term) migration from
        pfs.drp.stella.pfsTargetSpectra.PfsCalibratedSpectra
        to pfs.datamodel.drp.PfsCalibrated
        """
        return cls([pfsCalibrated[target] for target in pfsCalibrated])


class PfsObjectSpectra(pfs.datamodel.drp.PfsCoadd):
    @classmethod
    def fromPfsCoadd(cls, pfsCoadd):
        """Construct from a PfsCoadd

        This supports the (long-term) migration from
        pfs.drp.stella.pfsTargetSpectra.PfsObjectSpectra
        to pfs.datamodel.drp.PfsCoadd
        """
        return cls([pfsCoadd[target] for target in pfsCoadd])
