import pfs.datamodel
import pfs.datamodel.pfsTargetSpectra

__all__ = ["PfsTargetSpectra", "PfsCalibratedSpectra", "PfsObjectSpectra"]


"""The contents of this file have been moved to the 'datamodel' package with
some slightly different names. This file exists to maintain backwards
compatibility because the names have been baked into the registry database."""


class TargetSpectraLsfMixin:
    """Provides the ``lsf`` of target spectra as a `pfs.drp.stella.SpectraLsf`

    The same as `pfs.drp.stella.datamodel.PfsFiberArraySet.lsf`, for spectra
    identified by their target (e.g. a pfsCoadd).
    """

    @property
    def lsf(self):
        """Line-spread function of the spectra (`pfs.drp.stella.SpectraLsf`), or `None`

        The LSF read from a file is converted when first used.
        """
        lsf = getattr(self, "_lsf", None)
        if isinstance(lsf, pfs.datamodel.PfsLsf):
            from ..spectraLsf import spectraLsfFromDatamodel
            targets = list(self)
            wavelength = {target: self[target].wavelength for target in targets}
            length = len(next(iter(wavelength.values()))) if wavelength else 0
            lsf = self._lsf = spectraLsfFromDatamodel(lsf, targets, wavelength, length)
        return lsf

    @lsf.setter
    def lsf(self, value):
        self._lsf = value


class PfsTargetSpectra(TargetSpectraLsfMixin, pfs.datamodel.pfsTargetSpectra.PfsTargetSpectra):
    pass


class PfsCalibratedSpectra(TargetSpectraLsfMixin, pfs.datamodel.drp.PfsCalibrated):
    @classmethod
    def fromPfsCalibrated(cls, pfsCalibrated):
        """Construct from a PfsCalibrated

        This supports the (long-term) migration from
        pfs.drp.stella.pfsTargetSpectra.PfsCalibratedSpectra
        to pfs.datamodel.drp.PfsCalibrated
        """
        return cls([pfsCalibrated[target] for target in pfsCalibrated], pfsCalibrated.metadata,
                   lsf=getattr(pfsCalibrated, "lsf", None))


class PfsObjectSpectra(TargetSpectraLsfMixin, pfs.datamodel.drp.PfsCoadd):
    @classmethod
    def fromPfsCoadd(cls, pfsCoadd):
        """Construct from a PfsCoadd

        This supports the (long-term) migration from
        pfs.drp.stella.pfsTargetSpectra.PfsObjectSpectra
        to pfs.datamodel.drp.PfsCoadd
        """
        return cls([pfsCoadd[target] for target in pfsCoadd], pfsCoadd.metadata,
                   lsf=getattr(pfsCoadd, "lsf", None))
