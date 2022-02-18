
from lsst.utils import continueClass
from lsst.afw.fits import MemFileManager
from lsst.afw.detection import Psf

from .SpectralPsf import ImagingSpectralPsf

__all__ = ["ImagingSpectralPsf"]


@continueClass  # noqa: F811 (redefinition)
class ImagingSpectralPsf:  # noqa: F811 (redefinition)
    @classmethod
    def _fromPickle(cls, psfData, detectorMap):
        """Construct from pickle

        Parameters
        ----------
        psfData : `bytes`
            Data stream describing the PSF.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength <--> x,y.

        Returns
        -------
        self : cls
            Constructed object.
        """
        size = len(psfData)
        manager = MemFileManager(size)
        manager.setData(psfData, size)
        psf = Psf.readFits(manager)
        return cls(psf, detectorMap)

    def __reduce__(self):
        """Pickling support"""
        manager = MemFileManager()
        self.imagePsf.writeFits(manager)
        return ImagingSpectralPsf._fromPickle, (manager.getData(), self.detectorMap)
