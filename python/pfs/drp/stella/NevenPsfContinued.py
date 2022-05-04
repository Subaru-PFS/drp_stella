import os

import numpy as np

from lsst.utils import continueClass, getPackageDir
from lsst.geom import Extent2I

from .NevenPsf import NevenPsf


@continueClass  # noqa: F811 (redefinition)
class NevenPsf:  # type: ignore [no-redef] # noqa: F811 (redefinition)
    @classmethod
    def build(cls, detMap, version=None, oversampleFactor=None, targetSize=None, directory=None):
        """Generate a `NevenPsf` using the standard data

        Parameters
        ----------
        detMap : `pfs.drp.stella.DetectorMap`
            Mapping between fiberId,wavelength and x,y.
        version : `str`, optional
            Version string, used for identifying the file containing the data.
        oversampleFactor : `int`, optional
            Factor by which the data has been oversampled.
        targetSize : `int`, optional
            Desired size of the realised PSF images.
        directory : `str`, optional
            Directory containing the realisations from Neven. If not provided,
            defaults to ``/path/to/drp_pfs_data/nevenPsf``.

        Returns
        -------
        psf : `pfs.drp.stella.NevenPsf`
            Point-spread function model.
        """
        if directory is None:
            directory = os.path.join(getPackageDir("drp_pfs_data"), "nevenPsf")
        if version is None:
            version = "Jan0821_v3"
        if oversampleFactor is None:
            oversampleFactor = 9
        if targetSize is None:
            targetSize = 23

        # positions_of_simulation_00_from_<version>.npy contains: fiberId,x, y, wavelength
        positions = np.load(os.path.join(directory, f"positions_of_simulation_00_from_{version}.npy"),
                            allow_pickle=True)
        images = np.load(os.path.join(directory, f"array_of_simulation_00_from_{version}.npy"),
                         allow_pickle=True)

        return cls(detMap, positions[:, 0].astype(np.int32), positions[:, 3].astype(float), images,
                   oversampleFactor, Extent2I(targetSize, targetSize))

    def __reduce__(self):
        """Pickling"""
        return self.__class__, (self.detectorMap, self.getFiberId(), self.getWavelength(), self.getImages(),
                                self.oversampleFactor, self.targetSize)
