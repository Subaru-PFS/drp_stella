import os

import numpy as np

from lsst.utils import continueClass, getPackageDir
from lsst.geom import Extent2I

from .NevenPsf import NevenPsf


@continueClass  # noqa: F811 (redefinition)
class NevenPsf:
    @classmethod
    def build(cls, detMap, version=None, oversampleFactor=None, targetSize=None, xMaxDistance=20,
              directory=None):
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
        xMaxDistance : `float`, optional
            Maximum distance in x for selecting images for interpolation.
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
            version = "Jan2921_v1"
        if oversampleFactor is None:
            oversampleFactor = 9
        if targetSize is None:
            targetSize = 23

        # positions_of_simulation_00_from_<version>.npy contains: fiberId,x, y, wavelength
        xy = np.load(os.path.join(directory, f"positions_of_simulation_00_from_{version}.npy"))
        images = np.load(os.path.join(directory, f"array_of_simulation_00_from_{version}.npy"))

        return cls(detMap, xy[:, 1].astype(np.float32), xy[:, 2].astype(np.float32), images,
                   oversampleFactor, Extent2I(targetSize, targetSize), xMaxDistance)

    def __reduce__(self):
        """Pickling"""
        return self.__class__, (self.detectorMap, self.x, self.y, self.images, self.oversampleFactor,
                                self.targetSize, self.xMaxDistance)
