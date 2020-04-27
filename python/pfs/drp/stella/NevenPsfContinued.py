import os

import numpy as np

from lsst.utils import continueClass, getPackageDir
from lsst.geom import Extent2I

from .NevenPsf import NevenPsf


@continueClass  # noqa: F811 (redefinition)
class NevenPsf:
    @classmethod
    def build(cls, detMap, version="Apr1520_v3", oversampleFactor=9, targetSize=23, xMaxDistance=20):
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
        xMaxDistance : `float`
            Maximum distance in x for selecting images for interpolation.

        Returns
        -------
        psf : `pfs.drp.stella.NevenPsf`
            Point-spread function model.
        """
        obsPfsData = getPackageDir("drp_pfs_data")

        # positions_of_simulation_00_from_<version>.npy contains: fiberId,x, y, wavelength
        xy = np.load(os.path.join(obsPfsData, "nevenPsf",
                                  "positions_of_simulation_00_from_" + version + ".npy"))
        images = np.load(os.path.join(obsPfsData, "nevenPsf",
                                      "array_of_simulation_00_from_" + version + ".npy"))

        return cls(detMap, xy[:, 1].astype(np.float32), xy[:, 2].astype(np.float32), images,
                   oversampleFactor, Extent2I(targetSize, targetSize), xMaxDistance)

    def __reduce__(self):
        """Pickling"""
        return self.__class__, (self.detectorMap, self.x, self.y, self.images, self.oversampleFactor,
                                self.targetSize, self.xMaxDistance)
