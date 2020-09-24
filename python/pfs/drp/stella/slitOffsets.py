import numpy as np

from lsst.pex.config import Config, ListField

__all__ = ("SlitOffsetsConfig",)


class SlitOffsetsConfig(Config):
    """Configuration of slit offsets for DetectorMap"""
    spatial = ListField(dtype=float, default=[], doc="Spatial slit offsets for each fiber; or empty")
    spectral = ListField(dtype=float, default=[], doc="Spectral slit offsets for each fiber; or empty")

    # For backwards compatibility
    @property
    def x(self):
        return self.spatial

    # For backwards compatibility
    @property
    def y(self):
        return self.spectral

    @property
    def numOffsets(self):
        """The number of slit offsets"""
        if self.spatial:
            return len(self.spatial)
        if self.spectral:
            return len(self.spectral)
        return 0

    def validate(self):
        super().validate()
        numOffsets = self.numOffsets
        if self.spatial and len(self.spatial) != numOffsets:
            raise ValueError("Inconsistent number of spatial slit offsets: %d vs %d" %
                             (len(self.spatial), numOffsets))
        if self.spectral and len(self.spectral) != numOffsets:
            raise ValueError("Inconsistent number of spectral slit offsets: %d vs %d" %
                             (len(self.spectral), numOffsets))

    def apply(self, detectorMap, log=None):
        """Apply slit offsets to detectorMap

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            DetectorMap to which to apply slit offsets.
        log : `lsst.log.Log`, or ``None``
            Optional logger for reporting the application of slit offsets.
        """
        numOffsets = self.numOffsets
        if numOffsets == 0:
            return  # Nothing to do
        if len(detectorMap) != self.numOffsets:
            raise ValueError("Number of offsets (%d) doesn't match number of fibers (%d)" %
                             (numOffsets, len(detectorMap)))
        if log is not None:
            which = []
            if self.spatial:
                which += ["spatial"]
            if self.spectral:
                which += ["spectral"]
            log.info("Applying %s slit offsets to detectorMap" %
                     ("+".join(which),))

        spatial = detectorMap.getSpatialOffsets()
        spectral = detectorMap.getSpectralOffsets()
        if self.spatial:
            spatial += self.spatial
        if self.spectral:
            spatial += self.spectral
        detectorMap.setSlitOffsets(spatial.astype(np.float32), spectral.astype(np.float32))
