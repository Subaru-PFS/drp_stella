from lsst.pex.config import Config, Field
from lsst.pipe.base import Task

from .NevenPsfContinued import NevenPsf


class MeasurePsfConfig(Config):
    """Configuration for MeasurePsfTask

    No defaults are set for the ``NevenPsf`` parameters, so that the defaults
    can be specified by ``NevenPsf``.
    """
    directory = Field(dtype=str, optional=True, doc="Directory containing NevenPsf realisations")
    version = Field(dtype=str, optional=True, doc="Version of NevenPsf realisation to use")
    oversampleFactor = Field(dtype=int, optional=True,
                             doc="Factor by which the NevenPsf realisations have been oversampled")
    targetSize = Field(dtype=int, optional=True, doc="Desired size of the realised PSF images")


class MeasurePsfTask(Task):
    """Measure the PSF from an exposure

    This is currently a non-functional placeholder.
    """
    ConfigClass = MeasurePsfConfig
    _DefaultName = "measurePsf"

    def run(self, sensorRefList, exposureList, detectorMapList):
        """Measure the PSF for an exposure over the entire spectrograph

        We operate on the entire spectrograph in case there are parameters
        that are shared between spectrographs. However, this placeholder
        implementation iterates over the spectrographs, fitting the PSF of
        each individually.

        Parameters
        ----------
        sensorRefList : iterable of `lsst.daf.persistence.ButlerDataRef`
            List of data references for each sensor in an exposure.
        exposureList : iterable of `lsst.afw.image.Exposure`
            List of images of each sensor in an exposure.
        detectorMapList : iterable of `pfs.drp.stella.DetectorMap`
            List of detector maps for each sensor in an exposure.

        Returns
        -------
        psfList : `list` of `pfs.drp.stella.SpectralPsf`
            List of point-spread functions.
        """
        return [self.runSingle(sensorRef, exposure, detectorMap) for
                sensorRef, exposure, detectorMap in zip(sensorRefList, exposureList, detectorMapList)]

    def runSingle(self, sensorRef, exposure, detectorMap):
        """Measure the PSF for a single spectrograph

        This is currently a non-functional placeholder.

        Parameters
        ----------
        sensorRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for individual sensor.
        exposure : `lsst.afw.image.Exposure`
            Image of sensor.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength <--> x,y.

        Returns
        -------
        psf : `pfs.drp.stella.SpectralPsf`
            Point-spread function.
        """
        psf = NevenPsf.build(detectorMap, version=self.config.version,
                             oversampleFactor=self.config.oversampleFactor,
                             targetSize=self.config.targetSize,
                             directory=self.config.directory)
        exposure.setPsf(psf)
        return psf
