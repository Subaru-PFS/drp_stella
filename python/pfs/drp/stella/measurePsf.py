from lsst.pex.config import Config
from lsst.pipe.base import Task


class MeasurePsfConfig(Config):
    """Configuration for MeasurePsfTask"""
    pass


class MeasurePsfTask(Task):
    """Measure the PSF from an exposure

    This is currently a non-functional placeholder.
    """
    ConfigClass = MeasurePsfConfig

    def run(self, sensorRefList, exposureList):
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

        Returns
        -------
        psfList : `list` of PSFs (type TBD)
            List of point-spread functions.
        """
        return [self.runSingle(sensorRef, exposure) for
                sensorRef, exposure in zip(sensorRefList, exposureList)]

    def runSingle(self, sensorRef, exposure):
        """Measure the PSF for a single spectrograph

        This is currently a non-functional placeholder.

        Parameters
        ----------
        sensorRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for individual sensor.
        exposure : `lsst.afw.image.Exposure`
            Image of sensor.

        Returns
        -------
        psf : PSF (type TBD)
            Point-spread function.
        """
        raise NotImplementedError("Not coded yet!")
