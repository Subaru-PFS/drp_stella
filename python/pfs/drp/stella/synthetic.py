import math
import numpy as np

import lsst.afw.image
from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct
from pfs.drp.stella import DetectorMap

__all__ = ["makeSpectrumImage",
           "addNoiseToImage",
           "makeSyntheticFlat",
           "makeSyntheticArc",
           "makeSyntheticDetectorMap",
           ]


def makeSpectrumImage(spectrum, dims, traceCenters, traceOffsets, fwhm):
    """Make an image with multiple spectra

    This is a generic workhorse, so as to be able to make a variety of images.

    Parameters
    ----------
    spectrum : `ndarray.array`
        Array containing the spectrum to employ. There should be a number of
        values equal to the number of rows in the image. Each value is the
        integrated flux in the spectrum for that row.
    dims : `lsst.afw.geom.Extent2I`
        Dimensions of the image.
    traceCenters : `ndarray.array`
        Column centers of each of traces.
    traceOffsets : `ndarray.array`
        Offset from the column center for each row.
    fwhm : `float`
        Full width at half maximum of the trace.

    Returns
    -------
    image : `lsst.afw.image.Image`
        Image with spectra.
    """
    image = lsst.afw.image.ImageF(dims)
    sigma = fwhm/(2*math.sqrt(2*math.log(2)))
    width, height = dims
    norm = 1.0/(sigma*math.sqrt(2*math.pi))
    xx = np.arange(width, dtype=np.float32)

    if np.isscalar(spectrum):
        spectrum = spectrum*np.ones(height, dtype=np.float32)
    else:
        assert len(spectrum) == height
    assert len(traceOffsets) == height
    for row, (spec, offset) in enumerate(zip(spectrum, traceOffsets)):
        profile = np.zeros_like(xx)
        for center in traceCenters:
            profile += np.exp(-0.5*((xx - center + offset)/sigma)**2)
        image.array[row] = spec*norm*profile

    return image


def addNoiseToImage(image, gain, readnoise, rng=None):
    """Add noise to an image

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image to which to add noise. The units of the image is ADU.
    gain : `float`
        Gain, in electrons/ADU.
    readnoise : `float`
        Read noise, in electrons (or ADU if ``gain`` is zero).
    rng : `numpy.random.RandomState`
        Random number generator.
    """
    if rng is None:
        rng = np.random
    if gain != 0.0:
        image.array[:] = rng.poisson(image.array*gain)/gain
        rn = readnoise/gain
    else:
        rn = readnoise
    image.array += rng.normal(0.0, rn, image.array.shape)


class SyntheticConfig(Config):
    """Synthetic spectrograph configuration"""
    width = Field(dtype=int, default=512, doc="Width of image")
    height = Field(dtype=int, default=2048, doc="Height of image")
    separation = Field(dtype=float, default=50, doc="Separation between traces (pixels)")
    slope = Field(dtype=float, default=0.01, doc="Slope of the trace as a function of row")
    fwhm = Field(dtype=float, default=3.21, doc="Full width at half maximum (FWHM) of trace")
    gain = Field(dtype=float, default=1.23, doc="Detector gain (e/ADU)")
    readnoise = Field(dtype=float, default=4.321, doc="Detector read noise (e)")

    @property
    def dims(self):
        """Dimensions of the image"""
        return lsst.afw.geom.Extent2I(self.width, self.height)

    @property
    def traceCenters(self):
        """Center of each trace"""
        return np.arange(self.separation, self.width - self.separation, self.separation)

    @property
    def numFibers(self):
        """Number of fibers"""
        return len(self.traceCenters)

    @property
    def traceOffset(self):
        """Offset of trace from center as a function of row"""
        return self.slope*(np.arange(self.height) - 0.5*self.height)


def makeSyntheticFlat(config, xOffset=0.0, flux=1.0e5, addNoise=True, rng=None):
    """Make a flat image

    This provides a flat-field with a specific configuration.

    Parameters
    ----------
    config : `pfs.drp.stella.synthetic.SyntheticConfig`
        Configuration for synthetic spectrograph.
    xOffset : `float`
        Offset in x direction; this is like the slitOffset, but in pixels.
    flux : `float`
        Integrated flux for each row.
    addNoise : `bool`
        Add noise to the image?
    rng : `numpy.random.RandomState`
        Random number generator.

    Returns
    -------
    image : `lsst.afw.image.Image`
        Flat-field image.
    """
    image = makeSpectrumImage(flux, config.dims, config.traceCenters + xOffset, config.traceOffset,
                              config.fwhm)
    if addNoise:
        addNoiseToImage(image, config.gain, config.readnoise, rng)
    return image


def makeSyntheticArc(config, numLines=50, fwhm=4.321, flux=3.0e5, addNoise=True, rng=None):
    """Make an arc image

    This provides an arc with a specific configuration.

    Parameters
    ----------
    config : `pfs.drp.stella.synthetic.SyntheticConfig`
        Configuration for synthetic spectrograph.
    lines : `numpy.ndarray`, optional
        Centers for each line.
    fwhm : `float`
        Spectral full width at half maximum of lines.
    flux : `float`
        Flux of each line.
    addNoise : `bool`
        Add noise to the image?
    rng : `numpy.random.RandomState`
        Random number generator.

    Returns
    -------
    lines : `numpy.array`
        Line centers.
    spectrum : `numpy.array`
        Spectrum used in creation of image.
    image : `lsst.afw.image.Image`
        Arc image.
    """
    lines = np.linspace(0, config.height, numLines)
    yy = np.arange(config.height, dtype=np.float32)
    spectrum = np.zeros(config.height, dtype=np.float32)
    sigma = fwhm/(2*math.sqrt(2*math.log(2)))
    norm = 1.0/(sigma*math.sqrt(2*math.pi))
    for ll in lines:
        spectrum += np.exp(-0.5*((yy - ll)/sigma)**2)
    spectrum *= flux*norm
    image = makeSpectrumImage(spectrum, config.dims, config.traceCenters, config.traceOffset, config.fwhm)
    if addNoise:
        addNoiseToImage(image, config.gain, config.readnoise, rng)
    return Struct(lines=lines, spectrum=spectrum, image=image)


def makeSyntheticDetectorMap(config, numKnots=20):
    """Make a DetectorMap with a specific configuration

    Parameters
    ----------
    config : `pfs.drp.stella.synthetic.SyntheticConfig`
        Configuration for synthetic spectrograph.
    numKnots : `int`
        Number of spline knots for ``DetectorMap``.

    Returns
    -------
    detMap : `pfs.drp.stella.DetectorMap`
        Detector map.
    """
    bbox = lsst.afw.geom.Box2I(lsst.afw.geom.Point2I(0, 0), config.dims)
    fiberIds = np.arange(config.numFibers, dtype=np.int32)*10
    xCenter = np.ndarray((config.numFibers, config.height), dtype=np.float32)
    wavelength = np.ndarray((config.numFibers, config.height), dtype=np.float32)
    for ii in range(config.numFibers):
        xCenter[ii] = config.traceCenters[ii] + config.traceOffset
        wavelength[ii] = np.linspace(400.0, 950.0, config.height, dtype=np.float32)
    return DetectorMap(bbox, fiberIds, xCenter, wavelength, numKnots)
