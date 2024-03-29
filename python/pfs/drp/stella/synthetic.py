import math
import numpy as np
import astropy.units as u

import lsst.geom
import lsst.afw.image
from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct
from pfs.datamodel import PfsConfig, TargetType, FiberStatus, GuideStars
from pfs.drp.stella import SplinedDetectorMap
from pfs.drp.stella.utils.psf import fwhmToSigma

__all__ = ["makeSpectrumImage",
           "addNoiseToImage",
           "makeSyntheticFlat",
           "makeSyntheticArc",
           "makeSyntheticDetectorMap",
           "makeSyntheticPfsConfig",
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
    dims : `lsst.geom.Extent2I`
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
    sigma = fwhmToSigma(fwhm)
    width, height = dims
    xx = np.arange(width, dtype=float)

    if np.isscalar(spectrum):
        spectrum = spectrum*np.ones(height, dtype=float)
    else:
        assert len(spectrum) == height
    assert len(traceOffsets) == height
    for row, (spec, offset) in enumerate(zip(spectrum, traceOffsets)):
        profile = np.zeros_like(xx)
        for center in traceCenters:
            pp = (np.exp(-0.5*((xx - center - offset)/sigma)**2)).astype(np.float32)
            profile += pp/pp.sum()
        image.array[row] = spec*profile

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
        return lsst.geom.Extent2I(self.width, self.height)

    @property
    def traceCenters(self):
        """Center of each trace"""
        buffer = self.separation + self.slope*0.5*self.height
        return np.arange(buffer, self.width - buffer, self.separation)

    @property
    def numFibers(self):
        """Number of fibers"""
        return len(self.traceCenters)

    @property
    def fiberId(self):
        """Array of fiber identifiers"""
        return 1 + np.arange(self.numFibers, dtype=np.int32)*10

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
    numLines : `int`, optional
        Number of lines to generate.
    fwhm : `float`, optional
        Spectral full width at half maximum of lines.
    flux : `float`, optional
        Flux of each line.
    addNoise : `bool`, optional
        Add noise to the image?
    rng : `numpy.random.RandomState`, optional
        Random number generator.

    Returns
    -------
    lines : `numpy.array`
        Line centers (pixels).
    spectrum : `numpy.array`
        Spectrum used in creation of image.
    image : `lsst.afw.image.Image`
        Arc image.
    """
    lines = np.linspace(0, config.height - 1, numLines + 2)[1:-1]
    yy = np.arange(config.height, dtype=np.float32)
    spectrum = np.zeros(config.height, dtype=np.float32)
    sigma = fwhmToSigma(fwhm)
    norm = 1.0/(sigma*math.sqrt(2*math.pi))
    for ll in lines:
        spectrum += np.exp(-0.5*((yy - ll)/sigma)**2)
    spectrum *= flux*norm
    image = makeSpectrumImage(spectrum, config.dims, config.traceCenters, config.traceOffset, config.fwhm)
    if addNoise:
        addNoiseToImage(image, config.gain, config.readnoise, rng)
    return Struct(lines=lines, spectrum=spectrum, image=image)


def makeSyntheticDetectorMap(config, minWl=400.0, maxWl=950.0):
    """Make a DetectorMap with a specific configuration

    Parameters
    ----------
    config : `pfs.drp.stella.synthetic.SyntheticConfig`
        Configuration for synthetic spectrograph.
    minWl, maxWl : `float`, optional
        Minimum and maximum wavelengths.

    Returns
    -------
    detMap : `pfs.drp.stella.SplinedDetectorMap`
        Detector map.
    """
    bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), config.dims)
    fiberId = config.fiberId
    knots = np.arange(config.height, dtype=float)
    xCenter = []
    wavelength = []
    for ii in range(config.numFibers):
        xCenter.append((config.traceCenters[ii] + config.traceOffset).astype(float))
        wavelength.append(np.linspace(minWl, maxWl, config.height, dtype=float))
    return SplinedDetectorMap(bbox, fiberId, [knots]*config.numFibers, xCenter,
                              [knots]*config.numFibers, wavelength)


def makeSyntheticPfsConfig(config, pfsDesignId, visit, rng=None,
                           raBoresight=60.0*lsst.geom.degrees,
                           decBoresight=30.0*lsst.geom.degrees,
                           posAng=0.0*lsst.geom.degrees,
                           arms='brn',
                           fracSky=0.1, fracFluxStd=0.1):
    """Make a PfsConfig with a specific configuration

    Parameters
    ----------
    config : `pfs.drp.stella.synthetic.SyntheticConfig`
        Configuration for synthetic spectrograph.
    pfsDesignId : `int`
        Identifier for top-end design.
    visit : `int`
        Exposure identifier.
    rng : `numpy.random.RandomState`, optional
        Random number generator.
    raBoresight : `lsst.geom.Angle`, optional
        Right Ascension of boresight.
    decBoresight : `lsst.geom.Angle`, optional
        Declination of boresight.
    posAng : `lsst.geom.Angle`, optional
        Position Angle of PFI: the angle from the PFI_Y axis
        to the NCP, measured clockwise in direction
        of PFI_Z axis.
    arms : `str`, optional
        Arms exposed, eg 'brn'.
    fracSky : `float`, optional
        Fraction of fibers to claim are sky.
    fracFluxStd : `float`, optional
        Fraction of fibers to claim are flux standards.

    Returns
    -------
    pfsConfig : `pfs.datamodel.PfsConfig`
        Top-end configuration.
    """
    if rng is None:
        rng = np.random

    fiberId = config.fiberId
    numFibers = config.numFibers

    fov = 1.5*lsst.geom.degrees
    pfiScale = 800000.0/fov.asDegrees()  # microns/degree
    pfiErrors = 10  # microns

    rng = np.random.RandomState(12345)
    tract = rng.uniform(high=30000, size=numFibers).astype(int)
    patch = ["%d,%d" % tuple(xy.tolist()) for
             xy in rng.uniform(high=15, size=(numFibers, 2)).astype(int)]

    boresight = lsst.geom.SpherePoint(raBoresight, decBoresight)
    radius = np.sqrt(rng.uniform(size=numFibers))*0.5*fov.asDegrees()  # degrees
    theta = rng.uniform(size=numFibers)*2*np.pi  # radians
    coords = [boresight.offset(tt*lsst.geom.radians, rr*lsst.geom.degrees) for
              rr, tt in zip(radius, theta)]
    ra = np.array([cc.getRa().asDegrees() for cc in coords])
    dec = np.array([cc.getDec().asDegrees() for cc in coords])
    pfiNominal = (pfiScale*np.array([(rr*np.cos(tt), rr*np.sin(tt)) for
                                     rr, tt in zip(radius, theta)])).astype(np.float32)
    pfiCenter = (pfiNominal + rng.normal(scale=pfiErrors, size=(numFibers, 2))).astype(np.float32)

    catId = rng.uniform(high=23, size=numFibers).astype(int)
    objId = rng.uniform(high=2**63, size=numFibers).astype(int)

    numSky = int(fracSky*numFibers + 0.5)
    numFluxStd = int(fracFluxStd*numFibers + 0.5)
    numObject = numFibers - numSky - numFluxStd

    targetType = np.array([int(TargetType.SKY)]*numSky +
                          [int(TargetType.FLUXSTD)]*numFluxStd +
                          [int(TargetType.SCIENCE)]*numObject)
    rng.shuffle(targetType)

    fiberStatus = np.full_like(targetType, FiberStatus.GOOD)

    epoch = np.full(shape=numFibers, fill_value="J2000.0")
    pmRa = np.full(shape=numFibers, fill_value=0.0, dtype=np.float32)
    pmDec = np.full(shape=numFibers, fill_value=0.0, dtype=np.float32)
    parallax = np.full(shape=numFibers, fill_value=1e-5, dtype=np.float32)

    proposalId = np.full(numFibers, "S24B-001QN")
    obCode = np.array([f"obcode_{fibid:04d}" for fibid in range(numFibers)])

    fiberMagnitude = [22.0, 23.5, 25.0, 26.0]
    fluxes = [(f * u.ABmag).to_value(u.nJy) for f in fiberMagnitude]

    fiberFlux = [np.array(fluxes if
                          tt in (TargetType.SCIENCE, TargetType.FLUXSTD) else [])
                 for tt in targetType]

    # Assigning psfFlux and totalFlux the same values
    psfFlux = fiberFlux.copy()
    totalFlux = fiberFlux.copy()

    # All errors are 1% of the original fluxes.
    # Again, assigning the same values to
    # psfFluxErr and totalFluxErr
    # as to fiberFluxErr.
    fiberFluxErr = [0.01 * fFlux for fFlux in fiberFlux]
    psfFluxErr = fiberFluxErr.copy()
    totalFluxErr = fiberFluxErr.copy()

    filterNames = [["g", "i", "y", "H"] if tt in (TargetType.SCIENCE, TargetType.FLUXSTD) else []
                   for tt in targetType]

    return PfsConfig(pfsDesignId, visit, raBoresight.asDegrees(), decBoresight.asDegrees(),
                     posAng.asDegrees(),
                     arms,
                     fiberId, tract, patch, ra, dec, catId, objId, targetType, fiberStatus,
                     epoch, pmRa, pmDec, parallax,
                     proposalId, obCode,
                     fiberFlux, psfFlux, totalFlux,
                     fiberFluxErr, psfFluxErr, totalFluxErr,
                     filterNames, pfiCenter, pfiNominal, GuideStars.empty())
