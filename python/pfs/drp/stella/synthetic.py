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
from pfs.drp.stella.psfProfiles import PsfParams, WingParams, evaluateParametricPsf, gaussian1D
from pfs.drp.stella.psfPixelIntegration import integrateOverPixels

__all__ = [
    "makeSpectrumImage",
    "addNoiseToImage",
    "makeSyntheticFlat",
    "makeSyntheticArc",
    "makeSyntheticDetectorMap",
    "makeSyntheticPfsConfig",
    "SyntheticPsfConfig",
    "makeSyntheticPsfArc",
    "plotSyntheticPsfArc",
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
        spectrum = spectrum * np.ones(height, dtype=float)
    else:
        assert len(spectrum) == height
    assert len(traceOffsets) == height
    for row, (spec, offset) in enumerate(zip(spectrum, traceOffsets)):
        profile = np.zeros_like(xx)
        for center in traceCenters:
            pp = (np.exp(-0.5 * ((xx - center - offset) / sigma) ** 2)).astype(np.float32)
            profile += pp / pp.sum()
        image.array[row] = spec * profile

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
        image.array[:] = rng.poisson(image.array * gain) / gain
        rn = readnoise / gain
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
        buffer = self.separation + self.slope * 0.5 * self.height
        return np.arange(buffer, self.width - buffer, self.separation)

    @property
    def numFibers(self):
        """Number of fibers"""
        return len(self.traceCenters)

    @property
    def fiberId(self):
        """Array of fiber identifiers"""
        return 1 + np.arange(self.numFibers, dtype=np.int32) * 10

    @property
    def traceOffset(self):
        """Offset of trace from center as a function of row"""
        return self.slope * (np.arange(self.height) - 0.5 * self.height)


class SyntheticPsfConfig(SyntheticConfig):
    """Configuration for synthetic PSF-fitting test data

    Extends `SyntheticConfig` with a tightly-packed fiber layout (with slit
    gaps between blocks of fibers), a curved trace, an undersampled PSF core,
    and non-Gaussian wings with an injected non-parametric "bump" feature,
    suitable for exercising the hybrid parametric + regularized-spline PSF
    fit (PIPE2D-1823-psf).
    """

    separation = Field(dtype=float, default=6.5, doc="Cross-dispersion fiber pitch (pixels)")
    fwhm = Field(dtype=float, default=1.8, doc="Full width at half maximum (FWHM) of the PSF core (pixels)")
    coreWidth = Field(dtype=float, default=1.0, doc="Top-hat width of the fiber image core (pixels)")
    curvature = Field(
        dtype=float, default=2.0e-6, doc="Quadratic term of the trace offset as a function of row (pixels^-1)"
    )
    blockSize = Field(dtype=int, default=20, doc="Number of fibers per slit block, between gaps")
    gapWidth = Field(
        dtype=float, default=4.0, doc="Extra spacing between slit blocks, in units of the fiber pitch"
    )
    wingScaleX = Field(dtype=float, default=4.0, doc="Cross-dispersion scale length of the PSF wing (pixels)")
    wingScaleY = Field(
        dtype=float, default=5.5, doc="Dispersion-direction scale length of the PSF wing (pixels)"
    )
    wingBeta = Field(dtype=float, default=2.2, doc="Moffat exponent of the PSF wing")
    wingFraction = Field(dtype=float, default=0.03, doc="Fraction of flux in the PSF wing")
    bumpFraction = Field(
        dtype=float,
        default=0.01,
        doc="Fraction of flux in an injected, non-parametric wing feature, for testing "
        "recovery of real structure by the dP spline correction",
    )
    bumpOffsetX = Field(
        dtype=float, default=5.0, doc="Cross-dispersion offset of the injected bump (pixels)"
    )
    bumpOffsetY = Field(
        dtype=float, default=3.0, doc="Dispersion-direction offset of the injected bump (pixels)"
    )
    bumpWidth = Field(dtype=float, default=1.2, doc="Width (Gaussian sigma) of the injected bump (pixels)")
    psfHalfSize = Field(dtype=int, default=16, doc="Half-size of the PSF stamp (pixels)")
    oversampling = Field(dtype=int, default=4, doc="Oversampling factor for exact pixel integration")
    lineFluxMin = Field(dtype=float, default=3.0e3, doc="Minimum arc line flux (detector units)")
    lineFluxMax = Field(dtype=float, default=3.0e5, doc="Maximum arc line flux (detector units)")
    phaseJitter = Field(
        dtype=float,
        default=0.5,
        doc="Half-range of random sub-pixel jitter applied to nominal line rows (pixels)",
    )

    @property
    def traceOffset(self):
        """Offset of trace from center as a function of row (linear + quadratic term)"""
        rows = np.arange(self.height) - 0.5 * self.height
        return self.slope * rows + self.curvature * rows**2

    @property
    def traceCenters(self):
        """Center of each trace, grouped into blocks of ``blockSize`` fibers separated by slit gaps"""
        buffer = self.separation + np.max(np.abs(self.traceOffset))
        centers = []
        position = buffer
        count = 0
        while position < self.width - buffer:
            centers.append(position)
            count += 1
            step = self.separation * (1 + self.gapWidth) if count % self.blockSize == 0 else self.separation
            position += step
        return np.array(centers)

    @property
    def blockEdgeIndices(self):
        """Indices (into `traceCenters`/`fiberId`) of fibers at a slit edge or adjacent to a gap"""
        num = self.numFibers
        indices = {0, num - 1}
        for ii in range(self.blockSize - 1, num - 1, self.blockSize):
            indices.add(ii)
            indices.add(ii + 1)
        return np.array(sorted(indices))

    def makeGroundTruthPsf(self):
        """Construct the ground-truth parametric-backbone PSF parameters

        This is the part of the ground truth that ``P_param`` can, by
        construction, recover exactly; the injected "bump" feature (see
        `evaluatePsf`) is deliberately excluded from this parametric form, so
        that recovering it demonstrates the value of the ``dP`` correction.

        Returns
        -------
        params : `pfs.drp.stella.psfProfiles.PsfParams`
            Ground-truth parametric PSF parameters.
        """
        sigma = fwhmToSigma(self.fwhm)
        wing = WingParams(
            scaleX=self.wingScaleX, scaleY=self.wingScaleY, beta=self.wingBeta, fraction=self.wingFraction
        )
        return PsfParams(sigmaX=sigma, sigmaY=sigma, tophatWidth=self.coreWidth, wings=[wing])

    def evaluatePsf(self, dx, dy):
        """Evaluate the full ground-truth PSF, including the injected bump

        Parameters
        ----------
        dx, dy : `numpy.ndarray`
            Positions at which to evaluate, relative to the PSF center.

        Returns
        -------
        values : `numpy.ndarray`
            Unit-flux PSF evaluated at ``(dx, dy)``.
        """
        profile = evaluateParametricPsf(dx, dy, self.makeGroundTruthPsf())
        if self.bumpFraction > 0:
            bump = gaussian1D(dx - self.bumpOffsetX, self.bumpWidth) * gaussian1D(
                dy - self.bumpOffsetY, self.bumpWidth
            )
            profile = (1.0 - self.bumpFraction) * profile + self.bumpFraction * bump
        return profile


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
    image = makeSpectrumImage(
        flux, config.dims, config.traceCenters + xOffset, config.traceOffset, config.fwhm
    )
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
    norm = 1.0 / (sigma * math.sqrt(2 * math.pi))
    for ll in lines:
        spectrum += np.exp(-0.5 * ((yy - ll) / sigma) ** 2)
    spectrum *= flux * norm
    image = makeSpectrumImage(spectrum, config.dims, config.traceCenters, config.traceOffset, config.fwhm)
    if addNoise:
        addNoiseToImage(image, config.gain, config.readnoise, rng)
    return Struct(lines=lines, spectrum=spectrum, image=image)


def makeSyntheticPsfArc(config, numLines=40, addNoise=True, rng=None):
    """Make a synthetic arc image for testing the hybrid PSF fit

    Unlike `makeSyntheticArc` (which renders a fixed Gaussian trace profile
    sampled at pixel centers), this renders each arc line individually with
    ``config``'s ground-truth, undersampled, non-Gaussian PSF (see
    `SyntheticPsfConfig.evaluatePsf`), exactly integrated over pixels, so the
    resulting image and line list can be used to test recovery of the PSF
    model itself. Lines whose stamp would fall off the edge of the image are
    skipped.

    Parameters
    ----------
    config : `SyntheticPsfConfig`
        Configuration for the synthetic spectrograph and PSF.
    numLines : `int`, optional
        Nominal number of arc lines per fiber; actual row positions are
        jittered by up to ``config.phaseJitter`` pixels.
    addNoise : `bool`, optional
        Add noise to the image?
    rng : `numpy.random.RandomState`, optional
        Random number generator.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Result struct with elements:

        - ``image`` (`lsst.afw.image.Image`): the synthetic image.
        - ``fiberIndex`` (`numpy.ndarray` of `int`): 0-based fiber index of
          each line.
        - ``fiberId`` (`numpy.ndarray` of `int`): fiberId of each line.
        - ``row`` (`numpy.ndarray` of `float`): true dispersion-direction (y)
          center of each line.
        - ``xCenter`` (`numpy.ndarray` of `float`): true spatial-direction
          (x) center of each line.
        - ``amplitude`` (`numpy.ndarray` of `float`): true integrated flux of
          each line.
        - ``truePsf`` (`pfs.drp.stella.psfProfiles.PsfParams`): true
          parametric-backbone PSF parameters (excludes the injected
          non-parametric bump feature; see ``config.evaluatePsf``).
    """
    if rng is None:
        rng = np.random
    image = lsst.afw.image.ImageF(config.dims)
    image.array[:] = 0.0

    nominalRows = np.linspace(0, config.height - 1, numLines + 2)[1:-1]
    rowGrid = np.arange(config.height, dtype=float)
    traceCenters = config.traceCenters
    traceOffset = config.traceOffset
    halfSize = config.psfHalfSize
    logFluxMin, logFluxMax = np.log(config.lineFluxMin), np.log(config.lineFluxMax)

    fiberIndexList = []
    fiberIdList = []
    rowList = []
    xCenterList = []
    amplitudeList = []

    for index, fiberId in enumerate(config.fiberId):
        for nominalRow in nominalRows:
            lineRow = nominalRow + rng.uniform(-config.phaseJitter, config.phaseJitter)
            centerRow = int(np.round(lineRow))
            if centerRow - halfSize < 0 or centerRow + halfSize >= config.height:
                continue
            xTrace = traceCenters[index] + np.interp(lineRow, rowGrid, traceOffset)
            centerCol = int(np.round(xTrace))
            if centerCol - halfSize < 0 or centerCol + halfSize >= config.width:
                continue

            lineFlux = np.exp(rng.uniform(logFluxMin, logFluxMax))
            xIndices = np.arange(centerCol - halfSize, centerCol + halfSize + 1) - xTrace
            yIndices = np.arange(centerRow - halfSize, centerRow + halfSize + 1) - lineRow
            stamp = integrateOverPixels(config.evaluatePsf, xIndices, yIndices, config.oversampling)
            image.array[
                centerRow - halfSize : centerRow + halfSize + 1,
                centerCol - halfSize : centerCol + halfSize + 1,
            ] += (lineFlux * stamp).astype(np.float32)

            fiberIndexList.append(index)
            fiberIdList.append(fiberId)
            rowList.append(lineRow)
            xCenterList.append(xTrace)
            amplitudeList.append(lineFlux)

    if addNoise:
        addNoiseToImage(image, config.gain, config.readnoise, rng)

    return Struct(
        image=image,
        fiberIndex=np.array(fiberIndexList, dtype=int),
        fiberId=np.array(fiberIdList, dtype=np.int32),
        row=np.array(rowList, dtype=float),
        xCenter=np.array(xCenterList, dtype=float),
        amplitude=np.array(amplitudeList, dtype=float),
        truePsf=config.makeGroundTruthPsf(),
    )


def plotSyntheticPsfArc(result, config, index=0, show=True):
    """Plot a sample line stamp from a synthetic PSF arc

    Displays the pixel stamp around a single line (log scale) alongside its
    radial profile compared with the noise-free ground truth, for visual
    sanity-checking of `makeSyntheticPsfArc`.

    Parameters
    ----------
    result : `lsst.pipe.base.Struct`
        Return value of `makeSyntheticPsfArc`.
    config : `SyntheticPsfConfig`
        Configuration used to generate ``result``.
    index : `int`, optional
        Index of the line (into ``result.row``/``result.xCenter``) to plot.
    show : `bool`, optional
        Call ``matplotlib.pyplot.show()``?

    Returns
    -------
    figure : `matplotlib.figure.Figure`
        The resulting figure.
    """
    import matplotlib.pyplot as plt

    halfSize = config.psfHalfSize
    row = result.row[index]
    xCenter = result.xCenter[index]
    amplitude = result.amplitude[index]
    centerRow = int(np.round(row))
    centerCol = int(np.round(xCenter))
    stamp = np.array(
        result.image.array[
            centerRow - halfSize : centerRow + halfSize + 1, centerCol - halfSize : centerCol + halfSize + 1
        ]
    )

    xIndices = np.arange(centerCol - halfSize, centerCol + halfSize + 1) - xCenter
    yIndices = np.arange(centerRow - halfSize, centerRow + halfSize + 1) - row
    truth = amplitude * integrateOverPixels(config.evaluatePsf, xIndices, yIndices, config.oversampling)
    xGrid, yGrid = np.meshgrid(xIndices, yIndices, indexing="xy")
    radius = np.hypot(xGrid, yGrid)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    imagePlot = axes[0].imshow(np.log10(np.clip(stamp, 1.0e-3, None)), origin="lower")
    axes[0].set_title(f"fiberId={result.fiberId[index]}, row={row:.2f}")
    figure.colorbar(imagePlot, ax=axes[0])

    axes[1].semilogy(radius.ravel(), np.clip(stamp, 1.0e-3, None).ravel(), ".", alpha=0.4, label="data")
    axes[1].semilogy(radius.ravel(), truth.ravel(), ".", alpha=0.4, label="truth")
    axes[1].set_xlabel("radius (pixels)")
    axes[1].set_ylabel("flux")
    axes[1].legend()
    figure.tight_layout()

    if show:
        plt.show()
    return figure


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
    return SplinedDetectorMap(
        bbox, fiberId, [knots] * config.numFibers, xCenter, [knots] * config.numFibers, wavelength
    )


def makeSyntheticPfsConfig(
    config,
    pfsDesignId,
    visit,
    rng=None,
    raBoresight=60.0 * lsst.geom.degrees,
    decBoresight=30.0 * lsst.geom.degrees,
    posAng=0.0 * lsst.geom.degrees,
    arms="brn",
    fracSky=0.1,
    fracFluxStd=0.1,
):
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

    fov = 1.5 * lsst.geom.degrees
    pfiScale = 800000.0 / fov.asDegrees()  # microns/degree
    pfiErrors = 10  # microns

    rng = np.random.RandomState(12345)
    tract = rng.uniform(high=30000, size=numFibers).astype(int)
    patch = ["%d,%d" % tuple(xy.tolist()) for xy in rng.uniform(high=15, size=(numFibers, 2)).astype(int)]

    boresight = lsst.geom.SpherePoint(raBoresight, decBoresight)
    radius = np.sqrt(rng.uniform(size=numFibers)) * 0.5 * fov.asDegrees()  # degrees
    theta = rng.uniform(size=numFibers) * 2 * np.pi  # radians
    coords = [
        boresight.offset(tt * lsst.geom.radians, rr * lsst.geom.degrees) for rr, tt in zip(radius, theta)
    ]
    ra = np.array([cc.getRa().asDegrees() for cc in coords])
    dec = np.array([cc.getDec().asDegrees() for cc in coords])
    pfiNominal = (
        pfiScale * np.array([(rr * np.cos(tt), rr * np.sin(tt)) for rr, tt in zip(radius, theta)])
    ).astype(np.float32)
    pfiCenter = (pfiNominal + rng.normal(scale=pfiErrors, size=(numFibers, 2))).astype(np.float32)

    catId = rng.uniform(high=23, size=numFibers).astype(int)
    objId = rng.uniform(high=2**63, size=numFibers).astype(int)

    numSky = int(fracSky * numFibers + 0.5)
    numFluxStd = int(fracFluxStd * numFibers + 0.5)
    numObject = numFibers - numSky - numFluxStd

    targetType = np.array(
        [int(TargetType.SKY)] * numSky
        + [int(TargetType.FLUXSTD)] * numFluxStd
        + [int(TargetType.SCIENCE)] * numObject
    )
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

    fiberFlux = [
        np.array(fluxes if tt in (TargetType.SCIENCE, TargetType.FLUXSTD) else []) for tt in targetType
    ]

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

    filterNames = [
        ["g", "i", "y", "H"] if tt in (TargetType.SCIENCE, TargetType.FLUXSTD) else [] for tt in targetType
    ]

    return PfsConfig(
        pfsDesignId,
        visit,
        raBoresight.asDegrees(),
        decBoresight.asDegrees(),
        posAng.asDegrees(),
        arms,
        fiberId,
        tract,
        patch,
        ra,
        dec,
        catId,
        objId,
        targetType,
        fiberStatus,
        epoch,
        pmRa,
        pmDec,
        parallax,
        proposalId,
        obCode,
        fiberFlux,
        psfFlux,
        totalFlux,
        fiberFluxErr,
        psfFluxErr,
        totalFluxErr,
        filterNames,
        pfiCenter,
        pfiNominal,
        GuideStars.empty(),
    )
