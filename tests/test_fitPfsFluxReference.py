import lsst.utils.tests
import lsst.utils
from pfs.drp.stella.tests import runTests

from pfs.datamodel.identity import Identity
from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.pfsConfig import FiberStatus, PfsConfig, TargetType
from pfs.datamodel.pfsFiberArraySet import PfsFiberArraySet
from pfs.datamodel.pfsSimpleSpectrum import PfsSimpleSpectrum
from pfs.datamodel.target import Target
from pfs.drp.stella.dustMap import DustMap
from pfs.drp.stella.extinctionCurve import F99ExtinctionCurve
from pfs.drp.stella.fitPfsFluxReference import FitPfsFluxReferenceTask, FitPfsFluxReferenceConfig
from pfs.drp.stella.fitReference import FilterCurve
from pfs.drp.stella.fluxModelSet import FluxModelSet
from pfs.drp.stella.interpolate import interpolateFlux
from pfs.drp.stella.lsf import GaussianKernel1D
from pfs.drp.stella.utils.psf import fwhmToSigma

import numpy as np

import unittest

try:
    fluxmodeldataDir = lsst.utils.getPackageDir("fluxmodeldata")
    dustmapsDir = lsst.utils.getPackageDir("dustmaps_cachedata")
except LookupError:
    fluxmodeldataDir = None
    dustmapsDir = None


@unittest.skipIf(
    (fluxmodeldataDir is None) or (dustmapsDir is None),
    "fluxmodeldata or dustmaps_cachedata not setup")
class FitPfsFluxReferenceTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        try:
            self.np_random = np.random.default_rng(0x981808a8fa8a744c)
        except AttributeError:
            self.np_random = np.random
            self.np_random.seed(0xf6503311)

        self.task = FitPfsFluxReferenceTask(config=FitPfsFluxReferenceConfig())
        self.dustMap = DustMap()
        self.extinctionCurve = F99ExtinctionCurve()
        self.modelSet = FluxModelSet(fluxmodeldataDir)

    def testRun(self):
        """Test run() method
        """
        nFluxStd = 1
        parameters, pfsConfig, pfsMerged = self.inventPfsMerged(
            nFluxStd=nFluxStd, nFibers=10, nSamples=1000, snr=10, bbSnr=100
        )
        fluxReference = self.task.run(pfsConfig, pfsMerged)

        self.assertEqual(len(fluxReference.fiberId), nFluxStd)

    def inventPfsMerged(self, nFluxStd, nFibers, nSamples, snr, bbSnr):
        """Invent a ``PfsMerged`` object.

        Parameters
        ----------
        nFluxStd : `int`
            Number of spectra whose targets are ``TargetType.FLUXSTD``.
        nFibers : `int`
            Number of total fibers.
        nSamples : `int`
            Number of sampling points along the wavelength axis.
        snr : `float`
            Signal to noise ratio for spectra. (variance = (flux / snr)**2).
            This is used only for inventing ``pfsMerged.covar`` member.
            This amount of noise won't be added to the flux.
        bbSnr : `float`
            Signal to noise ratio for broadband photometries.
            This is used only for inventing ``pfsConfig.fluxErr`` member.
            This amount of noise won't be added to the flux.

        Returns
        -------
        parameters : `numpy.array`
            Structured array indicating input parameters.
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            Configuration of the PFS top-end.
        pfsMerged : `pfs.datamodel.pfsFiberArraySet.PfsFiberArraySet`
            Observed spectra.
        """
        parameters = self.modelSet.parameters[
            self.np_random.choice(len(self.modelSet.parameters), size=nFibers, replace=False)
        ]

        identity = Identity(visit=0, arm="r", spectrograph=1, pfsDesignId=0)
        fiberId = np.arange(1, nFibers + 1, dtype=np.int32)
        radecs = self.getRandomLonLat(size=nFibers)
        ebvs = self.dustMap(radecs[:, 0], radecs[:, 1])

        wavelength = np.empty(shape=(nFibers, nSamples), dtype=float)
        wavelength[...] = np.linspace(400, 1200, num=nSamples, dtype=float).reshape(1, -1)
        flux = np.empty(shape=wavelength.shape, dtype=float)

        for i, (m, ebv) in enumerate(zip(parameters, ebvs)):
            spectrum = self.modelSet.getSpectrum(
                m["teff"], m["logg"], m["m"], m["alpha"],
            )
            spectrum.flux *= self.extinctionCurve.attenuation(spectrum.wavelength, ebv)
            flux[i] = interpolateFlux(
                spectrum.wavelength,
                convolveLsf(spectrum.wavelength, spectrum.flux),
                wavelength[i]
            )

        mask = np.zeros(shape=wavelength.shape, dtype=int)
        sky = np.zeros(shape=wavelength.shape, dtype=float)
        norm = np.ones(shape=wavelength.shape, dtype=float)

        covar = np.zeros(shape=(len(fiberId), 3, wavelength.shape[-1]), dtype=float)
        covar[:, 0, :] = np.square(flux / snr)

        flags = MaskHelper()
        metadata = {}

        mask |= np.where(
            np.isfinite(flux) & (flux > 0),
            0,
            flags.add("BAD"),
        )
        pfsMerged = PfsFiberArraySet(
            identity, fiberId, wavelength, flux, mask, sky, norm, covar, flags, metadata
        )
        pfsConfig = self.inventPfsConfig(pfsMerged, radecs[:, 0], radecs[:, 1], nFluxStd, bbSnr)

        return parameters, pfsConfig, pfsMerged

    def inventPfsConfig(self, pfsMerged, ra, dec, nFluxStd, bbSnr):
        """Invent a ``PfsConfig`` object.

        Parameters
        ----------
        pfsMerged : `pfs.datamodel.pfsFiberArraySet.PfsFiberArraySet`
            Observed spectra. Broadband fluxes will be computed from these spectra.
        ra : `numpy.array` of `float`
            R.A. in degrees.
        dec : `numpy.array` of `float`
            Dec. in degrees.
        nFluxStd : `int`
            Number of spectra whose targets are ``TargetType.FLUXSTD``.
        bbSnr : `float`
            Signal to noise ratio for broadband photometries.
            This is used only for inventing ``pfsConfig.fluxErr`` member.
            This amount of noise won't be added to the flux.

        Returns
        -------
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            Configuration of the PFS top-end.
        """
        bbFilters = ["g", "r2", "i2", "z", "y"]

        fiberId = pfsMerged.fiberId
        flux = np.empty(shape=(len(fiberId), len(bbFilters)), dtype=float)
        for j, filter in enumerate(bbFilters):
            curve = FilterCurve(filter)
            for i in range(len(fiberId)):
                # We cannot use pfsMerged.extractFiber()
                # because it requires pfsConfig.
                spectrum = makePfsSimpleSpectrum(pfsMerged.wavelength[i], pfsMerged.flux[i])
                flux[i, j] = curve.photometer(spectrum)

        fluxErr = flux / bbSnr

        pfsDesignId = 0
        visit0 = 0
        raBoresight = 0.0
        decBoresight = 0.0
        posAng = 0.0
        arms = "brn"
        tract = np.full(fiberId.shape, 0, dtype=np.int32)
        patch = np.full(fiberId.shape, "0,0", dtype="U8")
        catId = np.full(fiberId.shape, 0, dtype=np.int32)
        objId = np.arange(100, 100+len(fiberId), dtype=np.int64)

        targetType = np.full(fiberId.shape, TargetType.SCIENCE, dtype=int)
        targetType[self.np_random.choice(len(targetType), nFluxStd, replace=False)] = TargetType.FLUXSTD

        fiberStatus = np.full(fiberId.shape, FiberStatus.GOOD, dtype=int)

        fiberFlux = np.copy(flux)
        psfFlux = np.copy(flux)
        totalFlux = np.copy(flux)

        fiberFluxErr = np.copy(fluxErr)
        psfFluxErr = np.copy(fluxErr)
        totalFluxErr = np.copy(fluxErr)

        filterNames = [list(bbFilters) for i in range(len(fiberId))]
        pfiCenter = np.full(fiberId.shape + (2,), 0.0, dtype=float)
        pfiNominal = np.full(fiberId.shape + (2,), 0.0, dtype=float)
        guideStars = None

        return PfsConfig(
            pfsDesignId,
            visit0,
            raBoresight,
            decBoresight,
            posAng,
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
            fiberFlux,
            psfFlux,
            totalFlux,
            fiberFluxErr,
            psfFluxErr,
            totalFluxErr,
            filterNames,
            pfiCenter,
            pfiNominal,
            guideStars,
        )

    def getRandomLonLat(self, size=()):
        """Get random points uniformly distributed
        in the surface of the unit sphere.

        Parameters
        ----------
        size : `int` or `tuple` of `int`
            Number of points to get.

        Returns
        -------
        lonlat : `numpy.array` of `float`
            Longitudes and latitudes (in degrees) of the random points.
            The shape of this array will be `size` + (2,).
        """
        if not hasattr(size, "__iter__"):
            size = (size,)

        xyz = np.empty(shape=size + (3,), dtype=float).reshape(-1, 3)
        n = len(xyz)
        while n > 0:
            chunk = self.np_random.uniform(-1, 1, size=(2*n, 3))
            r = np.hypot(chunk[:, 0], np.hypot(chunk[:, 1], chunk[:, 2]))
            chunk = chunk[(0.5 < r) & (r < 1.0)]
            chunk = chunk[:n]
            xyz[(len(xyz) - n):(len(xyz) - n + len(chunk))] = chunk
            n -= len(chunk)

        lonlat = np.empty(shape=(len(xyz), 2), dtype=float).reshape(-1, 2)
        lonlat[:, 0] = np.arctan2(xyz[:, 1], xyz[:, 0])
        lonlat[:, 1] = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 1], xyz[:, 0]))
        return np.degrees(lonlat).reshape(size + (2,))


def convolveLsf(wavelength, flux):
    """Convolve a typical LSF to ``flux``

    Parameters
    ----------
    wavelength : `numpy.array` of `float`
        Wavelength in nm.
    flux : `numpy.array` of `float`
        Flux.

    Returns
    -------
    convolvedFlux : `numpy.array` of `float`
        Flux with the typical LSF convolved to it.
    """
    fwhm = 0.2  # typical FWHM, in nm, of LSF.
    n = len(wavelength)
    dlambda = wavelength[n//2 + 1] - wavelength[n//2]
    sigma = fwhmToSigma(fwhm)
    return GaussianKernel1D(width=sigma / dlambda).convolve(flux)


def makePfsSimpleSpectrum(wavelength, flux):
    """Make a ``PfsSimpleSpectrum`` object.

    Parameters
    ----------
    wavelength : `numpy.array` of `float`
        Wavelength in nm.
    flux : `numpy.array` of `float`
        Flux.

    Returns
    -------
    pfsSimpleSpectrum : `pfs.datamodel.pfsSimpleSpectrum.PfsSimpleSpectrum`
        Constructed ``PfsSimpleSpectrum`` object.
    """
    target = Target(0, 0, "0,0", 0)
    mask = np.zeros(shape=wavelength.shape, dtype=int)
    flags = MaskHelper()
    return PfsSimpleSpectrum(target, wavelength, flux, mask, flags)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
