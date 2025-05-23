import lsst.utils.tests
import lsst.utils
from pfs.drp.stella.tests import runTests

from pfs.datamodel.identity import Identity
from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.observations import Observations
from pfs.datamodel.pfsConfig import FiberStatus, PfsConfig, TargetType
from pfs.datamodel.pfsSimpleSpectrum import PfsSimpleSpectrum
from pfs.datamodel.target import Target
from pfs.drp.stella.datamodel import PfsFiberArraySet, PfsSingle
from pfs.drp.stella.dustMap import DustMap
from pfs.drp.stella.extinctionCurve import F99ExtinctionCurve
from pfs.drp.stella.fitFluxReference import (
    _trapezoidal,
    FilterCurve,
    FitFluxReferenceTask,
    FitFluxReferenceConfig,
)
from pfs.drp.stella.fluxModelSet import FluxModelSet
from pfs.drp.stella.interpolate import interpolateFlux
from pfs.drp.stella.lsf import GaussianLsf
from pfs.drp.stella.utils.psf import fwhmToSigma

import numpy as np

import unittest


class FilterCurveTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        try:
            self.np_random = np.random.default_rng(0x981808A8FA8A744C)
        except AttributeError:
            self.np_random = np.random
            self.np_random.seed(0xF6503311)

    def testTrapezoidal(self):
        """Test ``_trapezoidal()`` method."""
        x = np.array([2, 3, 5, 7, 11, 13, 17], dtype=float)
        y = w = np.ones_like(x)
        # assert _exact_ equality
        self.assertEqual(_trapezoidal(x, y, w), x[-1] - x[0])

    def testPhotometer(self):
        """Test ``TransmissionCurve.photometer()`` method"""
        nSamples = 50
        wavelength = np.linspace(400, 1000, num=nSamples)
        expectedFlux = (wavelength - wavelength[0]) * (3 / (wavelength[-1] - wavelength[0]))
        variance = 2 + np.sin((wavelength - wavelength[0]) * (4 * np.pi / (wavelength[-1] - wavelength[0])))
        stddev = np.sqrt(variance)

        covar = np.zeros(shape=(3, nSamples))
        covar[0, :] = variance

        filterCurve = FilterCurve("i2_hsc")
        target = Target(0, 0, "0,0", 0)
        observations = Observations(
            visit=np.zeros(shape=1),
            arm=["b"],
            spectrograph=np.ones(shape=1),
            pfsDesignId=np.zeros(shape=1),
            fiberId=np.zeros(shape=1),
            pfiNominal=np.zeros(shape=(1, 2)),
            pfiCenter=np.zeros(shape=(1, 2)),
        )
        maskHelper = MaskHelper()

        photometries = []
        photoVars = []

        for i in range(1000):
            flux = expectedFlux + stddev * self.np_random.normal(size=nSamples)

            spectrum = PfsSingle(
                target=target,
                observations=observations,
                wavelength=wavelength,
                flux=flux,
                mask=np.zeros(shape=nSamples, dtype=int),
                sky=np.zeros(shape=nSamples, dtype=int),
                covar=covar,
                covar2=np.zeros(shape=(1, 1), dtype=int),
                flags=maskHelper,
            )

            photo, error = filterCurve.photometer(spectrum, doComputeError=True)
            photometries.append(photo)
            photoVars.append(error**2)

        measuredPhotoVar = np.var(photometries, ddof=1)
        estimatedPhotoVar = np.mean(photoVars)
        self.assertAlmostEqual(measuredPhotoVar, estimatedPhotoVar, places=1)


try:
    fluxmodeldataDir = lsst.utils.getPackageDir("fluxmodeldata")
    dustmapsDir = lsst.utils.getPackageDir("dustmaps_cachedata")
except LookupError:
    fluxmodeldataDir = None
    dustmapsDir = None


@unittest.skipIf(
    (fluxmodeldataDir is None) or (dustmapsDir is None), "fluxmodeldata or dustmaps_cachedata not setup"
)
class FitFluxReferenceTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        try:
            self.np_random = np.random.default_rng(0x981808A8FA8A744C)
        except AttributeError:
            self.np_random = np.random
            self.np_random.seed(0xF6503311)

        self.task = FitFluxReferenceTask(config=FitFluxReferenceConfig())
        self.dustMap = DustMap()
        self.extinctionCurve = F99ExtinctionCurve()
        self.modelSet = FluxModelSet(fluxmodeldataDir)

    def testRun(self):
        """Test run() method"""
        nFluxStd = 1
        parameters, pfsConfig, pfsMerged, pfsMergedLsf = self.inventPfsMerged(
            nFluxStd=nFluxStd, nFibers=10, nSamples=1000, snr=10, bbSnr=100
        )
        fluxReference = self.task.run(pfsConfig, pfsMerged, pfsMergedLsf)

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
        pfsMerged : `pfs.drp.stella.datamodel.PfsFiberArraySet`
            Observed spectra.
        pfsMergedLsf : `dict` (`int`: `pfs.drp.stella.Lsf`)
            Combined line-spread functions indexed by fiberId.
        """
        parameters = self.modelSet.parameters[
            self.np_random.choice(len(self.modelSet.parameters), size=nFibers, replace=False)
        ]

        identity = Identity(visit=0, arm="brn", spectrograph=1, pfsDesignId=0)
        fiberId = np.arange(1, nFibers + 1, dtype=np.int32)
        radecs = self.getRandomLonLat(size=nFibers)
        ebvs = self.dustMap(radecs[:, 0], radecs[:, 1])

        wavelength = np.empty(shape=(nFibers, nSamples), dtype=float)
        wavelength[...] = np.linspace(400, 1200, num=nSamples, dtype=float).reshape(1, -1)
        flux = np.empty(shape=wavelength.shape, dtype=float)
        pfsMergedLsf = {}

        for i, (m, ebv) in enumerate(zip(parameters, ebvs)):
            spectrum = self.modelSet.getSpectrum(
                m["teff"],
                m["logg"],
                m["m"],
                m["alpha"],
            )
            spectrum.flux *= self.extinctionCurve.attenuation(spectrum.wavelength, ebv)
            convolvedFlux, lsf = convolveLsf(spectrum.wavelength, spectrum.flux)
            flux[i] = interpolateFlux(spectrum.wavelength, convolvedFlux, wavelength[i])
            pfsMergedLsf[fiberId[i]] = lsf.warp(spectrum.wavelength, wavelength[i])

        mask = np.zeros(shape=wavelength.shape, dtype=np.uint32)
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
        ).astype(mask.dtype)

        pfsMerged = PfsFiberArraySet(
            identity, fiberId, wavelength, flux, mask, sky, norm, covar, flags, metadata
        )
        pfsConfig = self.inventPfsConfig(pfsMerged, radecs[:, 0], radecs[:, 1], nFluxStd, bbSnr)

        return parameters, pfsConfig, pfsMerged, pfsMergedLsf

    def inventPfsConfig(self, pfsMerged, ra, dec, nFluxStd, bbSnr):
        """Invent a ``PfsConfig`` object.

        Parameters
        ----------
        pfsMerged : `pfs.drp.stella.datamodel.PfsFiberArraySet`
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
        bbFilters = ["g_hsc", "r2_hsc", "i2_hsc", "z_hsc", "y_hsc"]

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
        visit = 0
        raBoresight = 0.0
        decBoresight = 0.0
        posAng = 0.0
        arms = "brn"
        tract = np.full(fiberId.shape, 0, dtype=np.int32)
        patch = np.full(fiberId.shape, "0,0", dtype="U8")
        catId = np.full(fiberId.shape, 0, dtype=np.int32)
        objId = np.arange(100, 100 + len(fiberId), dtype=np.int64)

        targetType = np.full(fiberId.shape, TargetType.SCIENCE, dtype=int)
        targetType[self.np_random.choice(len(targetType), nFluxStd, replace=False)] = TargetType.FLUXSTD

        fiberStatus = np.full(fiberId.shape, FiberStatus.GOOD, dtype=int)

        epoch = np.full(shape=fiberId.shape, fill_value="J2000.0")
        pmRa = np.full(shape=fiberId.shape, fill_value=0.0, dtype=np.float32)
        pmDec = np.full(shape=fiberId.shape, fill_value=0.0, dtype=np.float32)
        parallax = np.full(shape=fiberId.shape, fill_value=1e-5, dtype=np.float32)

        proposalId = np.full(fiberId.shape, "S24B-001QN")
        obCode = np.array([f"obcode_{fibid:04d}" for fibid in range(len(ra))])

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
            visit,
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
            chunk = self.np_random.uniform(-1, 1, size=(2 * n, 3))
            r = np.hypot(chunk[:, 0], np.hypot(chunk[:, 1], chunk[:, 2]))
            chunk = chunk[(0.5 < r) & (r < 1.0)]
            chunk = chunk[:n]
            xyz[(len(xyz) - n) : (len(xyz) - n + len(chunk))] = chunk
            n -= len(chunk)

        lonlat = np.empty(shape=(len(xyz), 2), dtype=float).reshape(-1, 2)
        lonlat[:, 0] = np.arctan2(xyz[:, 1], xyz[:, 0])
        lonlat[:, 1] = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 1], xyz[:, 0]))
        return np.degrees(lonlat).reshape(size + (2,))


def convolveLsf(wavelength, flux, fwhm=0.2):
    """Convolve a Gaussian LSF to ``flux``

    Parameters
    ----------
    wavelength : `numpy.array` of `float`
        Wavelength in nm.
    flux : `numpy.array` of `float`
        Flux.
    fwhm : `float`
        LSF's FWHM in nm.

    Returns
    -------
    convolvedFlux : `numpy.array` of `float`
        Flux with the LSF convolved to it.
    lsf : `pfs.drp.stella.Lsf`
        Used LSF.
    """
    n = len(wavelength)
    dlambda = wavelength[n // 2 + 1] - wavelength[n // 2]
    sigma = fwhmToSigma(fwhm)
    lsf = GaussianLsf(length=n, width=sigma / dlambda)
    convolvedFlux = lsf.computeKernel((n - 1) / 2.0).convolve(flux)
    return convolvedFlux, lsf


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
    mask = np.zeros(shape=wavelength.shape, dtype=np.uint32)
    flags = MaskHelper()
    return PfsSimpleSpectrum(target, wavelength, flux, mask, flags)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
