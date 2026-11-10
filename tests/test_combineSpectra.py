"""Claude: pin the current spectral combination (PIPE2D-1993)

The combination code is being refactored (a shared resampler; the weights split
into a policy and a common application step) without changing its results.
These tests compare the live code, bit for bit, with frozen copies of the code
as it was before the refactor (``_reference*``), on synthetic spectra with
overlapping wavelength ranges, masked, suspect and no-data pixels, zero and
non-finite norms and variances.
"""
import unittest

import numpy as np

import lsst.utils.tests
from pfs.datamodel import Identity, MaskHelper, Observations, Target
from pfs.drp.stella.coaddSpectra import CoaddSpectraConfig, CoaddSpectraTask
from pfs.drp.stella.combineSpectra import combineSpectraSets
from pfs.drp.stella.datamodel import PfsArm, PfsSingle
from pfs.drp.stella.interpolate import interpolate, interpolateFlux
from pfs.drp.stella.lsf import GaussianLsf


def _referenceResampleSet(self, wavelength, order=3, bad=None, minWeight=0.1):
    """``PfsFiberArraySet.resample`` (all fibres) as of master bff326cf"""
    fiberId = self.fiberId
    numCovar = order + 1
    numSpectra = len(fiberId)
    numSamples = len(wavelength)
    flux = np.empty((numSpectra, numSamples), dtype=self.flux.dtype)
    mask = np.empty((numSpectra, numSamples), dtype=self.mask.dtype)
    sky = np.empty((numSpectra, numSamples), dtype=self.sky.dtype)
    norm = np.empty((numSpectra, numSamples), dtype=self.norm.dtype)
    covar = np.zeros((numSpectra, numCovar, numSamples), dtype=self.covar.dtype)
    badMask = self.flags.get(*bad) if bad else 0
    kwargs = dict(fill=0.0, order=order)
    for ii, ff in enumerate(fiberId):
        jj = np.argwhere(self.fiberId == ff)[0][0]
        norm[ii] = interpolateFlux(self.wavelength[jj], self.norm[jj], wavelength, **kwargs)
        badNorm = (self.norm[jj] == 0) | ~np.isfinite(self.norm[jj])
        badFlux = ~np.isfinite(self.flux[jj])
        badVar = badNorm | ~np.isfinite(self.variance[jj])
        badSky = badNorm | ~np.isfinite(self.sky[jj])
        isBad = badFlux | badVar | badSky
        mm = self.mask[jj].copy()
        mm[isBad] |= self.flags["NO_DATA"]
        result = interpolate(
            self.wavelength[jj],
            np.where(badFlux, 0.0, self.flux[jj]).astype(np.float32),
            mm,
            np.where(badVar, 0.0, self.covar[jj][0]).astype(np.float32),
            wavelength,
            minWeight=minWeight,
            badMask=badMask,
            fillMask=self.flags.get("NO_DATA"),
            numCovar=numCovar,
            **kwargs,
        )
        flux[ii] = result.flux
        mask[ii] = result.mask
        covar[ii] = result.covariance
        sky[ii] = interpolateFlux(self.wavelength[jj], np.where(badSky, 0.0, self.sky[jj]), wavelength,
                                  **kwargs)
    return type(self)(self.identity, fiberId, np.concatenate([[wavelength]]*numSpectra),
                      flux, mask, sky, norm, covar, self.flags, self.metadata)


def _referenceCombineSpectraSets(spectra, flags, badMasks, suspectMasks=None):
    """``combineSpectraSets`` as of master bff326cf (returns flux, sky, norm, covar, mask)"""
    archetype = spectra[0]

    def makeData():
        return dict(
            mask=np.zeros_like(archetype.mask),
            flux=np.zeros_like(archetype.flux),
            sky=np.zeros_like(archetype.sky),
            norm=np.zeros_like(archetype.norm),
            covar=np.zeros_like(archetype.covar),
            count=np.zeros_like(archetype.flux, dtype=int),
            sumWeights=np.zeros_like(archetype.flux),
        )

    goodData = makeData()
    badData = makeData()
    suspectData = None
    if suspectMasks:
        badMasks = list(badMasks) + list(suspectMasks)
        suspectData = makeData()

    def accumulate(data, spectrum, weight, select):
        with np.errstate(invalid="ignore"):
            factor = weight[select]/spectrum.norm[select]
            data["flux"][select] += spectrum.flux[select]*factor
            data["sky"][select] += spectrum.sky[select]*factor
            data["norm"][select] += spectrum.norm[select]*weight[select]
        data["mask"][select] |= spectrum.mask[select]
        data["count"][select] += 1
        data["sumWeights"][select] += weight[select]

    for ss in spectra:
        with np.errstate(invalid="ignore", divide="ignore"):
            variance = ss.variance/ss.norm**2
            goodVariance = np.isfinite(variance) & (variance > 0)
        good = ((ss.mask & ss.flags.get(*badMasks)) == 0) & goodVariance
        suspect = None
        if suspectMasks:
            suspect = ((ss.mask & ss.flags.get(*suspectMasks)) != 0) & goodVariance
        bad = ((ss.mask & ss.flags.get(*badMasks)) != 0) & goodVariance
        weight = np.zeros_like(ss.flux)
        weight[goodVariance] = 1.0/variance[goodVariance]
        accumulate(goodData, ss, weight, good)
        if suspectMasks and np.any(suspect):
            accumulate(suspectData, ss, weight, suspect)
        accumulate(badData, ss, weight, bad)

    def calculate(target, source, select):
        with np.errstate(invalid="ignore", divide="ignore"):
            factor = 1.0/source["sumWeights"][select]
            target["flux"][select] = source["flux"][select]*factor
            target["sky"][select] = source["sky"][select]*factor
            target["norm"][select] = source["norm"][select]*factor
            target["covar"][:, 0][select] = factor

    good = (goodData["count"] > 0) & (goodData["sumWeights"] > 0)
    calculate(goodData, goodData, good)
    goodData["mask"][good] &= ~goodData["mask"].dtype.type(flags.get(*badMasks))
    bad = ~good
    if suspectMasks:
        useSuspect = (goodData["count"] == 0) & (suspectData["count"] > 0)
        if np.any(useSuspect):
            calculate(goodData, suspectData, useSuspect)
            goodData["mask"][useSuspect] |= suspectData["mask"][useSuspect]
        bad &= ~useSuspect
    calculate(goodData, badData, bad & (badData["sumWeights"] > 0))
    goodData["mask"][bad] |= flags["NO_DATA"]
    for ss in spectra:
        goodData["mask"][bad] |= ss.mask[bad]
    with np.errstate(invalid="ignore"):
        return dict(
            flux=goodData["flux"]*goodData["norm"],
            sky=goodData["sky"]*goodData["norm"],
            norm=goodData["norm"],
            covar=goodData["covar"]*goodData["norm"][:, np.newaxis, :]**2,
            mask=goodData["mask"],
        )


def _referenceCoaddCombine(config, spectraList, flags, wavelength):
    """The flux part of ``CoaddSpectraTask.combine`` as of master bff326cf"""
    resampled = [spectrum.resample(wavelength, order=config.resampleOrder,
                                   minWeight=config.resampleMinWeight, bad=config.mask)
                 for spectrum in spectraList]
    numCovar = config.resampleOrder + 1
    archetype = resampled[0]
    length = archetype.length
    mask = np.zeros(length, dtype=archetype.mask.dtype)
    flux = np.zeros(length, dtype=archetype.flux.dtype)
    sky = np.zeros(length, dtype=archetype.sky.dtype)
    covar = np.zeros((numCovar, length), dtype=archetype.covar.dtype)
    sumWeights = np.zeros(length, dtype=archetype.flux.dtype)
    for ss in resampled:
        weight = np.zeros_like(flux)
        with np.errstate(invalid="ignore", divide="ignore"):
            good = ((ss.mask & ss.flags.get(*config.mask)) == 0) & (ss.covar[0] > 0)
            ww = 1.0/ss.covar[0][good]
            weight[good] = ww
            flux[good] += ss.flux[good]*ww
            sky[good] += ss.sky[good]*ww
            mask[good] |= ss.mask[good]
            covar[:, good] += ss.covar[:, good]*ww**2
            sumWeights += weight
    good = sumWeights > 0
    ww = sumWeights[good]
    flux[good] /= ww
    sky[good] /= ww
    covar[:, good] /= ww**2
    covar[:, ~good] = np.inf
    mask[~good] = flags["NO_DATA"]
    return dict(flux=flux, sky=sky, covar=covar, mask=mask)


# Arm wavelength ranges (nm) with b/r and r/n overlaps, as in PFS
ARMS = dict(b=(380.0, 650.0, 4096), r=(630.0, 970.0, 4096), n=(940.0, 1260.0, 4096))
PLANES = ("NO_DATA", "SUSPECT", "BAD", "CR", "SAT", "BAD_SKY", "BAD_FLUXCAL", "BAD_FIBERNORMS")


def makeFlags():
    flags = MaskHelper()
    for name in PLANES:
        flags.add(name)
    return flags


def corrupt(rng, flags, flux, mask, variance, norm, sky):
    """Sprinkle masked, suspect and non-finite pixels into one spectrum (in place)"""
    size = flux.size
    for name, frac in (("CR", 0.01), ("BAD", 0.005), ("SUSPECT", 0.02), ("NO_DATA", 0.003), ("SAT", 0.002),
                       ("BAD_SKY", 0.002), ("BAD_FLUXCAL", 0.001), ("BAD_FIBERNORMS", 0.001)):
        mask[rng.random(size) < frac] |= flags[name]
    start = rng.integers(0, size - 50)
    mask[start:start + 40] |= flags["SUSPECT"]          # a run of suspect pixels
    flux[rng.random(size) < 0.002] = np.nan
    variance[rng.random(size) < 0.002] = np.nan
    variance[rng.random(size) < 0.002] = 0.0
    if norm is not None:
        norm[rng.random(size) < 0.002] = 0.0
    sky[rng.random(size) < 0.001] = np.nan


def makeArm(rng, arm, fiberId, flags):
    """A PfsArm with a smooth continuum, lines, noise and corrupted pixels"""
    lo, hi, length = ARMS[arm]
    num = len(fiberId)
    wavelength = np.array([np.linspace(lo, hi, length) + rng.normal(0.0, 0.05) for _ in range(num)])
    norm = 1000.0*(1.0 + 0.3*np.sin(wavelength/37.0))*rng.uniform(0.8, 1.2, size=(num, 1))
    flux = norm*(1.0 + 0.1*np.cos(wavelength/11.0))
    for center in rng.uniform(lo, hi, size=20):
        flux += 5000.0*np.exp(-0.5*((wavelength - center)/0.15)**2)
    sky = 0.2*flux + 10.0
    variance = np.abs(flux) + 100.0
    flux = flux + rng.normal(size=flux.shape)*np.sqrt(variance)
    mask = np.zeros(flux.shape, dtype=np.int32)
    for ii in range(num):
        corrupt(rng, flags, flux[ii], mask[ii], variance[ii], norm[ii], sky[ii])
    covar = np.zeros((num, 3, length))
    covar[:, 0] = variance
    identity = Identity(visit=1, arm=arm, spectrograph=1, pfsDesignId=0)
    return PfsArm(identity, np.array(fiberId), wavelength, flux, mask, sky, norm, covar, flags, {})


def makeSingle(rng, visit, arm, flags):
    """A PfsSingle (one visit's arm) for the coadd"""
    lo, hi, length = ARMS[arm]
    wavelength = np.linspace(lo, hi, length) + rng.normal(0.0, 0.05)
    flux = 1.0 + 0.1*np.cos(wavelength/11.0)
    for center in rng.uniform(lo, hi, size=20):
        flux += 5.0*np.exp(-0.5*((wavelength - center)/0.15)**2)
    sky = 0.2*flux + 0.01
    variance = (0.01 + 0.001*np.abs(flux))*rng.uniform(0.5, 2.0)
    flux = flux + rng.normal(size=flux.shape)*np.sqrt(variance)
    mask = np.zeros(length, dtype=np.int32)
    corrupt(rng, flags, flux, mask, variance, None, sky)
    covar = np.zeros((3, length))
    covar[0] = variance
    observations = Observations(visit=np.array([visit]), arm=[arm], spectrograph=np.array([1]),
                                pfsDesignId=np.array([0]), fiberId=np.array([7]),
                                pfiNominal=np.zeros((1, 2)), pfiCenter=np.zeros((1, 2)))
    return PfsSingle(Target(0, 0, "0,0", 0), observations, wavelength, flux, mask, sky, covar,
                     np.zeros((1, 1)), flags)


class CombineSpectraTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.flags = makeFlags()
        self.fiberId = [3, 7, 11]
        self.arms = [makeArm(self.rng, arm, self.fiberId, self.flags) for arm in ARMS]
        self.wavelength = np.linspace(380.0, 1260.0, 12000)

    def assertSame(self, live, reference, names):
        for name in names:
            np.testing.assert_array_equal(getattr(live, name) if not isinstance(live, dict) else live[name],
                                          reference[name], err_msg=name)

    def testResampleSet(self):
        """PfsFiberArraySet.resample is unchanged"""
        for order in (1, 3):
            for arm in self.arms:
                live = arm.resample(self.wavelength, order=order, bad=["NO_DATA"], minWeight=0.5)
                ref = _referenceResampleSet(arm, self.wavelength, order=order, bad=["NO_DATA"], minWeight=0.5)
                for name in ("flux", "mask", "sky", "norm", "covar"):
                    np.testing.assert_array_equal(getattr(live, name), getattr(ref, name),
                                                  err_msg=f"{name} order={order}")

    def testMergeArms(self):
        """The mergeArms combination (resample + combineSpectraSets) is unchanged"""
        for order in (1, 3):
            for suspect in (["SUSPECT"], None):
                resampled = [arm.resample(self.wavelength, order=order, bad=["NO_DATA"], minWeight=0.5)
                             for arm in self.arms]
                refResampled = [_referenceResampleSet(arm, self.wavelength, order=order, bad=["NO_DATA"],
                                                      minWeight=0.5) for arm in self.arms]
                live = combineSpectraSets(resampled, self.flags, ["NO_DATA"], suspect)
                ref = _referenceCombineSpectraSets(refResampled, self.flags, ["NO_DATA"], suspect)
                self.assertSame(live, ref, ("flux", "sky", "norm", "covar", "mask"))
                self.assertGreater(np.sum(np.isfinite(ref["flux"]) & (ref["flux"] != 0)), 1000)

    def testCoadd(self):
        """The CoaddSpectraTask.combine flux combination is unchanged"""
        singles = [makeSingle(self.rng, visit, arm, self.flags) for visit in (1, 2, 3) for arm in ARMS]
        lsfList = [{7: GaussianLsf(len(ss.wavelength), 1.5)} for ss in singles]
        for order in (1, 3):
            config = CoaddSpectraConfig()
            config.resampleOrder = order
            task = CoaddSpectraTask(config=config)
            live = task.combine(singles, lsfList, self.flags, self.wavelength)
            ref = _referenceCoaddCombine(config, singles, self.flags, self.wavelength)
            self.assertSame(live, ref, ("flux", "sky", "covar", "mask"))
            self.assertGreater(np.sum(np.isfinite(ref["flux"]) & (ref["flux"] != 0)), 1000)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
