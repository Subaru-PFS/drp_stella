from typing import Dict, Iterable, Optional

import numpy as np

import lsst.utils.tests
import lsst.afw.image
from lsst.afw.image import Exposure

from pfs.datamodel import CalibIdentity, PfsConfig
from pfs.drp.stella.fiberProfile import FiberProfile
from pfs.drp.stella.reduceProfiles import ReduceProfilesTask
from pfs.drp.stella.synthetic import (
    SyntheticConfig,
    addNoiseToImage,
    makeSpectrumImage,
    makeSyntheticDetectorMap,
    makeSyntheticPfsConfig,
)
from pfs.drp.stella.tests.utils import runTests


class ReduceProfilesDarkTestCase(lsst.utils.tests.TestCase):
    """Tests of the dark subtraction in ReduceProfilesTask

    We simulate a quartz exposure in which every ``numGroups``-th fiber is
    illuminated, together with some broken cobras that can't be hidden, and a
    dark exposure in which only the broken cobras are illuminated. Each broken
    cobra's neighbor on one side is illuminated in the quartz, so if the
    broken cobra's light is not completely subtracted by the dark, it ends up
    in the wing of that neighbor's profile (PIPE2D-2034).
    """

    def setUp(self):
        self.synth = SyntheticConfig()
        self.synth.height = 1024
        self.synth.width = 300
        self.synth.separation = 8.0
        self.synth.fwhm = 3.21
        self.synth.slope = 0.04  # Sample sub-pixel positions
        self.flux = 1.0e4  # Flux per row in each illuminated fiber (electrons)
        self.numGroups = 3  # Illuminate every numGroups-th fiber
        self.wingRadius = 4  # Distance at which light is considered to be in the wings (pixels)
        self.fluxRadius = 3  # Half-width of window for measuring a fiber's flux (pixels)

        self.rng = np.random.RandomState(12345)
        self.fiberId = self.synth.fiberId
        self.detectorMap = makeSyntheticDetectorMap(self.synth)
        self.pfsConfig = makeSyntheticPfsConfig(
            self.synth, 123456789, 54321, self.rng, fracSky=0, fracFluxStd=0
        )
        self.identity = CalibIdentity("2026-11-25", 1, "r", 12345)

        index = np.arange(len(self.fiberId))
        self.lit = set(self.fiberId[index % self.numGroups == 0].tolist())  # Illuminated in the quartz

        # Broken cobras can't be hidden, so are always illuminated. Each has an illuminated neighbor.
        brokenIndex = np.array([5, 20, 26])
        self.brokenCobras = set(self.fiberId[brokenIndex].tolist())
        self.neighbors = set(self.fiberId[brokenIndex + 1].tolist())
        assert not self.brokenCobras & self.lit
        assert self.neighbors <= self.lit

        # Fibers illuminated in the quartz, far from any broken cobra
        self.other = int(self.fiberId[12])  # Reference for a clean profile
        self.extra = int(self.fiberId[15])  # Used to simulate problems with the darks
        assert {self.other, self.extra} <= self.lit

        self.config = ReduceProfilesTask.ConfigClass()
        self.config.profiles.profileRadius = 10
        self.config.profiles.profileSwath = 256
        self.config.profiles.pruneMinLength = int(0.9 * self.synth.height)
        self.config.profiles.rowFwhm = self.synth.fwhm
        self.config.profiles.columnFwhm = self.synth.fwhm
        self.config.profiles.doScatteredLight = False

    def makeExposure(self, flux: Dict[int, float]) -> Exposure:
        """Make an exposure with the nominated fibers illuminated

        Parameters
        ----------
        flux : `dict` [`int`: `float`]
            Flux per row for each illuminated fiber, indexed by fiberId.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            Exposure, with units of electrons.
        """
        image = lsst.afw.image.ImageF(self.synth.dims)
        centers = dict(zip(self.fiberId.tolist(), self.synth.traceCenters))
        for ff, ss in flux.items():
            image += makeSpectrumImage(
                ss, self.synth.dims, [centers[ff]], self.synth.traceOffset, self.synth.fwhm
            )
        addNoiseToImage(image, 1.0, self.synth.readnoise, self.rng)

        mask = lsst.afw.image.Mask(image.getBBox())
        mask.set(0)
        variance = lsst.afw.image.ImageF(image.getBBox())
        variance.array[:] = np.where(image.array > 0, image.array, 0) + self.synth.readnoise**2
        exposure = lsst.afw.image.makeExposure(lsst.afw.image.makeMaskedImage(image, mask, variance))
        exposure.getInfo().setVisitInfo(lsst.afw.image.VisitInfo(darkTime=1.0))
        return exposure

    def makeConfig(self, lit: Iterable[int]) -> PfsConfig:
        """Make a pfsConfig in which only the nominated fibers are exposed

        Parameters
        ----------
        lit : iterable of `int`
            Fiber identifiers that are exposed; the others are hidden (NaN
            positions).

        Returns
        -------
        pfsConfig : `PfsConfig`
            Top-end configuration.
        """
        hidden = ~np.isin(self.fiberId, list(lit))
        pfiNominal = self.pfsConfig.pfiNominal.copy()
        pfiCenter = self.pfsConfig.pfiCenter.copy()
        pfiNominal[hidden] = np.nan
        pfiCenter[hidden] = np.nan
        return self.pfsConfig.copy(pfiNominal=pfiNominal, pfiCenter=pfiCenter)

    def runReduce(
        self,
        darkFlux: Optional[Dict[int, float]] = None,
        darkConfigLit: Optional[Iterable[int]] = None,
        darkConfigOmit: Iterable[int] = (),
    ):
        """Run ReduceProfilesTask on simulated quartz and dark

        Parameters
        ----------
        darkFlux : `dict` [`int`: `float`], optional
            Flux per row for each fiber illuminated in the dark, in addition
            to the broken cobras.
        darkConfigLit : iterable of `int`, optional
            Fibers that the dark's pfsConfig claims are exposed, in addition
            to the broken cobras.
        darkConfigOmit : iterable of `int`, optional
            Fibers that the dark's pfsConfig claims are hidden, even though
            they are exposed.

        Returns
        -------
        profiles : `pfs.drp.stella.FiberProfileSet`
            Fiber profiles.
        darkFibers : `set` of `int`
            Fibers that the task found to be illuminated in the darks.
        quartz : `lsst.afw.image.Exposure`
            Quartz exposure, after dark subtraction.
        """
        darkFlux = {**{ff: self.flux for ff in self.brokenCobras}, **(darkFlux or {})}
        darkConfigLit = (self.brokenCobras | set(darkConfigLit or [])) - set(darkConfigOmit)
        quartzLit = self.lit | self.brokenCobras

        quartz = self.makeExposure({ff: self.flux for ff in quartzLit})
        dark = self.makeExposure(darkFlux)
        brightList = {1: quartz}
        darkList = {2: dark}
        pfsConfigList = {1: self.makeConfig(quartzLit), 2: self.makeConfig(darkConfigLit)}
        detectorMapList = {1: self.detectorMap, 2: self.detectorMap}

        task = ReduceProfilesTask(config=self.config)
        result = task.run(self.identity, brightList, darkList, pfsConfigList, detectorMapList)
        return result.fiberProfiles, set(result.darkFibers["fiberId"]), quartz

    def getWingFraction(self, profile: FiberProfile) -> float:
        """Return the fraction of a profile's light in the wings

        Parameters
        ----------
        profile : `FiberProfile`
            Profile to measure.

        Returns
        -------
        fraction : `float`
            Fraction of the light at ``wingRadius`` or more from the center.
        """
        wings = np.abs(profile.index) >= self.wingRadius
        return float(np.sum(profile.profiles[:, wings]) / np.sum(profile.profiles))

    def getFlux(self, exposure: Exposure, fiberId: int) -> float:
        """Measure the flux of a fiber in an exposure

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to measure.
        fiberId : `int`
            Fiber identifier.

        Returns
        -------
        flux : `float`
            Median over rows of the flux within ``fluxRadius`` of the fiber
            center.
        """
        xCenter = self.detectorMap.getXCenter(fiberId)
        xx = np.arange(self.synth.width)[np.newaxis, :]
        select = np.abs(xx - xCenter[:, np.newaxis]) <= self.fluxRadius
        return float(np.median(np.sum(np.where(select, exposure.image.array, 0.0), axis=1)))

    def assertDarkSubtracted(self, profiles, darkFibers, quartz):
        """Assert that the broken cobras' light was removed

        The broken cobras should subtract to zero in the quartz, so the
        profiles of their illuminated neighbors have no more light in their
        wings than any other profile. The broken cobras should be reported as
        being illuminated in the darks.
        """
        self.assertEqual(darkFibers, self.brokenCobras)
        reference = self.getWingFraction(profiles[self.other])
        self.assertLess(reference, 0.01)
        for ff in self.brokenCobras:
            self.assertNotIn(ff, profiles)
            residual = self.getFlux(quartz, ff) / self.flux
            self.assertLess(abs(residual), 0.02, f"Broken cobra {ff} not subtracted to zero")
        for ff in self.neighbors:
            excess = self.getWingFraction(profiles[ff]) - reference
            self.assertLess(excess, 0.01, f"Neighbor {ff} of broken cobra has excess wing light")

    def testControl(self):
        """Control: the pfsConfig for the dark is correct"""
        profiles, darkFibers, quartz = self.runReduce()
        self.assertEqual(set(profiles), self.lit)
        self.assertDarkSubtracted(profiles, darkFibers, quartz)

    def testPhantomDarkFiber(self):
        """A fiber hidden in the dark, but claimed by the pfsConfig to be exposed

        This is fiber 2007 on r4 in group 1 of run31: hidden in the dark (but
        with a finite ``pfiCenter``), and illuminated in the quartz.
        """
        profiles, darkFibers, quartz = self.runReduce(darkConfigLit=[self.extra])
        self.assertEqual(set(profiles), self.lit)
        self.assertDarkSubtracted(profiles, darkFibers, quartz)

    def testPartlyHiddenDarkFiber(self):
        """A fiber partly exposed in the dark, and illuminated in the quartz

        This is fiber 1398 on r3 in group 4 of run31: half-hidden in the dark,
        and illuminated in the quartz.
        """
        profiles, darkFibers, quartz = self.runReduce(
            darkFlux={self.extra: 0.5 * self.flux}, darkConfigLit=[self.extra]
        )
        self.assertEqual(set(profiles), self.lit)
        self.assertDarkSubtracted(profiles, darkFibers, quartz)
        excess = self.getWingFraction(profiles[self.extra]) - self.getWingFraction(profiles[self.other])
        self.assertLess(excess, 0.01)

    def testMissingDarkFiber(self):
        """A fiber exposed in the dark, but claimed by the pfsConfig to be hidden"""
        profiles, darkFibers, quartz = self.runReduce(darkConfigOmit=[min(self.brokenCobras)])
        self.assertEqual(set(profiles), self.lit)
        self.assertDarkSubtracted(profiles, darkFibers, quartz)

    def testHiddenInBright(self):
        """A fiber partly exposed in the dark, but hidden in the quartz

        This shouldn't bias the scaling of the dark.
        """
        hidden = int(self.fiberId[14])
        assert hidden not in self.lit
        profiles, darkFibers, quartz = self.runReduce(
            darkFlux={hidden: 0.5 * self.flux}, darkConfigLit=[hidden]
        )
        self.assertEqual(set(profiles), self.lit)
        self.assertDarkSubtracted(profiles, darkFibers, quartz)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
