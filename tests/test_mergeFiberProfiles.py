from typing import Iterable

import numpy as np

import lsst.utils.tests

from pfs.datamodel import CalibIdentity, PfsConfig
from pfs.drp.stella.fiberProfile import FiberProfile
from pfs.drp.stella.fiberProfileSet import FiberProfileSet
from pfs.drp.stella.mergeFiberProfiles import MergeFiberProfilesTask
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticPfsConfig
from pfs.drp.stella.tests.utils import runTests


class MergeFiberProfilesTestCase(lsst.utils.tests.TestCase):
    """Tests for MergeFiberProfilesTask

    We simulate what ``ReduceProfilesTask`` hands to the merge: several groups
    of quartz exposures, each illuminating every ``numGroups``-th fiber, plus a
    broken cobra that is lit in every exposure (darks included). The broken
    cobra is removed from each profile set, because it's lit in the darks, so
    its profile must be replaced by the merge.
    """

    def setUp(self):
        self.numGroups = (
            4  # Number of quartz groups; > fiberInfluence so lit fibers don't influence each other
        )
        self.spectrograph = 1
        self.brokenCobra = 115  # Lit in every exposure, darks included
        self.sigma = 1.2  # Gaussian sigma of the profiles (pixels)
        self.radius = 10  # Profile half-width (pixels)
        self.oversample = 5  # Profile oversampling factor
        self.height = 4000  # Detector height (pixels)
        self.wingRadius = (
            4  # Distance from the center at which light is considered to be in the wings (pixels)
        )
        self.plateauOffset = -6.5  # Position of residual light in the contaminated profile (pixels)
        self.plateauSigma = 1.5  # Width of residual light in the contaminated profile (pixels)
        self.plateauAmplitude = 0.15  # Amplitude of residual light, relative to the profile peak

        synth = SyntheticConfig()
        synth.separation = 10.0
        synth.slope = 0.0
        synth.width = 340
        rng = np.random.RandomState(12345)
        base = makeSyntheticPfsConfig(synth, 123456789, 54321, rng, fracSky=0.0, fracFluxStd=0.0)
        self.fiberId = 100 + np.arange(len(base), dtype=int)
        self.pfsConfig = base.copy(fiberId=self.fiberId)
        assert np.all(self.pfsConfig.spectrograph == self.spectrograph)

        self.identity = CalibIdentity("2026-11-25", 5, "r1", 12345)
        self.clean = FiberProfile.makeGaussian(self.sigma, self.height, self.radius, self.oversample)

    def makeConfig(self, lit: Iterable[int]) -> PfsConfig:
        """Make a pfsConfig in which only the nominated fibers are exposed

        Fibers that are not exposed have NaN positions (hidden).

        Parameters
        ----------
        lit : iterable of `int`
            Fiber identifiers that are exposed.

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

    def makeContaminated(self) -> FiberProfile:
        """Make a profile contaminated with light from a neighboring fiber

        This simulates the residual light from a broken cobra that was not
        removed by the dark subtraction (PIPE2D-2034).

        Returns
        -------
        profile : `FiberProfile`
            Contaminated profile.
        """
        profile = self.clean.copy()
        peak = profile.profiles.max()
        plateau = np.exp(-0.5 * ((profile.index - self.plateauOffset) / self.plateauSigma) ** 2)
        profile.profiles = profile.profiles + self.plateauAmplitude * peak * plateau[np.newaxis, :]
        return profile

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

    def runMerge(self, contaminated: Iterable[int], useDarkFibers: bool = False) -> FiberProfileSet:
        """Run the merge on simulated profiles

        Parameters
        ----------
        contaminated : iterable of `int`
            Fiber identifiers whose measured profiles are contaminated with
            light from the broken cobra.
        useDarkFibers : `bool`, optional
            Provide the broken cobra as measured fibers illuminated in the
            darks (as done by ``ReduceProfilesTask``), and make the darks'
            pfsConfig claim that no fibers are exposed?

        Returns
        -------
        merged : `FiberProfileSet`
            Merged fiber profiles.
        """
        contaminated = set(contaminated)
        profiles = []
        brightConfigs = []
        darkConfigs = []
        for group in range(self.numGroups):
            lit = set(self.fiberId[self.fiberId % self.numGroups == group].tolist())
            brightConfigs.append([self.makeConfig(lit | {self.brokenCobra})])
            darkConfigs.append([self.makeConfig(set() if useDarkFibers else {self.brokenCobra})])

            # ReduceProfilesTask removes the fibers that are lit in the darks
            profileSet = FiberProfileSet.makeEmpty(self.identity)
            for ff in lit - {self.brokenCobra}:
                profileSet[ff] = self.makeContaminated() if ff in contaminated else self.clean.copy()
            profiles.append(profileSet)

        task = MergeFiberProfilesTask(config=MergeFiberProfilesTask.ConfigClass())
        darkFibers = [{self.brokenCobra}] * self.numGroups if useDarkFibers else None
        return task.run(profiles, brightConfigs, darkConfigs, self.spectrograph, darkFibers=darkFibers).merged

    def assertReplacementClean(self, merged: FiberProfileSet):
        """Assert that the broken cobra's replacement profile is clean"""
        self.assertIn(self.brokenCobra, merged)
        cleanWings = self.getWingFraction(self.clean)
        replacedWings = self.getWingFraction(merged[self.brokenCobra])
        self.assertLess(cleanWings, 2.0e-3)
        self.assertLess(replacedWings - cleanWings, 0.01)

    def testNoContamination(self):
        """Control: with clean donors, the replacement is clean"""
        merged = self.runMerge([])
        self.assertEqual(set(merged), set(self.fiberId.tolist()))
        self.assertReplacementClean(merged)

    def testOneContaminatedDonor(self):
        """A single contaminated neighbor shouldn't contaminate the replacement

        This is the configuration seen on r4 in run31 (e.g., fiber 2208 is
        replaced using 2209, which is contaminated by 2208's own light).
        """
        merged = self.runMerge([self.brokenCobra + 1])
        self.assertGreater(self.getWingFraction(self.makeContaminated()), 0.1)
        self.assertReplacementClean(merged)

    def testTwoContaminatedDonors(self):
        """Contaminated neighbors on both sides shouldn't contaminate the replacement

        The broken cobra's residual light can land in the profiles of the
        neighbors on both sides, so two of the three donors are contaminated.
        """
        merged = self.runMerge([self.brokenCobra - 1, self.brokenCobra + 1])
        self.assertReplacementClean(merged)

    def testDarkFibers(self):
        """The fibers exposed in the darks measured by ReduceProfilesTask

        These override the pfsConfig.
        """
        merged = self.runMerge([self.brokenCobra - 1, self.brokenCobra + 1], useDarkFibers=True)
        self.assertReplacementClean(merged)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
