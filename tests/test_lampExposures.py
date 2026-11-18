import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

import lsst.utils.tests

from pfs.datamodel import TargetType
from pfs.drp.stella.lineLists.lamps import (hasLamp, lampMismatches, lampSeries, integratedLineFluxes,
                                            warmupGrowth, warmupSummary, HE_LINES, heLineRates, heLampLevels)
from pfs.drp.stella.tests.utils import runTests

display = None


class FakeButler:
    """Just enough of a butler for lampSeries: visit records with timespans"""
    def __init__(self, starts):
        records = [SimpleNamespace(id=visit, timespan=SimpleNamespace(begin=SimpleNamespace(datetime=start)))
                   for visit, start in starts.items()]
        self.registry = SimpleNamespace(queryDimensionRecords=lambda *args, **kwargs: records)


class FakeHeButler:
    """Just enough of a butler for heLineRates: a pfsConfig and r-arm pfsArms"""
    sat = 4

    def __init__(self, rates, expTime=60.0, saturated=()):
        """``rates``: {visit: He I 706.72 counts/s}; fibres 1-3 are science, 4-12 engineering"""
        self.rates = rates
        self.expTime = expTime
        self.saturated = saturated
        self.fiberId = np.arange(1, 13)
        self.pfsConfig = SimpleNamespace(fiberId=self.fiberId,
                                         targetType=np.where(self.fiberId < 4, TargetType.SCIENCE,
                                                             TargetType.ENGINEERING))

    def get(self, datasetType, visit, arm=None, spectrograph=None):
        if datasetType == "pfsConfig":
            return self.pfsConfig
        if arm != "r" or spectrograph != 1 or visit not in self.rates:
            raise LookupError(datasetType)
        wavelength = np.tile(np.arange(660.0, 860.0, 0.02), (len(self.fiberId), 1))
        flux = np.full(wavelength.shape, 10.0)
        mask = np.zeros(wavelength.shape, dtype=np.int32)
        sigma = 0.1
        for wl in HE_LINES["r"]:
            total = self.rates[visit]*self.expTime*(0.01 if wl > 770 else 1.0)
            flux += total*0.02/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*((wavelength - wl)/sigma)**2)
        flux[self.fiberId < 4] *= 100   # science fibres: must be ignored
        for index in self.saturated:
            mask[index, np.abs(wavelength[index] - 706.72) < 0.05] = self.sat
        return SimpleNamespace(fiberId=self.fiberId, wavelength=wavelength, flux=flux, mask=mask,
                               flags={"SAT": self.sat}, identity=SimpleNamespace(expTime=self.expTime))


class LampExposuresTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def testHasLamp(self):
        """Select by header lamps, excluding quartz; mismatches with the registry"""
        scan = pd.DataFrame(dict(visit=[1, 2, 3, 4, 5],
                                 registry=["HgCd", "HgCd", "Ne", "HgCd,Quartz", "Ar"],
                                 header=["HgCd", "Ne_eng", "HgCd,Ne", "HgCd,Quartz_eng", "ERROR KeyError"]))
        np.testing.assert_array_equal(hasLamp(scan, "HgCd"), [True, False, True, False, False])
        np.testing.assert_array_equal(hasLamp(scan, "Ne_eng"), [False, True, False, False, False])
        np.testing.assert_array_equal(lampMismatches(scan).visit, [2, 3, 4])

    def testLampSeries(self):
        """A gap of more than 10 minutes starts a new series"""
        t0 = datetime.datetime(2026, 5, 5, 10, 0, 0)
        minutes = {11: 0, 12: 2, 13: 4, 21: 30, 22: 32, 31: 60}
        butler = FakeButler({visit: t0 + datetime.timedelta(minutes=mm) for visit, mm in minutes.items()})
        series = lampSeries(butler, [22, 11, 13, 12, 31, 21])
        np.testing.assert_array_equal(series.visit, [11, 12, 13, 21, 22, 31])
        np.testing.assert_array_equal(series.series, [1, 1, 1, 2, 2, 3])
        np.testing.assert_array_equal(series.idx, [0, 1, 2, 0, 1, 0])
        np.testing.assert_array_equal(series.tSeries, [0, 120, 240, 0, 120, 0])

    def testIntegratedLineFluxes(self):
        """Sum over the core minus the sideband continuum; masked or missing lines are NaN"""
        wavelength = np.arange(430.0, 440.0, 0.01)
        flux = np.full_like(wavelength, 5.0)
        flux += 100*np.exp(-0.5*((wavelength - 436.0)/0.1)**2)
        flux += 100*np.exp(-0.5*((wavelength - 433.0)/0.1)**2)
        mask = np.zeros(wavelength.size, dtype=np.int32)
        mask[np.abs(wavelength - 433.0) < 0.05] = 4
        fluxes = integratedLineFluxes(wavelength, flux, mask, 4, (436.0, 433.0, 450.0))
        self.assertFloatsAlmostEqual(fluxes[436.0], 100*0.1*np.sqrt(2*np.pi)/0.01, rtol=1e-3)
        self.assertTrue(np.isnan(fluxes[433.0]))
        self.assertTrue(np.isnan(fluxes[450.0]))

    def testWarmupGrowth(self):
        """Growth relative to each series' first exposure, averaged over spectrographs, by epoch"""
        seriesRows, ratioRows = [], []
        visit = 0
        for series, day, growth in ((1, 20260505, 1.2), (2, 20260506, 1.2), (3, 20260708, 1.4),
                                    (4, 20260709, 1.4), (5, 20260710, 1.4)):
            for idx in range(3):
                visit += 1
                seriesRows.append(dict(visit=visit, series=series, idx=idx, tSeries=120.0*idx, day_obs=day))
                for spectrograph in (1, 2, 3, 4):
                    ratio = 0.3*(1 + (growth - 1)*idx/2)*(1 + 1e-3*self.rng.normal())
                    ratioRows.append({"visit": visit, "spectrograph": spectrograph, "509/436": ratio,
                                      "405/436": 0.5, "I436": 1000.0})
        ratioRows.append({"visit": visit + 1, "spectrograph": 1, "509/436": 0.3, "405/436": 0.5, "I436": 1.0})
        seriesRows.append(dict(visit=visit + 1, series=6, idx=0, tSeries=0.0, day_obs=20260711))  # unlit
        growth = warmupGrowth(pd.DataFrame(seriesRows), pd.DataFrame(ratioRows))
        self.assertEqual(set(growth.series), {1, 2, 3, 4, 5})
        self.assertTrue(growth.clean.all())
        self.assertFloatsAlmostEqual(growth[growth.idx == 0].rel.to_numpy(), 1.0, atol=1e-12)
        summary = warmupSummary(growth).set_index(["epochStart", "idx"])
        self.assertFloatsAlmostEqual(summary.loc[(0, 2), "growth"], 1.2, atol=3e-3)
        self.assertFloatsAlmostEqual(summary.loc[(20260701, 2), "growth"], 1.4, atol=3e-3)
        self.assertFloatsAlmostEqual(summary.loc[(20260701, 1), "growth"], 1.2, atol=3e-3)
        self.assertEqual(summary.loc[(20260701, 2), "num"], 3)

    def testHeLineRates(self):
        """Rates on the engineering fibres only, per second; saturated fibres rejected"""
        butler = FakeHeButler({1: 800.0, 2: 400.0}, saturated=(5, 6))
        rates = heLineRates(butler, [1, 2, 3])
        self.assertEqual(len(rates), 2)
        np.testing.assert_array_equal(rates.numFibers, 9)
        self.assertFloatsAlmostEqual(rates["706.72"].to_numpy(), [800.0, 400.0], rtol=1e-3)
        self.assertFloatsAlmostEqual((rates["777.41"]/rates["706.72"]).to_numpy(), 0.01, rtol=1e-3)
        rates = heLineRates(FakeHeButler({1: 800.0}, saturated=(4, 5, 6, 7, 8)), [1])  # 4 good fibres left
        self.assertTrue(np.isnan(rates["706.72"].iloc[0]))

    def testHeLampLevels(self):
        """Exposures relative to the lamp-wide and each series' settled level, per spectrograph"""
        seriesRows, rateRows = [], []
        throughput = {1: 1.0, 2: 0.8, 3: 1.2, 4: 0.9}
        visit = 0
        for series, night, firstFactor, num in ((1, 1.0, 0.6, 3), (2, 1.04, 1.02, 2), (3, 0.96, 0.9, 1)):
            for idx in range(num):
                visit += 1
                seriesRows.append(dict(visit=visit, series=series, idx=idx, tSeries=240.0*idx,
                                       day_obs=20260500 + series))
                factor = night*(firstFactor if idx == 0 else 1.0)
                oxygen = 0.008*(2 if idx == 0 else 1)
                for spectrograph, through in throughput.items():
                    rate = 750*factor*through*(1 + 1e-4*self.rng.normal())
                    rateRows.append({"visit": visit, "arm": "r", "spectrograph": spectrograph,
                                     "expTime": 60.0, "706.72": rate, "777.41": oxygen*rate})
                    rateRows.append({"visit": visit, "arm": "b", "spectrograph": spectrograph,
                                     "expTime": 60.0, "587.73": 2*rate})
        levels = heLampLevels(pd.DataFrame(seriesRows), pd.DataFrame(rateRows)).set_index("visit")
        # settled exposures: visits 2, 3 (night 1.0) and 5 (1.04); lamp-wide level 1.0 per spectrograph
        self.assertFloatsAlmostEqual(levels.loc[[2, 3, 5], "706.72_norm"].to_numpy(), [1.0, 1.0, 1.04],
                                     rtol=1e-3)
        self.assertFloatsAlmostEqual(levels.loc[[1, 4, 6], "706.72_norm"].to_numpy(),
                                     [0.6, 1.04*1.02, 0.96*0.9], rtol=1e-3)
        self.assertFloatsAlmostEqual(levels.loc[[1, 2, 4, 5], "706.72_own"].to_numpy(), [0.6, 1.0, 1.02, 1.0],
                                     rtol=1e-3)
        self.assertTrue(np.isnan(levels.loc[6, "706.72_own"]))   # single exposure: no settled level
        self.assertFloatsAlmostEqual(levels.loc[[1, 2], "OI777/He707_own"].to_numpy(), [2.0, 1.0], rtol=1e-3)
        self.assertFloatsAlmostEqual(levels.loc[1, "587.73_norm"], 0.6, rtol=1e-3)  # combined over arms
        self.assertFloatsAlmostEqual(levels.loc[[1, 2, 3], "t"].to_numpy(), [0.0, 4.0, 8.0])
        self.assertLess(levels.loc[1, "706.72_normErr"], 1e-3)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
