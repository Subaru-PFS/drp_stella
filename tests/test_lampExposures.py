import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

import lsst.utils.tests

from pfs.drp.stella.lineLists.lamps import (hasLamp, lampMismatches, lampSeries, integratedLineFluxes,
                                            warmupGrowth, warmupSummary)
from pfs.drp.stella.tests.utils import runTests

display = None


class FakeButler:
    """Just enough of a butler for lampSeries: visit records with timespans"""
    def __init__(self, starts):
        records = [SimpleNamespace(id=visit, timespan=SimpleNamespace(begin=SimpleNamespace(datetime=start)))
                   for visit, start in starts.items()]
        self.registry = SimpleNamespace(queryDimensionRecords=lambda *args, **kwargs: records)


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


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
