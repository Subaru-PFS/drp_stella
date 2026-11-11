import numpy as np
import pandas as pd

import lsst.utils.tests

from pfs.drp.stella.referenceLine import ReferenceLineSet, ReferenceLineStatus
from pfs.drp.stella.lineLists.broad import (lineFamilies, flagBroadFamilies, familyWidthExcess,
                                            familyShifts, C_GHZ_NM)
from pfs.drp.stella.tests.utils import runTests

display = None
BROAD = int(ReferenceLineStatus.BROAD)
S2, S4, OTHER = "4s_2[1/2]*_1", "4s_2[3/2]*_1", "4p_2[1/2]_1"


def makeLineList(rows):
    """Write and read back a line list from (wavelength, description, status, transition)"""
    with lsst.utils.tests.getTempFilePath(".txt") as filename:
        with open(filename, "w") as fd:
            for wl, desc, status, tr in rows:
                fd.write(f"{wl:.5f} 100 {desc} {status} {tr} 2\n")
        return ReferenceLineSet.fromLineList(filename)


class BroadFamiliesTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def testLineFamilies(self):
        """Families from the lower level of any component, per species"""
        descriptions = ["ArI", "ArI", "ArI", "KrI", "ArII", "HgI", "ArI"]
        transitions = [f"{S2}->4p_2[1/2]_0", f"{OTHER}->5d_2[1/2]*_1;{S4}->4p_2[3/2]_2", "UNKNOWN",
                       "5s_2[3/2]*_1->5p_2[3/2]_2", f"{S2}->4p_2[1/2]_0", "6p_3P*_1->7s_3S_1",
                       f"{OTHER}->5d_2[1/2]*_1"]
        np.testing.assert_array_equal(lineFamilies(descriptions, transitions),
                                      ["1s2", "1s4", "", "1s4", "", "6p3P1", ""])

    def testFlagBroadFamilies(self):
        """BROAD set on family members, cleared on known non-members, kept on UNKNOWN lines"""
        lines = makeLineList([(700.0, "ArI", 0, f"{S2}->4p_2[1/2]_0"),
                              (710.0, "ArI", BROAD, f"{S4}->4p_2[3/2]_2"),
                              (720.0, "ArI", BROAD, "UNKNOWN"),
                              (730.0, "ArI", 0, f"{OTHER}->5d_2[1/2]*_1"),
                              (740.0, "ArI", 1, f"{S2}->4p_2[3/2]_1")])   # other status bits are kept
        newLines, counts = flagBroadFamilies(lines, ["1s2"])
        status = {round(ll.wavelength, 3): ll.status for ll in newLines.rows}
        self.assertEqual(status, {700.0: BROAD, 710.0: 0, 720.0: BROAD, 730.0: 0, 740.0: 1 | BROAD})
        self.assertEqual((counts["before"], counts["after"], counts["set"], counts["cleared"]), (2, 3, 2, 1))
        np.testing.assert_array_equal(counts["clearedWavelengths"], [710.0])

        # Lower-level tokens work too, and the result is idempotent
        again, counts = flagBroadFamilies(newLines, [S2])
        self.assertEqual((counts["set"], counts["cleared"]), (0, 0))
        np.testing.assert_array_equal(again.data.status, newLines.data.status)

    def testFamilyWidthExcess(self):
        """A broadened family is detected against brightness-matched lines; faint lines are ignored"""
        rows = []
        dispersion = 0.08
        excessGHz = 5.0
        lam = np.linspace(700, 800, 12)
        for ii, wl in enumerate(lam):
            family = ii < 4
            transition = f"{S2}->4p_2[1/2]_0" if family else f"{OTHER}->5d_2[1/2]*_1"
            peak = 1e4 if ii < 10 else 10.0   # two faint lines: their Lorentzian is larger
            width = 0.2 + (excessGHz/(C_GHZ_NM*dispersion/wl**2) if family else 0.0)
            width += 0.5 if peak < 100 else 0.0
            for fiberId in range(10):
                rows.append(dict(lamp="Ar", arm="r", description="ArI", transition=transition, wavelength=wl,
                                 lorentz=width + 0.002*self.rng.normal(), peak=peak, dispersion=dispersion))
        excess = familyWidthExcess(pd.DataFrame(rows))
        self.assertEqual(len(excess), 1)
        row = excess.iloc[0]
        self.assertEqual((row.family, row.nFamily, row.nReference), ("1s2", 4, 6))
        self.assertFloatsAlmostEqual(row.excessGHz, excessGHz, atol=0.1)
        self.assertTrue(row.broad)

        smaller = familyWidthExcess(pd.DataFrame(rows), minGHz=10.0)   # the GHz threshold applies
        self.assertFalse(smaller.iloc[0].broad)

    def testFamilyShifts(self):
        """Family shifts relative to the same lamp's lines, with per-fibre offsets removed"""
        rows = []
        shift = -0.015
        for visit, lamp, offset in ((1, "Ne", 0.0), (2, "Ar", 0.01)):
            for ii, wl in enumerate(np.linspace(700, 800, 10)):
                family = lamp == "Ar" and ii < 3
                transition = f"{S2}->4p_2[1/2]_0" if family else f"{OTHER}->5d_2[1/2]*_1"
                for fiberId in range(30):
                    fibre = 0.1*np.sin(fiberId)   # common to all lamps
                    rows.append(dict(file="r1", visit=visit, fiberId=fiberId, wavelength=wl,
                                     description=lamp + "I", transition=transition, flag=0,
                                     pixel=1000.0 + ii, refPixel=1000.0 + ii,
                                     dataCentroid=fibre + offset + (shift if family else 0.0)
                                     + 1e-3*self.rng.normal(), dispersion=0.08))
        rows[0]["flag"] = 0x1   # saturated: ignored
        rows[0]["dataCentroid"] = 100.0
        offsets, shifts = familyShifts(pd.DataFrame(rows), {1: "Ne", 2: "Ar"})
        byLamp = offsets.set_index("lamp")
        self.assertFloatsAlmostEqual(byLamp.loc["Ne", "offset"], 0.0, atol=1e-12)
        self.assertFloatsAlmostEqual(byLamp.loc["Ar", "offset"], 0.01, atol=1e-3)
        self.assertEqual(len(shifts), 1)
        self.assertEqual((shifts.lamp[0], shifts.family[0], shifts.num[0]), ("Ar", "1s2", 3))
        self.assertFloatsAlmostEqual(shifts["shift"][0], shift, atol=1e-3)
        self.assertFloatsAlmostEqual(shifts.shiftPm[0], 1e3*0.08*shifts["shift"][0], rtol=1e-12)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
