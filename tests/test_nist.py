import os
import tempfile
import urllib.parse

import lsst.utils.tests

from pfs.drp.stella.referenceLine import ReferenceLineSet, ReferenceLineSource
from pfs.drp.stella.lineLists.nist import (parseSpecies, nistLinesUrl, loadNist, addTransitions,
                                           buildLineList, readLevels, findTransitions)
from pfs.drp.stella.tests.utils import runTests

display = None
NIST = int(ReferenceLineSource.NIST)
LINES_HEADER = ["obs_wl_vac(nm)", "unc_obs_wl", "ritz_wl_vac(nm)", "unc_ritz_wl", "intens", "Aki(s^-1)",
                "Acc", "Ei(cm-1)", "Ek(cm-1)", "conf_i", "term_i", "J_i", "conf_k", "term_k", "J_k"]


def writeTable(path, header, rows):
    """Write a table in NIST's format: tab-separated, quoted values"""
    with open(path, "w") as fd:
        fd.write("\t".join(header) + "\t\n")
        for row in rows:
            fd.write("\t".join(f'"{v}"' for v in row) + "\t\n")


def nistRow(obs, ritz, intens, A, lo, hi, unc=""):
    """A NIST lines row; lo/hi are (configuration, term, J)"""
    return [obs, unc, ritz, "", intens, A, "", "", "", lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]


class NistTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def path(self, name):
        return os.path.join(self.dir.name, name)

    def testSpeciesAndUrl(self):
        self.assertEqual(parseSpecies("ArI"), ("Ar", 1))
        self.assertEqual(parseSpecies("HgII"), ("Hg", 2))
        self.assertIsNone(parseSpecies("OH"))
        query = urllib.parse.parse_qs(urllib.parse.urlparse(nistLinesUrl("Cd", 1, 370, 1270)).query)
        self.assertEqual(query["spectra"], ["Cd I"])     # one species per query: NIST rejects "Cd I;Hg I"
        self.assertEqual(query["show_av"], ["3"])        # vacuum wavelengths
        self.assertEqual(query["format"], ["3"])         # tab-separated
        self.assertEqual(query["page_size"], ["15"])     # NIST rejects page_size=0

    def testAddTransitions(self):
        """Matching by species and wavelength: single lines, blends, unclassified, unmatched, skipped"""
        s, p = ("3s.3p", "3P*", "1"), ("3s.4s", "3S", "1")
        writeTable(self.path("nist_Ne.dat"), LINES_HEADER, [
            nistRow("600.0001+", "600.0000", "100", "1e6", ("2p5.3s", "2[3/2]*", "1"),
                    ("2p5.3p", "2[1/2]", "0")),
            nistRow("650.0000", "650.0000", "50", "1e6", s, p),            # a blend of two components
            nistRow("650.0000", "650.0001", "50", "1e6", s, ("3s.4s", "3S", "2")),
            nistRow("700.0000", "", "10", "", ("", "", ""), ("", "", "")),  # unclassified
        ])
        with lsst.utils.tests.getTempFilePath(".txt") as filename:
            with open(filename, "w") as fd:
                for wl, tr in ((600.0, "UNKNOWN"), (650.0, "UNKNOWN"), (700.0, "UNKNOWN"), (800.0, "UNKNOWN"),
                               (900.0, "already_known")):
                    fd.write(f"{wl:.5f} 1 NeI 0 {tr} 0\n")
                fd.write("610.00000 1 ArI 0 UNKNOWN 0\n")
            lines = ReferenceLineSet.fromLineList(filename)
        newLines, counts = addTransitions(lines, loadNist([self.path("nist_Ne.dat")]))
        self.assertEqual(counts, dict(matched=1, blended=1, unmatched=1, unclassified=1, noSpecies=1,
                                      skipped=1))
        byLam = {(ll.description, round(ll.wavelength, 3)): ll for ll in newLines.rows}
        self.assertEqual(byLam["NeI", 600.0].transition, "3s_2[3/2]*_1->3p_2[1/2]_0")
        self.assertEqual(byLam["NeI", 650.0].transition, "3p_3P*_1->4s_3S_1;3p_3P*_1->4s_3S_2")
        self.assertEqual(byLam["NeI", 700.0].transition, "UNKNOWN")
        self.assertEqual(byLam["NeI", 900.0].transition, "already_known")
        self.assertTrue(byLam["NeI", 600.0].source & NIST)
        self.assertFalse(byLam["NeI", 800.0].source & NIST)

    def testBuildLineList(self):
        """Unresolved components merge at the g*A-weighted Ritz wavelength; minFrac; unclassified lines"""
        lo, hi = ("1s.2p", "3P*"), ("1s.3d", "3D")
        writeTable(self.path("nist_He.dat"), LINES_HEADER, [
            nistRow("587.7249", "587.7240", "500", "1e7", lo + ("2",), hi + ("3",)),   # g=7
            nistRow("587.7249", "587.7270", "500", "2e7", lo + ("1",), hi + ("2",)),   # g=5
            nistRow("", "600.0000", "", "1e7", lo + ("1",), hi + ("1",)),              # no intensity: dropped
            nistRow("650.0000", "650.0000", "4", "1e6", lo + ("1",), hi + ("1",)),     # < minFrac
            nistRow("700.0000", "700.0000", "200", "", ("", "", ""), ("", "", "")),    # unclassified
        ])
        lines = buildLineList(self.path("nist_He.dat"), "HeI", minFrac=0.01)
        byLam = {round(ll.wavelength, 4): ll for ll in lines.rows}
        expected = (587.7240*7e7 + 587.7270*5*2e7)/(7e7 + 5*2e7)
        self.assertEqual(sorted(byLam), [round(expected, 4), 700.0])
        merged = byLam[round(expected, 4)]
        self.assertEqual(merged.transition, "2p_3P*_1/2->3d_3D_2/3")
        self.assertEqual(merged.intensity, 500.0)
        self.assertTrue(merged.source & NIST)
        self.assertEqual(byLam[700.0].transition, "UNKNOWN")

    def testFindTransitions(self):
        """Ritz wavelengths from energy levels, with the parity and J selection rules"""
        header = ["Configuration", "Term", "J", "Prefix", "Level (cm-1)", "Suffix"]
        writeTable(self.path("levels.dat"), header, [
            ["4d10.5s2", "1S", "0", "", "0.000", ""],
            ["4d10.5s.5p", "1P*", "1", "", "43692.384", ""],
            ["4d10.5s.6s", "1S", "0", "", "53310.101", ""],
            ["4d10.5s.6s", "3S", "1", "", "51483.980", ""],
        ])
        levels = readLevels(self.path("levels.dat"))
        found = findTransitions(levels, 1039.75, tolerance=0.5)
        self.assertEqual(len(found), 1)
        self.assertEqual(found.transition[0], "5p_1P*_1->6s_1S_0")
        self.assertFloatsAlmostEqual(found.ritz[0], 1e7/(53310.101 - 43692.384), rtol=1e-12)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
