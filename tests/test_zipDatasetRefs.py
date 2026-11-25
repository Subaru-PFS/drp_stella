"""Claude: zipDatasetRefs iterates in a reproducible (sorted) order, also over a single list"""
import unittest

import numpy as np

import lsst.utils.tests
from lsst.daf.butler import DataCoordinate, DatasetRef, DatasetType, DimensionUniverse
from pfs.drp.stella.gen3 import DatasetRefList, zipDatasetRefs


class ZipDatasetRefsTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.universe = DimensionUniverse()
        self.rng = np.random.default_rng(2468)

    def makeRefs(self, name, dimensions, values):
        """DatasetRefs for the given data coordinate values, in the given order"""
        datasetType = DatasetType(name, dimensions, "StructuredDataDict", universe=self.universe)
        refs = []
        for vv in values:
            dataId = DataCoordinate.standardize(dict(zip(dimensions, vv)), universe=self.universe)
            refs.append(DatasetRef(datasetType, dataId, run="test"))
        return DatasetRefList.fromList(refs)

    def testOrder(self):
        names = ("instrument", "visit", "detector")
        values = [("HSC", visit, detector) for visit in (3, 1, 2) for detector in (5, 4)]
        # Sorted in the order of the universe's required dimensions (which needn't be ``names``)
        required = list(next(iter(self.makeRefs("probe", names, values[:1]))).dataId.dimensions.required)
        expect = sorted(values, key=lambda vv: tuple(vv[names.index(dd)] for dd in required))
        for _ in range(3):
            self.rng.shuffle(values)
            # Single list
            refs = self.makeRefs("one", names, values)
            got = [(ref.dataId["instrument"], ref.dataId["visit"], ref.dataId["detector"])
                   for ref, in zipDatasetRefs(refs)]
            self.assertEqual(got, expect)
            # Two lists, one with fewer dimensions
            perVisit = self.makeRefs("two", ("instrument", "visit"), [("HSC", vv) for vv in (2, 3, 1)])
            got = [(aa.dataId["visit"], aa.dataId["detector"], bb.dataId["visit"])
                   for aa, bb in zipDatasetRefs(refs, perVisit)]
            self.assertEqual(got, [(vv[1], vv[2], vv[1]) for vv in expect])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
