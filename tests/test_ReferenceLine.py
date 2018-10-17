import sys
import pickle
import unittest

import lsst.utils.tests
from pfs.drp.stella import ReferenceLine
from pfs.drp.stella.tests import BaseTestCase


class ReferenceLineTestCase(BaseTestCase):
    def testBasics(self):
        """Test construction, getters, setters and arithmetic with ``Status``"""
        self.assertReferenceLine(self.line)

        # Check that enums can be OR-ed together
        self.status = 0
        for name in ("NOWT", "FIT", "RESERVED", "MISIDENTIFIED", "CLIPPED", "SATURATED",
                     "INTERPOLATED", "CR"):
            self.status |= getattr(ReferenceLine.Status, name)

        # Check that attributes set at construction time can also be set directly
        self.line.description = self.description = "reference"
        self.line.status = self.status
        self.line.wavelength = self.wavelength = 9876.54321
        self.line.guessedIntensity = self.guessedIntensity = 0.98765
        self.assertReferenceLine(self.line)

    def testPickle(self):
        """Test that ``ReferenceLine`` can be pickled"""
        copy = pickle.loads(pickle.dumps(self.line))
        self.assertReferenceLine(copy)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main()
