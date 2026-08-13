import lsst.afw.image as afwImage
import lsst.utils.tests

from lsst.obs.pfs.maskPlanes import CALIB_MASK_PLANES, ISR_MASK_PLANES

import pfs.drp.stella  # noqa: F401; the import under test -- it claims the planes

from pfs.drp.stella.tests import runTests


class MaskPlaneRegistrationTestCase(lsst.utils.tests.TestCase):
    """Importing this package claims the whole PFS mask plane set.

    `SpectrumSet.toPfsArm` freezes the process mask plane dictionary into the
    pfsArm it writes, and `MaskHelper` compares those numbers between files
    without remapping them. Every process that can write a pfsArm imports this
    package, so claiming the planes as it imports is what gets those processes to
    agree -- whether or not they run ISR, and whichever arm they are reducing.
    """

    def testPlanesAreClaimedOnImport(self):
        planes = afwImage.Mask().getMaskPlaneDict()
        for name in CALIB_MASK_PLANES + ISR_MASK_PLANES:
            self.assertIn(name, planes)

    def testPartlyVignettedKeepsItsBit(self):
        """Bit 21, following this package's own ten planes at 11-20.

        Every pfsArm and combined calib already written carries it there, so a
        change here stops new files merging with old ones.
        """
        self.assertEqual(afwImage.Mask().getMaskPlaneDict()["PARTLY_VIGNETTED"], 21)

    def testNoTwoPlanesShareABit(self):
        """`MaskHelper.fromMerge` builds a helper with two names on one bit
        without complaining, so the dictionary handed to it must not have one.
        """
        planes = afwImage.Mask().getMaskPlaneDict()
        self.assertEqual(len(set(planes.values())), len(planes))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
