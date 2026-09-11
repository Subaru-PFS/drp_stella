# These tests exercise PsfMatchDiagnostic's event-handling logic by driving matplotlib's own event
# synthesis methods (button_press_event/motion_notify_event/button_release_event/key_press_event) under
# the non-interactive "Agg" backend. This validates the dispatch logic (click-vs-drag, hit-testing, stretch
# math, toolbar guard), but NOT the actual felt responsiveness under a real GUI event loop, nor genuine
# ipympl/Jupyter behaviour: those must be checked manually (see the "Notes" section of the plan/design that
# introduced this module).
import matplotlib

matplotlib.use("Agg")

from matplotlib.backend_bases import KeyEvent, MouseEvent  # noqa: E402
import numpy as np  # noqa: E402

import lsst.utils.tests  # noqa: E402

from pfs.drp.stella.utils.interactiveDisplay import PsfMatchDiagnostic  # noqa: E402
from pfs.drp.stella.tests.utils import runTests  # noqa: E402

display = None


# matplotlib >= 3.9 removed FigureCanvasBase's button_press_event/motion_notify_event/
# button_release_event/key_press_event convenience methods; the modern replacement is to construct the
# Event subclass directly and call its private _process() method, which is what these helpers do.
def pressButton(canvas, x, y, button):
    MouseEvent("button_press_event", canvas, x, y, button=button)._process()


def moveMouse(canvas, x, y):
    MouseEvent("motion_notify_event", canvas, x, y)._process()


def releaseButton(canvas, x, y, button):
    MouseEvent("button_release_event", canvas, x, y, button=button)._process()


def pressKey(canvas, key):
    KeyEvent("key_press_event", canvas, key)._process()


class PsfMatchDiagnosticTestCase(lsst.utils.tests.TestCase):
    """Test PsfMatchDiagnostic's construction and interactive event handling"""

    def setUp(self):
        rng = np.random.RandomState(12345)
        shape = (40, 50)
        self.source = 100.0 + 10.0 * rng.normal(size=shape)
        self.target = 100.0 + 10.0 * rng.normal(size=shape)
        self.convolved = 100.0 + 10.0 * rng.normal(size=shape)

        self.diagnostic = PsfMatchDiagnostic(self.source, self.target, self.convolved)
        self.diagnostic.fig.canvas.draw()

    def tearDown(self):
        import matplotlib.pyplot as plt

        self.diagnostic.close()
        plt.close(self.diagnostic.fig)
        del self.diagnostic

    def axisCenter(self, axis):
        """Return the (x, y) display-pixel coordinates of an axis's center"""
        bbox = axis.bbox
        return 0.5 * (bbox.x0 + bbox.x1), 0.5 * (bbox.y0 + bbox.y1)

    def testConstruction(self):
        """Panel layout, titles and auto-stretch match independent expectations"""
        self.assertEqual(self.diagnostic.axes.shape, (2, 2))
        titles = [axis.get_title() for axis in self.diagnostic.axes.ravel()]
        self.assertEqual(titles, ["source", "target", "convolved", "target - convolved"])

        allValues = np.concatenate([self.source.ravel(), self.target.ravel(), self.convolved.ravel()])
        expectedVmin = np.percentile(allValues, 0.5)
        expectedVmax = np.percentile(allValues, 99.5)
        shared = self.diagnostic._norms["shared"]
        self.assertAlmostEqual(shared.vmin, expectedVmin)
        self.assertAlmostEqual(shared.vmax, expectedVmax)

        diffValues = self.target - self.convolved
        expectedLimit = np.percentile(np.abs(diffValues), 99.5)
        diffNorm = self.diagnostic._norms["diff"]
        self.assertAlmostEqual(diffNorm.vmin, -expectedLimit)
        self.assertAlmostEqual(diffNorm.vmax, expectedLimit)

    def testAddMarkByClick(self):
        """A zero-displacement left click adds one mark, echoed in all 4 panels"""
        axis = self.diagnostic.axes[0, 0]
        x, y = self.axisCenter(axis)
        expectedXY = axis.transData.inverted().transform((x, y))

        pressButton(self.diagnostic.fig.canvas, x, y, 1)
        releaseButton(self.diagnostic.fig.canvas, x, y, 1)

        self.assertEqual(len(self.diagnostic.marks), 1)
        self.assertEqual(len(self.diagnostic.marks[0]["artists"]), 4)
        np.testing.assert_allclose(self.diagnostic.marks[0]["xy"], expectedXY)

    def testRemoveMarkByRightClick(self):
        """A zero-displacement right click removes the nearest mark"""
        axis = self.diagnostic.axes[0, 0]
        x, y = self.axisCenter(axis)

        pressButton(self.diagnostic.fig.canvas, x, y, 1)
        releaseButton(self.diagnostic.fig.canvas, x, y, 1)
        self.assertEqual(len(self.diagnostic.marks), 1)

        pressButton(self.diagnostic.fig.canvas, x, y, 3)
        releaseButton(self.diagnostic.fig.canvas, x, y, 3)
        self.assertEqual(len(self.diagnostic.marks), 0)

    def testRemoveMarkTooFarIsNoop(self):
        """A right click far from any mark does not remove it"""
        axis = self.diagnostic.axes[0, 0]
        x, y = self.axisCenter(axis)
        pressButton(self.diagnostic.fig.canvas, x, y, 1)
        releaseButton(self.diagnostic.fig.canvas, x, y, 1)
        self.assertEqual(len(self.diagnostic.marks), 1)

        self.diagnostic.removeNearestMark(x + 1000, y + 1000, axis)
        self.assertEqual(len(self.diagnostic.marks), 1)

    def testClickDragThreshold(self):
        """Small motion is still a click; motion past the threshold is a drag"""
        axis = self.diagnostic.axes[0, 0]
        x, y = self.axisCenter(axis)

        # Displacement below the (default 4px) threshold: still counts as a click.
        pressButton(self.diagnostic.fig.canvas, x, y, 1)
        moveMouse(self.diagnostic.fig.canvas, x + 1, y)
        releaseButton(self.diagnostic.fig.canvas, x + 1, y, 1)
        self.assertEqual(len(self.diagnostic.marks), 1)

        # Displacement past the threshold: a drag, so release must not add a second mark.
        pressButton(self.diagnostic.fig.canvas, x, y, 1)
        moveMouse(self.diagnostic.fig.canvas, x + 50, y)
        releaseButton(self.diagnostic.fig.canvas, x + 50, y, 1)
        self.assertEqual(len(self.diagnostic.marks), 1)

    def testDragAdjustsCorrectGroupOnly(self):
        """A drag in a shared-stretch panel never touches the difference stretch"""
        axis = self.diagnostic.axes[0, 0]
        x, y = self.axisCenter(axis)
        initShared = (self.diagnostic._norms["shared"].vmin, self.diagnostic._norms["shared"].vmax)
        initDiff = (self.diagnostic._norms["diff"].vmin, self.diagnostic._norms["diff"].vmax)

        pressButton(self.diagnostic.fig.canvas, x, y, 3)
        moveMouse(self.diagnostic.fig.canvas, x + 50, y + 30)
        releaseButton(self.diagnostic.fig.canvas, x + 50, y + 30, 3)

        newShared = (self.diagnostic._norms["shared"].vmin, self.diagnostic._norms["shared"].vmax)
        newDiff = (self.diagnostic._norms["diff"].vmin, self.diagnostic._norms["diff"].vmax)
        self.assertNotEqual(initShared, newShared)
        self.assertEqual(initDiff, newDiff)

    def testDragDirection(self):
        """Dragging right brightens (lowers the center); dragging up increases contrast"""
        axis = self.diagnostic.axes[0, 0]
        x, y = self.axisCenter(axis)
        norm = self.diagnostic._norms["shared"]

        centerStart = 0.5 * (norm.vmin + norm.vmax)
        halfStart = 0.5 * (norm.vmax - norm.vmin)
        pressButton(self.diagnostic.fig.canvas, x, y, 3)
        moveMouse(self.diagnostic.fig.canvas, x + 50, y)
        centerNew = 0.5 * (norm.vmin + norm.vmax)
        halfNew = 0.5 * (norm.vmax - norm.vmin)
        self.assertLess(centerNew, centerStart)
        self.assertAlmostEqual(halfNew, halfStart)
        releaseButton(self.diagnostic.fig.canvas, x + 50, y, 3)

        halfStart2 = 0.5 * (norm.vmax - norm.vmin)
        pressButton(self.diagnostic.fig.canvas, x, y, 3)
        moveMouse(self.diagnostic.fig.canvas, x, y + 50)
        halfNew2 = 0.5 * (norm.vmax - norm.vmin)
        self.assertLess(halfNew2, halfStart2)
        releaseButton(self.diagnostic.fig.canvas, x, y + 50, 3)

    def testToolbarGuard(self):
        """No interaction happens while the navigation toolbar's pan/zoom is engaged"""

        class FakeToolbar:
            mode = "pan/zoom"

        axis = self.diagnostic.axes[0, 0]
        x, y = self.axisCenter(axis)
        self.diagnostic.fig.canvas.toolbar = FakeToolbar()
        pressButton(self.diagnostic.fig.canvas, x, y, 1)
        releaseButton(self.diagnostic.fig.canvas, x, y, 1)
        self.assertEqual(len(self.diagnostic.marks), 0)

    def testClearMarksKey(self):
        """The 'c' key clears all marks"""
        axis = self.diagnostic.axes[0, 0]
        x, y = self.axisCenter(axis)
        pressButton(self.diagnostic.fig.canvas, x, y, 1)
        releaseButton(self.diagnostic.fig.canvas, x, y, 1)
        self.assertEqual(len(self.diagnostic.marks), 1)

        pressKey(self.diagnostic.fig.canvas, "c")
        self.assertEqual(len(self.diagnostic.marks), 0)

    def testResetStretchKey(self):
        """The '0' key resets the stretch to its initial, automatic values"""
        axis = self.diagnostic.axes[0, 0]
        x, y = self.axisCenter(axis)
        initShared = self.diagnostic._initRanges["shared"]

        pressButton(self.diagnostic.fig.canvas, x, y, 3)
        moveMouse(self.diagnostic.fig.canvas, x + 80, y + 80)
        releaseButton(self.diagnostic.fig.canvas, x + 80, y + 80, 3)
        norm = self.diagnostic._norms["shared"]
        self.assertNotEqual((norm.vmin, norm.vmax), initShared)

        pressKey(self.diagnostic.fig.canvas, "0")
        self.assertEqual((norm.vmin, norm.vmax), initShared)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
