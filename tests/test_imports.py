import importlib.util
import json
import subprocess
import sys

import lsst.utils.tests

from pfs.drp.stella.tests import runTests

# Plotting, display and opdb tools for notebooks; nothing in the pipeline needs them.
NOTEBOOK_MODULES = [
    f"pfs.drp.stella.utils.{name}" for name in (
        "ag_to_zenith_offset", "asinhNorm", "display", "fiberProfiles", "fiberThroughputs", "guiders",
        "patrolRegionFluxes", "pfiFocus", "plotting", "quality", "raster", "stability", "sunss",
    )
]


class ImportTestCase(lsst.utils.tests.TestCase):
    """Importing this package loads only what the pipeline needs.

    Every pipeline process imports it, so anything it pulls in costs every
    process; the notebook tools bring matplotlib and opdb access with them.
    """

    def testNotebookModulesExist(self):
        """A misspelt or removed name would make the check below vacuous."""
        for name in NOTEBOOK_MODULES:
            self.assertIsNotNone(importlib.util.find_spec(name), name)

    def testNotebookModulesNotLoaded(self):
        """Run in a fresh interpreter, because other tests in this process
        import some of these modules directly.
        """
        code = "import json, sys, pfs.drp.stella; print(json.dumps(sorted(sys.modules)))"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = set(json.loads(result.stdout.splitlines()[-1]))
        self.assertEqual([name for name in NOTEBOOK_MODULES if name in loaded], [])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
