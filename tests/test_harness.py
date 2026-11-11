import argparse
import ast
import glob
import importlib
import inspect
import logging
import os
import shutil
import tempfile

import numpy as np

import lsst.utils.tests
from lsst.pex.config import Config, Field, FieldValidationError
from lsst.pipe.base import Struct, Task

from pfs.drp.stella import harness
from pfs.drp.stella.arcLine import ArcLine, ArcLineSet
from pfs.drp.stella.DetectorMapContinued import DetectorMap
from pfs.drp.stella.referenceLine import ReferenceLineSource, ReferenceLineStatus
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella.tests.utils import runTests

display = None

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "runTask")


class DummyConfig(Config):
    value = Field(dtype=int, default=1, doc="Must be positive")

    def validate(self):
        super().validate()
        if self.value <= 0:
            raise FieldValidationError(DummyConfig.value, self, "must be positive")


class DummyTask(Task):
    """Task for testing the harness: returns its inputs"""

    ConfigClass = DummyConfig
    _DefaultName = "dummy"
    numRuns = 0

    def run(self, **kwargs):
        DummyTask.numRuns += 1
        return Struct(value=self.config.value, **kwargs)


class Writable:
    """Object that records what it was written to"""

    def __init__(self):
        self.written = []

    def writeFits(self, filename):
        self.written.append(filename)


class HarnessTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.tempDir = tempfile.mkdtemp()
        rootLogger = logging.getLogger()
        self.rootHandlers = list(rootLogger.handlers)
        self.rootLevel = rootLogger.level
        self.registry = dict(harness.TYPE_REGISTRY)
        DummyTask.numRuns = 0

    def tearDown(self):
        shutil.rmtree(self.tempDir, ignore_errors=True)
        rootLogger = logging.getLogger()
        for handler in list(rootLogger.handlers):
            if handler not in self.rootHandlers:
                rootLogger.removeHandler(handler)
                handler.close()
        rootLogger.setLevel(self.rootLevel)
        harness.TYPE_REGISTRY.clear()
        harness.TYPE_REGISTRY.update(self.registry)

    def writeFile(self, name, contents):
        path = os.path.join(self.tempDir, name)
        with open(path, "w") as fd:
            fd.write(contents)
        return path

    def testParseSpecs(self):
        """Test parsing of data and output specifications"""
        self.assertEqual(
            harness.parseDataSpec("lines:ArcLineSet=a=b.fits"), ("lines", "ArcLineSet", "a=b.fits")
        )
        self.assertEqual(harness.parseOutputSpec("detectorMap:dir/out.fits"), ("detectorMap", "dir/out.fits"))
        for spec in ("lines=foo.fits", "lines:ArcLineSet", ":ArcLineSet=foo.fits"):
            with self.assertRaises(argparse.ArgumentTypeError):
                harness.parseDataSpec(spec)
        for spec in ("out.fits", "detectorMap:"):
            with self.assertRaises(argparse.ArgumentTypeError):
                harness.parseOutputSpec(spec)

    def testReadData(self):
        """Test reading data, including unknown types and duplicate names"""
        harness.TYPE_REGISTRY["String"] = lambda path: f"read {path}"
        self.assertEqual(
            harness.readDataSpecs(["aa:String=foo", "bb:String=bar"]), dict(aa="read foo", bb="read bar")
        )
        with self.assertRaises(ValueError):
            harness.readDataSpecs(["aa:String=foo", "aa:String=bar"])
        with self.assertRaises(KeyError):
            harness.readData("NoSuchType", "foo")

    def testWriteResult(self):
        """Test choosing how to write a result"""
        obj = Writable()
        harness.writeResult(obj, "foo.fits")
        self.assertEqual(obj.written, ["foo.fits"])

        from matplotlib.figure import Figure

        filename = os.path.join(self.tempDir, "figure.png")
        harness.writeResult(Figure(), filename)
        self.assertTrue(os.path.exists(filename))

        with self.assertRaises(TypeError):
            harness.writeResult(object(), "foo.fits")

    def testWriteOutputs(self):
        """Test that all outputs are checked before any are written"""
        obj = Writable()
        result = Struct(good=obj)
        with self.assertRaises(AttributeError):
            harness.writeOutputs(result, ["good:good.fits", "bad:bad.fits"])
        self.assertEqual(obj.written, [])
        harness.writeOutputs(result, ["good:good.fits"])
        self.assertEqual(obj.written, ["good.fits"])

    def testLogging(self):
        """Test configuration of logging"""
        self.assertEqual(harness.parseLogLevel("debug"), (None, logging.DEBUG))
        self.assertEqual(harness.parseLogLevel("foo.bar=Warning"), ("foo.bar", logging.WARNING))
        with self.assertRaises(argparse.ArgumentTypeError):
            harness.parseLogLevel("foo=NOT_A_LEVEL")

        # A file handler doesn't count as a console handler
        rootLogger = logging.getLogger()
        fileHandler = logging.FileHandler(os.path.join(self.tempDir, "log.txt"))
        rootLogger.addHandler(fileHandler)

        logger = harness.configureLogging(["harnessTest=debug"])
        self.assertIs(logger, rootLogger)
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(logging.getLogger("harnessTest").level, logging.DEBUG)
        self.assertTrue(
            any(
                isinstance(hh, logging.StreamHandler) and not isinstance(hh, logging.FileHandler)
                for hh in logger.handlers
            )
        )

        harness.configureLogging(["warning"])
        self.assertEqual(logger.level, logging.WARNING)

        # Invalid specification doesn't change anything
        with self.assertRaises(argparse.ArgumentTypeError):
            harness.configureLogging(["debug", "foo=NOT_A_LEVEL"])
        self.assertEqual(logger.level, logging.WARNING)

    def testLoadConfig(self):
        """Test that the config is loaded, validated and frozen"""
        good = self.writeFile("good.py", "config.value = 3\n")
        config = harness.loadConfig(DummyTask, good)
        self.assertEqual(config.value, 3)
        with self.assertRaises(FieldValidationError):
            config.value = 4  # frozen

        bad = self.writeFile("bad.py", "config.value = -3\n")
        with self.assertRaises(FieldValidationError):
            harness.loadConfig(DummyTask, bad)

    def testApplyExtraData(self):
        """Test that the extra file can modify the data and knows its name"""
        extra = self.writeFile(
            "extra.py", 'data["file"] = __file__\ndata["name"] = __name__\ndata["aa"] += 1\n'
        )
        data = dict(aa=1)
        harness.applyExtraData(extra, data)
        self.assertEqual(data, dict(aa=2, file=extra, name="__runTask_extra__"))
        harness.applyExtraData(None, data)
        self.assertEqual(data["aa"], 2)

    def testRunTask(self):
        """Test running a task, including checking output specifications"""
        taskPath = f"{__name__}.DummyTask"
        harness.TYPE_REGISTRY["String"] = lambda path: f"read {path}"
        config = self.writeFile("config.py", "config.value = 5\n")
        extra = self.writeFile("extra.py", 'data["extra"] = 123\n')
        result = harness.runTask(taskPath, ["aa:String=foo"], configFile=config, extraFile=extra)
        self.assertEqual(result.value, 5)
        self.assertEqual(result.aa, "read foo")
        self.assertEqual(result.extra, 123)
        self.assertEqual(DummyTask.numRuns, 1)

        # An invalid output specification is caught before running the task
        with self.assertRaises(argparse.ArgumentTypeError):
            harness.runTask(taskPath, ["aa:String=foo"], outputSpecs=(spec for spec in ["noColon"]))
        self.assertEqual(DummyTask.numRuns, 1)


def parseTemplate(filename):
    """Parse a runTask template

    Parameters
    ----------
    filename : `str`
        Path to the template.

    Returns
    -------
    task : `str`
        Fully-qualified dotted path to the task (``TASK``).
    inputs : `dict` [`str`: `str`]
        Names and types of the data read from files (``INPUTS``).
    assigned : `set` of `str`
        Names assigned in the ``data`` dict.
    modules : `list` of (`str`, `list` of `str`)
        Imported modules and the names imported from them.
    """
    with open(filename) as fd:
        tree = ast.parse(fd.read(), filename)
    declarations = {}
    assigned = set()
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("TASK", "INPUTS"):
                    declarations[target.id] = ast.literal_eval(node.value)
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "data"
                    and isinstance(target.slice, ast.Constant)
                ):
                    assigned.add(target.slice.value)
        elif isinstance(node, ast.Import):
            modules += [(alias.name, []) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules.append((node.module, [alias.name for alias in node.names]))
    return declarations["TASK"], declarations["INPUTS"], assigned, modules


class TemplatesTestCase(lsst.utils.tests.TestCase):
    """Tests that the runTask templates are up to date"""

    def setUp(self):
        self.templates = sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*.py")))
        self.assertGreater(len(self.templates), 0)

    def testSignatures(self):
        """Test that the templates supply valid arguments for the task's run"""
        for filename in self.templates:
            with self.subTest(template=os.path.basename(filename)):
                task, inputs, assigned, _ = parseTemplate(filename)
                signature = inspect.signature(harness.resolveClass(task).run)
                parameters = {
                    name: param
                    for name, param in signature.parameters.items()
                    if name != "self" and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
                }
                required = {name for name, param in parameters.items() if param.default is param.empty}
                supplied = set(inputs) | assigned
                self.assertLessEqual(required, supplied, "Template doesn't supply required arguments")
                if not any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
                    self.assertLessEqual(supplied, set(parameters), "Template supplies unknown arguments")

    def testImports(self):
        """Test that the task, imports and input types in the templates exist"""
        for filename in self.templates:
            with self.subTest(template=os.path.basename(filename)):
                task, inputs, _, modules = parseTemplate(filename)
                harness.resolveClass(task)
                for moduleName, names in modules:
                    module = importlib.import_module(moduleName)
                    for name in names:
                        self.assertTrue(hasattr(module, name), f"{moduleName} has no {name}")
                for typeName in inputs.values():
                    self.assertIn(typeName, harness.TYPE_REGISTRY)


class FitDetectorMapTemplateTestCase(lsst.utils.tests.TestCase):
    """Test running the FitDetectorMapTask template on synthetic data"""

    def setUp(self):
        self.tempDir = tempfile.mkdtemp()
        rootLogger = logging.getLogger()
        self.rootHandlers = list(rootLogger.handlers)
        self.rootLevel = rootLogger.level

    def tearDown(self):
        shutil.rmtree(self.tempDir, ignore_errors=True)
        rootLogger = logging.getLogger()
        for handler in list(rootLogger.handlers):
            if handler not in self.rootHandlers:
                rootLogger.removeHandler(handler)
        rootLogger.setLevel(self.rootLevel)

    def testRun(self):
        """Copy and edit the template, as a user would, and run it"""
        template = os.path.join(EXAMPLES_DIR, "FitDetectorMapTask.py")
        task, _, _, _ = parseTemplate(template)
        with open(template) as fd:
            contents = fd.read()
        self.assertEqual(contents.count('ARM = "b"'), 1)
        extra = os.path.join(self.tempDir, "extra.py")
        with open(extra, "w") as fd:
            fd.write(contents.replace('ARM = "b"', 'ARM = "r"'))

        synthConfig = SyntheticConfig()
        base = makeSyntheticDetectorMap(synthConfig, 650.0, 950.0)
        baseFile = os.path.join(self.tempDir, "base.fits")
        base.writeFits(baseFile)

        rows = []
        for ff in synthConfig.fiberId:
            for yy in range(base.bbox.getMinY(), base.bbox.getMaxY(), 4):
                rows.append(
                    ArcLine(
                        ff,
                        base.getWavelength(ff, yy),
                        base.getXCenter(ff, yy),
                        float(yy),
                        0.01,
                        0.01,
                        np.nan,
                        np.nan,
                        np.nan,
                        1000.0,
                        1.0,
                        np.nan,
                        False,
                        ReferenceLineStatus.GOOD | ReferenceLineStatus.DETECTORMAP_USED,
                        "Fake",
                        None,
                        ReferenceLineSource.NONE,
                    )
                )
        linesFile = os.path.join(self.tempDir, "lines.fits")
        ArcLineSet.fromRows(rows).writeFits(linesFile)

        configFile = os.path.join(self.tempDir, "config.py")
        with open(configFile, "w") as fd:
            fd.write("config.order = 1\nconfig.doSlitOffsets = True\nconfig.exclusionRadius = 1.0\n")

        outFile = os.path.join(self.tempDir, "detectorMap.fits")
        result = harness.runTask(
            task,
            [f"lines:ArcLineSet={linesFile}", f"base:DetectorMap={baseFile}"],
            configFile=configFile,
            extraFile=extra,
            outputSpecs=[f"detectorMap:{outFile}"],
        )
        detMap = DetectorMap.readFits(outFile)
        self.assertEqual(detMap.bbox, base.bbox)
        self.assertFloatsAlmostEqual(detMap.opticsModel.distortions[0].getCoefficients(), 0.0, atol=2.0e-7)
        self.assertFloatsAlmostEqual(result.xRms, 0.0, atol=1.0e-3)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
