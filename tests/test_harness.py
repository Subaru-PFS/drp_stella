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


class DetectorMapTemplatesTestCase(lsst.utils.tests.TestCase):
    """Test running the FitDetectorMapTask and AdjustDetectorMapTask templates

    The templates are copied and edited, as a user would, and run on
    synthetic data.
    """

    def setUp(self):
        self.tempDir = tempfile.mkdtemp()
        rootLogger = logging.getLogger()
        self.rootHandlers = list(rootLogger.handlers)
        self.rootLevel = rootLogger.level

        self.synthConfig = SyntheticConfig()
        self.base = makeSyntheticDetectorMap(self.synthConfig, 650.0, 950.0)
        self.baseFile = os.path.join(self.tempDir, "base.fits")
        self.base.writeFits(self.baseFile)

        rows = []
        for ff in self.synthConfig.fiberId:
            for yy in range(self.base.bbox.getMinY(), self.base.bbox.getMaxY(), 4):
                rows.append(
                    ArcLine(
                        ff,
                        self.base.getWavelength(ff, yy),
                        self.base.getXCenter(ff, yy),
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
        self.linesFile = os.path.join(self.tempDir, "lines.fits")
        ArcLineSet.fromRows(rows).writeFits(self.linesFile)

    def tearDown(self):
        shutil.rmtree(self.tempDir, ignore_errors=True)
        rootLogger = logging.getLogger()
        for handler in list(rootLogger.handlers):
            if handler not in self.rootHandlers:
                rootLogger.removeHandler(handler)
        rootLogger.setLevel(self.rootLevel)

    def editTemplate(self, name, replacements=None):
        """Copy and edit a template, as a user would

        Parameters
        ----------
        name : `str`
            Name of the template file in ``EXAMPLES_DIR``.
        replacements : `dict` [`str`: `str`], optional
            Text to replace in the template. Each must appear exactly once.
            Defaults to setting the arm to ``r``.

        Returns
        -------
        task : `str`
            Fully-qualified dotted path to the task.
        extra : `str`
            Path to the edited copy of the template.
        """
        template = os.path.join(EXAMPLES_DIR, name)
        task, _, _, _ = parseTemplate(template)
        if replacements is None:
            replacements = {'ARM = "b"': 'ARM = "r"'}
        with open(template) as fd:
            contents = fd.read()
        for old, new in replacements.items():
            self.assertEqual(contents.count(old), 1, f"{old!r} in {name}")
            contents = contents.replace(old, new)
        extra = os.path.join(self.tempDir, f"extra_{name}")
        with open(extra, "w") as fd:
            fd.write(contents)
        return task, extra

    def writeConfig(self, name, overrides):
        """Write a config override file"""
        filename = os.path.join(self.tempDir, name)
        with open(filename, "w") as fd:
            fd.write(overrides)
        return filename

    def fitDetectorMap(self):
        """Run the FitDetectorMapTask template

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result from the task.
        filename : `str`
            Path to the fitted detectorMap.
        """
        task, extra = self.editTemplate("FitDetectorMapTask.py")
        configFile = self.writeConfig(
            "fitConfig.py", "config.order = 1\nconfig.doSlitOffsets = True\nconfig.exclusionRadius = 1.0\n"
        )
        outFile = os.path.join(self.tempDir, "detectorMap.fits")
        result = harness.runTask(
            task,
            [f"lines:ArcLineSet={self.linesFile}", f"base:DetectorMap={self.baseFile}"],
            configFile=configFile,
            extraFile=extra,
            outputSpecs=[f"detectorMap:{outFile}"],
        )
        return result, outFile

    def testFitDetectorMap(self):
        """Test the FitDetectorMapTask template"""
        result, outFile = self.fitDetectorMap()
        detMap = DetectorMap.readFits(outFile)
        self.assertEqual(detMap.bbox, self.base.bbox)
        self.assertFloatsAlmostEqual(detMap.opticsModel.distortions[0].getCoefficients(), 0.0, atol=2.0e-7)
        self.assertFloatsAlmostEqual(result.xRms, 0.0, atol=1.0e-3)

    def testAdjustDetectorMap(self):
        """Test the AdjustDetectorMapTask template

        We adjust the fitted detectorMap using the same lines, so the
        adjustment should be negligible.
        """
        _, fitFile = self.fitDetectorMap()
        task, extra = self.editTemplate("AdjustDetectorMapTask.py")
        configFile = self.writeConfig("adjustConfig.py", "config.exclusionRadius = 1.0\n")
        outFile = os.path.join(self.tempDir, "adjusted.fits")
        result = harness.runTask(
            task,
            [f"detectorMap:DetectorMap={fitFile}", f"lines:ArcLineSet={self.linesFile}"],
            configFile=configFile,
            extraFile=extra,
            outputSpecs=[f"detectorMap:{outFile}"],
        )
        fitted = DetectorMap.readFits(fitFile)
        adjusted = DetectorMap.readFits(outFile)
        self.assertFloatsAlmostEqual(result.xRms, 0.0, atol=1.0e-3)
        for fiberId in self.synthConfig.fiberId:
            self.assertFloatsAlmostEqual(
                adjusted.getXCenter(fiberId), fitted.getXCenter(fiberId), atol=1.0e-3
            )
            self.assertFloatsAlmostEqual(
                adjusted.getWavelength(fiberId), fitted.getWavelength(fiberId), atol=1.0e-4
            )


class FitDetectorMapCombinedTemplateTestCase(DetectorMapTemplatesTestCase):
    """Test running the FitDetectorMapCombinedTask template on synthetic data

    This follows ``test_fitDetectorMapCombined.py``, with arc lines only.
    """

    def setUp(self):
        super().setUp()
        # Recreate the inputs to match test_fitDetectorMapCombined.py
        self.synthConfig = SyntheticConfig()
        self.synthConfig.separation = 100  # Avoid having a fiber go down the middle
        self.base = makeSyntheticDetectorMap(self.synthConfig, 650.0, 950.0)
        self.base.writeFits(os.path.join(self.tempDir, "detectorMap-sim-n1.fits"))
        rows = []
        for ff in self.synthConfig.fiberId:
            for yy in np.arange(self.base.bbox.getMinY() + 1, self.base.bbox.getMaxY(), 50, dtype=float):
                rows.append(
                    ArcLine(
                        ff,
                        self.base.findWavelength(ff, yy),
                        self.base.getXCenter(ff, yy),
                        yy,
                        0.01,
                        0.01,
                        np.nan,
                        np.nan,
                        np.nan,
                        1000.0,
                        1.0,
                        np.nan,
                        False,
                        ReferenceLineStatus.GOOD,
                        "Fake",
                        None,
                        ReferenceLineSource.NONE,
                    )
                )
        self.linesFile = os.path.join(self.tempDir, "combinedLines.fits")
        ArcLineSet.fromRows(rows).writeFits(self.linesFile)

    def testFitDetectorMap(self):
        pass  # Covered by DetectorMapTemplatesTestCase

    def testAdjustDetectorMap(self):
        pass  # Covered by DetectorMapTemplatesTestCase

    def testRun(self):
        """Test the FitDetectorMapCombinedTask template"""
        dims = self.base.bbox.getDimensions()
        task, extra = self.editTemplate(
            "FitDetectorMapCombinedTask.py",
            {'ARM = "b"': 'ARM = "n"', "DIMENSIONS = (4096, 4176)": f"DIMENSIONS = ({dims.x}, {dims.y})"},
        )
        basePath = os.path.join(self.tempDir, "detectorMap-sim-%(arm)s%(spectrograph)s.fits")
        overrides = "".join(
            f"config.{sub}.{line}\n"
            for sub in ("fitDetectorMap", "adjustDetectorMap")
            for line in (
                "order = 1",
                "doSlitOffsets = False",
                "exclusionRadius = 1.0",
                f"base = {basePath!r}",
            )
        )
        configFile = self.writeConfig("combinedConfig.py", overrides)
        outFile = os.path.join(self.tempDir, "combined.fits")
        result = harness.runTask(
            task,
            [f"arcLines:ArcLineSet={self.linesFile}"],
            configFile=configFile,
            extraFile=extra,
            outputSpecs=[f"detectorMap:{outFile}"],
        )
        detMap = DetectorMap.readFits(outFile)
        self.assertFloatsAlmostEqual(detMap.getXCenter(), self.base.getXCenter(), atol=1.0e-3)
        self.assertFloatsAlmostEqual(detMap.getWavelength(), self.base.getWavelength(), rtol=1.0e-5)
        self.assertTrue(np.all(result.lines.description == "Fake"))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
