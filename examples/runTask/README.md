# Running a Task on local files: `runTask.py`

`runTask.py` configures and runs a single `Task` (or `PipelineTask`) on files on the local filesystem,
without a butler.
It is intended for debugging and development:
re-running one step of the pipeline on its saved inputs, with different configuration, more logging,
or a debugger.

It calls the task's `run` method directly (not `runQuantum`), so everything `run` needs must be supplied,
either from files on the command line or from an "extra" python file (see [Extra files](#extra-files)).

The underlying functions are in `pfs.drp.stella.harness`, and can also be used from python
(`harness.runTask`).

This directory also contains templates for the extra files (see [Templates](#templates)).


## Basic usage

Set up the stack and the packages you are working on, then:

    runTask.py TASK [name:type=filename ...] [options]

`TASK` is the fully-qualified dotted path to the task class,
e.g. `pfs.drp.stella.fitDetectorMap.FitDetectorMapTask`.

Each `name:type=filename` reads `filename` as an object of type `type` (see [Data types](#data-types)),
and passes it to `run` as the keyword argument `name`.
`name` must therefore match a parameter of the task's `run` method.
For example:

    runTask.py pfs.drp.stella.fitDetectorMap.FitDetectorMapTask \
        lines:ArcLineSet=detectorMapLines.fits base:DetectorMap=detectorMap.fits \
        --extra FitDetectorMapTask.py \
        --config overrides.py \
        --log-level fitDetectorMap=DEBUG \
        --output detectorMap:detectorMap-new.fits

Options:

- `--config FILE`: configuration overrides, in the usual `config.field = value` form.
  The configuration is validated and frozen before the task is constructed, as in the pipeline framework.
- `--extra FILE`: python file to supply or modify the arguments to `run`; see [Extra files](#extra-files).
- `--output name:filename`: write the attribute `name` of the task's result (usually a `Struct`)
  to `filename`. May be repeated.
  The syntax of all outputs is checked before the task runs,
  and all the attributes are checked before any are written.
  Most data products are written with their own `writeFits`;
  a matplotlib `Figure` is saved with `savefig`,
  and an afw `Psf` is written inside a small dummy `Exposure`.
  Other special cases can be added to `harness.OUTPUT_WRITERS`.
- `--log-level [name=]LEVEL`: set a log level (case-insensitive). May be repeated.
  Without `name=`, it sets the root logger, which otherwise defaults to `INFO`.
  The task's logger is named after the task's `_DefaultName` (e.g., `fitDetectorMap`),
  and its subtasks are children of that (e.g., `fitDetectorMap.subtaskName`).
- `--debug`: import a `debug` module (from the current directory or `PYTHONPATH`) to configure `lsstDebug`,
  for tasks with debug displays or plots.
- `--pdb`: drop into the python debugger if there is an exception.


## Data types

The `type` in `name:type=filename` is a key in `harness.TYPE_REGISTRY`. Currently:

| Type              | Class                                        |
|-------------------|----------------------------------------------|
| `PfsArm`          | `pfs.drp.stella.PfsArm`                      |
| `PfsConfig`       | `pfs.drp.stella.PfsConfig`                   |
| `DetectorMap`     | `pfs.drp.stella.DetectorMap` (any subclass)  |
| `ArcLineSet`      | `pfs.drp.stella.ArcLineSet`                  |
| `FiberProfileSet` | `pfs.drp.stella.FiberProfileSet`             |
| `Exposure`        | `lsst.afw.image.ExposureF`                   |

To add a type, add an entry mapping the name to either the dotted path of a class with a
`readFits(filename)` classmethod, or a function that takes a filename and returns the object.
The reader must accept an arbitrary filename.
For example, `FiberTraceSet` isn't included because its `readFits` parses the filename for the data
identifier; instead, read a `FiberProfileSet` and a `DetectorMap`, and make the traces in an extra file.

Arguments that aren't single objects read from a file (strings, numbers, lists, objects constructed in
code) are supplied with an extra file.


## Extra files

An extra file is a python file that is executed after the data on the command line have been read,
and before the task is constructed and run.
It is executed with these names defined:

- `data`: `dict` of the keyword arguments for `run`, containing the data read from the command line.
  Add, replace or delete entries to change what is passed to `run`, e.g., `data["arm"] = "r"`.
  The keys must match the parameters of `run`.
- `__file__`: the path to the extra file, e.g., for reading files alongside it.
- `__name__`: `"__runTask_extra__"`.

Typical uses:

- **Values that aren't files**: a data identifier, an arm name, a bounding box, a `VisitInfo`.
- **Lists**: `run` methods that take one entry per exposure need a list,
  which the command line can't express. Read each file with `harness.readData`:

      from pfs.drp.stella.harness import readData
      data["exposures"] = [readData("Exposure", fn) for fn in ("calexp1.fits", "calexp2.fits")]

- **Derived inputs**: objects built from other inputs, e.g., fiberTraces from a fiberProfileSet and a
  detectorMap, or an LSF from a detectorMap.
- **Modified inputs**: e.g., resetting status flags left by a previous run,
  or selecting a subset of lines to speed things up.

The configuration has been frozen by the time the extra file runs,
so use `--config` for configuration, not the extra file.

An extra file is arbitrary python code: read it before running it.


## Templates

The python files in this directory are templates for the extra files of some tasks,
named after the task.
Copy one, review and edit it, and pass the copy with `--extra`:

    cp $DRP_STELLA_DIR/examples/runTask/FitDetectorMapTask.py .
    # review and edit FitDetectorMapTask.py
    runTask.py pfs.drp.stella.fitDetectorMap.FitDetectorMapTask \
        lines:ArcLineSet=detectorMapLines.fits --extra FitDetectorMapTask.py

Each template has:

- a docstring giving the command line to use and what the template does;
- the declarations `TASK` (dotted path to the task) and `INPUTS`
  (a `dict` of the names and types that may be supplied on the command line),
  which are used by the unit tests;
- an "Edit these" block of the values that change from use to use (e.g., the arm),
  which should be reviewed every time;
- the code that sets the entries of `data`.

To add a template, follow the same structure.
`tests/test_harness.py` checks every template in this directory against the current code:
that the task, the imported modules and names, and the `INPUTS` types exist,
and that `INPUTS` plus the `data["..."]` assignments supply every required argument of `run`
and no unknown ones.
The check finds the assignments by parsing the file, so assign with a literal key (`data["name"] = ...`).
Where synthetic inputs are easy to make, also add a test that runs the template
(see `DetectorMapTemplatesTestCase` in `tests/test_harness.py`).
The templates are exempt from flake8's undefined-name check (F821), because `data` is provided by the
harness.
