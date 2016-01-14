#!/usr/bin/env python

#import hsc.pipe.base.log # Monkey-patch lsst.pex.logging
import pfs.drp.stella.log # Monkey-patch lsst.pex.logging
#import lsst.pex.logging

import re
import os
import os.path
import stat
import sys
import tempfile
import argparse
import traceback
import contextlib
import lsst.pex.logging as pexLog
import lsst.afw.cameraGeom as cameraGeom
from lsst.pipe.base import CmdLineTask
#from hsc.pipe.base.pool import startPool, NODE, abortOnError
from pfs.drp.stella.pool import startPool, NODE, abortOnError

__all__ = ["Batch", "PbsBatch", "SlurmBatch", "SmpBatch", "BATCH_TYPES", "BatchArgumentParser",
           "BatchCmdLineTask", "BatchPoolTask",]

UMASK = "002" # umask to set

# Functions to convert a list of arguments to a quoted shell command, provided by Dave Abrahams
# http://stackoverflow.com/questions/967443/python-module-to-shellquote-unshellquote
_quote_pos = re.compile('(?=[^-0-9a-zA-Z_./\n])')
def shQuote(arg):
    r"""Quote the argument for the shell.

    >>> quote('\t')
    '\\\t'
    >>> quote('foo bar')
    'foo\\ bar'
    """
    # This is the logic emacs uses
    if arg:
        return _quote_pos.sub('\\\\', arg).replace('\n',"'\n'")
    else:
        return "''"
def shCommandFromArgs(args):
    """Convert a list of shell arguments to a shell command-line"""
    return ' '.join([shQuote(a) for a in args])

def processStats():
    """Collect Linux-specific process statistics

    Parses the /proc/self/status file (N.B. Linux-specific!) into a dict
    which is returned.
    """
    result = {}
    with open("/proc/self/status") as f:
        for line in f:
            key, _, value = line.partition(":")
            result[key] = value.strip()
    return result

def printProcessStats():
    """Print the process statistics to the log"""
    from lsst.pex.logging import getDefaultLog
    getDefaultLog().info("Process stats for %s: %s" % (NODE, processStats()))

class Batch(object):
    """Base class for batch submission"""
    def __init__(self, outputDir=None, numNodes=1, numProcsPerNode=1, queue=None, jobName=None, walltime=None,
                 dryrun=False, doExec=False, mpiexec="", submit=None, options=None):
        """Constructor

        @param outputDir: output directory, or None
        @param numNodes: number of nodes
        @param numProcsPerNode: number of processors per node
        @param queue: name of queue, or None
        @param jobName: name of job, or None
        @param walltime: maximum wall clock time for job
        @param dryrun: Dry run (only print actions that would be taken)?
        @param doExec: exec the script instead of submitting to batch system?
        @param mpiexec: options for mpiexec
        @param submit: command-line options for batch submission (e.g., for qsub, sbatch)
        @param options: options to append to script header (e.g., #PBS or #SBATCH)
        """
        self.outputDir = outputDir
        self.numNodes = numNodes
        self.numProcsPerNode = numProcsPerNode
        self.queue = queue
        self.jobName = jobName
        self.walltime = walltime
        self.dryrun = dryrun
        self.doExec = doExec
        self.mpiexec = mpiexec
        self.submit = submit
        self.options = options

    def shebang(self):
        return "#!/bin/bash"

    def preamble(self, command, walltime=None):
        """Return preamble string for script to be submitted

        Most batch systems allow you to embed submission options as comments here.
        """
        raise NotImplementedError("Not implemented for base class")

    def execution(self, command):
        """Return execution string for script to be submitted"""
        return "\n".join([exportEnv(),
                          "date",
                          "echo \"mpiexec is at: $(which mpiexec)\"",
                          "ulimit -a",
                          "umask %s" % UMASK,
                          "echo 'umask: ' $(umask)",
                          "eups list -s",
                          "export",
                          "cd %s" % os.getcwd(),
                          "date",
                          "mpiexec %s %s" % (self.mpiexec, command),
                          "date",
                          "echo Done.",
                          ])

    def createScript(self, command, walltime=None):
        """Create script to be submitted

        @param command: command to run
        @param walltime: maximum wall clock time, overrides value to constructor
        @return name of script on filesystem
        """
        fd, scriptName = tempfile.mkstemp()
        with os.fdopen(fd, "w") as f:
            print >>f, self.shebang()
            print >>f, self.preamble(walltime)
            print >>f, self.execution(command)

        os.chmod(scriptName, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        return scriptName

    def submitCommand(self, scriptName):
        """Return command to submit script

        @param scriptName: name of script on filesystem
        """
        raise NotImplementedError("No implementation for base class")

    def run(self, command, walltime=None):
        """Run the batch system

        Creates and submits the script to execute the provided command

        @param command: command to run
        @param walltime: maximum wall clock time, overrides value to constructor
        @return name of script on filesystem
        """
        scriptName = self.createScript(command, walltime=walltime)
        command = self.submitCommand(scriptName)
        if self.dryrun:
            print "Would run: %s" % command
        elif self.doExec:
            os.execl(scriptName, scriptName)
        else:
            os.system(command)
        return scriptName


class PbsBatch(Batch):
    """Batch submission with PBS"""
    def preamble(self, walltime=None):
        if walltime is None:
            walltime = self.walltime
        return "\n".join([
                          "#PBS %s" % self.options if self.options is not None else "",
                          "#PBS -l nodes=%d:ppn=%d" % (self.numNodes, self.numProcsPerNode),
                          "#PBS -l walltime=%d" % walltime if walltime is not None else "",
                          "#PBS -o %s" % self.outputDir if self.outputDir is not None else "",
                          "#PBS -N %s" % self.jobName if self.jobName is not None else "",
                          "#PBS -q %s" % self.queue if self.queue is not None else "",
                          "#PBS -j oe",
                          "#PBS -W umask=%s" % UMASK,
                          ])

    def submitCommand(self, scriptName):
        return "qsub %s -V %s" % (self.submit if self.submit is not None else "", scriptName)

class SlurmBatch(Batch):
    """Batch submission with Slurm"""
    def preamble(self, walltime=None):
        if walltime is None:
            walltime = self.walltime
        outputDir = self.outputDir if self.outputDir is not None else os.getcwd()
        filename = os.path.join(outputDir, (self.jobName if self.jobName is not None else "slurm") + ".o%j")
        return "\n".join(["#SBATCH --nodes=%d" % self.numNodes,
                          "#SBATCH --ntasks-per-node=%d" % self.numProcsPerNode,
                          "#SBATCH --time=%d" % (walltime/60.0 + 0.5) if walltime is not None else "",
                          "#SBATCH --job-name=%s" % self.jobName if self.jobName is not None else "",
                          "#SBATCH -p %s" % self.queue if self.queue is not None else "",
                          "#SBATCH --output=%s" % filename,
                          "#SBATCH --error=%s" % filename,
                          "#SBATCH %s" % self.options if self.options is not None else "",
                          ])

    def submitCommand(self, scriptName):
        return "sbatch %s %s" % (self.submit if self.submit is not None else "", scriptName)

class SmpBatch(Batch):
    """Not-really-Batch submission with multiple cores on the current node

    The job is run immediately.
    """
    def __init__(self, *args, **kwargs):
        super(SmpBatch, self).__init__(*args, **kwargs)
        self.mpiexec = "%s -n %d" % (self.mpiexec if self.mpiexec is not None else "", self.numProcsPerNode)
        if self.numNodes != 1:
            raise RuntimeError("SMP requires only a single node be specified (--nodes), not %d" %
                               self.numNodes)

    def preamble(self, walltime=None):
        return ""

    def submitCommand(self, scriptName):
        return "exec %s" % scriptName


BATCH_TYPES = {'pbs': PbsBatch,
               'slurm': SlurmBatch,
               'smp': SmpBatch,
               } # Mapping batch type --> Batch class

class BatchArgumentParser(argparse.ArgumentParser):
    """An argument parser to get relevant parameters for batch submission

    We want to be able to display the help for a 'parent' ArgumentParser
    along with the batch-specific options we introduce in this class, but
    we don't want to swallow the parent (i.e., ArgumentParser(parents=[parent]))
    because we want to save the list of arguments that this particular
    BatchArgumentParser doesn't parse, so they can be passed on to a different
    program (though we also want to parse them to check that they can be parsed).
    """
    def __init__(self, parent=None, *args, **kwargs):
        super(BatchArgumentParser, self).__init__(*args, **kwargs)
        self._parent = parent
        group = self.add_argument_group("Batch submission options")
        group.add_argument("--queue", help="Queue name")
        group.add_argument("--job", help="Job name")
        group.add_argument("--nodes", type=int, default=1, help="Number of nodes")
        group.add_argument("--procs", type=int, default=1, help="Number of processors per node")
        group.add_argument("--time", type=float, default=1000,
                           help="Expected execution time per element (sec)")
        group.add_argument("--batch-type", dest="batchType", choices=BATCH_TYPES.keys(), default="pbs",
                           help="Batch system to use")
        group.add_argument("--batch-output", dest="batchOutput", help="Output directory")
        group.add_argument("--batch-submit", dest="batchSubmit", help="Batch submission command-line flags")
        group.add_argument("--batch-options", dest="batchOptions", help="Header options for batch script")
        group.add_argument("--batch-profile", dest="batchProfile", action="store_true", default=False,
                           help="Enable profiling on batch job?")
        group.add_argument("--batch-stats", dest="batchStats", action="store_true", default=False,
                           help="Print process stats on completion (Linux only)?")
        group.add_argument("--dry-run", dest="dryrun", default=False, action="store_true",
                           help="Dry run?")
        group.add_argument("--do-exec", dest="doExec", default=False, action="store_true",
                           help="Exec script instead of submit to batch system?")
        group.add_argument("--mpiexec", default="", help="mpiexec options")

    def parse_args(self, config=None, args=None, namespace=None, **kwargs):
        args, leftover = super(BatchArgumentParser, self).parse_known_args(args=args, namespace=namespace)
        args.parent = None
        args.leftover = None
        if len(leftover) > 0:
            # Save any leftovers for the parent
            if self._parent is None:
                self.error("Unrecognised arguments: %s" % leftover)
            args.parent = self._parent.parse_args(config, args=leftover, **kwargs)
            args.leftover = leftover
        args.batch = self.makeBatch(args)
        return args

    def makeBatch(self, args):
        """Create a Batch object from the command-line arguments"""
        argMapping = {'outputDir': 'batchOutput',
                      'numNodes': 'nodes',
                      'numProcsPerNode': 'procs',
                      'walltime': 'time',
                      'queue': 'queue',
                      'jobName': 'job',
                      'dryrun': 'dryrun',
                      'doExec': 'doExec',
                      'mpiexec': 'mpiexec',
                      'submit': 'batchSubmit',
                      'options': 'batchOptions',
                      } # Mapping Batch init kwargs --> argument parser elements
        kwargs = dict((k, getattr(args, v)) for k,v in argMapping.iteritems())
        return BATCH_TYPES[args.batchType](**kwargs)

    def format_help(self):
        text = """This is a script for queue submission of a wrapped script.

Use this program name and ignore that for the wrapped script (it will be
passed on to the batch system).  Arguments for *both* this wrapper script or the
wrapped script are valid (if it is required for the wrapped script, it
is required for the wrapper as well).

*** Batch system submission wrapper:

"""
        text += super(BatchArgumentParser, self).format_help()
        if self._parent is not None:
            text += """

*** Wrapped script:

"""
            text += self._parent.format_help()
        return text

    def format_usage(self):
        if self._parent is not None:
            prog = self._parent.prog
            self._parent.prog = self.prog
            usage = self._parent.format_usage()
            self._parent.prog = prog
            return usage
        return super(BatchArgumentParser, self).format_usage()


def exportEnv():
    """Generate bash script to regenerate the current environment"""
    output = ""
    for key, val in os.environ.items():
        if key in ("DISPLAY",):
            continue
        if val.startswith("() {"):
            # This is a function.
            # "Two parentheses, a single space, and a brace"
            # is exactly the same criterion as bash uses.

            # From 2014-09-25, the function name is prefixed by 'BASH_FUNC_'
            # and suffixed by '()', which we have to remove.
            if key.startswith("BASH_FUNC_") and key.endswith("()"):
                key = key[10:-2]

            output += "{key} {val}\nexport -f {key}\n".format(key=key, val=val)
        else:
            # This is a variable.
            output += "export {key}='{val}'\n".format(key=key, val=val.replace("'", "'\"'\"'"))
    return output


class BatchCmdLineTask(CmdLineTask):
    @classmethod
    def parseAndSubmit(cls, args=None, **kwargs):
        taskParser = cls._makeArgumentParser(doBatch=True, add_help=False)
        batchParser = BatchArgumentParser(parent=taskParser)
        batchArgs = batchParser.parse_args(config=cls.ConfigClass(), args=args, override=cls.applyOverrides,
                                           **kwargs)

        if not cls.RunnerClass(cls, batchArgs.parent).precall(batchArgs.parent): # Write config, schema
            taskParser.error("Error in task preparation")

        walltime = cls.batchWallTime(batchArgs.time, batchArgs.parent, batchArgs.nodes, batchArgs.procs)

        command = cls.batchCommand(batchArgs)
        batchArgs.batch.run(command, walltime=walltime)

    @classmethod
    def batchWallTime(cls, time, parsedCmd, numNodes, numProcs):
        """Return walltime request for batch job

        Subclasses should override if the walltime should be calculated
        differently (e.g., addition of some serial time).

        @param time: Requested time per iteration
        @param parsedCmd: Results of argument parsing
        @param numNodes: Number of nodes for processing
        @param numProcs: Number of processors per node
        """
        numTargets = len(cls.RunnerClass.getTargetList(parsedCmd))
        return time*numTargets/(numNodes*numProcs)

    @classmethod
    def batchCommand(cls, args):
        """Return command to run CmdLineTask

        @param args: Parsed batch job arguments (from BatchArgumentParser)
        """
        job = args.job if args.job is not None else "job"
        module = cls.__module__
        script = ("import os; os.umask(%s); " +
                  "import pfs.drp.stella.log; pfs.drp.stella.log.jobLog(\"%s\"); ") % (UMASK, job)
#                  "import hsc.pipe.base.log; hsc.pipe.base.log.jobLog(\"%s\"); ") % (UMASK, job)

        if args.batchStats:
            script += ("import pfs.drp.stella.parallel; import atexit; " +
                       "atexit.register(pfs.drp.stella.parallel.printProcessStats); ")
#            script += ("import hsc.pipe.base.parallel; import atexit; " +
#                       "atexit.register(hsc.pipe.base.parallel.printProcessStats); ")

        script += "import %s; %s.%s.parseAndRun();" % (module, module, cls.__name__)

        profilePre = "import cProfile; import os; cProfile.run(\"\"\""
        profilePost = "\"\"\", filename=\"profile-" + job + "-%s-%d.dat\" % (os.uname()[1], os.getpid()))"

        return ("python -c '" + (profilePre if args.batchProfile else "") + script +
                (profilePost if args.batchProfile else "") + "' " + shCommandFromArgs(args.leftover))

    @contextlib.contextmanager
    def logOperation(self, operation, catch=False, trace=True):
        """Provide a context manager for logging an operation

        @param operation: description of operation (string)
        @param catch: Catch all exceptions?
        @param trace: Log a traceback of caught exception?

        Note that if 'catch' is True, all exceptions are swallowed, but there may
        be other side-effects such as undefined variables.
        """
        self.log.info("%s: Start %s" % (NODE, operation))
        try:
            yield
        except:
            if catch:
                cls, e, _ = sys.exc_info()
                self.log.warn("%s: Caught %s while %s: %s" % (NODE, cls.__name__, operation, e))
                if trace:
                    self.log.info("%s: Traceback:\n%s" % (NODE, traceback.format_exc()))
                return
            raise
        finally:
            self.log.info("%s: Finished %s" % (NODE, operation))

class BatchPoolTask(BatchCmdLineTask):
    @classmethod
    @abortOnError
    def parseAndRun(cls, *args, **kwargs):
        """Run with a MPI process pool"""
        pool = startPool()
        super(BatchPoolTask, cls).parseAndRun(*args, **kwargs)
        pool.exit()

