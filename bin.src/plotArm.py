#!/usr/bin/env python

from lsst.pex.config import Config, Field
from lsst.pipe.base import CmdLineTask, TaskRunner
from pfs.drp.stella import SpectrumSet


class PlotArmConfig(Config):
    numRows = Field(dtype=int, default=3, doc="Number of rows in plot")


class PlotArmRunner(TaskRunner):
    @classmethod
    def getTargetList(cls, parsedCmd, **kwargs):
        return super().getTargetList(parsedCmd, filename=parsedCmd.filename)


class PlotArmTask(CmdLineTask):
    ConfigClass = PlotArmConfig
    RunnerClass = PlotArmRunner
    _DefaultName = "plotArm"

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = super()._makeArgumentParser(*args, **kwargs)
        parser.add_argument("--filename", required=True,
                            help=("Format for plot filename, e.g., "
                                  "spectra-%%(visit)d-%%(arm)s%%(spectrograph)d.png"))
        return parser

    def run(self, dataRef, filename):
        spectra = SpectrumSet.fromPfsArm(dataRef.get("pfsArm"))
        fnUse = filename % dataRef.dataId
        self.log.info("Plotting arm %s as %s", dataRef.dataId, fnUse)
        spectra.plot(numRows=self.config.numRows, filename=fnUse)

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None


PlotArmTask.parseAndRun()
