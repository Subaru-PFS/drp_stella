from lsst.pipe.base import ArgumentParser

class TraceArgumentParser(ArgumentParser):
    """Argument parser to support tracing spectra"""
    def __init__(self, *args, **kwargs):
        super(TraceArgumentParser, self).__init__(*args, **kwargs)
        self.add_argument("-n", "--dry-run", dest="dryrun", action="store_true", default=False,
                          help="Don't perform any action?")
        self.add_argument("--mode", choices=["move", "copy", "link", "skip"], default="link",
                          help="Mode of delivering the files to their destination")
        self.add_argument("--create", action="store_true", help="Create new registry (clobber old)?")
        self.add_id_argument("--badId", "raw", "Data identifier for bad data", doMakeDataRefList=False)
        self.add_argument("--badFile", nargs="*", default=[],
                          help="Names of bad files (no path; wildcards allowed)")
        self.add_argument("files", nargs="+", help="Names of file")

class SubaruIngestTask(IngestTask):
    ArgumentParser = SubaruIngestArgumentParser
