###############
Generate the defect list
###############

For the pipeline to fix defects on the CCD, fits files (one per CCD) are required in the
$OBS_PFS_DIR/pfs/defects directory. The generation of these fits files is
done in two steps. First one needs to create a list of defects and store
this list in the file $OBS_PFS_DIR/pfs/defects/<YYYY-MM-DD>/defects.dat.
In a second step, during the compilation of obs_pfs, scons will automatically
create the needed fits files from the defects.dat file.

Generating the list of defects
==============================

The list of defects ($OBS_PFS_DIR/pfs/defects/<YYYY-MM-DD>/defects.dat) is created
by the python command genDefectList.py, which is located in $OBS_PFS_DIR/python/lsst/obs/pfs.
This command has a number of parameters which control its behavior::

    --arm: help="Arm for which to create the defect list", type=str, default='r'
    --badCols: help="List of bad columns delimited by ','", type=str, default="1020,1021,1022,1023,1024, 1025, 1026,3068,3069,3070,3071,3072,3073, 3074"
    --display: help="Set to display outputs", action="store_true"
    --debug: help="Set to print debugging outputs", action="store_true"
    --expTimeHigh: help="Exposure time for stronger flats", type=int, default=30
    --expTimeLow: help="Exposure time for weaker flats", type=int, default=2
    --gapHigh: help="Number of column where gap between physical devices ends", type=int, default=2052
    --gapLow: help="Number of column where gap between physical devices starts", type=int, default=2045
    --maxSigma: help="Maximum sigma for good pixels", type=float, default=6.
    --medianFlatsOut: help="Median Flats output root (without '.fits'). Leave empty for not writing", type=str, default=""
    --nCols: help="Number of columns in images", type=int, default=4096
    --nPixCut: help="Number of rows/columns around the edge of images to ignore", type=int, default=40
    --nRows: help="Number of rows in images", type=int, default=4174
    --outFile: help="Output defect list relative to OBS_PFS_DIR", type=str, default="pfs/defects/2015-12-01/defects.dat"
    --rerun: help="Which rerun directory are the postISRCCD images in if not in root?", type=str, default=""
    --root: help="Path to PFS data directory", type=str, default="/Volumes/My Passport/Users/azuri/spectra/pfs/PFS"
    --spectrograph: help="Spectrograph number for which to create the defect list", type=int, default=1
    --verbose: help="Set to print info outputs", action="store_true"
    --visitLow: help="Lowest visit number to search for flats", type=int, default=6301
    --visitHigh: help="Highest visit number to search for flats", type=int, default=6758

If your PFS data repository is in ~/PFS, you don't want to display the defects
nor write the median flats per exposure time to disk, and all other defaults are
correct, you would simply type::

    genDefectList.py --root=~/PFS

This would create the file $OBS_PFS_DIR/pfs/defects/2015-12-01/defects.dat.

Generating the defects fits files
=================================

To generate the defects fits files needed by the ISR tasks, simply compile obs_pfs.
In $OBS_PFS_DIR, type::

    scons opt=3

This will compile obs_pfs and turn the defects list into the defects fits files.
