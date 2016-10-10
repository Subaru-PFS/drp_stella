###############
Getting Started
###############

The following notes should allow you to install the LSST stack and the PFS
DRP.  After the installation procedure example commands are given to show you
how to use the pipeline. The commands have been tested on an Arch Linux
machine as well as on Mac OS X.  The main goal of this quick-start guide is to
enable the user to extract and wavelength calibrate an Arc spectrum.

Installation
============

- Install the LSST binary distribution >= 12.0 (https://pipelines.lsst.io/install/conda.html)

  Here it should not matter whether you install anaconda or miniconda. Please
  note that there's only a need to install ``lsst-distrib``, not
  ``lsst-sims``.  Don’t forget to setup the LSST environment as mentioned on
  the website. You will need to do this every time you want to use the
  pipeline in a new shell, so you might want to add these 2 lines to your
  :file:`.bashrc`::

     source activate lsst
     source eups-setups.sh

- The test data are quite large (~38 MB) and are stored in git using git-lfs. We therefore
  need to install git-lfs (https://git-lfs.github.com/). Please follow the instructions at
  https://git-lfs.github.com to installing git-lfs (e.g. using homebrew on
  os/x), then type::

     git lfs install

  There's no need to issue any ``git lfs track`` commands.

- Install the Anaconda future package::

     conda install future

- On Linux, all the anaconda packages were compiled with gcc 4.small and are
  incompatible with modern versions. Therefore, if you are installing the PFS
  DRP on a Linux machine, install gcc in anaconda::

     conda install gcc

- For the convenience of this quick-start guide, create a directory for the PFS-DRP
  repositories and an environment variable to refer to it::

     export PFS_DRP=<your_target_directory>   # e.g. export PFS_DRP=$HOME/PFS
     mkdir -p $PFS_DRP
     cd $PFS_DRP

- Clone the relevant git repositories::

     git clone https://github.com/Subaru-PFS/drp_stella.git
     git clone https://github.com/Subaru-PFS/drp_stella_data.git
     git clone https://github.com/Subaru-PFS/obs_pfs.git

- The next 2 steps are temporary workarounds until fixed:

  - Set :envvar:`EUPS_PKGROOT`::

       export EUPS_PKGROOT="http://sw.lsstcorp.org/eupspkg/"

  - Install Eigen with the unsupported modules::

       eups distrib install eigen 3.2.5.lsst2

- Build and setup the pipeline. There are a few warnings when compiling drp_stella
  which will be silenced in future version and can be safely ignored. Note
  that setting up pipe_drivers and obs_subaru will be obsolete following the
  next LSST binary release::

     cd $PFS_DRP/pipe_drivers
     setup -r .
     scons -Q opt=3 -j8

     cd $PFS_DRP/obs_subaru
     setup -r . -j
     scons -Q opt=3 -j8

     cd $PFS_DRP/obs_pfs
     setup -r .
     scons -Q opt=3 -j8 --filterWarn

     cd $PFS_DRP/drp_stella_data
     setup -r .

     cd $PFS_DRP/drp_stella
     setup -r .
     scons -Q opt=3 -j8 --filterWarn

Usage
=====

Now for using the pipeline.

- Raw test data are in :file:`$DRP_STELLA_DATA_DIR/tests/data/raw/`:

    - 3 Biases: visit numbers 7251-7253;
    - 3 Darks: visit numbers 7291-7293;
    - 1 Flat: visit number 5;
    - 1 Arc: visit number 4.

  Configuration parameters for the pipeline tasks can be set either in config
  files (see :file:`$OBS_PFS_DIR/config/pfs/`) or by passing them on the
  command line (after ``--config``, e.g. ``--config isr.doDark=False``). You can
  list all configuration parameters by appending a ``--show config`` to the
  parameter list.

- First we need to create a directory (actually 2) where we want to store
  pipeline outputs. Let's assume you want to store the pipeline outputs in a
  directory :file:`$HOME/spectra/PFS`. For the convenience of this
  quick-start guide we define another environment variable::

     export PFS_DATA=$HOME/spectra/PFS
     mkdir -p $PFS_DATA/CALIB

- We need to tell the LSST stack which mapper to use. The mapper provides a logical view
  of both the raw data and pipeline outputs, and provides facilities for querying for
  particular data sets. It abstracts away the details of the underlying storage, so we
  can avoid worrying about implementation details::

     echo "lsst.obs.pfs.PfsMapper" > $PFS_DATA/_mapper

- We can now copy/symlink the raw images into the repository and ingest them into a
  registry stored in :file:`$PFS_DATA/registry.sqlite3`.

  The ``--mode link`` parameter tells the pipeline to create symbolic links
  instead of copying the raw images. If you like you can add a ``-L warn``
  parameter to set the log level to only print warnings, making the script
  much less verbose::

     ingestImages.py $PFS_DATA $DRP_STELLA_DATA_DIR/tests/data/raw/*.fits --mode link

- Now that we have our database we can start reducing things. We start with
  creating a master Bias, followed by a Bias-subtracted master Dark. We will
  then create a Bias- and Dark-subtracted master Flat, which we then use to
  identify and trace the apertures of the fiber traces. The fiber traces from
  the Arc image are then extracted and wavelength calibrated.

  The data we want to reduce were observed/simulated on 2015-12-22 on
  spectrograph 2, arm ``r`` (“red”) at site ``S`` (“Summit”).

  The parameter ``--rerun USERNAME/tmp`` (substitute USERNAME with your name)
  specifies where to store temporary pipeline outputs. Please refer to
  https://lsst-web.ncsa.illinois.edu/doxygen/x_masterDoxyDoc/pipe_base.html#pipeBase_argumentParser_rerun
  for a detailed description of the ``rerun`` parameter.

  The ``--id`` parameter specifies the identity of the inputs while the
  parameter ``--calibId`` specifies the output.

  Note the parameter ``--cores 1`` at the end. This parameter is required by
  tasks which are parallelized.  Sometimes running the code in parallel can
  lead to problems (in most cases caused by the 3rd-party libraries used), so
  setting cores to 1 is a safe choice::

     constructBias.py $PFS_DATA --rerun USERNAME/tmp --id field=BIAS dateObs=2015-12-22 arm=r spectrograph=2 --calibId calibVersion=bias calibDate=2015-12-22 arm=r spectrograph=2 --cores 1

- Now that we have a master bias we need to ingest that into our calibration
  database stored in :file:`$PFS_DATA/CALIB/calibRegistry.sqlite3`. The
  parameter ``--validity 180`` specifies that the calibration images are valid
  for 180 days. We will need to repeat this step every time we create a new
  calibration image so that successive tasks can find them::

     genCalibRegistry.py --root $PFS_DATA/CALIB --camera PFS --validity 180


- Now we can create a trimmed and scaled, Bias-subtracted master Dark and
  ingest that into our calibration registry. Again, you need to substitute
  ``USERNAME`` with your name::

     constructDark.py $PFS_DATA --rerun USERNAME/tmp --id field=DARK dateObs=2015-12-22 arm=r spectrograph=2 --calibId calibVersion=dark calibDate=2015-12-22 arm=r spectrograph=2 --cores 1
     genCalibRegistry.py --root $PFS_DATA/CALIB --camera PFS --validity 180

- Having the master Bias and Dark, we can now create our master Flat.
  Currently the master Flat is only used to trace the apertures of the fiber
  traces and to calculate the spatial profile for the optimal extraction.
  Note that for the actual flat-fielding dithered Flats will be used in the
  near future, and this kind of Flat here will be renamed to ``apDef``
  (aperture definition).

  In our data set only visit 5 is a flat, so specifying ``--id visit=5`` is
  all we need to specify for our flat to be found. If you wanted to reduce
  all Flats taken 2015-12-22 for spectrograph 2, red arm, you would replace
  ``visit=5`` with ``field=FLAT arm=r dateObs=2015-12-22 spectrograph=2``::

     reduceFlat.py $PFS_DATA --rerun USERNAME/tmp --id visit=5 --calibId calibVersion=flat calibDate=2015-12-22 arm=r spectrograph=2 --cores 1
     genCalibRegistry.py --root $PFS_DATA/CALIB --camera PFS --validity 180


- In order to extract the arc spectra we first need to identify and trace
  the apertures for each fiber. This is what constructFiberTrace.py does:
      
     constructFiberTrace.py $PFS_DATA --rerun USERNAME/tmp --id field=FLAT dateObs=2015-12-22 arm=r spectrograph=2 --calibId calibVersion=fiberTrace calibDate=2015-12-22 arm=r spectrograph=2
     genCalibRegistry.py --root $PFS_DATA/CALIB --camera PFS --validity 180

- Since we have the Bias and Dark we can now perform the
  Instrumental-Signature Removal (ISR) task for our Arc spectrum (visit=4).
  The program detrend.py will start the ISR task which will subtract the Bias
  and scaled Dark from our Arc image. As mentioned above, flat-fielding is not
  yet supported by the pipeline, but will be in the near future.

  If you want to reduce all Arcs taken 2015-12-22 for spectrograph 2, red arm,
  simply replace ``visit=4`` with ``arm=r spectrograph=2 dateObs=2015-12-22
  field=ARC``. Note that this time you need to specify the output directory as
  we will need the ``postISRCCD`` image in the next step::

     detrend.py $PFS_DATA --rerun USERNAME/tmp --id visit=4

- We now have the ``postISRCCD`` images for our Flat and Arc and can extract and
  wavelength-calibrate our CdHgKrNeXe Arc with the visit number 4::

     reduceArc.py $PFS_DATA --rerun USERNAME/tmp --id visit=4

  This program will write a pfsArm file as described in the data model
  (https://github.com/Subaru-PFS/datamodel/blob/master/datamodel.txt).

Initializing the Pipeline
=========================

During the above, we defined a number of environment variables which are local
to our current session. For convenience, we can create a :file:`setup.sh` file
to easily restore them in a new terminal or after a restart::

   echo "export EUPS_PATH=\"\"" > $DRP_STELLA_DIR/setup.sh
   echo "export EUPS_DIR=\"\"" >> $DRP_STELLA_DIR/setup.sh
   echo "source activate lsst" >> $DRP_STELLA_DIR/setup.sh
   echo "source eups-setups.sh" >> $DRP_STELLA_DIR/setup.sh
   echo "setup -r "$PIPE_DRIVERS_DIR" -j" >> $DRP_STELLA_DIR/setup.sh
   echo "setup -r "$OBS_SUBARU_DIR" -j" >> $DRP_STELLA_DIR/setup.sh
   echo "setup -r "$OBS_PFS_DIR >> $DRP_STELLA_DIR/setup.sh
   echo "setup -r "$DRP_STELLA_DATA_DIR >> $DRP_STELLA_DIR/setup.sh
   echo "setup -r "$DRP_STELLA_DIR >> $DRP_STELLA_DIR/setup.sh

To setup the pipeline again you can then type::

   source $DRP_STELLA_DIR/setup.sh

Note that you will need to set :envvar:`$DRP_STELLA_DIR` manually.
