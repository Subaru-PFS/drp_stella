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

- Install conda from http://conda.pydata.org/miniconda.html (or install anaconda).
- Install the LSST binary distribution >= 12.1::

    conda config --add channels http://conda.lsst.codes/stack/0.12.1
    conda create --name lsst-v12_1 python=2
    conda install -n lsst-v12_1 anaconda
    conda install -n lsst-v12_1 lsst-distrib

  Don’t forget to setup the LSST environment. You will need to do this every time
  you want to use the pipeline in a new shell, so you might want to add this line
  to your :file:`.bashrc` (or follow the steps at the bottom `Initializing the
  pipeline`)::

    source activate lsst-v12_1

- You need one workaround;  set the environment variable OPAL_PREFIX, e.g. in bash::
    
    export OPAL_PREFIX=XXX/envs/lsst-v12_1
    
where XXX is the root of your conda installation (check that it has an envs directory!).
Again, for your convenience you might want to add this line to your :file:`.bashrc` as well
(or follow the steps at in `Initializing the pipeline`).
    
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
  repositories and an environment variable to refer to it, plus an environment variable
  that contains your user name::

     export whoami=$(whoami)
     export PFS_DRP=<your_target_directory>   # e.g. export PFS_DRP=$HOME/PFS
     mkdir -p $PFS_DRP
     cd $PFS_DRP

- Clone the relevant git repositories::

     git clone git@github.com:lsst/ctrl_pool
     git clone git@github.com:lsst/pipe_drivers
     git clone git@github.com:lsst/display_ds9
     git clone https://github.com/Subaru-PFS/drp_stella.git
     git clone https://github.com/Subaru-PFS/drp_stella_data.git
     git clone https://github.com/Subaru-PFS/obs_pfs.git
     git clone https://github.com/Subaru-PFS/datamodel.git

- Build and setup the pipeline. There are a few warnings when compiling drp_stella
  which will be silenced in future version and can be safely ignored. Setting up
  ctrl_pool, pipe_drivers, and display_ds9 will be obsolete following the
  next LSST binary release. Note that we need to checkout a certain commit in
  pipe_drivers and at least for the moment certain branches in drp_stella and
  datamodel::

     cd $PFS_DRP/ctrl_pool
     setup -r .
     scons -Q opt=3 -j8

     cd $PFS_DRP/pipe_drivers
     git checkout e677033910672
     setup -r .
     scons -Q opt=3 -j8

     cd $PFS_DRP/display_ds9
     setup -r .
     scons -Q opt=3 -j8

     cd $PFS_DRP/obs_pfs
     setup -r .
     scons -Q opt=3 -j8 --filterWarn
     eups declare -r $OBS_PFS_DIR -c

     cd $PFS_DRP/drp_stella_data
     setup -r .
     eups declare -r $DRP_STELLA_DATA_DIR -c
     
     cd $PFS_DRP/datamodel
     git checkout tickets/PIPE2D-68
     setup -r .
     scons -Q opt=3 -j8 --filterWarn
     eups declare -r $DATAMODEL_DIR -c

     cd $PFS_DRP/drp_stella
     git checkout tickets/PIPE2D-48
     setup -r .
     scons -Q opt=3 -j8 --filterWarn


Initializing the Pipeline
=========================

During the above, we defined a number of environment variables which are local
to our current session. For convenience, we can create a :file:`setup.sh` file
to easily restore them in a new terminal or after a restart::

   echo "source activate lsst-v12_1" >> $PFS_DRP/setup.sh
   echo "export OPAL_PREFIX="$OPAL_PREFIX >> $PFS_DRP/setup.sh
   echo "source "$OPAL_PREFIX"/bin/eups-setups.sh" >> $PFS_DRP/setup.sh
   echo "setup -r "$DRP_STELLA_DIR >> $PFS_DRP/setup.sh
   echo "setup -r "$CTRL_POOL_DIR" -j" >> $PFS_DRP/setup.sh
   echo "setup -r "$PIPE_DRIVERS_DIR" -j" >> $PFS_DRP/setup.sh
   echo "setup -r "$DISPLAY_DS9_DIR" -j" >> $PFS_DRP/setup.sh

To initialize the pipeline now and again next time you want to use it, type::

   source $PFS_DRP/setup.sh

Note that next time you will need to set :envvar:`$PFS_DRP` manually before
executing this command, or simply replace :envvar:`$PFS_DRP` with the appropriate directory.


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

  The parameter ``--rerun $whoami/tmp``
  specifies where to store temporary pipeline outputs. Please refer to
  https://lsst-web.ncsa.illinois.edu/doxygen/x_masterDoxyDoc/pipe_base.html#pipeBase_argumentParser_rerun
  for a detailed description of the ``rerun`` parameter.

  The ``--id`` parameter specifies the identity of the inputs while the
  parameter ``--calibId`` specifies the output.

  Note the parameter ``--batch-type none`` at the end. This parameter is required by
  tasks which are parallelized.  Sometimes running the code in parallel can
  lead to problems (in most cases caused by the 3rd-party libraries used), so
  specifying ``--batch-type none`` is a safe choice. Note that we also add the
  config parameter ``doLinearize=False`` as we don't yet have the table needed
  for the linearizer::

     constructBias.py $PFS_DATA --rerun $whoami/calibs --id field=BIAS dateObs=2015-12-22 arm=r spectrograph=2 --calibId calibVersion=bias calibDate=2015-12-22 arm=r spectrograph=2 -c isr.doLinearize=False --batch-type none

- Now that we have a master bias we need to ingest that into our calibration
  database stored in :file:`$PFS_DATA/CALIB/calibRegistry.sqlite3`. The
  parameter ``--validity 180`` specifies that the calibration images are valid
  for 180 days. We will need to repeat this step every time we create a new
  calibration image so that successive tasks can find them::

     genCalibRegistry.py --root $PFS_DATA/CALIB --camera PFS --validity 180

- Now we can create a trimmed and scaled, Bias-subtracted master Dark and
  ingest that into our calibration registry::

     constructDark.py $PFS_DATA --rerun $whoami/calibs --id field=DARK dateObs=2015-12-22 arm=r spectrograph=2 --calibId calibVersion=dark calibDate=2015-12-22 arm=r spectrograph=2 -c isr.doBias=True isr.doLinearize=False --batch-type none
     genCalibRegistry.py --root $PFS_DATA/CALIB --camera PFS --validity 180

- In order to extract the arc spectra we first need to identify and trace
  the apertures for each fiber. This is what constructFiberTrace.py does.
  In our data set only visit 5 is a flat, so specifying ``--id visit=5`` is
  all we need to specify for our flat to be found::
      
     constructFiberTrace.py $PFS_DATA_DIR --rerun $whoami/calibs --id visit=5 --calibId calibVersion=fiberTrace calibDate=2015-12-22 arm=r spectrograph=2 -c isr.doBias=True isr.doDark=True isr.doFlat=False isr.doLinearize=False --batch-type none
     genCalibRegistry.py --root $PFS_DATA/CALIB --camera PFS --validity 180
     
- Since we have the master Bias and Dark we can now perform the
  Instrumental-Signature Removal (ISR) task for our Arc spectrum (visit=4).
  The program detrend.py will start the ISR task which will subtract the Bias
  and scaled Dark from our Arc image. Flat-fielding is not yet supported by the
  pipeline, but will be in the near future.

  If you want to reduce all Arcs taken 2015-12-22 for spectrograph 2, red arm,
  simply replace ``visit=4`` with ``arm=r spectrograph=2 dateObs=2015-12-22
  field=ARC``::

     detrend.py $PFS_DATA --rerun $whoami/tmp --id visit=4 -c isr.doBias=True isr.doDark=True isr.doFlat=False isr.doLinearize=False

- We now have the ``postISRCCD`` image for our Arc and can extract and
  wavelength-calibrate our CdHgKrNeXe Arc with the visit number 4::

     reduceArc.py $PFS_DATA --rerun $whoami/tmp --id visit=4

  Note that this program does not currently write an output file but will do so
  very soon.
