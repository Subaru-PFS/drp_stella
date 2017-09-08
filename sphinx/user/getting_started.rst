###############
Getting Started
###############

The following notes should allow you to install the LSST stack and the PFS
DRP.  After the installation procedure example commands are given to show you
how to use the pipeline. The main goal of this quick-start guide is to
enable the user to extract and wavelength calibrate an Arc spectrum.

The installation can be done using a :ref:`user-script-install`, or :ref:`user-manual-install`. You are
encouraged to use the :ref:`user-script-install` because it is intended to be convenient and we would like to
ensure the pipeline is simple to install.

.. _user-script-install:

Script-based install
====================

The script-based install was constructed using the :ref:`user-manual-install`, and is intended to simplify
the installation process. Please `report`_ any problems using the script so that we can improve it.

The script has currently only been tested on Linux; we hope to verify its functionality on Mac OSX soon.

.. _report: mailto:reduction-software@pfs.ipmu.jp


Installation
------------

The installation script, :file:`install_pfs.sh`, lives in the `pfs_pipe2d`_ package.
It will automatically install the LSST stack and the PFS DRP.

Unless you're a developer wanting to build a particular feature, the only argument you need is the directory
under which to install. For example::

    /path/to/pfs_pipe2d/bin/install_pfs.sh /data/pfs

The full usage information for :file:`install_pfs.sh` is::

    Install the PFS 2D pipeline.
    
    Usage: /path/to/pfs_pipe2d/bin/install_pfs.sh [-b <BRANCH>] <PREFIX>
    
        -b <BRANCH> : name of branch on PFS to install
        <PREFIX> : directory in which to install

.. _pfs_pipe2d: http://github.com/Subaru-PFS/pfs_pipe2d


Initializing the Pipeline
-------------------------

The :file:`install_pfs.sh` script writes a script called :file:`pfs_setups.sh` in the install directory that
you can use to set up your environment to use the PFS pipeline. For example, if you installed the PFS
pipeline in :file:`/data/pfs`, then you need to::

    . /data/pfs/pfs_setups.sh

You may want to add this to your :file:`~/.bashrc` file.


.. _user-manual-install:


Manual install
==============

Installation
------------

The following commands have been tested on an Arch Linux machine as well as on Mac OS X.

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
(or follow the steps in `Initializing the pipeline`).
    
- The test data are quite large (~38 MB) and are stored in git using git-lfs. We therefore
  need to install git-lfs (https://git-lfs.github.com/). Please follow the instructions at
  https://git-lfs.github.com to installing git-lfs (e.g. using homebrew on
  os/x), then type::

     git lfs install

  (note that this doesn't do the install -- you already did that --, it modifies
    your :file:`.gitconfig` file to use `git lfs`)

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
  next LSST binary release. Note that at the moment we need to checkout a certain
  commit in pipe_drivers::

     cd $PFS_DRP/ctrl_pool
     setup -r . -j
     scons -Q opt=3 -j8

     cd $PFS_DRP/pipe_drivers
     git checkout e677033910672
     setup -r . -j
     scons -Q opt=3 -j8

     cd $PFS_DRP/display_ds9
     setup -r . -j
     scons -Q opt=3 -j8

     cd $PFS_DRP/obs_pfs
     setup -r .
     scons -Q opt=3 -j8 --filterWarn
     eups declare -r $OBS_PFS_DIR -c

     cd $PFS_DRP/drp_stella_data
     setup -r .
     eups declare -r $DRP_STELLA_DATA_DIR -c
     
     cd $PFS_DRP/datamodel
     setup -r .
     scons -Q opt=3 -j8 --filterWarn
     eups declare -r $DATAMODEL_DIR -c

     cd $PFS_DRP/drp_stella
     setup -r .
     scons -Q opt=3 -j8 --filterWarn


Initializing the Pipeline
-------------------------

During the above, we defined a number of environment variables which are local
to our current session. For convenience, we can create a :file:`setup.sh` file
to easily restore them in a new terminal or after a restart::

   echo "source activate lsst-v12_1" >> $PFS_DRP/setup.sh
   echo "export OPAL_PREFIX="$OPAL_PREFIX >> $PFS_DRP/setup.sh
   echo "source "$OPAL_PREFIX"/bin/eups-setups.sh" >> $PFS_DRP/setup.sh
   echo "setup -r "$DATAMODEL_DIR >> $PFS_DRP/setup.sh
   echo "setup -r "$DRP_STELLA_DIR >> $PFS_DRP/setup.sh
   echo "setup -r "$CTRL_POOL_DIR" -j" >> $PFS_DRP/setup.sh
   echo "setup -r "$PIPE_DRIVERS_DIR" -j" >> $PFS_DRP/setup.sh
   echo "setup -r "$DISPLAY_DS9_DIR" -j" >> $PFS_DRP/setup.sh

To initialize the pipeline again next time you want to use it, type::

   source $PFS_DRP/setup.sh

Note that next time you will need to set :envvar:`$PFS_DRP` manually before
executing this command, or simply replace :envvar:`$PFS_DRP` with the appropriate directory.


Usage
=====

Now for using the pipeline.

- Raw test data are in :file:`$DRP_STELLA_DATA_DIR/tests/data/raw/`:

    - 5 Biases: visit numbers 7251-7255;
    - 3 Darks: visit numbers 7291-7293;
    - 9 dithered Flats: visit number 104-112;
    - 1 Arc: visit number 103.

  Configuration parameters for the pipeline tasks can be set either in config
  files (see :file:`$OBS_PFS_DIR/config/pfs/`) or by passing them on the
  command line (after ``--config``, e.g. ``--config isr.doDark=False``). You can
  list all configuration parameters by appending a ``--show config`` to the
  parameter list.

- First we need to create a directory (actually 2) where we want to store
  pipeline outputs. Let's assume you want to store the pipeline outputs in a
  directory :file:`$HOME/PFS/Data`. For the convenience of this
  quick-start guide we define another environment variable::

     export PFS_DATA=$HOME/PFS/Data
     mkdir -p $PFS_DATA/CALIB

- We need to tell the LSST stack which mapper to use. The mapper provides a logical view
  of both the raw data and pipeline outputs, and provides facilities for querying for
  particular data sets. It abstracts away the details of the underlying storage, so we
  can avoid worrying about implementation details::

     echo "lsst.obs.pfs.PfsMapper" > $PFS_DATA/_mapper
     
- We also need a Pfs State File which can conveniently be found in
  `$DRP_STELLA_DATA_DIR/tests/data/PFS/pfsState`. Please copy this directory into your
  `$PFS_DATA` directory:

    cp -pr $DRP_STELLA_DATA_DIR/tests/data/PFS/pfsState $PFS_DATA/

- We can now copy/symlink the raw images into the repository and ingest them into a
  registry stored in :file:`$PFS_DATA/registry.sqlite3`.

  The ``--mode link`` parameter tells the pipeline to create symbolic links
  instead of copying the raw images. If you like you can add a ``-L warn``
  parameter to set the log level to only print warnings, making the script
  much less verbose::

     ingestImages.py $PFS_DATA $DRP_STELLA_DATA_DIR/tests/data/raw/*.fits --mode link

- We also need a file describing the configuration of the cobras. For now we will
  use the one with all ra and dec values equal to 0.0 which has (as a special case)
  a pfsConfigId == 0x0
  
     cp -r $DRP_STELLA_DATA_DIR/tests/data/PFS/pfsState $PFS_DATA

- Now that we have our database we can start reducing our data. We start with
  creating a trimmed master Bias, followed by a Bias-subtracted master Dark. We
  will then create a Bias- and Dark-subtracted master Flat, which we then use to
  identify and trace the apertures of the fiber traces. The fiber traces from
  the Arc image are then extracted and wavelength calibrated.

  The data we want to reduce were observed/simulated on 2015-12-22/2016-11-11 on
  spectrograph 1, arm ``r`` (“red”) at site ``J`` (“JHU”)/``F`` (“Simulations”).

  The parameter ``--rerun $whoami/tmp``
  specifies where to store temporary pipeline outputs. Please refer to
  https://lsst-web.ncsa.illinois.edu/doxygen/x_masterDoxyDoc/pipe_base.html#pipeBase_argumentParser_rerun
  for a detailed description of the ``rerun`` parameter.

  The ``--id`` parameter specifies the identity of the inputs while the
  parameter ``--calibId`` specifies the output.

  Note the parameter ``--batch-type none`` at the end. This parameter is required by
  tasks which are parallelized.  Sometimes running the code in parallel can
  lead to problems (in most cases caused by the 3rd-party libraries used), so
  specifying ``--batch-type none`` is a safe choice::

     constructBias.py $PFS_DATA --rerun $whoami/tmp --id field=BIAS dateObs=2015-12-22 arm=r spectrograph=1 --calibId calibVersion=bias calibDate=2015-12-22 arm=r spectrograph=1 --batch-type none

- Now that we have a master bias we need to ingest that into our calibration
  database stored in :file:`$PFS_DATA/CALIB/calibRegistry.sqlite3`. The
  parameter ``--validity 360`` specifies that the calibration images are valid
  for 360 days. We will need to repeat this step every time we create a new
  calibration image so that successive tasks can find them::

     genCalibRegistry.py --root $PFS_DATA/CALIB --validity 360

- Now we can create a trimmed and scaled, Bias-subtracted master Dark and
  ingest that into our calibration registry::

     constructDark.py $PFS_DATA --rerun $whoami/tmp --id field=DARK dateObs=2015-12-22 arm=r spectrograph=1 --calibId calibVersion=dark calibDate=2015-12-22 arm=r spectrograph=1 --batch-type none
     genCalibRegistry.py --root $PFS_DATA/CALIB --validity 360

- In order to extract the arc spectra we first need to identify and trace
  the apertures for each fiber. This is what constructFiberTrace.py does.
  In our data set visit 104 is the only non-dithered flat, so specifying
  ``--id visit=104`` is all we need to specify for our flat to be found::

     constructFiberTrace.py $PFS_DATA --rerun $whoami/tmp --id visit=104 --calibId calibVersion=fiberTrace calibDate=2016-11-11 arm=r spectrograph=1 --batch-type none
     genCalibRegistry.py --root $PFS_DATA/CALIB --validity 360

- We can now construct our master Flat from all the dithered Flats, which have
  the visit numbers 104-112::
      
     constructFiberFlat.py $PFS_DATA --rerun $whoami/tmp --id visit=104..112 --calibId calibVersion=flat calibDate=2016-11-11 arm=r spectrograph=1 --batch-type none
     genCalibRegistry.py --root $PFS_DATA/CALIB --validity 360
     
- Since we have the master Bias, Dark, and Flat we can now perform the
  Instrumental-Signature Removal (ISR) task for our Arc spectrum (visit=103).
  The program detrend.py will start the ISR task which will subtract the Bias
  and scaled Dark from our Arc image and flatfield it.

  If you want to reduce all Arcs taken 2016-11-11 for spectrograph 1, red arm,
  simply replace ``visit=103`` with ``arm=r spectrograph=1 dateObs=2016-11-11
  field=ARC``::

     constructArc.py $PFS_DATA --rerun $whoami/calibs --id visit=58 --calibId calibVersion=arc calibDate=2015-12-19 arm=r spectrograph=2 -c isr.doBias=True isr.doDark=True isr.doFlat=False isr.doLinearize=False --batch-type none
     genCalibRegistry.py --root $PFS_DATA/CALIB --camera PFS --validity 360

- We now have our master Arc and can extract and wavelength-calibrate our
  CdHgKrNeXe Arc with the visit number 58::

     reduceArcRefSpec.py $PFS_DATA --rerun $whoami/tmp --id visit=58

  The output file created by reduceArcRefSpec.py is a pfsArm file as described
  in the data model.
