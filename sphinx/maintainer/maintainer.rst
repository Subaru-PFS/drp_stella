################
Maintainer notes
################

These notes are intended to assist in maintaining the PFS DRP software and associated products.

Build system
============

The script for building the pipeline, :file:`pfs_pipe2d/bin/install_pfs.sh`, includes several variables
configuring several elements of the install:

- `Conda`_ version, flavors and MD5s
- `LSST conda packages <http://pipelines.lsst.io/install/conda.html>`_
- `Git LFS`_ version and flavor

.. _Conda: http://www.continuum.io/anaconda-overview
.. _Git LFS: http://git-lfs.github.com


Updating LSST stack usage
-------------------------

When the next version of the LSST stack has been released:

- Update variables in :file:`pfs_pipe2d/bin/install_pfs.sh`: ``LSST_SOURCE``, ``LSST_VERSION``.
- Update list of packages/commits in :file:`pfs_pipe2d/lsst_overrides.txt`.
- Clear the Travis-CI cache (see :ref:`maint_travis_cache`).


Travis-CI
=========

You need admin rights to a repo on GitHub in order to activate (or deactivate) Travis-CI for that repo.
Usually this means ensuring that the "Princeton" team has admin access to the repo and that you are a member
of that team.

.. _maint_travis_cache:

Clearing the Cache
------------------

Travis caches the install of the LSST stack (conda, LSST packages, git-lfs, LSST overrides) to provide a small reduction in testing time.
This cache may need to be cleared if the contents are expected to change.

- Go to the appropriate page in Travis-CI, e.g., https://travis-ci.org/Subaru-PFS/obs_pfs
- On the right-hand side, under "More options", select "Caches".
- Click "Delete all repository caches".
