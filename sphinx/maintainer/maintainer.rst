################
Maintainer notes
################

These notes are intended to assist in maintaining the PFS DRP software and associated products.

Build status
============

+-----------------+---------------------------+
| pfs_pipe2d      | |pfs-pipe2d-status|_      |
+-----------------+---------------------------+
| drp_stella      | |drp-stella-status|_      |
+-----------------+---------------------------+
| obs_pfs         | |obs-pfs-status|_         |
+-----------------+---------------------------+
| datamodel       | |datamodel-status|_       |
+-----------------+---------------------------+
| drp_stella_data | |drp-stella-data-status|_ |
+-----------------+---------------------------+
| Docs            | |docs-status|_            |
+-----------------+---------------------------+

.. |pfs-pipe2d-status| image:: https://travis-ci.org/Subaru-PFS/pfs_pipe2d.svg?branch=master
   :alt: not available
.. _pfs-pipe2d-status: https://travis-ci.org/Subaru-PFS/pfs_pipe2d
.. |drp-stella-status| image:: https://travis-ci.org/Subaru-PFS/drp_stella.svg?branch=master
   :alt: not available
.. _drp-stella-status: https://travis-ci.org/Subaru-PFS/drp_stella
.. |obs-pfs-status| image:: https://travis-ci.org/Subaru-PFS/obs_pfs.svg?branch=master
   :alt: not available
.. _obs-pfs-status: https://travis-ci.org/Subaru-PFS/obs_pfs
.. |datamodel-status| image:: https://travis-ci.org/Subaru-PFS/datamodel.svg?branch=master
   :alt: not available
.. _datamodel-status: https://travis-ci.org/Subaru-PFS/datamodel
.. |drp-stella-data-status| image:: https://travis-ci.org/Subaru-PFS/drp_stella_data.svg?branch=master
   :alt: not available
.. _drp-stella-data-status: https://travis-ci.org/Subaru-PFS/drp_stella_data
.. |docs-status| image:: https://readthedocs.org/projects/pfs-2d-pipeline/badge/?version=latest
   :alt: not available
.. _docs-status: https://readthedocs.org/projects/pfs-2d-pipeline


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
