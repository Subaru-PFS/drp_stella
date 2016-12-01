#############################
PFS DRP Development Practices
#############################

In general, we expect PFS DRP development to follow effectively the same
development workflow and coding standards as LSST Data Management. Refer to
the `LSST Developer Guide`_ in conjunction with the notes below.

.. _LSST Developer Guide: https://developer.lsst.io/

PFS Code Repositories
=====================

All the PFS-specific code is hosted in the `Subaru-PFS`_ organization on
`GitHub`_. You will need to be added to the appropriate team (presumably
“princeton”) by the PFS Project Office. Refer to the `pfs_ipmu_jp account
policy`_, then provide `pfsprojectoffice@ipmu.jp`_ with the details they
request.

Significant packages include:

- `datamodel <http://github.com/Subaru-PFS/datamodel>`_: data model definition.
- `obs_pfs <http://github.com/Subaru-PFS/obs_pfs>`_: interface package for I/O.
- `drp_stella <http://github.com/Subaru-PFS/drp_stella>`_: algorithms for 2D spectra reduction.
- `drp_stella_data <http://github.com/Subaru-PFS/drp_stella_data>`_: test data.
- `pfs_pipe2d <http://github.com/Subaru-PFS/pfs_pipe2d>`_: 2D pipeline build and test scripts.

.. _Subaru-PFS: https://github.com/Subaru-PFS/
.. _GitHub: https://github.com/
.. _pfs_ipmu_jp account policy: http://sumire.pbworks.com/w/page/84391630/pfs_ipmu_jp%20account%20policy#Technicalteammember
.. _pfsprojectoffice@ipmu.jp: mailto:pfsprojectoffice@ipmu.jp

Relationship to LSST Development
================================

The PFS DRP code is built atop the LSST software stack.

Versioning and Installation
---------------------------

LSST follows a six-monthly release cycle. Thus, version 12 of the LSST stack
was released in June 2016, version 13 will follow in December 2016, etc.

PFS development will always target the latest released version of the LSST
stack. It should *not* be necessary to install an unstable snapshot of the
LSST codebase in order to compile, use and develop the PFS code.

Instructions for installing the LSST stack are available from the `LSST
Science Pipelines`_ site. The LSST software is made available through various
distribution channels. Where possible, you are encouraged to use a binary
installation (e.g. `Anaconda packages`_) rather than building from source.

On occasion, PFS work may absolutely require a newer version of an LSST
provided package than is available in an LSST release. In that case, it should
be possible to clone just that individual package from its git repository and
to set it up to work with the release stack.

Any problems arising from interoperability with the LSST stack should be
ticketed in `JIRA`_.

.. _LSST Science Pipelines: https://pipelines.lsst.io/
.. _Anaconda packages: https://pipelines.lsst.io/install/conda.html

Development on LSST
-------------------

Occasionally, it may be necessary to suggest modifications, improvements or
fixes to the LSST codebase as a part of PFS development. Follow the normal
LSST procedures (per their Developer Guide, linked above) for this. This may
require e.g. obtaining an account on LSST's JIRA.

Development Workflow
====================

Our development workflow is closely based on LSST: refer to `their
documentation`_ for details.

In particular, note the following:

.. _sec-jira:

JIRA
----

All work should be ticketed using `PFS JIRA`_ (*not* LSST's JIRA) before it is
undertaken.

When filing JIRA tickets, add them to the correct `JIRA project`_ (click on
the project titles on that list for further information on each project).

Unlike LSST, we do not currently make semantic distinctions between “story”,
“bug”, etc ticket types. Similarly, we do not need to set “team”, “component”
or other exotic fields: a summary and a description is generally enough,
together with links to other relevant issues.

You are encouraged to ticket any potential development, project, bug, etc that
comes to mind. This will avoid things being forgotten. The worst that can
happen is that we close them as “Won't Fix”.

All work *must* be ticketed before it is undertaken.

The normal progression of issue status is “Open” (for a freshly created
ticket), to “in progress”, to “In Review”, to “Reviewed”, to “Done” (when a
ticket has been completed). At any point, a ticket can also be set to “Won't
Fix”.

.. _their documentation: https://developer.lsst.io/processes/workflow.html
.. _PFS JIRA: https://pfs.ipmu.jp/jira/
.. _JIRA project: https://pfs.ipmu.jp/jira/secure/BrowseProjects.jspa#all

Sprinting
---------

We work on a two-week sprint cycle. At the start of the period, we plan the
work to be undertaken during the sprint; at the end, we review what was
achieved and assess if we could do better next time.

Sometimes, it may be necessary to carry out work which hasn't been explicitly
assigned to the sprint. However, this should be the exception rather than the
rule: please think carefully, and, if possible, discuss it with others, before
plunging in to unscheduled work.

Workflow
--------

As in LSST, development should be carried out on a branch named after the JIRA
ticket. This, when working on ticket “INFRA-26”, work should be committed to
the branch ``tickets/INFRA-26`` in one or more repositories.

Note that PFS does *not* provide a continuous integration / Jenkins server.
You may ignore LSST guidelines that refer to that facility. However, you
*must* ensure that all unit tests pass before merging code. Reviewers and
developers are both responsible for checking this.

When work has been completed, ask for it to be reviewed by setting the ticket
status to “In Review” and naming one or more reviewers. Make it clear to them
exactly what code they are supposed to be reviewing (e.g. by linking to the
relevant branches on GitHub).

Do not merge to the ``master`` branch until the reviewer is satisfied with the
work and the :ref:`dev-ci` passes. It is not required to agree with or implement every suggestion the
reviewer makes, but, when disagreeing, make this clear and iterate with the
reviewer to a solution you are both happy with.

Before merging, please use ``git rebase`` to re-write history for clarity.
Refer to the `LSST guidelines`_

When merging to ``master``, use the ``--no-ff`` option to git to generate a
merge commit.

For issues that do not involve changes to deployed code, review is optional:
please use your discretion as to whether a second pair of eyes would help
ensure that the work has been done properly.

.. _LSST guidelines: https://developer.lsst.io/processes/workflow.html#appendix-commit-organization-best-practices

Coding Standards
================

Follow the LSST DM Style Guides for `C++`_ and `Python`_. In general, follow
LSST's guidelines on use of external libraries (`Boost`_, `Astropy`_, etc). We
specifically except the use of `Eigen's unsupported modules`_ from this
policy: their use is specifically permitted.

.. _C++: https://developer.lsst.io/coding/cpp_style_guide.html
.. _Python: https://developer.lsst.io/coding/python_style_guide.html
.. _Boost: https://developer.lsst.io/coding/using_boost.html
.. _Astropy: https://developer.lsst.io/coding/using_astropy.html
.. _Eigen's unsupported modules: https://developer.lsst.io/coding/using_eigen.html


.. _dev-ci:

Continuous Integration
======================

An integration test is available in the `pfs_pipe2d`_ package. The integration test consists of more than just
running the unit tests of individual packages, but exercises the actual commands users will employ to reduce
data, ensuring that the individual packages still work together to achieve what we have declared the pipeline
is capable of doing. Therefore, this test should be run before merging any work to the ``master`` branch;
this ensures that the ``master`` branch always works. Timing will vary according to resource availability and
the number of cores used, but the test typically takes about half an hour to run.

The test can be run in one of two ways: on the developer's own system at the command-line, or on `Travis-CI`_.
The Travis-CI method is recommended because it is requires little effort, uses external resources and provides
a visible demonstration to the team that the code works.

.. _pfs_pipe2d: http://github.com/Subaru-PFS/pfs_pipe2d
.. _Travis-CI: http://travis-ci.org


Command-line use
----------------

You can run the integration test using :file:`pfs_integration_test.sh` on the command-line. Note that this
command only runs the integration test, and does *not* build or install the code. It is the user's
responsibility to set up the environment (e.g., sourcing the appropriate :file:`pfs_setups.sh` file; see
:ref:`user-script-install`), including the packages to be tested.

Unless you've modified the ``drp_stella_data`` package as part of your work (in which case you need to use
the ``-b <BRANCH>`` flag), the only argument you need is the directory under which to operate. For example::

  /path/to/pfs_pipe2d/bin/pfs_integration_test.sh /data/pfs

The full usage information for :file:`pfs_integration_test.sh` is::

    Exercise the PFS 2D pipeline code
    
    Usage: /path/to/pfs_pipe2d/bin/pfs_integration_test.sh [-b <BRANCH>] [-r <RERUN>] [-d DIRNAME] [-c CORES] [-n] <PREFIX>
    
        -b <BRANCH> : branch of drp_stella_data to use
        -r <RERUN> : rerun name to use (default: 'integration')
        -d <DIRNAME> : directory name to give data repo (default: 'INTEGRATION')
        -c <CORES> : number of cores to use (default: 1)
        -n : don't cleanup temporary products
        <PREFIX> : directory under which to operate


Travis use
----------

The integration test will run automatically under `Travis-CI`_ when GitHub pull requests are issued for any
of the following products:

- pfs_pipe2d
- drp_stella
- drp_stella_data
- obs_pfs
- datamodel

After issuing a pull request, all you need to do is wait for the test to finish successfully. You'll see a
yellow circle while the test runs; you can click on the "Details" button to see the build output. If it
finished successfully, you'll get a green circle and the statement "All checks have passed"; then you're
clear to merge after review. If the Travis check fails, you'll get a red circle, which signals that you need
to fix your contribution. In that case, have a look at the Travis log for clues as to what the problem might
be.

As you push new commits to the pull request, Travis will automatically trigger new integration tests. Because
Travis is triggered on GitHub pull requests, you should ensure you have pushed your work on a common ticket
branch to all appropriate repos before making pull requests. If you want to signal Travis `not to
automatically test a commit`_, add the text ``[ci skip]`` to your commit message.

Unfortunately, due to Travis resource limitations, only 4 MB of logs can be generated. We therefore trap the
build and test output and display only the last 100 lines when the process completes. If this makes it
difficult to determine what's causing your build to fail, you can always run the integration test on the
command-line of your own system.

.. _Travis-CI: http://travis-ci.org
.. _not to automatically test a commit: http://docs.travis-ci.com/user/customizing-the-build#Skipping-a-build
