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
work. It is not required to agree with or implement every suggestion the
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
