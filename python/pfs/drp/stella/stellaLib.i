// -*- lsst-c++ -*-

%define stellaLib_DOCSTRING
"
Interface to Stella
"
%enddef

%feature("autodoc", "1");
%module(package="pfs.drp.stella", docstring=stellaLib_DOCSTRING) stellaLib

%{
#define PY_ARRAY_UNIQUE_SYMBOL PFS_DRP_STELLA_NUMPY_ARRAY_API
#include "numpy/arrayobject.h"
#include "ndarray/swig.h"
#include "lsst/pex/logging.h"
#include "lsst/afw.h"
#include "pfs/drp/stella/FiberTraces.h"
#include "pfs/drp/stella/Example.h"
%}

%include "lsst/p_lsstSwig.i"

%lsst_exceptions();

%include "lsst/base.h"
%include "lsst/pex/config.h"

%import "lsst/afw/geom/geomLib.i"
%import "lsst/afw/image/imageLib.i"

%include "pfs/drp/stella/Example.h"
//
// Instantiate addImage* for desired types
//
%define %addImages(PIXEL_TYPE)
   %template(addImagesWithBlitz) pfs::drp::stella::addImagesWithBlitz<PIXEL_TYPE>;
   %template(addImagesWithEigen) pfs::drp::stella::addImagesWithEigen<PIXEL_TYPE>;
%enddef

%addImages(double);
%addImages(int);
%addImages(float);

/************************************************************************************************************/

%shared_ptr(pfs::drp::stella::FiberTrace);
%shared_ptr(pfs::drp::stella::ImageFiberTrace);

%include "pfs/drp/stella/FiberTraces.h"
