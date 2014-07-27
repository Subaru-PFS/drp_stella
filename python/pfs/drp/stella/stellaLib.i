// -*- lsst-c++ -*-

%define stellaLib_DOCSTRING
"
Interface to Stella
"
%enddef

%feature("autodoc", "1");
%module(package="pfs.drp.stella", docstring=stellaLib_DOCSTRING) stellaLib

%{
#include "lsst/pex/logging.h"
#include "lsst/afw.h"
#include "pfs/drp/stella/Example.h"
%}

%include "lsst/p_lsstSwig.i"

%lsst_exceptions();

%include "lsst/base.h"
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

%addImages(float);
