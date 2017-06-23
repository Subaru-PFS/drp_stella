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
#include "lsst/afw.h"
#include "lsst/afw/image/MaskedImage.h"
#include <vector>
#include "pfs/drp/stella/FiberTraces.h"
#include "pfs/drp/stella/utils/Utils.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/math/CurveFitting.h"
#include "pfs/drp/stella/Controls.h"
#include "pfs/drp/stella/Spectra.h"
#include "ndarray/Array.h"
#include "ndarray/swig/eigen.h"
%}

%include "ndarray.i"
%declareNumPyConverters(ndarray::Array<size_t,1,0>);
%declareNumPyConverters(ndarray::Array<size_t,1,1>);
%declareNumPyConverters(ndarray::Array<size_t,2,0>);
%declareNumPyConverters(ndarray::Array<size_t,2,1>);
%declareNumPyConverters(ndarray::Array<size_t,2,2>);
%declareNumPyConverters(ndarray::Array<size_t,3,0>);
%declareNumPyConverters(ndarray::Array<size_t,3,1>);
%declareNumPyConverters(ndarray::Array<size_t,3,2>);
%declareNumPyConverters(ndarray::Array<int,1,0>);
%declareNumPyConverters(ndarray::Array<int,1,1>);
%declareNumPyConverters(ndarray::Array<int,2,0>);
%declareNumPyConverters(ndarray::Array<int,2,1>);
%declareNumPyConverters(ndarray::Array<int,2,2>);
%declareNumPyConverters(ndarray::Array<int,3,0>);
%declareNumPyConverters(ndarray::Array<int,3,1>);
%declareNumPyConverters(ndarray::Array<int,3,2>);
%declareNumPyConverters(ndarray::Array<unsigned short,1,0>);
%declareNumPyConverters(ndarray::Array<unsigned short,1,1>);
%declareNumPyConverters(ndarray::Array<unsigned short,2,0>);
%declareNumPyConverters(ndarray::Array<unsigned short,2,1>);
%declareNumPyConverters(ndarray::Array<unsigned short,2,2>);
%declareNumPyConverters(ndarray::Array<unsigned long,1,0>);
%declareNumPyConverters(ndarray::Array<unsigned long,1,1>);
%declareNumPyConverters(ndarray::Array<unsigned long,2,0>);
%declareNumPyConverters(ndarray::Array<unsigned long,2,1>);
%declareNumPyConverters(ndarray::Array<unsigned long,2,2>);
%declareNumPyConverters(ndarray::Array<unsigned long,3,0>);
%declareNumPyConverters(ndarray::Array<unsigned long,3,1>);
%declareNumPyConverters(ndarray::Array<unsigned long,3,2>);
%declareNumPyConverters(ndarray::Array<float,1,0>);
%declareNumPyConverters(ndarray::Array<float,1,1>);
%declareNumPyConverters(ndarray::Array<float,2,0>);
%declareNumPyConverters(ndarray::Array<float,2,1>);
%declareNumPyConverters(ndarray::Array<float,2,2>);
%declareNumPyConverters(ndarray::Array<float,3,0>);
%declareNumPyConverters(ndarray::Array<float,3,1>);
%declareNumPyConverters(ndarray::Array<float,3,2>);
%declareNumPyConverters(ndarray::Array<double,1,0>);
%declareNumPyConverters(ndarray::Array<double,1,1>);
%declareNumPyConverters(ndarray::Array<double,2,0>);
%declareNumPyConverters(ndarray::Array<double,2,1>);
%declareNumPyConverters(ndarray::Array<double,2,2>);
%declareNumPyConverters(ndarray::Array<double,3,0>);
%declareNumPyConverters(ndarray::Array<double,3,1>);
%declareNumPyConverters(ndarray::Array<double,3,2>);

%include "lsst/p_lsstSwig.i"

%lsst_exceptions();

%init %{
    import_array();
%}

%include "lsst/base.h"
%include "lsst/pex/config.h"

%import "lsst/afw/geom/geomLib.i"
%import "lsst/afw/image/imageLib.i"

%shared_ptr(pfs::drp::stella::FiberTraceFunctionFindingControl);
%shared_ptr(pfs::drp::stella::FiberTraceFunction);
%shared_ptr(pfs::drp::stella::FiberTraceProfileFittingControl);

%shared_ptr(pfs::drp::stella::FiberTrace<float, unsigned short, float>);
%shared_ptr(std::vector<PTR(pfs::drp::stella::FiberTrace<float, unsigned short, float>)>);
%shared_ptr(pfs::drp::stella::FiberTraceSet<float, unsigned short, float>);

%shared_ptr(pfs::drp::stella::Spectrum<float, unsigned short, float, float>);
%shared_ptr(std::vector<PTR(pfs::drp::stella::Spectrum<float, unsigned short, float, float>)>);
%shared_ptr(pfs::drp::stella::SpectrumSet<float, unsigned short, float, float>);
%shared_ptr(std::vector<PTR(ndarray::Array<float, 2, 1>)>);

%include "std_vector.i"
%include "pfs/drp/stella/FiberTraces.h"

%template(FTVectorF) std::vector<PTR(pfs::drp::stella::FiberTrace<float, unsigned short, float>)>;

%include "pfs/drp/stella/Spectra.h"
%include "pfs/drp/stella/Controls.h"

%extend pfs::drp::stella::Spectrum{
    %template(identifyF) identify<float>;
}

%template(SpectrumF) pfs::drp::stella::Spectrum<float, unsigned short, float, float>;
%template(SpecPtrVectorF) std::vector<PTR(pfs::drp::stella::Spectrum<float, unsigned short, float, float>)>;
%template(SpecVectorF) std::vector<pfs::drp::stella::Spectrum<float, unsigned short, float, float>>;

%include "pfs/drp/stella/utils/Utils.h"
%include "pfs/drp/stella/math/Math.h"
%include "pfs/drp/stella/math/CurveFitting.h"

%template(FiberTraceF) pfs::drp::stella::FiberTrace<float, unsigned short, float>;
%template(FiberTraceSetF) pfs::drp::stella::FiberTraceSet<float, unsigned short, float>;

%template(markFiberTraceInMask) pfs::drp::stella::utils::markFiberTraceInMask<float, unsigned short, float>;
%template(SpectrumSetF) pfs::drp::stella::SpectrumSet<float, unsigned short, float, float>;
%template(findAndTraceAperturesF) pfs::drp::stella::math::findAndTraceApertures<float, unsigned short, float>;
%template(sortIndices) pfs::drp::stella::math::sortIndices<unsigned short>;
%template(sortIndices) pfs::drp::stella::math::sortIndices<int>;
%template(getRawPointerFTF) pfs::drp::stella::utils::getRawPointer<pfs::drp::stella::FiberTrace<float, unsigned short, float>>;
%template(indGenF) pfs::drp::stella::math::indGen<float>;
%template(indGenNdArrUS) pfs::drp::stella::math::indGenNdArr<unsigned short>;
%template(indGenNdArrF) pfs::drp::stella::math::indGenNdArr<float>;
%template(findCenterPositionsOneTraceF) pfs::drp::stella::math::findCenterPositionsOneTrace<float, float>;
%template(stretchAndCrossCorrelateSpecFF) pfs::drp::stella::math::stretchAndCrossCorrelateSpec<float, float>;
%template(StretchAndCrossCorrelateSpecResultFF) pfs::drp::stella::math::StretchAndCrossCorrelateSpecResult<float, float>;
%template(poly) pfs::drp::stella::math::Poly<float, double>;
%template(assignITrace) pfs::drp::stella::math::assignITrace< float, unsigned short, float, int, float, 1 >;
%template(unique) pfs::drp::stella::math::unique<long int, 1>;
%template(createLineList) pfs::drp::stella::math::createLineList< float, 1 >;
%template(addFiberTraceToCcdImage) pfs::drp::stella::math::addFiberTraceToCcdImage< float, unsigned short, float, float, float >;
%template(markFiberTraceInMask) pfs::drp::stella::utils::markFiberTraceInMask<float, unsigned short, float>;
%template(where) pfs::drp::stella::math::where< int, int, 1 >;
%template(isMonotonic) pfs::drp::stella::math::isMonotonic< int >;
%template(PolyFit) pfs::drp::stella::math::PolyFit<double>;
%template(calcMinCenMax) pfs::drp::stella::math::calcMinCenMax<float, float>;
%template(findITrace) pfs::drp::stella::math::findITrace<float, unsigned short, float, float, 0>;
%template(ccdToFiberTraceCoordinates) pfs::drp::stella::math::ccdToFiberTraceCoordinates<float, float, unsigned short, float>;
%template(fiberTraceCoordinatesRelativeTo) pfs::drp::stella::math::fiberTraceCoordinatesRelativeTo<float, float, unsigned short, float>;
%template(CoordinatesF) pfs::drp::stella::math::dataXY<float>;
