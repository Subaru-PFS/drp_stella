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
//#include "lsst/afw/image.h"
//#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
//#include "boost/numeric/ublas/matrix.hpp"
//#include "boost/numeric/ublas/matrix_proxy.hpp"
#include <vector>
#include "pfs/drp/stella/FiberTraces.h"
//#include "pfs/drp/stella/blitz.h"
//#include "pfs/drp/stella/Example.h"
#include "pfs/drp/stella/utils/Utils.h"
//#include "pfs/drp/stella/utils/UtilsBlitz.h"
#include "pfs/drp/stella/math/Math.h"
//#include "pfs/drp/stella/math/MathBlitz.h"
#include "pfs/drp/stella/math/SurfaceFitting.h"
#include "pfs/drp/stella/math/CurveFitting.h"
#include "pfs/drp/stella/Controls.h"
#include "pfs/drp/stella/PSF.h"
#include "pfs/drp/stella/Spectra.h"
#include "pfs/drp/stella/SurfaceFit.h"
//#include "pfs/drp/stella/spline.h"
    // Enable ndarray's NumPy typemaps; types are declared in %included files.
//#include "numpy/arrayobject.h"
//#include "ndarray/swig.h"
#include "ndarray/Array.h"
#include "ndarray/swig/eigen.h"
%}

//%init %{
//    import_array();
//%}
%include "ndarray.i"
%declareNumPyConverters(ndarray::Array<size_t,1,1>);
%declareNumPyConverters(ndarray::Array<size_t,2,1>);
%declareNumPyConverters(ndarray::Array<size_t,2,2>);
%declareNumPyConverters(ndarray::Array<unsigned short,1,1>);
%declareNumPyConverters(ndarray::Array<unsigned short,2,1>);
%declareNumPyConverters(ndarray::Array<unsigned short,2,2>);
%declareNumPyConverters(ndarray::Array<unsigned long,1,1>);
%declareNumPyConverters(ndarray::Array<unsigned long,2,1>);
%declareNumPyConverters(ndarray::Array<unsigned long,2,2>);
%declareNumPyConverters(ndarray::Array<unsigned long,3,1>);
%declareNumPyConverters(ndarray::Array<float,1,1>);
%declareNumPyConverters(ndarray::Array<float,2,1>);
%declareNumPyConverters(ndarray::Array<float,2,2>);
%declareNumPyConverters(ndarray::Array<float,3,1>);
%declareNumPyConverters(ndarray::Array<double,1,1>);
%declareNumPyConverters(ndarray::Array<double,2,1>);
%declareNumPyConverters(ndarray::Array<double,2,2>);
%declareNumPyConverters(ndarray::Array<double,3,1>);

%include "lsst/p_lsstSwig.i"

%lsst_exceptions();

%init %{
    import_array();
%}

%include "lsst/base.h"
%include "lsst/pex/config.h"

%import "lsst/afw/geom/geomLib.i"
%import "lsst/afw/image/imageLib.i"

//%include "pfs/drp/stella/blitz.h"
//%include "pfs/drp/stella/Example.h"
//%include "pfs/drp/stella/SurfaceFit.h"
//
// Instantiate addImage* for desired types
//
//%define %addImages(PIXEL_TYPE)
//   %template(addImagesWithBlitz) pfs::drp::stella::addImagesWithBlitz<PIXEL_TYPE>;
//   %template(addImagesWithEigen) pfs::drp::stella::addImagesWithEigen<PIXEL_TYPE>;
//%enddef

//%addImages(double);
//%addImages(int);
//%addImages(float);
//%include <std_shared_ptr.i>
/************************************************************************************************************/
//%shared_ptr(std::vector<float>);
//%shared_ptr(std::vector<double>);
//%shared_ptr(std::vector<unsigned short>);
//%shared_ptr(std::vector<unsigned int>);
//%shared_ptr(std::vector<int>);
//%shared_ptr(std::vector<long>);

//%shared_ptr(ndarray::Array<float, 2, 1>);
//%shared_ptr(ndarray::Array<double, 2, 1>);

%shared_ptr(pfs::drp::stella::FiberTraceFunctionFindingControl);
%shared_ptr(pfs::drp::stella::FiberTraceFunctionControl);
%shared_ptr(pfs::drp::stella::FiberTraceFunction);
%shared_ptr(pfs::drp::stella::FiberTraceProfileFittingControl);
%shared_ptr(pfs::drp::stella::TwoDPSFControl);

%shared_ptr(pfs::drp::stella::FiberTrace<float, unsigned short, float>);
%shared_ptr(pfs::drp::stella::FiberTrace<double, unsigned short, float>);

%shared_ptr(std::vector<PTR(pfs::drp::stella::FiberTrace<float, unsigned short, float>)>);
%shared_ptr(std::vector<PTR(pfs::drp::stella::FiberTrace<double, unsigned short, float>)>);

%shared_ptr(pfs::drp::stella::FiberTraceSet<float, unsigned short, float>);
%shared_ptr(pfs::drp::stella::FiberTraceSet<double, unsigned short, float>);

%shared_ptr(pfs::drp::stella::PSF<float>);
%shared_ptr(pfs::drp::stella::PSF<double>);

%shared_ptr(std::vector<PTR(pfs::drp::stella::PSF<float>)>);
%shared_ptr(std::vector<PTR(pfs::drp::stella::PSF<double>)>);

%shared_ptr(pfs::drp::stella::PSFSet<float>);
%shared_ptr(pfs::drp::stella::PSFSet<double>);

%shared_ptr(std::vector<PTR(pfs::drp::stella::PSFSet<float>)>);
%shared_ptr(std::vector<PTR(pfs::drp::stella::PSFSet<double>)>);

%shared_ptr(pfs::drp::stella::Spectrum<float, unsigned short, float, float>);
%shared_ptr(pfs::drp::stella::Spectrum<double, unsigned short, float, float>);
%shared_ptr(pfs::drp::stella::Spectrum<float, unsigned int, float, float>);
%shared_ptr(pfs::drp::stella::Spectrum<double, unsigned int, float, float>);

%shared_ptr(std::vector<PTR(pfs::drp::stella::Spectrum<float, unsigned short, float, float>)>);
%shared_ptr(std::vector<PTR(pfs::drp::stella::Spectrum<double, unsigned short, float, float>)>);
%shared_ptr(std::vector<PTR(pfs::drp::stella::Spectrum<float, unsigned int, float, float>)>);
%shared_ptr(std::vector<PTR(pfs::drp::stella::Spectrum<double, unsigned int, float, float>)>);

%shared_ptr(pfs::drp::stella::SpectrumSet<float, unsigned short, float, float>);
%shared_ptr(pfs::drp::stella::SpectrumSet<double, unsigned short, float, float>);
%shared_ptr(pfs::drp::stella::SpectrumSet<float, unsigned int, float, float>);
%shared_ptr(pfs::drp::stella::SpectrumSet<double, unsigned int, float, float>);

%shared_ptr(std::vector<PTR(ndarray::Array<float, 2, 1>)>);
%shared_ptr(std::vector<PTR(ndarray::Array<double, 2, 1>)>);

%include "std_vector.i"
// """Shared Pointers (!) to std::vectors"""
//%template(SPVectorF) std::vector<float>;
//%template(SPVectorUS) std::vector<unsigned short>;
//%template(SPVectorUI) std::vector<unsigned int>;
//%template(SPVectorUL) std::vector<unsigned long>;
//%template(SPVectorI) std::vector<int>;
//%template(SPVectorL) std::vector<long>;
//%template(SPVectorD) std::vector<double>;
//%template(VectorPVF) std::vector<PTR(std::vector<float>)>;

%include "pfs/drp/stella/FiberTraces.h"
%template(FTVectorF) std::vector<PTR(pfs::drp::stella::FiberTrace<float, unsigned short, float>)>;
%template(FTVectorD) std::vector<PTR(pfs::drp::stella::FiberTrace<double, unsigned short, float>)>;

%include "ndarray/Array.h"
%template(NDArr1I) ndarray::Array<int, 1, 1>;
%template(NDArr1L) ndarray::Array<long, 1, 1>;
%template(NDArr1UL) ndarray::Array<unsigned long, 1, 1>;
%template(NDArr1F) ndarray::Array<float, 1, 1>;
%template(NDArr1D) ndarray::Array<double, 1, 1>;
%template(NDArr21I) ndarray::Array<int, 2, 1>;
%template(NDArr21L) ndarray::Array<long, 2, 1>;
%template(NDArr21UL) ndarray::Array<unsigned long, 2, 1>;
%template(NDArr21F) ndarray::Array<float, 2, 1>;
%template(NDArr21D) ndarray::Array<double, 2, 1>;
%template(NDArr22I) ndarray::Array<int, 2, 2>;
%template(NDArr22L) ndarray::Array<long, 2, 2>;
%template(NDArr22UL) ndarray::Array<unsigned long, 2, 2>;
%template(NDArr22F) ndarray::Array<float, 2, 2>;
%template(NDArr22D) ndarray::Array<double, 2, 2>;
%template(NDArr31I) ndarray::Array<int, 3, 1>;
%template(NDArr31L) ndarray::Array<long, 3, 1>;
%template(NDArr31UL) ndarray::Array<unsigned long, 3, 1>;
%template(NDArr31F) ndarray::Array<float, 3, 1>;
%template(NDArr31D) ndarray::Array<double, 3, 1>;
%template(NDArr32I) ndarray::Array<int, 3, 2>;
%template(NDArr32L) ndarray::Array<long, 3, 2>;
%template(NDArr32UL) ndarray::Array<unsigned long, 3, 2>;
%template(NDArr32F) ndarray::Array<float, 3, 2>;
%template(NDArr32D) ndarray::Array<double, 3, 2>;
//%template(PNDArrF) PTR(ndarray::Array<float, 2, 1>);
//%template(PNDArrD) PTR(ndarray::Array<double, 2, 1>);
%template(PNDArrVectorF) std::vector<PTR(ndarray::Array<float, 2, 1>)>;
%template(PNDArrVectorD) std::vector<PTR(ndarray::Array<double, 2, 1>)>;
%template(NDArrVectorF) std::vector<ndarray::Array<float, 2, 1>>;
%template(NDArrVectorD) std::vector<ndarray::Array<double, 2, 1>>;

%include "pfs/drp/stella/PSF.h"
%template(PSFVectorF) std::vector<PTR(pfs::drp::stella::PSF<float>)>;
%template(PSFVectorD) std::vector<PTR(pfs::drp::stella::PSF<double>)>;
%template(PSFSetVectorF) std::vector<PTR(pfs::drp::stella::PSFSet<float>)>;
%template(PSFSetVectorD) std::vector<PTR(pfs::drp::stella::PSFSet<double>)>;

%include "pfs/drp/stella/Spectra.h"
%template(SpecVectorF) std::vector<PTR(pfs::drp::stella::Spectrum<float, unsigned short, float, float>)>;
%template(SpecVectorD) std::vector<PTR(pfs::drp::stella::Spectrum<double, unsigned short, float, float>)>;
%template(SpecVectorFI) std::vector<PTR(pfs::drp::stella::Spectrum<float, unsigned int, float, float>)>;
%template(SpecVectorDI) std::vector<PTR(pfs::drp::stella::Spectrum<double, unsigned int, float, float>)>;

%include "pfs/drp/stella/utils/Utils.h"
//%include "pfs/drp/stella/utils/UtilsBlitz.h"
%include "pfs/drp/stella/math/Math.h"
//%include "pfs/drp/stella/math/MathBlitz.h"
//%include "boost/numeric/ublas/matrix.hpp"
//%include "boost/numeric/ublas/matrix_proxy.hpp"
%include "pfs/drp/stella/math/CurveFitting.h"
%include "pfs/drp/stella/math/SurfaceFitting.h"
%include "pfs/drp/stella/Controls.h"
//%include "pfs/drp/stella/blitz.h"
//%include "pfs/drp/stella/spline.h"
%include "pfs/drp/stella/SurfaceFit.h"
%include "lsst/"

%template(PSFSetVectorF) std::vector<PTR(pfs::drp::stella::PSFSet<float>)>;
%template(PSFSetVectorD) std::vector<PTR(pfs::drp::stella::PSFSet<double>)>;

%template(FiberTraceF) pfs::drp::stella::FiberTrace<float, unsigned short, float>;
%template(FiberTraceD) pfs::drp::stella::FiberTrace<double, unsigned short, float>;

%extend pfs::drp::stella::PSF{
    %template(extractPSFFromCenterPositionsFF) extractPSFFromCenterPositions<float, unsigned short, float>;
    %template(extractPSFFromCenterPositionsFD) extractPSFFromCenterPositions<float, unsigned short, double>;
    %template(extractPSFFromCenterPositionsDF) extractPSFFromCenterPositions<double, unsigned short, float>;
    %template(extractPSFFromCenterPositionsDD) extractPSFFromCenterPositions<double, unsigned short, double>;
}

%template(PSFF) pfs::drp::stella::PSF<float>;
%template(PSFD) pfs::drp::stella::PSF<double>;

%template(SpectrumF) pfs::drp::stella::Spectrum<float, unsigned short, float, float>;
%template(SpectrumD) pfs::drp::stella::Spectrum<double, unsigned short, float, float>;
%template(SpectrumFI) pfs::drp::stella::Spectrum<float, unsigned int, float, float>;
%template(SpectrumDI) pfs::drp::stella::Spectrum<double, unsigned int, float, float>;

%template(FiberTraceSetF) pfs::drp::stella::FiberTraceSet<float, unsigned short, float>;
%template(FiberTraceSetD) pfs::drp::stella::FiberTraceSet<double, unsigned short, float>;

%template(PSFSetF) pfs::drp::stella::PSFSet<float>;
%template(PSFSetD) pfs::drp::stella::PSFSet<double>;

%template(SpectrumSetF) pfs::drp::stella::SpectrumSet<float, unsigned short, float, float>;
%template(SpectrumSetD) pfs::drp::stella::SpectrumSet<double, unsigned short, float, float>;
%template(SpectrumSetF) pfs::drp::stella::SpectrumSet<float, unsigned int, float, float>;
%template(SpectrumSetD) pfs::drp::stella::SpectrumSet<double, unsigned int, float, float>;

%template(findAndTraceAperturesF) pfs::drp::stella::math::findAndTraceApertures<float, unsigned short, float>;
%template(findAndTraceAperturesD) pfs::drp::stella::math::findAndTraceApertures<double, unsigned short, float>;

%template(calculate2dPSFPerBinFFF) pfs::drp::stella::math::calculate2dPSFPerBin<float, float, unsigned short, float, float>;
%template(calculate2dPSFPerBinDFF) pfs::drp::stella::math::calculate2dPSFPerBin<double, float, unsigned short, float, float>;
%template(calculate2dPSFPerBinFDF) pfs::drp::stella::math::calculate2dPSFPerBin<float, double, unsigned short, float, float>;
%template(calculate2dPSFPerBinDDF) pfs::drp::stella::math::calculate2dPSFPerBin<double, double, unsigned short, float, float>;
%template(calculate2dPSFPerBinFFD) pfs::drp::stella::math::calculate2dPSFPerBin<float, float, unsigned short, float, double>;
%template(calculate2dPSFPerBinDFD) pfs::drp::stella::math::calculate2dPSFPerBin<double, float, unsigned short, float, double>;
%template(calculate2dPSFPerBinFDD) pfs::drp::stella::math::calculate2dPSFPerBin<float, double, unsigned short, float, double>;
%template(calculate2dPSFPerBinDDD) pfs::drp::stella::math::calculate2dPSFPerBin<double, double, unsigned short, float, double>;

%template(FixU) pfs::drp::stella::math::Fix<unsigned short>;
%template(FixI) pfs::drp::stella::math::Fix<int>;
%template(FixL) pfs::drp::stella::math::Fix<long>;
%template(FixF) pfs::drp::stella::math::Fix<float>;
%template(FixD) pfs::drp::stella::math::Fix<double>;

%template(FixLU) pfs::drp::stella::math::FixL<unsigned short>;
%template(FixLI) pfs::drp::stella::math::FixL<int>;
%template(FixLL) pfs::drp::stella::math::FixL<long>;
%template(FixLF) pfs::drp::stella::math::FixL<float>;
%template(FixLD) pfs::drp::stella::math::FixL<double>;

%template(IntU) pfs::drp::stella::math::Int<unsigned short>;
%template(IntI) pfs::drp::stella::math::Int<int>;
%template(IntL) pfs::drp::stella::math::Int<long>;
%template(IntF) pfs::drp::stella::math::Int<float>;
%template(IntD) pfs::drp::stella::math::Int<double>;

%template(LongU) pfs::drp::stella::math::Long<unsigned short>;
%template(LongI) pfs::drp::stella::math::Long<int>;
%template(LongL) pfs::drp::stella::math::Long<long>;
%template(LongF) pfs::drp::stella::math::Long<float>;
%template(LongD) pfs::drp::stella::math::Long<double>;

%template(FloatU) pfs::drp::stella::math::Float<unsigned short>;
%template(FloatI) pfs::drp::stella::math::Float<int>;
%template(FloatL) pfs::drp::stella::math::Float<long>;
%template(FloatF) pfs::drp::stella::math::Float<float>;
%template(FloatD) pfs::drp::stella::math::Float<double>;

%template(DoubleU) pfs::drp::stella::math::Double<unsigned short>;
%template(DoubleI) pfs::drp::stella::math::Double<int>;
%template(DoubleL) pfs::drp::stella::math::Double<long>;
%template(DoubleF) pfs::drp::stella::math::Double<float>;
%template(DoubleD) pfs::drp::stella::math::Double<double>;

%template(RoundUS) pfs::drp::stella::math::Round<unsigned short>;
%template(RoundUI) pfs::drp::stella::math::Round<unsigned int>;
%template(RoundI) pfs::drp::stella::math::Round<int>;
%template(RoundL) pfs::drp::stella::math::Round<long>;
%template(RoundF) pfs::drp::stella::math::Round<float>;
%template(RoundD) pfs::drp::stella::math::Round<double>;

%template(RoundLUS) pfs::drp::stella::math::RoundL<unsigned short>;
%template(RoundLUI) pfs::drp::stella::math::RoundL<unsigned int>;
%template(RoundLI) pfs::drp::stella::math::RoundL<int>;
%template(RoundLL) pfs::drp::stella::math::RoundL<long>;
%template(RoundLF) pfs::drp::stella::math::RoundL<float>;
%template(RoundLD) pfs::drp::stella::math::RoundL<double>;

//%template(ReplicateU) pfs::drp::stella::math::Replicate<unsigned short>;
//%template(ReplicateI) pfs::drp::stella::math::Replicate<int>;
//%template(ReplicateL) pfs::drp::stella::math::Replicate<long>;
//%template(ReplicateF) pfs::drp::stella::math::Replicate<float>;
//%template(ReplicateD) pfs::drp::stella::math::Replicate<double>;

//%template(ReformU) pfs::drp::stella::math::Reform<unsigned short>;
//%template(ReformI) pfs::drp::stella::math::Reform<int>;
//%template(ReformL) pfs::drp::stella::math::Reform<long>;
//%template(ReformF) pfs::drp::stella::math::Reform<float>;
//%template(ReformD) pfs::drp::stella::math::Reform<double>;

//%template(GetSubArrCopyU) pfs::drp::stella::math::GetSubArrCopy<unsigned short>;
//%template(GetSubArrCopyI) pfs::drp::stella::math::GetSubArrCopy<int>;
//%template(GetSubArrCopyL) pfs::drp::stella::math::GetSubArrCopy<long>;
//%template(GetSubArrCopyF) pfs::drp::stella::math::GetSubArrCopy<float>;
//%template(GetSubArrCopyD) pfs::drp::stella::math::GetSubArrCopy<double>;

//%template(CountPixGTZeroU) pfs::drp::stella::math::CountPixGTZero<unsigned short>;
//%template(CountPixGTZeroI) pfs::drp::stella::math::CountPixGTZero<int>;
//%template(CountPixGTZeroL) pfs::drp::stella::math::CountPixGTZero<long>;
//%template(CountPixGTZeroF) pfs::drp::stella::math::CountPixGTZero<float>;
//%template(CountPixGTZeroD) pfs::drp::stella::math::CountPixGTZero<double>;

//%template(FirstIndexWithValueGEFromU) pfs::drp::stella::math::FirstIndexWithValueGEFrom<unsigned short>;
//%template(FirstIndexWithValueGEFromI) pfs::drp::stella::math::FirstIndexWithValueGEFrom<int>;
//%template(FirstIndexWithValueGEFromL) pfs::drp::stella::math::FirstIndexWithValueGEFrom<long>;
//%template(FirstIndexWithValueGEFromF) pfs::drp::stella::math::FirstIndexWithValueGEFrom<float>;
//%template(FirstIndexWithValueGEFromD) pfs::drp::stella::math::FirstIndexWithValueGEFrom<double>;

//%template(LastIndexWithZeroValueBeforeU) pfs::drp::stella::math::LastIndexWithZeroValueBefore<unsigned short>;
//%template(LastIndexWithZeroValueBeforeI) pfs::drp::stella::math::LastIndexWithZeroValueBefore<int>;
//%template(LastIndexWithZeroValueBeforeL) pfs::drp::stella::math::LastIndexWithZeroValueBefore<long>;
//%template(LastIndexWithZeroValueBeforeF) pfs::drp::stella::math::LastIndexWithZeroValueBefore<float>;
//%template(LastIndexWithZeroValueBeforeD) pfs::drp::stella::math::LastIndexWithZeroValueBefore<double>;

//%template(FirstIndexWithZeroValueFromU) pfs::drp::stella::math::FirstIndexWithZeroValueFrom<unsigned short>;
//%template(FirstIndexWithZeroValueFromI) pfs::drp::stella::math::FirstIndexWithZeroValueFrom<int>;
//%template(FirstIndexWithZeroValueFromL) pfs::drp::stella::math::FirstIndexWithZeroValueFrom<long>;
//%template(FirstIndexWithZeroValueFromF) pfs::drp::stella::math::FirstIndexWithZeroValueFrom<float>;
//%template(FirstIndexWithZeroValueFromD) pfs::drp::stella::math::FirstIndexWithZeroValueFrom<double>;

//%template(MedianU) pfs::drp::stella::math::Median<unsigned short>;
//%template(MedianI) pfs::drp::stella::math::Median<int>;
//%template(MedianL) pfs::drp::stella::math::Median<long>;
//%template(MedianF) pfs::drp::stella::math::Median<float>;
//%template(MedianD) pfs::drp::stella::math::Median<double>;

//%template(MedianVecU) pfs::drp::stella::math::MedianVec<unsigned short>;
//%template(MedianVecI) pfs::drp::stella::math::MedianVec<int>;
//%template(MedianVecL) pfs::drp::stella::math::MedianVec<long>;
//%template(MedianVecF) pfs::drp::stella::math::MedianVec<float>;
//%template(MedianVecD) pfs::drp::stella::math::MedianVec<double>;

//%template(SelectU) pfs::drp::stella::math::Select<unsigned short>;
//%template(SelectI) pfs::drp::stella::math::Select<int>;
//%template(SelectL) pfs::drp::stella::math::Select<long>;
//%template(SelectF) pfs::drp::stella::math::Select<float>;
//%template(SelectD) pfs::drp::stella::math::Select<double>;

//%template(BubbleSortU) pfs::drp::stella::math::BubbleSort<unsigned short>;
//%template(BubbleSortI) pfs::drp::stella::math::BubbleSort<int>;
//%template(BubbleSortL) pfs::drp::stella::math::BubbleSort<long>;
//%template(BubbleSortF) pfs::drp::stella::math::BubbleSort<float>;
//%template(BubbleSortD) pfs::drp::stella::math::BubbleSort<double>;

//%template(UniqI) pfs::drp::stella::math::Uniq<int>;
//%template(UniqL) pfs::drp::stella::math::Uniq<long>;
//%template(UniqF) pfs::drp::stella::math::Uniq<float>;
//%template(UniqD) pfs::drp::stella::math::Uniq<double>;

//%template(MomentI) pfs::drp::stella::math::Moment<int>;
//%template(MomentI) pfs::drp::stella::math::Moment<long>;
//%template(MomentI) pfs::drp::stella::math::Moment<float>;
//%template(MomentI) pfs::drp::stella::math::Moment<double>;

//%template(resizeU) pfs::drp::stella::math::resize<unsigned int>;
//%template(resizeU) pfs::drp::stella::math::resize<int>;
//%template(resizeU) pfs::drp::stella::math::resize<long>;
//%template(resizeU) pfs::drp::stella::math::resize<float>;
//%template(resizeU) pfs::drp::stella::math::resize<double>;

//%template(sortIndicesU) pfs::drp::stella::math::sortIndices<unsigned short>;
//%template(sortIndicesUI) pfs::drp::stella::math::sortIndices<unsigned int>;
//%template(sortIndicesI) pfs::drp::stella::math::sortIndices<int>;
//%template(sortIndicesL) pfs::drp::stella::math::sortIndices<long>;
//%template(sortIndicesF) pfs::drp::stella::math::sortIndices<float>;
//%template(sortIndicesD) pfs::drp::stella::math::sortIndices<double>;

//%template(WriteFitsU) pfs::drp::stella::utils::WriteFits<unsigned short>;
//%template(WriteFitsI) pfs::drp::stella::utils::WriteFits<int>;
//%template(WriteFitsL) pfs::drp::stella::utils::WriteFits<long>;
//%template(WriteFitsF) pfs::drp::stella::utils::WriteFits<float>;
//%template(WriteFitsD) pfs::drp::stella::utils::WriteFits<double>;

//%template(WriteArrayToFile1U) pfs::drp::stella::utils::WriteArrayToFile<unsigned short, 1>;
//%template(WriteArrayToFile1I) pfs::drp::stella::utils::WriteArrayToFile<int, 1>;
//%template(WriteArrayToFile1L) pfs::drp::stella::utils::WriteArrayToFile<long, 1>;
//%template(WriteArrayToFile1F) pfs::drp::stella::utils::WriteArrayToFile<float, 1>;
//%template(WriteArrayToFile1D) pfs::drp::stella::utils::WriteArrayToFile<double, 1>;

//%template(WriteArrayToFile2U) pfs::drp::stella::utils::WriteArrayToFile<unsigned short, 2>;
//%template(WriteArrayToFile2I) pfs::drp::stella::utils::WriteArrayToFile<int, 2>;
//%template(WriteArrayToFile2L) pfs::drp::stella::utils::WriteArrayToFile<long, 2>;
//%template(WriteArrayToFile2F) pfs::drp::stella::utils::WriteArrayToFile<float, 2>;
//%template(WriteArrayToFile2D) pfs::drp::stella::utils::WriteArrayToFile<double, 2>;

//%template(copyBlitzToNdarrayU) pfs::drp::stella::utils::copyBlitzToNdarray<unsigned short>;
//%template(copyBlitzToNdarrayI) pfs::drp::stella::utils::copyBlitzToNdarray<int>;
//%template(copyBlitzToNdarrayL) pfs::drp::stella::utils::copyBlitzToNdarray<long>;
//%template(copyBlitzToNdarrayF) pfs::drp::stella::utils::copyBlitzToNdarray<float>;
//%template(copyBlitzToNdarrayD) pfs::drp::stella::utils::copyBlitzToNdarray<double>;

%template(getPointerMIF) pfs::drp::stella::utils::getPointer<lsst::afw::image::MaskedImage<float, unsigned short, float>>;
%template(getPointerMID) pfs::drp::stella::utils::getPointer<lsst::afw::image::MaskedImage<double, unsigned short, float>>;
%template(getPointerVUS) pfs::drp::stella::utils::getPointer<std::vector<unsigned short>>;
%template(getPointerVUI) pfs::drp::stella::utils::getPointer<std::vector<unsigned int>>;
%template(getPointerVI) pfs::drp::stella::utils::getPointer<std::vector<int>>;
%template(getPointerVF) pfs::drp::stella::utils::getPointer<std::vector<float>>;
%template(getPointerVD) pfs::drp::stella::utils::getPointer<std::vector<double>>;
//%template(getPointerSF) pfs::drp::stella::utils::getPointer<std::vector<pfs::drp::stella::Spectrum<float, unsigned short, float, float>>>;
//%template(getPointerSD) pfs::drp::stella::utils::getPointer<std::vector<pfs::drp::stella::Spectrum<double, unsigned short, float, float>>>;

//%template(get1DBlitzArrayUS) pfs::drp::stella::utils::get1DBlitzArray<unsigned short>;
//%template(get1DBlitzArrayI) pfs::drp::stella::utils::get1DBlitzArray<int>;
//%template(get1DBlitzArrayF) pfs::drp::stella::utils::get1DBlitzArray<float>;
//%template(get1DBlitzArrayD) pfs::drp::stella::utils::get1DBlitzArray<double>;

//%template(get2DBlitzArrayUS) pfs::drp::stella::utils::get2DBlitzArray<unsigned short>;
//%template(get2DBlitzArrayI) pfs::drp::stella::utils::get2DBlitzArray<int>;
//%template(get2DBlitzArrayF) pfs::drp::stella::utils::get2DBlitzArray<float>;
//%template(get2DBlitzArrayD) pfs::drp::stella::utils::get2DBlitzArray<double>;

%template(getRawPointerMIF) pfs::drp::stella::utils::getRawPointer<lsst::afw::image::MaskedImage<float, unsigned short, float>>;
%template(getRawPointerMID) pfs::drp::stella::utils::getRawPointer<lsst::afw::image::MaskedImage<double, unsigned short, float>>;
%template(getRawPointerIF) pfs::drp::stella::utils::getRawPointer<lsst::afw::image::Image<float>>;
%template(getRawPointerII) pfs::drp::stella::utils::getRawPointer<lsst::afw::image::Image<int>>;
%template(getRawPointerIUI) pfs::drp::stella::utils::getRawPointer<lsst::afw::image::Image<unsigned int>>;
%template(getRawPointerID) pfs::drp::stella::utils::getRawPointer<lsst::afw::image::Image<double>>;
%template(getRawPointerFTF) pfs::drp::stella::utils::getRawPointer<pfs::drp::stella::FiberTrace<float, unsigned short, float>>;
%template(getRawPointerFTD) pfs::drp::stella::utils::getRawPointer<pfs::drp::stella::FiberTrace<double, unsigned short, float>>;

%template(copyVectorI) pfs::drp::stella::utils::copy<int>;
%template(copyVectorF) pfs::drp::stella::utils::copy<float>;
%template(copyVectorD) pfs::drp::stella::utils::copy<double>;

%template(indGenUS) pfs::drp::stella::math::indGen<unsigned short>;
%template(indGenUI) pfs::drp::stella::math::indGen<unsigned int>;
%template(indGenI) pfs::drp::stella::math::indGen<int>;
%template(indGenF) pfs::drp::stella::math::indGen<float>;
%template(indGenD) pfs::drp::stella::math::indGen<double>;

%template(indGenNdArrUS) pfs::drp::stella::math::indGenNdArr<unsigned short>;
%template(indGenNdArrUI) pfs::drp::stella::math::indGenNdArr<unsigned int>;
%template(indGenNdArrI) pfs::drp::stella::math::indGenNdArr<int>;
%template(indGenNdArrF) pfs::drp::stella::math::indGenNdArr<float>;
%template(indGenNdArrD) pfs::drp::stella::math::indGenNdArr<double>;

%template(ThinPlateSplineFF) pfs::drp::stella::math::ThinPlateSpline<float, float>;
%template(ThinPlateSplineFD) pfs::drp::stella::math::ThinPlateSpline<float, double>;
%template(ThinPlateSplineDF) pfs::drp::stella::math::ThinPlateSpline<double, float>;
%template(ThinPlateSplineDD) pfs::drp::stella::math::ThinPlateSpline<double, double>;

%template(interpolatePSFThinPlateSplineFF) pfs::drp::stella::math::interpolatePSFThinPlateSpline<float, float>;
%template(interpolatePSFThinPlateSplineDF) pfs::drp::stella::math::interpolatePSFThinPlateSpline<double, float>;
%template(interpolatePSFThinPlateSplineFD) pfs::drp::stella::math::interpolatePSFThinPlateSpline<float, double>;
%template(interpolatePSFThinPlateSplineDD) pfs::drp::stella::math::interpolatePSFThinPlateSpline<double, double>;

%template(interpolatePSFThinPlateSplineFFF) pfs::drp::stella::math::interpolatePSFThinPlateSpline<float, float, float>;
%template(interpolatePSFThinPlateSplineDFF) pfs::drp::stella::math::interpolatePSFThinPlateSpline<double, float, float>;
%template(interpolatePSFThinPlateSplineFDF) pfs::drp::stella::math::interpolatePSFThinPlateSpline<float, double, float>;
%template(interpolatePSFThinPlateSplineDDF) pfs::drp::stella::math::interpolatePSFThinPlateSpline<double, double, float>;
%template(interpolatePSFThinPlateSplineFFD) pfs::drp::stella::math::interpolatePSFThinPlateSpline<float, float, double>;
%template(interpolatePSFThinPlateSplineDFD) pfs::drp::stella::math::interpolatePSFThinPlateSpline<double, float, double>;
%template(interpolatePSFThinPlateSplineFDD) pfs::drp::stella::math::interpolatePSFThinPlateSpline<float, double, double>;
%template(interpolatePSFThinPlateSplineDDD) pfs::drp::stella::math::interpolatePSFThinPlateSpline<double, double, double>;

%template(interpolatePSFSetThinPlateSplineFF) pfs::drp::stella::math::interpolatePSFSetThinPlateSpline<float, float>;
%template(interpolatePSFSetThinPlateSplineDF) pfs::drp::stella::math::interpolatePSFSetThinPlateSpline<double, float>;
%template(interpolatePSFSetThinPlateSplineFD) pfs::drp::stella::math::interpolatePSFSetThinPlateSpline<float, double>;
%template(interpolatePSFSetThinPlateSplineDD) pfs::drp::stella::math::interpolatePSFSetThinPlateSpline<double, double>;

//%template(mkSlitFuncF) pfs::drp::stella::math::MkSlitFunc<float, unsigned short, float>;
//%template(mkSlitFuncD) pfs::drp::stella::math::MkSlitFunc<double, unsigned short, float>;

%template(xCorFF) pfs::drp::stella::math::xCor<float, float>;
%template(xCorDF) pfs::drp::stella::math::xCor<double, float>;
//%template(xCorFD) pfs::drp::stella::math::xCor<float, double>;
//%template(xCorDD) pfs::drp::stella::math::xCor<double, double>;

%template(collapsePSFFF) pfs::drp::stella::math::collapsePSF<float, float>;
%template(collapsePSFDF) pfs::drp::stella::math::collapsePSF<double, float>;
%template(collapsePSFFD) pfs::drp::stella::math::collapsePSF<float, double>;
%template(collapsePSFDD) pfs::drp::stella::math::collapsePSF<double, double>;

%template(collapseFittedPSFFF) pfs::drp::stella::math::collapseFittedPSF<float, float>;
%template(collapseFittedPSFDF) pfs::drp::stella::math::collapseFittedPSF<double, float>;
%template(collapseFittedPSFFD) pfs::drp::stella::math::collapseFittedPSF<float, double>;
%template(collapseFittedPSFDD) pfs::drp::stella::math::collapseFittedPSF<double, double>;

%template(compareCenterPositionsFF) pfs::drp::stella::math::compareCenterPositions<float, float>;
%template(compareCenterPositionsDF) pfs::drp::stella::math::compareCenterPositions<double, float>;
%template(compareCenterPositionsFD) pfs::drp::stella::math::compareCenterPositions<float, double>;
%template(compareCenterPositionsDD) pfs::drp::stella::math::compareCenterPositions<double, double>;

//%template(vecToNdArrayUS) pfs::drp::stella::math::vecToNdArray<unsigned short>;
//%template(vecToNdArrayUI) pfs::drp::stella::math::vecToNdArray<unsigned int>;
//%template(vecToNdArrayUL) pfs::drp::stella::math::vecToNdArray<unsigned long>;
//%template(vecToNdArrayI) pfs::drp::stella::math::vecToNdArray<int>;
//%template(vecToNdArrayF) pfs::drp::stella::math::vecToNdArray<float>;
//%template(vecToNdArrayD) pfs::drp::stella::math::vecToNdArray<double>;

%template(findCenterPositionsOneTraceF) pfs::drp::stella::math::findCenterPositionsOneTrace<float, float>;
%template(findCenterPositionsOneTraceD) pfs::drp::stella::math::findCenterPositionsOneTrace<double, float>;

%template(vectorToNdArrayUS) pfs::drp::stella::math::vectorToNdArray<unsigned short>;
%template(vectorToNdArrayUI) pfs::drp::stella::math::vectorToNdArray<unsigned int>;
%template(vectorToNdArrayUL) pfs::drp::stella::math::vectorToNdArray<unsigned long>;
%template(vectorToNdArrayI) pfs::drp::stella::math::vectorToNdArray<int>;
%template(vectorToNdArrayL) pfs::drp::stella::math::vectorToNdArray<long>;
%template(vectorToNdArrayF) pfs::drp::stella::math::vectorToNdArray<float>;
%template(vectorToNdArrayD) pfs::drp::stella::math::vectorToNdArray<double>;

%template(makeNdArray21I) pfs::drp::stella::math::ndArray21<int>;
%template(makeNdArray21F) pfs::drp::stella::math::ndArray21<float>;
%template(makeNdArray21D) pfs::drp::stella::math::ndArray21<double>;
%template(makeNdArray22I) pfs::drp::stella::math::ndArray22<int>;
%template(makeNdArray22F) pfs::drp::stella::math::ndArray22<float>;
%template(makeNdArray22D) pfs::drp::stella::math::ndArray22<double>;

%template(getSubArrayF) pfs::drp::stella::math::getSubArray<float, int>;
%template(getSubArrayD) pfs::drp::stella::math::getSubArray<double, int>;
%template(getSubArrayI) pfs::drp::stella::math::getSubArray<int, int>;
%template(getSubArrayL) pfs::drp::stella::math::getSubArray<long, int>;

%template(chebyshevFF) pfs::drp::stella::math::chebyshev<float, float>;
%template(chebyshevFD) pfs::drp::stella::math::chebyshev<float, double>;
%template(chebyshevDF) pfs::drp::stella::math::chebyshev<double, float>;
%template(chebyshevDD) pfs::drp::stella::math::chebyshev<double, double>;
   
%template(convertRangeToUnityFF) pfs::drp::stella::math::convertRangeToUnity<float, float>;
%template(convertRangeToUnityFD) pfs::drp::stella::math::convertRangeToUnity<float, double>;
%template(convertRangeToUnityDF) pfs::drp::stella::math::convertRangeToUnity<double, float>;
%template(convertRangeToUnityDD) pfs::drp::stella::math::convertRangeToUnity<double, double>;
