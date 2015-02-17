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
#include "lsst/afw/image/MaskedImage.h"
#include <vector>
#include "pfs/drp/stella/FiberTraces.h"
#include "pfs/drp/stella/blitz.h"
#include "pfs/drp/stella/Example.h"
#include "pfs/drp/stella/utils/Utils.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/Controls.h"
#include "pfs/drp/stella/PSF.h"
#include "pfs/drp/stella/Spectra.h"
#include "pfs/drp/stella/SurfaceFit.h"
#include "pfs/drp/stella/spline.h"
%}

%include "lsst/p_lsstSwig.i"

%lsst_exceptions();

%init %{
    import_array();
%}

%include "lsst/base.h"
%include "lsst/pex/config.h"

%import "lsst/afw/geom/geomLib.i"
%import "lsst/afw/image/imageLib.i"

%include "pfs/drp/stella/blitz.h"
%include "pfs/drp/stella/Example.h"
//%include "pfs/drp/stella/SurfaceFit.h"
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
%shared_ptr(std::vector<float>);
%shared_ptr(std::vector<double>);
%shared_ptr(std::vector<unsigned short>);
%shared_ptr(std::vector<unsigned int>);
%shared_ptr(std::vector<int>);
%shared_ptr(std::vector<long>);

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

%shared_ptr(pfs::drp::stella::PSF<float, unsigned short, float, float>);
%shared_ptr(pfs::drp::stella::PSF<double, unsigned short, float, float>);

%shared_ptr(std::vector<PTR(pfs::drp::stella::PSF<float, unsigned short, float, float>)>);
%shared_ptr(std::vector<PTR(pfs::drp::stella::PSF<double, unsigned short, float, float>)>);

%shared_ptr(pfs::drp::stella::PSFSet<float, unsigned short, float, float>);
%shared_ptr(pfs::drp::stella::PSFSet<double, unsigned short, float, float>);

%shared_ptr(std::vector<PTR(pfs::drp::stella::PSFSet<float, unsigned short, float, float>)>);
%shared_ptr(std::vector<PTR(pfs::drp::stella::PSFSet<double, unsigned short, float, float>)>);

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

%include "pfs/drp/stella/FiberTraces.h"
%include "std_vector.i"
%template(FTVectorF) std::vector<PTR(pfs::drp::stella::FiberTrace<float, unsigned short, float>)>;
%template(FTVectorD) std::vector<PTR(pfs::drp::stella::FiberTrace<double, unsigned short, float>)>;
%template(VectorPVF) std::vector<PTR(std::vector<float>)>;

%include "pfs/drp/stella/PSF.h"
%template(PSFVectorF) std::vector<PTR(pfs::drp::stella::PSF<float, unsigned short, float, float>)>;
%template(PSFVectorD) std::vector<PTR(pfs::drp::stella::PSF<double, unsigned short, float, float>)>;
%template(PSFSetVectorF) std::vector<PTR(pfs::drp::stella::PSFSet<float, unsigned short, float, float>)>;
%template(PSFSetVectorD) std::vector<PTR(pfs::drp::stella::PSFSet<double, unsigned short, float, float>)>;

%include "pfs/drp/stella/Spectra.h"
%template(SpecVectorF) std::vector<PTR(pfs::drp::stella::Spectrum<float, unsigned short, float, float>)>;
%template(SpecVectorD) std::vector<PTR(pfs::drp::stella::Spectrum<double, unsigned short, float, float>)>;
%template(SpecVectorFI) std::vector<PTR(pfs::drp::stella::Spectrum<float, unsigned int, float, float>)>;
%template(SpecVectorDI) std::vector<PTR(pfs::drp::stella::Spectrum<double, unsigned int, float, float>)>;

%include "pfs/drp/stella/utils/Utils.h"
%include "pfs/drp/stella/math/Math.h"
%include "pfs/drp/stella/Controls.h"
%include "pfs/drp/stella/blitz.h"
%include "pfs/drp/stella/spline.h"
%include "pfs/drp/stella/SurfaceFit.h"

# """Shared Pointers (!) to std::vectors"""
%template(SPVectorF) std::vector<float>;
#%template(SPVectorD) std::vector<double>;
%template(SPVectorUS) std::vector<unsigned short>;
%template(SPVectorUI) std::vector<unsigned int>;
%template(SPVectorI) std::vector<int>;
%template(SPVectorL) std::vector<long>;
%template(PSFSetVectorF) std::vector<PTR(pfs::drp::stella::PSFSet<float, unsigned short, float, float>)>;
%template(PSFSetVectorD) std::vector<PTR(pfs::drp::stella::PSFSet<double, unsigned short, float, float>)>;

%template(FiberTraceF) pfs::drp::stella::FiberTrace<float, unsigned short, float>;
%template(FiberTraceD) pfs::drp::stella::FiberTrace<double, unsigned short, float>;

%template(PSFF) pfs::drp::stella::PSF<float, unsigned short, float, float>;
%template(PSFD) pfs::drp::stella::PSF<double, unsigned short, float, float>;

%template(SpectrumF) pfs::drp::stella::Spectrum<float, unsigned short, float, float>;
%template(SpectrumD) pfs::drp::stella::Spectrum<double, unsigned short, float, float>;
%template(SpectrumFI) pfs::drp::stella::Spectrum<float, unsigned int, float, float>;
%template(SpectrumDI) pfs::drp::stella::Spectrum<double, unsigned int, float, float>;

%template(FiberTraceSetF) pfs::drp::stella::FiberTraceSet<float, unsigned short, float>;
%template(FiberTraceSetD) pfs::drp::stella::FiberTraceSet<double, unsigned short, float>;

%template(PSFSetF) pfs::drp::stella::PSFSet<float, unsigned short, float, float>;
%template(PSFSetD) pfs::drp::stella::PSFSet<double, unsigned short, float, float>;

%template(SpectrumSetF) pfs::drp::stella::SpectrumSet<float, unsigned short, float, float>;
%template(SpectrumSetD) pfs::drp::stella::SpectrumSet<double, unsigned short, float, float>;
%template(SpectrumSetF) pfs::drp::stella::SpectrumSet<float, unsigned int, float, float>;
%template(SpectrumSetD) pfs::drp::stella::SpectrumSet<double, unsigned int, float, float>;

%template(findAndTraceAperturesF) pfs::drp::stella::math::findAndTraceApertures<float, unsigned short, float>;
%template(findAndTraceAperturesD) pfs::drp::stella::math::findAndTraceApertures<double, unsigned short, float>;

%template(calculate2dPSFPerBinF) pfs::drp::stella::math::calculate2dPSFPerBin<float, unsigned short, float, float>;
%template(calculate2dPSFPerBinD) pfs::drp::stella::math::calculate2dPSFPerBin<double, unsigned short, float, float>;

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

%template(ReplicateU) pfs::drp::stella::math::Replicate<unsigned short>;
%template(ReplicateI) pfs::drp::stella::math::Replicate<int>;
%template(ReplicateL) pfs::drp::stella::math::Replicate<long>;
%template(ReplicateF) pfs::drp::stella::math::Replicate<float>;
%template(ReplicateD) pfs::drp::stella::math::Replicate<double>;

%template(ReformU) pfs::drp::stella::math::Reform<unsigned short>;
%template(ReformI) pfs::drp::stella::math::Reform<int>;
%template(ReformL) pfs::drp::stella::math::Reform<long>;
%template(ReformF) pfs::drp::stella::math::Reform<float>;
%template(ReformD) pfs::drp::stella::math::Reform<double>;

%template(GetSubArrCopyU) pfs::drp::stella::math::GetSubArrCopy<unsigned short>;
%template(GetSubArrCopyI) pfs::drp::stella::math::GetSubArrCopy<int>;
%template(GetSubArrCopyL) pfs::drp::stella::math::GetSubArrCopy<long>;
%template(GetSubArrCopyF) pfs::drp::stella::math::GetSubArrCopy<float>;
%template(GetSubArrCopyD) pfs::drp::stella::math::GetSubArrCopy<double>;

%template(CountPixGTZeroU) pfs::drp::stella::math::CountPixGTZero<unsigned short>;
%template(CountPixGTZeroI) pfs::drp::stella::math::CountPixGTZero<int>;
%template(CountPixGTZeroL) pfs::drp::stella::math::CountPixGTZero<long>;
%template(CountPixGTZeroF) pfs::drp::stella::math::CountPixGTZero<float>;
%template(CountPixGTZeroD) pfs::drp::stella::math::CountPixGTZero<double>;

%template(FirstIndexWithValueGEFromU) pfs::drp::stella::math::FirstIndexWithValueGEFrom<unsigned short>;
%template(FirstIndexWithValueGEFromI) pfs::drp::stella::math::FirstIndexWithValueGEFrom<int>;
%template(FirstIndexWithValueGEFromL) pfs::drp::stella::math::FirstIndexWithValueGEFrom<long>;
%template(FirstIndexWithValueGEFromF) pfs::drp::stella::math::FirstIndexWithValueGEFrom<float>;
%template(FirstIndexWithValueGEFromD) pfs::drp::stella::math::FirstIndexWithValueGEFrom<double>;

%template(LastIndexWithZeroValueBeforeU) pfs::drp::stella::math::LastIndexWithZeroValueBefore<unsigned short>;
%template(LastIndexWithZeroValueBeforeI) pfs::drp::stella::math::LastIndexWithZeroValueBefore<int>;
%template(LastIndexWithZeroValueBeforeL) pfs::drp::stella::math::LastIndexWithZeroValueBefore<long>;
%template(LastIndexWithZeroValueBeforeF) pfs::drp::stella::math::LastIndexWithZeroValueBefore<float>;
%template(LastIndexWithZeroValueBeforeD) pfs::drp::stella::math::LastIndexWithZeroValueBefore<double>;

%template(FirstIndexWithZeroValueFromU) pfs::drp::stella::math::FirstIndexWithZeroValueFrom<unsigned short>;
%template(FirstIndexWithZeroValueFromI) pfs::drp::stella::math::FirstIndexWithZeroValueFrom<int>;
%template(FirstIndexWithZeroValueFromL) pfs::drp::stella::math::FirstIndexWithZeroValueFrom<long>;
%template(FirstIndexWithZeroValueFromF) pfs::drp::stella::math::FirstIndexWithZeroValueFrom<float>;
%template(FirstIndexWithZeroValueFromD) pfs::drp::stella::math::FirstIndexWithZeroValueFrom<double>;

%template(MedianU) pfs::drp::stella::math::Median<unsigned short>;
%template(MedianI) pfs::drp::stella::math::Median<int>;
%template(MedianL) pfs::drp::stella::math::Median<long>;
%template(MedianF) pfs::drp::stella::math::Median<float>;
%template(MedianD) pfs::drp::stella::math::Median<double>;

%template(MedianVecU) pfs::drp::stella::math::MedianVec<unsigned short>;
%template(MedianVecI) pfs::drp::stella::math::MedianVec<int>;
%template(MedianVecL) pfs::drp::stella::math::MedianVec<long>;
%template(MedianVecF) pfs::drp::stella::math::MedianVec<float>;
%template(MedianVecD) pfs::drp::stella::math::MedianVec<double>;

%template(SelectU) pfs::drp::stella::math::Select<unsigned short>;
%template(SelectI) pfs::drp::stella::math::Select<int>;
%template(SelectL) pfs::drp::stella::math::Select<long>;
%template(SelectF) pfs::drp::stella::math::Select<float>;
%template(SelectD) pfs::drp::stella::math::Select<double>;

%template(BubbleSortU) pfs::drp::stella::math::BubbleSort<unsigned short>;
%template(BubbleSortI) pfs::drp::stella::math::BubbleSort<int>;
%template(BubbleSortL) pfs::drp::stella::math::BubbleSort<long>;
%template(BubbleSortF) pfs::drp::stella::math::BubbleSort<float>;
%template(BubbleSortD) pfs::drp::stella::math::BubbleSort<double>;

%template(UniqI) pfs::drp::stella::math::Uniq<int>;
%template(UniqL) pfs::drp::stella::math::Uniq<long>;
%template(UniqF) pfs::drp::stella::math::Uniq<float>;
%template(UniqD) pfs::drp::stella::math::Uniq<double>;

%template(MomentI) pfs::drp::stella::math::Moment<int>;
%template(MomentI) pfs::drp::stella::math::Moment<long>;
%template(MomentI) pfs::drp::stella::math::Moment<float>;
%template(MomentI) pfs::drp::stella::math::Moment<double>;

#%template(resizeU) pfs::drp::stella::math::resize<unsigned int>;
#%template(resizeU) pfs::drp::stella::math::resize<int>;
#%template(resizeU) pfs::drp::stella::math::resize<long>;
#%template(resizeU) pfs::drp::stella::math::resize<float>;
#%template(resizeU) pfs::drp::stella::math::resize<double>;

#%template(sortIndicesU) pfs::drp::stella::math::sortIndices<unsigned short>;
#%template(sortIndicesUI) pfs::drp::stella::math::sortIndices<unsigned int>;
#%template(sortIndicesI) pfs::drp::stella::math::sortIndices<int>;
#%template(sortIndicesL) pfs::drp::stella::math::sortIndices<long>;
#%template(sortIndicesF) pfs::drp::stella::math::sortIndices<float>;
#%template(sortIndicesD) pfs::drp::stella::math::sortIndices<double>;

%template(WriteFitsU) pfs::drp::stella::utils::WriteFits<unsigned short>;
%template(WriteFitsI) pfs::drp::stella::utils::WriteFits<int>;
%template(WriteFitsL) pfs::drp::stella::utils::WriteFits<long>;
%template(WriteFitsF) pfs::drp::stella::utils::WriteFits<float>;
%template(WriteFitsD) pfs::drp::stella::utils::WriteFits<double>;

%template(WriteArrayToFile1U) pfs::drp::stella::utils::WriteArrayToFile<unsigned short, 1>;
%template(WriteArrayToFile1I) pfs::drp::stella::utils::WriteArrayToFile<int, 1>;
%template(WriteArrayToFile1L) pfs::drp::stella::utils::WriteArrayToFile<long, 1>;
%template(WriteArrayToFile1F) pfs::drp::stella::utils::WriteArrayToFile<float, 1>;
%template(WriteArrayToFile1D) pfs::drp::stella::utils::WriteArrayToFile<double, 1>;

%template(WriteArrayToFile2U) pfs::drp::stella::utils::WriteArrayToFile<unsigned short, 2>;
%template(WriteArrayToFile2I) pfs::drp::stella::utils::WriteArrayToFile<int, 2>;
%template(WriteArrayToFile2L) pfs::drp::stella::utils::WriteArrayToFile<long, 2>;
%template(WriteArrayToFile2F) pfs::drp::stella::utils::WriteArrayToFile<float, 2>;
%template(WriteArrayToFile2D) pfs::drp::stella::utils::WriteArrayToFile<double, 2>;

%template(copyBlitzToNdarrayU) pfs::drp::stella::utils::copyBlitzToNdarray<unsigned short>;
%template(copyBlitzToNdarrayI) pfs::drp::stella::utils::copyBlitzToNdarray<int>;
%template(copyBlitzToNdarrayL) pfs::drp::stella::utils::copyBlitzToNdarray<long>;
%template(copyBlitzToNdarrayF) pfs::drp::stella::utils::copyBlitzToNdarray<float>;
%template(copyBlitzToNdarrayD) pfs::drp::stella::utils::copyBlitzToNdarray<double>;

%template(getPointerMIF) pfs::drp::stella::utils::getPointer<lsst::afw::image::MaskedImage<float, unsigned short, float>>;
%template(getPointerMID) pfs::drp::stella::utils::getPointer<lsst::afw::image::MaskedImage<double, unsigned short, float>>;
%template(getPointerVUS) pfs::drp::stella::utils::getPointer<std::vector<unsigned short>>;
%template(getPointerVUI) pfs::drp::stella::utils::getPointer<std::vector<unsigned int>>;
%template(getPointerVI) pfs::drp::stella::utils::getPointer<std::vector<int>>;
%template(getPointerVF) pfs::drp::stella::utils::getPointer<std::vector<float>>;
%template(getPointerVD) pfs::drp::stella::utils::getPointer<std::vector<double>>;
#%template(getPointerSF) pfs::drp::stella::utils::getPointer<std::vector<pfs::drp::stella::Spectrum<float, unsigned short, float, float>>>;
#%template(getPointerSD) pfs::drp::stella::utils::getPointer<std::vector<pfs::drp::stella::Spectrum<double, unsigned short, float, float>>>;

%template(get1DBlitzArrayUS) pfs::drp::stella::utils::get1DBlitzArray<unsigned short>;
%template(get1DBlitzArrayI) pfs::drp::stella::utils::get1DBlitzArray<int>;
%template(get1DBlitzArrayF) pfs::drp::stella::utils::get1DBlitzArray<float>;
%template(get1DBlitzArrayD) pfs::drp::stella::utils::get1DBlitzArray<double>;

%template(get2DBlitzArrayUS) pfs::drp::stella::utils::get2DBlitzArray<unsigned short>;
%template(get2DBlitzArrayI) pfs::drp::stella::utils::get2DBlitzArray<int>;
%template(get2DBlitzArrayF) pfs::drp::stella::utils::get2DBlitzArray<float>;
%template(get2DBlitzArrayD) pfs::drp::stella::utils::get2DBlitzArray<double>;

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

