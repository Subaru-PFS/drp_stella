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
#include "pfs/drp/stella/utils/Utils.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/math/SurfaceFitting.h"
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

%include "pfs/drp/stella/FiberTraces.h"
%template(FTVectorF) std::vector<PTR(pfs::drp::stella::FiberTrace<float, unsigned short, float>)>;
%template(FTVectorD) std::vector<PTR(pfs::drp::stella::FiberTrace<double, unsigned short, float>)>;

%include "pfs/drp/stella/Spectra.h"
%include "pfs/drp/stella/Controls.h"

%extend pfs::drp::stella::Spectrum{
    %template(identifyF) identify<float>;
    %template(identifyD) identify<double>;
}

%template(SpectrumF) pfs::drp::stella::Spectrum<float, unsigned short, float, float>;
%template(SpectrumD) pfs::drp::stella::Spectrum<double, unsigned short, float, float>;

%template(SpecPtrVectorF) std::vector<PTR(pfs::drp::stella::Spectrum<float, unsigned short, float, float>)>;
%template(SpecPtrVectorD) std::vector<PTR(pfs::drp::stella::Spectrum<double, unsigned short, float, float>)>;
%template(SpecVectorF) std::vector<pfs::drp::stella::Spectrum<float, unsigned short, float, float>>;
%template(SpecVectorD) std::vector<pfs::drp::stella::Spectrum<double, unsigned short, float, float>>;

%include "pfs/drp/stella/utils/Utils.h"
%include "pfs/drp/stella/math/Math.h"
%include "pfs/drp/stella/math/CurveFitting.h"
%include "pfs/drp/stella/math/SurfaceFitting.h"
%include "lsst/"

%template(FiberTraceF) pfs::drp::stella::FiberTrace<float, unsigned short, float>;
%template(FiberTraceD) pfs::drp::stella::FiberTrace<double, unsigned short, float>;

%template(FiberTraceSetF) pfs::drp::stella::FiberTraceSet<float, unsigned short, float>;
%template(FiberTraceSetD) pfs::drp::stella::FiberTraceSet<double, unsigned short, float>;

%template(markFiberTraceInMask) pfs::drp::stella::utils::markFiberTraceInMask<float, unsigned short, float>;

%template(SpectrumSetF) pfs::drp::stella::SpectrumSet<float, unsigned short, float, float>;
%template(SpectrumSetD) pfs::drp::stella::SpectrumSet<double, unsigned short, float, float>;

%template(findAndTraceAperturesF) pfs::drp::stella::math::findAndTraceApertures<float, unsigned short, float>;
%template(findAndTraceAperturesD) pfs::drp::stella::math::findAndTraceApertures<double, unsigned short, float>;

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

%template(sortIndices) pfs::drp::stella::math::sortIndices<unsigned short>;
%template(sortIndices) pfs::drp::stella::math::sortIndices<unsigned int>;
%template(sortIndices) pfs::drp::stella::math::sortIndices<int>;
%template(sortIndices) pfs::drp::stella::math::sortIndices<long>;
%template(sortIndices) pfs::drp::stella::math::sortIndices<float>;
%template(sortIndices) pfs::drp::stella::math::sortIndices<double>;

%template(getRawPointerFTF) pfs::drp::stella::utils::getRawPointer<pfs::drp::stella::FiberTrace<float, unsigned short, float>>;

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

%template(ThinPlateSplineBaseFF) pfs::drp::stella::math::ThinPlateSplineBase<float, float>;
%template(ThinPlateSplineBaseFD) pfs::drp::stella::math::ThinPlateSplineBase<float, double>;
%template(ThinPlateSplineBaseDF) pfs::drp::stella::math::ThinPlateSplineBase<double, float>;
%template(ThinPlateSplineBaseDD) pfs::drp::stella::math::ThinPlateSplineBase<double, double>;

%template(ThinPlateSplineFF) pfs::drp::stella::math::ThinPlateSpline<float, float>;
%template(ThinPlateSplineFD) pfs::drp::stella::math::ThinPlateSpline<float, double>;
%template(ThinPlateSplineDF) pfs::drp::stella::math::ThinPlateSpline<double, float>;
%template(ThinPlateSplineDD) pfs::drp::stella::math::ThinPlateSpline<double, double>;

%template(ThinPlateSplineChiSquareFF) pfs::drp::stella::math::ThinPlateSplineChiSquare<float, float>;
%template(ThinPlateSplineChiSquareFD) pfs::drp::stella::math::ThinPlateSplineChiSquare<float, double>;
%template(ThinPlateSplineChiSquareDF) pfs::drp::stella::math::ThinPlateSplineChiSquare<double, float>;
%template(ThinPlateSplineChiSquareDD) pfs::drp::stella::math::ThinPlateSplineChiSquare<double, double>;

%template(xCorFF) pfs::drp::stella::math::xCor<float, float>;
%template(xCorDF) pfs::drp::stella::math::xCor<double, float>;

%template(findCenterPositionsOneTraceF) pfs::drp::stella::math::findCenterPositionsOneTrace<float, float>;
%template(findCenterPositionsOneTraceD) pfs::drp::stella::math::findCenterPositionsOneTrace<double, float>;

%template(vectorToNdArrayUS) pfs::drp::stella::math::vectorToNdArray<unsigned short>;
%template(vectorToNdArrayUI) pfs::drp::stella::math::vectorToNdArray<unsigned int>;
%template(vectorToNdArrayUL) pfs::drp::stella::math::vectorToNdArray<unsigned long>;
%template(vectorToNdArrayI) pfs::drp::stella::math::vectorToNdArray<int>;
%template(vectorToNdArrayL) pfs::drp::stella::math::vectorToNdArray<long>;
%template(vectorToNdArrayF) pfs::drp::stella::math::vectorToNdArray<float>;
%template(vectorToNdArrayD) pfs::drp::stella::math::vectorToNdArray<double>;

%template(ndArrayToVectorUS) pfs::drp::stella::math::ndArrayToVector<unsigned short>;
%template(ndArrayToVectorUI) pfs::drp::stella::math::ndArrayToVector<unsigned int>;
%template(ndArrayToVectorUL) pfs::drp::stella::math::ndArrayToVector<unsigned long>;
%template(ndArrayToVectorI) pfs::drp::stella::math::ndArrayToVector<int>;
%template(ndArrayToVectorL) pfs::drp::stella::math::ndArrayToVector<long>;
%template(ndArrayToVectorF) pfs::drp::stella::math::ndArrayToVector<float>;
%template(ndArrayToVectorD) pfs::drp::stella::math::ndArrayToVector<double>;

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

%template(getZMinMaxInRangeF) pfs::drp::stella::math::getZMinMaxInRange<float>;
%template(getZMinMaxInRangeD) pfs::drp::stella::math::getZMinMaxInRange<double>;

%template(createRectangularGridD) pfs::drp::stella::math::createRectangularGrid<double>;

%template(createPolarGridD) pfs::drp::stella::math::createPolarGrid<double>;

%template(calculateChiSquareF) pfs::drp::stella::math::calculateChiSquare<float>;
%template(calculateChiSquareD) pfs::drp::stella::math::calculateChiSquare<double>;

%template(numberToString_dotToUnderscoreF) pfs::drp::stella::utils::numberToString_dotToUnderscore<float>;
%template(numberToString_dotToUnderscoreD) pfs::drp::stella::utils::numberToString_dotToUnderscore<double>;

%template(getDataInRangeF) pfs::drp::stella::math::getDataInRange<float>;
%template(getDataInRangeD) pfs::drp::stella::math::getDataInRange<double>;

%template(SpectrumBackgroundF) pfs::drp::stella::math::SpectrumBackground<float>;
%template(SpectrumBackgroundD) pfs::drp::stella::math::SpectrumBackground<double>;

%template(linFitBevingtonFF) pfs::drp::stella::math::LinFitBevingtonNdArray<float, float>;
%template(linFitBevingtonDF) pfs::drp::stella::math::LinFitBevingtonNdArray<double, float>;
%template(linFitBevingtonFD) pfs::drp::stella::math::LinFitBevingtonNdArray<float, double>;
%template(linFitBevingtonDD) pfs::drp::stella::math::LinFitBevingtonNdArray<double, double>;

%template(stretchAndCrossCorrelateSpecFF) pfs::drp::stella::math::stretchAndCrossCorrelateSpec<float, float>;
%template(stretchAndCrossCorrelateSpecDF) pfs::drp::stella::math::stretchAndCrossCorrelateSpec<double, float>;
%template(stretchAndCrossCorrelateSpecFD) pfs::drp::stella::math::stretchAndCrossCorrelateSpec<float, double>;
%template(stretchAndCrossCorrelateSpecDD) pfs::drp::stella::math::stretchAndCrossCorrelateSpec<double, double>;

%template(StretchAndCrossCorrelateSpecResultFF) pfs::drp::stella::math::StretchAndCrossCorrelateSpecResult<float, float>;
%template(StretchAndCrossCorrelateSpecResultDD) pfs::drp::stella::math::StretchAndCrossCorrelateSpecResult<double, double>;
%template(StretchAndCrossCorrelateSpecResultFD) pfs::drp::stella::math::StretchAndCrossCorrelateSpecResult<float, double>;
%template(StretchAndCrossCorrelateSpecResultDF) pfs::drp::stella::math::StretchAndCrossCorrelateSpecResult<double, float>;

%template(StretchAndCrossCorrelateResultF) pfs::drp::stella::math::StretchAndCrossCorrelateResult<float>;
%template(StretchAndCrossCorrelateResultD) pfs::drp::stella::math::StretchAndCrossCorrelateResult<double>;

%template(stretchAndCrossCorrelateF) pfs::drp::stella::math::stretchAndCrossCorrelate<float>;
%template(stretchAndCrossCorrelateD) pfs::drp::stella::math::stretchAndCrossCorrelate<double>;

%template(crossCorrelateF) pfs::drp::stella::math::crossCorrelate<float>;
%template(crossCorrelateD) pfs::drp::stella::math::crossCorrelate<double>;

%template(stretchF) pfs::drp::stella::math::stretch<float>;
%template(stretchD) pfs::drp::stella::math::stretch<double>;

%template(poly) pfs::drp::stella::math::Poly<float, float>;
%template(poly) pfs::drp::stella::math::Poly<double, float>;
%template(poly) pfs::drp::stella::math::Poly<float, double>;
%template(poly) pfs::drp::stella::math::Poly<double, double>;

%template(assignITrace) pfs::drp::stella::math::assignITrace< float, unsigned short, float, int, float, 0 >;
%template(assignITrace) pfs::drp::stella::math::assignITrace< float, unsigned short, float, int, double, 0 >;
%template(assignITrace) pfs::drp::stella::math::assignITrace< float, unsigned short, float, short int, float, 0 >;
%template(assignITrace) pfs::drp::stella::math::assignITrace< float, unsigned short, float, short int, double, 0 >;
%template(assignITrace) pfs::drp::stella::math::assignITrace< float, unsigned short, float, long int, float, 0 >;
%template(assignITrace) pfs::drp::stella::math::assignITrace< float, unsigned short, float, long int, double, 0 >;
%template(assignITrace) pfs::drp::stella::math::assignITrace< float, unsigned short, float, int, float, 1 >;
%template(assignITrace) pfs::drp::stella::math::assignITrace< float, unsigned short, float, int, double, 1 >;
%template(assignITrace) pfs::drp::stella::math::assignITrace< float, unsigned short, float, short int, float, 1 >;
%template(assignITrace) pfs::drp::stella::math::assignITrace< float, unsigned short, float, short int, double, 1 >;
%template(assignITrace) pfs::drp::stella::math::assignITrace< float, unsigned short, float, long int, float, 1 >;
%template(assignITrace) pfs::drp::stella::math::assignITrace< float, unsigned short, float, long int, double, 1 >;

%template(unique) pfs::drp::stella::math::unique<unsigned long, 0>;
%template(unique) pfs::drp::stella::math::unique<short int, 0>;
%template(unique) pfs::drp::stella::math::unique<long int, 0>;
%template(unique) pfs::drp::stella::math::unique<float, 0>;
%template(unique) pfs::drp::stella::math::unique<double, 0>;
%template(unique) pfs::drp::stella::math::unique<unsigned long, 1>;
%template(unique) pfs::drp::stella::math::unique<short int, 1>;
%template(unique) pfs::drp::stella::math::unique<long int, 1>;
%template(unique) pfs::drp::stella::math::unique<float, 1>;
%template(unique) pfs::drp::stella::math::unique<double, 1>;

%template(createLineList) pfs::drp::stella::math::createLineList< float, 0 >;
%template(createLineList) pfs::drp::stella::math::createLineList< double, 0 >;
%template(createLineList) pfs::drp::stella::math::createLineList< float, 1 >;
%template(createLineList) pfs::drp::stella::math::createLineList< double, 1 >;

%template(addFiberTraceToCcdImage) pfs::drp::stella::math::addFiberTraceToCcdImage< float, unsigned short, float, float, float >;
%template(addFiberTraceToCcdImage) pfs::drp::stella::math::addFiberTraceToCcdImage< float, unsigned short, float, unsigned short, unsigned short >;
%template(addFiberTraceToCcdImage) pfs::drp::stella::math::addFiberTraceToCcdImage< float, unsigned short, float, double, float >;
%template(addFiberTraceToCcdImage) pfs::drp::stella::math::addFiberTraceToCcdImage< float, unsigned short, float, float, unsigned short >;
%template(addFiberTraceToCcdImage) pfs::drp::stella::math::addFiberTraceToCcdImage< float, unsigned short, float, double, double >;

%template(markFiberTraceInMask) pfs::drp::stella::utils::markFiberTraceInMask<float, unsigned short, float>;

%template(where) pfs::drp::stella::math::where< size_t, size_t, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, size_t, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, short int, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, short int, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, int, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, int, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, long int, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, long int, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, float, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, float, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, double, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, double, 1 >;
%template(where) pfs::drp::stella::math::where< short int, size_t, 0 >;
%template(where) pfs::drp::stella::math::where< short int, size_t, 1 >;
%template(where) pfs::drp::stella::math::where< short int, short int, 0 >;
%template(where) pfs::drp::stella::math::where< short int, short int, 1 >;
%template(where) pfs::drp::stella::math::where< short int, int, 0 >;
%template(where) pfs::drp::stella::math::where< short int, int, 1 >;
%template(where) pfs::drp::stella::math::where< short int, long int, 0 >;
%template(where) pfs::drp::stella::math::where< short int, long int, 1 >;
%template(where) pfs::drp::stella::math::where< short int, float, 0 >;
%template(where) pfs::drp::stella::math::where< short int, float, 1 >;
%template(where) pfs::drp::stella::math::where< short int, double, 0 >;
%template(where) pfs::drp::stella::math::where< short int, double, 1 >;
%template(where) pfs::drp::stella::math::where< int, size_t, 0 >;
%template(where) pfs::drp::stella::math::where< int, size_t, 1 >;
%template(where) pfs::drp::stella::math::where< int, short int, 0 >;
%template(where) pfs::drp::stella::math::where< int, short int, 1 >;
%template(where) pfs::drp::stella::math::where< int, int, 0 >;
%template(where) pfs::drp::stella::math::where< int, int, 1 >;
%template(where) pfs::drp::stella::math::where< int, long int, 0 >;
%template(where) pfs::drp::stella::math::where< int, long int, 1 >;
%template(where) pfs::drp::stella::math::where< int, float, 0 >;
%template(where) pfs::drp::stella::math::where< int, float, 1 >;
%template(where) pfs::drp::stella::math::where< int, double, 0 >;
%template(where) pfs::drp::stella::math::where< int, double, 1 >;
%template(where) pfs::drp::stella::math::where< long int, size_t, 0 >;
%template(where) pfs::drp::stella::math::where< long int, size_t, 1 >;
%template(where) pfs::drp::stella::math::where< long int, short int, 0 >;
%template(where) pfs::drp::stella::math::where< long int, short int, 1 >;
%template(where) pfs::drp::stella::math::where< long int, int, 0 >;
%template(where) pfs::drp::stella::math::where< long int, int, 1 >;
%template(where) pfs::drp::stella::math::where< long int, long int, 0 >;
%template(where) pfs::drp::stella::math::where< long int, long int, 1 >;
%template(where) pfs::drp::stella::math::where< long int, float, 0 >;
%template(where) pfs::drp::stella::math::where< long int, float, 1 >;
%template(where) pfs::drp::stella::math::where< long int, double, 0 >;
%template(where) pfs::drp::stella::math::where< long int, double, 1 >;
%template(where) pfs::drp::stella::math::where< float, size_t, 0 >;
%template(where) pfs::drp::stella::math::where< float, size_t, 1 >;
%template(where) pfs::drp::stella::math::where< float, short int, 0 >;
%template(where) pfs::drp::stella::math::where< float, short int, 1 >;
%template(where) pfs::drp::stella::math::where< float, int, 0 >;
%template(where) pfs::drp::stella::math::where< float, int, 1 >;
%template(where) pfs::drp::stella::math::where< float, long int, 0 >;
%template(where) pfs::drp::stella::math::where< float, long int, 1 >;
%template(where) pfs::drp::stella::math::where< float, float, 0 >;
%template(where) pfs::drp::stella::math::where< float, float, 1 >;
%template(where) pfs::drp::stella::math::where< float, double, 0 >;
%template(where) pfs::drp::stella::math::where< float, double, 1 >;
%template(where) pfs::drp::stella::math::where< double, size_t, 0 >;
%template(where) pfs::drp::stella::math::where< double, size_t, 1 >;
%template(where) pfs::drp::stella::math::where< double, short int, 0 >;
%template(where) pfs::drp::stella::math::where< double, short int, 1 >;
%template(where) pfs::drp::stella::math::where< double, int, 0 >;
%template(where) pfs::drp::stella::math::where< double, int, 1 >;
%template(where) pfs::drp::stella::math::where< double, long int, 0 >;
%template(where) pfs::drp::stella::math::where< double, long int, 1 >;
%template(where) pfs::drp::stella::math::where< double, float, 0 >;
%template(where) pfs::drp::stella::math::where< double, float, 1 >;
%template(where) pfs::drp::stella::math::where< double, double, 0 >;
%template(where) pfs::drp::stella::math::where< double, double, 1 >;

%template(where) pfs::drp::stella::math::where< size_t, size_t, 0, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, size_t, 0, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, size_t, 0, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, size_t, 1, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, size_t, 1, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, size_t, 1, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, size_t, 2, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, size_t, 2, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, size_t, 2, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, short int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, short int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, short int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, short int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, short int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, short int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, short int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, short int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, short int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, long int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, long int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, long int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, long int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, long int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, long int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, long int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, long int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, long int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, float, 0, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, float, 0, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, float, 0, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, float, 1, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, float, 1, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, float, 1, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, float, 2, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, float, 2, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, float, 2, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, double, 0, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, double, 0, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, double, 0, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, double, 1, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, double, 1, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, double, 1, 2 >;
%template(where) pfs::drp::stella::math::where< size_t, double, 2, 0 >;
%template(where) pfs::drp::stella::math::where< size_t, double, 2, 1 >;
%template(where) pfs::drp::stella::math::where< size_t, double, 2, 2 >;
%template(where) pfs::drp::stella::math::where< short int, size_t, 0, 0 >;
%template(where) pfs::drp::stella::math::where< short int, size_t, 0, 1 >;
%template(where) pfs::drp::stella::math::where< short int, size_t, 0, 2 >;
%template(where) pfs::drp::stella::math::where< short int, size_t, 1, 0 >;
%template(where) pfs::drp::stella::math::where< short int, size_t, 1, 1 >;
%template(where) pfs::drp::stella::math::where< short int, size_t, 1, 2 >;
%template(where) pfs::drp::stella::math::where< short int, size_t, 2, 0 >;
%template(where) pfs::drp::stella::math::where< short int, size_t, 2, 1 >;
%template(where) pfs::drp::stella::math::where< short int, size_t, 2, 2 >;
%template(where) pfs::drp::stella::math::where< short int, short int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< short int, short int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< short int, short int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< short int, short int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< short int, short int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< short int, short int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< short int, short int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< short int, short int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< short int, short int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< short int, int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< short int, int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< short int, int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< short int, int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< short int, int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< short int, int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< short int, int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< short int, int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< short int, int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< short int, long int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< short int, long int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< short int, long int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< short int, long int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< short int, long int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< short int, long int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< short int, long int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< short int, long int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< short int, long int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< short int, float, 0, 0 >;
%template(where) pfs::drp::stella::math::where< short int, float, 0, 1 >;
%template(where) pfs::drp::stella::math::where< short int, float, 0, 2 >;
%template(where) pfs::drp::stella::math::where< short int, float, 1, 0 >;
%template(where) pfs::drp::stella::math::where< short int, float, 1, 1 >;
%template(where) pfs::drp::stella::math::where< short int, float, 1, 2 >;
%template(where) pfs::drp::stella::math::where< short int, float, 2, 0 >;
%template(where) pfs::drp::stella::math::where< short int, float, 2, 1 >;
%template(where) pfs::drp::stella::math::where< short int, float, 2, 2 >;
%template(where) pfs::drp::stella::math::where< short int, double, 0, 0 >;
%template(where) pfs::drp::stella::math::where< short int, double, 0, 1 >;
%template(where) pfs::drp::stella::math::where< short int, double, 0, 2 >;
%template(where) pfs::drp::stella::math::where< short int, double, 1, 0 >;
%template(where) pfs::drp::stella::math::where< short int, double, 1, 1 >;
%template(where) pfs::drp::stella::math::where< short int, double, 1, 2 >;
%template(where) pfs::drp::stella::math::where< short int, double, 2, 0 >;
%template(where) pfs::drp::stella::math::where< short int, double, 2, 1 >;
%template(where) pfs::drp::stella::math::where< short int, double, 2, 2 >;
%template(where) pfs::drp::stella::math::where< int, size_t, 0, 0 >;
%template(where) pfs::drp::stella::math::where< int, size_t, 0, 1 >;
%template(where) pfs::drp::stella::math::where< int, size_t, 0, 2 >;
%template(where) pfs::drp::stella::math::where< int, size_t, 1, 0 >;
%template(where) pfs::drp::stella::math::where< int, size_t, 1, 1 >;
%template(where) pfs::drp::stella::math::where< int, size_t, 1, 2 >;
%template(where) pfs::drp::stella::math::where< int, size_t, 2, 0 >;
%template(where) pfs::drp::stella::math::where< int, size_t, 2, 1 >;
%template(where) pfs::drp::stella::math::where< int, size_t, 2, 2 >;
%template(where) pfs::drp::stella::math::where< int, short int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< int, short int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< int, short int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< int, short int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< int, short int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< int, short int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< int, short int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< int, short int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< int, short int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< int, int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< int, int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< int, int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< int, int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< int, int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< int, int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< int, int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< int, int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< int, int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< int, long int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< int, long int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< int, long int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< int, long int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< int, long int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< int, long int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< int, long int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< int, long int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< int, long int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< int, float, 0, 0 >;
%template(where) pfs::drp::stella::math::where< int, float, 0, 1 >;
%template(where) pfs::drp::stella::math::where< int, float, 0, 2 >;
%template(where) pfs::drp::stella::math::where< int, float, 1, 0 >;
%template(where) pfs::drp::stella::math::where< int, float, 1, 1 >;
%template(where) pfs::drp::stella::math::where< int, float, 1, 2 >;
%template(where) pfs::drp::stella::math::where< int, float, 2, 0 >;
%template(where) pfs::drp::stella::math::where< int, float, 2, 1 >;
%template(where) pfs::drp::stella::math::where< int, float, 2, 2 >;
%template(where) pfs::drp::stella::math::where< int, double, 0, 0 >;
%template(where) pfs::drp::stella::math::where< int, double, 0, 1 >;
%template(where) pfs::drp::stella::math::where< int, double, 0, 2 >;
%template(where) pfs::drp::stella::math::where< int, double, 1, 0 >;
%template(where) pfs::drp::stella::math::where< int, double, 1, 1 >;
%template(where) pfs::drp::stella::math::where< int, double, 1, 2 >;
%template(where) pfs::drp::stella::math::where< int, double, 2, 0 >;
%template(where) pfs::drp::stella::math::where< int, double, 2, 1 >;
%template(where) pfs::drp::stella::math::where< int, double, 2, 2 >;
%template(where) pfs::drp::stella::math::where< long int, size_t, 0, 0 >;
%template(where) pfs::drp::stella::math::where< long int, size_t, 0, 1 >;
%template(where) pfs::drp::stella::math::where< long int, size_t, 0, 2 >;
%template(where) pfs::drp::stella::math::where< long int, size_t, 1, 0 >;
%template(where) pfs::drp::stella::math::where< long int, size_t, 1, 1 >;
%template(where) pfs::drp::stella::math::where< long int, size_t, 1, 2 >;
%template(where) pfs::drp::stella::math::where< long int, size_t, 2, 0 >;
%template(where) pfs::drp::stella::math::where< long int, size_t, 2, 1 >;
%template(where) pfs::drp::stella::math::where< long int, size_t, 2, 2 >;
%template(where) pfs::drp::stella::math::where< long int, short int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< long int, short int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< long int, short int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< long int, short int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< long int, short int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< long int, short int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< long int, short int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< long int, short int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< long int, short int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< long int, int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< long int, int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< long int, int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< long int, int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< long int, int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< long int, int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< long int, int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< long int, int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< long int, int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< long int, long int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< long int, long int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< long int, long int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< long int, long int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< long int, long int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< long int, long int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< long int, long int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< long int, long int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< long int, long int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< long int, float, 0, 0 >;
%template(where) pfs::drp::stella::math::where< long int, float, 0, 1 >;
%template(where) pfs::drp::stella::math::where< long int, float, 0, 2 >;
%template(where) pfs::drp::stella::math::where< long int, float, 1, 0 >;
%template(where) pfs::drp::stella::math::where< long int, float, 1, 1 >;
%template(where) pfs::drp::stella::math::where< long int, float, 1, 2 >;
%template(where) pfs::drp::stella::math::where< long int, float, 2, 0 >;
%template(where) pfs::drp::stella::math::where< long int, float, 2, 1 >;
%template(where) pfs::drp::stella::math::where< long int, float, 2, 2 >;
%template(where) pfs::drp::stella::math::where< long int, double, 0, 0 >;
%template(where) pfs::drp::stella::math::where< long int, double, 0, 1 >;
%template(where) pfs::drp::stella::math::where< long int, double, 0, 2 >;
%template(where) pfs::drp::stella::math::where< long int, double, 1, 0 >;
%template(where) pfs::drp::stella::math::where< long int, double, 1, 1 >;
%template(where) pfs::drp::stella::math::where< long int, double, 1, 2 >;
%template(where) pfs::drp::stella::math::where< long int, double, 2, 0 >;
%template(where) pfs::drp::stella::math::where< long int, double, 2, 1 >;
%template(where) pfs::drp::stella::math::where< long int, double, 2, 2 >;
%template(where) pfs::drp::stella::math::where< float, size_t, 0, 0 >;
%template(where) pfs::drp::stella::math::where< float, size_t, 0, 1 >;
%template(where) pfs::drp::stella::math::where< float, size_t, 0, 2 >;
%template(where) pfs::drp::stella::math::where< float, size_t, 1, 0 >;
%template(where) pfs::drp::stella::math::where< float, size_t, 1, 1 >;
%template(where) pfs::drp::stella::math::where< float, size_t, 1, 2 >;
%template(where) pfs::drp::stella::math::where< float, size_t, 2, 0 >;
%template(where) pfs::drp::stella::math::where< float, size_t, 2, 1 >;
%template(where) pfs::drp::stella::math::where< float, size_t, 2, 2 >;
%template(where) pfs::drp::stella::math::where< float, short int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< float, short int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< float, short int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< float, short int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< float, short int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< float, short int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< float, short int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< float, short int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< float, short int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< float, int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< float, int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< float, int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< float, int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< float, int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< float, int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< float, int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< float, int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< float, int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< float, long int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< float, long int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< float, long int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< float, long int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< float, long int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< float, long int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< float, long int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< float, long int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< float, long int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< float, float, 0, 0 >;
%template(where) pfs::drp::stella::math::where< float, float, 0, 1 >;
%template(where) pfs::drp::stella::math::where< float, float, 0, 2 >;
%template(where) pfs::drp::stella::math::where< float, float, 1, 0 >;
%template(where) pfs::drp::stella::math::where< float, float, 1, 1 >;
%template(where) pfs::drp::stella::math::where< float, float, 1, 2 >;
%template(where) pfs::drp::stella::math::where< float, float, 2, 0 >;
%template(where) pfs::drp::stella::math::where< float, float, 2, 1 >;
%template(where) pfs::drp::stella::math::where< float, float, 2, 2 >;
%template(where) pfs::drp::stella::math::where< float, double, 0, 0 >;
%template(where) pfs::drp::stella::math::where< float, double, 0, 1 >;
%template(where) pfs::drp::stella::math::where< float, double, 0, 2 >;
%template(where) pfs::drp::stella::math::where< float, double, 1, 0 >;
%template(where) pfs::drp::stella::math::where< float, double, 1, 1 >;
%template(where) pfs::drp::stella::math::where< float, double, 1, 2 >;
%template(where) pfs::drp::stella::math::where< float, double, 2, 0 >;
%template(where) pfs::drp::stella::math::where< float, double, 2, 1 >;
%template(where) pfs::drp::stella::math::where< float, double, 2, 2 >;
%template(where) pfs::drp::stella::math::where< double, size_t, 0, 0 >;
%template(where) pfs::drp::stella::math::where< double, size_t, 0, 1 >;
%template(where) pfs::drp::stella::math::where< double, size_t, 0, 2 >;
%template(where) pfs::drp::stella::math::where< double, size_t, 1, 0 >;
%template(where) pfs::drp::stella::math::where< double, size_t, 1, 1 >;
%template(where) pfs::drp::stella::math::where< double, size_t, 1, 2 >;
%template(where) pfs::drp::stella::math::where< double, size_t, 2, 0 >;
%template(where) pfs::drp::stella::math::where< double, size_t, 2, 1 >;
%template(where) pfs::drp::stella::math::where< double, size_t, 2, 2 >;
%template(where) pfs::drp::stella::math::where< double, short int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< double, short int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< double, short int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< double, short int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< double, short int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< double, short int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< double, short int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< double, short int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< double, short int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< double, int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< double, int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< double, int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< double, int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< double, int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< double, int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< double, int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< double, int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< double, int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< double, long int, 0, 0 >;
%template(where) pfs::drp::stella::math::where< double, long int, 0, 1 >;
%template(where) pfs::drp::stella::math::where< double, long int, 0, 2 >;
%template(where) pfs::drp::stella::math::where< double, long int, 1, 0 >;
%template(where) pfs::drp::stella::math::where< double, long int, 1, 1 >;
%template(where) pfs::drp::stella::math::where< double, long int, 1, 2 >;
%template(where) pfs::drp::stella::math::where< double, long int, 2, 0 >;
%template(where) pfs::drp::stella::math::where< double, long int, 2, 1 >;
%template(where) pfs::drp::stella::math::where< double, long int, 2, 2 >;
%template(where) pfs::drp::stella::math::where< double, float, 0, 0 >;
%template(where) pfs::drp::stella::math::where< double, float, 0, 1 >;
%template(where) pfs::drp::stella::math::where< double, float, 0, 2 >;
%template(where) pfs::drp::stella::math::where< double, float, 1, 0 >;
%template(where) pfs::drp::stella::math::where< double, float, 1, 1 >;
%template(where) pfs::drp::stella::math::where< double, float, 1, 2 >;
%template(where) pfs::drp::stella::math::where< double, float, 2, 0 >;
%template(where) pfs::drp::stella::math::where< double, float, 2, 1 >;
%template(where) pfs::drp::stella::math::where< double, float, 2, 2 >;
%template(where) pfs::drp::stella::math::where< double, double, 0, 0 >;
%template(where) pfs::drp::stella::math::where< double, double, 0, 1 >;
%template(where) pfs::drp::stella::math::where< double, double, 0, 2 >;
%template(where) pfs::drp::stella::math::where< double, double, 1, 0 >;
%template(where) pfs::drp::stella::math::where< double, double, 1, 1 >;
%template(where) pfs::drp::stella::math::where< double, double, 1, 2 >;
%template(where) pfs::drp::stella::math::where< double, double, 2, 0 >;
%template(where) pfs::drp::stella::math::where< double, double, 2, 1 >;
%template(where) pfs::drp::stella::math::where< double, double, 2, 2 >;

%template(where) pfs::drp::stella::math::where< size_t, size_t >;
%template(where) pfs::drp::stella::math::where< size_t, short int >;
%template(where) pfs::drp::stella::math::where< size_t, int >;
%template(where) pfs::drp::stella::math::where< size_t, long int >;
%template(where) pfs::drp::stella::math::where< size_t, float >;
%template(where) pfs::drp::stella::math::where< size_t, double >;
%template(where) pfs::drp::stella::math::where< short int, size_t >;
%template(where) pfs::drp::stella::math::where< short int, short int >;
%template(where) pfs::drp::stella::math::where< short int, int >;
%template(where) pfs::drp::stella::math::where< short int, long int >;
%template(where) pfs::drp::stella::math::where< short int, float >;
%template(where) pfs::drp::stella::math::where< short int, double >;
%template(where) pfs::drp::stella::math::where< int, size_t >;
%template(where) pfs::drp::stella::math::where< int, short int >;
%template(where) pfs::drp::stella::math::where< int, int >;
%template(where) pfs::drp::stella::math::where< int, long int >;
%template(where) pfs::drp::stella::math::where< int, float >;
%template(where) pfs::drp::stella::math::where< int, double >;
%template(where) pfs::drp::stella::math::where< long int, size_t >;
%template(where) pfs::drp::stella::math::where< long int, short int >;
%template(where) pfs::drp::stella::math::where< long int, int >;
%template(where) pfs::drp::stella::math::where< long int, long int >;
%template(where) pfs::drp::stella::math::where< long int, float >;
%template(where) pfs::drp::stella::math::where< long int, double >;
%template(where) pfs::drp::stella::math::where< float, size_t >;
%template(where) pfs::drp::stella::math::where< float, short int >;
%template(where) pfs::drp::stella::math::where< float, int >;
%template(where) pfs::drp::stella::math::where< float, long int >;
%template(where) pfs::drp::stella::math::where< float, float >;
%template(where) pfs::drp::stella::math::where< float, double >;
%template(where) pfs::drp::stella::math::where< double, size_t >;
%template(where) pfs::drp::stella::math::where< double, short int >;
%template(where) pfs::drp::stella::math::where< double, int >;
%template(where) pfs::drp::stella::math::where< double, long int >;
%template(where) pfs::drp::stella::math::where< double, float >;
%template(where) pfs::drp::stella::math::where< double, double >;

%template(isMonotonic) pfs::drp::stella::math::isMonotonic< int >;

%template(getIndices) pfs::drp::stella::math::getIndices< size_t >;
%template(getIndices) pfs::drp::stella::math::getIndices< int >;
%template(getIndices) pfs::drp::stella::math::getIndices< long >;
%template(getIndices) pfs::drp::stella::math::getIndices< float >;
%template(getIndices) pfs::drp::stella::math::getIndices< double >;

%template(PolyFit) pfs::drp::stella::math::PolyFit<double>;

%template(addArrayIntoArray) pfs::drp::stella::math::addArrayIntoArray< float const, float, 1, 1 >;
%template(addArrayIntoArray) pfs::drp::stella::math::addArrayIntoArray< float const, float, 2, 1 >;
%template(addArrayIntoArray) pfs::drp::stella::math::addArrayIntoArray< float const, float, 1, 2 >;
%template(addArrayIntoArray) pfs::drp::stella::math::addArrayIntoArray< float const, float, 2, 2 >;

%template(calcMinCenMax) pfs::drp::stella::math::calcMinCenMax<float, float>;
%template(calcMinCenMax) pfs::drp::stella::math::calcMinCenMax<double, float>;
%template(calcMinCenMax) pfs::drp::stella::math::calcMinCenMax<float, double>;
%template(calcMinCenMax) pfs::drp::stella::math::calcMinCenMax<double, double>;

%template(findITrace) pfs::drp::stella::math::findITrace<float, unsigned short, float, float, 0>;

%template(ccdToFiberTraceCoordinates) pfs::drp::stella::math::ccdToFiberTraceCoordinates<float, float, unsigned short, float>;
%template(fiberTraceCoordinatesRelativeTo) pfs::drp::stella::math::fiberTraceCoordinatesRelativeTo<float, float, unsigned short, float>;

%template(CoordinatesF) pfs::drp::stella::math::dataXY<float>;
