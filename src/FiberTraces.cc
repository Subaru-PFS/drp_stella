#include <algorithm>
#include "ndarray/eigen.h"

#include "lsst/log/Log.h"
#include "lsst/pex/exceptions/Exception.h"
#include "lsst/afw/image/MaskedImage.h"

#include "pfs/drp/stella/FiberTraces.h"
#include "pfs/drp/stella/cmpfit-1.2/MPFitting_ndarray.h"
#include "pfs/drp/stella/math/Chebyshev.h"
#include "pfs/drp/stella/math/CurveFitting.h"
#include "pfs/drp/stella/spline.h"
#include "pfs/drp/stella/utils/Utils.h"

//#define __DEBUG_FINDANDTRACE__ 1
namespace afwImage = lsst::afw::image;

namespace pfs { namespace drp { namespace stella {

  template<typename ImageT, typename MaskT, typename VarianceT>
  FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    PTR(const MaskedImageT) const& maskedImage,
    std::size_t fiberTraceId
  ) :
  _overSampledProfileFitXPerSwath(),
  _overSampledProfileFitYPerSwath(),
  _profileFittingInputXPerSwath(),
  _profileFittingInputYPerSwath(),
  _profileFittingInputXMeanPerSwath(),
  _profileFittingInputYMeanPerSwath(),
  _trace(new MaskedImageT(*maskedImage)),
  _xCenters(utils::get1DndArray(float(0))),
  _fiberId(fiberTraceId),
  _fiberTraceFunction(new FiberTraceFunction()),
  _fiberTraceProfileFittingControl(new FiberTraceProfileFittingControl)
  {}

  template<typename ImageT, typename MaskT, typename VarianceT>
  FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    PTR(const afwImage::MaskedImage<ImageT, MaskT, VarianceT>) const & maskedImage,
    PTR(const FiberTraceFunction) const& fiberTraceFunction,
    PTR(FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl,
    size_t fiberId) :
  _overSampledProfileFitXPerSwath(),
  _overSampledProfileFitYPerSwath(),
  _profileFittingInputXPerSwath(),
  _profileFittingInputYPerSwath(),
  _profileFittingInputXMeanPerSwath(),
  _profileFittingInputYMeanPerSwath(),
  _trace(new afwImage::MaskedImage<ImageT, MaskT, VarianceT>(
    fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1,
    int(fiberTraceFunction->fiberTraceFunctionControl->xHigh
        - fiberTraceFunction->fiberTraceFunctionControl->xLow + 1))),
  _fiberId(fiberId),
  _fiberTraceFunction(fiberTraceFunction),
  _fiberTraceProfileFittingControl(fiberTraceProfileFittingControl)
  {
    fiberTraceFunction->fiberTraceFunctionControl->nRows = maskedImage->getHeight();
    _xCenters = math::calculateXCenters(fiberTraceFunction,
                                        maskedImage->getHeight(),
                                        maskedImage->getWidth());
    _createTrace(maskedImage);
    _calcProfile();
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(FiberTrace<ImageT, MaskT, VarianceT> & fiberTrace, bool const deep) :
  _overSampledProfileFitXPerSwath(),
  _overSampledProfileFitYPerSwath(),
  _profileFittingInputXPerSwath(),
  _profileFittingInputYPerSwath(),
  _profileFittingInputXMeanPerSwath(),
  _profileFittingInputYMeanPerSwath(),
  _trace(fiberTrace.getTrace()),
  _xCenters(fiberTrace.getXCenters()),
  _fiberId(fiberTrace.getFiberId()),
  _fiberTraceFunction(new FiberTraceFunction()),
  _fiberTraceProfileFittingControl(new FiberTraceProfileFittingControl)
  {
    if (deep){
      PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) ptr(new afwImage::MaskedImage<ImageT, MaskT, VarianceT>(*(fiberTrace.getTrace()), true));
      _trace.reset();
      _trace = ptr;
    }
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(Spectrum)
  FiberTrace<ImageT, MaskT, VarianceT>::extractSpectrum(PTR(const MaskedImageT) spectrumImage,
                                                        const bool fitBackground,
                                                        const float clipNSigma,
                                                        const bool useProfile
                                                       )
  {
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.FiberTrace.extractFromProfile");
    auto const bbox = _trace->getBBox();
    MaskedImageT traceIm(*spectrumImage, bbox);
    const int height = bbox.getHeight();
    const int width = bbox.getWidth();

    ndarray::Array<ImageT, 1, 1> spec = ndarray::allocate(height);
    spec.deep() = 0.;
    ndarray::Array<ImageT, 1, 1> background = ndarray::allocate(height);
    background.deep() = 0.;
    ndarray::Array<VarianceT, 1, 1> var = ndarray::allocate(height);

    const MaskT ftMask = _trace->getMask()->getPlaneBitMask("FIBERTRACE");

    if (useProfile) {
        ndarray::Array<int, 2, 1> US_A2_Mask(height, width); // set to 1 for points in the fiberTrace
        auto itRowBin = _trace->getMask()->getArray().begin();
        for (auto itRow = US_A2_Mask.begin(); itRow != US_A2_Mask.end(); ++itRow, ++itRowBin){
            auto itColBin = itRowBin->begin();
            for (auto itCol = itRow->begin(); itCol != itRow->end(); ++itCol, ++itColBin){
                *itCol = (*itColBin & ftMask) ? 1 : 0;
            }
        }

        float rchi2 = math::fitProfile2d(traceIm,                        // input data
                                         US_A2_Mask,                     // mask of pixels to use
                                         _trace->getImage()->getArray(), // fibre trace profile
                                         fitBackground,                  // should I fit the background level?
                                         clipNSigma,                     // number of sigma to clip
                                         spec,                           // out: spectrum
                                         background,                     // out: background level
                                         var                             // out: spectrum's variance
                                        );
        if (rchi2 < 0) {
            std::string message("FiberTrace");
            message += std::to_string(_fiberId);
            message += std::string("::extractFromProfile: 2. ERROR: fitProfile2d(...) returned ");
            message += std::to_string(rchi2);
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
    } else {                            // simple profile fit
        MaskedImageT traceIm(*spectrumImage, bbox); // image box corresponding to trace
        
        auto specIt = spec.begin();
        auto varIt = var.begin();
        auto itMaskRow = _trace->getMask()->getArray().begin();
        auto itTraceRow = traceIm.getImage()->getArray().begin();
        auto itVarRow = traceIm.getVariance()->getArray().begin();
        for (int i = 0; i < _trace->getImage()->getHeight();
             ++i, ++specIt, ++varIt, ++itMaskRow, ++itTraceRow, ++itVarRow){
            *specIt = 0.0;
            *varIt = 0.0;
            auto itTraceCol = itTraceRow->begin();
            auto itVarCol = itVarRow->begin();
            for (auto itMaskCol = itMaskRow->begin(); itMaskCol != itMaskRow->end();
                 ++itMaskCol, ++itTraceCol, ++itVarCol) {
                if (*itMaskCol & ftMask){
                    *specIt += *itTraceCol;
                    *varIt += *itVarCol;
                }
            }
        }
    }

    auto spectrum = std::make_shared<Spectrum>(spectrumImage->getHeight());
    spectrum->setFiberId(_fiberId);

    ndarray::Array< ImageT, 1, 1 > spectrumSpecOut = spectrum->getSpectrum();
    ndarray::Array< ImageT, 1, 1 > backgroundOut = spectrum->getBackground();
    ndarray::Array< VarianceT, 1, 1 > spectrumVarOut = spectrum->getVariance();
    //
    // Copy the extracted spectra/stdev into the Spectrum (allowing for the ends)
    //
    {
        const auto& spectrumMask = *spectrumImage->getMask();
        const auto& traceMask = *_trace->getMask(); // we should be propagating these to spectrumMask

        auto mask = spectrum->getMask();
        const auto nodata = mask.getPlaneBitMask("NO_DATA");
        const auto fibertrace = traceMask.getPlaneBitMask("FIBERTRACE");

        int i = 0;
        for (; i < bbox.getMinY(); ++i) {
            spectrumSpecOut[i] = 0;
            mask(i, 0) |= nodata;
            backgroundOut[i] = 0;
            spectrumVarOut[i] = 0;
        }
        for (int j = 0; i < bbox.getMaxY(); ++i, ++j) {
            spectrumSpecOut[i] = spec[j];
            backgroundOut[i] =   background[j];
            spectrumVarOut[i] =  var[j];

            int maskVal = 0;
            for (int k = bbox.getMinX(); k <= bbox.getMaxX(); ++k) {
                if (traceMask(k - bbox.getMinX(), i - bbox.getMinY()) & fibertrace) { // we used this pixel
                    maskVal |= spectrumMask(k, i);
                }
            }
            mask(i, 0) |= maskVal;
        }
        for (; i < spectrumSpecOut.size(); ++i) {
            spectrumSpecOut[i] = 0;
            mask(i, 0) |= nodata;
            backgroundOut[i] = 0;
            spectrumVarOut[i]  = 0;
        }
    }

    return spectrum;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  void FiberTrace<ImageT, MaskT, VarianceT>::_createTrace( const PTR(const MaskedImageT) &maskedImage ){
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.FiberTrace.createTrace");

    _minCenMax = math::calcMinCenMax(
            _xCenters,
            _fiberTraceFunction->fiberTraceFunctionControl->xHigh,
            _fiberTraceFunction->fiberTraceFunctionControl->xLow,
            _fiberTraceFunction->fiberTraceFunctionControl->nPixCutLeft,
            _fiberTraceFunction->fiberTraceFunctionControl->nPixCutRight);
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": _minCenMax = " << _minCenMax);

    ndarray::Array<size_t, 1, 1> minPixX(_minCenMax[ndarray::view()(0)]);
    int xMin = *std::min_element(minPixX.begin(), minPixX.end());

    ndarray::Array<size_t, 1, 1> maxPixX(_minCenMax[ndarray::view()(2)]);
    int xMax = *std::max_element(maxPixX.begin(), maxPixX.end());

    lsst::afw::geom::Point<int> lowerLeft(std::pair<int, int>(
        xMin,
        _fiberTraceFunction->yCenter + _fiberTraceFunction->yLow));
    lsst::afw::geom::Extent<int, 2> extend(
        lsst::afw::geom::Point<int>(xMax - xMin + 1,
                                    _fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1));
    lsst::afw::geom::Box2I box(lowerLeft, extend);

    _trace.reset(new MaskedImageT(MaskedImageT(*maskedImage, box), true));

    /// mark FiberTrace in Mask
    _minCenMax.deep() -= static_cast<size_t>(xMin);
    std::string maskPlane = "FIBERTRACE";
    _trace->getMask()->addMaskPlane(maskPlane);
    const auto ftMask = _trace->getMask()->getPlaneBitMask(maskPlane);
    _markFiberTraceInMask(ftMask);
  }

  /// Return shared pointer to an image containing the reconstructed 2D spectrum of the FiberTrace
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(afwImage::Image<ImageT>)
      FiberTrace<ImageT, MaskT, VarianceT>::getReconstructed2DSpectrum(const Spectrum & spectrum) const
  {
      auto recon2d = std::make_shared<afwImage::Image<ImageT>>(_trace->getBBox());
      const auto FIBERTRACE = _trace->getMask()->getPlaneBitMask("FIBERTRACE");

      auto bkgd = spectrum.getBackground();
      auto spec = spectrum.getSpectrum();
      const int y0 = _trace->getY0();
      
      const int height = _trace->getHeight();
      for (int i = 0; i != height; ++i) {
          const float specVal = spec[y0 + i]; // value of spectrum
          const float bkgdVal = bkgd[y0 + i]; // value of background
          
          auto profilePtr = _trace->getImage()->row_begin(i);
          auto maskPtr  = _trace->getMask()->row_begin(i);
          auto reconPtr = recon2d->row_begin(i);
          const int width  = _trace->getImage()->getWidth();
          for (int j = 0; j != width; ++j) {
              if (maskPtr[j] & FIBERTRACE) {
                  reconPtr[j] = bkgdVal + specVal*profilePtr[j];
              }
          }
      }
      
      return recon2d;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  ndarray::Array<size_t, 2, 1> FiberTrace<ImageT, MaskT, VarianceT>::_calcSwathBoundY(size_t const& swathWidth) const{
    size_t nSwaths = 0;

    size_t swathWidth_mutable = swathWidth;
    if (swathWidth_mutable > _trace->getImage()->getHeight()){
      swathWidth_mutable = _trace->getImage()->getHeight();
      #ifdef __DEBUG_CALCSWATHBOUNDY__
        cout << "FiberTrace" << _fiberId << "::calcSwathBoundY: KeyWord_Set(SWATH_WIDTH): swathWidth_mutable too large: swathWidth_mutable set to " << swathWidth_mutable << endl;
      #endif
    }
    nSwaths = round(float(_trace->getImage()->getHeight()) / float(swathWidth));
    if (nSwaths == 0){
        nSwaths = 1;
    }
    size_t binHeight = _trace->getImage()->getHeight() / nSwaths;
    if (nSwaths > 1)
      nSwaths = (2 * nSwaths) - 1;

    #ifdef __DEBUG_CALCSWATHBOUNDY__
      cout << "FiberTrace" << _fiberId << "::calcSwathBoundY: fiberTraceNumber = " << _fiberId << endl;
      cout << "FiberTrace" << _fiberId << "::calcSwathBoundY: _trace->getImage()->getHeight() = "
           << _trace->getImage()->getHeight() << endl;
      cout << "FiberTrace" << _fiberId << "::calcSwathBoundY: binHeight = " << binHeight << endl;
      cout << "FiberTrace" << _fiberId << "::calcSwathBoundY: nSwaths set to " << nSwaths << endl;
    #endif

    /// Calculate boundaries of distinct slitf regions.
    /// Boundaries of bins
    /// Test run because if swathWidth is small rounding errors can have a big affect
    ndarray::Array<size_t, 2, 1> swathBoundYTemp = ndarray::allocate(int(nSwaths), 2);
    swathBoundYTemp[0][0] = 0;
    size_t I_BinHeight_Temp = binHeight;
    swathBoundYTemp[0][1] = I_BinHeight_Temp;
    for (size_t iSwath = 1; iSwath < nSwaths; iSwath++){
      I_BinHeight_Temp = binHeight;
      if (iSwath == 1)
        swathBoundYTemp[iSwath][0] = swathBoundYTemp[iSwath-1][0] + size_t(0.5*binHeight);
      else
        swathBoundYTemp[iSwath][0] = swathBoundYTemp[iSwath-2][1] + 1;
      swathBoundYTemp[iSwath][1] = swathBoundYTemp[iSwath][0] + I_BinHeight_Temp;
      if (swathBoundYTemp[iSwath][1] >= _trace->getImage()->getHeight()-1){
        nSwaths = iSwath + 1;
      }
      if (iSwath == (nSwaths-1)){
        swathBoundYTemp[iSwath][1] = _trace->getImage()->getHeight()-1;
      }
    }

    ndarray::Array<size_t, 2, 1> swathBoundY = ndarray::allocate(int(nSwaths), 2);
    #ifdef __DEBUG_CALCSWATHBOUNDY__
      ndarray::Array<float, 2, 1>::Index shape = swathBoundY.getShape();
      cout << "FiberTrace" << _fiberId << "::calcSwathBoundY: shape = " << shape << endl;
    #endif
    swathBoundY[0][0] = 0;
    #ifdef __DEBUG_CALCSWATHBOUNDY__
      cout << "FiberTrace" << _fiberId << "::calcSwathBoundY: 1. swathBoundY[0][0] set to " << swathBoundY[0][0] << endl;
    #endif
    I_BinHeight_Temp = binHeight;
    swathBoundY[0][1] = I_BinHeight_Temp;
    #ifdef __DEBUG_CALCSWATHBOUNDY__
      cout << "FiberTrace" << _fiberId << "::calcSwathBoundY: swathBoundY[0][1] set to " << swathBoundY[0][1] << endl;
    #endif
    for (size_t iSwath = 1; iSwath < nSwaths; iSwath++){
      I_BinHeight_Temp = binHeight;
      if (iSwath == 1)
        swathBoundY[iSwath][0] = swathBoundY[iSwath-1][0] + size_t(0.5*binHeight);
      else
        swathBoundY[iSwath][0] = swathBoundY[iSwath-2][1] + 1;
      #ifdef __DEBUG_CALCSWATHBOUNDY__
        cout << "FiberTrace" << _fiberId << "::calcSwathBoundY: swathBoundY[iSwath=" << iSwath << "][0] set to swathBoundY[iSwath-1=" << iSwath-1 << "][0] + (binHeight/2.=" << binHeight / 2. << ")" << endl;
      #endif
      swathBoundY[iSwath][1] = swathBoundY[iSwath][0] + I_BinHeight_Temp;
      if (iSwath == (nSwaths-1)){
        swathBoundY[iSwath][1] = _trace->getImage()->getHeight()-1;
        #ifdef __DEBUG_CALCSWATHBOUNDY__
          cout << "FiberTrace" << _fiberId << "::calcSwathBoundY: nSwaths = " << nSwaths << endl;
          cout << "FiberTrace" << _fiberId << "::calcSwathBoundY: _trace->getImage()->getHeight() = "
               << _trace->getImage()->getHeight() << endl;
          cout << "FiberTrace" << _fiberId << "::calcSwathBoundY: swathBoundY[" << iSwath << "][1] set to " << swathBoundY[iSwath][1] << endl;
        #endif
      }
      #ifdef __DEBUG_CALCSWATHBOUNDY__
        cout << "FiberTrace" << _fiberId << "::calcSwathBoundY: swathBoundY[" << iSwath << "][1] set to " << swathBoundY[iSwath][1] << endl;
      #endif
    }
    swathBoundY[ nSwaths - 1][ 1 ] = _trace->getImage()->getHeight() - 1;
    #ifdef __DEBUG_CALCSWATHBOUNDY__
      cout << "FiberTrace" << _fiberId << "::calcSwathBoundY: swathBoundY set to " << swathBoundY << endl;
    #endif
    return swathBoundY;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  void FiberTrace<ImageT, MaskT, VarianceT>::_calcProfile(){
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.FiberTrace._calcProfile");

    size_t nCols = _minCenMax[0][2] - _minCenMax[0][0] + 1;

    /// Calculate boundaries for swaths
    ndarray::Array<size_t, 2, 1> swathBoundsY = _calcSwathBoundY(_fiberTraceProfileFittingControl->swathWidth);
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": swathBoundsY = " << swathBoundsY);

    ndarray::Array<size_t, 1, 1> nPixArr = ndarray::allocate(swathBoundsY.getShape()[0]);
    nPixArr[ndarray::view()] = swathBoundsY[ndarray::view()(1)] - swathBoundsY[ndarray::view()(0)] + 1;
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": nPixArr = " << nPixArr);

    unsigned int nSwaths = swathBoundsY.getShape()[0];
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": nSwaths = " << nSwaths);
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": _trace->getImage()->getHeight() = "
                      << _trace->getImage()->getHeight());

    /// for each swath
    ndarray::Array<float, 3, 2> slitFuncsSwaths = ndarray::allocate(nPixArr[0],
                                                                    nCols,
                                                                    int(nSwaths-1));
    ndarray::Array<float, 2, 1> lastSlitFuncSwath = ndarray::allocate(nPixArr[nPixArr.getShape()[0] - 1],
                                                                      nCols);
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths.getShape() = "
                      << slitFuncsSwaths.getShape());
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths.getShape()[0] = "
                      << slitFuncsSwaths.getShape()[0] << ", slitFuncsSwaths.getShape()[1] = "
                      << slitFuncsSwaths.getShape()[1] << ", slitFuncsSwaths.getShape()[2] = "
                      << slitFuncsSwaths.getShape()[2]);
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths[view(0)()(0) = "
                      << slitFuncsSwaths[ndarray::view(0)()(0)]);
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths[view(0)(0,9)(0) = "
                      << slitFuncsSwaths[ndarray::view(0)(0,slitFuncsSwaths.getShape()[1])(0)]);

    _overSampledProfileFitXPerSwath.resize(0);
    _overSampledProfileFitYPerSwath.resize(0);
    _profileFittingInputXPerSwath.resize(0);
    _profileFittingInputYPerSwath.resize(0);
    _profileFittingInputXMeanPerSwath.resize(0);
    _profileFittingInputYMeanPerSwath.resize(0);

    std::string maskPlane = "FIBERTRACE";
    lsst::afw::image::Image<ImageT> profile(nCols, _trace->getImage()->getHeight());
    profile.getArray().deep() = 0.0;

    for (unsigned int iSwath = 0; iSwath < nSwaths; ++iSwath){
      int yMin = int(swathBoundsY[iSwath][0]);
      int yMax = int(swathBoundsY[iSwath][1] + 1);
      LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": yMin = "
                        << yMin << ", yMax = " << yMax);

      size_t nRows = yMax - yMin;
      ndarray::Array<ImageT, 2, 1> imageSwath = ndarray::allocate(nRows, nCols);
      ndarray::Array<MaskT, 2, 1> maskSwath = ndarray::allocate(nRows, nCols);
      ndarray::Array<VarianceT, 2, 1> varianceSwath = ndarray::allocate(nRows, nCols);
      ndarray::Array<float, 1, 1> xCentersSwath = ndarray::copy(
            _xCenters[ndarray::view(yMin, yMax)]);
      for (int iRow=0; iRow<nRows; ++iRow){
          imageSwath[ndarray::view(iRow)()] = _trace->getImage()->getArray()[
                  ndarray::view(yMin+iRow)(_minCenMax[iRow+yMin][0], _minCenMax[iRow+yMin][2]+1)];
          maskSwath[ndarray::view(iRow)()] = _trace->getMask()->getArray()[
                  ndarray::view(yMin+iRow)(_minCenMax[iRow+yMin][0], _minCenMax[iRow+yMin][2]+1)];
          varianceSwath[ndarray::view(iRow)()] = _trace->getVariance()->getArray()[
                  ndarray::view(yMin+iRow)(_minCenMax[iRow+yMin][0], _minCenMax[iRow+yMin][2]+1)];
      }
      LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": swath " << iSwath << ": imageSwath = " << imageSwath);

      if (iSwath < nSwaths - 1){
        slitFuncsSwaths[ndarray::view()()(iSwath)] = _calcProfileSwath(imageSwath,
                                                                      maskSwath,
                                                                      varianceSwath,
                                                                      xCentersSwath,
                                                                      iSwath);
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths.getShape() = "
                          << slitFuncsSwaths.getShape());
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": imageSwath.getShape() = " << imageSwath.getShape());
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": xCentersSwath.getShape() = "
                          << xCentersSwath.getShape());
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": swathBoundsY = " << swathBoundsY);
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": nPixArr = " << nPixArr);
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": swath " << iSwath
                          << ": slitFuncsSwaths[ndarray::view()()(iSwath)] = "
                          << slitFuncsSwaths[ndarray::view()()(iSwath)]);
      }
      else{
        lastSlitFuncSwath.deep() = _calcProfileSwath(imageSwath,
                                                    maskSwath,
                                                    varianceSwath,
                                                    xCentersSwath,
                                                    iSwath);
      }
    }

    if (nSwaths == 1){
      profile.getArray() = lastSlitFuncSwath;
      return;
    }
    int I_Bin = 0;
    double D_Weight_Bin0 = 0.;
    double D_Weight_Bin1 = 0.;
    double D_RowSum;
    for (size_t iSwath = 0; iSwath < nSwaths - 1; iSwath++){
      for (size_t i_row = 0; i_row < nPixArr[iSwath]; i_row++){
        D_RowSum = ndarray::sum(ndarray::Array<float, 1, 0>(
                   slitFuncsSwaths[ndarray::view(static_cast<int>(i_row))()(iSwath)]));
        if (std::fabs(D_RowSum) > 0.00000000000000001){
          for (int iPix = 0; iPix < slitFuncsSwaths.getShape()[1]; iPix++){
            slitFuncsSwaths[i_row][iPix][iSwath] = slitFuncsSwaths[i_row][iPix][iSwath] / D_RowSum;
          }
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths(" << i_row << ", *, "
                            << iSwath << ") = "
                            << slitFuncsSwaths[ndarray::view(static_cast<int>(i_row))()(iSwath)]);
            D_RowSum = ndarray::sum(ndarray::Array<float, 1, 0>(
                       slitFuncsSwaths[ndarray::view(static_cast<int>(i_row))()(iSwath)]));
            LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ": iSwath = "
                              << iSwath << ": D_RowSum = " << D_RowSum);
        }
      }
    }
    for (size_t i_row = 0; i_row < nPixArr[nPixArr.getShape()[0]-1]; i_row++){
      D_RowSum = ndarray::sum(lastSlitFuncSwath[ndarray::view(static_cast<int>(i_row))()]);
      if (std::fabs(D_RowSum) > 0.00000000000000001){
        lastSlitFuncSwath[ndarray::view(static_cast<int>(i_row))()] =
                lastSlitFuncSwath[ndarray::view(static_cast<int>(i_row))()] / D_RowSum;
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": lastSlitFuncSwath(" << i_row << ", *) = "
                          << lastSlitFuncSwath[ndarray::view(static_cast<int>(i_row))()]);
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ": D_RowSum = " << D_RowSum);
      }
    }
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": swathBoundsY.getShape() = "
                      << swathBoundsY.getShape() << ", nSwaths = " << nSwaths);
    int iRowSwath = 0;
    for (size_t i_row = 0; i_row < static_cast<size_t>(_trace->getImage()->getHeight()); ++i_row){
      iRowSwath = i_row - swathBoundsY[I_Bin][0];
      LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ", I_Bin = "
                        << I_Bin << ", iRowSwath = " << iRowSwath);
      if ((I_Bin == 0) && (i_row < swathBoundsY[1][0])){
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": I_Bin=" << I_Bin << " == 0 && i_row="
                          << i_row << " < swathBoundsY[1][0]=" << swathBoundsY[1][0]);
        profile.getArray()[
                ndarray::view(static_cast<int>(i_row))()]
                = ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()(0)]);
      }
      else if ((I_Bin == nSwaths-1) && (i_row >= swathBoundsY[I_Bin-1][1])){
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": I_Bin=" << I_Bin << " == nSwaths-1="
                          << nSwaths-1 << " && i_row=" << i_row << " >= swathBoundsY[I_Bin-1="
                          << I_Bin - 1 << "][0]=" << swathBoundsY[I_Bin-1][0]);
        profile.getArray()[
                ndarray::view(static_cast<int>(i_row))()]
                = lastSlitFuncSwath[ndarray::view(iRowSwath)()];
      }
      else{
        D_Weight_Bin1 = float(i_row - swathBoundsY[I_Bin+1][0])/(swathBoundsY[I_Bin][1]
                              - swathBoundsY[I_Bin+1][0]);
        D_Weight_Bin0 = 1. - D_Weight_Bin1;
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ": nSwaths = " << nSwaths);
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ": I_Bin = " << I_Bin);
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ": swathBoundsY(I_Bin, *) = "
                          << swathBoundsY[ndarray::view(I_Bin)()]);
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ": swathBoundsY(I_Bin+1, *) = "
                          << swathBoundsY[ndarray::view(I_Bin+1)()]);
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ": D_Weight_Bin0 = "
                          << D_Weight_Bin0);
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ": D_Weight_Bin1 = "
                          << D_Weight_Bin1);
        if (I_Bin == nSwaths - 2){
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths(iRowSwath, *, I_Bin) = "
                            << slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)]
                            << ", lastSlitFuncSwath[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0])="
                            << int(i_row - swathBoundsY[I_Bin+1][0]) << ")] = "
                            << lastSlitFuncSwath[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()]);
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": profile.getArray().getShape() = "
                            << profile.getArray().getShape());
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": slitFuncsSwath.getShape() = "
                            << slitFuncsSwaths.getShape());
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": lastSlitFuncSwath.getShape() = "
                            << lastSlitFuncSwath.getShape());
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ", iRowSwath = "
                            << iRowSwath << ", I_Bin = " << I_Bin << ", swathBoundsY[I_Bin+1][0] = "
                            << swathBoundsY[I_Bin+1][0] << ", i_row - swathBoundsY[I_Bin+1][0] = "
                            << i_row - swathBoundsY[I_Bin+1][0]);
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": profile.getArray()[ndarray::view(i_row)()] = "
                            << profile.getArray()[ndarray::view(i_row)()]);
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId
                            << ": slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)] = "
                            << slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)]);
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId
                            << ": ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()"
                            << "(I_Bin)]).getShape() = "
                            << slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)].getShape());
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": D_Weight_Bin0 = " << D_Weight_Bin0
                            << ", D_Weight_Bin1 = " << D_Weight_Bin1);
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId
                            << ": lastSlitFuncSwath[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()] = "
                            << lastSlitFuncSwath[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()]);
          profile.getArray()[
                  ndarray::view(static_cast<int>(i_row))()]
                  = (ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)])
                     * D_Weight_Bin0)
                    + (lastSlitFuncSwath[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()]
                       * D_Weight_Bin1);
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId
                            << ": profile.getArray()[ndarray::view(i_row)()] set to "
                            << profile.getArray()[ndarray::view(i_row)()]);
        }
        else{
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths(iRowSwath, *, I_Bin) = "
                            << slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)]
                            << ", slitFuncsSwaths(int(i_row - swathBoundsY[I_Bin+1][0])="
                            << int(i_row - swathBoundsY[I_Bin+1][0]) << ", *, I_Bin+1) = "
                            << slitFuncsSwaths[ndarray::view(
                               int(i_row - swathBoundsY[I_Bin+1][0]))()(I_Bin+1)]);
          profile.getArray()[
                  ndarray::view(i_row)()]
                  = (slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)]
                     * D_Weight_Bin0)
                    + (slitFuncsSwaths[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()(I_Bin+1)]
                       * D_Weight_Bin1);
        }
        int int_i_row = static_cast<int>(i_row);
        double dSumSFRow = ndarray::sum(profile.getArray()[ndarray::view(int_i_row)()]);
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ": I_Bin = "
                          << I_Bin << ": dSumSFRow = " << dSumSFRow);
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ": I_Bin = "
                          << I_Bin << ": profile.getArray().getShape() = "
                          << profile.getArray().getShape());
        if (std::fabs(dSumSFRow) >= 0.000001){
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ": I_Bin = "
                            << I_Bin << ": normalizing profile.getArray()[i_row = "
                            << i_row << ", *]");
          profile.getArray()[
                  ndarray::view(int_i_row)()]
                  = profile.getArray()[ndarray::view(int_i_row)()]
                    / dSumSFRow;
        }
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ": I_Bin = "
                          << I_Bin << ": profile.getArray()(" << i_row << ", *) set to "
                          << profile.getArray()[ndarray::view(int_i_row)()]);
      }

      if (i_row == swathBoundsY[I_Bin][1]){
        I_Bin++;
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": i_row = " << i_row << ": I_Bin set to " << I_Bin);
      }
    }/// end for (int i_row = 0; i_row < slitFuncsSwaths.rows(); i_row++){
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": profile.getArray() set to ["
                      << profile.getHeight() << ", " << profile.getWidth() << "]: "
                      << profile.getArray());
    int width = profile.getWidth();
    int xMin, xMax;

    _trace->getImage()->getArray().deep() = 0.0;
    for (int i=0; i<_minCenMax.getShape()[0]; ++i){
        xMax = _minCenMax[i][2] + 1;
        xMin = _minCenMax[i][0];
        if (xMax - xMin != width){
            std::string message("_minCenMax[");
            message += std::to_string(i) + "][2](=" + std::to_string(_minCenMax[i][2]);
            message += ") - _minCenMax[" + std::to_string(i) + "][0](=";
            message += std::to_string(_minCenMax[i][0]) + ") + 1 = ";
            message += std::to_string(_minCenMax[i][2] - _minCenMax[i][0] + 1) +" != width(=";
            message += std::to_string(width);
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        _trace->getImage()->getArray()[ndarray::view(i)(_minCenMax[i][0],
                                                        _minCenMax[i][2] + 1)]
                = profile.getArray()[ndarray::view(i)()];
    }
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  ndarray::Array<float, 2, 1> FiberTrace<ImageT, MaskT, VarianceT>::_calcProfileSwath(
        ndarray::Array<ImageT const, 2, 1> const& imageSwath,
        ndarray::Array<MaskT const, 2, 1> const& maskSwath,
        ndarray::Array<VarianceT const, 2, 1> const& varianceSwath,
        ndarray::Array<float const, 1, 1> const& xCentersSwath,
        size_t const iSwath)
  {
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.FiberTrace._calcProfileSwath");

    /// Check shapes of input arrays
    if (imageSwath.getShape()[0] != maskSwath.getShape()[0]){
      string message("pfs::drp::stella::FiberTrace::_calcProfileSwath: ERROR: imageSwath.getShape()[0](=");
      message += to_string(imageSwath.getShape()[0]) + ") != maskSwath.getShape()[0](=" + to_string(maskSwath.getShape()[0]);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (imageSwath.getShape()[0] != varianceSwath.getShape()[0]){
      string message("pfs::drp::stella::FiberTrace::_calcProfileSwath: ERROR: imageSwath.getShape()[0](=");
      message += to_string(imageSwath.getShape()[0]) + ") != varianceSwath.getShape()[0](=" + to_string(varianceSwath.getShape()[0]);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (imageSwath.getShape()[0] != xCentersSwath.getShape()[0]){
      string message("pfs::drp::stella::FiberTrace::_calcProfileSwath: ERROR: imageSwath.getShape()[0](=");
      message += to_string(imageSwath.getShape()[0]) + ") != xCentersSwath.getShape()[0](=" + to_string(xCentersSwath.getShape()[0]);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (imageSwath.getShape()[1] != maskSwath.getShape()[1]){
      string message("pfs::drp::stella::FiberTrace::_calcProfileSwath: ERROR: imageSwath.getShape()[1](=");
      message += to_string(imageSwath.getShape()[1]) + ") != maskSwath.getShape()[1](=" + to_string(maskSwath.getShape()[1]);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (imageSwath.getShape()[1] != varianceSwath.getShape()[1]){
      string message("pfs::drp::stella::FiberTrace::_calcProfileSwath: ERROR: imageSwath.getShape()[1](=");
      message += to_string(imageSwath.getShape()[1]) + ") != varianceSwath.getShape()[1](=" + to_string(varianceSwath.getShape()[1]);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    /// Normalize rows in imageSwath
    ndarray::Array<float, 2, 1> imageSwathNormalized = ndarray::allocate(imageSwath.getShape()[0], imageSwath.getShape()[1]);
    ndarray::Array<float, 1, 1> sumArr = ndarray::allocate(imageSwath.getShape()[1]);
    for (int iRow = 0; iRow < imageSwath.getShape()[0]; ++iRow){
      sumArr.deep() = ndarray::Array<ImageT const, 1, 1>(imageSwath[ndarray::view(iRow)()]);
      imageSwathNormalized[ndarray::view(iRow)()] = ndarray::Array<ImageT const, 1, 1>(imageSwath[ndarray::view(iRow)()]) / ndarray::sum(sumArr);
    }
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": imageSwath = " << imageSwath);
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": imageSwathNormalized = " << imageSwathNormalized);

    /// Calculate pixel offset to xCenter
    ndarray::Array<float, 1, 1> xCentersTemp = ndarray::allocate(xCentersSwath.getShape()[0]);
    xCentersTemp.deep() = xCentersSwath + 0.5;
    const ndarray::Array<size_t const, 1, 1> xCentersInt = math::floor(xCentersTemp, size_t(0));
    ndarray::Array<float, 1, 1> pixelOffset = ndarray::allocate(xCentersSwath.getShape()[0]);
    pixelOffset.deep() = 0.0;
    pixelOffset.deep() -= xCentersSwath;
    pixelOffset.deep() += xCentersInt;
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": pixelOffset = " << pixelOffset);
    const ndarray::Array<float const, 1, 1> xCenterArrayIndexX =
        math::indGenNdArr(float(imageSwath.getShape()[1]));
    const ndarray::Array<float const, 1, 1> xCenterArrayIndexY =
        math::replicate<float>(1.0, int(imageSwath.getShape()[0]));
    ndarray::Array<float, 2, 1> xArray = ndarray::allocate(imageSwath.getShape()[0], imageSwath.getShape()[1]);
    xArray.asEigen() = xCenterArrayIndexY.asEigen() * xCenterArrayIndexX.asEigen().transpose();
    double xMin = 1.;
    double xMax = -1.;
    auto itOffset = pixelOffset.begin();
    for (auto itX = xArray.begin(); itX != xArray.end(); ++itX){
      for (auto itXY = itX->begin(); itXY != itX->end(); ++itXY){
        *itXY += *itOffset;
        if (*itXY < xMin)
          xMin = *itXY;
        if (*itXY > xMax)
          xMax = *itXY;
      }
      ++itOffset;
    }
    PTR(vector<float>) xVecPtr(new vector<float>());
    xVecPtr->reserve(xArray.getShape()[0] * xArray.getShape()[1]);
    PTR(vector<float>) yVecPtr(new vector<float>());
    yVecPtr->reserve(xArray.getShape()[0] * xArray.getShape()[1]);
    auto itRowIm = imageSwathNormalized.begin();
    for (auto itRow = xArray.begin(); itRow != xArray.end(); ++itRow, ++itRowIm){
      auto itColIm = itRowIm->begin();
      for (auto itCol = itRow->begin(); itCol != itRow->end(); ++itCol, ++itColIm){
        xVecPtr->push_back(*itCol);
        yVecPtr->push_back(static_cast<float>(*itColIm));
      }
    }
    _profileFittingInputXPerSwath.push_back(xVecPtr);
    _profileFittingInputYPerSwath.push_back(yVecPtr);
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": xArray = " << xArray);
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": xMin = " << xMin
                      << ", xMax = " << xMax);
    double xOverSampleStep = 1. / _fiberTraceProfileFittingControl->overSample;
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": initial xOverSampleStep = "
                      << xOverSampleStep);

    ///adjust xOverSampleStep to cover x from xMin + xOverSampleStep/2 to xMax - xOverSampleStep/2
    int nSteps = (xMax - xMin) / xOverSampleStep + 1;
    xOverSampleStep = (xMax - xMin) / nSteps;
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": final xOverSampleStep = "
                      << xOverSampleStep);
    ndarray::Array<float, 1, 1> xOverSampled = ndarray::allocate(nSteps);
    double xStart = xMin + 0.5*xOverSampleStep;
    int iStep = 0;
    for (auto it = xOverSampled.begin(); it != xOverSampled.end(); ++it){
      *it = xStart + (iStep * xOverSampleStep);
      ++iStep;
    }
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": xOverSampled = "
                      << xOverSampled);
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                      << ": _fiberTraceProfileFittingControl->maxIterSig = "
                      << _fiberTraceProfileFittingControl->maxIterSig);
    PTR(vector<float>) xOverSampledFitVec(new vector<float>(xOverSampled.begin(), xOverSampled.end()));
    _overSampledProfileFitXPerSwath.push_back(xOverSampledFitVec);

    /// calculate oversampled profile values
    iStep = 0;
    std::vector<std::pair<float, float>> valOverSampledVec;
    int iStepsWithValues = 0;
    int bThisStepHasValues = false;
    ImageT mean = 0.;
    double rangeStart = xOverSampled[0] - 0.5*xOverSampleStep;
    double rangeEnd = rangeStart + xOverSampleStep;
    for (auto it = xOverSampled.begin(); it != xOverSampled.end(); ++it, ++iStep){
      bThisStepHasValues = false;
      if (iStep == nSteps - 1)
        rangeEnd += xOverSampleStep / 100.;
      size_t iterSig = 0;
      size_t nValues = 0;
      #ifdef __DEBUG_CALCPROFILESWATH__
        for (int iRow = 0; iRow < xArray.getShape()[0]; ++iRow){
          printf("xArray[%i][*] = ",iRow);
          for (int iCol = 0; iCol < xArray.getShape()[1]; ++iCol){
            printf("%.9f ", xArray[iRow][iCol]);
          }
          printf("\n");
        }
        cout << "FiberTrace::_calcProfileSwath: iSwath = " << iSwath << ": iStep" << iStep << ": rangeStart = " << rangeStart << ", rangeEnd = " << rangeEnd << endl;
        printf("rangeStart = %.9f, rangeEnd = %.9f\n", rangeStart, rangeEnd);
      #endif
      ndarray::Array<size_t, 2, 1> indicesInValueRange =
          math::getIndicesInValueRange<float>(xArray, rangeStart, rangeEnd);
      LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                        << ": iStep" << iStep << ": indicesInValueRange = " << indicesInValueRange);
      std::vector< std::pair<size_t, size_t> > indicesInValueRangeVec(indicesInValueRange.getShape()[0]);
      for (int i = 0; i < indicesInValueRange.getShape()[0]; ++i){
        indicesInValueRangeVec[i].first = indicesInValueRange[i][0];
        indicesInValueRangeVec[i].second = indicesInValueRange[i][1];
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                          << ": iStep" << iStep << ": indicesInValueRangeVec[" << i << "].first = "
                          << indicesInValueRangeVec[i].first << ", indicesInValueRangeVec[" << i
                          << "].second = " << indicesInValueRangeVec[i].second);
      }
      do{
        ndarray::Array<float, 1, 1> subArr = math::getSubArray(imageSwathNormalized, indicesInValueRangeVec);
        ndarray::Array<float, 1, 1> xSubArr = math::getSubArray(xArray, indicesInValueRangeVec);
        nValues = subArr.getShape()[0];
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                          << ": iStep" << iStep << ": iterSig = " << iterSig << ": nValues = " << nValues);
        if (nValues > 1){
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                            << ": iStep = " << iStep << ": iterSig = " << iterSig << ": xSubArr = ["
                            << xSubArr.getShape()[0] << "]: " << xSubArr);
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                            << ": iStep = " << iStep << ": iterSig = " << iterSig << ": subArr = ["
                            << subArr.getShape()[0] << "]: " << subArr);
          if (_fiberTraceProfileFittingControl->maxIterSig > iterSig){
            ndarray::Array<float, 1, 1> moments = math::moment(subArr, 2);
            LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                              << ": iStep = " << iStep << ": iterSig = " << iterSig << ": moments = "
                              << moments);
            for (int i = subArr.getShape()[0] - 1; i >= 0; --i){
              LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                                << ": iStep" << iStep << ": iterSig = " << iterSig << ": moments[0](="
                                << moments[0] << ") - subArr[" << i << "](=" << subArr[i] << ") = "
                                << moments[0] - subArr[i]
                                << ", 0. - (_fiberTraceProfileFittingControl->upperSigma(="
                                << _fiberTraceProfileFittingControl->upperSigma << ") * sqrt(moments[1](="
                                << moments[1] << "))(= " << sqrt(moments[1]) << ") = "
                                << 0. - (_fiberTraceProfileFittingControl->upperSigma * sqrt(moments[1])));
              LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                                << ": iStep" << iStep << ": iterSig = " << iterSig
                                << ": _fiberTraceProfileFittingControl->lowerSigma(="
                                << _fiberTraceProfileFittingControl->lowerSigma << ") * sqrt(moments[1](="
                                << moments[1] << "))(= " << sqrt(moments[1]) << ") = "
                                << _fiberTraceProfileFittingControl->lowerSigma * sqrt(moments[1]));
              if ((moments[0] - subArr[i] < 0. - (_fiberTraceProfileFittingControl->upperSigma * sqrt(moments[1])))
               || (moments[0] - subArr[i] > (_fiberTraceProfileFittingControl->lowerSigma * sqrt(moments[1])))){
                LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                                  << ": iStep = " << iStep << ": iterSig = " << iterSig
                                  << ": rejecting element " << i << "from subArr");
                indicesInValueRangeVec.erase(indicesInValueRangeVec.begin() + i);
              }
            }
          }
          ndarray::Array<float, 1, 1> moments = math::moment(subArr, 1);
          mean = moments[0];
          LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                            << ": iStep = " << iStep << ": iterSig = " << iterSig << ": mean = " << mean);
          ++iStepsWithValues;
          bThisStepHasValues = true;
        }
        ++iterSig;
      } while (iterSig <= _fiberTraceProfileFittingControl->maxIterSig);
      if (bThisStepHasValues){
        valOverSampledVec.push_back(std::pair<float, float>(*it, mean));
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                          << ": valOverSampledVec[" << iStep << "] = (" << valOverSampledVec[iStep].first
                          << ", " << valOverSampledVec[iStep].second << ")");
      }
      rangeStart = rangeEnd;
      rangeEnd += xOverSampleStep;
    }
    ndarray::Array<float, 1, 1> valOverSampled = ndarray::allocate(valOverSampledVec.size());
    ndarray::Array<float, 1, 1> xValOverSampled = ndarray::allocate(valOverSampledVec.size());
    for (int iRow = 0; iRow < valOverSampledVec.size(); ++iRow){
      xValOverSampled[iRow] = valOverSampledVec[iRow].first;
      valOverSampled[iRow] = valOverSampledVec[iRow].second;
      LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                        << ": (x)valOverSampled[" << iRow << "] = (" << xValOverSampled[iRow]
                        << "," << valOverSampled[iRow] << ")");
    }
    PTR(std::vector<float>) xVecMean(new vector<float>(xValOverSampled.begin(), xValOverSampled.end()));
    std::vector<ImageT> yVecMean(valOverSampled.begin(), valOverSampled.end());

    PTR(std::vector<float>) yVecMeanF(new vector<float>(yVecMean.size()));
    auto itF = yVecMeanF->begin();
    for (auto itT = yVecMean.begin(); itT != yVecMean.end(); ++itT, ++itF){
        *itF = static_cast<float>(*itT);
    }
    _profileFittingInputXMeanPerSwath.push_back(xVecMean);
    _profileFittingInputYMeanPerSwath.push_back(yVecMeanF);

    math::spline<float> spline(*xVecMean, *yVecMeanF, math::spline<float>::CUBIC_NATURAL); // X must be sorted

    PTR(vector<float>) yOverSampledFitVec(new vector<float>(nSteps));
    _overSampledProfileFitYPerSwath.push_back(yOverSampledFitVec);
    for (auto itX = xOverSampledFitVec->begin(), itY = yOverSampledFitVec->begin(); itX != xOverSampledFitVec->end(); ++itX, ++itY)
      *itY = spline(*itX);
    #ifdef __DEBUG_CALCPROFILESWATH__
      std::vector<float> yVecFit(yVecMean.size());
      for (int iRow = 0; iRow < yVecMean.size(); ++iRow){
        yVecFit[iRow] = spline((*xVecMean)[iRow]);
        cout << "FiberTrace::_calcProfileSwath: iSwath = " << iSwath << ": yVecMean[" << iRow << "] = " << yVecMean[iRow] << ", yVecFit[" << iRow << "] = " << yVecFit[iRow] << endl;
      }
    #endif

    /// calculate profile for each row in imageSwath
    ndarray::Array<float, 2, 1> profArraySwath = ndarray::allocate(imageSwath.getShape()[0], imageSwath.getShape()[1]);
    double tmpVal = 0.0;
    for (int iRow = 0; iRow < imageSwath.getShape()[0]; ++iRow){
      LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                        << ": xArray[" << iRow << "][*] = " << xArray[ndarray::view(iRow)()]);
      for (int iCol = 0; iCol < imageSwath.getShape()[1]; ++iCol){
        /// The spline's knots are calculated from bins in x centered at the
        /// oversampled positions in x.
        /// Outside the range in x on which the spline is defined, which is
        /// [min(xRange) + overSample/2., max(xRange) - overSample/2.], so to
        /// say in the outer (half overSample), the spline is extrapolated from
        /// the 1st derivative at the end points.
        tmpVal = spline(xArray[iRow][iCol]);

        /// Set possible negative profile values to Zero as they are not physical
        profArraySwath[iRow][iCol] = (tmpVal >= 0. ? tmpVal : 0.);
        #ifdef __DEBUG_CALCPROFILESWATH__
          if (xArray[iRow][iCol] < (*xVecMean)[0]){
            cout << "FiberTrace::_calcProfileSwath: xArray[" << iRow << "][" << iCol << "] = " << xArray[iRow][iCol] << endl;
            cout << "FiberTrace::_calcProfileSwath: profArraySwath[" << iRow << "][" << iCol << "] = " << profArraySwath[iRow][iCol] << endl;
          }
        #endif
      }
      LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                        << ": profArraySwath[" << iRow << "][*] = " << profArraySwath[ndarray::view(iRow)()]);
      profArraySwath[ndarray::view(iRow)()] = profArraySwath[ndarray::view(iRow)()] / ndarray::sum(profArraySwath[ndarray::view(iRow)()]);
      LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                        << ": normalized profArraySwath[" << iRow << "][*] = "
                        << profArraySwath[ndarray::view(iRow)()]);
    }
    LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                      << ": profArraySwath = " << profArraySwath);
    return profArraySwath;
  }

    template<typename ImageT, typename MaskT, typename VarianceT>
    void FiberTrace<ImageT, MaskT, VarianceT>::_markFiberTraceInMask(MaskT value){
        /// Call getMinCenMax which will reconstruct _minCenMax in case it is empty
        ndarray::Array<size_t, 2, -2> minCenMax = _getMinCenMax();

        for (int y = 0; y < minCenMax.getShape()[0]; ++y){
            for (int x = minCenMax[y][0]; x <= minCenMax[y][2]; ++x){
                (*_trace->getMask())(x, y) |= value;
            }
        }
    }

    template<typename ImageT, typename MaskT, typename VarianceT>
    ndarray::Array<size_t, 2, -2> FiberTrace<ImageT, MaskT, VarianceT>::_getMinCenMax(){
        if (_minCenMax.asEigen().maxCoeff() == 0){
            _reconstructMinCenMax();
        }
        ndarray::Array<size_t, 2, -2> minCenMax(_minCenMax.getShape());
        minCenMax.deep() = _minCenMax;
        return minCenMax;
    }

    template<typename ImageT, typename MaskT, typename VarianceT>
    void FiberTrace<ImageT, MaskT, VarianceT>::_reconstructMinCenMax(){
        LOG_LOGGER _log = LOG_GET("pfs.drp.stella.FiberTrace._reconstructMinCenMax");
        _minCenMax = ndarray::allocate(_trace->getImage()->getHeight(), 3);
        ndarray::Array<size_t, 1, 1> xMin = ndarray::allocate(_minCenMax.getShape()[0]);
        ndarray::Array<size_t, 1, 1> xMax = ndarray::allocate(_minCenMax.getShape()[0]);
        ndarray::Array<size_t, 1, 1> xCen = ndarray::allocate(_minCenMax.getShape()[0]);

        const auto ftMask = _trace->getMask()->getPlaneBitMask("FIBERTRACE");
        bool xMinFound;
        bool xMaxFound;
        int iY = 0;
        auto itMaskRow = _trace->getMask()->getArray().begin();
        for (auto itXMin=xMin.begin(), itXMax=xMax.begin(), itXCen=xCen.begin();
             itXMin != xMin.end();
             ++itXMin, ++itXMax, ++itXCen, ++itMaskRow, ++iY) {
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": *itMaskRow = " << *itMaskRow);
            xMinFound = false;
            xMaxFound = false;
            int iX=0;
            for (auto itMaskCol=itMaskRow->begin(); itMaskCol!=itMaskRow->end(); ++itMaskCol, ++iX){
                if (*itMaskCol & ftMask){
                    if (!xMinFound){
                        *itXMin = iX;
                        xMinFound = true;
                        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iY = " << iY << ", iX = " << iX << ": xMinFound");
                    }
                    *itXMax = iX;
                }
            }
            if (*itXMax > 0){
                xMaxFound = true;
                LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": iY = " << iY << ", iX = " << iX << ": xMaxFound");
            }
            if (!xMinFound || !xMaxFound){
                std::string message("_fiberId = ");
                message += to_string(_fiberId) + ": xMin or xMax not found";
                throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
            *itXCen = size_t((*itXMin + *itXMax) / 2);
        }
        _minCenMax[ndarray::view()(0)] = xMin;
        _minCenMax[ndarray::view()(2)] = xMax;
        _minCenMax[ndarray::view()(1)] = xCen;
        LOGLS_DEBUG(_log, "_fiberId = " << _fiberId << ": _minCenMax = " << _minCenMax);
    }

  /**
   * class FiberTraceSet
   **/
  template<typename ImageT, typename MaskT, typename VarianceT>
  FiberTraceSet<ImageT, MaskT, VarianceT>::FiberTraceSet(FiberTraceSet<ImageT, MaskT, VarianceT> const &fiberTraceSet, bool const deep)
      : _traces(fiberTraceSet.getTraces())
  {
    if (deep){
      PTR(std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>) ptr(new std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>(fiberTraceSet.getNtrace()));
      _traces.reset();
      _traces = ptr;
      for (size_t i = 0; i < fiberTraceSet.getNtrace(); ++i){
        PTR(FiberTrace<ImageT, MaskT, VarianceT>) tra(new FiberTrace<ImageT, MaskT, VarianceT>(*(fiberTraceSet.getFiberTrace(i)), true));
        (*_traces)[i] = tra;
      }
    }
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  void FiberTraceSet<ImageT, MaskT, VarianceT>::setFiberTrace(const size_t i,     ///< which aperture?
                                                              const PTR(FiberTrace<ImageT, MaskT, VarianceT>) &trace ///< the FiberTrace for the ith aperture
  ){
    if (i > static_cast<int>(_traces->size())){
      string message("FiberTraceSet::setFiberTrace: ERROR: position for trace outside range!");
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (i == static_cast<int>(_traces->size())){
      _traces->push_back(trace);
    } else {
      (*_traces)[i] = trace;
    }
  }

  template< typename ImageT, typename MaskT, typename VarianceT >
  PTR( FiberTrace< ImageT, MaskT, VarianceT >) FiberTraceSet< ImageT, MaskT, VarianceT >::getFiberTrace( const size_t i ){
    if (i >= _traces->size()){
      string message("FiberTraceSet::getFiberTrace(i=");
      message += to_string(i) + string("): ERROR: i > _traces->size()=") + to_string(_traces->size());
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return _traces->at(i);
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(FiberTrace<ImageT, MaskT, VarianceT>) const FiberTraceSet<ImageT, MaskT, VarianceT>::getFiberTrace(const size_t i) const {
    if (i >= _traces->size()){
      string message("FiberTraceSet::getFiberTrace(i=");
      message += to_string(i) + "): ERROR: i > nTrace=" + to_string(_traces->size());
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return _traces->at(i);
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  void FiberTraceSet<ImageT, MaskT, VarianceT>::addFiberTrace(const PTR(FiberTrace<ImageT, MaskT, VarianceT>) &trace, const size_t fiberId) ///< the FiberTrace for the ith aperture
  {
    size_t nTrace = getNtrace();
    _traces->push_back(trace);
    if (_traces->size() == nTrace) {
      string message("FiberTraceSet::addFiberTrace: ERROR: could not add trace to _traces");
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (fiberId > 0){
      (*_traces)[nTrace]->setFiberId(fiberId);
    }
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  void FiberTraceSet< ImageT, MaskT, VarianceT >::sortTracesByXCenter()
  {
    std::vector<float> xCenters;
    for (int fiberId = 0; fiberId < static_cast<int>(_traces->size()); ++fiberId) {
        auto const bbox = (*_traces)[fiberId]->getTrace()->getBBox();

        xCenters.push_back(0.5*(bbox.getMinX() + bbox.getMaxX())); // fixed typo Min -> Max
    }
    std::vector<int> sortedIndices(xCenters.size());
    sortedIndices = ::pfs::drp::stella::math::sortIndices(xCenters);

    std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)> sortedTraces(_traces->size());
    for (size_t i = 0; i < _traces->size(); ++i){
      sortedTraces[ i ] = ( *_traces )[ sortedIndices[ i ] ];
    }
    _traces.reset(new std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>(sortedTraces));

    return;
  }

  namespace math {
    template<typename ImageT, typename MaskT, typename VarianceT>
    PTR(FiberTraceSet<ImageT, MaskT, VarianceT>) findAndTraceApertures(
        const PTR(const afwImage::MaskedImage<ImageT, MaskT, VarianceT>) &maskedImage,
        DetectorMap const& detectorMap,                          
        const PTR(const FiberTraceFunctionFindingControl) &fiberTraceFunctionFindingControl,
        const PTR(FiberTraceProfileFittingControl) &fiberTraceProfileFittingControl)
    {
      LOG_LOGGER _log = LOG_GET("pfs.drp.stella.math.findAndTraceApertures");
      LOGLS_TRACE(_log, "::pfs::drp::stella::math::findAndTraceApertures started");

      if (static_cast<int>(fiberTraceFunctionFindingControl->apertureFWHM * 2.) + 1
          <= fiberTraceFunctionFindingControl->nTermsGaussFit)
      {
        std::string message("fiberTraceFunctionFindingControl->apertureFWHM too small for GaussFit ");
        message += "-> Try lower fiberTraceFunctionFindingControl->nTermsGaussFit!";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      PTR(FiberTraceSet<ImageT, MaskT, VarianceT>) fiberTraceSet(new FiberTraceSet<ImageT, MaskT, VarianceT>());

      afwImage::MaskedImage<ImageT, MaskT, VarianceT> maskedImageCopy(*maskedImage, true);
      PTR(afwImage::Image<ImageT>) ccdImage = maskedImageCopy.getImage();
      PTR(afwImage::Image<VarianceT>) ccdVarianceImage = maskedImageCopy.getVariance();
      ndarray::Array<ImageT, 2, 1> ccdArray = ndarray::copy(ccdImage->getArray());
      bool B_ApertureFound;

      std::vector<std::string> keywords(1);
      keywords[0] = "XRANGE";
      std::vector<void*> args(1);
      ndarray::Array<float, 1, 1> xRange = ndarray::allocate(2);
      PTR(ndarray::Array<float, 1, 1>) p_xRange(new ndarray::Array<float, 1, 1>(xRange));
      args[0] = &p_xRange;

      /// Set all pixels below fiberTraceFunctionFindingControl->signalThreshold to 0.
      for (auto i = ccdArray.begin(); i != ccdArray.end(); ++i){
        for (auto j = i->begin(); j != i->end(); ++j){
          if (*j < fiberTraceFunctionFindingControl->signalThreshold){
            *j = 0;
          }
        }
      }
      do{
        PTR(FiberTraceFunction) fiberTraceFunction(new FiberTraceFunction());
        fiberTraceFunction->fiberTraceFunctionControl.reset();
        fiberTraceFunction->fiberTraceFunctionControl =
                fiberTraceFunctionFindingControl->fiberTraceFunctionControl;
        LOGLS_TRACE(_log, "fiberTraceFunction.fiberTraceFunctionControl set");
        FindCenterPositionsOneTraceResult result = findCenterPositionsOneTrace( ccdImage,
                                                                                ccdVarianceImage,
                                                                                fiberTraceFunctionFindingControl );
        if (result.apertureCenterIndex.size() > fiberTraceFunctionFindingControl->minLength){
          B_ApertureFound = true;
          const ndarray::Array<float, 1, 1> D_A1_ApertureCenterIndex = vectorToNdArray(result.apertureCenterIndex);
          const ndarray::Array<float, 1, 1> D_A1_ApertureCenterPos = vectorToNdArray(result.apertureCenterPos);
          const ndarray::Array<float, 1, 1> D_A1_EApertureCenterPos = vectorToNdArray(result.eApertureCenterPos);

          LOGLS_TRACE(_log, "D_A1_ApertureCenterIndex = " << D_A1_ApertureCenterIndex);
          LOGLS_TRACE(_log, "D_A1_ApertureCenterPos = " << D_A1_ApertureCenterPos);

          fiberTraceFunction->xCenter = D_A1_ApertureCenterPos[int(D_A1_ApertureCenterIndex.size()/2.)];
          fiberTraceFunction->yCenter = int(D_A1_ApertureCenterIndex[int(D_A1_ApertureCenterIndex.size()/2.)]);
          fiberTraceFunction->yHigh = int(D_A1_ApertureCenterIndex[int(D_A1_ApertureCenterIndex.size()-1)] - fiberTraceFunction->yCenter);
          fiberTraceFunction->yLow = int(D_A1_ApertureCenterIndex[0]) - fiberTraceFunction->yCenter;
          LOGLS_TRACE(_log, "fiberTraceFunction->xCenter = " << fiberTraceFunction->xCenter);
          LOGLS_TRACE(_log, "fiberTraceFunction->yCenter = " << fiberTraceFunction->yCenter);
          LOGLS_TRACE(_log, "fiberTraceFunction->yHigh = " << fiberTraceFunction->yHigh);
          LOGLS_TRACE(_log, "fiberTraceFunction->yLow = " << fiberTraceFunction->yLow);

          if (fiberTraceFunction->fiberTraceFunctionControl->interpolation.compare("POLYNOMIAL") == 0)
          {
            /// Fit Polynomial
            (*p_xRange)[0] = D_A1_ApertureCenterIndex[0];
            (*p_xRange)[1] = D_A1_ApertureCenterIndex[int(D_A1_ApertureCenterIndex.size()-1)];
            fiberTraceFunction->coefficients = math::PolyFit(
                    D_A1_ApertureCenterIndex,
                    D_A1_ApertureCenterPos,
                    fiberTraceFunctionFindingControl->fiberTraceFunctionControl->order,
                    keywords,
                    args);
            LOGLS_TRACE(_log, "after PolyFit: fiberTraceFunction->coefficients = "
                              << fiberTraceFunction->coefficients);
          }
          else{
            std::string message("fiberTraceFunction->fiberTraceFunctionControl->interpolation ");
            message += fiberTraceFunction->fiberTraceFunctionControl->interpolation;
            message += " not supported. Please use POLYNOMIAL";
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }

          PTR(FiberTrace<ImageT, MaskT, VarianceT >) fiberTrace(new FiberTrace< ImageT, MaskT, VarianceT >(
                  maskedImage,
                  fiberTraceFunction,
                  fiberTraceProfileFittingControl));
          fiberTrace->setFiberId( fiberTraceSet->getNtrace() );
          fiberTraceSet->addFiberTrace(fiberTrace);
        } else {
          B_ApertureFound = false;
        }
      } while (B_ApertureFound);
      //
      // Set the fiberIds
      //
      for (auto fiberTrace: *fiberTraceSet->getTraces()) {
        const auto xCenters = fiberTrace->getXCenters();
        const int midRow = xCenters.getShape()[0]/2;
        const lsst::afw::geom::Point<double> cen(xCenters[midRow], midRow);
        fiberTrace->setFiberId(detectorMap.findFiberId(cen));
      }

      fiberTraceSet->sortTracesByXCenter();

      return fiberTraceSet;
    }

    template<typename ImageT, typename VarianceT>
    FindCenterPositionsOneTraceResult findCenterPositionsOneTrace( PTR(afwImage::Image<ImageT>) & ccdImage,
                                                                   PTR(afwImage::Image<VarianceT>) & ccdVarianceImage,
                                                                   PTR(const FiberTraceFunctionFindingControl) const& fiberTraceFunctionFindingControl){
      ndarray::Array<ImageT, 2, 1> ccdArray = ccdImage->getArray();
      ndarray::Array<VarianceT, 2, 1> ccdVarianceArray = ccdVarianceImage->getArray();
      int I_MinWidth = int(1.5 * fiberTraceFunctionFindingControl->apertureFWHM);
      if (I_MinWidth < fiberTraceFunctionFindingControl->nTermsGaussFit)
        I_MinWidth = fiberTraceFunctionFindingControl->nTermsGaussFit;
      float D_MaxTimesApertureWidth = 4.0;
      std::vector<float> gaussFitVariances(0);
      std::vector<float> gaussFitMean(0);
      ndarray::Array<float, 1, 1> xCorRange = ndarray::allocate(2);
      xCorRange[0] = -0.5;
      xCorRange[1] = 0.5;
      float xCorMinPos = 0.;
      int nInd = 100;
      ndarray::Array<float, 2, 1> indGaussArr = ndarray::allocate(nInd, 2);
      std::vector<pfs::drp::stella::math::dataXY<float>> xySorted(0);
      xySorted.reserve(((fiberTraceFunctionFindingControl->apertureFWHM * D_MaxTimesApertureWidth) + 2) * ccdImage->getHeight());

      int I_StartIndex;
      int I_FirstWideSignal;
      int I_FirstWideSignalEnd;
      int I_FirstWideSignalStart;
      unsigned int I_Length, I_ApertureLost;
      int I_ApertureLength;
      int I_RowBak;
      size_t apertureLength = 0;
      bool B_ApertureFound;
      ndarray::Array<float, 1, 1> D_A1_IndexCol =
          math::indGenNdArr(static_cast<float>(ccdArray.getShape()[1]));
      #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: D_A1_IndexCol = " << D_A1_IndexCol << endl;
      #endif
      ndarray::Array<float, 1, 1> D_A1_Guess = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit);
      ndarray::Array<float, 1, 1> D_A1_GaussFit_Coeffs = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit);
      ndarray::Array<float, 1, 1> D_A1_GaussFit_Coeffs_Bak = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit);
      ndarray::Array<int, 1, 1> I_A1_Signal = ndarray::allocate(ccdArray.getShape()[1]);
      I_A1_Signal[ndarray::view()] = 0;
      ndarray::Array<float, 1, 1> D_A1_ApertureCenter = ndarray::allocate(ccdArray.getShape()[0]);
      ndarray::Array<float, 1, 1> D_A1_EApertureCenter = ndarray::allocate(ccdArray.getShape()[0]);
      ndarray::Array<int, 1, 1> I_A1_ApertureCenterIndex = ndarray::allocate(ccdArray.getShape()[0]);
      ndarray::Array<int, 1, 1> I_A1_IndSignal;
      #if defined(__DEBUG_FINDANDTRACE__)
        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: started" << endl;
      #endif
      ndarray::Array<int, 1, 1> I_A1_Ind;
      ndarray::Array<int, 1, 1> I_A1_Where;
      #if defined(__DEBUG_FINDANDTRACE__)
        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: fiberTraceFunctionFindingControl->signalThreshold = " << fiberTraceFunctionFindingControl->signalThreshold << endl;
      #endif

      FindCenterPositionsOneTraceResult result;
      result.apertureCenterIndex.reserve(ccdArray.getShape()[0]);
      result.apertureCenterPos.reserve(ccdArray.getShape()[0]);
      result.eApertureCenterPos.reserve(ccdArray.getShape()[0]);

      /// Search for Apertures
      D_A1_ApertureCenter[ndarray::view()] = 0.;
      D_A1_EApertureCenter[ndarray::view()] = 0.;
      I_A1_ApertureCenterIndex[ndarray::view()] = 0;
      auto itIm = ccdArray.begin();
      for (int i_Row = 0; i_Row < ccdArray.getShape()[0]; i_Row++){
        #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
          cout << "i_Row = " << i_Row << ": ccdArray[i_Row][*] = " << ccdArray[ndarray::view(i_Row)()] << endl;
        #endif
        I_RowBak = i_Row;
        I_StartIndex = 0;
        B_ApertureFound = false;
        for (int i_Col = 0; i_Col < ccdArray.getShape()[1]; ++i_Col){
          if (i_Col == 0){
            if (ccdArray[i_Row][i_Col] > fiberTraceFunctionFindingControl->signalThreshold){
              I_A1_Signal[i_Col] = 1;
            }
            else{
              I_A1_Signal[i_Col] = 0;
            }
          }
          else{
            if (ccdArray[i_Row][i_Col] > fiberTraceFunctionFindingControl->signalThreshold){
              if (I_A1_Signal[i_Col - 1] > 0){
                I_A1_Signal[i_Col] = I_A1_Signal[i_Col - 1] + 1;
              }
              else{
                I_A1_Signal[i_Col] = 1;
              }
            }
          }
        }
        while (!B_ApertureFound){
          gaussFitVariances.resize(0);
          gaussFitMean.resize(0);
          #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
            cout << "pfs::drp::stella::math::findAndTraceApertures: while: I_A1_Signal = " << I_A1_Signal << endl;
            cout << "pfs::drp::stella::math::findAndTraceApertures: while: I_MinWidth = " << I_MinWidth << endl;
            cout << "pfs::drp::stella::math::findAndTraceApertures: while: I_StartIndex = " << I_StartIndex << endl;
          #endif
          I_FirstWideSignal = math::firstIndexWithValueGEFrom(I_A1_Signal, I_MinWidth, I_StartIndex);
          #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": I_FirstWideSignal found at index " << I_FirstWideSignal << ", I_StartIndex = " << I_StartIndex << endl;
          #endif
          if (I_FirstWideSignal < 0){
            #if defined(__DEBUG_FINDANDTRACE__)
              cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": No Aperture found in row " << i_Row << ", trying next row" << endl;
            #endif
            break;
          }
          else{
            I_FirstWideSignalStart = ::pfs::drp::stella::math::lastIndexWithZeroValueBefore(I_A1_Signal, I_FirstWideSignal) + 1;
            #if defined(__DEBUG_FINDANDTRACE__)
              cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": 1. I_FirstWideSignalStart = " << I_FirstWideSignalStart << endl;
            #endif

            I_FirstWideSignalEnd = math::firstIndexWithZeroValueFrom(I_A1_Signal, I_FirstWideSignal) - 1;
            #if defined(__DEBUG_FINDANDTRACE__)
              cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
            #endif

            if (I_FirstWideSignalStart < 0){
              #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": WARNING: No start of aperture found -> Going to next Aperture." << endl;
              #endif

              if (I_FirstWideSignalEnd < 0){
                #if defined(__DEBUG_FINDANDTRACE__)
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": 1. WARNING: No end of aperture found -> Going to next row." << endl;
                #endif
                break;
              }
              else{
                #if defined(__DEBUG_FINDANDTRACE__)
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": End of first wide signal found at index " << I_FirstWideSignalEnd << endl;
                #endif
                /// Set start index for next run
                I_StartIndex = I_FirstWideSignalEnd+1;
              }
            }
            else{ /// Fit Gaussian and Trace Aperture
              if (I_FirstWideSignalEnd < 0){
                #if defined(__DEBUG_FINDANDTRACE__)
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": 2. WARNING: No end of aperture found -> Going to next row." << endl;
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": B_ApertureFound = " << B_ApertureFound << endl;
                #endif
                break;
              }
              else{
                #if defined(__DEBUG_FINDANDTRACE__)
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": End of first wide signal found at index " << I_FirstWideSignalEnd << endl;
                #endif

                if (I_FirstWideSignalEnd - I_FirstWideSignalStart + 1 > fiberTraceFunctionFindingControl->apertureFWHM * D_MaxTimesApertureWidth){
                  I_FirstWideSignalEnd = I_FirstWideSignalStart + int(D_MaxTimesApertureWidth * fiberTraceFunctionFindingControl->apertureFWHM);
                }
                #if defined(__DEBUG_FINDANDTRACE__)
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": I_FirstWideSignalStart = " << I_FirstWideSignalStart << ", I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                #endif
                I_A1_Signal[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)] = 0;

                /// Set start index for next run
                I_StartIndex = I_FirstWideSignalEnd+1;
              }
              I_Length = I_FirstWideSignalEnd - I_FirstWideSignalStart + 1;
              #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": I_Length = " << I_Length << endl;
              #endif
              if (fiberTraceFunctionFindingControl->nTermsGaussFit == 0){/// look for maximum only
                D_A1_ApertureCenter[ndarray::view()] = 0.;
                D_A1_EApertureCenter[ndarray::view()] = 0.;
                B_ApertureFound = true;
                int maxPos = 0;
                ImageT tMax = 0.;
                for (int i = I_FirstWideSignalStart; i <= I_FirstWideSignalEnd; ++i){
                  if (ccdArray[i_Row][i] > tMax){
                    tMax = ccdArray[i_Row][i];
                    maxPos = i;
                  }
                }
                D_A1_ApertureCenter[i_Row] = maxPos;
                D_A1_EApertureCenter[i_Row] = 0.5;
                #if defined(__DEBUG_FINDANDTRACE__)
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": Aperture found at " << D_A1_ApertureCenter[i_Row] << endl;
                #endif

                /// Set signal to zero
                if ((I_FirstWideSignalEnd - 1) >=(I_FirstWideSignalStart + 1))
                  ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
              }
              else{// if (fiberTraceFunctionFindingControl->nTermsGaussFit > 0){
                if (I_Length <= fiberTraceFunctionFindingControl->nTermsGaussFit){
                  #if defined(__DEBUG_FINDANDTRACE__)
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": WARNING: Width of aperture <= " << fiberTraceFunctionFindingControl->nTermsGaussFit << "-> abandoning aperture" << endl;
                  #endif

                  /// Set signal to zero
                  if ((I_FirstWideSignalEnd - 1) >=(I_FirstWideSignalStart + 1))
                    ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                }
                else{
                  /// populate Arrays for GaussFit
                  ndarray::Array<float, 1, 1> D_A1_X = copy(ndarray::Array<float, 1, 1>(D_A1_IndexCol[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)]));
                  ndarray::Array<float, 1, 1> D_A1_Y = ndarray::allocate(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
                  D_A1_Y.deep() = ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)];
                  for (auto it = D_A1_Y.begin(); it != D_A1_Y.end(); ++it){
                    if (*it < 0.000001)
                      *it = 1.;
                  }
                  #if defined(__DEBUG_FINDANDTRACE__)
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: 1. D_A1_Y set to " << D_A1_Y << endl;
                  #endif
                  ndarray::Array<VarianceT, 1, 1> T_A1_MeasureErrors = ndarray::allocate(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
                  T_A1_MeasureErrors.deep() = ccdVarianceArray[ndarray::view(i_Row)(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)];
                  ndarray::Array<float, 1, 1> D_A1_MeasureErrors = ndarray::allocate(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
                  for (int ooo = 0; ooo < I_FirstWideSignalEnd - I_FirstWideSignalStart + 1; ++ooo){
                    if (T_A1_MeasureErrors[ooo] > 0)
                      D_A1_MeasureErrors[ooo] = sqrt(T_A1_MeasureErrors[ooo]);
                    else
                      D_A1_MeasureErrors[ooo] = 1;
                  }

                  /// Guess values for GaussFit
                  if (fiberTraceFunctionFindingControl->nTermsGaussFit == 3){
                    D_A1_Guess[0] = max(D_A1_Y);
                    D_A1_Guess[1] = I_FirstWideSignalStart +
                        0.5*(I_FirstWideSignalEnd - I_FirstWideSignalStart);
                    D_A1_Guess[2] = 0.5*fiberTraceFunctionFindingControl->apertureFWHM;
                  }
                  else if (fiberTraceFunctionFindingControl->nTermsGaussFit > 3){
                     D_A1_Guess[3] = std::min(D_A1_Y[0], D_A1_Y[D_A1_Y.getShape()[0]-1]);
                    if (D_A1_Guess[3] < 0.)
                      D_A1_Guess[3] = 0.1;
                    if (fiberTraceFunctionFindingControl->nTermsGaussFit > 4)
                      D_A1_Guess[4] = (D_A1_Y[D_A1_Y.getShape()[0]-1] - D_A1_Y[0]) / (D_A1_Y.getShape()[0] - 1);
                  }

                  D_A1_GaussFit_Coeffs[ndarray::view()] = 0.;
                  ndarray::Array<float, 1, 1> D_A1_GaussFit_ECoeffs = ndarray::allocate(D_A1_GaussFit_Coeffs.size());
                  D_A1_GaussFit_ECoeffs[ndarray::view()] = 0.;

                  #if defined(__DEBUG_FINDANDTRACE__)
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": D_A1_X = " << D_A1_X << ", D_A1_Y = " << D_A1_Y << endl;
                  #endif

                  ndarray::Array<int, 2, 1> I_A2_Limited = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit, 2);
                  I_A2_Limited[ndarray::view()] = 1;
                  ndarray::Array<float, 2, 1> D_A2_Limits = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit, 2);
                  D_A2_Limits[0][0] = 0.;/// Peak lower limit
                  D_A2_Limits[0][1] = 2. * D_A1_Guess[0];/// Peak upper limit
                  D_A2_Limits[1][0] = I_FirstWideSignalStart;      /// Centroid lower limit
                  D_A2_Limits[1][1] = I_FirstWideSignalEnd;/// Centroid upper limit
                  D_A2_Limits[2][0] = 0.25*fiberTraceFunctionFindingControl->apertureFWHM;/// Sigma lower limit
                  D_A2_Limits[2][1] = fiberTraceFunctionFindingControl->apertureFWHM; /// Sigma upper limit
                  if (fiberTraceFunctionFindingControl->nTermsGaussFit > 3){
                    D_A2_Limits[3][0] = 0.;
                    D_A2_Limits[3][1] = 2. * D_A1_Guess[3];
                    if (fiberTraceFunctionFindingControl->nTermsGaussFit > 4){
                      D_A2_Limits[4][0] = D_A1_Guess[4] / 10.;
                      D_A2_Limits[4][1] = D_A1_Guess[4] * 10.;
                    }
                  }
                  #if defined(__DEBUG_FINDANDTRACE__)
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": 1. starting MPFitGaussLim: D_A1_Guess = " << D_A1_Guess << ", I_A2_Limited = " << I_A2_Limited << ", D_A2_Limits = " << D_A2_Limits << endl;
                  #endif
                  if (!MPFitGaussLim(D_A1_X,
                                     D_A1_Y,
                                     D_A1_MeasureErrors,
                                     D_A1_Guess,
                                     I_A2_Limited,
                                     D_A2_Limits,
                                     0,
                                     false,
                                     D_A1_GaussFit_Coeffs,
                                     D_A1_GaussFit_ECoeffs)){
                    #if defined(__DEBUG_FINDANDTRACE__)
                      cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: WARNING: GaussFit FAILED -> abandoning aperture" << endl;
                    #endif

                    /// Set start index for next run
                    I_StartIndex = I_FirstWideSignalEnd+1;

                    ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                  }
                  else{
                    #if defined(__DEBUG_FINDANDTRACE__)
                      cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
                      if (D_A1_GaussFit_Coeffs[0] > fiberTraceFunctionFindingControl->saturationLevel){
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: WARNING: Signal appears to be saturated" << endl;
                      }
                      if (D_A1_GaussFit_Coeffs[1] < I_FirstWideSignalStart + 0.25*I_Length ||
                          D_A1_GaussFit_Coeffs[1] > I_FirstWideSignalStart + 0.75*I_Length) {
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: Warning: Center of Gaussian far away from middle of signal" << endl;
                      }
                    #endif
                    if (D_A1_GaussFit_Coeffs[1] < I_FirstWideSignalStart ||
                        D_A1_GaussFit_Coeffs[1] > I_FirstWideSignalEnd) {
                      #if defined(__DEBUG_FINDANDTRACE__)
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                      #endif
                      /// Set signal to zero
                      ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;

                      /// Set start index for next run
                      I_StartIndex = I_FirstWideSignalEnd+1;
                    }
                    else{
                      if ((D_A1_GaussFit_Coeffs[2] < fiberTraceFunctionFindingControl->apertureFWHM / 4.) || (D_A1_GaussFit_Coeffs[2] > fiberTraceFunctionFindingControl->apertureFWHM)){
                        #if defined(__DEBUG_FINDANDTRACE__)
                          cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: WARNING: FWHM = " << D_A1_GaussFit_Coeffs[2] << " outside range -> abandoning aperture" << endl;
                        #endif
                        /// Set signal to zero
                        ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                        #if defined(__DEBUG_FINDANDTRACE__)
                          cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: B_ApertureFound = " << B_ApertureFound << ": 1. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                          cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: 1. ccdArray(i_Row = " << i_Row << ", Range(I_FirstWideSignalStart = " << I_FirstWideSignalStart << ", I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << ")) set to " << ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart, I_FirstWideSignalEnd)] << endl;
                        #endif
                        /// Set start index for next run
                        I_StartIndex = I_FirstWideSignalEnd+1;
                      }
                      else{
                        D_A1_ApertureCenter[ndarray::view()] = 0.;
                        D_A1_EApertureCenter[ndarray::view()] = 0.;
                        B_ApertureFound = true;
                        //I_LastRowWhereApertureWasFound = i_Row;
                        D_A1_ApertureCenter[i_Row] = D_A1_GaussFit_Coeffs[1];
                        D_A1_EApertureCenter[i_Row] = D_A1_GaussFit_ECoeffs[1];
                        #if defined(__DEBUG_FINDANDTRACE__)
                          cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": Aperture found at " << D_A1_ApertureCenter[i_Row] << endl;
                        #endif
                        ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                      }
                    }
                  }
                }
              }
            }
          }
        }

        if (B_ApertureFound){
          /// Trace Aperture
          xySorted.resize(0);
          apertureLength = 1;
          I_Length = 1;
          I_ApertureLost = 0;
          #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": Starting to trace aperture " << endl;
          #endif
          D_A1_GaussFit_Coeffs_Bak[ndarray::view()].deep() = D_A1_GaussFit_Coeffs;
          while(B_ApertureFound && (I_ApertureLost < fiberTraceFunctionFindingControl->nLost) && (i_Row < ccdArray.getShape()[0]-1) && I_Length < fiberTraceFunctionFindingControl->maxLength){
            i_Row++;
            apertureLength++;
            I_Length++;
            if (fiberTraceFunctionFindingControl->nTermsGaussFit == 0){/// look for maximum only
              B_ApertureFound = true;
              int maxPos = 0;
              ImageT tMax = 0.;
              for (int i = I_FirstWideSignalStart; i <= I_FirstWideSignalEnd; ++i){
                if (ccdArray[i_Row][i] > tMax){
                  tMax = ccdArray[i_Row][i];
                  maxPos = i;
                }
              }
              if (tMax < fiberTraceFunctionFindingControl->signalThreshold){
                I_ApertureLost++;
              }
              else{
                D_A1_ApertureCenter[i_Row] = maxPos;
                D_A1_EApertureCenter[i_Row] = 0.5;/// Half a pixel uncertainty
                #if defined(__DEBUG_FINDANDTRACE__)
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": Aperture found at " << D_A1_ApertureCenter[i_Row] << endl;
                #endif
                if (D_A1_ApertureCenter[i_Row] < D_A1_ApertureCenter[i_Row-1]){
                  I_FirstWideSignalStart--;
                  I_FirstWideSignalEnd--;
                }
                if (D_A1_ApertureCenter[i_Row] > D_A1_ApertureCenter[i_Row-1]){
                  I_FirstWideSignalStart++;
                  I_FirstWideSignalEnd++;
                }
                ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
              }
            }
            else{
              I_FirstWideSignalStart = int(D_A1_GaussFit_Coeffs_Bak[1] - 1.6 * D_A1_GaussFit_Coeffs_Bak[2]);
              I_FirstWideSignalEnd = int(D_A1_GaussFit_Coeffs_Bak[1] + 1.6 * D_A1_GaussFit_Coeffs_Bak[2]) + 1;
              if (I_FirstWideSignalStart < 0. || I_FirstWideSignalEnd >= ccdArray.getShape()[1]){
                #if defined(__DEBUG_FINDANDTRACE__)
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": start or end of aperture outside CCD -> Aperture lost" << endl;
                #endif
                /// Set signal to zero
                if (I_FirstWideSignalStart < 0)
                  I_FirstWideSignalStart = 0;
                if (I_FirstWideSignalEnd >= ccdArray.getShape()[1])
                  I_FirstWideSignalEnd = ccdArray.getShape()[1] - 1;
                ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                I_ApertureLost++;
              }
              else{
                I_Length = I_FirstWideSignalEnd - I_FirstWideSignalStart + 1;

                if (I_Length <= fiberTraceFunctionFindingControl->nTermsGaussFit){
                  #if defined(__DEBUG_FINDANDTRACE__)
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": Warning: Width of Aperture <= " << fiberTraceFunctionFindingControl->nTermsGaussFit << " -> Lost Aperture" << endl;
                  #endif
                  /// Set signal to zero
                  ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                  I_ApertureLost++;
                }
                else{
                  ndarray::Array<float, 1, 1> D_A1_X = copy(ndarray::Array<float, 1, 1>(D_A1_IndexCol[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd+1)]));
                  ndarray::Array<float, 1, 1> D_A1_Y = ndarray::allocate(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
                  D_A1_Y.deep() = ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart, I_FirstWideSignalEnd+1)];
                  ndarray::Array<VarianceT, 1, 1> T_A1_MeasureErrors = copy(ccdVarianceArray[ndarray::view(i_Row)(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)]);
                  ndarray::Array<float, 1, 1> D_A1_MeasureErrors = ndarray::allocate(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
                  for (int ooo = 0; ooo < I_FirstWideSignalEnd - I_FirstWideSignalStart + 1; ++ooo){
                    if (T_A1_MeasureErrors[ooo] > 0)
                      D_A1_MeasureErrors[ooo] = ImageT(sqrt(T_A1_MeasureErrors[ooo]));
                    else
                      D_A1_MeasureErrors[ooo] = 1;
                  }
                  int iSum = 0;
                  for (auto it = D_A1_Y.begin(); it != D_A1_Y.end(); ++it){
                    if (*it < 0.000001)
                      *it = 1.;
                    if (*it >= fiberTraceFunctionFindingControl->signalThreshold)
                      ++iSum;
                  }
                  if (iSum < I_MinWidth){
                    /// Set signal to zero
                    ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                    I_ApertureLost++;
                    #if defined(__DEBUG_FINDANDTRACE__)
                      cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": Signal not wide enough => Aperture lost" << endl;
                    #endif
                  }
                  else{
                    #if defined(__DEBUG_FINDANDTRACE__)
                      cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": 2. D_A1_Y set to " << D_A1_Y << endl;
                    #endif
                    for (auto it = D_A1_MeasureErrors.begin(); it != D_A1_MeasureErrors.end(); ++it){
                      if (*it > 0)
                        *it = sqrt(*it);
                      else
                        *it = 1;
                    }
                    D_A1_Guess.deep() = D_A1_GaussFit_Coeffs_Bak;

                    D_A1_GaussFit_Coeffs[ndarray::view()] = 0.;
                    ndarray::Array<float, 1, 1> D_A1_GaussFit_ECoeffs = ndarray::allocate(D_A1_GaussFit_Coeffs.size());
                    D_A1_GaussFit_ECoeffs[ndarray::view()] = 0.;

                    #if defined(__DEBUG_FINDANDTRACE__)
                      cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: D_A1_X = " << D_A1_X << ", D_A1_Y = " << D_A1_Y << endl;
                    #endif

                    ndarray::Array<int, 2, 1> I_A2_Limited = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit, 2);
                    I_A2_Limited[ndarray::view()] = 1;
                    ndarray::Array<float, 2, 1> D_A2_Limits = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit, 2);
                    D_A2_Limits[0][0] = 0.;/// Peak lower limit
                    D_A2_Limits[0][1] = 2. * D_A1_Guess[0];/// Peak upper limit
                    D_A2_Limits[1][0] = I_FirstWideSignalStart;/// Centroid lower limit
                    D_A2_Limits[1][1] = I_FirstWideSignalEnd;/// Centroid upper limit
                    D_A2_Limits[2][0] = 0.25*fiberTraceFunctionFindingControl->apertureFWHM;/// Sigma lower limit
                    D_A2_Limits[2][1] = fiberTraceFunctionFindingControl->apertureFWHM;/// Sigma upper limit
                    if (gaussFitVariances.size() > 15){
                      double sum = std::accumulate(gaussFitMean.end()-10, gaussFitMean.end(), 0.0);
                      double mean = sum/10.; // we're using the last 10 points; see previous line
                      #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
                        for (int iMean = 0; iMean < gaussFitMean.size(); ++iMean){
                          cout << "gaussFitMean[" << iMean << "] = " << gaussFitMean[iMean] << endl;
                          cout << "gaussFitVariances[" << iMean << ") = " << gaussFitVariances[iMean] << endl;
                        }
                        cout << "sum = " << sum << ", mean = " << mean << endl;
                      #endif
                      double sq_sum = std::inner_product(gaussFitMean.end()-10, gaussFitMean.end(), gaussFitMean.end()-10, 0.0);
                      double stdev = std::sqrt(sq_sum / 10 - mean * mean);
                      #if defined(__DEBUG_FINDANDTRACE__)
                        cout << "GaussFitMean: sq_sum = " << sq_sum << ", stdev = " << stdev << endl;
                      #endif
                      D_A1_Guess[1] = mean;
                      D_A2_Limits[1][0] = mean - (3. * stdev) - 0.1;
                      D_A2_Limits[1][1] = mean + (3. * stdev) + 0.1;
                      #if defined(__DEBUG_FINDANDTRACE__)
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: D_A1_Guess[1] = " << D_A1_Guess[1] << ", Limits = " << D_A2_Limits[ndarray::view(1)()] << endl;
                      #endif
                      for (int iMean = 0; iMean < gaussFitMean.size(); ++iMean)
                      sum = std::accumulate(gaussFitVariances.end()-10, gaussFitVariances.end(), 0.0);
                      mean = sum / 10.;
                      #if defined(__DEBUG_FINDANDTRACE__)
                        cout << "GaussFitVariance: sum = " << sum << ", mean = " << mean << endl;
                      #endif
                      sq_sum = std::inner_product(gaussFitVariances.end()-10, gaussFitVariances.end(), gaussFitVariances.end()-10, 0.0);
                      stdev = std::sqrt(sq_sum / 10 - mean * mean);
                      #if defined(__DEBUG_FINDANDTRACE__)
                        cout << "sq_sum = " << sq_sum << ", stdev = " << stdev << endl;
                      #endif
                      D_A1_Guess[2] = mean;
                      D_A2_Limits[2][0] = mean - (3. * stdev);
                      D_A2_Limits[2][1] = mean + (3. * stdev);
                      #if defined(__DEBUG_FINDANDTRACE__)
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: D_A1_Guess[2] = " << D_A1_Guess[2] << ", Limits = " << D_A2_Limits[ndarray::view(2)()] << endl;
                      #endif
                    }
                    if (fiberTraceFunctionFindingControl->nTermsGaussFit > 3){
                      D_A2_Limits[3][0] = 0.;
                      D_A2_Limits[3][1] = 2. * D_A1_Guess[3];
                      if (fiberTraceFunctionFindingControl->nTermsGaussFit > 4){
                        D_A2_Limits[4][0] = D_A1_Guess[4] / 10.;
                        D_A2_Limits[4][1] = D_A1_Guess[4] * 10.;
                      }
                    }
                    #if defined(__DEBUG_FINDANDTRACE__)
                      cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: 2. starting MPFitGaussLim: D_A2_Limits = " << D_A2_Limits << endl;
                    #endif
                    ndarray::Array<float, 2, 1> D_A2_XY = ndarray::allocate(D_A1_X.getShape()[0], 2);
                    D_A2_XY[ndarray::view()(0)] = D_A1_X;
                    D_A2_XY[ndarray::view()(1)] = D_A1_Y;
                    D_A1_MeasureErrors.deep() = 1.;
                    if ((D_A2_Limits[1][0] > max(D_A1_X)) || (D_A2_Limits[1][1] < min(D_A1_X))){
                      string message("pfs::drp::stella::math::findCenterPositionsOneTrace: ERROR: (D_A2_Limits[1][0](=");
                      message += to_string(D_A2_Limits[1][0]) + ") > max(D_A1_X)(=" + to_string(max(D_A1_X)) + ")) || (D_A2_Limits[1][1](=";
                      message += to_string(D_A2_Limits[1][1]) + ") < min(D_A1_X)(=" + to_string(min(D_A1_X)) + "))";
                      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
                    }
                    if (!MPFitGaussLim(D_A1_X,
                                       D_A1_Y,
                                       D_A1_MeasureErrors,
                                       D_A1_Guess,
                                       I_A2_Limited,
                                       D_A2_Limits,
                                       0,
                                       false,
                                       D_A1_GaussFit_Coeffs,
                                       D_A1_GaussFit_ECoeffs)){
                      #if defined(__DEBUG_FINDANDTRACE__)
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": Warning: GaussFit FAILED" << endl;
                      #endif
                      /// Set signal to zero
                      ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart + 1, I_FirstWideSignalEnd)] = 0.;

                      I_ApertureLost++;
                    }
                    else{
                      gaussFitMean.push_back(D_A1_GaussFit_Coeffs[1]);
                      gaussFitVariances.push_back(D_A1_GaussFit_Coeffs[2]);

                      #if defined(__DEBUG_FINDANDTRACE__)
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
                        if (D_A1_GaussFit_Coeffs[0] < fiberTraceFunctionFindingControl->saturationLevel/5.){
                          cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": WARNING: Signal less than 20% of saturation level" << endl;
                        }
                        if (D_A1_GaussFit_Coeffs[0] > fiberTraceFunctionFindingControl->saturationLevel){
                          cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": WARNING: Signal appears to be saturated" << endl;
                        }
                      #endif
                      if (D_A1_GaussFit_Coeffs[0] < fiberTraceFunctionFindingControl->signalThreshold){
                          #if defined(__DEBUG_FINDANDTRACE__)
                            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": WARNING: peak = " << D_A1_GaussFit_Coeffs[1] << " lower than signalThreshold -> abandoning aperture" << endl;
                          #endif
                          /// Set signal to zero
                          ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart + 1, I_FirstWideSignalEnd)] = 0.;
                          #if defined(__DEBUG_FINDANDTRACE__)
                            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": 2. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                          #endif
                          I_ApertureLost++;
                      }
                      else{
                        if ((D_A1_GaussFit_Coeffs[1] < D_A1_GaussFit_Coeffs_Bak[1] - 1.) || (D_A1_GaussFit_Coeffs[1] > D_A1_GaussFit_Coeffs_Bak[1] + 1.)){
                          #if defined(__DEBUG_FINDANDTRACE__)
                            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                          #endif
                          /// Set signal to zero
                          ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;

                          I_ApertureLost++;
                        }
                        else{
                          if ((D_A1_GaussFit_Coeffs[2] < fiberTraceFunctionFindingControl->apertureFWHM / 4.) || (D_A1_GaussFit_Coeffs[2] > fiberTraceFunctionFindingControl->apertureFWHM)){
                            #if defined(__DEBUG_FINDANDTRACE__)
                              cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": WARNING: FWHM = " << D_A1_GaussFit_Coeffs[2] << " outside range -> abandoning aperture" << endl;
                            #endif
                            /// Set signal to zero
                            ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                            #if defined(__DEBUG_FINDANDTRACE__)
                              cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": 2. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                            #endif
                            I_ApertureLost++;
                          }
                          else{
                            I_ApertureLost = 0;
                            B_ApertureFound = true;
                            D_A1_ApertureCenter[i_Row] = D_A1_GaussFit_Coeffs[1];
                            D_A1_EApertureCenter[i_Row] = D_A1_GaussFit_ECoeffs[1];
                            #if defined(__DEBUG_FINDANDTRACE__)
                              cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": Aperture found at " << D_A1_ApertureCenter[i_Row] << endl;
                            #endif
                            D_A1_GaussFit_Coeffs_Bak[ndarray::view()] = D_A1_GaussFit_Coeffs[ndarray::view()];
                            xCorMinPos = 0.;
                            int ind = 0;
                            ndarray::Array<float, 2, 1> xyRelativeToCenter = ndarray::allocate(D_A1_X.getShape()[0] + 2, 2);
                            xyRelativeToCenter[0][0] = D_A1_X[0] - D_A1_GaussFit_Coeffs[1] - 1.;
                            xyRelativeToCenter[0][1] = 0.;
                            xyRelativeToCenter[xyRelativeToCenter.getShape()[0]-1][0] = D_A1_X[D_A1_X.getShape()[0]-1] - D_A1_GaussFit_Coeffs[1] + 1.;
                            xyRelativeToCenter[xyRelativeToCenter.getShape()[0]-1][1] = 0.;
                            for (int iX = 0; iX < D_A1_X.getShape()[0]; ++iX){
                              xyRelativeToCenter[iX+1][0] = D_A1_X[iX] - D_A1_GaussFit_Coeffs[1];
                              xyRelativeToCenter[iX+1][1] = D_A1_Y[iX];
                            }
                            indGaussArr[ndarray::view()(0)] = xyRelativeToCenter[0][0];
                            ind = 0;
                            float fac = (xyRelativeToCenter[xyRelativeToCenter.getShape()[0]-1][0] - xyRelativeToCenter[0][0]) / nInd;
                            for (auto itRow = indGaussArr.begin(); itRow != indGaussArr.end(); ++itRow, ++ind){
                              auto itCol = itRow->begin();
                              *itCol = *itCol + (ind * fac);
                              *(itCol + 1) = D_A1_GaussFit_Coeffs[0] * exp(0. - ((*itCol) * (*itCol)) / (2. * D_A1_Guess[2] * D_A1_Guess[2]));
                            }
                            if (gaussFitVariances.size() > 20){
                              ndarray::Array<float, 2, 1> xysRelativeToCenter = ndarray::allocate(xySorted.size(), 2);
                              ind = 0;
                              auto itSorted = xySorted.begin();
                              for (auto itRow = xysRelativeToCenter.begin(); itRow != xysRelativeToCenter.end(); ++itRow, ++ind, ++itSorted){
                                auto itCol = itRow->begin();
                                *itCol = itSorted->x;
                                *(itCol+1) = itSorted->y;
                              }
                              #if defined(__DEBUG_FINDANDTRACE__)
                                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": xCorMinPos = " << xCorMinPos << endl;
                              #endif
                            }
                            if (gaussFitVariances.size() > 10){
                              for (int iX = 0; iX < xyRelativeToCenter.getShape()[0]; ++iX){
                                dataXY<float> xyStruct;
                                xyStruct.x = xyRelativeToCenter[iX][0] + xCorMinPos;
                                xyStruct.y = xyRelativeToCenter[iX][1];
                                pfs::drp::stella::math::insertSorted(xySorted, xyStruct);
                              }
                              D_A1_ApertureCenter[i_Row] = D_A1_ApertureCenter[i_Row] + xCorMinPos;
                              #if defined(__DEBUG_FINDANDTRACE__)
                                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": Aperture position corrected to " << D_A1_ApertureCenter[i_Row] << endl;
                              #endif
                            }
                          }
                        }/// end else if ((D_A1_GaussFit_Coeffs(1) >= D_A1_Guess(1) - 1.) && (D_A1_GaussFit_Coeffs(1) <= D_A1_Guess(1) + 1.))
                      }/// end else if (D_A1_GaussFit_Coeffs(0) >= signalThreshold
                    }/// end else if (GaussFit(D_A1_X, D_A1_Y, D_A1_GaussFit_Coeffs, S_A1_KeyWords_GaussFit, PP_Args_GaussFit))
                    ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart + 1, I_FirstWideSignalEnd)] = 0.;
                  }/// end else if (sum(I_A1_Signal) >= I_MinWidth){
                }/// end if (I_Length > 3)
              }/// end else if (I_ApertureStart >= 0. && I_ApertureEnd < ccdArray.getShape()[1])
            }/// end else if GaussFit
            if ((I_ApertureLost == fiberTraceFunctionFindingControl->nLost) && (apertureLength < fiberTraceFunctionFindingControl->minLength)){
              i_Row = I_RowBak;
              #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row set to " << i_Row << endl;
              #endif
            }
          }///end while(B_ApertureFound && (I_ApertureLost < 3) && i_Row < ccdArray.getShape()[0] - 2))

          /// Fit Polynomial to traced aperture positions
          #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": D_A1_ApertureCenter = " << D_A1_ApertureCenter << endl;
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": I_A1_ApertureCenterIndex.getShape() = " << I_A1_ApertureCenterIndex.getShape() << endl;
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": D_A1_ApertureCenter.getShape() = " << D_A1_ApertureCenter.getShape() << endl;
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": D_A1_EApertureCenter.getShape() = " << D_A1_EApertureCenter.getShape() << endl;
          #endif

          auto itInd = I_A1_ApertureCenterIndex.begin();
          auto itCen = D_A1_ApertureCenter.begin();
          auto itECen = D_A1_EApertureCenter.begin();
          I_ApertureLength = 0;
          result.apertureCenterIndex.resize(0);
          result.apertureCenterPos.resize(0);
          result.eApertureCenterPos.resize(0);
          for (int iInd = 0; iInd < ccdArray.getShape()[0]; ++iInd){
            if (*(itCen + iInd) > 0.){
              (*(itInd + iInd)) = 1;
              ++I_ApertureLength;
              result.apertureCenterIndex.push_back(iInd);
              result.apertureCenterPos.push_back((*(itCen + iInd)));
              result.eApertureCenterPos.push_back((*(itECen + iInd)));
              #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": result.apertureCenterIndex[" << result.apertureCenterIndex.size()-1 << "] set to " << result.apertureCenterIndex[result.apertureCenterIndex.size()-1] << endl;
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": result.apertureCenterPos[" << result.apertureCenterPos.size()-1 << "] set to " << result.apertureCenterPos[result.apertureCenterPos.size()-1] << endl;
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": result.eApertureCenterPos[" << result.eApertureCenterPos.size()-1 << "] set to " << result.eApertureCenterPos[result.eApertureCenterPos.size()-1] << endl;
              #endif
            }
          }
          if (I_ApertureLength > fiberTraceFunctionFindingControl->minLength){
            #if defined(__DEBUG_FINDANDTRACE__)
              cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": result.apertureCenterIndex.size() = " << result.apertureCenterIndex.size() << endl;
            #endif
            return result;
          }
        }
      }
      #if defined(__DEBUG_FINDANDTRACE__)
        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: result.apertureCenterIndex.size() = " << result.apertureCenterIndex.size() << endl;
      #endif
      return result;
    }


    /** *******************************************************************************************************/

    /// Calculate the x-centers of the fiber trace
    ndarray::Array<float, 1, 1> calculateXCenters(PTR(const FiberTraceFunction) const& fiberTraceFunctionIn,
                                                  size_t const& ccdHeightIn,
                                                  size_t const& ccdWidthIn){
      ndarray::Array<float, 1, 1> xRowIndex = ndarray::allocate(fiberTraceFunctionIn->yHigh - fiberTraceFunctionIn->yLow + 1);
      float xRowInd = fiberTraceFunctionIn->yCenter + fiberTraceFunctionIn->yLow;
      for (auto i = xRowIndex.begin(); i != xRowIndex.end(); ++i){
        *i = xRowInd;
        ++xRowInd;
      }
      return calculateXCenters(fiberTraceFunctionIn,
                               xRowIndex,
                               ccdHeightIn,
                               ccdWidthIn);
    }

    ndarray::Array<float, 1, 1> calculateXCenters(PTR(const ::pfs::drp::stella::FiberTraceFunction) const& fiberTraceFunctionIn,
                                                  ndarray::Array<float, 1, 1> const& yIn,
                                                  size_t const& ccdHeightIn,
                                                  size_t const& ccdWidthIn){

      ndarray::Array<float, 1, 1> xCenters;

      #ifdef __DEBUG_XCENTERS__
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunctionIn->fiberTraceFunctionControl->interpolation = " << fiberTraceFunctionIn->fiberTraceFunctionControl->interpolation << endl;
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunctionIn->fiberTraceFunctionControl->order = " << fiberTraceFunctionIn->fiberTraceFunctionControl->order << endl;
      #endif
      ndarray::Array<float, 1, 1> range = ndarray::allocate(2);
      range[0] = fiberTraceFunctionIn->yCenter + fiberTraceFunctionIn->yLow;
      range[1] = fiberTraceFunctionIn->yCenter + fiberTraceFunctionIn->yHigh;
      #ifdef __DEBUG_XCENTERS__
        cout << "pfs::drp::stella::calculateXCenters: range = " << range << endl;
      #endif
      if (fiberTraceFunctionIn->fiberTraceFunctionControl->interpolation.compare("CHEBYSHEV") == 0)
      {
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::math::calculateXCenters: Calculating Chebyshev Polynomial" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Function = Chebyshev" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Coeffs = " << fiberTraceFunctionIn->coefficients << endl;
        #endif
        ndarray::Array<float, 1, 1> yNew = pfs::drp::stella::math::convertRangeToUnity(yIn, range);
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::calculateXCenters: CHEBYSHEV: yNew = " << yNew << endl;
          cout << "pfs::drp::stella::calculateXCenters: CHEBYSHEV: fiberTraceFunctionIn->coefficients = " << fiberTraceFunctionIn->coefficients << endl;
        #endif
        xCenters = pfs::drp::stella::math::chebyshev(yNew, fiberTraceFunctionIn->coefficients);
      }
      else /// Polynomial
      {
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::math::calculateXCenters: Calculating Polynomial" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Function = Polynomial" << endl;
          cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunctionIn->coefficients = " << fiberTraceFunctionIn->coefficients << endl;
        #endif
        xCenters = math::Poly(yIn,
                              fiberTraceFunctionIn->coefficients,
                              range[0],
                              range[1]);
      }
      #ifdef __DEBUG_XCENTERS__
        cout << "calculateXCenters: xCenters = " << xCenters.getShape()[0] << ": " << xCenters << endl;
      #endif
      return xCenters;
    }

    template PTR(FiberTraceSet<float, lsst::afw::image::MaskPixel, float>)
    findAndTraceApertures(PTR(const afwImage::MaskedImage<float, lsst::afw::image::MaskPixel, float>) const&,
                          DetectorMap const&,                          
                          PTR(const FiberTraceFunctionFindingControl) const&,
                          PTR(FiberTraceProfileFittingControl) const&);

    template FindCenterPositionsOneTraceResult
    findCenterPositionsOneTrace( PTR(afwImage::Image<float>) &,
                                 PTR(afwImage::Image<float>) &,
                                 PTR(const FiberTraceFunctionFindingControl) const&);

  }

template class FiberTrace<float, lsst::afw::image::MaskPixel, float>;
template class FiberTraceSet<float, lsst::afw::image::MaskPixel, float>;

}}}
