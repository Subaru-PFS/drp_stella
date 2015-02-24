#include "pfs/drp/stella/FiberTraces.h"

namespace pfsDRPStella = pfs::drp::stella;

  /** @brief Construct an FiberTrace with a blank MaskedImage of specified size (default 0x0)
   */
  template<typename ImageT, typename MaskT, typename VarianceT>
  pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    size_t width,                 ///< number of columns
    size_t height,                ///< number of rows
    size_t iTrace
  ) :
  _trace(new afwImage::MaskedImage<ImageT, MaskT, VarianceT>(width, height)),
  _profile(new afwImage::Image<float>(width, height)),
  _xCenters(pfsDRPStella::utils::get1DndArray(float(height))),
  _iTrace(iTrace),
  _isTraceSet(false),
  _isProfileSet(false),
  _isFiberTraceProfileFittingControlSet(false),
  _fiberTraceFunction(new pfsDRPStella::FiberTraceFunction),
  _fiberTraceProfileFittingControl(new FiberTraceProfileFittingControl),
  _overSampledProfileFitXPerSwath(),
  _overSampledProfileFitYPerSwath(),
  _profileFittingInputXPerSwath(),
  _profileFittingInputYPerSwath(),
  _profileFittingInputXMeanPerSwath(),
  _profileFittingInputYMeanPerSwath()
  {
    
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(PTR(const MaskedImageT) const & maskedImage, ///< desired image width/height
                                                                 PTR(const pfsDRPStella::FiberTraceFunction) const& fiberTraceFunction,
                                                                 ndarray::Array<float const, 1, 1> const& xCenters,
                                                                 size_t iTrace) :
  _trace(new afwImage::MaskedImage<ImageT, MaskT, VarianceT>(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1, int(fiberTraceFunction->fiberTraceFunctionControl.xHigh - fiberTraceFunction->fiberTraceFunctionControl.xLow + 1))),
  _profile(new afwImage::Image<float>(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1, int(fiberTraceFunction->fiberTraceFunctionControl.xHigh - fiberTraceFunction->fiberTraceFunctionControl.xLow + 1))),
  _xCenters(xCenters),//new std::vector<const float>(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1)),
  _iTrace(iTrace),
  _isTraceSet(false),
  _isProfileSet(false),
  _isFiberTraceProfileFittingControlSet(false),
  _fiberTraceFunction(fiberTraceFunction),
  _fiberTraceProfileFittingControl(new FiberTraceProfileFittingControl),
  _overSampledProfileFitXPerSwath(),
  _overSampledProfileFitYPerSwath(),
  _profileFittingInputXPerSwath(),
  _profileFittingInputYPerSwath(),
  _profileFittingInputXMeanPerSwath(),
  _profileFittingInputYMeanPerSwath()
  {
    createTrace(maskedImage);
  }
  
  template<typename ImageT, typename MaskT, typename VarianceT>
  pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT> & fiberTrace, bool const deep) :
  _trace(fiberTrace.getTrace()),
  _profile(fiberTrace.getProfile()),
  _xCenters(fiberTrace.getXCenters()),
  _iTrace(fiberTrace.getITrace()),
  _isTraceSet(fiberTrace.isTraceSet()),
  _isProfileSet(fiberTrace.isProfileSet()),
  _isFiberTraceProfileFittingControlSet(fiberTrace.isFiberTraceProfileFittingControlSet()),
  _fiberTraceFunction(fiberTrace.getFiberTraceFunction()),
  _fiberTraceProfileFittingControl(fiberTrace.getFiberTraceProfileFittingControl()),
  _overSampledProfileFitXPerSwath(),
  _overSampledProfileFitYPerSwath(),
  _profileFittingInputXPerSwath(),
  _profileFittingInputYPerSwath(),
  _profileFittingInputXMeanPerSwath(),
  _profileFittingInputYMeanPerSwath()
  {
    if (deep){
      PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) ptr(new afwImage::MaskedImage<ImageT, MaskT, VarianceT>(*(fiberTrace.getTrace()), true));
      _trace.reset();
      _trace = ptr;
      PTR(afwImage::Image<float>) prof(new afwImage::Image<float>(*(fiberTrace.getProfile()), true));
      _profile.reset();
      _profile = prof;
    }
  }
  
  /** **************************************************************/
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setFiberTraceProfileFittingControl(PTR(FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl){

    /// Check for valid values in fiberTraceFunctionControl
    bool isTelluricValid = false;
    #ifdef __DEBUG_SETFIBERTRACEEXTRACTIONCONTROL__
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->profileInterpolation = <" << fiberTraceProfileFittingControl->profileInterpolation << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->ccdReadOutNoise = <" << fiberTraceProfileFittingControl->ccdReadOutNoise << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->swathWidth = <" << fiberTraceProfileFittingControl->swathWidth << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->telluric = <" << fiberTraceProfileFittingControl->telluric << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->overSample = <" << fiberTraceProfileFittingControl->overSample << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->maxIterSF = <" << fiberTraceProfileFittingControl->maxIterSF << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->maxIterSig = <" << fiberTraceProfileFittingControl->maxIterSig << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->maxIterSky = <" << fiberTraceProfileFittingControl->maxIterSky << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->lambdaSF = <" << fiberTraceProfileFittingControl->lambdaSF << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->lambdaSP = <" << fiberTraceProfileFittingControl->lambdaSP << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->wingSmoothFactor = <" << fiberTraceProfileFittingControl->wingSmoothFactor << ">" << endl;
      //cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->xCorProf = <" << fiberTraceProfileFittingControl->xCorProf << ">" << endl;
    #endif

    if (!fiberTraceProfileFittingControl->isClassInvariant()){
      string message("FiberTrace::setFiberTraceProfileFittingControl: ERROR: fiberTraceProfileFittingControl is not ClassInvariant");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }

    /// test passed -> copy fiberTraceProfileFittingControl to _fiberTraceProfileFittingControl
    _fiberTraceProfileFittingControl.reset();
    _fiberTraceProfileFittingControl = fiberTraceProfileFittingControl;
    _isFiberTraceProfileFittingControlSet = true;

    return true;
  }

  /// Set the image pointer of this fiber trace to image
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setImage(const PTR(afwImage::Image<ImageT>) &image){

    /// Check input image size
    if (image->getWidth() != int(_trace->getWidth())){
      string message("FiberTrace.setImage: ERROR: image.getWidth(=");
      message += to_string(image->getWidth()) + string(") != _trace->getWidth(=") + to_string(_trace->getWidth()) + string(")");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    if (image->getHeight() != int(_trace->getHeight())){
      string message("FiberTrace.setImage: ERROR: image.getHeight(=");
      message += to_string(image->getHeight()) + string(") != _trace->getHeight(=") + to_string(_trace->getHeight()) + string(")");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }

    _trace->getImage() = image;

    return true;
  }

  /// Set the mask pointer of this fiber trace to mask
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setMask(const PTR(afwImage::Mask<MaskT>) &mask){

    /// Check input mask size
    if (mask->getWidth() != int(_trace->getWidth())){
      string message("FiberTrace.setMask: ERROR: mask.getWidth(=");
      message += to_string(mask->getWidth()) + string(") != _trace->getWidth()(=") + to_string(_trace->getWidth()) + string(")");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    if (mask->getHeight() != int(_trace->getHeight())){
      string message("FiberTrace.setMask: ERROR: mask.getHeight(=");
      message += to_string(mask->getHeight()) + string(") != _trace->getHeight()(=") + to_string(_trace->getHeight()) + string(")");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }

    _trace->getMask() = mask;

    return true;
  }

  /// Set the variance pointer of this fiber trace to variance
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setVariance(const PTR(afwImage::Image<VarianceT>) &variance){

    /// Check input variance size
    if (variance->getWidth() != int(_trace->getWidth())){
      string message("FiberTrace.setVariance: ERROR: variance.getWidth(=");
      message += to_string(variance->getWidth()) + string(") != _trace->getWidth(=") + to_string(_trace->getWidth()) + string(")");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    if (variance->getHeight() != int(_trace->getHeight())){
      string message("FiberTrace.setVariance: ERROR: variance.getHeight(=");
      message += to_string(variance->getHeight()) + string(") != _trace->getHeight(=") + to_string(_trace->getHeight()) + string(")");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }

    _trace->getVariance() = variance;

    return true;
  }

  /// Set the _trace of this fiber trace to trace
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setTrace(PTR(MaskedImageT) & trace){
    if (_isTraceSet && (trace->getHeight() != int(_trace->getHeight()))){
      string message("FiberTrace");
      message += to_string(_iTrace) + string("::setTrace: ERROR: trace->getHeight(=") + to_string(trace->getHeight()) + string(") != _trace->getHeight(=");
      message += to_string(_trace->getHeight()) + string(")");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    if (_isTraceSet && (trace->getWidth() != int(_trace->getWidth()))){
      string message ("FiberTrace");
      message += to_string(_iTrace) + string("::setTrace: ERROR: trace->getWidth(=") + to_string(trace->getWidth()) + string(") != _trace->getWidth(=");
      message += to_string(_trace->getWidth()) + string(")");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }

    _trace.reset();
    _trace = trace;

    _isTraceSet = true;
    return true;
  }

  /// Set the profile image of this fiber trace to profile
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setProfile(const PTR(afwImage::Image<float>) &profile){
    if (!_isTraceSet){
      string message("FiberTrace");
      message += to_string(_iTrace) + string("::setProfile: ERROR: _trace not set");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }

    /// Check input profile size
    if (profile->getWidth() != _trace->getWidth()){
      string message("FiberTrace.setProfile: ERROR: profile->getWidth(=");
      message += to_string(profile->getWidth()) + string(") != _trace->getWidth(=") + to_string(_trace->getWidth()) + string(")");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    if (profile->getHeight() != _trace->getHeight()){
      string message("FiberTrace.setProfile: ERROR: profile->getHeight(=");
      message += to_string(profile->getHeight()) + string(") != _trace->getHeight(=") + to_string(_trace->getHeight()) + string(")");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }

    _profile.reset();
    _profile = profile;

    _isProfileSet = true;
    return true;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::extractSum()
  {
    PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) spectrum(new pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>(_trace->getHeight(), _iTrace));
    ndarray::Array<ImageT, 1, 1> spec = ndarray::allocate(_trace->getHeight());
    ndarray::Array<MaskT, 1, 1> mask = ndarray::allocate(_trace->getHeight());
    ndarray::Array<VarianceT, 1, 1> var = ndarray::allocate(_trace->getHeight());
    auto specIt = spec.begin();
    auto varIt = var.begin();
    auto maskIt = mask.begin();
    for (int i = 0; i < _trace->getHeight(); ++i, ++specIt, ++varIt, ++maskIt){
      *specIt = sum(_trace->getImage()->getArray()[i]);
      *varIt = sum(_trace->getVariance()->getArray()[i]);
      *maskIt = sum(_trace->getMask()->getArray()[i]);
    }
    if (!spectrum->setSpectrum(spec)){
      string message("FiberTrace");
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setSpectrum(spec) returned FALSE";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    if (!spectrum->setVariance(var)){
      string message("FiberTrace");
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setVariance(var) returned FALSE";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    if (!spectrum->setMask(mask)){
      string message("FiberTrace");
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setMask(mask) returned FALSE";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    return spectrum;
  }
  
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::extractFromProfile()
  {
    if (!_isTraceSet){
      cout << "FiberTrace.extractFromProfile: ERROR: _trace is not set" << endl;
      throw LSST_EXCEPT(pexExcept::Exception, "FiberTrace.extractFromProfile: ERROR: _trace is not set");
    }
    if (!_isProfileSet){
      cout << "FiberTrace.extractFromProfile: ERROR: _profile is not set" << endl;
      throw LSST_EXCEPT(pexExcept::Exception, "FiberTrace.extractFromProfile: ERROR: _profile is not set");
    }
    if (_trace->getWidth() != _profile->getWidth()){
      std::string message("FiberTrace.extractFromProfile: ERROR: _trace->getWidth(=");
      message += std::to_string(_trace->getWidth());
      message += std::string(") != _profile.getWidth(=");
      message += std::to_string(_profile->getWidth());
      message += std::string(")");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (_trace->getHeight() != _profile->getHeight()){
      std::string message("FiberTrace.extractFromProfile: ERROR: _trace->getHeight(=");
      message += std::to_string(_trace->getHeight());
      message += std::string(") != _profile.getHeight(=");
      message += std::to_string(_profile->getHeight());
      message += std::string(")");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    ///TODO: change to sqrt(T_A2_VarianceArray)
    for (auto itRow = _trace->getVariance()->getArray().begin(); itRow != _trace->getVariance()->getArray().end(); ++itRow){
      for (auto itCol = itRow->begin(); itCol != itRow->end(); ++itCol){
        if (*itCol < 0.00000001)
          *itCol = 1.;
      }
    }

    ndarray::Array<unsigned short, 2, 2> US_A2_MaskArray = ndarray::allocate(_trace->getImage()->getArray().getShape());
    for (int i_row = 0; i_row < _trace->getHeight(); ++i_row){
      for (int i_col = 0; i_col < _trace->getWidth(); ++i_col){
        if (_trace->getMask()->getArray()[i_row][i_col] == 0)
          US_A2_MaskArray[i_row][i_col] = 1;
        else
          US_A2_MaskArray[i_row][i_col] = 0;
      }
    }
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "US_A2_MaskArray = " << US_A2_MaskArray << endl;
    #endif

    ndarray::Array<ImageT, 1, 1> D_A1_SP = ndarray::allocate(_trace->getHeight());
    D_A1_SP.deep() = 0.;
    ndarray::Array<ImageT, 1, 1> D_A1_Sky = ndarray::allocate(_trace->getHeight());
    D_A1_Sky.deep() = 0.;
    vector<string> S_A1_Args_Fit(3);
    vector<void *> P_Args_Fit(3);

    ndarray::Array<ImageT, 2, 2> D_A2_ErrArray = ndarray::allocate(_trace->getImage()->getArray().getShape());
    for (int i_row = 0; i_row < _trace->getHeight(); ++i_row){
      for (int i_col = 0; i_col < _trace->getWidth(); ++i_col){
        D_A2_ErrArray[i_row][i_col] = ImageT(sqrt(_trace->getVariance()->getArray()[i_row][i_col]));
      }
    }
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "D_A2_ErrArray = " << D_A2_ErrArray << endl;
    #endif
    S_A1_Args_Fit[0] = string("MEASURE_ERRORS_IN");
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "S_A1_Args_Fit[0] set to <" << S_A1_Args_Fit[0] << ">" << endl;
      cout << "D_A2_ErrArray.getShape() = " << D_A2_ErrArray.getShape() << endl;
    #endif
    PTR(ndarray::Array<ImageT, 2, 2>) P_D_A2_ErrArray(new ndarray::Array<ImageT, 2, 2>(D_A2_ErrArray));
    P_Args_Fit[0] = &P_D_A2_ErrArray;
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: *P_D_A2_ErrArray = " << *P_D_A2_ErrArray << endl;
    #endif

    S_A1_Args_Fit[1] = "MASK_INOUT";
    PTR(ndarray::Array<unsigned short, 2, 2>) P_US_A2_MaskArray(new ndarray::Array<unsigned short, 2, 2>(US_A2_MaskArray));
    P_Args_Fit[1] = &P_US_A2_MaskArray;
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "P_E_A2_MaskArray = " << *P_US_A2_MaskArray << endl;
    #endif

    S_A1_Args_Fit[2] = "SIGMA_OUT";
    ndarray::Array<ImageT, 2, 2> D_A2_Sigma_Fit = ndarray::allocate(_trace->getHeight(), 2);
    PTR(ndarray::Array<ImageT, 2, 2>) P_D_A2_Sigma_Fit(new ndarray::Array<ImageT, 2, 2>(D_A2_Sigma_Fit));
    P_Args_Fit[2] = &P_D_A2_Sigma_Fit;

    bool B_WithSky = false;
    if (_fiberTraceProfileFittingControl->telluric.compare(_fiberTraceProfileFittingControl->TELLURIC_NAMES[0]) != 0){
      D_A1_Sky.deep() = 1.;
      B_WithSky = true;
      cout << "extractFromProfile: Sky switched ON" << endl;
    }
    if (!math::LinFitBevingtonNdArray(_trace->getImage()->getArray(),      ///: in
                                      _profile->getArray(),             ///: in
                                      D_A1_SP,             ///: out
                                      D_A1_Sky,          ///: in/out
                                      B_WithSky,                   ///: with sky: in
                                      S_A1_Args_Fit,         ///: in
                                      P_Args_Fit)){          ///: in/out
      std::string message("FiberTrace");
      message += std::to_string(_iTrace);
      message += std::string("::extractFromProfile: 2. ERROR: LinFitBevington(...) returned FALSE");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    #ifdef __DEBUG_MkSLITFUNC_FILES__
      string S_MaskFinalOut = "Mask_Final" + S_SF_DebugFilesSuffix + ".fits";
      ::pfs::drp::stella::utils::WriteFits(&I_A2_MaskArray, S_MaskFinalOut);

      S_MaskFinalOut = "D_A2_CCD_Ap" + CS_SF_DebugFilesSuffix + ".fits";
      ::pfs::drp::stella::utils::WriteFits(&D_A2_CCDArray, S_MaskFinalOut);
    #endif
//    D_A1_SP.asEigen() = specEigen;
//    D_A1_Sky.asEigen() = skyEigen;
//    D_A2_Sigma_Fit.asEigen() = *P_D_A2_Sigma_Fit;

    PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) spectrum(new pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>(_trace->getHeight()));
    PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) background(new pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>(_trace->getHeight()));
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: D_A1_SP = " << D_A1_SP << endl;
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: D_A2_Sigma_Fit = " << D_A2_Sigma_Fit << endl;
      cout << "P_D_A2_Sigma_Fit = " << *P_D_A2_Sigma_Fit << endl;
    #endif
    for (int i = 0; i < _trace->getHeight(); i++) {
      spectrum->getSpectrum()[i] = ImageT(D_A1_SP[i]);
      spectrum->getVariance()[i] = VarianceT(pow(D_A2_Sigma_Fit[i][0], 2));
      background->getSpectrum()[i] = ImageT(D_A1_Sky[i]);
      background->getVariance()[i] = VarianceT(pow(D_A2_Sigma_Fit[i][1], 2));
    }
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: spectrum->getSpectrum() = " << spectrum->getSpectrum() << endl;
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: spectrum->getVariance() = " << spectrum->getVariance() << endl;
    #endif
    return spectrum;
  }

  /**************************************************************************
   * createTrace
   * ************************************************************************/
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::createTrace(const PTR(const MaskedImageT) &maskedImage){
    int oldTraceHeight = 0;
    if (_isTraceSet)
      oldTraceHeight = getHeight();

    if (_xCenters.getShape()[0] != (_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1)){
      string message("FiberTrace");
      message += to_string(_iTrace) + string("::createTrace: ERROR: _xCenters.getShape()[0]=") + to_string(_xCenters.getShape()[0]);
      message += string(" != (_fiberTraceFunction->yHigh(=") + to_string(_fiberTraceFunction->yHigh) + string(") - _fiberTraceFunction->yLow(=");
      message += to_string(_fiberTraceFunction->yLow) + string(") + 1)=") + to_string(_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1);
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }

    ndarray::Array<size_t, 2, 2> minCenMax = pfsDRPStella::math::calcMinCenMax(_xCenters,
                                                                               _fiberTraceFunction->fiberTraceFunctionControl.xHigh,
                                                                               _fiberTraceFunction->fiberTraceFunctionControl.xLow,
                                                                               1,
                                                                               1);
    #ifdef __DEBUG_CREATEFIBERTRACE__
      cout << "FiberTrace" << _iTrace << "::CreateFiberTrace: minCenMax = " << minCenMax << endl;
    #endif

    if ((_isTraceSet) && (_trace->getHeight() != (_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1))){
      string message("FiberTrace ");
      message += to_string(_iTrace) + string("::createTrace: ERROR: _trace.getHeight(=") + to_string(_trace->getHeight()) + string(") != (_fiberTraceFunction->yHigh(=");
      message += to_string(_fiberTraceFunction->yHigh) + string(") - _fiberTraceFunction->yLow(=") + to_string(_fiberTraceFunction->yLow) + string(") + 1) = ");
      message += to_string(_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1);
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    _trace.reset(new MaskedImageT(int(minCenMax[0][2] - minCenMax[0][0] + 1), _fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1));// minCenMax.rows());

    if (oldTraceHeight > 0){
      if (oldTraceHeight != getHeight()){
        string message("FiberTrace ");
        message += to_string(_iTrace) + string("::createTrace: ERROR: oldTraceHeight(=") + to_string(oldTraceHeight) + string(") != getHeight(=") + to_string(getHeight());
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    }

    ndarray::Array<ImageT, 2, 1> imageArray = maskedImage->getImage()->getArray();
    ndarray::Array<VarianceT, 2, 1> varianceArray = maskedImage->getVariance()->getArray();
    ndarray::Array<MaskT, 2, 1> maskArray = maskedImage->getMask()->getArray();
    ndarray::Array<ImageT, 2, 1> traceImageArray = _trace->getImage()->getArray();
    ndarray::Array<VarianceT, 2, 1> traceVarianceArray = _trace->getVariance()->getArray();
    ndarray::Array<MaskT, 2, 1> traceMaskArray = _trace->getMask()->getArray();
    typename ndarray::Array<ImageT, 2, 1>::Iterator yIterTrace = traceImageArray.begin();
    typename ndarray::Array<VarianceT, 2, 1>::Iterator yIterTraceVariance = traceVarianceArray.begin();
    typename ndarray::Array<MaskT, 2, 1>::Iterator yIterTraceMask = traceMaskArray.begin();
    int iy = 0;//_fiberTraceFunction->yCenter + _fiberTraceFunction->yLow;
    for (iy = 0; iy <= static_cast<int>(_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow); ++iy) {
      typename ndarray::Array<ImageT, 2, 1>::Iterator yIter = imageArray.begin() + _fiberTraceFunction->yCenter + _fiberTraceFunction->yLow + iy;
      typename ndarray::Array<VarianceT, 2, 1>::Iterator yIterV = varianceArray.begin() + _fiberTraceFunction->yCenter + _fiberTraceFunction->yLow + iy;
      typename ndarray::Array<MaskT, 2, 1>::Iterator yIterM = maskArray.begin() + _fiberTraceFunction->yCenter + _fiberTraceFunction->yLow + iy;
      typename ndarray::Array<ImageT, 2, 1>::Reference::Iterator ptrImageStart = yIter->begin() + minCenMax[iy][0];
      typename ndarray::Array<ImageT, 2, 1>::Reference::Iterator ptrImageEnd = yIter->begin() + minCenMax[iy][2] + 1;
      typename ndarray::Array<ImageT, 2, 1>::Reference::Iterator ptrTraceStart = yIterTrace->begin();
      std::copy(ptrImageStart, ptrImageEnd, ptrTraceStart);

      typename ndarray::Array<VarianceT, 2, 1>::Reference::Iterator ptrVarianceStart = yIterV->begin() + minCenMax[iy][0];
      typename ndarray::Array<VarianceT, 2, 1>::Reference::Iterator ptrVarianceEnd = yIterV->begin() + minCenMax[iy][2] + 1;
      typename ndarray::Array<VarianceT, 2, 1>::Reference::Iterator ptrTraceVarianceStart = yIterTraceVariance->begin();
      std::copy(ptrVarianceStart, ptrVarianceEnd, ptrTraceVarianceStart);

      typename ndarray::Array<MaskT, 2, 1>::Reference::Iterator ptrMaskStart = yIterM->begin() + minCenMax[iy][0];
      typename ndarray::Array<MaskT, 2, 1>::Reference::Iterator ptrMaskEnd = yIterM->begin() + minCenMax[iy][2] + 1;
      typename ndarray::Array<MaskT, 2, 1>::Reference::Iterator ptrTraceMaskStart = yIterTraceMask->begin();
      std::copy(ptrMaskStart, ptrMaskEnd, ptrTraceMaskStart);
      ++yIterTrace;
      ++yIterTraceVariance;
      ++yIterTraceMask;
      #ifdef __DEBUG_CREATETRACE__
        cout << "FiberTrace " << _iTrace << "::createTrace: iy = " << iy << endl;
      #endif
    }
    if (_trace->getHeight() != (_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1)){
      string message("FiberTrace ");
      message += to_string(_iTrace) + string("::createTrace: 2. ERROR: _trace.getHeight(=") + to_string(_trace->getHeight()) + string(") != (_fiberTraceFunction->yHigh(=");
      message += to_string(_fiberTraceFunction->yHigh) + string(") - _fiberTraceFunction->yLow(=") + to_string(_fiberTraceFunction->yLow) + string(") + 1) = ");
      message += to_string(_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1);
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (_xCenters.getShape()[0] != (_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1)){
      string message("FiberTrace");
      message += to_string(_iTrace) + string("::createTrace: 2. ERROR: xCenters.getShape()[0]=") + to_string(_xCenters.getShape()[0]);
      message += string(") != (_fiberTraceFunction->yHigh(=") + to_string(_fiberTraceFunction->yHigh) + string(") - _fiberTraceFunction->yLow(=");
      message += to_string(_fiberTraceFunction->yLow) + string(") + 1)=") + to_string(_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1);
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    #ifdef __DEBUG_CREATETRACE__
      cout << "FiberTrace::createFiberTrace: _trace set to " << _trace->getImage()->getArray() << endl;
    #endif
    if (!_isProfileSet){
      _profile.reset(new afwImage::Image<float>(_trace->getWidth(), _trace->getHeight()));
    }
    _isTraceSet = true;
    return true;
  }

  /// Return shared pointer to an image containing the reconstructed 2D spectrum of the FiberTrace
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(afwImage::Image<float>) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::getReconstructed2DSpectrum(const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT> & spectrum) const{
    ndarray::Array<float, 2, 1> F_A2_Rec = ndarray::allocate(_trace->getHeight(), _trace->getWidth());
    int i_row = 0;
    for (auto itProf = _profile->getArray().begin(), itRec = F_A2_Rec.begin(); itProf != _profile->getArray().end(); ++itProf, ++itRec, ++i_row)
      *itRec = (*itProf) * spectrum.getSpectrum()[i_row];
    PTR(afwImage::Image<float>) image(new afwImage::Image<float>(F_A2_Rec));
    return image;
  }

  /// Return shared pointer to an image containing the reconstructed background of the FiberTrace
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(afwImage::Image<float>) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::getReconstructedBackground(const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT> & background) const{
    ndarray::Array<float, 2, 1> F_A2_Rec = ndarray::allocate(_trace->getHeight(), _trace->getWidth());
    int i_row = 0;
    for (auto itProf = _profile->getArray().begin(), itRec = F_A2_Rec.begin(); itProf != _profile->getArray().end(); ++itProf, ++itRec, ++i_row)
      itRec->deep() = background.getSpectrum()[i_row];
    PTR(afwImage::Image<float>) image(new afwImage::Image<float>(F_A2_Rec));
    return image;
  }

  /// Return shared pointer to an image containing the reconstructed 2D spectrum + background of the FiberTrace
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(afwImage::Image<float>) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::getReconstructed2DSpectrum(const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT> & spectrum,
                                                                                                             const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT> & background) const
  {
    PTR(afwImage::Image<float>) imageSpectrum(getReconstructed2DSpectrum(spectrum));
    PTR(afwImage::Image<float>) imageBackground(getReconstructed2DSpectrum(background));
    *imageSpectrum += (*imageBackground);
    return imageSpectrum;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  ndarray::Array<size_t, 2, 1> pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::calcSwathBoundY(const size_t swathWidth) const{
    size_t I_NI = 0;
    size_t I_I = 0;
    size_t nSwaths = 0;

    size_t swathWidth_mutable = swathWidth;
    if (swathWidth_mutable > _trace->getHeight()){
      swathWidth_mutable = _trace->getHeight();
      #ifdef __DEBUG_CALCSWATHBOUNDY__
        cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: KeyWord_Set(SWATH_WIDTH): swathWidth_mutable too large: swathWidth_mutable set to " << swathWidth_mutable << endl;
      #endif
    }
/*    if (swathWidth_mutable == 0)/// Calculate swathWidth automatically
    { /// Estimate the Points of column crossing
      const ndarray::Array<const float, 1, 1> xCenters = ndarray::external(_xCenters.data(), ndarray::makeVector(int(_xCenters.size())), ndarray::makeVector(1));
      #ifdef __DEBUG_CALCSWATHBOUNDY__
        cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: xCenters.getShape()[0] = " << xCenters.getShape()[0] << endl;
      #endif
      for (int i=0; i<xCenters.getShape()[0]; ++i)
        cout << "xCenters[" << i << "] = " << xCenters[i] << endl;
      ndarray::Array<size_t, 1> xCentersInt = ::pfs::drp::stella::math::floor(xCenters, size_t(0));
      for (int i=0; i<xCentersInt.getShape()[0]; ++i)
        cout << "xCentersInt[" << i << "] = " << xCentersInt[i] << endl;
      #ifdef __DEBUG_CALCSWATHBOUNDY__
        cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: !KeyWord_Set(SWATH_WIDTH): tempIntArrA = " << tempIntArrA << endl;
      #endif
      ::pfs::drp::stella::math::Uniq(tempIntArrA, I_A1_I);
      #ifdef __DEBUG_CALCSWATHBOUNDY__
        cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: !KeyWord_Set(SWATH_WIDTH): I_A1_I set to " << I_A1_I << endl;
      #endif

      ///This is how many times this order crosses to the next column
      I_NI = I_A1_I.size();
      #ifdef __DEBUG_CALCSWATHBOUNDY__
        cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: !KeyWord_Set(SWATH_WIDTH): I_NI set to " << I_NI << endl;
      #endif

      ///Curved order crosses columns
      if (I_NI > 1)
      {
        I_I = blitz::sum(I_A1_I(blitz::Range(1, I_NI-1)) - I_A1_I(blitz::Range(0, I_NI - 2))) / (I_NI - 1);
        #ifdef __DEBUG_CALCSWATHBOUNDY__
          cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") > 1): I_I set to " << I_I << endl;
        #endif

        /// number of swaths along the order
        nSwaths = ::pfs::drp::stella::math::Round(static_cast<double>(_trace->getHeight()) / static_cast<double>(I_I) / 3.);
        #ifdef __DEBUG_CALCSWATHBOUNDY__
          cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") > 1): I_NBin = Round((double)_trace->getHeight()(=" << _trace->getHeight() << ") / (double)I_I(=" << I_I << ") / 3.) set to " << nSwaths << endl;
          cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") > 1): I_NBin set to " << nSwaths << endl;
        #endif
      }
      else
      { /// Perfectly aligned orders
        /// Still follow the changes in PSF
        nSwaths = ::pfs::drp::stella::math::Round(static_cast<double>(_trace->getHeight()) / 400.);
        #ifdef __DEBUG_CALCSWATHBOUNDY__
          cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") <= 1): I_NBin = int(_trace->getHeight()(=" << _trace->getHeight() << ") / 400.) set to " << nSwaths << endl;
          cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") <= 1): I_NBin set to " << nSwaths << endl;
        #endif
      }
      if (nSwaths < 3)
        nSwaths = 3;
      if (nSwaths > 20)
        nSwaths = 2;
      #ifdef __DEBUG_CALCSWATHBOUNDY__
        cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: !KeyWord_Set(SWATH_WIDTH): I_NBin set to " << nSwaths << endl;
      #endif
    }
*/
    nSwaths = round(float(_trace->getHeight()) / float(swathWidth));

    size_t binHeight = _trace->getHeight() / nSwaths;
    if (nSwaths > 1)
      nSwaths = (2 * nSwaths) - 1;

    #ifdef __DEBUG_CALCSWATHBOUNDY__
      cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: fiberTraceNumber = " << _iTrace << endl;
      cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: nSwaths set to " << nSwaths << endl;
    #endif

    /// Calculate boundaries of distinct slitf regions.
    /// Boundaries of bins
//    ndarray::Array<double,1,1>::Index shape = ndarray::makeVector(int(nSwaths));
//    ndarray::Array<double,1,1>::Index strides = ndarray::makeVector(1);
    ndarray::Array<size_t, 2, 1> swathBoundY = ndarray::allocate(int(nSwaths), 2);
    #ifdef __DEBUG_CALCSWATHBOUNDY__
      ndarray::Array<double, 2, 1>::Index shape = swathBoundY.getShape();
      cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: shape = " << shape << endl;
    #endif
    swathBoundY[0][0] = 0;
    #ifdef __DEBUG_CALCSWATHBOUNDY__
      cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: 1. swathBoundY[0][0] set to " << swathBoundY[0][0] << endl;
    #endif
    size_t I_BinHeight_Temp = binHeight;
    swathBoundY[0][1] = I_BinHeight_Temp;
    #ifdef __DEBUG_CALCSWATHBOUNDY__
      cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: swathBoundY[0][1] set to " << swathBoundY[0][1] << endl;
    #endif
    for (size_t iSwath = 1; iSwath < nSwaths; iSwath++){
      I_BinHeight_Temp = binHeight;
      if (iSwath == 1)
        swathBoundY[iSwath][0] = swathBoundY[iSwath-1][0] + size_t(double(binHeight) / 2.);
      else
        swathBoundY[iSwath][0] = swathBoundY[iSwath-2][1] + 1;
      #ifdef __DEBUG_CALCSWATHBOUNDY__
        cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: swathBoundY[iSwath=" << iSwath << "][0] set to swathBoundY[iSwath-1=" << iSwath-1 << "][0] + (binHeight/2.=" << binHeight / 2. << ")" << endl;
      #endif
      swathBoundY[iSwath][1] = swathBoundY[iSwath][0] + I_BinHeight_Temp;
      if (iSwath == (nSwaths-1)){
        swathBoundY[iSwath][1] = _trace->getHeight()-1;
        #ifdef __DEBUG_CALCSWATHBOUNDY__
          cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: nSwaths = " << nSwaths << endl;
          cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: _trace->getHeight() = " << _trace->getHeight() << endl;
          cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: swathBoundY[" << iSwath << "][1] set to " << swathBoundY[iSwath][1] << endl;
        #endif
      }
      #ifdef __DEBUG_CALCSWATHBOUNDY__
        cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: swathBoundY[" << iSwath << "][1] set to " << swathBoundY[iSwath][1] << endl;
      #endif
    }
    #ifdef __DEBUG_CALCSWATHBOUNDY__
      cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: swathBoundY set to " << swathBoundY << endl;
    #endif
    return swathBoundY;
  }
  
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::calcProfile(){
    if (!_isTraceSet){
      string message("FiberTrace ");
      message += to_string(_iTrace) + "::calcProfile: ERROR: _Trace is not set";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    if (!_isFiberTraceProfileFittingControlSet){
      string message("FiberTrace ");
      message += to_string(_iTrace) + "::calcProfile: ERROR: _fiberTraceProfileFittingControl is not set";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    
    /// Calculate boundaries for swaths
    const ndarray::Array<const size_t, 2, 1> swathBoundsY = calcSwathBoundY(_fiberTraceProfileFittingControl->swathWidth);
    #ifdef __DEBUG_CALCPROFILE__
      cout << "FiberTrace" << _iTrace << "::calcProfile: swathBoundsY = " << swathBoundsY << endl;
    #endif
    ndarray::Array<size_t, 1, 1> nPixArr = ndarray::allocate(swathBoundsY.getShape()[0]);
    nPixArr[ndarray::view()] = swathBoundsY[ndarray::view()(1)] - swathBoundsY[ndarray::view()(0)] + 1;
    #ifdef __DEBUG_CALCPROFILE__
      cout << "nPixArr = " << nPixArr << endl;
    #endif
    unsigned int nSwaths = swathBoundsY.getShape()[0];
    #ifdef __DEBUG_CALCPROFILE__
      cout << "FiberTrace::calcProfile: trace " << _iTrace << ": nSwaths = " << nSwaths << endl;
      cout << "FiberTrace::calcProfile: trace " << _iTrace << ": _trace->getHeight() = " << _trace->getHeight() << endl;
    #endif

    /// for each swath
    ndarray::Array<float, 3, 2> slitFuncsSwaths = ndarray::allocate(nPixArr[0], _trace->getWidth(), int(nSwaths-1));
    ndarray::Array<float, 2, 2> lastSlitFuncSwath = ndarray::allocate(nPixArr[nPixArr.getShape()[0] - 1], _trace->getWidth());
    #ifdef __DEBUG_CALCPROFILE__
      cout << "slitFuncsSwaths.getShape() = " << slitFuncsSwaths.getShape() << endl;
      cout << "slitFuncsSwaths.getShape()[0] = " << slitFuncsSwaths.getShape()[0] << ", slitFuncsSwaths.getShape()[1] = " << slitFuncsSwaths.getShape()[1] << ", slitFuncsSwaths.getShape()[2] = " << slitFuncsSwaths.getShape()[2] << endl;
      cout << "slitFuncsSwaths[view(0)()(0) = " << ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(0)()(0)]) << endl;
      cout << "slitFuncsSwaths[view(0)(0,9)(0) = " << ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(0)(0,slitFuncsSwaths.getShape()[1])(0)]) << endl;
    #endif
    ndarray::Array<double, 3, 2>::Index shapeSFsSwaths = slitFuncsSwaths.getShape();
    #ifdef __DEBUG_CALCPROFILE__
      cout << "FiberTrace" << _iTrace << "::calcProfile: shapeSFsSwaths = (" << shapeSFsSwaths[0] << ", " << shapeSFsSwaths[1] << ", " << shapeSFsSwaths[2] << ")" << endl;
    #endif
    _overSampledProfileFitXPerSwath.resize(0);
    _overSampledProfileFitYPerSwath.resize(0);
    _profileFittingInputXPerSwath.resize(0);
    _profileFittingInputYPerSwath.resize(0);
    _profileFittingInputXMeanPerSwath.resize(0);
    _profileFittingInputYMeanPerSwath.resize(0);
    for (unsigned int iSwath = 0; iSwath < nSwaths; ++iSwath){
      int iMin = int(swathBoundsY[iSwath][0]);
      int iMax = int(swathBoundsY[iSwath][1] + 1);
      #ifdef __DEBUG_CALCPROFILE__
        cout << "FiberTrace::calcProfile: trace " << _iTrace << ": iSwath = " << iSwath << ": iMin = " << iMin << ", iMax = " << iMax << endl;
      #endif
      const ndarray::Array<ImageT const, 2, 2> imageSwath = ndarray::copy(_trace->getImage()->getArray()[ndarray::view(iMin, iMax)()]);
      #ifdef __DEBUG_CALCPROFILE__
        cout << "FiberTrace::calcProfile: trace " << _iTrace << ": swath " << iSwath << ": imageSwath = " << imageSwath << endl;
      #endif
      const ndarray::Array<MaskT const, 2, 2> maskSwath = ndarray::copy(_trace->getMask()->getArray()[ndarray::view(iMin, iMax)()]);
      const ndarray::Array<VarianceT const, 2, 2> varianceSwath = ndarray::copy(_trace->getVariance()->getArray()[ndarray::view(iMin, iMax)()]);
      const ndarray::Array<float const, 1, 1> xCentersSwath = ndarray::copy(_xCenters[ndarray::view(iMin, iMax)]);

      if (iSwath < nSwaths - 1){
        slitFuncsSwaths[ndarray::view()()(iSwath)] = calcProfileSwath(imageSwath, 
                                                                      maskSwath,
                                                                      varianceSwath,
                                                                      xCentersSwath,
                                                                      iSwath);
        #ifdef __DEBUG_CALCPROFILE__
          cout << "slitFuncsSwaths.getShape() = " << slitFuncsSwaths.getShape() << endl;
          cout << "imageSwath.getShape() = " << imageSwath.getShape() << endl;
          cout << "xCentersSwath.getShape() = " << xCentersSwath.getShape() << endl;
          cout << "swathBoundsY = " << swathBoundsY << endl;
          cout << "nPixArr = " << nPixArr << endl;
          cout << "FiberTrace::calcProfile: trace " << _iTrace << ": swath " << iSwath << ": slitFuncsSwaths[ndarray::view()()(iSwath)] = " << slitFuncsSwaths[ndarray::view()()(iSwath)] << endl;
        #endif
      }
      else{
        lastSlitFuncSwath.deep() = calcProfileSwath(imageSwath, 
                                                    maskSwath,
                                                    varianceSwath,
                                                    xCentersSwath,
                                                    iSwath);
      }
    }
    
    if (nSwaths == 1){
      _profile->getArray() = lastSlitFuncSwath;
      return true;
    }
    int I_Bin = 0;
    float D_Weight_Bin0 = 0.;
    float D_Weight_Bin1 = 0.;
    float D_RowSum;
    for (int iSwath = 0; iSwath < nSwaths - 1; iSwath++){
      for (int i_row = 0; i_row < nPixArr[iSwath]; i_row++){
        D_RowSum = ndarray::sum(ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(i_row)()(iSwath)]));
        if (fabs(D_RowSum) > 0.00000000000000001){
          for (int iPix = 0; iPix < slitFuncsSwaths.getShape()[1]; iPix++){
            slitFuncsSwaths[i_row][iPix][iSwath] = slitFuncsSwaths[i_row][iPix][iSwath] / D_RowSum;
          }
          #ifdef __DEBUG_CALCPROFILE__
            cout << "slitFuncsSwaths(" << i_row << ", *, " << iSwath << ") = " << ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(i_row)()(iSwath)]) << endl;
            D_RowSum = ndarray::sum(ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(i_row)()(iSwath)]));
            cout << "i_row = " << i_row << ": iSwath = " << iSwath << ": D_RowSum = " << D_RowSum << endl;
          #endif
        }
      }
    }
    for (int i_row = 0; i_row < nPixArr[nPixArr.getShape()[0]-1]; i_row++){
      D_RowSum = ndarray::sum(lastSlitFuncSwath[ndarray::view(i_row)()]);
      if (fabs(D_RowSum) > 0.00000000000000001){
        lastSlitFuncSwath[ndarray::view(i_row)()] = lastSlitFuncSwath[ndarray::view(i_row)()] / D_RowSum;
        #ifdef __DEBUG_CALCPROFILE__
          cout << "lastSlitFuncSwath(" << i_row << ", *) = " << lastSlitFuncSwath[ndarray::view(i_row)()] << endl;
          cout << "i_row = " << i_row << ": D_RowSum = " << D_RowSum << endl;
        #endif
      }
    }
    #ifdef __DEBUG_CALCPROFILE__
      cout << "swathBoundsY.getShape() = " << swathBoundsY.getShape() << ", nSwaths = " << nSwaths << endl;
    #endif
    int iRowSwath = 0;
    for (int i_row = 0; i_row < _trace->getHeight(); ++i_row){
      iRowSwath = i_row - swathBoundsY[I_Bin][0];
      #ifdef __DEBUG_CALCPROFILE__
        cout << "swathBoundsY = " << swathBoundsY << endl;
        cout << "i_row = " << i_row << ", I_Bin = " << I_Bin << ", iRowSwath = " << iRowSwath << endl;
      #endif
      if ((I_Bin == 0) && (i_row < swathBoundsY[1][0])){
        #ifdef __DEBUG_CALCPROFILE__
          cout << "I_Bin=" << I_Bin << " == 0 && i_row=" << i_row << " < swathBoundsY[1][0]=" << swathBoundsY[1][0] << endl;
        #endif
        _profile->getArray()[ndarray::view(i_row)()] = ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()(0)]);
      }
      else if ((I_Bin == nSwaths-1) && (i_row >= swathBoundsY[I_Bin-1][1])){// && (i_row > (I_A2_IBound(I_Bin-1, 1) - (I_A2_IBound(0,1) / 2.)))){
        #ifdef __DEBUG_CALCPROFILE__
          cout << "I_Bin=" << I_Bin << " == nSwaths-1=" << nSwaths-1 << " && i_row=" << i_row << " >= swathBoundsY[I_Bin-1=" << I_Bin - 1 << "][0]=" << swathBoundsY[I_Bin-1][0] << endl;
        #endif
        _profile->getArray()[ndarray::view(i_row)()] = lastSlitFuncSwath[ndarray::view(iRowSwath)()];
      }
      else{
        D_Weight_Bin1 = double(i_row - swathBoundsY[I_Bin+1][0]) / double(swathBoundsY[I_Bin][1] - swathBoundsY[I_Bin+1][0]);
        D_Weight_Bin0 = 1. - D_Weight_Bin1;
        #ifdef __DEBUG_CALCPROFILE__
          cout << "FiberTrace" << _iTrace << "::calcProfile: i_row = " << i_row << ": nSwaths = " << nSwaths << endl;
          cout << "FiberTrace" << _iTrace << "::calcProfile: i_row = " << i_row << ": I_Bin = " << I_Bin << endl;
          cout << "FiberTrace" << _iTrace << "::calcProfile: i_row = " << i_row << ": swathBoundsY(I_Bin, *) = " << swathBoundsY[ndarray::view(I_Bin)()] << endl;
          cout << "FiberTrace" << _iTrace << "::calcProfile: i_row = " << i_row << ": swathBoundsY(I_Bin+1, *) = " << swathBoundsY[ndarray::view(I_Bin+1)()] << endl;
          cout << "FiberTrace" << _iTrace << "::calcProfile: i_row = " << i_row << ": D_Weight_Bin0 = " << D_Weight_Bin0 << endl;
          cout << "FiberTrace" << _iTrace << "::calcProfile: i_row = " << i_row << ": D_Weight_Bin1 = " << D_Weight_Bin1 << endl;
        #endif
        if (I_Bin == nSwaths - 2){
          #ifdef __DEBUG_CALCPROFILE__
            cout << "FiberTrace" << _iTrace << "::calcProfile: slitFuncsSwaths(iRowSwath, *, I_Bin) = " << ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)]) << ", lastSlitFuncSwath[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0])=" << int(i_row - swathBoundsY[I_Bin+1][0]) << ")] = " << lastSlitFuncSwath[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()] << endl;
            cout << "_profile->getArray().getShape() = " << _profile->getArray().getShape() << endl;
            cout << "slitFuncsSwath.getShape() = " << slitFuncsSwaths.getShape() << endl;
            cout << "lastSlitFuncSwath.getShape() = " << lastSlitFuncSwath.getShape() << endl;
          #endif
          _profile->getArray()[ndarray::view(i_row)()] = (ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)]) * D_Weight_Bin0) + (lastSlitFuncSwath[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()] * D_Weight_Bin1);
        }
        else{
          #ifdef __DEBUG_CALCPROFILE__
            cout << "FiberTrace" << _iTrace << "::calcProfile: slitFuncsSwaths(iRowSwath, *, I_Bin) = " << ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)]) << ", slitFuncsSwaths(int(i_row - swathBoundsY[I_Bin+1][0])=" << int(i_row - swathBoundsY[I_Bin+1][0]) << ", *, I_Bin+1) = " << ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()(I_Bin+1)]) << endl;
          #endif            
          _profile->getArray()[ndarray::view(i_row)()] = (ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)]) * D_Weight_Bin0) + (ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()(I_Bin+1)]) * D_Weight_Bin1);
        }
        double dSumSFRow = ndarray::sum(_profile->getArray()[ndarray::view(i_row)()]);
        if (fabs(dSumSFRow) >= 0.00000000000000001)
          _profile->getArray()[ndarray::view(i_row)()] = _profile->getArray()[ndarray::view(i_row)()] / dSumSFRow;
        #ifdef __DEBUG_CALCPROFILE__
          cout << "FiberTrace" << _iTrace << "::calcProfile: i_row = " << i_row << ": I_Bin = " << I_Bin << ": _profile->getArray()(" << i_row << ", *) set to " << _profile->getArray()[ndarray::view(i_row)()] << endl;
        #endif
      }

      if (i_row == swathBoundsY[I_Bin][1]){
        I_Bin++;
        #ifdef __DEBUG_CALCPROFILE__
          cout << "FiberTrace" << _iTrace << "::calcProfile: i_row = " << i_row << ": I_Bin set to " << I_Bin << endl;
        #endif
      }
    }/// end for (int i_row = 0; i_row < slitFuncsSwaths.rows(); i_row++){
    #ifdef __DEBUG_CALCPROFILE__
      cout << "FiberTrace" << _iTrace << "::calcProfile: _profile->getArray() set to " << _profile->getArray() << endl; 
    #endif  
    
    _isProfileSet = true;
    return true;
  }
  
  template<typename ImageT, typename MaskT, typename VarianceT>
  ndarray::Array<float, 2, 2> pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::calcProfileSwath(ndarray::Array<ImageT const, 2, 2> const& imageSwath,
                                                                                                   ndarray::Array<MaskT const, 2, 2> const& maskSwath,
                                                                                                   ndarray::Array<VarianceT const, 2, 2> const& varianceSwath,
                                                                                                   ndarray::Array<float const, 1, 1> const& xCentersSwath,
                                                                                                   size_t const iSwath){
    
    /// Check shapes of input arrays
    assert(imageSwath.getShape()[0] == maskSwath.getShape()[0]);
    assert(imageSwath.getShape()[0] == varianceSwath.getShape()[0]); 
    assert(imageSwath.getShape()[0] == xCentersSwath.getShape()[0]);
    assert(imageSwath.getShape()[1] == maskSwath.getShape()[1]);
    assert(imageSwath.getShape()[1] == varianceSwath.getShape()[1]);
    assert(imageSwath.getShape()[1] == xCentersSwath.getShape()[1]);
    
    /// Normalize rows in imageSwath
    ndarray::Array<ImageT, 2, 2> imageSwathNormalized = ndarray::allocate(imageSwath.getShape()[0], imageSwath.getShape()[1]);
    ndarray::Array<ImageT, 1, 1> sumArr = ndarray::allocate(imageSwath.getShape()[1]);
    for (int iRow = 0; iRow < imageSwath.getShape()[0]; ++iRow){
      sumArr.deep() = imageSwath[ndarray::view(iRow)()];
      imageSwathNormalized[ndarray::view(iRow)()] = imageSwath[ndarray::view(iRow)()] / ndarray::sum(sumArr);
    }
    #ifdef __DEBUG_CALCPROFILESWATH__
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": imageSwath = " << imageSwath << endl;
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": imageSwathNormalized = " << imageSwathNormalized << endl;
    #endif  
 
    /// Calculate pixel offset to xCenter
    const ndarray::Array<size_t const, 1, 1> xCentersInt = math::floor(xCentersSwath, size_t(0));
    ndarray::Array<float, 1, 1> pixelOffset = ndarray::allocate(xCentersSwath.getShape()[0]);
    pixelOffset.deep() = 0.5;
    pixelOffset.deep() -= xCentersSwath;
    pixelOffset.deep() += xCentersInt;
    #ifdef __DEBUG_CALCPROFILESWATH__
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": pixelOffset = " << pixelOffset << endl;
    #endif
    const ndarray::Array<float const, 1, 1> xCenterArrayIndexX = math::indGenNdArr(float(imageSwath.getShape()[1]));
    const ndarray::Array<float const, 1, 1> xCenterArrayIndexY = math::replicate(float(1), int(imageSwath.getShape()[0]));
    ndarray::Array<float, 2, 2> xArray = ndarray::allocate(imageSwath.getShape()[0], imageSwath.getShape()[1]);
    xArray.asEigen() = xCenterArrayIndexY.asEigen() * xCenterArrayIndexX.asEigen().transpose();
    float xMin = 1.;
    float xMax = -1.;
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
        yVecPtr->push_back(float(*itColIm));
      }
    }
    _profileFittingInputXPerSwath.push_back(xVecPtr);
    _profileFittingInputYPerSwath.push_back(yVecPtr);
    #ifdef __DEBUG_CALCPROFILESWATH__
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": xArray = " << xArray << endl;
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": xMin = " << xMin << ", xMax = " << xMax << endl;
    #endif
    float xOverSampleStep = 1. / _fiberTraceProfileFittingControl->overSample;
    #ifdef __DEBUG_CALCPROFILESWATH__
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": initial xOverSampleStep = " << xOverSampleStep << endl;
    #endif  
    
    ///adjust xOverSampleStep to cover x from xMin + xOverSampleStep/2 to xMax - xOverSampleStep/2
    int nSteps = (xMax - xMin) / xOverSampleStep + 1;
    xOverSampleStep = (xMax - xMin) / nSteps;
    #ifdef __DEBUG_CALCPROFILESWATH__
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": final xOverSampleStep = " << xOverSampleStep << endl;
    #endif
    ndarray::Array<float, 1, 1> xOverSampled = ndarray::allocate(nSteps);
    float xStart = xMin + (xOverSampleStep / 2.);
    int iStep = 0;
    for (auto it = xOverSampled.begin(); it != xOverSampled.end(); ++it){
      *it = xStart + (iStep * xOverSampleStep);
      ++iStep;
    }
    #ifdef __DEBUG_CALCPROFILESWATH__
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": xOverSampled = " << xOverSampled << endl;
    #endif
    PTR(vector<float>) xOverSampledFitVec(new vector<float>(xOverSampled.begin(), xOverSampled.end()));
    _overSampledProfileFitXPerSwath.push_back(xOverSampledFitVec);
    
    /// calculate oversampled profile values
    iStep = 0;
    std::vector< std::pair<float, ImageT> > valOverSampledVec;
    int iStepsWithValues = 0;
    int bThisStepHasValues = false;
    ImageT mean = 0.;
    float rangeStart = float(xOverSampled[0] - (xOverSampleStep / 2.));
    float rangeEnd = rangeStart + xOverSampleStep;
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
        cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": iStep" << iStep << ": rangeStart = " << rangeStart << ", rangeEnd = " << rangeEnd << endl;
        printf("rangeStart = %.9f, rangeEnd = %.9f\n", rangeStart, rangeEnd);
      #endif
      ndarray::Array<size_t, 2, 2> indicesInValueRange = math::getIndicesInValueRange(xArray, rangeStart, rangeEnd);
      #ifdef __DEBUG_CALCPROFILESWATH__
        cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": iStep" << iStep << ": indicesInValueRange = " << indicesInValueRange << endl;
      #endif
      std::vector< std::pair<size_t, size_t> > indicesInValueRangeVec(indicesInValueRange.getShape()[0]);
      for (int i = 0; i < indicesInValueRange.getShape()[0]; ++i){
        indicesInValueRangeVec[i].first = indicesInValueRange[i][0];
        indicesInValueRangeVec[i].second = indicesInValueRange[i][1];
      }
      do{
        ndarray::Array<ImageT, 1, 1> subArr = math::getSubArray(imageSwathNormalized, indicesInValueRangeVec);
        ndarray::Array<float, 1, 1> xSubArr = math::getSubArray(xArray, indicesInValueRangeVec);
        nValues = subArr.getShape()[0];
        #ifdef __DEBUG_CALCPROFILESWATH__
          cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": iStep" << iStep << ": iterSig = " << iterSig << ": nValues = " << nValues << endl;
        #endif
        if (nValues > 1){
          #ifdef __DEBUG_CALCPROFILESWATH__
            cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": iStep = " << iStep << ": iterSig = " << iterSig << ": xSubArr = [" << xSubArr.getShape()[0] << "]: " << xSubArr << endl;
            cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": iStep = " << iStep << ": iterSig = " << iterSig << ": subArr = [" << subArr.getShape()[0] << "]: " << subArr << endl;
          #endif
          if (_fiberTraceProfileFittingControl->maxIterSig > iterSig){
            ndarray::Array<ImageT, 1, 1> moments = math::moment(subArr, 2);
            #ifdef __DEBUG_CALCPROFILESWATH__
              cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": iStep = " << iStep << ": iterSig = " << iterSig << ": moments = " << moments << endl;
            #endif
            for (int i = subArr.getShape()[0] - 1; i >= 0; --i){
              #ifdef __DEBUG_CALCPROFILESWATH__
                cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": iStep" << iStep << ": iterSig = " << iterSig << ": moments[0](=" << moments[0] << ") - subArr[" << i << "](=" << subArr[i] << ") = " << moments[0] - subArr[i] << ", 0. - (_fiberTraceProfileFittingControl->upperSigma(=" << _fiberTraceProfileFittingControl->upperSigma << ") * sqrt(moments[1](=" << moments[1] << "))(= " << sqrt(moments[1]) << ") = " << 0. - (_fiberTraceProfileFittingControl->upperSigma * sqrt(moments[1])) << endl;
                cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": iStep" << iStep << ": iterSig = " << iterSig << ": _fiberTraceProfileFittingControl->lowerSigma(=" << _fiberTraceProfileFittingControl->lowerSigma << ") * sqrt(moments[1](=" << moments[1] << "))(= " << sqrt(moments[1]) << ") = " << _fiberTraceProfileFittingControl->lowerSigma * sqrt(moments[1]) << endl;
              #endif
              if ((moments[0] - subArr[i] < 0. - (_fiberTraceProfileFittingControl->upperSigma * sqrt(moments[1])))
               || (moments[0] - subArr[i] > (_fiberTraceProfileFittingControl->lowerSigma * sqrt(moments[1])))){
//                #ifdef __DEBUG_CALCPROFILESWATH__
                  cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": iStep = " << iStep << ": iterSig = " << iterSig << ": rejecting element " << i << "from subArr" << endl;
//                #endif
                indicesInValueRangeVec.erase(indicesInValueRangeVec.begin() + i);
              }
            }
          }
          ndarray::Array<ImageT, 1, 1> moments = math::moment(subArr, 1);
          mean = moments[0];
          #ifdef __DEBUG_CALCPROFILESWATH__
            cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": iStep = " << iStep << ": iterSig = " << iterSig << ": mean = " << mean << endl;
          #endif
          ++iStepsWithValues;
          bThisStepHasValues = true;
        }
        ++iterSig;
      } while (iterSig <= _fiberTraceProfileFittingControl->maxIterSig);
      if (bThisStepHasValues){
        valOverSampledVec.push_back(std::pair<float, ImageT>(*it, mean));
        #ifdef __DEBUG_CALCPROFILESWATH__
          cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": valOverSampledVec[" << iStep << "] = (" << valOverSampledVec[iStep].first << ", " << valOverSampledVec[iStep].second << ")" << endl;
        #endif
      }
      rangeStart = rangeEnd;
      rangeEnd += xOverSampleStep;
    }
    ndarray::Array<ImageT, 1, 1> valOverSampled = ndarray::allocate(valOverSampledVec.size());
    ndarray::Array<float, 1, 1> xValOverSampled = ndarray::allocate(valOverSampledVec.size());
    for (int iRow = 0; iRow < valOverSampledVec.size(); ++iRow){
      xValOverSampled[iRow] = valOverSampledVec[iRow].first;
      valOverSampled[iRow] = valOverSampledVec[iRow].second;
      #ifdef __DEBUG_CALCPROFILESWATH__
        cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": (x)valOverSampled[" << iRow << "] = (" << xValOverSampled[iRow] << "," << valOverSampled[iRow] << ")" << endl;
      #endif
    }
    PTR(std::vector<float>) xVecMean(new vector<float>(xValOverSampled.begin(), xValOverSampled.end()));
    std::vector<ImageT> yVecMean(valOverSampled.begin(), valOverSampled.end());
    
    PTR(std::vector<float>) yVecMeanF(new vector<float>(yVecMean.size()));
    auto itF = yVecMeanF->begin();
    for (auto itT = yVecMean.begin(); itT != yVecMean.end(); ++itT, ++itF){
      *itF = float(*itT);
    }
    _profileFittingInputXMeanPerSwath.push_back(xVecMean);
    _profileFittingInputYMeanPerSwath.push_back(yVecMeanF);
    
    math::spline<float> spline;
    spline.set_points(*xVecMean, *yVecMeanF);    // currently it is required that X is already sorted

    PTR(vector<float>) yOverSampledFitVec(new vector<float>(nSteps));
    _overSampledProfileFitYPerSwath.push_back(yOverSampledFitVec);
    for (auto itX = xOverSampledFitVec->begin(), itY = yOverSampledFitVec->begin(); itX != xOverSampledFitVec->end(); ++itX, ++itY)
      *itY = spline(*itX);
    #ifdef __DEBUG_CALCPROFILESWATH__
    std::vector<float> yVecFit(yVecMean.size());
      for (int iRow = 0; iRow < yVecMean.size(); ++iRow){
        yVecFit[iRow] = spline((*xVecMean)[iRow]);
        cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": yVecMean[" << iRow << "] = " << yVecMean[iRow] << ", yVecFit[" << iRow << "] = " << yVecFit[iRow] << endl;
      }
    #endif
    
    /// calculate profile for each row in imageSwath
    ndarray::Array<float, 2, 2> profArraySwath = ndarray::allocate(imageSwath.getShape()[0], imageSwath.getShape()[1]);
    for (int iRow = 0; iRow < imageSwath.getShape()[0]; ++iRow){
      #ifdef __DEBUG_CALCPROFILESWATH__
        cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": xArray[" << iRow << "][*] = " << xArray[ndarray::view(iRow)()] << endl;
      #endif
      for (int iCol = 0; iCol < imageSwath.getShape()[1]; ++iCol){
        if (xArray[iRow][iCol] < (*xVecMean)[0])
          profArraySwath[iRow][iCol] = (*yVecMeanF)[0];
        else if (xArray[iRow][iCol] > (*xVecMean)[xVecMean->size() - 1])
          profArraySwath[iRow][iCol] = (*yVecMeanF)[yVecMeanF->size() - 1];
        else
          profArraySwath[iRow][iCol] = spline(xArray[iRow][iCol]);
        if (profArraySwath[iRow][iCol] < 0.)
          profArraySwath[iRow][iCol] = 0.;
      }
      #ifdef __DEBUG_CALCPROFILESWATH__
        cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": profArraySwath[" << iRow << "][*] = " << profArraySwath[ndarray::view(iRow)()] << endl;
      #endif
      profArraySwath[ndarray::view(iRow)()] = profArraySwath[ndarray::view(iRow)()] / ndarray::sum(profArraySwath[ndarray::view(iRow)()]);
      #ifdef __DEBUG_CALCPROFILESWATH__
        cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": normalized profArraySwath[" << iRow << "][*] = " << profArraySwath[ndarray::view(iRow)()] << endl;
      #endif
    }
    #ifdef __DEBUG_CALCPROFILESWATH__
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": profArraySwath = " << profArraySwath << endl;
    #endif
    return profArraySwath;
  }
/*  
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::fitSpline(const blitz::Array<double, 2> &fiberTraceSwath_In,
                                                                     const blitz::Array<double, 1> &xOverSampled_In,
                                                                     blitz::Array<double, 1> &profileOverSampled_Out,
                                                                     const blitz::Array<double, 2> &profileXValuesPerRowOverSampled_In,
                                                                     const blitz::Array<double, 1> &profileXValuesAllRows_In,
                                                                     blitz::Array<double, 2> &profilePerRow_Out)
  {
    #ifdef __DEBUG_SPLINE__
      cout << "FiberTrace" << _iTrace << "::fitSpline: fiberTraceSwath_In = " << fiberTraceSwath_In << endl;
//      cout << "FiberTrace" << _iTrace << "::fitSpline: iFirst_In = " << iFirst_In << endl;
      cout << "FiberTrace" << _iTrace << "::fitSpline: xOverSampled_In = " << xOverSampled_In << endl;
      cout << "FiberTrace" << _iTrace << "::fitSpline: profileXValuesPerRowOverSampled_In = " << profileXValuesPerRowOverSampled_In << endl;
      cout << "FiberTrace" << _iTrace << "::fitSpline: profileXValuesAllRows_In = " << profileXValuesAllRows_In << endl;
    #endif
    /// check input paramters
    if (fiberTraceSwath_In.cols() != static_cast<int>(profileXValuesAllRows_In.size())){
      string message("FiberTrace");
      message += to_string(_iTrace) + "::fitSpline: ERROR: profileXValuesAllRows_In.size(=" + to_string(profileXValuesAllRows_In.size());
      message += ") != fiberTraceSwath_In.cols(=" + to_string(fiberTraceSwath_In.cols()) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    if (fiberTraceSwath_In.rows() != profileXValuesPerRowOverSampled_In.rows()){
      string message("FiberTrace");
      message += to_string(_iTrace) + "::fitSpline: ERROR: profileXValuesPerRowOverSampled_In.size(=" + to_string(profileXValuesPerRowOverSampled_In.size());
      message += ") != fiberTraceSwath_In.rows(=" + to_string(fiberTraceSwath_In.rows()) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    blitz::Array<double, 1> ccdRow(fiberTraceSwath_In.cols());
    std::vector<double> xVec(fiberTraceSwath_In.rows() * fiberTraceSwath_In.cols());
    std::vector<double> yVec(xVec.size());
    std::vector<double>::iterator iter_xVec = xVec.begin();
    std::vector<double>::iterator iter_yVec = yVec.begin();
    blitz::Array<int, 1> TempIVecArr(profileXValuesPerRowOverSampled_In.cols());
    int NInd, iFirst;
    blitz::Array<int, 1> IndVecArr(1);
    for (int i = 0; i < fiberTraceSwath_In.rows(); ++i){
      TempIVecArr = blitz::where((profileXValuesPerRowOverSampled_In(i, blitz::Range::all()) > 0), 1, 0);
      math::GetIndex(TempIVecArr, NInd, IndVecArr);
      iFirst = IndVecArr(0);
      ccdRow = fiberTraceSwath_In(i,blitz::Range::all()) / blitz::sum(fiberTraceSwath_In(i,blitz::Range::all()));
      for (int j = 0; j < fiberTraceSwath_In.cols(); ++j){
        *iter_xVec = double(j) * double(_fiberTraceProfileFittingControl->overSample) + double(iFirst) + (double(_fiberTraceProfileFittingControl->overSample)/2.);
        *iter_yVec = ccdRow(j);
        ++iter_xVec;
        ++iter_yVec;
      }
    }

    blitz::Array<double, 1> D_A1_xVec(xVec.data(), blitz::shape(xVec.size()), blitz::neverDeleteData);
    blitz::Array<double, 1> D_A1_yVec(yVec.data(), blitz::shape(yVec.size()), blitz::neverDeleteData);
    blitz::Array<int, 1> I_A1_Uniq(1);
    if (!::pfs::drp::stella::math::Uniq(D_A1_xVec, I_A1_Uniq)){
      string message("FiberTrace");
      message += to_string(_iTrace) + "::fitSpline: ERROR: Uniq returned FALSE";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    #ifdef __DEBUG_SPLINE__
      cout << "FiberTrace" << _iTrace << "::fitSpline: I_A1_Uniq = " << I_A1_Uniq << endl;
    #endif
    std::vector<double> xVecSorted(I_A1_Uniq.size());
    std::vector<double> yVecSorted(I_A1_Uniq.size());
    std::vector<double>::iterator iter_xVecSorted = xVecSorted.begin();
    std::vector<double>::iterator iter_yVecSorted = yVecSorted.begin();
    blitz::Array<int, 1> I_A1_Where(D_A1_xVec.size());
    blitz::Array<int, 1> *P_I_A1_Where;
    blitz::Array<double, 1> D_A1_SubArr(1);
    int count = 0;
    double median = 0.;
    blitz::Array<double, 1> D_A1_XTemp = ::pfs::drp::stella::math::DIndGenArr(xOverSampled_In.size());
    blitz::Array<double, 1> D_X(1);
    blitz::Array<double, 1> D_Y(1);
    for (size_t i = 0; i < I_A1_Uniq.size(); ++i){
      D_X(0) = D_A1_xVec(I_A1_Uniq(i));
      if (!::pfs::drp::stella::math::InterPol(xOverSampled_In, D_A1_XTemp, D_X, D_Y)){
        cout << "FiberTrace" << _iTrace << "::fitSpline: ERROR: InterPos(xOverSampled_In=" << xOverSampled_In << ", D_A1_XTemp=" << D_A1_XTemp << ", D_X=" << D_X << ", D_Y) returned FALSE" << endl;
        string message("FiberTrace");
        message += to_string(_iTrace) + "::fitSpline: ERROR: InterPol(xOverSampled_In, D_A1_XTemp, D_X, D_Y) returned FALSE";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      *iter_xVecSorted = D_Y(0);
      I_A1_Where = blitz::where(fabs(D_A1_xVec - D_A1_xVec(I_A1_Uniq(i))) < 0.000001, 1, 0);
      P_I_A1_Where = ::pfs::drp::stella::math::GetIndex(I_A1_Where, count);
      if (!::pfs::drp::stella::math::GetSubArrCopy(D_A1_yVec, *P_I_A1_Where, D_A1_SubArr)){
        string message("FiberTrace");
        message += to_string(_iTrace) + "::fitSpline: i=" + to_string(i) + ": ERROR: GetSubArrCopy returned false";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      median = ::pfs::drp::stella::math::Median(D_A1_SubArr);
      #ifdef __DEBUG_SPLINE__
        cout << "FiberTrace" << _iTrace << "::fitSpline: i=" << i << ": D_A1_xVec(I_A1_Uniq(i)=" << I_A1_Uniq(i) << ") = " << D_A1_xVec(I_A1_Uniq(i)) << ": *P_I_A1_Where = " << *P_I_A1_Where << endl;
        cout << "FiberTrace" << _iTrace << "::fitSpline: i=" << i << ": D_A1_SubArr = " << D_A1_SubArr << endl;
        cout << "FiberTrace" << _iTrace << "::fitSpline: i=" << i << ": median = " << median << endl;
      #endif
      *iter_yVecSorted = median;
      ++iter_xVecSorted;
      ++iter_yVecSorted;
      delete(P_I_A1_Where);
    }
    #ifdef __DEBUG_SPLINE__
      blitz::Array<double, 1> D_A1_XVecSorted(xVecSorted.data(), blitz::shape(xVecSorted.size()), blitz::neverDeleteData);
      blitz::Array<double, 1> D_A1_YVecSorted(yVecSorted.data(), blitz::shape(yVecSorted.size()), blitz::neverDeleteData);
      cout << "FiberTrace" << _iTrace << "::fitSpline: xVecSorted = " << D_A1_XVecSorted << endl;
      cout << "FiberTrace" << _iTrace << "::fitSpline: yVecSorted = " << D_A1_YVecSorted << endl;
    #endif
    ::pfs::drp::stella::math::spline<double> spline;
    spline.set_points(xVecSorted,yVecSorted);    // currently it is required that X is already sorted

    /// calculate oversampled profile for each x in xOverSampled_In
    if (profileOverSampled_Out.size() != xOverSampled_In.size())
      profileOverSampled_Out.resize(xOverSampled_In.size());
    for (int i=0; i < static_cast<int>(xOverSampled_In.size()); ++i){
      if ((xOverSampled_In(i) < xVecSorted[0]) || (xOverSampled_In(i) > xVecSorted[xVecSorted.size()-1]))
        profileOverSampled_Out(i) = 0.;
      else
        profileOverSampled_Out(i) = spline(xOverSampled_In(i));
    }
    #ifdef __DEBUG_SPLINE__
      cout << "FiberTrace" << _iTrace << "::fitSpline: xOverSampled_In = " << xOverSampled_In << endl;
      cout << "FiberTrace" << _iTrace << "::fitSpline: profileOverSampled_Out = " << profileOverSampled_Out << endl;
    #endif
    profileOverSampled_Out = profileOverSampled_Out * double(_fiberTraceProfileFittingControl->overSample) / blitz::sum(profileOverSampled_Out);

    if ((profilePerRow_Out.rows() != fiberTraceSwath_In.rows()) || (profilePerRow_Out.cols() != fiberTraceSwath_In.cols()))
      profilePerRow_Out.resize(fiberTraceSwath_In.rows(), fiberTraceSwath_In.cols());

    blitz::Array<double, 1> yProf(profileXValuesAllRows_In.size());
    for (int i_row = 0; i_row < fiberTraceSwath_In.rows(); i_row++){
      if (!::pfs::drp::stella::math::InterPol(profileOverSampled_Out, profileXValuesPerRowOverSampled_In(i_row, blitz::Range::all()), profileXValuesAllRows_In, yProf)){
        cout << "FiberTrace" << _iTrace << "::fitSpline: ERROR: InterPol(profileOverSampled_Out=" << profileOverSampled_Out << ", profileXValuesPerRowOverSampled_In(i_row=" << i_row << ", blitz::Range::all())=" << profileXValuesPerRowOverSampled_In(i_row, blitz::Range::all()) << ", profileXValuesAllRows_In=" << profileXValuesAllRows_In << ", yProf) returned FALSE" << endl;
        string message("FiberTrace");
        message += to_string(_iTrace) + "::fitSpline: ERROR: InterPol(profileOverSampled_Out, profileXValuesPerRowOverSampled_In(i_row=";
        message += to_string(i_row) + ", blitz::Range::all()), profileXValuesAllRows_In, yProf) returned FALSE";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      profilePerRow_Out(i_row, blitz::Range::all()) = blitz::where(yProf < 0., 0., yProf);
      profilePerRow_Out(i_row, blitz::Range::all()) = profilePerRow_Out(i_row, blitz::Range::all()) / blitz::sum(profilePerRow_Out(i_row, blitz::Range::all()));
    }
    #ifdef __DEBUG_SPLINE__
      cout << "FiberTrace" << _iTrace << "::fitSpline: profilePerRow_Out = " << profilePerRow_Out << endl;
    #endif
    return true;
  }
  */
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::getPointer(){
    PTR(FiberTrace) ptr(new FiberTrace(*this));
    return ptr;
  }


  /**
   * class FiberTraceSet
   **/
  template<typename ImageT, typename MaskT, typename VarianceT>
  pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::FiberTraceSet(size_t nTraces)
        : _traces(new std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>(nTraces))
  {
    for (size_t i=0; i<nTraces; ++i){
      PTR(FiberTrace<ImageT, MaskT, VarianceT>) fiberTrace(new FiberTrace<ImageT, MaskT, VarianceT>(0,0,i));
      (*_traces)[i] = fiberTrace;
    }
  }
  
  template<typename ImageT, typename MaskT, typename VarianceT>
  pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::FiberTraceSet(pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT> const &fiberTraceSet, bool const deep)
      : _traces(fiberTraceSet.getTraces())
  {
    if (deep){
      PTR(std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>) ptr(new std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>(fiberTraceSet.size()));
      _traces.reset();
      _traces = ptr;
      for (size_t i = 0; i < fiberTraceSet.size(); ++i){
        PTR(FiberTrace<ImageT, MaskT, VarianceT>) tra(new pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>(*(fiberTraceSet.getFiberTrace(i)), true));
//        _traces[i].reset();
        (*_traces)[i] = tra;
      }
    }
  }

      
  /// Extract FiberTraces from new MaskedImage
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::createTraces(const PTR(const MaskedImageT) &maskedImage){
    for (int i = 0; i < _traces->size(); ++i){
      if (!(*_traces)[i]->createTrace(maskedImage)){
        string message("FiberTraceSet::createTraces: ERROR: _traces[");
        message += to_string(i) + string("] returned FALSE");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    }
    return true;
  }
 
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::setFiberTrace(const size_t i,     ///< which aperture?
                                                                            const PTR(FiberTrace<ImageT, MaskT, VarianceT>) &trace ///< the FiberTrace for the ith aperture
  ){
    if (i > static_cast<int>(_traces->size())){
      string message("FiberTraceSet::setFiberTrace: ERROR: position for trace outside range!");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    if (i == static_cast<int>(_traces->size())){
      _traces->push_back(trace);
    }
    else{
      (*_traces)[i] = trace;
    }
    return true;
  }
  
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>)& pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::getFiberTrace(const size_t i){
    if (i >= _traces->size()){
      string message("FiberTraceSet::getFiberTrace(i=");
      message += to_string(i) + string("): ERROR: i > _traces->size()=") + to_string(_traces->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return _traces->at(i); 
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>) const& pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::getFiberTrace(const size_t i) const { 
    if (i >= _traces->size()){
      string message("FiberTraceSet::getFiberTrace(i=");
      message += to_string(i) + "): ERROR: i > _traces->size()=" + to_string(_traces->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return _traces->at(i); 
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::erase(const size_t iStart, const size_t iEnd){
    if (iStart >= _traces->size()){
      string message("FiberTraceSet::erase(iStart=");
      message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iStart >= _traces->size()=" + to_string(_traces->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (iEnd >= _traces->size()){
      string message("FiberTraceSet::erase(iStart=");
      message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iEnd >= _traces->size()=" + to_string(_traces->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (iEnd > 0){
      if (iStart > iEnd){
        string message("FiberTraceSet::erase(iStart=");
        message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iStart > iEnd";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    }
    if (iStart == (_traces->size()-1)){
      _traces->pop_back();
    }
    else{
      if (iEnd == 0)
        _traces->erase(_traces->begin() + iStart);
      else
        _traces->erase(_traces->begin() + iStart, _traces->begin() + iEnd);
    }
    return true;
  }
  
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::addFiberTrace(const PTR(FiberTrace<ImageT, MaskT, VarianceT>) &trace, const size_t iTrace) ///< the FiberTrace for the ith aperture
  {
    int size = _traces->size();
    _traces->push_back(trace);
    if (_traces->size() == size){
      string message("FiberTraceSet::addFiberTrace: ERROR: could not add trace to _traces");
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (iTrace > 0){
      (*_traces)[size]->setITrace(iTrace);
      cout << "FiberTraceSet::addFiberTrace: (*_traces)[" << size << "]->_iTrace set to " << (*_traces)[size]->getITrace() << endl;
    }
    return true;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  void pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::sortTracesByXCenter()
  {
    for (int i = 0; i < static_cast<int>(_traces->size()); ++i){
      cout << "FiberTraceSet::sortTracesByXCenter: (*_traces)[" << i << "]->getFiberTraceFunction()->xCenter = " << (*_traces)[i]->getFiberTraceFunction()->xCenter << endl;
    }
    std::vector<float> xCenters;
    for (int iTrace = 0; iTrace < static_cast<int>(_traces->size()); ++iTrace){
      xCenters.push_back((*_traces)[iTrace]->getFiberTraceFunction()->xCenter);
    }
    std::vector<int> sortedIndices(xCenters.size());
    sortedIndices = ::pfs::drp::stella::math::sortIndices(xCenters);
    #ifdef __DEBUG_SORTTRACESBYXCENTER__
      for (int iTrace = 0; iTrace < static_cast<int>(_traces->size()); ++iTrace)
        cout << "FiberTraceSet::sortTracesByXCenter: sortedIndices(" << iTrace << ") = " << sortedIndices[iTrace] << endl;
    #endif
    
    std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)> sortedTraces(_traces->size());
    for (int i = 0; i < static_cast<int>(_traces->size()); ++i){
      sortedTraces[i] = (*_traces)[sortedIndices[i]];
      sortedTraces[i]->setITrace(i);
    }
    _traces.reset(new std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>(sortedTraces));
    return;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::setFiberTraceProfileFittingControl(PTR(FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl){
    for (unsigned int i=0; i<_traces->size(); ++i){
      if (!(*_traces)[i]->setFiberTraceProfileFittingControl(fiberTraceProfileFittingControl)){
        string message("FiberTraceSet::setFiberTraceProfileFittingControl: ERROR: (*_traces)[");
        message += to_string(i) + "]->setFiberTraceProfileFittingControl(fiberTraceProfileFittingControl) returned FALSE";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
    }
    return true;
  }
/*
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::extractTraceNumber(const size_t traceNumber)
  {
    return (*_traces)[traceNumber]->MkSlitFunc();
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, VarianceT>) pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::extractAllTraces()
  {
    PTR(pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, VarianceT>) spectrumSet(new pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, VarianceT>(_traces->size()));
    blitz::Array<std::string, 1> keyWords(1);
    for (size_t i = 0; i < _traces->size(); ++i){
      (*(spectrumSet->getSpectra()))[i] = (*_traces)[i]->MkSlitFunc();//keyWords, args);
    }
    return spectrumSet;
  }
*/
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::extractTraceNumberFromProfile(const size_t traceNumber)
  {
    return (*_traces)[traceNumber]->extractFromProfile();
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, VarianceT>)  pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::extractAllTracesFromProfile()
  {
    PTR(pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, VarianceT>) spectrumSet(new pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, VarianceT>(_traces->size()));
    for (size_t i = 0; i < _traces->size(); ++i){
      cout << "extracting FiberTrace " << i << endl;
      (*(spectrumSet->getSpectra()))[i] = (*_traces)[i]->extractFromProfile();
    }
    return spectrumSet;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::setAllProfiles(const PTR(FiberTraceSet<ImageT, MaskT, VarianceT>) &fiberTraceSet){
    for (size_t i = 0; i < _traces->size(); ++i){
      if (!(*_traces)[i]->setProfile(fiberTraceSet->getFiberTrace(i)->getProfile())){
        string message("FiberTraceSet::copyAllProfiles: ERROR: (*_traces)[");
        message += to_string(i) + "].setProfile(fiberTraceSet->getFiberTrace(" + to_string(i) + ")->getProfile()) returned FALSE";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
    }
    return true;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::calcProfileAllTraces(){
    for (size_t i = 0; i < _traces->size(); ++i){
      if (!(*_traces)[i]->calcProfile()){
        string message("FiberTraceSet::copyAllProfiles: ERROR: (*_traces)[");
        message += to_string(i) + "].calcProfile() returned FALSE";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
    }
    return true;
  }

  namespace pfs { namespace drp { namespace stella { namespace math {
    template<typename ImageT, typename MaskT, typename VarianceT>
    PTR(pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>) findAndTraceApertures(const PTR(const afwImage::MaskedImage<ImageT, MaskT, VarianceT>) &maskedImage,
                                                                                     const PTR(const pfsDRPStella::FiberTraceFunctionFindingControl) &fiberTraceFunctionFindingControl){
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "::pfs::drp::stella::math::findAndTraceApertures started" << endl;
      #endif
      if (static_cast<int>(fiberTraceFunctionFindingControl->apertureFWHM * 2.) + 1 <= fiberTraceFunctionFindingControl->nTermsGaussFit){
        cout << "::pfs::drp::stella::math::findAndTraceApertures: WARNING: fiberTraceFunctionFindingControl->apertureFWHM too small for GaussFit -> Try lower fiberTraceFunctionFindingControl->nTermsGaussFit!" << endl;
        exit(EXIT_FAILURE);
      }
      PTR(pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>) fiberTraceSet(new pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>());
      pfsDRPStella::FiberTraceFunction fiberTraceFunction;
      fiberTraceFunction.fiberTraceFunctionControl = fiberTraceFunctionFindingControl->fiberTraceFunctionControl;
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.fiberTraceFunctionControl set" << endl;
      #endif

      int I_Aperture = 0;
      int I_StartIndex;
      int I_FirstWideSignal;
      int I_FirstWideSignalEnd;
      int I_FirstWideSignalStart;
      unsigned int I_Length, I_ApertureLost, I_Row_Bak;//, I_LastRowWhereApertureWasFound
      int I_ApertureLength;
      int I_NInd;
      double D_Max;
      bool B_ApertureFound;
      ndarray::Array<ImageT, 2, 2> ccdImage = ndarray::copy(maskedImage->getImage()->getArray());
      ndarray::Array<ImageT, 1, 1> D_A1_IndexCol = pfsDRPStella::math::indGenNdArr(ImageT(ccdImage.getShape()[1]));
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "::pfs::drp::stella::math::findAndTraceApertures: D_A1_IndexCol = " << D_A1_IndexCol << endl;
      #endif
//      ndarray::Array<VarianceT, 1, 1> D_A1_MeasureErrors = ndarray::allocate(ccdImage.getShape()[1]);
      ndarray::Array<ImageT, 1, 1> D_A1_Guess = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit);
      ndarray::Array<ImageT, 1, 1> D_A1_GaussFit_Coeffs = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit);
      ndarray::Array<ImageT, 1, 1> D_A1_GaussFit_Coeffs_Bak = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit);
      ndarray::Array<int, 1, 1> I_A1_Signal = ndarray::allocate(maskedImage->getWidth());
      I_A1_Signal[ndarray::view()] = 0;
      ndarray::Array<float, 1, 1> D_A1_ApertureCenter = ndarray::allocate(maskedImage->getHeight());
      ndarray::Array<int, 1, 1> I_A1_ApertureCenterIndex = ndarray::allocate(maskedImage->getHeight());
      ndarray::Array<double, 1, 1> D_A1_PolyFitCoeffs = ndarray::allocate(fiberTraceFunctionFindingControl->fiberTraceFunctionControl.order);
      ndarray::Array<int, 1, 1> I_A1_IndSignal;
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "::pfs::drp::stella::math::findAndTraceApertures: started" << endl;
      #endif
      ndarray::Array<int, 1, 1> I_A1_Ind;
      ndarray::Array<int, 1, 1> I_A1_Where;
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunctionFindingControl->signalThreshold = " << fiberTraceFunctionFindingControl->signalThreshold << endl;
      #endif

      /// Set all pixels below fiberTraceFunctionFindingControl->signalThreshold to 0.
      for (auto i = ccdImage.begin(); i != ccdImage.end(); ++i){
        for (auto j = i->begin(); j != i->end(); ++j){
          if (*j < fiberTraceFunctionFindingControl->signalThreshold){
            *j = 0;
          }
        }
      }
      #ifdef __DEBUG_FINDANDTRACE__
        maskedImage->getImage()->getArray()[0,0] = 3.;
        cout << "ccdImage[0][0] = " << ccdImage[0][0] << endl;
        cout << "ccdImage = " << ccdImage << endl;
      #endif
      int I_MinWidth = int(1.5 * fiberTraceFunctionFindingControl->apertureFWHM);
      if (I_MinWidth < fiberTraceFunctionFindingControl->nTermsGaussFit)
        I_MinWidth = fiberTraceFunctionFindingControl->nTermsGaussFit;
      double D_MaxTimesApertureWidth = 4.;

      /// Search for Apertures
      D_A1_ApertureCenter[ndarray::view()] = 0.;
      I_A1_ApertureCenterIndex[ndarray::view()] = 0;
      auto itIm = ccdImage.begin();
      for (int i_Row = 0; i_Row < maskedImage->getHeight(); i_Row++){
        I_StartIndex = 0;
        B_ApertureFound = false;
        for (int i_Col = 0; i_Col < maskedImage->getWidth(); ++i_Col){
          if (i_Col == 0){
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "i_Row = " << i_Row << ": i_Col == 0: ccdImage[i_Row][i_Col] = " << ccdImage[i_Row][i_Col] << endl;
            #endif
            if (ccdImage[i_Row][i_Col] > 0.){
              I_A1_Signal[i_Col] = 1;
            }
            else{
              I_A1_Signal[i_Col] = 0;
            }
          }
          else{
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "i_Row = " << i_Row << ": i_Col = " << i_Col << ": ccdImage[" << i_Row << "][" << i_Col << "] = " << ccdImage[i_Row][i_Col] << ", I_A1_Signal[" << i_Col-1 << "] = " << I_A1_Signal[i_Col - 1] << endl;
            #endif
            if (ccdImage[i_Row][i_Col] > 0.){
              if (I_A1_Signal[i_Col - 1] > 0){
                I_A1_Signal[i_Col] = I_A1_Signal[i_Col - 1] + 1;
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "i_Row = " << i_Row << ": i_Col = " << i_Col << " => I_A1_Signal[" << i_Col << "] set to " << I_A1_Signal[i_Col] << endl;
                #endif
              }
              else{
                I_A1_Signal[i_Col] = 1;
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "i_Row = " << i_Row << ": i_Col = " << i_Col << " => I_A1_Signal[" << i_Col << "] set to " << I_A1_Signal[i_Col] << endl;
                #endif
              }
            }
          }
        }
/*        auto itImRow = itIm->begin();
        int pos = 0;
        for (auto it = I_A1_Signal.begin(); it != I_A1_Signal.end(); ++it){
          if (pos == 0){
  //          cout << "i_Row = " << i_Row << ": pos == 0: *itImRow + " << pos << " = " << *(itImRow + pos) << endl;
            if ((*(itImRow + pos)) > 0.){
              *it  = 1;
            }
            else
              *it = 0;
          }
          else{
//            cout << "i_Row = " << i_Row << ": pos = " << pos << ": *itImRow + " << pos << " = " << *(itImRow + pos) << endl;
            if ((*(itImRow+pos)) > 0.){
              if (*(it-1) > 0){
                *it = (*(it-1)) + 1;
              }
              else
                *it = 1;
              cout << "i_Row = " << i_Row << ": *(itImRow + " << pos << ") = " << *(itImRow + pos) << " => *it set to " << *it << endl;
            }
            else
              *it = 0;
          }
          ++pos;
        }*/
        #ifdef __DEBUG_FINDANDTRACE__
          cout << "pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_A1_Signal = " << I_A1_Signal << endl;
        #endif
        while (!B_ApertureFound){
          I_FirstWideSignal = pfsDRPStella::math::firstIndexWithValueGEFrom(I_A1_Signal, I_MinWidth, I_StartIndex);
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_FirstWideSignal found at index " << I_FirstWideSignal << ", I_StartIndex = " << I_StartIndex << endl;
          #endif
          if (I_FirstWideSignal < 0){
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": No Aperture found in row " << i_Row << ", trying next row" << endl;
            #endif
            break;
          }
          else{
            I_FirstWideSignalStart = ::pfs::drp::stella::math::lastIndexWithZeroValueBefore(I_A1_Signal, I_FirstWideSignal) + 1;
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "::pfs::drp::stella::math::findAndTraceApertures: while: 1. i_Row = " << i_Row << ": I_FirstWideSignalStart = " << I_FirstWideSignalStart << endl;
            #endif

            I_FirstWideSignalEnd = pfsDRPStella::math::firstIndexWithZeroValueFrom(I_A1_Signal, I_FirstWideSignal) - 1;
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
            #endif

            if (I_FirstWideSignalStart < 0){
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": WARNING: No start of aperture found -> Going to next Aperture." << endl;
              #endif

              if (I_FirstWideSignalEnd < 0){
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": 1. WARNING: No end of aperture found -> Going to next row." << endl;
                #endif
                break;
              }
              else{
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": End of first wide signal found at index " << I_FirstWideSignalEnd << endl;
                #endif
                /// Set start index for next run
                I_StartIndex = I_FirstWideSignalEnd+1;
              }
            }
            else{ /// Fit Gaussian and Trace Aperture
              if (I_FirstWideSignalEnd < 0){
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": 2. WARNING: No end of aperture found -> Going to next row." << endl;
                  cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_Row_Bak = " << I_Row_Bak << endl;
                  cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": B_ApertureFound = " << B_ApertureFound << endl;
                #endif
                break;
              }
              else{
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": End of first wide signal found at index " << I_FirstWideSignalEnd << endl;
                #endif

                if (I_FirstWideSignalEnd - I_FirstWideSignalStart + 1 > fiberTraceFunctionFindingControl->apertureFWHM * D_MaxTimesApertureWidth){
                  I_FirstWideSignalEnd = I_FirstWideSignalStart + int(D_MaxTimesApertureWidth * fiberTraceFunctionFindingControl->apertureFWHM);
                }
                I_A1_Signal[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)] = 0;

                /// Set start index for next run
                I_StartIndex = I_FirstWideSignalEnd+1;
              }
              I_Length = I_FirstWideSignalEnd - I_FirstWideSignalStart + 1;

              if (fiberTraceFunctionFindingControl->nTermsGaussFit == 0){/// look for maximum only
                D_A1_ApertureCenter[ndarray::view()] = 0.;
                B_ApertureFound = true;
                int maxPos = 0;
                ImageT tMax = 0.;
                for (int i = I_FirstWideSignalStart; i <= I_FirstWideSignalEnd; ++i){
                  if (ccdImage[i_Row][i] > tMax){
                    tMax = ccdImage[i_Row][i];
                    maxPos = i;
                  }
                }
                D_A1_ApertureCenter[i_Row] = maxPos;
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": Aperture found at " << D_A1_ApertureCenter[i_Row] << endl;
                #endif

                /// Set signal to zero
                ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
              }
              else{
                if (I_Length <= fiberTraceFunctionFindingControl->nTermsGaussFit){
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": WARNING: Width of aperture <= " << fiberTraceFunctionFindingControl->nTermsGaussFit << "-> abandoning aperture" << endl;
                  #endif

                  /// Set signal to zero
                  ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                }
                else{
                  /// populate Arrays for GaussFit
                  ndarray::Array<ImageT, 1, 1> D_A1_X = copy(D_A1_IndexCol[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)]);
                  ndarray::Array<ImageT, 1, 1> D_A1_Y = copy(ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)]);
                  for (auto it = D_A1_Y.begin(); it != D_A1_Y.end(); ++it){
                    if (*it < 0.000001)
                      *it = 1.;
                  }
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: 1. D_A1_Y set to " << D_A1_Y << endl;
                  #endif
                  ndarray::Array<VarianceT, 1, 1> T_A1_MeasureErrors = copy(maskedImage->getVariance()->getArray()[ndarray::view(i_Row)(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)]);
                  ndarray::Array<ImageT, 1, 1> D_A1_MeasureErrors = ndarray::allocate(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
                  for (int ooo = 0; ooo < I_FirstWideSignalEnd - I_FirstWideSignalStart + 1; ++ooo){
                    if (T_A1_MeasureErrors[ooo] > 0)
                      D_A1_MeasureErrors[ooo] = ImageT(sqrt(T_A1_MeasureErrors[ooo]));
                    else
                      D_A1_MeasureErrors[ooo] = 1;
                  }
/*                  for (std::pair<itT, itD> i(T_A1_MeasureErrors.begin(), D_A1_MeasureErrors.begin()); i.first != T_A1_MeasureErrors.end(); ++itT, ++itD){
                    if (*itT > 0)
                      *itD = ImageT(sqrt(*itT));
                    else
                      *itD = 1;
                  }*/

                  /// Guess values for GaussFit
                  if (fiberTraceFunctionFindingControl->nTermsGaussFit == 3){
                    D_A1_Guess[0] = math::max(D_A1_Y);
                    D_A1_Guess[1] = double(I_FirstWideSignalStart) + (double((I_FirstWideSignalEnd - I_FirstWideSignalStart)) / 2.);
                    D_A1_Guess[2] = double(fiberTraceFunctionFindingControl->apertureFWHM) / 2.;
                  }
                  else if (fiberTraceFunctionFindingControl->nTermsGaussFit > 3){
                    D_A1_Guess[3] = std::min(D_A1_Y[0], D_A1_Y[D_A1_Y.getShape()[0]-1]);
                    if (D_A1_Guess[3] < 0.)
                      D_A1_Guess[3] = 0.1;
                    if (fiberTraceFunctionFindingControl->nTermsGaussFit > 4)
                      D_A1_Guess[4] = (D_A1_Y[D_A1_Y.getShape()[0]-1] - D_A1_Y[0]) / (D_A1_Y.getShape()[0] - 1);
                  }

                  D_A1_GaussFit_Coeffs[ndarray::view()] = 0.;
                  ndarray::Array<ImageT, 1, 1> D_A1_GaussFit_ECoeffs = ndarray::allocate(D_A1_GaussFit_Coeffs.size());
                  D_A1_GaussFit_ECoeffs[ndarray::view()] = 0.;

                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": D_A1_X = " << D_A1_X << ", D_A1_Y = " << D_A1_Y << endl;
                  #endif

                  ndarray::Array<int, 2, 2> I_A2_Limited = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit, 2);
                  I_A2_Limited[ndarray::view()] = 1;
                  ndarray::Array<ImageT, 2, 2> D_A2_Limits = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit, 2);
                  D_A2_Limits[0][0] = 0.;/// Peak lower limit
                  D_A2_Limits[0][1] = 2. * D_A1_Guess[0];/// Peak upper limit
                  D_A2_Limits[1][0] = double(I_FirstWideSignalStart);/// Centroid lower limit
                  D_A2_Limits[1][1] = double(I_FirstWideSignalEnd);/// Centroid upper limit
                  D_A2_Limits[2][0] = double(fiberTraceFunctionFindingControl->apertureFWHM) / 4.;/// Sigma lower limit
                  D_A2_Limits[2][1] = double(fiberTraceFunctionFindingControl->apertureFWHM);/// Sigma upper limit
                  if (fiberTraceFunctionFindingControl->nTermsGaussFit > 3){
                    D_A2_Limits[3][0] = 0.;
                    D_A2_Limits[3][1] = 2. * D_A1_Guess[3];
                    if (fiberTraceFunctionFindingControl->nTermsGaussFit > 4){
                      D_A2_Limits[4][0] = D_A1_Guess[4] / 10.;
                      D_A2_Limits[4][1] = D_A1_Guess[4] * 10.;
                    }
                  }
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": 1. starting MPFitGaussLim: D_A1_Guess = " << D_A1_Guess << ", I_A2_Limited = " << I_A2_Limited << ", D_A2_Limits = " << D_A2_Limits << endl;
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
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": WARNING: GaussFit FAILED -> abandoning aperture" << endl;
                    #endif

                    /// Set start index for next run
                    I_StartIndex = I_FirstWideSignalEnd+1;

                    ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                  }
                  else{
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
                      if (D_A1_GaussFit_Coeffs[0] < fiberTraceFunctionFindingControl->saturationLevel/5.){
                        cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": WARNING: Signal less than 20% of saturation level" << endl;
                      }
                      if (D_A1_GaussFit_Coeffs[0] > fiberTraceFunctionFindingControl->saturationLevel){
                        cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": WARNING: Signal appears to be saturated" << endl;
                      }
                      if ((D_A1_GaussFit_Coeffs[1] < double(I_FirstWideSignalStart) + (double(I_Length)/4.)) || (D_A1_GaussFit_Coeffs[1] > double(I_FirstWideSignalStart) + (double(I_Length) * 3./4.))){
                        cout << "::pfs::drp::stella::math::findAndTraceApertures: while: Warning: i_Row = " << i_Row << ": Center of Gaussian far away from middle of signal" << endl;
                      }
                    #endif
                    if ((D_A1_GaussFit_Coeffs[1] < double(I_FirstWideSignalStart)) || (D_A1_GaussFit_Coeffs[1] > double(I_FirstWideSignalEnd))){
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                      #endif
                      /// Set signal to zero
                      ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;

                      /// Set start index for next run
                      I_StartIndex = I_FirstWideSignalEnd+1;
                    }
                    else{
                      if ((D_A1_GaussFit_Coeffs[2] < fiberTraceFunctionFindingControl->apertureFWHM / 4.) || (D_A1_GaussFit_Coeffs[2] > fiberTraceFunctionFindingControl->apertureFWHM)){
                        #ifdef __DEBUG_FINDANDTRACE__
                          cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": WARNING: FWHM = " << D_A1_GaussFit_Coeffs[2] << " outside range -> abandoning aperture" << endl;
                        #endif
                        /// Set signal to zero
                        ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                        #ifdef __DEBUG_FINDANDTRACE__
                          cout << "::pfs::drp::stella::math::findAndTraceApertures: while: B_ApertureFound = " << B_ApertureFound << ": 1. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                          cout << "::pfs::drp::stella::math::findAndTraceApertures: while: 1. ccdImage(i_Row = " << i_Row << ", Range(I_FirstWideSignalStart = " << I_FirstWideSignalStart << ", I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << ")) set to " << ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart, I_FirstWideSignalEnd)] << endl;
                        #endif
                        /// Set start index for next run
                        I_StartIndex = I_FirstWideSignalEnd+1;
                      }
                      else{
                        D_A1_ApertureCenter[ndarray::view()] = 0.;
                        B_ApertureFound = true;
                        //I_LastRowWhereApertureWasFound = i_Row;
                        D_A1_ApertureCenter[i_Row] = D_A1_GaussFit_Coeffs[1];
                        #ifdef __DEBUG_FINDANDTRACE__
                          cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": Aperture found at " << D_A1_ApertureCenter[i_Row] << endl;
                        #endif
                        ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                      }
                    }/// end else if ((D_A1_GaussFit_Coeffs(1) > double(I_FirstWideSignalStart)) && (D_A1_GaussFit_Coeffs(1) < double(I_FirstWideSignalEnd)))
                  }/// else else if (GaussFit returned TRUE)
                }/// end else if (I_Length >= 4)
              }/// end else if GaussFit
            }/// end else if (I_FirstWideSignalStart > 0)
          }/// end if (I_FirstWideSignal > 0)
        }/// end while (!B_ApertureFound)

        if (B_ApertureFound){
          /// Trace Aperture
          I_Length = 1;
          I_ApertureLost = 0;
  //        #ifdef __DEBUG_FINDANDTRACE__
            cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": Starting to trace aperture " << I_Aperture << endl;
  //        #endif
          D_A1_GaussFit_Coeffs_Bak[ndarray::view()].deep() = D_A1_GaussFit_Coeffs;
          I_Row_Bak = i_Row;
          while(B_ApertureFound && (I_ApertureLost < fiberTraceFunctionFindingControl->nLost) && (i_Row < maskedImage->getHeight()-1) && I_Length < fiberTraceFunctionFindingControl->maxLength){
            i_Row++;
            I_Length++;
            if (fiberTraceFunctionFindingControl->nTermsGaussFit == 0){/// look for maximum only
              B_ApertureFound = true;
              int maxPos = 0;
              ImageT tMax = 0.;
              for (int i = I_FirstWideSignalStart; i <= I_FirstWideSignalEnd; ++i){
                if (ccdImage[i_Row][i] > tMax){
                  tMax = ccdImage[i_Row][i];
                  maxPos = i;
                }
              }
              if (tMax < fiberTraceFunctionFindingControl->signalThreshold){
                I_ApertureLost++;
              }
              else{
                D_A1_ApertureCenter[i_Row] = maxPos;
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": Aperture found at " << D_A1_ApertureCenter[i_Row] << endl;
                #endif
                if (D_A1_ApertureCenter[i_Row] < D_A1_ApertureCenter[i_Row-1]){
                  I_FirstWideSignalStart--;
                  I_FirstWideSignalEnd--;
                }
                if (D_A1_ApertureCenter[i_Row] > D_A1_ApertureCenter[i_Row-1]){
                  I_FirstWideSignalStart++;
                  I_FirstWideSignalEnd++;
                }
                ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
              }
            }
            else{
              I_FirstWideSignalStart = int(D_A1_GaussFit_Coeffs_Bak[1] - 1.6 * D_A1_GaussFit_Coeffs_Bak[2]);
              I_FirstWideSignalEnd = int(D_A1_GaussFit_Coeffs_Bak[1] + 1.6 * D_A1_GaussFit_Coeffs_Bak[2]) + 1;
              if (I_FirstWideSignalStart < 0. || I_FirstWideSignalEnd >= maskedImage->getWidth()){
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": start or end of aperture outside CCD -> Aperture lost" << endl;
                #endif
                /// Set signal to zero
                if (I_FirstWideSignalStart < 0)
                  I_FirstWideSignalStart = 0;
                if (I_FirstWideSignalEnd >= maskedImage->getWidth())
                  I_FirstWideSignalEnd = maskedImage->getWidth() - 1;
                ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                I_ApertureLost++;
              }
              else{
                I_Length = I_FirstWideSignalEnd - I_FirstWideSignalStart + 1;

                if (I_Length <= fiberTraceFunctionFindingControl->nTermsGaussFit){
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": Warning: Width of Aperture <= " << fiberTraceFunctionFindingControl->nTermsGaussFit << " -> Lost Aperture" << endl;
                  #endif
                  /// Set signal to zero
                  ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                  I_ApertureLost++;
                }
                else{
                  ndarray::Array<ImageT, 1, 1> D_A1_X = copy(D_A1_IndexCol[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd+1)]);
                  ndarray::Array<ImageT, 1, 1> D_A1_Y = copy(ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart, I_FirstWideSignalEnd+1)]);
                  ndarray::Array<VarianceT, 1, 1> T_A1_MeasureErrors = copy(maskedImage->getVariance()->getArray()[ndarray::view(i_Row)(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)]);
                  ndarray::Array<ImageT, 1, 1> D_A1_MeasureErrors = ndarray::allocate(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
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
                    ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                    I_ApertureLost++;
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "::pfs::drp::stella::math::findAndTraceApertures: Signal not wide enough => Aperture lost" << endl;
                    #endif
                  }
                  else{
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "::pfs::drp::stella::math::findAndTraceApertures: 2. D_A1_Y set to " << D_A1_Y << endl;
                    #endif
                    for (auto it = D_A1_MeasureErrors.begin(); it != D_A1_MeasureErrors.end(); ++it){
                      if (*it > 0)
                        *it = sqrt(*it);
                      else
                        *it = 1;
                    }
                    D_A1_Guess.deep() = D_A1_GaussFit_Coeffs_Bak;

                    D_A1_GaussFit_Coeffs[ndarray::view()] = 0.;
                    ndarray::Array<ImageT, 1, 1> D_A1_GaussFit_ECoeffs = ndarray::allocate(D_A1_GaussFit_Coeffs.size());
                    D_A1_GaussFit_ECoeffs[ndarray::view()] = 0.;

                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": D_A1_X = " << D_A1_X << ", D_A1_Y = " << D_A1_Y << endl;
                    #endif

                    ndarray::Array<int, 2, 2> I_A2_Limited = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit, 2);
                    I_A2_Limited[ndarray::view()] = 1;
                    ndarray::Array<ImageT, 2, 2> D_A2_Limits = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit, 2);
                    D_A2_Limits[0][0] = 0.;/// Peak lower limit
                    D_A2_Limits[0][1] = 2. * D_A1_Guess[0];/// Peak upper limit
                    D_A2_Limits[1][0] = ImageT(I_FirstWideSignalStart);/// Centroid lower limit
                    D_A2_Limits[1][1] = ImageT(I_FirstWideSignalEnd);/// Centroid upper limit
                    D_A2_Limits[2][0] = ImageT(fiberTraceFunctionFindingControl->apertureFWHM) / 4.;/// Sigma lower limit
                    D_A2_Limits[2][1] = ImageT(fiberTraceFunctionFindingControl->apertureFWHM);/// Sigma upper limit
                    if (fiberTraceFunctionFindingControl->nTermsGaussFit > 3){
                      D_A2_Limits[3][0] = 0.;
                      D_A2_Limits[3][1] = 2. * D_A1_Guess[3];
                      if (fiberTraceFunctionFindingControl->nTermsGaussFit > 4){
                        D_A2_Limits[4][0] = D_A1_Guess[4] / 10.;
                        D_A2_Limits[4][1] = D_A1_Guess[4] * 10.;
                      }
                    }
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": 2. starting MPFitGaussLim: D_A2_Limits = " << D_A2_Limits << endl;
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
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": Warning: GaussFit FAILED" << endl;
                      #endif
                      /// Set signal to zero
                      ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart + 1, I_FirstWideSignalEnd)] = 0.;

                      I_ApertureLost++;
                    }
                    else{
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
                        if (D_A1_GaussFit_Coeffs[0] < fiberTraceFunctionFindingControl->saturationLevel/5.){
                          cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": WARNING: Signal less than 20% of saturation level" << endl;
                        }
                        if (D_A1_GaussFit_Coeffs[0] > fiberTraceFunctionFindingControl->saturationLevel){
                          cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": WARNING: Signal appears to be saturated" << endl;
                        }
                      #endif
                      if (D_A1_GaussFit_Coeffs[0] < fiberTraceFunctionFindingControl->signalThreshold){
                          #ifdef __DEBUG_FINDANDTRACE__
                            cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": WARNING: peak = " << D_A1_GaussFit_Coeffs[1] << " lower than signalThreshold -> abandoning aperture" << endl;
                          #endif
                          /// Set signal to zero
                          ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart + 1, I_FirstWideSignalEnd)] = 0.;
                          #ifdef __DEBUG_FINDANDTRACE__
                            cout << "::pfs::drp::stella::math::findAndTraceApertures: 2. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                          #endif
                          I_ApertureLost++;
                      }
                      else{
                        if ((D_A1_GaussFit_Coeffs[1] < D_A1_GaussFit_Coeffs_Bak[1] - 1.) || (D_A1_GaussFit_Coeffs[1] > D_A1_GaussFit_Coeffs_Bak[1] + 1.)){
                          #ifdef __DEBUG_FINDANDTRACE__
                            cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                          #endif
                          /// Set signal to zero
                          ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;

                          I_ApertureLost++;
                        }
                        else{
                          if ((D_A1_GaussFit_Coeffs[2] < fiberTraceFunctionFindingControl->apertureFWHM / 4.) || (D_A1_GaussFit_Coeffs[2] > fiberTraceFunctionFindingControl->apertureFWHM)){
                            #ifdef __DEBUG_FINDANDTRACE__
                              cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": WARNING: FWHM = " << D_A1_GaussFit_Coeffs[2] << " outside range -> abandoning aperture" << endl;
                            #endif
                            /// Set signal to zero
                            ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                            #ifdef __DEBUG_FINDANDTRACE__
                              cout << "::pfs::drp::stella::math::findAndTraceApertures: 2. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                            #endif
                            I_ApertureLost++;
                          }
                          else{
                            I_ApertureLost = 0;
                            B_ApertureFound = true;
                            D_A1_ApertureCenter[i_Row] = D_A1_GaussFit_Coeffs[1];
                            #ifdef __DEBUG_FINDANDTRACE__
                              cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": Aperture found at " << D_A1_ApertureCenter[i_Row] << endl;
                            #endif
                            D_A1_GaussFit_Coeffs_Bak[ndarray::view()] = D_A1_GaussFit_Coeffs[ndarray::view()];
                            //I_LastRowWhereApertureWasFound = i_Row;
                          }
                        }/// end else if ((D_A1_GaussFit_Coeffs(1) >= D_A1_Guess(1) - 1.) && (D_A1_GaussFit_Coeffs(1) <= D_A1_Guess(1) + 1.))
                      }/// end else if (D_A1_GaussFit_Coeffs(0) >= signalThreshold
                    }/// end else if (GaussFit(D_A1_X, D_A1_Y, D_A1_GaussFit_Coeffs, S_A1_KeyWords_GaussFit, PP_Args_GaussFit))
                    ccdImage[ndarray::view(i_Row)(I_FirstWideSignalStart + 1, I_FirstWideSignalEnd)] = 0.;
                  }/// end else if (sum(I_A1_Signal) >= I_MinWidth){
                }/// end if (I_Length > 3)
              }/// end else if (I_ApertureStart >= 0. && I_ApertureEnd < maskedImage->getWidth())
            }/// end else if GaussFit
          }///end while(B_ApertureFound && (I_ApertureLost < 3) && i_Row < maskedImage->getHeight() - 2))

          /// Fit Polynomial to traced aperture positions
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": D_A1_ApertureCenter = " << D_A1_ApertureCenter << endl;
  //          ::pfs::drp::stella::utils::WriteArrayToFile(D_A1_ApertureCenter, std::string("xCenters_before_polyfit_ap")+to_string(I_ApertureNumber)+std::string(".dat"), std::string("ascii"));
          #endif

          auto itInd = I_A1_ApertureCenterIndex.begin();
          auto itCen = D_A1_ApertureCenter.begin();
          I_ApertureLength = 0;
          std::vector<double> apertureCenterIndVec;
          apertureCenterIndVec.reserve(maskedImage->getHeight());
          std::vector<double> apertureCenterPosVec;
          apertureCenterPosVec.reserve(maskedImage->getHeight());
          for (int iInd = 0; iInd < maskedImage->getHeight(); ++iInd){
            if (*(itCen + iInd) > 0.){
              (*(itInd + iInd)) = 1;
              ++I_ApertureLength;
              apertureCenterIndVec.push_back(double(iInd));
              apertureCenterPosVec.push_back((*(itCen + iInd)) + 0.5);
            }
          }
          const ndarray::Array<double, 1, 1> D_A1_ApertureCenterIndex = ndarray::external(apertureCenterIndVec.data(), ndarray::makeVector(int(apertureCenterIndVec.size())), ndarray::makeVector(1));
          const ndarray::Array<double, 1, 1> D_A1_ApertureCenterPos = ndarray::external(apertureCenterPosVec.data(), ndarray::makeVector(int(apertureCenterPosVec.size())), ndarray::makeVector(1));

          if (I_ApertureLength >= int(fiberTraceFunctionFindingControl->minLength)){
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": D_A1_ApertureCenterIndex = " << D_A1_ApertureCenterIndex << endl;
              cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": D_A1_ApertureCenterPos = " << D_A1_ApertureCenterPos << endl;
            #endif

            /// Fit Polynomial
            ndarray::Array<double, 1, 1> D_A1_PolyFitCoeffs = pfsDRPStella::math::PolyFit(D_A1_ApertureCenterIndex,
                                                                                          D_A1_ApertureCenterPos,
                                                                                          fiberTraceFunctionFindingControl->fiberTraceFunctionControl.order);
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "D_A1_PolyFitCoeffs = " << D_A1_PolyFitCoeffs << endl;
            #endif
 
            fiberTraceFunction.xCenter = D_A1_ApertureCenterPos[int(D_A1_ApertureCenterIndex.size()/2.)];
            fiberTraceFunction.yCenter = D_A1_ApertureCenterIndex[int(D_A1_ApertureCenterIndex.size()/2.)];
            fiberTraceFunction.yHigh = D_A1_ApertureCenterIndex[int(D_A1_ApertureCenterIndex.size()-1)] - fiberTraceFunction.yCenter;
            fiberTraceFunction.yLow = D_A1_ApertureCenterIndex[0] - fiberTraceFunction.yCenter;
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "::pfs::drp::stella::math::findAndTraceApertures: D_A1_PolyFitCoeffs = " << D_A1_PolyFitCoeffs << endl;
            #endif
            fiberTraceFunction.coefficients.resize(D_A1_PolyFitCoeffs.getShape()[0]);
            for (int iter=0; iter < int(D_A1_PolyFitCoeffs.getShape()[0]); iter++)
              fiberTraceFunction.coefficients[iter] = D_A1_PolyFitCoeffs[iter];
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.xCenter = " << fiberTraceFunction.xCenter << endl;
              cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.yCenter = " << fiberTraceFunction.yCenter << endl;
              cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.yLow = " << fiberTraceFunction.yLow << endl;
              cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.yHigh = " << fiberTraceFunction.yHigh << endl;
            #endif
            PTR(const pfsDRPStella::FiberTraceFunction) fiberTraceFunctionPTR(new pfsDRPStella::FiberTraceFunction(fiberTraceFunction));

            ndarray::Array<double, 1, 1> D_A1_XCenters_Y = pfsDRPStella::math::indGenNdArr(double(maskedImage->getHeight()));
            ndarray::Array<double, 1, 1> D_A1_XCenters = pfsDRPStella::math::Poly(D_A1_XCenters_Y,
                                                                                  D_A1_PolyFitCoeffs);
            ndarray::Array<double, 1, 1> D_A1_XCentersTrace = ndarray::allocate(fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1);
            D_A1_XCentersTrace[ndarray::view()] = D_A1_XCenters[ndarray::view(fiberTraceFunction.yCenter + fiberTraceFunction.yLow, fiberTraceFunction.yCenter + fiberTraceFunction.yHigh + 1)];
            ndarray::Array<float, 1, 1> F_A1_XCenters = pfsDRPStella::math::Float(D_A1_XCentersTrace);
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": F_A1_XCenters = " << F_A1_XCenters << endl;
            #endif
            if (F_A1_XCenters.getShape()[0] != (fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1)){
              string message("FindAndTraceApertures: iTrace = ");
              message += to_string(fiberTraceSet->getTraces()->size()) + string(": 1. ERROR: xCenters.size(=");
              message += to_string(F_A1_XCenters.getShape()[0]) + string(") != (fiberTraceFunction.yHigh(=");
              message += to_string(fiberTraceFunction.yHigh) + string(") - fiberTraceFunction.yLow(=") + to_string(fiberTraceFunction.yLow);
              message += string(") + 1) = ") + to_string(fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1);
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
            PTR(pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>) fiberTrace(new pfsDRPStella::FiberTrace< ImageT, MaskT, VarianceT >(maskedImage, 
                                                                                                                                        fiberTraceFunctionPTR,
                                                                                                                                        F_A1_XCenters, 
                                                                                                                                        0));
            fiberTrace->setITrace(fiberTraceSet->getTraces()->size());
            if (fiberTrace->getXCenters().getShape()[0] != (fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1)){
              string message("FindAndTraceApertures: iTrace = ");
              message += to_string(fiberTraceSet->getTraces()->size()) + string(": 2. ERROR: fiberTrace->getXCenters()->size(=");
              message += to_string(fiberTrace->getXCenters().getShape()[0]) + string(") != (fiberTraceFunction.yHigh(=");
              message += to_string(fiberTraceFunction.yHigh) + string(") - fiberTraceFunction.yLow(=") + to_string(fiberTraceFunction.yLow);
              message += string(") + 1) = ") + to_string(fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1);
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
            if (fiberTrace->getTrace()->getHeight() != (fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1)){
              string message("FindAndTraceApertures: iTrace = ");
              message += to_string(fiberTraceSet->getTraces()->size()) + string(": ERROR: fiberTrace->getTrace()->getHeight(=");
              message += to_string(fiberTrace->getTrace()->getHeight()) + string(")!= (fiberTraceFunction.yHigh(=");
              message += to_string(fiberTraceFunction.yHigh) + string(") - fiberTraceFunction.yLow(=") + to_string(fiberTraceFunction.yLow);
              message += string(") + 1) = ") + to_string(fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1);
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
            fiberTraceSet->addFiberTrace(fiberTrace);
            cout << "FindAndTraceApertures: aperture number " << I_Aperture << " added to fiberTraceSet" << endl;
            ++I_Aperture;
          }// end if (I_ApertureLength >= int(fiberTraceFunctionFindingControl->minLength))
          i_Row = I_Row_Bak - 1;
        }/// end if (B_ApertureFound)
      }/// end for(i_Row = 0; i_Row < maskedImage->getHeight(); i_Row++)

      fiberTraceSet->sortTracesByXCenter();
      return fiberTraceSet;
    }

    /** *******************************************************************************************************/

    /// Calculate the x-centers of the fiber trace
//    template<typename ImageT, typename MaskT, typename VarianceT>
    ndarray::Array<float, 1, 1> calculateXCenters(PTR(const pfsDRPStella::FiberTraceFunction) const& fiberTraceFunction,
                                                  size_t const& ccdHeight,
                                                  size_t const& ccdWidth){

      int cIndex = 0;
//      double D_YMin = 0.;
//      double D_YMax = 0.;

      ndarray::Array<float, 1, 1> xCenters;// = ndarray::allocate(ccdHeight);
//      xCenters[ndarray::view()] = 0.;

      PTR(pfsDRPStella::FiberTraceFunction) pFTF = const_pointer_cast<pfsDRPStella::FiberTraceFunction>(fiberTraceFunction);
      const ndarray::Array<double, 1, 1> fiberTraceFunctionCoefficients = ndarray::external(pFTF->coefficients.data(), ndarray::makeVector(int(fiberTraceFunction->coefficients.size())), ndarray::makeVector(1));

      #ifdef __DEBUG_XCENTERS__
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunctionCoefficients = " << fiberTraceFunctionCoefficients << endl;
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunction->fiberTraceFunctionControl.interpolation = " << fiberTraceFunction->fiberTraceFunctionControl.interpolation << endl;
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunction->fiberTraceFunctionControl.order = " << fiberTraceFunction->fiberTraceFunctionControl.order << endl;
      #endif
/*      if (fiberTraceFunction->fiberTraceFunctionControl.interpolation.compare("LEGENDRE") == 0){
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::calculateXCenters: Function = LEGENDRE" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
        #endif
        if (!::pfs::drp::stella::math::Legendre(xCenters,
                                          D_YMin,
                                          D_YMax,
                                          fiberTraceFunctionCoefficients,
                                          double(fiberTraceFunction->xCenter)+1.,
                                          double(fiberTraceFunction->yCenter)+1.,
                                          double(fiberTraceFunction->yCenter + fiberTraceFunction->yLow + 1),//1.
                                          double(int(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh)+1),//static_cast<double>(ccdHeight)
                                          double(fiberTraceFunction->fiberTraceFunctionControl.xLow),
                                          double(fiberTraceFunction->fiberTraceFunctionControl.xHigh),
                                          fiberTraceFunction->fiberTraceFunctionControl.order,
                                          int(ccdWidth),
                                          int(ccdHeight))){
          cout << "pfs::drp::stella::calculateXCenters: yCenter = " << fiberTraceFunction->yCenter << endl;
          cout << "pfs::drp::stella::calculateXCenters: yLow = " << fiberTraceFunction->yLow << endl;
          cout << "pfs::drp::stella::calculateXCenters: yHigh = " << fiberTraceFunction->yHigh << endl;
          std::string message("pfs::drp::stella::calculateXCenters: ERROR: Legendre(...) returned FALSE!!!");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::calculateXCenters: Legendre: D_YMin set to " << D_YMin << endl;
          cout << "pfs::drp::stella::calculateXCenters: Legendre: D_YMax set to " << D_YMax << endl;
        #endif

        xCenters -= 1.;
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::calculateXCenters: Legendre: xCenters set to " << xCenters << endl;
        #endif
      }
      else if (fiberTraceFunction->fiberTraceFunctionControl.interpolation.compare("CHEBYSHEV") == 0)
      {
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::calculateXCenters: Function = Chebyshev" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
        #endif
        if (!::pfs::drp::stella::math::Chebyshev(xCenters,
                             D_YMin,
                             D_YMax,
                             fiberTraceFunctionCoefficients,
                             double(fiberTraceFunction->xCenter)+1.,
                             double(fiberTraceFunction->yCenter)+1.,
                             double(fiberTraceFunction->yCenter + fiberTraceFunction->yLow + 1),//1.
                             double(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh + 1),//int(ccdHeight)
                             double(fiberTraceFunction->fiberTraceFunctionControl.xLow),
                             double(fiberTraceFunction->fiberTraceFunctionControl.xHigh),
                             int(fiberTraceFunction->fiberTraceFunctionControl.order),
                             int(ccdWidth),
                             int(ccdHeight))){
          std::string message("pfs::drp::stella::calculateXCenters: ERROR: Chebyshev(...) returned FALSE!!!");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        xCenters -= 1.;
      }
      else if (fiberTraceFunction->fiberTraceFunctionControl.interpolation.compare("LINEAR") == 0){
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::calculateXCenters: Function = spline1" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
        #endif
        if (!::pfs::drp::stella::math::LinearSpline(xCenters,
                                              fiberTraceFunctionCoefficients,
                                              double(fiberTraceFunction->xCenter) + 1.,
                                              double(fiberTraceFunction->yCenter) + 1.,
                                              double(fiberTraceFunction->yCenter + fiberTraceFunction->yLow + 1),//1.
                                              double(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh + 1),//int(ccdHeight)
                                              double(fiberTraceFunction->fiberTraceFunctionControl.xLow),
                                              double(fiberTraceFunction->fiberTraceFunctionControl.xHigh),
                                              int(fiberTraceFunction->fiberTraceFunctionControl.order),
                                              int(ccdWidth),
                                              int(ccdHeight))){
          std::string message("pfs::drp::stella::calculateXCenters: ERROR: LinearSpline(...) returned FALSE!!!");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        xCenters -= 1.;
      }
      else if (fiberTraceFunction->fiberTraceFunctionControl.interpolation.compare("CUBIC") == 0){
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::calculateXCenters: Function = spline3" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
        #endif
        if (!::pfs::drp::stella::math::CubicSpline(xCenters,
                                             fiberTraceFunctionCoefficients,
                                             double(fiberTraceFunction->xCenter) + 1.,
                                             double(fiberTraceFunction->yCenter) + 1.,
                                             double(fiberTraceFunction->yCenter + fiberTraceFunction->yLow + 1),//1.
                                             double(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh + 1),//int(ccdHeight)
                                             double(fiberTraceFunction->fiberTraceFunctionControl.xLow),
                                             double(fiberTraceFunction->fiberTraceFunctionControl.xHigh),
                                             int(fiberTraceFunction->fiberTraceFunctionControl.order),
                                             int(ccdWidth),
                                             int(ccdHeight))){
          std::string message("pfs::drp::stella::calculateXCenters: ERROR: CubicSpline(...) returned FALSE!!!");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        xCenters -= 1.;
      }
      else /// Polynomial
      {*/
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::calculateXCenters: Function = Polynomial" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
        #endif
        ndarray::Array<float, 1, 1> xRowIndex = ndarray::allocate(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1);
        float xRowInd = fiberTraceFunction->yCenter + fiberTraceFunction->yLow;
        for (auto i = xRowIndex.begin(); i != xRowIndex.end(); ++i){
          *i = xRowInd;
          ++xRowInd;
        }
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunctionCoefficients = " << fiberTraceFunctionCoefficients << endl;
        #endif
        xCenters = pfsDRPStella::math::Poly(xRowIndex, fiberTraceFunctionCoefficients);
      //}

/*      /// Check limits
      ndarray::Array<float, 1, 1> D_A1_XLow = ndarray::copy(xCenters + fiberTraceFunction->fiberTraceFunctionControl.xLow);
      #ifdef __DEBUG_XCENTERS__
        cout << "pfs::drp::stella::calculateXCenters: D_A1_XLow = " << D_A1_XLow << endl;
      #endif
      ndarray::Array<float, 1> D_A1_XHigh = ndarray::copy(xCenters + fiberTraceFunction->fiberTraceFunctionControl.xHigh);
      #ifdef __DEBUG_XCENTERS__
        cout << "pfs::drp::stella::calculateXCenters: D_A1_XHigh = " << D_A1_XHigh << endl;
      #endif
      ndarray::Array<int, 1> I_A1_Where = copy(D_A1_XLow < -0.5 ? 0 : 1);
      #ifdef __DEBUG_XCENTERS__
        cout << "pfs::drp::stella::calculateXCenters: I_A1_Where = " << I_A1_Where << endl;
      #endif
      int I_NInd;
      blitz::Array<int, 1> *P_I_A1_WhereInd = ::pfs::drp::stella::math::GetIndex(I_A1_Where, I_NInd);
      D_YMin = (*P_I_A1_WhereInd)(0);
      delete(P_I_A1_WhereInd);
      I_A1_Where = blitz::where(D_A1_XHigh < double(ccdWidth)-0.5, 1, 0);
      P_I_A1_WhereInd = ::pfs::drp::stella::math::GetIndex(I_A1_Where, I_NInd);
      D_YMax = (*P_I_A1_WhereInd)(P_I_A1_WhereInd->size()-1);
      delete(P_I_A1_WhereInd);

      #ifdef __DEBUG_XCENTERS__
        /// Coefficients for the trace functions
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunctionCoefficients = " << fiberTraceFunctionCoefficients << endl;

        /// Centres of the apertures in x (cols)
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunction->xCenter set to " << fiberTraceFunction->xCenter << endl;

        /// center position in y (row) for every aperture
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunction->yCenter set to " << fiberTraceFunction->yCenter << endl;

        /// lower aperture limit x (cols)
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunction->fiberTraceFunctionControl.xLow set to " << fiberTraceFunction->fiberTraceFunctionControl.xLow << endl;

        /// higher aperture limit x (cols)
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunction->fiberTraceFunctionControl.xHigh set to " << fiberTraceFunction->fiberTraceFunctionControl.xHigh << endl;

        /// lower aperture limit for every aperture y, rows
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunction->yLow set to " << fiberTraceFunction->yLow << endl;

        /// higher aperture limit for every aperture y, rows
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunction->yHigh set to " << fiberTraceFunction->yHigh << endl;

        /// order of aperture trace function
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunction->fiberTraceFunctionControl.order set to " << fiberTraceFunction->fiberTraceFunctionControl.order << endl;

        /// Name of function used to trace the apertures
        ///  0: chebyshev
        ///  1: legendre
        ///  2: cubic
        ///  3: linear
        ///  4: polynomial
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunction->FiberTraceFunctionControl.interpolation set to " << fiberTraceFunction->fiberTraceFunctionControl.interpolation << endl;

        cout << "pfs::drp::stella::calculateXCenters: D_YMin set to " << D_YMin << endl;
        cout << "pfs::drp::stella::calculateXCenters: D_YMax set to " << D_YMax << endl;

        cout << "pfs::drp::stella::calculateXCenters: xCenters set to " << xCenters << endl;
      #endif

      if ((D_YMin - fiberTraceFunction->yCenter) > fiberTraceFunction->yLow){
        string message("pfs::drp::stella::calculateXCenters: ERROR: (D_YMin - fiberTraceFunction->yCenter = ");
        message += to_string(D_YMin - fiberTraceFunction->yCenter) + string(") > fiberTraceFunction->yLow(=") + to_string(fiberTraceFunction->yLow);
        message += string(")");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
//        fiberTraceFunction->yLow = D_YMin - fiberTraceFunction->yCenter;
      }
      if ((D_YMax - fiberTraceFunction->yCenter) < fiberTraceFunction->yHigh){
        string message("pfs::drp::stella::calculateXCenters: ERROR: (D_YMax - fiberTraceFunction->yCenter = ");
        message += to_string(D_YMax - fiberTraceFunction->yCenter) + string(") < fiberTraceFunction->yHigh(=") + to_string(fiberTraceFunction->yHigh);
        message += string(")");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
//        fiberTraceFunction->yHigh = D_YMax - fiberTraceFunction->yCenter;
      }
      cout << "pfs::drp::stella::calculateXCenters: yCenter = " << fiberTraceFunction->yCenter << endl;
      cout << "pfs::drp::stella::calculateXCenters: yLow = " << fiberTraceFunction->yLow << endl;
      cout << "pfs::drp::stella::calculateXCenters: yHigh = " << fiberTraceFunction->yHigh << endl;
      //int xLowTooLowInLastRow = 2;
      //int xHighTooHighInLastRow = 2;
      for (int i = fiberTraceFunction->yCenter + fiberTraceFunction->yLow; i <= int(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh); i++){
        if (D_A1_XLow(i) < -0.5){
          string message("pfs::drp::stella::calculateXCenters: D_A1_XLow(");
          message += to_string(i) + string(") = ") +to_string(D_A1_XLow(i)) + string(" < -0.5");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
 //         if (xLowTooLowInLastRow == 0){
//            fiberTraceFunction->yHigh = i - fiberTraceFunction->yCenter - 1;
//            string message("pfs::drp::stella::calculateXCenters: xLowTooLowInLastRow == 0: fiberTraceFunction->yHigh set to " << fiberTraceFunction->yHigh << endl;
//          }
//          xLowTooLowInLastRow = 1;
        }
 //       else{
 //         if (xLowTooLowInLastRow == 1){
 //           fiberTraceFunction->yLow = i - fiberTraceFunction->yCenter + 1;
 //           cout << "pfs::drp::stella::calculateXCenters: xLowTooLowInLastRow == 1: fiberTraceFunction->yLow set to " << fiberTraceFunction->yLow << endl;
 //         }
 //         xLowTooLowInLastRow = 0;
 //       }
        if (D_A1_XHigh(i) > double(ccdWidth)-0.5){
          string message("pfs::drp::stella::calculateXCenters: D_A1_XHigh(");
          message += to_string(i) + string(")=") + to_string(D_A1_XHigh(i)) + string(" >= ccdWidth-0.5=") + to_string(ccdWidth-0.5);
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
//          if (xHighTooHighInLastRow == 0){
//            fiberTraceFunction->yHigh = i - fiberTraceFunction->yCenter - 1;
//            cout << "pfs::drp::stella::calculateXCenters: xHighTooHighInLastRow == 0: fiberTraceFunction->yHigh set to " << fiberTraceFunction->yHigh << endl;
//          }
//          xHighTooHighInLastRow = 1;
        }
//        else{
//          if (xHighTooHighInLastRow == 1){
//            fiberTraceFunction->yLow = i - fiberTraceFunction->yCenter + 1;
//            cout << "pfs::drp::stella::calculateXCenters: xHighTooHighInLastRow == 1: fiberTraceFunction->yLow set to " << fiberTraceFunction->yLow << endl;
//          }
//          xHighTooHighInLastRow = 0;
//        }
      }

      if (D_A1_XLow(fiberTraceFunction->yCenter + fiberTraceFunction->yLow) < -0.5){
        std::string message("pfs::drp::stella::calculateXCenters: ERROR: D_A1_XLow(fiberTraceFunction->yCenter + fiberTraceFunction->yLow=");
        message += to_string(fiberTraceFunction->yCenter + fiberTraceFunction->yLow) + string(")=");
        message += to_string(D_A1_XLow(fiberTraceFunction->yCenter + fiberTraceFunction->yLow)) + string(" < -0.5");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (D_A1_XLow(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh) < -0.5){
        string message("pfs::drp::stella::calculateXCenters: ERROR: D_A1_XLow(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh=");
        message += to_string(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh) + string(")=");
        message += to_string(D_A1_XLow(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh)) + string(" < -0.5");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (D_A1_XHigh(fiberTraceFunction->yCenter + fiberTraceFunction->yLow) > double(ccdWidth)-0.5){
        string message("pfs::drp::stella::calculateXCenters: ERROR: D_A1_XHigh(fiberTraceFunction->yCenter + fiberTraceFunction->yLow=");
        message += to_string(fiberTraceFunction->yCenter + fiberTraceFunction->yLow) + string(")=");
        message += to_string(D_A1_XHigh(fiberTraceFunction->yCenter + fiberTraceFunction->yLow)) + string(" > ccdWidth-0.5 =");
        message += to_string(double(ccdWidth)-0.5);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (D_A1_XHigh(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh) > double(ccdWidth)-0.5){
        string message("pfs::drp::stella::calculateXCenters: ERROR: D_A1_XHigh(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh=");
        message += to_string(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh) + string(")=");
        message += to_string(D_A1_XHigh(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh)) + string(" > ccdWidth-0.5=");
        message += to_string(double(ccdWidth)-0.5);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_XCENTERS__
        cout << "pfs::drp::stella::calculateXCenters: yLow set to " << fiberTraceFunction->yLow << endl;
        cout << "pfs::drp::stella::calculateXCenters: yHigh set to " << fiberTraceFunction->yHigh << endl;
      #endif

      /// populate _xCenters
      std::vector<float> xCenters(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1);
      for (int i = static_cast<int>(fiberTraceFunction->yCenter + fiberTraceFunction->yLow); i <= static_cast<int>(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh); i++) {
        xCenters[cIndex] = static_cast<float>(xCenters(i));
        cIndex++;
      }
      PTR(const std::vector<float>) pXCenters(new const std::vector<float>(xCenters));
      xCenters.resize(0);*/
      return xCenters;
    }
    
  }
  
  namespace utils{
  
    template<typename T>
    const T* getRawPointer(const PTR(const T) & ptr){
      return ptr.get();
    }
    
  }
  }}}

/*  int main(){
    cout << "Test that we can create a FiberTraceFunctionFindingControl" << endl;
    pfsDRPStella::FiberTraceFunctionFindingControl ftffc;

    cout << "Test that we can set the parameters of the FiberTraceFunctionFindingControl" << endl;
    ftffc.fiberTraceFunctionControl.interpolation = "POLYNOMIAL";
    ftffc.fiberTraceFunctionControl.order = 4;
    ftffc.fiberTraceFunctionControl.xLow = -4.2;
    ftffc.fiberTraceFunctionControl.xHigh = 4.2;
    ftffc.apertureFWHM = 3.2;
    ftffc.signalThreshold = 4500.;
    ftffc.nTermsGaussFit = 3;
    ftffc.saturationLevel = 65500.;

    afwImage::MaskedImage<float> maskedImageFlat = afwImage::MaskedImage<float>("/home/azuri/spectra/pfs/IR-23-0-centerFlatx2.fits");
    cout << "maskedImageFlat created" << endl;

    cout << "Test that we can trace fibers" << endl;
    pfsDRPStella::MaskedSpectrographImage<float> maskedSpectrographImageFlat(maskedImageFlat);
    cout << "maskedSpectrographImageFlat created" << endl;

    maskedSpectrographImageFlat.findAndTraceApertures(ftffc, 0, maskedImageFlat.getHeight(), 10);
    cout << "msi.findAndTraceApertures finished" << endl;

    cout << "calculate profile and extract from flat" << endl;
    pfsDRPStella::FiberTraceSet<float> fts = maskedSpectrographImageFlat.getFiberTraceSet();

    pfsDRPStella::FiberTrace<float> ft = fts.getFiberTrace(0);
    if (!ft.MkProfIm()){
      cout << "ERROR: ft.MkProfIm returned FALSE" << endl;
      return 0;
    }

    return 1;
  }*/

  template class pfsDRPStella::FiberTrace<float, unsigned short, float>;
  template class pfsDRPStella::FiberTrace<double, unsigned short, float>;
//  template class pfsDRPStella::FiberTrace<float, unsigned short, double>;
//  template class pfsDRPStella::FiberTrace<double, unsigned short, double>;

  template class pfsDRPStella::FiberTraceSet<float, unsigned short, float>;
  template class pfsDRPStella::FiberTraceSet<double, unsigned short, float>;
//  template class pfsDRPStella::FiberTraceSet<float, unsigned short, double>;
//  template class pfsDRPStella::FiberTraceSet<double, unsigned short, double>;

  template PTR(pfsDRPStella::FiberTraceSet<float, unsigned short, float>) pfsDRPStella::math::findAndTraceApertures(PTR(const afwImage::MaskedImage<float, unsigned short, float>) const&, 
                                                                                             PTR(const pfsDRPStella::FiberTraceFunctionFindingControl) const&);
  template PTR(pfsDRPStella::FiberTraceSet<double, unsigned short, float>) pfsDRPStella::math::findAndTraceApertures(PTR(const afwImage::MaskedImage<double, unsigned short, float>) const&, 
                                                                                              PTR(const pfsDRPStella::FiberTraceFunctionFindingControl) const&);

template const afwImage::MaskedImage<float, unsigned short, float>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::MaskedImage<float, unsigned short, float>) &ptr);
template const afwImage::MaskedImage<double, unsigned short, float>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::MaskedImage<double, unsigned short, float>) &ptr);
template const afwImage::Image<float>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<float>) &ptr);
template const afwImage::Image<double>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<double>) &ptr);
template const afwImage::Image<unsigned short>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<unsigned short>) &ptr);
template const afwImage::Image<unsigned int>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<unsigned int>) &ptr);
template const afwImage::Image<int>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<int>) &ptr);
template const pfsDRPStella::FiberTrace<float, unsigned short, float>* pfsDRPStella::utils::getRawPointer(const PTR(const pfsDRPStella::FiberTrace<float, unsigned short, float>) &ptr);
template const pfsDRPStella::FiberTrace<double, unsigned short, float>* pfsDRPStella::utils::getRawPointer(const PTR(const pfsDRPStella::FiberTrace<double, unsigned short, float>) &ptr);
