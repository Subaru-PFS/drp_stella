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
  _fiberTraceFunction(new FiberTraceFunction),
  _fiberTraceProfileFittingControl(new FiberTraceProfileFittingControl)
  {
    
  }

  /** @brief Construct an Exposure with a blank MaskedImage of specified size (default 0x0) and
   * a Wcs (which may be default constructed)
   */
  template<typename ImageT, typename MaskT, typename VarianceT>
  pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(PTR(const MaskedImageT) const & maskedImage, ///< desired image width/height
                                                                 PTR(const pfsDRPStella::FiberTraceFunction) const& fiberTraceFunction,
                                                                 ndarray::Array<float const, 1, 1> const& xCenters,
                                                                 size_t iTrace) :
  _trace(new afwImage::MaskedImage<ImageT, MaskT, VarianceT>(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1, int(fiberTraceFunction->fiberTraceFunctionControl.xHigh - fiberTraceFunction->fiberTraceFunctionControl.xLow + 1))),
  _profile(new afwImage::Image<float>(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1, int(fiberTraceFunction->fiberTraceFunctionControl.xHigh - fiberTraceFunction->fiberTraceFunctionControl.xLow + 1))),
  _xCenters(xCenters),//new std::vector<const float>(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1)),
  _iTrace(iTrace),
  _isXCentersCalculated(false),
  _isTraceSet(false),
  _isProfileSet(false),
  _isFiberTraceFunctionSet(false),
  _isFiberTraceProfileFittingControlSet(false),
  _fiberTraceFunction(new FiberTraceFunction),
  _fiberTraceProfileFittingControl(new FiberTraceProfileFittingControl)
  {
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(PTR(const MaskedImageT) const & maskedImage, ///< desired image width/height
                                                                 PTR(const pfsDRPStella::FiberTraceFunction) const& fiberTraceFunction,
                                                                 PTR(const std::vector<float>) const& xCenters,
                                                                 size_t iTrace) :
  _trace(new afwImage::MaskedImage<ImageT, MaskT, VarianceT>(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1, int(fiberTraceFunction->fiberTraceFunctionControl.xHigh - fiberTraceFunction->fiberTraceFunctionControl.xLow + 1))),
  _profile(new afwImage::Image<float>(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1, int(fiberTraceFunction->fiberTraceFunctionControl.xHigh - fiberTraceFunction->fiberTraceFunctionControl.xLow + 1))),
  _xCenters(xCenters),//new std::vector<const float>(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1)),
  _iTrace(iTrace),
  _isTraceSet(false),
  _isProfileSet(false),
  _isFiberTraceProfileFittingControlSet(false),
  _fiberTraceFunction(fiberTraceFunction),
  _fiberTraceProfileFittingControl(new FiberTraceProfileFittingControl)
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
  _fiberTraceProfileFittingControl(fiberTrace.getFiberTraceProfileFittingControl())
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
    PTR(std::vector<ImageT>) spec(new std::vector<ImageT>(_trace->getHeight()));
    PTR(std::vector<MaskT>) mask(new std::vector<MaskT>(_trace->getHeight()));
    PTR(std::vector<VarianceT>) var(new std::vector<VarianceT>(_trace->getHeight()));
    for (int i = 0; i < _trace->getHeight(); ++i){
      (*spec)[i] = sum(_trace->getImage()->getArray()[i]);
      (*var)[i] = sum(_trace->getVariance()->getArray()[i]);
      (*mask)[i] = sum(_trace->getMask()->getArray()[i]);
    }
    spectrum->setSpectrum(spec);
    spectrum->setVariance(var);
    spectrum->setMask(mask);
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

    blitz::Array<std::string, 1> keyWords(1);
    keyWords(0) = std::string("FIBERTRACENUMBER");
    void **args = (void**)malloc(sizeof(void*));
    args[0] = &_iTrace;

    blitz::Array<float, 2> F_A2_ProfArray = utils::ndarrayToBlitz(_profile->getArray());
    blitz::Array<double, 2> D_A2_ProfArray = ::pfs::drp::stella::math::Double(F_A2_ProfArray);
    blitz::Array<ImageT, 2> T_A2_CCDArray = utils::ndarrayToBlitz(_trace->getImage()->getArray());
    blitz::Array<double, 2> D_A2_CCDArray = ::pfs::drp::stella::math::Double(T_A2_CCDArray);
    blitz::Array<VarianceT, 2> T_A2_VarianceArray = utils::ndarrayToBlitz(_trace->getVariance()->getArray());
    blitz::Array<double, 2> D_A2_ErrArray = ::pfs::drp::stella::math::Double(T_A2_VarianceArray);
    ///TODO: change to sqrt(T_A2_VarianceArray)
    D_A2_ErrArray = sqrt(where(D_A2_CCDArray > 0., D_A2_CCDArray, 1.));
    blitz::Array<MaskT, 2> T_A2_MaskArray = utils::ndarrayToBlitz(_trace->getMask()->getArray());
    blitz::Array<int, 2> I_A2_MaskArray = ::pfs::drp::stella::math::Int(T_A2_MaskArray);
    I_A2_MaskArray = where(I_A2_MaskArray == 0, 1, 0);
    blitz::Array<double, 1> D_A1_SP(_trace->getHeight());
    D_A1_SP = 0.;
    blitz::Array<string, 1> S_A1_Args_Fit(3);
    void **PP_Args_Fit;
    PP_Args_Fit = (void**)malloc(sizeof(void*) * 3);
    S_A1_Args_Fit = " ";

    S_A1_Args_Fit(0) = "MEASURE_ERRORS_IN";
    PP_Args_Fit[0] = &D_A2_ErrArray;
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: D_A2_ErrArray = " << D_A2_ErrArray << endl;
    #endif

    S_A1_Args_Fit(1) = "MASK_INOUT";
    PP_Args_Fit[1] = &I_A2_MaskArray;
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "I_A2_MaskArray = " << I_A2_MaskArray << endl;
    #endif

    S_A1_Args_Fit(2) = "SIGMA_OUT";
    blitz::Array<double, 2> D_A2_Sigma_Fit(_trace->getHeight(),2);
    PP_Args_Fit[2] = &D_A2_Sigma_Fit;

    blitz::Array<double, 1> D_A1_Sky(_trace->getHeight());
    D_A1_Sky = 0.;
    bool B_WithSky = false;
    if (_fiberTraceProfileFittingControl->telluric.compare(_fiberTraceProfileFittingControl->TELLURIC_NAMES[0]) != 0){
      D_A1_Sky = 1.;
      B_WithSky = true;
      cout << "extractFromProfile: Sky switched ON" << endl;
    }
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: Before Fit: D_A2_CCDArray = " << D_A2_CCDArray << endl;
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: Before Fit: D_A2_ProfArray = " << D_A2_ProfArray << endl;
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: Before Fit: D_A1_SP = " << D_A1_SP << endl;
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: Before Fit: D_A1_Sky = " << D_A1_Sky << endl;
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: Before Fit: B_WithSky = " << B_WithSky << endl;
      for (int iargpos=0; iargpos < S_A1_Args_Fit.size(); iargpos++){
        cout << "FiberTrace" << _iTrace << "::extractFromProfile: Before Fit: S_A1_Args_Fit[" << iargpos << "] = " << S_A1_Args_Fit[iargpos] << endl;
      }
    #endif
    #ifdef __DEBUG_EXTRACTFROMPROFILE_FILES__
      string S_FileName_CCD_Ap = "CCD_Ap" + to_string(fiberTraceNumber) + "_Tel" + to_string(telluric) + ".fits";
      if (!::pfs::drp::stella::utils::WriteFits(&D_A2_CCDArray,S_FileName_CCD_Ap)){
        string message("FiberTrace");
        message += to_string(_iTrace) + string("::extractFromProfile: WriteFits(D_A2_CCD_Ap,") + S_FileName_CCD_Ap;
        message += string(") returned FALSE");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    #endif
    if (!::pfs::drp::stella::math::LinFitBevington(D_A2_CCDArray,      ///: in
                                             D_A2_ProfArray,             ///: in
                                             D_A1_SP,             ///: out
                                             D_A1_Sky,          ///: in/out
                                             B_WithSky,                   ///: with sky: in
                                             S_A1_Args_Fit,         ///: in
                                             PP_Args_Fit)){          ///: in/out
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

    PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) spectrum(new pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>(_trace->getHeight()));
    PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) background(new pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>(_trace->getHeight()));
    for (int i = 0; i < _trace->getHeight(); i++) {
      (*(spectrum->getSpectrum()))[i] = static_cast<ImageT>(D_A1_SP(i));
      (*(spectrum->getVariance()))[i] = static_cast<VarianceT>(blitz::pow2(D_A2_Sigma_Fit(i, 0)));
      (*(background->getSpectrum()))[i] = static_cast<ImageT>(D_A1_Sky(i));
      (*(background->getVariance()))[i] = static_cast<VarianceT>(pow(D_A2_Sigma_Fit(i, 1),2));
    }
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
    cout << "FiberTrace::createFiberTrace: _trace set to " << _trace->getImage()->getArray() << endl;
    if (!_isProfileSet){
      _profile.reset(new afwImage::Image<float>(_trace->getWidth(), _trace->getHeight()));
    }
    cout << "FiberTrace::createFiberTrace: _trace set to " << _trace->getImage()->getArray() << endl;
    if (!_isProfileSet){
      _profile.reset(new afwImage::Image<float>(_trace->getWidth(), _trace->getHeight()));
    }
    _isTraceSet = true;
    return true;
  }

  /// Return shared pointer to an image containing the reconstructed 2D spectrum of the FiberTrace
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(afwImage::Image<float>) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::getReconstructed2DSpectrum(const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT> & spectrum) const{
    PTR(afwImage::Image<float>) image(new afwImage::Image<float>(_trace->getWidth(), _trace->getHeight()));
    blitz::Array<float, 2> F_A2_Prof = ::pfs::drp::stella::utils::ndarrayToBlitz(_profile->getArray());
    blitz::Array<float, 2> F_A2_Rec(_trace->getHeight(), _trace->getWidth());
    for (int i_row=0; i_row<_trace->getHeight(); i_row++)
      F_A2_Rec(i_row, blitz::Range::all()) = F_A2_Prof(i_row, blitz::Range::all()) * (*(spectrum.getSpectrum()))[i_row];
    ndarray::Array<float, 2, 1> ndarrayRec(::pfs::drp::stella::utils::copyBlitzToNdarray(F_A2_Rec));
    afwImage::Image<float> imRec(ndarrayRec);
    *image = imRec;
    return image;
  }

  /// Return shared pointer to an image containing the reconstructed background of the FiberTrace
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(afwImage::Image<float>) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::getReconstructedBackground(const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT> & background) const{
    PTR(afwImage::Image<float>) image(new afwImage::Image<float>(_trace->getWidth(), _trace->getHeight()));
    blitz::Array<float, 2> F_A2_Rec(_trace->getHeight(), _trace->getWidth());
    for (int i_row=0; i_row<_trace->getHeight(); i_row++)
      F_A2_Rec(i_row, blitz::Range::all()) = (*(background.getSpectrum()))[i_row];
    ndarray::Array<float, 2, 1> ndarrayRec(::pfs::drp::stella::utils::copyBlitzToNdarray(F_A2_Rec));
    afwImage::Image<float> imRec(ndarrayRec);
    *image = imRec;
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
  ndarray::Array<int, 2, 1> pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::calculateBinBoundY(int swathWidth) const{
    int I_NI = 0;
    int I_I = 0;
    int nBins = 0;

    blitz::Array<int, 1> I_A1_I(1);
    I_A1_I = 0;

    int swathWidth_mutable = swathWidth;
    if (swathWidth_mutable > _trace->getHeight()){
      swathWidth_mutable = _trace->getHeight();
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: KeyWord_Set(SWATH_WIDTH): swathWidth_mutable too large: swathWidth_mutable set to " << swathWidth_mutable << endl;
      #endif
    }
/*    if (swathWidth_mutable == 0)/// Calculate swathWidth automatically
    { /// Estimate the Points of column crossing
      const ndarray::Array<const float, 1, 1> xCenters = ndarray::external(_xCenters.data(), ndarray::makeVector(int(_xCenters.size())), ndarray::makeVector(1));
      #ifdef __DEBUG_CALCSWATHBOUNDY__
        cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: xCenters.getShape()[0] = " << xCenters.getShape()[0] << endl;
      #endif
    }
    if (swathWidth_mutable == 0)
    { /// Estimate the Points of column crossing
      PTR(std::vector<float>) pXCenters = const_pointer_cast<vector<float>>(_xCenters);
      blitz::Array<float, 1> xCenters(pXCenters->data(), blitz::shape(_xCenters->size()), blitz::neverDeleteData);
      blitz::Array<int, 1> tempIntArrA = ::pfs::drp::stella::math::Fix(xCenters);
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: !KeyWord_Set(SWATH_WIDTH): tempIntArrA = " << tempIntArrA << endl;
      #endif
      ::pfs::drp::stella::math::Uniq(tempIntArrA, I_A1_I);
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: !KeyWord_Set(SWATH_WIDTH): I_A1_I set to " << I_A1_I << endl;
      #endif

      ///This is how many times this order crosses to the next column
      I_NI = I_A1_I.size();
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: !KeyWord_Set(SWATH_WIDTH): I_NI set to " << I_NI << endl;
      #endif

      ///Curved order crosses columns
      if (I_NI > 1)
      {
        I_I = blitz::sum(I_A1_I(blitz::Range(1, I_NI-1)) - I_A1_I(blitz::Range(0, I_NI - 2))) / (I_NI - 1);
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") > 1): I_I set to " << I_I << endl;
        #endif

        /// number of swaths along the order
        nBins = ::pfs::drp::stella::math::Round(static_cast<double>(_trace->getHeight()) / static_cast<double>(I_I) / 3.);
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") > 1): I_NBin = Round((double)_trace->getHeight()(=" << _trace->getHeight() << ") / (double)I_I(=" << I_I << ") / 3.) set to " << nBins << endl;
          cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") > 1): I_NBin set to " << nBins << endl;
        #endif
      }
      else
      { /// Perfectly aligned orders
        /// Still follow the changes in PSF
        nBins = ::pfs::drp::stella::math::Round(static_cast<double>(_trace->getHeight()) / 400.);
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") <= 1): I_NBin = int(_trace->getHeight()(=" << _trace->getHeight() << ") / 400.) set to " << nBins << endl;
          cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") <= 1): I_NBin set to " << nBins << endl;
        #endif
      }
      if (nBins < 3)
        nBins = 3;
      if (nBins > 20)
        nBins = 2;
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: !KeyWord_Set(SWATH_WIDTH): I_NBin set to " << nBins << endl;
      #endif

    }

    int binHeight = _trace->getHeight() / nBins;
    if (nBins > 1)
      nBins = (2 * nBins) - 1;

    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: fiberTraceNumber = " << _iTrace << endl;
      cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: nBins set to " << nBins << endl;
    #endif

    /// Calculate boundaries of distinct slitf regions.
    /// Boundaries of bins
    blitz::Array<int, 2> binBoundY(nBins,2);
    binBoundY = 0;
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: 1. I_A2_IBound(0,0) set to " << binBoundY(0,0) << endl;
    #endif
    int I_BinHeight_Temp = binHeight;
    binBoundY(0,1) = I_BinHeight_Temp;
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: binBoundY(0, 1) set to " << binBoundY(0, 1) << endl;
    #endif
    #ifdef __DEBUG_CHECK_INDICES__
      if(binBoundY(0,1) >= int(_trace->getHeight())){
        string message("FiberTrace");
        message += to_string(_iTrace) + string("::calculateBinBoundY: ERROR: binBoundY(0,1)=");
        message += to_string(binBoundY(0,1)) + string(" >= _trace->getHeight()=") + to_string(_trace->getHeight());
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    #endif
    for (int i_bin = 1; i_bin < nBins; i_bin++){
      I_BinHeight_Temp = binHeight;
      if (i_bin == 1)
        binBoundY(i_bin,0) = binBoundY(i_bin-1,0) + int(double(binHeight) / 2.);
      else
        binBoundY(i_bin,0) = binBoundY(i_bin-2,1) + 1;
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: binBoundY(i_bin=" << i_bin << ",0) set to binBoundY(i_bin-1=" << i_bin-1 << ",0) + (binHeight/2.=" << binHeight / 2. << ")" << endl;
      #endif
//      while(binBoundY(i_bin,0) < 0){
//        binBoundY(i_bin,0)++;
//        I_BinHeight_Temp--;
//        #ifdef __DEBUG_MKSLITFUNC__
//          cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: binBoundY(i_bin,0) < 0: binBoundY(" << i_bin << ", 0) set to " << binBoundY(i_bin, 0) << endl;
//        #endif
//      }
      binBoundY(i_bin,1) = binBoundY(i_bin,0) + I_BinHeight_Temp;
      if (i_bin == (nBins-1)){
        binBoundY(i_bin,1) = _trace->getHeight()-1;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: nBins = " << nBins << endl;
          cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: _trace->getHeight() = " << _trace->getHeight() << endl;
          cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: binBoundY(" << i_bin << ",1) set to " << binBoundY(i_bin, 1) << endl;
        #endif
      }
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: binBoundY(" << i_bin << ",1) set to " << binBoundY(i_bin, 1) << endl;
      #endif
      #ifdef __DEBUG_CHECK_INDICES__
        if(binBoundY(i_bin,1) >= _trace->getHeight()){
          string message("FiberTrace");
          message += to_string(_iTrace) + string("::calculateBinBoundY: ERROR: binBoundY(i_bin=");
          message += to_string(i_bin) + string(",1)=");
          message += to_string(binBoundY(i_bin,1)) + string(" >= _trace->getHeight()=") + to_string(_trace->getHeight());
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
      #endif
    }
//    #ifdef __DEBUG_MKSLITFUNC__
      for (int row=0; row < binBoundY.rows(); ++row)
        cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: binBoundY(" << row << ",*) set to " << binBoundY(row, blitz::Range::all()) << ", length = " << binBoundY(row, 1) - binBoundY(row, 0) + 1 << endl;
//    #endif
    #ifdef __DEBUG_CHECK_INDICES__
      if (binBoundY(binBoundY.rows()-1, 1) != (_trace->getHeight()-1)){
        string message("FiberTrace ");
        message += to_string(_iTrace) + string("::calculateBinBoundY: ERROR: binBoundY(binBoundY.rows()-1=");
        message += to_string(binBoundY.rows()-1) + string(", 1)=") + to_string(binBoundY(binBoundY.rows()-1, 1));
        message += string("!= (_trace->getHeight()-1)=") + to_string(_trace->getHeight()-1);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    #endif
//    binBoundY(nBins-1, 1) = _trace->getHeight()-1;
    ndarray::Array<int, 2, 1> binBoundY_Out = ndarray::copy(utils::blitzToNdarray(binBoundY));
//    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::calculateBinBoundY: binBoundY_Out set to " << binBoundY_Out << endl;
//    #endif

    return binBoundY_Out;
  }

  /**
   *  MkSlitFunc
   *  Make Slit Function
   **/
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::MkSlitFunc(const blitz::Array<string, 1> &S_A1_Args_In,           //: in
                                                                                                     void *ArgV_In[])                  //: in
  /**                     //PLOT        =
   * //                      Y_LOWER_LIM = int             : in
   * //                      Y_UPPER_LIM = int             : in
   * //                      LAMBDA_SF   = double          : in
   * //                      LAMBDA_SP   = int             : in
   * //                      WING_SMOOTH_FACTOR = double   : in
   * //                      SWATH_WIDTH = int             : in
   *                       BLZ         = blitz::Array<double, 1>: out
   * ///                       MASK        = blitz::Array<double, 2>: in
   * //                       CCD_GAIN    = double          : in
   * //                       CCD_READN   = double          : in
   * //                      NO_SCATTER  = void
   * //                      TELLURIC    = int[0-none, 1-Piskunov, 2-mine]        : in
   *                       FILENAME    = CString         : in For debugging purposes only
   * //                      XCOR_PROF   = int             : in
   *                       Y_PROF_START = int            : in 0 <= Y_PROF_START < NRows
   *                       Y_PROF_END  = int            : in Y_PROF_START < Y_PROF_END < NRows
   **/
  {
    /**
     *    blitz::Array<double, 1> bincen   => blitz::Array<double, 1> D_A1_BinCen
     *  ! blitz::Array<double, 1> BLZ=blz = fltarr(ncol)  => blitz::Array<double, 1> *P_D_A1_BLZ
     *    FILENAME=filename
     *  ! double GAIN=gain=CCD_gain       => double D_CCDGain       ; Gain
     *    blitz::Array<long, 1>/int i      => blitz::Array<int, 1> I_A1_I / int I_I ;Points of row crossing
     *  ! long ib                   => int I_IB
     *  ! long ie                   => int I_IE
     *  ! blitz::Array<long, 1> ibeg      => blitz::Array<int, 1> I_A1_IBeg
     *  ! blitz::Array<double, 1> ibound  => blitz::Array<double, 1> D_A1_IBound
     *  ! int icen = yc(ib+j)     => int I_ICen
     *  ! blitz::Array<long, 1> iend      => blitz::Array<int, 1> I_A1_IEnd
     *  ! blitz::Array<double, 2> im              => CFits P_CF_Im
     *  ! int imask = 0                    => int I_IMask
     * //  ! blitz::Array<long, 1> imax      => blitz::Array<int, 1> I_A2_MinCenMax(*,2)
     * //  ! blitz::Array<long, 1> imin      => blitz::Array<int, 1> I_A2_MinCenMax(*,0)
     *    blitz::Array<double, 1> irow    => blitz::Array<double, 1> D_A1_ICol
     *  ! blitz::Array<long, 1> j0        => blitz::Array<int, 1> I_A1_J0
     *  ! blitz::Array<long, 1> j1        => blitz::Array<ing, 1> I_A1_J1
     *   ! blitz::Array<int, 1> jbad      => blitz::Array<int, 1> I_A1_JBad
     *    !int jgood               => int I_JGood
     *  ! long k0                 => int I_A2_MinCenMax(I_IB + n, 0)
     *  ! long k1                 => int I_A2_MinCenMax(I_IB + n, 2)
     *  ! double LAMBDA_SF=lam_sf              => double D_LambdaSF
     *  ! int LAMBDA_SP=lam_sp                 => int I_LambdaSP
     *  ! blitz::Array<double, 2> MASK=mask            => blitz::Array<double, 2> I_A2_Mask
     *  ! bytarr msk(nc,nysf) / = 0  => blitz::Array<double, 2> I_A2_Msk
     *  ! int nbad                 => int I_NBad
     *  ! int nbin                 => int I_NBin
     *  ! long nc                  => int I_NR
     *  ! int ncol                 => int P_CF_Im->NRows
     *  ! int ni                   => int I_NI    ;This is how many times this order crosses to the next column
     *  NO_SCATTER=no_scatter
     *  ! int nrow                        => int P_CF_Im->NCols
     *  ! long nsf                        => int I_NSF
     *  ! int nslitf                     => int I_NSlitF
     *  ! long nysf                       => int I_NXSF
     *  ! int ord_num                             => int IOrdNum
     *  PLOT=iplot
     *  ! double READN=readn=CCD_readn=0. => double D_CCDReadN      ; Readout noise
     *  ! int OSAMPLE=osample                  => int I_OverSample
     *  blitz::Array<double, 1> dy_scatter         => blitz::Array<double, 1> D_A1_DXScatter
     *  blitz::Array<double, 1> yscatter_below     => blitz::Array<double, 1> D_A1_XScatterBelow
     *  blitz::Array<double, 1> yscatter_above     => blitz::Array<double, 1> D_A1_XScatterAbove
     *  blitz::Array<double, 1> scatter            => blitz::Array<double, 1> D_A1_Scatter
     *  blitz::Array<double, 1> scatter_above      => blitz::Array<double, 1> D_A1_ScatterAbove
     *  blitz::Array<double, 1> scatter_below      => blitz::Array<double, 1> D_A1_ScatterBelow
     *  ! blitz::Array<double, 2> sf(nc,nysf)      => blitz::Array<double, 2> D_A2_SlitFunc_Im_In
     *  ! blitz::Array<double, 2> sfbin            => blitz::Array<double, 2> D_A2_SFOut;
     *  ! blitz::Array<double, 1> sfpnt(nsf)       => blitz::Array<double, 1> D_A1_SFPnt
     *  ! blitz::Array<double, 1> sfsm             => blitz::Array<double, 1> D_A1_SFSM
     *  blitz::Array<double, 2> slitf
     *  ! blitz::Array<double, 1> ssf               => blitz::Array<double, 1> D_A1_SSF
     *  ! int SWATH_WIDTH=swath_width       => int _fiberTraceProfileFittingControl->swathWidth
     *  blitz::Array<double, 1> tel                => blitz::Array<double, 1> D_A1_Tel
     *  TELLURIC=telluric
     *  ! long X_LEFT_LIM=x_left_lim        => int (int)((*P_D_A1_XMin)(fiberTraceNumber))
     *  ! long X_RIGHT_LIM=x_right_lim       => int (int)((*P_D_A1_XMax)(fiberTraceNumber))
     *  ! blitz::Array<int, 1> yc                   => blitz::Array<int, 1> I_A1_XC
     *  yscatter_below
     *  yscatter_above
     *  ! blitz::Array<double, 1> ysfpnt(nsf)      => blitz::Array<double, 1> D_A1_XSFPnt
     *  ! blitz::Array<double, 1> ycen(ncol)        => blitz::Array<double, 1> P_D_A2_XCenters(fiberTraceNumber)
     *  ! blitz::Array<double, 1> ycene             => blitz::Array<double, 1> D_A1_XCentersE
     *  ! long y_lower_lim                   => int 0. - (_fiberTraceFunction->fiberTraceFunctionControl.xLow)(fiberTraceNumber), P_D_A1_XLow(fiberTraceNumber)
     *  ! long y_upper_lim                   => int (*P_D_A1_XHigh)(fiberTraceNumber), P_D_A1_XHigh(fiberTraceNumber)
     *  blitz::Array<double, 1> yslitf
     *  ! long yslitf0 = -y_lower_lim        => int I_XSlitFunc0
     *  ! long yslitf1 =  y_upper_lim        => int I_XSlitFunc1
     **/
    
    #ifdef __DEBUG_MKSLITFUNC__
      int oldYHigh = _fiberTraceFunction->yHigh;
      int oldYLow = _fiberTraceFunction->yLow;
    #endif
    if (!_isFiberTraceProfileFittingControlSet){
      std::string message("FiberTrace");
      message += std::to_string(_iTrace);
      message += std::string("::MkSlitFunc: ERROR: _fiberTraceProfileFittingControl is not set");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (!_isTraceSet){
      std::string message("FiberTrace");
      message += std::to_string(_iTrace);
      message += std::string("::MkSlitFunc: ERROR: _trace is not set");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    cout << "FiberTrace" << _iTrace << "::MkSlitFunc: Started: S_A1_Args_In = " << S_A1_Args_In << endl;
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: Started: S_A1_Args_In = " << S_A1_Args_In << endl;
    #endif

    string S_SF_DebugFilesSuffix = "";

    blitz::Array<ImageT, 2> T_A2_PixArray = utils::ndarrayToBlitz(_trace->getImage()->getArray());
    blitz::Array<double, 2> D_A2_PixArray = ::pfs::drp::stella::math::Double(T_A2_PixArray);
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: D_A2_PixArray.size = " << D_A2_PixArray.rows() << " rows x " << D_A2_PixArray.cols() << " cols" << endl;
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: D_A2_PixArray(0,*) = " << D_A2_PixArray(0,blitz::Range::all()) << endl;
    #endif

    blitz::Array<VarianceT, 2> T_A2_Variance = utils::ndarrayToBlitz(_trace->getVariance()->getArray());
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: T_A2_Variance(0,*) = " << T_A2_Variance(0,blitz::Range::all()) << endl;
    #endif
    ///TODO: change to sqrt(T_A2_Variance)
    blitz::Array<double, 2> D_A2_Errors(D_A2_PixArray.rows(), D_A2_PixArray.cols());
    D_A2_Errors = sqrt(where(D_A2_PixArray > 0., D_A2_PixArray, 1.));//::pfs::drp::stella::math::Double(T_A2_Variance);
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: D_A2_Errors(0,*) = " << D_A2_Errors(0,blitz::Range::all()) << endl;
    #endif

    blitz::Array<MaskT, 2> T_A2_MaskArray = utils::ndarrayToBlitz(_trace->getMask()->getArray());
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: T_A2_MaskArray(0,*) = " << T_A2_MaskArray(0,blitz::Range::all()) << endl;
    #endif
    blitz::Array<int, 2> I_A2_MaskArray(D_A2_PixArray.rows(), D_A2_PixArray.cols());// = ::pfs::drp::stella::math::Int(T_A2_MaskArray);
    ///TODO: use _trace->getMask
    I_A2_MaskArray = where(T_A2_MaskArray == 0, 1, 0);
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_A2_MaskArray(0,*) = " << I_A2_MaskArray(0,blitz::Range::all()) << endl;
    #endif

    blitz::Array<double, 2> D_A2_ProfArray(_trace->getHeight(), _trace->getWidth());

    blitz::Array<double, 1> *P_D_A1_BLZ = new blitz::Array<double, 1>(1);
    (*P_D_A1_BLZ) = 0.;

    blitz::Array<int, 2> I_A2_Mask(1, 1);
    I_A2_Mask = 0;

    blitz::Array<double, 2> D_A2_Mask(1, 1);
    D_A2_Mask = 0.;

    blitz::Array<int, 2> I_A2_MaskApTemp(1, 1);
    I_A2_MaskApTemp = 0;

    blitz::Array<int, 2> I_A2_Msk(1, 1);
    I_A2_Msk = 0.;

    blitz::Array<double, 1> D_A1_SC(1);
    D_A1_SC = 0.;

    blitz::Array<double, 1> D_A1_Scatter(1);
    D_A1_Scatter = 0.;

    blitz::Array<double, 1> D_A1_SF(1);
    D_A1_SF = 0.;

    blitz::Array<double, 2> D_A2_SlitFunc_Im_In(1, 1);
    D_A2_SlitFunc_Im_In = 0.;

    blitz::Array<double, 2> D_A2_CCD_Ap(2,2);
    D_A2_CCD_Ap = 0.;

    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: D_A2_Errors(0,*) = " << D_A2_Errors(0,blitz::Range::all()) << endl;
    #endif
    blitz::Array<double, 2> D_A2_Err(2,2);

    D_A2_Err = 0.;
    blitz::Array<double, 2> D_A2_Err_AllRows(1, 1);
    D_A2_Err_AllRows = 0.;
    blitz::Array<int, 2> I_A2_Mask_AllRows(1, 1);
    I_A2_Mask_AllRows = 1;

    blitz::Array<double, 1> D_A1_Err(1);
    D_A1_Err = 0.;

    blitz::Array<double, 2> D_A2_SFSM(1,1);
    D_A2_SFSM = 0.;
    blitz::Array<double, 3> D_A3_SFSM(1,1,1);
    D_A3_SFSM = 0.;

    blitz::Array<double, 2> D_A2_SlitFTemp(1,1);
    D_A2_SlitFTemp = 0.;

    blitz::Array<double, 1> D_A1_SP(1);
    D_A1_SP = 0.;
    blitz::Array<double, 2> D_A2_SP(1,1);
    D_A2_SP = 0.;
    blitz::Array<double, 2> D_A2_Errors_SP_Out(1,1);
    D_A2_Errors_SP_Out = 0.;
    blitz::Array<double, 2> D_A2_XCorProf(1,1);
    D_A2_XCorProf = 0.;
   // bool B_Run_XCor = false;

    blitz::Array<double, 2> D_A2_Sky(1,1);
    blitz::Array<double, 2> D_A2_ErrSky(1,1);

    blitz::Array<double, 1> D_A1_SSF(1);
    D_A1_SSF = 0.;

    blitz::Array<double, 1> D_A1_Tel(1);
    D_A1_Tel = 0.;

    blitz::Array<double, 2> D_A2_Tel(1,1);
    D_A2_Tel = 0.;

    blitz::Array<double, 1> D_A1_Temp(1);
    D_A1_Temp = 0.;

    blitz::Array<double, 1> D_A1_TempArr(1);
    D_A1_TempArr = 0.;

    blitz::Array<double, 1> D_A1_TempArrA(1);
    D_A1_TempArrA = 0.;

    blitz::Array<double, 1> D_A1_TempArrB(1);
    D_A1_TempArrB = 0.;

    blitz::Array<double, 1> D_A1_TempArrC(1);
    D_A1_TempArrC = 0.;

    blitz::Array<double, 1> D_A1_TempArrD(1);
    D_A1_TempArrD = 0.;

    blitz::Array<double, 1> D_A1_XInt(1);
    D_A1_XInt = 0.;

    blitz::Array<double, 1> D_A1_XSlitFTemp(1);
    D_A1_XSlitFTemp = 0.;

    blitz::Array<int, 1> I_A1_ISort(1);
    I_A1_ISort = 0;

    blitz::Array<int, 1> I_A1_ITel(1);
    I_A1_ITel = 0;

    blitz::Array<int, 1> I_A1_IX(1);
    I_A1_IX = 0;

    blitz::Array<double, 2> D_A2_ErrTel(1);
    blitz::Array<double, 1> D_A1_ErrTel(1);
    blitz::Array<int, 1> I_A1_ErrInd(1);
    blitz::Array<double, 1> D_A1_ErrTelSub(1);
    blitz::Array<double, 2> D_A2_SF(1,1);
    blitz::Array<double, 1> D_A1_ErrSky(1);
    blitz::Array<double, 1> D_A1_ErrOut(1);
    D_A1_ErrOut = 0.;
    blitz::Array<double, 1> D_A1_ErrFit(1);

    string S_TempNum = "";

    blitz::Array<string, 1> S_A1_Args_Median(3);
    S_A1_Args_Median = " ";
    S_A1_Args_Median(0) = "NORMAL";
    void **PP_Args_Median = (void**)malloc(sizeof(void*) * 3);
    int I_Val_Normal = 1;
    double D_Val_ErrOutMedian;

    ///TODO
    bool ErrorsRead = true;
    if (ErrorsRead){
      PP_Args_Median[0] = &I_Val_Normal;
      S_A1_Args_Median(1) = "ERRORS_IN";
      S_A1_Args_Median(2) = "ERR_OUT";
      PP_Args_Median[2] = &D_Val_ErrOutMedian;
    }

    int I_NR = 0;// = I_IE - I_IB + 1; /// Number of rows
//    int I_LambdaSP = 1;
    int I_NBins = 0;
    int I_NXSF = 0;
    int I_Pos = 0;
    int I_MaxIterSig = _fiberTraceProfileFittingControl->maxIterSig;
    int I_XCorProf = 0;//_fiberTraceProfileFittingControl->xCorProf;
    int pos = 0;
    int pppos = 0;

//    double D_LambdaSF = 1.;
//    double D_WingSmoothFactor = 0.;

    blitz::Array<double,1> D_A1_SF_Median(1);
    blitz::Array<double,1> D_A1_Sky(1);
    blitz::Array<double,1> D_A1_SPFit(1);
    blitz::Array<double,1> D_A1_SP_Out(1);
    blitz::Array<int, 2> I_A2_Mask_Tel(1,1);
    blitz::Array<int, 2> I_A2_Mask_TelTemp(1,1);
    blitz::Array<int, 1> I_A1_UseRow_Tel(1);
    blitz::Array<int, 1> I_A1_UseRow_TelTemp(1);
    blitz::Array<double, 2> D_A2_ErrIn_Tel(1,1);
    blitz::Array<double, 2> D_A2_Err_Temp(1,1);
    blitz::Array<double, 2> D_A2_SlitFuncOrig(1,1);
    blitz::Array<double, 2> D_A2_TempIm(1,1);
    blitz::Array<double, 1> D_A1_SFMax(1);
    int nind_temp;
    blitz::Array<int,1> I_A1_IndA(1);
    blitz::Array<int, 1> *P_I_A1_Ind;
    blitz::Array<int, 1> I_A1_Ind_Last(1);
    blitz::Array<int, 1> I_A1_UseRow_Tel_AllRows(1);
    blitz::Array<int, 1> I_A1_Ind_Temp(1);
    blitz::Array<int, 1> I_A1_SFMaxInd(1);
    int I_RunMax = 1;
    int I_Run_Tel = 0;
    blitz::Array<double, 2> D_A2_SlitFunc_Im_In_Tel(1,1);
    blitz::Array<double, 1> D_A1_Sky_Temp(1);
    blitz::Array<double, 2> D_A2_SFOld(1,1);
    double D_SFDev = 0.;
    blitz::Array<int, 2> I_A2_Mask_Temp(1,1);

    bool B_MaximaOnly = false;

    string sTemp = "Y_LOWER_LIM";
    string debugdir = DEBUGDIR;

    blitz::Array<string, 1> s_a1(23);
    void **args = (void**)malloc(sizeof(void*) * 23);

    blitz::firstIndex i;
    blitz::secondIndex j;

    int telluric = 0;
    for ( int fooInt = FiberTraceProfileFittingControl::NONE; fooInt != FiberTraceProfileFittingControl::NVALUES; fooInt++ ){
      if (_fiberTraceProfileFittingControl->telluric.compare(_fiberTraceProfileFittingControl->TELLURIC_NAMES[fooInt]) == 0){
        telluric = fooInt;
      }
    }
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: telluric set to " << telluric << endl;
    #endif

    if ((I_Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "BLZ")) >= 0)
    {
      if (P_D_A1_BLZ != NULL)
        delete P_D_A1_BLZ;
      P_D_A1_BLZ = (blitz::Array<double, 1>*)ArgV_In[I_Pos];
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::MkSlitFunc: KeyWord_Set(BLZ): P_D_A1_BLZ set to " << *P_D_A1_BLZ << endl;
      #endif
    }

    sTemp = "FIBERTRACENUMBER";
    unsigned int fiberTraceNumber = 0;
    if ((I_Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
    {
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_Pos = " << I_Pos << endl;
      #endif
      fiberTraceNumber = *(unsigned int*)ArgV_In[I_Pos];
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: KeyWord_Set(FIBERTRACENUMBER): fiberTraceNumber set to " << fiberTraceNumber << endl;
      s_a1(pppos) = "FIBERTRACENUMBER";
      args[pppos] = &fiberTraceNumber;
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "args[pppos=" << pppos << "] set to FIBERTRACENUMBER = " << *(unsigned int*)args[pppos] << endl;
      #endif
      pppos++;
    }

//    if (fiberTraceProfileFittingControl->xCorProf > 0)
//      B_Run_XCor = true;

    if (telluric == 3)
      B_MaximaOnly = true;

    blitz::Array<double, 1> D_A1_XCorProf_Out(_trace->getHeight());
    D_A1_XCorProf_Out = 0.;
    if (I_XCorProf > 0){
      s_a1(pppos) = "XCOR_PROF";
      args[pppos] = &I_XCorProf;
      #ifdef __DEBUG_FITS_MKSLITFUNC__
      cout << "args[pppos=" << pppos << "] set to I_XCorProf = " << *(int*)args[pppos] << endl;
      #endif
      pppos++;

      s_a1(pppos) = "XCOR_PROF_OUT";
      args[pppos] = &D_A1_XCorProf_Out;
      pppos++;
    }

    s_a1(pppos) = "SP_OUT";
    pppos++;

    s_a1(pppos) = "STOP";
    pppos++;

    s_a1(pppos) = "MASK";
    pppos++;

    if (telluric > 1)
    {
      s_a1(pppos) = "SKY";
      pppos++;
      s_a1(pppos) = "SP_FIT";
      pppos++;
    }

    blitz::Array<double, 1> D_A1_Errors_SP_Out(1);

    if (ErrorsRead){
      s_a1(pppos) = "ERRORS";
      pppos++;
      s_a1(pppos) = "ERRORS_OUT";
      pppos++;
      s_a1(pppos) = "ERRORS_SP_OUT";
      pppos++;
      if (telluric > 1)
      {
        s_a1(pppos) = "ERR_SKY";
        pppos++;
      }
    }
    s_a1(pppos) = "I_BIN";
    pppos++;
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: s_a1 = " << s_a1 << endl;
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: pppos = " << pppos << endl;
    #endif

    int I_Stop = 0;

    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: s_a1 set to " << s_a1 << endl;
    #endif

//    const blitz::Array<float, 1> xCenters(_xCenters.getData(), blitz::shape(_xCenters.getShape()[0]), blitz::neverDeleteData);
//    blitz::Array<double, 1> D_A1_XCenters(xCenters.size());
//    D_A1_XCenters = ::pfs::drp::stella::math::Double(xCenters);

    #ifdef __DEBUG_MKSLITFUNC__
      if (oldYHigh != _fiberTraceFunction->yHigh){
        string message("FiberTrace ");
        message += to_string(_iTrace) + string("::MkSlitFunc: before calculateSwathWidth: ERROR: oldYHigh(=") + to_string(oldYHigh) + string(") != _fiberTraceFunction->yHigh(=");
        message += to_string(_fiberTraceFunction->yHigh);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    #endif
    ndarray::Array<int, 2, 1> ndArr = calculateBinBoundY(_fiberTraceProfileFittingControl->swathWidth);
    cout << "FiberTrace" << _iTrace << "::MkSlitFunc: ndArr = " << ndArr << endl;
    blitz::Array<int, 2> I_A2_IBinBoundY = utils::ndarrayToBlitz(ndArr);
    I_NBins = I_A2_IBinBoundY.rows();
    cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_A2_IBinBoundY = " << I_A2_IBinBoundY << endl;
    if (I_A2_IBinBoundY(I_A2_IBinBoundY.rows()-1, 1) != (_trace->getHeight()-1)){
      string message("FiberTrace ");
      message += to_string(_iTrace) + string("::MkSlitFunc: after calculateSwathWidth: ERROR: I_A2_IBinBoundY(I_A2_IBinBoundY.rows()-1=");
      message += to_string(I_A2_IBinBoundY.rows()-1) + string(", 1)=") + to_string(I_A2_IBinBoundY(I_A2_IBinBoundY.rows()-1, 1));
      message += string("!= (_trace->getHeight()-1)=") + to_string(_trace->getHeight()-1);
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    #ifdef __DEBUG_MKSLITFUNC__
      if (oldYHigh != _fiberTraceFunction->yHigh){
        string message("FiberTrace ");
        message += to_string(_iTrace) + string("::MkSlitFunc: after calculateSwathWidth: ERROR: oldYHigh(=") + to_string(oldYHigh) + string(") != _fiberTraceFunction->yHigh(=");
        message += to_string(_fiberTraceFunction->yHigh);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    #endif

    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_A2_IBound set to " << I_A2_IBinBoundY << endl;
    #endif
    #ifdef __DEBUG_SLITFUNC_X__
      string boundFN = debugdir + "I_A2_IBoundY.dat";
      ::pfs::drp::stella::utils::WriteArrayToFile(I_A2_IBinBoundY, boundFN, string("ascii"));
    #endif
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: *P_D_A1_YHigh(fiberTraceNumber=" << fiberTraceNumber << ") = " << _fiberTraceFunction->yHigh << endl;
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: *P_D_A1_YLow(fiberTraceNumber=" << fiberTraceNumber << ") = " << _fiberTraceFunction->yLow << endl;
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_NBin = " << I_NBins << endl;
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_NBins = " << I_NBins << endl;
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_A2_IBinBoundY = " << I_A2_IBinBoundY << endl;
    #endif

    D_A3_SFSM.resize(_trace->getHeight(), _trace->getWidth(), I_NBins);

    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: D_A3_SFSM = " << D_A3_SFSM.rows() << " x " << D_A3_SFSM.cols() << " x " << I_NBins << endl;
    #endif
    D_A3_SFSM = 0.;
    D_A2_SP.resize(D_A3_SFSM.rows(), I_NBins);
    D_A2_SP = 0.;
    D_A2_Errors_SP_Out.resize(D_A3_SFSM.rows(), I_NBins);
    D_A2_Errors_SP_Out = 0.;
    D_A2_Sky.resize(D_A3_SFSM.rows(), I_NBins);
    D_A2_Sky = 0.;
    D_A2_ErrSky.resize(D_A3_SFSM.rows(), I_NBins);
    D_A2_ErrSky = 0.;

    /// Center of each bin
    ///  bincen = 0.5*(ibeg + iend)                    ;center of each bin
    blitz::Array<double, 1> D_A1_BinCen(I_NBins);
    D_A1_BinCen = 0.5 * (I_A2_IBinBoundY(blitz::Range::all(), 0) + I_A2_IBinBoundY(blitz::Range::all(),1));
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: D_A1_BinCen set to " << D_A1_BinCen << endl;
    #endif

    /// subpixel range required
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: P_D_A1_XLow = " << _fiberTraceFunction->fiberTraceFunctionControl.xLow << endl;
    #endif

    ///TODO: SF = 7.2 pix -> 0.9-8.1 -> 9 pixels, not 7+1!!!
    I_NXSF = _trace->getWidth();//I_A2_MinCenMax(0,2) - I_A2_MinCenMax(0,0) + 1;// - I_NPixCut_Left - I_NPixCut_Right;
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_NXSF set to " << I_NXSF << endl;
    #endif

    P_D_A1_BLZ->resize(_trace->getHeight());
    (*P_D_A1_BLZ) = 0.;

    D_A2_CCD_Ap.resize(D_A3_SFSM.rows(), D_A3_SFSM.cols());

    D_A2_Err_AllRows.resize(_trace->getHeight(), D_A3_SFSM.cols());
    I_A2_Mask_AllRows.resize(_trace->getHeight(), D_A3_SFSM.cols());
    I_A2_Mask_AllRows = 1;
    D_A2_Err_AllRows = 0.;

    for (int I_IBin = 0; I_IBin < I_NBins; I_IBin++) /// Loop thru sf regions
    {
      if (telluric == 3)
        B_MaximaOnly = true;
      
      #ifdef __DEBUG_MKSLITFUNC__
        if (oldYHigh != _fiberTraceFunction->yHigh){
          string message("FiberTrace ");
          message += to_string(_iTrace) + string("::MkSlitFunc: I_IBin = ") + to_string(I_IBin) + string(": 1. ERROR: oldYHigh(=") + to_string(oldYHigh) + string(") != _fiberTraceFunction->yHigh(=");
          message += to_string(_fiberTraceFunction->yHigh);
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
      #endif
      #ifdef __DEBUG_CHECK_INDICES__
        if (I_IBin >= I_A2_IBinBoundY.rows())
        {
          std::string message("FiberTrace");
          message += std::to_string(_iTrace);
          message += std::string("::MkSlitFunc: I_IBin = ");
          message += std::to_string(I_IBin);
          message += std::string(": ERROR: I_IBin >= I_A2_IBoundY.rows(=");
          message += std::to_string(I_A2_IBinBoundY.rows());
          message += std::string(")");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
      #endif

      I_NR = I_A2_IBinBoundY(I_IBin,1) - I_A2_IBinBoundY(I_IBin,0) + 1; /// Number of rows (Y-Direction)
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): Resizing D_A2_SlitFunc_Im_In: I_A2_IBinBoundY(I_IBin, *) = " << I_A2_IBinBoundY(I_IBin, blitz::Range::all()) << ": I_NR set to " << I_NR << endl;
      #endif

      D_A2_SlitFunc_Im_In.resize(I_NR, I_NXSF);
      D_A2_SlitFunc_Im_In = 0.;
      if (ErrorsRead){
        D_A2_Err.resize(I_NR, I_NXSF);
        D_A2_Err = 0.;
      }
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": D_A2_SlitFunc_Im_In initialized to " << D_A2_SlitFunc_Im_In.rows() << " x " << D_A2_SlitFunc_Im_In.cols() << endl;
      #endif

      I_A2_Msk.resize(I_NR, I_NXSF);
      I_A2_Msk = 1;
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): I_A2_Msk initialized to " << I_A2_Msk.rows() << " x " << I_A2_Msk.cols() << endl;
      #endif

      if (max(I_A2_MaskArray) > 1){
        std::string message("FiberTrace");
        message += std::to_string(_iTrace);
        message += std::string("::MkSlitFunc: I_IBin = ");
        message += std::to_string(I_IBin);
        message += std::string(": ERROR: max(P_I_A2_MaskArray) > 1");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }

      #ifdef __DEBUG_CHECK_INDICES__
        if (I_A2_IBinBoundY(I_IBin, 0) >= I_A2_IBinBoundY(I_IBin, 1)){
          std::string message("FiberTrace");
          message += std::to_string(_iTrace);
          message += std::string("::MkSlitFunc: I_IBin = ");
          message += std::to_string(I_IBin);
          message += std::string(": ERROR: I_A2_IBound(I_IBin, 0)=");
          message += std::to_string(I_A2_IBinBoundY(I_IBin, 0));
          message += std::string(" >= I_A2_IBound(I_IBin, 1)=");
          message += std::to_string(I_A2_IBinBoundY(I_IBin, 1));
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
      #endif
      #ifdef __DEBUG_CHECK_INDICES__
        if (I_A2_IBinBoundY(I_IBin,1) >= _trace->getHeight())
        {
          std::string message("FiberTrace");
          message += std::to_string(_iTrace);
          message += std::string("::MkSlitFunc: I_IBin = ");
          message += std::to_string(I_IBin);
          message += std::string(": ERROR: I_A2_IBound(I_IBin,1)(=");
          message += std::to_string(I_A2_IBinBoundY(I_IBin,1));
          message += std::string(") >= _trace->getHeight(=");
          message += std::to_string(_trace->getHeight());
          message += std::string(")");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
      #endif
      //blitz::Array<double, 1> xCentersTemp(I_A2_IBinBoundY(I_IBin,1) - I_A2_IBinBoundY(I_IBin,0) + 1);
      //xCentersTemp = D_A1_XCenters(blitz::Range(I_A2_IBinBoundY(I_IBin,0), I_A2_IBinBoundY(I_IBin,1)));/// USED TO ADD  + 0.5 HERE;
      ///TODO: check for the 0.5
///      blitz::Array<double, 1> D_A1_XCentersTemp(xCentersTemp.size());
      ndarray::Array<double, 1, 1> xCentersTemp = pfsDRPStella::math::Double(_xCenters);
      
      xCentersTemp[ndarray::view()] -= math::floor(_xCenters, double(0));
      blitz::Array<double, 1> D_A1_XCenMXC = utils::ndarrayToBlitz(xCentersTemp);
      #ifdef __DEBUG_MkSLITFUNC_FILES__
        string xCenMXC = debugdir + "D_A1_XCenMXC.dat";
        utils::WriteArrayToFile(D_A1_XCenMXC, xCenMXC, string("ascii"));
      #endif
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": for (int I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin(=" << I_NBins << "); I_IBin++): D_A1_XCenMXC set to " << D_A1_XCenMXC << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
      #endif
      /// loop thru rows in region
      for (int n = 0; n < I_NR; n++)
      {
        /// column closest to peak
        #ifdef __DEBUG_MKSLITFUNC__
          if (oldYHigh != _fiberTraceFunction->yHigh){
            string message("FiberTrace ");
            message += to_string(_iTrace) + string("::MkSlitFunc: I_IBin = ") + to_string(I_IBin) + string(": n = ") + to_string(n) + string(": 2. ERROR: oldYHigh(=") + to_string(oldYHigh) + string(") != _fiberTraceFunction->yHigh(=");
            message += to_string(_fiberTraceFunction->yHigh);
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
        #ifdef __DEBUG_CHECK_INDICES__
          if (I_A2_IBinBoundY(I_IBin,0) + n >= _trace->getHeight())
          {
            std::string message("FiberTrace");
            message += std::to_string(_iTrace);
            message += std::string("::MkSlitFunc: I_IBin = ");
            message += std::to_string(I_IBin);
            message += std::string(": ERROR: I_A2_IBound(I_IBin,0)(=");
            message += std::to_string(I_A2_IBinBoundY(I_IBin,0));
            message += std::string(") + n(=");
            message += std::to_string(n);
            message += std::string(") >= _trace->getHeight(=");
            message += std::to_string(_trace->getHeight());
            message += std::string(")");
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
        D_A1_SSF.resize(_trace->getWidth());
        D_A1_Err.resize(_trace->getWidth());
        D_A1_Err = D_A2_Errors(I_A2_IBinBoundY(I_IBin,0) + n, blitz::Range::all());
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): D_A1_Err set to " << D_A1_Err << endl;
        #endif
        if (max(abs(D_A1_Err)) < 0.00000001){
          std::string message("FiberTrace");
          message += std::to_string(_iTrace);
          message += std::string("::MkSlitFunc: I_IBin = ");
          message += std::to_string(I_IBin);
          message += std::string(": ERROR: max(abs(D_A1_Err))=");
          message += std::to_string(max(abs(D_A1_Err)));
          message += std::string(" < 0.00000001");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        D_A1_SSF = D_A2_PixArray(I_A2_IBinBoundY(I_IBin,0) + n, blitz::Range::all());
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): KeyWord_Set(NO_SCATTER): D_A1_SSF set to " << D_A1_SSF << endl;
        #endif
        #ifdef __DEBUG_CHECK_INDICES__
          if (I_A2_IBinBoundY(I_IBin,0)+n >= I_A2_MaskArray.rows())
          {
            std::string message("FiberTrace");
            message += std::to_string(_iTrace);
            message += std::string("::MkSlitFunc: I_IBin = ");
            message += std::to_string(I_IBin);
            message += std::string(": ERROR: I_A2_IBound(I_IBin,0)+n(=") + std::to_string(I_A2_IBinBoundY(I_IBin,0)+n);
            message += std::string(") >= I_A2_MaskArray.rows(=") + std::to_string(I_A2_MaskArray.rows()) + std::string(")");
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
        D_A2_SlitFunc_Im_In(n, blitz::Range::all()) = D_A1_SSF;
        #ifdef __DEBUG_CHECK_INDICES__
          if (static_cast<int>(D_A1_SSF.size()) != D_A2_CCD_Ap.cols()){
            std::string message("FiberTrace");
            message += std::to_string(_iTrace);
            message += std::string("::MkSlitFunc: I_IBin = ");
            message += std::to_string(I_IBin);
            message += std::string(": ERROR: D_A1_SSF.size() = ") + std::to_string(D_A1_SSF.size());
            message += std::string(" != D_A2_CCD_Ap.cols() = ") + std::to_string(D_A2_CCD_Ap.cols());
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
        D_A2_CCD_Ap(I_A2_IBinBoundY(I_IBin, 0) - I_A2_IBinBoundY(0, 0) + n, blitz::Range::all()) = D_A1_SSF;

        #ifdef __DEBUG_CHECK_INDICES__
          if (((I_A2_IBinBoundY(I_IBin, 0) + n) < 0) || ((I_A2_IBinBoundY(I_IBin, 0) + n) >= I_A2_MaskArray.rows())){
            std::string message("FiberTrace");
            message += std::to_string(_iTrace);
            message += std::string("::MkSlitFunc: I_IBin = ");
            message += std::to_string(I_IBin);
            message += std::string(": ERROR: ((I_A2_IBound(I_IBin, 0) + n)=") + std::to_string(I_A2_IBinBoundY(I_IBin, 0) + n);
            message += std::string(" < 0) || ((I_A2_IBound(I_IBin, 0) + n) >= I_A2_MaskArray.rows()=");
            message += std::to_string(I_A2_MaskArray.rows()) + std::string(")");
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
        I_A2_Msk(n, blitz::Range::all()) = I_A2_MaskArray(I_A2_IBinBoundY(I_IBin,0) + n, blitz::Range::all());
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): D_A1_XCenters(" << I_A2_IBinBoundY(I_IBin,0) + n << ") = " << D_A1_XCenters(I_A2_IBinBoundY(I_IBin,0) + n) << endl;
          double D_Temp = D_A1_XCenters(I_A2_IBinBoundY(I_IBin,0) + n) + _fiberTraceFunction->fiberTraceFunctionControl.xLow;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): D_A1_XCenters(" << I_A2_IBinBoundY(I_IBin,0)+n << ") + P_D_A1_XLow(" << fiberTraceNumber << ") = " << D_Temp << endl;
          D_Temp = D_A1_XCenters(I_A2_IBinBoundY(I_IBin,0) + n) + _fiberTraceFunction->fiberTraceFunctionControl.xHigh;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): D_A1_XCenters(" << I_A2_IBinBoundY(I_IBin,0)+n << ") + P_D_A1_XHigh(" << fiberTraceNumber << ") = " << D_Temp << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): I_A2_Msk(n, blitz::Range::all()) set to " << I_A2_Msk(n, blitz::Range::all()) << endl;
        #endif
        if (max(I_A2_Msk(n, blitz::Range::all())) > 1){
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": I_A2_Msk(n=" << n << ", *) = " << I_A2_Msk(n, blitz::Range::all()) << endl;
          std::string message("FiberTrace");
          message += std::to_string(_iTrace);
          message += std::string("::MkSlitFunc: I_IBin = ");
          message += std::to_string(I_IBin);
          message += std::string(": ERROR: max(I_A2_Msk) > 1");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        if (ErrorsRead){
          D_A2_Err(n, blitz::Range::all()) = D_A1_Err;
          #ifdef __DEBUG_CHECK_INDICES__
            if (D_A2_Err_AllRows.cols() != static_cast<int>(D_A1_Err.size())){
              std::string message("FiberTrace");
              message += std::to_string(_iTrace);
              message += std::string("::MkSlitFunc: I_IBin = ");
              message += std::to_string(I_IBin);
              message += std::string(": ERROR: D_A2_Err_AllRows.cols(=") + std::to_string(D_A2_Err_AllRows.cols());
              message += std::string(") != D_A1_Err.size(=") + std::to_string(D_A1_Err.size()) + std::string(")");
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
            if ((I_A2_IBinBoundY(I_IBin, 0) + n < 0) || (I_A2_IBinBoundY(I_IBin, 0) + n >= D_A2_Err_AllRows.rows())){
              std::string message("FiberTrace");
              message += std::to_string(_iTrace);
              message += std::string("::MkSlitFunc: I_IBin = ");
              message += std::to_string(I_IBin);
              message += std::string(": ERROR: (I_A2_IBound(I_IBin=") + std::to_string(I_IBin) + std::string(", 0)=");
              message += std::to_string(I_A2_IBinBoundY(I_IBin, 0)) + std::string(" + n = ") + std::to_string(I_A2_IBinBoundY(I_IBin, 0) + n);
              message += std::string(" < 0) || (I_A2_IBound(I_IBin, 0) + n >= D_A2_Err_AllRows.rows()=") + std::to_string(D_A2_Err_AllRows.rows());
              message += std::string(")");
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
          #endif
          D_A2_Err_AllRows(I_A2_IBinBoundY(I_IBin, 0) + n, blitz::Range::all()) = D_A1_Err;
        }
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): D_A2_SlitFunc_Im_In(n, 0) set to " << D_A2_SlitFunc_Im_In(n, 0) << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): D_A2_SlitFunc_Im_In(n, D_A2_SlitFunc_Im_In.cols()-1) set to " << D_A2_SlitFunc_Im_In(n, D_A2_SlitFunc_Im_In.cols()-1) << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): D_A2_SlitFunc_Im_In(n, *) set to " << D_A2_SlitFunc_Im_In(n, blitz::Range::all()) << endl;
        #endif
        #ifdef __DEBUG_CHECK_INDICES__
          if (n >= I_A2_Msk.rows())
          {
            std::string message("FiberTrace");
            message += std::to_string(_iTrace);
            message += std::string("::MkSlitFunc: I_IBin = ");
            message += std::to_string(I_IBin);
            message += std::string(": ERROR: n(=") + std::to_string(n) + std::string(") >= I_A2_Msk.rows(=") + std::to_string(I_A2_Msk.rows());
            message += std::string(")");
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
      } /// for (int n = 0; n < I_NR; n++)
      #ifdef __DEBUG_MKSLITFUNC__
        if (oldYHigh != _fiberTraceFunction->yHigh){
          string message("FiberTrace ");
          message += to_string(_iTrace) + string("::MkSlitFunc: I_IBin = ") + to_string(I_IBin) + string(": end for(n) ERROR: oldYHigh(=") + to_string(oldYHigh) + string(") != _fiberTraceFunction->yHigh(=");
          message += to_string(_fiberTraceFunction->yHigh);
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
      #endif
      if (max(I_A2_Msk) > 1){
        cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": A. I_A2_Msk = " << I_A2_Msk << endl;
        std::string message("FiberTrace");
        message += std::to_string(_iTrace);
        message += std::string("::MkSlitFunc: I_IBin = ");
        message += std::to_string(I_IBin);
        message += std::string(": ERROR: max(I_A2_Msk) = ") + std::to_string(max(I_A2_Msk)) + std::string(" > 1");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }

      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": end for: D_A2_Err = " << D_A2_Err << endl;
      #endif
      D_A2_Err_Temp.resize(D_A2_Err.rows(), D_A2_Err.cols());
      D_A2_Err_Temp = D_A2_Err;
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": end for: D_A2_SlitFunc_Im_In(*,0) set to " << D_A2_SlitFunc_Im_In(blitz::Range::all(),0) << endl;
        cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": end for: D_A2_SlitFunc_Im_In(*,ncols-1=" << D_A2_SlitFunc_Im_In.cols()-1 << ") set to " << D_A2_SlitFunc_Im_In(blitz::Range::all(),D_A2_SlitFunc_Im_In.cols()-1) << endl;
        cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": end for: I_A2_Msk(*,0) set to " << I_A2_Msk(blitz::Range::all(), 0) << endl;
        cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": end for: I_A2_Msk(*,ncols-1=" << I_A2_Msk.cols()-1 << ") set to " << I_A2_Msk(blitz::Range::all(), I_A2_Msk.cols()-1) << endl;
        string S_TempA(DEBUGDIR);
        S_TempA += string("I_A2_Msk.fits");
        ::pfs::drp::stella::utils::WriteFits(&I_A2_Msk, S_TempA);
        S_TempA = string(DEBUGDIR) + string("D_A2_SlitFunc_Im_In.fits");
        ::pfs::drp::stella::utils::WriteFits(&D_A2_SlitFunc_Im_In, S_TempA);
      #endif

      D_A2_SlitFuncOrig.resize(D_A2_SlitFunc_Im_In.rows(), D_A2_SlitFunc_Im_In.cols());
      /// Backup original Slit Function
      D_A2_SlitFuncOrig = D_A2_SlitFunc_Im_In;
      for (int p=0; p<D_A2_SlitFunc_Im_In.rows(); p++)
      {
        /// Set MySF to 0. where < 3.*(-RON)
        D_A2_SlitFunc_Im_In(p, blitz::Range::all()) = blitz::where(D_A2_SlitFunc_Im_In(p, blitz::Range::all()) < (3. * (0. - _fiberTraceProfileFittingControl->ccdReadOutNoise)), 0., D_A2_SlitFunc_Im_In(p, blitz::Range::all()));
      }
      if (telluric == 1)///Piskunov
      {
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": telluric == 1" << endl;
        #endif
        D_A1_Sky.resize(D_A2_SlitFunc_Im_In.rows());
        D_A1_ErrSky.resize(D_A2_SlitFunc_Im_In.rows());
        D_A1_Tel.resize(D_A2_SlitFunc_Im_In.cols());
        D_A1_Tel = blitz::sum(D_A2_SlitFunc_Im_In(j, i), j);
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: D_A2_SlitFunc_Im_In(*,0) = " << D_A2_SlitFunc_Im_In(blitz::Range::all(),0) << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: D_A1_Tel set to " << D_A1_Tel.size() << ": " << D_A1_Tel << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: D_A1_Tel set to " << D_A1_Tel << endl;
        #endif
        I_A1_ITel.resize(D_A1_Tel.size());
        I_A1_ITel = blitz::where(D_A1_Tel <= (max(D_A1_Tel) - min(D_A1_Tel)) / 100. + min(D_A1_Tel), 1, 0);
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: I_A1_ITel set to " << I_A1_ITel << endl;
        #endif
        D_A2_Tel.resize(D_A2_SlitFunc_Im_In.rows(), blitz::sum(I_A1_ITel));
        D_A2_ErrTel.resize(D_A2_SlitFunc_Im_In.rows(), blitz::sum(I_A1_ITel));
        pos = 0;
        for (int o = 0; o < static_cast<int>(I_A1_ITel.size()); o++)
        {
          if (I_A1_ITel(o) == 1)
          {
            D_A2_Tel(blitz::Range::all(), pos) = D_A2_SlitFunc_Im_In(blitz::Range::all(), o);
            D_A2_ErrTel(blitz::Range::all(), pos) = D_A2_Err(blitz::Range::all(), o);
            pos++;
          }
        }
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: D_A2_Tel set to " << D_A2_Tel << endl;
        #endif

        D_A1_SC.resize(I_NR);
        D_A1_SC = 0.;

        if (ErrorsRead){
          I_A1_ErrInd.resize(D_A2_Tel.cols());
          D_A1_ErrTel.resize(I_NR);
          D_A1_ErrTel = 0.;
        }

        for (int o = 0; o < I_NR; o++)
        {
          #ifdef __DEBUG_CHECK_INDICES__
            if (o >= D_A2_Tel.rows())
            {
              std::string message("FiberTrace");
              message += std::to_string(_iTrace);
              message += std::string("::MkSlitFunc: I_IBin = ");
              message += std::to_string(I_IBin);
              message += std::string(": ERROR: o(=") + std::to_string(o) + std::string(") >= D_A2_Tel.rows(=") + std::to_string(D_A2_Tel.rows());
              message += std::string(")");
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
          #endif
          if (ErrorsRead){
            D_A1_ErrTelSub.resize(D_A2_ErrTel.cols());
            D_A1_ErrTelSub = D_A2_ErrTel(o, blitz::Range::all());
            PP_Args_Median[1] = &D_A1_ErrTelSub;
          }
          D_A1_SC(o) = ::pfs::drp::stella::math::Median(D_A2_Tel(o, blitz::Range::all()), S_A1_Args_Median, PP_Args_Median);
          if (ErrorsRead){
            D_A1_ErrSky(o) = D_Val_ErrOutMedian;
            D_A2_Err(o, blitz::Range::all()) += D_Val_ErrOutMedian;
            D_A2_ErrTel(o, blitz::Range::all()) = D_A1_ErrTelSub;
          }
        }
        D_A1_Sky = D_A1_SC;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: D_A1_SC set to " << D_A1_SC << endl;
        #endif

        blitz::Array<double, 1> tempDblVecArr = ::pfs::drp::stella::math::Replicate(1., _trace->getWidth());// I_A2_MinCenMax(0, 2) - I_A2_MinCenMax(0, 0) + 1);
        #ifdef __DEBUG_CHECK_INDICES__
          if (D_A2_SlitFunc_Im_In.cols() != static_cast<int>(tempDblVecArr.size())){
            std::string message("FiberTrace");
            message += std::to_string(_iTrace);
            message += std::string("::MkSlitFunc: I_IBin = ");
            message += std::to_string(I_IBin);
            message += std::string(": ERROR: D_A2_SlitFunc_Im_In.cols()=") + std::to_string(D_A2_SlitFunc_Im_In.cols());
            message += std::string(" != tempDblVecArr.size()=") + std::to_string(tempDblVecArr.size());
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
        blitz::Array<double, 2> *p_d2mata = ::pfs::drp::stella::math::VecArrACrossB(D_A1_SC, tempDblVecArr);

        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc:  telluric = 1: I_IBin = " << I_IBin << ": I_NR = " << I_NR << ", tempDblVecArr.size() = " << tempDblVecArr.size() << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: D_A1_SC = " << D_A1_SC << ", VecArrACrossB(D_A1_SC, tempDblVecArr) = " << *p_d2mata << endl;
        #endif
        #ifdef __DEBUG_CHECK_INDICES__
          if (D_A2_SlitFunc_Im_In.rows() != static_cast<int>(D_A1_SC.size()))
          {
            std::string message("FiberTrace");
            message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
            message += std::string(": ERROR: D_A2_SlitFunc_Im_In.rows(=") + std::to_string(D_A2_SlitFunc_Im_In.rows());
            message += std::string(") != D_A1_SC.size(=") + std::to_string(D_A1_SC.size()) + std::string(")");
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          if (D_A2_SlitFunc_Im_In.cols() != static_cast<int>(tempDblVecArr.size()))
          {
            std::string message("FiberTrace");
            message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
            message += std::string(": ERROR: D_A2_SlitFunc_Im_In.cols(=") + std::to_string(D_A2_SlitFunc_Im_In.cols());
            message += std::string(") != tempDblVecArr.size(=") + std::to_string(tempDblVecArr.size()) + std::string(")");
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: p_d2mata = " << *p_d2mata << endl;
        #endif
        D_A2_SlitFunc_Im_In -= (*p_d2mata);
        delete p_d2mata;
//        #ifdef __DEBUG_MKSLITFUNC__
//          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": KeyWord_Set(TELLURIC): D_A2_SlitFunc_Im_In set to " << D_A2_SlitFunc_Im_In << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
//          CFits* p_tempfits = new CFits();
//          string S_TempB = debugdir;
//          S_TempB += "MkSlitFunc_TELLURIC_D_A2_SlitFunc_Im_In.fits";
//          p_tempfits->setFileName(S_TempB);
//          p_tempfits->SetNRows(D_A2_SlitFunc_Im_In.rows());
//          p_tempfits->SetNCols(D_A2_SlitFunc_Im_In.cols());
//          p_tempfits->GetPixArray() = D_A2_SlitFunc_Im_In;
//          p_tempfits->WriteArray();
//          delete(p_tempfits);
//        #endif

        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": I_NR(=" << I_NR << endl;
        #endif
      } /// end if (telluric == 1)
      else if (telluric == 3){

        I_A2_Mask_Tel.resize(I_A2_Msk.rows(), I_A2_Msk.cols());
        I_A2_Mask_Tel = I_A2_Msk;
        if (max(I_A2_Mask_Tel) > 1){
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": 0. I_A2_Mask_TelTemp = " << I_A2_Mask_TelTemp << endl;
          std::string message("FiberTrace");
          message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
          message += std::string(": 0. ERROR: max(I_A2_Mask_Tel) = ");
          message += std::to_string(max(I_A2_Mask_Tel)) + std::string(" > 1");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }

        I_A1_UseRow_Tel.resize(I_A2_Msk.rows());
        I_A1_UseRow_Tel = ::pfs::drp::stella::math::IndGenArr(I_A2_Msk.rows());

        D_A2_TempIm.resize(D_A2_SlitFunc_Im_In.rows(), D_A2_SlitFunc_Im_In.cols());

        blitz::Array<int, 1> I_A1_MaxPos(D_A2_SlitFunc_Im_In.rows());
        I_A1_MaxPos = 0;
        blitz::Array<int, 1> I_A1_MaxPosCol(D_A2_SlitFunc_Im_In.cols());
        I_A1_MaxPosCol = 0;
        blitz::Array<int, 1> *P_I_A1_MaxPosColInd;
        int I_NMax = 0;

        D_A1_SFMax.resize(D_A2_SlitFunc_Im_In.rows());
        for (int p=0; p<D_A2_SlitFunc_Im_In.rows(); p++)
        {
          #ifdef __DEBUG_TELLURIC__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: for(p(=" << p << "); p<D_A2_SlitFunc_Im_In.rows()=" << D_A2_SlitFunc_Im_In.rows() << "; p++): D_A2_SlitFunc_Im_In(p,*) = " << D_A2_SlitFunc_Im_In(p,blitz::Range::all()) << endl;
          #endif

          /// Normalize rows of D_A2_SlitFunc_Im_In to 1.
          if (fabs(blitz::sum(D_A2_SlitFunc_Im_In(p, blitz::Range::all()))) < 0.00000000000000001)
            D_A2_SlitFunc_Im_In(p, blitz::Range::all()) = 1.;
          D_A2_SlitFunc_Im_In(p,blitz::Range::all()) /= blitz::sum(D_A2_SlitFunc_Im_In(p,blitz::Range::all()));
          #ifdef __DEBUG_TELLURIC__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: for(p(=" << p << ")...): D_A2_SlitFunc_Im_In(p,*) = " << D_A2_SlitFunc_Im_In(p,blitz::Range::all()) << endl;
          #endif

          /// Get maximum of D_A2_SlitFunc_Im_In for every row
          D_A1_SFMax(p) = max(D_A2_SlitFunc_Im_In(p,blitz::Range::all()));

          /// Find MaxPos
          I_A1_MaxPosCol = blitz::where(fabs(D_A2_SlitFunc_Im_In(p,blitz::Range::all()) - D_A1_SFMax(p)) < 0.000001,1,0);
          P_I_A1_MaxPosColInd = ::pfs::drp::stella::math::GetIndex(I_A1_MaxPosCol, I_NMax);
          I_A1_MaxPos(p) = (*P_I_A1_MaxPosColInd)(0);

          #ifdef __DEBUG_TELLURIC__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: for(p(=" << p << ")...): D_A1_SFMax(p) = " << D_A1_SFMax(p) << endl;
          #endif
        }/// end for (int p=0; p<D_A2_SlitFunc_Im_In.rows(); p++)
        int I_MaxPos = ::pfs::drp::stella::math::Median(I_A1_MaxPos);
        for (int p=0; p<D_A2_SlitFunc_Im_In.rows(); p++){
          D_A1_SFMax(p) = max(D_A2_SlitFunc_Im_In(p,blitz::Range(I_MaxPos-1,I_MaxPos+1)));
        }
        #ifdef __DEBUG_TELLURIC__
          string S_MySF(debugdir);
          string S_MySFTemp;
          S_MySF += "MaxOnly_D_A2_SlitFunc_Im_In_Norm_IBin";
          S_MySFTemp = S_MySF.itoa(I_IBin);
          S_MySF += S_MySFTemp;
          S_MySF += "Tel3.fits";
          ::pfs::drp::stella::utils::WriteFits(&D_A2_SlitFunc_Im_In, S_MySF);
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: File " << S_MySF << " written" << endl;
        #endif
        /// /////////////////////////////////////////////////////////////////////////

        //      D_A2_SlitFunc_Im_In = D_A2_SlitFunc_Im_In_Max;

        /// /////////////////////////////////////////////////////////////////////////

        /// TODO: ONLY TAKE HIGHEST VALUES, NOT MIDDLE ONES?!? <- Already did median filtering!
        /// --- remove elements from D_A1_SFMax which are outside the median value +/- 2sigma
        I_A1_UseRow_Tel_AllRows.resize(D_A2_SlitFunc_Im_In.rows());
        I_A1_UseRow_Tel_AllRows = ::pfs::drp::stella::math::IndGenArr(D_A2_SlitFunc_Im_In.rows());
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: while: Median(D_A1_SFMax) = " << ::pfs::drp::stella::math::Median(D_A1_SFMax) << endl;
        #endif
        I_A1_IndA.resize(D_A1_SFMax.size());


        /** ************************************/


        double D_MedianSFMax = ::pfs::drp::stella::math::Median(D_A1_SFMax);
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: D_A1_SFMax = " << D_A1_SFMax << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: D_MedianSFMax = " << D_MedianSFMax << endl;
        #endif
        I_A1_IndA = blitz::where((D_A1_SFMax > D_MedianSFMax) & (D_A1_SFMax < 1.5 * D_MedianSFMax),1,0);


        /** ************************************/



        P_I_A1_Ind = ::pfs::drp::stella::math::GetIndex(I_A1_IndA, nind_temp);
        #ifdef __DEBUG_TELLURIC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: nind_temp = " << nind_temp << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: while: I_A1_IndA set to " << I_A1_IndA << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: while: *P_I_A1_Ind set to " << *P_I_A1_Ind << endl;
        #endif
        blitz::Array<double, 2> D_A2_SFMax(2,2);
        ::pfs::drp::stella::math::GetSubArrCopy(D_A2_SlitFunc_Im_In,
                                          *P_I_A1_Ind,
                                          0,
                                          D_A2_SFMax);
        blitz::Array<double, 1> D_A1_SFMa(D_A2_SFMax.rows());
        for (int i_c=0; i_c<D_A2_SlitFunc_Im_In.cols(); i_c++){
          D_A1_SFMa = ::pfs::drp::stella::math::MedianVec(D_A2_SFMax(blitz::Range::all(), i_c),5);
          D_A2_SFMax(blitz::Range::all(), i_c) = D_A1_SFMa;
        }
        for (int inde=0; inde<static_cast<int>(P_I_A1_Ind->size()); inde++){
          D_A2_SlitFunc_Im_In((*P_I_A1_Ind)(inde), blitz::Range::all()) = D_A2_SFMax(inde, blitz::Range::all());
        }
        #ifdef __DEBUG_TELLURIC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: while: D_A1_SFMax set to " << D_A1_SFMax << endl;
        #endif

        I_A1_UseRow_TelTemp.resize(I_A1_UseRow_Tel.size());
        I_A1_UseRow_TelTemp = I_A1_UseRow_Tel;
        I_A1_UseRow_Tel.resize(P_I_A1_Ind->size());
        I_A2_Mask_TelTemp.resize(I_A2_Mask_Tel.rows(), I_A2_Mask_Tel.cols());
        I_A2_Mask_TelTemp = 0;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": 1. I_A2_Mask_Tel = " << I_A2_Mask_Tel << endl;
        #endif
        if (max(I_A2_Mask_Tel) > 1){
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": 1A. I_A2_Mask_TelTemp = " << I_A2_Mask_TelTemp << endl;
          std::string message("FiberTrace");
          message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
          message += std::string(": 1A. ERROR: max(I_A2_Mask_Tel) = ") + std::to_string(max(I_A2_Mask_Tel)) + std::string(" > 1");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        for (int nnn=0; nnn<static_cast<int>(P_I_A1_Ind->size()); nnn++){
          I_A2_Mask_TelTemp((*P_I_A1_Ind)(nnn),blitz::Range::all()) = I_A2_Mask_Tel((*P_I_A1_Ind)(nnn), blitz::Range::all());
          #ifdef __DEBUG_CHECK_INDICES__
            if ((*P_I_A1_Ind)(nnn) >= static_cast<int>(I_A1_UseRow_Tel_AllRows.size())){
              std::string message("FiberTrace");
              message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
              message += std::string(": ERROR: (*P_I_A1_Ind)(nnn=") + std::to_string(nnn) + std::string(")=") + std::to_string((*P_I_A1_Ind)(nnn));
              message += std::string(") = ") + std::to_string((*P_I_A1_Ind)(nnn)) + std::string(" >= I_A1_UseRow_Tel_AllRows.size()=");
              message += std::to_string(I_A1_UseRow_Tel_AllRows.size());
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
          #endif
          I_A1_UseRow_Tel(nnn) = I_A1_UseRow_Tel_AllRows((*P_I_A1_Ind)(nnn));
        }
        #ifdef __DEBUG_MKSLITFUNC__
          string tempC = debugdir + "P_I_A1_Ind_1.dat";
          ::pfs::drp::stella::utils::WriteArrayToFile(*P_I_A1_Ind, tempC, string("ascii"));
        #endif

        blitz::Array<double, 1> D_A1_SFMax_SumRows(D_A2_SlitFunc_Im_In.cols());
        blitz::Array<double, 2> D_A2_SlitFunc_Im_In_Times_Mask(D_A2_SlitFunc_Im_In.rows(), D_A2_SlitFunc_Im_In.cols());
        D_A2_SlitFunc_Im_In_Times_Mask = D_A2_SlitFunc_Im_In * I_A2_Mask_TelTemp;
        D_A1_SFMax_SumRows = blitz::sum(D_A2_SlitFunc_Im_In_Times_Mask(j,i),j);
        #ifdef __DEBUG_SLITFUNC_FILES__
          string S_SFMax_SumRows("SFMax_SumRows_");
          if (B_MaximaOnly){
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: B_MaximaOnly: D_A1_SFMax_SumRows = " << D_A1_SFMax_SumRows << endl;
            S_SFMax_SumRows += "MaxOnly_";
          }
          else{
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: !B_MaximaOnly: D_A1_SFMax_SumRows = " << D_A1_SFMax_SumRows << endl;
          }
          S_SFMax_SumRows += "IBin" + to_string(I_IBin) + "_IRunTel" + to_string(I_Run_Tel) + ".fits";
          ::pfs::drp::stella::utils::WriteFits(&D_A1_SFMax_SumRows, S_SFMax_SumRows);
        #endif

        I_A1_Ind_Last.resize(P_I_A1_Ind->size());
        I_A1_Ind_Last = (*P_I_A1_Ind);
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": 2. I_A2_Mask_TelTemp = " << I_A2_Mask_TelTemp << endl;
        #endif
        I_A2_Mask_Tel = I_A2_Mask_TelTemp;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": 3. I_A2_Mask_Tel = " << I_A2_Mask_Tel << endl;
        #endif
        if (max(I_A2_Mask_Tel) > 1){
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": 3A. I_A2_Mask_TelTemp = " << I_A2_Mask_TelTemp << endl;
          std::string message("FiberTrace");
          message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
          message += std::string(": 3A. ERROR: max(I_A2_Mask_Tel) = ") + std::to_string(max(I_A2_Mask_Tel)) + std::string(" > 1");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        #ifdef __DEBUG_MKSLITFUNC__
          D_A2_TempIm.resize(D_A2_SlitFunc_Im_In.rows(), D_A2_SlitFunc_Im_In.cols());
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": D_A2_SlitFunc_Im_In = " << D_A2_SlitFunc_Im_In << endl;
          D_A2_TempIm = D_A2_SlitFunc_Im_In * I_A2_Mask_Tel;
          string S_TempC = debugdir + "D_A2_SlitFunc_Im_In_Times_Mask_1.fits";
          ::pfs::drp::stella::utils::WriteFits(&D_A2_TempIm, S_TempC);
          D_A2_TempIm.resize(1,1);
        #endif
        delete(P_I_A1_Ind);
      }/// end else if (Telluric == 3)
      #ifdef __DEBUG_MKSLITFUNC__
        if (oldYHigh != _fiberTraceFunction->yHigh){
          string message("FiberTrace ");
          message += to_string(_iTrace) + string("::MkSlitFunc: I_IBin = ") + to_string(I_IBin) + string(": after if (Telluric == 1 or 3): 2. ERROR: oldYHigh(=") + to_string(oldYHigh) + string(") != _fiberTraceFunction->yHigh(=");
          message += to_string(_fiberTraceFunction->yHigh);
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
      #endif
      I_RunMax = 1;
      if (telluric > 2){
        I_RunMax = _fiberTraceProfileFittingControl->maxIterSky;
        I_MaxIterSig = 0;
        B_MaximaOnly = true;
      }
      I_Run_Tel = 0;
      D_A2_SlitFunc_Im_In_Tel.resize(D_A2_SlitFunc_Im_In.rows(), D_A2_SlitFunc_Im_In.cols());
      D_A2_SlitFunc_Im_In_Tel = D_A2_SlitFunc_Im_In;
      D_A1_Sky_Temp.resize(D_A2_SlitFunc_Im_In.rows());
      D_A1_Sky_Temp = 0.;
      if (telluric != 1){
        D_A1_Sky.resize(D_A1_Sky_Temp.size());
        D_A1_Sky = 0.;
      }
      D_A2_SFOld.resize(D_A2_SlitFunc_Im_In.rows(), D_A2_SlitFunc_Im_In.cols());
      D_A2_SFOld = 0.;
      D_SFDev = 0.;
      do{
        pppos = 0;
        ///      if ((I_Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "FIBERTRACENUMBER")) >= 0)
        ///        s_a1(pppos) = "FIBERTRACENUMBER";
        ///      if (_fiberTraceProfileFittingControl->xCorProf > 0)
        ///        s_a1(pppos) = "XCOR_PROF_OUT";
        ///      s_a1(pppos) = "SP_OUT";
        ///      s_a1(pppos) = "STOP";
        ///      s_a1(pppos) = "MASK";
        //       if (_fiberTraceProfileFittingControl->telluric > 1)
        //       {
        //         s_a1(pppos) = "SKY";
        //         pppos++;
        //         s_a1(pppos) = "SP_FIT";
        //         pppos++;
        //       }
        //    if (ErrorsRead){
        //      s_a1(pppos) = "ERRORS";
        //      pppos++;
        //      s_a1(pppos) = "ERRORS_OUT";
        //      pppos++;
        //      s_a1(pppos) = "ERRORS_SP_OUT";
        //      pppos++;
        //      if (I_Telluric > 1)
        //      {
        //        s_a1(pppos) = "ERR_SKY";
        //        pppos++;
        //      }
        //    }
        //    s_a1(pppos) = "I_BIN";
        //    pppos++;
        //    s_a1(pppos) = "DEBUGFILES_SUFFIX";
        /////////////////////////////        if (I_Telluric > 0)
/////////////////////////////          pppos++;
        if ((I_Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "FIBERTRACENUMBER")) >= 0)
          pppos++;
        if (I_XCorProf > 0){
////////////////////////////          pppos++;
          pppos++;
          pppos++;
        }

//        args[pppos] = &D_XLow;
//        pppos++;

        D_A1_SP_Out.resize(D_A2_SlitFunc_Im_In_Tel.rows());
        D_A1_SP_Out = 0.;
        args[pppos] = &D_A1_SP_Out;
        pppos++;

        I_Stop = 0;
        args[pppos] = &I_Stop;
        pppos++;

        I_A2_Mask.resize(I_A2_Msk.rows(), I_A2_Msk.cols());
        I_A2_Mask_Temp.resize(I_A2_Msk.rows(), I_A2_Msk.cols());
        if (telluric < 3){
          I_A2_Mask_Temp = I_A2_Msk;
        }
        else{
          I_A2_Mask_Temp.resize(I_A2_Mask_Tel.rows(), I_A2_Mask_Tel.cols());
          I_A2_Mask_Temp = I_A2_Mask_Tel;
        }
        I_A2_Mask = I_A2_Mask_Temp;
        I_A2_MaskApTemp.resize(I_A2_Mask.rows(), I_A2_Mask.cols());
        I_A2_MaskApTemp = I_A2_Mask_Temp;
        args[pppos] = &I_A2_Mask_Temp;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": I_A2_Mask_Temp = " << I_A2_Mask_Temp << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": args[" << pppos << "] set to " << *((blitz::Array<int, 2>*)(args[pppos])) << endl;
        #endif
        if (max(I_A2_Mask_Temp) > 1){
          std::string message("FiberTrace");
          message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
          message += std::string(": ERROR: A. max(I_A2_Mask_Temp) > 1");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        pppos++;

        if (telluric > 1)
        {
          D_A1_Sky_Temp.resize(D_A2_SlitFunc_Im_In_Tel.rows());
          D_A1_Sky_Temp = 0.;
          args[pppos] = &D_A1_Sky_Temp;
          pppos++;
          D_A1_SPFit.resize(D_A2_SlitFunc_Im_In_Tel.rows());
          D_A1_SPFit = 0.;
          args[pppos] = &D_A1_SPFit;
          pppos++;
        }

        if (ErrorsRead){
          args[pppos] = &D_A2_Err;
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": D_A2_Err = " << D_A2_Err << endl;
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": args[" << pppos << "] set to " << *((blitz::Array<double, 2>*)(args[pppos])) << endl;
          #endif
          pppos++;

          D_A1_ErrOut.resize(D_A2_SlitFunc_Im_In_Tel.rows());
          D_A1_ErrOut = 0.;
          args[pppos] = &D_A1_ErrOut;
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": D_A1_ErrOut = " << D_A1_ErrOut << endl;
          #endif
          pppos++;

          D_A1_Errors_SP_Out.resize(D_A2_SlitFunc_Im_In_Tel.rows());
          D_A1_Errors_SP_Out = 0.;
          args[pppos] = &D_A1_Errors_SP_Out;
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": D_A1_Errors_SP_Out = " << D_A1_Errors_SP_Out << endl;
          #endif
          pppos++;

          if (telluric > 1)
          {
            D_A1_ErrSky.resize(D_A2_SlitFunc_Im_In_Tel.rows());
            D_A1_ErrSky = 0.;
            args[pppos] = &D_A1_ErrSky;
            pppos++;
          }
        }

        args[pppos] = &I_IBin;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": args[" << pppos << "] set to " << *((int*)(args[pppos])) << endl;
        #endif
        pppos++;

        s_a1(pppos) = "DEBUGFILES_SUFFIX";
        S_SF_DebugFilesSuffix = "_trace";
        if (fiberTraceNumber < 100)
          S_SF_DebugFilesSuffix += "0";
        if (fiberTraceNumber < 10)
          S_SF_DebugFilesSuffix += "0";
        S_SF_DebugFilesSuffix += std::to_string(fiberTraceNumber) + "_IBin";
        if (I_IBin < 10)
          S_SF_DebugFilesSuffix += "0";
        S_SF_DebugFilesSuffix += std::to_string(I_IBin) + "_IRunTel" + std::to_string(I_Run_Tel) + "_Tel" + std::to_string(telluric);
        if (B_MaximaOnly)
          S_SF_DebugFilesSuffix += "_MaxOnly";
        args[pppos] = &S_SF_DebugFilesSuffix;
        pppos++;

        cout << "FiberTrace" << _iTrace << "::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ", I_IBin = " << I_IBin << ": starting SlitFunc" << endl;
        #ifdef __DEBUG_MKSLITFUNC__
          if (I_IBin < 100)
          {
            S_TempNum = "0";
          }
          if (I_IBin < 10)
          {
            S_TempNum = "00";
          }
          int tempm = I_IBin;
          int tempmt = I_IBin / 10;
          if (tempmt > 0)
          {
            S_TempNum += to_string(tempmt);
            tempm -= 10 * tempmt;
          }
          S_TempNum += to_string(tempm);

          string S_FName_SF(DEBUGDIR);
          S_FName_SF += string("SlitFunc_SFIn_in_");
          if (B_MaximaOnly)
            S_FName_SF += string("MaxOnly_");
          S_FName_SF += string("IBin") + S_TempNum + string("_Tel") + to_string(telluric) + string("_IRunTel") + to_string(I_Run_Tel) + string(".fits");
          ::pfs::drp::stella::utils::WriteFits(&D_A2_SlitFunc_Im_In_Tel,  S_FName_SF);
        #endif

        blitz::Array<double, 2> D_A2_SlitFunc_Im_In_SumRows(D_A2_SlitFunc_Im_In.rows(), D_A2_SlitFunc_Im_In.cols());
        D_A2_SlitFunc_Im_In_SumRows = D_A2_SlitFunc_Im_In_Tel * I_A2_Mask_Temp;
        #ifdef __DEBUG_MkSLITFUNC_FILES__
          blitz::Array<double, 1> D_A1_SlitFunc_Im_In_SumRows(D_A2_SlitFunc_Im_In_SumRows.cols());
          D_A1_SlitFunc_Im_In_SumRows = blitz::sum(D_A2_SlitFunc_Im_In_SumRows(j,i),j);
          string S_Sum = "SlitFuncImInTel_SumCols" + S_SF_DebugFilesSuffix + ".fits";
          ::pfs::drp::stella::utils::WriteFits(&D_A1_SlitFunc_Im_In_SumRows, S_Sum);
        #endif
        if (!this->SlitFunc(D_A2_SlitFunc_Im_In_Tel,
                            I_MaxIterSig,
                            D_A1_XCenMXC,
                            D_A1_SP,
                            D_A2_SFSM,
                            s_a1,
                            args))
        {
          std::string message("FiberTrace");
          message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
          message += std::string(": ERROR: SlitFunc returned FALSE!");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        #ifdef __DEBUG_MKSLITFUNC__
          if (oldYHigh != _fiberTraceFunction->yHigh){
            string message("FiberTrace ");
            message += to_string(_iTrace) + string("::MkSlitFunc: I_IBin = ") + to_string(I_IBin) + string(": after SlitFunc: 2. ERROR: oldYHigh(=") + to_string(oldYHigh) + string(") != _fiberTraceFunction->yHigh(=");
            message += to_string(_fiberTraceFunction->yHigh);
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
        #ifdef __DEBUG_SLITFUNC_FILES__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ", I_IBin = " << I_IBin << ": D_A1_SP = " << D_A1_SP << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ", I_IBin = " << I_IBin << ": D_A1_SP_Out = " << D_A1_SP_Out << endl;
        #endif
        if (ErrorsRead){
          #ifdef __DEBUG_SLITFUNC_FILES__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ", I_IBin = " << I_IBin << ": D_A1_Errors_SP_Out = " << D_A1_Errors_SP_Out << endl;
          #endif
          blitz::Array<double, 1> D_A1_SNR(D_A1_SP.size());
          D_A1_SNR = D_A1_SP_Out / D_A1_Errors_SP_Out;
          #ifdef __DEBUG_SLITFUNC_FILES__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ", I_IBin = " << I_IBin << ": D_A1_SNR = " << D_A1_SNR << endl;
          #endif
        }
        #ifdef __DEBUG_MKSLITFUNC__
          blitz::Array<double, 2> D_A2_MaskTimesSlitFunc(I_A2_Mask_Temp.rows(), I_A2_Mask_Temp.cols());
          D_A2_MaskTimesSlitFunc = D_A2_SlitFunc_Im_In_Tel * I_A2_Mask_Temp;
          string S_MaskOut(DEBUGDIR);
          S_MaskOut += string("SlitFuncTimesMask_just_after_SlitFunc") + S_SF_DebugFilesSuffix + string(".fits");
          if (!::pfs::drp::stella::utils::WriteFits(&D_A2_MaskTimesSlitFunc, S_MaskOut)){
            std::string message("FiberTrace");
            message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
            message += std::string(": ERROR: WriteFits(I_A2_Mask_Temp, ") + S_MaskOut + std::string(") returned FALSE");
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
        I_A2_Mask = I_A2_Mask_Temp;
        I_A2_Mask_Temp = I_A2_MaskApTemp;
        if (!B_MaximaOnly){
          #ifdef __DEBUG_SLITFUNC_FILES__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: !B_MaximaOnly: After SlitFunc: I_A2_Mask = " << I_A2_Mask <<  endl;
          #endif
        }

        cout << "FiberTrace" << _iTrace << "::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ", I_IBin = " << I_IBin << ": SlitFunc ready" << endl;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_IBin = " << I_IBin << ": D_A2_SFSM = " << D_A2_SFSM << endl;
          if (ErrorsRead)
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_IBin = " << I_IBin << ": After SlitFunc: D_A2_Err = " << D_A2_Err << endl;
        #endif
        if (!B_MaximaOnly){
          #ifdef __DEBUG_CHECK_INDICES__
            if (D_A3_SFSM.cols() != D_A2_SFSM.cols()){
              std::string message("FiberTrace");
              message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
              message += std::string(": ERROR: D_A3_SFSM.cols(=") + std::to_string(D_A3_SFSM.cols()) + string(") != D_A2_SFSM.cols(=");
              message += to_string(D_A2_SFSM.cols()) + string(")");
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
            if (I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(0,0) - (I_A2_IBinBoundY(I_IBin, 0) - I_A2_IBinBoundY(0,0)) + 1 != D_A2_SFSM.rows()){
              std::string message("FiberTrace");
              message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
              message += std::string(": ERROR: I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(0,0) - (I_A2_IBinBoundY(I_IBin, 0) - I_A2_IBinBoundY(0,0)) + 1 (=");
              message += to_string(I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(0,0) - (I_A2_IBinBoundY(I_IBin, 0) - I_A2_IBinBoundY(0,0)) + 1);
              message += string(") != D_A2_SFSM.rows()=") + to_string(D_A2_SFSM.rows());
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
          #endif
          D_A3_SFSM(blitz::Range(I_A2_IBinBoundY(I_IBin, 0) - I_A2_IBinBoundY(0,0), I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(0,0)), blitz::Range::all(), I_IBin) = D_A2_SFSM;

          D_A2_SP(blitz::Range(I_A2_IBinBoundY(I_IBin, 0) - I_A2_IBinBoundY(0,0), I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(0,0)), I_IBin) = D_A1_SP;

          if (ErrorsRead)
            D_A2_Errors_SP_Out(blitz::Range(I_A2_IBinBoundY(I_IBin, 0) - I_A2_IBinBoundY(0,0), I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(0,0)), I_IBin) = D_A1_Errors_SP_Out;

          ///TODO: use iterator
          for (int i_row=0; i_row<I_A2_Mask.rows(); i_row++){
            if (max(I_A2_Mask(i_row, blitz::Range::all())) == 1){
              for (int i_col=0; i_col<I_A2_Mask.cols(); i_col++){
                if (I_A2_Mask(i_row, i_col) == 0)
                  I_A2_MaskArray(I_A2_IBinBoundY(I_IBin, 0)+i_row, i_col) = I_A2_Mask(i_row, i_col);
              }
              #ifdef __DEBUG_MKSLITFUNC__
                cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_A2_MaskArray(" << I_A2_IBinBoundY(I_IBin, 0)+i_row << ", *) = " << I_A2_MaskArray(I_A2_IBinBoundY(I_IBin, 0)+i_row, blitz::Range::all()) << endl;
              #endif
            }
          }
          if (telluric == 1){
            D_A2_Sky(blitz::Range(I_A2_IBinBoundY(I_IBin, 0) - I_A2_IBinBoundY(0,0), I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(0,0)), I_IBin) = D_A1_Sky;
            if (ErrorsRead)
              D_A2_ErrSky(blitz::Range(I_A2_IBinBoundY(I_IBin, 0) - I_A2_IBinBoundY(0,0), I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(0,0)), I_IBin) = D_A1_ErrSky;
          }
        }/// end if (!B_MaximaOnly)
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": fiberTraceNumber = " << fiberTraceNumber << ", IBin = " << I_IBin << ": D_A2_SFSM set to " << D_A2_SFSM << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": fiberTraceNumber = " << fiberTraceNumber << ", IBin = " << I_IBin << ": D_A1_SP set to " << D_A1_SP << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": fiberTraceNumber = " << fiberTraceNumber << ", IBin = " << I_IBin << ": I_A2_Mask_Temp set to " << I_A2_Mask_Temp << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": SlitFunc ready" << endl;

          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": After SlitFunc: D_A1_SP = " << D_A1_SP << endl;

          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": After SlitFunc: D_A2_SFSM = " << D_A2_SFSM << endl;

          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": After SlitFunc: D_A2_SFSM = Im_Out = " << D_A2_SFSM << endl;
          string S_FNameOut = debugdir + "SlitFunc_SFSMOut_out" + S_TempNum + S_SF_DebugFilesSuffix + ".fits";
          ::pfs::drp::stella::utils::WriteFits(&D_A2_SFSM,  S_FNameOut);
        #endif

        if (telluric == 3){/// Subtract minimum of each row > 0. from Slit Function
          double D_MinSF = 0.;
          for (int q=0; q<D_A2_SFSM.rows(); q++){


            /// TODO: Only include pixels which are not marked as bad (old strategy with some border pixels equal to zero)

            D_MinSF = min(D_A2_SFSM(q, blitz::Range::all()));
            D_A2_SFSM(q, blitz::Range::all()) = D_A2_SFSM(q, blitz::Range::all()) - D_MinSF;
            double D_Sum_D_A2_SFSM = blitz::sum(D_A2_SFSM(q,blitz::Range::all()));
            if (abs(D_Sum_D_A2_SFSM) < 0.00000000000000001){
              std::string message("FiberTrace");
              message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
              message += std::string(": ERROR: D_Sum_D_A2_SFSM == 0");
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
            D_A2_SFSM(q, blitz::Range::all()) = D_A2_SFSM(q, blitz::Range::all()) / D_Sum_D_A2_SFSM;
            #ifdef __DEBUG_MKSLITFUNC__
              cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": D_Sum_D_A2_SFSM = " << D_Sum_D_A2_SFSM << endl;
              cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": D_A2_SFSM(q=" << q << ", *) set to " << D_A2_SFSM(q, blitz::Range::all()) << endl;
            #endif
          }/// end for (int q=0; q<D_A2_SFSM.rows(); q++){
          #ifdef __DEBUG_MKSLITFUNC__
            string S_FName = "D_A2_SFOut_Tel3";
            if (B_MaximaOnly)
              S_FName += "_MaxOnly";
            S_FName += "_IRunTel" + to_string(I_Run_Tel) + ".fits";
            ::pfs::drp::stella::utils::WriteFits(&D_A2_SFSM, S_FName);
          #endif
        }/// end if (telluric == 3)

        ::pfs::drp::stella::math::Double(I_A2_Mask, D_A2_Mask);
        if (fabs(mean(I_A2_Mask) - mean(D_A2_Mask)) > 0.000001){
          std::string message("FiberTrace");
          message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
          message += std::string(": ERROR: mean(I_A2_Mask)(=") + to_string(mean(I_A2_Mask)) + string(") != mean(D_A2_Mask)(=");
          message += to_string(mean(D_A2_Mask)) + string(")");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        D_SFDev = sqrt(blitz::sum(blitz::pow2(D_A2_SFSM - D_A2_SFOld) * D_A2_Mask)/blitz::sum(D_A2_Mask));
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": I_Run_Tel = " << I_Run_Tel << ": D_SFDev = " << D_SFDev << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": I_Run_Tel = " << I_Run_Tel << ": mean(D_A2_SlitFunc_Im_In_Tel) / 1000. = " << mean(D_A2_SlitFunc_Im_In_Tel) / 1000. << endl;
        #endif
        D_A2_SFOld = D_A2_SFSM;

        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": Before Fit: 1. D_A2_Err = " << D_A2_Err << endl;
        #endif
        #ifdef __DEBUG_CHECK_INDICES__
          if (I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(I_IBin, 0) + 1 != I_A2_Mask.rows()){
            std::string message("FiberTrace");
            message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
            message += std::string(": ERROR: I_A2_IBound(I_IBin, 1) - I_A2_IBound(I_IBin, 0) + 1(=") + to_string(I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(I_IBin, 0) + 1);
            message += string(") != I_A2_Mask.rows())(=") + to_string(I_A2_Mask.rows()) + string(")");
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          if (I_A2_Mask_AllRows.cols() != I_A2_Mask.cols()){
            std::string message("FiberTrace");
            message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
            message += std::string(": ERROR: I_A2_Mask_AllRows.cols(=") + to_string(I_A2_Mask_AllRows.cols()) + string(") != I_A2_Mask.cols(=");
            message += to_string(I_A2_Mask.cols()) + string(")");
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": D_A2_SFSM.cols() = " << D_A2_SFSM.cols() << endl;
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": I_Run_Tel = " << I_Run_Tel << ": I_IBin = " << I_IBin << ", I_A2_IBound = " << I_A2_IBinBoundY.rows() << " x " << I_A2_IBinBoundY.cols() << ", I_A2_IBound(I_IBin, *) = " << I_A2_IBinBoundY(I_IBin, blitz::Range::all()) << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": I_Run_Tel = " << I_Run_Tel << ": I_IBin = " << I_IBin << ", I_A2_Mask_AllRows = " << I_A2_Mask_AllRows.rows() << " x " << I_A2_Mask_AllRows.cols() << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": I_Run_Tel = " << I_Run_Tel << ": I_IBin = " << I_IBin << ", I_A2_Mask = " << I_A2_Mask.rows() << " x " << I_A2_Mask.cols() << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: Before Fit: 2. D_A2_Err = " << D_A2_Err << endl;
        #endif

        #ifdef __DEBUG_CHECK_INDICES__
          if (I_A2_IBinBoundY(I_IBin, 0) < 0){
            std::string message("FiberTrace");
            message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
            message += std::string(": ERROR: I_A2_IBound(I_IBin=") + to_string(I_IBin) + string(", 0) = ") + to_string(I_A2_IBinBoundY(I_IBin, 0));
            message += string(" < 0");
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          if (I_A2_IBinBoundY(I_IBin, 1) >= I_A2_Mask_AllRows.rows()){
            std::string message("FiberTrace");
            message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
            message += std::string(": ERROR: I_A2_IBound(I_IBin=") + to_string(I_IBin) + string(", 1) = ") + to_string(I_A2_IBinBoundY(I_IBin, 1));
            message += string(" >= I_A2_Mask_AllRows.rows() = ") + to_string(I_A2_Mask_AllRows.rows()) + string(" => Returning FALSE");
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          if (I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(I_IBin, 0) + 1 != I_A2_Mask.rows()){
            std::string message("FiberTrace");
            message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
            message += std::string(": ERROR: I_A2_IBound(I_IBin=") + to_string(I_IBin) + string(", 1)(=") + to_string(I_A2_IBinBoundY(I_IBin, 1));
            message += string(") - I_A2_IBound(I_IBin, 0)(=") + to_string(I_A2_IBinBoundY(I_IBin, 0)) + string(") + 1 = ");
            message += to_string(I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(I_IBin, 0) + 1) + string(" != I_A2_Mask.rows()=");
            message += to_string(I_A2_Mask.rows());
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          if (I_A2_Mask_AllRows.cols() != I_A2_Mask.cols()){
            std::string message("FiberTrace");
            message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
            message += std::string(": ERROR: I_A2_Mask_AllRows.cols()=") + to_string(I_A2_Mask_AllRows.cols()) + string(" != I_A2_Mask.cols()=");
            message += to_string(I_A2_Mask.cols());
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_A2_Mask = " << I_A2_Mask << endl;
        #endif
        if (!B_MaximaOnly){
          for (int i_row=0; i_row<I_A2_Mask.rows(); i_row++){
            #ifdef __DEBUG_MKSLITFUNC__
              cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_A2_Mask_AllRows(" << I_A2_IBinBoundY(I_IBin, 0)+i_row << ", *) = " << I_A2_Mask_AllRows(I_A2_IBinBoundY(I_IBin, 0)+i_row, blitz::Range::all()) << endl;
            #endif
            for (int i_col=0; i_col<I_A2_Mask.cols(); i_col++){
              if (I_A2_Mask(i_row, i_col) == 0)
                I_A2_Mask_AllRows(I_A2_IBinBoundY(I_IBin, 0)+i_row, i_col) = 0;
            }
            #ifdef __DEBUG_MKSLITFUNC__
              cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_A2_Mask_AllRows(" << I_A2_IBinBoundY(I_IBin, 0)+i_row << ", *) = " << I_A2_Mask_AllRows(I_A2_IBinBoundY(I_IBin, 0)+i_row, blitz::Range::all()) << endl;
            #endif
          }
        }/// end if (!B_MaximaOnly)
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": I_A2_Mask_AllRows(blitz::Range(" << I_A2_IBinBoundY(I_IBin, 0) << ", " << I_A2_IBinBoundY(I_IBin, 1) << "), blitz::Range::all()) set to I_A2_Mask = " << I_A2_Mask << endl;
        #endif

        D_A2_Err.resize(D_A2_Err_Temp.rows(), D_A2_Err_Temp.cols());
        D_A2_Err = D_A2_Err_Temp;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": Before Fit: 4. D_A2_Err = " << D_A2_Err << endl;
        #endif
        if ((I_Run_Tel == (I_RunMax-1)) || (D_SFDev < mean(D_A2_SFSM) / 5000000.)){
          #ifdef __DEBUG_MKSLITFUNC__
            if (D_SFDev < mean(D_A2_SlitFunc_Im_In_Tel) / 5000000.){
              if (B_MaximaOnly)
                cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": B_MaximaOnly==TRUE: D_SFDev = " << D_SFDev << ": mean(D_A2_SFSM) = " << mean(D_A2_SFSM) << endl;
              else
                cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": D_SFDev = " << D_SFDev << ": mean(D_A2_SFSM) = " << mean(D_A2_SFSM) << endl;
            }
          #endif
          if (B_MaximaOnly){
            B_MaximaOnly = false;
            I_MaxIterSig = _fiberTraceProfileFittingControl->maxIterSig;
            if (telluric == 3){
              I_Run_Tel = -1;
              D_SFDev = 1.;
              I_A2_Mask_Tel = I_A2_Msk;
              I_A2_Mask_Temp = I_A2_Msk;
              #ifdef __DEBUG_MKSLITFUNC__
                cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": I_A2_Mask_Tel set to " << I_A2_Mask_Tel << endl;
              #endif
              D_A2_SlitFunc_Im_In = D_A2_SlitFuncOrig;
              D_A2_SlitFunc_Im_In_Tel = D_A2_SlitFuncOrig;
              #ifdef __DEBUG_MKSLITFUNC__
                cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": D_A2_SlitFunc_Im_In_Tel set to " << D_A2_SlitFunc_Im_In_Tel << endl;
              #endif
              D_A1_Sky = 0.;
              #ifdef __DEBUG_MKSLITFUNC__
                cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": D_A2_SFSM = " << D_A2_SFSM << endl;
              #endif
            }
            else{
              I_Run_Tel = I_RunMax;
            }
          }
          else{
            I_Run_Tel = I_RunMax;
          }
        }/// end if ((I_Run_Tel == (I_RunMax-1)) || (D_SFDev < mean(D_A2_SFSM) / 5000.))

        if (telluric == 3){
          blitz::Array<string, 1> S_A1_Args_Fit(4);
          void **PP_Args_Fit;
          PP_Args_Fit = (void**)malloc(sizeof(void*) * 4);
          S_A1_Args_Fit = " ";

          #ifdef __DEBUG_CHECK_INDICES__
            if (D_A2_SlitFuncOrig.rows() != D_A2_Err.rows()){
              std::string message("FiberTrace");
              message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
              message += std::string(": ERROR: D_A2_SlitFuncOrig.rows(=") + to_string(D_A2_SlitFuncOrig.rows()) + string(") != D_A2_Err.rows(=");
              message += to_string(D_A2_Err.rows()) + string(")");
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
          #endif
          if (ErrorsRead){
            S_A1_Args_Fit(0) = "MEASURE_ERRORS_IN";
            PP_Args_Fit[0] = &D_A2_Err;
            #ifdef __DEBUG_MKSLITFUNC__
              cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": Just before Fit: D_A2_Err = " << D_A2_Err << endl;
            #endif
          }

          #ifdef __DEBUG_CHECK_INDICES__
            if (D_A2_SlitFuncOrig.rows() != I_A2_Mask.rows()){
              std::string message("FiberTrace");
              message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
              message += std::string(": ERROR: D_A2_SlitFuncOrig.rows(=") + to_string(D_A2_SlitFuncOrig.rows()) + string(") != I_A2_Mask.rows(=");
              message += to_string(I_A2_Mask.rows()) + string(")");
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
          #endif
          S_A1_Args_Fit(1) = "MASK_INOUT";
          /// TODO ????????????????????????????????????????????????
          PP_Args_Fit[1] = &I_A2_Mask;
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": Just before Fit: I_A2_Mask = " << I_A2_Mask << endl;
          #endif

          S_A1_Args_Fit(2) = "SIGMA_OUT";
          blitz::Array<double, 2> D_A2_Sigma_Fit(D_A2_SFSM.rows(),2);
          D_A2_Sigma_Fit = 0.;
          PP_Args_Fit[2] = &D_A2_Sigma_Fit;

          ///TODO: change to function parameter
          double D_Reject = 5.;
          S_A1_Args_Fit(3) = "REJECT_IN";
          PP_Args_Fit[3] = &D_Reject;

          /// Use SlitFunc_Im_In for fit, and SlitFunc_im_in_tel for SlitFunc()
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": min(D_A2_SlitFunc_Im_In_Tel) = " << min(D_A2_SlitFunc_Im_In_Tel) << endl;
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": Just before Fit: D_A2_SFSM = " << D_A2_SFSM << endl;
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": Just before Fit: D_A2_SlitFuncOrig = " << D_A2_SlitFuncOrig << endl;
          #endif

          #ifdef __DEBUG_CHECK_INDICES__
            if (D_A2_SlitFuncOrig.rows() != D_A2_SFSM.rows()){
              std::string message("FiberTrace");
              message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
              message += std::string(": ERROR: D_A2_SlitFuncOrig.rows(=") + to_string(D_A2_SlitFuncOrig.rows()) + string(") != D_A2_SFSM.rows(=");
              message += to_string(D_A2_SFSM.rows()) + string(")");
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
          #endif
          if (!::pfs::drp::stella::math::LinFitBevington(D_A2_SlitFuncOrig,      ///: in
                                                   D_A2_SFSM,             ///: in
                                                   D_A1_SPFit,             ///: out
                                                   D_A1_Sky_Temp,          ///: in/out
                                                   true,                   ///: with sky: in
                                                   S_A1_Args_Fit,         ///: in
                                                   PP_Args_Fit)){          ///: in/out
              /// MEASURE_ERRORS_IN = blitz::Array<double,2>(D_A2_CCD_In.rows, D_A2_CCD_In.cols) : in
              /// REJECT_IN = double                                                      : in
              /// MASK_INOUT = blitz::Array<double,2>(D_A1_CCD_In.rows,D_A1_CCD_In.cols)         : in/out
              /// CHISQ_OUT = blitz::Array<double,1>(D_A2_CCD_In.rows)                           : out
              /// Q_OUT = blitz::Array<double,1>(D_A2_CCD_In.rows)                               : out
              /// SIGMA_OUT = blitz::Array<double,2>(D_A2_CCD_In.rows, 2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
            std::string message("FiberTrace");
            message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
            message += std::string(": ERROR: 1. telluric == 2: LinFitBevington(...) returned FALSE");
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
            #ifdef __DEBUG_MKSLITFUNC__
              cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": Just after Fit: D_A2_SFSM = " << D_A2_SFSM << endl;
            #endif
            D_A1_Sky_Temp = blitz::where(D_A1_Sky_Temp < 0., 0., D_A1_Sky_Temp);
            D_A1_Sky = D_A1_Sky_Temp;

//            #ifdef __DEBUG_MKSLITFUNC__
//              blitz::Array<double, 2> D_A2_MaskTimesSFOrig(D_A2_SlitFuncOrig.rows(), D_A2_SlitFuncOrig.cols());
//              D_A2_MaskTimesSFOrig = D_A2_SlitFuncOrig * I_A2_Mask;
//              string S_MaskFitOut = "SFOrigTimesMaskAfterFit") + CS_SF_DebugFilesSuffix + ".fits";
//              ::pfs::drp::stella::utils::WriteFits(&D_A2_MaskTimesSFOrig, S_MaskFitOut);
//            #endif

            ///Remove sky from D_A2_SlitFunc_Im_In_Tel???
            double D_ImMin;
            for (int tttt=0; tttt<D_A2_SlitFunc_Im_In_Tel.rows(); tttt++){
              D_A2_SlitFunc_Im_In_Tel(tttt, blitz::Range::all()) = D_A2_SlitFuncOrig(tttt, blitz::Range::all()) - D_A1_Sky(tttt);
              for (int uuuu=0; uuuu<D_A2_SlitFunc_Im_In_Tel.cols(); uuuu++){
                if (I_A2_Mask_Temp(tttt, uuuu) == 0)
                  D_A2_SlitFunc_Im_In_Tel(tttt, uuuu) = 0.;
              }
              for (int uuuu=0; uuuu<D_A2_SlitFunc_Im_In_Tel.cols(); uuuu++){
                if (I_A2_Mask_Temp(tttt, uuuu) == 0)
                  D_A2_SlitFunc_Im_In_Tel(tttt, uuuu) = 0.;
              }
              D_ImMin = min(D_A2_SlitFunc_Im_In_Tel(tttt, blitz::Range::all()));
              if (D_ImMin < 0. - (3. * _fiberTraceProfileFittingControl->ccdReadOutNoise)){
                D_ImMin = D_ImMin + (3. * _fiberTraceProfileFittingControl->ccdReadOutNoise);
                #ifdef __DEBUG_MKSLITFUNC__
                  cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": tttt=" << tttt << ": D_ImMin = " << D_ImMin << endl;
                  cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": tttt=" << tttt << ": D_A2_SlitFunc_Im_In_Tel(tttt,*) = " << D_A2_SlitFunc_Im_In_Tel(tttt, blitz::Range::all()) << endl;
                  cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": tttt=" << tttt << ": D_A1_Sky(tttt) = " << D_A1_Sky(tttt) << endl;
                #endif
                D_A2_SlitFunc_Im_In_Tel(tttt, blitz::Range::all()) -= D_ImMin;
                #ifdef __DEBUG_MKSLITFUNC__
                  cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": tttt=" << tttt << ": D_A2_SlitFunc_Im_In_Tel(tttt,*) set to " << D_A2_SlitFunc_Im_In_Tel(tttt, blitz::Range::all()) << endl;
                #endif
                D_A1_Sky(tttt) += D_ImMin;
                #ifdef __DEBUG_MKSLITFUNC__
                  cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": tttt=" << tttt << ": D_A1_Sky(tttt) set to " << D_A1_Sky(tttt) << endl;
                #endif
              }
            }


            #ifdef __DEBUG_MKSLITFUNC__
              cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": I_Run_Tel = " << I_Run_Tel << ": after sky subtraction: min(D_A2_SlitFunc_Im_In_Tel) = " << min(D_A2_SlitFunc_Im_In_Tel) << endl;
            #endif

            #ifdef __DEBUG_MKSLITFUNC__
              string sPFit = debugdir + "D_A1_SPFit_";
              if (B_MaximaOnly)
                sPFit += "MaxOnly_";
              sPFit += to_string(I_Run_Tel) + ".dat";
              ::pfs::drp::stella::utils::WriteArrayToFile(D_A1_SPFit, sPFit, string("ascii"));
              string skyFit = debugdir + "D_A1_SkyFit_";
              if (B_MaximaOnly)
                skyFit += "MaxOnly_" + to_string(I_Run_Tel) + ".dat";
              ::pfs::drp::stella::utils::WriteArrayToFile(D_A1_Sky, skyFit, string("ascii"));
              string S_SlitFunc_Im_In_Tel = debugdir + "D_A2_SlitFunc_Im_In_Tel_skySubtracted";
              if (B_MaximaOnly)
                S_SlitFunc_Im_In_Tel += "_MaxOnly";
              S_SlitFunc_Im_In_Tel += to_string(I_Run_Tel) + ".fits";
              ::pfs::drp::stella::utils::WriteFits(&D_A2_SlitFunc_Im_In_Tel, S_SlitFunc_Im_In_Tel);
            #endif

            if (ErrorsRead){
              D_A1_ErrSky.resize(D_A1_SPFit.size());
              D_A1_ErrSky(blitz::Range::all()) = D_A2_Sigma_Fit(blitz::Range::all(),1);

              D_A1_ErrOut.resize(D_A1_SPFit.size());
              D_A1_ErrOut(blitz::Range::all()) = D_A2_Sigma_Fit(blitz::Range::all(),0);
            }
            D_A1_Sky_Temp = 0.;
            free(PP_Args_Fit);
        }/// end if (telluric == 3)
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": I_Run_Tel = " << I_Run_Tel << endl;
        #endif
        I_Run_Tel++;
      } while (I_Run_Tel < I_RunMax);
      #ifdef __DEBUG_MKSLITFUNC__
        if (oldYHigh != _fiberTraceFunction->yHigh){
          string message("FiberTrace ");
          message += to_string(_iTrace) + string("::MkSlitFunc: I_IBin = ") + to_string(I_IBin) + string(": end do ... while(I_Run_Tel < I_RunMax): 2. ERROR: oldYHigh(=") + to_string(oldYHigh) + string(") != _fiberTraceFunction->yHigh(=");
          message += to_string(_fiberTraceFunction->yHigh);
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
      #endif
      for (int q=0; q < static_cast<int>(D_A1_SP.size()); q++){

        #ifdef __DEBUG_MKSLITFUNC__
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": I_A2_Msk(q=" << q << ", blitz::Range::all()) = " << I_A2_Msk(q,blitz::Range::all()) << endl;
          cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_IBin = " << I_IBin << ": max(I_A2_Msk(q=" << q << ", blitz::Range::all())) = " << max(I_A2_Msk(q,blitz::Range::all())) << endl;
        #endif
        if (max(I_A2_Msk(q, blitz::Range::all())) > 1){
          std::string message("FiberTrace");
          message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
          message += std::string(": ERROR: max(I_A2_Msk(q=") + to_string(q) + string(", blitz::Range::all())) > 1");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
      }/// end for (int q=0; q < D_A1_SP.size(); q++)
      if (max(I_A2_Msk > 1)){
        std::string message("FiberTrace");
        message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_IBin = ") + std::to_string(I_IBin);
        message += std::string(": ERROR: max(I_A2_Msk)=") + to_string(max(I_A2_Msk)) + string(" > 1");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    } /// end for (int I_IBin = 0; I_IBin < I_NBin; I_IBin++) /// Loop thru sf regions
    #ifdef __DEBUG_MKSLITFUNC__
      if (oldYHigh != _fiberTraceFunction->yHigh){
        string message("FiberTrace ");
        message += to_string(_iTrace) + string("::MkSlitFunc: after for(I_IBin...): 2. ERROR: oldYHigh(=") + to_string(oldYHigh) + string(") != _fiberTraceFunction->yHigh(=");
        message += to_string(_fiberTraceFunction->yHigh);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    #endif
    D_A1_Sky.resize(I_A2_IBinBoundY(I_NBins-1,1) - I_A2_IBinBoundY(0,0) + 1);
    D_A1_ErrSky.resize(I_A2_IBinBoundY(I_NBins-1,1) - I_A2_IBinBoundY(0,0) + 1);
    D_A1_SP.resize(I_A2_IBinBoundY(I_NBins-1,1) - I_A2_IBinBoundY(0,0) + 1);

    D_A1_Errors_SP_Out.resize(I_A2_IBinBoundY(I_NBins-1,1) - I_A2_IBinBoundY(0,0) + 1);

    D_A2_SF.resize(I_A2_IBinBoundY(I_NBins-1,1) - I_A2_IBinBoundY(0,0) + 1, _trace->getWidth());
    #ifdef __DEBUG_CHECK_INDICES__
      if (static_cast<int>(D_A1_SP.size()) != D_A2_SP.rows()){
        std::string message("FiberTrace");
        message += std::to_string(_iTrace) + std::string("::MkSlitFunc: ERROR: D_A1_SP.size(=") + to_string(D_A1_SP.size()) + string(") != D_A2_SP.rows(= ") + to_string(D_A2_SP.rows());
        message += string(")");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: I_A2_IBinBoundY = " << I_A2_IBinBoundY << endl;
      if (D_A2_SF.rows() != D_A3_SFSM.rows()){
        std::string message("FiberTrace");
        message += std::to_string(_iTrace) + std::string("::MkSlitFunc: ERROR: D_A2_SF.rows(=") + to_string(D_A2_SF.rows()) + string(") != D_A3_SFSM.rows(= ");
        message += to_string(D_A3_SFSM.rows()) + string(")");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (D_A2_SF.cols() != D_A3_SFSM.cols()){
        std::string message("FiberTrace");
        message += std::to_string(_iTrace) + std::string("::MkSlitFunc: ERROR: D_A2_SF.cols(=") + to_string(D_A2_SF.cols()) + string(") != D_A3_SFSM.cols(= ");
        message += to_string(D_A3_SFSM.cols()) + string(")");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    #endif
    if (I_NBins == 1){
      D_A1_SP = D_A2_SP(blitz::Range::all(), 0);

      if (telluric == 1){
        D_A1_Sky = D_A2_Sky(blitz::Range::all(), 0);
        if (ErrorsRead)
          D_A1_ErrSky = D_A2_ErrSky(blitz::Range::all(), 0);
      }

      D_A2_SF = D_A3_SFSM(blitz::Range::all(), blitz::Range::all(), 0);
    }
    else{
      int I_Bin = 0;
      double D_Weight_Bin0 = 0.;
      double D_Weight_Bin1 = 0.;
//      int I_Row_Rel=0;
      double D_RowSum;
      for (int i_row = 0; i_row < D_A3_SFSM.rows(); i_row++){
        for (int i_ibin = 0; i_ibin < I_NBins; i_ibin++){
          D_RowSum = blitz::sum(D_A3_SFSM(i_row, blitz::Range::all(), i_ibin));
          if (fabs(D_RowSum) > 0.00000000000000001){
            D_A3_SFSM(i_row, blitz::Range::all(), i_ibin) = D_A3_SFSM(i_row, blitz::Range::all(), i_ibin) / D_RowSum;
            #ifdef __DEBUG_MKSLITFUNC__
              cout << "D_A3_SFSM(" << i_row << ", *, " << i_ibin << ") = " << D_A3_SFSM(i_row, blitz::Range::all(), i_ibin) << endl;
              cout << "i_row = " << i_row << ": i_ibin = " << i_ibin << ": D_RowSum = " << D_RowSum << endl;
            #endif
          }
        }
        if ((I_Bin == 0) && (i_row < I_A2_IBinBoundY(1, 0))){
          D_A1_SP(i_row) = D_A2_SP(i_row, 0);

          D_A2_SF(i_row, blitz::Range::all()) = D_A3_SFSM(i_row, blitz::Range::all(), 0);
          if (telluric == 1){
            D_A1_Sky(i_row) = D_A2_Sky(i_row, 0);
            if (ErrorsRead)
              D_A1_ErrSky(i_row) = D_A2_ErrSky(i_row, 0);
          }

          if (ErrorsRead){
            D_A1_Errors_SP_Out(i_row) = D_A2_Errors_SP_Out(i_row, 0);
          }
        }
        else if ((I_Bin == I_NBins-1) && (i_row >= I_A2_IBinBoundY(I_Bin-1, 1))){// && (i_row > (I_A2_IBound(I_Bin-1, 1) - (I_A2_IBound(0,1) / 2.)))){
          D_A1_SP(i_row) = D_A2_SP(i_row, I_Bin);
          D_A2_SF(i_row, blitz::Range::all()) = D_A3_SFSM(i_row, blitz::Range::all(), I_Bin);
          if (telluric == 1){
            D_A1_Sky(i_row) = D_A2_Sky(i_row, I_Bin);
            if (ErrorsRead)
              D_A1_ErrSky(i_row) = D_A2_ErrSky(i_row, I_Bin);
          }
          if (ErrorsRead){
            D_A1_Errors_SP_Out(i_row) = D_A2_Errors_SP_Out(i_row, I_Bin);
          }
        }
        else{
          D_Weight_Bin1 = double(i_row - I_A2_IBinBoundY(I_Bin+1, 0)) / double(I_A2_IBinBoundY(I_Bin, 1) - I_A2_IBinBoundY(I_Bin+1, 0));
          D_Weight_Bin0 = 1. - D_Weight_Bin1;
//          #ifdef __DEBUG_MKSLITFUNC__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: i_row = " << i_row << ": I_NBins = " << I_NBins << endl;
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: i_row = " << i_row << ": I_Bin = " << I_Bin << endl;
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: i_row = " << i_row << ": I_A2_IBinBoundY(I_Bin, *) = " << I_A2_IBinBoundY(I_Bin, blitz::Range::all()) << endl;
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: i_row = " << i_row << ": I_A2_IBinBoundY(I_Bin+1, *) = " << I_A2_IBinBoundY(I_Bin+1, blitz::Range::all()) << endl;
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: i_row = " << i_row << ": D_Weight_Bin0 = " << D_Weight_Bin0 << endl;
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: i_row = " << i_row << ": D_Weight_Bin1 = " << D_Weight_Bin1 << endl;
//          #endif
          #ifdef __DEBUG_CHECK_INDICES__
            if (i_row >= D_A2_SP.rows()){
              std::string message("FiberTrace");
              message += std::to_string(_iTrace) + std::string("::MkSlitFunc: i_row = ") + to_string(i_row) + string(": I_Bin = ") + std::to_string(I_Bin);
              message += std::string(": ERROR: i_row = ") + to_string(i_row) + string(" >= D_A2_SP.rows() = ") + to_string(D_A2_SP.rows());
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
            if (I_Bin + 1 >= D_A2_SP.cols()){
              std::string message("FiberTrace");
              message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_Bin = ") + std::to_string(I_Bin);
              message += std::string(": ERROR: I_Bin + 1 = ") + to_string(I_Bin + 1) + string(" >= D_A2_SP.cols() = ") + to_string(D_A2_SP.cols());
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
            }
          #endif
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: i_row = " << i_row << ": I_Bin = " << I_Bin << ": D_A2_SP(i_row, I_Bin) = " << D_A2_SP(i_row, I_Bin) << ", D_A2_SP(i_row, I_Bin + 1) = " << D_A2_SP(i_row, I_Bin + 1) << endl;
          #endif
          D_A1_SP(i_row) = (D_A2_SP(i_row, I_Bin) * D_Weight_Bin0) + (D_A2_SP(i_row, I_Bin + 1) * D_Weight_Bin1);
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: i_row = " << i_row << ": I_Bin = " << I_Bin << ": D_A1_SP(" << i_row << ") set to " << D_A1_SP(i_row) << endl;
          #endif

          if (ErrorsRead){
            D_A1_Errors_SP_Out(i_row) = (D_A2_Errors_SP_Out(i_row, I_Bin) * D_Weight_Bin0) + (D_A2_Errors_SP_Out(i_row, I_Bin + 1) * D_Weight_Bin1);
          }

          #ifdef __DEBUG_MKSLITFUNC__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: D_A3_SFSM(i_row, *, I_Bin) = " << D_A3_SFSM(i_row, blitz::Range::all(), I_Bin) << ", D_A3_SFSM(i_row, *, I_Bin+1) = " << D_A3_SFSM(i_row, blitz::Range::all(), I_Bin+1) << endl;
          #endif
          D_A2_SF(i_row, blitz::Range::all()) = (D_A3_SFSM(i_row, blitz::Range::all(), I_Bin) * D_Weight_Bin0) + (D_A3_SFSM(i_row, blitz::Range::all(), I_Bin+1) * D_Weight_Bin1);
          double dSumSFRow = blitz::sum(D_A2_SF(i_row, blitz::Range::all()));
          if (fabs(dSumSFRow) >= 0.00000000000000001)
            D_A2_SF(i_row, blitz::Range::all()) = D_A2_SF(i_row, blitz::Range::all()) / dSumSFRow;
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: i_row = " << i_row << ": I_Bin = " << I_Bin << ": D_A2_SF(" << i_row << ", *) set to " << D_A2_SF(i_row, blitz::Range::all()) << endl;
          #endif

          if (telluric == 1){
            #ifdef __DEBUG_CHECK_INDICES__
              if (i_row >= D_A1_Sky.rows()){
                std::string message("FiberTrace");
                message += std::to_string(_iTrace) + std::string("::MkSlitFunc: I_Bin = ") + std::to_string(I_Bin);
                message += std::string(": ERROR: i_row(=") + to_string(i_row) + string(") >= D_A1_Sky.rows()=") + to_string(D_A1_Sky.rows());
                cout << message << endl;
                throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
              }
            #endif
            D_A1_Sky(i_row) = (D_A2_Sky(i_row, I_Bin) * D_Weight_Bin0) + (D_A2_Sky(i_row, I_Bin + 1) * D_Weight_Bin1);
            D_A1_ErrSky(i_row) = (D_A2_ErrSky(i_row, I_Bin) * D_Weight_Bin0) + (D_A2_ErrSky(i_row, I_Bin + 1) * D_Weight_Bin1);
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: i_row = " << i_row << ": I_Bin = " << I_Bin << ": D_A1_Sky(" << i_row << ") set to " << D_A1_Sky(i_row) << endl;
          }

        //  I_Row_Rel++;
        }

        if (i_row == (I_A2_IBinBoundY(I_Bin, 1))){
          I_Bin++;
        //  I_Row_Rel = 0;
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "FiberTrace" << _iTrace << "::MkSlitFunc: i_row = " << i_row << ": I_Bin set to " << I_Bin << endl;
          #endif
        }
      }/// end for (int i_row = 0; i_row < D_A3_SFSM.rows(); i_row++){
    }/// end if (I_NBin != 1){
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace" << _iTrace << "::MkSlitFunc: D_A2_SF set to " << D_A2_SF << endl;
    #endif

    ///populate _spectrum, _spectrumVariance, _background, _backgroundVariance
    PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) spectrum(new pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>(_trace->getHeight()));
    PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) background(new Spectrum<ImageT, MaskT, VarianceT, VarianceT>(_trace->getHeight()));
    for (int ind=0; ind<_trace->getHeight(); ind++){
      (*(spectrum->getSpectrum()))[ind] = static_cast<ImageT>(D_A1_SP(ind));
      (*(spectrum->getVariance()))[ind] = static_cast<VarianceT>(D_A1_Errors_SP_Out(ind));
      (*(background->getSpectrum()))[ind] = static_cast<ImageT>(D_A1_Sky(ind));
      (*(background->getVariance()))[ind] = static_cast<VarianceT>(D_A1_ErrSky(ind));
    }
    blitz::Array<float, 2> F_A2_ProfArray = ::pfs::drp::stella::math::Float(D_A2_SF);
    ndarray::Array<float, 2, 1> ndarrayProf(::pfs::drp::stella::utils::copyBlitzToNdarray(F_A2_ProfArray));
    PTR(afwImage::Image<float>) imageProf(new afwImage::Image<float>(ndarrayProf));
    _profile.reset();
    _profile = imageProf;
    _isProfileSet = true;
    blitz::Array<MaskT, 2> U_A2_Mask(_trace->getHeight(), _trace->getWidth());
    U_A2_Mask = where(I_A2_Mask_AllRows == 1, T_A2_MaskArray, 1);
    ndarray::Array<MaskT, 2, 1> ndarrayMask(::pfs::drp::stella::utils::copyBlitzToNdarray(U_A2_Mask));
    afwImage::Mask<MaskT> maskImage(ndarrayMask);
    *_trace->getMask() = maskImage;

    I_A2_IBinBoundY.resize(0);
    I_A2_Msk.resize(0, 0);
    D_A1_SC.resize(0);
    D_A1_Scatter.resize(0);
    D_A2_SFSM.resize(0, 0);
    D_A1_SF.resize(0);
    D_A2_SlitFunc_Im_In.resize(0,0);
    D_A2_SlitFTemp.resize(0,0);
    D_A1_SP.resize(0);
    D_A1_SSF.resize(0);
    D_A1_Tel.resize(0);
    D_A2_Tel.resize(0,0);
    D_A1_Temp.resize(0);
    D_A1_TempArr.resize(0);
    D_A1_TempArrA.resize(0);
    D_A1_TempArrB.resize(0);
    D_A1_TempArrC.resize(0);
    D_A1_TempArrD.resize(0);
    D_A1_XInt.resize(0);
    D_A1_XSlitFTemp.resize(0);
    I_A1_ISort.resize(0);
    I_A1_ITel.resize(0);
    I_A1_IX.resize(0);

    free(PP_Args_Median);
    free(args);
    
    #ifdef __DEBUG_MKSLITFUNC__
      if (oldYHigh != _fiberTraceFunction->yHigh){
        string message("FiberTrace ");
        message += to_string(_iTrace) + string("::MkSlitFunc: End: ERROR: oldYHigh(=") + to_string(oldYHigh) + string(") != _fiberTraceFunction->yHigh(=");
        message += to_string(_fiberTraceFunction->yHigh);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (oldYLow != _fiberTraceFunction->yLow){
        string message("FiberTrace ");
        message += to_string(_iTrace) + string("::MkSlitFunc: End: ERROR: oldYLow(=") + to_string(oldYLow) + string(") != _fiberTraceFunction->yLow(=");
        message += to_string(_fiberTraceFunction->yLow);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    #endif
    cout << "FiberTrace" << _iTrace << "::MkSlitFunc: fiberTraceNumber " << fiberTraceNumber << " finished" << endl;
    return spectrum;
  }

  /**
   * SlitFunc
   * SlitFunc
   **/
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::SlitFunc(const blitz::Array<double, 2> &D_A2_ImM,
                                                                    unsigned int maxIterSig_In,
                                                                    const blitz::Array<double, 1> &xCentersPixelFraction_In, //: in
                                                                    blitz::Array<double, 1> &spectrum_Out,   //: out
                                                                    blitz::Array<double, 2> &profile_Out,   //: out
                                                                    const blitz::Array<string, 1> &S_A1_Args_In,   //: in
                                                                    void *ArgV_In[])     //: in
  /**
   * //              TELLURIC   = int [0-none,1-Piskunov,2-mine]     : in
   *              FIBERTRACENUMBER = unsigned int: in: For debugging purposes only
   *              IM_OUT     = blitz::Array<double, 2>: out
   *              PROF_OUT   = blitz::Array<double, 2>: out
   * //             LAMBDA_SF  = double          : in
   * //             LAMBDA_SP  = int             : in
   * //             WING_SMOOTH_FACTOR = double  : in
   *              USE_ROW    = int             : in
   *              BAD        = blitz::Array<int, 1>   : out
   *              MASK       = blitz::Array<double, 2>: in/out
   *              STOP       = int [0,1]                          : in
   *              SKY        = blitz::Array<double, 1>(D_A2_ImM.rows())  : out
   *              ERRORS     = blitz::Array<double, 2>(D_A2_ImM.rows(), D_A2_ImM.cols()): in/out
   *              ERRORS_OUT = blitz::Array<double, 1>(D_A2_ImM.rows())  : out
   *              ERR_SKY    = blitz::Array<double, 1>(D_A2_ImM.rows())  : out
   *              SP_FIT     = blitz::Array<double, 1>(D_A2_ImM.rows())  : out
   *              I_BIN      = int                                : in
   *              DEBUGFILES_SUFFIX = CString: in
   *              ERRORS_SP_OUT = blitz::Array<double, 1>(D_A2_ImM.rows()) : out
   *              SP_OUT     = blitz::Array<double, 1>(D_A2_ImM.rows()) : out
   * //             XLOW       = double                            : in
   *              SFO_OUT    = blitz::Array<double, 1>(D_A2_ImM.cols()*overSample_In)
   **/
  {
    /**
     *  USAGE: SlitFunc(XCenters, SP, SF, NArgs, (*(new CString("Mask")), *(new CString(JBadVecArr)), ...), (<CFits*>P_Mask, <long>JBadVecArr, ...)
     **/

    /**
     *  Array Akl                      -> blitz::Array<double, 2> AKLArr
     *  Array bkl (size=[N,2*OverSample+1])-> blitz::Array<double, 2> BKLArr
     *  Vector bklind (size=osample+1) -> blitz::Array<int, 1> BKLIndVecArr
     *  Vector Bl                      -> blitz::Array<double, 1> BLVecArr
     *  double dev                     -> double Dev
     *  int    i                       -> int m
     *  int/Vector i1                  -> int IFirst/blitz::Array<double, 1> IFirstVecArr
     *  int/Vector i2                  -> int ILast/blitz::Array<double, 1> ILastVecArr
     *  Array  im                      -> blitz::Array<double, 2> D_A2_ImM
     *  Vector(use_col set)/Array imm  -> blitz::Array<double, 2> D_A2_ImM->(UseRowVecArr)
     *  Array  im_out                  -> blitz::Array<double, 2> *P_D_A2_Prof_Out
     *  Vector ind                     -> blitz::Array<double, 1> IndVecArr
     *  int    iter                    -> int  I_Iter_SF
     *  Vector/long   jbad             -> blitz::Array<int, 1> *P_I_A1_JBadVecArr / (*P_I_A1_JBadVecArr)(0)
     *  int    l                       -> int I_NPixSlitF
     *  double lamb_sf                 -> double Lamb_SF [: in]
     *  double lamb_sp                 -> double Lamb_SP [: in]
     *  double lambda                  -> double Lambda
     *  Array  mask, mmsk              -> blitz::Array<long, 2> P_Mask->PixArray: in
     *  Vector(use_col set)/Array msk  -> blitz::Array<double, 2> Mask(UseRowVecArr)
     *  int    n                       -> int I_NPixSlitF
     *  int    ncol                    -> ImM->cols, NColsOut -> OutArr->NCols
     *  int    nind                    -> int  NInd
     *  int    nrow                    -> ImM->rows
     *  Array/Vector  o                -> blitz::Array<double, 2> OArr
     *  Vector oo                      -> blitz::Array<double, 1> OOVecArr
     *  Vector oind                    -> blitz::Array<int, 1> OIndVecArr
     *  Vector olind  (size=osample+1) -> blitz::Array<int, 1> OLIndVecArr
     *  Vector omega                   -> blitz::Array<double, 1> OmegaVecArr
     *  long   osample                 -> int  OverSample [: in]
     *  int    oversample              -> int  OverSample [: in]
     *  Vector r                       -> blitz::Array<double, 1> RVecArr
     *  Vector sf                      -> blitz::Array<double, 2> profile_Out out
     *  Vector sp                      -> blitz::Array<double, 1> spectrum_Out: out
     *  Vector sp_old                  -> blitz::Array<double, 1> SPOldVecArr
     *  Array  ssf                     -> blitz::Array<double, 2> SSFArr
     *  int    use_col                 -> blitz::Array<long, 1> UseRowVecArr: in
     *  double weight                  -> double Weight
     *  Vector y (size=n)              -> blitz::Array<double, 1> XVecArr
     *  Vector ycen                    -> blitz::Array<double, 1> XCenVecArr: in
     *  Vector yy (size=n)             -> blitz::Array<double, 1> XXVecArr
     *  int(use_col set)/Vector yycen  -> blitz::Array<double, 1> XCenVecArr(0) <- XCenter: in
     *  double yyy                     -> double XXX
     *  IDL:     array(Column, Row)
     *  BLITZ++: array(Row, Column) !!!!!!!!!!!!!!!!!!!
     *  reform(Array, NCols, NRows)  -> blitz::Array<double, 2>& Reform(blitz::Array<double, 1>, NRows, NCols) <NOTE: Dim1 <=> Dim2>
     *  replicate(Value, Dim)      -> Replicate(Value, Dim)
     *  Matrix#Matrix              -> MatrixBTimesA()
     *  Vector#Vector              -> VecArrACrossB()
     *  Matrix#Vector              -> VecArrTimesMatrix() <Result: rows == 1>
     *  Matrix##Matrix             -> MatrixATimesB()
     *  Vector##Vector             -> VecArrACrossB()
     *  Matrix##Vector             -> MatrixTimesVecArr() <Result: cols == 1>
     **/

//    if (!_isImageSet){
//      cout << "FiberTrace" << _iTrace << "::SlitFunc: ERROR: _image is not set" << endl;
//      return false;
//    }
    #ifdef __DEBUG_SLITFUNC__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A2_ImM = " << D_A2_ImM << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: xCentersPixelFraction_In = " << xCentersPixelFraction_In << endl;
    #endif
    string debugFilesSuffix = "";
    string S_FileName_ImIn = " ";

    blitz::Array<double, 1> *P_D_A1_SPErrOut = new blitz::Array<double, 1>(D_A2_ImM.rows());
    blitz::Array<double, 1> *P_D_A1_SPOut = new blitz::Array<double, 1>(D_A2_ImM.rows());
    std::string S_SP = DEBUGDIR + std::string("SPVecArr1Final");

    profile_Out.resize(D_A2_ImM.rows(), D_A2_ImM.cols());
    profile_Out = 0.;
    int I_Bin = 0;
//    #ifdef __DEBUG_SLITFUNC__
//      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": D_A2_ImM = " << D_A2_ImM << endl;
//      CFits *p_testfitsP = new CFits();
//      S_FileName_ImIn = debugdir;
//      S_FileName_ImIn += "SlitFunc_In.fits";
//      p_testfitsP->setFileName(S_FileName_ImIn);
//      p_testfitsP->SetNRows(D_A2_ImM.rows());
//      p_testfitsP->SetNCols(D_A2_ImM.cols());
//      p_testfitsP->GetPixArray() = D_A2_ImM;
//      p_testfitsP->WriteArray();
//      delete(p_testfitsP);
//    #endif

    blitz::Array<double,2> D_A2_Im(D_A2_ImM.rows(), D_A2_ImM.cols());
    D_A2_Im = D_A2_ImM;

    int overSample_In = _fiberTraceProfileFittingControl->overSample;
    int I_NPixSlitF, TempInt, TempIntA, Pos;
    int I_Iter_SF = 0, NInd;
//    int IFirst, ILast;
    int I_Iter_Sky = 0;
    int tempcol;// = OLIndVecArr(n) / OArr.rows();
    int temprow;// = OLIndVecArr(n) - tempcol * OArr.rows();
    int tempint = 0;
    int i_tmp_sum;
    long TempLong;
    int I_NRows_Im = D_A2_Im.rows();
    int I_NCols_Im = D_A2_Im.cols();
    double d_sump;

    blitz::Array<int, 1> *P_I_A1_JBadVecArr = new blitz::Array<int, 1>(1);
    (*P_I_A1_JBadVecArr) = 0;

    blitz::Array<double, 1> a(1);
    a = 0.;

    blitz::Array<double, 2> AKLArr(1,1);
    AKLArr = 0.;

    blitz::Array<double, 1> b(1);
    b = 0.;

    blitz::Array<double, 2> BKLArr(1,1);
    BKLArr = 0.;

    blitz::Array<int, 1> BKLIndVecArr(1);
    BKLIndVecArr = 0;

    blitz::Array<double, 1> BLVecArr(1);
    BLVecArr = 0.;

    blitz::Array<double, 1> c(1);
    c = 0.;

    blitz::Array<double, 1> D_A1_Ind(1);
    D_A1_Ind = 0.;

    blitz::Array<double, 2> D_A2_AKLT(1,1);
    D_A2_AKLT = 0.;

    blitz::Array<int, 2> I_A2_Indarr_AKL;

    blitz::Array<double, 2> D_A2_OT(1,1);
    D_A2_OT = 0.;

    blitz::Array<double, 2> D_A2_SPVecTimesBKLArr(1, 1);
    D_A2_SPVecTimesBKLArr = 0.;

    blitz::Array<int, 1> IFirstVecArr(1);
    IFirstVecArr = 0;

//    blitz::Array<int, 1> ILastVecArr(1);
//    ILastVecArr = 0;

    blitz::Array<double, 2> *P_D_A2_Prof_Out = new blitz::Array<double, 2> (I_NRows_Im, I_NCols_Im);
    (*P_D_A2_Prof_Out) = 0.;

    blitz::Array<double, 2> *P_D_A2_Im_Out = new blitz::Array<double, 2> (I_NRows_Im, I_NCols_Im);
    (*P_D_A2_Im_Out) = 0.;

    blitz::Array<int, 1> IndVecArr(1);
    IndVecArr = 0;

    blitz::Array<double, 2> OArr(1,1);
    OArr = 0.;

    blitz::Array<int, 1> OIndVecArr(1);
    OIndVecArr = 0;

    blitz::Array<int, 1> OLIndVecArr(1);
    OLIndVecArr = 0;

    blitz::Array<double, 1> OOVecArr(I_NRows_Im);
    OOVecArr = 0.;

    blitz::Array<double, 1> OmegaVecArr(1);
    OmegaVecArr = 0.;

    blitz::Array<double, 2> OmegaArr(1,1);
    OmegaArr = 0.;

    blitz::Array<double, 2> D_A2_TempAA(1, 1);
    D_A2_TempAA = 0.;

    blitz::Array<double, 1> D_A1_TempDVecArr(1);
    D_A1_TempDVecArr = 0.;

    blitz::Array<double, 1> D_A1_TempDVecArr_Err(1);
    D_A1_TempDVecArr_Err = 0.;

    blitz::Array<double, 1> D_A1_TempDVecArrAA(1);
    D_A1_TempDVecArrAA = 0.;

    blitz::Array<double, 1> RVecArr(1);
    RVecArr = 0.;

    blitz::Array<double, 1> RVecArr_Err(1);
    RVecArr_Err = 0.;

    blitz::Array<double, 1> SFVecArr(I_NCols_Im);
    blitz::Array<double, 1> SPOldVecArr(1);
    SPOldVecArr = 0.;

    blitz::Array<double, 2> SSFArr(1,1);
    SSFArr = 0.;

    blitz::Array<double, 2> TempArray(1,1);
    TempArray = 0.;

    blitz::Array<double, 1> TempDVecArr(1);
    TempDVecArr = 0.;

    blitz::Array<double, 1> TempDVecArrB(1);
    TempDVecArrB = 0.;

    blitz::Array<double, 1> TempDVecArrC(1);
    TempDVecArrC = 0.;

    blitz::Array<double, 1> TempDVecArrD(1);
    TempDVecArrD = 0.;

    blitz::Array<double, 2> TempDArr(1,1);
    TempDArr = 0.;

    blitz::Array<int, 1> TempIVecArr(1);
    TempIVecArr = 0;

    blitz::Array<int, 2> *P_I_A2_Mask = new blitz::Array<int, 2>(1,1);
    blitz::Array<int, 2> *P_I_A2_MaskIn;

    blitz::Array<int, 1> UseRowVecArr(1);
    UseRowVecArr = 0;

    blitz::Array<int, 1> IVecArr(1);
    IVecArr = 0;

    blitz::Array<double, 1> XVecArr(1);
    XVecArr = 0.;

    blitz::Array<double, 1> XCenVecArr(1);
    XCenVecArr = 0.;

    blitz::Array<double, 2> D_A2_XX(1,1);
    D_A2_XX = 0.;
    blitz::Array<double, 1> D_A1_XX(1);
    D_A1_XX = 0.;

    blitz::Array<double, 2> D_A2_Sky(_trace->getHeight(), _trace->getWidth());
    D_A2_Sky = 0.;

    blitz::Array<double, 1> D_A1_MySF(1);
    blitz::Array<double, 1> D_A1_MySP(1);
    blitz::Array<double, 1> *P_D_A1_MySky = new blitz::Array<double,1>(1);
    blitz::Array<double, 2> D_A2_MySF(1,1);
    blitz::Array<double, 2> D_A2_MySP(1,1);
    blitz::Array<double, 2> D_A2_MySky(1,1);
    blitz::Array<double, 2> D_A2_ImTimesMask(I_NRows_Im,I_NCols_Im);
    blitz::Array<double, 2> D_A2_SFTimesMask(I_NRows_Im,I_NCols_Im);
    blitz::Array<double, 1> D_A1_OldSky(I_NRows_Im);
    blitz::Array<double, 2> *P_D_A2_Errors = new blitz::Array<double, 2>(I_NRows_Im, I_NCols_Im);
    *P_D_A2_Errors = blitz::sqrt(blitz::where(D_A2_Im > 0., D_A2_Im, 1.));
    blitz::Array<double, 1> *P_D_A1_ErrOut;
    blitz::Array<double, 1> *P_D_A1_ErrSky = new blitz::Array<double, 1>(1);
    blitz::Array<double, 1> *P_D_A1_SFO_Out = new blitz::Array<double, 1>(1);
    blitz::Array<double, 1> D_A1_ChiSquare_LinFit(1);
    blitz::Array<double, 1> D_A1_Probability_LinFit(1);
    blitz::Array<double, 2> D_A2_Sigma_LinFit(1,1);
    blitz::Array<double, 3> D_A3_CoVar_LinFit(1,1,1);

    blitz::Array<string, 1> S_A1_Args_Fit(10);
    S_A1_Args_Fit = " ";
    void **PP_Args_Fit = (void**)malloc(sizeof(void*) * 10);

    double Weight, Norm, Dev, tmpdbl, Lambda, XXX;
    double Lamb_SF = _fiberTraceProfileFittingControl->lambdaSF;
    double D_WingSmoothFactor = _fiberTraceProfileFittingControl->wingSmoothFactor;
    double Lamb_SP = _fiberTraceProfileFittingControl->lambdaSP;
    #ifdef __PISKUNOV_ORIG__
      double D_XLow = D_A2_Im.rows() / 2.;
    #endif
    #ifdef __DEBUG_SLITFUNC__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: Lamb_SF = " << Lamb_SF << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: Lamb_SP = " << Lamb_SP << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: D_WingSmoothFactor = " << D_WingSmoothFactor << endl;
    #endif

    std::string tempFileName(" ");
    std::string debugdir = DEBUGDIR;

    int I_NInd;
    int fiberTraceNumber = 0;
    Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "FIBERTRACENUMBER");
    if (Pos >= 0)
    {
      fiberTraceNumber = *(unsigned int*)ArgV_In[Pos];
    }

    bool ErrorsRead = true;

    blitz::firstIndex i;
    blitz::secondIndex j;

    Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "DEBUGFILES_SUFFIX");
    if (Pos >= 0)
    {
      debugFilesSuffix = *(string*)ArgV_In[Pos];
    }
    #ifdef __DEBUG_SLITFUNC_X__
      S_FileName_ImIn = DEBUGDIR + std::string("SlitFunc_ImIn") + debugFilesSuffix + std::string(".dat");
      ::pfs::drp::stella::utils::WriteArrayToFile(D_A2_Im, S_FileName_ImIn, std::string("ascii"));

      std::string sFileName_XCentersIn = DEBUGDIR + std::string("SlitFunc_XCentersIn") + debugFilesSuffix + std::string(".dat");
      ::pfs::drp::stella::utils::WriteArrayToFile(xCentersPixelFraction_In, sFileName_XCentersIn, std::string("ascii"));
    #endif

    Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "I_BIN");
    if (Pos >= 0)
    {
      I_Bin = *(int*)ArgV_In[Pos];
      #ifdef __DEBUG_SLITFUNC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: I_Bin set to " << I_Bin << endl;
      #endif
    }

//    Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "XLOW");
//    if (Pos >= 0)
//    {
//      D_XLow = *(double*)ArgV_In[Pos];
//      cout << "FiberTrace" << _iTrace << "::SlitFunc: D_XLow set to " << D_XLow << endl;
//    }

    /// TODO: get errors from maskedImage (Sigma or Variance???)
    if (ErrorsRead){
      Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "ERRORS");
//      #ifdef __DEBUG_SLITFUNC__
//        cout << "FiberTrace" << _iTrace << "::SlitFunc: KeyWord_Set(ERRORS): Pos = " << Pos << endl;
//        cout << "FiberTrace" << _iTrace << "::SlitFunc: KeyWord_Set(ERRORS): S_A1_Args_In = " << S_A1_Args_In << endl;
//        return false;
//      #endif
      if (Pos >= 0)
      {
        delete(P_D_A2_Errors);
        P_D_A2_Errors = (blitz::Array<double, 2>*)ArgV_In[Pos];
        #ifdef __DEBUG_SLITFUNC__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: *P_D_A2_Errors = " << *P_D_A2_Errors << endl;
        #endif
//        return false;
      }
      Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "ERRORS_OUT");
      if (Pos >= 0)
      {
        P_D_A1_ErrOut = (blitz::Array<double, 1>*)ArgV_In[Pos];
        P_D_A1_ErrOut->resize(D_A2_Im.rows());
        (*P_D_A1_ErrOut) = 0.;
      }
      Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "ERRORS_SP_OUT");
      if (Pos >= 0)
      {
        delete(P_D_A1_SPErrOut);
        P_D_A1_SPErrOut = (blitz::Array<double, 1>*)ArgV_In[Pos];
        P_D_A1_SPErrOut->resize(D_A2_Im.rows());
        (*P_D_A1_SPErrOut) = 0.;
      }
    }/// end if (this->ErrorsRead)

    Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "SP_OUT");
    if (Pos >= 0)
    {
      delete(P_D_A1_SPOut);
      P_D_A1_SPOut = (blitz::Array<double, 1>*)ArgV_In[Pos];
      P_D_A1_SPOut->resize(D_A2_Im.rows());
      (*P_D_A1_SPOut) = 0.;
    }

    Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "SFO_OUT");
    if (Pos >= 0)
    {
      delete(P_D_A1_SFO_Out);
      P_D_A1_SFO_Out = (blitz::Array<double, 1>*)ArgV_In[Pos];
      P_D_A1_SFO_Out->resize(D_A2_Im.cols() * overSample_In);
      (*P_D_A1_SFO_Out) = 0.;
    }

    blitz::Array<double, 1> SFVecArrTemp(1);
    SFVecArrTemp = 1.;
    blitz::Array<double, 1> *P_D_A1_XCorProfOut;
    int I_XCorProf = 0;
    int I_Pos = 0;
    if ((I_Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "XCOR_PROF")) >= 0)
    {
      I_XCorProf = *(int*)ArgV_In[I_Pos];
      P_D_A1_XCorProfOut = (blitz::Array<double, 1>*)ArgV_In[I_Pos+1];
      P_D_A1_XCorProfOut->resize(I_NRows_Im);
      (*P_D_A1_XCorProfOut) = 0.;
      if ((I_XCorProf > 0) && (I_XCorProf < 5))
        I_XCorProf = 5;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(XCOR_PROF): I_XCorProf set to " << I_XCorProf << endl;
    }
    else{
      P_D_A1_XCorProfOut = new blitz::Array<double, 1>(I_NRows_Im);
    }

    XCenVecArr.resize(xCentersPixelFraction_In.size());
    XCenVecArr = xCentersPixelFraction_In;
    #ifdef __DEBUG_SLITFUNC_XX__
//      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": XCenVecArr = " << XCenVecArr << endl;
      blitz::Array<double, 2> D_A2_XCentersPixelFraction(xCentersPixelFraction_In.size(), 2);
      D_A2_XCentersPixelFraction(blitz::Range::all(), 0) = i;
      D_A2_XCentersPixelFraction(blitz::Range::all(), 1) = XCenVecArr;
      std::string fnameXCen = DEBUGDIR + std::string("xCentersPixelFraction_In") + debugFilesSuffix + std::string(".dat");
      ::pfs::drp::stella::utils::WriteArrayToFile(D_A2_XCentersPixelFraction, fnameXCen, std::string("ascii"));
    #endif
    if (overSample_In < 1)
      overSample_In = 1;
    #ifdef __DEBUG_SLITFUNC__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": overSample_In = " << overSample_In << endl;
    #endif

    if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MASK")) >= 0)
    {
      P_I_A2_MaskIn = (blitz::Array<int, 2>*)ArgV_In[Pos];
      P_I_A2_Mask->resize(P_I_A2_MaskIn->rows(), P_I_A2_MaskIn->cols());
      (*P_I_A2_Mask) = (*P_I_A2_MaskIn);
      #ifdef __DEBUG_SLITFUNC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(MASK): P_I_A2_Mask read = " << *P_I_A2_Mask << endl;
      #endif
      if (max(*P_I_A2_Mask) > 1){
        string message("FiberTrace");
        message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ") + to_string(I_Bin);
        message += string(": ERROR: max(Mask) > 1");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      if (P_I_A2_Mask->size() != 1)
      {
        #ifdef __DEBUG_CHECK_INDICES__
          if (P_I_A2_Mask->rows() != I_NRows_Im ||
              P_I_A2_Mask->cols() != I_NCols_Im)
          {
            string message("SLIT_FUNC: Mask must have the same size as the image");
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
          }
        #endif
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(MASK): (*P_I_A2_Mask) set to " << (*P_I_A2_Mask) << endl;
        #endif

      }/// end if (P_TempMask->size() != 1)
    }/// end if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MASK")) >= 0)
    ///else
    if (Pos < 0 || (Pos >= 0 && P_I_A2_Mask->size() == 1))
    {
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(MASK): D_A2_Im.rows = " << I_NRows_Im << ", D_A2_Im.cols = " << I_NCols_Im << endl;
      #endif
      if (I_NCols_Im < 0)
      {
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(MASK): D_A2_Im = " << D_A2_Im << endl;
      }
      P_I_A2_MaskIn = new blitz::Array<int, 2>(I_NRows_Im, I_NCols_Im);
      (*P_I_A2_MaskIn) = 1;
      P_I_A2_Mask->resize(I_NRows_Im, I_NCols_Im);
      (*P_I_A2_Mask) = 1;
      #ifdef __DEBUG_SLITFUNC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(MASK): (*P_I_A2_Mask) set to " << (*P_I_A2_Mask) << endl;
      #endif

    }/// end if (Pos < 0 || (Pos >= 0 && P_I_A2_Mask->size() == 1))

    if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "PROF_OUT")) >= 0)
    {
      if (P_D_A2_Prof_Out != NULL)
        delete P_D_A2_Prof_Out;
      P_D_A2_Prof_Out = (blitz::Array<double, 2>*)ArgV_In[Pos];
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): P_D_A2_Prof_Out read " << endl;// to " << *P_D_A2_Prof_Out << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): P_D_A2_Prof_Out->size() = " << P_D_A2_Prof_Out->size() << endl;// to " << *P_D_A2_Prof_Out << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): P_D_A2_Prof_Out->rows() = " << P_D_A2_Prof_Out->rows() << endl;// to " << *P_D_A2_Prof_Out << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): P_D_A2_Prof_Out->cols() = " << P_D_A2_Prof_Out->cols() << endl;// to " << *P_D_A2_Prof_Out << endl;
      #endif
    }
    #ifdef __DEBUG_SLITFUNC__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: _fiberTraceFunction->fiberTraceFunctionControl.interpolation = " << _fiberTraceFunction->fiberTraceFunctionControl.interpolation << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: _fiberTraceFunction->fiberTraceFunctionControl.order = " << _fiberTraceFunction->fiberTraceFunctionControl.order << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: _fiberTraceFunction->fiberTraceFunctionControl.xLow = " << _fiberTraceFunction->fiberTraceFunctionControl.xLow << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: _fiberTraceFunction->fiberTraceFunctionControl.xHigh = " << _fiberTraceFunction->fiberTraceFunctionControl.xHigh << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: _fiberTraceFunction->xCenter = " << _fiberTraceFunction->xCenter << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: _fiberTraceFunction->yCenter = " << _fiberTraceFunction->yCenter << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: _fiberTraceFunction->yLow = " << _fiberTraceFunction->yLow << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: _fiberTraceFunction->yHigh = " << _fiberTraceFunction->yHigh << endl;
      for (int ii = 0; ii < static_cast<int>(_fiberTraceFunction->coefficients.size()); ii++)
        cout << "FiberTrace" << _iTrace << "::SlitFunc: _fiberTraceFunction->coefficients[" << ii << "] = " << _fiberTraceFunction->coefficients[ii] << endl;
    #endif
    P_D_A2_Prof_Out->resize(I_NRows_Im, I_NCols_Im);
    (*P_D_A2_Prof_Out) = 0.;
    #ifdef __DEBUG_SLITFUNC__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): P_D_A2_Prof_Out resized to (" << I_NRows_Im << ", " << I_NCols_Im << ")" << endl;
    #endif
    double D_SFMax = 0.;
    blitz::Array<double,1> D_A1_SFMax(I_NRows_Im);
    blitz::Array<double,1> D_A1_Sky(I_NRows_Im);
    D_A1_Sky = 0.;
    //  int I_NGoodSF = 0;
    blitz::Array<double,2> D_A2_MySFBest(1,1);
    blitz::Array<double,1> *P_D_A1_SFMax;
    blitz::Array<int,1> I_A1_SFMaxInd(1);
    int I_NGood;
    blitz::Array<int,1> *P_I_A1_SFMaxInd;
    blitz::Array<double,2> D_A2_MySF_Max(1);
    blitz::Array<double,1> D_A1_IndGen = ::pfs::drp::stella::math::Replicate(1., I_NRows_Im);
    blitz::Array<double,2> *P_D_A2_MySF;
    blitz::Array<double,1> D_A1_STDDEV(1);
    blitz::Array<double,1> D_A1_Covariance(1);
    blitz::Array<double, 2> D_A2_ImBak(D_A2_Im.rows(), D_A2_Im.cols());
    blitz::Array<double, 2> D_A2_ImBak_minus_Im(D_A2_Im.rows(), D_A2_Im.cols());
    int argpos=0;
    blitz::Array<double, 1> D_A1_Dev(D_A2_Im.cols());
    D_A1_Dev = 0.;
    #ifdef __PISKUNOV_ORIG__
      double D_Dev = 0.;
      double D_Dev_New = 0.;
    #endif
    blitz::Array<double, 2> D_A2_Mask(2,2);

    blitz::Array<int, 1> I_A1_Mask(1);
    I_A1_Mask = 0.;

    blitz::Array<double, 1> D_A1_Mask(1);
    D_A1_Mask = 0.;

    string S_MySF = "";
    string S_MySFTemp = "";

    #ifdef __DEBUG_TELLURIC__
      int I_Iter_Sky_Max=5;
    #endif

    int I_Telluric = 0;
    for ( int fooInt = FiberTraceProfileFittingControl::NONE; fooInt != FiberTraceProfileFittingControl::NVALUES; fooInt++ ){
      if (_fiberTraceProfileFittingControl->telluric.compare(_fiberTraceProfileFittingControl->TELLURIC_NAMES[fooInt]) == 0){
        I_Telluric = fooInt;
      }
    }
    #ifdef __DEBUG_SLITFUNC__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Telluric set to " << I_Telluric << endl;
    #endif
    if (I_Telluric == 2)
    {
      #ifdef __DEBUG_TELLURIC__
        cout << "__TELLURIC_MINE__ set" << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: S_A1_Args_In = " << S_A1_Args_In << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: S_A1_Args_In(0) = " << S_A1_Args_In(0) << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: S_A1_Args_In(1) = " << S_A1_Args_In(1) << endl;
      #endif

      /// take pointer to sky from argument for calculations
      Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "SKY");
      if (Pos >= 0)
      {
        delete(P_D_A1_MySky);
        P_D_A1_MySky = (blitz::Array<double,1>*)(ArgV_In[Pos]);
      }
      P_D_A1_MySky->resize(D_A2_Im.rows());
      (*P_D_A1_MySky) = 0.;

      if (ErrorsRead)
      {
        Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "ERR_SKY");
        if (Pos >= 0)
        {
          delete(P_D_A1_ErrSky);
          P_D_A1_ErrSky = (blitz::Array<double,1>*)(ArgV_In[Pos]);
        }
        P_D_A1_ErrSky->resize(D_A2_Im.rows());
        (*P_D_A1_ErrSky) = 0.;
      }

      D_A2_MySF.resize(I_NRows_Im, I_NCols_Im);
      #ifdef __DEBUG_TELLURIC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC = 2: D_A2_MySF.size() = " << D_A2_MySF << endl;
      #endif
      D_A2_MySF = D_A2_Im;
      #ifdef __DEBUG_TELLURIC__
        string S_MySF = DEBUGDIR + std::string("D_A2_Im") + debugFilesSuffix + std::string(".fits");
        ::pfs::drp::stella::utils::WriteFits(&D_A2_Im, S_MySF);
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: File " << *S_MySF << " written" << endl;
      #endif
      for (int p=0; p<I_NRows_Im; p++)
      {
        #ifdef __DEBUG_TELLURIC__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: for(p(=" << p << "); p<D_A2_MySF.rows()=" << D_A2_MySF.rows() << "; p++): D_A2_Im(p,*) = " << D_A2_Im(p,blitz::Range::all()) << endl;
        #endif

        /// Set MySF to 0. where < 3.*(-RON)
        D_A2_MySF(p, blitz::Range::all()) = blitz::where(D_A2_MySF(p, blitz::Range::all()) < (0. - (3. * _fiberTraceProfileFittingControl->ccdReadOutNoise)), 0., D_A2_MySF(p, blitz::Range::all()));

        /// Normalize rows of D_A2_MySF to 1.
        if (blitz::sum(D_A2_MySF(p, blitz::Range::all())) < 0.00000000000000001)
          D_A2_MySF(p, blitz::Range::all()) = 1.;
        D_A2_MySF(p,blitz::Range::all()) /= blitz::sum(D_A2_MySF(p,blitz::Range::all()));
        #ifdef __DEBUG_TELLURIC__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: for(p(=" << p << ")...): D_A2_MySF(p,*) = " << D_A2_MySF(p,blitz::Range::all()) << endl;
        #endif

        /// Get maximum of D_A2_MySF for every row
        D_A1_SFMax(p) = max(D_A2_MySF(p,blitz::Range::all()));

        #ifdef __DEBUG_TELLURIC__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: for(p(=" << p << ")...): D_A1_SFMax(p) = " << D_A1_SFMax(p) << endl;
        #endif
      }/// end for (int p=0; p<D_A2_MySF.rows(); p++)
      #ifdef __DEBUG_TELLURIC__
        S_MySF = DEBUGDIR + std::string("D_A2_MySF") + debugFilesSuffix + std::string(".fits");
        ::pfs::drp::stella::utils::WriteFits(&D_A2_MySF, S_MySF);
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: File " << S_MySF << " written" << endl;
      #endif

      /// remove elements with dev > 3. * stddev from D_A1_SFMax
      double D_DevOld = 0.;
      double D_DevTemp = 0.;
      int nind_temp;
      blitz::Array<int,1> I_A1_IndA(1);
      blitz::Array<double,2> D_A2_MySF_Max(I_NRows_Im, I_NCols_Im);
      blitz::Array<double,2> D_A2_MySF_MaxTemp(1, 1);
      for (int nn=0; nn < I_NCols_Im; nn++)
      {
        D_A2_MySF_Max(blitz::Range::all(), nn) = ::pfs::drp::stella::math::MedianVec(D_A2_MySF(blitz::Range::all(), nn), 5);
      }

      /// --- re-normalize D_A2_MySF_Max
      for (int nn=0; nn < I_NRows_Im; nn++)
      {
        double sum_sf = blitz::sum(D_A2_MySF_Max(nn, blitz::Range::all()));
        if (fabs(sum_sf) >= 0.00000000000000001)
          D_A2_MySF_Max(nn, blitz::Range::all()) /= sum_sf;
      }
      #ifdef __DEBUG_TELLURIC__
        S_MySF = DEBUGDIR + std::string("D_A2_MySF_Median") + debugFilesSuffix + std::string(".fits");
        ::pfs::drp::stella::utils::WriteFits(&D_A2_MySF_Max, S_MySF);
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: File " << S_MySF << " written" << endl;
      #endif

      /// TODO: ONLY TAKE HIGHEST VALUES, NOT MIDDLE ONES?!? <- Already did median filtering!
      /// --- remove elements from D_A1_SFMax which are outside the median value +/- 2sigma
      do
      {
        D_DevOld = D_DevTemp;
        D_DevTemp = sqrt(blitz::sum(pow((D_A1_SFMax) - ::pfs::drp::stella::math::Median(D_A1_SFMax),2)) / D_A1_SFMax.size());
        #ifdef __DEBUG_TELLURIC__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: while: D_DevTemp set to " << D_DevTemp << endl;
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: while: Median(D_A1_SFMax) = " << ::pfs::drp::stella::math::Median(D_A1_SFMax) << endl;
        #endif
        I_A1_IndA.resize(D_A1_SFMax.size());


        /** ************************************/


        I_A1_IndA = blitz::where(abs(D_A1_SFMax - ::pfs::drp::stella::math::Median(D_A1_SFMax)) < 2. * D_DevTemp,1,0);


        /** ************************************/



        blitz::Array<int,1> *P_I_A1_Ind = ::pfs::drp::stella::math::GetIndex(I_A1_IndA, nind_temp);
        #ifdef __DEBUG_TELLURIC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: while: I_A1_IndA set to " << I_A1_IndA << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: while: *P_I_A1_Ind set to " << *P_I_A1_Ind << endl;
        #endif
        P_D_A1_SFMax = new blitz::Array<double, 1>(D_A1_SFMax.size());
        *P_D_A1_SFMax = D_A1_SFMax;
        if (!::pfs::drp::stella::math::GetSubArrCopy(*P_D_A1_SFMax, *P_I_A1_Ind, D_A1_SFMax)){
          string message("FiberTrace");
          message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ") + to_string(I_Bin);
          message += string(": ERROR: GetSubArrCopy(P_D_A1_SFMax, P_I_A1_Ind, D_A1_SFMax) returned FALSE");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
        }
        delete(P_D_A1_SFMax);
        #ifdef __DEBUG_TELLURIC__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: while: D_A1_SFMax set to " << D_A1_SFMax << endl;
        #endif
        D_A2_MySF_MaxTemp.resize(D_A2_MySF_Max.rows(), D_A2_MySF_Max.cols());
        D_A2_MySF_MaxTemp = D_A2_MySF_Max;
        D_A2_MySF_Max.resize(P_I_A1_Ind->size(), I_NCols_Im);
        for (int nnn=0; nnn<static_cast<int>(P_I_A1_Ind->size()); nnn++){
          #ifdef __DEBUG_CHECK_INDICES__
            if (((*P_I_A1_Ind)(nnn) < 0) || ((*P_I_A1_Ind)(nnn) >= D_A2_MySF_MaxTemp.rows())){
              string message("FiberTrace");
              message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ") + to_string(I_Bin);
              message += string(": ERROR: ((*P_I_A1_Ind)(nnn=") + to_string(nnn) + string(")(=") + to_string((*P_I_A1_Ind)(nnn));
              message += string(") < 0) || ((*P_I_A1_Ind)(nnn) >= D_A2_MySF_MaxTemp.rows(=") + to_string(D_A2_MySF_MaxTemp.rows()) + string("))");
              delete(P_I_A1_Ind);
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
            }
          #endif
          D_A2_MySF_Max(nnn,blitz::Range::all()) = D_A2_MySF_MaxTemp((*P_I_A1_Ind)(nnn), blitz::Range::all());
        }
        #ifdef __DEBUG_TELLURIC__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: while: D_A2_MySF_Max set to " << D_A2_MySF_Max << endl;
          S_MySF = DEBUGDIR + std::string("D_A2_MySF_Max") + to_string(I_Iter_Sky_Max) + debugFilesSuffix + std::string(".fits");
          ::pfs::drp::stella::utils::WriteFits(&D_A2_MySF_Max, S_MySF);
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: File " << S_MySF << " written" << endl;
          I_Iter_Sky_Max++;
        #endif
        delete(P_I_A1_Ind);
      } while (D_DevTemp - D_DevOld > D_DevTemp / 100.);

      /// Get maximum of all maxima
      D_SFMax = max(D_A1_SFMax);

      /// Get index positions of D_A1_SFMax where D_A1_SFMax > D_SFMax - 10%
      I_A1_SFMaxInd.resize(D_A1_SFMax.size());
      I_A1_SFMaxInd = blitz::where(D_A1_SFMax > D_SFMax - (D_SFMax/50.),1,0);
      #ifdef __DEBUG_TELLURIC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_SFMax = " << D_SFMax << endl;
      #endif
      P_I_A1_SFMaxInd = ::pfs::drp::stella::math::GetIndex(I_A1_SFMaxInd, I_NGood);

      /// Create SF array of the highest Slit Functions
      D_A2_MySF_MaxTemp.resize(D_A2_MySF_Max.rows(), I_NCols_Im);
      D_A2_MySF_MaxTemp = D_A2_MySF_Max;
      D_A2_MySF_Max.resize(P_I_A1_SFMaxInd->size(), I_NCols_Im);
      ::pfs::drp::stella::math::GetSubArrCopy(D_A2_MySF_MaxTemp,
                                        *P_I_A1_SFMaxInd,
                                        0,
                                        D_A2_MySF_Max);
      #ifdef __DEBUG_TELLURIC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: P_I_A1_SFMaxInd set to " << *P_I_A1_SFMaxInd << endl;
        S_MySF = DEBUGDIR + std::string("D_A2_MySF_Max_Max") + debugFilesSuffix + std::string(".fits");
        ::pfs::drp::stella::utils::WriteFits(&D_A2_MySF_Max, S_MySF);
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: File " << S_MySF << " written" << endl;
      #endif
      delete(P_I_A1_SFMaxInd);

      /// Take median        or sum???           of all rows of D_A2_MySF_Max to define initial SlitFunction
      /// TODO: Switch D_A1_MySF to D_A2_MySF consistently
      D_A1_MySF.resize(I_NCols_Im);
      for (int p=0; p < I_NCols_Im; p++)
      {
        ///      D_A1_MySF(p) = blitz::sum(D_A2_MySF_Max(blitz::Range::all(),p));
        D_A1_MySF(p) = ::pfs::drp::stella::math::Median(D_A2_MySF_Max(blitz::Range::all(),p));
      }
      #ifdef __DEBUG_TELLURIC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A1_MySF set to " << D_A1_MySF << endl;
      #endif
      if (blitz::sum(D_A1_MySF) < 0.00000000000000001)
        D_A1_MySF = 1.;
      D_A1_MySF /= blitz::sum(D_A1_MySF);
      #ifdef __DEBUG_TELLURIC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A1_MySF set to " << D_A1_MySF << endl;
      #endif
      P_D_A1_MySky->resize(I_NRows_Im);

      /// Get initial values for Spectrum and Sky
      P_D_A2_MySF = ::pfs::drp::stella::math::VecArrACrossB(D_A1_IndGen,D_A1_MySF);
      D_A2_MySF.resize(P_D_A2_MySF->rows(), P_D_A2_MySF->cols());
      D_A2_MySF = *P_D_A2_MySF;
      delete(P_D_A2_MySF);
      #ifdef __DEBUG_TELLURIC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A2_MySF set to " << D_A2_MySF << endl;
        S_MySF = DEBUGDIR + std::string("D_A2_MySF_New") + debugFilesSuffix + std::string(".fits");
        ::pfs::drp::stella::utils::WriteFits(&D_A2_MySF, S_MySF);
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: File " << S_MySF << " written" << endl;
      #endif

      /// TODO: CHECK: are all D_A1_MySF.size() the same during one execution?
      SFVecArr.resize(D_A1_MySF.size());
      SFVecArr = D_A1_MySF;

      argpos = 0;
      if (ErrorsRead)
      {
        #ifdef __DEBUG_SLITFUNC__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": MEASURE_ERRORS_IN set to " << *P_D_A2_Errors << endl;
        #endif
        S_A1_Args_Fit(argpos) = "MEASURE_ERRORS_IN";
        PP_Args_Fit[argpos] = P_D_A2_Errors;
        argpos++;
      }

      S_A1_Args_Fit(argpos) = "CHISQ_OUT";
      PP_Args_Fit[argpos] = &D_A1_ChiSquare_LinFit;
      argpos++;

      S_A1_Args_Fit(argpos) = "SIGMA_OUT";
      PP_Args_Fit[argpos] = &D_A2_Sigma_LinFit;
      argpos++;

      S_A1_Args_Fit(argpos) = "Q_OUT";
      PP_Args_Fit[argpos] = &D_A1_Probability_LinFit;
      argpos++;

      D_A2_ImBak = D_A2_Im;
      #ifdef __DEBUG_TELLURIC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A2_Im = " << D_A2_Im << endl;
      #endif

      D_A1_Sky = 1.;
      #ifdef __DEBUG_TELLURIC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A2_Im = " << D_A2_Im << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: (*P_I_A2_Mask) = " << (*P_I_A2_Mask) << endl;
      #endif
      D_A2_ImTimesMask = D_A2_Im * (*P_I_A2_Mask);
      D_A2_SFTimesMask = D_A2_MySF * (*P_I_A2_Mask);
      #ifdef __DEBUG_TELLURIC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A2_ImTimesMask = " << D_A2_ImTimesMask << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A2_SFTimesMask = " << D_A2_SFTimesMask << endl;
      #endif
      if (!::pfs::drp::stella::math::LinFitBevington(D_A2_ImTimesMask,
                                               D_A2_SFTimesMask,
                                               D_A1_MySP,
                                               D_A1_Sky,
                                               S_A1_Args_Fit,
                                               PP_Args_Fit))
      {
        string message("FiberTrace");
        message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ") + to_string(I_Bin);
        message += string(": TELLURIC == 2: ERROR: Fit returned FALSE!");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      D_A1_OldSky = D_A1_Sky;
      #ifdef __DEBUG_TELLURIC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: after Fit: D_A1_MySP = " << D_A1_MySP << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: after Fit: D_A1_Sky = " << D_A1_Sky << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: after Fit: D_A1_ChiSquare_LinFit = " << D_A1_ChiSquare_LinFit << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: after Fit: D_A1_Probability_LinFit = " << D_A1_Probability_LinFit << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: after Fit: D_A2_Sigma_LinFit = " << D_A2_Sigma_LinFit << endl;
      #endif

      spectrum_Out.resize(D_A1_MySP.size());
      spectrum_Out = D_A1_MySP;

      /// Save initial sky to P_D_A1_MySky
      (*P_D_A1_MySky) = D_A1_Sky;
      #ifdef __DEBUG_TELLURIC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A1_MySP set to " << D_A1_MySP << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: *P_D_A1_MySky set to " << *P_D_A1_MySky << endl;
        string S_TempA = DEBUGDIR + std::string("D_A2_Im") + debugFilesSuffix + std::string(".fits");
        ::pfs::drp::stella::utils::WriteFits(&D_A2_Im, S_TempA);
      #endif
      for (int p=0; p < static_cast<int>(P_D_A1_MySky->size()); p++)
      {
        /// subtract sky from D_A2_Im
        D_A2_Im(p,blitz::Range::all()) -= D_A1_Sky(p);
        D_A2_Im(p,blitz::Range::all()) = blitz::where(D_A2_Im(p,blitz::Range::all()) < 0. - (3. * _fiberTraceProfileFittingControl->ccdReadOutNoise), 0., D_A2_Im(p,blitz::Range::all()));

        /// Add sky errors to error image
        if (ErrorsRead)
          (*P_D_A2_Errors)(p, blitz::Range::all()) += D_A2_Sigma_LinFit(p,1);
      }
      #ifdef __DEBUG_TELLURIC__
        S_MySF = DEBUGDIR + std::string("D_A2_Im_Minus_Sky") + debugFilesSuffix + std::string(".fits");
        ::pfs::drp::stella::utils::WriteFits(&D_A2_Im, S_MySF);
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: File " << S_MySF << " written" << endl;
      #endif
    }///end if (I_Telluric == 2)

    int Pos_Stop = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "STOP");
    int I_Stop = 0;
    if (Pos_Stop >= 0)
    {
      I_Stop = *(int*)(ArgV_In[Pos_Stop]);
    }

    for (int p=0; p < D_A2_Im.rows(); p++)
    {
      D_A2_Im(p,blitz::Range::all()) = blitz::where(D_A2_Im(p,blitz::Range::all()) < (0.-(3.*_fiberTraceProfileFittingControl->ccdReadOutNoise)), 0., D_A2_Im(p,blitz::Range::all()));
    }

    Weight = 1. / (double)overSample_In;

    /// Set OIndVecArr to blitz::Array<int,1>(overSample_In + 1) with values = index * (overSample_In + 2)
    UseRowVecArr = i;
    OIndVecArr.resize(overSample_In + 1);
    OIndVecArr = i;
    OIndVecArr *= (overSample_In + 2);
    #ifdef __DEBUG_SLITFUNC_N__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Weight = " << Weight << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": UseRowVecArr = " << UseRowVecArr << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": OIndVecArr = " << OIndVecArr << endl;
    #endif

    /// Set N to (number of columns in Im_In * Oversample) + OverSample + 1 (number of sub columns)
    I_NPixSlitF = ((I_NCols_Im + 1) * overSample_In) + 1;
    #ifdef __DEBUG_SLITFUNC_N__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_NPixSlitF set to " << I_NPixSlitF << endl;
    #endif

    /// Get Bad-pixel mask
    if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "BAD")) >= 0)
    {
      if (P_I_A1_JBadVecArr != NULL)
        delete P_I_A1_JBadVecArr;
      P_I_A1_JBadVecArr = (blitz::Array<int, 1>*)ArgV_In[Pos];
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(BAD): P_I_A1_JBadVecArr set to " << *P_I_A1_JBadVecArr << endl;
      #endif
    }/// end if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "BAD")) >= 0)

    if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "USE_ROWS")) >= 0)
    {
      blitz::Array<int, 1> *P_I_A1_UseRows = (blitz::Array<int, 1>*)ArgV_In[Pos];
      UseRowVecArr.resize(P_I_A1_UseRows->size());
      UseRowVecArr = *P_I_A1_UseRows;
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(USE_ROW): ArgV_In[Pos=" << Pos << "]=" << *(int*)(ArgV_In[Pos]) << " => UseRowVecArr set to " << UseRowVecArr << endl;
      #endif
      blitz::Array<double, 2> D_A2_ImUseRows(UseRowVecArr.size(), D_A2_Im.cols());
      if (!::pfs::drp::stella::math::GetSubArrCopy(D_A2_Im, UseRowVecArr, 0, D_A2_ImUseRows)){
        string message("FiberTrace");
        message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ") + to_string(I_Bin);
        message += string(": GetSubArrCopy(D_A2_Im, UseRowVecArr) returned FALSE");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      D_A2_Im.resize(D_A2_ImUseRows.rows(), D_A2_ImUseRows.cols());
      D_A2_Im = D_A2_ImUseRows;
      I_NRows_Im = D_A2_Im.rows();
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_Im = " << D_A2_Im << endl;
      #endif

      blitz::Array<double, 1> D_A1_XCentersUseRow(UseRowVecArr.size());
      if (!::pfs::drp::stella::math::GetSubArrCopy(XCenVecArr, UseRowVecArr, D_A1_XCentersUseRow)){
        string message("FiberTrace");
        message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ") + to_string(I_Bin);
        message += string(": GetSubArrCopy(XCenVecArr, UseRowVecArr) returned FALSE");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      XCenVecArr.resize(UseRowVecArr.size());
      XCenVecArr = D_A1_XCentersUseRow;
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": XCenVecArr = " << XCenVecArr << endl;
      #endif

      blitz::Array<int, 2> I_A2_MaskUseRow(UseRowVecArr.size(), P_I_A2_Mask->cols());
      if (!::pfs::drp::stella::math::GetSubArrCopy(*P_I_A2_Mask, UseRowVecArr, 0, I_A2_MaskUseRow)){
        string message("FiberTrace");
        message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ") + to_string(I_Bin);
        message += (": GetSubArrCopy(*P_I_A2_Mask, UseRowVecArr) returned FALSE");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      P_I_A2_Mask->resize(UseRowVecArr.size(), P_I_A2_Mask->cols());
      (*P_I_A2_Mask) = I_A2_MaskUseRow;
      #ifdef __DEBUG_SLITFUNC_A__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": (*P_I_A2_Mask) = " << (*P_I_A2_Mask) << endl;
      #endif
    }/// end if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "USE_ROW")) >= 0)
    if (Pos < 0)// || (Pos >= 0 && TempIntB == 0))
    {
      UseRowVecArr.resize(I_NRows_Im);
      UseRowVecArr = i;
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(USE_ROW): UseRowVecArr set to " << UseRowVecArr << endl;
      #endif
    }

    if (blitz::sum((*P_I_A2_Mask)) < 1)
    {
      string message("FiberTrace");
      message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ") + to_string(I_Bin);
      message += string(": ERROR: blitz::sum((*P_I_A2_Mask)) == 0");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }

    /** Set Norm to number of pixels in Im_In devided by blitz::sum((*P_I_A2_Mask)) **/
    Norm = (I_NRows_Im * I_NCols_Im) / blitz::sum((*P_I_A2_Mask));
    #ifdef __DEBUG_SLITFUNC_N__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Norm set to " << Norm << endl;
    #endif

    blitz::Array<double, 2> D_A2_SPTemp(spectrum_Out.size(), 1);

    blitz::Array<double, 2> D_A2_ImMedian(D_A2_Im.rows(), D_A2_Im.cols());
    D_A2_ImMedian = D_A2_Im;

    blitz::Array<double, 1> D_A1_XProf(D_A2_Im.cols());
    blitz::Array<double, 1> D_A1_YProf(D_A2_Im.cols());

    if (I_Telluric != 2)
    {
      #ifdef __PISKUNOV_ORIG__
        SFVecArr.resize(D_A2_Im.cols());
        spectrum_Out.resize(D_A2_Im.rows());
        blitz::Array<double, 2> D_A2_ImTimesMask_Guess(D_A2_Im.rows(), D_A2_Im.cols());
        D_A2_ImTimesMask_Guess = (D_A2_Im * (*P_I_A2_Mask));
        SFVecArr = blitz::sum(D_A2_ImTimesMask_Guess(j,i),j);
        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: PiskunovOrig: 1. SFVecArr set to " << SFVecArr << endl;
        #endif
        #ifdef __DEBUG_SLITFUNC_FILES__
          string S_SFOut = DEBUGDIR + std::string("PiskunovOrig_SFVecArr1") + debugFilesSuffix + std::string(".fits");
          ::pfs::drp::stella::utils::WriteFits(&SFVecArr, S_SFOut);
        #endif
        if ((overSample_In > 2) && (SFVecArr.size() > 5)){
          SFVecArr = ::pfs::drp::stella::math::MedianVec(SFVecArr, 5);
        }
        if (mean(blitz::sum(D_A2_Im(i,j),j)) < 1000.){
          blitz::Array<double, 1> D_A1_IndGenCols = ::pfs::drp::stella::math::DIndGenArr(D_A2_Im.cols());
          SFVecArr = exp(0. - blitz::pow2((D_A1_IndGenCols + D_XLow) / (D_A2_Im.cols() / 4.)));
        }
        SFVecArr = SFVecArr / blitz::sum(SFVecArr);
        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          string sSFOut = DEBUGDIR + std::string("PiskunovOrig_SFVecArr_DivBySum") + debugFilesSuffix + std::string(".fits");
          ::pfs::drp::stella::utils::WriteFits(&SFVecArr, sSFOut);
          cout << "FiberTrace" << _iTrace << "::SlitFunc: PiskunovOrig: 2. SFVecArr set to " << SFVecArr << endl;
        #endif
        blitz::Array<double, 1> D_A1_Rep = ::pfs::drp::stella::math::Replicate(1., D_A2_Im.rows());
        blitz::Array<double, 2> *P_D_A2_Mat = ::pfs::drp::stella::math::VecArrACrossB(D_A1_Rep, SFVecArr);
        D_A2_ImTimesMask_Guess = D_A2_ImTimesMask_Guess * (*P_D_A2_Mat);
        spectrum_Out = blitz::sum(D_A2_ImTimesMask_Guess(i,j),j) * P_I_A2_Mask->rows() * P_I_A2_Mask->cols() / blitz::sum(*P_I_A2_Mask);
        delete(P_D_A2_Mat);
        if (overSample_In > 2){
          spectrum_Out = ::pfs::drp::stella::math::MedianVec(spectrum_Out, 5);
        }
        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: PiskunovOrig: 1. spectrum_Out set to " << spectrum_Out << endl;
        #endif
        spectrum_Out = (spectrum_Out / blitz::sum(spectrum_Out)) * blitz::sum(D_A2_Im * (*P_I_A2_Mask));
        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: PiskunovOrig: 2. spectrum_Out set to " << spectrum_Out << endl;
        #endif
        P_D_A2_Mat = ::pfs::drp::stella::math::VecArrACrossB(spectrum_Out, SFVecArr);
        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: PiskunovOrig: (*P_D_A2_Mat) set to " << *P_D_A2_Mat << endl;
        #endif
        D_Dev = sqrt(blitz::sum((*P_I_A2_Mask) * blitz::pow2(D_A2_Im - (*P_D_A2_Mat))) / double(blitz::sum(*P_I_A2_Mask)));
        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: PiskunovOrig: D_Dev set to " << D_Dev << endl;
        #endif
        int I_NBad = 0;
        blitz::Array<int, 1> I_A1_WhereDev(D_A2_Im.cols());
        blitz::Array<double, 1> D_A1_WhereDev(D_A2_Im.cols());
        blitz::Array<int, 1> *P_I_A1_WhereDev;
        I_A1_WhereDev = 0;
        for (int i_row = 0; i_row < D_A2_Im.rows(); i_row++){
          D_A1_WhereDev = fabs(D_A2_Im(i_row, blitz::Range::all()) - (*P_D_A2_Mat)(i_row, blitz::Range::all()));
          #ifdef __DEBUG_SLITFUNC_PISKUNOV__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: PiskunovOrig: i_row = " << i_row << ": D_A1_WhereDev = " << D_A1_WhereDev  << endl;
          #endif
          I_A1_WhereDev = blitz::where(D_A1_WhereDev > 3. * D_Dev, 1, 0);
          P_I_A1_WhereDev = ::pfs::drp::stella::math::GetIndex(I_A1_WhereDev, I_NBad);
          if (I_NBad > 0){
            for (int i_ind = 0; i_ind < I_NBad; i_ind++){
              #ifdef __DEBUG_SLITFUNC_PISKUNOV__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: PiskunovOrig: i_row = " << i_row << ": i_ind = " << i_ind << ": (*P_I_A1_WhereDev)(i_ind) = " << (*P_I_A1_WhereDev)(i_ind)  << endl;
              #endif
              (*P_I_A2_Mask)(i_row, (*P_I_A1_WhereDev)(i_ind)) = 0;
            }
          }
          delete(P_I_A1_WhereDev);
        }
        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: PiskunovOrig: (*P_I_A2_Mask) set to " << *P_I_A2_Mask << endl;
        #endif
        D_A2_ImTimesMask_Guess = D_A2_Im * (*P_I_A2_Mask);
        #ifdef __DEBUG_SLITFUNC_FILES__
          string S_ImTimesMask = DEBUGDIR + std::string("PiskunovOrig_ImMTimesMask") + debugFilesSuffix + std::string(".fits");
          ::pfs::drp::stella::utils::WriteFits(&D_A2_ImTimesMask_Guess, S_ImTimesMask);
        #endif

        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          S_SP = DEBUGDIR + std::string("spectrum_Out1") + debugFilesSuffix + std::string(".fits");
          D_A2_SPTemp.resize(spectrum_Out.size(), 1);
          D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
          ::pfs::drp::stella::utils::WriteFits(&D_A2_SPTemp, S_SP);
        #endif
      #else
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_Im = " << D_A2_Im << endl;
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": *P_I_A2_Mask = " << *P_I_A2_Mask << endl;
        #endif
        if (maxIterSig_In > 0){
          for (int p = 0; p < I_NCols_Im; p++)
          {
            D_A2_ImMedian(blitz::Range::all(), p) = ::pfs::drp::stella::math::MedianVec(D_A2_Im(blitz::Range::all(),p), 5, string("NORMAL"));
          }
        }
        D_A2_ImTimesMask = D_A2_ImMedian * (*P_I_A2_Mask);
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_ImTimesMask = " << D_A2_ImTimesMask << endl;
        #endif

        /// MINE:
        for (int p = 0; p < I_NRows_Im; p++)
        {
          d_sump = blitz::sum(D_A2_ImTimesMask(p, blitz::Range::all()));
          if (d_sump > 0.)
            D_A2_ImTimesMask(p, blitz::Range::all()) /= d_sump;
          else
            D_A2_ImTimesMask(p, blitz::Range::all()) = 0.;
        }

        SFVecArr.resize(I_NCols_Im);
        SFVecArr = blitz::sum(D_A2_ImTimesMask(j, i), j); /** Initial guess for the **/
        if (abs(blitz::sum(SFVecArr)) < 0.000000001)
        {
          #ifdef __DEBUG_SLITFUNC_N__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": WARNING: blitz::sum(SFVecArr) == 0 => Setting to 1." << endl;
          #endif
          SFVecArr = 1.;
        }

        if (blitz::sum(SFVecArr) < 0.00000000000000001)
          SFVecArr = 1.;
        SFVecArr /= blitz::sum(SFVecArr);           /** Slit Function **/
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": 3.    Slit Function SFVecArr = SFVecArr / (blitz::sum(SFVecArr) = " << blitz::sum(SFVecArr) << ") = " << SFVecArr << endl;
        #endif

        /** Initial guess for the spectrum **/
        TempArray.resize(I_NRows_Im, I_NCols_Im);
        TempArray = D_A2_Im;
        TempArray *= (*P_I_A2_Mask);
        #ifdef __DEBUG_SLITFUNC_A__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": (*P_I_A2_Mask) = " << (*P_I_A2_Mask) << endl;
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TempArray = ImM*(*P_I_A2_Mask) = " << TempArray << endl;
        #endif

        /** weight rows of im_in with slit function, sum to estimate the spectrum, multiply with Norm, take median over 5 pixels, normalize to blitz::sum(row)=1, and multiply with sum of im_in **/
        blitz::Array<double, 1> d1rep = ::pfs::drp::stella::math::Replicate(1., I_NRows_Im);
        blitz::Array<double, 2> *p_d2mat = ::pfs::drp::stella::math::VecArrACrossB(d1rep, SFVecArr);
        TempArray *= (*p_d2mat);
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TempArray = TempArray*(VecArrACrossB(Replicate(1.,D_A2_Im.rows(=" << I_NRows_Im << "))=" << d1rep << ", SFVecArr(=" << SFVecArr << "))=" << *p_d2mat << ") = " << TempArray << endl;
        #endif
        delete p_d2mat;

        spectrum_Out.resize(I_NRows_Im);
        spectrum_Out = blitz::sum(TempArray, j);

        #ifdef __DEBUG_SLITFUNC_FILES__
          S_SP = DEBUGDIR + std::string("spectrum_Out1") + debugFilesSuffix + std::string(".fits");
          D_A2_SPTemp.resize(spectrum_Out.size(), 1);
          D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
          ::pfs::drp::stella::utils::WriteFits(&D_A2_SPTemp, S_SP);
        #endif
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": -3. spectrum_Out (set to blitz::sum(D_A2_Im * (*P_I_A2_Mask), j)) = " << spectrum_Out << endl;
        #endif
        spectrum_Out *= Norm;

        #ifdef __DEBUG_SLITFUNC_FILES__
          S_SP = DEBUGDIR + std::string("spectrum_Out1TimesNorm") + debugFilesSuffix + std::string(".fits");
          D_A2_SPTemp.resize(spectrum_Out.size(), 1);
          D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
          ::pfs::drp::stella::utils::WriteFits(&D_A2_SPTemp, S_SP);
        #endif
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": -2. spectrum_Out (*Norm=" << Norm << ") = " << spectrum_Out << endl;
        #endif

        spectrum_Out = blitz::where(spectrum_Out < 0., 0., spectrum_Out);
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": -1. spectrum_Out (*Norm=" << Norm << ") = " << spectrum_Out << endl;
        #endif
        spectrum_Out /= blitz::sum(spectrum_Out);
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": 0. spectrum_Out (set to /=blitz::sum(spectrum_Out)) = " << spectrum_Out << endl;
        #endif

        #ifdef __DEBUG_SLITFUNC_FILES__
          S_SP = DEBUGDIR + std::string("spectrum_Out1DivBySum") + debugFilesSuffix+ std::string(".fits");
          D_A2_SPTemp.resize(spectrum_Out.size(), 1);
          D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
          ::pfs::drp::stella::utils::WriteFits(&D_A2_SPTemp, S_SP);
        #endif

        /// used to be before spectrum_Out /= blitz::sum(spectrum_Out)
        if (abs(blitz::sum(spectrum_Out)) < 0.000000001)
        {
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": WARNING: blitz::sum(spectrum_Out=" << spectrum_Out << ") == 0" << endl;
          spectrum_Out = 1.;
        }
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum(spectrum_Out) =" << blitz::sum(spectrum_Out) << endl;
        #endif

        spectrum_Out *= blitz::sum(D_A2_Im * (*P_I_A2_Mask));

        #ifdef __DEBUG_SLITFUNC_FILES__
          S_SP = DEBUGDIR + std::string("spectrum_Out1TimesSumImTimesMask") + debugFilesSuffix + std::string(".fits");
          D_A2_SPTemp.resize(spectrum_Out.size(), 1);
          D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
          ::pfs::drp::stella::utils::WriteFits(&D_A2_SPTemp, S_SP);
        #endif
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": spectrum_Out (set to *=blitz::sum(D_A2_Im * (*P_I_A2_Mask))(=" << blitz::sum(D_A2_Im * (*P_I_A2_Mask)) << ")) = " << spectrum_Out << endl;
        #endif
      #endif
    }/// end if (I_Telluric != 2)

    /** Add too noisy pixels to bad-pixel mask **/
    Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "NOISE");
    blitz::Array<double, 2> D_A2_TempBB(P_I_A2_Mask->rows(), P_I_A2_Mask->cols());
    if (Pos >= 0)
    {
      Dev = *(double*)ArgV_In[Pos];
      D_A1_Dev = Dev;
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(NOISE): Dev set to " << Dev << endl;
      #endif

    }
    if (Pos < 0 || (Pos >= 0 && abs(Dev) < 0.00000000000000001))
    {
      if (blitz::sum((*P_I_A2_Mask)) < 1)
      {
        string message("FiberTrace");
        message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ") + to_string(I_Bin);
        message += string(": ERROR: blitz::sum(*P_I_A2_Mask) == 0");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
    }
    #ifdef __DEBUG_SLITFUNC_N__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": 1. spectrum_Out = " << spectrum_Out << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": SFVecArr = " << SFVecArr << endl;
    #endif
    #ifdef __DEBUG_CHECK_INDICES__
      if (D_A2_Im.cols() != static_cast<int>(SFVecArr.size())){
        string message("FiberTrace");
        message += to_string(_iTrace) + string("::SlitFunc: ERROR: D_A2_Im.cols() != SFVecArr.size()");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
    #endif
    blitz::Array<double, 1> D_A1_SP_Tmp(D_A2_Im.rows());
    D_A1_SP_Tmp = 0.;
    double D_Reject = 4.;
    blitz::Array<string, 1> S_A1_Args_FitSig(3);
    void **PP_Void_FitSig = (void**)malloc(sizeof(void*) * 3);
    S_A1_Args_FitSig(0) = "REJECT_IN";
    PP_Void_FitSig[0] = &D_Reject;
    if (ErrorsRead){
      S_A1_Args_FitSig(1) = "MEASURE_ERRORS_IN";
      PP_Void_FitSig[1] = P_D_A2_Errors;
    }
    blitz::Array<int, 2> I_A2_MaskFit(D_A2_Im.rows(), D_A2_Im.cols());
    I_A2_MaskFit = (*P_I_A2_Mask);
    S_A1_Args_FitSig(2) = "MASK_INOUT";
    PP_Void_FitSig[2] = &I_A2_MaskFit;
    //blitz::Array<double, 1> Rep = ::pfs::drp::stella::math::Replicate(1., D_A2_Im.rows());
    blitz::Array<double, 2> *p_SFArr = ::pfs::drp::stella::math::VecArrACrossB(::pfs::drp::stella::math::Replicate(1., D_A2_Im.rows()), SFVecArr);
    blitz::Array<double, 1> D_A1_Sky_Tmp(D_A2_Im.rows());
    bool B_WithSky = true;
    if (I_Telluric == 1 || I_Telluric == 3)
      B_WithSky = false;
    #ifdef __DEBUG_SLITFUNC__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: Before FitSig: SFVecArr = " << SFVecArr << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A2_Im = " << D_A2_Im << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: *p_SFArr = " << *p_SFArr << endl;
      if (B_WithSky)
        cout << "FiberTrace" << _iTrace << "::SlitFunc: B_WithSky = true" << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: D_Reject = " << D_Reject << endl;
      if (ErrorsRead)
        cout << "FiberTrace" << _iTrace << "::SlitFunc: *P_D_A2_Errors = " << *P_D_A2_Errors << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: I_A2_MaskFit = " << I_A2_MaskFit << endl;
    #endif
    if (!::pfs::drp::stella::math::LinFitBevington(D_A2_Im,
                                             *p_SFArr,
                                             D_A1_SP_Tmp,
                                             D_A1_Sky_Tmp,
                                             B_WithSky,
                                             S_A1_Args_FitSig,
                                             PP_Void_FitSig)){
      string message("FiberTrace");
      message += to_string(_iTrace) + string("::SlitFunc: ERROR: FitSig returned FALSE");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    *P_I_A2_Mask = I_A2_MaskFit;
    #ifdef __DEBUG_SLITFUNC_N__
      string S_MaskSig = DEBUGDIR + std::string("MaskFitSig") + debugFilesSuffix + std::string(".fits");
      ::pfs::drp::stella::utils::WriteFits(&I_A2_MaskFit, S_MaskSig);
      cout << "FiberTrace" << _iTrace << "::SlitFunc: After FitSig: I_A2_MaskFit = " << I_A2_MaskFit << endl;
    #endif
    delete(p_SFArr);
    free(PP_Void_FitSig);

    SFVecArr = blitz::where(SFVecArr < 0., 0., SFVecArr);
    #ifdef __DEBUG_SLITFUNC_N__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": 2c.    Slit Function SFVecArr = SFVecArr / (blitz::sum(SFVecArr) = " << blitz::sum(SFVecArr) << ") = " << SFVecArr << endl;
    #endif

    if (blitz::sum(SFVecArr) < 0.00000000000000001)
      SFVecArr = 1.;
    SFVecArr /= blitz::sum(SFVecArr);           /** Slit Function **/
    #ifdef __DEBUG_SLITFUNC__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: c) SFVecArr = " << SFVecArr << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A1_SP_Tmp = " << D_A1_SP_Tmp << endl;
    #endif
    blitz::Array<double, 2> *p_tempMatA = ::pfs::drp::stella::math::VecArrACrossB(D_A1_SP_Tmp, SFVecArr);
    blitz::Array<double, 2> *p_tempMat = new blitz::Array<double, 2>(D_A2_Im.rows(), D_A2_Im.cols());
    blitz::Array<double, 2> *p_tempMatB = new blitz::Array<double, 2>(D_A2_Im.rows(), D_A2_Im.cols());
    *p_tempMat = D_A2_Im - (*p_tempMatA);
    #ifdef __DEBUG_SLITFUNC_FILES__
      string S_ImMinusRec = DEBUGDIR + std::string("ImMinusRec") + debugFilesSuffix + std::string(".fits");
      ::pfs::drp::stella::utils::WriteFits(p_tempMat, S_ImMinusRec);
    #endif
    #ifdef __DEBUG_SLITFUNC__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(NOISE): *p_tempMatA " << *p_tempMatA << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(NOISE): *p_tempMat " << *p_tempMat << endl;
    #endif
    *p_tempMatB = blitz::pow2(*p_tempMat);
    for (unsigned int iter_sig = 0; iter_sig < maxIterSig_In; iter_sig++)
    {
      ::pfs::drp::stella::math::Double((*P_I_A2_Mask), D_A2_Mask);
      #ifdef __DEBUG_CHECK_INDICES__
        if (P_I_A2_Mask->rows() != D_A2_Mask.rows()){
          string message("FiberTrace");
          message += to_string(_iTrace) + string("::SlitFunc: ERROR: 1. P_I_A2_Mask->rows() != D_A2_Mask.rows()");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
        }
      #endif
      Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "NOISE");
      if (Pos < 0 || (Pos >= 0 && abs(Dev) < 0.00000000000000001))
      {
        for (int i_col=0; i_col < static_cast<int>(D_A2_Im.cols()); i_col++){
          if (blitz::sum((*P_I_A2_Mask)(blitz::Range::all(), i_col)) == 0){
            string message("FiberTrace");
            message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ") + to_string(I_Bin);
            message += string(": !KeyWord_Set(NOISE): iter_sig = ") + to_string(iter_sig) + string(": i_col = ") + to_string(i_col);
            message += string(": ERROR: blitz::sum((*P_I_A2_Mask)(blitz::Range::all(), i_col)) = ") + to_string(blitz::sum((*P_I_A2_Mask)(blitz::Range::all(), i_col)));
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
          }
          else{
            if (fabs(blitz::sum(D_A2_Mask)) < 0.000000000001){
              string message("FiberTrace");
              message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ");
              message += to_string(I_Bin) + string(": !KeyWord_Set(NOISE): iter_sig = ") + to_string(iter_sig) + string(": i_col = ") + to_string(i_col);
              message += string(": ERROR:  fabs(blitz::sum(D_A2_Mask)=") + to_string(blitz::sum(D_A2_Mask)) + string(") < 0.000000000001");
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
           }
            D_A1_Dev(i_col) = sqrt(blitz::sum(D_A2_Mask(blitz::Range::all(), i_col) * ((*p_tempMatB)(blitz::Range::all(), i_col))) / blitz::sum(D_A2_Mask(blitz::Range::all(), i_col)));
          }
        }
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(NOISE): iter_sig = " << iter_sig << ": D_A1_Dev set to " << D_A1_Dev << endl;
        #endif
      }
      else{
        cout << "FiberTrace" << _iTrace << "::SlitFunc: KeyWord_Set(NOISE): D_A1_Dev = " << D_A1_Dev << endl;
      }

      ::pfs::drp::stella::math::Double((*P_I_A2_Mask), D_A2_Mask);
      #ifdef __DEBUG_CHECK_INDICES__
        if (P_I_A2_Mask->rows() != D_A2_Mask.rows()){
          string message("FiberTrace");
          message += to_string(_iTrace) + string("::SlitFunc: ERROR: 0. P_I_A2_Mask->rows() != D_A2_Mask.rows()");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
        }
      #endif
      #ifdef __DEBUG_SLITFUNC_N__
        D_A2_TempBB = fabs(D_A2_Mask * (*p_tempMat));
        cout << " iter_sig = " << iter_sig << ": fabs(P_I_A2_Mask(= " << *P_I_A2_Mask << ") * p_tempMat(= " << *p_tempMat << ")) = " << D_A2_TempBB << ", D_A1_Dev = " << D_A1_Dev << endl;
      #endif

      for (int i_col = 0; i_col < D_A2_Im.cols(); i_col++){
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": iter_sig = " << iter_sig << ": (*p_tempMatB)(blitz::Range::all(), i_col=" << i_col << ") = " << (*p_tempMatB)(blitz::Range::all(), i_col) << endl;
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": iter_sig = " << iter_sig << ": D_A1_Dev(i_col=" << i_col << ") = " << D_A1_Dev(i_col) << endl;
        #endif
        #ifndef __PISKUNOV_ORIG__
          if (maxIterSig_In > 0){
            (*P_I_A2_Mask)(blitz::Range::all(), i_col) = blitz::where(sqrt(D_A2_Mask(blitz::Range::all(), i_col) * ((*p_tempMatB)(blitz::Range::all(), i_col))) > (6.5 * D_A1_Dev(i_col)), 0, (*P_I_A2_Mask)(blitz::Range::all(), i_col));
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": iter_sig = " << iter_sig << ": (*P_I_A2_Mask)(blitz::Range::all(), i_col=" << i_col << ") = " << (*P_I_A2_Mask)(blitz::Range::all(), i_col) << endl;
            #endif
          }
        #endif
      }///end for (int i_col = 0; i_col < D_A2_Im.cols(); i_col++){
      #ifdef __DEBUG_SLITFUNC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": iter_sig = " << iter_sig << ": (*P_I_A2_Mask) = " << (*P_I_A2_Mask) << endl;
      #endif
      ::pfs::drp::stella::math::Double((*P_I_A2_Mask), D_A2_Mask);
      #ifdef __DEBUG_CHECK_INDICES__
        if (P_I_A2_Mask->rows() != D_A2_Mask.rows()){
          string message("FiberTrace");
          message += to_string(_iTrace) + string("::SlitFunc: ERROR: P_I_A2_Mask->rows() != D_A2_Mask.rows()");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
        }
        for (int i_row=0; i_row<P_I_A2_Mask->rows(); i_row++){
          for (int i_col=0; i_col<P_I_A2_Mask->cols(); i_col++){
            if ((*P_I_A2_Mask)(i_row, i_col) != int(D_A2_Mask(i_row, i_col))){
              string message("FiberTrace");
              message += to_string(_iTrace) + string("::SlitFunc: ERROR: (*P_I_A2_Mask)(i_row, i_col)(=") + to_string((*P_I_A2_Mask)(i_row, i_col));
              message += string(") != int(D_A2_Mask(i_row, i_col))(=") + to_string(int(D_A2_Mask(i_row, i_col))) + string(")");
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
            }
          }
        }
      #endif
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "blitz::sum(*P_I_A2_MaskIn) = " << blitz::sum(*P_I_A2_MaskIn) << endl;
        cout << "blitz::sum(*P_I_A2_Mask) = " << blitz::sum(*P_I_A2_Mask) << endl;
        blitz::Array<double,2> D_A2_ImTemp(D_A2_Im.rows(), D_A2_Im.cols());
        D_A2_ImTemp = D_A2_Im - (*p_tempMatA);
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_Im[" << D_A2_Im.rows() << ", " << D_A2_Im.cols() << "]" << ", p_tempMatA[" << p_tempMatA->rows() << ", " << p_tempMatA->cols() << "]" << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_Im - spectrum_Out x SFVecArr = " << D_A2_ImTemp << endl;
      #endif

      #ifdef __DEBUG_SLITFUNC_FILES__
        string S_Mask = DEBUGDIR + std::string("Mask_IterSig") + to_string(iter_sig) + debugFilesSuffix + std::string(".fits");
        ::pfs::drp::stella::utils::WriteFits(P_I_A2_Mask, S_Mask);

        S_Mask = DEBUGDIR + std::string("ImInTimesMask_IterSig") + to_string(iter_sig) + debugFilesSuffix + std::string(".fits");
        D_A2_ImTimesMask = D_A2_Im * (*P_I_A2_Mask);
        ::pfs::drp::stella::utils::WriteFits(&D_A2_ImTimesMask, S_Mask);
      #endif
    }/// for (int iter_sig = 0; iter_sig < this->I_MaxIterSig; iter_sig++)
    delete p_tempMat;
    delete p_tempMatA;
    delete p_tempMatB;

    /** Set XVecArr to vector containing the subpixel numbers **/

    XVecArr.resize(I_NPixSlitF);
    XVecArr = (i + 0.5) / double(overSample_In) - 1.;
    ///TODO: check for the + 0.5 and - 1.
//    XVecArr = (i + 0.5) / double(overSample_In) - 1.;
//    #ifdef __DEBUG_SLITFUNC_X__
//      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": XVecArr = " << XVecArr << endl;
//    #endif
    BKLIndVecArr.resize(overSample_In + 1);
    BKLIndVecArr = i;
    BKLIndVecArr += (I_NPixSlitF * overSample_In);
    #ifdef __DEBUG_SLITFUNC_N__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": BKLIndVecArr = " << BKLIndVecArr << endl;
    #endif

    OLIndVecArr.resize(overSample_In + 1);
    #ifdef __DEBUG_CHECK_INDICES__
      if (static_cast<int>(OIndVecArr.size()) < overSample_In+1)
      {
        string message("FiberTrace");
        message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ") + to_string(I_Bin);
        message += string(": ERROR: size of OIndVecArr(=") + to_string(OIndVecArr.size()) + string(" < overSample_In + 1(=") + to_string(overSample_In + 1);
        message += string(")");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
    #endif
    OLIndVecArr = OIndVecArr(blitz::Range(0, overSample_In));
    #ifdef __DEBUG_SLITFUNC_N__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": OLIndVecArr = " << OLIndVecArr << endl;
    #endif

    for (long m=overSample_In + 1; m <= (2 * overSample_In); m++)
    {
      long mm = m - overSample_In;
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m=" << m << "): mm = " << mm << endl;
      #endif

      /// attach another Vector<int>(osample+1-mm) to BKLIndVecArr with values index + (N * m)
      TempIVecArr.resize(overSample_In + 1 - mm);
      TempIVecArr = i + (I_NPixSlitF * m);
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m=" << m << "): TempIVecArr = " << TempIVecArr << endl;
      #endif

      int oldsize = BKLIndVecArr.size();
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m=" << m << "): oldsize(BKLIndVecArr) = " << oldsize << endl;
      #endif
      BKLIndVecArr.resizeAndPreserve(oldsize + overSample_In + 1 - mm);
      BKLIndVecArr(blitz::Range(oldsize, blitz::toEnd))//oldsize + 1 + overSample_In - mm))
        = TempIVecArr(blitz::Range::all());
//      #ifdef __DEBUG_SLITFUNC_X__
//        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m=" << m << "): BKLIndVecArr = " << BKLIndVecArr << endl;
//      #endif

      oldsize = OLIndVecArr.size();
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m=" << m << "): oldsize(OLIndVecArr) = " << oldsize << endl;
      #endif
      OLIndVecArr.resizeAndPreserve(OLIndVecArr.size() + overSample_In + 1 - mm);
      #ifdef __DEBUG_CHECK_INDICES__
        if (static_cast<int>(OIndVecArr.size()) < overSample_In - mm + 1)
        {
          string message("FiberTrace");
          message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ") + to_string(I_Bin);
          message += string(": ERROR: size of OIndVecArr(=") + to_string(OIndVecArr.size()) + string(") < overSample_In - mm + 1(=");
          message += to_string(overSample_In - mm + 1) + string(")");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
        }
      #endif
      OLIndVecArr(blitz::Range(oldsize, blitz::toEnd))
        = OIndVecArr(blitz::Range(0, overSample_In - mm)) + mm;
    }/// end for (long m=overSample_In + 1; m <= (2 * overSample_In); m++)

    SPOldVecArr.resize(spectrum_Out.size());
    SPOldVecArr = 0.;
    #ifdef __DEBUG_SLITFUNC__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": SPOldVecArr = " << SPOldVecArr << endl;
    #endif

    ::pfs::drp::stella::math::Double((*P_I_A2_Mask), D_A2_Mask);

    blitz::Array<double, 1> D_A1_SFO(2);
    D_A1_SFO = 0.;
    blitz::Array<double, 2> D_A2_Weights(D_A2_Im.rows(), overSample_In + 1);
    D_A2_Weights = 0.;

    blitz::Array<int, 1> I_A1_IFirstPix(D_A2_Im.rows());
    blitz::Array<int, 1> I_A1_ILastPix(D_A2_Im.rows());
    blitz::Array<int, 1> I_A1_IFirstSpec(D_A2_Im.rows());
    blitz::Array<int, 1> I_A1_ILastSpec(D_A2_Im.rows());

    D_A2_XX.resize(I_NRows_Im, I_NPixSlitF);
    D_A2_XX = 0.;
    D_A1_XX.resize(I_NPixSlitF);
    for (int m = 0; m < I_NRows_Im; m++)  /** Fill up matrix and RHS **/
    {
      D_A2_XX(m,blitz::Range::all()) = XVecArr + XCenVecArr(m);    /** Offset SFVecArr **/
      #ifdef __DEBUG_CHECK_INDICES__
        if (D_A2_XX.cols() != static_cast<int>(XVecArr.size()))
        {
          string message("FiberTrace");
          message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ") + to_string(I_Bin);
          message += string(": I_Iter_SF = ") + to_string(I_Iter_SF) + string(": ERROR: D_A2_XX.cols(=") + to_string(D_A2_XX.cols());
          message += string(") != size of XVecArr(=") + to_string(XVecArr.size()) + string(")");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
        }
      #endif

      /** Weights are the same for all pixels except for the first and the last subpixels **/
      TempIVecArr.resize(D_A2_XX.cols());
      TempIVecArr = blitz::where((D_A2_XX(m,blitz::Range::all()) > 0) && (D_A2_XX(m,blitz::Range::all()) < 1), 1, 0);
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): TempIVecArr set to " << TempIVecArr << endl;
      #endif
      ::pfs::drp::stella::math::GetIndex(TempIVecArr, NInd, IndVecArr);
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): NInd = " << NInd << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): IndVecArr set to " << IndVecArr << endl;
      #endif
      I_A1_IFirstPix(m) = IndVecArr(0);

      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): I_A1_IFirstPix(" << m << ") = " << I_A1_IFirstPix(m) << endl;
      #endif
      I_A1_ILastPix(m)  = IndVecArr(NInd - 1);
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): I_A1_ILastPix(" << m << ") = " << I_A1_ILastPix(m) << endl;
      #endif

      #ifdef __DEBUG_SLITFUNC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): D_A2_XX(m,*) = " << D_A2_XX(m,blitz::Range::all()) << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): D_A2_Im->cols = " << I_NCols_Im << endl;
      #endif

      TempIVecArr = blitz::where((D_A2_XX(m,blitz::Range::all()) >= 0.) && (D_A2_XX(m,blitz::Range::all()) < (double)(I_NCols_Im)), 1, 0);

      #ifdef __DEBUG_SLITFUNC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): TempIVecArr(where) = " << TempIVecArr << endl;
      #endif

      i_tmp_sum = blitz::sum(TempIVecArr);
      if (i_tmp_sum == 0)
      {
        string message("FiberTrace");
        message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ") + to_string(I_Bin);
        message += string(": KeyWord_Set(PROF_OUT): ERROR: i_tmp_sum == 0!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      ::pfs::drp::stella::math::GetIndex(TempIVecArr, I_NInd, IFirstVecArr);
      I_A1_IFirstSpec(m) = IFirstVecArr(0);
      I_A1_ILastSpec(m) = IFirstVecArr(IFirstVecArr.size() - 1);
      #ifdef __DEBUG_SLITFUNC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): I_A1_IFirstSpec(m) set to " << I_A1_IFirstSpec(m) << endl;
        cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): I_A1_ILastSpec(m) set to " << I_A1_ILastSpec(m) << endl;
      #endif

    }/// end for (int m = 0; m < I_NRows_Im; m++)
    #ifdef __DEBUG_SLITFUNC_X__
      std::string fname_ifirst = DEBUGDIR + std::string("IFirstPix") + debugFilesSuffix + std::string(".dat");
      ::pfs::drp::stella::utils::WriteArrayToFile(I_A1_IFirstPix, fname_ifirst, std::string("ascii"));
    #endif

    for (int mm=0; mm < I_NCols_Im; mm++)
    {
      D_A1_XProf(mm) = double(mm) + 0.5 + (1. / (2. * static_cast<double>(_fiberTraceProfileFittingControl->overSample)));
    }

    /// fit spline3?
    if (_fiberTraceProfileFittingControl->profileInterpolation.compare(_fiberTraceProfileFittingControl->PROFILE_INTERPOLATION_NAMES[1]) == 0){
      if (!fitSpline(D_A2_Im,
                     I_A1_IFirstPix,
                     XVecArr,
                     SFVecArr,
                     D_A2_XX,
                     D_A1_XProf,
                     profile_Out)){
        string message("FiberTrace");
        message += to_string(_iTrace) + string("::SlitFunc: ERROR: fitSpline returned FALSE");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
    }
    else{/// Piskunov
      do
      {
        I_Iter_Sky++;
        I_Iter_SF = 0;
        while(I_Iter_SF < static_cast<int>(_fiberTraceProfileFittingControl->maxIterSF))   /** Iteration counter **/
        {
          I_Iter_SF++;

          #ifdef __DEBUG_SLITFUNC_SF_N__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": (max(abs(spectrum_Out - SPOldVecArr=" << SPOldVecArr << ") / max(spectrum_Out)=" << max(spectrum_Out) << ")=" << max(fabs(spectrum_Out - SPOldVecArr) / max(spectrum_Out)) << endl;
          #endif

          if ((I_Iter_SF == 1) || (max(fabs(spectrum_Out - SPOldVecArr) / max(spectrum_Out)) > 0.00001))
          {
            AKLArr.resize(I_NPixSlitF, (2*overSample_In) + 1); /** Initialize Matrix **/
            AKLArr = 0.;
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": if(max...): AKLArr initialized to 0.: size(AKLArr) = (" << AKLArr.rows() << "," << AKLArr.cols() << ")" << endl;
            #endif

            BLVecArr.resize(I_NPixSlitF);                      /** and RHS **/
            BLVecArr = 0.;
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": if(max...): BLVecArr = " << BLVecArr << endl;
            #endif

            OmegaVecArr.resize(overSample_In + 1);
            OmegaVecArr = Weight;                    /** Replicate constant Weights **/
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": if(max...): OmegaVecArr = " << OmegaVecArr << endl;
            #endif

            ::pfs::drp::stella::math::Double((*P_I_A2_Mask), D_A2_Mask);
            for (int m = 0; m < I_NRows_Im; m++)  /** Fill up matrix and RHS **/
            {
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Begin for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++)" << endl;
              #endif

              /** Fix the first and the last subpixels, here the weight is split between the two subpixels **/
              OmegaVecArr(0) = D_A2_XX(m,I_A1_IFirstPix(m));

              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OmegaVecArr(0) set to " << OmegaVecArr(0) << endl;
              #endif

              OmegaVecArr(overSample_In) = 1. - D_A2_XX(m, I_A1_ILastPix(m));
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OmegaVecArr(overSample_In=" << overSample_In << ") set to " << OmegaVecArr(overSample_In) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OmegaVecArr set to " << OmegaVecArr << endl;
              #endif
              D_A2_Weights(m, blitz::Range::all()) = OmegaVecArr;

              /** Band-diagonal part that will contain omega#omega **/
              BKLArr.resize(I_NPixSlitF, (2 * overSample_In) + 1);
              BKLArr = 0.;

              blitz::Array<double, 2> *p_OArr = ::pfs::drp::stella::math::VecArrACrossB(OmegaVecArr, OmegaVecArr);
              OArr.resize(p_OArr->rows(),p_OArr->cols());//OmegaVecArr.size(), OmegaVecArr.size());
              OArr = (*p_OArr);
              delete p_OArr;
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): 1. OArr set to " << OArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              #endif

              tmpdbl = OArr(overSample_In, overSample_In);
              tmpdbl += OArr(0, 0);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): tmpdbl set to " << tmpdbl << endl;
              #endif
              OArr(overSample_In, overSample_In) = tmpdbl;
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OArr(overSample_In=" << overSample_In << ", overSample_In) set to " << OArr(overSample_In, overSample_In) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OArr = " << OArr << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OLIndVecArr = " << OLIndVecArr << endl;
              #endif
              //cout << "resizing OOVecArr to size " << OLIndVecArr.size() << endl;
              OOVecArr.resize(OLIndVecArr.size());
              OOVecArr = 0.;
              for (unsigned int n = 0; n < OLIndVecArr.size(); n++)
              {
                tempcol = ::pfs::drp::stella::math::GetColFromIndex(OLIndVecArr(n), OArr.rows());
                temprow = ::pfs::drp::stella::math::GetRowFromIndex(OLIndVecArr(n), OArr.rows());
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for (int n(=" << n << ") = 0; n < OLIndVecArr.size(=" << OLIndVecArr.size() << "); n++): setting OOVecArr(n) to OArr(OLIndVecArr(temprow(=" << temprow << "), tempcol(=" << tempcol << "))=" << OArr(temprow, tempcol) << endl;
                #endif
                #ifdef __DEBUG_CHECK_INDICES__
                  if (temprow >= OArr.rows())
                  {
                    string message("FiberTrace");
                    message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ");
                    message += to_string(I_Bin) + string(": I_Iter_SF = ") + to_string(I_Iter_SF) + string(": ERROR: temprow(=") + to_string(temprow);
                    message += string(") >= OArr.rows(=") + to_string(OArr.rows()) + string(")");
                    cout << message << endl;
                    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                  }
                  if (tempcol >= OArr.cols())
                  {
                    string message("FiberTrace");
                    message += to_string(_iTrace) + string("::SlitFunc: fiberTraceNumber = ") + to_string(fiberTraceNumber) + string(": I_Bin = ");
                    message += to_string(I_Bin) + ": I_Iter_SF = " + to_string(I_Iter_SF) + ": ERROR: tempcol(=" + to_string(tempcol);
                    message += ") >= OArr.cols(=" to_string(OArr.cols()) ")";
                    cout << message << endl;
                    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                  }
                #endif
                OOVecArr(n) = OArr(temprow, tempcol);
              }
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OOVecArr set to " << OOVecArr << endl;
              #endif

              for (int n = 0; n < I_NCols_Im; n++)
              {
                for (unsigned int o = 0; o < BKLIndVecArr.size(); o++)
                {
                  #ifdef __DEBUG_SLITFUNC_N__
                    cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": n(=" << n << ") * overSample_In(=" << overSample_In << ") + I_A1_IFirstPix(m)(=" << I_A1_IFirstPix(m) << ") + BKLIndVecArr(o=" << o << ")=" << BKLIndVecArr(o) << " = " << (n * overSample_In) + I_A1_IFirstPix(m) + BKLIndVecArr(o) << endl;
                    cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": BKLIndVecArr = " << BKLIndVecArr << endl;
                    cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": OOVecArr = " << OOVecArr << endl;
                  #endif

                  tempcol = ::pfs::drp::stella::math::GetColFromIndex((n * overSample_In) + I_A1_IFirstPix(m) + BKLIndVecArr(o), BKLArr.rows());
                  temprow = ::pfs::drp::stella::math::GetRowFromIndex((n * overSample_In) + I_A1_IFirstPix(m) + BKLIndVecArr(o), BKLArr.rows());

                  #ifdef __DEBUG_CHECK_INDICES__
                    if (temprow >= BKLArr.rows())
                    {
                      string message("FiberTrace");
                      message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                      message += ": I_Iter_SF = " + to_string(I_Iter_SF) + ": ERROR: temprow(=" + to_string(temprow) ") >= BKLArr.rows(=" + to_string(BKLArr.rows());
                      message += ")";
                      cout << message << endl;
                      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                    }

                    if (tempcol >= BKLArr.cols())
                    {
                      string message("FiberTrace");
                      message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                      message += ": I_Iter_SF = " + to_string(I_Iter_SF) + ": ERROR: tempcol(=" + to_string(tempcol) + ") >= BKLArr.cols(=";
                      message += to_string(BKLArr.cols()) + ")";
                      cout << message << endl;
                      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                    }

                    if (o >= OOVecArr.size())
                    {
                      string message("FiberTrace");
                      message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                      message += ": I_Iter_SF = " + to_string(I_Iter_SF) + ": ERROR: o(=" + to_string(o) + ") >= OOVecArr.size(=" + to_string(OOVecArr.size());
                      message += ")";
                      cout << message << endl;
                      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                    }
                    if (m >= P_I_A2_Mask->rows())
                    {
                      string message("FiberTrace");
                      message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                      message += ": I_Iter_SF = " + to_string(I_Iter_SF) + ": ERROR: m(=" + to_string(m) + ") >= P_I_A2_Mask->rows(=";
                      message += to_string(P_I_A2_Mask->rows()) + "), P_I_A2_Mask->cols(=" + to_string(P_I_A2_Mask->cols()) + "), I_NRows_Im = ";
                      message += to_string(I_NRows_Im);
                      cout << message << endl;
                      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                    }
                    if (n >= P_I_A2_Mask->cols())
                    {
                      string message("FiberTrace");
                      message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                      message += ": I_Iter_SF = " + to_string(I_Iter_SF) + ": ERROR: n(=" + to_string(n) + ") >= P_I_A2_Mask->cols(=" + to_string(P_I_A2_Mask->cols());
                      message += "), (*P_I_A2_Mask).rows(=" + to_string(P_I_A2_Mask->rows()) + "), P_I_A2_Mask->size() = " + to_string(P_I_A2_Mask->size());
                      message += ", I_NCols_Im = " + to_string(I_NCols_Im);
                      cout << message << endl;
                      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                    }
                  #endif
                  BKLArr(temprow, tempcol) = OOVecArr(o) * D_A2_Mask(m,n);

                  #ifdef __DEBUG_SLITFUNC_N__
                    cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=0; n<NCols(=" << I_NCols_Im << "); n++): for(o(=" << o << ")=0; o<BKLIndVecArr(=" << BKLIndVecArr.size() << "); o++): BKLArr(((n * overSample_In) + I_A1_IFirstPix(m) + BKLIndVecArr(o) = " << (n * overSample_In) + I_A1_IFirstPix(m) + BKLIndVecArr(o) << ")(= temprow=" << temprow << ", tempcol=" << tempcol << ")) set to " << BKLArr(temprow, tempcol) << endl;
                  #endif
                }/// end for (int o = 0; o < BKLIndVecArr.size(); o++)
              }/// end for (int n = 0; n < I_NCols_Im; n++)
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): BKLArr set to " << BKLArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OOVecArr set to " << OOVecArr << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): BKLArr set to " << BKLArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              #endif

  //            OOVecArr.resize(1);
              double OOVal = OArr(overSample_In, overSample_In);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OOVal set to " << OOVal << endl;
              #endif

              for (int n = 1; n < I_NCols_Im; n++)
              {
                #ifdef __DEBUG_CHECK_INDICES__
                  if ((n*overSample_In) + I_A1_IFirstPix(m) >= BKLArr.rows())
                  {
                    string message("FiberTrace");
                    message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                    message += ": I_Iter_SF = " + to_string(I_Iter_SF) + ": ERROR: (n*overSample_In) + I_A1_IFirstPix(m) = ";
                    message += to_string((n*overSample_In) + I_A1_IFirstPix(m)) + " >= BKLArr.rows(=" + to_string(BKLArr.rows()) + ")";
                    cout << message << endl;
                    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                  }
                #endif
                BKLArr((n * overSample_In) + I_A1_IFirstPix(m), overSample_In) = OOVal * D_A2_Mask(m, n);
              }/// end for (int n = 1; n < I_NCols_Im; n++)
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): BKLArr set to " << BKLArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              #endif

              #ifdef __DEBUG_CHECK_INDICES__
                if (I_NCols_Im * overSample_In + I_A1_IFirstPix(m) >= BKLArr.rows())
                {
                  string message("FiberTrace");
                  message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                  message += ": I_Iter_SF = " + to_string(I_Iter_SF) + ": ERROR: I_NCols_Im(=" + to_string(I_NCols_Im) + ") * overSample_In(=";
                  message += to_string(overSample_In) + ") + I_A1_IFirstPix(m)(=" + to_string(I_A1_IFirstPix(m)) + ") = ";
                  message += to_string(I_NCols_Im * overSample_In + I_A1_IFirstPix(m)) + " >= BKLArr.rows(=" + to_string(BKLArr.rows()) + ");
                  cout << message << endl;
                  throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                }
              #endif
              BKLArr((I_NCols_Im * overSample_In) + I_A1_IFirstPix(m), overSample_In) = pow(OmegaVecArr(overSample_In),2) * D_A2_Mask(m, I_NCols_Im - 1);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): BKLArr((I_NCols_Im * overSample_In) + I_A1_IFirstPix(m)=" << (I_NCols_Im * overSample_In) + I_A1_IFirstPix(m) << ", overSample_In=" << overSample_In << ") set to " << BKLArr((I_NCols_Im * overSample_In) + I_A1_IFirstPix(m), overSample_In) << endl;
              #endif
              for (int o = 0; o < overSample_In; o++)
              {
                #ifdef __DEBUG_CHECK_INDICES__
                  if (I_NPixSlitF-1 >= BKLArr.rows())
                  {
                    string message("FiberTrace");
                    message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                    message += ": I_Iter_SF = " + to_string(I_Iter_SF) + ": ERROR: N-1 = " + to_string(I_NPixSlitF-1) + " >= BKLArr.rows(=";
                    message += to_string(BKLArr.rows()) + ")";
                    cout << message << endl;
                    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                  }
                  if (o >= BKLArr.cols())
                  {
                    string message("FiberTrace");
                    message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                    message += ": I_Iter_SF = " + to_string(I_Iter_SF) + ": ERROR: o = " + to_string(o) + " >= BKLArr.cols(=" + to_string(BKLArr.cols());
                    message += ")";
                    cout << message << endl;
                    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                  }
                  if (2*overSample_In-o >= BKLArr.cols())
                  {
                    string message("FiberTrace");
                    message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                    message += ": I_Iter_SF = " + to_string(I_Iter_SF) + ": ERROR: 2*overSample_In-o = " + to_string(2*overSample_In-o) + " >= BKLArr.cols(=";
                    message += to_string(BKLArr.cols()) + ")";
                    cout << message << endl;
                    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                  }
                #endif
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(o(=" << o << ")=0; o<overSample_In(=" << overSample_In << "); o++): BKLARR(blitz::Range = (overSample_In - o = " << overSample_In - o << ", " << endl;
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(o(=" << o << ")=0; o<overSample_In(=" << overSample_In << "); o++): BKLArr(blitz::Range(overSample_In(=" << overSample_In << ")-o(=" << o << ") = " << overSample_In - o << "), I_NPixSlitF(=" << I_NPixSlitF << " - 1 = " << I_NPixSlitF - 1 << "), o = " << o << ") = " << BKLArr(blitz::Range(overSample_In-o, I_NPixSlitF - 1), o) << endl;
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(o(=" << o << ")=0; o<overSample_In(=" << overSample_In << "); o++): BKLARR(blitz::Range = (0, I_NPixSlitF(=" << I_NPixSlitF - 1 << " - 1 - overSample_In + o) = " << I_NPixSlitF - 1 - overSample_In + o << "), 2 * overSample_In - o) = " << 2*overSample_In - o << ") = " << BKLArr(blitz::Range(0, I_NPixSlitF - 1 - overSample_In - o), 2 * overSample_In - o) << endl;
                #endif

                BKLArr(blitz::Range(overSample_In-o, I_NPixSlitF - 1), o) = BKLArr(blitz::Range(0, I_NPixSlitF - 1 - overSample_In + o), 2 * overSample_In - o);
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(o(=" << o << ")=0; o<overSample_In(=" << overSample_In << "); o++): BKLArr(blitz::Range(overSample_In-o=" << overSample_In-o << ", I_NPixSlitF-1=" << I_NPixSlitF-1 << "),o=" << o << ") set to " << BKLArr(blitz::Range(overSample_In-o, I_NPixSlitF - 1), o) << endl;
                #endif
              }/// end for (int o = 0; o < overSample_In; o++)
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): BKLArr set to " << BKLArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): AKLArr = " << AKLArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
                double D_SPVecPow = pow(spectrum_Out(m), 2);
                D_A2_SPVecTimesBKLArr.resize(BKLArr.rows(), BKLArr.cols());
                D_A2_SPVecTimesBKLArr = 0.;
                D_A2_SPVecTimesBKLArr = D_SPVecPow * BKLArr;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): D_SPVecPow(= " << D_SPVecPow << ") * BKLArr(= " << BKLArr << ") = " << D_A2_SPVecTimesBKLArr << endl;
              #endif
              #ifdef __DEBUG_CHECK_INDICES__
                if (AKLArr.size() != BKLArr.size())
                {
                  string message("FiberTrace");
                  message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                  message += ": I_Iter_SF = " + to_string(I_Iter_SF) + ": ERROR: size of AKLArr(=" + to_string(AKLArr.size()) + ") != size of BKLArr(=";
                  message += to_string(BKLArr.size()) + ")";
                  cout << message << endl;
                  throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                }
              #endif
              AKLArr += (pow(spectrum_Out(m), 2) * BKLArr);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): AKLArr(+= (pow(spectrum_Out(m), 2) * BKLArr)) set to " << AKLArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              #endif
              OArr.resize(I_NPixSlitF, 1);
              OArr = 0.;
              for (int n = 0; n < I_NCols_Im; n++)
              {
                OArr(blitz::Range((n * overSample_In) + I_A1_IFirstPix(m), ((n+1) * overSample_In) + I_A1_IFirstPix(m)), 0) = D_A2_Im(m, n) * Weight * D_A2_Mask(m, n);
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OArr(blitz::Range(n * overSample_In) + I_A1_IFirstPix(m)=" << (n * overSample_In) + I_A1_IFirstPix(m) << ", n * overSample_In + I_A1_IFirstPix(m) + overSample_In=" << n * overSample_In + I_A1_IFirstPix(m) + overSample_In << "), 0) set to " << OArr(blitz::Range((n * overSample_In) + I_A1_IFirstPix(m), n * overSample_In + I_A1_IFirstPix(m) + overSample_In), 0) << endl;
                #endif
              }
              #ifdef __DEBUG_SLITFUNC_X__
                //cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): 2. OArr set to " << OArr << endl;
                //cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): D_A2_Im(m,*) = " << D_A2_Im(m, blitz::Range::all()) << endl;
                std::string fName = DEBUGDIR + std::string("OArr") + debugFilesSuffix + std::string("_row");
                if (m < 100)
                  fName += std::string("0");
                if (m < 10)
                  fName += std::string("0");
                fName += to_string(m) + std::string(".dat");
                ::pfs::drp::stella::utils::WriteArrayToFile(OArr, fName, std::string("ascii"));
              #endif
              #ifdef __DEBUG_SLITFUNC_FILES__
                string sOArr = DEBUGDIR + std::string("OArrBySF") + debugFilesSuffix + std::string("_row");
                if (m < 100)
                  sOARR += std::string("0");
                if (m < 10)
                  sOARR += std::string("0");
                sOARR += to_string(m) + std::string(".dat");
                blitz::Array<double, 1> D_A1_OArrBySF(OArr.rows());
                D_A1_OArrBySF = OArr(blitz::Range::all(),0) * double(overSample_In) / blitz::sum(D_A2_Im(m, blitz::Range::all()));
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Writing file " << sOArr << endl;
                ::pfs::drp::stella::utils::WriteArrayToFile(D_A1_OArrBySF, sOArr, string("ascii"));

                sOArr = DEBUGDIR + std::string("D_A2_Im")+debugFilesSuffix+std::string("_row");
                if (m < 100)
                  sOARR += std::string("0");
                if (m < 10)
                  sOARR += std::string("0");
                sOARR += to_string(m) + std::string(".dat");
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Writing file " << sOArr << endl;
                ::pfs::drp::stella::utils::WriteArrayToFile(D_A2_Im(m,blitz::Range::all()), sOArr, string("ascii"));
              #endif
              #ifdef __DEBUG_CHECK_INDICES__
                if (P_I_A2_Mask->rows() != D_A2_Mask.rows()){
                  string message("FiberTrace");
                  message += to_string(_iTrace) + "::SlitFunc: ERROR: 2. P_I_A2_Mask->rows() != D_A2_Mask.rows()";
                  cout << message << endl;
                  throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                }
              #endif
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): 3. OArr set to " << OArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): *P_I_A2_Mask = " << *P_I_A2_Mask << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): D_A2_Mask = " << D_A2_Mask << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              #endif

              for (int n = 1; n < I_NCols_Im; n++)
              {
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=1; n<NCols(=" << I_NCols_Im << "); n++): (D_A2_Im(m,n-1)=" << D_A2_Im(m, n-1) << ")" << endl;
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=1; n<NCols(=" << I_NCols_Im << "); n++): ((*P_I_A2_Mask)(m,n-1)=" << (*P_I_A2_Mask)(m, n-1) << ")" << endl;
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=1; n<NCols(=" << I_NCols_Im << "); n++): OmegaVecArr(overSample_In)=" << OmegaVecArr(overSample_In) << ")" << endl;
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=1; n<NCols(=" << I_NCols_Im << "); n++): (D_A2_Im(m,n)=" << D_A2_Im(m, n) << ")" << endl;
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=1; n<NCols(=" << I_NCols_Im << "); n++): ((*P_I_A2_Mask)(m,n)=" << (*P_I_A2_Mask)(m, n) << ")" << endl;
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=1; n<NCols(=" << I_NCols_Im << "); n++): (OmegaVecArr(0)=" << OmegaVecArr(0) << ")" << endl;
                #endif
                OArr((n * overSample_In) + I_A1_IFirstPix(m), 0) = (D_A2_Im(m, n-1) * OmegaVecArr(overSample_In) * D_A2_Mask(m, n-1)) + (D_A2_Im(m, n) * OmegaVecArr(0) * D_A2_Mask(m, n));
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=1; n<NCols(=" << I_NCols_Im << "); n++): OArr((n * overSample_In) + I_A1_IFirstPix(m)=" << (n * overSample_In) + I_A1_IFirstPix(m) << ", 0) set to " << OArr((n * overSample_In) + I_A1_IFirstPix(m), 0) << endl;
                #endif
              }/// end for (int n = 1; n < I_NCols_Im; n++)
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): 4. OArr set to " << OArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              #endif
              OArr(I_A1_IFirstPix(m), 0) = D_A2_Im(m, 0) * OmegaVecArr(0) * D_A2_Mask(m, 0);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): D_A2_Im(m, I_NCols_Im-1) = " << D_A2_Im(m, I_NCols_Im-1) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OmegaVecArr(overSample_In) = " << OmegaVecArr(overSample_In) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): (*P_I_A2_Mask)(m, I_NCols_Im-1) = " << (*P_I_A2_Mask)(m, I_NCols_Im-1) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): D_A2_Mask(m, I_NCols_Im-1) = " << D_A2_Mask(m, I_NCols_Im-1) << endl;
              #endif
              OArr((I_NCols_Im * overSample_In) + I_A1_IFirstPix(m), 0) = D_A2_Im(m, I_NCols_Im - 1) * OmegaVecArr(overSample_In) * D_A2_Mask(m, I_NCols_Im - 1);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OArr((D_A2_Im.cols(=" << I_NCols_Im << ") * overSample_In=" << overSample_In << ") + I_A1_IFirstPix(m)=" << I_A1_IFirstPix(m) << ", 0) set to " << OArr((I_NCols_Im * overSample_In) + I_A1_IFirstPix(m), 0) << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): 5. OArr set to " << OArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): spectrum_Out(m) = " << spectrum_Out(m) << endl;
              #endif
              BLVecArr(blitz::Range::all()) += ((spectrum_Out(m) * OArr(blitz::Range::all(), 0)));
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": end for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): BLVecArr set to " << BLVecArr << endl;
              #endif
              #ifdef __DEBUG_SLITFUNC_X__
                fName = DEBUGDIR + std::string("OArr_new") + debugFilesSuffix + "_row";
                if (m < 100)
                  fName += "0";
                if (m < 10)
                  fName += "0";
                fName += to_string(m) + ".dat";
                ::pfs::drp::stella::utils::WriteArrayToFile(OArr, fName, std::string("ascii"));
              #endif
            } /** end for (int m = 0; m < I_NRows_Im; m++) **/

            #ifdef __DEBUG_SLITFUNC_N__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": end for(m=0; m<NRows; m++): OArr set to " << OArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": end for(m=0; m<NRows; m++): BLVecArr set to " << BLVecArr << endl;
            #endif

            Lambda = Lamb_SF * blitz::sum(AKLArr(blitz::Range::all(), overSample_In)) / I_NPixSlitF;
            blitz::Array<double, 1> D_A1_Lamb_SF(I_NPixSlitF);
            if (D_WingSmoothFactor > 0.){
              if (I_Iter_SF == 1){
                blitz::Array<double, 1> D_A1_DIndGen = ::pfs::drp::stella::math::DIndGenArr(I_NPixSlitF);
                D_A1_Lamb_SF = Lambda * (1. + D_WingSmoothFactor * blitz::pow2(2. * D_A1_DIndGen / (I_NPixSlitF - 1) - 1.));
              }
              else{
                #ifdef __DEBUG_CHECK_INDICES__
                  if (static_cast<int>(SFVecArr.size()) != I_NPixSlitF){
                    string message("FiberTrace");
                    message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                    message += ": KeyWord_Set(WING_SMOOTH_FACTOR): ERROR: SFVecArr.size(=" + to_string(SFVecArr.size()) + ") != I_NPixSlitF=";
                    message += to_string(I_NPixSlitF);
                    cout << message << endl;
                    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                  }
                #endif
                SFVecArrTemp.resize(SFVecArr.size());
                SFVecArrTemp = SFVecArr;
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(WING_SMOOTH_FACTOR): 1. SFVecArrTemp set to " << SFVecArrTemp << endl;
                #endif
                for (unsigned int m=0; m<SFVecArrTemp.size(); m++){
                  if (SFVecArrTemp(m) < 0.00001)
                    SFVecArrTemp(m) = 0.00001;
                }
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(WING_SMOOTH_FACTOR): 2. SFVecArrTemp set to " << SFVecArrTemp << endl;
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(WING_SMOOTH_FACTOR): Lambda = " << Lambda << endl;
                #endif
                D_A1_Lamb_SF = Lambda * (1. + D_WingSmoothFactor / (SFVecArrTemp));
              }
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(WING_SMOOTH_FACTOR): D_A1_Lamb_SF set to " << D_A1_Lamb_SF << endl;
              #endif
            }/// end if (Pos >= 0 && D_WingSmoothFactor > 0.){
            else{
              D_A1_Lamb_SF = ::pfs::drp::stella::math::Replicate(Lambda, I_NPixSlitF);
              #ifdef __DEBUG_SLITFUNC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(WING_SMOOTH_FACTOR): D_A1_Lamb_SF set to " << D_A1_Lamb_SF << endl;
              #endif
            }
              /**
              *        1st order Tikhonov regularization (minimum 1st derivatives)
              *        Add the following 3-diagonal matrix * lambda:
              *          1 -1  0  0  0  0
              *        -1  2 -1  0  0  0
              *          0 -1  2 -1  0  0
              *          0  0 -1  2 -1  0
              *              .  .  .
              **/
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Start 1st order Tikhonov: AKLArr = " << AKLArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Start 1st order Tikhonov: D_A1_Lamb_SF = " << D_A1_Lamb_SF << endl;
            #endif

            AKLArr(0, overSample_In) += D_A1_Lamb_SF(0); /** + Lambda to the upper-left element **/
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 1st order Tikhonov: AKLArr(0,overSample_In=" << overSample_In << ") set to " << AKLArr(0,overSample_In) << endl;
            #endif

            AKLArr(I_NPixSlitF-1,overSample_In) += D_A1_Lamb_SF(I_NPixSlitF - 1); /** and to the lower-right **/
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 1st order Tikhonov: AKLArr(I_NPixSlitF-1=" << I_NPixSlitF-1 << ",overSample_In=" << overSample_In << ") set to " << AKLArr(I_NPixSlitF-1,overSample_In) << endl;
            #endif

            AKLArr(blitz::Range(1,I_NPixSlitF-2), overSample_In) += 2. * D_A1_Lamb_SF(blitz::Range(1, I_NPixSlitF-2)); /** +2*Lambda to the rest of the main diagonal **/
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 1st order Tikhonov: AKLArr(blitz::Range(1,I_NPixSlitF-2=" << I_NPixSlitF-2 << "),overSample_In=" << overSample_In << ") set to " << AKLArr(blitz::Range(1,I_NPixSlitF-2),overSample_In) << endl;
            #endif

            AKLArr(blitz::Range(0, I_NPixSlitF - 2), overSample_In + 1) -= D_A1_Lamb_SF(blitz::Range(0, I_NPixSlitF - 2)); /** -Lambda to the upper sub-diagonal **/
            #ifdef __DEBUG_SLITFUNC__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 1st order Tikhonov: AKLArr(blitz::Range(0,I_NPixSlitF-2=" << I_NPixSlitF-2 << "),overSample_In+1=" << overSample_In+1 << ") set to " << AKLArr(blitz::Range(0,I_NCols_Im-2),overSample_In+1) << endl;
            #endif

            AKLArr(blitz::Range(1, I_NPixSlitF - 1), overSample_In - 1) -= D_A1_Lamb_SF(blitz::Range(1, I_NPixSlitF - 1)); /** -Lambda to the lower sub-diagonal **/
            #ifdef __DEBUG_SLITFUNC__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 1st order Tikhonov: AKLArr(blitz::Range(1,I_NPixSlitF-1=" << I_NPixSlitF-1 << "),overSample_In-1=" << overSample_In-1 << ") set to " << AKLArr(blitz::Range(1,I_NPixSlitF-1),overSample_In-1) << endl;
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Start. 1st order Tikhonov: AKLArr = " << AKLArr << endl;
            #endif

            /**
            *        2nd order Tikhonov regularization (minimum 2nd derivative)
            *        Add the following 5-diagonal matrix * lambda:
            *          1 -2  1  0  0  0
            *         -2  5 -4  1  0  0
            *          1 -4  6 -4  1  0
            *          0  1 -4  6 -4  1
            *              .  .  .
            **/
            void **PP_Void = (void**)malloc(sizeof(void*) * 4);
            D_A2_AKLT.resize(AKLArr.cols(), AKLArr.rows());
            D_A2_AKLT = AKLArr.transpose(blitz::secondDim, blitz::firstDim);
            #ifdef __DEBUG_SLITFUNC__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Start. 2nd order Tikhonov: AKLArr = " << AKLArr << endl;
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Start. 2nd order Tikhonov: D_A2_AKLT = " << D_A2_AKLT << endl;
            #endif
            PP_Void[0] = D_A2_AKLT.data();
            PP_Void[1] = BLVecArr.data();
            #ifdef __DEBUG_SLITFUNC__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Start. 2nd order Tikhonov: BLVecArr = " << BLVecArr << endl;
            #endif
            PP_Void[2] = &I_NPixSlitF;
            #ifdef __DEBUG_SLITFUNC__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Start. 2nd order Tikhonov: I_NPixSlitF = " << I_NPixSlitF << endl;
            #endif
            TempInt = (2 * overSample_In) + 1;
            PP_Void[3] = &TempInt;
            #ifdef __DEBUG_SLITFUNC__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 2nd order Tikhonov: TempInt = " << TempInt << endl;
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 2nd order Tikhonov: Starting BandSol(D_A2_AKLT=" << D_A2_AKLT << ", BLVecArr=" << BLVecArr << ", I_NPixSlitF=" << I_NPixSlitF << ", TempInt=" << TempInt << ")" << endl;
            #endif
            ::pfs::drp::stella::math::BandSol(4, PP_Void);
            free(PP_Void);
            AKLArr = D_A2_AKLT.transpose(blitz::secondDim, blitz::firstDim);
            #ifdef __DEBUG_SLITFUNC__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 2nd order Tikhonov: After BandSol: D_A2_AKLT=" << D_A2_AKLT << ", AKLArr=" << AKLArr << ", BLVecArr=" << BLVecArr << ", I_NPixSlitF=" << I_NPixSlitF << ", TempInt=" << TempInt << endl;
            #endif
            SFVecArr.resize(BLVecArr.size());
            if (abs(blitz::sum(BLVecArr)) < 0.000000001)
            {
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": WARNING: blitz::sum(BLVecArr) == 0 => Setting to 1." << endl;
              BLVecArr = 1.;
            }
            double D_SumBLVecArr = blitz::sum(BLVecArr);
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum(BLVecArr) = " << D_SumBLVecArr << endl;
            #endif
            if (D_SumBLVecArr == 0.){
              string message("FiberTrace");
              message += to_string(_iTrace);
              message += "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
              message += ": ERROR: blitz::sum(BLVecArr) == 0";
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
            }
            SFVecArr = BLVecArr;
            for (unsigned int mmm=0; mmm < SFVecArr.size(); mmm++){
              if (SFVecArr(mmm) < 0.)
                SFVecArr(mmm) = 0.;
            }
            SFVecArr = SFVecArr / D_SumBLVecArr * overSample_In;

            #ifdef __DEBUG_SLITFUNC_N__
              std::string sOArr = std::string(DEBUGDIR) + std::string("SFVecArr") + debugFilesSuffix + std::string(".dat");
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Writing File " << SFVecArr << endl;
              ::pfs::drp::stella::utils::WriteArrayToFile(SFVecArr, sOArr, string("ascii"));

              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 4.     2nd order Tikhonov: SFVecArr ( = BLVecArr / blitz::sum(BLVecArr)(=" << blitz::sum(BLVecArr) << ") * overSample_In(=" << overSample_In << ")) = " << SFVecArr << endl;
            #endif

            #ifdef __DEBUG_SLITFUNC__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 2nd order Tikhonov: spectrum_Out = " << spectrum_Out << endl;
            #endif
            SPOldVecArr.resize(spectrum_Out.size());
            SPOldVecArr = spectrum_Out;
            #ifdef __DEBUG_SLITFUNC__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 2nd order Tikhonov: SPOldVecArr = " << SPOldVecArr << endl;
            #endif

            RVecArr.resize(spectrum_Out.size());
            RVecArr_Err.resize(spectrum_Out.size());
            RVecArr = spectrum_Out;
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 2nd order Tikhonov: RVecArr = " << RVecArr << endl;
            #endif

            OmegaVecArr.resize(overSample_In);
            OmegaVecArr = Weight;
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 2nd order Tikhonov: OmegaVecArr = " << OmegaVecArr << endl;
              cout << "FiberTrace" << _iTrace << "::SlitFunc: Before TempDVecArrA: *P_I_A2_MaskIn = " << *P_I_A2_MaskIn << endl;
            #endif

            /** Evaluate the new Spectrum **/
            for (int m = 0; m < I_NRows_Im; m++)
            {
              OmegaVecArr(0) = D_A2_XX(m,I_A1_IFirstSpec(m));
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): OmegaVecArr(0) set to " << OmegaVecArr(0) << endl;
              #endif

              TempDVecArr.resize(I_A1_ILastSpec(m) - I_A1_IFirstSpec(m) + 1);
              TempDVecArr = SFVecArr(blitz::Range(I_A1_IFirstSpec(m), I_A1_ILastSpec(m)));
              #ifdef __DEBUG_SLITFUNC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): 1. TempDVecArr set to " << TempDVecArr << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Starting Reform: I_NCols_Im = " << I_NCols_Im << ", overSample_In = " << overSample_In << endl;
              #endif
              blitz::Array<double, 2> *p_SSFTArr = ::pfs::drp::stella::math::Reform(TempDVecArr, (int)I_NCols_Im, overSample_In);
              #ifdef __DEBUG_SLITFUNC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": fiberTraceNumber = " << fiberTraceNumber << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Reform finished: p_SSFTArr set to " << *p_SSFTArr << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_NCols_Im = " << I_NCols_Im << endl;
              #endif
              SSFArr.resize(overSample_In, (int)I_NCols_Im);
              SSFArr = p_SSFTArr->transpose(blitz::secondDim, blitz::firstDim);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": SSFArr set to " << SSFArr << endl;
              #endif
              delete p_SSFTArr;
              TempDVecArr.resize(0);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): SSFArr set to " << SSFArr << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
              #endif

              tempint = SSFArr.cols();

              D_A2_TempAA.resize(SSFArr.cols(), SSFArr.rows());
              D_A2_TempAA = SSFArr.transpose(blitz::secondDim,blitz::firstDim);

              blitz::Array<double, 1> *p_TempDVecArrBB = ::pfs::drp::stella::math::MatrixTimesVecArr(D_A2_TempAA,
                                                                                              OmegaVecArr);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): 2. TempDVecArr set to " << *p_TempDVecArrBB << endl;
              #endif
              OArr.resize(tempint, 1);
              OArr(blitz::Range(0,tempint-1), 0) = (*p_TempDVecArrBB)(blitz::Range(0,tempint-1));
              delete p_TempDVecArrBB;
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): OArr(blitz::Range(0,tempint-1), 0) set to " << OArr(blitz::Range(0,tempint-1), 0) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): D_A2_Im.cols(=" << I_NCols_Im << ") - D_A2_XX(" << m << ",I_A1_ILastSpec(m)=" << I_A1_ILastSpec(m) << ")(=" << D_A2_XX(m,I_A1_ILastSpec(m)) << ") = " << I_NCols_Im - D_A2_XX(m,I_A1_ILastSpec(m)) << endl;
              #endif
              XXX = I_NCols_Im - D_A2_XX(m,I_A1_ILastSpec(m));
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): XXX set to " << XXX << endl;
              #endif

              TempDVecArr.resize(I_NCols_Im-1);
              TempDVecArr = SSFArr(0, blitz::Range(1, I_NCols_Im-1));
              TempDVecArr *= XXX;
              OArr(blitz::Range(0, I_NCols_Im - 2), 0) += TempDVecArr;
              OArr(I_NCols_Im - 1, 0) += SFVecArr(I_A1_ILastSpec(m) + 1) * XXX;
              #ifdef __DEBUG_SLITFUNC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): OArr(I_NCols_Im - 1, 0) set to " << OArr(I_NCols_Im - 1, 0) << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): D_A2_Im(m, all()) = " << D_A2_Im(m, blitz::Range::all()) << endl;
              #endif

              D_A1_TempDVecArr.resize(I_NCols_Im);
              D_A1_TempDVecArr = D_A2_Im(m, blitz::Range::all());
              D_A1_TempDVecArr *= D_A2_Mask(m, blitz::Range::all());
              #ifdef __DEBUG_SLITFUNC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A1_TempDVecArr: D_A2_Im(m=" << m << ", blitz::Range::all()) = " << D_A2_Im(m, blitz::Range::all()) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A1_TempDVecArr: P_I_A2_Mask(m=" << m << ", blitz::Range::all()) = " << (*P_I_A2_Mask)(m, blitz::Range::all()) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A1_TempDVecArr: D_A2_Mask(m=" << m << ", blitz::Range::all()) = " << D_A2_Mask(m, blitz::Range::all()) << endl;
              #endif
              D_A1_TempDVecArr_Err.resize(I_NCols_Im);
              if (ErrorsRead){
                D_A1_TempDVecArr_Err = (*P_D_A2_Errors)(m, blitz::Range::all());
                D_A1_TempDVecArr_Err *= D_A2_Mask(m, blitz::Range::all());
              }
              else{
                D_A1_TempDVecArr_Err = 0.;
              }
              D_A1_TempDVecArrAA.resize(OArr.rows());
              D_A1_TempDVecArrAA = OArr(blitz::Range::all(), 0);
              #ifdef __DEBUG_SLITFUNC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A1_TempDVecArr = " << D_A1_TempDVecArr << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A1_TempDVecArrAA = " << D_A1_TempDVecArrAA << endl;
              #endif
              #ifdef __DEBUG_CHECK_INDICES__
                if (D_A1_TempDVecArr.size() != D_A1_TempDVecArrAA.size())
                {
                  string message("FiberTrace");
                  message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                  message += ": I_Iter_SF = " + to_string(I_Iter_SF) + ": ERROR: D_A1_TempDVecArr.size(=" + to_string(D_A1_TempDVecArr.size());
                  message += ")=(" + to_string(D_A1_TempDVecArr) + ") != D_A1_TempDVecArrAA.size(=" + to_string(D_A1_TempDVecArrAA.size()) + ") = (";
                  message += to_string(D_A1_TempDVecArrAA) + ")";
                  cout << message << endl;
                  throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                }
              #endif

              RVecArr(m) = ::pfs::drp::stella::math::VecArrAScalarB(D_A1_TempDVecArr, D_A1_TempDVecArrAA);
              RVecArr_Err(m) = ::pfs::drp::stella::math::VecArrAScalarB(D_A1_TempDVecArr_Err, D_A1_TempDVecArrAA);
              #ifdef __DEBUG_SLITFUNC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): RVecArr(m) set to " << RVecArr(m) << endl;
              #endif
              TempDVecArr.resize(OArr.rows());
              TempDVecArr = (blitz::pow2(OArr(blitz::Range::all(), 0)));
              TempDVecArr(blitz::Range::all()) *= D_A2_Mask(m, blitz::Range::all());
              spectrum_Out(m) = blitz::sum(TempDVecArr);
              #ifdef __DEBUG_SLITFUNC_SF__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): spectrum_Out(m=" << m << ") set to " << spectrum_Out(m) << endl;
              #endif
              if (fabs(spectrum_Out(m)) < 0.000001)
              {
                spectrum_Out(m) = blitz::sum(blitz::pow2(OArr));
                #ifdef __DEBUG_SLITFUNC_SF__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): was == 0: spectrum_Out(m=" << m << ") set to " << spectrum_Out << endl;
                #endif
              }

              ///Locate and mask outliers
              if (I_Iter_SF > 1)
              {
                TempDVecArr.resize(I_NCols_Im);
                TempDVecArr = 0.;
                TempDVecArrB.resize(OArr.rows());
                TempDVecArrB = 0.;
                TempDVecArrC.resize(OArr.rows());
                TempDVecArrC = 0.;
                TempDVecArrD.resize(OArr.rows());
                TempDVecArrD = 0.;

                TempDVecArr = D_A2_Im(m, blitz::Range::all());
                #ifdef __DEBUG_SLITFUNC__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m==" << m << "): I_Iter_SF(=" << I_Iter_SF << ") > 1: TempDVecArr set to " << TempDVecArr << endl;
                #endif

                TempDVecArrB = OArr(blitz::Range::all(), 0);
                #ifdef __DEBUG_SLITFUNC__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m==" << m << "): I_Iter_SF(=" << I_Iter_SF << ") > 1: TempDVecArrB set to " << TempDVecArrB << endl;
                #endif

                ::pfs::drp::stella::math::Double((*P_I_A2_Mask), D_A2_Mask);
                #ifdef __DEBUG_CHECK_INDICES__
                  if (P_I_A2_Mask->rows() != D_A2_Mask.rows()){
                    string message("FiberTrace");
                    message += to_string(_iTrace) + "::SlitFunc: ERROR: 3. P_I_A2_Mask->rows() != D_A2_Mask.rows()";
                    cout << message << endl;
                    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                  }
                #endif
                if (fabs(mean(*P_I_A2_Mask) - mean(D_A2_Mask)) > 0.0000001){
                  string message("FiberTrace");
                  message += to_string(_iTrace) + "::SlitFunc: ERROR: 3. mean(P_I_A2_Mask)(=" + to_string(mean(*P_I_A2_Mask)) + ") != mean(D_A2_Mask)(=";
                  message += to_string(mean(D_A2_Mask)) + ")";
                  cout << message << endl;
                  throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                }

                #ifdef __PISKUNOV_ORIG__
                  double D_Norm = RVecArr(m) / spectrum_Out(m);
                  blitz::Array<int, 1> I_A1_IndArr(D_A2_Im.cols());
                  I_A1_IndArr = blitz::where(fabs(D_A2_Im(m, blitz::Range::all()) - (D_Norm * TempDVecArrB)) > 6. * D_Dev, 1, 0);
                  for (int i_ind=0; i_ind<D_A2_Im.cols(); i_ind++){
                    if (I_A1_IndArr(i_ind) == 1){
                      (*P_I_A2_Mask)(m, i_ind) = 0;
                    }
                    else{
                      (*P_I_A2_Mask)(m, i_ind) = (*P_I_A2_MaskIn)(m, i_ind);
                    }
                  }
                  D_Dev_New += blitz::sum((*P_I_A2_Mask)(m, blitz::Range::all()) * blitz::pow2(D_A2_Im(m, blitz::Range::all()) - D_Norm * TempDVecArrB));
                #endif// __PISKUNOV_ORIG__
              } /// end if (I_Iter_SF > 1)
            } ///end for (int m = 0; m < I_NRows_Im; m++)

            #ifdef __PISKUNOV_ORIG__
              if (I_Iter_SF > 1)
              D_Dev = sqrt(D_Dev_New / blitz::sum(*P_I_A2_Mask));
            #endif

            #ifdef __DEBUG_SLITFUNC_FILES__
              S_SP = DEBUGDIR + "spectrum_Out1Rows.fits";
              D_A2_SPTemp.resize(spectrum_Out.size(), 1);
              D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
              ::pfs::drp::stella::utils::WriteFits(&D_A2_SPTemp, S_SP);
            #endif
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: After TempDVecArrA: *P_I_A2_Mask = " << *P_I_A2_Mask << endl;
              cout << "blitz::sum(*P_I_A2_MaskIn) = " << blitz::sum(*P_I_A2_MaskIn) << endl;
              cout << "blitz::sum(*P_I_A2_Mask) = " << blitz::sum(*P_I_A2_Mask) << endl;
            #endif

            if (abs(Lamb_SP) > 0.0000001)
            {
              Lambda = Lamb_SP * blitz::sum(spectrum_Out) / I_NRows_Im;
              #ifdef __DEBUG_SLITFUNC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": KeyWord_Set(LAMBDA_SP): Lambda set to " << Lambda << endl;
              #endif
              a.resize(I_NRows_Im);
              a = 0. - Lambda;
              a(0) = 0.;
              #ifdef __DEBUG_SLITFUNC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": KeyWord_Set(LAMBDA_SP): a set to " << a << endl;
              #endif
              b.resize(I_NRows_Im);
              b = (2. * Lambda) + 1.;
              b(0) = Lambda + 1.;
              b(I_NRows_Im - 1) = Lambda + 1.;
              #ifdef __DEBUG_SLITFUNC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": KeyWord_Set(LAMBDA_SP): b set to " << b << endl;
              #endif
              c.resize(I_NRows_Im);
              c = 0. - Lambda;
              c(I_NRows_Im - 1) = 0.;
              #ifdef __DEBUG_SLITFUNC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": KeyWord_Set(LAMBDA_SP): c set to " << c << endl;
              #endif

              #ifdef __DEBUG_CHECK_INDICES__
                if (I_NRows_Im != static_cast<int>(spectrum_Out.size()))
                {
                  string message("FiberTrace");
                  message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                  message += ": I_Iter_SF = " + to_string(I_Iter_SF) + ": ERROR: D_A2_Im.rows(=" + to_string(I_NRows_Im) + ") != spectrum_Out.size(=";
                  message += spectrum_Out.size() + ")";
                  cout << message << endl;
                  throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                }
              #endif
              #ifdef __DEBUG_SLITFUNC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: Before TriDag: RVecArr = " << RVecArr << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: Before TriDag: spectrum_Out = " << spectrum_Out << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: Before TriDag: a = " << a << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: Before TriDag: b = " << b << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: Before TriDag: c = " << c << endl;
              #endif
              TempDVecArr.resize(I_NRows_Im);
              TempDVecArr = RVecArr / spectrum_Out;
              #ifdef __DEBUG_SLITFUNC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: Before TriDag: TempDVecArr = " << TempDVecArr << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": KeyWord_Set(LAMBDA_SP): TempDVecArr set to " << TempDVecArr << endl;
              #endif
              ::pfs::drp::stella::math::TriDag(a, b, c, TempDVecArr, spectrum_Out);
              #ifdef __DEBUG_SLITFUNC_SF__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": KeyWord_Set(LAMBDA_SP): after TriDag: spectrum_Out set to " << spectrum_Out << endl;
              #endif

              #ifdef __DEBUG_SLITFUNC__
                S_SP = DEBUGDIR + std::string("spectrum_Out1TriDag_IterSF") + to_string(I_Iter_SF) + debugFilesSuffix + std::string(".fits");
                D_A2_SPTemp.resize(spectrum_Out.size(), 1);
                D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
                ::pfs::drp::stella::utils::WriteFits(&D_A2_SPTemp, S_SP);
              #endif
            }
            else{
              spectrum_Out = RVecArr / spectrum_Out;
              #ifdef __DEBUG_SLITFUNC_SF__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Not KeyWord_Set(LAMBDA_SP): spectrum_Out set to " << spectrum_Out << endl;
              #endif

              #ifdef __DEBUG_SLITFUNC__
                S_SP = "spectrum_Out1NoLambSP_IterSF" + to_string(I_Iter_SF) + debugFilesSuffix + ".fits";
                D_A2_SPTemp.resize(spectrum_Out.size(), 1);
                D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
                ::pfs::drp::stella::utils::WriteFits(&D_A2_SPTemp, S_SP);
              #endif
            } /// end else if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(const_cast<const CString**>(Args), NArgs, *P_TempString)) < 0)

            #ifdef __DEBUG_SLITFUNC_SF__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": end if (max(fabs(spectrum_Out - SPOldVecArr) / max(spectrum_Out)=" << max(spectrum_Out) << ")=" << max(fabs(spectrum_Out - SPOldVecArr) / max(spectrum_Out)) << " > 0.00001)" << endl;
            #endif

            #ifdef __DEBUG_SLITFUNC_X__
              tempFileName = DEBUGDIR + std::string("cfits_sp_");
              if (I_Iter_SF < 10)
                tempFileName += std::string("0");
              tempFileName += to_string(I_Iter_SF) + debugFilesSuffix + std::string(".dat");
              ofstream *P_SP_Log = new ofstream(tempFileName.c_str());
              for (int isp=0; isp < spectrum_Out.rows(); isp++){
                (*P_SP_Log) << spectrum_Out(isp) << endl;
              }
              delete(P_SP_Log);

              tempFileName = DEBUGDIR + std::string("cfits_sf_");
              if (I_Iter_SF < 10)
                tempFileName += std::string("0");
              tempFileName += to_string(I_Iter_SF) + debugFilesSuffix + std::string(".dat");
              ofstream *P_SF_Log = new ofstream(tempFileName.c_str());
              for (int isf=0; isf < SFVecArr.rows(); isf++){
                (*P_SF_Log) << SFVecArr(isf) << endl;
              }
              delete(P_SF_Log);
            #endif
          } /// end if (I_Iter_SF == 1 || max(abs(spectrum_Out-SPOldVecArr)/max(spectrum_Out)) > 0.00001)
          else
          {
            if (I_Iter_SF != 1)
            {
              #ifdef __DEBUG_SLITFUNC_SF__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": !if (max(fabs(spectrum_Out - SPOldVecArr) / max(spectrum_Out)=" << max(spectrum_Out) << ")=" << max(fabs(spectrum_Out - SPOldVecArr) / max(spectrum_Out)) << " > 0.00001) => breaking while loop" << endl;
              #endif
              break;
            }/// end if (I_Iter_SF != 1)
          }/// end else if (I_Iter_SF != 1 && max(abs(spectrum_Out-SPOldVecArr)/max(spectrum_Out)) <= 0.00001)
          blitz::Array<double, 2> D_A2_ImTimesMask_SF(D_A2_Im.rows(), D_A2_Im.cols());
          D_A2_ImTimesMask_SF = D_A2_Im * (*P_I_A2_Mask);
          #ifdef __DEBUG_SLITFUNC_FILES__
            string S_ImTimesMaskSF = DEBUGDIR + std::string("ImTimesMask_IterSF") + to_string(I_Iter_SF) + debugFilesSuffix + std::string(".fits");
            ::pfs::drp::stella::utils::WriteFits(&D_A2_ImTimesMask_SF, S_ImTimesMaskSF);
          #endif
        } /// end while(I_Iter_SF < I_MaxIterSF)
        if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "IM_OUT")) >= 0)
        {
          if (P_D_A2_Im_Out != NULL)
            delete P_D_A2_Im_Out;
          P_D_A2_Im_Out = (blitz::Array<double, 2>*)ArgV_In[Pos];
          P_D_A2_Im_Out->resize(D_A2_Im.rows(), D_A2_Im.cols());
          (*P_D_A2_Im_Out) = 0.;
        }
        #ifdef __DEBUG_SLITFUNC__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): I_NRows_Im set to " << I_NRows_Im << endl;
        #endif
        if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "USE_ROW")) >= 0)// && TempIntB != 0)
        {
          D_A1_Ind.resize(I_NRows_Im);
          D_A1_Ind = i;
          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): KeyWord_Set(USE_ROW): MARK: INTERPOL: spectrum_Out = " << spectrum_Out << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): KeyWord_Set(USE_ROW): MARK: INTERPOL: D_A1_Ind = " << D_A1_Ind << endl;
          #endif
          blitz::Array<double, 1> tempDblVecArrA = ::pfs::drp::stella::math::Double(UseRowVecArr);
          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": tempDblVecArrA = " << tempDblVecArrA << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A1_Ind = " << D_A1_Ind << endl;
          #endif
          blitz::Array<double, 1> tempDblVecArrB(D_A1_Ind.size());
          if (!::pfs::drp::stella::math::InterPol(spectrum_Out, tempDblVecArrA, D_A1_Ind, tempDblVecArrB)){
            string message("FiberTrace");
            message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
            message += ": ERROR: InterPol returned FALSE";
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
          }
          spectrum_Out = tempDblVecArrB;
          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): KeyWord_Set(USE_ROW): MARK: INTERPOL: spectrum_Out set to " << spectrum_Out << endl;
          #endif
        }///end if KeyWord_Set(USE_ROW)
        OmegaVecArr.resize(overSample_In);
        OmegaVecArr = Weight;
        #ifdef __DEBUG_SLITFUNC__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): OmegaVecArr set to " << OmegaVecArr << endl;
        #endif

        D_A1_SFO.resize(SFVecArr.size());
        D_A1_SFO = SFVecArr;
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: SFVecArr = " << SFVecArr << endl;
        #endif
        D_A1_SFO = blitz::where(D_A1_SFO < 0., 0., D_A1_SFO);
        D_A1_SFO = blitz::where(D_A1_SFO > (I_NCols_Im+1), 0., D_A1_SFO);
        if (I_Telluric == 3){
          double D_MinLeft = min(D_A1_SFO(blitz::Range(0, int(D_A1_SFO.size()/2))));
          double D_MinRight = min(D_A1_SFO(blitz::Range(int(D_A1_SFO.size()/2), D_A1_SFO.size()-1)));
          double D_MinWing = D_MinLeft;
          if (D_MinRight > D_MinLeft)
            D_MinWing = D_MinRight;
          D_A1_SFO = D_A1_SFO - D_MinWing;
          D_A1_SFO = blitz::where(D_A1_SFO < 0., 0., D_A1_SFO);
        }

        for (int m = 0; m < I_NRows_Im; m++)  /// Evaluate the new spectrum
        {
          OmegaVecArr(0) = D_A2_XX(m,I_A1_IFirstSpec(m));
          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): OmegaVecArr(0) set to " << OmegaVecArr(0) << endl;
          #endif

          SSFArr.resize(overSample_In, I_NCols_Im);
          SSFArr = 0.;
          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): SFVecArr = " << SFVecArr << endl;
          #endif
          TempDVecArr.resize(I_A1_ILastSpec(m) - I_A1_IFirstSpec(m) + 1);
          TempDVecArr = SFVecArr(blitz::Range(I_A1_IFirstSpec(m), I_A1_ILastSpec(m)));
          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): TempDVecArr set to " << TempDVecArr << endl;
          #endif
          #ifdef __DEBUG_CHECK_INDICES__
            if (static_cast<int>(TempDVecArr.size()) != I_NCols_Im * overSample_In)
            {
              string message("FiberTrace");
              message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
              message += ": ERROR: size of TempDVecArr(=" + to_string(TempDVecArr.size()) + ") != D_A2_Im.cols(=" + to_string(I_NCols_Im);
              message += ") * overSample_In(=" + to_string(overSample_In) + ")";
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
            }
          #endif
          blitz::Array<double, 2> *p_D_A2_SSFT = ::pfs::drp::stella::math::Reform(TempDVecArr, I_NCols_Im, overSample_In);
          TempDVecArr.resize(0);
          SSFArr = p_D_A2_SSFT->transpose(blitz::secondDim, blitz::firstDim);
          delete p_D_A2_SSFT;
          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): SSFArr set to " << SSFArr << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): OmegaVecArr set to " << OmegaVecArr << endl;
          #endif

          OArr.resize(SSFArr.cols(), 1);
          D_A2_OT.resize(SSFArr.cols(), SSFArr.rows());
          D_A2_OT = SSFArr.transpose(blitz::secondDim, blitz::firstDim);
          #ifdef __DEBUG_CHECK_INDICES__
            if (OArr.rows() != D_A2_OT.rows())
            {
              string message("FiberTrace");
              message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
              message += ": ERROR: OArr.rows(=" + to_string(OArr.rows()) + ") != D_A2_OT.rows(=" + to_string(D_A2_OT.rows()) + ")";
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
            }
          #endif
          blitz::Array<double, 1> *p_TempDVecArrAA = ::pfs::drp::stella::math::MatrixTimesVecArr(D_A2_OT, OmegaVecArr);
          OArr(blitz::Range::all(), 0) = (*p_TempDVecArrAA);
          delete p_TempDVecArrAA;
          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): OArr set to " << OArr << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
          #endif

          XXX = I_NCols_Im - D_A2_XX(m,I_A1_ILastSpec(m));
          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): XXX set to " << XXX << endl;
          #endif

          #ifdef __DEBUG_CHECK_INDICES__
            if (OArr.rows() < I_NCols_Im)
            {
              string message("FiberTrace");
              message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
              message += ": ERROR: OArr.rows(=" + to_string(OArr.rows()) + ") < D_A2_Im.cols(=" + to_string(I_NCols_Im) + ")";
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
            }
            if (SSFArr.cols() < I_NCols_Im)
            {
              string message("FiberTrace");
              message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
              message += ": ERROR: SSFArr.cols(=" + to_string(SSFArr.cols()) + ") < D_A2_Im.cols(=" + to_string(I_NCols_Im) + ")";
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
            }
            if (static_cast<int>(SFVecArr.size()) < I_A1_ILastSpec(m) + 2)
            {
              string message("FiberTrace");
              message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
              message += ": ERROR: SFVecArr.size(=" + to_string(SFVecArr.size()) + ") < I_A1_ILastSpec(m)(=" + to_string(I_A1_ILastSpec(m)) + ")+2";
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
            }
          #endif
          OArr(blitz::Range(0, I_NCols_Im - 2), 0) += (SSFArr(0, blitz::Range(1, I_NCols_Im - 1)) * XXX);
          OArr(I_NCols_Im - 1, 0) += (SFVecArr(I_A1_ILastSpec(m) + 1) * XXX);
          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): OArr set to " << OArr << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
          #endif

          #ifdef __DEBUG_CHECK_INDICES__
            if (OArr.rows() != I_NCols_Im)
            {
              string message("FiberTrace");
              message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
              message += ": ERROR: OArr.row(=" + to_string(OArr.rows()) + ") < D_A2_Im.cols(=" + to_string(I_NCols_Im) + ")";
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
            }
          #endif
          IVecArr.resize(I_NCols_Im);
          #ifdef __PISKUNOV_ORIG__
            blitz::Array<double, 1> D_A1_TempWhere(D_A2_Im.cols());
            blitz::Array<int, 1> I_A1_IndDev(D_A2_Im.cols());
            I_A1_IndDev = blitz::where(fabs(D_A2_Im(m, blitz::Range::all()) - spectrum_Out(m) * OArr(blitz::Range::all(), 0)) < 3. * D_Dev, 1, 0);
            blitz::Array<int, 1> *P_I_A1_IndDev = ::pfs::drp::stella::math::GetIndex(I_A1_IndDev, I_NInd);
            double D_SS = 0.;
            double D_XX = 0.;
            if (I_NInd > 2){
              blitz::Array<double, 1> D_A1_OArr(I_NInd);
              for (int i_ind=0; i_ind<I_NInd; i_ind++){
                D_A1_OArr(i_ind) = OArr((*P_I_A1_IndDev)(i_ind), 0);
              }
              double D_MeanO = mean(D_A1_OArr);
              for (int i_ind=0; i_ind<I_NInd; i_ind++){
                D_SS += blitz::pow2(D_A2_Im(m, (*P_I_A1_IndDev)(i_ind)) - spectrum_Out(m) * D_A1_OArr(i_ind));
                D_XX += blitz::pow2(D_A1_OArr(i_ind) - D_MeanO) * (I_NInd-2);
              }
              (*P_D_A1_SPErrOut)(m) = D_SS / D_XX;
            }
            else{
              (*P_D_A1_SPErrOut)(m) = 0.;
            }
            cout << "FiberTrace" << _iTrace << "::SlitFunc: (*P_D_A1_SPErrOut)(m) set to " << (*P_D_A1_SPErrOut)(m) << endl;
            delete(P_I_A1_IndDev);//      }
          #endif

          I_A1_Mask.resize(P_I_A2_Mask->cols());
          I_A1_Mask = (*P_I_A2_Mask)(m, blitz::Range::all());
          ::pfs::drp::stella::math::Double(I_A1_Mask, D_A1_Mask);
          D_A2_Mask(m, blitz::Range::all()) = D_A1_Mask;
          #ifdef __DEBUG_CHECK_INDICES__
            if (P_I_A2_Mask->rows() != D_A2_Mask.rows()){
              string message("FiberTrace");
              message += to_string(_iTrace) + "::SlitFunc: ERROR: 5. P_I_A2_Mask->rows() != D_A2_Mask.rows()";
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
            }
          #endif
          NInd = blitz::sum((*P_I_A2_Mask)(m, blitz::Range::all()));
          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum(*P_I_A2_MaskIn) = " << blitz::sum(*P_I_A2_MaskIn) << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum(*P_I_A2_Mask) = " << blitz::sum(*P_I_A2_Mask) << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): P_I_A2_Mask->size() = " << P_I_A2_Mask->size() << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): IVecArr.size() = " << IVecArr.size() << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): NInd set to " << NInd << endl;
          #endif
          #ifdef __DEBUG_CHECK_INDICES__
            if (static_cast<int>(IVecArr.size()) < NInd)
            {
              string message("FiberTrace");
              message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
              message += ": ERROR: IVecArr.size(=" + to_string(IVecArr.size()) + ") < NInd(=" + to_string(NInd) + ")";
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
            }
          #endif


          if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "BAD")) >= 0)
          {

            TempIVecArr.resize(IVecArr.size() - NInd);
            TempIVecArr = 0;
            if (NInd > 0)
              IFirstVecArr.resize(NInd);
            else
              IFirstVecArr.resize(1);
            IFirstVecArr = 0;
            TempInt = 0;
            TempIntA = 0;



            /// MARK MARK MARK!!!
            /// Indices below are still to be checked



            for (int n = 0; n < P_I_A2_Mask->cols(); n++)
            {
              if ((*P_I_A2_Mask)(m,n) > 0)
              {
                #ifdef __DEBUG_CHECK_INDICES__
                  if (TempInt >= static_cast<int>(IFirstVecArr.size()))
                  {
                    string message("FiberTrace");
                    message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                    message += ": ERROR: TempInt(=" + to_string(TempInt) + ") >= IFirstVecArr.size(=" + to_strin(IFirstVecArr.size()) + ")";
                    cout << message << endl;
                    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                  }
                #endif
                IFirstVecArr(TempInt) = n;
                #ifdef __DEBUG_SLITFUNC__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): for(n(==" << n << ")=0; n< P_I_A2_Mask->cols(=" << P_I_A2_Mask->cols() << "; n++): IFirstVecArr(TempInt=" << TempInt << ") set to " << IFirstVecArr(TempInt) << endl;
                #endif
                TempInt++;
              }
              else
              {
                #ifdef __DEBUG_CHECK_INDICES__
                  if (TempIntA >= static_cast<int>(TempIVecArr.size()))
                  {
                    string message("FiberTrace");
                    message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                    message += ": ERROR: TempIntA(=" + to_string(TempIntA) + ") >= TempIVecArr.size(=" + to_string(TempIVecArr.size()) + ")";
                    cout << message << endl;
                    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                  }
                #endif
                TempIVecArr(TempIntA) = n;
                #ifdef __DEBUG_SLITFUNC__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): for(n(==" << n << ")=0; n< P_I_A2_Mask->cols(=" << P_I_A2_Mask->cols() << "; n++): TempIVecArr(TempIntA=" << TempIntA << ") set to " << TempIVecArr(TempIntA) << endl;
                #endif

                TempIntA++;
              }

              (*P_I_A1_JBadVecArr)(0) = 0;
              if (NInd < I_NCols_Im)  /// Bad pixels in column m
              {
                TempLong = (*P_I_A1_JBadVecArr)(0);
                #ifdef __DEBUG_SLITFUNC__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): NInd(=" << NInd << ") < D_A2_Im.cols(=" << I_NCols_Im << "): TempLong set to " << TempLong << endl;
                #endif
                P_I_A1_JBadVecArr->resize(1 + TempIVecArr.size());
                (*P_I_A1_JBadVecArr)(0) = TempLong;
                #ifdef __DEBUG_CHECK_INDICES__
                  if (TempIVecArr.size() != P_I_A1_JBadVecArr->size()-1)
                  {
                    string message("FiberTrace");
                    message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                    message += ": ERROR: TempIVecArr.size(=" + to_string(TempIVecArr.size()) + ") != P_I_A1_JBadVecArr->size(=" + to_string(P_I_A1_JBadVecArr->size());
                    message += ")-1";
                    cout << message << endl;
                    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
                  }
                #endif
                (*P_I_A1_JBadVecArr)(blitz::Range(1, P_I_A1_JBadVecArr->size() - 1)) = TempIVecArr;
                (*P_I_A1_JBadVecArr)(blitz::Range(1, P_I_A1_JBadVecArr->size() - 1)) += (long)I_NCols_Im * m;
                #ifdef __DEBUG_SLITFUNC__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): NInd(=" << NInd << ") < D_A2_Im.cols(=" << I_NCols_Im << "): P_I_A1_JBadVecArr set to " << (*P_I_A1_JBadVecArr) << ")" << endl;
                #endif
              }// end if (NInd < I_NCols_Im)
            }// end for (int n = 0; n < P_I_A2_Mask->cols(); n++)
          }// end if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "BAD")) >= 0)

          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): 2. spectrum_Out = " << spectrum_Out << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): spectrum_Out(m) = " << spectrum_Out(m) << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): OArr = " << OArr << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
          #endif
          #ifdef __DEBUG_CHECK_INDICES__
            if (P_D_A2_Prof_Out->cols() != OArr.rows())
            {
              string message("FiberTrace");
              message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
              message += ": ERROR: P_D_A2_Prof_Out->cols(=" + to_string(P_D_A2_Prof_Out->cols()) + ") != OArr.rows(=" + to_string(OArr.rows()) + ")";
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
            }
          #endif
          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": m=" << m << ": P_D_A2_Prof_Out->size() = " << P_D_A2_Prof_Out->size() << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": m=" << m << ": P_D_A2_Prof_Out->rows() = " << P_D_A2_Prof_Out->rows() << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": m=" << m << ": P_D_A2_Prof_Out->cols() = " << P_D_A2_Prof_Out->cols() << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": m=" << m << ": OArr.size() = " << OArr.size() << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": m=" << m << ": OArr.rows() = " << OArr.rows() << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": m=" << m << ": OArr.cols() = " << OArr.cols() << endl;
          #endif

          double D_Error = 0.;
          double D_MinError = 10000000000000.;
          double D_Offset = -1./static_cast<double>(_fiberTraceProfileFittingControl->overSample);
          if (I_XCorProf == 0)
            D_Offset = 0.;
          double D_MinOffset = 0.;
          D_A1_XX = D_A2_XX(m, blitz::Range::all());
          D_A1_SFO = D_A1_SFO * _fiberTraceProfileFittingControl->overSample / sum(D_A1_SFO);
          #ifdef __DEBUG_SLITFUNC_X__
            std::string fname_sf = DEBUGDIR + std::string("SlitFuncOut") + debugFilesSuffix + std::string(".dat");
            ::pfs::drp::stella::utils::WriteArrayToFile(D_A1_SFO, fname_sf, std::string("ascii"));
          #endif
          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_XCorProf = " << I_XCorProf << endl;
          #endif
          double dSum = 0.;
          for (int i_cross=0; i_cross < I_XCorProf; i_cross++){
            D_A1_XX = D_A2_XX(m, blitz::Range::all()) + D_Offset;
            if (!::pfs::drp::stella::math::InterPol(D_A1_SFO, D_A1_XX, D_A1_XProf, D_A1_YProf)){
              cout << "FiberTrace" << _iTrace << "::SlitFunc: ERROR: InterPol(D_A2_SFO=" << D_A1_SFO << ", D_A1_XX=" << D_A1_XX << ", D_A1_XProf=" << D_A1_XProf << ", D_A1_YProf) returned FALSE" << endl;
              string message("FiberTrace");
              message += to_string(_iTrace) + "::SlitFunc: ERROR: InterPol(D_A1_SFO, D_A1_XX, D_A1_XProf, D_A1_YProf) returned FALSE";
              cout << message << endl;
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
            }
            profile_Out(m, blitz::Range::all()) = where(D_A1_YProf < 0., 0., D_A1_YProf);
            dSum = blitz::sum(profile_Out(m, blitz::Range::all()));
            if (fabs(dSum) > 0.00000000000000001)
              profile_Out(m, blitz::Range::all()) = profile_Out(m, blitz::Range::all()) / dSum;

            D_Error = sqrt(blitz::sum(blitz::pow2(D_A2_Im(m, blitz::Range::all()) - (profile_Out(m, blitz::Range::all()) * spectrum_Out(m)))));
            if (D_Error < D_MinError){
              D_MinError = D_Error;
              D_MinOffset = D_Offset;
            }
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiber " << fiberTraceNumber << ": bin = " << I_Bin << ": row " << m << ": i_cross=" << i_cross << ": offset=" << D_Offset << ": minError=" << D_MinError << ": error=" << D_Error << ": D_MinOffset=" << D_MinOffset << endl;
            #endif
            D_Offset += .2 / I_XCorProf;
          }
          (*P_D_A1_XCorProfOut)(m) = D_MinOffset;
          if (I_XCorProf > 0)
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiber " << fiberTraceNumber << ": bin " << I_Bin << ": row=" << m << ": D_MinOffset=" << D_MinOffset << endl;

        } /// end for (int m = 0; m < I_NRows_Im; m++)

        blitz::Array<double, 1> D_A1_Fit(P_D_A1_XCorProfOut->size());
        D_A1_Fit = 0.;
        if (I_XCorProf > 0){
          #ifdef __DEBUG_SLITFUNC_N__
            string sDebugFileName = DEBUGDIR + std::string("D_A1_XCorProfOut") + debugFilesSuffix + std::string(".dat");
            ::pfs::drp::stella::utils::WriteArrayToFile((*P_D_A1_XCorProfOut), sDebugFileName, string("ascii"));
          #endif
          int I_NDeg = 3;
          blitz::Array<double, 1> *P_D_A1_PolyCoeffs = new blitz::Array<double, 1>(6);
          blitz::Array<string, 1> S_A1_Args_PolyFit(1);
          S_A1_Args_PolyFit(0) = "YFIT";
          void **PP_Args_PolyFit = (void**)malloc(sizeof(void*) * 1);
          PP_Args_PolyFit[0] = &D_A1_Fit;
          blitz::Array<double, 1> D_A1_XPF = ::pfs::drp::stella::math::DIndGenArr(P_D_A1_XCorProfOut->size());
          if (!::pfs::drp::stella::math::PolyFit(D_A1_XPF,
                                          *P_D_A1_XCorProfOut,
                                          I_NDeg,
                                          S_A1_Args_PolyFit,
                                          PP_Args_PolyFit,
                                          P_D_A1_PolyCoeffs)){
            string message("FiberTrace");
            message += to_string(_iTrace) + "::SlitFunc: ERROR: PolyFit(XCorProf) returned FALSE";
            delete(P_D_A1_PolyCoeffs);
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
          }
          #ifdef __DEBUG_SLITFUNC_N__
            sDebugFileName = DEBUGDIR + std::string("D_A1_XCorProf_Fit") + debugFilesSuffix + std::string(".dat");
            ::pfs::drp::stella::utils::WriteArrayToFile(D_A1_Fit, sDebugFileName, string("ascii"));
            cout << "FiberTrace" << _iTrace << "::SlitFunc: after PolyFit: P_D_A1_XCorProfOut = " << *P_D_A1_XCorProfOut << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: after PolyFit: D_A1_Fit = " << D_A1_Fit << endl;
            blitz::Array<double, 1> D_A1_Diff(P_D_A1_XCorProfOut->size());
            D_A1_Diff = (*P_D_A1_XCorProfOut) - D_A1_Fit;
            sDebugFileName = DEBUGDIR + std::string("D_A1_XCorProf_Diff") + debugFilesSuffix + std::string(".dat");
            ::pfs::drp::stella::utils::WriteArrayToFile(D_A1_Diff, sDebugFileName, string("ascii"));
            cout << "FiberTrace" << _iTrace << "::SlitFunc: after PolyFit: D_A1_Diff = " << D_A1_Diff << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: after PolyFit: max(D_A1_Diff) = " << max(D_A1_Diff) << endl;
          #endif
          delete(P_D_A1_PolyCoeffs);
          free(PP_Args_PolyFit);
          (*P_D_A1_XCorProfOut) = D_A1_Fit;
        }
  //      #ifdef __DEBUG_SLITFUNC_X__
  //        cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A2_XX = " << D_A2_XX << endl;
  //        cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A1_Fit = " << D_A1_Fit << endl;
  //      #endif
        for (int m=0; m < I_NRows_Im; m++){
          //D_A1_XX(D_A2_XX.cols());
          D_A1_XX = D_A2_XX(m, blitz::Range::all()) + D_A1_Fit(m);
          #ifdef __DEBUG_SLITFUNC_X__
            //cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A2_XX(" << m << ", *) = " << D_A2_XX(m, blitz::Range::all()) << endl;
            //cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A1_Fit(" << m << ") = " << D_A1_Fit(m) << endl;
            //cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A1_XX = " << D_A1_XX << endl;
            //cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A1_SFO = " << D_A1_SFO << endl;
  //          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A1_Range = " << D_A1_Range << endl;

            std::string fname_x = DEBUGDIR + std::string("x")+debugFilesSuffix + std::string("_row");
            if (m < 100)
              fname_x += "0";
            if (m < 10)
              fname_x += "0";
            fname_x += to_string(m);// + ".dat";
            ::pfs::drp::stella::utils::WriteArrayToFile(D_A2_XX(m, blitz::Range::all()), fname_x+"_orig.dat", std::string("ascii"));
            ::pfs::drp::stella::utils::WriteArrayToFile(D_A1_XX, fname_x+".dat", std::string("ascii"));
          #endif
          if (!::pfs::drp::stella::math::InterPol(D_A1_SFO, D_A1_XX, D_A1_XProf, D_A1_YProf)){
            cout << "FiberTrace" << _iTrace << "::SlitFunc: ERROR: InterPol(D_A1_SFO=" << D_A1_SFO << ", D_A1_XX=" << D_A1_XX << ", D_A1_XProf=" << D_A1_XProf << ", D_A1_YProf) returned FALSE" << endl;
            string message("FiberTrace");
            message += to_string(_iTrace) + "::SlitFunc: ERROR: InterPol(D_A1_SFO, D_A1_XX, D_A1_XProf, D_A1_YProf) returned FALSE";
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
          }
          profile_Out(m, blitz::Range::all()) = where(D_A1_YProf(m) < 0., 0., D_A1_YProf);
          double D_SumSF = blitz::sum(profile_Out(m, blitz::Range::all()));
          if (fabs(D_SumSF) > 0.00000000000000001)
            profile_Out(m, blitz::Range::all()) = profile_Out(m, blitz::Range::all()) / D_SumSF;
          if (I_Telluric == 3){
            profile_Out(m, blitz::Range::all()) = profile_Out(m, blitz::Range::all()) - min(profile_Out(m, blitz::Range::all()));
            D_SumSF = blitz::sum(profile_Out(m, blitz::Range::all()));
            if (fabs(D_SumSF) > 0.00000000000000001){
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "D_SumSF = " << D_SumSF << endl;
              #endif
              profile_Out(m, blitz::Range::all()) = profile_Out(m, blitz::Range::all()) / D_SumSF;
            }
          }

          (*P_D_A2_Prof_Out)(m, blitz::Range::all()) = profile_Out(m,blitz::Range::all());/// * spectrum_Out(m)
          if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "IM_OUT")) >= 0)
          {
            (*P_D_A2_Im_Out)(m, blitz::Range::all()) = profile_Out(m,blitz::Range::all()) * spectrum_Out(m);
          }
          #ifdef __DEBUG_SLITFUNC_X__
            std::string fname_prof = DEBUGDIR + std::string("ProfileOut")+ debugFilesSuffix + "_row";
            if (m < 100)
              fname_prof += "0";
            if (m < 10)
              fname_prof += "0";
            fname_prof += to_string(m) + std::string(".dat");
            ::pfs::drp::stella::utils::WriteArrayToFile(profile_Out(m,blitz::Range::all()), fname_prof, std::string("ascii"));
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fname_prof=<" << fname_prof << "> written" << endl;
          #endif
        }/// end for (int m=0; m < I_NRows_Im; m++){
        #ifdef __DEBUG_SLITFUNC_X__
          std::string fname_rec = DEBUGDIR + std::string("ImRecOut") + debugFilesSuffix + std::string(".dat");
          ::pfs::drp::stella::utils::WriteArrayToFile(*P_D_A2_Im_Out, fname_rec, std::string("ascii"));
        #endif

        #ifdef __DEBUG_SLITFUNC_N__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": profile_Out set to " << profile_Out << endl;
        #endif
        if (I_Telluric == 2)
        {
          argpos = 0;
          if (ErrorsRead)
          {
            S_A1_Args_Fit(argpos) = "MEASURE_ERRORS_IN";
            PP_Args_Fit[argpos] = P_D_A2_Errors;
            argpos++;
          }

          S_A1_Args_Fit(argpos) = "CHISQ_OUT";
          PP_Args_Fit[argpos] = &D_A1_ChiSquare_LinFit;
          argpos++;

          S_A1_Args_Fit(argpos) = "SIGMA_OUT";
          PP_Args_Fit[argpos] = &D_A2_Sigma_LinFit;
          argpos++;

          S_A1_Args_Fit(argpos) = "Q_OUT";
          PP_Args_Fit[argpos] = &D_A1_Probability_LinFit;
          argpos++;

          #ifdef __DEBUG_TELLURIC__
            cout << endl << "FiberTrace" << _iTrace << "::SlitFunc: before Fit: D_A2_MySF = " << D_A2_MySF << endl << endl;
          #endif

          for (int pppp=0; pppp < P_D_A2_Prof_Out->rows(); pppp++){
            if (fabs(blitz::sum((*P_D_A2_Prof_Out)(pppp, blitz::Range::all()))) < 0.00000000000000001)
              (*P_D_A2_Prof_Out)(pppp, blitz::Range::all()) = 1.;
            (*P_D_A2_Prof_Out)(pppp, blitz::Range::all()) /= blitz::sum((*P_D_A2_Prof_Out)(pppp, blitz::Range::all()));
          }

          /// --- WHAT TODO?!?

          int I_RangeMinRow, I_RangeMaxRow, I_RangeMinCol, I_RangeMaxCol;
          int I_RangeWidth = 5;
          int xxx, yyy, i_nrows, i_ncols;//, i_row, i_col, i_indexTemp, zzz;
          blitz::Array<int,1> I_A1_IndicesRange(2 * I_RangeWidth + 1);
          blitz::Array<int,3> I_A3_IndicesRange(1,1,1);//(2 * I_RangeWidth + 1, 2 * I_RangeWidth + 1, 2);
          blitz::Array<int, 2> I_A2_TempArrA(1,1);
          blitz::Array<int, 2> I_A2_TempArr(1,1);
          blitz::Array<int, 2> I_A2_Temp(1,1);

          for (xxx = 0; xxx < P_I_A2_Mask->rows(); xxx++){
            for (yyy = 0; yyy < P_I_A2_Mask->cols(); yyy++){
              I_RangeMinRow = xxx - I_RangeWidth;
              if (I_RangeMinRow < 0)
                I_RangeMinRow = 0;
              I_RangeMaxRow = xxx + I_RangeWidth;
              if (I_RangeMaxRow >= P_I_A2_Mask->rows())
                I_RangeMaxRow = P_I_A2_Mask->rows() - 1;
              #ifdef __DEBUG_TELLURIC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": xxx = " << xxx << ", yyy = " << yyy << ": I_RangeMinRow = " << I_RangeMinRow << ", I_RangeMaxRow = " << I_RangeMaxRow << endl;
              #endif

              I_RangeMinCol = yyy - I_RangeWidth;
              if (I_RangeMinCol < 0)
                I_RangeMinCol = 0;
              I_RangeMaxCol = yyy + I_RangeWidth;
              if (I_RangeMaxCol >= P_I_A2_Mask->cols())
                I_RangeMaxCol = P_I_A2_Mask->cols() - 1;
              #ifdef __DEBUG_TELLURIC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": xxx = " << xxx << ", yyy = " << yyy << " I_RangeMinCol = " << I_RangeMinCol << ", I_RangeMaxCol = " << I_RangeMaxCol << endl;
              #endif

              i_nrows = (I_RangeMaxRow - I_RangeMinRow + 1);
              i_ncols = (I_RangeMaxCol - I_RangeMinCol + 1);
              #ifdef __DEBUG_TELLURIC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": i_nrows = " << i_nrows << ", i_ncols = " << i_ncols << endl;
              #endif

              I_A3_IndicesRange.resize(i_nrows, i_ncols, 2);
              for (int zzz = 0; zzz < i_nrows; zzz++){
                for (int qqq = 0; qqq < i_ncols; qqq++){
                  I_A3_IndicesRange(zzz,qqq,0) = I_RangeMinRow + zzz;
                  I_A3_IndicesRange(zzz,qqq,1) = I_RangeMinCol + qqq;
                }
              }
              #ifdef __DEBUG_TELLURIC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_A3_IndicesRange(Subarr) set to " << I_A3_IndicesRange << endl;
              #endif
              I_A2_Temp.resize(P_I_A2_Mask->rows(), P_I_A2_Mask->cols());
              I_A2_Temp = (*P_I_A2_Mask);
              I_A2_TempArr.resize(I_A3_IndicesRange.rows(), I_A3_IndicesRange.cols());
              I_A2_TempArr = ::pfs::drp::stella::math::GetSubArrCopy(I_A2_Temp, I_A3_IndicesRange);
              #ifdef __DEBUG_TELLURIC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_A2_TempArr(Subarr) set to " << I_A2_TempArr << endl;
              #endif

              if (((*P_I_A2_Mask)(xxx,yyy) == 1) ||
                (blitz::sum(I_A2_TempArr) < I_A2_TempArr.size() / 1.5) ||
                (blitz::sum((*P_I_A2_Mask)(xxx,blitz::Range::all())) == 0))
              {
                D_A2_ImTimesMask(xxx,yyy) = D_A2_ImBak(xxx,yyy);
                D_A2_SFTimesMask(xxx,yyy) = (*P_D_A2_Prof_Out)(xxx,yyy);
                #ifdef __DEBUG_TELLURIC__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": (*P_I_A2_Mask)(xxx,yyy) = " << (*P_I_A2_Mask)(xxx,yyy) << " == 1 ||" << endl;
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum(I_A2_TempArr) = " << blitz::sum(I_A2_TempArr) << " < I_A2_TempArr.size() / 1.5 = " << I_A2_TempArr.size() / 1.5 << " ||" << endl;
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum((*P_I_A2_Mask)(xxx,blitz::Range::all())) = " << blitz::sum((*P_I_A2_Mask)(xxx,blitz::Range::all())) << " == 0" << endl;
                #endif
              }
              else{
                D_A2_ImTimesMask(xxx,yyy) = D_A1_Sky(xxx);
                D_A2_SFTimesMask(xxx,yyy) = 0.;
                #ifdef __DEBUG_TELLURIC__
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": (*P_I_A2_Mask)(xxx,yyy) = " << (*P_I_A2_Mask)(xxx,yyy) << " != 1 &&" << endl;
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum(I_A2_TempArr) = " << blitz::sum(I_A2_TempArr) << " >= I_A2_TempArr.size() / 1.5 = " << I_A2_TempArr.size() / 1.5 << " &&" << endl;
                  cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum((*P_I_A2_Mask)(xxx,blitz::Range::all())) = " << blitz::sum((*P_I_A2_Mask)(xxx,blitz::Range::all())) << " != 0" << endl;
                #endif
              }
              #ifdef __DEBUG_TELLURIC__
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_ImTimesMask(xxx=" << xxx << ",yyy=" << yyy << ") set to " << D_A2_ImTimesMask(xxx, yyy) << endl;
                cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_SFTimesMask(xxx=" << xxx << ",yyy=" << yyy << ") set to " << D_A2_SFTimesMask(xxx, yyy) << endl;
              #endif
            }
          }

          D_A1_Sky = 1.;
          #ifdef __DEBUG_TELLURIC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_ImTimesMask = " << D_A2_ImTimesMask << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": *P_D_A2_Prof_Out = " << *P_D_A2_Prof_Out << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_SFTimesMask = " << D_A2_SFTimesMask << endl;
            S_MySF = DEBUGDIR + std::string("D_A2_ImTimesMask_beforeFit_iterSky") + to_string(I_Iter_Sky) + debugFilesSuffix + std::string(".fits");
            ::pfs::drp::stella::utils::WriteFits(&D_A2_ImTimesMask, S_MySF);
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": File " << S_MySF << " written" << endl;

            S_MySF = DEBUGDIR + std::string("D_A2_SFTimesMask_beforeFit_iterSky") + to_string(I_Iter_Sky) + debugFilesSuffix + std::string(".fits");
            ::pfs::drp::stella::utils::WriteFits(&D_A2_SFTimesMask, S_MySF);
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": File " << S_MySF << " written" << endl;
          #endif

          if (!::pfs::drp::stella::math::LinFitBevington(D_A2_ImTimesMask,
                                                  D_A2_SFTimesMask,
                                                  D_A1_MySP,
                                                  D_A1_Sky,
                                                  S_A1_Args_Fit,
                                                  PP_Args_Fit))
          {
            string message("FiberTrace");
            message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
            message += ": ERROR: Fit returned FALSE!";
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
          }

          #ifdef __DEBUG_TELLURIC__
            S_MySF = DEBUGDIR + std::string("D_A2_ImTimesMask_afterFit_iterSky") + to_string(I_Iter_Sky) + debugFilesSuffix + std::string(".fits");
            ::pfs::drp::stella::utils::WriteFits(&D_A2_ImTimesMask, S_MySF);
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": File " << S_MySF << " written" << endl;

            S_MySF = DEBUGDIR + std::string("D_A2_SFTimesMask_afterFit_iterSky") + to_string(I_Iter_Sky) + debugFilesSuffix + std::string(".fits");
            ::pfs::drp::stella::utils::WriteFits(&D_A2_SFTimesMask, S_MySF);
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": File " << S_MySF << " written" << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": 3. spectrum_Out = " << spectrum_Out << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Fit returned TRUE" << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": after Fit: D_A1_MySP = " << D_A1_MySP << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": after Fit: D_A1_Sky = " << D_A1_Sky << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": after Fit: D_A1_ChiSquare_LinFit = " << D_A1_ChiSquare_LinFit << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": after Fit: D_A1_Probability_LinFit = " << D_A1_Probability_LinFit << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": after Fit: D_A2_Sigma_LinFit = " << D_A2_Sigma_LinFit << endl;
          #endif
          if (I_Stop > 0)
            return false;

          #ifdef __DEBUG_TELLURIC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": __TELLURIC_MINE__: spectrum_Out = " << spectrum_Out << endl;
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": __TELLURIC_MINE__: D_A1_MySP set to " << D_A1_MySP << endl;
          #endif
          TempDVecArr.resize(D_A1_MySP.size());
          TempDVecArr = spectrum_Out - D_A1_MySP;

          /// subtract new sky from D_A2_Im
          for (int p=0; p < static_cast<int>(D_A1_Sky.size()); p++)
          {
            D_A2_Im(p,blitz::Range::all()) = D_A2_ImBak(p, blitz::Range::all()) - D_A1_Sky(p);
          }
          #ifdef __DEBUG_TELLURIC__
            tempFileName = DEBUGDIR + std::string("D_A1_MySky_iterSky") + to_string(I_Iter_Sky) + debugFilesSuffix + std::string(".dat");
            ofstream *P_SP_Log = new ofstream(tempFileName.c_str());
            for (int isp=0; isp < P_D_A1_MySky->size(); isp++)
            {
              (*P_SP_Log) << (*P_D_A1_MySky)(isp) << endl;
            }
            delete(P_SP_Log);


            S_MySF = DEBUGDIR + std::string("D_A2_Im_Minus_Sky_new_iterSky") + to_string(I_Iter_Sky) + debugFilesSuffix + std::string(".fits");
            ::pfs::drp::stella::utils::WriteFits(&D_A2_Im, S_MySF);
            cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": File " << S_MySF << " written" << endl;
          #endif

          Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "STOP");
          if (Pos >= 0)
          {
            if (*(int*)ArgV_In[Pos] == 1)
            {
              cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(STOP) == 1" << endl;
              if (I_Iter_SF == 1){
                string message("FiberTrace");
                message += to_string(_iTrace) + "::SlitFunc: fiberTraceNumber = " + to_string(fiberTraceNumber) + ": I_Bin = " + to_string(I_Bin);
                message += ": KeyWord_Set(STOP) == 1, I_Iter_SF == " + to_string(I_Iter_SF);
                cout << message << endl;
                throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
              }
            }
          }
        }/// end if (I_TELLURIC == 2)

        if (I_Telluric != 2)
          break;
        D_A1_OldSky = abs(D_A1_Sky - D_A1_OldSky) / D_A1_Sky;
        D_A1_OldSky = blitz::where(D_A1_Sky > 0., D_A1_OldSky, 0.);
        #ifdef __DEBUG_SLITFUNC__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_Sky = " << I_Iter_Sky << ": D_A1_OldSky - D_A1_Sky = " << D_A1_OldSky << endl;
        #endif
        if (mean(D_A1_OldSky) < 0.005)
          break;
        D_A1_OldSky = D_A1_Sky;
      } while(I_Iter_Sky < static_cast<int>(_fiberTraceProfileFittingControl->maxIterSky));
    }/// end if (!doSplineFit)

    blitz::Array<double, 1> D_A1_CY(D_A2_Im.rows());
    D_A1_CY = 0.;
    blitz::Array<double, 1> D_A1_CY_SP(D_A2_Im.rows());
    D_A1_CY_SP = 0.;
    blitz::Array<double, 1> D_A1_DY(D_A2_Im.rows());
    D_A1_DY = 0.;

    blitz::Array<double, 1> D_A1_SumMaskProf(D_A2_Im.rows());
    D_A1_SumMaskProf = 0.;
    blitz::Array<double, 1> D_A1_SumMaskProfSquaredDivByErr(D_A2_Im.rows());
    D_A1_SumMaskProfSquaredDivByErr = 0.;
    blitz::Array<double, 1> D_A1_SPErrHorne(D_A2_Im.rows());
    D_A1_SPErrHorne = 0.;

    double D_Sum_ProfTimesWeight = 0.;
    int I_SubPix = 0;
    #ifdef __DEBUG_SLITFUNC__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A2_Im.rows() = " << D_A2_Im.rows() << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A2_Im.cols() = " << D_A2_Im.cols() << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: SFVecArr = " << SFVecArr << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A2_Weights(0:3, *) = " << D_A2_Weights(blitz::Range(0,3), blitz::Range::all()) << endl;
    #endif

    if (I_Telluric == 3)
      SFVecArr = SFVecArr - min(SFVecArr);

    #ifdef __DEBUG_SLITFUNC_FILES__
      string S_SF = "SFVecArr_Out.fits";
      ::pfs::drp::stella::utils::WriteFits(&SFVecArr, S_SF);
    #endif

    SFVecArr = SFVecArr / blitz::sum(SFVecArr);

    double D_Sum_AxySquared = 0.;
    double D_Sum_SigmaSquared_AxySquared = 0.;

    for (int i_row=0; i_row<D_A2_Im.rows(); i_row++){
      D_A1_CY = 0.;
      D_A1_DY = 0.;
      D_Sum_AxySquared = 0.;
      D_Sum_SigmaSquared_AxySquared = 0.;
      double sumProfTimesWeightRow = 0.;
      for (int i_col=0; i_col<D_A2_Im.cols(); i_col++){
        D_Sum_ProfTimesWeight = 0.;
        I_SubPix = 0;
        for (int i_j=0; i_j<static_cast<int>(SFVecArr.size())-1; i_j++){
          #ifdef __DEBUG_SLITFUNC__
            cout << "FiberTrace" << _iTrace << "::SlitFunc: i_row = " << i_row << ": i_col = " << i_col << ": I_SubPix = " << I_SubPix << ": i_j = " << i_j << ": D_A2_XX(i_row, i_j) = " << D_A2_XX(i_row, i_j) << ": int(D_A2_XX(i_row, i_j)) = " << int(D_A2_XX(i_row, i_j)) << endl;
          #endif
          if ((int(D_A2_XX(i_row, i_j)) == i_col) || ((int(D_A2_XX(i_row, i_j)) <= i_col) && (D_A2_XX(i_row, i_j+1) > i_col))){
            #ifdef __DEBUG_TELLURIC__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: SubPixel found" << endl;
            #endif
            D_Sum_ProfTimesWeight += SFVecArr(i_j) * D_A2_Weights(i_row, I_SubPix);
            #ifdef __DEBUG_TELLURIC__
              cout << "FiberTrace" << _iTrace << "::SlitFunc: i_row = " << i_row << ": i_col = " << i_col << ": I_SubPix = " << I_SubPix << ": i_j = " << i_j << ": SFVecArr(i_j) = " << SFVecArr(i_j) << ": D_A2_Weights(i_row, I_SubPix) = " << D_A2_Weights(i_row, I_SubPix) << ": D_Sum_ProfTimesWeight = " << D_Sum_ProfTimesWeight << endl;
            #endif
            I_SubPix++;
            if (I_SubPix == overSample_In + 1)
              i_j = SFVecArr.size();
          }
        }/// end for (int i_j=0; i_j<SFVecArr.size()-1; i_j++){

        D_Sum_AxySquared += (*P_I_A2_Mask)(i_row, i_col) * blitz::pow2(D_Sum_ProfTimesWeight);
        D_Sum_SigmaSquared_AxySquared += (*P_I_A2_Mask)(i_row, i_col) * blitz::pow2((*P_D_A2_Errors)(i_row, i_col)) * blitz::pow2(D_Sum_ProfTimesWeight);

        sumProfTimesWeightRow += D_Sum_ProfTimesWeight;
        #ifdef __DEBUG_SLITFUNC__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: i_row = " << i_row << ": i_col = " << i_col << ": sumProfTimesWeightRow = " << sumProfTimesWeightRow << endl;
        #endif
        D_A1_CY_SP(i_row) += (*P_I_A2_Mask)(i_row, i_col) * D_A2_Im(i_row, i_col) * D_Sum_ProfTimesWeight;
        D_A1_DY(i_row) += (*P_I_A2_Mask)(i_row, i_col) * blitz::pow2(D_Sum_ProfTimesWeight);
        #ifdef __DEBUG_TELLURIC__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: i_row = " << i_row << ": i_col = " << i_col << ": D_A1_CY_SP(i_row) = " << D_A1_CY_SP(i_row) << endl;
          cout << "FiberTrace" << _iTrace << "::SlitFunc: i_row = " << i_row << ": i_col = " << i_col << ": D_A1_DY(i_row) = " << D_A1_DY(i_row) << endl;
        #endif

        D_A1_SumMaskProf(i_row) += (*P_I_A2_Mask)(i_row, i_col) * D_Sum_ProfTimesWeight;
        D_A1_SumMaskProfSquaredDivByErr(i_row) += (*P_I_A2_Mask)(i_row, i_col) * blitz::pow2(D_Sum_ProfTimesWeight) / blitz::pow2((*P_D_A2_Errors)(i_row, i_col));
      }/// end for (int i_col=0; i_col<D_A2_Im.cols(); i_col++){

      #ifndef __PISKUNOV_ORIG__
        (*P_D_A1_SPErrOut)(i_row) = sqrt(D_Sum_SigmaSquared_AxySquared / blitz::pow2(D_Sum_AxySquared)) / overSample_In;

        #ifdef __DEBUG_TELLURIC__
          cout << "FiberTrace" << _iTrace << "::SlitFunc: (*P_D_A1_SPErrOut)(i_row = " << i_row << ") = " << (*P_D_A1_SPErrOut)(i_row) << endl;
        #endif
      #endif

      (*P_D_A1_SPOut)(i_row) = (D_A1_CY_SP(i_row) / D_A1_DY(i_row)) / overSample_In;
      #ifdef __DEBUG_TELLURIC__
        cout << "FiberTrace" << _iTrace << "::SlitFunc: (*P_D_A1_SPOut)(i_row = " << i_row << ") = " << (*P_D_A1_SPOut)(i_row) << endl;
      #endif
    }/// end for (int i_row=0; i_row<D_A2_Im.rows(); i_row++)
    #ifdef __DEBUG_SLITFUNC__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A1_CY_SP = " << D_A1_CY_SP << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: D_A1_DY = " << D_A1_DY << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: spectrum_Out = " << spectrum_Out << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: P_D_A1_SPOut = " << *P_D_A1_SPOut << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: P_D_A1_SPErrOut = " << *P_D_A1_SPErrOut << endl;
    #endif
    blitz::Array<double, 1> D_A1_SNR(spectrum_Out.size());
    D_A1_SNR = spectrum_Out / (*P_D_A1_SPErrOut);
    #ifdef __DEBUG_SLITFUNC__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: SNR(spectrum_Out / P_D_A1_SPErrOut) = " << D_A1_SNR << endl;
    #endif
    D_A1_SNR = (*P_D_A1_SPOut) / (*P_D_A1_SPErrOut);
    #ifdef __DEBUG_SLITFUNC__
      cout << "FiberTrace" << _iTrace << "::SlitFunc: SNR(P_D_A1_SPOut / P_D_A1_SPErrOut) = " << D_A1_SNR << endl;
    #endif
    if (I_Telluric == 2)
    {
      /// set P_D_A1_MySky to new sky
      (*P_D_A1_MySky) = D_A1_Sky;
      P_D_A1_ErrSky->resize(D_A2_Im.rows());
      *P_D_A1_ErrSky = D_A2_Sigma_LinFit(blitz::Range::all(), 1);

      if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "ERRORS_OUT")) >= 0){
        P_D_A1_ErrOut->resize(D_A2_Im.rows());
        *P_D_A1_ErrOut = D_A2_Sigma_LinFit(blitz::Range::all(), 0);
      }

      for (int pp=0; pp < D_A2_Im.rows(); pp++)
        (*P_D_A2_Errors)(pp, blitz::Range::all()) += (*P_D_A1_ErrSky)(pp);
    }

    #ifdef __DEBUG_SLITFUNC__
      cout << "start P_I_A1_BadVecArr" << endl;
    #endif
    if (P_I_A1_JBadVecArr->size() > 1)
    {
      (*P_I_A1_JBadVecArr)(blitz::Range(0, P_I_A1_JBadVecArr->size() - 2)) = (*P_I_A1_JBadVecArr)(blitz::Range(1, P_I_A1_JBadVecArr->size() - 1));
      P_I_A1_JBadVecArr->resizeAndPreserve(P_I_A1_JBadVecArr->size() - 1);
    }
    else
    {
      P_I_A1_JBadVecArr->resize(1);
      (*P_I_A1_JBadVecArr) = -1;
    }
    spectrum_Out = blitz::where(spectrum_Out < 0., 0., spectrum_Out);
    #ifdef __DEBUG_SLITFUNC_SF__
      cout << "spectrum_Out (final) = " << spectrum_Out << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": READY: P_I_A1_JBadVecArr set to " << (*P_I_A1_JBadVecArr) << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": READY: SFVecArr set to " << SFVecArr << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": READY: spectrum_Out set to " << spectrum_Out << endl;
      cout << "FiberTrace" << _iTrace << "::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": READY: P_D_A2_Prof_Out set to " << *P_D_A2_Prof_Out << endl;//->transpose(blitz::secondDim, blitz::firstDim) << endl;
    #endif

    if (::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "PROF_OUT") < 0)
    {
      delete P_D_A2_Prof_Out;
    }

    if (::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "XCOR_PROF") < 0){
      delete(P_D_A1_XCorProfOut);
    }

    Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "SP_FIT");
    if (Pos >= 0)
    {
      blitz::Array<double, 1> *P_D_A1_SPFit = (blitz::Array<double, 1>*)ArgV_In[Pos];
      P_D_A1_SPFit->resize(D_A1_MySP.size());
      *P_D_A1_SPFit = D_A1_MySP;
    }

    #ifdef __DEBUG_SLITFUNC__
      cout << "deleting pointers" << endl;
    #endif
    if ((Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "IM_OUT")) < 0)
    {
      if (P_D_A2_Im_Out != NULL)
        delete P_D_A2_Im_Out;
    }

    #ifdef __DEBUG_TELLURIC__
      S_SP = "spectrum_Out1Out.fits";
      D_A2_SPTemp.resize(spectrum_Out.size(), 1);
      D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
      ::pfs::drp::stella::utils::WriteFits(&D_A2_SPTemp, S_SP);
    #endif

    #ifdef __DEBUG_SEDM__
      string sFileName_SPVecArrOut = DEBUGDIR + std::string("SEDM_spectrum_OutOut") + debugFilesSuffix + std::string(".dat");
      ::pfs::drp::stella::utils::WriteArrayToFile(spectrum_Out, sFileName_SPVecArrOut, string("ascii"));

      string sFileName_SFVecArrOut = DEBUGDIR + std::string("SEDM_SFVecArrOut") + debugFilesSuffix + std::string(".dat");
      ::pfs::drp::stella::utils::WriteArrayToFile(SFVecArr, sFileName_SFVecArrOut, string("ascii"));

      string S_MaskOut = "Mask_SF.fits";
      if (!::pfs::drp::stella::utils::WriteFits(P_I_A2_Mask, S_MaskOut)){
        string message("FiberTrace");
        message += to_string(_iTrace) + "::SlitFunc: ERROR: WriteFits(Mask) returned FALSE";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      cout << "FiberTrace" << _iTrace << "::SlitFunc: " << CS_MaskOut << " written" << endl;
    #endif
    if ((blitz::sum(*P_I_A2_Mask) > static_cast<int>(P_I_A2_Mask->size()) / 2) && (blitz::sum(*P_I_A2_Mask) == blitz::sum(*P_I_A2_MaskIn))){
      cout << "FiberTrace" << _iTrace << "::SlitFunc: WARNING: No cosmics detected" << endl;
    }
    P_D_A1_SFO_Out->resize(D_A1_SFO.size());
    (*P_D_A1_SFO_Out) = D_A1_SFO;

    #ifdef __DEBUG_SLITFUNC__
      cout << "deleting parameters" << endl;
    #endif
    a.resize(0);
    AKLArr.resize(0,0);
    b.resize(0);
    BKLArr.resize(0,0);
    BKLIndVecArr.resize(0);
    BLVecArr.resize(0);
    c.resize(0);
    D_A1_Ind.resize(0);
    D_A2_AKLT.resize(0,0);
    D_A2_OT.resize(0,0);
    D_A2_SPVecTimesBKLArr.resize(0, 0);
//    IFirstVecArr.resize(0);
//    ILastVecArr.resize(0);
    IndVecArr.resize(0);
    OArr.resize(0,0);
    OIndVecArr.resize(0);
    OLIndVecArr.resize(0);
    OOVecArr.resize(0);
    OmegaVecArr.resize(0);
    OmegaArr.resize(0,0);
    D_A2_TempAA.resize(0,0);
    D_A1_TempDVecArr.resize(0);
    RVecArr.resize(0);
    SPOldVecArr.resize(0);
    SSFArr.resize(0,0);
    TempArray.resize(0,0);
    TempDVecArr.resize(0);
    TempDVecArrB.resize(0);
    TempDVecArrC.resize(0);
    TempDArr.resize(0,0);
    TempIVecArr.resize(0);
    UseRowVecArr.resize(0);
    IVecArr.resize(0);
    XVecArr.resize(0);
    XCenVecArr.resize(0);
    *P_I_A2_MaskIn = *P_I_A2_Mask;
    delete(P_I_A2_Mask);
    Pos = ::pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MASK");
    if (Pos < 0)
      delete(P_I_A2_MaskIn);
    P_D_A1_ErrSky->resize(0);
    P_D_A1_MySky->resize(0);
    P_I_A1_JBadVecArr->resize(0);
    free(PP_Args_Fit);
    return true;
  }
  /**
   *  MkSlitFunc without arguments
   **/
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::MkSlitFunc()
  {
    blitz::Array<string, 1> S_A1_Args(1);
    S_A1_Args = "";
    void **PP_Args;
    PP_Args = (void**)malloc(sizeof(void*) * 1);

    int I_temp = 0;
    PP_Args[0] = &I_temp;

    return MkSlitFunc(S_A1_Args, PP_Args);
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::fitSpline(const blitz::Array<double, 2> &fiberTraceSwath_In,
                                                                     const blitz::Array<int, 1> &iFirst_In,
                                                                     const blitz::Array<double, 1> &xOverSampled_In,
                                                                     blitz::Array<double, 1> &profileOverSampled_Out,
                                                                     const blitz::Array<double, 2> &profileXValuesPerRowOverSampled_In,
                                                                     const blitz::Array<double, 1> &profileXValuesAllRows_In,
                                                                     blitz::Array<double, 2> &profilePerRow_Out)
  {
    #ifdef __DEBUG_SPLINE__
      cout << "FiberTrace" << _iTrace << "::fitSpline: fiberTraceSwath_In = " << fiberTraceSwath_In << endl;
      cout << "FiberTrace" << _iTrace << "::fitSpline: iFirst_In = " << iFirst_In << endl;
      cout << "FiberTrace" << _iTrace << "::fitSpline: xOverSampled_In = " << xOverSampled_In << endl;
      cout << "FiberTrace" << _iTrace << "::fitSpline: profileXValuesPerRowOverSampled_In = " << profileXValuesPerRowOverSampled_In << endl;
      cout << "FiberTrace" << _iTrace << "::fitSpline: profileXValuesAllRows_In = " << profileXValuesAllRows_In << endl;
    #endif
    /// check input paramters
    if (static_cast<int>(iFirst_In.size()) != fiberTraceSwath_In.rows()){
      string message("FiberTrace");
      message += to_string(_iTrace) + "::fitSpline: ERROR: iFirst_In.size(=" + to_string(iFirst_In.size()) + ") != fiberTraceSwath_In.rows(=";
      message += to_string(fiberTraceSwath_In.rows()) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
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
    for (int i = 0; i < fiberTraceSwath_In.rows(); ++i){
      ccdRow = fiberTraceSwath_In(i,blitz::Range::all()) / blitz::sum(fiberTraceSwath_In(i,blitz::Range::all()));
      for (int j = 0; j < fiberTraceSwath_In.cols(); ++j){
//        *iter_xVec = profileXValuesPerRowOverSampled_In(i,j);
        *iter_xVec = double(j) * double(_fiberTraceProfileFittingControl->overSample) + double(iFirst_In(i)) + (double(_fiberTraceProfileFittingControl->overSample)/2.);
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
    ::pfs::drp::stella::math::spline spline;
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
      message += to_string(i) + string("): ERROR: i > _traces->size()=") + to_string(_traces->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return _traces->at(i); 
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::erase(const size_t iStart, const size_t iEnd){
    if (iStart >= _traces->size()){
      string message("FiberTraceSet::erase(iStart=");
      message += to_string(iStart) + string("): ERROR: iStart >= _traces->size()=") + to_string(_traces->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (iEnd >= _traces->size()){
      string message("FiberTraceSet::erase(iEnd=");
      message += to_string(iEnd) + string("): ERROR: iEnd >= _traces->size()=") + to_string(_traces->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if ((iEnd > 0) && (iStart > iEnd)){
      string message("FiberTraceSet::erase(iStart=");
      message += to_string(iStart) + string("): ERROR: iStart > iEnd=") + to_string(iEnd);
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
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
      cout << "FiberTraceSet::addFiberTrace: (*_traces)[" << size << "]->_iTrace set to " << (*_traces)[size]->getITrace();
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

  namespace pfs { namespace drp { namespace stella { namespace math {
    template<typename ImageT, typename MaskT, typename VarianceT>
    PTR(pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>) findAndTraceApertures(const PTR(const afwImage::MaskedImage<ImageT, MaskT, VarianceT>) &maskedImage,
                                                                                     const PTR(const pfsDRPStella::FiberTraceFunctionFindingControl) &fiberTraceFunctionFindingControl){
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "::pfs::drp::stella::math::findAndTraceApertures started" << endl;
      #endif
      //try{
        if (static_cast<int>(fiberTraceFunctionFindingControl->apertureFWHM * 2.) + 1 <= fiberTraceFunctionFindingControl->nTermsGaussFit){
          cout << "::pfs::drp::stella::math::findAndTraceApertures: WARNING: fiberTraceFunctionFindingControl->apertureFWHM too small for GaussFit -> Try lower fiberTraceFunctionFindingControl->nTermsGaussFit!" << endl;
          exit(EXIT_FAILURE);
        }
  //      PTR(pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>) fiberTraceSet(new pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>());
        PTR(pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>) fiberTraceSet(new pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>());
        pfsDRPStella::FiberTraceFunction fiberTraceFunction;
        fiberTraceFunction.fiberTraceFunctionControl = fiberTraceFunctionFindingControl->fiberTraceFunctionControl;
        #ifdef __DEBUG_FINDANDTRACE__
          cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.fiberTraceFunctionControl set" << endl;
        #endif

        //  int I_PolyFitOrder = 3;
        unsigned int I_ApertureNumber = 0;
        int I_StartIndex;
        int I_FirstWideSignal;
        int I_FirstWideSignalEnd;
        int I_FirstWideSignalStart;
        unsigned int I_Length, I_ApertureLost, I_Row_Bak;//, I_LastRowWhereApertureWasFound
        int I_ApertureLength;
        int I_NInd;
        double D_Max;
        bool B_ApertureFound;
        blitz::Array<ImageT, 2> ccdTImage = ::pfs::drp::stella::utils::ndarrayToBlitz(maskedImage->getImage()->getArray());
        #ifdef __DEBUG_FINDANDTRACE__
          cout << "::pfs::drp::stella::math::findAndTraceApertures: ccdTImage(60, *) = " << ccdTImage(60, blitz::Range::all()) << endl;
        #endif
        blitz::Array<double, 2> ccdImage(2,2);
        ::pfs::drp::stella::math::Double(ccdTImage, ccdImage);
        #ifdef __DEBUG_FINDANDTRACE__
          cout << "::pfs::drp::stella::math::findAndTraceApertures: ccdImage(60,*) = " << ccdImage(60, blitz::Range::all()) << endl;
        #endif
        blitz::Array<double, 1> D_A1_IndexCol = ::pfs::drp::stella::math::DIndGenArr(ccdImage.cols());
        blitz::Array<double, 1> D_A1_IndexRow = ::pfs::drp::stella::math::DIndGenArr(ccdImage.rows());
        blitz::Array<double, 1> D_A1_X(10);
        blitz::Array<double, 1> D_A1_Y(10);
        blitz::Array<double, 1> D_A1_MeasureErrors(10);
        blitz::Array<double, 1> D_A1_Guess(fiberTraceFunctionFindingControl->nTermsGaussFit);
        blitz::Array<double, 1> D_A1_GaussFit_Coeffs(fiberTraceFunctionFindingControl->nTermsGaussFit);
        blitz::Array<double, 1> D_A1_GaussFit_Coeffs_Bak(fiberTraceFunctionFindingControl->nTermsGaussFit);
        #ifdef __DEBUG_FINDANDTRACE__
          cout << "::pfs::drp::stella::math::findAndTraceApertures: D_A1_IndexCol = " << D_A1_IndexCol << endl;
        #endif
        blitz::Array<int, 1> I_A1_Signal(maskedImage->getWidth());
        blitz::Array<double, 1> D_A1_ApertureCenter(maskedImage->getHeight());
        blitz::Array<int, 1> I_A1_ApertureCenterInd(maskedImage->getHeight());
        blitz::Array<double, 1> D_A1_ApertureCenterIndex(maskedImage->getHeight());
        blitz::Array<int, 1> I_A1_ApertureCenterIndex(maskedImage->getHeight());
        blitz::Array<double, 1> D_A1_ApertureCenterPos(1);
        blitz::Array<double, 1> *P_D_A1_PolyFitCoeffs = new blitz::Array<double, 1>(fiberTraceFunctionFindingControl->fiberTraceFunctionControl.order);
        blitz::Array<int, 1> I_A1_IndSignal(2);
        #ifdef __DEBUG_FINDANDTRACE__
          cout << "::pfs::drp::stella::math::findAndTraceApertures: started" << endl;
          blitz::Array<double, 2> D_A2_PixArrayNew(maskedImage->getHeight(), maskedImage->getWidth());
          D_A2_PixArrayNew = 0.;
        #endif
        blitz::Array<int, 1> I_A1_Ind(1);
        blitz::Array<int, 1> I_A1_Where(1);
        #ifdef __DEBUG_FINDANDTRACE__
          cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunctionFindingControl->signalThreshold = " << fiberTraceFunctionFindingControl->signalThreshold << endl;
        #endif

        /// Set all pixels below fiberTraceFunctionFindingControl->signalThreshold to 0.
        ccdImage = blitz::where(ccdImage < fiberTraceFunctionFindingControl->signalThreshold, 0., ccdImage);
        #ifdef __DEBUG_FINDANDTRACE__
          cout << "::pfs::drp::stella::math::findAndTraceApertures: ccdImage(60,*) = " << ccdImage(60, blitz::Range::all()) << endl;
          string S_TempFileName(DEBUGDIR);
          S_TempFileName += string("FindAndTraceApertures_thresh.fits");
          ::pfs::drp::stella::utils::WriteFits(&ccdImage, S_TempFileName);
        #endif

        int I_MinWidth = int(1.5 * fiberTraceFunctionFindingControl->apertureFWHM);
        if (I_MinWidth < fiberTraceFunctionFindingControl->nTermsGaussFit)
          I_MinWidth = fiberTraceFunctionFindingControl->nTermsGaussFit;
        double D_MaxTimesApertureWidth = 4.;

        /// Search for Apertures
        //  I_LastRowWhereApertureWasFound = 0;
        D_A1_ApertureCenter = 0.;
        D_A1_ApertureCenterIndex = 0.;
        D_A1_ApertureCenterPos = 0.;
        I_A1_ApertureCenterInd = 0;
        I_A1_ApertureCenterIndex = 0;
        for (int i_Row = 0; i_Row < maskedImage->getHeight(); i_Row++){
          I_StartIndex = 0;
          B_ApertureFound = false;
          while (!B_ApertureFound){
            I_A1_Signal = blitz::where(ccdImage(i_Row, blitz::Range::all()) > 0., 1, 0);
            if (!::pfs::drp::stella::math::CountPixGTZero(I_A1_Signal)){
              cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: ::pfs::drp::stella::math::CountPixGTZero(I_A1_Signal=" << I_A1_Signal << ") returned FALSE => Returning FALSE" << endl;
              exit(EXIT_FAILURE);
            }
            I_FirstWideSignal = ::pfs::drp::stella::math::FirstIndexWithValueGEFrom(I_A1_Signal, I_MinWidth, I_StartIndex);
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_FirstWideSignal found at index " << I_FirstWideSignal << ", I_StartIndex = " << I_StartIndex << endl;
            #endif
            if (I_FirstWideSignal < 0){
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": No Aperture found in row " << i_Row << ", trying next row" << endl;
              #endif
              break;
            }
            else{
              I_FirstWideSignalStart = ::pfs::drp::stella::math::LastIndexWithZeroValueBefore(I_A1_Signal, I_FirstWideSignal) + 1;
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "::pfs::drp::stella::math::findAndTraceApertures: while: 1. i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_FirstWideSignalStart = " << I_FirstWideSignalStart << endl;
              #endif

              I_FirstWideSignalEnd = ::pfs::drp::stella::math::FirstIndexWithZeroValueFrom(I_A1_Signal, I_FirstWideSignal) - 1;
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
              #endif

              if (I_FirstWideSignalStart < 0){
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: No start of aperture found -> Going to next Aperture." << endl;
                #endif

                if (I_FirstWideSignalEnd < 0){
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": 1. WARNING: No end of aperture found -> Going to next row." << endl;
                  #endif
                  break;
                }
                else{
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": End of first wide signal found at index " << I_FirstWideSignalEnd << endl;
                  #endif
                  /// Set start index for next run
                  I_StartIndex = I_FirstWideSignalEnd+1;
                }
              }
              else{ /// Fit Gaussian and Trace Aperture
                if (I_FirstWideSignalEnd < 0){
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": 2. WARNING: No end of aperture found -> Going to next row." << endl;
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_Row_Bak = " << I_Row_Bak << endl;
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": B_ApertureFound = " << B_ApertureFound << endl;
                  #endif
                  break;
                }
                else{
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": End of first wide signal found at index " << I_FirstWideSignalEnd << endl;
                  #endif

                  if (I_FirstWideSignalEnd - I_FirstWideSignalStart + 1 > fiberTraceFunctionFindingControl->apertureFWHM * D_MaxTimesApertureWidth){
                    I_FirstWideSignalEnd = I_FirstWideSignalStart + int(D_MaxTimesApertureWidth * fiberTraceFunctionFindingControl->apertureFWHM);
                  }

                  /// Set start index for next run
                  I_StartIndex = I_FirstWideSignalEnd+1;
                }
                I_Length = I_FirstWideSignalEnd - I_FirstWideSignalStart + 1;

                if (fiberTraceFunctionFindingControl->nTermsGaussFit == 0){/// look for maximum only
                  D_A1_ApertureCenter = 0.;
                  B_ApertureFound = true;
                  I_A1_Where.resize(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
                  I_A1_Where = blitz::where(fabs(ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)) - max(ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)))) < 0.00001, 1, 0);
                  if (!::pfs::drp::stella::math::GetIndex(I_A1_Where, I_NInd, I_A1_Ind)){
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: while: ERROR: GetIndex(I_A1_Where=" << I_A1_Where << ") returned FALSE => Returning FALSE" << endl;
                    exit(EXIT_FAILURE);
                  }
                  D_A1_ApertureCenter(i_Row) = I_FirstWideSignalStart + I_A1_Ind(0);
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Aperture found at " << D_A1_ApertureCenter(i_Row) << endl;
                  #endif

                  /// Set signal to zero
                  ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                }
                else{
                  if (I_Length <= fiberTraceFunctionFindingControl->nTermsGaussFit){
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Width of aperture <= " << fiberTraceFunctionFindingControl->nTermsGaussFit << "-> abandoning aperture" << endl;
                    #endif

                    /// Set signal to zero
                    ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                  }
                  else{
                    /// populate Arrays for GaussFit
                    D_A1_X.resize(I_Length);
                    D_A1_Y.resize(I_Length);
                    D_A1_MeasureErrors.resize(I_Length);

                    D_A1_X = D_A1_IndexCol(blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd));
                    D_A1_Y = ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd));
                    D_A1_Y = blitz::where(D_A1_Y < 0.000001, 1., D_A1_Y);
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "::pfs::drp::stella::math::findAndTraceApertures: 1. D_A1_Y set to " << D_A1_Y << endl;
                    #endif
                    D_A1_MeasureErrors = blitz::sqrt(blitz::where(D_A1_Y > 0., D_A1_Y, 1.));

                    /// Guess values for GaussFit
                    D_A1_Guess(0) = max(D_A1_Y);
                    D_A1_Guess(1) = double(I_FirstWideSignalStart) + (double((I_FirstWideSignalEnd - I_FirstWideSignalStart)) / 2.);
                    D_A1_Guess(2) = double(fiberTraceFunctionFindingControl->apertureFWHM) / 2.;

                    D_A1_GaussFit_Coeffs = 0.;
                    blitz::Array<double, 1> D_A1_GaussFit_ECoeffs(D_A1_GaussFit_Coeffs.size());
                    D_A1_GaussFit_ECoeffs = 0.;

                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_X = " << D_A1_X << ", D_A1_Y = " << D_A1_Y << endl;
                    #endif

                    blitz::Array<int, 2> I_A2_Limited(3,2);
                    I_A2_Limited = 1;
                    blitz::Array<double, 2> D_A2_Limits(3,2);
                    D_A2_Limits(0,0) = 0.;/// Peak lower limit
                    D_A2_Limits(0,1) = 2. * D_A1_Guess(0);/// Peak upper limit
                    D_A2_Limits(1,0) = static_cast<double>(I_FirstWideSignalStart);/// Centroid lower limit
                    D_A2_Limits(1,1) = static_cast<double>(I_FirstWideSignalEnd);/// Centroid upper limit
                    D_A2_Limits(2,0) = double(fiberTraceFunctionFindingControl->apertureFWHM) / 4.;/// Sigma lower limit
                    D_A2_Limits(2,1) = double(fiberTraceFunctionFindingControl->apertureFWHM);/// Sigma upper limit
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": 1. starting MPFitGaussLim: D_A1_Guess = " << D_A1_Guess << ", I_A2_Limited = " << I_A2_Limited << ", D_A2_Limits = " << D_A2_Limits << endl;
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
                        cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: GaussFit FAILED -> abandoning aperture" << endl;
                      #endif

                      /// Set start index for next run
                      I_StartIndex = I_FirstWideSignalEnd+1;

                      ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                    }
                    else{
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
                        if (D_A1_GaussFit_Coeffs(0) < fiberTraceFunctionFindingControl->saturationLevel/5.){
                          cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Signal less than 20% of saturation level" << endl;
                        }
                        if (D_A1_GaussFit_Coeffs(0) > fiberTraceFunctionFindingControl->saturationLevel){
                          cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Signal appears to be saturated" << endl;
                        }
                        if ((D_A1_GaussFit_Coeffs(1) < double(I_FirstWideSignalStart) + (double(I_Length)/4.)) || (D_A1_GaussFit_Coeffs(1) > double(I_FirstWideSignalStart) + (double(I_Length) * 3./4.))){
                          cout << "::pfs::drp::stella::math::findAndTraceApertures: while: Warning: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Center of Gaussian far away from middle of signal" << endl;
                        }
                      #endif
                      if ((D_A1_GaussFit_Coeffs(1) < double(I_FirstWideSignalStart)) || (D_A1_GaussFit_Coeffs(1) > double(I_FirstWideSignalEnd))){
                        #ifdef __DEBUG_FINDANDTRACE__
                          cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                        #endif
                        /// Set signal to zero
                        ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;


                        /// Set start index for next run
                        I_StartIndex = I_FirstWideSignalEnd+1;
                      }
                      else{
                        if ((D_A1_GaussFit_Coeffs(2) < fiberTraceFunctionFindingControl->apertureFWHM / 4.) || (D_A1_GaussFit_Coeffs(2) > fiberTraceFunctionFindingControl->apertureFWHM)){
                          #ifdef __DEBUG_FINDANDTRACE__
                            cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: FWHM = " << D_A1_GaussFit_Coeffs(2) << " outside range -> abandoning aperture" << endl;
                          #endif
                          /// Set signal to zero
                          ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                          #ifdef __DEBUG_FINDANDTRACE__
                            cout << "::pfs::drp::stella::math::findAndTraceApertures: while: B_ApertureFound = " << B_ApertureFound << ": 1. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                            cout << "::pfs::drp::stella::math::findAndTraceApertures: while: 1. ccdImage(i_Row = " << i_Row << ", blitz::Range(I_FirstWideSignalStart = " << I_FirstWideSignalStart << ", I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << ")) set to " << ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)) << endl;
                          #endif
                          /// Set start index for next run
                          I_StartIndex = I_FirstWideSignalEnd+1;
                        }
                        else{
                          D_A1_ApertureCenter = 0.;
                          B_ApertureFound = true;
                          //I_LastRowWhereApertureWasFound = i_Row;
                          D_A1_ApertureCenter(i_Row) = D_A1_GaussFit_Coeffs(1);
                          #ifdef __DEBUG_FINDANDTRACE__
                            cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Aperture found at " << D_A1_ApertureCenter(i_Row) << endl;
                          #endif
                          ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
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
              cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Starting to trace aperture" << endl;
    //        #endif
            D_A1_GaussFit_Coeffs_Bak = D_A1_GaussFit_Coeffs;
            I_Row_Bak = i_Row;
            while(B_ApertureFound && (I_ApertureLost < fiberTraceFunctionFindingControl->nLost) && (i_Row < maskedImage->getHeight()-1) && I_Length < fiberTraceFunctionFindingControl->maxLength){
              i_Row++;
              I_Length++;
              if (fiberTraceFunctionFindingControl->nTermsGaussFit == 0){/// look for maximum only
                B_ApertureFound = true;
                D_Max = max(ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)));
                if (D_Max < fiberTraceFunctionFindingControl->signalThreshold){
                  I_ApertureLost++;
                }
                else{
                  I_A1_Where.resize(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
                  I_A1_Where = blitz::where(fabs(ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)) - D_Max) < 0.00001, 1, 0);
                  if (!::pfs::drp::stella::math::GetIndex(I_A1_Where, I_NInd, I_A1_Ind)){
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: ERROR: GetIndex(I_A1_Where=" << I_A1_Where << ") returned FALSE => Returning FALSE" << endl;
                    exit(EXIT_FAILURE);
                  }
                  D_A1_ApertureCenter(i_Row) = I_FirstWideSignalStart + I_A1_Ind(0);
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Aperture found at " << D_A1_ApertureCenter(i_Row) << endl;
                  #endif
                  if (D_A1_ApertureCenter(i_Row) < D_A1_ApertureCenter(i_Row-1)){
                    I_FirstWideSignalStart--;
                    I_FirstWideSignalEnd--;
                  }
                  if (D_A1_ApertureCenter(i_Row) > D_A1_ApertureCenter(i_Row-1)){
                    I_FirstWideSignalStart++;
                    I_FirstWideSignalEnd++;
                  }
                  ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                }
              }
              else{
                I_FirstWideSignalStart = int(D_A1_GaussFit_Coeffs_Bak(1) - 1.6 * D_A1_GaussFit_Coeffs_Bak(2));
                I_FirstWideSignalEnd = int(D_A1_GaussFit_Coeffs_Bak(1) + 1.6 * D_A1_GaussFit_Coeffs_Bak(2)) + 1;
                if (I_FirstWideSignalStart < 0. || I_FirstWideSignalEnd >= maskedImage->getWidth()){
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": start or end of aperture outside CCD -> Aperture lost" << endl;
                  #endif
                  /// Set signal to zero
                  if (I_FirstWideSignalStart < 0)
                    I_FirstWideSignalStart = 0;
                  if (I_FirstWideSignalEnd >= maskedImage->getWidth())
                    I_FirstWideSignalEnd = maskedImage->getWidth() - 1;
                  ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                  I_ApertureLost++;
                }
                else{
                  I_Length = I_FirstWideSignalEnd - I_FirstWideSignalStart + 1;

                  if (I_Length <= fiberTraceFunctionFindingControl->nTermsGaussFit){
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Warning: Width of Aperture <= " << fiberTraceFunctionFindingControl->nTermsGaussFit << " -> Lost Aperture" << endl;
                    #endif
                    /// Set signal to zero
                    ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                    I_ApertureLost++;
                  }
                  else{
                    D_A1_X.resize(I_Length);
                    D_A1_Y.resize(I_Length);
                    D_A1_MeasureErrors.resize(I_Length);

                    D_A1_X = D_A1_IndexCol(blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd));
                    D_A1_Y = ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd));
                    I_A1_IndSignal.resize(D_A1_Y.size());
                    I_A1_IndSignal = blitz::where(D_A1_Y < fiberTraceFunctionFindingControl->signalThreshold, 0, 1);
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "::pfs::drp::stella::math::findAndTraceApertures: I_MinWidth = " << I_MinWidth << ": I_A1_IndSignal = " << I_A1_IndSignal << endl;
                    #endif
                    if (blitz::sum(I_A1_IndSignal) < I_MinWidth){
                      /// Set signal to zero
                      ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                      I_ApertureLost++;
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "::pfs::drp::stella::math::findAndTraceApertures: Signal not wide enough => Aperture lost" << endl;
                      #endif
                    }
                    else{
                      D_A1_Y = blitz::where(D_A1_Y < 0.00000001, 1., D_A1_Y);
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "::pfs::drp::stella::math::findAndTraceApertures: 2. D_A1_Y set to " << D_A1_Y << endl;
                      #endif
                      D_A1_MeasureErrors = sqrt(blitz::where(D_A1_Y > 0., D_A1_Y, 1.));
                      D_A1_Guess = D_A1_GaussFit_Coeffs_Bak;

                      D_A1_GaussFit_Coeffs = 0.;
                      blitz::Array<double, 1> D_A1_GaussFit_ECoeffs(D_A1_GaussFit_Coeffs.size());
                      D_A1_GaussFit_ECoeffs = 0.;

                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_X = " << D_A1_X << ", D_A1_Y = " << D_A1_Y << endl;
                      #endif

                      blitz::Array<int, 2> I_A2_Limited(3,2);
                      I_A2_Limited = 1;
                      blitz::Array<double, 2> D_A2_Limits(3,2);
                      D_A2_Limits(0,0) = 0.;/// Peak lower limit
                      D_A2_Limits(0,1) = 2. * D_A1_Guess(0);/// Peak upper limit
                      D_A2_Limits(1,0) = static_cast<double>(I_FirstWideSignalStart);/// Centroid lower limit
                      D_A2_Limits(1,1) = static_cast<double>(I_FirstWideSignalEnd);/// Centroid upper limit
                      D_A2_Limits(2,0) = fiberTraceFunctionFindingControl->apertureFWHM / 4.;/// Sigma lower limit
                      D_A2_Limits(2,1) = fiberTraceFunctionFindingControl->apertureFWHM;/// Sigma upper limit
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "::pfs::drp::stella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": 2. starting MPFitGaussLim: D_A2_Limits = " << D_A2_Limits << endl;
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
                          cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Warning: GaussFit FAILED" << endl;
                        #endif
                        /// Set signal to zero
                        ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart-1, I_FirstWideSignalEnd+1)) = 0.;

                        I_ApertureLost++;
                      }
                      else{
                        #ifdef __DEBUG_FINDANDTRACE__
                          cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
                          if (D_A1_GaussFit_Coeffs(0) < fiberTraceFunctionFindingControl->saturationLevel/5.){
                            cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Signal less than 20% of saturation level" << endl;
                          }
                          if (D_A1_GaussFit_Coeffs(0) > fiberTraceFunctionFindingControl->saturationLevel){
                            cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Signal appears to be saturated" << endl;
                          }
                        #endif
                        //          if ((D_A1_GaussFit_Coeffs(1) < double(I_FirstWideSignalStart) - (double(I_Length)/4.)) || (D_A1_GaussFit_Coeffs(1) > double(I_FirstWideSignalStart) + (double(I_Length) * 3./4.))){
                        //            cout << "::pfs::drp::stella::math::findAndTraceApertures: Warning: i_Row = " << i_Row << ": Center of Gaussian far away from middle of signal" << endl;
                        //          }
                        if (D_A1_GaussFit_Coeffs(0) < fiberTraceFunctionFindingControl->signalThreshold){
                            #ifdef __DEBUG_FINDANDTRACE__
                              cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: peak = " << D_A1_GaussFit_Coeffs(1) << " lower than signalThreshold -> abandoning aperture" << endl;
                            #endif
                            /// Set signal to zero
                            ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                            #ifdef __DEBUG_FINDANDTRACE__
                              cout << "::pfs::drp::stella::math::findAndTraceApertures: 2. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                            #endif
                            I_ApertureLost++;
                        }
                        else{
                          if ((D_A1_GaussFit_Coeffs(1) < D_A1_GaussFit_Coeffs_Bak(1) - 1.) || (D_A1_GaussFit_Coeffs(1) > D_A1_GaussFit_Coeffs_Bak(1) + 1.)){
                            #ifdef __DEBUG_FINDANDTRACE__
                              cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                            #endif
                            /// Set signal to zero
                            ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;

                            I_ApertureLost++;
                          }
                          else{
                            if ((D_A1_GaussFit_Coeffs(2) < fiberTraceFunctionFindingControl->apertureFWHM / 4.) || (D_A1_GaussFit_Coeffs(2) > fiberTraceFunctionFindingControl->apertureFWHM)){
                              #ifdef __DEBUG_FINDANDTRACE__
                                cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: FWHM = " << D_A1_GaussFit_Coeffs(2) << " outside range -> abandoning aperture" << endl;
                              #endif
                              /// Set signal to zero
                              ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                              #ifdef __DEBUG_FINDANDTRACE__
                                cout << "::pfs::drp::stella::math::findAndTraceApertures: 2. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                              #endif
                              I_ApertureLost++;
                            }
                            else{
                              I_ApertureLost = 0;
                              B_ApertureFound = true;
                              D_A1_ApertureCenter(i_Row) = D_A1_GaussFit_Coeffs(1);
                              #ifdef __DEBUG_FINDANDTRACE__
                                cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Aperture found at " << D_A1_ApertureCenter(i_Row) << endl;
                              #endif
                              D_A1_GaussFit_Coeffs_Bak = D_A1_GaussFit_Coeffs;
                              //I_LastRowWhereApertureWasFound = i_Row;
                            }
                          }/// end else if ((D_A1_GaussFit_Coeffs(1) >= D_A1_Guess(1) - 1.) && (D_A1_GaussFit_Coeffs(1) <= D_A1_Guess(1) + 1.))
                        }/// end else if (D_A1_GaussFit_Coeffs(0) >= signalThreshold
                      }/// end else if (GaussFit(D_A1_X, D_A1_Y, D_A1_GaussFit_Coeffs, S_A1_KeyWords_GaussFit, PP_Args_GaussFit))
                      ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                    }/// end else if (blitz::sum(I_A1_Signal) >= I_MinWidth){
                  }/// end if (I_Length > 3)
                }/// end else if (I_ApertureStart >= 0. && I_ApertureEnd < maskedImage->getWidth())
              }/// end else if GaussFit
            }///end while(B_ApertureFound && (I_ApertureLost < 3) && i_Row < maskedImage->getHeight() - 2))

            /// Fit Polynomial to traced aperture positions
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_ApertureCenter = " << D_A1_ApertureCenter << endl;
    //          ::pfs::drp::stella::utils::WriteArrayToFile(D_A1_ApertureCenter, std::string("xCenters_before_polyfit_ap")+to_string(I_ApertureNumber)+std::string(".dat"), std::string("ascii"));
            #endif
            I_A1_ApertureCenterIndex = blitz::where(D_A1_ApertureCenter > 0., 1, 0);
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_A1_ApertureCenterIndex = " << I_A1_ApertureCenterIndex << endl;
            #endif
            if (!::pfs::drp::stella::math::GetIndex(I_A1_ApertureCenterIndex, I_ApertureLength, I_A1_ApertureCenterInd)){
              cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: ::pfs::drp::stella::math::GetIndex(I_A1_ApertureCenterIndex=" << I_A1_ApertureCenterIndex << ", I_ApertureLength, I_A1_ApertureCenterInd) returned FALSE -> Returning FALSE" << endl;
              exit(EXIT_FAILURE);
            }
            if (I_ApertureLength >= static_cast<int>(fiberTraceFunctionFindingControl->minLength)){
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_IndexRow = " << D_A1_IndexRow << endl;
                cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_A1_ApertureCenterInd = " << I_A1_ApertureCenterInd << endl;
              #endif
              if (!::pfs::drp::stella::math::GetSubArrCopy(D_A1_IndexRow,
                                                    I_A1_ApertureCenterInd,
                                                    D_A1_ApertureCenterIndex)){
                cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: ::pfs::drp::stella::math::GetSubArrCopy(D_A1_IndexRow = " << D_A1_IndexRow << ", I_A1_ApertureCenterInd = " << I_A1_ApertureCenterInd << ", D_A1_ApertureCenterIndex) returned FALSE -> returning FALSE" << endl;
                exit(EXIT_FAILURE);
              }
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_ApertureCenterIndex = " << D_A1_ApertureCenterIndex << endl;
              #endif

              if (!::pfs::drp::stella::math::GetSubArrCopy(D_A1_ApertureCenter,
                                                    I_A1_ApertureCenterInd,
                                                    D_A1_ApertureCenterPos)){
                cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: ::pfs::drp::stella::math::GetSubArrCopy(D_A1_ApertureCenter = " << D_A1_ApertureCenter << ", I_A1_ApertureCenterInd = " << I_A1_ApertureCenterInd << ", D_A1_ApertureCenterIndex) returned FALSE -> returning FALSE" << endl;
                exit(EXIT_FAILURE);
              }
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_ApertureCenterPos = " << D_A1_ApertureCenterPos << endl;
              #endif

              /// Fit Polynomial
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_ApertureCenterIndex = " << D_A1_ApertureCenterIndex << endl;
                cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_ApertureCenterPos = " << D_A1_ApertureCenterPos << endl;
                cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": order = " << fiberTraceFunctionFindingControl->fiberTraceFunctionControl.order << endl;
              #endif
              if (!::pfs::drp::stella::math::PolyFit(D_A1_ApertureCenterIndex,
                                              D_A1_ApertureCenterPos,
                                              fiberTraceFunctionFindingControl->fiberTraceFunctionControl.order,
                                              P_D_A1_PolyFitCoeffs)){
                cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: PolyFit returned FALSE -> Returning FALSE" << endl;
                exit(EXIT_FAILURE);
              }

              fiberTraceFunction.xCenter = D_A1_ApertureCenterPos(int(D_A1_ApertureCenterIndex.size()/2.));
              fiberTraceFunction.yCenter = D_A1_ApertureCenterIndex(int(D_A1_ApertureCenterIndex.size()/2.));
              fiberTraceFunction.yHigh = D_A1_ApertureCenterIndex(int(D_A1_ApertureCenterIndex.size()-1)) - fiberTraceFunction.yCenter;
              fiberTraceFunction.yLow = D_A1_ApertureCenterIndex(0) - fiberTraceFunction.yCenter;
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "::pfs::drp::stella::math::findAndTraceApertures: P_D_A1_PolyFitCoeffs = " << *P_D_A1_PolyFitCoeffs << endl;
              #endif
              fiberTraceFunction.coefficients.resize(P_D_A1_PolyFitCoeffs->size());
              for (int iter=0; iter < static_cast<int>(P_D_A1_PolyFitCoeffs->size()); iter++)
                fiberTraceFunction.coefficients[iter] = (*P_D_A1_PolyFitCoeffs)(iter);
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.xCenter = " << fiberTraceFunction.xCenter << endl;
                cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.yCenter = " << fiberTraceFunction.yCenter << endl;
                cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.yLow = " << fiberTraceFunction.yLow << endl;
                cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.yHigh = " << fiberTraceFunction.yHigh << endl;
    //            cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.coefficients = " << fiberTraceFunction.coefficients << endl;
              #endif
              PTR(const pfsDRPStella::FiberTraceFunction) fiberTraceFunctionPTR(new pfsDRPStella::FiberTraceFunction(fiberTraceFunction));

              I_ApertureNumber++;

/*              if (fiberTraceFunction.yLow != fiberTrace->getFiberTraceFunction()->yLow){
                string message("FindAndTraceApertures: ERROR: fiberTraceFunction.yLow(=");
                message += to_string(fiberTraceFunction.yLow) + string("<< ) != fiberTrace->getFiberTraceFunction()->yLow(=");
                message += to_string(fiberTrace->getFiberTraceFunction()->yLow) + string(")");
                cout << message << endl;
                throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
              }
              if (fiberTraceFunction.yCenter != fiberTrace->getFiberTraceFunction()->yCenter){
                string message("FindAndTraceApertures: ERROR: fiberTraceFunction.yCenter(=");
                message += to_string(fiberTraceFunction.yCenter) + string("<< ) != fiberTrace->getFiberTraceFunction()->yCenter(=");
                message += to_string(fiberTrace->getFiberTraceFunction()->yCenter) + string(")");
                cout << message << endl;
                throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
              }
              if (fiberTraceFunction.yHigh != fiberTrace->getFiberTraceFunction()->yHigh){
                string message("FindAndTraceApertures: ERROR: fiberTraceFunction.yHigh(=");
                message += to_string(fiberTraceFunction.yHigh) + string("<< ) != fiberTrace->getFiberTraceFunction()->yHigh(=");
                message += to_string(fiberTrace->getFiberTraceFunction()->yHigh) + string(")");
                cout << message << endl;
                throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
              }
*/
              blitz::Array<double, 1> D_A1_XCenters_Y = ::pfs::drp::stella::math::DIndGenArr(ccdImage.rows());
              blitz::Array<double, 1> D_A1_XCenters = ::pfs::drp::stella::math::Poly(D_A1_XCenters_Y, *P_D_A1_PolyFitCoeffs);
              blitz::Array<float, 1> F_A1_XCenters(fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1);
              int fiter = 0;
              for (int iter = (fiberTraceFunction.yCenter + fiberTraceFunction.yLow); iter <= (fiberTraceFunction.yCenter + fiberTraceFunction.yHigh); ++iter){
                F_A1_XCenters(fiter) = static_cast<float>(D_A1_XCenters(iter));
                fiter++;
              }
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "::pfs::drp::stella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": F_A1_XCenters = " << F_A1_XCenters << endl;
              #endif
              const std::vector<float> xCenters(F_A1_XCenters.begin(), F_A1_XCenters.end());
              if (xCenters.size() != (fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1)){
                string message("FindAndTraceApertures: iTrace = ");
                message += to_string(fiberTraceSet->getTraces()->size()) + string(": 1. ERROR: xCenters.size(=");
                message += to_string(xCenters.size()) + string(") != (fiberTraceFunction.yHigh(=");
                message += to_string(fiberTraceFunction.yHigh) + string(") - fiberTraceFunction.yLow(=") + to_string(fiberTraceFunction.yLow);
                message += string(") + 1) = ") + to_string(fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1);
                cout << message << endl;
                throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
              }
//              for (int iter = 0; iter < static_cast<int>(D_A1_XCenters_Ap.size()); iter++)
//                xCenters[iter] = static_cast<float>(D_A1_XCenters_Ap(iter));

              #ifdef __DEBUG_FINDANDTRACE__
                for (int iter=0; iter < static_cast<int>(xCenters.size()); ++iter){
                  cout << "FindAndTraceApertures: xCenters[" << iter << "] = " << xCenters[iter] << endl;
                }
              #endif
              ndarray::Array<const float, 1, 1> xCentersND = ndarray::external(xCenters.data(), ndarray::makeVector(int(xCenters.size())));
              PTR(pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>) fiberTrace(new pfsDRPStella::FiberTrace< ImageT, MaskT, VarianceT >(maskedImage, fiberTraceFunctionPTR, xCentersND, I_ApertureNumber));
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
            }
            i_Row = I_Row_Bak - 1;
          }/// end if (B_ApertureFound)
        }/// end for(i_Row = 0; i_Row < maskedImage->getHeight(); i_Row++)

        #ifdef __DEBUG_FINDANDTRACE__
          std::string S_NewFileName = DEBUGDIR;
          S_NewFileName += "FindAndTraceApertures";
          std::string S_FileNameTrace = S_NewFileName + "_out.fits";
          pfs::drp::stella::utils::WriteFits(&ccdImage, S_FileNameTrace);
          blitz::Array<float, 2> F_A2_Trace(2,2);
          for (int i = 0; i < static_cast<int>(fiberTraceSet->size()); i++){
            blitz::Array<ImageT, 2> TImage = utils::ndarrayToBlitz(fiberTraceSet->getFiberTrace(i)->getTrace()->getImage()->getArray());
            pfs::drp::stella::math::Float(TImage, F_A2_Trace);
            S_FileNameTrace = S_NewFileName + "_trace_" + std::to_string(i) + ".fits";
            pfs::drp::stella::utils::WriteFits(&F_A2_Trace, S_FileNameTrace);
          }

        #endif
        fiberTraceSet->sortTracesByXCenter();
        delete(P_D_A1_PolyFitCoeffs);
        return fiberTraceSet;
//      }
//      catch (const std::exception &e){
//        cout << "findAndTraceAperture: ERROR: <" << e.what() << ">" << endl;
//        exit(EXIT_FAILURE);
//      }
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
        xCenters[ndarray::view()] += 0.5;
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
      else if (fiberTraceFunction->fiberTraceFunctionControl.interpolation.compare("CHEBYSHEV") == 0)
      {
        #ifdef __DEBUG_TRACEFUNC__
          cout << "pfs::drp::stella::calculateXCenters: Function = Chebyshev" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
        #endif
        if (!::pfs::drp::stella::math::Chebyshev(D_A1_TempCen,
                             D_YMin,
                             D_YMax,
                             fiberTraceFunctionCoefficients,
                             static_cast<double>(fiberTraceFunction->xCenter)+1.,
                             static_cast<double>(fiberTraceFunction->yCenter)+1.,
                             1.,//double(static_cast<int>(fiberTraceFunction->yCenter) + fiberTraceFunction->yLow + 1),
                             int(ccdHeight),//double(static_cast<int>(fiberTraceFunction->yCenter) + static_cast<int>(fiberTraceFunction->yHigh)+1),
                             static_cast<double>(fiberTraceFunction->fiberTraceFunctionControl.xLow),
                             static_cast<double>(fiberTraceFunction->fiberTraceFunctionControl.xHigh),
                             static_cast<int>(fiberTraceFunction->fiberTraceFunctionControl.order),
                             int(ccdWidth),
                             int(ccdHeight))){
          std::string message("pfs::drp::stella::calculateXCenters: ERROR: Chebyshev(...) returned FALSE!!!");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        D_A1_TempCen -= 1.;
      }
      else if (fiberTraceFunction->fiberTraceFunctionControl.interpolation.compare("LINEAR") == 0){
        #ifdef __DEBUG_TRACEFUNC__
          cout << "pfs::drp::stella::calculateXCenters: Function = spline1" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
        #endif
        if (!::pfs::drp::stella::math::LinearSpline(D_A1_TempCen,
                                              fiberTraceFunctionCoefficients,
                                              static_cast<double>(fiberTraceFunction->xCenter)+1.,
                                              static_cast<double>(fiberTraceFunction->yCenter)+1.,
                                              1.,//double(static_cast<int>(fiberTraceFunction->yCenter) + fiberTraceFunction->yLow + 1),
                                              int(ccdHeight),//double(static_cast<int>(fiberTraceFunction->yCenter) + static_cast<int>(fiberTraceFunction->yHigh)+1),
                                              static_cast<double>(fiberTraceFunction->fiberTraceFunctionControl.xLow),
                                              static_cast<double>(fiberTraceFunction->fiberTraceFunctionControl.xHigh),
                                              static_cast<int>(fiberTraceFunction->fiberTraceFunctionControl.order),
                                              int(ccdWidth),
                                              int(ccdHeight))){
          std::string message("pfs::drp::stella::calculateXCenters: ERROR: LinearSpline(...) returned FALSE!!!");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        D_A1_TempCen -= 1.;
      }
      else if (fiberTraceFunction->fiberTraceFunctionControl.interpolation.compare("CUBIC") == 0){
        #ifdef __DEBUG_TRACEFUNC__
          cout << "pfs::drp::stella::calculateXCenters: Function = spline3" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
        #endif
        if (!::pfs::drp::stella::math::CubicSpline(D_A1_TempCen,
                                             fiberTraceFunctionCoefficients,
                                             static_cast<double>(fiberTraceFunction->xCenter) + 1.,
                                             static_cast<double>(fiberTraceFunction->yCenter) + 1.,
                                             1.,//double(static_cast<int>(fiberTraceFunction->yCenter) + fiberTraceFunction->yLow + 1),
                                             int(ccdHeight),//double(static_cast<int>(fiberTraceFunction->yCenter) + static_cast<int>(fiberTraceFunction->yHigh) + 1),
                                             static_cast<double>(fiberTraceFunction->fiberTraceFunctionControl.xLow),
                                             static_cast<double>(fiberTraceFunction->fiberTraceFunctionControl.xHigh),
                                             static_cast<int>(fiberTraceFunction->fiberTraceFunctionControl.order),
                                             int(ccdWidth),
                                             int(ccdHeight))){
          std::string message("pfs::drp::stella::calculateXCenters: ERROR: CubicSpline(...) returned FALSE!!!");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        D_A1_TempCen -= 1.;
      }
      else /// Polynomial
      {
        #ifdef __DEBUG_TRACEFUNC__
          cout << "pfs::drp::stella::calculateXCenters: Function = Polynomial" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
        #endif
        blitz::Array<double,1> D_A1_XRow(ccdHeight);
        for (int i=0; i < static_cast<int>(ccdHeight); i++){
          D_A1_XRow(i) = double(i);
        }
        #ifdef __DEBUG_TRACEFUNC__
          cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunctionCoefficients = " << fiberTraceFunctionCoefficients << endl;
        #endif
        D_A1_TempCen = pfsDRPStella::math::Poly(D_A1_XRow, fiberTraceFunctionCoefficients);
      }

      /// Check limits
      blitz::Array<double, 1> D_A1_XLow(ccdHeight);
      D_A1_XLow = D_A1_TempCen + fiberTraceFunction->fiberTraceFunctionControl.xLow;
      blitz::Array<double, 1> D_A1_XHigh(ccdHeight);
      D_A1_XHigh = D_A1_TempCen + fiberTraceFunction->fiberTraceFunctionControl.xHigh;
      blitz::Array<int, 1> I_A1_Where(ccdHeight);
      I_A1_Where = blitz::where(D_A1_XLow < -0.5, 0, 1);
      int I_NInd;
      blitz::Array<int, 1> *P_I_A1_WhereInd = ::pfs::drp::stella::math::GetIndex(I_A1_Where, I_NInd);
      D_YMin = (*P_I_A1_WhereInd)(0);
      delete(P_I_A1_WhereInd);
      I_A1_Where = blitz::where(D_A1_XHigh < double(ccdWidth)-0.5, 1, 0);
      P_I_A1_WhereInd = ::pfs::drp::stella::math::GetIndex(I_A1_Where, I_NInd);
      D_YMax = (*P_I_A1_WhereInd)(P_I_A1_WhereInd->size()-1);
      delete(P_I_A1_WhereInd);

      #ifdef __DEBUG_TRACEFUNC__
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

        cout << "pfs::drp::stella::calculateXCenters: D_A1_TempCen set to " << D_A1_TempCen << endl;
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
      #ifdef __DEBUG_TRACEFUNC__
        cout << "pfs::drp::stella::calculateXCenters: yLow set to " << fiberTraceFunction->yLow << endl;
        cout << "pfs::drp::stella::calculateXCenters: yHigh set to " << fiberTraceFunction->yHigh << endl;
      #endif

      /// populate _xCenters
      std::vector<float> xCenters(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1);
      for (int i = static_cast<int>(fiberTraceFunction->yCenter + fiberTraceFunction->yLow); i <= static_cast<int>(fiberTraceFunction->yCenter + fiberTraceFunction->yHigh); i++) {
        xCenters[cIndex] = static_cast<float>(D_A1_TempCen(i));
        cIndex++;
      }
      PTR(const std::vector<float>) pXCenters(new const std::vector<float>(xCenters));
      D_A1_TempCen.resize(0);
      return pXCenters;
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

//  template class FiberTrace<unsigned short>;
//  template class FiberTrace<int>;
  template class pfsDRPStella::FiberTrace<float, unsigned short, float>;
  template class pfsDRPStella::FiberTrace<double, unsigned short, float>;
//  template class pfsDRPStella::FiberTrace<float, unsigned int, float>;
//  template class pfsDRPStella::FiberTrace<double, unsigned int, float>;

//  template class FiberTraceSet<unsigned short>;
//  template class FiberTraceSet<int>;
  template class pfsDRPStella::FiberTraceSet<float, unsigned short, float>;
  template class pfsDRPStella::FiberTraceSet<double, unsigned short, float>;
//  template class pfsDRPStella::FiberTraceSet<float, unsigned int, float>;
//  template class pfsDRPStella::FiberTraceSet<double, unsigned int, float>;

  template PTR(pfsDRPStella::FiberTraceSet<float, unsigned short, float>) pfsDRPStella::math::findAndTraceApertures(PTR(const afwImage::MaskedImage<float, unsigned short, float>) const&, 
                                                                                             PTR(const pfsDRPStella::FiberTraceFunctionFindingControl) const&);
  template PTR(pfsDRPStella::FiberTraceSet<double, unsigned short, float>) pfsDRPStella::math::findAndTraceApertures(PTR(const afwImage::MaskedImage<double, unsigned short, float>) const&, 
                                                                                              PTR(const pfsDRPStella::FiberTraceFunctionFindingControl) const&);
//  template PTR(pfsDRPStella::FiberTraceSet<float, unsigned int, float>) pfsDRPStella::math::findAndTraceApertures(PTR(const afwImage::MaskedImage<float, unsigned int, float>) const&, 
//                                                                                             PTR(const pfsDRPStella::FiberTraceFunctionFindingControl) const&);
//  template PTR(pfsDRPStella::FiberTraceSet<double, unsigned int, float>) pfsDRPStella::math::findAndTraceApertures(PTR(const afwImage::MaskedImage<double, unsigned int, float>) const&, 
//                                                                                              PTR(const pfsDRPStella::FiberTraceFunctionFindingControl) const&);

template const afwImage::MaskedImage<float, unsigned short, float>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::MaskedImage<float, unsigned short, float>) &ptr);
template const afwImage::MaskedImage<double, unsigned short, float>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::MaskedImage<double, unsigned short, float>) &ptr);
template const afwImage::Image<float>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<float>) &ptr);
template const afwImage::Image<double>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<double>) &ptr);
template const afwImage::Image<unsigned short>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<unsigned short>) &ptr);
template const afwImage::Image<unsigned int>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<unsigned int>) &ptr);
template const afwImage::Image<int>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<int>) &ptr);
template const pfsDRPStella::FiberTrace<float, unsigned short, float>* pfsDRPStella::utils::getRawPointer(const PTR(const pfsDRPStella::FiberTrace<float, unsigned short, float>) &ptr);
template const pfsDRPStella::FiberTrace<double, unsigned short, float>* pfsDRPStella::utils::getRawPointer(const PTR(const pfsDRPStella::FiberTrace<double, unsigned short, float>) &ptr);
//template const pfsDRPStella::FiberTrace<float, unsigned int, float>* pfsDRPStella::utils::getRawPointer(const PTR(const pfsDRPStella::FiberTrace<float, unsigned int, float>) &ptr);
//template const pfsDRPStella::FiberTrace<double, unsigned int, float>* pfsDRPStella::utils::getRawPointer(const PTR(const pfsDRPStella::FiberTrace<double, unsigned int, float>) &ptr);

//  template void math::resize(blitz::Array<unsigned int, 1> &arr_in, unsigned int newSize);
//  template void math::resize(blitz::Array<int, 1> &arr_in, unsigned int newSize);
//  template void math::resize(blitz::Array<long, 1> &arr_in, unsigned int newSize);
//  template void math::resize(blitz::Array<float, 1> &arr_in, unsigned int newSize);
//  template void math::resize(blitz::Array<double, 1> &arr_in, unsigned int newSize);

//  template std::vector<unsigned short> sortIndices(const std::vector<unsigned short> &vec_In);
//  template std::vector<unsigned int> sortIndices(const std::vector<unsigned int> &vec_In);
//  template std::vector<int> sortIndices(const std::vector<int> &vec_In);
//  template std::vector<long> sortIndices(const std::vector<long> &vec_In);
//  template std::vector<float> sortIndices(const std::vector<float> &vec_In);
//  template std::vector<double> sortIndices(const std::vector<double> &vec_In);


//  template PTR(afwImage::MaskedImage<float, unsigned short, float>) utils::getShared(afwImage::MaskedImage<float, unsigned short, float> const &maskedImage);
//  template PTR(afwImage::MaskedImage<double, unsigned short, double>) utils::getShared(afwImage::MaskedImage<double, unsigned short, double> const &maskedImage);
