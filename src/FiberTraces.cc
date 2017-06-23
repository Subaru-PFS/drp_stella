#include "pfs/drp/stella/FiberTraces.h"

namespace pfsDRPStella = pfs::drp::stella;

  template<typename ImageT, typename MaskT, typename VarianceT>
  pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    size_t width,
    size_t height,
    size_t iTrace
  ) :
  _overSampledProfileFitXPerSwath(),
  _overSampledProfileFitYPerSwath(),
  _profileFittingInputXPerSwath(),
  _profileFittingInputYPerSwath(),
  _profileFittingInputXMeanPerSwath(),
  _profileFittingInputYMeanPerSwath(),
  _trace(new afwImage::MaskedImage<ImageT, MaskT, VarianceT>(width, height)),
  _profile(new afwImage::Image<double>(width, height)),
  _xCentersMeas(pfsDRPStella::utils::get2DndArray(float(height), float(2))),
  _xCenters(pfsDRPStella::utils::get1DndArray(float(height))),
  _iTrace(iTrace),
  _isTraceSet(false),
  _isProfileSet(false),
  _isFiberTraceProfileFittingControlSet(false),
  _fiberTraceFunction(new pfsDRPStella::FiberTraceFunction),
  _fiberTraceProfileFittingControl(new FiberTraceProfileFittingControl)
  {

  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(PTR(const afwImage::MaskedImage<ImageT, MaskT, VarianceT>) const & maskedImage, ///< desired image width/height
                                                                 PTR(const pfsDRPStella::FiberTraceFunction) const& fiberTraceFunction,
                                                                 size_t iTrace) :
  _overSampledProfileFitXPerSwath(),
  _overSampledProfileFitYPerSwath(),
  _profileFittingInputXPerSwath(),
  _profileFittingInputYPerSwath(),
  _profileFittingInputXMeanPerSwath(),
  _profileFittingInputYMeanPerSwath(),
  _trace(new afwImage::MaskedImage<ImageT, MaskT, VarianceT>(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1, int(fiberTraceFunction->fiberTraceFunctionControl.xHigh - fiberTraceFunction->fiberTraceFunctionControl.xLow + 1))),
  _profile(new afwImage::Image<double>(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1, int(fiberTraceFunction->fiberTraceFunctionControl.xHigh - fiberTraceFunction->fiberTraceFunctionControl.xLow + 1))),
  _iTrace(iTrace),
  _isTraceSet(false),
  _isProfileSet(false),
  _isFiberTraceProfileFittingControlSet(false),
  _fiberTraceFunction(fiberTraceFunction),
  _fiberTraceProfileFittingControl(new FiberTraceProfileFittingControl)
  {
    _xCenters = ::pfs::drp::stella::math::calculateXCenters( fiberTraceFunction,
                                                             maskedImage->getHeight(),
                                                             maskedImage->getWidth() );//new std::vector<const float>(fiberTraceFunction->yHigh - fiberTraceFunction->yLow + 1)),
    _xCentersMeas = pfsDRPStella::utils::get2DndArray(float(_xCenters.getShape()[0]), float(2)),
    createTrace(maskedImage);
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace ) :
  _overSampledProfileFitXPerSwath(),
  _overSampledProfileFitYPerSwath(),
  _profileFittingInputXPerSwath(),
  _profileFittingInputYPerSwath(),
  _profileFittingInputXMeanPerSwath(),
  _profileFittingInputYMeanPerSwath(),
  _trace( new MaskedImageT( *( fiberTrace.getTrace() ) ) ),
  _profile(fiberTrace.getProfile()),
  _xCentersMeas(fiberTrace.getXCentersMeas()),
  _xCenters(fiberTrace.getXCenters()),
  _iTrace(fiberTrace.getITrace()),
  _isTraceSet(fiberTrace.isTraceSet()),
  _isProfileSet(fiberTrace.isProfileSet()),
  _isFiberTraceProfileFittingControlSet(fiberTrace.isFiberTraceProfileFittingControlSet()),
  _fiberTraceFunction(fiberTrace.getFiberTraceFunction()),
  _fiberTraceProfileFittingControl(fiberTrace.getFiberTraceProfileFittingControl())
  {
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT> & fiberTrace, bool const deep) :
  _overSampledProfileFitXPerSwath(),
  _overSampledProfileFitYPerSwath(),
  _profileFittingInputXPerSwath(),
  _profileFittingInputYPerSwath(),
  _profileFittingInputXMeanPerSwath(),
  _profileFittingInputYMeanPerSwath(),
  _trace(fiberTrace.getTrace()),
  _profile(fiberTrace.getProfile()),
  _xCentersMeas(fiberTrace.getXCentersMeas()),
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
      PTR(afwImage::Image<double>) prof(new afwImage::Image<double>(*(fiberTrace.getProfile()), true));
      _profile.reset();
      _profile = prof;
    }
  }

  /** **************************************************************/
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setFiberTraceProfileFittingControl(PTR(FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl){

    /// Check for valid values in fiberTraceFunctionControl
    #ifdef __DEBUG_SETFIBERTRACEPROFILEFITTINGCONTROL__
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->profileInterpolation = <" << fiberTraceProfileFittingControl->profileInterpolation << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->swathWidth = <" << fiberTraceProfileFittingControl->swathWidth << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->telluric = <" << fiberTraceProfileFittingControl->telluric << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->overSample = <" << fiberTraceProfileFittingControl->overSample << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->maxIterSF = <" << fiberTraceProfileFittingControl->maxIterSF << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->maxIterSig = <" << fiberTraceProfileFittingControl->maxIterSig << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->maxIterSky = <" << fiberTraceProfileFittingControl->maxIterSky << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->lambdaSF = <" << fiberTraceProfileFittingControl->lambdaSF << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->lambdaSP = <" << fiberTraceProfileFittingControl->lambdaSP << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->wingSmoothFactor = <" << fiberTraceProfileFittingControl->wingSmoothFactor << ">" << endl;
    #endif

    if (!fiberTraceProfileFittingControl->isClassInvariant()){
      string message("FiberTrace::setFiberTraceProfileFittingControl: ERROR: fiberTraceProfileFittingControl is not ClassInvariant");
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    /// test passed -> copy fiberTraceProfileFittingControl to _fiberTraceProfileFittingControl
    _fiberTraceProfileFittingControl.reset();
    _fiberTraceProfileFittingControl = fiberTraceProfileFittingControl;
    _isFiberTraceProfileFittingControlSet = true;

    return true;
  }

  /** **************************************************************/
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setFiberTraceProfileFittingControl( PTR( const FiberTraceProfileFittingControl ) const& fiberTraceProfileFittingControl ){

    /// Check for valid values in fiberTraceFunctionControl
    #ifdef __DEBUG_SETFIBERTRACEEXTRACTIONCONTROL__
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->profileInterpolation = <" << fiberTraceProfileFittingControl->profileInterpolation << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->swathWidth = <" << fiberTraceProfileFittingControl->swathWidth << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->telluric = <" << fiberTraceProfileFittingControl->telluric << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->overSample = <" << fiberTraceProfileFittingControl->overSample << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->maxIterSF = <" << fiberTraceProfileFittingControl->maxIterSF << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->maxIterSig = <" << fiberTraceProfileFittingControl->maxIterSig << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->maxIterSky = <" << fiberTraceProfileFittingControl->maxIterSky << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->lambdaSF = <" << fiberTraceProfileFittingControl->lambdaSF << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->lambdaSP = <" << fiberTraceProfileFittingControl->lambdaSP << ">" << endl;
      cout << "FiberTrace" << _iTrace << "::setFiberTraceProfileFittingControl: fiberTraceProfileFittingControl->wingSmoothFactor = <" << fiberTraceProfileFittingControl->wingSmoothFactor << ">" << endl;
    #endif

    if (!fiberTraceProfileFittingControl->isClassInvariant()){
      string message("FiberTrace::setFiberTraceProfileFittingControl: ERROR: fiberTraceProfileFittingControl is not ClassInvariant");
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    /// test passed -> copy fiberTraceProfileFittingControl to _fiberTraceProfileFittingControl
    pfsDRPStella::FiberTraceProfileFittingControl ftpfc( *fiberTraceProfileFittingControl );
    _fiberTraceProfileFittingControl.reset();
    _fiberTraceProfileFittingControl = fiberTraceProfileFittingControl->getPointer();
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
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (image->getHeight() != int(_trace->getHeight())){
      string message("FiberTrace.setImage: ERROR: image.getHeight(=");
      message += to_string(image->getHeight()) + string(") != _trace->getHeight(=") + to_string(_trace->getHeight()) + string(")");
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
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (mask->getHeight() != int(_trace->getHeight())){
      string message("FiberTrace.setMask: ERROR: mask.getHeight(=");
      message += to_string(mask->getHeight()) + string(") != _trace->getHeight()(=") + to_string(_trace->getHeight()) + string(")");
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
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (variance->getHeight() != int(_trace->getHeight())){
      string message("FiberTrace.setVariance: ERROR: variance.getHeight(=");
      message += to_string(variance->getHeight()) + string(") != _trace->getHeight(=") + to_string(_trace->getHeight()) + string(")");
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    _trace->getVariance() = variance;

    return true;
  }

  /// Set the _trace of this fiber trace to trace
  template< typename ImageT, typename MaskT, typename VarianceT >
  bool pfsDRPStella::FiberTrace< ImageT, MaskT, VarianceT >::setTrace( PTR( MaskedImageT ) & trace){
    if ( _isTraceSet && ( trace->getHeight() != int( _trace->getHeight() ) ) ){
      string message( "FiberTrace" );
      message += to_string( _iTrace ) + string( "::setTrace: ERROR: trace->getHeight(=" ) + to_string( trace->getHeight() ) + string( ") != _trace->getHeight(=" );
      message += to_string( _trace->getHeight() ) + string( ")" );
      throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
    }
    if ( _isTraceSet && ( trace->getWidth() != int( _trace->getWidth() ) ) ){
      string message ( "FiberTrace");
      message += to_string( _iTrace ) + string( "::setTrace: ERROR: trace->getWidth(=" ) + to_string( trace->getWidth() ) + string( ") != _trace->getWidth(=" );
      message += to_string( _trace->getWidth() ) + string( ")" );
      throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
    }

    _trace.reset();
    _trace = trace;

    _isTraceSet = true;
    return true;
  }

  /// Set the profile image of this fiber trace to profile
  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setProfile( PTR(afwImage::Image<double>) const& profile){
    if (!_isTraceSet){
      string message("FiberTrace");
      message += to_string(_iTrace) + string("::setProfile: ERROR: _trace not set");
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    /// Check input profile size
    if (profile->getWidth() != _trace->getWidth()){
      string message("FiberTrace.setProfile: ERROR: profile->getWidth(=");
      message += to_string(profile->getWidth()) + string(") != _trace->getWidth(=") + to_string(_trace->getWidth()) + string(")");
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (profile->getHeight() != _trace->getHeight()){
      string message("FiberTrace.setProfile: ERROR: profile->getHeight(=");
      message += to_string(profile->getHeight()) + string(") != _trace->getHeight(=") + to_string(_trace->getHeight()) + string(")");
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    _profile.reset();
    _profile = profile;

    _isProfileSet = true;
    return true;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTrace< ImageT, MaskT, VarianceT >::setFiberTraceFunction( PTR( const FiberTraceFunction ) fiberTraceFunction ){
    pfsDRPStella::FiberTraceFunction ftf( *fiberTraceFunction );
    _fiberTraceFunction.reset();
    _fiberTraceFunction = ftf.getPointer();
    return true;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::extractSum()
  {
    PTR( pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT> ) spectrum( new pfsDRPStella::Spectrum< ImageT, MaskT, VarianceT, VarianceT >( _trace->getHeight(), _iTrace ) );
    ndarray::Array<ImageT, 1, 1> spec = ndarray::allocate(_trace->getHeight());
    afwImage::Mask<MaskT> mask(_trace->getHeight(), 1);
    ndarray::Array<VarianceT, 1, 1> var = ndarray::allocate(_trace->getHeight());
    auto specIt = spec.begin();
    auto varIt = var.begin();
    for (int i = 0; i < _trace->getHeight(); ++i, ++specIt, ++varIt){
      *specIt = sum(_trace->getImage()->getArray()[i]);
      *varIt = sum(_trace->getVariance()->getArray()[i]);
    }
    if ( !spectrum->setSpectrum( spec ) ){
      string message("FiberTrace");
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setSpectrum(spec) returned FALSE";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (!spectrum->setVariance(var)){
      string message("FiberTrace");
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setVariance(var) returned FALSE";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (!spectrum->setMask(mask)){
      string message("FiberTrace");
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setMask(mask) returned FALSE";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if ( !spectrum->setNCCDRows( getHeight() ) ){
      string message( "FiberTrace" );
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setNCCDRows(getHeight()) returned FALSE";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if ( !spectrum->setYLow( _fiberTraceFunction->yCenter + _fiberTraceFunction->yLow ) ){
      string message( "FiberTrace" );
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setYLow(_fiberTraceFunction->yCenter + _fiberTraceFunction->yLow) returned FALSE";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if ( !spectrum->setYHigh( _fiberTraceFunction->yCenter + _fiberTraceFunction->yHigh ) ){
      string message( "FiberTrace" );
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setYHigh(_fiberTraceFunction->yCenter + _fiberTraceFunction->yHigh) returned FALSE";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return spectrum;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR( pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT> ) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::extractFromProfile()
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
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (_trace->getHeight() != _profile->getHeight()){
      std::string message("FiberTrace.extractFromProfile: ERROR: _trace->getHeight(=");
      message += std::to_string(_trace->getHeight());
      message += std::string(") != _profile.getHeight(=");
      message += std::to_string(_profile->getHeight());
      message += std::string(")");
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    ndarray::Array<VarianceT, 2, 1> variance = math::where(_trace->getVariance()->getArray(),
                                                           "<",
                                                           VarianceT(1.0),
                                                           VarianceT(1.0),
                                                           _trace->getVariance()->getArray());

    ndarray::Array<MaskT, 2, 1> US_A2_MaskArray = math::where(_trace->getMask()->getArray(),
                                                              "==",
                                                               MaskT(0),
                                                               MaskT(1),
                                                               MaskT(0));
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "US_A2_MaskArray = " << US_A2_MaskArray << endl;
    #endif

    ndarray::Array<ImageT, 1, 1> D_A1_SP = ndarray::allocate(_trace->getHeight());
    D_A1_SP.deep() = 0.;
    ndarray::Array<ImageT, 1, 1> D_A1_Sky = ndarray::allocate(_trace->getHeight());
    D_A1_Sky.deep() = 0.;
    vector<string> S_A1_Args_Fit(5);
    vector<void *> P_Args_Fit(5);

    ndarray::Array<ImageT, 2, 1> D_A2_ErrArray = ndarray::allocate(_trace->getImage()->getArray().getShape());
    for (int i_row = 0; i_row < _trace->getHeight(); ++i_row){
      for (int i_col = 0; i_col < _trace->getWidth(); ++i_col){
        D_A2_ErrArray[i_row][i_col] = ImageT(sqrt(variance[i_row][i_col]));
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
    PTR(ndarray::Array<ImageT, 2, 1>) P_D_A2_ErrArray(new ndarray::Array<ImageT, 2, 1>(D_A2_ErrArray));
    P_Args_Fit[0] = &P_D_A2_ErrArray;
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: *P_D_A2_ErrArray = " << *P_D_A2_ErrArray << endl;
    #endif

    S_A1_Args_Fit[1] = "MASK_INOUT";
    PTR(ndarray::Array<unsigned short, 2, 1>) P_US_A2_MaskArray(new ndarray::Array<unsigned short, 2, 1>(US_A2_MaskArray));
    P_Args_Fit[1] = &P_US_A2_MaskArray;
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "P_E_A2_MaskArray = " << *P_US_A2_MaskArray << endl;
    #endif

    S_A1_Args_Fit[2] = "SIGMA_OUT";
    ndarray::Array<ImageT, 2, 1> D_A2_Sigma_Fit = ndarray::allocate(_trace->getHeight(), 2);
    PTR(ndarray::Array<ImageT, 2, 1>) P_D_A2_Sigma_Fit(new ndarray::Array<ImageT, 2, 1>(D_A2_Sigma_Fit));
    P_Args_Fit[2] = &P_D_A2_Sigma_Fit;

    /// Disallow background and spectrum to go below Zero as it is not physical
    S_A1_Args_Fit[3] = "ALLOW_SKY_LT_ZERO";
    int allowSkyLtZero = 0;
    P_Args_Fit[3] = &allowSkyLtZero;

    S_A1_Args_Fit[4] = "ALLOW_SPEC_LT_ZERO";
    int allowSpecLtZero = 0;
    P_Args_Fit[4] = &allowSpecLtZero;

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
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    #ifdef __DEBUG_MkSLITFUNC_FILES__
      string S_MaskFinalOut = "Mask_Final" + S_SF_DebugFilesSuffix + ".fits";
      ::pfs::drp::stella::utils::WriteFits(&I_A2_MaskArray, S_MaskFinalOut);

      S_MaskFinalOut = "D_A2_CCD_Ap" + CS_SF_DebugFilesSuffix + ".fits";
      ::pfs::drp::stella::utils::WriteFits(&D_A2_CCDArray, S_MaskFinalOut);
    #endif

    PTR( pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT> ) spectrum( new pfsDRPStella::Spectrum< ImageT, MaskT, VarianceT, VarianceT >( _trace->getHeight() ) );
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: D_A1_SP = " << D_A1_SP << endl;
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: D_A2_Sigma_Fit = " << D_A2_Sigma_Fit << endl;
      cout << "P_D_A2_Sigma_Fit = " << *P_D_A2_Sigma_Fit << endl;
    #endif
    ndarray::Array< ImageT, 1, 1 > spectrumSpecOut = ndarray::allocate(_trace->getHeight());
    ndarray::Array< VarianceT, 1, 1 > spectrumVarOut = ndarray::allocate(_trace->getHeight());
    ndarray::Array< ImageT, 1, 1 > backgroundSpecOut = ndarray::allocate(_trace->getHeight());
    ndarray::Array< VarianceT, 1, 1 > backgroundVarOut = ndarray::allocate(_trace->getHeight());
    for (int i = 0; i < _trace->getHeight(); i++) {
      spectrumSpecOut[ i ] = ImageT( D_A1_SP[ i ] );
      spectrumVarOut[ i ] = VarianceT( pow( D_A2_Sigma_Fit[ i ][ 0 ], 2) );
      backgroundSpecOut[ i ] = ImageT( D_A1_Sky[ i ] );
      backgroundVarOut[ i ] = VarianceT( pow( D_A2_Sigma_Fit[ i ][ 1 ], 2 ) );
    }
    if (!spectrum->setSpectrum(spectrumSpecOut)){
      string message( "FiberTrace" );
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setSpectrum(spectrumSpecOut) returned FALSE";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (!spectrum->setVariance(spectrumVarOut)){
      string message( "FiberTrace" );
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setVariance(spectrumVarOut) returned FALSE";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (!spectrum->setSky(backgroundSpecOut)){
      string message( "FiberTrace" );
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setSky(backgroundSpecOut) returned FALSE";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    /// TODO: Add background(sky)Variance to Spectrum class
//    if (!background->setVariance(backgroundVarOut)){
//      string message( "FiberTrace" );
//      message += to_string(_iTrace) + "::extractSum: ERROR: background->setVariance(backgroundVarOut) returned FALSE";
//      cout << message << endl;
//      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
//    }
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: for loop finished" << endl;
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: starting spectrum->setNCCDRows( getHeight()=" << getHeight() << " )" << endl;
    #endif
    if ( !spectrum->setNCCDRows( getHeight() ) ){
      string message( "FiberTrace" );
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setNCCDRows(getHeight()) returned FALSE";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: starting spectrum->getNCCDRows() = " << spectrum->getNCCDRows() << endl;
      cout << "FiberTrace" << _iTrace << "::extractFromProfile: starting spectrum->setYLow( _fiberTraceFunction->yCenter + _fiberTraceFunction->yLow=" << _fiberTraceFunction->yCenter + _fiberTraceFunction->yLow << " )" << endl;
    #endif
    if ( !spectrum->setYLow( _fiberTraceFunction->yCenter + _fiberTraceFunction->yLow ) ){
      string message( "FiberTrace" );
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setYLow(_fiberTraceFunction->yCenter + _fiberTraceFunction->yLow) returned FALSE";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if ( !spectrum->setYHigh( _fiberTraceFunction->yCenter + _fiberTraceFunction->yHigh ) ){
      string message( "FiberTrace" );
      message += to_string(_iTrace) + "::extractSum: ERROR: spectrum->setYHigh(_fiberTraceFunction->yCenter + _fiberTraceFunction->yHigh) returned FALSE";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    spectrum->setITrace( _iTrace );
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
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::createTrace( const PTR(const MaskedImageT) &maskedImage ){
    size_t oldTraceHeight = 0;
    size_t oldTraceWidth = 0;
    if (_isTraceSet){
      oldTraceWidth = getWidth();
      oldTraceHeight = getHeight();
    }
    if (_xCenters.getShape()[0] >= 0 &&
        static_cast<size_t>(_xCenters.getShape()[0]) != (_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1)) {
      string message("FiberTrace");
      message += to_string(_iTrace) + string("::createTrace: ERROR: _xCenters.getShape()[0]=") + to_string(_xCenters.getShape()[0]);
      message += string(" != (_fiberTraceFunction->yHigh(=") + to_string(_fiberTraceFunction->yHigh) + string(") - _fiberTraceFunction->yLow(=");
      message += to_string(_fiberTraceFunction->yLow) + string(") + 1)=") + to_string(_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    ndarray::Array<size_t, 2, 1> minCenMax = pfsDRPStella::math::calcMinCenMax( _xCenters,
                                                                                _fiberTraceFunction->fiberTraceFunctionControl.xHigh,
                                                                                _fiberTraceFunction->fiberTraceFunctionControl.xLow,
                                                                                _fiberTraceFunction->fiberTraceFunctionControl.nPixCutLeft,
                                                                                _fiberTraceFunction->fiberTraceFunctionControl.nPixCutRight );
    #ifdef __DEBUG_CREATEFIBERTRACE__
      cout << "FiberTrace" << _iTrace << "::CreateFiberTrace: minCenMax = " << minCenMax << endl;
    #endif

    if ((_isTraceSet) && (static_cast<size_t>(_trace->getHeight()) != (_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1))){
      string message("FiberTrace ");
      message += to_string(_iTrace) + string("::createTrace: ERROR: _trace.getHeight(=") + to_string(_trace->getHeight()) + string(") != (_fiberTraceFunction->yHigh(=");
      message += to_string(_fiberTraceFunction->yHigh) + string(") - _fiberTraceFunction->yLow(=") + to_string(_fiberTraceFunction->yLow) + string(") + 1) = ");
      message += to_string(_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    if (oldTraceHeight > 0){
      if (oldTraceHeight != getHeight()){
        string message("FiberTrace ");
        message += to_string(_iTrace) + string("::createTrace: ERROR: oldTraceHeight(=") + to_string(oldTraceHeight) + string(") != getHeight(=") + to_string(getHeight());
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (oldTraceWidth != getWidth()){
        string message("FiberTrace ");
        message += to_string(_iTrace) + string("::createTrace: ERROR: oldTraceWidth(=") + to_string(oldTraceWidth) + string(") != getWidth(=") + to_string(getWidth());
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    }

    _trace.reset(new MaskedImageT(int(minCenMax[0][2] - minCenMax[0][0] + 1), _fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1));// minCenMax.rows());

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
    if (static_cast<size_t>(_trace->getHeight()) !=
        (_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1)) {
      string message("FiberTrace ");
      message += to_string(_iTrace) + string("::createTrace: 2. ERROR: _trace.getHeight(=") + to_string(_trace->getHeight()) + string(") != (_fiberTraceFunction->yHigh(=");
      message += to_string(_fiberTraceFunction->yHigh) + string(") - _fiberTraceFunction->yLow(=") + to_string(_fiberTraceFunction->yLow) + string(") + 1) = ");
      message += to_string(_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (static_cast<size_t>(_xCenters.getShape()[0]) !=
        (_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1)) {
      string message("FiberTrace");
      message += to_string(_iTrace) + string("::createTrace: 2. ERROR: xCenters.getShape()[0]=") + to_string(_xCenters.getShape()[0]);
      message += string(") != (_fiberTraceFunction->yHigh(=") + to_string(_fiberTraceFunction->yHigh) + string(") - _fiberTraceFunction->yLow(=");
      message += to_string(_fiberTraceFunction->yLow) + string(") + 1)=") + to_string(_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    #ifdef __DEBUG_CREATETRACE__
      cout << "FiberTrace::createFiberTrace: _trace set to " << _trace->getImage()->getArray() << endl;
    #endif
    if (!_isProfileSet){
      _profile.reset(new afwImage::Image<double>(_trace->getWidth(), _trace->getHeight()));
    }
    _isTraceSet = true;
    return true;
  }

  /// Return shared pointer to an image containing the reconstructed 2D spectrum of the FiberTrace
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR( afwImage::Image< ImageT > ) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::getReconstructed2DSpectrum(const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT> & spectrum) const{
    ndarray::Array<ImageT, 2, 1> F_A2_Rec = ndarray::allocate(_trace->getHeight(), _trace->getWidth());
    auto itRec = F_A2_Rec.begin();
    auto itSpec = spectrum.getSpectrum().begin();
    for (auto itProf = _profile->getArray().begin(); itProf != _profile->getArray().end(); ++itProf, ++itRec, ++itSpec)
      (*itRec) = (*itProf) * (*itSpec);
    PTR( afwImage::Image< ImageT > ) imagePtr( new afwImage::Image< ImageT >( F_A2_Rec ) );
    return imagePtr;
  }

  /// Return shared pointer to an image containing the reconstructed background of the FiberTrace
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(afwImage::Image<ImageT>) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::getReconstructedBackground( const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT> & background) const{
    ndarray::Array<ImageT, 1, 1> oneRow = ndarray::allocate(_trace->getWidth());
    oneRow.deep() = 1.;
    ndarray::Array<ImageT, 2, 1> F_A2_Rec = ndarray::allocate(_trace->getHeight(),
                                                              _trace->getWidth());
    F_A2_Rec.asEigen() = background.getSky().asEigen() * oneRow.asEigen().transpose();
    PTR(afwImage::Image<ImageT>) imagePtr( new afwImage::Image<ImageT>( F_A2_Rec ) );
    return imagePtr;
  }

  /// Return shared pointer to an image containing the reconstructed 2D spectrum + background of the FiberTrace
  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(afwImage::Image<ImageT>) pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::getReconstructed2DSpectrum(const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT> & spectrum,
                                                                                                             const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT> & background) const
  {
    PTR(afwImage::Image<ImageT>) imageSpectrum = getReconstructed2DSpectrum(spectrum);
    PTR(afwImage::Image<ImageT>) imageBackground = getReconstructed2DSpectrum(background);
    *imageSpectrum += *imageBackground;
    return imageSpectrum;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  ndarray::Array<size_t, 2, 1> pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::calcSwathBoundY(const size_t swathWidth) const{
    size_t nSwaths = 0;

    size_t swathWidth_mutable = swathWidth;
    if (swathWidth_mutable > _trace->getHeight()){
      swathWidth_mutable = _trace->getHeight();
      #ifdef __DEBUG_CALCSWATHBOUNDY__
        cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: KeyWord_Set(SWATH_WIDTH): swathWidth_mutable too large: swathWidth_mutable set to " << swathWidth_mutable << endl;
      #endif
    }
    nSwaths = round(float(_trace->getHeight()) / float(swathWidth));

    size_t binHeight = _trace->getHeight() / nSwaths;
    if (nSwaths > 1)
      nSwaths = (2 * nSwaths) - 1;

    #ifdef __DEBUG_CALCSWATHBOUNDY__
      cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: fiberTraceNumber = " << _iTrace << endl;
      cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: _trace->getHeight() = " << _trace->getHeight() << endl;
      cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: binHeight = " << binHeight << endl;
      cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: nSwaths set to " << nSwaths << endl;
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
        swathBoundYTemp[iSwath][0] = swathBoundYTemp[iSwath-1][0] + size_t(double(binHeight) / 2.);
      else
        swathBoundYTemp[iSwath][0] = swathBoundYTemp[iSwath-2][1] + 1;
      swathBoundYTemp[iSwath][1] = swathBoundYTemp[iSwath][0] + I_BinHeight_Temp;
      if (swathBoundYTemp[iSwath][1] >= _trace->getHeight()-1){
        nSwaths = iSwath + 1;
      }
      if (iSwath == (nSwaths-1)){
        swathBoundYTemp[iSwath][1] = _trace->getHeight()-1;
      }
    }

    ndarray::Array<size_t, 2, 1> swathBoundY = ndarray::allocate(int(nSwaths), 2);
    #ifdef __DEBUG_CALCSWATHBOUNDY__
      ndarray::Array<double, 2, 1>::Index shape = swathBoundY.getShape();
      cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: shape = " << shape << endl;
    #endif
    swathBoundY[0][0] = 0;
    #ifdef __DEBUG_CALCSWATHBOUNDY__
      cout << "FiberTrace" << _iTrace << "::calcSwathBoundY: 1. swathBoundY[0][0] set to " << swathBoundY[0][0] << endl;
    #endif
    I_BinHeight_Temp = binHeight;
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
    swathBoundY[ nSwaths - 1][ 1 ] = _trace->getHeight() - 1;
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
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (!_isFiberTraceProfileFittingControlSet){
      string message("FiberTrace ");
      message += to_string(_iTrace) + "::calcProfile: ERROR: _fiberTraceProfileFittingControl is not set";
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
    ndarray::Array<double, 3, 2> slitFuncsSwaths = ndarray::allocate(nPixArr[0], _trace->getWidth(), int(nSwaths-1));
    ndarray::Array<double, 2, 1> lastSlitFuncSwath = ndarray::allocate(nPixArr[nPixArr.getShape()[0] - 1], _trace->getWidth());
    #ifdef __DEBUG_CALCPROFILE__
      cout << "slitFuncsSwaths.getShape() = " << slitFuncsSwaths.getShape() << endl;
      cout << "slitFuncsSwaths.getShape()[0] = " << slitFuncsSwaths.getShape()[0] << ", slitFuncsSwaths.getShape()[1] = " << slitFuncsSwaths.getShape()[1] << ", slitFuncsSwaths.getShape()[2] = " << slitFuncsSwaths.getShape()[2] << endl;
      cout << "slitFuncsSwaths[view(0)()(0) = " << ndarray::Array<double, 1, 0>(slitFuncsSwaths[ndarray::view(0)()(0)]) << endl;
      cout << "slitFuncsSwaths[view(0)(0,9)(0) = " << ndarray::Array<double, 1, 0>(slitFuncsSwaths[ndarray::view(0)(0,slitFuncsSwaths.getShape()[1])(0)]) << endl;
    #endif
    #ifdef __DEBUG_CALCPROFILE__
      ndarray::Array<double, 3, 1>::Index shapeSFsSwaths = slitFuncsSwaths.getShape();
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
      const ndarray::Array<ImageT const, 2, 1> imageSwath = ndarray::copy(_trace->getImage()->getArray()[ndarray::view(iMin, iMax)()]);
      #ifdef __DEBUG_CALCPROFILE__
        cout << "FiberTrace::calcProfile: trace " << _iTrace << ": swath " << iSwath << ": imageSwath = " << imageSwath << endl;
      #endif
      const ndarray::Array<MaskT const, 2, 1> maskSwath = ndarray::copy(_trace->getMask()->getArray()[ndarray::view(iMin, iMax)()]);
      const ndarray::Array<VarianceT const, 2, 1> varianceSwath = ndarray::copy(_trace->getVariance()->getArray()[ndarray::view(iMin, iMax)()]);
      const ndarray::Array< float const, 1, 1> xCentersSwath = ndarray::copy(_xCenters[ndarray::view(iMin, iMax)]);

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
    double D_Weight_Bin0 = 0.;
    double D_Weight_Bin1 = 0.;
    double D_RowSum;
    for (size_t iSwath = 0; iSwath < nSwaths - 1; iSwath++){
      for (size_t i_row = 0; i_row < nPixArr[iSwath]; i_row++){
        D_RowSum = ndarray::sum(ndarray::Array<double, 1, 0>(slitFuncsSwaths[ndarray::view(static_cast<int>(i_row))()(iSwath)]));
        if (std::fabs(D_RowSum) > 0.00000000000000001){
          for (int iPix = 0; iPix < slitFuncsSwaths.getShape()[1]; iPix++){
            slitFuncsSwaths[i_row][iPix][iSwath] = slitFuncsSwaths[i_row][iPix][iSwath] / D_RowSum;
          }
          #ifdef __DEBUG_CALCPROFILE__
            cout << "slitFuncsSwaths(" << i_row << ", *, " << iSwath << ") = " << ndarray::Array<double, 1, 0>(slitFuncsSwaths[ndarray::view(static_cast<int>(i_row))()(iSwath)]) << endl;
            D_RowSum = ndarray::sum(ndarray::Array<double, 1, 0>(slitFuncsSwaths[ndarray::view(static_cast<int>(i_row))()(iSwath)]));
            cout << "i_row = " << i_row << ": iSwath = " << iSwath << ": D_RowSum = " << D_RowSum << endl;
          #endif
        }
      }
    }
    for (size_t i_row = 0; i_row < nPixArr[nPixArr.getShape()[0]-1]; i_row++){
      D_RowSum = ndarray::sum(lastSlitFuncSwath[ndarray::view(static_cast<int>(i_row))()]);
      if (std::fabs(D_RowSum) > 0.00000000000000001){
        lastSlitFuncSwath[ndarray::view(static_cast<int>(i_row))()] = lastSlitFuncSwath[ndarray::view(static_cast<int>(i_row))()] / D_RowSum;
        #ifdef __DEBUG_CALCPROFILE__
          cout << "lastSlitFuncSwath(" << i_row << ", *) = " << lastSlitFuncSwath[ndarray::view(static_cast<int>(i_row))()] << endl;
          cout << "i_row = " << i_row << ": D_RowSum = " << D_RowSum << endl;
        #endif
      }
    }
    #ifdef __DEBUG_CALCPROFILE__
      cout << "swathBoundsY.getShape() = " << swathBoundsY.getShape() << ", nSwaths = " << nSwaths << endl;
    #endif
    int iRowSwath = 0;
    for (size_t i_row = 0; i_row < static_cast<size_t>(_trace->getHeight()); ++i_row){
      iRowSwath = i_row - swathBoundsY[I_Bin][0];
      #ifdef __DEBUG_CALCPROFILE__
        cout << "i_row = " << i_row << ", I_Bin = " << I_Bin << ", iRowSwath = " << iRowSwath << endl;
      #endif
      if ((I_Bin == 0) && (i_row < swathBoundsY[1][0])){
        #ifdef __DEBUG_CALCPROFILE__
          cout << "I_Bin=" << I_Bin << " == 0 && i_row=" << i_row << " < swathBoundsY[1][0]=" << swathBoundsY[1][0] << endl;
        #endif
        _profile->getArray()[ndarray::view(static_cast<int>(i_row))()] = ndarray::Array<double, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()(0)]);
      }
      else if ((I_Bin == nSwaths-1) && (i_row >= swathBoundsY[I_Bin-1][1])){// && (i_row > (I_A2_IBound(I_Bin-1, 1) - (I_A2_IBound(0,1) / 2.)))){
        #ifdef __DEBUG_CALCPROFILE__
          cout << "I_Bin=" << I_Bin << " == nSwaths-1=" << nSwaths-1 << " && i_row=" << i_row << " >= swathBoundsY[I_Bin-1=" << I_Bin - 1 << "][0]=" << swathBoundsY[I_Bin-1][0] << endl;
        #endif
        _profile->getArray()[ndarray::view(static_cast<int>(i_row))()] = lastSlitFuncSwath[ndarray::view(iRowSwath)()];
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
            cout << "FiberTrace" << _iTrace << "::calcProfile: slitFuncsSwaths(iRowSwath, *, I_Bin) = " << ndarray::Array<double, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)]) << ", lastSlitFuncSwath[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0])=" << int(i_row - swathBoundsY[I_Bin+1][0]) << ")] = " << lastSlitFuncSwath[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()] << endl;
            cout << "FiberTrace" << _iTrace << "::calcProfile: _profile->getArray().getShape() = " << _profile->getArray().getShape() << endl;
            cout << "FiberTrace" << _iTrace << "::calcProfile: slitFuncsSwath.getShape() = " << slitFuncsSwaths.getShape() << endl;
            cout << "FiberTrace" << _iTrace << "::calcProfile: lastSlitFuncSwath.getShape() = " << lastSlitFuncSwath.getShape() << endl;
            cout << "FiberTrace" << _iTrace << "::calcProfile: i_row = " << i_row << ", iRowSwath = " << iRowSwath << ", I_Bin = " << I_Bin << ", swathBoundsY[I_Bin+1][0] = " << swathBoundsY[I_Bin+1][0] << ", i_row - swathBoundsY[I_Bin+1][0] = " << i_row - swathBoundsY[I_Bin+1][0] << endl;
            cout << "FiberTrace" << _iTrace << "::calcProfile: _profile->getArray()[ndarray::view(i_row)()] = " << ndarray::Array<double, 1, 1>(_profile->getArray()[ndarray::view(i_row)()]) << endl;
            cout << "FiberTrace" << _iTrace << "::calcProfile: slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)] = " << ndarray::Array<double, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)]) << endl;
            cout << "FiberTrace" << _iTrace << "::calcProfile: ndarray::Array<double, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)]).getShape() = " << ndarray::Array<double, 1, 1>(slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)]).getShape() << endl;
            cout << "FiberTrace" << _iTrace << "::calcProfile: D_Weight_Bin0 = " << D_Weight_Bin0 << ", D_Weight_Bin1 = " << D_Weight_Bin1 << endl;
            cout << "FiberTrace" << _iTrace << "::calcProfile: lastSlitFuncSwath[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()] = " << ndarray::Array<double, 1, 1>(lastSlitFuncSwath[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()]) << endl;
          #endif
          _profile->getArray()[ndarray::view(static_cast<int>(i_row))()] = (ndarray::Array<double, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)]) * D_Weight_Bin0) + (lastSlitFuncSwath[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()] * D_Weight_Bin1);
          #ifdef __DEBUG_CALCPROFILE__
            cout << "FiberTrace" << _iTrace << "::calcProfile: _profile->getArray()[ndarray::view(i_row)()] set to " << _profile->getArray()[ndarray::view(i_row)()] << endl;
          #endif
        }
        else{
          #ifdef __DEBUG_CALCPROFILE__
            cout << "FiberTrace" << _iTrace << "::calcProfile: slitFuncsSwaths(iRowSwath, *, I_Bin) = " << ndarray::Array<double, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)]) << ", slitFuncsSwaths(int(i_row - swathBoundsY[I_Bin+1][0])=" << int(i_row - swathBoundsY[I_Bin+1][0]) << ", *, I_Bin+1) = " << ndarray::Array<double, 1, 0>(slitFuncsSwaths[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()(I_Bin+1)]) << endl;
          #endif
          _profile->getArray()[ndarray::view(i_row)()] = (slitFuncsSwaths[ndarray::view(iRowSwath)()(I_Bin)] * D_Weight_Bin0) + (slitFuncsSwaths[ndarray::view(int(i_row - swathBoundsY[I_Bin+1][0]))()(I_Bin+1)] * D_Weight_Bin1);
        }
        int int_i_row = static_cast<int>(i_row);
        double dSumSFRow = ndarray::sum(_profile->getArray()[ndarray::view(int_i_row)()]);
        #ifdef __DEBUG_CALCPROFILE__
          cout << "FiberTrace" << _iTrace << "::calcProfile: i_row = " << i_row << ": I_Bin = " << I_Bin << ": dSumSFRow = " << dSumSFRow << endl;
          cout << "FiberTrace" << _iTrace << "::calcProfile: i_row = " << i_row << ": I_Bin = " << I_Bin << ": _profile->getArray().getShape() = " << _profile->getArray().getShape() << endl;
        #endif
        if (std::fabs(dSumSFRow) >= 0.000001){
          #ifdef __DEBUG_CALCPROFILE__
            cout << "FiberTrace" << _iTrace << "::calcProfile: i_row = " << i_row << ": I_Bin = " << I_Bin << ": normalizing _profile.getArray()[i_row = " << i_row << ", *]" << endl;
          #endif
          _profile->getArray()[ndarray::view(int_i_row)()] = _profile->getArray()[ndarray::view(int_i_row)()] / dSumSFRow;
        }
        #ifdef __DEBUG_CALCPROFILE__
          cout << "FiberTrace" << _iTrace << "::calcProfile: i_row = " << i_row << ": I_Bin = " << I_Bin << ": _profile->getArray()(" << i_row << ", *) set to " << _profile->getArray()[ndarray::view(int_i_row)()] << endl;
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
      cout << "FiberTrace" << _iTrace << "::calcProfile: _profile->getArray() set to [" << _profile->getHeight() << ", " << _profile->getWidth() << "]: " << _profile->getArray() << endl;
    #endif

    _isProfileSet = true;
    return true;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  void pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setXCenters( ndarray::Array< float, 1, 1 > const& xCenters){
    if (static_cast<size_t>(xCenters.getShape()[0])
        != (_fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1)){
      string message("pfs::drp::stella::FiberTrace::setXCenters: ERROR: xCenters.getShape()[ 0 ](=" );
      message += std::to_string( xCenters.getShape()[ 0 ] ) + ") != _fiberTraceFunction.yHigh - _fiberTraceFunction.yLow + 1(=";
      message += std::to_string( _fiberTraceFunction->yHigh - _fiberTraceFunction->yLow + 1 )+ ")";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    _xCenters = ndarray::allocate( xCenters.getShape()[0] );
    _xCenters.deep() = xCenters;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  void pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setXCentersMeas( ndarray::Array< float, 2, 1 > const& xCentersMeas){
    _xCentersMeas = ndarray::allocate(xCentersMeas.getShape()[0], xCentersMeas.getShape()[1]);
    _xCentersMeas.deep() = xCentersMeas;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  ndarray::Array<double, 2, 1> pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::calcProfileSwath(ndarray::Array<ImageT const, 2, 1> const& imageSwath,
                                                                                                    ndarray::Array<MaskT const, 2, 1> const& maskSwath,
                                                                                                    ndarray::Array<VarianceT const, 2, 1> const& varianceSwath,
                                                                                                    ndarray::Array<float const, 1, 1> const& xCentersSwath,
                                                                                                    size_t const iSwath){

    /// Check shapes of input arrays
    if (imageSwath.getShape()[0] != maskSwath.getShape()[0]){
      string message("pfs::drp::stella::FiberTrace::calcProfileSwath: ERROR: imageSwath.getShape()[0](=");
      message += to_string(imageSwath.getShape()[0]) + ") != maskSwath.getShape()[0](=" + to_string(maskSwath.getShape()[0]);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (imageSwath.getShape()[0] != varianceSwath.getShape()[0]){
      string message("pfs::drp::stella::FiberTrace::calcProfileSwath: ERROR: imageSwath.getShape()[0](=");
      message += to_string(imageSwath.getShape()[0]) + ") != varianceSwath.getShape()[0](=" + to_string(varianceSwath.getShape()[0]);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (imageSwath.getShape()[0] != xCentersSwath.getShape()[0]){
      string message("pfs::drp::stella::FiberTrace::calcProfileSwath: ERROR: imageSwath.getShape()[0](=");
      message += to_string(imageSwath.getShape()[0]) + ") != xCentersSwath.getShape()[0](=" + to_string(xCentersSwath.getShape()[0]);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (imageSwath.getShape()[1] != maskSwath.getShape()[1]){
      string message("pfs::drp::stella::FiberTrace::calcProfileSwath: ERROR: imageSwath.getShape()[1](=");
      message += to_string(imageSwath.getShape()[1]) + ") != maskSwath.getShape()[1](=" + to_string(maskSwath.getShape()[1]);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (imageSwath.getShape()[1] != varianceSwath.getShape()[1]){
      string message("pfs::drp::stella::FiberTrace::calcProfileSwath: ERROR: imageSwath.getShape()[1](=");
      message += to_string(imageSwath.getShape()[1]) + ") != varianceSwath.getShape()[1](=" + to_string(varianceSwath.getShape()[1]);
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    /// Normalize rows in imageSwath
    ndarray::Array<double, 2, 1> imageSwathNormalized = ndarray::allocate(imageSwath.getShape()[0], imageSwath.getShape()[1]);
    ndarray::Array<double, 1, 1> sumArr = ndarray::allocate(imageSwath.getShape()[1]);
    for (int iRow = 0; iRow < imageSwath.getShape()[0]; ++iRow){
      sumArr.deep() = ndarray::Array<ImageT const, 1, 1>(imageSwath[ndarray::view(iRow)()]);
      imageSwathNormalized[ndarray::view(iRow)()] = ndarray::Array<ImageT const, 1, 1>(imageSwath[ndarray::view(iRow)()]) / ndarray::sum(sumArr);
    }
    #ifdef __DEBUG_CALCPROFILESWATH__
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": imageSwath = " << imageSwath << endl;
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": imageSwathNormalized = " << imageSwathNormalized << endl;
    #endif

    /// Calculate pixel offset to xCenter
    ndarray::Array<float, 1, 1> xCentersTemp = ndarray::allocate(xCentersSwath.getShape()[0]);
    xCentersTemp.deep() = xCentersSwath + 0.5 - PIXEL_CENTER;
    const ndarray::Array<size_t const, 1, 1> xCentersInt = math::floor(xCentersTemp, size_t(0));
    ndarray::Array<double, 1, 1> pixelOffset = ndarray::allocate(xCentersSwath.getShape()[0]);
    pixelOffset.deep() = PIXEL_CENTER;
    pixelOffset.deep() -= xCentersSwath;
    pixelOffset.deep() += xCentersInt;
    #ifdef __DEBUG_CALCPROFILESWATH__
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": pixelOffset = " << pixelOffset << endl;
    #endif
    const ndarray::Array<double const, 1, 1> xCenterArrayIndexX = math::indGenNdArr(double(imageSwath.getShape()[1]));
    const ndarray::Array<double const, 1, 1> xCenterArrayIndexY = math::replicate(double(1), int(imageSwath.getShape()[0]));
    ndarray::Array<double, 2, 1> xArray = ndarray::allocate(imageSwath.getShape()[0], imageSwath.getShape()[1]);
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
    PTR(vector<double>) xVecPtr(new vector<double>());
    xVecPtr->reserve(xArray.getShape()[0] * xArray.getShape()[1]);
    PTR(vector<double>) yVecPtr(new vector<double>());
    yVecPtr->reserve(xArray.getShape()[0] * xArray.getShape()[1]);
    auto itRowIm = imageSwathNormalized.begin();
    for (auto itRow = xArray.begin(); itRow != xArray.end(); ++itRow, ++itRowIm){
      auto itColIm = itRowIm->begin();
      for (auto itCol = itRow->begin(); itCol != itRow->end(); ++itCol, ++itColIm){
        xVecPtr->push_back(*itCol);
        yVecPtr->push_back(double(*itColIm));
      }
    }
    _profileFittingInputXPerSwath.push_back(xVecPtr);
    _profileFittingInputYPerSwath.push_back(yVecPtr);
    #ifdef __DEBUG_CALCPROFILESWATH__
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": xArray = " << xArray << endl;
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": xMin = " << xMin << ", xMax = " << xMax << endl;
    #endif
    double xOverSampleStep = 1. / _fiberTraceProfileFittingControl->overSample;
    #ifdef __DEBUG_CALCPROFILESWATH__
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": initial xOverSampleStep = " << xOverSampleStep << endl;
    #endif

    ///adjust xOverSampleStep to cover x from xMin + xOverSampleStep/2 to xMax - xOverSampleStep/2
    int nSteps = (xMax - xMin) / xOverSampleStep + 1;
    xOverSampleStep = (xMax - xMin) / nSteps;
    #ifdef __DEBUG_CALCPROFILESWATH__
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": final xOverSampleStep = " << xOverSampleStep << endl;
    #endif
    ndarray::Array<double, 1, 1> xOverSampled = ndarray::allocate(nSteps);
    double xStart = xMin + (xOverSampleStep / 2.);
    int iStep = 0;
    for (auto it = xOverSampled.begin(); it != xOverSampled.end(); ++it){
      *it = xStart + (iStep * xOverSampleStep);
      ++iStep;
    }
    #ifdef __DEBUG_CALCPROFILESWATH__
      cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": xOverSampled = " << xOverSampled << endl;
      cout << "FiberTrace::calcProfileSwath: _fiberTraceProfileFittingControl->maxIterSig = " << _fiberTraceProfileFittingControl->maxIterSig << endl;
    #endif
    PTR(vector<double>) xOverSampledFitVec(new vector<double>(xOverSampled.begin(), xOverSampled.end()));
    _overSampledProfileFitXPerSwath.push_back(xOverSampledFitVec);

    /// calculate oversampled profile values
    iStep = 0;
    std::vector< std::pair<double, double> > valOverSampledVec;
    int iStepsWithValues = 0;
    int bThisStepHasValues = false;
    ImageT mean = 0.;
    double rangeStart = double(xOverSampled[0] - (xOverSampleStep / 2.));
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
        cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": iStep" << iStep << ": rangeStart = " << rangeStart << ", rangeEnd = " << rangeEnd << endl;
        printf("rangeStart = %.9f, rangeEnd = %.9f\n", rangeStart, rangeEnd);
      #endif
      ndarray::Array<size_t, 2, 1> indicesInValueRange = math::getIndicesInValueRange(xArray, rangeStart, rangeEnd);
      #ifdef __DEBUG_CALCPROFILESWATH__
        cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": iStep" << iStep << ": indicesInValueRange = " << indicesInValueRange << endl;
      #endif
      std::vector< std::pair<size_t, size_t> > indicesInValueRangeVec(indicesInValueRange.getShape()[0]);
      for (int i = 0; i < indicesInValueRange.getShape()[0]; ++i){
        indicesInValueRangeVec[i].first = indicesInValueRange[i][0];
        indicesInValueRangeVec[i].second = indicesInValueRange[i][1];
        #ifdef __DEBUG_CALCPROFILESWATH__
          cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": iStep" << iStep << ": indicesInValueRangeVec[" << i << "].first = " << indicesInValueRangeVec[i].first << ", indicesInValueRangeVec[" << i << "].second = " << indicesInValueRangeVec[i].second << endl;
        #endif
      }
      do{
        ndarray::Array<double, 1, 1> subArr = math::getSubArray(imageSwathNormalized, indicesInValueRangeVec);
        ndarray::Array<double, 1, 1> xSubArr = math::getSubArray(xArray, indicesInValueRangeVec);
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
            ndarray::Array<double, 1, 1> moments = math::moment(subArr, 2);
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
                #ifdef __DEBUG_CALCPROFILESWATH__
                  cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": iStep = " << iStep << ": iterSig = " << iterSig << ": rejecting element " << i << "from subArr" << endl;
                #endif
                indicesInValueRangeVec.erase(indicesInValueRangeVec.begin() + i);
              }
            }
          }
          ndarray::Array<double, 1, 1> moments = math::moment(subArr, 1);
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
        valOverSampledVec.push_back(std::pair<double, double>(*it, mean));
        #ifdef __DEBUG_CALCPROFILESWATH__
          cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": valOverSampledVec[" << iStep << "] = (" << valOverSampledVec[iStep].first << ", " << valOverSampledVec[iStep].second << ")" << endl;
        #endif
      }
      rangeStart = rangeEnd;
      rangeEnd += xOverSampleStep;
    }
    ndarray::Array<double, 1, 1> valOverSampled = ndarray::allocate(valOverSampledVec.size());
    ndarray::Array<double, 1, 1> xValOverSampled = ndarray::allocate(valOverSampledVec.size());
    for (int iRow = 0; iRow < valOverSampledVec.size(); ++iRow){
      xValOverSampled[iRow] = valOverSampledVec[iRow].first;
      valOverSampled[iRow] = valOverSampledVec[iRow].second;
      #ifdef __DEBUG_CALCPROFILESWATH__
        cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": (x)valOverSampled[" << iRow << "] = (" << xValOverSampled[iRow] << "," << valOverSampled[iRow] << ")" << endl;
      #endif
    }
    PTR(std::vector<double>) xVecMean(new vector<double>(xValOverSampled.begin(), xValOverSampled.end()));
    std::vector<ImageT> yVecMean(valOverSampled.begin(), valOverSampled.end());

    PTR(std::vector<double>) yVecMeanF(new vector<double>(yVecMean.size()));
    auto itF = yVecMeanF->begin();
    for (auto itT = yVecMean.begin(); itT != yVecMean.end(); ++itT, ++itF){
      *itF = double(*itT);
    }
    _profileFittingInputXMeanPerSwath.push_back(xVecMean);
    _profileFittingInputYMeanPerSwath.push_back(yVecMeanF);

    math::spline<double> spline;
    spline.set_points(*xVecMean, *yVecMeanF);    // currently it is required that X is already sorted

    PTR(vector<double>) yOverSampledFitVec(new vector<double>(nSteps));
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
    ndarray::Array<double, 2, 1> profArraySwath = ndarray::allocate(imageSwath.getShape()[0], imageSwath.getShape()[1]);
    double tmpVal = 0.0;
    for (int iRow = 0; iRow < imageSwath.getShape()[0]; ++iRow){
      #ifdef __DEBUG_CALCPROFILESWATH__
        cout << "FiberTrace::calcProfileSwath: iSwath = " << iSwath << ": xArray[" << iRow << "][*] = " << xArray[ndarray::view(iRow)()] << endl;
      #endif
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
            cout << "FiberTrace::calcProfileSwath: xArray[" << iRow << "][" << iCol << "] = " << xArray[iRow][iCol] << endl;
            cout << "FiberTrace::calcProfileSwath: profArraySwath[" << iRow << "][" << iCol << "] = " << profArraySwath[iRow][iCol] << endl;
          }
        #endif
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

  template< typename ImageT, typename MaskT, typename VarianceT >
  PTR( pfsDRPStella::FiberTrace< ImageT, MaskT, VarianceT >) & pfsDRPStella::FiberTraceSet< ImageT, MaskT, VarianceT >::getFiberTrace( const size_t i ){
    if (i >= _traces->size()){
      string message("FiberTraceSet::getFiberTrace(i=");
      message += to_string(i) + string("): ERROR: i > _traces->size()=") + to_string(_traces->size());
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return _traces->at(i);
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR(pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>) const& pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::getFiberTrace(const size_t i) const {
    if (i >= _traces->size()){
      string message("FiberTraceSet::getFiberTrace(i=");
      message += to_string(i) + "): ERROR: i > _traces->size()=" + to_string(_traces->size());
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return _traces->at(i);
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  ndarray::Array<float, 1, 1> pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::getTraceCoefficients() const{
    ndarray::Array<float, 1, 1> coeffs = ndarray::allocate(_fiberTraceFunction->coefficients.size());
    auto itCo = _fiberTraceFunction->coefficients.begin();
    for (auto it = coeffs.begin(); it != coeffs.end(); ++it, ++itCo)
      *it = *itCo;
    return coeffs;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::erase(const size_t iStart, const size_t iEnd){
    if (iStart >= _traces->size()){
      string message("FiberTraceSet::erase(iStart=");
      message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iStart >= _traces->size()=" + to_string(_traces->size());
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (iEnd > _traces->size()){
      string message("FiberTraceSet::erase(iStart=");
      message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iEnd >= _traces->size()=" + to_string(_traces->size());
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (iEnd > 0){
      if (iStart > iEnd){
        string message("FiberTraceSet::erase(iStart=");
        message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iStart > iEnd";
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
    size_t size = _traces->size();
    _traces->push_back(trace);
    if (_traces->size() == size){
      string message("FiberTraceSet::addFiberTrace: ERROR: could not add trace to _traces");
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (iTrace > 0){
      (*_traces)[size]->setITrace(iTrace);
    }
    return true;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  void pfsDRPStella::FiberTraceSet< ImageT, MaskT, VarianceT >::sortTracesByXCenter()
  {
    std::vector<float> xCenters;
    for (int iTrace = 0; iTrace < static_cast<int>(_traces->size()); ++iTrace){
      xCenters.push_back((*_traces)[iTrace]->getFiberTraceFunction()->xCenter);
    }
    std::vector<int> sortedIndices(xCenters.size());
    sortedIndices = ::pfs::drp::stella::math::sortIndices(xCenters);
    #ifdef __DEBUG_SORTTRACESBYXCENTER__
      for (int iTrace = 0; iTrace < static_cast<int>(_traces->size()); ++iTrace)
        cout << "FiberTraceSet::sortTracesByXCenter: sortedIndices(" << iTrace << ") = " << sortedIndices[iTrace] << ", xCenters[sortedIndices[iTrace]] = " << xCenters[sortedIndices[iTrace]] << endl;
    #endif

    std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)> sortedTraces(_traces->size());
    for (size_t i = 0; i < _traces->size(); ++i){
      sortedTraces[ i ] = ( *_traces )[ sortedIndices[ i ] ];
      sortedTraces[ i ]->setITrace( i );
      #ifdef __DEBUG_SORTTRACESBYXCENTER__
        cout << "FiberTraceSet::sortTracesByXCenter: sortedTraces[" << i << "]->_iTrace set to " << sortedTraces[i]->getITrace() << endl;
      #endif
    }
    _traces.reset(new std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)>(sortedTraces));
    #ifdef __DEBUG_SORTTRACESBYXCENTER__
      for (size_t i = 0; i < _traces->size(); ++i){
        cout << "FiberTraceSet::sortTracesByXCenter: _traces[" << i << "]->_iTrace set to " << (*_traces)[i]->getITrace() << endl;
      }
    #endif
    return;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::setFiberTraceProfileFittingControl(PTR(FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl){
    for (unsigned int i=0; i<_traces->size(); ++i){
      if (!(*_traces)[i]->setFiberTraceProfileFittingControl(fiberTraceProfileFittingControl)){
        string message("FiberTraceSet::setFiberTraceProfileFittingControl: ERROR: (*_traces)[");
        message += to_string(i) + "]->setFiberTraceProfileFittingControl(fiberTraceProfileFittingControl) returned FALSE";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    }
    return true;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR( pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT> ) pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::extractTraceNumberFromProfile(const size_t traceNumber)
  {
    return (*_traces)[traceNumber]->extractFromProfile();
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  PTR( pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, VarianceT> ) pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::extractAllTracesFromProfile()
  {
    LOG_LOGGER _log = LOG_GET("pfs::drp::stella::FiberTraceSet::extractAllTracesFromProfile");
    PTR( pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, VarianceT> ) spectrumSet ( new pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, VarianceT>( _traces->size() ) );
    for (size_t i = 0; i < _traces->size(); ++i){
      LOGLS_DEBUG(_log, "extracting FiberTrace " << i);
      spectrumSet->setSpectrum(i, (*_traces)[i]->extractFromProfile());
    }
    return spectrumSet;
  }

  template<typename ImageT, typename MaskT, typename VarianceT>
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::setAllProfiles(const PTR(FiberTraceSet<ImageT, MaskT, VarianceT>) &fiberTraceSet){
    for (size_t i = 0; i < _traces->size(); ++i){
      if (!(*_traces)[i]->setProfile(fiberTraceSet->getFiberTrace(i)->getProfile())){
        string message("FiberTraceSet::copyAllProfiles: ERROR: (*_traces)[");
        message += to_string(i) + "].setProfile(fiberTraceSet->getFiberTrace(" + to_string(i) + ")->getProfile()) returned FALSE";
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
      afwImage::MaskedImage<ImageT, MaskT, VarianceT> maskedImageCopy(*maskedImage, true);
      PTR(afwImage::Image<ImageT>) ccdImage = maskedImageCopy.getImage();
      PTR(afwImage::Image<VarianceT>) ccdVarianceImage = maskedImageCopy.getVariance();
      ndarray::Array<ImageT, 2, 1> ccdArray = ndarray::copy(ccdImage->getArray());
      bool B_ApertureFound;

      std::vector<std::string> keywords(1);
      keywords[0] = "XRANGE";
      std::vector<void*> args(1);
      ndarray::Array<double, 1, 1> xRange = ndarray::allocate(2);
      PTR(ndarray::Array<double, 1, 1>) p_xRange(new ndarray::Array<double, 1, 1>(xRange));
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
        FindCenterPositionsOneTraceResult result = findCenterPositionsOneTrace( ccdImage,
                                                                                ccdVarianceImage,
                                                                                fiberTraceFunctionFindingControl );
        if (result.apertureCenterIndex.size() > fiberTraceFunctionFindingControl->minLength){
          B_ApertureFound = true;
          const ndarray::Array<float const, 1, 1> F_A1_ApertureCenterIndex = vectorToNdArray(result.apertureCenterIndex);
          const ndarray::Array<float const, 1, 1> F_A1_ApertureCenterPos = vectorToNdArray(result.apertureCenterPos);
          const ndarray::Array<float const, 1, 1> F_A1_EApertureCenterPos = vectorToNdArray(result.eApertureCenterPos);
          ndarray::Array<double, 1, 1> D_A1_ApertureCenterIndex = utils::typeCastNdArray(F_A1_ApertureCenterIndex, double(0.));
          ndarray::Array<double, 1, 1> D_A1_ApertureCenterPos = utils::typeCastNdArray(F_A1_ApertureCenterPos, double(0.));
          ndarray::Array<double, 1, 1> D_A1_EApertureCenterPos = utils::typeCastNdArray(F_A1_EApertureCenterPos, double(0.));

          #ifdef __DEBUG_FINDANDTRACE__
            cout << "::pfs::drp::stella::math::findAndTraceApertures: D_A1_ApertureCenterIndex = " << D_A1_ApertureCenterIndex << endl;
            cout << "::pfs::drp::stella::math::findAndTraceApertures: D_A1_ApertureCenterPos = " << D_A1_ApertureCenterPos << endl;
          #endif

          ndarray::Array<double, 1, 1> D_A1_PolyFitCoeffs;
          if (fiberTraceFunction.fiberTraceFunctionControl.interpolation.compare("CHEBYSHEV") == 0)
          {
            LOGL_INFO("pfs.drp.stella.math.findAndTraceApertures", "Fitting CHEBYSHEV Polynomial");
            int n = fiberTraceFunction.fiberTraceFunctionControl.order + 1;
            double xRangeMin = double(D_A1_ApertureCenterIndex[0]);
            double xRangeMax = double(D_A1_ApertureCenterIndex[D_A1_ApertureCenterIndex.getShape()[0]-1]);
            D_A1_PolyFitCoeffs = pfs::drp::stella::math::t_project_coefficients_data(D_A1_ApertureCenterIndex,
                                                                                     D_A1_ApertureCenterPos,
                                                                                     xRangeMin,
                                                                                     xRangeMax,
                                                                                     n);
          }
          else{
            /// Fit Polynomial
            (*p_xRange)[0] = D_A1_ApertureCenterIndex[0];
            (*p_xRange)[1] = D_A1_ApertureCenterIndex[int(D_A1_ApertureCenterIndex.size()-1)];
            D_A1_PolyFitCoeffs = pfsDRPStella::math::PolyFit(D_A1_ApertureCenterIndex,
                                                             D_A1_ApertureCenterPos,
                                                             fiberTraceFunctionFindingControl->fiberTraceFunctionControl.order,
                                                             keywords,
                                                             args);
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "::pfs::drp::stella::math::findAndTraceApertures: after PolyFit: D_A1_PolyFitCoeffs = " << D_A1_PolyFitCoeffs << endl;
            #endif
          }
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "D_A1_PolyFitCoeffs = " << D_A1_PolyFitCoeffs << endl;
          #endif

          fiberTraceFunction.xCenter = D_A1_ApertureCenterPos[int(D_A1_ApertureCenterIndex.size()/2.)];
          fiberTraceFunction.yCenter = int(D_A1_ApertureCenterIndex[int(D_A1_ApertureCenterIndex.size()/2.)]);
          fiberTraceFunction.yHigh = int(D_A1_ApertureCenterIndex[int(D_A1_ApertureCenterIndex.size()-1)] - fiberTraceFunction.yCenter);
          fiberTraceFunction.yLow = int(D_A1_ApertureCenterIndex[0]) - fiberTraceFunction.yCenter;
          fiberTraceFunction.coefficients.resize(D_A1_PolyFitCoeffs.getShape()[0]);
          auto itDC = D_A1_PolyFitCoeffs.begin();
          for (auto itCoeff = fiberTraceFunction.coefficients.begin(); itCoeff != fiberTraceFunction.coefficients.end(); ++itCoeff, ++itDC)
            *itCoeff = float(*itDC);
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.xCenter = " << fiberTraceFunction.xCenter << endl;
            cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.yCenter = " << fiberTraceFunction.yCenter << endl;
            cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.yLow = " << fiberTraceFunction.yLow << endl;
            cout << "::pfs::drp::stella::math::findAndTraceApertures: fiberTraceFunction.yHigh = " << fiberTraceFunction.yHigh << endl;
          #endif
          PTR(const pfsDRPStella::FiberTraceFunction) fiberTraceFunctionPTR(new pfsDRPStella::FiberTraceFunction(fiberTraceFunction));

          PTR( pfsDRPStella::FiberTrace< ImageT, MaskT, VarianceT >) fiberTrace( new pfsDRPStella::FiberTrace< ImageT, MaskT, VarianceT >( maskedImage,
                                                                                                                                           fiberTraceFunctionPTR,
                                                                                                                                           0 ) );
          fiberTrace->setITrace( fiberTraceSet->getTraces()->size() );
          if (fiberTrace->getXCenters().getShape()[0] != (fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1)){
            string message("FindAndTraceApertures: iTrace = ");
            message += to_string(fiberTraceSet->getTraces()->size()) + string(": 2. ERROR: fiberTrace->getXCenters()->size(=");
            message += to_string(fiberTrace->getXCenters().getShape()[0]) + string(") != (fiberTraceFunction.yHigh(=");
            message += to_string(fiberTraceFunction.yHigh) + string(") - fiberTraceFunction.yLow(=") + to_string(fiberTraceFunction.yLow);
            message += string(") + 1) = ") + to_string(fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1);
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          if (fiberTrace->getTrace()->getHeight() != (fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1)){
            string message("FindAndTraceApertures: iTrace = ");
            message += to_string(fiberTraceSet->getTraces()->size()) + string(": ERROR: fiberTrace->getTrace()->getHeight(=");
            message += to_string(fiberTrace->getTrace()->getHeight()) + string(")!= (fiberTraceFunction.yHigh(=");
            message += to_string(fiberTraceFunction.yHigh) + string(") - fiberTraceFunction.yLow(=") + to_string(fiberTraceFunction.yLow);
            message += string(") + 1) = ") + to_string(fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1);
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          if (fiberTraceFunctionFindingControl->fiberTraceFunctionControl.xLow
              != fiberTrace->getFiberTraceFunction()->fiberTraceFunctionControl.xLow){
            string message("FindAndTraceApertures: iTrace = ");
            message += to_string(fiberTraceSet->getTraces()->size());
            message += ": ERROR: fiberTrace->getFiberTraceFunction().";
            message += "FiberTraceFunctionControl.xLow(=";
            message += to_string(fiberTraceFunctionFindingControl->fiberTraceFunctionControl.xLow);
            message += ") != fiberTrace->getFiberTraceFunction()->fiberTraceFunctionControl.xLow(=";
            message += to_string(fiberTrace->getFiberTraceFunction()->fiberTraceFunctionControl.xLow);
            message += ")";
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          if (fiberTraceFunctionFindingControl->fiberTraceFunctionControl.xHigh
              != fiberTrace->getFiberTraceFunction()->fiberTraceFunctionControl.xHigh){
            string message("FindAndTraceApertures: iTrace = ");
            message += to_string(fiberTraceSet->getTraces()->size());
            message += ": ERROR: fiberTrace->getFiberTraceFunction()->";
            message += "FiberTraceFunctionControl.xHigh(=";
            message += to_string(fiberTraceFunctionFindingControl->fiberTraceFunctionControl.xHigh);
            message += ") != fiberTrace->getFiberTraceFunction()->fiberTraceFunctionControl.xHigh(=";
            message += to_string(fiberTrace->getFiberTraceFunction()->fiberTraceFunctionControl.xHigh);
            message += ")";
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          ndarray::Array<float, 2, 1> xCentersMeas = ndarray::allocate(D_A1_ApertureCenterPos.getShape()[0], 2);
          xCentersMeas[ndarray::view()(0)].deep() = D_A1_ApertureCenterIndex;
          xCentersMeas[ndarray::view()(1)].deep() = D_A1_ApertureCenterPos;
          fiberTrace->setXCentersMeas(xCentersMeas);
          fiberTraceSet->addFiberTrace(fiberTrace);
          ++I_Aperture;
        }/// end if (B_ApertureFound)
        else{
          B_ApertureFound = false;
        }
      } while (B_ApertureFound);
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
      double D_MaxTimesApertureWidth = 4.;
      std::vector<double> gaussFitVariances(0);
      std::vector<double> gaussFitMean(0);
      ndarray::Array<double, 1, 1> xCorRange = ndarray::allocate(2);
      xCorRange[0] = -0.5;
      xCorRange[1] = 0.5;
      double xCorMinPos = 0.;
      int nInd = 100;
      ndarray::Array<double, 2, 1> indGaussArr = ndarray::allocate(nInd, 2);
      std::vector<pfs::drp::stella::math::dataXY<double>> xySorted(0);
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
      ndarray::Array<double, 1, 1> D_A1_IndexCol = pfsDRPStella::math::indGenNdArr(double(ccdArray.getShape()[1]));
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: D_A1_IndexCol = " << D_A1_IndexCol << endl;
      #endif
      ndarray::Array<double, 1, 1> D_A1_Guess = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit);
      ndarray::Array<double, 1, 1> D_A1_GaussFit_Coeffs = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit);
      ndarray::Array<double, 1, 1> D_A1_GaussFit_Coeffs_Bak = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit);
      ndarray::Array<int, 1, 1> I_A1_Signal = ndarray::allocate(ccdArray.getShape()[1]);
      I_A1_Signal[ndarray::view()] = 0;
      ndarray::Array<double, 1, 1> D_A1_ApertureCenter = ndarray::allocate(ccdArray.getShape()[0]);
      ndarray::Array<double, 1, 1> D_A1_EApertureCenter = ndarray::allocate(ccdArray.getShape()[0]);
      ndarray::Array<int, 1, 1> I_A1_ApertureCenterIndex = ndarray::allocate(ccdArray.getShape()[0]);
      ndarray::Array<int, 1, 1> I_A1_IndSignal;
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: started" << endl;
      #endif
      ndarray::Array<int, 1, 1> I_A1_Ind;
      ndarray::Array<int, 1, 1> I_A1_Where;
      #ifdef __DEBUG_FINDANDTRACE__
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
        #ifdef __DEBUG_FINDANDTRACE__
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
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "pfs::drp::stella::math::findAndTraceApertures: while: I_A1_Signal = " << I_A1_Signal << endl;
            cout << "pfs::drp::stella::math::findAndTraceApertures: while: I_MinWidth = " << I_MinWidth << endl;
            cout << "pfs::drp::stella::math::findAndTraceApertures: while: I_StartIndex = " << I_StartIndex << endl;
          #endif
          I_FirstWideSignal = pfsDRPStella::math::firstIndexWithValueGEFrom(I_A1_Signal, I_MinWidth, I_StartIndex);
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": I_FirstWideSignal found at index " << I_FirstWideSignal << ", I_StartIndex = " << I_StartIndex << endl;
          #endif
          if (I_FirstWideSignal < 0){
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": No Aperture found in row " << i_Row << ", trying next row" << endl;
            #endif
            break;
          }
          else{
            I_FirstWideSignalStart = ::pfs::drp::stella::math::lastIndexWithZeroValueBefore(I_A1_Signal, I_FirstWideSignal) + 1;
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": 1. I_FirstWideSignalStart = " << I_FirstWideSignalStart << endl;
            #endif

            I_FirstWideSignalEnd = pfsDRPStella::math::firstIndexWithZeroValueFrom(I_A1_Signal, I_FirstWideSignal) - 1;
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
            #endif

            if (I_FirstWideSignalStart < 0){
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": WARNING: No start of aperture found -> Going to next Aperture." << endl;
              #endif

              if (I_FirstWideSignalEnd < 0){
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": 1. WARNING: No end of aperture found -> Going to next row." << endl;
                #endif
                break;
              }
              else{
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": End of first wide signal found at index " << I_FirstWideSignalEnd << endl;
                #endif
                /// Set start index for next run
                I_StartIndex = I_FirstWideSignalEnd+1;
              }
            }
            else{ /// Fit Gaussian and Trace Aperture
              if (I_FirstWideSignalEnd < 0){
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": 2. WARNING: No end of aperture found -> Going to next row." << endl;
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": B_ApertureFound = " << B_ApertureFound << endl;
                #endif
                break;
              }
              else{
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": End of first wide signal found at index " << I_FirstWideSignalEnd << endl;
                #endif

                if (I_FirstWideSignalEnd - I_FirstWideSignalStart + 1 > fiberTraceFunctionFindingControl->apertureFWHM * D_MaxTimesApertureWidth){
                  I_FirstWideSignalEnd = I_FirstWideSignalStart + int(D_MaxTimesApertureWidth * fiberTraceFunctionFindingControl->apertureFWHM);
                }
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": I_FirstWideSignalStart = " << I_FirstWideSignalStart << ", I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                #endif
                I_A1_Signal[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)] = 0;

                /// Set start index for next run
                I_StartIndex = I_FirstWideSignalEnd+1;
              }
              I_Length = I_FirstWideSignalEnd - I_FirstWideSignalStart + 1;
              #ifdef __DEBUG_FINDANDTRACE__
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
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": Aperture found at " << D_A1_ApertureCenter[i_Row] << endl;
                #endif

                /// Set signal to zero
                if ((I_FirstWideSignalEnd - 1) >=(I_FirstWideSignalStart + 1))
                  ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
              }
              else{// if (fiberTraceFunctionFindingControl->nTermsGaussFit > 0){
                if (I_Length <= fiberTraceFunctionFindingControl->nTermsGaussFit){
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": WARNING: Width of aperture <= " << fiberTraceFunctionFindingControl->nTermsGaussFit << "-> abandoning aperture" << endl;
                  #endif

                  /// Set signal to zero
                  if ((I_FirstWideSignalEnd - 1) >=(I_FirstWideSignalStart + 1))
                    ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                }
                else{
                  /// populate Arrays for GaussFit
                  ndarray::Array<double, 1, 1> D_A1_X = copy(ndarray::Array<double, 1, 1>(D_A1_IndexCol[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)]));
                  ndarray::Array<double, 1, 1> D_A1_Y = ndarray::allocate(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
                  D_A1_Y.deep() = ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)];
                  for (auto it = D_A1_Y.begin(); it != D_A1_Y.end(); ++it){
                    if (*it < 0.000001)
                      *it = 1.;
                  }
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: 1. D_A1_Y set to " << D_A1_Y << endl;
                  #endif
                  ndarray::Array<VarianceT, 1, 1> T_A1_MeasureErrors = ndarray::allocate(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
                  T_A1_MeasureErrors.deep() = ccdVarianceArray[ndarray::view(i_Row)(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)];
                  ndarray::Array<double, 1, 1> D_A1_MeasureErrors = ndarray::allocate(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
                  for (int ooo = 0; ooo < I_FirstWideSignalEnd - I_FirstWideSignalStart + 1; ++ooo){
                    if (T_A1_MeasureErrors[ooo] > 0)
                      D_A1_MeasureErrors[ooo] = double(sqrt(T_A1_MeasureErrors[ooo]));
                    else
                      D_A1_MeasureErrors[ooo] = 1;
                  }

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
                  ndarray::Array<double, 1, 1> D_A1_GaussFit_ECoeffs = ndarray::allocate(D_A1_GaussFit_Coeffs.size());
                  D_A1_GaussFit_ECoeffs[ndarray::view()] = 0.;

                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": D_A1_X = " << D_A1_X << ", D_A1_Y = " << D_A1_Y << endl;
                  #endif

                  ndarray::Array<int, 2, 1> I_A2_Limited = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit, 2);
                  I_A2_Limited[ndarray::view()] = 1;
                  ndarray::Array<double, 2, 1> D_A2_Limits = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit, 2);
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
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: WARNING: GaussFit FAILED -> abandoning aperture" << endl;
                    #endif

                    /// Set start index for next run
                    I_StartIndex = I_FirstWideSignalEnd+1;

                    ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                  }
                  else{
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
                      if (D_A1_GaussFit_Coeffs[0] > fiberTraceFunctionFindingControl->saturationLevel){
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: WARNING: Signal appears to be saturated" << endl;
                      }
                      if ((D_A1_GaussFit_Coeffs[1] < double(I_FirstWideSignalStart) + (double(I_Length)/4.)) || (D_A1_GaussFit_Coeffs[1] > double(I_FirstWideSignalStart) + (double(I_Length) * 3./4.))){
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: Warning: Center of Gaussian far away from middle of signal" << endl;
                      }
                    #endif
                    if ((D_A1_GaussFit_Coeffs[1] < double(I_FirstWideSignalStart)) || (D_A1_GaussFit_Coeffs[1] > double(I_FirstWideSignalEnd))){
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                      #endif
                      /// Set signal to zero
                      ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;

                      /// Set start index for next run
                      I_StartIndex = I_FirstWideSignalEnd+1;
                    }
                    else{
                      if ((D_A1_GaussFit_Coeffs[2] < fiberTraceFunctionFindingControl->apertureFWHM / 4.) || (D_A1_GaussFit_Coeffs[2] > fiberTraceFunctionFindingControl->apertureFWHM)){
                        #ifdef __DEBUG_FINDANDTRACE__
                          cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: WARNING: FWHM = " << D_A1_GaussFit_Coeffs[2] << " outside range -> abandoning aperture" << endl;
                        #endif
                        /// Set signal to zero
                        ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                        #ifdef __DEBUG_FINDANDTRACE__
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
                        #ifdef __DEBUG_FINDANDTRACE__
                          cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << i_Row << ": Aperture found at " << D_A1_ApertureCenter[i_Row] << endl;
                        #endif
                        ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
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
          xySorted.resize(0);
          apertureLength = 1;
          I_Length = 1;
          I_ApertureLost = 0;
          #ifdef __DEBUG_FINDANDTRACE__
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
                #ifdef __DEBUG_FINDANDTRACE__
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
                #ifdef __DEBUG_FINDANDTRACE__
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
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": Warning: Width of Aperture <= " << fiberTraceFunctionFindingControl->nTermsGaussFit << " -> Lost Aperture" << endl;
                  #endif
                  /// Set signal to zero
                  ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                  I_ApertureLost++;
                }
                else{
                  ndarray::Array<double, 1, 1> D_A1_X = copy(ndarray::Array<double, 1, 1>(D_A1_IndexCol[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd+1)]));
                  ndarray::Array<double, 1, 1> D_A1_Y = ndarray::allocate(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
                  D_A1_Y.deep() = ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart, I_FirstWideSignalEnd+1)];
                  ndarray::Array<VarianceT, 1, 1> T_A1_MeasureErrors = copy(ccdVarianceArray[ndarray::view(i_Row)(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)]);
                  ndarray::Array<double, 1, 1> D_A1_MeasureErrors = ndarray::allocate(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
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
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": Signal not wide enough => Aperture lost" << endl;
                    #endif
                  }
                  else{
                    #ifdef __DEBUG_FINDANDTRACE__
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
                    ndarray::Array<double, 1, 1> D_A1_GaussFit_ECoeffs = ndarray::allocate(D_A1_GaussFit_Coeffs.size());
                    D_A1_GaussFit_ECoeffs[ndarray::view()] = 0.;

                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: D_A1_X = " << D_A1_X << ", D_A1_Y = " << D_A1_Y << endl;
                    #endif

                    ndarray::Array<int, 2, 1> I_A2_Limited = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit, 2);
                    I_A2_Limited[ndarray::view()] = 1;
                    ndarray::Array<double, 2, 1> D_A2_Limits = ndarray::allocate(fiberTraceFunctionFindingControl->nTermsGaussFit, 2);
                    D_A2_Limits[0][0] = 0.;/// Peak lower limit
                    D_A2_Limits[0][1] = 2. * D_A1_Guess[0];/// Peak upper limit
                    D_A2_Limits[1][0] = double(I_FirstWideSignalStart);/// Centroid lower limit
                    D_A2_Limits[1][1] = double(I_FirstWideSignalEnd);/// Centroid upper limit
                    D_A2_Limits[2][0] = double(fiberTraceFunctionFindingControl->apertureFWHM) / 4.;/// Sigma lower limit
                    D_A2_Limits[2][1] = double(fiberTraceFunctionFindingControl->apertureFWHM);/// Sigma upper limit
                    if (gaussFitVariances.size() > 15){
                      double sum = std::accumulate(gaussFitMean.end()-10, gaussFitMean.end(), 0.0);
                      double mean = sum / 10.;
                      #ifdef __DEBUG_FINDANDTRACE__
                        for (int iMean = 0; iMean < gaussFitMean.size(); ++iMean){
                          cout << "gaussFitMean[" << iMean << "] = " << gaussFitMean[iMean] << endl;
                          cout << "gaussFitVariances[" << iMean << ") = " << gaussFitVariances[iMean] << endl;
                        }
                        cout << "sum = " << sum << ", mean = " << mean << endl;
                      #endif
                      double sq_sum = std::inner_product(gaussFitMean.end()-10, gaussFitMean.end(), gaussFitMean.end()-10, 0.0);
                      double stdev = std::sqrt(sq_sum / 10 - mean * mean);
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "GaussFitMean: sq_sum = " << sq_sum << ", stdev = " << stdev << endl;
                      #endif
                      D_A1_Guess[1] = mean;
                      D_A2_Limits[1][0] = mean - (3. * stdev) - 0.1;
                      D_A2_Limits[1][1] = mean + (3. * stdev) + 0.1;
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: D_A1_Guess[1] = " << D_A1_Guess[1] << ", Limits = " << D_A2_Limits[ndarray::view(1)()] << endl;
                      #endif
                      for (int iMean = 0; iMean < gaussFitMean.size(); ++iMean)
                      sum = std::accumulate(gaussFitVariances.end()-10, gaussFitVariances.end(), 0.0);
                      mean = sum / 10.;
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "GaussFitVariance: sum = " << sum << ", mean = " << mean << endl;
                      #endif
                      sq_sum = std::inner_product(gaussFitVariances.end()-10, gaussFitVariances.end(), gaussFitVariances.end()-10, 0.0);
                      stdev = std::sqrt(sq_sum / 10 - mean * mean);
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "sq_sum = " << sq_sum << ", stdev = " << stdev << endl;
                      #endif
                      D_A1_Guess[2] = mean;
                      D_A2_Limits[2][0] = mean - (3. * stdev);
                      D_A2_Limits[2][1] = mean + (3. * stdev);
                      #ifdef __DEBUG_FINDANDTRACE__
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
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": while: 2. starting MPFitGaussLim: D_A2_Limits = " << D_A2_Limits << endl;
                    #endif
                    ndarray::Array<double, 2, 1> D_A2_XY = ndarray::allocate(D_A1_X.getShape()[0], 2);
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
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": Warning: GaussFit FAILED" << endl;
                      #endif
                      /// Set signal to zero
                      ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart + 1, I_FirstWideSignalEnd)] = 0.;

                      I_ApertureLost++;
                    }
                    else{
                      gaussFitMean.push_back(D_A1_GaussFit_Coeffs[1]);
                      gaussFitVariances.push_back(D_A1_GaussFit_Coeffs[2]);

                      #ifdef __DEBUG_FINDANDTRACE__
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
                          #ifdef __DEBUG_FINDANDTRACE__
                            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": WARNING: peak = " << D_A1_GaussFit_Coeffs[1] << " lower than signalThreshold -> abandoning aperture" << endl;
                          #endif
                          /// Set signal to zero
                          ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart + 1, I_FirstWideSignalEnd)] = 0.;
                          #ifdef __DEBUG_FINDANDTRACE__
                            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": 2. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                          #endif
                          I_ApertureLost++;
                      }
                      else{
                        if ((D_A1_GaussFit_Coeffs[1] < D_A1_GaussFit_Coeffs_Bak[1] - 1.) || (D_A1_GaussFit_Coeffs[1] > D_A1_GaussFit_Coeffs_Bak[1] + 1.)){
                          #ifdef __DEBUG_FINDANDTRACE__
                            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                          #endif
                          /// Set signal to zero
                          ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;

                          I_ApertureLost++;
                        }
                        else{
                          if ((D_A1_GaussFit_Coeffs[2] < fiberTraceFunctionFindingControl->apertureFWHM / 4.) || (D_A1_GaussFit_Coeffs[2] > fiberTraceFunctionFindingControl->apertureFWHM)){
                            #ifdef __DEBUG_FINDANDTRACE__
                              cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": WARNING: FWHM = " << D_A1_GaussFit_Coeffs[2] << " outside range -> abandoning aperture" << endl;
                            #endif
                            /// Set signal to zero
                            ccdArray[ndarray::view(i_Row)(I_FirstWideSignalStart+1, I_FirstWideSignalEnd)] = 0.;
                            #ifdef __DEBUG_FINDANDTRACE__
                              cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": 2. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                            #endif
                            I_ApertureLost++;
                          }
                          else{
                            I_ApertureLost = 0;
                            B_ApertureFound = true;
                            D_A1_ApertureCenter[i_Row] = D_A1_GaussFit_Coeffs[1];
                            D_A1_EApertureCenter[i_Row] = D_A1_GaussFit_ECoeffs[1];
                            #ifdef __DEBUG_FINDANDTRACE__
                              cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": Aperture found at " << D_A1_ApertureCenter[i_Row] << endl;
                            #endif
                            D_A1_GaussFit_Coeffs_Bak[ndarray::view()] = D_A1_GaussFit_Coeffs[ndarray::view()];
                            xCorMinPos = 0.;
                            int ind = 0;
                            ndarray::Array<double, 2, 1> xyRelativeToCenter = ndarray::allocate(D_A1_X.getShape()[0] + 2, 2);
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
                            double fac = (xyRelativeToCenter[xyRelativeToCenter.getShape()[0]-1][0] - xyRelativeToCenter[0][0]) / nInd;
                            for (auto itRow = indGaussArr.begin(); itRow != indGaussArr.end(); ++itRow, ++ind){
                              auto itCol = itRow->begin();
                              *itCol = *itCol + (ind * fac);
                              *(itCol + 1) = D_A1_GaussFit_Coeffs[0] * exp(0. - ((*itCol) * (*itCol)) / (2. * D_A1_Guess[2] * D_A1_Guess[2]));
                            }
                            if (gaussFitVariances.size() > 20){
                              ndarray::Array<double, 2, 1> xysRelativeToCenter = ndarray::allocate(xySorted.size(), 2);
                              ind = 0;
                              auto itSorted = xySorted.begin();
                              for (auto itRow = xysRelativeToCenter.begin(); itRow != xysRelativeToCenter.end(); ++itRow, ++ind, ++itSorted){
                                auto itCol = itRow->begin();
                                *itCol = itSorted->x;
                                *(itCol+1) = itSorted->y;
                              }
                              #ifdef __DEBUG_FINDANDTRACE__
                                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": xCorMinPos = " << xCorMinPos << endl;
                              #endif
                            }
                            if (gaussFitVariances.size() > 10){
                              for (int iX = 0; iX < xyRelativeToCenter.getShape()[0]; ++iX){
                                dataXY<double> xyStruct;
                                xyStruct.x = xyRelativeToCenter[iX][0] + xCorMinPos;
                                xyStruct.y = xyRelativeToCenter[iX][1];
                                pfs::drp::stella::math::insertSorted(xySorted, xyStruct);
                              }
                              D_A1_ApertureCenter[i_Row] = D_A1_ApertureCenter[i_Row] + xCorMinPos;
                              #ifdef __DEBUG_FINDANDTRACE__
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
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row set to " << i_Row << endl;
              #endif
            }
          }///end while(B_ApertureFound && (I_ApertureLost < 3) && i_Row < ccdArray.getShape()[0] - 2))

          /// Fit Polynomial to traced aperture positions
          #ifdef __DEBUG_FINDANDTRACE__
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
              result.apertureCenterIndex.push_back(double(iInd + PIXEL_CENTER));
              result.apertureCenterPos.push_back((*(itCen + iInd)) + PIXEL_CENTER);
              result.eApertureCenterPos.push_back((*(itECen + iInd)));
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": result.apertureCenterIndex[" << result.apertureCenterIndex.size()-1 << "] set to " << result.apertureCenterIndex[result.apertureCenterIndex.size()-1] << endl;
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": result.apertureCenterPos[" << result.apertureCenterPos.size()-1 << "] set to " << result.apertureCenterPos[result.apertureCenterPos.size()-1] << endl;
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": result.eApertureCenterPos[" << result.eApertureCenterPos.size()-1 << "] set to " << result.eApertureCenterPos[result.eApertureCenterPos.size()-1] << endl;
              #endif
            }
          }
          if (I_ApertureLength > fiberTraceFunctionFindingControl->minLength){
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << i_Row << ": result.apertureCenterIndex.size() = " << result.apertureCenterIndex.size() << endl;
            #endif
            return result;
          }
        }
      }
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: result.apertureCenterIndex.size() = " << result.apertureCenterIndex.size() << endl;
      #endif
      return result;
    }


    /** *******************************************************************************************************/

    /// Calculate the x-centers of the fiber trace
    ndarray::Array<float, 1, 1> calculateXCenters(PTR(const pfsDRPStella::FiberTraceFunction) const& fiberTraceFunctionIn,
                                                  size_t const& ccdHeightIn,
                                                  size_t const& ccdWidthIn){
      ndarray::Array<float, 1, 1> xRowIndex = ndarray::allocate(fiberTraceFunctionIn->yHigh - fiberTraceFunctionIn->yLow + 1);
      float xRowInd = fiberTraceFunctionIn->yCenter + fiberTraceFunctionIn->yLow + PIXEL_CENTER;
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

      PTR(pfsDRPStella::FiberTraceFunction) pFTF = const_pointer_cast<pfsDRPStella::FiberTraceFunction>(fiberTraceFunctionIn);
      const ndarray::Array<float, 1, 1> fiberTraceFunctionCoefficients = ndarray::external(pFTF->coefficients.data(), ndarray::makeVector(int(fiberTraceFunctionIn->coefficients.size())), ndarray::makeVector(1));

      #ifdef __DEBUG_XCENTERS__
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunctionCoefficients = " << fiberTraceFunctionCoefficients << endl;
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunctionIn->fiberTraceFunctionControl.interpolation = " << fiberTraceFunctionIn->fiberTraceFunctionControl.interpolation << endl;
        cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunctionIn->fiberTraceFunctionControl.order = " << fiberTraceFunctionIn->fiberTraceFunctionControl.order << endl;
      #endif
      if (fiberTraceFunctionIn->fiberTraceFunctionControl.interpolation.compare("CHEBYSHEV") == 0)
      {
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::math::calculateXCenters: Calculating Chebyshev Polynomial" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Function = Chebyshev" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
        #endif
        ndarray::Array<double, 1, 1> range = ndarray::allocate(2);
        range[0] = fiberTraceFunctionIn->yCenter + fiberTraceFunctionIn->yLow + PIXEL_CENTER;
        range[1] = fiberTraceFunctionIn->yCenter + fiberTraceFunctionIn->yHigh + PIXEL_CENTER;
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::calculateXCenters: CHEBYSHEV: yIn = " << yIn << endl;
          cout << "pfs::drp::stella::calculateXCenters: CHEBYSHEV: range = " << range << endl;
        #endif
        ndarray::Array<float, 1, 1> yNew = pfs::drp::stella::math::convertRangeToUnity(yIn, range);
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::calculateXCenters: CHEBYSHEV: yNew = " << yNew << endl;
        #endif
        std::vector<float> coeffsVec = fiberTraceFunctionIn->coefficients;
        ndarray::Array<float, 1, 1> coeffs = ndarray::external(coeffsVec.data(), ndarray::makeVector(int(coeffsVec.size())), ndarray::makeVector(1));
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::calculateXCenters: CHEBYSHEV: coeffs = " << coeffs << endl;
        #endif
        xCenters = pfs::drp::stella::math::chebyshev(yNew, coeffs);
      }
      else /// Polynomial
      {
        #ifdef __DEBUG_XCENTERS__
          cout << "pfs::drp::stella::math::calculateXCenters: Calculating Polynomial" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Function = Polynomial" << endl;
          cout << "pfs::drp::stella::calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
          cout << "pfs::drp::stella::calculateXCenters: fiberTraceFunctionCoefficients = " << fiberTraceFunctionCoefficients << endl;
        #endif
        xCenters = pfsDRPStella::math::Poly( yIn,
                                              fiberTraceFunctionCoefficients,
                                              float(fiberTraceFunctionIn->yCenter + fiberTraceFunctionIn->yLow + PIXEL_CENTER),
                                              float(fiberTraceFunctionIn->yCenter + fiberTraceFunctionIn->yHigh + PIXEL_CENTER));
      }

      return xCenters;
    }

    template< typename ImageT, typename MaskT, typename VarianceT >
    PTR(FiberTrace< ImageT, MaskT, VarianceT >) makeNormFlatFiberTrace( PTR( const afwImage::MaskedImage< ImageT, MaskT, VarianceT >) const& maskedImage,
                                                                         PTR( const ::pfs::drp::stella::FiberTraceFunction ) const& fiberTraceFunctionWide,
                                                                         PTR( const ::pfs::drp::stella::FiberTraceFunctionControl ) const& fiberTraceFunctionControlNarrow,
                                                                         PTR( const ::pfs::drp::stella::FiberTraceProfileFittingControl ) const& fiberTraceProfileFittingControl,
                                                                         ImageT minSNR,
                                                                         size_t iTrace ){
      /// calculate center positions for fiberTraces
      ndarray::Array< float, 1, 1 > xCenters = calculateXCenters( fiberTraceFunctionWide,
                                                                  maskedImage->getHeight(),
                                                                  maskedImage->getWidth() );

      /// extract wide dithered flat FiberTrace
      FiberTrace< ImageT, MaskT, VarianceT > flatFiberTrace( maskedImage,
                                                             fiberTraceFunctionWide,
                                                             iTrace );

      if ( xCenters.getShape()[ 0 ] != flatFiberTrace.getHeight() ){
        std::string message("pfs::drp::stella::math::makeNormFlatFiberTrace: ERROR: xCenters.getShape()[ 0 ](=");
        message += std::to_string(xCenters.getShape()[ 0 ]) + " != flatFiberTrace.getHeight()(=" + std::to_string(flatFiberTrace.getHeight()) + ")";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }

      /// set ProfileFittingControl, fit spatial profile, and optimally extract Blaze function
      flatFiberTrace.setFiberTraceProfileFittingControl( fiberTraceProfileFittingControl );
      flatFiberTrace.calcProfile();
      PTR(Spectrum<ImageT, MaskT, VarianceT, VarianceT>) blazeFunc = flatFiberTrace.extractFromProfile();

      /// reconstruct dithered Flat from spectrum and profile
      PTR(afwImage::Image< ImageT >) reconstructedFlatIm = flatFiberTrace.getReconstructed2DSpectrum( *blazeFunc );

      /// calculate minCenMax for wide and narrow FiberTraces
      ndarray::Array<size_t, 2, 1> minCenMaxWide = calcMinCenMax( xCenters,
                                                                  fiberTraceFunctionWide->fiberTraceFunctionControl.xHigh,/// >= 0
                                                                  fiberTraceFunctionWide->fiberTraceFunctionControl.xLow,/// <= 0
                                                                  fiberTraceFunctionWide->fiberTraceFunctionControl.nPixCutLeft,
                                                                  fiberTraceFunctionWide->fiberTraceFunctionControl.nPixCutRight );

      FiberTraceFunction fiberTraceFunctionNarrow( *fiberTraceFunctionWide );
      fiberTraceFunctionNarrow.fiberTraceFunctionControl = *fiberTraceFunctionControlNarrow;

      ndarray::Array<size_t, 2, 1> minCenMaxNarrow = calcMinCenMax( xCenters,
                                                                    fiberTraceFunctionNarrow.fiberTraceFunctionControl.xHigh,/// >= 0
                                                                    fiberTraceFunctionNarrow.fiberTraceFunctionControl.xLow,/// <= 0
                                                                    fiberTraceFunctionNarrow.fiberTraceFunctionControl.nPixCutLeft,
                                                                    fiberTraceFunctionNarrow.fiberTraceFunctionControl.nPixCutRight );

      if ( xCenters.getShape()[ 0 ] != minCenMaxWide.getShape()[0] ){
        std::string message("pfs::drp::stella::math::makeNormFlatFiberTrace: ERROR: xCenters.getShape()[ 0 ](=");
        message += std::to_string(xCenters.getShape()[ 0 ]) + " != minCenMaxWide.getShape()[0](=" + std::to_string(minCenMaxWide.getShape()[0]) + ")";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }

      /// create normalized flat
      ndarray::Array< ImageT, 2, 1 > normFlat = ndarray::allocate( flatFiberTrace.getImage()->getArray().getShape()[ 0 ], minCenMaxNarrow[0][2] - minCenMaxNarrow[0][0] + 1 );
      auto itRowIm = flatFiberTrace.getImage()->getArray().begin();
      auto itRowVar = flatFiberTrace.getVariance()->getArray().begin();
      auto itRowRec = reconstructedFlatIm->getArray().begin();
      int iRow = fiberTraceFunctionWide->yCenter + fiberTraceFunctionWide->yLow;
      for (auto itRowNorm = normFlat.begin(); itRowNorm != normFlat.end(); ++itRowIm, ++itRowVar, ++itRowNorm, ++itRowRec, ++iRow ){
        auto itColIm = itRowIm->begin() + minCenMaxWide[iRow][0] - minCenMaxNarrow[iRow][0];
        auto itColVar = itRowVar->begin() + minCenMaxWide[iRow][0] - minCenMaxNarrow[iRow][0];
        auto itColRec = itRowRec->begin() + minCenMaxWide[iRow][0] - minCenMaxNarrow[iRow][0];
        for (auto itColNorm = itRowNorm->begin(); itColNorm != itRowNorm->end(); ++itColIm, ++itColVar, ++itColNorm, ++itColRec ){
          if ( ( *itColIm / sqrt( *itColVar ) ) < minSNR )
            *itColNorm = 1.;
          else
            *itColNorm = *itColIm / *itColRec;
        }
      }
      FiberTrace< ImageT, MaskT, VarianceT > normFlatFiberTrace( normFlat.getShape()[ 1 ],
                                                                 normFlat.getShape()[ 0 ],
                                                                 iTrace );
      if ( normFlatFiberTrace.getHeight() != flatFiberTrace.getHeight() ){
        std::string message( "pfs::drp::stella::math::makeNormFlatFiberTrace: ERROR: normFlatFiberTrace.getHeight(=" );
        message += std::to_string( normFlatFiberTrace.getHeight() ) + " != flatFiberTrace.getHeight()(=" + std::to_string( flatFiberTrace.getHeight() ) + ")";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      normFlatFiberTrace.setFiberTraceFunction( fiberTraceFunctionNarrow.getPointer() );
      normFlatFiberTrace.setXCenters( xCenters );
      return normFlatFiberTrace.getPointer();
    }

    template< typename ImageT, typename MaskT, typename VarianceT, typename T, typename U, int I >
    void assignITrace( FiberTraceSet< ImageT, MaskT, VarianceT > & fiberTraceSet,
                       ndarray::Array< T, 1, I > const& fiberIds,
                       ndarray::Array< U, 1, I > const& xCenters ){
      LOG_LOGGER _log = LOG_GET("pfs::drp::stella::math::assignITrace");

      size_t iTrace;
      size_t startPos = 0;
      ndarray::Array< T, 1, 1 > fiberIdsUnique = ndarray::allocate(fiberIds.getShape());
      fiberIdsUnique.deep() = fiberIds;
      const auto newEndIt = std::unique( fiberIdsUnique.begin(), fiberIdsUnique.end() );
      size_t nTraces = newEndIt - fiberIdsUnique.begin();
      LOGLS_DEBUG(_log, "assignITrace: nTraces = " << nTraces);
      size_t nRows = fiberIds.getShape()[0] / nTraces;
      LOGLS_DEBUG(_log, "assignITrace: nRows = " << nRows);

      for (size_t i = 0; i < fiberTraceSet.size(); ++i) {
        PTR( FiberTrace<ImageT, MaskT, VarianceT> ) fiberTrace = fiberTraceSet.getFiberTrace( i );
        iTrace = findITrace( *fiberTrace,
                             xCenters,
                             nTraces,
                             nRows,
                             startPos );
        startPos = iTrace + 1;
        LOGLS_DEBUG(_log, "assignITrace: i = " << i << ": iTrace = " << iTrace);
        LOGLS_DEBUG(_log, "assignITrace: i = " << i << ": fiberIdsUnique[" << iTrace << "] = "
                    << fiberIdsUnique[iTrace]);
        fiberTrace->setITrace(fiberIdsUnique[iTrace]);
      }
      return;
    }

    template< typename ImageT, typename MaskT, typename VarianceT, typename U, int I >
    size_t findITrace( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace,
                       ndarray::Array< U, 1, I > const& xCentersArr,
                       size_t nTraces,
                       size_t nRows,
                       size_t startPos ){
      LOG_LOGGER _log = LOG_GET("pfs::drp::stella::math::findITrace");
      float xCenter = fiberTrace.getFiberTraceFunction()->xCenter;
      unsigned int yCenter = fiberTrace.getFiberTraceFunction()->yCenter;
      LOGLS_DEBUG(_log, "xCenter = " << xCenter);
      LOGLS_DEBUG(_log, "yCenter = " << yCenter);

      float minDist = 100000.0;
      size_t iTrace = 0;
      for ( size_t i = startPos; i < nTraces; ++i){
        float dist = std::fabs( xCentersArr[ i * nRows + yCenter ] - xCenter );
        if (dist < minDist){
          minDist = dist;
          iTrace = i;
        }
      }
      LOGLS_DEBUG(_log, "iTrace = " << iTrace);
      return iTrace;
    }

    template< typename ImageT, typename MaskT, typename VarianceT, typename arrayT, typename ccdImageT, int dim >
    void addFiberTraceToCcdArray( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace,
                                  afwImage::Image< arrayT > const& fiberTraceRepresentation,
                                  ndarray::Array< ccdImageT, 2, dim > & ccdArray ){

      const PTR(const FiberTraceFunction) ftf = fiberTrace.getFiberTraceFunction();
      const ndarray::Array< float, 1, 1 > xCenters = fiberTrace.getXCenters();
      ndarray::Array< size_t, 2, 1 > minCenMax = pfsDRPStella::math::calcMinCenMax( xCenters,
                                                                                    ftf->fiberTraceFunctionControl.xHigh,
                                                                                    ftf->fiberTraceFunctionControl.xLow,
                                                                                    ftf->fiberTraceFunctionControl.nPixCutLeft,
                                                                                    ftf->fiberTraceFunctionControl.nPixCutRight);
      /// Check fiberTraceRepresentation dimension
      if ( fiberTraceRepresentation.getDimensions()[ 0 ] != ( minCenMax[ 0 ][ 2 ] - minCenMax[ 0 ][ 0 ] + 1 ) ){
        string message("pfs::drp::stella::math::addFiberTraceToCcdArray: ERROR: fiberTraceRepresentation.getDimensions()[ 0 ](=");
        message += to_string( fiberTraceRepresentation.getDimensions()[ 0 ] ) + " != ( minCenMax[ 0 ][ 2 ] - minCenMax[ 0 ][ 0 ] + 1 )(= ";
        message += to_string( minCenMax[ 0 ][ 2 ] - minCenMax[ 0 ][ 0 ] + 1 ) + ")";
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
      }
      if ( fiberTraceRepresentation.getDimensions()[ 1 ] != minCenMax.getShape()[ 0 ] ){
        string message( "pfs::drp::stella::math::addFiberTraceToCcdArray: ERROR: fiberTraceRepresentation.getDimensions()[ 1 ] )(=");
        message += to_string( fiberTraceRepresentation.getDimensions()[ 1 ] ) + ") != minCenMax.getShape()[ 0 ](=" + to_string( minCenMax.getShape()[ 0 ] ) + ")";
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
      }

      int y = ftf->yCenter + ftf->yLow;
      ndarray::Array<size_t, 2, 1> xMinMax = ndarray::allocate(minCenMax.getShape()[0], 2);
      xMinMax[ndarray::view()(0)] = minCenMax[ndarray::view()(0)];
      xMinMax[ndarray::view()(1)] = minCenMax[ndarray::view()(2)];
      addArrayIntoArray(fiberTraceRepresentation.getArray(),
                        xMinMax,
                        y,
                        ccdArray);
      return;
    }

    template< typename ImageT, typename MaskT, typename VarianceT, typename arrayT, typename ccdImageT >
    void addFiberTraceToCcdImage( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace,
                                  afwImage::Image< arrayT > const& fiberTraceRepresentation,
                                  afwImage::Image< ccdImageT > & ccdImage ){
      ndarray::Array< ccdImageT, 2, 1 > arr = ccdImage.getArray();
      return addFiberTraceToCcdArray( fiberTrace, fiberTraceRepresentation, arr );
    }

    template< typename smallT, typename bigT, int I, int J >
    void addArrayIntoArray( ndarray::Array< smallT, 2, I > const& smallArr,
                            ndarray::Array< size_t, 2, 1 > const& xMinMax,
                            size_t const& yMin,
                            ndarray::Array< bigT, 2, J > & bigArr ){
      assert( smallArr.getShape()[ 0 ] == xMinMax.getShape()[ 0 ] );
      assert( xMinMax.getShape()[ 1 ] == 2 );
      assert( smallArr.getShape()[ 0 ] + yMin <= bigArr.getShape()[ 0 ] );
      assert( smallArr.getShape()[ 1 ] + xMinMax[ 0 ][ 0 ] <= bigArr.getShape()[ 1 ] );
      for ( int i = 0; i < smallArr.getShape()[ 0 ]; ++i ){
        bigArr[ndarray::view(int(yMin) + i)(xMinMax[i][0], xMinMax[i][1] + 1)] += smallArr[ndarray::view(i)()];
      }
      return;
    }

    template<typename CoordT, typename ImageT, typename MaskT, typename VarianceT>
    dataXY<CoordT> ccdToFiberTraceCoordinates(
        dataXY<CoordT> const& ccdCoordinates,
        pfs::drp::stella::FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace)
    {
      LOG_LOGGER _log = LOG_GET("pfs::drp::stella::math::ccdToFiberTraceCoordinates");
      dataXY<CoordT> coordsOut;

      ndarray::Array< size_t, 2, 1 > minCenMax = calcMinCenMax(
            fiberTrace.getXCenters(),
            double( fiberTrace.getFiberTraceFunction()->fiberTraceFunctionControl.xHigh ),
            double( fiberTrace.getFiberTraceFunction()->fiberTraceFunctionControl.xLow ),
            fiberTrace.getFiberTraceFunction()->fiberTraceFunctionControl.nPixCutLeft,
            fiberTrace.getFiberTraceFunction()->fiberTraceFunctionControl.nPixCutRight );
      LOGLS_DEBUG(_log, "minCenMax[coordsOut.y=" << coordsOut.y << "] = " << minCenMax[coordsOut.y]);

      const PTR(const pfs::drp::stella::FiberTraceFunction) fiberTraceFunction = fiberTrace.getFiberTraceFunction();
      LOGLS_DEBUG(_log, "fiberTraceFunction->yCenter = " << fiberTraceFunction->yCenter);
      LOGLS_DEBUG(_log, "fiberTraceFunction->yLow = " << fiberTraceFunction->yLow);
      LOGLS_DEBUG(_log, "ccdCoordinates = (" << ccdCoordinates.x << ", " << ccdCoordinates.y << ")");
      coordsOut.y = ccdCoordinates.y
                    - (fiberTraceFunction->yCenter
                       + fiberTraceFunction->yLow);
      coordsOut.x = ccdCoordinates.x - minCenMax[coordsOut.y][0];
      LOGLS_DEBUG(_log, "coordsOut = (" << coordsOut.x << ", " << coordsOut.y << ")");

      return coordsOut;
    }

    template<typename CoordT, typename ImageT, typename MaskT, typename VarianceT>
    dataXY<CoordT> fiberTraceCoordinatesRelativeTo(
        dataXY<CoordT> const& fiberTraceCoordinates,
        dataXY<CoordT> const& ccdCoordinatesCenter,
        pfs::drp::stella::FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace)
    {
      LOG_LOGGER _log = LOG_GET("pfs::drp::stella::math::fiberTraceCoordinatesRelativeTo");

      dataXY<CoordT> traceCoordinatesCenter = ccdToFiberTraceCoordinates(
              ccdCoordinatesCenter,
              fiberTrace);
      string message("traceCoordinatesCenter = (");
      message += to_string(traceCoordinatesCenter.x) + ", ";
      message += to_string(traceCoordinatesCenter.y) + ")";
      LOGLS_DEBUG(_log, message);

      dataXY<CoordT> fiberTraceCoordinatesRelativeToCenter;
      message = "fiberTraceCoordinates = (";
      message += to_string(fiberTraceCoordinates.x) + ", ";
      message += to_string(fiberTraceCoordinates.y) + ")";
      LOGLS_DEBUG(_log, message);
      fiberTraceCoordinatesRelativeToCenter.x = fiberTraceCoordinates.x - traceCoordinatesCenter.x;
      fiberTraceCoordinatesRelativeToCenter.y = fiberTraceCoordinates.y - traceCoordinatesCenter.y;
      message = "fiberTraceCoordinatesRelativeToCenter = (";
      message += to_string(fiberTraceCoordinatesRelativeToCenter.x) + ", ";
      message += to_string(fiberTraceCoordinatesRelativeToCenter.y) + ")";
      LOGLS_DEBUG(_log, message);

      return fiberTraceCoordinatesRelativeToCenter;
    }
  }

  namespace utils{

    template<typename T>
    const T* getRawPointer(const PTR(const T) & ptr){
      return ptr.get();
    }

    template< typename ImageT, typename MaskT, typename VarianceT >
    bool markFiberTraceInMask( PTR( FiberTrace< ImageT, MaskT, VarianceT > ) const& fiberTrace,
                               PTR( afwImage::Mask< MaskT > ) const& mask,
                               MaskT value){
      const PTR(const FiberTraceFunction) ftf = fiberTrace->getFiberTraceFunction();
      const ndarray::Array< float, 1, 1 > xCenters = fiberTrace->getXCenters();
      ndarray::Array<size_t, 2, 1> minCenMax = pfsDRPStella::math::calcMinCenMax( xCenters,
                                                                                  ftf->fiberTraceFunctionControl.xHigh,
                                                                                  ftf->fiberTraceFunctionControl.xLow,
                                                                                  ftf->fiberTraceFunctionControl.nPixCutLeft,
                                                                                  ftf->fiberTraceFunctionControl.nPixCutRight);
      int y = ftf->yCenter + ftf->yLow;
      for (int iY = 0; iY < xCenters.getShape()[ 0 ]; ++iY){
        for (int x = minCenMax[iY][0]; x <= minCenMax[iY][2]; ++x){
          (*mask)(x, y + iY) |= value;
        }
      }
      return true;
    }

  }
}}}

template void pfsDRPStella::math::assignITrace( pfsDRPStella::FiberTraceSet< float, unsigned short, float > &,
                                                ndarray::Array< int, 1, 0 > const&,
                                                ndarray::Array< float, 1, 0 > const& );
template void pfsDRPStella::math::assignITrace( pfsDRPStella::FiberTraceSet< float, unsigned short, float > &,
                                                ndarray::Array< int, 1, 0 > const&,
                                                ndarray::Array< double, 1, 0 > const& );
template void pfsDRPStella::math::assignITrace( pfsDRPStella::FiberTraceSet< float, unsigned short, float > &,
                                                ndarray::Array< short int, 1, 0 > const&,
                                                ndarray::Array< float, 1, 0 > const& );
template void pfsDRPStella::math::assignITrace( pfsDRPStella::FiberTraceSet< float, unsigned short, float > &,
                                                ndarray::Array< short int, 1, 0 > const&,
                                                ndarray::Array< double, 1, 0 > const& );
template void pfsDRPStella::math::assignITrace( pfsDRPStella::FiberTraceSet< float, unsigned short, float > &,
                                                ndarray::Array< long int, 1, 0 > const&,
                                                ndarray::Array< float, 1, 0 > const& );
template void pfsDRPStella::math::assignITrace( pfsDRPStella::FiberTraceSet< float, unsigned short, float > &,
                                                ndarray::Array< long int, 1, 0 > const&,
                                                ndarray::Array< double, 1, 0 > const& );

template void pfsDRPStella::math::assignITrace( pfsDRPStella::FiberTraceSet< float, unsigned short, float > &,
                                                ndarray::Array< int, 1, 1 > const&,
                                                ndarray::Array< float, 1, 1 > const& );
template void pfsDRPStella::math::assignITrace( pfsDRPStella::FiberTraceSet< float, unsigned short, float > &,
                                                ndarray::Array< int, 1, 1 > const&,
                                                ndarray::Array< double, 1, 1 > const& );
template void pfsDRPStella::math::assignITrace( pfsDRPStella::FiberTraceSet< float, unsigned short, float > &,
                                                ndarray::Array< short int, 1, 1 > const&,
                                                ndarray::Array< float, 1, 1 > const& );
template void pfsDRPStella::math::assignITrace( pfsDRPStella::FiberTraceSet< float, unsigned short, float > &,
                                                ndarray::Array< short int, 1, 1 > const&,
                                                ndarray::Array< double, 1, 1 > const& );
template void pfsDRPStella::math::assignITrace( pfsDRPStella::FiberTraceSet< float, unsigned short, float > &,
                                                ndarray::Array< long int, 1, 1 > const&,
                                                ndarray::Array< float, 1, 1 > const& );
template void pfsDRPStella::math::assignITrace( pfsDRPStella::FiberTraceSet< float, unsigned short, float > &,
                                                ndarray::Array< long int, 1, 1 > const&,
                                                ndarray::Array< double, 1, 1 > const& );

template size_t pfsDRPStella::math::findITrace( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                ndarray::Array< float, 1, 0 > const&,
                                                size_t,
                                                size_t,
                                                size_t );
template size_t pfsDRPStella::math::findITrace( pfsDRPStella::FiberTrace< double, unsigned short, double > const&,
                                                ndarray::Array< float, 1, 0 > const&,
                                                size_t,
                                                size_t,
                                                size_t );
template size_t pfsDRPStella::math::findITrace( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                ndarray::Array< double, 1, 0 > const&,
                                                size_t,
                                                size_t,
                                                size_t );
template size_t pfsDRPStella::math::findITrace( pfsDRPStella::FiberTrace< double, unsigned short, double > const&,
                                                ndarray::Array< double, 1, 0 > const&,
                                                size_t,
                                                size_t,
                                                size_t );

template pfs::drp::stella::math::dataXY<float> pfsDRPStella::math::ccdToFiberTraceCoordinates(
    pfs::drp::stella::math::dataXY<float> const&,
    pfsDRPStella::FiberTrace<float, unsigned short, float> const&
);
template pfs::drp::stella::math::dataXY<double> pfsDRPStella::math::ccdToFiberTraceCoordinates(
    pfs::drp::stella::math::dataXY<double> const&,
    pfsDRPStella::FiberTrace<float, unsigned short, float> const&
);
template pfs::drp::stella::math::dataXY<float> pfsDRPStella::math::ccdToFiberTraceCoordinates(
    pfs::drp::stella::math::dataXY<float> const&,
    pfsDRPStella::FiberTrace<double, unsigned short, float> const&
);
template pfs::drp::stella::math::dataXY<double> pfsDRPStella::math::ccdToFiberTraceCoordinates(
    pfs::drp::stella::math::dataXY<double> const&,
    pfsDRPStella::FiberTrace<double, unsigned short, float> const&
);

template pfs::drp::stella::math::dataXY<float> pfsDRPStella::math::fiberTraceCoordinatesRelativeTo(
    pfs::drp::stella::math::dataXY<float> const&,
    pfs::drp::stella::math::dataXY<float> const&,
    pfs::drp::stella::FiberTrace<float, unsigned short, float> const&
);
template pfs::drp::stella::math::dataXY<double> pfsDRPStella::math::fiberTraceCoordinatesRelativeTo(
    pfs::drp::stella::math::dataXY<double> const&,
    pfs::drp::stella::math::dataXY<double> const&,
    pfs::drp::stella::FiberTrace<float, unsigned short, float> const&
);
template pfs::drp::stella::math::dataXY<float> pfsDRPStella::math::fiberTraceCoordinatesRelativeTo(
    pfs::drp::stella::math::dataXY<float> const&,
    pfs::drp::stella::math::dataXY<float> const&,
    pfs::drp::stella::FiberTrace<double, unsigned short, float> const&
);
template pfs::drp::stella::math::dataXY<double> pfsDRPStella::math::fiberTraceCoordinatesRelativeTo(
    pfs::drp::stella::math::dataXY<double> const&,
    pfs::drp::stella::math::dataXY<double> const&,
    pfs::drp::stella::FiberTrace<double, unsigned short, float> const&
);

template class pfsDRPStella::FiberTrace<float, unsigned short, float>;
template class pfsDRPStella::FiberTrace<double, unsigned short, float>;

template class pfsDRPStella::FiberTraceSet<float, unsigned short, float>;
template class pfsDRPStella::FiberTraceSet<double, unsigned short, float>;

template PTR(pfsDRPStella::FiberTraceSet<float, unsigned short, float>) pfsDRPStella::math::findAndTraceApertures(PTR(const afwImage::MaskedImage<float, unsigned short, float>) const&,
                                                                                           PTR(const pfsDRPStella::FiberTraceFunctionFindingControl) const&);
template PTR(pfsDRPStella::FiberTraceSet<double, unsigned short, float>) pfsDRPStella::math::findAndTraceApertures(PTR(const afwImage::MaskedImage<double, unsigned short, float>) const&,
                                                                                            PTR(const pfsDRPStella::FiberTraceFunctionFindingControl) const&);

template pfsDRPStella::math::FindCenterPositionsOneTraceResult pfsDRPStella::math::findCenterPositionsOneTrace( PTR(afwImage::Image<float>) &,
                                                                                                                PTR(afwImage::Image<float>) &,
                                                                                                                PTR(const FiberTraceFunctionFindingControl) const&);
template pfsDRPStella::math::FindCenterPositionsOneTraceResult pfsDRPStella::math::findCenterPositionsOneTrace( PTR(afwImage::Image<double>) &,
                                                                                                                PTR(afwImage::Image<float>) &,
                                                                                                                PTR(const FiberTraceFunctionFindingControl) const&);

template const afwImage::MaskedImage<float, unsigned short, float>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::MaskedImage<float, unsigned short, float>) &ptr);
template const afwImage::MaskedImage<double, unsigned short, float>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::MaskedImage<double, unsigned short, float>) &ptr);
template const afwImage::Image<float>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<float>) &ptr);
template const afwImage::Image<double>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<double>) &ptr);
template const afwImage::Image<unsigned short>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<unsigned short>) &ptr);
template const afwImage::Image<unsigned int>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<unsigned int>) &ptr);
template const afwImage::Image<int>* pfsDRPStella::utils::getRawPointer(const PTR(const afwImage::Image<int>) &ptr);
template const pfsDRPStella::FiberTrace<float, unsigned short, float>* pfsDRPStella::utils::getRawPointer(const PTR(const pfsDRPStella::FiberTrace<float, unsigned short, float>) &ptr);
template const pfsDRPStella::FiberTrace<double, unsigned short, float>* pfsDRPStella::utils::getRawPointer(const PTR(const pfsDRPStella::FiberTrace<double, unsigned short, float>) &ptr);

template bool pfsDRPStella::utils::markFiberTraceInMask( PTR( FiberTrace< float, unsigned short, float > ) const& fiberTrace,
                                                         PTR( afwImage::Mask< unsigned short > ) const& mask,
                                                         unsigned short value );

template PTR(pfsDRPStella::FiberTrace< float, unsigned short, float >) pfsDRPStella::math::makeNormFlatFiberTrace( PTR( const afwImage::MaskedImage< float, unsigned short, float >) const&,
                                                                                                PTR( const ::pfs::drp::stella::FiberTraceFunction ) const&,
                                                                                                PTR( const ::pfs::drp::stella::FiberTraceFunctionControl ) const&,
                                                                                                PTR( const ::pfs::drp::stella::FiberTraceProfileFittingControl ) const&,
                                                                                                float,
                                                                                                size_t );
template PTR(pfsDRPStella::FiberTrace< double, unsigned short, float >) pfsDRPStella::math::makeNormFlatFiberTrace( PTR( const afwImage::MaskedImage< double, unsigned short, float >) const&,
                                                                                                 PTR( const ::pfs::drp::stella::FiberTraceFunction ) const&,
                                                                                                 PTR( const ::pfs::drp::stella::FiberTraceFunctionControl ) const&,
                                                                                                 PTR( const ::pfs::drp::stella::FiberTraceProfileFittingControl ) const&,
                                                                                                 double,
                                                                                                 size_t );

template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< float > const&,
                                                           ndarray::Array< float, 2, 0 > & );
template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< unsigned short > const&,
                                                           ndarray::Array< unsigned short, 2, 0 > & );
template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< double > const&,
                                                           ndarray::Array< float, 2, 0 > & );
template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< float > const&,
                                                           ndarray::Array< unsigned short, 2, 0 > & );
template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< double > const&,
                                                           ndarray::Array< double, 2, 0 > & );

template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< float > const&,
                                                           ndarray::Array< float, 2, 1 > & );
template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< unsigned short > const&,
                                                           ndarray::Array< unsigned short, 2, 1 > & );
template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< double > const&,
                                                           ndarray::Array< float, 2, 1 > & );
template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< float > const&,
                                                           ndarray::Array< unsigned short, 2, 1 > & );
template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< double > const&,
                                                           ndarray::Array< double, 2, 1 > & );

template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< float > const&,
                                                           ndarray::Array< float, 2, 2 > & );
template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< unsigned short > const&,
                                                           ndarray::Array< unsigned short, 2, 2 > & );
template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< double > const&,
                                                           ndarray::Array< float, 2, 2 > & );
template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< float > const&,
                                                           ndarray::Array< unsigned short, 2, 2 > & );
template void pfsDRPStella::math::addFiberTraceToCcdArray( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< double > const&,
                                                           ndarray::Array< double, 2, 2 > & );

template void pfsDRPStella::math::addFiberTraceToCcdImage( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< float > const&,
                                                           afwImage::Image< float > & );
template void pfsDRPStella::math::addFiberTraceToCcdImage( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< unsigned short > const&,
                                                           afwImage::Image< unsigned short > & );
template void pfsDRPStella::math::addFiberTraceToCcdImage( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< double > const&,
                                                           afwImage::Image< float > & );
template void pfsDRPStella::math::addFiberTraceToCcdImage( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< float > const&,
                                                           afwImage::Image< unsigned short > & );
template void pfsDRPStella::math::addFiberTraceToCcdImage( pfsDRPStella::FiberTrace< float, unsigned short, float > const&,
                                                           afwImage::Image< double > const&,
                                                           afwImage::Image< double > & );

#define INSTANTIATE_ADDARRAYINTOARRAY(T1, T2, I1, I2) \
template void pfsDRPStella::math::addArrayIntoArray( ndarray::Array< T1, 2, I1 > const&, \
                                                     ndarray::Array< size_t, 2, 1 > const&, \
                                                     size_t const&, \
                                                     ndarray::Array< T2, 2, I2 > & ); \

INSTANTIATE_ADDARRAYINTOARRAY(float const, float, 1, 1);
INSTANTIATE_ADDARRAYINTOARRAY(float const, float, 1, 2);
INSTANTIATE_ADDARRAYINTOARRAY(float const, float, 2, 1);
INSTANTIATE_ADDARRAYINTOARRAY(float const, float, 2, 2);
