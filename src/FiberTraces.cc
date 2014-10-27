#include "pfs/drp/stella/FiberTraces.h"

namespace pfsDRPStella = pfs::drp::stella;

  /** @brief Construct an Exposure with a blank MaskedImage of specified size (default 0x0)
   */          
  template<typename ImageT, typename MaskT, typename VarianceT> 
  pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    unsigned int width,                 ///< number of columns
    unsigned int height                ///< number of rows
  ) :
  _trace(10, height),
  _profile(new afwImage::Image<float>(10, height)),
  _ccdWidth(width),
  _ccdHeight(height),
//  _maskedImage(new MaskedImageT(width, height)),
  _xCenters(height),
  _spectrum(height),
  _spectrumVariance(height),
  _background(height),
  _backgroundVariance(height),
  _fiberTraceFunction(), 
  _fiberTraceExtractionControl(new FiberTraceExtractionControl)
  {
    _isXCentersCalculated = false;
//    _isImageSet = false;
    _isTraceSet = false;
    _isProfileSet = false;
    _isSpectrumExtracted = false;
    _isFiberTraceFunctionSet = false;
    _isFiberTraceExtractionControlSet = false;
  }
  
  /** @brief Construct an Exposure with a blank MaskedImage of specified size (default 0x0) and
   * a Wcs (which may be default constructed)
   */          
  template<typename ImageT, typename MaskT, typename VarianceT> 
  pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    afwGeom::Extent2I const & dimensions ///< desired image width/height
  ) :
  _trace(dimensions),
  _profile(new afwImage::Image<float>(dimensions)),
//  _maskedImage(new MaskedImageT(dimensions)),
  _ccdWidth(dimensions.getX()),
  _ccdHeight(dimensions.getY()),
  _xCenters(dimensions.getY()),
  _spectrum(dimensions.getY()),
  _spectrumVariance(dimensions.getY()),
  _background(dimensions.getY()),
  _backgroundVariance(dimensions.getY()),
  _fiberTraceFunction(), 
  _fiberTraceExtractionControl(new FiberTraceExtractionControl)
  {
    _isXCentersCalculated = false;
//    _isImageSet = false;
    _isTraceSet = false;
    _isProfileSet = false;
    _isSpectrumExtracted = false;
    _isFiberTraceFunctionSet = false;
    _isFiberTraceExtractionControlSet = false;
  }

  template<typename ImageT, typename MaskT, typename VarianceT> 
  pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    PTR(MaskedImageT) const & maskedImage ///< desired image width/height
  ) :
  _trace(maskedImage->getWidth(), maskedImage->getHeight()),
  _profile(new afwImage::Image<float>(maskedImage->getWidth(), maskedImage->getHeight())),
//  _maskedImage(maskedImage),
  _ccdWidth(maskedImage->getWidth()),
  _ccdHeight(maskedImage->getHeight()),
  _xCenters(maskedImage->getHeight()),
  _spectrum(maskedImage->getHeight()),
  _spectrumVariance(maskedImage->getHeight()),
  _background(maskedImage->getHeight()),
  _backgroundVariance(maskedImage->getHeight()),
  _fiberTraceFunction(), 
  _fiberTraceExtractionControl(new FiberTraceExtractionControl)
  {
    _isXCentersCalculated = false;
//    _isImageSet = true;
    _isTraceSet = false;
    _isProfileSet = false;
    _isSpectrumExtracted = false;
    _isFiberTraceFunctionSet = false;
    _isFiberTraceExtractionControlSet = false;
  }
  
  /** **************************************************************/
  
  /// Set the CCD image to image
/*  template<typename ImageT, typename MaskT, typename VarianceT> 
  void pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setMaskedImage( PTR(MaskedImageT) const & image){ 
    _maskedImage = image; 
    _isImageSet = true;
    _trace = MaskedImageT(_trace.getWidth(), _ccdHeight);
  }*/
  
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setFiberTraceFunction(const FiberTraceFunction &fiberTraceFunction){
    
    /// Check for valid values in fiberTraceFunctionControl
    bool isFunctionValid = false;
    #ifdef __DEBUG_SETFIBERTRACEFUNCTION__
      cout << "FiberTrace::setFiberTraceFunction: fiberTraceFunction.fiberTraceFunctionControl.interpolation = <" << fiberTraceFunction.fiberTraceFunctionControl.interpolation << ">" << endl;
      cout << "FiberTrace::setFiberTraceFunction: fiberTraceFunction.fiberTraceFunctionControl.order = <" << fiberTraceFunction.fiberTraceFunctionControl.order << ">" << endl;
      cout << "FiberTrace::setFiberTraceFunction: fiberTraceFunction.fiberTraceFunctionControl.xLow = <" << fiberTraceFunction.fiberTraceFunctionControl.xLow << ">" << endl;
      cout << "FiberTrace::setFiberTraceFunction: fiberTraceFunction.fiberTraceFunctionControl.xHigh = <" << fiberTraceFunction.fiberTraceFunctionControl.xHigh << ">" << endl;
      cout << "FiberTrace::setFiberTraceFunction: fiberTraceFunction.xCenter = <" << fiberTraceFunction.xCenter << ">" << endl;
      cout << "FiberTrace::setFiberTraceFunction: fiberTraceFunction.yCenter = <" << fiberTraceFunction.yCenter << ">" << endl;
      cout << "FiberTrace::setFiberTraceFunction: fiberTraceFunction.yLow = <" << fiberTraceFunction.yLow << ">" << endl;
      cout << "FiberTrace::setFiberTraceFunction: fiberTraceFunction.yHigh = <" << fiberTraceFunction.yHigh << ">" << endl;
      cout << "FiberTrace::setFiberTraceFunction: fiberTraceFunction.coefficients = <";
      for (int i = 0; i < static_cast<int>(fiberTraceFunction.coefficients.size()); i++)
        cout << fiberTraceFunction.coefficients[i] << " ";
      cout << ">" << endl;
    #endif
      
    for ( int fooInt = FiberTraceFunctionControl::CHEBYSHEV; fooInt != FiberTraceFunctionControl::NVALUES; fooInt++ ){
      #ifdef __DEBUG_SETFIBERTRACEFUNCTION__
        cout << "FiberTrace::setFiberTraceFunction: INTERPOLATION_NAMES[fooInt] = <" << fiberTraceFunction.fiberTraceFunctionControl.INTERPOLATION_NAMES[fooInt] << ">" << endl;
      #endif
      if (fiberTraceFunction.fiberTraceFunctionControl.interpolation.compare(fiberTraceFunction.fiberTraceFunctionControl.INTERPOLATION_NAMES[fooInt]) == 0){
        isFunctionValid = true;
        #ifdef __DEBUG_SETFIBERTRACEFUNCTION__
          cout << "FiberTrace::setFiberTraceFunction: " << fiberTraceFunction.fiberTraceFunctionControl.interpolation << " is valid" << endl;
        #endif
      }
    }
    if (!isFunctionValid){
      cout << "FiberTrace::setFiberTraceFunction: ERROR: interpolation function is not valid! => Returning FALSE" << endl;
      return false;
    }
    
    if (fiberTraceFunction.fiberTraceFunctionControl.order < 0){
      cout << "FiberTrace::setFiberTraceFunction: ERROR: fiberTraceFunction.fiberTraceFunctionControl.order(=" << fiberTraceFunction.fiberTraceFunctionControl.order << ") < 0 => Returning FALSE" << endl;
      return false;
    }
    
    if (fiberTraceFunction.fiberTraceFunctionControl.xLow > 0.){
      cout << "FiberTrace::setFiberTraceFunction: ERROR: (fiberTraceFunction.fiberTraceFunctionControl.xLow(=" << fiberTraceFunction.fiberTraceFunctionControl.xLow << ") > 0 => Returning FALSE" << endl;
      return false;
    }
    
    if (fiberTraceFunction.fiberTraceFunctionControl.xHigh < 0.){
      cout << "FiberTrace::setFiberTraceFunction: ERROR: (fiberTraceFunction.fiberTraceFunctionControl.xHigh(=" << fiberTraceFunction.fiberTraceFunctionControl.xHigh << ") < 0 => Returning FALSE" << endl;
      return false;
    }
        
    if (fiberTraceFunction.coefficients.size() < fiberTraceFunction.fiberTraceFunctionControl.order){
      cout << "FiberTrace::setFiberTraceFunction: ERROR: fiberTraceFunction.coefficients(= << fiberTraceFunction.coefficients << ).size(=" << fiberTraceFunction.coefficients.size() << ") < fiberTraceFunction.fiberTraceFunctionControl.order(=" << fiberTraceFunction.fiberTraceFunctionControl.order << ") => Returning FALSE" << endl;
      return false;
    }
    
    if (fiberTraceFunction.xCenter < 0.){
      cout << "FiberTrace::setFiberTraceFunction: ERROR: fiberTraceFunction.xCenter(=" << fiberTraceFunction.xCenter << ") < 0 => Returning FALSE" << endl;
      return false;
    }
    
    if (fiberTraceFunction.yCenter < 0.){
      cout << "FiberTrace::setFiberTraceFunction: ERROR: fiberTraceFunction.yCenter(=" << fiberTraceFunction.yCenter << ") < 0 => Returning FALSE" << endl;
      return false;
    }
    
    if (fiberTraceFunction.fiberTraceFunctionControl.xLow + fiberTraceFunction.xCenter < 0.){
      cout << "FiberTrace::setFiberTraceFunction: ERROR: (fiberTraceFunction.fiberTraceFunctionControl.xLow(=" << fiberTraceFunction.fiberTraceFunctionControl.xLow << ") + fiberTraceFunction.xCenter(=" << fiberTraceFunction.xCenter << ") = " << fiberTraceFunction.fiberTraceFunctionControl.xLow + fiberTraceFunction.xCenter << " < 0 => Returning FALSE" << endl;
      return false;
    }

    if (fiberTraceFunction.yLow > 0.){
      cout << "FiberTrace::setFiberTraceFunction: ERROR: (fiberTraceFunction.yLow(=" << fiberTraceFunction.yLow << ") > 0 => Returning FALSE" << endl;
      return false;
    }
    
    if (fiberTraceFunction.yLow + fiberTraceFunction.yCenter < 0.){
      cout << "FiberTrace::setFiberTraceFunction: ERROR: (fiberTraceFunction.yLow(=" << fiberTraceFunction.yLow << ") + fiberTraceFunction.yCenter(=" << fiberTraceFunction.yCenter << ") = " << fiberTraceFunction.yLow + fiberTraceFunction.yCenter << " < 0 => Returning FALSE" << endl;
      return false;
    }
    
    if (fiberTraceFunction.yHigh < 0.){
      cout << "FiberTrace::setFiberTraceFunction: ERROR: (fiberTraceFunction.yHigh(=" << fiberTraceFunction.yHigh << ") < 0 => Returning FALSE" << endl;
      return false;
    }
    
    /// test passed -> copy fiberTraceFunctionControl to _fiberTraceFunctionControl
    _fiberTraceFunction = fiberTraceFunction;
    _isFiberTraceFunctionSet = true;
    
    return true;
  }
  
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setFiberTraceExtractionControl(PTR(FiberTraceExtractionControl) fiberTraceExtractionControl){
    
    /// Check for valid values in fiberTraceFunctionControl
    bool isTelluricValid = false;
    #ifdef __DEBUG_SETFIBERTRACEEXTRACTIONCONTROL__
      cout << "FiberTrace::setFiberTraceExtractionControl: fiberTraceExtractionControl->profileInterpolation = <" << fiberTraceExtractionControl->profileInterpolation << ">" << endl;
      cout << "FiberTrace::setFiberTraceExtractionControl: fiberTraceExtractionControl->ccdReadOutNoise = <" << fiberTraceExtractionControl->ccdReadOutNoise << ">" << endl;
      cout << "FiberTrace::setFiberTraceExtractionControl: fiberTraceExtractionControl->swathWidth = <" << fiberTraceExtractionControl->swathWidth << ">" << endl;
      cout << "FiberTrace::setFiberTraceExtractionControl: fiberTraceExtractionControl->telluric = <" << fiberTraceExtractionControl->telluric << ">" << endl;
      cout << "FiberTrace::setFiberTraceExtractionControl: fiberTraceExtractionControl->overSample = <" << fiberTraceExtractionControl->overSample << ">" << endl;
      cout << "FiberTrace::setFiberTraceExtractionControl: fiberTraceExtractionControl->maxIterSF = <" << fiberTraceExtractionControl->maxIterSF << ">" << endl;
      cout << "FiberTrace::setFiberTraceExtractionControl: fiberTraceExtractionControl->maxIterSig = <" << fiberTraceExtractionControl->maxIterSig << ">" << endl;
      cout << "FiberTrace::setFiberTraceExtractionControl: fiberTraceExtractionControl->maxIterSky = <" << fiberTraceExtractionControl->maxIterSky << ">" << endl;
      cout << "FiberTrace::setFiberTraceExtractionControl: fiberTraceExtractionControl->lambdaSF = <" << fiberTraceExtractionControl->lambdaSF << ">" << endl;
      cout << "FiberTrace::setFiberTraceExtractionControl: fiberTraceExtractionControl->lambdaSP = <" << fiberTraceExtractionControl->lambdaSP << ">" << endl;
      cout << "FiberTrace::setFiberTraceExtractionControl: fiberTraceExtractionControl->wingSmoothFactor = <" << fiberTraceExtractionControl->wingSmoothFactor << ">" << endl;
      //cout << "FiberTrace::setFiberTraceExtractionControl: fiberTraceExtractionControl->xCorProf = <" << fiberTraceExtractionControl->xCorProf << ">" << endl;
    #endif

    int isProfileInterpolationValid = false;
    for ( int fooInt = FiberTraceExtractionControl::PISKUNOV; fooInt != FiberTraceExtractionControl::NVALUES_P; fooInt++ ){
      #ifdef __DEBUG_SETFIBERTRACEEXTRACTIONCONTROL__
        cout << "FiberTrace::setFiberTraceExtractionControl: PROFILE_INTERPOLATION_NAMES[fooInt] = <" << fiberTraceExtractionControl->PROFILE_INTERPOLATION_NAMES[fooInt] << ">" << endl;
      #endif
      if (fiberTraceExtractionControl->profileInterpolation.compare(fiberTraceExtractionControl->PROFILE_INTERPOLATION_NAMES[fooInt]) == 0){
        isProfileInterpolationValid = true;
        #ifdef __DEBUG_SETFIBERTRACEEXTRACTIONCONTROL__
          cout << "FiberTrace::setFiberTraceExtractionControl: " << fiberTraceExtractionControl->profileInterpolation << " is valid" << endl;
        #endif
      }
    }
    
    if (!isProfileInterpolationValid){
      cout << "FiberTrace::setFiberTraceExtractionControl: ERROR: fiberTraceExtractionControl.profileInterpolation is not valid! => Returning FALSE" << endl;
      return false;
    }
      
    for ( int fooInt = FiberTraceExtractionControl::NONE; fooInt != FiberTraceExtractionControl::NVALUES; fooInt++ ){
      #ifdef __DEBUG_SETFIBERTRACEEXTRACTIONCONTROL__
        cout << "FiberTrace::setFiberTraceExtractionControl: TELLURIC_NAMES[fooInt] = <" << fiberTraceExtractionControl->TELLURIC_NAMES[fooInt] << ">" << endl;
      #endif
      if (fiberTraceExtractionControl->telluric.compare(fiberTraceExtractionControl->TELLURIC_NAMES[fooInt]) == 0){
        isTelluricValid = true;
        #ifdef __DEBUG_SETFIBERTRACEEXTRACTIONCONTROL__
          cout << "FiberTrace::setFiberTraceExtractionControl: " << fiberTraceExtractionControl->telluric << " is valid" << endl;
        #endif
      }
    }
    if (!isTelluricValid){
      cout << "FiberTrace::setFiberTraceExtractionControl: ERROR: telluric(=" << fiberTraceExtractionControl->telluric << ") is not valid! => Returning FALSE" << endl;
      return false;
    }
    
    if (fiberTraceExtractionControl->ccdReadOutNoise < 0.){
      cout << "FiberTrace::setFiberTraceExtractionControl: ERROR: fiberTraceExtractionControl->ccdReadOutNoise(=" << fiberTraceExtractionControl->ccdReadOutNoise << ") < 0 => Returning FALSE" << endl;
      return false;
    }
    
    if (fiberTraceExtractionControl->overSample == 0){
      cout << "FiberTrace::setFiberTraceExtractionControl: ERROR: (fiberTraceExtractionControl->overSample(=" << fiberTraceExtractionControl->overSample << ") == 0 => Returning FALSE" << endl;
      return false;
    }
    
    if (fiberTraceExtractionControl->maxIterSF == 0){
      cout << "FiberTrace::setFiberTraceExtractionControl: ERROR: (fiberTraceExtractionControl->maxIterSF(=" << fiberTraceExtractionControl->maxIterSF << ") == 0 => Returning FALSE" << endl;
      return false;
    }
    
    if ((fiberTraceExtractionControl->telluric.compare(fiberTraceExtractionControl->TELLURIC_NAMES[0]) != 0) && (fiberTraceExtractionControl->maxIterSky == 0)){
      cout << "FiberTrace::setFiberTraceExtractionControl: ERROR: telluric set to not NONE and (fiberTraceExtractionControl->maxIterSky(=" << fiberTraceExtractionControl->maxIterSky << ") == 0 => Returning FALSE" << endl;
      return false;
    }
    
    if (fiberTraceExtractionControl->lambdaSF < 0.){
      cout << "FiberTrace::setFiberTraceExtractionControl: ERROR: (fiberTraceExtractionControl->lambdaSF(=" << fiberTraceExtractionControl->lambdaSF << ") < 0. => Returning FALSE" << endl;
      return false;
    }
    
    if (fiberTraceExtractionControl->lambdaSP < 0.){
      cout << "FiberTrace::setFiberTraceExtractionControl: ERROR: (fiberTraceExtractionControl->lambdaSP(=" << fiberTraceExtractionControl->lambdaSP << ") < 0. => Returning FALSE" << endl;
      return false;
    }
    
    if (fiberTraceExtractionControl->wingSmoothFactor < 0.){
      cout << "FiberTrace::setFiberTraceExtractionControl: ERROR: (fiberTraceExtractionControl->wingSmoothFactor(=" << fiberTraceExtractionControl->wingSmoothFactor << ") < 0. => Returning FALSE" << endl;
      return false;
    }
    
    /// test passed -> copy fiberTraceExtractionControl to _fiberTraceExtractionControl
    _fiberTraceExtractionControl.reset();
    _fiberTraceExtractionControl = fiberTraceExtractionControl;
    _isFiberTraceExtractionControlSet = true;
    
    return true;
  }
  
  /** *******************************************************************************************************/
  
  /// Calculate the x-centers of the fiber trace
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::calculateXCenters(){
//    if (!_isImageSet){
//      cout << "FiberTrace::calculateXCenters: ERROR: _maskedImage is not set => returning FALSE" << endl;
//      return false;
//    }
    
    int cIndex = 0;
    double D_YMin = 0.;
    double D_YMax = 0.;
      
    blitz::Array<double, 1> D_A1_TempCen(_ccdHeight);
    D_A1_TempCen = 0.;
      
    blitz::Array<double, 1> fiberTraceFunctionCoefficients(_fiberTraceFunction.coefficients.data(), blitz::shape(_fiberTraceFunction.coefficients.size()), blitz::neverDeleteData);
//    blitz::Array<double, 1> D_A1_TempCoef(_fiberTraceFunction.coefficients.size());// = fiberTraceFunctionCoefficients;
      
    #ifdef __DEBUG_TRACEFUNC__
      cout << "FiberTrace.calculateXCenters: _fiberTraceFunction.fiberTraceFunctionControl.interpolation = " << _fiberTraceFunction.fiberTraceFunctionControl.interpolation << endl;
      cout << "FiberTrace.calculateXCenters: _fiberTraceFunction.fiberTraceFunctionControl.order = " << _fiberTraceFunction.fiberTraceFunctionControl.order << endl;
    #endif
    if (_fiberTraceFunction.fiberTraceFunctionControl.interpolation.compare("LEGENDRE") == 0){
      #ifdef __DEBUG_TRACEFUNC__
        cout << "FiberTrace.calculateXCenters: Function = LEGENDRE" << endl;
        cout << "FiberTrace.calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
      #endif
      if (!pfsDRPStella::math::Legendre(D_A1_TempCen,
                                        D_YMin,
                                        D_YMax,
                                        fiberTraceFunctionCoefficients,
                                        static_cast<double>(_fiberTraceFunction.xCenter)+1.,
                                        static_cast<double>(_fiberTraceFunction.yCenter)+1.,
                                        1.,//double(int(_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow)+1),
                                        double(_ccdHeight),//double(int(_fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh)+1),
                                        static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xLow),
                                        static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xHigh),
                                        _fiberTraceFunction.fiberTraceFunctionControl.order,
                                        _ccdWidth,
                                        _ccdHeight)){
        cout << "FiberTrace.calculateXCenters: yCenter = " << _fiberTraceFunction.yCenter << endl;
        cout << "FiberTrace.calculateXCenters: yLow = " << _fiberTraceFunction.yLow << endl;
        cout << "FiberTrace.calculateXCenters: yHigh = " << _fiberTraceFunction.yHigh << endl;
        cout << "FiberTrace.calculateXCenters: ERROR: Legendre(...) returned FALSE!!!" << endl;
        return false;
      }
      #ifdef __DEBUG_TRACEFUNC__
        cout << "FiberTrace.calculateXCenters: Legendre: D_YMin set to " << D_YMin << endl;
        cout << "FiberTrace.calculateXCenters: Legendre: D_YMax set to " << D_YMax << endl;
      #endif
      
      D_A1_TempCen -= 1.;
      #ifdef __DEBUG_TRACEFUNC__
        cout << "FiberTrace.calculateXCenters: Legendre: D_A1_TempCen set to " << D_A1_TempCen << endl;
      #endif
    }
    else if (_fiberTraceFunction.fiberTraceFunctionControl.interpolation.compare("CHEBYSHEV") == 0)
    {
      #ifdef __DEBUG_TRACEFUNC__
        cout << "FiberTrace.calculateXCenters: Function = Chebyshev" << endl;
        cout << "FiberTrace.calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
      #endif
      if (!pfsDRPStella::math::Chebyshev(D_A1_TempCen,
                                         D_YMin,
                                         D_YMax,
                                         fiberTraceFunctionCoefficients,
                                         static_cast<double>(_fiberTraceFunction.xCenter)+1.,
                                         static_cast<double>(_fiberTraceFunction.yCenter)+1.,
                                         1.,//double(static_cast<int>(_fiberTraceFunction.yCenter) + _fiberTraceFunction.yLow + 1),
                                         _ccdHeight,//double(static_cast<int>(_fiberTraceFunction.yCenter) + static_cast<int>(_fiberTraceFunction.yHigh)+1),
                                         static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xLow),
                                         static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xHigh),
                                         static_cast<int>(_fiberTraceFunction.fiberTraceFunctionControl.order),
                                         _ccdWidth,
                                         _ccdHeight)){
        cout << "FiberTrace.calculateXCenters: ERROR: Chebyshev(...) returned FALSE!!!" << endl;
        return false;
      }
      D_A1_TempCen -= 1.;
    }
    else if (_fiberTraceFunction.fiberTraceFunctionControl.interpolation.compare("LINEAR") == 0){
      #ifdef __DEBUG_TRACEFUNC__
        cout << "FiberTrace.calculateXCenters: Function = spline1" << endl;
        cout << "FiberTrace.calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
      #endif
      if (!pfsDRPStella::math::LinearSpline(D_A1_TempCen,
                                            fiberTraceFunctionCoefficients,
                                            static_cast<double>(_fiberTraceFunction.xCenter)+1.,
                                            static_cast<double>(_fiberTraceFunction.yCenter)+1.,
                                            1.,//double(static_cast<int>(_fiberTraceFunction.yCenter) + _fiberTraceFunction.yLow + 1),
                                            _ccdHeight,//double(static_cast<int>(_fiberTraceFunction.yCenter) + static_cast<int>(_fiberTraceFunction.yHigh)+1),
                                            static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xLow),
                                            static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xHigh),
                                            static_cast<int>(_fiberTraceFunction.fiberTraceFunctionControl.order),
                                            _ccdWidth,
                                            _ccdHeight)){
        cout << "FiberTrace.calculateXCenters: ERROR: LinearSpline(...) returned FALSE!!!" << endl;
        return false;
      }
      D_A1_TempCen -= 1.;
    }
    else if (_fiberTraceFunction.fiberTraceFunctionControl.interpolation.compare("CUBIC") == 0){
      #ifdef __DEBUG_TRACEFUNC__
        cout << "FiberTrace.calculateXCenters: Function = spline3" << endl;
        cout << "FiberTrace.calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
      #endif
      if (!pfsDRPStella::math::CubicSpline(D_A1_TempCen,
                                           fiberTraceFunctionCoefficients,
                                           static_cast<double>(_fiberTraceFunction.xCenter) + 1.,
                                           static_cast<double>(_fiberTraceFunction.yCenter) + 1.,
                                           1.,//double(static_cast<int>(_fiberTraceFunction.yCenter) + _fiberTraceFunction.yLow + 1),
                                           _ccdHeight,//double(static_cast<int>(_fiberTraceFunction.yCenter) + static_cast<int>(_fiberTraceFunction.yHigh) + 1),
                                           static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xLow),
                                           static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xHigh),
                                           static_cast<int>(_fiberTraceFunction.fiberTraceFunctionControl.order),
                                           _ccdWidth,
                                           _ccdHeight)){
        cout << "FiberTrace.calculateXCenters: ERROR: CubicSpline(...) returned FALSE!!!" << endl;
        return false;
      }
      D_A1_TempCen -= 1.;
    }
    else /// Polynomial
    {
      #ifdef __DEBUG_TRACEFUNC__
        cout << "FiberTrace.calculateXCenters: Function = Polynomial" << endl;
        cout << "FiberTrace.calculateXCenters: Coeffs = " << fiberTraceFunctionCoefficients << endl;
      #endif
      blitz::Array<double,1> D_A1_XRow(_ccdHeight);
      for (int i=0; i < _ccdHeight; i++){
        D_A1_XRow(i) = double(i);
      }
      #ifdef __DEBUG_TRACEFUNC__
        cout << "FiberTrace.calculateXCenters: fiberTraceFunctionCoefficients = " << fiberTraceFunctionCoefficients << endl;
      #endif
      D_A1_TempCen = pfsDRPStella::math::Poly(D_A1_XRow, fiberTraceFunctionCoefficients);
    }
    
    /// Check limits
    blitz::Array<double, 1> D_A1_XLow(_ccdHeight);
    D_A1_XLow = D_A1_TempCen + _fiberTraceFunction.fiberTraceFunctionControl.xLow;
    blitz::Array<double, 1> D_A1_XHigh(_ccdHeight);
    D_A1_XHigh = D_A1_TempCen + _fiberTraceFunction.fiberTraceFunctionControl.xHigh;
    blitz::Array<int, 1> I_A1_Where(_ccdHeight);
    I_A1_Where = blitz::where(D_A1_XLow < -0.5, 0, 1);
    int I_NInd;
    blitz::Array<int, 1> *P_I_A1_WhereInd = pfsDRPStella::math::GetIndex(I_A1_Where, I_NInd);
    D_YMin = (*P_I_A1_WhereInd)(0);
    delete(P_I_A1_WhereInd);
    I_A1_Where = blitz::where(D_A1_XHigh < _ccdWidth-0.5, 1, 0);
    P_I_A1_WhereInd = pfsDRPStella::math::GetIndex(I_A1_Where, I_NInd);
    D_YMax = (*P_I_A1_WhereInd)(P_I_A1_WhereInd->size()-1);
    delete(P_I_A1_WhereInd);
    
    #ifdef __DEBUG_TRACEFUNC__
      /// Coefficients for the trace functions
      cout << "FiberTrace.calculateXCenters: fiberTraceFunctionCoefficients = " << fiberTraceFunctionCoefficients << endl;
    
      /// Centres of the apertures in x (cols)
      cout << "FiberTrace.calculateXCenters: _fiberTraceFunction.xCenter set to " << _fiberTraceFunction.xCenter << endl;
      
      /// center position in y (row) for every aperture
      cout << "FiberTrace.calculateXCenters: _fiberTraceFunction.yCenter set to " << _fiberTraceFunction.yCenter << endl;
      
      /// lower aperture limit x (cols)
      cout << "FiberTrace.calculateXCenters: _fiberTraceFunction.fiberTraceFunctionControl.xLow set to " << _fiberTraceFunction.fiberTraceFunctionControl.xLow << endl;
    
      /// higher aperture limit x (cols)
      cout << "FiberTrace.calculateXCenters: _fiberTraceFunction.fiberTraceFunctionControl.xHigh set to " << _fiberTraceFunction.fiberTraceFunctionControl.xHigh << endl;
    
      /// lower aperture limit for every aperture y, rows
      cout << "FiberTrace.calculateXCenters: _fiberTraceFunction.yLow set to " << _fiberTraceFunction.yLow << endl;
    
      /// higher aperture limit for every aperture y, rows
      cout << "FiberTrace.calculateXCenters: _fiberTraceFunction.yHigh set to " << _fiberTraceFunction.yHigh << endl;
    
      /// order of aperture trace function
      cout << "FiberTrace.calculateXCenters: _fiberTraceFunction.fiberTraceFunctionControl.order set to " << _fiberTraceFunction.fiberTraceFunctionControl.order << endl;
    
      /// Name of function used to trace the apertures
      ///  0: chebyshev
      ///  1: legendre
      ///  2: cubic
      ///  3: linear
      ///  4: polynomial
      cout << "FiberTrace.calculateXCenters: _fiberTraceFunction.FiberTraceFunctionControl.interpolation set to " << _fiberTraceFunction.fiberTraceFunctionControl.interpolation << endl;

      cout << "FiberTrace.calculateXCenters: D_YMin set to " << D_YMin << endl;
      cout << "FiberTrace.calculateXCenters: D_YMax set to " << D_YMax << endl;
      
      cout << "FiberTrace.calculateXCenters: D_A1_TempCen set to " << D_A1_TempCen << endl;
      return false;
    #endif
      
    if (D_YMin - _fiberTraceFunction.yCenter > _fiberTraceFunction.yLow)
      _fiberTraceFunction.yLow = D_YMin - _fiberTraceFunction.yCenter;
    if (D_YMax - _fiberTraceFunction.yCenter < _fiberTraceFunction.yHigh)
      _fiberTraceFunction.yHigh = D_YMax - _fiberTraceFunction.yCenter;
    
    cout << "FiberTrace.calculateXCenters: yCenter = " << _fiberTraceFunction.yCenter << endl;
    cout << "FiberTrace.calculateXCenters: yLow = " << _fiberTraceFunction.yLow << endl;
    cout << "FiberTrace.calculateXCenters: yHigh = " << _fiberTraceFunction.yHigh << endl;
    int xLowTooLowInLastRow = 2;
    int xHighTooHighInLastRow = 2;
    for (int i=_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow; i <= int(_fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh); i++){
      if (D_A1_XLow(i) < -0.5){
        cout << "FiberTrace.calculateXCenters: D_A1_XLow(" << i << ") = " << D_A1_XLow(i) << " < 0." << endl;
        if (xLowTooLowInLastRow == 0){
          _fiberTraceFunction.yHigh = i - _fiberTraceFunction.yCenter - 1;
          cout << "FiberTrace.calculateXCenters: xLowTooLowInLastRow == 0: _fiberTraceFunction.yHigh set to " << _fiberTraceFunction.yHigh << endl;
        }
        xLowTooLowInLastRow = 1;
      }
      else{
        if (xLowTooLowInLastRow == 1){
          _fiberTraceFunction.yLow = i - _fiberTraceFunction.yCenter + 1;
          cout << "FiberTrace.calculateXCenters: xLowTooLowInLastRow == 1: _fiberTraceFunction.yLow set to " << _fiberTraceFunction.yLow << endl;
        }
        xLowTooLowInLastRow = 0;
      }
      if (D_A1_XHigh(i) > _ccdWidth-0.5){
        cout << "FiberTrace.calculateXCenters: D_A1_XHigh(" << i << ")=" << D_A1_XHigh(i) << " >= NCols-0.5" << endl;
        if (xHighTooHighInLastRow == 0){
          _fiberTraceFunction.yHigh = i - _fiberTraceFunction.yCenter - 1;
          cout << "FiberTrace.calculateXCenters: xHighTooHighInLastRow == 0: _fiberTraceFunction.yHigh set to " << _fiberTraceFunction.yHigh << endl;
        }
        xHighTooHighInLastRow = 1;
      }
      else{
        if (xHighTooHighInLastRow == 1){
          _fiberTraceFunction.yLow = i - _fiberTraceFunction.yCenter + 1;
          cout << "FiberTrace.calculateXCenters: xHighTooHighInLastRow == 1: _fiberTraceFunction.yLow set to " << _fiberTraceFunction.yLow << endl;
        }
        xHighTooHighInLastRow = 0;
      }
    }
    
    if (D_A1_XLow(_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow) < -0.5){
      cout << "FiberTrace.calculateXCenters: ERROR: D_A1_XLow(_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow=" << _fiberTraceFunction.yCenter + _fiberTraceFunction.yLow << ")=" << D_A1_XLow(_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow) << " < -0.5" << endl;
      return false;
    }
    if (D_A1_XLow(_fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh) < -0.5){
      cout << "FiberTrace.calculateXCenters: ERROR: D_A1_XLow(_fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh=" << _fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh << ")=" << D_A1_XLow(_fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh) << " < -0.5" << endl;
      return false;
    }
    if (D_A1_XHigh(_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow) > _ccdWidth-0.5){
      cout << "FiberTrace.calculateXCenters: ERROR: D_A1_XHigh(_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow=" << _fiberTraceFunction.yCenter + _fiberTraceFunction.yLow << ")=" << D_A1_XHigh(_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow) << " > _ccdWidth-0.5 =" << _ccdWidth-0.5 << endl;
      return false;
    }
    if (D_A1_XHigh(_fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh) > _ccdWidth-0.5){
      cout << "FiberTrace.calculateXCenters: ERROR: D_A1_XHigh(_fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh=" << _fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh << ")=" << D_A1_XHigh(_fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh) << " > _ccdWidth-0.5=" << _ccdWidth-0.5 << endl;
      return false;
    }
    #ifdef __DEBUG_TRACEFUNC__
      cout << "FiberTrace.calculateXCenters: yLow set to " << _fiberTraceFunction.yLow << endl;
      cout << "FiberTrace.calculateXCenters: yHigh set to " << _fiberTraceFunction.yHigh << endl;
    #endif
      
    /// populate _xCenters
    _xCenters.resize(_fiberTraceFunction.yHigh - _fiberTraceFunction.yLow + 1);
    for (int i = static_cast<int>(_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow); i <= static_cast<int>(_fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh); i++) {
      _xCenters[cIndex] = static_cast<float>(D_A1_TempCen(i));
      cIndex++;
    }
    
    D_A1_TempCen.resize(0);
    _isXCentersCalculated = true;
    return true;
  }  
  
  /// Set the x-centers of the fiber trace
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setXCenters(const std::vector<float> &xCenters){
//    if (!_isImageSet){
//      cout << "FiberTrace::setXCenters: ERROR: _maskedImage has not been set" << endl;
//      return false;
//    }
    if (!_isFiberTraceFunctionSet){
      cout << "FiberTrace::setXCenters: ERROR: _fiberTraceFunction has not been set" << endl;
      return false;
    }
    
    /// Check input vector size
    if (xCenters.size() != (_fiberTraceFunction.yHigh - _fiberTraceFunction.yLow + 1)){
      cout << "FiberTrace.setXCenters: ERROR: xCenters.size(=" << xCenters.size() << ") != (_fiberTraceFunction.yHigh - _fiberTraceFunction.yLow + 1)=" << (_fiberTraceFunction.yHigh - _fiberTraceFunction.yLow + 1) << ") => Returning false" << endl;
      return false;
    }
//    if (xCenters.size() != _xCenters.size()){
//      cout << "FiberTrace.setXCenters: ERROR: xCenters.size(=" << xCenters.size() << ") != _xCenters.size(=" << _xCenters.size() << ") => Returning FALSE" << endl;
//      return false;
//    }

    /// Check that xCenters are within image
    for (int i = 0; i < static_cast<int>(xCenters.size()); i++){
      if ((xCenters[i] < -0.5) || (xCenters[i] > _ccdWidth-0.5)){
        cout << "FiberTrace.setXCenters: ERROR: xCenters[" << i << "] = " << xCenters[i] << " outside range" << endl;
        return false;
      }
    }
    
    _xCenters.resize(xCenters.size());
    
    std::vector<float>::iterator iter_xCenters_begin = _xCenters.begin();
    std::vector<float>::const_iterator iter_xCenters_In_begin = xCenters.begin();
    std::vector<float>::const_iterator iter_xCenters_In_end = xCenters.end();
    std::copy(iter_xCenters_In_begin, iter_xCenters_In_end, iter_xCenters_begin);
    
    _isXCentersCalculated = true;
    return true;
  }
  
  /// Set the image pointer of this fiber trace to image
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setImage(PTR(afwImage::Image<ImageT>) image){
    
    /// Check input image size
//    if (_isImageSet){
      if (image->getWidth() != _ccdWidth){
        cout << "FiberTrace.setXCenters: ERROR: image.getWidth(=" << image->getWidth() << ") != _ccdWidth(=" << _ccdWidth << ") => Returning false" << endl;
        return false;
      }
      if (image->getHeight() != _ccdHeight){
        cout << "FiberTrace.setXCenters: ERROR: image.getHeight(=" << image->getHeight() << ") != _ccdHeight(=" << _ccdHeight << ") => Returning false" << endl;
        return false;
      }
//    }

    _trace.getImage() = image;
    
//    _isImageSet = true;
    return true;
  }
  
  /// Set the mask pointer of this fiber trace to mask
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setMask(PTR(afwImage::Mask<MaskT>) mask){
    
    /// Check input mask size
//    if (_isImageSet){
      if (mask->getWidth() != _ccdWidth){
        cout << "FiberTrace.setXCenters: ERROR: mask.getWidth(=" << mask->getWidth() << ") != _ccdWidth(=" << _ccdWidth << ") => Returning false" << endl;
        return false;
      }
      if (mask->getHeight() != _ccdHeight){
        cout << "FiberTrace.setXCenters: ERROR: mask.getHeight(=" << mask->getHeight() << ") != _ccdHeight(=" << _ccdHeight << ") => Returning false" << endl;
        return false;
      }
//    }
    
    _trace.getMask() = mask;
    
    return true;
  }
  
  /// Set the variance pointer of this fiber trace to variance
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setVariance(PTR(afwImage::Image<VarianceT>) variance){
    
    /// Check input variance size
//    if (_isImageSet){
      if (variance->getWidth() != _ccdWidth){
        cout << "FiberTrace.setXCenters: ERROR: variance.getWidth(=" << variance->getWidth() << ") != _maskedImage->getWidth(=" << _ccdWidth << ") => Returning false" << endl;
        return false;
      }
      if (variance->getHeight() != _ccdHeight){
        cout << "FiberTrace.setXCenters: ERROR: variance.getHeight(=" << variance->getHeight() << ") != _maskedImage->getHeight(=" << _ccdHeight << ") => Returning false" << endl;
        return false;
      }
//    }
    
    _trace.getVariance() = variance;
    
    return true;
  }
  
  /// Set the _trace of this fiber trace to trace
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setTrace(MaskedImageT & trace){
    if (!_isFiberTraceFunctionSet){
      cout << "FiberTrace::setTrace: ERROR: _fiberTraceFunction not set => Returning FALSE" << endl;
      return false;
    }
    if (trace.getHeight() > _ccdHeight){
      cout << "FiberTrace::setTrace: ERROR: trace.getHeight(=" << trace.getHeight() << ") > _ccdHeight(=" << _ccdHeight << ") => Returning FALSE" << endl;
      return false;
    }
    if (trace.getWidth() > _ccdWidth){
      cout << "FiberTrace::setTrace: ERROR: trace.getWidth(=" << trace.getWidth() << ") > _ccdWidth(=" << _ccdWidth << ") => Returning FALSE" << endl;
      return false;
    }
    
    /// Check input profile size
    if (trace.getWidth() != _fiberTraceFunction.fiberTraceFunctionControl.xHigh - _fiberTraceFunction.fiberTraceFunctionControl.xLow + 1){
      cout << "FiberTrace.setTrace: ERROR: trace.getWidth(=" << trace.getWidth() << ") != _fiberTraceFunction.fiberTraceFunctionControl.xHigh - _fiberTraceFunction.fiberTraceFunctionControl.xLow + 1(=" << _fiberTraceFunction.fiberTraceFunctionControl.xHigh - _fiberTraceFunction.fiberTraceFunctionControl.xLow + 1 << ") => Returning false" << endl;
      return false;
    }
    if (trace.getHeight() != static_cast<int>(_fiberTraceFunction.yHigh - _fiberTraceFunction.yLow + 1)){
      cout << "FiberTrace.setTrace: ERROR: trace.getHeight(=" << trace.getHeight() << ") != _fiberTraceFunction.yHigh - _fiberTraceFunction.yLow + 1(=" << _fiberTraceFunction.yHigh - _fiberTraceFunction.yLow + 1 << ") => Returning false" << endl;
      return false;
    }
    
    _trace = MaskedImageT(trace.getDimensions());
    _trace = trace;
    
    _isTraceSet = true;
    return true;
  }
  
  /// Set the profile image of this fiber trace to profile
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setProfile(PTR(afwImage::Image<float>) profile){
    if (!_isFiberTraceFunctionSet){
      cout << "FiberTrace::setProfile: ERROR: _fiberTraceFunction not set => Returning FALSE" << endl;
      return false;
    }
    if (!_isTraceSet){
      cout << "FiberTrace::setProfile: ERROR: _trace not set => Returning FALSE" << endl;
      return false;
    }
    
    /// Check input profile size
    if (profile->getWidth() != _trace.getWidth()){
      cout << "FiberTrace.setProfile: ERROR: profile->getWidth(=" << profile->getWidth() << ") != _trace.getWidth(=" << _trace.getWidth() << ") => Returning false" << endl;
      return false;
    }
    if (profile->getHeight() != _trace.getHeight()){
      cout << "FiberTrace.setProfile: ERROR: profile->getHeight(=" << profile->getHeight() << ") != _trace.getHeight(=" << _trace.getHeight() << ") => Returning false" << endl;
      return false;
    }

    _profile.reset();
    _profile = profile;
    
    _isProfileSet = true;
    return true;
  }

  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::extractFromProfile()
  {
    blitz::Array<string, 1> S_A1_Args(1);
    S_A1_Args = "";
    void **PP_Args;
    PP_Args = (void**)malloc(sizeof(void*) * 1);
    
    int I_temp = 0;
    PP_Args[0] = &I_temp;
    
    if (!extractFromProfile(S_A1_Args, PP_Args))
    {
      cout << "CFits::MkSlitFunc(): ERROR: extractFromProfile(S_A1_Args=" << S_A1_Args << ") returned FALSE => Returning FALSE" << endl;
      return false;
    }
    return true;
  }
  
  /// Set the profile image of this fiber trace to profile
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::extractFromProfile(const blitz::Array<string, 1> &S_A1_Args_In,
                                                                              void *ArgV_In[]){
    if (!_isTraceSet){
      cout << "FiberTrace.extractFromProfile: ERROR: _trace is not set" << endl;
      return false;
    }
    if (!_isProfileSet){
      cout << "FiberTrace.extractFromProfile: ERROR: _profile is not set" << endl;
      return false;
    }
    if (_trace.getWidth() != _profile->getWidth()){
      cout << "FiberTrace.extractFromProfile: ERROR: _trace.getWidth(=" << _trace.getWidth() << ") != _profile.getWidth(=" << _profile->getWidth() << ") => Returning FALSE" << endl;
      return false;
    }
    if (_trace.getHeight() != _profile->getHeight()){
      cout << "FiberTrace.extractFromProfile: ERROR: _trace.getHeight(=" << _trace.getHeight() << ") != _profile.getHeight(=" << _profile->getHeight() << ") => Returning FALSE" << endl;
      return false;
    }

    unsigned int fiberTraceNumber = 0;
    
    blitz::Array<std::string, 1> keyWords(1);
    keyWords(0) = std::string("FIBERTRACENUMBER");
    void **args = (void**)malloc(sizeof(void*));
    args[0] = &fiberTraceNumber;
    std::string sTemp = "FIBERTRACENUMBER";
    int I_Pos = 0;
    if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
    {
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "FiberTrace::extractFromProfile: I_Pos = " << I_Pos << endl;
      #endif
      fiberTraceNumber = *(unsigned int*)ArgV_In[I_Pos];
      cout << "FiberTrace::extractFromProfile: KeyWord_Set(FIBERTRACENUMBER): fiberTraceNumber set to " << fiberTraceNumber << endl;
      args[0] = &fiberTraceNumber;
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "args[pppos=" << pppos << "] set to FIBERTRACENUMBER = " << *(unsigned int*)args[pppos] << endl;
      #endif
    }
    
    blitz::Array<float, 2> F_A2_ProfArray = utils::ndarrayToBlitz(_profile->getArray());
    blitz::Array<double, 2> D_A2_ProfArray = pfsDRPStella::math::Double(F_A2_ProfArray);
    blitz::Array<ImageT, 2> T_A2_CCDArray = utils::ndarrayToBlitz(_trace.getImage()->getArray());
    blitz::Array<double, 2> D_A2_CCDArray = pfsDRPStella::math::Double(T_A2_CCDArray);
    blitz::Array<VarianceT, 2> T_A2_VarianceArray = utils::ndarrayToBlitz(_trace.getVariance()->getArray());
    blitz::Array<double, 2> D_A2_ErrArray = pfsDRPStella::math::Double(T_A2_VarianceArray);
    ///TODO: change to sqrt(T_A2_VarianceArray)
    D_A2_ErrArray = sqrt(D_A2_CCDArray);
    blitz::Array<MaskT, 2> T_A2_MaskArray = utils::ndarrayToBlitz(_trace.getMask()->getArray());
    blitz::Array<int, 2> I_A2_MaskArray = pfsDRPStella::math::Int(T_A2_MaskArray);
    I_A2_MaskArray = where(I_A2_MaskArray == 0, 1, 0);
    blitz::Array<double, 1> D_A1_SP(_trace.getHeight());
    D_A1_SP = 0.;
    blitz::Array<string, 1> S_A1_Args_Fit(3);
    void **PP_Args_Fit;
    PP_Args_Fit = (void**)malloc(sizeof(void*) * 3);
    S_A1_Args_Fit = " ";
    
    S_A1_Args_Fit(0) = "MEASURE_ERRORS_IN";
    PP_Args_Fit[0] = &D_A2_ErrArray;
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "CFits::extractFromProfile: D_A2_ErrArray = " << D_A2_ErrArray << endl;
    #endif
    
    S_A1_Args_Fit(1) = "MASK_INOUT";
    PP_Args_Fit[1] = &I_A2_MaskArray;
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "I_A2_MaskArray = " << I_A2_MaskArray << endl;
    #endif
    
    S_A1_Args_Fit(2) = "SIGMA_OUT";
    blitz::Array<double, 2> D_A2_Sigma_Fit(_trace.getHeight(),2);
    PP_Args_Fit[2] = &D_A2_Sigma_Fit;
    
    blitz::Array<double, 1> D_A1_Sky(_trace.getHeight());
    D_A1_Sky = 0.;
    bool B_WithSky = false;
    if (_fiberTraceExtractionControl->telluric.compare(_fiberTraceExtractionControl->TELLURIC_NAMES[0]) != 0){
      D_A1_Sky = 1.;
      B_WithSky = true;
      cout << "extractFromProfile: Sky switched ON" << endl;
    }
    #ifdef __DEBUG_EXTRACTFROMPROFILE__
      cout << "CFits::extractFromProfile: Before Fit: D_A2_CCDArray = " << D_A2_CCDArray << endl;
    #endif
    #ifdef __DEBUG_EXTRACTFROMPROFILE_FILES__
      string S_FileName_CCD_Ap = "CCD_Ap" + to_string(fiberTraceNumber) + "_Tel" + to_string(telluric) + ".fits";
      if (!pfsDRPStella::utils::WriteFits(&D_A2_CCDArray,S_FileName_CCD_Ap)){
        cout << "CFits::extractFromProfile: WriteFits(D_A2_CCD_Ap," << S_FileName_CCD_Ap << ") returned FALSE!" << endl;
        return false;
      }
    #endif
    if (!pfsDRPStella::math::LinFitBevington(D_A2_CCDArray,      ///: in
                                             D_A2_ProfArray,             ///: in
                                             D_A1_SP,             ///: out
                                             D_A1_Sky,          ///: in/out
                                             B_WithSky,                   ///: with sky: in
                                             S_A1_Args_Fit,         ///: in
                                             PP_Args_Fit)){          ///: in/out
      cout << "CFits::extractFromProfile: 2. ERROR: LinFitBevington(...) returned FALSE => Returning FALSE" << endl;
      return false;
    }
    #ifdef __DEBUG_MkSLITFUNC_FILES__
      string S_MaskFinalOut = "Mask_Final" + S_SF_DebugFilesSuffix + ".fits";
      pfsDRPStella::utils::WriteFits(&I_A2_MaskArray, S_MaskFinalOut);
      
      S_MaskFinalOut = "D_A2_CCD_Ap" + CS_SF_DebugFilesSuffix + ".fits";
      pfsDRPStella::utils::WriteFits(&D_A2_CCDArray, S_MaskFinalOut);
    #endif
      
      cout << "Just after Fit: D_A1_SP = " << D_A1_SP << endl;
      cout << "Just after Fit: D_A1_Sky = " << D_A1_Sky << endl;
      cout << "Just after Fit: D_A2_CCDArray = " << D_A2_CCDArray << endl;
      cout << "Just after Fit: D_A2_ProfArray = " << D_A2_ProfArray << endl;

    _spectrum.resize(_trace.getHeight());
    _spectrumVariance.resize(_trace.getHeight());
    _background.resize(_trace.getHeight());
    _backgroundVariance.resize(_trace.getHeight());
    for (int i = 0; i < _trace.getHeight(); i++) {
      _spectrum[i] = static_cast<float>(D_A1_SP(i));
      _spectrumVariance[i] = float(blitz::pow2(D_A2_Sigma_Fit(i, 0)));
      _background[i] = static_cast<float>(D_A1_Sky(i));
      _backgroundVariance[i] = static_cast<float>(pow(D_A2_Sigma_Fit(i, 1),2));
    }
    
    _isSpectrumExtracted = true;
    
    return true;
  }
  
  /**************************************************************************
   * createTrace
   * ************************************************************************/
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::createTrace(PTR(MaskedImageT) const &maskedImage){
//    if (!_isImageSet){
//      cout << "FiberTrace::createTrace: ERROR: _maskedImage has not been set" << endl;
//      return false;
//    }
    if (maskedImage->getWidth() != _ccdWidth){
      cout << "FiberTrace::createTrace: ERROR: maskedImage->getWidth(=" << maskedImage->getWidth() << ") != _ccdWidth(=" << _ccdWidth << ")" << endl;
      return false;
    }
    if (maskedImage->getHeight() != _ccdHeight){
      cout << "FiberTrace::createTrace: ERROR: maskedImage->getHeight(=" << maskedImage->getHeight() << ") != _ccdHeight(=" << _ccdHeight << ")" << endl;
      return false;
    }
    if (!_isXCentersCalculated){
      cout << "FiberTrace::createTrace: ERROR: _xCenters has not been set/calculated" << endl;
      return false;
    }
    
    blitz::Array<int, 2> minCenMax(2,2);
    minCenMax = 0;
    blitz::Array<float, 1> xCenters(_xCenters.data(), blitz::shape(_xCenters.size()), blitz::neverDeleteData);
//    pfsDRPStella::utils::WriteArrayToFile(xCenters, DEBUGDIR + std::string("trace0_xCenters.dat"), std::string("ascii"));
//    return false;
    if (!pfsDRPStella::math::calcMinCenMax(xCenters,
                                           _fiberTraceFunction.fiberTraceFunctionControl.xHigh,
                                           _fiberTraceFunction.fiberTraceFunctionControl.xLow,
                                           _fiberTraceFunction.yCenter,
                                           _fiberTraceFunction.yLow,
                                           _fiberTraceFunction.yHigh,
                                           1,
                                           1,
                                           minCenMax)){
      cout << "FiberTrace::createTrace: ERROR: calcMinCenMax returned FALSE" << endl;
      return false;
    }
//    pfsDRPStella::utils::WriteArrayToFile(minCenMax, DEBUGDIR + std::string("trace0_minCenMax.dat"), std::string("ascii"));
//    return false;
    #ifdef __DEBUG_CREATEFIBERTRACE__
      cout << "FiberTrace::CreateFiberTrace: minCenMax = " << minCenMax << endl;
    #endif
    
    _trace = MaskedImageT(minCenMax(0,2) - minCenMax(0,0) + 1, _fiberTraceFunction.yHigh - _fiberTraceFunction.yLow + 1);// minCenMax.rows());
//    _profile.reset(new afwImage::Image<float>(_trace.getDimensions()));

    ndarray::Array<ImageT, 2, 1> imageArray = maskedImage->getImage()->getArray();
    ndarray::Array<VarianceT, 2, 1> varianceArray = maskedImage->getVariance()->getArray();
    ndarray::Array<MaskT, 2, 1> maskArray = maskedImage->getMask()->getArray();
    ndarray::Array<ImageT, 2, 1> traceImageArray = _trace.getImage()->getArray();
    ndarray::Array<VarianceT, 2, 1> traceVarianceArray = _trace.getVariance()->getArray();
    ndarray::Array<MaskT, 2, 1> traceMaskArray = _trace.getMask()->getArray();
    typename ndarray::Array<ImageT, 2, 1>::Iterator yIterTrace = traceImageArray.begin();
    typename ndarray::Array<VarianceT, 2, 1>::Iterator yIterTraceVariance = traceVarianceArray.begin();
    typename ndarray::Array<MaskT, 2, 1>::Iterator yIterTraceMask = traceMaskArray.begin();
    int iy = 0;//_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow;
    for (iy = 0; iy <= static_cast<int>(_fiberTraceFunction.yHigh - _fiberTraceFunction.yLow); ++iy) {
      typename ndarray::Array<ImageT, 2, 1>::Iterator yIter = imageArray.begin() + _fiberTraceFunction.yCenter + _fiberTraceFunction.yLow + iy;
      typename ndarray::Array<VarianceT, 2, 1>::Iterator yIterV = varianceArray.begin() + _fiberTraceFunction.yCenter + _fiberTraceFunction.yLow + iy;
      typename ndarray::Array<MaskT, 2, 1>::Iterator yIterM = maskArray.begin() + _fiberTraceFunction.yCenter + _fiberTraceFunction.yLow + iy;
      typename ndarray::Array<ImageT, 2, 1>::Reference::Iterator ptrImageStart = yIter->begin() + minCenMax(iy, 0);
      typename ndarray::Array<ImageT, 2, 1>::Reference::Iterator ptrImageEnd = yIter->begin() + minCenMax(iy, 2) + 1;
      typename ndarray::Array<ImageT, 2, 1>::Reference::Iterator ptrTraceStart = yIterTrace->begin();
      std::copy(ptrImageStart, ptrImageEnd, ptrTraceStart);

      typename ndarray::Array<VarianceT, 2, 1>::Reference::Iterator ptrVarianceStart = yIterV->begin() + minCenMax(iy, 0);
      typename ndarray::Array<VarianceT, 2, 1>::Reference::Iterator ptrVarianceEnd = yIterV->begin() + minCenMax(iy, 2) + 1;
      typename ndarray::Array<VarianceT, 2, 1>::Reference::Iterator ptrTraceVarianceStart = yIterTraceVariance->begin();
      std::copy(ptrVarianceStart, ptrVarianceEnd, ptrTraceVarianceStart);
      
      typename ndarray::Array<MaskT, 2, 1>::Reference::Iterator ptrMaskStart = yIterM->begin() + minCenMax(iy, 0);
      typename ndarray::Array<MaskT, 2, 1>::Reference::Iterator ptrMaskEnd = yIterM->begin() + minCenMax(iy, 2) + 1;
      typename ndarray::Array<MaskT, 2, 1>::Reference::Iterator ptrTraceMaskStart = yIterTraceMask->begin();
      std::copy(ptrMaskStart, ptrMaskEnd, ptrTraceMaskStart);
      ++yIterTrace;
      ++yIterTraceVariance;
      ++yIterTraceMask;
    }
    _isTraceSet = true;
    return true;
  }
  
  /// Return shared pointer to an image containing the reconstructed 2D spectrum of the FiberTrace
  template<typename ImageT, typename MaskT, typename VarianceT> 
  afwImage::Image<float> pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::getReconstructed2DSpectrum() const{
    afwImage::Image<float> image(_trace.getWidth(), _trace.getHeight());
    if (_isSpectrumExtracted){
      blitz::Array<float, 2> F_A2_Prof = pfsDRPStella::utils::ndarrayToBlitz(_profile->getArray());
      blitz::Array<float, 2> F_A2_Rec(_trace.getHeight(), _trace.getWidth());
      for (int i_row=0; i_row<_trace.getHeight(); i_row++)
        F_A2_Rec(i_row, blitz::Range::all()) = F_A2_Prof(i_row, blitz::Range::all()) * _spectrum[i_row];
      ndarray::Array<float, 2, 1> ndarrayRec(pfsDRPStella::utils::copyBlitzToNdarray(F_A2_Rec));
      afwImage::Image<float> imRec(ndarrayRec);
      image = imRec;
    }
    return image;
  }
  
  /// Return shared pointer to an image containing the reconstructed background of the FiberTrace
  template<typename ImageT, typename MaskT, typename VarianceT> 
  afwImage::Image<float> pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::getReconstructedBackground() const{
    afwImage::Image<float> image(_trace.getWidth(), _trace.getHeight());
    if (_isSpectrumExtracted){
      blitz::Array<float, 2> F_A2_Rec(_trace.getHeight(), _trace.getWidth());
      for (int i_row=0; i_row<_trace.getHeight(); i_row++)
        F_A2_Rec(i_row, blitz::Range::all()) = _background[i_row];
      ndarray::Array<float, 2, 1> ndarrayRec(pfsDRPStella::utils::copyBlitzToNdarray(F_A2_Rec));
      afwImage::Image<float> imRec(ndarrayRec);
      image = imRec;
    }
    return image;
  }
  
  /**
   *  MkSlitFunc
   *  Make Slit Function
   **/
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::MkSlitFunc(const blitz::Array<string, 1> &S_A1_Args_In,           //: in
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
     *  ! int SWATH_WIDTH=swath_width       => int I_SwathWidth
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
     *  ! long y_lower_lim                   => int 0. - (_fiberTraceFunction.fiberTraceFunctionControl.xLow)(fiberTraceNumber), P_D_A1_XLow(fiberTraceNumber)
     *  ! long y_upper_lim                   => int (*P_D_A1_XHigh)(fiberTraceNumber), P_D_A1_XHigh(fiberTraceNumber)
     *  blitz::Array<double, 1> yslitf
     *  ! long yslitf0 = -y_lower_lim        => int I_XSlitFunc0
     *  ! long yslitf1 =  y_upper_lim        => int I_XSlitFunc1
     **/
    
    if (!_isFiberTraceExtractionControlSet){
      cout << "MkSlitFunc: ERROR: _fiberTraceExtractionControl is not set => Returning FALSE" << endl;
      return false;
    }
    if (!_isTraceSet){
      cout << "MkSlitFunc: ERROR: _trace is not set => Returning FALSE" << endl;
      return false;
    }
    if (!_isXCentersCalculated){
      cout << "MkSlitFunc: ERROR: _fiberTraceExtractionControl is not set => Returning FALSE" << endl;
      return false;
    }
    
    cout << "CFits::MkSlitFunc: Started: S_A1_Args_In = " << S_A1_Args_In << endl;
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "CFits::MkSlitFunc: Started: S_A1_Args_In = " << S_A1_Args_In << endl;
    #endif
    
    string S_SF_DebugFilesSuffix = "";

    blitz::Array<ImageT, 2> T_A2_PixArray = utils::ndarrayToBlitz(_trace.getImage()->getArray());
    blitz::Array<double, 2> D_A2_PixArray = pfsDRPStella::math::Double(T_A2_PixArray);
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "MkSlitFunc: D_A2_PixArray.size = " << D_A2_PixArray.rows() << " rows x " << D_A2_PixArray.cols() << " cols" << endl;
      cout << "MkSlitFunc: D_A2_PixArray(0,*) = " << D_A2_PixArray(0,blitz::Range::all()) << endl;
    #endif
      
    blitz::Array<VarianceT, 2> T_A2_Variance = utils::ndarrayToBlitz(_trace.getVariance()->getArray());
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "MkSlitFunc: T_A2_Variance(0,*) = " << T_A2_Variance(0,blitz::Range::all()) << endl;
    #endif
    ///TODO: change to sqrt(T_A2_Variance)
    blitz::Array<double, 2> D_A2_Errors(D_A2_PixArray.rows(), D_A2_PixArray.cols());
    D_A2_Errors = sqrt(D_A2_PixArray);//pfsDRPStella::math::Double(T_A2_Variance);
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "MkSlitFunc: D_A2_Errors(0,*) = " << D_A2_Errors(0,blitz::Range::all()) << endl;
    #endif
    
    blitz::Array<MaskT, 2> T_A2_MaskArray = utils::ndarrayToBlitz(_trace.getMask()->getArray());
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "MkSlitFunc: T_A2_MaskArray(0,*) = " << T_A2_MaskArray(0,blitz::Range::all()) << endl;
    #endif
    blitz::Array<int, 2> I_A2_MaskArray(D_A2_PixArray.rows(), D_A2_PixArray.cols());// = pfsDRPStella::math::Int(T_A2_MaskArray);
    ///TODO: use _trace.getMask
    I_A2_MaskArray = where(T_A2_MaskArray == 0, 1, 0);
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "MkSlitFunc: I_A2_MaskArray(0,*) = " << I_A2_MaskArray(0,blitz::Range::all()) << endl;
    #endif
    
    blitz::Array<double, 2> D_A2_ProfArray(_trace.getHeight(), _trace.getWidth());
    
    blitz::Array<double, 1> *P_D_A1_BLZ = new blitz::Array<double, 1>(1);
    (*P_D_A1_BLZ) = 0.;
    
//    blitz::Array<double, 1> D_A1_DXScatter(1);
//    D_A1_DXScatter = 0.;
    
    blitz::Array<int, 2> I_A2_IBinBoundY(1,1);
    I_A2_IBinBoundY = 0.;
    
    blitz::Array<double, 1> D_A1_ICol(1);
    D_A1_ICol = 0.;
    
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
    
    D_A2_Errors = sqrt(D_A2_Errors);
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "MkSlitFunc: D_A2_Errors(0,*) = " << D_A2_Errors(0,blitz::Range::all()) << endl;
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
    
    blitz::Array<double, 1> D_A1_XCenMXC(1);
    D_A1_XCenMXC = 0.;
    
    blitz::Array<double, 1> D_A1_XCentersE(1);
    D_A1_XCentersE = 0.;
    
    blitz::Array<double, 1> D_A1_XInt(1);
    D_A1_XInt = 0.;
    
    blitz::Array<double, 1> D_A1_XSlitFTemp(1);
    D_A1_XSlitFTemp = 0.;
    
    blitz::Array<int, 1> I_A1_I(1);
    I_A1_I = 0;
    
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
//    blitz::Array<int, 2> I_A2_MinCenMax(_ccdHeight, 3);
//    I_A2_MinCenMax = 0;
    
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
    
    int I_I = 0;
    int I_NR = 0;// = I_IE - I_IB + 1; /// Number of rows
//    int I_LambdaSP = 1;
    int I_NBins = 0;
    int I_NI = 0;
    int I_NXSF = 0;
    int I_Pos = 0;
    int I_SwathWidth = _fiberTraceExtractionControl->swathWidth;
    //int I_MaxIterSF = _fiberTraceExtractionControl->maxIterSF;
    //int I_MaxIterSky = _fiberTraceExtractionControl->maxIterSky;
    int I_MaxIterSig = _fiberTraceExtractionControl->maxIterSig;
    int I_MaxIterSig_Temp = _fiberTraceExtractionControl->maxIterSig;
    int I_XCorProf = 0;//_fiberTraceExtractionControl->xCorProf;
    double D_ReadOutNoise = _fiberTraceExtractionControl->ccdReadOutNoise;
    //    int overSample_In = _fiberTraceExtractionControl->overSample;
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
    for ( int fooInt = FiberTraceExtractionControl::NONE; fooInt != FiberTraceExtractionControl::NVALUES; fooInt++ ){
      if (_fiberTraceExtractionControl->telluric.compare(_fiberTraceExtractionControl->TELLURIC_NAMES[fooInt]) == 0){
        telluric = fooInt;
      }
    }
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "FiberTrace::MkSlitFunc: telluric set to " << telluric << endl;
    #endif
    
    if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "BLZ")) >= 0)
    {
      if (P_D_A1_BLZ != NULL)
        delete P_D_A1_BLZ;
      P_D_A1_BLZ = (blitz::Array<double, 1>*)ArgV_In[I_Pos];
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: KeyWord_Set(BLZ): P_D_A1_BLZ set to " << *P_D_A1_BLZ << endl;
      #endif
    }
    
    sTemp = "FIBERTRACENUMBER";
    unsigned int fiberTraceNumber = 0;
    if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
    {
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: I_Pos = " << I_Pos << endl;
      #endif
      fiberTraceNumber = *(unsigned int*)ArgV_In[I_Pos];
      cout << "CFits::MkSlitFunc: KeyWord_Set(FIBERTRACENUMBER): fiberTraceNumber set to " << fiberTraceNumber << endl;
      s_a1(pppos) = "FIBERTRACENUMBER";
      args[pppos] = &fiberTraceNumber;
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "args[pppos=" << pppos << "] set to FIBERTRACENUMBER = " << *(unsigned int*)args[pppos] << endl;
      #endif
      pppos++;
    }
    
//    if (fiberTraceExtractionControl->xCorProf > 0)
//      B_Run_XCor = true;

    if (telluric == 3)
      B_MaximaOnly = true;

    blitz::Array<double, 1> D_A1_XCorProf_Out(_trace.getHeight());
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
      cout << "CFits::MkSlitFunc: s_a1 = " << s_a1 << endl;
      cout << "CFits::MkSlitFunc: pppos = " << pppos << endl;
    #endif
    
    int I_Stop = 0;
    
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "CFits::MkSlitFunc: s_a1 set to " << s_a1 << endl;
    #endif
    
    int I_BinHeight;
    
    /// Add 0.5 pixels to get real subpixels TODO: REALLY???????
    blitz::Array<float, 1> xCenters(_xCenters.data(), blitz::shape(_xCenters.size()), blitz::neverDeleteData);
    blitz::Array<double, 1> D_A1_XCenters(xCenters.size());
    D_A1_XCenters = pfsDRPStella::math::Double(xCenters);// + 0.5;
    
    if (I_SwathWidth > _trace.getHeight()){
      I_SwathWidth = _trace.getHeight();
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: KeyWord_Set(SWATH_WIDTH): I_SwathWidth too large: I_SwathWidth set to " << I_SwathWidth << endl;
      #endif
    }
    if (I_SwathWidth != 0)
    {
      I_NBins = pfsDRPStella::math::Round(_trace.getHeight() / I_SwathWidth);
      if (I_NBins < 1)
        I_NBins = 1;
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: KeyWord_Set(SWATH_WIDTH): I_NBin = Round((_fiberTraceFunction.yHigh(=" << _fiberTraceFunction.yHigh << ") - _fiberTraceFunction.yLow(=" << _fiberTraceFunction.yLow << ") + 1.) / I_SwathWidth(=" << I_SwathWidth << ")) set to " << I_NBins << endl;
      #endif
    }
    if (I_SwathWidth == 0)
    { /// Estimate the Points of column crossing
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: !KeyWord_Set(SWATH_WIDTH): D_A1_XCenters = " << D_A1_XCenters << endl;
      #endif
      blitz::Array<int, 1> tempIntArrA = pfsDRPStella::math::Fix(D_A1_XCenters);
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: !KeyWord_Set(SWATH_WIDTH): tempIntArrA = " << tempIntArrA << endl;
      #endif
      pfsDRPStella::math::Uniq(tempIntArrA, I_A1_I);
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: !KeyWord_Set(SWATH_WIDTH): I_A1_I set to " << I_A1_I << endl;
      #endif
      
      ///This is how many times this order crosses to the next column
      I_NI = I_A1_I.size();
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: !KeyWord_Set(SWATH_WIDTH): I_NI set to " << I_NI << endl;
      #endif
      
      ///Curved order crosses columns
      if (I_NI > 1)
      {
        I_I = blitz::sum(I_A1_I(blitz::Range(1, I_NI-1)) - I_A1_I(blitz::Range(0, I_NI - 2))) / (I_NI - 1);
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") > 1): I_I set to " << I_I << endl;
        #endif
        
        /// number of swaths along the order
        I_NBins = pfsDRPStella::math::Round(static_cast<double>(_trace.getHeight()) / static_cast<double>(I_I) / 3.);
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") > 1): I_NBin = Round((double)_trace.getHeight()(=" << _trace.getHeight() << ") / (double)I_I(=" << I_I << ") / 3.) set to " << I_NBins << endl;
          cout << "CFits::MkSlitFunc: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") > 1): I_NBin set to " << I_NBins << endl;
        #endif
      }
      else
      { /// Perfectly aligned orders
        /// Still follow the changes in PSF
        I_NBins = pfsDRPStella::math::Round(static_cast<double>(_trace.getHeight()) / 400.);
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") <= 1): I_NBin = int(_trace.getHeight()(=" << _trace.getHeight() << ") / 400.) set to " << I_NBins << endl;
          cout << "CFits::MkSlitFunc: !KeyWord_Set(SWATH_WIDTH): if(I_NI(=" << I_NI << ") <= 1): I_NBin set to " << I_NBins << endl;
        #endif
      }
      if (I_NBins < 3)
        I_NBins = 3;
      if (I_NBins > 20)
        I_NBins = 2;
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: !KeyWord_Set(SWATH_WIDTH): I_NBin set to " << I_NBins << endl;
      #endif

      /// Adjust for the true order length
    } /// if (!(I_Pos = pfsDRPStella::utils::KeyWord_Set(const_cast<const CString**>(PP_CS_Args), I_NArgs, CS_Temp)))
    
    I_BinHeight = _trace.getHeight() / I_NBins;
    if (I_NBins > 1)
      I_NBins = (2 * I_NBins) - 1;
    
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "CFits::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << endl;
      cout << "CFits::MkSlitFunc: I_NBins set to " << I_NBins << endl;
    #endif

    /// Calculate boundaries of distinct slitf regions.
    /// Boundaries of bins
    I_A2_IBinBoundY.resize(I_NBins,2);
    I_A2_IBinBoundY = 0;
    I_A2_IBinBoundY(0,0) = 0;//int(_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow);// + 1
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "CFits::MkSlitFunc: 1. I_A2_IBound(0,0) set to " << I_A2_IBinBoundY(0,0) << endl;
    #endif
    int I_BinHeight_Temp = I_BinHeight;
    I_A2_IBinBoundY(0,1) = I_BinHeight_Temp;
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "MkSlitFunc: I_A2_IBinBoundY(0, 1) set to " << I_A2_IBinBoundY(0, 1) << endl;
    #endif
    while(I_A2_IBinBoundY(0,1) >= _ccdHeight){
      I_A2_IBinBoundY(0,1)--;
      _fiberTraceFunction.yHigh--;
      I_BinHeight_Temp--;
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "MkSlitFunc: I_A2_IBinBoundY(0,1) >= _trace.getHeight(): I_A2_IBinBoundY(0, 1) set to " << I_A2_IBinBoundY(0, 1) << endl;
      #endif
    }
    for (int i_bin = 1; i_bin < I_NBins; i_bin++){
      I_BinHeight_Temp = I_BinHeight;
      I_A2_IBinBoundY(i_bin,0) = I_A2_IBinBoundY(i_bin-1,0) + int(double(I_BinHeight) / 2.);
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "MkSlitFunc: I_A2_IBinBoundY(i_bin,1) >= _trace.getHeight(): I_A2_IBinBoundY(" << i_bin << ",0) set to " << I_A2_IBinBoundY(i_bin, 0) << endl;
      #endif
      while(I_A2_IBinBoundY(i_bin,0) < 0){
        I_A2_IBinBoundY(i_bin,0)++;
        I_BinHeight_Temp--;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "MkSlitFunc: I_A2_IBinBoundY(i_bin,0) < 0: I_A2_IBinBoundY(" << i_bin << ", 0) set to " << I_A2_IBinBoundY(i_bin, 0) << endl;
        #endif
      }
      I_A2_IBinBoundY(i_bin,1) = I_A2_IBinBoundY(i_bin,0) + I_BinHeight_Temp;
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "MkSlitFunc: I_A2_IBinBoundY(" << i_bin << ",1) set to " << I_A2_IBinBoundY(i_bin, 1) << endl;
      #endif
      while(I_A2_IBinBoundY(i_bin,1) >= _trace.getHeight()){
        I_A2_IBinBoundY(i_bin,1)--;
        _fiberTraceFunction.yHigh--;
        I_BinHeight_Temp--;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "MkSlitFunc: I_A2_IBinBoundY(i_bin,1) >= _trace.getHeight(): I_A2_IBinBoundY(" << i_bin << ",1) set to " << I_A2_IBinBoundY(i_bin, 1) << endl;
        #endif
      }
    }
    I_A2_IBinBoundY(I_NBins-1, 1) = _trace.getHeight()-1;
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "CFits::MkSlitFunc: I_A2_IBound set to " << I_A2_IBinBoundY << endl;
    #endif
    #ifdef __DEBUG_SLITFUNC_X__
      string boundFN = debugdir + "I_A2_IBoundY.dat";
      pfsDRPStella::utils::WriteArrayToFile(I_A2_IBinBoundY, boundFN, string("ascii"));
    #endif
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "CFits::MkSlitFunc: *P_D_A1_YHigh(fiberTraceNumber=" << fiberTraceNumber << ") = " << _fiberTraceFunction.yHigh << endl;
      cout << "CFits::MkSlitFunc: *P_D_A1_YLow(fiberTraceNumber=" << fiberTraceNumber << ") = " << _fiberTraceFunction.yLow << endl;
      cout << "CFits::MkSlitFunc: I_NBin = " << I_NBins << endl;
      cout << "CFits::MkSlitFunc: _ccdHeight = " << _ccdHeight << endl;
      cout << "CFits::MkSlitFunc: I_NBins = " << I_NBins << endl;
      cout << "CFits::MkSlitFunc: I_BinHeight = " << I_BinHeight << endl;
      cout << "CFits::MkSlitFunc: I_BinHeight_Temp = " << I_BinHeight_Temp << endl;
      cout << "CFits::MkSlitFunc: I_A2_IBinBoundY = " << I_A2_IBinBoundY << endl;
    #endif

    D_A3_SFSM.resize(_trace.getHeight(), _trace.getWidth(), I_NBins);
    
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "CFits::MkSlitFunc: D_A3_SFSM = " << D_A3_SFSM.rows() << " x " << D_A3_SFSM.cols() << " x " << I_NBins << endl;
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
      cout << "CFits::MkSlitFunc: D_A1_BinCen set to " << D_A1_BinCen << endl;
    #endif
    D_A1_ICol.resize(_ccdWidth);
    D_A1_ICol = i;
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "CFits::MkSlitFunc: D_A1_ICol set to " << D_A1_ICol << endl;
    #endif
    
    D_A1_XCentersE.resize(D_A1_XCenters.size());//int(_fiberTraceFunction.yHigh - _fiberTraceFunction.yLow + 1.));
    D_A1_XCentersE = D_A1_XCenters;//(blitz::Range((int)(_fiberTraceFunction.yCenter+_fiberTraceFunction.yLow), (int)(_fiberTraceFunction.yCenter+_fiberTraceFunction.yHigh)));
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "CFits::MkSlitFunc: D_A1_XCentersE set to " << D_A1_XCentersE << endl;
    #endif
    
    /// subpixel range required
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "CFits::MkSlitFunc: P_D_A1_XLow = " << _fiberTraceFunction.fiberTraceFunctionControl.xLow << endl;
    #endif
    
    ///TODO: SF = 7.2 pix -> 0.9-8.1 -> 9 pixels, not 7+1!!!
    I_NXSF = _trace.getWidth();//I_A2_MinCenMax(0,2) - I_A2_MinCenMax(0,0) + 1;// - I_NPixCut_Left - I_NPixCut_Right;
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "CFits::MkSlitFunc: I_NXSF set to " << I_NXSF << endl;
    #endif
    
    P_D_A1_BLZ->resize(_trace.getHeight());
    (*P_D_A1_BLZ) = 0.;
    
    D_A2_CCD_Ap.resize(D_A3_SFSM.rows(), D_A3_SFSM.cols());
    
    D_A2_Err_AllRows.resize(_trace.getHeight(), D_A3_SFSM.cols());
    I_A2_Mask_AllRows.resize(_trace.getHeight(), D_A3_SFSM.cols());
    I_A2_Mask_AllRows = 1;
    D_A2_Err_AllRows = 0.;
    
    for (int I_IBin = 0; I_IBin < I_NBins; I_IBin++) /// Loop thru sf regions
    {
      if (telluric == 3)
        B_MaximaOnly = true;

      #ifdef __DEBUG_CHECK_INDICES__      
        if (I_IBin >= I_A2_IBinBoundY.rows())
        {
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: I_IBin(=" << I_IBin << ") >= I_A2_IBoundY.rows(=" << I_A2_IBinBoundY.rows() << ")" << endl;
          return false;
        }
      #endif
      
      I_NR = I_A2_IBinBoundY(I_IBin,1) - I_A2_IBinBoundY(I_IBin,0) + 1; /// Number of rows (Y-Direction)
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): Resizing D_A2_SlitFunc_Im_In: I_A2_IBinBoundY(I_IBin, *) = " << I_A2_IBinBoundY(I_IBin, blitz::Range::all()) << ": I_NR set to " << I_NR << endl;
      #endif
      
      D_A2_SlitFunc_Im_In.resize(I_NR, I_NXSF);
      D_A2_SlitFunc_Im_In = 0.;
      if (ErrorsRead){
        D_A2_Err.resize(I_NR, I_NXSF);
        D_A2_Err = 0.;
      }
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": D_A2_SlitFunc_Im_In initialized to " << D_A2_SlitFunc_Im_In.rows() << " x " << D_A2_SlitFunc_Im_In.cols() << endl;
      #endif
      
      I_A2_Msk.resize(I_NR, I_NXSF);
      I_A2_Msk = 1;
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): I_A2_Msk initialized to " << I_A2_Msk.rows() << " x " << I_A2_Msk.cols() << endl;
      #endif
      
      if (max(I_A2_MaskArray) > 1){
        cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": I_IBin=" << I_IBin << ": ERROR: max(P_I_A2_MaskArray) > 1" << endl;
        return false;
      }
      
      #ifdef __DEBUG_CHECK_INDICES__      
        if (I_A2_IBinBoundY(I_IBin, 0) >= I_A2_IBinBoundY(I_IBin, 1)){
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: I_A2_IBound(I_IBin=" << I_IBin << ", 0)=" << I_A2_IBinBoundY(I_IBin, 0) << " >= I_A2_IBound(I_IBin, 1)=" << I_A2_IBinBoundY(I_IBin, 1) << " => Returning FALSE" << endl;
          return false;
        }
      #endif
      D_A1_XCenMXC.resize(I_A2_IBinBoundY(I_IBin,1) - I_A2_IBinBoundY(I_IBin,0) + 1);
      #ifdef __DEBUG_CHECK_INDICES__      
        if (I_A2_IBinBoundY(I_IBin,1) >= _trace.getHeight())
        {
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: I_A2_IBound(I_IBin,1)(=" << I_A2_IBinBoundY(I_IBin,1) << ") >= _trace.getHeight(=" << _trace.getHeight() << ")" << endl;
          return false;
        }
      #endif
      blitz::Array<double, 1> xCentersTemp(I_A2_IBinBoundY(I_IBin,1) - I_A2_IBinBoundY(I_IBin,0) + 1);
      xCentersTemp = D_A1_XCenters(blitz::Range(I_A2_IBinBoundY(I_IBin,0), I_A2_IBinBoundY(I_IBin,1))) + 0.5;
      ///TODO: check for the 0.5
      blitz::Array<double, 1> D_A1_XCentersTemp(xCentersTemp.size());
      D_A1_XCenMXC = xCentersTemp - pfsDRPStella::math::Int(xCentersTemp);
      #ifdef __DEBUG_MkSLITFUNC_FILES__
        string xCenMXC = debugdir + "D_A1_XCenMXC.dat";
        pfsDRPStella::utils::WriteArrayToFile(D_A1_XCenMXC, xCenMXC, string("ascii"));
      #endif
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (int I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin(=" << I_NBins << "); I_IBin++): D_A1_XCenMXC set to " << D_A1_XCenMXC << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
      #endif
      /// loop thru rows in region
      for (int n = 0; n < I_NR; n++)
      {
        /// column closest to peak
        #ifdef __DEBUG_CHECK_INDICES__      
          if (I_A2_IBinBoundY(I_IBin,0) + n >= _trace.getHeight())
          {
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: I_A2_IBound(I_IBin,0)(=" << I_A2_IBinBoundY(I_IBin,0) << ") + n(=" << n << ") >= _trace.getHeight(=" << _trace.getHeight() << ")" << endl;
            return false;
          }
        #endif
        D_A1_SSF.resize(_trace.getWidth());
        D_A1_Err.resize(_trace.getWidth());
        D_A1_Err = D_A2_Errors(I_A2_IBinBoundY(I_IBin,0) + n, blitz::Range::all());
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "MkSlitFunc: for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): D_A1_Err set to " << D_A1_Err << endl;
        #endif
        if (max(abs(D_A1_Err)) < 0.00000001){
          cout << "MkSlitFunc: ERROR: max(abs(D_A1_Err))=" << max(abs(D_A1_Err)) << " < 0.00000001 => Returning FALSE" << endl;
          return false;
        }
//        sTemp = "NO_SCATTER";
//        if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0 && *(int*)ArgV_In[I_Pos] != 0)
//        {
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ":  for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): KeyWord_Set(NO_SCATTER): P_D_A2_PixArray has " << _ccdHeight << " and " << _ccdWidth << endl;
          #endif
          D_A1_SSF = D_A2_PixArray(I_A2_IBinBoundY(I_IBin,0) + n, blitz::Range::all());
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): KeyWord_Set(NO_SCATTER): D_A1_SSF set to " << D_A1_SSF << endl;
          #endif
/*        }///      endif else begin
        else
        {
          /// Interpolate background
          D_A1_DXScatter.resize(I_NXSF);
          D_A1_DXScatter = i;
          
          if (I_A2_IBinBoundY(I_IBin,0) + n >= static_cast<int>(D_A1_XScatterBelow.size()))
          {
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: I_A2_IBound(I_IBin,0)(=" << I_A2_IBinBoundY(I_IBin,0) << ") + n(=" << n << ") >= D_A1_XScatterBelow.size(=" << D_A1_XScatterBelow.size() << ")" << endl;
            return false;
          }
          if (I_A2_IBinBoundY(I_IBin,0) + n >= static_cast<int>(D_A1_XScatterAbove.size()))
          {
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: I_A2_IBound(I_IBin,0)(=" << I_A2_IBinBoundY(I_IBin,0) << ") + n(=" << n << ") >= D_A1_XScatterAbove.size(=" << D_A1_XScatterAbove.size() << ")" << endl;
            return false;
          }
          D_A1_DXScatter += (double)I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 0) - D_A1_XScatterBelow(I_A2_IBinBoundY(I_IBin,0) + n);
          D_A1_DXScatter /= (D_A1_XScatterAbove(I_A2_IBinBoundY(I_IBin,0) + n)
          - D_A1_XScatterBelow(I_A2_IBinBoundY(I_IBin,0) + n));
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): Scatter+: D_A1_DXScatter set to " << D_A1_DXScatter << endl;
          #endif
          D_A1_Scatter.resize(I_NXSF);
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): Scatter+: D_A1_ScatterAbove(I_A2_IBound(I_IBin,0)+n) = " << D_A1_ScatterAbove(I_A2_IBinBoundY(I_IBin,0)+n) << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): Scatter+: D_A1_ScatterBelow(I_A2_IBound(I_IBin,0)+n) = " << D_A1_ScatterBelow(I_A2_IBinBoundY(I_IBin,0)+n) << endl;
          #endif
          
          D_A1_Scatter = D_A1_ScatterAbove(I_A2_IBinBoundY(I_IBin,0) + n) - D_A1_ScatterBelow(I_A2_IBinBoundY(I_IBin,0) + n);
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): Scatter+: D_A1_Scatter set to " << D_A1_Scatter << endl;
          #endif
          ///                  * dy_scatter+scatter_below(ib+j)
          if (D_A1_Scatter.size() != D_A1_DXScatter.size())
          {
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: D_A1_Scatter.size(=" << D_A1_Scatter.size() << ") != D_A1_DXScatter.size(=" << D_A1_DXScatter.size() << ")" << endl;
            return false;
          }
          D_A1_Scatter *= D_A1_DXScatter;
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): Scatter+: D_A1_Scatter set to " << D_A1_Scatter << endl;
          #endif
          D_A1_Scatter += D_A1_ScatterBelow(I_A2_IBinBoundY(I_IBin,0) + n);
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): Scatter+: D_A1_Scatter set to " << D_A1_Scatter << endl;
          #endif
          
          /// compute normalized slit func
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): Scatter+: D_A2_PixArray(I_A2_IBound(I_IBin,0)(=" << I_A2_IBinBoundY(I_IBin,0) << ") + n(=" << n << ") = " << I_A2_IBinBoundY(I_IBin,0) + n << ", blitz::Range(I_A2_MinCenMax(I_A2_IBound(I_IBin,0) + n, 0)(=" << I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 0) << "), I_A2_MinCenMax(I_A2_IBound(I_IBin,0) + n, 2)(=" << I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 2) << ") = " << blitz::Range(I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 0), I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 2)) << ") = " << D_A2_PixArray(I_A2_IBinBoundY(I_IBin,0) + n, blitz::Range(I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 0), I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 2))) << endl;
          #endif
          /// compute normalized slit func
          if (D_A1_Scatter.size() != D_A1_SSF.size())
          {
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: D_A1_Scatter.size(=" << D_A1_Scatter << ") != D_A1_SSF.size(=" << D_A1_SSF.size() << ")" << endl;
            return false;
          }
          if (static_cast<int>(D_A1_SSF.size()) != I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 2) - I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 0) + 1)
          {
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: D_A1_SSF.size(=" << D_A1_SSF.size() << ") != I_A2_MinCenMax(I_A2_IBound(I_IBin,0) + n, 2)(=" << I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 2) << ") - I_A2_MinCenMax(I_A2_IBound(I_IBin,0) + n, 0)(=" << I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 0) << ") + 1" << endl;
            return false;
          }
          if (_ccdHeight <= I_A2_IBinBoundY(I_IBin,0) + n)
          {
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: P_D_A2_PixArray.rows(=" << _ccdHeight << ") <= I_A2_IBound(I_IBin,0)(=" << I_A2_IBinBoundY(I_IBin,0) << ") + n(=" << n << ")" << endl;
            return false;
          }
          if (_ccdWidth <= I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 2))
          {
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: P_D_A2_PixArray->cols(=" << _ccdWidth << ") <= I_A2_MinCenMax(I_A2_IBound(I_IBin,0) + n, 2)(=" << I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 2) << ")" << endl;
            return false;
          }
          D_A1_SSF = D_A2_PixArray(I_A2_IBinBoundY(I_IBin,0) + n, blitz::Range(I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 0), I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 2))) - D_A1_Scatter;
          
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): Scatter+: D_A1_SSF set to " << D_A1_SSF << endl;
          #endif
        }/// end if (SCATTER)
        */
/*        if (I_A2_MaskArray.cols() <= I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 2))
        {
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: P_I_A2_Mask->cols(=" << I_A2_MaskArray.cols() << ") <= I_A2_MinCenMax(I_A2_IBound(I_IBin,0) + n, 2)(=" << I_A2_MinCenMax(I_A2_IBinBoundY(I_IBin,0) + n, 2) << ")" << endl;
          return false;
        }
        */
        #ifdef __DEBUG_CHECK_INDICES__      
          if (I_A2_IBinBoundY(I_IBin,0)+n >= I_A2_MaskArray.rows())
          {
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: I_A2_IBound(I_IBin,0)+n(=" << I_A2_IBinBoundY(I_IBin,0)+n << ") >= I_A2_MaskArray.rows(=" << I_A2_MaskArray.rows() << ")" << endl;
            return false;
          }
        #endif
        D_A2_SlitFunc_Im_In(n, blitz::Range::all()) = D_A1_SSF;
        #ifdef __DEBUG_CHECK_INDICES__      
          if (I_A2_IBinBoundY(I_IBin, 0) - I_A2_IBinBoundY(0, 0) + n >= D_A2_CCD_Ap.rows()){
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (int n = 0; n < I_NR(=" << I_NR << "; n++): _fiberTraceFunction.yCenter = " << _fiberTraceFunction.yCenter << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (int n = 0; n < I_NR(=" << I_NR << "; n++): _fiberTraceFunction.yLow = " << _fiberTraceFunction.yLow << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (int n = 0; n < I_NR(=" << I_NR << "; n++): _fiberTraceFunction.yHigh = " << _fiberTraceFunction.yHigh << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (int n = 0; n < I_NR(=" << I_NR << "; n++): _fiberTraceFunction.yHigh - _fiberTraceFunction.yLow + 1 = " << _fiberTraceFunction.yHigh - _fiberTraceFunction.yLow + 1 << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (int n = 0; n < I_NR(=" << I_NR << "; n++): D_A3_SFSM.rows() = " << D_A3_SFSM.rows() << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (int n = 0; n < I_NR(=" << I_NR << "; n++): I_BinHeight = " << I_BinHeight << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (int n = 0; n < I_NR(=" << I_NR << "; n++): I_NBin = " << I_NBins << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (int n = 0; n < I_NR(=" << I_NR << "; n++): I_A2_IBinBoundY(*,0) = " << I_A2_IBinBoundY(blitz::Range::all(), 0) << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (int n = 0; n < I_NR(=" << I_NR << "; n++): I_A2_IBinBoundY(*,1) = " << I_A2_IBinBoundY(blitz::Range::all(), 1) << endl;
            blitz::Array<int, 1> I_A1_IBinBoundYTemp(I_A2_IBinBoundY.rows());
            I_A1_IBinBoundYTemp = I_A2_IBinBoundY(blitz::Range::all(),1) - I_A2_IBinBoundY(blitz::Range::all(),0);
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (int n = 0; n < I_NR(=" << I_NR << "; n++): I_A2_IBinBoundY(*,1) - I_A2_IBinBoundY(*,0) = " << I_A1_IBinBoundYTemp << endl;
          
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (int n = 0; n < I_NR(=" << I_NR << "; n++): I_A2_IBinBoundY(I_IBin, *) = " << I_A2_IBinBoundY(I_IBin, blitz::Range::all()) << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (int n = 0; n < I_NR(=" << I_NR << "; n++): I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(I_IBin, 0) + 1 = " << I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(I_IBin, 0) + 1 << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (int n = 0; n < I_NR(=" << I_NR << "; n++): ERROR: (0.5 * I_IBin * I_BinHeight(=" << I_BinHeight << ")) + n(=" << n << ") = " << (0.5 * I_IBin * I_BinHeight) + n << " D_A2_CCD_Ap.rows() = " << D_A2_CCD_Ap.rows() << " => Returning FALSE" << endl;
            return false;
          }
          if (static_cast<int>(D_A1_SSF.size()) != D_A2_CCD_Ap.cols()){
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: D_A1_SSF.size() = " << D_A1_SSF.size() << " != D_A2_CCD_Ap.cols() = " << D_A2_CCD_Ap.cols() << " => Returning FALSE" << endl;
            return false;
          }
        #endif
        D_A2_CCD_Ap(I_A2_IBinBoundY(I_IBin, 0) - I_A2_IBinBoundY(0, 0) + n, blitz::Range::all()) = D_A1_SSF;
        
        #ifdef __DEBUG_CHECK_INDICES__      
          if (((I_A2_IBinBoundY(I_IBin, 0) + n) < 0) || ((I_A2_IBinBoundY(I_IBin, 0) + n) >= I_A2_MaskArray.rows())){
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: ((I_A2_IBound(I_IBin, 0) + n)=" << I_A2_IBinBoundY(I_IBin, 0) + n << " < 0) || ((I_A2_IBound(I_IBin, 0) + n) >= I_A2_MaskArray.rows()=" << I_A2_MaskArray.rows() << ") => Returning FALSE" << endl;
            return false;
          }
        #endif
        I_A2_Msk(n, blitz::Range::all()) = I_A2_MaskArray(I_A2_IBinBoundY(I_IBin,0) + n, blitz::Range::all());
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): D_A1_XCenters(" << I_A2_IBinBoundY(I_IBin,0) + n << ") = " << D_A1_XCenters(I_A2_IBinBoundY(I_IBin,0) + n) << endl;
          double D_Temp = D_A1_XCenters(I_A2_IBinBoundY(I_IBin,0) + n) + _fiberTraceFunction.fiberTraceFunctionControl.xLow;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): D_A1_XCenters(" << I_A2_IBinBoundY(I_IBin,0)+n << ") + P_D_A1_XLow(" << fiberTraceNumber << ") = " << D_Temp << endl;
          D_Temp = D_A1_XCenters(I_A2_IBinBoundY(I_IBin,0) + n) + _fiberTraceFunction.fiberTraceFunctionControl.xHigh;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): D_A1_XCenters(" << I_A2_IBinBoundY(I_IBin,0)+n << ") + P_D_A1_XHigh(" << fiberTraceNumber << ") = " << D_Temp << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): I_A2_Msk(n, blitz::Range::all()) set to " << I_A2_Msk(n, blitz::Range::all()) << endl;
        #endif
        if (max(I_A2_Msk(n, blitz::Range::all())) > 1){
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": I_A2_Msk(n=" << n << ", *) = " << I_A2_Msk(n, blitz::Range::all()) << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: max(I_A2_Msk) > 1" << endl;
          return false;
        }
        if (ErrorsRead){
          D_A2_Err(n, blitz::Range::all()) = D_A1_Err;
          #ifdef __DEBUG_CHECK_INDICES__      
            if (D_A2_Err_AllRows.cols() != static_cast<int>(D_A1_Err.size())){
              cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: D_A2_Err_AllRows.cols(=" << D_A2_Err_AllRows.cols() << ") != D_A1_Err.size(=" << D_A1_Err.size() << ")" << endl;
              return false;
            }
            if ((I_A2_IBinBoundY(I_IBin, 0) + n < 0) || (I_A2_IBinBoundY(I_IBin, 0) + n >= D_A2_Err_AllRows.rows())){
              cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: (I_A2_IBound(I_IBin=" << I_IBin << ", 0)=" << I_A2_IBinBoundY(I_IBin, 0) << " + n = " << I_A2_IBinBoundY(I_IBin, 0) + n << " < 0) || (I_A2_IBound(I_IBin, 0) + n >= D_A2_Err_AllRows.rows()=" << D_A2_Err_AllRows.rows() << ") => Returning FALSE" << endl;
              return false;
            }
          #endif
          D_A2_Err_AllRows(I_A2_IBinBoundY(I_IBin, 0) + n, blitz::Range::all()) = D_A1_Err;
        }
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): D_A2_SlitFunc_Im_In(n, 0) set to " << D_A2_SlitFunc_Im_In(n, 0) << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): D_A2_SlitFunc_Im_In(n, D_A2_SlitFunc_Im_In.cols()-1) set to " << D_A2_SlitFunc_Im_In(n, D_A2_SlitFunc_Im_In.cols()-1) << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": for (I_IBin(=" << I_IBin << ") = 0; I_IBin < I_NBin=" << I_NBins << "; I_IBin++): for (n(=" << n << ") = 0; n < I_NR(=" << I_NR << "); n++): D_A2_SlitFunc_Im_In(n, *) set to " << D_A2_SlitFunc_Im_In(n, blitz::Range::all()) << endl;
        #endif
        #ifdef __DEBUG_CHECK_INDICES__      
          if (n >= I_A2_Msk.rows())
          {
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: n(=" << n << ") >= I_A2_Msk.rows(=" << I_A2_Msk.rows() << ")" << endl;
            return false;
          }
        #endif
      } /// for (int n = 0; n < I_NR; n++)
      if (max(I_A2_Msk) > 1){
        cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": A. I_A2_Msk = " << I_A2_Msk << endl;
        cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": A. ERROR: max(I_A2_Msk) = " << max(I_A2_Msk) << " > 1" << endl;
        return false;
      }
      
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": end for: D_A2_Err = " << D_A2_Err << endl;
      #endif
      D_A2_Err_Temp.resize(D_A2_Err.rows(), D_A2_Err.cols());
      D_A2_Err_Temp = D_A2_Err;
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": end for: D_A2_SlitFunc_Im_In(*,0) set to " << D_A2_SlitFunc_Im_In(blitz::Range::all(),0) << endl;
        cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": end for: D_A2_SlitFunc_Im_In(*,ncols-1=" << D_A2_SlitFunc_Im_In.cols()-1 << ") set to " << D_A2_SlitFunc_Im_In(blitz::Range::all(),D_A2_SlitFunc_Im_In.cols()-1) << endl;
        cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": end for: I_A2_Msk(*,0) set to " << I_A2_Msk(blitz::Range::all(), 0) << endl;
        cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": end for: I_A2_Msk(*,ncols-1=" << I_A2_Msk.cols()-1 << ") set to " << I_A2_Msk(blitz::Range::all(), I_A2_Msk.cols()-1) << endl;
        string S_TempA = DEBUGDIR + "I_A2_Msk.fits";
        pfsDRPStella::utils::WriteFits(&I_A2_Msk, S_TempA);
        S_TempA = DEBUGDIR + "D_A2_SlitFunc_Im_In.fits";
        pfsDRPStella::utils::WriteFits(&D_A2_SlitFunc_Im_In, S_TempA);
      #endif
      
      D_A2_SlitFuncOrig.resize(D_A2_SlitFunc_Im_In.rows(), D_A2_SlitFunc_Im_In.cols());
      /// Backup original Slit Function
      D_A2_SlitFuncOrig = D_A2_SlitFunc_Im_In;
      for (int p=0; p<D_A2_SlitFunc_Im_In.rows(); p++)
      {
        /// Set MySF to 0. where < 3.*(-RON)
        D_A2_SlitFunc_Im_In(p, blitz::Range::all()) = blitz::where(D_A2_SlitFunc_Im_In(p, blitz::Range::all()) < (3. * (0.-D_ReadOutNoise)), 0., D_A2_SlitFunc_Im_In(p, blitz::Range::all()));
      }
      if (telluric == 1)///Piskunov
      {
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": telluric == 1" << endl;
        #endif
        D_A1_Sky.resize(D_A2_SlitFunc_Im_In.rows());
        D_A1_ErrSky.resize(D_A2_SlitFunc_Im_In.rows());
        D_A1_Tel.resize(D_A2_SlitFunc_Im_In.cols());
        D_A1_Tel = blitz::sum(D_A2_SlitFunc_Im_In(j, i), j);
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: D_A2_SlitFunc_Im_In(*,0) = " << D_A2_SlitFunc_Im_In(blitz::Range::all(),0) << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: D_A1_Tel set to " << D_A1_Tel.size() << ": " << D_A1_Tel << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: D_A1_Tel set to " << D_A1_Tel << endl;
        #endif
        I_A1_ITel.resize(D_A1_Tel.size());
        I_A1_ITel = blitz::where(D_A1_Tel <= (max(D_A1_Tel) - min(D_A1_Tel)) / 100. + min(D_A1_Tel), 1, 0);
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: I_A1_ITel set to " << I_A1_ITel << endl;
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
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: D_A2_Tel set to " << D_A2_Tel << endl;
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
              cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: o(=" << o << ") >= D_A2_Tel.rows(=" << D_A2_Tel.rows() << ")" << endl;
              return false;
            }
          #endif
          if (ErrorsRead){
            D_A1_ErrTelSub.resize(D_A2_ErrTel.cols());
            D_A1_ErrTelSub = D_A2_ErrTel(o, blitz::Range::all());
            PP_Args_Median[1] = &D_A1_ErrTelSub;
          }
          D_A1_SC(o) = pfsDRPStella::math::Median(D_A2_Tel(o, blitz::Range::all()), S_A1_Args_Median, PP_Args_Median);
          if (ErrorsRead){
            D_A1_ErrSky(o) = D_Val_ErrOutMedian;
            D_A2_Err(o, blitz::Range::all()) += D_Val_ErrOutMedian;
            D_A2_ErrTel(o, blitz::Range::all()) = D_A1_ErrTelSub;
          }
        }
        D_A1_Sky = D_A1_SC;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: D_A1_SC set to " << D_A1_SC << endl;
        #endif
        
        blitz::Array<double, 1> tempDblVecArr = pfsDRPStella::math::Replicate(1., _trace.getWidth());// I_A2_MinCenMax(0, 2) - I_A2_MinCenMax(0, 0) + 1);
        #ifdef __DEBUG_CHECK_INDICES__      
          if (D_A2_SlitFunc_Im_In.cols() != static_cast<int>(tempDblVecArr.size())){
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: D_A2_SlitFunc_Im_In.cols()=" << D_A2_SlitFunc_Im_In.cols() << " != tempDblVecArr.size()=" << tempDblVecArr.size() << " => Returning FALSE" << endl;
            return false;
          }
        #endif
        blitz::Array<double, 2> *p_d2mata = pfsDRPStella::math::VecArrACrossB(D_A1_SC, tempDblVecArr);
        
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc:  telluric = 1: I_IBin = " << I_IBin << ": I_NR = " << I_NR << ", tempDblVecArr.size() = " << tempDblVecArr.size() << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: D_A1_SC = " << D_A1_SC << ", VecArrACrossB(D_A1_SC, tempDblVecArr) = " << *p_d2mata << endl;
        #endif
        #ifdef __DEBUG_CHECK_INDICES__      
          if (D_A2_SlitFunc_Im_In.rows() != static_cast<int>(D_A1_SC.size()))
          {
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: D_A2_SlitFunc_Im_In.rows(=" << D_A2_SlitFunc_Im_In.rows() << ") != D_A1_SC.size(=" << D_A1_SC.size() << ")" << endl;
            return false;
          }
          if (D_A2_SlitFunc_Im_In.cols() != static_cast<int>(tempDblVecArr.size()))
          {
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: D_A2_SlitFunc_Im_In.cols(=" << D_A2_SlitFunc_Im_In.cols() << ") != tempDblVecArr.size(=" << tempDblVecArr.size() << ")" << endl;
            return false;
          }
        #endif
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": telluric = 1: p_d2mata = " << *p_d2mata << endl;
        #endif
        D_A2_SlitFunc_Im_In -= (*p_d2mata);
        delete p_d2mata;
//        #ifdef __DEBUG_MKSLITFUNC__
//          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": KeyWord_Set(TELLURIC): D_A2_SlitFunc_Im_In set to " << D_A2_SlitFunc_Im_In << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
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
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": I_NR(=" << I_NR << endl;
        #endif
      } /// end if (telluric == 1)
      else if (telluric == 3){
        
        I_A2_Mask_Tel.resize(I_A2_Msk.rows(), I_A2_Msk.cols());
        I_A2_Mask_Tel = I_A2_Msk;
        if (max(I_A2_Mask_Tel) > 1){
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": 0. I_A2_Mask_TelTemp = " << I_A2_Mask_TelTemp << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": 0. ERROR: max(I_A2_Mask_Tel = " << I_A2_Mask_Tel << ") = " << max(I_A2_Mask_Tel) << " > 1" << endl;
          return false;
        }
        
        I_A1_UseRow_Tel.resize(I_A2_Msk.rows());
        I_A1_UseRow_Tel = pfsDRPStella::math::IndGenArr(I_A2_Msk.rows());
        
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
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: for(p(=" << p << "); p<D_A2_SlitFunc_Im_In.rows()=" << D_A2_SlitFunc_Im_In.rows() << "; p++): D_A2_SlitFunc_Im_In(p,*) = " << D_A2_SlitFunc_Im_In(p,blitz::Range::all()) << endl;
          #endif
          
          /// Normalize rows of D_A2_SlitFunc_Im_In to 1.
          if (fabs(blitz::sum(D_A2_SlitFunc_Im_In(p, blitz::Range::all()))) < 0.00000000000000001)
            D_A2_SlitFunc_Im_In(p, blitz::Range::all()) = 1.;
          D_A2_SlitFunc_Im_In(p,blitz::Range::all()) /= blitz::sum(D_A2_SlitFunc_Im_In(p,blitz::Range::all()));
          #ifdef __DEBUG_TELLURIC__
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: for(p(=" << p << ")...): D_A2_SlitFunc_Im_In(p,*) = " << D_A2_SlitFunc_Im_In(p,blitz::Range::all()) << endl;
          #endif
          
          /// Get maximum of D_A2_SlitFunc_Im_In for every row
          D_A1_SFMax(p) = max(D_A2_SlitFunc_Im_In(p,blitz::Range::all()));
          
          /// Find MaxPos
          I_A1_MaxPosCol = blitz::where(fabs(D_A2_SlitFunc_Im_In(p,blitz::Range::all()) - D_A1_SFMax(p)) < 0.000001,1,0);
          P_I_A1_MaxPosColInd = pfsDRPStella::math::GetIndex(I_A1_MaxPosCol, I_NMax);
          I_A1_MaxPos(p) = (*P_I_A1_MaxPosColInd)(0);
          
          #ifdef __DEBUG_TELLURIC__
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: for(p(=" << p << ")...): D_A1_SFMax(p) = " << D_A1_SFMax(p) << endl;
          #endif
        }/// end for (int p=0; p<D_A2_SlitFunc_Im_In.rows(); p++)
        int I_MaxPos = pfsDRPStella::math::Median(I_A1_MaxPos);
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
          pfsDRPStella::utils::WriteFits(&D_A2_SlitFunc_Im_In, S_MySF);
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: File " << S_MySF << " written" << endl;
        #endif
        /// /////////////////////////////////////////////////////////////////////////
        
        //      D_A2_SlitFunc_Im_In = D_A2_SlitFunc_Im_In_Max;
        
        /// /////////////////////////////////////////////////////////////////////////
        
        /// TODO: ONLY TAKE HIGHEST VALUES, NOT MIDDLE ONES?!? <- Already did median filtering!
        /// --- remove elements from D_A1_SFMax which are outside the median value +/- 2sigma
        I_A1_UseRow_Tel_AllRows.resize(D_A2_SlitFunc_Im_In.rows());
        I_A1_UseRow_Tel_AllRows = pfsDRPStella::math::IndGenArr(D_A2_SlitFunc_Im_In.rows());
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: while: Median(D_A1_SFMax) = " << pfsDRPStella::math::Median(D_A1_SFMax) << endl;
        #endif
        I_A1_IndA.resize(D_A1_SFMax.size());
        
        
        /** ************************************/
        
        
        double D_MedianSFMax = pfsDRPStella::math::Median(D_A1_SFMax);
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: D_A1_SFMax = " << D_A1_SFMax << endl;
          cout << "CFits::MkSlitFunc: D_MedianSFMax = " << D_MedianSFMax << endl;
        #endif
        I_A1_IndA = blitz::where((D_A1_SFMax > D_MedianSFMax) & (D_A1_SFMax < 1.5 * D_MedianSFMax),1,0);
        
        
        /** ************************************/
        
        
        
        P_I_A1_Ind = pfsDRPStella::math::GetIndex(I_A1_IndA, nind_temp);
        #ifdef __DEBUG_TELLURIC__
          cout << "CFits::MkSlitFunc: nind_temp = " << nind_temp << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: while: I_A1_IndA set to " << I_A1_IndA << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: while: *P_I_A1_Ind set to " << *P_I_A1_Ind << endl;
        #endif
        blitz::Array<double, 2> D_A2_SFMax(2,2);
        pfsDRPStella::math::GetSubArrCopy(D_A2_SlitFunc_Im_In,
                                          *P_I_A1_Ind,
                                          0,
                                          D_A2_SFMax);
        blitz::Array<double, 1> D_A1_SFMa(D_A2_SFMax.rows());
        for (int i_c=0; i_c<D_A2_SlitFunc_Im_In.cols(); i_c++){
          D_A1_SFMa = pfsDRPStella::math::MedianVec(D_A2_SFMax(blitz::Range::all(), i_c),5);
          D_A2_SFMax(blitz::Range::all(), i_c) = D_A1_SFMa;
        }
        for (int inde=0; inde<static_cast<int>(P_I_A1_Ind->size()); inde++){
          D_A2_SlitFunc_Im_In((*P_I_A1_Ind)(inde), blitz::Range::all()) = D_A2_SFMax(inde, blitz::Range::all());
        }
        #ifdef __DEBUG_TELLURIC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": TELLURIC == 3: while: D_A1_SFMax set to " << D_A1_SFMax << endl;
        #endif
        
        I_A1_UseRow_TelTemp.resize(I_A1_UseRow_Tel.size());
        I_A1_UseRow_TelTemp = I_A1_UseRow_Tel;
        I_A1_UseRow_Tel.resize(P_I_A1_Ind->size());
        I_A2_Mask_TelTemp.resize(I_A2_Mask_Tel.rows(), I_A2_Mask_Tel.cols());
        I_A2_Mask_TelTemp = 0;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": 1. I_A2_Mask_Tel = " << I_A2_Mask_Tel << endl;
        #endif
        if (max(I_A2_Mask_Tel) > 1){
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": 1A. I_A2_Mask_TelTemp = " << I_A2_Mask_TelTemp << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": 1A. ERROR: max(I_A2_Mask_Tel = " << I_A2_Mask_Tel << ") = " << max(I_A2_Mask_Tel) << " > 1" << endl;
          return false;
        }
        for (int nnn=0; nnn<static_cast<int>(P_I_A1_Ind->size()); nnn++){
          I_A2_Mask_TelTemp((*P_I_A1_Ind)(nnn),blitz::Range::all()) = I_A2_Mask_Tel((*P_I_A1_Ind)(nnn), blitz::Range::all());
          #ifdef __DEBUG_CHECK_INDICES__      
            if ((*P_I_A1_Ind)(nnn) >= static_cast<int>(I_A1_UseRow_Tel_AllRows.size())){
              cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: (*P_I_A1_Ind)(nnn=" << nnn << ")=" << (*P_I_A1_Ind)(nnn) << ") = " << (*P_I_A1_Ind)(nnn) << " >= I_A1_UseRow_Tel_AllRows.size()=" << I_A1_UseRow_Tel_AllRows.size() << endl;
              return false;
            }
          #endif
          I_A1_UseRow_Tel(nnn) = I_A1_UseRow_Tel_AllRows((*P_I_A1_Ind)(nnn));
        }
        #ifdef __DEBUG_MKSLITFUNC__
          string tempC = debugdir + "P_I_A1_Ind_1.dat";
          pfsDRPStella::utils::WriteArrayToFile(*P_I_A1_Ind, tempC, string("ascii"));
        #endif
        
        blitz::Array<double, 1> D_A1_SFMax_SumRows(D_A2_SlitFunc_Im_In.cols());
        blitz::Array<double, 2> D_A2_SlitFunc_Im_In_Times_Mask(D_A2_SlitFunc_Im_In.rows(), D_A2_SlitFunc_Im_In.cols());
        D_A2_SlitFunc_Im_In_Times_Mask = D_A2_SlitFunc_Im_In * I_A2_Mask_TelTemp;
        D_A1_SFMax_SumRows = blitz::sum(D_A2_SlitFunc_Im_In_Times_Mask(j,i),j);
        #ifdef __DEBUG_SLITFUNC_FILES__
          string S_SFMax_SumRows("SFMax_SumRows_");
          if (B_MaximaOnly){
            cout << "CFits::MkSlitFunc: B_MaximaOnly: D_A1_SFMax_SumRows = " << D_A1_SFMax_SumRows << endl;
            S_SFMax_SumRows += "MaxOnly_";
          }
          else{
            cout << "CFits::MkSlitFunc: !B_MaximaOnly: D_A1_SFMax_SumRows = " << D_A1_SFMax_SumRows << endl;
          }
          S_SFMax_SumRows += "IBin" + to_string(I_IBin) + "_IRunTel" + to_string(I_Run_Tel) + ".fits";
          pfsDRPStella::utils::WriteFits(&D_A1_SFMax_SumRows, S_SFMax_SumRows);
        #endif
        
        I_A1_Ind_Last.resize(P_I_A1_Ind->size());
        I_A1_Ind_Last = (*P_I_A1_Ind);
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": 2. I_A2_Mask_TelTemp = " << I_A2_Mask_TelTemp << endl;
        #endif
        I_A2_Mask_Tel = I_A2_Mask_TelTemp;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": 3. I_A2_Mask_Tel = " << I_A2_Mask_Tel << endl;
        #endif
        if (max(I_A2_Mask_Tel) > 1){
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": 3A. I_A2_Mask_TelTemp = " << I_A2_Mask_TelTemp << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": 3A. ERROR: max(I_A2_Mask_Tel = " << I_A2_Mask_Tel << ") = " << max(I_A2_Mask_Tel) << " > 1" << endl;
          return false;
        }
        #ifdef __DEBUG_MKSLITFUNC__
          D_A2_TempIm.resize(D_A2_SlitFunc_Im_In.rows(), D_A2_SlitFunc_Im_In.cols());
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": D_A2_SlitFunc_Im_In = " << D_A2_SlitFunc_Im_In << endl;
          D_A2_TempIm = D_A2_SlitFunc_Im_In * I_A2_Mask_Tel;
          string S_TempC = debugdir + "D_A2_SlitFunc_Im_In_Times_Mask_1.fits";
          pfsDRPStella::utils::WriteFits(&D_A2_TempIm, S_TempC);
          D_A2_TempIm.resize(1,1);
        #endif
        delete(P_I_A1_Ind);
      }/// end else if (Telluric == 3)
      
      I_RunMax = 1;
      if (telluric > 2){
        I_RunMax = _fiberTraceExtractionControl->maxIterSky;
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
        ///      if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "FIBERTRACENUMBER")) >= 0)
        ///        s_a1(pppos) = "FIBERTRACENUMBER";
        ///      if (_fiberTraceExtractionControl->xCorProf > 0)
        ///        s_a1(pppos) = "XCOR_PROF_OUT";
        ///      s_a1(pppos) = "SP_OUT";
        ///      s_a1(pppos) = "STOP";
        ///      s_a1(pppos) = "MASK";
        //       if (_fiberTraceExtractionControl->telluric > 1)
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
        if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "FIBERTRACENUMBER")) >= 0)
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
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": I_A2_Mask_Temp = " << I_A2_Mask_Temp << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": args[" << pppos << "] set to " << *((blitz::Array<int, 2>*)(args[pppos])) << endl;
        #endif
        if (max(I_A2_Mask_Temp) > 1){
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: A. max(I_A2_Mask_Temp = " << I_A2_Mask_Temp << ") > 1 => Returning FALSE" << endl;
          return false;
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
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": D_A2_Err = " << D_A2_Err << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": args[" << pppos << "] set to " << *((blitz::Array<double, 2>*)(args[pppos])) << endl;
//            return false;
          #endif
          pppos++;
          
          D_A1_ErrOut.resize(D_A2_SlitFunc_Im_In_Tel.rows());
          D_A1_ErrOut = 0.;
          args[pppos] = &D_A1_ErrOut;
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": D_A1_ErrOut = " << D_A1_ErrOut << endl;
          #endif
          pppos++;
          
          D_A1_Errors_SP_Out.resize(D_A2_SlitFunc_Im_In_Tel.rows());
          D_A1_Errors_SP_Out = 0.;
          args[pppos] = &D_A1_Errors_SP_Out;
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": D_A1_Errors_SP_Out = " << D_A1_Errors_SP_Out << endl;
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
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": args[" << pppos << "] set to " << *((int*)(args[pppos])) << endl;
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
        
        cout << "CFits::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ", I_IBin = " << I_IBin << ": starting SlitFunc" << endl;
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
        
          string S_FName_SF = DEBUGDIR + "SlitFunc_SFIn_in_";
          if (B_MaximaOnly)
            S_FName_SF += "MaxOnly_";
          S_FName_SF += "IBin" + S_TempNum + "_Tel" + to_string(telluric) + "_IRunTel" + to_string(I_Run_Tel) + ".fits";
          pfsDRPStella::utils::WriteFits(&D_A2_SlitFunc_Im_In_Tel,  S_FName_SF);
        #endif
        
        blitz::Array<double, 2> D_A2_SlitFunc_Im_In_SumRows(D_A2_SlitFunc_Im_In.rows(), D_A2_SlitFunc_Im_In.cols());
        D_A2_SlitFunc_Im_In_SumRows = D_A2_SlitFunc_Im_In_Tel * I_A2_Mask_Temp;
        #ifdef __DEBUG_MkSLITFUNC_FILES__
          blitz::Array<double, 1> D_A1_SlitFunc_Im_In_SumRows(D_A2_SlitFunc_Im_In_SumRows.cols());
          D_A1_SlitFunc_Im_In_SumRows = blitz::sum(D_A2_SlitFunc_Im_In_SumRows(j,i),j);
          string S_Sum = "SlitFuncImInTel_SumCols" + S_SF_DebugFilesSuffix + ".fits";
          pfsDRPStella::utils::WriteFits(&D_A1_SlitFunc_Im_In_SumRows, S_Sum);
        #endif
        if (!this->SlitFunc(D_A2_SlitFunc_Im_In_Tel,
                            I_MaxIterSig,
                            D_A1_XCenMXC,
                            D_A1_SP,
                            D_A2_SFSM,
                            s_a1,
                            args))
        {
          cout << "CFits::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ", I_IBin = " << I_IBin << ": ERROR: SlitFunc returned FALSE!" << endl;
          return false;
        }
        #ifdef __DEBUG_SLITFUNC_FILES__
          cout << "CFits::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ", I_IBin = " << I_IBin << ": D_A1_SP = " << D_A1_SP << endl;
          cout << "CFits::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ", I_IBin = " << I_IBin << ": D_A1_SP_Out = " << D_A1_SP_Out << endl;
        #endif
        if (ErrorsRead){
          #ifdef __DEBUG_SLITFUNC_FILES__
            cout << "CFits::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ", I_IBin = " << I_IBin << ": D_A1_Errors_SP_Out = " << D_A1_Errors_SP_Out << endl;
          #endif
          blitz::Array<double, 1> D_A1_SNR(D_A1_SP.size());
          D_A1_SNR = D_A1_SP_Out / D_A1_Errors_SP_Out;
          #ifdef __DEBUG_SLITFUNC_FILES__
            cout << "CFits::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ", I_IBin = " << I_IBin << ": D_A1_SNR = " << D_A1_SNR << endl;
          #endif
        }
        #ifdef __DEBUG_MKSLITFUNC__
          blitz::Array<double, 2> D_A2_MaskTimesSlitFunc(I_A2_Mask_Temp.rows(), I_A2_Mask_Temp.cols());
          D_A2_MaskTimesSlitFunc = D_A2_SlitFunc_Im_In_Tel * I_A2_Mask_Temp;
          string S_MaskOut = DEBUGDIR + "SlitFuncTimesMask_just_after_SlitFunc" + S_SF_DebugFilesSuffix + ".fits";
          if (!pfsDRPStella::utils::WriteFits(&D_A2_MaskTimesSlitFunc, S_MaskOut)){
            cout << "CFits::MkSlitFunc: ERROR: WriteFits(I_A2_Mask_Temp, " << S_MaskOut << ") returned FALSE" << endl;
            return false;
          }
        #endif
        I_A2_Mask = I_A2_Mask_Temp;
        I_A2_Mask_Temp = I_A2_MaskApTemp;
        if (!B_MaximaOnly){
          #ifdef __DEBUG_SLITFUNC_FILES__
            cout << "CFits::MkSlitFunc: !B_MaximaOnly: After SlitFunc: I_A2_Mask = " << I_A2_Mask <<  endl;
          #endif
        }
        
        cout << "CFits::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ", I_IBin = " << I_IBin << ": SlitFunc ready" << endl;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_IBin = " << I_IBin << ": D_A2_SFSM = " << D_A2_SFSM << endl;
          if (ErrorsRead)
            cout << "CFits::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_IBin = " << I_IBin << ": After SlitFunc: D_A2_Err = " << D_A2_Err << endl;
        #endif
        if (!B_MaximaOnly){
          #ifdef __DEBUG_CHECK_INDICES__      
            if (D_A3_SFSM.cols() != D_A2_SFSM.cols()){
              cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: D_A3_SFSM.cols(=" << D_A3_SFSM.cols() << ") != D_A2_SFSM.cols(=" << D_A2_SFSM.cols() << ")" << endl;
              return false;
            }
            if (I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(0,0) - (I_A2_IBinBoundY(I_IBin, 0) - I_A2_IBinBoundY(0,0)) + 1 != D_A2_SFSM.rows()){
              cout << "CFits::MkSlitFunc: ERROR: I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(0,0) - (I_A2_IBinBoundY(I_IBin, 0) - I_A2_IBinBoundY(0,0)) + 1 (=" << I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(0,0) - (I_A2_IBinBoundY(I_IBin, 0) - I_A2_IBinBoundY(0,0)) + 1 << ") != D_A2_SFSM.rows()=" << D_A2_SFSM.rows() << " => Returning FALSE" << endl;
              return false;
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
                cout << "CFits::MkSlitFunc: I_A2_MaskArray(" << I_A2_IBinBoundY(I_IBin, 0)+i_row << ", *) = " << I_A2_MaskArray(I_A2_IBinBoundY(I_IBin, 0)+i_row, blitz::Range::all()) << endl;
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
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": fiberTraceNumber = " << fiberTraceNumber << ", IBin = " << I_IBin << ": D_A2_SFSM set to " << D_A2_SFSM << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": fiberTraceNumber = " << fiberTraceNumber << ", IBin = " << I_IBin << ": D_A1_SP set to " << D_A1_SP << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": fiberTraceNumber = " << fiberTraceNumber << ", IBin = " << I_IBin << ": I_A2_Mask_Temp set to " << I_A2_Mask_Temp << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": SlitFunc ready" << endl;
        
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": After SlitFunc: D_A1_SP = " << D_A1_SP << endl;
        
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": After SlitFunc: D_A2_SFSM = " << D_A2_SFSM << endl;
        
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": After SlitFunc: D_A2_SFSM = Im_Out = " << D_A2_SFSM << endl;
          string S_FNameOut = debugdir + "SlitFunc_SFSMOut_out" + S_TempNum + S_SF_DebugFilesSuffix + ".fits";
          pfsDRPStella::utils::WriteFits(&D_A2_SFSM,  S_FNameOut);
        #endif
          
        if (telluric == 3){/// Subtract minimum of each row > 0. from Slit Function
          double D_MinSF = 0.;
          for (int q=0; q<D_A2_SFSM.rows(); q++){
            
            
            /// TODO: Only include pixels which are not marked as bad (old strategy with some border pixels equal to zero)
            
            D_MinSF = min(D_A2_SFSM(q, blitz::Range::all()));
            D_A2_SFSM(q, blitz::Range::all()) = D_A2_SFSM(q, blitz::Range::all()) - D_MinSF;
            double D_Sum_D_A2_SFSM = blitz::sum(D_A2_SFSM(q,blitz::Range::all()));
            if (abs(D_Sum_D_A2_SFSM) < 0.00000000000000001){
              cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: D_Sum_D_A2_SFSM == 0 => Returning FALSE" << endl;
              return false;
            }
            D_A2_SFSM(q, blitz::Range::all()) = D_A2_SFSM(q, blitz::Range::all()) / D_Sum_D_A2_SFSM;
            #ifdef __DEBUG_MKSLITFUNC__
              cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": D_Sum_D_A2_SFSM = " << D_Sum_D_A2_SFSM << endl;
              cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": D_A2_SFSM(q=" << q << ", *) set to " << D_A2_SFSM(q, blitz::Range::all()) << endl;
            #endif
          }/// end for (int q=0; q<D_A2_SFSM.rows(); q++){
          #ifdef __DEBUG_MKSLITFUNC__
            string S_FName = "D_A2_SFOut_Tel3";
            if (B_MaximaOnly)
              S_FName += "_MaxOnly";
            S_FName += "_IRunTel" + to_string(I_Run_Tel) + ".fits";
            pfsDRPStella::utils::WriteFits(&D_A2_SFSM, S_FName);
          #endif
        }/// end if (telluric == 3)
        
        pfsDRPStella::math::Double(I_A2_Mask, D_A2_Mask);
        if (fabs(mean(I_A2_Mask) - mean(D_A2_Mask)) > 0.000001){
          cout << "CFits::MkSlitFunc: ERROR: mean(I_A2_Mask)(=" << mean(I_A2_Mask) << ") != mean(D_A2_Mask)(=" << mean(D_A2_Mask) << ")" << endl;
          return false;
        }
        D_SFDev = sqrt(blitz::sum(blitz::pow2(D_A2_SFSM - D_A2_SFOld) * D_A2_Mask)/blitz::sum(D_A2_Mask));
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": I_Run_Tel = " << I_Run_Tel << ": D_SFDev = " << D_SFDev << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": I_Run_Tel = " << I_Run_Tel << ": mean(D_A2_SlitFunc_Im_In_Tel) / 1000. = " << mean(D_A2_SlitFunc_Im_In_Tel) / 1000. << endl;
        #endif
        D_A2_SFOld = D_A2_SFSM;
        
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": Before Fit: 1. D_A2_Err = " << D_A2_Err << endl;
        #endif
        #ifdef __DEBUG_CHECK_INDICES__      
          if (I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(I_IBin, 0) + 1 != I_A2_Mask.rows()){
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: I_A2_IBound(I_IBin, 1) - I_A2_IBound(I_IBin, 0) + 1(=" << I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(I_IBin, 0) + 1 << ") != I_A2_Mask.rows())(=" << I_A2_Mask.rows() << ")" << endl;
            return false;
          }
          if (I_A2_Mask_AllRows.cols() != I_A2_Mask.cols()){
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: I_A2_Mask_AllRows.cols(=" << I_A2_Mask_AllRows.cols() << ") != I_A2_Mask.cols(=" << I_A2_Mask.cols() << ")" << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": D_A2_SFSM.cols() = " << D_A2_SFSM.cols() << endl;
            return false;
          }
        #endif
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": I_Run_Tel = " << I_Run_Tel << ": I_IBin = " << I_IBin << ", I_A2_IBound = " << I_A2_IBinBoundY.rows() << " x " << I_A2_IBinBoundY.cols() << ", I_A2_IBound(I_IBin, *) = " << I_A2_IBinBoundY(I_IBin, blitz::Range::all()) << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": I_Run_Tel = " << I_Run_Tel << ": I_IBin = " << I_IBin << ", I_A2_Mask_AllRows = " << I_A2_Mask_AllRows.rows() << " x " << I_A2_Mask_AllRows.cols() << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": I_Run_Tel = " << I_Run_Tel << ": I_IBin = " << I_IBin << ", I_A2_Mask = " << I_A2_Mask.rows() << " x " << I_A2_Mask.cols() << endl;
          cout << "CFits::MkSlitFunc: Before Fit: 2. D_A2_Err = " << D_A2_Err << endl;
        #endif
        
        #ifdef __DEBUG_CHECK_INDICES__      
          if (I_A2_IBinBoundY(I_IBin, 0) < 0){
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: I_A2_IBound(I_IBin=" << I_IBin << ", 0) = " << I_A2_IBinBoundY(I_IBin, 0) << " < 0 => Return false" << endl;
            return false;
          }
          if (I_A2_IBinBoundY(I_IBin, 1) >= I_A2_Mask_AllRows.rows()){
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: I_A2_IBound(I_IBin=" << I_IBin << ", 1) = " << I_A2_IBinBoundY(I_IBin, 1) << " >= I_A2_Mask_AllRows.rows() = " << I_A2_Mask_AllRows.rows() << " => Returning FALSE" << endl;
            return false;
          }
          if (I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(I_IBin, 0) + 1 != I_A2_Mask.rows()){
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: I_A2_IBound(I_IBin=" << I_IBin << ", 1)(=" << I_A2_IBinBoundY(I_IBin, 1) << ") - I_A2_IBound(I_IBin, 0)(=" << I_A2_IBinBoundY(I_IBin, 0) << ") + 1 = " << I_A2_IBinBoundY(I_IBin, 1) - I_A2_IBinBoundY(I_IBin, 0) + 1 << " != I_A2_Mask.rows()=" << I_A2_Mask.rows() << " => Returning FALSE" << endl;
            return false;
          }
          if (I_A2_Mask_AllRows.cols() != I_A2_Mask.cols()){
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: I_A2_Mask_AllRows.cols()=" << I_A2_Mask_AllRows.cols() << " != I_A2_Mask.cols()=" << I_A2_Mask.cols() << " => Returning FALSE" << endl;
            return false;
          }
        #endif
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_A2_Mask = " << I_A2_Mask << endl;
        #endif
        if (!B_MaximaOnly){
          for (int i_row=0; i_row<I_A2_Mask.rows(); i_row++){
            #ifdef __DEBUG_MKSLITFUNC__
              cout << "CFits::MkSlitFunc: I_A2_Mask_AllRows(" << I_A2_IBinBoundY(I_IBin, 0)+i_row << ", *) = " << I_A2_Mask_AllRows(I_A2_IBinBoundY(I_IBin, 0)+i_row, blitz::Range::all()) << endl;
            #endif
            for (int i_col=0; i_col<I_A2_Mask.cols(); i_col++){
              if (I_A2_Mask(i_row, i_col) == 0)
                I_A2_Mask_AllRows(I_A2_IBinBoundY(I_IBin, 0)+i_row, i_col) = 0;
            }
            #ifdef __DEBUG_MKSLITFUNC__
              cout << "CFits::MkSlitFunc: I_A2_Mask_AllRows(" << I_A2_IBinBoundY(I_IBin, 0)+i_row << ", *) = " << I_A2_Mask_AllRows(I_A2_IBinBoundY(I_IBin, 0)+i_row, blitz::Range::all()) << endl;
            #endif
          }
        }/// end if (!B_MaximaOnly)
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": I_A2_Mask_AllRows(blitz::Range(" << I_A2_IBinBoundY(I_IBin, 0) << ", " << I_A2_IBinBoundY(I_IBin, 1) << "), blitz::Range::all()) set to I_A2_Mask = " << I_A2_Mask << endl;
        #endif
        
        D_A2_Err.resize(D_A2_Err_Temp.rows(), D_A2_Err_Temp.cols());
        D_A2_Err = D_A2_Err_Temp;
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": Before Fit: 4. D_A2_Err = " << D_A2_Err << endl;
        #endif
        if ((I_Run_Tel == (I_RunMax-1)) || (D_SFDev < mean(D_A2_SFSM) / 5000000.)){
          #ifdef __DEBUG_MKSLITFUNC__
            if (D_SFDev < mean(D_A2_SlitFunc_Im_In_Tel) / 5000000.){
              if (B_MaximaOnly)
                cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": B_MaximaOnly==TRUE: D_SFDev = " << D_SFDev << ": mean(D_A2_SFSM) = " << mean(D_A2_SFSM) << endl;
              else
                cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": D_SFDev = " << D_SFDev << ": mean(D_A2_SFSM) = " << mean(D_A2_SFSM) << endl;
            }
          #endif
          if (B_MaximaOnly){
            B_MaximaOnly = false;
            I_MaxIterSig = I_MaxIterSig_Temp;
            if (telluric == 3){
              I_Run_Tel = -1;
              D_SFDev = 1.;
              I_A2_Mask_Tel = I_A2_Msk;
              I_A2_Mask_Temp = I_A2_Msk;
              #ifdef __DEBUG_MKSLITFUNC__
                cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": I_A2_Mask_Tel set to " << I_A2_Mask_Tel << endl;
              #endif
              D_A2_SlitFunc_Im_In = D_A2_SlitFuncOrig;
              D_A2_SlitFunc_Im_In_Tel = D_A2_SlitFuncOrig;
              #ifdef __DEBUG_MKSLITFUNC__
                cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": D_A2_SlitFunc_Im_In_Tel set to " << D_A2_SlitFunc_Im_In_Tel << endl;
              #endif
              D_A1_Sky = 0.;
              #ifdef __DEBUG_MKSLITFUNC__
                cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": D_A2_SFSM = " << D_A2_SFSM << endl;
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
              cout << "CFits::MkSlitFunc: ERROR: D_A2_SlitFuncOrig.rows() != D_A2_Err.rows()" << endl;
              return false;
            }
          #endif
          if (ErrorsRead){
            S_A1_Args_Fit(0) = "MEASURE_ERRORS_IN";
            PP_Args_Fit[0] = &D_A2_Err;
            #ifdef __DEBUG_MKSLITFUNC__
              cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": Just before Fit: D_A2_Err = " << D_A2_Err << endl;
            #endif
          }
          
          #ifdef __DEBUG_CHECK_INDICES__      
            if (D_A2_SlitFuncOrig.rows() != I_A2_Mask.rows()){
              cout << "CFits::MkSlitFunc: ERROR: D_A2_SlitFuncOrig.rows() != I_A2_Mask.rows()" << endl;
              return false;
            }
          #endif
          S_A1_Args_Fit(1) = "MASK_INOUT";
          /// TODO ????????????????????????????????????????????????
          PP_Args_Fit[1] = &I_A2_Mask;
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": Just before Fit: I_A2_Mask = " << I_A2_Mask << endl;
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
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": min(D_A2_SlitFunc_Im_In_Tel) = " << min(D_A2_SlitFunc_Im_In_Tel) << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": Just before Fit: D_A2_SFSM = " << D_A2_SFSM << endl;
            cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": Just before Fit: D_A2_SlitFuncOrig = " << D_A2_SlitFuncOrig << endl;
          #endif
          
          #ifdef __DEBUG_CHECK_INDICES__      
            if (D_A2_SlitFuncOrig.rows() != D_A2_SFSM.rows()){
              cout << "CFits::MkSlitFunc: ERROR: D_A2_SlitFuncOrig.rows() != D_A2_SFSM.rows()" << endl;
              return false;
            }
          #endif
          if (!pfsDRPStella::math::LinFitBevington(D_A2_SlitFuncOrig,      ///: in
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
              cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": 1. telluric == 2: ERROR: LinFitBevington(...) returned FALSE => Returning FALSE" << endl;
              return false;
            }
            #ifdef __DEBUG_MKSLITFUNC__
              cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": Just after Fit: D_A2_SFSM = " << D_A2_SFSM << endl;
            #endif
            D_A1_Sky_Temp = blitz::where(D_A1_Sky_Temp < 0., 0., D_A1_Sky_Temp);
            D_A1_Sky = D_A1_Sky_Temp;
            
//            #ifdef __DEBUG_MKSLITFUNC__
//              blitz::Array<double, 2> D_A2_MaskTimesSFOrig(D_A2_SlitFuncOrig.rows(), D_A2_SlitFuncOrig.cols());
//              D_A2_MaskTimesSFOrig = D_A2_SlitFuncOrig * I_A2_Mask;
//              string S_MaskFitOut = "SFOrigTimesMaskAfterFit" + CS_SF_DebugFilesSuffix + ".fits";
//              pfsDRPStella::utils::WriteFits(&D_A2_MaskTimesSFOrig, S_MaskFitOut);
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
              if (D_ImMin < 0. - (3. * D_ReadOutNoise)){
                D_ImMin = D_ImMin + (3. * D_ReadOutNoise);
                #ifdef __DEBUG_MKSLITFUNC__
                  cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": tttt=" << tttt << ": D_ImMin = " << D_ImMin << endl;
                  cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": tttt=" << tttt << ": D_A2_SlitFunc_Im_In_Tel(tttt,*) = " << D_A2_SlitFunc_Im_In_Tel(tttt, blitz::Range::all()) << endl;
                  cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": tttt=" << tttt << ": D_A1_Sky(tttt) = " << D_A1_Sky(tttt) << endl;
                #endif
                D_A2_SlitFunc_Im_In_Tel(tttt, blitz::Range::all()) -= D_ImMin;
                #ifdef __DEBUG_MKSLITFUNC__
                  cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": tttt=" << tttt << ": D_A2_SlitFunc_Im_In_Tel(tttt,*) set to " << D_A2_SlitFunc_Im_In_Tel(tttt, blitz::Range::all()) << endl;
                #endif
                D_A1_Sky(tttt) += D_ImMin;
                #ifdef __DEBUG_MKSLITFUNC__
                  cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": tttt=" << tttt << ": D_A1_Sky(tttt) set to " << D_A1_Sky(tttt) << endl;
                #endif
              }
            }
            
            
            #ifdef __DEBUG_MKSLITFUNC__
              cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": I_Run_Tel = " << I_Run_Tel << ": after sky subtraction: min(D_A2_SlitFunc_Im_In_Tel) = " << min(D_A2_SlitFunc_Im_In_Tel) << endl;
            #endif
            
            #ifdef __DEBUG_MKSLITFUNC__
              string sPFit = debugdir + "D_A1_SPFit_";
              if (B_MaximaOnly)
                sPFit += "MaxOnly_";
              sPFit += to_string(I_Run_Tel) + ".dat";
              pfsDRPStella::utils::WriteArrayToFile(D_A1_SPFit, sPFit, string("ascii"));
              string skyFit = debugdir + "D_A1_SkyFit_";
              if (B_MaximaOnly)
                skyFit += "MaxOnly_" + to_string(I_Run_Tel) + ".dat";
              pfsDRPStella::utils::WriteArrayToFile(D_A1_Sky, skyFit, string("ascii"));
              string S_SlitFunc_Im_In_Tel = debugdir + "D_A2_SlitFunc_Im_In_Tel_skySubtracted";
              if (B_MaximaOnly)
                S_SlitFunc_Im_In_Tel += "_MaxOnly";
              S_SlitFunc_Im_In_Tel += to_string(I_Run_Tel) + ".fits";
              pfsDRPStella::utils::WriteFits(&D_A2_SlitFunc_Im_In_Tel, S_SlitFunc_Im_In_Tel);
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
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": I_Run_Tel = " << I_Run_Tel << endl;
        #endif
        I_Run_Tel++;
      } while (I_Run_Tel < I_RunMax);
      
      for (int q=0; q < static_cast<int>(D_A1_SP.size()); q++){
        
        #ifdef __DEBUG_MKSLITFUNC__
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": I_A2_Msk(q=" << q << ", blitz::Range::all()) = " << I_A2_Msk(q,blitz::Range::all()) << endl;
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": max(I_A2_Msk(q=" << q << ", blitz::Range::all())) = " << max(I_A2_Msk(q,blitz::Range::all())) << endl;
        #endif
        if (max(I_A2_Msk(q, blitz::Range::all())) > 1){
          cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: max(I_A2_Msk(q=" << q << ", blitz::Range::all())) > 1" << endl;
          return false;
        }
      }/// end for (int q=0; q < D_A1_SP.size(); q++)
      if (max(I_A2_Msk > 1)){
        cout << "CFits::MkSlitFunc: I_IBin = " << I_IBin << ": ERROR: max(I_A2_Msk=" << I_A2_Msk << ") > 1" << endl;
        return false;
      }
    } /// end for (int I_IBin = 0; I_IBin < I_NBin; I_IBin++) /// Loop thru sf regions
    
    D_A1_Sky.resize(I_A2_IBinBoundY(I_NBins-1,1) - I_A2_IBinBoundY(0,0) + 1);
    D_A1_ErrSky.resize(I_A2_IBinBoundY(I_NBins-1,1) - I_A2_IBinBoundY(0,0) + 1);
    D_A1_SP.resize(I_A2_IBinBoundY(I_NBins-1,1) - I_A2_IBinBoundY(0,0) + 1);
    
    D_A1_Errors_SP_Out.resize(I_A2_IBinBoundY(I_NBins-1,1) - I_A2_IBinBoundY(0,0) + 1);
    
    D_A2_SF.resize(I_A2_IBinBoundY(I_NBins-1,1) - I_A2_IBinBoundY(0,0) + 1, _trace.getWidth());
    #ifdef __DEBUG_CHECK_INDICES__      
      if (static_cast<int>(D_A1_SP.size()) != D_A2_SP.rows()){
        cout << "CFits::MkSlitFunc: ERROR: D_A1_SP.size(=" << D_A1_SP.size() << ") != D_A2_SP.rows(= " << D_A2_SP.rows() << ")" << endl;
        return false;
      }
      cout << "MkSlitFunc: I_A2_IBinBoundY = " << I_A2_IBinBoundY << endl;
      if (D_A2_SF.rows() != D_A3_SFSM.rows()){
        cout << "CFits::MkSlitFunc: ERROR: D_A2_SF.rows(=" << D_A2_SF.rows() << ") != D_A3_SFSM.rows(= " << D_A3_SFSM.rows() << ")" << endl;
        return false;
      }
      if (D_A2_SF.cols() != D_A3_SFSM.cols()){
        cout << "CFits::MkSlitFunc: ERROR: D_A2_SF.cols(=" << D_A2_SF.cols() << ") != D_A3_SFSM.cols(= " << D_A3_SFSM.cols() << ")" << endl;
        return false;
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
      int I_Row_Rel=0;
      double D_RowSum;
      for (int i_row = 0; i_row < D_A3_SFSM.rows(); i_row++){
        for (int i_ibin=0; i_ibin<I_NBins; i_ibin++){
          D_RowSum = blitz::sum(D_A3_SFSM(i_row, blitz::Range::all(), i_ibin));
          if (fabs(D_RowSum) > 0.00000000000000001){
            D_A3_SFSM(i_row, blitz::Range::all(), i_ibin) = D_A3_SFSM(i_row, blitz::Range::all(), i_ibin) / D_RowSum;
            #ifdef __DEBUG_MKSLITFUNC__
              cout << "D_A3_SFSM(" << i_row << ", *, " << i_ibin << ") = " << D_A3_SFSM(i_row, blitz::Range::all(), i_ibin) << endl;
              cout << "i_row = " << i_row << ": i_ibin = " << i_ibin << ": D_RowSum = " << D_RowSum << endl;
            #endif
          }
        }
        if ((I_Bin == 0) && (i_row < I_BinHeight/2)){
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
        else if ((I_Bin == I_NBins-1)){// && (i_row > (I_A2_IBound(I_Bin-1, 1) - (I_A2_IBound(0,1) / 2.)))){
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
          D_Weight_Bin1 = 2. * double(I_Row_Rel) / double(I_BinHeight);
          D_Weight_Bin0 = 1. - D_Weight_Bin1;
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: i_row = " << i_row << ": I_NBin = " << I_NBins << endl;
            cout << "CFits::MkSlitFunc: i_row = " << i_row << ": I_Bin = " << I_Bin << endl;
            cout << "CFits::MkSlitFunc: i_row = " << i_row << ": I_Row_Rel = " << I_Row_Rel << endl;
            cout << "CFits::MkSlitFunc: i_row = " << i_row << ": D_Weight_Bin0 = " << D_Weight_Bin0 << endl;
            cout << "CFits::MkSlitFunc: i_row = " << i_row << ": D_Weight_Bin1 = " << D_Weight_Bin1 << endl;
          #endif
          #ifdef __DEBUG_CHECK_INDICES__      
            if (i_row >= D_A2_SP.rows()){
              cout << "CFits::MkSlitFunc: ERROR: i_row = " << i_row << " >= D_A2_SP.rows() = " << D_A2_SP.rows() << " => Returning FALSE" << endl;
              return false;
            }
            if (I_Bin + 1 >= D_A2_SP.cols()){
              cout << "CFits::MkSlitFunc: ERROR: I_Bin + 1 = " << I_Bin + 1 << " >= D_A2_SP.cols() = " << D_A2_SP.cols() << " => Returning FALSE" << endl;
              return false;
            }
          #endif
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: i_row = " << i_row << ": I_Bin = " << I_Bin << ": D_A2_SP(i_row, I_Bin) = " << D_A2_SP(i_row, I_Bin) << ", D_A2_SP(i_row, I_Bin + 1) = " << D_A2_SP(i_row, I_Bin + 1) << endl;
          #endif
          D_A1_SP(i_row) = (D_A2_SP(i_row, I_Bin) * D_Weight_Bin0) + (D_A2_SP(i_row, I_Bin + 1) * D_Weight_Bin1);
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: i_row = " << i_row << ": I_Bin = " << I_Bin << ": D_A1_SP(" << i_row << ") set to " << D_A1_SP(i_row) << endl;
          #endif
          
          if (ErrorsRead){
            D_A1_Errors_SP_Out(i_row) = (D_A2_Errors_SP_Out(i_row, I_Bin) * D_Weight_Bin0) + (D_A2_Errors_SP_Out(i_row, I_Bin + 1) * D_Weight_Bin1);
          }
          
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: D_A3_SFSM(i_row, *, I_Bin) = " << D_A3_SFSM(i_row, blitz::Range::all(), I_Bin) << ", D_A3_SFSM(i_row, *, I_Bin+1) = " << D_A3_SFSM(i_row, blitz::Range::all(), I_Bin+1) << endl;
          #endif
          D_A2_SF(i_row, blitz::Range::all()) = (D_A3_SFSM(i_row, blitz::Range::all(), I_Bin) * D_Weight_Bin0) + (D_A3_SFSM(i_row, blitz::Range::all(), I_Bin+1) * D_Weight_Bin1);
          double dSumSFRow = blitz::sum(D_A2_SF(i_row, blitz::Range::all()));
          if (fabs(dSumSFRow) >= 0.00000000000000001)
            D_A2_SF(i_row, blitz::Range::all()) = D_A2_SF(i_row, blitz::Range::all()) / dSumSFRow;
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: i_row = " << i_row << ": I_Bin = " << I_Bin << ": D_A2_SF(" << i_row << ", *) set to " << D_A2_SF(i_row, blitz::Range::all()) << endl;
          #endif
          
          if (telluric == 1){
            #ifdef __DEBUG_CHECK_INDICES__      
              if (i_row >= D_A1_Sky.rows()){
                cout << "CFits::MkSlitFunc: ERROR: i_row >= D_A1_Sky.rows()=" << D_A1_Sky.rows() << " => Returning FALSE" << endl;
                return false;
              }
            #endif
            D_A1_Sky(i_row) = (D_A2_Sky(i_row, I_Bin) * D_Weight_Bin0) + (D_A2_Sky(i_row, I_Bin + 1) * D_Weight_Bin1);
            D_A1_ErrSky(i_row) = (D_A2_ErrSky(i_row, I_Bin) * D_Weight_Bin0) + (D_A2_ErrSky(i_row, I_Bin + 1) * D_Weight_Bin1);
            cout << "CFits::MkSlitFunc: i_row = " << i_row << ": I_Bin = " << I_Bin << ": D_A1_Sky(" << i_row << ") set to " << D_A1_Sky(i_row) << endl;
          }
          
          I_Row_Rel++;
        }
        
        if (I_Row_Rel == (I_BinHeight/2)){
          I_Bin++;
          I_Row_Rel = 0;
          #ifdef __DEBUG_MKSLITFUNC__
            cout << "CFits::MkSlitFunc: i_row = " << i_row << ": I_Bin set to " << I_Bin << endl;
          #endif
        }
      }/// end for (int i_row = 0; i_row < D_A3_SFSM.rows(); i_row++){
    }/// end if (I_NBin != 1){
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "CFits::MkSlitFunc: D_A2_SF set to " << D_A2_SF << endl;
    #endif
      
    ///populate _spectrum, _spectrumVariance, _background, _backgroundVariance
    _spectrum.resize(_trace.getHeight());
    _spectrumVariance.resize(_trace.getHeight());
    _background.resize(_trace.getHeight());
    _backgroundVariance.resize(_trace.getHeight());
    for (int ind=0; ind<_trace.getHeight(); ind++){
      _spectrum[ind] = static_cast<float>(D_A1_SP(ind));
      _spectrumVariance[ind] = static_cast<float>(D_A1_Errors_SP_Out(ind));
      _background[ind] = static_cast<float>(D_A1_Sky(ind));
      _backgroundVariance[ind] = static_cast<float>(D_A1_ErrSky(ind));
    }
    
/*    D_A1_SPFit.resize(D_A1_SP.size());
    #ifdef __DEBUG_CHECK_INDICES__      
      if (_fiberTraceFunction.yHigh - _fiberTraceFunction.yLow + 1 != D_A1_SP.size()){
        cout << "CFits::MkSlitFunc: ERROR: D_A1_SP.size(=" << D_A1_SP.size() << ") is wrong" << endl;
        return false;
      }
    #endif
    
    blitz::Array<string, 1> S_A1_Args_Fit(3);
    void **PP_Args_Fit;
    PP_Args_Fit = (void**)malloc(sizeof(void*) * 3);
    S_A1_Args_Fit = " ";
    
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "CFits::MkSlitFunc: D_A2_Err = " << D_A2_Err << endl;
    #endif
    
    blitz::Array<double, 2> D_A2_ErrAp(I_A2_IBinBoundY(I_NBins-1, 1) - I_A2_IBinBoundY(0,0) + 1, D_A2_Err_AllRows.cols());
    D_A2_ErrAp = 0.;
    if (ErrorsRead){
      S_A1_Args_Fit(0) = "MEASURE_ERRORS_IN";
      D_A2_ErrAp = D_A2_Err_AllRows(blitz::Range(I_A2_IBinBoundY(0,0), I_A2_IBinBoundY(I_NBins-1, 1)), blitz::Range::all());
      PP_Args_Fit[0] = &D_A2_ErrAp;
      #ifdef __DEBUG_MKSLITFUNC__
        cout << "CFits::MkSlitFunc: D_A2_ErrAp = " << D_A2_ErrAp << endl;
      #endif
    }
    
    S_A1_Args_Fit(1) = "MASK_INOUT";
    blitz::Array<int, 2> I_A2_MaskAp(I_A2_IBinBoundY(I_NBins-1, 1) - I_A2_IBinBoundY(0,0) + 1, I_A2_Mask_AllRows.cols());
    I_A2_MaskAp = I_A2_Mask_AllRows(blitz::Range(I_A2_IBinBoundY(0,0), I_A2_IBinBoundY(I_NBins-1, 1)), blitz::Range::all());
    PP_Args_Fit[1] = &I_A2_MaskAp;
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "I_A2_MaskAp = " << I_A2_MaskAp << endl;
    #endif
    
    S_A1_Args_Fit(2) = "SIGMA_OUT";
    blitz::Array<double, 2> D_A2_Sigma_Fit(D_A2_SF.rows(),2);
    PP_Args_Fit[2] = &D_A2_Sigma_Fit;
    
    blitz::Array<double, 1> D_A1_SkyFit(D_A1_SPFit.size());
    D_A1_SkyFit = 0.;
    bool B_WithSky = false;
    if (telluric > 0){
      D_A1_SkyFit = 1.;
      B_WithSky = true;
    }
    #ifdef __DEBUG_MKSLITFUNC__
      cout << "CFits::MkSlitFunc: Before Fit: D_A2_CCD_Ap = " << D_A2_CCD_Ap << endl;
    #endif
    #ifdef __DEBUG_MkSLITFUNC_FILES__
      string S_FileName_CCD_Ap = "CCD_Ap" + to_string(fiberTraceNumber) + "_Tel" + to_string(telluric) + ".fits";
      if (!pfsDRPStella::utils::WriteFits(&D_A2_CCD_Ap,S_FileName_CCD_Ap)){
        cout << "CFits::MkSlitFunc: WriteFits(D_A2_CCD_Ap," << S_FileName_CCD_Ap << ") returned FALSE!" << endl;
        return false;
      }
      cout << "CFits::MkSlitFunc: Before Fit: D_A2_SF = " << D_A2_SF << endl;
    #endif
    if (!pfsDRPStella::math::LinFitBevington(D_A2_CCD_Ap,      ///: in
                                             D_A2_SF,             ///: in
                                             D_A1_SPFit,             ///: out
                                             D_A1_SkyFit,          ///: in/out
                                             B_WithSky,                   ///: with sky: in
                                             S_A1_Args_Fit,         ///: in
                                             PP_Args_Fit)){          ///: in/out
        /// MEASURE_ERRORS_IN = blitz::Array<double,2>(D_A2_CCD_In.rows, D_A2_CCD_In.cols) : in
        /// REJECT_IN = double                                                      : in
        /// MASK_INOUT = blitz::Array<double,2>(D_A1_CCD_In.rows,D_A1_CCD_In.cols)         : in/out
        /// CHISQ_OUT = blitz::Array<double,1>(D_A2_CCD_In.rows)                           : out
        /// Q_OUT = blitz::Array<double,1>(D_A2_CCD_In.rows)                               : out
        /// SIGMA_OUT = blitz::Array<double,2>(D_A2_CCD_In.rows, 2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
      cout << "CFits::MkSlitFunc: 2. ERROR: LinFitBevington(...) returned FALSE => Returning FALSE" << endl;
      return false;
    }
    #ifdef __DEBUG_MkSLITFUNC_FILES__
      string S_MaskFinalOut = "Mask_Final" + S_SF_DebugFilesSuffix + ".fits";
      pfsDRPStella::utils::WriteFits(&I_A2_MaskAp, S_MaskFinalOut);
      
      S_MaskFinalOut = "D_A2_CCD_Ap" + CS_SF_DebugFilesSuffix + ".fits";
      pfsDRPStella::utils::WriteFits(&D_A2_CCD_Ap, S_MaskFinalOut);
      
      cout << "Just after Fit: D_A1_SPFit = " << D_A1_SPFit << endl;
      cout << "Just after Fit: D_A1_SkyFit = " << D_A1_SkyFit << endl;
      cout << "Just after Fit: D_A2_SF = " << D_A2_SF << endl;
    #endif
      
    #ifdef __DEBUG_CHECK_INDICES__      
      if (I_A2_IBinBoundY(I_NBins-1, 1) - I_A2_IBinBoundY(0,0) + 1 != static_cast<int>(D_A1_SPFit.size())){
        cout << "CFits::MkSlitFunc: ERROR: I_A2_IBound(I_NBin-1, 1)(=" << I_A2_IBinBoundY(I_NBins-1, 1) << ") - I_A2_IBound(0,0)(=" << I_A2_IBinBoundY(0,0) << ") + 1 != D_A1_SPFit.size()(=" << D_A1_SPFit.size() << ")" << endl;
        return false;
      }
    #endif
      
//    blitz::Array<double, 1> D_A1_BLZ(D_A1_SP.size());
//    D_A1_BLZ = D_A1_SP;
//    #ifdef __DEBUG_MKSLITFUNC__
//      cout << "CFits::MkSlitFunc: D_A1_BLZ set to " << D_A1_BLZ << endl;
//    #endif
//    blitz::Array<double, 2> D_A2_RecArray(_trace.getHeight(), _trace.getWidth());
//    D_A2_RecArray = 0.;
//    blitz::Array<double, 2> D_A2_RecFitArray(_trace.getHeight(), _trace.getWidth());
//    D_A2_RecFitArray = 0.;
//    blitz::Array<double, 1> D_A1_SkyFitArray(_trace.getHeight());
//    D_A1_SkyFitArray = 0.;
//    blitz::Array<double, 2> D_A2_RecSkyFitArray(_trace.getHeight(), _trace.getWidth());
//    D_A2_RecSkyFitArray = 0.;
//    blitz::Array<double, 1> D_A1_SkyArray(_trace.getHeight());
//    D_A1_SkyArray = 0.;
//    blitz::Array<double, 2> D_A2_RecSkyArray(_trace.getHeight(), _trace.getWidth());
//    D_A2_RecSkyArray = 0.;
//    blitz::Array<double, 1> D_A1_Errors_Ec(_trace.getHeight());
//    D_A1_Errors_Ec = 0.;
//    blitz::Array<double, 1> D_A1_Errors_EcFit(_trace.getHeight());
//    D_A1_Errors_EcFit = 0.;
//    blitz::Array<double, 1> D_A1_SkyFitError(_trace.getHeight());
//    D_A1_SkyFitError = 0.;
//    blitz::Array<double, 1> D_A1_SkyError(_trace.getHeight());
//    D_A1_SkyError = 0.;
  */  

    ///TODO: Safe errors and mask
/*    blitz::Array<double, 2> D_A2_ErrArray(_trace.getHeight(), _trace.getWidth());
    D_A2_ErrArray = D_A2_Errors;
    for (int i_row=0; i_row < static_cast<int>(D_A1_SPFit.size()); i_row++){
      if (ErrorsRead)
      {
        /// Mark bad pixels with large errors in this->P_D_A2_ErrArray
        D_A2_ErrArray(i_row, blitz::Range::all()) = blitz::where(I_A2_Mask_AllRows(i_row, blitz::Range::all()) < 1, 10000., D_A2_ErrArray(i_row, blitz::Range::all()));
      }// end if (this->ErrorsRead)
    }// end for (int i_row=0; i_row < D_A1_SPFit.size(); i_row++){
*/
    blitz::Array<float, 2> F_A2_ProfArray = pfsDRPStella::math::Float(D_A2_SF);
    ndarray::Array<float, 2, 1> ndarrayProf(pfsDRPStella::utils::copyBlitzToNdarray(F_A2_ProfArray));
    PTR(afwImage::Image<float>) imageProf(new afwImage::Image<float>(ndarrayProf));
    _profile.reset();
    _profile = imageProf;
    _isProfileSet = true;
    _isSpectrumExtracted = true;
    blitz::Array<unsigned short, 2> U_A2_Mask(_trace.getHeight(), _trace.getWidth());
    U_A2_Mask = where(I_A2_Mask_AllRows == 1, T_A2_MaskArray, 1);
    ndarray::Array<unsigned short, 2, 1> ndarrayMask(pfsDRPStella::utils::copyBlitzToNdarray(U_A2_Mask));
    afwImage::Mask<unsigned short> maskImage(ndarrayMask);
    *_trace.getMask() = maskImage;
    
//    if (ErrorsRead){
//      D_A1_Errors_Ec(blitz::Range::all()) = D_A1_Errors_SP_Out;
//    }
      
//    blitz::Array<double, 1> D_A1_BLZSmooth(D_A1_BLZ.size());
//    D_A1_BLZSmooth = 0.;
//    if (pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "FLAT") >= 0)
//    {
//      D_A1_BLZSmooth = pfsDRPStella::math::MedianVec(D_A1_BLZ, static_cast<int>(_fiberTraceExtractionControl->lambdaSP));
//    }
      
//    blitz::Array<double, 1> D_A1_Blaze(_trace.getHeight());
//    if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "FLAT")) >= 0)
//    {
//      D_A1_Blaze(blitz::Range::all()) = D_A1_BLZSmooth(blitz::Range::all());
//    }
//    else
//    {
//      D_A1_Blaze(blitz::Range::all()) = D_A1_BLZ(blitz::Range::all());
//    }
//    #ifdef __DEBUG_MKSLITFUNC__
//      cout << "CFits::MkSlitFunc: P_D_A2_Blaze(fiberTraceNumber=" << fiberTraceNumber << ", *) set to " << D_A1_Blaze(fiberTraceNumber, blitz::Range::all()) << endl;
//      
//      string sD_A2_SlitFTemp = debugdir + "D_A1_SPFit_Tel" + to_string(telluric) + ".dat";
//      pfsDRPStella::utils::WriteArrayToFile(D_A1_SPFit, sD_A2_SlitFTemp, string("ascii"));
//      cout << "CFits::MkSlitFunc: " << sD_A2_SlitFTemp << " written" << endl;
//    #endif

//    D_A2_RecArray = blitz::where(D_A2_RecArray < 0., 0., D_A2_RecArray);
//    D_A2_RecFitArray = blitz::where(D_A2_RecFitArray < 0., 0., D_A2_RecFitArray);
  
//    #ifdef __DEBUG_MKSLITFUNC__
//      cout << "CFits::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": D_A2_SF = " << D_A2_SF << endl;
//      cout << "CFits::MkSlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": D_A1_SP = " << D_A1_SP << endl;
//    #endif
      
    //    D_A1_DXScatter.resize(0);
    I_A2_IBinBoundY.resize(0);
    D_A1_ICol.resize(0);
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
    D_A1_XCenMXC.resize(0);
    D_A1_XCentersE.resize(0);
    D_A1_XInt.resize(0);
    D_A1_XSlitFTemp.resize(0);
    I_A1_I.resize(0);
    I_A1_ISort.resize(0);
    I_A1_ITel.resize(0);
    I_A1_IX.resize(0);
//    delete P_D_A1_BLZSmooth;
  
//    free(PP_Args_Fit);
    free(PP_Args_Median);
    free(args);

    cout << "MkSlitFunc: fiberTraceNumber " << fiberTraceNumber << " finished" << endl;
    return true;
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
//      cout << "FiberTrace::SlitFunc: ERROR: _image is not set" << endl;
//      return false;
//    }
    #ifdef __DEBUG_SLITFUNC__
      cout << "SlitFunc: D_A2_ImM = " << D_A2_ImM << endl;
      cout << "SlitFunc: xCentersPixelFraction_In = " << xCentersPixelFraction_In << endl;
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
//      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": D_A2_ImM = " << D_A2_ImM << endl;
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

    int overSample_In = _fiberTraceExtractionControl->overSample;
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
    
    blitz::Array<double, 2> D_A2_Sky(_ccdHeight, _ccdWidth);
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
    *P_D_A2_Errors = sqrt(D_A2_Im);
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
    double Lamb_SF = _fiberTraceExtractionControl->lambdaSF;
    double D_WingSmoothFactor = _fiberTraceExtractionControl->wingSmoothFactor;
    double Lamb_SP = _fiberTraceExtractionControl->lambdaSP;
    #ifdef __PISKUNOV_ORIG__
      double D_XLow = D_A2_Im.rows() / 2.;
    #endif
    #ifdef __DEBUG_SLITFUNC__
      cout << "SlitFunc: Lamb_SF = " << Lamb_SF << endl;
      cout << "SlitFunc: Lamb_SP = " << Lamb_SP << endl;
      cout << "SlitFunc: D_WingSmoothFactor = " << D_WingSmoothFactor << endl;
    #endif    
    
    std::string tempFileName(" ");
    std::string debugdir = DEBUGDIR;
    
    int I_NInd;
    int fiberTraceNumber = 0;
    Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "FIBERTRACENUMBER");
    if (Pos >= 0)
    {
      fiberTraceNumber = *(unsigned int*)ArgV_In[Pos];
    }
    
    bool ErrorsRead = true;
    
    blitz::firstIndex i;
    blitz::secondIndex j;
    
    Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "DEBUGFILES_SUFFIX");
    if (Pos >= 0)
    {
      debugFilesSuffix = *(string*)ArgV_In[Pos];
    }
    #ifdef __DEBUG_SLITFUNC_X__
      S_FileName_ImIn = DEBUGDIR + std::string("SlitFunc_ImIn") + debugFilesSuffix + std::string(".dat");
      pfsDRPStella::utils::WriteArrayToFile(D_A2_Im, S_FileName_ImIn, std::string("ascii"));

      std::string sFileName_XCentersIn = DEBUGDIR + std::string("SlitFunc_XCentersIn") + debugFilesSuffix + std::string(".dat");
      pfsDRPStella::utils::WriteArrayToFile(xCentersPixelFraction_In, sFileName_XCentersIn, std::string("ascii"));
    #endif
    
    Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "I_BIN");
    if (Pos >= 0)
    {
      I_Bin = *(int*)ArgV_In[Pos];
      #ifdef __DEBUG_SLITFUNC__
        cout << "CFits::SlitFunc: I_Bin set to " << I_Bin << endl;
      #endif
    }
    
//    Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "XLOW");
//    if (Pos >= 0)
//    {
//      D_XLow = *(double*)ArgV_In[Pos];
//      cout << "CFits::SlitFunc: D_XLow set to " << D_XLow << endl;
//    }

    /// TODO: get errors from maskedImage (Sigma or Variance???)
    if (ErrorsRead){
      Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "ERRORS");
//      #ifdef __DEBUG_SLITFUNC__
//        cout << "SlitFunc: KeyWord_Set(ERRORS): Pos = " << Pos << endl;
//        cout << "SlitFunc: KeyWord_Set(ERRORS): S_A1_Args_In = " << S_A1_Args_In << endl;
//        return false;
//      #endif
      if (Pos >= 0)
      {
        delete(P_D_A2_Errors);
        P_D_A2_Errors = (blitz::Array<double, 2>*)ArgV_In[Pos];
        #ifdef __DEBUG_SLITFUNC__
          cout << "SlitFunc: *P_D_A2_Errors = " << *P_D_A2_Errors << endl;
        #endif
//        return false;
      }
      Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "ERRORS_OUT");
      if (Pos >= 0)
      {
        P_D_A1_ErrOut = (blitz::Array<double, 1>*)ArgV_In[Pos];
        P_D_A1_ErrOut->resize(D_A2_Im.rows());
        (*P_D_A1_ErrOut) = 0.;
      }
      Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "ERRORS_SP_OUT");
      if (Pos >= 0)
      {
        delete(P_D_A1_SPErrOut);
        P_D_A1_SPErrOut = (blitz::Array<double, 1>*)ArgV_In[Pos];
        P_D_A1_SPErrOut->resize(D_A2_Im.rows());
        (*P_D_A1_SPErrOut) = 0.;
      }
    }/// end if (this->ErrorsRead)
    
    Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "SP_OUT");
    if (Pos >= 0)
    {
      delete(P_D_A1_SPOut);
      P_D_A1_SPOut = (blitz::Array<double, 1>*)ArgV_In[Pos];
      P_D_A1_SPOut->resize(D_A2_Im.rows());
      (*P_D_A1_SPOut) = 0.;
    }
    
    Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "SFO_OUT");
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
    if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "XCOR_PROF")) >= 0)
    {
      I_XCorProf = *(int*)ArgV_In[I_Pos];
      P_D_A1_XCorProfOut = (blitz::Array<double, 1>*)ArgV_In[I_Pos+1];
      P_D_A1_XCorProfOut->resize(I_NRows_Im);
      (*P_D_A1_XCorProfOut) = 0.;
      if ((I_XCorProf > 0) && (I_XCorProf < 5))
        I_XCorProf = 5;
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(XCOR_PROF): I_XCorProf set to " << I_XCorProf << endl;
    }
    else{
      P_D_A1_XCorProfOut = new blitz::Array<double, 1>(I_NRows_Im);
    }
    
    XCenVecArr.resize(xCentersPixelFraction_In.size());
    XCenVecArr = xCentersPixelFraction_In;
    #ifdef __DEBUG_SLITFUNC_XX__
//      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": XCenVecArr = " << XCenVecArr << endl;
      blitz::Array<double, 2> D_A2_XCentersPixelFraction(xCentersPixelFraction_In.size(), 2);
      D_A2_XCentersPixelFraction(blitz::Range::all(), 0) = i;
      D_A2_XCentersPixelFraction(blitz::Range::all(), 1) = XCenVecArr;
      std::string fnameXCen = DEBUGDIR + std::string("xCentersPixelFraction_In") + debugFilesSuffix + std::string(".dat");
      pfsDRPStella::utils::WriteArrayToFile(D_A2_XCentersPixelFraction, fnameXCen, std::string("ascii"));
    #endif
    if (overSample_In < 1)
      overSample_In = 1;
    #ifdef __DEBUG_SLITFUNC__
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": overSample_In = " << overSample_In << endl;
    #endif
    
    if ((Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "MASK")) >= 0)
    {
      P_I_A2_MaskIn = (blitz::Array<int, 2>*)ArgV_In[Pos];
      P_I_A2_Mask->resize(P_I_A2_MaskIn->rows(), P_I_A2_MaskIn->cols());
      (*P_I_A2_Mask) = (*P_I_A2_MaskIn);
      #ifdef __DEBUG_SLITFUNC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(MASK): P_I_A2_Mask read = " << *P_I_A2_Mask << endl;
      #endif
      if (max(*P_I_A2_Mask) > 1){
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: max(Mask) > 1" << endl;
        return false;
      }
      if (P_I_A2_Mask->size() != 1)
      {
        #ifdef __DEBUG_CHECK_INDICES__      
          if (P_I_A2_Mask->rows() != I_NRows_Im ||
              P_I_A2_Mask->cols() != I_NCols_Im)
          {
            cout << "SLIT_FUNC: Mask must have the same size as the image" << endl;
            return false;
          }
        #endif
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(MASK): (*P_I_A2_Mask) set to " << (*P_I_A2_Mask) << endl;
        #endif
        
      }/// end if (P_TempMask->size() != 1)
    }/// end if ((Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "MASK")) >= 0)
    ///else
    if (Pos < 0 || (Pos >= 0 && P_I_A2_Mask->size() == 1))
    {
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(MASK): D_A2_Im.rows = " << I_NRows_Im << ", D_A2_Im.cols = " << I_NCols_Im << endl;
      #endif
      if (I_NCols_Im < 0)
      {
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(MASK): D_A2_Im = " << D_A2_Im << endl;
      }
      P_I_A2_MaskIn = new blitz::Array<int, 2>(I_NRows_Im, I_NCols_Im);
      (*P_I_A2_MaskIn) = 1;
      P_I_A2_Mask->resize(I_NRows_Im, I_NCols_Im);
      (*P_I_A2_Mask) = 1;
      #ifdef __DEBUG_SLITFUNC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(MASK): (*P_I_A2_Mask) set to " << (*P_I_A2_Mask) << endl;
      #endif
      
    }/// end if (Pos < 0 || (Pos >= 0 && P_I_A2_Mask->size() == 1))
    
    if ((Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "PROF_OUT")) >= 0)
    {
      if (P_D_A2_Prof_Out != NULL)
        delete P_D_A2_Prof_Out;
      P_D_A2_Prof_Out = (blitz::Array<double, 2>*)ArgV_In[Pos];
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): P_D_A2_Prof_Out read " << endl;// to " << *P_D_A2_Prof_Out << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): P_D_A2_Prof_Out->size() = " << P_D_A2_Prof_Out->size() << endl;// to " << *P_D_A2_Prof_Out << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): P_D_A2_Prof_Out->rows() = " << P_D_A2_Prof_Out->rows() << endl;// to " << *P_D_A2_Prof_Out << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): P_D_A2_Prof_Out->cols() = " << P_D_A2_Prof_Out->cols() << endl;// to " << *P_D_A2_Prof_Out << endl;
      #endif
    }
    #ifdef __DEBUG_SLITFUNC__
      cout << "SlitFunc: _fiberTraceFunction.fiberTraceFunctionControl.interpolation = " << _fiberTraceFunction.fiberTraceFunctionControl.interpolation << endl;
      cout << "SlitFunc: _fiberTraceFunction.fiberTraceFunctionControl.order = " << _fiberTraceFunction.fiberTraceFunctionControl.order << endl;
      cout << "SlitFunc: _fiberTraceFunction.fiberTraceFunctionControl.xLow = " << _fiberTraceFunction.fiberTraceFunctionControl.xLow << endl;
      cout << "SlitFunc: _fiberTraceFunction.fiberTraceFunctionControl.xHigh = " << _fiberTraceFunction.fiberTraceFunctionControl.xHigh << endl;
      cout << "SlitFunc: _fiberTraceFunction.xCenter = " << _fiberTraceFunction.xCenter << endl;
      cout << "SlitFunc: _fiberTraceFunction.yCenter = " << _fiberTraceFunction.yCenter << endl;
      cout << "SlitFunc: _fiberTraceFunction.yLow = " << _fiberTraceFunction.yLow << endl;
      cout << "SlitFunc: _fiberTraceFunction.yHigh = " << _fiberTraceFunction.yHigh << endl;
      for (int ii = 0; ii < static_cast<int>(_fiberTraceFunction.coefficients.size()); ii++)
        cout << "SlitFunc: _fiberTraceFunction.coefficients[" << ii << "] = " << _fiberTraceFunction.coefficients[ii] << endl;
    #endif
    P_D_A2_Prof_Out->resize(I_NRows_Im, I_NCols_Im);
    (*P_D_A2_Prof_Out) = 0.;
    #ifdef __DEBUG_SLITFUNC__
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): P_D_A2_Prof_Out resized to (" << I_NRows_Im << ", " << I_NCols_Im << ")" << endl;
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
    blitz::Array<double,1> D_A1_IndGen = pfsDRPStella::math::Replicate(1., I_NRows_Im);
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
    for ( int fooInt = FiberTraceExtractionControl::NONE; fooInt != FiberTraceExtractionControl::NVALUES; fooInt++ ){
      if (_fiberTraceExtractionControl->telluric.compare(_fiberTraceExtractionControl->TELLURIC_NAMES[fooInt]) == 0){
        I_Telluric = fooInt;
      }
    }
    #ifdef __DEBUG_SLITFUNC__
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Telluric set to " << I_Telluric << endl;
    #endif
    if (I_Telluric == 2)
    {
      #ifdef __DEBUG_TELLURIC__
        cout << "__TELLURIC_MINE__ set" << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: S_A1_Args_In = " << S_A1_Args_In << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: S_A1_Args_In(0) = " << S_A1_Args_In(0) << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: S_A1_Args_In(1) = " << S_A1_Args_In(1) << endl;
      #endif
      
      /// take pointer to sky from argument for calculations
      Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "SKY");
      if (Pos >= 0)
      {
        delete(P_D_A1_MySky);
        P_D_A1_MySky = (blitz::Array<double,1>*)(ArgV_In[Pos]);
      }
      P_D_A1_MySky->resize(D_A2_Im.rows());
      (*P_D_A1_MySky) = 0.;
      
      if (ErrorsRead)
      {
        Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "ERR_SKY");
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
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC = 2: D_A2_MySF.size() = " << D_A2_MySF << endl;
      #endif
      D_A2_MySF = D_A2_Im;
      #ifdef __DEBUG_TELLURIC__
        string S_MySF = DEBUGDIR + std::string("D_A2_Im") + debugFilesSuffix + std::string(".fits");
        pfsDRPStella::utils::WriteFits(&D_A2_Im, S_MySF);
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: File " << *S_MySF << " written" << endl;
      #endif
      for (int p=0; p<I_NRows_Im; p++)
      {
        #ifdef __DEBUG_TELLURIC__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: for(p(=" << p << "); p<D_A2_MySF.rows()=" << D_A2_MySF.rows() << "; p++): D_A2_Im(p,*) = " << D_A2_Im(p,blitz::Range::all()) << endl;
        #endif
        
        /// Set MySF to 0. where < 3.*(-RON)
        D_A2_MySF(p, blitz::Range::all()) = blitz::where(D_A2_MySF(p, blitz::Range::all()) < (0. - (3. * _fiberTraceExtractionControl->ccdReadOutNoise)), 0., D_A2_MySF(p, blitz::Range::all()));
        
        /// Normalize rows of D_A2_MySF to 1.
        if (blitz::sum(D_A2_MySF(p, blitz::Range::all())) < 0.00000000000000001)
          D_A2_MySF(p, blitz::Range::all()) = 1.;
        D_A2_MySF(p,blitz::Range::all()) /= blitz::sum(D_A2_MySF(p,blitz::Range::all()));
        #ifdef __DEBUG_TELLURIC__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: for(p(=" << p << ")...): D_A2_MySF(p,*) = " << D_A2_MySF(p,blitz::Range::all()) << endl;
        #endif
        
        /// Get maximum of D_A2_MySF for every row
        D_A1_SFMax(p) = max(D_A2_MySF(p,blitz::Range::all()));
        
        #ifdef __DEBUG_TELLURIC__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: for(p(=" << p << ")...): D_A1_SFMax(p) = " << D_A1_SFMax(p) << endl;
        #endif
      }/// end for (int p=0; p<D_A2_MySF.rows(); p++)
      #ifdef __DEBUG_TELLURIC__
        S_MySF = DEBUGDIR + std::string("D_A2_MySF") + debugFilesSuffix + std::string(".fits");
        pfsDRPStella::utils::WriteFits(&D_A2_MySF, S_MySF);
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: File " << S_MySF << " written" << endl;
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
        D_A2_MySF_Max(blitz::Range::all(), nn) = pfsDRPStella::math::MedianVec(D_A2_MySF(blitz::Range::all(), nn), 5);
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
        pfsDRPStella::utils::WriteFits(&D_A2_MySF_Max, S_MySF);
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: File " << S_MySF << " written" << endl;
      #endif
      
      /// TODO: ONLY TAKE HIGHEST VALUES, NOT MIDDLE ONES?!? <- Already did median filtering!
      /// --- remove elements from D_A1_SFMax which are outside the median value +/- 2sigma
      do
      {
        D_DevOld = D_DevTemp;
        D_DevTemp = sqrt(blitz::sum(pow((D_A1_SFMax) - pfsDRPStella::math::Median(D_A1_SFMax),2)) / D_A1_SFMax.size());
        #ifdef __DEBUG_TELLURIC__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: while: D_DevTemp set to " << D_DevTemp << endl;
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: while: Median(D_A1_SFMax) = " << pfsDRPStella::math::Median(D_A1_SFMax) << endl;
        #endif
        I_A1_IndA.resize(D_A1_SFMax.size());
        
        
        /** ************************************/
        
        
        I_A1_IndA = blitz::where(abs(D_A1_SFMax - pfsDRPStella::math::Median(D_A1_SFMax)) < 2. * D_DevTemp,1,0);
        
        
        /** ************************************/
        
        
        
        blitz::Array<int,1> *P_I_A1_Ind = pfsDRPStella::math::GetIndex(I_A1_IndA, nind_temp);
        #ifdef __DEBUG_TELLURIC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: while: I_A1_IndA set to " << I_A1_IndA << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: while: *P_I_A1_Ind set to " << *P_I_A1_Ind << endl;
        #endif
        P_D_A1_SFMax = new blitz::Array<double, 1>(D_A1_SFMax.size());
        *P_D_A1_SFMax = D_A1_SFMax;
        if (!pfsDRPStella::math::GetSubArrCopy(*P_D_A1_SFMax, *P_I_A1_Ind, D_A1_SFMax)){
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: GetSubArrCopy(P_D_A1_SFMax, P_I_A1_Ind, D_A1_SFMax) returned FALSE" << endl;
          return false;
        }
        delete(P_D_A1_SFMax);
        #ifdef __DEBUG_TELLURIC__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: while: D_A1_SFMax set to " << D_A1_SFMax << endl;
        #endif
        D_A2_MySF_MaxTemp.resize(D_A2_MySF_Max.rows(), D_A2_MySF_Max.cols());
        D_A2_MySF_MaxTemp = D_A2_MySF_Max;
        D_A2_MySF_Max.resize(P_I_A1_Ind->size(), I_NCols_Im);
        for (int nnn=0; nnn<static_cast<int>(P_I_A1_Ind->size()); nnn++){
          #ifdef __DEBUG_CHECK_INDICES__      
            if (((*P_I_A1_Ind)(nnn) < 0) || ((*P_I_A1_Ind)(nnn) >= D_A2_MySF_MaxTemp.rows())){
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: ((*P_I_A1_Ind)(nnn=" << nnn << ")(=" << (*P_I_A1_Ind)(nnn) << ") < 0) || ((*P_I_A1_Ind)(nnn) >= D_A2_MySF_MaxTemp.rows(=" << D_A2_MySF_MaxTemp.rows() << "))" << endl;
              delete(P_I_A1_Ind);
              return false;
            }
          #endif
          D_A2_MySF_Max(nnn,blitz::Range::all()) = D_A2_MySF_MaxTemp((*P_I_A1_Ind)(nnn), blitz::Range::all());
        }
        #ifdef __DEBUG_TELLURIC__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: while: D_A2_MySF_Max set to " << D_A2_MySF_Max << endl;
          S_MySF = DEBUGDIR + std::string("D_A2_MySF_Max") + to_string(I_Iter_Sky_Max) + debugFilesSuffix + std::string(".fits");
          pfsDRPStella::utils::WriteFits(&D_A2_MySF_Max, S_MySF);
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: File " << S_MySF << " written" << endl;
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
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_SFMax = " << D_SFMax << endl;
      #endif
      P_I_A1_SFMaxInd = pfsDRPStella::math::GetIndex(I_A1_SFMaxInd, I_NGood);
      
      /// Create SF array of the highest Slit Functions
      D_A2_MySF_MaxTemp.resize(D_A2_MySF_Max.rows(), I_NCols_Im);
      D_A2_MySF_MaxTemp = D_A2_MySF_Max;
      D_A2_MySF_Max.resize(P_I_A1_SFMaxInd->size(), I_NCols_Im);
      pfsDRPStella::math::GetSubArrCopy(D_A2_MySF_MaxTemp,
                                        *P_I_A1_SFMaxInd,
                                        0,
                                        D_A2_MySF_Max);
      #ifdef __DEBUG_TELLURIC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: P_I_A1_SFMaxInd set to " << *P_I_A1_SFMaxInd << endl;
        S_MySF = DEBUGDIR + std::string("D_A2_MySF_Max_Max") + debugFilesSuffix + std::string(".fits");
        pfsDRPStella::utils::WriteFits(&D_A2_MySF_Max, S_MySF);
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: File " << S_MySF << " written" << endl;
      #endif
      delete(P_I_A1_SFMaxInd);
      
      /// Take median        or sum???           of all rows of D_A2_MySF_Max to define initial SlitFunction
      /// TODO: Switch D_A1_MySF to D_A2_MySF consistently
      D_A1_MySF.resize(I_NCols_Im);
      for (int p=0; p < I_NCols_Im; p++)
      {
        ///      D_A1_MySF(p) = blitz::sum(D_A2_MySF_Max(blitz::Range::all(),p));
        D_A1_MySF(p) = pfsDRPStella::math::Median(D_A2_MySF_Max(blitz::Range::all(),p));
      }
      #ifdef __DEBUG_TELLURIC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A1_MySF set to " << D_A1_MySF << endl;
      #endif
      if (blitz::sum(D_A1_MySF) < 0.00000000000000001)
        D_A1_MySF = 1.;
      D_A1_MySF /= blitz::sum(D_A1_MySF);
      #ifdef __DEBUG_TELLURIC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A1_MySF set to " << D_A1_MySF << endl;
      #endif
      P_D_A1_MySky->resize(I_NRows_Im);

      /// Get initial values for Spectrum and Sky
      P_D_A2_MySF = pfsDRPStella::math::VecArrACrossB(D_A1_IndGen,D_A1_MySF);
      D_A2_MySF.resize(P_D_A2_MySF->rows(), P_D_A2_MySF->cols());
      D_A2_MySF = *P_D_A2_MySF;
      delete(P_D_A2_MySF);
      #ifdef __DEBUG_TELLURIC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A2_MySF set to " << D_A2_MySF << endl;
        S_MySF = DEBUGDIR + std::string("D_A2_MySF_New") + debugFilesSuffix + std::string(".fits");
        pfsDRPStella::utils::WriteFits(&D_A2_MySF, S_MySF);
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: File " << S_MySF << " written" << endl;
      #endif
      
      /// TODO: CHECK: are all D_A1_MySF.size() the same during one execution?
      SFVecArr.resize(D_A1_MySF.size());
      SFVecArr = D_A1_MySF;
      
      argpos = 0;
      if (ErrorsRead)
      {
        #ifdef __DEBUG_SLITFUNC__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": MEASURE_ERRORS_IN set to " << *P_D_A2_Errors << endl;
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
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A2_Im = " << D_A2_Im << endl;
      #endif
      
      D_A1_Sky = 1.;
      #ifdef __DEBUG_TELLURIC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A2_Im = " << D_A2_Im << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: (*P_I_A2_Mask) = " << (*P_I_A2_Mask) << endl;
      #endif
      D_A2_ImTimesMask = D_A2_Im * (*P_I_A2_Mask);
      D_A2_SFTimesMask = D_A2_MySF * (*P_I_A2_Mask);
      #ifdef __DEBUG_TELLURIC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A2_ImTimesMask = " << D_A2_ImTimesMask << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A2_SFTimesMask = " << D_A2_SFTimesMask << endl;
      #endif
      if (!pfsDRPStella::math::LinFitBevington(D_A2_ImTimesMask,
                                               D_A2_SFTimesMask,
                                               D_A1_MySP,
                                               D_A1_Sky,
                                               S_A1_Args_Fit,
                                               PP_Args_Fit))
      {
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: ERROR: Fit returned FALSE!" << endl;
        return false;
      }
      D_A1_OldSky = D_A1_Sky;
      #ifdef __DEBUG_TELLURIC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: after Fit: D_A1_MySP = " << D_A1_MySP << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: after Fit: D_A1_Sky = " << D_A1_Sky << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: after Fit: D_A1_ChiSquare_LinFit = " << D_A1_ChiSquare_LinFit << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: after Fit: D_A1_Probability_LinFit = " << D_A1_Probability_LinFit << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: after Fit: D_A2_Sigma_LinFit = " << D_A2_Sigma_LinFit << endl;
      #endif
      
      spectrum_Out.resize(D_A1_MySP.size());
      spectrum_Out = D_A1_MySP;
      
      /// Save initial sky to P_D_A1_MySky
      (*P_D_A1_MySky) = D_A1_Sky;
      #ifdef __DEBUG_TELLURIC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: D_A1_MySP set to " << D_A1_MySP << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: *P_D_A1_MySky set to " << *P_D_A1_MySky << endl;
        string S_TempA = DEBUGDIR + std::string("D_A2_Im") + debugFilesSuffix + std::string(".fits");
        pfsDRPStella::utils::WriteFits(&D_A2_Im, S_TempA);
      #endif
      for (int p=0; p < static_cast<int>(P_D_A1_MySky->size()); p++)
      {
        /// subtract sky from D_A2_Im
        D_A2_Im(p,blitz::Range::all()) -= D_A1_Sky(p);
        D_A2_Im(p,blitz::Range::all()) = blitz::where(D_A2_Im(p,blitz::Range::all()) < 0. - (3. * _fiberTraceExtractionControl->ccdReadOutNoise), 0., D_A2_Im(p,blitz::Range::all()));
        
        /// Add sky errors to error image
        if (ErrorsRead)
          (*P_D_A2_Errors)(p, blitz::Range::all()) += D_A2_Sigma_LinFit(p,1);
      }
      #ifdef __DEBUG_TELLURIC__
        S_MySF = DEBUGDIR + std::string("D_A2_Im_Minus_Sky") + debugFilesSuffix + std::string(".fits");
        pfsDRPStella::utils::WriteFits(&D_A2_Im, S_MySF);
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TELLURIC == 2: File " << S_MySF << " written" << endl;
      #endif
    }///end if (I_Telluric == 2)
    
    int Pos_Stop = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "STOP");
    int I_Stop = 0;
    if (Pos_Stop >= 0)
    {
      I_Stop = *(int*)(ArgV_In[Pos_Stop]);
    }
    
    for (int p=0; p < D_A2_Im.rows(); p++)
    {
      D_A2_Im(p,blitz::Range::all()) = blitz::where(D_A2_Im(p,blitz::Range::all()) < (0.-(3.*_fiberTraceExtractionControl->ccdReadOutNoise)), 0., D_A2_Im(p,blitz::Range::all()));
    }

    Weight = 1. / (double)overSample_In;
    
    /// Set OIndVecArr to blitz::Array<int,1>(overSample_In + 1) with values = index * (overSample_In + 2)
    UseRowVecArr = i;
    OIndVecArr.resize(overSample_In + 1);
    OIndVecArr = i;
    OIndVecArr *= (overSample_In + 2);
    #ifdef __DEBUG_SLITFUNC_N__
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Weight = " << Weight << endl;
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": UseRowVecArr = " << UseRowVecArr << endl;
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": OIndVecArr = " << OIndVecArr << endl;
    #endif
    
    /// Set N to (number of columns in Im_In * Oversample) + OverSample + 1 (number of sub columns)
    I_NPixSlitF = ((I_NCols_Im + 1) * overSample_In) + 1;
    #ifdef __DEBUG_SLITFUNC_N__
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_NPixSlitF set to " << I_NPixSlitF << endl;
    #endif
    
    /// Get Bad-pixel mask
    if ((Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "BAD")) >= 0)
    {
      if (P_I_A1_JBadVecArr != NULL)
        delete P_I_A1_JBadVecArr;
      P_I_A1_JBadVecArr = (blitz::Array<int, 1>*)ArgV_In[Pos];
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(BAD): P_I_A1_JBadVecArr set to " << *P_I_A1_JBadVecArr << endl;
      #endif
    }/// end if ((Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "BAD")) >= 0)

    if ((Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "USE_ROWS")) >= 0)
    {
      blitz::Array<int, 1> *P_I_A1_UseRows = (blitz::Array<int, 1>*)ArgV_In[Pos];
      UseRowVecArr.resize(P_I_A1_UseRows->size());
      UseRowVecArr = *P_I_A1_UseRows;
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(USE_ROW): ArgV_In[Pos=" << Pos << "]=" << *(int*)(ArgV_In[Pos]) << " => UseRowVecArr set to " << UseRowVecArr << endl;
      #endif
      blitz::Array<double, 2> D_A2_ImUseRows(UseRowVecArr.size(), D_A2_Im.cols());
      if (!pfsDRPStella::math::GetSubArrCopy(D_A2_Im, UseRowVecArr, 0, D_A2_ImUseRows)){
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": GetSubArrCopy(D_A2_Im, UseRowVecArr) returned FALSE" << endl;
        return false;
      }
      D_A2_Im.resize(D_A2_ImUseRows.rows(), D_A2_ImUseRows.cols());
      D_A2_Im = D_A2_ImUseRows;
      I_NRows_Im = D_A2_Im.rows();
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_Im = " << D_A2_Im << endl;
      #endif
      
      blitz::Array<double, 1> D_A1_XCentersUseRow(UseRowVecArr.size());
      if (!pfsDRPStella::math::GetSubArrCopy(XCenVecArr, UseRowVecArr, D_A1_XCentersUseRow)){
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": GetSubArrCopy(XCenVecArr, UseRowVecArr) returned FALSE" << endl;
        return false;
      }
      XCenVecArr.resize(UseRowVecArr.size());
      XCenVecArr = D_A1_XCentersUseRow;
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": XCenVecArr = " << XCenVecArr << endl;
      #endif
      
      blitz::Array<int, 2> I_A2_MaskUseRow(UseRowVecArr.size(), P_I_A2_Mask->cols());
      if (!pfsDRPStella::math::GetSubArrCopy(*P_I_A2_Mask, UseRowVecArr, 0, I_A2_MaskUseRow)){
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": GetSubArrCopy(*P_I_A2_Mask, UseRowVecArr) returned FALSE" << endl;
        return false;
      }
      P_I_A2_Mask->resize(UseRowVecArr.size(), P_I_A2_Mask->cols());
      (*P_I_A2_Mask) = I_A2_MaskUseRow;
      #ifdef __DEBUG_SLITFUNC_A__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": (*P_I_A2_Mask) = " << (*P_I_A2_Mask) << endl;
      #endif
    }/// end if ((Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "USE_ROW")) >= 0)
    if (Pos < 0)// || (Pos >= 0 && TempIntB == 0))
    {
      UseRowVecArr.resize(I_NRows_Im);
      UseRowVecArr = i;
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(USE_ROW): UseRowVecArr set to " << UseRowVecArr << endl;
      #endif
    }
    
    if (blitz::sum((*P_I_A2_Mask)) < 1)
    {
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: blitz::sum((*P_I_A2_Mask)=" << (*P_I_A2_Mask) << ") == 0" << endl;
      return false;
    }
    
    /** Set Norm to number of pixels in Im_In devided by blitz::sum((*P_I_A2_Mask)) **/
    Norm = (I_NRows_Im * I_NCols_Im) / blitz::sum((*P_I_A2_Mask));
    #ifdef __DEBUG_SLITFUNC_N__
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Norm set to " << Norm << endl;
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
          cout << "CFits::SlitFunc: PiskunovOrig: 1. SFVecArr set to " << SFVecArr << endl;
        #endif
        #ifdef __DEBUG_SLITFUNC_FILES__
          string S_SFOut = DEBUGDIR + std::string("PiskunovOrig_SFVecArr1") + debugFilesSuffix + std::string(".fits");
          pfsDRPStella::utils::WriteFits(&SFVecArr, S_SFOut);
        #endif
        if ((overSample_In > 2) && (SFVecArr.size() > 5)){
          SFVecArr = pfsDRPStella::math::MedianVec(SFVecArr, 5);
        }
        if (mean(blitz::sum(D_A2_Im(i,j),j)) < 1000.){
          blitz::Array<double, 1> D_A1_IndGenCols = pfsDRPStella::math::DIndGenArr(D_A2_Im.cols());
          SFVecArr = exp(0. - blitz::pow2((D_A1_IndGenCols + D_XLow) / (D_A2_Im.cols() / 4.)));
        }
        SFVecArr = SFVecArr / blitz::sum(SFVecArr);
        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          string sSFOut = DEBUGDIR + std::string("PiskunovOrig_SFVecArr_DivBySum") + debugFilesSuffix + std::string(".fits");
          pfsDRPStella::utils::WriteFits(&SFVecArr, sSFOut);
          cout << "CFits::SlitFunc: PiskunovOrig: 2. SFVecArr set to " << SFVecArr << endl;
        #endif
        blitz::Array<double, 1> D_A1_Rep = pfsDRPStella::math::Replicate(1., D_A2_Im.rows());
        blitz::Array<double, 2> *P_D_A2_Mat = pfsDRPStella::math::VecArrACrossB(D_A1_Rep, SFVecArr);
        D_A2_ImTimesMask_Guess = D_A2_ImTimesMask_Guess * (*P_D_A2_Mat);
        spectrum_Out = blitz::sum(D_A2_ImTimesMask_Guess(i,j),j) * P_I_A2_Mask->rows() * P_I_A2_Mask->cols() / blitz::sum(*P_I_A2_Mask);
        delete(P_D_A2_Mat);
        if (overSample_In > 2){
          spectrum_Out = pfsDRPStella::math::MedianVec(spectrum_Out, 5);
        }
        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          cout << "CFits::SlitFunc: PiskunovOrig: 1. spectrum_Out set to " << spectrum_Out << endl;
        #endif
        spectrum_Out = (spectrum_Out / blitz::sum(spectrum_Out)) * blitz::sum(D_A2_Im * (*P_I_A2_Mask));
        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          cout << "CFits::SlitFunc: PiskunovOrig: 2. spectrum_Out set to " << spectrum_Out << endl;
        #endif
        P_D_A2_Mat = pfsDRPStella::math::VecArrACrossB(spectrum_Out, SFVecArr);
        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          cout << "CFits::SlitFunc: PiskunovOrig: (*P_D_A2_Mat) set to " << *P_D_A2_Mat << endl;
        #endif
        D_Dev = sqrt(blitz::sum((*P_I_A2_Mask) * blitz::pow2(D_A2_Im - (*P_D_A2_Mat))) / double(blitz::sum(*P_I_A2_Mask)));
        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          cout << "CFits::SlitFunc: PiskunovOrig: D_Dev set to " << D_Dev << endl;
        #endif
        int I_NBad = 0;
        blitz::Array<int, 1> I_A1_WhereDev(D_A2_Im.cols());
        blitz::Array<double, 1> D_A1_WhereDev(D_A2_Im.cols());
        blitz::Array<int, 1> *P_I_A1_WhereDev;
        I_A1_WhereDev = 0;
        for (int i_row = 0; i_row < D_A2_Im.rows(); i_row++){
          D_A1_WhereDev = fabs(D_A2_Im(i_row, blitz::Range::all()) - (*P_D_A2_Mat)(i_row, blitz::Range::all()));
          #ifdef __DEBUG_SLITFUNC_PISKUNOV__
            cout << "CFits::SlitFunc: PiskunovOrig: i_row = " << i_row << ": D_A1_WhereDev = " << D_A1_WhereDev  << endl;
          #endif
          I_A1_WhereDev = blitz::where(D_A1_WhereDev > 3. * D_Dev, 1, 0);
          P_I_A1_WhereDev = pfsDRPStella::math::GetIndex(I_A1_WhereDev, I_NBad);
          if (I_NBad > 0){
            for (int i_ind = 0; i_ind < I_NBad; i_ind++){
              #ifdef __DEBUG_SLITFUNC_PISKUNOV__
                cout << "CFits::SlitFunc: PiskunovOrig: i_row = " << i_row << ": i_ind = " << i_ind << ": (*P_I_A1_WhereDev)(i_ind) = " << (*P_I_A1_WhereDev)(i_ind)  << endl;
              #endif
              (*P_I_A2_Mask)(i_row, (*P_I_A1_WhereDev)(i_ind)) = 0;
            }
          }
          delete(P_I_A1_WhereDev);
        }
        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          cout << "CFits::SlitFunc: PiskunovOrig: (*P_I_A2_Mask) set to " << *P_I_A2_Mask << endl;
        #endif
        D_A2_ImTimesMask_Guess = D_A2_Im * (*P_I_A2_Mask);
        #ifdef __DEBUG_SLITFUNC_FILES__
          string S_ImTimesMask = DEBUGDIR + std::string("PiskunovOrig_ImMTimesMask") + debugFilesSuffix + std::string(".fits");
          pfsDRPStella::utils::WriteFits(&D_A2_ImTimesMask_Guess, S_ImTimesMask);
        #endif
      
        #ifdef __DEBUG_SLITFUNC_PISKUNOV__
          S_SP = DEBUGDIR + std::string("spectrum_Out1") + debugFilesSuffix + std::string(".fits");
          D_A2_SPTemp.resize(spectrum_Out.size(), 1);
          D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
          pfsDRPStella::utils::WriteFits(&D_A2_SPTemp, S_SP);
        #endif
      #else
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_Im = " << D_A2_Im << endl;
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": *P_I_A2_Mask = " << *P_I_A2_Mask << endl;
        #endif
        if (maxIterSig_In > 0){
          for (int p = 0; p < I_NCols_Im; p++)
          {
            D_A2_ImMedian(blitz::Range::all(), p) = pfsDRPStella::math::MedianVec(D_A2_Im(blitz::Range::all(),p), 5, string("NORMAL"));
          }
        }
        D_A2_ImTimesMask = D_A2_ImMedian * (*P_I_A2_Mask);
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_ImTimesMask = " << D_A2_ImTimesMask << endl;
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
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": WARNING: blitz::sum(SFVecArr) == 0 => Setting to 1." << endl;
          #endif
          SFVecArr = 1.;
        }
      
        if (blitz::sum(SFVecArr) < 0.00000000000000001)
          SFVecArr = 1.;
        SFVecArr /= blitz::sum(SFVecArr);           /** Slit Function **/
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": 3.    Slit Function SFVecArr = SFVecArr / (blitz::sum(SFVecArr) = " << blitz::sum(SFVecArr) << ") = " << SFVecArr << endl;
        #endif
      
        /** Initial guess for the spectrum **/
        TempArray.resize(I_NRows_Im, I_NCols_Im);
        TempArray = D_A2_Im;
        TempArray *= (*P_I_A2_Mask);
        #ifdef __DEBUG_SLITFUNC_A__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": (*P_I_A2_Mask) = " << (*P_I_A2_Mask) << endl;
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TempArray = ImM*(*P_I_A2_Mask) = " << TempArray << endl;
        #endif
      
        /** weight rows of im_in with slit function, sum to estimate the spectrum, multiply with Norm, take median over 5 pixels, normalize to blitz::sum(row)=1, and multiply with sum of im_in **/
        blitz::Array<double, 1> d1rep = pfsDRPStella::math::Replicate(1., I_NRows_Im);
        blitz::Array<double, 2> *p_d2mat = pfsDRPStella::math::VecArrACrossB(d1rep, SFVecArr);
        TempArray *= (*p_d2mat);
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": TempArray = TempArray*(VecArrACrossB(Replicate(1.,D_A2_Im.rows(=" << I_NRows_Im << "))=" << d1rep << ", SFVecArr(=" << SFVecArr << "))=" << *p_d2mat << ") = " << TempArray << endl;
        #endif
        delete p_d2mat;
      
        spectrum_Out.resize(I_NRows_Im);
        spectrum_Out = blitz::sum(TempArray, j);
      
        #ifdef __DEBUG_SLITFUNC_FILES__
          S_SP = DEBUGDIR + std::string("spectrum_Out1") + debugFilesSuffix + std::string(".fits");
          D_A2_SPTemp.resize(spectrum_Out.size(), 1);
          D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
          pfsDRPStella::utils::WriteFits(&D_A2_SPTemp, S_SP);
        #endif
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": -3. spectrum_Out (set to blitz::sum(D_A2_Im * (*P_I_A2_Mask), j)) = " << spectrum_Out << endl;
        #endif
        spectrum_Out *= Norm;
      
        #ifdef __DEBUG_SLITFUNC_FILES__
          S_SP = DEBUGDIR + std::string("spectrum_Out1TimesNorm") + debugFilesSuffix + std::string(".fits");
          D_A2_SPTemp.resize(spectrum_Out.size(), 1);
          D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
          pfsDRPStella::utils::WriteFits(&D_A2_SPTemp, S_SP);
        #endif
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": -2. spectrum_Out (*Norm=" << Norm << ") = " << spectrum_Out << endl;
        #endif
      
        spectrum_Out = blitz::where(spectrum_Out < 0., 0., spectrum_Out);
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": -1. spectrum_Out (*Norm=" << Norm << ") = " << spectrum_Out << endl;
        #endif
        spectrum_Out /= blitz::sum(spectrum_Out);
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": 0. spectrum_Out (set to /=blitz::sum(spectrum_Out)) = " << spectrum_Out << endl;
        #endif
          
        #ifdef __DEBUG_SLITFUNC_FILES__
          S_SP = DEBUGDIR + std::string("spectrum_Out1DivBySum") + debugFilesSuffix+ std::string(".fits");
          D_A2_SPTemp.resize(spectrum_Out.size(), 1);
          D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
          pfsDRPStella::utils::WriteFits(&D_A2_SPTemp, S_SP);
        #endif
      
        /// used to be before spectrum_Out /= blitz::sum(spectrum_Out)
        if (abs(blitz::sum(spectrum_Out)) < 0.000000001)
        {
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": WARNING: blitz::sum(spectrum_Out=" << spectrum_Out << ") == 0" << endl;
          spectrum_Out = 1.;
        }
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum(spectrum_Out) =" << blitz::sum(spectrum_Out) << endl;
        #endif
      
        spectrum_Out *= blitz::sum(D_A2_Im * (*P_I_A2_Mask));
      
        #ifdef __DEBUG_SLITFUNC_FILES__
          S_SP = DEBUGDIR + std::string("spectrum_Out1TimesSumImTimesMask") + debugFilesSuffix + std::string(".fits");
          D_A2_SPTemp.resize(spectrum_Out.size(), 1);
          D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
          pfsDRPStella::utils::WriteFits(&D_A2_SPTemp, S_SP);
        #endif
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": spectrum_Out (set to *=blitz::sum(D_A2_Im * (*P_I_A2_Mask))(=" << blitz::sum(D_A2_Im * (*P_I_A2_Mask)) << ")) = " << spectrum_Out << endl;
        #endif
      #endif
    }/// end if (I_Telluric != 2)
//    return false;
    
    /** Add too noisy pixels to bad-pixel mask **/
    Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "NOISE");
    blitz::Array<double, 2> D_A2_TempBB(P_I_A2_Mask->rows(), P_I_A2_Mask->cols());
    if (Pos >= 0)
    {
      Dev = *(double*)ArgV_In[Pos];
      D_A1_Dev = Dev;
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(NOISE): Dev set to " << Dev << endl;
      #endif
      
    }
    if (Pos < 0 || (Pos >= 0 && abs(Dev) < 0.00000000000000001))
    {
      if (blitz::sum((*P_I_A2_Mask)) < 1)
      {
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: blitz::sum((*P_I_A2_Mask)=" << (*P_I_A2_Mask) << ") == 0" << endl;
        return false;
      }
    }
    #ifdef __DEBUG_SLITFUNC_N__
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": 1. spectrum_Out = " << spectrum_Out << endl;
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": SFVecArr = " << SFVecArr << endl;
    #endif
    #ifdef __DEBUG_CHECK_INDICES__      
      if (D_A2_Im.cols() != static_cast<int>(SFVecArr.size())){
        cout << "CFits::SlitFunc: ERROR: D_A2_Im.cols() != SFVecArr.size() => Returning FALSE" << endl;
        return false;
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
    //blitz::Array<double, 1> Rep = pfsDRPStella::math::Replicate(1., D_A2_Im.rows());
    blitz::Array<double, 2> *p_SFArr = pfsDRPStella::math::VecArrACrossB(pfsDRPStella::math::Replicate(1., D_A2_Im.rows()), SFVecArr);
    blitz::Array<double, 1> D_A1_Sky_Tmp(D_A2_Im.rows());
    bool B_WithSky = true;
    if (I_Telluric == 1 || I_Telluric == 3)
      B_WithSky = false;
    #ifdef __DEBUG_SLITFUNC__
      cout << "CFits::SlitFunc: Before FitSig: SFVecArr = " << SFVecArr << endl;
      cout << "CFits::SlitFunc: D_A2_Im = " << D_A2_Im << endl;
      cout << "CFits::SlitFunc: *p_SFArr = " << *p_SFArr << endl;
      if (B_WithSky)
        cout << "SlitFunc: B_WithSky = true" << endl;
      cout << "SlitFunc: D_Reject = " << D_Reject << endl;
      if (ErrorsRead)
        cout << "SlitFunc: *P_D_A2_Errors = " << *P_D_A2_Errors << endl;
      cout << "SlitFunc: I_A2_MaskFit = " << I_A2_MaskFit << endl;
    #endif
    if (!pfsDRPStella::math::LinFitBevington(D_A2_Im,
                                             *p_SFArr,
                                             D_A1_SP_Tmp,
                                             D_A1_Sky_Tmp,
                                             B_WithSky,
                                             S_A1_Args_FitSig,
                                             PP_Void_FitSig)){
      cout << "CFits::SlitFunc: ERROR: FitSig returned FALSE" << endl;
      return false;
    }
    *P_I_A2_Mask = I_A2_MaskFit;
    #ifdef __DEBUG_SLITFUNC_N__
      string S_MaskSig = DEBUGDIR + std::string("MaskFitSig") + debugFilesSuffix + std::string(".fits");
      pfsDRPStella::utils::WriteFits(&I_A2_MaskFit, S_MaskSig);
      cout << "CFits::SlitFunc: After FitSig: I_A2_MaskFit = " << I_A2_MaskFit << endl;
    #endif
    delete(p_SFArr);
    free(PP_Void_FitSig);
      
    SFVecArr = blitz::where(SFVecArr < 0., 0., SFVecArr);
    #ifdef __DEBUG_SLITFUNC_N__
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": 2c.    Slit Function SFVecArr = SFVecArr / (blitz::sum(SFVecArr) = " << blitz::sum(SFVecArr) << ") = " << SFVecArr << endl;
    #endif
      
    if (blitz::sum(SFVecArr) < 0.00000000000000001)
      SFVecArr = 1.;
    SFVecArr /= blitz::sum(SFVecArr);           /** Slit Function **/
    #ifdef __DEBUG_SLITFUNC__
      cout << "CFits::SlitFunc: c) SFVecArr = " << SFVecArr << endl;
      cout << "CFits::SlitFunc: D_A1_SP_Tmp = " << D_A1_SP_Tmp << endl;
    #endif
    blitz::Array<double, 2> *p_tempMatA = pfsDRPStella::math::VecArrACrossB(D_A1_SP_Tmp, SFVecArr);
    blitz::Array<double, 2> *p_tempMat = new blitz::Array<double, 2>(D_A2_Im.rows(), D_A2_Im.cols());
    blitz::Array<double, 2> *p_tempMatB = new blitz::Array<double, 2>(D_A2_Im.rows(), D_A2_Im.cols());
    *p_tempMat = D_A2_Im - (*p_tempMatA);
    #ifdef __DEBUG_SLITFUNC_FILES__
      string S_ImMinusRec = DEBUGDIR + std::string("ImMinusRec") + debugFilesSuffix + std::string(".fits");
      pfsDRPStella::utils::WriteFits(p_tempMat, S_ImMinusRec);
    #endif
    #ifdef __DEBUG_SLITFUNC__
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(NOISE): *p_tempMatA " << *p_tempMatA << endl;
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(NOISE): *p_tempMat " << *p_tempMat << endl;
    #endif
    *p_tempMatB = blitz::pow2(*p_tempMat);
    for (unsigned int iter_sig = 0; iter_sig < maxIterSig_In; iter_sig++)
    {
      pfsDRPStella::math::Double((*P_I_A2_Mask), D_A2_Mask);
      #ifdef __DEBUG_CHECK_INDICES__      
        if (P_I_A2_Mask->rows() != D_A2_Mask.rows()){
          cout << "CFits::SlitFunc: ERROR: 1. P_I_A2_Mask->rows() != D_A2_Mask.rows()" << endl;
          return false;
        }
      #endif
      Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "NOISE");
      if (Pos < 0 || (Pos >= 0 && abs(Dev) < 0.00000000000000001))
      {
        for (int i_col=0; i_col < static_cast<int>(D_A2_Im.cols()); i_col++){
          if (blitz::sum((*P_I_A2_Mask)(blitz::Range::all(), i_col)) == 0){
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(NOISE): iter_sig = " << iter_sig << ": i_col = " << i_col << ": ERROR: blitz::sum((*P_I_A2_Mask)(blitz::Range::all(), i_col)) = " << blitz::sum((*P_I_A2_Mask)(blitz::Range::all(), i_col)) << endl;
            return false;
          }
          else{
            if (fabs(blitz::sum(D_A2_Mask)) < 0.000000000001){
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(NOISE): iter_sig = " << iter_sig << ": i_col = " << i_col << ": ERROR:  fabs(blitz::sum(D_A2_Mask)=" << blitz::sum(D_A2_Mask) << ") < 0.000000000001" << endl;
              return false;
            }
            D_A1_Dev(i_col) = sqrt(blitz::sum(D_A2_Mask(blitz::Range::all(), i_col) * ((*p_tempMatB)(blitz::Range::all(), i_col))) / blitz::sum(D_A2_Mask(blitz::Range::all(), i_col)));
          }
        }
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(NOISE): iter_sig = " << iter_sig << ": D_A1_Dev set to " << D_A1_Dev << endl;
        #endif
      }
      else{
        cout << "CFits::SlitFunc: KeyWord_Set(NOISE): D_A1_Dev = " << D_A1_Dev << endl;
      }
        
      pfsDRPStella::math::Double((*P_I_A2_Mask), D_A2_Mask);
      #ifdef __DEBUG_CHECK_INDICES__      
        if (P_I_A2_Mask->rows() != D_A2_Mask.rows()){
          cout << "CFits::SlitFunc: ERROR: 0. P_I_A2_Mask->rows() != D_A2_Mask.rows()" << endl;
          return false;
        }
      #endif
      #ifdef __DEBUG_SLITFUNC_N__
        D_A2_TempBB = fabs(D_A2_Mask * (*p_tempMat));
        cout << " iter_sig = " << iter_sig << ": fabs(P_I_A2_Mask(= " << *P_I_A2_Mask << ") * p_tempMat(= " << *p_tempMat << ")) = " << D_A2_TempBB << ", D_A1_Dev = " << D_A1_Dev << endl;
      #endif
        
      for (int i_col = 0; i_col < D_A2_Im.cols(); i_col++){
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": iter_sig = " << iter_sig << ": (*p_tempMatB)(blitz::Range::all(), i_col=" << i_col << ") = " << (*p_tempMatB)(blitz::Range::all(), i_col) << endl;
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": iter_sig = " << iter_sig << ": D_A1_Dev(i_col=" << i_col << ") = " << D_A1_Dev(i_col) << endl;
        #endif
        #ifndef __PISKUNOV_ORIG__
          if (maxIterSig_In > 0){
            (*P_I_A2_Mask)(blitz::Range::all(), i_col) = blitz::where(sqrt(D_A2_Mask(blitz::Range::all(), i_col) * ((*p_tempMatB)(blitz::Range::all(), i_col))) > (6.5 * D_A1_Dev(i_col)), 0, (*P_I_A2_Mask)(blitz::Range::all(), i_col));
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": iter_sig = " << iter_sig << ": (*P_I_A2_Mask)(blitz::Range::all(), i_col=" << i_col << ") = " << (*P_I_A2_Mask)(blitz::Range::all(), i_col) << endl;
            #endif
          }
        #endif
      }///end for (int i_col = 0; i_col < D_A2_Im.cols(); i_col++){
      #ifdef __DEBUG_SLITFUNC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": iter_sig = " << iter_sig << ": (*P_I_A2_Mask) = " << (*P_I_A2_Mask) << endl;
      #endif
      pfsDRPStella::math::Double((*P_I_A2_Mask), D_A2_Mask);
      #ifdef __DEBUG_CHECK_INDICES__      
        if (P_I_A2_Mask->rows() != D_A2_Mask.rows()){
          cout << "CFits::SlitFunc: ERROR: P_I_A2_Mask->rows() != D_A2_Mask.rows()" << endl;
          return false;
        }
        for (int i_row=0; i_row<P_I_A2_Mask->rows(); i_row++){
          for (int i_col=0; i_col<P_I_A2_Mask->cols(); i_col++){
            if ((*P_I_A2_Mask)(i_row, i_col) != int(D_A2_Mask(i_row, i_col))){
              cout << "CFits::SlitFunc: ERROR: (*P_I_A2_Mask)(i_row, i_col)(=" << (*P_I_A2_Mask)(i_row, i_col) << ") != int(D_A2_Mask(i_row, i_col))(=" << int(D_A2_Mask(i_row, i_col)) << ")" << endl;
              return false;
            }
          }
        }
      #endif
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "blitz::sum(*P_I_A2_MaskIn) = " << blitz::sum(*P_I_A2_MaskIn) << endl;
        cout << "blitz::sum(*P_I_A2_Mask) = " << blitz::sum(*P_I_A2_Mask) << endl;
        blitz::Array<double,2> D_A2_ImTemp(D_A2_Im.rows(), D_A2_Im.cols());
        D_A2_ImTemp = D_A2_Im - (*p_tempMatA);
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_Im[" << D_A2_Im.rows() << ", " << D_A2_Im.cols() << "]" << ", p_tempMatA[" << p_tempMatA->rows() << ", " << p_tempMatA->cols() << "]" << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_Im - spectrum_Out x SFVecArr = " << D_A2_ImTemp << endl;
      #endif
        
      #ifdef __DEBUG_SLITFUNC_FILES__
        string S_Mask = DEBUGDIR + std::string("Mask_IterSig") + to_string(iter_sig) + debugFilesSuffix + std::string(".fits");
        pfsDRPStella::utils::WriteFits(P_I_A2_Mask, S_Mask);
        
        S_Mask = DEBUGDIR + std::string("ImInTimesMask_IterSig") + to_string(iter_sig) + debugFilesSuffix + std::string(".fits");
        D_A2_ImTimesMask = D_A2_Im * (*P_I_A2_Mask);
        pfsDRPStella::utils::WriteFits(&D_A2_ImTimesMask, S_Mask);
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
//      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": XVecArr = " << XVecArr << endl;
//    #endif
    BKLIndVecArr.resize(overSample_In + 1);
    BKLIndVecArr = i;
    BKLIndVecArr += (I_NPixSlitF * overSample_In);
    #ifdef __DEBUG_SLITFUNC_N__
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": BKLIndVecArr = " << BKLIndVecArr << endl;
    #endif
      
    OLIndVecArr.resize(overSample_In + 1);
    #ifdef __DEBUG_CHECK_INDICES__      
      if (static_cast<int>(OIndVecArr.size()) < overSample_In+1)
      {
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: size of OIndVecArr(=" << OIndVecArr.size() << " < overSample_In + 1(=" << overSample_In + 1 << ")" << endl;
        return false;
      }
    #endif
    OLIndVecArr = OIndVecArr(blitz::Range(0, overSample_In));
    #ifdef __DEBUG_SLITFUNC_N__
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": OLIndVecArr = " << OLIndVecArr << endl;
    #endif
      
    for (long m=overSample_In + 1; m <= (2 * overSample_In); m++)
    {
      long mm = m - overSample_In;
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m=" << m << "): mm = " << mm << endl;
      #endif
        
      /// attach another Vector<int>(osample+1-mm) to BKLIndVecArr with values index + (N * m)
      TempIVecArr.resize(overSample_In + 1 - mm);
      TempIVecArr = i + (I_NPixSlitF * m);
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m=" << m << "): TempIVecArr = " << TempIVecArr << endl;
      #endif
        
      int oldsize = BKLIndVecArr.size();
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m=" << m << "): oldsize(BKLIndVecArr) = " << oldsize << endl;
      #endif
      BKLIndVecArr.resizeAndPreserve(oldsize + overSample_In + 1 - mm);
      BKLIndVecArr(blitz::Range(oldsize, blitz::toEnd))//oldsize + 1 + overSample_In - mm))
        = TempIVecArr(blitz::Range::all());
//      #ifdef __DEBUG_SLITFUNC_X__
//        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m=" << m << "): BKLIndVecArr = " << BKLIndVecArr << endl;
//      #endif
        
      oldsize = OLIndVecArr.size();
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m=" << m << "): oldsize(OLIndVecArr) = " << oldsize << endl;
      #endif
      OLIndVecArr.resizeAndPreserve(OLIndVecArr.size() + overSample_In + 1 - mm);
      #ifdef __DEBUG_CHECK_INDICES__      
        if (static_cast<int>(OIndVecArr.size()) < overSample_In - mm + 1)
        {
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: size of OIndVecArr(=" << OIndVecArr.size() << ") < overSample_In - mm + 1(=" << overSample_In - mm + 1 << ")" << endl;
          return false;
        }
      #endif
      OLIndVecArr(blitz::Range(oldsize, blitz::toEnd))
        = OIndVecArr(blitz::Range(0, overSample_In - mm)) + mm;
//      #ifdef __DEBUG_SLITFUNC_X__
//        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m=" << m << "): OLIndVecArr = " << OLIndVecArr << endl;
//      #endif
    }/// end for (long m=overSample_In + 1; m <= (2 * overSample_In); m++)
//    #ifdef __DEBUG_SLITFUNC_X__
//      cout << "SlitFunc: OLIndVecArr = " << OLIndVecArr << endl;
//      cout << "SlitFunc: BKLIndVecArr = " << BKLIndVecArr << endl;
//      return false;
//    #endif
      
    SPOldVecArr.resize(spectrum_Out.size());
    SPOldVecArr = 0.;
    #ifdef __DEBUG_SLITFUNC__
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": SPOldVecArr = " << SPOldVecArr << endl;
    #endif
      
    pfsDRPStella::math::Double((*P_I_A2_Mask), D_A2_Mask);
      
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
      ///TODO OR PLUS XCenVecArr(m)???????????????
      D_A2_XX(m,blitz::Range::all()) = XVecArr + XCenVecArr(m);    /** Offset SFVecArr **/
//      #ifdef __DEBUG_SLITFUNC_X__
//        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": line 10600: XVecArr = " << XVecArr << endl;
//        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": line 10600: _xCenters[m=" << m << "] = " << _xCenters[m] << endl;
//        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": XCenVecArr(m=" << m << ") = " << XCenVecArr(m) << endl;
//        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): D_A2_XX(" << m << ", *) set to " << D_A2_XX(m,blitz::Range::all()) << endl;
//      #endif
      #ifdef __DEBUG_CHECK_INDICES__      
        if (D_A2_XX.cols() != static_cast<int>(XVecArr.size()))
        {
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: D_A2_XX.cols(=" << D_A2_XX.cols() << ") != size of XVecArr(=" << XVecArr.size() << ")" << endl;
          return false;
        }
      #endif
      
      /** Weights are the same for all pixels except for the first and the last subpixels **/
      TempIVecArr.resize(D_A2_XX.cols());
      TempIVecArr = blitz::where((D_A2_XX(m,blitz::Range::all()) > 0) && (D_A2_XX(m,blitz::Range::all()) < 1), 1, 0);
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): TempIVecArr set to " << TempIVecArr << endl;
      #endif
      pfsDRPStella::math::GetIndex(TempIVecArr, NInd, IndVecArr);
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): NInd = " << NInd << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): IndVecArr set to " << IndVecArr << endl;
      #endif
      I_A1_IFirstPix(m) = IndVecArr(0);
      
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): I_A1_IFirstPix(" << m << ") = " << I_A1_IFirstPix(m) << endl;
      #endif
      I_A1_ILastPix(m)  = IndVecArr(NInd - 1);
      #ifdef __DEBUG_SLITFUNC_N__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): I_A1_ILastPix(" << m << ") = " << I_A1_ILastPix(m) << endl;
      #endif
      
      #ifdef __DEBUG_SLITFUNC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): D_A2_XX(m,*) = " << D_A2_XX(m,blitz::Range::all()) << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): D_A2_Im->cols = " << I_NCols_Im << endl;
      #endif
        
      TempIVecArr = blitz::where((D_A2_XX(m,blitz::Range::all()) >= 0.) && (D_A2_XX(m,blitz::Range::all()) < (double)(I_NCols_Im)), 1, 0);
        
      #ifdef __DEBUG_SLITFUNC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): TempIVecArr(where) = " << TempIVecArr << endl;
      #endif
        
      i_tmp_sum = blitz::sum(TempIVecArr);
      if (i_tmp_sum == 0)
      {
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): ERROR: i_tmp_sum == 0!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        return false;
      }
      pfsDRPStella::math::GetIndex(TempIVecArr, I_NInd, IFirstVecArr);
      I_A1_IFirstSpec(m) = IFirstVecArr(0);
      I_A1_ILastSpec(m) = IFirstVecArr(IFirstVecArr.size() - 1);
      #ifdef __DEBUG_SLITFUNC__
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): I_A1_IFirstSpec(m) set to " << I_A1_IFirstSpec(m) << endl;
        cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): I_A1_ILastSpec(m) set to " << I_A1_ILastSpec(m) << endl;
      #endif
        
    }/// end for (int m = 0; m < I_NRows_Im; m++)
    #ifdef __DEBUG_SLITFUNC_X__
      std::string fname_ifirst = DEBUGDIR + std::string("IFirstPix") + debugFilesSuffix + std::string(".dat");
      pfsDRPStella::utils::WriteArrayToFile(I_A1_IFirstPix, fname_ifirst, std::string("ascii"));
    #endif

    for (int mm=0; mm < I_NCols_Im; mm++)
    {
      D_A1_XProf(mm) = double(mm) + 0.5 + (1. / (2. * static_cast<double>(_fiberTraceExtractionControl->overSample)));
    }

    /// fit spline3?
    if (_fiberTraceExtractionControl->profileInterpolation.compare(_fiberTraceExtractionControl->PROFILE_INTERPOLATION_NAMES[1]) == 0){
      if (!fitSpline(D_A2_Im,
                     I_A1_IFirstPix,
                     XVecArr,
                     SFVecArr,
                     D_A2_XX,
                     D_A1_XProf,
                     profile_Out)){
        cout << "SlitFunc: ERROR: fitSpline returned FALSE" << endl;
        return false;
      }
    }
    else{/// Piskunov
      do
      {
        I_Iter_Sky++;
        I_Iter_SF = 0;
        while(I_Iter_SF < static_cast<int>(_fiberTraceExtractionControl->maxIterSF))   /** Iteration counter **/
        {
          I_Iter_SF++;
            
          #ifdef __DEBUG_SLITFUNC_SF_N__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": (max(abs(spectrum_Out - SPOldVecArr=" << SPOldVecArr << ") / max(spectrum_Out)=" << max(spectrum_Out) << ")=" << max(fabs(spectrum_Out - SPOldVecArr) / max(spectrum_Out)) << endl;
          #endif
            
          if ((I_Iter_SF == 1) || (max(fabs(spectrum_Out - SPOldVecArr) / max(spectrum_Out)) > 0.00001))
          {
  //          if ((AKLArr.rows() != AKLArrTemp.rows()) || (AKLArr.cols() != AKLArrTemp.cols())){
  //            cout << "SlitFunc: ERROR: (AKLArr.rows() != AKLArrTemp.rows()) || (AKLArr.cols() != AKLArrTemp.cols()) => returning FALSE" << endl;
  //            return false;
  //          }
  //          AKLArr = AKLArrTemp
            AKLArr.resize(I_NPixSlitF, (2*overSample_In) + 1); /** Initialize Matrix **/
            AKLArr = 0.;
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": if(max...): AKLArr initialized to 0.: size(AKLArr) = (" << AKLArr.rows() << "," << AKLArr.cols() << ")" << endl;
            #endif
              
            BLVecArr.resize(I_NPixSlitF);                      /** and RHS **/
            BLVecArr = 0.;
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": if(max...): BLVecArr = " << BLVecArr << endl;
            #endif
              
            OmegaVecArr.resize(overSample_In + 1);
            OmegaVecArr = Weight;                    /** Replicate constant Weights **/
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": if(max...): OmegaVecArr = " << OmegaVecArr << endl;
            #endif
              
            pfsDRPStella::math::Double((*P_I_A2_Mask), D_A2_Mask);
            for (int m = 0; m < I_NRows_Im; m++)  /** Fill up matrix and RHS **/
            {
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Begin for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++)" << endl;
              #endif
                
              /** Fix the first and the last subpixels, here the weight is split between the two subpixels **/
              OmegaVecArr(0) = D_A2_XX(m,I_A1_IFirstPix(m));
                
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OmegaVecArr(0) set to " << OmegaVecArr(0) << endl;
              #endif
                
              OmegaVecArr(overSample_In) = 1. - D_A2_XX(m, I_A1_ILastPix(m));
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OmegaVecArr(overSample_In=" << overSample_In << ") set to " << OmegaVecArr(overSample_In) << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OmegaVecArr set to " << OmegaVecArr << endl;
              #endif
              D_A2_Weights(m, blitz::Range::all()) = OmegaVecArr;
                
              /** Band-diagonal part that will contain omega#omega **/
              BKLArr.resize(I_NPixSlitF, (2 * overSample_In) + 1);
              BKLArr = 0.;
                
              blitz::Array<double, 2> *p_OArr = pfsDRPStella::math::VecArrACrossB(OmegaVecArr, OmegaVecArr);
              OArr.resize(p_OArr->rows(),p_OArr->cols());//OmegaVecArr.size(), OmegaVecArr.size());
              OArr = (*p_OArr);
              delete p_OArr;
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): 1. OArr set to " << OArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              #endif
                
              tmpdbl = OArr(overSample_In, overSample_In);
              tmpdbl += OArr(0, 0);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): tmpdbl set to " << tmpdbl << endl;
              #endif
              OArr(overSample_In, overSample_In) = tmpdbl;
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OArr(overSample_In=" << overSample_In << ", overSample_In) set to " << OArr(overSample_In, overSample_In) << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OArr = " << OArr << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OLIndVecArr = " << OLIndVecArr << endl;
              #endif
              //cout << "resizing OOVecArr to size " << OLIndVecArr.size() << endl;
              OOVecArr.resize(OLIndVecArr.size());
              OOVecArr = 0.;
            // cout << "OOVecArr resized to " << OOVecArr.size() << endl;
  //            return false;
              for (unsigned int n = 0; n < OLIndVecArr.size(); n++)
              {
                tempcol = pfsDRPStella::math::GetColFromIndex(OLIndVecArr(n), OArr.rows());
                temprow = pfsDRPStella::math::GetRowFromIndex(OLIndVecArr(n), OArr.rows());
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for (int n(=" << n << ") = 0; n < OLIndVecArr.size(=" << OLIndVecArr.size() << "); n++): setting OOVecArr(n) to OArr(OLIndVecArr(temprow(=" << temprow << "), tempcol(=" << tempcol << "))=" << OArr(temprow, tempcol) << endl;
                #endif
                #ifdef __DEBUG_CHECK_INDICES__      
                  if (temprow >= OArr.rows())
                  {
                    cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: temprow(=" << temprow << ") >= OArr.rows(=" << OArr.rows() << ")" << endl;
                    return false;
                  }
                  if (tempcol >= OArr.cols())
                  {
                    cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: tempcol(=" << tempcol << ") >= OArr.cols(=" << OArr.cols() << ")" << endl;
                    return false;
                  }
                #endif
                OOVecArr(n) = OArr(temprow, tempcol);
              }
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OOVecArr set to " << OOVecArr << endl;
              #endif
                
              for (int n = 0; n < I_NCols_Im; n++)
              {
                for (unsigned int o = 0; o < BKLIndVecArr.size(); o++)
                {
                  #ifdef __DEBUG_SLITFUNC_N__
                    cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": n(=" << n << ") * overSample_In(=" << overSample_In << ") + I_A1_IFirstPix(m)(=" << I_A1_IFirstPix(m) << ") + BKLIndVecArr(o=" << o << ")=" << BKLIndVecArr(o) << " = " << (n * overSample_In) + I_A1_IFirstPix(m) + BKLIndVecArr(o) << endl;
                    cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": BKLIndVecArr = " << BKLIndVecArr << endl;
                    cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": OOVecArr = " << OOVecArr << endl;
                  #endif
                    
                  tempcol = pfsDRPStella::math::GetColFromIndex((n * overSample_In) + I_A1_IFirstPix(m) + BKLIndVecArr(o), BKLArr.rows());
                  temprow = pfsDRPStella::math::GetRowFromIndex((n * overSample_In) + I_A1_IFirstPix(m) + BKLIndVecArr(o), BKLArr.rows());
                    
                  #ifdef __DEBUG_CHECK_INDICES__      
                    if (temprow >= BKLArr.rows())
                    {
                      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: temprow(=" << temprow << ") >= BKLArr.rows(=" << BKLArr.rows() << ")" << endl;
                      return false;
                    }
                    
                    if (tempcol >= BKLArr.cols())
                    {
                      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: tempcol(=" << tempcol << ") >= BKLArr.cols(=" << BKLArr.cols() << ")" << endl;
                      return false;
                    }
                    
                    if (o >= OOVecArr.size())
                    {
                      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: o(=" << o << ") >= OOVecArr.size(=" << OOVecArr.size() << ")" << endl;
                      return false;
                    }
                    if (m >= P_I_A2_Mask->rows())
                    {
                      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: m(=" << m << ") >= P_I_A2_Mask->rows(=" << P_I_A2_Mask->rows() << ")" << ", P_I_A2_Mask->cols(=" << P_I_A2_Mask->cols() << ")" << ", I_NRows_Im = " << I_NRows_Im << endl;
                      return false;
                    }
                    if (n >= P_I_A2_Mask->cols())
                    {
                      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: n(=" << n << ") >= P_I_A2_Mask->cols(=" << P_I_A2_Mask->cols() << "), (*P_I_A2_Mask).rows(=" << P_I_A2_Mask->rows() << "), P_I_A2_Mask->size() = " << P_I_A2_Mask->size() << ", I_NCols_Im = " << I_NCols_Im << endl;
                      return false;
                    }
                  #endif
                  BKLArr(temprow, tempcol) = OOVecArr(o) * D_A2_Mask(m,n);
                    
                  #ifdef __DEBUG_SLITFUNC_N__
                    cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=0; n<NCols(=" << I_NCols_Im << "); n++): for(o(=" << o << ")=0; o<BKLIndVecArr(=" << BKLIndVecArr.size() << "); o++): BKLArr(((n * overSample_In) + I_A1_IFirstPix(m) + BKLIndVecArr(o) = " << (n * overSample_In) + I_A1_IFirstPix(m) + BKLIndVecArr(o) << ")(= temprow=" << temprow << ", tempcol=" << tempcol << ")) set to " << BKLArr(temprow, tempcol) << endl;
                  #endif
                }/// end for (int o = 0; o < BKLIndVecArr.size(); o++)
              }/// end for (int n = 0; n < I_NCols_Im; n++)
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): BKLArr set to " << BKLArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OOVecArr set to " << OOVecArr << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): BKLArr set to " << BKLArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              #endif
                
  //            OOVecArr.resize(1);
              double OOVal = OArr(overSample_In, overSample_In);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OOVal set to " << OOVal << endl;
              #endif
                
              for (int n = 1; n < I_NCols_Im; n++)
              {
                #ifdef __DEBUG_CHECK_INDICES__      
                  if ((n*overSample_In) + I_A1_IFirstPix(m) >= BKLArr.rows())
                  {
                    cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: (n*overSample_In) + I_A1_IFirstPix(m) = " << (n*overSample_In) + I_A1_IFirstPix(m) << " >= BKLArr.rows(=" << BKLArr.rows() << ")" << endl;
                    return false;
                  }
                #endif
                BKLArr((n * overSample_In) + I_A1_IFirstPix(m), overSample_In) = OOVal * D_A2_Mask(m, n);
              }/// end for (int n = 1; n < I_NCols_Im; n++)
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): BKLArr set to " << BKLArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              #endif
                
              #ifdef __DEBUG_CHECK_INDICES__      
                if (I_NCols_Im * overSample_In + I_A1_IFirstPix(m) >= BKLArr.rows())
                {
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: I_NCols_Im(=" << I_NCols_Im << ") * overSample_In(=" << overSample_In << ") + I_A1_IFirstPix(m)(=" << I_A1_IFirstPix(m) << ") = " << I_NCols_Im * overSample_In + I_A1_IFirstPix(m) << " >= BKLArr.rows(=" << BKLArr.rows() << ")" << endl;
                  return false;
                }
              #endif
              BKLArr((I_NCols_Im * overSample_In) + I_A1_IFirstPix(m), overSample_In) = pow(OmegaVecArr(overSample_In),2) * D_A2_Mask(m, I_NCols_Im - 1);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): BKLArr((I_NCols_Im * overSample_In) + I_A1_IFirstPix(m)=" << (I_NCols_Im * overSample_In) + I_A1_IFirstPix(m) << ", overSample_In=" << overSample_In << ") set to " << BKLArr((I_NCols_Im * overSample_In) + I_A1_IFirstPix(m), overSample_In) << endl;
              #endif
              for (int o = 0; o < overSample_In; o++)
              {
                #ifdef __DEBUG_CHECK_INDICES__      
                  if (I_NPixSlitF-1 >= BKLArr.rows())
                  {
                    cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: N-1 = " << I_NPixSlitF-1 << " >= BKLArr.rows(=" << BKLArr.rows() << ")" << endl;
                    return false;
                  }
                  if (o >= BKLArr.cols())
                  {
                    cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: o = " << o << " >= BKLArr.cols(=" << BKLArr.cols() << ")" << endl;
                    return false;
                  }
                  if (2*overSample_In-o >= BKLArr.cols())
                  {
                    cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: 2*overSample_In-o = " << 2*overSample_In-o << " >= BKLArr.cols(=" << BKLArr.cols() << ")" << endl;
                    return false;
                  }
                #endif
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(o(=" << o << ")=0; o<overSample_In(=" << overSample_In << "); o++): BKLARR(blitz::Range = (overSample_In - o = " << overSample_In - o << ", " << endl;
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(o(=" << o << ")=0; o<overSample_In(=" << overSample_In << "); o++): BKLArr(blitz::Range(overSample_In(=" << overSample_In << ")-o(=" << o << ") = " << overSample_In - o << "), I_NPixSlitF(=" << I_NPixSlitF << " - 1 = " << I_NPixSlitF - 1 << "), o = " << o << ") = " << BKLArr(blitz::Range(overSample_In-o, I_NPixSlitF - 1), o) << endl;
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(o(=" << o << ")=0; o<overSample_In(=" << overSample_In << "); o++): BKLARR(blitz::Range = (0, I_NPixSlitF(=" << I_NPixSlitF - 1 << " - 1 - overSample_In + o) = " << I_NPixSlitF - 1 - overSample_In + o << "), 2 * overSample_In - o) = " << 2*overSample_In - o << ") = " << BKLArr(blitz::Range(0, I_NPixSlitF - 1 - overSample_In - o), 2 * overSample_In - o) << endl;
                #endif
                  
                BKLArr(blitz::Range(overSample_In-o, I_NPixSlitF - 1), o) = BKLArr(blitz::Range(0, I_NPixSlitF - 1 - overSample_In + o), 2 * overSample_In - o);
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(o(=" << o << ")=0; o<overSample_In(=" << overSample_In << "); o++): BKLArr(blitz::Range(overSample_In-o=" << overSample_In-o << ", I_NPixSlitF-1=" << I_NPixSlitF-1 << "),o=" << o << ") set to " << BKLArr(blitz::Range(overSample_In-o, I_NPixSlitF - 1), o) << endl;
                #endif
              }/// end for (int o = 0; o < overSample_In; o++)
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): BKLArr set to " << BKLArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): AKLArr = " << AKLArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
                double D_SPVecPow = pow(spectrum_Out(m), 2);
                D_A2_SPVecTimesBKLArr.resize(BKLArr.rows(), BKLArr.cols());
                D_A2_SPVecTimesBKLArr = 0.;
                D_A2_SPVecTimesBKLArr = D_SPVecPow * BKLArr;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): D_SPVecPow(= " << D_SPVecPow << ") * BKLArr(= " << BKLArr << ") = " << D_A2_SPVecTimesBKLArr << endl;
              #endif
              #ifdef __DEBUG_CHECK_INDICES__      
                if (AKLArr.size() != BKLArr.size())
                {
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: size of AKLArr(=" << AKLArr.size() << ") != size of BKLArr(=" << BKLArr.size() << ")";
                  return false;
                }
              #endif
              AKLArr += (pow(spectrum_Out(m), 2) * BKLArr);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): AKLArr(+= (pow(spectrum_Out(m), 2) * BKLArr)) set to " << AKLArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              #endif
              OArr.resize(I_NPixSlitF, 1);
              OArr = 0.;
              for (int n = 0; n < I_NCols_Im; n++)
              {
                OArr(blitz::Range((n * overSample_In) + I_A1_IFirstPix(m), ((n+1) * overSample_In) + I_A1_IFirstPix(m)), 0) = D_A2_Im(m, n) * Weight * D_A2_Mask(m, n);
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OArr(blitz::Range(n * overSample_In) + I_A1_IFirstPix(m)=" << (n * overSample_In) + I_A1_IFirstPix(m) << ", n * overSample_In + I_A1_IFirstPix(m) + overSample_In=" << n * overSample_In + I_A1_IFirstPix(m) + overSample_In << "), 0) set to " << OArr(blitz::Range((n * overSample_In) + I_A1_IFirstPix(m), n * overSample_In + I_A1_IFirstPix(m) + overSample_In), 0) << endl;
                #endif
              }
              #ifdef __DEBUG_SLITFUNC_X__
                //cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): 2. OArr set to " << OArr << endl;
                //cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): D_A2_Im(m,*) = " << D_A2_Im(m, blitz::Range::all()) << endl;
                std::string fName = DEBUGDIR + std::string("OArr") + debugFilesSuffix + std::string("_row");
                if (m < 100)
                  fName += std::string("0");
                if (m < 10)
                  fName += std::string("0");
                fName += to_string(m) + std::string(".dat");
                pfsDRPStella::utils::WriteArrayToFile(OArr, fName, std::string("ascii"));
                //if (m == 220)
                //  return false;
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
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Writing file " << sOArr << endl;
                pfsDRPStella::utils::WriteArrayToFile(D_A1_OArrBySF, sOArr, string("ascii"));
                
                sOArr = DEBUGDIR + std::string("D_A2_Im")+debugFilesSuffix+std::string("_row");
                if (m < 100)
                  sOARR += std::string("0");
                if (m < 10)
                  sOARR += std::string("0");
                sOARR += to_string(m) + std::string(".dat");
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Writing file " << sOArr << endl;
                pfsDRPStella::utils::WriteArrayToFile(D_A2_Im(m,blitz::Range::all()), sOArr, string("ascii"));
              #endif
              #ifdef __DEBUG_CHECK_INDICES__      
                if (P_I_A2_Mask->rows() != D_A2_Mask.rows()){
                  cout << "CFits::SlitFunc: ERROR: 2. P_I_A2_Mask->rows() != D_A2_Mask.rows()" << endl;
                  return false;
                }
              #endif
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): 3. OArr set to " << OArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): *P_I_A2_Mask = " << *P_I_A2_Mask << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): D_A2_Mask = " << D_A2_Mask << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              #endif
                
              for (int n = 1; n < I_NCols_Im; n++)
              {
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=1; n<NCols(=" << I_NCols_Im << "); n++): (D_A2_Im(m,n-1)=" << D_A2_Im(m, n-1) << ")" << endl;
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=1; n<NCols(=" << I_NCols_Im << "); n++): ((*P_I_A2_Mask)(m,n-1)=" << (*P_I_A2_Mask)(m, n-1) << ")" << endl;
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=1; n<NCols(=" << I_NCols_Im << "); n++): OmegaVecArr(overSample_In)=" << OmegaVecArr(overSample_In) << ")" << endl;
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=1; n<NCols(=" << I_NCols_Im << "); n++): (D_A2_Im(m,n)=" << D_A2_Im(m, n) << ")" << endl;
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=1; n<NCols(=" << I_NCols_Im << "); n++): ((*P_I_A2_Mask)(m,n)=" << (*P_I_A2_Mask)(m, n) << ")" << endl;
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=1; n<NCols(=" << I_NCols_Im << "); n++): (OmegaVecArr(0)=" << OmegaVecArr(0) << ")" << endl;
                #endif
                OArr((n * overSample_In) + I_A1_IFirstPix(m), 0) = (D_A2_Im(m, n-1) * OmegaVecArr(overSample_In) * D_A2_Mask(m, n-1)) + (D_A2_Im(m, n) * OmegaVecArr(0) * D_A2_Mask(m, n));
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): for(n(=" << n << ")=1; n<NCols(=" << I_NCols_Im << "); n++): OArr((n * overSample_In) + I_A1_IFirstPix(m)=" << (n * overSample_In) + I_A1_IFirstPix(m) << ", 0) set to " << OArr((n * overSample_In) + I_A1_IFirstPix(m), 0) << endl;
                #endif
              }/// end for (int n = 1; n < I_NCols_Im; n++)
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): 4. OArr set to " << OArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              #endif
              OArr(I_A1_IFirstPix(m), 0) = D_A2_Im(m, 0) * OmegaVecArr(0) * D_A2_Mask(m, 0);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): D_A2_Im(m, I_NCols_Im-1) = " << D_A2_Im(m, I_NCols_Im-1) << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OmegaVecArr(overSample_In) = " << OmegaVecArr(overSample_In) << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): (*P_I_A2_Mask)(m, I_NCols_Im-1) = " << (*P_I_A2_Mask)(m, I_NCols_Im-1) << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): D_A2_Mask(m, I_NCols_Im-1) = " << D_A2_Mask(m, I_NCols_Im-1) << endl;
              #endif
              OArr((I_NCols_Im * overSample_In) + I_A1_IFirstPix(m), 0) = D_A2_Im(m, I_NCols_Im - 1) * OmegaVecArr(overSample_In) * D_A2_Mask(m, I_NCols_Im - 1);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): OArr((D_A2_Im.cols(=" << I_NCols_Im << ") * overSample_In=" << overSample_In << ") + I_A1_IFirstPix(m)=" << I_A1_IFirstPix(m) << ", 0) set to " << OArr((I_NCols_Im * overSample_In) + I_A1_IFirstPix(m), 0) << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): 5. OArr set to " << OArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): spectrum_Out(m) = " << spectrum_Out(m) << endl;
              #endif
              BLVecArr(blitz::Range::all()) += ((spectrum_Out(m) * OArr(blitz::Range::all(), 0)));
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": end for(m(=" << m << ")=0; m<NRows(=" << I_NRows_Im << "); m++): BLVecArr set to " << BLVecArr << endl;
              #endif
              #ifdef __DEBUG_SLITFUNC_X__
                fName = DEBUGDIR + std::string("OArr_new") + debugFilesSuffix + "_row";
                if (m < 100)
                  fName += "0";
                if (m < 10)
                  fName += "0";
                fName += to_string(m) + ".dat";
                pfsDRPStella::utils::WriteArrayToFile(OArr, fName, std::string("ascii"));
                //              if (m == 20)
                //                return false;
  //              cout << "xCentersPixelFraction_In(" << m << ") = " << xCentersPixelFraction_In(m) << endl;
              #endif
            } /** end for (int m = 0; m < I_NRows_Im; m++) **/
            
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": end for(m=0; m<NRows; m++): OArr set to " << OArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": end for(m=0; m<NRows; m++): BLVecArr set to " << BLVecArr << endl;
            //  return false;
            #endif
              
            Lambda = Lamb_SF * blitz::sum(AKLArr(blitz::Range::all(), overSample_In)) / I_NPixSlitF;
            blitz::Array<double, 1> D_A1_Lamb_SF(I_NPixSlitF);
            if (D_WingSmoothFactor > 0.){
              if (I_Iter_SF == 1){
                blitz::Array<double, 1> D_A1_DIndGen = pfsDRPStella::math::DIndGenArr(I_NPixSlitF);
                D_A1_Lamb_SF = Lambda * (1. + D_WingSmoothFactor * blitz::pow2(2. * D_A1_DIndGen / (I_NPixSlitF - 1) - 1.));
              }
              else{
                #ifdef __DEBUG_CHECK_INDICES__      
                  if (static_cast<int>(SFVecArr.size()) != I_NPixSlitF){
                    cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(WING_SMOOTH_FACTOR): ERROR: SFVecArr.size(=" << SFVecArr.size() << ") != I_NPixSlitF=" << I_NPixSlitF << " => Returning FALSE" << endl;
                    return false;
                  }
                #endif
                SFVecArrTemp.resize(SFVecArr.size());
                SFVecArrTemp = SFVecArr;
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(WING_SMOOTH_FACTOR): 1. SFVecArrTemp set to " << SFVecArrTemp << endl;
                #endif
                for (unsigned int m=0; m<SFVecArrTemp.size(); m++){
                  if (SFVecArrTemp(m) < 0.00001)
                    SFVecArrTemp(m) = 0.00001;
                }
                #ifdef __DEBUG_SLITFUNC_N__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(WING_SMOOTH_FACTOR): 2. SFVecArrTemp set to " << SFVecArrTemp << endl;
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(WING_SMOOTH_FACTOR): Lambda = " << Lambda << endl;
                #endif
                D_A1_Lamb_SF = Lambda * (1. + D_WingSmoothFactor / (SFVecArrTemp));
              }
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(WING_SMOOTH_FACTOR): D_A1_Lamb_SF set to " << D_A1_Lamb_SF << endl;
              #endif
            }/// end if (Pos >= 0 && D_WingSmoothFactor > 0.){
            else{
              D_A1_Lamb_SF = pfsDRPStella::math::Replicate(Lambda, I_NPixSlitF);
              #ifdef __DEBUG_SLITFUNC__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": !KeyWord_Set(WING_SMOOTH_FACTOR): D_A1_Lamb_SF set to " << D_A1_Lamb_SF << endl;
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
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Start 1st order Tikhonov: AKLArr = " << AKLArr << endl;//.transpose(blitz::secondDim,blitz::firstDim) << endl;
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Start 1st order Tikhonov: D_A1_Lamb_SF = " << D_A1_Lamb_SF << endl;
            #endif
  //          return false;
              
            AKLArr(0, overSample_In) += D_A1_Lamb_SF(0); /** + Lambda to the upper-left element **/
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 1st order Tikhonov: AKLArr(0,overSample_In=" << overSample_In << ") set to " << AKLArr(0,overSample_In) << endl;
            #endif
              
            AKLArr(I_NPixSlitF-1,overSample_In) += D_A1_Lamb_SF(I_NPixSlitF - 1); /** and to the lower-right **/
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 1st order Tikhonov: AKLArr(I_NPixSlitF-1=" << I_NPixSlitF-1 << ",overSample_In=" << overSample_In << ") set to " << AKLArr(I_NPixSlitF-1,overSample_In) << endl;
            #endif
              
            AKLArr(blitz::Range(1,I_NPixSlitF-2), overSample_In) += 2. * D_A1_Lamb_SF(blitz::Range(1, I_NPixSlitF-2)); /** +2*Lambda to the rest of the main diagonal **/
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 1st order Tikhonov: AKLArr(blitz::Range(1,I_NPixSlitF-2=" << I_NPixSlitF-2 << "),overSample_In=" << overSample_In << ") set to " << AKLArr(blitz::Range(1,I_NPixSlitF-2),overSample_In) << endl;
            #endif
              
            AKLArr(blitz::Range(0, I_NPixSlitF - 2), overSample_In + 1) -= D_A1_Lamb_SF(blitz::Range(0, I_NPixSlitF - 2)); /** -Lambda to the upper sub-diagonal **/
            #ifdef __DEBUG_SLITFUNC__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 1st order Tikhonov: AKLArr(blitz::Range(0,I_NPixSlitF-2=" << I_NPixSlitF-2 << "),overSample_In+1=" << overSample_In+1 << ") set to " << AKLArr(blitz::Range(0,I_NCols_Im-2),overSample_In+1) << endl;
            #endif
              
            AKLArr(blitz::Range(1, I_NPixSlitF - 1), overSample_In - 1) -= D_A1_Lamb_SF(blitz::Range(1, I_NPixSlitF - 1)); /** -Lambda to the lower sub-diagonal **/
            #ifdef __DEBUG_SLITFUNC__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 1st order Tikhonov: AKLArr(blitz::Range(1,I_NPixSlitF-1=" << I_NPixSlitF-1 << "),overSample_In-1=" << overSample_In-1 << ") set to " << AKLArr(blitz::Range(1,I_NPixSlitF-1),overSample_In-1) << endl;
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Start. 1st order Tikhonov: AKLArr = " << AKLArr << endl;
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
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Start. 2nd order Tikhonov: AKLArr = " << AKLArr << endl;
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Start. 2nd order Tikhonov: D_A2_AKLT = " << D_A2_AKLT << endl;
            #endif
            PP_Void[0] = D_A2_AKLT.data();
            PP_Void[1] = BLVecArr.data();
            #ifdef __DEBUG_SLITFUNC__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Start. 2nd order Tikhonov: BLVecArr = " << BLVecArr << endl;
            #endif
            PP_Void[2] = &I_NPixSlitF;
            #ifdef __DEBUG_SLITFUNC__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Start. 2nd order Tikhonov: I_NPixSlitF = " << I_NPixSlitF << endl;
            #endif
            TempInt = (2 * overSample_In) + 1;
            PP_Void[3] = &TempInt;
            #ifdef __DEBUG_SLITFUNC__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 2nd order Tikhonov: TempInt = " << TempInt << endl;
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 2nd order Tikhonov: Starting BandSol(D_A2_AKLT=" << D_A2_AKLT << ", BLVecArr=" << BLVecArr << ", I_NPixSlitF=" << I_NPixSlitF << ", TempInt=" << TempInt << ")" << endl;
            #endif
  //          return false;
            pfsDRPStella::math::BandSol(4, PP_Void);
            free(PP_Void);
            AKLArr = D_A2_AKLT.transpose(blitz::secondDim, blitz::firstDim);
            #ifdef __DEBUG_SLITFUNC__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 2nd order Tikhonov: After BandSol: D_A2_AKLT=" << D_A2_AKLT << ", AKLArr=" << AKLArr << ", BLVecArr=" << BLVecArr << ", I_NPixSlitF=" << I_NPixSlitF << ", TempInt=" << TempInt << endl;
            #endif
            SFVecArr.resize(BLVecArr.size());
            if (abs(blitz::sum(BLVecArr)) < 0.000000001)
            {
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": WARNING: blitz::sum(BLVecArr) == 0 => Setting to 1." << endl;
              BLVecArr = 1.;
            }
            double D_SumBLVecArr = blitz::sum(BLVecArr);
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum(BLVecArr) = " << D_SumBLVecArr << endl;
            #endif
            if (D_SumBLVecArr == 0.){
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: blitz::sum(BLVecArr) == 0 => Returning FALSE" << endl;
              return false;
            }
            SFVecArr = BLVecArr;
            for (unsigned int mmm=0; mmm < SFVecArr.size(); mmm++){
              if (SFVecArr(mmm) < 0.)
                SFVecArr(mmm) = 0.;
            }
            SFVecArr = SFVecArr / D_SumBLVecArr * overSample_In;
              
            #ifdef __DEBUG_SLITFUNC_N__
              std::string sOArr = std::string(DEBUGDIR) + std::string("SFVecArr") + debugFilesSuffix + std::string(".dat");
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Writing File " << SFVecArr << endl;
              pfsDRPStella::utils::WriteArrayToFile(SFVecArr, sOArr, string("ascii"));
              
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 4.     2nd order Tikhonov: SFVecArr ( = BLVecArr / blitz::sum(BLVecArr)(=" << blitz::sum(BLVecArr) << ") * overSample_In(=" << overSample_In << ")) = " << SFVecArr << endl;
  //            return false;
            #endif
              
            #ifdef __DEBUG_SLITFUNC__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 2nd order Tikhonov: spectrum_Out = " << spectrum_Out << endl;
            #endif
            SPOldVecArr.resize(spectrum_Out.size());
            SPOldVecArr = spectrum_Out;
            #ifdef __DEBUG_SLITFUNC__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 2nd order Tikhonov: SPOldVecArr = " << SPOldVecArr << endl;
            #endif
              
            RVecArr.resize(spectrum_Out.size());
            RVecArr_Err.resize(spectrum_Out.size());
            RVecArr = spectrum_Out;
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 2nd order Tikhonov: RVecArr = " << RVecArr << endl;
            #endif
              
            OmegaVecArr.resize(overSample_In);
            OmegaVecArr = Weight;
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": 2nd order Tikhonov: OmegaVecArr = " << OmegaVecArr << endl;
              cout << "CFits::SlitFunc: Before TempDVecArrA: *P_I_A2_MaskIn = " << *P_I_A2_MaskIn << endl;
            #endif
              
            /** Evaluate the new Spectrum **/
            for (int m = 0; m < I_NRows_Im; m++)
            {
              OmegaVecArr(0) = D_A2_XX(m,I_A1_IFirstSpec(m));
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): OmegaVecArr(0) set to " << OmegaVecArr(0) << endl;
              #endif
                
              TempDVecArr.resize(I_A1_ILastSpec(m) - I_A1_IFirstSpec(m) + 1);
              TempDVecArr = SFVecArr(blitz::Range(I_A1_IFirstSpec(m), I_A1_ILastSpec(m)));
              #ifdef __DEBUG_SLITFUNC__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): 1. TempDVecArr set to " << TempDVecArr << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Starting Reform: I_NCols_Im = " << I_NCols_Im << ", overSample_In = " << overSample_In << endl;
              #endif
              blitz::Array<double, 2> *p_SSFTArr = pfsDRPStella::math::Reform(TempDVecArr, (int)I_NCols_Im, overSample_In);
              #ifdef __DEBUG_SLITFUNC__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": fiberTraceNumber = " << fiberTraceNumber << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Reform finished: p_SSFTArr set to " << *p_SSFTArr << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_NCols_Im = " << I_NCols_Im << endl;
              #endif
              SSFArr.resize(overSample_In, (int)I_NCols_Im);
              SSFArr = p_SSFTArr->transpose(blitz::secondDim, blitz::firstDim);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": SSFArr set to " << SSFArr << endl;
              #endif
              delete p_SSFTArr;
              TempDVecArr.resize(0);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): SSFArr set to " << SSFArr << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
              #endif
                
              tempint = SSFArr.cols();
                
              D_A2_TempAA.resize(SSFArr.cols(), SSFArr.rows());
              D_A2_TempAA = SSFArr.transpose(blitz::secondDim,blitz::firstDim);
                
              blitz::Array<double, 1> *p_TempDVecArrBB = pfsDRPStella::math::MatrixTimesVecArr(D_A2_TempAA,
                                                                                              OmegaVecArr);
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): 2. TempDVecArr set to " << *p_TempDVecArrBB << endl;
              #endif
              OArr.resize(tempint, 1);
              OArr(blitz::Range(0,tempint-1), 0) = (*p_TempDVecArrBB)(blitz::Range(0,tempint-1));
              delete p_TempDVecArrBB;
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): OArr(blitz::Range(0,tempint-1), 0) set to " << OArr(blitz::Range(0,tempint-1), 0) << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): D_A2_Im.cols(=" << I_NCols_Im << ") - D_A2_XX(" << m << ",I_A1_ILastSpec(m)=" << I_A1_ILastSpec(m) << ")(=" << D_A2_XX(m,I_A1_ILastSpec(m)) << ") = " << I_NCols_Im - D_A2_XX(m,I_A1_ILastSpec(m)) << endl;
              #endif
              XXX = I_NCols_Im - D_A2_XX(m,I_A1_ILastSpec(m));
              #ifdef __DEBUG_SLITFUNC_N__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): XXX set to " << XXX << endl;
              #endif
                
              TempDVecArr.resize(I_NCols_Im-1);
              TempDVecArr = SSFArr(0, blitz::Range(1, I_NCols_Im-1));
              TempDVecArr *= XXX;
              OArr(blitz::Range(0, I_NCols_Im - 2), 0) += TempDVecArr;
              OArr(I_NCols_Im - 1, 0) += SFVecArr(I_A1_ILastSpec(m) + 1) * XXX;
              #ifdef __DEBUG_SLITFUNC__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): OArr(I_NCols_Im - 1, 0) set to " << OArr(I_NCols_Im - 1, 0) << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): D_A2_Im(m, all()) = " << D_A2_Im(m, blitz::Range::all()) << endl;
              #endif
                
              D_A1_TempDVecArr.resize(I_NCols_Im);
              D_A1_TempDVecArr = D_A2_Im(m, blitz::Range::all());
              D_A1_TempDVecArr *= D_A2_Mask(m, blitz::Range::all());
              #ifdef __DEBUG_SLITFUNC__
                cout << "CFits::SlitFunc: D_A1_TempDVecArr: D_A2_Im(m=" << m << ", blitz::Range::all()) = " << D_A2_Im(m, blitz::Range::all()) << endl;
                cout << "CFits::SlitFunc: D_A1_TempDVecArr: P_I_A2_Mask(m=" << m << ", blitz::Range::all()) = " << (*P_I_A2_Mask)(m, blitz::Range::all()) << endl;
                cout << "CFits::SlitFunc: D_A1_TempDVecArr: D_A2_Mask(m=" << m << ", blitz::Range::all()) = " << D_A2_Mask(m, blitz::Range::all()) << endl;
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
                cout << "CFits::SlitFunc: D_A1_TempDVecArr = " << D_A1_TempDVecArr << endl;
                cout << "CFits::SlitFunc: D_A1_TempDVecArrAA = " << D_A1_TempDVecArrAA << endl;
              #endif
              #ifdef __DEBUG_CHECK_INDICES__      
                if (D_A1_TempDVecArr.size() != D_A1_TempDVecArrAA.size())
                {
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: D_A1_TempDVecArr.size(=" << D_A1_TempDVecArr.size() << ")=(" << D_A1_TempDVecArr << ") != D_A1_TempDVecArrAA.size(=" << D_A1_TempDVecArrAA.size() << ") = (" << D_A1_TempDVecArrAA << ")" << endl;
                  return false;
                }
              #endif
                
              RVecArr(m) = pfsDRPStella::math::VecArrAScalarB(D_A1_TempDVecArr, D_A1_TempDVecArrAA);
              RVecArr_Err(m) = pfsDRPStella::math::VecArrAScalarB(D_A1_TempDVecArr_Err, D_A1_TempDVecArrAA);
              #ifdef __DEBUG_SLITFUNC__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): RVecArr(m) set to " << RVecArr(m) << endl;
              #endif
              TempDVecArr.resize(OArr.rows());
              TempDVecArr = (blitz::pow2(OArr(blitz::Range::all(), 0)));
              TempDVecArr(blitz::Range::all()) *= D_A2_Mask(m, blitz::Range::all());
              spectrum_Out(m) = blitz::sum(TempDVecArr);
              #ifdef __DEBUG_SLITFUNC_SF__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): spectrum_Out(m=" << m << ") set to " << spectrum_Out(m) << endl;
              #endif
              if (fabs(spectrum_Out(m)) < 0.000001)
              {
                spectrum_Out(m) = blitz::sum(blitz::pow2(OArr));
                #ifdef __DEBUG_SLITFUNC_SF__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": for(m==" << m << "): was == 0: spectrum_Out(m=" << m << ") set to " << spectrum_Out << endl;
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
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m==" << m << "): I_Iter_SF(=" << I_Iter_SF << ") > 1: TempDVecArr set to " << TempDVecArr << endl;
                #endif
                  
                TempDVecArrB = OArr(blitz::Range::all(), 0);
                #ifdef __DEBUG_SLITFUNC__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": for(m==" << m << "): I_Iter_SF(=" << I_Iter_SF << ") > 1: TempDVecArrB set to " << TempDVecArrB << endl;
                #endif
                  
                pfsDRPStella::math::Double((*P_I_A2_Mask), D_A2_Mask);
                #ifdef __DEBUG_CHECK_INDICES__      
                  if (P_I_A2_Mask->rows() != D_A2_Mask.rows()){
                    cout << "CFits::SlitFunc: ERROR: 3. P_I_A2_Mask->rows() != D_A2_Mask.rows()" << endl;
                    return false;
                  }
                #endif
                if (fabs(mean(*P_I_A2_Mask) - mean(D_A2_Mask)) > 0.0000001){
                  cout << "CFits::SlitFunc: ERROR: 3. mean(P_I_A2_Mask)(=" << mean(*P_I_A2_Mask) << ") != mean(D_A2_Mask)(=" << mean(D_A2_Mask) << ")" << endl;
                  return false;
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
              pfsDRPStella::utils::WriteFits(&D_A2_SPTemp, S_SP);
            #endif
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "CFits::SlitFunc: After TempDVecArrA: *P_I_A2_Mask = " << *P_I_A2_Mask << endl;
              cout << "blitz::sum(*P_I_A2_MaskIn) = " << blitz::sum(*P_I_A2_MaskIn) << endl;
              cout << "blitz::sum(*P_I_A2_Mask) = " << blitz::sum(*P_I_A2_Mask) << endl;
            #endif

            if (abs(Lamb_SP) > 0.0000001)
            {
              Lambda = Lamb_SP * blitz::sum(spectrum_Out) / I_NRows_Im;
              #ifdef __DEBUG_SLITFUNC__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": KeyWord_Set(LAMBDA_SP): Lambda set to " << Lambda << endl;
              #endif
              a.resize(I_NRows_Im);
              a = 0. - Lambda;
              a(0) = 0.;
              #ifdef __DEBUG_SLITFUNC__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": KeyWord_Set(LAMBDA_SP): a set to " << a << endl;
              #endif
              b.resize(I_NRows_Im);
              b = (2. * Lambda) + 1.;
              b(0) = Lambda + 1.;
              b(I_NRows_Im - 1) = Lambda + 1.;
              #ifdef __DEBUG_SLITFUNC__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": KeyWord_Set(LAMBDA_SP): b set to " << b << endl;
              #endif
              c.resize(I_NRows_Im);
              c = 0. - Lambda;
              c(I_NRows_Im - 1) = 0.;
              #ifdef __DEBUG_SLITFUNC__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": KeyWord_Set(LAMBDA_SP): c set to " << c << endl;
              #endif
                
              #ifdef __DEBUG_CHECK_INDICES__      
                if (I_NRows_Im != static_cast<int>(spectrum_Out.size()))
                {
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": ERROR: D_A2_Im.rows(=" << I_NRows_Im << ") != spectrum_Out.size(=" << spectrum_Out.size() << ")" << endl;
                  return false;
                }
              #endif
              #ifdef __DEBUG_SLITFUNC__
                cout << "CFits::SlitFunc: Before TriDag: RVecArr = " << RVecArr << endl;
                cout << "CFits::SlitFunc: Before TriDag: spectrum_Out = " << spectrum_Out << endl;
                cout << "CFits::SlitFunc: Before TriDag: a = " << a << endl;
                cout << "CFits::SlitFunc: Before TriDag: b = " << b << endl;
                cout << "CFits::SlitFunc: Before TriDag: c = " << c << endl;
              #endif
              TempDVecArr.resize(I_NRows_Im);
              TempDVecArr = RVecArr / spectrum_Out;
              #ifdef __DEBUG_SLITFUNC__
                cout << "CFits::SlitFunc: Before TriDag: TempDVecArr = " << TempDVecArr << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": KeyWord_Set(LAMBDA_SP): TempDVecArr set to " << TempDVecArr << endl;
              #endif
              pfsDRPStella::math::TriDag(a, b, c, TempDVecArr, spectrum_Out);
              #ifdef __DEBUG_SLITFUNC_SF__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": KeyWord_Set(LAMBDA_SP): after TriDag: spectrum_Out set to " << spectrum_Out << endl;
              #endif
                
              #ifdef __DEBUG_SLITFUNC__
                S_SP = DEBUGDIR + std::string("spectrum_Out1TriDag_IterSF") + to_string(I_Iter_SF) + debugFilesSuffix + std::string(".fits");
                D_A2_SPTemp.resize(spectrum_Out.size(), 1);
                D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
                pfsDRPStella::utils::WriteFits(&D_A2_SPTemp, S_SP);
              #endif
            }
            else{
              spectrum_Out = RVecArr / spectrum_Out;
              #ifdef __DEBUG_SLITFUNC_SF__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": Not KeyWord_Set(LAMBDA_SP): spectrum_Out set to " << spectrum_Out << endl;
              #endif
                
              #ifdef __DEBUG_SLITFUNC__
                S_SP = "spectrum_Out1NoLambSP_IterSF" + to_string(I_Iter_SF) + debugFilesSuffix + ".fits";
                D_A2_SPTemp.resize(spectrum_Out.size(), 1);
                D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
                pfsDRPStella::utils::WriteFits(&D_A2_SPTemp, S_SP);
              #endif
            } /// end else if ((Pos = pfsDRPStella::utils::KeyWord_Set(const_cast<const CString**>(Args), NArgs, *P_TempString)) < 0)
              
            #ifdef __DEBUG_SLITFUNC_SF__
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": end if (max(fabs(spectrum_Out - SPOldVecArr) / max(spectrum_Out)=" << max(spectrum_Out) << ")=" << max(fabs(spectrum_Out - SPOldVecArr) / max(spectrum_Out)) << " > 0.00001)" << endl;
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
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_SF = " << I_Iter_SF << ": !if (max(fabs(spectrum_Out - SPOldVecArr) / max(spectrum_Out)=" << max(spectrum_Out) << ")=" << max(fabs(spectrum_Out - SPOldVecArr) / max(spectrum_Out)) << " > 0.00001) => breaking while loop" << endl;
              #endif
              break;
            }/// end if (I_Iter_SF != 1)
          }/// end else if (I_Iter_SF != 1 && max(abs(spectrum_Out-SPOldVecArr)/max(spectrum_Out)) <= 0.00001)
          blitz::Array<double, 2> D_A2_ImTimesMask_SF(D_A2_Im.rows(), D_A2_Im.cols());
          D_A2_ImTimesMask_SF = D_A2_Im * (*P_I_A2_Mask);
          #ifdef __DEBUG_SLITFUNC_FILES__
            string S_ImTimesMaskSF = DEBUGDIR + std::string("ImTimesMask_IterSF") + to_string(I_Iter_SF) + debugFilesSuffix + std::string(".fits");
            pfsDRPStella::utils::WriteFits(&D_A2_ImTimesMask_SF, S_ImTimesMaskSF);
          #endif
        } /// end while(I_Iter_SF < I_MaxIterSF)
        if ((Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "IM_OUT")) >= 0)
        {
          if (P_D_A2_Im_Out != NULL)
            delete P_D_A2_Im_Out;
          P_D_A2_Im_Out = (blitz::Array<double, 2>*)ArgV_In[Pos];
          P_D_A2_Im_Out->resize(D_A2_Im.rows(), D_A2_Im.cols());
          (*P_D_A2_Im_Out) = 0.;
        }
        #ifdef __DEBUG_SLITFUNC__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): I_NRows_Im set to " << I_NRows_Im << endl;
        #endif
        if ((Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "USE_ROW")) >= 0)// && TempIntB != 0)
        {
          D_A1_Ind.resize(I_NRows_Im);
          D_A1_Ind = i;
          #ifdef __DEBUG_SLITFUNC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): KeyWord_Set(USE_ROW): MARK: INTERPOL: spectrum_Out = " << spectrum_Out << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): KeyWord_Set(USE_ROW): MARK: INTERPOL: D_A1_Ind = " << D_A1_Ind << endl;
          #endif
          blitz::Array<double, 1> tempDblVecArrA = pfsDRPStella::math::Double(UseRowVecArr);
          #ifdef __DEBUG_SLITFUNC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": tempDblVecArrA = " << tempDblVecArrA << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A1_Ind = " << D_A1_Ind << endl;
          #endif
          blitz::Array<double, 1> tempDblVecArrB(D_A1_Ind.size());
          if (!pfsDRPStella::math::InterPol(spectrum_Out, tempDblVecArrA, D_A1_Ind, tempDblVecArrB)){
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: InterPol returned FALSE" << endl;
            return false;
          }
          spectrum_Out = tempDblVecArrB;
          #ifdef __DEBUG_SLITFUNC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): KeyWord_Set(USE_ROW): MARK: INTERPOL: spectrum_Out set to " << spectrum_Out << endl;
          #endif
        }///end if KeyWord_Set(USE_ROW)
        OmegaVecArr.resize(overSample_In);
        OmegaVecArr = Weight;
        #ifdef __DEBUG_SLITFUNC__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): OmegaVecArr set to " << OmegaVecArr << endl;
        #endif
          
        D_A1_SFO.resize(SFVecArr.size());
        D_A1_SFO = SFVecArr;
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "SlitFunc: SFVecArr = " << SFVecArr << endl;
  //        return false;
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
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): OmegaVecArr(0) set to " << OmegaVecArr(0) << endl;
          #endif
            
          SSFArr.resize(overSample_In, I_NCols_Im);
          SSFArr = 0.;
          #ifdef __DEBUG_SLITFUNC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): SFVecArr = " << SFVecArr << endl;
          #endif
          TempDVecArr.resize(I_A1_ILastSpec(m) - I_A1_IFirstSpec(m) + 1);
          TempDVecArr = SFVecArr(blitz::Range(I_A1_IFirstSpec(m), I_A1_ILastSpec(m)));
          #ifdef __DEBUG_SLITFUNC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): TempDVecArr set to " << TempDVecArr << endl;
          #endif
          #ifdef __DEBUG_CHECK_INDICES__      
            if (static_cast<int>(TempDVecArr.size()) != I_NCols_Im * overSample_In)
            {
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: size of TempDVecArr(=" << TempDVecArr.size() << ") != D_A2_Im.cols(=" << I_NCols_Im << ") * overSample_In(=" << overSample_In << ")" << endl;
              return false;
            }
          #endif
          blitz::Array<double, 2> *p_D_A2_SSFT = pfsDRPStella::math::Reform(TempDVecArr, I_NCols_Im, overSample_In);
          TempDVecArr.resize(0);
          SSFArr = p_D_A2_SSFT->transpose(blitz::secondDim, blitz::firstDim);
          delete p_D_A2_SSFT;
          #ifdef __DEBUG_SLITFUNC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): SSFArr set to " << SSFArr << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): OmegaVecArr set to " << OmegaVecArr << endl;
          #endif
            
          OArr.resize(SSFArr.cols(), 1);
          D_A2_OT.resize(SSFArr.cols(), SSFArr.rows());
          D_A2_OT = SSFArr.transpose(blitz::secondDim, blitz::firstDim);
          #ifdef __DEBUG_CHECK_INDICES__      
            if (OArr.rows() != D_A2_OT.rows())
            {
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: OArr.rows(=" << OArr.rows() << ") != D_A2_OT.rows(=" << D_A2_OT.rows() << ")" << endl;
              return false;
            }
          #endif
          blitz::Array<double, 1> *p_TempDVecArrAA = pfsDRPStella::math::MatrixTimesVecArr(D_A2_OT, OmegaVecArr);
          OArr(blitz::Range::all(), 0) = (*p_TempDVecArrAA);
          delete p_TempDVecArrAA;
          #ifdef __DEBUG_SLITFUNC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): OArr set to " << OArr << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
          #endif
            
          XXX = I_NCols_Im - D_A2_XX(m,I_A1_ILastSpec(m));
          #ifdef __DEBUG_SLITFUNC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): XXX set to " << XXX << endl;
          #endif
            
          #ifdef __DEBUG_CHECK_INDICES__      
            if (OArr.rows() < I_NCols_Im)
            {
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: OArr.rows(=" << OArr.rows() << ") < D_A2_Im.cols(=" << I_NCols_Im << ")" << endl;
              return false;
            }
            if (SSFArr.cols() < I_NCols_Im)
            {
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: SSFArr.cols(=" << SSFArr.cols() << ") < D_A2_Im.cols(=" << I_NCols_Im << ")" << endl;
              return false;
            }
            if (static_cast<int>(SFVecArr.size()) < I_A1_ILastSpec(m) + 2)
            {
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: SFVecArr.size(=" << SFVecArr.size() << ") < I_A1_ILastSpec(m)(=" << I_A1_ILastSpec(m) << ")+2" << endl;
              return false;
            }
          #endif
          OArr(blitz::Range(0, I_NCols_Im - 2), 0) += (SSFArr(0, blitz::Range(1, I_NCols_Im - 1)) * XXX);
          OArr(I_NCols_Im - 1, 0) += (SFVecArr(I_A1_ILastSpec(m) + 1) * XXX);
          #ifdef __DEBUG_SLITFUNC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): OArr set to " << OArr << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
          #endif
            
          #ifdef __DEBUG_CHECK_INDICES__      
            if (OArr.rows() != I_NCols_Im)
            {
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: OArr.row(=" << OArr.rows() << ") < D_A2_Im.cols(=" << I_NCols_Im << ")" << endl;
              return false;
            }
          #endif
          IVecArr.resize(I_NCols_Im);
          #ifdef __PISKUNOV_ORIG__
            blitz::Array<double, 1> D_A1_TempWhere(D_A2_Im.cols());
            blitz::Array<int, 1> I_A1_IndDev(D_A2_Im.cols());
            I_A1_IndDev = blitz::where(fabs(D_A2_Im(m, blitz::Range::all()) - spectrum_Out(m) * OArr(blitz::Range::all(), 0)) < 3. * D_Dev, 1, 0);
            blitz::Array<int, 1> *P_I_A1_IndDev = pfsDRPStella::math::GetIndex(I_A1_IndDev, I_NInd);
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
            cout << "CFits::SlitFunc: (*P_D_A1_SPErrOut)(m) set to " << (*P_D_A1_SPErrOut)(m) << endl;
            delete(P_I_A1_IndDev);//      }
          #endif
            
          I_A1_Mask.resize(P_I_A2_Mask->cols());
          I_A1_Mask = (*P_I_A2_Mask)(m, blitz::Range::all());
          pfsDRPStella::math::Double(I_A1_Mask, D_A1_Mask);
          D_A2_Mask(m, blitz::Range::all()) = D_A1_Mask;
          #ifdef __DEBUG_CHECK_INDICES__      
            if (P_I_A2_Mask->rows() != D_A2_Mask.rows()){
              cout << "CFits::SlitFunc: ERROR: 5. P_I_A2_Mask->rows() != D_A2_Mask.rows()" << endl;
              return false;
            }
          #endif
          NInd = blitz::sum((*P_I_A2_Mask)(m, blitz::Range::all()));
          #ifdef __DEBUG_SLITFUNC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum(*P_I_A2_MaskIn) = " << blitz::sum(*P_I_A2_MaskIn) << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum(*P_I_A2_Mask) = " << blitz::sum(*P_I_A2_Mask) << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): P_I_A2_Mask->size() = " << P_I_A2_Mask->size() << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): IVecArr.size() = " << IVecArr.size() << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): NInd set to " << NInd << endl;
          #endif
          #ifdef __DEBUG_CHECK_INDICES__      
            if (static_cast<int>(IVecArr.size()) < NInd)
            {
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: IVecArr.size(=" << IVecArr.size() << ") < NInd(=" << NInd << ")" << endl;
              return false;
            }
          #endif
          
          
          if ((Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "BAD")) >= 0)
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
                    cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: TempInt(=" << TempInt << ") >= IFirstVecArr.size(=" << IFirstVecArr.size() << ")" << endl;
                    return false;
                  }
                #endif
                IFirstVecArr(TempInt) = n;
                #ifdef __DEBUG_SLITFUNC__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): for(n(==" << n << ")=0; n< P_I_A2_Mask->cols(=" << P_I_A2_Mask->cols() << "; n++): IFirstVecArr(TempInt=" << TempInt << ") set to " << IFirstVecArr(TempInt) << endl;
                #endif
                TempInt++;
              }
              else
              {
                #ifdef __DEBUG_CHECK_INDICES__      
                  if (TempIntA >= static_cast<int>(TempIVecArr.size()))
                  {
                    cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: TempIntA(=" << TempIntA << ") >= TempIVecArr.size(=" << TempIVecArr.size() << ")" << endl;
                    return false;
                  }
                #endif
                TempIVecArr(TempIntA) = n;
                #ifdef __DEBUG_SLITFUNC__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): for(n(==" << n << ")=0; n< P_I_A2_Mask->cols(=" << P_I_A2_Mask->cols() << "; n++): TempIVecArr(TempIntA=" << TempIntA << ") set to " << TempIVecArr(TempIntA) << endl;
                #endif
                
                TempIntA++;
              }
            
              (*P_I_A1_JBadVecArr)(0) = 0;
              if (NInd < I_NCols_Im)  /// Bad pixels in column m
              {
                TempLong = (*P_I_A1_JBadVecArr)(0);
                #ifdef __DEBUG_SLITFUNC__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): NInd(=" << NInd << ") < D_A2_Im.cols(=" << I_NCols_Im << "): TempLong set to " << TempLong << endl;
                #endif
                P_I_A1_JBadVecArr->resize(1 + TempIVecArr.size());
                (*P_I_A1_JBadVecArr)(0) = TempLong;
                #ifdef __DEBUG_CHECK_INDICES__      
                  if (TempIVecArr.size() != P_I_A1_JBadVecArr->size()-1)
                  {
                    cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: TempIVecArr.size(=" << TempIVecArr.size() << ") != P_I_A1_JBadVecArr->size(=" << P_I_A1_JBadVecArr->size() << ")-1" << endl;
                    return false;
                  }
                #endif
                (*P_I_A1_JBadVecArr)(blitz::Range(1, P_I_A1_JBadVecArr->size() - 1)) = TempIVecArr;
                (*P_I_A1_JBadVecArr)(blitz::Range(1, P_I_A1_JBadVecArr->size() - 1)) += (long)I_NCols_Im * m;
                #ifdef __DEBUG_SLITFUNC__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): NInd(=" << NInd << ") < D_A2_Im.cols(=" << I_NCols_Im << "): P_I_A1_JBadVecArr set to " << (*P_I_A1_JBadVecArr) << ")" << endl;
                #endif
              }// end if (NInd < I_NCols_Im)
            }// end for (int n = 0; n < P_I_A2_Mask->cols(); n++)
          }// end if ((Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "BAD")) >= 0)
          
          #ifdef __DEBUG_SLITFUNC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): 2. spectrum_Out = " << spectrum_Out << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): spectrum_Out(m) = " << spectrum_Out(m) << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): OArr = " << OArr << endl;//.transpose(blitz::secondDim, blitz::firstDim) << endl;
          #endif
          #ifdef __DEBUG_CHECK_INDICES__      
            if (P_D_A2_Prof_Out->cols() != OArr.rows())
            {
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: P_D_A2_Prof_Out->cols(=" << P_D_A2_Prof_Out->cols() << ") != OArr.rows(=" << OArr.rows() << ")" << endl;
              return false;
            }
          #endif
          #ifdef __DEBUG_SLITFUNC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": m=" << m << ": P_D_A2_Prof_Out->size() = " << P_D_A2_Prof_Out->size() << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": m=" << m << ": P_D_A2_Prof_Out->rows() = " << P_D_A2_Prof_Out->rows() << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": m=" << m << ": P_D_A2_Prof_Out->cols() = " << P_D_A2_Prof_Out->cols() << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": m=" << m << ": OArr.size() = " << OArr.size() << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": m=" << m << ": OArr.rows() = " << OArr.rows() << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": m=" << m << ": OArr.cols() = " << OArr.cols() << endl;
          #endif
            
          double D_Error = 0.;
          double D_MinError = 10000000000000.;
          double D_Offset = -1./static_cast<double>(_fiberTraceExtractionControl->overSample);
          if (I_XCorProf == 0)
            D_Offset = 0.;
          double D_MinOffset = 0.;
          D_A1_XX = D_A2_XX(m, blitz::Range::all());
  //        #ifdef __DEBUG_SLITFUNC_X__
  //          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_XCorProf = " << I_XCorProf << ": row = " << m << ": D_A1_XX = " << D_A1_XX << endl;
  //        #endif
          D_A1_SFO = D_A1_SFO * _fiberTraceExtractionControl->overSample / sum(D_A1_SFO);
  //        if (!pfsDRPStella::math::IntegralNormalise(D_A1_XX, D_A1_SFO)){
  //          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: IntegralNormalise returned FALSE" << endl;
  //          return false;
  //        }
          #ifdef __DEBUG_SLITFUNC_X__
            std::string fname_sf = DEBUGDIR + std::string("SlitFuncOut") + debugFilesSuffix + std::string(".dat");
            pfsDRPStella::utils::WriteArrayToFile(D_A1_SFO, fname_sf, std::string("ascii"));
  //          cout << "D_A1_SFO written to <" << fname_sf << ">" << endl;
          //        return false;
          #endif
  //          return false;
          #ifdef __DEBUG_SLITFUNC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_XCorProf = " << I_XCorProf << endl;
          #endif
          double dSum = 0.;
          for (int i_cross=0; i_cross < I_XCorProf; i_cross++){
            D_A1_XX = D_A2_XX(m, blitz::Range::all()) + D_Offset;
            if (!pfsDRPStella::math::InterPol(D_A1_SFO, D_A1_XX, D_A1_XProf, D_A1_YProf)){
              cout << "FiberTrace::SlitFunc: ERROR: InterPol(D_A1_SFO=" << D_A1_SFO << ", D_A1_XX=" << D_A1_XX << ", D_A1_XProf=" << D_A1_XProf << ", D_A1_YProf) returned FALSE => Returning FALSE" << endl;
              return false;
            }
            profile_Out(m, blitz::Range::all()) = where(D_A1_YProf < 0., 0., D_A1_YProf);
            dSum = blitz::sum(profile_Out(m, blitz::Range::all()));
            if (fabs(dSum) > 0.00000000000000001)
              profile_Out(m, blitz::Range::all()) = profile_Out(m, blitz::Range::all()) / dSum;
            
            //cout << "FiberTrace::SlitFunc: 2. profile_Out(m=" << m << ", *) = " << profile_Out(m, blitz::Range::all()) << endl;
  //          return false;
            D_Error = sqrt(blitz::sum(blitz::pow2(D_A2_Im(m, blitz::Range::all()) - (profile_Out(m, blitz::Range::all()) * spectrum_Out(m)))));
            if (D_Error < D_MinError){
              D_MinError = D_Error;
              D_MinOffset = D_Offset;
            }
            #ifdef __DEBUG_SLITFUNC_N__
              cout << "SlitFunc: fiber " << fiberTraceNumber << ": bin = " << I_Bin << ": row " << m << ": i_cross=" << i_cross << ": offset=" << D_Offset << ": minError=" << D_MinError << ": error=" << D_Error << ": D_MinOffset=" << D_MinOffset << endl;
            #endif
            D_Offset += .2 / I_XCorProf;
          }
          (*P_D_A1_XCorProfOut)(m) = D_MinOffset;
          if (I_XCorProf > 0)
            cout << "SlitFunc: fiber " << fiberTraceNumber << ": bin " << I_Bin << ": row=" << m << ": D_MinOffset=" << D_MinOffset << endl;
          
  //        #ifdef __DEBUG_SLITFUNC_X__
  //          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_XCorProf = " << I_XCorProf << ": m=" << m << ": D_A2_Im(m=" << m << ", *) = " << D_A2_Im(m, blitz::Range::all()) << endl;
  //          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_XCorProf = " << I_XCorProf << ": m=" << m << ": D_MinOffset = " << D_MinOffset << endl;
  //          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_XCorProf = " << I_XCorProf << ": m=" << m << ": XVecArr = " << XVecArr << endl;
  //          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_XCorProf = " << I_XCorProf << ": m=" << m << ": xCentersPixelFraction_In(row=" << m << ") = " << xCentersPixelFraction_In(m) << endl;
  //          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_XCorProf = " << I_XCorProf << ": m=" << m << ": D_A1_SFO = " << D_A1_SFO << endl;
  //          if (m == 20)
  //            return false;
  //        #endif
        } /// end for (int m = 0; m < I_NRows_Im; m++)
          
        blitz::Array<double, 1> D_A1_Fit(P_D_A1_XCorProfOut->size());
        D_A1_Fit = 0.;
        if (I_XCorProf > 0){
          #ifdef __DEBUG_SLITFUNC_N__
            string sDebugFileName = DEBUGDIR + std::string("D_A1_XCorProfOut") + debugFilesSuffix + std::string(".dat");
            pfsDRPStella::utils::WriteArrayToFile((*P_D_A1_XCorProfOut), sDebugFileName, string("ascii"));
          #endif
          int I_NDeg = 3;
          blitz::Array<double, 1> *P_D_A1_PolyCoeffs = new blitz::Array<double, 1>(6);
          blitz::Array<string, 1> S_A1_Args_PolyFit(1);
          S_A1_Args_PolyFit(0) = "YFIT";
          void **PP_Args_PolyFit = (void**)malloc(sizeof(void*) * 1);
          PP_Args_PolyFit[0] = &D_A1_Fit;
          blitz::Array<double, 1> D_A1_XPF = pfsDRPStella::math::DIndGenArr(P_D_A1_XCorProfOut->size());
          if (!pfsDRPStella::math::PolyFit(D_A1_XPF,
                                          *P_D_A1_XCorProfOut,
                                          I_NDeg,
                                          S_A1_Args_PolyFit,
                                          PP_Args_PolyFit,
                                          P_D_A1_PolyCoeffs)){
            cout << "CFits::SlitFunc: ERROR: PolyFit(XCorProf) returned FALSE => Returning FALSE" << endl;
            delete(P_D_A1_PolyCoeffs);
            return false;
          }
          #ifdef __DEBUG_SLITFUNC_N__
            sDebugFileName = DEBUGDIR + std::string("D_A1_XCorProf_Fit") + debugFilesSuffix + std::string(".dat");
            pfsDRPStella::utils::WriteArrayToFile(D_A1_Fit, sDebugFileName, string("ascii"));
            cout << "CFits::SlitFunc: after PolyFit: P_D_A1_XCorProfOut = " << *P_D_A1_XCorProfOut << endl;
            cout << "CFits::SlitFunc: after PolyFit: D_A1_Fit = " << D_A1_Fit << endl;
            blitz::Array<double, 1> D_A1_Diff(P_D_A1_XCorProfOut->size());
            D_A1_Diff = (*P_D_A1_XCorProfOut) - D_A1_Fit;
            sDebugFileName = DEBUGDIR + std::string("D_A1_XCorProf_Diff") + debugFilesSuffix + std::string(".dat");
            pfsDRPStella::utils::WriteArrayToFile(D_A1_Diff, sDebugFileName, string("ascii"));
            cout << "CFits::SlitFunc: after PolyFit: D_A1_Diff = " << D_A1_Diff << endl;
            cout << "CFits::SlitFunc: after PolyFit: max(D_A1_Diff) = " << max(D_A1_Diff) << endl;
          #endif
          delete(P_D_A1_PolyCoeffs);
          free(PP_Args_PolyFit);
          (*P_D_A1_XCorProfOut) = D_A1_Fit;
        }
  //      #ifdef __DEBUG_SLITFUNC_X__
  //        cout << "SlitFunc: D_A2_XX = " << D_A2_XX << endl;
  //        cout << "SlitFunc: D_A1_Fit = " << D_A1_Fit << endl;
  //      #endif
        for (int m=0; m < I_NRows_Im; m++){
          //D_A1_XX(D_A2_XX.cols());
          D_A1_XX = D_A2_XX(m, blitz::Range::all()) + D_A1_Fit(m);
          #ifdef __DEBUG_SLITFUNC_X__
            //cout << "SlitFunc: D_A2_XX(" << m << ", *) = " << D_A2_XX(m, blitz::Range::all()) << endl;
            //cout << "SlitFunc: D_A1_Fit(" << m << ") = " << D_A1_Fit(m) << endl;
            //cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A1_XX = " << D_A1_XX << endl;
            //cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A1_SFO = " << D_A1_SFO << endl;
  //          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A1_Range = " << D_A1_Range << endl;
            
            std::string fname_x = DEBUGDIR + std::string("x")+debugFilesSuffix + std::string("_row");
            if (m < 100)
              fname_x += "0";
            if (m < 10)
              fname_x += "0";
            fname_x += to_string(m);// + ".dat";
            pfsDRPStella::utils::WriteArrayToFile(D_A2_XX(m, blitz::Range::all()), fname_x+"_orig.dat", std::string("ascii"));
            pfsDRPStella::utils::WriteArrayToFile(D_A1_XX, fname_x+".dat", std::string("ascii"));
          #endif
          if (!pfsDRPStella::math::InterPol(D_A1_SFO, D_A1_XX, D_A1_XProf, D_A1_YProf)){
            cout << "FiberTrace::SlitFunc: ERROR: InterPol(D_A1_SFO=" << D_A1_SFO << ", D_A1_XX=" << D_A1_XX << ", D_A1_XProf=" << D_A1_XProf << ", D_A1_YProf) returned FALSE => Returning FALSE" << endl;
            return false;
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
          if ((Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "IM_OUT")) >= 0)
          {
            (*P_D_A2_Im_Out)(m, blitz::Range::all()) = profile_Out(m,blitz::Range::all()) * spectrum_Out(m);
          }
          #ifdef __DEBUG_SLITFUNC_X__
  //          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(PROF_OUT): for(m(==" << m << ")=0; m< I_NRows_Im(=" << I_NRows_Im << "; m++): P_D_A2_Prof_Out(m=" << m << ", *) set to " << (*P_D_A2_Prof_Out)(m, blitz::Range::all()) << endl;
            
            std::string fname_prof = DEBUGDIR + std::string("ProfileOut")+ debugFilesSuffix + "_row";
            if (m < 100)
              fname_prof += "0";
            if (m < 10)
              fname_prof += "0";
            fname_prof += to_string(m) + std::string(".dat");
            pfsDRPStella::utils::WriteArrayToFile(profile_Out(m,blitz::Range::all()), fname_prof, std::string("ascii"));
            cout << "SlitFunc: fname_prof=<" << fname_prof << "> written" << endl;
  //          return false;
          #endif
        }/// end for (int m=0; m < I_NRows_Im; m++){
        #ifdef __DEBUG_SLITFUNC_X__
          std::string fname_rec = DEBUGDIR + std::string("ImRecOut") + debugFilesSuffix + std::string(".dat");
          pfsDRPStella::utils::WriteArrayToFile(*P_D_A2_Im_Out, fname_rec, std::string("ascii"));
          //cout << "SlitFunc: fname_rec=<" << fname_rec << "> written" << endl;
        //          return false;
        #endif
        //      return false;
          
        #ifdef __DEBUG_SLITFUNC_N__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": profile_Out set to " << profile_Out << endl;
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
            cout << endl << "CFits::SlitFunc: before Fit: D_A2_MySF = " << D_A2_MySF << endl << endl;
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
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": xxx = " << xxx << ", yyy = " << yyy << ": I_RangeMinRow = " << I_RangeMinRow << ", I_RangeMaxRow = " << I_RangeMaxRow << endl;
              #endif
                
              I_RangeMinCol = yyy - I_RangeWidth;
              if (I_RangeMinCol < 0)
                I_RangeMinCol = 0;
              I_RangeMaxCol = yyy + I_RangeWidth;
              if (I_RangeMaxCol >= P_I_A2_Mask->cols())
                I_RangeMaxCol = P_I_A2_Mask->cols() - 1;
              #ifdef __DEBUG_TELLURIC__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": xxx = " << xxx << ", yyy = " << yyy << " I_RangeMinCol = " << I_RangeMinCol << ", I_RangeMaxCol = " << I_RangeMaxCol << endl;
              #endif
                
              i_nrows = (I_RangeMaxRow - I_RangeMinRow + 1);
              i_ncols = (I_RangeMaxCol - I_RangeMinCol + 1);
              #ifdef __DEBUG_TELLURIC__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": i_nrows = " << i_nrows << ", i_ncols = " << i_ncols << endl;
              #endif
                
              I_A3_IndicesRange.resize(i_nrows, i_ncols, 2);
              for (int zzz = 0; zzz < i_nrows; zzz++){
                for (int qqq = 0; qqq < i_ncols; qqq++){
                  I_A3_IndicesRange(zzz,qqq,0) = I_RangeMinRow + zzz;
                  I_A3_IndicesRange(zzz,qqq,1) = I_RangeMinCol + qqq;
                }
              }
              #ifdef __DEBUG_TELLURIC__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_A3_IndicesRange(Subarr) set to " << I_A3_IndicesRange << endl;
              #endif
              I_A2_Temp.resize(P_I_A2_Mask->rows(), P_I_A2_Mask->cols());
              I_A2_Temp = (*P_I_A2_Mask);
              I_A2_TempArr.resize(I_A3_IndicesRange.rows(), I_A3_IndicesRange.cols());
              I_A2_TempArr = pfsDRPStella::math::GetSubArrCopy(I_A2_Temp, I_A3_IndicesRange);
              #ifdef __DEBUG_TELLURIC__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_A2_TempArr(Subarr) set to " << I_A2_TempArr << endl;
              #endif
                
              if (((*P_I_A2_Mask)(xxx,yyy) == 1) ||
                (blitz::sum(I_A2_TempArr) < I_A2_TempArr.size() / 1.5) ||
                (blitz::sum((*P_I_A2_Mask)(xxx,blitz::Range::all())) == 0))
              {
                D_A2_ImTimesMask(xxx,yyy) = D_A2_ImBak(xxx,yyy);
                D_A2_SFTimesMask(xxx,yyy) = (*P_D_A2_Prof_Out)(xxx,yyy);
                #ifdef __DEBUG_TELLURIC__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": (*P_I_A2_Mask)(xxx,yyy) = " << (*P_I_A2_Mask)(xxx,yyy) << " == 1 ||" << endl;
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum(I_A2_TempArr) = " << blitz::sum(I_A2_TempArr) << " < I_A2_TempArr.size() / 1.5 = " << I_A2_TempArr.size() / 1.5 << " ||" << endl;
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum((*P_I_A2_Mask)(xxx,blitz::Range::all())) = " << blitz::sum((*P_I_A2_Mask)(xxx,blitz::Range::all())) << " == 0" << endl;
                #endif
              }
              else{
                D_A2_ImTimesMask(xxx,yyy) = D_A1_Sky(xxx);
                D_A2_SFTimesMask(xxx,yyy) = 0.;
                #ifdef __DEBUG_TELLURIC__
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": (*P_I_A2_Mask)(xxx,yyy) = " << (*P_I_A2_Mask)(xxx,yyy) << " != 1 &&" << endl;
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum(I_A2_TempArr) = " << blitz::sum(I_A2_TempArr) << " >= I_A2_TempArr.size() / 1.5 = " << I_A2_TempArr.size() / 1.5 << " &&" << endl;
                  cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": blitz::sum((*P_I_A2_Mask)(xxx,blitz::Range::all())) = " << blitz::sum((*P_I_A2_Mask)(xxx,blitz::Range::all())) << " != 0" << endl;
                #endif
              }
              #ifdef __DEBUG_TELLURIC__
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_ImTimesMask(xxx=" << xxx << ",yyy=" << yyy << ") set to " << D_A2_ImTimesMask(xxx, yyy) << endl;
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_SFTimesMask(xxx=" << xxx << ",yyy=" << yyy << ") set to " << D_A2_SFTimesMask(xxx, yyy) << endl;
              #endif
            }
          }

          D_A1_Sky = 1.;
          #ifdef __DEBUG_TELLURIC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_ImTimesMask = " << D_A2_ImTimesMask << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": *P_D_A2_Prof_Out = " << *P_D_A2_Prof_Out << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": D_A2_SFTimesMask = " << D_A2_SFTimesMask << endl;
            S_MySF = DEBUGDIR + std::string("D_A2_ImTimesMask_beforeFit_iterSky") + to_string(I_Iter_Sky) + debugFilesSuffix + std::string(".fits");
            pfsDRPStella::utils::WriteFits(&D_A2_ImTimesMask, S_MySF);
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": File " << S_MySF << " written" << endl;
            
            S_MySF = DEBUGDIR + std::string("D_A2_SFTimesMask_beforeFit_iterSky") + to_string(I_Iter_Sky) + debugFilesSuffix + std::string(".fits");
            pfsDRPStella::utils::WriteFits(&D_A2_SFTimesMask, S_MySF);
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": File " << S_MySF << " written" << endl;
          #endif
            
          if (!pfsDRPStella::math::LinFitBevington(D_A2_ImTimesMask,
                                                  D_A2_SFTimesMask,
                                                  D_A1_MySP,
                                                  D_A1_Sky,
                                                  S_A1_Args_Fit,
                                                  PP_Args_Fit))
          {
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": ERROR: Fit returned FALSE!" << endl;
            return false;
          }
            
          #ifdef __DEBUG_TELLURIC__
            S_MySF = DEBUGDIR + std::string("D_A2_ImTimesMask_afterFit_iterSky") + to_string(I_Iter_Sky) + debugFilesSuffix + std::string(".fits");
            pfsDRPStella::utils::WriteFits(&D_A2_ImTimesMask, S_MySF);
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": File " << S_MySF << " written" << endl;
            
            S_MySF = DEBUGDIR + std::string("D_A2_SFTimesMask_afterFit_iterSky") + to_string(I_Iter_Sky) + debugFilesSuffix + std::string(".fits");
            pfsDRPStella::utils::WriteFits(&D_A2_SFTimesMask, S_MySF);
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": File " << S_MySF << " written" << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": 3. spectrum_Out = " << spectrum_Out << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": Fit returned TRUE" << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": after Fit: D_A1_MySP = " << D_A1_MySP << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": after Fit: D_A1_Sky = " << D_A1_Sky << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": after Fit: D_A1_ChiSquare_LinFit = " << D_A1_ChiSquare_LinFit << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": after Fit: D_A1_Probability_LinFit = " << D_A1_Probability_LinFit << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": after Fit: D_A2_Sigma_LinFit = " << D_A2_Sigma_LinFit << endl;
          #endif
          if (I_Stop > 0)
            return false;
            
          #ifdef __DEBUG_TELLURIC__
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": __TELLURIC_MINE__: spectrum_Out = " << spectrum_Out << endl;
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": __TELLURIC_MINE__: D_A1_MySP set to " << D_A1_MySP << endl;
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
            pfsDRPStella::utils::WriteFits(&D_A2_Im, S_MySF);
            cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": File " << S_MySF << " written" << endl;
          #endif
            
          Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "STOP");
          if (Pos >= 0)
          {
            if (*(int*)ArgV_In[Pos] == 1)
            {
              cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(STOP) == 1" << endl;
              if (I_Iter_SF == 1){
                cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": KeyWord_Set(STOP) == 1, I_Iter_SF == " << I_Iter_SF << " => returning false" << endl;
                return false;
              }
            }
          }
        }/// end if (I_TELLURIC == 2)

        if (I_Telluric != 2)
          break;
        D_A1_OldSky = abs(D_A1_Sky - D_A1_OldSky) / D_A1_Sky;
        D_A1_OldSky = blitz::where(D_A1_Sky > 0., D_A1_OldSky, 0.);
        #ifdef __DEBUG_SLITFUNC__
          cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": I_Iter_Sky = " << I_Iter_Sky << ": D_A1_OldSky - D_A1_Sky = " << D_A1_OldSky << endl;
        #endif
        if (mean(D_A1_OldSky) < 0.005)
          break;
        D_A1_OldSky = D_A1_Sky;
      } while(I_Iter_Sky < static_cast<int>(_fiberTraceExtractionControl->maxIterSky));
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
      cout << "CFits::SlitFunc: D_A2_Im.rows() = " << D_A2_Im.rows() << endl;
      cout << "CFits::SlitFunc: D_A2_Im.cols() = " << D_A2_Im.cols() << endl;
      cout << "CFits::SlitFunc: SFVecArr = " << SFVecArr << endl;
      cout << "CFits::SlitFunc: D_A2_Weights(0:3, *) = " << D_A2_Weights(blitz::Range(0,3), blitz::Range::all()) << endl;
    #endif
      
    if (I_Telluric == 3)
      SFVecArr = SFVecArr - min(SFVecArr);
      
    #ifdef __DEBUG_SLITFUNC_FILES__
      string S_SF = "SFVecArr_Out.fits";
      pfsDRPStella::utils::WriteFits(&SFVecArr, S_SF);
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
            cout << "CFits::SlitFunc: i_row = " << i_row << ": i_col = " << i_col << ": I_SubPix = " << I_SubPix << ": i_j = " << i_j << ": D_A2_XX(i_row, i_j) = " << D_A2_XX(i_row, i_j) << ": int(D_A2_XX(i_row, i_j)) = " << int(D_A2_XX(i_row, i_j)) << endl;
          #endif
          if ((int(D_A2_XX(i_row, i_j)) == i_col) || ((int(D_A2_XX(i_row, i_j)) <= i_col) && (D_A2_XX(i_row, i_j+1) > i_col))){
            #ifdef __DEBUG_TELLURIC__
              cout << "CFits::SlitFunc: SubPixel found" << endl;
            #endif
            D_Sum_ProfTimesWeight += SFVecArr(i_j) * D_A2_Weights(i_row, I_SubPix);
            #ifdef __DEBUG_TELLURIC__
              cout << "CFits::SlitFunc: i_row = " << i_row << ": i_col = " << i_col << ": I_SubPix = " << I_SubPix << ": i_j = " << i_j << ": SFVecArr(i_j) = " << SFVecArr(i_j) << ": D_A2_Weights(i_row, I_SubPix) = " << D_A2_Weights(i_row, I_SubPix) << ": D_Sum_ProfTimesWeight = " << D_Sum_ProfTimesWeight << endl;
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
          cout << "CFits::SlitFunc: i_row = " << i_row << ": i_col = " << i_col << ": sumProfTimesWeightRow = " << sumProfTimesWeightRow << endl;
        #endif
        D_A1_CY_SP(i_row) += (*P_I_A2_Mask)(i_row, i_col) * D_A2_Im(i_row, i_col) * D_Sum_ProfTimesWeight;
        D_A1_DY(i_row) += (*P_I_A2_Mask)(i_row, i_col) * blitz::pow2(D_Sum_ProfTimesWeight);
        #ifdef __DEBUG_TELLURIC__
          cout << "CFits::SlitFunc: i_row = " << i_row << ": i_col = " << i_col << ": D_A1_CY_SP(i_row) = " << D_A1_CY_SP(i_row) << endl;
          cout << "CFits::SlitFunc: i_row = " << i_row << ": i_col = " << i_col << ": D_A1_DY(i_row) = " << D_A1_DY(i_row) << endl;
        #endif
          
        D_A1_SumMaskProf(i_row) += (*P_I_A2_Mask)(i_row, i_col) * D_Sum_ProfTimesWeight;
        D_A1_SumMaskProfSquaredDivByErr(i_row) += (*P_I_A2_Mask)(i_row, i_col) * blitz::pow2(D_Sum_ProfTimesWeight) / blitz::pow2((*P_D_A2_Errors)(i_row, i_col));
      }/// end for (int i_col=0; i_col<D_A2_Im.cols(); i_col++){
        
      #ifndef __PISKUNOV_ORIG__
        (*P_D_A1_SPErrOut)(i_row) = sqrt(D_Sum_SigmaSquared_AxySquared / blitz::pow2(D_Sum_AxySquared)) / overSample_In;
        
        #ifdef __DEBUG_TELLURIC__
          cout << "CFits::SlitFunc: (*P_D_A1_SPErrOut)(i_row = " << i_row << ") = " << (*P_D_A1_SPErrOut)(i_row) << endl;
        #endif
      #endif
        
      (*P_D_A1_SPOut)(i_row) = (D_A1_CY_SP(i_row) / D_A1_DY(i_row)) / overSample_In;
      #ifdef __DEBUG_TELLURIC__
        cout << "CFits::SlitFunc: (*P_D_A1_SPOut)(i_row = " << i_row << ") = " << (*P_D_A1_SPOut)(i_row) << endl;
      #endif
    }/// end for (int i_row=0; i_row<D_A2_Im.rows(); i_row++)
    #ifdef __DEBUG_SLITFUNC__
      cout << "CFits::SlitFunc: D_A1_CY_SP = " << D_A1_CY_SP << endl;
      cout << "CFits::SlitFunc: D_A1_DY = " << D_A1_DY << endl;
      cout << "CFits::SlitFunc: spectrum_Out = " << spectrum_Out << endl;
      cout << "CFits::SlitFunc: P_D_A1_SPOut = " << *P_D_A1_SPOut << endl;
      cout << "CFits::SlitFunc: P_D_A1_SPErrOut = " << *P_D_A1_SPErrOut << endl;
    #endif
    blitz::Array<double, 1> D_A1_SNR(spectrum_Out.size());
    D_A1_SNR = spectrum_Out / (*P_D_A1_SPErrOut);
    #ifdef __DEBUG_SLITFUNC__
      cout << "CFits::SlitFunc: SNR(spectrum_Out / P_D_A1_SPErrOut) = " << D_A1_SNR << endl;
    #endif
    D_A1_SNR = (*P_D_A1_SPOut) / (*P_D_A1_SPErrOut);
    #ifdef __DEBUG_SLITFUNC__
      cout << "CFits::SlitFunc: SNR(P_D_A1_SPOut / P_D_A1_SPErrOut) = " << D_A1_SNR << endl;
    #endif
    if (I_Telluric == 2)
    {
      /// set P_D_A1_MySky to new sky
      (*P_D_A1_MySky) = D_A1_Sky;
      P_D_A1_ErrSky->resize(D_A2_Im.rows());
      *P_D_A1_ErrSky = D_A2_Sigma_LinFit(blitz::Range::all(), 1);
        
      if ((Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "ERRORS_OUT")) >= 0){
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
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": READY: P_I_A1_JBadVecArr set to " << (*P_I_A1_JBadVecArr) << endl;
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": READY: SFVecArr set to " << SFVecArr << endl;
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": READY: spectrum_Out set to " << spectrum_Out << endl;
      cout << "CFits::SlitFunc: fiberTraceNumber = " << fiberTraceNumber << ": I_Bin = " << I_Bin << ": READY: P_D_A2_Prof_Out set to " << *P_D_A2_Prof_Out << endl;//->transpose(blitz::secondDim, blitz::firstDim) << endl;
    #endif
      
    if (pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "PROF_OUT") < 0)
    {
      delete P_D_A2_Prof_Out;
    }
      
    if (pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "XCOR_PROF") < 0){
      delete(P_D_A1_XCorProfOut);
    }
      
    Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "SP_FIT");
    if (Pos >= 0)
    {
      blitz::Array<double, 1> *P_D_A1_SPFit = (blitz::Array<double, 1>*)ArgV_In[Pos];
      P_D_A1_SPFit->resize(D_A1_MySP.size());
      *P_D_A1_SPFit = D_A1_MySP;
    }
      
    #ifdef __DEBUG_SLITFUNC__
      cout << "deleting pointers" << endl;
    #endif
    if ((Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "IM_OUT")) < 0)
    {
      if (P_D_A2_Im_Out != NULL)
        delete P_D_A2_Im_Out;
    }
      
    #ifdef __DEBUG_TELLURIC__
      S_SP = "spectrum_Out1Out.fits";
      D_A2_SPTemp.resize(spectrum_Out.size(), 1);
      D_A2_SPTemp(blitz::Range::all(), 0) = spectrum_Out;
      pfsDRPStella::utils::WriteFits(&D_A2_SPTemp, S_SP);
    #endif
      
    #ifdef __DEBUG_SEDM__
      string sFileName_SPVecArrOut = DEBUGDIR + std::string("SEDM_spectrum_OutOut") + debugFilesSuffix + std::string(".dat");
      pfsDRPStella::utils::WriteArrayToFile(spectrum_Out, sFileName_SPVecArrOut, string("ascii"));
      
      string sFileName_SFVecArrOut = DEBUGDIR + std::string("SEDM_SFVecArrOut") + debugFilesSuffix + std::string(".dat");
      pfsDRPStella::utils::WriteArrayToFile(SFVecArr, sFileName_SFVecArrOut, string("ascii"));
      
      string S_MaskOut = "Mask_SF.fits";
      if (!pfsDRPStella::utils::WriteFits(P_I_A2_Mask, S_MaskOut)){
        cout << "CFits::SlitFunc: ERROR: WriteFits(Mask) returned FALSE" << endl;
        return false;
      }
      cout << "CFits::SlitFunc: " << CS_MaskOut << " written" << endl;
    #endif
    if ((blitz::sum(*P_I_A2_Mask) > static_cast<int>(P_I_A2_Mask->size()) / 2) && (blitz::sum(*P_I_A2_Mask) == blitz::sum(*P_I_A2_MaskIn))){
      cout << "CFits::SlitFunc: WARNING: No cosmics detected" << endl;
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
    Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "MASK");
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
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::MkSlitFunc()
  {
    blitz::Array<string, 1> S_A1_Args(1);
    S_A1_Args = "";
    void **PP_Args;
    PP_Args = (void**)malloc(sizeof(void*) * 1);
    
    int I_temp = 0;
    PP_Args[0] = &I_temp;
    
    if (!MkSlitFunc(S_A1_Args, PP_Args))
    {
      cout << "CFits::MkSlitFunc(): ERROR: MkSlitFunc(S_A1_Args=" << S_A1_Args << ") returned FALSE => Returning FALSE" << endl;
      return false;
    }
    return true;
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
      cout << "FiberTrace::fitSpline: fiberTraceSwath_In = " << fiberTraceSwath_In << endl;
      cout << "FiberTrace::fitSpline: iFirst_In = " << iFirst_In << endl;
      cout << "FiberTrace::fitSpline: xOverSampled_In = " << xOverSampled_In << endl;
      cout << "FiberTrace::fitSpline: profileXValuesPerRowOverSampled_In = " << profileXValuesPerRowOverSampled_In << endl;
      cout << "FiberTrace::fitSpline: profileXValuesAllRows_In = " << profileXValuesAllRows_In << endl;
    #endif
    /// check input paramters
    if (iFirst_In.size() != fiberTraceSwath_In.rows()){
      cout << "FiberTrace::fitSpline: ERROR: iFirst_In.size(=" << iFirst_In.size() << ") != fiberTraceSwath_In.rows(=" << fiberTraceSwath_In.rows() << ") => Returning FALSE" << endl;
      return false;
    }
    if (fiberTraceSwath_In.cols() != profileXValuesAllRows_In.size()){
      cout << "FiberTrace::fitSpline: ERROR: profileXValuesAllRows_In.size(=" << profileXValuesAllRows_In.size() << ") != fiberTraceSwath_In.cols(=" << fiberTraceSwath_In.cols() << ") => Returning FALSE" << endl;
      return false;
    }
    if (fiberTraceSwath_In.rows() != profileXValuesPerRowOverSampled_In.rows()){
      cout << "FiberTrace::fitSpline: ERROR: profileXValuesPerRowOverSampled_In.size(=" << profileXValuesPerRowOverSampled_In.size() << ") != fiberTraceSwath_In.rows(=" << fiberTraceSwath_In.rows() << ") => Returning FALSE" << endl;
      return false;
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
        *iter_xVec = double(j) * double(_fiberTraceExtractionControl->overSample) + double(iFirst_In(i)) + (double(_fiberTraceExtractionControl->overSample)/2.);
        *iter_yVec = ccdRow(j);
        ++iter_xVec;
        ++iter_yVec;
      }
    }
    
    blitz::Array<double, 1> D_A1_xVec(xVec.data(), blitz::shape(xVec.size()), blitz::neverDeleteData);
    blitz::Array<double, 1> D_A1_yVec(yVec.data(), blitz::shape(yVec.size()), blitz::neverDeleteData);
    blitz::Array<int, 1> I_A1_Uniq(1);
    if (!pfsDRPStella::math::Uniq(D_A1_xVec, I_A1_Uniq)){
      cout << "FiberTrace::fitSpine: ERROR: Uniq returned FALSE" << endl;
      return false;
    }
    #ifdef __DEBUG_SPLINE__
      cout << "FiberTrace::fitSpline: I_A1_Uniq = " << I_A1_Uniq << endl;
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
    blitz::Array<double, 1> D_A1_XTemp = pfsDRPStella::math::DIndGenArr(xOverSampled_In.size());
    blitz::Array<double, 1> D_X(1);
    blitz::Array<double, 1> D_Y(1);
    for (size_t i = 0; i < I_A1_Uniq.size(); ++i){
      D_X(0) = D_A1_xVec(I_A1_Uniq(i));
      if (!pfsDRPStella::math::InterPol(xOverSampled_In, D_A1_XTemp, D_X, D_Y)){
        cout << "FiberTrace::fitSpline: ERROR: InterPol(xOverSampled_In=" << xOverSampled_In << ", D_A1_XTemp=" << D_A1_XTemp << ", D_X=" << D_X << ", D_Y) returned FALSE" << endl;
        return false;
      }
      *iter_xVecSorted = D_Y(0);
      I_A1_Where = blitz::where(fabs(D_A1_xVec - D_A1_xVec(I_A1_Uniq(i))) < 0.000001, 1, 0);
      P_I_A1_Where = pfsDRPStella::math::GetIndex(I_A1_Where, count);
      if (!pfsDRPStella::math::GetSubArrCopy(D_A1_yVec, *P_I_A1_Where, D_A1_SubArr)){
        cout << "FiberTrace::fitSpline: i=" << i << ": ERROR: GetSubArrCopy returned false" << endl;
        return false;
      }
      median = pfsDRPStella::math::Median(D_A1_SubArr);
      #ifdef __DEBUG_SPLINE__
        cout << "FiberTrace::fitSpline: i=" << i << ": D_A1_xVec(I_A1_Uniq(i)=" << I_A1_Uniq(i) << ") = " << D_A1_xVec(I_A1_Uniq(i)) << ": *P_I_A1_Where = " << *P_I_A1_Where << endl;
        cout << "FiberTrace::fitSpline: i=" << i << ": D_A1_SubArr = " << D_A1_SubArr << endl;
        cout << "FiberTrace::fitSpline: i=" << i << ": median = " << median << endl;
      #endif
      *iter_yVecSorted = median;
      ++iter_xVecSorted;
      ++iter_yVecSorted;
      delete(P_I_A1_Where);
    }
    #ifdef __DEBUG_SPLINE__
      blitz::Array<double, 1> D_A1_XVecSorted(xVecSorted.data(), blitz::shape(xVecSorted.size()), blitz::neverDeleteData);
      blitz::Array<double, 1> D_A1_YVecSorted(yVecSorted.data(), blitz::shape(yVecSorted.size()), blitz::neverDeleteData);
      cout << "FiberTrace::fitSpline: xVecSorted = " << D_A1_XVecSorted << endl;
      cout << "FiberTrace::fitSpline: yVecSorted = " << D_A1_YVecSorted << endl;
    #endif
    pfsDRPStella::math::spline spline;
    spline.set_points(xVecSorted,yVecSorted);    // currently it is required that X is already sorted

    /// calculate oversampled profile for each x in xOverSampled_In
    if (profileOverSampled_Out.size() != xOverSampled_In.size())
      profileOverSampled_Out.resize(xOverSampled_In.size());
    for (int i=0; i< xOverSampled_In.size(); ++i){
      if ((xOverSampled_In(i) < xVecSorted[0]) || (xOverSampled_In(i) > xVecSorted[xVecSorted.size()-1]))
        profileOverSampled_Out(i) = 0.;
      else
        profileOverSampled_Out(i) = spline(xOverSampled_In(i));
    }
    #ifdef __DEBUG_SPLINE__
      cout << "FiberTrace::fitSpline: xOverSampled_In = " << xOverSampled_In << endl;
      cout << "FiberTrace::fitSpline: profileOverSampled_Out = " << profileOverSampled_Out << endl;
    #endif
    profileOverSampled_Out = profileOverSampled_Out * double(_fiberTraceExtractionControl->overSample) / blitz::sum(profileOverSampled_Out);
    
    if ((profilePerRow_Out.rows() != fiberTraceSwath_In.rows()) || (profilePerRow_Out.cols() != fiberTraceSwath_In.cols()))
      profilePerRow_Out.resize(fiberTraceSwath_In.rows(), fiberTraceSwath_In.cols());
    
    blitz::Array<double, 1> yProf(profileXValuesAllRows_In.size());
    for (int i_row = 0; i_row < fiberTraceSwath_In.rows(); i_row++){
      if (!pfsDRPStella::math::InterPol(profileOverSampled_Out, profileXValuesPerRowOverSampled_In(i_row, blitz::Range::all()), profileXValuesAllRows_In, yProf)){
        cout << "FiberTrace::fitSpline: ERROR: InterPol(profileOverSampled_Out=" << profileOverSampled_Out << ", profileXValuesPerRowOverSampled_In(i_row=" << i_row << ", blitz::Range::all())=" << profileXValuesPerRowOverSampled_In(i_row, blitz::Range::all()) << ", profileXValuesAllRows_In=" << profileXValuesAllRows_In << ", yProf) returned FALSE => Returning FALSE" << endl;
        return false;
      }
      profilePerRow_Out(i_row, blitz::Range::all()) = blitz::where(yProf < 0., 0., yProf);
      profilePerRow_Out(i_row, blitz::Range::all()) = profilePerRow_Out(i_row, blitz::Range::all()) / blitz::sum(profilePerRow_Out(i_row, blitz::Range::all()));
    }
    #ifdef __DEBUG_SPLINE__
      cout << "FiberTrace::fitSpline: profilePerRow_Out = " << profilePerRow_Out << endl;
    #endif
    return true;
  }

  /** 
   * class FiberTraceSet
   **/
  
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::setFiberTrace(int const i,     ///< which aperture?
                                                                            PTR(FiberTrace<ImageT, MaskT, VarianceT>) trace ///< the FiberTrace for the ith aperture
  ){
    if (i > static_cast<int>(_traces.size())){
      cout << "FiberTraceSet::setFiberTrace: ERROR: position for trace outside range!" << endl;
      return false;
    }
    if (i == static_cast<int>(_traces.size())){
      _traces.push_back(trace);
    }
    else{
      _traces[i] = trace;
    }
    return true;
  }
  
  template<typename ImageT, typename MaskT, typename VarianceT> 
  void pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::addFiberTrace(PTR(FiberTrace<ImageT, MaskT, VarianceT>) trace) ///< the FiberTrace for the ith aperture
  {
    _traces.push_back(trace);
  }
  
  template<typename ImageT, typename MaskT, typename VarianceT> 
  void pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::sortTracesByXCenter()
  {
    for (int i = 0; i < static_cast<int>(_traces.size()); ++i){
      cout << "FiberTraceSet::sortTracesByXCenter: _traces[" << i << "]->getFiberTraceFunction().xCenter = " << _traces[i]->getFiberTraceFunction().xCenter << endl;
    }
    std::vector<float> xCenters;
    for (int iTrace = 0; iTrace < static_cast<int>(_traces.size()); ++iTrace){
      xCenters.push_back(_traces[iTrace]->getFiberTraceFunction().xCenter);
    }
    std::vector<int> sortedIndices(xCenters.size());
    sortedIndices = pfsDRPStella::math::sortIndices(xCenters);
    
    std::vector<PTR(FiberTrace<ImageT, MaskT, VarianceT>)> sortedTraces(_traces.size());
    for (int i = 0; i < static_cast<int>(_traces.size()); ++i){
      sortedTraces[i] = _traces[sortedIndices[i]];
    }
    _traces = sortedTraces;
    return;
  }

  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::setFiberTraceExtractionControl(FiberTraceExtractionControl &fiberTraceExtractionControl){
    PTR(FiberTraceExtractionControl) p_fiberTraceExtractionControl(new FiberTraceExtractionControl(fiberTraceExtractionControl));
    for (unsigned int i=0; i<_traces.size(); ++i){
      if (!_traces[i]->setFiberTraceExtractionControl(p_fiberTraceExtractionControl)){
        cout << "FiberTraceSet::setFiberTraceExtractionControl: ERROR: _traces[" << i << "]->setFiberTraceExtractionControl(fiberTraceExtractionControl) returned FALSE => Returning FALSE" << endl;
        return false;
      }
    }
    return true;
  }
  
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::extractTraceNumber(int traceNumber)
  {
    blitz::Array<std::string, 1> keyWords(1);
    keyWords(0) = "FIBERTRACENUMBER";
    void **args = (void**)malloc(sizeof(void*));
    args[0] = &traceNumber;
    if (!_traces[traceNumber]->MkSlitFunc(keyWords, args)){
      cout << "FiberTraceSet::extractAllTraces: ERROR: _traces[" << traceNumber << "].MkSlitFunc() returned FALSE => Returning FALSE" << endl;
      return false;
    }
    return true;
  }
  
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::extractAllTraces()
  {
    blitz::Array<std::string, 1> keyWords(1);
    keyWords(0) = "FIBERTRACENUMBER";
    void **args = (void**)malloc(sizeof(void*));
    unsigned int i=0;
    args[0] = &i;
    for (i = 0; i < _traces.size(); ++i){
      if (!_traces[i]->MkSlitFunc(keyWords, args)){
        cout << "FiberTraceSet::extractAllTraces: ERROR: _traces[" << i << "].MkSlitFunc() returned FALSE => Returning FALSE" << endl;
        return false;
      }
    }
    return true;
  }

  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::extractTraceNumberFromProfile(int traceNumber)
  {
    blitz::Array<std::string, 1> keyWords(1);
    keyWords(0) = "FIBERTRACENUMBER";
    void **args = (void**)malloc(sizeof(void*));
    args[0] = &traceNumber;
    if (!_traces[traceNumber]->extractFromProfile(keyWords, args)){
      cout << "FiberTraceSet::extractAllTraces: ERROR: _traces[" << traceNumber << "].extractFromProfile() returned FALSE => Returning FALSE" << endl;
      return false;
    }
    return true;
  }
  
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::extractAllTracesFromProfile()
  {
    blitz::Array<std::string, 1> keyWords(1);
    keyWords(0) = "FIBERTRACENUMBER";
    void **args = (void**)malloc(sizeof(void*));
    unsigned int i = 0;
    args[0] = &i;
    for (i = 0; i < _traces.size(); ++i){
      if (!_traces[i]->extractFromProfile(keyWords, args)){
        cout << "FiberTraceSet::extractAllTraces: ERROR: _traces[" << i << "].extractFromProfile() returned FALSE => Returning FALSE" << endl;
        return false;
      }
    }
    return true;
  }

  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::setAllProfiles(FiberTraceSet<ImageT, MaskT, VarianceT> &fiberTraceSet){
    for (unsigned int i = 0; i < _traces.size(); ++i){
      if (!_traces[i]->setProfile(fiberTraceSet.getFiberTrace(i).getProfile())){
        cout << "FiberTraceSet::copyAllProfiles: ERROR: _traces[" << i << "].setProfile(fiberTraceSet.getFiberTrace(" << i << ").getProfile()) returned FALSE => Returning FALSE" << endl;
        return false;
      }
    }
    return true;
  }
  
  namespace pfs{ namespace drp{ namespace stella{ namespace math{
    
/*    template<typename ImageT, typename MaskT, typename VarianceT> 
    PTR(pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>) findAndTraceApertures(const PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) &maskedImage,*/
    template<typename ImageT, typename MaskT, typename VarianceT> 
    pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT> findAndTraceApertures(const PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) &maskedImage,
                              const pfsDRPStella::FiberTraceFunctionFindingControl &fiberTraceFunctionFindingControl){
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "pfsDRPStella::math::findAndTraceApertures started" << endl;
      #endif
      
      if (static_cast<int>(fiberTraceFunctionFindingControl.apertureFWHM * 2.) + 1 <= fiberTraceFunctionFindingControl.nTermsGaussFit){
        cout << "pfsDRPStella::math::findAndTraceApertures: WARNING: fiberTraceFunctionFindingControl.apertureFWHM too small for GaussFit -> Try lower fiberTraceFunctionFindingControl.nTermsGaussFit!" << endl;
        exit(EXIT_FAILURE);
      }
//      PTR(pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>) fiberTraceSet(new pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>());
      pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT> fiberTraceSet = pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>();
      FiberTraceFunction fiberTraceFunction;
      fiberTraceFunction.fiberTraceFunctionControl = fiberTraceFunctionFindingControl.fiberTraceFunctionControl;
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "pfsDRPStella::math::findAndTraceApertures: fiberTraceFunction.fiberTraceFunctionControl set" << endl;
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
      blitz::Array<ImageT, 2> ccdTImage = utils::ndarrayToBlitz(maskedImage->getImage()->getArray());
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "pfsDRPStella::math::findAndTraceApertures: ccdTImage(60, *) = " << ccdTImage(60, blitz::Range::all()) << endl;
      #endif
      blitz::Array<double, 2> ccdImage(2,2);
      pfsDRPStella::math::Double(ccdTImage, ccdImage);
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "pfsDRPStella::math::findAndTraceApertures: ccdImage(60,*) = " << ccdImage(60, blitz::Range::all()) << endl;
      #endif
      blitz::Array<double, 1> D_A1_IndexCol = pfsDRPStella::math::DIndGenArr(ccdImage.cols());
      blitz::Array<double, 1> D_A1_IndexRow = pfsDRPStella::math::DIndGenArr(ccdImage.rows());
      blitz::Array<double, 1> D_A1_X(10);
      blitz::Array<double, 1> D_A1_Y(10);
      blitz::Array<double, 1> D_A1_MeasureErrors(10);
      blitz::Array<double, 1> D_A1_Guess(fiberTraceFunctionFindingControl.nTermsGaussFit);
      blitz::Array<double, 1> D_A1_GaussFit_Coeffs(fiberTraceFunctionFindingControl.nTermsGaussFit);
      blitz::Array<double, 1> D_A1_GaussFit_Coeffs_Bak(fiberTraceFunctionFindingControl.nTermsGaussFit);
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "pfsDRPStella::math::findAndTraceApertures: D_A1_IndexCol = " << D_A1_IndexCol << endl;
      #endif
      blitz::Array<int, 1> I_A1_Signal(maskedImage->getWidth());
      blitz::Array<double, 1> D_A1_ApertureCenter(maskedImage->getHeight());
      blitz::Array<int, 1> I_A1_ApertureCenterInd(maskedImage->getHeight());
      blitz::Array<double, 1> D_A1_ApertureCenterIndex(maskedImage->getHeight());
      blitz::Array<int, 1> I_A1_ApertureCenterIndex(maskedImage->getHeight());
      blitz::Array<double, 1> D_A1_ApertureCenterPos(1);
      blitz::Array<double, 1> *P_D_A1_PolyFitCoeffs = new blitz::Array<double, 1>(fiberTraceFunctionFindingControl.fiberTraceFunctionControl.order);
      blitz::Array<int, 1> I_A1_IndSignal(2);
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "pfsDRPStella::math::findAndTraceApertures: started" << endl;
        blitz::Array<double, 2> D_A2_PixArrayNew(maskedImage->getHeight(), maskedImage->getWidth());
        D_A2_PixArrayNew = 0.;
      #endif
      blitz::Array<int, 1> I_A1_Ind(1);
      blitz::Array<int, 1> I_A1_Where(1);
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "pfsDRPStella::math::findAndTraceApertures: fiberTraceFunctionFindingControl.signalThreshold = " << fiberTraceFunctionFindingControl.signalThreshold << endl;
  //      return false;
      #endif
      
      /// Set all pixels below fiberTraceFunctionFindingControl.signalThreshold to 0.
      ccdImage = blitz::where(ccdImage < fiberTraceFunctionFindingControl.signalThreshold, 0., ccdImage);
      #ifdef __DEBUG_FINDANDTRACE__
        cout << "pfsDRPStella::math::findAndTraceApertures: ccdImage(60,*) = " << ccdImage(60, blitz::Range::all()) << endl;
        string S_TempFileName = DEBUGDIR + "FindAndTraceApertures_thresh.fits";
        pfsDRPStella::utils::WriteFits(&ccdImage, S_TempFileName);
      #endif
      
      int I_MinWidth = int(1.5 * fiberTraceFunctionFindingControl.apertureFWHM);
      if (I_MinWidth < fiberTraceFunctionFindingControl.nTermsGaussFit)
        I_MinWidth = fiberTraceFunctionFindingControl.nTermsGaussFit;
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
          if (!pfsDRPStella::math::CountPixGTZero(I_A1_Signal)){
            cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: pfsDRPStella::math::CountPixGTZero(I_A1_Signal=" << I_A1_Signal << ") returned FALSE => Returning FALSE" << endl;
            exit(EXIT_FAILURE);
          }
          I_FirstWideSignal = pfsDRPStella::math::FirstIndexWithValueGEFrom(I_A1_Signal, I_MinWidth, I_StartIndex);
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_FirstWideSignal found at index " << I_FirstWideSignal << ", I_StartIndex = " << I_StartIndex << endl;
          #endif
          if (I_FirstWideSignal < 0){
            cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": No Aperture found in row " << i_Row << ", trying next row" << endl;
            break;
          }
          else{
            I_FirstWideSignalStart = pfsDRPStella::math::LastIndexWithZeroValueBefore(I_A1_Signal, I_FirstWideSignal) + 1;
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "pfsDRPStella::math::findAndTraceApertures: while: 1. i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_FirstWideSignalStart = " << I_FirstWideSignalStart << endl;
            #endif
              
            I_FirstWideSignalEnd = pfsDRPStella::math::FirstIndexWithZeroValueFrom(I_A1_Signal, I_FirstWideSignal) - 1;
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
            #endif
              
            if (I_FirstWideSignalStart < 0){
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: No start of aperture found -> Going to next Aperture." << endl;
              #endif

              if (I_FirstWideSignalEnd < 0){
                cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": 1. WARNING: No end of aperture found -> Going to next row." << endl;
                break;
              }
              else{
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": End of first wide signal found at index " << I_FirstWideSignalEnd << endl;
                #endif
                /// Set start index for next run
                I_StartIndex = I_FirstWideSignalEnd+1;
              }
            }
            else{ /// Fit Gaussian and Trace Aperture
              if (I_FirstWideSignalEnd < 0){
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": 2. WARNING: No end of aperture found -> Going to next row." << endl;
                  cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_Row_Bak = " << I_Row_Bak << endl;
                  cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": B_ApertureFound = " << B_ApertureFound << endl;
                #endif
                break;
              }
              else{
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": End of first wide signal found at index " << I_FirstWideSignalEnd << endl;
                #endif
                
                if (I_FirstWideSignalEnd - I_FirstWideSignalStart + 1 > fiberTraceFunctionFindingControl.apertureFWHM * D_MaxTimesApertureWidth){
                  I_FirstWideSignalEnd = I_FirstWideSignalStart + int(D_MaxTimesApertureWidth * fiberTraceFunctionFindingControl.apertureFWHM);
                }
                
                /// Set start index for next run
                I_StartIndex = I_FirstWideSignalEnd+1;
              }
              I_Length = I_FirstWideSignalEnd - I_FirstWideSignalStart + 1;
              
              if (fiberTraceFunctionFindingControl.nTermsGaussFit == 0){/// look for maximum only
                D_A1_ApertureCenter = 0.;
                B_ApertureFound = true;
                I_A1_Where.resize(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
                I_A1_Where = blitz::where(fabs(ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)) - max(ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)))) < 0.00001, 1, 0);
                if (!pfsDRPStella::math::GetIndex(I_A1_Where, I_NInd, I_A1_Ind)){
                  cout << "pfsDRPStella::math::findAndTraceApertures: while: ERROR: GetIndex(I_A1_Where=" << I_A1_Where << ") returned FALSE => Returning FALSE" << endl;
                  exit(EXIT_FAILURE);
                }
                D_A1_ApertureCenter(i_Row) = I_FirstWideSignalStart + I_A1_Ind(0);
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Aperture found at " << D_A1_ApertureCenter(i_Row) << endl;
                #endif
                
                /// Set signal to zero
                ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
              }
              else{
                if (I_Length <= fiberTraceFunctionFindingControl.nTermsGaussFit){
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Width of aperture <= " << fiberTraceFunctionFindingControl.nTermsGaussFit << "-> abandoning aperture" << endl;
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
                    cout << "pfsDRPStella::math::findAndTraceApertures: 1. D_A1_Y set to " << D_A1_Y << endl;
                  #endif
                  D_A1_MeasureErrors = sqrt(fabs(D_A1_Y));
                  
                  /// Guess values for GaussFit
                  D_A1_Guess(0) = max(D_A1_Y);
                  D_A1_Guess(1) = double(I_FirstWideSignalStart) + (double((I_FirstWideSignalEnd - I_FirstWideSignalStart)) / 2.);
                  D_A1_Guess(2) = double(fiberTraceFunctionFindingControl.apertureFWHM) / 2.;

                  D_A1_GaussFit_Coeffs = 0.;
                  blitz::Array<double, 1> D_A1_GaussFit_ECoeffs(D_A1_GaussFit_Coeffs.size());
                  D_A1_GaussFit_ECoeffs = 0.;
                  
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_X = " << D_A1_X << ", D_A1_Y = " << D_A1_Y << endl;
                  #endif
                  
                  blitz::Array<int, 2> I_A2_Limited(3,2);
                  I_A2_Limited = 1;
                  blitz::Array<double, 2> D_A2_Limits(3,2);
                  D_A2_Limits(0,0) = 0.;/// Peak lower limit
                  D_A2_Limits(0,1) = 2. * D_A1_Guess(0);/// Peak upper limit
                  D_A2_Limits(1,0) = static_cast<double>(I_FirstWideSignalStart);/// Centroid lower limit
                  D_A2_Limits(1,1) = static_cast<double>(I_FirstWideSignalEnd);/// Centroid upper limit
                  D_A2_Limits(2,0) = double(fiberTraceFunctionFindingControl.apertureFWHM) / 4.;/// Sigma lower limit
                  D_A2_Limits(2,1) = double(fiberTraceFunctionFindingControl.apertureFWHM);/// Sigma upper limit
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": 1. starting MPFitGaussLim: D_A1_Guess = " << D_A1_Guess << ", I_A2_Limited = " << I_A2_Limited << ", D_A2_Limits = " << D_A2_Limits << endl;
                  #endif
                  if (!MPFitGaussLim(D_A1_X,
                                    D_A1_Y,
                                    D_A1_MeasureErrors,
                                    D_A1_Guess,
                                    I_A2_Limited,
                                    D_A2_Limits,
                                    false,
                                    false,
                                    D_A1_GaussFit_Coeffs,
                                    D_A1_GaussFit_ECoeffs)){
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: GaussFit FAILED -> abandoning aperture" << endl;
                    #endif
                  
                    /// Set start index for next run
                    I_StartIndex = I_FirstWideSignalEnd+1;
                  
                    ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                  }
                  else{
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
  //                    return false;
                      if (D_A1_GaussFit_Coeffs(0) < fiberTraceFunctionFindingControl.saturationLevel/5.){
                        cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Signal less than 20% of saturation level" << endl;
                      }
                      if (D_A1_GaussFit_Coeffs(0) > fiberTraceFunctionFindingControl.saturationLevel){
                        cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Signal appears to be saturated" << endl;
                      }
                      if ((D_A1_GaussFit_Coeffs(1) < double(I_FirstWideSignalStart) + (double(I_Length)/4.)) || (D_A1_GaussFit_Coeffs(1) > double(I_FirstWideSignalStart) + (double(I_Length) * 3./4.))){
                        cout << "pfsDRPStella::math::findAndTraceApertures: while: Warning: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Center of Gaussian far away from middle of signal" << endl;
                      }
                    #endif
                    if ((D_A1_GaussFit_Coeffs(1) < double(I_FirstWideSignalStart)) || (D_A1_GaussFit_Coeffs(1) > double(I_FirstWideSignalEnd))){
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                      #endif
                      /// Set signal to zero
                      ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                        
                        
                      /// Set start index for next run
                      I_StartIndex = I_FirstWideSignalEnd+1;
                    }
                    else{
                      if ((D_A1_GaussFit_Coeffs(2) < fiberTraceFunctionFindingControl.apertureFWHM / 4.) || (D_A1_GaussFit_Coeffs(2) > fiberTraceFunctionFindingControl.apertureFWHM)){
                        #ifdef __DEBUG_FINDANDTRACE__
                          cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: FWHM = " << D_A1_GaussFit_Coeffs(2) << " outside range -> abandoning aperture" << endl;
                        #endif
                        /// Set signal to zero
                        ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                        #ifdef __DEBUG_FINDANDTRACE__
                          cout << "pfsDRPStella::math::findAndTraceApertures: while: B_ApertureFound = " << B_ApertureFound << ": 1. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                          cout << "pfsDRPStella::math::findAndTraceApertures: while: 1. ccdImage(i_Row = " << i_Row << ", blitz::Range(I_FirstWideSignalStart = " << I_FirstWideSignalStart << ", I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << ")) set to " << ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)) << endl;
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
                          cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Aperture found at " << D_A1_ApertureCenter(i_Row) << endl;
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
            cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Starting to trace aperture" << endl;
  //        #endif
          D_A1_GaussFit_Coeffs_Bak = D_A1_GaussFit_Coeffs;
          I_Row_Bak = i_Row;
          while(B_ApertureFound && (I_ApertureLost < fiberTraceFunctionFindingControl.nLost) && (i_Row < maskedImage->getHeight()-1) && I_Length < fiberTraceFunctionFindingControl.maxLength){
            i_Row++;
            I_Length++;
            if (fiberTraceFunctionFindingControl.nTermsGaussFit == 0){/// look for maximum only
              B_ApertureFound = true;
              D_Max = max(ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)));
              if (D_Max < fiberTraceFunctionFindingControl.signalThreshold){
                I_ApertureLost++;
              }
              else{
                I_A1_Where.resize(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
                I_A1_Where = blitz::where(fabs(ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)) - D_Max) < 0.00001, 1, 0);
                if (!pfsDRPStella::math::GetIndex(I_A1_Where, I_NInd, I_A1_Ind)){
                  cout << "pfsDRPStella::math::findAndTraceApertures: ERROR: GetIndex(I_A1_Where=" << I_A1_Where << ") returned FALSE => Returning FALSE" << endl;
                  exit(EXIT_FAILURE);
                }
                D_A1_ApertureCenter(i_Row) = I_FirstWideSignalStart + I_A1_Ind(0);
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Aperture found at " << D_A1_ApertureCenter(i_Row) << endl;
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
                  cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": start or end of aperture outside CCD -> Aperture lost" << endl;
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
                
                if (I_Length <= fiberTraceFunctionFindingControl.nTermsGaussFit){
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Warning: Width of Aperture <= " << fiberTraceFunctionFindingControl.nTermsGaussFit << " -> Lost Aperture" << endl;
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
                  I_A1_IndSignal = blitz::where(D_A1_Y < fiberTraceFunctionFindingControl.signalThreshold, 0, 1);
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "pfsDRPStella::math::findAndTraceApertures: I_MinWidth = " << I_MinWidth << ": I_A1_IndSignal = " << I_A1_IndSignal << endl;
                  #endif
                  if (blitz::sum(I_A1_IndSignal) < I_MinWidth){
                    /// Set signal to zero
                    ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                    I_ApertureLost++;
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "pfsDRPStella::math::findAndTraceApertures: Signal not wide enough => Aperture lost" << endl;
                    #endif
                  }
                  else{
                    D_A1_Y = blitz::where(D_A1_Y < 0.00000001, 1., D_A1_Y);
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "pfsDRPStella::math::findAndTraceApertures: 2. D_A1_Y set to " << D_A1_Y << endl;
                    #endif
                    D_A1_MeasureErrors = sqrt(fabs(D_A1_Y));
                    D_A1_Guess = D_A1_GaussFit_Coeffs_Bak;
                    
                    D_A1_GaussFit_Coeffs = 0.;
                    blitz::Array<double, 1> D_A1_GaussFit_ECoeffs(D_A1_GaussFit_Coeffs.size());
                    D_A1_GaussFit_ECoeffs = 0.;
                    
                    #ifdef __DEBUG_FINDANDTRACE__
                    cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_X = " << D_A1_X << ", D_A1_Y = " << D_A1_Y << endl;
                    #endif
                    
                    blitz::Array<int, 2> I_A2_Limited(3,2);
                    I_A2_Limited = 1;
                    blitz::Array<double, 2> D_A2_Limits(3,2);
                    D_A2_Limits(0,0) = 0.;/// Peak lower limit
                    D_A2_Limits(0,1) = 2. * D_A1_Guess(0);/// Peak upper limit
                    D_A2_Limits(1,0) = static_cast<double>(I_FirstWideSignalStart);/// Centroid lower limit
                    D_A2_Limits(1,1) = static_cast<double>(I_FirstWideSignalEnd);/// Centroid upper limit
                    D_A2_Limits(2,0) = fiberTraceFunctionFindingControl.apertureFWHM / 4.;/// Sigma lower limit
                    D_A2_Limits(2,1) = fiberTraceFunctionFindingControl.apertureFWHM;/// Sigma upper limit
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "pfsDRPStella::math::findAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": 2. starting MPFitGaussLim: D_A2_Limits = " << D_A2_Limits << endl;
                    #endif
                    if (!MPFitGaussLim(D_A1_X,
                                      D_A1_Y,
                                      D_A1_MeasureErrors,
                                      D_A1_Guess,
                                      I_A2_Limited,
                                      D_A2_Limits,
                                      false,
                                      false,
                                      D_A1_GaussFit_Coeffs,
                                      D_A1_GaussFit_ECoeffs)){
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Warning: GaussFit FAILED" << endl;
                      #endif
                      /// Set signal to zero
                      ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart-1, I_FirstWideSignalEnd+1)) = 0.;
                    
                      I_ApertureLost++;
                      //          return false;
                    }
                    else{
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
                        if (D_A1_GaussFit_Coeffs(0) < fiberTraceFunctionFindingControl.saturationLevel/5.){
                          cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Signal less than 20% of saturation level" << endl;
                        }
                        if (D_A1_GaussFit_Coeffs(0) > fiberTraceFunctionFindingControl.saturationLevel){
                          cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Signal appears to be saturated" << endl;
                        }
                      #endif
                      //          if ((D_A1_GaussFit_Coeffs(1) < double(I_FirstWideSignalStart) - (double(I_Length)/4.)) || (D_A1_GaussFit_Coeffs(1) > double(I_FirstWideSignalStart) + (double(I_Length) * 3./4.))){
                      //            cout << "pfsDRPStella::math::findAndTraceApertures: Warning: i_Row = " << i_Row << ": Center of Gaussian far away from middle of signal" << endl;
                      //          }
                      if ((D_A1_GaussFit_Coeffs(1) < D_A1_GaussFit_Coeffs_Bak(1) - 1.) || (D_A1_GaussFit_Coeffs(1) > D_A1_GaussFit_Coeffs_Bak(1) + 1.)){
                        #ifdef __DEBUG_FINDANDTRACE__
                          cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                        #endif
                        /// Set signal to zero
                        ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                          
                        I_ApertureLost++;
                      }
                      else{
                        if ((D_A1_GaussFit_Coeffs(2) < fiberTraceFunctionFindingControl.apertureFWHM / 4.) || (D_A1_GaussFit_Coeffs(2) > fiberTraceFunctionFindingControl.apertureFWHM)){
                          #ifdef __DEBUG_FINDANDTRACE__
                            cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: FWHM = " << D_A1_GaussFit_Coeffs(2) << " outside range -> abandoning aperture" << endl;
                          #endif
                          /// Set signal to zero
                          ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                          #ifdef __DEBUG_FINDANDTRACE__
                            cout << "pfsDRPStella::math::findAndTraceApertures: 2. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                          #endif
                          I_ApertureLost++;
                        }
                        else{
                          I_ApertureLost = 0;
                          B_ApertureFound = true;
                          D_A1_ApertureCenter(i_Row) = D_A1_GaussFit_Coeffs(1);
                          #ifdef __DEBUG_FINDANDTRACE__
                            cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Aperture found at " << D_A1_ApertureCenter(i_Row) << endl;
                          #endif
                          D_A1_GaussFit_Coeffs_Bak = D_A1_GaussFit_Coeffs;
                          //I_LastRowWhereApertureWasFound = i_Row;
                        }
                      }/// end else if ((D_A1_GaussFit_Coeffs(1) >= D_A1_Guess(1) - 1.) && (D_A1_GaussFit_Coeffs(1) <= D_A1_Guess(1) + 1.))
                    }/// end else if (GaussFit(D_A1_X, D_A1_Y, D_A1_GaussFit_Coeffs, S_A1_KeyWords_GaussFit, PP_Args_GaussFit))
                    ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                  }/// end else if (blitz::sum(I_A1_Signal) >= I_MinWidth){
                }/// end if (I_Length > 3)
              }/// end else if (I_ApertureStart >= 0. && I_ApertureEnd < maskedImage->getWidth())
            }/// end else if GaussFit
          }///end while(B_ApertureFound && (I_ApertureLost < 3) && i_Row < maskedImage->getHeight() - 2))
          
          /// Fit Polynomial to traced aperture positions
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_ApertureCenter = " << D_A1_ApertureCenter << endl;
  //          pfsDRPStella::utils::WriteArrayToFile(D_A1_ApertureCenter, std::string("xCenters_before_polyfit_ap")+to_string(I_ApertureNumber)+std::string(".dat"), std::string("ascii"));
          #endif
          I_A1_ApertureCenterIndex = blitz::where(D_A1_ApertureCenter > 0., 1, 0);
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_A1_ApertureCenterIndex = " << I_A1_ApertureCenterIndex << endl;
          #endif
          if (!pfsDRPStella::math::GetIndex(I_A1_ApertureCenterIndex, I_ApertureLength, I_A1_ApertureCenterInd)){
            cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: pfsDRPStella::math::GetIndex(I_A1_ApertureCenterIndex=" << I_A1_ApertureCenterIndex << ", I_ApertureLength, I_A1_ApertureCenterInd) returned FALSE -> Returning FALSE" << endl;
            exit(EXIT_FAILURE);
          }
          if (I_ApertureLength >= static_cast<int>(fiberTraceFunctionFindingControl.minLength)){
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_IndexRow = " << D_A1_IndexRow << endl;
              cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_A1_ApertureCenterInd = " << I_A1_ApertureCenterInd << endl;
            #endif
            if (!pfsDRPStella::math::GetSubArrCopy(D_A1_IndexRow,
                                                  I_A1_ApertureCenterInd, 
                                                  D_A1_ApertureCenterIndex)){
              cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: pfsDRPStella::math::GetSubArrCopy(D_A1_IndexRow = " << D_A1_IndexRow << ", I_A1_ApertureCenterInd = " << I_A1_ApertureCenterInd << ", D_A1_ApertureCenterIndex) returned FALSE -> returning FALSE" << endl;
              exit(EXIT_FAILURE);
            }
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_ApertureCenterIndex = " << D_A1_ApertureCenterIndex << endl;
            #endif
              
            if (!pfsDRPStella::math::GetSubArrCopy(D_A1_ApertureCenter,
                                                  I_A1_ApertureCenterInd,
                                                  D_A1_ApertureCenterPos)){
              cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: pfsDRPStella::math::GetSubArrCopy(D_A1_ApertureCenter = " << D_A1_ApertureCenter << ", I_A1_ApertureCenterInd = " << I_A1_ApertureCenterInd << ", D_A1_ApertureCenterIndex) returned FALSE -> returning FALSE" << endl;
            exit(EXIT_FAILURE);
            }
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_ApertureCenterPos = " << D_A1_ApertureCenterPos << endl;
            #endif
                
            /// Fit Polynomial
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_ApertureCenterIndex = " << D_A1_ApertureCenterIndex << endl;
              cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_ApertureCenterPos = " << D_A1_ApertureCenterPos << endl;
              cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": order = " << fiberTraceFunctionFindingControl.fiberTraceFunctionControl.order << endl;
            #endif
            if (!pfsDRPStella::math::PolyFit(D_A1_ApertureCenterIndex, 
                                            D_A1_ApertureCenterPos, 
                                            fiberTraceFunctionFindingControl.fiberTraceFunctionControl.order,
                                            P_D_A1_PolyFitCoeffs)){
              cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: PolyFit returned FALSE -> Returning FALSE" << endl;
              exit(EXIT_FAILURE);
            }
            PTR(pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>) fiberTrace(new pfsDRPStella::FiberTrace< ImageT, MaskT, VarianceT >(maskedImage));
                
            I_ApertureNumber++;
          
            fiberTraceFunction.xCenter = D_A1_ApertureCenterPos(int(D_A1_ApertureCenterIndex.size()/2.));
            fiberTraceFunction.yCenter = D_A1_ApertureCenterIndex(int(D_A1_ApertureCenterIndex.size()/2.));
            fiberTraceFunction.yHigh = D_A1_ApertureCenterIndex(int(D_A1_ApertureCenterIndex.size()-1)) - fiberTraceFunction.yCenter;
            fiberTraceFunction.yLow = D_A1_ApertureCenterIndex(0) - fiberTraceFunction.yCenter;
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "pfsDRPStella::math::findAndTraceApertures: P_D_A1_PolyFitCoeffs = " << *P_D_A1_PolyFitCoeffs << endl;
            #endif
            fiberTraceFunction.coefficients.resize(P_D_A1_PolyFitCoeffs->size());
            for (int iter=0; iter < static_cast<int>(P_D_A1_PolyFitCoeffs->size()); iter++)
              fiberTraceFunction.coefficients[iter] = (*P_D_A1_PolyFitCoeffs)(iter);
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "pfsDRPStella::math::findAndTraceApertures: fiberTraceFunction.xCenter = " << fiberTraceFunction.xCenter << endl;
              cout << "pfsDRPStella::math::findAndTraceApertures: fiberTraceFunction.yCenter = " << fiberTraceFunction.yCenter << endl;
              cout << "pfsDRPStella::math::findAndTraceApertures: fiberTraceFunction.yLow = " << fiberTraceFunction.yLow << endl;
              cout << "pfsDRPStella::math::findAndTraceApertures: fiberTraceFunction.yHigh = " << fiberTraceFunction.yHigh << endl;
  //            cout << "pfsDRPStella::math::findAndTraceApertures: fiberTraceFunction.coefficients = " << fiberTraceFunction.coefficients << endl;
            #endif
            if (!fiberTrace->setFiberTraceFunction(fiberTraceFunction)){
              cout << "FindAndTraceApertures: ERROR: setFiberTraceFunction returned FALSE" << endl;
              exit(EXIT_FAILURE);
            }
          
            blitz::Array<double, 1> D_A1_XCenters_Y = pfsDRPStella::math::DIndGenArr(ccdImage.rows());
            blitz::Array<double, 1> D_A1_XCenters = pfsDRPStella::math::Poly(D_A1_XCenters_Y, *P_D_A1_PolyFitCoeffs);
            blitz::Array<double, 1> D_A1_XCenters_Ap(fiberTraceFunction.yHigh - fiberTraceFunction.yLow + 1);
            D_A1_XCenters_Ap = D_A1_XCenters(blitz::Range(fiberTraceFunction.yCenter + fiberTraceFunction.yLow, fiberTraceFunction.yCenter + fiberTraceFunction.yHigh));
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "pfsDRPStella::math::findAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_XCenters = " << D_A1_XCenters << endl;
            #endif
            std::vector<float> xCenters(D_A1_XCenters_Ap.size());
            for (int iter = 0; iter < static_cast<int>(D_A1_XCenters_Ap.size()); iter++)
              xCenters[iter] = static_cast<float>(D_A1_XCenters_Ap(iter));
          
            #ifdef __DEBUG_FINDANDTRACE__
              for (int iter=0; iter < static_cast<int>(xCenters.size()); ++iter){
                cout << "FindAndTraceApertures: xCenters[" << iter << "] = " << xCenters[iter] << endl;
              }
            #endif
            if (!fiberTrace->setXCenters(xCenters)){
              cout << "FindAndTraceApertures: ERROR: setXCenters returned FALSE" << endl;
              exit(EXIT_FAILURE);
            }
            #ifdef __DEBUG_FINDANDTRACE__
              for (int iter=0; iter < static_cast<int>(xCenters.size()); ++iter){
                cout << "FindAndTraceApertures: xCenters[" << iter << "] = " << xCenters[iter] << " =? fiberTrace->getXCenters()[" << iter << "] = fiberTrace->getXCenters()[iter] = " << fiberTrace->getXCenters()[iter] << endl;
              }
            #endif
            if (!fiberTrace->createTrace(maskedImage)){
              cout << "FindAndTraceApertures: ERROR: createTrace returned FALSE" << endl;
              exit(EXIT_FAILURE);
            }
            fiberTraceSet.addFiberTrace(fiberTrace);
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
        for (int i = 0; i < static_cast<int>(fiberTraceSet.size()); i++){
          blitz::Array<ImageT, 2> TImage = utils::ndarrayToBlitz(fiberTraceSet.getFiberTrace(i).getTrace().getImage()->getArray());
          pfs::drp::stella::math::Float(TImage, F_A2_Trace);
          S_FileNameTrace = S_NewFileName + "_trace_" + std::to_string(i) + ".fits";
          pfs::drp::stella::utils::WriteFits(&F_A2_Trace, S_FileNameTrace);
        }
        
      #endif
      delete(P_D_A1_PolyFitCoeffs);
      return fiberTraceSet;
    }
    
    /*****************************************************************/
    /*  Sub method for CubicSpline, Legendre, and Chebyshev          */
    /*****************************************************************/
    double GetNormalized(const double XVal, const double XMin, const double XMax)
    {
      double Normalized;
      Normalized = ((2 * XVal) - (XMax + XMin)) / (XMax - XMin);
      #ifdef __DEBUG_GET__
        cout << "GetNormalized: XVal = " << XVal << ", Normalized = " << Normalized << endl;
      #endif
      return Normalized;
    }
    
    /*****************************************************************/
    /*  Sub method for LinearSpline and CubicSpline                  */
    /*****************************************************************/
    double GetA(const double XVal, const double XMin, const double XMax, const int Order)
    {
      double A;
      A = (pfs::drp::stella::math::GetJ(XVal, XMin, XMax, Order) + 1.) - pfs::drp::stella::math::GetS(XVal, XMin, XMax, Order);
      #ifdef __DEBUG_GET__
        cout << "GetA: XVal = " << XVal << ", A = " << A << endl;
      #endif
      return A;
    }
    
    /*****************************************************************/
    /*  Sub method for LinearSpline and CubicSpline                  */
    /*****************************************************************/
    double GetB(const double XVal, const double XMin, const double XMax, const int Order)
    {
      double B;
      B = pfs::drp::stella::math::GetS(XVal, XMin, XMax, Order) - pfs::drp::stella::math::GetJ(XVal, XMin, XMax, Order);
      #ifdef __DEBUG_GET__
        cout << "GetB: XVal = " << XVal << ", A = " << B << endl;
      #endif
      return B;
    }
    
    /*****************************************************************/
    /* Sub method for LinearSpline and CubicSpline                   */
    /*****************************************************************/
    long GetJ(const double XVal, const double XMin, const double XMax, const int Order)
    {
      long J;
      double S;
      S = pfs::drp::stella::math::GetS(XVal, XMin, XMax, Order);
      J = (long)S;
      #ifdef __DEBUG_GET__
        cout << "GetJ: XVal = " << XVal << ", S = " << S << ", J = " << J << endl;
      #endif
      return J;
    }
    
    /** **************************************************/
    /** Sub method for LinearSpline and CubicSpline      */
    /** **************************************************/
    double GetS(const double XVal, const double XMin, const double XMax, const int Order)
    {
      double S;
      S = (XVal - XMin) / (XMax - XMin) * Order;//(GetJ(XVal, XMin, XMax) + 1.) - GetS(XVal, XMin, XMax);
      #ifdef __DEBUG_GET__
        cout << "GetS: XVal = " << XVal << ", S = " << S << endl;
      #endif
      return S;
    }
    
    /** **************************************************/
    bool LinearSpline(blitz::Array<double, 1> &D_A1_XCenters_Out,
                      const blitz::Array<double, 1> &D_A1_Coeffs_In,
                      double D_XCenter_In,
                      double D_YCenter_In,
                      double D_YMin_In,
                      double D_YMax_In,
                      double D_XLow_In,
                      double D_XHigh_In,
                      int I_Order_In,
                      int I_NCols_In,
                      int I_NRows_In){
      for (int m = 0; m < I_NRows_In; m++)
      {
        int n = pfs::drp::stella::math::GetJ(0. - D_YCenter_In + (double)m, D_YMin_In, D_YMax_In, I_Order_In);
        
        D_A1_XCenters_Out(m) = D_XCenter_In + (D_A1_Coeffs_In(n) * pfs::drp::stella::math::GetA(0. - D_YCenter_In + (double)m, D_YMin_In, D_YMax_In, I_Order_In)) + (D_A1_Coeffs_In(n+1) * pfs::drp::stella::math::GetB(0. - D_YCenter_In + m, D_YMin_In, D_YMax_In, I_Order_In));
        if (D_A1_XCenters_Out(m) < 0. - D_XLow_In)
          D_A1_XCenters_Out(m) = 0. - D_XLow_In;
        if (D_A1_XCenters_Out(m) > I_NCols_In - D_XHigh_In)
          D_A1_XCenters_Out(m) = I_NCols_In - D_XHigh_In;
      }
      return true;
    }
    
    /** **************************************************/
    /**       */
    /**       */
    /** **************************************************/
    bool CubicSpline(blitz::Array<double, 1> &D_A1_XCenters_Out,
                     const blitz::Array<double, 1> &D_A1_Coeffs_In,
                     double D_XCenter_In,
                     double D_YCenter_In,
                     double D_YMin_In,
                     double D_YMax_In,
                     double D_XLow_In,
                     double D_XHigh_In,
                     int I_Order_In,
                     int I_NCols_In,
                     int I_NRows_In){
      int m, o;
      blitz::Array<double, 1> D_A1_ZArr(4);
      
      cout << "CubicSpline: D_A1_XCenters_Out.size() returned " << D_A1_XCenters_Out.size() << endl;
      
      for (m = 0; m < I_NRows_In; m++)
      {
        #ifdef __DEBUG_TRACEFUNC__
          double D_Normalized = GetNormalized((double)m - D_YMin_In, D_YMin_In, D_YMax_In);
          cout << "CubicSpline: m = " << m << ": D_Normalized = " << D_Normalized << endl;
        #endif
        /*y = sum from i=0 to 3 {c_{i+j} * z_i}
         *      z_0 = a**3
         *      z_1 = 1 + 3 * a * (1 + a * b)
         *      z_2 = 1 + 3 * b * (1 + a * b)
         *      z_3 = b**3*/
        D_A1_ZArr(0) = std::pow(pfs::drp::stella::math::GetA(m, D_YMin_In, D_YMax_In, I_Order_In), 3);
        D_A1_ZArr(1) = 1. + (3. * pfs::drp::stella::math::GetA(m, D_YMin_In, D_YMax_In, I_Order_In) * (1. + (pfs::drp::stella::math::GetA(m, D_YMin_In, D_YMax_In, I_Order_In) * pfs::drp::stella::math::GetB(m, D_YMin_In, D_YMax_In, I_Order_In))));
        D_A1_ZArr(2) = 1. + (3. * pfs::drp::stella::math::GetB(m, D_YMin_In, D_YMax_In, I_Order_In) * (1. + (pfs::drp::stella::math::GetA(m, D_YMin_In, D_YMax_In, I_Order_In) * pfs::drp::stella::math::GetB(m, D_YMin_In, D_YMax_In, I_Order_In))));
        D_A1_ZArr(3) = std::pow(pfs::drp::stella::math::GetB(m, D_YMin_In, D_YMax_In, I_Order_In), 3);
        #ifdef __DEBUG_TRACEFUNC__
          cout << "CubicSpline: P_ZArr(0) = " << D_A1_ZArr(0) << ", P_ZArr(1) = " << D_A1_ZArr(1) << ", P_ZArr(2) = " << D_A1_ZArr(2) << ", P_ZArr(3) = " << D_A1_ZArr(3) << endl;
        #endif
        D_A1_XCenters_Out(m) = D_XCenter_In;
        for (o = 0; o < 4; o++)
        {
          D_A1_XCenters_Out(m) += D_A1_Coeffs_In(o + pfsDRPStella::math::GetJ(0. - D_YCenter_In + m, D_YMin_In, D_YMax_In, I_Order_In)) * D_A1_ZArr(o);
        }
        if (D_A1_XCenters_Out(m) < 0. - D_XLow_In)
          D_A1_XCenters_Out(m) = 0. - D_XLow_In;
        if (D_A1_XCenters_Out(m) > I_NCols_In - D_XHigh_In)
          D_A1_XCenters_Out(m) = I_NCols_In - D_XHigh_In;
      }// end for (i = 0; i < XMax; i++)
      D_A1_ZArr.resize(0);
      return true;
    }
    
    /** **************************************************/
    /**       */
    /**       */
    /** **************************************************/
    bool ChebyLegend(blitz::Array<double, 1> &D_A1_XCenters_Out,
                     double &D_YMin_Out,
                     double &D_YMax_Out,
                     const blitz::Array<double, 1> &D_A1_Coeffs_In,
                     const double D_XCenter_In,
                     const double D_YCenter_In,
                     const double D_YMin_In,
                     const double D_YMax_In,
                     const double D_XLow_In,
                     const double D_XHigh_In,
                     const int I_Order_In,
                     const int I_NCols_In,
                     const int I_NRows_In,
                     const string &S_Function_In){
      string sLegendre("legendre");
      string sChebyshev("chebyshev");
      int m, n;
      double D_Normalized;
      bool B_XMax_Set = false;
      bool B_Begin_Found = false;
      blitz::Array<double, 1> D_A1_ZArr(3);
      
      #ifdef __DEBUG_TRACEFUNC__
        cout << "ChebyLegend: started: S_Function_In = " << S_Function_In << ", D_XCenter_In = " << D_XCenter_In << ", D_YCenter_In = " << D_YCenter_In << ", D_YMin_In = " << D_YMin_In << ", D_YMax_In = " << D_YMax_In << ", D_XLow_In = " << D_XLow_In << ", D_XHigh_In = " << D_XHigh_In << ", I_Order_In = " << I_Order_In << ", Coeffs = " << D_A1_Coeffs_In << endl;
        cout << "ChebyLegend: D_A1_XCenters_Out.size() returned " << D_A1_XCenters_Out.size() << endl;
      #endif
      
      if (D_A1_XCenters_Out.size() < D_YMax_In - D_YMin_In + 1.)
      {
        cout << "ChebyLegend: D_YMax_In = " << D_YMax_In << ", D_YMin_In = " << D_YMin_In << endl;
        cout << "ChebyLegend: D_A1_XCenters_Out.size(=" << D_A1_XCenters_Out.size() << ") < D_YMax_In(=" << D_YMax_In << ") - D_YMin_In(=" << D_YMin_In << ") + 1 = " << D_YMax_In - D_YMin_In + 1. << " => Returning FALSE" << endl;
        return false;
      }
      
      /// m: Pixel No of 2nd dim (x-direction, independent variable)
      D_YMin_Out = 0.;
      D_YMax_Out = I_NRows_In-1.;
      for (m = 0; m < I_NRows_In; m++)
      {
        D_Normalized = pfs::drp::stella::math::GetNormalized(m, D_YMin_In, D_YMax_In);
        #ifdef __DEBUG_TRACEFUNC__
          cout << "ChebyLegend: m = " << m << ": D_Normalized set to " << D_Normalized << endl;
        #endif
        D_A1_XCenters_Out(m) = D_XCenter_In;
        for (n = 0; n < I_Order_In; n++)
        {
          if (n == 0)
          {
            D_A1_ZArr(0) = 0.;
            D_A1_ZArr(1) = 0.;
            D_A1_ZArr(2) = 1.;
          }
          else if (n == 1)
          {
            D_A1_ZArr(0) = 0;
            D_A1_ZArr(1) = 1;
            D_A1_ZArr(2) = D_Normalized;
          }
          else
          {
            D_A1_ZArr(0) = D_A1_ZArr(1);
            D_A1_ZArr(1) = D_A1_ZArr(2);
            if (S_Function_In.compare(sChebyshev) == 0)
            {
              D_A1_ZArr(2) = (2. * D_Normalized * D_A1_ZArr(1)) - D_A1_ZArr(0);
              #ifdef __DEBUG_TRACEFUNC__
                cout << "ChebyLegend: S_Function_In = <" << S_Function_In << "> == 'chebyshev' : D_A1_ZArr(2) = " << D_A1_ZArr(2) << endl;
              #endif
            }
            else if (S_Function_In.compare(sLegendre) == 0)
            {
              //Y = sum from i=1!!! to order {c_i * x_i}
              D_A1_ZArr(2) = ((((2. * ((double)n + 1.)) - 3.) * D_Normalized * D_A1_ZArr(1)) - (((double)n - 1.) * D_A1_ZArr(0))) / (double)n;
              #ifdef __DEBUG_TRACEFUNC__
                cout << "ChebyLegend: S_Function_In = <" << S_Function_In << "> == 'legendre' : D_A1_ZArr(2) = " << D_A1_ZArr(2) << endl;
              #endif
            }
            else
            {
              #ifdef __DEBUG_TRACEFUNC__
                cout << "ChebyLegend: Cannot associate S_Function = <" << S_Function_In << "> to '" << sChebyshev << "' of '" << sLegendre << "'" << endl;
              #endif
              return false;
            }
          }
          //Y = sum from i=1!!! to order {c_i * x_i}
          D_A1_XCenters_Out(m) += D_A1_Coeffs_In(n) * D_A1_ZArr(2);
          #ifdef __DEBUG_TRACEFUNC__
            cout << "ChebyLegend: D_A1_XCenters_Out(m=" << m << ") set to += D_A1_Coeffs_In(n=" << n << ")=" << D_A1_Coeffs_In(n) << " * D_A1_ZArr(2)=" << D_A1_ZArr(2) << " = " << D_A1_XCenters_Out(m) << endl;
          #endif
        }// end for (n = 0; n < Order; n++)
        #ifdef __DEBUG_TRACEFUNC__
          cout << "ChebyLegend: D_XLow_In = " << D_XLow_In << endl;
          cout << "ChebyLegend: I_NCols_In(=" << I_NCols_In <<") - D_XHigh_In(=" << D_XHigh_In << ") = " << I_NCols_In - D_XHigh_In << endl;
        #endif
        if (D_A1_XCenters_Out(m) < 0. - D_XLow_In || D_A1_XCenters_Out(m) > I_NCols_In - D_XHigh_In){
          if (!B_Begin_Found){
            D_YMin_Out = m;
            #ifdef __DEBUG_TRACEFUNC__
              cout << "ChebyLegend: D_YMin_Out set to " << D_YMin_Out << endl;
            #endif
          }
          else{
            if (!B_XMax_Set){
              D_YMax_Out = m;
              B_XMax_Set = true;
              #ifdef __DEBUG_TRACEFUNC__
                cout << "ChebyLegend: B_XMax_Set set to TRUE: D_YMax_Out set to " << D_YMax_Out << endl;
              #endif
            }
          }
          //      D_A1_XCenters_Out(m) = 0. - D_XLow_In;
        }
        else
          B_Begin_Found = true;
      }// end for (i = 0; i < XMax; i++)
      D_A1_ZArr.resize(0);
      return true;
    }
    
    /*****************************************************************/
    /*       */
    /*       */
    /*****************************************************************/
    bool Legendre(blitz::Array<double, 1> &D_A1_XCenters_Out,
                  double &D_YMin_Out,
                  double &D_YMax_Out,
                  const blitz::Array<double, 1> &D_A1_Coeffs_In,
                  const double D_XCenter_In,
                  const double D_YCenter_In,
                  const double D_YMin_In,
                  const double D_YMax_In,
                  const double D_XLow_In,
                  const double D_XHigh_In,
                  const int I_Order_In,
                  const int I_NCols_In,
                  const int I_NRows_In
                 ){
      string sLegendre = "legendre";
      
      #ifdef __DEBUG_LEGENDRE__
        cout << "Legendre: starting  (ChebyLegend(D_A1_XCenters_Out=" << D_A1_XCenters_Out << ", D_A1_Coeffs_In=" << D_A1_Coeffs_In << ", D_XCenter_In=" << D_XCenter_In << ", D_YCenter_In=" << D_YCenter_In << ", D_YMin_In=" << D_YMin_In << ", D_YMax_In=" << D_YMax_In << ", D_XLow_In=" << D_XLow_In << ", D_XHigh_In=" << D_XHigh_In << ", I_Order_In=" << I_Order_In << ", I_NCols_In=" << I_NCols_In << ", sLegendre=" << sLegendre << "));" << endl;
      #endif
      return (ChebyLegend(D_A1_XCenters_Out,
                          D_YMin_Out,
                          D_YMax_Out,
                          D_A1_Coeffs_In,
                          D_XCenter_In,
                          D_YCenter_In,
                          D_YMin_In,
                          D_YMax_In,
                          D_XLow_In,
                          D_XHigh_In,
                          I_Order_In,
                          I_NCols_In,
                          I_NRows_In,
                          sLegendre));
    }
    
    /*****************************************************************/
    /*       */
    /*       */
    /*****************************************************************/
    bool Chebyshev(blitz::Array<double, 1> &D_A1_XCenters_Out,
                   double &D_YMin_Out,
                   double &D_YMax_Out,
                   const blitz::Array<double, 1> &D_A1_Coeffs_In,
                   const double D_XCenter_In,
                   const double D_YCenter_In,
                   const double D_YMin_In,
                   const double D_YMax_In,
                   const double D_XLow_In,
                   const double D_XHigh_In,
                   const int I_Order_In,
                   const int I_NCols_In,
                   const int I_NRows_In)
    {
      string sChebyshev = "chebyshev";
      return (ChebyLegend(D_A1_XCenters_Out,
                          D_YMin_Out,
                          D_YMax_Out,
                          D_A1_Coeffs_In,
                          D_XCenter_In,
                          D_YCenter_In,
                          D_YMin_In,
                          D_YMax_In,
                          D_XLow_In,
                          D_XHigh_In,
                          I_Order_In,
                          I_NCols_In,
                          I_NRows_In,
                          sChebyshev));
    }
    
    blitz::Array<double, 1> Poly(const blitz::Array<double, 1> &xVec,
                                  const blitz::Array<double, 1> &coeffsVec){
      int ii = 0;
      blitz::Array<double, 1> D_A1_Out(xVec.size());
      #ifdef __DEBUG_POLY__
        cout << "Poly: xVec = " << xVec << endl;
        cout << "Poly: coeffsVec = " << coeffsVec << endl;
        cout << "Poly: D_A1_Out set to " << D_A1_Out << endl;
      #endif
      int I_PolynomialOrder = coeffsVec.size() - 1;
      #ifdef __DEBUG_POLY__
        cout << "Poly: I_PolynomialOrder set to " << I_PolynomialOrder << endl;
      #endif
      if (I_PolynomialOrder == 0){
        D_A1_Out = coeffsVec(0);
        #ifdef __DEBUG_POLY__
          cout << "Poly: I_PolynomialOrder == 0: D_A1_Out set to " << D_A1_Out << endl;
        #endif
        return D_A1_Out;
      }
      D_A1_Out = coeffsVec(I_PolynomialOrder);
      #ifdef __DEBUG_POLY__
        cout << "Poly: I_PolynomialOrder != 0: D_A1_Out set to " << D_A1_Out << endl;
      #endif
      for (ii = I_PolynomialOrder-1; ii >= 0; ii--){
        D_A1_Out = D_A1_Out * xVec + coeffsVec(ii);
        #ifdef __DEBUG_POLY__
          cout << "Poly: I_PolynomialOrder != 0: for (ii = " << ii << "; ii >= 0; ii--) D_A1_Out set to " << D_A1_Out << endl;
        #endif
      }
      return D_A1_Out;
    }

    double Poly(const double D_X_In,
                const blitz::Array<double, 1> &D_A1_Coeffs){
      blitz::Array<double, 1> D_A1_X(1);
      D_A1_X = D_X_In;
      blitz::Array<double, 1> D_A1_Y = pfs::drp::stella::math::Poly(D_A1_X, D_A1_Coeffs);
      double D_Y = D_A1_Y(0);
      return D_Y;
    }
    
    /**
     *  Returns Indexes of I_A1_Where where I_A1_Where equals 1 and writes of blitz::sum(I_A1_Where) to I_NInd_Out
     **/
    blitz::Array<int,1>* GetIndex(const blitz::Array<int,1> &I_A1_Where, int &I_NInd_Out){
      I_NInd_Out = blitz::sum(I_A1_Where);
      int arrsize = I_NInd_Out;
      if (arrsize == 0){
        arrsize = 1;
      }
      blitz::Array<int,1> *P_Index_Out = new blitz::Array<int,1>(arrsize);
      (*P_Index_Out) = -1;
      unsigned int j=0;
      for (unsigned int i=0;i<I_A1_Where.size();i++){
        if (I_A1_Where(i) == 1){
          (*P_Index_Out)(j) = i;
          j++;
        }
      }
      return P_Index_Out;
    }
    
    /**
     *  Returns Indexes of I_A1_Where where I_A1_Where equals 1 and writes of blitz::sum(I_A1_Where) to I_NInd_Out
     **/
    bool GetIndex(const blitz::Array<int,1> &I_A1_Where, 
                  int &I_NInd_Out, 
                  blitz::Array<int, 1> &I_IndArr_Out){
      I_NInd_Out = blitz::sum(I_A1_Where);
      int arrsize = I_NInd_Out;
      if (arrsize == 0)
        arrsize = 1;
      I_IndArr_Out.resize(arrsize);
      I_IndArr_Out = -1;
      if (I_NInd_Out == 0)
        return false;
      unsigned int j=0;
      for (unsigned int i=0;i<I_A1_Where.size();i++){
        if (I_A1_Where(i) == 1){
          I_IndArr_Out(j) = i;
          j++;
        }
      }
      return true;
    }
    
    /**
     * Returns Indexes of I_A2_Where where I_A2_Where equals 1 and writes of blitz::sum(I_A2_Where) to I_NInd_Out
     **/
    blitz::Array<int,2>* GetIndex(const blitz::Array<int,2> &I_A2_Where, 
                                  int &I_NInd_Out){
      I_NInd_Out = blitz::sum(I_A2_Where);
      int arrsize = I_NInd_Out;
      if (arrsize == 0)
        arrsize = 1;
      blitz::Array<int,2> *P_Index_Out = new blitz::Array<int,2>(arrsize,2);
      (*P_Index_Out) = -1;
      int j=0;
      for (int i_row = 0; i_row < I_A2_Where.rows(); i_row++){
        for (int i_col = 0; i_col < I_A2_Where.cols(); i_col++){
          if (I_A2_Where(i_row, i_col) == 1){
            (*P_Index_Out)(j,0) = i_row;
            (*P_Index_Out)(j,1) = i_col;
            j++;
          }
        }
      }
      return P_Index_Out;
    }
    
    /**
     * Returns Indexes of I_A2_Where where I_A2_Where equals 1 and writes of blitz::sum(I_A2_Where) to I_NInd_Out
     **/
    bool GetIndex(const blitz::Array<int,2> &I_A2_Where, 
                  int &I_NInd_Out, 
                  blitz::Array<int, 2> &I_IndArr_Out){
      I_NInd_Out = blitz::sum(I_A2_Where);
      int arrsize = I_NInd_Out;
      if (arrsize == 0)
        arrsize = 1;
      I_IndArr_Out.resize(arrsize,2);
      I_IndArr_Out = -1;
      if (I_NInd_Out == 0)
        return false;
      int j=0;
      for (int i_row = 0; i_row < I_A2_Where.rows(); i_row++){
        for (int i_col = 0; i_col < I_A2_Where.cols(); i_col++){
          if (I_A2_Where(i_row, i_col) == 1){
            I_IndArr_Out(j,0) = i_row;
            I_IndArr_Out(j,1) = i_col;
            j++;
          }
        }
      }
      return true;
    }
    
    
    /**
     * Calculates aperture minimum pixel, central position, and maximum pixel for the trace,
     * and writes result to I_A2_MinCenMax_Out
     **/
    bool calcMinCenMax(const blitz::Array<float, 1> &xCenters_In,
                       float xHigh_In,
                       float xLow_In,
                       int yCenter_In,
                       int yLow_In,
                       int yHigh_In,
                       int nPixCutLeft_In,
                       int nPixCutRight_In,
                       blitz::Array<int, 2> &I_A2_MinCenMax_Out){
      if (static_cast<int>(xCenters_In.size()) != static_cast<int>(yHigh_In - yLow_In + 1)){
        cout << "calcMinCenMax: ERROR: xCenters_In.size(=" << xCenters_In.size() << ") != yHigh_In - yLow_In + 1=" << yHigh_In - yLow_In + 1 << endl;
        return false;
      }
      blitz::Array<float, 1> F_A1_XCenters(xCenters_In.size());
      F_A1_XCenters = xCenters_In + 0.5;
      #ifdef __DEBUG_MINCENMAX__
        cout << "CFits::calcMinCenMax: F_A1_XCenters = " << F_A1_XCenters << endl;
      #endif
      blitz::Array<int, 1> I_A1_Fix = pfsDRPStella::math::Int(F_A1_XCenters);
      I_A2_MinCenMax_Out.resize(xCenters_In.size(), 3);
      I_A2_MinCenMax_Out = 0;
      
      I_A2_MinCenMax_Out(blitz::Range::all(), 1) = I_A1_Fix;
      
      #ifdef __DEBUG_MINCENMAX__
        cout << "CFits::calcMinCenMax: I_A2_MinCenMax_Out(*,1) = " << I_A2_MinCenMax_Out(blitz::Range::all(), 1) << endl;
      #endif
      blitz::Array<float, 1> F_A1_Temp(F_A1_XCenters.size());
      F_A1_Temp = F_A1_XCenters + xLow_In;
      
      I_A2_MinCenMax_Out(blitz::Range::all(), 0) = pfsDRPStella::math::Fix(F_A1_Temp);// - I_NPixCut_Left;///(*P_I_A1_Temp); /// Left column of order
      
      #ifdef __DEBUG_MINCENMAX__
        cout << "CFits::calcMinCenMax: I_A2_MinCenMax_Out(*,0) = " << I_A2_MinCenMax_Out(blitz::Range::all(), 0) << endl;
      #endif
      F_A1_Temp = F_A1_XCenters + xHigh_In;
      
      I_A2_MinCenMax_Out(blitz::Range::all(), 2) = pfsDRPStella::math::Fix(F_A1_Temp);
      
      #ifdef __DEBUG_MINCENMAX__
        cout << "CFits::calcMinCenMax: I_A2_MinCenMax_Out(*,2) = " << I_A2_MinCenMax_Out(blitz::Range::all(), 2) << endl;
      #endif
      
      blitz::Array<int, 1> I_A1_NPixLeft(I_A2_MinCenMax_Out.rows());
      I_A1_NPixLeft = I_A2_MinCenMax_Out(blitz::Range::all(),1) - I_A2_MinCenMax_Out(blitz::Range::all(),0);

      blitz::Array<int, 1> I_A1_NPixRight(I_A2_MinCenMax_Out.rows());
      I_A1_NPixRight = I_A2_MinCenMax_Out(blitz::Range::all(),2) - I_A2_MinCenMax_Out(blitz::Range::all(),1);

      #ifdef __DEBUG_MINCENMAX__
        cout << "CFits::calcMinCenMax: I_A1_NPixLeft(=" << I_A1_NPixLeft << endl;
        cout << "CFits::calcMinCenMax: I_A1_NPixRight(=" << I_A1_NPixRight << endl;
      #endif
      
      blitz::Array<int, 1> I_A1_I_NPixX(I_A2_MinCenMax_Out.rows());
      I_A1_I_NPixX = I_A2_MinCenMax_Out(blitz::Range::all(), 2) - I_A2_MinCenMax_Out(blitz::Range::all(), 0) + 1;
      
      #ifdef __DEBUG_MINCENMAX__
        cout << "CFits::calcMinCenMax: I_A1_I_NPixX = " << I_A1_I_NPixX << endl;
      #endif
      
      int I_MaxPixLeft = max(I_A1_NPixLeft);
      int I_MaxPixRight = max(I_A1_NPixRight);
      int I_MinPixLeft = min(I_A1_NPixLeft);
      int I_MinPixRight = min(I_A1_NPixRight);
      
      if (I_MaxPixLeft > I_MinPixLeft)
        I_A2_MinCenMax_Out(blitz::Range::all(),0) = I_A2_MinCenMax_Out(blitz::Range::all(),1) - I_MaxPixLeft + nPixCutLeft_In;
      
      if (I_MaxPixRight > I_MinPixRight)
        I_A2_MinCenMax_Out(blitz::Range::all(),2) = I_A2_MinCenMax_Out(blitz::Range::all(),1) + I_MaxPixRight - nPixCutRight_In;

      #ifdef __DEBUG_MINCENMAX__
        cout << "CFits::calcMinCenMax: I_A2_MinCenMax_Out = " << I_A2_MinCenMax_Out << endl;
      #endif
      
      return true;
    }
    
    /**
     * Calculates Slit Function for each pixel in an aperture row from oversampled Slit Function oSlitFunc_In,
     * and writes result to slitFunc_Out
     **
    bool CalcSF(const blitz::Array<double, 1> &xCenters_In,
                unsigned int row_In,
                float xHigh_In,
                float xLow_In,
                unsigned int overSample_In_In,
                const blitz::Array<double, 1> &oSlitFunc_In,
                const PTR(afwImage::Image<float>) image_In,
                blitz::Array<double, 1> &slitFunc_Out){
      if ((row_In < 0) || (row_In >= image_In->getHeight())){
        cout << "CFits::CalcSF: ERROR: row_In=" << row_In << " outside range" << endl;
        return false;
      }
  
      blitz::firstIndex i;
      int I_NXSF = xHigh_In - xLow_In + 1;

      blitz::Array<double, 1> XVecArr(oSlitFunc_In.size());
      XVecArr = (i + 0.5) / double(overSample_In_In) - 1.;
      double D_XCenMXC = xCenters_In(row_In) - pfsDRPStella::math::Int(xCenters_In(row_In));
      XVecArr += D_XCenMXC;
  
      blitz::Array<double, 1> D_A1_Range(2);
  
      slitFunc_Out.resize(I_NXSF);
      for (int i_col=0; i_col<I_NXSF; i_col++){
        D_A1_Range(0) = i_col;
        D_A1_Range(1) = i_col+1;
        if (!pfsDRPStella::math::IntegralUnderCurve(XVecArr, oSlitFunc_In, D_A1_Range, slitFunc_Out(i_col))){
          cout << "CFits::CalcSF: row_In = " << row_In << ": ERROR: IntegralUnderCurve returned FALSE" << endl;
          return false;
        }
      }
  
      return true;  
    }*/
    
    /**
     * Fix(double)
     * Returns integer value cut at decimal point. If D_In is negative the integer value greater or equal than D_In is returned.
     **/
    template <typename T> 
    int Fix(T D_In){
      return (((D_In < T(0.)) && (T(static_cast<int>(D_In)) < D_In)) ? static_cast<int>(D_In) + 1 : static_cast<int>(D_In));
    }
    
    /**
     *      Fix(blitz::Array<double, 1> &VecArr)
     *      Returns an Array of the same size containing the Fix integer values of VecArr.
     **/
    template <typename T> 
    blitz::Array<int, 1> Fix(const blitz::Array<T, 1> &VecArr){
      blitz::Array<int, 1> TempIntVecArr(VecArr.size());
      for (unsigned int m = 0; m < VecArr.size(); m++)
        TempIntVecArr(m) = pfsDRPStella::math::Fix(VecArr(m));
      return TempIntVecArr;
    }
    
    /**
     *     Fix(blitz::Array<double, 2> &Arr)
     *     Returns an Array of the same size containing the Fix integer values of Arr (see int Fix(double D_In)).
     **/
    template <typename T> 
    blitz::Array<int, 2> Fix(const blitz::Array<T, 2> &Arr){
      blitz::Array<int, 2> TempIntArr(Arr.rows(), Arr.cols());
      for (int m = 0; m < Arr.rows(); m++){
        for (int n = 0; n < Arr.cols(); m++){
          TempIntArr(m, n) = pfsDRPStella::math::Fix(Arr(m, n));
        }
      }
      return TempIntArr;
    }
    
    /**
     * Fix(double)
     * Returns long integer value cut at decimal point (See int Fix(double)).
     **/
    template <typename T> 
    long FixL(T D_In){
      return ((D_In < 0.) && (T(static_cast<long>(D_In)) < D_In)) ? static_cast<long>(D_In) + 1 : static_cast<long>(D_In);
    }
    
    /**
     *      FixL(blitz::Array<double, 1> &VecArr)
     *      Returns an Array of the same size containing the fix long integer values of VecArr (see int Fix(double D_In)).
     **/
    template <typename T> 
    blitz::Array<long, 1> FixL(const blitz::Array<T, 1> &VecArr){
      blitz::Array<long, 1> TempLongVecArr(VecArr.size());
      for (unsigned int m = 0; m < VecArr.size(); m++)
        TempLongVecArr(m) = pfsDRPStella::math::FixL(VecArr(m));
      return TempLongVecArr;
    }
    
    /**
     *     FixL(blitz::Array<double, 2> &Arr, CString Mode)
     *     Returns an Array of the same size containing the long integer values of Arr (see int Fix(double D_In)).
     **/
    template <typename T> 
    blitz::Array<long, 2> FixL(const blitz::Array<T, 2> &Arr){
      blitz::Array<long, 2> TempIntArr(Arr.rows(), Arr.cols());
      for (int m = 0; m < Arr.rows(); m++){
        for (int n = 0; n < Arr.cols(); m++){
          TempIntArr(m, n) = pfsDRPStella::math::FixL(Arr(m, n));
        }
      }
      return TempIntArr;
    }
    
    template <typename T> 
    int Int(T D_In){
      return static_cast<int>(D_In);
    }
    
    template <typename T> 
    blitz::Array<int, 1> Int(const blitz::Array<T, 1> &VecArr){
      blitz::Array<int, 1> TempIntVecArr(VecArr.size());
      for (unsigned int m = 0; m < VecArr.size(); m++)
        TempIntVecArr(m) = pfsDRPStella::math::Int(VecArr(m));
      return TempIntVecArr;
    }
    
    template <typename T> 
    blitz::Array<int, 2> Int(const blitz::Array<T, 2> &Arr){
      blitz::Array<int, 2> TempIntArr(Arr.rows(), Arr.cols());
      for (int m = 0; m < Arr.rows(); m++){
        for (int n = 0; n < Arr.cols(); n++){
          TempIntArr(m, n) = pfsDRPStella::math::Int(Arr(m, n));
        }
      }
      return TempIntArr;
    }
    
    template <typename T> 
    long Long(T D_In){
      return static_cast<long>(D_In);
    }
    
    template <typename T> 
    blitz::Array<long, 1> Long(const blitz::Array<T, 1> &VecArr){
      blitz::Array<long, 1> TempLongVecArr(VecArr.size());
      for (unsigned int m = 0; m < VecArr.size(); m++)
        TempLongVecArr(m) = pfsDRPStella::math::Long(VecArr(m));
      return TempLongVecArr;
    }
    
    template <typename T> 
    blitz::Array<long, 2> Long(const blitz::Array<T, 2> &Arr){
      blitz::Array<long, 2> TempIntArr(Arr.rows(), Arr.cols());
      for (int m = 0; m < Arr.rows(); m++){
        for (int n = 0; n < Arr.cols(); n++){
          TempIntArr(m, n) = pfsDRPStella::math::Long(Arr(m, n));
        }
      }
      return TempIntArr;
    }
    
    template <typename T> 
    void Float(const blitz::Array<T, 1> &Arr, blitz::Array<float, 1>& Arr_Out){
      Arr_Out.resize(Arr.size());
      for (int m = 0; m < static_cast<int>(Arr.size()); m++){
        Arr_Out(m) = static_cast<float>(Arr(m));
      }
      return;
    }
    
    template <typename T> 
    blitz::Array<float, 1> Float(const blitz::Array<T, 1> &Arr){
      blitz::Array<float, 1> Arr_Out(Arr.size());
      for (int m = 0; m < static_cast<int>(Arr.size()); m++){
        Arr_Out(m) = static_cast<float>(Arr(m));
      }
      return Arr_Out;
    }
    
    template <typename T> 
    void Float(const blitz::Array<T, 2> &Arr, blitz::Array<float, 2>& Arr_Out){
      Arr_Out.resize(Arr.rows(), Arr.cols());
      for (int m = 0; m < Arr.rows(); m++){
        for (int n = 0; n < Arr.cols(); n++){
          Arr_Out(m, n) = static_cast<float>(Arr(m, n));
        }
      }
      return;
    }
    
    template <typename T> 
    blitz::Array<float, 2> Float(const blitz::Array<T, 2> &Arr){
      blitz::Array<float, 2> Arr_Out(Arr.rows(), Arr.cols());
      for (int m = 0; m < Arr.rows(); m++){
        for (int n = 0; n < Arr.cols(); n++){
          Arr_Out(m, n) = static_cast<float>(Arr(m, n));
        }
      }
      return Arr_Out;
    }
    
    template <typename T> 
    void Double(const blitz::Array<T, 1> &Arr, blitz::Array<double, 1>& Arr_Out){
      Arr_Out.resize(Arr.size());
      for (int m = 0; m < static_cast<int>(Arr.size()); m++){
        Arr_Out(m) = double(Arr(m));
      }
      return;
    }
    
    template <typename T> 
    blitz::Array<double, 1> Double(const blitz::Array<T, 1> &Arr){
      blitz::Array<double, 1> Arr_Out(Arr.size());
      for (int m = 0; m < static_cast<int>(Arr.size()); m++){
        Arr_Out(m) = double(Arr(m));
      }
      return Arr_Out;
    }
    
    template <typename T> 
    void Double(const blitz::Array<T, 2> &Arr, blitz::Array<double, 2>& Arr_Out){
      Arr_Out.resize(Arr.rows(), Arr.cols());
      for (int m = 0; m < Arr.rows(); m++){
        for (int n = 0; n < Arr.cols(); n++){
          Arr_Out(m, n) = double(Arr(m, n));
        }
      }
      return;
    }
    
    template <typename T> 
    blitz::Array<double, 2> Double(const blitz::Array<T, 2> &Arr){
      blitz::Array<double, 2> Arr_Out(Arr.rows(), Arr.cols());
      for (int m = 0; m < Arr.rows(); m++){
        for (int n = 0; n < Arr.cols(); n++){
          Arr_Out(m, n) = double(Arr(m, n));
        }
      }
      return Arr_Out;
    }
    
    /**
     * Calculate Integral under curve from D_A1_XInt(0) to D_A1_XInt(1)
     **/
    bool IntegralUnderCurve(const blitz::Array<double, 1> &D_A1_XIn,
                            const blitz::Array<double, 1> &D_A1_YIn,
                            const blitz::Array<double, 1> &D_A1_XInt,
                            double &D_Integral_Out){
      #ifdef __DEBUG_INTEGRAL__
        cout << "DFits::IntegralUnderCurve: D_A1_XIn = " << D_A1_XIn << endl;
        cout << "DFits::IntegralUnderCurve: D_A1_YIn = " << D_A1_YIn << endl;
        cout << "DFits::IntegralUnderCurve: D_A1_XInt = " << D_A1_XInt << endl;
      #endif
      if (D_A1_XIn.size() < 2){
        cout << "CFits::IntegralUnderCurve: ERROR: D_A1_XIn.size() < 2 => Returning FALSE" << endl;
        return false;
      }
      if (D_A1_YIn.size() < 2){
        cout << "CFits::IntegralUnderCurve: ERROR: D_A1_YIn.size() < 2 => Returning FALSE" << endl;
        return false;
      }
      if (D_A1_XIn.size() != D_A1_YIn.size()){
        cout << "CFits::IntegralUnderCurve: ERROR: D_A1_XIn.size() != D_A1_YIn.size() => Returning FALSE" << endl;
        return false;
      }
      if (D_A1_XInt.size() != 2){
        cout << "CFits::IntegralUnderCurve: ERROR: D_A1_XInt.size() != 2 => Returning FALSE" << endl;
        return false;
      }
      if (D_A1_XInt(0) > D_A1_XIn(D_A1_XIn.size()-1)){
        cout << "CFits::IntegralUnderCurve: WARNING: D_A1_XInt(0)(=" << D_A1_XInt(0) << ") > D_A1_XIn(" << D_A1_XIn.size()-1 << ")(=" << D_A1_XIn(D_A1_XIn.size()-1) << ")" << endl;
        D_Integral_Out = 0.;
        return true;
      }
      if (D_A1_XInt(1) < D_A1_XIn(0)){
        cout << "CFits::IntegralUnderCurve: WARNING: D_A1_XInt(1)(=" << D_A1_XInt(1) << ") < D_A1_XIn(0)(=" << D_A1_XIn(0) << ")" << endl;
        D_Integral_Out = 0.;
        return true;
      }

      blitz::Array<double, 1> D_A1_XTemp(D_A1_XIn.size() + 2);
      D_A1_XTemp = 0.;
      int I_IndXIn = 0;
      int I_IndX = 1;
      while ((D_A1_XIn(I_IndXIn) < D_A1_XInt(0)) && (I_IndXIn < static_cast<int>(D_A1_XIn.size()))){
        #ifdef __DEBUG_INTEGRAL__
          cout << "CFits::IntegralUnderCurve: D_A1_XIn(I_IndXIn=" << I_IndXIn << ") = " << D_A1_XIn(I_IndXIn) << " < D_A1_XInt(0) = " << D_A1_XInt(0) << endl;
        #endif
        I_IndXIn++;
      }
      #ifdef __DEBUG_INTEGRAL__
        cout << "CFits::IntegralUnderCurve: I_IndXIn = " << I_IndXIn << endl;
      #endif
      D_A1_XTemp(0) = D_A1_XInt(0);
      if ((I_IndXIn < 0) || (I_IndXIn >= static_cast<int>(D_A1_XIn.size()))){
        cout << "CFits::IntegralUnderCurve: ERROR: (I_IndXIn=" << I_IndXIn << " < 0) || (I_IndXIn >= D_A1_XIn.size()=" << D_A1_XIn.size() << ")" << endl;
        return false;
      }
      while (D_A1_XIn(I_IndXIn) < D_A1_XInt(1)){
        #ifdef __DEBUG_INTEGRAL__
          cout << "CFits::IntegralUnderCurve: D_A1_XIn(I_IndXIn=" << I_IndXIn << ") = " << D_A1_XIn(I_IndXIn) << " < D_A1_XInt(1) = " << D_A1_XInt(1) << endl;
        #endif
        D_A1_XTemp(I_IndX) = D_A1_XIn(I_IndXIn);
        I_IndX++;
        I_IndXIn++;
        if (I_IndXIn >= static_cast<int>(D_A1_XIn.size())){
          break;
        }
      }
      #ifdef __DEBUG_INTEGRAL__
        cout << "CFits::IntegralUnderCurve: D_A1_XTemp set to " << D_A1_XTemp << endl;
      #endif
      D_A1_XTemp(I_IndX) = D_A1_XInt(1);
      blitz::Array<double, 1> D_A1_X(I_IndX+1);
      D_A1_X = D_A1_XTemp(blitz::Range(0, I_IndX));
      #ifdef __DEBUG_INTEGRAL__
        cout << "CFits::IntegralUnderCurve: D_A1_X set to " << D_A1_X << endl;
      #endif
  
      blitz::Array<double, 1> D_A1_Y(D_A1_X.size());
//      blitz::Array<double, 1> *P_D_A1_Y = new blitz::Array<double, 1>(D_A1_X.size());
      if (!pfsDRPStella::math::InterPol(D_A1_YIn, D_A1_XIn, D_A1_X, D_A1_Y)){
        cout << "CFits::IntegralUnderCurve: ERROR: InterPol returned FALSE" << endl;
        return false;
      }
//      D_A1_Y = (*P_D_A1_Y);
//      delete(P_D_A1_Y);
      #ifdef __DEBUG_INTEGRAL__
        cout << "CFits::IntegralUnderCurve: D_A1_X = " << D_A1_X << endl;
        cout << "CFits::IntegralUnderCurve: D_A1_Y = " << D_A1_Y << endl;
      #endif
    //  return false;
      D_Integral_Out = 0.;
      double D_Integral = 0.;
      blitz::Array<double, 2> D_A2_Coords(2,2);
      for (unsigned int i=1; i<D_A1_X.size(); i++){
        D_A2_Coords(0,0) = D_A1_X(i-1);
        D_A2_Coords(1,0) = D_A1_X(i);
        D_A2_Coords(0,1) = D_A1_Y(i-1);
        D_A2_Coords(1,1) = D_A1_Y(i);
        if (!pfsDRPStella::math::IntegralUnderLine(D_A2_Coords, D_Integral)){
          cout << "CFits::IntegralUnderCurve: ERROR: IntegralUnderLine(" << D_A2_Coords << ", " << D_Integral << ") returned FALSE" << endl;
          return false;
        }
        D_Integral_Out += D_Integral;
        #ifdef __DEBUG_INTEGRAL__
          cout << "CFits::IntegralUnderCurve: i=" << i << ": D_Integral_Out = " << D_Integral_Out << endl;
        #endif
      }
      return true;
    }

    /**
     * Calculate Integral under line between two points
     * **/
    bool IntegralUnderLine(const blitz::Array<double, 2> &D_A2_Coords_In,
                           double &D_Integral_Out){
      if (D_A2_Coords_In(0,0) == D_A2_Coords_In(1,0)){
        D_Integral_Out = 0.;
        return true;
      }
      blitz::Array<double, 1> D_A1_X(2);
      blitz::Array<double, 1> D_A1_Y(2);
      if (D_A2_Coords_In(0,0) > D_A2_Coords_In(1,0)){
        D_A1_X(0) = D_A2_Coords_In(1,0);
        D_A1_Y(0) = D_A2_Coords_In(1,1);
        D_A1_X(1) = D_A2_Coords_In(0,0);
        D_A1_Y(1) = D_A2_Coords_In(0,1);
      }
      else{
        D_A1_X(0) = D_A2_Coords_In(0,0);
        D_A1_Y(0) = D_A2_Coords_In(0,1);
        D_A1_X(1) = D_A2_Coords_In(1,0);
        D_A1_Y(1) = D_A2_Coords_In(1,1);
      }
      #ifdef __DEBUG_INTEGRAL__
        cout << "CFits::IntegralUnderLine: D_A1_X = " << D_A1_X << endl;
        cout << "CFits::IntegralUnderLine: D_A1_Y = " << D_A1_Y << endl;
      #endif
      if (fabs((D_A1_X(0) - D_A1_X(1)) / D_A1_X(0)) < 5.e-8){
        D_Integral_Out = 0.;
        return true;
      }
      double D_BinStart_X, D_BinEnd_X;//, D_Bin_Ya, D_Bin_Yb;
      D_Integral_Out = 0.;
    
      D_BinStart_X = D_A1_X(0);
      D_BinEnd_X = D_A1_X(1);
    
      ///fit straight line to coordinates
      if (fabs(D_A1_X(0) - D_A1_X(1)) < 0.000002){
        return true;
      }
      blitz::Array<double, 1> *P_D_A1_Coeffs = new blitz::Array<double, 1>(2);
      if (!pfsDRPStella::math::PolyFit(D_A1_X,
                                       D_A1_Y,
                                       1,
                                       P_D_A1_Coeffs)){
        cout << "CFits::IntegralUnderLine: fabs(D_A1_X(0)(=" << D_A1_X(0) << ") - D_A1_X(1)(=" << D_A1_X(1) << ")) = " << fabs(D_A1_X(0) - D_A1_X(1)) << endl;
        cout << "CFits::IntegralUnderLine: ERROR: PolyFit(" << D_A1_X << ", " << D_A1_Y << ", 1, " << *P_D_A1_Coeffs << ") returned false" << endl;
        delete(P_D_A1_Coeffs);
        return false;
      }
      D_A1_X.resize(3,2);
      D_A1_X(0) = D_BinStart_X;
      D_A1_X(1) = D_BinStart_X + ((D_BinEnd_X - D_BinStart_X)/2.);
      D_A1_X(2) = D_BinEnd_X;
      blitz::Array<double, 1> D_A1_YFit = pfsDRPStella::math::Poly(D_A1_X, *P_D_A1_Coeffs);

      /// Calculate Integral
      D_Integral_Out += (D_BinEnd_X - D_BinStart_X) * D_A1_YFit(1);

      #ifdef __DEBUG_INTEGRAL__
        cout << "CFits::IntegralUnderLine: D_Integral_Out = " << D_Integral_Out << endl;
      #endif

      /// clean up
      delete(P_D_A1_Coeffs);
      return true;
    }

    /**
    * Integral-normalise a function
    **/
    bool IntegralNormalise(const blitz::Array<double, 1> &D_A1_XIn,
                           const blitz::Array<double, 1> &D_A1_YIn,
                           blitz::Array<double, 1> &D_A1_YOut)
    {
      if (D_A1_XIn.size() < 2){
        cout << "CFits::IntegralNormalise: ERROR: D_A1_XIn.size() < 2" << endl;
        return false;
      }
      if (D_A1_XIn.size() != D_A1_YIn.size()){
        cout << "CFits::IntegralNormalise: ERROR: D_A1_XIn.size() != D_A1_YIn.size()" << endl;
        return false;
      }
      D_A1_YOut.resize(D_A1_YIn.size());
      double D_Integral;
      blitz::Array<double, 1> D_A1_XInt(2);
      D_A1_XInt(0) = D_A1_XIn(0);
      D_A1_XInt(1) = D_A1_XIn(D_A1_XIn.size()-1);
      #ifdef __DEBUG_INTEGRAL__
        cout << "CFits::IntegralNormalise: D_A1_XIn = " << D_A1_XIn << endl;
        cout << "CFits::IntegralNormalise: D_A1_YIn = " << D_A1_YIn << endl;
        cout << "CFits::IntegralNormalise: D_A1_XInt = " << D_A1_XInt << endl;
      #endif
      if (!pfsDRPStella::math::IntegralUnderCurve(D_A1_XIn, D_A1_YIn, D_A1_XInt, D_Integral)){
        cout << "CFits::IntegralNormalise: ERROR: IntegralUnderCurve returned FALSE" << endl;
        return false;
      }
      #ifdef __DEBUG_INTEGRAL__
        cout << "CFits::IntegralNormalise: D_Integral = " << D_Integral << endl;
      #endif
      D_A1_YOut = D_A1_YIn / D_Integral;
      #ifdef __DEBUG_INTEGRAL__
        cout << "CFits::IntegralNormalise: D_A1_YIn = " << D_A1_YIn << endl;
        cout << "CFits::IntegralNormalise: D_A1_YOut = " << D_A1_YOut << endl;
      #endif
      return true;
    }

    /**
    * Integral-normalise a function
    **/
    bool IntegralNormalise(const blitz::Array<double, 1> &D_A1_XIn,
                           blitz::Array<double, 1> &D_A1_YInOut)
    {
      blitz::Array<double, 1> D_A1_YTemp(D_A1_YInOut.size());
      #ifdef __DEBUG_INTEGRAL__
        cout << "CFits::IntegralNormalise: D_A1_XIn = " << D_A1_XIn << endl;
        cout << "CFits::IntegralNormalise: D_A1_YInOut = " << D_A1_YInOut << endl;
      #endif
      if (!pfsDRPStella::math::IntegralNormalise(D_A1_XIn, D_A1_YInOut, D_A1_YTemp)){
        cout << "CFits::IntegralNormalise: ERROR: IntegralNormalise returned FALSE" << endl;
        return false;
      }
      D_A1_YInOut = D_A1_YTemp;
      D_A1_YTemp.resize(0);
      return true;
    }

    bool PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                 const blitz::Array<double, 1> &D_A1_Y_In,
                 unsigned int I_Degree_In,
                 double D_Reject_In,
                 const blitz::Array<string, 1> &S_A1_Args_In,
                 void *ArgV[],
                 blitz::Array<double, 1>* out){
      return pfsDRPStella::math::PolyFit(D_A1_X_In,
                           D_A1_Y_In,
                           I_Degree_In,
                           D_Reject_In,
                           D_Reject_In,
                           -1,
                           S_A1_Args_In,
                           ArgV,
                           out);
    }

    bool PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                 const blitz::Array<double, 1> &D_A1_Y_In,
                 unsigned int I_Degree_In,
                 double D_LReject_In,
                 double D_UReject_In,
                 unsigned int I_NIter,
                 const blitz::Array<string, 1> &S_A1_Args_In,
                 void *ArgV[],
                 blitz::Array<double, 1>* out){

      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: Starting " << endl;
      #endif

      int I_NReject = 0;
      blitz::Array<double, 1> D_A1_X(D_A1_X_In.size());
      D_A1_X = D_A1_X_In;
      blitz::Array<double, 1> D_A1_Y(D_A1_Y_In.size());
      D_A1_Y = D_A1_Y_In;
      blitz::Array<double, 1> D_A1_X_New(D_A1_X.size());
      blitz::Array<double, 1> D_A1_Y_New(D_A1_Y.size());
      blitz::Array<double, 1> D_A1_MeasureErrors(D_A1_X.size());
      blitz::Array<double, 1> D_A1_MeasureErrors_New(D_A1_X.size());
      blitz::Array<double, 1> *P_D_A1_MeasureErrors;
      int I_DataValues_New = 0;
      int I_NRejected = 0;
      bool B_HaveMeasureErrors = false;
    
      int I_Pos = -1;
      I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "MEASURE_ERRORS");
      if (I_Pos >= 0){
        P_D_A1_MeasureErrors = (blitz::Array<double,1>*)ArgV[I_Pos];
        D_A1_MeasureErrors = *P_D_A1_MeasureErrors;
        B_HaveMeasureErrors = true;
        if (P_D_A1_MeasureErrors->size() != D_A1_X_In.size()){
          cout << "CFits::PolyFit: ERROR: P_D_A1_MeasureErrors->size(=" << P_D_A1_MeasureErrors->size() << ") != D_A1_X_In.size(=" << D_A1_X_In.size() << ")" << endl;
          return false;
        }
      }
      else{
        P_D_A1_MeasureErrors = new blitz::Array<double, 1>(D_A1_X_In.size());
        D_A1_MeasureErrors = sqrt(D_A1_Y_In);
        (*P_D_A1_MeasureErrors) = D_A1_MeasureErrors;
      }

      blitz::Array<int, 1> *P_I_A1_NotRejected = new blitz::Array<int, 1>(1);
      bool B_KeyWordSet_NotRejected = false;
      I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "NOT_REJECTED");
      if (I_Pos >= 0){
        cout << "CFits::PolyFit: Reading KeyWord NOT_REJECTED" << endl;
        cout << "CFits::PolyFit: I_Pos = " << I_Pos << endl;
        P_I_A1_NotRejected = (blitz::Array<int,1>*)(ArgV[I_Pos]);
        cout << "CFits::PolyFit: *P_I_A1_NotRejected = " << *P_I_A1_NotRejected << endl;
        B_KeyWordSet_NotRejected = true;
        cout << "CFits::PolyFit: KeyWord NOT_REJECTED read" << endl;
      }

      blitz::Array<int, 1> *P_I_A1_Rejected = new blitz::Array<int, 1>(1);
      blitz::Array<int, 1> I_A1_Rejected(D_A1_X_In.size());
      I_A1_Rejected = 0;
      bool B_KeyWordSet_Rejected = false;
      I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "REJECTED");
      if (I_Pos >= 0){
        cout << "CFits::PolyFit: Reading KeyWord REJECTED" << endl;
        cout << "CFits::PolyFit: I_Pos = " << I_Pos << endl;
        delete(P_I_A1_Rejected);
        P_I_A1_Rejected = (blitz::Array<int,1>*)(ArgV[I_Pos]);
        cout << "CFits::PolyFit: *P_I_A1_Rejected = " << *P_I_A1_Rejected << endl;
        B_KeyWordSet_Rejected = true;
        cout << "CFits::PolyFit: KeyWord REJECTED read" << endl;
      }

      int *P_I_NRejected = new int(0);
      bool B_KeyWordSet_NRejected = false;
      I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "N_REJECTED");
      if (I_Pos >= 0){
        cout << "CFits::PolyFit: Reading KeyWord N_REJECTED" << endl;
        cout << "CFits::PolyFit: I_Pos = " << I_Pos << endl;
        delete(P_I_NRejected);
        P_I_NRejected = (int*)(ArgV[I_Pos]);
        cout << "CFits::PolyFit: P_I_NRejected = " << *P_I_NRejected << endl;
        B_KeyWordSet_NRejected = true;
        cout << "CFits::PolyFit: KeyWord N_REJECTED read" << endl;
      }

      blitz::Array<int, 1> I_A1_OrigPos(D_A1_X_In.size());
      I_A1_OrigPos = pfsDRPStella::math::IndGenArr(D_A1_X_In.size());
      blitz::Array<double, 1> D_A1_PolyRes;
      int I_NRejected_Old=0;
      blitz::Array<int, 1> I_A1_Rejected_Old(D_A1_X_In.size());
      bool B_Run = true;
      unsigned int i_iter = 0;
      while (B_Run){
        I_A1_Rejected_Old.resize(I_A1_Rejected.size());
        I_A1_Rejected_Old = I_A1_Rejected;
        I_A1_Rejected.resize(D_A1_X_In.size());
        I_NRejected_Old = I_NRejected;
        I_NReject = 0;
        I_NRejected = 0;
        I_DataValues_New = 0;
        if (!pfsDRPStella::math::PolyFit(D_A1_X,
                                         D_A1_Y,
                                         I_Degree_In,
                                         S_A1_Args_In,
                                         ArgV,
                                         out)){
          cout << "CFits::PolyFit: ERROR: PolyFit returned FALSE" << endl;
          if (!B_HaveMeasureErrors)
            delete(P_D_A1_MeasureErrors);
          return false;
        }
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: PolyFit(D_A1_X, D_A1_Y, I_Degree_In, S_A1_Args_In, ArgV, out) returned *out = " << *out << endl;
        #endif
        blitz::Array<double, 1> D_A1_YFit = pfsDRPStella::math::Poly(D_A1_X, *out);
        double D_SDev = sqrt(blitz::sum(blitz::pow2(D_A1_Y - D_A1_YFit) / D_A1_Y.size()));

        D_A1_PolyRes = pfsDRPStella::math::Poly(D_A1_X_In, *out);
        for (unsigned int i_pos=0; i_pos < D_A1_Y_In.size(); i_pos++){
          double D_Dev = D_A1_Y_In(i_pos) - D_A1_PolyRes(i_pos);
          if (((D_Dev < 0) && (D_Dev > (D_LReject_In * D_SDev))) || ((D_Dev >= 0) && (D_Dev < (D_UReject_In * D_SDev)))){
            D_A1_X_New(I_DataValues_New) = D_A1_X_In(i_pos);
            D_A1_Y_New(I_DataValues_New) = D_A1_Y_In(i_pos);
            if (B_HaveMeasureErrors)
              D_A1_MeasureErrors_New(I_DataValues_New) = (*P_D_A1_MeasureErrors)(i_pos);
            I_A1_OrigPos(I_DataValues_New) = D_A1_Y_In(i_pos);
  
            I_DataValues_New++;
          }
          else{
            I_A1_Rejected(I_NRejected) = i_pos;
            cout << "CFits::PolyFit: Rejecting D_A1_X_In(i_pos) = " << D_A1_X_In(i_pos) << endl;
            I_NReject++;
            I_NRejected++;
          }
        }
        D_A1_X.resize(I_DataValues_New);
        D_A1_Y.resize(I_DataValues_New);
        D_A1_X = D_A1_X_New(blitz::Range(0,I_DataValues_New-1));
        D_A1_Y = D_A1_Y_New(blitz::Range(0,I_DataValues_New-1));
        if (B_HaveMeasureErrors){
          D_A1_MeasureErrors.resize(I_DataValues_New);
          D_A1_MeasureErrors = D_A1_MeasureErrors_New(blitz::Range(0,I_DataValues_New-1));
        }
    
        B_Run = false;
        if (I_NRejected != I_NRejected_Old)
          B_Run = true;
        else{
          for (int i_pos=0; i_pos < I_NRejected; i_pos++){
            if (fabs(I_A1_Rejected(i_pos) - I_A1_Rejected_Old(i_pos)) > 0.0001)
              B_Run = true;
          }
        }
        i_iter++;
        if ((I_NIter >= 0) && (i_iter >= I_NIter))
          B_Run = false;
      }
      cout << "CFits::PolyFit: I_NRejected = " << I_NRejected << endl;
    
      cout << "CFits::PolyFit: I_DataValues_New = " << I_DataValues_New << endl;
      blitz::Array<int, 1> I_A1_NotRejected(I_DataValues_New);
      I_A1_NotRejected = I_A1_OrigPos(blitz::Range(0, I_DataValues_New-1));
      if (B_KeyWordSet_NotRejected){
        P_I_A1_NotRejected->resize(I_DataValues_New);
        (*P_I_A1_NotRejected) = I_A1_NotRejected;
        cout << "CFits::PolyFit: *P_I_A1_NotRejected = " << *P_I_A1_NotRejected << endl;
      }
      I_A1_OrigPos.resize(D_A1_X_In.size());
      I_A1_OrigPos = pfsDRPStella::math::IndGenArr(D_A1_X_In.size());
      if (!pfsDRPStella::math::removeSubArrayFromArray(I_A1_OrigPos, I_A1_NotRejected)){
        cout << "CFits::PolyFit: ERROR: Remove_SubArrayFromArray(" << I_A1_OrigPos << ", " << I_A1_NotRejected << ") returned FALSE" << endl;
        if (!B_HaveMeasureErrors)
          delete(P_D_A1_MeasureErrors);
        return false;
      }
      if (B_KeyWordSet_Rejected){
        P_I_A1_Rejected->resize(I_NRejected);
        (*P_I_A1_Rejected) = I_A1_Rejected(blitz::Range(0, I_NRejected-1));
        cout << "CFits::PolyFit: *P_I_A1_Rejected = " << *P_I_A1_Rejected << endl;
      }
      else{
        delete(P_I_A1_Rejected);
      }
      if (!B_KeyWordSet_NotRejected)
        delete(P_I_A1_NotRejected);
      if (B_KeyWordSet_NRejected){
        *P_I_NRejected = I_NRejected;
      }
      else{
        delete(P_I_NRejected);
      }
      if (!B_HaveMeasureErrors)
        delete(P_D_A1_MeasureErrors);
      
      return true;
    }
    
    /** **********************************************************************/
    
    bool PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                 const blitz::Array<double, 1> &D_A1_Y_In,
                 int I_Degree_In,
                 blitz::Array<double, 1>* P_D_A1_Out){
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: Starting " << endl;
      #endif
      blitz::Array<string, 1> S_A1_Args(1);
      S_A1_Args = " ";
      void **PP_Args = (void**)malloc(sizeof(void*));
      if (!pfsDRPStella::math::PolyFit(D_A1_X_In, D_A1_Y_In, I_Degree_In, S_A1_Args, PP_Args, P_D_A1_Out)){
        cout << "CFits::PolyFit: ERROR: PolyFit(" << D_A1_X_In << ", " << D_A1_Y_In << ", " << I_Degree_In << "...) returned FALSE" << endl;
        cout << "CFits::PolyFit: ERROR: PolyFit returned *P_D_A1_Out = " << *P_D_A1_Out << endl;
        free(PP_Args);
        return false;
      }
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: PolyFit returned *P_D_A1_Out = " << *P_D_A1_Out << endl;
      #endif
    //  free(*PP_Args);
      free(PP_Args);
      return true;
    }
    
    

/** 
    CHISQ=double(chisq): out
    COVAR=covar: out
    MEASURE_ERRORS=measure_errors: in
    SIGMA=sigma: out
    STATUS=status: out
    YERROR=yerror
    YFIT=yfit: out
    LSIGMA=lsigma: lower sigma rejection threshold
    USIGMA=usigma:
    ;**/
    bool PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                 const blitz::Array<double, 1> &D_A1_Y_In,
                 int I_Degree_In,
                 const blitz::Array<string, 1> &S_A1_Args_In,
                 void *ArgV[],
                 blitz::Array<double, 1>* P_D_A1_Out){

      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: Starting " << endl;
      #endif
      int I_M = I_Degree_In + 1;
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: I_M set to " << I_M << endl;
      #endif
      if (P_D_A1_Out == NULL)
        P_D_A1_Out = new blitz::Array<double, 1>(1);
      P_D_A1_Out->resize(I_M);
      (*P_D_A1_Out) = 0.;
      int i,j,I_Pos;

      int I_N = D_A1_X_In.size();
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: I_N set to " << I_N << endl;
      #endif

      if (I_N != static_cast<int>(D_A1_Y_In.size())){
        cout << "CFits::PolyFit: ERROR: X and Y must have same number of elements!" << endl;
        return false;
      }

      blitz::Array<double, 1> D_A1_SDev(D_A1_X_In.size());
      D_A1_SDev= 1.;
      blitz::Array<double, 1> D_A1_SDevSquare(D_A1_X_In.size());
      
      bool B_HaveMeasureError = false;
      blitz::Array<double, 1> *P_D_A1_MeasureErrors = new blitz::Array<double, 1>(D_A1_X_In.size());
      *P_D_A1_MeasureErrors = 1.;
      string sTemp = "MEASURE_ERRORS";
      if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
      {
        B_HaveMeasureError = true;
        delete(P_D_A1_MeasureErrors);
        P_D_A1_MeasureErrors = (blitz::Array<double, 1>*)ArgV[I_Pos];
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: B_HaveMeasureError set to TRUE" << endl;
          cout << "CFits::PolyFit: *P_D_A1_MeasureErrors set to " << *P_D_A1_MeasureErrors << endl;
        #endif
      }
      D_A1_SDev = (*P_D_A1_MeasureErrors);
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: D_A1_SDev set to " << D_A1_SDev << endl;
      #endif
      
      D_A1_SDevSquare = pow(D_A1_SDev,2);
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: D_A1_SDevSquare set to " << D_A1_SDevSquare << endl;
      #endif
      blitz::Array<double,1> *P_D_A1_YFit;
      sTemp = "YFIT";
      if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
      {
        P_D_A1_YFit = (blitz::Array<double,1>*)ArgV[I_Pos];
        P_D_A1_YFit->resize(D_A1_X_In.size());
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: KeyWord_Set(YFIT)" << endl;
        #endif
      }
      else{
        P_D_A1_YFit = new blitz::Array<double,1>(D_A1_X_In.size());
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: !KeyWord_Set(YFIT)" << endl;
        #endif
      }
      (*P_D_A1_YFit) = 0.;

      blitz::Array<double,1>* P_D_A1_Sigma = new blitz::Array<double,1>(1);
      sTemp = "SIGMA";
      if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
      {
        delete(P_D_A1_Sigma);
        P_D_A1_Sigma = (blitz::Array<double,1>*)ArgV[I_Pos];
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: KeyWord_Set(SIGMA): *P_D_A1_Sigma set to " << (*P_D_A1_Sigma) << endl;
        #endif
      }

      blitz::Array<double, 2> *P_D_A2_Covar = new blitz::Array<double,2>(I_M,I_M);
      sTemp = "COVAR";
      if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
      {
        delete(P_D_A2_Covar);
        P_D_A2_Covar = (blitz::Array<double,2>*)ArgV[I_Pos];
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: KeyWord_Set(COVAR): *P_D_A2_Covar set to " << (*P_D_A2_Covar) << endl;
        #endif
      }

      blitz::Array<double, 1> D_A1_B(I_M);

      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: D_A1_X_In.size() = " << D_A1_X_In.size() << endl;
      #endif
      blitz::Array<double, 1> D_A1_Z(D_A1_X_In.size());
      D_A1_Z = 1.;
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: D_A1_Z set to " << D_A1_Z << endl;
      #endif

      blitz::Array<double, 1> D_A1_WY(D_A1_Y_In.size());
      D_A1_WY = D_A1_Y_In;
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: D_A1_WY set to " << D_A1_WY << endl;
      #endif

      if (B_HaveMeasureError){
        D_A1_WY = D_A1_WY / D_A1_SDevSquare;
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: B_HaveMeasureError: D_A1_WY set to " << D_A1_WY << endl;
        #endif
      }

      if (B_HaveMeasureError){
        (*P_D_A2_Covar)(0,0) = blitz::sum(1./D_A1_SDevSquare);
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: B_HaveMeasureError: (*P_D_A2_Covar)(0,0) set to " << (*P_D_A2_Covar)(0,0) << endl;
        #endif
      }
      else{
        (*P_D_A2_Covar)(0,0) = I_N;
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: !B_HaveMeasureError: (*P_D_A2_Covar)(0,0) set to " << (*P_D_A2_Covar)(0,0) << endl;
        #endif
      }

      D_A1_B(0) = blitz::sum(D_A1_WY);
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: D_A1_B(0) set to " << D_A1_B(0) << endl;
      #endif

      double D_Sum;
      for (int p = 1; p <= 2 * I_Degree_In; p++){
        D_A1_Z *= D_A1_X_In;
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: for(p(=" << p << ")...): D_A1_Z set to " << D_A1_Z << endl;
        #endif
        if (p < I_M){
          D_A1_B(p) = blitz::sum(D_A1_WY * D_A1_Z);
          #ifdef __DEBUG_POLYFIT__
            cout << "CFits::PolyFit: for(p(=" << p << ")...): p < I_M(=" << I_M << "): D_A1_B(p) set to " << D_A1_B(p) << endl;
          #endif
        }
        if (B_HaveMeasureError){
          D_Sum = blitz::sum(D_A1_Z / D_A1_SDevSquare);
          #ifdef __DEBUG_POLYFIT__
            cout << "CFits::PolyFit: for(p(=" << p << ")...): B_HaveMeasureError: D_Sum set to " << D_Sum << endl;
          #endif
        }
        else{
          D_Sum = blitz::sum(D_A1_Z);
          #ifdef __DEBUG_POLYFIT__
            cout << "CFits::PolyFit: for(p(=" << p << ")...): !B_HaveMeasureError: D_Sum set to " << D_Sum << endl;
          #endif
        }
        if (p-I_Degree_In > 0){
          i = p-I_Degree_In;
        }
        else{
          i = 0;
        }
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: for(p(=" << p << ")...): i set to " << i << endl;
        #endif
        for (j = i; j <= I_Degree_In; j++){
          (*P_D_A2_Covar)(j,p-j) = D_Sum;
          #ifdef __DEBUG_POLYFIT__
            cout << "CFits::PolyFit: for(p(=" << p << ")...): for(j(=" << j << ")...): (*P_D_A2_Covar)(j,p-j=" << p-j << ") set to " << (*P_D_A2_Covar)(j,p-j) << endl;
          #endif
        }
      }

      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: before InvertGaussJ: (*P_D_A2_Covar) = " << (*P_D_A2_Covar) << endl;
      #endif
      if (!pfsDRPStella::math::InvertGaussJ(*P_D_A2_Covar)){
        cout << "CFits::PolyFit: ERROR! InvertGaussJ(*P_D_A2_Covar=" << *P_D_A2_Covar << ") returned false!" << endl;
        return false;
      }
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: InvertGaussJ: (*P_D_A2_Covar) set to " << (*P_D_A2_Covar) << endl;
        cout << "CFits::PolyFit: MatrixTimesVecArr: P_D_A2_Covar->rows() = " << P_D_A2_Covar->rows() << endl;
        cout << "CFits::PolyFit: MatrixTimesVecArr: P_D_A2_Covar->cols() = " << P_D_A2_Covar->cols() << endl;
        cout << "CFits::PolyFit: MatrixTimesVecArr: (*P_D_A2_Covar) = " << (*P_D_A2_Covar) << endl;
        cout << "CFits::PolyFit: MatrixTimesVecArr: D_A1_B = " << D_A1_B.size() << ": " << D_A1_B << endl;
      #endif
      blitz::Array<double,1> *P_D_A1_TempA = MatrixTimesVecArr(*P_D_A2_Covar, D_A1_B);
      P_D_A1_Out->resize(P_D_A1_TempA->size());
      (*P_D_A1_Out) = (*P_D_A1_TempA);
      delete(P_D_A1_TempA);
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: MatrixTimesVecArr: P_D_A1_YFit->size() = " << P_D_A1_YFit->size() << ": (*P_D_A1_Out) set to " << (*P_D_A1_Out) << endl;
      #endif

      (*P_D_A1_YFit) = (*P_D_A1_Out)(I_Degree_In);
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: InvertGaussJ: (*P_D_A1_YFit) set to " << (*P_D_A1_YFit) << endl;
      #endif

      for (int k=I_Degree_In-1; k >= 0; k--){
        (*P_D_A1_YFit) = (*P_D_A1_Out)(k) + (*P_D_A1_YFit) * D_A1_X_In;
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: for(k(=" << k << ")...): (*P_D_A1_YFit) set to " << (*P_D_A1_YFit) << endl;
        #endif
      }

      P_D_A1_Sigma->resize(I_M);
      for (int k=0;k < I_M; k++){
        (*P_D_A1_Sigma)(k) = (*P_D_A2_Covar)(k,k);
      }
      (*P_D_A1_Sigma) = sqrt(abs(*P_D_A1_Sigma));
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: (*P_D_A1_Sigma) set to " << (*P_D_A1_Sigma) << endl;
      #endif

      double D_ChiSq = 0.;
      if (B_HaveMeasureError){
        blitz::Array<double,1> D_A1_Diff(D_A1_Y_In.size());
        D_A1_Diff = pow(D_A1_Y_In - (*P_D_A1_YFit),2);
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: B_HaveMeasureError: D_A1_Diff set to " << D_A1_Diff << endl;
        #endif

        D_ChiSq = blitz::sum(D_A1_Diff / D_A1_SDevSquare);
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: B_HaveMeasureError: D_ChiSq set to " << D_ChiSq << endl;
        #endif

      }
      else{
        D_ChiSq = blitz::sum(pow(D_A1_Y_In - (*P_D_A1_YFit),2));
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: !B_HaveMeasureError: D_ChiSq set to " << D_ChiSq << endl;
      #endif

      (*P_D_A1_Sigma) *= sqrt(D_ChiSq / (I_N - I_M));
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: !B_HaveMeasureError: (*P_D_A1_Sigma) set to " << (*P_D_A1_Sigma) << endl;
      #endif
      }
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: returning *P_D_A1_Out = " << (*P_D_A1_Out) << endl;
      #endif

      sTemp = "YFIT";
      if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) < 0)
      {
        delete(P_D_A1_YFit);
      }
      sTemp = "SIGMA";
      if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) < 0)
      {
        delete(P_D_A1_Sigma);
      }
      sTemp = "COVAR";
      if ((I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) < 0)
      {
        delete(P_D_A2_Covar);
      }
      if (!B_HaveMeasureError)
        delete(P_D_A1_MeasureErrors);
      return true;
    }
    
    /** **********************************************************************/

    bool PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                 const blitz::Array<double, 1> &D_A1_Y_In,
                 unsigned int I_Degree_In,
                 double D_Reject_In,
                 blitz::Array<double, 1>* P_D_A1_Out){
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: Starting " << endl;
      #endif
      blitz::Array<string, 1> S_A1_Args(1);
      S_A1_Args = " ";
      void **PP_Args = (void**)malloc(sizeof(void*) * 1);
      if (!pfsDRPStella::math::PolyFit(D_A1_X_In, 
                         D_A1_Y_In, 
                         I_Degree_In, 
                         D_Reject_In, 
                         S_A1_Args, 
                         PP_Args, 
                         P_D_A1_Out)){
        cout << "CFits::PolyFit: ERROR: PolyFit(" << D_A1_X_In << ", " << D_A1_Y_In << ", " << I_Degree_In << "...) returned FALSE" << endl;
        cout << "CFits::PolyFit: ERROR: PolyFit returned *P_D_A1_Out = " << *P_D_A1_Out << endl;
        free(PP_Args);
        return false;
      }
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: PolyFit returned *P_D_A1_Out = " << *P_D_A1_Out << endl;
      #endif
      free(PP_Args);
      return true;
    }

    /** **********************************************************************/

    bool PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                 const blitz::Array<double, 1> &D_A1_Y_In,
                 unsigned int I_Degree_In,
                 double D_LReject_In,
                 double D_HReject_In,
                 unsigned int I_NIter,
                 blitz::Array<double, 1>* P_D_A1_Out){
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: Starting " << endl;
      #endif
      blitz::Array<string, 1> S_A1_Args;
      S_A1_Args = " ";
      void **PP_Args = (void**)malloc(sizeof(void*) * 1);
      if (!pfsDRPStella::math::PolyFit(D_A1_X_In, 
                         D_A1_Y_In, 
                         I_Degree_In, 
                         D_LReject_In, 
                         D_HReject_In, 
                         I_NIter,
                         S_A1_Args, 
                         PP_Args, 
                         P_D_A1_Out)){
        cout << "CFits::PolyFit: ERROR: PolyFit(" << D_A1_X_In << ", " << D_A1_Y_In << ", " << I_Degree_In << "...) returned FALSE" << endl;
        cout << "CFits::PolyFit: ERROR: PolyFit returned *P_D_A1_Out = " << *P_D_A1_Out << endl;
        free(PP_Args);
        return false;
      }
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: PolyFit returned *P_D_A1_Out = " << *P_D_A1_Out << endl;
      #endif
      free(PP_Args);
      return true;
    }

    blitz::Array<int, 1> IndGenArr(int len){
      blitz::Array<int, 1> I_A1_Result(len);
      blitz::firstIndex i;
      I_A1_Result = i;
      #ifdef __DEBUG_INDGENARR__
        cout << "CFits::IndGenArr: len = " << len << ": (*P_I_A1_Result) set to " << *P_I_A1_Result << endl;
      #endif
      return I_A1_Result;
    }
    
    blitz::Array<float, 1> FIndGenArr(int len){
      blitz::Array<float, 1> F_A1_Return(len);
      blitz::firstIndex i;
      F_A1_Return = i;
//      for (int i=0; i<len; i++)
//        (*P_F_A1_Return)(i) = float(i);   // [ -3 -2 -1  0  1  2  3 ]
      return (F_A1_Return);
    }
    
    blitz::Array<double, 1> DIndGenArr(int len){
      blitz::Array<double, 1> D_A1_Return(len);
      blitz::firstIndex i;
      D_A1_Return = i;
      return D_A1_Return;
    }
    
    blitz::Array<long, 1> LIndGenArr(int len){
      blitz::Array<long, 1> L_A1_Return(len);
      blitz::firstIndex i;
      L_A1_Return = i;
      return L_A1_Return;
    }
    
    bool removeSubArrayFromArray(blitz::Array<int, 1> &A1_Array_InOut, 
                                  const blitz::Array<int, 1> &A1_SubArray){
      blitz::Array<int, 1> A1_Array_Out(A1_Array_InOut.size());
      int I_NElements = 0;
      bool B_InSubArray = false;
      for (unsigned int i_orig=0; i_orig<A1_Array_InOut.size(); i_orig++){
        B_InSubArray = false;
        for (unsigned int i_sub=0; i_sub<A1_SubArray.size(); i_sub++){
          if (A1_Array_InOut(i_orig) == A1_SubArray(i_sub))
            B_InSubArray = true;
        }
        if (!B_InSubArray){
          A1_Array_Out(I_NElements) = A1_Array_InOut(i_orig);
          I_NElements++;
        }
      }
      A1_Array_InOut.resize(I_NElements);
      A1_Array_InOut = A1_Array_Out(blitz::Range(0, I_NElements-1));
      return true;
    }
    
    /**
     *     InterPol linear, not regular
     **/
    bool InterPol(const blitz::Array<double, 1> &v_In,
                  const blitz::Array<double, 1> &x_In,
                  const blitz::Array<double, 1> &u_In,
                  blitz::Array<double, 1> &y_Out){
      return pfsDRPStella::math::InterPol(v_In, x_In, u_In, y_Out, false);
    }
    
    bool InterPol(const blitz::Array<double, 1> &v_In,
                  const blitz::Array<double, 1> &x_In,
                  const blitz::Array<double, 1> &u_In,
                  blitz::Array<double, 1> &y_Out,
                  bool preserveFlux){
      blitz::Array<string, 1> s_a1(1);
      s_a1 = " ";
      y_Out.resize(u_In.size());
      if (preserveFlux){
        blitz::Array<double, 1> D_A1_U(2);
        blitz::Array<double, 1> D_A1_X(x_In.size() + 1);
        D_A1_X(0) = x_In(0) - ((x_In(1) - x_In(0))/2.);
        D_A1_X(D_A1_X.size()-1) = x_In(x_In.size()-1) + ((x_In(x_In.size()-1) - x_In(x_In.size()-2))/2.);
        for (unsigned int i_pix=1; i_pix<x_In.size(); i_pix++){
          D_A1_X(i_pix) = x_In(i_pix-1) + ((x_In(i_pix) - x_In(i_pix-1))/2.);
        }
        #ifdef __DEBUG_INTERPOL__
          cout << "CFits::InterPol: x_In = " << x_In << endl;
          cout << "CFits::InterPol: D_A1_X = " << D_A1_X << endl;
        #endif
        
        blitz::Array<int, 1> I_A1_Ind(D_A1_X.size());
        blitz::Array<int, 1> *P_I_A1_Ind;
        int I_Start = 0;
        int I_NInd = 0;
        double D_Start, D_End;
        for (unsigned int i_pix=0; i_pix<u_In.size(); i_pix++){
          if (i_pix == 0){
            D_A1_U(0) = u_In(0) - ((u_In(1) - u_In(0)) / 2.);
            D_A1_U(1) = u_In(0) + ((u_In(1) - u_In(0)) / 2.);
          }
          else if (i_pix == u_In.size()-1){
            D_A1_U(0) = u_In(u_In.size()-1) - ((u_In(u_In.size()-1) - u_In(u_In.size()-2)) / 2.);
            D_A1_U(1) = u_In(u_In.size()-1) + ((u_In(u_In.size()-1) - u_In(u_In.size()-2)) / 2.);
          }
          else{
            D_A1_U(0) = u_In(i_pix) - ((u_In(i_pix) - u_In(i_pix-1)) / 2.);
            D_A1_U(1) = u_In(i_pix) + ((u_In(i_pix+1) - u_In(i_pix)) / 2.);
          }
          I_A1_Ind = blitz::where(D_A1_X < D_A1_U(0), 1, 0);
          P_I_A1_Ind = pfsDRPStella::math::GetIndex(I_A1_Ind, I_NInd);
          if (I_NInd < 1){
            #ifdef __DEBUG_INTERPOL__
              cout << "CFits::InterPol: WARNING: 1. I_A1_Ind = " << I_A1_Ind << ": I_NInd < 1" << endl;
            #endif
            I_Start = 0;
          }
          else{
            I_Start = (*P_I_A1_Ind)(P_I_A1_Ind->size()-1);
          }
          #ifdef __DEBUG_INTERPOL__
            cout << "CFits::InterPol: i_pix = " << i_pix << ": D_A1_U = " << D_A1_U << endl;
          #endif
          delete(P_I_A1_Ind);
          I_A1_Ind = blitz::where(D_A1_X > D_A1_U(1), 1, 0);
          P_I_A1_Ind = pfsDRPStella::math::GetIndex(I_A1_Ind, I_NInd);
          #ifdef __DEBUG_INTERPOL__
            int I_End = 0;
            if (I_NInd < 1){
              cout << "CFits::InterPol: WARNING: 2. I_A1_Ind = " << I_A1_Ind << ": I_NInd < 1" << endl;
              I_End = D_A1_X.size()-1;
            }
            else{
              I_End = (*P_I_A1_Ind)(0);
            }
            cout << "CFits::InterPol: i_pix = " << i_pix << ": D_A1_X(" << I_Start << ":" << I_End << ") = " << D_A1_X(blitz::Range(I_Start, I_End)) << endl;
          #endif
          delete(P_I_A1_Ind);
          
          D_Start = D_A1_U(0);
          if (D_A1_X(I_Start) > D_A1_U(0))
            D_Start = D_A1_X(I_Start);
          y_Out(i_pix) = 0.;
          if ((D_A1_U(1) > D_A1_X(0)) && (D_A1_U(0) < D_A1_X(D_A1_X.size()-1))){
            do {
              if (D_A1_U(1) < D_A1_X(I_Start + 1)){
                D_End = D_A1_U(1);
              }
              else{
                D_End = D_A1_X(I_Start + 1);
              }
              #ifdef __DEBUG_INTERPOL__
                cout << "CFits::InterPol: i_pix = " << i_pix << ": I_Start = " << I_Start << ", I_End = " << I_End << endl;
                cout << "CFits::InterPol: i_pix = " << i_pix << ": D_Start = " << D_Start << ", D_End = " << D_End << endl;
              #endif
              y_Out(i_pix) += v_In(I_Start) * (D_End - D_Start) / (D_A1_X(I_Start + 1) - D_A1_X(I_Start));
              D_Start = D_End;
              if (D_A1_U(1) >= D_A1_X(I_Start + 1))
                I_Start++;
              #ifdef __DEBUG_INTERPOL__
                cout << "CFits::InterPol: i_pix = " << i_pix << ": y_Out(" << i_pix << ") = " << y_Out(i_pix) << endl;
              #endif
              if (I_Start + 1 >= static_cast<int>(D_A1_X.size()))
                break;
            } while (D_End < D_A1_U(1)-((D_A1_U(1) - D_A1_U(0)) / 100000000.));
          }
        }
        return true;
      }
      
      if (!pfsDRPStella::math::InterPol(v_In, x_In, u_In, s_a1, y_Out)){
        cout << "CFits::InterPol: ERROR: InterPol returned FALSE" << endl;
        return false;
      }
      #ifdef __DEBUG_INTERPOL__
        cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, P_A1_Out): Ready " << endl;
      #endif
      
      s_a1.resize(0);
      
      return true;
    }
    
    /**
     *      InterPol
     *       The InterPol function performs linear, quadratic, or spline interpolation on vectors with an irregular grid.
     **/
    bool InterPol(const blitz::Array<double, 1> &v_In,
                  const blitz::Array<double, 1> &x_In,
                  const blitz::Array<double, 1> &u_In,
                  const blitz::Array<string, 1> &keyWords_In,
                  blitz::Array<double,1> &y_Out){
      #ifdef __DEBUG_INTERPOL__
        cout << "CFits::InterPol: v_In.size() = " << v_In.size() << endl;
        cout << "CFits::InterPol: x_In.size() = " << x_In.size() << endl;
        cout << "CFits::InterPol: u_In.size() = " << u_In.size() << endl;
        cout << "CFits::InterPol: keyWords_In.size() = " << keyWords_In.size() << endl;
      #endif
      
      #ifdef __DEBUG_INTERPOL__
        cout << "CFits::InterPol(v_In = " << v_In << ", x_In = " << x_In << ", u_In = " << u_In << ", keyWords_In) Started" << endl;
      #endif
      
      int M = v_In.size();
      #ifdef __DEBUG_INTERPOL__
      cout << "CFits::InterPol(v_In, x_In, u_In, keyWords_In): M set to " << M << endl;
      #endif
      blitz::firstIndex i;
      
      if (static_cast<int>(x_In.size()) != M)
      {
        cout << "CFits::InterPol: ERROR: x_In and v_In must have same # of elements!" << endl;
        return false;
      }
      blitz::Array<int, 1> *p_SVecArr = pfsDRPStella::math::valueLocate(x_In, u_In);
      #ifdef __DEBUG_INTERPOL__
        cout << "CFits::InterPol(v_In, x_In, u_In, keyWords_In): SVecArr set to " << *p_SVecArr << endl;
      #endif
      blitz::Array<int, 1> SVecArr(p_SVecArr->size());
      SVecArr = (*p_SVecArr);
      delete p_SVecArr;
      SVecArr = blitz::where(SVecArr < 0, 0, SVecArr);
      #ifdef __DEBUG_INTERPOL__
        cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, keyWords_In): SVecArr set to " << SVecArr << endl;
      #endif
      
      SVecArr = blitz::where(SVecArr > M-2, M-2, SVecArr);
      #ifdef __DEBUG_INTERPOL__
        cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, keyWords_In): SVecArr set to " << SVecArr << endl;
      #endif
      
      #ifdef __DEBUG_INTERPOL__
        cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, keyWords_In): Starting HInterPol " << endl;
      #endif
      //  blitz::Array<double, 1> *P_ResultVecArr;
      if (!pfsDRPStella::math::HInterPol(v_In, x_In, SVecArr, u_In, keyWords_In, y_Out)){
        cout << "CFits::InterPol: ERROR: HInterPol returned FALSE" << endl;
        return false;
      }
      
      SVecArr.resize(0);
      #ifdef __DEBUG_INTERPOL__
        cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, keyWords_In): Ready " << endl;
      #endif
      
      return true;
    }
    
    /**
     *      InterPol
     *       This function performs linear, quadratic, or spline interpolation on vectors with a regular grid.
     **
    bool InterPol(blitz::Array<double, 1> &v_In,
                  long n_In,
                  const blitz::Array<string, 1> &keyWords_In,
                  blitz::Array<double,1> &y_Out){
      int M = v_In.size();
      # ifdef __DEBUG_INTERPOL__
      cout << "CFits::InterPol: M set to " << M << endl;
      #endif
      blitz::firstIndex i;
      blitz::Array<double, 1> UVecArr(n_In);       /// Grid points
      UVecArr = i;
      double divisor = n_In - 1.0;
      # ifdef __DEBUG_INTERPOL__
        cout << "CFits::InterPol: divisor set to " << divisor << endl;
      #endif
      blitz::Array<double, 1> RVecArr(n_In);
      RVecArr = i;
      blitz::Array<double, 1> DifVecArr(v_In.size() - 1);
      DifVecArr = 0.;
      blitz::Array<double, 1> DA1_VTemp(RVecArr.size());
      DA1_VTemp = 0.;
      blitz::Array<double, 1> DA1_DifTemp(RVecArr.size());
      DA1_DifTemp = 0.;
      double n = (double)n_In;
      blitz::Array<double,1> *P_D_A1_TempB;
      
      if (pfsDRPStella::utils::KeyWord_Set(S_A1_In, "LSQUADRATIC") < 0
        && pfsDRPStella::utils::KeyWord_Set(S_A1_In, "QUADRATIC") < 0
        && pfsDRPStella::utils::KeyWord_Set(S_A1_In, "SPLINE") < 0)
      {
        if (n < 2.0)
          n = 1.0;
        RVecArr *= (M - 1.0) / ((n - 1.0));  /// Grid points in v_In
        # ifdef __DEBUG_INTERPOL__
          cout << "CFits::InterPol: RVecArr set to " << RVecArr << endl;
        #endif
        
        blitz::Array<int, 1> RLVecArr(RVecArr.size());
        RLVecArr = Int(RVecArr);   // Conversion to Integer
        DifVecArr(blitz::Range::all()) = v_In(blitz::Range(1, v_In.size() - 1)) - v_In(blitz::Range(1, v_In.size() - 1));
        
        /// Interpolate
        pfsDRPStella::math::GetSubArrCopy(v_In,
                            *p_RLVecArr,
                            DA1_VTemp);
        pfsDRPStella::math::GetSubArrCopy(DifVecArr,
                            *p_RLVecArr,
                            DA1_DifTemp);
        
        P_D_A1_TempB = new blitz::Array<double, 1>(DA1_VTemp + (RVecArr - (*p_RLVecArr)) * DA1_DifTemp);
        P_D_A1_Out->resize(P_D_A1_TempB->size());
        (*P_D_A1_Out) = (*P_D_A1_TempB);
        delete(P_D_A1_TempB);
        # ifdef __DEBUG_INTERPOL__
        cout << "CFits::InterPol: *P_D_A1_TempB set to " << *P_D_A1_TempB << endl;
        #endif
        
        UVecArr.resize(0);       /// Grid points
        RVecArr.resize(0);
        RLVecArr.resize(0);
        DifVecArr.resize(0);
        DA1_VTemp.resize(0);
        DA1_DifTemp.resize(0);
        
        return true;
      }
      if (divisor < 1.0)
        divisor = 1.0;
      UVecArr *= ((M - 1.0) / divisor);
      # ifdef __DEBUG_INTERPOL__
      cout << "CFits::InterPol: UVecArr set to " << UVecArr << endl;
      #endif
      SVecArr = Int(UVecArr);   /// Subscripts
      blitz::Array<double, 1> XVecArr(1);
      XVecArr = n_In;
      
      //  blitz::Array<double, 1> *P_PResultArr;
      if (!HInterPol(v_In, XVecArr, SVecArr, UVecArr, S_A1_In, P_D_A1_Out)){
        cout << "CFits::InterPol: ERROR: HInterPol returned FALSE" << endl;
        return false;
      }
      UVecArr.resize(0);       /// Grid points
      RVecArr.resize(0);
      SVecArr.resize(0);
      DifVecArr.resize(0);
      DA1_VTemp.resize(0);
      DA1_DifTemp.resize(0);
      XVecArr.resize(0);
      
      return true;
    }*/
    
    /**
     *      HInterPol
     *      Help function for InterPol methods
     **/
    bool HInterPol(const blitz::Array<double, 1> &v_In,
                   const blitz::Array<double, 1> &x_In,
                   blitz::Array<int, 1> &s_InOut,
                   const blitz::Array<double, 1> &u_In,
                   const blitz::Array<string, 1> &keyWords_In,
                   blitz::Array<double,1> &y_Out){
      #ifdef __DEBUG_INTERPOL__
      cout << "CFits::HInterPol: v_In.size() = " << v_In.size() << endl;
      cout << "CFits::HInterPol: x_In.size() = " << x_In.size() << endl;
      cout << "CFits::HInterPol: s_InOut.size() = " << s_InOut.size() << endl;
      cout << "CFits::HInterPol: u_In.size() = " << u_In.size() << endl;
      cout << "CFits::HInterPol: keyWords_In.size() = " << keyWords_In.size() << endl;
      #endif
      
      int M = v_In.size();
      blitz::firstIndex i;
      
      blitz::Array<int, 1> IA1_Temp(s_InOut.size());
      IA1_Temp = 0;
      
      blitz::Array<double, 1> DA1_Temp(s_InOut.size());
      DA1_Temp = 0.;
      
      blitz::Array<double, 1> DA1_TempA(s_InOut.size());
      DA1_TempA = 0.;
      
      blitz::Array<double, 1> DA1_VTempP1(s_InOut.size());
      DA1_VTempP1 = 0.;
      
      blitz::Array<double, 1> DA1_VTemp(s_InOut.size());
      DA1_VTemp = 0.;
      
      blitz::Array<double, 1> DA1_XTempP1(s_InOut.size());
      DA1_XTempP1 = 0.;
      
      blitz::Array<double, 1> DA1_XTemp(s_InOut.size());
      DA1_XTemp = 0.;
      
      blitz::Array<int, 1> IA1_STemp(s_InOut.size());
      IA1_STemp = 0;
      
      blitz::Array<double, 1> PVecArr(s_InOut.size());
      PVecArr = 0.;
      
      blitz::Array<double, 1> TmpVecArr(4);
      TmpVecArr = i;
      
      blitz::Array<double, 1> T1VecArr(4);
      T1VecArr = 0.;
      
      blitz::Array<double, 1> T2VecArr(4);
      T2VecArr = 0.;
      
      blitz::Array<double, 1> X1VecArr(s_InOut.size());
      X1VecArr = 0.;
      
      blitz::Array<double, 1> X0VecArr(s_InOut.size());
      X0VecArr = 0.;
      
      blitz::Array<double, 1> X2VecArr(s_InOut.size());
      X2VecArr = 0.;
      
      blitz::Array<double, 1> X0Arr(4);
      X0Arr = 0.;
      
      blitz::Array<double, 1> V0Arr(4);
      V0Arr = 0.;
      
      blitz::Array<double, 1> QArr(s_InOut.size());
      QArr = 0.;
      
      int s0int;
      double s0;
      /// Least square fit quadratic, 4 points
      if (pfsDRPStella::utils::KeyWord_Set(keyWords_In, "LSQUADRATIC") >= 0)
      {
        # ifdef __DEBUG_INTERPOL__
          cout << "CFits::HInterPol: KeywordSet(LSQUADRATIC)" << endl;
        #endif
        s_InOut = blitz::where(s_InOut < 1, 1, s_InOut);
        s_InOut = blitz::where(s_InOut > M-3, M-3, s_InOut);
        # ifdef __DEBUG_INTERPOL__
          cout << "CFits::HInterPol: LSQUADRATIC: s_InOut.size() set to " << s_InOut.size() << endl;
        #endif
        PVecArr = v_In(0);   /// Result
        for (unsigned int m = 0; m < s_InOut.size(); m++)
        {
          s0 = double(s_InOut(m)) - 1.;
          s0int = (int)s0;
          TmpVecArr += s0;
          T1VecArr = x_In(blitz::Range(s0int, (s0int)+3));
          T2VecArr = v_In(blitz::Range(s0int, (s0int)+3));
          #ifdef __DEBUG_INTERPOL__
            cout << "CFits::HInterPol: Starting LsToFit(T1VecArr, T2VecArr, u_In(m)" << endl;
          #endif
          if (!pfsDRPStella::math::LsToFit(*(const_cast<const blitz::Array<double, 1>*>(&T1VecArr)), 
                       *(const_cast<const blitz::Array<double, 1>*>(&T2VecArr)), 
                       u_In(m), 
                       PVecArr(m)))
            return false;
        }
      }
      else if (pfsDRPStella::utils::KeyWord_Set(keyWords_In, "QUADRATIC") >= 0)
      {
        # ifdef __DEBUG_INTERPOL__
          cout << "CFits::HInterPol: KeywordSet(QUADRATIC)" << endl;
        #endif
        s_InOut = blitz::where(s_InOut < 1, 1, s_InOut);
        s_InOut = blitz::where(s_InOut > M-2, M-2, s_InOut);
        # ifdef __DEBUG_INTERPOL__
          cout << "CFits::HInterPol: QUADRATIC: s_InOut.size() set to " << s_InOut.size() << endl;
        #endif
        
        if (!pfsDRPStella::math::GetSubArrCopy(x_In,
                                 s_InOut,
                                 X1VecArr)){
          cout << "CFits::HInterPol: ERROR: GetSubArrCopy(x_In, s_InOut, X1VecArr) returned FALSE" << endl;
          return false;
        }
          
        IA1_Temp = s_InOut - 1;
          
        if (!pfsDRPStella::math::GetSubArrCopy(x_In,
                                 IA1_Temp,
                                 X0VecArr)){
          cout << "CFits::HInterPol: ERROR: GetSubArrCopy(x_In, IA1_Temp, X0VecArr) returned FALSE" << endl;
          return false;
        }
            
        IA1_Temp = s_InOut + 1;
        if (!pfsDRPStella::math::GetSubArrCopy(x_In,
                                 IA1_Temp,
                                 X2VecArr)){
          cout << "CFits::HInterPol: ERROR: GetSubArrCopy(x_In, IA1_Temp, X2VecArr) returned FALSE" << endl;
          return false;
        }
              
        IA1_Temp = s_InOut - 1;
        if (!pfsDRPStella::math::GetSubArrCopy(v_In,
                                 IA1_Temp,
                                 DA1_Temp)){
          cout << "CFits::HInterPol: ERROR: GetSubArrCopy(v_In, IA1_Temp, DA1_Temp) returned FALSE" << endl;
          return false;
        }
        IA1_Temp = s_InOut + 1;
        if (!pfsDRPStella::math::GetSubArrCopy(v_In,
                                 IA1_Temp,
                                 DA1_TempA)){
          cout << "CFits::HInterPol: ERROR: GetSubArrCopy(v_In, IA1_Temp, DA1_TempA) returned FALSE" << endl;
          return false;
        }
        PVecArr = DA1_Temp
                  * (u_In - X1VecArr) * (u_In - X2VecArr)
                  / ((X0VecArr - X1VecArr) * (X0VecArr - X2VecArr))
                  + DA1_TempA
                  * (u_In - X0VecArr) * (u_In - X1VecArr)
                  / ((X2VecArr - X0VecArr) * (X2VecArr - X1VecArr));
      }
      else if (pfsDRPStella::utils::KeyWord_Set(keyWords_In, "SPLINE") >= 0){
        # ifdef __DEBUG_INTERPOL__
          cout << "CFits::HInterPol: KeywordSet(SPLINE)" << endl;
        #endif
        s_InOut = blitz::where(s_InOut < 1, 1, s_InOut);
        s_InOut = blitz::where(s_InOut > M-3, M-3, s_InOut);
        # ifdef __DEBUG_INTERPOL__
          cout << "CFits::HInterPol: SPLINE: s_InOut.size() set to " << s_InOut.size() << endl;
        #endif
        PVecArr.resize(s_InOut.size());
        PVecArr = v_In(0);
        int SOld = -1;
        for (unsigned int m = 0; m < s_InOut.size(); m++){
          s0 = s_InOut(m) - 1.;
          s0int = (int)s0;
          if (abs(SOld - s0int) > 0){
            X0Arr.resize(4);
            X0Arr = x_In(blitz::Range(s0int, (s0int)+3));
            V0Arr.resize(4);
            V0Arr(blitz::Range::all()) = x_In(blitz::Range(s0int, (s0int)+3));
            if (!pfsDRPStella::math::Spline(X0Arr, V0Arr, QArr)){
              cout << "CFits::HInterPol: ERROR: Spline(X0Arr, V0Arr, QArr) returned FALSE" << endl;
              return false;
            }
            SOld = s0int;
          }
          if (!pfsDRPStella::math::SplInt(X0Arr, V0Arr, QArr, u_In(m), &(PVecArr(m)))){
            cout << "CFits::HInterPol: ERROR: SplInt(X0Arr, V0Arr, QArr, u_In(m), PVecArr(m)) returned FALSE" << endl;
            return false;
          }
        }
      }
      /*
       *  ELSE: $              ;Linear, not regular
       *  p = (u-x[s])*(v[s+1]-v[s])/(x[s+1] - x[s]) + v[s]
       *  ENDCASE
       * 
       *  RETURN, p
       *  end
       */
      else  /// Linear, not regular
      {
        ///    p = (u-x[s])*(v[s+1]-v[s])/(x[s+1] - x[s]) + v[s]
        
        if (!pfsDRPStella::math::GetSubArrCopy(x_In,
          s_InOut,
          DA1_XTemp)){
          cout << "CFits::HInterPol: ERROR: GetSubArrCopy(x_In, s_InOut, DA1_XTemp) returned FALSE" << endl;
        return false;
          }
          # ifdef __DEBUG_INTERPOL__
          cout << "CFits::HInterPol: DA1_XTemp set to " << DA1_XTemp << endl;
          #endif
          if (!pfsDRPStella::math::GetSubArrCopy(v_In,
            s_InOut,
            DA1_VTemp)){
            cout << "CFits::HInterPol: ERROR: GetSubArrCopy(v_In, s_InOut, DA1_VTemp) returned FALSE" << endl;
          return false;
            }
            # ifdef __DEBUG_INTERPOL__
            cout << "CFits::HInterPol: DA1_VTemp set to " << DA1_VTemp << endl;
            #endif
            
            IA1_STemp = s_InOut + 1;
            # ifdef __DEBUG_INTERPOL__
            cout << "CFits::HInterPol: IA1_STemp set to " << IA1_STemp << endl;
            #endif
            
            if (!pfsDRPStella::math::GetSubArrCopy(x_In,
              IA1_STemp,
              DA1_XTempP1)){
              cout << "CFits::HInterPol: ERROR: GetSubArrCopy(x_In, IA1_STemp, DA1_XTempP1) returned FALSE" << endl;
            return false;
              }
              # ifdef __DEBUG_INTERPOL__
              cout << "CFits::HInterPol: DA1_XTempP1 set to " << DA1_XTempP1 << endl;
              #endif
              
              if (!pfsDRPStella::math::GetSubArrCopy(v_In,
                IA1_STemp,
                DA1_VTempP1)){
                cout << "CFits::HInterPol: ERROR: GetSubArrCopy(v_In, IA1_STemp, DA1_VTempP1) returned FALSE" << endl;
              return false;
                }
                # ifdef __DEBUG_INTERPOL__
                cout << "CFits::HInterPol: DA1_VTempP1 set to " << DA1_VTempP1 << endl;
                #endif
                
                //    IA1_STemp = s_InOut - 1;
                //    pfsDRPStella::math::GetSubArrCopy(x_In, IA1_STemp, DA1_XTempM1);
                //    pfsDRPStella::math::GetSubArrCopy(v_In, IA1_STemp, DA1_VTempM1);
                
                PVecArr = (u_In - DA1_XTemp)
                * (DA1_VTempP1 - DA1_VTemp)
                / (DA1_XTempP1 - DA1_XTemp)
                + DA1_VTemp;
      }
      #ifdef __DEBUG_INTERPOL__
      cout << "CFits::HInterPol: Ready: Returning PVecArr = " << PVecArr << endl;
      #endif
      
      /**  IA1_Temp.resize(0);
       *  DA1_Temp.resize(0);
       *  DA1_TempA.resize(0);
       *  DA1_VTempP1.resize(0);
       *  DA1_VTemp.resize(0);
       *  DA1_XTempP1.resize(0);
       *  DA1_XTemp.resize(0);
       *  IA1_STemp.resize(0);
       *  TmpVecArr.resize(0);
       *  T1VecArr.resize(0);
       *  T2VecArr.resize(0);
       *  X1VecArr.resize(0);
       *  X0VecArr.resize(0);
       *  X2VecArr.resize(0);
       *  X0Arr.resize(0);
       *  V0Arr.resize(0);
       *  QArr.resize(0);
       **/
//      blitz::Array<double, 1> *P_PVecArr = new blitz::Array<double, 1>(PVecArr.size());
//      (*P_PVecArr) = PVecArr;
      
      
      y_Out.resize(PVecArr.size());
      y_Out = PVecArr;
//      delete(P_PVecArr);
      PVecArr.resize(0);
      return true;
    }
    
    blitz::Array<int, 1>* valueLocate(const blitz::Array<double, 1> &vec_In, 
                                      const blitz::Array<double, 1> &valueVec_In){
      #ifdef __DEBUG_INTERPOL__
        cout << "CFits::ValueLocate: vec_In = " << vec_In << endl;
        cout << "CFits::ValueLocate: valueVec_In = " << valueVec_In << endl;
      #endif
      if (vec_In.size() < 1){
        cout << "CFits::ValueLocate: ERROR: vec_In.size() < 1 => Returning FALSE" << endl;
        exit(EXIT_FAILURE);
      }
      if (valueVec_In.size() < 1){
        cout << "CFits::ValueLocate: ERROR: valueVec_In.size() < 1 => Returning FALSE" << endl;
        exit(EXIT_FAILURE);
      }
      blitz::Array<int, 1> IntVecArr(valueVec_In.size());
      
      int n;
      int N = vec_In.size();
      int M = valueVec_In.size();
      
      bool Increasing = false;
      int ii=0;
      while((ii < static_cast<int>(vec_In.size())) && (vec_In(ii) == vec_In(ii+1))){
        ii++;
      }
      if (vec_In(ii+1) > vec_In(ii))
        Increasing = true;
      
      #ifdef __DEBUG_INTERPOL__
        if (Increasing)
          cout << "CFits::ValueLocate: input vector Increasing = TRUE" << endl;
        else
          cout << "CFits::ValueLocate: input vector Increasing = FALSE" << endl;
      #endif
      
      /// For every element in valueVec_In
      for (int m = 0; m < M; m++){
        #ifdef __DEBUG_INTERPOL__
          cout << "CFits::ValueLocate: valueVec_In(m) = " << valueVec_In(m) << endl;
        #endif
        if (Increasing){
          if (valueVec_In(m) < vec_In(0)){
            IntVecArr(m) = 0 - 1;
          }
          else if (vec_In(N-1) <= valueVec_In(m)){
            IntVecArr(m) = N - 1;
          }
          else{
            n = -1;
            while (n < N-1){
              n++;
              if (vec_In(n) <= valueVec_In(m) && valueVec_In(m) < vec_In(n+1)){
                IntVecArr(m) = n;
                break;
              }
            }
          }
          #ifdef __DEBUG_INTERPOL__
            cout << "CFits::ValueLocate: Increasing = TRUE: IntVecArr(m) = " << IntVecArr(m) << endl;
          #endif
        }
        else{/// if (Decreasing)
          if (vec_In(0) <= valueVec_In(m))
            IntVecArr(m) = 0 - 1;
          else if (valueVec_In(m) < vec_In(N-1))
            IntVecArr(m) = N - 1;
          else{
            n = -1;
            while (n < N-1){
              n++;
              if ((vec_In(n+1) <= valueVec_In(m)) && (valueVec_In(m) < vec_In(n))){
                IntVecArr(m) = n;
                break;
              }
            }
          }
          #ifdef __DEBUG_INTERPOL__
            cout << "CFits::ValueLocate: Increasing = FALSE: IntVecArr(m) = " << IntVecArr(m) << endl;
          #endif
        }
      }
      #ifdef __DEBUG_INTERPOL__
        cout << "CFits::ValueLocate: IntVecArr = " << IntVecArr << endl;
      #endif
      blitz::Array<int, 1> *P_I_Result = new blitz::Array<int, 1>(IntVecArr.size());
      (*P_I_Result) = IntVecArr;
      #ifdef __DEBUG_INTERPOL__
        cout << "CFits::ValueLocate: *P_I_Result = " << (*P_I_Result) << endl;
      #endif
      IntVecArr.resize(0);
      return P_I_Result;
    }
    
    bool LsToFit(const blitz::Array<double, 1> &XXVecArr, 
                 const blitz::Array<double, 1> &YVecArr, 
                 const double &XM, 
                 double &D_Out){
      #ifdef __DEBUG_INTERPOL__
        cout << "CFits::LsToFit(XXVecArr = " << XXVecArr << ", YVecArr = " << YVecArr << ", XM = " << XM << ") Started" << endl;
      #endif
        
      blitz::Array<double, 1> XVecArr(XXVecArr.size());
      XVecArr = XXVecArr - XXVecArr(0);
      
      long NDegree = 2;
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: NDegree set to " << NDegree << endl;
      #endif
      
      long N = XXVecArr.size();
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: N set to " << N << endl;
      #endif
      
      blitz::Array<double, 2> CorrMArr(NDegree + 1, NDegree + 1);
      
      blitz::Array<double, 1> BVecArr(NDegree + 1);
      
      CorrMArr(0, 0) = N;
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: CorrMArr(0,0) set to " << CorrMArr(0,0) << endl;
      #endif
      
      BVecArr(0) = blitz::sum(YVecArr);
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: BVecArr(0) set to " << BVecArr(0) << endl;
      #endif
      
      blitz::Array<double, 1> ZVecArr(XXVecArr.size());
      ZVecArr = XVecArr;
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: ZVecArr set to " << ZVecArr << endl;
      #endif
      
      blitz::Array<double, 1> TempVecArr(YVecArr.size());
      TempVecArr = YVecArr;
      TempVecArr *= ZVecArr;
      BVecArr(1) = blitz::sum(TempVecArr);
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: BVecArr(1) set to " << BVecArr(1) << endl;
      #endif
      
      CorrMArr(0, 1) = blitz::sum(ZVecArr);
      CorrMArr(1, 0) = blitz::sum(ZVecArr);
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: CorrMArr(0,1) set to " << CorrMArr(0,1) << endl;
        cout << "CFits::LsToFit: CorrMArr(1,0) set to " << CorrMArr(1,0) << endl;
      #endif
      
      ZVecArr *= XVecArr;
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: ZVecArr set to " << ZVecArr << endl;
      #endif
      
      TempVecArr.resize(YVecArr.size());
      TempVecArr = YVecArr;
      TempVecArr *= ZVecArr;
      BVecArr(2) = blitz::sum(TempVecArr);
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: BVecArr(2) set to " << BVecArr(2) << endl;
      #endif
      
      CorrMArr(0, 2) = blitz::sum(ZVecArr);
      CorrMArr(1, 1) = blitz::sum(ZVecArr);
      CorrMArr(2, 0) = blitz::sum(ZVecArr);
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: CorrMArr(0,2) set to " << CorrMArr(0,2) << endl;
        cout << "CFits::LsToFit: CorrMArr(1,1) set to " << CorrMArr(1,1) << endl;
        cout << "CFits::LsToFit: CorrMArr(2,0) set to " << CorrMArr(2,0) << endl;
      #endif
      
      ZVecArr *= XVecArr;
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: ZVecArr set to " << ZVecArr << endl;
      #endif
      
      CorrMArr(1, 2) = blitz::sum(ZVecArr);
      CorrMArr(2, 1) = blitz::sum(ZVecArr);
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: CorrMArr(1,2) set to " << CorrMArr(1,2) << endl;
        cout << "CFits::LsToFit: CorrMArr(2,1) set to " << CorrMArr(2,1) << endl;
      #endif
      
      TempVecArr.resize(ZVecArr.size());
      TempVecArr = ZVecArr;
      TempVecArr *= XVecArr;
      CorrMArr(2, 2) = blitz::sum(TempVecArr);
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: CorrMArr(2,2) set to " << CorrMArr(2,2) << endl;
      #endif
      
      blitz::Array<double, 2> CorrInvMArr;
      CorrInvMArr.resize(CorrMArr.rows(), CorrMArr.cols());
      CorrInvMArr = CorrMArr;
      if (!pfsDRPStella::math::InvertGaussJ(CorrInvMArr)){
        cout << "CFits::LsToFit: ERROR: InvertGaussJ(CorrInvMArr) returned FALSE" << endl;
        return false;
      }
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: CorrInvMArr set to " << CorrInvMArr << endl;
      #endif
      blitz::Array<double, 1> *p_CVecArr = pfsDRPStella::math::VecArrTimesMatrix(BVecArr, CorrInvMArr);
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: p_CVecArr set to " << *p_CVecArr << endl;
      #endif
      
      //xm0 = xm - xx[0]
      double XM0 = XM - XXVecArr(0);
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: XM0 set to " << XM0 << endl;
      #endif
      
      D_Out = (*p_CVecArr)(0) + ((*p_CVecArr)(1) * XM0) + ((*p_CVecArr)(2) * pow(XM0, 2));
      #ifdef __DEBUG_LSTOFIT__
        cout << "CFits::LsToFit: D_Out set to " << D_Out << endl;
      #endif
      XVecArr.resize(0);
      CorrMArr.resize(0,0);
      BVecArr.resize(0);
      ZVecArr.resize(0);
      TempVecArr.resize(0);
      delete p_CVecArr;
      CorrInvMArr.resize(0,0);
      return true;
    }
    
    /**
     *  InvertGaussJ(AArray, BArray)
     *  Linear equation solution by Gauss-Jordan elimination
     *  AArray(0:N-1, 0:N-1) is the input matrix. BArray(0:N-1, 0:M-1) is input containing the m right-hand side vectors.
     *  On output, AArray is replaced by its matrix inverse, and BArray is replaced by the corresponding set of solution vectors.
     **/
    bool InvertGaussJ(blitz::Array<double, 2> &AArray, 
                      blitz::Array<double, 2> &BArray){
      /// The integer arrays IPivVecArr, IndXCVecArr, and IndXRVecArr are used for bookkeeping on the pivoting
      blitz::Array<int, 1> IndXCVecArr(AArray.extent(blitz::firstDim));
      blitz::Array<int, 1> IndXRVecArr(AArray.extent(blitz::firstDim));
      blitz::Array<int, 1> IPivVecArr(AArray.extent(blitz::firstDim));
      
      #ifdef __DEBUG_INVERT__
        cout << "CFits::InvertGaussJ: AArray = " << AArray << endl;
        cout << "CFits::InvertGaussJ: BArray = " << BArray << endl;
        cout << "CFits::InvertGaussJ: IndXCVecArr = " << IndXCVecArr << endl;
        cout << "CFits::InvertGaussJ: IndXRVecArr = " << IndXRVecArr << endl;
        cout << "CFits::InvertGaussJ: IPivVecArr = " << IPivVecArr << endl;
      #endif
      
      int m = 0, icol = 0, irow = 0, n = 0, o = 0, p = 0, pp = 0, N = 0, M = 0;
      double Big = 0., dum = 0., PivInv = 0.;
      
      N = AArray.rows();
      M = BArray.cols();
      
      IPivVecArr = 0;
      #ifdef __DEBUG_INVERT__
        cout << "CFits::InvertGaussJ: N = " << N << ", M = " << M << endl;
        cout << "CFits::InvertGaussJ: IPivVecArr = " << IPivVecArr << endl;
      #endif
      
      /// This is the main loop over the columns to be reduced
      for (m = 0; m < N; m++){ /// m == i
        Big = 0.;
        
        /// This is the outer loop of the search for a pivot element
        for (n = 0; n < N; n++){ /// n == j
          if (IPivVecArr(n) != 1){
            for (o = 0; o < N; o++){ /// o == k
              if (IPivVecArr(o) == 0){
                if (std::fabs(AArray(n, o)) >= Big){
                  Big = std::fabs(AArray(n, o));
                  irow = n;
                  icol = o;
                }
              }
            }
          } /// end if (IPivVecArr(n) != 1)
        } /// end for (n = 0; n < N; n++)     Outer Loop
        ++(IPivVecArr(icol));
        
        /** We now have the pivot element, so we interchange rows, if needed, to put the pivot element on the diagonal. The columns are not physically interchanged, only relabled: IndXCVecArr(m), the column of the mth pivot element, is the mth column that is reduced, while IndXRVecArr(m) is the row in which that pivot element was originally located. If IndXCVecArr(i) != IndXRVecArr there is an implied column interchange. With this form of bookkeeping, the solution BVecArr's will end up in the correct order, and the inverse matrix will be scrambled by columns.
         **/
        if (irow != icol){
          for (p = 0; p < N; p++){
            std::swap(AArray(irow, p), AArray(icol, p));
          }
          for (p = 0; p < M; p++){
            std::swap(BArray(irow, p), BArray(icol, p));
          }
        }
        IndXRVecArr(m) = irow;
        IndXCVecArr(m) = icol;
        
        /** We are now ready to divide the pivot row by the pivot element, located at irow and icol
         **/
        if (AArray(icol, icol) == 0.0){
          cout << "CFits::InvertGaussJ: Error 1: Singular Matrix!" << endl;
          cout << "CFits::InvertGaussJ: AArray = " << AArray << endl;
          cout << "CFits::InvertGaussJ: AArray(" << icol << ", " << icol << ") == 0." << endl;
          return false;
        }
        PivInv = 1.0 / AArray(icol, icol);
        AArray(icol, icol) = 1.0;
        for (p = 0; p < N; p++){
          AArray(icol, p) *= PivInv;
        }
        for (p = 0; p < M; p++){
          BArray(icol, p) *= PivInv;
        }
        
        /**
         *      Next, we reduce the rows...
         *          ... except for the pivot one, of course
         **/
        for (pp = 0; pp < N; pp++){
          if (pp != icol){
            dum = AArray(pp, icol);
            AArray(pp, icol) = 0.0;
            for (p = 0; p < N; p++){
              AArray(pp, p) -= AArray(icol, p) * dum;
            }
            for (p = 0; p < M; p++){
              BArray(pp, p) -= BArray(icol, p) * dum;
            }
          } /// end if (pp != icol)
        } /// end for (pp = 0; pp < N; pp++)
      } /// end for (m = 0; m < N; m++)       Main Loop
      /** This is the end of the main loop over columns of the reduction. It only remains to unscramble the solution in view of the column interchanges. We do this by interchanging pairs of columns int the reverse order that the permutation was built up.
       **/
      for (p = N-1; p >= 0; p--){
        if (IndXRVecArr(p) != IndXCVecArr(p)){
          for (o = 0; o < N; o++){
            std::swap(AArray(o, IndXRVecArr(p)), AArray(o, IndXCVecArr(p)));
            /// And we are done.
          }
        }
      }
      IndXCVecArr.resize(0);
      IndXRVecArr.resize(0);
      IPivVecArr.resize(0);
      return true;
    }
    
    /**
     *  InvertGaussJ(AArray)
     *  From: Numerical Recipies
     *  Linear equation solution by Gauss-Jordan elimination with B == Unity
     *  AArray(0:N-1, 0:N-1) is the input matrix.
     *  On output, AArray is replaced by its matrix inverse.
     **/
    bool InvertGaussJ(blitz::Array<double, 2> &AArray){
      int N = AArray.cols();
      if (N != AArray.rows()){
        cout << "CFits::InvertGaussJ(AArray=" << AArray << "): ERROR: AArray is not quadratic!" << endl;
        return false;
      }
      blitz::Array<double, 2> Unity(N, N);
      Unity = 0.;
      for (int m = 0; m < N; m ++){
        Unity(m, m) = 1.;
      }
      if (!pfsDRPStella::math::InvertGaussJ(AArray, Unity)){
        cout << "CFits::InvertGaussJ: ERROR: InvertGaussJ(AArray=" << AArray << ", Unity=" << Unity << ") retuned FALSE" << endl;
        return false;
      }
      Unity.resize(0);
      return true;
    }
    
    /**
     * MatrixATimesB(blitz::Array<double, 2> &Arr, blitz::Array<double, 2> &B);
     **/
    blitz::Array<double, 2>* MatrixATimesB(const blitz::Array<double, 2> &A, 
                                                  const blitz::Array<double, 2> &B){
      #ifdef __DEBUG_MULT__
        cout << "CFits::MatrixATimesB(A = " << A << ", B = " << B << ") started" << endl;
      #endif
      blitz::Array<double, 2> *P_TempArray = new blitz::Array<double, 2>(A.rows(), B.cols());
      int m, n, o;
      double dtemp;
      (*P_TempArray) = 0.;
      #ifdef __DEBUG_MULT__
        cout << "CFits::MatrixATimesB: TempArray = " << TempArray << endl;
      #endif
      for (m = 0; m < A.rows(); m++){
        for (n = 0; n < B.cols(); n++){
          for (o = 0; o < A.cols(); o++){
            dtemp = A(m, o);
            dtemp = dtemp * B(o, n);
            (*P_TempArray)(m, n) += dtemp;
          }
        }
      }
      #ifdef __DEBUG_MULT__
        cout << "CFits::MatrixATimesB: End: TempArray = " << *P_TempArray << endl;
      #endif
      return (P_TempArray);
    }
    
    /**
     * MatrixBTimesA(blitz::Array<double, 2> &Arr, blitz::Array<double, 2> &B);
     **/
    blitz::Array<double, 2>* MatrixBTimesA(const blitz::Array<double, 2> &A, 
                                           const blitz::Array<double, 2> &B){
      return pfsDRPStella::math::MatrixATimesB(B, A);
    }
    
    /**
     * MatrixTimesVecArr(blitz::Array<double, 2> &Arr, blitz::Array<double, 1> &B);
     **/
    blitz::Array<double, 1>* MatrixTimesVecArr(const blitz::Array<double, 2> &A, 
                                               const blitz::Array<double, 1> &B){
      if (static_cast<int>(B.size()) != A.extent(blitz::secondDim))
        return (new blitz::Array<double, 1>(A.extent(blitz::firstDim)));
      blitz::Array<double, 2> ProductArr(B.size(), 1);
      for (int m = 0; m < static_cast<int>(B.size()); m++)
        ProductArr(m, 0) = B(m);
      #ifdef __DEBUG_MULT__
        cout << "CFits::MatrixTimesVecArr: A = " << A << endl;
        cout << "CFits::MatrixTimesVecArr: B = " << B << endl;
        cout << "CFits::MatrixTimesVecArr: ProductArr = " << ProductArr << endl;
      #endif
      blitz::Array<double, 2> *p_TempMatrix = pfsDRPStella::math::MatrixATimesB(A, ProductArr);
      #ifdef __DEBUG_MULT__
        cout << "CFits::MatrixTimesVecArr: TempMatrix = " << *p_TempMatrix << endl;
      #endif
      //  ProductArr.resize(0,0);
      blitz::Array<double, 1> *p_RefArr = pfsDRPStella::math::Reform(*p_TempMatrix);
      delete p_TempMatrix;
      return p_RefArr;
    }
    
    /**
     * VecArrTimesMatrix(blitz::Array<double, 1> &Arr, blitz::Array<double, 2> &B);
     **/
    blitz::Array<double, 1>* VecArrTimesMatrix(const blitz::Array<double, 1> &A, 
                                               const blitz::Array<double, 2> &B){
      if (static_cast<int>(A.size()) != B.extent(blitz::firstDim)){
        #ifdef __DEBUG_MULT__
          cout << "CFits::VecArrTimesMatrix: A(=" << A << ").size(=" << A.size() << ") != B(=" << B << ").extent(blitz::firstDim)=" << B.extent(blitz::firstDim) << " => Returning new VecArr(" << B.extent(blitz::secondDim) << ")" << endl;
        #endif
        return (new blitz::Array<double, 1>(B.extent(blitz::secondDim)));
      }
      blitz::Array<double, 2> ProductArr(1, A.size());
      for (int m = 0; m < static_cast<int>(A.size()); m++)
        ProductArr(0, m) = A(m);
      #ifdef __DEBUG_MULT__
        cout << "CFits::VecArrTimesMatrix: A = " << A << endl;
        cout << "CFits::VecArrTimesMatrix: B = " << B << endl;
        cout << "CFits::VecArrTimesMatrix: ProductArr = " << ProductArr << endl;
      #endif
      blitz::Array<double, 2> *p_temp = pfsDRPStella::math::MatrixATimesB(ProductArr, B);
      blitz::Array<double, 1> *p_tempA = pfsDRPStella::math::Reform(*p_temp);
      delete p_temp;
      return p_tempA;
    }
    
    /**
     * VecArrACrossB(blitz::Array<double, 1> &Arr, blitz::Array<double, 1> &B);
     **/
    blitz::Array<double, 2>* VecArrACrossB(const blitz::Array<double, 1> &A, 
                                           const blitz::Array<double, 1> &B){
      blitz::Array<double, 2> *P_TempArray = new blitz::Array<double, 2>(A.size(), B.size());
      int m, n;
      double dtemp;
      (*P_TempArray) = 0.;
      for (m = 0; m < static_cast<int>(A.size()); m++){
        for (n = 0; n < static_cast<int>(B.size()); n++){
          dtemp = A(m);
          dtemp = dtemp * B(n);
          (*P_TempArray)(m, n) = dtemp;
        }
      }
      return (P_TempArray);
    }
    
    /**
     * VecArrACrossB(blitz::Array<int, 1> &Arr, blitz::Array<int, 1> &B);
     **/
    blitz::Array<int, 2>* VecArrACrossB(const blitz::Array<int, 1> &A, 
                                        const blitz::Array<int, 1> &B){
      blitz::Array<int, 2> *P_TempArray = new blitz::Array<int, 2>(A.size(), B.size());
      int m, n;
      int dtemp;
      (*P_TempArray) = 0.;
      for (m = 0; m < static_cast<int>(A.size()); m++){
        for (n = 0; n < static_cast<int>(B.size()); n++){
          dtemp = A(m);
          dtemp = dtemp * B(n);
          (*P_TempArray)(m, n) = dtemp;
        }
      }
      return (P_TempArray);
    }
    
    /**
     * VecArrAScalarB(blitz::Array<double, 1> &Arr, blitz::Array<double, 1> &B);
     **/
    double VecArrAScalarB(const blitz::Array<double, 1> &A, 
                          const blitz::Array<double, 1> &B){
      if (A.extent(blitz::firstDim) != B.extent(blitz::firstDim))
        return 0.;
      return blitz::sum(A * B);
    }
    
    /**
     * Reform(blitz::Array<double, 1> &Arr, int DimA, int DimB);
     **/
    template<typename T>
    blitz::Array<T, 2>* Reform(const blitz::Array<T, 1> &VecArr, 
                               int NRow, 
                               int NCol){
      const T *data = VecArr.data();
      #ifdef __DEBUG_REFORM__
        cout << "CFits::Reform(VecArr, NRow, NCol): VecArr.data() returns data=<" << *data << ">" << endl;
        for (int m = 0; m < VecArr.size(); m++)
          cout << "CFits::Reform(VecArr, NRow, NCol): data[m=" << m << "]=<" << data[m] << ">" << endl;
      #endif
      blitz::Array<T, 2> *P_TempArray = new blitz::Array<T, 2>(NRow, NCol);
      for (int i_row=0; i_row < NRow; i_row++){
        for (int i_col=0; i_col < NCol; i_col++){
          (*P_TempArray)(i_row, i_col) = data[(i_row*NCol) + i_col];
        }
      }
      #ifdef __DEBUG_REFORM__
        cout << "CFits::Reform(VecArr, NRow=" << NRow << ", NCol=" << NCol << "): returning <" << *P_TempArray << ">" << endl;
      #endif
      return P_TempArray;
    }
    
    /**
     * Reform(blitz::Array<double, 1> &Arr, int DimA, int DimB);
     * Reformates a 2-dimensional array into a vector
     **/
    template<typename T>
    blitz::Array<T, 1>* Reform(const blitz::Array<T, 2> &Arr){
      int na = Arr.extent(blitz::firstDim);
      int nb = Arr.extent(blitz::secondDim);
      int n;
      if (na == 1){
        n = nb;
      }
      else if (nb == 1){
        n = na;
      }
      else{
        n = na * nb;
      }
      blitz::Array<T, 1> *P_TempVecArr = new blitz::Array<T, 1>(n);
      if ((na == 1) || (nb == 1)){
        for (int m = 0; m < n; m++){
          if (na == 1)
            (*P_TempVecArr)(m) = Arr(0, m);
          else if (nb == 1)
            (*P_TempVecArr)(m) = Arr(m, 0);
        }
      }
      else{
        int pos = 0;
        for (int m = 0; m < na; m++){
          for (int n = 0; n < nb; n++){
            (*P_TempVecArr)(pos) = Arr(m, n);
            pos++;
          }
        }
      }
      return (P_TempVecArr);
    }
    
    /**
      GetSubArrCopy(blitz::Array<double, 1> &DA1_In, blitz::Array<int, 1> &IA1_Indices, blitz::Array<double, 1> &DA1_Out) const
    **/
    template<typename T>
    bool GetSubArrCopy(const blitz::Array<T, 1> &DA1_In,
                       const blitz::Array<int, 1> &IA1_Indices,
                       blitz::Array<T, 1> &DA1_Out){
      #ifdef __DEBUG_GETSUBARRCOPY__
        cout << "CFits::GetSubArrCopy: IA1_Indices = " << IA1_Indices << endl;
      #endif
      DA1_Out.resize(IA1_Indices.size());
      if (static_cast<int>(DA1_In.size()) < max(IA1_Indices)){
        cout << "CFits::GetSubArrCopy: ERROR: DA1_In.size(=" << DA1_In.size() << ") < max(IA1_Indices=" << max(IA1_Indices) << endl;
        return false;
      }
      for (unsigned int m = 0; m < IA1_Indices.size(); m++){
        DA1_Out(m) = DA1_In(IA1_Indices(m));
      }
      return true;
    }
    
    /**
     *  GetSubArrCopy(blitz::Array<int, 2> &IA2_In, blitz::Array<int, 1> &IA1_Indices, int I_Mode_In, blitz::Array<int, 2> &IA2_Out) const
     *  Copies the values of IA1_In(IA1_Indices) to IA1_Out
     *  I_Mode_In: 0: IA1_Indices are row numbers
     *             1: IA1_Indices are column numbers
     **/
    template<typename T>
    bool GetSubArrCopy(const blitz::Array<T, 2> &A2_In,
                       const blitz::Array<int, 1> &I_A1_Indices,
                       int I_Mode_In,
                       blitz::Array<T, 2> &A2_Out)
    {
      if (I_Mode_In > 1){
        cout << "CFits::GetSubArrCopy: ERROR: I_Mode_In > 1" << endl;
        return false;
      }
      if (I_Mode_In == 0){
        A2_Out.resize(I_A1_Indices.size(),A2_In.cols());
        if (max(I_A1_Indices) >= A2_In.rows()){
          cout << "CFits::GetSubArrCopy: ERROR: max(I_A1_Indices) >= A2_In.rows()" << endl;
          return false;
        }
      }
      else{// if (I_Mode_In == 1){
        A2_Out.resize(A2_In.rows(),I_A1_Indices.size());
        if (max(I_A1_Indices) >= A2_In.cols()){
          cout << "CFits::GetSubArrCopy: ERROR: max(I_A1_Indices) >= A2_In.cols()" << endl;
          return false;
        }
      }

      for (int m=0; m < static_cast<int>(I_A1_Indices.size()); m++){
        if (I_Mode_In == 0){
          A2_Out(m,blitz::Range::all()) = A2_In(I_A1_Indices(m),blitz::Range::all());
        }
        else{// if (I_Mode_In == 1){
          A2_Out(blitz::Range::all(),m) = A2_In(blitz::Range::all(),I_A1_Indices(m));
        }
      }
      return true;
    }
    
    /**
     *  GetSubArr(blitz::Array<double, 1> &DA1_In, blitz::Array<int, 3> &I_A3_Indices) const
     *  Copies the values of DA1_In(I_A3_Indices(row,col,0), I_A3_Indices(row,col,1)) to D_A2_Out
     **/
    template<typename T>
    blitz::Array<T, 2> GetSubArrCopy(const blitz::Array<T, 2> &A2_In, 
                                     const blitz::Array<int, 3> &I_A3_Indices)
    {
      blitz::Array<T, 2> A2_Out(I_A3_Indices.rows(), I_A3_Indices.cols());
      for (int u=0; u < I_A3_Indices.rows(); u++){
        for (int v=0; v < I_A3_Indices.cols(); v++){
          A2_Out(u,v) = A2_In(I_A3_Indices(u,v,0), I_A3_Indices(u,v,1));
        }
      }
      return (A2_Out);
    }
    
    /**
     *  Spline
     *  Given Arrays x_In(0:N-1) and y_In(0:N-1) containing a tabulated function, i.e., y_i = f(x_i), with x_1 < x_2 < ... < x_N, and given values yP1 and yPN for the first derivative of the interpolating function at points 1 and N, respectively, this routine returns an Array y2(0:N-1) that contains the second derivatives of the interpolating function at the tabulated points x_i. If yP1 and/or yPN are equal to 1x10^30 or larger, the routine is signaled to set the corresponding boundary condition for a natural spline, with zero second derivative on that boundary.
     **/
    bool Spline(const blitz::Array<double, 1> &x_In, 
                const blitz::Array<double, 1> &y_In, 
                double yP1, 
                double yPN, 
                blitz::Array<double, 1> &y_Out){
      int m, o, N = x_In.size();
      double p, qn, sig, un;
      blitz::Array<double, 1> UVecArr(N-1);
      y_Out.resize(N);
      
      if (yP1 > 0.99e30)  /// The lower boundary condition is set either to be "natural"
      {
        y_Out(0) = UVecArr(0) = 0.0;
      }
      else                /// or else to have a specified first derivative
      {
        y_Out(0) = -0.5;
        UVecArr(0)  = (3.0 / (x_In(1) - x_In(0))) * ((y_In(1) - y_In(0)) / (x_In(1) - x_In(0)) - yP1);
      }
      
      /**
       *  This is the decomposition loop of the tridiagonal algorithm. y_Out and UVecArr are used for temporary storage of the decomposed factors.
       **/
      for (m = 1; m < N-1; m++)
      {
        sig = (x_In(m) - x_In(m-1)) / (x_In(m + 1) - x_In(m-1));
        p = sig * y_Out(m - 1) + 2.0;
        y_Out(m) = (sig - 1.0) / p;
        UVecArr(m)  = (y_In(m+1) - y_In(m)) / (x_In(m+1) - x_In(m)) - (y_In(m) - y_In(m-1)) / (x_In(m) - x_In(m-1));
        UVecArr(m)  = (6.0 * UVecArr(m) / (x_In(m+1) - x_In(m-1)) - sig * UVecArr(m-1)) / p;
      }
      if (yPN > 0.99e30)  /// The upper boundary condition is set either to be "natural"
        qn = un = 0.0;
      else                /// or else to have a specified first derivative
      {
        qn = 0.5;
        un = (3.0 / (x_In(N-1) - x_In(N-2))) * (yPN - (y_In(N-1) - y_In(N-2)) / (x_In(N-1) - x_In(N-2)));
      }
      y_Out(N-1) = (un - qn * UVecArr(N-2)) / (qn * y_Out(N-2) + 1.0);
      
      /// This is the backsubstitution loop of the tridiagonal algorithm
      for (o = N - 2; o >= 0; o--)
      {
        y_Out(o) = y_Out(o) * y_Out(o+1) + UVecArr(o);
      }
      UVecArr.resize(0);
      return true;
    }
    
    /**
     *  Spline
     *  Given Arrays x_In(0:N-1) and y_In(0:N-1) containing a tabulated function, i.e., y_i = f(x_i), with x_1 < x_2 < ... < x_N, this routine returns an Array y2(0:N-1) that contains the second derivatives of the interpolating function at the tabulated points x_i. The routine is signaled to set the corresponding boundary condition for a natural spline, with zero second derivative on that boundary.
     **/
    bool Spline(const blitz::Array<double, 1> &x_In, 
                const blitz::Array<double, 1> &y_In, 
                blitz::Array<double, 1> &y_Out){
      return pfsDRPStella::math::Spline(x_In, y_In, 1.0e30, 1.0e30, y_Out);
    }
    
    /**
     *  SplInt
     *  Given the Arrays xVec_In(0:N-1) and y1_In(0:N-1), which tabulate a function (whith the xVec_In(i)'s in order), and given the array y2_In(0:N-1), which is the output from Spline above, and given a value of x_In, this routine returns a cubic-spline interpolated value y_Out;
     **/
    bool SplInt(const blitz::Array<double, 1> &xVec_In, 
                blitz::Array<double, 1> &y1_In, 
                blitz::Array<double, 1> &y2_In, 
                double x_In, 
                double *y_Out){
      int klo, khi, o, N;
      double h, b, a;
      
      N = xVec_In.size();
      /**
       *  We will find the right place in the table by means of bisection. This is optimal if sequential calls to this routine are at random values of x_In. If sequential calls are in order, and closely spaced, one would do better to store previous values of klo and khi and test if they remain appropriate on the next call.
       **/
      klo = 1;
      khi = N;
      while (khi - klo > 1)
      {
        o = (khi + klo) >> 1;
        if (xVec_In(o) > x_In)
          khi = o;
        else
          klo = o;
      } /// klo and khi now bracket the input value of x_In
      h = xVec_In(khi) - xVec_In(klo);
      if (h == 0.0)  /// The xVec_In(i)'s must be distinct
      {
        cout << "CFits::SplInt: ERROR: Bad xVec_In input to routine SplInt" << endl;
        return false;
      }
      a = (xVec_In(khi) - x_In) / h;
      b = (x_In - xVec_In(klo)) / h; /// Cubic Spline polynomial is now evaluated.
      *y_Out = a * y1_In(klo) + b * y1_In(khi) + ((a * a * a - a) * y2_In(khi)) * (h * h) / 6.0;
      return true;
    }
    
    /**
     *  LsToFit
     **/
    bool LsToFit(const blitz::Array<double, 1> &XXVecArr, 
                 const blitz::Array<double, 1> &YVecArr, 
                 double XM, double &D_Out){
      //  Function ls2fit, xx, y, xm
      
      //COMPILE_OPT hidden, strictarr
      
      //x = xx - xx[0]
      ///Normalize to preserve significance.
      blitz::Array<double, 1> XVecArr(XXVecArr.size());
      XVecArr = XXVecArr - XXVecArr(0);
      
      //ndegree = 2L
      long NDegree = 2;
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: NDegree set to " << NDegree << endl;
      #endif
      
      //n = n_elements(xx)
      long N = XXVecArr.size();
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: N set to " << N << endl;
      #endif
      
      //corrm = fltarr(ndegree+1, ndegree+1)
      ///Correlation matrix
      blitz::Array<double, 2> CorrMArr(NDegree + 1, NDegree + 1);
      
      //b = fltarr(ndegree+1)
      blitz::Array<double, 1> BVecArr(NDegree + 1);
      
      //corrm[0,0] = n
      ///0 - Form the normal equations
      CorrMArr(0, 0) = N;
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: CorrMArr(0,0) set to " << CorrMArr(0,0) << endl;
      #endif
      
      //b[0] = total(y)
      BVecArr(0) = blitz::sum(YVecArr);
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: BVecArr(0) set to " << BVecArr(0) << endl;
      #endif
      
      //z = x
      ///1
      blitz::Array<double, 1> ZVecArr(XXVecArr.size());
      ZVecArr = XVecArr;
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: ZVecArr set to " << ZVecArr << endl;
      #endif
      
      //b[1] = total(y*z)
      blitz::Array<double, 1> TempVecArr(YVecArr.size());
      TempVecArr = YVecArr;
      TempVecArr *= ZVecArr;
      BVecArr(1) = blitz::sum(TempVecArr);
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: BVecArr(1) set to " << BVecArr(1) << endl;
      #endif
      
      //corrm[[0,1],[1,0]] = total(z)
      CorrMArr(0, 1) = blitz::sum(ZVecArr);
      CorrMArr(1, 0) = blitz::sum(ZVecArr);
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: CorrMArr(0,1) set to " << CorrMArr(0,1) << endl;
      cout << "CFits::LsToFit: CorrMArr(1,0) set to " << CorrMArr(1,0) << endl;
      #endif
      
      //z = z * x
      ///2
      ZVecArr *= XVecArr;
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: ZVecArr set to " << ZVecArr << endl;
      #endif
      
      //b[2] = total(y*z)
      TempVecArr.resize(YVecArr.size());
      TempVecArr = YVecArr;
      TempVecArr *= ZVecArr;
      BVecArr(2) = blitz::sum(TempVecArr);
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: BVecArr(2) set to " << BVecArr(2) << endl;
      #endif
      
      //corrm[[0,1,2], [2,1,0]] = total(z)
      CorrMArr(0, 2) = blitz::sum(ZVecArr);
      CorrMArr(1, 1) = blitz::sum(ZVecArr);
      CorrMArr(2, 0) = blitz::sum(ZVecArr);
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: CorrMArr(0,2) set to " << CorrMArr(0,2) << endl;
      cout << "CFits::LsToFit: CorrMArr(1,1) set to " << CorrMArr(1,1) << endl;
      cout << "CFits::LsToFit: CorrMArr(2,0) set to " << CorrMArr(2,0) << endl;
      #endif
      
      //z = z * x
      ///3
      ZVecArr *= XVecArr;
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: ZVecArr set to " << ZVecArr << endl;
      #endif
      
      //corrm[[1,2],[2,1]] = total(z)
      CorrMArr(1, 2) = blitz::sum(ZVecArr);
      CorrMArr(2, 1) = blitz::sum(ZVecArr);
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: CorrMArr(1,2) set to " << CorrMArr(1,2) << endl;
      cout << "CFits::LsToFit: CorrMArr(2,1) set to " << CorrMArr(2,1) << endl;
      #endif
      
      //corrm[2,2] = total(z*x)
      ///4
      TempVecArr.resize(ZVecArr.size());
      TempVecArr = ZVecArr;
      TempVecArr *= XVecArr;
      CorrMArr(2, 2) = blitz::sum(TempVecArr);
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: CorrMArr(2,2) set to " << CorrMArr(2,2) << endl;
      #endif
      
      //c = b # invert(corrm)
      blitz::Array<double, 2> CorrInvMArr;
      CorrInvMArr.resize(CorrMArr.rows(), CorrMArr.cols());
      CorrInvMArr = CorrMArr;
      if (!pfsDRPStella::math::InvertGaussJ(CorrInvMArr)){
        cout << "CFits::LsToFit: ERROR: InvertGaussJ(CorrInvMArr) returned FALSE" << endl;
        return false;
      }
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: CorrInvMArr set to " << CorrInvMArr << endl;
      #endif
      blitz::Array<double, 1> *p_CVecArr = pfsDRPStella::math::VecArrTimesMatrix(BVecArr, CorrInvMArr);
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: p_CVecArr set to " << *p_CVecArr << endl;
      #endif
      
      //xm0 = xm - xx[0]
      double XM0 = XM - XXVecArr(0);
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: XM0 set to " << XM0 << endl;
      #endif
      
      D_Out = (*p_CVecArr)(0) + ((*p_CVecArr)(1) * XM0) + ((*p_CVecArr)(2) * pow(XM0, 2));
      #ifdef __DEBUG_LSTOFIT__
      cout << "CFits::LsToFit: D_Out set to " << D_Out << endl;
      #endif
      XVecArr.resize(0);
      CorrMArr.resize(0,0);
      BVecArr.resize(0);
      ZVecArr.resize(0);
      TempVecArr.resize(0);
      delete p_CVecArr;
      CorrInvMArr.resize(0,0);
      return true;
    }
    
    
    /**
     *  function CountPixGTZero
     *  replaces input vector with vector of the same size where values are zero where the input vector is zero and all other values represent the number of non-zero values since the last zero value
     **/
    template<typename T>
    bool CountPixGTZero(blitz::Array<T, 1> &vec_InOut){
      int count = 0;
      if (vec_InOut.size() < 1)
        return false;
      for (unsigned int i=0; i < vec_InOut.size(); i++){
        #ifdef __DEBUG_COUNTPIXGTZERO__
          cout << "CFits::CountPixGTZero: i = " << i << ": vec_InOut(i) = " << vec_InOut(i) << endl;
        #endif
        if (vec_InOut(i) <= T(0))
          count = 0;
        else
          count++;
        #ifdef __DEBUG_COUNTPIXGTZERO__
          cout << "CFits::CountPixGTZero: i = " << i << ": count set to " << count << endl;
        #endif
        vec_InOut(i) = T(count);
        #ifdef __DEBUG_COUNTPIXGTZERO__
          cout << "CFits::CountPixGTZero: i = " << i << ": vec_InOut(i) set to " << vec_InOut(i) << endl;
        #endif
      }
      return true;
    }
    
    template<typename T>
    int FirstIndexWithValueGEFrom(const blitz::Array<T, 1> &vec_In, 
                                  const T minValue_In, 
                                  const int fromIndex_In){
      if (vec_In.size() < 1 || fromIndex_In >= static_cast<int>(vec_In.size())){
        cout << "CFits::FirstIndexWithValueGEFrom: Error: vec_In.size(=" << vec_In.size() << ") < 1 or fromIndex_In >= vec_In.size() => Returning -1" << endl;
        return -1;
      }
      for (int i=fromIndex_In; i < static_cast<int>(vec_In.size()); i++){
        if (vec_In(i) >= minValue_In)
          return i;
      }
      #ifdef __DEBUG_INDEX__
        cout << "CFits::FirstIndexWithValueGEFrom: not found => Returning -1" << endl;
      #endif
      return -1;
    }
    
    /**
     *  function LastIndexWithZeroValueBefore
     *  returns last index of integer input vector where value is equal to zero before index I_StartPos
     *  returns -1 if values are always greater than 0 before I_StartPos
     **/
    template<typename T>
    int LastIndexWithZeroValueBefore(const blitz::Array<T, 1> &vec_In, const int startPos_In){
      if ( ( startPos_In < 0 ) || ( startPos_In >= static_cast<int>(vec_In.size()) ) )
        return -1;
      for (int i=startPos_In; i >= 0; i--){
        if (fabs(double(vec_In(i))) < 0.00000000000000001)
          return i;
      }
      return -1;
    }
    
    /**
     *  function FirstIndexWithZeroValueFrom
     *  returns first index of integer input vector where value is equal to zero, starting at index I_StartPos
     *  returns -1 if values are always greater than 0
     **/
    template<typename T>
    int FirstIndexWithZeroValueFrom(const blitz::Array<T, 1> &vec_In, const int startPos_In){
      if (startPos_In < 0 || startPos_In >= static_cast<int>(vec_In.size()))
        return -1;
      for (int i=startPos_In; i < static_cast<int>(vec_In.size()); i++){
        #ifdef __DEBUG_FINDANDTRACE__
          cout << "CFits::FirstIndexWithZeroValueFrom: i = " << i << ":" << endl;
          cout << "CFits::FirstIndexWithZeroValueFrom: I_A1_VecIn(i) = " << vec_In(i) << endl;
        #endif
        if (fabs(T(vec_In(i))) < 0.00000000000000001)
          return i;
      }
      return -1;
    }


    /**
      Fit(blitz::Array<double, 1> y, const blitz::Array<double, 1> x, a1, a0);
      calculates a0 and a1 for the system of equations yvec = a0 + a1 * xvec
     **/
    bool LinFitBevington(const blitz::Array<double, 1> &D_A1_CCD_In,      /// yvec: in
                         const blitz::Array<double, 1> &D_A1_SF_In,       /// xvec: in
                         double &D_SP_Out,                         /// a1: out
                         double &D_Sky_Out,                        /// a0: in/out
                         const blitz::Array<string, 1> &S_A1_Args_In,   ///: in
                         void *ArgV_In[])                    ///: in
    {
      bool B_WithSky = true;
      if (D_Sky_Out < 0.00000001)
        B_WithSky = false;
      return pfsDRPStella::math::LinFitBevington(D_A1_CCD_In,
                                                 D_A1_SF_In,
                                                 D_SP_Out,
                                                 D_Sky_Out,
                                                 B_WithSky,
                                                 S_A1_Args_In,
                                                 ArgV_In);
    }

    /**
      Fit(blitz::Array<double, 1> y, const blitz::Array<double, 1> x, a1, a0);
      calculates a0 and a1 for the system of equations yvec = a0 + a1 * xvec
     **/
    bool LinFitBevington(const blitz::Array<double, 2> &D_A2_CCD_In,      /// yvec: in
                         const blitz::Array<double, 2> &D_A2_SF_In,       /// xvec: in
                         blitz::Array<double, 1> &D_A1_SP_Out,                         /// a1: out
                         blitz::Array<double, 1> &D_A1_Sky_Out,                        /// a0: out
                         const blitz::Array<string, 1> &S_A1_Args_In,   ///: in
                         void *ArgV_In[])                    ///: in
    {
      bool B_WithSky = true;
      if (max(D_A1_Sky_Out) < 0.0000001)
        B_WithSky = false;
      return pfsDRPStella::math::LinFitBevington(D_A2_CCD_In,
                                                 D_A2_SF_In,
                                                 D_A1_SP_Out,
                                                 D_A1_Sky_Out,
                                                 B_WithSky,
                                                 S_A1_Args_In,
                                                 ArgV_In);
    }

    /**
      Fit(blitz::Array<double, 1> y, const blitz::Array<double, 1> x, a1, a0);
      calculates a0 and a1 for the system of equations yvec = a0 + a1 * xvec
    **/
    bool LinFitBevington(const blitz::Array<double, 2> &D_A2_CCD_In,      /// yvec: in
                         const blitz::Array<double, 2> &D_A2_SF_In,       /// xvec: in
                         blitz::Array<double, 1> &D_A1_SP_Out,                         /// a1: out
                         blitz::Array<double, 1> &D_A1_Sky_Out,                        /// a0: out
                         bool B_WithSky,                           /// with sky: in
                         const blitz::Array<string, 1> &S_A1_Args_In,   ///: in
                         void *ArgV_In[])                    ///: in
    /// MEASURE_ERRORS_IN = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
    /// REJECT_IN         = double
    /// MASK_INOUT        = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
    /// CHISQ_OUT         = blitz::Array<double, 1>(D_A2_CCD_In.rows())
    /// Q_OUT             = blitz::Array<double, 1>(D_A2_CCD_In.rows())
    /// SIGMA_OUT         = blitz::Array<double, 2>(D_A2_CCD_In.rows(),2)
    /// YFIT_OUT          = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
    {
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr) started" << endl;
        cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr): D_A2_CCD_In = " << D_A2_CCD_In << endl;
        cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr): D_A2_SF_In = " << D_A2_SF_In << endl;
      #endif
      if (D_A2_CCD_In.size() != D_A2_SF_In.size()){
        cout << "CFits::LinFitBevington: ERROR: D_A2_CCD_In.size(=" << D_A2_CCD_In.size() << ") != D_A2_SF_In.size(=" << D_A2_SF_In.size() << ") => returning false" << endl;
        D_A1_SP_Out = 0.;
        D_A1_Sky_Out = 0.;
        return false;
      }
      int i, I_ArgPos = 0;
      int I_KeywordSet_MeasureErrors, I_KeywordSet_Reject, I_KewordSet_Mask, I_KeywordSet_ChiSq, I_KeywordSet_Q, I_KeywordSet_Sigma, I_KeywordSet_YFit;
      if (static_cast<int>(D_A1_SP_Out.size()) != D_A2_CCD_In.rows())
      {
        D_A1_SP_Out.resize(D_A2_CCD_In.rows());
      }
      D_A1_SP_Out = 0.;
      if (static_cast<int>(D_A1_Sky_Out.size()) != D_A2_CCD_In.rows())
      {
        D_A1_Sky_Out.resize(D_A2_CCD_In.rows());
      }
      D_A1_Sky_Out = 0.;
    
      double *P_D_Reject;

      blitz::Array<double, 1> *P_D_A1_YFit;
      blitz::Array<double, 2> *P_D_A2_YFit;
      blitz::Array<int, 1> *P_I_A1_Mask;
      blitz::Array<int, 2> *P_I_A2_Mask;
  
      blitz::Array<double, 1> *P_D_A1_Sigma;
      blitz::Array<double, 1> *P_D_A1_Sigma_Out;
      blitz::Array<string, 1> S_A1_Args_Fit(10);
      S_A1_Args_Fit = " ";
      void **PP_Args_Fit = (void**)malloc(sizeof(void*) * 10);

      blitz::Array<double, 2> *P_D_A2_Sigma;
      I_KeywordSet_MeasureErrors = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "MEASURE_ERRORS_IN");
      if (I_KeywordSet_MeasureErrors >= 0)
      {
        P_D_A2_Sigma = (blitz::Array<double,2>*)ArgV_In[I_KeywordSet_MeasureErrors];
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A2_Sigma = " << *P_D_A2_Sigma << endl;
        #endif
        S_A1_Args_Fit(I_ArgPos) = "MEASURE_ERRORS_IN";
        I_ArgPos++;
      }

      blitz::Array<double, 1> *P_D_A1_ChiSq;
      I_KeywordSet_ChiSq = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "CHISQ_OUT");
      if (I_KeywordSet_ChiSq >= 0)
      {
        P_D_A1_ChiSq = (blitz::Array<double,1>*)ArgV_In[I_KeywordSet_ChiSq];
        P_D_A1_ChiSq->resize(D_A2_CCD_In.rows());
        S_A1_Args_Fit(I_ArgPos) = "CHISQ_OUT";
        I_ArgPos++;
      }

      blitz::Array<double, 1> *P_D_A1_Q;
      I_KeywordSet_Q = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "Q_OUT");
      if (I_KeywordSet_Q >= 0)
      {
        P_D_A1_Q = (blitz::Array<double,1>*)ArgV_In[I_KeywordSet_Q];
        P_D_A1_Q->resize(D_A2_CCD_In.rows());
        S_A1_Args_Fit(I_ArgPos) = "Q_OUT";
        I_ArgPos++;
      }

      blitz::Array<double, 2> *P_D_A2_Sigma_Out;
      I_KeywordSet_Sigma = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "SIGMA_OUT");
      if (I_KeywordSet_Sigma >= 0)
      {
        P_D_A2_Sigma_Out = (blitz::Array<double,2>*)ArgV_In[I_KeywordSet_Sigma];
        P_D_A2_Sigma_Out->resize(D_A2_CCD_In.rows(), 2);
        S_A1_Args_Fit(I_ArgPos) = "SIGMA_OUT";
        I_ArgPos++;
      }

      I_KeywordSet_YFit = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "YFIT_OUT");
      if (I_KeywordSet_YFit >= 0)
      {
        P_D_A2_YFit = (blitz::Array<double,2>*)ArgV_In[I_KeywordSet_YFit];
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington2D: P_D_A2_YFit = " << *P_D_A2_YFit << endl;
        #endif
        S_A1_Args_Fit(I_ArgPos) = "YFIT_OUT";
        I_ArgPos++;
      }
    
      I_KeywordSet_Reject = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "REJECT_IN");
      if (I_KeywordSet_Reject >= 0)
      {
        P_D_Reject = (double*)ArgV_In[I_KeywordSet_Reject];
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington2D: P_D_Reject = " << *P_D_Reject << endl;
        #endif
        S_A1_Args_Fit(I_ArgPos) = "REJECT_IN";
        I_ArgPos++;
      }
  
      I_KewordSet_Mask = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "MASK_INOUT");
      if (I_KewordSet_Mask >= 0)
      {
        P_I_A2_Mask = (blitz::Array<int,2>*)ArgV_In[I_KewordSet_Mask];
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington2D: P_I_A2_Mask = " << *P_I_A2_Mask << endl;
        #endif
        S_A1_Args_Fit(I_ArgPos) = "MASK_INOUT";
        I_ArgPos++;
      }

      bool B_DoFit = true;
      for (i = 0; i < D_A2_CCD_In.rows(); i++)
      {
        I_ArgPos = 0;
        if (I_KeywordSet_MeasureErrors >= 0){
          P_D_A1_Sigma = new blitz::Array<double,1>((*P_D_A2_Sigma)(i, blitz::Range::all()));
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A1_Sigma set to " << *P_D_A1_Sigma << endl;
          #endif
          PP_Args_Fit[I_ArgPos] = P_D_A1_Sigma;
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington(Array, Array, Array, Array): PP_Args_Fit[I_ArgPos=" << I_ArgPos << "] = " << *((blitz::Array<double,1>*)PP_Args_Fit[I_ArgPos]) << endl;
          #endif
          I_ArgPos++;
        }

        if (I_KeywordSet_ChiSq >= 0){
          PP_Args_Fit[I_ArgPos] = &((*P_D_A1_ChiSq)(i));
          I_ArgPos++;
        }

        if (I_KeywordSet_Q >= 0){
          PP_Args_Fit[I_ArgPos] = &((*P_D_A1_Q)(i));
          I_ArgPos++;
        }

        if (I_KeywordSet_Sigma >= 0){
          P_D_A1_Sigma_Out = new blitz::Array<double,1>((*P_D_A2_Sigma_Out)(i, blitz::Range::all()));
          PP_Args_Fit[I_ArgPos] = P_D_A1_Sigma_Out;
          I_ArgPos++;
        }

        if (I_KeywordSet_YFit >= 0){
          P_D_A1_YFit = new blitz::Array<double,1>((*P_D_A2_YFit)(i, blitz::Range::all()));
          PP_Args_Fit[I_ArgPos] = P_D_A1_YFit;
          I_ArgPos++;
        }

        if (I_KeywordSet_Reject >= 0){
          PP_Args_Fit[I_ArgPos] = P_D_Reject;
          I_ArgPos++;
        }

        B_DoFit = true;
        if (I_KewordSet_Mask >= 0){
          P_I_A1_Mask = new blitz::Array<int,1>((*P_I_A2_Mask)(i, blitz::Range::all()));
          PP_Args_Fit[I_ArgPos] = P_I_A1_Mask;
          I_ArgPos++;
          if (blitz::sum(*P_I_A1_Mask) == 0)
            B_DoFit = false;
        }

        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: Starting Fit1D: D_A2_CCD_In(i=" << i << ", *) = " << D_A2_CCD_In(i,blitz::Range::all()) << endl;
        #endif
        if (B_DoFit){
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: D_A2_SF_In(i=" << i << ", *) = " << D_A2_SF_In(i, blitz::Range::all()) << endl;
          #endif
          if (!pfsDRPStella::math::LinFitBevington(D_A2_CCD_In(i,blitz::Range::all()),
                                                   D_A2_SF_In(i,blitz::Range::all()),
                                                   D_A1_SP_Out(i),
                                                   D_A1_Sky_Out(i),
                                                   B_WithSky,
                                                   S_A1_Args_Fit,
                                                   PP_Args_Fit)){
            cout << "CFits::LinFitBevington: ERROR: LinFitBevington(D_A2_CCD_In(i,blitz::Range::all()),D_A2_SF_In(i,blitz::Range::all()),D_A1_SP_Out(i),D_A1_Sky_Out(i),D_A1_STDDEV_Out(i),D_A1_Covariance_Out(i)) returned false" << endl;
            free(PP_Args_Fit);
            cout << "CFits::LinFitBevington: D_A2_SF_In(0, *) = " << D_A2_SF_In(0,blitz::Range::all()) << endl;
            return false;
          }
        }

        I_ArgPos = 0;
        if (I_KeywordSet_MeasureErrors >= 0){
          (*P_D_A2_Sigma)(i, blitz::Range::all()) = (*P_D_A1_Sigma);
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A2_Sigma(i=" << i << ",*) set to " << (*P_D_A2_Sigma)(i, blitz::Range::all()) << endl;
          #endif
          I_ArgPos++;
          delete(P_D_A1_Sigma);/// or not
        }

        if (I_KeywordSet_Sigma >= 0){
          (*P_D_A2_Sigma_Out)(i, blitz::Range::all()) = (*P_D_A1_Sigma_Out);
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A2_Sigma_Out(i=" << i << ",*) set to " << (*P_D_A2_Sigma_Out)(i, blitz::Range::all()) << endl;
          #endif
          I_ArgPos++;
          delete(P_D_A1_Sigma_Out);// or not
        }

        if (I_KeywordSet_YFit >= 0){
          (*P_D_A2_YFit)(i, blitz::Range::all()) = (*P_D_A1_YFit);
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A2_YFit(i=" << i << ",*) set to " << (*P_D_A2_YFit)(i, blitz::Range::all()) << endl;
          #endif
          I_ArgPos++;
          delete(P_D_A1_YFit);// or not
        }

        if (I_KewordSet_Mask >= 0){
          (*P_I_A2_Mask)(i, blitz::Range::all()) = (*P_I_A1_Mask);
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_I_A1_Mask = " << (*P_I_A1_Mask) << endl;
            cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_I_A2_Mask(i=" << i << ",*) set to " << (*P_I_A2_Mask)(i, blitz::Range::all()) << endl;
          #endif
          I_ArgPos++;
          delete(P_I_A1_Mask);// or not
        }
      }
      free(PP_Args_Fit);
      return true;
    }

    /**
      Fit(blitz::Array<double, 1> y, const blitz::Array<double, 1> x, a1, a0);
      calculates a0 and a1 for the system of equations yvec = a0 + a1 * xvec
    **/
    bool LinFitBevington(const blitz::Array<double, 1> &D_A1_CCD_In,      /// yvec: in
                         const blitz::Array<double, 1> &D_A1_SF_In,       /// xvec: in
                         double &D_SP_Out,                         /// a1: out
                         double &D_Sky_Out,                        /// a0: in/out
                         bool B_WithSky,                        /// with sky: in
                         const blitz::Array<string, 1> &S_A1_Args_In,   ///: in
                         void *ArgV_In[])                    ///: in
    /// MEASURE_ERRORS_IN = blitz::Array<double,1>(D_A1_CCD_In.size)
    /// REJECT_IN = double
    /// MASK_INOUT = blitz::Array<double,1>(D_A1_CCD_In.size)
    /// CHISQ_OUT = double
    /// Q_OUT = double
    /// SIGMA_OUT = blitz::Array<double,1>(2): [0]: sigma_sp, [1]: sigma_sky
    /// YFIT_OUT = blitz::Array<double, 1>(D_A1_CCD_In.size)
    /// ALLOW_SKY_LT_ZERO = 1
    /// ALLOW_SPEC_LT_ZERO = 1
    {
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington(Array, Array, double, double, bool, CSArr, PPArr) started" << endl;
        cout << "CFits::LinFitBevington: D_A1_CCD_In = " << D_A1_CCD_In << endl;
        cout << "CFits::LinFitBevington: D_A1_SF_In = " << D_A1_SF_In << endl;
      #endif

      if (D_A1_CCD_In.size() != D_A1_SF_In.size()){
        cout << "CFits::LinFitBevington: ERROR: D_A1_CCD_In.size(=" << D_A1_CCD_In.size() << ") != D_A1_SF_In.size(=" << D_A1_SF_In.size() << ") => returning false" << endl;
        D_SP_Out = 0.;
        D_Sky_Out = 0.;
        return false;
      }

      //  /// Set D_A1_SF_In to zero where D_A1_CCD_In == zero
      blitz::Array<double, 1> D_A1_SF(D_A1_SF_In.size());
      D_A1_SF = D_A1_SF_In;//(fabs(D_A1_CCD_In) < 0.000000001, 0., D_A1_SF_In);
      blitz::Array<double, 1> D_A1_CCD(D_A1_CCD_In.size());
      D_A1_CCD = D_A1_CCD_In;//(fabs(D_A1_CCD_In) < 0.000000001, 0., D_A1_SF_In);

      if (blitz::sum(D_A1_CCD_In) == 0. || blitz::sum(D_A1_SF) == 0.){
        cout << "CFits::LinFitBevington: Warning: (blitz::sum(D_A1_CCD_In)=" << blitz::sum(D_A1_CCD_In) << " == 0. || blitz::sum(D_A1_SF)=" << blitz::sum(D_A1_SF) << " == 0.) => returning false" << endl;
        D_SP_Out = 0.;
        D_Sky_Out = 0.;
        return true;
      }
      int i, I_Pos;
      int I_KeywordSet_Reject, I_KeywordSet_Mask, I_KeywordSet_MeasureErrors, I_KeywordSet_SigmaOut, I_KeywordSet_ChiSqOut, I_KeywordSet_QOut, I_KeywordSet_YFitOut, I_KeywordSet_AllowSkyLTZero, I_KeywordSet_AllowSpecLTZero;
      double sigdat;
      int ndata = D_A1_CCD_In.size();
      blitz::Array<double, 1> *P_D_A1_Sig = new blitz::Array<double, 1>(D_A1_CCD_In.size());
      (*P_D_A1_Sig) = 0.;
      blitz::Array<double, 1> D_A1_Sig(D_A1_CCD_In.size());
      blitz::Array<double, 1> D_A1_WT(ndata);

      /// a: D_Sky_Out
      /// b: D_SP_Out
      /// x: D_A1_SF_In
      /// y: D_A1_CCD_In

      int *P_I_TempInt = new int(0);

      bool B_AllowSkyLTZero = false;
      I_KeywordSet_AllowSkyLTZero = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "ALLOW_SKY_LT_ZERO");
      if (I_KeywordSet_AllowSkyLTZero >= 0)
      {
        delete(P_I_TempInt);
        P_I_TempInt = (int*)ArgV_In[I_KeywordSet_AllowSkyLTZero];

        if (*P_I_TempInt > 0){
          B_AllowSkyLTZero = true;
          cout << "CFits::LinFitBevington: KeyWord_Set(ALLOW_SKY_LT_ZERO)" << endl;
        }
      }

      bool B_AllowSpecLTZero = false;
      I_KeywordSet_AllowSpecLTZero = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "ALLOW_SPEC_LT_ZERO");
      if (I_KeywordSet_AllowSpecLTZero >= 0)
      {
        if (I_KeywordSet_AllowSkyLTZero < 0)
          delete(P_I_TempInt);
        P_I_TempInt = (int*)ArgV_In[I_KeywordSet_AllowSkyLTZero];
  
        if (*P_I_TempInt > 0){
          B_AllowSpecLTZero = true;
          cout << "CFits::LinFitBevington: KeyWord_Set(ALLOW_SPEC_LT_ZERO)" << endl;
        }
      }

      double *P_D_Reject = new double(-1.);
      I_KeywordSet_Reject = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "REJECT_IN");
      if (I_KeywordSet_Reject >= 0)
      {
        delete(P_D_Reject);
        P_D_Reject = (double*)ArgV_In[I_KeywordSet_Reject];
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: KeyWord_Set(REJECT_IN): *P_D_Reject = " << *P_D_Reject << endl;
        #endif
      }
      bool B_Reject = false;
      if (*P_D_Reject > 0.)
        B_Reject = true;
    
      blitz::Array<int, 1> *P_I_A1_Mask = new blitz::Array<int, 1>(D_A1_CCD_In.size());
      blitz::Array<int, 1> I_A1_Mask_Orig(D_A1_CCD_In.size());
      *P_I_A1_Mask = 1;

      I_KeywordSet_Mask = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "MASK_INOUT");
      if (I_KeywordSet_Mask >= 0)
      {
        delete(P_I_A1_Mask);
        P_I_A1_Mask = (blitz::Array<int, 1>*)ArgV_In[I_KeywordSet_Mask];
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: KeyWord_Set(MASK_INOUT): *P_I_A1_Mask = " << *P_I_A1_Mask << endl;
        #endif
      }
      I_A1_Mask_Orig = (*P_I_A1_Mask);
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: *P_I_A1_Mask set to " << *P_I_A1_Mask << endl;
        cout << "CFits::LinFitBevington: I_A1_Mask_Orig set to " << I_A1_Mask_Orig << endl;
      #endif

      blitz::Array<double, 1> *P_D_A1_Sigma_Out;
      blitz::Array<double, 1> D_A1_Sigma_Out(2);
      I_KeywordSet_SigmaOut = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "SIGMA_OUT");
      if (I_KeywordSet_SigmaOut >= 0)
      {
        P_D_A1_Sigma_Out = (blitz::Array<double, 1>*)ArgV_In[I_KeywordSet_SigmaOut];
        P_D_A1_Sigma_Out->resize(2);
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: KeyWord_Set(SIGMA_OUT)" << endl;
        #endif
      }
      else
      {
        P_D_A1_Sigma_Out = new blitz::Array<double, 1>(2);
      }
      *P_D_A1_Sigma_Out = 0.;
      D_A1_Sigma_Out = *P_D_A1_Sigma_Out;
    
      double* P_D_ChiSqr_Out;
      I_KeywordSet_ChiSqOut = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "CHISQ_OUT");
      if (I_KeywordSet_ChiSqOut >= 0)
      {
        P_D_ChiSqr_Out = (double*)ArgV_In[I_KeywordSet_ChiSqOut];
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: KeyWord_Set(CHISQ_OUT)" << endl;
        #endif
      }
      else
      {
        P_D_ChiSqr_Out = new double();
      }
      *P_D_ChiSqr_Out = 0.;

      double* P_D_Q_Out;
      I_KeywordSet_QOut = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "Q_OUT");
      if (I_KeywordSet_QOut >= 0)
      {
        P_D_Q_Out = (double*)ArgV_In[I_KeywordSet_QOut];
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: KeyWord_Set(Q_OUT)" << endl;
        #endif
      }
      else
      {
        P_D_Q_Out = new double();
      }
      *P_D_Q_Out = 1.;
    
      D_SP_Out = 0.0;
      I_KeywordSet_MeasureErrors = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "MEASURE_ERRORS_IN");
      if (I_KeywordSet_MeasureErrors >= 0)
      {
        /// Accumulate sums...
        delete(P_D_A1_Sig);
        P_D_A1_Sig = (blitz::Array<double, 1>*)ArgV_In[I_KeywordSet_MeasureErrors];
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: ArgV_In[I_KeywordSet_MeasureErrors=" << I_KeywordSet_MeasureErrors << "] = " << *((blitz::Array<double,1>*)ArgV_In[I_KeywordSet_MeasureErrors]) << endl;
          cout << "CFits::LinFitBevington: *P_D_A1_Sig = " << *P_D_A1_Sig << endl;
        #endif
        if (D_A1_CCD_In.size() != P_D_A1_Sig->size()){
          cout << "CFits::LinFitBevington: ERROR: D_A1_CCD_In.size(=" << D_A1_CCD_In.size() << ") != P_D_A1_Sig->size(=" << P_D_A1_Sig->size() << ") => returning false" << endl;
          D_SP_Out = 0.;
          D_Sky_Out = 0.;
          return false;
        }
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: KeyWord_Set(MEASURE_ERRORS_IN): *P_D_A1_Sig = " << *P_D_A1_Sig << endl;
        #endif
      }

      blitz::Array<double, 1> D_A1_YFit(1);
      blitz::Array<double, 1> *P_D_A1_YFit = new blitz::Array<double, 1>(D_A1_CCD_In.size());
      I_KeywordSet_YFitOut = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "YFIT_OUT");
      if (I_KeywordSet_YFitOut >= 0)
      {
        delete(P_D_A1_YFit);
        P_D_A1_YFit = (blitz::Array<double, 1>*)ArgV_In[I_KeywordSet_YFitOut];
        P_D_A1_YFit->resize(D_A1_CCD_In.size());
        (*P_D_A1_YFit) = 0.;
      }
      if (blitz::sum(*P_I_A1_Mask) == 0){
        cout << "CFits::LinFitBevington: WARNING: blitz::sum(*P_I_A1_Mask = " << *P_I_A1_Mask << ") == 0" << endl;
        D_SP_Out = 0.;
        D_Sky_Out = 0.;
        return true;
      }

      int I_SumMaskLast;
      double D_SDevReject;
      blitz::Array<double, 1> D_A1_Check(D_A1_CCD_In.size());
      blitz::Array<int, 1> I_A1_LastMask(P_I_A1_Mask->size());
      blitz::Array<double, 1> D_A1_Diff(D_A1_CCD_In.size());
      D_A1_Diff = 0.;
      double D_Sum_Weights = 0.;
      double D_Sum_XSquareTimesWeight = 0;
      double D_Sum_XTimesWeight = 0.;
      double D_Sum_YTimesWeight = 0.;
      double D_Sum_XYTimesWeight = 0.;
      double D_Delta = 0.;
    
      bool B_Run = true;
      int I_Run = -1;
      int I_MaskSum;
      while (B_Run){
        D_SP_Out = 0.0;
  
        I_Run++;
        /// remove bad pixels marked by mask
        I_MaskSum = blitz::sum(*P_I_A1_Mask);
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": I_MaskSum = " << I_MaskSum << endl;
        #endif
        D_A1_Sig.resize(I_MaskSum);
        D_A1_CCD.resize(I_MaskSum);
        D_A1_SF.resize(I_MaskSum);
        D_A1_WT.resize(I_MaskSum);
        D_A1_YFit.resize(I_MaskSum);
        D_A1_Sig = 0.;
        D_A1_CCD = 0.;
        D_A1_SF = 0.;
        D_A1_WT = 0.;
        D_A1_YFit = 0.;

        I_Pos = 0;
        for (unsigned int ii = 0; ii < P_I_A1_Mask->size(); ii++){
          if ((*P_I_A1_Mask)(ii) == 1){
            D_A1_CCD(I_Pos) = D_A1_CCD_In(ii);
            D_A1_SF(I_Pos) = D_A1_SF_In(ii);
            D_A1_Sig(I_Pos) = (*P_D_A1_Sig)(ii);
            I_Pos++;
          }
        }
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_CCD set to " << D_A1_CCD << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_SF set to " << D_A1_SF << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_Sig set to " << D_A1_Sig << endl;
        #endif

        D_Sum_Weights = 0.;
        D_Sum_XSquareTimesWeight = 0.;
        D_Sum_XTimesWeight = 0.;
        D_Sum_XYTimesWeight = 0.;
        D_Sum_YTimesWeight = 0.;
        if (I_KeywordSet_MeasureErrors >= 0)
        {
          ///    D_A1_WT = D_A1_SF;
          for (i=0; i < I_MaskSum; i++)
          {
            /// ... with weights
            if (fabs(D_A1_Sig(i)) < 0.00000000000000001){
              cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": ERROR: D_A1_Sig = " << D_A1_Sig << endl;
              cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": ERROR: D_A1_Sig(" << i << ") == 0. => Returning FALSE" << endl;
              return false;
            }
            D_A1_WT(i) = 1. / blitz::pow2(D_A1_Sig(i));
          }
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ":: D_A1_WT set to " << D_A1_WT << endl;
          #endif
          for (i=0; i < I_MaskSum; i++)
          {
            D_Sum_Weights += D_A1_WT(i);
            D_Sum_XTimesWeight += D_A1_SF(i) * D_A1_WT(i);
            D_Sum_YTimesWeight += D_A1_CCD(i) * D_A1_WT(i);
            D_Sum_XYTimesWeight += D_A1_SF(i) * D_A1_CCD(i) * D_A1_WT(i);
            D_Sum_XSquareTimesWeight += D_A1_SF(i) * D_A1_SF(i) * D_A1_WT(i);
          }
        }
        else
        {
          for (i = 0; i < I_MaskSum; i++)
          {
            /// ... or without weights
            D_Sum_XTimesWeight += D_A1_SF(i);
            D_Sum_YTimesWeight += D_A1_CCD(i);
            D_Sum_XYTimesWeight += D_A1_SF(i) * D_A1_CCD(i);
            D_Sum_XSquareTimesWeight += D_A1_SF(i) * D_A1_SF(i);
          }
          D_Sum_Weights = I_MaskSum;
        }
        D_Delta = D_Sum_Weights * D_Sum_XSquareTimesWeight - blitz::pow2(D_Sum_XTimesWeight);

        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_Weights set to " << D_Sum_Weights << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XTimesWeight set to " << D_Sum_XTimesWeight << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_YTimesWeight set to " << D_Sum_YTimesWeight << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XYTimesWeight set to " << D_Sum_XYTimesWeight << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XSquareTimesWeight set to " << D_Sum_XSquareTimesWeight << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Delta set to " << D_Delta << endl;
        #endif


        if (!B_WithSky)
        {
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out < 0. = setting D_Sky_Out to 0 " << endl;
          #endif
          D_SP_Out = D_Sum_XYTimesWeight / D_Sum_XSquareTimesWeight;
          D_Sky_Out = 0.0;
        }
        else
        {
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out >= 0." << D_Sky_Out << endl;
          #endif
          D_Sky_Out = ((D_Sum_XSquareTimesWeight * D_Sum_YTimesWeight) - (D_Sum_XTimesWeight * D_Sum_XYTimesWeight)) / D_Delta;

          D_SP_Out = ((D_Sum_Weights * D_Sum_XYTimesWeight) - (D_Sum_XTimesWeight * D_Sum_YTimesWeight)) / D_Delta;
          (*P_D_A1_Sigma_Out)(0) = sqrt(D_Sum_Weights / D_Delta);
          (*P_D_A1_Sigma_Out)(1) = sqrt(D_Sum_XSquareTimesWeight / D_Delta);
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_SP_Out set to " << D_SP_Out << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out set to " << D_Sky_Out << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": P_D_A1_Sigma_Out(0) set to " << (*P_D_A1_Sigma_Out)(0) << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": P_D_A1_Sigma_Out(1) set to " << (*P_D_A1_Sigma_Out)(1) << endl;
          #endif
        }
        if ((!B_AllowSpecLTZero) && (D_SP_Out < 0.))
          D_SP_Out = 0.;

        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out set to " << D_Sky_Out << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_SP_Out set to " << D_SP_Out << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": fabs(D_SP_Out) = " << fabs(D_SP_Out) << endl;
        #endif

        *P_D_A1_YFit = D_Sky_Out + D_SP_Out * D_A1_SF_In;
        D_A1_YFit = D_Sky_Out + D_SP_Out * D_A1_SF;
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": *P_D_A1_YFit set to " << *P_D_A1_YFit << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_YFit set to " << D_A1_YFit << endl;
        #endif
        *P_D_ChiSqr_Out = 0.;
        if (I_KeywordSet_MeasureErrors < 0)
        {
          for (i = 0; i < I_MaskSum; i++)
          {
            *P_D_ChiSqr_Out += blitz::pow2(D_A1_CCD(i) - D_A1_YFit(i));
            #ifdef __DEBUG_FIT__
              cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": P_D_ChiSqr_Out set to " << *P_D_ChiSqr_Out << endl;
            #endif
          }

          /// for unweighted data evaluate typical sig using chi2, and adjust the standard deviations
          if (I_MaskSum == 2){
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": ERROR: Sum of Mask (=" << I_MaskSum << ") must be greater than 2 => Returning FALSE" << endl;
            return false;
          }
          sigdat = sqrt((*P_D_ChiSqr_Out) / (I_MaskSum - 2));
          (*P_D_A1_Sigma_Out)(0) *= sigdat;
          (*P_D_A1_Sigma_Out)(1) *= sigdat;
        }
        else
        {
          for (i = 0; i < I_MaskSum; i++)
          {
            #ifdef __DEBUG_FIT__
              cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_CCD(" << i << ") = " << D_A1_CCD(i) << endl;
              cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_SF(" << i << ") = " << D_A1_SF(i) << endl;
              cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_Sig(" << i << ") = " << D_A1_Sig(i) << endl;
              cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_YFit(" << i << ") = " << D_A1_YFit(i) << endl;
            #endif
            if (abs(D_A1_Sig(i)) < 0.00000000000000001){
              cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": ERROR: D_A1_Sig(" << i << ") == 0. => Returning FALSE" << endl;
              return false;
            }
            *P_D_ChiSqr_Out += pow((D_A1_CCD(i) - D_A1_YFit(i)) / D_A1_Sig(i),2);
            #ifdef __DEBUG_FIT__
              cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": P_D_ChiSqr_Out set to " << *P_D_ChiSqr_Out << endl;
            #endif
          }
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": P_D_ChiSqr_Out set to " << *P_D_ChiSqr_Out << endl;
          #endif
          if (I_MaskSum > 2)
          {
            if (!pfsDRPStella::math::GammQ(0.5 * (I_MaskSum - 2), 0.5 * (*P_D_ChiSqr_Out), P_D_Q_Out))
            {
              cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": ERROR: GammQ returned FALSE" << endl;
              return false;
            }
          }
        }
        if (fabs(D_SP_Out) < 0.000001)
          B_Reject = false;
        if (!B_Reject)
          B_Run = false;
        else{

          I_SumMaskLast = blitz::sum(*P_I_A1_Mask);
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: I_SumMaskLast = " << I_SumMaskLast << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_CCD = " << D_A1_CCD << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_YFit = " << D_A1_YFit << endl;
          #endif
          D_SDevReject = sqrt(blitz::sum(blitz::pow2(D_A1_CCD - D_A1_YFit)) / double(I_SumMaskLast));//(blitz::sum(pow(D_A1_CCD - (D_A1_YFit),2)) / I_SumMaskLast);
      
          /// NOTE: Should be square! Test!!!
          D_A1_Diff = D_A1_CCD_In - (*P_D_A1_YFit);
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_SDevReject = " << D_SDevReject << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_CCD_In = " << D_A1_CCD_In << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: *P_D_A1_YFit = " << *P_D_A1_YFit << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_CCD_In - (*P_D_A1_YFit) = " << D_A1_Diff << endl;
          #endif
          D_A1_Check = fabs(D_A1_Diff);
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_Check = " << D_A1_Check << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": before Reject: *P_I_A1_Mask = " << *P_I_A1_Mask << endl;
          #endif
          I_A1_LastMask = *P_I_A1_Mask;
          *P_I_A1_Mask = blitz::where(D_A1_Check > (*P_D_Reject) * D_SDevReject, 0, 1);
          *P_I_A1_Mask = blitz::where(I_A1_Mask_Orig < 1, 0, *P_I_A1_Mask);
          if (blitz::sum(*P_I_A1_Mask) == blitz::sum(I_A1_Mask_Orig))
            B_Reject = false;
          else
            *P_I_A1_Mask = blitz::where(I_A1_LastMask < 1, 0, *P_I_A1_Mask);
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: *P_I_A1_Mask = " << *P_I_A1_Mask << endl;
          #endif
          if (I_SumMaskLast == blitz::sum(*P_I_A1_Mask)){
            B_Run = false;
            #ifdef __DEBUG_FIT__
              cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": leaving while loop" << endl;
            #endif
          }
          else{
            D_Sky_Out = 0.;
          }
        }
        if ((!B_AllowSkyLTZero) && (D_Sky_Out < 0.)){
          B_Run = true;
          B_WithSky = false;
        }
      }/// end while (B_Run)

      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: *P_D_A1_YFit set to " << *P_D_A1_YFit << endl;
        cout << "CFits::LinFitBevington: *P_I_A1_Mask set to " << *P_I_A1_Mask << endl;
      #endif


      /// clean up
      if (I_KeywordSet_Mask < 0)
      {
        delete(P_I_A1_Mask);
      }
      if (I_KeywordSet_Reject < 0)
      {
        delete(P_D_Reject);
      }
      if ((I_KeywordSet_AllowSkyLTZero < 0) && (I_KeywordSet_AllowSpecLTZero < 0)){
        delete(P_I_TempInt);
      }
      if (I_KeywordSet_MeasureErrors < 0)
      {
        delete(P_D_A1_Sig);
      }
      if (I_KeywordSet_ChiSqOut < 0)
      {
        delete(P_D_ChiSqr_Out);
      }
      if (I_KeywordSet_QOut < 0)
      {
        delete(P_D_Q_Out);
      }
      if (I_KeywordSet_SigmaOut < 0)
      {
        delete(P_D_A1_Sigma_Out);
      }
      if (I_KeywordSet_YFitOut < 0){
        delete(P_D_A1_YFit);
      }

      return true;
    }
    
    /**
     * Helper function to calculate incomplete Gamma Function
     **/
    bool GSER(double *P_D_Gamser_Out, double a, double x, double *P_D_GLn_Out)
    {
      int n;
      int ITMax = 100;
      double d_sum, del, ap;
      
      #ifdef __DEBUG_LINFIT__
      cout << "CFits::GSER: *P_D_Gamser_Out = " << *P_D_Gamser_Out << endl;
      cout << "CFits::GSER: a = " << a << endl;
      cout << "CFits::GSER: x = " << x << endl;
      #endif
      
      *P_D_GLn_Out = GammLn(a);
      #ifdef __DEBUG_LINFIT__
      cout << "CFits::GSER: *P_D_GLn_Out = " << *P_D_GLn_Out << endl;
      #endif
      if (x <= 0.){
        if (x < 0.){
          cout << "CFits::GSER: ERROR: x less than 0!" << endl;
          return false;
        }
        *P_D_Gamser_Out = 0.;
        #ifdef __DEBUG_LINFIT__
        cout << "CFits::GSER: x<=0: *P_D_Gamser_Out = " << *P_D_Gamser_Out << endl;
        cout << "CFits::GSER: x<=0: *P_D_GLn_Out = " << *P_D_GLn_Out << endl;
        #endif
        return true;
      }
      else{
        ap = a;
        del = d_sum = 1. / a;
        for (n=1; n <= ITMax; n++){
          ++ap;
          del *= x/ap;
          d_sum += del;
          if (fabs(del) < fabs(d_sum) * 3.e-7){
            *P_D_Gamser_Out = d_sum * exp(-x+a*log(x) - (*P_D_GLn_Out));
            #ifdef __DEBUG_LINFIT__
            cout << "CFits::GSER: x>0: *P_D_Gamser_Out = " << *P_D_Gamser_Out << endl;
            cout << "CFits::GSER: x>0: *P_D_GLn_Out = " << *P_D_GLn_Out << endl;
            #endif
            return true;
          }
        }
        cout << "CFits::GSER: ERROR: a too large, ITMax too small in routine GSER" << endl;
        return false;
      }
    }
    
    /**
     * Helper function to calculate incomplete Gamma Function
     **/
    double GammLn(double xx)
    {
      double x,y,tmp,ser;
      static double cof[6]={76.18009172947146, -86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5};
      
      #ifdef __DEBUG_LINFIT__
        cout << "CFits::GammLn: xx = " << xx << endl;
      #endif
      
      y = x = xx;
      tmp = x + 5.5;
      tmp -= (x+0.5) * log(tmp);
      #ifdef __DEBUG_LINFIT__
        cout << "CFits::GammLn: tmp = " << tmp << endl;
      #endif
      ser = 1.000000000190015;
      for (int o = 0; o <= 5; o++){
        ser += cof[o] / ++y;
      }
      double D_Result = (-tmp + log(2.5066282746310005 * ser / x));
      #ifdef __DEBUG_LINFIT__
        cout << "CFits::GammLn: ser = " << ser << endl;
        cout << "CFits::GammLn: returning (-tmp + log(2.5066282746310005 * ser / xx)) = " << D_Result << endl;
      #endif
      return D_Result;
    }
    
    /**
     * Helper function to calculate incomplete Gamma Function
     **/
    bool GCF(double *P_D_GammCF_Out, double a, double x, double *P_D_GLn_Out)
    {
      int n;
      int ITMAX = 100;             /// Maximum allowed number of iterations
      double an, b, c, d, del, h;
      double FPMIN = 1.0e-30;      /// Number near the smallest representable floating-point number
      double EPS = 1.0e-7;         /// Relative accuracy
      
      *P_D_GLn_Out = GammLn(a);
      #ifdef __DEBUG_FIT__
        cout << "CFits::GCF: P_D_GLn_Out set to " << *P_D_GLn_Out << endl;
      #endif
      
      b = x + 1. - a;
      #ifdef __DEBUG_FIT__
        cout << "CFits::GCF: x=" << x << ", a=" << a << ": b set to " << b << endl;
      #endif
      c = 1. / FPMIN;
      #ifdef __DEBUG_FIT__
        cout << "CFits::GCF: c set to " << c << endl;
      #endif
      d = 1. / b;
      #ifdef __DEBUG_FIT__
        cout << "CFits::GCF: d set to " << d << endl;
      #endif
      h = d;
      for (n=1; n <= ITMAX; n++){
        an = -n * (n - a);
        #ifdef __DEBUG_FIT__
          cout << "CFits::GCF: n = " << n << ": an set to " << an << endl;
        #endif
        b += 2.;
        #ifdef __DEBUG_FIT__
          cout << "CFits::GCF: n = " << n << ": b set to " << b << endl;
        #endif
        d = an * d + b;
        #ifdef __DEBUG_FIT__
          cout << "CFits::GCF: n = " << n << ": d set to " << d << endl;
        #endif
        if (fabs(d) < FPMIN)
          d = FPMIN;
        c = b + an / c;
        #ifdef __DEBUG_FIT__
          cout << "CFits::GCF: n = " << n << ": c set to " << c << endl;
        #endif
        if (fabs(c) < FPMIN)
          c = FPMIN;
        d = 1. / d;
        #ifdef __DEBUG_FIT__
          cout << "CFits::GCF: n = " << n << ": d set to " << d << endl;
        #endif
        del = d * c;
        #ifdef __DEBUG_FIT__
          cout << "CFits::GCF: n = " << n << ": del set to " << del << endl;
        #endif
        
        h *= del;
        if (fabs(del-1.) < EPS)
          break;
      }
      if (n > ITMAX){
        cout << "CFits::GCF: ERROR: a too large, ITMAX too small in GCF" << endl;
        return false;
      }
      *P_D_GammCF_Out = exp(-x+a*log(x) - (*P_D_GLn_Out)) * h;
      return true;
    }
    
    /**
     * Function to calculate incomplete Gamma Function P(a,x)
     **/
    bool GammP(double a, double x, double* D_Out){
      #ifdef __DEBUG_FIT__
        cout << "CFits::GammP started: a = " << a << ", x = " << x << endl;
      #endif
      double gamser, gammcf, gln;
      if (x < 0. || a <= 0.){
        cout << "CFits::GammP: ERROR: Invalid arguments in routine GammP" << endl;
        return false;
      }
      if (x < (a+1.)){
        if (!GSER(&gamser, a, x, &gln)){
          cout << "CFits::GammP: ERROR: GSER returned FALSE" << endl;
          return false;
        }
        *D_Out = gamser;
        return true;
      }
      else{
        if (!GCF(&gammcf, a, x, &gln))
        {
          cout << "CFits::GammP: ERROR: GCF returned FALSE" << endl;
          return false;
        }
        *D_Out = 1. - gammcf;
        return true;
      }
    }
    
    /**
     * Function to calculate incomplete Gamma Function Q(a,x) = 1. - P(a,x)
     **/
    bool GammQ(double a, double x, double* D_Out){
      #ifdef __DEBUG_FIT__
      cout << "CFits::GammQ started: a = " << a << ", x = " << x << endl;
      #endif
      double gamser = 0.;
      double gammcf = 0.;
      double gln = 0.;
      if (x < 0. || a <= 0.){
        cout << "CFits::GammQ: ERROR: Invalid arguments in routine GammQ" << endl;
        return false;
      }
      if (x < (a+1.)){
        if (!GSER(&gamser, a, x, &gln)){
          cout << "CFits::GammQ: ERROR: GSER returned FALSE" << endl;
          return false;
        }
        #ifdef __DEBUG_FIT__
        cout << "CFits::GammQ: x < (a+1.): gamser = " << gamser << endl;
        #endif
        *D_Out = 1. - gamser;
        return true;
      }
      else{
        if (!GCF(&gammcf, a, x, &gln))
        {
          cout << "CFits::GammQ: ERROR: GCF returned false" << endl;
          return false;
        }
        #ifdef __DEBUG_FIT__
        cout << "CFits::GammQ: x < (a+1.): gammcf = " << gammcf << endl;
        #endif
        *D_Out = gammcf;
        return true;
      }
    }
    
    template<typename T>
    blitz::Array<T, 1> Replicate(T Val, int Len)
    {
      blitz::Array<T, 1> TempVecArr(Len);
      TempVecArr = Val;
      return (TempVecArr);
    }

    /**
    double Median(blitz::Array<double, int>) const;
    **/
    template<typename T>
    T Median(const blitz::Array<T, 1> &Arr)
    {
      blitz::Array<string, 1> S_A1_Args_Median(1);
      void **PP_Args_Median;
      PP_Args_Median = (void**)malloc(sizeof(void*) * 1);

      S_A1_Args_Median = " ";

      S_A1_Args_Median(0) = "MODE";
      string Mode = "NORMAL";
      PP_Args_Median[0] = &Mode;

      T T_Out = pfsDRPStella::math::Median(Arr, S_A1_Args_Median, PP_Args_Median);
      free(PP_Args_Median);
      return T_Out;
    }

    /**
    double Median(blitz::Array<double, 2>) const;
    **/
    template<typename T>
    T Median(const blitz::Array<T, 2> &Arr, bool B_IgnoreZeros)
    {
      blitz::Array<T, 1> D_A1_Arr(Arr.rows() * Arr.cols());
      int I_Pos=0;
      int I_N_NoZeros = 0;
      for (int i_row=0; i_row<Arr.rows(); i_row++){
        for (int i_col=0; i_col<Arr.cols(); i_col++){
          if ((B_IgnoreZeros && (fabs(double(Arr(i_row, i_col))) > 0.0000001)) || (!B_IgnoreZeros)){
            D_A1_Arr(I_Pos) = Arr(i_row, i_col);
            I_Pos++;
            I_N_NoZeros++;
          }
        }
      }
      if (I_N_NoZeros > 0){
        D_A1_Arr.resizeAndPreserve(I_N_NoZeros);
        return (pfsDRPStella::math::Median(D_A1_Arr));
      }
      else{
        return T(0);
      }
    }

    /**
      T Median(blitz::Array<T, int> &Arr, CString &Mode) const;
      Args: MODE(CString)
      //        ERRORS_IN(blitz::Array<double, 1>)
//            ERR_OUT(double)
    **/
    template<typename T>
    T Median(const blitz::Array<T, 1> &Arr, 
             const blitz::Array<string, 1> &S_A1_Args_In, 
             void *PP_Args_In[])
    {
      int Length = Arr.size();
      T median;
      string Mode = "NORMAL";
      //double *P_D_ErrOut = new double(0.);

      int I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "MODE");
      if (I_Pos >= 0)
        Mode = *(string*)PP_Args_In[I_Pos];

//      I_Pos = pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "ERR_OUT");
//      if (I_Pos >= 0){
//        delete(P_D_ErrOut);
//        P_D_ErrOut = (double*)PP_Args_In[I_Pos];
//      }
//      else{
//        cout << "CFits::Median: ERROR: KeyWord 'ERRORS_IN' set, but 'ERR_OUT' is not" << endl;
//        return 0.;
//      }

      /** Odd Array **/
      if (pfsDRPStella::math::IsOddNumber(Length))
      {
        median = pfsDRPStella::math::Select(Arr, (int)((double)Length / 2.)+1);
        #ifdef __DEBUG_MEDIAN__
          cout << "CFits::Median(blitz::Array<double, int Length = " << Length << ">, Mode = " << Mode << "): IsOddNumber: median(Arr=" << Arr << ") from Select(" << (int)((double)Length / 2.) + 1 << ") = " << median << endl;
        #endif
      }
      else /** Even Array **/
      {
        /** Return Mean of both Numbers next to Length / 2. **/
        if (Mode.compare("NORMAL") == 0)
        {
          median = ((pfsDRPStella::math::Select(Arr, (int)((double)Length / 2.))) +
                    (pfsDRPStella::math::Select(Arr, (int)((double)Length / 2.) + 1))) / 2.;
          #ifdef __DEBUG_MEDIAN__
            cout << "CFits::Median(blitz::Array<double, int Length = " << Length << ">, Mode = " << Mode << "): !IsOddNumber: mean of medians(" << pfsDRPStella::math::Select(Arr, (int)((double)Length / 2.)) << " and " << pfsDRPStella::math::Select(Arr, (int)((double)Length / 2.) + 1) << ") from Select() = " << median << endl;
          #endif
        }
        else/** Return Number lower next to Length / 2. **/
        {
          median = pfsDRPStella::math::Select(Arr, (int)((double)Length / 2.));
          #ifdef __DEBUG_MEDIAN__
            cout << "CFits::Median(blitz::Array<double, int Length = " << Length << ">, Mode = " << Mode << "): !IsOddNumber: median from Select(" << (int)((double)Length / 2.) << ") = " << median << endl;
          #endif
        }
      }

//      if (pfsDRPStella::utils::KeyWord_Set(S_A1_Args_In, "ERRORS_IN") >= 0){
//        *P_D_ErrOut = 0.;
//        for (unsigned int i=0; i<Arr.size(); i++){
//          *P_D_ErrOut += blitz::pow2((Arr)(i) - median);
//        }
//        *P_D_ErrOut = sqrt(*P_D_ErrOut) / Arr.size();
//      }

      return (median);
    }
    
//    template<typename T>
//    blitz::Array<T, 1> MedianVec(const blitz::Array<T, 1> &VecArr, 
//                                 int Width)
//    {
//      std::string mode = "NORMAL";
//      return pfsDRPStella::math::MedianVec(VecArr, Width, mode);
//    }
    
    template<typename T>
    blitz::Array<T, 1> MedianVec(const blitz::Array<T, 1> &VecArr, 
                                 int Width, 
                                 const std::string &Mode=std::string("NORMAL"))
    {
      #ifdef __DEBUG_MEDIAN__
        cout << "CFits::MedianVec: VecArr = " << VecArr << endl;
      #endif
      //  CString *P_TempMode = new CString(Mode);
      blitz::Array<T, 1> TempVecArr(VecArr.size());
      TempVecArr = VecArr;
      #ifdef __DEBUG_MEDIAN__
        cout << "CFits::MedianVec: TempVecArr = " << TempVecArr << endl;
      #endif
      int              m;
      int              Start, End, Length;
      bool             Odd;
      blitz::Array<string, 1> S_A1_Args_Median(1);
      S_A1_Args_Median = " ";
      S_A1_Args_Median(0) = "MODE";
      void **PP_Args_Median = (void**)malloc(sizeof(void*) * 1);
      PP_Args_Median[0] = (void*)(&Mode);

      Length = VecArr.size();
      Odd = pfsDRPStella::math::IsOddNumber(Width);
  //  if (Odd)
  //    (*P_TempMode).Set("NORMAL");

      if (Width < 2)
        return (TempVecArr);
      /** Calculate Median for every Pixel**/
      blitz::Array<T, 1> TmpVecArr(Length);
      TmpVecArr = TempVecArr;
      for (m = Width/2; m < Length - Width/2; m++)
      {
        /** Check Start end End Indizes **/
        Start = m - (int)((Width) / 2.);
        End   = m + (int)((Width-1) / 2.);
        if (Start < 0)
          Start = 0;
        if (End > Length - 1)
          End = Length - 1;
        #ifdef __DEBUG_MEDIAN__
          cout << "CFits::MedianVec: run m = " << m << ": Start = " << Start << ", End = " << End << endl;
        #endif
        blitz::Range tempRange(Start, End);/**!!!!!!!!!!!!!!!!!!!!!!!**/
        #ifdef __DEBUG_MEDIAN__
          cout << "CFits::MedianVec: run m = " << m << ": tempRange set to " << tempRange << endl;
          cout << "CFits::MedianVec: TempVecArr(tempRange) = " << TempVecArr(tempRange) << endl;
        #endif
        TmpVecArr(m) = pfsDRPStella::math::Median(TempVecArr(tempRange), S_A1_Args_Median, PP_Args_Median);
        if (!Odd)
        {
          /** Mode == "NORMAL" **/
          if ((Mode.compare(string("NORMAL")) == 0) && (End + 1 < Length))
          {
            if (Start != End)
            {
              TmpVecArr(m) += pfsDRPStella::math::Median(TempVecArr(blitz::Range(Start+1, End+1)), S_A1_Args_Median, PP_Args_Median);
              TmpVecArr(m) /= 2.;
            }
            #ifdef __DEBUG_MEDIAN__
              cout << "CFits::MedianVec: run m = " << m << ": Odd = false: Mode == Normal: OutArr(m) set to " << TmpVecArr(m) << endl;
            #endif
          }
        }
      }
      free(PP_Args_Median);
      return (TmpVecArr);
    }
    
    /**
     *  Select(blitz::Array<double, int> &Arr, int KThSmallest) const;
     *  Returns the <KThSmallest> value of <Arr>.
     **/
    template<typename T>
    T Select(const blitz::Array<T, 1> &Arr, int KThSmallest)
    {
      blitz::Array<T, 1> TempArr = pfsDRPStella::math::BubbleSort(Arr);
      if (KThSmallest == 0)
        KThSmallest = 1;
      T result = TempArr(KThSmallest-1);
      return result;
    }
    
    /**
     *  bool IsOddNumber(long No) const
     *  Returns TRUE, if <No> is an Odd Number, FALSE if <No> is an Even Number.
     **/
    bool IsOddNumber(long No)
    {
      return (fabs((double)(((double)No) / 2.) - (double)(int)(((double)No) / 2.)) > 0.3);
    }
    
    /**
     * function Bubble Sort
     **/
    template<typename T>
    blitz::Array<T, 1> BubbleSort(const blitz::Array<T, 1> &A1_ArrIn)
    {
      long Size = A1_ArrIn.size();
      blitz::Array<T, 1> A1_Out(Size);
      A1_Out = A1_ArrIn;
      long UpperLimit = Size - 1;
      long LastSwap;
      long Pos;
      
      while(UpperLimit > 0)
      {
        LastSwap = 0;
        for(Pos = 0;Pos < UpperLimit; ++Pos)
        {
          if(A1_Out(Pos) > A1_Out(Pos+1))
          {
            std::swap(A1_Out(Pos), A1_Out(Pos+1));
            LastSwap = Pos;
          }
        }
        UpperLimit = LastSwap;
      }
      return (A1_Out);
    }

    template<typename T>
    std::vector<int> sortIndices(const std::vector<T> &vec_In){
      #ifdef __DEBUG_FITS_SORT__
        cout << "CFits::SortIndices(vec_In = " << vec_In << ") started" << endl;
      #endif
      std::vector<T> vec(vec_In.size());
      vec = vec_In;
      blitz::Array<T, 1> D_A1_In(vec.data(), blitz::shape(vec.size()), blitz::neverDeleteData);
      int I_M = 7;
      int I_NStack = 50;
      blitz::firstIndex i;
      
      int I_I, I_Indxt, I_Ir, I_J, I_K, I_L, I_SizeIn;
      int I_JStack = 0;
      blitz::Array<int, 1> I_A1_IStack(I_NStack);
      blitz::Array<int, 1> I_A1_Indx;
      T D_A;
      
      I_SizeIn = D_A1_In.size();
      I_Ir = I_SizeIn - 1;
      I_L = 0;
      
      I_A1_IStack = 0;
      I_A1_Indx.resize(I_SizeIn);
      I_A1_Indx = i;
      
      #ifdef __DEBUG_FITS_SORT__
        cout << "CFits::SortIndices() starting for(;;)" << endl;
      #endif
      for(;;)
      {
        if (I_Ir - I_L < I_M)
        {
          for (I_J = I_L + 1; I_J <= I_Ir; I_J++)
          {
            I_Indxt = I_A1_Indx(I_J);
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(): I_Indxt set to " << I_Indxt << endl;
            #endif
            D_A = D_A1_In(I_Indxt);
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(): D_A set to " << D_A << endl;
            #endif
            for (I_I = I_J - 1; I_I >= I_L; I_I--)
            {
              if (D_A1_In(I_A1_Indx(I_I)) <= D_A)
              {
                #ifdef __DEBUG_FITS_SORT__
                  cout << "CFits::SortIndices(): D_A1_In(P_I_A1_Indx(I_I = " << I_I << ") = " << I_A1_Indx(I_I) << " <= D_A = " << D_A << " =>  BREAK" << endl;
                #endif
                break;
              }
              I_A1_Indx(I_I + 1) = I_A1_Indx(I_I);
              #ifdef __DEBUG_FITS_SORT__
                cout << "CFits::SortIndices(): 1. P_I_A1_Indx(I_I+1 = " << I_I + 1 << ") set to " << I_A1_Indx(I_I+1) << " => P_I_A1_Indx = " << I_A1_Indx << endl;
              #endif
              
            }
            I_A1_Indx(I_I + 1) = I_Indxt;
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(): 2. P_I_A1_Indx(I_I+1 = " << I_I + 1 << ") set to " << I_A1_Indx(I_I+1) << " => P_I_A1_Indx = " << I_A1_Indx << endl;
            #endif
            
          }
          if (I_JStack == 0)
          {
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(): I_JStack <= 0 =>  BREAK" << endl;
            #endif
            break;
          }
          I_Ir = I_A1_IStack(I_JStack--);
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(): I_Ir(=" << I_Ir << ") set to I_A1_IStack(I_JStack--=" << I_JStack << ") = " << I_A1_IStack(I_JStack) << endl;
          #endif
          I_L  = I_A1_IStack(I_JStack--);
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(): I_L(=" << I_L << ") set to I_A1_IStack(I_JStack--=" << I_JStack << ") = " << I_A1_IStack(I_JStack) << endl;
          #endif
          
        }
        else
        {
          I_K = (I_L + I_Ir) >> 1;
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(): I_K(=" << I_K << ") set to (I_L[=" << I_L << "] + I_Ir[=" << I_Ir << "] >> 1)  = " << ((I_L + I_Ir) >> 1) << endl;
          #endif
          std::swap(I_A1_Indx(I_K),
               I_A1_Indx(I_L + 1));
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): P_I_A1_Indx(I_K=" << I_K << ")=" << I_A1_Indx(I_K) << " and P_I_A1_Indx(I_L(=" << I_L << ")+1)=" << I_A1_Indx(I_L+1) << " std::swapped" << " => P_I_A1_Indx = " << I_A1_Indx << endl;
          #endif
          if (D_A1_In(I_A1_Indx(I_L))
            > D_A1_In(I_A1_Indx(I_Ir)))
          {
            std::swap(I_A1_Indx(I_L),
                 I_A1_Indx(I_Ir));
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): P_I_A1_Indx(I_L=" << I_L << ")=" << I_A1_Indx(I_L) << " and P_I_A1_Indx(I_Ir(=" << I_Ir << "))=" << I_A1_Indx(I_Ir) << " std::swapped" << " => P_I_A1_Indx = " << I_A1_Indx << endl;
            #endif
            
          }
          if (D_A1_In(I_A1_Indx(I_L + 1))
            > D_A1_In(I_A1_Indx(I_Ir)))
          {
            std::swap(I_A1_Indx(I_L + 1),
                 I_A1_Indx(I_Ir));
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): P_I_A1_Indx(I_L=" << I_L << "+1)=" << I_A1_Indx(I_L + 1) << " and P_I_A1_Indx(I_Ir(=" << I_Ir << "))=" << I_A1_Indx(I_L+1) << " std::swapped" << " => P_I_A1_Indx = " << I_A1_Indx << endl;
            #endif
            
          }
          if (D_A1_In(I_A1_Indx(I_L))
            > D_A1_In(I_A1_Indx(I_L + 1)))
          {
            std::swap(I_A1_Indx(I_L),
                 I_A1_Indx(I_L + 1));
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): P_I_A1_Indx(I_L=" << I_L << ")=" << I_A1_Indx(I_L) << " and P_I_A1_Indx(I_L(=" << I_L << ")+1)=" << I_A1_Indx(I_L+1) << " std::swapped" << " => P_I_A1_Indx = " << I_A1_Indx << endl;
            #endif
            
          }
          I_I = I_L + 1;
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): I_I(=" << I_I << ") set to (I_L[=" << I_L << "] + 1)  = " << I_L + 1 << endl;
          #endif
          I_J = I_Ir;
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): I_J(=" << I_J << ") set to I_Ir[=" << I_Ir << "]" << endl;
          #endif
          I_Indxt = I_A1_Indx(I_L + 1);
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): I_Indxt(=" << I_Indxt << ") set to P_I_A1_Indx(I_L = " << I_L << "+1)" << endl;
          #endif
          D_A = D_A1_In(I_Indxt);
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): D_A(=" << D_A << ") set to D_A1_In(I_Indxt = " << I_Indxt << ")" << endl;
          #endif
          for (;;)
          {
            do
            {
              I_I++;
              #ifdef __DEBUG_FITS_SORT__
                cout << "CFits::SortIndices(D_A1_In): I_I set to " << I_I << " => D_A1_In(P_I_A1_Indx(I_I)) = " << D_A1_In(I_A1_Indx(I_I)) << endl;
              #endif
              
            }
            while(D_A1_In(I_A1_Indx(I_I)) < D_A && I_I < I_SizeIn - 2);
            do
            {
              I_J--;
              #ifdef __DEBUG_FITS_SORT__
                cout << "CFits::SortIndices(D_A1_In): I_J set to " << I_J << " => D_A1_In(P_I_A1_Indx(I_J)) = " << D_A1_In(I_A1_Indx(I_J)) << endl;
              #endif
              
            }
            while(D_A1_In(I_A1_Indx(I_J)) > D_A && I_J > 0);
            if (I_J < I_I)
            {
              #ifdef __DEBUG_FITS_SORT__
                cout << "CFits::SortIndices(D_A1_In): I_J(=" << I_J << ") < I_I(=" << I_I << ") => BREAK" << endl;
              #endif
              break;
            }
            std::swap(I_A1_Indx(I_I),
                 I_A1_Indx(I_J));
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): P_I_A1_Indx(I_I=" << I_I << ")=" << I_A1_Indx(I_I) << " and P_I_A1_Indx(I_J(=" << I_J << "))=" << I_A1_Indx(I_J) << " std::swapped" << " => P_I_A1_Indx = " << I_A1_Indx << endl;
            #endif
            
          }
          I_A1_Indx(I_L + 1) = I_A1_Indx(I_J);
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): P_I_A1_Indx(I_L=" << I_L << "+1) set to P_I_A1_Indx(I_J=" << I_J << ") = " << I_A1_Indx(I_L+1) << ") => P_I_A1_Indx = " << I_A1_Indx << endl;
          #endif
          I_A1_Indx(I_J) = I_Indxt;
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): P_I_A1_Indx(I_J=" << I_J << ") set to I_Indxt(=" << I_Indxt << ") => P_I_A1_Indx = " << I_A1_Indx << endl;
          #endif
          I_JStack += 2;
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): I_JStack = " << I_JStack << endl;
          #endif
          if (I_JStack > I_NStack)
          {
            cout << "CFits::SortIndices: ERROR: I_NStack ( = " << I_NStack << ") too small!!!";
            exit(EXIT_FAILURE);
          }
          if (I_Ir - I_I + 1 >= I_J - I_L)
          {
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): I_Ir(= " << I_Ir << ") - I_I(=" << I_I << ") + 1 = " << I_Ir - I_I + 1 << " >= I_J(="<< I_J << ") + I_L(=" << I_L << ") = " << I_J - I_L << endl;
            #endif
            I_A1_IStack(I_JStack) = I_Ir;
            I_A1_IStack(I_JStack - 1) = I_I;
            I_Ir = I_J - 1;
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): I_I set to I_J(=" << I_J << ") - 1" << endl;
            #endif
            
          }
          else
          {
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): I_Ir(= " << I_Ir << ") - I_I(=" << I_I << ") + 1 = " << I_Ir - I_I + 1 << " < I_J(="<< I_J << ") + I_L(=" << I_L << ") = " << I_J - I_L << endl;
            #endif
            I_A1_IStack(I_JStack) = I_J - 1;
            I_A1_IStack(I_JStack - 1) = I_L;
            I_L = I_I;
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): I_L set to I_I(=" << I_I << endl;
            #endif
            
          }
        }
      }
      I_A1_IStack.resize(0);
      std::vector<int> vecIndx(I_A1_Indx.size());
      for (int i = 0; i < static_cast<int>(I_A1_Indx.size()); ++i){
        vecIndx[i] = I_A1_Indx(i);
      }
      return (vecIndx);
    }
    
    /**
     * function GetRowFromIndex(int I_Index_In, int I_NRows_In) const
     * task: Returns Row specified by I_Index_In from the formula
     *       Col = (int)(I_Index_In / I_NRows_In)
     *       Row = fiberTraceNumber - Col * I_NRows_In
     **/
    int GetRowFromIndex(int I_Index_In, int I_NRows_In)
    {
      return (I_Index_In - (I_NRows_In * pfsDRPStella::math::GetColFromIndex(I_Index_In, I_NRows_In)));
    }
    
    /**
     * function GetColFromIndex(int I_Index_In, int I_NRows_In) const
     * task: Returns Col specified by I_Index_In from the formula
     *       Col = (int)(I_Index_In / I_NRows_In)
     *       Row = fiberTraceNumber - Col * I_NRows_In
     **/
    int GetColFromIndex(int I_Index_In, int I_NRows_In)
    {
      return ((int)(I_Index_In / I_NRows_In));
    }
    
    /**
     *  BandSol() const
     **/
    void BandSol(int Argc, void *Argv[])
    {
      double *a, *r;
      double aa;
      int n, nd, o, p, q;
      
      /* Double arrays are passed by reference. */
      a = (double*)Argv[0];
      r = (double*)Argv[1];
      /* The size of the system and the number of diagonals are passed by value */
      n = *(int *)Argv[2];
      nd = *(int *)Argv[3];
      #ifdef __DEBUG_BANDSOL__
        cout << "CFits::BandSol: *a = " << *a << endl;
        cout << "CFits::BandSol: *r = " << *r << endl;
        cout << "CFits::BandSol: n = " << n << endl;
        cout << "CFits::BandSol: nd = " << nd << endl;
      #endif
//      exit(EXIT_FAILURE);
      /*
       *  bandsol solve a sparse system of linear equations with band-diagonal matrix.
       *     Band is assumed to be symmetrix relative to the main diaginal. Usage:
       *  CALL_EXTERNAL('bandsol.so', 'bandsol', a, r, n, nd)
       *  where a is 2D array [n,m] where n - is the number of equations and nd
       *  is the width of the band (3 for tri-diagonal system),
       *  nd is always an odd number. The main diagonal should be in a(*,nd/2)
       *  The first lower subdiagonal should be in a(1:n-1,nd-2-1), the first
       *             upper subdiagonal is in a(0:n-2,nd/2+1) etc. For example:
       *                    / 0 0 X X X \
       *  | 0 X X X X |
       *  | X X X X X |
       *  | X X X X X |
       *  A = | X X X X X |
       *  | X X X X X |
       *  | X X X X X |
       *  | X X X X 0 |
       *  \ X X X 0 0 /
       *  r is the array of RHS of size n.
       */
      
      /* Forward sweep */
      for(o = 0; o < n - 1; o++)
      {
        aa=a[o + n * (nd / 2)];
        #ifdef __DEBUG_BANDSOL__
          cout << "bandsol: for(o(=" << o << ")=0; o<n(=" << n << "); o++): o+n*(nd/2) = " << o+n*(nd/2) << endl;
          cout << "bandsol: for(o(=" << o << ")=0; o<n(=" << n << "); o++): aa set to " << aa << endl;
        #endif
        r[o] /= aa;
        #ifdef __DEBUG_BANDSOL__
          cout << "bandsol: for(o(=" << o << ")=0; o<n(=" << n << "); o++): r[o] set to " << r[o] << endl;
        #endif
        for(p = 0; p < nd; p++){
          a[o + p * n] /= aa;
          #ifdef __DEBUG_BANDSOL__
            cout << "bandsol: for(o(=" << o << ")=0; o<n(=" << n << "); o++): for(p(=" << p << ")=0; p<nd(=" << nd << "); p++): a[o+p*n=" << o+p*n << "] set to " << a[o+p*n] << endl;
          #endif
        }
        for(p = 1; p < MIN(nd / 2 + 1, n - o); p++)
        {
          aa=a[o + p + n * (nd / 2 - p)];
          #ifdef __DEBUG_BANDSOL__
            cout << "bandsol: for(o(=" << o << ")=0; o<n(=" << n << "); o++): for(p(=" << p << ")=0; p<nd(=" << nd << "); p++): aa set to " << aa << endl;
          #endif
          r[o + p] -= r[o] * aa;
          #ifdef __DEBUG_BANDSOL__
            cout << "bandsol: for(o(=" << o << ")=0; o<n(=" << n << "); o++): for(p(=" << p << ")=0; p<nd(=" << nd << "); p++): r[o+p=" << o+p << "] set to " << r[o+p] << endl;
          #endif
          for(q = 0; q < n * (nd - p); q += n){
            a[o + p + q] -= a[o + q + n * p] * aa;
            #ifdef __DEBUG_BANDSOL__
              cout << "bandsol: for(o(=" << o << ")=0; o<n(=" << n << "); o++): for(p(=" << p << ")=0; p<nd(=" << nd << "); p++): for(q(=" << q << ")=0; q<n*(nd-p)(=" << n*(nd-p) << "); q++): a[o+p+q=" << o+p+q << "] set to " << a[o+p+q] << endl;
            #endif
          }
        }
      }
      
      /* Backward sweep */
      r[n-1] /= a[n - 1 + n * (nd / 2)];
      #ifdef __DEBUG_BANDSOL__
        cout << "bandsol: r[n-1=" << n-1 << "] set to " << r[n-1] << endl;
      #endif
      for(o=n-1; o>0; o--)
      {
        for(p=1; p <= MIN(nd/2,o); p++){
          r[o-p] -= r[o] *
          a[o - p + n * (nd / 2 + p)];
          #ifdef __DEBUG_BANDSOL__
            cout << "bandsol: for(o(=" << o << ")=n-1=" << n-1 << "; o>0; o--): for(p(=" << p << ")=1; p<=Min(nd/2=" << nd/2 << ",o=" << o << "); p++): r[o-p=" << o-p << "] set to " << r[o-p] << endl;
          #endif
        }
        r[o-1] /= a[o-1+n*(nd/2)];
        #ifdef __DEBUG_BANDSOL__
          cout << "bandsol: for(o(=" << o << ")=n-1=" << n-1 << "; o>0; o--): r[o-1=" << o-1 << "] set to " << r[o-1] << endl;
        #endif
      }
      r[0] /= a[n*(nd/2)];
      #ifdef __DEBUG_BANDSOL__
        cout << "bandsol: r[0] set to " << r[0] << endl;

        for (int i=0; i<nd; i++){
          if (to_string(r[i]).compare("-nan") == 0)
            exit(EXIT_FAILURE);
          for (int j=0; j<n; j++){
            if (to_string(a[i*j]).compare("-nan") == 0)
              exit(EXIT_FAILURE);
          }
        }
      #endif
        
      return;
    }
    
    /**
     * void TriDag
     * Solves for a vector blitz::Array<double, N> UVecArr the tridiagonal linear set given by equation
     *  [ b_1  c_1  0  ...                       ]   [  u_1  ]   [  r_1  ]
     *  [ a_2  b_2  c_2 ...                      ]   [  u_2  ]   [  r_2  ]
     *  [            ...                         ] * [  ...  ] = [  ...  ]
     *  [            ...  a_(N-1) b_(N-1) c_(N-1)]   [u_(N-1)]   [r_(N-1)]
     *  [            ...     0     a_N      b_N  ]   [  u_N  ]   [  r_N  ]
     * BVecArr(0..N-1), CVecArr(0..N-1), and RVecArr(0..N-1) are input vectors and are not modified.
     **/
    bool TriDag(blitz::Array<double, 1> &AVecArr, blitz::Array<double, 1> &BVecArr, blitz::Array<double, 1> &CVecArr, blitz::Array<double, 1> &RVecArr, blitz::Array<double, 1> &UVecArr)
    {
      int m;
      double Bet;
      int N = UVecArr.size();
      blitz::Array<double, 1> Gam(N);
      
      if (BVecArr(0) == 0.0)
      {
        cout << "CFits::TriDag: Error 1 in TriDag: BVecArr(0) == 0" << endl;
        /// If this happens then you should rewrite your equations as a set of order N-1, with u2 trivially eliminated
        return false;
      }
      UVecArr(0) = RVecArr(0) / (Bet = BVecArr(0));
      for (m = 1; m < N; m++) /// Decomposition and forward substitution
      {
        Gam(m) = CVecArr(m-1) / Bet;
        Bet = BVecArr(m) - AVecArr(m) * Gam(m);
        if (Bet == 0.0)
        {
          cout << "CFits::TriDag: Error 2 in TriDag: Bet == 0.0" << endl;
          /// Algorithm fails, see below
          return false;
        }
        UVecArr(m) = (RVecArr(m) - AVecArr(m) * UVecArr(m-1)) / Bet;
      }
      for (m = (N-2); m >= 0; m--)
      {
        UVecArr(m) -= Gam(m+1) * UVecArr(m+1); /// Backsubstitution
      }
      Gam.resize(0);
      
      return true;
    }
    
    template<typename T>
    bool Uniq(const blitz::Array<T, 1> &IA1_In, blitz::Array<int, 1> &IA1_Result)
    {
      blitz::Array<T, 1> VecArr_In(IA1_In.size());
      VecArr_In = IA1_In;
      std::vector<T> vec_In(VecArr_In.data(),VecArr_In.data() + VecArr_In.size());
      std::sort(vec_In.begin(), vec_In.end());
      auto last = std::unique(vec_In.begin(), vec_In.end());
      vec_In.erase(last, vec_In.end());
      #ifdef __DEBUG_UNIQ__
        std::cout << "Uniq: IA1_In = " << IA1_In << endl;
        cout << "Uniq: vec_In.size() = " << vec_In.size() << endl;
        for (const auto& i : vec_In)
          std::cout << i << " ";
        std::cout << "\n";
        int a=2;
        int b=2;
        double c=fabs(a-b);
        cout << "c = " << c << endl;
      #endif
      IA1_Result.resize(vec_In.size());
      blitz::Array<int, 1> I_A1_Ind(IA1_In.size());
      blitz::Array<int, 1> *P_I_A1_Ind;
      int count = 0;
      for (int i=0; i<vec_In.size(); i++){
        I_A1_Ind = blitz::where(fabs(IA1_In - vec_In[i]) < 0.5, 1, 0);
        P_I_A1_Ind = pfsDRPStella::math::GetIndex(I_A1_Ind, count);
        if (count == 0){
          #ifdef __DEBUG_UNIQ__
          for (int j=0; j<IA1_In.size(); j++)
            cout << "Uniq: fabs(IA1_In(j=" << j << ")=" << IA1_In(j) << " - vec_In[i=" << i << "]=" << vec_In[i] << ") = " << fabs(IA1_In(j) - vec_In[i]) << endl;
          #endif
          cout << "Uniq: I_A1_Ind = " << I_A1_Ind << endl;
          cout << "Uniq: ERROR: vec_In[" << i << "]=" << vec_In[i] << ": count == 0 => Returning FALSE" << endl;
          return false;
        }
        IA1_Result(i) = (*P_I_A1_Ind)(0);
        delete(P_I_A1_Ind);
      }
      return true;// (*(new blitz::Array<int, 1>(IA1_Indices.copy())));
    }
    
    template<typename T>
    T Round(const T ToRound, int DigitsBehindDot){
      long TempLong;
      int TempInt;
      T TempDbl;
      
      bool B_IsNegative = ToRound < 0.;
      TempLong = long(ToRound * pow(10., DigitsBehindDot));
      TempDbl = (ToRound - T(TempLong)) * pow(10., DigitsBehindDot);
      TempInt = int(abs(TempDbl * 10.));
      if (TempInt > 4){
        if (B_IsNegative)
          TempLong--;
        else
          TempLong++;
      }
      return (T(TempLong) / pow(10., DigitsBehindDot));
    }
  
    /************************************************************/
  
    template<typename T>
    long RoundL(const T ToRound){
      return long(Round(ToRound, 0));
    }
  
    /************************************************************/
  
    template<typename T>
    int Round(const T ToRound){
      return (int)Round(ToRound, 0);
    }
    
//    template<typename T>
//    void resize(blitz::Array<T, 1> &arr_InOut, unsigned int newSize){
//      blitz::Array<T, 1> *newArr = new blitz::Array<T, 1>(newSize);
//      *newArr = 0;
//      arr_InOut.resize(0);
//      &arr_InOut = newArr;
//      return;
//    }
  }/// end namespace math
  
  namespace utils{
    
    /**
     *       Returns Position of <str_In> in Array of strings <keyWords_In>, if <keyWords_In> contains string <str_In>, else returns -1.
     **/
    int KeyWord_Set(const blitz::Array<string, 1> &keyWords_In, 
                    const string &str_In){
      for (int m = 0; m < static_cast<int>(keyWords_In.size()); m++){
        if (keyWords_In(m).compare(str_In) == 0)
          return m;
      }
      return -1;
    }
    
    template<typename T>
    bool WriteFits(const blitz::Array<T,1>* image_In, const string &fileName_In){
      blitz::Array<T, 2> image(image_In->size(), 1);
      image(blitz::Range::all(), 0) = (*image_In);
      return WriteFits(&image, fileName_In);
    }
      
    template<typename T>
    bool WriteFits(const blitz::Array<T,2>* image_In, const string &fileName_In){
      fitsfile *P_Fits;
      int Status;
      long fpixel, nelements;
      void *p_void;
      
      Status=0;
      remove(fileName_In.c_str());
      fits_create_file(&P_Fits, fileName_In.c_str(), &Status);//{
      if (Status !=0){
        cout << "CFits::WriteFits: Error <" << Status << "> while creating file " << fileName_In << endl;
        char* P_ErrMsg = new char[255];
        ffgerr(Status, P_ErrMsg);
        cout << "CFits::WriteFits: <" << P_ErrMsg << "> => Returning FALSE" << endl;
        delete[] P_ErrMsg;
        return false;
      }
      
      ///  fits_write_img(P_FitsFile, TDOUBLE, fpixel, nelements,
      ///    p_void, &Status);
      long naxes[2] = {image_In->cols(), image_In->rows()};
      int naxis = 2;
      fits_create_img(P_Fits, DOUBLE_IMG, naxis, naxes, &Status);
      if (Status !=0){
        cout << "CFits::WriteFits: Error <" << Status << "> while creating image " << fileName_In << endl;
        char* P_ErrMsg = new char[255];
        ffgerr(Status, P_ErrMsg);
        cout << "CFits::WriteFits: <" << P_ErrMsg << "> => Returning FALSE" << endl;
        delete[] P_ErrMsg;
        return false;
      }
      #ifdef __DEBUG_WRITEFITS__
        cout << "CFits::WriteFits: size of image_In = <" << image_In->rows() << "x" << image_In->cols() << ">" << endl;
        cout << "CFits::WriteFits: size of image_In = <" << image_In->rows() << "x" << image_In->cols() << ">" << endl;
      #endif
      
      fpixel = 1;
      //nullval = 0.;
      nelements = image_In->cols() * image_In->rows();
      
      p_void = (const_cast<blitz::Array<T, 2>*>(image_In))->data();// = new blitz::Array<double,2>(p_Array, blitz::shape(naxes[0], naxes[1]),
      #ifdef __DEBUG_WRITEFITS__
        cout << "CFits::WriteFits: p_void = <" << (*((double*)p_void)) << ">" << endl;
      #endif

      int nbits = TDOUBLE;
      if (typeid(T) == typeid(short))
          nbits = TSHORT;
      else if (typeid(T) == typeid(unsigned short))
          nbits = TUSHORT;
      else if (typeid(T) == typeid(int))
          nbits = TINT;
      else if (typeid(T) == typeid(unsigned int))
          nbits = TUINT;
      else if (typeid(T) == typeid(long))
          nbits = TLONG;
      else if (typeid(T) == typeid(unsigned long))
          nbits = TULONG;
      else if (typeid(T) == typeid(float))
          nbits = TFLOAT;
      else
          nbits = TDOUBLE;
      fits_write_img(P_Fits, nbits, fpixel, nelements, p_void, &Status);
      
      if (Status !=0){
        cout << "CFits::WriteFits: Error " << Status << " while writing file " << fileName_In << endl;
        char* P_ErrMsg = new char[255];
        ffgerr(Status, P_ErrMsg);
        cout << "CFits::WriteFits: <" << P_ErrMsg << "> => Returning FALSE" << endl;
        delete[] P_ErrMsg;
        return false;
      }
      
      fits_close_file(P_Fits, &Status);
      cout << "CFits::WriteFits: FitsFileName <" << fileName_In << "> closed" << endl;
      if (Status !=0){
        cout << "CFits::WriteFits: Error " << Status << " while closing file " << fileName_In << endl;
        char* P_ErrMsg = new char[255];
        ffgerr(Status, P_ErrMsg);
        cout << "CFits::WriteFits: <" << P_ErrMsg << "> => Returning FALSE" << endl;
        delete[] P_ErrMsg;
        return false;
      }
      return true;
    }
    
    /**
     *  task: Writes Array <Array_In> to file <CS_FileName_In>
     **/
    template<typename T, int N>
    bool WriteArrayToFile(const blitz::Array<T, N> &Array_In, 
                          const string &S_FileName_In, 
                          const string &S_Mode)
    {
      int m, n;
//      ofstream ofs(S_FileName_In.c_str());
      //  FILE *p_file;
      FILE *p_file;
      p_file = fopen(S_FileName_In.c_str(), "w");
      
      if (S_Mode.compare(string("binary")) == 0){
        if (N == 1){
          fwrite(Array_In.data(), sizeof(T), Array_In.size(), p_file);
        }
        else if (N == 2){
          for (m = 0; m < Array_In.rows(); m++)
          {
            fwrite(Array_In(m, blitz::Range::all()).data(), sizeof(T), Array_In.cols(), p_file);
//          for (n = 0; n < D_A2_In.cols(); n++)
//            ofs << D_A2_In(m, n) << " ";
//          ofs << endl;
          }
        }
      }
      else{
        blitz::Array<bool, 1> B_A1_Exp(1);
        bool B_Exp = false;
        if (N == 1){
          if (max(Array_In) < 1e-7)
            B_Exp = true;
        }
        else{
          B_A1_Exp.resize(Array_In.cols());
          B_A1_Exp = false;
          for (m = 0; m < Array_In.cols(); m++){
            if (max(Array_In(blitz::Range::all(), m)) < 1e-7)
              B_A1_Exp(m) = true;
          }
        }
        for (m = 0; m < Array_In.rows(); m++)
        {
          if (N == 1){
            if (!B_Exp){
              if (typeid(T) == typeid(short))
                fprintf(p_file, "%hd\n", static_cast<short>(Array_In(m)));
              else if (typeid(T) == typeid(unsigned short))
                fprintf(p_file, "%u\n", static_cast<unsigned short>(Array_In(m)));
              else if (typeid(T) == typeid(int))
                fprintf(p_file, "%d\n", static_cast<int>(Array_In(m)));
              else if (typeid(T) == typeid(unsigned int))
                fprintf(p_file, "%u\n", static_cast<unsigned int>(Array_In(m)));
              else if (typeid(T) == typeid(long))
                fprintf(p_file, "%ld\n", static_cast<long>(Array_In(m)));
              else if (typeid(T) == typeid(unsigned long))
                fprintf(p_file, "%lu\n", static_cast<unsigned long>(Array_In(m)));
              else if (typeid(T) == typeid(float))
                fprintf(p_file, "%.17f\n", static_cast<float>(Array_In(m)));
              else
                fprintf(p_file, "%.17f\n", static_cast<double>(Array_In(m)));
            }
            else{
              fprintf(p_file, "%.8e\n", static_cast<double>(Array_In(m)));
            }
          }
          else{/// N == 2 
            for (n = 0; n < Array_In.cols(); n++){
              if (B_A1_Exp(n))
                fprintf(p_file, " %.8e", double(Array_In(m,n)));
              else{
                if (typeid(T) == typeid(short))
                  fprintf(p_file, " %hd", static_cast<short>(Array_In(m,n)));
                else if (typeid(T) == typeid(unsigned short))
                  fprintf(p_file, " %u", static_cast<unsigned short>(Array_In(m,n)));
                else if (typeid(T) == typeid(int))
                  fprintf(p_file, " %d", static_cast<int>(Array_In(m,n)));
                else if (typeid(T) == typeid(unsigned int))
                  fprintf(p_file, " %u", static_cast<unsigned int>(Array_In(m,n)));
                else if (typeid(T) == typeid(long))
                  fprintf(p_file, " %ld", static_cast<long>(Array_In(m,n)));
                else if (typeid(T) == typeid(unsigned long))
                  fprintf(p_file, " %lu", static_cast<unsigned long>(Array_In(m,n)));
                else if (typeid(T) == typeid(float))
                  fprintf(p_file, " %.10f", static_cast<float>(Array_In(m,n)));
                else
                  fprintf(p_file, " %.10f", static_cast<double>(Array_In(m,n)));
              }
            }
            fprintf(p_file, "\n");
          }
        }
      }
      fclose(p_file);
      return true;
    }
    
//    template<typename ImageT, typename MaskT, typename VarianceT>
//    PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) getShared(afwImage::MaskedImage<ImageT, MaskT, VarianceT> const &maskedImage){
//      PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) ptr = PTR(const new afwImage::MaskedImage<ImageT, MaskT, VarianceT>(maskedImage));
//      return ptr;
//    }
  }
  
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
  template class FiberTrace<float>;
  template class FiberTrace<double>;
  
//  template class FiberTraceSet<unsigned short>;
//  template class FiberTraceSet<int>;
  template class FiberTraceSet<float>;
  template class FiberTraceSet<double>;

  template pfsDRPStella::FiberTraceSet<float, unsigned short, float> math::findAndTraceApertures(const PTR(afwImage::MaskedImage<float, unsigned short, float>) &, const pfsDRPStella::FiberTraceFunctionFindingControl &fiberTraceFunctionFindingControl);
  template pfsDRPStella::FiberTraceSet<double, unsigned short, float> math::findAndTraceApertures(const PTR(afwImage::MaskedImage<double, unsigned short, float>) &, const pfsDRPStella::FiberTraceFunctionFindingControl &fiberTraceFunctionFindingControl);
  
  template int math::Fix(unsigned short);
  template int math::Fix(int);
  template int math::Fix(long);
  template int math::Fix(float);
  template int math::Fix(double);
  
  template blitz::Array<int, 1> math::Fix(const blitz::Array<unsigned short, 1> &);
  template blitz::Array<int, 1> math::Fix(const blitz::Array<int, 1> &);
  template blitz::Array<int, 1> math::Fix(const blitz::Array<long, 1> &);
  template blitz::Array<int, 1> math::Fix(const blitz::Array<float, 1> &);
  template blitz::Array<int, 1> math::Fix(const blitz::Array<double, 1> &);
  
  template blitz::Array<int, 2> math::Fix(const blitz::Array<unsigned short, 2> &);
  template blitz::Array<int, 2> math::Fix(const blitz::Array<int, 2> &);
  template blitz::Array<int, 2> math::Fix(const blitz::Array<long, 2> &);
  template blitz::Array<int, 2> math::Fix(const blitz::Array<float, 2> &);
  template blitz::Array<int, 2> math::Fix(const blitz::Array<double, 2> &);

  template long math::FixL(unsigned short);
  template long math::FixL(int);
  template long math::FixL(long);
  template long math::FixL(float);
  template long math::FixL(double);
  
  template blitz::Array<long, 1> math::FixL(const blitz::Array<unsigned short, 1> &);
  template blitz::Array<long, 1> math::FixL(const blitz::Array<int, 1> &);
  template blitz::Array<long, 1> math::FixL(const blitz::Array<long, 1> &);
  template blitz::Array<long, 1> math::FixL(const blitz::Array<float, 1> &);
  template blitz::Array<long, 1> math::FixL(const blitz::Array<double, 1> &);
  
  template blitz::Array<long, 2> math::FixL(const blitz::Array<unsigned short, 2> &);
  template blitz::Array<long, 2> math::FixL(const blitz::Array<int, 2> &);
  template blitz::Array<long, 2> math::FixL(const blitz::Array<long, 2> &);
  template blitz::Array<long, 2> math::FixL(const blitz::Array<float, 2> &);
  template blitz::Array<long, 2> math::FixL(const blitz::Array<double, 2> &);
  
  template int math::Int(unsigned short);
  template int math::Int(int);
  template int math::Int(long);
  template int math::Int(float);
  template int math::Int(double);
  
  template blitz::Array<int, 1> math::Int(const blitz::Array<unsigned short, 1> &);
  template blitz::Array<int, 1> math::Int(const blitz::Array<int, 1> &);
  template blitz::Array<int, 1> math::Int(const blitz::Array<long, 1> &);
  template blitz::Array<int, 1> math::Int(const blitz::Array<float, 1> &);
  template blitz::Array<int, 1> math::Int(const blitz::Array<double, 1> &);
  
  template blitz::Array<int, 2> math::Int(const blitz::Array<short unsigned int, 2> &);
  template blitz::Array<int, 2> math::Int(const blitz::Array<int, 2> &);
  template blitz::Array<int, 2> math::Int(const blitz::Array<long, 2> &);
  template blitz::Array<int, 2> math::Int(const blitz::Array<float, 2> &);
  template blitz::Array<int, 2> math::Int(const blitz::Array<double, 2> &);

  template long math::Long(unsigned short);
  template long math::Long(int);
  template long math::Long(long);
  template long math::Long(float);
  template long math::Long(double);
  
  template blitz::Array<long, 1> math::Long(const blitz::Array<unsigned short, 1> &);
  template blitz::Array<long, 1> math::Long(const blitz::Array<int, 1> &);
  template blitz::Array<long, 1> math::Long(const blitz::Array<long, 1> &);
  template blitz::Array<long, 1> math::Long(const blitz::Array<float, 1> &);
  template blitz::Array<long, 1> math::Long(const blitz::Array<double, 1> &);
  
  template blitz::Array<long, 2> math::Long(const blitz::Array<unsigned short, 2> &);
  template blitz::Array<long, 2> math::Long(const blitz::Array<int, 2> &);
  template blitz::Array<long, 2> math::Long(const blitz::Array<long, 2> &);
  template blitz::Array<long, 2> math::Long(const blitz::Array<float, 2> &);
  template blitz::Array<long, 2> math::Long(const blitz::Array<double, 2> &);
  
  
  template void math::Float(const blitz::Array<unsigned short, 1> &, blitz::Array<float, 1>&);
  template void math::Float(const blitz::Array<int, 1> &, blitz::Array<float, 1>&);
  template void math::Float(const blitz::Array<long, 1> &, blitz::Array<float, 1>&);
  template void math::Float(const blitz::Array<float, 1> &, blitz::Array<float, 1>&);
  template void math::Float(const blitz::Array<double, 1> &, blitz::Array<float, 1>&);

  template blitz::Array<float, 1> math::Float(const blitz::Array<unsigned short, 1> &);
  template blitz::Array<float, 1> math::Float(const blitz::Array<int, 1> &);
  template blitz::Array<float, 1> math::Float(const blitz::Array<long, 1> &);
  template blitz::Array<float, 1> math::Float(const blitz::Array<float, 1> &);
  template blitz::Array<float, 1> math::Float(const blitz::Array<double, 1> &);
  
  template void math::Float(const blitz::Array<unsigned short, 2> &, blitz::Array<float, 2>&);
  template void math::Float(const blitz::Array<int, 2> &, blitz::Array<float, 2>&);
  template void math::Float(const blitz::Array<long, 2> &, blitz::Array<float, 2>&);
  template void math::Float(const blitz::Array<float, 2> &, blitz::Array<float, 2>&);
  template void math::Float(const blitz::Array<double, 2> &, blitz::Array<float, 2>&);
  
  template blitz::Array<float, 2> math::Float(const blitz::Array<unsigned short, 2> &);
  template blitz::Array<float, 2> math::Float(const blitz::Array<int, 2> &);
  template blitz::Array<float, 2> math::Float(const blitz::Array<long, 2> &);
  template blitz::Array<float, 2> math::Float(const blitz::Array<float, 2> &);
  template blitz::Array<float, 2> math::Float(const blitz::Array<double, 2> &);
  
  template void math::Double(const blitz::Array<unsigned short, 1> &, blitz::Array<double, 1>&);
  template void math::Double(const blitz::Array<int, 1> &, blitz::Array<double, 1>&);
  template void math::Double(const blitz::Array<long, 1> &, blitz::Array<double, 1>&);
  template void math::Double(const blitz::Array<float, 1> &, blitz::Array<double, 1>&);
  template void math::Double(const blitz::Array<double, 1> &, blitz::Array<double, 1>&);
  
  template void math::Double(const blitz::Array<unsigned short, 2> &, blitz::Array<double, 2>&);
  template void math::Double(const blitz::Array<int, 2> &, blitz::Array<double, 2>&);
  template void math::Double(const blitz::Array<long, 2> &, blitz::Array<double, 2>&);
  template void math::Double(const blitz::Array<float, 2> &, blitz::Array<double, 2>&);
  template void math::Double(const blitz::Array<double, 2> &, blitz::Array<double, 2>&);

  template blitz::Array<double, 1> math::Double(const blitz::Array<unsigned short, 1> &Arr);
  template blitz::Array<double, 1> math::Double(const blitz::Array<int, 1> &Arr);
  template blitz::Array<double, 1> math::Double(const blitz::Array<long, 1> &Arr);
  template blitz::Array<double, 1> math::Double(const blitz::Array<float, 1> &Arr);
  template blitz::Array<double, 1> math::Double(const blitz::Array<double, 1> &Arr);
  
  template blitz::Array<double, 2> math::Double(const blitz::Array<unsigned short, 2> &Arr);
  template blitz::Array<double, 2> math::Double(const blitz::Array<int, 2> &Arr);
  template blitz::Array<double, 2> math::Double(const blitz::Array<long, 2> &Arr);
  template blitz::Array<double, 2> math::Double(const blitz::Array<float, 2> &Arr);
  template blitz::Array<double, 2> math::Double(const blitz::Array<double, 2> &Arr);
  
  template int math::Round(const float ToRound);
  template int math::Round(const double ToRound);
  
  template float math::Round(const float ToRound, int DigitsBehindDot);
  template double math::Round(const double ToRound, int DigitsBehindDot);
  
  template long math::RoundL(const float ToRound);
  template long math::RoundL(const double ToRound);

  template blitz::Array<unsigned short, 1> math::Replicate(unsigned short val, int Len);
  template blitz::Array<int, 1> math::Replicate(int val, int Len);
  template blitz::Array<long, 1> math::Replicate(long val, int Len);
  template blitz::Array<float, 1> math::Replicate(float val, int Len);
  template blitz::Array<double, 1> math::Replicate(double val, int Len);
  
  template blitz::Array<unsigned short, 1>* math::Reform(const blitz::Array<unsigned short, 2> &Arr);
  template blitz::Array<int, 1>* math::Reform(const blitz::Array<int, 2> &Arr);
  template blitz::Array<long, 1>* math::Reform(const blitz::Array<long, 2> &Arr);
  template blitz::Array<float, 1>* math::Reform(const blitz::Array<float, 2> &Arr);
  template blitz::Array<double, 1>* math::Reform(const blitz::Array<double, 2> &Arr);
  
  template blitz::Array<unsigned short, 2>* math::Reform(const blitz::Array<unsigned short, 1> &Arr, int DimA, int DimB);
  template blitz::Array<int, 2>* math::Reform(const blitz::Array<int, 1> &Arr, int DimA, int DimB);
  template blitz::Array<long, 2>* math::Reform(const blitz::Array<long, 1> &Arr, int DimA, int DimB);
  template blitz::Array<float, 2>* math::Reform(const blitz::Array<float, 1> &Arr, int DimA, int DimB);
  template blitz::Array<double, 2>* math::Reform(const blitz::Array<double, 1> &Arr, int DimA, int DimB);
  
  template bool math::GetSubArrCopy(const blitz::Array<unsigned short, 1> &DA1_In, const blitz::Array<int, 1> &IA1_Indices, blitz::Array<unsigned short, 1> &DA1_Out);
  template bool math::GetSubArrCopy(const blitz::Array<int, 1> &DA1_In, const blitz::Array<int, 1> &IA1_Indices, blitz::Array<int, 1> &DA1_Out);
  template bool math::GetSubArrCopy(const blitz::Array<long, 1> &DA1_In, const blitz::Array<int, 1> &IA1_Indices, blitz::Array<long, 1> &DA1_Out);
  template bool math::GetSubArrCopy(const blitz::Array<float, 1> &DA1_In, const blitz::Array<int, 1> &IA1_Indices, blitz::Array<float, 1> &DA1_Out);
  template bool math::GetSubArrCopy(const blitz::Array<double, 1> &DA1_In, const blitz::Array<int, 1> &IA1_Indices, blitz::Array<double, 1> &DA1_Out);
  
  template bool math::GetSubArrCopy(const blitz::Array<unsigned short, 2> &A2_In, const blitz::Array<int, 1> &I_A1_Indices, int I_Mode_In, blitz::Array<unsigned short, 2> &A2_Out);
  template bool math::GetSubArrCopy(const blitz::Array<int, 2> &A2_In, const blitz::Array<int, 1> &I_A1_Indices, int I_Mode_In, blitz::Array<int, 2> &A2_Out);
  template bool math::GetSubArrCopy(const blitz::Array<long, 2> &A2_In, const blitz::Array<int, 1> &I_A1_Indices, int I_Mode_In, blitz::Array<long, 2> &A2_Out);
  template bool math::GetSubArrCopy(const blitz::Array<float, 2> &A2_In, const blitz::Array<int, 1> &I_A1_Indices, int I_Mode_In, blitz::Array<float, 2> &A2_Out);
  template bool math::GetSubArrCopy(const blitz::Array<double, 2> &A2_In, const blitz::Array<int, 1> &I_A1_Indices, int I_Mode_In, blitz::Array<double, 2> &A2_Out);
  
  template blitz::Array<unsigned short, 2> math::GetSubArrCopy(const blitz::Array<unsigned short, 2> &A2_In, const blitz::Array<int, 3> &I_A3_Indices);
  template blitz::Array<int, 2> math::GetSubArrCopy(const blitz::Array<int, 2> &A2_In, const blitz::Array<int, 3> &I_A3_Indices);
  template blitz::Array<long, 2> math::GetSubArrCopy(const blitz::Array<long, 2> &A2_In, const blitz::Array<int, 3> &I_A3_Indices);
  template blitz::Array<float, 2> math::GetSubArrCopy(const blitz::Array<float, 2> &A2_In, const blitz::Array<int, 3> &I_A3_Indices);
  template blitz::Array<double, 2> math::GetSubArrCopy(const blitz::Array<double, 2> &A2_In, const blitz::Array<int, 3> &I_A3_Indices);
  
  template bool math::CountPixGTZero(blitz::Array<unsigned short, 1> &vec_InOut);
  template bool math::CountPixGTZero(blitz::Array<int, 1> &vec_InOut);
  template bool math::CountPixGTZero(blitz::Array<long, 1> &vec_InOut);
  template bool math::CountPixGTZero(blitz::Array<float, 1> &vec_InOut);
  template bool math::CountPixGTZero(blitz::Array<double, 1> &vec_InOut);
  
  template int math::FirstIndexWithValueGEFrom(const blitz::Array<unsigned short, 1> &vecIn, 
                                               const unsigned short minValue, 
                                               const int fromIndex);
  template int math::FirstIndexWithValueGEFrom(const blitz::Array<int, 1> &vecIn, 
                                               const int minValue, 
                                               const int fromIndex);
  template int math::FirstIndexWithValueGEFrom(const blitz::Array<long, 1> &vecIn, 
                                               const long minValue, 
                                               const int fromIndex);
  template int math::FirstIndexWithValueGEFrom(const blitz::Array<float, 1> &vecIn, 
                                               const float minValue, 
                                               const int fromIndex);
  template int math::FirstIndexWithValueGEFrom(const blitz::Array<double, 1> &vecIn, 
                                               const double minValue, 
                                               const int fromIndex);
  
  template int math::LastIndexWithZeroValueBefore(const blitz::Array<unsigned short, 1> &vec_In, 
                                                  const int startPos_In);
  template int math::LastIndexWithZeroValueBefore(const blitz::Array<int, 1> &vec_In, 
                                                  const int startPos_In);
  template int math::LastIndexWithZeroValueBefore(const blitz::Array<long, 1> &vec_In, 
                                                  const int startPos_In);
  template int math::LastIndexWithZeroValueBefore(const blitz::Array<float, 1> &vec_In, 
                                                  const int startPos_In);
  template int math::LastIndexWithZeroValueBefore(const blitz::Array<double, 1> &vec_In, 
                                                  const int startPos_In);
  
  template int math::FirstIndexWithZeroValueFrom(const blitz::Array<unsigned short, 1> &vec_In, 
                                                 const int startPos_In);
  template int math::FirstIndexWithZeroValueFrom(const blitz::Array<int, 1> &vec_In, 
                                                 const int startPos_In);
  template int math::FirstIndexWithZeroValueFrom(const blitz::Array<long, 1> &vec_In, 
                                                 const int startPos_In);
  template int math::FirstIndexWithZeroValueFrom(const blitz::Array<float, 1> &vec_In, 
                                                 const int startPos_In);
  template int math::FirstIndexWithZeroValueFrom(const blitz::Array<double, 1> &vec_In, 
                                                 const int startPos_In);
  
  template unsigned short math::Median(const blitz::Array<unsigned short, 1> &Arr);
  template int math::Median(const blitz::Array<int, 1> &Arr);
  template long math::Median(const blitz::Array<long, 1> &Arr);
  template float math::Median(const blitz::Array<float, 1> &Arr);
  template double math::Median(const blitz::Array<double, 1> &Arr);

  template unsigned short math::Median(const blitz::Array<unsigned short, 2> &Arr, bool b);
  template int math::Median(const blitz::Array<int, 2> &Arr, bool b);
  template long math::Median(const blitz::Array<long, 2> &Arr, bool b);
  template float math::Median(const blitz::Array<float, 2> &Arr, bool b);
  template double math::Median(const blitz::Array<double, 2> &Arr, bool b);

  template unsigned short math::Median(const blitz::Array<unsigned short, 1> &Arr, const blitz::Array<string, 1> &S_A1_Args_In, void *PP_Args[]);
  template int math::Median(const blitz::Array<int, 1> &Arr, const blitz::Array<string, 1> &S_A1_Args_In, void *PP_Args[]);
  template long math::Median(const blitz::Array<long, 1> &Arr, const blitz::Array<string, 1> &S_A1_Args_In, void *PP_Args[]);
  template float math::Median(const blitz::Array<float, 1> &Arr, const blitz::Array<string, 1> &S_A1_Args_In, void *PP_Args[]);
  template double math::Median(const blitz::Array<double, 1> &Arr, const blitz::Array<string, 1> &S_A1_Args_In, void *PP_Args[]);

  template blitz::Array<unsigned short, 1> math::MedianVec(const blitz::Array<unsigned short, 1> &arr, int Width, const string &Mode="NORMAL");
  template blitz::Array<int, 1> math::MedianVec(const blitz::Array<int, 1> &arr, int Width, const string &Mode="NORMAL");
  template blitz::Array<long, 1> math::MedianVec(const blitz::Array<long, 1> &arr, int Width, const string &Mode="NORMAL");
  template blitz::Array<float, 1> math::MedianVec(const blitz::Array<float, 1> &arr, int Width, const string &Mode="NORMAL");
  template blitz::Array<double, 1> math::MedianVec(const blitz::Array<double, 1> &arr, int Width, const string &Mode="NORMAL");
  
  template unsigned short math::Select(const blitz::Array<unsigned short, 1> &arr, int KThSmallest);
  template int math::Select(const blitz::Array<int, 1> &arr, int KThSmallest);
  template long math::Select(const blitz::Array<long, 1> &arr, int KThSmallest);
  template float math::Select(const blitz::Array<float, 1> &arr, int KThSmallest);
  template double math::Select(const blitz::Array<double, 1> &arr, int KThSmallest);
  
  template blitz::Array<unsigned short, 1> math::BubbleSort(const blitz::Array<unsigned short, 1> &I_A1_ArrIn);
  template blitz::Array<int, 1> math::BubbleSort(const blitz::Array<int, 1> &I_A1_ArrIn);
  template blitz::Array<long, 1> math::BubbleSort(const blitz::Array<long, 1> &I_A1_ArrIn);
  template blitz::Array<float, 1> math::BubbleSort(const blitz::Array<float, 1> &I_A1_ArrIn);
  template blitz::Array<double, 1> math::BubbleSort(const blitz::Array<double, 1> &I_A1_ArrIn);

  template bool math::Uniq(const blitz::Array<int, 1> &IA1_In, blitz::Array<int, 1> &IA1_Result);
  template bool math::Uniq(const blitz::Array<long, 1> &IA1_In, blitz::Array<int, 1> &IA1_Result);
  template bool math::Uniq(const blitz::Array<float, 1> &IA1_In, blitz::Array<int, 1> &IA1_Result);
  template bool math::Uniq(const blitz::Array<double, 1> &IA1_In, blitz::Array<int, 1> &IA1_Result);
  
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
  
  template bool utils::WriteFits(const blitz::Array<unsigned short, 2>* image_In, const string &fileName_In);
  template bool utils::WriteFits(const blitz::Array<int, 2>* image_In, const string &fileName_In);
  template bool utils::WriteFits(const blitz::Array<long, 2>* image_In, const string &fileName_In);
  template bool utils::WriteFits(const blitz::Array<float, 2>* image_In, const string &fileName_In);
  template bool utils::WriteFits(const blitz::Array<double, 2>* image_In, const string &fileName_In);
  
  template bool utils::WriteFits(const blitz::Array<unsigned short, 1>* image_In, const string &fileName_In);
  template bool utils::WriteFits(const blitz::Array<int, 1>* image_In, const string &fileName_In);
  template bool utils::WriteFits(const blitz::Array<long, 1>* image_In, const string &fileName_In);
  template bool utils::WriteFits(const blitz::Array<float, 1>* image_In, const string &fileName_In);
  template bool utils::WriteFits(const blitz::Array<double, 1>* image_In, const string &fileName_In);
  
  template bool utils::WriteArrayToFile(const blitz::Array<unsigned short, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  template bool utils::WriteArrayToFile(const blitz::Array<int, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  template bool utils::WriteArrayToFile(const blitz::Array<long, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  template bool utils::WriteArrayToFile(const blitz::Array<float, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  template bool utils::WriteArrayToFile(const blitz::Array<double, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  
  template bool utils::WriteArrayToFile(const blitz::Array<unsigned short, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  template bool utils::WriteArrayToFile(const blitz::Array<int, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  template bool utils::WriteArrayToFile(const blitz::Array<long, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  template bool utils::WriteArrayToFile(const blitz::Array<float, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  template bool utils::WriteArrayToFile(const blitz::Array<double, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);

//  template PTR(afwImage::MaskedImage<float, unsigned short, float>) utils::getShared(afwImage::MaskedImage<float, unsigned short, float> const &maskedImage);
//  template PTR(afwImage::MaskedImage<double, unsigned short, double>) utils::getShared(afwImage::MaskedImage<double, unsigned short, double> const &maskedImage);
  }}}
