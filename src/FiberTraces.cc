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
  _profile(10, height),
  _maskedImage(width, height),
  _xCenters(height),
  _spectrum(height),
  _background(height),
  _fiberTraceFunction(), 
  _fiberTraceExtractionControl()
  {
    _isXCentersCalculated = false;
    _isImageSet = false;
    _isTraceSet = false;
    _isProfileSet = false;
    _isFiberTraceFunctionSet = false;
  }
  
  /** @brief Construct an Exposure with a blank MaskedImage of specified size (default 0x0) and
   * a Wcs (which may be default constructed)
   */          
  template<typename ImageT, typename MaskT, typename VarianceT> 
  pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    afwGeom::Extent2I const & dimensions ///< desired image width/height
  ) :
  _trace(dimensions),
  _profile(dimensions),
  _maskedImage(dimensions),
  _xCenters(dimensions.getY()),
  _spectrum(dimensions.getY()),
  _background(dimensions.getY()),
  _fiberTraceFunction(), 
  _fiberTraceExtractionControl()
  {
    _isXCentersCalculated = false;
    _isImageSet = false;
    _isTraceSet = false;
    _isProfileSet = false;
    _isFiberTraceFunctionSet = false;
  }

  template<typename ImageT, typename MaskT, typename VarianceT> 
  pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    MaskedImageT const & maskedImage ///< desired image width/height
  ) :
  _trace(maskedImage.getWidth(), maskedImage.getHeight()),
  _profile(maskedImage.getWidth(), maskedImage.getHeight()),
  _maskedImage(maskedImage),
  _xCenters(maskedImage.getHeight()),
  _spectrum(maskedImage.getHeight()),
  _background(maskedImage.getHeight()),
  _fiberTraceFunction(), 
  _fiberTraceExtractionControl()
  {
    _isXCentersCalculated = false;
    _isImageSet = true;
    _isTraceSet = false;
    _isProfileSet = false;
    _isFiberTraceFunctionSet = false;
  }
  
  /** **************************************************************/
  
  /// Set the CCD image to image
  template<typename ImageT, typename MaskT, typename VarianceT> 
  void pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setMaskedImage( MaskedImageT & image){ 
    _maskedImage = image; 
    _isImageSet = true;
    _trace = MaskedImageT(_trace.getWidth(), _maskedImage.getHeight());
  }
  
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

  /** *******************************************************************************************************/
  
  /// Calculate the x-centers of the fiber trace
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::calculateXCenters(){//FiberTraceFunctionControl const& fiberTraceFunctionControl){
    if (!_isImageSet){
      cout << "FiberTrace::calculateXCenters: ERROR: _maskedImage is not set => returning FALSE" << endl;
      return false;
    }
    
    int cIndex = 0;
    double D_YMin = 0.;
    double D_YMax = 0.;
      
    blitz::Array<double, 1> D_A1_TempCen(_maskedImage.getHeight());
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
                                        double(_maskedImage.getHeight()),//double(int(_fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh)+1),
                                        static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xLow),
                                        static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xHigh),
                                        _fiberTraceFunction.fiberTraceFunctionControl.order,
                                        _maskedImage.getWidth(),
                                        _maskedImage.getHeight())){
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
                                         _maskedImage.getHeight(),//double(static_cast<int>(_fiberTraceFunction.yCenter) + static_cast<int>(_fiberTraceFunction.yHigh)+1),
                                         static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xLow),
                                         static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xHigh),
                                         static_cast<int>(_fiberTraceFunction.fiberTraceFunctionControl.order),
                                         _maskedImage.getWidth(),
                                         _maskedImage.getHeight())){
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
                                            _maskedImage.getHeight(),//double(static_cast<int>(_fiberTraceFunction.yCenter) + static_cast<int>(_fiberTraceFunction.yHigh)+1),
                                            static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xLow),
                                            static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xHigh),
                                            static_cast<int>(_fiberTraceFunction.fiberTraceFunctionControl.order),
                                            _maskedImage.getWidth(),
                                            _maskedImage.getHeight())){
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
                                           _maskedImage.getHeight(),//double(static_cast<int>(_fiberTraceFunction.yCenter) + static_cast<int>(_fiberTraceFunction.yHigh) + 1),
                                           static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xLow),
                                           static_cast<double>(_fiberTraceFunction.fiberTraceFunctionControl.xHigh),
                                           static_cast<int>(_fiberTraceFunction.fiberTraceFunctionControl.order),
                                           _maskedImage.getWidth(),
                                           _maskedImage.getHeight())){
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
      blitz::Array<double,1> D_A1_XRow(_maskedImage.getHeight());
      for (int i=0; i < _maskedImage.getHeight(); i++){
        D_A1_XRow(i) = double(i);
      }
      #ifdef __DEBUG_TRACEFUNC__
        cout << "FiberTrace.calculateXCenters: fiberTraceFunctionCoefficients = " << fiberTraceFunctionCoefficients << endl;
      #endif
      blitz::Array<double,1> *P_D_A1_TempCen = pfsDRPStella::math::Poly(D_A1_XRow, fiberTraceFunctionCoefficients);
      D_A1_TempCen = (*P_D_A1_TempCen);
      delete(P_D_A1_TempCen);
    }
    
    /// Check limits
    blitz::Array<double, 1> D_A1_XLow(_maskedImage.getHeight());
    D_A1_XLow = D_A1_TempCen + _fiberTraceFunction.fiberTraceFunctionControl.xLow;
    blitz::Array<double, 1> D_A1_XHigh(_maskedImage.getHeight());
    D_A1_XHigh = D_A1_TempCen + _fiberTraceFunction.fiberTraceFunctionControl.xHigh;
    blitz::Array<int, 1> I_A1_Where(_maskedImage.getHeight());
    I_A1_Where = blitz::where(D_A1_XLow < -0.5, 0, 1);
    int I_NInd;
    blitz::Array<int, 1> *P_I_A1_WhereInd = pfsDRPStella::math::GetIndex(I_A1_Where, I_NInd);
    D_YMin = (*P_I_A1_WhereInd)(0);
    delete(P_I_A1_WhereInd);
    I_A1_Where = blitz::where(D_A1_XHigh < _maskedImage.getWidth()-0.5, 1, 0);
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
      if (D_A1_XHigh(i) > _maskedImage.getWidth()-0.5){
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
    if (D_A1_XHigh(_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow) > _maskedImage.getWidth()-0.5){
      cout << "FiberTrace.calculateXCenters: ERROR: D_A1_XHigh(_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow=" << _fiberTraceFunction.yCenter + _fiberTraceFunction.yLow << ")=" << D_A1_XHigh(_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow) << " > _maskedImage.getWidth()-0.5 =" << _maskedImage.getWidth()-0.5 << endl;
      return false;
    }
    if (D_A1_XHigh(_fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh) > _maskedImage.getWidth()-0.5){
      cout << "FiberTrace.calculateXCenters: ERROR: D_A1_XHigh(_fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh=" << _fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh << ")=" << D_A1_XHigh(_fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh) << " > _maskedImage.getWidth()-0.5=" << _maskedImage.getWidth()-0.5 << endl;
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
    if (!_isImageSet){
      cout << "FiberTrace::setXCenters: ERROR: _maskedImage has not been set" << endl;
      return false;
    }
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
      if ((xCenters[i] < -0.5) || (xCenters[i] > _maskedImage.getWidth()-0.5)){
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
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setImage(PTR(afwImage::Image<ImageT>) & image){
    
    /// Check input image size
    if (_isImageSet){
      if (image->getWidth() != _maskedImage.getWidth()){
        cout << "FiberTrace.setXCenters: ERROR: image.getWidth(=" << image->getWidth() << ") != _maskedImage.getWidth(=" << _maskedImage.getWidth() << ") => Returning false" << endl;
        return false;
      }
      if (image->getHeight() != _maskedImage.getHeight()){
        cout << "FiberTrace.setXCenters: ERROR: image.getHeight(=" << image->getHeight() << ") != _maskedImage.getHeight(=" << _maskedImage.getHeight() << ") => Returning false" << endl;
        return false;
      }
    }

    _maskedImage.getImage() = image;
    
    _isImageSet = true;
    return true;
  }
  
  /// Set the mask pointer of this fiber trace to mask
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setMask(PTR(afwImage::Mask<MaskT>) & mask){
    
    /// Check input mask size
    if (_isImageSet){
      if (mask->getWidth() != _maskedImage.getWidth()){
        cout << "FiberTrace.setXCenters: ERROR: mask.getWidth(=" << mask->getWidth() << ") != _maskedImage.getWidth(=" << _maskedImage.getWidth() << ") => Returning false" << endl;
        return false;
      }
      if (mask->getHeight() != _maskedImage.getHeight()){
        cout << "FiberTrace.setXCenters: ERROR: mask.getHeight(=" << mask->getHeight() << ") != _maskedImage.getHeight(=" << _maskedImage.getHeight() << ") => Returning false" << endl;
        return false;
      }
    }
    
    _maskedImage.getMask() = mask;
    
    return true;
  }
  
  /// Set the variance pointer of this fiber trace to variance
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setVariance(PTR(afwImage::Image<VarianceT>) & variance){
    
    /// Check input variance size
    if (_isImageSet){
      if (variance->getWidth() != _maskedImage.getWidth()){
        cout << "FiberTrace.setXCenters: ERROR: variance.getWidth(=" << variance->getWidth() << ") != _maskedImage.getWidth(=" << _maskedImage.getWidth() << ") => Returning false" << endl;
        return false;
      }
      if (variance->getHeight() != _maskedImage.getHeight()){
        cout << "FiberTrace.setXCenters: ERROR: variance.getHeight(=" << variance->getHeight() << ") != _maskedImage.getHeight(=" << _maskedImage.getHeight() << ") => Returning false" << endl;
        return false;
      }
    }
    
    _maskedImage.getVariance() = variance;
    
    return true;
  }
  
  /// Set the profile image of this fiber trace to profile
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setTrace(MaskedImageT & trace){
    if (!_isFiberTraceFunctionSet){
      cout << "FiberTrace::setTrace: ERROR: _fiberTraceFunction not set => Returning FALSE" << endl;
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
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::setProfile(afwImage::Image<float> &profile){
    if (!_isFiberTraceFunctionSet){
      cout << "FiberTrace::setProfile: ERROR: _fiberTraceFunction not set => Returning FALSE" << endl;
      return false;
    }
    if (!_isTraceSet){
      cout << "FiberTrace::setProfile: ERROR: _trace not set => Returning FALSE" << endl;
      return false;
    }
    
    /// Check input profile size
    if (profile.getWidth() != _trace.getWidth()){
      cout << "FiberTrace.setProfile: ERROR: profile.getWidth(=" << profile.getWidth() << ") != _trace.getWidth(=" << _trace.getWidth() << ") => Returning false" << endl;
      return false;
    }
    if (profile.getHeight() != _trace.getHeight()){
      cout << "FiberTrace.setProfile: ERROR: profile.getHeight(=" << profile.getHeight() << ") != _trace.getHeight(=" << _trace.getHeight() << ") => Returning false" << endl;
      return false;
    }

    _profile = afwImage::Image<float>(profile.getDimensions());
    _profile = profile;
    
    _isProfileSet = true;
    return true;
  }
  
  /// Set the profile image of this fiber trace to profile
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::extractFromProfile(){
    if (!_isTraceSet){
      cout << "FiberTrace.extractFromProfile: ERROR: _trace is not set" << endl;
    }
    if (!_isProfileSet){
      cout << "FiberTrace.extractFromProfile: ERROR: _profile is not set" << endl;
    }
    return false;
  }
  /**************************************************************************
   * createTrace
   * ************************************************************************/
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTrace<ImageT, MaskT, VarianceT>::createTrace(){
    if (!_isImageSet){
      cout << "FiberTrace::createTrace: ERROR: _maskedImage has not been set" << endl;
      return false;
    }
    if (!_isXCentersCalculated){
      cout << "FiberTrace::createTrace: ERROR: _xCenters has not been set/calculated" << endl;
      return false;
    }
    
    blitz::Array<int, 2> minCenMax(2,2);
    minCenMax = 0;
    blitz::Array<float, 1> xCenters(_xCenters.data(), blitz::shape(_xCenters.size()), blitz::neverDeleteData);
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
    #ifdef __DEBUG_CREATEFIBERTRACE__
      cout << "FiberTrace::CreateFiberTrace: minCenMax = " << minCenMax << endl;
    #endif
    
    _trace = MaskedImageT(minCenMax(0,2) - minCenMax(0,0) + 1, _fiberTraceFunction.yHigh - _fiberTraceFunction.yLow + 1);// minCenMax.rows());

    ndarray::Array<ImageT, 2, 1> imageArray = _maskedImage.getImage()->getArray();
    ndarray::Array<ImageT, 2, 1> traceArray = _trace.getImage()->getArray();
    typename ndarray::Array<ImageT, 2, 1>::Iterator yIterTrace = traceArray.begin();// + _fiberTraceFunction.yCenter + _fiberTraceFunction.yLow;
    int iy = 0;//_fiberTraceFunction.yCenter + _fiberTraceFunction.yLow;
    for (typename ndarray::Array<ImageT, 2, 1>::Iterator yIter = imageArray.begin() + _fiberTraceFunction.yCenter + _fiberTraceFunction.yLow; yIter != imageArray.begin() + _fiberTraceFunction.yCenter + _fiberTraceFunction.yHigh + 1; ++yIter) {
      typename ndarray::Array<ImageT, 2, 1>::Reference::Iterator ptrImageStart = yIter->begin() + minCenMax(iy, 0);
      typename ndarray::Array<ImageT, 2, 1>::Reference::Iterator ptrImageEnd = yIter->begin() + minCenMax(iy, 2) + 1;
      typename ndarray::Array<ImageT, 2, 1>::Reference::Iterator ptrTraceStart = yIterTrace->begin();
      std::copy(ptrImageStart, ptrImageEnd, ptrTraceStart);
      ++yIterTrace;
      ++iy;
    }
    return true;
  }

  /** 
   * class FiberTraceSet
   **/
  
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::setFiberTrace(int const i,     ///< which aperture?
                                                                            PTR(FiberTrace<ImageT>) trace ///< the FiberTrace for the ith aperture
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
  void pfsDRPStella::FiberTraceSet<ImageT, MaskT, VarianceT>::addFiberTrace(PTR(FiberTrace<ImageT>) trace) ///< the FiberTrace for the ith aperture
  {
    _traces.push_back(trace);
  }

  
  template<typename ImageT, typename MaskT, typename VarianceT> 
  bool pfsDRPStella::MaskedSpectrographImage<ImageT, MaskT, VarianceT>::findAndTraceApertures(
    const pfsDRPStella::FiberTraceFunctionFindingControl &fiberTraceFunctionFindingControl,
    int minLength_In,
    int maxLength_In,
    int nLost_In){

    #ifdef __DEBUG_FINDANDTRACE__
      cout << "pfsDRPStella::MaskedSpectrographImage::findAndTraceApertures started" << endl;
    #endif
    
    if (static_cast<int>(fiberTraceFunctionFindingControl.apertureFWHM * 2.) + 1 <= fiberTraceFunctionFindingControl.nTermsGaussFit){
      cout << "CFits::FindAndTraceApertures: WARNING: fiberTraceFunctionFindingControl.apertureFWHM too small for GaussFit -> Try lower fiberTraceFunctionFindingControl.nTermsGaussFit!" << endl;
      return false;
    }
    _fiberTraceSet.getTraces().resize(0);
    FiberTraceFunction fiberTraceFunction;
    fiberTraceFunction.fiberTraceFunctionControl = fiberTraceFunctionFindingControl.fiberTraceFunctionControl;
    #ifdef __DEBUG_FINDANDTRACE__
      cout << "pfsDRPStella::MaskedSpectrographImage::findAndTraceApertures: fiberTraceFunction.fiberTraceFunctionControl set" << endl;
    #endif
    
    //  int I_PolyFitOrder = 3;
    int I_ApertureNumber = 0;
    int I_StartIndex;
    int I_FirstWideSignal;
    int I_FirstWideSignalEnd;
    int I_FirstWideSignalStart;
    int I_Length, I_ApertureLost, I_Row_Bak;//, I_LastRowWhereApertureWasFound
    int I_ApertureLength;
    int I_NInd;
    double D_Max;
    bool B_ApertureFound;
    blitz::Array<ImageT, 2> ccdTImage = utils::ndarrayToBlitz(_maskedImage.getImage()->getArray());
    #ifdef __DEBUG_FINDANDTRACE__
      cout << "pfsDRPStella::MaskedSpectrographImage::findAndTraceApertures: ccdTImage(60, *) = " << ccdTImage(60, blitz::Range::all()) << endl;
    #endif
    blitz::Array<double, 2> ccdImage(2,2);
    pfsDRPStella::math::Double(ccdTImage, ccdImage);
    #ifdef __DEBUG_FINDANDTRACE__
      cout << "pfsDRPStella::MaskedSpectrographImage::findAndTraceApertures: ccdImage(60,*) = " << ccdImage(60, blitz::Range::all()) << endl;
    #endif
    blitz::Array<double, 1> D_A1_IndexCol = pfsDRPStella::math::DIndGenArr(ccdImage.cols());
    blitz::Array<double, 1> D_A1_X(10);
    blitz::Array<double, 1> D_A1_Y(10);
    blitz::Array<double, 1> D_A1_MeasureErrors(10);
    blitz::Array<double, 1> D_A1_Guess(fiberTraceFunctionFindingControl.nTermsGaussFit);
    blitz::Array<double, 1> D_A1_GaussFit_Coeffs(fiberTraceFunctionFindingControl.nTermsGaussFit);
    blitz::Array<double, 1> D_A1_GaussFit_Coeffs_Bak(fiberTraceFunctionFindingControl.nTermsGaussFit);
    #ifdef __DEBUG_FINDANDTRACE__
      cout << "pfsDRPStella::MaskedSpectrographImage::findAndTraceApertures: D_A1_IndexCol = " << D_A1_IndexCol << endl;
    #endif
    blitz::Array<int, 1> I_A1_Signal(_maskedImage.getWidth());
    blitz::Array<double, 1> D_A1_ApertureCenter(_maskedImage.getHeight());
    blitz::Array<int, 1> I_A1_ApertureCenterInd(_maskedImage.getHeight());
    blitz::Array<double, 1> D_A1_ApertureCenterIndex(_maskedImage.getHeight());
    blitz::Array<int, 1> I_A1_ApertureCenterIndex(_maskedImage.getHeight());
    blitz::Array<double, 1> D_A1_ApertureCenterPos(1);
    blitz::Array<double, 1> *P_D_A1_PolyFitCoeffs = new blitz::Array<double, 1>(fiberTraceFunctionFindingControl.fiberTraceFunctionControl.order);
    blitz::Array<int, 1> I_A1_IndSignal(2);
    #ifdef __DEBUG_FINDANDTRACE__
      cout << "CFits::FindAndTraceApertures: started" << endl;
      blitz::Array<double, 2> D_A2_PixArrayNew(_maskedImage.getHeight(), _maskedImage.getWidth());
      D_A2_PixArrayNew = 0.;
    #endif
    blitz::Array<int, 1> I_A1_Ind(1);
    blitz::Array<int, 1> I_A1_Where(1);
    #ifdef __DEBUG_FINDANDTRACE__
      cout << "pfsDRPStella::MaskedSpectrographImage::findAndTraceApertures: fiberTraceFunctionFindingControl.signalThreshold = " << fiberTraceFunctionFindingControl.signalThreshold << endl;
    #endif
    
    /// Set all pixels below fiberTraceFunctionFindingControl.signalThreshold to 0.
    ccdImage = blitz::where(ccdImage < fiberTraceFunctionFindingControl.signalThreshold, 0., ccdImage);
    #ifdef __DEBUG_FINDANDTRACE__
      cout << "pfsDRPStella::MaskedSpectrographImage::findAndTraceApertures: ccdImage(60,*) = " << ccdImage(60, blitz::Range::all()) << endl;
      string S_TempFileName = "/home/azuri/spectra/pfs/FindAndTraceApertures_thresh.fits";
      pfsDRPStella::util::WriteFits(&ccdImage, S_TempFileName);
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
    for (int i_Row = 0; i_Row < _maskedImage.getHeight(); i_Row++){
//      if (i_Row == 60)
//        return false;
      I_StartIndex = 0;
      B_ApertureFound = false;
      while (!B_ApertureFound){
        #ifdef __DEBUG_FINDANDTRACE__
          cout << "CFits::FindAndTraceApertures: ccdImage(i_Row = " << i_Row << ", *) = " << ccdImage(i_Row, blitz::Range::all()) << endl;
        #endif
        
        I_A1_Signal = blitz::where(ccdImage(i_Row, blitz::Range::all()) > 0., 1, 0);
        if (!pfsDRPStella::math::CountPixGTZero(I_A1_Signal)){
          cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: pfsDRPStella::math::CountPixGTZero(I_A1_Signal=" << I_A1_Signal << ") returned FALSE => Returning FALSE" << endl;
          return false;
        }
        I_FirstWideSignal = pfsDRPStella::math::FirstIndexWithValueGEFrom(I_A1_Signal, I_MinWidth, I_StartIndex);
        #ifdef __DEBUG_FINDANDTRACE__
          cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_FirstWideSignal found at index " << I_FirstWideSignal << ", I_StartIndex = " << I_StartIndex << endl;
        #endif
        if (I_FirstWideSignal > 0){
          I_FirstWideSignalStart = pfsDRPStella::math::LastIndexWithZeroValueBefore(I_A1_Signal, I_FirstWideSignal) + 1;
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "CFits::FindAndTraceApertures: while: 1. i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_FirstWideSignalStart = " << I_FirstWideSignalStart << endl;
          #endif
            
          I_FirstWideSignalEnd = pfsDRPStella::math::FirstIndexWithZeroValueFrom(I_A1_Signal, I_FirstWideSignal) - 1;
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
          #endif
            
          if (I_FirstWideSignalStart < 0){
            #ifdef __DEBUG_FINDANDTRACE__
              cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: No start of aperture found -> Going to next Aperture." << endl;
            #endif

            if (I_FirstWideSignalEnd < 0){
              cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": 1. WARNING: No end of aperture found -> Going to next row." << endl;
              break;
            }
            else{
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": End of first wide signal found at index " << I_FirstWideSignalEnd << endl;
              #endif
              /// Set start index for next run
              I_StartIndex = I_FirstWideSignalEnd+1;
            }
          }
          else{ /// Fit Gaussian and Trace Aperture
            if (I_FirstWideSignalEnd < 0){
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": 2. WARNING: No end of aperture found -> Going to next row." << endl;
                cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_Row_Bak = " << I_Row_Bak << endl;
                cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": B_ApertureFound = " << B_ApertureFound << endl;
              #endif
              break;
            }
            else{
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": End of first wide signal found at index " << I_FirstWideSignalEnd << endl;
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
              //I_LastRowWhereApertureWasFound = i_Row;
              I_A1_Where.resize(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
              I_A1_Where = blitz::where(fabs(ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)) - max(ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)))) < 0.00001, 1, 0);
              if (!pfsDRPStella::math::GetIndex(I_A1_Where, I_NInd, I_A1_Ind)){
                cout << "CFits::FindAndTraceApertures: while: ERROR: GetIndex(I_A1_Where=" << I_A1_Where << ") returned FALSE => Returning FALSE" << endl;
                return false;
              }
              D_A1_ApertureCenter(i_Row) = I_FirstWideSignalStart + I_A1_Ind(0);
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Aperture found at " << D_A1_ApertureCenter(i_Row) << endl;
              #endif
              
              /// Set signal to zero
              ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
            }
            else{
              if (I_Length <= fiberTraceFunctionFindingControl.nTermsGaussFit){
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Width of aperture <= " << fiberTraceFunctionFindingControl.nTermsGaussFit << "-> abandoning aperture" << endl;
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
                  cout << "CFits::FindAndTraceApertures: 1. D_A1_Y set to " << D_A1_Y << endl;
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
                  cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_X = " << D_A1_X << ", D_A1_Y = " << D_A1_Y << endl;
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
                  cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": 1. starting MPFitGaussLim: D_A1_Guess = " << D_A1_Guess << ", I_A2_Limited = " << I_A2_Limited << ", D_A2_Limits = " << D_A2_Limits << endl;
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
                    cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: GaussFit FAILED -> abandoning aperture" << endl;
                  #endif
                
                  /// Set start index for next run
                  I_StartIndex = I_FirstWideSignalEnd+1;
                
                  ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                }
                else{
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
//                    return false;
                    if (D_A1_GaussFit_Coeffs(0) < fiberTraceFunctionFindingControl.saturationLevel/5.){
                      cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Signal less than 20% of saturation level" << endl;
                    }
                    if (D_A1_GaussFit_Coeffs(0) > fiberTraceFunctionFindingControl.saturationLevel){
                      cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Signal appears to be saturated" << endl;
                    }
                    if ((D_A1_GaussFit_Coeffs(1) < double(I_FirstWideSignalStart) + (double(I_Length)/4.)) || (D_A1_GaussFit_Coeffs(1) > double(I_FirstWideSignalStart) + (double(I_Length) * 3./4.))){
                      cout << "CFits::FindAndTraceApertures: while: Warning: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Center of Gaussian far away from middle of signal" << endl;
                    }
                  #endif
                  if ((D_A1_GaussFit_Coeffs(1) < double(I_FirstWideSignalStart)) || (D_A1_GaussFit_Coeffs(1) > double(I_FirstWideSignalEnd))){
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                    #endif
                    /// Set signal to zero
                    ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                      
                      
                    /// Set start index for next run
                    I_StartIndex = I_FirstWideSignalEnd+1;
                  }
                  else{
                    if ((D_A1_GaussFit_Coeffs(2) < fiberTraceFunctionFindingControl.apertureFWHM / 4.) || (D_A1_GaussFit_Coeffs(2) > fiberTraceFunctionFindingControl.apertureFWHM)){
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: FWHM = " << D_A1_GaussFit_Coeffs(2) << " outside range -> abandoning aperture" << endl;
                      #endif
                      /// Set signal to zero
                      ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "CFits::FindAndTraceApertures: while: B_ApertureFound = " << B_ApertureFound << ": 1. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                        cout << "CFits::FindAndTraceApertures: while: 1. ccdImage(i_Row = " << i_Row << ", blitz::Range(I_FirstWideSignalStart = " << I_FirstWideSignalStart << ", I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << ")) set to " << ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)) << endl;
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
                        cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Aperture found at " << D_A1_ApertureCenter(i_Row) << endl;
                      #endif
                      ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                    }
                  }/// end else if ((D_A1_GaussFit_Coeffs(1) > double(I_FirstWideSignalStart)) && (D_A1_GaussFit_Coeffs(1) < double(I_FirstWideSignalEnd)))
                }/// else else if (GaussFit returned TRUE)
              }/// end else if (I_Length >= 4)
            }/// end else if GaussFit
          }/// end else if (I_FirstWideSignalStart > 0)
        }/// end if (I_FirstWideSignal > 0)
        else{
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": No Aperture found in row " << i_Row << ", trying next row" << endl;
          #endif
          break;
        }/// end else if (I_FirstWideSignal < 0)
      }/// end while (!B_ApertureFound)
      
      if (B_ApertureFound){
        /// Trace Aperture
        int I_Length = 1;
        I_ApertureLost = 0;
//        #ifdef __DEBUG_FINDANDTRACE__
          cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Starting to trace aperture" << endl;
//        #endif
        D_A1_GaussFit_Coeffs_Bak = D_A1_GaussFit_Coeffs;
        I_Row_Bak = i_Row;
        while(B_ApertureFound && (I_ApertureLost < nLost_In) && (i_Row < _maskedImage.getHeight() - 2) && I_Length < maxLength_In){
          i_Row++;
          I_Length++;
          if (fiberTraceFunctionFindingControl.nTermsGaussFit == 0){/// look for maximum only
            B_ApertureFound = true;
            //I_LastRowWhereApertureWasFound = i_Row;
            //I_FirstWideSignalStart
            D_Max = max(ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)));
            if (D_Max < fiberTraceFunctionFindingControl.signalThreshold){
              I_ApertureLost++;
            }
            else{
              I_A1_Where.resize(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
              I_A1_Where = blitz::where(fabs(ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)) - D_Max) < 0.00001, 1, 0);
              if (!pfsDRPStella::math::GetIndex(I_A1_Where, I_NInd, I_A1_Ind)){
                cout << "CFits::FindAndTraceApertures: ERROR: GetIndex(I_A1_Where=" << I_A1_Where << ") returned FALSE => Returning FALSE" << endl;
                return false;
              }
              D_A1_ApertureCenter(i_Row) = I_FirstWideSignalStart + I_A1_Ind(0);
              #ifdef __DEBUG_FINDANDTRACE__
              cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Aperture found at " << D_A1_ApertureCenter(i_Row) << endl;
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
            if (I_FirstWideSignalStart < 0. || I_FirstWideSignalEnd >= _maskedImage.getWidth()){
              #ifdef __DEBUG_FINDANDTRACE__
                cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": start or end of aperture outside CCD -> Aperture lost";
              #endif
              /// Set signal to zero
              if (I_FirstWideSignalStart < 0)
                I_FirstWideSignalStart = 0;
              if (I_FirstWideSignalEnd >= _maskedImage.getWidth())
                I_FirstWideSignalEnd = _maskedImage.getWidth() - 1;
              ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
              I_ApertureLost++;
            }
            else{
              I_Length = I_FirstWideSignalEnd - I_FirstWideSignalStart + 1;
              
              if (I_Length <= fiberTraceFunctionFindingControl.nTermsGaussFit){
                #ifdef __DEBUG_FINDANDTRACE__
                  cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Warning: Width of Aperture <= " << fiberTraceFunctionFindingControl.nTermsGaussFit << " -> Lost Aperture" << endl;
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
                  cout << "CFits::FindAndTraceApertures: I_MinWidth = " << I_MinWidth << ": I_A1_IndSignal = " << I_A1_IndSignal << endl;
                #endif
                if (blitz::sum(I_A1_IndSignal) < I_MinWidth){
                  /// Set signal to zero
                  ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                  I_ApertureLost++;
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "CFits::FindAndTraceApertures: Signal not wide enough => Aperture lost" << endl;
                  #endif
                }
                else{
                  D_A1_Y = blitz::where(D_A1_Y < 0.00000001, 1., D_A1_Y);
                  #ifdef __DEBUG_FINDANDTRACE__
                    cout << "CFits::FindAndTraceApertures: 2. D_A1_Y set to " << D_A1_Y << endl;
                  #endif
                  D_A1_MeasureErrors = sqrt(fabs(D_A1_Y));
                  D_A1_Guess = D_A1_GaussFit_Coeffs_Bak;
                  
                  D_A1_GaussFit_Coeffs = 0.;
                  blitz::Array<double, 1> D_A1_GaussFit_ECoeffs(D_A1_GaussFit_Coeffs.size());
                  D_A1_GaussFit_ECoeffs = 0.;
                  
                  #ifdef __DEBUG_FINDANDTRACE__
                  cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_X = " << D_A1_X << ", D_A1_Y = " << D_A1_Y << endl;
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
                    cout << "CFits::FindAndTraceApertures: while: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": 2. starting MPFitGaussLim: D_A2_Limits = " << D_A2_Limits << endl;
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
                      cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Warning: GaussFit FAILED" << endl;
                    #endif
                    /// Set signal to zero
                    ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart-1, I_FirstWideSignalEnd+1)) = 0.;
                  
                    I_ApertureLost++;
                    //          return false;
                  }
                  else{
                    #ifdef __DEBUG_FINDANDTRACE__
                      cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
                      if (D_A1_GaussFit_Coeffs(0) < fiberTraceFunctionFindingControl.saturationLevel/5.){
                        cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Signal less than 20% of saturation level" << endl;
                      }
                      if (D_A1_GaussFit_Coeffs(0) > fiberTraceFunctionFindingControl.saturationLevel){
                        cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: Signal appears to be saturated" << endl;
                      }
                    #endif
                    //          if ((D_A1_GaussFit_Coeffs(1) < double(I_FirstWideSignalStart) - (double(I_Length)/4.)) || (D_A1_GaussFit_Coeffs(1) > double(I_FirstWideSignalStart) + (double(I_Length) * 3./4.))){
                    //            cout << "CFits::FindAndTraceApertures: Warning: i_Row = " << i_Row << ": Center of Gaussian far away from middle of signal" << endl;
                    //          }
                    if ((D_A1_GaussFit_Coeffs(1) < D_A1_GaussFit_Coeffs_Bak(1) - 1.) || (D_A1_GaussFit_Coeffs(1) > D_A1_GaussFit_Coeffs_Bak(1) + 1.)){
                      #ifdef __DEBUG_FINDANDTRACE__
                        cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                      #endif
                      /// Set signal to zero
                      ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                        
                      I_ApertureLost++;
                    }
                    else{
                      if ((D_A1_GaussFit_Coeffs(2) < fiberTraceFunctionFindingControl.apertureFWHM / 4.) || (D_A1_GaussFit_Coeffs(2) > fiberTraceFunctionFindingControl.apertureFWHM)){
                        #ifdef __DEBUG_FINDANDTRACE__
                          cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": WARNING: FWHM = " << D_A1_GaussFit_Coeffs(2) << " outside range -> abandoning aperture" << endl;
                        #endif
                        /// Set signal to zero
                        ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                        #ifdef __DEBUG_FINDANDTRACE__
                          cout << "CFits::FindAndTraceApertures: 2. Signal set to zero from I_FirstWideSignalStart = " << I_FirstWideSignalStart << " to I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
                        #endif
                        I_ApertureLost++;
                      }
                      else{
                        I_ApertureLost = 0;
                        B_ApertureFound = true;
                        D_A1_ApertureCenter(i_Row) = D_A1_GaussFit_Coeffs(1);
                        #ifdef __DEBUG_FINDANDTRACE__
                          cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": Aperture found at " << D_A1_ApertureCenter(i_Row) << endl;
                        #endif
                        D_A1_GaussFit_Coeffs_Bak = D_A1_GaussFit_Coeffs;
                        //I_LastRowWhereApertureWasFound = i_Row;
                      }
                    }/// end else if ((D_A1_GaussFit_Coeffs(1) >= D_A1_Guess(1) - 1.) && (D_A1_GaussFit_Coeffs(1) <= D_A1_Guess(1) + 1.))
                  }/// end else if (GaussFit(D_A1_X, D_A1_Y, D_A1_GaussFit_Coeffs, S_A1_KeyWords_GaussFit, PP_Args_GaussFit))
                  ccdImage(i_Row, blitz::Range(I_FirstWideSignalStart+1, I_FirstWideSignalEnd-1)) = 0.;
                }/// end else if (blitz::sum(I_A1_Signal) >= I_MinWidth){
              }/// end if (I_Length > 3)
            }/// end else if (I_ApertureStart >= 0. && I_ApertureEnd < _maskedImage.getWidth())
          }/// end else if GaussFit
        }///end while(B_ApertureFound && (I_ApertureLost < 3) && i_Row < _maskedImage.getHeight() - 2))
        
        /// Fit Polynomial to traced aperture positions
        I_A1_ApertureCenterIndex = blitz::where(D_A1_ApertureCenter > 0., 1, 0);
        #ifdef __DEBUG_FINDANDTRACE__
          cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_A1_ApertureCenterIndex = " << I_A1_ApertureCenterIndex << endl;
        #endif
        if (!pfsDRPStella::math::GetIndex(I_A1_ApertureCenterIndex, I_ApertureLength, I_A1_ApertureCenterInd)){
          cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: pfsDRPStella::math::GetIndex(I_A1_ApertureCenterIndex=" << I_A1_ApertureCenterIndex << ", I_ApertureLength, I_A1_ApertureCenterInd) returned FALSE -> Returning FALSE" << endl;
          return false;
        }
        //        if ((I_ApertureLength >= 20)){//} <= maxLength_In) && (I_ApertureLength >= I_In_MinLength)){
        if (I_ApertureLength >= minLength_In){
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": I_A1_ApertureCenterInd = " << I_A1_ApertureCenterInd << endl;
          #endif
          if (!pfsDRPStella::math::GetSubArrCopy(D_A1_IndexCol,
                                                 I_A1_ApertureCenterInd, 
                                                  D_A1_ApertureCenterIndex)){
            cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: pfsDRPStella::math::GetSubArrCopy(*P_D_A1_IndexRow = " << D_A1_IndexCol << ", I_A1_ApertureCenterInd = " << I_A1_ApertureCenterInd << ", D_A1_ApertureCenterIndex) returned FALSE -> returning FALSE" << endl;
            return false;
          }
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_ApertureCenterIndex = " << D_A1_ApertureCenterIndex << endl;
          #endif
            
          if (!pfsDRPStella::math::GetSubArrCopy(D_A1_ApertureCenter,
                                                 I_A1_ApertureCenterInd,
                                                 D_A1_ApertureCenterPos)){
            cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: pfsDRPStella::math::GetSubArrCopy(D_A1_ApertureCenter = " << D_A1_ApertureCenter << ", I_A1_ApertureCenterInd = " << I_A1_ApertureCenterInd << ", D_A1_ApertureCenterIndex) returned FALSE -> returning FALSE" << endl;
            return false;
          }
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_ApertureCenterPos = " << D_A1_ApertureCenterPos << endl;
          #endif
              
          /// Fit Polynomial
          blitz::Array<std::string, 1> S_A1_KeyWords_PolyFit(1);
          S_A1_KeyWords_PolyFit(0) = "YFIT";
          void **PP_Args = (void**)malloc(sizeof(void*));
          blitz::Array<double, 1> D_A1_XCenters(ccdImage.rows());
          PP_Args[0] = &D_A1_XCenters;
          if (!pfsDRPStella::math::PolyFit(D_A1_ApertureCenterIndex, 
                                           D_A1_ApertureCenterPos, 
                                           fiberTraceFunctionFindingControl.fiberTraceFunctionControl.order,
                                           S_A1_KeyWords_PolyFit,
                                           PP_Args, 
                                           P_D_A1_PolyFitCoeffs)){
            cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": ERROR: PolyFit returned FALSE -> Returning FALSE" << endl;
            return false;
          }
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "CFits::FindAndTraceApertures: i_Row = " << i_Row << ": I_ApertureNumber = " << I_ApertureNumber << ": D_A1_XCenters = " << D_A1_XCenters << endl;
          #endif
          std::vector<float> xCenters(D_A1_XCenters.size());
          for (int iter = 0; iter < static_cast<int>(D_A1_XCenters.size()); iter++)
            xCenters[iter] = static_cast<float>(D_A1_XCenters(iter));
          
          PTR(pfsDRPStella::FiberTrace<ImageT>) fiberTrace(new pfsDRPStella::FiberTrace< ImageT >(_maskedImage));
              
          I_ApertureNumber++;
        
          fiberTraceFunction.xCenter = D_A1_ApertureCenterPos(int(D_A1_ApertureCenterIndex.size()/2.));
          fiberTraceFunction.yCenter = D_A1_ApertureCenterIndex(int(D_A1_ApertureCenterIndex.size()/2.));
          fiberTraceFunction.yHigh = D_A1_ApertureCenterIndex(int(D_A1_ApertureCenterIndex.size()-1)) - fiberTraceFunction.yCenter;
          fiberTraceFunction.yLow = D_A1_ApertureCenterIndex(0) - fiberTraceFunction.yCenter;
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "CFits::FindAndTraceApertures: P_D_A1_PolyFitCoeffs = " << *P_D_A1_PolyFitCoeffs << endl;
          #endif
          fiberTraceFunction.coefficients.resize(P_D_A1_PolyFitCoeffs->size());
          for (int iter=0; iter < static_cast<int>(P_D_A1_PolyFitCoeffs->size()); iter++)
            fiberTraceFunction.coefficients[iter] = (*P_D_A1_PolyFitCoeffs)(iter);
          #ifdef __DEBUG_FINDANDTRACE__
            cout << "CFits::FindAndTraceApertures: fiberTraceFunction.xCenter = " << fiberTraceFunction.xCenter << endl;
            cout << "CFits::FindAndTraceApertures: fiberTraceFunction.yCenter = " << fiberTraceFunction.yCenter << endl;
            cout << "CFits::FindAndTraceApertures: fiberTraceFunction.yLow = " << fiberTraceFunction.yLow << endl;
            cout << "CFits::FindAndTraceApertures: fiberTraceFunction.yHigh = " << fiberTraceFunction.yHigh << endl;
//            cout << "CFits::FindAndTraceApertures: fiberTraceFunction.coefficients = " << fiberTraceFunction.coefficients << endl;
          #endif
          if (!fiberTrace->setFiberTraceFunction(fiberTraceFunction)){
            cout << "FindAndTraceApertures: ERROR: setFiberTraceFunction returned FALSE" << endl;
            return false;
          }
          #ifdef __DEBUG_FINDANDTRACE__
            for (int iter=0; iter < static_cast<int>(xCenters.size()); ++iter){
              cout << "FindAndTraceApertures: xCenters[" << iter << "] = " << xCenters[iter] << endl;
            }
          #endif
          if (!fiberTrace->setXCenters(xCenters)){
            cout << "FindAndTraceApertures: ERROR: setXCenters returned FALSE" << endl;
            return false;
          }
          #ifdef __DEBUG_FINDANDTRACE__
            for (int iter=0; iter < static_cast<int>(xCenters.size()); ++iter){
              cout << "FindAndTraceApertures: xCenters[" << iter << "] = " << xCenters[iter] << " =? fiberTrace->getXCenters()[" << iter << "] = fiberTrace->getXCenters()[iter] = " << fiberTrace->getXCenters()[iter] << endl;
            }
          #endif
          if (!fiberTrace->createTrace()){
            cout << "FindAndTraceApertures: ERROR: createTrace returned FALSE" << endl;
            return false;
          }
          _fiberTraceSet.addFiberTrace(fiberTrace);
        }
        i_Row = I_Row_Bak - 1;
      }/// end if (B_ApertureFound)
    }/// end for(i_Row = 0; i_Row < _maskedImage.getHeight(); i_Row++)
    
    #ifdef __DEBUG_FINDANDTRACE__
      std::string S_NewFileName = "/home/azuri/spectra/pfs/FindAndTraceApertures";
      std::string S_FileNameTrace = S_NewFileName + "_out.fits";
      pfs::drp::stella::util::WriteFits(&ccdImage, S_FileNameTrace);
      blitz::Array<float, 2> F_A2_Trace(2,2);
      for (int i = 0; i < static_cast<int>(_fiberTraceSet.size()); i++){
        blitz::Array<ImageT, 2> TImage = utils::ndarrayToBlitz(_fiberTraceSet.getFiberTrace(i).getTrace().getImage()->getArray());
        pfs::drp::stella::math::Float(TImage, F_A2_Trace);
        S_FileNameTrace = S_NewFileName + "_trace_" + std::to_string(i) + ".fits";
        pfs::drp::stella::util::WriteFits(&F_A2_Trace, S_FileNameTrace);
      }
    #endif
    delete(P_D_A1_PolyFitCoeffs);
    return true;
  }
                                                                                                       
  namespace pfs{ namespace drp { namespace stella{ namespace math{
    
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
    
    blitz::Array<double, 1>* Poly(const blitz::Array<double, 1> &xVec,
                                  const blitz::Array<double, 1> &coeffsVec){
      int ii = 0;
      blitz::Array<double, 1>* P_D_A1_Out = new blitz::Array<double, 1>(xVec.size());
      #ifdef __DEBUG_POLY__
        cout << "Poly: xVec = " << xVec << endl;
        cout << "Poly: coeffsVec = " << coeffsVec << endl;
        cout << "Poly: *P_D_A1_Out set to " << *P_D_A1_Out << endl;
      #endif
      int I_PolynomialOrder = coeffsVec.size() - 1;
      #ifdef __DEBUG_POLY__
        cout << "Poly: I_PolynomialOrder set to " << I_PolynomialOrder << endl;
      #endif
      if (I_PolynomialOrder == 0){
        *P_D_A1_Out = coeffsVec(0);
        #ifdef __DEBUG_POLY__
          cout << "Poly: I_PolynomialOrder == 0: *P_D_A1_Out set to " << *P_D_A1_Out << endl;
        #endif
        return P_D_A1_Out;
      }
      *P_D_A1_Out = coeffsVec(I_PolynomialOrder);
      #ifdef __DEBUG_POLY__
        cout << "Poly: I_PolynomialOrder != 0: *P_D_A1_Out set to " << *P_D_A1_Out << endl;
      #endif
      for (ii = I_PolynomialOrder-1; ii >= 0; ii--){
        *P_D_A1_Out = (*P_D_A1_Out) * xVec + coeffsVec(ii);
        #ifdef __DEBUG_POLY__
          cout << "Poly: I_PolynomialOrder != 0: for (ii = " << ii << "; ii >= 0; ii--) *P_D_A1_Out set to " << *P_D_A1_Out << endl;
        #endif
      }
      return P_D_A1_Out;
    }

    double Poly(const double D_X_In,
                const blitz::Array<double, 1> &D_A1_Coeffs){
      blitz::Array<double, 1> D_A1_X(1);
      D_A1_X = D_X_In;
      blitz::Array<double, 1> *P_D_A1_Y = pfs::drp::stella::math::Poly(D_A1_X, D_A1_Coeffs);
      double D_Y = (*P_D_A1_Y)(0);
      delete(P_D_A1_Y);
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
      blitz::Array<int, 1> I_A1_Fix(F_A1_XCenters.size());
      I_A1_Fix = pfsDRPStella::math::Int(F_A1_XCenters);
      I_A2_MinCenMax_Out.resize(xCenters_In.size(), 3);
      I_A2_MinCenMax_Out = 0;
      
      I_A2_MinCenMax_Out(blitz::Range::all(), 1) = I_A1_Fix;
      
      #ifdef __DEBUG_MINCENMAX__
        cout << "CFits::calcMinCenMax: I_A2_MinCenMax_Out(*,1) = " << I_A2_MinCenMax_Out(blitz::Range::all(), 1) << endl;
      #endif
      blitz::Array<float, 1> F_A1_Temp(F_A1_XCenters.size());
      F_A1_Temp = F_A1_XCenters + xLow_In;
      
      I_A2_MinCenMax_Out(blitz::Range::all(), 0) = pfsDRPStella::math::Int(F_A1_Temp);// - I_NPixCut_Left;///(*P_I_A1_Temp); /// Left column of order
      
      #ifdef __DEBUG_MINCENMAX__
        cout << "CFits::calcMinCenMax: I_A2_MinCenMax_Out(*,0) = " << I_A2_MinCenMax_Out(blitz::Range::all(), 0) << endl;
      #endif
      F_A1_Temp = F_A1_XCenters + xHigh_In;
      
      I_A2_MinCenMax_Out(blitz::Range::all(), 2) = pfsDRPStella::math::Int(F_A1_Temp);
      
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
    }
    
    /**
     * Fix(double)
     * Returns integer value cut at decimal point. If D_In is negative the integer value greater or equal than D_In is returned.
     **/
    template <typename T> 
    int Fix(T D_In){
      return ((D_In < T(0.)) && (T(static_cast<int>(D_In)) < D_In)) ? static_cast<int>(D_In) + 1 : static_cast<int>(D_In);
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
        for (int n = 0; n < Arr.cols(); m++){
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
        for (int n = 0; n < Arr.cols(); m++){
          TempIntArr(m, n) = pfsDRPStella::math::Long(Arr(m, n));
        }
      }
      return TempIntArr;
    }
    
    template <typename T> 
    void Float(const blitz::Array<T, 1> &Arr, blitz::Array<float, 1>& Arr_Out){
      Arr_Out.resize(Arr.rows(), Arr.cols());
      for (int m = 0; m < static_cast<int>(Arr.size()); m++){
        Arr_Out(m) = float(Arr(m));
      }
      return;
    }
    
    template <typename T> 
    void Float(const blitz::Array<T, 2> &Arr, blitz::Array<float, 2>& Arr_Out){
      Arr_Out.resize(Arr.rows(), Arr.cols());
      for (int m = 0; m < Arr.rows(); m++){
        for (int n = 0; n < Arr.cols(); n++){
          Arr_Out(m, n) = float(Arr(m, n));
        }
      }
      return;
    }
    
    template <typename T> 
    void Double(const blitz::Array<T, 1> &Arr, blitz::Array<double, 1>& Arr_Out){
      Arr_Out.resize(Arr.rows(), Arr.cols());
      for (int m = 0; m < static_cast<int>(Arr.size()); m++){
        Arr_Out(m) = double(Arr(m));
      }
      return;
    }
    
    template <typename T> 
    blitz::Array<double, 1> Double(const blitz::Array<T, 1> &Arr){
      blitz::Array<double, 1> Arr_Out(Arr.rows(), Arr.cols());
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
        cout << "CFits::IntegralUnderCurve: WARNING: D_A1_XInt(1)(=" << D_A1_XInt(1) << ") > D_A1_XIn(0)(=" << D_A1_XIn(0) << ")" << endl;
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
      blitz::Array<double, 1> *P_D_A1_YFit;
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
      P_D_A1_YFit = pfsDRPStella::math::Poly(D_A1_X, *P_D_A1_Coeffs);

      /// Calculate Integral
      D_Integral_Out += (D_BinEnd_X - D_BinStart_X) * (*P_D_A1_YFit)(1);

      #ifdef __DEBUG_INTEGRAL__
        cout << "CFits::IntegralUnderLine: D_Integral_Out = " << D_Integral_Out << endl;
      #endif

      /// clean up
      delete(P_D_A1_YFit);
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
      I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "MEASURE_ERRORS");
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

      blitz::Array<int, 1> *P_I_A1_NotRejected;
      bool B_KeyWordSet_NotRejected = false;
      I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "NOT_REJECTED");
      if (I_Pos >= 0){
        cout << "CFits::PolyFit: Reading KeyWord NOT_REJECTED" << endl;
        cout << "CFits::PolyFit: I_Pos = " << I_Pos << endl;
        P_I_A1_NotRejected = (blitz::Array<int,1>*)(ArgV[I_Pos]);
        cout << "CFits::PolyFit: *P_I_A1_NotRejected = " << *P_I_A1_NotRejected << endl;
        B_KeyWordSet_NotRejected = true;
        cout << "CFits::PolyFit: KeyWord NOT_REJECTED read" << endl;
      }

      blitz::Array<int, 1> *P_I_A1_Rejected;
      blitz::Array<int, 1> I_A1_Rejected(D_A1_X_In.size());
      I_A1_Rejected = 0;
      bool B_KeyWordSet_Rejected = false;
      I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "REJECTED");
      if (I_Pos >= 0){
        cout << "CFits::PolyFit: Reading KeyWord REJECTED" << endl;
        cout << "CFits::PolyFit: I_Pos = " << I_Pos << endl;
        P_I_A1_Rejected = (blitz::Array<int,1>*)(ArgV[I_Pos]);
        cout << "CFits::PolyFit: *P_I_A1_Rejected = " << *P_I_A1_Rejected << endl;
        B_KeyWordSet_Rejected = true;
        cout << "CFits::PolyFit: KeyWord REJECTED read" << endl;
      }

      int *P_I_NRejected;
      bool B_KeyWordSet_NRejected = false;
      I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "N_REJECTED");
      if (I_Pos >= 0){
        cout << "CFits::PolyFit: Reading KeyWord N_REJECTED" << endl;
        cout << "CFits::PolyFit: I_Pos = " << I_Pos << endl;
        P_I_NRejected = (int*)(ArgV[I_Pos]);
        cout << "CFits::PolyFit: P_I_NRejected = " << *P_I_NRejected << endl;
        B_KeyWordSet_NRejected = true;
        cout << "CFits::PolyFit: KeyWord N_REJECTED read" << endl;
      }

      blitz::Array<int, 1> I_A1_OrigPos(D_A1_X_In.size());
      I_A1_OrigPos = pfsDRPStella::math::IndGenArr(D_A1_X_In.size());
      blitz::Array<double, 1> *P_D_A1_PolyRes;
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
        blitz::Array<double, 1> *P_D_A1_YFit = pfsDRPStella::math::Poly(D_A1_X, *out);
        double D_SDev = sqrt(blitz::sum(blitz::pow2(D_A1_Y - (*P_D_A1_YFit)) / D_A1_Y.size()));

        P_D_A1_PolyRes = pfsDRPStella::math::Poly(D_A1_X_In, *out);
        for (unsigned int i_pos=0; i_pos < D_A1_Y_In.size(); i_pos++){
          double D_Dev = D_A1_Y_In(i_pos) - (*P_D_A1_PolyRes)(i_pos);
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
        delete(P_D_A1_PolyRes);
        D_A1_X.resize(I_DataValues_New);
        D_A1_Y.resize(I_DataValues_New);
        D_A1_X = D_A1_X_New(blitz::Range(0,I_DataValues_New-1));
        D_A1_Y = D_A1_Y_New(blitz::Range(0,I_DataValues_New-1));
        if (B_HaveMeasureErrors){
          D_A1_MeasureErrors.resize(I_DataValues_New);
          D_A1_MeasureErrors = D_A1_MeasureErrors_New(blitz::Range(0,I_DataValues_New-1));
        }
    
        delete(P_D_A1_YFit);

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
      if (B_KeyWordSet_NRejected){
        *P_I_NRejected = I_NRejected;
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
      void **PP_Args = (void**)malloc(sizeof(void*) * 1);
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
      if ((I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
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
      if ((I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
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
      if ((I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
      {
        delete(P_D_A1_Sigma);
        P_D_A1_Sigma = (blitz::Array<double,1>*)ArgV[I_Pos];
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: KeyWord_Set(SIGMA): *P_D_A1_Sigma set to " << (*P_D_A1_Sigma) << endl;
        #endif
      }

      blitz::Array<double, 2> *P_D_A2_Covar = new blitz::Array<double,2>(I_M,I_M);
      sTemp = "COVAR";
      if ((I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
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
      if ((I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, sTemp)) < 0)
      {
        delete(P_D_A1_YFit);
      }
      sTemp = "SIGMA";
      if ((I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, sTemp)) < 0)
      {
        delete(P_D_A1_Sigma);
      }
      sTemp = "COVAR";
      if ((I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, sTemp)) < 0)
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
      
      if (pfsDRPStella::util::KeyWord_Set(S_A1_In, "LSQUADRATIC") < 0
        && pfsDRPStella::util::KeyWord_Set(S_A1_In, "QUADRATIC") < 0
        && pfsDRPStella::util::KeyWord_Set(S_A1_In, "SPLINE") < 0)
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
    }
    
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
      if (pfsDRPStella::util::KeyWord_Set(keyWords_In, "LSQUADRATIC") >= 0)
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
      else if (pfsDRPStella::util::KeyWord_Set(keyWords_In, "QUADRATIC") >= 0)
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
      else if (pfsDRPStella::util::KeyWord_Set(keyWords_In, "SPLINE") >= 0){
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
      while(valueVec_In(ii) == valueVec_In(ii+1)){
        ii++;
      }
      if (valueVec_In(ii+1) > valueVec_In(ii))
        Increasing = true;
      
      #ifdef __DEBUG_INTERPOL__
        if (Increasing)
          cout << "CFits::ValueLocate: Increasing = TRUE" << endl;
        else
          cout << "CFits::ValueLocate: Increasing = FALSE" << endl;
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
              if (vec_In(n+1) <= valueVec_In(m) && valueVec_In(m) < vec_In(n)){
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
      cout << "CFits::FirstIndexWithValueGEFrom: not found => Returning -1" << endl;
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
      I_KeywordSet_MeasureErrors = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "MEASURE_ERRORS_IN");
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
      I_KeywordSet_ChiSq = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "CHISQ_OUT");
      if (I_KeywordSet_ChiSq >= 0)
      {
        P_D_A1_ChiSq = (blitz::Array<double,1>*)ArgV_In[I_KeywordSet_ChiSq];
        P_D_A1_ChiSq->resize(D_A2_CCD_In.rows());
        S_A1_Args_Fit(I_ArgPos) = "CHISQ_OUT";
        I_ArgPos++;
      }

      blitz::Array<double, 1> *P_D_A1_Q;
      I_KeywordSet_Q = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "Q_OUT");
      if (I_KeywordSet_Q >= 0)
      {
        P_D_A1_Q = (blitz::Array<double,1>*)ArgV_In[I_KeywordSet_Q];
        P_D_A1_Q->resize(D_A2_CCD_In.rows());
        S_A1_Args_Fit(I_ArgPos) = "Q_OUT";
        I_ArgPos++;
      }

      blitz::Array<double, 2> *P_D_A2_Sigma_Out;
      I_KeywordSet_Sigma = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "SIGMA_OUT");
      if (I_KeywordSet_Sigma >= 0)
      {
        P_D_A2_Sigma_Out = (blitz::Array<double,2>*)ArgV_In[I_KeywordSet_Sigma];
        P_D_A2_Sigma_Out->resize(D_A2_CCD_In.rows(), 2);
        S_A1_Args_Fit(I_ArgPos) = "SIGMA_OUT";
        I_ArgPos++;
      }

      I_KeywordSet_YFit = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "YFIT_OUT");
      if (I_KeywordSet_YFit >= 0)
      {
        P_D_A2_YFit = (blitz::Array<double,2>*)ArgV_In[I_KeywordSet_YFit];
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington2D: P_D_A2_YFit = " << *P_D_A2_YFit << endl;
        #endif
        S_A1_Args_Fit(I_ArgPos) = "YFIT_OUT";
        I_ArgPos++;
      }
    
      I_KeywordSet_Reject = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "REJECT_IN");
      if (I_KeywordSet_Reject >= 0)
      {
        P_D_Reject = (double*)ArgV_In[I_KeywordSet_Reject];
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington2D: P_D_Reject = " << *P_D_Reject << endl;
        #endif
        S_A1_Args_Fit(I_ArgPos) = "REJECT_IN";
        I_ArgPos++;
      }
  
      I_KewordSet_Mask = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "MASK_INOUT");
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
      I_KeywordSet_AllowSkyLTZero = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "ALLOW_SKY_LT_ZERO");
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
      I_KeywordSet_AllowSpecLTZero = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "ALLOW_SPEC_LT_ZERO");
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
      I_KeywordSet_Reject = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "REJECT_IN");
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

      I_KeywordSet_Mask = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "MASK_INOUT");
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
      I_KeywordSet_SigmaOut = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "SIGMA_OUT");
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
      I_KeywordSet_ChiSqOut = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "CHISQ_OUT");
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
      I_KeywordSet_QOut = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "Q_OUT");
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
      I_KeywordSet_MeasureErrors = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "MEASURE_ERRORS_IN");
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
      I_KeywordSet_YFitOut = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "YFIT_OUT");
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
      int median;
      string Mode = "NORMAL";
      double *P_D_ErrOut = new double(0.);

      int I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "MODE");
      if (I_Pos >= 0)
        Mode = *(string*)PP_Args_In[I_Pos];

//      I_Pos = pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "ERR_OUT");
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

//      if (pfsDRPStella::util::KeyWord_Set(S_A1_Args_In, "ERRORS_IN") >= 0){
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
    
    /**
     * function GetRowFromIndex(int I_Index_In, int I_NRows_In) const
     * task: Returns Row specified by I_Index_In from the formula
     *       Col = (int)(I_Index_In / I_NRows_In)
     *       Row = I_IAperture_In - Col * I_NRows_In
     **/
    int GetRowFromIndex(int I_Index_In, int I_NRows_In)
    {
      return (I_Index_In - (I_NRows_In * pfsDRPStella::math::GetColFromIndex(I_Index_In, I_NRows_In)));
    }
    
    /**
     * function GetColFromIndex(int I_Index_In, int I_NRows_In) const
     * task: Returns Col specified by I_Index_In from the formula
     *       Col = (int)(I_Index_In / I_NRows_In)
     *       Row = I_IAperture_In - Col * I_NRows_In
     **/
    int GetColFromIndex(int I_Index_In, int I_NRows_In)
    {
      return ((int)(I_Index_In / I_NRows_In));
    }
    
    /**
     *  BandSol() const
     **/
    int BandSol(int Argc, void *Argv[])
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
      #ifdef __DEBUG_FITS_BANDSOL__
      cout << "CFits::BandSol: *a = " << *a << endl;
      cout << "CFits::BandSol: *r = " << *r << endl;
      cout << "CFits::BandSol: n = " << n << endl;
      cout << "CFits::BandSol: nd = " << nd << endl;
      (*P_OFS_Log) << "CFits::BandSol: *a = " << *a << endl;
      (*P_OFS_Log) << "CFits::BandSol: *r = " << *r << endl;
      (*P_OFS_Log) << "CFits::BandSol: n = " << n << endl;
      (*P_OFS_Log) << "CFits::BandSol: nd = " << nd << endl;
      #endif
      
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
        #ifdef __DEBUG_FITS_BANDSOL__
        (*P_OFS_Log) << "bandsol: for(o(=" << o << ")=0; o<n(=" << n << "); o++): o+n*(nd/2) = " << o+n*(nd/2) << endl;
        (*P_OFS_Log) << "bandsol: for(o(=" << o << ")=0; o<n(=" << n << "); o++): aa set to " << aa << endl;
        #endif
        r[o] /= aa;
        #ifdef __DEBUG_FITS_BANDSOL__
        (*P_OFS_Log) << "bandsol: for(o(=" << o << ")=0; o<n(=" << n << "); o++): r[o] set to " << r[o] << endl;
        #endif
        for(p = 0; p < nd; p++){
          a[o + p * n] /= aa;
          #ifdef __DEBUG_FITS_BANDSOL__
          (*P_OFS_Log) << "bandsol: for(o(=" << o << ")=0; o<n(=" << n << "); o++): for(p(=" << p << ")=0; p<nd(=" << nd << "); p++): a[o+p*n=" << o+p*n << "] set to " << a[o+p*n] << endl;
          #endif
        }
        for(p = 1; p < MIN(nd / 2 + 1, n - o); p++)
        {
          aa=a[o + p + n * (nd / 2 - p)];
          #ifdef __DEBUG_FITS_BANDSOL__
          (*P_OFS_Log) << "bandsol: for(o(=" << o << ")=0; o<n(=" << n << "); o++): for(p(=" << p << ")=0; p<nd(=" << nd << "); p++): aa set to " << aa << endl;
          #endif
          r[o + p] -= r[o] * aa;
          #ifdef __DEBUG_FITS_BANDSOL__
          (*P_OFS_Log) << "bandsol: for(o(=" << o << ")=0; o<n(=" << n << "); o++): for(p(=" << p << ")=0; p<nd(=" << nd << "); p++): r[o+p=" << o+p << "] set to " << r[o+p] << endl;
          #endif
          for(q = 0; q < n * (nd - p); q += n){
            a[o + p + q] -= a[o + q + n * p] * aa;
            #ifdef __DEBUG_FITS_BANDSOL__
            (*P_OFS_Log) << "bandsol: for(o(=" << o << ")=0; o<n(=" << n << "); o++): for(p(=" << p << ")=0; p<nd(=" << nd << "); p++): for(q(=" << q << ")=0; q<n*(nd-p)(=" << n*(nd-p) << "); q++): a[o+p+q=" << o+p+q << "] set to " << a[o+p+q] << endl;
            #endif
          }
        }
      }
      
      /* Backward sweep */
      r[n-1] /= a[n - 1 + n * (nd / 2)];
      #ifdef __DEBUG_FITS_BANDSOL__
      (*P_OFS_Log) << "bandsol: r[n-1=" << n-1 << "] set to " << r[n-1] << endl;
      #endif
      for(o=n-1; o>0; o--)
      {
        for(p=1; p <= MIN(nd/2,o); p++){
          r[o-p] -= r[o] *
          a[o - p + n * (nd / 2 + p)];
          #ifdef __DEBUG_FITS_BANDSOL__
          (*P_OFS_Log) << "bandsol: for(o(=" << o << ")=n-1=" << n-1 << "; o>0; o--): for(p(=" << p << ")=1; p<=Min(nd/2=" << nd/2 << ",o=" << o << "); p++): r[o-p=" << o-p << "] set to " << r[o-p] << endl;
          #endif
        }
        r[o-1] /= a[o-1+n*(nd/2)];
        #ifdef __DEBUG_FITS_BANDSOL__
        (*P_OFS_Log) << "bandsol: for(o(=" << o << ")=n-1=" << n-1 << "; o>0; o--): r[o-1=" << o-1 << "] set to " << r[o-1] << endl;
        #endif
      }
      r[0] /= a[n*(nd/2)];
      #ifdef __DEBUG_FITS_BANDSOL__
      (*P_OFS_Log) << "bandsol: r[0] set to " << r[0] << endl;
      #endif
      
      return 0;
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
    
    bool Uniq(const blitz::Array<int, 1> &IA1_In, blitz::Array<int, 1> &IA1_Result)
    {
      int I_Size = IA1_In.size();
      int I_Count;
      blitz::Array<int, 1> IA1_Res(IA1_In.size()-1);
      if (I_Size == 0)
      {
        cout << "CFits::Uniq: ERROR: Size of input array == 0" << endl;
        return false;
      }
      IA1_Res = blitz::where(IA1_In(blitz::Range(0, IA1_In.size()-2)) != IA1_In(blitz::Range(1, IA1_In.size()-1)), 1, 0);
      I_Count = blitz::sum(IA1_Res);
      if (I_Count <= 0)
      {
        IA1_Result.resize(1);
        IA1_Result(0) = IA1_In.size() - 1;
      }
      else
      {
        IA1_Result.resize(I_Count);
        int pos = 0;
        for (unsigned int m = 0; m < IA1_Res.size(); m++)
        {
          if (IA1_Res(m) == 1)
          {
            IA1_Result(pos) = m;
            pos++;
          }
        }
        if (IA1_In(IA1_In.size() - 2) != IA1_In(IA1_In.size() - 1))
        {
          int oldsize = IA1_Result.size();
          IA1_Result.resizeAndPreserve(I_Count + 1);
          if (oldsize < static_cast<int>(IA1_Result.size()))
            IA1_Result(blitz::Range(oldsize, IA1_Result.size() - 1)) = 0;
          IA1_Result(I_Count) = IA1_In.size() - 1;
        }
      }
      IA1_Res.resize(0);
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
  
  }/// end namespace math
  
  namespace util{
    
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
                fprintf(p_file, "%d\n", Array_In(m));
              else if (typeid(T) == typeid(unsigned short))
                fprintf(p_file, "%d\n", Array_In(m));
              else if (typeid(T) == typeid(int))
                fprintf(p_file, "%d\n", Array_In(m));
              else if (typeid(T) == typeid(unsigned int))
                fprintf(p_file, "%d\n", Array_In(m));
              else if (typeid(T) == typeid(long))
                fprintf(p_file, "%d\n", Array_In(m));
              else if (typeid(T) == typeid(unsigned long))
                fprintf(p_file, "%d\n", Array_In(m));
              else if (typeid(T) == typeid(float))
                fprintf(p_file, "%.17f\n", double(Array_In(m)));
              else
                fprintf(p_file, "%.17f\n", double(Array_In(m)));
            }
            else{
              fprintf(p_file, "%.8e\n", double(Array_In(m)));
            }
          }
          else{/// N == 2 
            for (n = 0; n < Array_In.cols(); n++){
              if (B_A1_Exp(n))
                fprintf(p_file, " %.8e", double(Array_In(m,n)));
              else{
                if (typeid(T) == typeid(short))
                  fprintf(p_file, " %d", Array_In(m,n));
                else if (typeid(T) == typeid(unsigned short))
                  fprintf(p_file, " %d", Array_In(m,n));
                else if (typeid(T) == typeid(int))
                  fprintf(p_file, " %d", Array_In(m,n));
                else if (typeid(T) == typeid(unsigned int))
                  fprintf(p_file, " %d", Array_In(m,n));
                else if (typeid(T) == typeid(long))
                  fprintf(p_file, " %d", Array_In(m,n));
                else if (typeid(T) == typeid(unsigned long))
                  fprintf(p_file, " %d", Array_In(m,n));
                else if (typeid(T) == typeid(float))
                  fprintf(p_file, " %.10f", double(Array_In(m,n)));
                else
                  fprintf(p_file, " %.10f", double(Array_In(m,n)));
              }
            }
            fprintf(p_file, "\n");
          }
        }
      }
      fclose(p_file);
      return true;
    }
  }
  
  template class FiberTrace<unsigned short>;
  template class FiberTrace<int>;
//  template class FiberTrace<long>;
  template class FiberTrace<float>;
  template class FiberTrace<double>;
  
  template class FiberTraceSet<unsigned short>;
  template class FiberTraceSet<int>;
//  template class FiberTraceSet<long>;
  template class FiberTraceSet<float>;
  template class FiberTraceSet<double>;
  
  template class MaskedSpectrographImage<unsigned short>;
  template class MaskedSpectrographImage<int>;
//  template class MaskedSpectrographImage<long>;
  template class MaskedSpectrographImage<float>;
  template class MaskedSpectrographImage<double>;
  
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
  
  template void math::Float(const blitz::Array<unsigned short, 2> &, blitz::Array<float, 2>&);
  template void math::Float(const blitz::Array<int, 2> &, blitz::Array<float, 2>&);
  template void math::Float(const blitz::Array<long, 2> &, blitz::Array<float, 2>&);
  template void math::Float(const blitz::Array<float, 2> &, blitz::Array<float, 2>&);
  template void math::Float(const blitz::Array<double, 2> &, blitz::Array<float, 2>&);
  
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
  
  template bool util::WriteFits(const blitz::Array<unsigned short, 2>* image_In, const string &fileName_In);
  template bool util::WriteFits(const blitz::Array<int, 2>* image_In, const string &fileName_In);
  template bool util::WriteFits(const blitz::Array<long, 2>* image_In, const string &fileName_In);
  template bool util::WriteFits(const blitz::Array<float, 2>* image_In, const string &fileName_In);
  template bool util::WriteFits(const blitz::Array<double, 2>* image_In, const string &fileName_In);
  
  template bool util::WriteFits(const blitz::Array<unsigned short, 1>* image_In, const string &fileName_In);
  template bool util::WriteFits(const blitz::Array<int, 1>* image_In, const string &fileName_In);
  template bool util::WriteFits(const blitz::Array<long, 1>* image_In, const string &fileName_In);
  template bool util::WriteFits(const blitz::Array<float, 1>* image_In, const string &fileName_In);
  template bool util::WriteFits(const blitz::Array<double, 1>* image_In, const string &fileName_In);
  
  template bool util::WriteArrayToFile(const blitz::Array<unsigned short, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  template bool util::WriteArrayToFile(const blitz::Array<int, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  template bool util::WriteArrayToFile(const blitz::Array<long, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  template bool util::WriteArrayToFile(const blitz::Array<float, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  template bool util::WriteArrayToFile(const blitz::Array<double, 1> &I_A1_In, const string &S_FileName_In, const string &S_Mode);
  
  template bool util::WriteArrayToFile(const blitz::Array<unsigned short, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  template bool util::WriteArrayToFile(const blitz::Array<int, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  template bool util::WriteArrayToFile(const blitz::Array<long, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  template bool util::WriteArrayToFile(const blitz::Array<float, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  template bool util::WriteArrayToFile(const blitz::Array<double, 2> &D_A2_In, const string &S_FileName_In, const string &S_Mode);
  
  }}}
