#include "pfs/drp/stella/Spectra.h"

namespace pfsDRPStella = pfs::drp::stella;

/** @brief Construct a Spectrum with empty vectors of specified size (default 0)
 */
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::Spectrum(size_t length,
                                                                           size_t iTrace) 
  : _length(length),
    _iTrace(iTrace),
    _isWavelengthSet(false)
{
  _spectrum = ndarray::allocate(length);
  _mask = ndarray::allocate(length);
  _variance = ndarray::allocate(length);
  _wavelength = ndarray::allocate(length);
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::Spectrum(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> const& spectrum,
                                                                           size_t iTrace) 
  : _length(spectrum.getLength()),
    _iTrace(spectrum.getITrace()),
    _isWavelengthSet(spectrum.isWavelengthSet())
{
  _spectrum = ndarray::allocate(spectrum.getSpectrum().getShape()[0]);
  _mask = ndarray::allocate(spectrum.getMask().getShape()[0]);
  _variance = ndarray::allocate(spectrum.getVariance().getShape()[0]);
  _wavelength = ndarray::allocate(spectrum.getWavelength().getShape()[0]);
  _spectrum.deep() = spectrum.getSpectrum();
  _mask.deep() = spectrum.getMask();
  _variance.deep() = spectrum.getVariance();
  _wavelength.deep() = spectrum.getWavelength();
  if (iTrace != 0)
    _iTrace = iTrace;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setSpectrum(const ndarray::Array<SpectrumT, 1, 1> & spectrum)
{
  /// Check length of input spectrum
  if (spectrum.getShape()[0] != _length){
    string message("pfsDRPStella::Spectrum::setSpectrum: ERROR: spectrum->size()=");
    message += to_string(spectrum.getShape()[0]) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _spectrum.deep() = spectrum;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setVariance(const ndarray::Array<VarianceT, 1, 1> & variance)
{
  /// Check length of input variance
  if (variance.getShape()[0] != _length){
    string message("pfsDRPStella::Spectrum::setVariance: ERROR: variance->size()=");
    message += to_string(variance.getShape()[0]) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _variance.deep() = variance;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setWavelength(const ndarray::Array<WavelengthT, 1, 1> & wavelength)
{
  /// Check length of input wavelength
  if (wavelength.getShape()[0] != _length){
    string message("pfsDRPStella::Spectrum::setWavelength: ERROR: wavelength->size()=");
    message += to_string(wavelength.getShape()[0]) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _wavelength.deep() = wavelength;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setMask(const ndarray::Array<MaskT, 1, 1> & mask)
{
  /// Check length of input mask
  if (mask.getShape()[0] != _length){
    string message("pfsDRPStella::Spectrum::setMask: ERROR: mask->size()=");
    message += to_string(mask.getShape()[0]) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _mask.deep() = mask;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setLength(const size_t length){
  pfsDRPStella::math::resize(_spectrum, length);
  pfsDRPStella::math::resize(_mask, length);
  pfsDRPStella::math::resize(_variance, length);
  pfsDRPStella::math::resize(_wavelength, length);
  if (length > _length){
    WavelengthT val = _wavelength[_length = 1];
    for (auto it = _wavelength.begin() + length; it != _wavelength.end(); ++it)
      *it = val;
  }
  _length = length;
  return true;
}

/**
 * Identify
 * Identifies calibration lines, given in D_A2_LineList_In the format [wlen, approx_pixel] in
 * wavelength-calibration spectrum D_A2_Spec_In [pixel_number, flux]
 * within the given position plus/minus I_Radius_In,
 * fits Gaussians to each line, fits Polynomial of order I_PolyFitOrder_In, and
 * returns calibrated spectrum D_A2_CalibratedSpec_Out in the format
 * [WLen, flux] and PolyFit coefficients D_A1_PolyFitCoeffs_Out
 * 
 * If D_A2_LineList_In contains 3 columns, the 3rd column will be used to decide which line
 * to keep in case a weak line close to a strong line gets wrongly identified as the strong
 * line
 **
bool CFits::Identify(const Array<double, 1> &D_A1_Spec_In,
                     const Array<double, 2> &D_A2_LineList_In,
                     const int I_Radius_In,
                     const double D_FWHM_In,
                     const int I_PolyFitOrder_In,
                     const CString &CS_FName_In,
                     Array<double, 2> &D_A2_CalibratedSpec_Out,
                     Array<double, 1> &D_A1_PolyFitCoeffs_Out,
                     double &D_RMS_Out,
                     Array<double, 2> &D_A2_PixWLen_Out) const{
*/
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::identify( ndarray::Array< double, 2, 1 > const& lineList,
                                                                                 DispCorControl const& dispCorControl ){
  DispCorControl tempDispCorControl( dispCorControl );
  _dispCorControl.reset();
  _dispCorControl = tempDispCorControl.getPointer();

  ///for each line in line list, find maximum in spectrum and fit Gaussian
  int I_MaxPos = 0;
  int I_Start = 0;
  int I_End = 0;
  int I_NTerms = 4;
  std::vector< int > V_Index( 2 * dispCorControl.searchRadius + 1, 0 );
  std::vector< double > V_GaussSpec( 1, 0. );
  ndarray::Array< double, 1, 1 > D_A1_GaussCoeffs = ndarray::allocate( I_NTerms );
  D_A1_GaussCoeffs.deep() = 0.;
  ndarray::Array< double, 1, 1 > D_A1_EGaussCoeffs = ndarray::allocate( I_NTerms );
  D_A1_EGaussCoeffs.deep() = 0.;
  ndarray::Array< int, 2, 1 > I_A2_Limited = ndarray::allocate( I_NTerms, 2 );
  I_A2_Limited.deep() = 1;
  ndarray::Array< double, 2, 1 > D_A2_Limits = ndarray::allocate( I_NTerms, 2 );
  D_A2_Limits.deep() = 0.;
  ndarray::Array< double, 1, 1 > D_A1_Guess = ndarray::allocate( I_NTerms );
  std::vector< double > V_MeasureErrors( 2, 0.);
  ndarray::Array< double, 1, 1 > D_A1_Ind = math::indGenNdArr( double( _spectrum.getShape()[ 0 ] ) );
  std::vector< double > V_X( 1, 0. );
  ndarray::Array< double, 1, 1 > D_A1_GaussPos = ndarray::allocate( lineList.getShape()[0] );
  D_A1_GaussPos.deep() = 0.;
  #ifdef __WITH_PLOTS__
    CString CS_PlotName("");
    CString *P_CS_Num;
  #endif
  for ( int i_line = 0; i_line < lineList.getShape()[ 0 ]; ++i_line ){
    I_Start = int( lineList[ ndarray::makeVector( i_line, 1 ) ] ) - dispCorControl.searchRadius;
    if ( I_Start < 0 )
      I_Start = 0;
    #ifdef __DEBUG_IDENTIFY__
      cout << "identify: i_line = " << i_line << ": I_Start = " << I_Start << endl;
    #endif
    I_End = int( lineList[ ndarray::makeVector( i_line, 1 ) ] ) + dispCorControl.searchRadius;
    if ( I_End >= _spectrum.getShape()[ 0 ] )
      I_End = _spectrum.getShape()[ 0 ] - 1;
    #ifdef __DEBUG_IDENTIFY__
      cout << "identify: i_line = " << i_line << ": I_End = " << I_End << endl;
    #endif
    if ( I_Start >= I_End ){
      cout << "identify: Warning: I_Start(=" << I_Start << ") >= I_End(=" << I_End << ")" << endl;// => Returning FALSE" << endl;
      cout << "identify: _spectrum = " << _spectrum << endl;
      cout << "identify: lineList = " << lineList << endl;
    }
    else{
      auto itMaxElement = std::max_element( _spectrum.begin() + I_Start, _spectrum.begin() + I_End + 1 );
      I_MaxPos = std::distance(_spectrum.begin(), itMaxElement);
//        #ifdef __DEBUG_IDENTIFY__
//          cout << "identify: i_line = " << i_line << ": indexPos = " << indexPos << endl;
//        #endif
//        I_MaxPos = indexPos;// + I_Start;
      #ifdef __DEBUG_IDENTIFY__
        cout << "identify: I_MaxPos = " << I_MaxPos << endl;
      #endif
      I_Start = std::round( double( I_MaxPos ) - ( 1.5 * dispCorControl.fwhm ) );
      if (I_Start < 0)
        I_Start = 0;
      #ifdef __DEBUG_IDENTIFY__
        cout << "identify: I_Start = " << I_Start << endl;
      #endif
      I_End = std::round( double( I_MaxPos ) + ( 1.5 * dispCorControl.fwhm ) );
      if ( I_End >= _spectrum.getShape()[ 0 ] )
        I_End = _spectrum.getShape()[ 0 ] - 1;
      #ifdef __DEBUG_IDENTIFY__
        cout << "identify: I_End = " << I_End << endl;
      #endif
      if ( I_End < I_Start + 4 ){
        cout << "identify: WARNING: Line position outside spectrum" << endl;
      }
      else{
        V_GaussSpec.resize( I_End - I_Start + 1 );
        V_MeasureErrors.resize( I_End - I_Start + 1 );
        V_X.resize( I_End - I_Start + 1 );
        auto itSpec = _spectrum.begin() + I_Start;
        for ( auto itGaussSpec = V_GaussSpec.begin(); itGaussSpec != V_GaussSpec.end(); ++itGaussSpec, ++itSpec )
          *itGaussSpec = *itSpec;
        #ifdef __DEBUG_IDENTIFY__
          cout << "identify: V_GaussSpec = ";
          for ( int iPos = 0; iPos < V_GaussSpec.size(); ++iPos )
            cout << V_GaussSpec[iPos] << " ";
          cout << endl;
        #endif
        for( auto itMeasErr = V_MeasureErrors.begin(), itGaussSpec = V_GaussSpec.begin(); itMeasErr != V_MeasureErrors.end(); ++itMeasErr, ++itGaussSpec ){
          *itMeasErr = sqrt( std::fabs( *itGaussSpec ) );
          if (*itMeasErr < 0.00001)
            *itMeasErr = 1.;
        }
        #ifdef __DEBUG_IDENTIFY__
          cout << "identify: V_MeasureErrors = ";
          for (int iPos = 0; iPos < V_MeasureErrors.size(); ++iPos )
            cout << V_MeasureErrors[iPos] << " ";
          cout << endl;
        #endif
        auto itInd = D_A1_Ind.begin() + I_Start;
        for ( auto itX = V_X.begin(); itX != V_X.end(); ++itX, ++itInd )
          *itX = *itInd;
        #ifdef __DEBUG_IDENTIFY__
          cout << "identify: V_X = ";
          for (int iPos = 0; iPos < V_X.size(); ++iPos )
            cout << V_X[iPos] << " ";
          cout << endl;
        #endif
//        if (!this->GaussFit(D_A1_X,
//                            D_A1_GaussSpec,
//                            D_A1_GaussCoeffs,
//                            CS_A1_KeyWords,
//                            PP_Args)){

      /*     p[0] = constant offset
       *     p[1] = peak y value
       *     p[2] = x centroid position
       *     p[3] = gaussian sigma width
       */
//          ndarray::Array< double, 2, 1 > toFit = ndarray::allocate( D_A1_X.getShape()[ 0 ], 2 );
//          toFit[ ndarray::view()(0) ] = D_A1_X;
//          toFit[ ndarray::view()(1) ] = D_A1_GaussSpec;
//            ndarray::Array< double, 1, 1 > gaussFitResult = gaussFit()
        D_A1_Guess[ 3 ] = *min_element( V_GaussSpec.begin(), V_GaussSpec.end() );
        D_A1_Guess[ 0 ] = *max_element( V_GaussSpec.begin(), V_GaussSpec.end() ) - D_A1_Guess(3);
        D_A1_Guess[ 1 ] = V_X[ 0 ] + ( V_X[ V_X.size() - 1 ] - V_X[ 0 ] ) / 2.;
        D_A1_Guess[ 2 ] = dispCorControl.fwhm;
        #ifdef __DEBUG_IDENTIFY__
          cout << "identify: D_A1_Guess = " << D_A1_Guess << endl;
        #endif
        D_A2_Limits[ ndarray::makeVector( 0, 0 ) ] = 0.;
        D_A2_Limits[ ndarray::makeVector( 0, 1 ) ] = std::fabs( 1.5 * D_A1_Guess[ 0 ] );
        D_A2_Limits[ ndarray::makeVector( 1, 0 ) ] = V_X[ 1 ];
        D_A2_Limits[ ndarray::makeVector( 1, 1 ) ] = V_X[ V_X.size() - 2 ];
        D_A2_Limits[ ndarray::makeVector( 2, 0 ) ] = D_A1_Guess[ 2 ] / 3.;
        D_A2_Limits[ ndarray::makeVector( 2, 1 ) ] = 2. * D_A1_Guess[ 2 ];
        D_A2_Limits[ ndarray::makeVector( 3, 1 ) ] = std::fabs( 1.5 * D_A1_Guess[ 3 ] ) + 1;
        #ifdef __DEBUG_IDENTIFY__
          cout << "identify: D_A2_Limits = " << D_A2_Limits << endl;
        #endif
        ndarray::Array< double, 1, 1 > D_A1_X = ndarray::external( V_X.data(), ndarray::makeVector( int( V_X.size() ) ), ndarray::makeVector( 1 ) );
        ndarray::Array< double, 1, 1 > D_A1_GaussSpec = ndarray::external( V_GaussSpec.data(), ndarray::makeVector( int( V_GaussSpec.size() ) ), ndarray::makeVector( 1 ) );
        ndarray::Array< double, 1, 1 > D_A1_MeasureErrors = ndarray::external( V_MeasureErrors.data(), ndarray::makeVector( int( V_MeasureErrors.size() ) ), ndarray::makeVector( 1 ) );
        if (!MPFitGaussLim(D_A1_X,
                           D_A1_GaussSpec,
                           D_A1_MeasureErrors,
                           D_A1_Guess,
                           I_A2_Limited,
                           D_A2_Limits,
                           true,
                           false,
                           D_A1_GaussCoeffs,
                           D_A1_EGaussCoeffs,
                           true)){
          cout << "identify: WARNING: GaussFit returned FALSE" << endl;
        //        return false;
        }
        else{
          #ifdef __DEBUG_IDENTIFY__
            cout << "identify: i_line = " << i_line << ": D_A1_GaussCoeffs = " << D_A1_GaussCoeffs << endl;
          #endif
          if ( std::fabs( double( I_MaxPos ) - D_A1_GaussCoeffs[ 1 ] ) < 2.5 ){//D_FWHM_In){
            D_A1_GaussPos[ i_line ] = D_A1_GaussCoeffs[ 1 ];
            #ifdef __DEBUG_IDENTIFY__
              cout << "identify: D_A1_GaussPos[" << i_line << "] = " << D_A1_GaussPos[ i_line ] << endl;
            #endif
            if ( i_line > 0 ){
              if ( std::fabs( D_A1_GaussPos[ i_line ] - D_A1_GaussPos[ i_line - 1 ] ) < 1.5 ){/// wrong line identified!
                if ( lineList.getShape()[ 1 ] > 2 ){
                  if ( lineList[ ndarray::makeVector( i_line, 2 ) ] < lineList[ ndarray::makeVector( i_line - 1, 2 ) ] ){
                    cout << "identify: WARNING: i_line=" << i_line << ": line " << i_line << " at " << D_A1_GaussPos[ i_line ] << " has probably been misidentified (D_A1_GaussPos(" << i_line-1 << ")=" << D_A1_GaussPos[ i_line - 1 ] << ") => removing line from line list" << endl;
                    D_A1_GaussPos[ i_line ] = 0.;
                  }
                  else{
                    cout << "identify: WARNING: i_line=" << i_line << ": line at D_A1_GaussPos[" << i_line-1 << "] = " << D_A1_GaussPos[ i_line - 1 ] << " has probably been misidentified (D_A1_GaussPos(" << i_line << ")=" << D_A1_GaussPos[ i_line ] << ") => removing line from line list" << endl;
                    D_A1_GaussPos[ i_line - 1 ] = 0.;
                  }
//                  exit(EXIT_FAILURE);
                }
              }
            }
          }
          else{
            cout << "identify: WARNING: I_MaxPos=" << I_MaxPos << " - D_A1_GaussCoeffs[ 1 ]=" << D_A1_GaussCoeffs[ 1 ] << " >= 2.5 => Skipping line" << endl;
          }
        }
      }
    }
  }/// end for (int i_line=0; i_line < D_A2_LineList_In.rows(); i_line++){
  ///remove lines which could not be found from line list
  V_Index.resize( D_A1_GaussPos.getShape()[ 0 ] );
  size_t pos = 0;
  for (auto it = D_A1_GaussPos.begin(); it != D_A1_GaussPos.end(); ++it, ++pos ){
    if ( *it > 0. )
      V_Index[ pos ] = 1;
    else
      V_Index[ pos ] = 0;
  }
  #ifdef __DEBUG_IDENTIFY__
    cout << "identify: D_A1_GaussPos = " << D_A1_GaussPos << endl;
    cout << "identify: V_Index = ";
    for (int iPos = 0; iPos < V_Index.size(); ++iPos)
      cout << V_Index[iPos] << " ";
    cout << endl;
  #endif
  std::vector< size_t > indices = math::getIndices( V_Index );
  size_t nInd = std::accumulate( V_Index.begin(), V_Index.end(), 0 );
  #ifdef __DEBUG_IDENTIFY__
    cout << "CFits::Identify: " << nInd << " lines identified" << endl;
    cout << "CFits::Identify: indices = ";
    for (int iPos = 0; iPos < indices.size(); ++iPos )
      cout << indices[iPos] << " ";
    cout << endl;
  #endif
  if ( nInd < ( std::round( double( lineList.getShape()[ 0 ] ) * 0.66 ) ) ){
    std::string message("pfs::drp::stella::identify: ERROR: ");
    message += "identify: ERROR: less than " + std::to_string( std::round( double( lineList.getShape()[ 0 ] ) * 0.66 ) ) + " lines identified";
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  else{
    ndarray::Array< size_t, 1, 1 > I_A1_IndexPos = ndarray::external( indices.data(), ndarray::makeVector( int( indices.size() ) ), ndarray::makeVector( 1 ) );
    ndarray::Array< double, 1, 1 > D_A1_WLen = ndarray::allocate( lineList.getShape()[ 0 ] );
    ndarray::Array< double, 1, 1 > D_A1_FittedPos = math::getSubArray( D_A1_GaussPos, I_A1_IndexPos );
    #ifdef __DEBUG_IDENTIFY__
      cout << "identify: D_A1_FittedPos = " << D_A1_FittedPos << endl;
    #endif

    D_A1_WLen[ ndarray::view() ] = lineList[ ndarray::view()( 0 ) ];
    ndarray::Array< double, 1, 1 > D_A1_FittedWLen = math::getSubArray( D_A1_WLen, I_A1_IndexPos );
    cout << "CFits::Identify: found D_A1_FittedWLen = " << D_A1_FittedWLen << endl;

    _dispCoeffs = ndarray::allocate( dispCorControl.order + 1 );
    _dispCoeffs.deep() = math::PolyFit( D_A1_FittedPos,
                                        D_A1_FittedWLen,
                                        dispCorControl.order );
    ndarray::Array< double, 1, 1 > D_A1_WLen_Gauss = math::Poly( D_A1_FittedPos, 
                                                                 _dispCoeffs );
    cout << "CFits::Identify: D_A1_WLen_PolyFit = " << D_A1_WLen_Gauss << endl;
    cout << "identify: _dispCoeffs = " << _dispCoeffs << endl;

    ///Calculate RMS
    ndarray::Array< double, 1, 1 > D_A1_WLenMinusFit = ndarray::allocate( D_A1_WLen_Gauss.getShape()[ 0 ] );
    D_A1_WLenMinusFit.deep() = D_A1_FittedWLen - D_A1_WLen_Gauss;
    cout << "CFits::Identify: D_A1_WLenMinusFit = " << D_A1_WLenMinusFit << endl;
    _dispRms = math::calcRMS( D_A1_WLenMinusFit );
    cout << "CFits::Identify: _dispRms = " << _dispRms << endl;
    cout << "======================================" << endl;

    ///calibrate spectrum
    ndarray::Array< double, 1, 1 > D_A1_Indices = math::indGenNdArr( double( _spectrum.getShape()[ 0 ] ) );
    _wavelength = ndarray::allocate( _spectrum.getShape()[ 0 ] );
    _wavelength.deep() = math::Poly( D_A1_Indices, _dispCoeffs );
    #ifdef __DEBUG_IDENTIFY__
      cout << "identify: _wavelength = " << _wavelength << endl;
    #endif

    /// Check for monotonic
    if ( math::isMonotonic( _wavelength ) == 0 ){
      cout << "CFits::Identify: WARNING: Wavelength solution is not monotonic => Setting identifyResult.rms to 1000" << endl;
      _dispRms = 1000.;
      cout << "CFits::Identify: RMS = " << _dispRms << endl;
      cout << "======================================" << endl;
    }

    #ifdef __WITH_PLOTS__
      mglGraph gr;
      mglData MGLData_X;
      MGLData_X.Link(P_D_A1_Ind->data(), P_D_A1_Ind->size(), 0, 0);
      mglData MGLData_WLen_Fit;
      MGLData_WLen_Fit.Link(P_D_A1_WLen_Out->data(), P_D_A1_WLen_Out->size(), 0, 0);
      mglData MGLData_OrigPos;// = new mglData(2);
      MGLData_OrigPos.Link(D_A1_FittedPos.data(), D_A1_FittedPos.size(), 0, 0);
      mglData MGLData_OrigWLen;// = new mglData(2);
      MGLData_OrigWLen.Link(D_A1_FittedWLen.data(), D_A1_FittedWLen.size(), 0, 0);
      mglData MGLData_FittedPos;// = new mglData(2);
      MGLData_FittedPos.Link(D_A1_FittedPos.data(), D_A1_FittedPos.size(), 0, 0);
      mglData MGLData_FittedWLen;// = new mglData(2);
      MGLData_FittedWLen.Link(P_D_A1_WLen_Gauss->data(), P_D_A1_WLen_Gauss->size(), 0, 0);

      gr.SetSize(1900,1200);
      gr.SetRanges(min(*P_D_A1_Ind),max(*P_D_A1_Ind),min(*P_D_A1_WLen_Out),max(*P_D_A1_WLen_Out)+(max(*P_D_A1_WLen_Out) - min(*P_D_A1_WLen_Out)) / 4.5);
      gr.Axis();
      gr.Label('y',"Wavelength",0);
      gr.Label('x',"Pixel Number",0);
      gr.Plot(MGLData_X, MGLData_WLen_Fit, "r");
      gr.AddLegend("Fit", "r");

      gr.Plot(MGLData_OrigPos, MGLData_OrigWLen, "bo ");
      gr.AddLegend("Line List", "bo ");

      gr.Plot(MGLData_FittedPos, MGLData_FittedWLen, "Yx ");
      gr.AddLegend("Fitted Positions", "Yx ");

      gr.Box();
      gr.Legend();

      CS_PlotName.Set(CS_FName_In);
      CS_PlotName.Add(CString("_Pix_WLen.png"));
      gr.WriteFrame(CS_PlotName.Get());

    #endif
  }

  _isWavelengthSet = true;
  return _isWavelengthSet;
}

///SpectrumSet
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::SpectrumSet(size_t nSpectra, size_t length)
        : _spectra(new std::vector<PTR(pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>)>())
{
  for (size_t i = 0; i < nSpectra; ++i){
    PTR(pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) spec(new pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>(length));
    _spectra->push_back(spec);
    (*_spectra)[i]->setITrace(i);
  }
}
    
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::SpectrumSet(const PTR(std::vector<PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>)>) &spectrumVector)
        : _spectra(spectrumVector)
{}
//  for (int i = 0; i < spectrumVector->size(); ++i){
//    (*_spectra)[i]->setITrace(i);
//  }
//}
    
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::setSpectrum(const size_t i,     /// which spectrum?
                     const PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) & spectrum /// the Spectrum for the ith aperture
                      )
{
  if (i > _spectra->size()){
    string message("SpectrumSet::setSpectrum(i=");
    message += to_string(i) + "): ERROR: i > _spectra->size()=" + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  if (i == static_cast<int>(_spectra->size())){
    _spectra->push_back(spectrum);
  }
  else{
    (*_spectra)[i] = spectrum;
  }
  return true;
}

/// add one Spectrum to the set
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
void pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::addSpectrum(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> const& spectrum /// the Spectrum to add
)
{
  PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) ptr(new Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>(spectrum));
  _spectra->push_back(ptr);
  return;
}
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
void pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::addSpectrum(PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) const& spectrum /// the Spectrum to add
)
{
  _spectra->push_back(spectrum);
  return;
}
  
template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, WavelengthT>)& pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getSpectrum(const size_t i){
  if (i >= _spectra->size()){
    string message("SpectrumSet::getSpectrum(i=");
    message += to_string(i) + "): ERROR: i >= _spectra->size()=" + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  return _spectra->at(i); 
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR(const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, WavelengthT>) const& pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getSpectrum(const size_t i) const { 
  if (i >= _spectra->size()){
    string message("SpectrumSet::getSpectrum(i=");
    message += to_string(i) + "): ERROR: i >= _spectra->size()=" + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  return PTR(const pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, WavelengthT>)(_spectra->at(i)); 
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::erase(const size_t iStart, const size_t iEnd){
  if (iStart >= _spectra->size()){
    string message("SpectrumSet::erase(iStart=");
    message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iStart >= _spectra->size()=" + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  if (iEnd >= _spectra->size()){
    string message("SpectrumSet::erase(iStart=");
    message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iEnd >= _spectra->size()=" + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  if (iEnd > 0){
    if (iStart > iEnd){
      string message("SpectrumSet::erase(iStart=");
      message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iStart > iEnd";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
  }
  if (iStart == (_spectra->size()-1)){
    _spectra->pop_back();
  }
  else{
    if (iEnd == 0)
      _spectra->erase(_spectra->begin() + iStart);
    else
      _spectra->erase(_spectra->begin() + iStart, _spectra->begin() + iEnd);
  }
  return true;
}

namespace pfs { namespace drp { namespace stella { namespace math {

    template< typename T, typename U >
    StretchAndCrossCorrelateSpecResult< T, U > stretchAndCrossCorrelateSpec( ndarray::Array< T, 1, 1 > const& spec,
                                                                             ndarray::Array< T, 1, 1 > const& specRef,
                                                                             ndarray::Array< U, 2, 1 > const& lineList_WLenPix,
                                                                             DispCorControl const& dispCorControl ){
//                                                            int const dispCorControl.radiusXCor,
//                                                            int const dispCorControl.stretchMinLength,
//                                                            int const dispCorControl.stretchMaxLength,
//                                                            int const dispCorControl.nStretches,
//                                                            int const dispCorControl.lengthPieces,
//                                                            int const nCalcs ){
//                                              int const polyFitOrder_Stretch,
//                                              int const polyFitOrder_Shift,
//                                              ndarray::Array< T, 2, 1 > & lineList_WLenPix_Out){
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
        cout << "stretchAndCrossCorrelateSpec: spec = " << spec.getShape() << ": " << spec << endl;
        cout << "stretchAndCrossCorrelateSpec: specRef = " << specRef.getShape() << ": " << specRef << endl;
        cout << "stretchAndCrossCorrelateSpec: lineList_WLenPix = " << lineList_WLenPix.getShape() << ": " << lineList_WLenPix << endl;
      #endif
      double fac = double( specRef.getShape()[ 0 ] ) / double( spec.getShape()[ 0 ] );
      ndarray::Array< T, 1, 1 > stretchedSpec = stretch( spec, specRef.getShape()[ 0 ] );
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
        cout << "stretchAndCrossCorrelateSpec: fac = " << fac << endl;
        cout << "stretchAndCrossCorrelateSpec: stretchedSpec = " << stretchedSpec.getShape() << ": " << stretchedSpec << endl;
      #endif

      if ( stretchedSpec.getShape()[ 0 ] != specRef.getShape()[ 0 ] ){
        cout << "stretchAndCrossCorrelate: ERROR: stretchedSpec.getShape()[0](=" << stretchedSpec.getShape()[ 0 ] << " != specRef.getShape()[0](=" << specRef.getShape() << ") => Returning FALSE" << endl;
        exit( EXIT_FAILURE );
      }
      StretchAndCrossCorrelateSpecResult< T, U > stretchAndCrossCorrelateSpecResult;
      stretchAndCrossCorrelateSpecResult.lineList = ndarray::allocate( lineList_WLenPix.getShape() );
////      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
        stretchAndCrossCorrelateSpecResult.specPieces = ndarray::allocate( ndarray::makeVector( dispCorControl.stretchMaxLength, 2, dispCorControl.nCalcs ) );
        stretchAndCrossCorrelateSpecResult.specPieces.deep() = 0.;
////      #endif
      int nCalcs = dispCorControl.nCalcs;
      if ( nCalcs < spec.getShape()[ 0 ] / dispCorControl.lengthPieces ){
        nCalcs = 2 * int( spec.getShape()[ 0 ] / dispCorControl.lengthPieces );
        cout << "stretchAndCrossCorrelate: Warning: dispCorControl.nCalcs(=" << dispCorControl.nCalcs << ") < spec.getShape()[0](=" << spec.getShape()[ 0 ] << ") / dispCorControl.lengthPieces(=" << dispCorControl.lengthPieces << ") => nCalcs set to " << nCalcs << endl;
      }
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
        cout << "stretchAndCrossCorrelateSpec: nCalcs = " << nCalcs << endl;
      #endif
      ndarray::Array< double, 1, 1 > chiSqMin_Stretch = ndarray::allocate( nCalcs );
      chiSqMin_Stretch.deep() = 0.;
      ndarray::Array< double, 1, 1 > chiSqMin_Shift = ndarray::allocate( nCalcs );
      chiSqMin_Shift.deep() = 0.;
      ndarray::Array< double, 1, 1 > xCenter = ndarray::allocate( nCalcs );
      xCenter.deep() = 0.;
      ndarray::Array< double, 1, 1 > specPiece = ndarray::allocate( dispCorControl.lengthPieces );
      ndarray::Array< double, 1, 1 > specRefPiece = ndarray::allocate( dispCorControl.lengthPieces );
      int start = 0;
      int end = 0;
      ndarray::Array< double, 2, 1 > specPieceStretched_MinChiSq;
      ndarray::Array< double, 2, 1 > lineList_Pixels_AllPieces = ndarray::allocate( lineList_WLenPix.getShape()[ 0 ], nCalcs );
      lineList_Pixels_AllPieces.deep() = 0.;
      ndarray::Array< double, 1, 1 > x = indGenNdArr( double( specRef.getShape()[ 0 ] ) );
      ndarray::Array< double, 1, 1 > xPiece;
      ndarray::Array< double, 1, 1 > xPieceStretched;

      for ( int i_run = 0; i_run < nCalcs; i_run++ ){
        end = start + dispCorControl.lengthPieces;
        if ( end >= stretchedSpec.getShape()[ 0 ] )
          end = spec.getShape()[ 0 ] - 1;
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
          cout << "stretchAndCrossCorrelateSpec: i_run = " << i_run << ": start = " << start << ", end = " << end << endl;
        #endif
        if ( end <= start ){
          cout << "stretchAndCrossCorrelateSpec: i_run = " << i_run << ": ERROR: end <= start" << endl;
          exit( EXIT_FAILURE );
        }
        xCenter[ i_run ] = double( start ) + ( double( end - start ) / 2. );
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
          cout << "stretchAndCrossCorrelateSpec: i_run = " << i_run << ": xCenter = " << xCenter[ i_run ] << endl;
        #endif

        specPiece = ndarray::allocate( end - start + 1);
        specPiece.deep() = stretchedSpec[ndarray::view( start, end + 1 ) ];
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
          cout << "stretchAndCrossCorrelateSpec: i_run = " << i_run << ": specPiece = " << specPiece.getShape() << ": " << specPiece << endl;
        #endif
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
          cout << "stretchAndCrossCorrelateSpec: i_run = " << i_run << ": specRefPiece = " << specRefPiece.getShape() << endl;
        #endif
        specRefPiece = ndarray::allocate( end - start + 1 );
        specRefPiece.deep() = specRef[ ndarray::view( start, end + 1 ) ];
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
          cout << "stretchAndCrossCorrelateSpec: i_run = " << i_run << ": specRefPiece = " << specRefPiece.getShape() << ": " << specRefPiece << endl;
        #endif
        /// stretch and crosscorrelate pieces
        StretchAndCrossCorrelateResult< double > stretchAndCrossCorrelateResult = stretchAndCrossCorrelate( specPiece,
                                                                                                            specRefPiece,
                                                                                                            dispCorControl.radiusXCor,
                                                                                                            dispCorControl.stretchMinLength,
                                                                                                            dispCorControl.stretchMaxLength,
                                                                                                            dispCorControl.nStretches );
        chiSqMin_Stretch[ i_run ] = stretchAndCrossCorrelateResult.stretch;
        chiSqMin_Shift[ i_run ] = stretchAndCrossCorrelateResult.shift;
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
          cout << "stretchAndCrossCorrelateSpec: i_run=" << i_run << ": chiSqMin_Stretch = " << chiSqMin_Stretch[i_run] << endl;
          cout << "stretchAndCrossCorrelateSpec: i_run=" << i_run << ": chiSqMin_Shift = " << chiSqMin_Shift[i_run] << endl;
          cout << "stretchAndCrossCorrelateSpec: i_run=" << i_run << ": stretchAndCrossCorrelateResult.specStretchedMinChiSq = " << stretchAndCrossCorrelateResult.specStretchedMinChiSq.getShape() << ": " << stretchAndCrossCorrelateResult.specStretchedMinChiSq << endl;
        #endif

        specPieceStretched_MinChiSq = ndarray::allocate( stretchAndCrossCorrelateResult.specStretchedMinChiSq.getShape() );
        specPieceStretched_MinChiSq.deep() = stretchAndCrossCorrelateResult.specStretchedMinChiSq;
        for ( int iSpecPos = 0; iSpecPos < specPieceStretched_MinChiSq.getShape()[ 0 ]; ++iSpecPos ){
          stretchAndCrossCorrelateSpecResult.specPieces[ ndarray::makeVector( iSpecPos, 0, i_run ) ] = specPieceStretched_MinChiSq[ ndarray::makeVector( iSpecPos, 0 ) ] + start;
          stretchAndCrossCorrelateSpecResult.specPieces[ ndarray::makeVector( iSpecPos, 1, i_run ) ] = specPieceStretched_MinChiSq[ ndarray::makeVector( iSpecPos, 1 ) ];
        }
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
          cout << "stretchAndCrossCorrelateSpec: stretchAndCrossCorrelateSpecResult.specPieces.getShape() = " << stretchAndCrossCorrelateSpecResult.specPieces.getShape() << endl;
          cout << "stretchAndCrossCorrelateSpec: stretchAndCrossCorrelateSpecResult.specPieces[ ndarray::view( 0, " << specPieceStretched_MinChiSq.getShape()[ 0 ] << " )( 0 )( " << i_run << " ) ] = " << stretchAndCrossCorrelateSpecResult.specPieces[ ndarray::view( 0, specPieceStretched_MinChiSq.getShape()[ 0 ] )( 0 )( i_run ) ] << endl;
          cout << "stretchAndCrossCorrelateSpec: stretchAndCrossCorrelateSpecResult.specPieces[ ndarray::view( 0, " << specPieceStretched_MinChiSq.getShape()[ 0 ] << " )( 1 )( " << i_run << " ) ] = " << stretchAndCrossCorrelateSpecResult.specPieces[ ndarray::view( 0, specPieceStretched_MinChiSq.getShape()[ 0 ] )( 1 )( i_run ) ] << endl;
        #endif
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
          cout << "stretchAndCrossCorrelateSpec: i_run=" << i_run << ": after stretchAndCrossCorrelate: stretchAndCrossCorrelateResult.specStretchedMinChiSq.getShape() = " << stretchAndCrossCorrelateResult.specStretchedMinChiSq.getShape() << endl;
          cout << "stretchAndCrossCorrelateSpec: i_run=" << i_run << ": after stretchAndCrossCorrelate: specPieceStretched_MinChiSq = " << specPieceStretched_MinChiSq.getShape() << ": " << specPieceStretched_MinChiSq << endl;
        #endif

        xPiece = ndarray::allocate( end - start + 1 );
        xPiece.deep() = x[ndarray::view( start, end + 1 ) ];

        xPieceStretched = ndarray::allocate( chiSqMin_Stretch[ i_run ] );
        xPieceStretched[ 0 ] = start;
        for ( int i_pix=1; i_pix < xPieceStretched.getShape()[ 0 ]; i_pix++ ){
          xPieceStretched[ i_pix ] = xPieceStretched[ i_pix - 1 ] + (xPiece.getShape()[ 0 ] / chiSqMin_Stretch[ i_run ] );
        }
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
          cout << "stretchAndCrossCorrelateSpec: i_run=" << i_run << ": xPieceStretched = " << xPiece.getShape() << ": " << xPiece << endl;
          cout << "stretchAndCrossCorrelateSpec: i_run=" << i_run << ": xPieceStretched = " << xPieceStretched.getShape() << ": " << xPieceStretched << endl;
        #endif

        double weightLeft = 0.;
        double weightRight = 0.;
        ndarray::Array< double, 1, 1 > lineListPix = ndarray::allocate( lineList_WLenPix.getShape()[ 0 ] );
        auto itTemp = lineListPix.begin();
        for ( auto itList = lineList_WLenPix.begin(); itList != lineList_WLenPix.end(); ++itList, ++itTemp ){
          auto itListCol = itList->begin() + 1;
          *itTemp = double( *itListCol );//lineList_WLenPix[ ndarray::view()( 1 ) ] );
        }
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
          cout << "stretchAndCrossCorrelateSpec: i_run=" << i_run << ": lineListPix = " << lineListPix.getShape() << ": " << lineListPix << endl;
        #endif

        ndarray::Array< int, 1, 1 > valueLocated = valueLocate( xPieceStretched, 
                                                                lineListPix );
        for ( int i_line = 0; i_line < lineList_Pixels_AllPieces.getShape()[ 0 ]; i_line++ ){//i_line < lineList_Pixels_AllPieces.rows()
          #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
            cout << "stretchAndCrossCorrelateSpec: i_run=" << i_run << ": i_line = " << i_line << ": valueLocated[ i_line ] = " << valueLocated[i_line] << ", xPieceStretched.getShape() = " << xPieceStretched.getShape() << endl;
          #endif
          if ( ( valueLocated[ i_line ] >= 0 ) && ( valueLocated[ i_line ] < xPieceStretched.getShape()[ 0 ] - 1 ) ){
            weightRight = ( xPieceStretched[ valueLocated[ i_line ] + 1 ] - xPieceStretched[ valueLocated[ i_line ] ] ) * ( lineList_WLenPix[ ndarray::makeVector( i_line, 1 ) ] - xPieceStretched[ valueLocated[ i_line ] ] );
            weightLeft = 1. - weightRight;
            #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
              cout << "stretchAndCrossCorrelateSpec: i_run=" << i_run << ": i_line = " << i_line << ": xPieceStretched[ valueLocated[ i_line ]=" << valueLocated[i_line] << ") = " << xPieceStretched[valueLocated[i_line]] << ", xPieceStretched[valueLocate[i_line]+1=" << valueLocated[i_line]+1 << ") = " << xPieceStretched[valueLocated[i_line]+1] << ", weightRight = " << weightRight << ": weightLeft = " << weightLeft << endl;
            #endif
            lineList_Pixels_AllPieces[ ndarray::makeVector( i_line, i_run ) ] = start + ( valueLocated[ i_line ] * weightLeft ) + ( ( valueLocated[ i_line ] + 1 ) * weightRight ) - chiSqMin_Shift[ i_run ];
            #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
              cout << "stretchAndCrossCorrelateSpec: i_run=" << i_run << ": i_line = " << i_line << ": lineList_Pixels_AllPieces[i_line][i_run] = " << lineList_Pixels_AllPieces[ndarray::makeVector( i_line, i_run ) ] << endl;
            #endif
          }
        }

        // for next run
        start += ( stretchedSpec.getShape()[ 0 ] - dispCorControl.lengthPieces ) / ( nCalcs - 1 );
      }/// end for (int i_run = 0; i_run < I_NStretches_In; i_run++){

      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
        cout << "stretchAndCrossCorrelateSpec: chiSqMin_Shift = " << chiSqMin_Shift << endl;
        cout << "stretchAndCrossCorrelateSpec: chiSqMin_Stretch = " << chiSqMin_Stretch << endl;
        cout << "stretchAndCrossCorrelateSpec: lineList_Pixels_AllPieces = " << lineList_Pixels_AllPieces << endl;
      #endif

      int nInd = 0;
      for ( int iLine = 0; iLine < lineList_WLenPix.getShape()[ 0 ]; ++iLine ){
        stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( iLine, 0 ) ] = lineList_WLenPix[ ndarray::makeVector( iLine, 0 ) ];
      }
      ndarray::Array< double, 1, 1 > tempArr = ndarray::allocate( lineList_Pixels_AllPieces.getShape()[ 1 ] );
      for (int i_line=0; i_line < lineList_WLenPix.getShape()[ 0 ]; i_line++){
        tempArr[ ndarray::view() ] = lineList_Pixels_AllPieces[ ndarray::view( i_line )() ];
        ndarray::Array< int, 1, 1 > whereVec = where( tempArr,
                                                      ">", 
                                                      0.001, 
                                                      1, 
                                                      0 );
        nInd = std::accumulate( whereVec.begin(), whereVec.end(), 0 );
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
          cout << "stretchAndCrossCorrelateSpec: i_line = " << i_line << ": whereVec = " << whereVec << ": nInd = " << nInd << endl;
        #endif
        ndarray::Array< size_t, 1, 1 > indWhere = getIndices( whereVec );
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
          cout << "stretchAndCrossCorrelateSpec: i_line = " << i_line << ": indWhere = " << indWhere << ", nInd = " << nInd << endl;
        #endif
        if ( nInd == 1 ){
          stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] = lineList_Pixels_AllPieces[ ndarray::makeVector( i_line, int( indWhere[ 0 ] ) ) ];
          #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
            cout << "stretchAndCrossCorrelateSpec: i_line = " << i_line << ": nInd == 1: stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] set to " << stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] << endl;
          #endif
        }
        else{
          stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] = 0.;
          for (int i_ind = 0; i_ind < nInd; i_ind++){
            stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] += lineList_Pixels_AllPieces[ ndarray::makeVector( i_line, int( indWhere[ i_ind ] ) ) ];
            #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
              cout << "stretchAndCrossCorrelateSpec: i_line = " << i_line << ": nInd != 1: stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] set to " << stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] << endl;
            #endif
          }
          stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] = stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] / nInd;
          #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
            cout << "stretchAndCrossCorrelateSpec: i_line = " << i_line << ": nInd != 1: stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] set to " << stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] << endl;
          #endif
        }
        if ( lineList_WLenPix.getShape()[ 1 ] == 3 ){
          stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 2 ) ] = lineList_WLenPix[ ndarray::makeVector( i_line, 2 ) ];
          #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
            cout << "stretchAndCrossCorrelateSpec: i_line = " << i_line << ": stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 2 ) ] set to " << stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 2 ) ] << endl;
          #endif
        }
      }
      cout << "stretchAndCrossCorrelateSpec: stretchAndCrossCorrelateSpecResult.lineList = " << stretchAndCrossCorrelateSpecResult.lineList << endl;

      /// Check positions
      ndarray::Array< double, 2, 1 > dist = ndarray::allocate( lineList_Pixels_AllPieces.getShape() );
      dist.deep() = 0.;
      for ( int i_row = 0; i_row < lineList_Pixels_AllPieces.getShape()[ 0 ]; i_row++){
        for (int i_col = 0; i_col < lineList_Pixels_AllPieces.getShape()[ 1 ]; i_col++){
          if ( std::fabs( lineList_Pixels_AllPieces[ ndarray::makeVector( i_row, i_col ) ] ) > 0.00000000000001 )
            dist[ ndarray::makeVector( i_row, i_col ) ] = lineList_Pixels_AllPieces[ ndarray::makeVector( i_row, i_col ) ] - lineList_WLenPix[ ndarray::makeVector( i_row, 1 ) ];
        }
      }
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
        cout << "stretchAndCrossCorrelateSpec: dist = " << dist << endl;
        cout << "stretchAndCrossCorrelateSpec: lineList_Pixels_AllPieces = " << lineList_Pixels_AllPieces << endl;
      #endif
      ndarray::Array< int, 2, 1 > whereArr = where( lineList_Pixels_AllPieces,
                                                    ">",
                                                    0.000001, 
                                                    1, 
                                                    0);
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
        cout << "stretchAndCrossCorrelateSpec: whereArr = " << whereArr << endl;
      #endif
      ndarray::Array< size_t, 2, 1 > indWhereArr = getIndices( whereArr );
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
        cout << "stretchAndCrossCorrelateSpec: indWhereArr = " << indWhereArr << endl;
      #endif
      ndarray::Array< double, 1, 1 > dist_SubArr = getSubArray( dist, indWhereArr );
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
        cout << "stretchAndCrossCorrelateSpec: dist_SubArr = " << dist_SubArr << endl;
      #endif
      double medianDiff = median( dist_SubArr );
      ndarray::Array< double, 1, 1 > sorted = ndarray::allocate( dist_SubArr.getShape()[ 0 ] );
      sorted.deep() = dist_SubArr;
      std::sort( sorted.begin(), sorted.end() );
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
        cout << "stretchAndCrossCorrelateSpec: medianDiff = " << medianDiff << endl;
        cout << "stretchAndCrossCorrelateSpec: sorted = " << sorted << endl;
      #endif
      ndarray::Array< double, 1, 1 > dist_Temp = ndarray::allocate( dist_SubArr.getShape()[ 0 ] - 4 );
      dist_Temp = sorted[ ndarray::view( 2, dist_SubArr.getShape()[ 0 ] - 2 ) ];
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
        cout << "stretchAndCrossCorrelateSpec: dist_Temp = " << dist_Temp << endl;
      #endif
      ndarray::Array< double, 1, 1 > moments = moment( dist_Temp, 2 );
      double stdDev_Diff = moments[ 1 ];
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
        cout << "stretchAndCrossCorrelateSpec: stdDev_Diff = " << stdDev_Diff << endl;
      #endif
      ndarray::Array< double, 1, 1 > tempDist = ndarray::copy( dist_SubArr - medianDiff );
      for ( auto itDist = tempDist.begin(); itDist != tempDist.end(); ++itDist )
        *itDist = std::fabs( *itDist );
      ndarray::Array< int, 1, 1 > whereVec = where( tempDist, 
                                                    ">", 
                                                    3. * stdDev_Diff, 
                                                    1, 
                                                    0 );
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
        cout << "stretchAndCrossCorrelateSpec: whereVec = " << whereVec << endl;
      #endif
      int nBad = std::accumulate( whereVec.begin(), whereVec.end(), 0 );
      if ( nBad > 0 ){
        ndarray::Array< size_t, 1, 1 > indWhereA = getIndices( whereVec );
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
          cout << "stretchAndCrossCorrelateSpec: nBad = " << nBad << ": indWhereA = " << indWhereA << endl;
        #endif
        for ( int i_bad = 0; i_bad < nBad; ++i_bad ){
          lineList_Pixels_AllPieces[ ndarray::makeVector( int( indWhereArr[ ndarray::makeVector( int( indWhereA[ i_bad ] ), 0 ) ] ), int( indWhereArr[ ndarray::makeVector( int( indWhereA[ i_bad ] ), 1 ) ] ) ) ] = 0.;
          #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
            cout << "stretchAndCrossCorrelateSpec: i_bad = " << i_bad << ": lineList_Pixels_AllPieces[" << indWhereArr[ ndarray::makeVector( int(indWhereA[ i_bad ]), 0 ) ] << "][" << indWhereArr[ ndarray::makeVector( int(indWhereA[ i_bad ]), 1 ) ] << "] set to " << lineList_Pixels_AllPieces[ ndarray::makeVector( int(indWhereArr[ ndarray::makeVector( int(indWhereA[i_bad]), 0 ) ]), int(indWhereArr[ ndarray::makeVector( int(indWhereA[ i_bad ]), 1 ) ] ) ) ] << endl;
          #endif
        }
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
          cout << "stretchAndCrossCorrelateSpec: lineList_Pixels_AllPieces = " << lineList_Pixels_AllPieces << endl;
        #endif

        stretchAndCrossCorrelateSpecResult.lineList[ ndarray::view()(1) ] = 0.;
        ndarray::Array< double, 1, 1 > tempList = ndarray::allocate( lineList_Pixels_AllPieces.getShape()[ 1 ] );
        for (int i_line=0; i_line < lineList_WLenPix.getShape()[ 0 ]; i_line++){
          tempList = lineList_Pixels_AllPieces[ ndarray::view(i_line)() ];
          for ( auto itTemp = tempList.begin(); itTemp != tempList.end(); ++itTemp )
            *itTemp = std::fabs( *itTemp );
          ndarray::Array< int, 1, 1 > whereVec = where( tempList,
                                                        ">",
                                                        0.001, 
                                                        1, 
                                                        0 );
          nInd = std::accumulate( whereVec.begin(), whereVec.end(), 0 );
          ndarray::Array< size_t, 1, 1 > indWhereB = getIndices( whereVec );
          #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
            cout << "stretchAndCrossCorrelateSpec: i_line = " << i_line << ": nInd = " << nInd << endl;
          #endif
          if ( nInd == 0 )
            stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] = lineList_WLenPix[ ndarray::makeVector( i_line, 1 ) ] + medianDiff;
          else if ( nInd == 1 )
            stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] = lineList_Pixels_AllPieces[ ndarray::makeVector( i_line, int( indWhereB[ 0 ] ) ) ];
          else{
            for (int i_ind = 0; i_ind < nInd; i_ind++ ){
              stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] += lineList_Pixels_AllPieces[ ndarray::makeVector( i_line, int( indWhereB[ i_ind ] ) ) ];
              #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
                cout << "stretchAndCrossCorrelateSpec: i_line = " << i_line << ": i_ind = " << i_ind << ": indWhereB[" << i_ind << "] = " << indWhereB[i_ind] << endl;
                cout << "stretchAndCrossCorrelateSpec: i_line = " << i_line << ": i_ind = " << i_ind << ": stretchAndCrossCorrelateSpecResult.lineList[" << i_line << "][1] set to " << stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] << endl;
              #endif
            }
            stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] = stretchAndCrossCorrelateSpecResult.lineList[ ndarray::makeVector( i_line, 1 ) ] / nInd;
          }
          #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
            cout << "stretchAndCrossCorrelateSpec: stretchAndCrossCorrelateSpecResult.lineList[" << i_line << "][1] set to " << stretchAndCrossCorrelateSpecResult.lineList[ndarray::makeVector(i_line,1)] << endl;
          #endif
        }
      }

      stretchAndCrossCorrelateSpecResult.lineList[ ndarray::view()( 1 ) ] = stretchAndCrossCorrelateSpecResult.lineList[ ndarray::view()( 1 ) ] / fac;
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
        cout << "stretchAndCrossCorrelateSpec: lineList_Pixels_AllPieces = " << lineList_Pixels_AllPieces << endl;
        cout << "stretchAndCrossCorrelateSpec: lineList_WLenPix = " << lineList_WLenPix << endl;
        cout << "stretchAndCrossCorrelateSpec: stretchAndCrossCorrelateSpecResult.lineList = " << stretchAndCrossCorrelateSpecResult.lineList << endl;
      #endif

      return stretchAndCrossCorrelateSpecResult;
    }
    
    template StretchAndCrossCorrelateSpecResult< float, float >;
    template StretchAndCrossCorrelateSpecResult< double, double >;
    template StretchAndCrossCorrelateSpecResult< float, double >;
    template StretchAndCrossCorrelateSpecResult< double, float >;

    template StretchAndCrossCorrelateSpecResult< float, float > stretchAndCrossCorrelateSpec( ndarray::Array< float, 1, 1 > const&,
                                                                                              ndarray::Array< float, 1, 1 > const&,
                                                                                              ndarray::Array< float, 2, 1 > const&,
                                                                                              DispCorControl const& );
    template StretchAndCrossCorrelateSpecResult< double, double > stretchAndCrossCorrelateSpec( ndarray::Array< double, 1, 1 > const&,
                                                                                                ndarray::Array< double, 1, 1 > const&,
                                                                                                ndarray::Array< double, 2, 1 > const&,
                                                                                                DispCorControl const& );
    template StretchAndCrossCorrelateSpecResult< double, float > stretchAndCrossCorrelateSpec( ndarray::Array< double, 1, 1 > const&,
                                                                                               ndarray::Array< double, 1, 1 > const&,
                                                                                               ndarray::Array< float, 2, 1 > const&,
                                                                                               DispCorControl const& );
    template StretchAndCrossCorrelateSpecResult< float, double > stretchAndCrossCorrelateSpec( ndarray::Array< float, 1, 1 > const&,
                                                                                               ndarray::Array< float, 1, 1 > const&,
                                                                                               ndarray::Array< double, 2, 1 > const&,
                                                                                               DispCorControl const& );

}}}}
  
template<typename T>
PTR(T) pfsDRPStella::utils::getPointer(T &obj){
  PTR(T) pointer(new T(obj));
  return pointer;
}

//template class pfsDRPStella::Spectrum<float>;
//template class pfsDRPStella::Spectrum<double>;
template class pfsDRPStella::Spectrum<float, unsigned int, float, float>;
template class pfsDRPStella::Spectrum<double, unsigned int, float, float>;
template class pfsDRPStella::Spectrum<float, unsigned short, float, float>;
template class pfsDRPStella::Spectrum<double, unsigned short, float, float>;
//template class pfsDRPStella::Spectrum<float, unsigned int, double, double>;
//template class pfsDRPStella::Spectrum<double, unsigned int, double, double>;
//template class pfsDRPStella::Spectrum<float, unsigned short, double, double>;
//template class pfsDRPStella::Spectrum<double, unsigned short, double, double>;

//template class pfsDRPStella::SpectrumSet<float>;
//template class pfsDRPStella::SpectrumSet<double>;
template class pfsDRPStella::SpectrumSet<float, unsigned int, float, float>;
template class pfsDRPStella::SpectrumSet<double, unsigned int, float, float>;
template class pfsDRPStella::SpectrumSet<float, unsigned short, float, float>;
template class pfsDRPStella::SpectrumSet<double, unsigned short, float, float>;
template class pfsDRPStella::SpectrumSet<float, unsigned int, float, double>;
template class pfsDRPStella::SpectrumSet<double, unsigned int, float, double>;
template class pfsDRPStella::SpectrumSet<float, unsigned short, float, double>;
template class pfsDRPStella::SpectrumSet<double, unsigned short, float, double>;

template PTR(afwImage::MaskedImage<float, unsigned short, float>) pfsDRPStella::utils::getPointer(afwImage::MaskedImage<float, unsigned short, float> &);
template PTR(afwImage::MaskedImage<double, unsigned short, float>) pfsDRPStella::utils::getPointer(afwImage::MaskedImage<double, unsigned short, float> &);
template PTR(std::vector<unsigned short>) pfsDRPStella::utils::getPointer(std::vector<unsigned short> &);
template PTR(std::vector<unsigned int>) pfsDRPStella::utils::getPointer(std::vector<unsigned int> &);
template PTR(std::vector<int>) pfsDRPStella::utils::getPointer(std::vector<int> &);
template PTR(std::vector<float>) pfsDRPStella::utils::getPointer(std::vector<float> &);
template PTR(std::vector<double>) pfsDRPStella::utils::getPointer(std::vector<double> &);
template PTR(pfsDRPStella::Spectrum<float, unsigned short, float, float>) pfsDRPStella::utils::getPointer(pfsDRPStella::Spectrum<float, unsigned short, float, float> &);
template PTR(pfsDRPStella::Spectrum<double, unsigned short, float, float>) pfsDRPStella::utils::getPointer(pfsDRPStella::Spectrum<double, unsigned short, float, float> &);

