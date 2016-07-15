template< typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT >
pfs::drp::stella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT >::Spectrum( Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > & spectrum,
                                                                                  size_t iTrace,
                                                                                  bool deep ) 
:       _yLow( spectrum.getYLow() ),
        _yHigh( spectrum.getYHigh() ),
        _length( spectrum.getLength() ),
        _nCCDRows( spectrum.getNCCDRows() ),
        _spectrum( spectrum.getSpectrum() ),
        _sky( spectrum.getSky() ),
        _mask( spectrum.getMask() ),
        _covar( spectrum.getCovar() ),
        _wavelength( spectrum.getWavelength() ),
        _dispersion( spectrum.getDispersion() ),
        _iTrace( spectrum.getITrace() ),
        _dispCoeffs( spectrum.getDispCoeffs() ),
        _dispRms( spectrum.getDispRms() ),
        _isWavelengthSet( spectrum.isWavelengthSet() ),
        _dispCorControl( spectrum.getDispCorControl() )
{
    if ( deep ){
        /// allocate memory
        _spectrum = ndarray::allocate(spectrum.getSpectrum().getShape()[0]);
        _sky = ndarray::allocate(spectrum.getSky().getShape()[0]);
        _mask = ndarray::allocate(spectrum.getMask().getShape()[0]);
        _covar = ndarray::allocate(spectrum.getCovar().getShape());
        _wavelength = ndarray::allocate(spectrum.getWavelength().getShape()[0]);
        _dispersion = ndarray::allocate(spectrum.getDispersion().getShape()[0]);
        _dispCoeffs = ndarray::allocate(spectrum.getDispCoeffs().getShape()[0]);

        /// copy variables
        _spectrum.deep() = spectrum.getSpectrum();
        _sky.deep() = spectrum.getSky();
        _mask.deep() = spectrum.getMask();
        _covar.deep() = spectrum.getCovar();
        _wavelength.deep() = spectrum.getWavelength();
        _dispersion.deep() = spectrum.getDispersion();
        _dispCoeffs.deep() = spectrum.getDispCoeffs();
    }
    if (iTrace != 0)
        _iTrace = iTrace;
}

template< typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT >
pfs::drp::stella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT >::Spectrum( Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > const& spectrum) //,
                                                                                  //int i ) 
:       _yLow( spectrum.getYLow() ),
        _yHigh( spectrum.getYHigh() ),
        _length( spectrum.getLength() ),
        _nCCDRows( spectrum.getNCCDRows() ),
        _iTrace( spectrum.getITrace() ),
        _dispRms( spectrum.getDispRms() ),
        _isWavelengthSet( spectrum.isWavelengthSet() ),
        _dispCorControl( spectrum.getDispCorControl() )
{
    /// allocate memory
    _spectrum = ndarray::allocate(spectrum.getSpectrum().getShape()[0]);
    _sky = ndarray::allocate(spectrum.getSky().getShape()[0]);
    _mask = ndarray::allocate(spectrum.getMask().getShape()[0]);
    _covar = ndarray::allocate(spectrum.getCovar().getShape());
    _wavelength = ndarray::allocate(spectrum.getWavelength().getShape()[0]);
    _dispersion = ndarray::allocate(spectrum.getDispersion().getShape()[0]);
    _dispCoeffs = ndarray::allocate(spectrum.getDispCoeffs().getShape()[0]);

    /// copy variables
    _spectrum.deep() = spectrum.getSpectrum();
    _sky.deep() = spectrum.getSky();
    _mask.deep() = spectrum.getMask();
    _covar.deep() = spectrum.getCovar();
    _wavelength.deep() = spectrum.getWavelength();
    _dispersion.deep() = spectrum.getDispersion();
    _dispCoeffs.deep() = spectrum.getDispCoeffs();
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfs::drp::stella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setSpectrum( ndarray::Array<SpectrumT, 1, 1> const& spectrum )
{
  /// Check length of input spectrum
  if (static_cast<size_t>(spectrum.getShape()[0]) != _length) {
    string message("pfs::drp::stella::Spectrum::setSpectrum: ERROR: spectrum->size()=");
    message += to_string(spectrum.getShape()[0]) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _spectrum.deep() = spectrum;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfs::drp::stella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setVariance( ndarray::Array<VarianceT, 1, 1> const& variance )
{
  /// Check length of input covar
  if (static_cast<size_t>(variance.getShape()[0]) != _length) {
    string message("pfs::drp::stella::Spectrum::setVariance: ERROR: variance->size()=");
    message += to_string( variance.getShape()[ 0 ] ) + string( " != _length=" ) + to_string( _length );
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _covar[ ndarray::view( 3 )( ) ] = variance;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfs::drp::stella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setCovar(const ndarray::Array<VarianceT, 2, 1> & covar )
{
    /// Check length of input covar
    if (static_cast<size_t>(covar.getShape()[1]) != _length) {
      string message("pfs::drp::stella::Spectrum::setCovar: ERROR: covar->size()=");
      message += to_string( covar.getShape()[ 1 ] ) + string( " != _length=" ) + to_string( _length );
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    _covar.deep() = covar;
    return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfs::drp::stella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setMask(const ndarray::Array<MaskT, 1, 1> & mask)
{
  /// Check length of input mask
  if (static_cast<size_t>(mask.getShape()[0]) != _length){
    string message("pfs::drp::stella::Spectrum::setMask: ERROR: mask->size()=");
    message += to_string(mask.getShape()[0]) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _mask.deep() = mask;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfs::drp::stella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setLength(const size_t length)
{
    #ifdef __DEBUG_SETLENGTH__
      cout << "pfs::drp::stella::Spectrum::setLength: starting to set _length to " << length << endl;
    #endif
    pfs::drp::stella::math::resize(_spectrum, length);
    #ifdef __DEBUG_SETLENGTH__
      cout << "pfs::drp::stella::Spectrum::setLength: _spectrum resized to " << _spectrum.getShape()[0] << endl;
    #endif
    pfs::drp::stella::math::resize(_mask, length);
    #ifdef __DEBUG_SETLENGTH__
      cout << "pfs::drp::stella::Spectrum::setLength: _mask resized to " << _mask.getShape()[0] << endl;
    #endif
    pfs::drp::stella::math::resize(_covar, 5, length);
    #ifdef __DEBUG_SETLENGTH__
      cout << "pfs::drp::stella::Spectrum::setLength: _covar resized to " << _covar.getShape()[0] << "x" << _covar.getShape()[1] << endl;
    #endif
    pfs::drp::stella::math::resize(_wavelength, length);
    #ifdef __DEBUG_SETLENGTH__
      cout << "pfs::drp::stella::Spectrum::setLength: _wavelength resized to " << _wavelength.getShape()[0] << endl;
    #endif
    if (length > _length){
      WavelengthT val = _wavelength[_length = 1];
      for (auto it = _wavelength.begin() + length; it != _wavelength.end(); ++it)
        *it = val;
    }
    _length = length;
    _yHigh = _yLow + _length - 1;
    #ifdef __DEBUG_SETLENGTH__
      cout << "pfs::drp::stella::Spectrum::setLength: finishing: _length to " << _length << endl;
    #endif
    return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfs::drp::stella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT >::setYLow( const size_t yLow )
{
  if ( yLow > _nCCDRows ){
    string message("pfs::drp::stella::Spectrum::setYLow: ERROR: yLow=");
    message += to_string( yLow ) + string(" > _nCCDRows=") + to_string(_nCCDRows);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _yLow = yLow;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfs::drp::stella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setYHigh(const size_t yHigh)
{
  if ( yHigh > _nCCDRows ){
    _nCCDRows = _yLow + yHigh;
  }
  _yHigh = yHigh;
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfs::drp::stella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setNCCDRows(const size_t nCCDRows)
{
  if ( _yLow > nCCDRows ){
    string message("pfs::drp::stella::Spectrum::setYLow: ERROR: _yLow=");
    message += to_string( _yLow ) + string(" > nCCDRows=") + to_string(nCCDRows);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _nCCDRows = nCCDRows;
  return true;
}

template< typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT >
template< typename T >
ndarray::Array< double, 1, 1 > pfs::drp::stella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::hIdentify( ndarray::Array< T, 2, 1 > const& lineList )
{
  ///for each line in line list, find maximum in spectrum and fit Gaussian
  int I_MaxPos = 0;
  int I_Start = 0;
  int I_End = 0;
  int I_NTerms = 4;
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
    I_Start = int( lineList[ ndarray::makeVector( i_line, 1 ) ] ) - _dispCorControl->searchRadius;
    if ( I_Start < 0 )
      I_Start = 0;
    #ifdef __DEBUG_IDENTIFY__
      cout << "identify: i_line = " << i_line << ": I_Start = " << I_Start << endl;
    #endif
    I_End = int( lineList[ ndarray::makeVector( i_line, 1 ) ] ) + _dispCorControl->searchRadius;
    if ( I_End >= _spectrum.getShape()[ 0 ] )
      I_End = _spectrum.getShape()[ 0 ] - 1;
    if ( ( I_End - I_Start ) > ( 1.5 * _dispCorControl->searchRadius ) ){
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
        I_Start = std::round( double( I_MaxPos ) - ( 1.5 * _dispCorControl->fwhm ) );
        if (I_Start < 0)
          I_Start = 0;
        #ifdef __DEBUG_IDENTIFY__
          cout << "identify: I_Start = " << I_Start << endl;
        #endif
        I_End = std::round( double( I_MaxPos ) + ( 1.5 * _dispCorControl->fwhm ) );
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

        /*     p[3] = constant offset
         *     p[0] = peak y value
         *     p[1] = x centroid position
         *     p[2] = gaussian sigma width
         */
  //          ndarray::Array< double, 2, 1 > toFit = ndarray::allocate( D_A1_X.getShape()[ 0 ], 2 );
  //          toFit[ ndarray::view()(0) ] = D_A1_X;
  //          toFit[ ndarray::view()(1) ] = D_A1_GaussSpec;
  //            ndarray::Array< double, 1, 1 > gaussFitResult = gaussFit()
          D_A1_Guess[ 3 ] = *min_element( V_GaussSpec.begin(), V_GaussSpec.end() );
          D_A1_Guess[ 0 ] = *max_element( V_GaussSpec.begin(), V_GaussSpec.end() ) - D_A1_Guess(3);
          D_A1_Guess[ 1 ] = V_X[ 0 ] + ( V_X[ V_X.size() - 1 ] - V_X[ 0 ] ) / 2.;
          D_A1_Guess[ 2 ] = _dispCorControl->fwhm;
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
    }
  }/// end for (int i_line=0; i_line < D_A2_LineList_In.rows(); i_line++){
  return D_A1_GaussPos;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfs::drp::stella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT >::setDispCoeffs( ndarray::Array< double, 1, 1 > const& dispCoeffs )
{
  /// Check length of input dispCoeffs
  if (dispCoeffs.getShape()[0] != ( _dispCorControl->order + 1 ) ){
    string message("pfsDRPStella::Spectrum::setDispCoeffs: ERROR: dispCoeffs.size()=");
    message += to_string(dispCoeffs.getShape()[0]) + string(" != _dispCorControl->order + 1 =") + to_string( _dispCorControl->order + 1 );
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _dispCoeffs = ndarray::allocate( dispCoeffs.getShape()[ 0 ] );
  _dispCoeffs.deep() = dispCoeffs;
  cout << "pfsDRPStella::setDispCoeffs: _dispCoeffs set to " << _dispCoeffs << endl;
  return true;
}

template< typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT >
template< typename T >
bool pfs::drp::stella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::identify( ndarray::Array< T, 2, 1 > const& lineList,
                                                                                     pfs::drp::stella::DispCorControl const& dispCorControl,
                                                                                     size_t nLinesCheck ){
    pfs::drp::stella::DispCorControl tempDispCorControl( dispCorControl );
    _dispCorControl.reset();
    _dispCorControl = tempDispCorControl.getPointer();

    ///for each line in line list, find maximum in spectrum and fit Gaussian
    ndarray::Array< double, 1, 1 > D_A1_GaussPos = hIdentify( lineList );

    ///remove lines which could not be found from line list
    std::vector< int > V_Index( D_A1_GaussPos.getShape()[ 0 ], 0 );
    size_t pos = 0;
    for (auto it = D_A1_GaussPos.begin(); it != D_A1_GaussPos.end(); ++it, ++pos ){
      if ( *it > 0. )
        V_Index[ pos ] = 1;
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
      cout << "Identify: " << nInd << " lines identified" << endl;
      cout << "Identify: indices = ";
      for (int iPos = 0; iPos < indices.size(); ++iPos )
        cout << indices[iPos] << " ";
      cout << endl;
    #endif

    /// separate lines to fit and lines for RMS test
    std::vector< size_t > indCheck;
    for ( size_t i = 0; i < nLinesCheck; ++i ){
      srand( 0 ); //seed initialization
      int randNum = rand() % ( indices.size() - 2 ) + 1; // Generate a random number between 0 and 1
      indCheck.push_back( size_t( randNum ) );
      indices.erase( indices.begin() + randNum );
    }

    if ( nInd < ( std::round( double( lineList.getShape()[ 0 ] ) * 0.66 ) ) ){
      std::string message("pfs::drp::stella::identify: ERROR: ");
      message += "identify: ERROR: less than " + std::to_string( std::round( double( lineList.getShape()[ 0 ] ) * 0.66 ) ) + " lines identified";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    ndarray::Array< size_t, 1, 1 > I_A1_IndexPos = ndarray::external( indices.data(), ndarray::makeVector( int( indices.size() ) ), ndarray::makeVector( 1 ) );
    ndarray::Array< double, 1, 1 > D_A1_WLen = ndarray::allocate( lineList.getShape()[ 0 ] );
    ndarray::Array< double, 1, 1 > D_A1_FittedPos = math::getSubArray( D_A1_GaussPos, 
                                                                       I_A1_IndexPos );
    ndarray::Array< size_t, 1, 1 > I_A1_IndexCheckPos = ndarray::external( indCheck.data(), ndarray::makeVector( int( indCheck.size() ) ), ndarray::makeVector( 1 ) );
    ndarray::Array< double, 1, 1 > D_A1_FittedCheckPos = math::getSubArray( D_A1_GaussPos, 
                                                                            I_A1_IndexCheckPos );
    #ifdef __DEBUG_IDENTIFY__
      cout << "identify: D_A1_FittedPos = " << D_A1_FittedPos << endl;
    #endif

    D_A1_WLen[ ndarray::view() ] = lineList[ ndarray::view()( 0 ) ];
    ndarray::Array< double, 1, 1 > D_A1_FittedWLen = math::getSubArray( D_A1_WLen, I_A1_IndexPos );
    cout << "Identify: found D_A1_FittedWLen = " << D_A1_FittedWLen << endl;

    ndarray::Array< double, 1, 1 > D_A1_FittedWLenCheck = math::getSubArray( D_A1_WLen, I_A1_IndexCheckPos );

    _dispCoeffs = ndarray::allocate( dispCorControl.order + 1 );
    _dispCoeffs.deep() = math::PolyFit( D_A1_FittedPos,
                                        D_A1_FittedWLen,
                                        dispCorControl.order );
    ndarray::Array< double, 1, 1 > D_A1_WLen_Gauss = math::Poly( D_A1_FittedPos, 
                                                                 _dispCoeffs );
    ndarray::Array< double, 1, 1 > D_A1_WLen_GaussCheck = math::Poly( D_A1_FittedCheckPos, 
                                                                      _dispCoeffs );
    cout << "Identify: D_A1_WLen_PolyFit = " << D_A1_WLen_Gauss << endl;
    cout << "identify: _dispCoeffs = " << _dispCoeffs << endl;

    ///Calculate RMS
    ndarray::Array< double, 1, 1 > D_A1_WLenMinusFit = ndarray::allocate( D_A1_WLen_Gauss.getShape()[ 0 ] );
    D_A1_WLenMinusFit.deep() = D_A1_FittedWLen - D_A1_WLen_Gauss;
    cout << "Identify: D_A1_WLenMinusFit = " << D_A1_WLenMinusFit << endl;
    _dispRms = math::calcRMS( D_A1_WLenMinusFit );
    cout << "Identify: _dispRms = " << _dispRms << endl;
    cout << "======================================" << endl;

    ///Calculate RMS for test lines
    ndarray::Array< double, 1, 1 > D_A1_WLenMinusFitCheck = ndarray::allocate( D_A1_WLen_GaussCheck.getShape()[ 0 ] );
    D_A1_WLenMinusFitCheck.deep() = D_A1_FittedWLenCheck - D_A1_WLen_GaussCheck;
    cout << "Identify: D_A1_WLenMinusFitCheck = " << D_A1_WLenMinusFitCheck << endl;
    double dispRmsCheck = math::calcRMS( D_A1_WLenMinusFitCheck );
    cout << "Identify: dispRmsCheck = " << dispRmsCheck << endl;
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
      cout << "Identify: WARNING: Wavelength solution is not monotonic => Setting identifyResult.rms to 1000" << endl;
      _dispRms = 1000.;
      cout << "Identify: RMS = " << _dispRms << endl;
      cout << "======================================" << endl;
    }

    _isWavelengthSet = true;
    return _isWavelengthSet;
}

template< typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT >
template< typename T >
bool pfs::drp::stella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::identify( ndarray::Array< T, 2, 1 > const& lineList,
//                                                                                     ndarray::Array< T, 1, 0 > const& predicted,
                                                                                     ndarray::Array< T, 1, 0 > const& predictedWLenAllPix,
                                                                                     pfs::drp::stella::DispCorControl const& dispCorControl,
                                                                                     size_t nLinesCheck ){
    pfs::drp::stella::DispCorControl tempDispCorControl( dispCorControl );
    _dispCorControl.reset();
    _dispCorControl = tempDispCorControl.getPointer();

    ///for each line in line list, find maximum in spectrum and fit Gaussian
    ndarray::Array< double, 1, 1 > D_A1_GaussPos = hIdentify( lineList );

    ///remove lines which could not be found from line list
    std::vector< int > V_Index( D_A1_GaussPos.getShape()[ 0 ], 0 );
    size_t pos = 0;
    for (auto it = D_A1_GaussPos.begin(); it != D_A1_GaussPos.end(); ++it, ++pos ){
      if ( *it > 0. )
        V_Index[ pos ] = 1;
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
      cout << "Identify: " << nInd << " lines identified" << endl;
      cout << "Identify: indices = ";
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
    ndarray::Array< size_t, 1, 1 > I_A1_IndexPos = ndarray::external( indices.data(), ndarray::makeVector( int( indices.size() ) ), ndarray::makeVector( 1 ) );
    ndarray::Array< double, 1, 1 > fittedPosLinesFound = math::getSubArray( D_A1_GaussPos, 
                                                                            I_A1_IndexPos );
    #ifdef __DEBUG_IDENTIFY__
      cout << "identify: fittedPosLinesFound = " << fittedPosLinesFound << endl;
    #endif

    ndarray::Array< double, 1, 1 > predictedWLen = ndarray::allocate( lineList.getShape()[ 0 ] );
    predictedWLen[ ndarray::view() ] = lineList[ ndarray::view()( 0 ) ];
    ndarray::Array< double, 1, 1 > fittedWLenLinesFound = math::getSubArray( predictedWLen, 
                                                                             I_A1_IndexPos );
    cout << "Identify: found fittedWLenLinesFound = " << fittedWLenLinesFound << endl;

/*    ndarray::Array< double, 1, 1 > predictedPos = ndarray::allocate( predicted.getShape()[ 0 ] );
    predictedPos[ ndarray::view() ] = predicted[ ndarray::view() ];

    ndarray::Array< double, 1, 1 > predictedPosFound = math::getSubArray( predictedPos, 
                                                                          I_A1_IndexPos );

    ndarray::Array< double, 1, 1 > pixOffsetToFit = ndarray::allocate( nInd );
    pixOffsetToFit[ ndarray::view() ] = fittedPosLinesFound[ ndarray::view() ] - predictedPosFound[ ndarray::view() ];

    _dispCoeffs = ndarray::allocate( dispCorControl.order + 1 );
    _dispCoeffs.deep() = math::PolyFit( fittedPosLinesFound,
                                        pixOffsetToFit,
                                        dispCorControl.order );
    ndarray::Array< double, 1, 1 > D_A1_PixOffsetFit = math::Poly( fittedPosLinesFound, 
                                                                   _dispCoeffs );

    ///Interpolate wavelength from predicted wavelengths and measured pixel offset
    ndarray::Array< double, 1, 1 > pixIndex = math::indGenNdArr( double( _length ) );
    std::vector< std::string > interpolKeyWords( 1 );
    interpolKeyWords[ 0 ] = std::string( "SPLINE" );
    ndarray::Array< double, 1, 1 > predictedWLenAllPixA = ndarray::allocate( predictedWLenAllPix.getShape()[ 0 ] );
    predictedWLenAllPixA.deep() = predictedWLenAllPix;
    ndarray::Array< double, 1, 1 > wLenLinesFoundCheck = math::interPol( predictedWLenAllPixA, 
                                                                         pixIndex, 
                                                                         fittedPosLinesFound,
                                                                         interpolKeyWords );
    wLenLinesFoundCheck[ ndarray::view() ] = wLenLinesFoundCheck[ ndarray::view() ] + predictedPosFound[ ndarray::view() ];
    cout << "Identify: wLenLinesFoundCheck = " << wLenLinesFoundCheck << endl;
    cout << "identify: _dispCoeffs = " << _dispCoeffs << endl;

    ///Calculate RMS
    ndarray::Array< double, 1, 1 > D_A1_WLenMinusFit = ndarray::allocate( wLenLinesFoundCheck.getShape()[ 0 ] );
    D_A1_WLenMinusFit.deep() = fittedWLenLinesFound - wLenLinesFoundCheck;
    cout << "Identify: D_A1_WLenMinusFit = " << D_A1_WLenMinusFit << endl;
    _dispRms = math::calcRMS( D_A1_WLenMinusFit );
    cout << "Identify: _dispRms = " << _dispRms << endl;
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
      cout << "Identify: WARNING: Wavelength solution is not monotonic => Setting identifyResult.rms to 1000" << endl;
      _dispRms = 1000.;
      cout << "Identify: RMS = " << _dispRms << endl;
      cout << "======================================" << endl;
    }
    _isWavelengthSet = true;
    return _isWavelengthSet;*/
    return true;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR( pfs::drp::stella::Spectrum< ImageT, MaskT, VarianceT, WavelengthT > ) pfs::drp::stella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getSpectrum( const size_t i ) const 
{
//    cout << "getSpectrum(size_t) const" << endl;
    if (i >= _spectra->size()){
        string message("SpectrumSet::getSpectrum(i=");
        message += to_string(i) + "): ERROR: i >= _spectra->size()=" + to_string(_spectra->size());
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
//    pfs::drp::stella::Spectrum< ImageT, MaskT, VarianceT, WavelengthT > spec( _spectra->at( i ) );
    return PTR( pfs::drp::stella::Spectrum< ImageT, MaskT, VarianceT, WavelengthT > )( new pfs::drp::stella::Spectrum< ImageT, MaskT, VarianceT, WavelengthT >( *( _spectra->at( i ) ) ) );
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR( pfs::drp::stella::Spectrum< ImageT, MaskT, VarianceT, WavelengthT > ) pfs::drp::stella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getSpectrum( const size_t i )
{
 //   cout << "getSpectrum(size_t) NOT CONST" << endl;
    if (i >= _spectra->size()){
        string message("SpectrumSet::getSpectrum(i=");
        message += to_string(i) + "): ERROR: i >= _spectra->size()=" + to_string(_spectra->size());
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return PTR( pfs::drp::stella::Spectrum< ImageT, MaskT, VarianceT, WavelengthT > )( new pfs::drp::stella::Spectrum< ImageT, MaskT, VarianceT, WavelengthT >( *( _spectra->at( i ) ) ) );
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfs::drp::stella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::setSpectrum( size_t const i,
                                                                                           PTR( Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) const& spectrum )
{
  if (i > _spectra->size()){
    string message("SpectrumSet::setSpectrum(i=");
    message += to_string(i) + "): ERROR: i > _spectra->size()=" + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  if ( i == static_cast< int >( _spectra->size() ) ){
    _spectra->push_back( spectrum );
  }
  else{
    ( *_spectra )[ i ] = spectrum;
  }
  return true;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfs::drp::stella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::setSpectrum( size_t const i,
                                                                                           Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> const& spectrum )
{
  if (i > _spectra->size()){
    string message("SpectrumSet::setSpectrum(i=");
    message += to_string(i) + "): ERROR: i > _spectra->size()=" + to_string(_spectra->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  PTR( Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> ) spectrumPtr( new Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT >( spectrum ) );
  
  if ( i == static_cast< int >( _spectra->size() ) ){
    _spectra->push_back(  spectrumPtr );
  }
  else{
    ( *_spectra )[ i ] = spectrumPtr;
  }
  return true;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfs::drp::stella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::erase(const size_t iStart, const size_t iEnd){
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
