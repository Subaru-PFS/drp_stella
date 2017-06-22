template< typename T >
template< typename ImageT, typename MaskT, typename VarianceT >
pfs::drp::stella::ExtractPSFResult<T> pfs::drp::stella::PSF<T>::extractPSFFromCenterPosition(
    pfs::drp::stella::FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace_In,
    T const centerPositionXCCD_In,
    T const centerPositionYCCD_In)
{
    LOG_LOGGER _log = LOG_GET("pfs::drp::stella::PSF::extractPSFFromCenterPosition");
    LOGLS_DEBUG(_log, "PSF::extractPSFFromCenterPosition(FiberTrace, centerPositionXCCD, centerPositionYCCD) started");
    for ( int i = 0; i < fiberTrace_In.getImage()->getHeight(); ++i )
        LOGLS_DEBUG(_log, "fiberTrace_In.getImage()->getArray()[" << i << ",*] = " << fiberTrace_In.getImage()->getArray()[ndarray::view(i)()]);
    LOGLS_DEBUG(_log, "centerPositionXCCD_In = " << centerPositionXCCD_In);
    LOGLS_DEBUG(_log, "centerPositionYCCD_In = " << centerPositionYCCD_In);

    // Calculate x and y position for the center of the PSF to be extracted in the swath
    // of the FiberTrace that contains the PSF
    LOGLS_DEBUG(_log, "_yMin = " << _yMin << ", _yMax = " << _yMax);
    pfs::drp::stella::math::dataXY<T> xyCenterCCD;
    xyCenterCCD.x = centerPositionXCCD_In;
    xyCenterCCD.y = centerPositionYCCD_In;

    // Define output
    ExtractPSFResult< T > result_Out;

    // Calculate offset of our coordinates to the Center of the PSF
    size_t nPix = 0;
    double sumPSF = 0.;
    pfs::drp::stella::math::dataXY<T> psfCoordinates;
    ndarray::Array<ImageT, 2, 1> trace = fiberTrace_In.getImage()->getArray();
    double halfLength = 2. * _twoDPSFControl->yFWHM;
    pfs::drp::stella::math::dataXY<T> xyCenterTrace = pfs::drp::stella::math::ccdToFiberTraceCoordinates(
            xyCenterCCD,
            fiberTrace_In
    );
    int rowMin = xyCenterTrace.y - halfLength;
    if (rowMin < 0)
        rowMin = 0;
    int rowMax = xyCenterTrace.y + halfLength;
    if (rowMax >= fiberTrace_In.getHeight())
        rowMax = fiberTrace_In.getHeight() - 1;
    for (size_t iY = rowMin; iY <= rowMax; ++iY){
        for (size_t iX = 0; iX < fiberTrace_In.getWidth(); ++iX){
            LOGLS_DEBUG(_log, "iX = " << iX << ": iY = " << iY);
            psfCoordinates.x = iX;
            psfCoordinates.y = iY;
            pfs::drp::stella::math::dataXY<T> relativeToCenter = pfs::drp::stella::math::fiberTraceCoordinatesRelativeTo(
                    psfCoordinates,
                    xyCenterCCD,
                    fiberTrace_In
            );
            result_Out.xRelativeToCenter.push_back(relativeToCenter.x);
            result_Out.yRelativeToCenter.push_back(relativeToCenter.y);
            T zTrace = trace[ndarray::makeVector(int(iY), int(iX))];
            result_Out.zNormalized.push_back(zTrace);
            result_Out.zTrace.push_back(zTrace);
            result_Out.weight.push_back(std::fabs(zTrace) > 0.000001 ? T(1. / sqrt(std::fabs(zTrace))) : 0.1);
            result_Out.xTrace.push_back( T ( iX ) );
            result_Out.yTrace.push_back( T ( iY ) );
            string message("x = ");
            message += to_string(_imagePSF_XRelativeToCenter[nPix]) + ", y = ";
            message += to_string(_imagePSF_YRelativeToCenter[nPix]) + ": val = " + to_string(zTrace);
            message += " = " + to_string(_imagePSF_ZNormalized[nPix]) + "; XOrig = ";
            message += to_string(_imagePSF_XTrace[nPix]) + ", YOrig = " + to_string(_imagePSF_YTrace[nPix]);
            LOGLS_DEBUG(_log, message);
            ++nPix;
            sumPSF += zTrace;
            LOGLS_DEBUG(_log, "nPix = " << nPix << ", sumPSF = " << sumPSF);
        }
    }
    result_Out.xCenterPSFCCD = T( centerPositionXCCD_In );
    result_Out.yCenterPSFCCD = T( centerPositionYCCD_In );
    if ( std::fabs( sumPSF ) < 0.00000001 ){
        string message("PSF::extractPSFs: ERROR: sumPSF == 0");
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
    }
    int pixelNo = 0;
    for ( auto iter = result_Out.zNormalized.begin(); iter != result_Out.zNormalized.end(); ++iter ){
        string message("result_Out.zNormalized[pixelNo=");
        message += to_string(pixelNo) + "] = " + to_string(result_Out.zNormalized[ pixelNo ]);
        message += ", sumPSF = " + to_string(sumPSF);
        LOGLS_DEBUG(_log, message);
        *iter = *iter / sumPSF;
        message = "result_Out.zNormalized[pixelNo=";
        message += to_string(pixelNo) + "] = " + to_string(result_Out.zNormalized[pixelNo]);
        LOGLS_DEBUG(_log, message);
        ++pixelNo;
    }
    LOGLS_DEBUG(_log, "PSF::extractPSFFromCenterPosition(FiberTrace, centerPositionXCCD, centerPositionYCCD) finished");
    return result_Out;
}

template< typename T > template< typename ImageT, typename MaskT, typename VarianceT >
bool pfs::drp::stella::PSF< T >::extractPSFFromCenterPositions(
    pfs::drp::stella::FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace_In,
    ndarray::Array< T, 1, 1 > const& centerPositionsXCCD_In,
    ndarray::Array< T, 1, 1 > const& centerPositionsYCCD_In )
{
    LOG_LOGGER _log = LOG_GET("pfs::drp::stella::PSF::extractPSFFromCenterPositions");
    LOGLS_DEBUG(_log, "PSF::extractPSFFromCenterPositions(FiberTrace, centerPositionXCCDArray, centerPositionYCCDArray) started");
    if (centerPositionsXCCD_In.getShape()[ 0 ] != centerPositionsYCCD_In.getShape()[ 0 ]){
      string message("PSF::extractPSFFromCenterPositions: ERROR: centerPositionsXCCD_In.getShape()[0]=");
      message += to_string( centerPositionsXCCD_In.getShape()[ 0 ] );
      message += " != centerPositionsYCCD_In.getShape()[0]=";
      message += to_string(centerPositionsYCCD_In.getShape()[0]);
      throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
    }
    _imagePSF_XRelativeToCenter.resize( 0 );
    _imagePSF_YRelativeToCenter.resize( 0 );
    _imagePSF_XTrace.resize( 0 );
    _imagePSF_YTrace.resize( 0 );
    _imagePSF_ZNormalized.resize( 0 );
    _imagePSF_ZTrace.resize( 0 );
    _imagePSF_Weight.resize( 0 );
    _xCentersPSFCCD.resize( 0 );
    _yCentersPSFCCD.resize( 0 );
    _nPixPerPSF.resize( 0 );

    _imagePSF_XRelativeToCenter.reserve( 1000 );
    _imagePSF_YRelativeToCenter.reserve( 1000 );
    _imagePSF_XTrace.reserve( 1000 );
    _imagePSF_YTrace.reserve( 1000 );
    _imagePSF_ZNormalized.reserve( 1000 );
    _imagePSF_ZTrace.reserve( 1000 );
    _imagePSF_Weight.reserve( 1000 );
    for ( int iPSF = 0; iPSF < centerPositionsXCCD_In.getShape()[ 0 ]; ++iPSF ){
        ExtractPSFResult< T > result = extractPSFFromCenterPosition( fiberTrace_In,
                                                                     centerPositionsXCCD_In[ iPSF ],
                                                                     centerPositionsYCCD_In[ iPSF ] );
        for ( int iPix = 0; iPix < result.xRelativeToCenter.size(); ++iPix ){
            _imagePSF_XRelativeToCenter.push_back( result.xRelativeToCenter[ iPix ] );
            _imagePSF_YRelativeToCenter.push_back( result.yRelativeToCenter[ iPix ] );
            _imagePSF_ZNormalized.push_back( result.zNormalized[ iPix ] );
            _imagePSF_ZTrace.push_back( result.zTrace[ iPix ] );
            _imagePSF_Weight.push_back( result.weight[ iPix ] );
            _imagePSF_XTrace.push_back( result.xTrace[ iPix ] );
            _imagePSF_YTrace.push_back( result.yTrace[ iPix ] );
        }
        _xCentersPSFCCD.push_back( result.xCenterPSFCCD );
        _yCentersPSFCCD.push_back( result.yCenterPSFCCD );
        _nPixPerPSF.push_back( result.xRelativeToCenter.size() );
        LOGLS_DEBUG(_log, "_nPixPerPSF[" << _nPixPerPSF.size()-1 << "] = " << _nPixPerPSF[_nPixPerPSF.size()-1]);
        if ( _nPixPerPSF[ _nPixPerPSF.size() - 1 ] == 0 ){
            string message( "PSF trace" );
            message += to_string(_iTrace) + " bin" + to_string( _iBin ) + "::extractPSFFromCenterPositions: ";
            message += "ERROR: _nPixPerPSF[_nPixPerPSF.size()-1=" + to_string(_nPixPerPSF.size()-1);
            message += "]=" + to_string(_nPixPerPSF[_nPixPerPSF.size()-1]) + " == 0";
            throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
        }
    }
    LOGLS_DEBUG(_log, "PSF::extractPSFFromCenterPosition(FiberTrace, centerPositionXCCDArray, centerPositionYCCDArray) finished");
    return true;
}

template< typename T > template< typename ImageT, typename MaskT, typename VarianceT >
bool pfs::drp::stella::PSF< T >::extractPSFFromCenterPositions(
    pfs::drp::stella::FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace_In)
{
    LOG_LOGGER _log = LOG_GET("pfs::drp::stella::PSF::extractPSFFromCenterPositions");
    LOGLS_DEBUG(_log, "PSF::extractPSFFromCenterPositions(FiberTrace) started");
    ndarray::Array< T, 1, 1 > centerPositionsX = ndarray::allocate( _xCentersPSFCCD.size() );
    ndarray::Array< T, 1, 1 > centerPositionsY = ndarray::allocate( _yCentersPSFCCD.size() );
    auto itXVec = _xCentersPSFCCD.begin();
    auto itYVec = _yCentersPSFCCD.begin();
    for ( auto itXArr = centerPositionsX.begin(), itYArr = centerPositionsY.begin(); itXArr != centerPositionsX.end(); ++itXArr, ++itYArr, ++itXVec, ++itYVec ){
        *itXArr = *itXVec;
        *itYArr = *itYVec;
    }
    _imagePSF_XRelativeToCenter.resize( 0 );
    _imagePSF_YRelativeToCenter.resize( 0 );
    _imagePSF_XTrace.resize( 0 );
    _imagePSF_YTrace.resize( 0 );
    _imagePSF_ZNormalized.resize( 0 );
    _imagePSF_ZTrace.resize( 0 );
    _imagePSF_Weight.resize( 0 );
    _xCentersPSFCCD.resize( 0 );
    _yCentersPSFCCD.resize( 0 );
    _nPixPerPSF.resize( 0 );

    _imagePSF_XRelativeToCenter.reserve( 1000 );
    _imagePSF_YRelativeToCenter.reserve( 1000 );
    _imagePSF_XTrace.reserve( 1000 );
    _imagePSF_YTrace.reserve( 1000 );
    _imagePSF_ZNormalized.reserve( 1000 );
    _imagePSF_ZTrace.reserve( 1000 );
    _imagePSF_Weight.reserve( 1000 );
    for ( int iPSF = 0; iPSF < centerPositionsX.getShape()[0]; ++iPSF ){
        ExtractPSFResult< T > result = extractPSFFromCenterPosition( fiberTrace_In,
                                                                     centerPositionsX[ iPSF ],
                                                                     centerPositionsY[ iPSF ]);
        for ( int iPix = 0; iPix < result.xRelativeToCenter.size(); ++iPix ){
            _imagePSF_XRelativeToCenter.push_back( result.xRelativeToCenter[ iPix ] );
            _imagePSF_YRelativeToCenter.push_back( result.yRelativeToCenter[ iPix ] );
            _imagePSF_ZNormalized.push_back( result.zNormalized[ iPix ] );
            _imagePSF_ZTrace.push_back( result.zTrace[ iPix ] );
            _imagePSF_Weight.push_back( result.weight[ iPix ] );
            _imagePSF_XTrace.push_back( result.xTrace[ iPix ] );
            _imagePSF_YTrace.push_back( result.yTrace[ iPix ] );
        }
        _xCentersPSFCCD.push_back( result.xCenterPSFCCD );
        _yCentersPSFCCD.push_back( result.yCenterPSFCCD );
        _nPixPerPSF.push_back( result.xRelativeToCenter.size() );
        if ( _nPixPerPSF[ _nPixPerPSF.size() - 1 ] == 0 ){
            string message( "PSF trace" );
            message += to_string(_iTrace) + " bin" + to_string( _iBin ) + "::extractPSFFromCenterPositionsA: ERROR: _nPixPerPSF[_nPixPerPSF.size()-1=" + to_string(_nPixPerPSF.size()-1);
            message += "]=" + to_string(_nPixPerPSF[_nPixPerPSF.size()-1]) + " == 0";
            throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
        }
    }
    LOGLS_DEBUG(_log, "PSF::extractPSFFromCenterPositions(FiberTrace) finished");
    return true;
}

template<typename T> template< typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfs::drp::stella::PSF<T>::extractPSFs(
    pfs::drp::stella::FiberTrace<ImageT, MaskT, VarianceT> const& fiberTraceIn,
    pfs::drp::stella::Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrumIn)
{
  #ifdef __DEBUG_PSF__
    cout << "PSF::extractPSFs(FiberTrace, Spectrum) started" << endl;
  #endif
  ndarray::Array<T, 2, 1> collapsedPSF = ndarray::allocate(1, 1);
  return extractPSFs(fiberTraceIn, spectrumIn, collapsedPSF);
}

template< typename T > template< typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT >
bool pfs::drp::stella::PSF< T >::extractPSFs(
    pfs::drp::stella::FiberTrace< ImageT, MaskT, VarianceT > const& fiberTraceIn,
    pfs::drp::stella::Spectrum< ImageT, MaskT, VarianceT, WavelengthT> const& spectrumIn,
    ndarray::Array< T, 2, 1 > const& collapsedPSF)
{
  #ifdef __DEBUG_PSF__
    cout << "PSF::extractPSFs(FiberTrace, Spectrum, collapsedPSF) started" << endl;
  #endif
  if (! _isTwoDPSFControlSet ){
    string message("PSF trace");
    message += to_string(_iTrace) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: _twoDPSFControl is not set";
    throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
  }
  if ( fiberTraceIn.getHeight() < _yMax ){
    string message("PSF trace");
    message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: fiberTraceIn.getHeight(=" + to_string( fiberTraceIn.getHeight() );
    message += " < _yMax = " + to_string( _yMax );
    throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
  }
  #ifdef __DEBUG_CALC2DPSF__
    cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: collapsedPSF = " << collapsedPSF << endl;
    cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: spectrumIn.getSpectrum() = " << spectrumIn.getSpectrum() << endl;
  #endif

  ndarray::Array< float, 1, 1 > xCentersTrace = copy( fiberTraceIn.getXCenters() );
  ndarray::Array< double, 1, 1 > spectrumSwath = ndarray::allocate( _yMax - _yMin + 1);
  ndarray::Array< double, 1, 1 > spectrumVarianceSwath = ndarray::allocate( _yMax - _yMin + 1 );

  #ifdef __DEBUG_CALC2DPSF__
    ndarray::Array< double, 2, 1 > imageTrace = pfs::drp::stella::math::Double( fiberTraceIn.getImage()->getArray() );
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: imageTrace.getShape() = " << imageTrace.getShape() << ", _yMin = " << _yMin << ", _yMax = " << _yMax << endl;
    ndarray::Array< double, 2, 1 > imageSwath = ndarray::allocate( _yMax - _yMin + 1, fiberTraceIn.getImage()->getWidth() );
    imageSwath = imageTrace[ ndarray::view( _yMin, _yMax + 1 )() ];
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: traceIn = " << imageSwath << endl;

    ndarray::Array< double, 2, 1 > stddevSwath = ndarray::allocate( _yMax - _yMin + 1, fiberTraceIn.getImage()->getWidth() );
    ndarray::Array< double, 2, 1 > stddevTrace = pfs::drp::stella::math::Double( fiberTraceIn.getVariance()->getArray() );
    for (auto itRowSDA = stddevTrace.begin() + _yMin, itRowSD = stddevSwath.begin(); itRowSDA != stddevTrace.begin() + _yMax + 1; ++itRowSDA, ++itRowSD){
      for (auto itColSDA = itRowSDA->begin(), itColSD = itRowSD->begin(); itColSDA != itRowSDA->end(); ++itColSDA, ++itColSD){
        *itColSD = sqrt( *itColSDA > 0. ? *itColSDA : 1. );
      }
    }
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: stddevTrace = " << stddevTrace << endl;
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: stddevSwath = " << stddevSwath << endl;

    int i, j;
    if ( std::fabs( stddevSwath.asEigen().maxCoeff( &i, &j ) ) < 0.000001 ){
      double D_MaxStdDev = std::fabs( stddevSwath[ ndarray::makeVector( i, j ) ] );
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: std::fabs(max(stddevSwath))=" + to_string( D_MaxStdDev ) + " < 0.000001";
      throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
    }
    ndarray::Array< double, 2, 1 > maskSwath = ndarray::allocate( _yMax - _yMin + 1, fiberTraceIn.getImage()->getWidth() );
    auto itMRow = maskSwath.begin();
    i = 0;
    for ( auto itRow = fiberTraceIn.getMask()->getArray().begin() + _yMin; itRow <= fiberTraceIn.getMask()->getArray().begin() + _yMax; ++itRow, ++itMRow ){
      auto itMCol = itMRow->begin();
      j = 0;
      for ( auto itCol = itRow->begin(); itCol != itRow->end(); ++itCol, ++itMCol ){
        *itMCol = *itCol == 0 ? 1 : 0;
        cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: row " << i << ", col " << j << ": *itCol = " << *itCol << " => *itMCol set to " << *itMCol << endl;
        ++j;
      }
      ++i;
    }
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: maskSwath = " << maskSwath << endl;
  #endif

  spectrumSwath = pfs::drp::stella::math::Double( ndarray::Array< ImageT, 1, 1 >( spectrumIn.getSpectrum()[ ndarray::view( _yMin, _yMax + 1 ) ] ) );
  #ifdef __DEBUG_CALC2DPSF__
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: spectrumSwath = " << spectrumSwath << endl;
  #endif

  spectrumVarianceSwath = pfs::drp::stella::math::Double( ndarray::Array< VarianceT, 1, 1 >( spectrumIn.getVariance()[ ndarray::view( _yMin, _yMax + 1 ) ] ) );
  #ifdef __DEBUG_CALC2DPSF__
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: spectrumVarianceSwath = " << spectrumVarianceSwath << endl;
  #endif

  ndarray::Array< double, 1, 1 > spectrumSigmaSwath = ndarray::allocate( spectrumVarianceSwath.size() );
  spectrumSigmaSwath.asEigen() = Eigen::Array< double, Eigen::Dynamic, 1 >( spectrumVarianceSwath.asEigen() ).sqrt();
  #ifdef __DEBUG_CALC2DPSF__
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: spectrumSigmaSwath = " << spectrumSigmaSwath << endl;
  #endif

  ndarray::Array< float, 1, 1 > xCentersSwath = ndarray::copy( xCentersTrace[ ndarray::view( _yMin, _yMax + 1 ) ] );
  ndarray::Array< float, 1, 1 > xCentersSwathTemp = ndarray::copy(xCentersSwath);
  xCentersSwathTemp.deep() += 0.5 - PIXEL_CENTER;
  ndarray::Array< float, 1, 1 > xCentersSwathFloor = pfs::drp::stella::math::floor(xCentersSwathTemp, float(0));
  #ifdef __DEBUG_CALC2DPSF__
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: xCentersSwath = " << xCentersSwath << endl;
  #endif

  ndarray::Array< float, 1, 1 > xCentersSwathOffset = ndarray::copy( xCentersSwath );
  xCentersSwathOffset.deep() -= xCentersSwathFloor;
  #ifdef __DEBUG_CALC2DPSF__
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: xCentersSwathOffset = " << xCentersSwathOffset << endl;
  #endif

  _imagePSF_XRelativeToCenter.resize(0);
  _imagePSF_YRelativeToCenter.resize(0);
  _imagePSF_XTrace.resize(0);
  _imagePSF_YTrace.resize(0);
  _imagePSF_ZNormalized.resize(0);
  _imagePSF_ZTrace.resize(0);
  _imagePSF_Weight.resize(0);
  _xCentersPSFCCD.resize(0);
  _yCentersPSFCCD.resize(0);
  _nPixPerPSF.resize(0);

  _imagePSF_XRelativeToCenter.reserve(1000);
  _imagePSF_YRelativeToCenter.reserve(1000);
  _imagePSF_XTrace.reserve(1000);
  _imagePSF_YTrace.reserve(1000);
  _imagePSF_ZNormalized.reserve(1000);
  _imagePSF_ZTrace.reserve(1000);
  _imagePSF_Weight.reserve(1000);
  size_t nPix = 0;

  /// Find emission lines
  /// set everything below signal threshold to zero
  ndarray::Array< int, 1, 1 > spectrumSwathZeroBelowThreshold = ndarray::allocate( spectrumSwath.size() );
  auto itSpec = spectrumSwath.begin();
  for (auto itSig = spectrumSwathZeroBelowThreshold.begin(); itSig != spectrumSwathZeroBelowThreshold.end(); ++itSig, ++itSpec){
    *itSig = *itSpec < _twoDPSFControl->signalThreshold ? 0. : *itSpec;
  }
  #ifdef __DEBUG_CALC2DPSF__
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: 1. spectrumSwathZeroBelowThreshold = " << spectrumSwathZeroBelowThreshold << endl;
  #endif

  /// look for signal wider than 2 FWHM
  if ( !pfs::drp::stella::math::countPixGTZero( spectrumSwathZeroBelowThreshold ) ){
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: ERROR: CountPixGTZero(spectrumSwathZeroBelowThreshold=" << spectrumSwathZeroBelowThreshold << ") returned FALSE" << endl;
    string message( "PSF trace" );
    message += to_string(_iTrace) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: CountPixGTZero(spectrumSwathZeroBelowThreshold) returned FALSE";
    throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
  }
  #ifdef __DEBUG_CALC2DPSF__
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: 2. spectrumSwathZeroBelowThreshold = " << spectrumSwathZeroBelowThreshold << endl;
  #endif

  /// identify emission lines
  int firstWideSignalSwath, firstWideSignalSwathStart, firstWideSignalSwathEnd;
  int minWidth = int( _twoDPSFControl->yFWHM ) + 1;
  int startIndexSwath = 0;
  double maxTimesFWHM = 3.5;
  ndarray::Array< int, 2, 1 > I_A2_Limited = ndarray::allocate( _twoDPSFControl->nTermsGaussFit, 2 );
  I_A2_Limited[ ndarray::view()() ] = 1;
  if ( _twoDPSFControl->nTermsGaussFit == 5 )
    I_A2_Limited[ ndarray::view(4)() ] = 0;
  ndarray::Array< double, 2, 1 > D_A2_Limits = ndarray::allocate( _twoDPSFControl->nTermsGaussFit, 2 );
  /// 0: peak value
  /// 1: center position
  /// 2: sigma
  /// 3: constant background
  /// 4: linear background
  D_A2_Limits[ ndarray::makeVector( 0, 0 ) ] = _twoDPSFControl->signalThreshold;
  D_A2_Limits[ ndarray::makeVector( 2, 0 ) ] = _twoDPSFControl->yFWHM / ( 2. * 2.3548 );
  D_A2_Limits[ ndarray::makeVector( 2, 1 ) ] = _twoDPSFControl->yFWHM;
  if ( _twoDPSFControl->nTermsGaussFit > 3 )
    D_A2_Limits[ ndarray::makeVector( 3, 0 ) ] = 0.;
  if ( _twoDPSFControl->nTermsGaussFit > 4 ){
    D_A2_Limits[ ndarray::makeVector( 4, 0 ) ] = 0.;
    D_A2_Limits[ ndarray::makeVector( 4, 1 ) ] = 1000.;
  }
  ndarray::Array< double, 1, 1 > D_A1_GaussFit_Coeffs = ndarray::allocate( _twoDPSFControl->nTermsGaussFit );
  ndarray::Array< double, 1, 1 > D_A1_GaussFit_ECoeffs = ndarray::allocate( _twoDPSFControl->nTermsGaussFit );

  ndarray::Array< double, 1, 1 > D_A1_Guess = ndarray::allocate( _twoDPSFControl->nTermsGaussFit );
//    float gaussCenterX;
  double gaussCenterY;
  int emissionLineNumber = 0;
  do{
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: starting firstIndexWithValueGEFrom: spectrumSwathZeroBelowThreshold.getShape()[0] = " << spectrumSwathZeroBelowThreshold.getShape()[ 0 ] << ", startIndexSwath = " << startIndexSwath << endl;
    #endif
    firstWideSignalSwath = pfs::drp::stella::math::firstIndexWithValueGEFrom( spectrumSwathZeroBelowThreshold, minWidth, startIndexSwath );
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: firstWideSignalSwath found at index " << firstWideSignalSwath << ", startIndexSwath = " << startIndexSwath << endl;
    #endif
    if ( firstWideSignalSwath < 0 ){
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: No emission line found" << endl;
      break;
    }
    firstWideSignalSwathStart = pfs::drp::stella::math::lastIndexWithZeroValueBefore( spectrumSwathZeroBelowThreshold, firstWideSignalSwath );
    if ( firstWideSignalSwathStart < 0 ){
      firstWideSignalSwathStart = 0;
    }
    else if (firstWideSignalSwathStart < startIndexSwath){
        firstWideSignalSwathStart = startIndexSwath;
    }
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: firstWideSignalSwathStart = " << firstWideSignalSwathStart << endl;
    #endif

    firstWideSignalSwathEnd = pfs::drp::stella::math::firstIndexWithZeroValueFrom( spectrumSwathZeroBelowThreshold, firstWideSignalSwath );
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: firstWideSignalSwathEnd = " << firstWideSignalSwathEnd << endl;
    #endif

    if ( firstWideSignalSwathEnd < 0 ){
      firstWideSignalSwathEnd = spectrumSwathZeroBelowThreshold.size()-1;
    }
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: End of first wide signal found at index " << firstWideSignalSwathEnd << endl;
    #endif

    if ( ( firstWideSignalSwathEnd - firstWideSignalSwathStart + 1 ) > ( _twoDPSFControl->yFWHM * maxTimesFWHM ) ){
      firstWideSignalSwathEnd = firstWideSignalSwathStart + int( maxTimesFWHM * _twoDPSFControl->yFWHM );
    }

    /// Set start index for next run
    startIndexSwath = firstWideSignalSwathEnd + 1;

    /// Fit Gaussian and Trace Aperture
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: End of first wide signal found at index " << firstWideSignalSwathEnd << endl;
    #endif

    if ( pfs::drp::stella::math::max( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) ) < _twoDPSFControl->saturationLevel ){
      D_A2_Limits[ ndarray::makeVector( 0, 1 )  ] = 1.5 * pfs::drp::stella::math::max( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) );
      D_A2_Limits[ ndarray::makeVector( 1, 0 ) ] = firstWideSignalSwathStart;
      D_A2_Limits[ ndarray::makeVector( 1, 1 ) ] = firstWideSignalSwathEnd;
      if ( _twoDPSFControl->nTermsGaussFit > 3 )
        D_A2_Limits[ ndarray::makeVector( 3, 1 ) ] = pfs::drp::stella::math::min( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) );

      D_A1_Guess[ 0 ] = pfs::drp::stella::math::max( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) );
      D_A1_Guess[ 1 ] = firstWideSignalSwathStart + ( pfs::drp::stella::math::maxIndex( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) ) );
      D_A1_Guess[ 2 ] = _twoDPSFControl->yFWHM / 2.3548;
      if ( _twoDPSFControl->nTermsGaussFit > 3 )
        D_A1_Guess[ 3 ] = pfs::drp::stella::math::min( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) );
      if ( _twoDPSFControl->nTermsGaussFit > 4 )
        D_A1_Guess[ 4 ] = ( spectrumSwath( firstWideSignalSwathEnd ) - spectrumSwath( firstWideSignalSwathStart ) ) / ( firstWideSignalSwathEnd - firstWideSignalSwathStart );

      ndarray::Array< double, 1, 1 > xSwathGaussFit = pfs::drp::stella::math::indGenNdArr( double( firstWideSignalSwathEnd - firstWideSignalSwathStart + 1 ) );
      xSwathGaussFit[ ndarray::view() ] += firstWideSignalSwathStart;

      ndarray::Array< double, 1, 1 > ySwathGaussFit = ndarray::allocate( firstWideSignalSwathEnd - firstWideSignalSwathStart + 1 );
      for ( int ooo = 0; ooo < firstWideSignalSwathEnd - firstWideSignalSwathStart + 1; ++ooo ){
        ySwathGaussFit[ ooo ] = spectrumSwath( firstWideSignalSwathStart + ooo );
        if ( ySwathGaussFit[ ooo ] < 0.01 )
          ySwathGaussFit[ ooo ] = 0.01;
      }
      ndarray::Array< double, 1, 1 > stdDevGaussFit = ndarray::allocate( firstWideSignalSwathEnd - firstWideSignalSwathStart + 1 );
      for ( int ooo = 0; ooo < firstWideSignalSwathEnd - firstWideSignalSwathStart + 1; ++ooo ){
        if (std::fabs(spectrumSigmaSwath(firstWideSignalSwathStart + ooo)) < 0.1)
          stdDevGaussFit[ ooo ] = 0.1;
        else
          stdDevGaussFit[ ooo ] = spectrumSigmaSwath( firstWideSignalSwathStart + ooo );
      }
      #ifdef __DEBUG_CALC2DPSF__
        cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: emissionLineNumber = " << emissionLineNumber << ": Starting GaussFit: xSwathGaussFit=" << xSwathGaussFit << ", ySwathGaussFit = " << ySwathGaussFit << ", stdDevGaussFit = " << stdDevGaussFit << ", D_A1_Guess = " << D_A1_Guess << ", I_A2_Limited = " << I_A2_Limited << ", D_A2_Limits = " << D_A2_Limits << endl;
      #endif
      int iBackground = 0;
      if ( _twoDPSFControl->nTermsGaussFit == 4 )
        iBackground = 1;
      else if ( _twoDPSFControl->nTermsGaussFit == 5 )
        iBackground = 2;
      bool success;
      success = MPFitGaussLim(xSwathGaussFit,
                ySwathGaussFit,
                stdDevGaussFit,
                D_A1_Guess,
                I_A2_Limited,
                D_A2_Limits,
                iBackground,
                false,
                D_A1_GaussFit_Coeffs,
                D_A1_GaussFit_ECoeffs );
      if ( !success ){
        #ifdef __DEBUG_CALC2DPSF__
          cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: WARNING: Fit FAILED" << endl;
        #endif
      }
      else{
        success = false;
        #ifdef __DEBUG_CALC2DPSF__
          cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
        #endif
        if ( ( D_A1_GaussFit_Coeffs[ 2 ] < ( _twoDPSFControl->yFWHM / 1.5 ) ) &&
             ( ( _twoDPSFControl->nTermsGaussFit < 5 ) ||
               ( (_twoDPSFControl->nTermsGaussFit > 4 ) && ( D_A1_GaussFit_Coeffs[ 4 ] < 1000. ) ) ) ){
          ++emissionLineNumber;
          gaussCenterY = D_A1_GaussFit_Coeffs[ 1 ] + PIXEL_CENTER;
          success = true;
        }
        else{
          cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: D_A1_GaussFit_Coeffs(2)(=" << D_A1_GaussFit_Coeffs[ 2 ] << ") >= (_twoDPSFControl->yFWHM / 1.5)(=" << (_twoDPSFControl->yFWHM / 1.5) << ") || ((_twoDPSFControl->nTermsGaussFit(=" << _twoDPSFControl->nTermsGaussFit << ") < 5) || ((_twoDPSFControl->nTermsGaussFit > 4) && (D_A1_GaussFit_Coeffs(4)(=" << D_A1_GaussFit_Coeffs[4] << ") >= 1000.)) => Skipping emission line" << endl;
        }
      }
      if ( D_A1_GaussFit_Coeffs[ 1 ] < ( 2. * _twoDPSFControl->yFWHM ) ){
        cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: D_A1_GaussFit_Coeffs(2)(=" << D_A1_GaussFit_Coeffs[ 2 ] << ") < 2*yFWHM -> spot not fully sampled" << endl;
        success = false;
      }
      if ( success && ( collapsedPSF.getShape()[ 0 ] > 2 ) ){
        ndarray::Array< double, 2, 1 > indexRelToCenter = pfs::drp::stella::math::calcPositionsRelativeToCenter( double( gaussCenterY ), double( 4. * _twoDPSFControl->yFWHM ) );
        #ifdef __DEBUG_CALC2DPSF__
          cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: indexRelToCenter = " << indexRelToCenter << endl;
        #endif
        if ( indexRelToCenter[ ndarray::makeVector<size_t>( indexRelToCenter.getShape()[ 0 ] - 1, 0 ) ] < spectrumSwath.getShape()[ 0 ] ){
          cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": indexRelToCenter[indexRelToCenter.getShape()[0]=" << indexRelToCenter.getShape()[0] << "][0](=" << indexRelToCenter[ ndarray::makeVector<size_t>( indexRelToCenter.getShape()[ 0 ], 0 ) ] << " < spectrumSwath.getShape()[0] = " << spectrumSwath.getShape()[0] << endl;
          ndarray::Array< double, 2, 1 > arrA = ndarray::allocate( indexRelToCenter.getShape()[ 0 ], 2 );
          arrA[ ndarray::view()( 0 )].deep() = indexRelToCenter[ ndarray::view()( 1 ) ];
          ndarray::Array< size_t, 1, 1 > indVec = ndarray::allocate( indexRelToCenter.getShape()[ 0 ] );
          for (int iInd = 0; iInd < indexRelToCenter.getShape()[ 0 ]; ++iInd)
            indVec[ iInd ] = size_t( indexRelToCenter[ ndarray::makeVector( iInd, 0 ) ] );
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: indVec = " << indVec << endl;
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: spectrumSwath = " << spectrumSwath.getShape() << ": " << spectrumSwath << endl;
          #endif
          ndarray::Array< double, 1, 1 > yDataVec = pfs::drp::stella::math::getSubArray( spectrumSwath, indVec );
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: ySwathGaussFit = " << ySwathGaussFit << ": indVec = " << indVec << endl;
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: yDataVec = " << yDataVec << endl;
          #endif
          arrA[ ndarray::view()( 1 ) ].deep() = yDataVec;//yDataVec;

          ndarray::Array< double, 1, 1 > range = ndarray::allocate(2);
          range[ 0 ] = _twoDPSFControl->xCorRangeLowLimit;
          range[ 1 ] = _twoDPSFControl->xCorRangeHighLimit;
          const double stepSize = _twoDPSFControl->xCorStepSize;

          double maxA = yDataVec.asEigen().maxCoeff();
          ndarray::Array< double, 1, 1 > yValCollapsedPSF = ndarray::allocate( collapsedPSF.getShape()[ 0 ] );
          yValCollapsedPSF.deep() = collapsedPSF[ ndarray::view()( 1 ) ];
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: yValCollapsedPSF = " << yValCollapsedPSF << endl;
          #endif
          double maxCollapsedPSF = yValCollapsedPSF.asEigen().maxCoeff();
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: maxA = " << maxA << ", maxCollapsedPSF = " << maxCollapsedPSF << ", collapsedPSF = " << collapsedPSF << endl;
          #endif
          collapsedPSF[ ndarray::view()( 1 ) ].deep() = collapsedPSF[ ndarray::view()( 1 )] * maxA / maxCollapsedPSF;

          double xCorMinPos = pfs::drp::stella::math::xCor( arrA,/// x must be 'y' relative to center
                                                            collapsedPSF,/// x is 'y' relative to center
                                                            range,
                                                            stepSize);
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": gaussCenterY = " << gaussCenterY << endl;
          #endif
          gaussCenterY += xCorMinPos;
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": xCorMinPos = " << xCorMinPos << ": gaussCenterY = " << gaussCenterY << endl;
          #endif
        }
        else{
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber - 1 << ": WARNING indexRelToCenter[indexRelToCenter.getShape()[0]][0](=" << indexRelToCenter[ ndarray::makeVector<size_t>( indexRelToCenter.getShape()[0], 0 ) ] << " >= spectrumSwath.getShape()[0] = " << spectrumSwath.getShape()[0] << endl;
          #endif
          success = false;
        }
      }
      if ( !success ){
        #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace " << _iTrace << " bin " << _iBin << "::extractPSFs: WARNING: MPFitGaussLim failed" << endl;
        #endif
      }
      else{
        const T yCenterCCD(gaussCenterY + fiberTraceIn.getFiberTraceFunction()->yCenter + fiberTraceIn.getFiberTraceFunction()->yLow + _yMin);
        ndarray::Array< float, 1, 1 > yCenterFromGaussCenter = ndarray::allocate( 1 );
        yCenterFromGaussCenter[0] = yCenterCCD;
        ndarray::Array< float, 1, 1 > xCenterCCDFromYCenterCCD = pfs::drp::stella::math::calculateXCenters( fiberTraceIn.getFiberTraceFunction(),
                                                                                                            yCenterFromGaussCenter );
        const T xTemp(xCenterCCDFromYCenterCCD[0]);
        ExtractPSFResult< T > result = extractPSFFromCenterPosition( fiberTraceIn,
                                                                     xTemp,
                                                                     yCenterCCD );
        if (result.zTrace.size() > 0){
          for (size_t iPix = 0; iPix < result.xRelativeToCenter.size(); ++iPix){
            _imagePSF_XRelativeToCenter.push_back( result.xRelativeToCenter[ iPix ] );
            _imagePSF_YRelativeToCenter.push_back( result.yRelativeToCenter[ iPix ] );
            _imagePSF_ZNormalized.push_back( result.zNormalized[ iPix ] );
            _imagePSF_ZTrace.push_back( result.zTrace[ iPix ] );
            _imagePSF_Weight.push_back( result.weight[ iPix ] );
            _imagePSF_XTrace.push_back( result.xTrace[ iPix ] );
            _imagePSF_YTrace.push_back( result.yTrace[ iPix ] );
          }
          _xCentersPSFCCD.push_back( result.xCenterPSFCCD );
          _yCentersPSFCCD.push_back( result.yCenterPSFCCD );
          _nPixPerPSF.push_back( result.xRelativeToCenter.size() );
          #ifdef __DEBUG_CALC2DPSF__
              cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: _nPixPerPSF[" << _nPixPerPSF.size()-1 << "] = " << _nPixPerPSF[_nPixPerPSF.size()-1] << endl;
          #endif
          if ( _nPixPerPSF[ _nPixPerPSF.size() - 1 ] == 0 ){
            string message( "PSF trace" );
            message += to_string(_iTrace) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: _nPixPerPSF[_nPixPerPSF.size()-1=" + to_string(_nPixPerPSF.size()-1);
            message += "]=" + to_string(_nPixPerPSF[_nPixPerPSF.size()-1]) + " == 0";
            throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
          }
          nPix += result.xRelativeToCenter.size();
        }
        else{
          #ifdef __DEBUG_CALC2DPSF__
              cout << "PSF trace " << _iTrace << " bin " << _iBin << "::extractPSFs: WARNING: result.size() = 0" << endl;
          #endif
        }
      }/// end if MPFitGaussLim
    }/// end if (max(spectrumSwath(Range(firstWideSignalSwathStart, firstWideSignalSwathEnd))) < _twoDPSFControl->saturationLevel){
    else{
      #ifdef __DEBUG_CALC2DPSF__
        cout << "PSF trace " << _iTrace << " bin " << _iBin << "::extractPSFs: WARNING: math::max( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) )(=" << pfs::drp::stella::math::max( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) ) << ") >= _twoDPSFControl->saturationLevel(=" << _twoDPSFControl->saturationLevel << ")" << endl;
      #endif
    }
  } while( true );
  #ifdef __DEBUG_CALC2DPSF__
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: nPix = " << nPix << ", _imagePSF_XRelativeToCenter.size() = " << _imagePSF_XRelativeToCenter.size() << endl;
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: nPix = " << nPix << ", _imagePSF_XTrace.size() = " << _imagePSF_XTrace.size() << endl;
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: nPix = " << nPix << ", _imagePSF_YRelativeToCenter.size() = " << _imagePSF_YRelativeToCenter.size() << endl;
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: nPix = " << nPix << ", _imagePSF_YTrace.size() = " << _imagePSF_YTrace.size() << endl;
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: nPix = " << nPix << ", _imagePSF_ZTrace.size() = " << _imagePSF_ZTrace.size() << endl;
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: nPix = " << nPix << ", _imagePSF_ZNormalized.size() = " << _imagePSF_ZNormalized.size() << endl;
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: nPix = " << nPix << ", _imagePSF_Weight.size() = " << _imagePSF_Weight.size() << endl;
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: xFWHM = " << _twoDPSFControl->xFWHM << ", yFWHM = " << _twoDPSFControl->yFWHM << endl;
    std::ofstream of;
    std::string ofname = __DEBUGDIR__ + std::string("pix_x_xo_y_yo_val_norm_weight_iBin");
    if (_iBin < 10)
      ofname += std::string("0");
    ofname += to_string(_iBin)+std::string(".dat");
    of.open(ofname);
    if (!of){
      string message("PSF::extractPSFs: ERROR: Could not open file <");
      message += ofname + ">";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    for (size_t iPix=0; iPix<nPix; ++iPix){
      of << _imagePSF_XRelativeToCenter[iPix];
      of << " " << _imagePSF_XTrace[iPix];
      of << " " << _imagePSF_YRelativeToCenter[iPix];
      of << " " << _imagePSF_YTrace[iPix];
      of << " " << _imagePSF_ZTrace[iPix];
      of << " " << _imagePSF_ZNormalized[iPix];
      of << " " << _imagePSF_Weight[iPix] << endl;
    }
    of.close();
    cout << "ofname = <" << ofname << "> written" << endl;
  #endif
  if ( nPix != _imagePSF_XRelativeToCenter.size() ){
    string message( "PSF trace" );
    message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: nPix != _imagePSF_XRelativeToCenter.size()";
    throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
  }
  if ( nPix != _imagePSF_XTrace.size() ){
    string message("PSF trace");
    message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: nPix != _imagePSF_XTrace.size()";
    throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
  }
  if (nPix != _imagePSF_YRelativeToCenter.size()){
    string message("PSF trace");
    message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: nPix != _imagePSF_YRelativeToCenter.size()";
    throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
  }
  if ( nPix != _imagePSF_YTrace.size() ){
    string message("PSF trace");
    message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: nPix != _imagePSF_YTrace.size()";
    throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
  }
  if ( nPix != _imagePSF_ZTrace.size() ){
    string message("PSF trace");
    message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: nPix != _imagePSF_ZTrace.size()";
    throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
  }
  if ( nPix != _imagePSF_ZNormalized.size() ){
    string message("PSF trace");
    message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: nPix != _imagePSF_ZNormalized.size()";
    throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
  }
  if ( nPix != _imagePSF_Weight.size() ){
    string message("PSF trace");
    message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: nPix != _imagePSF_Weight.size()";
    throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
  }
  _isPSFsExtracted = true;
  #ifdef __DEBUG_PSF__
    cout << "PSF::extractPSFs(FiberTrace, Spectrum, collapsedPSF) finished" << endl;
  #endif
  return true;
}
