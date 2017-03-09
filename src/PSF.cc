#include <limits>
#include "pfs/drp/stella/PSF.h"

namespace pfs{ namespace drp{ namespace stella{

//  template<typename T>//, typename WavelengthT>
//  PSF<T>::PSF(PSF &psf, const bool deep)

//  template class PSF< float >;
//  template class PSF< double >;

  /// Set the _twoDPSFControl
  template<typename T>
  bool PSF<T>::setTwoDPSFControl(PTR(TwoDPSFControl) &twoDPSFControl){
    #ifdef __DEBUG_PSF__
      cout << "PSF::Constructor(TwoDPSFControl) started" << endl;
    #endif
    assert(twoDPSFControl->signalThreshold >= 0.);
    assert(twoDPSFControl->signalThreshold < twoDPSFControl->saturationLevel / 2.);
    assert(twoDPSFControl->swathWidth > twoDPSFControl->yFWHM * 10);
    assert(twoDPSFControl->xFWHM * 4. > twoDPSFControl->nTermsGaussFit);
    assert(twoDPSFControl->yFWHM * 4. > twoDPSFControl->nTermsGaussFit);
    assert(twoDPSFControl->nTermsGaussFit > 2);
    assert(twoDPSFControl->nTermsGaussFit < 6);
    assert(twoDPSFControl->saturationLevel > 0.);
    assert(twoDPSFControl->nKnotsX > 10);
    assert(twoDPSFControl->nKnotsY > 10);
    _twoDPSFControl->signalThreshold = twoDPSFControl->signalThreshold;
    _twoDPSFControl->swathWidth = twoDPSFControl->swathWidth;
    _twoDPSFControl->xFWHM = twoDPSFControl->xFWHM;
    _twoDPSFControl->yFWHM = twoDPSFControl->yFWHM;
    _twoDPSFControl->nTermsGaussFit = twoDPSFControl->nTermsGaussFit;
    _twoDPSFControl->saturationLevel = twoDPSFControl->saturationLevel;
    _twoDPSFControl->nKnotsX = twoDPSFControl->nKnotsX;
    _twoDPSFControl->nKnotsY = twoDPSFControl->nKnotsY;
    _isTwoDPSFControlSet = true;
    #ifdef __DEBUG_PSF__
      cout << "PSF::Constructor(TwoDPSFControl) finished" << endl;
    #endif
    return true;
  }
  
//  template< typename T > template< typename ImageT, typename MaskT, typename VarianceT >
//  ExtractPSFResult< T > PSF< T >::extractPSFFromCenterPosition( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace_In,
//                                                                T const centerPositionXCCD_In,
//                                                                T const centerPositionYCCD_In){
 
//  template<> template<> ExtractPSFResult< float > PSF< float >::extractPSFFromCenterPosition(FiberTrace< float, unsigned short, float > const&, float const, float const);
//  template<> template<> ExtractPSFResult< float > PSF< float >::extractPSFFromCenterPosition(FiberTrace< float, unsigned short, double > const&, float const, float const);
//  template<> template<> ExtractPSFResult< float > PSF< float >::extractPSFFromCenterPosition<double, unsigned short, float>(FiberTrace< double, unsigned short, float > const&, float const, float const);
//  template<> template<> ExtractPSFResult< float > PSF< float >::extractPSFFromCenterPosition(FiberTrace< double, unsigned short, double > const&, float const, float const);
//  template<> template<> ExtractPSFResult< double > PSF< double >::extractPSFFromCenterPosition(FiberTrace< float, unsigned short, float > const&, double const, double const);
//  template<> template<> ExtractPSFResult< double > PSF< double >::extractPSFFromCenterPosition(FiberTrace< float, unsigned short, double > const&, double const, double const);
//  template<> template<> ExtractPSFResult< double > PSF< double >::extractPSFFromCenterPosition<double, unsigned short, float>(FiberTrace< double, unsigned short, float > const&, double const, double const);
//  template<> template<> ExtractPSFResult< double > PSF< double >::extractPSFFromCenterPosition(FiberTrace< double, unsigned short, double > const&, double const, double const);
  
  template< typename T > template< typename ImageT, typename MaskT, typename VarianceT >
  bool PSF< T >::extractPSFFromCenterPositions( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace_In,
                                                ndarray::Array< T, 1, 1 > const& centerPositionsXCCD_In,
                                                ndarray::Array< T, 1, 1 > const& centerPositionsYCCD_In ){
    #ifdef __DEBUG_PSF__
      cout << "PSF::extractPSFFromCenterPositions(FiberTrace, centerPositionXCCDArray, centerPositionYCCDArray) started" << endl;
    #endif
    if (centerPositionsXCCD_In.getShape()[ 0 ] != centerPositionsYCCD_In.getShape()[ 0 ]){
      string message("PSF::extractPSFFromCenterPositions: ERROR: centerPositionsXCCD_In.getShape()[0]=");
      message += to_string( centerPositionsXCCD_In.getShape()[ 0 ] ) + " != centerPositionsYCCD_In.getShape()[0]=" + to_string( centerPositionsYCCD_In.getShape()[ 0 ] );
      cout << message << endl;
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
      cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFFromCenterPositions: _nPixPerPSF[" << _nPixPerPSF.size()-1 << "] = " << _nPixPerPSF[_nPixPerPSF.size()-1] << endl;
      if ( _nPixPerPSF[ _nPixPerPSF.size() - 1 ] == 0 ){
        string message( "PSF trace" );
        message += to_string(_iTrace) + " bin" + to_string( _iBin ) + "::extractPSFFromCenterPositions: ERROR: _nPixPerPSF[_nPixPerPSF.size()-1=" + to_string(_nPixPerPSF.size()-1);
        message += "]=" + to_string(_nPixPerPSF[_nPixPerPSF.size()-1]) + " == 0";
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
      }
    }
    #ifdef __DEBUG_PSF__
      cout << "PSF::extractPSFFromCenterPosition(FiberTrace, centerPositionXCCDArray, centerPositionYCCDArray) finished" << endl;
    #endif
    return true;
  }
      
  template< typename T > template< typename ImageT, typename MaskT, typename VarianceT >
  bool PSF< T >::extractPSFFromCenterPositions(FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace_In){
    #ifdef __DEBUG_PSF__
      cout << "PSF::extractPSFFromCenterPositions(FiberTrace) started" << endl;
    #endif
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
//      cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFFromCenterPositionsA: _nPixPerPSF[" << _nPixPerPSF.size()-1 << "] = " << _nPixPerPSF[_nPixPerPSF.size()-1] << endl;
      if ( _nPixPerPSF[ _nPixPerPSF.size() - 1 ] == 0 ){
        string message( "PSF trace" );
        message += to_string(_iTrace) + " bin" + to_string( _iBin ) + "::extractPSFFromCenterPositionsA: ERROR: _nPixPerPSF[_nPixPerPSF.size()-1=" + to_string(_nPixPerPSF.size()-1);
        message += "]=" + to_string(_nPixPerPSF[_nPixPerPSF.size()-1]) + " == 0";
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
      }
    }
    #ifdef __DEBUG_PSF__
      cout << "PSF::extractPSFFromCenterPositions(FiberTrace) finished" << endl;
    #endif
    return true;
  }

  template<typename T> template< typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
  bool PSF<T>::extractPSFs(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTraceIn,
                           Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrumIn){
    #ifdef __DEBUG_PSF__
      cout << "PSF::extractPSFs(FiberTrace, Spectrum) started" << endl;
    #endif
    ndarray::Array<T, 2, 1> collapsedPSF = ndarray::allocate(1, 1);
    return extractPSFs(fiberTraceIn, spectrumIn, collapsedPSF);
  }
  
  template< typename T > template< typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT >
  bool PSF< T >::extractPSFs( FiberTrace< ImageT, MaskT, VarianceT > const& fiberTraceIn,
                              Spectrum< ImageT, MaskT, VarianceT, WavelengthT> const& spectrumIn,
                              ndarray::Array< T, 2, 1 > const& collapsedPSF)
  {
    #ifdef __DEBUG_PSF__
      cout << "PSF::extractPSFs(FiberTrace, Spectrum, collapsedPSF) started" << endl;
    #endif
    if (! _isTwoDPSFControlSet ){
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: _twoDPSFControl is not set";
      cout << message << endl;
      throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
    }
    if ( fiberTraceIn.getHeight() < _yMax ){
      string message("PSF trace");
      message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: fiberTraceIn.getHeight(=" + to_string( fiberTraceIn.getHeight() );
      message += " < _yMax = " + to_string( _yMax );
      cout << message << endl;
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
      ndarray::Array< double, 2, 1 > imageTrace = math::Double( fiberTraceIn.getImage()->getArray() );
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: imageTrace.getShape() = " << imageTrace.getShape() << ", _yMin = " << _yMin << ", _yMax = " << _yMax << endl;
      ndarray::Array< double, 2, 1 > imageSwath = ndarray::allocate( _yMax - _yMin + 1, fiberTraceIn.getImage()->getWidth() );
      imageSwath = imageTrace[ ndarray::view( _yMin, _yMax + 1 )() ];
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: traceIn = " << imageSwath << endl;

      ndarray::Array< double, 2, 1 > stddevSwath = ndarray::allocate( _yMax - _yMin + 1, fiberTraceIn.getImage()->getWidth() );
      ndarray::Array< double, 2, 1 > stddevTrace = math::Double( fiberTraceIn.getVariance()->getArray() );
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
        cout << message << endl;
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

    spectrumSwath = math::Double( ndarray::Array< ImageT, 1, 1 >( spectrumIn.getSpectrum()[ ndarray::view( _yMin, _yMax + 1 ) ] ) );
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: spectrumSwath = " << spectrumSwath << endl;
    #endif

    spectrumVarianceSwath = math::Double( ndarray::Array< VarianceT, 1, 1 >( spectrumIn.getVariance()[ ndarray::view( _yMin, _yMax + 1 ) ] ) );
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: spectrumVarianceSwath = " << spectrumVarianceSwath << endl;
    #endif

    ndarray::Array< double, 1, 1 > spectrumSigmaSwath = ndarray::allocate( spectrumVarianceSwath.size() );
    spectrumSigmaSwath.asEigen() = Eigen::Array< double, Eigen::Dynamic, 1 >( spectrumVarianceSwath.asEigen() ).sqrt();
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: spectrumSigmaSwath = " << spectrumSigmaSwath << endl;
    #endif

    ndarray::Array< float, 1, 1 > xCentersSwath = ndarray::copy( xCentersTrace[ ndarray::view( _yMin, _yMax + 1 ) ] );
    ndarray::Array< float, 1, 1 > xCentersSwathFloor = math::floor( ndarray::Array< const float, 1, 1 >( xCentersSwath ), float( 0 ) );
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
//    double dX, dY;//, pixOffsetY, xStart, yStart;
//    int i_Left, i_Right, i_Down, i_Up;//, i_xCenter, i_yCenter;
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
    if ( !math::countPixGTZero( spectrumSwathZeroBelowThreshold ) ){
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
      firstWideSignalSwath = math::firstIndexWithValueGEFrom( spectrumSwathZeroBelowThreshold, minWidth, startIndexSwath );
      #ifdef __DEBUG_CALC2DPSF__
        cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: firstWideSignalSwath found at index " << firstWideSignalSwath << ", startIndexSwath = " << startIndexSwath << endl;
      #endif
      if ( firstWideSignalSwath < 0 ){
        cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: No emission line found" << endl;
        break;
      }
      firstWideSignalSwathStart = math::lastIndexWithZeroValueBefore( spectrumSwathZeroBelowThreshold, firstWideSignalSwath );
      if ( firstWideSignalSwathStart < 0 ){
        firstWideSignalSwathStart = 0;
      }
      #ifdef __DEBUG_CALC2DPSF__
        cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: firstWideSignalSwathStart = " << firstWideSignalSwathStart << endl;
      #endif

      firstWideSignalSwathEnd = math::firstIndexWithZeroValueFrom( spectrumSwathZeroBelowThreshold, firstWideSignalSwath );
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

      if ( math::max( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) ) < _twoDPSFControl->saturationLevel ){
        D_A2_Limits[ ndarray::makeVector( 0, 1 )  ] = 1.5 * math::max( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) );
        D_A2_Limits[ ndarray::makeVector( 1, 0 ) ] = firstWideSignalSwathStart;
        D_A2_Limits[ ndarray::makeVector( 1, 1 ) ] = firstWideSignalSwathEnd;
        if ( _twoDPSFControl->nTermsGaussFit > 3 )
          D_A2_Limits[ ndarray::makeVector( 3, 1 ) ] = math::min( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) );

        D_A1_Guess[ 0 ] = math::max( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) );
        D_A1_Guess[ 1 ] = firstWideSignalSwathStart + ( math::maxIndex( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) ) );
        D_A1_Guess[ 2 ] = _twoDPSFControl->yFWHM / 2.3548;
        if ( _twoDPSFControl->nTermsGaussFit > 3 )
          D_A1_Guess[ 3 ] = math::min( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) );
        if ( _twoDPSFControl->nTermsGaussFit > 4 )
          D_A1_Guess[ 4 ] = ( spectrumSwath( firstWideSignalSwathEnd ) - spectrumSwath( firstWideSignalSwathStart ) ) / ( firstWideSignalSwathEnd - firstWideSignalSwathStart );
        
        ndarray::Array< double, 1, 1 > xSwathGaussFit = math::indGenNdArr( double( firstWideSignalSwathEnd - firstWideSignalSwathStart + 1 ) );
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
        success = MPFitGaussLim( xSwathGaussFit,
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
            gaussCenterY = D_A1_GaussFit_Coeffs[ 1 ] + 0.5;
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
          ndarray::Array< double, 2, 1 > indexRelToCenter = math::calcPositionsRelativeToCenter( double( gaussCenterY ), double( 4. * _twoDPSFControl->yFWHM ) );
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
            ndarray::Array< double, 1, 1 > yDataVec = math::getSubArray( spectrumSwath, indVec );
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

            double xCorMinPos = math::xCor( arrA,/// x must be 'y' relative to center
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
              cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber - 1 << ": WARNING indexRelToCenter[indexRelToCenter.getShape()[0]][0](=" << indexRelToCenter[ ndarray::makeVector<size_t>( indexRelToCenter.getShape()[0], 0 ) ] << " >= spectrumSwath.getShape()[0] = " << spectrumSwath.getShape()[0] << endl;
            success = false;
          }
        }
        if ( !success ){
          cout << "PSF trace " << _iTrace << " bin " << _iBin << "::extractPSFs: WARNING: MPFitGaussLim failed" << endl;
        }
        else{
          float yCenterCCD = gaussCenterY + fiberTraceIn.getFiberTraceFunction()->yCenter + fiberTraceIn.getFiberTraceFunction()->yLow + _yMin;
          ndarray::Array< float, 1, 1 > yCenterFromGaussCenter = math::indGenNdArr( float( 1 ) );
          yCenterFromGaussCenter.deep() = yCenterCCD;
          ndarray::Array< float, 1, 1 > xCenterCCDFromYCenterCCD = math::calculateXCenters( fiberTraceIn.getFiberTraceFunction(), 
                                                                                            yCenterFromGaussCenter );

          ExtractPSFResult< T > result = extractPSFFromCenterPosition( fiberTraceIn,
                                                                       xCenterCCDFromYCenterCCD[ 0 ],
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
            cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: _nPixPerPSF[" << _nPixPerPSF.size()-1 << "] = " << _nPixPerPSF[_nPixPerPSF.size()-1] << endl;
            if ( _nPixPerPSF[ _nPixPerPSF.size() - 1 ] == 0 ){
              string message( "PSF trace" );
              message += to_string(_iTrace) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: _nPixPerPSF[_nPixPerPSF.size()-1=" + to_string(_nPixPerPSF.size()-1);
              message += "]=" + to_string(_nPixPerPSF[_nPixPerPSF.size()-1]) + " == 0";
              throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
            }
            nPix += result.xRelativeToCenter.size();
          }
          else{
            cout << "PSF trace " << _iTrace << " bin " << _iBin << "::extractPSFs: WARNING: result.size() = 0" << endl;
          }
        }/// end if MPFitGaussLim
      }/// end if (max(spectrumSwath(Range(firstWideSignalSwathStart, firstWideSignalSwathEnd))) < _twoDPSFControl->saturationLevel){
      else{
        cout << "PSF trace " << _iTrace << " bin " << _iBin << "::extractPSFs: WARNING: math::max( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) )(=" << math::max( ndarray::Array< double, 1, 1 >( spectrumSwath[ ndarray::view( firstWideSignalSwathStart, firstWideSignalSwathEnd + 1 ) ] ) ) << ") >= _twoDPSFControl->saturationLevel(=" << _twoDPSFControl->saturationLevel << ")" << endl;
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
        cout << message << endl;
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
      cout << message << endl;
      throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
    }
    if ( nPix != _imagePSF_XTrace.size() ){
      string message("PSF trace");
      message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: nPix != _imagePSF_XTrace.size()";
      cout << message << endl;
      throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
    }
    if (nPix != _imagePSF_YRelativeToCenter.size()){
      string message("PSF trace");
      message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: nPix != _imagePSF_YRelativeToCenter.size()";
      cout << message << endl;
      throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
    }
    if ( nPix != _imagePSF_YTrace.size() ){
      string message("PSF trace");
      message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: nPix != _imagePSF_YTrace.size()";
      cout << message << endl;
      throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
    }
    if ( nPix != _imagePSF_ZTrace.size() ){
      string message("PSF trace");
      message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: nPix != _imagePSF_ZTrace.size()";
      cout << message << endl;
      throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
    }
    if ( nPix != _imagePSF_ZNormalized.size() ){
      string message("PSF trace");
      message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: nPix != _imagePSF_ZNormalized.size()";
      cout << message << endl;
      throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
    }
    if ( nPix != _imagePSF_Weight.size() ){
      string message("PSF trace");
      message += to_string( _iTrace ) + " bin" + to_string( _iBin ) + "::extractPSFs: ERROR: nPix != _imagePSF_Weight.size()";
      cout << message << endl;
      throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
    }
    _isPSFsExtracted = true;
    #ifdef __DEBUG_PSF__
      cout << "PSF::extractPSFs(FiberTrace, Spectrum, collapsedPSF) finished" << endl;
    #endif
    return true;
  }
 
//  bool PSF<ImageT, MaskT, VarianceT>::extractPSFs(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTraceIn,
//                                                               Spectrum<ImageT, MaskT, VarianceT, float> const& spectrumIn,
//                                                               ndarray::Array<double, 2, 1> const& collapsedPSF);
//  bool PSF<ImageT, MaskT, VarianceT>::extractPSFs(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTraceIn,
//                                                               Spectrum<ImageT, MaskT, VarianceT, double> const& spectrumIn,
//                                                               ndarray::Array<double, 2, 1> const& collapsedPSF);
  
  template< typename T >    
  std::vector< T > PSF< T >::reconstructFromThinPlateSplineFit(double const regularization) const {
    #ifdef __DEBUG_PSF__
      cout << "PSF::reconstructFromThinPlateSplineFit(regularization) started" << endl;
    #endif
    ndarray::Array<T, 1, 1> xArr = ndarray::external( ( const_cast< std::vector< T >* >( &_imagePSF_XRelativeToCenter ) )->data(), ndarray::makeVector( int( _imagePSF_XRelativeToCenter.size() ) ), ndarray::makeVector( 1 ) );
    ndarray::Array<T, 1, 1> yArr = ndarray::external( ( const_cast< std::vector< T >* >( &_imagePSF_YRelativeToCenter ) )->data(), ndarray::makeVector( int( _imagePSF_YRelativeToCenter.size() ) ), ndarray::makeVector( 1 ) );
    ndarray::Array<T, 1, 1> zArr = ndarray::external( ( const_cast< std::vector< T >* >( &_imagePSF_ZNormalized ) )->data(), ndarray::makeVector( int( _imagePSF_ZNormalized.size() ) ), ndarray::makeVector( 1 ) );
    TPSControl tpsControl;
    tpsControl.regularization = regularization;
    math::ThinPlateSpline< T, T > tps = math::ThinPlateSpline< T, T >( xArr,
                                                                       yArr,
                                                                       zArr,
                                                                       tpsControl );
    ndarray::Array< T, 2, 1 > zFitArr = tps.fitArray( xArr,
                                                      yArr,
                                                      false );
    std::vector< T > zVec( _imagePSF_XRelativeToCenter.size() );
    for (int i = 0; i < _imagePSF_XRelativeToCenter.size(); ++i)
      zVec[ i ] = T( zFitArr[ ndarray::makeVector( i, 0 ) ] );
    #ifdef __DEBUG_PSF__
      cout << "PSF::reconstructFromThinPlateSplineFit(regularization) finished" << endl;
    #endif
    return zVec;
  }
  
  template< typename T >
  bool PSF< T >::setImagePSF_ZFit(ndarray::Array<T, 1, 1> const& zFit){
    #ifdef __DEBUG_PSF__
      cout << "PSF::setImagePSF_ZFit(zFit) started" << endl;
    #endif
    if (zFit.getShape()[0] != _imagePSF_XRelativeToCenter.size()){
      cout << "PSF::setImagePSF_ZFit: zFit.getShape()[0]=" << zFit.getShape()[0] << " != _imagePSF_XRelativeToCenter.size()=" << _imagePSF_XRelativeToCenter.size() << " => returning FALSE" << endl;
    }
    _imagePSF_ZFit.resize(zFit.getShape()[0]);
    auto it_In = zFit.begin();
    for (auto it = _imagePSF_ZFit.begin(); it != _imagePSF_ZFit.end(); ++it, ++it_In)
      *it = *it_In;
    #ifdef __DEBUG_PSF__
      cout << "PSF::setImagePSF_ZFit(zFit) finished" << endl;
    #endif
    return true;
  }
  
  template< typename T >
  bool PSF< T >::setImagePSF_ZTrace(ndarray::Array<T, 1, 1> const& zTrace){
    #ifdef __DEBUG_PSF__
      cout << "PSF::setImagePSF_ZTrace(zTrace) started" << endl;
    #endif
    if (zTrace.getShape()[0] != _imagePSF_XRelativeToCenter.size()){
      cout << "PSF::setImagePSF_ZTrace: zTrace.getShape()[0]=" << zTrace.getShape()[0] << " != _imagePSF_XRelativeToCenter.size()=" << _imagePSF_XRelativeToCenter.size() << " => returning FALSE" << endl;
    }
    _imagePSF_ZTrace.resize(zTrace.getShape()[0]);
    auto it_In = zTrace.begin();
    for (auto it = _imagePSF_ZTrace.begin(); it != _imagePSF_ZTrace.end(); ++it, ++ it_In)
      *it = *it_In;
    #ifdef __DEBUG_PSF__
      cout << "PSF::setImagePSF_ZTrace(zTrace) finished" << endl;
    #endif
    return true;
  }
  
  template< typename T >
  bool PSF< T >::setImagePSF_ZTrace( std::vector< T > const& zTrace ){
    ndarray::Array< T, 1, 1 > zTraceTemp = ndarray::external( ( const_cast< std::vector< T >* >( &zTrace ) )->data(), ndarray::makeVector( int( zTrace.size() ) ), ndarray::makeVector( 1 ) );
    return setImagePSF_ZTrace( zTraceTemp );
  }
  
  template< typename T >
  bool PSF< T >::setImagePSF_ZNormalized(ndarray::Array<T, 1, 1> const& zNormalized){
    #ifdef __DEBUG_PSF__
      cout << "PSF::setImagePSF_ZNormalized(zNormalized) started" << endl;
    #endif
    if (zNormalized.getShape()[0] != _imagePSF_XRelativeToCenter.size()){
      cout << "PSF::setImagePSF_ZNormalized: zNormalized.getShape()[0]=" << zNormalized.getShape()[0] << " != _imagePSF_XRelativeToCenter.size()=" << _imagePSF_XRelativeToCenter.size() << " => returning FALSE" << endl;
    }
    _imagePSF_ZNormalized.resize(zNormalized.getShape()[0]);
    auto it_In = zNormalized.begin();
    for (auto it = _imagePSF_ZNormalized.begin(); it != _imagePSF_ZNormalized.end(); ++it, ++ it_In)
      *it = *it_In;
    #ifdef __DEBUG_PSF__
      cout << "PSF::setImagePSF_ZNormalized(zNormalized) finished" << endl;
    #endif
    return true;
  }
  
  template< typename T >
  bool PSF< T >::setImagePSF_ZNormalized( std::vector< T > const& zNormalized ){
    ndarray::Array< T, 1, 1 > zNorm = ndarray::external( ( const_cast< std::vector< T >* >( &zNormalized ) )->data(), ndarray::makeVector( int( zNormalized.size() ) ), ndarray::makeVector( 1 ) );
    return setImagePSF_ZNormalized( zNorm );
  }
  
  template< typename T >
  bool PSF< T >::setXCentersPSFCCD(std::vector<T> const& xCentersPSFCCD_In){
    #ifdef __DEBUG_PSF__
      cout << "PSF::setXCentersPSFCCD(xCentersPSFCCD) started" << endl;
    #endif
/*    size_t iPos = 0;
    auto itMin = std::min_element(_imagePSF_XTrace.begin(), _imagePSF_XTrace.end());
    auto itMax = std::max_element(_imagePSF_XTrace.begin(), _imagePSF_XTrace.end());
    for (auto itX = xCentersPSFCCD_In.begin(); itX != xCentersPSFCCD_In.end(); ++itX, ++iPos){
      if (*itX < *itMin){
        string message("PSF::setXCentersPSFCCD: ERROR: xCentersPSFCCD_In[iPos=");
        message += to_string(iPos) + "]=" + to_string(*itX) + " < min(_imagePSF_XTrace)=" + to_string(*itMin);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (*itX > *itMax){
        string message("PSF::setXCentersPSFCCD: ERROR: xCentersPSFCCD_In[iPos=");
        message += to_string(iPos) + "]=" + to_string(*itX) + " > max(_imagePSF_XTrace)=" + to_string(*itMax);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    }*/
    _xCentersPSFCCD = xCentersPSFCCD_In;
    #ifdef __DEBUG_PSF__
      cout << "PSF::setXCentersPSFCCD(xCentersPSFCCD) finished" << endl;
    #endif
    return true;
  }
  
  template< typename T >
  bool PSF< T >::setYCentersPSFCCD(std::vector<T> const& yCentersPSFCCD_In){
    #ifdef __DEBUG_PSF__
      cout << "PSF::setYCentersPSFCCD(yCentersPSFCCD) started" << endl;
    #endif
/*    size_t iPos = 0;
    auto itMin = std::min_element(_imagePSF_YTrace.begin(), _imagePSF_YTrace.end());
    auto itMax = std::max_element(_imagePSF_YTrace.begin(), _imagePSF_YTrace.end());
    for (auto it = yCentersPSFCCD_In.begin(); it != yCentersPSFCCD_In.end(); ++it, ++iPos){
      if (*it < *itMin){
        string message("PSF::setYCentersPSFCCD: ERROR: yCentersPSFCCD_In[iPos=");
        message += to_string(iPos) + "]=" + to_string(*it) + " < min(_imagePSF_YTrace)=" + to_string(*itMin);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (*it > *itMax){
        string message("PSF::setXCentersPSFCCD: ERROR: yCentersPSFCCD_In[iPos=");
        message += to_string(iPos) + "]=" + to_string(*it) + " > max(_imagePSF_YTrace)=" + to_string(*itMax);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    }*/
    _yCentersPSFCCD = yCentersPSFCCD_In;
    #ifdef __DEBUG_PSF__
      cout << "PSF::setYCentersPSFCCD(yCentersPSFCCD) started" << endl;
    #endif
    return true;
  }
  
  template< typename T >
  std::vector< T > PSF< T >::reconstructPSFFromFit( ndarray::Array< T, 1, 1 > const& xGridRelativeToCenterFit_In,
                                                    ndarray::Array< T, 1, 1 > const& yGridRelativeToCenterFit_In,
                                                    ndarray::Array< T, 2, 1 > const& zFit_In,
                                                    double regularization) const{
    #ifdef __DEBUG_PSF__
      cout << "PSF::reconstructPSFFromFit(xGridRelativeToCenterFit, yGridRelativeToCenterFit, zFit, regularization) started" << endl;
    #endif
    int nX = xGridRelativeToCenterFit_In.getShape()[ 0 ];
    int nY = yGridRelativeToCenterFit_In.getShape()[ 0 ];
    if ( zFit_In.getShape()[ 0 ] * zFit_In.getShape()[ 1 ] != nX * nY ){
      string message("PSF::reconstructPSFFromFit: ERROR: zFit_In.getShape()[ 0 ](=");
      message += to_string( zFit_In.getShape()[ 0 ] ) + ") * zFit_In.getShape()[ 1 ](=" + to_string( zFit_In.getShape()[ 1 ] );
      message += ") (=" + to_string( zFit_In.getShape()[ 0 ] * zFit_In.getShape()[ 1 ] ) + ") != xGridRelativeToCenterFit_In.getShape()[ 0 ](=";
      message += to_string( nX ) + ") * yGridRelativeToCenterFit_In.getShape()[ 0 ](=" + to_string( nY ) + ") (=" + to_string( nX * nY ) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    
    ndarray::Array< T, 1, 1 > xRelativeToCenter = ndarray::allocate(nX * nY);
    ndarray::Array< T, 1, 1 > yRelativeToCenter = ndarray::allocate(nX * nY);
    ndarray::Array< T, 1, 1 > zRelativeToCenter = ndarray::allocate(nX * nY);
    
    auto itXRel = xRelativeToCenter.begin();
    auto itYRel = yRelativeToCenter.begin();
    auto itZRel = zRelativeToCenter.begin();
    auto itZGridRow = zFit_In.begin();
    for ( auto itY = yGridRelativeToCenterFit_In.begin(); itY != yGridRelativeToCenterFit_In.end(); ++itY, ++itZGridRow ){
      auto itZGridCol = itZGridRow->begin();
      for ( auto itX = xGridRelativeToCenterFit_In.begin(); itX != xGridRelativeToCenterFit_In.end(); ++itX, ++itZGridCol ){
        *itXRel = *itX;
        *itYRel = *itY;
        *itZRel = *itZGridCol;
        ++itXRel;
        ++itYRel;
        ++itZRel;
      }
    }
    #ifdef __DEBUG_PFS_RECONSTRUCTPFSFROMFIT__
      cout << "PFS::reconstructPFSFromFit: xRelativeToCenter set to " << xRelativeToCenter << endl;
      cout << "PFS::reconstructPFSFromFit: yRelativeToCenter set to " << yRelativeToCenter << endl;
      cout << "PFS::reconstructPFSFromFit: zRelativeToCenter set to " << zRelativeToCenter << endl;
      cout << "PFS::reconstructPFSFromFit: starting construct ThinPlateSpline" << endl;
    #endif
    TPSControl tpsControl;
    tpsControl.regularization = regularization;
    math::ThinPlateSpline< T, T > tps = math::ThinPlateSpline< T, T >( xRelativeToCenter,
                                                                       yRelativeToCenter,
                                                                       zRelativeToCenter,
                                                                       tpsControl );
    ndarray::Array< T, 1, 1 > xArr = ndarray::external( ( const_cast< std::vector< T >* >( &_imagePSF_XRelativeToCenter ) )->data(), ndarray::makeVector( int( _imagePSF_XRelativeToCenter.size() ) ), ndarray::makeVector( 1 ) );
    ndarray::Array< T, 1, 1 > yArr = ndarray::external( ( const_cast< std::vector< T >* >( &_imagePSF_YRelativeToCenter ) )->data(), ndarray::makeVector( int( _imagePSF_YRelativeToCenter.size() ) ), ndarray::makeVector( 1 ) );
    #ifdef __DEBUG_PFS_RECONSTRUCTPFSFROMFIT__
      cout << "PFS::reconstructPFSFromFit: starting tps.fitArray" << endl;
    #endif
    ndarray::Array< T, 2, 1 > zFitArr = tps.fitArray( xArr,
                                                      yArr,
                                                      false );
    std::vector< T > zVec( _imagePSF_XRelativeToCenter.size() );
    for ( int i = 0; i < _imagePSF_XRelativeToCenter.size(); ++i )
      zVec[ i ] = zFitArr[ ndarray::makeVector( i, 0 ) ];
    #ifdef __DEBUG_PFS_RECONSTRUCTPFSFROMFIT__
      cout << "PFS::reconstructPFSFromFit: zVec set to " << zVec << endl;
    #endif
    #ifdef __DEBUG_PSF__
      cout << "PSF::reconstructPSFFromFit(xGridRelativeToCenterFit, yGridRelativeToCenterFit, zFit, regularization) finished" << endl;
    #endif
    return zVec;
  }
  
  template< typename T >
  std::vector< T > PSF< T >::reconstructPSFFromFit( ndarray::Array< T, 1, 1 > const& xGridRelativeToCenterFit_In,
                                                    ndarray::Array< T, 1, 1 > const& yGridRelativeToCenterFit_In,
                                                    ndarray::Array< T, 2, 1 > const& zFit_In,
                                                    ndarray::Array< T, 2, 1 > const& weights_In ) const{
    #ifdef __DEBUG_PSF__
      cout << "PSF::reconstructPSFFromFit(xGridRelativeToCenterFit, yGridRelativeToCenterFit, zFit, weights) started" << endl;
    #endif
    int nX = xGridRelativeToCenterFit_In.getShape()[ 0 ];
    int nY = yGridRelativeToCenterFit_In.getShape()[ 0 ];
    if ( zFit_In.getShape()[ 0 ] * zFit_In.getShape()[ 1 ] != nX * nY ){
      string message("PSF::reconstructPSFFromFit: ERROR: zFit_In.getShape()[ 0 ](=");
      message += to_string( zFit_In.getShape()[ 0 ] ) + ") * zFit_In.getShape()[ 1 ](=" + to_string( zFit_In.getShape()[ 1 ] );
      message += ") (=" + to_string( zFit_In.getShape()[ 0 ] * zFit_In.getShape()[ 1 ] ) + ") != xGridRelativeToCenterFit_In.getShape()[ 0 ](=";
      message += to_string( nX ) + ") * yGridRelativeToCenterFit_In.getShape()[ 0 ](=" + to_string( nY ) + ") (=" + to_string( nX * nY ) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if ( weights_In.getShape()[ 0 ] * weights_In.getShape()[ 1 ] != nX * nY ){
      string message("PSF::reconstructPSFFromFit: ERROR: weights_In.getShape()[ 0 ](=");
      message += to_string( weights_In.getShape()[ 0 ] ) + ") * weights_In.getShape()[ 1 ](=" + to_string( weights_In.getShape()[ 1 ] );
      message += ") (=" + to_string( zFit_In.getShape()[ 0 ] * zFit_In.getShape()[ 1 ] ) + ") != xGridRelativeToCenterFit_In.getShape()[ 0 ](=";
      message += to_string( nX ) + ") * yGridRelativeToCenterFit_In.getShape()[ 0 ](=" + to_string( nY ) + ") (=" + to_string( nX * nY ) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    
    ndarray::Array< T, 1, 1 > xRelativeToCenter = ndarray::allocate(nX * nY);
    ndarray::Array< T, 1, 1 > yRelativeToCenter = ndarray::allocate(nX * nY);
    ndarray::Array< T, 1, 1 > zRelativeToCenter = ndarray::allocate(nX * nY);
    ndarray::Array< T, 1, 1 > weights = ndarray::allocate(nX * nY);
    
    auto itXRel = xRelativeToCenter.begin();
    auto itYRel = yRelativeToCenter.begin();
    auto itZRel = zRelativeToCenter.begin();
    auto itWeights = weights.begin();
    auto itZGridRow = zFit_In.begin();
    auto itWeightsRow = weights_In.begin();
    for ( auto itY = yGridRelativeToCenterFit_In.begin(); itY != yGridRelativeToCenterFit_In.end(); ++itY, ++itZGridRow, ++itWeightsRow ){
      auto itZGridCol = itZGridRow->begin();
      auto itWeightsCol = itWeightsRow->begin();
      for ( auto itX = xGridRelativeToCenterFit_In.begin(); itX != xGridRelativeToCenterFit_In.end(); ++itX, ++itZGridCol, ++itWeightsCol, ++itXRel, ++itYRel, ++itZRel, ++itWeights ){
        *itXRel = *itX;
        *itYRel = *itY;
        *itZRel = *itZGridCol;
        *itWeights = *itWeightsCol;
//        ++itXRel;
//        ++itYRel;
//        ++itZRel;
      }
    }
    #ifdef __DEBUG_PFS_RECONSTRUCTPFSFROMFIT__
      cout << "PFS::reconstructPFSFromFit: xRelativeToCenter set to " << xRelativeToCenter << endl;
      cout << "PFS::reconstructPFSFromFit: yRelativeToCenter set to " << yRelativeToCenter << endl;
      cout << "PFS::reconstructPFSFromFit: zRelativeToCenter set to " << zRelativeToCenter << endl;
      cout << "PFS::reconstructPFSFromFit: weights set to " << weights << endl;
      cout << "PFS::reconstructPFSFromFit: starting construct ThinPlateSpline" << endl;
    #endif
    TPSControl tpsControl;
    math::ThinPlateSpline< T, T > tps = math::ThinPlateSpline< T, T >( xRelativeToCenter,
                                                                       yRelativeToCenter,
                                                                       zRelativeToCenter,
                                                                       weights,
                                                                       tpsControl );
    ndarray::Array< T, 1, 1 > xArr = ndarray::external( ( const_cast< std::vector< T >* >( &_imagePSF_XRelativeToCenter ) )->data(), ndarray::makeVector( int( _imagePSF_XRelativeToCenter.size() ) ), ndarray::makeVector( 1 ) );
    ndarray::Array< T, 1, 1 > yArr = ndarray::external( ( const_cast< std::vector< T >* >( &_imagePSF_YRelativeToCenter ) )->data(), ndarray::makeVector( int( _imagePSF_YRelativeToCenter.size() ) ), ndarray::makeVector( 1 ) );
    #ifdef __DEBUG_PFS_RECONSTRUCTPFSFROMFIT__
      cout << "PFS::reconstructPFSFromFit: starting tps.fitArray" << endl;
    #endif
    ndarray::Array< T, 2, 1 > zFitArr = tps.fitArray( xArr,
                                                      yArr,
                                                      false );
    std::vector< T > zVec( _imagePSF_XRelativeToCenter.size() );
    for ( int i = 0; i < _imagePSF_XRelativeToCenter.size(); ++i )
      zVec[ i ] = zFitArr[ ndarray::makeVector( i, 0 ) ];
    #ifdef __DEBUG_PFS_RECONSTRUCTPFSFROMFIT__
      cout << "PFS::reconstructPFSFromFit: zVec set to " << zVec << endl;
    #endif
    #ifdef __DEBUG_PSF__
      cout << "PSF::reconstructPSFFromFit(xGridRelativeToCenterFit, yGridRelativeToCenterFit, zFit, weights) finished" << endl;
    #endif
    return zVec;
  }

  template< typename T >
  double PSF< T >::fitFittedPSFToZTrace( std::vector< T > const& zFit_In ){
    #ifdef __DEBUG_PSF__
      cout << "PSF::fitFittedPSFToZTrace(zFitVec) started" << endl;
    #endif
    std::vector< T > measureErrors;
    measureErrors.assign( zFit_In.size(), 0. );
    double result = fitFittedPSFToZTrace( zFit_In,
                                          measureErrors );
    #ifdef __DEBUG_PSF__
      cout << "PSF::fitFittedPSFToZTrace(zFit) finished" << endl;
    #endif
    return result;
  }

  template< typename T >
  double PSF< T >::fitFittedPSFToZTrace( ndarray::Array< T, 1, 1 > const& zFit_In ){
    #ifdef __DEBUG_PSF__
      cout << "PSF::fitFittedPSFToZTrace(zFitArr) started" << endl;
    #endif
    ndarray::Array< T, 1, 1 > measureErrors = ndarray::allocate( zFit_In.getShape()[ 0 ] );
    measureErrors.deep() = 0.;
    double result = fitFittedPSFToZTrace( zFit_In,
                                          measureErrors );
    #ifdef __DEBUG_PSF__
      cout << "PSF::fitFittedPSFToZTrace(zFitArr) finished" << endl;
    #endif
    return result;
  }
  
  template< typename T >
  double PSF< T >::fitFittedPSFToZTrace( std::vector< T > const& zFit_In,
                                         std::vector< T > const& measureErrors_In ){
    #ifdef __DEBUG_PSF__
      cout << "PSF::fitFittedPSFToZTrace(zFitVec, measureErrors) started" << endl;
    #endif
    ndarray::Array< T, 1, 1 > measureErrors = ndarray::external( (const_cast< std::vector< T >* >( &measureErrors_In ) )->data(), ndarray::makeVector( int( measureErrors_In.size() ) ), ndarray::makeVector( 1 ) );
    ndarray::Array< T, 1, 1 > zFit = ndarray::external( ( const_cast< std::vector< T >* >( &zFit_In ) )->data(), ndarray::makeVector( int( zFit_In.size() ) ), ndarray::makeVector( 1 ) );
//    cout << "PSF::fitFittedPSFToZTrace(zFitVec, measureErrors): measureErrors = " << measureErrors << endl;
    double result = fitFittedPSFToZTrace( zFit,
                                          measureErrors );
    #ifdef __DEBUG_PSF__
      cout << "PSF::fitFittedPSFToZTrace(zFitVec, measureErrors) finished" << endl;
    #endif
    return result;
  }
  
  template< typename T >
  double PSF< T >::fitFittedPSFToZTrace( ndarray::Array< T, 1, 1 > const& zFit_In,
                                         ndarray::Array< T, 1, 1 > const& measureErrors_In ){
    #ifdef __DEBUG_PSF__
      cout << "PSF::fitFittedPSFToZTrace(zFitArr, measureErrors) started" << endl;
    #endif
    std::vector< std::string > args(1);
    args[0] = "MEASURE_ERRORS_IN";
    std::vector< void* > argV(1);
    PTR(ndarray::Array< T, 1, 1 >) p_measureErrors(new ndarray::Array< T, 1, 1 >(measureErrors_In));
    argV[0] = &p_measureErrors;
    T fittedValue, fittedConstant;
    int result;
    ndarray::Array< T, 1, 1 > zTrace = ndarray::external( _imagePSF_ZTrace.data(), ndarray::makeVector( int( _imagePSF_ZTrace.size() ) ), ndarray::makeVector( 1 ) );
    result = pfs::drp::stella::math::LinFitBevingtonNdArray( zTrace,
                                                             zFit_In,
                                                             fittedValue,
                                                             fittedConstant,
                                                             false,
                                                             args,
                                                             argV );
    #ifdef __DEBUG_PSF__
      cout << "PSF::fitFittedPSFToZTrace(zFitArr, measureErrors) finished" << endl;
    #endif
    return fittedValue;
  }
  
  template< typename T >
  bool PSFSet< T >::setPSF(const size_t i,
                           const PTR(PSF< T >) & psf)
  {
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::setPSF(i, psf) started" << endl;
    #endif
    if (i > _psfs->size()){
      string message("PSFSet::setPSF: ERROR: i=");
      message += to_string(i) + " > _psfs->size()=" + to_string(_psfs->size()) + " => Returning FALSE";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (i == _psfs->size()){
      _psfs->push_back(psf);
    }
    else{
      (*_psfs)[i] = psf;
    }
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::setPSF(i, psf) finished" << endl;
    #endif
    return true;
  }

  /// add one PSF to the set
  template< typename T >
  void PSFSet< T >::addPSF(const PTR(PSF< T >) & psf )/// the Spectrum to add
  {
    _psfs->push_back(psf);
  }

  template< typename T >
  PTR(PSF< T >)& PSFSet< T >::getPSF(const size_t i){
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::getPSF(i) started" << endl;
    #endif
    if (i >= _psfs->size()){
      string message("PSFSet::getPSF(i=");
      message += to_string(i) + string("): ERROR: i > _psfs->size()=") + to_string(_psfs->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::getPSF(i) finished" << endl;
    #endif
    return _psfs->at(i); 
  }

  template< typename T >
  const PTR(const PSF< T >) PSFSet< T >::getPSF(const size_t i) const { 
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::getPSF(i) const started" << endl;
    #endif
    if (i >= _psfs->size()){
      string message("PSFSet::getPSF(i=");
      message += to_string(i) + string("): ERROR: i > _psfs->size()=") + to_string(_psfs->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::getPSF(i) const started" << endl;
    #endif
    return _psfs->at(i); 
  }

  template< typename T >
  bool PSFSet< T >::erase(const size_t iStart, const size_t iEnd){
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::erase(iStart, iEnd) started" << endl;
    #endif
    if (iStart >= _psfs->size()){
      string message("PSFSet::erase(iStart=");
      message += to_string(iStart) + string("): ERROR: iStart >= _psfs->size()=") + to_string(_psfs->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (iEnd >= _psfs->size()){
      string message("PSFSet::erase(iEnd=");
      message += to_string(iEnd) + string("): ERROR: iEnd >= _psfs->size()=") + to_string(_psfs->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if ((iEnd > 0) && (iStart > iEnd)){
      string message("PSFSet::erase(iStart=");
      message += to_string(iStart) + string("): ERROR: iStart > iEnd=") + to_string(iEnd);
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (iStart == (_psfs->size()-1)){
      _psfs->pop_back();
    }
    else{
      if (iEnd == 0)
        _psfs->erase(_psfs->begin() + iStart);
      else
        _psfs->erase(_psfs->begin() + iStart, _psfs->begin() + iEnd);
    }
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::erase(iStart, iEnd) started" << endl;
    #endif
    return true;
  }

  namespace math{

    template< typename PsfT, typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
    PTR(PSFSet< PsfT >) calculate2dPSFPerBin(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace,
                                          Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrum,
                                          TwoDPSFControl const& twoDPSFControl){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::calculate2dPSFPerBin(fiberTrace, spectrum, twoDPSFControl) started" << endl;
      #endif
      ndarray::Array< PsfT, 2, 1 > collapsedPSF = ndarray::allocate(1,1);
      return calculate2dPSFPerBin(fiberTrace, spectrum, twoDPSFControl, collapsedPSF);
    }
    
    template< typename PsfT, typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
    PTR(PSFSet< PsfT >) calculate2dPSFPerBin(FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace,
                                             Spectrum< ImageT, MaskT, VarianceT, WavelengthT > const& spectrum,
                                             TwoDPSFControl const& twoDPSFControl,
                                             ndarray::Array< PsfT, 2, 1 > const& collapsedPSF){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::calculate2dPSFPerBin(fiberTrace, spectrum, twoDPSFControl, collapsedPSF) started" << endl;
      #endif
      int swathWidth = twoDPSFControl.swathWidth;
      ndarray::Array<size_t, 2, 1> binBoundY = fiberTrace.calcSwathBoundY(swathWidth);
      if ( binBoundY[ ndarray::makeVector<size_t>( binBoundY.getShape()[0]-1, 1 ) ] != fiberTrace.getHeight()-1){
        string message("calculate2dPSFPerBin: FiberTrace");
        message += to_string(fiberTrace.getITrace()) + ": ERROR: binBoundY[binBoundY.getShape()[0]-1=";
        message += to_string(binBoundY.getShape()[0]-1) + "][1] = " + to_string( binBoundY[ ndarray::makeVector<size_t>( binBoundY.getShape()[0]-1, 1 ) ] ) + "!= fiberTrace.getHeight()-1 = ";
        message += to_string(fiberTrace.getHeight()-1);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_CALC2DPSF__
        cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": fiberTrace.getHeight() = " << fiberTrace.getHeight() << ": binBoundY = " << binBoundY << endl;
//        exit(EXIT_FAILURE);
      #endif

      PTR(PSFSet< PsfT >) psfSet(new PSFSet< PsfT >());
      for (int iBin = 0; iBin < binBoundY.getShape()[0]; ++iBin){
        #ifdef __DEBUG_CALC2DPSF__
          cout << "binBoundY = " << binBoundY << endl;
        #endif
        /// start calculate2dPSF for bin iBin
        if ( binBoundY[ ndarray::makeVector( iBin, 1 ) ] >= fiberTrace.getHeight()){
          string message("calculate2dPSFPerBin: FiberTrace");
          message += to_string(fiberTrace.getITrace()) + ": iBin " + to_string(iBin) + ": ERROR: binBoundY[" + to_string(iBin) + "][1]=";
          message += to_string( binBoundY[ ndarray::makeVector( iBin, 1 ) ] ) + " >= fiberTrace.getHeight()=" + to_string(fiberTrace.getHeight());
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        #ifdef __DEBUG_CALC2DPSF__
          cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": calculating PSF for iBin " << iBin << ": binBoundY[" << iBin << "][0] = " << binBoundY[ ndarray::makeVector( iBin, 0 ) ] << ", binBoundY[" << iBin << "][1] = " << binBoundY[ndarray::makeVector( iBin, 1 ) ] << endl;
        #endif
        PTR(TwoDPSFControl) pTwoDPSFControl(new TwoDPSFControl(twoDPSFControl));
        PTR(PSF< PsfT >) psf(new PSF< PsfT >((unsigned int)(binBoundY[ ndarray::makeVector( iBin, 0 ) ] ),
                                             (unsigned int)(binBoundY[ ndarray::makeVector( iBin, 1 ) ] ),
                                             pTwoDPSFControl,
                                             fiberTrace.getITrace(),
                                             iBin));
        if (psf->getYHigh() != binBoundY[ ndarray::makeVector( iBin, 1 ) ] ){
          string message("calculate2dPSFPerBin: FiberTrace");
          message += to_string(fiberTrace.getITrace()) + ": iBin " + to_string(iBin) + ": ERROR: psf->getYHigh(=";
          message += to_string(psf->getYHigh()) + ") != binBoundY[" + to_string(iBin) + "][1]=" + to_string(binBoundY[ ndarray::makeVector( iBin, 1 ) ] );
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        #ifdef __DEBUG_CALC2DPSF__
          cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": iBin " << iBin << ": starting extractPSFs()" << endl;
        #endif
        if (!psf->extractPSFs(fiberTrace, 
                              spectrum,
                              collapsedPSF)){
          string message("calculate2dPSFPerBin: FiberTrace");
          message += to_string(fiberTrace.getITrace()) + string(": iBin ") + to_string(iBin) + string(": ERROR: psf->extractPSFs() returned FALSE");
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        if (psf->getITrace() != fiberTrace.getITrace()){
          string message("calculate2dPSFPerBin: FiberTrace");
          message += to_string(fiberTrace.getITrace()) + string(": iBin ") + to_string(iBin) + string(": ERROR: psf->getITrace(=");
          message += to_string(psf->getITrace()) + ") != fiberTrace.getITrace(=" + to_string(fiberTrace.getITrace());
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        #ifdef __DEBUG_CALC2DPSF__
          cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": iBin " << iBin << ": extractPSFs() finished" << endl;
          cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": iBin = " << iBin << ": psf->getImagePSF_XTrace().size() = " << psf->getImagePSF_XTrace().size() << endl;
        #endif
        if (psf->getImagePSF_XRelativeToCenter().size() > 3)
          psfSet->addPSF(psf);
//        #ifdef __DEBUG_CALC2DPSF__
//          if (iBin == 21)
//            exit(EXIT_FAILURE);
//        #endif
      }
      #ifdef __DEBUG_PSF__
        cout << "psfMath::calculate2dPSFPerBin(fiberTrace, spectrum, twoDPSFControl, collapsedPSF) finished" << endl;
      #endif
      return psfSet;
    }
    
    template< typename PsfT, typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT >
    std::vector< PTR( PSFSet< PsfT > ) > calculate2dPSFPerBin( FiberTraceSet< ImageT, MaskT, VarianceT > const& fiberTraceSet,
                                                               SpectrumSet< ImageT, MaskT, VarianceT, WavelengthT > const& spectrumSet,
                                                               TwoDPSFControl const& twoDPSFControl ){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::calculate2dPSFPerBin(fiberTraceSet, spectrumSet, twoDPSFControl) started" << endl;
      #endif
      std::vector< PTR(PSFSet< PsfT >)> vecOut(0);
      for (size_t i = 0; i < fiberTraceSet.size(); ++i){
        PTR(PSFSet< PsfT >) psfSet = calculate2dPSFPerBin< PsfT, ImageT, MaskT, VarianceT, WavelengthT >( *( fiberTraceSet.getFiberTrace( i ) ), 
                                                                                                          *( spectrumSet.getSpectrum( i ) ), 
                                                                                                          twoDPSFControl );
        vecOut.push_back(psfSet);
      }
      #ifdef __DEBUG_PSF__
        cout << "psfMath::calculate2dPSFPerBin(fiberTraceSet, spectrumSet, twoDPSFControl) finished" << endl;
      #endif
      return vecOut;
    }
    
    template< typename PsfT, typename CoordsT >
    ndarray::Array< PsfT, 2, 1 > interpolatePSFThinPlateSpline(PSF< PsfT > & psf,
                                                               ndarray::Array< CoordsT, 1, 1 > const& xPositions,
                                                               ndarray::Array< CoordsT, 1, 1 > const& yPositions,
                                                               bool const isXYPositionsGridPoints,
                                                               double const regularization,
                                                               PsfT const shapeParameter,
                                                               unsigned short const mode ){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFThinPlateSpline(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization) started" << endl;
      #endif
      if (mode > 1){
        string message("PSF::InterPolateThinPlateSplineChiSquare: ERROR: mode must be 0 or 1");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      std::vector< PsfT > xRelativeToCenter = psf.getImagePSF_XRelativeToCenter();
      std::vector< PsfT > yRelativeToCenter = psf.getImagePSF_YRelativeToCenter();
      //std::vector< CoordsT > xRelativeToCenter( xRel.begin(), xRel.end() );
      //std::vector< CoordsT > yRelativeToCenter( yRel.begin(), yRel.end() );
      if (xRelativeToCenter.size() < 3){
        string message("PSF::InterPolateThinPlateSpline: ERROR: xRelativeToCenter.size()(=");
        message += to_string( xRelativeToCenter.size() ) + ") < 3";
        cout << message << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
      }

      std::vector< PsfT > zVec;
      if ( mode == 0 )
        zVec = psf.getImagePSF_ZNormalized();
      else
        zVec = psf.getImagePSF_ZTrace();
      
      ndarray::Array< PsfT, 1, 1 > xArr = ndarray::external( xRelativeToCenter.data(), ndarray::makeVector( int( xRelativeToCenter.size() ) ), ndarray::makeVector( 1 ) );
      ndarray::Array< PsfT, 1, 1 > yArr = ndarray::external( yRelativeToCenter.data(), ndarray::makeVector( int( yRelativeToCenter.size() ) ), ndarray::makeVector( 1 ) );
      ndarray::Array< PsfT, 1, 1 > zArr = ndarray::external( zVec.data(), ndarray::makeVector( int( zVec.size() ) ), ndarray::makeVector( 1 ) );
      cout << "PSF::interpolatePSFThinPlateSpline: starting interpolateThinPlateSplineEigen" << endl;
      TPSControl tpsControl;
      tpsControl.regularization = regularization;
      tpsControl.shapeParameter = shapeParameter;
      math::ThinPlateSpline<PsfT, PsfT> tps = math::ThinPlateSpline<PsfT, PsfT>( xArr, 
                                                                                 yArr, 
                                                                                 zArr, 
                                                                                 tpsControl );
      ndarray::Array< PsfT, 1, 1 > xPos = ndarray::allocate( xPositions.getShape() );
      ndarray::Array< PsfT, 1, 1 > yPos = ndarray::allocate( yPositions.getShape() );
      auto itXPos = xPos.begin();
      for (auto itX = xPositions.begin(); itX != xPositions.end(); ++itX, ++itXPos ){
        *itXPos = PsfT(*itX);
      }
      auto itYPos = yPos.begin();
      for (auto itY = yPositions.begin(); itY != yPositions.end(); ++itY, ++itYPos ){
        *itYPos = PsfT(*itY);
      }
      cout << "PSF::interpolatePSFThinPlateSpline: xPos = " << xPos << endl;
      cout << "PSF::interpolatePSFThinPlateSpline: yPos = " << yPos << endl;
      ndarray::Array< PsfT, 2, 1 > arr_Out = ndarray::copy( tps.fitArray( xPos, 
                                                                          yPos,
                                                                          isXYPositionsGridPoints ) );
      cout << "PSF::interpolatePSFThinPlateSpline: arr_Out = " << arr_Out << endl;
      ndarray::Array< PsfT, 1, 1 > zRec = ndarray::copy( tps.getZFit() );
      if ( !psf.setImagePSF_ZFit( zRec ) ){
        cout << "PSF::interpolatePSFThinPlateSpline: WARNING: psf.setImagePSF_ZFit( zRec ) returned FALSE" << endl;
      }
      psf.setThinPlateSpline( tps );
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFThinPlateSpline(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization, shapeParameter, mode): tps.getDataPointsX().getShape()[ 0 ] = " << tps.getDataPointsX().getShape()[ 0 ] << endl;
        cout << "psfMath::interpolatePSFThinPlateSpline(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization, shapeParameter, mode): psf.getThinPlateSpline().getDataPointsX().getShape()[ 0 ] = " << psf.getThinPlateSpline().getDataPointsX().getShape()[ 0 ] << endl;
        cout << "psfMath::interpolatePSFThinPlateSpline(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization, shapeParameter, mode) finished" << endl;
      #endif
      return arr_Out;
    }
    
    template< typename PsfT, typename WeightT, typename CoordsT >
    ndarray::Array< PsfT, 2, 1> interpolatePSFThinPlateSpline( PSF< PsfT > & psf,
                                                               ndarray::Array<WeightT, 1, 1> const& weights,
                                                               ndarray::Array<CoordsT, 1, 1> const& xPositions,
                                                               ndarray::Array<CoordsT, 1, 1> const& yPositions,
                                                               bool const isXYPositionsGridPoints,
                                                               PsfT const shapeParameter,
                                                               unsigned short const mode ){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFThinPlateSpline(psf, weights, xPositions, yPositions, isXYPositionsGridPoints) started" << endl;
      #endif
      if (mode > 1){
        string message("PSF::InterPolateThinPlateSplineChiSquare: ERROR: mode must be 0 or 1");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      std::vector< PsfT > xRelativeToCenter = psf.getImagePSF_XRelativeToCenter();
      std::vector< PsfT > yRelativeToCenter = psf.getImagePSF_YRelativeToCenter();
      //std::vector< CoordsT > xRelativeToCenter( xRel.begin(), xRel.end() );
      //std::vector< CoordsT > yRelativeToCenter( yRel.begin(), yRel.end() );
      if (xRelativeToCenter.size() < 3){
        string message("PSF::InterPolateThinPlateSpline: ERROR: xRelativeToCenter.size()(=");
        message += to_string( xRelativeToCenter.size() ) + ") < 3";
        cout << message << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
      }

      std::vector< PsfT > zVec;
      if ( mode == 0 )
        zVec = psf.getImagePSF_ZNormalized();
      else
        zVec = psf.getImagePSF_ZTrace();
      
      ndarray::Array< PsfT, 1, 1 > xArr = ndarray::external( xRelativeToCenter.data(), ndarray::makeVector( int( xRelativeToCenter.size() ) ), ndarray::makeVector( 1 ) );
      ndarray::Array< PsfT, 1, 1 > yArr = ndarray::external( yRelativeToCenter.data(), ndarray::makeVector( int( yRelativeToCenter.size() ) ), ndarray::makeVector( 1 ) );
      ndarray::Array< PsfT, 1, 1 > zArr = ndarray::external( zVec.data(), ndarray::makeVector( int( zVec.size() ) ), ndarray::makeVector( 1 ) );
/*      for (int i = 0; i < xArr.getShape()[0]; ++i){
        xArr[i] = psf.getImagePSF_XRelativeToCenter()[i];
        yArr[i] = psf.getImagePSF_YRelativeToCenter()[i];
        zArr[i] = psf.getImagePSF_ZNormalized()[i];
      }*/
      #ifdef __DEBUG_CALC_TPS__
        cout << "PSF::interpolatePSFThinPlateSpline: psf.getImagePSF_ZNormalized() = " << psf.getImagePSF_ZNormalized() << endl;
        cout << "PSF::interpolatePSFThinPlateSpline: xArr = " << xArr << endl;
        cout << "PSF::interpolatePSFThinPlateSpline: yArr = " << yArr << endl;
        cout << "PSF::interpolatePSFThinPlateSpline: zArr = " << zArr << endl;
      #endif
/*      ndarray::Array< PsfT, 1, 1 > zT = ndarray::allocate(zArr.size());
      auto it = zArr.begin();
      for (auto itT = zT.begin(); itT != zT.end(); ++itT, ++it)
        *itT = PsfT(*it);
      #ifdef __DEBUG_CALC_TPS__
        cout << "PSF::interpolatePSFThinPlateSpline: zT = " << zT << endl;
      #endif*/
      cout << "PSF::interpolatePSFThinPlateSpline: starting interpolateThinPlateSplineEigen" << endl;
      ndarray::Array<PsfT, 1, 1> weightsT = ndarray::allocate(weights.getShape()[0]);
      auto itWeightsT = weightsT.begin();
      for (auto itWeights = weights.begin(); itWeights != weights.end(); ++itWeights, ++itWeightsT)
        *itWeightsT = PsfT(*itWeights);
      TPSControl tpsControl;
      math::ThinPlateSpline< PsfT, PsfT > tps = math::ThinPlateSpline< PsfT, PsfT >( xArr, 
                                                                                     yArr, 
                                                                                     zArr, 
                                                                                     weightsT,
                                                                                     tpsControl );
      
      ndarray::Array< PsfT, 1, 1 > xPos = ndarray::allocate( xPositions.getShape() );
      ndarray::Array< PsfT, 1, 1 > yPos = ndarray::allocate( yPositions.getShape() );
      auto itXPos = xPos.begin();
      for (auto itX = xPositions.begin(); itX != xPositions.end(); ++itX, ++itXPos ){
        *itXPos = PsfT(*itX);
      }
      auto itYPos = yPos.begin();
      for (auto itY = yPositions.begin(); itY != yPositions.end(); ++itY, ++itYPos ){
        *itYPos = PsfT(*itY);
      }
      ndarray::Array< PsfT, 2, 1 > zFit = ndarray::copy( tps.fitArray( xPos, 
                                                                       yPos, 
                                                                       isXYPositionsGridPoints ) );
      ndarray::Array< PsfT, 1, 1 > zRec = ndarray::copy( tps.getZFit() );
      if ( !psf.setImagePSF_ZFit( zRec ) ){
        cout << "PSF::interpolatePSFThinPlateSpline: WARNING: psf.setImagePSF_ZFit( zRec ) returned FALSE" << endl;
      }
      psf.setThinPlateSpline( tps );
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFThinPlateSpline(): tps.getDataPointsX().getShape()[ 0 ] = " << tps.getDataPointsX().getShape()[ 0 ] << endl;
        cout << "psfMath::interpolatePSFThinPlateSpline(): psf.getThinPlateSpline().getDataPointsX().getShape()[ 0 ] = " << psf.getThinPlateSpline().getDataPointsX().getShape()[ 0 ] << endl;
        cout << "psfMath::interpolatePSFThinPlateSpline(psf, weights, xPositions, yPositions, isXYPositionsGridPoints) finished" << endl;
      #endif
      return zFit;
    }
    
    template< typename PsfT, typename CoordsT >
    ndarray::Array< PsfT, 2, 1 > interpolatePSFThinPlateSplineChiSquare( PSF< PsfT > & psf,
                                                                         ndarray::Array< CoordsT, 1, 1 > const& xPositions,
                                                                         ndarray::Array< CoordsT, 1, 1 > const& yPositions,
                                                                         bool const isXYPositionsGridPoints,
                                                                         PsfT const regularization,
                                                                         PsfT const shapeParameter,
                                                                         unsigned short const mode ){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFThinPlateSplineChiSquare(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization) started" << endl;
      #endif
      if (mode > 1){
        string message("PSF::InterPolateThinPlateSplineChiSquare: ERROR: mode must be 0 or 1");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      std::vector< PsfT > xRelativeToCenter = psf.getImagePSF_XRelativeToCenter();
      std::vector< PsfT > yRelativeToCenter = psf.getImagePSF_YRelativeToCenter();
      //std::vector< CoordsT > xRelativeToCenter( xRel.begin(), xRel.end() );
      //std::vector< CoordsT > yRelativeToCenter( yRel.begin(), yRel.end() );
      if (xRelativeToCenter.size() < 3){
        string message("PSF::InterPolateThinPlateSplineChiSquare: ERROR: xRelativeToCenter.size()(=");
        message += to_string( xRelativeToCenter.size() ) + ") < 3";
        cout << message << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
      }

      std::vector< PsfT > zVec;
      if ( mode == 0 )
        zVec = psf.getImagePSF_ZNormalized();
      else
        zVec = psf.getImagePSF_ZTrace();
      
      ndarray::Array< PsfT, 1, 1 > xArr = ndarray::external( xRelativeToCenter.data(), ndarray::makeVector( int( xRelativeToCenter.size() ) ), ndarray::makeVector( 1 ) );
      ndarray::Array< PsfT, 1, 1 > yArr = ndarray::external( yRelativeToCenter.data(), ndarray::makeVector( int( yRelativeToCenter.size() ) ), ndarray::makeVector( 1 ) );
      ndarray::Array< PsfT, 1, 1 > zArr = ndarray::external( zVec.data(), ndarray::makeVector( int( zVec.size() ) ), ndarray::makeVector( 1 ) );
      #ifdef __DEBUG_CALC_TPS__
        cout << "PSF::interpolatePSFThinPlateSplineChiSquare: xArr = " << xArr << endl;
        cout << "PSF::interpolatePSFThinPlateSplineChiSquare: yArr = " << yArr << endl;
        cout << "PSF::interpolatePSFThinPlateSplineChiSquare: psf.getImagePSF_ZNormalized() = " << psf.getImagePSF_ZNormalized() << endl;
        cout << "PSF::interpolatePSFThinPlateSplineChiSquare: zArr = " << zArr << endl;
      #endif
      ndarray::Array< PsfT, 1, 1 > xPos = ndarray::allocate( xPositions.getShape() );
      ndarray::Array< PsfT, 1, 1 > yPos = ndarray::allocate( yPositions.getShape() );
      auto itXPos = xPos.begin();
      for (auto itX = xPositions.begin(); itX != xPositions.end(); ++itX, ++itXPos ){
        *itXPos = PsfT(*itX);
      }
      auto itYPos = yPos.begin();
      for (auto itY = yPositions.begin(); itY != yPositions.end(); ++itY, ++itYPos ){
        *itYPos = PsfT(*itY);
      }
//      ndarray::Array< PsfT, 1, 1 > zT = ndarray::allocate(zArr.size());
//      auto it = zArr.begin();
//      for (auto itT = zT.begin(); itT != zT.end(); ++itT, ++it)
//        *itT = double(*it);
//      #ifdef __DEBUG_CALC_TPS__
//        cout << "PSF::interpolatePSFThinPlateSplineChiSquare: zT = " << zT << endl;
//      #endif
      cout << "PSF::interpolatePSFThinPlateSplineChiSquare: xArr = " << xArr << endl;
      cout << "PSF::interpolatePSFThinPlateSplineChiSquare: yArr = " << yArr << endl;
      cout << "PSF::interpolatePSFThinPlateSplineChiSquare: zArr = " << zArr << endl;
      cout << "PSF::interpolatePSFThinPlateSplineChiSquare: xPos = " << xPos << endl;
      cout << "PSF::interpolatePSFThinPlateSplineChiSquare: yPos = " << yPos << endl;
      cout << "PSF::interpolatePSFThinPlateSplineChiSquare: starting interpolateThinPlateSplineChiSquare" << endl;
      TPSControl tpsControl;
      tpsControl.regularization = regularization;
      tpsControl.shapeParameter = shapeParameter;
      math::ThinPlateSplineChiSquare< PsfT, PsfT > tps = math::ThinPlateSplineChiSquare< PsfT, PsfT >( xArr, 
                                                                                                       yArr, 
                                                                                                       zArr,
                                                                                                       xPos,
                                                                                                       yPos,
                                                                                                       isXYPositionsGridPoints,
                                                                                                       tpsControl );
//                                                                                                    ndarray::Array< PsfT const, 1, 1 >(zT), 
//                                                                                       regularization );
      ndarray::Array< PsfT, 2, 1 > arr_Out = ndarray::copy( tps.fitArray( xPos, 
                                                                          yPos, 
                                                                          isXYPositionsGridPoints ) );
      ndarray::Array< PsfT, 1, 1 > zRec = ndarray::copy( tps.getZFit() );
      if ( !psf.setImagePSF_ZFit( zRec ) ){
        cout << "PSF::interpolatePSFThinPlateSplineChiSquare: WARNING: psf.setImagePSF_ZFit( zRec ) returned FALSE" << endl;
      }
      psf.setThinPlateSplineChiSquare( tps );
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFThinPlateSplineChiSquare(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization) finished" << endl;
      #endif
      return arr_Out;
    }
    
    template < typename PsfT, typename CoordsT >
    ndarray::Array< PsfT, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< PsfT > & psfSet,
                                                                  ndarray::Array< CoordsT, 1, 1 > const& xPositions,
                                                                  ndarray::Array< CoordsT, 1, 1 > const& yPositions,
                                                                  bool const isXYPositionsGridPoints,
                                                                  double const regularization){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFSetThinPlateSpline(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization) started" << endl;
      #endif
      ndarray::Array< PsfT, 3, 1 > arrOut = ndarray::allocate(yPositions.getShape()[0], xPositions.getShape()[0], psfSet.size());
      for (size_t i = 0; i < psfSet.size(); ++i){
        #ifdef __DEBUG_CALC_TPS__
          if (psfSet.getPSF(i)->getImagePSF_XRelativeToCenter().size() < 3){
            string message("PSF::InterPolatePSFSetThinPlateSpline: ERROR: i=");
            message += to_string(i) + ": imagePSF_XRelativeToCenter().size()=" + to_string(psfSet.getPSF(i)->getImagePSF_XRelativeToCenter().size()) + " < 3";
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
        ndarray::Array< PsfT, 2, 1 > arr = ndarray::allocate(yPositions.getShape()[0], xPositions.getShape()[0]);
        arr.deep() = interpolatePSFThinPlateSpline( *( psfSet.getPSF( i ) ), 
                                                    xPositions, 
                                                    yPositions, 
                                                    isXYPositionsGridPoints, 
                                                    regularization );
        #ifdef __DEBUG_CALC_TPS__
          cout << "interpolatePSFSetThinPlateSpline: arr.getShape() = " << arr.getShape() << ", arrOut.getShape() = " << arrOut.getShape() << endl;
          cout << "interpolatePSFSetThinPlateSpline: arr = " << arr << endl;
        #endif
        arrOut[ndarray::view()()(i)].deep() = arr;
//        PTR(ndarray::Array<ImageT, 2, 1>) pArr(new ndarray::Array<ImageT, 2, 1>(arr));
//        vecOut.push_back(arr);
      }
      #ifdef __DEBUG_CALC_TPS__
        cout << "interpolatePSFSetThinPlateSpline: arrOut[:][:][0] = " << arrOut[ndarray::view()()(0)] << endl;
      #endif
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFSetThinPlateSpline(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization) finished" << endl;
      #endif
      return arrOut;
    }
    
    template < typename PsfT, typename WeightT, typename CoordsT >
    ndarray::Array< PsfT, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< PsfT > & psfSet,
                                                                  ndarray::Array< WeightT, 2, 1 > const& weightArr,
                                                                  ndarray::Array< CoordsT, 1, 1 > const& xPositions,
                                                                  ndarray::Array< CoordsT, 1, 1 > const& yPositions,
                                                                  bool const isXYPositionsGridPoints){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFSetThinPlateSpline(psf, weights, xPositions, yPositions, isXYPositionsGridPoints) started" << endl;
      #endif
      ndarray::Array< PsfT, 3, 1 > arrOut = ndarray::allocate(yPositions.getShape()[0], xPositions.getShape()[0], psfSet.size());
      for (size_t i = 0; i < psfSet.size(); ++i){
        #ifdef __DEBUG_CALC_TPS__
        if (psfSet.getPSF(i)->getImagePSF_XRelativeToCenter().size() < 3){
          string message("PSF::InterPolatePSFSetThinPlateSpline: ERROR: i=");
          message += to_string(i) + ": imagePSF_XRelativeToCenter().size()=" + to_string(psfSet.getPSF(i)->getImagePSF_XRelativeToCenter().size()) + " < 3";
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        #endif
        size_t size = psfSet.getPSF(i)->getImagePSF_ZNormalized().size();
        ndarray::Array< WeightT, 1, 1 > weights = ndarray::allocate(size);
        weights.deep() = weightArr[ndarray::view(0, size)(i)];
        ndarray::Array< PsfT, 2, 1 > arr = ndarray::copy(interpolatePSFThinPlateSpline(*(psfSet.getPSF(i)), 
                                                                                       weights,
                                                                                       xPositions, 
                                                                                       yPositions,
                                                                                       isXYPositionsGridPoints ));
        #ifdef __DEBUG_CALC_TPS__
          cout << "interpolatePSFSetThinPlateSpline: arr.getShape() = " << arr.getShape() << ", arrOut.getShape() = " << arrOut.getShape() << endl;
          cout << "interpolatePSFSetThinPlateSpline: arr = " << arr << endl;
        #endif
        arrOut[ndarray::view()()(i)].deep() = arr;
//        PTR(ndarray::Array<ImageT, 2, 1>) pArr(new ndarray::Array<ImageT, 2, 1>(arr));
//        vecOut.push_back(arr);
      }
      #ifdef __DEBUG_CALC_TPS__
        cout << "interpolatePSFSetThinPlateSpline: arrOut[:][:][0] = " << arrOut[ndarray::view()()(0)] << endl;
      #endif
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFSetThinPlateSpline(psf, weights, xPositions, yPositions, isXYPositionsGridPoints) started" << endl;
      #endif
      return arrOut;
    }
    
    template< typename PsfT, typename CoordT >
    ndarray::Array< PsfT, 2, 1 > collapseFittedPSF( ndarray::Array< CoordT, 1, 1 > const& xGridVec_In,
                                                    ndarray::Array< CoordT, 1, 1 > const& yGridVec_In,
                                                    ndarray::Array< PsfT, 2, 1 > const& zArr_In,
                                                    int const direction){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::collapseFittedPSF(xGridVec, yGridVec, zArr, direction) started" << endl;
      #endif
      if (xGridVec_In.getShape()[0] != zArr_In.getShape()[1]){
        string message("PSF::math::collapseFittedPSF: ERROR: xGridVec_In.getShape()[0]=");
        message += to_string(xGridVec_In.getShape()[0]) + " != zArr_In.getShape()[1]=" + to_string(zArr_In.getShape()[1]);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (yGridVec_In.getShape()[0] != zArr_In.getShape()[0]){
        string message("PSF::math::collapseFittedPSF: ERROR: yGridVec_In.getShape()[0]=");
        message += to_string(yGridVec_In.getShape()[0]) + " != zArr_In.getShape()[0]=" + to_string(zArr_In.getShape()[0]);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      size_t collapsedPSFLength;
      ndarray::Array< PsfT, 1, 1 > tempVec;
      ndarray::Array< PsfT, 2, 1 > collapsedPSF;
      if (direction == 0){
        collapsedPSFLength = yGridVec_In.getShape()[0];
        tempVec = ndarray::allocate(xGridVec_In.getShape()[0]);
        collapsedPSF = ndarray::allocate(collapsedPSFLength, 2);
        collapsedPSF[ndarray::view()(0)].deep() = yGridVec_In;
      }
      else{
        collapsedPSFLength = xGridVec_In.getShape()[0];
        tempVec = ndarray::allocate(yGridVec_In.getShape()[0]);
        collapsedPSF = ndarray::allocate(collapsedPSFLength, 2);
        collapsedPSF[ndarray::view()(0)].deep() = xGridVec_In;
      }
      for (size_t iPos = 0; iPos < collapsedPSFLength; ++iPos){
        assert(iPos < std::numeric_limits<int>::max());
        if (direction == 0){
          tempVec.deep() = zArr_In[ndarray::view(static_cast<int>(iPos))()];
        }
        else{
          tempVec.deep() = zArr_In[ndarray::view()(static_cast<int>(iPos))];
        }
        collapsedPSF[ ndarray::makeVector( static_cast<int>(iPos), 1 ) ] = tempVec.asEigen().array().sum();
      }
      #ifdef __DEBUG_PSF__
        cout << "psfMath::collapseFittedPSF(xGridVec, yGridVec, zArr, direction) finished" << endl;
      #endif
      return collapsedPSF;
    }
    
    template< typename PsfT, typename CoordT >
    ndarray::Array< PsfT, 2, 1 > collapsePSF( PSF< PsfT > const& psf_In,
                                               ndarray::Array< CoordT, 1, 1 > const& coordinatesX_In,
                                               ndarray::Array< CoordT, 1, 1 > const& coordinatesY_In,
                                               int const direction,
                                               double const regularization){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::collapsePSF(, psf, coordinatesX, coordinatesY, direction, regularization) started" << endl;
      #endif
      ndarray::Array< CoordT, 1, 1 > xRelativeToCenter = ndarray::allocate(psf_In.getImagePSF_XRelativeToCenter().size());//ndarray::external(psf_In.getImagePSF_XRelativeToCenter().data(), ndarray::makeVector(int(psf_In.getImagePSF_XRelativeToCenter().size())), ndarray::makeVector(1));
      ndarray::Array< CoordT, 1, 1 > yRelativeToCenter = ndarray::allocate(psf_In.getImagePSF_YRelativeToCenter().size());//ndarray::external(psf_In.getImagePSF_YRelativeToCenter().data(), ndarray::makeVector(int(psf_In.getImagePSF_YRelativeToCenter().size())), ndarray::makeVector(1));
      auto itXVec = psf_In.getImagePSF_XRelativeToCenter().begin();
      auto itYVec = psf_In.getImagePSF_XRelativeToCenter().begin();
      for (auto itXArr = xRelativeToCenter.begin(), itYArr = yRelativeToCenter.begin(); itXArr != xRelativeToCenter.end(); ++itXVec, ++itYVec, ++itXArr, ++itYArr){
        *itXArr = CoordT(*itXVec);
        *itYArr = CoordT(*itYVec);
      }
      TPSControl tpsControl;
      tpsControl.regularization = regularization;
      ndarray::Array< const PsfT, 1, 1 > zNormalized = ndarray::external(psf_In.getImagePSF_ZNormalized().data(), ndarray::makeVector(int(psf_In.getImagePSF_ZNormalized().size())), ndarray::makeVector(1));
      math::ThinPlateSpline< PsfT, CoordT > tpsA = math::ThinPlateSpline< PsfT, CoordT >( xRelativeToCenter,
                                                                                          yRelativeToCenter,
                                                                                          zNormalized,
                                                                                          tpsControl );
      ndarray::Array< PsfT, 2, 1 > interpolatedPSF = tpsA.fitArray( coordinatesX_In,
                                                                    coordinatesY_In,
                                                                    true);
      #ifdef __DEBUG_PSF__
        cout << "psfMath::collapsePSF(psf, coordinatesX, coordinatesY, direction, regularization) finishing" << endl;
      #endif
      return math::collapseFittedPSF(coordinatesX_In,
                                     coordinatesY_In,
                                     interpolatedPSF,
                                     direction);
    }
    
    template< typename T >
    ndarray::Array< T, 2, 1> calcPositionsRelativeToCenter(T const centerPos_In,
                                                           T const width_In){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::calcPositionsRelativeToCenter(centerPos, width) started" << endl;
      #endif
      #ifdef __DEBUG_CPRTC__
        cout << "PSF::math::calcPositionsRelativeToCenter: centerPos_In = " << centerPos_In << ", width_In = " << width_In << endl;
      #endif
      size_t low = std::floor(centerPos_In - (width_In / 2.));
      size_t high = std::floor(centerPos_In + (width_In / 2.));
      #ifdef __DEBUG_CPRTC__
        cout << "PSF::math::calcPositionsRelativeToCenter: low = " << low << ", high = " << high << endl;
      #endif
      ndarray::Array< T, 2, 1 > arr_Out = ndarray::allocate(high - low + 1, 2);
      double dCenter = PIXEL_CENTER - centerPos_In + std::floor(centerPos_In);
      #ifdef __DEBUG_CPRTC__
        cout << "PSF::math::calcPositionsRelativeToCenter: dCenter = " << dCenter << endl;
      #endif
      for (size_t iPos = 0; iPos <= high - low; ++iPos){
        assert(iPos < std::numeric_limits<int>::max()); // safe to cast
        arr_Out[ ndarray::makeVector(static_cast<int>(iPos), 0)] = T(low + iPos);
        arr_Out[ ndarray::makeVector(static_cast<int>(iPos), 1)] =
            arr_Out[ ndarray::makeVector(static_cast<int>(iPos), 0 ) ] -
            T(std::floor(centerPos_In)) + dCenter;
        #ifdef __DEBUG_CPRTC__
          cout << "PSF::math::calcPositionsRelativeToCenter: arr_Out[" << iPos << "][0] = "
               << arr_Out[ndarray::makeVector(static_cast<int>(iPos), 0 ) ] << ", arr_Out[" << iPos
               << "][1] = " << arr_Out[ndarray::makeVector(static_cast<int>(iPos), 1 ) ] << endl;
        #endif
      }
      #ifdef __DEBUG_CPRTC__
        cout << "PSF::math::calcPositionsRelativeToCenter: arr_Out = " << arr_Out << endl;
      #endif
      #ifdef __DEBUG_PSF__
        cout << "psfMath::calcPositionsRelativeToCenter(centerPos, width) finished" << endl;
      #endif
      return arr_Out;
    }


    template< typename PsfT, typename CoordsT >
    ndarray::Array< CoordsT, 2, 1 > compareCenterPositions(PSF< PsfT > & psf,
                                                           ndarray::Array< const CoordsT, 1, 1 > const& xPositions,
                                                           ndarray::Array< const CoordsT, 1, 1 > const& yPositions,
                                                           float dXMax,
                                                           float dYMax,
                                                           bool setPsfXY){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::compareCenterPositions(psf, xPositions, yPositions, dXMax, dYMax, setPsfXY) started" << endl;
      #endif
      if (xPositions.getShape()[0] != yPositions.getShape()[0]){
        string message("pfs::drp::stella::math::compareCenterPositions: ERROR: xPositions.getShape()[0]=");
        message += to_string(xPositions.getShape()[0]) + " != yPositions.getShape()[1]=" + to_string(yPositions.getShape()[1]);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_COMPARECENTERPOSITIONS__
        cout << "pfs::drp::stella::math::compareCenterPositions: xPosititions = " << xPositions.getShape()[0] << ": " << xPositions << endl;
        cout << "pfs::drp::stella::math::compareCenterPositions: yPosititions = " << yPositions.getShape()[0] << ": " << yPositions << endl;
      #endif
      size_t nPSFs = psf.getXCentersPSFCCD().size();
      #ifdef __DEBUG_COMPARECENTERPOSITIONS__
        cout << "pfs::drp::stella::math::compareCenterPositions: nPSFs = " << nPSFs << endl;
      #endif
      ndarray::Array< CoordsT, 2, 1 > dXdYdR_Out = ndarray::allocate(nPSFs, 3);
      dXdYdR_Out.deep() = -1.;
      CoordsT xPSF, yPSF, dX, dY, dR;
      size_t iList;
      bool found;
      std::vector< PsfT > xCenters_Out;
      std::vector< PsfT > yCenters_Out;
      for (size_t iPSF = 0; iPSF < nPSFs; ++iPSF){
        xPSF = psf.getXCentersPSFCCD()[iPSF];
        yPSF = psf.getYCentersPSFCCD()[iPSF];
        found = false;
        iList = 0;
        while ((!found) && (iList < static_cast<size_t>(xPositions.getShape()[0]))){
          dX = xPositions[iList] - xPSF;
          #ifdef __DEBUG_COMPARECENTERPOSITIONS__
            cout << "pfs::drp::stella::math::compareCenterPositions: spot number " << iPSF << ": xPSF = " << xPSF << ", xPosititions[" << iList << "] = " << xPositions[iList] << ": dX = " << dX << endl;
          #endif
          if (std::fabs(dX) < dXMax){
            dY = yPositions[iList] - yPSF;
            #ifdef __DEBUG_COMPARECENTERPOSITIONS__
              cout << "pfs::drp::stella::math::compareCenterPositions: spot number " << iPSF << " yPosititions[" << iList << "] = " << yPositions[iList] << ": dY = " << dY << endl;
            #endif
            if (std::fabs(dY) < dYMax){
              dR = sqrt(pow(dX, 2) + pow(dY, 2));
              dXdYdR_Out[ ndarray::makeVector( int( iPSF ), 0 ) ] = dX;
              dXdYdR_Out[ ndarray::makeVector( int( iPSF ), 1 ) ] = dY;
              dXdYdR_Out[ ndarray::makeVector( int( iPSF ), 2 ) ] = dR;
              found = true;
              if (setPsfXY){
                xCenters_Out.push_back(xPositions[iList]);
                yCenters_Out.push_back(yPositions[iList]);
              }
              #ifdef __DEBUG_COMPARECENTERPOSITIONS__
                cout << "pfs::drp::stella::math::compareCenterPositions: spot number " << iPSF << " found in input lists at position " << iList << endl;
              #endif
            }
          }
          ++iList;
          #ifdef __DEBUG_COMPARECENTERPOSITIONS__
            cout << "pfs::drp::stella::math::compareCenterPositions: spot number " << iPSF << " iList = " << iList << endl;
          #endif
//          if (iList == xPositions.getShape()[0])
//            break;
        }
        if (!found){
          cout << "pfs::drp::stella::math::compareCenterPositions: WARNING: spot number " << iPSF << " not found in input lists" << endl;
          xCenters_Out.push_back(xPSF);
          yCenters_Out.push_back(yPSF);
        }
      }
      if (setPsfXY){
        if (!psf.setXCentersPSFCCD(xCenters_Out)){
          string message("pfs::drp::stella::math::compareCenterPositions: ERROR: psf.setXCentersPSFCCD(xCenters_Out) returned FALSE");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        if (!psf.setYCentersPSFCCD(yCenters_Out)){
          string message("pfs::drp::stella::math::compareCenterPositions: ERROR: psf.setYCentersPSFCCD(yCenters_Out) returned FALSE");
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
      }
      #ifdef __DEBUG_PSF__
        cout << "psfMath::compareCenterPositions(psf, xPositions, yPositions, dXMax, dYMax, setPsfXY) finished" << endl;
      #endif
      return dXdYdR_Out;
    }
    
    template ndarray::Array< float, 2, 1 > compareCenterPositions(PSF< float > &,
                                                                  ndarray::Array< const float, 1, 1 > const&,
                                                                  ndarray::Array< const float, 1, 1 > const&,
                                                                  float,
                                                                  float,
                                                                  bool);
    template ndarray::Array< float, 2, 1 > compareCenterPositions(PSF< double > &,
                                                                  ndarray::Array< const float, 1, 1 > const&,
                                                                  ndarray::Array< const float, 1, 1 > const&,
                                                                  float,
                                                                  float,
                                                                  bool);
    template ndarray::Array< double, 2, 1 > compareCenterPositions(PSF< float > &,
                                                                   ndarray::Array< const double, 1, 1 > const&,
                                                                   ndarray::Array< const double, 1, 1 > const&,
                                                                   float,
                                                                   float,
                                                                   bool);
    template ndarray::Array< double, 2, 1 > compareCenterPositions(PSF< double > &,
                                                                   ndarray::Array< const double, 1, 1 > const&,
                                                                   ndarray::Array< const double, 1, 1 > const&,
                                                                   float,
                                                                   float,
                                                                   bool);
    
    template ndarray::Array< float, 2, 1 > calcPositionsRelativeToCenter(float const, float const);
    template ndarray::Array< double, 2, 1 > calcPositionsRelativeToCenter(double const, double const);

    template ndarray::Array< float, 2, 1 > collapseFittedPSF( ndarray::Array< float, 1, 1 > const&,
                                                              ndarray::Array< float, 1, 1 > const&,
                                                              ndarray::Array< float, 2, 1 > const&,
                                                              int const);
    template ndarray::Array< double, 2, 1 > collapseFittedPSF( ndarray::Array< float, 1, 1 > const&,
                                                               ndarray::Array< float, 1, 1 > const&,
                                                               ndarray::Array< double, 2, 1 > const&,
                                                               int const);
    template ndarray::Array< float, 2, 1 > collapseFittedPSF( ndarray::Array< double, 1, 1 > const&,
                                                              ndarray::Array< double, 1, 1 > const&,
                                                              ndarray::Array< float, 2, 1 > const&,
                                                              int const);
    template ndarray::Array< double, 2, 1 > collapseFittedPSF( ndarray::Array< double, 1, 1 > const&,
                                                               ndarray::Array< double, 1, 1 > const&,
                                                               ndarray::Array< double, 2, 1 > const&,
                                                               int const);

    template ndarray::Array< float, 2, 1 > collapsePSF( PSF< float > const&,
                                                        ndarray::Array< double, 1, 1 > const&,
                                                        ndarray::Array< double, 1, 1 > const&,
                                                        int const,
                                                        double const);
    template ndarray::Array< double, 2, 1 > collapsePSF( PSF< double > const&,
                                                        ndarray::Array< double, 1, 1 > const&,
                                                        ndarray::Array< double, 1, 1 > const&,
                                                        int const,
                                                        double const);
    template ndarray::Array< float, 2, 1 > collapsePSF( PSF< float > const&,
                                                        ndarray::Array< float, 1, 1 > const&,
                                                        ndarray::Array< float, 1, 1 > const&,
                                                        int const,
                                                        double const);
    template ndarray::Array< double, 2, 1 > collapsePSF( PSF< double > const&,
                                                        ndarray::Array< float, 1, 1 > const&,
                                                        ndarray::Array< float, 1, 1 > const&,
                                                        int const,
                                                        double const);

/*    template ndarray::Array< float, 2, 1 > interpolatePSFThinPlateSpline(PSF< float > &, 
                                                                         ndarray::Array< float, 1, 1 > const&, 
                                                                         ndarray::Array< float, 1, 1 > const&,
                                                                         bool const,
                                                                         double const,
                                                                         float const,
                                                                         unsigned short const);
    template ndarray::Array< double, 2, 1 > interpolatePSFThinPlateSpline(PSF< double > &, 
                                                                          ndarray::Array<float, 1, 1> const&, 
                                                                          ndarray::Array<float, 1, 1> const&,
                                                                          bool const,
                                                                          double const,
                                                                          double const,
                                                                          unsigned short const);
    template ndarray::Array< float, 2, 1 > interpolatePSFThinPlateSpline(PSF< float > &, 
                                                                         ndarray::Array< double, 1, 1 > const&, 
                                                                         ndarray::Array< double, 1, 1 > const&,
                                                                         bool const,
                                                                         double const,
                                                                         float const,
                                                                         unsigned short const);*/
    template ndarray::Array< double, 2, 1 > interpolatePSFThinPlateSpline(PSF< double > &, 
                                                                          ndarray::Array< double, 1, 1 > const&, 
                                                                          ndarray::Array< double, 1, 1 > const&,
                                                                          bool const,
                                                                          double const,
                                                                          double const,
                                                                          unsigned short const);

/*    template ndarray::Array<float, 2, 1> interpolatePSFThinPlateSpline(PSF< float > &, 
                                                                       ndarray::Array< float, 1, 1 > const&, 
                                                                       ndarray::Array< float, 1, 1 > const&, 
                                                                       ndarray::Array< float, 1, 1 > const&,
                                                                       bool const,
                                                                       float const,
                                                                       unsigned short const);
    template ndarray::Array<double, 2, 1> interpolatePSFThinPlateSpline(PSF< double > &, 
                                                                        ndarray::Array< float, 1, 1 > const&, 
                                                                        ndarray::Array< float, 1, 1 > const&, 
                                                                        ndarray::Array< float, 1, 1 > const&,
                                                                        bool const,
                                                                        double const,
                                                                        unsigned short const);
    template ndarray::Array<float, 2, 1> interpolatePSFThinPlateSpline(PSF< float > &, 
                                                                       ndarray::Array< double, 1, 1 > const&, 
                                                                       ndarray::Array< float, 1, 1 > const&, 
                                                                       ndarray::Array< float, 1, 1 > const&,
                                                                       bool const,
                                                                       float const,
                                                                       unsigned short const);
    template ndarray::Array<double, 2, 1> interpolatePSFThinPlateSpline(PSF< double > &, 
                                                                        ndarray::Array< double, 1, 1 > const&, 
                                                                        ndarray::Array< float, 1, 1 > const&, 
                                                                        ndarray::Array< float, 1, 1 > const&,
                                                                        bool const,
                                                                        double const,
                                                                        unsigned short const);
    template ndarray::Array<float, 2, 1> interpolatePSFThinPlateSpline(PSF< float > &, 
                                                                       ndarray::Array< float, 1, 1 > const&, 
                                                                       ndarray::Array< double, 1, 1 > const&, 
                                                                       ndarray::Array< double, 1, 1 > const&,
                                                                       bool const,
                                                                       float const,
                                                                       unsigned short const);
    template ndarray::Array<double, 2, 1> interpolatePSFThinPlateSpline(PSF< double > &, 
                                                                        ndarray::Array< float, 1, 1 > const&, 
                                                                        ndarray::Array< double, 1, 1 > const&, 
                                                                        ndarray::Array< double, 1, 1 > const&,
                                                                        bool const,
                                                                        double const,
                                                                        unsigned short const);
    template ndarray::Array<float, 2, 1> interpolatePSFThinPlateSpline(PSF< float > &, 
                                                                       ndarray::Array< double, 1, 1 > const&, 
                                                                       ndarray::Array< double, 1, 1 > const&, 
                                                                       ndarray::Array< double, 1, 1 > const&,
                                                                       bool const,
                                                                       float const,
                                                                       unsigned short const);*/
    template ndarray::Array<double, 2, 1> interpolatePSFThinPlateSpline(PSF< double > &, 
                                                                        ndarray::Array< double, 1, 1 > const&, 
                                                                        ndarray::Array< double, 1, 1 > const&, 
                                                                        ndarray::Array< double, 1, 1 > const&,
                                                                        bool const,
                                                                        double const,
                                                                        unsigned short const);

//    template ndarray::Array< float, 2, 1 > interpolatePSFThinPlateSplineChiSquare( PSF< float > &, 
//                                                                                   ndarray::Array< float, 1, 1 > const&, 
//                                                                                   ndarray::Array< float, 1, 1 > const& );
//    template ndarray::Array< double, 2, 1 > interpolatePSFThinPlateSplineChiSquare( PSF< double > &, 
//                                                                                    ndarray::Array<float, 1, 1> const&, 
//                                                                                    ndarray::Array<float, 1, 1> const& );
//    template ndarray::Array< float, 2, 1 > interpolatePSFThinPlateSplineChiSquare( PSF< float > &, 
//                                                                                   ndarray::Array< double, 1, 1 > const&, 
//                                                                                   ndarray::Array< double, 1, 1 > const& );
    template ndarray::Array< double, 2, 1 > interpolatePSFThinPlateSplineChiSquare( PSF< double > &, 
                                                                                    ndarray::Array< double, 1, 1 > const&, 
                                                                                    ndarray::Array< double, 1, 1 > const&,
                                                                                    bool const,
                                                                                    double const,
                                                                                    double const,
                                                                                    unsigned short const );
    
    template ndarray::Array< float, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< float > &, 
                                                                            ndarray::Array< float, 1, 1 > const&, 
                                                                            ndarray::Array< float, 1, 1 > const&,
                                                                            bool const,
                                                                            double const);
    template ndarray::Array< double, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< double > &, 
                                                                             ndarray::Array< float, 1, 1 > const&, 
                                                                             ndarray::Array< float, 1, 1 > const&,
                                                                             bool const,
                                                                             double const);
    template ndarray::Array< float, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< float > &, 
                                                                            ndarray::Array< double, 1, 1 > const&, 
                                                                            ndarray::Array< double, 1, 1 > const&,
                                                                            bool const,
                                                                            double const);
    template ndarray::Array< double, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< double > &, 
                                                                             ndarray::Array< double, 1, 1 > const&, 
                                                                             ndarray::Array< double, 1, 1 > const&,
                                                                             bool const,
                                                                             double const);
    
    template ndarray::Array< float, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< float > &, 
                                                                            ndarray::Array< float, 2, 1 > const&, 
                                                                            ndarray::Array< float, 1, 1 > const&, 
                                                                            ndarray::Array< float, 1, 1 > const&,
                                                                            bool const);
    template ndarray::Array< double, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< double > &, 
                                                                             ndarray::Array< float, 2, 1 > const&, 
                                                                             ndarray::Array< float, 1, 1 > const&, 
                                                                             ndarray::Array< float, 1, 1 > const&,
                                                                             bool const);
    template ndarray::Array< float, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< float > &, 
                                                                            ndarray::Array< double, 2, 1 > const&, 
                                                                            ndarray::Array< float, 1, 1 > const&, 
                                                                            ndarray::Array< float, 1, 1 > const&,
                                                                            bool const);
    template ndarray::Array< double, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< double > &, 
                                                                             ndarray::Array< double, 2, 1 > const&, 
                                                                             ndarray::Array< float, 1, 1 > const&, 
                                                                             ndarray::Array< float, 1, 1 > const&,
                                                                             bool const);
    template ndarray::Array< float, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< float > &, 
                                                                            ndarray::Array< float, 2, 1 > const&, 
                                                                            ndarray::Array< double, 1, 1 > const&, 
                                                                            ndarray::Array< double, 1, 1 > const&,
                                                                            bool const);
    template ndarray::Array< double, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< double > &, 
                                                                             ndarray::Array< float, 2, 1 > const&, 
                                                                             ndarray::Array< double, 1, 1 > const&, 
                                                                             ndarray::Array< double, 1, 1 > const&,
                                                                             bool const);
    template ndarray::Array< float, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< float > &, 
                                                                            ndarray::Array< double, 2, 1 > const&, 
                                                                            ndarray::Array< double, 1, 1 > const&, 
                                                                            ndarray::Array< double, 1, 1 > const&,
                                                                            bool const);
    template ndarray::Array< double, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< double > &, 
                                                                             ndarray::Array< double, 2, 1 > const&, 
                                                                             ndarray::Array< double, 1, 1 > const&, 
                                                                             ndarray::Array< double, 1, 1 > const&,
                                                                             bool const);
    
    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&, 
                                                       Spectrum< float, unsigned short, float, float > const&,
                                                       TwoDPSFControl const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&, 
                                                        Spectrum< float, unsigned short, float, float > const&,
                                                        TwoDPSFControl const&);
    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&, 
                                                       Spectrum< double, unsigned short, float, float > const&,
                                                       TwoDPSFControl const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&, 
                                                        Spectrum< double, unsigned short, float, float > const&,
                                                        TwoDPSFControl const&);
    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&, 
                                                       Spectrum< float, unsigned short, float, double > const&,
                                                       TwoDPSFControl const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&, 
                                                        Spectrum< float, unsigned short, float, double > const&,
                                                        TwoDPSFControl const&);
    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&, 
                                                       Spectrum< double, unsigned short, float, double > const&,
                                                       TwoDPSFControl const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&, 
                                                        Spectrum< double, unsigned short, float, double > const&,
                                                        TwoDPSFControl const&);
    
    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&, 
                                                       Spectrum< float, unsigned short, float, float > const&,
                                                       TwoDPSFControl const&,
                                                        ndarray::Array< float, 2, 1 > const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&, 
                                                        Spectrum< float, unsigned short, float, float > const&,
                                                        TwoDPSFControl const&,
                                                        ndarray::Array< double, 2, 1 > const&);
    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&, 
                                                       Spectrum< double, unsigned short, float, float > const&,
                                                       TwoDPSFControl const&,
                                                        ndarray::Array< float, 2, 1 > const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&, 
                                                        Spectrum< double, unsigned short, float, float > const&,
                                                        TwoDPSFControl const&,
                                                        ndarray::Array< double, 2, 1 > const&);
    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&, 
                                                       Spectrum< float, unsigned short, float, double > const&,
                                                       TwoDPSFControl const&,
                                                        ndarray::Array< float, 2, 1 > const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&, 
                                                        Spectrum< float, unsigned short, float, double > const&,
                                                        TwoDPSFControl const&,
                                                        ndarray::Array< double, 2, 1 > const&);
    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&, 
                                                       Spectrum< double, unsigned short, float, double > const&,
                                                       TwoDPSFControl const&,
                                                        ndarray::Array< float, 2, 1 > const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&, 
                                                        Spectrum< double, unsigned short, float, double > const&,
                                                        TwoDPSFControl const&,
                                                        ndarray::Array< double, 2, 1 > const&);
    
    template std::vector<PTR(PSFSet< float >)> calculate2dPSFPerBin( FiberTraceSet< float, unsigned short, float > const&, 
                                                                     SpectrumSet< float, unsigned short, float, float > const&,
                                                                     TwoDPSFControl const&);
    template std::vector<PTR(PSFSet< double >)> calculate2dPSFPerBin( FiberTraceSet< float, unsigned short, float > const&, 
                                                                      SpectrumSet< float, unsigned short, float, float > const&,
                                                                      TwoDPSFControl const&);
    template std::vector<PTR(PSFSet< float >)> calculate2dPSFPerBin( FiberTraceSet< double, unsigned short, float > const&, 
                                                                     SpectrumSet< double, unsigned short, float, float > const&,
                                                                     TwoDPSFControl const&);
    template std::vector<PTR(PSFSet< double >)> calculate2dPSFPerBin( FiberTraceSet< double, unsigned short, float > const&, 
                                                                      SpectrumSet< double, unsigned short, float, float > const&,
                                                                      TwoDPSFControl const&);
    template std::vector<PTR(PSFSet< float >)> calculate2dPSFPerBin( FiberTraceSet< float, unsigned short, float > const&, 
                                                                     SpectrumSet< float, unsigned short, float, double > const&,
                                                                     TwoDPSFControl const&);
    template std::vector<PTR(PSFSet< double >)> calculate2dPSFPerBin( FiberTraceSet< float, unsigned short, float > const&, 
                                                                      SpectrumSet< float, unsigned short, float, double > const&,
                                                                      TwoDPSFControl const&);
    template std::vector<PTR(PSFSet< float >)> calculate2dPSFPerBin( FiberTraceSet< double, unsigned short, float > const&, 
                                                                     SpectrumSet< double, unsigned short, float, double > const&,
                                                                     TwoDPSFControl const&);
    template std::vector<PTR(PSFSet< double >)> calculate2dPSFPerBin( FiberTraceSet< double, unsigned short, float > const&, 
                                                                      SpectrumSet< double, unsigned short, float, double > const&,
                                                                      TwoDPSFControl const&);
  }

  template<> template<> bool PSF< float >::extractPSFFromCenterPositions(FiberTrace< float, unsigned short, float > const&,
                                                            ndarray::Array<float, 1, 1> const&,
                                                            ndarray::Array<float, 1, 1> const&);
  template<> template<> bool PSF< float >::extractPSFFromCenterPositions(FiberTrace< float, unsigned short, double > const&,
                                                            ndarray::Array<float, 1, 1> const&,
                                                            ndarray::Array<float, 1, 1> const&);
  template<> template<> bool PSF< float >::extractPSFFromCenterPositions(FiberTrace< double, unsigned short, float > const&,
                                                            ndarray::Array<float, 1, 1> const&,
                                                            ndarray::Array<float, 1, 1> const&);
  template<> template<> bool PSF< float >::extractPSFFromCenterPositions(FiberTrace< double, unsigned short, double > const&,
                                                            ndarray::Array<float, 1, 1> const&,
                                                            ndarray::Array<float, 1, 1> const&);
  template<> template<> bool PSF< double >::extractPSFFromCenterPositions(FiberTrace< float, unsigned short, float > const&,
                                                            ndarray::Array<double, 1, 1> const&,
                                                            ndarray::Array<double, 1, 1> const&);
  template<> template<> bool PSF< double >::extractPSFFromCenterPositions(FiberTrace< float, unsigned short, double > const&,
                                                            ndarray::Array<double, 1, 1> const&,
                                                            ndarray::Array<double, 1, 1> const&);
  template<> template<> bool PSF< double >::extractPSFFromCenterPositions(FiberTrace< double, unsigned short, float > const&,
                                                            ndarray::Array<double, 1, 1> const&,
                                                            ndarray::Array<double, 1, 1> const&);
  template<> template<> bool PSF< double >::extractPSFFromCenterPositions(FiberTrace< double, unsigned short, double > const&,
                                                            ndarray::Array<double, 1, 1> const&,
                                                            ndarray::Array<double, 1, 1> const&);
 
  template<> template<> bool PSF< float >::extractPSFFromCenterPositions(FiberTrace< float, unsigned short, float > const&);
  template<> template<> bool PSF< float >::extractPSFFromCenterPositions(FiberTrace< float, unsigned short, double > const&);
  template<> template<> bool PSF< float >::extractPSFFromCenterPositions(FiberTrace< double, unsigned short, float > const&);
  template<> template<> bool PSF< float >::extractPSFFromCenterPositions(FiberTrace< double, unsigned short, double > const&);
  template<> template<> bool PSF< double >::extractPSFFromCenterPositions(FiberTrace< float, unsigned short, float > const&);
  template<> template<> bool PSF< double >::extractPSFFromCenterPositions(FiberTrace< float, unsigned short, double > const&);
  template<> template<> bool PSF< double >::extractPSFFromCenterPositions(FiberTrace< double, unsigned short, float > const&);
  template<> template<> bool PSF< double >::extractPSFFromCenterPositions(FiberTrace< double, unsigned short, double > const&);
  
  template class PSFSet< float >;
  template class PSFSet< double >;

}}}
