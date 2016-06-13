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
  _spectrum = ndarray::allocate( length );
  _sky = ndarray::allocate( length );
  _mask = ndarray::allocate( length );
  _covar = ndarray::allocate( 5, length );
  _wavelength = ndarray::allocate( length );
  _dispersion = ndarray::allocate( length );
  _yLow = 0;
  _yHigh = length - 1;
  _nCCDRows = length;
}

//template< typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT >
//pfsDRPStella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT >::Spectrum( Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > const& spectrum,
//                                                                              size_t iTrace ) 

template class pfsDRPStella::Spectrum<float, unsigned int, float, float>;
template class pfsDRPStella::Spectrum<double, unsigned int, float, float>;
template class pfsDRPStella::Spectrum<float, unsigned short, float, float>;
template class pfsDRPStella::Spectrum<double, unsigned short, float, float>;
template class pfsDRPStella::Spectrum<float, int, float, float>;
template class pfsDRPStella::Spectrum<double, int, float, float>;
template class pfsDRPStella::Spectrum<double, int, double, double>;
template class pfsDRPStella::Spectrum<float, unsigned int, double, double>;
template class pfsDRPStella::Spectrum<double, unsigned int, double, double>;
template class pfsDRPStella::Spectrum<float, unsigned short, double, double>;
template class pfsDRPStella::Spectrum<double, unsigned short, double, double>;

//template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
//bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setSpectrum( ndarray::Array<SpectrumT, 1, 1> const& spectrum )

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setSky(const ndarray::Array<SpectrumT, 1, 1> & sky )
{
  /// Check length of input spectrum
  if (sky.getShape()[0] != _length){
    string message("pfsDRPStella::Spectrum::setSky: ERROR: spectrum->size()=");
    message += to_string(sky.getShape()[0]) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _sky.deep() = sky;
  return true;
}

//template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
//bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setVariance( ndarray::Array<VarianceT, 1, 1> const& variance )

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setCovar(const ndarray::Array<VarianceT, 2, 1> & covar )
{
  /// Check length of input covar
  if (covar.getShape()[ 1 ] != _length){
    string message("pfsDRPStella::Spectrum::setCovar: ERROR: covar->size()=");
    message += to_string( covar.getShape()[ 1 ] ) + string( " != _length=" ) + to_string( _length );
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _covar.deep() = covar;
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
bool pfsDRPStella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT >::setDispersion( ndarray::Array< WavelengthT, 1, 1 > const& dispersion )
{
  /// Check length of input wavelength
  if ( dispersion.getShape()[ 0 ] != _length ){
    string message("pfsDRPStella::Spectrum::setDispersion: ERROR: dispersion->size()=");
    message += to_string( dispersion.getShape()[ 0 ]) + string(" != _length=") + to_string( _length );
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _dispersion.deep() = dispersion;
  return true;
}

//template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
//bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setMask(const ndarray::Array<MaskT, 1, 1> & mask)

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setLength(const size_t length){
  #ifdef __DEBUG_SETLENGTH__
    cout << "pfsDRPStella::Spectrum::setLength: starting to set _length to " << length << endl;
  #endif
  pfsDRPStella::math::resize(_spectrum, length);
  #ifdef __DEBUG_SETLENGTH__
    cout << "pfsDRPStella::Spectrum::setLength: _spectrum resized to " << _spectrum.getShape()[0] << endl;
  #endif
  pfsDRPStella::math::resize(_mask, length);
  #ifdef __DEBUG_SETLENGTH__
    cout << "pfsDRPStella::Spectrum::setLength: _mask resized to " << _mask.getShape()[0] << endl;
  #endif
  pfsDRPStella::math::resize(_covar, 5, length);
  #ifdef __DEBUG_SETLENGTH__
    cout << "pfsDRPStella::Spectrum::setLength: _covar resized to " << _covar.getShape()[0] << "x" << _covar.getShape()[1] << endl;
  #endif
  pfsDRPStella::math::resize(_wavelength, length);
  #ifdef __DEBUG_SETLENGTH__
    cout << "pfsDRPStella::Spectrum::setLength: _wavelength resized to " << _wavelength.getShape()[0] << endl;
  #endif
  if (length > _length){
    WavelengthT val = _wavelength[_length = 1];
    for (auto it = _wavelength.begin() + length; it != _wavelength.end(); ++it)
      *it = val;
  }
  _length = length;
  _yHigh = _yLow + _length - 1;
  #ifdef __DEBUG_SETLENGTH__
    cout << "pfsDRPStella::Spectrum::setLength: finishing: _length to " << _length << endl;
  #endif
  return true;
}

//template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
//bool pfsDRPStella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT >::setYLow( const size_t yLow )

//template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
//bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setYHigh(const size_t yHigh)

//template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
//bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setNCCDRows(const size_t nCCDRows)

//template< typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT >
//template< typename T >
//ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::hIdentify( ndarray::Array< T, 2, 1 > const& lineList )

//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<float, unsigned int, float, float>::hIdentify(ndarray::Array< float, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<double, unsigned int, float, float>::hIdentify(ndarray::Array< float, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<float, unsigned short, float, float>::hIdentify(ndarray::Array< float, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<double, unsigned short, float, float>::hIdentify(ndarray::Array< float, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<float, unsigned int, float, float>::hIdentify(ndarray::Array< double, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<double, unsigned int, float, float>::hIdentify(ndarray::Array< double, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<float, unsigned short, float, float>::hIdentify(ndarray::Array< double, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<double, unsigned short, float, float>::hIdentify(ndarray::Array< double, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<float, int, float, float>::hIdentify(ndarray::Array< float, 2, 1 > const& );

template< typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT >
template< typename T >
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::identify( ndarray::Array< T, 2, 1 > const& lineList,
                                                                                 ndarray::Array< T, 1, 0 > const& predicted,
                                                                                 ndarray::Array< T, 1, 0 > const& predictedWLenAllPix,
                                                                                 DispCorControl const& dispCorControl,
                                                                                 size_t nLinesCheck ){
  DispCorControl tempDispCorControl( dispCorControl );
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

  ndarray::Array< double, 1, 1 > predictedPos = ndarray::allocate( predicted.getShape()[ 0 ] );
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
  return _isWavelengthSet;
}
/**
 * Identify
 * Identifies calibration lines, given in D_A2_LineList_In the format [wlen, approx_pixel] in
 * wavelength-calibration spectrum D_A2_Spec_In [pixel_number, flux]
 * within the given position plus/minus I_Radius_In,
 * fits Gaussians to each line, fits Polynomial of order I_PolyFitOrder_In, and
 * _wavelength and PolyFit coefficients to _dispCoeffs
 */
template< typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT >
template< typename T >
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::identify( ndarray::Array< T, 2, 1 > const& lineList,
                                                                                 DispCorControl const& dispCorControl,
                                                                                 size_t nLinesCheck ){
  DispCorControl tempDispCorControl( dispCorControl );
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

namespace {

// Helper functions for SpectrumSet FITS ctor.

void checkExtType(
    lsst::afw::fits::Fits & fitsfile,
    PTR(lsst::daf::base::PropertySet) metadata,
    std::string const & expected
) {
    try {
        std::string exttype = boost::algorithm::trim_right_copy(metadata->getAsString("EXTTYPE"));
        if (exttype != "" && exttype != expected) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              (boost::format("Reading %s (hdu %d) Expected EXTTYPE==\"%s\", saw \"%s\"") %
                               expected % fitsfile.getFileName() % fitsfile.getHdu() % exttype).str());
        }
        metadata->remove("EXTTYPE");
    } catch(lsst::pex::exceptions::NotFoundError) {
        lsst::pex::logging::Log log(lsst::pex::logging::Log::getDefaultLog(), "afw.image.MaskedImage");
        log.warn(boost::format("Expected extension type not found: %s") % expected);
    }
}

void ensureMetadata(PTR(lsst::daf::base::PropertySet) & metadata) {
    if (!metadata) {
        metadata.reset(new lsst::daf::base::PropertyList());
    }
}

} // anonymous

///SpectrumSet
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::SpectrumSet( size_t nSpectra, size_t length )
        : _spectra()// std::vector< pfsDRPStella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > > )
{
  for (size_t i = 0; i < nSpectra; ++i){
    pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> spec( length );
    _spectra.push_back(spec);
    _spectra[i].setITrace( i );
  }
}
    
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::SpectrumSet( std::vector< Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT> > const& spectrumVector )
        ://     lsst::daf::base::Citizen(typeid(this)),
              _spectra(spectrumVector)
{}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::SpectrumSet( std::string const & fileName,
        PTR(lsst::daf::base::PropertySet) metadata,
        PTR(lsst::daf::base::PropertySet) fluxMetadata,
        PTR(lsst::daf::base::PropertySet) covarMetadata,
        PTR(lsst::daf::base::PropertySet) maskMetadata,
        PTR(lsst::daf::base::PropertySet) wLenMetadata,
        PTR(lsst::daf::base::PropertySet) wDispMetadata,
        PTR(lsst::daf::base::PropertySet) skyMetadata )
:
      _spectra()
{
    lsst::afw::fits::Fits fitsfile(fileName, "r", lsst::afw::fits::Fits::AUTO_CLOSE | lsst::afw::fits::Fits::AUTO_CHECK);
    *this = SpectrumSet(fitsfile, metadata,
                        fluxMetadata, covarMetadata, maskMetadata, wLenMetadata, wDispMetadata, skyMetadata);
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::SpectrumSet( lsst::afw::fits::MemFileManager const& manager,
        PTR(lsst::daf::base::PropertySet) metadata,
        PTR(lsst::daf::base::PropertySet) fluxMetadata,
        PTR(lsst::daf::base::PropertySet) covarMetadata,
        PTR(lsst::daf::base::PropertySet) maskMetadata,
        PTR(lsst::daf::base::PropertySet) wLenMetadata,
        PTR(lsst::daf::base::PropertySet) wDispMetadata,
        PTR(lsst::daf::base::PropertySet) skyMetadata )
://     lsst::daf::base::Citizen(typeid(this)),
      _spectra()
{
    lsst::afw::fits::Fits fitsfile(const_cast< lsst::afw::fits::MemFileManager& >(manager), "r", lsst::afw::fits::Fits::AUTO_CLOSE | lsst::afw::fits::Fits::AUTO_CHECK);
    *this = SpectrumSet(fitsfile, metadata, 
                        fluxMetadata, covarMetadata, maskMetadata, wLenMetadata, wDispMetadata, skyMetadata);
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::SpectrumSet(         
        lsst::afw::fits::Fits const& fitsfile,
        PTR(lsst::daf::base::PropertySet) metadata,
        PTR(lsst::daf::base::PropertySet) fluxMetadata,
        PTR(lsst::daf::base::PropertySet) covarMetadata,
        PTR(lsst::daf::base::PropertySet) maskMetadata,
        PTR(lsst::daf::base::PropertySet) wLenMetadata,
        PTR(lsst::daf::base::PropertySet) wDispMetadata,
        PTR(lsst::daf::base::PropertySet) skyMetadata
 )
://     lsst::daf::base::Citizen(typeid(this)),
      _spectra()
{
    lsst::pex::logging::Log log(lsst::pex::logging::Log::getDefaultLog(), "pfs.drp.stella.SpectrumSet");

/*    typedef boost::mpl::vector<
        unsigned char, 
        unsigned short, 
        short, 
        int,
        unsigned int,
        float,
        double,
        boost::uint64_t
    > fits_image_types;
*/    
    enum class Hdu {
        Primary = 1,
        Flux,
        Covariance,
        Mask,
        WLen,
        WDisp,
        Sky
    };

    // If the user has requested a non-default HDU and we require all HDUs, we fail.
    //if (needAllHdus && fitsfile.getHdu() > static_cast<int>(Hdu::Image)) {
    //    throw LSST_EXCEPT(lsst::afw::fits::FitsError,
    //                      "Cannot read all HDUs starting from non-default");
    //}

    auto origHdu = (const_cast< lsst::afw::fits::Fits& >(fitsfile)).getHdu();
    if (metadata) {
        // Read primary metadata - only if user asks for it.
        // If the primary HDU is not empty, this may be the same as imageMetadata.
        auto prevHdu = (const_cast< lsst::afw::fits::Fits& >(fitsfile)).getHdu();
        (const_cast< lsst::afw::fits::Fits& >(fitsfile)).setHdu(static_cast<int>(Hdu::Primary));
        (const_cast< lsst::afw::fits::Fits& >(fitsfile)).readMetadata(*metadata);
        (const_cast< lsst::afw::fits::Fits& >(fitsfile)).setHdu(prevHdu);
    }

    // setHdu(0) jumps to the first extension iff the primary HDU is both
    // empty and currently selected.
    (const_cast< lsst::afw::fits::Fits& >(fitsfile)).setHdu(0);
    ensureMetadata(fluxMetadata);
//    _image.reset(new Image(fitsfile, imageMetadata, bbox, origin));

    if (!metadata) {
        metadata.reset(new lsst::daf::base::PropertyList());
    }

//    const lsst::afw::geom::Box2I bbox();
//    ImageOrigin origin=PARENT;
    lsst::afw::geom::Point2I xy0;
    cout << "xy0 = " << xy0 << endl;
    ndarray::Array< float, 2, 2 > array;
    
    lsst::afw::image::fits_read_array( const_cast< lsst::afw::fits::Fits& >(fitsfile), array, xy0, *metadata );
    cout << "SpectrumSet::SpectrumSet(fitsfile): array.getShape() = " << array.getShape() << endl;

    for (int i = 0; i < metadata->names().size(); ++i){
      std::cout << "Image::Image(fitsfile, metadata,...): metadata.names()[" << i << "] = " << metadata->names()[i] << std::endl;
      if (metadata->names()[i].compare("EXPTIME") == 0)
        std::cout << "Image::Image(fitsfile, metadata,...): metadata->get(EXPTIME) = " << metadata->getAsDouble("EXPTIME") << std::endl;
    }
    for (int i = 0; i < metadata->paramNames().size(); ++i){
      std::cout << "Image::Image(fitsfile, metadata,...): metadata.paramNames()[" << i << "] = " << metadata->paramNames()[i] << std::endl;
      if (metadata->paramNames()[i].compare("EXPTIME") == 0)
        std::cout << "Image::Image(fitsfile, metadata,...): metadata->get(EXPTIME) = " << metadata->getAsDouble("EXPTIME") << std::endl;
    }
    for (int i = 0; i < metadata->propertySetNames().size(); ++i)
      std::cout << "Image::Image(fitsfile, metadata,...): metadata.propertySetNames()[" << i << "] = " << metadata->propertySetNames()[i] << std::endl;

    checkExtType( const_cast< lsst::afw::fits::Fits& >(fitsfile), fluxMetadata, "IMAGE" );

/*    if (fitsfile.getHdu() != static_cast<int>(Hdu::Image)) {
        // Reading the image from a non-default HDU means we do not attempt to
        // read mask and variance.
        _mask.reset(new Mask(_image->getBBox()));
        _variance.reset(new Variance(_image->getBBox()));
    } else {
        try {
            fitsfile.setHdu(static_cast<int>(Hdu::Mask));
            ensureMetadata(maskMetadata);
            _mask.reset(new Mask(fitsfile, maskMetadata, bbox, origin, conformMasks));
            checkExtType(fitsfile, maskMetadata, "MASK");
        } catch(lsst::afw::fits::FitsError &e) {
            if (needAllHdus) {
                LSST_EXCEPT_ADD(e, "Reading Mask");
                throw e;
            }
            log.warn("Mask unreadable; using default");
            // By resetting the status we are able to read the next HDU (the variance).
            fitsfile.status = 0;
            _mask.reset(new Mask(_image->getBBox()));
        }

        try {
            fitsfile.setHdu(static_cast<int>(Hdu::Variance));
            ensureMetadata(varianceMetadata);
            _variance.reset(new Variance(fitsfile, varianceMetadata, bbox, origin));
            checkExtType(fitsfile, varianceMetadata, "VARIANCE");
        } catch(lsst::afw::fits::FitsError &e) {
            if (needAllHdus) {
                LSST_EXCEPT_ADD(e, "Reading Variance");
                throw e;
            }
            log.warn("Variance unreadable; using default");
            fitsfile.status = 0;
            _variance.reset(new Variance(_image->getBBox()));
        }
    }*/
    (const_cast< lsst::afw::fits::Fits& >(fitsfile)).setHdu(origHdu);
}

template class pfsDRPStella::SpectrumSet<float, int, float, float>;
template class pfsDRPStella::SpectrumSet<double, int, double, double>;
template class pfsDRPStella::SpectrumSet<float, unsigned int, float, float>;
template class pfsDRPStella::SpectrumSet<double, unsigned int, float, float>;
template class pfsDRPStella::SpectrumSet<float, unsigned short, float, float>;
template class pfsDRPStella::SpectrumSet<double, unsigned short, float, float>;
template class pfsDRPStella::SpectrumSet<float, unsigned int, float, double>;
template class pfsDRPStella::SpectrumSet<double, unsigned int, float, double>;
template class pfsDRPStella::SpectrumSet<float, unsigned short, float, double>;
template class pfsDRPStella::SpectrumSet<double, unsigned short, float, double>;
template class pfsDRPStella::SpectrumSet<float, unsigned int, double, double>;
template class pfsDRPStella::SpectrumSet<double, unsigned int, double, double>;
template class pfsDRPStella::SpectrumSet<float, unsigned short, double, double>;
template class pfsDRPStella::SpectrumSet<double, unsigned short, double, double>;

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::setSpectrum( size_t const i,
                                                                                       Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> const& spectrum )
{
  if (i > _spectra.size()){
    string message("SpectrumSet::setSpectrum(i=");
    message += to_string(i) + "): ERROR: i > _spectra.size()=" + to_string(_spectra.size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  if (i == static_cast<int>(_spectra.size())){
    _spectra.push_back(spectrum);
  }
  else{
    _spectra[i] = spectrum;
  }
  return true;
}

/*template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
void pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::addSpectrum( Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> const& spectrum )
{
  _spectra.push_back( spectrum );
  return;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
void pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::addSpectrum( PTR(Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT> ) const& spectrum /// the Spectrum to add
)
{
  _spectra.push_back( *spectrum );
  return;
}*/
  
template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR( pfsDRPStella::Spectrum< ImageT, MaskT, VarianceT, WavelengthT > ) pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getSpectrum( const size_t i )
{
  if (i >= _spectra.size()){
    string message("SpectrumSet::getSpectrum(i=");
    message += to_string(i) + "): ERROR: i >= _spectra.size()=" + to_string(_spectra.size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  return PTR( pfsDRPStella::Spectrum< ImageT, MaskT, VarianceT, WavelengthT > )( new pfsDRPStella::Spectrum< ImageT, MaskT, VarianceT, WavelengthT >( _spectra.at( i ) ) ); 
}

//template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
//pfsDRPStella::Spectrum< ImageT, MaskT, VarianceT, WavelengthT > pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getSpectrum( const size_t i ) const 

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::erase(const size_t iStart, const size_t iEnd){
  if (iStart >= _spectra.size()){
    string message("SpectrumSet::erase(iStart=");
    message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iStart >= _spectra.size()=" + to_string(_spectra.size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  if (iEnd >= _spectra.size()){
    string message("SpectrumSet::erase(iStart=");
    message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iEnd >= _spectra.size()=" + to_string(_spectra.size());
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
  if (iStart == (_spectra.size()-1)){
    _spectra.pop_back();
  }
  else{
    if (iEnd == 0)
      _spectra.erase(_spectra.begin() + iStart);
    else
      _spectra.erase(_spectra.begin() + iStart, _spectra.begin() + iEnd);
  }
  return true;
}

    template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
    void pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::writeFits( std::string const& fileName, int flags ) const
    {
      lsst::afw::fits::Fits fitsfile( fileName, "w", lsst::afw::fits::Fits::AUTO_CLOSE | lsst::afw::fits::Fits::AUTO_CHECK );
      writeFits( fitsfile, flags );
    }

    template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
    void pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::writeFits( lsst::afw::fits::MemFileManager & manager, int flags ) const
    {
      lsst::afw::fits::Fits fitsfile(manager, "w", lsst::afw::fits::Fits::AUTO_CLOSE | lsst::afw::fits::Fits::AUTO_CHECK);
      writeFits( fitsfile, flags );
    }
    
    template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
    void pfsDRPStella::SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::writeFits( lsst::afw::fits::Fits & fitsfile, int flags ) const
    {
      int nFibers = int( size() );
      int nCCDRows = getSpectrum( 0 ).getNCCDRows();
      /// allocate memory for the arrays
      ndarray::Array< float, 2, 1 > flux = ndarray::allocate( nCCDRows, nFibers );
      ndarray::Array< float, 3, 1 > covar = ndarray::allocate( 5, nCCDRows, nFibers );
      ndarray::Array< int, 2, 1 > mask = ndarray::allocate( nCCDRows, nFibers );
      ndarray::Array< float, 2, 1 > wLen = ndarray::allocate( nCCDRows, nFibers );
      ndarray::Array< float, 2, 1 > wDisp = ndarray::allocate( nCCDRows, nFibers );
      ndarray::Array< float, 2, 1 > sky = ndarray::allocate( nCCDRows, nFibers );
      
      /// write data to arrays
      flux.deep() = 0.;
      covar.deep() = 0.;
      mask.deep() = 0;
      wLen.deep() = 0.;
      wDisp.deep() = 0.;
      sky.deep() = 0.;
      
      for ( int iFiber = 0; iFiber < _spectra.size(); ++iFiber ){
//        PTR( pfsDRPStella::Spectrum< ImageT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra[ iFiber ];

        int yLow = _spectra.at( iFiber ).getYLow();
        int yHigh = _spectra.at( iFiber ).getYHigh();
        if ( yHigh - yLow + 1 != _spectra.at( iFiber ).getSpectrum().getShape()[ 0 ] ){
          cout << "SpectrumSet::writeFits: yHigh=" << yHigh << " - yLow=" << yLow << " + 1 (=" << yHigh - yLow + 1 << ") = " << yHigh-yLow + 1 << " != _spectra.at( iFiber )->getSpectrum().getShape()[ 0 ] = " << _spectra.at( iFiber ).getSpectrum().getShape()[ 0 ] << endl;
          throw LSST_EXCEPT(
            lsst::pex::exceptions::LogicError,
            "SpectrumSet::writeFits: spectrum does not have expected shape"
          );
        }

        flux[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = _spectra.at( iFiber ).getSpectrum()[ ndarray::view() ];
        covar[ ndarray::view( )( yLow, yHigh + 1 )( iFiber ) ] = _spectra.at( iFiber ).getCovar()[ ndarray::view() ];
        mask[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = _spectra.at( iFiber ).getMask()[ ndarray::view() ];
        wLen[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = _spectra.at( iFiber ).getWavelength()[ ndarray::view() ];
        wDisp[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = _spectra.at( iFiber ).getDispersion()[ ndarray::view() ];
        sky[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = _spectra.at( iFiber ).getSky()[ ndarray::view() ];
      }

      CONST_PTR(lsst::daf::base::PropertySet) fluxMetadata = CONST_PTR(lsst::daf::base::PropertySet)();
      PTR(lsst::daf::base::PropertySet) hdr(new lsst::daf::base::PropertyList());
//      if (metadata) {
//          hdr = metadata->deepCopy();
//      } else {
//          hdr.reset(new lsst::daf::base::PropertyList());
//      }

      if (fitsfile.countHdus() != 0) {
          throw LSST_EXCEPT(
              lsst::pex::exceptions::LogicError,
              "SpectrumSet::writeFits::writeFits can only write to an empty file"
          );
      }
      if (fitsfile.getHdu() <= 1) {
          // Don't ever write images to primary; instead we make an empty primary.
          fitsfile.createEmpty();
      } else {
          fitsfile.setHdu(1);
      }
      fitsfile.writeMetadata(*hdr);

//      processPlaneMetadata( fluxMetadata, hdr, "FLUX" );
      cout << "writing flux" << endl;
      utils::fits_write_ndarray( fitsfile, flux, fluxMetadata );

//      processPlaneMetadata( fluxMetadata, hdr, "COVARIANCE" );
      cout << "writing covar" << endl;
      utils::fits_write_ndarray( fitsfile, covar, fluxMetadata );

//      processPlaneMetadata( fluxMetadata, hdr, "MASK" );
      cout << "writing mask" << endl;
      utils::fits_write_ndarray( fitsfile, mask, fluxMetadata );

//      processPlaneMetadata( fluxMetadata, hdr, "WAVELENGTH" );
      cout << "writing wLen" << endl;
      utils::fits_write_ndarray( fitsfile, wLen, fluxMetadata );

//      processPlaneMetadata( fluxMetadata, hdr, "DISPERSION" );
      cout << "writing wDisp" << endl;
      utils::fits_write_ndarray( fitsfile, wDisp, fluxMetadata );

//      processPlaneMetadata( fluxMetadata, hdr, "SKY" );
      cout << "writing sky" << endl;
      utils::fits_write_ndarray( fitsfile, sky, fluxMetadata );

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
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
          cout << "stretchAndCrossCorrelateSpec: i_run = " << i_run << ": dispCorControl.lengthPieces = " << dispCorControl.lengthPieces << endl;
          cout << "stretchAndCrossCorrelateSpec: i_run = " << i_run << ": start = " << start << ", end = " << end << endl;
          cout << "stretchAndCrossCorrelateSpec: i_run = " << i_run << ": stretchedSpec.getShape()[0] = " << stretchedSpec.getShape()[0] << endl;
          cout << "stretchAndCrossCorrelateSpec: i_run = " << i_run << ": spec.getShape()[0] = " << spec.getShape()[0] << endl;
        #endif
        if ( end >= stretchedSpec.getShape()[ 0 ] )
          end = stretchedSpec.getShape()[ 0 ] - 1;
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

        specPiece = ndarray::allocate( end - start + 1 );
        specPiece.deep() = stretchedSpec[ ndarray::view( start, end + 1 ) ];
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
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
        cout << "stretchAndCrossCorrelateSpec: stretchAndCrossCorrelateSpecResult.lineList = " << stretchAndCrossCorrelateSpecResult.lineList << endl;
      #endif

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
    
    template< typename T >
    ndarray::Array< T, 2, 1 > createLineList( ndarray::Array< T, 1, 1 > const& wLen,
                                              ndarray::Array< T, 1, 1 > const& lines ){
      ndarray::Array< size_t, 1, 1 > ind = pfsDRPStella::math::getIndicesInValueRange( wLen, T( 1 ), T( 15000 ) );
      ndarray::Array< T, 1, 1 > indT = ndarray::allocate( ind.getShape()[ 0 ] );
      indT.deep() = ind;
      #ifdef __DEBUG_CREATELINELIST__
        cout << "Spectra::createLineList: indT = " << indT.getShape() << ": " << indT << endl;
        cout << "Spectra::createLineList: ind[0] = " << ind[0] << ", ind[ind.getShape()[0](=" << ind.getShape()[0] << ")-1] = " << ind[ind.getShape()[0]-1] << endl;
      #endif
      ndarray::Array< T, 1, 1 > wavelengths = ndarray::copy( wLen[ ndarray::view( ind[ 0 ], ind[ ind.getShape()[ 0 ] - 1 ] + 1 ) ] );
      #ifdef __DEBUG_CREATELINELIST__
        for (int i = 0; i < indT.getShape()[ 0 ]; ++i )
          cout << "Spectra::createLineList: indT[" << i << "] = " << indT[ i ] << ": wavelengths[" << i << "] = " << wavelengths[ i ] << endl;
      #endif
      std::vector< std::string > args( 1 );
//      args[ 0 ] = "SPLINE";
      #ifdef __DEBUG_CREATELINELIST__
        cout << "Spectra::createLineList: lines = " << lines.getShape() << ": " << lines << endl;
      #endif
      ndarray::Array< T, 1, 1 > pix = pfsDRPStella::math::interPol( indT, wavelengths, lines, args );
      #ifdef __DEBUG_CREATELINELIST__
        cout << "Spectra::createLineList: pix = " << pix.getShape() << ": " << pix << endl;
      #endif
      ndarray::Array< T, 2, 1 > out = ndarray::allocate( lines.getShape()[ 0 ], 2 );
      #ifdef __DEBUG_CREATELINELIST__
        cout << "Spectra::createLineList: out = " << out.getShape() << endl;
        cout << "Spectra::createLineList: out[ ndarray::view( )( 0 ) ].getShape() = " << out[ ndarray::view( )( 0 ) ].getShape() << endl;
      #endif
      out[ ndarray::view( )( 0 ) ] = lines;//[ ndarray::view( ) ];
      out[ ndarray::view( )( 1 ) ] = pix;//[ ndarray::view( ) ];
      #ifdef __DEBUG_CREATELINELIST__
        cout << "Spectra::createLineList: out = " << out << endl;
      #endif
      return out;
    }
    
    template ndarray::Array< float, 2, 1 > createLineList( ndarray::Array< float, 1, 1 > const&, ndarray::Array< float, 1, 1 > const& );
    template ndarray::Array< double, 2, 1 > createLineList( ndarray::Array< double, 1, 1 > const&, ndarray::Array< double, 1, 1 > const& );
    
//    template StretchAndCrossCorrelateSpecResult< float, float >;
//    template StretchAndCrossCorrelateSpecResult< double, double >;
//    template StretchAndCrossCorrelateSpecResult< float, double >;
//    template StretchAndCrossCorrelateSpecResult< double, float >;

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
  

//template PTR( pfsDRPStella::Spectrum<float, unsigned int, float, float> ) pfsDRPStella::utils::getPointer( pfsDRPStella::Spectrum<float, unsigned int, float, float> & );

template<> template<> bool pfsDRPStella::Spectrum<float, unsigned int, float, float>::identify( ndarray::Array< float, 2, 1 > const&,
                                                                                                DispCorControl const&,
                                                                                                size_t );
template<> template<> bool pfsDRPStella::Spectrum<double, unsigned int, float, float>::identify(ndarray::Array< float, 2, 1 > const&,
                                                                                   DispCorControl const&,
                                                                                  size_t );
template<> template<> bool pfsDRPStella::Spectrum<float, unsigned short, float, float>::identify(ndarray::Array< float, 2, 1 > const&,
                                                                                    DispCorControl const&,
                                                                                  size_t );
template<> template<> bool pfsDRPStella::Spectrum<double, unsigned short, float, float>::identify(ndarray::Array< float, 2, 1 > const&,
                                                                                     DispCorControl const&,
                                                                                  size_t );
template<> template<> bool pfsDRPStella::Spectrum<float, unsigned int, float, float>::identify(ndarray::Array< double, 2, 1 > const&,
                                                                                  DispCorControl const&,
                                                                                  size_t );
template<> template<> bool pfsDRPStella::Spectrum<double, unsigned int, float, float>::identify(ndarray::Array< double, 2, 1 > const&,
                                                                                   DispCorControl const&,
                                                                                  size_t );
template<> template<> bool pfsDRPStella::Spectrum<float, unsigned short, float, float>::identify(ndarray::Array< double, 2, 1 > const&,
                                                                                    DispCorControl const&,
                                                                                  size_t );
template<> template<> bool pfsDRPStella::Spectrum<double, unsigned short, float, float>::identify(ndarray::Array< double, 2, 1 > const&,
                                                                                     DispCorControl const&,
                                                                                  size_t );
template<> template<> bool pfsDRPStella::Spectrum<float, int, float, float>::identify(ndarray::Array< float, 2, 1 > const&, 
                                                                         DispCorControl const&,
                                                                                  size_t );

template<> template<> bool pfsDRPStella::Spectrum<float, unsigned int, float, float>::identify(ndarray::Array< float, 2, 1 > const&,
                                                                                  ndarray::Array< float, 1, 0 > const&,
                                                                                  ndarray::Array< float, 1, 0 > const&,
                                                                                  DispCorControl const&,
                                                                                  size_t );
template<> template<> bool pfsDRPStella::Spectrum<double, unsigned int, float, float>::identify(ndarray::Array< float, 2, 1 > const&,
                                                                                   ndarray::Array< float, 1, 0 > const&,
                                                                                   ndarray::Array< float, 1, 0 > const&,
                                                                                   DispCorControl const&,
                                                                                  size_t );
template bool pfsDRPStella::Spectrum<float, unsigned short, float, float>::identify(ndarray::Array< float, 2, 1 > const&,
                                                                                   ndarray::Array< float, 1, 0 > const&,
                                                                                   ndarray::Array< float, 1, 0 > const&,
                                                                                   DispCorControl const&,
                                                                                  size_t );
template<> template<> bool pfsDRPStella::Spectrum<double, unsigned short, float, float>::identify(ndarray::Array< float, 2, 1 > const&,
                                                                                     ndarray::Array< float, 1, 0 > const&,
                                                                                     ndarray::Array< float, 1, 0 > const&,
                                                                                     DispCorControl const&,
                                                                                  size_t );
template<> template<> bool pfsDRPStella::Spectrum<float, unsigned int, float, float>::identify(ndarray::Array< double, 2, 1 > const&,
                                                                                  ndarray::Array< double, 1, 0 > const&,
                                                                                  ndarray::Array< double, 1, 0 > const&,
                                                                                  DispCorControl const&,
                                                                                  size_t );
template<> template<> bool pfsDRPStella::Spectrum<double, unsigned int, float, float>::identify(ndarray::Array< double, 2, 1 > const&,
                                                                                   ndarray::Array< double, 1, 0 > const&,
                                                                                   ndarray::Array< double, 1, 0 > const&,
                                                                                   DispCorControl const&,
                                                                                  size_t );
template<> template<> bool pfsDRPStella::Spectrum<float, unsigned short, float, float>::identify(ndarray::Array< double, 2, 1 > const&,
                                                                                    ndarray::Array< double, 1, 0 > const&,
                                                                                    ndarray::Array< double, 1, 0 > const&,
                                                                                    DispCorControl const&,
                                                                                  size_t );
template<> template<> bool pfsDRPStella::Spectrum<double, unsigned short, float, float>::identify(ndarray::Array< double, 2, 1 > const&,
                                                                                     ndarray::Array< double, 1, 0 > const&,
                                                                                     ndarray::Array< double, 1, 0 > const&,
                                                                                     DispCorControl const&,
                                                                                  size_t );
template<> template<> bool pfsDRPStella::Spectrum<float, int, float, float>::identify(ndarray::Array< float, 2, 1 > const&, 
                                                                         ndarray::Array< float, 1, 0 > const&,
                                                                         ndarray::Array< float, 1, 0 > const&,
                                                                         DispCorControl const&,
                                                                                  size_t );

//template PTR(afwImage::MaskedImage<float, unsigned short, float>) pfsDRPStella::utils::getPointer(afwImage::MaskedImage<float, unsigned short, float> &);
//template PTR(afwImage::MaskedImage<double, unsigned short, float>) pfsDRPStella::utils::getPointer(afwImage::MaskedImage<double, unsigned short, float> &);
//template PTR(std::vector<unsigned short>) pfsDRPStella::utils::getPointer(std::vector<unsigned short> &);
//template PTR(std::vector<unsigned int>) pfsDRPStella::utils::getPointer(std::vector<unsigned int> &);
//template PTR(std::vector<int>) pfsDRPStella::utils::getPointer(std::vector<int> &);
//template PTR(std::vector<float>) pfsDRPStella::utils::getPointer(std::vector<float> &);
//template PTR(std::vector<double>) pfsDRPStella::utils::getPointer(std::vector<double> &);
//template PTR(pfsDRPStella::Spectrum<float, int, float, float>) pfsDRPStella::utils::getPointer(pfsDRPStella::Spectrum<float, int, float, float> &);
//template PTR(pfsDRPStella::Spectrum<float, unsigned short, float, float>) pfsDRPStella::utils::getPointer(pfsDRPStella::Spectrum<float, unsigned short, float, float> &);
//template PTR(pfsDRPStella::Spectrum<double, unsigned short, float, float>) pfsDRPStella::utils::getPointer(pfsDRPStella::Spectrum<double, unsigned short, float, float> &);

