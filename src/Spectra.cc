#include "lsst/log/Log.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/MaskedImage.h"

#include "pfs/drp/stella/Spectra.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/math/CurveFitting.h"
#include "pfs/drp/stella/cmpfit-1.2/MPFitting_ndarray.h"

namespace pfs { namespace drp { namespace stella {

/** @brief Construct a Spectrum with empty vectors of specified size (default 0)
 */
template<typename ImageT, typename MaskT, typename VarianceT>
Spectrum<ImageT, MaskT, VarianceT>::Spectrum(size_t length, size_t iTrace )
  : _length(length),
    _mask(length, 1),
    _iTrace(iTrace),
    _isWavelengthSet(false),
    _dispCorControl( new DispCorControl )
{
  _spectrum = ndarray::allocate( length );
  _spectrum.deep() = 0.;
  _covar = ndarray::allocate( length, 3 );
  _covar.deep() = 0.;
  _wavelength = ndarray::allocate( length );
  _wavelength.deep() = 0.;
  _dispCoeffs = ndarray::allocate( _dispCorControl->order + 1 );
  _dispCoeffs.deep() = 0.;
  _dispRms = 0.;
  _dispRmsCheck = 0.;
  _nGoodLines = 0;
  _minY = 0;
  _maxY = length - 1;
  _nCCDRows = length;
}

template<typename ImageT, typename MaskT, typename VarianceT>
void Spectrum<ImageT, MaskT, VarianceT>::setWavelength( ndarray::Array<float, 1, 1> const& wavelength )
{
  /// Check length of input wavelength
  if (static_cast<size_t>(wavelength.getShape()[0]) != _length){
    string message("pfsDRPStella::Spectrum::setWavelength: ERROR: wavelength->size()=");
    message += to_string(wavelength.getShape()[0]) + string(" != _length=") + to_string(_length);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  _wavelength.deep() = wavelength;
}

///SpectrumSet
template<typename ImageT, typename MaskT, typename VarianceT>
SpectrumSet<ImageT, MaskT, VarianceT>::SpectrumSet( size_t nSpectra, size_t length )
        : _spectra( new std::vector< PTR( Spectrum< ImageT, MaskT, VarianceT> ) >() )
{
  for (size_t i = 0; i < nSpectra; ++i){
    PTR(Spectrum<ImageT, MaskT, VarianceT>) spec( new Spectrum<ImageT, MaskT, VarianceT>( length, i ) );
    _spectra->push_back(spec);
  }
}

template<typename ImageT, typename MaskT, typename VarianceT>
SpectrumSet<ImageT, MaskT, VarianceT>::SpectrumSet( std::vector< PTR( Spectrum< ImageT, MaskT, VarianceT> > ) const& spectrumVector )
        : _spectra( new std::vector< PTR( Spectrum< ImageT, MaskT, VarianceT> ) >() )
{
  for (size_t i = 0; i < spectrumVector.size(); ++i){
    _spectra->push_back( spectrumVector.at(i) );
  }
}

template<typename ImageT, typename MaskT, typename VarianceT>
ndarray::Array< float, 2, 1 > SpectrumSet<ImageT, MaskT, VarianceT>::getAllFluxes() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 2, 1 > flux = ndarray::allocate( nCCDRows, nFibers );

  flux.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( Spectrum< ImageT, MaskT, VarianceT> ) spectrum = _spectra->at( iFiber );

    int minY = spectrum->getMinY();
    int maxY = spectrum->getMaxY();
    if ( maxY - minY + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      string message("SpectrumSet::writeFits: spectrum does not have expected shape: maxY=");
      message += to_string(maxY) + " - minY=" + to_string(minY) + " + 1 (=" + to_string(maxY - minY + 1);
      message += + ") = " +to_string(maxY-minY + 1) + " != spectrum->getSpectrum().getShape()[ 0 ] = ";
      message += to_string(spectrum->getSpectrum().getShape()[ 0 ]);
      throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,message.c_str());
    }

    flux[ ndarray::view( minY, maxY + 1 )( iFiber ) ] = spectrum->getSpectrum()[ ndarray::view() ];
  }
  return flux;
}

template<typename ImageT, typename MaskT, typename VarianceT>
ndarray::Array< float, 2, 1 > SpectrumSet<ImageT, MaskT, VarianceT>::getAllWavelengths() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 2, 1 > lambda = ndarray::allocate( nCCDRows, nFibers );

  lambda.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( Spectrum< ImageT, MaskT, VarianceT> ) spectrum = _spectra->at( iFiber );

    int minY = spectrum->getMinY();
    int maxY = spectrum->getMaxY();
    if ( maxY - minY + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      string message("SpectrumSet::writeFits: spectrum does not have expected shape: maxY=");
      message += to_string(maxY) + " - minY=" +to_string(minY) + " + 1 (=" + to_string(maxY - minY + 1);
      message += ") = " + to_string(maxY-minY + 1) + " != spectrum->getSpectrum().getShape()[ 0 ] = ";
      message += to_string(spectrum->getSpectrum().getShape()[ 0 ]);
      throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,message.c_str());
    }
    lambda[ ndarray::view( minY, maxY + 1 )( iFiber ) ] = spectrum->getWavelength();
  }
  return lambda;
}

template<typename ImageT, typename MaskT, typename VarianceT>
ndarray::Array< int, 2, 1 > SpectrumSet<ImageT, MaskT, VarianceT>::getAllMasks() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< int, 2, 1 > mask = ndarray::allocate( nCCDRows, nFibers );

  mask.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( Spectrum< ImageT, MaskT, VarianceT > ) spectrum = _spectra->at( iFiber );

    int minY = spectrum->getMinY();
    int maxY = spectrum->getMaxY();
    if ( maxY - minY + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      string message("SpectrumSet::writeFits: spectrum does not have expected shape: maxY=");
      message += to_string(maxY) + " - minY=" +to_string(minY) + " + 1 (=" + to_string(maxY - minY + 1);
      message += ") = " + to_string(maxY-minY + 1) + " != spectrum->getSpectrum().getShape()[ 0 ] = ";
      message += to_string(spectrum->getSpectrum().getShape()[ 0 ]);
      throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,message.c_str());
    }
  }
  return mask;
}

template<typename ImageT, typename MaskT, typename VarianceT>
ndarray::Array< float, 3, 1 > SpectrumSet<ImageT, MaskT, VarianceT>::getAllCovars() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 3, 1 > covar = ndarray::allocate( nCCDRows, 3, nFibers );

  covar.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( Spectrum< ImageT, MaskT, VarianceT > ) spectrum = _spectra->at( iFiber );

    int minY = spectrum->getMinY();
    int maxY = spectrum->getMaxY();
    if ( maxY - minY + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      string message("SpectrumSet::writeFits: spectrum does not have expected shape: maxY=");
      message += to_string(maxY) + " - minY=" + to_string(minY) + " + 1 (=" + to_string(maxY - minY + 1);
      message += ") = " + to_string(maxY-minY + 1) + " != spectrum->getSpectrum().getShape()[ 0 ] = ";
      message += to_string(spectrum->getSpectrum().getShape()[ 0 ]);
      throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,message.c_str());
    }
    #ifdef __DEBUG_GETALLCOVARS__
      cout << "covar[ ndarray::view( " << minY << ", " << maxY + 1 << " )( )( " << iFiber << " ) ].getShape() = " << covar[ ndarray::view( minY, maxY + 1 )( )( iFiber ) ].getShape() << ", spectrum->getCovar().getShape() = " << spectrum->getCovar().getShape() << endl;
    #endif
    covar[ ndarray::view( minY, maxY + 1 )( )( iFiber ) ] = spectrum->getCovar()[ ndarray::view() ];
  }
  return covar;
}

template< typename ImageT, typename MaskT, typename VarianceT >
Spectrum< ImageT, MaskT, VarianceT >::Spectrum( Spectrum< ImageT, MaskT, VarianceT > & spectrum,
                                                                                  std::size_t iTrace,
                                                bool deep ) :
        _minY( spectrum.getMinY() ),
        _maxY( spectrum.getMaxY() ),
        _length( spectrum._length ),
        _nCCDRows( spectrum.getNCCDRows() ),
        _spectrum( spectrum.getSpectrum() ),
        _mask( spectrum.getMask(), deep ),
        _covar( spectrum.getCovar() ),
        _wavelength( spectrum.getWavelength() ),
        _iTrace( spectrum.getITrace() ),
        _dispCoeffs( spectrum._dispCoeffs ),
        _dispRms( spectrum.getDispRms() ),
        _dispRmsCheck( spectrum.getDispRmsCheck() ),
        _nGoodLines( spectrum.getNGoodLines() ),
        _isWavelengthSet( spectrum.isWavelengthSet() ),
        _dispCorControl( spectrum._dispCorControl )
{
    if ( deep ){
        /// allocate memory
        _spectrum = ndarray::allocate(spectrum.getSpectrum().getShape()[0]);
        _covar = ndarray::allocate(spectrum.getCovar().getShape());
        _wavelength = ndarray::allocate(spectrum.getWavelength().getShape()[0]);
        _dispCoeffs = ndarray::allocate(spectrum._dispCoeffs.getShape()[0]);

        /// copy variables
        _spectrum.deep() = spectrum.getSpectrum();
        _covar.deep() = spectrum.getCovar();
        _wavelength.deep() = spectrum.getWavelength();
        _dispCoeffs.deep() = spectrum._dispCoeffs;
    }
    if (iTrace != 0)
        _iTrace = iTrace;
    _mask.addMaskPlane("REJECTED_LINES");
}

template< typename ImageT, typename MaskT, typename VarianceT >
Spectrum< ImageT, MaskT, VarianceT >::Spectrum( Spectrum< ImageT, MaskT, VarianceT > const& spectrum) :
        _minY( spectrum.getMinY() ),
        _maxY( spectrum.getMaxY() ),
        _length( spectrum._length ),
        _nCCDRows( spectrum.getNCCDRows() ),
        _mask(spectrum.getMask()),
        _iTrace( spectrum.getITrace() ),
        _dispRms( spectrum.getDispRms() ),
        _dispRmsCheck( spectrum.getDispRmsCheck() ),
        _nGoodLines( spectrum.getNGoodLines() ),
        _isWavelengthSet( spectrum.isWavelengthSet() ),
        _dispCorControl( spectrum._dispCorControl )
{
    /// allocate memory
    _spectrum = ndarray::allocate(spectrum.getSpectrum().getShape()[0]);
    _covar = ndarray::allocate(spectrum.getCovar().getShape());
    _wavelength = ndarray::allocate(spectrum.getWavelength().getShape()[0]);
    _dispCoeffs = ndarray::allocate(spectrum._dispCoeffs.getShape()[0]);

    /// copy variables
    _spectrum.deep() = spectrum.getSpectrum();
    _covar.deep() = spectrum.getCovar();
    _wavelength.deep() = spectrum.getWavelength();
    _dispCoeffs.deep() = spectrum._dispCoeffs;
}

template<typename ImageT, typename MaskT, typename VarianceT>
void
Spectrum<ImageT, MaskT, VarianceT>::setSpectrum( ndarray::Array<ImageT, 1, 1> const& spectrum )
{
  /// Check length of input spectrum
  if (static_cast<std::size_t>(spectrum.getShape()[0]) != _length) {
    string message("pfs::drp::stella::Spectrum::setSpectrum: ERROR: spectrum->size()=");
    message += to_string(spectrum.getShape()[0]) + string(" != _length=") + to_string(_length);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _spectrum.deep() = spectrum;
}

template<typename ImageT, typename MaskT, typename VarianceT>
ndarray::Array<VarianceT, 1, 1>
Spectrum<ImageT, MaskT, VarianceT>::getVariance() const
{
    ndarray::Array< VarianceT, 1, 1 > variance = ndarray::allocate( _covar.getShape()[ 0 ] );
    variance.deep() = _covar[ ndarray::view( )( 1 ) ];
    return variance;
}

template<typename ImageT, typename MaskT, typename VarianceT>
ndarray::Array<VarianceT, 1, 1>
Spectrum<ImageT, MaskT, VarianceT>::getVariance()
{
    ndarray::Array< VarianceT, 1, 1 > variance = ndarray::allocate(_covar.getShape()[0]);
    variance[ndarray::view()] = _covar[ ndarray::view( )( 1 ) ];
    return variance;
}

template<typename ImageT, typename MaskT, typename VarianceT>
void
Spectrum<ImageT, MaskT, VarianceT>::setVariance( ndarray::Array<VarianceT, 1, 1> const& variance )
{
  /// Check length of input variance
  if (static_cast<std::size_t>(variance.getShape()[0]) != _length) {
    string message("pfs::drp::stella::Spectrum::setVariance: ERROR: variance->size()=");
    message += to_string( variance.getShape()[ 0 ] ) + string( " != _length=" ) + to_string( _length );
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _covar[ ndarray::view()(1) ] = variance;
}

template<typename ImageT, typename MaskT, typename VarianceT>
void
Spectrum<ImageT, MaskT, VarianceT>::setCovar(const ndarray::Array<VarianceT, 2, 1> & covar )
{
    /// Check length of input covar
    if (static_cast<std::size_t>(covar.getShape()[0]) != _length) {
      string message("pfs::drp::stella::Spectrum::setCovar: ERROR: covar->size()=");
      message += to_string( covar.getShape()[0] ) + string( " != _length=" ) + to_string( _length );
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    if (covar.getShape()[1] != 3) {
      string message("pfs::drp::stella::Spectrum::setCovar: ERROR: covar->size()=");
      message += to_string( covar.getShape()[1] ) + string( " != 3" );
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    _covar.deep() = covar;
}

template<typename ImageT, typename MaskT, typename VarianceT>
void
Spectrum<ImageT, MaskT, VarianceT>::setMask(const lsst::afw::image::Mask<MaskT> & mask)
{
  /// Check length of input mask
  if (static_cast<std::size_t>(mask.getWidth()) != _length){
    string message("pfs::drp::stella::Spectrum::setMask: ERROR: mask.getWidth()=");
    message += to_string(mask.getWidth()) + string(" != _length=") + to_string(_length);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _mask = mask;
}

template<typename ImageT, typename MaskT, typename VarianceT>
void
Spectrum< ImageT, MaskT, VarianceT >::setMinY( const std::size_t minY )
{
  if ( minY > _nCCDRows ){
    string message("pfs::drp::stella::Spectrum::setMinY: ERROR: minY=");
    message += to_string( minY ) + string(" > _nCCDRows=") + to_string(_nCCDRows);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _minY = minY;
}

template<typename ImageT, typename MaskT, typename VarianceT>
void
Spectrum<ImageT, MaskT, VarianceT>::setMaxY(const std::size_t maxY)
{
  if ( maxY > _nCCDRows ){
    _nCCDRows = _minY + maxY;            // this is strange, but so is the definition of yLow/High (now min/maxY)
  }
  _maxY = maxY;
}

template<typename ImageT, typename MaskT, typename VarianceT>
void
Spectrum<ImageT, MaskT, VarianceT>::setNCCDRows(const std::size_t nCCDRows)
{
  if ( _minY > nCCDRows ){
    string message("pfs::drp::stella::Spectrum::setMinY: ERROR: _minY=");
    message += to_string( _minY ) + string(" > nCCDRows=") + to_string(nCCDRows);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _nCCDRows = nCCDRows;
}

template< typename ImageT, typename MaskT, typename VarianceT >
ndarray::Array<float, 1, 1 >
Spectrum<ImageT, MaskT, VarianceT>::hIdentify( ndarray::Array< float, 2, 1 > const& lineList )
{
  LOG_LOGGER _log = LOG_GET("pfs.drp.stella.Spectra.identify");
  ///for each line in line list, find maximum in spectrum and fit Gaussian
  int I_MaxPos = 0;
  int I_Start = 0;
  int I_End = 0;
  int I_NTerms = 4;
  std::vector<float> V_GaussSpec( 1, 0. );
  ndarray::Array<float, 1, 1 > D_A1_GaussCoeffs = ndarray::allocate( I_NTerms );
  D_A1_GaussCoeffs.deep() = 0.;
  ndarray::Array< float, 1, 1 > D_A1_EGaussCoeffs = ndarray::allocate( I_NTerms );
  D_A1_EGaussCoeffs.deep() = 0.;
  ndarray::Array< int, 2, 1 > I_A2_Limited = ndarray::allocate( I_NTerms, 2 );
  I_A2_Limited.deep() = 1;
  ndarray::Array< float, 2, 1 > D_A2_Limits = ndarray::allocate( I_NTerms, 2 );
  D_A2_Limits.deep() = 0.;
  ndarray::Array< float, 1, 1 > D_A1_Guess = ndarray::allocate( I_NTerms );
  std::vector< float > V_MeasureErrors( 2, 0.);
  ndarray::Array< float, 1, 1 > D_A1_Ind = math::indGenNdArr( float( _spectrum.getShape()[ 0 ] ) );
  std::vector< float > V_X( 1, 0. );
  ndarray::Array< float, 1, 1 > D_A1_GaussPos = ndarray::allocate( lineList.getShape()[0] );
  D_A1_GaussPos.deep() = 0.;
  #ifdef __WITH_PLOTS__
    CString CS_PlotName("");
    CString *P_CS_Num;
  #endif
  for ( int i_line = 0; i_line < lineList.getShape()[ 0 ]; ++i_line ){
    I_Start = int( lineList[ ndarray::makeVector( i_line, 1 ) ] ) - _dispCorControl->searchRadius;
    if ( I_Start < 0 )
      I_Start = 0;
    LOGLS_DEBUG(_log, "i_line = " << i_line << ": I_Start = " << I_Start);
    I_End = int( lineList[ ndarray::makeVector( i_line, 1 ) ] ) + _dispCorControl->searchRadius;
    if ( I_End >= _spectrum.getShape()[ 0 ] )
      I_End = _spectrum.getShape()[ 0 ] - 1;
    if ( ( I_End - I_Start ) > ( 1.5 * _dispCorControl->searchRadius ) ){
      LOGLS_DEBUG(_log, "i_line = " << i_line << ": I_End = " << I_End);
      if ( I_Start >= I_End ){
        LOGLS_WARN(_log, "I_Start(=" << I_Start << ") >= I_End(=" << I_End << ")");
        LOGLS_DEBUG(_log, "_spectrum = " << _spectrum);
        LOGLS_DEBUG(_log, "lineList = " << lineList);
      }
      else{
        auto itMaxElement = std::max_element( _spectrum.begin() + I_Start, _spectrum.begin() + I_End + 1 );
        I_MaxPos = std::distance(_spectrum.begin(), itMaxElement);
        LOGLS_DEBUG(_log, "I_MaxPos = " << I_MaxPos);
        I_Start = std::round( float( I_MaxPos ) - ( 1.5 * _dispCorControl->fwhm ) );
        if (I_Start < 0)
          I_Start = 0;
        LOGLS_DEBUG(_log, "I_Start = " << I_Start);
        I_End = std::round( float( I_MaxPos ) + ( 1.5 * _dispCorControl->fwhm ) );
        if ( I_End >= _spectrum.getShape()[ 0 ] )
          I_End = _spectrum.getShape()[ 0 ] - 1;
        LOGLS_DEBUG(_log, "I_End = " << I_End);
        if ( I_End < I_Start + 4 ){
          LOGLS_WARN(_log, "WARNING: Line position outside spectrum");
        }
        else{
          V_GaussSpec.resize( I_End - I_Start + 1 );
          V_MeasureErrors.resize( I_End - I_Start + 1 );
          V_X.resize( I_End - I_Start + 1 );
          auto itSpec = _spectrum.begin() + I_Start;
          for ( auto itGaussSpec = V_GaussSpec.begin(); itGaussSpec != V_GaussSpec.end(); ++itGaussSpec, ++itSpec )
            *itGaussSpec = *itSpec;
          LOGLS_DEBUG(_log, "V_GaussSpec = ");
          for ( int iPos = 0; iPos < V_GaussSpec.size(); ++iPos )
              LOGLS_DEBUG(_log, V_GaussSpec[iPos] << " ");
          for( auto itMeasErr = V_MeasureErrors.begin(), itGaussSpec = V_GaussSpec.begin(); itMeasErr != V_MeasureErrors.end(); ++itMeasErr, ++itGaussSpec ){
            *itMeasErr = sqrt( std::fabs( *itGaussSpec ) );
            if (*itMeasErr < 0.00001)
              *itMeasErr = 1.;
          }
          LOGLS_DEBUG(_log, "V_MeasureErrors = ");
          for (int iPos = 0; iPos < V_MeasureErrors.size(); ++iPos )
              LOGLS_DEBUG(_log, V_MeasureErrors[iPos] << " ");
          auto itInd = D_A1_Ind.begin() + I_Start;
          for ( auto itX = V_X.begin(); itX != V_X.end(); ++itX, ++itInd )
            *itX = *itInd;
          LOGLS_DEBUG(_log, "V_X = ");
          for (int iPos = 0; iPos < V_X.size(); ++iPos )
              LOGLS_DEBUG(_log, V_X[iPos] << " ");

        /*     D_A1_Guess[3] = constant offset
         *     D_A1_Guess[0] = peak y value
         *     D_A1_Guess[1] = x centroid position
         *     D_A1_Guess[2] = gaussian sigma width
         */
          D_A1_Guess[ 3 ] = *min_element( V_GaussSpec.begin(), V_GaussSpec.end() );
          D_A1_Guess[ 0 ] = *max_element( V_GaussSpec.begin(), V_GaussSpec.end() ) - D_A1_Guess(3);
          D_A1_Guess[ 1 ] = V_X[ 0 ] + ( V_X[ V_X.size() - 1 ] - V_X[ 0 ] ) / 2.;
          D_A1_Guess[ 2 ] = _dispCorControl->fwhm;
          LOGLS_DEBUG(_log, "D_A1_Guess = " << D_A1_Guess);
          D_A2_Limits[ ndarray::makeVector( 0, 0 ) ] = 0.;
          D_A2_Limits[ ndarray::makeVector( 0, 1 ) ] = std::fabs( 1.5 * D_A1_Guess[ 0 ] );
          D_A2_Limits[ ndarray::makeVector( 1, 0 ) ] = V_X[ 1 ];
          D_A2_Limits[ ndarray::makeVector( 1, 1 ) ] = V_X[ V_X.size() - 2 ];
          D_A2_Limits[ ndarray::makeVector( 2, 0 ) ] = D_A1_Guess[ 2 ] / 3.;
          D_A2_Limits[ ndarray::makeVector( 2, 1 ) ] = 2. * D_A1_Guess[ 2 ];
          D_A2_Limits[ ndarray::makeVector( 3, 0 ) ] = (D_A1_Guess[ 3 ] > 0) ? 0 : D_A1_Guess[ 3 ];
          D_A2_Limits[ ndarray::makeVector( 3, 1 ) ] = std::fabs( 1.5 * D_A1_Guess[ 3 ] ) + 1;
          LOGLS_DEBUG(_log, "D_A2_Limits = " << D_A2_Limits);
          ndarray::Array< float, 1, 1 > D_A1_X = ndarray::external( V_X.data(), ndarray::makeVector( int( V_X.size() ) ), ndarray::makeVector( 1 ) );
          ndarray::Array< float, 1, 1 > D_A1_GaussSpec = ndarray::external( V_GaussSpec.data(), ndarray::makeVector( int( V_GaussSpec.size() ) ), ndarray::makeVector( 1 ) );
          ndarray::Array< float, 1, 1 > D_A1_MeasureErrors = ndarray::external( V_MeasureErrors.data(), ndarray::makeVector( int( V_MeasureErrors.size() ) ), ndarray::makeVector( 1 ) );
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
            LOGLS_WARN(_log, "WARNING: GaussFit returned FALSE");
          }
          else{
            LOGLS_DEBUG(_log, "i_line = " << i_line << ": D_A1_GaussCoeffs = " << D_A1_GaussCoeffs);
            if ( std::fabs( float( I_MaxPos ) - D_A1_GaussCoeffs[ 1 ] ) < _dispCorControl->maxDistance ){
              D_A1_GaussPos[ i_line ] = D_A1_GaussCoeffs[ 1 ];
              LOGLS_DEBUG(_log, "D_A1_GaussPos[" << i_line << "] = " << D_A1_GaussPos[ i_line ]);
              if ( i_line > 0 ){
                if ( std::fabs( D_A1_GaussPos[ i_line ] - D_A1_GaussPos[ i_line - 1 ] ) < 1.5 ){/// wrong line identified!
                  if ( lineList.getShape()[ 0 ] > 2 ){
                    if ( lineList[ ndarray::makeVector( i_line, 2 ) ] < lineList[ ndarray::makeVector( i_line - 1, 2 ) ] ){
                      LOGLS_WARN(_log, "WARNING: i_line=" << i_line << ": line " << i_line << " at " << D_A1_GaussPos[ i_line ] << " has probably been misidentified (D_A1_GaussPos(" << i_line-1 << ")=" << D_A1_GaussPos[ i_line - 1 ] << ") => removing line from line list");
                      D_A1_GaussPos[ i_line ] = 0.;
                    }
                    else{
                      LOGLS_WARN(_log, "WARNING: i_line=" << i_line << ": line at D_A1_GaussPos[" << i_line-1 << "] = " << D_A1_GaussPos[ i_line - 1 ] << " has probably been misidentified (D_A1_GaussPos(" << i_line << ")=" << D_A1_GaussPos[ i_line ] << ") => removing line from line list");
                      D_A1_GaussPos[ i_line - 1 ] = 0.;
                    }
                  }
                }
              }
            }
            else{
                string message("WARNING: I_MaxPos=");
                message += to_string(I_MaxPos) + " - D_A1_GaussCoeffs[ 1 ]=" + to_string(D_A1_GaussCoeffs[ 1 ]);
                message += "(=" + to_string(std::fabs( float( I_MaxPos ) - D_A1_GaussCoeffs[ 1 ] ));
                message += ") >= " + to_string(_dispCorControl->maxDistance) + " => Skipping line";
              LOGLS_WARN(_log, message);
            }
          }
        }
      }
    }
  }/// end for (int i_line=0; i_line < D_A2_LineList_In.rows(); i_line++){
  return D_A1_GaussPos;
}

template< typename ImageT, typename MaskT, typename VarianceT >
void
Spectrum<ImageT, MaskT, VarianceT>::identify( ndarray::Array< float, 2, 1 > const& lineList,
                                                                                     DispCorControl const& dispCorControl,
                                                                                     std::size_t nLinesCheck ){
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.Spectra.identify");

    DispCorControl tempDispCorControl( dispCorControl );
    _dispCorControl.reset();
    _dispCorControl = tempDispCorControl.getPointer();

    ///for each line in line list, find maximum in spectrum and fit Gaussian
    ndarray::Array< float, 1, 1 > D_A1_GaussPos = hIdentify( lineList );

    ///remove lines which could not be found from line list
    std::vector< int > V_Index( D_A1_GaussPos.getShape()[ 0 ], 0 );
    std::size_t pos = 0;
    for (auto it = D_A1_GaussPos.begin(); it != D_A1_GaussPos.end(); ++it, ++pos ){
      if ( *it > 0. )
        V_Index[ pos ] = 1;
    }
    LOGLS_DEBUG(_log, "D_A1_GaussPos = " << D_A1_GaussPos);
    LOGLS_DEBUG(_log, "V_Index = ");
    for (int iPos = 0; iPos < V_Index.size(); ++iPos)
        LOGLS_DEBUG(_log, V_Index[iPos] << " ");
    std::vector< std::size_t > indices = math::getIndices( V_Index );
    std::size_t nInd = std::accumulate( V_Index.begin(), V_Index.end(), 0 );
    if (nInd == 0){
        std::string message("identify: No lines identified");
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    LOGLS_DEBUG(_log, nInd << " lines identified");
    LOGLS_DEBUG(_log, "indices = ");
    for (int iPos = 0; iPos < indices.size(); ++iPos )
        LOGLS_DEBUG(_log, indices[iPos] << " ");

    /// separate lines to fit and lines for RMS test
    std::vector< std::size_t > indCheck;
    for ( std::size_t i = 0; i < nLinesCheck; ++i ){
      srand( 0 ); //seed initialization
      int randNum = rand() % ( indices.size() - 2 ) + 1; // Generate a random number between 0 and 1
      indCheck.push_back( std::size_t( randNum ) );
      indices.erase( indices.begin() + randNum );
    }

    _nGoodLines = nInd;
    const long nLinesIdentifiedMin(std::lround(float(lineList.getShape()[0])
                                               * dispCorControl.minPercentageOfLines / 100.));
    if ( _nGoodLines < nLinesIdentifiedMin ){
      std::string message("identify: ERROR: less than ");
      message += std::to_string(nLinesIdentifiedMin) + " lines identified";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    ndarray::Array< std::size_t, 1, 1 > I_A1_IndexPos = ndarray::external( indices.data(), ndarray::makeVector( int( indices.size() ) ), ndarray::makeVector( 1 ) );
    ndarray::Array< float, 1, 1 > D_A1_WLen = ndarray::allocate( lineList.getShape()[ 0 ] );
    ndarray::Array< float, 1, 1 > D_A1_FittedPos = math::getSubArray( D_A1_GaussPos, 
                                                                       I_A1_IndexPos );
    ndarray::Array< std::size_t, 1, 1 > I_A1_IndexCheckPos = ndarray::external( indCheck.data(), ndarray::makeVector( int( indCheck.size() ) ), ndarray::makeVector( 1 ) );
    ndarray::Array< float, 1, 1 > D_A1_FittedCheckPos = math::getSubArray( D_A1_GaussPos, 
                                                                            I_A1_IndexCheckPos );
    LOGLS_DEBUG(_log, "D_A1_FittedPos = " << D_A1_FittedPos << endl);

    D_A1_WLen[ ndarray::view() ] = lineList[ ndarray::view()( 0 ) ];
    ndarray::Array< float, 1, 1 > D_A1_FittedWLen = math::getSubArray( D_A1_WLen, I_A1_IndexPos );
    LOGLS_DEBUG(_log, "found D_A1_FittedWLen = " << D_A1_FittedWLen);

    ndarray::Array< float, 1, 1 > D_A1_FittedWLenCheck = math::getSubArray( D_A1_WLen, I_A1_IndexCheckPos );

    std::vector<string> S_A1_Args(3);
    std::vector<void *> PP_Args(3);
    S_A1_Args[0] = "XRANGE";
    ndarray::Array<float, 1, 1> xRange = ndarray::allocate(2);
    xRange[0] = 0.;
    xRange[1] = _length-1;
    PTR(ndarray::Array<float, 1, 1>) pXRange(new ndarray::Array<float, 1, 1>(xRange));
    PP_Args[0] = &pXRange;
    S_A1_Args[1] = "REJECTED";
    PTR(std::vector<std::size_t>) rejected(new std::vector<std::size_t>());
    PP_Args[1] = &rejected;
    S_A1_Args[2] = "NOT_REJECTED";
    PTR(std::vector<std::size_t>) notRejected(new std::vector<std::size_t>());
    PP_Args[2] = &notRejected;

    _dispCoeffs = ndarray::allocate( dispCorControl.order + 1 );
    _dispCoeffs.deep() = math::PolyFit( D_A1_FittedPos,
                                        D_A1_FittedWLen,
                                        dispCorControl.order,
                                        float(0. - dispCorControl.sigmaReject),
                                        float(dispCorControl.sigmaReject),
                                        dispCorControl.nIterReject,
                                        S_A1_Args,
                                        PP_Args);
    LOGLS_DEBUG(_log, "_dispCoeffs = " << _dispCoeffs);
    
    /// Remove lines rejected by PolyFit from D_A1_FittedPos and D_A1_FittedWLen
    lsst::afw::image::MaskPixel maskVal = 1 << _mask.getMaskPlane("REJECTED_LINES");
    for (int i = 0; i < rejected->size(); ++i){
        LOGLS_DEBUG(_log, "rejected D_A1_FittedPos[" << (*rejected)[i] << "] = " << D_A1_FittedPos[(*rejected)[i]]);
        for (int p = (D_A1_FittedPos[(*rejected)[i]]-2 < 0 ? 0 : D_A1_FittedPos[(*rejected)[i]]-2);
                 p < (D_A1_FittedPos[(*rejected)[i]]+2 >= _length ? _length-1 : D_A1_FittedPos[(*rejected)[i]]+2); ++p){
            _mask(p, 0) |= maskVal;
            LOGLS_DEBUG(_log, "i=" << i << ": (*rejected)[i] _mask(" << p << ", 0) set to " << _mask(p,0));
        }
    }
    _nGoodLines = notRejected->size();
    if ( _nGoodLines < nLinesIdentifiedMin ){
      string message ("identify: ERROR: less than ");
      message += to_string(nLinesIdentifiedMin) + " lines identified";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    ndarray::Array< std::size_t, 1, 1 > notRejectedArr = ndarray::external(notRejected->data(),
                                                                      ndarray::makeVector(_nGoodLines),
                                                                      ndarray::makeVector( 1 ) );

    ndarray::Array<float, 1, 1> fittedPosNotRejected = math::getSubArray(D_A1_FittedPos, notRejectedArr);
    LOGLS_DEBUG(_log, "fittedPosNotRejected = " << _nGoodLines << ": " << fittedPosNotRejected);
    
    ndarray::Array<float, 1, 1> fittedWLenNotRejected = math::getSubArray(D_A1_FittedWLen, notRejectedArr);
    LOGLS_DEBUG(_log, "fittedWLenNotRejected = " << _nGoodLines << ": " << fittedWLenNotRejected);
    ndarray::Array< float, 1, 1 > D_A1_WLen_Gauss = math::Poly( fittedPosNotRejected,
                                                                 _dispCoeffs,
                                                                 xRange[0],
                                                                 xRange[1]);
    ndarray::Array< float, 1, 1 > D_A1_WLen_GaussCheck = math::Poly( D_A1_FittedCheckPos,
                                                                      _dispCoeffs,
                                                                      xRange[0],
                                                                      xRange[1]);
    LOGLS_DEBUG(_log, "D_A1_WLen_PolyFit = " << D_A1_WLen_Gauss);
    LOGLS_DEBUG(_log, "_dispCoeffs = " << _dispCoeffs);

    ///Calculate RMS
    ndarray::Array< float, 1, 1 > D_A1_WLenMinusFit = ndarray::allocate( D_A1_WLen_Gauss.getShape()[ 0 ] );
    D_A1_WLenMinusFit.deep() = fittedWLenNotRejected - D_A1_WLen_Gauss;
    LOGLS_DEBUG(_log, "D_A1_WLenMinusFit = " << D_A1_WLenMinusFit);
    _dispRms = math::calcRMS( D_A1_WLenMinusFit );
    LOGLS_DEBUG(_log, "_nGoodLines = " << _nGoodLines);
    LOGLS_DEBUG(_log, "_dispRms = " << _dispRms);

    ///Calculate RMS for test lines
    ndarray::Array< float, 1, 1 > D_A1_WLenMinusFitCheck = ndarray::allocate( D_A1_WLen_GaussCheck.getShape()[ 0 ] );
    D_A1_WLenMinusFitCheck.deep() = D_A1_FittedWLenCheck - D_A1_WLen_GaussCheck;
    LOGLS_DEBUG(_log, "D_A1_WLenMinusFitCheck = " << D_A1_WLenMinusFitCheck);
    _dispRmsCheck = math::calcRMS( D_A1_WLenMinusFitCheck );
    LOGLS_DEBUG(_log, "dispRmsCheck = " << _dispRmsCheck);

    ///calibrate spectrum
    ndarray::Array< float, 1, 1 > D_A1_Indices = math::indGenNdArr( float( _spectrum.getShape()[ 0 ] ) );
    _wavelength = ndarray::allocate( _spectrum.getShape()[ 0 ] );
    _wavelength.deep() = math::Poly( D_A1_Indices, _dispCoeffs, xRange[0], xRange[1] );
    LOGLS_DEBUG(_log, "_wavelength = " << _wavelength);

    /// Check for monotonic
    if ( math::isMonotonic( _wavelength ) == 0 ){
      LOGLS_DEBUG(_log, "WARNING: Wavelength solution is not monotonic => Setting identifyResult.rms to 1000");
      LOGLS_DEBUG(_log, "_wavelength = " << _wavelength);
      _dispRms = 1000.;
      LOGLS_WARN(_log, "Identify: RMS = " << _dispRms);
    } else {
      LOGLS_DEBUG(_log, "_wavelength is monotonic ");
    }

    _isWavelengthSet = true;
}

template<typename ImageT, typename MaskT, typename VarianceT>
PTR( Spectrum< ImageT, MaskT, VarianceT > )
SpectrumSet<ImageT, MaskT, VarianceT>::getSpectrum( const std::size_t i ) const
{
    if (i >= _spectra->size()){
        string message("SpectrumSet::getSpectrum(i=");
        message += to_string(i) + "): ERROR: i >= _spectra->size()=" + to_string(_spectra->size());
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return PTR( Spectrum< ImageT, MaskT, VarianceT > )( new Spectrum< ImageT, MaskT, VarianceT >( *( _spectra->at( i ) ) ) );
}

template<typename ImageT, typename MaskT, typename VarianceT>
PTR( Spectrum< ImageT, MaskT, VarianceT > )
SpectrumSet<ImageT, MaskT, VarianceT>::getSpectrum( const std::size_t i )
{
    if (i >= _spectra->size()){
        string message("SpectrumSet::getSpectrum(i=");
        message += to_string(i) + "): ERROR: i >= _spectra->size()=" + to_string(_spectra->size());
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return PTR( Spectrum< ImageT, MaskT, VarianceT > )( new Spectrum< ImageT, MaskT, VarianceT >( *( _spectra->at( i ) ) ) );
}

template<typename ImageT, typename MaskT, typename VarianceT>
void
SpectrumSet<ImageT, MaskT, VarianceT>::setSpectrum(std::size_t const i,
                                                   Spectrum<ImageT, MaskT, VarianceT> const& spectrum)
{
  if (i > _spectra->size()){
    string message("SpectrumSet::setSpectrum(i=");
    message += to_string(i) + "): ERROR: i > _spectra->size()=" + to_string(_spectra->size());
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  PTR( Spectrum<ImageT, MaskT, VarianceT> ) spectrumPtr( new Spectrum< ImageT, MaskT, VarianceT >( spectrum ) );

  if ( i == _spectra->size() ){
    _spectra->push_back( spectrumPtr );
  }
  else{
    ( *_spectra )[ i ] = spectrumPtr;
  }
}

template<typename ImageT, typename MaskT, typename VarianceT>
void
SpectrumSet<ImageT, MaskT, VarianceT>::setSpectrum(std::size_t const i,
                                                   PTR( Spectrum<ImageT, MaskT, VarianceT>) const& spectrum )
{
  if (i > _spectra->size()){
    string message("SpectrumSet::setSpectrum(i=");
    message += to_string(i) + "): ERROR: i > _spectra->size()=" + to_string(_spectra->size());
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  PTR( Spectrum<ImageT, MaskT, VarianceT> ) spectrumPtr( new Spectrum< ImageT, MaskT, VarianceT >( *spectrum ) );
  if ( i == _spectra->size() ){
    _spectra->push_back( spectrumPtr );
  }
  else{
    ( *_spectra )[ i ] = spectrumPtr;
  }
}

template<typename ImageT, typename MaskT, typename VarianceT>
void SpectrumSet<ImageT, MaskT, VarianceT>::erase(const std::size_t iStart, const std::size_t iEnd){
  if (iStart >= _spectra->size()){
    string message("SpectrumSet::erase(iStart=");
    message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iStart >= _spectra->size()=" + to_string(_spectra->size());
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  if (iEnd >= _spectra->size()){
    string message("SpectrumSet::erase(iStart=");
    message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iEnd >= _spectra->size()=" + to_string(_spectra->size());
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  if (iEnd > 0){
    if (iStart > iEnd){
      string message("SpectrumSet::erase(iStart=");
      message += to_string(iStart) + ", iEnd=" + to_string(iEnd) + "): ERROR: iStart > iEnd";
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
}


namespace math {
    template< typename T, int I >
    ndarray::Array< T, 2, 1 > createLineList( ndarray::Array< T, 1, I > const& wLen,
                                              ndarray::Array< T, 1, I > const& linesWLen ){
      LOG_LOGGER _log = LOG_GET("pfs.drp.stella.createLineList");
      LOGLS_TRACE(_log, "wLen = " << wLen);
      ndarray::Array< size_t, 1, 1 > ind = math::getIndicesInValueRange( wLen, T( 1 ), T( 15000 ) );
      if (ind.getShape()[0] <= 0){
        string message("ind.getShape()[0](=");
        message += to_string(ind.getShape()[0]) + ") <= 0";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      LOGLS_TRACE(_log, "ind = " << ind.getShape() << ": " << ind);
      ndarray::Array< T, 1, 1 > indT = ndarray::allocate( ind.getShape()[ 0 ] );
      indT.deep() = ind;
      LOGLS_TRACE(_log, "indT = " << indT.getShape() << ": " << indT);
      LOGLS_TRACE(_log, "ind[0] = " << ind[0] << ", ind[ind.getShape()[0](="
                                    << ind.getShape()[0] << ")-1] = " << ind[ind.getShape()[0]-1]);
      ndarray::Array< T, 1, 1 > wavelengths = ndarray::copy( wLen[ ndarray::view( ind[ 0 ], ind[ ind.getShape()[ 0 ] - 1 ] + 1 ) ] );
      for (int i = 0; i < indT.getShape()[ 0 ]; ++i )
          LOGLS_TRACE(_log, "indT[" << i << "] = " << indT[i] << ": wavelengths[" << i << "] = "
                            << wavelengths[i]);
      std::vector< std::string > args( 1 );
      LOGLS_TRACE(_log, "intT = " << indT.getShape() << ": " << indT);
      LOGLS_TRACE(_log, "wavelengths = " << wavelengths.getShape() << ": " << wavelengths);
      LOGLS_TRACE(_log, "linesWLen = " << linesWLen.getShape() << ": " << linesWLen);
      LOGLS_TRACE(_log, "args = " << args.size());
      ndarray::Array< T, 1, 1 > linesPix = math::interPol( indT, wavelengths, linesWLen, args );
      LOGLS_TRACE(_log, "linesWLen = " << linesWLen.getShape() << ": " << linesWLen);
      ndarray::Array< T, 2, 1 > out = ndarray::allocate( linesWLen.getShape()[ 0 ], 2 );
      LOGLS_TRACE(_log, "out = " << out.getShape());
      LOGLS_TRACE(_log, "out[ndarray::view()(0)].getShape() = " << out[ ndarray::view()(0) ].getShape());
      out[ ndarray::view()(0) ] = linesWLen;
      out[ ndarray::view()(1) ] = linesPix;
      LOGLS_TRACE(_log, "out = " << out);
      return out;
    }

    template ndarray::Array< float, 2, 1 > createLineList( ndarray::Array< float, 1, 0 > const&, ndarray::Array< float, 1, 0 > const& );
    template ndarray::Array< float, 2, 1 > createLineList( ndarray::Array< float, 1, 1 > const&, ndarray::Array< float, 1, 1 > const& );

}

/************************************************************************************************************/
/*
 * Explicit instantiations
 */
template class Spectrum<float, lsst::afw::image::MaskPixel, float>;
template class SpectrumSet<float, lsst::afw::image::MaskPixel, float>;
                
}}}
