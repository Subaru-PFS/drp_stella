#include "pfs/drp/stella/Spectra.h"
#include "pfs/drp/stella/cmpfit-1.2/MPFitting_ndarray.h"

namespace pfs { namespace drp { namespace stella {

/** @brief Construct a Spectrum with empty vectors of specified size (default 0)
 */
template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
Spectrum<ImageT, MaskT, VarianceT, WavelengthT>::Spectrum(size_t length,
                                                                           size_t iTrace )
  : _length(length),
    _mask(length, 1),
    _iTrace(iTrace),
    _isWavelengthSet(false),
    _dispCorControl( new DispCorControl )
{
  _spectrum = ndarray::allocate( length );
  _spectrum.deep() = 0.;
  _sky = ndarray::allocate( length );
  _sky.deep() = 0.;
  _covar = ndarray::allocate( length, 3 );
  _covar.deep() = 0.;
  _wavelength = ndarray::allocate( length );
  _wavelength.deep() = 0.;
  _dispersion = ndarray::allocate( length );
  _dispersion.deep() = 0.;
  _dispCoeffs = ndarray::allocate( _dispCorControl->order + 1 );
  _dispCoeffs.deep() = 0.;
  _dispRms = 0.;
  _dispRmsCheck = 0.;
  _nGoodLines = 0;
  _yLow = 0;
  _yHigh = length - 1;
  _nCCDRows = length;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void Spectrum<ImageT, MaskT, VarianceT, WavelengthT>::setSky( ndarray::Array<ImageT, 1, 1> const& sky )
{
  /// Check length of input spectrum
  if (static_cast<size_t>(sky.getShape()[0]) != _length){
    string message("pfsDRPStella::Spectrum::setSky: ERROR: spectrum->size()=");
    message += to_string(sky.getShape()[0]) + string(" != _length=") + to_string(_length);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  _sky.deep() = sky;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void Spectrum<ImageT, MaskT, VarianceT, WavelengthT>::setWavelength( ndarray::Array< WavelengthT, 1, 1 > const& wavelength )
{
  /// Check length of input wavelength
  if (static_cast<size_t>(wavelength.getShape()[0]) != _length){
    string message("pfsDRPStella::Spectrum::setWavelength: ERROR: wavelength->size()=");
    message += to_string(wavelength.getShape()[0]) + string(" != _length=") + to_string(_length);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  _wavelength.deep() = wavelength;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void Spectrum< ImageT, MaskT, VarianceT, WavelengthT >::setDispersion( ndarray::Array< WavelengthT, 1, 1 > const& dispersion )
{
  /// Check length of input wavelength
  if (static_cast<size_t>(dispersion.getShape()[0]) != _length ){
    string message("pfsDRPStella::Spectrum::setDispersion: ERROR: dispersion->size()=");
    message += to_string( dispersion.getShape()[ 0 ]) + string(" != _length=") + to_string( _length );
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  _dispersion.deep() = dispersion;
}

///SpectrumSet
template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::SpectrumSet( size_t nSpectra, size_t length )
        : _spectra( new std::vector< PTR( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > ) >() )
{
  for (size_t i = 0; i < nSpectra; ++i){
    PTR(Spectrum<ImageT, MaskT, VarianceT, WavelengthT>) spec( new Spectrum<ImageT, MaskT, VarianceT, WavelengthT>( length, i ) );
    _spectra->push_back(spec);
  }
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::SpectrumSet( std::vector< PTR( Spectrum< ImageT, MaskT, VarianceT, WavelengthT> > ) const& spectrumVector )
        : _spectra( new std::vector< PTR( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > ) >() )
{
  for (size_t i = 0; i < spectrumVector.size(); ++i){
    _spectra->push_back( spectrumVector.at(i) );
  }
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array< float, 2, 1 > SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getAllFluxes() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 2, 1 > flux = ndarray::allocate( nCCDRows, nFibers );

  flux.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra->at( iFiber );

    int yLow = spectrum->getYLow();
    int yHigh = spectrum->getYHigh();
    if ( yHigh - yLow + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      string message("SpectrumSet::writeFits: spectrum does not have expected shape: yHigh=");
      message += to_string(yHigh) + " - yLow=" + to_string(yLow) + " + 1 (=" + to_string(yHigh - yLow + 1);
      message += + ") = " +to_string(yHigh-yLow + 1) + " != spectrum->getSpectrum().getShape()[ 0 ] = ";
      message += to_string(spectrum->getSpectrum().getShape()[ 0 ]);
      throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,message.c_str());
    }

    flux[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = spectrum->getSpectrum()[ ndarray::view() ];
  }
  return flux;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array< float, 2, 1 > SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getAllWavelengths() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 2, 1 > lambda = ndarray::allocate( nCCDRows, nFibers );

  lambda.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra->at( iFiber );

    int yLow = spectrum->getYLow();
    int yHigh = spectrum->getYHigh();
    if ( yHigh - yLow + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      string message("SpectrumSet::writeFits: spectrum does not have expected shape: yHigh=");
      message += to_string(yHigh) + " - yLow=" +to_string(yLow) + " + 1 (=" + to_string(yHigh - yLow + 1);
      message += ") = " + to_string(yHigh-yLow + 1) + " != spectrum->getSpectrum().getShape()[ 0 ] = ";
      message += to_string(spectrum->getSpectrum().getShape()[ 0 ]);
      throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,message.c_str());
    }

    lambda[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = spectrum->getWavelength()[ ndarray::view() ];
  }
  return lambda;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array< float, 2, 1 > SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getAllDispersions() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 2, 1 > dispersion = ndarray::allocate( nCCDRows, nFibers );

  dispersion.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra->at( iFiber );

    int yLow = spectrum->getYLow();
    int yHigh = spectrum->getYHigh();
    if ( yHigh - yLow + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      string message("SpectrumSet::writeFits: spectrum does not have expected shape: yHigh=");
      message += to_string(yHigh) + " - yLow=" + to_string(yLow) + " + 1 (=" + to_string(yHigh - yLow + 1);
      message += ") = " + to_string(yHigh-yLow + 1) + " != spectrum->getSpectrum().getShape()[ 0 ] = ";
      message += to_string(spectrum->getSpectrum().getShape()[ 0 ]);
      throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,message.c_str());
    }

    dispersion[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = spectrum->getDispersion()[ ndarray::view() ];
  }
  return dispersion;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array< int, 2, 1 > SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getAllMasks() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< int, 2, 1 > mask = ndarray::allocate( nCCDRows, nFibers );

  mask.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra->at( iFiber );

    int yLow = spectrum->getYLow();
    int yHigh = spectrum->getYHigh();
    if ( yHigh - yLow + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      string message("SpectrumSet::writeFits: spectrum does not have expected shape: yHigh=");
      message += to_string(yHigh) + " - yLow=" +to_string(yLow) + " + 1 (=" + to_string(yHigh - yLow + 1);
      message += ") = " + to_string(yHigh-yLow + 1) + " != spectrum->getSpectrum().getShape()[ 0 ] = ";
      message += to_string(spectrum->getSpectrum().getShape()[ 0 ]);
      throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,message.c_str());
    }
  }
  return mask;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array< float, 2, 1 > SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getAllSkies() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 2, 1 > sky = ndarray::allocate( nCCDRows, nFibers );

  sky.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra->at( iFiber );

    int yLow = spectrum->getYLow();
    int yHigh = spectrum->getYHigh();
    if ( yHigh - yLow + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      cout << "SpectrumSet::getAllSkies: yHigh=" << yHigh << " - yLow=" << yLow << " + 1 (=" << yHigh - yLow + 1 << ") = " << yHigh-yLow + 1 << " != spectrum->getSpectrum().getShape()[ 0 ] = " << spectrum->getSpectrum().getShape()[ 0 ] << endl;
      throw LSST_EXCEPT(
        lsst::pex::exceptions::LogicError,
        "SpectrumSet::getAllSkies: spectrum does not have expected shape"
      );
    }

    sky[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = spectrum->getSky()[ ndarray::view() ];
  }
  return sky;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array< float, 2, 1 > SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getAllVariances() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 2, 1 > var = ndarray::allocate( nCCDRows, nFibers );

  var.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra->at( iFiber );

    int yLow = spectrum->getYLow();
    int yHigh = spectrum->getYHigh();
    if ( yHigh - yLow + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      string message("SpectrumSet::writeFits: spectrum does not have expected shape: yHigh=");
      message += to_string(yHigh) + " - yLow=" + to_string(yLow) + " + 1 (=" + to_string(yHigh - yLow + 1);
      message += ") = " + to_string(yHigh-yLow + 1) + " != spectrum->getSpectrum().getShape()[ 0 ] = ";
      message += to_string(spectrum->getSpectrum().getShape()[ 0 ]);
      throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,message.c_str());
    }

    var[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = spectrum->getVariance()[ ndarray::view() ];
  }
  return var;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array< float, 3, 1 > SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getAllCovars() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 3, 1 > covar = ndarray::allocate( nCCDRows, 3, nFibers );

  covar.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra->at( iFiber );

    int yLow = spectrum->getYLow();
    int yHigh = spectrum->getYHigh();
    if ( yHigh - yLow + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      string message("SpectrumSet::writeFits: spectrum does not have expected shape: yHigh=");
      message += to_string(yHigh) + " - yLow=" + to_string(yLow) + " + 1 (=" + to_string(yHigh - yLow + 1);
      message += ") = " + to_string(yHigh-yLow + 1) + " != spectrum->getSpectrum().getShape()[ 0 ] = ";
      message += to_string(spectrum->getSpectrum().getShape()[ 0 ]);
      throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,message.c_str());
    }
    #ifdef __DEBUG_GETALLCOVARS__
      cout << "covar[ ndarray::view( " << yLow << ", " << yHigh + 1 << " )( )( " << iFiber << " ) ].getShape() = " << covar[ ndarray::view( yLow, yHigh + 1 )( )( iFiber ) ].getShape() << ", spectrum->getCovar().getShape() = " << spectrum->getCovar().getShape() << endl;
    #endif
    covar[ ndarray::view( yLow, yHigh + 1 )( )( iFiber ) ] = spectrum->getCovar()[ ndarray::view() ];
  }
  return covar;
}

template< typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT >
Spectrum< ImageT, MaskT, VarianceT, WavelengthT >::Spectrum( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > & spectrum,
                                                                                  std::size_t iTrace,
                                                                                  bool deep ) 
:       _yLow( spectrum.getYLow() ),
        _yHigh( spectrum.getYHigh() ),
        _length( spectrum.getLength() ),
        _nCCDRows( spectrum.getNCCDRows() ),
        _spectrum( spectrum.getSpectrum() ),
        _sky( spectrum.getSky() ),
        _mask( spectrum.getMask(), deep ),
        _covar( spectrum.getCovar() ),
        _wavelength( spectrum.getWavelength() ),
        _dispersion( spectrum.getDispersion() ),
        _iTrace( spectrum.getITrace() ),
        _dispCoeffs( spectrum.getDispCoeffs() ),
        _dispRms( spectrum.getDispRms() ),
        _dispRmsCheck( spectrum.getDispRmsCheck() ),
        _nGoodLines( spectrum.getNGoodLines() ),
        _isWavelengthSet( spectrum.isWavelengthSet() ),
        _dispCorControl( spectrum.getDispCorControl() )
{
    if ( deep ){
        /// allocate memory
        _spectrum = ndarray::allocate(spectrum.getSpectrum().getShape()[0]);
        _sky = ndarray::allocate(spectrum.getSky().getShape()[0]);
        _covar = ndarray::allocate(spectrum.getCovar().getShape());
        _wavelength = ndarray::allocate(spectrum.getWavelength().getShape()[0]);
        _dispersion = ndarray::allocate(spectrum.getDispersion().getShape()[0]);
        _dispCoeffs = ndarray::allocate(spectrum.getDispCoeffs().getShape()[0]);

        /// copy variables
        _spectrum.deep() = spectrum.getSpectrum();
        _sky.deep() = spectrum.getSky();
        _covar.deep() = spectrum.getCovar();
        _wavelength.deep() = spectrum.getWavelength();
        _dispersion.deep() = spectrum.getDispersion();
        _dispCoeffs.deep() = spectrum.getDispCoeffs();
    }
    if (iTrace != 0)
        _iTrace = iTrace;
    _mask.addMaskPlane("REJECTED_LINES");
}

template< typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT >
Spectrum< ImageT, MaskT, VarianceT, WavelengthT >::Spectrum( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > const& spectrum) //,
                                                                                  //int i ) 
:       _yLow( spectrum.getYLow() ),
        _yHigh( spectrum.getYHigh() ),
        _length( spectrum.getLength() ),
        _nCCDRows( spectrum.getNCCDRows() ),
        _mask(spectrum.getMask()),
        _iTrace( spectrum.getITrace() ),
        _dispRms( spectrum.getDispRms() ),
        _dispRmsCheck( spectrum.getDispRmsCheck() ),
        _nGoodLines( spectrum.getNGoodLines() ),
        _isWavelengthSet( spectrum.isWavelengthSet() ),
        _dispCorControl( spectrum.getDispCorControl() )
{
    /// allocate memory
    _spectrum = ndarray::allocate(spectrum.getSpectrum().getShape()[0]);
    _sky = ndarray::allocate(spectrum.getSky().getShape()[0]);
    _covar = ndarray::allocate(spectrum.getCovar().getShape());
    _wavelength = ndarray::allocate(spectrum.getWavelength().getShape()[0]);
    _dispersion = ndarray::allocate(spectrum.getDispersion().getShape()[0]);
    _dispCoeffs = ndarray::allocate(spectrum.getDispCoeffs().getShape()[0]);

    /// copy variables
    _spectrum.deep() = spectrum.getSpectrum();
    _sky.deep() = spectrum.getSky();
    _covar.deep() = spectrum.getCovar();
    _wavelength.deep() = spectrum.getWavelength();
    _dispersion.deep() = spectrum.getDispersion();
    _dispCoeffs.deep() = spectrum.getDispCoeffs();
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void
Spectrum<ImageT, MaskT, VarianceT, WavelengthT>::setSpectrum( ndarray::Array<ImageT, 1, 1> const& spectrum )
{
  /// Check length of input spectrum
  if (static_cast<std::size_t>(spectrum.getShape()[0]) != _length) {
    string message("pfs::drp::stella::Spectrum::setSpectrum: ERROR: spectrum->size()=");
    message += to_string(spectrum.getShape()[0]) + string(" != _length=") + to_string(_length);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _spectrum.deep() = spectrum;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array<VarianceT, 1, 1>
Spectrum<ImageT, MaskT, VarianceT, WavelengthT>::getVariance() const
{
    ndarray::Array< VarianceT, 1, 1 > variance = ndarray::allocate( _covar.getShape()[ 0 ] );
    variance.deep() = _covar[ ndarray::view( )( 1 ) ];
    return variance;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array<VarianceT, 1, 1>
Spectrum<ImageT, MaskT, VarianceT, WavelengthT>::getVariance()
{
    ndarray::Array< VarianceT, 1, 1 > variance = ndarray::allocate(_covar.getShape()[0]);
    variance[ndarray::view()] = _covar[ ndarray::view( )( 1 ) ];
    return variance;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void
Spectrum<ImageT, MaskT, VarianceT, WavelengthT>::setVariance( ndarray::Array<VarianceT, 1, 1> const& variance )
{
  /// Check length of input variance
  if (static_cast<std::size_t>(variance.getShape()[0]) != _length) {
    string message("pfs::drp::stella::Spectrum::setVariance: ERROR: variance->size()=");
    message += to_string( variance.getShape()[ 0 ] ) + string( " != _length=" ) + to_string( _length );
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _covar[ ndarray::view()(1) ] = variance;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void
Spectrum<ImageT, MaskT, VarianceT, WavelengthT>::setCovar(const ndarray::Array<VarianceT, 2, 1> & covar )
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

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void
Spectrum<ImageT, MaskT, VarianceT, WavelengthT>::setMask(const afwImage::Mask<MaskT> & mask)
{
  /// Check length of input mask
  if (static_cast<std::size_t>(mask.getWidth()) != _length){
    string message("pfs::drp::stella::Spectrum::setMask: ERROR: mask.getWidth()=");
    message += to_string(mask.getWidth()) + string(" != _length=") + to_string(_length);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _mask = mask;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void
Spectrum<ImageT, MaskT, VarianceT, WavelengthT>::setLength(const std::size_t length)
{
    #ifdef __DEBUG_SETLENGTH__
      cout << "pfs::drp::stella::Spectrum::setLength: starting to set _length to " << length << endl;
    #endif
    math::resize(_spectrum, length);
    #ifdef __DEBUG_SETLENGTH__
      cout << "pfs::drp::stella::Spectrum::setLength: _spectrum resized to " << _spectrum.getShape()[0] << endl;
    #endif
    _mask = afwImage::Mask<MaskT>(length, 1);
    #ifdef __DEBUG_SETLENGTH__
      cout << "pfs::drp::stella::Spectrum::setLength: _mask resized to " << _mask.getWidth() << endl;
    #endif
    math::resize(_covar, length, 3);
    #ifdef __DEBUG_SETLENGTH__
      cout << "pfs::drp::stella::Spectrum::setLength: _covar resized to " << _covar.getShape()[0] << "x" << _covar.getShape()[1] << endl;
    #endif
    math::resize(_wavelength, length);
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
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void
Spectrum< ImageT, MaskT, VarianceT, WavelengthT >::setYLow( const std::size_t yLow )
{
  if ( yLow > _nCCDRows ){
    string message("pfs::drp::stella::Spectrum::setYLow: ERROR: yLow=");
    message += to_string( yLow ) + string(" > _nCCDRows=") + to_string(_nCCDRows);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _yLow = yLow;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void
Spectrum<ImageT, MaskT, VarianceT, WavelengthT>::setYHigh(const std::size_t yHigh)
{
  if ( yHigh > _nCCDRows ){
    _nCCDRows = _yLow + yHigh;
  }
  _yHigh = yHigh;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void
Spectrum<ImageT, MaskT, VarianceT, WavelengthT>::setNCCDRows(const std::size_t nCCDRows)
{
  if ( _yLow > nCCDRows ){
    string message("pfs::drp::stella::Spectrum::setYLow: ERROR: _yLow=");
    message += to_string( _yLow ) + string(" > nCCDRows=") + to_string(nCCDRows);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _nCCDRows = nCCDRows;
}

template< typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT >
template< typename T >
ndarray::Array<float, 1, 1 >
Spectrum<ImageT, MaskT, VarianceT, WavelengthT>::hIdentify( ndarray::Array< T, 2, 1 > const& lineList )
{
  LOG_LOGGER _log = LOG_GET("Spectra::identify");
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
                  if ( lineList.getShape()[ 1 ] > 2 ){
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

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void
Spectrum< ImageT, MaskT, VarianceT, WavelengthT >::setDispCoeffs( ndarray::Array< float, 1, 1 > const& dispCoeffs )
{
  /// Check length of input dispCoeffs
  if (dispCoeffs.getShape()[0] != ( _dispCorControl->order + 1 ) ){
    string message("pfsDRPStella::Spectrum::setDispCoeffs: ERROR: dispCoeffs.size()=");
    message += to_string(dispCoeffs.getShape()[0]) + string(" != _dispCorControl->order + 1 =") + to_string( _dispCorControl->order + 1 );
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _dispCoeffs = ndarray::allocate( dispCoeffs.getShape()[ 0 ] );
  _dispCoeffs.deep() = dispCoeffs;
  cout << "pfsDRPStella::setDispCoeffs: _dispCoeffs set to " << _dispCoeffs << endl;
}

template< typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT >
template< typename T >
void
Spectrum<ImageT, MaskT, VarianceT, WavelengthT>::identify( ndarray::Array< T, 2, 1 > const& lineList,
                                                                                     DispCorControl const& dispCorControl,
                                                                                     std::size_t nLinesCheck ){
    LOG_LOGGER _log = LOG_GET("Spectra::identify");

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

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > )
SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getSpectrum( const std::size_t i ) const
{
    if (i >= _spectra->size()){
        string message("SpectrumSet::getSpectrum(i=");
        message += to_string(i) + "): ERROR: i >= _spectra->size()=" + to_string(_spectra->size());
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return PTR( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > )( new Spectrum< ImageT, MaskT, VarianceT, WavelengthT >( *( _spectra->at( i ) ) ) );
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > )
SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::getSpectrum( const std::size_t i )
{
    if (i >= _spectra->size()){
        string message("SpectrumSet::getSpectrum(i=");
        message += to_string(i) + "): ERROR: i >= _spectra->size()=" + to_string(_spectra->size());
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return PTR( Spectrum< ImageT, MaskT, VarianceT, WavelengthT > )( new Spectrum< ImageT, MaskT, VarianceT, WavelengthT >( *( _spectra->at( i ) ) ) );
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void
SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::setSpectrum(std::size_t const i,
                                                                                          Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrum)
{
  if (i > _spectra->size()){
    string message("SpectrumSet::setSpectrum(i=");
    message += to_string(i) + "): ERROR: i > _spectra->size()=" + to_string(_spectra->size());
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  PTR( Spectrum<ImageT, MaskT, VarianceT, WavelengthT> ) spectrumPtr( new Spectrum< ImageT, MaskT, VarianceT, WavelengthT >( spectrum ) );

  if ( i == _spectra->size() ){
    _spectra->push_back( spectrumPtr );
  }
  else{
    ( *_spectra )[ i ] = spectrumPtr;
  }
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void
SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::setSpectrum(std::size_t const i,
                                                                PTR( Spectrum<ImageT, MaskT, VarianceT, WavelengthT>) const& spectrum )
{
  if (i > _spectra->size()){
    string message("SpectrumSet::setSpectrum(i=");
    message += to_string(i) + "): ERROR: i > _spectra->size()=" + to_string(_spectra->size());
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  PTR( Spectrum<ImageT, MaskT, VarianceT, WavelengthT> ) spectrumPtr( new Spectrum< ImageT, MaskT, VarianceT, WavelengthT >( *spectrum ) );
  if ( i == _spectra->size() ){
    _spectra->push_back( spectrumPtr );
  }
  else{
    ( *_spectra )[ i ] = spectrumPtr;
  }
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
void SpectrumSet<ImageT, MaskT, VarianceT, WavelengthT>::erase(const std::size_t iStart, const std::size_t iEnd){
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

    template< typename T, typename U >
    StretchAndCrossCorrelateSpecResult< T, U > stretchAndCrossCorrelateSpec( ndarray::Array< T, 1, 1 > const& spec,
                                                                             ndarray::Array< T, 1, 1 > const& specRef,
                                                                             ndarray::Array< U, 2, 1 > const& lineList_WLenPix,
                                                                             DispCorControl const& dispCorControl ){
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
        cout << "stretchAndCrossCorrelateSpec: spec = " << spec.getShape() << ": " << spec << endl;
        cout << "stretchAndCrossCorrelateSpec: specRef = " << specRef.getShape() << ": " << specRef << endl;
        cout << "stretchAndCrossCorrelateSpec: lineList_WLenPix = " << lineList_WLenPix.getShape() << ": " << lineList_WLenPix << endl;
      #endif
      float fac = float( specRef.getShape()[ 0 ] ) / float( spec.getShape()[ 0 ] );
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
      stretchAndCrossCorrelateSpecResult.specPieces = ndarray::allocate( ndarray::makeVector( dispCorControl.stretchMaxLength, 2, dispCorControl.nCalcs ) );
      stretchAndCrossCorrelateSpecResult.specPieces.deep() = 0.;
      int nCalcs = dispCorControl.nCalcs;
      if ( nCalcs < spec.getShape()[ 0 ] / dispCorControl.lengthPieces ){
        nCalcs = 2 * int( spec.getShape()[ 0 ] / dispCorControl.lengthPieces );
        cout << "stretchAndCrossCorrelate: Warning: dispCorControl.nCalcs(=" << dispCorControl.nCalcs << ") < spec.getShape()[0](=" << spec.getShape()[ 0 ] << ") / dispCorControl.lengthPieces(=" << dispCorControl.lengthPieces << ") => nCalcs set to " << nCalcs << endl;
      }
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
        cout << "stretchAndCrossCorrelateSpec: nCalcs = " << nCalcs << endl;
      #endif
      ndarray::Array< float, 1, 1 > chiSqMin_Stretch = ndarray::allocate( nCalcs );
      chiSqMin_Stretch.deep() = 0.;
      ndarray::Array< float, 1, 1 > chiSqMin_Shift = ndarray::allocate( nCalcs );
      chiSqMin_Shift.deep() = 0.;
      ndarray::Array< float, 1, 1 > xCenter = ndarray::allocate( nCalcs );
      xCenter.deep() = 0.;
      ndarray::Array< float, 1, 1 > specPiece = ndarray::allocate( dispCorControl.lengthPieces );
      ndarray::Array< float, 1, 1 > specRefPiece = ndarray::allocate( dispCorControl.lengthPieces );
      int start = 0;
      int end = 0;
      ndarray::Array< float, 2, 1 > specPieceStretched_MinChiSq;
      ndarray::Array< float, 2, 1 > lineList_Pixels_AllPieces = ndarray::allocate( lineList_WLenPix.getShape()[ 0 ], nCalcs );
      lineList_Pixels_AllPieces.deep() = 0.;
      ndarray::Array< float, 1, 1 > x = indGenNdArr( float( specRef.getShape()[ 0 ] ) );
      ndarray::Array< float, 1, 1 > xPiece;
      ndarray::Array< float, 1, 1 > xPieceStretched;

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
        xCenter[ i_run ] = float( start ) + ( float( end - start ) / 2. );
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
          cout << "stretchAndCrossCorrelateSpec: i_run = " << i_run << ": xCenter = " << xCenter[ i_run ] << endl;
        #endif

        specPiece = ndarray::allocate( end - start + 1 );
        specPiece.deep() = stretchedSpec[ ndarray::view( start, end + 1 ) ];
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
          cout << "stretchAndCrossCorrelateSpec: i_run = " << i_run << ": specPiece = " << specPiece.getShape() << ": " << specPiece << endl;
          cout << "stretchAndCrossCorrelateSpec: i_run = " << i_run << ": specRefPiece = " << specRefPiece.getShape() << endl;
        #endif
        specRefPiece = ndarray::allocate( end - start + 1 );
        specRefPiece.deep() = specRef[ ndarray::view( start, end + 1 ) ];
        #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC__
          cout << "stretchAndCrossCorrelateSpec: i_run = " << i_run << ": specRefPiece = " << specRefPiece.getShape() << ": " << specRefPiece << endl;
        #endif
        /// stretch and crosscorrelate pieces
        StretchAndCrossCorrelateResult< float > stretchAndCrossCorrelateResult = stretchAndCrossCorrelate( specPiece,
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

        float weightLeft = 0.;
        float weightRight = 0.;
        ndarray::Array< float, 1, 1 > lineListPix = ndarray::allocate( lineList_WLenPix.getShape()[ 0 ] );
        auto itTemp = lineListPix.begin();
        for ( auto itList = lineList_WLenPix.begin(); itList != lineList_WLenPix.end(); ++itList, ++itTemp ){
          auto itListCol = itList->begin() + 1;
          *itTemp = float( *itListCol );//lineList_WLenPix[ ndarray::view()( 1 ) ] );
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
      ndarray::Array< float, 1, 1 > tempArr = ndarray::allocate( lineList_Pixels_AllPieces.getShape()[ 1 ] );
      for (int i_line=0; i_line < lineList_WLenPix.getShape()[ 0 ]; i_line++){
        tempArr[ ndarray::view() ] = lineList_Pixels_AllPieces[ ndarray::view( i_line )() ];
        ndarray::Array< int, 1, 1 > whereVec = where<float, int>( tempArr,
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
      ndarray::Array< float, 2, 1 > dist = ndarray::allocate( lineList_Pixels_AllPieces.getShape() );
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
      ndarray::Array< int, 2, 1 > whereArr = where<float, int>(lineList_Pixels_AllPieces,
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
      ndarray::Array< float, 1, 1 > dist_SubArr = getSubArray( dist, indWhereArr );
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
        cout << "stretchAndCrossCorrelateSpec: dist_SubArr = " << dist_SubArr << endl;
      #endif
      float medianDiff = median( dist_SubArr );
      ndarray::Array< float, 1, 1 > sorted = ndarray::allocate( dist_SubArr.getShape()[ 0 ] );
      sorted.deep() = dist_SubArr;
      std::sort( sorted.begin(), sorted.end() );
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
        cout << "stretchAndCrossCorrelateSpec: medianDiff = " << medianDiff << endl;
        cout << "stretchAndCrossCorrelateSpec: sorted = " << sorted << endl;
      #endif
      ndarray::Array< float, 1, 1 > dist_Temp = ndarray::allocate( dist_SubArr.getShape()[ 0 ] - 4 );
      dist_Temp = sorted[ ndarray::view( 2, dist_SubArr.getShape()[ 0 ] - 2 ) ];
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
        cout << "stretchAndCrossCorrelateSpec: dist_Temp = " << dist_Temp << endl;
      #endif
      ndarray::Array< float, 1, 1 > moments = moment( dist_Temp, 2 );
      float stdDev_Diff = moments[ 1 ];
      #ifdef __DEBUG_STRETCHANDCROSSCORRELATESPEC_LINELIST__
        cout << "stretchAndCrossCorrelateSpec: stdDev_Diff = " << stdDev_Diff << endl;
      #endif
      ndarray::Array< float, 1, 1 > tempDist = ndarray::copy( dist_SubArr - medianDiff );
      for ( auto itDist = tempDist.begin(); itDist != tempDist.end(); ++itDist )
        *itDist = std::fabs( *itDist );
      ndarray::Array< int, 1, 1 > whereVec = where<float, int>(tempDist,
                                                               ">",
                                                               3. * stdDev_Diff,
                                                               1,
                                                               0);
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
        ndarray::Array< float, 1, 1 > tempList = ndarray::allocate( lineList_Pixels_AllPieces.getShape()[ 1 ] );
        for (int i_line=0; i_line < lineList_WLenPix.getShape()[ 0 ]; i_line++){
          tempList = lineList_Pixels_AllPieces[ ndarray::view(i_line)() ];
          for ( auto itTemp = tempList.begin(); itTemp != tempList.end(); ++itTemp )
            *itTemp = std::fabs( *itTemp );
          ndarray::Array< int, 1, 1 > whereVec = where<float, int>(tempList,
                                                                   ">",
                                                                   0.001,
                                                                   1,
                                                                   0);
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

    template< typename T, int I >
    ndarray::Array< T, 2, 1 > createLineList( ndarray::Array< T, 1, I > const& wLen,
                                              ndarray::Array< T, 1, I > const& linesWLen ){
      ndarray::Array< size_t, 1, 1 > ind = math::getIndicesInValueRange( wLen, T( 1 ), T( 15000 ) );
      assert(ind.getShape()[0] > 0);
      #ifdef __DEBUG_CREATELINELIST__
        cout << "Spectra::createLineList: ind = " << ind.getShape() << ": " << ind << endl;
      #endif
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
      #ifdef __DEBUG_CREATELINELIST__
        cout << "Spectra::createLineList: intT = " << indT.getShape() << ": " << indT << endl;
        cout << "Spectra::createLineList: wavelengths = " << wavelengths.getShape() << ": " << wavelengths << endl;
        cout << "Spectra::createLineList: linesWLen = " << linesWLen.getShape() << ": " << linesWLen << endl;
        cout << "Spectra::createLineList: args = " << args.size() << endl;
      #endif
      ndarray::Array< T, 1, 1 > linesPix = math::interPol( indT, wavelengths, linesWLen, args );
      #ifdef __DEBUG_CREATELINELIST__
        cout << "Spectra::createLineList: linesWLen = " << linesWLen.getShape() << ": " << linesWLen << endl;
      #endif
      ndarray::Array< T, 2, 1 > out = ndarray::allocate( linesWLen.getShape()[ 0 ], 2 );
      #ifdef __DEBUG_CREATELINELIST__
        cout << "Spectra::createLineList: out = " << out.getShape() << endl;
        cout << "Spectra::createLineList: out[ ndarray::view( )( 0 ) ].getShape() = " << out[ ndarray::view( )( 0 ) ].getShape() << endl;
      #endif
      out[ ndarray::view( )( 0 ) ] = linesWLen;//[ ndarray::view( ) ];
      out[ ndarray::view( )( 1 ) ] = linesPix;//[ ndarray::view( ) ];
      #ifdef __DEBUG_CREATELINELIST__
        cout << "Spectra::createLineList: out = " << out << endl;
      #endif
      return out;
    }

    template ndarray::Array< float, 2, 1 > createLineList( ndarray::Array< float, 1, 0 > const&, ndarray::Array< float, 1, 0 > const& );
    template ndarray::Array< float, 2, 1 > createLineList( ndarray::Array< float, 1, 1 > const&, ndarray::Array< float, 1, 1 > const& );

    template StretchAndCrossCorrelateSpecResult< float, float > stretchAndCrossCorrelateSpec( ndarray::Array< float, 1, 1 > const&,
                                                                                              ndarray::Array< float, 1, 1 > const&,
                                                                                              ndarray::Array< float, 2, 1 > const&,
                                                                                              DispCorControl const& );
}

/************************************************************************************************************/
/*
 * Explicit instantiations
 */
template class Spectrum<float, lsst::afw::image::MaskPixel, float, float>;
template void Spectrum<float, lsst::afw::image::MaskPixel, float>::identify(ndarray::Array<float, 2, 1> const&,
                                                                            DispCorControl const&,
                                                                            std::size_t);
template class SpectrumSet<float, lsst::afw::image::MaskPixel, float, float>;
                
}}}
