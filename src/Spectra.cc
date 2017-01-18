#include "pfs/drp/stella/Spectra.h"
#include "lsst/log/Log.h"

namespace pfsDRPStella = pfs::drp::stella;

/** @brief Construct a Spectrum with empty vectors of specified size (default 0)
 */
template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::Spectrum(size_t length,
                                                                           size_t iTrace )
  : _length(length),
    _mask(length, 1),
    _iTrace(iTrace),
    _isWavelengthSet(false),
    _dispCorControl( new DispCorControl )
{
  _spectrum = ndarray::allocate( length );
  _sky = ndarray::allocate( length );
  _covar = ndarray::allocate( length, 3 );
  _wavelength = ndarray::allocate( length );
  _dispersion = ndarray::allocate( length );
  _dispCoeffs = ndarray::allocate( _dispCorControl->order + 1 );
  _dispRms = 0.;
  _yLow = 0;
  _yHigh = length - 1;
  _nCCDRows = length;
}

//template class pfsDRPStella::Spectrum<float, unsigned int, float, float>;
//template class pfsDRPStella::Spectrum<double, unsigned int, float, float>;
template class pfsDRPStella::Spectrum<float, unsigned short, float, float>;
template class pfsDRPStella::Spectrum<double, unsigned short, float, float>;
//template class pfsDRPStella::Spectrum<float, int, float, float>;
//template class pfsDRPStella::Spectrum<double, int, float, float>;
//template class pfsDRPStella::Spectrum<double, int, double, double>;
//template class pfsDRPStella::Spectrum<float, unsigned int, double, double>;
//template class pfsDRPStella::Spectrum<double, unsigned int, double, double>;
template class pfsDRPStella::Spectrum<float, unsigned short, double, double>;
template class pfsDRPStella::Spectrum<double, unsigned short, double, double>;

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setSky( ndarray::Array<SpectrumT, 1, 1> const& sky )
{
  /// Check length of input spectrum
  if (static_cast<size_t>(sky.getShape()[0]) != _length){
    string message("pfsDRPStella::Spectrum::setSky: ERROR: spectrum->size()=");
    message += to_string(sky.getShape()[0]) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  _sky.deep() = sky;
  return true;
}

template bool pfsDRPStella::Spectrum< double, unsigned short, float, float >::setSky( ndarray::Array< double, 1, 1 > const& );
template bool pfsDRPStella::Spectrum< float, unsigned short, float, float >::setSky( ndarray::Array< float, 1, 1 > const& );

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::setWavelength( ndarray::Array< WavelengthT, 1, 1 > const& wavelength )
{
  /// Check length of input wavelength
  if (static_cast<size_t>(wavelength.getShape()[0]) != _length){
    string message("pfsDRPStella::Spectrum::setWavelength: ERROR: wavelength->size()=");
    message += to_string(wavelength.getShape()[0]) + string(" != _length=") + to_string(_length);
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  _wavelength.deep() = wavelength;
  return true;
}

template bool pfsDRPStella::Spectrum< double, unsigned short, float, float >::setWavelength( ndarray::Array< float, 1, 1 > const& );
template bool pfsDRPStella::Spectrum< float, unsigned short, float, float >::setWavelength( ndarray::Array< float, 1, 1 > const& );

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
bool pfsDRPStella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT >::setDispersion( ndarray::Array< WavelengthT, 1, 1 > const& dispersion )
{
  /// Check length of input wavelength
  if (static_cast<size_t>(dispersion.getShape()[0]) != _length ){
    string message("pfsDRPStella::Spectrum::setDispersion: ERROR: dispersion->size()=");
    message += to_string( dispersion.getShape()[ 0 ]) + string(" != _length=") + to_string( _length );
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  _dispersion.deep() = dispersion;
  return true;
}

template bool pfsDRPStella::Spectrum< double, unsigned short, float, float >::setDispersion( ndarray::Array< float, 1, 1 > const& );
template bool pfsDRPStella::Spectrum< float, unsigned short, float, float >::setDispersion( ndarray::Array< float, 1, 1 > const& );

//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<float, unsigned int, float, float>::hIdentify(ndarray::Array< float, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<double, unsigned int, float, float>::hIdentify(ndarray::Array< float, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<float, unsigned short, float, float>::hIdentify(ndarray::Array< float, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<double, unsigned short, float, float>::hIdentify(ndarray::Array< float, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<float, unsigned int, float, float>::hIdentify(ndarray::Array< double, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<double, unsigned int, float, float>::hIdentify(ndarray::Array< double, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<float, unsigned short, float, float>::hIdentify(ndarray::Array< double, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<double, unsigned short, float, float>::hIdentify(ndarray::Array< double, 2, 1 > const& );
//template<> template<> ndarray::Array< double, 1, 1 > pfsDRPStella::Spectrum<float, int, float, float>::hIdentify(ndarray::Array< float, 2, 1 > const& );

//template< typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT >
//template< typename T >
//bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::identify( ndarray::Array< T, 2, 1 > const& lineList,
//                                                                                 ndarray::Array< T, 1, 0 > const& predicted,
//                                                                                 ndarray::Array< T, 1, 0 > const& predictedWLenAllPix,
//                                                                                 DispCorControl const& dispCorControl,
//                                                                                 size_t nLinesCheck )

//template< typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT >
//template< typename T >
//bool pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>::identify( ndarray::Array< T, 2, 1 > const& lineList,
//                                                                                 DispCorControl const& dispCorControl,
//                                                                                 size_t nLinesCheck )

namespace {

// Helper functions for SpectrumSet FITS ctor.

void checkExtType( lsst::afw::fits::Fits & fitsfile,
                   PTR(lsst::daf::base::PropertySet) metadata,
                   std::string const & expected) {
    try {
        std::string exttype = boost::algorithm::trim_right_copy(metadata->getAsString("EXTTYPE"));
        if (exttype != "" && exttype != expected) {
            throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                              (boost::format("Reading %s (hdu %d) Expected EXTTYPE==\"%s\", saw \"%s\"") %
                               expected % fitsfile.getFileName() % fitsfile.getHdu() % exttype).str());
        }
        metadata->remove("EXTTYPE");
    } catch(lsst::pex::exceptions::NotFoundError) {
        LOGL_WARN("afw.image.MaskedImage", "Expected extension type not found: %s", expected.c_str());
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
        : _spectra( new std::vector< PTR( Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) >() )
{
  for (size_t i = 0; i < nSpectra; ++i){
    PTR(pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>) spec( new pfsDRPStella::Spectrum<SpectrumT, MaskT, VarianceT, WavelengthT>( length, i ) );
    _spectra->push_back(spec);
  }
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::SpectrumSet( PTR( std::vector< PTR( Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT> ) > ) const& spectrumVector )
        : _spectra( new std::vector< PTR( Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) >() )
{
  for (size_t i = 0; i < spectrumVector->size(); ++i){
    _spectra->push_back( spectrumVector->at(i) );
  }
}

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
      _spectra( new std::vector< PTR(Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) >() )
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
      _spectra( new std::vector< PTR(Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) >( ) )
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
      _spectra(new std::vector< PTR(Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) >( ) )
{
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

    for (size_t i = 0; i < metadata->names().size(); ++i){
      std::cout << "Image::Image(fitsfile, metadata,...): metadata.names()[" << i << "] = " << metadata->names()[i] << std::endl;
      if (metadata->names()[i].compare("EXPTIME") == 0)
        std::cout << "Image::Image(fitsfile, metadata,...): metadata->get(EXPTIME) = " << metadata->getAsDouble("EXPTIME") << std::endl;
    }
    for (size_t i = 0; i < metadata->paramNames().size(); ++i){
      std::cout << "Image::Image(fitsfile, metadata,...): metadata.paramNames()[" << i << "] = " << metadata->paramNames()[i] << std::endl;
      if (metadata->paramNames()[i].compare("EXPTIME") == 0)
        std::cout << "Image::Image(fitsfile, metadata,...): metadata->get(EXPTIME) = " << metadata->getAsDouble("EXPTIME") << std::endl;
    }
    for (size_t i = 0; i < metadata->propertySetNames().size(); ++i)
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
            LOGL_WARN("pfs.drp.stella.SpectrumSet", "Mask unreadable; using default");
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
            LOGL_WARN("pfs.drp.stella.SpectrumSet", "Variance unreadable; using default");
            fitsfile.status = 0;
            _variance.reset(new Variance(_image->getBBox()));
        }
    }*/
    (const_cast< lsst::afw::fits::Fits& >(fitsfile)).setHdu(origHdu);
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array< float, 2, 1 > pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::getAllFluxes() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 2, 1 > flux = ndarray::allocate( nCCDRows, nFibers );

  flux.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( pfs::drp::stella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra->at( iFiber );

    int yLow = spectrum->getYLow();
    int yHigh = spectrum->getYHigh();
    if ( yHigh - yLow + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      cout << "SpectrumSet::writeFits: yHigh=" << yHigh << " - yLow=" << yLow << " + 1 (=" << yHigh - yLow + 1 << ") = " << yHigh-yLow + 1 << " != spectrum->getSpectrum().getShape()[ 0 ] = " << spectrum->getSpectrum().getShape()[ 0 ] << endl;
      throw LSST_EXCEPT(
        lsst::pex::exceptions::LogicError,
        "SpectrumSet::writeFits: spectrum does not have expected shape"
      );
    }

    flux[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = spectrum->getSpectrum()[ ndarray::view() ];
  }
  return flux;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array< float, 2, 1 > pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::getAllWavelengths() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 2, 1 > lambda = ndarray::allocate( nCCDRows, nFibers );

  lambda.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( pfs::drp::stella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra->at( iFiber );

    int yLow = spectrum->getYLow();
    int yHigh = spectrum->getYHigh();
    if ( yHigh - yLow + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      cout << "SpectrumSet::writeFits: yHigh=" << yHigh << " - yLow=" << yLow << " + 1 (=" << yHigh - yLow + 1 << ") = " << yHigh-yLow + 1 << " != spectrum->getSpectrum().getShape()[ 0 ] = " << spectrum->getSpectrum().getShape()[ 0 ] << endl;
      throw LSST_EXCEPT(
        lsst::pex::exceptions::LogicError,
        "SpectrumSet::writeFits: spectrum does not have expected shape"
      );
    }

    lambda[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = spectrum->getWavelength()[ ndarray::view() ];
  }
  return lambda;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array< float, 2, 1 > pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::getAllDispersions() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 2, 1 > dispersion = ndarray::allocate( nCCDRows, nFibers );

  dispersion.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( pfs::drp::stella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra->at( iFiber );

    int yLow = spectrum->getYLow();
    int yHigh = spectrum->getYHigh();
    if ( yHigh - yLow + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      cout << "SpectrumSet::writeFits: yHigh=" << yHigh << " - yLow=" << yLow << " + 1 (=" << yHigh - yLow + 1 << ") = " << yHigh-yLow + 1 << " != spectrum->getSpectrum().getShape()[ 0 ] = " << spectrum->getSpectrum().getShape()[ 0 ] << endl;
      throw LSST_EXCEPT(
        lsst::pex::exceptions::LogicError,
        "SpectrumSet::writeFits: spectrum does not have expected shape"
      );
    }

    dispersion[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = spectrum->getDispersion()[ ndarray::view() ];
  }
  return dispersion;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array< int, 2, 1 > pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::getAllMasks() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< int, 2, 1 > mask = ndarray::allocate( nCCDRows, nFibers );

  mask.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( pfs::drp::stella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra->at( iFiber );

    int yLow = spectrum->getYLow();
    int yHigh = spectrum->getYHigh();
    if ( yHigh - yLow + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      cout << "SpectrumSet::writeFits: yHigh=" << yHigh << " - yLow=" << yLow << " + 1 (=" << yHigh - yLow + 1 << ") = " << yHigh-yLow + 1 << " != spectrum->getSpectrum().getShape()[ 0 ] = " << spectrum->getSpectrum().getShape()[ 0 ] << endl;
      throw LSST_EXCEPT(
        lsst::pex::exceptions::LogicError,
        "SpectrumSet::writeFits: spectrum does not have expected shape"
      );
    }
  }
  return mask;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array< float, 2, 1 > pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::getAllSkies() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 2, 1 > sky = ndarray::allocate( nCCDRows, nFibers );

  sky.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( pfs::drp::stella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra->at( iFiber );

    int yLow = spectrum->getYLow();
    int yHigh = spectrum->getYHigh();
    if ( yHigh - yLow + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      cout << "SpectrumSet::writeFits: yHigh=" << yHigh << " - yLow=" << yLow << " + 1 (=" << yHigh - yLow + 1 << ") = " << yHigh-yLow + 1 << " != spectrum->getSpectrum().getShape()[ 0 ] = " << spectrum->getSpectrum().getShape()[ 0 ] << endl;
      throw LSST_EXCEPT(
        lsst::pex::exceptions::LogicError,
        "SpectrumSet::writeFits: spectrum does not have expected shape"
      );
    }

    sky[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = spectrum->getSky()[ ndarray::view() ];
  }
  return sky;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array< float, 2, 1 > pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::getAllVariances() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 2, 1 > var = ndarray::allocate( nCCDRows, nFibers );

  var.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( pfs::drp::stella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra->at( iFiber );

    int yLow = spectrum->getYLow();
    int yHigh = spectrum->getYHigh();
    if ( yHigh - yLow + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      cout << "SpectrumSet::writeFits: yHigh=" << yHigh << " - yLow=" << yLow << " + 1 (=" << yHigh - yLow + 1 << ") = " << yHigh-yLow + 1 << " != spectrum->getSpectrum().getShape()[ 0 ] = " << spectrum->getSpectrum().getShape()[ 0 ] << endl;
      throw LSST_EXCEPT(
        lsst::pex::exceptions::LogicError,
        "SpectrumSet::writeFits: spectrum does not have expected shape"
      );
    }

    var[ ndarray::view( yLow, yHigh + 1 )( iFiber ) ] = spectrum->getVariance()[ ndarray::view() ];
  }
  return var;
}

template<typename SpectrumT, typename MaskT, typename VarianceT, typename WavelengthT>
ndarray::Array< float, 3, 1 > pfsDRPStella::SpectrumSet<SpectrumT, MaskT, VarianceT, WavelengthT>::getAllCovars() const{
  int nFibers = int( size() );
  int nCCDRows = getSpectrum( 0 )->getNCCDRows();
  /// allocate memory for the array
  ndarray::Array< float, 3, 1 > covar = ndarray::allocate( nCCDRows, 3, nFibers );

  covar.deep() = 0.;

  for ( int iFiber = 0; iFiber < _spectra->size(); ++iFiber ){
    PTR( pfs::drp::stella::Spectrum< SpectrumT, MaskT, VarianceT, WavelengthT > ) spectrum = _spectra->at( iFiber );

    int yLow = spectrum->getYLow();
    int yHigh = spectrum->getYHigh();
    if ( yHigh - yLow + 1 != spectrum->getSpectrum().getShape()[ 0 ] ){
      cout << "SpectrumSet::writeFits: yHigh=" << yHigh << " - yLow=" << yLow << " + 1 (=" << yHigh - yLow + 1 << ") = " << yHigh-yLow + 1 << " != spectrum->getSpectrum().getShape()[ 0 ] = " << spectrum->getSpectrum().getShape()[ 0 ] << endl;
      throw LSST_EXCEPT(
        lsst::pex::exceptions::LogicError,
        "SpectrumSet::writeFits: spectrum does not have expected shape"
      );
    }
    #ifdef __DEBUG_GETALLCOVARS__
      cout << "covar[ ndarray::view( " << yLow << ", " << yHigh + 1 << " )( )( " << iFiber << " ) ].getShape() = " << covar[ ndarray::view( yLow, yHigh + 1 )( )( iFiber ) ].getShape() << ", spectrum->getCovar().getShape() = " << spectrum->getCovar().getShape() << endl;
    #endif
    covar[ ndarray::view( yLow, yHigh + 1 )( )( iFiber ) ] = spectrum->getCovar()[ ndarray::view() ];
  }
  return covar;
}


//template class pfsDRPStella::SpectrumSet<float, int, float, float>;
//template class pfsDRPStella::SpectrumSet<double, int, double, double>;
//template class pfsDRPStella::SpectrumSet<float, unsigned int, float, float>;
//template class pfsDRPStella::SpectrumSet<double, unsigned int, float, float>;
template class pfsDRPStella::SpectrumSet<float, unsigned short, float, float>;
template class pfsDRPStella::SpectrumSet<double, unsigned short, float, float>;
//template class pfsDRPStella::SpectrumSet<float, unsigned int, float, double>;
//template class pfsDRPStella::SpectrumSet<double, unsigned int, float, double>;
template class pfsDRPStella::SpectrumSet<float, unsigned short, float, double>;
template class pfsDRPStella::SpectrumSet<double, unsigned short, float, double>;
//template class pfsDRPStella::SpectrumSet<float, unsigned int, double, double>;
//template class pfsDRPStella::SpectrumSet<double, unsigned int, double, double>;
//template class pfsDRPStella::SpectrumSet<float, unsigned short, double, double>;
//template class pfsDRPStella::SpectrumSet<double, unsigned short, double, double>;

namespace pfs { namespace drp { namespace stella { namespace math {

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

    template< typename T, int I >
    ndarray::Array< T, 2, 1 > createLineList( ndarray::Array< T, 1, I > const& wLen,
                                              ndarray::Array< T, 1, I > const& linesWLen ){
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
        cout << "Spectra::createLineList: intT = " << indT.getShape() << ": " << indT << endl;
        cout << "Spectra::createLineList: wavelengths = " << wavelengths.getShape() << ": " << wavelengths << endl;
        cout << "Spectra::createLineList: linesWLen = " << linesWLen.getShape() << ": " << linesWLen << endl;
        cout << "Spectra::createLineList: args = " << args.size() << endl;
      #endif
      ndarray::Array< T, 1, 1 > linesPix = pfsDRPStella::math::interPol( indT, wavelengths, linesWLen, args );
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
    template ndarray::Array< double, 2, 1 > createLineList( ndarray::Array< double, 1, 0 > const&, ndarray::Array< double, 1, 0 > const& );
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

//template<> template<> bool pfsDRPStella::Spectrum<float, unsigned short, float, float>::identify( ndarray::Array< float, 2, 1 > const&,
//                                                                                                  DispCorControl const&,
//                                                                                                  size_t );
//template<> template<> bool pfsDRPStella::Spectrum<double, unsigned short, float, float>::identify( ndarray::Array< float, 2, 1 > const&,
//                                                                                                   DispCorControl const&,
//                                                                                                   size_t );
//template<> template<> bool pfsDRPStella::Spectrum<float, unsigned short, float, float>::identify( ndarray::Array< double, 2, 1 > const&,
//                                                                                                  DispCorControl const&,
//                                                                                                  size_t );
//template<> template<> bool pfsDRPStella::Spectrum<double, unsigned short, float, float>::identify( ndarray::Array< double, 2, 1 > const&,
//                                                                                                   DispCorControl const&,
//                                                                                                   size_t );
