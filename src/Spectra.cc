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
Spectrum::Spectrum(size_t length, size_t iTrace )
  : _length(length),
    _mask(length, 1),
    _iTrace(iTrace),
    _isWavelengthSet(false)
{
  _spectrum = ndarray::allocate( length );
  _spectrum.deep() = 0.;
  _covar = ndarray::allocate(3, length);
  _covar.deep() = 0.;
  _wavelength = ndarray::allocate( length );
  _wavelength.deep() = 0.;
  _dispCoeffs = ndarray::allocate( 0 ); // The number of elements comes from DispCorControl
  _dispRms = 0.;
  _dispRmsCheck = 0.;
  _nGoodLines = 0;

  _mask.addMaskPlane("REJECTED_LINES");
  _mask.addMaskPlane("FIBERTRACE");
}

void Spectrum::setWavelength( ndarray::Array<float, 1, 1> const& wavelength )
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
SpectrumSet::SpectrumSet( size_t nSpectra, size_t length )
        : _spectra( new std::vector< PTR(Spectrum) >() )
{
  for (size_t i = 0; i < nSpectra; ++i){
    PTR(Spectrum) spec( new Spectrum( length, i ) );
    _spectra->push_back(spec);
  }
}

ndarray::Array< float, 2, 1 > SpectrumSet::getAllFluxes() const{
  int nFibers = int( getNtrace() );
  int nCCDRows = getSpectrum(0)->getNpix();
  ndarray::Array<float, 2, 1> flux = ndarray::allocate(nFibers, nCCDRows);

  flux.deep() = 0.;

  for ( int iFiber = 0; iFiber < getNtrace(); ++iFiber ){
      flux[iFiber] = (*_spectra)[iFiber]->getSpectrum();
  }
  return flux;
}

ndarray::Array< float, 2, 1 > SpectrumSet::getAllWavelengths() const{
  int nFibers = int( getNtrace() );
  int nCCDRows = getSpectrum( 0 )->getNpix();
  /// allocate memory for the array
  ndarray::Array< float, 2, 1 > lambda = ndarray::allocate( nFibers, nCCDRows );

  lambda.deep() = 0.;

  for ( int iFiber = 0; iFiber < getNtrace(); ++iFiber ){
      lambda[iFiber] = (*_spectra)[iFiber]->getWavelength();
  }
  return lambda;
}

ndarray::Array< int, 2, 1 > SpectrumSet::getAllMasks() const{
  int nFibers = int( getNtrace() );
  int nCCDRows = getSpectrum( 0 )->getNpix();
  /// allocate memory for the array
  ndarray::Array< int, 2, 1 > mask = ndarray::allocate( nFibers, nCCDRows );

  mask.deep() = 0.;

  for ( int iFiber = 0; iFiber < getNtrace(); ++iFiber) {
      mask[iFiber] = (*_spectra)[iFiber]->getMask().getArray()[0];
  }
  return mask;
}

ndarray::Array< float, 3, 1 > SpectrumSet::getAllCovars() const{
  int nFibers = int( getNtrace() );
  int nCCDRows = getSpectrum( 0 )->getNpix();
  /// allocate memory for the array
  ndarray::Array< float, 3, 1 > covar = ndarray::allocate(nFibers, 3, nCCDRows);

  covar.deep() = 0.;

  for ( int iFiber = 0; iFiber < getNtrace(); ++iFiber ){
      covar[iFiber] = (*_spectra)[iFiber]->getCovar();
  }
  return covar;
}

void
Spectrum::setSpectrum( ndarray::Array<Spectrum::ImageT, 1, 1> const& spectrum )
{
  /// Check length of input spectrum
  if (static_cast<std::size_t>(spectrum.getShape()[0]) != _length) {
    string message("pfs::drp::stella::Spectrum::setSpectrum: ERROR: spectrum->size()=");
    message += to_string(spectrum.getShape()[0]) + string(" != _length=") + to_string(_length);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _spectrum.deep() = spectrum;
}

ndarray::Array<Spectrum::VarianceT, 1, 1>
Spectrum::getVariance() const
{
    ndarray::Array< VarianceT, 1, 1 > variance = ndarray::allocate(getNpix());
    variance.deep() = _covar[0];
    return variance;
}

ndarray::Array<Spectrum::VarianceT, 1, 1>
Spectrum::getVariance()
{
    return _covar[0];
}

void
Spectrum::setVariance( ndarray::Array<VarianceT, 1, 1> const& variance )
{
  /// Check length of input variance
  if (static_cast<std::size_t>(variance.getShape()[0]) != _length) {
    string message("pfs::drp::stella::Spectrum::setVariance: ERROR: variance->size()=");
    message += to_string( variance.getShape()[ 0 ] ) + string( " != _length=" ) + to_string( _length );
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _covar[0].deep() = variance;
}

void
Spectrum::setCovar(const ndarray::Array<VarianceT, 2, 1> & covar )
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

void
Spectrum::setMask(const Mask & mask)
{
  /// Check length of input mask
  if (static_cast<std::size_t>(mask.getWidth()) != _length){
    string message("pfs::drp::stella::Spectrum::setMask: ERROR: mask.getWidth()=");
    message += to_string(mask.getWidth()) + string(" != _length=") + to_string(_length);
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _mask = mask;
}

ndarray::Array<float, 1, 1 >
Spectrum::hIdentify(ndarray::Array< float, 1, 1 > const& lineListLambda,
                    ndarray::Array< float, 1, 1 > const& lineListPixel,
                    DispCorControl const& dispCorControl
                   )
{
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.Spectra.identify");

    const auto nodata = _mask.getPlaneBitMask("NO_DATA");
    
    ///for each line in line list, find maximum in spectrum and fit Gaussian
    const int PEAK = 0;           // peak y value
    const int XC = 1;             // x centroid position
    const int WIDTH = 2;          // gaussian sigma width
    const int BASELINE = 3;       // constant offset
    const int nTerms = 4;         // number of terms in fit (i.e. BASELINE + 1)

    auto Guess = ndarray::Array< float, 1, 1 >( nTerms );
    ndarray::Array< float, 2, 1 > Limits = ndarray::allocate(nTerms, 2);
    ndarray::Array<float, 1, 1 >  GaussCoeffs = ndarray::allocate(nTerms);
    ndarray::Array< float, 1, 1 > EGaussCoeffs = ndarray::allocate(nTerms);
    ndarray::Array< int, 2, 1 > Limited = ndarray::allocate(nTerms, 2);
    Limited.deep() = 1;
    ndarray::Array< float, 1, 1 > Ind = math::indGenNdArr( float( _length ) );

    // Returned line positions
    const int nLine = lineListPixel.size();
    ndarray::Array< float, 1, 1 > GaussPos = ndarray::allocate(nLine);
    GaussPos.deep() = 0.;

    _referenceLines.reserve(nLine);

    for (int i = 0; i < nLine; ++i) {
        int start = int(lineListPixel[i]) - dispCorControl.searchRadius;
        int end   = start + 2*dispCorControl.searchRadius;
        if (start < 0 || end >= _length) {
            continue;
        }

        auto refLine = std::make_shared<ReferenceLine>(ReferenceLine::NOWT, lineListLambda[i], lineListPixel[i]);
        _referenceLines.push_back(refLine);

        auto itMaxElement = std::max_element( _spectrum.begin() + start, _spectrum.begin() + end + 1 );
        const float maxPos = std::distance(_spectrum.begin(), itMaxElement);
        start = std::round( maxPos - ( 1.5 * dispCorControl.fwhm ) );
        if (start < 0) start = 0;

        end = std::round(maxPos + 1.5*dispCorControl.fwhm);
        if ( end >= _length ) end = _length - 1;
        if ( end < start + 4) {
            LOGLS_WARN(_log, "WARNING: Line position outside spectrum");
        } else {
            const int n = end - start + 1;
            auto X             = ndarray::Array<float, 1, 1>(n);
            auto GaussSpec     = ndarray::Array<float, 1, 1>(n);
            auto MeasureErrors = ndarray::Array<float, 1, 1>(n);

            auto itSpec = _spectrum.begin() + start;
            for (auto itGaussSpec = GaussSpec.begin(); itGaussSpec != GaussSpec.end(); ++itGaussSpec, ++itSpec )
                *itGaussSpec = *itSpec;
            for(auto itMeasErr = MeasureErrors.begin(), itGaussSpec = GaussSpec.begin();
                itMeasErr != MeasureErrors.end(); ++itMeasErr, ++itGaussSpec) {
                *itMeasErr = sqrt(std::fabs(*itGaussSpec));
                if (*itMeasErr < 0.00001) *itMeasErr = 1.;
            }
            auto itInd = Ind.begin() + start;
            for ( auto itX = X.begin(); itX != X.end(); ++itX, ++itInd ) {
                *itX = *itInd;
            }

            Guess[BASELINE] = *min_element(GaussSpec.begin(), GaussSpec.end());
            Guess[PEAK]     = *max_element(GaussSpec.begin(), GaussSpec.end()) - Guess[BASELINE];
            Guess[XC]       = 0.5*(X[0] +  X[X.size() - 1]);
            Guess[WIDTH]    = dispCorControl.fwhm;
            LOGLS_DEBUG(_log, "Guess = " << Guess);

            Limits[ndarray::makeVector(PEAK,     0 )] = 0.;
            Limits[ndarray::makeVector(PEAK,     1 )] = std::fabs(1.5*Guess[PEAK]);
            Limits[ndarray::makeVector(XC,       0 )] = X[1];
            Limits[ndarray::makeVector(XC,       1 )] = X[X.size() - 2];
            Limits[ndarray::makeVector(WIDTH,    0 )] = 0.33*Guess[WIDTH];
            Limits[ndarray::makeVector(WIDTH,    1 )] = 2.0*Guess[WIDTH];
            Limits[ndarray::makeVector(BASELINE, 0 )] = 0.0;
            Limits[ndarray::makeVector(BASELINE, 1 )] = std::fabs(1.5*Guess[BASELINE]) + 1;
            LOGLS_DEBUG(_log, "Limits = " << Limits);

            if (!MPFitGaussLim(X,
                               GaussSpec,
                               MeasureErrors,
                               Guess,
                               Limited,
                               Limits,
                               true,
                               false,
                               GaussCoeffs,
                               EGaussCoeffs,
                               true)) {
                if ((_mask((start + end)/2, 0) & nodata) == 0) {
                    LOGLS_WARN(_log, "GaussFit returned FALSE for fibre " + to_string(getITrace()) +
                               ": xc = " + to_string(Guess[XC]) + " lambda = " + to_string(lineListLambda[i]));
                }
                GaussCoeffs.deep() = 0.0;
                EGaussCoeffs.deep() = 0.0;
            } else {
                if (std::fabs(maxPos - GaussCoeffs[1] ) < dispCorControl.maxDistance) {
                    GaussPos[i] = GaussCoeffs[1];
                    refLine->status |= ReferenceLine::FIT;
                    refLine->fitPixelPos = GaussCoeffs[1];
                    refLine->fitPixelPosErr = EGaussCoeffs[1];

                    if (i > 0 && std::fabs(GaussPos[i] - GaussPos[i - 1]) < 1.5) { // wrong line identified!
                        if (nLine > 2) {
                            int badIndx, goodIndx;
                            if (lineListPixel[i] < lineListPixel[i - 1]) {
                                badIndx = i;
                                goodIndx = i - 1;
                            } else {
                                badIndx = i - 1;
                                goodIndx = i;
                            }
                            if ((_mask((start + end)/2, 0) & nodata) == 0) {
                                LOGLS_WARN(_log, "Fibre " << to_string(getITrace()) << 
                                           " i=" << i << ": line " <<
                                           " at  GaussPos[" << badIndx << "] = " <<
                                           GaussPos[badIndx] <<
                                           " has probably been misidentified " <<
                                           "(GaussPos[" << goodIndx << "] =" <<
                                           GaussPos[goodIndx] <<
                                           ") => removing line from line list");
                            }
                            
                            GaussPos[badIndx] = 0.0;
                            _referenceLines[badIndx]->status |= ReferenceLine::MISIDENTIFIED;
                        }
                    }
                } else {
                    string message("WARNING: maxPos=");
                    message += to_string(maxPos) + " - GaussCoeffs[ 1 ]=" + to_string(GaussCoeffs[1]);
                    message += "(=" + to_string(std::fabs(maxPos - GaussCoeffs[1]));
                    message += ") >= " + to_string(dispCorControl.maxDistance) + " => Skipping line";
                    LOGLS_WARN(_log, message);
                }
            }
        }
    }

    return GaussPos;
}

void
Spectrum::identify(ndarray::Array< float, 2, 1 > const& lineList,
                   DispCorControl const& dispCorControl,
                   std::size_t nLinesCheck )
{
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.Spectra.identify");

    ///for each line in line list, find maximum in spectrum and fit Gaussian
    auto const lineListLambda = lineList[0];
    auto const lineListPixel = lineList[1];
    auto GaussPos = hIdentify(lineListLambda, lineListPixel, dispCorControl);

    ///remove lines which could not be found from line list
    std::vector<int> Index(GaussPos.getShape()[0], 0 );
    std::size_t pos = 0;
    for (auto it = GaussPos.begin(); it != GaussPos.end(); ++it, ++pos ){
        if (*it > 0.0) {
            Index[pos] = 1;
        }
    }
    std::vector<std::size_t> indices = math::getIndices(Index);
    std::size_t nInd = std::accumulate(Index.begin(), Index.end(), 0);
    if (nInd == 0) {
        std::string message("identify: No lines identified");
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    /// separate lines to fit and lines for RMS test
    std::vector< std::size_t > indCheck;
    for ( std::size_t i = 0; i < nLinesCheck; ++i ){
      srand( 0 ); //seed initialization
      int randNum = rand() % ( indices.size() - 2 ) + 1; // Generate a random number between 0 and 1
      indCheck.push_back( std::size_t( randNum ) );
      indices.erase( indices.begin() + randNum );
    }

    _nGoodLines = nInd;
    const int nLine = lineListPixel.size();
    const long nLinesIdentifiedMin(std::lround(float(nLine)*dispCorControl.minPercentageOfLines/100.0));
    if ( _nGoodLines < nLinesIdentifiedMin ){
      std::string message("identify: ERROR: less than ");
      message += std::to_string(nLinesIdentifiedMin) + " lines identified";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    ndarray::Array< std::size_t, 1, 1 > IndexPos = ndarray::external( indices.data(), ndarray::makeVector( int( indices.size() ) ), ndarray::makeVector( 1 ) );
    ndarray::Array< float, 1, 1 > FittedPos = math::getSubArray( GaussPos, 
                                                                       IndexPos );
    ndarray::Array< std::size_t, 1, 1 > IndexCheckPos = ndarray::external( indCheck.data(), ndarray::makeVector( int( indCheck.size() ) ), ndarray::makeVector( 1 ) );
    ndarray::Array< float, 1, 1 > FittedCheckPos = math::getSubArray(GaussPos, IndexCheckPos);
    LOGLS_DEBUG(_log, "FittedPos = " << FittedPos << endl);

    ndarray::Array< float, 1, 1 > WLen = ndarray::allocate(nLine); // do we need a copy here?
    WLen.deep() = lineListLambda;
    auto FittedWLen = math::getSubArray(WLen, IndexPos);
    LOGLS_DEBUG(_log, "found FittedWLen = " << FittedWLen);

    auto FittedWLenCheck = math::getSubArray( WLen, IndexCheckPos );

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
    _dispCoeffs.deep() = math::PolyFit( FittedPos,
                                        FittedWLen,
                                        dispCorControl.order,
                                        float(0. - dispCorControl.sigmaReject),
                                        float(dispCorControl.sigmaReject),
                                        dispCorControl.nIterReject,
                                        S_A1_Args,
                                        PP_Args);
    LOGLS_DEBUG(_log, "_dispCoeffs = " << _dispCoeffs);
    
    /// Remove lines rejected by PolyFit from FittedPos and FittedWLen
    auto maskVal = _mask.getPlaneBitMask("REJECTED_LINES");
    for (int i = 0; i < rejected->size(); ++i){
        LOGLS_DEBUG(_log, "rejected FittedPos[" << (*rejected)[i] << "] = " << FittedPos[(*rejected)[i]]);
        for (int p = (FittedPos[(*rejected)[i]]-2 < 0 ? 0 : FittedPos[(*rejected)[i]]-2);
                 p < (FittedPos[(*rejected)[i]]+2 >= _length ? _length-1 : FittedPos[(*rejected)[i]]+2); ++p){
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

    ndarray::Array<float, 1, 1> fittedPosNotRejected = math::getSubArray(FittedPos, notRejectedArr);
    LOGLS_DEBUG(_log, "fittedPosNotRejected = " << _nGoodLines << ": " << fittedPosNotRejected);
    
    ndarray::Array<float, 1, 1> fittedWLenNotRejected = math::getSubArray(FittedWLen, notRejectedArr);
    LOGLS_DEBUG(_log, "fittedWLenNotRejected = " << _nGoodLines << ": " << fittedWLenNotRejected);
    ndarray::Array< float, 1, 1 > WLen_Gauss =      math::Poly(fittedPosNotRejected, _dispCoeffs,
                                                               xRange[0], xRange[1]);
    ndarray::Array< float, 1, 1 > WLen_GaussCheck = math::Poly(FittedCheckPos, _dispCoeffs,
                                                               xRange[0], xRange[1]);
    LOGLS_DEBUG(_log, "WLen_PolyFit = " << WLen_Gauss);
    LOGLS_DEBUG(_log, "_dispCoeffs = " << _dispCoeffs);

    ///Calculate RMS
    ndarray::Array< float, 1, 1 > WLenMinusFit = ndarray::allocate( WLen_Gauss.getShape()[ 0 ] );
    WLenMinusFit.deep() = fittedWLenNotRejected - WLen_Gauss;
    LOGLS_DEBUG(_log, "WLenMinusFit = " << WLenMinusFit);
    _dispRms = math::calcRMS( WLenMinusFit );
    LOGLS_DEBUG(_log, "_nGoodLines = " << _nGoodLines);
    LOGLS_DEBUG(_log, "_dispRms = " << _dispRms);

    ///Calculate RMS for test lines
    ndarray::Array< float, 1, 1 > WLenMinusFitCheck = ndarray::allocate( WLen_GaussCheck.getShape()[ 0 ] );
    WLenMinusFitCheck.deep() = FittedWLenCheck - WLen_GaussCheck;
    LOGLS_DEBUG(_log, "WLenMinusFitCheck = " << WLenMinusFitCheck);
    _dispRmsCheck = math::calcRMS( WLenMinusFitCheck );
    LOGLS_DEBUG(_log, "dispRmsCheck = " << _dispRmsCheck);

    ///calibrate spectrum
    ndarray::Array< float, 1, 1 > Indices = math::indGenNdArr( float( _length ) );
    _wavelength = ndarray::allocate( _length );
    _wavelength.deep() = math::Poly( Indices, _dispCoeffs, xRange[0], xRange[1] );
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

PTR(const Spectrum)
SpectrumSet::getSpectrum( const std::size_t i ) const
{
    if (i >= _spectra->size()){
        string message("SpectrumSet::getSpectrum(i=");
        message += to_string(i) + "): ERROR: i >= _spectra->size()=" + to_string(_spectra->size());
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return (*_spectra)[i];
}

PTR(Spectrum)
SpectrumSet::getSpectrum( const std::size_t i )
{
    if (i >= _spectra->size()){
        string message("SpectrumSet::getSpectrum(i=");
        message += to_string(i) + "): ERROR: i >= _spectra->size()=" + to_string(_spectra->size());
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return (*_spectra)[i];
}

void
SpectrumSet::setSpectrum(std::size_t const i,
                                                   PTR( Spectrum) spectrum )
{
    if (i > _spectra->size()) {
        string message("SpectrumSet::setSpectrum(i=");
        message += to_string(i) + "): ERROR: i > _spectra->size()=" + to_string(_spectra->size());
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    
    if ( i == _spectra->size() ){
        _spectra->push_back(spectrum);
    } else{
        (*_spectra )[i] = spectrum;
    }
}

namespace math {
    /**
     * Return a 2-d array representing a set of lines
     *   arr[0]:   line positions in nm
     *   arr[1]:   line positions in pixels on the detector
     */
    template< typename T, int I >
    ndarray::Array< T, 2, 1 > createLineList(ndarray::Array<T, 1, I> const& wLen,
                                             ndarray::Array<T, 1, I> const& linesWLen )
    {
        const ndarray::Array<size_t, 1, 1> ind = math::getIndicesInValueRange(wLen, T(1), T(15000));
        if (ind.getShape()[0] <= 0) {
            string message("ind.getShape()[0](=");
            message += to_string(ind.getShape()[0]) + ") <= 0";
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        
        ndarray::Array<T, 1, 1> indT = ndarray::allocate(ind.getShape()[ 0 ]);
        indT.deep() = ind;
        ndarray::Array<T, 1, 1> wavelengths =
            ndarray::copy(wLen[ndarray::view(ind[0], ind[ind.getShape()[0] - 1] + 1)] );
        std::vector<std::string> args(1);
        ndarray::Array< T, 1, 1 > linesPix = math::interPol(indT, wavelengths, linesWLen, args);

        ndarray::Array< T, 2, 1 > out = ndarray::allocate(2, linesWLen.size());
        out[0] = linesWLen;
        out[1] = linesPix;

        return out;
    }

    template ndarray::Array< float, 2, 1 > createLineList( ndarray::Array< float, 1, 0 > const&, ndarray::Array< float, 1, 0 > const& );
    template ndarray::Array< float, 2, 1 > createLineList( ndarray::Array< float, 1, 1 > const&, ndarray::Array< float, 1, 1 > const& );

}
                
}}}
