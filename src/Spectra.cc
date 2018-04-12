#include <iostream>
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
Spectrum::Spectrum(size_t length, size_t fiberId )
  : _length(length),
    _mask(length, 1),
    _fiberId(fiberId),
    _isWavelengthSet(false)
{
  _spectrum = ndarray::allocate( length );
  _spectrum.deep() = 0.;
  _background = ndarray::allocate( length );
  _background.deep() = 0.;
  _covar = ndarray::allocate(3, length);
  _covar.deep() = 0.;
  _wavelength = ndarray::allocate( length );
  _wavelength.deep() = 0.;

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
  _isWavelengthSet = true;
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

void
Spectrum::setBackground(ndarray::Array<Spectrum::ImageT, 1, 1> const& background)
{
  /// Check length of input variance
  if (static_cast<std::size_t>(background.getShape()[0]) != _length) {
    string message("pfs::drp::stella::Spectrum::setBackground: ERROR: background->size()=");
    message += to_string( background.getShape()[ 0 ] ) + string( " != _length=" ) + to_string( _length );
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
  }
  _background.deep() = background;
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
    if (static_cast<std::size_t>(covar.getShape()[1]) != _length) {
      string message("pfs::drp::stella::Spectrum::setCovar: ERROR: covar->size()=");
      message += to_string( covar.getShape()[1] ) + string( " != _length=" ) + to_string( _length );
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
    }
    if (covar.getShape()[0] != 3) {
      string message("pfs::drp::stella::Spectrum::setCovar: ERROR: covar->size()=");
      message += to_string( covar.getShape()[0] ) + string( " != 3" );
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

void
Spectrum::identify(std::vector<std::shared_ptr<const ReferenceLine>> const& lineList, ///< List of arc lines
                   DispCorControl const& dispCorControl,
                   int nLinesCheck)
{
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.Spectra.identify");

    const auto NO_DATA = _mask.getPlaneBitMask("NO_DATA");
    const auto CR = _mask.getPlaneBitMask("CR");
    const auto INTRP = _mask.getPlaneBitMask("INTRP");
    const auto SAT = _mask.getPlaneBitMask("SAT");
    
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

    // fit line positions
    const int nLine = lineList.size();
    ndarray::Array< float, 1, 1 > GaussPos = ndarray::allocate(nLine);
    GaussPos.deep() = 0.;

    _referenceLines.reserve(nLine);
    for (int i = 0; i < nLine; ++i) {
        auto refLine = std::make_shared<ReferenceLine>(*lineList[i]);
        _referenceLines.push_back(refLine);
    }

    for (int i = 0; i < nLine; ++i) {
        auto refLine = _referenceLines[i];

        int start = int(refLine->guessedPixelPos) - dispCorControl.searchRadius;
        int end   = start + 2*dispCorControl.searchRadius;
        if (start < 0 || end >= _length) {
            continue;
        }

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
            auto X = ndarray::Array<float, 1, 1>(n);
            {
                float x = start;
                for (auto itX = X.begin(); itX != X.end(); ++itX ) {
                    *itX = x++;
                }
            }

            for (int j = start; j <= end; ++j) {
                const auto mval = _mask(j, 0);
                if ((mval & CR) != 0) {
                    refLine->status |= ReferenceLine::CR;
                }
                if ((mval & INTRP) != 0) {
                    refLine->status |= ReferenceLine::INTERPOLATED;
                }
                if ((mval & SAT) != 0) {
                    refLine->status |= ReferenceLine::SATURATED;
                }
            }

            Guess[BASELINE] = *min_element(GaussSpec.begin(), GaussSpec.end());
            Guess[PEAK]     = *max_element(GaussSpec.begin(), GaussSpec.end()) - Guess[BASELINE];
            Guess[XC]       = 0.5*(X[0] + X[X.size() - 1]);
            Guess[WIDTH]    = dispCorControl.fwhm;
            LOGLS_DEBUG(_log, "Guess = " << Guess);

            Limits[PEAK    ][0] = 0.0;
            Limits[PEAK    ][1] = std::fabs(1.5*Guess[PEAK]);
            Limits[XC      ][0] = X[1];
            Limits[XC      ][1] = X[X.size() - 2];
            Limits[WIDTH   ][0] = 0.33*Guess[WIDTH];
            Limits[WIDTH   ][1] = 2.0*Guess[WIDTH];
            Limits[BASELINE][0] = 0.0;
            Limits[BASELINE][1] = std::fabs(1.5*Guess[BASELINE]) + 1;

            // fitter fails if initial guess not within limits
            if (Guess[BASELINE] < 0.0) {
                Limits[BASELINE][0] = Guess[BASELINE];
            }

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
                if ((_mask((start + end)/2, 0) & NO_DATA) == 0) {
                    LOGLS_WARN(_log, "GaussFit returned FALSE for fibre " + to_string(getFiberId()) +
                               ": xc = " + to_string(Guess[XC]) +
                               " lambda = " + to_string(refLine->wavelength) + " i = " + to_string(i));
                }
                GaussCoeffs.deep() = 0.0;
                EGaussCoeffs.deep() = 0.0;
            } else {
                if (std::fabs(maxPos - GaussCoeffs[XC] ) < dispCorControl.maxDistance) {
                    GaussPos[i] = GaussCoeffs[XC];
                    refLine->status |= ReferenceLine::FIT;
                    refLine->fitIntensity = GaussCoeffs[PEAK];
                    refLine->fitPixelPos = GaussCoeffs[XC];
                    refLine->fitPixelPosErr = ::sqrt(EGaussCoeffs[XC]);

                    if (i > 0 && std::fabs(GaussPos[i] - GaussPos[i - 1]) < 1.5) { // wrong line identified!
                        if (nLine > 2) {
                            int badIndx, goodIndx;
                            if (_referenceLines[i]->guessedPixelPos < _referenceLines[i - 1]->guessedPixelPos) {
                                badIndx = i;
                                goodIndx = i - 1;
                            } else {
                                badIndx = i - 1;
                                goodIndx = i;
                            }
                            if ((_mask((start + end)/2, 0) & (SAT | NO_DATA)) == 0) {
                                LOGLS_WARN(_log, "Fibre " << getFiberId() << 
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
                    std::ostringstream msg;

                    msg << "fiberId = " << getFiberId() <<
                        " |(maxPos=" << maxPos << ") - (GaussCoeffs[XC]=" <<
                        GaussCoeffs[XC] << ")| = " << std::fabs(maxPos - GaussCoeffs[XC]) << " >= " <<
                        dispCorControl.maxDistance << " => Skipping line";
                    
                    if (refLine->status & ReferenceLine::INTERPOLATED) {
                        LOGLS_TRACE(_log, msg.str());
                    } else {
                        LOGLS_WARN(_log, msg.str());
                    }
                }
            }
        }
    }
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
                
}}}
