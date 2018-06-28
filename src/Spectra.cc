#include <sstream>
#include "lsst/log/Log.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/image/MaskedImage.h"

#include "pfs/drp/stella/Spectra.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/math/CurveFitting.h"
#include "pfs/drp/stella/cmpfit-1.2/MPFitting_ndarray.h"
#include "pfs/drp/stella/utils/checkSize.h"

namespace pfs { namespace drp { namespace stella {

Spectrum::Spectrum(std::size_t length, std::size_t fiberId)
  : _length(length),
    _spectrum(ndarray::allocate(length)),
    _mask(length, 1),
    _background(ndarray::allocate(length)),
    _covariance(ndarray::allocate(3, length)),
    _wavelength(ndarray::allocate(length)),
    _fiberId(fiberId),
    _isWavelengthSet(false)
{
    _spectrum.deep() = 0.;
    _background.deep() = 0.;
    _covariance.deep() = 0.;
    _wavelength.deep() = 0.;
    _mask.addMaskPlane("REJECTED_LINES");
    _mask.addMaskPlane("FIBERTRACE");
}


Spectrum::Spectrum(
    ImageArray const& spectrum,
    Mask const& mask,
    ImageArray const& background,
    CovarianceMatrix const& covariance,
    ImageArray const& wavelength,
    ReferenceLineList const& lines,
    std::size_t fiberId
) : _length(spectrum.getNumElements()),
    _spectrum(spectrum),
    _mask(mask),
    _background(background),
    _covariance(covariance),
    _wavelength(wavelength),
    _fiberId(fiberId),
    _referenceLines(lines),
    _isWavelengthSet(!wavelength.empty())
{}


void Spectrum::setSpectrum(Spectrum::ImageArray const& spectrum) {
    utils::checkSize(spectrum.getShape()[0], _length, "Spectrum::setSpectrum");
    _spectrum.deep() = spectrum;
}


void Spectrum::setMask(const Mask & mask) {
    utils::checkSize(mask.getDimensions(), lsst::afw::geom::Extent2I(_length, 1), "Spectrum::setMask");
    _mask = mask;
}


void Spectrum::setBackground(Spectrum::ImageArray const& background) {
    utils::checkSize(background.getShape()[0], _length, "Spectrum::setBackground");
    _background.deep() = background;
}


void Spectrum::setVariance(Spectrum::VarianceArray const& variance) {
    utils::checkSize(variance.getShape()[0], _length, "Spectrum::setVariance");
    _covariance[0].deep() = variance;
}


void Spectrum::setCovariance(Spectrum::CovarianceMatrix const& covariance) {
    utils::checkSize(covariance.getShape(), ndarray::makeVector(static_cast<std::size_t>(3), _length),
              "Spectrum::setCovariance");
    _covariance.deep() = covariance;
}


void Spectrum::setWavelength(Spectrum::ImageArray const& wavelength) {
    utils::checkSize(wavelength.getShape()[0], _length, "Spectrum::setWavelength");
    _wavelength.deep() = wavelength;
    _isWavelengthSet = true;
}



SpectrumSet::SpectrumSet(std::size_t nSpectra, std::size_t length)
  : _length(length)
{
    _spectra.reserve(nSpectra);
    for (std::size_t i = 0; i < nSpectra; ++i) {
        _spectra.push_back(std::make_shared<Spectrum>(length, i));
    }
}


SpectrumSet::SpectrumSet(SpectrumSet::Collection const& spectra)
  : _length(spectra.size() > 0 ? spectra[0]->getNumPixels() : 0)
{
    if (spectra.size() == 0) {
        throw LSST_EXCEPT(pexExcept::LengthError, "Empty vector supplied; can't determine length of spectra");
    }
    _spectra.reserve(spectra.size());
    for (auto const& ss : spectra) {
        utils::checkSize(ss->getNumPixels(), _length, "SpectrumSet constructor");
        _spectra.push_back(ss);
    }
}


SpectrumSet::ImageArray SpectrumSet::getAllFluxes() const {
    ImageArray flux = ndarray::allocate(size(), _length);
    for (std::size_t ii = 0; ii < size(); ++ii) {
        flux[ii] = get(ii)->getSpectrum();
    }
    return flux;
}


SpectrumSet::ImageArray SpectrumSet::getAllWavelengths() const {
    ImageArray lambda = ndarray::allocate(size(), _length);
    for (std::size_t ii = 0; ii < size(); ++ii) {
        lambda[ii] = get(ii)->getWavelength();
    }
    return lambda;
}


SpectrumSet::MaskArray SpectrumSet::getAllMasks() const {
    MaskArray mask = ndarray::allocate(size(), _length);
    for (std::size_t ii = 0; ii < size(); ++ii) {
        mask[ii] = get(ii)->getMask().getArray()[0];
    }
    return mask;
}


SpectrumSet::CovarianceArray SpectrumSet::getAllCovariances() const {
    CovarianceArray covar = ndarray::allocate(size(), 3, _length);
    for (std::size_t ii = 0; ii < size(); ++ii ){
        covar[ii] = get(ii)->getCovariance();
    }
    return covar;
}


SpectrumSet::ImageArray SpectrumSet::getAllBackgrounds() const {
    ImageArray backgrounds = ndarray::allocate(size(), _length);
    for (std::size_t ii = 0; ii < size(); ++ii) {
        backgrounds[ii] = get(ii)->getBackground();
    }
    return backgrounds;
}


void SpectrumSet::set(std::size_t ii, SpectrumSet::SpectrumPtr spectrum) {
    utils::checkSize(spectrum->getNumPixels(), _length, "SpectrumSet::set");
    _spectra[ii] = spectrum;
}


void SpectrumSet::add(SpectrumSet::SpectrumPtr spectrum) {
    utils::checkSize(spectrum->getNumPixels(), _length, "SpectrumSet::add");
    _spectra.push_back(spectrum);
}


void Spectrum::identify(
    std::vector<std::shared_ptr<const ReferenceLine>> const& lineList,
    DispersionCorrectionControl const& dispCorControl,
    int nLinesCheck
) {
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.Spectrum.identify");

    auto const NO_DATA = _mask.getPlaneBitMask("NO_DATA");
    auto const CR = _mask.getPlaneBitMask("CR");
    auto const INTRP = _mask.getPlaneBitMask("INTRP");
    auto const SAT = _mask.getPlaneBitMask("SAT");
    
    // for each line in line list, find maximum in spectrum and fit Gaussian
    int const PEAK = 0;           // peak y value
    int const XC = 1;             // x centroid position
    int const WIDTH = 2;          // gaussian sigma width
    int const BASELINE = 3;       // constant offset
    int const nTerms = 4;         // number of terms in fit (i.e. BASELINE + 1)

    auto Guess = ndarray::Array<float, 1, 1>( nTerms );
    ndarray::Array<float, 2, 1> Limits = ndarray::allocate(nTerms, 2);
    ndarray::Array<float, 1, 1>  GaussCoeffs = ndarray::allocate(nTerms);
    ndarray::Array<float, 1, 1> EGaussCoeffs = ndarray::allocate(nTerms);
    ndarray::Array<int, 2, 1> Limited = ndarray::allocate(nTerms, 2);
    Limited.deep() = 1;

    // fit line positions
    std::size_t const nLine = lineList.size();
    ndarray::Array<float, 1, 1> GaussPos = ndarray::allocate(nLine);
    GaussPos.deep() = 0.;

    _referenceLines.reserve(nLine);
    for (std::size_t i = 0; i < nLine; ++i) {
        auto refLine = std::make_shared<ReferenceLine>(*lineList[i]);
        _referenceLines.push_back(refLine);
    }

    for (std::size_t i = 0; i < nLine; ++i) {
        auto refLine = _referenceLines[i];

        std::size_t start = std::size_t(refLine->guessedPosition) - dispCorControl.searchRadius;
        std::size_t end = start + 2*dispCorControl.searchRadius;
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
            continue;
        }
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

        for (std::size_t j = start; j <= end; ++j) {
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
            continue;
        }
        if (std::fabs(maxPos - GaussCoeffs[XC] ) < dispCorControl.maxDistance) {
            GaussPos[i] = GaussCoeffs[XC];
            refLine->status |= ReferenceLine::FIT;
            refLine->fitIntensity = GaussCoeffs[PEAK];
            refLine->fitPosition = GaussCoeffs[XC];
            refLine->fitPositionErr = ::sqrt(EGaussCoeffs[XC]);

            if (i > 0 && std::fabs(GaussPos[i] - GaussPos[i - 1]) < 1.5) { // wrong line identified!
                if (nLine > 2) {
                    int badIndx, goodIndx;
                    if (_referenceLines[i]->guessedPosition <
                        _referenceLines[i - 1]->guessedPosition) {
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
            continue;
        }
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

                
}}}
