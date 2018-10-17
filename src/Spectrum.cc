#include <string>
#include <sstream>
#include <algorithm>
#include "lsst/log/Log.h"

#include "pfs/drp/stella/Spectrum.h"
//#include "pfs/drp/stella/math/Math.h"
//#include "pfs/drp/stella/math/CurveFitting.h"
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
    utils::checkSize(mask.getDimensions(), lsst::geom::Extent2I(_length, 1), "Spectrum::setMask");
    _mask = Mask(mask, true);
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

    auto guess = ndarray::Array<float, 1, 1>(nTerms);
    ndarray::Array<float, 2, 1> limits = ndarray::allocate(nTerms, 2);
    ndarray::Array<float, 1, 1>  gaussCoeffs = ndarray::allocate(nTerms);
    ndarray::Array<float, 1, 1> gaussCoeffsErr = ndarray::allocate(nTerms);
    ndarray::Array<int, 2, 1> limited = ndarray::allocate(nTerms, 2);
    limited.deep() = 1;

    // fit line positions
    std::size_t const nLine = lineList.size();
    ndarray::Array<float, 1, 1> gaussPos = ndarray::allocate(nLine);
    gaussPos.deep() = 0.;

    _referenceLines.reserve(nLine);
    for (std::size_t i = 0; i < nLine; ++i) {
        auto refLine = std::make_shared<ReferenceLine>(*lineList[i]);
        _referenceLines.push_back(refLine);
    }

    for (std::size_t i = 0; i < nLine; ++i) {
        auto refLine = _referenceLines[i];
        if (!std::isfinite(refLine->guessedPosition)) {
            continue;
        }

        std::size_t start = std::size_t(refLine->guessedPosition) - dispCorControl.searchRadius;
        std::size_t end = start + 2*dispCorControl.searchRadius;
        if (start < 0 || start >= _length || end < 0 || end >= _length) {
            continue;
        }

        auto itMaxElement = std::max_element(_spectrum.begin() + start, _spectrum.begin() + end + 1);
        const float maxPos = std::distance(_spectrum.begin(), itMaxElement);
        start = std::round(maxPos - (1.5*dispCorControl.fwhm));
        if (start < 0) start = 0;

        end = std::round(maxPos + 1.5*dispCorControl.fwhm);
        if (end >= _length) end = _length - 1;
        if (end < start + 4) {
            LOGLS_DEBUG(_log, "Line position outside spectrum");
            continue;
        }
        const int n = end - start + 1;
        auto gaussSpec = ndarray::Array<float, 1, 1>(n);
        auto measureErrors = ndarray::Array<float, 1, 1>(n);

        bool allFinite = true;
        auto itSpec = _spectrum.begin() + start;
        for (auto itGaussSpec = gaussSpec.begin(); itGaussSpec != gaussSpec.end(); ++itGaussSpec, ++itSpec) {
            if (!::isfinite(*itSpec)) {
                allFinite = false;
            }
            *itGaussSpec = *itSpec;
        }
        for (auto itMeasErr = measureErrors.begin(), itGaussSpec = gaussSpec.begin();
            itMeasErr != measureErrors.end(); ++itMeasErr, ++itGaussSpec) {
            *itMeasErr = ::sqrt(std::fabs(*itGaussSpec));
            if (!::isfinite(*itMeasErr)) {
                allFinite = false;
            }
            if (*itMeasErr < 0.00001) *itMeasErr = 1.;
        }
        if (!allFinite) {
            LOGLS_WARN(_log, "Fibre " << getFiberId() << " line " << i << " (between " << start << " and " <<
                        end << " has NANs: skipping");
            continue;
        }
        auto array = ndarray::Array<float, 1, 1>(n);
        {
            float x = start;
            for (auto itX = array.begin(); itX != array.end(); ++itX ) {
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

        guess[BASELINE] = *min_element(gaussSpec.begin(), gaussSpec.end());
        guess[PEAK] = *max_element(gaussSpec.begin(), gaussSpec.end()) - guess[BASELINE];
        guess[XC] = 0.5*(array[0] + array[array.size() - 1]);
        guess[WIDTH] = dispCorControl.fwhm;
        LOGLS_DEBUG(_log, "Guess = " << guess);

        limits[PEAK][0] = 0.0;
        limits[PEAK][1] = std::fabs(1.5*guess[PEAK]);
        limits[XC][0] = array[1];
        limits[XC][1] = array[array.size() - 2];
        limits[WIDTH][0] = 0.33*guess[WIDTH];
        limits[WIDTH][1] = 2.0*guess[WIDTH];
        limits[BASELINE][0] = 0.0;
        limits[BASELINE][1] = std::fabs(1.5*guess[BASELINE]) + 1;

        // fitter fails if initial guess not within limits
        if (guess[BASELINE] < 0.0) {
            limits[BASELINE][0] = guess[BASELINE];
        }

        LOGLS_DEBUG(_log, "Limits = " << limits);

        if (!MPFitGaussLim(array,
                           gaussSpec,
                           measureErrors,
                           guess,
                           limited,
                           limits,
                           true,
                           false,
                           gaussCoeffs,
                           gaussCoeffsErr,
                           true)) {
            if ((_mask((start + end)/2, 0) & NO_DATA) == 0) {
                LOGLS_WARN(_log, "GaussFit returned FALSE for fibre " + to_string(getFiberId()) +
                           ": xc = " + to_string(guess[XC]) +
                           " lambda = " + to_string(refLine->wavelength) + " i = " + to_string(i));
            }
            gaussCoeffs.deep() = 0.0;
            gaussCoeffsErr.deep() = 0.0;
            continue;
        }
        if (std::fabs(maxPos - gaussCoeffs[XC]) < dispCorControl.maxDistance) {
            gaussPos[i] = gaussCoeffs[XC];
            refLine->status |= ReferenceLine::FIT;
            refLine->fitIntensity = gaussCoeffs[PEAK];
            refLine->fitPosition = gaussCoeffs[XC];
            refLine->fitPositionErr = ::sqrt(gaussCoeffsErr[XC]);

            if (i > 0 && std::fabs(gaussPos[i] - gaussPos[i - 1]) < 1.5) { // wrong line identified!
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
                        LOGLS_DEBUG(_log, "Fibre " << getFiberId() <<
                                   " i=" << i << ": line " <<
                                   " at  GaussPos[" << badIndx << "] = " <<
                                   gaussPos[badIndx] <<
                                   " has probably been misidentified " <<
                                   "(GaussPos[" << goodIndx << "] =" <<
                                   gaussPos[goodIndx] <<
                                   ") => removing line from line list");
                    }

                    gaussPos[badIndx] = 0.0;
                    _referenceLines[badIndx]->status |= ReferenceLine::MISIDENTIFIED;
                }
            }
            continue;
        }
        std::ostringstream msg;

        msg << "fiberId = " << getFiberId() <<
            " |(maxPos=" << maxPos << ") - (GaussCoeffs[XC]=" <<
            gaussCoeffs[XC] << ")| = " << std::fabs(maxPos - gaussCoeffs[XC]) << " >= " <<
            dispCorControl.maxDistance << " => Skipping line";

        if (refLine->status & ReferenceLine::INTERPOLATED) {
            LOGLS_TRACE(_log, msg.str());
        } else {
            LOGLS_WARN(_log, msg.str());
        }
    }
}

}}} // namespace pfs::drp::stella
