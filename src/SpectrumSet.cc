#include <sstream>
#include "lsst/pex/exceptions.h"

#include "pfs/drp/stella/SpectrumSet.h"
//#include "pfs/drp/stella/math/Math.h"
//#include "pfs/drp/stella/math/CurveFitting.h"
//#include "pfs/drp/stella/cmpfit-1.2/MPFitting_ndarray.h"
#include "pfs/drp/stella/utils/checkSize.h"

namespace pfs { namespace drp { namespace stella {

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


ndarray::Array<std::size_t, 1, 1> SpectrumSet::getAllFiberIds() const {
    ndarray::Array<std::size_t, 1, 1> fiberId = ndarray::allocate(size());
    for (std::size_t ii = 0; ii < size(); ++ii) {
        fiberId[ii] = get(ii)->getFiberId();
    }
    return fiberId;
}


SpectrumSet::ImageArray SpectrumSet::getAllFluxes() const {
    ImageArray flux = ndarray::allocate(size(), _length);
    for (std::size_t ii = 0; ii < size(); ++ii) {
        flux[ii] = get(ii)->getSpectrum();
    }
    return flux;
}


SpectrumSet::WavelengthArray SpectrumSet::getAllWavelengths() const {
    WavelengthArray lambda = ndarray::allocate(size(), _length);
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

}}} // namespace pfs::drp::stella
