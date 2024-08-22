#include <string>
#include <sstream>
#include <algorithm>
#include "ndarray.h"
#include "ndarray/eigen.h"
#include "lsst/log/Log.h"

#include "pfs/drp/stella/Spectrum.h"
#include "pfs/drp/stella/utils/checkSize.h"


namespace pfs { namespace drp { namespace stella {

Spectrum::Spectrum(std::size_t length, int fiberId)
  : _length(length),
    _flux(ndarray::allocate(length)),
    _mask(length, 1),
    _norm(ndarray::allocate(length)),
    _variance(ndarray::allocate(length)),
    _wavelength(ndarray::allocate(length)),
    _fiberId(fiberId),
    _isWavelengthSet(false),
    _notes(std::make_shared<lsst::daf::base::PropertySet>(true))
{
    _flux.deep() = 0.;
    _mask.getArray().deep() = 0;
    _norm.deep() = 1.0;
    _variance.deep() = 0.;
    _wavelength.deep() = 0.;
}


Spectrum::Spectrum(
    ImageArray const& flux,
    Mask const& mask,
    ImageArray const& norm,
    VarianceArray const& variance,
    WavelengthArray const& wavelength,
    int fiberId,
    std::shared_ptr<lsst::daf::base::PropertySet> notes
) : _length(flux.getNumElements()),
    _flux(flux),
    _mask(mask),
    _norm(norm),
    _variance(variance),
    _wavelength(wavelength),
    _fiberId(fiberId),
    _isWavelengthSet(!wavelength.empty()),
    _notes(notes ? notes : std::make_shared<lsst::daf::base::PropertySet>(true))
{}


void Spectrum::setFlux(Spectrum::ImageArray const& flux) {
    utils::checkSize(flux.getShape()[0], _length, "Spectrum::setFlux");
    _flux.deep() = flux;
}


void Spectrum::setMask(const Mask & mask) {
    utils::checkSize(mask.getDimensions(), lsst::geom::Extent2I(_length, 1), "Spectrum::setMask");
    _mask = Mask(mask, true);
}


void Spectrum::setNorm(Spectrum::ImageArray const& norm) {
    utils::checkSize(norm.getShape()[0], _length, "Spectrum::setNorm");
    _norm.deep() = norm;
}


void Spectrum::setVariance(Spectrum::VarianceArray const& variance) {
    utils::checkSize(variance.getShape()[0], _length, "Spectrum::setVariance");
    _variance.deep() = variance;
}


void Spectrum::setWavelength(Spectrum::WavelengthArray const& wavelength) {
    utils::checkSize(wavelength.getShape()[0], _length, "Spectrum::setWavelength");
    _wavelength.deep() = wavelength;
    _isWavelengthSet = true;
}


Spectrum::ImageArray Spectrum::getNormFlux() const {
    Spectrum::ImageArray normFlux = ndarray::allocate(_length);
    ndarray::asEigenArray(normFlux) = ndarray::asEigenArray(_flux)/ndarray::asEigenArray(_norm);
    return normFlux;
}

}}} // namespace pfs::drp::stella
