#include <string>
#include <sstream>
#include <algorithm>
#include "lsst/log/Log.h"

#include "pfs/drp/stella/Spectrum.h"
#include "pfs/drp/stella/utils/checkSize.h"


namespace pfs { namespace drp { namespace stella {

Spectrum::Spectrum(std::size_t length, int fiberId)
  : _length(length),
    _flux(ndarray::allocate(length)),
    _mask(length, 1),
    _background(ndarray::allocate(length)),
    _covariance(ndarray::allocate(3, length)),
    _wavelength(ndarray::allocate(length)),
    _fiberId(fiberId),
    _isWavelengthSet(false)
{
    _flux.deep() = 0.;
    _mask.getArray().deep() = 0;
    _background.deep() = 0.;
    _covariance.deep() = 0.;
    _wavelength.deep() = 0.;
}


Spectrum::Spectrum(
    ImageArray const& flux,
    Mask const& mask,
    ImageArray const& background,
    CovarianceMatrix const& covariance,
    WavelengthArray const& wavelength,
    ReferenceLineList const& lines,
    int fiberId
) : _length(flux.getNumElements()),
    _flux(flux),
    _mask(mask),
    _background(background),
    _covariance(covariance),
    _wavelength(wavelength),
    _fiberId(fiberId),
    _referenceLines(lines),
    _isWavelengthSet(!wavelength.empty())
{}


void Spectrum::setFlux(Spectrum::ImageArray const& flux) {
    utils::checkSize(flux.getShape()[0], _length, "Spectrum::setFlux");
    _flux.deep() = flux;
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


void Spectrum::setWavelength(Spectrum::WavelengthArray const& wavelength) {
    utils::checkSize(wavelength.getShape()[0], _length, "Spectrum::setWavelength");
    _wavelength.deep() = wavelength;
    _isWavelengthSet = true;
}


}}} // namespace pfs::drp::stella
