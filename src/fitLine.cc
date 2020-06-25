#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/FCNBase.h"

#include "boost/format.hpp"
#include "ndarray.h"
#include "lsst/pex/exceptions.h"

#include "pfs/drp/stella/fitLine.h"

namespace pfs {
namespace drp {
namespace stella {

namespace {

template <typename T>
class MinimizationFunction : public ROOT::Minuit2::FCNBase {
  public:
    using Array = ndarray::Array<T, 1, 1>;
    using Mask = ndarray::Array<bool, 1, 1>;

    explicit MinimizationFunction(
        Array const& indices,
        Array const& values,
        Mask const& mask
    ) : _num(indices.getNumElements()),
        _indices(indices),
        _values(values),
        _mask(mask) {
            assert(values.getNumElements() == _num);
            assert(mask.getNumElements() == _num);
    }

    MinimizationFunction(MinimizationFunction const &) = default;
    MinimizationFunction(MinimizationFunction &&) = default;
    MinimizationFunction &operator=(MinimizationFunction const &) = default;
    MinimizationFunction &operator=(MinimizationFunction &&) = default;
    ~MinimizationFunction() override = default;

    double operator()(std::vector<double> const& parameters) const override;
    double Up() const override { return 1.0; }  // 1.0 --> fitting chi^2

  private:
    std::size_t const _num;
    Array const& _indices;
    Array const& _values;
    Mask const& _mask;
};


template <typename T>
double MinimizationFunction<T>::operator()(std::vector<double> const& parameters) const {
    assert(parameters.size() == 5);
    double const amplitude = parameters[0];
    double const center = parameters[1];
    double const rmsSize = parameters[2];
    double const bg0 = parameters[3];
    double const bg1 = parameters[4];

    double chi2 = 0.0;
    for (std::size_t ii = 0; ii < _num; ++ii) {
        if (_mask[ii]) {
            continue;
        }
        double const xx = _indices[ii];
        double const background = bg0 + bg1*(xx - center);
        double const value = amplitude*std::exp(-0.5*std::pow((xx - center)/rmsSize, 2)) + background;
        chi2 += std::pow(value - _values[ii], 2);
    }
    return chi2;
}

}  // anonymous namespace


FitLineResult fitLine(
    Spectrum const& spectrum,
    float peakPosition,
    float rmsSize,
    lsst::afw::image::MaskPixel badBitMask,
    std::size_t fittingHalfSize
) {
    utils::checkSize(mask.getNumElements(), flux.getNumElements(), "mask vs flux");
    std::size_t const length = flux.getNumElements();
    std::size_t const low = std::max(
        std::size_t(0),
        fittingHalfSize == 0 ? std::size_t(0) : std::size_t(peakPosition - fittingHalfSize)
    );
    std::size_t const high = std::min(
        std::size_t(length - 1),
        fittingHalfSize == 0 ? std::size_t(length - 1) : std::size_t(peakPosition + fittingHalfSize + 0.5)
    );
    std::size_t const size = high - low + 1;
    ndarray::Array<Spectrum::ImageT, 1, 1> indices = ndarray::allocate(size);
    ndarray::Array<Spectrum::ImageT, 1, 1> values = ndarray::allocate(size);
    ndarray::Array<bool, 1, 1> mask = ndarray::allocate(size);
    std::size_t count = 0;
    for (std::size_t ii = low, jj = 0; ii <= high; ++ii, ++jj) {
        indices[jj] = ii;
        values[jj] = spectrum.getSpectrum()[ii];
        mask[jj] = spectrum.getMask()(ii, 0) & badBitMask;
        if (!mask[jj]) {
            ++count;
        }
    }
    if (count < 5) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          (boost::format("Insufficient good points (%d) fitting line at %f") %
                           count % peakPosition).str());
    }
    MinimizationFunction<Spectrum::ImageT> func(indices, values, mask);

    double const amplitude = spectrum.getSpectrum()[int(peakPosition + 0.5)];
    if (!std::isfinite(amplitude) || !std::isfinite(peakPosition) || !std::isfinite(rmsSize)) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          (boost::format("Non-finite initial guess: %f,%f,%f") %
                           amplitude % peakPosition % rmsSize).str());
    }
    std::vector<double> parameters = {amplitude, peakPosition, rmsSize, 0.0, 0.0};
    std::vector<double> steps = {0.1*amplitude, 0.1, 0.1, 1.0, 0.01};

    auto const min = ROOT::Minuit2::MnMigrad(func, parameters, steps)();
    assert(min.UserParameters().Params().size() == 5);
    return FitLineResult(
        std::sqrt(min.Fval()/count),
        min.IsValid() && std::isfinite(min.Fval()),
        min.UserParameters().Params()[0],
        min.UserParameters().Params()[1],
        min.UserParameters().Params()[2],
        min.UserParameters().Params()[3],
        min.UserParameters().Params()[4],
        count
    );
}


}}}  // namespace pfs::drp::stella
