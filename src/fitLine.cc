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
    using IndexArray = ndarray::Array<std::ptrdiff_t, 1, 1>;
    using Array = ndarray::Array<T const, 1, 1>;
    using BoolArray = ndarray::Array<bool, 1, 1>;

    explicit MinimizationFunction(
        IndexArray const& indices,
        Array const& values,
        BoolArray const& ignore
    ) : _num(indices.getNumElements()),
        _indices(indices),
        _values(values),
        _ignore(ignore) {
            assert(values.getNumElements() == _num);
            assert(ignore.getNumElements() == _num);
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
    IndexArray const _indices;
    Array const _values;
    BoolArray const _ignore;
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
        if (_ignore[ii]) {
            continue;
        }
        std::ptrdiff_t const xx = _indices[ii];
        double const background = bg0 + bg1*(xx - center);
        double const value = amplitude*std::exp(-0.5*std::pow((xx - center)/rmsSize, 2)) + background;
        chi2 += std::pow(value - _values[ii], 2);
    }
    return chi2;
}

}  // anonymous namespace

template <typename T>
FitLineResult fitLine(
    ndarray::Array<T const, 1, 1> const& flux,
    ndarray::Array<lsst::afw::image::MaskPixel const, 1, 1> const& mask,
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
    ndarray::Array<std::ptrdiff_t, 1, 1> indices = ndarray::allocate(size);
    ndarray::Array<bool, 1, 1> ignore = ndarray::allocate(size);
    std::size_t count = 0;
    for (std::size_t ii = low, jj = 0; ii <= high; ++ii, ++jj) {
        indices[jj] = ii;
        ignore[jj] = mask[ii] & badBitMask;
        if (!ignore[jj]) {
            ++count;
        }
    }
    if (count < 5) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                          (boost::format("Insufficient good points (%d) fitting line at %f") %
                           count % peakPosition).str());
    }

    MinimizationFunction<T> func(indices, flux[ndarray::view(low, high + 1)].shallow(), ignore);

    double const amplitude = flux[int(peakPosition + 0.5)];
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
        count,
        min.UserParameters().Params()[0],
        min.UserParameters().Params()[1],
        min.UserParameters().Params()[2],
        min.UserParameters().Params()[3],
        min.UserParameters().Params()[4],
        min.UserParameters().Errors()[0],
        min.UserParameters().Errors()[1],
        min.UserParameters().Errors()[2],
        min.UserParameters().Errors()[3],
        min.UserParameters().Errors()[4]
    );
}


// Explicit instantiation
#define INSTANTIATE(TYPE) \
template FitLineResult fitLine<TYPE>( \
    ndarray::Array<TYPE const, 1, 1> const&, \
    ndarray::Array<lsst::afw::image::MaskPixel const, 1, 1> const&, \
    float, \
    float, \
    lsst::afw::image::MaskPixel, \
    std::size_t \
);


INSTANTIATE(float);


}}}  // namespace pfs::drp::stella
