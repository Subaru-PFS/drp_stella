#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/FCNBase.h"

#include "ndarray.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/PolynomialDistortion.h"
#include "pfs/drp/stella/LayeredDetectorMap.h"
#include "pfs/drp/stella/math/AffineTransform.h"

#include "pfs/drp/stella/fitDetectorMap.h"

namespace pfs {
namespace drp {
namespace stella {


namespace {


LayeredDetectorMap makeLayeredDetectorMap(
    SplinedDetectorMap const& base,
    int order,
    ndarray::Array<double, 1, 1> const& parameters,
    ndarray::Array<double, 1, 1> const& spatialOffsets,
    ndarray::Array<double, 1, 1> const& spectralOffsets,
    lsst::afw::image::VisitInfo const& visitInfo
) {
    std::size_t const numDistortion = 2*PolynomialDistortion::getNumDistortionForOrder(order);
    lsst::geom::AffineTransform affine = math::makeAffineTransform(
        parameters[ndarray::view(numDistortion, numDistortion + 6)].shallow()
    );

    LayeredDetectorMap::DistortionList distortions;
    distortions.push_back(
        std::make_shared<PolynomialDistortion>(
            order, lsst::geom::Box2D(base.getBBox()), parameters[ndarray::view(0, numDistortion)].deep()
        )
    );

    return LayeredDetectorMap(
        base.getBBox(), spatialOffsets, spectralOffsets, base, distortions, true, affine, visitInfo
    );
}


template <typename T>
class MinimizationFunction : public ROOT::Minuit2::FCNBase {
  public:
    explicit MinimizationFunction(
        int order,
        SplinedDetectorMap const& base,
        ndarray::Array<int, 1, 1> const& fiberIdLine,
        ndarray::Array<double, 1, 1> const& wavelength,
        ndarray::Array<T, 1, 1> const& xLine,
        ndarray::Array<T, 1, 1> const& yLine,
        ndarray::Array<T, 1, 1> const& xErrLine,
        ndarray::Array<T, 1, 1> const& yErrLine,
        ndarray::Array<int, 1, 1> const& fiberIdTrace,
        ndarray::Array<T, 1, 1> const& xTrace,
        ndarray::Array<T, 1, 1> const& yTrace,
        ndarray::Array<T, 1, 1> const& xErrTrace
    ) : _order(order),
        _numDistortion(PolynomialDistortion::getNumDistortionForOrder(order)),
        _base(base),
        _numLines(fiberIdLine.size()),
        _fiberIdLine(fiberIdLine),
        _wavelength(wavelength),
        _xLine(xLine),
        _yLine(yLine),
        _xErrLine(xErrLine),
        _yErrLine(yErrLine),
        _numTraces(fiberIdTrace.size()),
        _fiberIdTrace(fiberIdTrace),
        _xTrace(xTrace),
        _yTrace(yTrace),
        _xErrTrace(xErrTrace),
        _slitOffsets(ndarray::allocate(base.getNumFibers()))
    {
        _slitOffsets.deep() = 0.0;
    }

    MinimizationFunction(MinimizationFunction const &) = default;
    MinimizationFunction(MinimizationFunction &&) = default;
    MinimizationFunction &operator=(MinimizationFunction const &) = default;
    MinimizationFunction &operator=(MinimizationFunction &&) = default;
    ~MinimizationFunction() override = default;

    double operator()(std::vector<double> const& parameters) const override;
    double Up() const override { return 1.0; }  // 1.0 --> fitting chi^2

  private:
    int _order;
    std::size_t _numDistortion;
    SplinedDetectorMap const& _base;

    std::size_t const _numLines;
    ndarray::Array<int, 1, 1> const& _fiberIdLine;
    ndarray::Array<double, 1, 1> const& _wavelength;
    ndarray::Array<T, 1, 1> const& _xLine;
    ndarray::Array<T, 1, 1> const& _yLine;
    ndarray::Array<T, 1, 1> const& _xErrLine;
    ndarray::Array<T, 1, 1> const& _yErrLine;

    std::size_t const _numTraces;
    ndarray::Array<int, 1, 1> const& _fiberIdTrace;
    ndarray::Array<T, 1, 1> const& _xTrace;
    ndarray::Array<T, 1, 1> const& _yTrace;
    ndarray::Array<T, 1, 1> const& _xErrTrace;

    ndarray::Array<double, 1, 1> _slitOffsets;
};


template <typename T>
double MinimizationFunction<T>::operator()(std::vector<double> const& parameters) const {
    LayeredDetectorMap detMap = makeLayeredDetectorMap(
        _base, _order, utils::vectorToArray(parameters), _slitOffsets, _slitOffsets, _base.getVisitInfo()
    );

    double chi2 = 0.0;

    std::size_t goodLines = 0;
    for (std::size_t ii = 0; ii < _numLines; ++ii) {
        auto const point = detMap.findPoint(_fiberIdLine[ii], _wavelength[ii]);
        double const xFit = point.getX();
        double const yFit = point.getY();
        if (!std::isfinite(xFit) || !std::isfinite(yFit)) {
            continue;
        }
        chi2 += std::pow((xFit - _xLine[ii]) / _xErrLine[ii], 2);
        chi2 += std::pow((yFit - _yLine[ii]) / _yErrLine[ii], 2);
        ++goodLines;
    }

    std::size_t goodTraces = 0;
    for (std::size_t ii = 0; ii < _numTraces; ++ii) {
        double const xFit = detMap.getXCenter(_fiberIdTrace[ii], _yTrace[ii]);
        if (!std::isfinite(xFit)) {
            continue;
        }
        chi2 += std::pow((xFit - _xTrace[ii]) / _xErrTrace[ii], 2);
        ++goodTraces;
    }
    chi2 /= (2*goodLines + goodTraces);
    std::cerr << "params=" << utils::vectorToArray(parameters) << " chi2=" << chi2 << " goodLines=" << goodLines << " goodTraces=" << goodTraces << std::endl;
    return chi2;
}


}  // anonymous namespace


template <typename T>
std::shared_ptr<DetectorMap> fitDetectorMap(
    int order,
    SplinedDetectorMap const& base,
    ndarray::Array<int, 1, 1> const& fiberIdLine,
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<T, 1, 1> const& xLine,
    ndarray::Array<T, 1, 1> const& yLine,
    ndarray::Array<T, 1, 1> const& xErrLine,
    ndarray::Array<T, 1, 1> const& yErrLine,
    ndarray::Array<int, 1, 1> const& fiberIdTrace,
    ndarray::Array<T, 1, 1> const& xTrace,
    ndarray::Array<T, 1, 1> const& yTrace,
    ndarray::Array<T, 1, 1> const& xErrTrace,
    ndarray::Array<double, 1, 1> const& start
) {
    std::size_t numLines = fiberIdLine.size();
    utils::checkSize(wavelength.size(), numLines, "wavelength");
    utils::checkSize(xLine.size(), numLines, "xLine");
    utils::checkSize(yLine.size(), numLines, "yLine");
    utils::checkSize(xErrLine.size(), numLines, "xErrLine");
    utils::checkSize(yErrLine.size(), numLines, "yErrLine");

    std::size_t numTraces = fiberIdTrace.size();
    utils::checkSize(xTrace.size(), numTraces, "xTrace");
    utils::checkSize(yTrace.size(), numTraces, "yTrace");
    utils::checkSize(xErrTrace.size(), numTraces, "xErrTrace");

    std::size_t const numParameters = PolynomialDistortion::getNumDistortionForOrder(order);
    if (!start.empty()) {
        utils::checkSize(start.size(), 2*numParameters + 6, "start vs order");
    }

    MinimizationFunction<T> func{
        order, base,
        fiberIdLine, wavelength, xLine, yLine, xErrLine, yErrLine,
        fiberIdTrace, xTrace, yTrace, xErrTrace
    };



    ROOT::Minuit2::MnUserParameterState state;
    for (std::size_t ii = 0; ii < numParameters; ++ii) {
        state.Add("xDistortion" + std::to_string(ii), start.empty() ? 0.0 : start[ii], 1.0);
    }
    for (std::size_t ii = 0; ii < numParameters; ++ii) {
        state.Add(
            "yDistortion" + std::to_string(ii),
            start.empty() ? 0.0 : start[numParameters + ii],
            1.0
        );
    }
    for (std::size_t ii = 0; ii < 6; ++ii) {
        state.Add(
            "affine" + std::to_string(ii),
            (ii % 3 == 0) ? 1.0 : 0.0,
            0.1
        );
    }

    ROOT::Minuit2::MnStrategy const strategy{1};
    auto const min = ROOT::Minuit2::MnMigrad(func, state, strategy)();
    assert(min.UserParameters().Params().size() == 2*numParameters + 6);

    ndarray::Array<double, 1, 1> slitOffsets = ndarray::allocate(base.getNumFibers());
    slitOffsets.deep() = 0.0;
    return std::make_shared<LayeredDetectorMap>(
        makeLayeredDetectorMap(
            base, order, utils::vectorToArray(min.UserParameters().Params()), slitOffsets, slitOffsets, base.getVisitInfo()
        )
    );
}


// Explicit instantiation
#define INSTANTIATE(T) \
template std::shared_ptr<DetectorMap> fitDetectorMap( \
    int order, \
    SplinedDetectorMap const& base, \
    ndarray::Array<int, 1, 1> const& fiberIdLine, \
    ndarray::Array<double, 1, 1> const& wavelength, \
    ndarray::Array<T, 1, 1> const& xLine, \
    ndarray::Array<T, 1, 1> const& yLine, \
    ndarray::Array<T, 1, 1> const& xErrLine, \
    ndarray::Array<T, 1, 1> const& yErrLine, \
    ndarray::Array<int, 1, 1> const& fiberIdTrace, \
    ndarray::Array<T, 1, 1> const& xTrace, \
    ndarray::Array<T, 1, 1> const& yTrace, \
    ndarray::Array<T, 1, 1> const& xErrTrace, \
    ndarray::Array<double, 1, 1> const& start \
);


INSTANTIATE(float);

}}}  // namespace pfs::drp::stella
