#ifndef PFS_DRP_STELLA_NORMALIZEDPOLYNOMIAL_H
#define PFS_DRP_STELLA_NORMALIZEDPOLYNOMIAL_H

#include <memory>
#include <vector>
#include <algorithm>

#include "ndarray_fwd.h"

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"
#include "lsst/afw/math/FunctionLibrary.h"

#include "pfs/drp/stella/utils/math.h"


namespace pfs {
namespace drp {
namespace stella {
namespace math {


/// 1D ordinary polynomial, with inputs normalized to be in the range [-1,1)
///
/// Subclass of lsst::afw::math::PolynomialFunction1 (ordinary polynomial) with
/// the input normalization copied from lsst::afw::math::Chebyshev1Function1.
template <typename T>
class NormalizedPolynomial1 : public lsst::afw::math::PolynomialFunction1<T> {
  public:
    /// Ctor
    ///
    /// @param order : Polynomial order (identical for x and y)
    /// @param min : Minimum input value
    /// @param max : Maximum input value
    explicit NormalizedPolynomial1(
        unsigned int order,
        double min=-1,
        double max=1
    ) : lsst::afw::math::PolynomialFunction1<T>(order) {
        _initialize(min, max);
    }

    /// Ctor
    ///
    /// @param parameters : Polynomial coefficients
    /// @param min : Minimum input value
    /// @param max : Maximum input value
    explicit NormalizedPolynomial1(
        ndarray::Array<double, 1, 1> const& parameters,
        double min=-1,
        double max=1
    ) : NormalizedPolynomial1(utils::arrayToVector(parameters), min, max)
    {}

    /// Ctor
    ///
    /// @param parameters : Polynomial coefficients
    /// @param min : Minimum input value
    /// @param max : Maximum input value
    explicit NormalizedPolynomial1(
        std::vector<double> const& parameters,
        double min=-1,
        double max=1
    ) : lsst::afw::math::PolynomialFunction1<T>(parameters) {
        _initialize(min, max);
    }

    NormalizedPolynomial1(NormalizedPolynomial1 const&) = default;
    NormalizedPolynomial1(NormalizedPolynomial1&&) = default;
    NormalizedPolynomial1& operator=(NormalizedPolynomial1 const&) = default;
    NormalizedPolynomial1& operator=(NormalizedPolynomial1&&) = default;
    ~NormalizedPolynomial1() noexcept override = default;

    /// Polymorphic clone
    std::shared_ptr<lsst::afw::math::Function1<T>> clone() const override {
        return std::make_shared<NormalizedPolynomial1<T>>(this->getParameters(), getMin(), getMax());
    }

    //@{
    /// Return the bounds of input coordinates used for normalization
    double getMin() const { return _min; }
    double getMax() const { return _max; }
    //@}

    /// Normalize input coordinate
    double normalize(double x) const {
        return (x - _offset)*_scale;
    }

    //@{
    /// Evaluate at coordinates
    T operator()(double x) const noexcept override {
        return lsst::afw::math::PolynomialFunction1<T>::operator()(normalize(x));
    }
    ndarray::Array<T, 1, 1> operator()(ndarray::Array<double, 1, 1> const& x) const noexcept {
        ndarray::Array<T, 1, 1> result = ndarray::allocate(x.size());
        std::transform(x.begin(), x.end(), result.begin(),
                       [this](double value) { return operator()(value); });
        return result;
    }
    //@}

    /// Calculate the derivatives of the function w.r.t. the parameters
    ///
    /// Useful for constructing the design matrix when fitting.
    ///
    /// Strangely, this isn't implemented in the base class for 1D functions,
    /// so we implement it here.
    std::vector<double> getDFuncDParameters(double x) const {
        double const xNorm = (x - _offset)*_scale;
        int const num = this->getNParameters();
        std::vector<double> result;
        assert(num >= 1);
        result.push_back(1.0);
        for (int ii = 0; ii < num - 1; ++ii) {
            result.push_back(result[ii]*xNorm);
        }
        return result;
    }

    /// Not persistable because we haven't written the persistence code
    bool isPersistable() const noexcept override { return false; }

  private:

    /// Initialize values used for normalization
    void _initialize(double min, double max) {
        _min = min;
        _max = max;
        _offset = 0.5*(min + max);
        _scale = 2.0/(max - min);
    }

    double _min, _max;  ///< Range of input coordinates
    double _offset;  ///< Offset for normalization
    double _scale;  ///< Scale for normalization
};


/// 2D ordinary polynomial, with inputs normalized to be in the range [-1,1)
///
/// Subclass of lsst::afw::math::PolynomialFunction2 (ordinary polynomial) with
/// the input normalization copied from lsst::afw::math::Chebyshev1Function2.
template <typename T>
class NormalizedPolynomial2 : public lsst::afw::math::PolynomialFunction2<T> {
  public:
    /// Ctor
    ///
    /// @param order : Polynomial order (identical for x and y)
    /// @param range : Bounds of input coordinates, for normalization
    explicit NormalizedPolynomial2(
        unsigned int order,
        lsst::geom::Box2D const& range=lsst::geom::Box2D(
            lsst::geom::Point2D(-1.0, -1.0),
            lsst::geom::Point2D(1.0, 1.0))
    ) : lsst::afw::math::PolynomialFunction2<T>(order) {
        _initialize(range);
    }

    /// Ctor
    ///
    /// @param parameters : Polynomial coefficients
    /// @param range : Bounds of input coordinates, for normalization
    explicit NormalizedPolynomial2(
        ndarray::Array<double, 1, 1> const& parameters,
        lsst::geom::Box2D const& range=lsst::geom::Box2D(
            lsst::geom::Point2D(-1.0, -1.0),
            lsst::geom::Point2D(1.0, 1.0))
    ) : NormalizedPolynomial2(utils::arrayToVector(parameters), range)
    {}

    /// Ctor
    ///
    /// @param parameters : Polynomial coefficients
    /// @param range : Bounds of input coordinates, for normalization
    explicit NormalizedPolynomial2(
        std::vector<double> const& parameters,
        lsst::geom::Box2D const& range=lsst::geom::Box2D(
            lsst::geom::Point2D(-1.0, -1.0),
            lsst::geom::Point2D(1.0, 1.0))
    ) : lsst::afw::math::PolynomialFunction2<T>(parameters) {
        _initialize(range);
    }

    NormalizedPolynomial2(NormalizedPolynomial2 const&) = default;
    NormalizedPolynomial2(NormalizedPolynomial2&&) = default;
    NormalizedPolynomial2& operator=(NormalizedPolynomial2 const&) = default;
    NormalizedPolynomial2& operator=(NormalizedPolynomial2&&) = default;
    ~NormalizedPolynomial2() noexcept override = default;

    /// Polymorphic clone
    std::shared_ptr<lsst::afw::math::Function2<T>> clone() const override {
        return std::make_shared<NormalizedPolynomial2<T>>(this->getParameters(), getXYRange());
    }

    /// Return the bounds of input coordinates used for normalization
    lsst::geom::Box2D getXYRange() const { return _range; }

    /// Normalize input coordinates
    lsst::geom::Point2D normalize(lsst::geom::Point2D const& xy) const {
        return lsst::geom::Point2D(
            (xy.getX() - _xOffset)*_xScale,
            (xy.getY() - _yOffset)*_yScale
        );
    }

    //@{
    /// Evaluate at coordinates
    T operator()(double x, double y) const noexcept override {
        auto const norm = normalize(lsst::geom::Point2D(x, y));
        return lsst::afw::math::PolynomialFunction2<T>::operator()(norm.getX(), norm.getY());
    }
    ndarray::Array<double, 1, 1> operator()(
        ndarray::Array<double, 1, 1> const& x,
        ndarray::Array<double, 1, 1> const& y
    ) const noexcept {
        utils::checkSize(x.size(), y.size(), "x vs y");
        std::size_t const num = x.size();
        ndarray::Array<double, 1, 1> result = ndarray::allocate(num);
        for (std::size_t ii = 0; ii < num; ++ii) {
            result[ii] = operator()(x[ii], y[ii]);
        }
        return result;
    }
    //@}

    /// Calculate the derivatives of the function w.r.t. the parameters
    ///
    /// Useful for constructing the design matrix when fitting.
    std::vector<double> getDFuncDParameters(double x, double y) const override {
        return lsst::afw::math::PolynomialFunction2<T>::getDFuncDParameters(
            (x - _xOffset)*_xScale,
            (y - _yOffset)*_yScale
        );
    }

    /// Not persistable because we haven't written the persistence code
    bool isPersistable() const noexcept override { return false; }

  private:

    /// Initialize values used for normalization
    void _initialize(lsst::geom::Box2D const& range) {
        _range = range;
        _xOffset = 0.5*(range.getMinX() + range.getMaxX());
        _yOffset = 0.5*(range.getMinY() + range.getMaxY());
        _xScale = 2.0/(range.getMaxX() - range.getMinX());
        _yScale = 2.0/(range.getMaxY() - range.getMinY());
    }

    lsst::geom::Box2D _range;  ///< Bounds of input coordinates
    double _xOffset, _yOffset;  ///< Offset for normalization
    double _xScale, _yScale;  ///< Scale for normalization
};


}}}}  // namespace pfs::drp::stella::math

#endif  // include guard
