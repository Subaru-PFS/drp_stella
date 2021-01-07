#ifndef PFS_DRP_STELLA_NORMALIZEDPOLYNOMIAL_H
#define PFS_DRP_STELLA_NORMALIZEDPOLYNOMIAL_H

#include <memory>
#include <vector>

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"
#include "lsst/afw/math/FunctionLibrary.h"


namespace pfs {
namespace drp {
namespace stella {
namespace math {


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

    /// Evaluate at coordinates
    T operator()(double x, double y) const noexcept override {
        return lsst::afw::math::PolynomialFunction2<T>::operator()(
            (x - _xOffset)*_xScale,
            (y - _yOffset)*_yScale
        );
    }

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
