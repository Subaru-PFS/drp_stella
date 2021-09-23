#ifndef PFS_DRP_STELLA_DISTORTION_H
#define PFS_DRP_STELLA_DISTORTION_H

#include "ndarray_fwd.h"

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"

#include "pfs/drp/stella/utils/checkSize.h"


namespace pfs {
namespace drp {
namespace stella {

/// Distortion of the detector
///
/// This abstract base class defines the interface for distortion. We use CRTP
/// in order to force the subclass to define the static methods, e.g.,
///
///     class MyDistortion : public BaseDistortion<MyDistortion> { ... };
///     template<> ndarray::Array<...> BaseDistortion<MyDistortion>::calculateDesignMatrix(...) { ... }
///
/// See https://stackoverflow.com/a/23292378/834250 .
///
/// This is required for 'calculateDesignMatrix' and 'getNumParametersForOrder'.
/// The pure-virtual methods 'evaluate', 'merge' and 'removeLowOrder' also need
/// to be defined.
template <typename Derived>
class BaseDistortion {
  public:
    using Array1D = ndarray::Array<double, 1, 1>;
    using Array2D = ndarray::Array<double, 2, 1>;

    BaseDistortion(
        int order,
        lsst::geom::Box2D const& range,
        Array1D const& coeff
    );

    virtual ~BaseDistortion() {}
    BaseDistortion(BaseDistortion const&) = default;
    BaseDistortion(BaseDistortion &&) = default;
    BaseDistortion & operator=(BaseDistortion const&) = default;
    BaseDistortion & operator=(BaseDistortion &&) = default;

    //@{
    /// Evaluate the model
    ///
    /// @param x : x position (pixels)
    /// @param y : y position (pixels)
    /// @return dx,dy distortion for detector
    lsst::geom::Point2D operator()(lsst::geom::Point2D const& xy) const {
        return evaluate(xy);
    }
    lsst::geom::Point2D operator()(double x, double y) const {
        return operator()(lsst::geom::Point2D(x, y));
    }
    Array2D operator()(
        Array1D const& xx,
        Array1D const& yy
    ) const;
    Array2D operator()(Array2D const& xy) const {
        return operator()(xy[ndarray::view(0)], xy[ndarray::view(1)]);
    }
    //@}

    /// Fit the distortion
    ///
    /// @param order : polynomial order for distortion
    /// @param range : range for input x,y values
    /// @param xx : Distortion in x
    /// @param yy : Distortion in y
    /// @param xMeas : Measured x position
    /// @param yMeas : Measured y position
    /// @param xErr : Error in measured x position
    /// @param yErr : Error in measured y position
    /// @param fitStatic : fit static features?
    /// @returns design matrix
    static Derived fit(
        int order,
        lsst::geom::Box2D const& range,
        Array1D const& xx,
        Array1D const& yy,
        Array1D const& xMeas,
        Array1D const& yMeas,
        Array1D const& xErr,
        Array1D const& yErr,
        bool fitStatic=true
    );

    /// Calculate chi^2 for a particular dataset
    ///
    /// @param xOrig : x values of undistorted position for arc lines
    /// @param yOrig : x values of undistorted position for arc lines
    /// @param xMeas : Measured x values for arc lines
    /// @param yMeas : Measured y values for arc lines
    /// @param xErr : error in x values for arc lines
    /// @param yErr : error in y values for arc lines
    /// @param good : whether value should be used in the fit
    /// @param sysErr : systematic error to add in quadrature
    /// @return chi2 and number of degrees of freedom
    std::pair<double, std::size_t> calculateChi2(
        Array1D const& xOrig,
        Array1D const& yOrig,
        Array1D const& xMeas,
        Array1D const& yMeas,
        Array1D const& xErr,
        Array1D const& yErr,
        ndarray::Array<bool, 1, 1> const& good=ndarray::Array<bool, 1, 1>(),
        float sysErr=0.0
    ) const;

    /// Zero out low-order coefficients
    virtual Derived removeLowOrder(int order) const = 0;

    /// Merge distortions using low-order coefficients from another distortion
    virtual Derived merge(Derived const& other) const = 0;

    //@{
    /// Return the total number of parameters for the model
    static std::size_t getNumParametersForOrder(int order);
    std::size_t getNumParameters() const {
        return getNumParametersForOrder(getOrder());
    }
    //@}

    //@{
    /// Accessors
    int getOrder() const { return _order; }
    lsst::geom::Box2D getRange() const { return _range; }
    Array1D getCoefficients() const { return _coeff; }
    //@}

  protected:
    int _order;  // Order for distortion
    lsst::geom::Box2D _range;  // Range of input x,y positions
    Array1D _coeff;  // Coefficients

  private:

    /// Evaluate the distortion
    virtual lsst::geom::Point2D evaluate(lsst::geom::Point2D const& xy) const = 0;
};


}}}  // namespace pfs::drp::stella

#endif  // include guard
