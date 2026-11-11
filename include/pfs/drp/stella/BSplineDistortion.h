#ifndef PFS_DRP_STELLA_BSPLINEDISTORTION_H
#define PFS_DRP_STELLA_BSPLINEDISTORTION_H

#include <vector>

#include "ndarray_fwd.h"

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"
#include "lsst/afw/table/io/Persistable.h"

#include "pfs/drp/stella/Distortion.h"

namespace pfs {
namespace drp {
namespace stella {


/// Tensor-product B-spline in two dimensions
///
/// The spline is defined by full knot vectors in x and y (including the
/// repeated boundary knots) and a two-dimensional array of coefficients. The
/// spline is evaluated within the range of the knots; positions outside the
/// range are clamped to the range.
///
/// An empty spline (no coefficients) evaluates to zero.
class BSpline2D {
  public:
    using Array1D = ndarray::Array<double, 1, 1>;
    using Array2D = ndarray::Array<double, 2, 1>;

    /// Maximum supported order
    static constexpr int MAX_ORDER = 5;

    /// Ctor for an empty spline, which evaluates to zero
    ///
    /// @param order : order of the spline (3 for cubic)
    explicit BSpline2D(int order=3);

    /// Ctor
    ///
    /// @param order : order of the spline (3 for cubic)
    /// @param xKnots : full knot vector in x
    /// @param yKnots : full knot vector in y
    /// @param coeff : coefficients, with shape (numX, numY), where
    ///     numX = xKnots.size() - order - 1, and similarly for numY.
    BSpline2D(int order, Array1D const& xKnots, Array1D const& yKnots, Array2D const& coeff);

    BSpline2D(BSpline2D const&) = default;
    BSpline2D(BSpline2D &&) = default;
    BSpline2D & operator=(BSpline2D const&) = default;
    BSpline2D & operator=(BSpline2D &&) = default;
    ~BSpline2D() = default;

    /// Return a deep copy
    BSpline2D copy() const;

    //@{
    /// Accessors
    int getOrder() const { return _order; }
    Array1D const& getXKnots() const { return _xKnots; }
    Array1D const& getYKnots() const { return _yKnots; }
    Array2D const& getCoefficients() const { return _coeff; }
    bool isEmpty() const { return _coeff.getNumElements() == 0; }
    std::size_t getNumParameters() const { return _coeff.getNumElements(); }
    //@}

    /// Return the range of the spline
    ///
    /// Throws if the spline is empty.
    lsst::geom::Box2D getRange() const;

    //@{
    /// Evaluate the spline
    double operator()(double x, double y) const;
    double operator()(lsst::geom::Point2D const& xy) const { return operator()(xy.getX(), xy.getY()); }
    Array1D operator()(Array1D const& x, Array1D const& y) const;
    //@}

  private:
    /// Calculate the non-zero basis functions at a position
    ///
    /// @param knots : full knot vector
    /// @param value : position at which to evaluate
    /// @param[out] basis : values of the order + 1 non-zero basis functions
    /// @return index of the first non-zero basis function
    std::size_t calculateBasis(Array1D const& knots, double value, double * basis) const;

    int _order;  ///< order of spline
    Array1D _xKnots;  ///< full knot vector in x
    Array1D _yKnots;  ///< full knot vector in y
    Array2D _coeff;  ///< coefficients, with shape (numX, numY)
};


std::ostream& operator<<(std::ostream& os, BSpline2D const& spline);


/// Distortion given by tensor-product B-splines in x and y
///
/// The detector is divided into regions in x (e.g., the two CCDs of a divided
/// detector), each with its own splines for the distortion in x and in y. This
/// allows the distortion to be discontinuous at the region boundaries. Region
/// ii covers xBoundaries[ii - 1] <= x < xBoundaries[ii].
///
/// Either spline in a region may be empty, in which case it contributes no
/// distortion in that dimension.
class BSplineDistortion : public Distortion {
  public:
    using SplineList = std::vector<BSpline2D>;

    /// Ctor
    ///
    /// @param xBoundaries : boundaries in x between the regions (ascending)
    /// @param xSplines : splines for the distortion in x for each region
    /// @param ySplines : splines for the distortion in y for each region
    BSplineDistortion(
        Array1D const& xBoundaries,
        SplineList const& xSplines,
        SplineList const& ySplines
    );

    virtual ~BSplineDistortion() {}
    BSplineDistortion(BSplineDistortion const&) = default;
    BSplineDistortion(BSplineDistortion &&) = default;
    BSplineDistortion & operator=(BSplineDistortion const&) = default;
    BSplineDistortion & operator=(BSplineDistortion &&) = default;

    virtual std::shared_ptr<Distortion> clone() const override;

    //@{
    /// Accessors
    Array1D const& getXBoundaries() const { return _xBoundaries; }
    SplineList const& getXSplines() const { return _xSplines; }
    SplineList const& getYSplines() const { return _ySplines; }
    std::size_t getNumRegions() const { return _xSplines.size(); }
    //@}

    /// Return the index of the region containing x
    std::size_t getRegion(double x) const;

    /// Return the number of parameters in the model
    virtual std::size_t getNumParameters() const override;

    /// Evaluate the distortion
    virtual lsst::geom::Point2D evaluate(lsst::geom::Point2D const& xy) const override;

    bool isPersistable() const noexcept override { return true; }

    class Factory;

  protected:
    friend std::ostream& operator<<(std::ostream& os, BSplineDistortion const& model);

    std::string getPersistenceName() const override { return "BSplineDistortion"; }
    std::string getPythonModule() const override { return "pfs.drp.stella"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override;

  private:
    Array1D _xBoundaries;  ///< boundaries in x between the regions
    SplineList _xSplines;  ///< splines for the distortion in x for each region
    SplineList _ySplines;  ///< splines for the distortion in y for each region
};


std::ostream& operator<<(std::ostream& os, BSplineDistortion const& model);


}}}  // namespace pfs::drp::stella

#endif  // include guard
