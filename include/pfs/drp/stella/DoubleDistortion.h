#ifndef PFS_DRP_STELLA_DOUBLEDISTORTION_H
#define PFS_DRP_STELLA_DOUBLEDISTORTION_H

#include <map>
#include "ndarray_fwd.h"

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"

#include "pfs/drp/stella/math/NormalizedPolynomial.h"
#include "pfs/drp/stella/Distortion.h"
#include "pfs/drp/stella/PolynomialDistortion.h"

namespace pfs {
namespace drp {
namespace stella {

/// Distortion of the detector
///
/// A two-dimensional polynomial distortion field accounts for the camera
/// optics and features on the detector. We use one of these distortion fields
/// separately for each of the two CCDs.
///
/// The model parameters are:
/// * xLeft, yLeft: 2D polynomial distortion field coefficients for left CCD.
/// * xRight, yRight: 2D polynomial distortion field coefficients for right CCD.
class DoubleDistortion : public AnalyticDistortion<DoubleDistortion> {
  public:
    using Polynomial = math::NormalizedPolynomial2<double>;

    /// Ctor
    ///
    /// @param order : polynomial order for distortion
    /// @param range : range of input values in x and y
    /// @param coeff : distortion coefficients
    DoubleDistortion(
        int order,
        lsst::geom::Box2D const& range,
        Array1D const& coeff
    );

    /// Ctor
    ///
    /// @param order : polynomial order for distortion
    /// @param range : range of input values in x and y
    /// @param xLeft : distortion field parameters for x, left CCD
    /// @param yLeft : distortion field parameters for y, left CCD
    /// @param xRight : distortion field parameters for x, right CCD
    /// @param yRight : distortion field parameters for y, right CCD
    DoubleDistortion(
        int order,
        lsst::geom::Box2D const& range,
        Array1D const& xLeft,
        Array1D const& yLeft,
        Array1D const& xRight,
        Array1D const& yRight
    );

    virtual ~DoubleDistortion() {}
    DoubleDistortion(DoubleDistortion const&) = default;
    DoubleDistortion(DoubleDistortion &&) = default;
    DoubleDistortion & operator=(DoubleDistortion const&) = default;
    DoubleDistortion & operator=(DoubleDistortion &&) = default;

    //@{
    /// Evaluate the model
    ///
    /// @param x : x position (pixels)
    /// @param y : y position (pixels)
    /// @param onRightCcd : whether position is on right CCD
    /// @return dx,dy distortion for detector
    virtual lsst::geom::Point2D evaluate(lsst::geom::Point2D const& xy) const override {
        return evaluate(xy, getOnRightCcd(xy[0]));
    }
    lsst::geom::Point2D evaluate(lsst::geom::Point2D const& xy, bool onRightCcd) const;
    //@}

    /// Zero out low-order coefficients
    virtual DoubleDistortion removeLowOrder(int order) const override;

    /// Merge distortions using low-order coefficients from another distortion
    virtual DoubleDistortion merge(DoubleDistortion const& other) const override;

    //@{
    /// Accessors
    PolynomialDistortion const& getLeft() const { return _left; }
    PolynomialDistortion const& getRight() const { return _right; }
    Polynomial const& getXLeft() const { return getLeft().getXPoly(); }
    Polynomial const& getYLeft() const { return getLeft().getYPoly(); }
    Polynomial const& getXRight() const { return getRight().getXPoly(); }
    Polynomial const& getYRight() const { return getRight().getYPoly(); }
    //@}

    //@{
    /// Return the distortion polynomial coefficients
    Array1D getXLeftCoefficients() const { return getLeft().getXCoefficients(); }
    Array1D getYLeftCoefficients() const { return getLeft().getYCoefficients(); }
    Array1D getXRightCoefficients() const { return getRight().getXCoefficients(); }
    Array1D getYRightCoefficients() const { return getRight().getYCoefficients(); }
    //@}

    //@{
    /// Return number of distortion parameters per axis
    static std::size_t getNumDistortionForOrder(int distortionOrder) {
        return Polynomial::nParametersFromOrder(distortionOrder);
    }
    std::size_t getNumDistortion() const {
        return getNumDistortionForOrder(getOrder());
    }
    //@}

    //@{
    /// Return whether the x position is on the right CCD
    bool getOnRightCcd(double xx) const {
        return xx > _range.getCenterX();
    }
    ndarray::Array<bool, 1, 1> getOnRightCcd(ndarray::Array<double, 1, 1> const& xx) const;
    //@}


    class Factory;

  protected:
    friend std::ostream& operator<<(std::ostream& os, DoubleDistortion const& model);

    virtual std::string getPersistenceName() const override { return "DoubleDistortion"; }
    virtual void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override;

  private:
    // Calculation parameters

    PolynomialDistortion _left;  // distortion for left CCD
    PolynomialDistortion _right;  // distortion for right CCD

    /// Split single coefficients array into xLeft, yLeft, xRight, yRight
    ///
    /// @param order : Distortion order
    /// @param coeff : Coefficients array
    /// @return xLeft, yLeft, xRight, yRight coefficients as columns for distortion
    static Array2D splitCoefficients(int order, Array1D const& coeff);

    /// Join separate (x/y)(Left/Right) coefficient arrays into a single array
    ///
    /// @param order : Distortion order
    /// @param xLeft : distortion coefficients in x for left CCD
    /// @param yLeft : distortion coefficients in y for left CCD
    /// @param xRight : distortion coefficients in x for right CCD
    /// @param yRight : distortion coefficients in x for right CCD
    /// @return single coefficients array
    static Array1D joinCoefficients(
        int order,
        Array1D const& xLeft,
        Array1D const& yLeft,
        Array1D const& xRight,
        Array1D const& yRight
    );

    /// Ctor
    ///
    /// Construct from the 2D array representation of the coefficients (output
    /// of splitCoefficients).
    ///
    /// @param order : polynomial order for distortion
    /// @param range : range of input values in x and y
    /// @param coeff : 2D coefficients arrays (output of splitCoefficients)
    DoubleDistortion(
        int order,
        lsst::geom::Box2D const& range,
        Array2D const& coeff
    );

};


}}}  // namespace pfs::drp::stella

#endif  // include guard
