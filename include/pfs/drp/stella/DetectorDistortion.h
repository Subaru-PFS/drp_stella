#ifndef PFS_DRP_STELLA_DETECTORDISTORTION_H
#define PFS_DRP_STELLA_DETECTORDISTORTION_H

#include <tuple>
#include "ndarray_fwd.h"

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"
#include "lsst/geom/AffineTransform.h"

#include "pfs/drp/stella/Distortion.h"
#include "pfs/drp/stella/math/NormalizedPolynomial.h"


namespace pfs {
namespace drp {
namespace stella {

/// Distortion of the detector
///
/// A two-dimensional polynomial distortion field accounts for the camera
/// optics. We allow for the detector to be comprised of two discrete elements
/// with a relative offset and rotation.
///
/// The model parameters are:
/// * xDistortion, yDistortion: 2D polynomial distortion field coefficients.
/// * rightCcd: 2D affine transformation coefficients for the right CCD.
class DetectorDistortion : public AnalyticDistortion<DetectorDistortion> {
  public:
    using Polynomial = math::NormalizedPolynomial2<double>;

    /// Ctor
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param range : range of input values in x and y
    /// @param coeff : distortion coefficients
    DetectorDistortion(
        int distortionOrder,
        lsst::geom::Box2D const& range,
        Array1D const& coeff
    ) : DetectorDistortion(distortionOrder, range, splitCoefficients(distortionOrder, coeff))
    {}

    /// Ctor
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param range : range of input values in x and y
    /// @param xDistortion : distortion field parameters for x
    /// @param yDistortion : distortion field parameters for y
    /// @param rightCcd : affine transformation parameters for the right CCD
    DetectorDistortion(
        int order,
        lsst::geom::Box2D const& range,
        Array1D const& xDistortion,
        Array1D const& yDistortion,
        Array1D const& rightCcd
    );

    virtual ~DetectorDistortion() {}
    DetectorDistortion(DetectorDistortion const&) = default;
    DetectorDistortion(DetectorDistortion &&) = default;
    DetectorDistortion & operator=(DetectorDistortion const&) = default;
    DetectorDistortion & operator=(DetectorDistortion &&) = default;

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

    //@{
    /// Accessors
    Polynomial const& getXDistortion() const { return _xDistortion; }
    Polynomial const& getYDistortion() const { return _yDistortion; }
    lsst::geom::AffineTransform getRightCcd() const { return _rightCcd; }
    //@}

    //@{
    /// Return the distortion polynomial coefficients
    Array1D getXCoefficients() const;
    Array1D getYCoefficients() const;
    //@}

    /// Return the right ccd affine transform coefficients
    Array1D getRightCcdCoefficients() const;

    /// Generate coefficients for the right CCD affine transform
    ///
    /// @param x : Offset in x
    /// @param y : Offset in y
    /// @param xx : Term in x for the x coordinate
    /// @param xy : Term in y for the x coordinate
    /// @param yx : Term in x for the y coordinate
    /// @param yy : Term in y for the y coordinate
    /// @return coefficient array
    static Array1D makeRightCcdCoefficients(
        double x, double y,
        double xx, double xy,
        double yx, double yy
    );

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
    friend std::ostream& operator<<(std::ostream& os, DetectorDistortion const& model);

    virtual std::string getPersistenceName() const override { return "DetectorDistortion"; }
    virtual void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override;

  private:
    // Calculation parameters
    Polynomial _xDistortion;  // distortion polynomial in x
    Polynomial _yDistortion;  // distortion polynomial in y
    lsst::geom::AffineTransform _rightCcd;  // transformation for right CCD

    /// Split single coefficients array into x, y, rightCcd
    ///
    /// @param order : Distortion order
    /// @param coeff : Coefficients array
    /// @return Separate arrays for x, y, rightCcd coefficients
    static std::tuple<Array1D, Array1D, Array1D> splitCoefficients(int order, Array1D const& coeff);

    /// Join separate x, y, right coefficient arrays into a single array
    ///
    /// @param order : Distortion order
    /// @param xCoeff : distortion field coefficients for x
    /// @param yCoeff : distortion field coefficients for y
    /// @param rightCcd : affine transformation parameters for the right CCD
    /// @return single coefficients array
    static Array1D joinCoefficients(
        int order,
        Array1D const& xCoeff,
        Array1D const& yCoeff,
        Array1D const& rightCcd
    );

    /// Ctor
    ///
    /// Used for construction from the results of 'splitCoefficients'
    ///
    /// @param order : Distortion order
    /// @param range : range of input values in x and y
    /// @param coeff : split coefficients (output from splitCoefficients)
    DetectorDistortion(
        int order,
        lsst::geom::Box2D const& range,
        std::tuple<Array1D, Array1D, Array1D> const& coeff
    ) : DetectorDistortion(order, range, std::get<0>(coeff), std::get<1>(coeff), std::get<2>(coeff))
    {}

};


}}}  // namespace pfs::drp::stella

#endif  // include guard
