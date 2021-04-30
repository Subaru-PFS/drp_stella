#ifndef PFS_DRP_STELLA_DETECTORDISTORTION_H
#define PFS_DRP_STELLA_DETECTORDISTORTION_H

#include <map>
#include "ndarray_fwd.h"

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"
#include "lsst/geom/AffineTransform.h"
#include "lsst/afw/table/io/Persistable.h"

#include "pfs/drp/stella/spline.h"
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
class DetectorDistortion : public lsst::afw::table::io::Persistable {
  public:
    using Polynomial = math::NormalizedPolynomial2<double>;

    /// Ctor
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param range : range of input values in x and y
    /// @param xDistortion : distortion field parameters for x
    /// @param yDistortion : distortion field parameters for y
    /// @param rightCcd : affine transformation parameters for the right CCD
    DetectorDistortion(
        int distortionOrder,
        lsst::geom::Box2D const& range,
        ndarray::Array<double, 1, 1> const& xDistortion,
        ndarray::Array<double, 1, 1> const& yDistortion,
        ndarray::Array<double, 1, 1> const& rightCcd
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
    lsst::geom::Point2D operator()(double x, double y) const {
        return operator()(lsst::geom::Point2D(x, y));
    }
    lsst::geom::Point2D operator()(lsst::geom::Point2D const& xy) const {
        return operator()(xy, getOnRightCcd(xy[0]));
    }
    lsst::geom::Point2D operator()(lsst::geom::Point2D const& xy, bool onRightCcd) const;
    ndarray::Array<double, 2, 1> operator()(
        ndarray::Array<double, 1, 1> const& xx,
        ndarray::Array<double, 1, 1> const& yy
    ) const {
        return operator()(xx, yy, getOnRightCcd(xx));
    }
    ndarray::Array<double, 2, 1> operator()(
        ndarray::Array<double, 1, 1> const& xx,
        ndarray::Array<double, 1, 1> const& yy,
        ndarray::Array<bool, 1, 1> const& onRightCcd
    ) const;
    ndarray::Array<double, 2, 1> operator()(ndarray::Array<double, 2, 1> const& xy) const {
        return operator()(xy, getOnRightCcd(xy[ndarray::view(0)]));
    }
    ndarray::Array<double, 2, 1> operator()(
        ndarray::Array<double, 2, 1> const& xy,
        ndarray::Array<bool, 1, 1> const& onRightCcd
    ) const {
        return operator()(xy[ndarray::view(0)], xy[ndarray::view(1)], onRightCcd);
    }
    //@}

    /// Calculate the design matrix for the distortion polynomials
    ///
    /// The design matrix is the "X" matrix in the classic linear least-squares
    /// equation: X^T*X*beta = X^T*y (where beta are the parameters, y are the
    /// measurements, and "*" denotes matrix multiplication). Rows correspond to
    /// the input data points, and columns correspond to the individual
    /// polynomial terms. The dimensions of the matrix are (Ndata, Ndistortion),
    /// where Ndata is the number of data points provided, and Ndistortion is
    /// the number of distortion polynomial terms.
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param range : range for input x,y values
    /// @param xy : x,y values for data points
    /// @returns design matrix
    static ndarray::Array<double, 2, 1> calculateDesignMatrix(
        int distortionOrder,
        lsst::geom::Box2D const& range,
        ndarray::Array<double, 1, 1> const& xx,
        ndarray::Array<double, 1, 1> const& yy
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
        ndarray::Array<double, 1, 1> const& xOrig,
        ndarray::Array<double, 1, 1> const& yOrig,
        ndarray::Array<bool, 1, 1> const& onRightCcd,
        ndarray::Array<double, 1, 1> const& xMeas,
        ndarray::Array<double, 1, 1> const& yMeas,
        ndarray::Array<double, 1, 1> const& xErr,
        ndarray::Array<double, 1, 1> const& yErr,
        ndarray::Array<bool, 1, 1> const& good=ndarray::Array<bool, 1, 1>(),
        float sysErr=0.0
    ) const;

    //@{
    /// Return the total number of parameters for the model
    static std::size_t getNumParameters(int distortionOrder) {
        return 6 + 2*DetectorDistortion::getNumDistortion(distortionOrder);
    }
    std::size_t getNumParameters() const {
        return getNumParameters(getDistortionOrder());
    }
    //@}

    //@{
    /// Accessors
    int getDistortionOrder() const { return _distortionOrder; }
    lsst::geom::Box2D getRange() const { return _range; }
    Polynomial const& getXDistortion() const { return _xDistortion; }
    Polynomial const& getYDistortion() const { return _yDistortion; }
    lsst::geom::AffineTransform getRightCcd() const { return _rightCcd; }
    //@}

    //@{
    /// Return the distortion polynomial coefficients
    ndarray::Array<double, 1, 1> getXCoefficients() const;
    ndarray::Array<double, 1, 1> getYCoefficients() const;
    //@}

    /// Return the right ccd affine transform coefficients
    ndarray::Array<double, 1, 1> getRightCcdCoefficients() const;

    /// Generate coefficients for the right CCD affine transform
    ///
    /// @param x : Offset in x
    /// @param y : Offset in y
    /// @param xx : Term in x for the x coordinate
    /// @param xy : Term in y for the x coordinate
    /// @param yx : Term in x for the y coordinate
    /// @param yy : Term in y for the y coordinate
    /// @return coefficient array
    static ndarray::Array<double, 1, 1> makeRightCcdCoefficients(
        double x, double y,
        double xx, double xy,
        double yx, double yy
    );

    //@{
    /// Return number of distortion parameters per axis
    static std::size_t getNumDistortion(int distortionOrder) {
        return Polynomial::nParametersFromOrder(distortionOrder);
    }
    std::size_t getNumDistortion() const {
        return getNumDistortion(getDistortionOrder());
    }
    //@}

    //@{
    /// Return whether the x position is on the right CCD
    bool getOnRightCcd(double xx) const {
        return xx > _range.getCenterX();
    }
    ndarray::Array<bool, 1, 1> getOnRightCcd(ndarray::Array<double, 1, 1> const& xx) const;
    //@}


    bool isPersistable() const noexcept { return true; }

    class Factory;

  protected:
    friend std::ostream& operator<<(std::ostream& os, DetectorDistortion const& model);

    std::string getPersistenceName() const { return "DetectorDistortion"; }
    std::string getPythonModule() const { return "pfs.drp.stella"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const;

  private:
    // Configuration
    int _distortionOrder;  // Order for distortion polynomials
    lsst::geom::Box2D _range;  // Range of input x,y positions

    // Calculation parameters
    Polynomial _xDistortion;  // distortion polynomial in x
    Polynomial _yDistortion;  // distortion polynomial in y
    lsst::geom::AffineTransform _rightCcd;  // transformation for right CCD
};


}}}  // namespace pfs::drp::stella

#endif  // include guard
