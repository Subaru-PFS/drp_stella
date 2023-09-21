#ifndef PFS_DRP_STELLA_POLYNOMIALDISTORTION_H
#define PFS_DRP_STELLA_POLYNOMIALDISTORTION_H

#include <map>
#include "ndarray_fwd.h"

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"
#include "lsst/afw/table/io/Persistable.h"

#include "pfs/drp/stella/math/NormalizedPolynomial.h"
#include "pfs/drp/stella/Distortion.h"

namespace pfs {
namespace drp {
namespace stella {

/// Distortion of the detector
///
/// A two-dimensional polynomial distortion field accounts for the camera
/// optics and features on the detector.
///
/// The model parameters are:
/// * xCoeff, yCoeff: 2D polynomial distortion field coefficients.
class PolynomialDistortion :
  public AnalyticDistortion<PolynomialDistortion> {
  public:
    using Polynomial = math::NormalizedPolynomial2<double>;

    /// Ctor
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param range : range of input values in x and y
    /// @param coeff : distortion coefficients
    PolynomialDistortion(
        int distortionOrder,
        lsst::geom::Box2D const& range,
        Array1D const& coeff
    );

    /// Ctor
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param range : range of input values in x and y
    /// @param xCoeff : distortion field parameters for x
    /// @param yCoeff : distortion field parameters for y
    PolynomialDistortion(
        int distortionOrder,
        lsst::geom::Box2D const& range,
        Array1D const& xCoeff,
        Array1D const& yCoeff
    );

    virtual ~PolynomialDistortion() {}
    PolynomialDistortion(PolynomialDistortion const&) = default;
    PolynomialDistortion(PolynomialDistortion &&) = default;
    PolynomialDistortion & operator=(PolynomialDistortion const&) = default;
    PolynomialDistortion & operator=(PolynomialDistortion &&) = default;

    //@{
    /// Evaluate the model
    ///
    /// @param x : x position (pixels)
    /// @param y : y position (pixels)
    /// @return dx,dy distortion for detector
    virtual lsst::geom::Point2D evaluate(lsst::geom::Point2D const& xy) const override;
    //@}

    //@{
    /// Accessors
    Polynomial const& getXPoly() const { return _xPoly; }
    Polynomial const& getYPoly() const { return _yPoly; }
    //@}

    //@{
    /// Return the distortion polynomial coefficients
    Array1D getXCoefficients() const;
    Array1D getYCoefficients() const;
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


    bool isPersistable() const noexcept { return true; }

    class Factory;

  protected:
    friend std::ostream& operator<<(std::ostream& os, PolynomialDistortion const& model);

    std::string getPersistenceName() const { return "PolynomialDistortion"; }
    std::string getPythonModule() const { return "pfs.drp.stella"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const;

  private:
    // Calculation parameters
    Polynomial _xPoly;  // distortion polynomial in x
    Polynomial _yPoly;  // distortion polynomial in y

    /// Split single coefficients array into xCoeff, yCoeff arrays
    ///
    /// @param order : Distortion order
    /// @param coeff : Coefficients array
    /// @return xCoeff, yCoeff as columns
    static Array2D splitCoefficients(int order, Array1D const& coeff);

    /// Join separate (x/y)(Left/Right) coefficient arrays into a single array
    ///
    /// @param order : Distortion order
    /// @param xCoeff : distortion coefficients in x
    /// @param yCoeff : distortion coefficients in y
    /// @return single coefficients array
    static Array1D joinCoefficients(
        int order,
        Array1D const& xLeft,
        Array1D const& yLeft
    );

    /// Ctor
    ///
    /// Construct from the 2D array representation of the coefficients (output
    /// of splitCoefficients).
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param range : range of input values in x and y
    /// @param coeff : 2D coefficients array (output of splitCoefficients)
    PolynomialDistortion(
        int order,
        lsst::geom::Box2D const& range,
        Array2D const& coeff
    ) : PolynomialDistortion(order, range, coeff[ndarray::view(0)], coeff[ndarray::view(1)])
    {}

};


}}}  // namespace pfs::drp::stella

#endif  // include guard
