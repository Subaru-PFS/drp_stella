#ifndef PFS_DRP_STELLA_MOSAICPOLYNOMIALDISTORTION_H
#define PFS_DRP_STELLA_MOSAICPOLYNOMIALDISTORTION_H

#include <map>
#include "ndarray_fwd.h"

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"
#include "lsst/geom/AffineTransform.h"
#include "lsst/afw/table/io/Persistable.h"

#include "pfs/drp/stella/math/NormalizedPolynomial.h"
#include "pfs/drp/stella/PolynomialDistortion.h"

namespace pfs {
namespace drp {
namespace stella {

/// Distortion of a mosaicked detector
///
/// PFS optical detectors (b, r and m) are actually composed of two CCDs. The
/// following distortion model puts the CCDs in the same frame through an affine
/// transformation, and then applies a polynomial distortion field over the
/// whole.
///
/// The model parameters are:
/// * affineCoeff: affine transformation coefficients (dx,dy m00, m01, m10, m11)
/// * xCoeff, yCoeff: 2D polynomial distortion field coefficients.
class MosaicPolynomialDistortion :
  public AnalyticDistortion<MosaicPolynomialDistortion> {
  public:
    using Polynomial = math::NormalizedPolynomial2<double>;

    /// Ctor
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param range : range of input values in x and y
    /// @param coeff : distortion coefficients
    MosaicPolynomialDistortion(
        int distortionOrder,
        lsst::geom::Box2D const& range,
        Array1D const& coeff
    );

    /// Ctor
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param range : range of input values in x and y
    /// @param affineCoeff : affine transformation coefficients
    /// @param xCoeff : distortion field parameters for x
    /// @param yCoeff : distortion field parameters for y
    MosaicPolynomialDistortion(
        int distortionOrder,
        lsst::geom::Box2D const& range,
        Array1D const& affineCoeff,
        Array1D const& xCoeff,
        Array1D const& yCoeff
    );

    virtual ~MosaicPolynomialDistortion() {}
    MosaicPolynomialDistortion(MosaicPolynomialDistortion const&) = default;
    MosaicPolynomialDistortion(MosaicPolynomialDistortion &&) = default;
    MosaicPolynomialDistortion & operator=(MosaicPolynomialDistortion const&) = default;
    MosaicPolynomialDistortion & operator=(MosaicPolynomialDistortion &&) = default;

    //@{
    /// Evaluate the model
    ///
    /// @param x : x position (pixels)
    /// @param y : y position (pixels)
    /// @return dx,dy distortion for detector
    virtual lsst::geom::Point2D evaluate(lsst::geom::Point2D const& xy) const override;
    //@}

    //@{
    /// Return whether the x position is on the right CCD
    bool getOnRightCcd(double xx) const {
        return xx > _range.getCenterX();
    }
    ndarray::Array<bool, 1, 1> getOnRightCcd(ndarray::Array<double, 1, 1> const& xx) const;
    //@}

    //@{
    /// Accessors
    lsst::geom::AffineTransform const& getAffine() const { return _affine; }
    Polynomial const& getXPoly() const { return _poly.getXPoly(); }
    Polynomial const& getYPoly() const { return _poly.getYPoly(); }
    //@}

    //@{
    /// Return the distortion coefficients
    Array1D getAffineCoefficients() const;
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
    friend std::ostream& operator<<(std::ostream& os, MosaicPolynomialDistortion const& model);

    std::string getPersistenceName() const { return "MosaicPolynomialDistortion"; }
    std::string getPythonModule() const { return "pfs.drp.stella"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const;

    /// Split single coefficients array into affineCoeff, xCoeff, yCoeff arrays
    ///
    /// @param order : Distortion order
    /// @param coeff : Coefficients array
    /// @return affineCoeff, xCoeff, yCoeff as columns
    static std::tuple<Array1D, Array2D> splitCoefficients(int order, Array1D const& coeff);

    /// Join separate (x/y)(Left/Right) coefficient arrays into a single array
    ///
    /// @param order : Distortion order
    /// @param affineCoeff : affine transformation coefficients
    /// @param xCoeff : distortion coefficients in x
    /// @param yCoeff : distortion coefficients in y
    /// @return single coefficients array
    static Array1D joinCoefficients(
        int order,
        Array1D const& affineCoeff,
        Array1D const& xCoeff,
        Array1D const& yCoeff
    );

    /// Ctor
    ///
    /// Construct from the 2D array representation of the coefficients (output
    /// of splitCoefficients).
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param range : range of input values in x and y
    /// @param coeff : 2D coefficients array (output of splitCoefficients)
    MosaicPolynomialDistortion(
        int order,
        lsst::geom::Box2D const& range,
        std::tuple<Array1D, Array2D> const& coeff
    ) : MosaicPolynomialDistortion(
        order,
        range,
        std::get<0>(coeff),
        std::get<1>(coeff)
     ) {}

    /// Ctor
    ///
    /// Construct from the 2D array representation of the coefficients (output
    /// of splitCoefficients).
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param range : range of input values in x and y
    /// @param affineCoeff : coefficients for affine transformation
    /// @param polyCoeff : 2D coefficients array
    MosaicPolynomialDistortion(
        int order,
        lsst::geom::Box2D const& range,
        Array1D const& affineCoeff,
        Array2D const& polyCoeff
    ) : MosaicPolynomialDistortion(
        order, range, affineCoeff, polyCoeff[ndarray::view(0)], polyCoeff[ndarray::view(1)]
    ) {}

    lsst::geom::Point2D _evaluate(lsst::geom::Point2D const& xy, bool onRightCcd) const;

  private:
    // Calculation parameters
    lsst::geom::AffineTransform _affine;  // affine transformation
    PolynomialDistortion _poly;  // polynomial distortion

};


}}}  // namespace pfs::drp::stella

#endif  // include guard
