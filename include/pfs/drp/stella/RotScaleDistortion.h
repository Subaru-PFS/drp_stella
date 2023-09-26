#ifndef PFS_DRP_STELLA_ROTSCALEDISTORTION_H
#define PFS_DRP_STELLA_ROTSCALEDISTORTION_H

#include <map>
#include "ndarray_fwd.h"

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"

#include "pfs/drp/stella/math/NormalizedPolynomial.h"
#include "pfs/drp/stella/Distortion.h"

namespace pfs {
namespace drp {
namespace stella {

/// Rotation and scale distortion of two CCDs (separately)
///
/// The model parameters are:
/// * dx, dy: translation
/// * x0Rot, y0Rot: center of rotation (relative to the center)
/// * rot: rotation angle (radians)
/// * x0Scale, y0Scale: center of scale change (relative to the center)
/// * scale: scale factor
///
/// We implement a translation plus a rotation about x0Rot,y0Rot and scale
/// change about x0Scale,y0Scale.
///
/// All positions are relative to x0,y0:
///
/// X = x - x0
/// Y = y - y0
///
/// The rotation gives:
///
/// X' = cos(theta).X - sin(theta).Y
/// X' = sin(theta).X + cos(theta).Y
///
/// The distortion is the difference between the transformed position and the
/// starting position:
///
/// X' - X = cos(theta).X - sin(theta).Y - X = [cos(theta) - 1].X - sin(theta).Y
/// Y' - Y = sin(theta).X + cos(theta).Y - Y = sin(theta).X + [cos(theta) - 1].Y
///
/// Now, introduce a scale:
///
/// X' = xScale.X
/// Y' = yScale.Y
///
/// X' - X = xScale.X - X = (xScale - 1).X
/// Y' - Y = yScale.Y - Y = (yScale - 1).Y
///
/// Let's write "xScale - 1" as "dxScale".
///
/// Adding translation, rotation and scale, we get:
///
/// X' - X = dx + [cos(theta) - 1].X - sin(theta).Y + dxScale.X
/// Y' - Y = dy + sin(theta).X + [cos(theta) - 1].Y + dyScale.Y
///
/// X' - X = dx + [cos(theta) - 1 + dxScale].X - sin(theta).Y
/// Y' - Y = dy + sin(theta).X + [cos(theta) - 1 + dyScale].Y
///
/// X' - X = dx + xTerm.X - sin(theta).Y
/// Y' - Y = dy + sin(theta).X + yTerm.Y
///
class RotScaleDistortion : public Distortion {
  public:
    //@{
    /// Ctor
    ///
    /// @param range : range of input values in x and y
    /// @param parameters : distortion parameters
    RotScaleDistortion(
        lsst::geom::Box2D const& range,
        Array1D const& parameters
    ) : RotScaleDistortion(range, ndarray::copy(parameters)) {}

    RotScaleDistortion(
        lsst::geom::Box2D const& range,
        Array1D && parameters
    );
    //@}

    virtual ~RotScaleDistortion() {}
    RotScaleDistortion(RotScaleDistortion const&) = default;
    RotScaleDistortion(RotScaleDistortion &&) = default;
    RotScaleDistortion & operator=(RotScaleDistortion const&) = default;
    RotScaleDistortion & operator=(RotScaleDistortion &&) = default;

    virtual std::shared_ptr<Distortion> clone() const {
        return std::make_shared<RotScaleDistortion>(getRange(), getParameters());
    }

    virtual std::size_t getNumParameters() const override { return NUM_PARAMS; }

    //@{
    /// Accessors
    lsst::geom::Box2D const& getRange() const { return _range; }
    Array1D const& getParameters() const { return _params; }
    //@}

    //@{
    /// Evaluate the model
    ///
    /// @param x : x position (pixels)
    /// @param y : y position (pixels)
    /// @return distortion for detector
    virtual lsst::geom::Point2D evaluate(lsst::geom::Point2D const& xy) const override;
    //@}

    /// Fit the distortion
    ///
    /// @param range : range for input x,y values
    /// @param xx : Distortion in x
    /// @param yy : Distortion in y
    /// @param xMeas : Measured distortion in x position
    /// @param yMeas : Measured distortion in y position
    /// @param xErr : Error in measured x distortion
    /// @param yErr : Error in measured y distortion
    /// @param slope : (Inverse) slope of trace (dx/dy; pixels per pixel)
    /// @param isLine : Is this a line? Otherwise it's a trace
    /// @param threshold : Least-squares solving threshold
    /// @param forced : Whether to force a parameter to the provided value
    /// @param params : Values of parameters to be forced (only those parameters for
    ///     which forced is true are used)
    /// @returns distortion object
    static RotScaleDistortion fit(
        lsst::geom::Box2D const& range,
        Array1D const& xx,
        Array1D const& yy,
        Array1D const& xMeas,
        Array1D const& yMeas,
        Array1D const& xErr,
        Array1D const& yErr,
        Array1D const& slope,
        ndarray::Array<bool, 1, 1> const& isLine,
        double threshold=1.0e-6,
        ndarray::Array<bool, 1, 1> const& forced=ndarray::Array<bool, 1, 1>(),
        ndarray::Array<double, 1, 1> const& params=ndarray::Array<double, 1, 1>()
    );

    class Factory;

  protected:

    enum Parameters {
        DX = 0,  // x translation
        DY = 1,  // y translation
        DIAG = 2,  // diagonal terms
        OFFDIAG = 3,  // off-diagonal terms
        NUM_PARAMS  // number of parameters
    };

    friend std::ostream& operator<<(std::ostream& os, RotScaleDistortion const& model);

    virtual std::string getPersistenceName() const override { return "RotScaleDistortion"; }
    virtual void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override;

    lsst::geom::Box2D _range;  // range of input values in x and y
    Array1D _params;  // distortion parameters

    // pre-computed values
    double _x0, _y0;  // Center
    double _norm;  // Normalisation for relative positions
};


class DoubleRotScaleDistortion : public Distortion {
  public:
    //@{
    /// Ctor
    ///
    /// @param range : range of input values in x and y
    /// @param parameters : distortion parameters
    DoubleRotScaleDistortion(
        lsst::geom::Box2D const& range,
        Array1D const& parameters
    );

    DoubleRotScaleDistortion(
        lsst::geom::Box2D const& range,
        Array1D const& leftParameters,
        Array1D const& rightParameters
    );

    //@}

    virtual ~DoubleRotScaleDistortion() {}
    DoubleRotScaleDistortion(DoubleRotScaleDistortion const&) = default;
    DoubleRotScaleDistortion(DoubleRotScaleDistortion &&) = default;
    DoubleRotScaleDistortion & operator=(DoubleRotScaleDistortion const&) = default;
    DoubleRotScaleDistortion & operator=(DoubleRotScaleDistortion &&) = default;

    virtual std::shared_ptr<Distortion> clone() const {
        return std::make_shared<DoubleRotScaleDistortion>(getRange(), getParameters());
    }

    virtual std::size_t getNumParameters() const override {
        return _left.getNumParameters() + _right.getNumParameters();
    }

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
    lsst::geom::Point2D evaluate(lsst::geom::Point2D const& xy, bool onRightCcd) const {
        return onRightCcd ? _right.evaluate(xy) : _left.evaluate(xy);
    }
    //@}

    //@{
    /// Accessors
    lsst::geom::Box2D const& getRange() const { return _range; }
    RotScaleDistortion const& getLeft() const { return _left; }
    RotScaleDistortion const& getRight() const { return _right; }
    //@}

    //@{
    /// Return the distortion parameters
    Array1D getParameters() const;
    Array1D getLeftParameters() const { return getLeft().getParameters(); }
    Array1D getRightParameters() const { return getRight().getParameters(); }
    //@}

    //@{
    /// Return whether the x position is on the right CCD
    bool getOnRightCcd(double xx) const {
        return xx > _center;
    }
    ndarray::Array<bool, 1, 1> getOnRightCcd(ndarray::Array<double, 1, 1> const& xx) const;
    //@}

    /// Fit the distortion
    ///
    /// @param range : range for input x,y values
    /// @param xx : Distortion in x
    /// @param yy : Distortion in y
    /// @param xMeas : Measured distortion in x position
    /// @param yMeas : Measured distortion in y position
    /// @param xErr : Error in measured x distortion
    /// @param yErr : Error in measured y distortion
    /// @param slope : (Inverse) slope of trace (dx/dy; pixels per pixel)
    /// @param isLine : Is this a line? Otherwise it's a trace
    /// @param threshold : Least-squares solving threshold
    /// @param forced : Whether to force a parameter to the provided value
    /// @param params : Values of parameters to be forced (only those parameters for
    ///     which forced is true are used)
    /// @returns distortion object
    static DoubleRotScaleDistortion fit(
        lsst::geom::Box2D const& range,
        Array1D const& xx,
        Array1D const& yy,
        Array1D const& xMeas,
        Array1D const& yMeas,
        Array1D const& xErr,
        Array1D const& yErr,
        Array1D const& slope,
        ndarray::Array<bool, 1, 1> const& isLine,
        double threshold=1.0e-6,
        ndarray::Array<bool, 1, 1> const& forced=ndarray::Array<bool, 1, 1>(),
        ndarray::Array<double, 1, 1> const& params=ndarray::Array<double, 1, 1>()
    );

    class Factory;

  protected:
    friend std::ostream& operator<<(std::ostream& os, DoubleRotScaleDistortion const& model);

    virtual std::string getPersistenceName() const override { return "DoubleRotScaleDistortion"; }
    virtual void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override;

  private:
    // Calculation parameters

    lsst::geom::Box2D _range;  // range of input values in x and y
    double _center;  // Center of CCDs in x
    RotScaleDistortion _left;  // distortion for left CCD
    RotScaleDistortion _right;  // distortion for right CCD

    /// Split single parameters array into left, right
    ///
    /// @param parameters : parameters array
    /// @return left, right parameters as columns
    static Array2D splitParameters(Array1D const& parameters);

    /// Join separate left/right parameter arrays into a single array
    ///
    /// @param left : parameters for left CCD
    /// @param right : parameters for right CCD
    /// @return single parameters array
    static Array1D joinParameters(
        Array1D const& left,
        Array1D const& right
    );

    /// Ctor
    ///
    /// Construct from the 2D array representation of the parameters (output
    /// of splitParameters).
    ///
    /// @param order : polynomial order for distortion
    /// @param range : range of input values in x and y
    /// @param coeff : 2D parameters arrays (output of splitParameters)
    DoubleRotScaleDistortion(
        lsst::geom::Box2D const& range,
        Array2D const& parameters
    );

};


}}}  // namespace pfs::drp::stella

#endif  // include guard
