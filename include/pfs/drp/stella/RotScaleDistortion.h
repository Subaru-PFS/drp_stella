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
/// * dxLeft, dyLeft: shift for left CCD
/// * x0Left, y0Left: center of rotation/scale for left CCD (relative to the center of the CCD)
/// * rotLeft: rotation angle for left CCD
/// * xScaleLeft, yScaleLeft: scale factors for left CCD
/// * dxRight, dyRight: shift for right CCD
/// * x0Right, y0Right: center of rotation/scale for right CCD (relative to the center of the CCD)
/// * rotRight: rotation angle for right CCD
/// * xScaleRight, yScaleRight: scale factors for right CCD
///
/// For each CCD, we implement a translation plus a rotation and scale change
/// about x0,y0.
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
    /// @param onRightCcd : whether position is on right CCD
    /// @return distortion for detector
    virtual lsst::geom::Point2D evaluate(lsst::geom::Point2D const& xy) const override {
        return evaluate(xy, getOnRightCcd(xy[0]));
    }
    lsst::geom::Point2D evaluate(lsst::geom::Point2D const& xy, bool onRightCcd) const;
    //@}

    //@{
    /// Return whether the x position is on the right CCD
    bool getOnRightCcd(lsst::geom::Point2D const& point) const { return getOnRightCcd(point.getX()); }
    bool getOnRightCcd(double xx) const {
        return xx > _range.getCenterX();
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
    /// @param useForWavelength : Use line for fitting wavelength solution?
    /// @param maxFuncCalls : Maximum number of function calls in non-linear fitting
    /// @returns distortion object
    static RotScaleDistortion fit(
        lsst::geom::Box2D const& range,
        Array1D const& xx,
        Array1D const& yy,
        Array1D const& xMeas,
        Array1D const& yMeas,
        Array1D const& xErr,
        Array1D const& yErr,
        ndarray::Array<bool, 1, 1> const& useForWavelength,
        std::size_t maxFuncCalls=10000
    );

    class Factory;

  protected:

    enum Parameters {
        DX_LEFT = 0,  // x translation for left CCD
        DY_LEFT = 1,  // y translation for left CCD
        XCEN_LEFT = 2,  // x center of rotation/scale for left CCD
        YCEN_LEFT = 3,  // y center of rotation/scale for right CCD
        ROT_LEFT = 4,  // rotation angle for left CCD
        XSCALE_LEFT = 5,  // x scale factor for left CCD
        YSCALE_LEFT = 6,  // y scale factor for left CCD
        DX_RIGHT = 7,  // x translation for right CCD
        DY_RIGHT = 8,  // y translation for right CCD
        XCEN_RIGHT = 9,  // x center of rotation/scale for right CCD
        YCEN_RIGHT = 10,  // y center of rotation/scale for right CCD
        ROT_RIGHT = 11,  // rotation angle for right CCD
        XSCALE_RIGHT = 12,  // x scale factor for right CCD
        YSCALE_RIGHT = 13,  // y scale factor for right CCD
        NUM_PARAMS  // number of parameters
    };

    friend std::ostream& operator<<(std::ostream& os, RotScaleDistortion const& model);

    virtual std::string getPersistenceName() const override { return "RotScaleDistortion"; }
    virtual void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override;

    lsst::geom::Box2D _range;  // range of input values in x and y
    Array1D _params;  // distortion parameters

    // pre-computed values
    double _x0Left, _y0Left;  // Center of the left CCD
    double _x0Right, _y0Right;  // Center of the right CCD
    double _xTermLeft, _yTermLeft;  // Diagonal terms for left CCD
    double _sinRotLeft;  // Off-diagonal terms for left CCD
    double _xTermRight, _yTermRight;  // Diagonal terms for right CCD
    double _sinRotRight;  // Off-diagonal terms for right CCD
};


}}}  // namespace pfs::drp::stella

#endif  // include guard
