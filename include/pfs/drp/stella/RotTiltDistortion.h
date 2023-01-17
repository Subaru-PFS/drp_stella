#ifndef PFS_DRP_STELLA_ROTTILTDISTORTION_H
#define PFS_DRP_STELLA_ROTTILTDISTORTION_H

#include <map>
#include "ndarray_fwd.h"

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"

#include "pfs/drp/stella/math/NormalizedPolynomial.h"
#include "pfs/drp/stella/Distortion.h"

namespace pfs {
namespace drp {
namespace stella {

/// Rot+Tilt distortion of two CCDs
///
/// The model parameters are:
/// * dxLeft, dyLeft: shift for left CCD
/// * rotLeft: rotation angle for left CCD
/// * scaleLeft: scale factor for left CCD
/// * xTiltLeft, yTiltLeft: tilt angle for left CCD
/// * xTiltZeroLeft, yTiltZeroLeft: center of tilt for left CCD
/// * magTiltLeft: magnitude of tilt distortion for left CCD
/// * dxRight, dyRight: shift for right CCD
/// * rotRight: rotation angle for right CCD
/// * scaleRight: scale factor for right CCD
/// * xTiltRight, yTiltRight: tilt angle for right CCD
/// * xTiltZeroRight, yTiltZeroRight: center of tilt for right CCD
/// * magTiltRight: magnitude of tilt distortion for right CCD
class RotTiltDistortion : public Distortion {
  public:
    /// Ctor
    ///
    /// @param range : range of input values in x and y
    /// @param coeff : distortion coefficients
    RotTiltDistortion(
        lsst::geom::Box2D const& range,
        Array1D const& coeff
    );

    virtual ~RotTiltDistortion() {}
    RotTiltDistortion(RotTiltDistortion const&) = default;
    RotTiltDistortion(RotTiltDistortion &&) = default;
    RotTiltDistortion & operator=(RotTiltDistortion const&) = default;
    RotTiltDistortion & operator=(RotTiltDistortion &&) = default;

    virtual std::shared_ptr<Distortion> clone() const {
        return std::make_shared<RotTiltDistortion>(getRange(), getParameters());
    }

    virtual std::size_t getNumParameters() const override { return 18; }

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

    static RotTiltDistortion fit(
        
        lsst::geom::Box2D const& range,
        ndarray::Array<double, 1, 1> const& xx,
        ndarray::Array<double, 1, 1> const& yy,
        ndarray::Array<double, 1, 1> const& xMeas,
        ndarray::Array<double, 1, 1> const& yMeas,
        ndarray::Array<double, 1, 1> const& xErr,
        ndarray::Array<double, 1, 1> const& yErr,
        ndarray::Array<bool, 1, 1> const& useForWavelength,
        bool fit
    );

    class Factory;

  protected:

    enum Parameters {
        DX_LEFT = 0,
        DY_LEFT = 1,
        ROT_LEFT = 2,
        SCALE_LEFT = 3,
        XTILT_LEFT = 4,
        YTILT_LEFT = 5,
        XTILTZERO_LEFT = 6,
        YTILTZERO_LEFT = 7,
        MAGTILT_LEFT = 8,
        DX_RIGHT = 9,
        DY_RIGHT = 10,
        ROT_RIGHT = 11,
        SCALE_RIGHT = 12,
        XTILT_RIGHT = 13,
        YTILT_RIGHT = 14,
        XTILTZERO_RIGHT = 15,
        YTILTZERO_RIGHT = 16,
        MAGTILT_RIGHT = 17,
    };

    lsst::geom::Extent2D evaluateRot(lsst::geom::Point2D const& point) const {
        return evaluateRot(point, getOnRightCcd(point));
    }
    lsst::geom::Extent2D evaluateRot(
        lsst::geom::Point2D const& point,
        bool onRightCcd
    ) const;
    static lsst::geom::Extent2D evaluateRot(
        lsst::geom::Point2D const& point,
        double dx,
        double dy,
        double cosRot,
        double sinRot,
        double scale
    );
    lsst::geom::Extent2D evaluateTilt(lsst::geom::Point2D const& point) const {
        return evaluateTilt(point, getOnRightCcd(point));
    }
    lsst::geom::Extent2D evaluateTilt(
        lsst::geom::Point2D const& point,
        bool onRightCcd
    ) const;
    static lsst::geom::Extent2D evaluateTilt(
        lsst::geom::Point2D const& point,
        double xTilt,
        double yTilt,
        double xTiltZero,
        double yTiltZero,
        double magTilt
    );

    friend std::ostream& operator<<(std::ostream& os, RotTiltDistortion const& model);

    virtual std::string getPersistenceName() const override { return "RotTiltDistortion"; }
    virtual void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override;

    lsst::geom::Box2D _range;  // range of input values in x and y
    Array1D _params;  // distortion parameters

    double _cosRotLeft;
    double _sinRotLeft;
    double _cosRotRight;
    double _sinRotRight;
};


}}}  // namespace pfs::drp::stella

#endif  // include guard
