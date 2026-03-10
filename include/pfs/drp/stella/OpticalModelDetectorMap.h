#ifndef PFS_DRP_STELLA_OPTICALMODELDETECTORMAP_H
#define PFS_DRP_STELLA_OPTICALMODELDETECTORMAP_H

#include "ndarray_fwd.h"

#include "lsst/geom/AffineTransform.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/cpputils/Cache.h"

#include "pfs/drp/stella/SplinedDetectorMap.h"
#include "pfs/drp/stella/Distortion.h"
#include "pfs/drp/stella/OpticalModel.h"


namespace pfs {
namespace drp {
namespace stella {


/// DetectorMap implemented as a series of layers, following the optical path
/// from the slit to the detector.
///
/// The layers are:
/// 1. SlitModel: spectrograph coordinates --> slit coordinates. This is a
///    relatively minor perturbation of the spectrograph coordinates, which
///    includes per-fiber offsets and a (low-order) distortion (e.g., due to
///    movement of the slit head).
/// 2. OpticalModel: slit coordinates --> detector coordinates. This is provided
///    by JEG's optical model (a grid of fiberId,wavelength vs x,y on the
///    detector) with additional distortion applied.
/// 3. DetectorModel: detector coordinates --> pixel coordinates. For the NIR
///    detectors, this is a no-op. For the optical detectors, this accounts for
///    the chip gap and slight rotation differences between the chips, and
///    includes any distortion within the detector (e.g., epoxy blobs and other
///    detector geography).
class OpticalModelDetectorMap : public DetectorMap {
  public:
    using Spline = math::Spline<double>;
    using SplinePtr = std::shared_ptr<Spline>;
    using SplineTuple = std::tuple<SplinePtr, SplinePtr, SplinePtr, SplinePtr>;

    OpticalModelDetectorMap(
        lsst::geom::Box2I const& bbox,
        SlitModel const& slitModel,
        OpticalModel const& opticalModel,
        DetectorModel const& detectorModel,
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr
    );

    virtual ~OpticalModelDetectorMap() {}
    OpticalModelDetectorMap(OpticalModelDetectorMap const&) = default;
    OpticalModelDetectorMap(OpticalModelDetectorMap &&) = default;
    OpticalModelDetectorMap & operator=(OpticalModelDetectorMap const&) = default;
    OpticalModelDetectorMap & operator=(OpticalModelDetectorMap &&) = default;

    virtual std::shared_ptr<DetectorMap> clone() const override;

    SlitModel const& getSlitModel() const { return _slitModel; }
    OpticalModel const& getOpticalModel() const { return _opticalModel; }
    DetectorModel const& getDetectorModel() const { return _detectorModel; }

    // row -> wavelength
    Spline const& getWavelengthSpline(int fiberId) const {
        return *std::get<0>(_splines(fiberId, [this](int fiberId) { return makeSplines(fiberId); }));
    }
    Spline const& getRowSpline(int fiberId) const {
        return *std::get<1>(_splines(fiberId, [this](int fiberId) { return makeSplines(fiberId); }));
    }
    Spline const& getXDetectorSpline(int fiberId) const {
        return *std::get<2>(_splines(fiberId, [this](int fiberId) { return makeSplines(fiberId); }));
    }
    Spline const& getYDetectorSpline(int fiberId) const {
        return *std::get<3>(_splines(fiberId, [this](int fiberId) { return makeSplines(fiberId); }));
    }

    lsst::geom::Point2D findPointFull(int fiberId, double wavelength) const;

    bool isPersistable() const noexcept override { return true; }
    class Factory;

  protected:

    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    ///
    /// Implementation of findPoint, for subclasses to define.
    virtual lsst::geom::PointD findPointImpl(int fiberId, double wavelength) const override;

    /// Return the wavelength of a point on the detector, given a fiberId and row
    ///
    /// Implementation of findWavelength, for subclasses to define.
    virtual double findWavelengthImpl(int fiberId, double row) const override;

    /// Return the center of the fiber trace, given a fiberId and row
    ///
    /// Implementation of getXCenter, for subclasses to define.
    virtual double getXCenterImpl(int fiberId, double row) const override;

    /// Return the position of the fiber trace
    ///
    /// Implementation of getTracePosition, allowing subclasses to override.
    virtual std::pair<int, ndarray::Array<double, 1, 1>> getTracePositionImpl(
        int fiberId,
        int row,
        int halfWidth
    ) const override;

    SplineTuple makeSplines(int fiberId) const;

    /// Reset cached elements after setting slit offsets
    virtual void _resetSlitOffsets() override;

    std::string getPythonModule() const override { return "pfs.drp.stella"; }
    std::string getPersistenceName() const override { return "OpticalModelDetectorMap"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override;

  private:
    SlitModel _slitModel;
    OpticalModel _opticalModel;
    DetectorModel _detectorModel;
    std::pair<double, double> _wavelengthRange;  ///< (min, max) wavelength
    std::size_t _numKnots;  ///< number of knots to use for the per-fiber splines

    /// Per-fiber splines, behind a cache to avoid constructing them until needed.
    ///
    /// 0: row -> wavelength
    /// 1: wavelength -> row
    /// 2: row -> x on detector
    /// 3: row -> y on detector
    mutable lsst::cpputils::Cache<int, SplineTuple> _splines;
};


}}}

#endif // include guard
