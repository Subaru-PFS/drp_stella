#ifndef PFS_DRP_STELLA_LAYEREDDETECTORMAP_H
#define PFS_DRP_STELLA_LAYEREDDETECTORMAP_H

#include "ndarray_fwd.h"

#include "lsst/geom/AffineTransform.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/cpputils/Cache.h"

#include "pfs/drp/stella/SplinedDetectorMap.h"
#include "pfs/drp/stella/ModelBasedDetectorMap.h"
#include "pfs/drp/stella/Distortion.h"
#include "pfs/drp/stella/PolynomialDistortion.h"


namespace pfs {
namespace drp {
namespace stella {


/// DetectorMap modelled with layers
///
/// The layers are:
/// 1. Slit offsets: fiberId,wavelength --> fiberId',wavelength'. The slit offsets are applied to the input
///    fiberId,wavelength. Note that the fiberId' is a floating-point value. The slit offsets are specified
///    in pixels but used to modify the input fiberId,wavelength using the mean fiber pitch and wavelength
///    dispersion (the scaling doesn't need to be very precise).
/// 2. Optical model: fiberId',wavelength' --> xJEG,yJEG. This transformation is provided through JEG's
///    optical model (the "base" detectorMap). I can interpolate between the JEG optical model splines to get
///    the xJEG,yJEG for any fiberId',wavelength'.
/// 3. Distortion models: xJEG,yJEG --> xDist,yDist. This transformation is provided by a list of
///    distortion models.
/// 4. Detector: xDist,yDist --> xDet,yDet. This is a linear transformation, accounting for the chip gap
///    and slight rotation differences between the chips. For the NIR detectors, this is a no-op.
class LayeredDetectorMap : public ModelBasedDetectorMap {
  public:
    using DistortionList = std::vector<std::shared_ptr<Distortion>>;

    //@{
    /// Ctor
    ///
    /// @param bbox : bounding box of the detector
    /// @param spatialOffsets : slit offsets in the spatial dimension
    /// @param spectralOffsets : slit offsets in the spectral dimension
    /// @param base : detectorMap from optical model
    /// @param distortions : distortions applied
    /// @param dividedDetector : whether the detector is divided into two CCDs
    /// @param rightCcd : transformation for CCD with larger fiberId
    /// @param visitInfo : visit information
    /// @param metadata : FITS header
    /// @param sampling : period to sample distortion field for cached spline (pixels)
    /// @param precision : sampling for precision at spline ends (pixels)
    LayeredDetectorMap(
        lsst::geom::Box2I const& bbox,
        ndarray::Array<double, 1, 1> const& spatialOffsets,
        ndarray::Array<double, 1, 1> const& spectralOffsets,
        SplinedDetectorMap const& base,
        DistortionList const& distortions,
        bool dividedDetector,
        lsst::geom::AffineTransform const& rightCcd=lsst::geom::AffineTransform(),
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr,
        float sampling=10.0,
        float precision=1.0e-3
    );
    LayeredDetectorMap(
        lsst::geom::Box2I const& bbox,
        ndarray::Array<double, 1, 1> const& spatialOffsets,
        ndarray::Array<double, 1, 1> const& spectralOffsets,
        SplinedDetectorMap const& base,
        DistortionList const& distortions,
        bool dividedDetector,
        ndarray::Array<double, 1, 1> const& rightCcd=ndarray::Array<double, 1, 1>(),
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr,
        float sampling=10.0,
        float precision=1.0e-3
    );

    //@}

    virtual ~LayeredDetectorMap() {}
    LayeredDetectorMap(LayeredDetectorMap const&) = default;
    LayeredDetectorMap(LayeredDetectorMap &&) = default;
    LayeredDetectorMap & operator=(LayeredDetectorMap const&) = default;
    LayeredDetectorMap & operator=(LayeredDetectorMap &&) = default;

    virtual std::shared_ptr<DetectorMap> clone() const override;

    SplinedDetectorMap const& getBase() const { return _base; }
    DistortionList const& getDistortions() const { return _distortions; }
    bool getDividedDetector() const { return _dividedDetector; }
    lsst::geom::AffineTransform const& getRightCcd() const { return _rightCcd; }

    ndarray::Array<double, 1, 1> getRightCcdParameters() const;

    bool isRightCcd(float x) const;

    bool isPersistable() const noexcept override { return true; }

    class Factory;

  protected:
    /// Evaluate the model
    ///
    /// This handles the first three layers.
    virtual lsst::geom::PointD evalModel(int fiberId, double wavelength) const override;

    /// Transform the point to the detector frame
    ///
    /// This handles the final layer (layer 4).
    lsst::geom::Point2D toDetectorFrame(lsst::geom::Point2D const& point) const;

    /// Transform the point from the detector frame to the model frame
    ///
    /// This is the inverse of the layer 4 transformation.
    lsst::geom::Point2D fromDetectorFrame(lsst::geom::Point2D const& point) const;

    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    virtual lsst::geom::PointD findPointImpl(int fiberId, double wavelength) const override;

    /// Find the position of a fiber on the detector
    ///
    /// Allows the position to be off the detector bbox (contrary to findPointImpl).
    lsst::geom::Point2D findPointPermissive(int fiberId, double wavelength) const;

    /// Return the fiber center
    virtual double getXCenterImpl(int fiberId, double row) const override;

    /// Return the wavelength of a point on the detector, given a fiberId and row
    virtual double findWavelengthImpl(int fiberId, double row) const override;

    /// Return the position of the fiber trace
    virtual std::pair<int, ndarray::Array<double, 1, 1>> getTracePositionImpl(
        int fiberId, int row, int halfWidth
    ) const override;

    std::string getPythonModule() const override { return "pfs.drp.stella"; }
    std::string getPersistenceName() const override { return "LayeredDetectorMap"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override;

    /// Get fiber indices for interpolation
    std::pair<std::pair<int, double>, std::pair<int, double>> _interpolateFiberIndices(double fiberId) const;

  private:
    double _fiberPitch;  ///< Average distance between fibers (pixels)
    double _wavelengthDispersion;  ///< Average dispersion (nm/pixel)
    SplinedDetectorMap _base;
    DistortionList _distortions;
    bool _dividedDetector;
    lsst::geom::AffineTransform _rightCcd;
    int _lowMiddleFiber, _highMiddleFiber;  ///< Low and high middle fiberIds
    std::size_t _lowMiddleIndex, _highMiddleIndex;  ///< Fiber index to use for interpolating near middle
    float _middleFiber;  ///< Middle fiberId
    bool _increasing;  ///< x position increases with increasing fiberId?
    lsst::geom::Box2D _precisionBBox;  ///< Bounding box allowing for precision

    double _xOffset, _xScale, _yOffset, _yScale;  ///< Normalization for right CCD transformation

    mutable lsst::cpputils::Cache<std::ptrdiff_t, std::size_t> _fiberIndexCache;
};


}}}

#endif // include guard
