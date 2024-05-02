#if !defined(PFS_DRP_STELLA_MODELBASEDDETECTORMAP_H)
#define PFS_DRP_STELLA_MODELBASEDDETECTORMAP_H

#include "ndarray_fwd.h"
#include "lsst/cpputils/Cache.h"

#include "pfs/drp/stella/spline.h"
#include "pfs/drp/stella/DetectorMap.h"


namespace pfs { namespace drp { namespace stella {


/// DetectorMap implemented with a global model
///
/// Subclasses implement a model mapping fiberId,wavelength to x,y
/// (findPointImpl), and this takes care of the DetectorMap interfaces.
///
/// We use some splines to cache the row->wavelength and row->xCenter
/// transforms.
class ModelBasedDetectorMap : public DetectorMap {
  public:

    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param wavelengthCenter : Central wavelength (nm)
    /// @param wavelengthSampling : Wavelength sampling interval (nm)
    /// @param spatialOffsets : slit offsets in the spatial dimension
    /// @param spectralOffsets : slit offsets in the spectral dimension
    /// @param visitInfo : visit information
    /// @param metadata : FITS header
    ModelBasedDetectorMap(
        lsst::geom::Box2I const& bbox,
        double wavelengthCenter,
        double wavelengthSampling,
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& spatialOffsets,
        ndarray::Array<double, 1, 1> const& spectralOffsets,
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr
    );

    virtual ~ModelBasedDetectorMap() = default;
    ModelBasedDetectorMap(ModelBasedDetectorMap const&) = default;
    ModelBasedDetectorMap(ModelBasedDetectorMap &&) = default;
    ModelBasedDetectorMap & operator=(ModelBasedDetectorMap const&) = default;
    ModelBasedDetectorMap & operator=(ModelBasedDetectorMap &&) = default;

  protected:
    /// Return the wavelength of a point on the detector, given a fiberId and row
    virtual double findWavelengthImpl(int fiberId, double row) const override {
        return _getWavelengthSpline(fiberId)(row);
    }

    /// Reset cached elements after setting slit offsets
    virtual void _resetSlitOffsets() override;

  private:
    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    ///
    /// Implementation of findPoint, for subclasses to define.
    virtual lsst::geom::PointD findPointImpl(int fiberId, double wavelength) const override = 0;

    /// Return the fiber center
    virtual double getXCenterImpl(int fiberId, double row) const override {
        return _getXCenterSpline(fiberId)(row);
    }

    using SplineCoeffT = double;
    using Spline = math::Spline<SplineCoeffT>;
    using SplinePair = std::pair<Spline, Spline>;
    using SplineCache = lsst::utils::Cache<int, std::shared_ptr<SplinePair>>;

    /// Construct wavelength and xCenter splines for a fiber
    SplinePair _makeSplines(int fiberId) const;

    /// Retrieve wavelength and xCenter splines for a fiber from the cache
    ///
    /// Generates them if they are not already in the cache.
    SplinePair const& _getSplines(int fiberId) const {
        return *_splines(
            fiberId,
            [this](int fiberId) { return std::make_shared<SplinePair>(_makeSplines(fiberId)); }
        );
    }

    /// Retrieve wavelength spline for a fiber from the cache
    ///
    /// Generates it (and the xCenter spline) if they are not already in the
    /// cache.
    Spline const& _getWavelengthSpline(int fiberId) const { return _getSplines(fiberId).first; }

    /// Retrieve xCenter spline for a fiber from the cache
    ///
    /// Generates it (and the wavelength spline) if they are not already in the
    /// cache.
    Spline const& _getXCenterSpline(int fiberId) const { return _getSplines(fiberId).second; }

    /// Clear splines cache
    void _clearSplines() const { _splines.flush(); }

    double _wavelengthCenter;
    double _wavelengthSampling;
    mutable SplineCache _splines;
};


}}}  // namespace pfs::drp::stella

#endif  // include guard
