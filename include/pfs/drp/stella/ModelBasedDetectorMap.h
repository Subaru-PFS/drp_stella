#if !defined(PFS_DRP_STELLA_MODELBASEDDETECTORMAP_H)
#define PFS_DRP_STELLA_MODELBASEDDETECTORMAP_H

#include "ndarray_fwd.h"

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

    virtual ~ModelBasedDetectorMap() {}
    ModelBasedDetectorMap(ModelBasedDetectorMap const&) = default;
    ModelBasedDetectorMap(ModelBasedDetectorMap &&) = default;
    ModelBasedDetectorMap & operator=(ModelBasedDetectorMap const&) = default;
    ModelBasedDetectorMap & operator=(ModelBasedDetectorMap &&) = default;

    /// Return the fiberId given a position on the detector
    virtual int findFiberId(lsst::geom::PointD const& point) const override;

  protected:
    /// Return the wavelength of a point on the detector, given a fiberId and row
    virtual double findWavelengthImpl(int fiberId, double row) const override;

    /// Reset cached elements after setting slit offsets
    virtual void _resetSlitOffsets() override;

    /// Ensure that splines are initialized
    void _ensureSplinesInitialized() const;

  private:
    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    ///
    /// Implementation of findPoint, for subclasses to define.
    virtual lsst::geom::PointD findPointImpl(int fiberId, double wavelength) const override = 0;

    /// Return the fiber center
    virtual double getXCenterImpl(int fiberId, double row) const override;

    /// Set splines
    void _setSplines() const;

    double _wavelengthCenter;
    double _wavelengthSampling;
    using SplineCoeffT = double;
    using Spline = math::Spline<SplineCoeffT>;
    mutable std::vector<Spline> _rowToWavelength;
    mutable std::vector<Spline> _rowToXCenter;
    mutable bool _splinesInitialized;
};


}}}  // namespace pfs::drp::stella

#endif  // include guard
