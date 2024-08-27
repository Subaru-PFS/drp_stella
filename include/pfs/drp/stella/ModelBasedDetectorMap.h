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
/// (evalModel), and this takes care of the DetectorMap interfaces.
///
/// We use some splines to cache the row->wavelength and row->xCenter
/// transforms.
class ModelBasedDetectorMap : public DetectorMap {
  public:
    using SplineCoeffT = double;
    using Spline = math::Spline<SplineCoeffT>;

    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param wavelengthCenter : Central wavelength (nm)
    /// @param dispersion : Approximate wavelength dispersion (nm/pixel)
    /// @param sampling : Spline sampling interval (pixel)
    /// @param fiberId : fiber IDs
    /// @param spatialOffsets : slit offsets in the spatial dimension
    /// @param spectralOffsets : slit offsets in the spectral dimension
    /// @param visitInfo : visit information
    /// @param metadata : FITS header
    /// @param extrapolation : extrapolation mode for splines
    /// @param precision : sampling for precision at spline ends (pixels)
    ModelBasedDetectorMap(
        lsst::geom::Box2I const& bbox,
        double wavelengthCenter,
        double dispersion,
        double sampling,
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& spatialOffsets,
        ndarray::Array<double, 1, 1> const& spectralOffsets,
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr,
        Spline::ExtrapolationTypes extrapolation=Spline::EXTRAPOLATE_ALL,
        double precision=1.0e-3
    );

    virtual ~ModelBasedDetectorMap() = default;
    ModelBasedDetectorMap(ModelBasedDetectorMap const&) = default;
    ModelBasedDetectorMap(ModelBasedDetectorMap &&) = default;
    ModelBasedDetectorMap & operator=(ModelBasedDetectorMap const&) = default;
    ModelBasedDetectorMap & operator=(ModelBasedDetectorMap &&) = default;

  protected:

    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    virtual lsst::geom::PointD findPointImpl(int fiberId, double wavelength) const override;

    /// Return the wavelength of a point on the detector, given a fiberId and row
    virtual double findWavelengthImpl(int fiberId, double row) const override;

    /// Return the fiber center
    virtual double getXCenterImpl(int fiberId, double row) const override;

    /// Reset cached elements after setting slit offsets
    virtual void _resetSlitOffsets() override;

    /// Evaluate the model
    ///
    /// The model evaluations will be cached and splined.
    ///
    /// @param fiberId : fiber identifier
    /// @param wavelength : wavelength (nm)
    /// @return position on the detector
    virtual lsst::geom::PointD evalModel(int fiberId, double wavelength) const = 0;

    //@{
    /// Accessors
    double getWavelengthCenter() const { return _wavelengthCenter; }
    double getDispersion() const { return _dispersion; }
    double getSampling() const { return _sampling; }
    double getPrecision() const { return _precision; }
    //@}

  private:
    // Splines are: row->wavelength, row->xCenter, wavelength->row
    using SplineGroup = std::tuple<Spline, Spline, Spline>;
    using SplineCache = lsst::utils::Cache<int, std::shared_ptr<SplineGroup>>;

    /// Construct wavelength and xCenter splines for a fiber
    SplineGroup _makeSplines(int fiberId) const;

    /// Retrieve wavelength and xCenter splines for a fiber from the cache
    ///
    /// Generates them if they are not already in the cache.
    SplineGroup const& _getSplines(int fiberId) const {
        return *_splines(
            fiberId,
            [this](int fiberId) { return std::make_shared<SplineGroup>(_makeSplines(fiberId)); }
        );
    }

    /// Retrieve wavelength spline for a fiber from the cache
    ///
    /// Generates splines for this fiber if they are not already in the cache.
    Spline const& _getWavelengthSpline(int fiberId) const { return std::get<0>(_getSplines(fiberId)); }

    /// Retrieve xCenter spline for a fiber from the cache
    ///
    /// Generates splines for this fiber if they are not already in the cache.
    Spline const& _getXCenterSpline(int fiberId) const { return std::get<1>(_getSplines(fiberId)); }

    /// Retrieve row spline for a fiber from the cache
    ///
    /// Generates splines for this fiber if they are not already in the cache.
    Spline const& _getRowSpline(int fiberId) const { return std::get<2>(_getSplines(fiberId)); }

    /// Clear splines cache
    void _clearSplines() const { _splines.flush(); }

    double _wavelengthCenter;
    double _dispersion;
    double _sampling;
    double _precision;
    mutable SplineCache _splines;
    Spline::ExtrapolationTypes _extrapolation;
};


}}}  // namespace pfs::drp::stella

#endif  // include guard
