#ifndef PFS_DRP_STELLA_SPLINEDDETECTORMAP_H
#define PFS_DRP_STELLA_SPLINEDDETECTORMAP_H

#include <vector>

#include "lsst/afw/table/io/Persistable.h"

#include "pfs/drp/stella/spline.h"
#include "pfs/drp/stella/DetectorMap.h"


namespace pfs { namespace drp { namespace stella {
/// DetectorMap implemented with splines.
///
/// The slitOffsets are the different zeropoints of the wavelengths of each
/// fiber (in floating-point pixels), due to imperfect manufacture of the
/// slithead. The values are *subtracted* from the CCD coordinates when
/// calculating the wavelength for a pixel -- i.e. we assume that the spots are
/// deflected up by slitOffset
class SplinedDetectorMap : public DetectorMap {
  public:
    using Spline = math::Spline<double>;

    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param fiberId : fiber identifiers
    /// @param xCenterKnots : xCenter spline knots for each fiber
    /// @param xCenterValues : xCenter spline values for each fiber
    /// @param wavelengthKnots : wavelength spline knots for each fiber
    /// @param wavelengthValues : wavelength spline values for each fiber
    /// @param spatialOffsets : per-fiber offsets in the spatial dimension
    /// @param spectralOffsets : per-fiber offsets in the spectral dimension
    /// @param visitInfo : visit information
    /// @param metadata : FITS header
    SplinedDetectorMap(
        lsst::geom::Box2I bbox,
        FiberIds const& fiberId,
        std::vector<ndarray::Array<double, 1, 1>> const& xCenterKnots,
        std::vector<ndarray::Array<double, 1, 1>> const& xCenterValues,
        std::vector<ndarray::Array<double, 1, 1>> const& wavelengthKnots,
        std::vector<ndarray::Array<double, 1, 1>> const& wavelengthValues,
        Array1D const& spatialOffsets=Array1D(),
        Array1D const& spectralOffsets=Array1D(),
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr
    );

    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param fiberId : fiber identifiers
    /// @param center : xCenter splines for each fiber
    /// @param wavelength : wavelength splines for each fiber
    /// @param spatialOffsets : per-fiber offsets in the spatial dimension
    /// @param spectralOffsets : per-fiber offsets in the spectral dimension
    /// @param visitInfo : visit information
    /// @param metadata : FITS header
    SplinedDetectorMap(
        lsst::geom::Box2I bbox,
        FiberIds const& fiberId,
        std::vector<Spline> const& xCenter,
        std::vector<Spline> const& wavelength,
        Array1D const& spatialOffsets=Array1D(),
        Array1D const& spectralOffsets=Array1D(),
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr
    );

    virtual ~SplinedDetectorMap() {}
    SplinedDetectorMap(SplinedDetectorMap const&) = default;
    SplinedDetectorMap(SplinedDetectorMap &&) = default;
    SplinedDetectorMap & operator=(SplinedDetectorMap const&) = default;
    SplinedDetectorMap & operator=(SplinedDetectorMap &&) = default;

    virtual std::shared_ptr<DetectorMap> clone() const override;

    Spline const& getXCenterSpline(int fiberId) const;
    Spline const& getWavelengthSpline(int fiberId) const;

    /// Set the fiber center spline
    void setXCenter(
        int fiberId,
        ndarray::Array<double, 1, 1> const& knots,
        ndarray::Array<double, 1, 1> const& xCenter
    );

    /// Set the wavelength spline
    void setWavelength(
        int fiberId,
        ndarray::Array<double, 1, 1> const& knots,
        ndarray::Array<double, 1, 1> const& wavelength
    );

    bool isPersistable() const noexcept override { return true; }

    class Factory;

  protected:
    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    virtual lsst::geom::PointD findPointImpl(int fiberId, double wavelength) const override;

    /// Return the wavelength of a point on the detector, given a fiberId and row
    virtual double findWavelengthImpl(int fiberId, double row) const override;

    /// Return the fiber center
    virtual double getXCenterImpl(int fiberId, double row) const override;

    std::string getPersistenceName() const override { return "SplinedDetectorMap"; }
    std::string getPythonModule() const override { return "pfs.drp.stella"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override;

  private:
    std::vector<Spline> _xCenter;  ///< convert y pixel value to trace position
    std::vector<Spline> _wavelength;  ///< convert a y pixel value to wavelength
    std::vector<Spline> _row;  ///< convert wavelength to y pixel value

    void _set_xToFiberId();
    ndarray::Array<int, 1, 1> _xToFiberId;  ///< An array that gives the fiberId half way up the chip
};


}}}  // namespace pfs::drp::stella

#endif  // include guard
