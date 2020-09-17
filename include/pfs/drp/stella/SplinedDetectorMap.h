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
    using Spline = math::Spline<float>;

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
        std::vector<ndarray::Array<float, 1, 1>> const& xCenterKnots,
        std::vector<ndarray::Array<float, 1, 1>> const& xCenterValues,
        std::vector<ndarray::Array<float, 1, 1>> const& wavelengthKnots,
        std::vector<ndarray::Array<float, 1, 1>> const& wavelengthValues,
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

    //@{
    /// Return the fiber centers
    virtual Array1D getXCenter(int fiberId) const override;
    virtual float getXCenter(int fiberId, float row) const override;
    //@}

    //@{
    /// Return the wavelength values for each row
    virtual Array1D getWavelength(int fiberId) const override;
    virtual float getWavelength(int fiberId, float row) const override;
    //@}

    /// Return the fiberId given a position on the detector
    virtual int findFiberId(lsst::geom::PointD const& point) const override;

    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    virtual lsst::geom::PointD findPoint(int fiberId, float wavelength) const override;

    /// Return the wavelength of a point on the detector, given a fiberId and row
    virtual float findWavelength(int fiberId, float row) const override;

    math::Spline<float> const& getXCenterSpline(int fiberId) const;
    math::Spline<float> const& getWavelengthSpline(int fiberId) const;

    /// Set the fiber center spline
    void setXCenter(int fiberId, Array1D const& knots, Array1D const& xCenter);

    /// Set the wavelength spline
    void setWavelength(int fiberId, Array1D const& knots, Array1D const& wavelength);

    bool isPersistable() const noexcept { return true; }

    class Factory;

  protected:
    std::string getPersistenceName() const { return "SplinedDetectorMap"; }
    std::string getPythonModule() const { return "pfs.drp.stella"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const;

  private:
    std::vector<Spline> _xCenter;  ///< convert y pixel value to trace position
    std::vector<Spline> _wavelength;  ///< convert a y pixel value to wavelength

    void _set_xToFiberId();
    ndarray::Array<int, 1, 1> _xToFiberId;  ///< An array that gives the fiberId half way up the chip
};


}}}  // namespace pfs::drp::stella

#endif  // include guard
