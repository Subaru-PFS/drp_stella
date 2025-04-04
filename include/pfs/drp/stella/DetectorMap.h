#if !defined(PFS_DRP_STELLA_DETECTORMAP_H)
#define PFS_DRP_STELLA_DETECTORMAP_H

#include <map>
#include <utility>
#include <unordered_map>
#include "ndarray_fwd.h"

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"
#include "lsst/daf/base/PropertySet.h"
#include "lsst/afw/image/VisitInfo.h"
#include "lsst/afw/table/io/Persistable.h"

namespace pfs { namespace drp { namespace stella {

/// Describe the geometry of the focal plane.
///
/// This pure-virtual base class provides a mapping between fiberId,wavelength
/// and x,y on the detector.
class DetectorMap : public lsst::afw::table::io::Persistable {
  public:
    using FiberIds = ndarray::Array<int, 1, 1>;
    using Array2D = ndarray::Array<double, 2, 1>;
    using Array1D = ndarray::Array<double, 1, 1>;
    using VisitInfo = lsst::afw::image::VisitInfo;

    /// Polymorphic copy
    virtual std::shared_ptr<DetectorMap> clone() const = 0;

    /// Return the detector bounding box
    lsst::geom::Box2I getBBox() const { return _bbox; }

    /// Return the fiberIds
    FiberIds const& getFiberId() const { return _fiberId; }

    /// Return the number of fibers
    std::size_t getNumFibers() const { return _fiberId.size(); }

    /// Does this include a particular fiberId?
    bool contains(int fiberId) const { return _fiberMap.find(fiberId) != _fiberMap.end(); }

    /// Apply an offset to the slit position
    void applySlitOffset(double spatial, double spectral);

    //@{
    /// Return the slit offsets
    Array1D & getSpatialOffsets() { return _spatialOffsets; }
    Array1D const& getSpatialOffsets() const { return _spatialOffsets; }
    double getSpatialOffset(int fiberId) const { return _spatialOffsets[getFiberIndex(fiberId)]; }
    Array1D & getSpectralOffsets() { return _spectralOffsets; }
    Array1D const& getSpectralOffsets() const { return _spectralOffsets; }
    double getSpectralOffset(int fiberId) const { return _spectralOffsets[getFiberIndex(fiberId)]; }
    //@}

    //@{
    /// Set the slit positions
    void setSlitOffsets(Array1D const& spatial, Array1D const& spectral);
    void setSlitOffsets(int fiberId, double spatial, double spectral);
    //@}

    //@{
    /// Return the fiber centers
    Array2D getXCenter() const;
    Array1D getXCenter(int fiberId) const;
    double getXCenter(int fiberId, double row) const { return getXCenterImpl(fiberId, row); }
    Array1D getXCenter(int fiberId, ndarray::Array<double, 1, 1> const& row) const;
    Array1D getXCenter(
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& row
    ) const;
    //@}

    //@{
    /// Return the wavelength values
    Array2D getWavelength() const;
    Array1D getWavelength(int fiberId) const;
    double getWavelength(int fiberId, double row) const { return findWavelengthImpl(fiberId, row); }
    //@}

    //@{
    /// Return the dispersion (nm/pixel) at the center of a fiber
    double getDispersionAtCenter(int fiberId) const;
    double getDispersionAtCenter() const {
        return getDispersionAtCenter(_fiberId[getNumFibers()/2]);
    }
    //@}

    //@{
    /// Return the dispersion (nm/pixel) at a particular wavelength
    ///
    /// @param fiberId : Fiber identifier.
    /// @param wavelength : Wavelength of interest (nm).
    /// @param dWavelength : Change in wavelength to use in calculation (nm).
    double getDispersion(int fiberId, double wavelength, double dWavelength=0.1) const;
    Array1D getDispersion(FiberIds const& fiberId, Array1D const& wavelength, double dWavelength=0.1) const;
    //@}

    //@{
    /// Return the (inverse) slope (dx/dy) of the fiber trace
    ///
    /// @param fiberId : Fiber identifier.
    /// @param row : Row on the detector (pixels).
    /// @param calculate : Calculate value for this point? Allows excluding points from calculation (e.g,.
    ///     because it's an actual line rather than a trace measurement).
    double getSlope(int fiberId, double row) const;
    Array1D getSlope(
        FiberIds const& fiberId,
        Array1D const& row,
        ndarray::Array<bool, 1, 1> const& calculate=ndarray::Array<bool, 1, 1>()
    ) const;
    //@}

    //@{
    /// Return the fiberId given a position on the detector
    int findFiberId(double x, double y) const { return findFiberId(lsst::geom::Point2D(x, y)); }
    int findFiberId(lsst::geom::PointD const& point) const;
    //@}

    //@{
    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    lsst::geom::PointD findPoint(
        int fiberId,
        double wavelength,
        bool throwOnError=false,
        bool enforceLimits=true
    ) const;
    ndarray::Array<double, 2, 1> findPoint(
        int fiberId,
        ndarray::Array<double, 1, 1> const& wavelength
    ) const;
    ndarray::Array<double, 2, 1> findPoint(
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& wavelength
    ) const;
    //@}

    //@{
    /// Return the wavelength of a point on the detector, given a fiberId and row
    double findWavelength(int fiberId, double row, bool throwOnError=false) const;
    ndarray::Array<double, 1, 1> findWavelength(
        int fiberId,
        ndarray::Array<double, 1, 1> const& row
    ) const;
    ndarray::Array<double, 1, 1> findWavelength(
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& row
    ) const;
    //@}

    //@{
    /// Return the position of the fiber trace
    ///
    /// Returns the minimum x index and an array containing the distance from
    /// the center of the trace for each pixel. Note that because of chip gaps,
    /// this may be shorter than the range implied by the halfWidth, and the
    /// difference between successive values may not be constant.
    ///
    /// @param fiberId : Fiber identifier.
    /// @param row : Row on the detector (pixels).
    /// @param halfWidth : Half-width of the trace (pixels).
    /// @returns xMin, dxCenter
    std::pair<int, ndarray::Array<double, 1, 1>> getTracePosition(
        int fiberId,
        int row,
        int halfWidth
    ) const {
        return getTracePositionImpl(fiberId, row, halfWidth);
    }
    std::vector<std::pair<int, ndarray::Array<double, 1, 1>>> getTracePosition(
        int fiberId,
        int halfWidth
    ) const;
    //@}

    VisitInfo getVisitInfo() const { return _visitInfo; }
    void setVisitInfo(VisitInfo &visitInfo) { _visitInfo = visitInfo; };

    std::shared_ptr<lsst::daf::base::PropertySet> getMetadata() { return _metadata; }
    std::shared_ptr<lsst::daf::base::PropertySet const> getMetadata() const { return _metadata; }

    virtual ~DetectorMap() {}
    DetectorMap(DetectorMap const&) = default;
    DetectorMap(DetectorMap &&) = default;
    DetectorMap & operator=(DetectorMap const&) = default;
    DetectorMap & operator=(DetectorMap &&) = default;

  protected:
    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param fiberId : fiber identifiers
    /// @param spatialOffsets : per-fiber offsets in the spatial dimension
    /// @param spectralOffsets : per-fiber offsets in the spectral dimension
    /// @param visitInfo : visit information
    /// @param metadata : FITS header
    DetectorMap(
        lsst::geom::Box2I bbox,
        FiberIds const& fiberId,
        Array1D const& spatialOffsets=Array1D(),
        Array1D const& spectralOffsets=Array1D(),
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr
    );

    /// Return the index of a fiber, given its fiber ID
    std::size_t getFiberIndex(int fiberId) const { return _fiberMap.at(fiberId); }

    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    ///
    /// Implementation of findPoint, for subclasses to define.
    virtual lsst::geom::PointD findPointImpl(int fiberId, double wavelength) const = 0;

    /// Return the wavelength of a point on the detector, given a fiberId and row
    ///
    /// Implementation of findWavelength, for subclasses to define.
    virtual double findWavelengthImpl(int fiberId, double row) const = 0;

    /// Return the center of the fiber trace, given a fiberId and row
    ///
    /// Implementation of getXCenter, for subclasses to define.
    virtual double getXCenterImpl(int fiberId, double row) const = 0;

    /// Return the position of the fiber trace
    ///
    /// Implementation of getTracePosition, allowing subclasses to override.
    virtual std::pair<int, ndarray::Array<double, 1, 1>> getTracePositionImpl(
        int fiberId,
        int row,
        int halfWidth
    ) const;

    /// Reset cached elements after setting slit offsets
    virtual void _resetSlitOffsets() {}

  private:
    lsst::geom::Box2I _bbox;  ///< bounding box of detector
    FiberIds _fiberId;  ///< fiber identifiers (between 1 and c. 2400) present on this detector
    std::unordered_map<int, std::size_t> _fiberMap;  // Mapping fiberId -> fiber index

    Array1D _spatialOffsets;   ///< per-fiber offsets in the spatial dimension
    Array1D _spectralOffsets;   ///< per-fiber offsets in the spectral dimension

    lsst::afw::image::VisitInfo _visitInfo;
    std::shared_ptr<lsst::daf::base::PropertySet> _metadata;  // FITS header
};

}}}  // namespace pfs::drp::stella

#endif  // include guard
