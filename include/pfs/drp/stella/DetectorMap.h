#if !defined(PFS_DRP_STELLA_DETECTORMAP_H)
#define PFS_DRP_STELLA_DETECTORMAP_H

#include <utility>
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

    /// Measure and apply slit offsets
    ///
    /// @param fiberId : Fiber identifiers for reference lines
    /// @param wavelength : Wavelength of reference lines (nm)
    /// @param x, y : Position of reference lines (pixels)
    /// @param xErr, yErr : Error in position of reference lines (pixels)
    virtual void measureSlitOffsets(
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& wavelength,
        ndarray::Array<double, 1, 1> const& x,
        ndarray::Array<double, 1, 1> const& y,
        ndarray::Array<double, 1, 1> const& xErr,
        ndarray::Array<double, 1, 1> const& yErr
    ) = 0;

    //@{
    /// Return the fiber centers
    Array2D getXCenter() const;
    virtual Array1D getXCenter(int fiberId) const = 0;
    virtual double getXCenter(int fiberId, double row) const = 0;
    //@}

    //@{
    /// Return the wavelength values for each row
    Array2D getWavelength() const;
    virtual Array1D getWavelength(int fiberId) const = 0;

    virtual double getWavelength(int fiberId, double row) const = 0;
    //@}

    /// Return the dispersion (nm/pixel) at the center
    double getDispersion(int fiberId) const;

    /// Return the fiberId given a position on the detector
    virtual int findFiberId(lsst::geom::PointD const& point) const = 0;

    //@{
    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    lsst::geom::PointD findPoint(int fiberId, double wavelength) const {
        return findPointImpl(fiberId, wavelength);
    }
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
    double findWavelength(int fiberId, double row) const {
        return findWavelengthImpl(fiberId, row);
    }
    ndarray::Array<double, 1, 1> findWavelength(
        int fiberId,
        ndarray::Array<double, 1, 1> const& row
    ) const;
    ndarray::Array<double, 1, 1> findWavelength(
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& row
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

    /// Reset cached elements after setting slit offsets
    virtual void _resetSlitOffsets() {}

  private:
    lsst::geom::Box2I _bbox;  ///< bounding box of detector
    FiberIds _fiberId;  ///< fiber identifiers (between 1 and c. 2400) present on this detector
    std::map<int, std::size_t> _fiberMap;  // Mapping fiberId -> fiber index

    Array1D _spatialOffsets;   ///< per-fiber offsets in the spatial dimension
    Array1D _spectralOffsets;   ///< per-fiber offsets in the spectral dimension

    lsst::afw::image::VisitInfo _visitInfo;
    std::shared_ptr<lsst::daf::base::PropertySet> _metadata;  // FITS header
};

}}}  // namespace pfs::drp::stella

#endif  // include guard
