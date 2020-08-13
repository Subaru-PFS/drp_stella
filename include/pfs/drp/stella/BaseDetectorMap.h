#if !defined(PFS_DRP_STELLA_BASEDETECTORMAP_H)
#define PFS_DRP_STELLA_BASEDETECTORMAP_H

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
class BaseDetectorMap : public lsst::afw::table::io::Persistable {
  public:
    using FiberIds = ndarray::Array<int, 1, 1>;
    using Array2D = ndarray::Array<float, 2, 1>;
    using Array1D = ndarray::Array<float, 1, 1>;
    using VisitInfo = lsst::afw::image::VisitInfo;

    /// Return the detector bounding box
    lsst::geom::Box2I getBBox() const { return _bbox; }

    /// Return the fiberIds
    FiberIds const& getFiberId() const { return _fiberId; }

    /// Return the number of fibers
    std::size_t getNumFibers() const { return _fiberId.size(); }

    /// Apply an offset to the slit position
    void applySlitOffset(float spatial, float spectral);

    //@{
    /// Return the slit offsets
    Array1D & getSpatialOffsets() { return _spatialOffsets; }
    Array1D const& getSpatialOffsets() const { return _spatialOffsets; }
    float getSpatialOffset(int fiberId) const { return _spatialOffsets[getFiberIndex(fiberId)]; }
    Array1D & getSpectralOffsets() { return _spectralOffsets; }
    Array1D const& getSpectralOffsets() const { return _spectralOffsets; }
    float getSpectralOffset(int fiberId) const { return _spectralOffsets[getFiberIndex(fiberId)]; }
    //@}

    //@{
    /// Set the slit positions
    void setSlitOffsets(Array1D const& spatial, Array1D const& spectral);
    void setSlitOffsets(int fiberId, float spatial, float spectral);
    //@}

    //@{
    /// Return the fiber centers
    Array2D getXCenter() const;
    virtual Array1D getXCenter(int fiberId) const = 0;
    virtual float getXCenter(int fiberId, float row) const = 0;
    //@}

    //@{
    /// Return the wavelength values for each row
    Array2D getWavelength() const;
    virtual Array1D getWavelength(int fiberId) const = 0;
    virtual float getWavelength(int fiberId, float row) const = 0;
    //@}

    /// Return the fiberId given a position on the detector
    virtual int findFiberId(lsst::geom::PointD const& point) const = 0;

    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    virtual lsst::geom::PointD findPoint(int fiberId, float wavelength) const = 0;

    /// Return the wavelength of a point on the detector, given a fiberId and row
    virtual float findWavelength(int fiberId, float row) const = 0;

    VisitInfo getVisitInfo() const { return _visitInfo; }
    void setVisitInfo(VisitInfo &visitInfo) { _visitInfo = visitInfo; };

    std::shared_ptr<lsst::daf::base::PropertySet> getMetadata() { return _metadata; }
    std::shared_ptr<lsst::daf::base::PropertySet const> getMetadata() const { return _metadata; }

    virtual ~BaseDetectorMap() {}
    BaseDetectorMap(BaseDetectorMap const&) = default;
    BaseDetectorMap(BaseDetectorMap &&) = default;
    BaseDetectorMap & operator=(BaseDetectorMap const&) = default;
    BaseDetectorMap & operator=(BaseDetectorMap &&) = default;

  protected:
    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param fiberId : fiber identifiers
    /// @param spatialOffsets : per-fiber offsets in the spatial dimension
    /// @param spectralOffsets : per-fiber offsets in the spectral dimension
    /// @param visitInfo : visit information
    /// @param metadata : FITS header
    BaseDetectorMap(
        lsst::geom::Box2I bbox,
        FiberIds const& fiberId,
        Array1D const& spatialOffsets=Array1D(),
        Array1D const& spectralOffsets=Array1D(),
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr
    );

    /// Return the index of a fiber, given its fiber ID
    std::size_t getFiberIndex(int fiberId) const { return _fiberMap.at(fiberId); }

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
