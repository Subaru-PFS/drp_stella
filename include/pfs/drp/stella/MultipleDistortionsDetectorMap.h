#ifndef PFS_DRP_STELLA_MULTIPLEDISTORTIONSDETECTORMAP_H
#define PFS_DRP_STELLA_MULTIPLEDISTORTIONSDETECTORMAP_H

#include "ndarray_fwd.h"

#include "lsst/afw/table/io/Persistable.h"

#include "pfs/drp/stella/SplinedDetectorMap.h"
#include "pfs/drp/stella/ModelBasedDetectorMap.h"
#include "pfs/drp/stella/Distortion.h"


namespace pfs {
namespace drp {
namespace stella {


/// DetectorMap referenced to a base detectorMap and a series of distortions
///
/// Base detectorMap is a SplinedDetectorMap (in theory, we could use any kind of detectorMap,
/// but that would make it difficult to describe the datamodel of persisted data).
/// The Distortions map x,y --> dx,dy.
class MultipleDistortionsDetectorMap : public ModelBasedDetectorMap {
  public:
    using DistortionList = std::vector<std::shared_ptr<Distortion>>;

    /// Ctor
    ///
    /// @param base : foundational detectorMap
    /// @param distortions : distortions applied to base
    /// @param visitInfo : visit information
    /// @param metadata : FITS header
    /// @param sampling : period to sample distortion field for cached spline (pixels)
    MultipleDistortionsDetectorMap(
        SplinedDetectorMap const& base,
        DistortionList const& distortions,
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr,
        float sampling=50.0
    );

    virtual ~MultipleDistortionsDetectorMap() {}
    MultipleDistortionsDetectorMap(MultipleDistortionsDetectorMap const&) = default;
    MultipleDistortionsDetectorMap(MultipleDistortionsDetectorMap &&) = default;
    MultipleDistortionsDetectorMap & operator=(MultipleDistortionsDetectorMap const&) = default;
    MultipleDistortionsDetectorMap & operator=(MultipleDistortionsDetectorMap &&) = default;

    virtual std::shared_ptr<DetectorMap> clone() const override;

    SplinedDetectorMap const& getBase() const { return _base; }
    DistortionList const& getDistortions() const { return _distortions; }

    bool isPersistable() const noexcept override { return true; }

    class Factory;

  protected:
    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    virtual lsst::geom::PointD evalModel(int fiberId, double wavelength) const override;

    std::string getPythonModule() const override { return "pfs.drp.stella"; }
    std::string getPersistenceName() const override { return "MultipleDistortionsDetectorMap"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override;

    /// Reset cached elements after setting slit offsets
    virtual void _resetSlitOffsets() override;

  private:
    SplinedDetectorMap _base;
    DistortionList _distortions;
};


}}}

#endif // include guard
