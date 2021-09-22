#ifndef PFS_DRP_STELLA_DISTORTIONBASEDDETECTORMAP_H
#define PFS_DRP_STELLA_DISTORTIONBASEDDETECTORMAP_H

#include "ndarray_fwd.h"

#include "lsst/afw/table/io/Persistable.h"

#include "pfs/drp/stella/SplinedDetectorMap.h"
#include "pfs/drp/stella/ModelBasedDetectorMap.h"


namespace pfs {
namespace drp {
namespace stella {


/// DetectorMap referenced to a base detectorMap and a distortion
///
/// Base detectorMap is a SplinedDetectorMap (in theory, we could use any kind of detectorMap,
/// but that would make it difficult to describe the datamodel of persisted data).
/// The Distortion maps x,y --> dx,dy.
template <typename DistortionT>
class DistortionBasedDetectorMap : public ModelBasedDetectorMap {
  public:
    using Distortion = DistortionT;

    /// Ctor
    ///
    /// @param base : foundational detectorMap
    /// @param distortion : distortion applied to base
    /// @param visitInfo : visit information
    /// @param metadata : FITS header
    /// @param samplingFactor : period to sample distortion field for cached spline (pixels)
    DistortionBasedDetectorMap(
        SplinedDetectorMap const& base,
        Distortion const& distortion,
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr,
        float samplingFactor=50.0
    );

    virtual ~DistortionBasedDetectorMap() {}
    DistortionBasedDetectorMap(DistortionBasedDetectorMap const&) = default;
    DistortionBasedDetectorMap(DistortionBasedDetectorMap &&) = default;
    DistortionBasedDetectorMap & operator=(DistortionBasedDetectorMap const&) = default;
    DistortionBasedDetectorMap & operator=(DistortionBasedDetectorMap &&) = default;

    virtual std::shared_ptr<DetectorMap> clone() const override;

    SplinedDetectorMap const& getBase() const { return _base; }
    Distortion const& getDistortion() const { return _distortion; }

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
    ) override;

    bool isPersistable() const noexcept override { return true; }

  protected:
    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    virtual lsst::geom::PointD findPointImpl(int fiberId, double wavelength) const override;

    std::string getPythonModule() const override { return "pfs.drp.stella"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override;

    template <typename Derived>
    static std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    );

    /// Reset cached elements after setting slit offsets
    virtual void _resetSlitOffsets() override;

  private:
    SplinedDetectorMap _base;
    Distortion _distortion;
};


}}}

#endif // include guard
