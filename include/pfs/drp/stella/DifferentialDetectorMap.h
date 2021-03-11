#ifndef PFS_DRP_STELLA_DIFFERENTIALDETECTORMAP_H
#define PFS_DRP_STELLA_DIFFERENTIALDETECTORMAP_H

#include "ndarray_fwd.h"

#include "lsst/afw/table/io/Persistable.h"

#include "pfs/drp/stella/SplinedDetectorMap.h"
#include "pfs/drp/stella/GlobalDetectorModel.h"
#include "pfs/drp/stella/ModelBasedDetectorMap.h"


namespace pfs {
namespace drp {
namespace stella {


/// DetectorMap referenced to a base detectorMap and a correction via a global detector model
///
/// Base detectorMap is a SplinedDetectorMap (in theory, we could use any kind of detectorMap,
/// but that would make it difficult to describe the datamodel of persisted data).
/// The GlobalDetectorModel maps fiberId,wavelength --> dx,dy.
class DifferentialDetectorMap : public ModelBasedDetectorMap {
  public:

    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param model : spectrograph model
    /// @param visitInfo : visit information
    /// @param metadata : FITS header
    DifferentialDetectorMap(
        std::shared_ptr<SplinedDetectorMap> base,
        GlobalDetectorModel const& model,
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr
    );

    virtual ~DifferentialDetectorMap() {}
    DifferentialDetectorMap(DifferentialDetectorMap const&) = default;
    DifferentialDetectorMap(DifferentialDetectorMap &&) = default;
    DifferentialDetectorMap & operator=(DifferentialDetectorMap const&) = default;
    DifferentialDetectorMap & operator=(DifferentialDetectorMap &&) = default;

    virtual std::shared_ptr<DetectorMap> clone() const override;

    std::shared_ptr<SplinedDetectorMap> getBase() const { return _base; }
    GlobalDetectorModel getModel() const { return _model; }

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

    bool isPersistable() const noexcept { return true; }

    class Factory;

  protected:
    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    virtual lsst::geom::PointD findPointImpl(int fiberId, double wavelength) const override;

    std::string getPersistenceName() const { return "DifferentialDetectorMap"; }
    std::string getPythonModule() const { return "pfs.drp.stella"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const;

    /// Reset cached elements after setting slit offsets
    virtual void _resetSlitOffsets() override;

  private:
    std::shared_ptr<SplinedDetectorMap> _base;
    GlobalDetectorModel _model;
};


}}}

#endif // include guard
