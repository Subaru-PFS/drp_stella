#ifndef PFS_DRP_STELLA_GLOBALDETECTORMAP_H
#define PFS_DRP_STELLA_GLOBALDETECTORMAP_H

#include <map>
#include "ndarray_fwd.h"

#include "lsst/afw/table/io/Persistable.h"

#include "pfs/drp/stella/spline.h"
#include "pfs/drp/stella/GlobalDetectorModel.h"
#include "pfs/drp/stella/DetectorMap.h"


namespace pfs {
namespace drp {
namespace stella {


/// DetectorMap implemented with a global detector model
class GlobalDetectorMap : public DetectorMap {
  public:
    using ParamType = float;

    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param fiberId : fiber identifiers
    /// @param distortionOrder : polynomial order for the distortion field
    /// @param dualDetector : model as a pair of detectors?
    /// @param parameters : model parameters
    /// @param spatialOffsets : per-fiber offsets in the spatial dimension
    /// @param spectralOffsets : per-fiber offsets in the spectral dimension
    /// @param visitInfo : visit information
    /// @param metadata : FITS header
    GlobalDetectorMap(
        lsst::geom::Box2I bbox,
        FiberIds const& fiberId,
        int distortionOrder,
        bool dualDetector,
        ndarray::Array<double, 1, 1> const& parameters,
        Array1D const& spatialOffsets=Array1D(),
        Array1D const& spectralOffsets=Array1D(),
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr
    );

    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param model : spectrograph model
    /// @param visitInfo : visit information
    /// @param metadata : FITS header
    GlobalDetectorMap(
        GlobalDetectorModel const& model,
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr
    );


    virtual ~GlobalDetectorMap() {}
    GlobalDetectorMap(GlobalDetectorMap const&) = default;
    GlobalDetectorMap(GlobalDetectorMap &&) = default;
    GlobalDetectorMap & operator=(GlobalDetectorMap const&) = default;
    GlobalDetectorMap & operator=(GlobalDetectorMap &&) = default;

    //@{
    /// Return the fiber centers
    virtual Array1D getXCenter(int fiberId) const override;
    virtual float getXCenter(int fiberId, float row) const override;
    //@}

    //@{
    /// Return the wavelength values
    virtual Array1D getWavelength(int fiberId) const override;
    virtual float getWavelength(int fiberId, float row) const override;
    //@}

    /// Return the fiberId given a position on the detector
    virtual int findFiberId(lsst::geom::PointD const& point) const override;

    //@{
    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    virtual lsst::geom::PointD findPoint(int fiberId, float wavelength) const override;
    virtual Array2D findPoint(FiberIds const& fiberId, Array1D const& wavelength) const;
    //@}

    /// Return the wavelength of a point on the detector, given a fiberId and row
    virtual float findWavelength(int fiberId, float row) const override;

    GlobalDetectorModel getModel() const { return _model; }
    int getDistortionOrder() const { return _model.getDistortionOrder(); }
    bool getDualDetector() const { return _model.getDualDetector(); }
    ndarray::Array<double, 1, 1> getParameters() const { return ndarray::copy(_model.getParameters()); }

    bool isPersistable() const noexcept { return true; }

    class Factory;

  protected:
    std::string getPersistenceName() const { return "GlobalDetectorMap"; }
    std::string getPythonModule() const { return "pfs.drp.stella"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const;

  private:
    /// Reset cached elements after setting slit offsets
    virtual void _resetSlitOffsets() override;

    using Spline = math::Spline<ParamType>;
    void _setSplines();

    GlobalDetectorModel _model;
    std::vector<Spline> _rowToWavelength;
    std::vector<Spline> _rowToXCenter;
};


}}}

#endif // include guard
