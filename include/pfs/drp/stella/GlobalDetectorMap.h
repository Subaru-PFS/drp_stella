#ifndef PFS_DRP_STELLA_GLOBALDETECTORMAP_H
#define PFS_DRP_STELLA_GLOBALDETECTORMAP_H

#include "ndarray_fwd.h"

#include "lsst/afw/table/io/Persistable.h"

#include "pfs/drp/stella/GlobalDetectorModel.h"
#include "pfs/drp/stella/ModelBasedDetectorMap.h"


namespace pfs {
namespace drp {
namespace stella {


/// DetectorMap implemented with a global detector model
class GlobalDetectorMap : public ModelBasedDetectorMap {
  public:
    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param model : spectrograph model
    /// @param visitInfo : visit information
    /// @param metadata : FITS header
    GlobalDetectorMap(
        lsst::geom::Box2I const& bbox,
        GlobalDetectorModel const& model,
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr
    );

    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param distortionOrder : order of distortion polynomials
    /// @param fiberId : fiber identifiers
    /// @param fiberPitch : distance between fibers (pixels)
    /// @param dispersion : linear wavelength dispersion (nm per pixel)
    /// @param wavelengthCenter : central wavelength (nm)
    /// @param buffer : fraction of expected wavelength range by which to expand
    ///                 the wavelength range in the polynomials; this accounts
    ///                 for distortion or small inaccuracies in the dispersion.
    /// @param fiberCenter : fiberId separating low- and high-fiberId CCDs
    /// @param xCoefficients : x distortion polynomial coefficients
    /// @param yCoefficients : y distortion polynomial coefficients
    /// @param highCcd : affine transformation parameters for the high-fiberId CCD
    /// @param spatialOffsets : slit offsets in the spatial dimension
    /// @param spectralOffsets : slit offsets in the spectral dimension
    /// @param visitInfo : visit information
    /// @param metadata : FITS header
    GlobalDetectorMap(
        lsst::geom::Box2I const& bbox,
        int distortionOrder,
        ndarray::Array<int, 1, 1> const& fiberId,
        double fiberPitch,
        double dispersion,
        double wavelengthCenter,
        float buffer,
        float fiberCenter,
        ndarray::Array<double, 1, 1> const& xCoefficients,
        ndarray::Array<double, 1, 1> const& yCoefficients,
        ndarray::Array<double, 1, 1> const& highCcd,
        ndarray::Array<double, 1, 1> const& spatialOffsets,
        ndarray::Array<double, 1, 1> const& spectralOffsets,
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr
    );

    virtual ~GlobalDetectorMap() {}
    GlobalDetectorMap(GlobalDetectorMap const&) = default;
    GlobalDetectorMap(GlobalDetectorMap &&) = default;
    GlobalDetectorMap & operator=(GlobalDetectorMap const&) = default;
    GlobalDetectorMap & operator=(GlobalDetectorMap &&) = default;

    virtual std::shared_ptr<DetectorMap> clone() const override;

    GlobalDetectorModel getModel() const { return _model; }
    int getDistortionOrder() const { return _model.getDistortionOrder(); }

    /// Measure and apply slit offsets
    ///
    /// This implementation fits for a single pair of spatial,spectral offsets
    /// that minimises chi^2. Simply measuring the mean x and y offsets is not
    /// sufficient here, because the detectorMap includes distortion (so the
    /// effects of a shift at the edge can be different from the effects of the
    /// same shift at the center). Instead, we fit for x,y slit offsets using
    /// the detectorMap.
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

    /// Reset cached elements after setting slit offsets
    virtual void _resetSlitOffsets() override;

    std::string getPersistenceName() const { return "GlobalDetectorMap"; }
    std::string getPythonModule() const { return "pfs.drp.stella"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const;

  private:
    GlobalDetectorModel _model;
};


}}}

#endif // include guard
