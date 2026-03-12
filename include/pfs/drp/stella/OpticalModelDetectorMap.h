#ifndef PFS_DRP_STELLA_OPTICALMODELDETECTORMAP_H
#define PFS_DRP_STELLA_OPTICALMODELDETECTORMAP_H

#include "ndarray_fwd.h"

#include "lsst/geom/AffineTransform.h"
#include "lsst/afw/table/io/Persistable.h"
#include "lsst/cpputils/Cache.h"
#include "lsst/cpputils/hashCombine.h"

#include "pfs/drp/stella/SplinedDetectorMap.h"
#include "pfs/drp/stella/Distortion.h"
#include "pfs/drp/stella/OpticalModel.h"


namespace pfs {
namespace drp {
namespace stella {


/// DetectorMap implemented as a series of layers, following the optical path
/// from the slit to the detector.
///
/// The layers are:
/// 1. SlitModel: spectrograph coordinates --> slit coordinates. This is a
///    relatively minor perturbation of the spectrograph coordinates, which
///    includes per-fiber offsets and a (low-order) distortion (e.g., due to
///    movement of the slit head).
/// 2. OpticsModel: slit coordinates --> detector coordinates. This is provided
///    by JEG's optical model (a grid of fiberId,wavelength vs x,y on the
///    detector) with additional distortion applied.
/// 3. DetectorModel: detector coordinates --> pixel coordinates. For the NIR
///    detectors, this is a no-op. For the optical detectors, this accounts for
///    the chip gap and slight rotation differences between the chips, and
///    includes any distortion within the detector (e.g., epoxy blobs and other
///    detector geography).
class OpticalModelDetectorMap : public DetectorMap {
  public:

    enum Coordinate {
        WAVELENGTH = 1,
        SLIT_SPATIAL = 2,
        SLIT_SPECTRAL = 3,
        DETECTOR_X = 4,
        DETECTOR_Y = 5,
        PIXELS_P = 6,
        PIXELS_Q = 7,
        ROW = 7,  // Alias for PIXELS_Q
        COL = 6  // Alias for PIXELS_P
    };

    using Spline = math::Spline<double>;
    using SplinePtr = std::shared_ptr<Spline>;
    using SplineKey = std::tuple<int, Coordinate, Coordinate>;

    /// Data from the optical model, for a single fiber
    ///
    /// Each wavelength is associated with a position in slit coordinates
    /// (spatial, spectral), detector coordinates (x, y), and pixel coordinates
    /// (p, q).
    struct Data {
        Array1D wavelength;  ///< wavelengths for each point along the fiber trace
        ndarray::Array<double, 2, 2> slit;  ///< corresponding slit coordinates (spatial, spectral)
        ndarray::Array<double, 2, 2> detector;  ///< corresponding detector coordinates (x, y)
        ndarray::Array<double, 2, 2> pixels;  ///< corresponding pixel coordinates (p, q)

        Data(
            Array1D const& wavelength,
            ndarray::Array<double, 2, 2> const& slit,
            ndarray::Array<double, 2, 2> const& detector,
            ndarray::Array<double, 2, 2> const& pixels
        ) : wavelength(wavelength), slit(slit), detector(detector), pixels(pixels) {}

        /// Return the array for the given coordinate
        Array1D getArray(Coordinate coord) const;

        /// Generate a spline between the given x and y coordinates
        ///
        /// The user is responsible for ensuring that the x coordinates are strictly
        /// increasing; no checks are performed.
        SplinePtr getSpline(Coordinate x, Coordinate y) const {
            return std::make_shared<Spline>(getArray(x), getArray(y));
        }
    };

    /// Ctor
    OpticalModelDetectorMap(
        lsst::geom::Box2I const& bbox,
        SlitModel const& slitModel,
        OpticsModel const& opticsModel,
        DetectorModel const& detectorModel,
        VisitInfo const& visitInfo=VisitInfo(lsst::daf::base::PropertyList()),
        std::shared_ptr<lsst::daf::base::PropertySet> metadata=nullptr
    );

    virtual ~OpticalModelDetectorMap() noexcept override {}
    OpticalModelDetectorMap(OpticalModelDetectorMap const&) = default;
    OpticalModelDetectorMap(OpticalModelDetectorMap &&) = default;
    OpticalModelDetectorMap & operator=(OpticalModelDetectorMap const&) = default;
    OpticalModelDetectorMap & operator=(OpticalModelDetectorMap &&) = default;

    virtual std::shared_ptr<DetectorMap> clone() const override;

    SlitModel const& getSlitModel() const { return _slitModel; }
    OpticsModel const& getOpticsModel() const { return _opticsModel; }
    DetectorModel const& getDetectorModel() const { return _detectorModel; }

    /// Return the data for a particular fiber
    Data getData(int fiberId) const {
        return _data(fiberId, [this](int fiberId) { return makeData(fiberId); });
    }

    //@{
    /// Return a spline for a particular fiber and coordinate pair
    Spline const& getSpline(int fiberId, Coordinate coordFrom, Coordinate coordTo) const {
        return getSpline(std::make_tuple(fiberId, coordFrom, coordTo));
    }
    Spline const& getSpline(SplineKey const& key) const {
        return *_splines(key, [this](SplineKey const& key) { return makeSpline(key); });
    }
    //@}

    //@{
    /// Calculate any coordinate given any other coordinate, using the splines
    ///
    /// @param fiberId : fiber identifier
    /// @param coordFrom : coordinate to convert from
    /// @param coordTo : coordinate to convert to
    /// @param value : value of the coordFrom coordinate to convert
    double calculate(int fiberId, Coordinate coordFrom, Coordinate coordTo, double value) const;
    Array1D calculate(
        FiberIds const& fiberId, Coordinate coordFrom, Coordinate coordTo, Array1D const& value
    ) const;
    //@}

    /// row -> wavelength
    Spline const& getWavelengthSpline(int fiberId) const {
        return getSpline(fiberId, ROW, WAVELENGTH);
    }

    /// wavelength -> row
    Spline const& getRowSpline(int fiberId) const {
        return getSpline(fiberId, WAVELENGTH, ROW);
    }

    /// row -> x on detector
    Spline const& getXDetectorSpline(int fiberId) const {
        return getSpline(fiberId, ROW, DETECTOR_X);
    }

    /// row -> y on detector
    Spline const& getYDetectorSpline(int fiberId) const {
        return getSpline(fiberId, ROW, DETECTOR_Y);
    }

    /// Full-fidelity findPoint
    ///
    /// Rather than using the splines, trace the point through each layer of the
    /// optical model.
    lsst::geom::Point2D findPointFull(int fiberId, double wavelength) const;

    bool isPersistable() const noexcept override { return true; }
    class Factory;

  protected:

    /// Return the position of the fiber trace on the detector, given a fiberId and wavelength
    ///
    /// Implementation of findPoint, for subclasses to define.
    virtual lsst::geom::PointD findPointImpl(int fiberId, double wavelength) const override;

    /// Return the wavelength of a point on the detector, given a fiberId and row
    ///
    /// Implementation of findWavelength, for subclasses to define.
    virtual double findWavelengthImpl(int fiberId, double row) const override;

    /// Return the center of the fiber trace, given a fiberId and row
    ///
    /// Implementation of getXCenter, for subclasses to define.
    virtual double getXCenterImpl(int fiberId, double row) const override;

    /// Return the position of the fiber trace
    ///
    /// Implementation of getTracePosition, allowing subclasses to override.
    virtual std::pair<int, ndarray::Array<double, 1, 1>> getTracePositionImpl(
        int fiberId,
        int row,
        int halfWidth
    ) const override;

    /// Construct the data for a particular fiber
    Data makeData(int fiberId) const;

    //@{
    /// Construct spline
    SplinePtr makeSpline(int fiberId, Coordinate coordFrom, Coordinate coordTo) const {
        return getData(fiberId).getSpline(coordFrom, coordTo);
    }
    SplinePtr makeSpline(SplineKey const& key) const {
        return makeSpline(std::get<0>(key), std::get<1>(key), std::get<2>(key));
    }
    //@}

    /// Reset cached elements after setting slit offsets
    virtual void _resetSlitOffsets() override;

    std::string getPythonModule() const override { return "pfs.drp.stella"; }
    std::string getPersistenceName() const override { return "OpticalModelDetectorMap"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const override;

  private:
    SlitModel _slitModel;
    OpticsModel _opticsModel;
    DetectorModel _detectorModel;
    std::pair<double, double> _wavelengthRange;  ///< (min, max) wavelength
    std::size_t _numKnots;  ///< number of knots to use for the per-fiber splines

    /// Per-fiber data from the optical model, behind a cache to avoid constructing it until needed.
    ///
    /// Use getData to get the data (and make it if it's not already cached).
    mutable lsst::cpputils::Cache<int, Data> _data;

    struct SplineKeyHash {
        std::size_t operator()(SplineKey const& key) const {
            std::size_t seed = 0;
            return lsst::cpputils::hashCombine(seed, std::get<0>(key), std::get<1>(key), std::get<2>(key));
        }
    };

    /// Per-fiber splines, behind a cache to avoid constructing them until needed.
    /// The key is (fiberId, coordFrom, coordTo).
    /// Use getSpline to get a spline (and make it if it's not already cached).
    mutable lsst::cpputils::Cache<SplineKey, SplinePtr, SplineKeyHash> _splines;
};


}}}

#endif // include guard
