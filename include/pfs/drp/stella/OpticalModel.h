#ifndef PFS_DRP_STELLA_OPTICALMODEL_H
#define PFS_DRP_STELLA_OPTICALMODEL_H

#include "ndarray_fwd.h"
#include "lsst/geom/Point.h"
#include "lsst/geom/AffineTransform.h"

#include "pfs/drp/stella/Distortion.h"
#include "pfs/drp/stella/SplinedDetectorMap.h"
#include "pfs/drp/stella/spline.h"
#include "pfs/drp/stella/GridTransform.h"


/// Optical model of the spectrograph camera
///
/// The purpose of these modules is to provide a mapping from the slit to
/// detector pixels. We divide the mapping into four layers:
///
/// 1. Spectrograph coordinates (fiberId, wavelength): the basic coordinates of
///    the spectrograph, used for science. The fiberId is an integer identifier
///    for each fiber, which runs regularly across the detector. The wavelength
///    is specified in nm, and runs principally vertically on the detector.
/// 2. Slit coordinates (spatial, spectral): the coordinates relative to the
///    slit head. The spatial coordinate is in units of fibers, and the spectral
///    coordinate is in units of nm.
/// 3. Detector coordinates (x, y): the coordinates on the detector, in pixels.
///    These are where light from the spectrograph falls on the detector focal
///    plane. The units are pixels, and the origin is in the lower left corner
///    of the detector.
/// 4. Pixel coordinates (p, q): the coordinates of the physical pixels
///    recording the light. This differs from the detector coordinates for the
///    optical detectors (arm=b,r,m) because the detector is composed of two
///    CCDs, with a gap in between, and possibly a small rotation. This also
///    accounts for any distortion within the detector.
///
/// Mapping between these layers is handled by three classes:
///
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

namespace pfs {
namespace drp {
namespace stella {


using FiberMap = std::unordered_map<int, std::size_t>;

/// Make a mapping from fiberId to fiber index
FiberMap makeFiberMap(ndarray::Array<int, 1, 1> const& fiberId);


namespace detail {


/// Apply a method to two arrays
///
/// This is used to implement the array versions of the methods below.
///
/// @tparam Output : output type of the array
/// @param self : the object to call the method on
/// @param func : the method to call
/// @param input1 : the first input array
/// @param input2 : the second input array
/// @param name1 : the name of the first input array (for error messages)
/// @param name2 : the name of the second input array (for error messages)
/// @return an array of the outputs of the method applied to each pair of inputs
template <typename Output, typename Cls, typename Input1, int C1, typename Input2, int C2>
ndarray::Array<Output, 2, 2> applyMethod(
    Cls const& self,
    lsst::geom::Point2D (Cls::*func)(Input1, Input2) const,
    ndarray::Array<Input1, 1, C1> const& input1,
    ndarray::Array<Input2, 1, C2> const& input2,
    char const* name1,
    char const* name2
) {
    utils::checkSize(input1.size(), input2.size(), std::string(name1) + " vs " + name2);
    ndarray::Array<Output, 2, 2> result = ndarray::allocate(input1.size(), 2);
    for (std::size_t ii = 0; ii < input1.size(); ++ii) {
        lsst::geom::Point2D const output = (self.*func)(input1[ii], input2[ii]);
        result[ii][0] = output.getX();
        result[ii][1] = output.getY();
    }
    return result;
}


/// Apply a method to a 2D array
///
/// @tparam Output : output type of the array
/// @param self : the object to call the method on
/// @param func : the method to call
/// @param input : the input array, which must have two columns
template <typename Output, typename Cls, typename Input, int C>
ndarray::Array<Output, 2, 2> applyMethod(
    Cls const& self,
    lsst::geom::Point2D (Cls::*func)(Input, Input) const,
    ndarray::Array<Input, 2, C> const& input
) {
    utils::checkSize(input.getShape()[1], 2UL, "input columns");
    ndarray::Array<Output, 2, 2> result = ndarray::allocate(input.getShape());
    for (std::size_t ii = 0; ii < input.size(); ++ii) {
        lsst::geom::Point2D const output = (self.*func)(input[ii][0], input[ii][1]);
        result[ii][0] = output.getX();
        result[ii][1] = output.getY();
    }
    return result;
}


}  // namespace detail


/// Model of the slit head
///
/// This class provides a mapping from the spectrograph's coordinate system
/// (fiberId,wavelength) to the slit coordinate system (spatial, spectral).
/// The slit coordinate system is in units of fibers and nm (because that's
/// how the inputs to the optical model are defined).
///
/// Conversion from spectrograph (fiberId,wavelength) to slit (spatial,spectral)
/// coordinates includes:
/// 1. Slit offsets: per-fiber offsets in the spatial and spectral dimensions.
/// 2. Distortion: a (low-order) distortion (e.g., due to movement of the slit
///    head).
///
/// The slit offsets are in units of pixels, but everything else is happening
/// in fiber and wavelength units. For that reason, we also need the fiberPitch
/// (average pixels per fiber) and wavelengthDispersion (nm/pixels)
class SlitModel {
  public:
    using Array1I = ndarray::Array<int, 1, 1>;
    using Array1D = ndarray::Array<double, 1, 1>;
    using Array2D = ndarray::Array<double, 2, 1>;
    using DistortionList = std::vector<std::shared_ptr<Distortion>>;

    /// Ctor
    ///
    /// @param fiberId : fiber identifiers
    /// @param fiberPitch : average number of pixels per fiber
    /// @param fiberMin : minimum fiber (fiber units)
    /// @param wavelengthDispersion : dispersion (nm/pixel)
    /// @param wavelengthMin : minimum wavelength (nm)
    /// @param spatialOffsets : per-fiber offsets in the spatial dimension (pixels)
    /// @param spectralOffsets : per-fiber offsets in the spectral dimension (pixels)
    /// @param distortion : distortion for spectrograph -> slit
    SlitModel(
        ndarray::Array<int, 1, 1> const& fiberId,
        double fiberPitch,
        double fiberMin,
        double wavelengthDispersion,
        double wavelengthMin,
        Array1D const& spatialOffsets,
        Array1D const& spectralOffsets,
        DistortionList const& distortions=DistortionList()
    );

    /// Construct from a SplinedDetectorMap
    SlitModel(SplinedDetectorMap const& source, DistortionList const& distortions=DistortionList());

    SlitModel(SlitModel const&) = default;
    SlitModel(SlitModel &&) = default;
    SlitModel & operator=(SlitModel const&) = default;
    SlitModel & operator=(SlitModel &&) = default;
    ~SlitModel() = default;

    SlitModel copy() const;

    /// Return a copy of this SlitModel with an additional distortion applied
    SlitModel withDistortion(std::shared_ptr<Distortion> distortion) const;

    /// Return a copy of this SlitModel without any distortions applied
    SlitModel withoutDistortion() const;

    //@{
    /// Accessors
    Array1I const& getFiberId() const { return _fiberId; }
    double getFiberPitch() const { return _fiberPitch; }
    double getFiberMin() const { return _fiberMin; }
    double getWavelengthDispersion() const { return _wavelengthDispersion; }
    double getWavelengthMin() const { return _wavelengthMin; }
    Array1D const& getSpatialOffsets() const { return _spatialOffsets; }
    Array1D const& getSpectralOffsets() const { return _spectralOffsets; }
    DistortionList const& getDistortions() const { return _distortions; }
    //@}

    //@{
    /// Return slit offset for a single fiber, in pixels
    double getSpatialOffset(int fiberId) const;
    double getSpectralOffset(int fiberId) const;
    //@}

    //@{
    /// Convert from spectrograph coordinates (fiberId,wavelength) to slit coordinates (spatial,spectral)
    ///
    /// This applies the slit offsets and distortion. The output is in units of
    /// fibers and nm.
    ///
    /// There's no spectrographToSlit(Point2D) because the fiberId is an integer
    /// that identifies a specific slit offset.
    lsst::geom::Point2D spectrographToSlit(int fiberId, double wavelength) const {
        return preSlitToSlit(spectrographToPreSlit(fiberId, wavelength));
    }
    Array2D spectrographToSlit(Array1I const& fiberId, Array1D const& wavelength) const {
        return detail::applyMethod<double>(
            *this, &SlitModel::_spectrographToSlit, fiberId, wavelength, "fiberId", "wavelength"
        );
    }
    //@}

    //@{
    /// Convert from spectrograph coordinates (fiberId,wavelength) to pre-slit coordinates
    ///
    /// This applies the slit offsets and distortion, but doesn't convert back to fiber-wavelength units.
    /// This is used for getting coordinates in pixel units, which is needed for measuring and applying
    /// the distortion.
    ///
    /// There's no spectrographToPreSlit(Point2D) because the fiberId is an integer
    /// that identifies a specific slit offset.
    lsst::geom::Point2D spectrographToPreSlit(int fiberId, double wavelength) const {
        return _spectrographToPreSlit(fiberId, wavelength);
    }
    Array2D spectrographToPreSlit(Array1I const& fiberId, Array1D const& wavelength) const {
        return detail::applyMethod<double>(
            *this, &SlitModel::_spectrographToPreSlit, fiberId, wavelength, "fiberId", "wavelength"
        );
    }
    //@}

    //@{
    /// Convert from pre-slit coordinates to slit coordinates (spatial, spectral)
    lsst::geom::Point2D preSlitToSlit(double spatial, double spectral) const {
        return _preSlitToSlit(spatial, spectral);
    }
    Array2D preSlitToSlit(Array1D const& spatial, Array1D const& spectral) const {
        return detail::applyMethod<double>(
            *this, &SlitModel::_preSlitToSlit, spatial, spectral, "spatial", "spectral"
        );
    }
    lsst::geom::Point2D preSlitToSlit(lsst::geom::Point2D const& spatialSpectral) const {
        return preSlitToSlit(spatialSpectral.getX(), spatialSpectral.getY());
    }
    Array2D preSlitToSlit(Array2D const& spatialSpectral) const {
        return detail::applyMethod<double>(*this, &SlitModel::_preSlitToSlit, spatialSpectral);
    }
    //@}}

    //@{
    /// Convert from slit coordinates (spatial, spectral) to pre-slit coordinates
    ///
    /// This scales the coordinates to the same units, but doesn't apply the slit offsets or distortion.
    /// This is used for getting coordinates in pixel units, which is needed for measuring and applying
    /// the distortion.
    lsst::geom::Point2D slitToPreSlit(double spatial, double spectral) const {
        return _slitToPreSlit(spatial, spectral);
    }
    Array2D slitToPreSlit(Array1D const& spatial, Array1D const& spectral) const {
        return detail::applyMethod<double>(
            *this, &SlitModel::_slitToPreSlit, spatial, spectral, "spatial", "spectral"
        );
    }
    lsst::geom::Point2D slitToPreSlit(lsst::geom::Point2D const& spatialSpectral) const {
        return slitToPreSlit(spatialSpectral.getX(), spatialSpectral.getY());
    }
    Array2D slitToPreSlit(Array2D const& spatialSpectral) const {
        return detail::applyMethod<double>(*this, &SlitModel::_slitToPreSlit, spatialSpectral);
    }
    //@}

#if 0 // We need an inverse distortion to be able to convert from slit to spectrograph coordinates
    // NOTE: the slitToSpectrograph mapping does NOT include the slit offsets.
    // Since the slit offsets for different fibers are uncoordinated, it is
    // technically possible for the slit offsets to map a single point in slit
    // space to multiple points in spectrograph space.
    lsst::geom::Point2D slitToSpectrograph(double spatial, double spectral) const;
    lsst::geom::Point2D slitToSpectrograph(lsst::geom::Point2D const& spatialSpectral) const;
    Array2D slitToSpectrograph(Array1D const& spatial, Array1D const& spectral) const;
#endif

  private:
    //@{
    /// Non-overloaded versions of the methods above, which take scalar inputs and outputs.
    /// These are used with applyMethod, and save us from repeatedly specifying the overload.
    lsst::geom::Point2D _spectrographToSlit(int fiberId, double wavelength) const {
        lsst::geom::Point2D preslit = _spectrographToPreSlit(fiberId, wavelength);
        return _preSlitToSlit(preslit.getX(), preslit.getY());
    }
    lsst::geom::Point2D _spectrographToPreSlit(int fiberId, double wavelength) const;
    lsst::geom::Point2D _preSlitToSlit(double spatial, double spectral) const {
        return pixelsToFiberWavelength(lsst::geom::Point2D(spatial, spectral));
    }
    lsst::geom::Point2D _slitToPreSlit(double spatial, double spectral) const {
        return fiberWavelengthToPixels(spatial, spectral);
    }
    //@}

    //@{
    /// Apply scaling to convert between fiber,wavelength units and pixels
    lsst::geom::Point2D fiberWavelengthToPixels(double fiber, double wavelength) const {
        return lsst::geom::Point2D(
            (fiber - _fiberMin)*_fiberPitch,
            (wavelength - _wavelengthMin)/_wavelengthDispersion
        );
    }
    lsst::geom::Point2D pixelsToFiberWavelength(lsst::geom::Point2D const& pixels) const {
        return lsst::geom::Point2D(
            pixels.getX()/_fiberPitch + _fiberMin,
            pixels.getY()*_wavelengthDispersion + _wavelengthMin
        );
    }
    //@}

    double _fiberPitch;  ///< average pixels per fiber
    double _fiberMin;  ///< minimum fiberId
    double _wavelengthDispersion;  ///< nm/pixels
    double _wavelengthMin;  ///< minimum wavelength
    ndarray::Array<int, 1, 1> _fiberId;  ///< fiber identifiers
    Array1D _spatialOffsets;  ///< per-fiber offsets in the spatial dimension (pixels)
    Array1D _spectralOffsets;  ///< per-fiber offsets in the spectral dimension (pixels)
    std::unordered_map<int, std::size_t> _fiberMap;  // Mapping fiberId -> fiber index
    DistortionList _distortions;  ///< distortions for spectrograph -> slit
};


/// Optics model of the spectrograph camera
///
/// This provides a mapping from the slit coordinate system (spatial, spectral)
/// to the detector coordinate system (x, y).
///
/// The OpticsModel is built from the JEG optical model, which provides a
/// mapping between fiberId,wavelength and x,y on the detector. Additionally,
/// distortions can be applied after the JEG optical model. For efficiency,
/// we apply the distortions to the grid points of the JEG optical model in the
/// constructor, but save the original grid points.
///
/// The JEG optical model is defined in terms of fiberId and wavelength, but we
/// here call these "spatial" and "spectral" coordinates, to attempt to avoid
/// confusion with the "fiberId" and "wavelength" coordinates going into the
/// slit model. The "spatial" coordinate is in units of fibers, and the
/// "spectral" coordinate is in wavelength units of nm, so it's almost
/// equivalent, except that here we allow for interpolation between fibers.
class OpticsModel {
  public:
    using Array1I = ndarray::Array<int, 1, 1>;
    using Array2D = ndarray::Array<double, 2, 1>;
    using Array1D = ndarray::Array<double, 1, 1>;
    using DistortionList = std::vector<std::shared_ptr<Distortion>>;

    /// Ctor
    ///
    /// @param spatial : spatial coordinates of grid points (in units of fibers)
    /// @param spectral : spectral coordinates of grid points (in units of nm)
    /// @param x : x coordinates of grid points on the detector (in pixels)
    /// @param y : y coordinates of grid points on the detector (in pixels)
    /// @param distortions : distortions to apply after the JEG optical model
    OpticsModel(
        Array2D const& spatial,
        Array2D const& spectral,
        Array2D const& x,
        Array2D const& y,
        DistortionList const& distortions=DistortionList()
    );

    /// Construct from a SplinedDetectorMap
    ///
    /// The SplinedDetectorMap includes the JEG optical model. We extract the
    /// grid points from the splines, and use those to construct the
    /// OpticsModel. The distortions are applied on top.
    OpticsModel(SplinedDetectorMap const& source, DistortionList const& distortions=DistortionList());

    OpticsModel(OpticsModel const&) = default;
    OpticsModel(OpticsModel &&) = default;
    OpticsModel & operator=(OpticsModel const&) = default;
    OpticsModel & operator=(OpticsModel &&) = default;
    ~OpticsModel() = default;

    OpticsModel copy() const;

    //@{
    /// Accessors
    ///
    /// Note that getX() and getY() return the original grid points, without the
    /// distortions applied. The distorted x,y grid points can be obtained by
    // getSlitToDetector().getX() and getSlitToDetector().getY().
    Array2D const& getSpatial() const { return _slitToDetector.getU(); }
    Array2D const& getSpectral() const { return _slitToDetector.getV(); }
    Array2D const& getX() const { return _xOrig; }
    Array2D const& getY() const { return _yOrig; }
    GridTransform const& getSlitToDetector() const { return _slitToDetector; }
    GridTransform const& getDetectorToSlit() const { return _detectorToSlit; }
    DistortionList const& getDistortions() const { return _distortions; }
    //@}

    //@{
    /// Convert from slit coordinates (spatial, spectral) to detector coordinates (x, y)
    lsst::geom::Point2D slitToDetector(double spatial, double spectral) const;
    Array2D slitToDetector(Array1D const& spatial, Array1D const& spectral) const {
        return detail::applyMethod<double>(
            *this, &OpticsModel::slitToDetector, spatial, spectral, "spatial", "spectral"
        );
    }
    lsst::geom::Point2D slitToDetector(lsst::geom::Point2D const& spatialSpectral) const {
        return slitToDetector(spatialSpectral.getX(), spatialSpectral.getY());
    }
    Array2D slitToDetector(Array2D const& spatialSpectral) const {
        return detail::applyMethod<double>(*this, &OpticsModel::slitToDetector, spatialSpectral);
    }
    //@}

    //@{
    /// Convert from detector coordinates (x, y) to slit coordinates (spatial, spectral)
    lsst::geom::Point2D detectorToSlit(double x, double y) const;
    Array2D detectorToSlit(Array1D const& x, Array1D const& y) const {
        return detail::applyMethod<double>(*this, &OpticsModel::detectorToSlit, x, y, "x", "y");
    }
    lsst::geom::Point2D detectorToSlit(lsst::geom::Point2D const& xy) const {
        return detectorToSlit(xy.getX(), xy.getY());
    }
    Array2D detectorToSlit(Array2D const& xy) const {
        return detail::applyMethod<double>(*this, &OpticsModel::detectorToSlit, xy);
    }
    //@}

  private:

    /// Convenience constructor for constructing from a SplinedDetectorMap
    OpticsModel(
        std::tuple<Array2D, Array2D, Array2D, Array2D> const& grid,
         DistortionList const& distortions=DistortionList()
    );

    // Original x,y grid (the GridTransform will apply the distortions, so we save the original)
    Array2D _xOrig;  ///< original x coordinates of points on the detector
    Array2D _yOrig;  ///< original y coordinates of points on the detector
    DistortionList _distortions;  ///< distortions between slit and detector

    // Active grid, including the distortions
    GridTransform _slitToDetector;  ///< transform for interpolating between slit and detector coordinates
    GridTransform _detectorToSlit;  ///< transform for interpolating between detector and slit coordinates
};


/// Model for the position of the detector(s) in the spectrograph camera
///
/// This maps positions from their position on the detector to the pixels in the
/// detector.
///
/// The PFS detectors are either two 2k x 4k CCDs, or a monolithic 4k square
/// HgCdTe. When using two CCDs, we allow for one CCD (we choose the right-hand
/// one, with large x) to have an affine transform relative to the other.
///
/// After determining the position on the detector, we apply distortions to
/// account for effects such as the epoxy bumps and other detector geography.
class DetectorModel {
  public:
    using Array1D = ndarray::Array<double, 1, 1>;
    using Array2D = ndarray::Array<double, 2, 1>;
    using DistortionList = std::vector<std::shared_ptr<Distortion>>;

    //@{
    /// Ctor
    ///
    /// @param bbox : bounding box of the detector in pixels
    /// @param isDivided : whether the detector is divided into two halves (i.e, two CCDs)
    /// @param rightCcd : affine transform for the right CCD (if the detector is divided)
    /// @param distortions : distortions to apply within the detector
    DetectorModel(
        lsst::geom::Box2I const& bbox,
        lsst::geom::AffineTransform const& rightCcd,
        DistortionList const& distortions=DistortionList()
    );
    DetectorModel(
        lsst::geom::Box2I const& bbox,
        DistortionList const& distortions=DistortionList()
    );
    //@}

    DetectorModel(DetectorModel const&) = default;
    DetectorModel(DetectorModel &&) = default;
    DetectorModel & operator=(DetectorModel const&) = default;
    DetectorModel & operator=(DetectorModel &&) = default;
    ~DetectorModel() = default;

    DetectorModel copy() const;

    //@{
    /// Accessors
    lsst::geom::Box2I const& getBBox() const { return _bbox; }
    bool getIsDivided() const { return _isDivided; }
    lsst::geom::AffineTransform const& getRightCcd() const { return _rightCcd; }
    DistortionList const& getDistortions() const { return _distortions; }
    //@}

    //@{
    /// Convert from detector coordinates (x, y) to pixel coordinates (p, q)
    lsst::geom::Point2D detectorToPixels(lsst::geom::Point2D const& detector) const;
    lsst::geom::Point2D detectorToPixels(double x, double y) const {
        return detectorToPixels(lsst::geom::Point2D(x, y));
    }
    Array2D detectorToPixels(Array1D const& x, Array1D const& y) const {
        return detail::applyMethod<double>(*this, &DetectorModel::detectorToPixels, x, y, "x", "y");
    }
    Array2D detectorToPixels(Array2D const& xy) const {
        return detail::applyMethod<double>(*this, &DetectorModel::detectorToPixels, xy);
    }
    //@}

    //@{
    /// Convert from pixel coordinates (p, q) to detector coordinates (x, y)
    lsst::geom::Point2D pixelsToDetector(lsst::geom::Point2D const& pixels) const;
    lsst::geom::Point2D pixelsToDetector(double p, double q) const {
        return pixelsToDetector(lsst::geom::Point2D(p, q));
    }
    Array2D pixelsToDetector(Array1D const& p, Array1D const& q) const {
        return detail::applyMethod<double>(*this, &DetectorModel::pixelsToDetector, p, q, "p", "q");
    }
    Array2D pixelsToDetector(Array2D const& pq) const {
        return detail::applyMethod<double>(*this, &DetectorModel::pixelsToDetector, pq);
    }
    //@}

    /// Return neighboring columns
    ///
    /// This is used for determining which pixels to extract (see
    /// DetectorMap::getTracePosition). Given a center position in the detector
    /// frame and the half-width of the region of interest, we return the
    /// minimum p value for columns in the region of interest, and the position
    /// of each column relative to the center position in the pixels frame.
    std::pair<int, ndarray::Array<double, 1, 1>> detectorToPixelsColumns(
        lsst::geom::Point2D const& detector,
        int halfWidth
    ) const;

  private:

    /// Base constructor for the two public ctors
    DetectorModel(
        lsst::geom::Box2I const& bbox,
        bool isDivided,
        lsst::geom::AffineTransform const& rightCcd,
        DistortionList const& distortions
    );

    lsst::geom::Box2I _bbox;  ///< detector size
    bool _isDivided;  ///< is the detector divided into two halves?
    lsst::geom::AffineTransform _rightCcd;  ///< transformation for the right CCD
    DistortionList _distortions;  ///< distortions on the detector (e.g., epoxy bumps)

    double _xCenter;  ///< location of the division
    double _xOffset, _xScale, _yOffset, _yScale;  ///< Normalization for right CCD transformation
    std::size_t INVERSE_MAX_ITER = 1000;  ///< maximum number of iterations for inverse transformations
    double INVERSE_PRECISION = 1.0e-3;  ///< precision for inverse transformations
};


}}}  // namespace pfs::drp::stella

#endif  // include guard
