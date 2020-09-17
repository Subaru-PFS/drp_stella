#ifndef PFS_DRP_STELLA_GLOBALDETECTORMODEL_H
#define PFS_DRP_STELLA_GLOBALDETECTORMODEL_H

#include <map>
#include "ndarray_fwd.h"

#include "lsst/afw/table/io/Persistable.h"
#include "lsst/afw/math/FunctionLibrary.h"

#include "pfs/drp/stella/spline.h"
#include "pfs/drp/stella/DetectorMap.h"


namespace pfs {
namespace drp {
namespace stella {


/// Model for transforming fiberId,wavelength to x,y on the detector
///
/// The model assumes that fibers are inserted in the slit in numeric order,
/// with a constant spacing between, but individual fiber offsets are applied
/// to account for any small deviations from this assumption. The wavelength
/// is linear with distance from the slit.
///
/// A two-dimensional polynomial distortion field is applied to this simple
/// setup to account for the camera optics, whereupon the positions are mapped
/// onto the detector. We allow for the detector to be comprised of two
/// discrete elements with a relative offset and rotation.
///
/// The model parameters are:
/// * fiberCenter: the fiberId which is the optical center of the slit.
/// * fiberPitch: distance between fibers (undistorted pixels).
/// * wavelengthCenter: the central wavelength (nm).
/// * dispersion: the linear dispersion (nm per undistorted pixel).
/// * ccdRotation: the rotation of the right-hand detector relative to the
///       left-hand detector (radians).
/// * x0, y0: optical center in the distorted frame.
/// * xGap, yGap: distance between the left-hand and right-hand detectors.
/// * xDistortion, yDistortion: 2D polynomial distortion field coefficients.
/// * spatialOffset, spectralOffset: per-fiber offsets in the spatial and
///       spectral dimensions.
///
/// There is a strong degeneracy between some parameters, especially
/// fiberCenter, fiberPitch, wavelengthCenter, dispersion, x0, y0 and the
/// linear part of the distortion polynomials; the distortion can also couple
/// into the per-fiber offsets. This requires care when fitting.
///
/// We want to be able to construct the model quickly, since it will be used in
/// tight loops (e.g., a non-linear fitter), where the model evaluation time is
/// a major limitation on the runtime; and the user shouldn't have to supply all
/// the model parameters independently. For this reason, we have constructors
/// that take an array of parameters (either as a std::vector or ndarray::Array)
/// and store that array as an ndarray::Array (because std::vector can be
/// converted to an ndarray::Array without copying, but not vice versa). But in
/// order to support the DetectorMap class, where the slit offsets are stored
/// separately, we also store the slit offsets separately from the rest of the
/// parameters (using ndarray::Array, this can be done without any copying).
class GlobalDetectorModel {
  public:
    using ParamArray = ndarray::Array<double, 1, 1>;
    using Polynomial = lsst::afw::math::Chebyshev1Function2<double>;
    using FiberMap = std::unordered_map<int, std::size_t>;

    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param distortionOrder : polynomial order for distortion
    /// @param fiberId : fiberId values for fibers
    /// @param dualDetector : detector is two individual CCDs?
    /// @param parameters : model parameters
    /// @param copyParameters : copy parameter array (e.g., to avoid parameters going out of scope)?
    GlobalDetectorModel(
        lsst::geom::Box2I const& bbox,
        int distortionOrder,
        ndarray::Array<int, 1, 1> const& fiberId,
        bool dualDetector,
        std::vector<double> const& parameters,
        bool copyParameters=true
    );

    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param distortionOrder : polynomial order for distortion
    /// @param fiberId : fiberId values for fibers
    /// @param dualDetector : detector is two individual CCDs?
    /// @param parameters : model parameters
    GlobalDetectorModel(
        lsst::geom::Box2I const& bbox,
        int distortionOrder,
        ndarray::Array<int, 1, 1> const& fiberId,
        bool dualDetector,
        ndarray::Array<double const, 1, 1> const& parameters
    );

    //@{
    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param distortionOrder : polynomial order for distortion
    /// @param fiberId : fiberId values for fibers
    /// @param dualDetector : detector is two individual CCDs?
    /// @param parameters : distortion parameters
    /// @param spatialOffsets : slit offsets in the spatial dimension
    /// @param spectralOffsets : slit offsets in the spectral dimension
    GlobalDetectorModel(
        lsst::geom::Box2I const& bbox,
        int distortionOrder,
        ndarray::Array<int, 1, 1> const& fiberId,
        bool dualDetector,
        ndarray::Array<double const, 1, 1> const& parameters,
        ndarray::Array<double const, 1, 1> const& spatialOffsets,
        ndarray::Array<double const, 1, 1> const& spectralOffsets
    );
    GlobalDetectorModel(
        lsst::geom::Box2I const& bbox,
        int distortionOrder,
        ndarray::Array<int, 1, 1> const& fiberId,
        bool dualDetector,
        ndarray::Array<double const, 1, 1> const& parameters,
        ndarray::Array<float, 1, 1> const& spatialOffsets,
        ndarray::Array<float, 1, 1> const& spectralOffsets
    );
    //@}

    //@{
    /// Evaluate the model
    ///
    /// @param fiberId : fiber identifier
    /// @param wavelength : wavelength (nm)
    /// @param fiberIndex : index for fiber
    /// @return x,y position on detector
    lsst::geom::Point2D operator()(int fiberId, double wavelength) const {
        return operator()(fiberId, wavelength, _fiberMap.at(fiberId));
    }
    ndarray::Array<double, 2, 1> operator()(
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& wavelength
    ) const;
    lsst::geom::Point2D operator()(int fiberId, double wavelength, std::size_t fiberIndex) const;
    ndarray::Array<double, 2, 1> operator()(
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& wavelength,
        ndarray::Array<std::size_t, 1, 1> const& fiberIndex
    ) const;
    //@}

    /// Fit a model to measured arc line positions
    ///
    /// @param bbox : detector bounding box
    /// @param distortionOrder : polynomial order for distortion
    /// @param dualDetector : detector is two individual CCDs?
    /// @param fiberId : fiberId values for arc lines
    /// @param wavelength : wavelength values for arc lines
    /// @param xx : x values for arc lines
    /// @param yy : y values for arc lines
    /// @param xErr : error in x values for arc lines
    /// @param yErr : error in y values for arc lines
    /// @param good : whether value should be used in the fit
    /// @param parameters : guess for parameters
    static GlobalDetectorModel fit(
        lsst::geom::Box2I const& bbox,
        int distortionOrder,
        bool dualDetector,
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& wavelength,
        ndarray::Array<double, 1, 1> const& xx,
        ndarray::Array<double, 1, 1> const& yy,
        ndarray::Array<double, 1, 1> const& xErr,
        ndarray::Array<double, 1, 1> const& yErr,
        ndarray::Array<bool, 1, 1> const& good=ndarray::Array<bool, 1, 1>(),
        ndarray::Array<double, 1, 1> const& parameters=ndarray::Array<double, 1, 1>()
    );

    //@{
    /// Return the total number of parameters for the model
    static std::size_t getNumParameters(int distortionOrder, std::size_t numFibers) {
        return BULK + GlobalDetectorModel::getNumDistortion(distortionOrder) + 2*numFibers;
    }
    std::size_t getNumParameters() const {
        return getNumParameters(getDistortionOrder(), getNumFibers());
    }
    //@}

    /// Return the number of fibers
    std::size_t getNumFibers() const { return _fiberMap.size(); }

    //@{
    /// Accessors
    lsst::geom::Box2I getBBox() const { return _bbox; }
    int getDistortionOrder() const { return _distortionOrder; }
    ndarray::Array<int, 1, 1> getFiberId() const;
    bool getDualDetector() const { return _dualDetector; }
    ndarray::Array<double const, 1, 1> const& getParameters() const { return _parameters; }
    Polynomial const& getXDistortion() const { return _xDistortion; }
    Polynomial const& getYDistortion() const { return _yDistortion; }
    double getSpatialOffset(std::size_t index) const { return _spatialOffsets[index]; }
    double getSpectralOffset(std::size_t index) const { return _spectralOffsets[index]; }
    ndarray::Array<double const, 1, 1> const getSpatialOffsets() const { return _spatialOffsets; }
    ndarray::Array<double const, 1, 1> const getSpectralOffsets() const { return _spectralOffsets; }
    double getFiberCenter() const { return _parameters[FIBER_CENTER]; }
    double getFiberPitch() const { return _parameters[FIBER_PITCH]; }
    double getWavelengthCenter() const { return _parameters[WAVELENGTH_CENTER]; }
    double getDispersion() const { return _parameters[DISPERSION]; }
    double getCcdRotation() const { return _parameters[CCD_ROTATION]; }
    double getX0() const { return _parameters[X0]; }
    double getY0() const { return _parameters[Y0]; }
    double getXGap() const { return _parameters[X_GAP]; }
    double getYGap() const { return _parameters[Y_GAP]; }
    double getCosCcdRotation() const { return _cosCcdRotation; }
    double getSinCcdRotation() const { return _sinCcdRotation; }
    //@}

    /// Return the detector x center
    ///
    /// This is the dividing line between the left and right CCDs if dualDetector
    float getXCenter() const { return 0.5*(_bbox.getMinX() + _bbox.getMaxX()); }

    //@{
    /// Return the distortion polynomial coefficients
    ndarray::Array<double const, 1, 1> const getXCoefficients() const;
    ndarray::Array<double const, 1, 1> const getYCoefficients() const;
    //@}

    /// Generate a vector of parameters suitable for fitting
    ///
    /// Puts named parameters in the right order and position in the vector.
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param numFibers : number of fibers
    /// @param fiberCenter : center of the slit, in fiberId
    /// @param fiberPitch : distance between the fibers (pixels/fiber)
    /// @param wavelengthCenter : center of the slit, in wavelength (nm)
    /// @param dispersion : linear dispersion (nm/pixel)
    /// @param ccdRotation : rotation of right-hand CCD relative to left-hand (radians)
    /// @param x0,y0 : optical center
    /// @param xGap, yGap : offset between CCDs (pixels)
    /// @param distortionCoeff : distortion coefficients
    /// @param spatialOffsets : slit offsets in the spatial dimension
    /// @param spectralOffsets : slit offsets in the spectral dimension
    /// @returns parameter vector
    static std::vector<double> makeParameters(
        int distortionOrder,
        std::size_t numFibers,
        double fiberCenter,
        double fiberPitch,
        double wavelengthCenter,
        double dispersion,
        double ccdRotation,
        double x0,
        double y0,
        double xGap=0,
        double yGap=0,
        ndarray::Array<double, 1, 1> const& distortionCoeff=ndarray::Array<double, 1, 1>(),
        ndarray::Array<double, 1, 1> const& spatialOffsets=ndarray::Array<double, 1, 1>(),
        ndarray::Array<double, 1, 1> const& spectralOffsets=ndarray::Array<double, 1, 1>()
    );

    //@{
    /// Return number of distortion parameters
    static std::size_t getNumDistortion(int distortionOrder) {
        return 2*Polynomial::nParametersFromOrder(distortionOrder);
    }
    std::size_t getNumDistortion() const {
        return getNumDistortion(getDistortionOrder());
    }
    //@}

  protected:
    friend class GlobalDetectorMap;

    /// Parameter indices
    enum ParameterIndex {
        FIBER_CENTER = 0,  // Slit center in fiberId
        FIBER_PITCH = 1,  // Distance between fibers
        WAVELENGTH_CENTER = 2,  // Slit center in wavelength
        DISPERSION = 3,  // linear dispersion
        CCD_ROTATION = 4,  // rotation of right-hand CCD relative to left-hand
        X0 = 5,  // optical center in x
        Y0 = 6,  // optical center in y
        X_GAP = 7,  // offset between CCDs in x
        Y_GAP = 8,  // offset between CCDs in y
        // After this point, these cease to refer to individual parameters!
        // You can use BULK to refer to the start index of these individual parameters.
        BULK = 9,
        LINEAR = 10,  // Linear distortion parameters
        DISTORTION = 11,  // Non-linear distortion parameters
        OFFSETS = 12,  // ALL slit offset parameters
    };

    /// Return coefficients for polynomial
    ///
    /// Used to construct the Polynomial objects
    ///
    /// @param start : parameter index at which to start
    /// @param num : number of parameters to copy
    std::vector<double> getPolynomialCoefficients(std::size_t start, std::size_t num);

    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param distortionOrder : polynomial order for distortion
    /// @param fiberMap : mapping of fiberId to fiber index
    /// @param dualDetector : detector is two individual CCDs?
    /// @param parameters : distortion parameters
    /// @param copyParameters : copy parameter array (e.g., to avoid parameters going out of scope)?
    GlobalDetectorModel(
        lsst::geom::Box2I const& bbox,
        int distortionOrder,
        FiberMap const& fiberMap,
        bool dualDetector,
        std::vector<double> const& parameters,
        bool copyParameters=true
    );

    /// Ctor
    ///
    /// @param bbox : detector bounding box
    /// @param distortionOrder : polynomial order for distortion
    /// @param fiberMap : mapping of fiberId to fiber index
    /// @param dualDetector : detector is two individual CCDs?
    /// @param parameters : distortion parameters
    GlobalDetectorModel(
        lsst::geom::Box2I const& bbox,
        int distortionOrder,
        FiberMap const& fiberMap,
        bool dualDetector,
        ndarray::Array<double const, 1, 1> const& parameters
    );

    /// Ctor
    ///
    /// This is the master constructor, to which all others delegate.
    ///
    /// @param bbox : detector bounding box
    /// @param distortionOrder : polynomial order for distortion
    /// @param fiberMap : mapping of fiberId to fiber index
    /// @param dualDetector : detector is two individual CCDs?
    /// @param parameters : distortion parameters
    /// @param spatialOffsets : slit offsets in the spatial dimension
    /// @param spectralOffsets : slit offsets in the spectral dimension
    GlobalDetectorModel(
        lsst::geom::Box2I const& bbox,
        int distortionOrder,
        FiberMap const& fiberMap,
        bool dualDetector,
        ndarray::Array<double const, 1, 1> const& parameters,
        ndarray::Array<double const, 1, 1> const& spatialOffsets,
        ndarray::Array<double const, 1, 1> const& spectralOffsets
    );

    friend std::ostream& operator<<(std::ostream& os, GlobalDetectorModel const& model);

  private:
    // Configuration
    lsst::geom::Box2I _bbox;
    int _distortionOrder;
    FiberMap _fiberMap;
    std::size_t _numFibers;
    bool _dualDetector;

    // Calculation parameters
    ndarray::Array<double const, 1, 1> _parameters;
    Polynomial _xDistortion;
    Polynomial _yDistortion;
    ndarray::Array<double const, 1, 1> _spatialOffsets;
    ndarray::Array<double const, 1, 1> _spectralOffsets;

    // Derived calculation parameters
    double _cosCcdRotation;
    double _sinCcdRotation;
};


}}}  // namespace pfs::drp::stella

#endif  // include guard
