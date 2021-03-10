#ifndef PFS_DRP_STELLA_GLOBALDETECTORMODEL_H
#define PFS_DRP_STELLA_GLOBALDETECTORMODEL_H

#include <map>
#include "ndarray_fwd.h"

#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"
#include "lsst/geom/AffineTransform.h"
#include "lsst/afw/table/io/Persistable.h"

#include "pfs/drp/stella/spline.h"
#include "pfs/drp/stella/math/NormalizedPolynomial.h"
#include "pfs/drp/stella/DetectorMap.h"


namespace pfs {
namespace drp {
namespace stella {


/// Scaling of fiberId,wavelength to nominal pixels
///
/// This is a container for parameters that convert fiberId and wavelength to
/// nominal pixels. This is used to set the scaling for the distortion
/// polynomials and the spatial and spectral offsets in the GlobalDetectorModel.
/// As such, getting these values absolutely correct is not necessary, but it is
/// convenient for interpreting the model parameters if they are close.
///
/// The coordinate system in which fiberId,wavelength have been scaled is
/// referred to as 'xi,eta'.
struct GlobalDetectorModelScaling {
    double fiberPitch;  ///< distance between fibers (pixels)
    double dispersion;  ///< linear wavelength dispersion (nm per pixel)
    double wavelengthCenter;  ///< central wavelength (nm)
    int minFiberId;  ///< minimum fiberId value
    int maxFiberId;  ///< maximum fiberId value
    std::size_t height;  ///< height of detector (wavelength dimension; pixel)
    float buffer;  ///< fraction of expected wavelength range by which to expand the wavelength range in the
                   ///< polynomials; this accounts for distortion or small inaccuracies in the dispersion.

    /// Ctor
    ///
    /// @param fiberPitch_ : distance between fibers (pixels)
    /// @param dispersion_ : linear wavelength dispersion (nm per pixel)
    /// @param wavelengthCenter_ : central wavelength (nm)
    /// @param minFiberId : minimum fiberId value
    /// @param maxFiberId : maximum fiberId value
    /// @param height : height of detector (wavelength dimension; pixel)
    /// @param buffer : fraction of expected wavelength range by which to expand
    ///                 the wavelength range in the polynomials; this accounts
    ///                 for distortion or small inaccuracies in the dispersion.
    GlobalDetectorModelScaling(
        double fiberPitch_,
        double dispersion_,
        double wavelengthCenter_,
        int minFiberId_,
        int maxFiberId_,
        std::size_t height_,
        float buffer_=0.05
    ) : fiberPitch(std::abs(fiberPitch_)), dispersion(dispersion_), wavelengthCenter(wavelengthCenter_),
        minFiberId(minFiberId_), maxFiberId(maxFiberId_), height(height_), buffer(buffer_) {}

    virtual ~GlobalDetectorModelScaling() {}
    GlobalDetectorModelScaling(GlobalDetectorModelScaling const&) = default;
    GlobalDetectorModelScaling(GlobalDetectorModelScaling &&) = default;
    GlobalDetectorModelScaling & operator=(GlobalDetectorModelScaling const&) = default;
    GlobalDetectorModelScaling & operator=(GlobalDetectorModelScaling &&) = default;

    //@{
    /// Convert fiberId,wavelength to xi,eta in pixels
    ///
    /// The scaling is applied; there may be an offset to the appropriate
    /// position on the detector.
    ///
    /// @param fiberId : fiber identifier
    /// @param wavelength : wavelength (nm)
    /// @returns xi,eta position
    lsst::geom::Point2D operator()(int fiberId, double wavelength) const;
    ndarray::Array<double, 2, 1> operator()(
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& wavelength
    ) const;
    //@}

    /// Convert xi,eta to approximate fiberId,wavelength
    ///
    /// This is useful for identifying troublesome xi,eta points.
    /// @param xiEta : xi,eta position
    /// @returns fiberId,wavelength values
    lsst::geom::Point2D inverse(lsst::geom::Point2D const& xiEta) const;

    /// Get range of xi,eta values
    ///
    /// All fiberId,wavelength values for a detector should fall in this range;
    /// it may be used for scaling polynomial inputs.
    lsst::geom::Box2D getRange() const;

    friend std::ostream& operator<<(std::ostream& os, GlobalDetectorModelScaling const& model);
};


/// Mapping from fiberId to fiberIndex
///
/// fiberId (int) are identifiers assigned to fibers. fiberIndex (std::size_t)
/// is the position in an array of that fiber. This class therefore provides an
/// array index for a particular fiberId.
///
/// Given the same input fiberIds, the same mapping should result.
class FiberMap {
    using Map = std::unordered_map<int, std::size_t>;
  public:

    /// Ctor
    ///
    /// @param fiberId : fiber identifiers
    FiberMap(ndarray::Array<int, 1, 1> const& fiberId);

    virtual ~FiberMap() {}
    FiberMap(FiberMap const&) = default;
    FiberMap(FiberMap &&) = default;
    FiberMap & operator=(FiberMap const&) = default;
    FiberMap & operator=(FiberMap &&) = default;

    /// Number of fibers in mapping
    std::size_t size() const { return _map.size(); }

    //@{
    /// Map fiberId to fiberIndex
    std::size_t operator()(int fiberId) const { return _map.at(fiberId); }
    ndarray::Array<std::size_t, 1, 1> operator()(ndarray::Array<int, 1, 1> const& fiberId) const;
    //@}

    //@{
    /// Iteration
    Map::iterator begin() { return _map.begin(); }
    Map::const_iterator begin() const { return _map.begin(); }
    Map::iterator end() { return _map.end(); }
    Map::const_iterator end() const { return _map.end(); }
    //@}

    friend std::ostream& operator<<(std::ostream& os, FiberMap const& fiberMap);

  private:
    std::unordered_map<int, std::size_t> _map;
};


/// Model for transforming fiberId,wavelength to x,y on the detector
///
/// The model assumes that fibers are inserted in the slit in numeric order,
/// with a constant spacing between, but individual fiber offsets are applied
/// to account for any small deviations from this assumption. The wavelength
/// is linear with distance from the slit.
///
/// A two-dimensional polynomial distortion field is applied to this simple
/// setup to account for the camera optics. We allow for the detector to be
/// comprised of two discrete elements with a relative offset and rotation.
///
/// The model parameters are:
/// * scaling: these parameters convert fiberId and wavelength to nominal
///   pixels. It is not necessary to get these values exactly correct, but they
///   set the scaling for the distortion polynomials and the spatial and
///   spectral offsets.
/// * xDistortion, yDistortion: 2D polynomial distortion field coefficients.
/// * highCcd: 2D affine transformation coefficients for the high-fiberId
///   CCD.
/// * spatialOffset, spectralOffset: per-fiber offsets in the spatial and
///       spectral dimensions.
class GlobalDetectorModel : public lsst::afw::table::io::Persistable {
  public:
    using Polynomial = math::NormalizedPolynomial2<double>;

    /// Ctor
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param fiberId : fiberId values for fibers
    /// @param scaling : scaling of fiberId,wavelength to xi,eta
    /// @param fiberCenter : central fiberId value; for separating left and right CCDs
    /// @param xDistortion : distortion field parameters for x
    /// @param yDistortion : distortion field parameters for y
    /// @param highCcd : affine transformation parameters for the high-fiberId CCD
    /// @param spatialOffsets : slit offsets in the spatial dimension
    /// @param spectralOffsets : slit offsets in the spectral dimension
    GlobalDetectorModel(
        int distortionOrder,
        ndarray::Array<int, 1, 1> const& fiberId,
        GlobalDetectorModelScaling const& scaling,
        float fiberCenter,
        ndarray::Array<double, 1, 1> const& xDistortion,
        ndarray::Array<double, 1, 1> const& yDistortion,
        ndarray::Array<double, 1, 1> const& highCcd,
        ndarray::Array<double, 1, 1> const& spatialOffsets=ndarray::Array<double, 1, 1>(),
        ndarray::Array<double, 1, 1> const& spectralOffsets=ndarray::Array<double, 1, 1>()
    );

    /// Ctor
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param fiberMap : mapping for fiberId to fiberIndex
    /// @param scaling : scaling of fiberId,wavelength to xi,eta
    /// @param fiberCenter : central fiberId value; for separating left and right CCDs
    /// @param xDistortion : distortion field parameters for x
    /// @param yDistortion : distortion field parameters for y
    /// @param highCcd : affine transformation parameters for the high-fiberId CCD
    /// @param spatialOffsets : slit offsets in the spatial dimension
    /// @param spectralOffsets : slit offsets in the spectral dimension
    GlobalDetectorModel(
        int distortionOrder,
        FiberMap const& fiberMap,
        GlobalDetectorModelScaling const& scaling,
        float fiberCenter,
        ndarray::Array<double, 1, 1> const& xDistortion,
        ndarray::Array<double, 1, 1> const& yDistortion,
        ndarray::Array<double, 1, 1> const& highCcd,
        ndarray::Array<double, 1, 1> const& spatialOffsets,
        ndarray::Array<double, 1, 1> const& spectralOffsets
    );

    virtual ~GlobalDetectorModel() {}
    GlobalDetectorModel(GlobalDetectorModel const&) = default;
    GlobalDetectorModel(GlobalDetectorModel &&) = default;
    GlobalDetectorModel & operator=(GlobalDetectorModel const&) = default;
    GlobalDetectorModel & operator=(GlobalDetectorModel &&) = default;

    //@{
    /// Evaluate the model
    ///
    /// @param fiberId : fiber identifier
    /// @param wavelength : wavelength (nm)
    /// @param fiberIndex : index for fiber
    /// @param onHighCcd : whether fiber is on high-fiberId CCD
    /// @return x,y position on detector
    lsst::geom::Point2D operator()(int fiberId, double wavelength) const {
        return operator()(getScaling()(fiberId, wavelength), getFiberIndex(fiberId), getOnHighCcd(fiberId));
    }
    ndarray::Array<double, 2, 1> operator()(
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& wavelength
    ) const {
        return operator()(getScaling()(fiberId, wavelength), getFiberIndex(fiberId), getOnHighCcd(fiberId));
    }
    lsst::geom::Point2D operator()(int fiberId, double wavelength, std::size_t fiberIndex) const {
        return operator()(getScaling()(fiberId, wavelength), fiberIndex, getOnHighCcd(fiberId));
    }
    ndarray::Array<double, 2, 1> operator()(
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& wavelength,
        ndarray::Array<std::size_t, 1, 1> const& fiberIndex
    ) const {
        return operator()(getScaling()(fiberId, wavelength), fiberIndex, getOnHighCcd(fiberId));
    }
    lsst::geom::Point2D operator()(lsst::geom::Point2D const& xiEta, std::size_t fiberIndex,
                                   bool onHighCcd) const;
    ndarray::Array<double, 2, 1> operator()(
        ndarray::Array<double, 2, 1> const& xiEta,
        ndarray::Array<std::size_t, 1, 1> const& fiberIndex,
        ndarray::Array<bool, 1, 1> const& onHighCcd
    ) const;
    //@}

    /// Calculate the design matrix for the distortion polynomials
    ///
    /// The design matrix is the "X" matrix in the classic linear least-squares
    /// equation: X^T*X*beta = X^T*y (where beta are the parameters, y are the
    /// measurements, and "*" denotes matrix multiplication). Rows correspond to
    /// the input data points, and columns correspond to the individual
    /// polynomial terms. The dimensions of the matrix are (Ndata, Ndistortion),
    /// where Ndata is the number of data points provided, and Ndistortion is
    /// the number of distortion polynomial terms.
    ///
    /// @param distortionOrder : polynomial order for distortion
    /// @param xiEtaRange : range for xi,eta values
    /// @param xiEta : xi,eta values for data points
    /// @returns design matrix
    static ndarray::Array<double, 2, 1> calculateDesignMatrix(
        int distortionOrder,
        lsst::geom::Box2D const& xiEtaRange,
        ndarray::Array<double, 2, 1> const& xiEta
    );

    //@{
    /// Calculate chi^2 for a particular dataset
    ///
    /// @param xiEta : xi,eta (scaled fiberId,wavelength) values for arc lines
    /// @param fiberId : fiberId values for arc lines
    /// @param wavelength : wavelength values for arc lines
    /// @param xx : x values for arc lines
    /// @param yy : y values for arc lines
    /// @param xErr : error in x values for arc lines
    /// @param yErr : error in y values for arc lines
    /// @param good : whether value should be used in the fit
    /// @param sysErr : systematic error to add in quadrature
    /// @return chi2 and number of degrees of freedom
    std::pair<double, std::size_t> calculateChi2(
        ndarray::Array<double, 2, 1> const& xiEta,
        ndarray::Array<std::size_t, 1, 1> const& fiberIndex,
        ndarray::Array<bool, 1, 1> const& onHighCcd,
        ndarray::Array<double, 1, 1> const& xx,
        ndarray::Array<double, 1, 1> const& yy,
        ndarray::Array<double, 1, 1> const& xErr,
        ndarray::Array<double, 1, 1> const& yErr,
        ndarray::Array<bool, 1, 1> const& good=ndarray::Array<bool, 1, 1>(),
        float sysErr=0.0
    ) const;
    std::pair<double, std::size_t> calculateChi2(
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& wavelength,
        ndarray::Array<double, 1, 1> const& xx,
        ndarray::Array<double, 1, 1> const& yy,
        ndarray::Array<double, 1, 1> const& xErr,
        ndarray::Array<double, 1, 1> const& yErr,
        ndarray::Array<bool, 1, 1> const& good=ndarray::Array<bool, 1, 1>(),
        float sysErr=0.0
    ) const {
        return calculateChi2(getScaling()(fiberId, wavelength), getFiberIndex(fiberId), getOnHighCcd(fiberId),
                             xx, yy, xErr, yErr, good, sysErr);
    }
    //@}

    //@{
    /// Measure slit offsets for data
    ///
    /// After applying a distortion, individual fibers may be offset in x or y.
    /// This method measures the mean offset for each fiber, and sets it in the
    /// model.
    ///
    /// @param xiEta : xi,eta values for data points
    /// @param fiberIndex : fiber index (mapped fiberId) for each data point
    /// @param xx : x coordinate values for data points
    /// @param yy : y coordinate values for data points
    /// @param xErr : x coordinate error values for data points
    /// @param yErr : y coordinate error values for data points
    /// @returns spatial and spectral offsets for each fiber
    ndarray::Array<double, 2, 1> measureSlitOffsets(
        ndarray::Array<double, 2, 1> const& xiEta,
        ndarray::Array<std::size_t, 1, 1> const& fiberIndex,
        ndarray::Array<bool, 1, 1> const& onHighCcd,
        ndarray::Array<double, 1, 1> const& xx,
        ndarray::Array<double, 1, 1> const& yy,
        ndarray::Array<double, 1, 1> const& xErr,
        ndarray::Array<double, 1, 1> const& yErr
    );
    ndarray::Array<double, 2, 1> measureSlitOffsets(
        ndarray::Array<int, 1, 1> const& fiberId,
        ndarray::Array<double, 1, 1> const& wavelength,
        ndarray::Array<double, 1, 1> const& xx,
        ndarray::Array<double, 1, 1> const& yy,
        ndarray::Array<double, 1, 1> const& xErr,
        ndarray::Array<double, 1, 1> const& yErr
    ) {
        return measureSlitOffsets(getScaling()(fiberId, wavelength), getFiberIndex(fiberId),
                                  getOnHighCcd(fiberId), xx, yy, xErr, yErr);
    }
    //@}

    //@{
    /// Return the total number of parameters for the model
    static std::size_t getNumParameters(int distortionOrder, std::size_t numFibers) {
        return 6 + 2*GlobalDetectorModel::getNumDistortion(distortionOrder) + 2*numFibers;
    }
    std::size_t getNumParameters() const {
        return getNumParameters(getDistortionOrder(), getNumFibers());
    }
    //@}

    /// Return the number of fibers
    std::size_t getNumFibers() const { return _fiberMap.size(); }

    //@{
    /// Accessors
    int getDistortionOrder() const { return _distortionOrder; }
    ndarray::Array<int, 1, 1> getFiberId() const;
    GlobalDetectorModelScaling getScaling() const { return _scaling; }
    double getFiberPitch() const { return _scaling.fiberPitch; }
    double getDispersion() const { return _scaling.dispersion; }
    double getWavelengthCenter() const { return _scaling.wavelengthCenter; }
    int getHeight() const { return _scaling.height; }
    float getBuffer() const { return _scaling.buffer; }
    float getFiberCenter() const { return _fiberCenter; }
    Polynomial const& getXDistortion() const { return _xDistortion; }
    Polynomial const& getYDistortion() const { return _yDistortion; }
    lsst::geom::AffineTransform getHighCcd() const { return _highCcd; }
    double getSpatialOffset(std::size_t index) const { return _spatialOffsets[index]; }
    double getSpectralOffset(std::size_t index) const { return _spectralOffsets[index]; }
    ndarray::Array<double, 1, 1> const& getSpatialOffsets() const { return _spatialOffsets; }
    ndarray::Array<double, 1, 1> & getSpatialOffsets() { return _spatialOffsets; }
    ndarray::Array<double, 1, 1> const& getSpectralOffsets() const { return _spectralOffsets; }
    ndarray::Array<double, 1, 1> & getSpectralOffsets() { return _spectralOffsets; }
    //@}

    //@{
    /// Return the distortion polynomial coefficients
    ndarray::Array<double, 1, 1> getXCoefficients() const;
    ndarray::Array<double, 1, 1> getYCoefficients() const;
    //@}

    /// Return the right ccd affine transform coefficients
    ndarray::Array<double, 1, 1> getHighCcdCoefficients() const;

    /// Generate coefficients for the right CCD affine transform
    ///
    /// @param x : Offset in x
    /// @param y : Offset in y
    /// @param xx : Term in x for the x coordinate
    /// @param xy : Term in y for the x coordinate
    /// @param yx : Term in x for the y coordinate
    /// @param yy : Term in y for the y coordinate
    /// @return coefficient array
    static ndarray::Array<double, 1, 1> makeHighCcdCoefficients(
        double x, double y,
        double xx, double xy,
        double yx, double yy
    );

    //@{
    /// Return number of distortion parameters per axis
    static std::size_t getNumDistortion(int distortionOrder) {
        return Polynomial::nParametersFromOrder(distortionOrder);
    }
    std::size_t getNumDistortion() const {
        return getNumDistortion(getDistortionOrder());
    }
    //@}

    //@{
    /// Map fiberId to fiberIndex
    ///
    /// @param fiberId : fiber index
    /// @returns fiber index
    std::size_t getFiberIndex(int fiberId) const {
        return _fiberMap(fiberId);
    }
    ndarray::Array<std::size_t, 1, 1> getFiberIndex(ndarray::Array<int, 1, 1> const& fiberId) const {
        return _fiberMap(fiberId);
    }
    //@}

    //@{
    /// Return whether the fibers are on the high-fiberId CCD
    bool getOnHighCcd(int fiberId) const {
        return fiberId > _fiberCenter;
    }
    ndarray::Array<bool, 1, 1> getOnHighCcd(ndarray::Array<int, 1, 1> const& fiberId) const;
    //@}


    bool isPersistable() const noexcept { return true; }

    class Factory;

  protected:
    friend std::ostream& operator<<(std::ostream& os, GlobalDetectorModel const& model);

    std::string getPersistenceName() const { return "GlobalDetectorModel"; }
    std::string getPythonModule() const { return "pfs.drp.stella"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const;

  private:
    // Configuration
    int _distortionOrder;  // Order for distortion polynomials
    FiberMap _fiberMap;  // Mapping from fiberId to fiberIndex

    // Calculation parameters
    GlobalDetectorModelScaling _scaling;  // Scaling of fiberId,wavelength to xi,eta
    float _fiberCenter;  // central fiberId value; for separating low- and high-fiberId CCDs
    Polynomial _xDistortion;  // distortion polynomial in x
    Polynomial _yDistortion;  // distortion polynomial in y
    lsst::geom::AffineTransform _highCcd;  // transformation for high-fiberId CCD
    ndarray::Array<double, 1, 1> _spatialOffsets;  // fiber offsets in the spatial dimension
    ndarray::Array<double, 1, 1> _spectralOffsets;  // fiber offsets in the spectral dimension
};


}}}  // namespace pfs::drp::stella

#endif  // include guard
