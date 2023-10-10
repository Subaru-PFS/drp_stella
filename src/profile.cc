#include "ndarray.h"

#include "lsst/log/Log.h"

#include "pfs/drp/stella/profile.h"

#include "pfs/drp/stella/math/quartiles.h"
#include "pfs/drp/stella/utils/checkSize.h"

#include "pfs/drp/stella/profile.h"
#include "pfs/drp/stella/math/SparseSquareMatrix.h"
#include "pfs/drp/stella/math/solveLeastSquares.h"
#include "pfs/drp/stella/utils/math.h"

namespace pfs {
namespace drp {
namespace stella {


namespace {

LOG_LOGGER _log = LOG_GET("pfs.drp.stella.profile");


// The following is used to represent bound methods as a callable
template<typename ReturnT, class ClassT, typename... Args>
struct BoundMethod {
    using Function = ReturnT (ClassT::*)(Args...);
    BoundMethod(ClassT & obj_, Function func_) : obj(obj_), func(func_) {}
    ClassT & obj;
    Function func;
    ReturnT operator()(Args&... args) const { return (obj.*func)(args...); }
};
template<typename ReturnT, class ClassT, typename... Args>
struct BoundConstMethod {
    using Function = ReturnT (ClassT::*)(Args...) const;
    BoundConstMethod(ClassT const& obj_, Function func_) : obj(obj_), func(func_) {}
    ClassT const& obj;
    Function func;
    ReturnT operator()(Args&... args) const { return (obj.*func)(args...); }
};

template<typename ReturnT, class ClassT, class... Args>
BoundMethod<ReturnT, ClassT, Args...> bind_method(
    ClassT & obj,
    ReturnT (ClassT::* func)(Args...)
) {
    return BoundMethod<ReturnT, ClassT, Args...>(obj, func);
}
template<typename ReturnT, class ClassT, class... Args>
BoundConstMethod<ReturnT, ClassT, Args...> bind_method(
    ClassT const& obj,
    ReturnT (ClassT::* func)(Args...) const
) {
    return BoundConstMethod<ReturnT, ClassT, Args...>(obj, func);
}


// Class to support the simultaneous fit of multiple fiber profiles
//
// This class constructs and solves the least-squares matrix problem.
// Model parameters are the amplitudes of the fiber profiles in each
// over-sampled pixel. In order for this to be a linear problem, the user
// needs to supply the spectrum of each fiber.
class SwathProfileBuilder {
  public:

    // Constructor
    //
    // @param numFibers : Number of fibers
    // @param oversample : Oversampling factor
    // @param radius : Radius of fiber profile, in pixels
    SwathProfileBuilder(
        std::size_t numFibers,
        int oversample,
        int radius
    ) : _oversample(oversample),
        _radius(radius),
        _profileSize(2*std::size_t((radius + 1)*oversample + 0.5) + 1),
        _profileCenter((radius + 1)*oversample + 0.5),
        _numFibers(numFibers),
        _numParameters(numFibers*_profileSize),
        _matrix(_numParameters),
        _vector(ndarray::allocate(_numParameters)),
        _isZero(ndarray::allocate(_numParameters)) {
            reset();
        }

    std::size_t getNumParameters() const { return _numParameters; }

    // Reset for a new swath
    void reset() {
        _matrix.reset();
        _vector.deep() = 0;
        _isZero.deep() = true;
    }

    // Accumulate the data from a single image
    //
    // @param image : Image to accumulate
    // @param centers : Fiber centers for each row, fiber
    // @param spectra : Spectrum for each fiber
    // @param yMin : Minimum row to accumulate
    // @param yMax : Maximum row to accumulate
    // @param rejected : Rejected pixels
    void accumulateImage(
        lsst::afw::image::MaskedImage<float> const& image,
        ndarray::Array<double, 2, 1> const& centers,
        ndarray::Array<float, 2, 1> const& spectra,
        int yMin,
        int yMax,
        ndarray::Array<bool, 2, 1> const& rejected
    ) {
        assert(centers.getShape() == spectra.getShape());
        assert(rejected.getShape()[1] == std::size_t(image.getWidth()));
        assert(rejected.getShape()[0] == std::size_t(yMax - yMin));
        std::size_t ii = 0;
        for (int yy = yMin; yy < yMax; ++yy, ++ii) {
            iterateRow(image, yy, centers[ndarray::view()(yy)],
                       bind_method<void>(*this, &SwathProfileBuilder::accumulatePixel),
                       spectra[ndarray::view()(yy)], rejected[ii]);
        }
    }

    // Accumulate the data from a single pixel
    //
    // @param xx : Column of pixel
    // @param iter : Iterator to pixel (we only use the image and variance; the
    //     mask is applied through the 'rejected' parameter)
    // @param centers : Fiber centers in this row for each fiber
    // @param norm : Spectrum values in this row for each fiber
    // @param left : Index of first fiber to consider (inclusive)
    // @param right : Index of last fiber to consider (exclusive)
    // @param rejected : Rejected pixels for this row
    void accumulatePixel(
        int xx,
        int yy,
        typename lsst::afw::image::MaskedImage<float>::x_iterator iter,
        ndarray::ArrayRef<double, 1, 0> const& centers,
        std::size_t left,
        std::size_t right,
        ndarray::ArrayRef<float, 1, 0> const& norm,
        ndarray::ArrayRef<bool, 1, 1> const& rejected
    ) {
        assert(norm.size() == centers.size());
        assert(std::size_t(xx) < rejected.size());
        if (rejected[xx]) return;
        float const pixelValue = iter.image();
        assert(iter.variance() > 0);
        float const invVariance = 1.0/iter.variance();

        // Iterate over fibers of interest
#if 1
        // Linear interpolation
        for (std::size_t ii = left; ii < right; ++ii) {
            double const iModel = norm[ii];
            if (iModel == 0) continue;
            std::size_t iLower;
            double iLowerFrac;
            std::tie(iLower, iLowerFrac) = getProfileInterpolation(ii, xx, centers[ii]);
            double const iLowerModel = iModel*iLowerFrac;
            std::size_t const iUpper = iLower + 1;
            double const iUpperModel = iModel*(1.0 - iLowerFrac);

            _matrix.add(iLower, iLower, std::pow(iLowerModel, 2)*invVariance);
            _matrix.add(iUpper, iUpper, std::pow(iUpperModel, 2)*invVariance);
            _matrix.add(iLower, iUpper, iLowerModel*iUpperModel*invVariance);

            _isZero[iLower] &= iLowerFrac == 0;
            _isZero[iUpper] &= iLowerFrac == 1;
            _vector[iLower] += iLowerModel*pixelValue*invVariance;
            _vector[iUpper] += iUpperModel*pixelValue*invVariance;

            for (std::size_t jj = ii + 1; jj < right; ++jj) {
                double const jModel = norm[jj];
                if (jModel == 0) continue;
                std::size_t jLower;
                double jLowerFrac;
                std::tie(jLower, jLowerFrac) = getProfileInterpolation(jj, xx, centers[jj]);
                double const jLowerModel = jModel*jLowerFrac;
                std::size_t const jUpper = jLower + 1;
                double const jUpperModel = jModel*(1.0 - jLowerFrac);

                _matrix.add(iLower, jLower, iLowerModel*jLowerModel*invVariance);
                _matrix.add(iLower, jUpper, iLowerModel*jUpperModel*invVariance);
                _matrix.add(iUpper, jLower, iUpperModel*jLowerModel*invVariance);
                _matrix.add(iUpper, jUpper, iUpperModel*jUpperModel*invVariance);
            }
        }
#else
        // Nearest-neighbour interpolation
        // This is much simpler than linear interpolation, but it is subject to imprecisions
        for (std::size_t ii = left; ii < right; ++ii) {
            double const iModel = norm[ii];
            if (iModel == 0) continue;
            std::size_t const iPixel = getProfileNearest(ii, xx, centers[ii]);

            _matrix.add(iPixel, iPixel, std::pow(iModel, 2)*invVariance);
            _isZero[iPixel] = false;
            _vector[iPixel] += iModel*pixelValue*invVariance;

            for (std::size_t jj = ii + 1; jj < right; ++jj) {
                double const jModel = norm[jj];
                if (jModel == 0) continue;
                std::size_t const jPixel = getProfileNearest(jj, xx, centers[jj]);
                _matrix.add(iPixel, jPixel, iModel*jModel*invVariance);

                _isZero[jPixel] = false;
            }
        }
#endif

    }

    // Iterate over a row of an image, calling a function for each pixel
    //
    // @param image : Image to iterate over
    // @param yy : Row to iterate over
    // @param centers : Fiber centers in this row for each fiber
    // @param func : Function to call for each pixel
    // @param args : Arguments to pass to func
    template<typename ImageT, typename Function, class... Args>
    void iterateRow(
        ImageT const& image,
        int yy,
        ndarray::ArrayRef<double, 1, 0> const& centers,
        Function func,
        Args... args
    ) const {
        int const width = image.getWidth();
        int const height = image.getHeight();
        utils::checkSize(centers.size(), _numFibers, "centers");
        if (yy < 0 || yy >= height) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "Row not in image");
        }

        // Determine position and bounds of each fiber
        ndarray::Array<int, 1, 1> lower = ndarray::allocate(_numFibers);  // inclusive
        ndarray::Array<int, 1, 1> upper = ndarray::allocate(_numFibers);  // exclusive
        for (std::size_t fiberIndex = 0; fiberIndex < _numFibers; ++fiberIndex) {
            if (fiberIndex > 0 && centers[fiberIndex] < centers[fiberIndex - 1]) {
                // Having the fiberIds sorted in order from left to right on the image makes
                // things so much easier, so we insist upon it.
                throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError,
                                  "fiberIds aren't sorted from left to right");
            }
            lower[fiberIndex] = std::max(0, int(std::ceil(centers[fiberIndex] - _radius - 1)));
            upper[fiberIndex] = std::min(width, int(std::floor(centers[fiberIndex] + _radius + 1)) + 1);
        }

        // Iterate over pixels in the row
        std::size_t left = 0;  // fiber index of lower bound of consideration for this pixel (inclusive)
        std::size_t right = 1;  // fiber index of upper bound of consideration for this pixel (exclusive)
        int xx = lower[0];  // column for pixel of interest
        auto const stop = image.row_begin(yy) + upper[_numFibers - 1];
        for (auto ptr = image.row_begin(yy) + xx; ptr != stop; ++ptr, ++xx) {
            // Which fibers are we dealing with?
            // pixel:                X
            // fiber 0: |-----------|
            // fiber 1:     |-----------|
            // fiber 2:         |-----------|
            // fiber 3:             |-----------|
            // fiber 4:                 |-----------|
            // Our pixel (X, at top) overlaps 1,2,3; therefore get left=1 (inclusive), right=4 (exclusive)
            while (left < _numFibers && xx >= upper[left]) ++left;
            while (right < _numFibers && xx >= lower[right]) ++right;

            func(xx, yy, ptr, centers, left, right, args...);
        }
    }

    //@{
    // Solve the matrix equation
    void solve(ndarray::Array<double, 1, 1> & solution, float matrixTol=1.0e-4) {
        // Add in the symmetric elements that we didn't bother to accumulate
        _matrix.symmetrize();

        // Deal with singular values
        for (std::size_t ii = 0; ii < getNumParameters(); ++ii) {
            if (_isZero[ii]) {
                _matrix.add(ii, ii, 1.0);
                assert(_vector[ii] == 0.0);  // or we've done something wrong
            }
        }

        solution.deep() = std::numeric_limits<double>::quiet_NaN();

        // Solve the matrix equation
        // We use a preconditioned conjugate gradient solver, which is much less
        // picky about the matrix being close to singular.
        using Matrix = math::NonsymmetricSparseSquareMatrix::Matrix;
        using Solver = Eigen::ConjugateGradient<
            Matrix, Eigen::Lower|Eigen::Upper, Eigen::DiagonalPreconditioner<double>
        >;
        Solver solver;
        solver.setMaxIterations(getNumParameters()*10);
        solver.setTolerance(matrixTol);
        _matrix.solve(solution, _vector, solver);

        for (std::size_t ii = 0; ii < getNumParameters(); ++ii) {
            if (_isZero[ii]) {
                solution[ii] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    ndarray::Array<double, 1, 1> solve(float matrixTol=1.0e-4) {
        ndarray::Array<double, 1, 1> solution = ndarray::allocate(_numParameters);
        solve(solution, matrixTol);
        return solution;
    }
    //@}

    // Repackage the solution into a 2-d array
    //
    // @param solution : Solution array
    // @return profile and mask arrays
    auto repackageSolution(ndarray::Array<double, 1, 1> const& solution) {
        // Repackage the solution
        ndarray::Array<double, 2, 1> profile = ndarray::allocate(_numFibers, _profileSize);
        ndarray::Array<bool, 2, 1> mask = ndarray::allocate(_numFibers, _profileSize);
        std::size_t index = 0;
        for (std::size_t ff = 0; ff < _numFibers; ++ff) {
            for (std::size_t pp = 0; pp < _profileSize; ++pp, ++index) {
                if (_isZero[index]) {
                    profile[ff][pp] = NAN;
                    mask[ff][pp] = true;
                } else {
                    profile[ff][pp] = solution[index];
                    mask[ff][pp] = false;
                }
            }
        }
        return std::make_pair(profile, mask);
    }

    // Reject pixels that are too far from the model
    //
    // @param solution : Solution array
    // @param image : Image to fit (mask is unused: we use the rejected array instead)
    // @param centers : Fiber centers for each row, fiber
    // @param spectra : Spectra for each row, fiber
    // @param yMin : Minimum row to fit
    // @param yMax : Maximum row to fit (exclusive)
    // @param rejected : Array of pixels to reject (true=rejected)
    // @param rejThreshold : Threshold for rejecting pixels (in sigma)
    void reject(
        ndarray::Array<double, 1, 1> const& solution,
        lsst::afw::image::MaskedImage<float> const& image,
        ndarray::Array<double, 2, 1> const& centers,
        ndarray::Array<float, 2, 1> const& spectra,
        int yMin,
        int yMax,
        ndarray::Array<bool, 2, 1> rejected,
        float rejThreshold
    ) const {
        assert(centers.getShape() == spectra.getShape());
        assert(rejected.getShape()[0] == std::size_t(yMax - yMin));
        assert(rejected.getShape()[1] == std::size_t(image.getWidth()));
        std::size_t ii = 0;
        for (int yy = yMin; yy < yMax; ++yy, ++ii) {
            iterateRow(image, yy, centers[ndarray::view()(yy)],
                       bind_method<void>(*this, &SwathProfileBuilder::applyPixelRejection),
                       rejected[ii], spectra[ndarray::view()(yy)], solution, rejThreshold);
        }
    }

    // Test a single pixel for rejection
    //
    // @param xx : Column of pixel
    // @param iter : Iterator to pixel
    // @param centers : Fiber centers for each row, fiber
    // @param left : Fiber index of lower bound of consideration for this pixel (inclusive)
    // @param right : Fiber index of upper bound of consideration for this pixel (exclusive)
    // @param rejected : Array of pixels to reject (true=rejected)
    // @param norm : Normalization for each fiber
    // @param solution : Solution array
    // @param rejThreshold : Threshold for rejecting pixels (in sigma)
    void applyPixelRejection(
        int xx,
        int yy,
        typename lsst::afw::image::MaskedImage<float>::x_iterator iter,
        ndarray::ArrayRef<double, 1, 0> const& centers,
        std::size_t left,
        std::size_t right,
        ndarray::ArrayRef<bool, 1, 1> & rejected,
        ndarray::ArrayRef<float, 1, 0> const& norm,
        ndarray::Array<double, 1, 1> const& solution,
        float rejThreshold
    ) const {
        assert(centers.size() == norm.size());
        assert(std::size_t(xx) < rejected.size());
        float const image = iter.image();
        float const variance = iter.variance();
        double model = 0.0;
        for (std::size_t ii = left; ii < right; ++ii) {
            double const iModel = norm[ii];
            if (iModel == 0) continue;

#if 1
            // Linear interpolation
            std::size_t iLower;
            double iLowerFrac;
            std::tie(iLower, iLowerFrac) = getProfileInterpolation(ii, xx, centers[ii]);
            double const iLowerModel = iModel*iLowerFrac;
            std::size_t const iUpper = iLower + 1;
            double const iUpperModel = iModel*(1.0 - iLowerFrac);
            model += iLowerModel*solution[iLower] + iUpperModel*solution[iUpper];
#else
            // Nearest neighbor interpolation
            std::size_t const iPixel = getProfileNearest(ii, xx, centers[ii]);
            model += iModel*solution[iPixel];
#endif
        }
        float const diff = (image - model)/std::sqrt(variance);
        if (std::abs(diff) > rejThreshold) {
            rejected[xx] = true;
        }
    }

  private:
    // Indices:
    // profile[fiberIndex=0, position=0]
    // ...
    // profile[fiberIndex=0, position=profileSize-1]
    // profile[fiberIndex=1, position=0]
    // ... ...
    // profile[fiberIndex=numFibers-1, position=profileSize-1]

    // Convert pixel --> index
    //
    // Given a fiber (with associated center) and the pixel column, return the
    // appropriate model index.
    std::size_t getProfileIndex(std::size_t fiberIndex, int xx, double center) const {
        return getProfileIndex(fiberIndex, getProfilePosition(xx, center));
    }

    // Convert profile position --> index
    //
    // Given the integer index of a pixel in the oversampled space relative to
    // the start of the profile, return the appropriate model index.
    std::size_t getProfileIndex(std::size_t fiberIndex, std::ptrdiff_t profilePosition) const {
        // "profile index" is the index in the matrix for the profile position
        assert(fiberIndex < _numFibers);
        assert(profilePosition >= 0 && profilePosition < std::ptrdiff_t(_profileSize));
        return profilePosition + _profileSize*fiberIndex;
    }

    // Convert pixel --> profile position
    //
    // Given the pixel column and fiber center, return the integer index of the
    // pixel in the oversampled space, relative to the start of the profile.
    std::ptrdiff_t getProfilePosition(int xx, double center) const {
        // "profile position" is the integer index of the pixel in the
        // oversampled space, relative to the start of the profile (not the
        // center).
        std::ptrdiff_t profilePosition = std::round((xx - center)*_oversample) + _profileCenter;
        assert(profilePosition >= 0 && profilePosition < std::ptrdiff_t(_profileSize));
        return profilePosition;
    }

    // Linear interpolation between pixels: more accurate, but lots slower
    std::pair<std::size_t, double> getProfileInterpolation(
        std::size_t fiberIndex,
        int xx,
        double center
    ) const {
        double const actual = (xx - center)*_oversample;
        std::ptrdiff_t const truncated = std::ptrdiff_t(std::floor(actual));
        std::ptrdiff_t const position = truncated + _profileCenter;
        assert(position >= 0 && std::size_t(position) < _profileSize - 1);
        std::size_t index = getProfileIndex(fiberIndex, position);
        double const diff = actual - truncated;
        double const frac = 1 - diff;
        assert(frac >= 0 && frac <= 1.0);
        return std::make_pair(index, frac);
    }

    // Nearest-neighbor interpolation
    //
    // Given a fiber (with associated center) and the pixel column, return the
    // appropriate model index.
    std::size_t getProfileNearest(
        std::size_t fiberIndex,
        int xx,
        double center
    ) const {
        //          X
        // |-1-|-2-|-3-|-4-|-5-|
        double const actual = (xx - center)*_oversample;
        std::size_t const position = std::round(actual) + _profileCenter;
        return getProfileIndex(fiberIndex, position);
    }

    int _oversample;  // oversampling factor for profiles
    int _radius;  // radius of profiles
    std::size_t _profileSize;  // size of profiles = 2*radius*oversample + 1
    std::size_t _profileCenter;  // index of center of profile = radius*oversample
    std::size_t _numFibers;  // number of fibers
    std::size_t _numParameters;  // number of parameters = profileSize*numFibers
    // We use a non-symmetric matrix because the conjugate gradient solver
    // doesn't work with a self-adjoint view. So we just add the symmetric
    // elements manually.
    math::NonsymmetricSparseSquareMatrix _matrix;  // least-squares (Fisher) matrix
    ndarray::Array<double, 1, 1> _vector;  // least-squares (right-hand side) vector
    ndarray::Array<bool, 1, 1> _isZero;  // any zero matrix elements for this parameter?
};

}  // anonymous namespace


std::pair<ndarray::Array<double, 2, 1>, ndarray::Array<bool, 2, 1>>
fitSwathProfiles(
    std::vector<lsst::afw::image::MaskedImage<float>> const& images,
    std::vector<ndarray::Array<double, 2, 1>> const& centers,
    std::vector<ndarray::Array<float, 2, 1>> const& spectra,
    ndarray::Array<int, 1, 1> const& fiberIds,
    int yMin,
    int yMax,
    lsst::afw::image::MaskPixel badBitMask,
    int oversample,
    int radius,
    int rejIter,
    float rejThresh,
    float matrixTol
) {
    std::size_t const num = images.size();
    if (num == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "No images to fit");
    }
    if (fiberIds.size() == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "No fibers to fit");
    }
    utils::checkSize(centers.size(), num, "images vs centers");
    utils::checkSize(spectra.size(), num, "images vs spectra");
    auto const dims = images[0].getDimensions();
    int const width = dims.getX();
    for (std::size_t ii = 1; ii < num; ++ii) {
        if (images[ii].getDimensions() != dims) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "Image dimension mismatch");
        }
        if (centers[ii].getShape() != ndarray::makeVector(fiberIds.size(), std::size_t(dims.getY()))) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "Centers dimension mismatch");
        }
    }
    if (yMax < yMin) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "No pixels to fit");
    }
    if (yMin < 0 || yMin >= images[0].getHeight() || yMax < 0 || yMax >= images[0].getHeight()) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "yMin/yMax outside of image");
    }

    // Reject pixels based on mask
    ndarray::Array<bool, 3, 1> rejected = ndarray::allocate(num, yMax - yMin, width);
    for (std::size_t ii = 0; ii < num; ++ii) {
        std::size_t jj = 0;
        for (int yy = yMin; yy < yMax; ++yy, ++jj) {
            auto mask = images[ii].getMask()->row_begin(yy);
            auto rej = rejected[ii][jj].begin();
            for (int xx = 0; xx < width; ++xx, ++mask, ++rej) {
                *rej = (*mask & badBitMask) != 0;
            }
        }
    }

    SwathProfileBuilder builder{fiberIds.size(), oversample, radius};

    for (int iter = 0; iter < rejIter; ++iter) {
        // Solve for the profiles
        builder.reset();
        LOGL_DEBUG(_log, "Fitting profile for rows %d-%d: iteration %d", yMin, yMax, iter);
        for (std::size_t ii = 0; ii < images.size(); ++ii) {
            LOGL_DEBUG(_log, "    Accumulating image %d", ii);
            builder.accumulateImage(images[ii], centers[ii], spectra[ii], yMin, yMax, rejected[ii]);
        }
        auto const solution = builder.solve(matrixTol);

        // Reject bad pixels
        LOGL_DEBUG(_log, "Rejecting pixels for rows %d-%d: iteration %d", yMin, yMax, iter);
        for (std::size_t ii = 0; ii < images.size(); ++ii) {
            builder.reject(solution, images[ii], centers[ii], spectra[ii], yMin, yMax,
                           rejected[ii], rejThresh);
        }
    }
    // Final solution after iteration
    builder.reset();
    LOGL_DEBUG(_log, "Fitting profile for rows %d-%d: final iteration", yMin, yMax);
    for (std::size_t ii = 0; ii < images.size(); ++ii) {
        LOGL_DEBUG(_log, "    Accumulating image %d", ii);
        builder.accumulateImage(images[ii], centers[ii], spectra[ii], yMin, yMax, rejected[ii]);
    }
    auto const solution = builder.solve(matrixTol);

    return builder.repackageSolution(solution);
}


ndarray::Array<double, 2, 1>
fitAmplitudes(
    lsst::afw::image::MaskedImage<float> const& image,
    ndarray::Array<double, 2, 1> const& centers,
    float sigma,
    lsst::afw::image::MaskPixel badBitMask,
    float nSigma
) {
    std::size_t const width = image.getWidth();
    std::size_t const height = image.getHeight();
    std::size_t const numFibers = centers.getShape()[0];
    utils::checkSize(centers.getShape()[1], height, "centers vs image height");
    double const invSigma = 1.0/sigma;
    auto const pixels = image.getImage()->getArray();
    auto const mask = image.getMask()->getArray();

    ndarray::Array<double, 2, 1> amplitudes = ndarray::allocate(centers.getShape());
    ndarray::Array<double, 2, 1> design = ndarray::allocate(width, numFibers);
    ndarray::Array<double, 1, 1> meas = ndarray::allocate(width);
    for (std::size_t yy = 0; yy < height; ++yy) {
        for (std::size_t xx = 0; xx < width; ++xx) {
            if (mask[yy][xx] & badBitMask) {
                design[ndarray::view(xx)()] = 0;
                meas[xx] = 0;
                continue;
            }
            meas[xx] = pixels[yy][xx];
            for (std::size_t ii = 0; ii < numFibers; ++ii) {
                double const radius = (xx - centers[ii][yy])*invSigma;
                design[xx][ii] = std::abs(radius) > nSigma ? 0.0 : std::exp(-0.5*std::pow(radius, 2));
            }
        }

        auto const solution = math::solveLeastSquaresDesign(design, meas);
        amplitudes[ndarray::view()(yy)] = solution;
    }
    return amplitudes;
}


std::pair<ndarray::Array<double, 1, 1>, ndarray::Array<bool, 1, 1>>
calculateSwathProfile(
    ndarray::Array<double, 2, 1> const& values,
    ndarray::Array<bool, 2, 1> const& mask,
    int rejIter,
    float rejThresh
) {
    utils::checkSize(mask.getShape(), values.getShape(), "mask");
    std::size_t const height = values.getShape()[0];
    std::size_t const width = values.getShape()[1];
    ndarray::Array<double, 1, 1> outValues = ndarray::allocate(width);
    ndarray::Array<bool, 1, 1> outMask = ndarray::allocate(width);

    for (std::size_t col = 0; col < width; ++col) {
        bool anyRejected = true;
        for (int ii = 0; ii < rejIter && anyRejected; ++ii) {
            double lower, median, upper;
            std::tie(lower, median, upper) = math::calculateQuartiles(values[ndarray::view()(col)],
                                                                      mask[ndarray::view()(col)]);
            double const threshold = 0.741*(upper - lower)*rejThresh;
            anyRejected = false;
            for (std::size_t row = 0; row < height; ++row) {
                if (!mask[row][col] && std::abs(values[row][col] - median) > threshold) {
                    mask[row][col] = true;
                    anyRejected = true;
                }
            }
        }
        double sum = 0.0;
        std::size_t num = 0;
        for (std::size_t row = 0; row < height; ++row) {
            if (!mask[row][col]) {
                sum += values[row][col];
                ++num;
            }
        }
        outValues[col] = num > 0 ? sum/num : 0.0;
        outMask[col] = num == 0 || !std::isfinite(sum);
    }

    return std::make_pair(outValues, outMask);
}


}}} // namespace pfs::drp::stella
