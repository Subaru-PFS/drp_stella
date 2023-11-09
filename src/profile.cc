#include <omp.h>

#include "ndarray.h"

#include "lsst/log/Log.h"

#include "pfs/drp/stella/profile.h"

#include "pfs/drp/stella/math/quartiles.h"
#include "pfs/drp/stella/utils/checkSize.h"

#include "pfs/drp/stella/profile.h"
#include "pfs/drp/stella/math/SparseSquareMatrix.h"
#include "pfs/drp/stella/math/solveLeastSquares.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/backgroundIndices.h"

namespace pfs {
namespace drp {
namespace stella {


#pragma GCC optimize("O0")

namespace {

LOG_LOGGER _log = LOG_GET("pfs.drp.stella.profile");


// The matrix equation we're solving for the profiles
struct ProfileEquation {
    // We use a non-symmetric matrix because the conjugate gradient solver
    // doesn't work with a self-adjoint view. So we just add the symmetric
    // elements manually.
    math::SymmetricSparseSquareMatrix matrix;  // least-squares (Fisher) matrix
    ndarray::Array<double, 1, 1> vector;  // least-squares (right-hand side) vector

    ProfileEquation(std::size_t numParameters) :
        matrix(numParameters), vector(ndarray::allocate(numParameters)) {
        reset();
    }

    ProfileEquation(ProfileEquation const& other) = delete;
    ProfileEquation(ProfileEquation && other) = default;
    ProfileEquation & operator=(ProfileEquation const& other) = delete;
    ProfileEquation & operator=(ProfileEquation && other) = default;
    ~ProfileEquation() = default;

    void reset() {
        matrix.reset();
        vector.deep() = 0;
    }

    void addDiagonal(std::size_t ii, double value) {
        matrix.add(ii, ii, value);
    }

    void addOffDiagonal(std::size_t ii, std::size_t jj, double value) {
        std::size_t const first = std::min(ii, jj);
        std::size_t const second = std::max(ii, jj);
        matrix.add(first, second, value);
    }

    void addVector(std::size_t ii, double value) {
        vector[ii] += value;
    }

    ProfileEquation & operator+=(ProfileEquation const& other) {
        matrix += other.matrix;
        asEigenArray(vector) += asEigenArray(other.vector);
        return *this;
    }

    ProfileEquation & operator-=(ProfileEquation const& other) {
        matrix -= other.matrix;
        asEigenArray(vector) -= asEigenArray(other.vector);
        return *this;
    }
};
#pragma omp declare reduction(+:ProfileEquation:omp_out += omp_in) \
    initializer(omp_priv = ProfileEquation(omp_orig.vector.size()))


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
    // @param numImages : Number of images
    // @param numFibers : Number of fibers
    // @param ySwaths : y positions of swath centers
    // @param oversample : Oversampling factor
    // @param radius : Radius of fiber profile, in pixels
    // @param sigma : Gaussian sigma of main fiber profile
    // @param dims : Dimensions of image
    // @param bgNum : Number of background pixels in each dimension
    SwathProfileBuilder(
        std::size_t numImages,
        std::size_t numFibers,
        ndarray::Array<float, 1, 1> const& ySwaths,
        int oversample,
        int radius,
        float sigma,
        lsst::geom::Extent2I const& dims,
        lsst::geom::Extent2I const& bgSize
    ) : _ySwaths(ySwaths),
        _oversample(oversample),
        _radius(radius),
        _invSigma(1.0/sigma),
        _profileSize(2*std::size_t(radius*oversample) + oversample + 1),
        _profileCenter((radius + 1)*oversample),
        _numImages(numImages),
        _numFibers(numFibers),
        _dims(dims),
        _bgSize(bgSize),
        _numProfileParameters(numFibers*_profileSize),
        _numSwathsParameters(_numProfileParameters*ySwaths.size()),
        _numBackgroundParameters(getNumBackgroundIndices(dims, bgSize)),
        _numParameters(_numSwathsParameters + numImages*_numBackgroundParameters + 1),
        _iGaussian(_numParameters - 1),
//        _gaussian(ndarray::allocate(_profileSize + oversample)),
        _bgIndex(calculateBackgroundIndices(dims, bgSize, _numSwathsParameters))
        {
#if 0
            double invSigma = 1.0/sigma;
            for (std::size_t ii = -(radius + 1); ii <= radius + 1; ++ii) {
                double const xx = (ii - _profileCenter)/_oversample;
                _gaussian[ii] = std::exp(-0.5*std::pow(xx*invSigma, 2));
            }
#endif
        }

    std::size_t getNumParameters() const { return _numParameters; }

    // Accumulate the data from a single image
    //
    // @param imageIndex : Index of image
    // @param image : Image to accumulate
    // @param centers : Fiber centers for each row, fiber
    // @param spectra : Spectrum for each fiber
    // @param rejected : Rejected pixels
    void accumulateImage(
        std::size_t imageIndex,
        lsst::afw::image::MaskedImage<float> const& image,
        ProfileEquation & equation,
        ndarray::Array<double, 2, 1> const& centers,
        ndarray::Array<float, 2, 1> const& spectra,
        ndarray::Array<bool, 2, 1> const& rejected
    ) const {
        assert(centers.getShape() == spectra.getShape());
        assert(rejected.getShape()[1] == std::size_t(image.getWidth()));
        assert(rejected.getShape()[0] == std::size_t(image.getHeight()));

        std::size_t const bgOffset = imageIndex*_numBackgroundParameters;  // additional background offset

//        std::cerr << "Image " << imageIndex << ": " << spectra[ndarray::view()(512)] << std::endl;

        #pragma omp parallel for reduction(+:equation) schedule(guided)
        for (int yy = 0; yy < image.getHeight(); ++yy) {
            // Protect ndarray reference counting
            // These are declared as pointers because I need to create them in the critical block
            // (which has its own scope, so anything declared inside doesn't exist outside), but they have
            // no default constructor so I can't declare them as values outside the critical block.
            // Also, I need to control when they are destructed, and declaring them as values would
            // cause them to be destructed outside a critical block.
            std::unique_ptr<ndarray::ArrayRef<double, 1, 0> const> cen;
            std::unique_ptr<ndarray::ArrayRef<float, 1, 0> const> norm;
            std::unique_ptr<ndarray::ArrayRef<bool, 1, 0> const> rej;
            #pragma omp critical  // protect ndarray reference counting
            {
                cen = std::make_unique<ndarray::ArrayRef<double, 1, 0>>(centers[ndarray::view()(yy)]);
                norm = std::make_unique<ndarray::ArrayRef<float, 1, 0>>(spectra[ndarray::view()(yy)]);
                rej = std::make_unique<ndarray::ArrayRef<bool, 1, 0>>(rejected[yy]);
            }

            ProfileEquation eqn(_numParameters);

            // Can't allow exceptions to leak out of the parallel region (they would also get around
            // our ndarray reference counting protection), so catch them here.
            try {
                auto const swathInterp = getSwathInterpolation(yy);
                iterateRow(
                    image, yy, *cen,
                    [&](int xx, int yy, typename lsst::afw::image::MaskedImage<float>::x_iterator iter,
                        ndarray::ArrayRef<double, 1, 0> const& centers,
                        std::size_t left,
                        std::size_t right
                    ) {
                        accumulatePixel(
                            xx, yy, iter, centers, left, right, eqn, *norm, *rej, swathInterp, bgOffset
                        );
                    }
                );
            } catch (std::exception const& exc) {
                std::cerr << "Caught exception: " << exc.what() << std::endl;
                std::terminate();
            } catch (...) {
                std::cerr << "Caught unknown exception" << std::endl;
                std::terminate();
            }

            #pragma omp critical  // protect ndarray reference counting
            {
                cen.reset();
                norm.reset();
                rej.reset();
            }

            equation += eqn;
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
    // @param swathInterp : Swath interpolation for this row
    // @param bgOffset : Offset into the background parameters
    void accumulatePixel(
        int xx,
        int yy,
        typename lsst::afw::image::MaskedImage<float>::x_iterator & iter,
        ndarray::ArrayRef<double, 1, 0> const& centers,
        std::size_t left,
        std::size_t right,
        ProfileEquation & equation,
        ndarray::ArrayRef<float, 1, 0> const& norm,
        ndarray::ArrayRef<bool, 1, 0> const& rejected,
        std::pair<std::pair<std::size_t, double>, std::pair<std::size_t, double>> const& swathInterp,
        std::size_t bgOffset
    ) const {
        assert(norm.size() == centers.size());
        assert(std::size_t(xx) < rejected.size());
        if (rejected[xx]) return;


        // if (yy < 497 || yy > 519) return;


        float const pixelValue = iter.image();
        assert(iter.variance() > 0);
        double const invVariance = 1.0; // /iter.variance();
        double const pixelTimesInvVariance = pixelValue*invVariance;
        double const diffValue = (pixelValue - (iter + 1).image())*invVariance;

#if 0
        auto const bgIndex = bgOffset + _bgIndex[yy][xx];
        equation.addDiagonal(bgIndex, invVariance);
        equation.addVector(bgIndex, pixelValue*invVariance);
#endif

        double gTerm = 0.0;  // sum of norm*gaussian; for vector[iGaussian] and matrix[iGaussian][bg]
        double ggTerm = 0.0;  // sum of (norm*gaussian)^2; for matrix[iGaussian][iGaussian]
        double dgTerm = 0.0;  // sum of norm*dgaussian; for matrix[iGaussian][iProfile]
        for (std::size_t ii = left; ii < right; ++ii) {
            double const iNorm = norm[ii];
            if (iNorm == 0) continue;

            double const normGaussian = iNorm*_gaussian(xx - centers[ii]);
            gTerm += normGaussian;
            ggTerm += std::pow(normGaussian, 2);
            dgTerm += normGaussian - iNorm*_gaussian(xx - centers[ii] + 1);
        }

        equation.addDiagonal(_iGaussian, ggTerm*invVariance);
        equation.addVector(_iGaussian, gTerm*pixelTimesInvVariance);
#if 0
        equation.addOffDiagonal(_iGaussian, bgIndex, gTerm*invVariance);
#endif

#if 1
        // Iterate over fibers of interest
        for (std::size_t ii = left; ii < right; ++ii) {
            double const iNorm = norm[ii];
            if (iNorm == 0) continue;

            // We're doing two interpolations here: one in the spatial dimension (profile)
            // and one in the spectral dimension (swath). That means we need to add terms
            // for all four combinations of the two interpolations.
            // The matrix is symmetric, so we only need to add elements for one of the upper/lower triangles;
            // we'll set the symmetric terms after everything's been accumulated.
            auto const iInterp = getSwathProfileInterpolation(swathInterp, ii, xx, centers[ii], iNorm);
            std::array<std::size_t, 4> const iIndex = std::get<0>(iInterp);
            std::array<double, 4> const iValue = std::get<1>(iInterp);
            int const prevPosition = std::get<2>(iInterp);
            int const nextPosition = prevPosition + 1;

//            if (yy == 512) std::cerr << "xx=" << xx << " ii=" << ii << " dx=" << (xx - centers[ii]) << " " << iIndex[0] << "," << iValue[0] << " " << iIndex[1] << "," << iValue[1] << " " << iIndex[2] << "," << iValue[2] << " " << iIndex[3] << "," << iValue[3] << " " << nextPosition << " " << (nextPosition + _oversample < _profileSize) << std::endl;

            bool const notEnd = nextPosition + _oversample < _profileSize;

            for (int iTerm = 0; iTerm < 4; ++iTerm) {
                std::size_t const iIndexPlus = iIndex[iTerm] + _oversample;

                equation.addOffDiagonal(iIndex[iTerm], _iGaussian, iValue[iTerm]*dgTerm*invVariance);
                equation.addDiagonal(iIndex[iTerm], 2*std::pow(iValue[iTerm], 2)*invVariance);
                equation.addVector(iIndex[iTerm], iValue[iTerm]*diffValue);

                for (int jTerm = iTerm; jTerm < 4; ++jTerm) {
                    double const value = iValue[iTerm]*iValue[jTerm]*invVariance;
                    equation.addOffDiagonal(iIndex[iTerm], iIndex[jTerm], 2*value);
                    if (notEnd) {
                        std::size_t const jIndexPlus = iIndex[jTerm] + _oversample;
                        equation.addOffDiagonal(iIndexPlus, iIndex[jTerm], -value);
                        equation.addOffDiagonal(iIndex[iTerm], jIndexPlus, -value);
                        equation.addOffDiagonal(iIndexPlus, jIndexPlus, 2*value);
                    }
                }
            }

            for (std::size_t jj = ii + 1; jj < right; ++jj) {
                double const jNorm = norm[jj];
                if (jNorm == 0) continue;

                auto const jInterp = getSwathProfileInterpolation(swathInterp, jj, xx, centers[jj], jNorm);
                std::array<std::size_t, 4> const jIndex = std::get<0>(jInterp);
                std::array<double, 4> const jValue = std::get<1>(jInterp);

                for (int iTerm = 0; iTerm < 4; ++iTerm) {
                    std::size_t const iIndexPlus = iIndex[iTerm] + _oversample;
                    for (int jTerm = 0; jTerm < 4; ++jTerm) {
                        double const value = iValue[iTerm]*jValue[jTerm]*invVariance;

                        // if (yy == 512 && ii == 0 && jj == 1) std::cerr << "x=" << xx << " bgOffset=" << bgOffset << " i=" << ii << " j=" << jj << " " << iIndex[iTerm] << " " << jIndex[jTerm] << " " << iValue[iTerm] << " " << jValue[jTerm] << " " << value << std::endl;

                        equation.addOffDiagonal(iIndex[iTerm], jIndex[jTerm], 2*value);
                        if (notEnd) {
                            std::size_t const jIndexPlus = jIndex[jTerm] + _oversample;
                            equation.addOffDiagonal(iIndexPlus, jIndex[jTerm], -value);
                            equation.addOffDiagonal(iIndex[iTerm], jIndexPlus, -value);
                            equation.addOffDiagonal(iIndexPlus, jIndexPlus, 2*value);
                        }
                    }
                }
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
    template<typename ImageT, typename Function>
    void iterateRow(
        ImageT const& image,
        int yy,
        ndarray::ArrayRef<double, 1, 0> const& centers,
        Function func
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
            upper[fiberIndex] = std::min(width - 1, int(std::floor(centers[fiberIndex] + _radius) + 1));
        }
//        std::cerr << "y=" << yy << " centers=" << centers << " lower=" << lower << " upper=" << upper << std::endl;
        // if (yy == 512) std::cerr << "Lower: " << lower << std::endl;
        // if (yy == 512) std::cerr << "Upper: " << upper << std::endl;
        // if (yy == 512) std::cerr << "Centers: " << centers << std::endl;

        // Iterate over pixels in the row
        std::size_t left = 0;  // fiber index of lower bound of consideration for this pixel (inclusive)
        std::size_t right = 0;  // fiber index of upper bound of consideration for this pixel (exclusive)
        int xx = 0;  // column for pixel of interest
        auto const stop = image.row_begin(yy) + (image.getWidth() - 1);  // exclude last pixel
        for (auto ptr = image.row_begin(yy); ptr != stop; ++ptr, ++xx) {
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

            func(xx, yy, ptr, centers, left, right);
        }
    }

    //@{
    // Solve the matrix equation
    void solve(ndarray::Array<double, 1, 1> & solution, ProfileEquation & equation, float matrixTol=1.0e-4) {
        // Detect and deal with singular values
        ndarray::Array<bool, 1, 1> isZero = ndarray::allocate(getNumParameters());
        isZero.deep() = true;
        for (std::size_t ii = 0; ii < getNumParameters(); ++ii) {
            for (std::size_t jj = ii; jj < getNumParameters(); ++jj) {
                if (equation.matrix.get(ii, jj) != 0.0) {
                    isZero[ii] = false;
                    break;
                }
            }
            if (isZero[ii]) {
                equation.addDiagonal(ii, 1.0);
                assert(equation.vector[ii] == 0.0);  // or we've done something wrong
            }
        }


#if 0
        {
            lsst::afw::image::Image<float> matrixImage(_numParameters, _numParameters);
            for (auto const& triplet : equation.matrix.getTriplets()) {
                matrixImage(triplet.col(), triplet.row()) = triplet.value();
            }
            matrixImage.writeFits("matrix.fits");
        }
#endif


        solution.deep() = std::numeric_limits<double>::quiet_NaN();

        // Solve the matrix equation
        // We use a preconditioned conjugate gradient solver, which is much less
        // picky about the matrix being close to singular.
        using Matrix = math::SymmetricSparseSquareMatrix::Matrix;
#if 0
        using Solver = Eigen::ConjugateGradient<
            Matrix, Eigen::Lower|Eigen::Upper, Eigen::DiagonalPreconditioner<double>
        >;
        solver.setMaxIterations(getNumParameters()*1000);
        solver.setTolerance(matrixTol);
#else
        using Solver = Eigen::SparseLU<Matrix>;
#endif
        Solver solver;
        equation.matrix.solve(solution, equation.vector, solver, true);

        for (std::size_t ii = 0; ii < getNumParameters(); ++ii) {
            if (isZero[ii]) {
                solution[ii] = 0.0; // std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    ndarray::Array<double, 1, 1> solve(ProfileEquation & equation, float matrixTol=1.0e-4) {
        ndarray::Array<double, 1, 1> solution = ndarray::allocate(_numParameters);
        solve(solution, equation, matrixTol);
        return solution;
    }
    //@}

    // Repackage the solution into a format that is useful for the user
    //
    // @param solution : Solution array
    // @return results struct
    auto repackageSolution(ndarray::Array<double, 1, 1> const& solution) {
        // Repackage the solution
        std::size_t const profileSize = _profileSize + _oversample;  // Add the extra pixel
        ndarray::Array<double, 3, 1> profiles = ndarray::allocate(_numFibers, _ySwaths.size(), profileSize);
        ndarray::Array<bool, 3, 1> masks = ndarray::allocate(_numFibers, _ySwaths.size(), profileSize);
        std::size_t index = 0;

        profiles.deep() = 0.0;
        masks.deep() = false;

        double const norm = solution[_iGaussian];

        for (std::size_t ss = 0; ss < _ySwaths.size(); ++ss) {
            for (std::size_t ff = 0; ff < _numFibers; ++ff) {
                double xx = -(_radius + 1)*_oversample;
                for (std::size_t pp = 0; pp < _profileSize; ++pp, ++index, xx += 1) {
                    double const value = solution[index];
                    profiles[ff][ss][pp] += norm*_gaussian(xx/_oversample) + value;
//                    std::cerr << "xx=" << xx << " " << norm*_gaussian(xx/_oversample) << " " << value << " " << profiles[ff][ss][pp] << std::endl;
                    assert(pp + _oversample < profileSize);
                    profiles[ff][ss][pp + _oversample] -= value;
                    bool const isMasked = std::isnan(solution[index]);
                    masks[ff][ss][pp] |= isMasked;
                    masks[ff][ss][pp + _oversample] |= isMasked;
                }
                // Add the extra pixel at the end
                for (std::size_t pp = _profileSize; pp < profileSize; ++pp, xx += 1) {
                    profiles[ff][ss][pp] += norm*_gaussian(xx/_oversample);
//                    std::cerr << "xx=" << xx << " " << norm*_gaussian(xx/_oversample) << " " << profiles[ff][ss][pp] << std::endl;
                }
            }
        }

        // std::cerr << "Solution:" << std::endl;
        // std::cerr << solution[ndarray::view(0, _profileSize)] << std::endl;
        // std::cerr << solution[ndarray::view(_profileSize, _numSwathsParameters)] << std::endl;

        std::vector<std::shared_ptr<lsst::afw::image::Image<float>>> backgrounds{_numImages};
        for (std::size_t ii = 0; ii < _numImages; ++ii) {
            std::size_t const offset = _numSwathsParameters + ii*_numBackgroundParameters;
            backgrounds[ii] = makeBackgroundImage(_dims, _bgSize, solution, offset);
        }

        return FitProfilesResults(profiles, masks, backgrounds);
    }

    // Reject pixels that are too far from the model
    //
    // @param solution : Solution array
    // @param imageIndex : Index of image to fit
    // @param image : Image to fit (mask is unused: we use the rejected array instead)
    // @param centers : Fiber centers for each row, fiber
    // @param spectra : Spectra for each row, fiber
    // @param rejected : Array of pixels to reject (true=rejected)
    // @param rejThreshold : Threshold for rejecting pixels (in sigma)
    ProfileEquation reject(
        ndarray::Array<double, 1, 1> const& solution,
        std::size_t imageIndex,
        lsst::afw::image::MaskedImage<float> const& image,
        ndarray::Array<double, 2, 1> const& centers,
        ndarray::Array<float, 2, 1> const& spectra,
        ndarray::Array<bool, 2, 1> rejected,
        float rejThreshold
    ) const {
        assert(centers.getShape() == spectra.getShape());
        assert(rejected.getShape()[0] == std::size_t(image.getHeight()));
        assert(rejected.getShape()[1] == std::size_t(image.getWidth()));
        std::size_t const bgOffset = imageIndex*_numBackgroundParameters;  // additional offset for background

        lsst::afw::image::Image<float> model{image.getBBox()};
        model = 0.0;

        ProfileEquation equation(_numParameters);

        #pragma omp parallel for reduction(+:equation) schedule(guided)
        for (int yy = 0; yy < image.getHeight(); ++yy) {
            // Protect ndarray reference counting
            std::unique_ptr<ndarray::ArrayRef<double, 1, 0> const> cen;
            std::unique_ptr<ndarray::ArrayRef<float, 1, 0> const> norm;
            std::unique_ptr<ndarray::ArrayRef<bool, 1, 0>> rej;
            #pragma omp critical
            {
                cen = std::make_unique<ndarray::ArrayRef<double, 1, 0>>(centers[ndarray::view()(yy)]);
                norm = std::make_unique<ndarray::ArrayRef<float, 1, 0>>(spectra[ndarray::view()(yy)]);
                rej = std::make_unique<ndarray::ArrayRef<bool, 1, 0>>(rejected[yy]);
            }

            ProfileEquation eqn(_numParameters);
            auto const swathInterp = getSwathInterpolation(yy);
            try {
                iterateRow(
                    image, yy, *cen,
                    [&](int xx, int yy, typename lsst::afw::image::MaskedImage<float>::x_iterator iter,
                        ndarray::ArrayRef<double, 1, 0> const& centers,
                        std::size_t left, std::size_t right
                    ) {
                        applyPixelRejection(
                            xx, yy, iter, *cen, left, right,
                            eqn, *rej, *norm, solution, rejThreshold, swathInterp, bgOffset, model
                        );
                    }
                );
            } catch (std::exception const& exc) {
                std::cerr << "Caught exception: " << exc.what() << std::endl;
                std::terminate();
            } catch (...) {
                std::cerr << "Caught unknown exception" << std::endl;
                std::terminate();
            }

            #pragma omp critical  // protect ndarray reference counting
            {
                cen.reset();
                norm.reset();
                rej.reset();
            }

            equation += eqn;
        }

        {
            std::ostringstream filename;
            filename << "model-" << imageIndex << ".fits";
            model.writeFits(filename.str());
        }
        for (int yy = 0; yy < image.getHeight(); ++yy) {
            for (int xx = 0; xx < image.getWidth(); ++xx) {
                model(xx, yy) = (*image.getImage())(xx, yy) - model(xx, yy);
            }
        }
        {
            std::ostringstream filename;
            filename << "resid-" << imageIndex << ".fits";
            model.writeFits(filename.str());
        }

        std::size_t numRejected = 0;
        for (int yy = 0; yy < image.getHeight(); ++yy) {
            for (int xx = 0; xx < image.getWidth(); ++xx) {
                if (rejected[yy][xx]) ++numRejected;
            }
        }
        std::cerr << "Image " << imageIndex << ": " << numRejected << " pixels rejected" << std::endl;

//        equation.matrix.symmetrize();
        return equation;
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
    // @param swathInterp : Swath interpolation for this row
    // @param bgOffset : Offset into the background parameters
    void applyPixelRejection(
        int xx,
        int yy,
        typename lsst::afw::image::MaskedImage<float>::x_iterator & iter,
        ndarray::ArrayRef<double, 1, 0> const& centers,
        std::size_t left,
        std::size_t right,
        ProfileEquation & equation,
        ndarray::ArrayRef<bool, 1, 0> & rejected,
        ndarray::ArrayRef<float, 1, 0> const& norm,
        ndarray::Array<double, 1, 1> const& solution,
        float const& rejThreshold,
        std::pair<std::pair<std::size_t, double>, std::pair<std::size_t, double>> const& swathInterp,
        std::size_t bgOffset,
        lsst::afw::image::Image<float> & modelImage
    ) const {
        assert(centers.size() == norm.size());
        assert(std::size_t(xx) < rejected.size());

        if (rejected[xx]) {
            // Already rejected; nothing to do
            return;
        }

        double model = 0.0;

        float const image = iter.image();
        float const variance = iter.variance();
        auto const bgIndex = bgOffset + _bgIndex[yy][xx];
        model += solution[bgIndex];

        for (std::size_t ii = left; ii < right; ++ii) {
            double const iNorm = norm[ii];
            if (iNorm == 0) continue;
            model = iNorm*_gaussian(xx - centers[ii])*solution[_iGaussian];

            auto const iInterp = getSwathProfileInterpolation(swathInterp, ii, xx, centers[ii], iNorm);
            std::array<std::size_t, 4> const iIndex = std::get<0>(iInterp);
            std::array<double, 4> const iValue = std::get<1>(iInterp);
            int const prevPosition = std::get<2>(iInterp);
            int const nextPosition = prevPosition + 1;

            model += solution[iIndex[0]]*iValue[0];
            model += solution[iIndex[1]]*iValue[1];
            model += solution[iIndex[2]]*iValue[2];
            model += solution[iIndex[3]]*iValue[3];

            if (prevPosition > _oversample) {
                model -= solution[iIndex[0] - _oversample]*iValue[0];
                model -= solution[iIndex[1] - _oversample]*iValue[1];
                model -= solution[iIndex[2] - _oversample]*iValue[2];
                model -= solution[iIndex[3] - _oversample]*iValue[3];
            }
        }

        modelImage(xx, yy) = model;

#if 0
        float const diff = (image - model)/std::sqrt(variance);
        if (std::abs(diff) > rejThreshold) {
            // Figure out this pixel's contribution, so we can subtract it
            accumulatePixel(
                xx, yy, iter, centers, left, right, equation, norm, rejected, swathInterp, bgOffset
            );
            rejected[xx] = true;
        }
#endif
    }

  private:
    // Indices:
    // profile[swath=0, fiberIndex=0, position=0]
    // ...
    // profile[swath=0, fiberIndex=0, position=profileSize-1]
    // profile[swath=0, fiberIndex=1, position=0]
    // ... ...
    // profile[swath=0, fiberIndex=numFibers-1, position=profileSize-1]
    // profile[swath=1, fiberIndex=0, position=0]
    // ... ...
    // profile[swath=numSwaths-1, fiberIndex=numFibers-1, position=profileSize-1]
    // background[image=0, superpixel=0]
    // ...
    // background[image=0, superpixel=numBackground]
    // background[image=1, superpixel=0]
    // ...
    // background[image=numImages-1, superpixel=numBackground]
    // norm

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

    // Linear interpolation between pixels
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

#if 0
        // Nearest-neighbour
        return std::make_pair(index, frac > 0.5 ? 1.0 : 0.0);
#endif

        assert(frac >= 0 && frac <= 1.0);
        return std::make_pair(index, frac);
    }

    // Get index for a profile within a swath
    std::size_t getSwathProfileIndex(std::size_t swathIndex, std::size_t profileIndex) const {
        assert(swathIndex < _ySwaths.size());
        assert(profileIndex < _numProfileParameters);
        return swathIndex*_numProfileParameters + profileIndex;
    }

    // Linear interpolation between swaths
    //
    // Given a row, return the swath indices and linear interpolation factors.
    std::pair<std::pair<std::size_t, double>, std::pair<std::size_t, double>> getSwathInterpolation(
        int yy
    ) const {

#if 0
        // Single swath, to simplify the matrix
        return std::make_pair(std::make_pair(0UL, 1.0), std::make_pair(0UL, 0.0));
#endif

        std::size_t const last = _ySwaths.size() - 1;
        if (yy <= _ySwaths[0]) {
            return std::make_pair(std::make_pair(0UL, 1.0), std::make_pair(0UL, 0.0));
        }
        if (yy >= _ySwaths[last]) {
            return std::make_pair(std::make_pair(last, 1.0), std::make_pair(last, 0.0));
        }
        std::size_t nextIndex = 0;
        while (_ySwaths[nextIndex] < yy && nextIndex < last) {
            ++nextIndex;
        }
        std::size_t const prevIndex = std::max(0UL, nextIndex - 1);

#if 0
        // Nearest neighbour
        return std::make_pair(
            std::make_pair(prevIndex, 1.0),
            std::make_pair(nextIndex, 0.0)
        );
#endif

        double const yPrev = _ySwaths[prevIndex];
        double const yNext = _ySwaths[nextIndex];
        double const nextWeight = (yy - yPrev)/(yNext - yPrev);
        double const prevWeight = 1.0 - nextWeight;

        return std::make_pair(
            std::make_pair(prevIndex, prevWeight),
            std::make_pair(nextIndex, nextWeight)
        );
    }

    // Linear interpolation between swaths and profiles
    //
    // Given a row, return the swath and profile indices and linear interpolation factors.
    // Since there are two linear interpolations (one in the spatial dimension across the
    // profile, and one in the spectral dimension across the swaths), we return four indices
    // and factors.
    std::tuple<
        std::array<std::size_t, 4>,
        std::array<double, 4>,
        int
    > getSwathProfileInterpolation(
        std::pair<std::pair<std::size_t, double>, std::pair<std::size_t, double>> const& swathInterp,
        std::size_t fiberIndex,
        int xx,
        double center,
        double model
    ) const {
        std::array<std::size_t, 4> indices;
        std::array<double, 4> weights;

        // Interpolation over swaths
        std::size_t const prevSwath = swathInterp.first.first;
        double const prevSwathFrac = swathInterp.first.second;
        std::size_t const nextSwath = swathInterp.second.first;
        double const nextSwathFrac = swathInterp.second.second;

        // Interpolation over profile
        std::size_t prevPosition;
        double prevPositionFrac;
        std::tie(prevPosition, prevPositionFrac) = getProfileInterpolation(fiberIndex, xx, center);
        std::size_t const nextPosition = prevPosition + 1;
        double const nextPositionFrac = 1.0 - prevPositionFrac;

        assert(prevPositionFrac >= 0);
        assert(nextPositionFrac >= 0);

        indices[0] = getSwathProfileIndex(prevSwath, prevPosition);
        indices[1] = getSwathProfileIndex(prevSwath, nextPosition);
        indices[2] = getSwathProfileIndex(nextSwath, prevPosition);
        indices[3] = getSwathProfileIndex(nextSwath, nextPosition);

        weights[0] = prevSwathFrac*prevPositionFrac*model;
        weights[1] = prevSwathFrac*nextPositionFrac*model;
        weights[2] = nextSwathFrac*prevPositionFrac*model;
        weights[3] = nextSwathFrac*nextPositionFrac*model;

        int const position = std::floor((xx - center)*_oversample) + _profileCenter;

        return std::make_tuple(indices, weights, position);
    }

#if 0
    std::pair<std::array<std::size_t, 4>, std::array<double, 4>> getBackgroundInterpolation(
        int xx,
        int yy
    ) {
        // XXX this can be optimised by remembering the y interpolation as we go along a row

    }
#endif

#if 1
    // Return the value of the Gaussian used for normalisation
    double _gaussian(double x) const {
        return std::exp(-0.5*std::pow(x*_invSigma, 2));
    }
#endif

    ndarray::Array<float, 1, 1> const& _ySwaths;  // y positions of swath centers
    int const _oversample;  // oversampling factor for profiles
    int const _radius;  // radius of profiles
    double const _invSigma;  // Gaussian sigma
    std::size_t const _profileSize;  // size of profiles = 2*radius*oversample + 1
    std::size_t const _profileCenter;  // index of center of profile = radius*oversample
    std::size_t const _numImages;  // number of images
    std::size_t const _numFibers;  // number of fibers
    lsst::geom::Extent2I const _dims;  // dimensions of image
    lsst::geom::Extent2I const _bgSize;  // size of background super-pixels (in regular pixels)
    std::size_t const _numProfileParameters;  // number of parameters for profiles per swath
    std::size_t const _numSwathsParameters;  // number of parameters for profiles for all swaths
    std::size_t const _numBackgroundParameters;  // number of parameters for background per image
    std::size_t const _numParameters;  // total number of parameters
    std::size_t const _iGaussian;  // index of Gaussian term
//    ndarray::Array<double, 1, 1> _gaussian;  // Gaussian profile
    ndarray::Array<int, 2, 1> const _bgIndex;  // index of background super-pixel for each pixel
};

}  // anonymous namespace


FitProfilesResults fitProfiles(
    std::vector<lsst::afw::image::MaskedImage<float>> const& images,
    std::vector<ndarray::Array<double, 2, 1>> const& centers,
    std::vector<ndarray::Array<float, 2, 1>> const& spectra,
    ndarray::Array<int, 1, 1> const& fiberIds,
    ndarray::Array<float, 1, 1> const& yKnots,
    lsst::afw::image::MaskPixel badBitMask,
    int oversample,
    int radius,
    float sigma,
    lsst::geom::Extent2I const& bgSize,
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
    int const height = dims.getY();
    for (std::size_t ii = 1; ii < num; ++ii) {
        if (images[ii].getDimensions() != dims) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "Image dimension mismatch");
        }
        if (centers[ii].getShape() != ndarray::makeVector(fiberIds.size(), std::size_t(dims.getY()))) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "Centers dimension mismatch");
        }
    }

    // Reject pixels based on mask
    ndarray::Array<bool, 3, 1> rejected = ndarray::allocate(num, height, width);
    for (std::size_t ii = 0; ii < num; ++ii) {
        std::size_t jj = 0;
        for (int yy = 0; yy < height; ++yy, ++jj) {
            auto mask = images[ii].getMask()->row_begin(yy);
            auto rej = rejected[ii][jj].begin();
            for (int xx = 0; xx < width; ++xx, ++mask, ++rej) {
                *rej = (*mask & badBitMask) != 0;
            }
        }
    }

    SwathProfileBuilder builder{
        images.size(), fiberIds.size(), yKnots, oversample, radius, sigma, dims, bgSize
    };
    LOGL_DEBUG(_log, "Building least-squares equation...");
    ProfileEquation equation{builder.getNumParameters()};
    for (std::size_t ii = 0; ii < images.size(); ++ii) {
        LOGL_DEBUG(_log, "    Accumulating image %d", ii);
        builder.accumulateImage(ii, images[ii], equation, centers[ii], spectra[ii], rejected[ii]);
    }
    // Add in the symmetric elements that we didn't bother to accumulate
//    equation.matrix.symmetrize();

    for (int iter = 0; iter < rejIter; ++iter) {
        // Solve for the profiles
        LOGL_DEBUG(_log, "Solving profiles: iteration %d", iter);
        auto const solution = builder.solve(equation, matrixTol);

        // Reject bad pixels
        LOGL_DEBUG(_log, "Rejecting pixels: iteration %d", iter);
        for (std::size_t ii = 0; ii < images.size(); ++ii) {
            equation -= builder.reject(
                solution, ii, images[ii], centers[ii], spectra[ii], rejected[ii].shallow(), rejThresh
            );
        }
    }

    // Final solution after iteration
    LOGL_DEBUG(_log, "Solving profiles: final");
    auto const solution = builder.solve(equation, matrixTol);

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
