#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "ndarray.h"

#include "lsst/pex/exceptions.h"
#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"

#include "pfs/drp/stella/AlardLupton.h"
#include "pfs/drp/stella/math/NormalizedPolynomial.h"
#include "pfs/drp/stella/math/quartiles.h"
#include "pfs/drp/stella/math/solveLeastSquares.h"

namespace pfs {
namespace drp {
namespace stella {

namespace {


/// Divide [0, length) into numBlocks contiguous, nearly-equal blocks
///
/// @return (start, stop) pairs, with stop exclusive; the blocks tile
///     [0, length) exactly, with no gaps or overlaps.
std::vector<std::pair<int, int>> partitionRange(int length, int numBlocks) {
    std::vector<std::pair<int, int>> result;
    result.reserve(numBlocks);
    for (int ii = 0; ii < numBlocks; ++ii) {
        int const start = int(std::size_t(ii)*length/numBlocks);
        int const stop = int(std::size_t(ii + 1)*length/numBlocks);
        result.emplace_back(start, stop);
    }
    return result;
}


/// Is a single pixel usable for the fit?
template <typename ImageT, typename VarianceT>
bool isGoodPixel(ImageT value, lsst::afw::image::MaskPixel mask, VarianceT variance,
                  lsst::afw::image::MaskPixel badBitMask) {
    return (mask & badBitMask) == 0 && std::isfinite(value) && std::isfinite(variance) && variance > 0;
}


/// Determine, for every pixel, whether a single image has usable data there
ndarray::Array<bool, 2, 2> computeGoodImage(
    lsst::afw::image::MaskedImage<float> const& image,
    lsst::afw::image::MaskPixel badBitMask
) {
    int const height = image.getHeight();
    int const width = image.getWidth();
    ndarray::Array<bool, 2, 2> good = ndarray::allocate(height, width);
    auto const imageArray = image.getImage()->getArray();
    auto const maskArray = image.getMask()->getArray();
    auto const varianceArray = image.getVariance()->getArray();
    for (int yy = 0; yy < height; ++yy) {
        for (int xx = 0; xx < width; ++xx) {
            good[yy][xx] = isGoodPixel(
                imageArray[yy][xx], maskArray[yy][xx], varianceArray[yy][xx], badBitMask
            );
        }
    }
    return good;
}


/// Determine, for every pixel, whether both images have usable data there
ndarray::Array<bool, 2, 2> computeGood(
    ndarray::Array<bool, 2, 2> const& sourceGood,
    ndarray::Array<bool, 2, 2> const& targetGood
) {
    int const height = sourceGood.getShape()[0];
    int const width = sourceGood.getShape()[1];
    ndarray::Array<bool, 2, 2> good = ndarray::allocate(height, width);
    for (int yy = 0; yy < height; ++yy) {
        for (int xx = 0; xx < width; ++xx) {
            good[yy][xx] = sourceGood[yy][xx] && targetGood[yy][xx];
        }
    }
    return good;
}


/// Erode the "good" mask by the kernel footprint
///
/// A pixel is "computable" only if it, and every pixel within
/// kernelHalfWidth of it in x and y, is good; this requires the full
/// footprint to lie within the image, so a border of kernelHalfWidth
/// pixels around the edge of the image is never computable. We use a
/// summed-area table (integral image) so this is O(N) overall, rather
/// than O(N*kernelSize^2).
ndarray::Array<bool, 2, 2> erodeGood(ndarray::Array<bool, 2, 2> const& good, int kernelHalfWidth) {
    int const height = good.getShape()[0];
    int const width = good.getShape()[1];

    ndarray::Array<int, 2, 2> integral = ndarray::allocate(height + 1, width + 1);
    for (int xx = 0; xx <= width; ++xx) {
        integral[0][xx] = 0;
    }
    for (int yy = 0; yy < height; ++yy) {
        integral[yy + 1][0] = 0;
        for (int xx = 0; xx < width; ++xx) {
            integral[yy + 1][xx + 1] = (
                integral[yy][xx + 1] + integral[yy + 1][xx] - integral[yy][xx] + (good[yy][xx] ? 1 : 0)
            );
        }
    }

    int const footprintSize = (2*kernelHalfWidth + 1)*(2*kernelHalfWidth + 1);
    ndarray::Array<bool, 2, 2> result = ndarray::allocate(height, width);
    result.deep() = false;
    for (int yy = kernelHalfWidth; yy < height - kernelHalfWidth; ++yy) {
        int const yMin = yy - kernelHalfWidth;
        int const yMax = yy + kernelHalfWidth;
        for (int xx = kernelHalfWidth; xx < width - kernelHalfWidth; ++xx) {
            if (!good[yy][xx]) {
                continue;
            }
            int const xMin = xx - kernelHalfWidth;
            int const xMax = xx + kernelHalfWidth;
            int const numGood = (
                integral[yMax + 1][xMax + 1] - integral[yMin][xMax + 1] -
                integral[yMax + 1][xMin] + integral[yMin][xMin]
            );
            result[yy][xx] = (numGood == footprintSize);
        }
    }
    return result;
}


/// Construct a KernelSolution representing a failed (or untried) fit
KernelSolution makeFailure(
    lsst::geom::Box2I const& globalBox,
    int kernelHalfWidth,
    int backgroundOrder,
    std::size_t numPixels
) {
    int const kernelSize = 2*kernelHalfWidth + 1;
    std::size_t const numBackgroundParams = math::NormalizedPolynomial2<double>(backgroundOrder).getNParameters();

    double const nan = std::numeric_limits<double>::quiet_NaN();

    ndarray::Array<double, 2, 2> kernel = ndarray::allocate(kernelSize, kernelSize);
    kernel.deep() = 0.0;
    ndarray::Array<double, 1, 1> background = ndarray::allocate(numBackgroundParams);
    background.deep() = 0.0;
    ndarray::Array<bool, 2, 2> rejected = ndarray::allocate(globalBox.getHeight(), globalBox.getWidth());
    rejected.deep() = false;
    ndarray::Array<double, 2, 2> kernelError = ndarray::allocate(kernelSize, kernelSize);
    kernelError.deep() = nan;
    ndarray::Array<double, 1, 1> backgroundError = ndarray::allocate(numBackgroundParams);
    backgroundError.deep() = nan;

    return KernelSolution(
        globalBox, kernelHalfWidth, backgroundOrder, kernel, background, rejected,
        false, numPixels, 0, 0, 0, nan, nan, kernelError, backgroundError, nan
    );
}


/// Fit a spatially-constant kernel and differential background within a single region
///
/// @param source, target : full input images (not restricted to the region)
/// @param computable : for every pixel in the full image, whether it (and its kernel
///     footprint) has usable data in both source and target
/// @param localBox : bounding box of the region, in local (0-indexed) array coordinates
/// @param globalBox : bounding box of the region, in the coordinate system of the input images
/// @param commonKernelSumTarget : value to which the sum of the region's kernel taps is exactly
///     constrained (via variable elimination on the center kernel tap), or NaN to fit unconstrained
/// @param minSignalToNoise : minimum signal-to-noise ratio (source pixel value over its noise) required
///     for a pixel to be used in the fit; 0 (the default) applies no cut
KernelSolution fitRegion(
    lsst::afw::image::MaskedImage<float> const& source,
    lsst::afw::image::MaskedImage<float> const& target,
    ndarray::Array<bool, 2, 2> const& computable,
    lsst::geom::Box2I const& localBox,
    lsst::geom::Box2I const& globalBox,
    int kernelHalfWidth,
    int backgroundOrder,
    int rejIter,
    double rejThresh,
    double lsqThreshold,
    double commonKernelSumTarget = std::numeric_limits<double>::quiet_NaN(),
    double minSignalToNoise = 0.0
) {
    int const kernelSize = 2*kernelHalfWidth + 1;
    std::size_t const numKernelParams = std::size_t(kernelSize)*std::size_t(kernelSize);

    math::NormalizedPolynomial2<double> backgroundBasis(backgroundOrder, lsst::geom::Box2D(localBox));
    std::size_t const numBackgroundParams = backgroundBasis.getNParameters();
    std::size_t const numParams = numKernelParams + numBackgroundParams;

    auto const sourceImage = source.getImage()->getArray();
    auto const targetImage = target.getImage()->getArray();
    auto const sourceVariance = source.getVariance()->getArray();
    auto const targetVariance = target.getVariance()->getArray();

    // Candidate pixels within the region that have usable data and (if requested) adequate source
    // signal-to-noise: a pixel where the source is essentially flat/noise-only carries little or no
    // information about the kernel *shape* (every kernel tap sees nearly the same value there), so
    // including it can destabilize the shape fit even though it doesn't visibly hurt chi2.
    std::vector<lsst::geom::Point2I> pixels;
    for (int yy = localBox.getMinY(); yy <= localBox.getMaxY(); ++yy) {
        for (int xx = localBox.getMinX(); xx <= localBox.getMaxX(); ++xx) {
            if (!computable[yy][xx]) {
                continue;
            }
            if (minSignalToNoise > 0.0) {
                double const snr = double(sourceImage[yy][xx])/std::sqrt(double(sourceVariance[yy][xx]));
                if (!(snr >= minSignalToNoise)) {
                    continue;
                }
            }
            pixels.emplace_back(xx, yy);
        }
    }
    std::size_t const numPixels = pixels.size();
    if (numPixels <= numParams) {
        return makeFailure(globalBox, kernelHalfWidth, backgroundOrder, numPixels);
    }

    std::vector<bool> rejectedFlags(numPixels, false);
    std::size_t numActive = numPixels;

    for (int iter = 0; ; ++iter) {
        ndarray::Array<double, 2, 1> design = ndarray::allocate(numActive, numParams);
        ndarray::Array<double, 1, 1> meas = ndarray::allocate(numActive);
        ndarray::Array<double, 1, 1> err = ndarray::allocate(numActive);
        std::vector<std::size_t> activeIndices;
        activeIndices.reserve(numActive);

        std::size_t row = 0;
        for (std::size_t ii = 0; ii < numPixels; ++ii) {
            if (rejectedFlags[ii]) {
                continue;
            }
            int const xx = pixels[ii].getX();
            int const yy = pixels[ii].getY();
            for (int dy = -kernelHalfWidth; dy <= kernelHalfWidth; ++dy) {
                for (int dx = -kernelHalfWidth; dx <= kernelHalfWidth; ++dx) {
                    std::size_t const col = (
                        std::size_t(dy + kernelHalfWidth)*kernelSize + std::size_t(dx + kernelHalfWidth)
                    );
                    design[row][col] = sourceImage[yy - dy][xx - dx];
                }
            }
            std::vector<double> const basis = backgroundBasis.getDFuncDParameters(double(xx), double(yy));
            for (std::size_t jj = 0; jj < numBackgroundParams; ++jj) {
                design[row][numKernelParams + jj] = basis[jj];
            }
            meas[row] = targetImage[yy][xx];
            double const variance = double(sourceVariance[yy][xx]) + double(targetVariance[yy][xx]);
            err[row] = std::sqrt(variance);
            activeIndices.push_back(ii);
            ++row;
        }

        // Precondition the design matrix columns before solving: the kernel columns hold
        // raw pixel flux (which may be many orders of magnitude larger than 1), while the
        // background columns are order-unity polynomial terms. Left unscaled, this disparity
        // skews the eigenvalue spectrum of the normal equations enough that the relative SVD
        // threshold in solveLeastSquaresDesign can truncate the (perfectly well-determined)
        // background modes. Rescaling each column to unit RMS before solving, then undoing
        // the scaling on the solution, avoids that without changing the fit itself.
        ndarray::Array<double, 2, 2> designScaled = ndarray::allocate(numActive, numParams);
        ndarray::Array<double, 1, 1> colScale = ndarray::allocate(numParams);
        for (std::size_t col = 0; col < numParams; ++col) {
            double sumSq = 0.0;
            for (std::size_t row = 0; row < numActive; ++row) {
                double const value = design[row][col];
                sumSq += value*value;
            }
            double const rms = std::sqrt(sumSq/numActive);
            colScale[col] = (rms > 0.0) ? rms : 1.0;
            for (std::size_t row = 0; row < numActive; ++row) {
                designScaled[row][col] = design[row][col]/colScale[col];
            }
        }

        ndarray::Array<double, 1, 1> paramsSolution;
        ndarray::Array<double, 2, 2> covariance;
        try {
            auto const solved = math::solveLeastSquaresDesignCovariance(
                designScaled, meas, err, lsqThreshold
            );
            paramsSolution = solved.first;
            covariance = solved.second;
        } catch (lsst::pex::exceptions::Exception const&) {
            return makeFailure(globalBox, kernelHalfWidth, backgroundOrder, numPixels);
        }
        for (std::size_t col = 0; col < numParams; ++col) {
            paramsSolution[col] /= colScale[col];
        }
        // Un-scale the covariance to match: since actualParam = scaledParam/colScale,
        // Cov(actual_i, actual_j) = Cov(scaled_i, scaled_j)/(colScale[i]*colScale[j]).
        for (std::size_t row = 0; row < numParams; ++row) {
            for (std::size_t col = 0; col < numParams; ++col) {
                covariance[row][col] /= colScale[row]*colScale[col];
            }
        }

        double chi2 = 0.0;
        double sumSqResid = 0.0;
        std::vector<double> chiValues(numActive);
        for (std::size_t rr = 0; rr < numActive; ++rr) {
            double model = 0.0;
            for (std::size_t col = 0; col < numParams; ++col) {
                model += design[rr][col]*paramsSolution[col];
            }
            double const resid = meas[rr] - model;
            double const chi = resid/err[rr];
            chiValues[rr] = chi;
            chi2 += chi*chi;
            sumSqResid += resid*resid;
        }
        double const rms = std::sqrt(sumSqResid/numActive);

        std::vector<std::size_t> toReject;
        for (std::size_t rr = 0; rr < numActive; ++rr) {
            if (std::abs(chiValues[rr]) > rejThresh) {
                toReject.push_back(rr);
            }
        }

        bool const canIterate = (
            iter < rejIter && !toReject.empty() && (numActive - toReject.size() > numParams)
        );
        if (!canIterate) {
            ndarray::Array<double, 1, 1> finalParams = paramsSolution;
            ndarray::Array<double, 2, 2> finalCovariance = covariance;
            double finalChi2 = chi2;
            double finalRms = rms;
            bool sumIsFixed = false;

            if (std::isfinite(commonKernelSumTarget)) {
                // Exactly constrain the sum of the kernel taps to commonKernelSumTarget by variable
                // elimination: substitute k_pivot = commonKernelSumTarget - sum(other kernel taps) into
                // the model (pivoting on the center tap, which typically has the largest coefficient),
                // and solve the resulting numParams-1 unknown system. The active-pixel rejection above
                // is left completely unaffected by this (it has already converged), so the constraint
                // cannot bias which pixels are used; it only affects the final parameter values (and,
                // below, the reported chi2/rms).
                std::size_t const pivotCol = (
                    std::size_t(kernelHalfWidth)*kernelSize + std::size_t(kernelHalfWidth)
                );
                std::size_t const numReduced = numParams - 1;
                std::vector<std::size_t> colMap;
                colMap.reserve(numReduced);
                for (std::size_t col = 0; col < numParams; ++col) {
                    if (col != pivotCol) {
                        colMap.push_back(col);
                    }
                }

                ndarray::Array<double, 2, 2> reducedDesign = ndarray::allocate(numActive, numReduced);
                ndarray::Array<double, 1, 1> reducedMeas = ndarray::allocate(numActive);
                for (std::size_t rr = 0; rr < numActive; ++rr) {
                    double const pivotValue = design[rr][pivotCol];
                    for (std::size_t jj = 0; jj < numReduced; ++jj) {
                        std::size_t const col = colMap[jj];
                        double value = design[rr][col];
                        if (col < numKernelParams) {
                            value -= pivotValue;
                        }
                        reducedDesign[rr][jj] = value;
                    }
                    reducedMeas[rr] = meas[rr] - commonKernelSumTarget*pivotValue;
                }

                // Precondition the reduced design matrix columns, exactly as for the unconstrained
                // system above: its columns differ from the original (kernel columns hold the
                // difference from the pivot column), so the scaling must be recomputed.
                ndarray::Array<double, 2, 2> reducedDesignScaled = ndarray::allocate(numActive, numReduced);
                ndarray::Array<double, 1, 1> reducedColScale = ndarray::allocate(numReduced);
                for (std::size_t jj = 0; jj < numReduced; ++jj) {
                    double sumSq = 0.0;
                    for (std::size_t rr = 0; rr < numActive; ++rr) {
                        double const value = reducedDesign[rr][jj];
                        sumSq += value*value;
                    }
                    double const colRms = std::sqrt(sumSq/numActive);
                    reducedColScale[jj] = (colRms > 0.0) ? colRms : 1.0;
                    for (std::size_t rr = 0; rr < numActive; ++rr) {
                        reducedDesignScaled[rr][jj] = reducedDesign[rr][jj]/reducedColScale[jj];
                    }
                }

                bool constrainedOk = false;
                ndarray::Array<double, 1, 1> reducedSolution;
                ndarray::Array<double, 2, 2> reducedCovariance;
                try {
                    auto const reducedSolved = math::solveLeastSquaresDesignCovariance(
                        reducedDesignScaled, reducedMeas, err, lsqThreshold
                    );
                    reducedSolution = reducedSolved.first;
                    reducedCovariance = reducedSolved.second;
                    constrainedOk = true;
                } catch (lsst::pex::exceptions::Exception const&) {
                    constrainedOk = false;
                }

                if (constrainedOk) {
                    for (std::size_t jj = 0; jj < numReduced; ++jj) {
                        reducedSolution[jj] /= reducedColScale[jj];
                    }
                    for (std::size_t jj = 0; jj < numReduced; ++jj) {
                        for (std::size_t kk = 0; kk < numReduced; ++kk) {
                            reducedCovariance[jj][kk] /= reducedColScale[jj]*reducedColScale[kk];
                        }
                    }

                    ndarray::Array<double, 1, 1> constrainedSolution = ndarray::allocate(numParams);
                    for (std::size_t jj = 0; jj < numReduced; ++jj) {
                        constrainedSolution[colMap[jj]] = reducedSolution[jj];
                    }
                    double sumOtherKernelTaps = 0.0;
                    for (std::size_t col = 0; col < numKernelParams; ++col) {
                        if (col != pivotCol) {
                            sumOtherKernelTaps += constrainedSolution[col];
                        }
                    }
                    constrainedSolution[pivotCol] = commonKernelSumTarget - sumOtherKernelTaps;
                    finalParams = constrainedSolution;

                    // Reconstruct the full (numParams x numParams) covariance from the reduced solve's
                    // covariance: the eliminated pivot tap is a deterministic linear function of the
                    // other kernel taps (k_pivot = commonKernelSumTarget - sum(other kernel taps)), so
                    // its variance and its covariance with every other parameter follow by standard
                    // error propagation through that linear relation (Var(c - X) = Var(X), and
                    // Cov(c - X, Y) = -Cov(X, Y)).
                    std::vector<std::size_t> otherKernelIndices;
                    for (std::size_t jj = 0; jj < numReduced; ++jj) {
                        if (colMap[jj] < numKernelParams) {
                            otherKernelIndices.push_back(jj);
                        }
                    }
                    ndarray::Array<double, 2, 2> constrainedCovariance = ndarray::allocate(
                        numParams, numParams
                    );
                    constrainedCovariance.deep() = 0.0;
                    for (std::size_t jj = 0; jj < numReduced; ++jj) {
                        for (std::size_t kk = 0; kk < numReduced; ++kk) {
                            constrainedCovariance[colMap[jj]][colMap[kk]] = reducedCovariance[jj][kk];
                        }
                    }
                    double pivotVariance = 0.0;
                    for (std::size_t jj : otherKernelIndices) {
                        for (std::size_t kk : otherKernelIndices) {
                            pivotVariance += reducedCovariance[jj][kk];
                        }
                    }
                    constrainedCovariance[pivotCol][pivotCol] = pivotVariance;
                    for (std::size_t kk = 0; kk < numReduced; ++kk) {
                        double cov = 0.0;
                        for (std::size_t jj : otherKernelIndices) {
                            cov -= reducedCovariance[jj][kk];
                        }
                        constrainedCovariance[pivotCol][colMap[kk]] = cov;
                        constrainedCovariance[colMap[kk]][pivotCol] = cov;
                    }
                    finalCovariance = constrainedCovariance;
                    sumIsFixed = true;

                    // Recompute chi2/rms from the constrained parameters, but summed only over the
                    // real pixel rows against the original (unscaled) design: a region whose data
                    // genuinely disagrees with the shared-sum assumption will show it as elevated
                    // chi2/rms here, rather than the disagreement being invisible.
                    double chi2Constrained = 0.0;
                    double sumSqResidConstrained = 0.0;
                    for (std::size_t rr = 0; rr < numActive; ++rr) {
                        double model = 0.0;
                        for (std::size_t col = 0; col < numParams; ++col) {
                            model += design[rr][col]*finalParams[col];
                        }
                        double const resid = meas[rr] - model;
                        double const chi = resid/err[rr];
                        chi2Constrained += chi*chi;
                        sumSqResidConstrained += resid*resid;
                    }
                    finalChi2 = chi2Constrained;
                    finalRms = std::sqrt(sumSqResidConstrained/numActive);
                }
            }

            ndarray::Array<double, 2, 2> kernelArray = ndarray::allocate(kernelSize, kernelSize);
            for (int dy = 0; dy < kernelSize; ++dy) {
                for (int dx = 0; dx < kernelSize; ++dx) {
                    kernelArray[dy][dx] = finalParams[std::size_t(dy)*kernelSize + std::size_t(dx)];
                }
            }
            ndarray::Array<double, 1, 1> backgroundArray = ndarray::allocate(numBackgroundParams);
            for (std::size_t jj = 0; jj < numBackgroundParams; ++jj) {
                backgroundArray[jj] = finalParams[numKernelParams + jj];
            }
            ndarray::Array<bool, 2, 2> rejectedArray = ndarray::allocate(
                localBox.getHeight(), localBox.getWidth()
            );
            rejectedArray.deep() = false;
            for (std::size_t ii = 0; ii < numPixels; ++ii) {
                if (rejectedFlags[ii]) {
                    rejectedArray[pixels[ii].getY() - localBox.getMinY()]
                                 [pixels[ii].getX() - localBox.getMinX()] = true;
                }
            }

            // Raw formal (statistical) parameter errors, derived directly from the least-squares
            // covariance matrix computed from the pixel noise model (source + target variance): these
            // are NOT rescaled by the fit's reduced chi^2.
            ndarray::Array<double, 2, 2> kernelErrorArray = ndarray::allocate(kernelSize, kernelSize);
            for (int dy = 0; dy < kernelSize; ++dy) {
                for (int dx = 0; dx < kernelSize; ++dx) {
                    std::size_t const col = std::size_t(dy)*kernelSize + std::size_t(dx);
                    kernelErrorArray[dy][dx] = std::sqrt(std::max(0.0, finalCovariance[col][col]));
                }
            }
            ndarray::Array<double, 1, 1> backgroundErrorArray = ndarray::allocate(numBackgroundParams);
            for (std::size_t jj = 0; jj < numBackgroundParams; ++jj) {
                std::size_t const col = numKernelParams + jj;
                backgroundErrorArray[jj] = std::sqrt(std::max(0.0, finalCovariance[col][col]));
            }
            // The kernel sum is fixed by construction under the commonKernelSum constraint, so its
            // variance is exactly zero in that case; otherwise it's the full (correlated) sum over the
            // kernel-tap covariance submatrix.
            double kernelSumError;
            if (sumIsFixed) {
                kernelSumError = 0.0;
            } else {
                double sumVar = 0.0;
                for (std::size_t ii = 0; ii < numKernelParams; ++ii) {
                    for (std::size_t jj = 0; jj < numKernelParams; ++jj) {
                        sumVar += finalCovariance[ii][jj];
                    }
                }
                kernelSumError = std::sqrt(std::max(0.0, sumVar));
            }

            return KernelSolution(
                globalBox, kernelHalfWidth, backgroundOrder,
                kernelArray, backgroundArray, rejectedArray,
                true, numPixels, numActive, numPixels - numActive, iter, finalChi2, finalRms,
                kernelErrorArray, backgroundErrorArray, kernelSumError
            );
        }

        for (std::size_t rr : toReject) {
            rejectedFlags[activeIndices[rr]] = true;
        }
        numActive -= toReject.size();
    }
}


/// Compute a robust common kernel-sum estimate (and its scatter) from a set of region fits
///
/// @param solutions : kernel solutions for all regions (failed fits are ignored)
/// @return (commonSum, scatter), both NaN if fewer than 2 regions fit successfully
std::pair<double, double> computeCommonKernelSum(std::vector<KernelSolution> const& solutions) {
    double const nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> sums;
    sums.reserve(solutions.size());
    for (auto const& solution : solutions) {
        if (solution.success) {
            sums.push_back(solution.getKernelSum());
        }
    }
    if (sums.size() < 2) {
        return std::make_pair(nan, nan);
    }

    ndarray::Array<double, 1, 1> sumsArray = ndarray::allocate(sums.size());
    ndarray::Array<bool, 1, 1> mask = ndarray::allocate(sums.size());
    for (std::size_t ii = 0; ii < sums.size(); ++ii) {
        sumsArray[ii] = sums[ii];
        mask[ii] = false;
    }

    double const commonSum = math::calculateMedian(sumsArray, mask);
    double scatter = math::robustRms<double, 1>(sumsArray, mask);
    // Floor the scatter to avoid a degenerate, near-zero prior error if the regions happen to
    // already agree almost exactly.
    double const minScatter = 1.0e-3*std::abs(commonSum);
    if (!(scatter > minScatter)) {
        scatter = minScatter;
    }
    return std::make_pair(commonSum, scatter);
}


/// Fit a KernelSolution independently in each region of a grid
///
/// @param source, target : full input images (not restricted to any region)
/// @param computable : for every pixel in the full image, whether it (and its kernel footprint) has
///     usable data in both source and target
/// @param xBlocks, yBlocks : (start, stop) pairs, in local (0-indexed) array coordinates, dividing the
///     images into regions in x and y respectively
/// @param xy0 : offset from local (0-indexed) array coordinates to the coordinate system of the input
///     images
/// @param commonKernelSumTarget : passed through to fitRegion (see there); NaN (the default) fits
///     every region unconstrained
/// @param minSignalToNoise : passed through to fitRegion (see there); 0 (the default) applies no cut
/// @return kernel solution for each region, in row-major (y, x) order
std::vector<KernelSolution> fitAllRegions(
    lsst::afw::image::MaskedImage<float> const& source,
    lsst::afw::image::MaskedImage<float> const& target,
    ndarray::Array<bool, 2, 2> const& computable,
    std::vector<std::pair<int, int>> const& xBlocks,
    std::vector<std::pair<int, int>> const& yBlocks,
    lsst::geom::Extent2I const& xy0,
    int kernelHalfWidth,
    int backgroundOrder,
    int rejIter,
    double rejThresh,
    double lsqThreshold,
    double commonKernelSumTarget = std::numeric_limits<double>::quiet_NaN(),
    double minSignalToNoise = 0.0
) {
    std::vector<KernelSolution> solutions;
    solutions.reserve(xBlocks.size()*yBlocks.size());
    for (auto const& yRange : yBlocks) {
        for (auto const& xRange : xBlocks) {
            lsst::geom::Point2I const localMin(xRange.first, yRange.first);
            lsst::geom::Point2I const localMax(xRange.second - 1, yRange.second - 1);
            lsst::geom::Box2I const localBox(localMin, localMax);
            lsst::geom::Box2I const globalBox(localMin + xy0, localMax + xy0);

            solutions.push_back(fitRegion(
                source, target, computable, localBox, globalBox,
                kernelHalfWidth, backgroundOrder, rejIter, rejThresh, lsqThreshold,
                commonKernelSumTarget, minSignalToNoise
            ));
        }
    }
    return solutions;
}


}  // anonymous namespace


KernelSolution::KernelSolution(
    lsst::geom::Box2I const& bbox_,
    int kernelHalfWidth_,
    int backgroundOrder_,
    ndarray::Array<double, 2, 2> const& kernel_,
    ndarray::Array<double, 1, 1> const& background_,
    ndarray::Array<bool, 2, 2> const& rejected_,
    bool success_,
    std::size_t numPixels_,
    std::size_t numFit_,
    std::size_t numRejected_,
    int numIter_,
    double chi2_,
    double rms_,
    ndarray::Array<double, 2, 2> const& kernelError_,
    ndarray::Array<double, 1, 1> const& backgroundError_,
    double kernelSumError_
) : bbox(bbox_),
    kernelHalfWidth(kernelHalfWidth_),
    backgroundOrder(backgroundOrder_),
    kernel(kernel_),
    background(background_),
    rejected(rejected_),
    success(success_),
    numPixels(numPixels_),
    numFit(numFit_),
    numRejected(numRejected_),
    numIter(numIter_),
    chi2(chi2_),
    rms(rms_),
    kernelError(kernelError_),
    backgroundError(backgroundError_),
    kernelSumError(kernelSumError_)
{}


double KernelSolution::getKernelSum() const {
    double sum = 0.0;
    auto const shape = kernel.getShape();
    for (std::size_t yy = 0; yy < shape[0]; ++yy) {
        for (std::size_t xx = 0; xx < shape[1]; ++xx) {
            sum += kernel[yy][xx];
        }
    }
    return sum;
}


double KernelSolution::getReducedChi2() const {
    std::size_t const dof = getDegreesOfFreedom();
    if (dof == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return chi2/dof;
}


lsst::geom::Point2D KernelSolution::getFirstMoment() const {
    double const nan = std::numeric_limits<double>::quiet_NaN();
    double const sum0 = getKernelSum();
    if (!success || sum0 == 0.0) {
        return lsst::geom::Point2D(nan, nan);
    }
    double xSum = 0.0;
    double ySum = 0.0;
    auto const shape = kernel.getShape();
    for (std::size_t yy = 0; yy < shape[0]; ++yy) {
        double const dy = double(yy) - kernelHalfWidth;
        for (std::size_t xx = 0; xx < shape[1]; ++xx) {
            double const dx = double(xx) - kernelHalfWidth;
            double const value = kernel[yy][xx];
            xSum += value*dx;
            ySum += value*dy;
        }
    }
    return lsst::geom::Point2D(xSum/sum0, ySum/sum0);
}


lsst::afw::geom::ellipses::Quadrupole KernelSolution::getSecondMoment() const {
    double const nan = std::numeric_limits<double>::quiet_NaN();
    lsst::geom::Point2D const first = getFirstMoment();
    if (!std::isfinite(first.getX()) || !std::isfinite(first.getY())) {
        return lsst::afw::geom::ellipses::Quadrupole(nan, nan, nan);
    }
    double const sum0 = getKernelSum();
    double xxSum = 0.0;
    double yySum = 0.0;
    double xySum = 0.0;
    auto const shape = kernel.getShape();
    for (std::size_t yy = 0; yy < shape[0]; ++yy) {
        double const dy = double(yy) - kernelHalfWidth - first.getY();
        for (std::size_t xx = 0; xx < shape[1]; ++xx) {
            double const dx = double(xx) - kernelHalfWidth - first.getX();
            double const value = kernel[yy][xx];
            xxSum += value*dx*dx;
            yySum += value*dy*dy;
            xySum += value*dx*dy;
        }
    }
    return lsst::afw::geom::ellipses::Quadrupole(xxSum/sum0, yySum/sum0, xySum/sum0);
}


std::ostream& operator<<(std::ostream& os, KernelSolution const& solution) {
    os << "KernelSolution(bbox=" << solution.bbox << ", success=" << solution.success <<
        ", numPixels=" << solution.numPixels << ", numFit=" << solution.numFit <<
        ", numRejected=" << solution.numRejected << ", numIter=" << solution.numIter <<
        ", chi2=" << solution.chi2 << ", reducedChi2=" << solution.getReducedChi2() <<
        ", rms=" << solution.rms << ", kernelSum=" << solution.getKernelSum() <<
        ", kernelSumError=" << solution.kernelSumError << ")";
    return os;
}


AlardLuptonResult::AlardLuptonResult(
    lsst::afw::image::MaskedImage<float> const& convolved_,
    lsst::afw::image::MaskedImage<float> const& difference_,
    std::vector<KernelSolution> const& solutions_,
    int numRegionsX_,
    int numRegionsY_,
    int kernelHalfWidth_,
    double commonKernelSum_,
    double commonKernelSumScatter_
) : convolved(convolved_),
    difference(difference_),
    solutions(solutions_),
    numRegionsX(numRegionsX_),
    numRegionsY(numRegionsY_),
    kernelHalfWidth(kernelHalfWidth_),
    commonKernelSum(commonKernelSum_),
    commonKernelSumScatter(commonKernelSumScatter_)
{}


KernelSolution const* AlardLuptonResult::getSolutionAt(int xx, int yy) const {
    lsst::geom::Box2I const bbox = difference.getBBox();
    if (!bbox.contains(lsst::geom::Point2I(xx, yy))) {
        return nullptr;
    }
    int const localX = xx - bbox.getMinX();
    int const localY = yy - bbox.getMinY();
    std::vector<std::pair<int, int>> const xBlocks = partitionRange(bbox.getWidth(), numRegionsX);
    std::vector<std::pair<int, int>> const yBlocks = partitionRange(bbox.getHeight(), numRegionsY);

    int regionX = 0;
    while (regionX < numRegionsX - 1 && localX >= xBlocks[regionX].second) {
        ++regionX;
    }
    int regionY = 0;
    while (regionY < numRegionsY - 1 && localY >= yBlocks[regionY].second) {
        ++regionY;
    }

    std::size_t const index = std::size_t(regionY)*std::size_t(numRegionsX) + std::size_t(regionX);
    return &solutions[index];
}


double AlardLuptonResult::getChi2() const {
    double total = 0.0;
    for (auto const& solution : solutions) {
        total += solution.chi2;
    }
    return total;
}


std::size_t AlardLuptonResult::getNumFit() const {
    std::size_t total = 0;
    for (auto const& solution : solutions) {
        total += solution.numFit;
    }
    return total;
}


std::size_t AlardLuptonResult::getNumRejected() const {
    std::size_t total = 0;
    for (auto const& solution : solutions) {
        total += solution.numRejected;
    }
    return total;
}


std::pair<ndarray::Array<double, 2, 2>, ndarray::Array<double, 2, 2>>
AlardLuptonResult::getKernelFirstMoments() const {
    ndarray::Array<double, 2, 2> momentX = ndarray::allocate(numRegionsY, numRegionsX);
    ndarray::Array<double, 2, 2> momentY = ndarray::allocate(numRegionsY, numRegionsX);
    for (int regionY = 0; regionY < numRegionsY; ++regionY) {
        for (int regionX = 0; regionX < numRegionsX; ++regionX) {
            std::size_t const index = std::size_t(regionY)*std::size_t(numRegionsX) + std::size_t(regionX);
            lsst::geom::Point2D const moment = solutions[index].getFirstMoment();
            momentX[regionY][regionX] = moment.getX();
            momentY[regionY][regionX] = moment.getY();
        }
    }
    return std::make_pair(momentX, momentY);
}


std::tuple<ndarray::Array<double, 2, 2>, ndarray::Array<double, 2, 2>, ndarray::Array<double, 2, 2>>
AlardLuptonResult::getKernelSecondMoments() const {
    ndarray::Array<double, 2, 2> momentXX = ndarray::allocate(numRegionsY, numRegionsX);
    ndarray::Array<double, 2, 2> momentYY = ndarray::allocate(numRegionsY, numRegionsX);
    ndarray::Array<double, 2, 2> momentXY = ndarray::allocate(numRegionsY, numRegionsX);
    for (int regionY = 0; regionY < numRegionsY; ++regionY) {
        for (int regionX = 0; regionX < numRegionsX; ++regionX) {
            std::size_t const index = std::size_t(regionY)*std::size_t(numRegionsX) + std::size_t(regionX);
            lsst::afw::geom::ellipses::Quadrupole const moment = solutions[index].getSecondMoment();
            momentXX[regionY][regionX] = moment.getIxx();
            momentYY[regionY][regionX] = moment.getIyy();
            momentXY[regionY][regionX] = moment.getIxy();
        }
    }
    return std::make_tuple(momentXX, momentYY, momentXY);
}


AlardLuptonResult fitAlardLuptonKernel(
    lsst::afw::image::MaskedImage<float> const& source,
    lsst::afw::image::MaskedImage<float> const& target,
    int kernelHalfWidth,
    int numRegionsX,
    int numRegionsY,
    int backgroundOrder,
    lsst::afw::image::MaskPixel badBitMask,
    int rejIter,
    double rejThresh,
    double lsqThreshold,
    bool commonKernelSum,
    double minSignalToNoise
) {
    if (source.getBBox() != target.getBBox()) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthError, "source and target have different bounding boxes"
        );
    }
    if (kernelHalfWidth < 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "kernelHalfWidth must be >= 0");
    }
    if (numRegionsX < 1 || numRegionsY < 1) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError, "numRegionsX and numRegionsY must be >= 1"
        );
    }
    if (backgroundOrder < 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "backgroundOrder must be >= 0");
    }
    if (rejIter < 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "rejIter must be >= 0");
    }
    if (!(rejThresh > 0)) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "rejThresh must be > 0");
    }
    if (!(lsqThreshold > 0)) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "lsqThreshold must be > 0");
    }
    if (minSignalToNoise < 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "minSignalToNoise must be >= 0");
    }

    lsst::geom::Box2I const bbox = source.getBBox();
    int const width = bbox.getWidth();
    int const height = bbox.getHeight();
    int const kernelSize = 2*kernelHalfWidth + 1;
    if (width < kernelSize || height < kernelSize) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::LengthError, "images are too small for the requested kernelHalfWidth"
        );
    }

    int const x0 = bbox.getMinX();
    int const y0 = bbox.getMinY();
    lsst::geom::Extent2I const xy0{x0, y0};

    ndarray::Array<bool, 2, 2> const sourceGood = computeGoodImage(source, badBitMask);
    ndarray::Array<bool, 2, 2> const targetGood = computeGoodImage(target, badBitMask);
    ndarray::Array<bool, 2, 2> const good = computeGood(sourceGood, targetGood);
    ndarray::Array<bool, 2, 2> const computable = erodeGood(good, kernelHalfWidth);

    // A pixel is within the footprint if the full kernel footprint centred on it lies within the image;
    // outside that border we can never evaluate a model, regardless of how much data is good.
    auto const withinFootprint = [height, width, kernelHalfWidth](int yy, int xx) {
        return (
            yy >= kernelHalfWidth && yy < height - kernelHalfWidth &&
            xx >= kernelHalfWidth && xx < width - kernelHalfWidth
        );
    };

    lsst::afw::image::MaskedImage<float> diff{bbox};
    *diff.getImage() = 0.0;
    *diff.getMask() = 0;
    *diff.getVariance() = 0.0;

    lsst::afw::image::MaskedImage<float> convolved{bbox};
    *convolved.getImage() = 0.0;
    *convolved.getMask() = 0;
    *convolved.getVariance() = 0.0;

    lsst::afw::image::MaskPixel const noData = 1 << diff.getMask()->addMaskPlane("NO_DATA");
    lsst::afw::image::MaskPixel const diffimRejected = 1 << diff.getMask()->addMaskPlane("DIFFIM_REJECTED");
    lsst::afw::image::MaskPixel const diffimPartial = 1 << diff.getMask()->addMaskPlane("DIFFIM_PARTIAL");

    auto const sourceImage = source.getImage()->getArray();
    auto const targetImage = target.getImage()->getArray();
    auto const sourceVariance = source.getVariance()->getArray();
    auto const targetVariance = target.getVariance()->getArray();
    auto const sourceMask = source.getMask()->getArray();
    auto const targetMask = target.getMask()->getArray();
    auto diffImage = diff.getImage()->getArray();
    auto diffVariance = diff.getVariance()->getArray();
    auto diffMask = diff.getMask()->getArray();
    auto convolvedImage = convolved.getImage()->getArray();
    auto convolvedVariance = convolved.getVariance()->getArray();
    auto convolvedMask = convolved.getMask()->getArray();

    for (int yy = 0; yy < height; ++yy) {
        for (int xx = 0; xx < width; ++xx) {
            diffMask[yy][xx] = sourceMask[yy][xx] | targetMask[yy][xx];
            if (!withinFootprint(yy, xx) || !targetGood[yy][xx]) {
                diffMask[yy][xx] |= noData;
            }
            convolvedMask[yy][xx] = diffMask[yy][xx];
        }
    }

    std::vector<std::pair<int, int>> const xBlocks = partitionRange(width, numRegionsX);
    std::vector<std::pair<int, int>> const yBlocks = partitionRange(height, numRegionsY);

    std::vector<KernelSolution> solutions = fitAllRegions(
        source, target, computable, xBlocks, yBlocks, xy0,
        kernelHalfWidth, backgroundOrder, rejIter, rejThresh, lsqThreshold,
        std::numeric_limits<double>::quiet_NaN(), minSignalToNoise
    );

    double commonKernelSumValue = std::numeric_limits<double>::quiet_NaN();
    double commonKernelSumScatter = std::numeric_limits<double>::quiet_NaN();
    if (commonKernelSum) {
        std::tie(commonKernelSumValue, commonKernelSumScatter) = computeCommonKernelSum(solutions);
        if (std::isfinite(commonKernelSumValue)) {
            solutions = fitAllRegions(
                source, target, computable, xBlocks, yBlocks, xy0,
                kernelHalfWidth, backgroundOrder, rejIter, rejThresh, lsqThreshold,
                commonKernelSumValue, minSignalToNoise
            );
        }
    }

    std::size_t regionIndex = 0;
    for (auto const& yRange : yBlocks) {
        for (auto const& xRange : xBlocks) {
            lsst::geom::Point2I const localMin(xRange.first, yRange.first);
            lsst::geom::Point2I const localMax(xRange.second - 1, yRange.second - 1);
            lsst::geom::Box2I const localBox(localMin, localMax);

            KernelSolution const& solution = solutions[regionIndex];
            ++regionIndex;

            if (solution.success) {
                math::NormalizedPolynomial2<double> const backgroundPoly(
                    solution.background, lsst::geom::Box2D(localBox)
                );
                double const kernelSum = solution.getKernelSum();
                double const minValidWeight = std::max(1.0e-3*std::abs(kernelSum), 1.0e-6);
                for (int yy = localBox.getMinY(); yy <= localBox.getMaxY(); ++yy) {
                    for (int xx = localBox.getMinX(); xx <= localBox.getMaxX(); ++xx) {
                        if (!withinFootprint(yy, xx) || !targetGood[yy][xx]) {
                            continue;
                        }
                        double model = backgroundPoly(double(xx), double(yy));
                        double modelVariance = 0.0;
                        if (computable[yy][xx]) {
                            // Every source pixel in the footprint is good: sum over the full footprint.
                            for (int dy = -kernelHalfWidth; dy <= kernelHalfWidth; ++dy) {
                                for (int dx = -kernelHalfWidth; dx <= kernelHalfWidth; ++dx) {
                                    double const kernelValue = (
                                        solution.kernel[dy + kernelHalfWidth][dx + kernelHalfWidth]
                                    );
                                    model += kernelValue*sourceImage[yy - dy][xx - dx];
                                    modelVariance += (
                                        kernelValue*kernelValue*sourceVariance[yy - dy][xx - dx]
                                    );
                                }
                            }
                        } else {
                            // Some source pixels in the footprint are bad: sum over only the good
                            // ones, and rescale by the ratio of the full kernel sum to the sum of the
                            // kernel weights actually used, to approximately compensate for the missing
                            // flux (assuming the source is locally flat over the footprint).
                            double validWeight = 0.0;
                            double rawModel = 0.0;
                            double rawModelVariance = 0.0;
                            for (int dy = -kernelHalfWidth; dy <= kernelHalfWidth; ++dy) {
                                for (int dx = -kernelHalfWidth; dx <= kernelHalfWidth; ++dx) {
                                    if (!sourceGood[yy - dy][xx - dx]) {
                                        continue;
                                    }
                                    double const kernelValue = (
                                        solution.kernel[dy + kernelHalfWidth][dx + kernelHalfWidth]
                                    );
                                    validWeight += kernelValue;
                                    rawModel += kernelValue*sourceImage[yy - dy][xx - dx];
                                    rawModelVariance += (
                                        kernelValue*kernelValue*sourceVariance[yy - dy][xx - dx]
                                    );
                                }
                            }
                            if (std::abs(validWeight) < minValidWeight) {
                                diffMask[yy][xx] |= noData;
                                convolvedMask[yy][xx] |= noData;
                                continue;
                            }
                            double const scale = kernelSum/validWeight;
                            model += scale*rawModel;
                            modelVariance += scale*scale*rawModelVariance;
                            diffMask[yy][xx] |= diffimPartial;
                            convolvedMask[yy][xx] |= diffimPartial;
                        }
                        diffImage[yy][xx] = targetImage[yy][xx] - model;
                        diffVariance[yy][xx] = targetVariance[yy][xx] + modelVariance;
                        convolvedImage[yy][xx] = model;
                        convolvedVariance[yy][xx] = modelVariance;
                        if (solution.rejected[yy - localBox.getMinY()][xx - localBox.getMinX()]) {
                            diffMask[yy][xx] |= diffimRejected;
                            convolvedMask[yy][xx] |= diffimRejected;
                        }
                    }
                }
            } else {
                for (int yy = localBox.getMinY(); yy <= localBox.getMaxY(); ++yy) {
                    for (int xx = localBox.getMinX(); xx <= localBox.getMaxX(); ++xx) {
                        diffMask[yy][xx] |= noData;
                        convolvedMask[yy][xx] |= noData;
                    }
                }
            }
        }
    }

    return AlardLuptonResult(
        convolved, diff, solutions, numRegionsX, numRegionsY, kernelHalfWidth,
        commonKernelSumValue, commonKernelSumScatter
    );
}


}}}  // namespace pfs::drp::stella
