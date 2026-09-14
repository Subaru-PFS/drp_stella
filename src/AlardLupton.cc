#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "ndarray.h"

#include "lsst/pex/exceptions.h"
#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"

#include "pfs/drp/stella/AlardLupton.h"
#include "pfs/drp/stella/math/NormalizedPolynomial.h"
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


/// Determine, for every pixel, whether both images have usable data there
ndarray::Array<bool, 2, 2> computeGood(
    lsst::afw::image::MaskedImage<float> const& source,
    lsst::afw::image::MaskedImage<float> const& target,
    lsst::afw::image::MaskPixel badBitMask
) {
    int const height = source.getHeight();
    int const width = source.getWidth();
    ndarray::Array<bool, 2, 2> good = ndarray::allocate(height, width);
    auto const sourceImage = source.getImage()->getArray();
    auto const sourceMask = source.getMask()->getArray();
    auto const sourceVariance = source.getVariance()->getArray();
    auto const targetImage = target.getImage()->getArray();
    auto const targetMask = target.getMask()->getArray();
    auto const targetVariance = target.getVariance()->getArray();
    for (int yy = 0; yy < height; ++yy) {
        for (int xx = 0; xx < width; ++xx) {
            good[yy][xx] = (
                isGoodPixel(sourceImage[yy][xx], sourceMask[yy][xx], sourceVariance[yy][xx], badBitMask) &&
                isGoodPixel(targetImage[yy][xx], targetMask[yy][xx], targetVariance[yy][xx], badBitMask)
            );
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

    ndarray::Array<double, 2, 2> kernel = ndarray::allocate(kernelSize, kernelSize);
    kernel.deep() = 0.0;
    ndarray::Array<double, 1, 1> background = ndarray::allocate(numBackgroundParams);
    background.deep() = 0.0;
    ndarray::Array<bool, 2, 2> rejected = ndarray::allocate(globalBox.getHeight(), globalBox.getWidth());
    rejected.deep() = false;

    return KernelSolution(
        globalBox, kernelHalfWidth, backgroundOrder, kernel, background, rejected,
        false, numPixels, 0, 0, 0,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    );
}


/// Fit a spatially-constant kernel and differential background within a single region
///
/// @param source, target : full input images (not restricted to the region)
/// @param computable : for every pixel in the full image, whether it (and its kernel
///     footprint) has usable data in both source and target
/// @param localBox : bounding box of the region, in local (0-indexed) array coordinates
/// @param globalBox : bounding box of the region, in the coordinate system of the input images
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
    double lsqThreshold
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

    // Candidate pixels within the region that have usable data
    std::vector<lsst::geom::Point2I> pixels;
    for (int yy = localBox.getMinY(); yy <= localBox.getMaxY(); ++yy) {
        for (int xx = localBox.getMinX(); xx <= localBox.getMaxX(); ++xx) {
            if (computable[yy][xx]) {
                pixels.emplace_back(xx, yy);
            }
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
        try {
            paramsSolution = math::solveLeastSquaresDesign(designScaled, meas, err, lsqThreshold);
        } catch (lsst::pex::exceptions::Exception const&) {
            return makeFailure(globalBox, kernelHalfWidth, backgroundOrder, numPixels);
        }
        for (std::size_t col = 0; col < numParams; ++col) {
            paramsSolution[col] /= colScale[col];
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
            ndarray::Array<double, 2, 2> kernelArray = ndarray::allocate(kernelSize, kernelSize);
            for (int dy = 0; dy < kernelSize; ++dy) {
                for (int dx = 0; dx < kernelSize; ++dx) {
                    kernelArray[dy][dx] = paramsSolution[std::size_t(dy)*kernelSize + std::size_t(dx)];
                }
            }
            ndarray::Array<double, 1, 1> backgroundArray = ndarray::allocate(numBackgroundParams);
            for (std::size_t jj = 0; jj < numBackgroundParams; ++jj) {
                backgroundArray[jj] = paramsSolution[numKernelParams + jj];
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

            return KernelSolution(
                globalBox, kernelHalfWidth, backgroundOrder,
                kernelArray, backgroundArray, rejectedArray,
                true, numPixels, numActive, numPixels - numActive, iter, chi2, rms
            );
        }

        for (std::size_t rr : toReject) {
            rejectedFlags[activeIndices[rr]] = true;
        }
        numActive -= toReject.size();
    }
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
    double rms_
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
    rms(rms_)
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


std::ostream& operator<<(std::ostream& os, KernelSolution const& solution) {
    os << "KernelSolution(bbox=" << solution.bbox << ", success=" << solution.success <<
        ", numPixels=" << solution.numPixels << ", numFit=" << solution.numFit <<
        ", numRejected=" << solution.numRejected << ", numIter=" << solution.numIter <<
        ", chi2=" << solution.chi2 << ", reducedChi2=" << solution.getReducedChi2() <<
        ", rms=" << solution.rms << ", kernelSum=" << solution.getKernelSum() << ")";
    return os;
}


AlardLuptonResult::AlardLuptonResult(
    lsst::afw::image::MaskedImage<float> const& difference_,
    std::vector<KernelSolution> const& solutions_,
    int numRegionsX_,
    int numRegionsY_,
    int kernelHalfWidth_
) : difference(difference_),
    solutions(solutions_),
    numRegionsX(numRegionsX_),
    numRegionsY(numRegionsY_),
    kernelHalfWidth(kernelHalfWidth_)
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
    double lsqThreshold
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

    ndarray::Array<bool, 2, 2> const good = computeGood(source, target, badBitMask);
    ndarray::Array<bool, 2, 2> const computable = erodeGood(good, kernelHalfWidth);

    lsst::afw::image::MaskedImage<float> diff{bbox};
    *diff.getImage() = 0.0;
    *diff.getMask() = 0;
    *diff.getVariance() = 0.0;

    lsst::afw::image::MaskPixel const noData = 1 << diff.getMask()->addMaskPlane("NO_DATA");
    lsst::afw::image::MaskPixel const diffimRejected = 1 << diff.getMask()->addMaskPlane("DIFFIM_REJECTED");

    auto const sourceImage = source.getImage()->getArray();
    auto const targetImage = target.getImage()->getArray();
    auto const sourceVariance = source.getVariance()->getArray();
    auto const targetVariance = target.getVariance()->getArray();
    auto const sourceMask = source.getMask()->getArray();
    auto const targetMask = target.getMask()->getArray();
    auto diffImage = diff.getImage()->getArray();
    auto diffVariance = diff.getVariance()->getArray();
    auto diffMask = diff.getMask()->getArray();

    for (int yy = 0; yy < height; ++yy) {
        for (int xx = 0; xx < width; ++xx) {
            diffMask[yy][xx] = sourceMask[yy][xx] | targetMask[yy][xx];
            if (!computable[yy][xx]) {
                diffMask[yy][xx] |= noData;
            }
        }
    }

    std::vector<std::pair<int, int>> const xBlocks = partitionRange(width, numRegionsX);
    std::vector<std::pair<int, int>> const yBlocks = partitionRange(height, numRegionsY);

    std::vector<KernelSolution> solutions;
    solutions.reserve(std::size_t(numRegionsX)*std::size_t(numRegionsY));

    for (auto const& yRange : yBlocks) {
        for (auto const& xRange : xBlocks) {
            lsst::geom::Point2I const localMin(xRange.first, yRange.first);
            lsst::geom::Point2I const localMax(xRange.second - 1, yRange.second - 1);
            lsst::geom::Box2I const localBox(localMin, localMax);
            lsst::geom::Box2I const globalBox(localMin + xy0, localMax + xy0);

            KernelSolution const solution = fitRegion(
                source, target, computable, localBox, globalBox,
                kernelHalfWidth, backgroundOrder, rejIter, rejThresh, lsqThreshold
            );

            if (solution.success) {
                math::NormalizedPolynomial2<double> const backgroundPoly(
                    solution.background, lsst::geom::Box2D(localBox)
                );
                for (int yy = localBox.getMinY(); yy <= localBox.getMaxY(); ++yy) {
                    for (int xx = localBox.getMinX(); xx <= localBox.getMaxX(); ++xx) {
                        if (!computable[yy][xx]) {
                            continue;
                        }
                        double model = backgroundPoly(double(xx), double(yy));
                        double modelVariance = 0.0;
                        for (int dy = -kernelHalfWidth; dy <= kernelHalfWidth; ++dy) {
                            for (int dx = -kernelHalfWidth; dx <= kernelHalfWidth; ++dx) {
                                double const kernelValue = (
                                    solution.kernel[dy + kernelHalfWidth][dx + kernelHalfWidth]
                                );
                                model += kernelValue*sourceImage[yy - dy][xx - dx];
                                modelVariance += kernelValue*kernelValue*sourceVariance[yy - dy][xx - dx];
                            }
                        }
                        diffImage[yy][xx] = targetImage[yy][xx] - model;
                        diffVariance[yy][xx] = targetVariance[yy][xx] + modelVariance;
                        if (solution.rejected[yy - localBox.getMinY()][xx - localBox.getMinX()]) {
                            diffMask[yy][xx] |= diffimRejected;
                        }
                    }
                }
            } else {
                for (int yy = localBox.getMinY(); yy <= localBox.getMaxY(); ++yy) {
                    for (int xx = localBox.getMinX(); xx <= localBox.getMaxX(); ++xx) {
                        diffMask[yy][xx] |= noData;
                    }
                }
            }

            solutions.push_back(solution);
        }
    }

    return AlardLuptonResult(diff, solutions, numRegionsX, numRegionsY, kernelHalfWidth);
}


}}}  // namespace pfs::drp::stella
