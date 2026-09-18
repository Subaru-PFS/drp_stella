#ifndef PFS_DRP_STELLA_ALARDLUPTON_H
#define PFS_DRP_STELLA_ALARDLUPTON_H

#include <cstddef>
#include <limits>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

#include "ndarray_fwd.h"
#include "lsst/geom/Box.h"
#include "lsst/geom/Point.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/image/MaskedImage.h"

namespace pfs {
namespace drp {
namespace stella {


/// Result of fitting a spatially-constant kernel within a single region of an image
///
/// The kernel uses a delta-function basis (one free parameter per pixel
/// offset within the kernel footprint), following Alard & Lupton (1998), but
/// without their Gauss-Hermite basis. The model for a pixel (x, y) within the
/// region is
///
///     model(x, y) = sum_{dy=-kernelHalfWidth}^{kernelHalfWidth}
///                   sum_{dx=-kernelHalfWidth}^{kernelHalfWidth}
///                       kernel[dy][dx]*source(x - dx, y - dy)
///                   + background(x, y)
///
/// where "background" is a low-order 2D polynomial (see
/// pfs::drp::stella::math::NormalizedPolynomial2) fit simultaneously with the
/// kernel, to account for differential sky/bias offsets between the two
/// images. Spatial variation of the kernel across the full image is achieved
/// by fitting independently in each of a grid of regions (see
/// fitAlardLuptonKernel and AlardLuptonResult); this struct holds the result
/// for a single such region.
struct KernelSolution {
    lsst::geom::Box2I bbox;  ///< Bounding box of the region, in the coordinate system of the input images
    int kernelHalfWidth;  ///< Half-width of the kernel in x and y
    int backgroundOrder;  ///< Order of the differential background polynomial
    ndarray::Array<double, 2, 2> kernel;  ///< Convolution kernel, shape (2*kernelHalfWidth + 1) square
    ndarray::Array<double, 1, 1> background;  ///< Differential background polynomial coefficients
    ndarray::Array<bool, 2, 2> rejected;  ///< Whether each pixel in the region was rejected from the fit
    bool success;  ///< Did the fit succeed?
    std::size_t numPixels;  ///< Number of pixels in the region with valid data in both images
    std::size_t numFit;  ///< Number of pixels used in the final iteration of the fit
    std::size_t numRejected;  ///< Number of pixels rejected during the fit (subset of numPixels)
    int numIter;  ///< Number of rejection iterations actually performed
    double chi2;  ///< chi^2 of the final fit, summed over the pixels used in the fit
    double rms;  ///< RMS of the difference image, over the pixels used in the fit

    /// Ctor
    KernelSolution(
        lsst::geom::Box2I const& bbox,
        int kernelHalfWidth,
        int backgroundOrder,
        ndarray::Array<double, 2, 2> const& kernel,
        ndarray::Array<double, 1, 1> const& background,
        ndarray::Array<bool, 2, 2> const& rejected,
        bool success,
        std::size_t numPixels,
        std::size_t numFit,
        std::size_t numRejected,
        int numIter,
        double chi2,
        double rms
    );

    /// Number of parameters in the fit (kernel pixels plus background terms)
    std::size_t getNumParams() const { return kernel.getNumElements() + background.getNumElements(); }

    /// Sum of the kernel values (i.e., the flux scaling applied by the kernel)
    double getKernelSum() const;

    /// Degrees of freedom of the fit (numFit - getNumParams(), or 0 if negative)
    std::size_t getDegreesOfFreedom() const {
        std::size_t const numParams = getNumParams();
        return numFit > numParams ? numFit - numParams : 0;
    }

    /// Reduced chi^2 of the fit (chi2/dof), or NaN if there are no degrees of freedom
    double getReducedChi2() const;

    /// Unweighted first moment (centroid) of the kernel, as an (x, y) offset from the kernel center
    ///
    /// This is the kernel's "center of mass": zero for a kernel that is symmetric about its center,
    /// and non-zero if the fit found an overall positional shift between source and target.
    /// Returns (NaN, NaN) if the fit failed or the kernel sum is zero.
    lsst::geom::Point2D getFirstMoment() const;

    /// Unweighted second moment (shape) of the kernel, about its first moment
    ///
    /// This is a simple (unweighted, non-adaptive) moment calculation, analogous to
    /// pfs::drp::stella::OversampledPsf::doComputeShape. Because the delta-function kernel can take
    /// negative values (unlike a physical PSF), the result is not guaranteed to be positive-definite.
    /// Returns NaN components if the fit failed or the kernel sum is zero.
    lsst::afw::geom::ellipses::Quadrupole getSecondMoment() const;
};


/// Print a summary of the solution to a stream
std::ostream& operator<<(std::ostream& os, KernelSolution const& solution);


/// Result of fitting a spatially-varying kernel across an entire image
///
/// The two input images are divided into a grid of numRegionsX x numRegionsY
/// regions, and a KernelSolution is fit independently in each region (see
/// fitAlardLuptonKernel). This provides the spatial variation of the kernel
/// across the image, in place of the spatial polynomial used by the original
/// Alard & Lupton (1998) algorithm.
struct AlardLuptonResult {
    lsst::afw::image::MaskedImage<float> convolved;  ///< Source image convolved with the fitted kernel(s)
        ///< and differential background added, i.e. the model for target (also known as "matched")
    lsst::afw::image::MaskedImage<float> difference;  ///< Difference image (target - convolved)
    std::vector<KernelSolution> solutions;  ///< Kernel solution for each region, in row-major (y, x) order
    int numRegionsX;  ///< Number of regions in x
    int numRegionsY;  ///< Number of regions in y
    int kernelHalfWidth;  ///< Half-width of the kernel used for the fit
    double commonKernelSum;  ///< Robust common kernel-sum prior used to constrain the regions, or NaN
        ///< if the common-kernel-sum constraint was not requested (see fitAlardLuptonKernel)
    double commonKernelSumScatter;  ///< Robust scatter of the per-region kernel sums about
        ///< commonKernelSum, or NaN if the common-kernel-sum constraint was not requested

    /// Ctor
    AlardLuptonResult(
        lsst::afw::image::MaskedImage<float> const& convolved,
        lsst::afw::image::MaskedImage<float> const& difference,
        std::vector<KernelSolution> const& solutions,
        int numRegionsX,
        int numRegionsY,
        int kernelHalfWidth,
        double commonKernelSum=std::numeric_limits<double>::quiet_NaN(),
        double commonKernelSumScatter=std::numeric_limits<double>::quiet_NaN()
    );

    /// Get the solution for the region containing the point (x, y)
    ///
    /// Returns a null pointer if the point is not within any region (e.g., it
    /// is within kernelHalfWidth of the edge of the image).
    KernelSolution const* getSolutionAt(int xx, int yy) const;

    //@{
    /// Aggregate statistics, summed over all regions
    double getChi2() const;
    std::size_t getNumFit() const;
    std::size_t getNumRejected() const;
    //@}

    /// Unweighted first moments (centroids) of the fitted kernel in each region
    ///
    /// See KernelSolution::getFirstMoment.
    ///
    /// @return (x, y) components of the first moment, each of shape (numRegionsY, numRegionsX), in the
    ///     same row-major (y, x) order as `solutions`
    std::pair<ndarray::Array<double, 2, 2>, ndarray::Array<double, 2, 2>> getKernelFirstMoments() const;

    /// Unweighted second moments (shapes) of the fitted kernel in each region
    ///
    /// See KernelSolution::getSecondMoment.
    ///
    /// @return (xx, yy, xy) components of the second moment, each of shape (numRegionsY, numRegionsX)
    std::tuple<ndarray::Array<double, 2, 2>, ndarray::Array<double, 2, 2>, ndarray::Array<double, 2, 2>>
        getKernelSecondMoments() const;
};


/// Fit a spatially-varying kernel that convolves 'source' to match 'target'
///
/// This implements image differencing following Alard & Lupton (1998), using
/// a delta-function basis for the kernel (i.e., each kernel pixel is an
/// independent free parameter) instead of their Gauss-Hermite basis.
/// Spatial variation of the kernel is achieved by dividing the images into a
/// grid of numRegionsX x numRegionsY regions and fitting a kernel
/// independently in each (rather than fitting a single kernel with
/// spatially-varying coefficients, as in the original algorithm). A
/// low-order 2D polynomial "differential background" is fit simultaneously
/// with the kernel in each region, to account for any residual sky/bias
/// offset between the images.
///
/// Within each region, we perform a linear least-squares fit for the kernel
/// and background, weighting by the combined variance of the two images.
/// Pixels are then rejected (sigma-clipped) if their fit residual exceeds
/// rejThresh standard deviations, and the fit is repeated; this continues
/// for up to rejIter rounds (or until no further pixels are rejected).
///
/// A pixel of the output convolved and difference images can be computed
/// only if it lies at least kernelHalfWidth from the edge of the images (so
/// the kernel footprint centred on it lies entirely within the images) and
/// it has usable target data (i.e., is finite, has finite positive
/// variance, and is not flagged with any of the bits in badBitMask). If, in
/// addition, every source pixel within the kernel footprint has usable
/// data, the pixel is computed directly from the full footprint, as in
/// Alard & Lupton (1998). If some (but not all) of those source pixels are
/// unusable, the pixel is instead computed from only the usable source
/// pixels in the footprint, with the result rescaled by the ratio of the
/// full kernel sum to the sum of the kernel weights actually used
/// (compensating for the missing flux under the assumption that the source
/// is locally flat over the footprint); such pixels are flagged with the
/// "DIFFIM_PARTIAL" mask plane. A pixel is flagged "NO_DATA" instead if it
/// is too close to the edge of the images, its target data is unusable,
/// none (or too little) of the source data within its footprint is usable,
/// or it fails for other reasons (e.g. an entire region having too few good
/// pixels to constrain the fit). Pixels that were rejected during the fit
/// are flagged with the "DIFFIM_REJECTED" mask plane. The convolved and
/// difference images share identical masks (aside from their variance
/// planes: the convolved image's variance is that of the model alone,
/// while the difference image's variance also includes the target's). All
/// other mask planes are propagated from the two input images (bitwise-
/// OR'd together).
///
/// If commonKernelSum is true, the kernel sum (the overall flux
/// normalization; see KernelSolution::getKernelSum) is constrained to be the
/// same in every region, using a two-pass approach: regions are first fit
/// independently (as when commonKernelSum is false), a robust common kernel
/// sum and its scatter are estimated from the successful regions, and then
/// every region is refit with a pseudo-measurement pulling its kernel sum
/// towards that common value, weighted by the derived scatter. This is a
/// soft constraint: a region whose data genuinely disagrees with the shared-
/// value assumption will show it via degraded chi2, rather than being
/// silently forced to match. The rejection of discrepant pixels is
/// performed identically to the unconstrained case, before the constraint is
/// applied, so it cannot bias which pixels are used. If fewer than two
/// regions fit successfully, the constraint cannot be estimated and is
/// silently skipped (equivalent to commonKernelSum=false). The derived
/// common value and its scatter are recorded in
/// AlardLuptonResult::commonKernelSum and
/// AlardLuptonResult::commonKernelSumScatter (NaN if the constraint was not
/// applied).
///
/// @param source : Image to be convolved to match target (e.g., a template image)
/// @param target : Image to match (e.g., a science image)
/// @param kernelHalfWidth : Half-width of the kernel in x and y
/// @param numRegionsX : Number of regions in x, for spatial variation of the kernel
/// @param numRegionsY : Number of regions in y, for spatial variation of the kernel
/// @param backgroundOrder : Order of the differential background polynomial fit in each region
/// @param badBitMask : Mask bits that indicate a pixel should not be used
/// @param rejIter : Number of rejection iterations to perform in each region
/// @param rejThresh : Rejection threshold (standard deviations)
/// @param lsqThreshold : Threshold (relative to the largest eigenvalue) for singular values to be
///     ignored when solving the least-squares matrix equation
/// @param commonKernelSum : Constrain the kernel sum to be the same in every region (see above)
/// @return the convolved (matched source) image, the difference image, and the kernel solution for each
///     region
/// @throws lsst::pex::exceptions::LengthError if source and target have different bounding boxes, or if
///     the images are too small to fit a kernel of the requested half-width
/// @throws lsst::pex::exceptions::InvalidParameterError if kernelHalfWidth, numRegionsX, numRegionsY,
///     backgroundOrder, rejIter, rejThresh or lsqThreshold is out of range
AlardLuptonResult fitAlardLuptonKernel(
    lsst::afw::image::MaskedImage<float> const& source,
    lsst::afw::image::MaskedImage<float> const& target,
    int kernelHalfWidth=10,
    int numRegionsX=1,
    int numRegionsY=1,
    int backgroundOrder=1,
    lsst::afw::image::MaskPixel badBitMask=0,
    int rejIter=2,
    double rejThresh=3.0,
    double lsqThreshold=1.0e-6,
    bool commonKernelSum=false
);


}}}  // namespace pfs::drp::stella

#endif  // include guard
