#ifndef PFS_DRP_STELLA_ALARDLUPTON_H
#define PFS_DRP_STELLA_ALARDLUPTON_H

#include <cstddef>
#include <ostream>
#include <vector>

#include "ndarray_fwd.h"
#include "lsst/geom/Box.h"
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
    lsst::afw::image::MaskedImage<float> difference;  ///< Difference image (target - matched source)
    std::vector<KernelSolution> solutions;  ///< Kernel solution for each region, in row-major (y, x) order
    int numRegionsX;  ///< Number of regions in x
    int numRegionsY;  ///< Number of regions in y
    int kernelHalfWidth;  ///< Half-width of the kernel used for the fit

    /// Ctor
    AlardLuptonResult(
        lsst::afw::image::MaskedImage<float> const& difference,
        std::vector<KernelSolution> const& solutions,
        int numRegionsX,
        int numRegionsY,
        int kernelHalfWidth
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
/// A pixel of the output difference image can be computed only if it, and
/// every source pixel within the kernel footprint centred on it, has usable
/// data (i.e., is finite, has finite positive variance, and is not flagged
/// with any of the bits in badBitMask); this means a border of
/// kernelHalfWidth pixels around the edge of the images is not computed. Such
/// pixels, along with any that fail for other reasons (e.g. an entire region
/// having too few good pixels to constrain the fit) are flagged with the
/// "NO_DATA" mask plane in the output, and pixels that were rejected during
/// the fit are flagged with the "DIFFIM_REJECTED" mask plane. All other mask
/// planes are propagated from the two input images (bitwise-OR'd together).
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
/// @return the difference image and the kernel solution for each region
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
    double lsqThreshold=1.0e-6
);


}}}  // namespace pfs::drp::stella

#endif  // include guard
