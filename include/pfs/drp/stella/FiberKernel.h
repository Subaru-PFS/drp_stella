#ifndef PFS_DRP_STELLA_FIBERKERNEL_H
#define PFS_DRP_STELLA_FIBERKERNEL_H

#include <utility>  // for std::pair

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/log/Log.h"

#include "pfs/drp/stella/FiberTraceSet.h"


namespace pfs {
namespace drp {
namespace stella {

namespace detail {


/// Helper class for linear interpolation of kernel blocks
template <typename T>
class LinearInterpolationHelper {
  public:
    using IndexWeightPair = std::pair<std::size_t, T>;

    /// Ctor
    ///
    /// @param x : x-coordinates of the block centers (must be monotonic increasing)
    /// @param length : length of the output array to interpolate to
    LinearInterpolationHelper(
        ndarray::Array<T, 1, 1> const& x,
        std::size_t length
    );

    LinearInterpolationHelper(LinearInterpolationHelper const&) = default;
    LinearInterpolationHelper(LinearInterpolationHelper&&) = default;
    LinearInterpolationHelper& operator=(LinearInterpolationHelper const&) = default;
    LinearInterpolationHelper& operator=(LinearInterpolationHelper&&) = default;
    ~LinearInterpolationHelper() = default;

    //@{
    /// Accessors
    ndarray::Array<T, 1, 1> getX() const { return _x; }
    std::size_t getLength() const { return _length; }
    //@}

    /// Get the indices and weights for linear interpolation at pixel x
    ///
    /// We return the indices and weights of the two blocks to interpolate
    /// between. The caller can then compute the interpolated value as:
    /// weight1*array[index1] + weight2*array[index2].
    std::pair<IndexWeightPair, IndexWeightPair> operator()(std::size_t x) const;

  private:
    ndarray::Array<T, 1, 1> _x;  ///< x-coordinates of the block centers
    std::size_t _length;  ///< length of the output array to interpolate to
    ndarray::Array<T, 1, 1> _index;  ///< index of the left block center for each pixel
    ndarray::Array<T, 1, 1> _weight;  ///< weight of the left block center for each pixel
};


}  // namespace detail


/// Base class for 1D kernels, used to convolve fiber traces and images.
///
/// This sets the interface, and provides holding for parameter values.
///
/// Subclasses must implement the following pure-virtual methods:
/// * convolveImpl for FiberTrace, Image, and MaskedImage
/// * makeOffsetImagesImpl
class BaseKernel {
  public:
    BaseKernel(BaseKernel const&) = default;
    BaseKernel(BaseKernel &&) = default;
    BaseKernel & operator=(BaseKernel const&) = default;
    BaseKernel & operator=(BaseKernel &&) = default;
    virtual ~BaseKernel() = default;

    //@{
    /// Accessors
    lsst::geom::Extent2I getDims() const { return _dims; }
    int getHalfWidth() const { return _halfWidth; }
    std::size_t getNumParams() const { return _numParams; }
    ndarray::Array<double, 1, 1> getValues() const { return _values; }
    //@}

    //@{
    /// Convolve fiber traces and images with the kernel
    std::shared_ptr<FiberTrace<float>> convolve(
        FiberTrace<float> const& trace,  ///< trace to convolve
        lsst::geom::Box2I const& bbox  ///< bounding box of image
    ) const {
        return convolveImpl(trace, bbox);
    }
    FiberTraceSet<float> convolve(
        FiberTraceSet<float> const& traces,  ///< traces to convolve
        lsst::geom::Box2I const& bbox  ///< bounding box of image
    ) const;
    lsst::afw::image::Image<float> convolve(lsst::afw::image::Image<float> const& image) const {
        return convolveImpl(image);
    }
    lsst::afw::image::MaskedImage<float> convolve(
        lsst::afw::image::MaskedImage<float> const& image,  ///< image to convolve
        lsst::afw::image::MaskPixel badBitMask,  ///< bitmask for bad pixels
        double maskThreshold=0.1  ///< Minimum fraction of good pixels for successful convolution
    ) const {
        return convolveImpl(image, badBitMask, maskThreshold);
    }
    //@}

    //@{
    /// Make images of the kernel at each integer offset within the half-width
    ndarray::Array<double, 3, 3> makeOffsetImages(lsst::geom::Extent2I const& dims) const {
        return makeOffsetImagesImpl(dims);
    }
    ndarray::Array<double, 3, 3> makeOffsetImages(int width, int height) const {
        return makeOffsetImages(lsst::geom::Extent2I(width, height));
    }
    //@}

  protected:

    /// Ctor
    ///
    /// @param dims : image dimensions
    /// @param halfWidth : half-width of the kernel in x
    /// @param numParams : number of parameters in the kernel
    /// @param values : kernel parameter values (length must be numParams)
    BaseKernel(
        lsst::geom::Extent2I const& dims,
        int halfWidth,
        std::size_t numParams,
        ndarray::Array<double const, 1, 1> const& values
    );

    lsst::geom::Extent2I _dims;  ///< image dimensions
    int _halfWidth;  ///< half-width of the kernel in x
    std::size_t _numParams;  ///< number of parameters in the kernel
    ndarray::Array<double, 1, 1> _values;  ///< kernel parameter values

  private:

    //@{
    /// convolve() implementations
    virtual std::shared_ptr<FiberTrace<float>> convolveImpl(
        FiberTrace<float> const& trace,
        lsst::geom::Box2I const& bbox
    ) const = 0;
    virtual lsst::afw::image::Image<float> convolveImpl(
        lsst::afw::image::Image<float> const& image
    ) const = 0;
    virtual lsst::afw::image::MaskedImage<float> convolveImpl(
        lsst::afw::image::MaskedImage<float> const& image,
        lsst::afw::image::MaskPixel badBitMask,
        double maskThreshold
    ) const = 0;
    //@}

    /// makeOffsetImages() implementation
    virtual ndarray::Array<double, 3, 3> makeOffsetImagesImpl(
        lsst::geom::Extent2I const& dims
    ) const = 0;
};


///
class FiberKernel : public BaseKernel {
  public:

    /// Ctor
    ///
    /// @param dims : image dimensions
    /// @param halfWidth : half-width of the kernel in x
    /// @param xNumBlocks : number of blocks in x to use for interpolation
    /// @param yNumBlocks : number of blocks in y to use for interpolation
    /// @param values : kernel parameter values (length must be (2*halfWidth)*xNumBlocks*yNumBlocks)
    FiberKernel(
        lsst::geom::Extent2I const& dims,
        int halfWidth,
        int xNumBlocks,
        int yNumBlocks,
        ndarray::Array<double const, 1, 1> const& values
    );

    //@{
    /// Accessors
    int getXNumBlocks() const { return _xNumBlocks; }
    int getYNumBlocks() const { return _yNumBlocks; }
    //@}

    //@{
    /// Evaluate the kernel at position (x, y)
    ndarray::Array<double, 1, 1> evaluate(double x, double y) const;
    ndarray::Array<double, 1, 1> evaluate(lsst::geom::Point2D const& xy) const {
        return evaluate(xy.getX(), xy.getY());
    }
    //@}

  private:
    //@{
    /// Virtual method implementations
    std::shared_ptr<FiberTrace<float>> convolveImpl(
        FiberTrace<float> const& trace,
        lsst::geom::Box2I const& bbox
    ) const override;
    lsst::afw::image::Image<float> convolveImpl(
        lsst::afw::image::Image<float> const& image
    ) const override;
    lsst::afw::image::MaskedImage<float> convolveImpl(
        lsst::afw::image::MaskedImage<float> const& image,
        lsst::afw::image::MaskPixel badBitMask,
        double maskThreshold
    ) const override;
    ndarray::Array<double, 3, 3> makeOffsetImagesImpl(
        lsst::geom::Extent2I const& dims
    ) const override;
    //@}

    /// Evaluate the kernel at position (x, y), modifying the result array in-place
    template <int C>
    void _evaluate(
        ndarray::Array<double, 1, C> & result,
        double x,
        double y
    ) const;

    int _xNumBlocks;  ///< number of blocks in x to use for interpolation
    int _yNumBlocks;  ///< number of blocks in y to use for interpolation
    detail::LinearInterpolationHelper<double> _xInterp;  ///< interpolation helper for x
    detail::LinearInterpolationHelper<double> _yInterp;  ///< interpolation helper for y
};


/// Fit a kernel that convolves the fiberTraces to match the image
///
/// This involves alternating least-squares fitting of the kernel and the
/// fluxes.
///
/// @param image : image to match
/// @param fiberTraces : fiber traces to convolve
/// @param badBitMask : bitmask for bad pixels in image
/// @param kernelHalfWidth : half-width of the kernel in x
/// @param xKernelNum : number of blocks in x to use for interpolation
/// @param yKernelNum : number of blocks in y to use for interpolation
/// @param rows : rows to use for fitting (if empty, use all rows)
/// @param maxIter : maximum number of iterations to perform
/// @param andersonDepth : number of previous iterations to use for Anderson acceleration (0 to disable)
/// @param andersonDamping : damping factor to use for Anderson acceleration (between 0 and 1)
/// @param fluxTol : tolerance for convergence of fluxes (relative change)
/// @param lsqThreshold : threshold for least-squares fit SVD
std::pair<FiberKernel, lsst::afw::image::Image<float>> fitFiberKernel(
    lsst::afw::image::MaskedImage<float> const& image,
    FiberTraceSet<float> const& fiberTraces,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int xKernelNum,
    int yKernelNum,
    ndarray::Array<int, 1, 1> const& rows=ndarray::Array<int, 1, 1>(),
    int maxIter=20,
    int andersonDepth=5,
    double andersonDamping=0.25,
    double fluxTol=1.0e-3,
    double lsqThreshold=1.0e-16
);


/// Fit a kernel that convolves the source image to match the target image
///
/// @param source : image to convolve
/// @param target : image to match
/// @param badBitMask : bitmask for bad pixels in source and target
/// @param kernelHalfWidth : half-width of the kernel in x
/// @param xKernelNum : number of blocks in x to use for interpolation
/// @param yKernelNum : number of blocks in y to use for interpolation
/// @param rows : rows to use for fitting (if empty, use all rows)
/// @param lsqThreshold : threshold for least-squares fit SVD
FiberKernel fitFiberKernel(
    lsst::afw::image::MaskedImage<float> const& source,
    lsst::afw::image::MaskedImage<float> const& target,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int xKernelNum,
    int yKernelNum,
    ndarray::Array<int, 1, 1> const& rows=ndarray::Array<int, 1, 1>(),
    double lsqThreshold=1.0e-16
);


}}}  // namespace pfs::drp::stella

#endif  // include guard
