#ifndef PFS_DRP_STELLA_FIBERKERNEL_H
#define PFS_DRP_STELLA_FIBERKERNEL_H

#include <utility>  // for std::pair

#include "lsst/afw/image/MaskedImage.h"

#include "pfs/drp/stella/FiberTraceSet.h"
#include "pfs/drp/stella/SpectrumSet.h"
#include "pfs/drp/stella/math/NormalizedPolynomial.h"
#include "pfs/drp/stella/GridTransform.h"


namespace pfs {
namespace drp {
namespace stella {

namespace detail {


template <typename T>
class LinearInterpolationHelper {
  public:
    using IndexWeightPair = std::pair<std::size_t, T>;

    LinearInterpolationHelper(
        ndarray::Array<T, 1, 1> const& x,
        std::size_t length
    );

    LinearInterpolationHelper(LinearInterpolationHelper const&) = default;
    LinearInterpolationHelper(LinearInterpolationHelper&&) = default;
    LinearInterpolationHelper& operator=(LinearInterpolationHelper const&) = default;
    LinearInterpolationHelper& operator=(LinearInterpolationHelper&&) = default;
    ~LinearInterpolationHelper() = default;

    ndarray::Array<T, 1, 1> getX() const { return _x; }
    std::size_t getLength() const { return _length; }

    std::pair<IndexWeightPair, IndexWeightPair> operator()(std::size_t x) const;


  private:
    ndarray::Array<T, 1, 1> _x;
    std::size_t _length;
    ndarray::Array<T, 1, 1> _index;
    ndarray::Array<T, 1, 1> _weight;
};


}  // namespace detail


class BaseKernel {
  public:
    using Polynomial = math::NormalizedPolynomial2<double>;

    BaseKernel(BaseKernel const&) = default;
    BaseKernel(BaseKernel &&) = default;
    BaseKernel & operator=(BaseKernel const&) = default;
    BaseKernel & operator=(BaseKernel &&) = default;
    virtual ~BaseKernel() = default;

    int getHalfWidth() const { return _halfWidth; }
    std::size_t getNumParams() const { return _numParams; }
    ndarray::Array<double, 1, 1> getCoefficients() const { return _coefficients; }

    std::shared_ptr<FiberTrace<float>> convolve(
        FiberTrace<float> const& trace,
        lsst::geom::Box2I const& bbox
    ) const {
        return convolveImpl(trace, bbox);
    }
    FiberTraceSet<float> convolve(
        FiberTraceSet<float> const& traces,
        lsst::geom::Box2I const& bbox
    ) const;
    lsst::afw::image::Image<float> convolve(lsst::afw::image::Image<float> const& image) const {
        return convolveImpl(image);
    }

    ndarray::Array<double, 3, 3> makeOffsetImages(lsst::geom::Extent2I const& dims) const {
        return makeOffsetImagesImpl(dims);
    }
    ndarray::Array<double, 3, 3> makeOffsetImages(int width, int height) const {
        return makeOffsetImages(lsst::geom::Extent2I(width, height));
    }

  protected:
    BaseKernel(
        lsst::geom::Extent2I const& dims,
        int halfWidth,
        std::size_t numParams,
        ndarray::Array<double const, 1, 1> const& coefficients
    );

    lsst::geom::Extent2I _dims;
    int _halfWidth;
    std::size_t _numParams;  ///< number of parameters in the kernel
    ndarray::Array<double, 1, 1> _coefficients;

  private:
    virtual std::shared_ptr<FiberTrace<float>> convolveImpl(
        FiberTrace<float> const& trace,
        lsst::geom::Box2I const& bbox
    ) const = 0;
    virtual lsst::afw::image::Image<float> convolveImpl(
        lsst::afw::image::Image<float> const& image
    ) const = 0;
    virtual ndarray::Array<double, 3, 3> makeOffsetImagesImpl(
        lsst::geom::Extent2I const& dims
    ) const = 0;
};


class FiberKernel : public BaseKernel {
  public:

    FiberKernel(
        lsst::geom::Extent2I const& dims,
        int halfWidth,
        int xNumBlocks,
        int yNumBlocks,
        ndarray::Array<double const, 1, 1> const& coefficients
    );

    ndarray::Array<double, 1, 1> evaluate(double x, double y) const;
    ndarray::Array<double, 1, 1> evaluate(lsst::geom::Point2D const& xy) const {
        return evaluate(xy.getX(), xy.getY());
    }

  private:
    std::shared_ptr<FiberTrace<float>> convolveImpl(
        FiberTrace<float> const& trace,
        lsst::geom::Box2I const& bbox
    ) const override;
    lsst::afw::image::Image<float> convolveImpl(
        lsst::afw::image::Image<float> const& image
    ) const override;
    ndarray::Array<double, 3, 3> makeOffsetImagesImpl(
        lsst::geom::Extent2I const& dims
    ) const override;

    template <int C>
    void _evaluate(
        ndarray::Array<double, 1, C> & result,
        double x,
        double y
    ) const;

    int _xNumBlocks;
    int _yNumBlocks;
    detail::LinearInterpolationHelper<double> _xInterp;
    detail::LinearInterpolationHelper<double> _yInterp;
};


FiberKernel fitFiberKernel(
    lsst::afw::image::MaskedImage<float> const& image,
    FiberTraceSet<float> const& fiberTraces,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int xKernelNum,
    int yKernelNum,
    ndarray::Array<int, 1, 1> const& rows=ndarray::Array<int, 1, 1>(),
    int maxIter=20,
    int andersonDepth=5,
    double fluxTol=1.0e-3,
    double lsqThreshold=1.0e-16
);


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
