#ifndef PFS_DRP_STELLA_FIBERKERNEL_H
#define PFS_DRP_STELLA_FIBERKERNEL_H

#include <utility>  // for std::pair

#include "lsst/afw/image/MaskedImage.h"

#include "pfs/drp/stella/FiberTraceSet.h"
#include "pfs/drp/stella/SpectrumSet.h"
#include "pfs/drp/stella/math/NormalizedPolynomial.h"


namespace pfs {
namespace drp {
namespace stella {

namespace detail {


class PiecewiseConstantInterpolator {
  public:
    PiecewiseConstantInterpolator(
        lsst::geom::Extent2I const& dims,
        int xNumBlocks,
        int yNumBlocks,
        std::size_t step=1
    );

    std::size_t getBlock(int x, int y) const {
        return _yBlocks[y] + _xBlocks[x];
    }
    std::size_t getBlock(double x, int y) const {
        return getBlock(static_cast<int>(std::round(x)), y);
    }
    std::size_t getBlock(double x, double y) const {
        return getBlock(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
    }

    std::size_t getIndex(int x, int y) const {
        return _yIndex[y] + _xIndex[x];
    }
    std::size_t getIndex(double x, int y) const {
        return getIndex(static_cast<int>(std::round(x)), y);
    }
    std::size_t getIndex(double x, double y) const {
        return getIndex(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
    }

    std::size_t getXNumBlocks() const { return _xNumBlocks; }
    std::size_t getYNumBlocks() const { return _yNumBlocks; }
    std::size_t getNumBlocks() const { return _xNumBlocks*_yNumBlocks; }
    ndarray::Array<std::size_t const, 1, 1> getXBlocks() const { return _xBlocks; }
    ndarray::Array<std::size_t const, 1, 1> getYBlocks() const { return _yBlocks; }
    ndarray::Array<std::size_t const, 1, 1> getXIndex() const { return _xIndex; }
    ndarray::Array<std::size_t const, 1, 1> getYIndex() const { return _yIndex; }

  private:
    std::size_t _xNumBlocks;
    std::size_t _yNumBlocks;
    std::size_t _step;
    ndarray::Array<std::size_t, 1, 1> _xBlocks;
    ndarray::Array<std::size_t, 1, 1> _yBlocks;
    ndarray::Array<std::size_t, 1, 1> _xIndex;
    ndarray::Array<std::size_t, 1, 1> _yIndex;
};


}  // namespace detail


class BaseKernel {
  public:
    using Polynomial = math::NormalizedPolynomial2<double>;

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
        ndarray::ArrayRef<double const, 1, 1> const& coefficients
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
    virtual ndarray::Array<double, 3, 3> makeOffsetImagesImpl(lsst::geom::Extent2I const& dims) const = 0;
};


class FiberKernel : public BaseKernel {
  public:

    FiberKernel(
        lsst::geom::Extent2I const& dims,
        int halfWidth,
        int xNumBlocks,
        int yNumBlocks,
        ndarray::ArrayRef<double const, 1, 1> const& coefficients
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

    detail::PiecewiseConstantInterpolator _interp;
};


std::tuple<FiberKernel, lsst::afw::image::Image<float>, ndarray::Array<double, 2, 2>> fitFiberKernel(
    lsst::afw::image::MaskedImage<float> const& image,
    FiberTraceSet<float> const& fiberTraces,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int xKernelNum,
    int yKernelNum,
    int xBackgroundSize,
    int yBackgroundSize,
    ndarray::Array<int, 1, 1> const& rows=ndarray::Array<int, 1, 1>(),
    int maxIter=20,
    int andersonDepth=5,
    double fluxTol=1.0e-3,
    double lsqThreshold=1.0e-16
);


std::pair<FiberKernel, lsst::afw::image::Image<float>> fitFiberKernel(
    lsst::afw::image::MaskedImage<float> const& source,
    lsst::afw::image::MaskedImage<float> const& target,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int xKernelNum,
    int yKernelNum,
    int xBackgroundSize,
    int yBackgroundSize,
    ndarray::Array<int, 1, 1> const& rows=ndarray::Array<int, 1, 1>(),
    double lsqThreshold=1.0e-16
);


}}}  // namespace pfs::drp::stella

#endif  // include guard
