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


class PolynomialKernel {
  public:
    using Polynomial = math::NormalizedPolynomial2<double>;

    int getHalfWidth() const { return _halfWidth; }
    int getOrder() const { return _order; }
    std::size_t getNumPoly() const { return _numPoly; }
    std::size_t getNumParams() const { return _numParams; }
    ndarray::Array<double, 1, 1> getCoefficients() const { return _coefficients; }
    std::vector<Polynomial> const& getPolynomials() const { return _polynomials; }

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
    PolynomialKernel(
        lsst::geom::Box2D const& range,
        int halfWidth,
        int order,
        std::size_t numPoly,
        ndarray::ArrayRef<double const, 1, 1> const& coefficients
    );

    int _halfWidth;
    int _order;
    std::size_t _numCoeffs;  ///< number of coefficients in each polynomial
    std::size_t _numPoly;  ///< number of polynomials
    std::size_t _numParams;  ///< number of parameters in the kernel (should be numPoly * numOffsets)
    ndarray::Array<double, 1, 1> _coefficients;
    std::vector<Polynomial> _polynomials;

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


class FiberKernel : public PolynomialKernel {
  public:

    FiberKernel(
        lsst::geom::Box2D const& range,
        int halfWidth,
        int order,
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
};


class ImageKernel : public PolynomialKernel {
  public:
    ImageKernel(
        lsst::geom::Box2D const& range,
        int halfWidth,
        int order,
        ndarray::ArrayRef<double const, 1, 1> const& coefficients
    );

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
};


std::tuple<FiberKernel, lsst::afw::image::Image<float>, ndarray::Array<double, 2, 2>> fitFiberKernel(
    lsst::afw::image::MaskedImage<float> const& image,
    FiberTraceSet<float> const& fiberTraces,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int kernelOrder,
    int xBackgroundSize,
    int yBackgroundSize,
    ndarray::Array<int, 1, 1> const& rows=ndarray::Array<int, 1, 1>(),
    int maxIter=20,
    int andersonDepth=5,
    double fluxTol=1.0e-3,
    double lsqThreshold=1.0e-16
);


std::pair<ImageKernel, lsst::afw::image::Image<float>> fitImageKernel(
    lsst::afw::image::MaskedImage<float> const& source,
    lsst::afw::image::MaskedImage<float> const& target,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int kernelOrder,
    int xBackgroundSize,
    int yBackgroundSize,
    ndarray::Array<int, 1, 1> const& rows=ndarray::Array<int, 1, 1>(),
    double lsqThreshold=1.0e-16
);


}}}  // namespace pfs::drp::stella

#endif  // include guard
