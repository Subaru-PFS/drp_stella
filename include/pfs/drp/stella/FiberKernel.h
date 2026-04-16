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


class FiberKernel {
  public:
    using Polynomial = math::NormalizedPolynomial2<double>;

    FiberKernel(
        lsst::geom::Box2D const& range,
        int halfWidth,
        int order,
        ndarray::ArrayRef<double const, 1, 1> const& coefficients
    );

    int getHalfWidth() const { return _halfWidth; }
    int getOrder() const { return _order; }
    std::size_t getNumPoly() const { return _numPoly; }
    std::size_t getNumParams() const { return _numParams; }
    ndarray::Array<double, 1, 1> getCoefficients() const { return _coefficients; }
    std::vector<Polynomial> const& getPolynomials() const { return _polynomials; }

    std::shared_ptr<FiberTrace<float>> operator()(
        FiberTrace<float> const& trace,
        lsst::geom::Box2I const& bbox
    ) const;
    FiberTraceSet<float> operator()(
        FiberTraceSet<float> const& trace,
        lsst::geom::Box2I const& bbox
    ) const;

    ndarray::Array<double, 1, 1> evaluate(double x, double y) const;
    ndarray::Array<double, 1, 1> evaluate(lsst::geom::Point2D const& xy) const {
        return evaluate(xy.getX(), xy.getY());
    }

    std::vector<lsst::afw::image::Image<double>> makeOffsetImages(lsst::geom::Extent2I const& dims) const;
    std::vector<lsst::afw::image::Image<double>> makeOffsetImages(int width, int height) const {
        return makeOffsetImages(lsst::geom::Extent2I(width, height));
    }

  private:
    int _halfWidth;
    int _order;
    std::size_t _numPoly;
    std::size_t _numParams;
    ndarray::Array<double, 1, 1> _coefficients;
    std::vector<Polynomial> _polynomials;
};


std::tuple<FiberKernel, lsst::afw::image::Image<float>, ndarray::Array<double, 2, 2>> fitFiberKernel(
    lsst::afw::image::MaskedImage<float> const& image,
    FiberTraceSet<float> const& fiberTraces,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int kernelOrder,
    int xBackgroundSize,
    int yBackgroundSize,
    ndarray::Array<int, 1, 1> const& rows,
    int maxIter=20,
    int andersonDepth=5,
    double fluxTol=1.0e-3,
    double lsqThreshold=1.0e-16
);


//std::pair<



}}}  // namespace pfs::drp::stella

#endif  // include guard
