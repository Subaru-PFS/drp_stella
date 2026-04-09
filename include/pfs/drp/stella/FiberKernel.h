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

    std::shared_ptr<FiberTrace<float>> operator()(FiberTrace<float> const& trace) const;
    FiberTraceSet<float> operator()(FiberTraceSet<float> const& trace) const;

  private:
    int _halfWidth;
    int _order;
    std::size_t _numPoly;
    std::size_t _numParams;
    ndarray::Array<double, 1, 1> _coefficients;
    std::vector<Polynomial> _polynomials;
};


std::tuple<FiberKernel, lsst::afw::image::Image<double>> fitFiberKernel(
    lsst::afw::image::MaskedImage<float> const& image,
    FiberTraceSet<float> const& fiberTraces,
    SpectrumSet const& spectra,
    lsst::afw::image::MaskPixel badBitMask,
    int kernelHalfWidth,
    int kernelOrder,
    int xBackgroundSize,
    int yBackgroundSize
);


}}}  // namespace pfs::drp::stella

#endif  // include guard
