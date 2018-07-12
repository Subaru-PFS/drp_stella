#include "ndarray.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/Controls.h"

namespace pfs {
namespace drp {
namespace stella {

ndarray::Array<std::size_t, 2, -2>
FiberTraceFunction::calcMinCenMax(ndarray::Array<float, 1, 1> const& xCenters) {
    std::size_t const num = xCenters.getNumElements();
    ndarray::Array<float, 1, 1> tempCenter = ndarray::allocate(num);
    tempCenter.deep() = xCenters + 0.5;

    ndarray::Array<std::size_t, 2, -2> minCenMax = ndarray::allocate(num, 3);

    minCenMax[ndarray::view()(0)] = math::floor<std::size_t>(ndarray::copy(tempCenter + ctrl.xLow));
    minCenMax[ndarray::view()(1)] = math::floor<std::size_t>(tempCenter);
    minCenMax[ndarray::view()(2)] = math::floor<std::size_t>(ndarray::copy(tempCenter + ctrl.xHigh));

    auto leftPixels = minCenMax[ndarray::view()(1)] - minCenMax[ndarray::view()(0)];
    auto rightPixels = minCenMax[ndarray::view()(2)] - minCenMax[ndarray::view()(1)];

    auto const leftMinMax = std::minmax_element(leftPixels.begin(), leftPixels.end());
    auto const rightMinMax = std::minmax_element(rightPixels.begin(), rightPixels.end());
    if (*leftMinMax.second > *leftMinMax.first) {
        minCenMax[ndarray::view()(0)] = minCenMax[ndarray::view()(1)] - *leftMinMax.second + ctrl.nPixCutLeft;
    }
    if (*rightMinMax.second > *rightMinMax.first) {
        minCenMax[ndarray::view()(2)] = minCenMax[ndarray::view()(1)] + *rightMinMax.second - ctrl.nPixCutRight;
    }

    return minCenMax;
}

}}} // namespace pfs::drp::stella
