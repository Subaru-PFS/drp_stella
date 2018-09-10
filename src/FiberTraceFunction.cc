#include <algorithm>
#include "ndarray.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/Controls.h"
#include "pfs/drp/stella/math/CurveFitting.h"

namespace pfs {
namespace drp {
namespace stella {

ndarray::Array<std::size_t, 2, -2>
FiberTraceFunction::calcMinCenMax(ndarray::Array<float const, 1, 1> const& xCenters) {
    std::size_t const num = xCenters.getNumElements();
    ndarray::Array<float, 1, 1> tempCenter = ndarray::allocate(num);
    tempCenter.deep() = xCenters + 0.5;
    // Check that we're not going to have unsigned problems when we convert to size_t.
    assert(std::all_of(tempCenter.begin(), tempCenter.end(),
           [this](auto const xx) { return xx > - ctrl.xLow; }));

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


ndarray::Array<float, 1, 1>
FiberTraceFunction::calculateXCenters() const {
    ndarray::Array<float, 1, 1> xRowIndex = ndarray::allocate(yHigh - yLow + 1);
    float xRowInd = yCenter + yLow;
    for (auto i = xRowIndex.begin(); i != xRowIndex.end(); ++i, ++xRowInd) {
        *i = xRowInd;
    }
    return calculateXCenters(xRowIndex);
}


ndarray::Array<float, 1, 1>
FiberTraceFunction::calculateXCenters(
    ndarray::Array<float, 1, 1> const& yIn
) const {
    #ifdef __DEBUG_XCENTERS__
    cout << "pfs::drp::stella::calculateXCenters: function.ctrl.order = " << ctrl.order << endl;
    #endif
    float const rangeMin = yCenter + yLow;
    float const rangeMax = yCenter + yHigh;
    #ifdef __DEBUG_XCENTERS__
    cout << "pfs::drp::stella::calculateXCenters: range = " << rangeMin << "," << rangeMax << endl;
    #endif
    #ifdef __DEBUG_XCENTERS__
      cout << "pfs::drp::stella::math::calculateXCenters: Calculating Polynomial" << endl;
      cout << "pfs::drp::stella::calculateXCenters: Function = Polynomial" << endl;
      cout << "pfs::drp::stella::calculateXCenters: function.coefficients = " << coefficients << endl;
    #endif
    return math::calculatePolynomial(yIn, coefficients, rangeMin, rangeMax);
}


}}} // namespace pfs::drp::stella
