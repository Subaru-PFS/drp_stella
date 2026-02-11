#include <unordered_map>

#include "ndarray.h"
#include "lsst/cpputils/hashCombine.h"

#include "pfs/drp/stella/lineConsistency.h"
#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/math/quartiles.h"

namespace pfs {
namespace drp {
namespace stella {

namespace {


struct PairHash {
    template <typename PairT>
    std::size_t operator()(PairT const& pair) const noexcept {
        std::size_t seed = 0;
        return lsst::cpputils::hashCombine(seed, pair.first, pair.second);
    }
};


template <typename T>
void checkConsistency(
    ndarray::Array<bool, 1, 1> & accept,
    std::vector<std::size_t> const& indices,
    ndarray::Array<T, 1, 1> const& array,
    ndarray::Array<T, 1, 1> const& error,
    float threshold
) {
    std::size_t const numPoints = indices.size();
    if (numPoints <= 1) {
        return;
    }
    ndarray::Array<T, 1, 1> values = ndarray::allocate(numPoints);
    for (std::size_t ii = 0; ii < numPoints; ++ii) {
        values[ii] = array[indices[ii]];
    }
    double const median = math::calculateMedian(values);
    for (std::size_t ii = 0; ii < numPoints; ++ii) {
        if (std::abs(values[ii] - median) > 3.0*error[indices[ii]]) {
            accept[indices[ii]] = false;
        }
    }
}


}  // anonymous namespace


ndarray::Array<bool, 1, 1> checkLineConsistency(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<float, 1, 1> const& xx,
    ndarray::Array<float, 1, 1> const& yy,
    ndarray::Array<float, 1, 1> const& xErr,
    ndarray::Array<float, 1, 1> const& yErr,
    float threshold
) {
    std::size_t const length = fiberId.size();
    utils::checkSize(wavelength.size(), length, "wavelength");
    utils::checkSize(xx.size(), length, "x");
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");

    ndarray::Array<bool, 1, 1> accept = ndarray::allocate(length);
    accept.deep() = true;

    std::unordered_map<std::pair<int, double>, std::vector<std::size_t>, PairHash> points;
    for (std::size_t ii = 0; ii < length; ++ii) {
        points[std::make_pair(fiberId[ii], wavelength[ii])].push_back(ii);
    }
    for (auto const& pp : points) {
        checkConsistency(accept, pp.second, xx, xErr, threshold);
        checkConsistency(accept, pp.second, yy, yErr, threshold);
    }

    return accept;
}


ndarray::Array<bool, 1, 1> checkTraceConsistency(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<float, 1, 1> const& xx,
    ndarray::Array<float, 1, 1> const& yy,
    ndarray::Array<float, 1, 1> const& xErr,
    float threshold
) {
    std::size_t const length = fiberId.size();
    utils::checkSize(xx.size(), length, "x");
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xErr.size(), length, "xErr");

    ndarray::Array<bool, 1, 1> accept = ndarray::allocate(length);
    accept.deep() = true;

    std::unordered_map<std::pair<int, int>, std::vector<std::size_t>, PairHash> points;
    for (std::size_t ii = 0; ii < length; ++ii) {
        points[std::make_pair(fiberId[ii], static_cast<int>(std::round(yy[ii])))].push_back(ii);
    }
    for (auto const& pp : points) {
        checkConsistency(accept, pp.second, xx, xErr, threshold);
    }

    return accept;
}



}}}  // namespace pfs::drp::stella
