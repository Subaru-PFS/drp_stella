#include <unordered_map>

#include "ndarray.h"
#include "lsst/cpputils/hashCombine.h"

#include "pfs/drp/stella/lineConsistency.h"
#include "pfs/drp/stella/utils/checkSize.h"

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
std::pair<T, T> averageValues(
    std::vector<std::size_t> const& indices,
    ndarray::Array<T, 1, 1> const& array,
    ndarray::Array<T, 1, 1> const& error,
    lsst::afw::math::StatisticsControl const& control
) {
    auto const MEAN_STAT = lsst::afw::math::MEANCLIP;
    auto const STDEV_STAT = lsst::afw::math::STDEVCLIP;

    std::size_t const numPoints = indices.size();
    if (numPoints <= 1) {
        return std::make_pair(array[indices[0]], error[indices[0]]);
    }
    lsst::afw::image::MaskedImage<T> image{numPoints, 1};
    auto iter = image.begin();
    for (std::size_t ii = 0; ii < numPoints; ++ii, ++iter) {
        iter.image() = array[indices[ii]];
        iter.variance() = error[indices[ii]]*error[indices[ii]];
        iter.mask() = 0;
    }

    auto const stats = lsst::afw::math::makeStatistics(image, MEAN_STAT | STDEV_STAT, control);
    return std::make_pair(stats.getValue(MEAN_STAT), stats.getValue(STDEV_STAT));
}


}  // anonymous namespace


ConsistencyResult checkLineConsistency(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<float, 1, 1> const& xx,
    ndarray::Array<float, 1, 1> const& yy,
    ndarray::Array<float, 1, 1> const& xErr,
    ndarray::Array<float, 1, 1> const& yErr,
    ndarray::Array<float, 1, 1> const& flux,
    ndarray::Array<float, 1, 1> const& fluxErr,
    lsst::afw::math::StatisticsControl const& control
) {
    std::size_t const length = fiberId.size();
    utils::checkSize(wavelength.size(), length, "wavelength");
    utils::checkSize(xx.size(), length, "x");
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");
    utils::checkSize(flux.size(), length, "flux");
    utils::checkSize(fluxErr.size(), length, "fluxErr");

    ndarray::Array<bool, 1, 1> accept = ndarray::allocate(length);
    accept.deep() = true;

    std::unordered_map<std::pair<int, double>, std::vector<std::size_t>, PairHash> points;
    for (std::size_t ii = 0; ii < length; ++ii) {
        points[std::make_pair(fiberId[ii], wavelength[ii])].push_back(ii);
    }
    ConsistencyResult result{points.size()};
    std::size_t ii = 0;
    for (auto const& pp : points) {
        auto const xAvg = averageValues(pp.second, xx, xErr, control);
        auto const yAvg = averageValues(pp.second, yy, yErr, control);
        auto const fluxAvg = averageValues(pp.second, flux, fluxErr, control);
        result.fiberId[ii] = pp.first.first;
        result.wavelength[ii] = pp.first.second;
        result.x[ii] = xAvg.first;
        result.y[ii] = yAvg.first;
        result.xErr[ii] = xAvg.second;
        result.yErr[ii] = yAvg.second;
        result.flux[ii] = fluxAvg.first;
        result.fluxErr[ii] = fluxAvg.second;
        ++ii;
    }

    return result;
}


ConsistencyResult checkTraceConsistency(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<float, 1, 1> const& xx,
    ndarray::Array<float, 1, 1> const& yy,
    ndarray::Array<float, 1, 1> const& xErr,
    ndarray::Array<float, 1, 1> const& flux,
    ndarray::Array<float, 1, 1> const& fluxErr,
    lsst::afw::math::StatisticsControl const& control
) {
    std::size_t const length = fiberId.size();
    utils::checkSize(wavelength.size(), length, "wavelength");
    utils::checkSize(xx.size(), length, "x");
    utils::checkSize(yy.size(), length, "y");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(flux.size(), length, "flux");
    utils::checkSize(fluxErr.size(), length, "fluxErr");

    std::unordered_map<std::pair<int, int>, std::vector<std::size_t>, PairHash> points;
    for (std::size_t ii = 0; ii < length; ++ii) {
        points[std::make_pair(fiberId[ii], static_cast<int>(std::round(yy[ii])))].push_back(ii);
    }

    ConsistencyResult result{points.size()};
    std::size_t ii = 0;
    for (auto const& pp : points) {
        auto const xAvg = averageValues(pp.second, xx, xErr, control);
        auto const fluxAvg = averageValues(pp.second, flux, fluxErr, control);

        result.fiberId[ii] = pp.first.first;
        result.wavelength[ii] = wavelength[pp.second[0]];
        result.y[ii] = pp.first.second;
        result.x[ii] = xAvg.first;
        result.xErr[ii] = xAvg.second;
        result.yErr[ii] = std::numeric_limits<float>::quiet_NaN();
        result.flux[ii] = fluxAvg.first;
        result.fluxErr[ii] = fluxAvg.second;
        ++ii;
    }

    return result;
}


}}}  // namespace pfs::drp::stella
