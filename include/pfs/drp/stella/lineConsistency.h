#ifndef PFS_DRP_STELLA_LINECONSISTENCY_H
#define PFS_DRP_STELLA_LINECONSISTENCY_H

#include "ndarray_fwd.h"
#include "lsst/afw/math/Statistics.h"

namespace pfs {
namespace drp {
namespace stella {


struct ConsistencyResult {
    ndarray::Array<int, 1, 1> fiberId;
    ndarray::Array<double, 1, 1> wavelength;
    ndarray::Array<float, 1, 1> x;
    ndarray::Array<float, 1, 1> y;
    ndarray::Array<float, 1, 1> xErr;
    ndarray::Array<float, 1, 1> yErr;
    ndarray::Array<float, 1, 1> flux;
    ndarray::Array<float, 1, 1> fluxErr;

    explicit ConsistencyResult(std::size_t length) :
        fiberId(ndarray::allocate(length)),
        wavelength(ndarray::allocate(length)),
        x(ndarray::allocate(length)),
        y(ndarray::allocate(length)),
        xErr(ndarray::allocate(length)),
        yErr(ndarray::allocate(length)),
        flux(ndarray::allocate(length)),
        fluxErr(ndarray::allocate(length))
    {}
};


ConsistencyResult checkLineConsistency(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<float, 1, 1> const& xx,
    ndarray::Array<float, 1, 1> const& yy,
    ndarray::Array<float, 1, 1> const& xErr,
    ndarray::Array<float, 1, 1> const& yErr,
    ndarray::Array<float, 1, 1> const& flux,
    ndarray::Array<float, 1, 1> const& fluxErr,
    lsst::afw::math::StatisticsControl const& control=lsst::afw::math::StatisticsControl()
);


ConsistencyResult checkTraceConsistency(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<float, 1, 1> const& xx,
    ndarray::Array<float, 1, 1> const& yy,
    ndarray::Array<float, 1, 1> const& xErr,
    ndarray::Array<float, 1, 1> const& flux,
    ndarray::Array<float, 1, 1> const& fluxErr,
    lsst::afw::math::StatisticsControl const& control=lsst::afw::math::StatisticsControl()
);


}}}  // namespace pfs::drp::stella

#endif  // include guard
