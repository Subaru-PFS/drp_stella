#ifndef PFS_DRP_STELLA_LINECONSISTENCY_H
#define PFS_DRP_STELLA_LINECONSISTENCY_H

#include "ndarray_fwd.h"

namespace pfs {
namespace drp {
namespace stella {


ndarray::Array<bool, 1, 1> checkLineConsistency(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<float, 1, 1> const& xx,
    ndarray::Array<float, 1, 1> const& yy,
    ndarray::Array<float, 1, 1> const& xErr,
    ndarray::Array<float, 1, 1> const& yErr,
    float threshold=3.0
);


ndarray::Array<bool, 1, 1> checkTraceConsistency(
    ndarray::Array<int, 1, 1> const& fiberId,
    ndarray::Array<float, 1, 1> const& xx,
    ndarray::Array<float, 1, 1> const& yy,
    ndarray::Array<float, 1, 1> const& xErr,
    float threshold=3.0
);


}}}  // namespace pfs::drp::stella

#endif  // include guard
