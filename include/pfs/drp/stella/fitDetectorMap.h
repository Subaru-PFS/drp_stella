#ifndef PFS_DRP_STELLA_FITDETECTORMAP_H
#define PFS_DRP_STELLA_FITDETECTORMAP_H

#include "ndarray_fwd.h"

#include "pfs/drp/stella/SplinedDetectorMap.h"

namespace pfs {
namespace drp {
namespace stella {


template <typename T>
std::shared_ptr<DetectorMap> fitDetectorMap(
    int order,
    SplinedDetectorMap const& base,
    ndarray::Array<int, 1, 1> const& fiberIdLine,
    ndarray::Array<double, 1, 1> const& wavelength,
    ndarray::Array<T, 1, 1> const& xLine,
    ndarray::Array<T, 1, 1> const& yLine,
    ndarray::Array<T, 1, 1> const& xErrLine,
    ndarray::Array<T, 1, 1> const& yErrLine,
    ndarray::Array<int, 1, 1> const& fiberIdTrace,
    ndarray::Array<T, 1, 1> const& xTrace,
    ndarray::Array<T, 1, 1> const& yTrace,
    ndarray::Array<T, 1, 1> const& xErrTrace,
    ndarray::Array<double, 1, 1> const& start = ndarray::Array<double, 1, 1>()
);


}}}  // namespace pfs::drp::stella



#endif  // include guard
