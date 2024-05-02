#ifndef PFS_DRP_STELLA_maskNearby_H
#define PFS_DRP_STELLA_maskNearby_H

#include "ndarray_fwd.h"

namespace pfs {
namespace drp {
namespace stella {


/// Mask values that are within a certain distance of any other value.
///
/// @param values : Array of values.
/// @param distance : Distance within which to mask values.
/// @return Mask of values that are within distance of any other value.
ndarray::Array<bool, 1, 1> maskNearby(
    ndarray::Array<double, 1, 1> const& values,
    double distance
);


}}}  // namespace pfs::drp::stella

#endif  // include guard
