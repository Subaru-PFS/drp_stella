#ifndef PFS_DRP_STELLA_PROFILE_H
#define PFS_DRP_STELLA_PROFILE_H

#include <utility>

#include "ndarray_fwd.h"

namespace pfs {
namespace drp {
namespace stella {

/// Calculate the profile for a swath
///
/// When measuring a fiber profile, we have a 2D numpy MaskedArray of values
/// (measurements of the profile for each row) which we want to collapse to a
/// 1D average profile.
///
/// @param values : Measurement values
/// @param mask : Measurement mask (true = bad value)
/// @param rejIter : Number of rejection iterations
/// @param, rejThresh : Rejection threshold (sigma)
/// @return collapsed values and mask
std::pair<ndarray::Array<double, 1, 1>, ndarray::Array<bool, 1, 1>>
calculateSwathProfile(
    ndarray::Array<double, 2, 1> const& values,
    ndarray::Array<bool, 2, 1> const& mask,
    int rejIter=1,
    float rejThresh=3.0
);


}}} // namespace pfs::drp::stella

#endif
