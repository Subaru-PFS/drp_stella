#ifndef PFS_DRP_STELLA_UTILS_MATH_H
#define PFS_DRP_STELLA_UTILS_MATH_H

#include <ndarray_fwd.h>


namespace pfs {
namespace drp {
namespace stella {
namespace utils {


/// Convert an ndarray::Array from one type to another
template <typename T, typename U, int N, int C>
ndarray::Array<T, N, C> convertArray(ndarray::Array<U, N, C> const& array) {
    ndarray::Array<T, N, C> out = ndarray::allocate(array.getShape());
    std::copy(array.begin(), array.end(), out.begin());
    return out;
}


}}}}  // namespace pfs::drp::stella::utils

#endif
