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


/// Convert a std::vector to an ndarray::Array
///
/// This is useful for conversions in APIs, as the data is reused, with no
/// copying. The Array's data will disappear when the vector goes out of scope,
/// so be sure to use ndarray::copy as needed.
template <typename T>
ndarray::Array<T, 1, 1> vectorToArray(std::vector<T> vector) {
    return ndarray::Array<T, 1, 1>(ndarray::external(vector.data(), ndarray::makeVector(vector.size())));
}


/// Convert an ndarray::Array to a std::vector
///
/// Some things in LSST still use std::vector, and there's no simple way to
/// construct a std::vector from an ndarray::Array.
template <typename T>
std::vector<T> arrayToVector(ndarray::Array<T, 1, 1> const& array) {
    std::vector<T> vector(array.size());
    std::copy(array.begin(), array.end(), vector.begin());
    return vector;
}


}}}}  // namespace pfs::drp::stella::utils

#endif
