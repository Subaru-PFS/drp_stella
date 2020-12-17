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


/// Return whether two arrays compare according to a user-provided function
///
/// The default ndarray compare operators check that the arrays share data. This
/// checks that the individual values compare equal.
template <typename T, int N1, int N2, int C1, int C2, typename BinaryFunction>
bool arraysCompare(
    ndarray::Array<T, N1, C1> const& left,
    ndarray::Array<T, N2, C2> const& right,
    BinaryFunction compare
) {
    if (left.getShape() != right.getShape()) {
        return false;
    }
    auto ll = left.begin();
    auto rr = right.begin();
    for (; ll != left.end() && rr != right.end(); ++ll, ++rr) {
        if (!compare(*ll, *rr)) {
            return false;
        }
    }
    return true;
}


/// Return whether two arrays compare equal
///
/// The default ndarray compare operators check that the arrays share data. This
/// checks that the individual values compare equal.
template <typename T, int N1, int N2, int C1, int C2>
bool arraysEqual(ndarray::Array<T, N1, C1> const& left, ndarray::Array<T, N2, C2> const& right) {
    return arraysCompare(left, right, [](T ll, T rr) { return ll == rr; });
}


}}}}  // namespace pfs::drp::stella::utils

#endif
