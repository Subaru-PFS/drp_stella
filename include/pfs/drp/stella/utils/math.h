#ifndef PFS_DRP_STELLA_UTILS_MATH_H
#define PFS_DRP_STELLA_UTILS_MATH_H

#include <ndarray_fwd.h>

#include "pfs/drp/stella/utils/checkSize.h"

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


/// Return elements selected from an array
///
/// @param array : Array from which to select
/// @param selection : Selection to apply
/// @return 1D array with selected elements
template <typename T, int N, int C1, int C2>
ndarray::Array<T, 1, 1> arraySelect(
    ndarray::Array<T, N, C1> const& array,
    ndarray::Array<bool, N, C2> const& selection
) {
    checkSize(array.size(), selection.size(), "array vs selection");
    std::size_t const num = std::count_if(selection.begin(), selection.end(), [](bool ss) { return ss; });
    ndarray::Array<T, N, C1> out = ndarray::allocate(num);
    auto arr = array.begin();
    auto sel = selection.begin();
    for (std::size_t ii = 0; arr != array.end(); ++arr, ++sel) {
        if (!*sel) continue;
        out[ii] = *arr;
        ++ii;
    }
    return out;
}


/// Return an array filled with a particular value
///
/// @param shape : shape of array
/// @param value : value with which to fill array
/// @return filled array
template <typename T, int N, int C, typename U>
ndarray::Array<T, N, C> arrayFilled(
    U const& shape,
    T value
) {
    ndarray::Array<T, N, C> out = ndarray::allocate(shape);
    out.deep() = value;
    return out;
}


}}}}  // namespace pfs::drp::stella::utils

#endif
