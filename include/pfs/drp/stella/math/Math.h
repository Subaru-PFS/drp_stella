#ifndef __PFS_DRP_STELLA_MATH_H__
#define __PFS_DRP_STELLA_MATH_H__

#include <numeric>      // std::accumulate
#include <vector>

#include "ndarray.h"

#include "lsst/afw/geom/Point.h"

#include "pfs/drp/stella/utils/Utils.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math{

//@{
/**
 * Rounds x downward, returning the largest integral value that is not greater than x.
 *
 * @param rhs: value to be rounded down
 * @return rounded down value of rhs, type of outType
 */
template <typename T, typename U>
ndarray::Array<T, 1, 1> floor(ndarray::ArrayRef<U, 1, 1> const& rhs) {
    ndarray::Array<T, 1, 1> out = ndarray::allocate(rhs.getShape());
    std::transform(rhs.begin(), rhs.end(), out.begin(),
                   [](U val) { return T(std::llround(std::floor(val))); });
    return out;
}

template <typename T, typename U>
ndarray::Array<T, 1, 1> floor(ndarray::Array<U, 1, 1> const& rhs) {
    return floor<T>(rhs.deep());
}
//@}

/**
 * @brief: Return vector of indices where lowRangen <= array < highRange
 *
 * @param[in] array       1D array to search for values in given range
 * @param[in] lowRange    lower range
 * @param[in] lowRange    upper range
 */
template <typename T>
std::vector<std::size_t> getIndicesInValueRange(
    ndarray::Array<T const, 1, 1> const& array,
    T lowRange,
    T highRange
);

/**
 * @brief: Return vector of indices where lowRange <= array < highRange
 *
 * @param[in] array       2D array to search for values in given range
 * @param[in] lowRange    lower range
 * @param[in] lowRange    upper range
 */
template <typename T>
std::vector<lsst::afw::geom::Point2I> getIndicesInValueRange(
    ndarray::Array<T, 2, 1> const& array,
    T lowRange,
    T highRange
);

/**
 * @brief: Returns array to copies of specified elements of array
 *
 * @param[in] array   1D array to create subarray from
 * @param[in] indices indices of array which shall be copied to output subarray
 */
template <typename T>
ndarray::Array<T, 1, 1> getSubArray(
    ndarray::Array<T, 1, 1> const& array,
    std::vector<std::size_t> const& indices
);

/**
 * @brief: Returns array to copies of specified elements of array
 *
 * @param[in] array   2D array to create subarray from
 * @param[in] indices indices of array which shall be copied to output subarray
 */
template <typename T>
ndarray::Array<T, 1, 1> getSubArray(
    ndarray::Array<T, 2, 1 > const& array,
    std::vector<lsst::afw::geom::Point2I> const& indices
);

/**
 * Calculate moments from array
 *
 * The moments are:
 * - 0: Mean
 * - 1: Variance
 * - 2: Skew
 * - 3: Kurtosis
 *
 * @param[in] array  Array of values from which to compute statistics
 * @param[in] maxMoment  Maximum moment to calculate
 * @return array of moments
 */
template <typename T>
ndarray::Array<T, 1, 1> moment(
    ndarray::Array<T const, 1, 1> const& array,
    int maxMoment
);


/**
 * Returns an integer array of the same size like <data>,
 * containing the indixes of <data> in sorted order.
 *
 * @param[in] data       vector to sort
 **/
template <typename T>
std::vector<std::size_t> sortIndices(std::vector<T> const& data);

/**
 * @brief convert given numbers in given range to a number in range [-1,1]
 *
 * @param numbers: numbers to be converted
 * @param range: range numbers are from
 */
template <typename T>
ndarray::Array<T, 1, 1> convertRangeToUnity(
    ndarray::Array<T, 1, 1> const& numbers,
    T low,
    T high
) {
    ndarray::Array<T, 1, 1> out = ndarray::allocate(numbers.getNumElements());
    out.deep() = (numbers - low)*2./(high - low) - 1.;
    return out;
}

/**
 * returns first index of integer input vector where value is greater than or
 * equal to minValue, starting at index fromIndex
 *
 * returns -1 if values are always smaller than minValue
 *
 * @param[in] array     1D array to search for number >= minValue
 * @param[in] minValue  minimum value to search for
 * @param[in] fromIndex  index position to start search
 **/
template <typename T>
std::ptrdiff_t firstIndexWithValueGEFrom(
    ndarray::Array<T, 1, 1> const& array,
    T minValue,
    std::size_t fromIndex
);

/**
 * returns last index of integer input vector where value is equal to zero, starting at index startPos
 *
 * returns -1 if values are always greater than 0 before startPos
 *
 * @param[in] vec_In      1D array to search
 * @param[in] startPos_In  index position to start search
 **/
template <typename T>
std::ptrdiff_t lastIndexWithZeroValueBefore(
    ndarray::Array<T, 1, 1> const& array,
    std::ptrdiff_t startPos
);

/**
 * returns first index of integer input vector where value is equal to zero, starting at index startPos
 *
 * returns -1 if values are always greater than 0 past startPos
 *
 * @param[in] array     1D array to search
 * @param[in] startPos  index position to start search
 **/
template <typename T>
std::ptrdiff_t firstIndexWithZeroValueFrom(
    ndarray::Array<T, 1, 1> const& array,
    std::ptrdiff_t startPos
);


/**
 * Convert a std::vector to an ndarray::Array
 *
 * This is useful for conversions in APIs, as the data is reused, with no
 * copying. It is less useful for other purposes, as the Array's data will
 * disappear when the vector goes out of scope.
 *
 * @param[in] vector  Vector of numeric values
 * @returns Array of values
 */
template <typename T>
ndarray::Array<T, 1, 1> vectorToArray(std::vector<T> vector) {
    return ndarray::Array<T, 1, 1>(ndarray::external(vector.data(), ndarray::makeVector(vector.size())));
}

}}}} // namespace pfs::drp::stella::math

template <typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& obj);

#endif
