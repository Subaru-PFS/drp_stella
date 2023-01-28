#ifndef PFS_DRP_STELLA_MATH_QUARTILES_H
#define PFS_DRP_STELLA_MATH_QUARTILES_H

#include <climits>
#include <utility>
#include <algorithm>

#include "ndarray_fwd.h"

#include "lsst/pex/exceptions.h"
#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {

double const NaN = std::numeric_limits<double>::quiet_NaN();


/// Comparison for masked arrays
///
/// This functor can be passed to algorithmic codes like std::sort or
/// std::nth_element in combination with an index array in order to sort masked
/// arrays without having to do an additional copy.
template<bool maskMeansInclude, typename T, int C1, int C2>
struct MaskedArrayIndexCompare {
    /// Constructor
    ///
    /// @param _values : Numbers to be compared
    /// @param _masks : Whether the numbers are masked
    MaskedArrayIndexCompare(
        ndarray::ArrayRef<T, 1, C1> const& _values,
        ndarray::ArrayRef<bool, 1, C2> const& _masks
    ) : values(_values), masks(_masks) {
        utils::checkSize(masks.getShape(), values.getShape(), "masks");
    }

    /// Return whether the value at index is masked
    bool isMasked(std::size_t index) const {
        return ((maskMeansInclude && !masks[index]) || (!maskMeansInclude && masks[index]));
    }

    /// Perform comparison
    ///
    /// Unmasked values compare as less than masked values (function returns true).
    /// This, in effect, puts masked values at the end of the array.
    ///
    /// @param left : Index of the first element being compared
    /// @param right : Index of the second element being compared
    /// @return Whether the left value compares less than the right value.
    bool operator()(std::size_t const left, std::size_t const right) const {
        if (isMasked(right)) {
            if (isMasked(left)) {
                // Make the comparison stable by comparing indices
                return left < right;
            }
            return true;
        }
        if (isMasked(left)) {
            return false;
        }
        return values[left] < values[right];
    }

    std::size_t size() const { return values.size(); }

    ndarray::ArrayRef<T, 1, C1> const values;  ///< Numbers to be compared
    ndarray::ArrayRef<bool, 1, C2> const masks;  ///< Mask
};


template<bool maskMeansInclude, typename T, int C1, int C2>
struct MaskedArrayPartitioner {
    using Indices = ndarray::Array<std::size_t, 1, 1>;
    using Iterator = typename Indices::iterator;
    using Data = T;

    /// Constructor
    ///
    /// @param _values : Numbers to be compared
    /// @param _masks : Whether the numbers are to be considered as masked (true means bad value)
    MaskedArrayPartitioner(
        ndarray::ArrayRef<T, 1, C1> const& _values,
        ndarray::ArrayRef<bool, 1, C2> const& _masks
    ) : compare(_values, _masks),
        indices(utils::arange(0UL, compare.size())),
        unmasked(std::count_if(_masks.begin(), _masks.end(), [_masks](bool mm) {
            return (maskMeansInclude && mm) || (!maskMeansInclude && !mm); }
        ))
        {}

    /// Return size of array
    ///
    /// This is the effective size (number of unmasked values), not the actual
    /// size.
    std::size_t size() const { return unmasked; }

    //@{
    /// Return value at partition
    ///
    /// @param start : Beginning of partition
    /// @param target : Partition target
    /// @param stop : End of partition
    // Data operator()(std::size_t start, std::size_t target, std::size_t stop) {
    //     Iterator begin = indices.begin();
    //     return operator()(begin + start, begin + target, begin + stop);
    // }
    Data operator()(Iterator start, Iterator target, Iterator stop) {
        std::nth_element(start, target, stop, compare);
        return compare.values[*target];
    }
    //@}

    Iterator begin() { return indices.begin(); }
    Iterator end() { return indices.end(); }

    MaskedArrayIndexCompare<maskMeansInclude, T, C1, C2> compare;  ///< Compare function
    ndarray::Array<std::size_t, 1, 1> indices;  ///< Indices for arrays
    std::size_t unmasked;  ///< Number of unmasked entries
};


/// Comparison for array
///
/// This functor can be passed to algorithmic codes like std::sort or
/// std::nth_element in combination with an index array in order to sort an
/// array without having to do an additional copy.
template<typename T, int C>
struct ArrayIndexCompare {
    /// Constructor
    ///
    /// @param _values : Numbers to be compared
    ArrayIndexCompare(
        ndarray::ArrayRef<T, 1, C> const& _values
    ) : values(_values) {}

    /// Perform comparison
    ///
    /// @param left : Index of the first element being compared
    /// @param right : Index of the second element being compared
    /// @return Whether the left value compares less than the right value.
    bool operator()(std::size_t const left, std::size_t const right) const {
        return values[left] < values[right];
    }

    std::size_t size() const { return values.size(); }

    ndarray::ArrayRef<T, 1, C> const values;  ///< Numbers to be compared
};


template<typename T, int C>
struct ArrayPartitioner {
    using Indices = ndarray::Array<std::size_t, 1, 1>;
    using Iterator = typename Indices::iterator;
    using Data = T;

    /// Constructor
    ///
    /// @param _values : Numbers to be compared
    ArrayPartitioner(
        ndarray::ArrayRef<T, 1, C> const& _values
    ) : compare(_values),
        indices(utils::arange(0UL, compare.size()))
        {}

    /// Return size of array
    std::size_t size() const { return compare.size(); }

    //@{
    /// Return value at partition
    ///
    /// @param start : Beginning of partition
    /// @param target : Partition target
    /// @param stop : End of partition
    // Data operator()(std::size_t start, std::size_t target, std::size_t stop) {
    //     Iterator begin = indices.begin();
    //     return operator()(begin + start, begin + target, begin + stop);
    // }
    Data operator()(Iterator start, Iterator target, Iterator stop) {
        std::nth_element(start, target, stop, compare);
        return compare.values[*target];
    }
    //@}

    Iterator begin() { return indices.begin(); }
    Iterator end() { return indices.end(); }

    ArrayIndexCompare<T, C> compare;  ///< Compare function
    ndarray::Array<std::size_t, 1, 1> indices;  ///< Indices for arrays
};


/// Calculate median of an array
template <typename Partitioner>
typename Partitioner::Data calculateMedian(
    Partitioner & partitioner
) {
    std::size_t const num = partitioner.size();
    if (num == 0) {
        return NaN;
    }
    if (num == 1) {
        return partitioner(partitioner.begin(), partitioner.begin(), partitioner.end());
    }

    double const idx50 = 0.50 * (num - 1);

    std::size_t const q50a = static_cast<std::size_t>(idx50);
    std::size_t const q50b = q50a + 1;

    // get the 50th percentile
    auto mid50a = partitioner.begin() + q50a;
    auto mid50b = partitioner.begin() + q50b;
    double const val50a = partitioner(partitioner.begin(), mid50a, partitioner.end());
    double const val50b = partitioner(mid50a, mid50b, partitioner.end());

    // interpolate linearly between the adjacent values
    double const w50a = (static_cast<double>(q50b) - idx50);
    double const w50b = (idx50 - static_cast<double>(q50a));
    double const median = w50a * val50a + w50b * val50b;

    return typename Partitioner::Data(median);
}


//@{
/// Calculate median of a masked array
///
/// @param values : Numbers for which to calculate median
/// @param masks : Whether the numbers are to be considered as masked (true means bad value)
/// @return Median
template <bool maskMeansInclude, typename T, int C1, int C2>
T calculateMedian(
    ndarray::ArrayRef<T, 1, C1> const values,
    ndarray::ArrayRef<bool, 1, C2> const masks
) {
    utils::checkSize(masks.getShape(), values.getShape(), "masks");
    MaskedArrayPartitioner<maskMeansInclude, T, C1, C2> partitioner(values, masks);
    return calculateMedian(partitioner);
}
template <typename T, int C1, int C2>
T calculateMedian(
    ndarray::ArrayRef<T, 1, C1> const values,
    ndarray::ArrayRef<bool, 1, C2> const masks
) {
    return calculateMedian<false>(values, masks);
}
template <bool maskMeansInclude, typename T, int C1, int C2>
T calculateMedian(
    ndarray::Array<T, 1, C1> const& values,
    ndarray::Array<bool, 1, C2> const& masks
) {
    return calculateMedian<maskMeansInclude>(values.deep(), masks.deep());
}
template <typename T, int C1, int C2>
T calculateMedian(
    ndarray::Array<T, 1, C1> const& values,
    ndarray::Array<bool, 1, C2> const& masks
) {
    return calculateMedian<false>(values.deep(), masks.deep());
}
template <typename T, int C>
T calculateMedian(
    ndarray::ArrayRef<T, 1, C> const values
) {
    ArrayPartitioner<T, C> partitioner{values};
    return calculateMedian(partitioner);
}
template <typename T, int C>
T calculateMedian(
    ndarray::Array<T, 1, C> const& values
) {
    return calculateMedian(values.deep());
}
//@}


/// Calculate quartiles of an array
template <typename Partitioner>
std::tuple<typename Partitioner::Data, typename Partitioner::Data, typename Partitioner::Data>
calculateQuartiles(
    Partitioner & partitioner
) {
    using Data = typename Partitioner::Data;
    std::size_t const num = partitioner.size();
    if (num == 0) {
        return std::make_tuple(NaN, NaN, NaN);
    }
    if (num == 1) {
        Data const value = partitioner(partitioner.begin(), partitioner.begin(), partitioner.end());
        return std::make_tuple(value, value, value);
    }

    double const idx50 = 0.50 * (num - 1);
    double const idx25 = 0.25 * (num - 1);
    double const idx75 = 0.75 * (num - 1);

    // For efficiency:
    // - partition at 50th, then partition the two half further to get 25th and 75th
    // - to get the adjacent points (for interpolation), partition between 25/50, 50/75, 75/end
    //   these should be much smaller partitions

    std::size_t const q50a = static_cast<std::size_t>(idx50);
    std::size_t const q50b = q50a + 1;
    std::size_t const q25a = static_cast<std::size_t>(idx25);
    std::size_t const q25b = q25a + 1;
    std::size_t const q75a = static_cast<std::size_t>(idx75);
    std::size_t const q75b = q75a + 1;

    auto mid50a = partitioner.begin() + q50a;
    auto mid50b = partitioner.begin() + q50b;
    auto mid25a = partitioner.begin() + q25a;
    auto mid25b = partitioner.begin() + q25b;
    auto mid75a = partitioner.begin() + q75a;
    auto mid75b = partitioner.begin() + q75b;

    // get the 50th percentile, then get the 25th and 75th on the smaller partitions
    double const val50a = partitioner(partitioner.begin(), mid50a, partitioner.end());
    double const val75a = partitioner(mid50a, mid75a, partitioner.end());
    double const val25a = partitioner(partitioner.begin(), mid25a, mid50a);

    // and the adjacent points for each ... use the smallest segments available.
    double const val50b = partitioner(mid50a, mid50b, mid75a);
    double const val25b = partitioner(mid25a, mid25b, mid50a);
    double const val75b = partitioner(mid75a, mid75b, partitioner.end());

    // interpolate linearly between the adjacent values
    double const w50a = (static_cast<double>(q50b) - idx50);
    double const w50b = (idx50 - static_cast<double>(q50a));
    double const median = w50a * val50a + w50b * val50b;

    double const w25a = (static_cast<double>(q25b) - idx25);
    double const w25b = (idx25 - static_cast<double>(q25a));
    double const q1 = w25a * val25a + w25b * val25b;

    double const w75a = (static_cast<double>(q75b) - idx75);
    double const w75b = (idx75 - static_cast<double>(q75a));
    double const q3 = w75a * val75a + w75b * val75b;

    return std::make_tuple(Data(q1), Data(median), Data(q3));
}


//@{
/// Calculate quartiles of a masked array
///
/// @param values : Numbers for which to calculate quartiles
/// @param masks : Whether the numbers are to be considered as masked (true means bad value)
/// @return lower quartile, median, upper quartile
template <bool maskMeansInclude, typename T, int C1, int C2>
std::tuple<T, T, T>
calculateQuartiles(
    ndarray::ArrayRef<T, 1, C1> const values,
    ndarray::ArrayRef<bool, 1, C2> const masks
) {
    utils::checkSize(masks.getShape(), values.getShape(), "masks");
    MaskedArrayPartitioner<maskMeansInclude, T, C1, C2> partitioner(values, masks);
    return calculateQuartiles(partitioner);
}
template <typename T, int C1, int C2>
std::tuple<T, T, T>
calculateQuartiles(
    ndarray::ArrayRef<T, 1, C1> const values,
    ndarray::ArrayRef<bool, 1, C2> const masks
) {
    return calculateQuartiles<false>(values, masks);
}
template <bool maskMeansInclude, typename T, int C1, int C2>
std::tuple<T, T, T>
calculateQuartiles(
    ndarray::Array<T, 1, C1> const& values,
    ndarray::Array<bool, 1, C2> const& masks
) {
    return calculateQuartiles<maskMeansInclude>(values.deep(), masks.deep());
}
template <typename T, int C1, int C2>
std::tuple<T, T, T>
calculateQuartiles(
    ndarray::Array<T, 1, C1> const& values,
    ndarray::Array<bool, 1, C2> const& masks
) {
    return calculateQuartiles<false>(values.deep(), masks.deep());
}
template <typename T, int C>
std::tuple<T, T, T>
calculateQuartiles(
    ndarray::ArrayRef<T, 1, C> const values
) {
    ArrayPartitioner<T, C> partitioner{values};
    return calculateQuartiles(partitioner);
}
template <typename T, int C>
std::tuple<T, T, T>
calculateQuartiles(
    ndarray::Array<T, 1, C> const& values
) {
    return calculateQuartiles(values.deep());
}
//@}

}}}}  // namespace pfs::drp::stella::math

#endif
