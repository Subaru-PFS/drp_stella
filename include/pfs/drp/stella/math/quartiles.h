#ifndef PFS_DRP_STELLA_MATH_QUARTILES_H
#define PFS_DRP_STELLA_MATH_QUARTILES_H

#include <climits>
#include <utility>
#include <algorithm>

#include "ndarray_fwd.h"

#include "lsst/pex/exceptions.h"
#include "pfs/drp/stella/utils/checkSize.h"

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
template<typename T, int C>
struct MaskedArrayIndexCompare {
    /// Constructor
    ///
    /// @param _values : Numbers to be compared
    /// @param _masks : Whether the numbers are to be considered as masked (true means bad value)
    MaskedArrayIndexCompare(
        ndarray::ArrayRef<T, 1, C> const& _values,
        ndarray::ArrayRef<bool, 1, C> const& _masks
    ) : values(_values), masks(_masks) {}

    /// Perform comparison
    ///
    /// Unmasked values compare as less than masked values (function returns true).
    ///
    /// @param left : Index of the first element being compared
    /// @param right : Index of the second element being compared
    /// @return Whether the left value compares less than the right value.
    bool operator()(std::size_t const left, std::size_t const right) const {
        if (masks[right]) {
            if (masks[left]) {
                // Make the comparison stable by comparing indices
                return left < right;
            }
            return true;
        }
        if (masks[left]) {
            return false;
        }
        return values[left] < values[right];
    }

    ndarray::ArrayRef<T, 1, C> const values;  ///< Numbers to be compared
    ndarray::ArrayRef<bool, 1, C> const masks;  ///< Mask
};


/// Calculate quartiles of a masked array
///
/// @param values : Numbers for which to calculate quartiles
/// @param masks : Whether the numbers are to be considered as masked (true means bad value)
/// @return lower quartile, median, upper quartile
template <typename T, int C>
std::tuple<T, T, T>
calculateQuartiles(
    ndarray::ArrayRef<T, 1, C> const values,
    ndarray::ArrayRef<bool, 1, C> const masks
) {
    utils::checkSize(masks.getShape(), values.getShape(), "masks");
    std::size_t const total = values.getNumElements();
    std::size_t const numMasked = std::count_if(masks.begin(), masks.end(), [](bool mm) { return mm; });
    std::size_t const num = total - numMasked;
    if (num == 0) {
        return std::make_tuple(NaN, NaN, NaN);
    }
    if (num == 1) {
        auto unmasked = std::find_if(masks.begin(), masks.end(), [](bool mm) { return mm; });
        std::ptrdiff_t index = unmasked - masks.begin();
        return std::make_tuple(values[index], values[index], values[index]);
    }

    double const idx50 = 0.50 * (num - 1);
    double const idx25 = 0.25 * (num - 1);
    double const idx75 = 0.75 * (num - 1);

    ndarray::Array<std::size_t, 1, 1> indices = ndarray::allocate(total);
    for (std::size_t ii = 0; ii < total; ++ii) {
        indices[ii] = ii;
    }
    MaskedArrayIndexCompare<T, C> compare{values, masks};

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

    auto mid50a = indices.begin() + q50a;
    auto mid50b = indices.begin() + q50b;
    auto mid25a = indices.begin() + q25a;
    auto mid25b = indices.begin() + q25b;
    auto mid75a = indices.begin() + q75a;
    auto mid75b = indices.begin() + q75b;

    // get the 50th percentile, then get the 25th and 75th on the smaller partitions
    std::nth_element(indices.begin(), mid50a, indices.end(), compare);
    std::nth_element(mid50a, mid75a, indices.end(), compare);
    std::nth_element(indices.begin(), mid25a, mid50a, compare);

    // and the adjacent points for each ... use the smallest segments available.
    std::nth_element(mid50a, mid50b, mid75a, compare);
    std::nth_element(mid25a, mid25b, mid50a, compare);
    std::nth_element(mid75a, mid75b, indices.end(), compare);

    // interpolate linearly between the adjacent values
    double const val50a = static_cast<double>(values[*mid50a]);
    double const val50b = static_cast<double>(values[*mid50b]);
    double const w50a = (static_cast<double>(q50b) - idx50);
    double const w50b = (idx50 - static_cast<double>(q50a));
    double const median = w50a * val50a + w50b * val50b;

    double const val25a = static_cast<double>(values[*mid25a]);
    double const val25b = static_cast<double>(values[*mid25b]);
    double const w25a = (static_cast<double>(q25b) - idx25);
    double const w25b = (idx25 - static_cast<double>(q25a));
    double const q1 = w25a * val25a + w25b * val25b;

    double const val75a = static_cast<double>(values[*mid75a]);
    double const val75b = static_cast<double>(values[*mid75b]);
    double const w75a = (static_cast<double>(q75b) - idx75);
    double const w75b = (idx75 - static_cast<double>(q75a));
    double const q3 = w75a * val75a + w75b * val75b;

    return std::make_tuple(T(q1), T(median), T(q3));
}


}}}}  // namespace pfs::drp::stella::math

#endif
