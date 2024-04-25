#include "ndarray.h"

#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/maskNearby.h"

namespace pfs {
namespace drp {
namespace stella {


ndarray::Array<bool, 1, 1> maskNearby(
    ndarray::Array<double, 1, 1> const& values,
    double distance
) {
    std::size_t const num = values.size();
    ndarray::Array<bool, 1, 1> mask = ndarray::allocate(num);
    if (num <= 1) {
        mask.deep() = false;
        return mask;
    }

    // Ensure the inputs are sorted
    bool isSorted = true;  // innocent until proven guilty
    for (std::size_t ii = 1; ii < values.size(); ++ii) {
        if (values[ii] < values[ii - 1]) {
            isSorted = false;
            break;
        }
    }
    ndarray::Array<std::size_t, 1, 1> sortIndices;  // indices that sort "values"
    ndarray::Array<double, 1, 1> valuesSorted;  // "values" sorted
    ndarray::Array<bool, 1, 1> maskSorted;  // mask for valuesSorted
    if (isSorted) {
        valuesSorted = values;
        maskSorted = mask;
    } else {
        sortIndices = utils::argsort(values);
        valuesSorted = ndarray::allocate(num);
        for (std::size_t ii = 0; ii < values.size(); ++ii) {
            valuesSorted[ii] = values[sortIndices[ii]];
        }
        maskSorted = ndarray::allocate(num);
    }

    // Do the actual identification of nearby values
    maskSorted.deep() = false;
    std::size_t low = 0;  // Index of low bound of nearby values; inclusive
    std::size_t high = 1;  // Index of high bound of nearby values; exclusive
    for (std::size_t ii = 0; ii < num; ++ii) {
        double const vv = valuesSorted[ii];
        double const lowBound = vv - distance;
        double const highBound = vv + distance;
        while (low < ii && valuesSorted[low] <= lowBound) ++low;
        while (high < num && valuesSorted[high] < highBound) ++high;
        for (std::size_t jj = low; jj < ii; ++jj) {
            if (valuesSorted[jj] != vv) {
                maskSorted[jj] = true;
            }
        }
        for (std::size_t jj = ii + 1; jj < high; ++jj) {
            if (valuesSorted[jj] != vv) {
                maskSorted[jj] = true;
            }
        }
    }

    // Return to the original order
    if (!isSorted) {
        for (std::size_t ii = 0; ii < values.size(); ++ii) {
            mask[sortIndices[ii]] = maskSorted[ii];
        }
    } else {
        mask = maskSorted;
    }

    return mask;
}


}}}  // namespace pfs::drp::stella
