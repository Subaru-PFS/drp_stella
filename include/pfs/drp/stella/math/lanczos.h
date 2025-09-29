#ifndef PFS_DRP_STELLA_LANCZOS_H
#define PFS_DRP_STELLA_LANCZOS_H

#include "ndarray_fwd.h"

#include "lsst/afw/math/FunctionLibrary.h"

#include "pfs/drp/stella/utils/checkSize.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {


int const INTERPOLATE_DEFAULT_ORDER = 3;
double const INTERPOLATE_DEFAULT_FILL = 0.0;
float const INTERPOLATE_DEFAULT_MIN_WEIGHT = 0.3;


namespace impl {


//@{
/// Lanczos interpolation of 1D spectrum
///
/// This is a general implementation, handling flux, mask and variance, and used
/// by lanczosInterpolateFlux() and lanczosInterpolateVariance(). Template
/// parameters control which of the input and output arrays are provided and
/// used, allowing the compiler to optimize away unused branches. If an array
/// is not used according to the template parameters, any input array
/// will be ignored; using an empty array is recommended in this case.
///
/// @tparam wantValues  True if we want to compute resultValues.
/// @tparam wantMask  True if we want to compute resultMask.
/// @tparam wantVariance  True if we want to compute resultVariance.
/// @tparam haveMask  True if inputMask is provided.
/// @tparam T  The values and variance array element type.
/// @tparam U  The indices array element type.
/// @tparam C1  The resultValues array's contiguity.
/// @tparam C2  The resultMask array's contiguity.
/// @tparam C3  The resultVariance array's contiguity.
/// @tparam C4  The inputValues array's contiguity.
/// @tparam C5  The inputMask array's contiguity.
/// @tparam C6  The inputVariance array's contiguity.
/// @tparam C7  The indices array's contiguity.
/// @param[in,out] resultValues  The output values array. If wantValue is true,
///    this must be preallocated to the size of the indices array; otherwise
///    this is ignored.
/// @param[in,out] resultMask  The output mask array. If wantMask is true, this
///    must be preallocated to the size of the indices array; otherwise this is
///    ignored.
/// @param[in,out] resultVariance  The output variance array. If wantVariance is
///    true, this must be preallocated to the size of the indices array;
///    otherwise this is ignored.
/// @param[in] inputValues  The input values array. If wantValues is true, this
///    must be non-empty; otherwise this is ignored.
/// @param[in] inputMask  The input mask array. If haveMask is true, this must
///    be non-empty; otherwise this is ignored.
/// @param[in] inputVariance  The input variance array. If wantVariance is true,
///    this must be non-empty; otherwise this is ignored.
/// @param[in] indices  The indices array, with floating-point indices at which
///    to interpolate.
/// @param[in] fill  The value to use when interpolation is not possible.
/// @param[in] order  The order of the Lanczos function.
/// @param[in] minWeight  The minimum sum of weights to accept; if the sum of
///    weights is less than this value, the output will be masked.
template <
    bool wantValues,  // true means have resultValues and inputValues
    bool wantMask,  // true means have resultMask
    bool wantVariance,  // true means have resultVariance and inputVariance
    bool haveMask,  // true means have inputMask
    typename T,
    typename U,
    int C1, int C2, int C3, int C4, int C5, int C6, int C7
>
void lanczosInterpolate(
    ndarray::Array<T, 1, C1> & resultValues,
    ndarray::Array<bool, 1, C2> & resultMask,
    ndarray::Array<T, 1, C3> & resultVariance,
    ndarray::Array<T, 1, C4> const& inputValues,
    ndarray::Array<bool, 1, C5> const& inputMask,
    ndarray::Array<T, 1, C6> const& inputVariance,
    ndarray::Array<U, 1, C7> const& indices,
    T fill,
    unsigned int order,
    double minWeight=INTERPOLATE_DEFAULT_MIN_WEIGHT
) {
    static_assert(
        wantValues || wantMask || wantVariance,
        "At least one of wantValue, wantMask, or wantVariance must be provided"
    );

    std::size_t const numOut = indices.size();
    if (numOut == 0) {
        return;  // Nothing to do
    }
    std::size_t numIn = 0;
    if (wantValues) {
        numIn = inputValues.size();
        utils::checkSize(resultValues.size(), numOut, "resultValues");
    }
    if (haveMask) {
        if (numIn == 0) {
            numIn = inputMask.size();
        } else {
            utils::checkSize(inputMask.size(), numIn, "inputMask");
        }
    }
    if (wantMask) {
        utils::checkSize(resultMask.size(), numOut, "resultMask");
    }
    if (wantVariance) {
        if (numIn == 0) {
            numIn = inputVariance.size();
        } else {
            utils::checkSize(inputVariance.size(), numIn, "inputVariance");
        }
        utils::checkSize(resultVariance.size(), numOut, "resultVariance");
    }
    if (numIn == 0) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError, "A non-empty input array must be provided"
        );
    }

    lsst::afw::math::LanczosFunction1<double> const lanczos{order};
    for (std::size_t ii = 0; ii != numOut; ++ii) {
        T const target = indices[ii];
        if (target < 0 || target > numIn - 1) {
            // Outside the range of the input data
            if (wantValues) {
                resultValues[ii] = fill;
            }
            if (wantMask) {
                resultMask[ii] = true;
            }
            if (wantVariance) {
                resultVariance[ii] = fill;
            }
            continue;
        }
        std::ptrdiff_t const index = std::ceil(target);

        // |-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|
        //              ^ target = 2.75
        //               ^ index = 3
        // Want the center of the Lanczos window to be between index-1 and index
        // frac = 0.75
        // For order=3:
        // start = 0
        // end = 5
        // dx = -2.75, -1.75, -0.75, 0.25, 1.25, 2.25

        // Interpolate using a window of size 2*order centered on target
        double value = 0.0;
        double variance = 0.0;
        double sumWeights = 0.0;
        std::ptrdiff_t const start = std::max(std::ptrdiff_t(0), index - order);  // inclusive
        std::ptrdiff_t const end = std::min(std::ptrdiff_t(numIn), index + order);  // exclusive
        double frac = double(start) - target;
        for (std::ptrdiff_t jj = start; jj < end; ++jj, frac += 1.0) {
            if (haveMask && inputMask[jj]) {
                continue;
            }
            double weight = lanczos(frac);
            sumWeights += weight;
            if (wantValues) {
                value += weight * inputValues[jj];
            }
            if (wantVariance) {
                variance += weight * weight * inputVariance[jj];
            }
        }
        if (wantMask) {
            resultMask[ii] = (sumWeights < minWeight);
        }
        if (sumWeights <= 0.0) {
            if (wantValues) {
                resultValues[ii] = fill;
            }
            if (wantVariance) {
                resultVariance[ii] = fill;
            }
            continue;
        }
        if (wantValues) {
            resultValues[ii] = value / sumWeights;
        }
        if (wantVariance) {
            resultVariance[ii] = variance / (sumWeights*sumWeights);
        }
    }
}
//@}


}  // namespace impl


/// Lanczos interpolation of spectrum values, mask and variance
///
/// This is a full-featured interpolation function, wrapping the general
/// implementation in impl::lanczosInterpolate().
///
/// @tparam T  The values and variance array element type.
/// @tparam U  The indices array element type.
/// @tparam C1  The resultValues array's contiguity.
/// @tparam C2  The resultMask array's contiguity.
/// @tparam C3  The resultVariance array's contiguity.
/// @tparam C4  The inputValues array's contiguity.
/// @tparam C5  The inputMask array's contiguity.
/// @tparam C6  The inputVariance array's contiguity.
/// @tparam C7  The indices array's contiguity.
/// @param[in,out] resultValues  The output values array. This must be
///    preallocated to the size of the indices array.
/// @param[in,out] resultMask  The output mask array. This must be preallocated
///    to the size of the indices array.
/// @param[in,out] resultVariance  The output variance array. This must be
///    preallocated to the size of the indices array.
/// @param[in] inputValues  The input values array.
/// @param[in] inputMask  The input mask array.
/// @param[in] inputVariance  The input variance array.
/// @param[in] indices  The indices array, with floating-point indices at which
///    to interpolate.
/// @param[in] fill  The value to use when interpolation is not possible.
/// @param[in] order  The order of the Lanczos function.
/// @param[in] minWeight  The minimum sum of weights to accept; if the sum of
///    weights is less than this value, the output will be masked.
template <typename T, typename U, int C1, int C2, int C3, int C4, int C5, int C6, int C7>
void lanczosInterpolate(
    ndarray::Array<T, 1, C1> & resultValues,
    ndarray::Array<bool, 1, C2> & resultMask,
    ndarray::Array<T, 1, C3> & resultVariance,
    ndarray::Array<T, 1, C4> const& inputValues,
    ndarray::Array<bool, 1, C5> const& inputMask,
    ndarray::Array<T, 1, C6> const& inputVariance,
    ndarray::Array<U, 1, C7> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER,
    double minWeight=INTERPOLATE_DEFAULT_MIN_WEIGHT
) {
    impl::lanczosInterpolate<true, true, true, true>(
        resultValues, resultMask, resultVariance,
        inputValues, inputMask, inputVariance,
        indices, fill, order, minWeight
    );
}


//@{
/// Lanczos interpolation of 1D flux spectrum
///
/// For use when variance is not needed.
///
/// @tparam T  The values array element type.
/// @tparam U  The indices array element type.
/// @tparam C1  The resultValues array's contiguity.
/// @tparam C2  The resultMask array's contiguity.
/// @tparam C3  The inputValues array's contiguity.
/// @tparam C4  The inputMask array's contiguity.
/// @tparam C5  The indices array's contiguity.
/// @param[in,out] resultValues  The output values array. This must be
///    preallocated to the size of the indices array.
/// @param[in,out] resultMask  The output mask array. This must be preallocated
///    to the size of the indices array.
/// @param[in] inputValues  The input values array.
/// @param[in] inputMask  The input mask array.
/// @param[in] indices  The indices array, with floating-point indices at which
///    to interpolate.
/// @param[in] fill  The value to use when interpolation is not possible.
/// @param[in] order  The order of the Lanczos function.
/// @param[in] minWeight  The minimum sum of weights to accept; if the sum of
///    weights is less than this value, the output will be masked.
template <typename T, typename U, int C1, int C2, int C3, int C4, int C5>
void lanczosInterpolateFlux(
    ndarray::Array<T, 1, C1> & resultValues,
    ndarray::Array<bool, 1, C2> & resultMask,
    ndarray::Array<T, 1, C3> const& inputValues,
    ndarray::Array<bool, 1, C4> const& inputMask,
    ndarray::Array<U, 1, C5> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER,
    double minWeight=INTERPOLATE_DEFAULT_MIN_WEIGHT
) {
    ndarray::Array<T, 1, 1> emptyVar;  // Empty array for variance
    impl::lanczosInterpolate<true, true, false, true>(
        resultValues, resultMask, emptyVar, inputValues, inputMask, emptyVar, indices, fill, order, minWeight
    );
}
template <typename T, typename U, int C1, int C3, int C4, int C5>
ndarray::Array<T, 1, C1> & lanczosInterpolateFlux(
    ndarray::Array<T, 1, C1> & resultValues,
    ndarray::Array<T, 1, C3> const& inputValues,
    ndarray::Array<bool, 1, C4> const& inputMask,
    ndarray::Array<U, 1, C5> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER
) {
    ndarray::Array<bool, 1, 1> emptyMask;  // Empty array for mask
    ndarray::Array<T, 1, 1> emptyVar;  // Empty array for variance
    impl::lanczosInterpolate<true, false, false, true>(
        resultValues, emptyMask, emptyVar, inputValues, inputMask, emptyVar, indices, fill, order
    );
    return resultValues;
}
template <typename T, typename U, int C3, int C4, int C5>
ndarray::Array<T, 1, 1> lanczosInterpolateFlux(
    ndarray::Array<T, 1, C3> const& inputValues,
    ndarray::Array<bool, 1, C4> const& inputMask,
    ndarray::Array<U, 1, C5> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER
) {
    ndarray::Array<T, 1, 1> result = ndarray::allocate(indices.size());
    return lanczosInterpolateFlux(result, inputValues, inputMask, indices, fill, order);
}
template <typename T, typename U, int C3, int C5>
ndarray::Array<T, 1, 1> lanczosInterpolateFlux(
    ndarray::Array<T, 1, C3> const& inputValues,
    ndarray::Array<U, 1, C5> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER
) {
    ndarray::Array<T, 1, 1> result = ndarray::allocate(indices.size());
    ndarray::Array<bool, 1, 1> emptyMask;  // Empty array for mask
    ndarray::Array<T, 1, 1> emptyVar;  // Empty array for variance
    impl::lanczosInterpolate<true, false, false, false>(
        result, emptyMask, emptyVar, inputValues, emptyMask, emptyVar, indices, fill, order
    );
    return result;
}
//@}


//@{
/// Lanczos interpolation of 1D variance spectrum
///
/// For use when flux is not needed.
///
/// @tparam T  The variance array element type.
/// @tparam U  The indices array element type.
/// @tparam C1  The resultVariance array's contiguity.
/// @tparam C2  The resultMask array's contiguity.
/// @tparam C3  The inputVariance array's contiguity.
/// @tparam C4  The inputMask array's contiguity.
/// @tparam C5  The indices array's contiguity.
/// @param[in,out] resultVariance  The output variance array. This must be
///    preallocated to the size of the indices array.
/// @param[in,out] resultMask  The output mask array. If wantMask is true, this
///    must be preallocated to the size of the indices array; otherwise this is
///    ignored.
/// @param[in] inputVariance  The input variance array.
/// @param[in] inputMask  The input mask array.
/// @param[in] indices  The indices array, with floating-point indices at which
///    to interpolate.
/// @param[in] fill  The variance to use when interpolation is not possible.
/// @param[in] order  The order of the Lanczos function.
/// @param[in] minWeight  The minimum sum of weights to accept; if the sum of
///    weights is less than this value, the output will be masked.
template <typename T, typename U, int C1, int C2, int C3, int C4, int C5>
void lanczosInterpolateVariance(
    ndarray::Array<T, 1, C1> & resultVariance,
    ndarray::Array<bool, 1, C2> & resultMask,
    ndarray::Array<T, 1, C3> const& inputVariance,
    ndarray::Array<bool, 1, C4> const& inputMask,
    ndarray::Array<U, 1, C5> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER,
    double minWeight=INTERPOLATE_DEFAULT_MIN_WEIGHT
) {
    ndarray::Array<T, 1, 1> empty;  // Empty array for values
    impl::lanczosInterpolate<false, true, true, true>(
        empty, resultMask, resultVariance, empty, inputMask, inputVariance, indices, fill, order, minWeight
    );
}
template <typename T, typename U, int C1, int C3, int C4, int C5>
ndarray::Array<T, 1, C1> & lanczosInterpolateVariance(
    ndarray::Array<T, 1, C1> & resultVariance,
    ndarray::Array<T, 1, C3> const& inputVariance,
    ndarray::Array<bool, 1, C4> const& inputMask,
    ndarray::Array<U, 1, C5> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER
) {
    ndarray::Array<T, 1, 1> emptyVal;  // Empty array for variance
    ndarray::Array<bool, 1, 1> emptyMask;  // Empty array for mask
    impl::lanczosInterpolate<false, false, true, true>(
        emptyVal, emptyMask, resultVariance, emptyVal, inputMask, inputVariance, indices, fill, order
    );
    return resultVariance;
}
template <typename T, typename U, int C3, int C4, int C5>
ndarray::Array<T, 1, 1> lanczosInterpolateVariance(
    ndarray::Array<T, 1, C3> const& inputVariance,
    ndarray::Array<bool, 1, C4> const& inputMask,
    ndarray::Array<U, 1, C5> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER
) {
    ndarray::Array<T, 1, 1> result = ndarray::allocate(indices.size());
    return lanczosInterpolateVariance(result, inputVariance, inputMask, indices, fill, order);
}
template <typename T, typename U, int C3, int C5>
ndarray::Array<T, 1, 1> lanczosInterpolateVariance(
    ndarray::Array<T, 1, C3> const& inputVariance,
    ndarray::Array<U, 1, C5> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER
) {
    ndarray::Array<T, 1, 1> result = ndarray::allocate(indices.size());
    ndarray::Array<bool, 1, 1> emptyMask;  // Empty array for mask
    ndarray::Array<T, 1, 1> emptyVal;  // Empty array for variance
    impl::lanczosInterpolate<false, false, true, false>(
        emptyVal, emptyMask, result, emptyVal, emptyMask, inputVariance, indices, fill, order
    );
    return result;
}
//@}


}}}}  // namespace pfs::drp::stella::math

#endif  // include guard
