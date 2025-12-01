#ifndef PFS_DRP_STELLA_INTERPOLATE_H
#define PFS_DRP_STELLA_INTERPOLATE_H

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


/// Triangle function, for linear interpolation kernel
///
/// This is a simple triangular function that goes to zero at +/- 1.
class TriangleFunction1 : public lsst::afw::math::Function1<double> {
public:
    TriangleFunction1() : lsst::afw::math::Function1<double>(0) {}
    ~TriangleFunction1() noexcept override = default;

    std::shared_ptr<lsst::afw::math::Function1<double>> clone() const override {
        return std::make_shared<TriangleFunction1>(*this);
    }

    double operator()(double x) const override {
        double const xAbs = std::abs(x);
        if (xAbs >= 1.0) {
            return 0.0;
        }
        return 1.0 - xAbs;
    }

    std::string toString(std::string const& prefix) const override { return "TriangleFunction1()"; }
};


template <typename KernelT>
int getKernelHalfSize(KernelT const& kernel);

template <>
int getKernelHalfSize(TriangleFunction1 const& kernel) {
    return 1;
}

template <>
int getKernelHalfSize(lsst::afw::math::LanczosFunction1<double> const& kernel) {
    return kernel.getOrder();
}


//@{
/// Interpolation of 1D spectrum
///
/// This is a general implementation, handling flux, mask and variance, and used
/// by interpolateFlux() and interpolateVariance(). Template parameters control
/// which of the input and output arrays are provided and used, allowing the
/// compiler to optimize away unused branches. If an array is not used according
/// to the template parameters, any input array will be ignored; using an empty
/// array is recommended in this case.
///
/// @tparam wantValues  True if we want to compute resultValues.
/// @tparam wantMask  True if we want to compute resultMask.
/// @tparam wantVariance  True if we want to compute resultVariance.
/// @tparam haveMask  True if inputMask is provided.
/// @tparam KernelT  The kernel function type.
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
/// @param[in] order  The order of the interpolation kernel.
/// @param[in] minWeight  The minimum sum of weights to accept; if the sum of
///    weights is less than this value, the output will be masked.
template <
    bool wantValues,  // true means have resultValues and inputValues
    bool wantMask,  // true means have resultMask
    bool wantVariance,  // true means have resultVariance and inputVariance
    bool haveMask,  // true means have inputMask
    typename KernelT,
    typename T,
    typename U,
    int C1, int C2, int C3, int C4, int C5, int C6, int C7
>
void interpolate(
    KernelT const& kernel,
    ndarray::Array<T, 1, C1> & resultValues,
    ndarray::Array<bool, 1, C2> & resultMask,
    ndarray::Array<T, 1, C3> & resultVariance,
    ndarray::Array<T, 1, C4> const& inputValues,
    ndarray::Array<bool, 1, C5> const& inputMask,
    ndarray::Array<T, 1, C6> const& inputVariance,
    ndarray::Array<U, 1, C7> const& indices,
    T fill,
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

    int const halfSize = getKernelHalfSize(kernel);
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
        // Want the center of the kernel window to be between index-1 and index
        // frac = 0.75
        // For order=3 --> halfSize=3:
        // start = 0
        // end = 5
        // dx = -2.75, -1.75, -0.75, 0.25, 1.25, 2.25

        // Interpolate using a window of size 2*halfSize centered on target
        double value = 0.0;
        double variance = 0.0;
        double sumWeights = 0.0;
        std::ptrdiff_t const start = std::max(std::ptrdiff_t(0), index - halfSize);  // inclusive
        std::ptrdiff_t const end = std::min(std::ptrdiff_t(numIn), index + halfSize);  // exclusive
        double frac = double(start) - target;
        for (std::ptrdiff_t jj = start; jj < end; ++jj, frac += 1.0) {
            if (haveMask && inputMask[jj]) {
                continue;
            }
            double const weight = kernel(frac);
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
template <
    bool wantValues,  // true means have resultValues and inputValues
    bool wantMask,  // true means have resultMask
    bool wantVariance,  // true means have resultVariance and inputVariance
    bool haveMask,  // true means have inputMask
    typename T,
    typename U,
    int C1, int C2, int C3, int C4, int C5, int C6, int C7
>
void interpolate(
    ndarray::Array<T, 1, C1> & resultValues,
    ndarray::Array<bool, 1, C2> & resultMask,
    ndarray::Array<T, 1, C3> & resultVariance,
    ndarray::Array<T, 1, C4> const& inputValues,
    ndarray::Array<bool, 1, C5> const& inputMask,
    ndarray::Array<T, 1, C6> const& inputVariance,
    ndarray::Array<U, 1, C7> const& indices,
    T fill,
    int order,
    double minWeight=INTERPOLATE_DEFAULT_MIN_WEIGHT
) {
    if (order <= 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Order must be positive");
    }
    if (order == 1) {
        interpolate<wantValues, wantMask, wantVariance, haveMask>(
            impl::TriangleFunction1(),
            resultValues, resultMask, resultVariance,
            inputValues, inputMask, inputVariance,
            indices, fill, minWeight
        );
    } else {
        interpolate<wantValues, wantMask, wantVariance, haveMask>(
            lsst::afw::math::LanczosFunction1<double>(order),
            resultValues, resultMask, resultVariance,
            inputValues, inputMask, inputVariance,
            indices, fill, minWeight
        );
    }
}
//@}


}  // namespace impl


/// Interpolation of spectrum values, mask and variance
///
/// This is a full-featured interpolation function, wrapping the general
/// implementation in impl::interpolate().
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
/// @param[in] order  The order of the interpolation (1 = Triangle, >1 = Lanczos of given order).
/// @param[in] minWeight  The minimum sum of weights to accept; if the sum of
///    weights is less than this value, the output will be masked.
template <typename T, typename U, int C1, int C2, int C3, int C4, int C5, int C6, int C7>
void interpolate(
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
    return impl::interpolate<true, true, true, true>(
        resultValues, resultMask, resultVariance,
        inputValues, inputMask, inputVariance,
        indices, fill, order, minWeight
    );
}


//@{
/// Interpolation of 1D flux spectrum
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
/// @param[in] order  Interpolation order (1 = Triangle, >1 = Lanczos of given order).
/// @param[in] minWeight  The minimum sum of weights to accept; if the sum of
///    weights is less than this value, the output will be masked.
template <typename T, typename U, int C1, int C2, int C3, int C4, int C5>
void interpolateFlux(
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
    impl::interpolate<true, true, false, true>(
        resultValues, resultMask, emptyVar, inputValues, inputMask, emptyVar, indices, fill, order, minWeight
    );
}
template <typename T, typename U, int C1, int C3, int C4, int C5>
ndarray::Array<T, 1, C1> & interpolateFlux(
    ndarray::Array<T, 1, C1> & resultValues,
    ndarray::Array<T, 1, C3> const& inputValues,
    ndarray::Array<bool, 1, C4> const& inputMask,
    ndarray::Array<U, 1, C5> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER
) {
    ndarray::Array<bool, 1, 1> emptyMask;  // Empty array for mask
    ndarray::Array<T, 1, 1> emptyVar;  // Empty array for variance
    impl::interpolate<true, false, false, true>(
        resultValues, emptyMask, emptyVar, inputValues, inputMask, emptyVar, indices, fill, order
    );
    return resultValues;
}
template <typename T, typename U, int C3, int C4, int C5>
ndarray::Array<T, 1, 1> interpolateFlux(
    ndarray::Array<T, 1, C3> const& inputValues,
    ndarray::Array<bool, 1, C4> const& inputMask,
    ndarray::Array<U, 1, C5> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER
) {
    ndarray::Array<T, 1, 1> result = ndarray::allocate(indices.size());
    return interpolateFlux(result, inputValues, inputMask, indices, fill, order);
}
template <typename T, typename U, int C3, int C5>
ndarray::Array<T, 1, 1> interpolateFlux(
    ndarray::Array<T, 1, C3> const& inputValues,
    ndarray::Array<U, 1, C5> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER
) {
    ndarray::Array<T, 1, 1> result = ndarray::allocate(indices.size());
    ndarray::Array<bool, 1, 1> emptyMask;  // Empty array for mask
    ndarray::Array<T, 1, 1> emptyVar;  // Empty array for variance
    impl::interpolate<true, false, false, false>(
        result, emptyMask, emptyVar, inputValues, emptyMask, emptyVar, indices, fill, order
    );
    return result;
}
//@}


//@{
/// Interpolation of 1D variance spectrum
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
/// @param[in] order  Interpolation order (1 = Triangle, >1 = Lanczos of given order).
/// @param[in] minWeight  The minimum sum of weights to accept; if the sum of
///    weights is less than this value, the output will be masked.
template <typename T, typename U, int C1, int C2, int C3, int C4, int C5>
void interpolateVariance(
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
    impl::interpolate<false, true, true, true>(
        empty, resultMask, resultVariance, empty, inputMask, inputVariance, indices, fill, order, minWeight
    );
}
template <typename T, typename U, int C1, int C3, int C4, int C5>
ndarray::Array<T, 1, C1> & interpolateVariance(
    ndarray::Array<T, 1, C1> & resultVariance,
    ndarray::Array<T, 1, C3> const& inputVariance,
    ndarray::Array<bool, 1, C4> const& inputMask,
    ndarray::Array<U, 1, C5> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER
) {
    ndarray::Array<T, 1, 1> emptyVal;  // Empty array for variance
    ndarray::Array<bool, 1, 1> emptyMask;  // Empty array for mask
    impl::interpolate<false, false, true, true>(
        emptyVal, emptyMask, resultVariance, emptyVal, inputMask, inputVariance, indices, fill, order
    );
    return resultVariance;
}
template <typename T, typename U, int C3, int C4, int C5>
ndarray::Array<T, 1, 1> interpolateVariance(
    ndarray::Array<T, 1, C3> const& inputVariance,
    ndarray::Array<bool, 1, C4> const& inputMask,
    ndarray::Array<U, 1, C5> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER
) {
    ndarray::Array<T, 1, 1> result = ndarray::allocate(indices.size());
    return interpolateVariance(result, inputVariance, inputMask, indices, fill, order);
}
template <typename T, typename U, int C3, int C5>
ndarray::Array<T, 1, 1> interpolateVariance(
    ndarray::Array<T, 1, C3> const& inputVariance,
    ndarray::Array<U, 1, C5> const& indices,
    T fill=INTERPOLATE_DEFAULT_FILL,
    unsigned int order=INTERPOLATE_DEFAULT_ORDER
) {
    ndarray::Array<T, 1, 1> result = ndarray::allocate(indices.size());
    ndarray::Array<bool, 1, 1> emptyMask;  // Empty array for mask
    ndarray::Array<T, 1, 1> emptyVal;  // Empty array for variance
    impl::interpolate<false, false, true, false>(
        emptyVal, emptyMask, result, emptyVal, emptyMask, inputVariance, indices, fill, order
    );
    return result;
}
//@}


}}}}  // namespace pfs::drp::stella::math

#endif  // include guard
