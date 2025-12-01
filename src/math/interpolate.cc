#include "ndarray.h"
#include "pfs/drp/stella/math/interpolate.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {


// Explicit instantiation
#define INSTANTIATE(T, U, C1, C2) \
template void interpolate( \
    ndarray::Array<T, 1, C1> & resultValues, \
    ndarray::Array<bool, 1, C1> & resultMask, \
    ndarray::Array<T, 1, C1> & resultVariance, \
    ndarray::Array<T, 1, C1> const& inputValues, \
    ndarray::Array<bool, 1, C1> const& inputMask, \
    ndarray::Array<T, 1, C1> const& inputVariance, \
    ndarray::Array<U, 1, C2> const& indices, \
    T fill, \
    unsigned int order, \
    double minWeight \
); \
template void interpolateFlux( \
    ndarray::Array<T, 1, C1> & resultValues, \
    ndarray::Array<bool, 1, C1> & resultMask, \
    ndarray::Array<T, 1, C1> const& inputValues, \
    ndarray::Array<bool, 1, C1> const& inputMask, \
    ndarray::Array<U, 1, C2> const& indices, \
    T fill, \
    unsigned int order, \
    double minWeight \
); \
template ndarray::Array<T, 1, C1> & interpolateFlux( \
    ndarray::Array<T, 1, C1> & resultValues, \
    ndarray::Array<T, 1, C1> const& inputValues, \
    ndarray::Array<bool, 1, C1> const& inputMask, \
    ndarray::Array<U, 1, C2> const& indices, \
    T fill, \
    unsigned int order \
); \
template ndarray::Array<T, 1, 1> interpolateFlux( \
    ndarray::Array<T, 1, C1> const& inputValues, \
    ndarray::Array<bool, 1, C1> const& inputMask, \
    ndarray::Array<U, 1, C2> const& indices, \
    T fill, \
    unsigned int order \
); \
template ndarray::Array<T, 1, 1> interpolateFlux( \
    ndarray::Array<T, 1, C1> const& inputValues, \
    ndarray::Array<U, 1, C2> const& indices, \
    T fill, \
    unsigned int order \
); \
template void interpolateVariance( \
    ndarray::Array<T, 1, C1> & resultVariance, \
    ndarray::Array<bool, 1, C1> & resultMask, \
    ndarray::Array<T, 1, C1> const& inputVariance, \
    ndarray::Array<bool, 1, C1> const& inputMask, \
    ndarray::Array<U, 1, C2> const& indices, \
    T fill, \
    unsigned int order, \
    double minWeight \
); \
template ndarray::Array<T, 1, C1> & interpolateVariance( \
    ndarray::Array<T, 1, C1> & resultVariance, \
    ndarray::Array<T, 1, C1> const& inputVariance, \
    ndarray::Array<bool, 1, C1> const& inputMask, \
    ndarray::Array<U, 1, C2> const& indices, \
    T fill, \
    unsigned int order=3 \
); \
template ndarray::Array<T, 1, 1> interpolateVariance( \
    ndarray::Array<T, 1, C1> const& inputVariance, \
    ndarray::Array<bool, 1, C1> const& inputMask, \
    ndarray::Array<U, 1, C2> const& indices, \
    T fill, \
    unsigned int order \
); \
template ndarray::Array<T, 1, 1> interpolateVariance( \
    ndarray::Array<T, 1, C1> const& inputVariance, \
    ndarray::Array<U, 1, C2> const& indices, \
    T fill, \
    unsigned int order \
);


INSTANTIATE(double, double, 1, 1)
INSTANTIATE(double, double, 0, 1)
INSTANTIATE(float, double, 1, 1)
INSTANTIATE(float, double, 0, 1)


}}}}  // namespace pfs::drp::stella::math
