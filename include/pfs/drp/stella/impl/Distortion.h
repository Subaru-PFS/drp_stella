#ifndef PFS_DRP_STELLA_IMPL_DISTORTION_H
#define PFS_DRP_STELLA_IMPL_DISTORTION_H

#include "pfs/drp/stella/Distortion.h"
#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"


namespace pfs {
namespace drp {
namespace stella {


template <typename Derived>
AnalyticDistortion<Derived>::AnalyticDistortion(
    int order,
    lsst::geom::Box2D const& range,
    ndarray::Array<double, 1, 1> const& coeff
) : _order(order),
    _range(range),
    _coeff(ndarray::copy(coeff))
{}


namespace detail {

/// Return the elements of an optional fitting array at the indices
///
/// This is for selecting the elements of the "forced" and "params" arrays that
/// apply to a subset of the parameters (e.g., a single dimension). If the
/// array is empty, an empty array is returned.
///
/// @param array : Optional array (may be empty)
/// @param indices : Indices of elements to select
/// @return Selected elements, or an empty array
template <typename T>
ndarray::Array<T, 1, 1> takeOptional(
    ndarray::Array<T, 1, 1> const& array,
    std::vector<std::size_t> const& indices
) {
    if (array.isEmpty()) {
        return ndarray::Array<T, 1, 1>();
    }
    return utils::arrayTake(array, indices);
}

}  // namespace detail


}}}  // namespace pfs::drp::stella

#endif  // include guard
