#ifndef PFS_DRP_STELLA_IMPL_DISTORTION_H
#define PFS_DRP_STELLA_IMPL_DISTORTION_H

#include "pfs/drp/stella/Distortion.h"
#include "pfs/drp/stella/utils/checkSize.h"


namespace pfs {
namespace drp {
namespace stella {


template <typename Derived>
AnalyticDistortion<Derived>::AnalyticDistortion(
    int xOrder,
    int yOrder,
    lsst::geom::Box2D const& range,
    ndarray::Array<double, 1, 1> const& coeff
) : _xOrder(xOrder),
    _yOrder(yOrder),
    _range(range),
    _coeff(ndarray::copy(coeff))
{}


}}}  // namespace pfs::drp::stella

#endif  // include guard
