#ifndef PFS_DRP_STELLA_IMPL_DISTORTION_H
#define PFS_DRP_STELLA_IMPL_DISTORTION_H

#include "pfs/drp/stella/BaseDistortion.h"
#include "pfs/drp/stella/utils/checkSize.h"


namespace pfs {
namespace drp {
namespace stella {


template <typename Derived>
BaseDistortion<Derived>::BaseDistortion(
    int order,
    lsst::geom::Box2D const& range,
    ndarray::Array<double, 1, 1> const& coeff
) : _order(order),
    _range(range),
    _coeff(ndarray::copy(coeff))
{
    utils::checkSize(coeff.size(), getNumParametersForOrder(order), "coefficients vs order");
}


template <typename Derived>
ndarray::Array<double, 2, 1> BaseDistortion<Derived>::operator()(
    ndarray::Array<double, 1, 1> const& xx,
    ndarray::Array<double, 1, 1> const& yy
) const {
    std::size_t const length = xx.size();
    utils::checkSize(yy.size(), length, "x vs y");
    ndarray::Array<double, 2, 1> out = ndarray::allocate(length, 2);
    for (std::size_t ii = 0; ii < length; ++ii) {
        auto const point = operator()(lsst::geom::Point2D(xx[ii], yy[ii]));
        out[ii][0] = point.getX();
        out[ii][1] = point.getY();
    }
    return out;
}


template <typename Derived>
std::pair<double, std::size_t> BaseDistortion<Derived>::calculateChi2(
    ndarray::Array<double, 1, 1> const& xOrig,
    ndarray::Array<double, 1, 1> const& yOrig,
    ndarray::Array<double, 1, 1> const& xMeas,
    ndarray::Array<double, 1, 1> const& yMeas,
    ndarray::Array<double, 1, 1> const& xErr,
    ndarray::Array<double, 1, 1> const& yErr,
    ndarray::Array<bool, 1, 1> const& goodOrig,
    float sysErr
) const {
    std::size_t const length = xOrig.size();
    utils::checkSize(yOrig.size(), length, "yOrig");
    utils::checkSize(xMeas.size(), length, "xMeas");
    utils::checkSize(yMeas.size(), length, "yMeas");
    utils::checkSize(xErr.size(), length, "xErr");
    utils::checkSize(yErr.size(), length, "yErr");
    ndarray::Array<bool, 1, 1> good;
    if (goodOrig.isEmpty()) {
        good = ndarray::allocate(length);
        good.deep() = true;
    } else {
        good = goodOrig;
        utils::checkSize(good.size(), length, "good");
    }
    double const sysErr2 = std::pow(sysErr, 2);
    double chi2 = 0.0;
    std::size_t num = 0;
    for (std::size_t ii = 0; ii < length; ++ii) {
        if (!good[ii]) continue;
        double const xErr2 = std::pow(xErr[ii], 2) + sysErr2;
        double const yErr2 = std::pow(yErr[ii], 2) + sysErr2;
        lsst::geom::Point2D const fit = operator()(lsst::geom::Point2D(xOrig[ii], yOrig[ii]));
        chi2 += std::pow(xMeas[ii] - fit.getX(), 2)/xErr2 + std::pow(yMeas[ii] - fit.getY(), 2)/yErr2;
        num += 2;  // one for x, one for y
    }
    std::size_t const numFitParams = getNumParameters();
    std::size_t const dof = num - numFitParams;
    return std::make_pair(chi2, dof);
}


}}}  // namespace pfs::drp::stella

#endif  // include guard
