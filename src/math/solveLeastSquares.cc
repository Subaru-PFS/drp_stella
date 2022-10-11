#include "ndarray.h"
#include "ndarray/eigen.h"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/LeastSquares.h"

#include "pfs/drp/stella/math/solveLeastSquares.h"
#include "pfs/drp/stella/utils/checkSize.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {


ndarray::Array<double, 1, 1> solveLeastSquaresDesign(
    ndarray::Array<double, 2, 1> const& design,
    ndarray::Array<double, 1, 1> const& meas,
    ndarray::Array<double, 1, 1> const& err,
    double threshold
) {
    std::size_t const numValues = design.getShape()[0];
    std::size_t const numParams = design.getShape()[1];
    utils::checkSize(meas.size(), numValues, "meas vs design");
    utils::checkSize(err.size(), numValues, "err vs design");

    ndarray::Array<double, 2, 1> designNorm = ndarray::copy(design);
    for (std::size_t ii = 0; ii < numParams; ++ii) {
        designNorm[ndarray::view()(ii)] /= err;
    }
    ndarray::Array<double, 1, 1> measNorm = ndarray::copy(meas);
    measNorm.deep() /= err;

    auto const dd = asEigenMatrix(designNorm);

    ndarray::Array<double, 2, 1> fisher = ndarray::allocate(numParams, numParams);
    ndarray::Array<double, 1, 1> rhs = ndarray::allocate(numParams);
    asEigenMatrix(fisher) = dd.transpose()*dd;
    asEigenMatrix(rhs) = dd.transpose()*asEigenMatrix(measNorm);

    if (asEigenMatrix(fisher).isZero(0.0)) {
        rhs.deep() = 0.0;
        return rhs;
    }

    auto lsq = lsst::afw::math::LeastSquares::fromNormalEquations(fisher, rhs);
    auto const diag = lsq.getDiagnostic(lsst::afw::math::LeastSquares::NORMAL_EIGENSYSTEM);
    double const sum = asEigenArray(diag).sum();
    lsq.setThreshold(threshold*sum/diag[0]);
    return copy(lsq.getSolution());
}


}}}}  // namespace pfs::drp::stella::math
