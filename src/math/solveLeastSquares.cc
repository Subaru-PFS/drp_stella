#include <numeric>

#include "ndarray.h"
#include "ndarray/eigen.h"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/LeastSquares.h"

#include "pfs/drp/stella/math/solveLeastSquares.h"
#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {


ndarray::Array<double, 1, 1> solveLeastSquaresDesign(
    ndarray::Array<double, 2, 1> const& design,
    ndarray::Array<double, 1, 1> const& meas,
    ndarray::Array<double, 1, 1> const& err,
    double threshold,
    ndarray::Array<bool, 1, 1> const& forced,
    ndarray::Array<double, 1, 1> const& params
) {
    std::size_t const numValues = design.getShape()[0];
    std::size_t numParams = design.getShape()[1];
    utils::checkSize(meas.size(), numValues, "meas vs design");
    if (!err.isEmpty()) {
        utils::checkSize(err.size(), numValues, "err vs design");
    }

    // Deal with any forced parameters
    ndarray::Array<double, 2, 1> modifiedDesign;
    ndarray::Array<double, 1, 1> modifiedMeas;
    std::size_t numForced = 0;
    if (!forced.isEmpty()) {
        utils::checkSize(forced.size(), numParams, "forced vs design");
        numForced = std::accumulate(
            forced.begin(), forced.end(), 0UL, [](bool ff, std::size_t num) { return ff ? num + 1 : num; }
        );
        if (!params.isEmpty()) {
            utils::checkSize(params.size(), numParams, "params vs design");
        }

        modifiedDesign = ndarray::allocate(numValues, numParams - numForced);
        modifiedMeas = ndarray::copy(meas);
        for (std::size_t ii = 0, jj = 0; ii < numParams; ++ii) {
            if (forced[ii]) {
                // Remove the contribution of the forced parameter from the measurements.
                double value = params.isEmpty() ? 0.0 : params[ii];
                asEigenArray(modifiedMeas) -= value*asEigenArray(design[ndarray::view()(ii)]);
            } else {
                // Copy the design matrix for the non-forced parameters.
                asEigenArray(modifiedDesign[ndarray::view()(jj)]) = asEigenArray(design[ndarray::view()(ii)]);
                ++jj;
            }
        }
        numParams -= numForced;
    } else {
        modifiedDesign = design;
        modifiedMeas = meas;
    }

    ndarray::Array<double, 2, 1> designNorm;
    ndarray::Array<double, 1, 1> measNorm;
    if (err.isEmpty()) {
        // No normalisation required
        designNorm = modifiedDesign;
        measNorm = modifiedMeas;
    } else {
        auto const invErr = 1.0/asEigenArray(err);
        designNorm = ndarray::allocate(modifiedDesign.getShape());
        measNorm = ndarray::allocate(modifiedMeas.getShape());
        for (std::size_t ii = 0; ii < numParams; ++ii) {
            asEigenArray(designNorm[ndarray::view()(ii)]) = asEigenArray(modifiedDesign[ndarray::view()(ii)])*invErr;
        }
        asEigenArray(measNorm) = asEigenArray(modifiedMeas)*invErr;
    }
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
    ndarray::Array<double const, 1, 1> const& solution = lsq.getSolution();  // owned by lsq

    if (!forced.isEmpty()) {
        ndarray::Array<double, 1, 1> modifiedSolution = ndarray::allocate(forced.size());
        for (std::size_t ii = 0, jj = 0; ii < forced.size(); ++ii) {
            if (forced[ii]) {
                modifiedSolution[ii] = params.isEmpty() ? 0.0 : params[ii];
            } else {
                modifiedSolution[ii] = lsq.getSolution()[jj];
                ++jj;
            }
        }
        return modifiedSolution;
    }

    return copy(solution);
}


}}}}  // namespace pfs::drp::stella::math
