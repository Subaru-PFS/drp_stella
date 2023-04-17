#ifndef PFS_DRP_STELLA_MATH_SOLVELEASTSQUARES_H
#define PFS_DRP_STELLA_MATH_SOLVELEASTSQUARES_H

#include "ndarray_fwd.h"
#include "ndarray/eigen_fwd.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {


/// Solve a matrix equation
///
/// Generates the normal equations and solves them using
/// lsst::afw::math::LeastSquares::fromNormalEquations .
///
/// For more information and description of terminology used (e.g., "design
/// matrix", "normal equations") see the doxygen of
/// lsst::afw::math::LeastSquares.
///
/// @param design : Design matrix of the equation
/// @param meas : Measurements
/// @param err : Measurement errors
/// @return equation solution
ndarray::Array<double, 1, 1> solveLeastSquaresDesign(
    ndarray::Array<double, 2, 1> const& design,
    ndarray::Array<double, 1, 1> const& meas,
    ndarray::Array<double, 1, 1> const& err=ndarray::Array<double, 1, 1>(),
    double threshold = 1.0e-6
);


}}}}  // namespace pfs::drp::stella::math

#endif  // include guard
