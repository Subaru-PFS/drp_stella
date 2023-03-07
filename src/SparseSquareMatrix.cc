#include "ndarray.h"
#include "ndarray/eigen.h"

#include "lsst/afw/image.h"
#include "pfs/drp/stella/math/SparseSquareMatrix.h"
#include "pfs/drp/stella/utils/checkSize.h"

#include <unsupported/Eigen/SparseExtra>


namespace pfs {
namespace drp {
namespace stella {
namespace math {


// Explicit instantiation
#define INSTANTIATE(SYMMETRIC) \
template ndarray::Array<typename SparseSquareMatrix<SYMMETRIC>::ElemT, 1, 1> \
SparseSquareMatrix<SYMMETRIC>::solve(ndarray::Array<ElemT, 1, 1> const&) const; \

INSTANTIATE(true);
INSTANTIATE(false);


}}}}  // namespace pfs::drp::stella::math
