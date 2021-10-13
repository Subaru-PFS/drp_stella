#include "ndarray.h"
#include "ndarray/eigen.h"

#include "pfs/drp/stella/math/SparseSquareMatrix.h"
#include "pfs/drp/stella/utils/checkSize.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {


ndarray::Array<SparseSquareMatrix::ElemT, 1, 1>
SparseSquareMatrix::solve(ndarray::Array<SparseSquareMatrix::ElemT, 1, 1> const& rhs) {
    using Matrix = Eigen::SparseMatrix<ElemT, 0, IndexT>;
    utils::checkSize(rhs.size(), std::size_t(_num), "rhs");
    Matrix matrix(_num, _num);
    matrix.setFromTriplets(_triplets.begin(), _triplets.end());
    Eigen::SparseQR<Matrix, Eigen::NaturalOrdering<IndexT>> solver;
    solver.compute(matrix);
    if (solver.info() != Eigen::Success) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Sparse matrix decomposition failed.");
    }
    ndarray::Array<ElemT, 1, 1> solution = ndarray::allocate(_num);
    ndarray::asEigenMatrix(solution) = solver.solve(ndarray::asEigenMatrix(rhs));
    if (solver.info() != Eigen::Success) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Sparse matrix solving failed.");
    }
    return solution;
}


}}}}  // namespace pfs::drp::stella::math
