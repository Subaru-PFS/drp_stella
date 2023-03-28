#ifndef PFS_DRP_STELLA_MATH_SparseSquareMatrix_H
#define PFS_DRP_STELLA_MATH_SparseSquareMatrix_H

#include <vector>

#include "ndarray_fwd.h"
#include "ndarray/eigen.h"
#include "Eigen/Sparse"
#include "unsupported/Eigen/SparseExtra"
#include "lsst/pex/exceptions.h"
#include "pfs/drp/stella/utils/checkSize.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {


/// Sparse representation of a square matrix
///
/// Used for solving matrix problems.
template <bool symmetric=false>
class SparseSquareMatrix {
  public:
    using ElemT = double;
    using IndexT = std::ptrdiff_t;  // must be signed
    using Matrix = Eigen::SparseMatrix<ElemT, 0, IndexT>;

    /// Ctor
    ///
    /// The matrix is initialised to zero.
    ///
    /// @param num : Number of columns/rows
    /// @param nonZeroPerCol : Estimated mean number of non-zero entries per column
    SparseSquareMatrix(std::size_t num, float nonZeroPerCol=2.0)
      : _num(num) {
        if (num < 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "Number of columns is negative");
        }
        _triplets.reserve(std::size_t(num*nonZeroPerCol));
    }

    virtual ~SparseSquareMatrix() {}
    SparseSquareMatrix(SparseSquareMatrix const&) = delete;
    SparseSquareMatrix(SparseSquareMatrix &&) = default;
    SparseSquareMatrix & operator=(SparseSquareMatrix const&) = delete;
    SparseSquareMatrix & operator=(SparseSquareMatrix &&) = default;

    /// Add an entry to the matrix
    void add(IndexT ii, IndexT jj, ElemT value) {
        if (ii < 0 || ii >= std::ptrdiff_t(_num)) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "Index i is out of range");
        }
        if (jj < 0 || jj >= std::ptrdiff_t(_num)) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "Index j is out of range");
        }
        if (symmetric && jj < ii) {
            // we work with the upper triangle
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "Index j < i for symmetric matrix");
        }
        _triplets.emplace_back(ii, jj, value);
    }

    //@{
    /// Solve the matrix equation
    template <class Solver=Eigen::SparseQR<
        typename SparseSquareMatrix<symmetric>::Matrix,
        Eigen::NaturalOrdering<typename SparseSquareMatrix<symmetric>::IndexT>>
        >
    ndarray::Array<ElemT, 1, 1> solve(ndarray::Array<ElemT, 1, 1> const& rhs, bool debug=false) const {
        ndarray::Array<ElemT, 1, 1> solution = ndarray::allocate(_num);
        solve(solution, rhs, debug);
        return solution;
    }
    template <class Solver=Eigen::SparseQR<
        typename SparseSquareMatrix<symmetric>::Matrix,
        Eigen::NaturalOrdering<typename SparseSquareMatrix<symmetric>::IndexT>>
        >
    void solve(
        ndarray::Array<ElemT, 1, 1> & solution, ndarray::Array<ElemT, 1, 1> const& rhs, bool debug=false
    ) const {
        Solver solver;
        solve(solution, rhs, solver, debug);
    }
    template <class Solver>
    void solve(
        ndarray::Array<ElemT, 1, 1> & solution,
        ndarray::Array<ElemT, 1, 1> const& rhs,
        Solver & solver,
        bool debug=false
    ) const {
        utils::checkSize(rhs.size(), std::size_t(_num), "rhs");
        Matrix matrix{_num, _num};
        matrix.setFromTriplets(_triplets.begin(), _triplets.end());
        matrix.makeCompressed();

        if (debug) {
            // Save in a form that can be read by Eigen tools
            Eigen::saveMarket(matrix, "matrix.mtx");
            Eigen::saveMarketVector(ndarray::asEigenMatrix(rhs), "matrix_b.mtx");
        }

        _compute(solver, matrix);
        if (solver.info() != Eigen::Success) {
            std::ostringstream os;
            os << "Sparse matrix decomposition failed: " << solver.info();
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, os.str());
        }
        ndarray::asEigenMatrix(solution) = solver.solve(ndarray::asEigenMatrix(rhs));
        if (solver.info() != Eigen::Success) {
            std::ostringstream os;
            os << "Sparse matrix solving failed: " << solver.info();
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, os.str());
        }

        if (debug) {
            Eigen::saveMarketVector(ndarray::asEigenMatrix(solution), "solution.mtx");
        }
    }
    //@}

    /// Reset the matrix to zero
    void reset() {
        _triplets.clear();
    }

    /// Return the triplets
    std::vector<Eigen::Triplet<ElemT, IndexT>> const& getTriplets() const { return _triplets; }

  protected:

    /// Compute the solution factorization
    template <class Solver>
    void _compute(Solver & solver, Matrix const& matrix) const;

    IndexT _num;  ///< Number of rows/columns
    std::vector<Eigen::Triplet<ElemT, IndexT>> _triplets;  ///< Non-zero matrix elements (i,j,value)
};


using NonsymmetricSparseSquareMatrix = SparseSquareMatrix<false>;  // more explicit
using SymmetricSparseSquareMatrix = SparseSquareMatrix<true>;  // more explicit


template <bool sym>
std::ostream& operator<<(std::ostream& os, SparseSquareMatrix<sym> const& matrix) {
    return os << matrix.asEigen();
}

// Specialisations
template <>
template <class Solver>
void SparseSquareMatrix<true>::_compute(Solver & solver, Matrix const& matrix) const {
    solver.compute(matrix.selfadjointView<Eigen::Upper>());
}
template <>
template <class Solver>
void SparseSquareMatrix<false>::_compute(Solver & solver, Matrix const& matrix) const {
    solver.compute(matrix);
}


}}}}  // namespace pfs::drp::stella::math

#endif  // include guard
