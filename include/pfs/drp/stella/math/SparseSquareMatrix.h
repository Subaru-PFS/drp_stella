#ifndef PFS_DRP_STELLA_MATH_SparseSquareMatrix_H
#define PFS_DRP_STELLA_MATH_SparseSquareMatrix_H

#include <vector>

#include "ndarray_fwd.h"
#include "Eigen/Sparse"

namespace pfs {
namespace drp {
namespace stella {
namespace math {


/// Sparse representation of a square matrix
///
/// Used for solving matrix problems.
class SparseSquareMatrix {
  public:
    using ElemT = double;
    using IndexT = std::ptrdiff_t;  // must be signed

    /// Ctor
    ///
    /// The matrix is initialised to zero.
    ///
    /// @param num : Number of columns/rows
    /// @param nonZeroPerCol : Estimated mean number of non-zero entries per column
    SparseSquareMatrix(std::size_t num, float nonZeroPerCol=2.0)
      : _num(num) {
        _triplets.reserve(std::size_t(num*nonZeroPerCol));
    }

    virtual ~SparseSquareMatrix() {}
    SparseSquareMatrix(SparseSquareMatrix const&) = delete;
    SparseSquareMatrix(SparseSquareMatrix &&) = default;
    SparseSquareMatrix & operator=(SparseSquareMatrix const&) = delete;
    SparseSquareMatrix & operator=(SparseSquareMatrix &&) = default;

    /// Add an entry to the matrix
    void add(IndexT ii, IndexT jj, ElemT value) {
        assert(ii >= 0 && jj >= 0 && ii < std::ptrdiff_t(_num) && jj < std::ptrdiff_t(_num));
        _triplets.emplace_back(ii, jj, value);
    }

    /// Solve the matrix equation
    ndarray::Array<ElemT, 1, 1> solve(ndarray::Array<ElemT, 1, 1> const& rhs);

  private:
    std::size_t _num;  ///< Number of rows/columns
    std::vector<Eigen::Triplet<ElemT>> _triplets;  ///< Triplets (two coordinates and a value) of matrix elems
};


}}}}  // namespace pfs::drp::stella::math

#endif  // include guard
