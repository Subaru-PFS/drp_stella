#ifndef PFS_DRP_STELLA_MATH_SparseSquareMatrix_H
#define PFS_DRP_STELLA_MATH_SparseSquareMatrix_H

#include <vector>
#include <unordered_set>

#include "ndarray_fwd.h"
#include "Eigen/Sparse"
#include "unsupported/Eigen/SparseExtra"
#include "lsst/pex/exceptions.h"
#include "lsst/cpputils/hashCombine.h"
#include "pfs/drp/stella/utils/checkSize.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {

template <typename ElemT, typename IndexT>
class Triplet;

}}}}  // namespace pfs::drp::stella::math


template <typename ElemT, typename IndexT>
struct std::hash<typename pfs::drp::stella::math::Triplet<ElemT, IndexT>> {
    using argument_type = typename pfs::drp::stella::math::Triplet<ElemT, IndexT>;
    using result_type = std::size_t;
    std::size_t operator()(argument_type const& obj) const noexcept {
        std::size_t seed = 0;
        return lsst::cpputils::hashCombine(seed, obj.row(), obj.col());
    }
};


template <typename ElemT, typename IndexT>
struct std::equal_to<typename pfs::drp::stella::math::Triplet<ElemT, IndexT>> {
    using argument_type = typename pfs::drp::stella::math::Triplet<ElemT, IndexT>;
    constexpr bool operator()(
        argument_type const& left,
        argument_type const& right
    ) const noexcept {
        return left.row() == right.row() && left.col() == right.col();
    }
};



namespace pfs {
namespace drp {
namespace stella {
namespace math {


// Quacks like an Eigen::Triplet, but allows modifying the value.
//
// We could, instead, change TripletCollection to use a std::unordered_map
// where the value is indexed by (row,col), but that would require adding
// an iterator that provides an Eigen::Triplet (or converting the map to a
// std::vector<Eigen::Triplet>, but that would be an inefficient use of memory
// in an area where we very much care about memory). So this solution, while
// just a little dirty through its use of "mutable", gets the job done simply.
template <typename ElemT, typename IndexT>
class Triplet {
  public:
    Triplet(IndexT & row, IndexT & col, ElemT & value)
      : _row(row), _col(col), _value(value) {}

    Triplet(Triplet const&) = delete;
    Triplet(Triplet &&) = default;
    Triplet & operator=(Triplet const&) = delete;
    Triplet & operator=(Triplet &&) = default;

    IndexT & row() { return _row; }
    IndexT & col() { return _col; }
    ElemT & value() { return _value; }

    IndexT const& row() const { return _row; }
    IndexT const& col() const { return _col; }
    ElemT const& value() const { return _value; }

    ElemT operator+=(ElemT & value) const {
        _value += value;
        return _value;
    }
    ElemT operator-=(ElemT & value) const {
        _value -= value;
        return _value;
    }

  private:
    IndexT _row;
    IndexT _col;
    mutable ElemT _value;
};


template <typename ElemT, typename IndexT>
class TripletCollection {
  public:
    using Collection = std::unordered_set<Triplet<ElemT, IndexT>>;

    TripletCollection(std::size_t num) : _triplets(num) {
        _triplets.reserve(num);
    }

    TripletCollection(TripletCollection const&) = delete;
    TripletCollection(TripletCollection &&) = default;
    TripletCollection & operator=(TripletCollection const&) = delete;
    TripletCollection & operator=(TripletCollection &&) = default;

    void add(IndexT row, IndexT col, ElemT value) {
        if (value == 0.0) {
            // Doesn't add anything of value
            return;
        }
        Triplet<ElemT, IndexT> triplet{row, col, value};
        auto elem = _triplets.find(triplet);
        if (elem == _triplets.end()) {
            _triplets.emplace(std::move(triplet));
        } else {
            *elem += value;
        }
    }

    void clear() {
        _triplets.clear();
    }

    typename Collection::const_iterator begin() const { return _triplets.begin(); }
    typename Collection::const_iterator end() const { return _triplets.end(); }

    std::size_t size() const { return _triplets.size(); }

  private:
    std::unordered_set<Triplet<ElemT, IndexT>> _triplets;
};


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
    /// @param nnz : Number of non-zero entries
    SparseSquareMatrix(std::size_t num, std::size_t nnz)
      : _num(num), _triplets(nnz) {}

    bool debug = false;

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
        _triplets.add(ii, jj, value);
    }

    //@{
    /// Solve the matrix equation
    template <class Solver=Eigen::SparseQR<
        typename SparseSquareMatrix<symmetric>::Matrix,
        Eigen::NaturalOrdering<typename SparseSquareMatrix<symmetric>::IndexT>>
        >
    ndarray::Array<ElemT, 1, 1> solve(ndarray::Array<ElemT, 1, 1> const& rhs) const {
        ndarray::Array<ElemT, 1, 1> solution = ndarray::allocate(_num);
        solve(solution, rhs);
        return solution;
    }
    template <class Solver=Eigen::SparseQR<
        typename SparseSquareMatrix<symmetric>::Matrix,
        Eigen::NaturalOrdering<typename SparseSquareMatrix<symmetric>::IndexT>>
        >
    void solve(ndarray::Array<ElemT, 1, 1> & solution, ndarray::Array<ElemT, 1, 1> const& rhs) const {
        utils::checkSize(rhs.size(), _num, "rhs");
        Matrix matrix{std::ptrdiff_t(_num), std::ptrdiff_t(_num)};
        matrix.setFromTriplets(_triplets.begin(), _triplets.end());

        if (debug) {
            Eigen::saveMarket(matrix, "matrix.mtx");
            Eigen::saveMarketVector(asEigenMatrix(rhs), "matrix_b.mtx");
        }

        Solver solver;
        solver.compute(symmetric ? matrix.selfadjointView<Eigen::Upper>() : matrix);
        if (solver.info() != Eigen::Success) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Sparse matrix decomposition failed.");
        }
        ndarray::asEigenMatrix(solution) = solver.solve(ndarray::asEigenMatrix(rhs));
        if (solver.info() != Eigen::Success) {
            throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Sparse matrix solving failed.");
        }
    }
    //@}

    /// Reset the matrix to zero
    void reset() {
        _triplets.clear();
    }

    TripletCollection<ElemT, IndexT> const& getTriplets() const { return _triplets; }

  protected:
    std::size_t _num;  ///< Number of rows/columns
    TripletCollection<ElemT, IndexT> _triplets;  ///< Non-zero matrix elements (i,j,value)
};


using SymmetricSparseSquareMatrix = SparseSquareMatrix<true>;  // more explicit


template <bool sym>
std::ostream& operator<<(std::ostream& os, SparseSquareMatrix<sym> const& matrix) {
    return os << "SparseSquareMatrix<" << (sym ? "true" : "") << ">[" << matrix.getTriplets() << "]";
}

template <typename ElemT, typename IndexT>
std::ostream& operator<<(std::ostream& os, TripletCollection<ElemT, IndexT> const& triplets) {
    os << "TripletCollection[";
    for (auto & tt : triplets) {
        os << "(" << tt.row() << "," << tt.col() << "," << tt.value() << ")" << ",";
    }
    os << "]";
    return os;
}



}}}}  // namespace pfs::drp::stella::math

#endif  // include guard
