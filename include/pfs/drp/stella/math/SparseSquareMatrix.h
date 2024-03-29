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


// Collection of triplets (row, column, value) of sparse matrix
//
// A std::vector of triplets is fine if all the elements have been accumulated
// ahead of time, but if not then simply collecting values can produce a
// collection of triplets much larger than desired. This collection of triplets
// does the accumulation of matrix elements automatically.
template <typename ElemT=double, typename IndexT=std::ptrdiff_t>
class MatrixTriplets {
  public:
    using Map = std::unordered_map<IndexT, ElemT>;  // column --> value
    using List = std::vector<Map>;  // row --> (column, value)

    /// Ctor
    ///
    /// @param numCols : number of columns
    /// @param numRows : number of rows
    /// @param nonZeroPerRow : Estimated mean number of non-zero entries per row
    MatrixTriplets(IndexT numRows, IndexT numCols, float nonZeroPerRow=2.0)
      : _numRows(numRows), _numCols(numCols), _numNonZero(std::ceil(nonZeroPerRow)) {
        if (numRows < 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "Number of rows is negative");
        }
        if (numCols < 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "Number of columns is negative");
        }
        _triplets.reserve(numRows);
        clear();
    }

    virtual ~MatrixTriplets() {}
    MatrixTriplets(MatrixTriplets const&) = delete;
    MatrixTriplets(MatrixTriplets &&) = default;
    MatrixTriplets & operator=(MatrixTriplets const&) = delete;
    MatrixTriplets & operator=(MatrixTriplets &&) = default;

    std::size_t size() const {
        std::size_t num = 0;
        for (auto const& map : _triplets) {
            num += map.size();
        }
        return num;
    }

    //@{
    /// Add a matrix element
    ///
    /// If the element exists, the value is added to the existing value.
    void add(IndexT row, IndexT col, ElemT value) {
        _checkIndex(row, col);
        if (value == 0.0) {
            return;
        }
        Map & map = _triplets[row];
        auto result = map.insert({col, value});
        if (!result.second) {  // there's already an entry
            result.first->second += value;
        }
    }
    //@}

    /// Clear the list of triplets
    void clear() {
        _triplets.clear();
        for (IndexT ii = 0; ii < _numRows; ++ii) {
            _triplets.emplace_back(_numNonZero);
        }
    }

    /// Copy-constructable and assignable version of Eigen::Triplet
    /// This allows our iterator to work.
    class Triplet : public Eigen::Triplet<ElemT, IndexT> {
      public:
        using Base = Eigen::Triplet<ElemT, IndexT>;

        Triplet(IndexT row, IndexT col, ElemT value)
          : Base(row, col, value) {}

        Triplet(Triplet const&) = default;
        Triplet(Triplet &&) = default;
        Triplet & operator=(Triplet const&) = default;
        Triplet & operator=(Triplet &&) = default;
    };


    class Iterator {
      public:
        using ListIterator = typename List::const_iterator;
        using MapIterator = typename Map::const_iterator;
        using iterator_category = std::input_iterator_tag;
        using value_type = Triplet;
        using reference = Triplet;
        using pointer = Triplet*;

        Iterator(IndexT row, List const& triplets)
          : _row(row),
            _triplets(triplets),
            _listIter(triplets.cbegin() + row),
            _current(IndexT(), IndexT(), ElemT()) {
            if (_listIter != _triplets.end()) {
                _mapIter = _listIter->cbegin();
                _ensureExists();
            }
        }

        Iterator& operator++() {
            ++_mapIter;
            _ensureExists();
            return *this;
        }
        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        value_type operator*() const {
            return Triplet(row(), col(), value());
        }
        pointer operator->() const {
            _current = **this;
            return &_current;
        }

        bool operator==(Iterator const& other) const {
            return (_listIter == other._listIter &&
                    (_listIter == _triplets.end() || _mapIter == other._mapIter));
        }
        bool operator!=(Iterator const& other) const {
            return !(this->operator==(other));
        }

        IndexT row() const { return _row; }
        IndexT col() const { return _mapIter->first; }
        ElemT value() const { return _mapIter->second; }

      private:

        // Ensure that the iterators point to a valid entry (or the end)
        void _ensureExists() {
            while (_listIter != _triplets.end() && _mapIter == _listIter->cend()) {
                ++_row;
                ++_listIter;
                if (_listIter != _triplets.end()) {
                    _mapIter = _listIter->cbegin();
                }
            }
        }

        IndexT _row;
        List const& _triplets;
        ListIterator _listIter;
        MapIterator _mapIter;
        mutable value_type _current;
    };

    Iterator begin() const { return Iterator(0, _triplets); }
    Iterator end() const { return Iterator(_numRows, _triplets); }

    /// Convert to a classic list of Eigen triplets
    std::vector<Eigen::Triplet<ElemT, IndexT>> getEigen() const {
        std::vector<Eigen::Triplet<ElemT, IndexT>> result;
        result.reserve(_triplets.size());
        for (auto const& triplet : *this) {
            result.emplace_back(triplet.row(), triplet.col(), triplet.value());
        }
        return result;
    }

  private:

    void _checkIndex(IndexT row, IndexT col) const {
        if (row < 0 || row >= _numRows) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "Row index is out of range");
        }
        if (col < 0 || col >= _numCols) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "Column index is out of range");
        }
    }

    IndexT _numRows;  ///< Number of rows
    IndexT _numCols;  ///< Number of columns
    std::size_t _numNonZero;  ///< Number of non-zero elements per row
    List _triplets;  ///< Triplets
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
    using DefaultSolver = Eigen::SparseQR<
        typename SparseSquareMatrix<symmetric>::Matrix,
        Eigen::NaturalOrdering<typename SparseSquareMatrix<symmetric>::IndexT>
    >;

    /// Ctor
    ///
    /// The matrix is initialised to zero.
    ///
    /// @param num : Number of columns/rows
    /// @param nonZeroPerRow : Estimated mean number of non-zero entries per row
    SparseSquareMatrix(std::size_t num, float nonZeroPerRow=2.0)
      : _num(num),
        _triplets(num, num, nonZeroPerRow)
       {
        if (num < 0) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "Number of columns is negative");
        }
    }

    virtual ~SparseSquareMatrix() {}
    SparseSquareMatrix(SparseSquareMatrix const&) = delete;
    SparseSquareMatrix(SparseSquareMatrix &&) = default;
    SparseSquareMatrix & operator=(SparseSquareMatrix const&) = delete;
    SparseSquareMatrix & operator=(SparseSquareMatrix &&) = default;

    /// Get the number of entries
    std::size_t size() const { return _triplets.size(); }

    /// Add an entry to the matrix
    void add(IndexT ii, IndexT jj, ElemT value) {
#if 0  // Triplets does this for us
        if (ii < 0 || ii >= std::ptrdiff_t(_num)) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "Index i is out of range");
        }
        if (jj < 0 || jj >= std::ptrdiff_t(_num)) {
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "Index j is out of range");
        }
#endif
        if (symmetric && jj < ii) {
            // we work with the upper triangle
            throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError, "Index j < i for symmetric matrix");
        }
        _triplets.add(ii, jj, value);
    }

    //@{
    /// Solve the matrix equation
    template <class Solver=DefaultSolver>
    ndarray::Array<ElemT, 1, 1> solve(ndarray::Array<ElemT, 1, 1> const& rhs, bool debug=false) const {
        ndarray::Array<ElemT, 1, 1> solution = ndarray::allocate(_num);
        solve<Solver>(solution, rhs, debug);
        return solution;
    }
    template <class Solver=DefaultSolver>
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
    MatrixTriplets<ElemT, IndexT> const& getTriplets() const { return _triplets; }

  protected:

    /// Compute the solution factorization
    template <class Solver>
    void _compute(Solver & solver, Matrix const& matrix) const;

    IndexT _num;  ///< Number of rows/columns
    MatrixTriplets<ElemT, IndexT> _triplets;  ///< Non-zero matrix elements (i,j,value)
};


using NonsymmetricSparseSquareMatrix = SparseSquareMatrix<false>;  // more explicit
using SymmetricSparseSquareMatrix = SparseSquareMatrix<true>;  // more explicit


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
