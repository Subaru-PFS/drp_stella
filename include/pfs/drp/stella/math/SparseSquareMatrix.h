#ifndef PFS_DRP_STELLA_MATH_SparseSquareMatrix_H
#define PFS_DRP_STELLA_MATH_SparseSquareMatrix_H

#include <vector>
#include <map>
#include <set>

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
    using Map = std::map<IndexT, ElemT>;  // column --> value; std::map is faster than std::unordered_map
    using List = std::vector<Map>;  // row --> (column, value)

    /// Ctor
    ///
    /// @param numCols : number of columns
    /// @param numRows : number of rows
    MatrixTriplets(IndexT numRows, IndexT numCols)
      : _numRows(numRows), _numCols(numCols) {
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

    /// Get the number of non-zero matrix elements in a row
    std::size_t size(IndexT row) const {
        _checkIndex(row, 0);
        return _triplets[row].size();
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

    /// Remove a matrix element
    ///
    /// If the element exists, the value is removed.
    ElemT remove(IndexT row, IndexT col) {
        _checkIndex(row, col);
        Map & map = _triplets[row];
        auto iter = map.find(col);
        if (iter == map.end()) {
            return 0.0;
        }
        ElemT value = iter->second;
        map.erase(iter);
        return value;
    }

    /// Get a matrix element
    ElemT get(IndexT row, IndexT col) const {
        _checkIndex(row, col);
        auto const& map = _triplets[row];
        auto iter = map.find(col);
        if (iter == map.end()) {
            return 0.0;
        }
        return iter->second;
    }

    /// Clear the list of triplets
    void clear() {
        _triplets.resize(_numRows);
        for (IndexT ii = 0; ii < _numRows; ++ii) {
            _triplets[ii].clear();
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

        bool operator<(Triplet const& other) const {
            return (this->row() < other.row() || (this->row() == other.row() && this->col() < other.col()));
        }
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

      protected:

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

    // Iterator that produces symmetric elements in the process
    //
    // For non-diagonal elements, this produces first the original value and
    // then the symmetric value.
    class SymmetricIterator : public Iterator {
      public:
        SymmetricIterator(IndexT row, List const& triplets) : Iterator(row, triplets), _symmetric(false) {}

        Iterator& operator++() {
            if (isDiagonal() || _symmetric) {
                ++this->_mapIter;
                this->_ensureExists();
                _symmetric = false;
            } else {
                _symmetric = true;
            }
            return *this;
        }

        bool operator==(Iterator const& other) const {
            return (this->_listIter == other._listIter &&
                    (this->_listIter == this->_triplets.end() || this->_mapIter == other._mapIter) &&
                    (_symmetric == other._symmetric));
        }

        IndexT row() const { return _symmetric ? this->_mapIter->first : this->_row; }
        IndexT col() const { return _symmetric ? this->_row : this->_mapIter->first; }

        bool isDiagonal() const { return this->_row == this->_mapIter->first; }

      protected:
        bool _symmetric;
    };

    Iterator begin() const { return Iterator(0, _triplets); }
    Iterator end() const { return Iterator(_numRows, _triplets); }
    SymmetricIterator beginSymmetric() const { return SymmetricIterator(0, _triplets); }
    SymmetricIterator endSymmetric() const { return SymmetricIterator(_numRows, _triplets); }

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
    List _triplets;  ///< Triplets
};


namespace detail {

template <int uplo>
struct SparseMatrixView {
    template <class Matrix>
    Matrix get(Matrix const& matrix) const {
        return matrix.template selfadjointView<Eigen::UpLoType(uplo)>();
    }
};


template <>
template <class Matrix>
Matrix SparseMatrixView<0>::get(Matrix const& matrix) const {
    return matrix;
}


#if 0
template <Eigen::UpLoType>
struct SparseMatrixFactorization {
    template <class Matrix, class Solver>
    void operator()(Matrix const& matrix, Solver & solver) const;
};

template <>
template <class Matrix, class Solver>
void SparseMatrixFactorization<Eigen::Upper>::operator()(Matrix const& matrix, Solver & solver) const {
    solver.compute(matrix.template selfadjointView<Eigen::Upper>());
}

template <>
template <class Matrix, class Solver>
void SparseMatrixFactorization<false>::operator()(Matrix const& matrix, Solver & solver) const {
    solver.compute(matrix);
}

template <bool useUpperView>
struct SparseMatrixMultiplication {
    template <class Matrix>
    Eigen::VectorXd operator()(Matrix const& matrix, Eigen::VectorXd const& vector) const;
};

template <>
template <class Matrix>
Eigen::VectorXd SparseMatrixMultiplication<true>::operator()(
    Matrix const& matrix, Eigen::VectorXd const& vector
) const {
    return matrix.template selfadjointView<Eigen::Upper>() * vector;
}

template <>
template <class Matrix>
Eigen::VectorXd SparseMatrixMultiplication<false>::operator()(
    Matrix const& matrix,
    Eigen::VectorXd const& vector
) const {
    return matrix * vector;
}
#endif

}  // namespace detail


template <class Matrix, class Solver>
void solveSparseMatrix(
    Matrix const& matrix,
    ndarray::Array<double, 1, 1> const& rhs,
    ndarray::Array<double, 1, 1> & solution,
    Solver & solver,
    bool debug=false
) {
//    matrix.makeCompressed();

    if (debug) {
        // Save in a form that can be read by Eigen tools
        Eigen::saveMarket(matrix, "matrix.mtx");
        Eigen::saveMarketVector(ndarray::asEigenMatrix(rhs), "matrix_b.mtx");
    }

    solver.compute(matrix);
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


/// Sparse representation of a square matrix
///
/// Used for solving matrix problems.
class SparseSquareMatrix {
  public:
    using ElemT = double;
    using IndexT = std::ptrdiff_t;  // must be signed
    using Matrix = Eigen::SparseMatrix<ElemT, 0, IndexT>;
    using DefaultSolver = Eigen::SparseQR<Matrix, Eigen::NaturalOrdering<IndexT>>;

    /// Ctor
    ///
    /// The matrix is initialised to zero.
    ///
    /// @param num : Number of columns/rows
    SparseSquareMatrix(std::size_t num)
      : _num(num),
        _triplets(num, num)
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

    //@{
    /// Get the number of entries
    std::size_t size() const { return _triplets.size(); }
    std::size_t size(IndexT ii) const { return _triplets.size(ii); }
    //@}

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
        _triplets.add(ii, jj, value);
    }

    /// Add all entries in another matrix to this one
    void add(SparseSquareMatrix const& other) {
        if (other._num != _num) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "Matrices have different sizes");
        }
        for (auto const& triplet : other._triplets) {
            add(triplet.row(), triplet.col(), triplet.value());
        }
    }
    /// Subtract all entries in another matrix from this one
    void subtract(SparseSquareMatrix const& other) {
        if (other._num != _num) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LengthError, "Matrices have different sizes");
        }
        for (auto const& triplet : other._triplets) {
            add(triplet.row(), triplet.col(), -triplet.value());
        }
    }

    /// Remove an entry from the matrix
    ElemT remove(IndexT ii, IndexT jj) {
        return _triplets.remove(ii, jj);
    }

    SparseSquareMatrix & operator+=(SparseSquareMatrix const& other) {
        add(other);
        return *this;
    }
    SparseSquareMatrix & operator-=(SparseSquareMatrix const& other) {
        subtract(other);
        return *this;
    }

    /// Retrieve an entry from the matrix
    ElemT get(IndexT ii, IndexT jj) const {
        return _triplets.get(ii, jj);
    }

    //@{
    /// Solve the matrix equation
    template <int uplo=0, class Solver=DefaultSolver>
    ndarray::Array<ElemT, 1, 1> solve(
        ndarray::Array<ElemT, 1, 1> const& rhs,
        bool makeSymmetric=false,
        bool debug=false
    ) const {
        ndarray::Array<ElemT, 1, 1> solution = ndarray::allocate(_num);
        solve<uplo, Solver>(solution, rhs, makeSymmetric, debug);
        return solution;
    }
    template <int uplo=0, class Solver=DefaultSolver>
    void solve(
        ndarray::Array<ElemT, 1, 1> & solution,
        ndarray::Array<ElemT, 1, 1> const& rhs,
        bool makeSymmetric=false,
        bool debug=false
    ) const {
        Solver solver;
        solve<Solver, uplo>(solution, rhs, solver, makeSymmetric, debug);
    }
    template <class Solver, int uplo=0>
    void solve(
        ndarray::Array<ElemT, 1, 1> & solution,
        ndarray::Array<ElemT, 1, 1> const& rhs,
        Solver & solver,
        bool makeSymmetric=false,
        bool debug=false
    ) const {
        utils::checkSize(rhs.size(), std::size_t(_num), "rhs");
        solveSparseMatrix(getEigen<uplo>(makeSymmetric), rhs, solution, solver, debug);
    }
    //@}

    /// Reset the matrix to zero
    void reset() {
        _triplets.clear();
    }

    /// Add symmetric elements to the matrix
    void symmetrize() {
        std::set<typename MatrixTriplets<ElemT, IndexT>::Triplet> offDiag;
        for (auto const& triplet : _triplets) {
            if (triplet.row() != triplet.col()) {
                offDiag.insert(triplet);
            }
        }
        for (auto const& triplet : offDiag) {
            add(triplet.col(), triplet.row(), triplet.value());
        }
    }

    /// Return the triplets
    MatrixTriplets<ElemT, IndexT> const& getTriplets() const { return _triplets; }

    /// Return the Eigen representation of the matrix
    template <int uplo=0>
    Matrix getEigen(bool makeSymmetric=false) const {
        Matrix matrix{_num, _num};
        if (makeSymmetric) {
            matrix.setFromTriplets(_triplets.beginSymmetric(), _triplets.endSymmetric());
        } else {
            matrix.setFromTriplets(_triplets.begin(), _triplets.end());
        }
        return detail::SparseMatrixView<uplo>().get(matrix);
    }

  protected:
    IndexT _num;  ///< Number of rows/columns
    MatrixTriplets<ElemT, IndexT> _triplets;  ///< Non-zero matrix elements (i,j,value)
};


}}}}  // namespace pfs::drp::stella::math

#endif  // include guard
