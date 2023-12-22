#ifndef PFS_DRP_STELLA_MATH_LeastSquaresEquation_H
#define PFS_DRP_STELLA_MATH_LeastSquaresEquation_H

#include "ndarray_fwd.h"
#include "ndarray/eigen.h"
#include "Eigen/Sparse"

#include "lsst/pex/exceptions.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/math/SparseSquareMatrix.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {

/// Calculate the largest eigenvalue and eigenvector of a square matrix
///
/// This uses the power iteration method. The matrix must be an Eigen matrix
/// type (either sparse or dense).
///
/// @param matrix : Matrix of interest
/// @param tolerance : Tolerance for convergence
/// @param maxIterations : Maximum number of iterations
/// @param eigenvector : Initial guess for eigenvector
/// @return largest eigenvalue and eigenvector
template <bool symmetric, typename Matrix>
std::pair<double, ndarray::Array<double, 1, 1>> calculateLargestEigenvalue(
    Matrix const& matrix,
    double tolerance=1.0e-6,
    std::size_t maxIterations=100,
    ndarray::Array<double, 1, 1> & eigenvector=ndarray::Array<double, 1, 1>()
);


// A matrix equation, Mx = v
class LeastSquaresEquation {
  public:
    using Matrix = SparseSquareMatrix<true>;
    using IndexT = Matrix::IndexT;
    using ElemT = Matrix::ElemT;
    using Vector = ndarray::Array<ElemT, 1, 1>;

    using SparseMatrix = Matrix::Matrix;
    using DefaultSolver = Eigen::SimplicialLDLT<SparseMatrix, Eigen::Upper, Eigen::NaturalOrdering<IndexT>>;

    /// Constructor
    ///
    /// @param num : Number of parameters
    LeastSquaresEquation(IndexT num);

    LeastSquaresEquation(LeastSquaresEquation const& other) = delete;
    LeastSquaresEquation(LeastSquaresEquation && other) = default;
    LeastSquaresEquation & operator=(LeastSquaresEquation const& other) = delete;
    LeastSquaresEquation & operator=(LeastSquaresEquation && other) = default;
    ~LeastSquaresEquation() = default;

    //@{
    /// Accessors
    IndexT size() const { return _num; }
    Matrix const& getMatrix() const { return _matrix; }
    Vector const& getVector() const { return _vector; }
    //@}

    /// Reset the equation
    void reset();

    /// Add diagonal element
    ///
    /// @param ii : Index of diagonal element
    /// @param value : Value to add
    void addDiagonal(IndexT ii, ElemT value) {
        assert(ii >= 0 && ii < _num);
        if (value == 0) return;
        _matrix.add(ii, ii, value);
    }

    /// Add off-diagonal element
    ///
    /// @param ii : First index of off-diagonal element
    /// @param jj : Second index of off-diagonal element
    /// @param value : Value to add
    void addOffDiagonal(IndexT ii, IndexT jj, ElemT value) {
        assert(ii >= 0 && ii < _num);
        assert(jj >= 0 && jj < _num);
        if (value == 0) return;
        if (ii == jj) {
            throw LSST_EXCEPT(lsst::pex::exceptions::LogicError,
                              "Cannot add off-diagonal element to diagonal");
        }
        if (ii < jj) {
            _matrix.add(ii, jj, value);
        } else {
            _matrix.add(jj, ii, value);
        }
    }

    /// Add vector element
    ///
    /// @param ii : Index of vector element
    /// @param value : Value to add
    void addVector(IndexT ii, ElemT value) {
        assert(ii >= 0 && ii < _num);
        _vector[ii] += value;
    }

    /// Add equation elements to this one
    ///
    /// This is useful for combining elements from multiple threads.
    LeastSquaresEquation & operator+=(LeastSquaresEquation const& other);

    /// Subtract equation elements from this one
    ///
    /// This is useful for applying rejection to some observations used to
    /// construct the original equation.
    LeastSquaresEquation & operator-=(LeastSquaresEquation const& other);

    /// Calculate the sparse matrix
    SparseMatrix calculateEigenMatrix() const;

    /// Get a list of empty rows
    std::set<IndexT> getEmptyRows() const;

    /// Make the equation non-singular
    ///
    /// This adds a diagonal element to any row that has no non-zero elements.
    ///
    /// @param value : Value to add to diagonal
    /// @return Set of rows that were made non-singular
    std::set<IndexT> makeNonSingular(double value=1.0);

    /// Undo the effect of makeNonSingular
    ///
    /// @param zeroRows : Set of rows that were made non-singular
    void undoMakeNonSingular(std::set<IndexT> const& zeroRows);

    //@{
    /// Solve the equation
    template <class Solver>
    void solve(Vector & solution, Solver & solver, bool debug=false) {
        utils::checkSize(solution.size(), std::size_t(_num), "solution");
        SparseMatrix matrix = calculateEigenMatrix();
        solveSparseMatrix<true>(matrix, _vector, solution, solver, debug);
    }
    template <class Solver=DefaultSolver>
    void solve(Vector & solution, bool debug=false) {
        Solver solver;
        solve(solution, solver, debug);
    }
    template <class Solver=DefaultSolver>
    Vector solve(bool debug=false) {
        Vector solution = ndarray::allocate(_num);
        solve(solution, debug);
        return solution;
    }
    //@}

    /// Solve a constrained equation using the Proximal Gradient Method
    ///
    /// This minimizes the objective function f(x) = g(x) + h(x), where g(x) is
    /// the usual least-squares objective function and h(x) is an indicator
    /// function (0 when constraints are met and infinity otherwise). The
    /// solution is obtained by iterating with the Proximal Gradient Method.
    /// Some care is taken to ensure that the step size is small enough to
    /// ensure convergence (by using the power iteration method to estimate the
    /// largest eigenvalue of the Hessian matrix and by checking the Armijo
    /// condition).
    ///
    /// The proximal operator is a functor that projects the solution onto the
    /// constrained space. It receives a reference to a Vector, and should
    /// modify its values so that it satisfies the constraints.
    ///
    /// @param proxOperator : Functor that projects the solution onto the constrained space
    /// @param tolerance : Tolerance for convergence
    /// @param maxIterations : Maximum number of iterations
    /// @param armijoMultiplier : Multiplier for Armijo condition
    /// @param powerIterationMultiplier : Multiplier for step size
    /// @param powerIterationTolerance : Tolerance for power iteration method
    /// @param powerIterationMaxIterations : Maximum number of iterations for power iteration method
    /// @return solution vector
    template <class Callable, class Solver=DefaultSolver>
    Vector solveConstrained(
        Callable proxOperator,
        double tolerance=1.0e-4,
        std::size_t maxIterations=1000,
        double armijoMultiplier=0.5,
        double powerIterationMultiplier=0.9,
        double powerIterationTolerance=1.0e-4,
        std::size_t powerIterationMaxIterations=100
    ) {
        SparseMatrix matrix = calculateEigenMatrix();
        detail::SparseMatrixMultiplication<true> const multiplier;
        Vector solution = solve<Solver>();
        proxOperator(solution);
        std::pair<double, Vector> eigen = calculateLargestEigenvalue<true>(
            matrix, powerIterationTolerance, powerIterationMaxIterations, solution
        );
        double step = powerIterationMultiplier / eigen.first;

        Eigen::VectorXd last{_num};
        Eigen::VectorXd next = ndarray::asEigenMatrix(solution);
        Eigen::VectorXd gradient{_num};
        Eigen::VectorXd diff{_num};
        bool retry = false;
        bool converged = false;

        for (std::size_t iter = 0; iter < maxIterations; ++iter) {
            if (!retry) {
                last = next;
                gradient = multiplier(matrix, next) - ndarray::asEigenMatrix(_vector);
            }
            next = last - step*gradient;
            proxOperator(solution);

            diff = next - last;
            double const armijoLeft = diff.dot(multiplier(matrix, diff));
            double const armijoRight = diff.squaredNorm()/step;
            std::cerr << "solveConstrained: armijoLeft = " << armijoLeft
                      << ", armijoRight = " << armijoRight << std::endl;
            if (armijoLeft > armijoRight) {
                step *= armijoMultiplier;
                std::cerr << "solveConstrained: retrying with step = " << step << std::endl;
                retry = true;
                continue;
            }
            retry = false;

            if (diff.squaredNorm() < std::pow(tolerance, 2)) {
                converged = true;
                std::cerr << "solveConstrained converged after " << iter << " iterations" << std::endl;
                break;
            }
        }

        if (!converged) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::RuntimeError, "Proximal gradient method failed to converge"
            );
        }
        return solution;
    }

    /// Write the equation to a FITS file
    ///
    /// This might be a useful visualisation tool for debugging small equations.
    void writeFits(std::string const& filename) const;

  protected:
    IndexT _num;  ///< Number of parameters
    SymmetricSparseSquareMatrix _matrix;  ///< least-squares (Fisher/Hessian) matrix
    Vector _vector;  ///< least-squares (right-hand side) vector
};
#pragma omp declare reduction(+:LeastSquaresEquation:omp_out += omp_in) \
    initializer(omp_priv = LeastSquaresEquation(omp_orig.size()))

}}}}  // namespace pfs::drp::stella::math

#endif  // include guard
