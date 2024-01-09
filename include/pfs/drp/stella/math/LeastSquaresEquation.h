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
template <typename Matrix>
std::pair<double, ndarray::Array<double, 1, 1>> calculateLargestEigenvalue(
    Matrix const& matrix,
    double tolerance=1.0e-4,
    std::size_t maxIterations=100,
    ndarray::Array<double, 1, 1> & eigenvector=ndarray::Array<double, 1, 1>()
);


// A matrix equation, Mx = v
class LeastSquaresEquation {
  public:
    using Matrix = SparseSquareMatrix;
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
    ///
    /// When the solver is not specified, the Solver is the first template
    /// parameter.
    template <int uplo=0, class Solver=DefaultSolver>
    Vector solve(Vector & solution, Solver & solver, bool makeSymmetric=false) {
        _matrix.solve<uplo, Solver>(_vector, solution, solver, makeSymmetric);
        return solution;
    }
    template <class Solver, int uplo=0>
    Vector solve(Vector & solution, bool makeSymmetric=false) {
        Solver solver;
        return solve<uplo>(solution, solver, makeSymmetric);
    }
    template <int uplo=0, class Solver=DefaultSolver>
    Vector solve(Vector & solution, bool makeSymmetric=false) {
        return solve<Solver, uplo>(solution, makeSymmetric);
    }
    template <class Solver, int uplo=0>
    Vector solve(bool makeSymmetric=false) {
        Vector solution = ndarray::allocate(_num);
        return solve<Solver, uplo>(solution, makeSymmetric);
    }
    template <int uplo=0, class Solver=DefaultSolver>
    Vector solve(Solver & solver, bool makeSymmetric=false) {
        Vector solution = ndarray::allocate(_num);
        solve<uplo, Solver>(solution, solver, makeSymmetric);
        return solution;
    }
    //@}

    /// Control parameters for solveConstrained
    struct SolveConstrainedControl {
        double tolerance;  ///< Tolerance for convergence
        std::size_t maxIterations;  ///< Maximum number of iterations
        double armijoMultiplier;  ///< Multiplier for Armijo condition
        double powerIterationMultiplier;  ///< Multiplier for step size
        double powerIterationTolerance;  ///< Tolerance for power iteration method
        std::size_t powerIterationMaxIterations;  ///< Maximum number of iterations for power iteration method

        SolveConstrainedControl(
            double tolerance_=1.0e-4,
            std::size_t maxIterations_=1000,
            double armijoMultiplier_=0.5,
            double powerIterationMultiplier_=0.9,
            double powerIterationTolerance_=1.0e-3,
            std::size_t powerIterationMaxIterations_=1000
        ) : tolerance(tolerance_),
            maxIterations(maxIterations_),
            armijoMultiplier(armijoMultiplier_),
            powerIterationMultiplier(powerIterationMultiplier_),
            powerIterationTolerance(powerIterationTolerance_),
            powerIterationMaxIterations(powerIterationMaxIterations_) {}
    };

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
    /// @param solver : Eigen solver to use
    /// @param makeSymmetric : Make the matrix symmetric before solving?
    /// @param control : Control parameters
    /// @return solution vector
    template <class Callable, int uplo=0>
    Vector solveConstrained(
        Callable proxOperator,
        bool makeSymmetric=false,
        SolveConstrainedControl const& ctrl=SolveConstrainedControl()
    ) {
        DefaultSolver solver;
        return solveConstrained<Callable, uplo>(proxOperator, solver, makeSymmetric, ctrl);
    }
    template <class Callable, int uplo=0, class Solver>
    Vector solveConstrained(
        Callable proxOperator,
        Solver & solver,
        bool makeSymmetric=false,
        SolveConstrainedControl const& ctrl=SolveConstrainedControl()
    ) {


        writeFits("equation.fits");

        std::cerr << "Solving constrained problem..." << std::endl;
        SparseMatrix matrix = _matrix.getEigen<uplo>(makeSymmetric);
        matrix.makeCompressed();
        Eigen::saveMarket(matrix, "matrixToSolve.mtx");

        std::cerr << "Wrote matrixToSolve.mtx" << std::endl;

        Vector solution = ndarray::allocate(_num);
        solveSparseMatrix(matrix, _vector, solution, solver);
        proxOperator(solution);
        std::pair<double, Vector> eigen = calculateLargestEigenvalue(
            matrix, ctrl.powerIterationTolerance, ctrl.powerIterationMaxIterations, solution
        );
        double step = ctrl.powerIterationMultiplier / eigen.first;

        Eigen::VectorXd last{_num};
        Eigen::VectorXd next = ndarray::asEigenMatrix(solution);
        Eigen::VectorXd gradient{_num};
        Eigen::VectorXd diff{_num};
        bool retry = false;
        bool converged = false;

        for (std::size_t iter = 0; iter < ctrl.maxIterations; ++iter) {
            if (!retry) {
                last = next;
                gradient = matrix * next - ndarray::asEigenMatrix(_vector);
            }
            next = last - step*gradient;
            proxOperator(solution);

            diff = next - last;
            double const armijoLeft = diff.dot(matrix * diff);
            double const armijoRight = diff.squaredNorm()/step;
            std::cerr << "solveConstrained: armijoLeft = " << armijoLeft
                      << ", armijoRight = " << armijoRight << std::endl;
            if (armijoLeft > armijoRight) {
                step *= ctrl.armijoMultiplier;
                std::cerr << "solveConstrained: retrying with step = " << step << std::endl;
                retry = true;
                continue;
            }
            retry = false;

            std::cerr << "solveConstrained " << iter << ": " << diff.squaredNorm() << " vs " << std::pow(ctrl.tolerance, 2) << std::endl;
            if (diff.squaredNorm() < std::pow(ctrl.tolerance, 2)) {
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

    /// Write to files
    ///
    /// This is intended for debugging, as writing the matrix involves
    /// constructing the Eigen version and writing that.
    ///
    /// The filenames are "matrix.mtx" for the matrix and and "matrix_b.mtx"
    /// for the vector; this is the format used by Eigen's command-line solver.
    void write(bool makeSymmetric=false) {
        _matrix.write("matrix.mtx", makeSymmetric);
        Eigen::saveMarketVector(ndarray::asEigenArray(_vector), "matrix_b.mtx");
    }

    /// Write the equation to a FITS file
    ///
    /// This might be a useful visualisation tool for debugging small equations.
    void writeFits(std::string const& filename) const;

  protected:
    IndexT _num;  ///< Number of parameters
    SparseSquareMatrix _matrix;  ///< least-squares (Fisher/Hessian) matrix
    Vector _vector;  ///< least-squares (right-hand side) vector
};
#pragma omp declare reduction(+:LeastSquaresEquation:omp_out += omp_in) \
    initializer(omp_priv = LeastSquaresEquation(omp_orig.size()))


}}}}  // namespace pfs::drp::stella::math

#endif  // include guard
