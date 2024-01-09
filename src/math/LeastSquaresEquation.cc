#include "lsst/afw/image/Image.h"
#include "pfs/drp/stella/math/LeastSquaresEquation.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {


template <typename Matrix>
std::pair<double, ndarray::Array<double, 1, 1>> calculateLargestEigenvalue(
    Matrix const& matrix,
    double tolerance,
    std::size_t maxIterations,
    ndarray::Array<double, 1, 1> & eigenvector
) {
    auto const size = matrix.rows();
    if (matrix.cols() != size) {
        // Don't want to have to worry about left eigenvectors vs right eigenvectors
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError, "Matrix must be square");
    }
    if (eigenvector.empty()) {
        eigenvector = ndarray::allocate(size);
        eigenvector.deep() = 1.0;  // initial guess
    } else {
        utils::checkSize(eigenvector.size(), std::size_t(size), "eigenvector");
    }

    Eigen::VectorXd last(size);
    Eigen::VectorXd next = ndarray::asEigenMatrix(eigenvector);
    bool converged = false;
    for (std::size_t iter = 0; iter < maxIterations; ++iter) {
        last = next;
        next = matrix * last;
        next /= next.norm();
        if ((next - last).squaredNorm() < std::pow(tolerance, 2)) {
            converged = true;
            std::cerr << "calculateLargestEigenvalue converged after " << iter << " iterations: ";
            break;
        }
    }
    if (!converged) {
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "Power iteration method failed to converge");
    }
    double const eigenvalue = next.dot(matrix * next) / next.squaredNorm();
    std::cerr << "eigenvalue = " << eigenvalue << std::endl;
    return std::make_pair(eigenvalue, eigenvector);
}


LeastSquaresEquation::LeastSquaresEquation(IndexT num) :
    _num(num), _matrix(num), _vector(ndarray::allocate(num)) {
    reset();
}


void LeastSquaresEquation::reset() {
    _matrix.reset();
    _vector.deep() = 0;
}


LeastSquaresEquation & LeastSquaresEquation::operator+=(LeastSquaresEquation const& other) {
    _matrix += other._matrix;
    asEigenArray(_vector) += asEigenArray(other._vector);
    return *this;
}


LeastSquaresEquation & LeastSquaresEquation::operator-=(LeastSquaresEquation const& other) {
    _matrix -= other._matrix;
    asEigenArray(_vector) -= asEigenArray(other._vector);
    return *this;
}

#if 0
LeastSquaresEquation::SparseMatrix LeastSquaresEquation::calculateEigenMatrix() const {
    SparseMatrix matrix(_num, _num);
    matrix.setFromTriplets(_matrix.getTriplets().begin(), _matrix.getTriplets().end());
    matrix.makeCompressed();
    return matrix;
}
#endif

std::set<LeastSquaresEquation::IndexT> LeastSquaresEquation::getEmptyRows() const {
    std::set<IndexT> empty;
    for (IndexT ii = 0; ii < _num; ++ii) {
        if (_matrix.size(ii) == 0) {
            empty.insert(ii);
        }
    }
    return empty;
}


std::set<LeastSquaresEquation::IndexT> LeastSquaresEquation::makeNonSingular(double value) {
    std::set<IndexT> empty = getEmptyRows();
    for (IndexT ii : empty) {
        if (_vector[ii] != 0.0) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LogicError,
                "Cannot make equation non-singular: non-zero vector element with zero matrix element"
            );
        }
        addDiagonal(ii, value);
    }
    return empty;
}


void LeastSquaresEquation::undoMakeNonSingular(std::set<LeastSquaresEquation::IndexT> const& zeroRows) {
    for (IndexT ii : zeroRows) {
        _matrix.remove(ii, ii);
    }
}


void LeastSquaresEquation::writeFits(std::string const& filename) const {
    lsst::afw::image::Image<double> image(_num + 1, _num);
    image = 0.0;
    for (auto const& tt : _matrix.getTriplets()) {
        image(tt.col(), tt.row()) = tt.value();
    }
    image.getArray()[ndarray::view()(_num)].deep() = _vector;
    image.writeFits(filename);
}


// Explicit instantiation
template std::pair<double, ndarray::Array<double, 1, 1>> calculateLargestEigenvalue<>(
    typename SparseSquareMatrix::Matrix const& matrix,
    double tolerance,
    std::size_t maxIterations,
    ndarray::Array<double, 1, 1> & eigenvector
);


}}}} // namespace pfs::drp::stella::math
