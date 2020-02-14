#include "ndarray.h"
#include "lsst/pex/exceptions.h"
#include "pfs/drp/stella/math/symmetricTridiagonal.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {

//
// Implementation of the "tridiagonal matrix algorithm", which provides a
// solution to the matrix equation for a tridiagonal matrix
// (https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm), modified to
// solve a symmetric tridiagonal matrix equation.
//
// Note that we are using the version of this algorithm that requires a
// workspace, because I didn't work out how to modify the in-place algorithm
// to accomodate the lack of the second off-diagonal array.
//
// Using the notation in the function declaration, we calculate:
//
// cPrime[1] = c[1]/b[1]
// dPrime[1] = d[1]/b[1]
// cPrime[i] = c[i]/(b[i] - c[i-1]*cPrime[i-1])
// dPrime[i] = (d[i] - c[i-1]*dPrime[i-1])/(b[i] - c[i-1]*cPrime[i-1])
//
// Then the solution to the equation is:
// x[n] = dPrime[n]
// x[i] = (dPrime[i] - cPrime[i]*x[i+1])
//
template <typename T>
ndarray::Array<T, 1, 1> solveSymmetricTridiagonal(
    ndarray::Array<T, 1, 1> const& diagonal,
    ndarray::Array<T, 1, 1> const& offDiag,
    ndarray::Array<T, 1, 1> const& answer,
    SymmetricTridiagonalWorkspace<T> & workspace
) {
    std::size_t const num = diagonal.getNumElements();  // number of elements
    std::size_t const last = num - 1;  // index of last element
    std::size_t const penultimate = num - 2;  // index of penultimate element

    assert(offDiag.getNumElements() == num - 1);
    assert(answer.getNumElements() == num);

    workspace.reset(num);
    auto & cPrime = workspace.shortArray;
    auto & dPrime = workspace.longArray1;
    auto solution = workspace.longArray2;

    if (num == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
                          "Nothing to solve");
    }
    if (num == 1) {
        solution[0] = answer[0]/diagonal[0];
        return solution;
    }

    cPrime[0] = offDiag[0]/diagonal[0];
    dPrime[0] = answer[0]/diagonal[0];
    for (std::size_t ii = 1, jj = 0; ii < last; ++ii, ++jj) {  // jj = ii - 1
        T const denominator = diagonal[ii] - offDiag[jj]*cPrime[jj];
        cPrime[ii] = offDiag[ii]/denominator;
        dPrime[ii] = (answer[ii] - offDiag[jj]*dPrime[jj])/denominator;
    }
    T const denominator = diagonal[last] - offDiag[penultimate]*cPrime[penultimate];
    dPrime[last] = (answer[last] - offDiag[penultimate]*dPrime[penultimate])/denominator;

    solution[last] = dPrime[last];
    for (std::ptrdiff_t ii = penultimate, jj = last; ii >= 0; --ii, --jj) {  // jj = ii + 1
        solution[ii] = (dPrime[ii] - cPrime[ii]*solution[jj]);
    }

    return solution;
}

//
// Implementation of an algorithm to invert a tridiagonal matrix, modified to
// operate on a symmetric tridiagonal matrix.
//
// The algorithm is from Theorem 2.3 of "A Review on the Inverse of
// Symmetric Tridiagonal and Block Tridiagonal Matrices" (Meurant, 1992,
// DOI:10.1137/0613045;
// https://www.researchgate.net/publication/244450365_A_Review_on_the_Inverse_of_Symmetric_Tridiagonal_and_Block_Tridiagonal_Matrices)
//
// Using the notation in the function declaration, we calculate:
//
// delta[1] = b[1]
// epsilon[n] = b[n]
// delta[i] = b[i] - c[i-1]^2/delta[i-1]
// epsilon[i] = b[i] - c[i]^2/epsilon[i+1]
//
// Then the diagonal and off-diagonal elements of the matrix inverse are:
// M^-1[i,i] = epsilon[i+1] . Product_{j=i+2..n}(epsilon[j]) / Product_{j=i..n}(delta[j])
// M^-1[i,i+1] = -c[i] . Product_{j=i+2..n}(epsilon[j]) / Product_{j=i..n}(delta[j])
//
template <typename T>
std::tuple<ndarray::Array<T, 1, 1>, ndarray::Array<T, 1, 1>>
invertSymmetricTridiagonal(
    ndarray::Array<T, 1, 1> const& diagonal,
    ndarray::Array<T, 1, 1> const& offDiag,
    SymmetricTridiagonalWorkspace<T> & workspace
) {
    std::size_t const num = diagonal.getNumElements();  // number of elements
    std::size_t const last = num - 1;  // index of last element
    std::size_t const penultimate = num - 2;  // index of penultimate element
    assert(offDiag.getNumElements() == num - 1);

    workspace.reset(num);
    auto & delta = workspace.longArray1;
    auto variance = workspace.longArray2;
    auto covariance = workspace.shortArray;

    if (num == 0) {
        throw LSST_EXCEPT(lsst::pex::exceptions::LengthError,
                          "Nothing to invert");
    }
    if (num == 1) {
        variance[0] = 1.0/diagonal[0];
        return std::make_tuple(variance, covariance);
    }

    delta[0] = diagonal[0];
    for (std::size_t ii = 1, jj = 0; ii < num; ++ii, ++jj) {  // jj = ii - 1
        delta[ii] = diagonal[ii] - std::pow(offDiag[jj], 2)/delta[jj];
    }

    double epsilon1 = diagonal[last];  // epsilon[i+1]
    double epsilon2 = 1.0;  // epsilon[i+2]
    double ratioProduct = 1.0/delta[last]; // Product_{j=i+2..n}(epsilon[j]) / Product_{j=i..n}(delta[j])
    variance[last] = 1.0/delta[last];
    for (std::ptrdiff_t ii = penultimate, jj = last; ii >= 0; --ii, --jj) {  // jj = ii + 1
        ratioProduct *= epsilon2/delta[ii];
        epsilon2 = epsilon1;
        epsilon1 = diagonal[ii] - std::pow(offDiag[ii], 2)/epsilon1;

        variance[ii] = epsilon2*ratioProduct;
        covariance[ii] = -offDiag[ii]*ratioProduct;
    }

    return std::make_tuple(variance, covariance);
}


}}}}  // namespace pfs::drp::stella::math


// Explicit instantiations
#define INSTANTIATE(TYPE) \
    template ndarray::Array<TYPE, 1, 1> \
    pfs::drp::stella::math::solveSymmetricTridiagonal( \
        ndarray::Array<TYPE, 1, 1> const&, \
        ndarray::Array<TYPE, 1, 1> const&, \
        ndarray::Array<TYPE, 1, 1> const&, \
        pfs::drp::stella::math::SymmetricTridiagonalWorkspace<TYPE> & \
    ); \
    \
    template std::tuple<ndarray::Array<TYPE, 1, 1>, ndarray::Array<TYPE, 1, 1>> \
    pfs::drp::stella::math::invertSymmetricTridiagonal( \
        ndarray::Array<TYPE, 1, 1> const&, \
        ndarray::Array<TYPE, 1, 1> const&, \
        pfs::drp::stella::math::SymmetricTridiagonalWorkspace<TYPE> & \
    );

INSTANTIATE(double);