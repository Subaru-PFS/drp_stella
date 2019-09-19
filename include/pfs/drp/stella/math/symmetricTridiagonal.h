#include <tuple>
#include "ndarray_fwd.h"

namespace pfs {
namespace drp {
namespace stella {
namespace math {

/** Workspace for solveSymmetricTridiagonal and invertSymmetricTridiagonal
 *
 * This user-opaque object is simply a cache of arrays to limit the number of
 * allocations that are necessary with repeated calls to
 * solveSymmetricTridiagonal and/or invertSymmetricTridiagonal (the same struct
 * serves for both, but since the result arrays are held by the workspace,
 * beware of reusing the workspace when also holding the result of a previous
 * calculation).
 */
template <typename T>
struct SymmetricTridiagonalWorkspace {
    ndarray::Array<T, 1, 1> longArray1;  // Array of length N
    ndarray::Array<T, 1, 1> longArray2;  // Array of length N
    ndarray::Array<T, 1, 1> shortArray;  // Array of length N-1
    std::size_t _num;  // Number of elements

    explicit SymmetricTridiagonalWorkspace() : _num(0) {}

    // Reset for the next calculation
    void reset(std::size_t num) {
        if (num == _num) return;
        longArray1 = ndarray::allocate(num);
        longArray2 = ndarray::allocate(num);
        shortArray = ndarray::allocate(num - 1);
        _num = num;
    }
};


/** Solve a matrix equation involving a symmetric tridiagonal matrix
 *
 * We use the following notation:
 * We are solving the matrix equation, Mx = d
 * where x is the vector of unknown values, d is a known vector, and the matrix
 * The matrix, M, is symmetric and tridiagonal, with entries like:
 * M = [[b1, c1, 0, 0],
 *      [c1, b2, c2, 0],
 *      [0, c2, b3, c3],
 *      [0, 0, c3, b4]]
 *
 * @param diagonal : an array of length N, containing the diagonal elements of
 *        the matrix; the b[i] in the above notation.
 * @param offDiag : an array of length N-1, containing the off-diagonal elements
          of the matrix; the c[i] in the above notation.
 * @param answer : an array of length N, containing the right-hand-side vector;
 *        the d[i] in the above notation.
 * @param workspace : an opaque object of no relevance to the user; this allows
 *        us to limit the memory allocations by repeated invocations of this
 *        function.
 * @return an array of length N, containing the solution to the equation;
 *         the x[i] in the above notation.
 */
template <typename T>
ndarray::Array<T, 1, 1>
solveSymmetricTridiagonal(
    ndarray::Array<T, 1, 1> const& diagonal,
    ndarray::Array<T, 1, 1> const& offDiag,
    ndarray::Array<T, 1, 1> const& answer,
    SymmetricTridiagonalWorkspace<T> & workspace=SymmetricTridiagonalWorkspace<T>()
);


/** Partially invert a symmetric tridiagonal matrix
 *
 * We use the following notation:
 * The matrix, M, is symmetric tridiagonal, with entries like:
 * M = [[b1, c1, 0, 0],
 *      [c1, b2, c2, 0],
 *      [0, c2, b3, c3],
 *      [0, 0, c3, b4]]
 *
 * We calculate and return only the diagonal and off-diagonal elements of the
 * matrix inverse, M^-1.
 *
 * @param diagonal : an array of length N, containing the diagonal elements of
 *        the matrix; the b[i] in the above notation.
 * @param offDiag : an array of length N-1, containing the off-diagonal elements
          of the matrix; the c[i] in the above notation.
 * @param workspace : an opaque object of no relevance to the user; this allows
 *        us to limit the memory allocations by repeated invocations of this
 *        function.
 * @return a tuple containing two arrays: the first is an array of length N,
 *         containing the diagonal elements of the matrix inverse (the
 *         M^-1[i,i]); the second is an array of length N-1, containing the
 *         off-diagonal elements of the matrix inverse (the M^-1[i,i+1]).
 */
template <typename T>
std::tuple<ndarray::Array<T, 1, 1>, ndarray::Array<T, 1, 1>>
invertSymmetricTridiagonal(
    ndarray::Array<T, 1, 1> const& diagonal,
    ndarray::Array<T, 1, 1> const& offDiag,
    SymmetricTridiagonalWorkspace<T> & workspace=SymmetricTridiagonalWorkspace<T>()
);


}}}} // namespace pfs::drp::stella::math
