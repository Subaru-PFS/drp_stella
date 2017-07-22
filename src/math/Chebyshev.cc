# include <cstring>
# include <iomanip>
# include <iostream>
# include <stdio.h>

#include "lsst/pex/exceptions/Exception.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/math/Chebyshev.h"

namespace pfs { namespace drp { namespace stella { namespace math {
namespace {

  //****************************************************************************
  //
  //  Purpose:
  //
  //    T_POLYNOMIAL evaluates Chebyshev polynomials T(n,x).
  //
  //  Discussion:
  //
  //    Chebyshev polynomials are useful as a basis for representing the
  //    approximation of functions since they are well conditioned, in the sense
  //    that in the interval [-1,1] they each have maximum absolute value 1.
  //    Hence an error in the value of a coefficient of the approximation, of
  //    size epsilon, is exactly reflected in an error of size epsilon between
  //    the computed approximation and the theoretical approximation.
  //
  //    Typical usage is as follows, where we assume for the moment
  //    that the interval of approximation is [-1,1].  The value
  //    of N is chosen, the highest polynomial to be used in the
  //    approximation.  Then the function to be approximated is
  //    evaluated at the N+1 points XJ which are the zeroes of the N+1-th
  //    Chebyshev polynomial.  Let these values be denoted by F(XJ).
  //
  //    The coefficients of the approximation are now defined by
  //
  //      C(I) = 2/(N+1) * sum ( 1 <= J <= N+1 ) F(XJ) T(I,XJ)
  //
  //    except that C(0) is given a value which is half that assigned
  //    to it by the above formula,
  //
  //    and the representation is
  //
  //    F(X) approximated by sum ( 0 <= J <= N ) C(J) T(J,X)
  //
  //    Now note that, again because of the fact that the Chebyshev polynomials
  //    have maximum absolute value 1, if the higher order terms of the
  //    coefficients C are small, then we have the option of truncating
  //    the approximation by dropping these terms, and we will have an
  //    exact value for maximum perturbation to the approximation that
  //    this will cause.
  //
  //    It should be noted that typically the error in approximation
  //    is dominated by the first neglected basis function (some multiple of
  //    T(N+1,X) in the example above).  If this term were the exact error,
  //    then we would have found the minimax polynomial, the approximating
  //    polynomial of smallest maximum deviation from the original function.
  //    The minimax polynomial is hard to compute, and another important
  //    feature of the Chebyshev approximation is that it tends to behave
  //    like the minimax polynomial while being easy to compute.
  //
  //    To evaluate a sum like
  //
  //      sum ( 0 <= J <= N ) C(J) T(J,X),
  //
  //    Clenshaw's recurrence formula is recommended instead of computing the
  //    polynomial values, forming the products and summing.
  //
  //    Assuming that the coefficients C(J) have been computed
  //    for J = 0 to N, then the coefficients of the representation of the
  //    indefinite integral of the function may be computed by
  //
  //      B(I) = ( C(I-1) - C(I+1))/2*(I-1) for I=1 to N+1,
  //
  //    with
  //
  //      C(N+1)=0
  //      B(0) arbitrary.
  //
  //    Also, the coefficients of the representation of the derivative of the
  //    function may be computed by:
  //
  //      D(I) = D(I+2)+2*I*C(I) for I=N-1, N-2, ..., 0,
  //
  //    with
  //
  //      D(N+1) = D(N)=0.
  //
  //    Some of the above may have to adjusted because of the irregularity of C(0).
  //
  //    The formula is:
  //
  //      T(N,X) = COS(N*ARCCOS(X))
  //
  //  Differential equation:
  //
  //    (1-X*X) Y'' - X Y' + N N Y = 0
  //
  //  First terms:
  //
  //    T(0,X) =  1
  //    T(1,X) =  1 X
  //    T(2,X) =  2 X^2 -   1
  //    T(3,X) =  4 X^3 -   3 X
  //    T(4,X) =  8 X^4 -   8 X^2 +  1
  //    T(5,X) = 16 X^5 -  20 X^3 +  5 X
  //    T(6,X) = 32 X^6 -  48 X^4 + 18 X^2 - 1
  //    T(7,X) = 64 X^7 - 112 X^5 + 56 X^3 - 7 X
  //
  //  Inequality:
  //
  //    abs ( T(N,X) ) <= 1 for -1 <= X <= 1
  //
  //  Orthogonality:
  //
  //    For integration over [-1,1] with weight
  //
  //      W(X) = 1 / sqrt(1-X*X),
  //
  //    if we write the inner product of T(I,X) and T(J,X) as
  //
  //      < T(I,X), T(J,X) > = integral ( -1 <= X <= 1 ) W(X) T(I,X) T(J,X) dX
  //
  //    then the result is:
  //
  //      0 if I /= J
  //      PI/2 if I == J /= 0
  //      PI if I == J == 0
  //
  //    A discrete orthogonality relation is also satisfied at each of
  //    the N zeroes of T(N,X):  sum ( 1 <= K <= N ) T(I,X) * T(J,X)
  //                              = 0 if I /= J
  //                              = N/2 if I == J /= 0
  //                              = N if I == J == 0
  //
  //  Recursion:
  //
  //    T(0,X) = 1,
  //    T(1,X) = X,
  //    T(N,X) = 2 * X * T(N-1,X) - T(N-2,X)
  //
  //    T'(N,X) = N * ( -X * T(N,X) + T(N-1,X) ) / ( 1 - X^2 )
  //
  //  Special values:
  //
  //    T(N,1) = 1
  //    T(N,-1) = (-1)^N
  //    T(2N,0) = (-1)^N
  //    T(2N+1,0) = 0
  //    T(N,X) = (-1)**N * T(N,-X)
  //
  //  Zeroes:
  //
  //    M-th zero of T(N,X) is cos((2*M-1)*PI/(2*N)), M = 1 to N
  //
  //  Extrema:
  //
  //    M-th extremum of T(N,X) is cos(PI*M/N), M = 0 to N
  //
  //  Licensing:
  //
  //    This code is distributed under the GNU LGPL license.
  //
  //  Modified:
  //
  //    12 May 2003
  //
  //  Author:
  //
  //    John Burkardt
  //
  //  Parameters:
  //
  //    Input, int M, the number of evaluation points.
  //
  //    Input, int N, the highest polynomial to compute.
  //
  //    Input, float X[M], the evaluation points.
  //
  //    Output, float T_POLYNOMIAL[M*(N+1)], the values of the Chebyshev polynomials.
  //
  ndarray::Array<float, 2, 1>
  t_polynomial(ndarray::Array<float, 1, 1> const& x, int n)
  {
    int m = x.getShape()[0];
    int i;
    int j;

    if ( n < 0 )
    {
      throw LSST_EXCEPT(lsst::pex::exceptions::Exception, "t_polynomial: ERROR: n < 0");
    }

    ndarray::Array<float, 2, 1> v = ndarray::allocate(m, n+1);

    for ( auto itRow = v.begin(); itRow != v.end(); ++itRow )
    {
      for (auto itCol = itRow->begin(); itCol != itRow->end(); ++itCol)
      *itCol = 1.0;
    }
    if ( n < 1 )
    {
      return v;
    }

    for (  i = 0; i < m; i++ )
    {
      v[ ndarray::makeVector( i, 1 ) ] = x[i];
    }

    for ( j = 2; j <= n; j++ )
    {
      for ( i = 0; i < m; i++ )
      {
        v[ ndarray::makeVector( i, j ) ] = 2.0 * x[i] * v[ ndarray::makeVector( i, j-1 ) ] - v[ ndarray::makeVector( i, j-2 ) ];
      }
    }

    return v;
  }

  //****************************************************************************
  //
  //  Purpose:
  //
  //    T_POLYNOMIAL_AB: Chebyshev polynomials T(n,x) in [A,B].
  //
  //  Licensing:
  //
  //    This code is distributed under the GNU LGPL license.
  //
  //  Modified:
  //
  //    17 April 2012
  //
  //  Author:
  //
  //    John Burkardt
  //
  //  Parameters:
  //
  //    Input, float A, B, the domain of definition.
  //
  //    Input, int M, the number of evaluation points.
  //
  //    Input, int N, the highest polynomial to compute.
  //
  //    Input, float X[M], the evaluation points.
  //    It must be the case that A <= X(*) <= B.
  //
  //    Output, float T_POLYNOMIAL_AB[M*(N+1)], the values.
  //
  ndarray::Array<float, 2, 1>
  t_polynomial_ab(ndarray::Array<float, 1, 1> const& x, int n, float a, float b)
  {
    int m = x.getShape()[0];
    ndarray::Array<float, 2, 1> v;
    ndarray::Array<float, 1, 1> y = ndarray::allocate(m);

    for ( auto itX = x.begin(), itY = y.begin(); itX < x.end(); ++itX, ++itY )
    {
      *itY = ( ( b - *itX     )
             - (     *itX - a ) )
             / ( b        - a );
    }

    v = t_polynomial ( x, n );

    return v;
  }

  //****************************************************************************
  //
  //  Purpose:
  //
  //    T_PROJECT_COEFFICIENTS: function projected onto Chebyshev polynomials T(n,x).
  //
  //  Discussion:
  //
  //    It is assumed that the interval of definition is -1 <= x <= +1.
  //
  //    Over this interval, f(x) will be well approximated by
  //
  //      f(x) approx sum ( 0 <= i <= n ) c(i) * T(i,x)
  //
  //  Licensing:
  //
  //    This code is distributed under the GNU LGPL license.
  //
  //  Modified:
  //
  //    17 April 2012
  //
  //  Author:
  //
  //    John Burkardt
  //
  //  Parameters:
  //
  //    Input, int N, the highest order polynomial to compute.
  //
  //    Input, double F ( double X ), evaluates the function.
  //
  //    Output, double T_PROJECT_COEFFICIENTS[N+1], the projection coefficients
  //    of f(x) onto T(0,x) through T(n,x).
  //

  double *
  t_project_coefficients (int n, double f ( double x ))
  {
    double *c;
    double *d;
    double fac;
    int j;
    int k;
    double pi = 3.141592653589793;
    double total;
    double y;

    d = new double[n+1];

    for ( k = 0; k <= n; k++ )
    {
      y = cos ( pi * ( ( double ) ( k ) + 0.5 ) / ( double ) ( n + 1 ) );
      d[k] = f ( y );
    }

    fac = 2.0 / ( double ) ( n + 1 );

    c = new double[n+1];

    for ( j = 0; j <= n; j++ )
    {
      total = 0.0;
      for ( k = 0; k <= n; k++ )
      {
        total = total + d[k] * cos ( ( pi * ( double ) ( j ) )
          * ( ( ( double ) ( k ) + 0.5 ) / ( double ) ( n + 1 ) ) );
      }
      c[j] = fac * total;
    }

    c[0] = c[0] / 2.0;

    delete [] d;

    return c;
  }
}  
                
  //****************************************************************************
  //
  //  Purpose:
  //
  //    T_PROJECT_COEFFICIENTS_DATA: project data onto Chebyshev polynomials T(n,x).
  //
  //  Licensing:
  //
  //    This code is distributed under the GNU LGPL license.
  //
  //  Modified:
  //
  //    22 April 2012
  //
  //  Author:
  //
  //    John Burkardt
  //
  //  Parameters:
  //
  //    Input, float A, B, the domain of definition (both of size M)
  //
  //    Input, int N, the desired order of the Chebyshev
  //    expansion.
  //
  //    Input, float X[M], the data abscissas.  These need not
  //    be sorted.  It must be the case that A <= X() <= B.
  //
  //    Input, float D[M], the data values.
  //
  //    Output, float T_PROJECT_COEFFICIENTS_DATA[N+1], the approximate
  //    Chebshev coefficients.
  //
  ndarray::Array<float, 1, 1>
  t_project_coefficients_data(ndarray::Array<float, 1, 1> const& x,
                              ndarray::Array<float, 1, 1> const& d,
                              float a,
                              float b,
                              int n)
  {
    ndarray::Array<float, 1, 1> c = ndarray::allocate(n);
    ndarray::Array<float, 2, 1> v;
    ndarray::Array<float, 1, 1> range = ndarray::allocate(2);
    range[0] = a;
    range[1] = b;

    if(x.getShape() != d.getShape()) {
      throw LSST_EXCEPT(lsst::pex::exceptions::Exception, "x and d must have same dimension");
    }

    if(!pfs::drp::stella::math::checkIfValuesAreInRange(x, range)) {
      throw LSST_EXCEPT(lsst::pex::exceptions::Exception,
                       "T_PROJECT_COEFFICIENTS_DATA- Fatal error!: Some X not in [A,B].");
    }
    //
    //  Compute the M by N+1 Chebyshev Vandermonde matrix V.
    //
    //cout << "Compute the M by N+1 Chebyshev Vandermonde matrix V." << endl;
    v = t_polynomial_ab(x, n, a, b);
    //cout << "Compute the M by N+1 Chebyshev Vandermonde matrix V finished. v.getShape() = " << v.getShape() << endl;
    //
    //  Compute the least-squares solution C.
    //
    //cout << "Compute the least-squares solution C. d.getShape() = " << d.getShape() << endl;
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(v.asEigen(), Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto cEigen = svd.solve(d.asEigen());
    c.asEigen() = cEigen;
    //cout << "Compute the least-squares solution C finished. cEigen = " << cEigen << endl;
    //cout << "Compute the least-squares solution C finished. c = " << c << endl;

    return c;
  }
                
}}}}
