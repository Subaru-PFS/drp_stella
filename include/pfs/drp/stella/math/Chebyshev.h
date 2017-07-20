#ifndef PFS_DRP_STELLA_MATH_CHEBYSHEV_H
#define PFS_DRP_STELLA_MATH_CHEBYSHEV_H

namespace ndarray {
    template<typename T, int I, int J> class Array;
}

namespace pfs { namespace drp { namespace stella { namespace math {
  /**
   * \brief project data onto Chebyshev polynomials T(n,x).
   *
   *  Licensing:
   *
   *    This code is distributed under the GNU LGPL license.
   *
   *  Author:
   *
   *    John Burkardt
   *
   *  Parameters:
   *
   *  \param x, the data abscissas.  These need not
   *    be sorted.  It must be the case that A <= X() <= B.
   *  \param d the data values (same dimension as x)
   *  \param a Lower edge of the domain of definition.
   *  \param b Upper edge of the domain of definition.
   *  \param n the desired order of the Chebyshev expansion.
   *
   * \returns the approximate Chebyshev coefficients (n + 1 values)
   */
    ndarray::Array<double, 1, 1> t_project_coefficients_data(ndarray::Array<double, 1, 1> const& x, 
                                                             ndarray::Array<double, 1, 1> const& d, 
                                                             double a, 
                                                             double b, 
                                                             int n);                
}}}}
#endif
