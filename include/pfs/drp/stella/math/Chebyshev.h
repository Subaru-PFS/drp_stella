# include <cstring>
# include <iomanip>
# include <iostream>
# include <stdio.h>

#include "ndarray.h"
#include "ndarray/eigen.h"
#include "lsst/pex/exceptions/Exception.h"
#include "pfs/drp/stella/math/Math.h"

using namespace std;
namespace pexExcept = lsst::pex::exceptions;

namespace pfs{ namespace drp{ namespace stella{ namespace math{
    void daxpy ( int n, double da, double dx[], int incx, double dy[], int incy );
    double ddot ( int n, double dx[], int incx, double dy[], int incy );
    double dnrm2 ( int n, double x[], int incx );
    void drot ( int n, double x[], int incx, double y[], int incy, double c,
      double s );
    void drotg ( double *sa, double *sb, double *c, double *s );
    void dscal ( int n, double sa, double x[], int incx );
    int dsvdc ( double a[], int lda, int m, int n, double s[], double e[], 
      double u[], int ldu, double v[], int ldv, double work[], int job );
    void dswap ( int n, double x[], int incx, double y[], int incy );
    int i4_max ( int i1, int i2 );
    int i4_min ( int i1, int i2 );
    string i4_to_string ( int i4, string format );
    int i4_uniform ( int a, int b, int &seed );
    void imtqlx ( int n, double d[], double e[], double z[] );
    int r4_nint ( float x );
    double r8_choose ( int n, int k );
    double r8_epsilon ( );
    double r8_max ( double x, double y );
    double r8_sign ( double x );
    double *r8mat_copy_new ( int m, int n, double a1[] );
    double *r8mat_mtv_new ( int m, int n, double a[], double x[] );
    double *r8mat_mv_new ( int m, int n, double a[], double x[] );
    double r8vec_dot_product ( int n, double a1[], double a2[] );
    bool r8vec_in_ab ( int n, double x[], double a, double b );
    double *r8vec_linspace_new ( int n, double a_first, double a_last );
    double r8vec_max ( int n, double r8vec[] );
    void r8vec_print ( int n, double a[], string title );
    double *r8vec_uniform_new ( int n, double b, double c, int *seed );
    double *r8vec_uniform_01_new ( int n, int *seed );
    void r8vec2_print ( int n, double a1[], double a2[], string title );
    double *svd_solve ( int m, int n, double a[], double b[] );
    double t_double_product_integral ( int i, int j );
    double t_integral ( int e );
    double *t_polynomial ( int m, int n, double x[] );
    ndarray::Array<double, 2, 1> t_polynomial ( ndarray::Array<double, 1, 1> const& x, int n );
    double *t_polynomial_ab ( double a, double b, int m, int n, double x[] );
    ndarray::Array<double, 2, 1> t_polynomial_ab ( ndarray::Array<double, 1, 1> const& x, int n, double a, double b );
    double *t_polynomial_coefficients ( int n );
    double t_polynomial_value ( int n, double x );
    void t_polynomial_values ( int &n_data, int &n, double &x, double &fx );
    double *t_polynomial_zeros ( int n );
    double *t_project_coefficients ( int n, double f ( double x ) );
    double *t_project_coefficients_ab ( int n, double f ( double x ), double a, 
      double b );
    double *t_project_coefficients_data ( double a, double b, int m, int n, 
      double x[], double d[] );
    ndarray::Array<double, 1, 1> t_project_coefficients_data ( ndarray::Array<double, 1, 1> const& x, 
                                                               ndarray::Array<double, 1, 1> const& d, 
                                                               double a, 
                                                               double b, 
                                                               int n);
    double *t_project_value ( int m, int n, double x[], double c[] );
    double *t_project_value_ab ( int m, int n, double x[], double c[], double a, 
      double b );
    void t_quadrature_rule ( int n, double t[], double w[] );
    double t_triple_product_integral ( int i, int j, int k );
    void timestamp ( );
    double u_double_product_integral ( int i, int j );
    double u_integral ( int e );
    double *u_polynomial ( int m, int n, double x[] );
    double *u_polynomial_coefficients ( int n );
    void u_polynomial_values ( int &n_data, int &n, double &x, double &fx );
    double *u_polynomial_zeros ( int n );
    void u_quadrature_rule ( int n, double t[], double w[] );
    double v_double_product_integral ( int i, int j );
    double *v_polynomial ( int m, int n, double x[] );
    void v_polynomial_values ( int &n_data, int &n, double &x, double &fx );
    double *v_polynomial_zeros ( int n );
    double w_double_product_integral ( int i, int j );
    double *w_polynomial ( int m, int n, double x[] );
    void w_polynomial_values ( int &n_data, int &n, double &x, double &fx );
    double *w_polynomial_zeros ( int n );
}}}}
