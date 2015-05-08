/*
 * MINPACK-1 Least Squares Fitting Library
 *
 * Test routines
 *
 * These test routines provide examples for users to familiarize
 * themselves with the mpfit library.  They also provide a baseline
 * test data set for users to be sure that the library is functioning
 * properly on their platform.
 *
 * By default, testmpfit is built by the distribution Makefile.
 *
 * To test the function of the mpfit library,
 *   1. Build testmpfit   ("make testmpfit")
 *   2. Run testmpfit     ("./mpfit")
 *   3. Compare results of your run with the distributed file testmpfit.log
 *
 * This file contains several test user functions:
 *   1. linfunc() linear fit function, y = f(x) = a - b*x
 *      - Driver is testlinfit()
 *   2. quadfunc() quadratic polynomial function, y = f(x) = a + b*x + c*x^2
 *      - Driver is testquadfit() - all parameters free
 *      - Driver is testquadfix() - linear parameter fixed
 *   3. gaussfunc() gaussian peak
 *      - Driver is testgaussfit() - all parameters free
 *      - Driver is testgaussfix() - constant & centroid fixed
 *           (this routine demonstrates in comments how to impose parameter limits)
 *   4. main() routine calls all five driver functions
 *
 * Copyright (C) 2003,2006,2009,2010, Craig Markwardt
 *
 */

/* Test routines for mpfit library
   $Id: testmpfit.c,v 1.6 2010/11/13 08:18:02 craigm Exp $
*/

#ifndef __MYFIT_H__
#define __MYFIT_H__

#include "mpfit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ndarray.h"
#include "lsst/base.h"
#include "lsst/pex/config.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/exceptions/Exception.h"
#include "pfs/drp/stella/math/Math.h"

using namespace std;

#define D_PI 3.14159265359

#define __DEBUG_GAUSSFITLIM__

/* This is the private data structure which contains the data points
   and their uncertainties */
struct vars_struct {
  double *x;
  double *y;
  double *ey;
};

//template<typename T>
//typedef struct result {
//    ndarray::Array<
//};

/* Simple routine to print the fit results */
void PrintResult(double *x, mp_result *result);
//void PrintResult(double *x, double *xact, mp_result *result);

/*
 * linear fit function
 *
 * m - number of data points
 * n - number of parameters (2)
 * p - array of fit parameters
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 *
int linfunc(int m, int n, double *p, double *dy, double **dvec, void *vars);

* Test harness routine, which contains test data, invokes mpfit() *
int linfit(Array<double, 1> &D_A1_X, Array);

*
 * quadratic fit function
 *
 * m - number of data points
 * n - number of parameters (2)
 * p - array of fit parameters
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 *
int quadfunc(int m, int n, double *p, double *dy, double **dvec, void *vars);

* Test harness routine, which contains test quadratic data, invokes
   mpfit() *
int testquadfit();

* Test harness routine, which contains test quadratic data;

   Example of how to fix a parameter
*
int testquadfix();
*/
/*
 * gaussian fit function plus constant background (CB)
 *
 * m - number of data points
 * n - number of parameters (4)
 * p - array of fit parameters
 *     p[0] = peak y value
 *     p[1] = x centroid position
 *     p[2] = gaussian sigma width
 *     p[3] = constant offset
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitGaussFuncCB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * gaussian fit function without background (NB)
 *
 * m - number of data points
 * n - number of parameters (3)
 * p - array of fit parameters
 *     p[0] = peak y value
 *     p[1] = x centroid position
 *     p[2] = gaussian sigma width
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitGaussFuncNB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * gaussian fit function area with constant background (ACB)
 *
 * m - number of data points
 * n - number of parameters (4)
 * p - array of fit parameters
 *     p[0] = constant offset
 *     p[1] = area under curve value
 *     p[2] = x centroid position
 *     p[3] = gaussian sigma width
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitGaussFuncACB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * gaussian fit function area no background (ANB)
 *
 * m - number of data points
 * n - number of parameters (3)
 * p - array of fit parameters
 *     p[0] = area under curve value
 *     p[1] = x centroid position
 *     p[2] = gaussian sigma width
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitGaussFuncANB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * gaussian fit function plus constant and linear background (LB)
 *
 * m - number of data points
 * n - number of parameters (5)
 * p - array of fit parameters
 *     p[0] = peak y value
 *     p[1] = x centroid position
 *     p[2] = gaussian sigma width
 *     p[3] = constant offset
 *     p[4] = linear offset
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitGaussFuncLB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * gaussian fit function area plus constant and linear background
 *
 * m - number of data points
 * n - number of parameters (5)
 * p - array of fit parameters
 *     p[0] = area under curve value
 *     p[1] = x centroid position
 *     p[2] = gaussian sigma width
 *     p[3] = constant offset
 *     p[4] = linear offset
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitGaussFuncALB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * gaussian fit function 2 Gaussians plus constant background (CB)
 *
 * m - number of data points
 * n - number of parameters (6)
 * p - array of fit parameters
 *     p[0] = constant offset
 *     p[1] = peak y value 1st Gauss
 *     p[2] = x centroid position 1st Gauss
 *     p[3] = gaussian sigma width
 *     p[4] = peak y value 2nd Gauss
 *     p[5] = x centroid position 2nd Gauss
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitTwoGaussFuncCB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * gaussian fit function 2 Gaussians no background (NB)
 *
 * m - number of data points
 * n - number of parameters (5)
 * p - array of fit parameters
 *     p[0] = peak y value 1st Gauss
 *     p[1] = x centroid position 1st Gauss
 *     p[2] = gaussian sigma width
 *     p[3] = peak y value 2nd Gauss
 *     p[4] = x centroid position 2nd Gauss
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitTwoGaussFuncNB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * gaussian fit function 2 Gaussians area plus constant background (ACB)
 *
 * m - number of data points
 * n - number of parameters (6)
 * p - array of fit parameters
 *     p[0] = constant offset
 *     p[1] = area under curve value 1st Gauss
 *     p[2] = x centroid position 1st Gauss
 *     p[3] = gaussian sigma width
 *     p[4] = area under curve value 2nd Gauss
 *     p[5] = x centroid position 2nd Gauss
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitTwoGaussFuncACB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * gaussian fit function 2 Gaussians area no background (ANB)
 *
 * m - number of data points
 * n - number of parameters (5)
 * p - array of fit parameters
 *     p[0] = area under curve value 1st Gauss
 *     p[1] = x centroid position 1st Gauss
 *     p[2] = gaussian sigma width
 *     p[3] = area under curve value 2nd Gauss
 *     p[4] = x centroid position 2nd Gauss
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitTwoGaussFuncANB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * gaussian fit function 3 Gaussians plus constant background
 *
 * m - number of data points
 * n - number of parameters (8)
 * p - array of fit parameters
 *     p[0] = constant offset
 *     p[1] = peak y value 1st Gauss
 *     p[2] = x centroid position 1st Gauss
 *     p[3] = gaussian sigma width
 *     p[4] = peak y value 2nd Gauss
 *     p[5] = x centroid position 2nd Gauss
 *     p[6] = peak y value 3rd Gauss
 *     p[7] = x centroid position 3rd Gauss
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitThreeGaussFuncCB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * gaussian fit function 2D Gaussian plus constant background
 *
 * m - number of data points
 * n - number of parameters (5)
 * p - array of fit parameters
 *     p[0] = constant offset
 *     p[1] = peak z value Gauss
 *     p[2] = x centroid position Gauss
 *     p[3] = y centroid position Gauss
 *     p[4] = gaussian sigma width
 * dz - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFit2DGaussFuncCB(int m, int n, double *p, double *dz, double **dvec, void *vars);

/*
 * gaussian fit function 3 Gaussians no background (NB)
 *
 * m - number of data points
 * n - number of parameters (7)
 * p - array of fit parameters
 *     p[0] = peak y value 1st Gauss
 *     p[1] = x centroid position 1st Gauss
 *     p[2] = gaussian sigma width 1st Gauss
 *     p[3] = peak y value 2nd Gauss
 *     p[4] = x centroid position 2nd Gauss
 *     p[5] = peak y value 3rd Gauss
 *     p[6] = x centroid position 3rd Gauss
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitThreeGaussFuncNB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * gaussian fit function 3 Gaussians area plus constant background
 *
 * m - number of data points
 * n - number of parameters (8)
 * p - array of fit parameters
 *     p[0] = constant offset
 *     p[1] = area under curve value 1st Gauss
 *     p[2] = x centroid position 1st Gauss
 *     p[3] = gaussian sigma width 1st Gauss
 *     p[4] = area under curve value 2nd Gauss
 *     p[5] = x centroid position 2nd Gauss
 *     p[6] = area under curve value 3rd Gauss
 *     p[7] = x centroid position 3rd Gauss
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitThreeGaussFuncACB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * gaussian fit function 3 Gaussians area no background
 *
 * m - number of data points
 * n - number of parameters (7)
 * p - array of fit parameters
 *     p[0] = area under curve value 1st Gauss
 *     p[1] = x centroid position 1st Gauss
 *     p[2] = gaussian sigma width 1st Gauss
 *     p[3] = area under curve value 2nd Gauss
 *     p[4] = x centroid position 2nd Gauss
 *     p[5] = area under curve value 3rd Gauss
 *     p[6] = x centroid position 3rd Gauss
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitThreeGaussFuncANB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * Chebyshev polynomial 1st kind
 * m - number of data points
 * n - number of parameters (order + 1)
 * p - array of fitting parameters
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 * 
 * RETURNS: error code (0 = success)
 */
int Chebyshev1stKind(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * Fit a Chebyshev polynomial of the 1st kind to x and y. X will be shifted and rescaled to fit into the range [-1, 1].
 * The order of the polynomial will be determined from the length of D_A1_Coeffs_Out (must be equal to length of D_A1_ECoeffs_Out).
 * 
 */
template<typename T> 
bool MPFitChebyshev1stKind(const ndarray::Array<T, 1, 1> & D_A1_X_In,
                           const ndarray::Array<T, 1, 1> & D_A1_Y_In,
                           const ndarray::Array<T, 1, 1> & D_A1_EY_In,
                           ndarray::Array<T, 1, 1> & D_A1_Coeffs_Out,
                           ndarray::Array<T, 1, 1> & D_A1_ECoeffs_Out);
template<typename T> 
bool MPFitChebyshev1stKind(const ndarray::Array<T, 1, 1> & D_A1_X_In,
                           const ndarray::Array<T, 1, 1> & D_A1_Y_In,
                           const ndarray::Array<T, 1, 1> & D_A1_EY_In,
                           const ndarray::Array<T, 1, 1> & D_A1_Guess_In,
                           ndarray::Array<T, 1, 1> & D_A1_Coeffs_Out,
                           ndarray::Array<T, 1, 1> & D_A1_ECoeffs_Out);

/* Test harness routine, which contains test gaussian-peak data */
template<typename T>
bool MPFitGauss(const ndarray::Array<T, 1, 1> &D_A1_X_In,
                const ndarray::Array<T, 1, 1> &D_A1_Y_In,
                const ndarray::Array<T, 1, 1> &D_A1_EY_In,
                const ndarray::Array<T, 1, 1> &D_A1_Guess_In,
                const bool B_WithConstantBackground,
                const bool B_FitArea,
                ndarray::Array<T, 1, 1> &D_A1_Coeffs_Out,
                ndarray::Array< T, 1, 1>& D_A1_ECoeffs_Out);

template<typename T>
bool MPFitGaussFix(const ndarray::Array<T, 1, 1> &D_A1_X_In,
                   const ndarray::Array<T, 1, 1> &D_A1_Y_In,
                   const ndarray::Array<T, 1, 1> &D_A1_EY_In,
                   const ndarray::Array<T, 1, 1> &D_A1_Guess_In,
                   const ndarray::Array<int, 1, 1> &I_A1_Fix,
                   const bool B_WithConstantBackground,
                   const bool B_FitArea,
                   ndarray::Array<T, 1, 1> &D_A1_Coeffs_Out,
                   ndarray::Array<T, 1, 1> & D_A1_ECoeffs_Out);

template<typename T>
bool MPFitGaussLim(const ndarray::Array<T, 1, 1> &D_A1_X_In,
                   const ndarray::Array<T, 1, 1> &D_A1_Y_In,
                   const ndarray::Array<T, 1, 1> &D_A1_EY_In,
                   const ndarray::Array<T, 1, 1> &D_A1_Guess_In,
                   const ndarray::Array<int, 2, 1> &I_A2_Limited,
                   const ndarray::Array<T, 2, 1> &D_A2_Limits,
                   const int Background,///0-none, 1-constant, 2-linear
                   const bool B_FitArea,
                   ndarray::Array<T, 1, 1> &D_A1_Coeffs_Out,
                   ndarray::Array< T, 1, 1>& D_A1_ECoeffs_Out,
                   bool Debug = false);

/* Test harness routine, which contains test gaussian-peak data */
template<typename T>
bool MPFitTwoGauss(const ndarray::Array<T, 1, 1> &D_A1_X_In,
                   const ndarray::Array<T, 1, 1> &D_A1_Y_In,
                   const ndarray::Array<T, 1, 1> &D_A1_EY_In,
                   const ndarray::Array<T, 1, 1> &D_A1_Guess_In,
                   const bool B_WithConstantBackground,
                   const bool B_FitArea,
                   ndarray::Array<T, 1, 1> &D_A1_Coeffs_Out,
                   ndarray::Array< T, 1, 1>& D_A1_ECoeffs_Out);

template<typename T>
bool MPFitTwoGaussFix(const ndarray::Array<T, 1, 1> &D_A1_X_In,
                      const ndarray::Array<T, 1, 1> &D_A1_Y_In,
                      const ndarray::Array<T, 1, 1> &D_A1_EY_In,
                      const ndarray::Array<T, 1, 1> &D_A1_Guess_In,
                      const ndarray::Array< int, 1, 1 > &I_A1_Fix,
                      const bool B_WithConstantBackground,
                      const bool B_FitArea,
                      ndarray::Array<T, 1, 1> &D_A1_Coeffs_Out,
                      ndarray::Array< T, 1, 1>& D_A1_ECoeffs_Out);

template<typename T>
bool MPFitTwoGaussLim(const ndarray::Array<T, 1, 1> &D_A1_X_In,
                      const ndarray::Array<T, 1, 1> &D_A1_Y_In,
                      const ndarray::Array<T, 1, 1> &D_A1_EY_In,
                      const ndarray::Array<T, 1, 1> &D_A1_Guess_In,
                      const ndarray::Array<int, 2, 1> &I_A2_Limited,
                      const ndarray::Array<T, 2, 1> &D_A2_Limits,
                      const bool B_WithConstantBackground,
                      const bool B_FitArea,
                      ndarray::Array<T, 1, 1> &D_A1_Coeffs_Out,
                      ndarray::Array< T, 1, 1>& D_A1_ECoeffs_Out);

template<typename T>
bool MPFitThreeGauss(const ndarray::Array<T, 1, 1> &D_A1_X_In,
                     const ndarray::Array<T, 1, 1> &D_A1_Y_In,
                     const ndarray::Array<T, 1, 1> &D_A1_EY_In,
                     const ndarray::Array<T, 1, 1> &D_A1_Guess_In,
                     const bool B_WithConstantBackground,
                     const bool B_FitArea,
                     ndarray::Array<T, 1, 1> &D_A1_Coeffs_Out,
                     ndarray::Array< T, 1, 1>& D_A1_ECoeffs_Out);

template<typename T>
bool MPFitThreeGaussFix(const ndarray::Array<T, 1, 1> &D_A1_X_In,
                        const ndarray::Array<T, 1, 1> &D_A1_Y_In,
                        const ndarray::Array<T, 1, 1> &D_A1_EY_In,
                        const ndarray::Array<T, 1, 1> &D_A1_Guess_In,
                        const ndarray::Array< int, 1, 1 > &I_A1_Fix,
                        const bool B_WithConstantBackground,
                        const bool B_FitArea,
                        ndarray::Array<T, 1, 1> &D_A1_Coeffs_Out,
                        ndarray::Array< T, 1, 1>& D_A1_ECoeffs_Out);

template<typename T>
bool MPFitThreeGaussLim(const ndarray::Array<T, 1, 1> &D_A1_X_In,
                        const ndarray::Array<T, 1, 1> &D_A1_Y_In,
                        const ndarray::Array<T, 1, 1> &D_A1_EY_In,
                        const ndarray::Array<T, 1, 1> &D_A1_Guess_In,
                        const ndarray::Array<int, 2, 1> &I_A2_Limited,
                        const ndarray::Array<T, 2, 1> &D_A2_Limits,
                        const bool B_WithConstantBackground,
                        const bool B_FitArea,
                        ndarray::Array<T, 1, 1> &D_A1_Coeffs_Out,
                        ndarray::Array< T, 1, 1>& D_A1_ECoeffs_Out);

template<typename T>
bool MPFit2DGaussLim(const ndarray::Array< T, 1, 1>& D_A1_X_In,
                     const ndarray::Array< T, 1, 1>& D_A1_Y_In,
                     const ndarray::Array< T, 1, 1>& D_A1_Z_In,
                     const ndarray::Array< T, 1, 1>& D_A1_Guess_In,
                     const ndarray::Array<int, 2, 1> &I_A2_Limited,
                     const ndarray::Array<T, 2, 1> &D_A2_Limits,
                     ndarray::Array< T, 1, 1>& D_A1_Coeffs_Out,
                     ndarray::Array< T, 1, 1>& D_A1_ECoeffs_Out);

#endif
