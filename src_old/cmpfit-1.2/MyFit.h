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
#include <blitz/array.h>
#include <blitz/vector.h>
#include <blitz/vector-et.h>
#include <blitz/vecwhere.h>
#include <blitz/tinyvec.h>
#include <blitz/tinyvec-et.h>
#include <mgl2/type.h>

using namespace std;
//using namespace blitz;

#define D_PI 3.14159265359

/* This is the private data structure which contains the data points
   and their uncertainties */
struct vars_struct {
  double *x;
  double *y;
  double *ey;
};

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

/* Test harness routine, which contains test data, invokes mpfit() *
int linfit(Array<double, 1> &D_A1_X, Array);

/*
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

/* Test harness routine, which contains test quadratic data, invokes
   mpfit() *
int testquadfit();

/* Test harness routine, which contains test quadratic data;

   Example of how to fix a parameter
*
int testquadfix();

/*
 * gaussian fit function
 *
 * m - number of data points
 * n - number of parameters (4)
 * p - array of fit parameters
 *     p[0] = constant offset
 *     p[1] = peak y value
 *     p[2] = x centroid position
 *     p[3] = gaussian sigma width
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int MPFitGaussFuncCB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/*
 * gaussian fit function
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
 * gaussian fit function
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
 * gaussian fit function
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
 * gaussian fit function
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
 * gaussian fit function
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
 * gaussian fit function
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
 * gaussian fit function
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
 * gaussian fit function
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
 * gaussian fit function
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
 * gaussian fit function
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
 * gaussian fit function
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
 * gaussian fit function
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

/* Test harness routine, which contains test gaussian-peak data */
bool MPFitGauss(const blitz::Array<double, 1> &D_A1_X_In,
                const blitz::Array<double, 1> &D_A1_Y_In,
                const blitz::Array<double, 1> &D_A1_EY_In,
                const blitz::Array<double, 1> &D_A1_Guess_In,
                const bool B_WithConstantBackground,
                const bool B_FitArea,
                blitz::Array<double, 1> &D_A1_Coeffs_Out,
                blitz::Array< double, 1 >& D_A1_ECoeffs_Out);

bool MPFitGaussFix(const blitz::Array<double, 1> &D_A1_X_In,
                   const blitz::Array<double, 1> &D_A1_Y_In,
                   const blitz::Array<double, 1> &D_A1_EY_In,
                   const blitz::Array<double, 1> &D_A1_Guess_In,
                   const blitz::Array<int, 1> &I_A1_Fix,
                   const bool B_WithConstantBackground,
                   const bool B_FitArea,
                   blitz::Array<double, 1> &D_A1_Coeffs_Out,
                   blitz::Array< double, 1 >& D_A1_ECoeffs_Out);

bool MPFitGaussLim(const blitz::Array<double, 1> &D_A1_X_In,
                   const blitz::Array<double, 1> &D_A1_Y_In,
                   const blitz::Array<double, 1> &D_A1_EY_In,
                   const blitz::Array<double, 1> &D_A1_Guess_In,
                   const blitz::Array<int, 2> &I_A2_Limited,
                   const blitz::Array<double, 2> &D_A2_Limits,
                   const bool B_WithConstantBackground,
                   const bool B_FitArea,
                   blitz::Array<double, 1> &D_A1_Coeffs_Out,
                   blitz::Array< double, 1 >& D_A1_ECoeffs_Out);

/* Test harness routine, which contains test gaussian-peak data */
bool MPFitTwoGauss(const blitz::Array<double, 1> &D_A1_X_In,
                   const blitz::Array<double, 1> &D_A1_Y_In,
                   const blitz::Array<double, 1> &D_A1_EY_In,
                   const blitz::Array<double, 1> &D_A1_Guess_In,
                   const bool B_WithConstantBackground,
                   const bool B_FitArea,
                   blitz::Array<double, 1> &D_A1_Coeffs_Out,
                   blitz::Array< double, 1 >& D_A1_ECoeffs_Out);

bool MPFitTwoGaussFix(const blitz::Array<double, 1> &D_A1_X_In,
                      const blitz::Array<double, 1> &D_A1_Y_In,
                      const blitz::Array<double, 1> &D_A1_EY_In,
                      const blitz::Array<double, 1> &D_A1_Guess_In,
                      const blitz::Array<int, 1> &I_A1_Fix,
                      const bool B_WithConstantBackground,
                      const bool B_FitArea,
                      blitz::Array<double, 1> &D_A1_Coeffs_Out,
                      blitz::Array< double, 1 >& D_A1_ECoeffs_Out);

bool MPFitTwoGaussLim(const blitz::Array<double, 1> &D_A1_X_In,
                      const blitz::Array<double, 1> &D_A1_Y_In,
                      const blitz::Array<double, 1> &D_A1_EY_In,
                      const blitz::Array<double, 1> &D_A1_Guess_In,
                      const blitz::Array<int, 2> &I_A2_Limited,
                      const blitz::Array<double, 2> &D_A2_Limits,
                      const bool B_WithConstantBackground,
                      const bool B_FitArea,
                      blitz::Array<double, 1> &D_A1_Coeffs_Out,
                      blitz::Array< double, 1 >& D_A1_ECoeffs_Out);

bool MPFitThreeGauss(const blitz::Array<double, 1> &D_A1_X_In,
                     const blitz::Array<double, 1> &D_A1_Y_In,
                     const blitz::Array<double, 1> &D_A1_EY_In,
                     const blitz::Array<double, 1> &D_A1_Guess_In,
                     const bool B_WithConstantBackground,
                     const bool B_FitArea,
                     blitz::Array<double, 1> &D_A1_Coeffs_Out,
                     blitz::Array< double, 1 >& D_A1_ECoeffs_Out);

bool MPFitThreeGaussFix(const blitz::Array<double, 1> &D_A1_X_In,
                        const blitz::Array<double, 1> &D_A1_Y_In,
                        const blitz::Array<double, 1> &D_A1_EY_In,
                        const blitz::Array<double, 1> &D_A1_Guess_In,
                        const blitz::Array<int, 1> &I_A1_Fix,
                        const bool B_WithConstantBackground,
                        const bool B_FitArea,
                        blitz::Array<double, 1> &D_A1_Coeffs_Out,
                        blitz::Array< double, 1 >& D_A1_ECoeffs_Out);

bool MPFitThreeGaussLim(const blitz::Array<double, 1> &D_A1_X_In,
                        const blitz::Array<double, 1> &D_A1_Y_In,
                        const blitz::Array<double, 1> &D_A1_EY_In,
                        const blitz::Array<double, 1> &D_A1_Guess_In,
                        const blitz::Array<int, 2> &I_A2_Limited,
                        const blitz::Array<double, 2> &D_A2_Limits,
                        const bool B_WithConstantBackground,
                        const bool B_FitArea,
                        blitz::Array<double, 1> &D_A1_Coeffs_Out,
                        blitz::Array< double, 1 >& D_A1_ECoeffs_Out);

bool MPFit2DGaussLim(const blitz::Array< double, 1 >& D_A1_X_In,
                     const blitz::Array< double, 1 >& D_A1_Y_In,
                     const blitz::Array< double, 1 >& D_A1_Z_In,
                     const blitz::Array< double, 1 >& D_A1_Guess_In,
                     const blitz::Array<int, 2> &I_A2_Limited,
                     const blitz::Array<double, 2> &D_A2_Limits,
                     blitz::Array< double, 1 >& D_A1_Coeffs_Out,
                     blitz::Array< double, 1 >& D_A1_ECoeffs_Out);

#endif
