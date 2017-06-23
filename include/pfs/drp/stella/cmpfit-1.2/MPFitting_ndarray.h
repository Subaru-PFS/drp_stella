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

/**
 * \file
 * \brief Implementation of the Levenberg-Markwardt fitting procedures
 */

#ifndef __MYFIT_H__
#define __MYFIT_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mpfit.h"
#include "ndarray.h"
#include "lsst/pex/exceptions/Exception.h"
#include "pfs/drp/stella/math/Math.h"

namespace pexExcept = lsst::pex::exceptions;
using namespace std;

#define D_PI 3.14159265359

/* This is the private data structure which contains the data points
   and their uncertainties */
struct vars_struct {
  double *x;
  double *y;
  double *ey;
};

/** 
 * \brief Simple routine to print the fit results 
 * 
 * \param[in] x vector of results
 * \param[in] result structure containing the uncertainties for x plus information about the fit
 */
void PrintResult(double *x, mp_result *result);

/** \brief gaussian fit function plus constant background (CB)
 * \param m number of data points
 * \param n number of parameters (4)
 * \param p array of fit parameters
 *           p[0] = peak y value
 *           p[1] = x centroid position
 *           p[2] = gaussian sigma width
 *           p[3] = constant offset
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFitGaussFuncCB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief gaussian fit function without background (NB)
 * \param m number of data points
 * \param n number of parameters (3)
 * \param p array of fit parameters
 *           p[0] = peak y value
 *           p[1] = x centroid position
 *           p[2] = gaussian sigma width
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFitGaussFuncNB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief gaussian fit function area with constant background (ACB)
 * \param m number of data points
 * \param n number of parameters (4)
 * \param p array of fit parameters
 *           p[0] = area under curve value
 *           p[1] = x centroid position
 *           p[2] = gaussian sigma width
 *           p[3] = constant offset
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFitGaussFuncACB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief gaussian fit function area no background (ANB)
 * \param m number of data points
 * \param n number of parameters (3)
 * \param p array of fit parameters
 *           p[0] = area under curve value
 *           p[1] = x centroid position
 *           p[2] = gaussian sigma width
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFitGaussFuncANB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief gaussian fit function plus constant and linear background (LB)
 * \param m number of data points
 * \param n number of parameters (5)
 * \param p array of fit parameters
 *           p[0] = peak y value
 *           p[1] = x centroid position
 *           p[2] = gaussian sigma width
 *           p[3] = constant offset
 *           p[4] = linear offset
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFitGaussFuncLB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief gaussian fit function area plus constant and linear background
 * \param m number of data points
 * \param n number of parameters (5)
 * \param p array of fit parameters
 *           p[0] = area under curve value
 *           p[1] = x centroid position
 *           p[2] = gaussian sigma width
 *           p[3] = constant offset
 *           p[4] = linear offset
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFitGaussFuncALB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief gaussian fit function 2 Gaussians plus constant background (CB)
 * \param m number of data points
 * \param n number of parameters (6)
 * \param p array of fit parameters
 *           p[0] = peak y value 1st Gauss
 *           p[1] = x centroid position 1st Gauss
 *           p[2] = gaussian sigma width
 *           p[3] = peak y value 2nd Gauss
 *           p[4] = x centroid position 2nd Gauss
 *           p[5] = constant offset
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFitTwoGaussFuncCB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief gaussian fit function 2 Gaussians no background (NB)
 * \param m number of data points
 * \param n number of parameters (5)
 * \param p array of fit parameters
 *           p[0] = peak y value 1st Gauss
 *           p[1] = x centroid position 1st Gauss
 *           p[2] = gaussian sigma width
 *           p[3] = peak y value 2nd Gauss
 *           p[4] = x centroid position 2nd Gauss
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFitTwoGaussFuncNB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief gaussian fit function 2 Gaussians area plus constant background (ACB)
 * \param m number of data points
 * \param n number of parameters (6)
 * \param p array of fit parameters
 *           p[0] = area under curve value 1st Gauss
 *           p[1] = x centroid position 1st Gauss
 *           p[2] = gaussian sigma width
 *           p[3] = area under curve value 2nd Gauss
 *           p[4] = x centroid position 2nd Gauss
 *           p[5] = constant offset
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFitTwoGaussFuncACB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief gaussian fit function 2 Gaussians area no background (ANB)
 * \param m number of data points
 * \param n number of parameters (5)
 * \param p array of fit parameters
 *           p[0] = area under curve value 1st Gauss
 *           p[1] = x centroid position 1st Gauss
 *           p[2] = gaussian sigma width
 *           p[3] = area under curve value 2nd Gauss
 *           p[4] = x centroid position 2nd Gauss
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFitTwoGaussFuncANB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief gaussian fit function 3 Gaussians plus constant background
 * \param m number of data points
 * \param n number of parameters (8)
 * \param p array of fit parameters
 *           p[0] = peak y value 1st Gauss
 *           p[1] = x centroid position 1st Gauss
 *           p[2] = gaussian sigma width
 *           p[3] = peak y value 2nd Gauss
 *           p[4] = x centroid position 2nd Gauss
 *           p[5] = peak y value 3rd Gauss
 *           p[6] = x centroid position 3rd Gauss
 *           p[7] = constant offset
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFitThreeGaussFuncCB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief gaussian fit function 2D Gaussian plus constant background
 * \param m number of data points
 * \param n number of parameters (5)
 * \param p array of fit parameters
 *           p[0] = peak z value Gauss
 *           p[1] = x centroid position Gauss
 *           p[2] = y centroid position Gauss
 *           p[3] = gaussian sigma width
 *           p[4] = constant offset
 * \param dz array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFit2DGaussFuncCB(int m, int n, double *p, double *dz, double **dvec, void *vars);

/** \brief gaussian fit function 3 Gaussians no background (NB)
 * \param m number of data points
 * \param n number of parameters (7)
 * \param p array of fit parameters
 *           p[0] = peak y value 1st Gauss
 *           p[1] = x centroid position 1st Gauss
 *           p[2] = gaussian sigma width 1st Gauss
 *           p[3] = peak y value 2nd Gauss
 *           p[4] = x centroid position 2nd Gauss
 *           p[5] = peak y value 3rd Gauss
 *           p[6] = x centroid position 3rd Gauss
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFitThreeGaussFuncNB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief gaussian fit function 3 Gaussians area plus constant background
 * \param m number of data points
 * \param n number of parameters (8)
 * \param p array of fit parameters
 *           p[0] = constant offset
 *           p[1] = area under curve value 1st Gauss
 *           p[2] = x centroid position 1st Gauss
 *           p[3] = gaussian sigma width 1st Gauss
 *           p[4] = area under curve value 2nd Gauss
 *           p[5] = x centroid position 2nd Gauss
 *           p[6] = area under curve value 3rd Gauss
 *           p[7] = x centroid position 3rd Gauss
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFitThreeGaussFuncACB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief gaussian fit function 3 Gaussians area no background
 * \param m number of data points
 * \param n number of parameters (7)
 * \param p array of fit parameters
 *           p[0] = area under curve value 1st Gauss
 *           p[1] = x centroid position 1st Gauss
 *           p[2] = gaussian sigma width 1st Gauss
 *           p[3] = area under curve value 2nd Gauss
 *           p[4] = x centroid position 2nd Gauss
 *           p[5] = area under curve value 3rd Gauss
 *           p[6] = x centroid position 3rd Gauss
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 *
 * RETURNS error code (0 = success)
 */
int MPFitThreeGaussFuncANB(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief Chebyshev polynomial 1st kind
 * \param m number of data points
 * \param n number of parameters (order + 1)
 * \param p array of fitting parameters
 * \param dy array of residuals to be returned
 * \param dvec place holder to comply with template structure ( (void *) &v )
 * \param vars private data (struct vars_struct *)
 * 
 * RETURNS error code (0 = success)
 */
int Chebyshev1stKind(int m, int n, double *p, double *dy, double **dvec, void *vars);

/** \brief Fit a Chebyshev polynomial of the 1st kind to x and y. 
 *         X will be shifted and rescaled to fit into the range [-1, 1].
 *         The order of the polynomial will be determined from the length of D_A1_Coeffs_Out (must be equal to length of D_A1_ECoeffs_Out).
 * \param D_A1_X_In x values to fit
 * \param D_A1_Y_In y values to fit
 * \param D_A1_EY_In uncertainies in y values
 * \param D_A1_Coeffs_Out fitted coefficients
 * \param D_A1_ECoeffs_Out uncertainties in fitted coefficients
 */
template<typename T> 
bool MPFitChebyshev1stKind(const ndarray::Array<T, 1, 1> & D_A1_X_In,
                           const ndarray::Array<T, 1, 1> & D_A1_Y_In,
                           const ndarray::Array<T, 1, 1> & D_A1_EY_In,
                           ndarray::Array<T, 1, 1> & D_A1_Coeffs_Out,
                           ndarray::Array<T, 1, 1> & D_A1_ECoeffs_Out);

/** \brief Fit a Chebyshev polynomial of the 1st kind to x and y. 
 *         X will be shifted and rescaled to fit into the range [-1, 1].
 *         The order of the polynomial will be determined from the length of D_A1_Coeffs_Out (must be equal to length of D_A1_ECoeffs_Out).
 * \param D_A1_X_In x values to fit
 * \param D_A1_Y_In y values to fit
 * \param D_A1_EY_In uncertainies in y values
 * \param D_A1_Guess_In guessed values for fitting coefficients
 * \param D_A1_Coeffs_Out fitted coefficients
 * \param D_A1_ECoeffs_Out uncertainties in fitted coefficients
 */
template<typename T> 
bool MPFitChebyshev1stKind(const ndarray::Array<T, 1, 1> & D_A1_X_In,
                           const ndarray::Array<T, 1, 1> & D_A1_Y_In,
                           const ndarray::Array<T, 1, 1> & D_A1_EY_In,
                           const ndarray::Array<T, 1, 1> & D_A1_Guess_In,
                           ndarray::Array<T, 1, 1> & D_A1_Coeffs_Out,
                           ndarray::Array<T, 1, 1> & D_A1_ECoeffs_Out);

/** \brief procedure for fitting one Gaussian
 * \param D_A1_X_In x values to fit
 * \param D_A1_Y_In y values to fit
 * \param D_A1_EY_In uncertainties in y values to fit
 * \param D_A1_Guess_In guessed values for fitting parameters
 * \param B_WithConstantBackground Fit one Gaussian plus a constant background?
 * \param B_FitArea Fit area (YES) or amplitude (No)?
 * \param D_A1_Coeffs_Out fitted coefficients (output parameter)
 * \param D_A1_ECoeffs_Out uncertainties of fitted coefficients (output parameter)
 */
template<typename T>
bool MPFitGauss(const ndarray::Array<T, 1, 1> &D_A1_X_In,
                const ndarray::Array<T, 1, 1> &D_A1_Y_In,
                const ndarray::Array<T, 1, 1> &D_A1_EY_In,
                const ndarray::Array<T, 1, 1> &D_A1_Guess_In,
                const bool B_WithConstantBackground,
                const bool B_FitArea,
                ndarray::Array<T, 1, 1> &D_A1_Coeffs_Out,
                ndarray::Array< T, 1, 1>& D_A1_ECoeffs_Out);

/** \brief procedure for fitting one Gaussian holding some fitting parameters fixed at the guessed values
 * \param D_A1_X_In x values to fit
 * \param D_A1_Y_In y values to fit
 * \param D_A1_EY_In uncertainties in y values to fit
 * \param D_A1_Guess_In guessed values for fitting parameters
 * \param I_A1_Fix For each fitting parameter: 0 for not fixed, 1 for fixed at guessed value
 * \param B_WithConstantBackground Fit one Gaussian plus a constant background?
 * \param B_FitArea Fit area (YES) or amplitude (No)?
 * \param D_A1_Coeffs_Out fitted coefficients (output parameter)
 * \param D_A1_ECoeffs_Out uncertainties of fitted coefficients (output parameter)
 */
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

/** 
 * @brief Procedure for fitting one Gaussian holding some fitting parameters within certain limits
 * 
 * @param[in] D_A1_X_In             :: x values to fit
 * @param[in] D_A1_Y_In             :: y values to fit
 * @param[in] D_A1_EY_In            :: uncertainties in y values to fit
 * @param[in] D_A1_Guess_In         :: guessed values for fitting parameters
 * @param[in] I_A2_Limited          :: For each fitting parameter: 2 values: 1st value for lower limit, 2nd value for upper limit:
 *                      0 for not limited, 1 for limited at corresponding value given in D_A2_Limits
 * @param[in] D_A2_Limits           :: For each fitting parameter: 2 values: lower limits and upper limit
 * @param[in] Background            :: Fit one Gaussian plus background? 0-none, 1-constant, 2-linear background
 * @param[in] B_FitArea             :: Fit area (YES) or amplitude (No)?
 * @param[in,out] D_A1_Coeffs_Out   :: fitted coefficients (output parameter)
 * @param[in,out] D_A1_ECoeffs_Out  :: uncertainties of fitted coefficients (output parameter)
 * @param[in] Debug                 :: print out debugging output?
 */
template<typename T>
bool MPFitGaussLim(const ndarray::Array<T, 1, 1> &D_A1_X_In,
                   const ndarray::Array<T, 1, 1> &D_A1_Y_In,
                   const ndarray::Array<T, 1, 1> &D_A1_EY_In,
                   const ndarray::Array<T, 1, 1> &D_A1_Guess_In,
                   const ndarray::Array<int, 2, 1> &I_A2_Limited,
                   const ndarray::Array<T, 2, 1> &D_A2_Limits,
                   const int Background,
                   const bool B_FitArea,
                   ndarray::Array<T, 1, 1> &D_A1_Coeffs_Out,
                   ndarray::Array< T, 1, 1>& D_A1_ECoeffs_Out,
                   bool Debug = false);

/** \brief procedure for fitting two Gaussians
 * \param D_A1_X_In x values to fit
 * \param D_A1_Y_In y values to fit
 * \param D_A1_EY_In uncertainties in y values to fit
 * \param D_A1_Guess_In guessed values for fitting parameters
 * \param B_WithConstantBackground Fit two Gaussians plus constant background?
 * \param B_FitArea Fit area (YES) or amplitude (No)?
 * \param D_A1_Coeffs_Out fitted coefficients (output parameter)
 * \param D_A1_ECoeffs_Out uncertainties of fitted coefficients (output parameter)
 */
template<typename T>
bool MPFitTwoGauss(const ndarray::Array<T, 1, 1> &D_A1_X_In,
                   const ndarray::Array<T, 1, 1> &D_A1_Y_In,
                   const ndarray::Array<T, 1, 1> &D_A1_EY_In,
                   const ndarray::Array<T, 1, 1> &D_A1_Guess_In,
                   const bool B_WithConstantBackground,
                   const bool B_FitArea,
                   ndarray::Array<T, 1, 1> &D_A1_Coeffs_Out,
                   ndarray::Array< T, 1, 1>& D_A1_ECoeffs_Out);

/** \brief procedure for fitting two Gaussian holding some fitting parameters fixed at guessed values
 * \param D_A1_X_In x values to fit
 * \param D_A1_Y_In y values to fit
 * \param D_A1_EY_In uncertainties in y values to fit
 * \param D_A1_Guess_In guessed values for fitting parameters
 * \param I_A1_Fix for each fitting parameter: 0 for not fixed, 1 for fixed at value given in D_A2_Limits
 * \param B_WithConstantBackground Fit one Gaussian plus constant background?
 * \param B_FitArea Fit area (YES) or amplitude (No)?
 * \param D_A1_Coeffs_Out fitted coefficients (output parameter)
 * \param D_A1_ECoeffs_Out uncertainties of fitted coefficients (output parameter)
 */
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

/** \brief procedure for fitting two Gaussians holding some fitting parameters within certain limits
 * \param D_A1_X_In x values to fit
 * \param D_A1_Y_In y values to fit
 * \param D_A1_EY_In uncertainties in y values to fit
 * \param D_A1_Guess_In guessed values for fitting parameters
 * \param I_A2_Limited For each fitting parameter: 2 values: 1st value for lower limit, 2nd value for upper limit:
 *                      0 for not fixed, 1 for fixed at value given in D_A2_Limits
 * \param D_A2_Limits For each fitting parameter: 2 values: lower limits and upper limit
 * \param B_WithConstantBackground Fit one Gaussian plus constant background?
 * \param B_FitArea Fit area (YES) or amplitude (No)?
 * \param D_A1_Coeffs_Out fitted coefficients (output parameter)
 * \param D_A1_ECoeffs_Out uncertainties of fitted coefficients (output parameter)
 */
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

/** \brief procedure for fitting three Gaussians
 * \param D_A1_X_In x values to fit
 * \param D_A1_Y_In y values to fit
 * \param D_A1_EY_In uncertainties in y values to fit
 * \param D_A1_Guess_In guessed values for fitting parameters
 * \param B_WithConstantBackground Fit three Gaussians plus constant background?
 * \param B_FitArea Fit area (YES) or amplitude (No)?
 * \param D_A1_Coeffs_Out fitted coefficients (output parameter)
 * \param D_A1_ECoeffs_Out uncertainties of fitted coefficients (output parameter)
 */
template<typename T>
bool MPFitThreeGauss(const ndarray::Array<T, 1, 1> &D_A1_X_In,
                     const ndarray::Array<T, 1, 1> &D_A1_Y_In,
                     const ndarray::Array<T, 1, 1> &D_A1_EY_In,
                     const ndarray::Array<T, 1, 1> &D_A1_Guess_In,
                     const bool B_WithConstantBackground,
                     const bool B_FitArea,
                     ndarray::Array<T, 1, 1> &D_A1_Coeffs_Out,
                     ndarray::Array< T, 1, 1>& D_A1_ECoeffs_Out);

/** \brief procedure for fitting three Gaussians holding some fitting parameters fixed at guessed values
 * \param D_A1_X_In x values to fit
 * \param D_A1_Y_In y values to fit
 * \param D_A1_EY_In uncertainties in y values to fit
 * \param D_A1_Guess_In guessed values for fitting parameters
 * \param I_A1_Fix For each fitting parameter: 0 for not fixed, 1 for fixed at value given in D_A1_Guess_In
 * \param B_WithConstantBackground Fit three Gaussians plus constant background?
 * \param B_FitArea Fit area (YES) or amplitude (No)?
 * \param D_A1_Coeffs_Out fitted coefficients (output parameter)
 * \param D_A1_ECoeffs_Out uncertainties of fitted coefficients (output parameter)
 */
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

/** \brief procedure for fitting three Gaussian holding some fitting parameters within certain limits
 * \param D_A1_X_In x values to fit
 * \param D_A1_Y_In y values to fit
 * \param D_A1_EY_In uncertainties in y values to fit
 * \param D_A1_Guess_In guessed values for fitting parameters
 * \param I_A2_Limited For each fitting parameter: 2 values: 1st value for lower limit, 2nd value for upper limit:
 *                      0 for not limited, 1 for limited at corresponging value given in D_A2_Limits
 * \param D_A2_Limits For each fitting parameter: 2 values: lower limits and upper limit
 * \param B_WithConstantBackground Fit three Gaussians plus constant background?
 * \param B_FitArea Fit area (YES) or amplitude (No)?
 * \param D_A1_Coeffs_Out fitted coefficients (output parameter)
 * \param D_A1_ECoeffs_Out uncertainties of fitted coefficients (output parameter)
 */
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

/** \brief procedure for fitting one 2-dimensional Gaussian holding certain fitting parameters within limits
 * \param D_A1_X_In x values to fit
 * \param D_A1_Y_In y values to fit
 * \param D_A1_Z_In z values to fit
 * \param D_A1_Guess_In guessed values for fitting parameters
 * \param I_A2_Limited For each fitting parameter: 2 values: 1st value for lower limit, 2nd value for upper limit:
 *                      0 for not limited, 1 for limited at corresponding value given in D_A2_Limits
 * \param D_A2_Limits For each fitting parameter: 2 values: lower limits and upper limit
 * \param D_A1_Coeffs_Out fitted coefficients (output parameter)
 * \param D_A1_ECoeffs_Out uncertainties of fitted coefficients (output parameter)
 */
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
