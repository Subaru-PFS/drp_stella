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

#include "mpfit.h"
#include "ndarray.h"

#define D_PI 3.14159265359

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
#endif
