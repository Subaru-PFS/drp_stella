///TODO: Replace all pointers with sharedPointers!

#ifndef __PFS_DRP_STELLA_MATHBLITZ_H__
#define __PFS_DRP_STELLA_MATHBLITZ_H__

#include <vector>
#include <iostream>
#include "lsst/base.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/config.h"
#include "../blitz.h"
#include "../utils/Utils.h"
#include "../utils/UtilsBlitz.h"
#include "../Controls.h"
#include "../Spectra.h"
#include "Math.h"
#include "ndarray.h"
#include "ndarray/eigen.h"

//#define __DEBUG_FIT__
//#define __DEBUG_FITARR__
//#define __DEBUG_POLY__
//#define __DEBUG_POLYFIT__
//#define __DEBUG_MINCENMAX__
//#define __DEBUG_INDGEN__
#define DEBUGDIR "/Users/azuri/spectra/pfs/2014-11-02/debug/"// /home/azuri/entwicklung/idl/REDUCE/16_03_2013/"//stella/ses-pipeline/c/msimulateskysubtraction/data/"//spectra/elaina/eso_archive/red_564/red_r/"

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace pfsDRPStella = pfs::drp::stella;
using namespace std;

namespace pfs { namespace drp { namespace stella {
  namespace math{
    
    /*****************************************************************/
    /*  Sub method for CubicSpline, Legendre, and Chebyshev          */
    /*****************************************************************/
    double GetNormalized(double XVal,
                         double XMin,
                         double XMax);

    /** **************************************************/
    /** Sub method for LinearSpline and CubicSpline      */
    /** **************************************************/
    double GetA(double XVal,
                double XMin,
                double XMax,
                int Order);

    /** **************************************************/
    /** Sub method for LinearSpline and CubicSpline      */
    /** **************************************************/
    double GetB(double XVal,
                double XMin,
                double XMax,
                int Order);

    /** **************************************************/
    /** Sub method for LinearSpline and CubicSpline      */
    /** **************************************************/
    long GetJ(double XVal,
              double XMin,
              double XMax,
              int Order);

    /** **************************************************/
    /** Sub method for LinearSpline and CubicSpline      */
    /** **************************************************/
    double GetS(double XVal,
                double XMin,
                double XMax,
                int Order);

    bool LinearSpline(blitz::Array<double, 1> &D_A1_XCenters_Out,
                      const blitz::Array<double, 1> &D_A1_Coeffs_In,
                      double D_XCenter_In,
                      double D_YCenter_In,
                      double D_YMin_In,
                      double D_YMax_In,
                      double D_Low_In,
                      double D_High_In,
                      int I_Order_In,
                      int I_NCols_In,
                      int I_NRows_In);

    bool CubicSpline(blitz::Array<double, 1> &D_A1_XCenters_Out,
                     const blitz::Array<double, 1> &D_A1_Coeffs_In,
                     double D_XCenter_In,
                     double D_YCenter_In,
                     double D_YMin_In,
                     double D_YMax_In,
                     double D_XLow_In,
                     double D_XHigh_In,
                     int I_Order_In,
                     int I_NCols_In,
                     int I_NRows_In);


    /**
     *  Spline
     *  Given Arrays x_In(0:N-1) and y_In(0:N-1) containing a tabulated function,
     *  i.e., y_i = f(x_i), with x_1 < x_2 < ... < x_N, and given values yP1 and
     *  yPN for the first derivative of the interpolating function at points 1 and
     *  N, respectively, this routine returns an Array y2(0:N-1) that contains the
     *  second derivatives of the interpolating function at the tabulated points
     *  x_i. If yP1 and/or yPN are equal to 1x10^30 or larger, the routine is
     *  signaled to set the corresponding boundary condition for a natural spline,
     *  with zero second derivative on that boundary.
     **/
    bool Spline(const blitz::Array<double, 1> &x_In,
                const blitz::Array<double, 1> &y_In,
                double yP1,
                double yPN,
                blitz::Array<double, 1> &y_Out);

    /**
     *  Spline
     *  Given Arrays x_In(0:N-1) and y_In(0:N-1) containing a tabulated function,
     *  i.e., y_i = f(x_i), with x_1 < x_2 < ... < x_N, this routine returns an
     *  Array y2(0:N-1) that contains the second derivatives of the interpolating
     *  function at the tabulated points x_i. The routine is signaled to set the
     *  corresponding boundary condition for a natural spline, with zero second
     *  derivative on that boundary.
     **/
    bool Spline(const blitz::Array<double, 1> &x_In,
                const blitz::Array<double, 1> &y_In,
                blitz::Array<double, 1> &y_Out);

    /**
     *  SplInt
     *  Given the Arrays xVec_In(0:N-1) and y1_In(0:N-1), which tabulate a
     *  function (whith the xVec_In(i)'s in order), and given the array y2_In(0:N-1),
     *  which is the output from Spline above, and given a value of x_In, this
     *  routine returns a cubic-spline interpolated value y_Out;
     **/
    bool SplInt(const blitz::Array<double, 1> &xVec_In,
                blitz::Array<double, 1> &y1_In,
                blitz::Array<double, 1> &y2_In,
                double x_In,
                double *y_Out);

    bool ChebyLegend(blitz::Array<double, 1> &D_A1_XCenters_Out,
                     double &D_YMin_Out,
                     double &D_YMax_Out,
                     const blitz::Array<double, 1> &D_A1_Coeffs_In,
                     double D_XCenter_In,
                     double D_YCenter_In,
                     double D_YMin_In,
                     double D_YMax_In,
                     double D_XLow_In,
                     double D_XHigh_In,
                     int I_Order_In,
                     int I_NCols_In,
                     int I_NRows_In,
                     const string &S_Function_In);

    bool Legendre(blitz::Array<double, 1> &D_A1_XCenters_Out,
                  double &D_YMin_Out,
                  double &D_YMax_Out,
                  const blitz::Array<double, 1> &D_A1_Coeffs_In,
                  double D_XCenter_In,
                  double D_YCenter_In,
                  double D_YMin_In,
                  double D_YMax_In,
                  double D_XLow_In,
                  double D_XHigh_In,
                  int I_Order_In,
                  int I_NCols_In,
                  const int I_NRows_In);

    bool Chebyshev(blitz::Array<double, 1> &D_A1_XCenters_Out,
                   double &D_YMin_Out,
                   double &D_YMax_Out,
                   const blitz::Array<double, 1> &D_A1_Coeffs_In,
                   double D_XCenter_In,
                   double D_YCenter_In,
                   double D_YMin_In,
                   double D_YMax_In,
                   double D_XLow_In,
                   double D_XHigh_In,
                   int I_Order_In,
                   int I_NCols_In,
                   const int I_NRows_In);

    /*************************************************************
     * Poly
     *
     * INPUTS:
     *       D_A1_X_In:      The variable.  1D array.
     *
     *       D_A1_Coeffs_In: The 1D array of polynomial coefficients.  The degree of
     *                       of the polynomial is N_ELEMENTS(VecCoeffs) - 1.
     *
     * OUTPUTS:
     *       POLY returns a result equal to:
     *                C[0] + c[1] * X + c[2]*X^2 + ...
     *
    **/
    blitz::Array<double, 1> Poly(const blitz::Array<double, 1> &D_A1_X_In,
                                 const blitz::Array<double, 1> &D_A1_Coeffs_In);
    
    double Poly(const double D_X_In,
                const blitz::Array<double, 1> &D_A1_Coeffs_In);

    /**
     *        Returns Indexes of I_A1_Where where I_A1_Where equals 1 and writes sum(I_A1_Where) to I_NInd_Out
     **/
    blitz::Array<int,1>* GetIndex(const blitz::Array<int,1> &I_A1_Where,
                                  int &I_NInd_Out);

    /**
     *      Returns Indexes of I_A1_Where where I_A1_Where equals 1 and writes sum(I_A1_Where) to I_NInd_Out
     **/
    bool GetIndex(const blitz::Array<int,1> &I_A1_Where,
                  int &I_NInd_Out,
                  blitz::Array<int, 1> &I_IndArr_Out);

    /**
     *      Returns Indexes of I_A2_Where where I_A2_Where equals 1 and writes sum(I_A2_Where) to I_NInd_Out
     *      blitz::Array<int, 2> *P_I_A2_Out(I_NInd_Out, 2)
     **/
    blitz::Array<int,2>* GetIndex(const blitz::Array<int,2> &I_A2_Where,
                                  int &I_NInd_Out);

    /**
     *      Returns Indexes of I_A2_Where where I_A2_Where equals 1 and writes sum(I_A2_Where) to I_NInd_Out
     *      blitz::Array<int, 2> I_IndArr_Out(I_NInd_Out, 2)
     **/
    bool GetIndex(const blitz::Array<int,2> &I_A2_Where,
                  int &I_NInd_Out,
                  blitz::Array<int, 2> &I_IndArr_Out);

    /**
     *      ValueLocate
     *      Returns the successive indices of the Range of the two indices of the monotonically increasing or decreasing Vector vec_In,
     *        in which Val falls.
     *      If Vector is monotonically increasing, the result is
     *      if j = -1       valueVec_In(i) < vec_In(0)
     *      if 0 <= j < N-1 vec_In(j) <= valueVec_In(i) < vec_In(j+1)
     *      if j = N-1      vec_In(N-1) <= valueVec_In(i)
     *
     *      If Vector is monotonically decreasing, the result is
     *      if j = -1       vec_In(0) <= valueVec_In(i)
     *      if 0 <= j < N-1 vec_In(j+1) <= valueVec_In(i) < vec_In(j)
     *      if j = N-1      valueVec_In(i) < vec_In(N-1)
     **/
    blitz::Array<int, 1>* valueLocate(const blitz::Array<double, 1> &vec_In,
                                      const blitz::Array<double, 1> &valueVec_In);

    /**
     * Calculates Slit Function for each pixel in an aperture row from oversampled Slit Function D_A1_OSF_In,
     * and writes result to D_A1_SF_Out
     **
    bool CalcSF(const blitz::Array<double, 1> &xCenters_In,
                unsigned int I_Row_In,
                float xHigh_In,
                float xLow_In,
                unsigned int overSample_In,
                const blitz::Array<double, 1> &D_A1_OSF_In,
                const pfs::drp::stella::FiberTrace<float>::FiberTrace::MaskedImageT &image_In,
                blitz::Array<double, 1> &D_A1_SF_Out);
     */


    /**
      Fix(blitz::Array<double, 1> &VecArr)
      Returns an Array of the same size containing the Fix integer values of VecArr.
    **/
    template <typename T>
    blitz::Array<int, 1> Fix(const blitz::Array<T, 1> &VecArr);

    /**
     Fix(blitz::Array<double, 2> &Arr)
     Returns an Array of the same size containing the Fix integer values of Arr (see int Fix(double D_In)).
     **/
    template <typename T>
    blitz::Array<int, 2> Fix(const blitz::Array<T, 2> &Arr);

    /**
      FixL(blitz::Array<double, 1> &VecArr)
      Returns an Array of the same size containing the fix long integer values of VecArr (see int Fix(double D_In)).
     **/
    template <typename T>
    blitz::Array<long, 1> FixL(const blitz::Array<T, 1> &VecArr);

    /**
     FixL(blitz::Array<double, 2> &Arr, CString Mode)
     Returns an Array of the same size containing the long integer values of Arr (see int Fix(double D_In)).
     **/
    template <typename T>
    blitz::Array<long, 2> FixL(const blitz::Array<T, 2> &Arr);
    
    /**
     *      Int(blitz::Array<double, 1> &VecArr)
     *      Returns an Array of the same size containing the Int integer values of VecArr.
     **/
    template <typename T>
    blitz::Array<int, 1> Int(const blitz::Array<T, 1> &VecArr);

    /**
     *     Fix(blitz::Array<double, 2> &Arr)
     *     Returns an Array of the same size containing the Int integer values of Arr (see int Int(double D_In)).
     **/
    template <typename T>
    blitz::Array<int, 2> Int(const blitz::Array<T, 2> &Arr);


    /**
     *      Returns an Array of the same size containing the Int long integer values of VecArr (see int Int(double D_In)).
     **/
    template <typename T>
    blitz::Array<long, 1> Long(const blitz::Array<T, 1> &VecArr);

    /**
     *     Returns an Array of the same size containing the long integer values of Arr (see int Int(double D_In)).
     **/
    template <typename T>
    blitz::Array<long, 2> Long(const blitz::Array<T, 2> &Arr);

    /**
     *      Returns an Array of the same size containing the float values of VecArr.
     **/
    template <typename T>
    blitz::Array<float, 1> Float(const blitz::Array<T, 1> &VecArr);

    /**
     *      Returns an Array of the same size containing the float values of VecArr.
     **/
    template <typename T>
    void Float(const blitz::Array<T, 1> &VecArr, blitz::Array<float, 1> &VecArr_Out);

    /**
     *     Returns an Array of the same size containing the float values of Arr (see int Int(double D_In)).
     **/
    template <typename T>
    blitz::Array<float, 2> Float(const blitz::Array<T, 2> &Arr);

    /**
     *     Returns an Array of the same size containing the float values of Arr (see int Int(double D_In)).
     **/
    template <typename T>
    void Float(const blitz::Array<T, 2> &Arr, blitz::Array<float, 2> &Arr_Out);

    /**
     *     Returns the double representation of Arr.
     **/
    template <typename T>
    void Double(const blitz::Array<T, 1> &Arr, blitz::Array<double, 1> &Arr_Out);

    /**
     *     Returns the double representation of Arr.
     **/
    template <typename T>
    blitz::Array<double, 1> Double(const blitz::Array<T, 1> &Arr);
    
    /**
     *     Returns the double representation of Arr.
     **/
    template <typename T>
    void Double(const blitz::Array<T, 2> &Arr, blitz::Array<double, 2> &Arr_Out);

    /**
     *     Returns the double representation of Arr.
     **/
    template <typename T>
    blitz::Array<double, 2> Double(const blitz::Array<T, 2> &Arr);


    /**
     *      Replicate(double val, int Len);
     *      Out: blitz::Array<double, 1>(Len)
     **/
    template<typename T>
    blitz::Array<T, 1> Replicate(T val, int Len);

    /**
     * Calculate Integral under curve from D_A1_XInt(0) to D_A1_XInt(1)
     **/
    bool IntegralUnderCurve(const blitz::Array<double, 1> &D_A1_XIn,
                            const blitz::Array<double, 1> &D_A1_YIn,
                            const blitz::Array<double, 1> &D_A1_XInt,
                            double &D_Integral_Out);

    /**
     * Calculate Integral under line between two points
     * D_A2_Coords_In(0,0) = x0
     * D_A2_Coords_In(0,1) = y0
     * D_A2_Coords_In(1,0) = x1
     * D_A2_Coords_In(1,1) = y1
     * **/
    bool IntegralUnderLine(const blitz::Array<double, 2> &D_A2_Coords_In,
                           double &D_Integral_Out);

    /**
     * Integral-normalise a function
     **/
    bool IntegralNormalise(const blitz::Array<double, 1> &D_A1_XIn,
                           const blitz::Array<double, 1> &D_A1_YIn,
                           blitz::Array<double, 1> &D_A1_YOut);

    /**
     * Integral-normalise a function
     **/
    bool IntegralNormalise(const blitz::Array<double, 1> &D_A1_XIn,
                           blitz::Array<double, 1> &D_A1_YInOut);

    /**
     * PURPOSE:
     *   Perform a least-square polynomial fit with optional error estimates.
     *
     *   This routine uses matrix inversion.  A newer version of this routine,
     *   SVDFIT, uses Singular Value Decomposition.  The SVD technique is more
     *   flexible, but slower.
     *
     * INPUTS:
     *   X:  The independent variable vector.
     *
     *   Y:  The dependent variable vector, should be same length as x.
     *
     *   Degree: The degree of the polynomial to fit.
     *
     * OUTPUTS:
     *   POLY_FIT returns a vector of coefficients with a length of NDegree+1.
     *
     * KEYWORDS:
     *   CHISQ=chisq: double: out:
     *     Sum of squared errors divided by MEASURE_ERRORS if specified.
     *
     *   COVAR=covar: blitz::Array<double, 2>(I_Degree+1, I_Degree+1): out:
     *     Covariance matrix of the coefficients.
     *
     *   MEASURE_ERRORS=measure_errors: blitz::Array<double, 1>(D_A1_X_In.size()): in:
     *     Set this keyword to a vector containing standard
     *     measurement errors for each point Y[i].  This vector must be the same
     *     length as X and Y.
     *
     *     Note - For Gaussian errors (e.g. instrumental uncertainties),
     *       MEASURE_ERRORS should be set to the standard
     *       deviations of each point in Y. For Poisson or statistical weighting
     *       MEASURE_ERRORS should be set to sqrt(Y).
     *
     *   SIGMA=sigma: blitz::Array<double, 1>(I_Degree+1): out:
     *     The 1-sigma error estimates of the returned parameters.
     *
     *     Note: if MEASURE_ERRORS is omitted, then you are assuming that
     *       your model is correct. In this case, SIGMA is multiplied
     *       by SQRT(CHISQ/(N-M)), where N is the number of points
     *       in X and M is the number of terms in the fitting function.
     *       See section 15.2 of Numerical Recipes in C (2nd ed) for details.
     *
     *   STATUS=status: int: out:
     *     Set this keyword to a named variable to receive the status
     *     of the operation. Possible status values are:
     *     0 for successful completion, 1 for a singular array (which
     *     indicates that the inversion is invalid), and 2 which is a
     *     warning that a small pivot element was used and that significant
     *     accuracy was probably lost.
     *
     *   YFIT:   blitz::Vector of calculated Y's. These values have an error
     *           of + or - YBAND.
     *
    CHISQ=chisq: double: out
    COVAR=covar: blitz::Array<double, 2>(I_Degree+1, I_Degree+1): out
    MEASURE_ERRORS=measure_errors: blitz::Array<double, 1>(D_A1_X_In.size()): in
    SIGMA=sigma: blitz::Array<double, 1>(I_Degree+1): out
    STATUS=status: int: out
    YFIT=yfit: blitz::Array<double, 1>(D_A1_X_In.size()): out
    **/
    blitz::Array<double, 1> PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                                    const blitz::Array<double, 1> &D_A1_Y_In,
                                    int I_Degree,
                                    const blitz::Array<string, 1> &S_A1_Args,
                                    void *ArgV[]);

    blitz::Array<double, 1> PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                                    const blitz::Array<double, 1> &D_A1_Y_In,
                                    int I_Degree);

/** Additional Keywords:
    REJECTED=blitz::Array<int, 1>
    NOT_REJECTED=blitz::Array<int, 1>
    N_REJECTED=int
    **/
    blitz::Array<double, 1> PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                                    const blitz::Array<double, 1> &D_A1_Y_In,
                                    unsigned int I_Degree,
                                    double D_Reject,
                                    const blitz::Array<string, 1> &S_A1_Args,
                                    void *ArgV[]);
    blitz::Array<double, 1> PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                 const blitz::Array<double, 1> &D_A1_Y_In,
                 unsigned int I_Degree,
                 double D_LReject,
                 double D_HReject,
                 unsigned int I_NIter,
                 const blitz::Array<string, 1> &S_A1_Args,
                 void *ArgV[]);

    blitz::Array<double, 1> PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                                    const blitz::Array<double, 1> &D_A1_Y_In,
                                    unsigned int I_Degree,
                                    double D_Reject);
    
    blitz::Array<double, 1> PolyFit(const blitz::Array<double, 1> &D_A1_X_In,
                                    const blitz::Array<double, 1> &D_A1_Y_In,
                                    unsigned int I_Degree,
                                    double D_LReject,
                                    double D_HReject,
                                    unsigned int I_NIter);
    
    /**
     *  Creates float array containing the index numbers as values
     **/
    blitz::Array<float, 1> FIndGenArr(int len);

    /**
     *  Creates double array containing the index numbers as values
     **/
    blitz::Array<double, 1> DIndGenArr(int len);

    /**
     *  Creates long array containing the index numbers as values
     **/
    blitz::Array<long, 1> LIndGenArr(int len);

    /**
     *  Creates int array containing the index numbers as values
     **/
    blitz::Array<int, 1> IndGenArr(int len);

    bool removeSubArrayFromArray(blitz::Array<int, 1> &A1_Array_InOut,
                                 const blitz::Array<int, 1> &A1_SubArray);

    /**
      PURPOSE:
               Linearly interpolate vectors with a regular or irregular grid.
               Quadratic or a 4 point least-square fit to a quadratic
               interpolation may be used as an option.

      INPUTS:
             V:      The input vector can be any type except string.

         For regular grids:
             N:      The number of points in the result when both
                     input and output grids are regular.

         Irregular grids:
             X:      The absicissae values for V.  This vector must
                     have same # of elements as V.  The values MUST be
                     monotonically ascending or descending.

             U:      The absicissae values for the result.  The result
                     will have the same number of elements as U.  U
                     does not need to be monotonic.  If U is outside
                     the range of X, then the closest two endpoints of
                     (X,V) are linearly extrapolated.

      Keyword Input Parameters:
             LSQUADRATIC = if set, interpolate using a least squares
               quadratic fit to the equation y = a + bx + cx^2, for
               each 4 point neighborhood (x[i-1], x[i], x[i+1], x[i+2])
               surrounding the interval, x[i] <= u < x[i+1].

             QUADRATIC = if set, interpolate by fitting a quadratic
                         y = a + bx + cx^2, to the three point neighborhood
                         (x[i-1], x[i], x[i+1]) surrounding the interval
                         x[i] <= u < x[i+1].

             SPLINE = if set, interpolate by fitting a cubic spline to
                      the 4 point neighborhood (x[i-1], x[i], x[i+1],
                      x[i+2]) surrounding the interval, x[i] <= u <
                      x[i+1].

             Note: if LSQUADRATIC or QUADRATIC or SPLINE is not set,
                   the default linear interpolation is used.

      OUTPUTS:
             INTERPOL returns a floating-point vector of N points
             determined by interpolating the input vector by the
             specified method.

   PROCEDURE:
             For linear interpolation,
               Result(i) = V(x) + (x - FIX(x)) * (V(x+1) - V(x))
               where   x = i*(m-1)/(N-1) for regular grids.
                       m = # of elements in V, i=0 to N-1.

             For irregular grids, x = U(i).
                       m = number of points of input vector.

             For QUADRATIC interpolation, the equation y=a+bx+cx^2 is
               solved explicitly for each three point interval, and is
               then evaluated at the interpolate.

             For LSQUADRATIC interpolation, the coefficients a, b,
               and c, from the above equation are found, for the four
               point interval surrounding the interpolate using a
               least square fit.  Then the equation is evaluated at
               the interpolate.
               For SPLINE interpolation, a cubic spline is fit over
               the 4 point interval surrounding each interpolate,
               using the routine SPL_INTERP().
       **/
    /**
     InterPol linear, not regular
    **/
    bool InterPol(const blitz::Array<double, 1> &v,
                  const blitz::Array<double, 1> &x,
                  const blitz::Array<double, 1> &u,
                  blitz::Array<double, 1> &out);

    bool InterPol(const blitz::Array<double, 1> &v,
                  const blitz::Array<double, 1> &x,
                  const blitz::Array<double, 1> &u,
                  blitz::Array<double, 1> &out,
                  bool preserveFlux);

    /**
      InterPol
       The InterPol function performs linear, quadratic, or spline interpolation on vectors with an irregular grid.
     **/
    bool InterPol(const blitz::Array<double, 1> &v,
                  const blitz::Array<double, 1> &x,
                  const blitz::Array<double, 1> &u,
                  const blitz::Array<string, 1> &keyWords_In,
                  blitz::Array<double,1> &out);

    /**
      InterPol
       This function performs linear, quadratic, or spline interpolation on vectors with a regular grid.
     **
    bool InterPol(blitz::Array<double, 1> &v,
                  long n,
                  const blitz::Array<string, 1> &keyWords_In,
                  blitz::Array<double,1> &out);*/

    /**
      HInterPol
      Help function for InterPol methods
     **/
    bool HInterPol(const blitz::Array<double, 1> &v,
                   const blitz::Array<double, 1> &x,
                   blitz::Array<int, 1> &s,
                   const blitz::Array<double, 1> &u,
                   const blitz::Array<string, 1> &keyWords_In,
                   blitz::Array<double,1> &out);


    /**
     *        LsToFit
     **/
    bool LsToFit(const blitz::Array<double, 1> &XXVecArr,
                 const blitz::Array<double, 1> &YVecArr,
                 double XM,
                 double &D_Out);

    /**
     *        InvertGaussJ(AArray, BArray)
     *        Linear equation solution by Gauss-Jordan elimination
     *        AArray(0:N-1, 0:N-1) is the input matrix. BArray(0:N-1,
     *        0:M-1) is input containing the m right-hand side vectors.
     *        On output, AArray is replaced by its matrix inverse, and
     *        BArray is replaced by the corresponding set of solution
     *        vectors.
     **/
    bool InvertGaussJ(blitz::Array<double, 2> &AArray,
                      blitz::Array<double, 2> &BArray);

    /**
     *        InvertGaussJ(AArray)
     *        Linear equation solution by Gauss-Jordan elimination with B == Unity
     *        AArray(0:N-1, 0:N-1) is the input matrix.
     *        On output, AArray is replaced by its matrix inverse.
     **/
    bool InvertGaussJ(blitz::Array<double, 2> &AArray);

    /**
     *      MatrixATimesB(blitz::Array<double, 2> &Arr, blitz::Array<double, 2> &B);
     *      Out: blitz::Array<double, 2>(A.rows(), B.cols())
     **/
    blitz::Array<double, 2>* MatrixATimesB(const blitz::Array<double, 2> &A,
                                           const blitz::Array<double, 2> &B);

    /**
     *      MatrixBTimesA(blitz::Array<double, 2> &Arr, blitz::Array<double, 2> &B);
     *      Out: blitz::Array<double, 2>(B.rows(), A.cols())
     **/
    blitz::Array<double, 2>* MatrixBTimesA(const blitz::Array<double, 2> &A,
                                           const blitz::Array<double, 2> &B);

    /**
     *      MatrixTimesVecArr(blitz::Array<double, 2> &Arr, blitz::Array<double, 1> &B);
     *      Out: blitz::Array<double, 1>(A.rows())
     **/
    blitz::Array<double, 1>* MatrixTimesVecArr(const blitz::Array<double, 2> &A,
                                               const blitz::Array<double, 1> &B);

    /**
     *      VecArrTimesMatrix(blitz::Array<double, 1> &Arr, blitz::Array<double, 2> &B);
     *      Out: blitz::Array<double, 1>(B.cols())
     *      equivalent to IDL::operator #
     *      computes array elements by multiplying the rows of the first array by the columns of the second array
     **/
    blitz::Array<double, 1>* VecArrTimesMatrix(const blitz::Array<double, 1> &A,
                                               const blitz::Array<double, 2> &B);

    /**
     *      VecArrACrossB(blitz::Array<double, 1> &Arr, blitz::Array<double, 1> &B);
     *      Out: blitz::Array<double, 2>(A.size(), B.size())
     **/
    blitz::Array<double, 2>* VecArrACrossB(const blitz::Array<double, 1> &A,
                                           const blitz::Array<double, 1> &B);

    /**
     *      VecArrACrossB(blitz::Array<int, 1> &Arr, blitz::Array<int, 1> &B);
     *      Out: blitz::Array<int, 2>(A.size(), B.size())
     **/
    blitz::Array<int, 2>* VecArrACrossB(const blitz::Array<int, 1> &A,
                                        const blitz::Array<int, 1> &B);

    /**
     *      VecArrAScalarB(blitz::Array<double, 1> &Arr, blitz::Array<double, 1> &B);
     *      Out: double
     **/
    double VecArrAScalarB(const blitz::Array<double, 1> &A,
                          const blitz::Array<double, 1> &B);

    /**
     *      Reform(blitz::Array<double, 1> &Arr, int DimA, int DimB);
     *      Reformats blitz::Vector to Array of given size
     **/
    template<typename T>
    blitz::Array<T, 2>* Reform(const blitz::Array<T, 1> &Arr,
                               int DimA,
                               int DimB);

    /**
     *      Reform(blitz::Array<double, 2> &Arr);
     *      Reformates an Array to a blitz::Vector
     **/
    template<typename T>
    blitz::Array<T, 1>* Reform(const blitz::Array<T, 2> &Arr);

    /**
     *        GetSubArrCopy(blitz::Array<double, 1> &DA1_In, blitz::Array<int, 1> &IA1_Indices, blitz::Array<double, 1> &DA1_Out) const
     *        Copies the values of DA1_In(IA1_Indices) to DA1_Out
     **/
    template<typename T>
    bool GetSubArrCopy(const blitz::Array<T, 1> &DA1_In,
                       const blitz::Array<int, 1> &IA1_Indices,
                       blitz::Array<T, 1> &DA1_Out);

    /**
     *        GetSubArrCopy(blitz::Array<double, 2> &DA2_In, blitz::Array<int, 1> &IA1_Indices, int I_Mode_In, blitz::Array<double, 2> &DA2_Out) const
     *        Copies the values of DA1_In(IA1_Indices) to DA2_Out
     *        I_Mode_In: 0: IA1_Indices are row numbers
     *                   1: IA1_Indices are column numbers
     **/
    template<typename T>
    bool GetSubArrCopy(const blitz::Array<T, 2> &DA2_In,
                       const blitz::Array<int, 1> &IA1_Indices,
                       int I_Mode_In,
                       blitz::Array<T, 2> &DA2_Out);

    /**
     *        GetSubArrCopy(blitz::Array<double, 2> &DA1_In, blitz::Array<int, 3> &I_A3_Indices) const
     *        Copies the values of D_A2_In(I_A3_Indices(row,col,0), I_A3_Indices(row,col,1)) to D_A2_Out
     **/
    template<typename T>
    blitz::Array<T, 2> GetSubArrCopy(const blitz::Array<T, 2> &D_A2_In,
                                      const blitz::Array<int, 3> &I_A3_Indices);

    /**
     *        function CountPixGTZero
     *        replaces input vector with vector of the same size where values are zero where the input vector is lower than
     *        or equal to zero and all other values represent the number of gt-zero values since the last zero value
     **/
    template<typename T>
    bool CountPixGTZero(blitz::Array<T, 1> &vec_InOut);
    
    /**
     *        function FirstIndexWithValueGEFrom
     *        returns first index of integer input vector where value is greater than or equal to I_MinValue, starting at index I_FromIndex
     *        returns -1 if values are always smaller than I_MinValue
     **/
    template<typename T>
    int FirstIndexWithValueGEFrom(const blitz::Array<T, 1> &vecIn, const T minValue, const int fromIndex);

    /**
     *        function LastIndexWithZeroValueBefore
     *        returns last index of integer input vector where value is equal to zero, starting at index I_StartPos
     *        returns -1 if values are always greater than 0 before I_StartPos
     **/
    template<typename T>
    int LastIndexWithZeroValueBefore(const blitz::Array<T, 1> &vec_In, const int startPos_In);

    /**
     *        function FirstIndexWithZeroValueFrom
     *        returns first index of integer input vector where value is equal to zero, starting at index I_StartPos
     *        returns -1 if values are always greater than 0 past I_StartPos
     **/
    template<typename T>
    int FirstIndexWithZeroValueFrom(const blitz::Array<T, 1> &vec_In, const int startPos_In);

    /**
       CHANGES to original function:
         * D_Sky_Out must be >= 0. unless stated otherwise by the ALLOW_SKY_LT_ZERO parameter
         * D_SP_Out must be >= 0. unless stated otherwise by the ALLOW_SPEC_LT_ZERO parameter
         * added REJECT_IN as optinal parameter to reject cosmic rays from fit (times sigma)
         * added MASK_INOUT as optional parameter
     **/
    bool LinFitBevington(const blitz::Array<double, 1> &D_A1_CCD_In,      /// y: in
                         const blitz::Array<double, 1> &D_A1_SF_In,       /// x: in
                         double &D_SP_Out,                         /// a1: out
                         double &D_Sky_Out,                        /// a0: in/out
                         bool B_WithSky,                           /// with sky: in
                         const blitz::Array<string, 1> &S_A1_Args_In,   ///: in
                         void *ArgV_In[]);                   ///: in/out
    /// MEASURE_ERRORS_IN = blitz::Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = double                                                : in
    /// MASK_INOUT = blitz::Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = blitz::Array<double,1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = blitz::Array<double, 1>(D_A1_CCD_In.size)                     : out
    /// ALLOW_SKY_LT_ZERO = 1
    /// ALLOW_SPEC_LT_ZERO = 1
      
    /**
       CHANGES to original function:
         * D_Sky_Out must be >= 0. unless stated otherwise by the ALLOW_SKY_LT_ZERO parameter
         * D_SP_Out must be >= 0. unless stated otherwise by the ALLOW_SPEC_LT_ZERO parameter
         * if D_Sky_Out is set to be < -1.e-10 in input it is set to 0. and D_SP_Out is calculated as if there was no sky at all
         * added REJECT_IN as optinal parameter to reject cosmic rays from fit (times sigma)
         * added MASK_INOUT as optional parameter
     **/
    bool LinFitBevington(const blitz::Array<double, 1> &D_A1_CCD_In,      /// y: in
                         const blitz::Array<double, 1> &D_A1_SF_In,       /// x: in
                         double &D_SP_Out,                         /// a1: out
                         double &D_Sky_Out,                        /// a0: in/out
                         const blitz::Array<string, 1> &S_A1_Args_In,   ///: in
                         void *ArgV_In[]);                   ///: in/out
    /// MEASURE_ERRORS_IN = blitz::Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = double                                                : in
    /// MASK_INOUT = blitz::Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = blitz::Array<double,1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = blitz::Array<double, 1>(D_A1_CCD_In.size)                     : out
    /// ALLOW_SKY_LT_ZERO = 1
    /// ALLOW_SPEC_LT_ZERO = 1
     /**
            CHANGES to original function:
              * D_Sky_Out must be >= 0.
              * D_SP_Out must be >= 0.
              * if D_Sky_Out is set to be < -1.e-10 in input it is set to 0. and D_SP_Out is calculated as if there was no sky at all
              * added REJECT_IN as optinal parameter to reject cosmic rays from fit (times sigma)
              * added MASK_INOUT as optional parameter
      **/
      bool LinFitBevington(const blitz::Array<double, 2> &D_A2_CCD_In,      ///: in
                           const blitz::Array<double, 2> &D_A2_SF_In,       ///: in
                           blitz::Array<double,1> &D_A1_SP_Out,             ///: out
                           blitz::Array<double,1> &D_A1_Sky_Out,            ///: in/out
                           const blitz::Array<string, 1> &S_A1_Args_In,   ///: in
                           void *ArgV_In[]);                   ///: in/out
  /// MEASURE_ERRORS_IN = blitz::Array<double,2>(D_A2_CCD_In.rows, D_A2_CCD_In.cols) : in
  /// REJECT_IN = double                                                      : in
  /// MASK_INOUT = blitz::Array<double,2>(D_A1_CCD_In.rows,D_A1_CCD_In.cols)         : in/out
  /// CHISQ_OUT = blitz::Array<double,1>(D_A2_CCD_In.rows)                           : out
  /// Q_OUT = blitz::Array<double,1>(D_A2_CCD_In.rows)                               : out
  /// SIGMA_OUT = blitz::Array<double,2>(D_A2_CCD_In.rows, 2): [*,0]: sigma_sp, [*,1]: sigma_sky : out

      bool LinFitBevington(const blitz::Array<double, 2> &D_A2_CCD_In,      ///: in
                           const blitz::Array<double, 2> &D_A2_SF_In,       ///: in
                           blitz::Array<double,1> &D_A1_SP_Out,             ///: out
                           blitz::Array<double,1> &D_A1_Sky_Out,            ///: in/out
                           bool B_WithSky,                           ///: with sky: in
                           const blitz::Array<string, 1> &S_A1_Args_In,   ///: in
                           void *ArgV_In[]);                   ///: in/out
    template<typename T>
    T Median(const blitz::Array<T, 1> &Arr);

    template<typename T>
    T Median(const blitz::Array<T, 2> &Arr, bool B_IgnoreZeros);

    template<typename T>
    T Median(const blitz::Array<T, 1> &Arr,
             const blitz::Array<string, 1> &S_A1_Args_In,
             void *PP_Args[]);

//    template<typename T>
//    blitz::Array<T, 1> MedianVec(const blitz::Array<T, 1> &arr,
//                                  int Width);

    template<typename T>
    blitz::Array<T, 1> MedianVec(const blitz::Array<T, 1> &arr,
                                  int Width,
                                  const std::string &Mode = std::string("NORMAL"));

    template<typename T>
    T Select(const blitz::Array<T, 1> &arr, int KThSmallest);

    template<typename T>
    blitz::Array<T, 1> BubbleSort(const blitz::Array<T, 1> &T_A1_ArrIn);

    /**
     *      BandSol(blitz::Array<double, 2> &D_A2_A_In, blitz::Array<double, 1> &D_A1_R_In, int N, int I_ND) const
     *
     *      bandsol solve a sparse system of linear equations with
     *      band-diagonal matrix.
     *      Band is assumed to be symmetrix relative to the main diaginal.
     *      Usage:
     *      bandsol(D_A2_A_In, D_A1_R_In, I_N, I_ND)
     *      where D_A2_A_In is 2D array [I_N,m] where I_N - is the number
     *      of equations and I_ND is the width of the band (3 for
     *      tri-diagonal system), I_ND is always an odd number. The main
     *      diagonal should be in D_A2_A_In(*,I_ND/2)
     *      The first lower subdiagonal should be in D_A2_A_In(1:I_N-1,I_ND-2-1),
     *      the first upper subdiagonal is in D_A2_A_In(0:I_N-2,I_ND/2+1)
     *      etc. For example:
     *                    / 0 0 X X X \
     *                    | 0 X X X X |
     *                    | X X X X X |
     *                    | X X X X X |
     *      D_A2_A_In =   | X X X X X |
     *                    | X X X X X |
     *                    | X X X X X |
     *                    | X X X X 0 |
     *                    \ X X X 0 0 /
     *      D_A1_R_In is the array of RHS of size I_N.
     *      Argv: blitz::Array<double, 2> &D_A2_A_InOut, blitz::Array<double, 1> &D_A1_R_InOut, int N, int I_ND
     **/
    void BandSol(int Argc, void *Argv[]);

    /**
     *       bool TriDag
     *       Solves for a vector blitz::Array<double, N> UVecArr the tridiagonal
     *       linear set given by equation
     * [ b_1  c_1  0  ...                       ]   [  u_1  ]   [  r_1  ]
     * [ a_2  b_2  c_2 ...                      ]   [  u_2  ]   [  r_2  ]
     * [            ...                         ] * [  ...  ] = [  ...  ]
     * [            ...  a_(N-1) b_(N-1) c_(N-1)]   [u_(N-1)]   [r_(N-1)]
     * [            ...     0     a_N      b_N  ]   [  u_N  ]   [  r_N  ]
     *      BVecArr, CVecArr, and RVecArr are input vectors and are not
     *      modified.
     **/
    bool TriDag(blitz::Array<double, 1> &AVecArr,
                blitz::Array<double, 1> &BVecArr,
                blitz::Array<double, 1> &CVecArr,
                blitz::Array<double, 1> &RVecArr,
                blitz::Array<double, 1> &UVecArr);
    /**
     *        NAME:
     *            UNIQ
     *
     *        PURPOSE:
     *            Return the subscripts of the unique elements in an array.
     *
     *            This command is inspired by the Unix uniq(1) command.
     *
     *        CATEGORY:
     *            Array manipulation.
     *
     *        CALLING SEQUENCE:
     *            Uniq(blitz::Array<int, 1> IA1_In, blitz::Array<int, 1> IA1_Out)
     *
     *        INPUTS:
     *            blitz::Array<int, 1> IA1_In:  The array to be scanned. The number of dimensions of the array is not important.  The array must be sorted into monotonic order.
     *
     *        OUTPUTS:
     *            An array of indicies into ARRAY (blitz::Array<int, 1> IA1_Out) is returned.  The expression:
     *
     *            Uniq(IA1_In, IA1_Out);
     *            blitz::Array<int, 1> SubArr(this->GetSubArr(IA1_In, I_A1_Out))
     *
     *        will be a copy of the sorted Array with duplicate adjacent elements removed.
     *
     **/
    template<typename T>
    bool Uniq(const blitz::Array<T, 1> &Vec_In,
              blitz::Array<int, 1> &Ind_Out);
    //template<typename T>
    //bool Uniq(const std::vector<T> &Vec_In,
    //          std::vector<int> &Ind_Out);

//    template<typename T>
//    void resize(blitz::Array<T, 1> &arr_InOut, unsigned int newSize);

    template<typename T>
    blitz::Array<double,1> Moment(const blitz::Array<T, 1> &D_A1_Arr_In, int I_MaxMoment_In);

    template<typename T>
    blitz::Array<double,1> Moment(const blitz::Array<T, 2> &D_A1_Arr_In, int I_MaxMoment_In);

    template< typename ImageT, typename MaskT, typename VarianceT>
    PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) MkSlitFunc(PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) const& trace,
                                                                                ndarray::Array<float const, 1, 1> const& xCenters,
                                                                                blitz::Array<size_t, 2> const& binBoundY,
                                                                                PTR(pfsDRPStella::FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl,
                                                                                PTR(afwImage::Image<float>) & profile);
    
    template< typename ImageT, typename MaskT, typename VarianceT>
    PTR(pfsDRPStella::Spectrum<ImageT, MaskT, VarianceT, VarianceT>) MkSlitFunc(PTR(afwImage::MaskedImage<ImageT, MaskT, VarianceT>) const& trace,
                                                                                ndarray::Array<float const, 1, 1> const& xCenters,
                                                                                blitz::Array<size_t, 2> const& binBoundY,
                                                                                PTR(pfsDRPStella::FiberTraceProfileFittingControl) const& fiberTraceProfileFittingControl,
                                                                                PTR(afwImage::Image<float>) & profile,
                                                                                const blitz::Array<string, 1> &S_A1_Args_In,
                                                                                void *ArgV_In[]);
    /* S_A1_Args_In:
     * FIBERTRACENUMBER: size_t
     * */
    
    bool SlitFunc(const blitz::Array<double, 2> &D_A2_ImM,
                  unsigned int maxIterSig_In,
                  const blitz::Array<double, 1> &xCentersPixelFraction_In, //: in
                  const PTR(pfs::drp::stella::FiberTraceProfileFittingControl) &profileFittingControl,
                  blitz::Array<double, 1> &spectrum_Out,   //: out
                  blitz::Array<double, 2> &profile_Out,   //: out
                  const blitz::Array<string, 1> &S_A1_Args_In,   //: in
                  void *ArgV_In[]);
    
  }
}}}
#endif
