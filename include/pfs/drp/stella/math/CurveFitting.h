#ifndef __PFS_DRP_STELLA_MATH_CURVEFITTING_H__
#define __PFS_DRP_STELLA_MATH_CURVEFITTING_H__

#include <vector>
#include <iostream>
#include "lsst/base.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/config.h"
//#include "../blitz.h"
#include "../utils/Utils.h"
#include "Math.h"
#include "ndarray.h"
#include "ndarray/eigen.h"

//#define __DEBUG_FIT__
//#define __DEBUG_FITARR__
//#define __DEBUG_POLY__
//#define __DEBUG_POLYFIT__

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
using namespace std;

namespace pfs { namespace drp { namespace stella {
  namespace math{
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
    template<typename T, typename U>
    ndarray::Array<T, 1, 1> Poly(ndarray::Array<T, 1, 1> const& x_In,
                                 ndarray::Array<U, 1, 1> const& coeffs_In);

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

    template< typename T >
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In,
                                         std::vector<string> const& S_A1_Args_In,
                                         std::vector<void *> & ArgV);

    template< typename T >
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In);

/** Additional Keywords:
    REJECTED=blitz::Array<int, 1>
    NOT_REJECTED=blitz::Array<int, 1>
    N_REJECTED=int
    **/
    template<typename T>
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In,
                                         T const D_Reject_In,
                                         std::vector<string> const& S_A1_Args_In,
                                         std::vector<void *> & ArgV);

    template< typename T >
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                    ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                    size_t const I_Degree_In,
                                    T const D_LReject_In,
                                    T const D_UReject_In,
                                    size_t const I_NIter,
                                    std::vector<string> const& S_A1_Args_In,
                                    std::vector<void *> & ArgV);

    template< typename T >
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In,
                                         T const D_Reject_In);

    template< typename T>
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In,
                                         T const D_LReject_In,
                                         T const D_HReject_In,
                                         size_t const I_NIter);

    /**
       CHANGES to original function:
         * D_Sky_Out must be >= 0. unless stated otherwise by the ALLOW_SKY_LT_ZERO parameter
         * D_SP_Out must be >= 0. unless stated otherwise by the ALLOW_SPEC_LT_ZERO parameter
         * added REJECT_IN as optinal parameter to reject cosmic rays from fit (times sigma)
         * added MASK_INOUT as optional parameter
     **/
      
      template< typename ImageT, typename SlitFuncT >
      int LinFitBevingtonEigen(Eigen::Array<ImageT, Eigen::Dynamic, 1> const& D_A1_CCD_In,      /// yvec: in
                               Eigen::Array<SlitFuncT, Eigen::Dynamic, 1> const& D_A1_SF_In,       /// xvec: in
                               ImageT &D_SP_Out,                         /// a1: out
                               ImageT &D_Sky_Out,                        /// a0: in/out
                               bool B_WithSky,                        /// with sky: in
                               std::vector<string> const& S_A1_Args_In,   ///: in
                               std::vector<void *> & ArgV_In);                   ///: in
    /// MEASURE_ERRORS_IN = Eigen::Array<ImageT, Eigen::Dynamic, 1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = float                                                : in
    /// MASK_INOUT = Eigen::Array<unsigned short, Eigen::Dynamic, 1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = ImageT                                                : out
    /// Q_OUT = float                                                    : out
    /// SIGMA_OUT = Eigen::Array<ImageT, Eigen::Dynamic, 1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = ndarray::Array<ImageT, 1>(D_A1_CCD_In.size)                     : out
    /// ALLOW_SKY_LT_ZERO = int[0,1]
    /// ALLOW_SPEC_LT_ZERO = int[0,1]
      
      template< typename ImageT, typename SlitFuncT >
      int LinFitBevingtonNdArray(ndarray::Array<ImageT, 1, 1> const& D_A1_CCD_In,      /// yvec: in
                                 ndarray::Array<SlitFuncT, 1, 1> const& D_A1_SF_In,       /// xvec: in
                                 ImageT &D_SP_Out,                         /// a1: out
                                 ImageT &D_Sky_Out,                        /// a0: in/out
                                 bool B_WithSky,                        /// with sky: in
                                 std::vector<string> const& S_A1_Args_In,   ///: in
                                 std::vector<void *> & ArgV_In);                   ///: in
    /// MEASURE_ERRORS_IN = blitz::Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = float                                                : in
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
  /// MEASURE_ERRORS_IN = blitz::Array<double,2>(D_A2_CCD_In.rows, D_A2_CCD_In.cols) : in
  /// REJECT_IN = double                                                      : in
  /// MASK_INOUT = blitz::Array<double,2>(D_A1_CCD_In.rows,D_A1_CCD_In.cols)         : in/out
  /// CHISQ_OUT = blitz::Array<double,1>(D_A2_CCD_In.rows)                           : out
  /// Q_OUT = blitz::Array<double,1>(D_A2_CCD_In.rows)                               : out
  /// SIGMA_OUT = blitz::Array<double,2>(D_A2_CCD_In.rows, 2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
      template< typename ImageT, typename SlitFuncT>
      bool LinFitBevingtonEigen(Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic> const& D_A2_CCD_In,      ///: in
                                Eigen::Array<SlitFuncT, Eigen::Dynamic, Eigen::Dynamic> const& D_A2_SF_In,       ///: in
                                Eigen::Array<ImageT, Eigen::Dynamic, 1> & D_A1_SP_Out,             ///: out
                                Eigen::Array<ImageT, Eigen::Dynamic, 1> & D_A1_Sky_Out,            ///: in/out
                                bool B_WithSky,                           ///: with sky: in
                                vector<string> const& S_A1_Args_In,   ///: in
                                vector<void *> &ArgV_In);                   ///: in/out
      template< typename ImageT, typename SlitFuncT>
      bool LinFitBevingtonNdArray(ndarray::Array<ImageT, 2, 1> const& D_A2_CCD_In,      ///: in
                                  ndarray::Array<SlitFuncT, 2, 1> const& D_A2_SF_In,       ///: in
                                  ndarray::Array<ImageT, 1, 1> & D_A1_SP_Out,             ///: out
                                  ndarray::Array<ImageT, 1, 1> & D_A1_Sky_Out,            ///: in/out
                                  bool B_WithSky,                           ///: with sky: in
                                  vector<string> const& S_A1_Args_In,   ///: in
                                  vector<void *> &ArgV_In);                   ///: in/out
    /// MEASURE_ERRORS_IN = blitz::Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = double                                                : in
    /// MASK_INOUT = blitz::Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = blitz::Array<double,1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = blitz::Array<double, 1>(D_A1_CCD_In.size)                     : out

    /**
     *       Helper function to calculate incomplete Gamma Function
     **/
    template< typename T >
    T GammLn(T const D_X_In);

    /**
     *      Helper function to calculate incomplete Gamma Function
     **/
    template< typename T >
    T GCF(T & D_Gamser_In, T const a, T const x);

    /**
     *      Function to calculate incomplete Gamma Function P
     **/
    template< typename T>
    T GammP(T const a, T const x);

    /**
     *      Function to calculate incomplete Gamma Function Q = 1 - P
     **/
    template<typename T>
    T GammQ(T const a, T const x);

    /**
     *      Helper function to calculate incomplete Gamma Function
     **/
    template< typename T >
    T GSER(T & D_Gamser_In, T const a, T const x);

      
  }
}}}
#endif
