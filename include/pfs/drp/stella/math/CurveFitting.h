#ifndef __PFS_DRP_STELLA_MATH_CURVEFITTING_H__
#define __PFS_DRP_STELLA_MATH_CURVEFITTING_H__

#include <vector>
#include <iostream>
#include "lsst/base.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/config.h"
#include "lsst/pex/exceptions/Exception.h"
#include "../utils/Utils.h"
#include "Math.h"
#include "ndarray.h"
#include "ndarray/eigen.h"
#include "CurveFittingGaussian.h"

//#define __DEBUG_CURVEFIT__
//#define __DEBUG_FIT__
//#define __DEBUG_FITARR__
//#define __DEBUG_POLY__
//#define __DEBUG_POLYFIT__

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
using namespace std;

namespace pfs { namespace drp { namespace stella {
  namespace math{
      
    template< typename T >
    struct SpectrumBackground { 
        T spectrum;
        T background;
    };
      
    /*
     * @brief calculate y positions for given x positions and a polynomial of given coefficients
     * @param x_In: given x positions for which the y positions shall be calculated
     * @param coeffs_In: given polynomial coefficients. The degree of the polynomial is coeffs_In.shape[0] - 1
     * @param xRangeMin_In: minimum range from where x was coming from when the polynomial coefficients were fitted (default = -1.)
     * @param xRangeMax_In: maximum range from where x was coming from when the polynomial coefficients were fitted (default = +1.)
     * 
     *       POLY returns a result equal to:
     *                C[0] + c[1] * X + c[2]*X^2 + ...
     *       with X shifted and rescaled to fit in the range [-1,1]
     *
     */
    template<typename T, typename U>
    ndarray::Array<T, 1, 1> Poly(ndarray::Array<T, 1, 1> const& x_In,
                                 ndarray::Array<U, 1, 1> const& coeffs_In,
                                 T xRangeMin_In = -1.,
                                 T xRangeMax_In = 1.);
//    template<typename T, typename U>
//    ndarray::Array<T, 1, 1> Poly(ndarray::Array<T, 1, 1> const& x_In,
//                                 ndarray::Array<U, 1, 1> const& coeffs_In,
//                                 ndarray::Array<double, 1, 1> const& xRange_In);///shift and rescale x_In to fit into specified range

    /**
     * @brief  Perform a least-square polynomial fit using matrix inversion with optional error estimates.
     *
     * @param  x_In:  The independent variable vector.
     * @param  y_In:  The dependent variable vector, should be same length as x_In.
     * @param  degree_In: The degree of the polynomial to fit.
     * @param argsKeyWords_In: vector of keywords used to control the behavior of the function (See below)
     * @param argsValues_In: Keyword values corresponding to the keywords in argsKeyWords_In
     *
     * OUTPUTS:
     *   POLY_FIT returns a vector of coefficients with a length of NDegree+1.
     *
     * NOTES: * x_In will be shifted and rescaled to fit into the range [-1,1]
     *          if XRANGE is given the original x range will be assumed to be [xrange[0], xrange[1]],
     *          otherwise [min(x_In), max(x_In)]
     * 
     * KEYWORDS and values:
     *   CHISQ=chisq: double: out:
     *     Sum of squared errors divided by MEASURE_ERRORS if specified.
     *
     *   COVAR=covar: ndarray::Array<double, 2, 2>(I_Degree+1, I_Degree+1): out:
     *     Covariance matrix of the coefficients.
     *
     *   MEASURE_ERRORS=measure_errors: ndarray::Array<double, 1, 1>(D_A1_X_In.size()): in:
     *     Set this keyword to a vector containing standard
     *     measurement errors for each point Y[i].  This vector must be the same
     *     length as X and Y.
     *
     *     Note - For Gaussian errors (e.g. instrumental uncertainties),
     *       MEASURE_ERRORS should be set to the standard
     *       deviations of each point in Y. For Poisson or statistical weighting
     *       MEASURE_ERRORS should be set to sqrt(Y).
     *
     *   SIGMA=sigma: ndarray::Array<double, 1, 1>(I_Degree+1): out:
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
     *   YFIT:   ndarray::Array<double, 1, 1>(D_A1_X_In.size()) of calculated Y's. These values have an error
     *           of + or - YBAND.
     * 
     *   XRANGE: x range from which the original x_In values are from
     *           x will be rescaled from [xrange[0], xrange[1]] to [-1.,1.]
     *
    CHISQ=chisq: double: out
    COVAR=covar: PTR(ndarray::Array<double, 2, 1>(I_Degree+1, I_Degree+1)): out
    MEASURE_ERRORS=measure_errors: PTR(ndarray::Array<double, 1, 1>(D_A1_X_In.size())): in
    SIGMA=sigma: PTR(ndarray::Array<double, 1, 1>(I_Degree+1)): out
    STATUS=status: int: out
    YFIT=yfit: PTR(ndarray::Array<T, 1, 1>(D_A1_X_In.size())): out
    XRANGE: PTR(ndarray::Array<double, 1, 1>(2)): in
    YERROR=yerror
    LSIGMA=lsigma: lower sigma rejection threshold
    USIGMA=usigma:
    **/
    template< typename T >
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& x_In,
                                         ndarray::Array<T, 1, 1> const& y_In,
                                         size_t const degree_In,
                                         std::vector<string> const& argsKeyWords_In,
                                         std::vector<void *> & argsValues_In);

    template< typename T >
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In,
                                         T xRangeMin_In = -1.,
                                         T xRangeMax_In = 1.);

/** Additional Keywords:
    REJECTED=ndarray::Array<int, 1>
    NOT_REJECTED=ndarray::Array<int, 1>
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

    template< typename T>
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In,
                                         T const D_LReject_In,
                                         T const D_HReject_In,
                                         size_t const I_NIter,
                                         T xRangeMin_In = -1.,
                                         T xRangeMax_In = 1.);

    /**
     * @brief Scale the spatial profile to the FiberTrace row
       calculates D_Sky_Out and D_SP_Out for the system of equations D_A1_CCD_In = D_Sky_Out + D_SP_Out * D_A1_SF_In
       CHANGES to original function:
         * D_Sky_Out must be >= 0. unless stated otherwise by the ALLOW_SKY_LT_ZERO parameter
         * D_SP_Out must be >= 0. unless stated otherwise by the ALLOW_SPEC_LT_ZERO parameter
         * added REJECT_IN as optinal parameter to reject cosmic rays from fit (times sigma)
         * added MASK_INOUT as optional parameter
     * @param[in]       D_A1_CCD_In    :: FiberTrace row to scale the spatial profile D_A1_SF_In to
     * @param[in]       D_A1_SF_In     :: Spatial profile to scale to the FiberTrace row
     * @param[in,out]   D_SP_Out       :: Fitted scaling factor
     * @param[in,out]   D_Sky_Out      :: Fitted constant under the fitted signal
     * @param[in]       B_WithSky      :: Scale spatial profile plus constant background (sky)?
     * @param[in]       S_A1_Args_In   :: Vector of keywords controlling the procedure
     * @param[in,out]   ArgV_In        :: Vector of keyword values
     **/
      
      template< typename ImageT, typename SlitFuncT >
      int LinFitBevingtonEigen(Eigen::Array<ImageT, Eigen::Dynamic, 1> const& D_A1_CCD_In,
                               Eigen::Array<SlitFuncT, Eigen::Dynamic, 1> const& D_A1_SF_In,
                               ImageT &D_SP_Out,                         
                               ImageT &D_Sky_Out,                        
                               bool B_WithSky,                        
                               std::vector<string> const& S_A1_Args_In,
                               std::vector<void *> & ArgV_In);         
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
      SpectrumBackground< ImageT > LinFitBevingtonNdArray( ndarray::Array<ImageT, 1, 1> const& D_A1_CCD_In,
                                                           ndarray::Array<SlitFuncT, 1, 1> const& D_A1_SF_In,
                                                           bool B_WithSky);
      
      template< typename ImageT, typename SlitFuncT >
      int LinFitBevingtonNdArray(ndarray::Array<ImageT, 1, 1> const& D_A1_CCD_In,
                                 ndarray::Array<SlitFuncT, 1, 1> const& D_A1_SF_In,
                                 ImageT &D_SP_Out,
                                 ImageT &D_Sky_Out,
                                 bool B_WithSky,
                                 std::vector<string> const& S_A1_Args_In,
                                 std::vector<void *> & ArgV_In);
    /// MEASURE_ERRORS_IN = PTR(ndarray::Array<ImageT, 1, 1>(D_A1_CCD_In.size))             : in
    /// REJECT_IN = float                                                : in
    /// MASK_INOUT = ndarray::Array<unsigned short, 1, 1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = ndarray::Array<ImageT,1, 1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = ndarray::Array<ImageT, 1, 1>(D_A1_CCD_In.size)                     : out
    /// ALLOW_SKY_LT_ZERO = 1
    /// ALLOW_SPEC_LT_ZERO = 1

    /**
     * @brief Scale the spatial profile to the FiberTrace row
       calculates D_Sky_Out and D_SP_Out for the system of equations D_A1_CCD_In = D_Sky_Out + D_SP_Out * D_A1_SF_In
       CHANGES to original function:
         * D_Sky_Out must be >= 0.
         * D_SP_Out must be >= 0.
         * if D_Sky_Out is set to be < -1.e-10 in input it is set to 0. and D_SP_Out is calculated as if there was no sky at all
         * added REJECT_IN as optinal parameter to reject cosmic rays from fit (times sigma)
         * added MASK_INOUT as optional parameter
     * @param[in]       D_A2_CCD_In    :: FiberTrace row to scale the spatial profile D_A1_SF_In to
     * @param[in]       D_A2_SF_In     :: Spatial profile to scale to the FiberTrace row
     * @param[in,out]   D_A1_SP_Out       :: Fitted scaling factor
     * @param[in,out]   D_A1_Sky_Out      :: Fitted constant under the fitted signal
     * @param[in]       B_WithSky      :: Scale spatial profile plus constant background (sky)?
     * @param[in]       S_A1_Args_In   :: Vector of keywords controlling the procedure
                                            MEASURE_ERRORS_IN = ndarray::Array<double,2>(D_A2_CCD_In.rows, D_A2_CCD_In.cols) : in
                                            REJECT_IN = double                                                      : in
                                            MASK_INOUT = ndarray::Array<double,2>(D_A1_CCD_In.rows,D_A1_CCD_In.cols)         : in/out
                                            CHISQ_OUT = ndarray::Array<double,1>(D_A2_CCD_In.rows)                           : out
                                            Q_OUT = ndarray::Array<double,1>(D_A2_CCD_In.rows)                               : out
                                            SIGMA_OUT = ndarray::Array<double,2>(D_A2_CCD_In.rows, 2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
     * @param[in,out]   ArgV_In        :: Vector of keyword values
     * */
      template< typename ImageT, typename SlitFuncT>
      bool LinFitBevingtonEigen(Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic> const& D_A2_CCD_In,
                                Eigen::Array<SlitFuncT, Eigen::Dynamic, Eigen::Dynamic> const& D_A2_SF_In,
                                Eigen::Array<ImageT, Eigen::Dynamic, 1> & D_A1_SP_Out,
                                Eigen::Array<ImageT, Eigen::Dynamic, 1> & D_A1_Sky_Out,
                                bool B_WithSky,
                                std::vector<string> const& S_A1_Args_In,
                                std::vector<void *> &ArgV_In);
      
      template< typename ImageT, typename SlitFuncT>
      bool LinFitBevingtonNdArray(ndarray::Array<ImageT, 2, 1> const& D_A2_CCD_In,
                                  ndarray::Array<SlitFuncT, 2, 1> const& D_A2_SF_In,
                                  ndarray::Array<ImageT, 1, 1> & D_A1_SP_Out,
                                  ndarray::Array<ImageT, 1, 1> & D_A1_Sky_Out,
                                  bool B_WithSky,
                                  std::vector<string> const& S_A1_Args_In,
                                  std::vector<void *> &ArgV_In);
    /// MEASURE_ERRORS_IN = ndarray::Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = double                                                : in
    /// MASK_INOUT = ndarray::Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = ndarray::Array<double,1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = ndarray::Array<double, 1>(D_A1_CCD_In.size)                     : out

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

    /**
     */
    template< typename T, typename U >
    ndarray::Array<T, 1, 1> chebyshev(ndarray::Array<T, 1, 1> const& x_In, ndarray::Array<U, 1, 1> const& coeffs_In);
                
    /*
     * @brief fit a Gaussian to noisy data using Eigen's Levenberg Marquardt
     * @param xy_In 2d array (ndata, 2) [*][0]: x, [*][1]: y
     * @param guess_In [0]: peak, [1]: center, [2]: sigma
     */
    ndarray::Array<double, 1, 1> gaussFit(ndarray::Array<double, 2, 1> const& xy_In,
                                          ndarray::Array<double, 1, 1> const& guess_In);

  }
}}}
#endif
