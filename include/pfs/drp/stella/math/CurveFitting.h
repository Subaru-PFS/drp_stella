#ifndef __PFS_DRP_STELLA_MATH_CURVEFITTING_H__
#define __PFS_DRP_STELLA_MATH_CURVEFITTING_H__

#include <string>
#include <vector>

#include "ndarray.h"

namespace pfs { namespace drp { namespace stella {
  namespace math {
      
    template< typename T >
    struct SpectrumBackground { 
        T spectrum;
        T background;
    };

    struct GaussCoeffs {
        float constantBackground;
        float linearBackground;
        float mu;
        float sigma;
        float strength;

        /**
         * @brief Standard Constructor
         */
        GaussCoeffs():
        constantBackground(0.0),
        linearBackground(0.0),
        mu(0.0),
        sigma(0.0),
        strength(0.0)
        {}

        /**
         * @brief Copy Constructor
         * @param gaussCoeffs : GaussCoeffs to be copied to this
         */
        GaussCoeffs(GaussCoeffs const& gaussCoeffs):
        constantBackground(gaussCoeffs.constantBackground),
        linearBackground(gaussCoeffs.linearBackground),
        mu(gaussCoeffs.mu),
        sigma(gaussCoeffs.sigma),
        strength(gaussCoeffs.strength)
        {}

        /**
         * @brief Destructor
         */
        ~GaussCoeffs(){}

        /**
         * @brief Return a ndarray<float> of shape(5) containing the coefficients
         * @return ndarray<float>(5) [0: strength, 1: mu, 2: sigma, 3: constant background,
         *                            4: linear background]
         */
        ndarray::Array<float, 1, 1> toNdArray();

        /**
         * @brief convert an ndarray of shape>=3 to coefficients in this
         * @param coeffs : ndarray containing the coefficients
         *                 [0: strength, 1: mu, 2: sigma, (3: constant background,
         *                  (4: linear background))]
         */
        template<typename T>
        void set(ndarray::Array<T, 1, 1> const& coeffs);
    };

    /**
     * @brief write GaussCoeffs to basic_ostream
     * @param os : basic_ostream to be written to
     * @param coeffs : GaussCoeffs to be written to basic_ostream
     * @return basic_ostream
     */
    std::ostream& operator<< (std::ostream& os, const GaussCoeffs& coeffs);
      
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
     *   CHISQ=chisq: float: out:
     *     Sum of squared errors divided by MEASURE_ERRORS if specified.
     *
     *   COVAR=covar: ndarray::Array<float, 2, 2>(I_Degree+1, I_Degree+1): out:
     *     Covariance matrix of the coefficients.
     *
     *   MEASURE_ERRORS=measure_errors: ndarray::Array<float, 1, 1>(D_A1_X_In.size()): in:
     *     Set this keyword to a vector containing standard
     *     measurement errors for each point Y[i].  This vector must be the same
     *     length as X and Y.
     *
     *     Note - For Gaussian errors (e.g. instrumental uncertainties),
     *       MEASURE_ERRORS should be set to the standard
     *       deviations of each point in Y. For Poisson or statistical weighting
     *       MEASURE_ERRORS should be set to sqrt(Y).
     *
     *   SIGMA=sigma: ndarray::Array<float, 1, 1>(I_Degree+1): out:
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
     *   YFIT:   ndarray::Array<float, 1, 1>(D_A1_X_In.size()) of calculated Y's. These values have an error
     *           of + or - YBAND.
     * 
     *   XRANGE: x range from which the original x_In values are from
     *           x will be rescaled from [xrange[0], xrange[1]] to [-1.,1.]
     *
    CHISQ=chisq: float: out
    COVAR=covar: PTR(ndarray::Array<float, 2, 1>(I_Degree+1, I_Degree+1)): out
    MEASURE_ERRORS=measure_errors: PTR(ndarray::Array<float, 1, 1>(D_A1_X_In.size())): in
    SIGMA=sigma: PTR(ndarray::Array<float, 1, 1>(I_Degree+1)): out
    STATUS=status: int: out
    YFIT=yfit: PTR(ndarray::Array<T, 1, 1>(D_A1_X_In.size())): out
    XRANGE: PTR(ndarray::Array<float, 1, 1>(2)): in
    YERROR=yerror
    LSIGMA=lsigma: lower sigma rejection threshold
    USIGMA=usigma: upper sigma rejection threshold
    MIN_ERR=minErr: T minimum error, replace errors smaller than this by this number
    **/
    template< typename T >
    ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& x_In,
                                        ndarray::Array<T, 1, 1> const& y_In,
                                        size_t const degree_In,
                                        std::vector<std::string> const& argsKeyWords_In,
                                        std::vector<void *> & argsValues_In);

    template< typename T >
    ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In,
                                         T xRangeMin_In = -1.,
                                         T xRangeMax_In = 1.);

/** Additional Keywords:
    REJECTED=vector<int>
    NOT_REJECTED=vector<int>
    N_REJECTED=int
    **/
    template<typename T>
    ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                        ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                        size_t const I_Degree_In,
                                        T const D_Reject_In,
                                        std::vector<std::string> const& S_A1_Args_In,
                                        std::vector<void *> & ArgV);

    template< typename T >
    ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                        ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                        size_t const I_Degree_In,
                                        T const D_LReject_In,
                                        T const D_UReject_In,
                                        size_t const I_NIter,
                                        std::vector<std::string> const& S_A1_Args_In,
                                        std::vector<void *> & ArgV);

    template< typename T>
    ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
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
      SpectrumBackground< ImageT > LinFitBevingtonNdArray( ndarray::Array<ImageT, 1, 1> const& D_A1_CCD_In,
                                                           ndarray::Array<SlitFuncT, 1, 1> const& D_A1_SF_In,
                                                           bool B_WithSky);
      
      template< typename ImageT, typename SlitFuncT >
      int LinFitBevingtonNdArray(ndarray::Array<ImageT, 1, 1> const& D_A1_CCD_In,
                                 ndarray::Array<SlitFuncT, 1, 1> const& D_A1_SF_In,
                                 ImageT &D_SP_Out,
                                 ImageT &D_Sky_Out,
                                 bool B_WithSky,
                                 std::vector<std::string> const& S_A1_Args_In,
                                 std::vector<void *> & ArgV_In);
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
     *                MEASURE_ERRORS_IN = ndarray::Array<ImageT,2,1>(D_A2_CCD_In.getShape())   : in
     *                REJECT_IN = ImageT                                                       : in
     *                MASK_INOUT = ndarray::Array<unsigned short, 1, 1>(D_A2_CCD_In.getShape()): in/out
     *                CHISQ_OUT = ndarray::Array<ImageT, 1, 1>(D_A2_CCD_In.getShape()[0])      : out
     *                Q_OUT = ndarray::Array<ImageT, 1, 1>(D_A2_CCD_In.getShape()[0])          : out
     *                SIGMA_OUT = ndarray::Array<ImageT, 2, 1>(D_A2_CCD_In.getShape()[0], 2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
     *                YFIT_OUT = ndarray::Array<ImageT, 2, 1>(D_A2_CCD_In.getShape()[0], D_A2_CCD_In.getShape()[1]) : out
     * @param[in,out]   ArgV_In        :: Vector of keyword values
     * */
      template< typename ImageT, typename SlitFuncT>
      bool LinFitBevingtonNdArray(ndarray::Array<ImageT, 2, 1> const& D_A2_CCD_In,
                                  ndarray::Array<SlitFuncT, 2, 1> const& D_A2_SF_In,
                                  ndarray::Array<ImageT, 1, 1> & D_A1_SP_Out,
                                  ndarray::Array<ImageT, 1, 1> & D_A1_Sky_Out,
                                  bool B_WithSky,
                                  std::vector<std::string> const& S_A1_Args_In,
                                  std::vector<void *> &ArgV_In);

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
     * @brief fit a Gaussian to noisy data using Levenberg Marquardt
     * @param xy_In 2d array (ndata, 2) [*][0]: x, [*][1]: y
     * @param guess_In [0]: peak, [1]: center, [2]: sigma
     */
    ndarray::Array<float, 1, 1> gaussFit(ndarray::Array<float, 2, 1> const& xy_In,
                                         ndarray::Array<float, 1, 1> const& guess_In);

  }
}}}
#endif
