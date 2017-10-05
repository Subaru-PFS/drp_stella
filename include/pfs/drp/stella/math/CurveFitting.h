#ifndef __PFS_DRP_STELLA_MATH_CURVEFITTING_H__
#define __PFS_DRP_STELLA_MATH_CURVEFITTING_H__

#include <string>
#include <vector>

#include "ndarray.h"
#include "lsst/afw/image/MaskedImage.h"

namespace pfs { namespace drp { namespace stella {
  namespace math {
      
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
    USIGMA=usigma:
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

    /***
     * @brief  Fit the spatial profile to a FiberTrace
     * calculates  D_SP_Out for the system of equations ccdData = D_SP_Out*D_A1_SF_In + D_Bkgd_Out
     */
  template< typename ImageT>
  bool fitProfile2d(ndarray::Array<ImageT, 2, 1> const& ccdData,                      ///< The image
                    ndarray::Array<ImageT, 2, 1> const& ccdDataVar,                   ///< data's variance
                    ndarray::Array<lsst::afw::image::MaskPixel, 2, 1> const& traceMask, ///< set to 1 for points in the fiberTrace
                    ndarray::Array<ImageT, 2, 1> const& profile2d, ///< profile of fibre trace
                    const float clipNSigma,                        ///< clip at this many sigma
                    ndarray::Array<ImageT, 1, 1> & specAmp,        ///< returned spectrum
                    ndarray::Array<ImageT, 1, 1> & bkgd,           ///< returned background
                    ndarray::Array<ImageT, 1, 1> & specAmpVar      ///< the spectrum's variance
                   );
      
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
