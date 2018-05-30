#ifndef __PFS_DRP_STELLA_MATH_CURVEFITTING_H__
#define __PFS_DRP_STELLA_MATH_CURVEFITTING_H__

#include <string>
#include <vector>

#include "ndarray.h"
#include "lsst/afw/image/MaskedImage.h"

namespace pfs {
namespace drp {
namespace stella {
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
template <typename T, typename U>
ndarray::Array<T, 1, 1> calculatePolynomial(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<U, 1, 1> const& coeffs,
    T xRangeMin_In=-1.,
    T xRangeMax_In=1.
);


template <typename T>
struct PolynomialFitResults {
    using CoeffT = double;
    using Array = ndarray::Array<T, 1, 1>;
    using CoeffArray = ndarray::Array<double, 1, 1>;
    using BoolArray = ndarray::Array<bool, 1, 1>;
    using Matrix = ndarray::Array<double, 2, 2>;

    CoeffT chi2;

    // Data-related
    Array yFit;
    BoolArray rejected;

    // Coefficient-related
    CoeffArray coeffs;
    Array sigma;
    Matrix covar;
    std::size_t numRejected;

    PolynomialFitResults(std::size_t numData, std::size_t numCoeffs) :
        chi2(std::numeric_limits<CoeffT>::quiet_NaN()),
        yFit(ndarray::allocate(numData)),
        rejected(ndarray::allocate(numData)),
        coeffs(ndarray::allocate(numCoeffs)),
        sigma(ndarray::allocate(numCoeffs)),
        covar(ndarray::allocate(numCoeffs, numCoeffs)),
        numRejected(0) {
            rejected.deep() = false;
        }

    PolynomialFitResults(PolynomialFitResults const&) = default;
    PolynomialFitResults& operator=(PolynomialFitResults const&) = default;
    PolynomialFitResults(PolynomialFitResults &&) = default;
    PolynomialFitResults& operator=(PolynomialFitResults &&) = default;
};


template <typename T>
struct PolynomialFitControl {
    T xRangeMin;
    T xRangeMax;
    T rejectLow;
    T rejectHigh;
    int nIter;

    PolynomialFitControl(T xRangeMin_, T xRangeMax_, T rejectLow_, T rejectHigh_, int nIter_) :
        xRangeMin(xRangeMin_), xRangeMax(xRangeMax_), rejectLow(rejectLow_), rejectHigh(rejectHigh_),
        nIter(nIter_)
        {}
    explicit PolynomialFitControl(T xRangeMin_=-1.0, T xRangeMax_=1.0) :
        PolynomialFitControl(xRangeMin_, xRangeMax,
                             -std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(),
                             0)
        {}
};


/// Vanilla fit polynomial, no rejection applied
template <typename T>
PolynomialFitResults<T> fitPolynomial(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<T, 1, 1> const& y,
    std::size_t degree,
    ndarray::Array<T, 1, 1> const& yErr=ndarray::Array<T, 1, 1>(),
    PolynomialFitResults<T> *results=nullptr
);


/// Fit polynomial with optional scaling and rejection
template <typename T>
PolynomialFitResults<T> fitPolynomial(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<T, 1, 1> const& y,
    std::size_t degree,
    PolynomialFitControl<T> const& ctrl,
    ndarray::Array<T, 1, 1> const& yErr=ndarray::Array<T, 1, 1>()
);

template <typename T>
PolynomialFitResults<T> fitPolynomial(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<T, 1, 1> const& y,
    std::size_t degree,
    T xRangeMin,
    T xRangeMax,
    T rejectLow,
    T rejectHigh,
    int nIter,
    ndarray::Array<T, 1, 1> const& yErr=ndarray::Array<T, 1, 1>()
) {
    PolynomialFitControl<T> ctrl{xRangeMin, xRangeMax, rejectLow, rejectHigh, nIter};
    return fitPolynomial(x, y, degree, ctrl, yErr);
}

template <typename T>
PolynomialFitResults<T> fitPolynomial(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<T, 1, 1> const& y,
    std::size_t degree,
    T xRangeMin,
    T xRangeMax,
    ndarray::Array<T, 1, 1> const& yErr=ndarray::Array<T, 1, 1>()
) {
    PolynomialFitControl<T> ctrl{xRangeMin, xRangeMax};
    return fitPolynomial(x, y, degree, ctrl, yErr);
}

#if 0
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
template <typename T>
ndarray::Array<float, 1, 1> fitPolynomial(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<T, 1, 1> const& y,
    std::size_t degree,
    std::vector<std::string> const& argKeywords,
    std::vector<void *> & argValues
);

template <typename T>
ndarray::Array<float, 1, 1> fitPolynomial(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<T, 1, 1> const& y,
    std::size_t degree,
    T xRangeMin_In=-1.,
    T xRangeMax_In=1.
);

/** Additional Keywords:
REJECTED=vector<int>
NOT_REJECTED=vector<int>
N_REJECTED=int
**/
template <typename T>
ndarray::Array<float, 1, 1> fitPolynomial(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<T, 1, 1> const& y,
    std::size_t degree,
    T reject,
    std::vector<std::string> const& argKeywords,
    std::vector<void *> & argValues
) {
    return fitPolynomial(x, y, degree, -reject, reject, -1, argKeywords, argValues);
}

template <typename T>
ndarray::Array<float, 1, 1> fitPolynomial(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<T, 1, 1> const& y,
    std::size_t degree,
    T const D_LReject_In,
    T const D_UReject_In,
    std::size_t I_NIter,
    std::vector<std::string> const& argKeywords,
    std::vector<void *> & argValues
);

template< typename T>
ndarray::Array<float, 1, 1> fitPolynomial(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<T, 1, 1> const& y,
    std::size_t degree,
    T lowReject,
    T highReject,
    std::size_t nIter,
    T xRangeMin_In=-1.,
    T xRangeMax_In=1.
);

#endif

/***
 * @brief  Fit the spatial profile to a FiberTrace
 * calculates  D_SP_Out for the system of equations ccdData = D_SP_Out*D_A1_SF_In + D_Bkgd_Out
 */
template<typename ImageT>
std::tuple<ndarray::Array<ImageT, 1, 1>, // extracted spectrum
           ndarray::Array<ImageT, 1, 1>, // background
           ndarray::Array<ImageT, 1, 1> // variance
           >
fitProfile2d(
    lsst::afw::image::MaskedImage<ImageT> const& data,  ///< the input data containing the spectrum
    ndarray::Array<bool, 2, 1> const& traceMask,  ///< true for points in the fiberTrace
    ndarray::Array<ImageT, 2, 1> const& profile2d,  ///< profile of fibre trace
    bool fitBackground,  ///< should I fit the background level?
    float clipNSigma,  ///< clip at this many sigma
    std::vector<std::string> const& badMaskPlanes={"BAD", "SAT"} ///< mask planes to reject
);

/**
 */
template <typename T, typename U>
ndarray::Array<T, 1, 1> calculateChebyshev(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<U, 1, 1> const& coeffs,
    T xRangeMin=-1.0,
    T xRangeMax=1.0
);

/*
 * @brief fit a Gaussian to noisy data using Levenberg Marquardt
 * @param xy_In 2d array (ndata, 2) [*][0]: x, [*][1]: y
 * @param guess_In [0]: peak, [1]: center, [2]: sigma
 */
ndarray::Array<float, 1, 1> gaussFit(ndarray::Array<float, 2, 1> const& xy,
                                     ndarray::Array<float, 1, 1> const& guess);

}}}} // namespace pfs::drp::stella::math

#endif
