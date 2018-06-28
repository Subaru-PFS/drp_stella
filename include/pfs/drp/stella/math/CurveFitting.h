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
 *
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


/// Results from fitting a polynomial
template <typename T>
struct PolynomialFitResults {
    using CoeffT = double;
    using Array = ndarray::Array<T, 1, 1>;
    using CoeffArray = ndarray::Array<double, 1, 1>;
    using BoolArray = ndarray::Array<bool, 1, 1>;
    using Matrix = ndarray::Array<double, 2, 2>;

    CoeffT chi2;  ///< chi^2 from the fit

    // Data-related
    Array yFit; ///< Fit values
    BoolArray rejected;  ///< Whether points were rejected; 'true' means rejected
    std::size_t numRejected;  ///< Number of rejected points

    // Coefficient-related
    CoeffArray coeffs;  ///< Fit coefficients
    Array sigma;  ///< Estimated errors in fit coefficients
    Matrix covar;  ///< Covariance matrix for fit coefficients

    PolynomialFitResults(std::size_t numData, std::size_t numCoeffs) :
        chi2(std::numeric_limits<CoeffT>::quiet_NaN()),
        yFit(ndarray::allocate(numData)),
        rejected(ndarray::allocate(numData)),
        numRejected(0),
        coeffs(ndarray::allocate(numCoeffs)),
        sigma(ndarray::allocate(numCoeffs)),
        covar(ndarray::allocate(numCoeffs, numCoeffs)) {
            rejected.deep() = false;
        }

    PolynomialFitResults(PolynomialFitResults const&) = default;
    PolynomialFitResults& operator=(PolynomialFitResults const&) = default;
    PolynomialFitResults(PolynomialFitResults &&) = default;
    PolynomialFitResults& operator=(PolynomialFitResults &&) = default;
};


/// Parameters controlling polynomial fit
template <typename T>
struct PolynomialFitControl {
    T xRangeMin;  ///< Minimum for normalised x range (e.g., -1)
    T xRangeMax;  ///< Maximum for normalised x range (e.g., +1)
    T rejectLow;  ///< Low rejection threshold (factor of standard deviation)
    T rejectHigh;  ///< High rejection threshold (factor of standard deviation)
    int nIter;  ///< Number of rejection iterations

    PolynomialFitControl(T xRangeMin_, T xRangeMax_, T rejectLow_, T rejectHigh_, int nIter_) :
        xRangeMin(xRangeMin_), xRangeMax(xRangeMax_), rejectLow(rejectLow_), rejectHigh(rejectHigh_),
        nIter(nIter_)
        {}
    explicit PolynomialFitControl(T xRangeMin_=-1.0, T xRangeMax_=1.0) :
        PolynomialFitControl(xRangeMin_, xRangeMax_,
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


/// Fit polynomial, with scaling and rejection
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


/// Fit polynomial, with scaling
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


/***
 * @brief  Fit the spatial profile to a FiberTrace
 *
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
 * Evaluate Chebyshev polynomial at positions
 */
template <typename T, typename U>
ndarray::Array<T, 1, 1> calculateChebyshev(
    ndarray::Array<T, 1, 1> const& x,  ///< Values at which to evaluate Chebyshev
    ndarray::Array<U, 1, 1> const& coeffs,  ///< Chebyshev polynomial coefficients
    T xRangeMin=-1.0,  ///< Minimum for normalised range
    T xRangeMax=1.0  ///< Maximum for normalised range
);

/*
 * @brief fit a Gaussian to noisy data using Levenberg Marquardt
 *
 * @param xy  2d array (ndata, 2) [*][0]: x, [*][1]: y
 * @param guess  [0]: peak, [1]: center, [2]: sigma
 */
ndarray::Array<float, 1, 1> gaussFit(ndarray::Array<float, 2, 1> const& xy,
                                     ndarray::Array<float, 1, 1> const& guess);

}}}} // namespace pfs::drp::stella::math

#endif
