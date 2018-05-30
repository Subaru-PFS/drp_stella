#include "ndarray/eigen.h"

#include "unsupported/Eigen/LevenbergMarquardt"
#include "Eigen/Dense"

#include "lsst/log/Log.h"
#include "lsst/pex/exceptions/Exception.h"
#include "lsst/afw/image/MaskedImage.h"

#include "pfs/drp/stella/math/CurveFitting.h"
#include "pfs/drp/stella/math/CurveFittingGaussian.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/utils/Utils.h"
#include "pfs/drp/stella/utils/checkSize.h"

namespace pexExcept = lsst::pex::exceptions;

namespace pfs {
namespace drp {
namespace stella {
namespace math {

// XXX replace this implementation with lsst::afw::math::PolynomialFunction1
// once we have tests
template<typename T, typename U>
ndarray::Array<T, 1, 1> calculatePolynomial(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<U, 1, 1> const& coeffs,
    T xRangeMin,
    T xRangeMax
) {
    #ifdef __DEBUG_CURVEFIT__
    std::cout << "CurveFitting::Poly(x, coeffs, xRangeMin, xRangeMax) started" << std::endl;
    #endif
    std::size_t const num = x.getNumElements();
    ndarray::Array<T, 1, 1> xNew;
    /// shift and rescale x to fit into range [-1.,1.]
    if ((std::fabs(xRangeMin + 1.) > 0.00000001) || (std::fabs(xRangeMax - 1.) > 0.00000001)) {
        xNew = convertRangeToUnity(x, xRangeMin, xRangeMax);
    } else {
        xNew = x;
    }
    #ifdef __DEBUG_POLY__
    std::cout << "pfs::drp::stella::math::CurveFitting::Poly: x_In = " << x << std::endl;
    std::cout << "pfs::drp::stella::math::CurveFitting::Poly: xNew = " << xNew << std::endl;
    #endif

    ndarray::Array<T, 1, 1> out = ndarray::allocate(int(num));
    #ifdef __DEBUG_POLY__
    std::cout << "Poly: coeffs_In = " << coeffs_In << std::endl;
    #endif
    std::size_t order = coeffs.size() - 1;
    #ifdef __DEBUG_POLY__
    std::cout << "Poly: I_PolynomialOrder set to " << order << std::endl;
    #endif
    if (order == 0) {
      out.deep() = coeffs(0);
      #ifdef __DEBUG_POLY__
      std::cout << "Poly: I_PolynomialOrder == 0: arr_Out set to " << out << std::endl;
      #endif
      return out;
    }
    out.deep() = coeffs(order);
    #ifdef __DEBUG_POLY__
    std::cout << "Poly: I_PolynomialOrder != 0: arr_Out set to " << out << std::endl;
    #endif

    for (std::size_t ii = order - 1; ii >= 0; --ii) {
        for (std::size_t jj = 0; jj < num; ++jj) {
            out[jj] = out[jj]*xNew[jj] + coeffs[ii];
        }
        #ifdef __DEBUG_POLY__
        std::cout << "Poly: I_PolynomialOrder != 0: for (ii = " << ii << "; ii >= 0; ii--) arr_Out set to " <<
            out << std::endl;
        #endif
    }
    #ifdef __DEBUG_CURVEFIT__
    std::cout << "CurveFitting::Poly(x, coeffs, xRangeMin, xRangeMax) finished" << std::endl;
    #endif
    return out;
}



// template <typename T>
// ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
//                                      ndarray::Array<T, 1, 1> const& D_A1_Y_In,
//                                      size_t const I_Degree_In,
//                                      T const D_LReject_In,
//                                      T const D_UReject_In,
//                                      size_t const I_NIter,
//                                      std::vector<std::string> const& S_A1_Args_In,
//                                      std::vector<void *> &ArgV){

template <typename T>
PolynomialFitResults<T> fitPolynomial(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<T, 1, 1> const& y,
    std::size_t degree,
    PolynomialFitControl<T> const& ctrl,
    ndarray::Array<T, 1, 1> const& yErr
) {
    using Array = ndarray::Array<T, 1, 1>;
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.math.CurveFitting.fitPolynomial");

    LOGLS_DEBUG(_log, "fitPolynomial(x, y, deg, lReject, uReject, nIter, Args, ArgV) started");
    utils::checkSize(x.getShape(), y.getShape(), "fitPolynomial: x vs y");

    std::size_t const nData = x.getNumElements();
    std::size_t const nCoeffs = degree + 1;

    bool haveMeasureErrors = yErr.isEmpty();
    if (haveMeasureErrors) {
        utils::checkSize(nData, yErr.getNumElements(), "fitPolynomial: yErr");
    }

    Array xNew;
    if ((std::fabs(ctrl.xRangeMin + 1.) > 0.00000001) || (std::fabs(ctrl.xRangeMax - 1.) > 0.00000001)){
        xNew = ndarray::copy(convertRangeToUnity(x, ctrl.xRangeMin, ctrl.xRangeMax));
    } else {
        xNew = x;
    }

    LOGLS_DEBUG(_log, "D_A1_X_In = " << x);
    LOGLS_DEBUG(_log, "xNew = " << xNew);
    LOGLS_DEBUG(_log, "xRange = " << ctrl.xRangeMin << "," << ctrl.xRangeMax);

    PolynomialFitResults<T> results(nData, nCoeffs);

    ndarray::Array<bool, 1, 1> rejectedOld;
    for (int iter = 0; iter < ctrl.nIter; ++iter) {
        rejectedOld = ndarray::copy(results.rejected);
        results = fitPolynomial(x, y, degree, yErr, &results);

        LOGLS_DEBUG(_log, "fitPolynomial vanilla returned coeffs = " << results.coeffs);
        LOGLS_DEBUG(_log, "yFit = " << results.yFit);

        Array dev = ndarray::copy(y - results.yFit);
        float const sdev = std::sqrt(sum(dev*dev/T(nData)));
        float const low = ctrl.rejectLow*sdev;
        float const high = ctrl.rejectHigh*sdev;
        results.rejected.deep() = ndarray::logical_or(ndarray::less(dev, low), ndarray::greater(dev, high));
        results.numRejected = std::accumulate(
            results.rejected.begin(),
            results.rejected.end(),
            std::size_t(0),
            [](std::size_t left, bool right) { return right ? left + 1 : left; }
        );
        LOGLS_DEBUG(_log, "rejected " << results.numRejected << ": " << results.rejected);

        if (ndarray::all(results.rejected == rejectedOld)) {
            break;
        }
    }
    LOGLS_DEBUG(_log, "PolyFit(x, y, deg, lReject, uReject, nIter, Args, ArgV) finished");
    return results;
}

/** **********************************************************************/

template< typename T >
PolynomialFitResults<T> fitPolynomial(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<T, 1, 1> const& y,
    std::size_t degree,
    ndarray::Array<T, 1, 1> const& yErr,
    PolynomialFitResults<T> *resultsIn
) {
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.math.CurveFitting.fitPolynomial");
    LOGLS_DEBUG(_log, "fitPolynomial(x, y, deg, Args, ArgV) started");
    utils::checkSize(x.getNumElements(), y.getNumElements(), "fitPolynomial: x vs y");
    if (!yErr.isEmpty()) {
        utils::checkSize(yErr.getNumElements(), x.getNumElements(), "fitPolynomial: yErr vs x");
    }

    LOGLS_DEBUG(_log, "y = " << y);
    std::size_t const nDataPoints = x.getNumElements();
    std::size_t const nCoeffs = degree + 1;
    LOGLS_DEBUG(_log, "nCoeffs set to " << nCoeffs);
    if (resultsIn) {
        utils::checkSize(resultsIn->yFit.getNumElements(), nDataPoints, "fitPolynomial: results yFit");
        utils::checkSize(resultsIn->coeffs.getNumElements(), nCoeffs, "fitPolynomial: results coeffs");
    }
    PolynomialFitResults<T> results = resultsIn ? *resultsIn : PolynomialFitResults<T>(nDataPoints, nCoeffs);

    // Measurement errors
    bool haveMeasureError = yErr.isEmpty();
    ndarray::Array<double, 1, 1> sdevSquare = ndarray::allocate(nDataPoints);
    if (haveMeasureError) {
        sdevSquare.deep() = yErr*yErr;
    } else {
        sdevSquare.deep() = 1.0;
    }
    LOGLS_DEBUG(_log, "D_A1_SDevSquare set to " << sdevSquare);

    ndarray::Array<double, 1, 1> b = ndarray::allocate(nCoeffs);
    ndarray::Array<double, 1, 1> z = ndarray::allocate(nDataPoints);
    ndarray::Array<double, 1, 1> wy = ndarray::allocate(nDataPoints);
    z.deep() = x;

    if (haveMeasureError) {
        wy.deep() = wy/sdevSquare;
        results.covar[0][0] = sum(1./sdevSquare);
        LOGLS_DEBUG(_log, "B_HaveMeasureError: (*P_D_A2_Covar)(0,0) set to "
                          << results.covar[ndarray::makeVector(0, 0)]);
    } else {
        wy.deep() = ndarray::copy(y);
        results.covar[0][0] = nDataPoints;
        LOGLS_DEBUG(_log, "!B_HaveMeasureError: (*P_D_A2_Covar)(0,0) set to "
                          << results.covar[ndarray::makeVector(0, 0)]);
    }

    // Apply rejection
    std::size_t numGood = nDataPoints;
    if (resultsIn) {
        z.deep() *= ndarray::logical_not(resultsIn->rejected);
        numGood -= resultsIn->numRejected;
    }

    b[0] = sum(wy);
    for (std::size_t p = 1; p <= 2*degree; ++p) {
        if (p < nCoeffs) {
            b[p] = ndarray::sum(wy*z);
        }
        double sum;
        if (haveMeasureError){
            sum = ndarray::sum(z/sdevSquare);
        } else {
            sum = ndarray::sum(z);
        }
        for (std::size_t j = (p - degree > 0) ? p - degree : 0; j <= degree; ++j) {
            results.covar[ndarray::makeVector(j, p - j)] = sum;
        }
    }

    LOGLS_DEBUG(_log, "before InvertGaussJ: (*P_D_A2_Covar) = " << results.covar);
    results.covar.asEigen() = results.covar.asEigen().inverse();
    LOGLS_DEBUG(_log, "(*P_D_A2_Covar) set to " << results.covar);
    results.coeffs.asEigen() = results.covar.asEigen()*b.asEigen();

    results.yFit.deep() = calculatePolynomial(x, results.coeffs);

    for (std::size_t k = 0; k < nCoeffs; ++k) {
        double const covar = results.covar[k][k];
        results.sigma[k] = covar > 0 ? std::sqrt(covar) : 1.0;
    }
    LOGLS_DEBUG(_log, "(*P_D_A1_Sigma) set to " << results.sigma);
    LOGLS_DEBUG(_log, "*P_D_A1_YFit = " << results.yFit);

    Eigen::Array<T, Eigen::Dynamic, 1> diff = y.asEigen() - results.yFit.asEigen();
    LOGLS_DEBUG(_log, "Diff set to " << diff);
    ndarray::Array<T, 1, 1> errTemp = ndarray::allocate(nDataPoints);
    errTemp.asEigen() = diff.pow(2);
    errTemp.deep() *= ndarray::logical_not(resultsIn->rejected);
    LOGLS_DEBUG(_log, "Err_Temp set to " << errTemp);
    if (haveMeasureError){
        results.chi2 = sum(errTemp/sdevSquare);
        LOGLS_DEBUG(_log, "B_HaveMeasureError: D_ChiSq set to " << results.chi2);
    } else {
        results.chi2 = sum(errTemp);
        LOGLS_DEBUG(_log, "!B_HaveMeasureError: D_ChiSq set to " << results.chi2);
        results.sigma.deep() *= std::sqrt(float(results.chi2)/(numGood - nCoeffs));
        LOGLS_DEBUG(_log, "!B_HaveMeasureError: (*P_D_A1_Sigma) set to " << (results.sigma));
    }
    LOGLS_DEBUG(_log, "returning D_A1_Out = " << results.coeffs);
    LOGLS_DEBUG(_log, "fitPolynomial(x, y, deg, Args, ArgV) finished");

    return results;
}


/************************************************************************************************************/
/*
 * Fit the model
 *  data = bkgd + amp*profile + epsilon
 * where epsilon ~ N(0, dataVar)
 *
 * Return reduced chi^2 (or -ve in case of problems)
 */
namespace {

template<typename ImageT>
std::tuple<float, // reduced chi^2 or error status
           ImageT, // amplitude of fit
           ImageT, // background
           ImageT> // variance
fitProfile1d(ndarray::ArrayRef<ImageT const, 1, 1> const& data, // data
             ndarray::ArrayRef<ImageT const, 1, 1> const& dataVar,  // errors in data
             ndarray::ArrayRef<bool, 1, 1> const& traceMask,   // true for points in fiberTrace
             ndarray::ArrayRef<ImageT, 1, 1> const& profile,   // profile to fit
             float clipNSigma, // clip at this many sigma
             bool fitBackground // Should I fit the background?
) {
    using Tuple = std::tuple<float, ImageT, ImageT, ImageT>;
    std::size_t const nData = data.getNumElements();
    assert(nData == profile.size());
    assert(nData == traceMask.size());
    assert(nData == dataVar.size());
    bool const clipData = (clipNSigma > 0.) ? true : false;

    ImageT amp = 0.0;                        // amplitude of fit
    ImageT bkgd = 0.0;                       // sky level
    ImageT ampVar = 0.0;                    // amp's variance

    if ((data.asEigen().sum() == 0.) || (profile.asEigen().sum() == 0.)) {
        return Tuple(-1.0, amp, bkgd, ampVar);
    }
    //
    // Check that the pixel variances are not 0
    //
    for (std::size_t i = 0; i < traceMask.size(); ++i) {
        if (traceMask[i] == 0) { // bad pixels
            continue;
        }

        if (!std::isfinite(dataVar[i])) {
            std::string message("fitProfile1d:");
            message += ": i = " + std::to_string(i) + ": ERROR: dataVar(i) is not finite";
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        } else if (dataVar[i] < 0.00000000000000001){
            std::string message("fitProfile1d:");
            message += ": i = " + std::to_string(i) + ": ERROR: dataVar(i) == 0.";
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
    }

#if defined(BKGD_VAR)
    double bkgdVar = 0.0;            // bkgd's variance
    double ampBkgdCovar = 0;        // covariance between amp and bkgd
#endif

    if (traceMask.asEigen().sum() == 0) {
        return Tuple(-2, amp, bkgd, ampVar);
    }

    auto model = ndarray::Array<ImageT, 1, 1>(nData); // fit to data

    float rchi2 = -1; // reduced chi^2 (we're only fitting an amplitude)
    for (;;) {
        amp = 0.0;

        /// remove bad pixels marked by mask
        const int nGood = traceMask.asEigen().sum();
        if (nGood == 0) {
            return Tuple(-3.0, amp, bkgd, ampVar);
        }
        // Let us call the profile P and the intensity D in naming our sums
        double sum = 0.;
        double sumPP = 0.;
        double sumP = 0.;
        double sumPD = 0.;
        double sumD = 0.;
        for (std::size_t i = 0; i < traceMask.size(); ++i) {
            if (traceMask[i] == 0) { // bad pixels
                continue;
            }

            double weight = 1.0/dataVar[i];

            sum += weight;
            sumP += weight*profile[i];
            sumD += weight*data[i];
            sumPD += weight*profile[i]*data[i];
            sumPP += weight*profile[i]*profile[i];
        }
        double const delta = sum*sumPP - std::pow(sumP, 2);

        if (fitBackground) {
            amp = (sum*sumPD - sumP*sumD)/delta;
            bkgd = (sumPP*sumD - sumP*sumPD)/delta;

            ampVar = sum/delta;
#if defined(BKGD_VAR)
            bkgdVar = sumPP/delta;
            amp_bkgdCovar = -sum_P/D_Delta;
#endif
        } else {
            amp = sumPD/sumPP;
            bkgd = 0;

            ampVar = 1.0/sumPP;
#if defined(BKGD_VAR)
            bkgdVar = 0;
            ampBkgdCovar = 0;
#endif
        }

        model.deep() = bkgd + amp*profile;

        float chiSqr = 0.;
        int nPix = 0;                // number of unmasked pixels
        int nClip = 0;               // number of newly clipped pixels
        for (std::size_t i = 0; i < traceMask.size(); ++i) {
            if (traceMask[i] == 0) { // bad pixels
                continue;
            }

            float const dchi2 = pow(data[i] - model[i], 2)/dataVar[i];
            nPix++;
            chiSqr += dchi2;

            if (clipData && dchi2 > clipNSigma*clipNSigma) {
                traceMask[i] = 0;
                ++nClip;
            }
        }
        rchi2 = chiSqr/(nPix - 1);

        if (std::fabs(amp) < 0.000001) {
            break;
        }
        if (nClip == 0) {          // we didn't clip any new pixels
            break;
        }
    }

    return Tuple(rchi2, amp, bkgd, ampVar);
}

} // anonymous namespace

/************************************************************************************************************/

template <typename ImageT>
std::tuple<ndarray::Array<ImageT, 1, 1>,
           ndarray::Array<ImageT, 1, 1>,
           ndarray::Array<ImageT, 1, 1>
           >
fitProfile2d(
    lsst::afw::image::MaskedImage<ImageT> const& data,
    ndarray::Array<bool, 2, 1> const& traceMask,
    ndarray::Array<ImageT, 2, 1> const& profile2d,
    bool fitBackground,
    float clipNSigma,
    std::vector<std::string> const& badMaskPlanes
) {
    using Array = ndarray::Array<ImageT, 1, 1>;
    using MaskPixel = lsst::afw::image::MaskPixel;

    std::size_t const height = data.getHeight();

    ndarray::Array<ImageT const, 2, 1> const& image = data.getImage()->getArray();
    ndarray::Array<MaskPixel const, 2, 1> const& mask = data.getMask()->getArray();
    ndarray::Array<ImageT const, 2, 1> const& variance = data.getVariance()->getArray();
    MaskPixel const badBitmask = data.getMask()->getPlaneBitMask(badMaskPlanes);

    utils::checkSize(image.getShape(), profile2d.getShape(), "fitProfile2d: data vs profile2d");

    Array spectrumImage = ndarray::allocate(height);
    Array spectrumBackground = ndarray::allocate(height);
    Array spectrumVariance = ndarray::allocate(height);

    for (std::size_t y = 0; y < height; ++y) {
        /*
         * Add bad pixels to the traceMask
         */
        traceMask[y].deep() &= ndarray::equal(mask[y] & badBitmask, 0);

        auto const result = fitProfile1d(image[y], variance[y], traceMask[y], profile2d[y], clipNSigma,
                                         fitBackground);

        float rchi2;
        std::tie(rchi2, spectrumImage[y], spectrumBackground[y], spectrumVariance[y]) = result;
        if (rchi2 < 0) { // failed
            ; // XXX need to set a bit in the output mask, but it's still binary (grr)
        }
    }
    return std::make_tuple(spectrumImage, spectrumBackground, spectrumVariance);
}


// XXX replace this implementation with lsst::afw::math::Chebyshev1Function1
// (or maybe the guts of lsst::afw::math::ChebyshevBoundedField) once we have tests
template <typename T, typename U>
ndarray::Array<T, 1, 1> calculateChebyshev(
    ndarray::Array<T, 1, 1> const& x,
    ndarray::Array<U, 1, 1> const& coeffs,
    T xRangeMin,
    T xRangeMax
) {
    std::size_t nCoeffs = coeffs.getNumElements();
    #ifdef __DEBUG_CURVEFIT__
    std::cout << "CurveFitting::chebyshev(a, coeffs) started" << std::endl;
    std::cout << "pfs::drp::stella::math::CurveFitting::chebyshev: coeffs_In = " << nCoeffs << ": " <<
        coeffs << std::endl;
    #endif

    std::size_t num = x.getNumElements();
    ndarray::Array<T, 1, 1> xScaled = ndarray::allocate(num);
    xScaled.deep() = convertRangeToUnity(x, xRangeMin, xRangeMax);
    ndarray::Array<T, 1, 1> c0Arr;
    ndarray::Array<T, 1, 1> c1Arr;
    T c0 = 0;
    T c1 = 0;
    if (nCoeffs == 1) {
        c0 = coeffs[0];
        c1 = 0;
    } else if (nCoeffs == 2) {
        c0 = coeffs[0];
        c1 = coeffs[1];
    } else {
        ndarray::Array<float, 1, 1> x2 = ndarray::allocate(xScaled.getShape()[0]);
        x2.deep() = 2.*xScaled;
        T c0 = coeffs[nCoeffs - 2];
        c1 = coeffs[nCoeffs - 1];
        #ifdef __DEBUG_CURVEFIT__
        std::cout << "chebyshev: c0 = " << c0 << ", c1 = " << c1 << std::endl;
        #endif
        c0Arr = ndarray::allocate(num);
        c1Arr = ndarray::allocate(num);
        for (std::size_t i = 3; i <= nCoeffs; ++i){
            if (i == 3) {
                T tmp = c0;
                c0 = coeffs[nCoeffs - i] - c1;
                c1Arr.deep() = tmp + c1*x2;
            } else if (i == 4) {
                T tmp = c0;
                c0Arr.deep() = coeffs[nCoeffs - i] - c1Arr;
                c1Arr.deep() = tmp + c1Arr * x2;
            } else {
                ndarray::Array<T, 1, 1> tmp = ndarray::copy(c0Arr);
                c0Arr.deep() = coeffs[nCoeffs - i] - c1Arr;
                c1Arr.deep() = tmp + c1Arr * x2;
            }
            #ifdef __DEBUG_CURVEFIT__
            std::cout << "chebyshev: i = " << i << ": c0 = " << c0 << ", c0Arr = " << c0Arr <<
                ", c1Arr = " << c1Arr << std::endl;
            #endif
        }
    }
    ndarray::Array<T, 1, 1> yCalc = ndarray::allocate(num);
    if (nCoeffs < 3) {
        yCalc.deep() = c0 + c1*xScaled;
    } else if (nCoeffs == 3) {
        yCalc.deep() = c0 + c1Arr*xScaled;
    } else {
        yCalc.deep() = c0Arr + c1Arr*xScaled;
    }
    #ifdef __DEBUG_CURVEFIT__
    std::cout << "chebyshev: yCalc = " << yCalc << std::endl;
    std::cout << "CurveFitting::chebyshev(a, coeffs) finished" << std::endl;
    #endif
    return yCalc;
}

namespace {

struct gaussianFunctor : Eigen::DenseFunctor<float> {
    Eigen::VectorXf x0;
    Eigen::VectorXf y0;

    gaussianFunctor(Eigen::MatrixX2f const& f0) :
        Eigen::DenseFunctor<float>(3, f0.rows()),
        x0(f0.col(0)),
        y0(f0.col(1))
        {}

    int operator()(InputType const& x, ValueType & vec)  {
        auto const num = -(x0 - ValueType::Constant(values(), x[1])).array().square();
        auto const den = 2*std::pow(x[2], 2);
        vec = x[0]*(num/den).exp() - y0.array();
        return 0;
    }

    int df(InputType const& x, JacobianType & jac)  {
        auto const bv = ValueType::Constant(values(), x[1]);
        auto const tmp = (x0 - bv).array();
        auto const num = -tmp.square();
        auto const c2 = std::pow(x[2], 2);
        auto const den = 2*c2;

        auto const j0 = (num/den).exp();
        auto const j1 = x[0]*tmp*j0/c2;

        jac.col(0) = j0;
        jac.col(1) = j1;
        jac.col(2) = tmp*j1/x[2];

        return 0;
    }
};

} // anonymous namespace


// XXX convert this to use lsst::afw::math::LeastSquares (once we have tests)
// because Eigen's LM solver is "unsupported".
ndarray::Array<float, 1, 1> fitGaussian(
    ndarray::Array<float, 2, 1> const& xy,
    ndarray::Array<float, 1, 1> const& guess
) {
    #ifdef __DEBUG_CURVEFIT__
    std::cout << "CurveFitting::gaussFit(xy, guess) started" << std::endl;
    #endif
    utils::checkSize(guess.getNumElements(), std::size_t(3), "fitGaussian guess");
    gaussianFunctor func{xy.asEigen()};
    Eigen::LevenbergMarquardt<gaussianFunctor> solver(func);
    solver.setXtol(1.0e-6);
    solver.setFtol(1.0e-6);

    Eigen::VectorXf eigen(3);
    eigen = guess.asEigen();

    solver.minimize(eigen);
    #ifdef __DEBUG_CURVEFIT__
    std::cout << "CurveFitting::gaussFit(xy, guess) finished" << std::endl;
    #endif

    ndarray::Array<float, 1, 1> result = ndarray::allocate(3);
    result.asEigen() = eigen;
    return result;
}


// Explicit instantiations
template ndarray::Array<float, 1, 1> calculatePolynomial(
    ndarray::Array<float, 1, 1> const&,
    ndarray::Array<float, 1, 1> const&,
    float,
    float
);

template ndarray::Array<float, 1, 1>
calculateChebyshev(
    ndarray::Array<float, 1, 1> const&,
    ndarray::Array<float, 1, 1> const&,
    float,
    float
);

template struct PolynomialFitResults<float>;
template struct PolynomialFitControl<float>;

template PolynomialFitResults<float> fitPolynomial(
    ndarray::Array<float, 1, 1> const&,
    ndarray::Array<float, 1, 1> const&,
    std::size_t,
    ndarray::Array<float, 1, 1> const&,
    PolynomialFitResults<float>*
);

template PolynomialFitResults<float> fitPolynomial(
    ndarray::Array<float, 1, 1> const&,
    ndarray::Array<float, 1, 1> const&,
    std::size_t,
    PolynomialFitControl<float> const&,
    ndarray::Array<float, 1, 1> const&
);

template std::tuple<ndarray::Array<float, 1, 1>,
                    ndarray::Array<float, 1, 1>,
                    ndarray::Array<float, 1, 1>> fitProfile2d(
    lsst::afw::image::MaskedImage<float> const&,
    ndarray::Array<bool, 2, 1> const&,
    ndarray::Array<float, 2, 1> const&,
    bool fitBackground,
    float clipNSigma,
    std::vector<std::string> const&
);

}}}}
