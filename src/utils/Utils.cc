#include <iostream>

#include "pfs/drp/stella/math/CurveFitting.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/utils/Utils.h"

namespace pfs {
namespace drp {
namespace stella {
namespace utils {

void testPolyFit() {
    std::size_t nX = 10;
    ndarray::Array<float, 1, 1> xArr = ndarray::allocate(nX);
    std::size_t nDeg = 1;
    std::size_t badPos = 6;
    float badVal = -10.;
    ndarray::Array<float, 1, 1> coeffsIn = ndarray::allocate(nDeg + 1);
    for (std::size_t i = 0; i <= nDeg; ++i)
        coeffsIn[i] = i + 1.;
    for (std::size_t pos = 0; pos < xArr.getShape()[0]; ++pos) {
        xArr[pos] = float(pos);
    }
    ndarray::Array<float, 1, 1> yArr = math::calculatePolynomial(xArr, coeffsIn);
    float goodVal = yArr[badPos];
    yArr[badPos] = badVal;

    std::cout << "Test PolyFit(xArr, yArr, nDeg)" << std::endl;
    #ifdef __DEBUG_POLYFIT__
    std::cout << "testPolyFit: Testing PolyFit(xArr=" << xArr << ", yArr=" << yArr << ", nDeg=" << nDeg <<
        ")" << std::endl;
    #endif
    auto results = math::fitPolynomial(xArr, yArr, nDeg);
    #ifdef __DEBUG_POLYFIT__
    std::cout << "testPolyFit: coeffs = " << coeffs << std::endl;
    #endif
    ndarray::Array<float, 1, 1> yFit = math::calculatePolynomial(xArr, results.coeffs);
    #ifdef __DEBUG_POLYFIT__
    std::cout << "testPolyFit: yFit = " << yFit << std::endl;
    #endif
    for (std::size_t i = 0; i < xArr.getShape()[0]; ++i) {
        if (i != badPos) {
            if (fabs(yArr[i] - yFit[i]) < 0.01) {
                std::string message("error: fabs(yArr[");
                message += std::to_string(i) + "] - yFit[" + std::to_string(i) + "]) = " +
                    std::to_string(fabs(yArr[i] - yFit[i])) + " < 0.01";
                throw std::runtime_error(message);
            }
        }
    }

    std::cout << "Test PolyFit(xArr, yArr, nDeg, lSig, uSig, nIter)" << std::endl;
    float const lSig = -2.;
    float const uSig = 2.;
    std::size_t const nIter = 2;
    float const rangeMin = 0.0;
    float const rangeMax = xArr[xArr.getShape()[0] - 1];

    results = math::fitPolynomial(xArr, yArr, nDeg, rangeMin, rangeMax, lSig, uSig, nIter);
    #ifdef __DEBUG_POLYFIT__
    std::cout << "testPolyFit:coeffs = " << results.coeffs << std::endl;
    #endif
    yFit = math::calculatePolynomial(xArr, results.coeffs);
    #ifdef __DEBUG_POLYFIT__
    std::cout << "testPolyFit: yFit = " << yFit << std::endl;
    #endif
    for (std::size_t i = 0; i < xArr.getShape()[0]; ++i) {
        if (i != badPos){
            if (fabs(yArr[i] - yFit[i]) > 0.01) {
              std::string message("error: fabs(yArr[");
              message += std::to_string(i) + "] - yFit[" + std::to_string(i) + "]) = " +
                  std::to_string(fabs(yArr[i]-yFit[i])) + " > 0.01";
              throw std::runtime_error(message);
            }
        }
    }
    for (std::size_t i = 0; i <= nDeg; ++i){
        float const tol = 7e-6;
        if (fabs(results.coeffs[i] - coeffsIn[i]) > tol) {
          std::string message("error: fabs(coeffs[i](=");
          message += std::to_string(results.coeffs[i]) + ") - coeffsIn[i](=" +
              std::to_string(coeffsIn[i]) + ")) > " + std::to_string(tol);
          throw std::runtime_error(message);
        }
    }

    ndarray::Array<float, 1, 1> xNorm = math::convertRangeToUnity(xArr, rangeMin, rangeMax);
    #ifdef __DEBUG_POLYFIT__
      std::cout << "testPolyFit: xNorm = " << xNorm << std::endl;
    #endif

    std::cout << "Test PolyFit without MeasureErrors and without re-scaling the xRange, using the already re-scaled xRange 'xNorm'" << std::endl;
    results = math::fitPolynomial(xNorm, yArr, nDeg, float(-1.0), float(1.0), lSig, uSig, nIter);
    #ifdef __DEBUG_POLYFIT__
    std::cout << "testPolyFit: coeffs = " << results.coeffs << std::endl;
    std::cout << "testPolyFit: P_I_A1_Rejected = ";
    for (std::size_t pos = 0; pos < results.rejected.size(); ++pos) {
        std::cout << results.rejected[pos] << ", ";
    }
    std::cout << std::endl;
    std::cout << "testPolyFit: n rejected = " << results.numRejected << std::endl;
    #endif
    if (results.numRejected != 1) {
        std::string message("testPolyFit: ERROR: *P_I_NRejected(=");
        message += std::to_string(results.numRejected) + " != 1";
        throw std::runtime_error(message);
    }
    if (results.rejected.size() != 1) {
        std::string message("testPolyFit: ERROR: P_I_A1_Rejected->size()=");
        message += std::to_string(results.rejected.size()) + " != 1";
        throw std::runtime_error(message);
    }
    if (!results.rejected[badPos]) {
        std::string message("testPolyFit: ERROR: P_I_A1_Rejected[0]=");
        message += std::to_string(results.rejected[badPos]) + "!= true=" + std::to_string(badPos);
        throw std::runtime_error(message);
    }
    if (results.sigma.getShape()[0] != (nDeg + 1)) {
        std::string message("testPolyFit: ERROR: pSigma->getShape()[0] != nDeg+1(=");
        message += std::to_string(nDeg+1);
        throw std::runtime_error(message);
    }
    if (results.covar.getShape() != ndarray::makeVector(nDeg + 1, nDeg + 1)) {
        std::string message("testPolyFit: ERROR: pCovar not (nDeg+1)x(nDeg+1)( = ");
        message += std::to_string(nDeg + 1) + "x" + std::to_string(nDeg + 1);
        throw std::runtime_error(message);
    }

    yFit = results.yFit;
    for (std::size_t i = 0; i < yFit.getShape()[0]; ++i) {
        #ifdef __DEBUG_POLYFIT__
        std::cout << "testPolyFit: yArr[" << i << "] = " << yArr[i] << ", yFit[" << i << "] = " << yFit[i] << std::endl;
        #endif
        if ((i != badPos) && (std::fabs(yFit[i] - yArr[i]) > 0.1)) {
            std::string message("testPolyFit: ERROR1: fabs(yFit[");
            message += std::to_string(i) + "] - yArr[" + std::to_string(i) + "])=" +
                std::to_string(std::fabs(yFit[i]-yArr[i])) + " > 0.1";
            throw std::runtime_error(message);
        }
        if ((i == badPos) && (std::fabs(yFit[i] - goodVal) > 0.1)) {
            std::string message("testPolyFit: ERROR1: fabs(yFit[");
            message += std::to_string(i) + "] - goodVal=" + std::to_string(goodVal) + " = " +
                std::to_string(std::fabs(yFit[i]-goodVal)) + " > 0.1";
            throw std::runtime_error(message);
        }
    }

    std::cout << "Test with Measure Errors (wrong length) and with re-scaling the xRange to [-1,1]" <<  std::endl;
    ndarray::Array<float, 1, 1> measureErrorsWrongSize = ndarray::allocate(xArr.getShape()[0] - 1);
    for (std::size_t pos = 0; pos < measureErrorsWrongSize.getShape()[0]; ++pos) {
        measureErrorsWrongSize[pos] = std::sqrt(std::fabs(yArr[pos]));
    }
    #ifdef __DEBUG_POLYFIT__
    std::cout << "=================================================================" << std::endl;
    std::cout << "testPolyFit: Testing PolyFit(xArr=" << xArr << ", yArr=" << yArr << ", nDeg=" << nDeg <<
        ", lSig=" << lSig << ", uSig=" << uSig << ", nIter=" << nIter << ", keyWords, args)" << std::endl;
    #endif
    bool caught = false;
    try {
        results = math::fitPolynomial(xArr, yArr, nDeg, rangeMin, rangeMax, lSig, uSig, nIter,
                                      measureErrorsWrongSize);
    } catch (const std::exception& e) {
        // successfull detected problem
        caught = true;
    }
    if (!caught) {
        throw std::runtime_error("Exception was not thrown.");
    }

    std::cout << "Test with Measure Errors (correct length) and with re-scaling the xRange to [-1,1]" << std::endl;
    ndarray::Array<float, 1, 1> measureErrors = ndarray::allocate(xArr.getShape()[0]);
    for (std::size_t pos = 0; pos<xArr.getShape()[0]; ++pos) {
        measureErrors[pos] = sqrt(fabs(yArr[pos]));
    }
    #ifdef __DEBUG_POLYFIT__
    std::cout << "=================================================================" << std::endl;
    std::cout << "testPolyFit: Testing PolyFit(xArr=" << xArr << ", yArr=" << yArr << ", nDeg=" << nDeg <<
        ", lSig=" << lSig << ", uSig=" << uSig << ", nIter=" << nIter << ", keyWords, args)" << std::endl;
    #endif
    results = math::fitPolynomial(xArr, yArr, nDeg, rangeMin, rangeMax, lSig, uSig, nIter, measureErrors);
    #ifdef __DEBUG_POLYFIT__
    std::cout << "testPolyFit: coeffs = " << coeffs << std::endl;
    std::cout << "testPolyFit: P_I_A1_Rejected = ";
    for (std::size_t pos = 0; pos < results.rejected->size(); ++pos) {
        std::cout << results.rejected[pos] << ", ";
    }
    std::cout << std::endl;
    std::cout << "testPolyFit: n rejected = " << results.numRejected << std::endl;
    #endif
    if (results.numRejected != 1) {
        std::string message("testPolyFit: error: *P_I_NRejected=");
        message += std::to_string(results.numRejected) + " != 1";
        throw std::runtime_error(message);
    }
    yFit = math::calculatePolynomial(xArr, results.coeffs, rangeMin, rangeMax);
    #ifdef __DEBUG_POLYFIT__
    std::cout << "testPolyFit: yFit = " << yFit << std::endl;
    std::cout << "testPolyFit: yFitCheck = " << results.yFit << std::endl;
    #endif
    for (std::size_t i = 0; i < yFit.getShape()[0]; ++i) {
        #ifdef __DEBUG_POLYFIT__
        std::cout << "testPolyFit: yArr[" << i << "] = " << yArr[i] << ", yFit[" << i << "] = " << yFit[i] << std::endl;
        #endif
        if ((i != badPos) && (fabs(yArr[i] - yFit[i]) > 0.0001)) {
            std::string message("testPolyFit: ERROR: fabs(yArr[");
            message += std::to_string(i) + "]=" + std::to_string(yArr[i]) + ") - yFit[" +
                std::to_string(i) + "]=" + std::to_string(yFit[i]);
            message += ") = " + std::to_string(fabs(yArr[i] - yFit[i])) + " > 0.0001";
            throw std::runtime_error(message);
        }
        if ((i == badPos) && (fabs(yFit[i] - goodVal) > 0.0001)) {
            std::string message("testPolyFit: ERROR2: fabs(yFit[");
            message += std::to_string(i) + "]=" + std::to_string(yFit[i]) + " - goodVal=" +
                std::to_string(goodVal);
            message += " = " + std::to_string(fabs(yFit[i] - goodVal)) + " > 0.0001";
            throw std::runtime_error(message);
        }
        if (fabs(yFit[i] - results.yFit[i]) > 0.0001) {
            std::string message("testPolyFit: ERROR2: fabs(yFit[");
            message += std::to_string(i) + "]=" + std::to_string(yFit[i]) + ") - yFitCheck[" +
                std::to_string(i) + "]=" + std::to_string(results.yFit[i]);
            message += ") = " + std::to_string(fabs(yFit[i] - results.yFit[i])) + " > 0.0001";
            throw std::runtime_error(message);
        }
    }
    if (results.sigma.getShape()[0] != nDeg + 1) {
          std::string message("testPolyFit: ERROR: pSigma->getShape()[0](=");
          message += std::to_string(results.sigma.getShape()[0]) + ") != nDeg+1(=" + std::to_string(nDeg + 1);
          throw std::runtime_error(message);
    }
    if (results.covar.getShape() != ndarray::makeVector(nDeg + 1, nDeg + 1)) {
          std::string message("testPolyFit: ERROR: pCovar->getShape()(=");
          message += std::to_string(results.covar.getShape()[0]) + "," +
          std::to_string(results.covar.getShape()[0]) + ") != nDeg+1(=" + std::to_string(nDeg + 1);
          throw std::runtime_error(message);
    }
}


}}}} // namespace pfs::drp::stella::utils
