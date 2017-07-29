#include <iostream>
#include "lsst/base.h"

#include "pfs/drp/stella/math/CurveFitting.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/utils/Utils.h"

namespace pfs { namespace drp { namespace stella { namespace utils{

   int KeyWord_Set(std::vector<std::string> const& keyWords_In,
                   std::string const& str_In)
  {
    for (int m = 0; m < int(keyWords_In.size()); ++m){
      if (keyWords_In[m].compare(str_In) == 0)
        return m;
    }
    return -1;
  }

  /**************************************************************************/

  template<typename T>
  ndarray::Array<T, 2, 1> get2DndArray(T nRows, T nCols){
    ndarray::Array<T, 2, 1> out = ndarray::allocate(nRows, nCols);
    out[ndarray::view()()] = 0;
    return out;
  }

  /**************************************************************************/

  template<typename T>
  ndarray::Array<T, 1, 1> get1DndArray(T size){
    ndarray::Array<T, 1, 1> out = ndarray::allocate(size);
    out[ndarray::view()] = 0;
    return out;
  }

  /**************************************************************************/

  void testPolyFit(){
    size_t nX = 10;
    ndarray::Array<float, 1, 1> xArr = ndarray::allocate(nX);
    size_t nDeg = 1;
    size_t badPos = 6;
    float badVal = -10.;
    ndarray::Array<float, 1, 1> coeffsIn = ndarray::allocate(nDeg+1);
    for (size_t i = 0; i <= nDeg; ++i)
      coeffsIn[i] = i + 1.;
    for (size_t pos = 0; pos < xArr.getShape()[0]; ++pos){
      xArr[pos] = float(pos);
    }
    ndarray::Array<float, 1, 1> yArr = pfs::drp::stella::math::Poly(xArr, coeffsIn);
    float goodVal = yArr[badPos];
    yArr[badPos] = badVal;

    std::cout << "Test PolyFit(xArr, yArr, nDeg)" << std::endl;
    #ifdef __DEBUG_POLYFIT__
      std::cout << "testPolyFit: Testing PolyFit(xArr=" << xArr << ", yArr=" << yArr << ", nDeg=" << nDeg << ")" << std::endl;
    #endif
    ndarray::Array<float, 1, 1> coeffs = pfs::drp::stella::math::PolyFit(xArr, yArr, nDeg);
    #ifdef __DEBUG_POLYFIT__
      std::cout << "testPolyFit: coeffs = " << coeffs << std::endl;
    #endif
    ndarray::Array<float, 1, 1> yFit = pfs::drp::stella::math::Poly(xArr, coeffs);
    #ifdef __DEBUG_POLYFIT__
      std::cout << "testPolyFit: yFit = " << yFit << std::endl;
    #endif
    for (size_t i = 0; i < xArr.getShape()[0]; ++i){
      if (i != badPos){
        if (fabs(yArr[i] - yFit[i]) < 0.01){
          std::string message("error: fabs(yArr[");
          message += std::to_string(i) + "] - yFit[" + std::to_string(i) + "]) = " + std::to_string(fabs(yArr[i]-yFit[i])) + " < 0.01";
          throw std::runtime_error(message);
        }
      }
    }

    std::cout << "Test PolyFit(xArr, yArr, nDeg, lSig, uSig, nIter)" << std::endl;
    float lSig = -2.;
    float uSig = 2.;
    size_t nIter = 2;
    coeffs = pfs::drp::stella::math::PolyFit(xArr, yArr, nDeg, lSig, uSig, nIter);
    #ifdef __DEBUG_POLYFIT__
      std::cout << "testPolyFit:coeffs = " << coeffs << std::endl;
    #endif
    yFit = pfs::drp::stella::math::Poly(xArr, coeffs);
    #ifdef __DEBUG_POLYFIT__
      std::cout << "testPolyFit: yFit = " << yFit << std::endl;
    #endif
    for (size_t i = 0; i < xArr.getShape()[0]; ++i){
      if (i != badPos){
        if (fabs(yArr[i] - yFit[i]) > 0.01){
          std::string message("error: fabs(yArr[");
          message += std::to_string(i) + "] - yFit[" + std::to_string(i) + "]) = " + std::to_string(fabs(yArr[i]-yFit[i])) + " > 0.01";
          throw std::runtime_error(message);
        }
      }
    }
    for (size_t i = 0; i <= nDeg; ++i){
      float const tol = 7e-6;
      if (fabs(coeffs[i] - coeffsIn[i]) > tol) {
        std::string message("error: fabs(coeffs[i](=");
        message += std::to_string(coeffs[i]) + ") - coeffsIn[i](=" + std::to_string(coeffsIn[i]) + ")) > " +
            std::to_string(tol);
        throw std::runtime_error(message);
      }
    }

    ndarray::Array<float, 1, 1> xRange = ndarray::allocate(2);
    xRange[0] = 0.;
    xRange[1] = xArr[xArr.getShape()[0]-1];

    ndarray::Array<float, 1, 1> xNorm = pfs::drp::stella::math::convertRangeToUnity(xArr, xRange);
    #ifdef __DEBUG_POLYFIT__
      std::cout << "testPolyFit: xNorm = " << xNorm << std::endl;
    #endif

    int nArgs = 8;
    std::vector< std::string > keyWords(nArgs);
    keyWords[0] = std::string("REJECTED");
    keyWords[1] = std::string("NOT_REJECTED");
    keyWords[2] = std::string("N_REJECTED");
    PTR(std::vector<size_t>) P_I_A1_Rejected(new std::vector<size_t>());
    PTR(std::vector<size_t>) P_I_A1_NotRejected(new std::vector<size_t>());
    int I_NRejected = 3;
    PTR(int) P_I_NRejected(new int(I_NRejected));
    PTR(ndarray::Array<float, 1, 1>) pXRange(new ndarray::Array<float, 1, 1>(xRange));
    ndarray::Array<float, 1, 1> xRangeBak = ndarray::allocate(xRange.getShape()[0]);
    xRangeBak.deep() = xRange;
    std::vector<void*> args(nArgs);
    args[0] = &P_I_A1_Rejected;
    args[1] = &P_I_A1_NotRejected;
    args[2] = &P_I_NRejected;
    ndarray::Array<float, 1, 1> sigma = ndarray::allocate(nDeg+1);
    PTR(ndarray::Array<float, 1, 1>) pSigma(new ndarray::Array<float, 1, 1>(sigma));
    keyWords[6] = std::string("SIGMA");
    args[6] = &pSigma;
    ndarray::Array<float, 2, 2> covar = ndarray::allocate(ndarray::makeVector(nDeg+1,nDeg+1));
    PTR(ndarray::Array<float, 2, 2>) pCovar(new ndarray::Array<float, 2, 2>(covar));
    keyWords[7] = std::string("COVAR");
    args[7] = &pCovar;
    #ifdef __DEBUG_POLYFIT__
      std::cout << "=================================================================" << std::endl;
      std::cout << "testPolyFit: Testing PolyFit(xNorm=" << xNorm << ", yArr=" << yArr << ", nDeg=" << nDeg << ", lSig=" << lSig << ", uSig=" << uSig << ", nIter=" << nIter << ", keyWords, args)" << std::endl;
    #endif

    std::cout << "Test PolyFit without MeasureErrors and without re-scaling the xRange, using the already re-scaled xRange 'xNorm'" << std::endl;
    coeffs = pfs::drp::stella::math::PolyFit(xNorm, yArr, nDeg, lSig, uSig, nIter, keyWords, args);
    #ifdef __DEBUG_POLYFIT__
      std::cout << "testPolyFit: xRange = " << xRange << std::endl;
      std::cout << "testPolyFit: xRangeBak = " << xRangeBak << std::endl;
      std::cout << "testPolyFit: coeffs = " << coeffs << std::endl;
      std::cout << "testPolyFit: P_I_A1_Rejected = ";
      for (size_t pos = 0; pos < P_I_A1_Rejected->size(); ++pos)
        std::cout << (*P_I_A1_Rejected)[pos] << ", ";
      std::cout << std::endl;
      std::cout << "testPolyFit: not rejected = ";
      for (size_t pos = 0; pos < P_I_A1_NotRejected->size(); ++pos)
        std::cout << (*P_I_A1_NotRejected)[pos] << ", ";
      std::cout << std::endl;
      std::cout << "testPolyFit: n rejected = " << *P_I_NRejected << std::endl;
    #endif
    if (*P_I_NRejected != 1){
      std::string message("testPolyFit: ERROR: *P_I_NRejected(=");
      message += std::to_string(*P_I_NRejected) + " != 1";
      throw std::runtime_error(message);
    }
    if (P_I_A1_Rejected->size() != 1){
      std::string message("testPolyFit: ERROR: P_I_A1_Rejected->size()=");
      message += std::to_string(P_I_A1_Rejected->size()) + " != 1";
      throw std::runtime_error(message);
    }
    if ((*P_I_A1_Rejected)[0] != badPos){
      std::string message("testPolyFit: ERROR: P_I_A1_Rejected[0]=");
      message += std::to_string((*P_I_A1_Rejected)[0]) + "!= badPos=" + std::to_string(badPos);
      throw std::runtime_error(message);
    }
    if (P_I_A1_NotRejected->size() != (nX - 1)){
      std::string message("testPolyFit: ERROR: P_I_A1_NotRejected-size(=");
      message += std::to_string(P_I_A1_NotRejected->size()) + ") != " + std::to_string(nX-1);
      throw std::runtime_error(message);
    }
    if (pfs::drp::stella::math::find(pfs::drp::stella::math::vectorToNdArray(*P_I_A1_NotRejected, false), (*P_I_A1_Rejected)[0]) >= 0){
      std::string message("testPolyFit: can still find (*P_I_A1_Rejected)[0] in P_I_A1_NotRejected");
      throw std::runtime_error(message);
    }
    if (pSigma->getShape()[0] != (nDeg + 1)){
      std::string message("testPolyFit: ERROR: pSigma->getShape()[0] != nDeg+1(=");
      message += std::to_string(nDeg+1);
      throw std::runtime_error(message);
    }
    if ((pCovar->getShape()[0] != (nDeg + 1))
        || (pCovar->getShape()[1] != (nDeg + 1))){
      std::string message("testPolyFit: ERROR: pCovar not (nDeg+1)x(nDeg+1)( = ");
      message += std::to_string(nDeg+1) + "x" + std::to_string(nDeg+1);
      throw std::runtime_error(message);
    }

    yFit = pfs::drp::stella::math::Poly<float>(xNorm, coeffs, -1., 1.);
    for (size_t i = 0; i < yFit.getShape()[0]; ++i){
      #ifdef __DEBUG_POLYFIT__
        std::cout << "testPolyFit: yArr[" << i << "] = " << yArr[i] << ", yFit[" << i << "] = " << yFit[i] << std::endl;
      #endif
      if ((i != badPos) && (fabs(yFit[i] - yArr[i]) > 0.1)){
        std::string message("testPolyFit: ERROR1: fabs(yFit[");
        message += std::to_string(i) + "] - yArr[" + std::to_string(i) + "])=" + std::to_string(fabs(yFit[i]-yArr[i])) + " > 0.1";
        throw std::runtime_error(message);
      }
      if ((i == badPos) && (fabs(yFit[i] - goodVal) > 0.1)){
        std::string message("testPolyFit: ERROR1: fabs(yFit[");
        message += std::to_string(i) + "] - goodVal=" + std::to_string(goodVal) + " = " + std::to_string(fabs(yFit[i]-goodVal)) + " > 0.1";
        throw std::runtime_error(message);
      }
    }

    std::cout << "Test with Measure Errors (wrong length) and with re-scaling the xRange to [-1,1]" << std::endl;
    keyWords[3] = std::string("XRANGE");
    args[3] = &pXRange;
    keyWords[4] = std::string("MEASURE_ERRORS");
    ndarray::Array<float, 1, 1> measureErrorsWrongSize = ndarray::allocate(xArr.getShape()[0]-1);
    for (size_t pos = 0; pos < measureErrorsWrongSize.getShape()[0]; ++pos)
      measureErrorsWrongSize[pos] = sqrt(fabs(yArr[pos]));
    PTR(ndarray::Array<float, 1, 1>) pMeasureErrorsWrongSize(new ndarray::Array<float, 1, 1>(measureErrorsWrongSize));
    args[4] = &pMeasureErrorsWrongSize;
    #ifdef __DEBUG_POLYFIT__
      std::cout << "=================================================================" << std::endl;
      std::cout << "testPolyFit: Testing PolyFit(xArr=" << xArr << ", yArr=" << yArr << ", nDeg=" << nDeg << ", lSig=" << lSig << ", uSig=" << uSig << ", nIter=" << nIter << ", keyWords, args)" << std::endl;
    #endif
    try{
      coeffs = pfs::drp::stella::math::PolyFit(xArr, yArr, nDeg, lSig, uSig, nIter, keyWords, args);
    }
    catch (const std::exception& e) {
      std::string errorMessage = e.what();
      std::string testMessage("pfs::drp::stella::math::CurveFitting::PolyFit: Error:");
      testMessage += " P_D_A1_MeasureErrors->getShape()[0](=" + std::to_string(nX-1);
      testMessage += ") != D_A1_X_In.getShape()[0](=" + std::to_string(nX) + ")";
      if (errorMessage.compare(testMessage) != 0)
        throw;
    }
    for (size_t i = 0; i < pXRange->getShape()[0]; ++i){
      if (fabs((*pXRange)[i] - xRangeBak[i]) > 0.0000001){
        std::string message("error: fabs((*pXRange)[");
        message += std::to_string(i) +"](=" + std::to_string((*pXRange)[i]) + ") - xRangeBak[" + std::to_string(i) + "](=" + std::to_string(xRangeBak[i]) + ")) = ";
        message += std::to_string(fabs((*pXRange)[i] - xRangeBak[i])) + " > 0.0000001";
        throw std::runtime_error(message);
      }
    }

    std::cout << "Test with Measure Errors (correct length) and with re-scaling the xRange to [-1,1]" << std::endl;
    ndarray::Array<float, 1, 1> measureErrors = ndarray::allocate(xArr.getShape()[0]);
    for (size_t pos=0; pos<xArr.getShape()[0]; ++pos)
      measureErrors[pos] = sqrt(fabs(yArr[pos]));
    PTR(ndarray::Array<float, 1, 1>) pMeasureErrors(new ndarray::Array<float, 1, 1>(measureErrors));
    args[4] = &pMeasureErrors;
    ndarray::Array<float, 1, 1> yFitCheck = ndarray::allocate(xArr.getShape()[0]);
    PTR(ndarray::Array<float, 1, 1>) pYFitCheck(new ndarray::Array<float, 1, 1>(yFitCheck));
    keyWords[5] = std::string("YFIT");
    args[5] = &pYFitCheck;
    #ifdef __DEBUG_POLYFIT__
      std::cout << "=================================================================" << std::endl;
      std::cout << "testPolyFit: Testing PolyFit(xArr=" << xArr << ", yArr=" << yArr << ", nDeg=" << nDeg << ", lSig=" << lSig << ", uSig=" << uSig << ", nIter=" << nIter << ", keyWords, args)" << std::endl;
    #endif
    coeffs = pfs::drp::stella::math::PolyFit(xArr, yArr, nDeg, lSig, uSig, nIter, keyWords, args);
    for (size_t i = 0; i < pXRange->getShape()[0]; ++i){
      if (fabs((*pXRange)[i] - xRangeBak[i]) > 0.0000001){
        std::string message("error: 2. fabs((*pXRange)[");
        message += std::to_string(i) +"](=" + std::to_string((*pXRange)[i]) + ") - xRangeBak[" + std::to_string(i) + "](=" + std::to_string(xRangeBak[i]) + ")) = ";
        message += std::to_string(fabs((*pXRange)[i] - xRangeBak[i])) + " > 0.0000001";
        throw std::runtime_error(message);
      }
    }
    #ifdef __DEBUG_POLYFIT__
      std::cout << "testPolyFit: coeffs = " << coeffs << std::endl;
      std::cout << "testPolyFit: P_I_A1_Rejected = ";
      for (size_t pos = 0; pos < P_I_A1_Rejected->size(); ++pos)
        std::cout << (*P_I_A1_Rejected)[pos] << ", ";
      std::cout << std::endl;
      std::cout << "testPolyFit: not rejected = ";
      for (size_t pos = 0; pos < P_I_A1_NotRejected->size(); ++pos)
        std::cout << (*P_I_A1_NotRejected)[pos] << ", ";
      std::cout << std::endl;
      std::cout << "testPolyFit: n rejected = " << *P_I_NRejected << std::endl;
    #endif
    if (pMeasureErrors->getShape()[0] != xArr.getShape()[0]){
      std::string message("error: pMeasureErrors->getShape()[0](=");
      message += std::to_string(pMeasureErrors->getShape()[0]) + ") != xArr.getShape()[0](=" + std::to_string(xArr.getShape()[0]) + ")";
      throw std::runtime_error(message);
    }
    for (size_t pos = 0; pos < P_I_A1_NotRejected->size(); ++pos){
      if ((*P_I_A1_NotRejected)[pos] > xArr.getShape()[0]){
        std::string message("testPolyFit: ERROR: (*P_I_A1_NotRejected)[pos] = ");
        message += std::to_string((*P_I_A1_NotRejected)[pos]) + " outside limits";
        throw std::runtime_error(message);
      }
    }
    if (*P_I_NRejected != 1){
      std::string message("testPolyFit: error: *P_I_NRejected=");
      message += std::to_string(*P_I_NRejected) + " != 1";
      throw std::runtime_error(message);
    }
    yFit = pfs::drp::stella::math::Poly(xArr, coeffs, xRange[0], xRange[1]);
    #ifdef __DEBUG_POLYFIT__
      std::cout << "testPolyFit: yFit = " << yFit << std::endl;
      std::cout << "testPolyFit: yFitCheck = " << yFitCheck << std::endl;
    #endif
    for (size_t i = 0; i < yFit.getShape()[0]; ++i){
      #ifdef __DEBUG_POLYFIT__
        std::cout << "testPolyFit: yArr[" << i << "] = " << yArr[i] << ", yFit[" << i << "] = " << yFit[i] << std::endl;
      #endif
      if ((i != badPos) && (fabs(yArr[i] - yFit[i]) > 0.0001)){
        std::string message("testPolyFit: ERROR: fabs(yArr[");
        message += std::to_string(i) + "]=" + std::to_string(yArr[i]) + ") - yFit[" + std::to_string(i) + "]=" + std::to_string(yFit[i]);
        message += ") = " + std::to_string(fabs(yArr[i] - yFit[i])) + " > 0.0001";
        throw std::runtime_error(message);
      }
      if ((i == badPos) && (fabs(yFit[i] - goodVal) > 0.0001)){
        std::string message("testPolyFit: ERROR2: fabs(yFit[");
        message += std::to_string(i) + "]=" + std::to_string(yFit[i]) + " - goodVal=" + std::to_string(goodVal);
        message += " = " + std::to_string(fabs(yFit[i] - goodVal)) + " > 0.0001";
        throw std::runtime_error(message);
      }
      if (fabs(yFit[i] - yFitCheck[i]) > 0.0001){
        std::string message("testPolyFit: ERROR2: fabs(yFit[");
        message += std::to_string(i) + "]=" + std::to_string(yFit[i]) + ") - yFitCheck[" + std::to_string(i) + "]=" + std::to_string(yFitCheck[i]);
        message += ") = " + std::to_string(fabs(yFit[i] - yFitCheck[i])) + " > 0.0001";
        throw std::runtime_error(message);
      }
    }
    if (pSigma->getShape()[0] != nDeg+1){
        std::string message("testPolyFit: ERROR: pSigma->getShape()[0](=");
        message += std::to_string(pSigma->getShape()[0]) + ") != nDeg+1(=" + std::to_string(nDeg+1);
        throw std::runtime_error(message);
    }
    if (pCovar->getShape()[0] != nDeg+1){
        std::string message("testPolyFit: ERROR: pCovar->getShape()[0](=");
        message += std::to_string(pCovar->getShape()[0]) + ") != nDeg+1(=" + std::to_string(nDeg+1);
        throw std::runtime_error(message);
    }
    if (pCovar->getShape()[1] != nDeg+1){
        std::string message("testPolyFit: ERROR: pCovar->getShape()[0](=");
        message += std::to_string(pCovar->getShape()[1]) + ") != nDeg+1(=" + std::to_string(nDeg+1);
        throw std::runtime_error(message);
    }
    return;
  }

  template<typename T>
  ndarray::Array<T, 1, 1> vectorToNdArray(std::vector<T> & vec){
    ndarray::Array<T, 1, 1> array = external(vec.data(),
                                             ndarray::makeVector(int(vec.size())),
                                             ndarray::makeVector(1));
    return array;
  }

  template<typename T>
  ndarray::Array<T const, 1, 1> vectorToNdArray(std::vector<T> const& vec){
    const ndarray::Array<T const, 1, 1> array = external(vec.data(),
                                                         ndarray::makeVector(int(vec.size())),
                                                         ndarray::makeVector(1));
    return array;
  }

  template<typename T, typename U>
  ndarray::Array<U, 1, 1> typeCastNdArray(ndarray::Array<T, 1, 1> const& arr, U const& newType){
    ndarray::Array<U, 1, 1> out = ndarray::allocate(arr.getShape()[0]);
    auto itOut = out.begin();
    for (auto itIn = arr.begin(); itIn != arr.end(); ++itIn, ++itOut){
      *itOut = U(*itIn);
    }
    return out;
  }

  template ndarray::Array<float, 1, 1> get1DndArray(float);
  template ndarray::Array<float, 2, 1> get2DndArray(float, float);
  template ndarray::Array<float, 1, 1> typeCastNdArray(ndarray::Array<double, 1, 1> const&, float const&);
  template ndarray::Array<float, 1, 1> typeCastNdArray(ndarray::Array<float, 1, 1> const&, float const&);
  template ndarray::Array<double, 1, 1> typeCastNdArray(ndarray::Array<float, 1, 1> const&, double const&);
}
}}}
