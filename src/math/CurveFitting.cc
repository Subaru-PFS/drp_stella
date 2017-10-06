#include "ndarray/eigen.h"

#include <unsupported/Eigen/LevenbergMarquardt>

#include "lsst/log/Log.h"
#include "lsst/pex/exceptions/Exception.h"
#include "lsst/afw/image/MaskedImage.h"

#include "pfs/drp/stella/math/CurveFitting.h"
#include "pfs/drp/stella/math/CurveFittingGaussian.h"
#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/utils/Utils.h"

namespace pexExcept = lsst::pex::exceptions;

namespace pfs { namespace drp { namespace stella { namespace math {

  template<typename T, typename U>
  ndarray::Array<T, 1, 1> Poly(ndarray::Array<T, 1, 1> const& x_In,
                               ndarray::Array<U, 1, 1> const& coeffs_In,
                               T xRangeMin_In,
                               T xRangeMax_In){
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::Poly(x, coeffs, xRangeMin, xRangeMax) started" << std::endl;
    #endif
    ndarray::Array<T, 1, 1> xNew;
    /// shift and rescale x_In to fit into range [-1.,1.]
    if ((std::fabs(xRangeMin_In + 1.) > 0.00000001) || (std::fabs(xRangeMax_In - 1.) > 0.00000001)){
      ndarray::Array<T, 1, 1> xRange = ndarray::allocate(2);
      xRange[0] = xRangeMin_In;
      xRange[1] = xRangeMax_In;
      xNew = pfs::drp::stella::math::convertRangeToUnity(x_In, xRange);
    }
    else{
      xNew = x_In;
    }
    #ifdef __DEBUG_POLY__
      std::cout << "pfs::drp::stella::math::CurveFitting::Poly: x_In = " << x_In << std::endl;
      std::cout << "pfs::drp::stella::math::CurveFitting::Poly: xNew = " << xNew << std::endl;
    #endif

    int ii = 0;
    ndarray::Array<T, 1, 1> arr_Out = ndarray::allocate(int(x_In.size()));
    #ifdef __DEBUG_POLY__
      std::cout << "Poly: coeffs_In = " << coeffs_In << std::endl;
    #endif
    int I_PolynomialOrder = coeffs_In.size() - 1;
    #ifdef __DEBUG_POLY__
      std::cout << "Poly: I_PolynomialOrder set to " << I_PolynomialOrder << std::endl;
    #endif
    if (I_PolynomialOrder == 0){
      arr_Out[ndarray::view()] = coeffs_In(0);
      #ifdef __DEBUG_POLY__
        std::cout << "Poly: I_PolynomialOrder == 0: arr_Out set to " << arr_Out << std::endl;
      #endif
      return arr_Out;
    }
    arr_Out[ndarray::view()] = coeffs_In(I_PolynomialOrder);
    #ifdef __DEBUG_POLY__
      std::cout << "Poly: I_PolynomialOrder != 0: arr_Out set to " << arr_Out << std::endl;
    #endif

    auto arr_Out_begin = arr_Out.begin();
    auto xNew_begin = xNew.begin();
    auto coeffs_In_begin = coeffs_In.begin();
    for (ii = I_PolynomialOrder-1; ii >= 0; ii--){
      for (int i = 0; i < arr_Out.getShape()[0]; ++i)
        *(arr_Out_begin + i) = (*(arr_Out_begin + i)) * (*(xNew_begin + i)) + (*(coeffs_In_begin + ii));
      #ifdef __DEBUG_POLY__
        std::cout << "Poly: I_PolynomialOrder != 0: for (ii = " << ii << "; ii >= 0; ii--) arr_Out set to " << arr_Out << std::endl;
      #endif
    }
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::Poly(x, coeffs, xRangeMin, xRangeMax) finished" << std::endl;
    #endif
    return arr_Out;
  }

  template<typename T>
  ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                       ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                       size_t const I_Degree_In,
                                       T const D_Reject_In,
                                       std::vector<std::string> const& S_A1_Args_In,
                                       std::vector<void *> & ArgV){
    return pfs::drp::stella::math::PolyFit(D_A1_X_In,
                                           D_A1_Y_In,
                                           I_Degree_In,
                                           -1.*D_Reject_In,
                                           D_Reject_In,
                                           -1,
                                           S_A1_Args_In,
                                           ArgV);
  }

  template< typename T >
  ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                       ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                       size_t const I_Degree_In,
                                       T const D_LReject_In,
                                       T const D_UReject_In,
                                       size_t const I_NIter,
                                       std::vector<std::string> const& S_A1_Args_In,
                                       std::vector<void *> &ArgV){
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.math.CurveFitting.PolyFit");

    LOGLS_DEBUG(_log, "PolyFit(x, y, deg, lReject, uReject, nIter, Args, ArgV) started");
    if (D_A1_X_In.getShape()[0] != D_A1_Y_In.getShape()[0]){
      std::string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: D_A1_X_In.getShape()[0](=");
      message += std::to_string(D_A1_X_In.getShape()[0]) + " != D_A1_Y_In.getShape()[0](=" + std::to_string(D_A1_Y_In.getShape()[0]) + ")";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    int I_NRejected = 0;
    bool B_HaveMeasureErrors = false;
    ndarray::Array<float, 1, 1> D_A1_Coeffs_Out = ndarray::allocate(I_Degree_In + 1);

    // We need at least an existing array to which P_D_A1_MeasureErrors points
    // so we can define the vector V_MeasureErrors and P_D_A1_MeasureErrosTemp
    // later in the block/namespace where they are possibly needed
    ndarray::Array<T, 1, 1> D_A1_MeasureErrors = ndarray::allocate(1);
    PTR(ndarray::Array<T, 1, 1>) P_D_A1_MeasureErrors(new ndarray::Array<T, 1, 1>(D_A1_MeasureErrors));

    int I_Pos = -1;
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MEASURE_ERRORS")) >= 0) {
      LOGLS_DEBUG(_log, "Reading MEASURE_ERRORS");
      P_D_A1_MeasureErrors.reset();
      P_D_A1_MeasureErrors = (*((PTR(ndarray::Array<T, 1, 1>)*)ArgV[I_Pos]));
      B_HaveMeasureErrors = true;
      if (P_D_A1_MeasureErrors->getShape()[0] != D_A1_X_In.getShape()[0]){
        std::string message("pfs::drp::stella::math::CurveFitting::PolyFit: Error: P_D_A1_MeasureErrors->getShape()[0](=");
        message += std::to_string(P_D_A1_MeasureErrors->getShape()[0]) + ") != D_A1_X_In.getShape()[0](=" + std::to_string(D_A1_X_In.getShape()[0]) + ")";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    }

    PTR(std::vector<size_t>) P_I_A1_NotRejected(new std::vector<size_t>());
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "NOT_REJECTED")) >= 0) {
      P_I_A1_NotRejected.reset();
      P_I_A1_NotRejected = *((PTR(std::vector<size_t>)*)(ArgV[I_Pos]));
      LOGLS_DEBUG(_log, "KeyWord NOT_REJECTED read");
    }

    PTR(std::vector<size_t>) P_I_A1_Rejected(new std::vector<size_t>());
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "REJECTED")) >= 0) {
      P_I_A1_Rejected.reset();
      P_I_A1_Rejected = *((PTR(std::vector<size_t>)*)(ArgV[I_Pos]));
      LOGLS_DEBUG(_log, "KeyWord REJECTED read");
    }

    PTR(int) P_I_NRejected(new int(I_NRejected));
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "N_REJECTED")) >= 0) {
      P_I_NRejected.reset();
      P_I_NRejected = *((PTR(int)*)(ArgV[I_Pos]));
      LOGLS_DEBUG(_log, "KeyWord N_REJECTED read");
    }
    *P_I_NRejected = 0;

    ndarray::Array<float, 1, 1> xRange = ndarray::allocate(2);
    xRange[0] = -1.;
    xRange[1] = 1.;
    PTR(ndarray::Array<float, 1, 1>) P_D_A1_XRange(new ndarray::Array<float, 1, 1>(xRange));
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "XRANGE")) >= 0) {
      P_D_A1_XRange.reset();
      P_D_A1_XRange = *((PTR(ndarray::Array<float, 1, 1>)*)ArgV[I_Pos]);
      if (P_D_A1_XRange->getShape()[0] != 2){
        std::string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: P_D_A1_XRange->getShape()[0](=");
        message += std::to_string(P_D_A1_XRange->getShape()[0]) + " != 2";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      xRange.deep() = *P_D_A1_XRange;
      LOGLS_DEBUG(_log, "*P_D_A1_XRange set to " << *P_D_A1_XRange);
    }
    ndarray::Array<T, 1, 1> xNew = ndarray::allocate(D_A1_X_In.getShape()[0]);

    /// shift and rescale x_In to fit into range [-1.,1.]
    if ((std::fabs(xRange[0] + 1.) > 0.00000001) || (std::fabs(xRange[1] - 1.) > 0.00000001)){
      xNew = pfs::drp::stella::math::convertRangeToUnity(D_A1_X_In, xRange);
    }
    else{
      xNew = D_A1_X_In;
    }
    LOGLS_DEBUG(_log, "D_A1_X_In = " << D_A1_X_In);
    LOGLS_DEBUG(_log, "xNew = " << xNew);
    LOGLS_DEBUG(_log, "xRange = " << xRange);

    std::vector<T> D_A1_X(D_A1_X_In.begin(), D_A1_X_In.end());
    std::vector<T> D_A1_Y(D_A1_Y_In.begin(), D_A1_Y_In.end());
    std::vector<size_t> I_A1_OrigPos(0);
    I_A1_OrigPos.reserve(D_A1_X_In.getShape()[0]);

    // Copy (deep) P_D_A1_MeasureErrors to a vector and then to another temporary
    // array pointer which is used for the measure errors during the sigma-rejection
    // iterations.
    // We create the vector of the correct size here because it will be used again
    // later in a different context
    std::vector<T> V_MeasureErrors(P_D_A1_MeasureErrors->begin(), P_D_A1_MeasureErrors->end());
    LOGLS_DEBUG(_log, "V_MeasureErrors = " << V_MeasureErrors);

    PTR(ndarray::Array<T, 1, 1>) P_D_A1_MeasureErrorsTemp(new ndarray::Array<T, 1, 1>(ndarray::external(V_MeasureErrors.data(),
                                                                                                        ndarray::makeVector(int(V_MeasureErrors.size())),
                                                                                                        ndarray::makeVector(1))));
    LOGLS_DEBUG(_log, "P_D_A1_MeasureErrorsTemp = " << *P_D_A1_MeasureErrorsTemp);

    int I_NRejected_Old = 0;
    std::vector<size_t> I_A1_Rejected_Old(0);
    bool B_Run = true;
    unsigned int i_iter = 0;
    ndarray::Array<T, 1, 1> D_A1_YFit = ndarray::allocate(D_A1_X_In.getShape()[0]);
    D_A1_YFit.deep() = 0.;
    PTR(ndarray::Array<T, 1, 1>) P_D_A1_YFit(new ndarray::Array<T, 1, 1>(D_A1_YFit));
    bool haveYFit = false;
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "YFIT")) >= 0){
      haveYFit = true;
      P_D_A1_YFit.reset();
      P_D_A1_YFit = (*((PTR(ndarray::Array<T, 1, 1>)*)ArgV[I_Pos]));
      if (P_D_A1_YFit->getShape()[0] != D_A1_X_In.getShape()[0]){
        std::string message("pfs::drp::stella::math::CurveFitting::PolyFit: KeyWord_Set(YFIT): ERROR: P_D_A1_YFit->getShape()[0](=");
        message += std::to_string(P_D_A1_YFit->getShape()[0]) + ") != D_A1_X_In.getShape()[0](=" + std::to_string(D_A1_X_In.getShape()[0]) + ")";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      LOGLS_DEBUG(_log, "1. KeyWord_Set(YFIT): P_D_A1_YFit = " << *P_D_A1_YFit);
    }
    size_t nArgs = S_A1_Args_In.size();
    std::vector<void*> args(ArgV);
    std::vector<std::string> keyWords(S_A1_Args_In);
    if (!haveYFit){
      ++nArgs;
      keyWords.push_back("YFIT");
      args.resize(nArgs);
    }
    while (B_Run){
      I_A1_Rejected_Old = *P_I_A1_Rejected;
      I_NRejected_Old = *P_I_NRejected;
      *P_I_NRejected = 0;
      ndarray::Array<T, 1, 1> D_A1_XArr = ndarray::external(D_A1_X.data(), ndarray::makeVector(int(D_A1_X.size())), ndarray::makeVector(1));
      ndarray::Array<T, 1, 1> D_A1_YArr = ndarray::external(D_A1_Y.data(), ndarray::makeVector(int(D_A1_X.size())), ndarray::makeVector(1));
      ndarray::Array<T, 1, 1> yFit = ndarray::allocate(D_A1_X.size());
      PTR(ndarray::Array<T, 1, 1>) pYFit(new ndarray::Array<T, 1, 1>(yFit));
      for (size_t iArg=0; iArg < nArgs; ++iArg){
        if (iArg >= S_A1_Args_In.size() || S_A1_Args_In[iArg] == "YFIT"){
          args[iArg] = &pYFit;
        }
        else if (S_A1_Args_In[iArg] == "MEASURE_ERRORS"){
          args[iArg] = &P_D_A1_MeasureErrorsTemp;
        }
      }
      D_A1_Coeffs_Out = pfs::drp::stella::math::PolyFit(D_A1_XArr,
                                                        D_A1_YArr,
                                                        I_Degree_In,
                                                        keyWords,
                                                        args);
      LOGLS_DEBUG(_log, "PolyFit(D_A1_XArr, D_A1_YArr, I_Degree_In, keyWords, args) returned D_A1_Coeffs_Out = " << D_A1_Coeffs_Out);
      LOGLS_DEBUG(_log, "yFit = " << *pYFit);
      ndarray::Array<T, 1, 1> D_A1_Temp = ndarray::allocate(D_A1_Y.size());
      auto itY = D_A1_YArr.begin();
      auto itYFit = pYFit->begin();
      for (auto itTemp = D_A1_Temp.begin(); itTemp < D_A1_Temp.end(); ++itTemp, ++itY, ++itYFit)
        *itTemp = (*itY) - (*itYFit);
      Eigen::Array<T, Eigen::Dynamic, 1> tempEArr = D_A1_Temp.asEigen();
      D_A1_Temp.asEigen() = tempEArr.pow(2) / T(D_A1_Y.size());
      float D_SDev = float(sqrt(D_A1_Temp.asEigen().sum()));
      P_D_A1_YFit->deep() = pfs::drp::stella::math::Poly(D_A1_X_In,
                                                         D_A1_Coeffs_Out,
                                                         xRange[0],
                                                         xRange[1]);
      LOGLS_DEBUG(_log, "P_D_A1_YFit = " << *P_D_A1_YFit);
      float D_Dev;
      D_A1_X.resize(0);
      D_A1_Y.resize(0);
      V_MeasureErrors.resize(0);
      I_A1_OrigPos.resize(0);
      P_I_A1_Rejected->resize(0);
      for (size_t i_pos=0; i_pos < D_A1_Y_In.getShape()[0]; i_pos++){
        D_Dev = D_A1_Y_In[i_pos] - (*P_D_A1_YFit)[i_pos];
        LOGLS_DEBUG(_log, "i_pos = " << i_pos << ": D_Dev = " << D_Dev << ", D_SDev = " << D_SDev);
        if ((I_NIter == 0) ||
            ((D_Dev < 0) && (D_Dev >= (D_LReject_In * D_SDev))) ||
            ((D_Dev >= 0) && (D_Dev <= (D_UReject_In * D_SDev)))){
          D_A1_X.push_back(D_A1_X_In[i_pos]);
          D_A1_Y.push_back(D_A1_Y_In[i_pos]);
          if (B_HaveMeasureErrors)
            V_MeasureErrors.push_back((*P_D_A1_MeasureErrors)[i_pos]);
          I_A1_OrigPos.push_back(i_pos);
        }
        else{
          P_I_A1_Rejected->push_back(i_pos);
          LOGLS_DEBUG(_log, "Rejecting D_A1_X_In(" << i_pos << ") = " << D_A1_X_In[i_pos]);
          ++(*P_I_NRejected);
        }
      }
      if (B_HaveMeasureErrors){
        P_D_A1_MeasureErrorsTemp.reset(new ndarray::Array<T, 1, 1>(ndarray::external(V_MeasureErrors.data(),
                                                                                     ndarray::makeVector(int(V_MeasureErrors.size())),
                                                                                     ndarray::makeVector(1))));
      }

      B_Run = false;
      if (*P_I_NRejected != I_NRejected_Old) {
        B_Run = true;
      } else {
        for (int i_pos=0; i_pos < *P_I_NRejected; i_pos++){
          if ((*P_I_A1_Rejected)[i_pos] != I_A1_Rejected_Old[i_pos])
            B_Run = true;
        }
      }
      i_iter++;
      if ( i_iter >= I_NIter ) {
        B_Run = false;
      }
    }
    LOGLS_DEBUG(_log, "*P_I_NRejected = " << *P_I_NRejected);
    *P_I_A1_NotRejected = I_A1_OrigPos;
    LOGLS_DEBUG(_log, "PolyFit(x, y, deg, lReject, uReject, nIter, Args, ArgV) finished");
    return D_A1_Coeffs_Out;
  }

  /** **********************************************************************/

  template< typename T >
  ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                       ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                       size_t const I_Degree_In,
                                       T xRangeMin_In,
                                       T xRangeMax_In){
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.math.CurveFitting.PolyFit");
    LOGLS_DEBUG(_log, "PolyFit(x, y, deg, xRangeMin, xRangeMax) started");
    if (D_A1_X_In.getShape()[0] != D_A1_Y_In.getShape()[0]){
      std::string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: D_A1_X_In.getShape()[0](=");
      message += std::to_string(D_A1_X_In.getShape()[0]) +") != D_A1_Y_In.getShape()[0](=" + std::to_string(D_A1_Y_In.getShape()[0]) + ")";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    std::vector<std::string> S_A1_Args(1);
    S_A1_Args[0] = "XRANGE";
    std::vector<void *> PP_Args(1);
    ndarray::Array<float, 1, 1> xRange = ndarray::allocate(2);
    xRange[0] = xRangeMin_In;
    xRange[1] = xRangeMax_In;
    PTR(ndarray::Array<float, 1, 1>) pXRange(new ndarray::Array<float, 1, 1>(xRange));
    PP_Args[0] = &pXRange;
    LOGLS_DEBUG(_log, "PolyFit(x, y, deg, xRangeMin, xRangeMax) finishing");
    return PolyFit(D_A1_X_In,
                   D_A1_Y_In,
                   I_Degree_In,
                   S_A1_Args,
                   PP_Args);
  }

  template< typename T >
  ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                       ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                       size_t const I_Degree_In,
                                       std::vector<std::string> const& S_A1_Args_In,
                                       std::vector<void *> & ArgV){
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.math.CurveFitting.PolyFit");
    LOGLS_DEBUG(_log, "PolyFit(x, y, deg, Args, ArgV) started");
    if (D_A1_X_In.getShape()[0] != D_A1_Y_In.getShape()[0]){
      std::string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: D_A1_X_In.getShape()[0](=");
      message += std::to_string(D_A1_X_In.getShape()[0]) + ") != D_A1_Y_In.getShape()[0](=" + std::to_string(D_A1_Y_In.getShape()[0]) + ")";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    const int degree(I_Degree_In);
    LOGLS_DEBUG(_log, "D_A1_Y_In = " << D_A1_Y_In);
    size_t const nCoeffs(I_Degree_In + 1);
    LOGLS_DEBUG(_log, "nCoeffs set to " << nCoeffs);
    ndarray::Array<double, 1, 1> D_A1_Out = ndarray::allocate(nCoeffs);
    D_A1_Out.deep() = 0.;
    int i, j, I_Pos;

    const int nDataPoints(D_A1_X_In.getShape()[0]);

    ndarray::Array<double, 1, 1> D_A1_SDevSquare = ndarray::allocate(nDataPoints);

    bool B_HaveMeasureError = false;
    ndarray::Array<T, 1, 1> D_A1_MeasureErrors = ndarray::allocate(nDataPoints);
    std::string sTemp = "MEASURE_ERRORS";
    PTR(ndarray::Array<T, 1, 1>) P_D_A1_MeasureErrors(new ndarray::Array<T, 1, 1>(D_A1_MeasureErrors));
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
    {
      B_HaveMeasureError = true;
      P_D_A1_MeasureErrors.reset();
      P_D_A1_MeasureErrors = *((PTR(ndarray::Array<T, 1, 1>)*)ArgV[I_Pos]);
      if (P_D_A1_MeasureErrors->getShape()[0] != nDataPoints){
        std::string message("pfs::drp::stella::math::CurveFitting::PolyFit: KeyWordSet(MEASURE_ERRORS): ERROR:");
        message += "P_D_A1_MeasureErrors->getShape()[0](=" + std::to_string(P_D_A1_MeasureErrors->getShape()[0]);
        message += ") != nDataPoints(=" + std::to_string(nDataPoints) + ")";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      D_A1_MeasureErrors.deep() = *P_D_A1_MeasureErrors;
      LOGLS_DEBUG(_log, "B_HaveMeasureError set to TRUE");
      LOGLS_DEBUG(_log, "*P_D_A1_MeasureErrors set to " << *P_D_A1_MeasureErrors);
    }
    else{
      D_A1_MeasureErrors.deep() = 1.;
    }

    ndarray::Array<float, 1, 1> D_A1_XRange = ndarray::allocate(2);
    ndarray::Array<T, 1, 1> xNew;
    PTR(ndarray::Array<float, 1, 1>) P_D_A1_XRange(new ndarray::Array<float, 1, 1>(D_A1_XRange));
    sTemp = "XRANGE";
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
    {
      P_D_A1_XRange.reset();
      P_D_A1_XRange = *((PTR(ndarray::Array<float, 1, 1>)*)ArgV[I_Pos]);
      if (P_D_A1_XRange->getShape()[0] != 2){
        std::string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: P_D_A1_XRange->getShape()[0](=");
        message += std::to_string(P_D_A1_XRange->getShape()[0]) +") != 2";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }

      D_A1_XRange.deep() = *P_D_A1_XRange;
      LOGLS_DEBUG(_log, "*P_D_A1_XRange set to " << *P_D_A1_XRange);
      xNew = pfs::drp::stella::math::convertRangeToUnity(D_A1_X_In,
                                                         D_A1_XRange);
    }
    else{
      D_A1_XRange[0] = -1.;
      D_A1_XRange[1] = 1.;
      xNew = D_A1_X_In;
    }
    LOGLS_DEBUG(_log, "xNew = " << xNew);

    auto itM = D_A1_MeasureErrors.begin();
    for (auto it=D_A1_SDevSquare.begin(); it!=D_A1_SDevSquare.end(); ++it, ++itM){
        *it = ::pow(static_cast<double>(*itM), 2.0);
    }
    LOGLS_DEBUG(_log, "D_A1_SDevSquare set to " << D_A1_SDevSquare);
    ndarray::Array<T, 1, 1> D_A1_YFit = ndarray::allocate(nDataPoints);
    PTR(ndarray::Array<T, 1, 1>) P_D_A1_YFit(new ndarray::Array<T, 1, 1>(D_A1_YFit));
    sTemp = "YFIT";
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
    {
      P_D_A1_YFit.reset();
      P_D_A1_YFit = *((PTR(ndarray::Array<T, 1, 1>)*)ArgV[I_Pos]);
      LOGLS_DEBUG(_log, "KeyWord_Set(YFIT)");
      if (P_D_A1_YFit->getShape()[0] != D_A1_X_In.getShape()[0]){
        std::string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: P_D_A1_YFit->getShape()[0] != D_A1_X_In.getShape()[0]");
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    }
    P_D_A1_YFit->deep() = 0.;

    ndarray::Array<T, 1, 1> D_A1_Sigma = ndarray::allocate(nCoeffs);
    PTR(ndarray::Array<T, 1, 1>) P_D_A1_Sigma(new ndarray::Array<T, 1, 1>(D_A1_Sigma));
    sTemp = "SIGMA";
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
    {
      P_D_A1_Sigma.reset();
      P_D_A1_Sigma = *((PTR(ndarray::Array<T, 1, 1>)*)ArgV[I_Pos]);
      if (P_D_A1_Sigma->getShape()[0] != nCoeffs){
        std::string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: P_D_A1_Sigma->getShape()[0](=");
        message += std::to_string(P_D_A1_Sigma->getShape()[0]) +") != nCoeffs(=" + std::to_string(nCoeffs) + ")";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      LOGLS_DEBUG(_log, "KeyWord_Set(SIGMA): *P_D_A1_Sigma set to " << (*P_D_A1_Sigma));
    }
    P_D_A1_Sigma->deep() = 0.;

    ndarray::Array<T, 2, 2> D_A2_Covar = ndarray::allocate(nCoeffs, nCoeffs);
    PTR(ndarray::Array<T, 2, 2>) P_D_A2_Covar(new ndarray::Array<T, 2, 2>(D_A2_Covar));
    sTemp = "COVAR";
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
    {
      P_D_A2_Covar.reset();
      P_D_A2_Covar = *((PTR(ndarray::Array<T, 2, 2>)*)ArgV[I_Pos]);
      if (P_D_A2_Covar->getShape()[0] != nCoeffs){
        std::string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: P_D_A2_Covar->getShape()[0](=");
        message += std::to_string(P_D_A2_Covar->getShape()[0]) + ") != nCoeffs(=" + std::to_string(nCoeffs) + ")";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (P_D_A2_Covar->getShape()[1] != nCoeffs){
        std::string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: P_D_A2_Covar->getShape()[1](=");
        message += std::to_string(P_D_A2_Covar->getShape()[1]) + ") != nCoeffs(=" + std::to_string(nCoeffs) + ")";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      LOGLS_DEBUG(_log, "KeyWord_Set(COVAR): *P_D_A2_Covar set to " << (*P_D_A2_Covar));
    }
    P_D_A2_Covar->deep() = 0.;
    ndarray::Array<double, 2, 2> covarTemp = ndarray::allocate(P_D_A2_Covar->getShape());

    ndarray::Array<double, 1, 1> D_A1_B = ndarray::allocate(nCoeffs);
    ndarray::Array<double, 1, 1> D_A1_Z = ndarray::allocate(nDataPoints);
    D_A1_Z.deep() = 1.0;

    ndarray::Array<double, 1, 1> D_A1_WY = pfs::drp::stella::utils::typeCastNdArray(D_A1_Y_In, double(0.0));

    if (B_HaveMeasureError){
      D_A1_WY.deep() = D_A1_WY / D_A1_SDevSquare;
      covarTemp[ ndarray::makeVector( 0, 0 ) ] = sum(1./D_A1_SDevSquare);
      LOGLS_DEBUG(_log, "B_HaveMeasureError: (*P_D_A2_Covar)(0,0) set to "
                        << covarTemp[ ndarray::makeVector( 0, 0 ) ]);
    }
    else{
      covarTemp[ ndarray::makeVector( 0, 0 ) ] = nDataPoints;
      LOGLS_DEBUG(_log, "!B_HaveMeasureError: (*P_D_A2_Covar)(0,0) set to "
                        << covarTemp[ ndarray::makeVector(0, 0 ) ]);
    }

    D_A1_B[0] = sum(D_A1_WY);

    double D_Sum;
    for (int p = 1; p <= 2 * degree; p++){
      D_A1_Z.deep() = D_A1_Z * xNew;
      if (p < nCoeffs){
        D_A1_B[p] = sum(D_A1_WY * D_A1_Z);
      }
      if (B_HaveMeasureError){
        D_Sum = sum(D_A1_Z / D_A1_SDevSquare);
      }
      else{
        D_Sum = sum(D_A1_Z);
      }
      if (p - degree > 0){
        i = p - degree;
      }
      else{
        i = 0;
      }
      for (j = i; j <= degree; j++){
        covarTemp[ ndarray::makeVector( j, p-j ) ] = D_Sum;
      }
    }

    LOGLS_DEBUG(_log, "before InvertGaussJ: (*P_D_A2_Covar) = " << covarTemp);
    covarTemp.asEigen() = covarTemp.asEigen().inverse();
    LOGLS_DEBUG(_log, "(*P_D_A2_Covar) set to " << covarTemp);
    D_A1_Out.asEigen() = covarTemp.asEigen() * D_A1_B.asEigen();
    ndarray::Array<T, 1, 1> T_A1_Out = pfs::drp::stella::utils::typeCastNdArray(D_A1_Out, T(0));
    LOGLS_DEBUG(_log, "P_D_A1_YFit->size() = " << P_D_A1_YFit->getShape()[0]
                      << ": T_A1_Out set to " << T_A1_Out);

    P_D_A1_YFit->deep() = pfs::drp::stella::math::Poly(D_A1_X_In,
                                                       T_A1_Out,
                                                       D_A1_XRange[0],
                                                       D_A1_XRange[1]);

    for (int k = 0; k < static_cast<int>(nCoeffs); k++){
      (*P_D_A1_Sigma)[k] = covarTemp[ ndarray::makeVector( k, k ) ];
    }
    for (auto it = P_D_A1_Sigma->begin(); it != P_D_A1_Sigma->end(); ++it){
      *it = (*it > 0) ? sqrt(*it) : 1.;
    }
    LOGLS_DEBUG(_log, "(*P_D_A1_Sigma) set to " << (*P_D_A1_Sigma));
    LOGLS_DEBUG(_log, "*P_D_A1_YFit = " << *P_D_A1_YFit);

    float D_ChiSq = 0.;
    Eigen::Array<T, Eigen::Dynamic, 1> Diff = D_A1_Y_In.asEigen() - P_D_A1_YFit->asEigen();
    LOGLS_DEBUG(_log, "Diff set to " << Diff);
    ndarray::Array<T, 1, 1> Err_Temp = ndarray::allocate(nDataPoints);
    Err_Temp.asEigen() = Diff.pow(2);
    LOGLS_DEBUG(_log, "Err_Temp set to " << Err_Temp);
    if (B_HaveMeasureError){
      D_ChiSq = sum(Err_Temp / D_A1_SDevSquare);
      LOGLS_DEBUG(_log, "B_HaveMeasureError: D_ChiSq set to " << D_ChiSq);
    }
    else{
      D_ChiSq = sum(Err_Temp);
      LOGLS_DEBUG(_log, "!B_HaveMeasureError: D_ChiSq set to " << D_ChiSq);

      float dTemp = sqrt(D_ChiSq / (nDataPoints - nCoeffs));
      P_D_A1_Sigma->deep() = (*P_D_A1_Sigma) * dTemp;
      LOGLS_DEBUG(_log, "!B_HaveMeasureError: (*P_D_A1_Sigma) set to " << (*P_D_A1_Sigma));
    }
    LOGLS_DEBUG(_log, "returning D_A1_Out = " << D_A1_Out);
    LOGLS_DEBUG(_log, "PolyFit(x, y, deg, Args, ArgV) finished");
    ndarray::Array<float, 1, 1> out = pfs::drp::stella::utils::typeCastNdArray(T_A1_Out, float(0.0));
    return out;
  }

  template< typename T>
  ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                       ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                       size_t const I_Degree_In,
                                       T const D_LReject_In,
                                       T const D_HReject_In,
                                       size_t const I_NIter,
                                       T xRangeMin_In,
                                       T xRangeMax_In){
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.math.CurveFitting.PolyFit");
    LOGLS_DEBUG(_log, "PolyFit(x, y, deg, lReject, hReject, nIter, xRangeMin, xRangeMax) started");
    std::vector<std::string> S_A1_Args(1);
    S_A1_Args[0] = "XRANGE";
    std::vector<void *> PP_Args(1);
    ndarray::Array<float, 1, 1> xRange = ndarray::allocate(2);
    xRange[0] = xRangeMin_In;
    xRange[1] = xRangeMax_In;
    PTR(ndarray::Array<float, 1, 1>) p_xRange(new ndarray::Array<float, 1, 1>(xRange));
    PP_Args[0] = &p_xRange;
    ndarray::Array<float, 1, 1> D_A1_Out = ndarray::allocate(I_Degree_In + 1);
    D_A1_Out = pfs::drp::stella::math::PolyFit(D_A1_X_In,
                                               D_A1_Y_In,
                                               I_Degree_In,
                                               D_LReject_In,
                                               D_HReject_In,
                                               I_NIter,
                                               S_A1_Args,
                                               PP_Args);
    LOGLS_DEBUG(_log, "PolyFit returned D_A1_Out = " << D_A1_Out);
    LOGLS_DEBUG(_log, "PolyFit(x, y, deg, lReject, hReject, nIter, xRangeMin, xRangeMax) finished");
    return D_A1_Out;
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
  static float
  fitProfile1d(ndarray::ArrayRef<ImageT, 1, 1> const& data, // data
                           ndarray::ArrayRef<ImageT, 1, 1> const& dataVar,  // errors in data
                           ndarray::ArrayRef<lsst::afw::image::MaskPixel, 1, 1> const& traceMask, // set to 1 for points in the fiberTrace
                           ndarray::ArrayRef<ImageT, 1, 1> const& profile,   // profile to fit
                           const float clipNSigma, // clip at this many sigma
                           const bool fitBackground, // Should I fit the background?
                           ImageT &amp,              // amplitude of fit
                           ImageT &bkgd,             // sky level
                           ImageT &ampVar           // amp's variance
                          )
  {
      assert(data.size() == profile.size());

      if ((data.asEigen().sum() == 0.) || (profile.asEigen().sum() == 0.)){
          amp = 0.;
          return -1;
      }
      //
      // Check that the pixel variances are not 0
      //
      for (int i = 0; i < traceMask.size(); i++) {
          if (traceMask[i] == 0) { // bad pixels
              continue;
          }

          if (dataVar[i] < 0.00000000000000001){
              std::cout << "fitProfile1d: i = " << i << ": ERROR: dataVar = " << dataVar << std::endl;
              std::string message("fitProfile1d:");
              message += ": i = " + std::to_string(i) + ": ERROR: dataVar(" + std::to_string(i) + ") == 0.";
              throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
      }
      
      const int ndata = data.getShape()[0];
      const bool clipData = (clipNSigma > 0.) ? true : false;

      amp = 0.0;                        // amplitude of fit
      ampVar = 0.0;                    // amp's variance
      bkgd = 0.0;                       // sky level
#if defined(BKGD_VAR)
      double bkgdVar = 0.0;            // bkgd's variance
      double amp_bkgdCovar = 0;        // covariance between amp and bkgd
#endif

      if (traceMask.asEigen().sum() == 0) {
          return -2;
      }

      auto model  = ndarray::Array<ImageT, 1, 1>(ndata); // fit to data

      float rchi2 = -1; // reduced chi^2 (we're only fitting an amplitude)
      for (;;) {
          amp = 0.0;

          /// remove bad pixels marked by mask
          const int nGood = traceMask.asEigen().sum();
          if (nGood == 0) {
              return -3;                 // BAD
          }
          // Let us call the profile P and the intensity D in naming our sums
          double sum = 0.;
          double sum_PP = 0.;
          double sum_P = 0.;
          double sum_PD = 0.;
          double sum_D = 0.;
          for (int i=0; i < traceMask.size(); i++) {
              if (traceMask[i] == 0) { // bad pixels
                  continue;
              }

              double weight = 1/dataVar[i];
                  
              sum += weight;
              sum_P += weight*profile[i];
              sum_D += weight*data[i];
              sum_PD += weight*profile[i]*data[i];
              sum_PP += weight*profile[i]*profile[i];
          }
          const double D_Delta = sum*sum_PP - pow(sum_P, 2);

          if (fitBackground) {
              amp = (sum*sum_PD - sum_P*sum_D)/D_Delta;
              bkgd = (sum_PP*sum_D - sum_P*sum_PD)/D_Delta;

              ampVar = sum/D_Delta;
#if defined(BKGD_VAR)
              bkgdVar = sum_PP/D_Delta;
              amp_bkgdCovar = -sum_P/D_Delta;
#endif
          } else {
              amp = sum_PD/sum_PP;
              bkgd = 0;

              ampVar = 1/sum_PP;
#if defined(BKGD_VAR)
              bkgdVar = 0;
              amp_bkgdCovar = 0;
#endif
          }

          model.deep() = bkgd + amp*profile;

          float ChiSqr = 0.;
          int nPix = 0;                // number of unmasked pixels
          int nClip = 0;               // number of newly clipped pixels
          for (int i=0; i < traceMask.size(); i++) {
              if (traceMask[i] == 0) { // bad pixels
                  continue;
              }

              const float dchi2 = pow(data[i] - model[i], 2)/dataVar[i];
              nPix++;
              ChiSqr += dchi2;

              if (clipData && dchi2 > clipNSigma*clipNSigma) {
                  traceMask[i] = 0;
                  ++nClip;
              }
          }
          rchi2 = ChiSqr/(nPix - 1);
          
          if (std::fabs(amp) < 0.000001) {
              break;
          }
          if (nClip == 0) {          // we didn't clip any new pixels
              break;
          }
      }

      return rchi2;
  }
}

/************************************************************************************************************/

  template< typename ImageT>
  bool fitProfile2d(ndarray::Array<ImageT, 2, 1> const& ccdData, // data
                    ndarray::Array<ImageT, 2, 1> const& ccdDataVar,  // data's variance
                    ndarray::Array<lsst::afw::image::MaskPixel, 2, 1> const& traceMask, // set to 1 for points in the fiberTrace
                    ndarray::Array<ImageT, 2, 1> const& profile2d,   // profile of fibre trace
                    const bool fitBackground,                        // should I fit the background level?
                    const float clipNSigma,                          // clip at this many sigma
                    ndarray::Array<ImageT, 1, 1> & specAmp,          // returned spectrum
                    ndarray::Array<ImageT, 1, 1> & bkgd,             // returned background
                    ndarray::Array<ImageT, 1, 1> & specAmpVar        // spectrum's variance
                   )
  {
      const int height = ccdData.getShape()[0];
      const int width  = ccdData.getShape()[1];

    if (height != profile2d.getShape()[0]){
      std::string message("pfs::drp::stella::math::CurveFitting::fitProfile2d: ERROR: height(=");
      message += std::to_string(height) + ") != profile2d.getShape()[0](=" + std::to_string(profile2d.getShape()[0]) + ")";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (width != profile2d.getShape()[1]){
      std::string message("pfs::drp::stella::math::CurveFitting::fitProfile2d: ERROR: width(=");
      message += std::to_string(width) + ") != profile2d.getShape()[1](=" + std::to_string(profile2d.getShape()[1]) + ")";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (specAmp.getShape()[0] != height){
      std::string message("pfs::drp::stella::math::CurveFitting::fitProfile2d: ERROR: height(=");
      message += std::to_string(height) + ") != specAmp.getShape()[0](=" + std::to_string(specAmp.getShape()[0]) + ")";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    specAmp.deep() = 0;

    if (ccdDataVar.getShape()[0] != height) {
        std::string message("pfs::drp::stella::math::CurveFitting::fitProfile2d: ERROR: height(=");
        message += std::to_string(height) + ") != ccdDataVar.getShape()[0](=" + std::to_string(ccdDataVar.getShape()[0]) + ")";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (ccdDataVar.getShape()[1] != width) {
        std::string message("pfs::drp::stella::math::CurveFitting::fitProfile2d: ERROR: width(=");
        message += std::to_string(width) + ") != ccdDataVar.getShape()[1](=" + std::to_string(ccdDataVar.getShape()[1]) + ")";
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    for (int i = 0; i < height; i++) {
        float rchi2 = fitProfile1d(ccdData[i],
                                   ccdDataVar[i],
                                   traceMask[i],
                                   profile2d[i],
                                   clipNSigma,
                                   fitBackground, // should I fit the background?
                                   specAmp[i],    // output spectrum
                                   bkgd[i],       // output background level
                                   specAmpVar[i]  // output spectrum's variance
                                  );
      if (rchi2 < 0) {              // failed
          ;                         // need to set a bit in the output mask, but it's still binary (grr)
      }
    }

    return true;
  }

  template< typename T, typename U >
  ndarray::Array<T, 1, 1> chebyshev(ndarray::Array<T, 1, 1> const& x_In, ndarray::Array<U, 1, 1> const& coeffs_In){
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::chebyshev(a, coeffs) started" << std::endl;
    #endif
    int nCoeffs = coeffs_In.getShape()[0];
    std::cout << "pfs::drp::stella::math::CurveFitting::chebyshev: coeffs_In = " << nCoeffs << ": ";
    for (int i = 0; i < nCoeffs; ++i)
      printf("%.12f ", coeffs_In[i]);
    printf("\n");

    ndarray::Array<float, 1, 1> range = ndarray::allocate(2);
    range[0] = min(x_In);
    range[1] = max(x_In);
    std::cout << "chebyshev: range = " << range << std::endl;
    ndarray::Array<T, 1, 1> xScaled = pfs::drp::stella::math::convertRangeToUnity(x_In, range);
    std::cout << "chebyshev: xScaled = " << xScaled << std::endl;

    ndarray::Array<T, 1, 1> tmpArr = ndarray::allocate(x_In.getShape()[0]);
    ndarray::Array<T, 1, 1> c0Arr = ndarray::allocate(x_In.getShape()[0]);
    ndarray::Array<T, 1, 1> c1Arr = ndarray::allocate(x_In.getShape()[0]);
    ndarray::Array<T, 1, 1> yCalc = ndarray::allocate(x_In.getShape()[0]);
    float c0, c1, tmp;
    if (coeffs_In.getShape()[0] == 1){
        c0 = coeffs_In[0];
        c1 = 0;
    }
    else if (coeffs_In.getShape()[0] == 2){
        c0 = coeffs_In[0];
        c1 = coeffs_In[1];
    }
    else{
      ndarray::Array<float, 1, 1> x2 = ndarray::allocate(xScaled.getShape()[0]);
      x2.deep() = 2. * xScaled;
      c0 = coeffs_In[coeffs_In.getShape()[0] - 2];
      c1 = coeffs_In[coeffs_In.getShape()[0] - 1];
      std::cout << "chebyshev: c0 = " << c0 << ", c1 = " << c1 << std::endl;
      for (int i = 3; i <= coeffs_In.getShape()[0]; ++i){
        if (i == 3){
          tmp = c0;
          c0 = coeffs_In[coeffs_In.getShape()[0] - i] - c1;
          c1Arr.deep() = tmp + c1*x2;
        }
        else if (i == 4){
          tmp = c0;
          c0Arr.deep() = coeffs_In[coeffs_In.getShape()[0] - i] - c1Arr;
          c1Arr.deep() = tmp + c1Arr * x2;
        }
        else{
          tmpArr.deep() = c0Arr;
          c0Arr.deep() = coeffs_In[coeffs_In.getShape()[0] - i] - c1Arr;
          c1Arr.deep() = tmpArr + c1Arr * x2;
        }
        std::cout << "chebyshev: i = " << i << ": c0 = " << c0 << ", c0Arr = " << c0Arr << ", c1Arr = " << c1Arr << std::endl;
      }
    }
    if (coeffs_In.getShape()[0] < 3)
      yCalc.deep() = c0 + c1 * xScaled;
    else if (coeffs_In.getShape()[0] == 3)
      yCalc.deep() = c0 + c1Arr * xScaled;
    else
      yCalc.deep() = c0Arr + c1Arr * xScaled;
    std::cout << "chebyshev: yCalc = " << yCalc << std::endl;
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::chebyshev(a, coeffs) finished" << std::endl;
    #endif
    return yCalc;
  }

  ndarray::Array<float, 1, 1> gaussFit(ndarray::Array<float, 2, 1> const& xy_In,
                                       ndarray::Array<float, 1, 1> const& guess_In){
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::gaussFit(xy, guess) started" << std::endl;
    #endif
    gaussian_functor gf(xy_In.asEigen());
    Eigen::VectorXf guess(3);
    guess[0] = guess_In[0];
    guess[1] = guess_In[1];
    guess[2] = guess_In[2];
    Eigen::LevenbergMarquardt<gaussian_functor> solver(gf);
    solver.setXtol(1.0e-6);
    solver.setFtol(1.0e-6);
    solver.minimize(guess);
    ndarray::Array<float, 1, 1> result = ndarray::allocate(guess_In.getShape()[0]);
    result[0] = guess[0];
    result[1] = guess[1];
    result[2] = guess[2];
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::gaussFit(xy, guess) finished" << std::std::endl;
    #endif
    return result;
  }

  template ndarray::Array<float, 1, 1> chebyshev(ndarray::Array<float, 1, 1> const& x_In, ndarray::Array<float, 1, 1> const& coeffs_In);

  template ndarray::Array<float, 1, 1> Poly(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, float, float);

  template ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const, float const, float const, size_t const, std::vector<std::string> const&, std::vector<void *> &);
  template ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const, float, float);
  template ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const, float const, float const, size_t const, float, float);

  template bool fitProfile2d(ndarray::Array<float, 2, 1> const&,
                             ndarray::Array<float, 2, 1> const&,
                             ndarray::Array<lsst::afw::image::MaskPixel, 2, 1> const&,
                             ndarray::Array<float, 2, 1> const&,
                             const bool,
                             const float,
                             ndarray::Array<float, 1, 1> &,
                             ndarray::Array<float, 1, 1> &,
                             ndarray::Array<float, 1, 1> &);
                
}}}}
