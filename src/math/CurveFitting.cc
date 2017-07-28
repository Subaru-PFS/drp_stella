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
    LOG_LOGGER _log = LOG_GET("pfs::drp::stella::math::CurfFitting::PolyFit");

    LOGLS_DEBUG(_log, "CurveFitting::PolyFit(x, y, deg, lReject, uReject, nIter, Args, ArgV) started");
    if (D_A1_X_In.getShape()[0] != D_A1_Y_In.getShape()[0]){
      std::string message("pfs::drp::stella::math::CurfFitting::PolyFit: ERROR: D_A1_X_In.getShape()[0](=");
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
            ((D_Dev < 0) && (D_Dev > (D_LReject_In * D_SDev))) ||
            ((D_Dev >= 0) && (D_Dev < (D_UReject_In * D_SDev)))){
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
      if (*P_I_NRejected != I_NRejected_Old)
        B_Run = true;
      else{
        for (int i_pos=0; i_pos < *P_I_NRejected; i_pos++){
          if ((*P_I_A1_Rejected)[i_pos] != I_A1_Rejected_Old[i_pos])
            B_Run = true;
        }
      }
      i_iter++;
      if ( i_iter >= I_NIter )
        B_Run = false;
    }
    LOGLS_DEBUG(_log, "*P_I_NRejected = " << *P_I_NRejected);
    *P_I_A1_NotRejected = I_A1_OrigPos;
    LOGLS_DEBUG(_log, "CurveFitting::PolyFit(x, y, deg, lReject, uReject, nIter, Args, ArgV) finished");
    return D_A1_Coeffs_Out;
  }

  /** **********************************************************************/

  template< typename T >
  ndarray::Array<float, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                       ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                       size_t const I_Degree_In,
                                       T xRangeMin_In,
                                       T xRangeMax_In){
    LOG_LOGGER _log = LOG_GET("pfs::drp::stella::math::CurfFitting::PolyFit");
    LOGLS_DEBUG(_log, "CurveFitting::PolyFit(x, y, deg, xRangeMin, xRangeMax) started");
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
    LOG_LOGGER _log = LOG_GET("pfs::drp::stella::math::CurfFitting::PolyFit");
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
    LOG_LOGGER _log = LOG_GET("pfs::drp::stella::math::CurfFitting::PolyFit");
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

  template< typename ImageT, typename SlitFuncT>
  bool LinFitBevingtonNdArray(ndarray::Array<ImageT, 2, 1> const& D_A2_CCD_In,
                              ndarray::Array<SlitFuncT, 2, 1> const& D_A2_SF_In,
                              ndarray::Array<ImageT, 1, 1> & D_A1_SP_Out,
                              ndarray::Array<ImageT, 1, 1> & D_A1_Sky_Out,
                              bool B_WithSky,
                              std::vector<std::string> const& S_A1_Args_In,
                              std::vector<void *> & ArgV_In)
    /// MEASURE_ERRORS_IN = ndarray::Array<ImageT,2,1>(D_A2_CCD_In.getShape())   : in
    /// REJECT_IN = ImageT                                                       : in
    /// MASK_INOUT = ndarray::Array<unsigned short, 1, 1>(D_A2_CCD_In.getShape()): in/out
    /// CHISQ_OUT = ndarray::Array<ImageT, 1, 1>(D_A2_CCD_In.getShape()[0])      : out
    /// Q_OUT = ndarray::Array<ImageT, 1, 1>(D_A2_CCD_In.getShape()[0])          : out
    /// SIGMA_OUT = ndarray::Array<ImageT, 2, 1>(D_A2_CCD_In.getShape()[0], 2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = ndarray::Array<ImageT, 2, 1>(D_A2_CCD_In.getShape()[0], D_A2_CCD_In.getShape()[1]) : out
  {
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::LinFitBevingtonNdArray(D_A2_CCD, D_A2_SF, SP, Sky, withSky, Args, ArgV) started" << std::endl;
    #endif
    #ifdef __DEBUG_FITARR__
      std::cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr) started" << std::endl;
      std::cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr): D_A2_CCD_In = " << D_A2_CCD_In << std::endl;
      std::cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr): D_A2_SF_In = " << D_A2_SF_In << std::endl;
    #endif
    if (D_A2_CCD_In.getShape()[0] != D_A2_SF_In.getShape()[0]){
      std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
      message += std::to_string(D_A2_CCD_In.getShape()[0]) + ") != D_A2_SF_In.getShape()[0](=" + std::to_string(D_A2_SF_In.getShape()[0]) + ")";
      std::cout << message << std::endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (D_A2_CCD_In.getShape()[1] != D_A2_SF_In.getShape()[1]){
      std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[1](=");
      message += std::to_string(D_A2_CCD_In.getShape()[1]) + ") != D_A2_SF_In.getShape()[1](=" + std::to_string(D_A2_SF_In.getShape()[1]) + ")";
      std::cout << message << std::endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (D_A1_SP_Out.getShape()[0] != D_A2_CCD_In.getShape()[0]){
      std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
      message += std::to_string(D_A2_CCD_In.getShape()[0]) + ") != D_A1_SP_Out.getShape()[0](=" + std::to_string(D_A1_SP_Out.getShape()[0]) + ")";
      std::cout << message << std::endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (D_A1_Sky_Out.getShape()[0] != D_A2_CCD_In.getShape()[0]){
      std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
      message += std::to_string(D_A2_CCD_In.getShape()[0]) + ") != D_A1_Sky_Out.getShape()[0](=" + std::to_string(D_A1_Sky_Out.getShape()[0]) + ")";
      std::cout << message << std::endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    int i, I_ArgPos = 0;
    int I_KeywordSet_MeasureErrors, I_KeywordSet_Reject, I_KeywordSet_Mask, I_KeywordSet_ChiSq, I_KeywordSet_Q, I_KeywordSet_Sigma, I_KeywordSet_YFit;
    D_A1_SP_Out.deep() = 0;
    D_A1_Sky_Out.deep() = 0;

    std::vector<std::string> S_A1_Args_Fit(10);
    for (auto it = S_A1_Args_Fit.begin(); it != S_A1_Args_Fit.end(); ++it)
      *it = " ";
    std::vector<void *> Args_Fit(10);

    ndarray::Array<ImageT, 1, 1> D_A1_Sigma = ndarray::allocate(D_A2_CCD_In.getShape()[1]);
    PTR(ndarray::Array<ImageT, 1, 1>) P_D_A1_Sigma(new ndarray::Array<ImageT, 1, 1>(D_A1_Sigma));

    ndarray::Array<ImageT, 2, 1> D_A2_Sigma = ndarray::allocate(D_A2_CCD_In.getShape()[0], D_A2_CCD_In.getShape()[1]);
    PTR(ndarray::Array<ImageT, 2, 1>) P_D_A2_Sigma(new ndarray::Array<ImageT, 2, 1>(D_A2_Sigma));
    I_KeywordSet_MeasureErrors = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MEASURE_ERRORS_IN");
    if (I_KeywordSet_MeasureErrors >= 0)
    {
      P_D_A2_Sigma.reset();
      P_D_A2_Sigma = *((PTR(ndarray::Array<ImageT, 2, 1>)*)ArgV_In[I_KeywordSet_MeasureErrors]);
      if (P_D_A2_Sigma->getShape()[0] != D_A2_CCD_In.getShape()[0]){
        std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
        message += std::to_string(D_A2_CCD_In.getShape()[0]) + ") != P_D_A2_Sigma->getShape()[0](=" + std::to_string(P_D_A2_Sigma->getShape()[0]) + ")";
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (P_D_A2_Sigma->getShape()[1] != D_A2_CCD_In.getShape()[1]){
        std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[1](=");
        message += std::to_string(D_A2_CCD_In.getShape()[1]) + ") != P_D_A2_Sigma->getShape()[1](=" + std::to_string(P_D_A2_Sigma->getShape()[1]) + ")";
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_FITARR__
        std::cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A2_Sigma = " << *P_D_A2_Sigma << std::endl;
      #endif
      S_A1_Args_Fit[I_ArgPos] = "MEASURE_ERRORS_IN";
      I_ArgPos++;
    }

    ndarray::Array<ImageT, 1, 1> D_A1_ChiSq = ndarray::allocate(D_A2_CCD_In.getShape()[0]);
    PTR(ndarray::Array<ImageT, 1, 1>) P_D_A1_ChiSq(new ndarray::Array<ImageT, 1, 1>(D_A1_ChiSq));
    I_KeywordSet_ChiSq = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "CHISQ_OUT");
    if (I_KeywordSet_ChiSq >= 0)
    {
      P_D_A1_ChiSq.reset();
      P_D_A1_ChiSq = *((PTR(ndarray::Array<ImageT, 1, 1>)*)ArgV_In[I_KeywordSet_ChiSq]);
      if (P_D_A1_ChiSq->getShape()[0] != D_A2_CCD_In.getShape()[0]){
        std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
        message += std::to_string(D_A2_CCD_In.getShape()[0]) + ") != P_D_A1_ChiSq->getShape()[0](=" + std::to_string(P_D_A1_ChiSq->getShape()[0]) + ")";
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      S_A1_Args_Fit[I_ArgPos] = "CHISQ_OUT";
      I_ArgPos++;
    }

    ndarray::Array<ImageT, 1, 1> D_A1_Q = ndarray::allocate(D_A2_CCD_In.getShape()[0]);
    PTR(ndarray::Array<ImageT, 1, 1>) P_D_A1_Q(new ndarray::Array<ImageT, 1, 1>(D_A1_Q));
    I_KeywordSet_Q = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "Q_OUT");
    if (I_KeywordSet_Q >= 0)
    {
      P_D_A1_Q.reset();
      P_D_A1_Q = *((PTR(ndarray::Array<ImageT, 1, 1>)*)ArgV_In[I_KeywordSet_Q]);
      if (P_D_A1_Q->getShape()[0] != D_A2_CCD_In.getShape()[0]){
        std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
        message += std::to_string(D_A2_CCD_In.getShape()[0]) + ") != P_D_A1_Q->getShape()[0](=" + std::to_string(P_D_A1_Q->getShape()[0]) + ")";
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      S_A1_Args_Fit[I_ArgPos] = "Q_OUT";
      I_ArgPos++;
    }

    ndarray::Array<ImageT, 1, 1> D_A1_Sigma_Out = ndarray::allocate(2);
    PTR(ndarray::Array<ImageT, 1, 1>) P_D_A1_Sigma_Out(new ndarray::Array<ImageT, 1, 1>(D_A1_Sigma_Out));

    ndarray::Array<ImageT, 2, 1> D_A2_Sigma_Out = ndarray::allocate(D_A2_CCD_In.getShape()[0], 2);
    PTR(ndarray::Array<ImageT, 2, 1>) P_D_A2_Sigma_Out(new ndarray::Array<ImageT, 2, 1>(D_A2_Sigma_Out));
    I_KeywordSet_Sigma = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "SIGMA_OUT");
    if (I_KeywordSet_Sigma >= 0)
    {
      P_D_A2_Sigma_Out.reset();
      P_D_A2_Sigma_Out = *((PTR(ndarray::Array<ImageT, 2, 1>)*)ArgV_In[I_KeywordSet_Sigma]);
      if (P_D_A2_Sigma_Out->getShape()[0] != D_A2_CCD_In.getShape()[0]){
        std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
        message += std::to_string(D_A2_CCD_In.getShape()[0]) + ") != P_D_A2_Sigma_Out->getShape()[0](=" + std::to_string(P_D_A2_Sigma_Out->getShape()[0]) + ")";
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (P_D_A2_Sigma_Out->getShape()[1] != 2){
        std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: P_D_A2_Sigma->getShape()[1](=");
        message += std::to_string(P_D_A2_Sigma->getShape()[0]) + ") != 2";
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      S_A1_Args_Fit[I_ArgPos] = "SIGMA_OUT";
      I_ArgPos++;
    }

    ndarray::Array<ImageT, 1, 1> D_A1_YFit = ndarray::allocate(D_A2_CCD_In.getShape()[1]);
    PTR(ndarray::Array<ImageT, 1, 1>) P_D_A1_YFit(new ndarray::Array<ImageT, 1, 1>(D_A1_YFit));

    ndarray::Array<ImageT, 2, 1> D_A2_YFit = ndarray::allocate(D_A2_CCD_In.getShape()[0], D_A2_CCD_In.getShape()[1]);
    PTR(ndarray::Array<ImageT, 2, 1>) P_D_A2_YFit(new ndarray::Array<ImageT, 2, 1>(D_A2_YFit));
    I_KeywordSet_YFit = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "YFIT_OUT");
    if (I_KeywordSet_YFit >= 0)
    {
      P_D_A2_YFit.reset();
      P_D_A2_YFit = *((PTR(ndarray::Array<ImageT, 2, 1>)*)ArgV_In[I_KeywordSet_YFit]);
      if (P_D_A2_YFit->getShape()[0] != D_A2_CCD_In.getShape()[0]){
        std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
        message += std::to_string(D_A2_CCD_In.getShape()[0]) + ") != P_D_A2_YFit->getShape()[0](=" + std::to_string(P_D_A2_YFit->getShape()[0]) + ")";
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (P_D_A2_YFit->getShape()[1] != D_A2_CCD_In.getShape()[1]){
        std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[1](=");
        message += std::to_string(D_A2_CCD_In.getShape()[1]) + ") != P_D_A2_YFit->getShape()[1](=" + std::to_string(P_D_A2_YFit->getShape()[1]) + ")";
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      S_A1_Args_Fit[I_ArgPos] = "YFIT_OUT";
      I_ArgPos++;
    }

    PTR(ImageT) P_D_Reject(new ImageT(-1));
    I_KeywordSet_Reject = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "REJECT_IN");
    if (I_KeywordSet_Reject >= 0)
    {
      P_D_Reject.reset();
      P_D_Reject = *((PTR(ImageT)*)ArgV_In[I_KeywordSet_Reject]);
      #ifdef __DEBUG_FITARR__
        std::cout << "CFits::LinFitBevington2D: P_D_Reject = " << *P_D_Reject << std::endl;
      #endif
      S_A1_Args_Fit[I_ArgPos] = "REJECT_IN";
      I_ArgPos++;
    }

    ndarray::Array<lsst::afw::image::MaskPixel, 1, 1> I_A1_Mask = ndarray::allocate(D_A2_CCD_In.getShape()[1]);
    PTR(ndarray::Array<lsst::afw::image::MaskPixel, 1, 1>) P_I_A1_Mask(new ndarray::Array<lsst::afw::image::MaskPixel, 1, 1>(I_A1_Mask));

    ndarray::Array<lsst::afw::image::MaskPixel, 2, 1> I_A2_Mask = ndarray::allocate(D_A2_CCD_In.getShape()[0], D_A2_CCD_In.getShape()[1]);
    I_A2_Mask.deep() = 1;
    PTR(ndarray::Array<lsst::afw::image::MaskPixel, 2, 1>) P_I_A2_Mask(new ndarray::Array<lsst::afw::image::MaskPixel, 2, 1>(I_A2_Mask));
    I_KeywordSet_Mask = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MASK_INOUT");
    if (I_KeywordSet_Mask >= 0)
    {
      P_I_A2_Mask.reset();
      P_I_A2_Mask = *((PTR(ndarray::Array<lsst::afw::image::MaskPixel, 2, 1>)*)ArgV_In[I_KeywordSet_Mask]);
      if (P_I_A2_Mask->getShape()[0] != D_A2_CCD_In.getShape()[0]){
        std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
        message += std::to_string(D_A2_CCD_In.getShape()[0]) + ") != P_I_A2_Mask->getShape()[0](=" + std::to_string(P_I_A2_Mask->getShape()[0]) + ")";
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (P_I_A2_Mask->getShape()[1] != D_A2_CCD_In.getShape()[1]){
        std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[1](=");
        message += std::to_string(D_A2_CCD_In.getShape()[1]) + ") != P_I_A2_Mask->getShape()[1](=" + std::to_string(P_I_A2_Mask->getShape()[1]) + ")";
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_FITARR__
        std::cout << "CFits::LinFitBevington2D: P_I_A2_Mask = " << *P_I_A2_Mask << std::endl;
      #endif
      S_A1_Args_Fit[I_ArgPos] = "MASK_INOUT";
      I_ArgPos++;
    }

    bool B_DoFit = true;
    for (i = 0; i < D_A2_CCD_In.getShape()[0]; i++)
    {
      I_ArgPos = 0;
      if (I_KeywordSet_MeasureErrors >= 0){
        *P_D_A1_Sigma = (*P_D_A2_Sigma)[ndarray::view(i)()];
        #ifdef __DEBUG_FITARR__
          std::cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A1_Sigma set to " << *P_D_A1_Sigma << std::endl;
        #endif
        Args_Fit[I_ArgPos] = &P_D_A1_Sigma;
        #ifdef __DEBUG_FITARR__
          std::cout << "CFits::LinFitBevington(Array, Array, Array, Array): PP_Args_Fit[I_ArgPos=" << I_ArgPos << "] = " << *((PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>)*)Args_Fit[I_ArgPos]) << std::endl;
        #endif
        I_ArgPos++;
      }

      if (I_KeywordSet_ChiSq >= 0){
        Args_Fit[I_ArgPos] = &((*P_D_A1_ChiSq)[i]);
        I_ArgPos++;
      }

      if (I_KeywordSet_Q >= 0){
        Args_Fit[I_ArgPos] = &((*P_D_A1_Q)[i]);
        I_ArgPos++;
      }

      if (I_KeywordSet_Sigma >= 0){
        *P_D_A1_Sigma_Out = (*P_D_A2_Sigma_Out)[ndarray::view(i)()];
        Args_Fit[I_ArgPos] = &P_D_A1_Sigma_Out;
        I_ArgPos++;
      }

      if (I_KeywordSet_YFit >= 0){
        *P_D_A1_YFit = (*P_D_A2_YFit)[ndarray::view(i)()];
        Args_Fit[I_ArgPos] = &P_D_A1_YFit;
        I_ArgPos++;
      }

      if (I_KeywordSet_Reject >= 0){
        Args_Fit[I_ArgPos] = &P_D_Reject;
        I_ArgPos++;
      }

      B_DoFit = true;
      if (I_KeywordSet_Mask >= 0){
        *P_I_A1_Mask = (*P_I_A2_Mask)[ndarray::view(i)()];
        Args_Fit[I_ArgPos] = &P_I_A1_Mask;
        I_ArgPos++;
        if (ndarray::sum(*P_I_A1_Mask) == 0)
          B_DoFit = false;
      }

      #ifdef __DEBUG_FITARR__
        std::cout << "CFits::LinFitBevington: Starting Fit1D: D_A2_CCD_In(i=" << i << ", *) = " << D_A2_CCD_In[ndarray::view(i)()] << std::endl;
      #endif
      if (B_DoFit){
        #ifdef __DEBUG_FITARR__
          std::cout << "CFits::LinFitBevington: D_A2_SF_In(i=" << i << ", *) = " << D_A2_SF_In[ndarray::view(i)()] << std::endl;
        #endif
        ndarray::Array<ImageT, 1, 1> D_A1_CCD(D_A2_CCD_In[ndarray::view(i)()]);
        ndarray::Array<SlitFuncT, 1, 1> D_A1_SF(D_A2_SF_In[ndarray::view(i)()]);
        int status = math::LinFitBevingtonNdArray(D_A1_CCD,
                                                  D_A1_SF,
                                                  D_A1_SP_Out[i],
                                                  D_A1_Sky_Out[i],
                                                  B_WithSky,
                                                  S_A1_Args_Fit,
                                                  Args_Fit);
        if (status != 1){
          #ifdef __WARNINGS_ON__
            std::string message("CFits::LinFitBevington: WARNING: LinFitBevington(" +
                           "D_A1_CCD, D_A1_SF, D_A1_SP_Out[i], D_A1_Sky_Out[i], " +
                           "B_WithSky, S_A1_Args_Fit, Args_Fit" +
                           ") returned status = ");
            message += std::to_string(status);
            std::cout << message << std::endl;
            std::cout << "CFits::LinFitBevington: D_A2_SF_In(0, *) = " << D_A2_SF_In[ndarray::view(0)()] << ": LinFitBevingtonNdArray returned status = " << status << std::endl;
          #endif
        }
      }
      #ifdef __DEBUG_FITARR__
        std::cout << "CFits::LinFitBevington(Array, Array, Array, Array): D_A1_SP_Out(i=" << i << ") set to " << D_A1_SP_Out[i] << std::endl;
        std::cout << "CFits::LinFitBevington(Array, Array, Array, Array): D_A1_Sky_Out(i=" << i << ") set to " << D_A1_Sky_Out[i] << std::endl;
      #endif

      if (I_KeywordSet_Sigma >= 0){
        #ifdef __DEBUG_FITARR__
          std::cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A1_Sigma_Out = " << (*P_D_A1_Sigma_Out) << std::endl;
          std::cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A2_Sigma_Out(i=" << i << ",*) = " << (*P_D_A2_Sigma_Out)[ndarray::view(i)()] << std::endl;
        #endif
        (*P_D_A2_Sigma_Out)[ndarray::view(i)()] = (*P_D_A1_Sigma_Out);
      }

      if (I_KeywordSet_YFit >= 0){
        (*P_D_A2_YFit)[ndarray::view(i)()] = (*P_D_A1_YFit);
        #ifdef __DEBUG_FITARR__
          std::cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A2_YFit(i=" << i << ",*) set to " << (*P_D_A2_YFit)[ndarray::view(i)()] << std::endl;
        #endif
      }

      if (I_KeywordSet_Mask >= 0){
        (*P_I_A2_Mask)[ndarray::view(i)()] = (*P_I_A1_Mask);
        #ifdef __DEBUG_FITARR__
          std::cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_I_A1_Mask = " << (*P_I_A1_Mask) << std::endl;
          std::cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_I_A2_Mask(i=" << i << ",*) set to " << (*P_I_A2_Mask)[ndarray::view(i)()] << std::endl;
        #endif
      }
    }
    #ifdef __DEBUG_FITARR__
      std::cout << "CFits::LinFitBevington(Array, Array, Array, Array): D_A1_SP_Out = " << D_A1_SP_Out << std::endl;
    #endif
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::LinFitBevingtonNdArray(D_A2_CCD, D_A2_SF, SP, Sky, withSky, Args, ArgV) finished" << std::endl;
    #endif
    return true;
  }

  template<typename ImageT, typename SlitFuncT>
  SpectrumBackground< ImageT > LinFitBevingtonNdArray( ndarray::Array<ImageT, 1, 1> const& D_A1_CCD_In,
                                                       ndarray::Array<SlitFuncT, 1, 1> const& D_A1_SF_In,
                                                       bool B_WithSky )
  {
    SpectrumBackground< ImageT > out;
    std::vector< std::string > args(1, " ");
    std::vector< void * > argV(1);
    LinFitBevingtonNdArray( D_A1_CCD_In,
                            D_A1_SF_In,
                            out.spectrum,
                            out.background,
                            B_WithSky,
                            args,
                            argV );
    return out;
  }

  template<typename ImageT, typename SlitFuncT>
  int LinFitBevingtonNdArray(ndarray::Array<ImageT, 1, 1> const& D_A1_CCD_In,
                             ndarray::Array<SlitFuncT, 1, 1> const& D_A1_SF_In,
                             ImageT &D_SP_Out,
                             ImageT &D_Sky_Out,
                             bool B_WithSky,
                             std::vector<std::string> const& S_A1_Args_In,
                             std::vector<void *> & ArgV_In)
  {
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::LinFitBevingtonNdArray(D_A1_CCD, D_A1_SF, SP, Sky, withSky, Args, ArgV) started" << std::endl;
    #endif
    int status = 1;
    #ifdef __DEBUG_FIT__
      std::cout << "CFits::LinFitBevington(Array, Array, float, float, bool, CSArr, PPArr) started" << std::endl;
      std::cout << "CFits::LinFitBevington: D_A1_CCD_In = " << D_A1_CCD_In << std::endl;
      std::cout << "CFits::LinFitBevington: D_A1_SF_In = " << D_A1_SF_In << std::endl;
    #endif

    if (D_A1_CCD_In.size() != D_A1_SF_In.size()){
      std::string message("CFits::LinFitBevington: ERROR: D_A1_CCD_In.size(=");
      message += std::to_string(D_A1_CCD_In.size()) + ") != D_A1_SF_In.size(=" + std::to_string(D_A1_SF_In.size()) + ")";
      std::cout << message << std::endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    //  /// Set D_A1_SF_In to zero where D_A1_CCD_In == zero
    ndarray::Array<SlitFuncT, 1, 1> D_A1_SF = ndarray::allocate(D_A1_SF_In.getShape()[0]);
    D_A1_SF.deep() = D_A1_SF_In;
    ndarray::Array<ImageT, 1, 1> D_A1_CCD = ndarray::allocate(D_A1_CCD_In.getShape()[0]);
    D_A1_CCD.deep() = D_A1_CCD_In;

    if ((D_A1_CCD_In.asEigen().sum() == 0.) || (D_A1_SF.asEigen().sum() == 0.)){
      #ifdef __WARNINGS_ON__
        std::cout << "CFits::LinFitBevington: Warning: (D_A1_CCD_In.sum(=" << D_A1_CCD_In.asEigen().sum() << " == 0.) || (D_A1_SF.sum(=" << D_A1_SF.asEigen().sum() << ") == 0.) => returning false" << std::endl;
      #endif
      D_SP_Out = 0.;
      D_Sky_Out = 0.;
      status = 0;
      return status;
    }
    int i, I_Pos;
    int I_KeywordSet_Reject, I_KeywordSet_Mask, I_KeywordSet_MeasureErrors, I_KeywordSet_SigmaOut, I_KeywordSet_ChiSqOut, I_KeywordSet_QOut, I_KeywordSet_YFitOut, I_KeywordSet_AllowSkyLTZero, I_KeywordSet_AllowSpecLTZero;
    float sigdat;
    const int ndata(D_A1_CCD_In.getShape()[0]);
    ndarray::Array<ImageT, 1, 1> D_A1_Sig = ndarray::allocate(ndata);
    D_A1_Sig.deep() = 0.;
    PTR(ndarray::Array<ImageT, 1, 1>) P_D_A1_Sig(new ndarray::Array<ImageT, 1, 1>(D_A1_Sig));
    ndarray::Array<ImageT, 1, 1> D_A1_WT = ndarray::allocate(ndata);

    /// a: D_Sky_Out
    /// b: D_SP_Out
    /// x: D_A1_SF_In
    /// y: D_A1_CCD_In
    bool B_AllowSkyLTZero = false;
    I_KeywordSet_AllowSkyLTZero = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "ALLOW_SKY_LT_ZERO");
    if (I_KeywordSet_AllowSkyLTZero >= 0){
      if(*((int*)ArgV_In[I_KeywordSet_AllowSkyLTZero]) > 0){
        B_AllowSkyLTZero = true;
        std::cout << "CFits::LinFitBevington: KeyWord_Set(ALLOW_SKY_LT_ZERO)" << std::endl;
      }
    }

    bool B_AllowSpecLTZero = false;
    I_KeywordSet_AllowSpecLTZero = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "ALLOW_SPEC_LT_ZERO");
    if (I_KeywordSet_AllowSpecLTZero >= 0){
      if (I_KeywordSet_AllowSkyLTZero < 0){
        if (*((int*)ArgV_In[I_KeywordSet_AllowSkyLTZero]) > 0){
          B_AllowSpecLTZero = true;
          std::cout << "CFits::LinFitBevington: KeyWord_Set(ALLOW_SPEC_LT_ZERO)" << std::endl;
        }
      }
    }

    float D_Reject(-1.);
    I_KeywordSet_Reject = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "REJECT_IN");
    if (I_KeywordSet_Reject >= 0)
    {
      D_Reject = *(float*)ArgV_In[I_KeywordSet_Reject];
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: KeyWord_Set(REJECT_IN): D_Reject = " << D_Reject << std::endl;
      #endif
    }
    bool B_Reject = false;
    if (D_Reject > 0.)
      B_Reject = true;

    ndarray::Array<lsst::afw::image::MaskPixel, 1, 1> I_A1_Mask_Orig = ndarray::allocate(ndata);
    ndarray::Array<lsst::afw::image::MaskPixel, 1, 1> I_A1_Mask = ndarray::allocate(ndata);
    I_A1_Mask.deep() = 1;
    PTR(ndarray::Array<lsst::afw::image::MaskPixel, 1, 1>) P_I_A1_Mask(new ndarray::Array<lsst::afw::image::MaskPixel, 1, 1>(I_A1_Mask));
    I_KeywordSet_Mask = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MASK_INOUT");
    if (I_KeywordSet_Mask >= 0)
    {
      P_I_A1_Mask.reset();
      P_I_A1_Mask = *((PTR(ndarray::Array<lsst::afw::image::MaskPixel, 1, 1>)*)ArgV_In[I_KeywordSet_Mask]);
      if (P_I_A1_Mask->getShape()[0] != ndata){
        std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: P_I_A1_Mask->getShape()[0](=");
        message += std::to_string(P_I_A1_Mask->getShape()[0]) + ") != ndata(=" + std::to_string(ndata) + ")";
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: KeyWord_Set(MASK_INOUT): *P_I_A1_Mask = " << *P_I_A1_Mask << std::endl;
      #endif
    }
    I_A1_Mask_Orig.deep() = *P_I_A1_Mask;
    #ifdef __DEBUG_FIT__
      std::cout << "CFits::LinFitBevington: *P_I_A1_Mask set to " << *P_I_A1_Mask << std::endl;
      std::cout << "CFits::LinFitBevington: I_A1_Mask_Orig set to " << I_A1_Mask_Orig << std::endl;
    #endif

    ndarray::Array<ImageT, 1, 1> D_A1_Sigma_Out = ndarray::allocate(2);
    PTR(ndarray::Array<ImageT, 1, 1>) P_D_A1_Sigma_Out(new ndarray::Array<ImageT, 1, 1>(D_A1_Sigma_Out));
    I_KeywordSet_SigmaOut = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "SIGMA_OUT");
    if (I_KeywordSet_SigmaOut >= 0)
    {
      P_D_A1_Sigma_Out.reset();
      P_D_A1_Sigma_Out = *(PTR(ndarray::Array<ImageT, 1, 1>)*)ArgV_In[I_KeywordSet_SigmaOut];
      if (P_D_A1_Sigma_Out->getShape()[0] != 2){
        std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: P_D_A1_Sigma_Out->getShape()[0](=");
        message += std::to_string(P_D_A1_Sigma_Out->getShape()[0]) + ") != 2";
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: KeyWord_Set(SIGMA_OUT)" << std::endl;
      #endif
    }
    P_D_A1_Sigma_Out->deep() = 0.;
    #ifdef __DEBUG_FIT__
      std::cout << "CFits::LinFitBevington: *P_D_A1_Sigma_Out set to " << *P_D_A1_Sigma_Out << std::endl;
    #endif

    PTR(ImageT) P_D_ChiSqr_Out(new ImageT(0.));
    I_KeywordSet_ChiSqOut = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "CHISQ_OUT");
    if (I_KeywordSet_ChiSqOut >= 0)
    {
      P_D_ChiSqr_Out.reset();
      P_D_ChiSqr_Out = *(PTR(ImageT)*)ArgV_In[I_KeywordSet_ChiSqOut];
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: KeyWord_Set(CHISQ_OUT)" << std::endl;
      #endif
    }
    *P_D_ChiSqr_Out = 0.;
    #ifdef __DEBUG_FIT__
      std::cout << "CFits::LinFitBevington: *P_D_ChiSqr_Out set to " << *P_D_ChiSqr_Out << std::endl;
    #endif

    PTR(ImageT) P_D_Q_Out(new ImageT(0.));
    I_KeywordSet_QOut = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "Q_OUT");
    if (I_KeywordSet_QOut >= 0)
    {
      P_D_Q_Out.reset();
      P_D_Q_Out = *(PTR(ImageT)*)ArgV_In[I_KeywordSet_QOut];
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: KeyWord_Set(Q_OUT)" << std::endl;
      #endif
    }
    *P_D_Q_Out = 1.;
    #ifdef __DEBUG_FIT__
      std::cout << "CFits::LinFitBevington: *P_D_Q_Out set to " << *P_D_Q_Out << std::endl;
    #endif

    D_SP_Out = 0.0;
    I_KeywordSet_MeasureErrors = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MEASURE_ERRORS_IN");
    if (I_KeywordSet_MeasureErrors >= 0)
    {
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: keyword MEASURE_ERRORS_IN set" << std::endl;
      #endif
      P_D_A1_Sig.reset();
      P_D_A1_Sig = *(PTR(ndarray::Array<ImageT, 1, 1>)*)ArgV_In[I_KeywordSet_MeasureErrors];
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: *P_D_A1_Sig = " << *P_D_A1_Sig << std::endl;
      #endif
      if (P_D_A1_Sig->getShape()[0] != ndata){
        std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: P_D_A1_Sig->size()(=");
        message += std::to_string(P_D_A1_Sig->size()) + ") != ndata(=" + std::to_string(ndata) + ")";
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      D_A1_Sig.deep() = *P_D_A1_Sig;
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: KeyWord_Set(MEASURE_ERRORS_IN): *P_D_A1_Sig = " << *P_D_A1_Sig << std::endl;
      #endif
    }

    ndarray::Array<ImageT, 1, 1> D_A1_YFit = ndarray::allocate(ndata);
    PTR(ndarray::Array<ImageT, 1, 1>) P_D_A1_YFit(new ndarray::Array<ImageT, 1, 1>(D_A1_YFit));
    I_KeywordSet_YFitOut = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "YFIT_OUT");
    if (I_KeywordSet_YFitOut >= 0)
    {
      P_D_A1_YFit.reset();
      P_D_A1_YFit = *(PTR(ndarray::Array<ImageT, 1, 1>)*)ArgV_In[I_KeywordSet_YFitOut];
      if (P_D_A1_YFit->getShape()[0] != ndata){
        std::string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: P_D_A1_YFit->size()(=");
        message += std::to_string(P_D_A1_YFit->size()) + ") != ndata(=" + std::to_string(ndata) + ")";
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    }
    P_D_A1_YFit->deep() = 0.;
    #ifdef __DEBUG_FIT__
      std::cout << "CFits::LinFitBevington: *P_D_A1_YFit set to " << *P_D_A1_YFit << std::endl;
    #endif

    if (P_I_A1_Mask->asEigen().sum() == 0){
      #ifdef __WARNINGS_ON__
        std::cout << "CFits::LinFitBevington: WARNING: P_I_A1_Mask->sum() == 0" << std::endl;
      #endif
      D_SP_Out = 0.;
      D_Sky_Out = 0.;
      status = 0;
      #ifdef __DEBUG_CURVEFIT__
        std::cout << "CurveFitting::LinFitBevingtonNdArray(D_A1_CCD, D_A1_SF, SP, Sky, withSky, Args, ArgV) finished, status = " << status << std::endl;
      #endif
      return status;
    }

    int I_SumMaskLast;
    ImageT D_SDevReject;
    ndarray::Array<ImageT, 1, 1> D_A1_Check = ndarray::allocate(ndata);
    ndarray::Array<lsst::afw::image::MaskPixel, 1, 1> I_A1_LastMask = ndarray::allocate(P_I_A1_Mask->getShape()[0]);
    ndarray::Array<ImageT, 1, 1> D_A1_Diff = ndarray::allocate(ndata);
    D_A1_Diff.deep() = 0.;
    ImageT D_Sum_Weights = 0.;
    ImageT D_Sum_XSquareTimesWeight = 0;
    ImageT D_Sum_XTimesWeight = 0.;
    ImageT D_Sum_YTimesWeight = 0.;
    ImageT D_Sum_XYTimesWeight = 0.;
    ImageT D_Delta = 0.;

    bool B_Run = true;
    int I_Run = -1;
    int I_MaskSum;
    #ifdef __DEBUG_FIT__
      std::cout << "CFits::LinFitBevington: starting while loop" << std::endl;
    #endif
    while (B_Run){
      D_SP_Out = 0.0;

      I_Run++;
      /// remove bad pixels marked by mask
      I_MaskSum = P_I_A1_Mask->asEigen().sum();
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": I_MaskSum = " << I_MaskSum << std::endl;
      #endif
      if (I_MaskSum == 0){
        std::string message("LinFitBevington: WARNING: I_MaskSum == 0");
        std::cout << message << std::endl;
        status = 0;
        return status;
      }
      math::resize(D_A1_Sig, I_MaskSum);
      math::resize(D_A1_CCD, I_MaskSum);
      math::resize(D_A1_SF, I_MaskSum);
      math::resize(D_A1_WT, I_MaskSum);
      math::resize(D_A1_YFit, I_MaskSum);

      I_Pos = 0;
      for (size_t ii = 0; ii < P_I_A1_Mask->size(); ii++){
        if ((*P_I_A1_Mask)[ii] == 1){
          D_A1_CCD[I_Pos] = D_A1_CCD_In[ii];
          D_A1_SF[I_Pos] = D_A1_SF_In[ii];
          D_A1_Sig[I_Pos] = (*P_D_A1_Sig)[ii];
          I_Pos++;
        }
      }
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_CCD set to " << D_A1_CCD << std::endl;
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_SF set to " << D_A1_SF << std::endl;
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_Sig set to " << D_A1_Sig << std::endl;
      #endif

      D_Sum_Weights = 0.;
      D_Sum_XSquareTimesWeight = 0.;
      D_Sum_XTimesWeight = 0.;
      D_Sum_XYTimesWeight = 0.;
      D_Sum_YTimesWeight = 0.;
      if (I_KeywordSet_MeasureErrors >= 0)
      {
        for (i=0; i < I_MaskSum; i++)
        {
          /// ... with weights
          if (std::fabs(D_A1_Sig[i]) < 0.00000000000000001){
            std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": ERROR: D_A1_Sig = " << D_A1_Sig << std::endl;
            std::string message("CFits::LinFitBevington: I_Run=");
            message += std::to_string(I_Run) + ": i = " + std::to_string(i) + ": ERROR: D_A1_Sig(" + std::to_string(i) + ") == 0.";
            std::cout << message << std::endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          D_A1_WT[i] = 1. / pow(D_A1_Sig[i], 2);
        }
        #ifdef __DEBUG_FIT__
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ":: D_A1_WT set to " << D_A1_WT << std::endl;
        #endif
        for (i=0; i < I_MaskSum; i++)
        {
          D_Sum_Weights += D_A1_WT[i];
          D_Sum_XTimesWeight += D_A1_SF[i] * D_A1_WT[i];
          D_Sum_YTimesWeight += D_A1_CCD[i] * D_A1_WT[i];
          D_Sum_XYTimesWeight += D_A1_SF[i] * D_A1_CCD[i] * D_A1_WT[i];
          D_Sum_XSquareTimesWeight += D_A1_SF[i] * D_A1_SF[i] * D_A1_WT[i];
        }
      }
      else
      {
        for (i = 0; i < I_MaskSum; i++)
        {
          /// ... or without weights
          D_Sum_XTimesWeight += D_A1_SF[i];
          D_Sum_YTimesWeight += D_A1_CCD[i];
          D_Sum_XYTimesWeight += D_A1_SF[i] * D_A1_CCD[i];
          D_Sum_XSquareTimesWeight += D_A1_SF[i] * D_A1_SF[i];
        }
        D_Sum_Weights = I_MaskSum;
      }
      D_Delta = D_Sum_Weights * D_Sum_XSquareTimesWeight - pow(D_Sum_XTimesWeight, 2);

      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_Weights set to " << D_Sum_Weights << std::endl;
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XTimesWeight set to " << D_Sum_XTimesWeight << std::endl;
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_YTimesWeight set to " << D_Sum_YTimesWeight << std::endl;
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XYTimesWeight set to " << D_Sum_XYTimesWeight << std::endl;
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XSquareTimesWeight set to " << D_Sum_XSquareTimesWeight << std::endl;
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Delta set to " << D_Delta << std::endl;
      #endif


      if (!B_WithSky)
      {
        #ifdef __DEBUG_FIT__
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out < 0. = setting D_Sky_Out to 0 " << std::endl;
        #endif
        D_SP_Out = D_Sum_XYTimesWeight / D_Sum_XSquareTimesWeight;
        D_Sky_Out = 0.0;
      }
      else
      {
        #ifdef __DEBUG_FIT__
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out >= 0." << D_Sky_Out << std::endl;
        #endif
        D_Sky_Out = ((D_Sum_XSquareTimesWeight * D_Sum_YTimesWeight) - (D_Sum_XTimesWeight * D_Sum_XYTimesWeight)) / D_Delta;

        D_SP_Out = ((D_Sum_Weights * D_Sum_XYTimesWeight) - (D_Sum_XTimesWeight * D_Sum_YTimesWeight)) / D_Delta;
        #ifdef __DEBUG_FIT__
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_SP_Out set to " << D_SP_Out << std::endl;
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out set to " << D_Sky_Out << std::endl;
        #endif
      }
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_Weights >= " << D_Sum_Weights << std::endl;
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XSquareTimesWeight >= " << D_Sum_XSquareTimesWeight << std::endl;
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Delta >= " << D_Delta << std::endl;
      #endif
      (*P_D_A1_Sigma_Out)[0] = sqrt(D_Sum_Weights / D_Delta);
      (*P_D_A1_Sigma_Out)[1] = sqrt(D_Sum_XSquareTimesWeight / D_Delta);
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": P_D_A1_Sigma_Out(0) set to " << (*P_D_A1_Sigma_Out)[0] << std::endl;
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": P_D_A1_Sigma_Out(1) set to " << (*P_D_A1_Sigma_Out)[1] << std::endl;
      #endif
      if ((!B_AllowSpecLTZero) && (D_SP_Out < 0.))
        D_SP_Out = 0.;

      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out set to " << D_Sky_Out << std::endl;
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_SP_Out set to " << D_SP_Out << std::endl;
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": std::fabs(D_SP_Out) = " << std::fabs(D_SP_Out) << std::endl;
      #endif

      P_D_A1_YFit->deep() = D_Sky_Out + D_SP_Out * D_A1_SF_In;//.template cast<ImageT>();
      D_A1_YFit.deep() = D_Sky_Out + D_SP_Out * D_A1_SF;//.template cast<ImageT>();
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": *P_D_A1_YFit set to " << *P_D_A1_YFit << std::endl;
        std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_YFit set to " << D_A1_YFit << std::endl;
      #endif
      *P_D_ChiSqr_Out = 0.;
      if (I_KeywordSet_MeasureErrors < 0)
      {
        for (i = 0; i < I_MaskSum; i++)
        {
          *P_D_ChiSqr_Out += pow(D_A1_CCD[i] - D_A1_YFit[i], 2);
          #ifdef __DEBUG_FIT__
            std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": P_D_ChiSqr_Out set to " << *P_D_ChiSqr_Out << std::endl;
          #endif
        }

        /// for unweighted data evaluate typical sig using chi2, and adjust the standard deviations
        if (I_MaskSum == 2){
          std::string message("CFits::LinFitBevington: I_Run=");
          message += std::to_string(I_Run) + ": ERROR: Sum of Mask (=" + std::to_string(I_MaskSum) + ") must be greater than 2";
          std::cout << message << std::endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        sigdat = sqrt((*P_D_ChiSqr_Out) / (I_MaskSum - 2));
        (*P_D_A1_Sigma_Out)[0] *= sigdat;
        (*P_D_A1_Sigma_Out)[1] *= sigdat;
      }
      else
      {
        for (i = 0; i < I_MaskSum; i++)
        {
          #ifdef __DEBUG_FIT__
            std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_CCD(" << i << ") = " << D_A1_CCD[i] << std::endl;
            std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_SF(" << i << ") = " << D_A1_SF[i] << std::endl;
            std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_Sig(" << i << ") = " << D_A1_Sig[i] << std::endl;
            std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_YFit(" << i << ") = " << D_A1_YFit[i] << std::endl;
          #endif
          if (std::abs(D_A1_Sig[i]) < 0.00000000000000001){
            std::string message("CFits::LinFitBevington: I_Run=");
            message += std::to_string(I_Run) + ": i = " + std::to_string(i) + ": ERROR: D_A1_Sig(" + std::to_string(i) + ") == 0.";
            std::cout << message << std::endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          *P_D_ChiSqr_Out += pow((D_A1_CCD[i] - D_A1_YFit[i]) / D_A1_Sig[i], 2);
          #ifdef __DEBUG_FIT__
            std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": P_D_ChiSqr_Out set to " << *P_D_ChiSqr_Out << std::endl;
          #endif
        }
        #ifdef __DEBUG_FIT__
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": P_D_ChiSqr_Out set to " << *P_D_ChiSqr_Out << std::endl;
        #endif
        if (I_MaskSum > 2)
          *P_D_Q_Out = pfs::drp::stella::math::GammQ(0.5 * (I_MaskSum - 2), 0.5 * (*P_D_ChiSqr_Out));
      }
      if (std::fabs(D_SP_Out) < 0.000001)
        B_Reject = false;
      if (!B_Reject)
        B_Run = false;
      else{

        I_SumMaskLast = P_I_A1_Mask->asEigen().sum();
        #ifdef __DEBUG_FIT__
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: I_SumMaskLast = " << I_SumMaskLast << std::endl;
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_CCD = " << D_A1_CCD << std::endl;
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_YFit = " << D_A1_YFit << std::endl;
        #endif
        ndarray::Array<ImageT, 1, 1> tempArr = ndarray::allocate(D_A1_CCD.getShape()[0]);
        tempArr.deep() = D_A1_CCD - D_A1_YFit;
        Eigen::Array<ImageT, Eigen::Dynamic, 1> tempEArr = tempArr.asEigen();
        tempArr.asEigen() = tempEArr.pow(2);
        D_SDevReject = sqrt(tempArr.asEigen().sum() / ImageT(I_SumMaskLast));

        D_A1_Diff.deep() = D_A1_CCD_In - (*P_D_A1_YFit);
        #ifdef __DEBUG_FIT__
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_SDevReject = " << D_SDevReject << std::endl;
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_CCD_In = " << D_A1_CCD_In << std::endl;
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: *P_D_A1_YFit = " << *P_D_A1_YFit << std::endl;
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_CCD_In - (*P_D_A1_YFit) = " << D_A1_Diff << std::endl;
        #endif
        tempEArr = D_A1_Diff.asEigen();
        D_A1_Check.asEigen() = tempEArr.abs();
        #ifdef __DEBUG_FIT__
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_Check = " << D_A1_Check << std::endl;
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": before Reject: *P_I_A1_Mask = " << *P_I_A1_Mask << std::endl;
        #endif
        I_A1_LastMask = *P_I_A1_Mask;
        for (size_t pos = 0; pos < D_A1_Check.getShape()[0]; ++pos){
          (*P_I_A1_Mask)[pos] = (D_A1_Check[pos] > (D_Reject * D_SDevReject)) ? 0 : 1;
          if (I_A1_Mask_Orig[pos] < 1)
            (*P_I_A1_Mask)[pos] = 0;
        }
        if (P_I_A1_Mask->asEigen().sum() == I_A1_Mask_Orig.asEigen().sum())
          B_Reject = false;
        else{
          for (size_t pos = 0; pos < P_I_A1_Mask->getShape()[0]; ++pos)
            if (I_A1_LastMask[pos] < 1)
              (*P_I_A1_Mask)[pos] = 0;
        }
        #ifdef __DEBUG_FIT__
          std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: *P_I_A1_Mask = " << *P_I_A1_Mask << std::endl;
        #endif
        if (I_SumMaskLast == P_I_A1_Mask->asEigen().sum()){
          B_Run = false;
          #ifdef __DEBUG_FIT__
            std::cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": leaving while loop" << std::endl;
          #endif
        }
        else{
          D_Sky_Out = 0.;
        }
      }
      if ((!B_AllowSkyLTZero) && (D_Sky_Out < 0.)){
        B_Run = true;
        B_WithSky = false;
      }
    }/// end while (B_Run)

    #ifdef __DEBUG_FIT__
      std::cout << "CFits::LinFitBevington: *P_D_A1_YFit set to " << *P_D_A1_YFit << std::endl;
      std::cout << "CFits::LinFitBevington: *P_I_A1_Mask set to " << *P_I_A1_Mask << std::endl;
    #endif
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::LinFitBevingtonNdArray(D_A1_CCD, D_A1_SF, SP, Sky, withSky, Args, ArgV) finished" << std::endl;
    #endif
    return status;
  }

  /**
   * Helper function to calculate incomplete Gamma Function
   **/
  template< typename T>
  T GSER(T & D_Gamser_Out, T const a, T const x)
  {
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::GSER(Gamser, a, x) started" << std::endl;
    #endif
    T D_GLn_Out = 0;
    int n;
    int ITMax = 100;
    float d_sum, del, ap;

    #ifdef __DEBUG_LINFIT__
      std::cout << "CFits::GSER: D_Gamser_Out = " << D_Gamser_Out << std::endl;
      std::cout << "CFits::GSER: a = " << a << std::endl;
      std::cout << "CFits::GSER: x = " << x << std::endl;
    #endif

    D_GLn_Out = GammLn(a);
    #ifdef __DEBUG_LINFIT__
      std::cout << "CFits::GSER: D_GLn_Out = " << D_GLn_Out << std::endl;
    #endif
    if (x <= 0.){
      if (x < 0.){
        std::string message("CFits::GSER: ERROR: x less than 0!");
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      D_Gamser_Out = 0.;
      #ifdef __DEBUG_LINFIT__
        std::cout << "CFits::GSER: x<=0: D_Gamser_Out = " << D_Gamser_Out << std::endl;
        std::cout << "CFits::GSER: x<=0: D_GLn_Out = " << D_GLn_Out << std::endl;
      #endif
      #ifdef __DEBUG_CURVEFIT__
        std::cout << "CurveFitting::GSER(Gamser, a, x) finished" << std::endl;
      #endif
      return D_GLn_Out;
    }
    else{
      ap = a;
      del = d_sum = 1. / a;
      for (n=1; n <= ITMax; n++){
        ++ap;
        del *= x/ap;
        d_sum += del;
        if (std::fabs(del) < std::fabs(d_sum) * 3.e-7){
          D_Gamser_Out = d_sum * exp(-x+a*log(x) - D_GLn_Out);
          #ifdef __DEBUG_LINFIT__
            std::cout << "CFits::GSER: x>0: D_Gamser_Out = " << D_Gamser_Out << std::endl;
            std::cout << "CFits::GSER: x>0: D_GLn_Out = " << D_GLn_Out << std::endl;
          #endif
          #ifdef __DEBUG_CURVEFIT__
            std::cout << "CurveFitting::GSER(Gamser, a, x) finished" << std::endl;
          #endif
          return D_GLn_Out;
        }
      }
      std::string message("CFits::GSER: ERROR: a too large, ITMax too small in routine GSER");
      std::cout << message << std::endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
  }

  /**
   * Helper function to calculate incomplete Gamma Function
   **/
  template< typename T >
  T GammLn(T const xx)
  {
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::GammLn(xx) started" << std::endl;
    #endif
    float x,y,tmp,ser;
    static float cof[6]={76.18009172947146, -86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5};

    #ifdef __DEBUG_LINFIT__
      std::cout << "CFits::GammLn: xx = " << xx << std::endl;
    #endif

    y = x = xx;
    tmp = x + 5.5;
    tmp -= (x+0.5) * log(tmp);
    #ifdef __DEBUG_LINFIT__
      std::cout << "CFits::GammLn: tmp = " << tmp << std::endl;
    #endif
    ser = 1.000000000190015;
    for (int o = 0; o <= 5; o++){
      ser += cof[o] / ++y;
    }
    T D_Result = T(-tmp + log(2.5066282746310005 * ser / x));
    #ifdef __DEBUG_LINFIT__
      std::cout << "CFits::GammLn: ser = " << ser << std::endl;
      std::cout << "CFits::GammLn: returning (-tmp + log(2.5066282746310005 * ser / xx)) = " << D_Result << std::endl;
    #endif
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::GammLn(xx) finished" << std::endl;
    #endif
    return D_Result;
  }

  /**
   * Helper function to calculate incomplete Gamma Function
   **/
  template<typename T>
  T GCF(T & D_GammCF_Out, T const a, T const x)
  {
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::GCF(GammCF, a, x) Started" << std::endl;
    #endif
    T D_GLn_Out = 0;
    int n;
    int ITMAX = 100;             /// Maximum allowed number of iterations
    T an, b, c, d, del, h;
    float FPMIN = 1.0e-30;      /// Number near the smallest representable floating-point number
    float EPS = 1.0e-7;         /// Relative accuracy

    D_GLn_Out = GammLn(a);
    #ifdef __DEBUG_FIT__
      std::cout << "CFits::GCF: D_GLn_Out set to " << D_GLn_Out << std::endl;
    #endif

    b = x + 1. - a;
    #ifdef __DEBUG_FIT__
      std::cout << "CFits::GCF: x=" << x << ", a=" << a << ": b set to " << b << std::endl;
    #endif
    c = 1. / FPMIN;
    #ifdef __DEBUG_FIT__
      std::cout << "CFits::GCF: c set to " << c << std::endl;
    #endif
    d = 1. / b;
    #ifdef __DEBUG_FIT__
      std::cout << "CFits::GCF: d set to " << d << std::endl;
    #endif
    h = d;
    for (n=1; n <= ITMAX; n++){
      an = -n * (n - a);
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::GCF: n = " << n << ": an set to " << an << std::endl;
      #endif
      b += 2.;
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::GCF: n = " << n << ": b set to " << b << std::endl;
      #endif
      d = an * d + b;
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::GCF: n = " << n << ": d set to " << d << std::endl;
      #endif
      if (std::fabs(d) < FPMIN)
        d = FPMIN;
      c = b + an / c;
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::GCF: n = " << n << ": c set to " << c << std::endl;
      #endif
      if (std::fabs(c) < FPMIN)
        c = FPMIN;
      d = 1. / d;
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::GCF: n = " << n << ": d set to " << d << std::endl;
      #endif
      del = d * c;
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::GCF: n = " << n << ": del set to " << del << std::endl;
      #endif

      h *= del;
      if (std::fabs(del-1.) < EPS)
        break;
    }
    if (n > ITMAX){
      std::string message("CFits::GCF: ERROR: a too large, ITMAX too small in GCF");
      std::cout << message << std::endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    D_GammCF_Out = exp(-x+a*log(x) - D_GLn_Out) * h;
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::GCF(GammCF, a, x) finished" << std::endl;
    #endif
    return D_GLn_Out;
  }

  /**
   * Function to calculate incomplete Gamma Function P(a,x)
   **/
  template<typename T>
  T GammP(T const a, T const x){
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::GammP(a, x) Started" << std::endl;
    #endif
    T D_Out = 0;
    #ifdef __DEBUG_FIT__
      std::cout << "CFits::GammP started: a = " << a << ", x = " << x << std::endl;
    #endif
    T gamser, gammcf;
    if (x < 0.){
      std::string message("pfs::drp::stella::math::CurveFitting::GammP: ERROR: x(=");
      message += std::to_string(x) + ") < 0.";
      std::cout << message << std::endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (a <= 0.){
      std::string message("pfs::drp::stella::math::CurveFitting::GammP: ERROR: a(=");
      message += std::to_string(a) + ") <= 0.";
      std::cout << message << std::endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (x < (a+1.)){
      GSER(gamser, a, x);
      D_Out = gamser;
      #ifdef __DEBUG_CURVEFIT__
        std::cout << "CurveFitting::GammP(a, x) finished" << std::endl;
      #endif
      return D_Out;
    }
    else{
      GCF(gammcf, a, x);
      D_Out = 1. - gammcf;
      #ifdef __DEBUG_CURVEFIT__
        std::cout << "CurveFitting::GammP(a, x) finished" << std::endl;
      #endif
      return D_Out;
    }
  }

  /**
   * Function to calculate incomplete Gamma Function Q(a,x) = 1. - P(a,x)
   **/
  template<typename T>
  T GammQ(T const a, T const x){
    #ifdef __DEBUG_CURVEFIT__
      std::cout << "CurveFitting::GammQ(a, x) started" << std::endl;
    #endif
    T D_Out = 0;
    #ifdef __DEBUG_FIT__
      std::cout << "CFits::GammQ started: a = " << a << ", x = " << x << std::endl;
    #endif
    T gamser = 0.;
    T gammcf = 0.;
    if (x < 0.){
      std::string message("pfs::drp::stella::math::CurveFitting::GammQ: ERROR: x(=");
      message += std::to_string(x) + ") < 0.";
      std::cout << message << std::endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if(a <= 0.){
      std::string message("pfs::drp::stella::math::CurveFitting::GammQ: ERROR: a(=");
      message += std::to_string(a) + ") < 0.";
      std::cout << message << std::endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (x < (a+1.)){
      GSER(gamser, a, x);
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::GammQ: x < (a+1.): gamser = " << gamser << std::endl;
      #endif
      D_Out = 1. - gamser;
      #ifdef __DEBUG_CURVEFIT__
        std::cout << "CurveFitting::GammQ(a, x) finished" << std::endl;
      #endif
      return D_Out;
    }
    else{
      GCF(gammcf, a, x);
      #ifdef __DEBUG_FIT__
        std::cout << "CFits::GammQ: x < (a+1.): gammcf = " << gammcf << std::endl;
      #endif
      D_Out = gammcf;
      #ifdef __DEBUG_CURVEFIT__
        std::cout << "CurveFitting::GammQ(a, x) finished" << std::endl;
      #endif
      return D_Out;
    }
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

  template bool LinFitBevingtonNdArray(ndarray::Array<float, 2, 1> const&,
                                             ndarray::Array<float, 2, 1> const&,
                                             ndarray::Array<float, 1, 1> &,
                                             ndarray::Array<float, 1, 1> &,
                                             bool,
                                             const std::vector<std::string> &,
                                             std::vector<void *> &);
}}}}
