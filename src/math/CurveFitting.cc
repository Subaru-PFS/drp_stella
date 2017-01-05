#include "pfs/drp/stella/math/CurveFitting.h"
namespace pfs{ namespace drp{ namespace stella{ namespace math{

  template<typename T, typename U>
  ndarray::Array<T, 1, 1> Poly(ndarray::Array<T, 1, 1> const& x_In,
                               ndarray::Array<U, 1, 1> const& coeffs_In,
                               T xRangeMin_In,
                               T xRangeMax_In){
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::Poly(x, coeffs, xRangeMin, xRangeMax) started" << endl;
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
      cout << "pfs::drp::stella::math::CurveFitting::Poly: x_In = " << x_In << endl;
      cout << "pfs::drp::stella::math::CurveFitting::Poly: xNew = " << xNew << endl;
    #endif

    int ii = 0;
    ndarray::Array<T, 1, 1> arr_Out = ndarray::allocate(int(x_In.size()));
    #ifdef __DEBUG_POLY__
      cout << "Poly: coeffs_In = " << coeffs_In << endl;
    #endif
    int I_PolynomialOrder = coeffs_In.size() - 1;
    #ifdef __DEBUG_POLY__
      cout << "Poly: I_PolynomialOrder set to " << I_PolynomialOrder << endl;
    #endif
    if (I_PolynomialOrder == 0){
      arr_Out[ndarray::view()] = coeffs_In(0);
      #ifdef __DEBUG_POLY__
        cout << "Poly: I_PolynomialOrder == 0: arr_Out set to " << arr_Out << endl;
      #endif
      return arr_Out;
    }
    arr_Out[ndarray::view()] = coeffs_In(I_PolynomialOrder);
    #ifdef __DEBUG_POLY__
      cout << "Poly: I_PolynomialOrder != 0: arr_Out set to " << arr_Out << endl;
    #endif

    auto arr_Out_begin = arr_Out.begin();
    auto xNew_begin = xNew.begin();
    auto coeffs_In_begin = coeffs_In.begin();
    for (ii = I_PolynomialOrder-1; ii >= 0; ii--){
      for (int i = 0; i < arr_Out.getShape()[0]; ++i)
        *(arr_Out_begin + i) = (*(arr_Out_begin + i)) * (*(xNew_begin + i)) + (*(coeffs_In_begin + ii));
      #ifdef __DEBUG_POLY__
        cout << "Poly: I_PolynomialOrder != 0: for (ii = " << ii << "; ii >= 0; ii--) arr_Out set to " << arr_Out << endl;
      #endif
    }
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::Poly(x, coeffs, xRangeMin, xRangeMax) finished" << endl;
    #endif
    return arr_Out;
  }

/*  template<typename T, typename U>
  ndarray::Array<T, 1, 1> Poly(ndarray::Array<T, 1, 1> const& x_In,
                               ndarray::Array<U, 1, 1> const& coeffs_In,
                               ndarray::Array<double, 1, 1> const& xRange_In){
    ndarray::Array<T, 1, 1> xNew = ndarray::allocate(x_In.getShape()[0]);
    T xMin = pfs::drp::stella::math::min(x_In);
    T xMax = pfs::drp::stella::math::max(x_In);
    for (int i = 0; i < x_In.getShape()[0]; ++i){
      xNew[i] = ((xRange_In[1] - xRange_In[0]) * (x_In[i] - xMin) / (xMax - xMin)) + xRange_In[0];
    }
    #ifdef __DEBUG_POLY__
      cout << "pfs::drp::stella::math::CurveFitting::Poly: x_In = " << x_In << endl;
      cout << "pfs::drp::stella::math::CurveFitting::Poly: xNew = " << xNew << endl;
    #endif
    return Poly(xNew, coeffs_In);
  }*/

  template<typename T>
  ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                       ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                       size_t const I_Degree_In,
                                       T const D_Reject_In,
                                       std::vector<string> const& S_A1_Args_In,
                                       std::vector<void *> & ArgV){
    return pfs::drp::stella::math::PolyFit(D_A1_X_In,
                                           D_A1_Y_In,
                                           I_Degree_In,
                                           D_Reject_In,
                                           D_Reject_In,
                                           -1,
                                           S_A1_Args_In,
                                           ArgV);
  }

  template< typename T >
  ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                       ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                       size_t const I_Degree_In,
                                       T const D_LReject_In,
                                       T const D_UReject_In,
                                       size_t const I_NIter,
                                       std::vector<string> const& S_A1_Args_In,
                                       std::vector<void *> &ArgV){
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::PolyFit(x, y, deg, lReject, uReject, nIter, Args, ArgV) started" << endl;
    #endif
    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: Starting " << endl;
    #endif
    if (D_A1_X_In.getShape()[0] != D_A1_Y_In.getShape()[0]){
      string message("pfs::drp::stella::math::CurfFitting::PolyFit: ERROR: D_A1_X_In.getShape()[0](=");
      message += to_string(D_A1_X_In.getShape()[0]) + " != D_A1_Y_In.getShape()[0](=" + to_string(D_A1_Y_In.getShape()[0]) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    int I_NReject = 0;
    int I_DataValues_New = 0;
    int I_NRejected = 0;
    bool B_HaveMeasureErrors = false;
    ndarray::Array<double, 1, 1> D_A1_Coeffs_Out = ndarray::allocate(I_Degree_In + 1);
    ndarray::Array<T, 1, 1> measureErrors = ndarray::allocate(D_A1_X_In.getShape()[0]);
    PTR(ndarray::Array<T, 1, 1>) P_D_A1_MeasureErrors(new ndarray::Array<T, 1, 1>(measureErrors));

    int I_Pos = -1;
    I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MEASURE_ERRORS");
    if (I_Pos >= 0){
      P_D_A1_MeasureErrors.reset();
      P_D_A1_MeasureErrors = (*((PTR(ndarray::Array<T, 1, 1>)*)ArgV[I_Pos]));
      measureErrors.deep() = *P_D_A1_MeasureErrors;
      B_HaveMeasureErrors = true;
      if (P_D_A1_MeasureErrors->getShape()[0] != D_A1_X_In.getShape()[0]){
        string message("pfs::drp::stella::math::CurfFitting::PolyFit: Error: P_D_A1_MeasureErrors->getShape()[0](=");
        message += to_string(P_D_A1_MeasureErrors->getShape()[0]) + ") != D_A1_X_In.getShape()[0](=" + to_string(D_A1_X_In.getShape()[0]) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    }
    else{
      for (auto it = measureErrors.begin(); it != measureErrors.end(); ++it)
        *it = (*it > 0) ? sqrt(*it) : 1;
    }
    ndarray::Array<T, 1, 1> D_A1_MeasureErrorsBak = copy(measureErrors);

    PTR(std::vector<size_t>) P_I_A1_NotRejected(new std::vector<size_t>());
    I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "NOT_REJECTED");
    if (I_Pos >= 0){
      P_I_A1_NotRejected.reset();
      P_I_A1_NotRejected = *((PTR(std::vector<size_t>)*)(ArgV[I_Pos]));
      #ifdef __DEBUG_POLYFIT__
        cout << "PolyFit: P_I_A1_NotRejected = ";
        for (auto it = P_I_A1_NotRejected->begin(); it != P_I_A1_NotRejected->end(); ++it)
          cout << *it << " ";
        cout << endl;
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: KeyWord NOT_REJECTED read" << endl;
      #endif
    }

    PTR(std::vector<size_t>) P_I_A1_Rejected(new std::vector<size_t>());
    I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "REJECTED");
    if (I_Pos >= 0){
      P_I_A1_Rejected.reset();
      P_I_A1_Rejected = *((PTR(std::vector<size_t>)*)(ArgV[I_Pos]));
      #ifdef __DEBUG_POLYFIT__
        cout << "PolyFit: P_I_A1_NotRejected = ";
        for (auto it = P_I_A1_Rejected->begin(); it != P_I_A1_Rejected->end(); ++it)
          cout << *it << " ";
        cout << endl;
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: KeyWord REJECTED read" << endl;
      #endif
    }

    PTR(int) P_I_NRejected(new int(I_NRejected));
    I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "N_REJECTED");
    if (I_Pos >= 0){
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: Reading KeyWord N_REJECTED" << endl;
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: I_Pos = " << I_Pos << endl;
      #endif
      P_I_NRejected.reset();
      P_I_NRejected = *((PTR(int)*)(ArgV[I_Pos]));
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: P_I_NRejected = " << *P_I_NRejected << endl;
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: KeyWord N_REJECTED read" << endl;
      #endif
    }

    ndarray::Array<double, 1, 1> xRange = ndarray::allocate(2);
    xRange[0] = pfs::drp::stella::math::min(D_A1_X_In);
    xRange[1] = pfs::drp::stella::math::max(D_A1_X_In);;
    PTR(ndarray::Array<double, 1, 1>) P_D_A1_XRange(new ndarray::Array<double, 1, 1>(xRange));
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "XRANGE")) >= 0)
    {
      P_D_A1_XRange.reset();
      P_D_A1_XRange = *((PTR(ndarray::Array<double, 1, 1>)*)ArgV[I_Pos]);
      if (P_D_A1_XRange->getShape()[0] != 2){
        string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: P_D_A1_XRange->getShape()[0](=");
        message += to_string(P_D_A1_XRange->getShape()[0]) + " != 2";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      xRange.deep() = *P_D_A1_XRange;
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: *P_D_A1_XRange set to " << *P_D_A1_XRange << endl;
      #endif
    }
    ndarray::Array<T, 1, 1> xNew;
    /// shift and rescale x_In to fit into range [-1.,1.]
    if ((std::fabs(xRange[0] + 1.) > 0.00000001) || (std::fabs(xRange[1] - 1.) > 0.00000001)){
      xNew = pfs::drp::stella::math::convertRangeToUnity(D_A1_X_In, xRange);
    }
    else{
      xNew = D_A1_X_In;
    }
    #ifdef __DEBUG_POLY__
      cout << "pfs::drp::stella::math::CurveFitting::Poly: D_A1_X_In = " << D_A1_X_In << endl;
      cout << "pfs::drp::stella::math::CurveFitting::Poly: xNew = " << xNew << endl;
    #endif

    std::vector<T> D_A1_X(xNew.begin(), xNew.end());
    std::vector<T> D_A1_Y(D_A1_Y_In.begin(), D_A1_Y_In.end());
    std::vector<T> D_A1_X_New(D_A1_X.size());
    std::vector<T> D_A1_Y_New(D_A1_Y.size());
    std::vector<T> D_A1_MeasureErrors(D_A1_X.size());
    std::vector<T> D_A1_MeasureErrors_New(D_A1_X.size());
    std::vector<size_t> I_A1_OrigPos(D_A1_X_In.getShape()[0]);
    for (size_t i = 0; i < I_A1_OrigPos.size(); ++i)
      I_A1_OrigPos[i] = i;
    ndarray::Array<T, 1, 1> D_A1_PolyRes = ndarray::allocate(D_A1_X_In.getShape()[0]);
    int I_NRejected_Old = 0;
    std::vector<size_t> I_A1_Rejected_Old(D_A1_X_In.size());
    bool B_Run = true;
    unsigned int i_iter = 0;
    while (B_Run){
      I_A1_Rejected_Old = *P_I_A1_Rejected;
      I_NRejected_Old = *P_I_NRejected;
      I_NReject = 0;
      *P_I_NRejected = 0;
      I_DataValues_New = 0;
      ndarray::Array<T, 1, 1> D_A1_XArr = ndarray::external(D_A1_X.data(), ndarray::makeVector(int(D_A1_X.size())), ndarray::makeVector(1));
      ndarray::Array<T, 1, 1> D_A1_YArr = ndarray::external(D_A1_Y.data(), ndarray::makeVector(int(D_A1_X.size())), ndarray::makeVector(1));
      D_A1_Coeffs_Out = pfs::drp::stella::math::PolyFit(D_A1_XArr,
                                                        D_A1_YArr,
                                                        I_Degree_In,
                                                        S_A1_Args_In,
                                                        ArgV);
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: PolyFit(D_A1_XArr, D_A1_YArr, I_Degree_In, S_A1_Args_In, ArgV) returned D_A1_Coeffs_Out = " << D_A1_Coeffs_Out << endl;
      #endif
      ndarray::Array<T, 1, 1> D_A1_YFit;
      D_A1_YFit = pfs::drp::stella::math::Poly(D_A1_XArr,
                                               D_A1_Coeffs_Out,
                                               T(xRange[0]),
                                               T(xRange[1]));

      ndarray::Array<T, 1, 1> D_A1_Temp = ndarray::allocate(D_A1_Y.size());
      for (int pos = 0; pos < D_A1_Y.size(); ++pos)
        D_A1_Temp[pos] = D_A1_Y[pos] - D_A1_YFit[pos];
      Eigen::Array<T, Eigen::Dynamic, 1> tempEArr = D_A1_Temp.asEigen();
      D_A1_Temp.asEigen() = tempEArr.pow(2) / T(D_A1_Y.size());
//        D_A1_Temp.deep() = D_A1_Temp / D_A1_Y.size();
      double D_SDev = double(sqrt(D_A1_Temp.asEigen().sum()));

      D_A1_PolyRes.deep() = pfs::drp::stella::math::Poly(xNew,
                                                         D_A1_Coeffs_Out,
                                                         T(xRange[0]),
                                                         T(xRange[1]));
      double D_Dev;
      D_A1_X.resize(0);
      D_A1_Y.resize(0);
      D_A1_MeasureErrors.resize(0);
      I_A1_OrigPos.resize(0);
      P_I_A1_Rejected->resize(0);
      for (size_t i_pos=0; i_pos < D_A1_Y_In.getShape()[0]; i_pos++){
        D_Dev = D_A1_Y_In[i_pos] - D_A1_PolyRes[i_pos];
        if (((D_Dev < 0) && (D_Dev > (D_LReject_In * D_SDev))) ||
            ((D_Dev >= 0) && (D_Dev < (D_UReject_In * D_SDev)))){
          D_A1_X.push_back(xNew[i_pos]);
          D_A1_Y.push_back(D_A1_Y_In[i_pos]);
          if (B_HaveMeasureErrors)
            D_A1_MeasureErrors.push_back(D_A1_MeasureErrorsBak[i_pos]);
          I_A1_OrigPos.push_back(D_A1_Y_In[i_pos]);

          I_DataValues_New++;
        }
        else{
          P_I_A1_Rejected->push_back(i_pos);
          #ifdef __DEBUG_POLYFIT__
            cout << "pfs::drp::stella::math::CurveFitting::PolyFit: Rejecting D_A1_X_In(" << i_pos << ") = " << D_A1_X_In[i_pos] << endl;
          #endif
          I_NReject++;
          ++(*P_I_NRejected);
        }
      }

      B_Run = false;
      if (*P_I_NRejected != I_NRejected_Old)
        B_Run = true;
      else{
        for (int i_pos=0; i_pos < *P_I_NRejected; i_pos++){
          if (std::fabs((*P_I_A1_Rejected)[i_pos] - I_A1_Rejected_Old[i_pos]) > 0.0001)
            B_Run = true;
        }
      }
      i_iter++;
      if ( i_iter >= I_NIter )
        B_Run = false;
    }
    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: *P_I_NRejected = " << *P_I_NRejected << endl;
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: I_DataValues_New = " << I_DataValues_New << endl;
    #endif
    *P_I_A1_NotRejected = I_A1_OrigPos;
    I_A1_OrigPos.resize(D_A1_X_In.getShape()[0]);
    for (size_t i_pos = 0; i_pos < I_A1_OrigPos.size(); ++i_pos)
      I_A1_OrigPos[i_pos] = i_pos;
    std::vector<size_t> V_OrigPos(I_A1_OrigPos.begin(), I_A1_OrigPos.end());
    std::vector<size_t> V_NotRejected(P_I_A1_NotRejected->begin(), P_I_A1_NotRejected->end());
    *P_I_A1_Rejected = pfs::drp::stella::math::removeSubArrayFromArray(V_OrigPos, V_NotRejected);

    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::PolyFit(x, y, deg, lReject, uReject, nIter, Args, ArgV) finished" << endl;
    #endif
    return D_A1_Coeffs_Out;
  }

  /** **********************************************************************/

  template< typename T >
  ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                       ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                       size_t const I_Degree_In,
                                       T xRangeMin_In,
                                       T xRangeMax_In){
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::PolyFit(x, y, deg, xRangeMin, xRangeMax) started" << endl;
    #endif
    if (D_A1_X_In.getShape()[0] != D_A1_Y_In.getShape()[0]){
      string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: D_A1_X_In.getShape()[0](=");
      message += to_string(D_A1_X_In.getShape()[0]) +") != D_A1_Y_In.getShape()[0](=" + to_string(D_A1_Y_In.getShape()[0]) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    std::vector<string> S_A1_Args(1);
    S_A1_Args[0] = "XRANGE";
    std::vector<void *> PP_Args(1);
    ndarray::Array<double, 1, 1> xRange = ndarray::allocate(2);
    xRange[0] = xRangeMin_In;
    xRange[1] = xRangeMax_In;
    PTR(ndarray::Array<double, 1, 1>) pXRange(new ndarray::Array<double, 1, 1>(xRange));
    PP_Args[0] = &pXRange;
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::PolyFit(x, y, deg, xRangeMin, xRangeMax) finishing" << endl;
    #endif
    return PolyFit(D_A1_X_In,
                   D_A1_Y_In,
                   I_Degree_In,
                   S_A1_Args,
                   PP_Args);
  }

  template< typename T >
  ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                       ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                       size_t const I_Degree_In,
                                       std::vector<string> const& S_A1_Args_In,
                                       std::vector<void *> & ArgV){
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::PolyFit(x, y, deg, Args, ArgV) started" << endl;
    #endif
    if (D_A1_X_In.getShape()[0] != D_A1_Y_In.getShape()[0]){
      string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: D_A1_X_In.getShape()[0](=");
      message += to_string(D_A1_X_In.getShape()[0]) + ") != D_A1_Y_In.getShape()[0](=" + to_string(D_A1_Y_In.getShape()[0]) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: Starting " << endl;
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: D_A1_Y_In = " << D_A1_Y_In << endl;
    #endif
    size_t const nCoeffs(I_Degree_In + 1);
    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: nCoeffs set to " << nCoeffs << endl;
    #endif
    ndarray::Array<double, 1, 1> D_A1_Out = ndarray::allocate(nCoeffs);
    D_A1_Out.deep() = 0.;
    int i, j, I_Pos;

    const int nDataPoints(D_A1_X_In.getShape()[0]);
    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: nDataPoints set to " << nDataPoints << endl;
    #endif

    ndarray::Array<T, 1, 1> D_A1_SDevSquare = ndarray::allocate(nDataPoints);

    bool B_HaveMeasureError = false;
    ndarray::Array<T, 1, 1> D_A1_MeasureErrors = ndarray::allocate(nDataPoints);
    string sTemp = "MEASURE_ERRORS";
    PTR(ndarray::Array<T, 1, 1>) P_D_A1_MeasureErrors(new ndarray::Array<T, 1, 1>(D_A1_MeasureErrors));
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
    {
      B_HaveMeasureError = true;
      P_D_A1_MeasureErrors.reset();
      P_D_A1_MeasureErrors = *((PTR(ndarray::Array<T, 1, 1>)*)ArgV[I_Pos]);
      if (P_D_A1_MeasureErrors->getShape()[0] != nDataPoints){
        string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: P_D_A1_MeasureErrors->getShape()[0](=");
        message += to_string(P_D_A1_MeasureErrors->getShape()[0]) + ") != nDataPoints(=" + to_string(nDataPoints) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      D_A1_MeasureErrors.deep() = *P_D_A1_MeasureErrors;
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: B_HaveMeasureError set to TRUE" << endl;
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: *P_D_A1_MeasureErrors set to " << *P_D_A1_MeasureErrors << endl;
      #endif
    }
    else{
      D_A1_MeasureErrors.deep() = 1.;
    }

    ndarray::Array<double, 1, 1> D_A1_XRange = ndarray::allocate(2);
    ndarray::Array<T, 1, 1> xNew;
    PTR(ndarray::Array<double, 1, 1>) P_D_A1_XRange(new ndarray::Array<double, 1, 1>(D_A1_XRange));
    sTemp = "XRANGE";
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
    {
      P_D_A1_XRange.reset();
      P_D_A1_XRange = *((PTR(ndarray::Array<double, 1, 1>)*)ArgV[I_Pos]);
      if (P_D_A1_XRange->getShape()[0] != 2){
        string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: P_D_A1_XRange->getShape()[0](=");
        message += to_string(P_D_A1_XRange->getShape()[0]) +") != 2";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }

      D_A1_XRange.deep() = *P_D_A1_XRange;
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: *P_D_A1_XRange set to " << *P_D_A1_XRange << endl;
      #endif
    }
    else{
      D_A1_XRange[0] = pfs::drp::stella::math::min(D_A1_X_In);
      D_A1_XRange[1] = pfs::drp::stella::math::max(D_A1_X_In);
    }
    xNew = pfs::drp::stella::math::convertRangeToUnity(D_A1_X_In,
                                                       D_A1_XRange);

    D_A1_SDevSquare.deep() = D_A1_MeasureErrors * D_A1_MeasureErrors;
    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: D_A1_SDevSquare set to " << D_A1_SDevSquare << endl;
    #endif
    ndarray::Array<T, 1, 1> D_A1_YFit = ndarray::allocate(nDataPoints);
    PTR(ndarray::Array<T, 1, 1>) P_D_A1_YFit(new ndarray::Array<T, 1, 1>(D_A1_YFit));
    sTemp = "YFIT";
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
    {
      P_D_A1_YFit.reset();
      P_D_A1_YFit = *((PTR(ndarray::Array<T, 1, 1>)*)ArgV[I_Pos]);
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: KeyWord_Set(YFIT)" << endl;
      #endif
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
        string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: P_D_A1_Sigma->getShape()[0](=");
        message += to_string(P_D_A1_Sigma->getShape()[0]) +") != nCoeffs(=" + to_string(nCoeffs) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: KeyWord_Set(SIGMA): *P_D_A1_Sigma set to " << (*P_D_A1_Sigma) << endl;
      #endif
    }

    ndarray::Array<T, 2, 2> D_A2_Covar = ndarray::allocate(nCoeffs, nCoeffs);
    PTR(ndarray::Array<T, 2, 2>) P_D_A2_Covar(new ndarray::Array<T, 2, 2>(D_A2_Covar));
    sTemp = "COVAR";
    if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
    {
      P_D_A2_Covar.reset();
      P_D_A2_Covar = *((PTR(ndarray::Array<T, 2, 2>)*)ArgV[I_Pos]);
      if (P_D_A2_Covar->getShape()[0] != nCoeffs){
        string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: P_D_A2_Covar->getShape()[0](=");
        message += to_string(P_D_A2_Covar->getShape()[0]) + ") != nCoeffs(=" + to_string(nCoeffs) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (P_D_A2_Covar->getShape()[1] != nCoeffs){
        string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR: P_D_A2_Covar->getShape()[1](=");
        message += to_string(P_D_A2_Covar->getShape()[1]) + ") != nCoeffs(=" + to_string(nCoeffs) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: KeyWord_Set(COVAR): *P_D_A2_Covar set to " << (*P_D_A2_Covar) << endl;
      #endif
    }

    ndarray::Array<T, 1, 1> D_A1_B = ndarray::allocate(nCoeffs);
    ndarray::Array<T, 1, 1> D_A1_Z = ndarray::allocate(nDataPoints);
    D_A1_Z.deep() = 1;
//    #ifdef __DEBUG_POLYFIT__
//      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: D_A1_Z set to " << D_A1_Z << endl;
//    #endif

    ndarray::Array<T, 1, 1> D_A1_WY = ndarray::allocate(nDataPoints);
    D_A1_WY.deep() = D_A1_Y_In;
//    #ifdef __DEBUG_POLYFIT__
//      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: D_A1_WY set to " << D_A1_WY << endl;
//    #endif

    if (B_HaveMeasureError){
      D_A1_WY.deep() = D_A1_WY / D_A1_SDevSquare;
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: B_HaveMeasureError: D_A1_WY set to " << D_A1_WY << endl;
      #endif
    }

    if (B_HaveMeasureError){
      (*P_D_A2_Covar)[ ndarray::makeVector( 0, 0 ) ] = sum(1./D_A1_SDevSquare);
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: B_HaveMeasureError: (*P_D_A2_Covar)(0,0) set to " << (*P_D_A2_Covar)[ ndarray::makeVector( 0, 0 ) ] << endl;
      #endif
    }
    else{
      (*P_D_A2_Covar)[ ndarray::makeVector( 0, 0 ) ] = nDataPoints;
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: !B_HaveMeasureError: (*P_D_A2_Covar)(0,0) set to " << (*P_D_A2_Covar)[ ndarray::makeVector(0, 0 ) ] << endl;
      #endif
    }

    D_A1_B[0] = sum(D_A1_WY);
    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: D_A1_B(0) set to " << D_A1_B[0] << endl;
    #endif

    T D_Sum;
    for (int p = 1; p <= 2 * I_Degree_In; p++){
      D_A1_Z.deep() = D_A1_Z * xNew;
//      #ifdef __DEBUG_POLYFIT__
//        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: for(p(=" << p << ")...): D_A1_Z set to " << D_A1_Z << endl;
//      #endif
      if (p < nCoeffs){
        D_A1_B[p] = sum(D_A1_WY * D_A1_Z);
        #ifdef __DEBUG_POLYFIT__
          cout << "pfs::drp::stella::math::CurveFitting::PolyFit: for(p(=" << p << ")...): p < nCoeffs(=" << nCoeffs << "): D_A1_B(p) set to " << D_A1_B[p] << endl;
        #endif
      }
      if (B_HaveMeasureError){
        D_Sum = sum(D_A1_Z / D_A1_SDevSquare);
        #ifdef __DEBUG_POLYFIT__
          cout << "pfs::drp::stella::math::CurveFitting::PolyFit: for(p(=" << p << ")...): B_HaveMeasureError: D_Sum set to " << D_Sum << endl;
        #endif
      }
      else{
        D_Sum = sum(D_A1_Z);
        #ifdef __DEBUG_POLYFIT__
          cout << "pfs::drp::stella::math::CurveFitting::PolyFit: for(p(=" << p << ")...): !B_HaveMeasureError: D_Sum set to " << D_Sum << endl;
        #endif
      }
      if (p - int(I_Degree_In) > 0){
        i = p - int(I_Degree_In);
      }
      else{
        i = 0;
      }
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: for(p(=" << p << ")...): I_Degree_In = " << I_Degree_In << ": i set to " << i << endl;
      #endif
      for (j = i; j <= I_Degree_In; j++){
        (*P_D_A2_Covar)[ ndarray::makeVector( j, p-j ) ] = D_Sum;
//        #ifdef __DEBUG_POLYFIT__
//          cout << "pfs::drp::stella::math::CurveFitting::PolyFit: for(p(=" << p << ")...): for(j(=" << j << ")...): (*P_D_A2_Covar)(j,p-j=" << p-j << ") set to " << (*P_D_A2_Covar)[ ndArray::makeVector( j, p-j ) ] << endl;
//        #endif
      }
    }

    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: before InvertGaussJ: (*P_D_A2_Covar) = " << (*P_D_A2_Covar) << endl;
    #endif
    P_D_A2_Covar->asEigen() = P_D_A2_Covar->asEigen().inverse();
//      if (!pfs::drp::stella::math::InvertGaussJ(*P_D_A2_Covar)){
//        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: ERROR! InvertGaussJ(*P_D_A2_Covar=" << *P_D_A2_Covar << ") returned false!" << endl;
//        string message("pfs::drp::stella::math::CurveFitting::PolyFit: ERROR! InvertGaussJ(*P_D_A2_Covar) returned false!");
//        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
//      }
    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: InvertGaussJ: (*P_D_A2_Covar) set to " << (*P_D_A2_Covar) << endl;
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: MatrixTimesVecArr: P_D_A2_Covar->rows() = " << P_D_A2_Covar->getShape()[0] << endl;
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: MatrixTimesVecArr: P_D_A2_Covar->cols() = " << P_D_A2_Covar->getShape()[1] << endl;
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: MatrixTimesVecArr: (*P_D_A2_Covar) = " << (*P_D_A2_Covar) << endl;
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: MatrixTimesVecArr: D_A1_B = " << D_A1_B.getShape()[0] << ": " << D_A1_B << endl;
    #endif
    ndarray::Array<T, 1, 1> T_A1_Out = ndarray::allocate(D_A1_Out.getShape()[0]);
    T_A1_Out.asEigen() = P_D_A2_Covar->asEigen() * D_A1_B.asEigen();
    for (int pos = 0; pos < D_A1_Out.getShape()[0]; ++pos)
      D_A1_Out[pos] = T_A1_Out[pos];
//      D_A1_Out.deep() = MatrixTimesVecArr(*P_D_A2_Covar, D_A1_B);
    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: MatrixTimesVecArr: P_D_A1_YFit->size() = " << P_D_A1_YFit->getShape()[0] << ": D_A1_Out set to " << D_A1_Out << endl;
    #endif

    P_D_A1_YFit->deep() = D_A1_Out[I_Degree_In];
//    #ifdef __DEBUG_POLYFIT__
//      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: InvertGaussJ: (*P_D_A1_YFit) set to " << (*P_D_A1_YFit) << endl;
//    #endif

    for (int k=I_Degree_In-1; k >= 0; k--){
      P_D_A1_YFit->deep() = D_A1_Out[k] + (*P_D_A1_YFit) * xNew;
    }
    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: after for(k...): (*P_D_A1_YFit) set to " << (*P_D_A1_YFit) << endl;
    #endif

    for (int k = 0; k < nCoeffs; k++){
      (*P_D_A1_Sigma)[k] = (*P_D_A2_Covar)[ ndarray::makeVector( k, k ) ];
    }
    for (auto it = P_D_A1_Sigma->begin(); it != P_D_A1_Sigma->end(); ++it){
      *it = (*it > 0) ? sqrt(*it) : 1.;
    }
    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: (*P_D_A1_Sigma) set to " << (*P_D_A1_Sigma) << endl;
    #endif

    double D_ChiSq = 0.;
    ndarray::Array<T, 1, 1> D_A1_Diff = ndarray::allocate(nDataPoints);
    Eigen::Array<T, Eigen::Dynamic, 1> EArr_Temp = D_A1_Y_In.asEigen() - P_D_A1_YFit->asEigen();
    D_A1_Diff.asEigen() = EArr_Temp.pow(2);
    if (B_HaveMeasureError){
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: B_HaveMeasureError: D_A1_Diff set to " << D_A1_Diff << endl;
      #endif
      D_ChiSq = sum(D_A1_Diff / D_A1_SDevSquare);
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: B_HaveMeasureError: D_ChiSq set to " << D_ChiSq << endl;
      #endif

    }
    else{
      D_ChiSq = sum(D_A1_Diff);
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: !B_HaveMeasureError: D_ChiSq set to " << D_ChiSq << endl;
      #endif

      double dTemp = sqrt(D_ChiSq / (nDataPoints - nCoeffs));
      P_D_A1_Sigma->deep() = (*P_D_A1_Sigma) * dTemp;
      #ifdef __DEBUG_POLYFIT__
        cout << "pfs::drp::stella::math::CurveFitting::PolyFit: !B_HaveMeasureError: (*P_D_A1_Sigma) set to " << (*P_D_A1_Sigma) << endl;
      #endif
    }
    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: returning D_A1_Out = " << D_A1_Out << endl;
    #endif
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::PolyFit(x, y, deg, Args, ArgV) finished" << endl;
    #endif
    return D_A1_Out;
  }

  /** **********************************************************************/

  template< typename T >
  ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                       ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                       size_t const I_Degree_In,
                                       T const D_Reject_In,
                                       T xRangeMin_In,
                                       T xRangeMax_In){
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::PolyFit(x, y, deg, reject, xRangeMin, xRangeMax) started" << endl;
    #endif
    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: Starting " << endl;
    #endif
    std::vector<string> S_A1_Args(1);
    S_A1_Args[0] = "XRANGE";
    std::vector<void *> PP_Args(1);
    ndarray::Array<double, 1, 1> xRange = ndarray::allocate(2);
    xRange[0] = xRangeMin_In;
    xRange[1] = xRangeMax_In;
    PTR(ndarray::Array<double, 1, 1>) p_xRange(new ndarray::Array<double, 1, 1>(xRange));
    PP_Args[0] = &p_xRange;
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::PolyFit(x, y, deg, reject, xRangeMin, xRangeMax) finishing" << endl;
    #endif
    return pfs::drp::stella::math::PolyFit(D_A1_X_In,
                                           D_A1_Y_In,
                                           I_Degree_In,
                                           D_Reject_In,
                                           S_A1_Args,
                                           PP_Args);
  }

  template< typename T>
  ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                       ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                       size_t const I_Degree_In,
                                       T const D_LReject_In,
                                       T const D_HReject_In,
                                       size_t const I_NIter,
                                       T xRangeMin_In,
                                       T xRangeMax_In){
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::PolyFit(x, y, deg, lReject, hReject, nIter, xRangeMin, xRangeMax) started" << endl;
    #endif
    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: Starting " << endl;
    #endif
    std::vector<string> S_A1_Args(1);
    S_A1_Args[0] = "XRANGE";
    std::vector<void *> PP_Args(1);
    ndarray::Array<double, 1, 1> xRange = ndarray::allocate(2);
    xRange[0] = xRangeMin_In;
    xRange[1] = xRangeMax_In;
    PTR(ndarray::Array<double, 1, 1>) p_xRange(new ndarray::Array<double, 1, 1>(xRange));
    PP_Args[0] = &p_xRange;
    ndarray::Array<double, 1, 1> D_A1_Out = ndarray::allocate(I_Degree_In + 1);
    D_A1_Out = pfs::drp::stella::math::PolyFit(D_A1_X_In,
                                               D_A1_Y_In,
                                               I_Degree_In,
                                               D_LReject_In,
                                               D_HReject_In,
                                               I_NIter,
                                               S_A1_Args,
                                               PP_Args);
    #ifdef __DEBUG_POLYFIT__
      cout << "pfs::drp::stella::math::CurveFitting::PolyFit: PolyFit returned D_A1_Out = " << D_A1_Out << endl;
    #endif
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::PolyFit(x, y, deg, lReject, hReject, nIter, xRangeMin, xRangeMax) finished" << endl;
    #endif
    return D_A1_Out;
  }

  template< typename ImageT, typename SlitFuncT>
  bool LinFitBevingtonEigen(Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic> const& D_A2_CCD_In,
                            Eigen::Array<SlitFuncT, Eigen::Dynamic, Eigen::Dynamic> const& D_A2_SF_In,
                            Eigen::Array<ImageT, Eigen::Dynamic, 1> & D_A1_SP_Out,
                            Eigen::Array<ImageT, Eigen::Dynamic, 1> & D_A1_Sky_Out,
                            bool B_WithSky,
                            std::vector<string> const& S_A1_Args_In,
                            std::vector<void *> & ArgV_In)
  /// MEASURE_ERRORS_IN = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
  /// REJECT_IN         = double
  /// MASK_INOUT        = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
  /// CHISQ_OUT         = blitz::Array<double, 1>(D_A2_CCD_In.rows())
  /// Q_OUT             = blitz::Array<double, 1>(D_A2_CCD_In.rows())
  /// SIGMA_OUT         = blitz::Array<double, 2>(D_A2_CCD_In.rows(),2)
  /// YFIT_OUT          = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
  {
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::LinFitBevingtonEigen(D_A2_CCD, D_A2_SF, SP, Sky, withSky, Args, ArgV) started" << endl;
    #endif
    #ifdef __DEBUG_FITARR__
      cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr) started" << endl;
      cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr): D_A2_CCD_In = " << D_A2_CCD_In << endl;
      cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr): D_A2_SF_In = " << D_A2_SF_In << endl;
    #endif
    if (D_A2_CCD_In.size() != D_A2_SF_In.size()){
      string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: D_A2_CCD_In.size()(=");
      message += to_string(D_A2_CCD_In.size()) + "} != D_A2_SF_In.size()(=" + to_string(D_A2_SF_In.size()) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (D_A1_SP_Out.size() != D_A2_CCD_In.rows()){
      string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: D_A1_SP_Out.size()(=");
      message += to_string(D_A1_SP_Out.size()) + ") != D_A2_CCD_In.rows()(=" + to_string(D_A2_CCD_In.rows()) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (D_A1_Sky_Out.size() != D_A2_CCD_In.rows()){
      string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: D_A1_Sky_Out.size()(=");
      message += to_string(D_A1_Sky_Out.size()) + ") != D_A2_CCD_In.rows()(=" + to_string(D_A2_CCD_In.rows()) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    int i, I_ArgPos = 0;
    int I_KeywordSet_MeasureErrors, I_KeywordSet_Reject, I_KeywordSet_Mask, I_KeywordSet_ChiSq, I_KeywordSet_Q, I_KeywordSet_Sigma, I_KeywordSet_YFit;
    D_A1_SP_Out.setConstant(0);
    D_A1_Sky_Out.setConstant(0);

    PTR(ImageT) P_D_Reject(new ImageT(-1));

    PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>) P_D_A1_YFit(new Eigen::Array<ImageT, Eigen::Dynamic, 1>(D_A2_CCD_In.cols()));
    PTR(Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic>) P_D_A2_YFit(new Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols()));
    PTR(Eigen::Array<unsigned short, Eigen::Dynamic, 1>) P_I_A1_Mask(new Eigen::Array<unsigned short, Eigen::Dynamic, 1>(D_A2_CCD_In.cols()));
    PTR(Eigen::Array<unsigned short, Eigen::Dynamic, Eigen::Dynamic>) P_I_A2_Mask(new Eigen::Array<unsigned short, Eigen::Dynamic, Eigen::Dynamic>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols()));

    PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>) P_D_A1_Sigma(new Eigen::Array<ImageT, Eigen::Dynamic, 1>(D_A2_CCD_In.cols()));
    PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>) P_D_A1_Sigma_Out(new Eigen::Array<ImageT, Eigen::Dynamic, 1>(2));
    std::vector<string> S_A1_Args_Fit(10);
    for (auto it = S_A1_Args_Fit.begin(); it != S_A1_Args_Fit.end(); ++it)
      *it = " ";
    std::vector<void *> Args_Fit(10);

    PTR(Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic>) P_D_A2_Sigma(new Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols()));
    I_KeywordSet_MeasureErrors = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MEASURE_ERRORS_IN");
    if (I_KeywordSet_MeasureErrors >= 0)
    {
      P_D_A2_Sigma.reset();
      P_D_A2_Sigma = *((PTR(Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic>)*)ArgV_In[I_KeywordSet_MeasureErrors]);
      if (P_D_A2_Sigma->rows() != D_A2_CCD_In.rows()){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: P_D_A2_Sigma->rows()(=");
        message += to_string(P_D_A2_Sigma->rows()) +") != D_A2_CCD_In.rows()(=" + to_string(D_A2_CCD_In.rows()) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_FITARR__
        cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A2_Sigma = " << *P_D_A2_Sigma << endl;
      #endif
      S_A1_Args_Fit[I_ArgPos] = "MEASURE_ERRORS_IN";
      I_ArgPos++;
    }

    PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>) P_D_A1_ChiSq(new Eigen::Array<ImageT, Eigen::Dynamic, 1>(D_A2_CCD_In.rows()));
    I_KeywordSet_ChiSq = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "CHISQ_OUT");
    if (I_KeywordSet_ChiSq >= 0)
    {
      P_D_A1_ChiSq.reset();
      P_D_A1_ChiSq = *((PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>)*)ArgV_In[I_KeywordSet_ChiSq]);
      if (P_D_A1_ChiSq->rows() != D_A2_CCD_In.rows()){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: P_D_A1_ChiSq->rows()(=");
        message += to_string(P_D_A1_ChiSq->rows()) +") != D_A2_CCD_In.rows()(=" + to_string(D_A2_CCD_In.rows()) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      S_A1_Args_Fit[I_ArgPos] = "CHISQ_OUT";
      I_ArgPos++;
    }

    PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>) P_D_A1_Q(new Eigen::Array<ImageT, Eigen::Dynamic, 1>(D_A2_CCD_In.rows()));
    I_KeywordSet_Q = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "Q_OUT");
    if (I_KeywordSet_Q >= 0)
    {
      P_D_A1_Q.reset();
      P_D_A1_Q = *((PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>)*)ArgV_In[I_KeywordSet_Q]);
      if (P_D_A1_Q->rows() != D_A2_CCD_In.rows()){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: P_D_A1_Q->rows()(=");
        message += to_string(P_D_A1_Q->rows()) +") != D_A2_CCD_In.rows()(=" + to_string(D_A2_CCD_In.rows()) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      S_A1_Args_Fit[I_ArgPos] = "Q_OUT";
      I_ArgPos++;
    }

    PTR(Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic>) P_D_A2_Sigma_Out(new Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic>(D_A2_CCD_In.rows(), 2));
    I_KeywordSet_Sigma = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "SIGMA_OUT");
    if (I_KeywordSet_Sigma >= 0)
    {
      P_D_A2_Sigma_Out.reset();
      P_D_A2_Sigma_Out = *((PTR(Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic>)*)ArgV_In[I_KeywordSet_Sigma]);
      if (P_D_A2_Sigma_Out->rows() != D_A2_CCD_In.rows()){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: P_D_A2_Sigma_Out->rows()(=");
        message += to_string(P_D_A2_Sigma_Out->rows()) +") != D_A2_CCD_In.rows()(=" + to_string(D_A2_CCD_In.rows()) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (P_D_A2_Sigma_Out->cols() != 2){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: P_D_A2_Sigma_Out->cols()(=");
        message += to_string(P_D_A2_Sigma_Out->cols()) +") != 2";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      S_A1_Args_Fit[I_ArgPos] = "SIGMA_OUT";
      I_ArgPos++;
    }

    I_KeywordSet_YFit = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "YFIT_OUT");
    if (I_KeywordSet_YFit >= 0)
    {
      P_D_A2_YFit.reset();
      P_D_A2_YFit = *((PTR(Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic>)*)ArgV_In[I_KeywordSet_YFit]);
      if (P_D_A2_YFit->rows() != D_A2_CCD_In.rows()){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: P_D_A2_YFig->rows()(=");
        message += to_string(P_D_A2_YFit->rows()) +") != D_A2_CCD_In.rows()(=" + to_string(D_A2_CCD_In.rows()) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (P_D_A2_YFit->cols() != D_A2_CCD_In.cols()){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: P_D_A2_YFit->cols()(=");
        message += to_string(P_D_A2_YFit->cols()) +") != D_A2_CCD_In.cols()(=" + to_string(D_A2_CCD_In.cols()) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      S_A1_Args_Fit[I_ArgPos] = "YFIT_OUT";
      I_ArgPos++;
    }

    I_KeywordSet_Reject = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "REJECT_IN");
    if (I_KeywordSet_Reject >= 0)
    {
      P_D_Reject.reset();
      P_D_Reject = *((PTR(ImageT)*)ArgV_In[I_KeywordSet_Reject]);
      #ifdef __DEBUG_FITARR__
        cout << "CFits::LinFitBevington2D: P_D_Reject = " << *P_D_Reject << endl;
      #endif
      S_A1_Args_Fit[I_ArgPos] = "REJECT_IN";
      I_ArgPos++;
    }

    I_KeywordSet_Mask = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MASK_INOUT");
    if (I_KeywordSet_Mask >= 0)
    {
      P_I_A2_Mask.reset();
      P_I_A2_Mask = *((PTR(Eigen::Array<unsigned short, Eigen::Dynamic, Eigen::Dynamic>)*)ArgV_In[I_KeywordSet_Mask]);
      if (P_I_A2_Mask->rows() != D_A2_CCD_In.rows()){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: P_I_A2_Mask->rows()(=");
        message += to_string(P_I_A2_Mask->rows()) +") != D_A2_CCD_In.rows()(=" + to_string(D_A2_CCD_In.rows()) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (P_I_A2_Mask->cols() != D_A2_CCD_In.cols()){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: P_I_A2_Mask->cols()(=");
        message += to_string(P_I_A2_Mask->cols()) +") != D_A2_CCD_In.cols()(=" + to_string(D_A2_CCD_In.cols()) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_FITARR__
        cout << "CFits::LinFitBevington2D: P_I_A2_Mask = " << *P_I_A2_Mask << endl;
      #endif
      S_A1_Args_Fit[I_ArgPos] = "MASK_INOUT";
      I_ArgPos++;
    }

    bool B_DoFit = true;
    for (i = 0; i < D_A2_CCD_In.rows(); i++)
    {
      I_ArgPos = 0;
      if (I_KeywordSet_MeasureErrors >= 0){
        *P_D_A1_Sigma = P_D_A2_Sigma->row(i);
        #ifdef __DEBUG_FITARR__
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A1_Sigma set to " << *P_D_A1_Sigma << endl;
        #endif
        Args_Fit[I_ArgPos] = &P_D_A1_Sigma;
        #ifdef __DEBUG_FITARR__
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): PP_Args_Fit[I_ArgPos=" << I_ArgPos << "] = " << *((PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>)*)Args_Fit[I_ArgPos]) << endl;
        #endif
        I_ArgPos++;
      }

      if (I_KeywordSet_ChiSq >= 0){
        Args_Fit[I_ArgPos] = &((*P_D_A1_ChiSq)(i));
        I_ArgPos++;
      }

      if (I_KeywordSet_Q >= 0){
        Args_Fit[I_ArgPos] = &((*P_D_A1_Q)(i));
        I_ArgPos++;
      }

      if (I_KeywordSet_Sigma >= 0){
        *P_D_A1_Sigma_Out = P_D_A2_Sigma_Out->row(i);
        Args_Fit[I_ArgPos] = &P_D_A1_Sigma_Out;
        I_ArgPos++;
      }

      if (I_KeywordSet_YFit >= 0){
        *P_D_A1_YFit = P_D_A2_YFit->row(i);
        Args_Fit[I_ArgPos] = &P_D_A1_YFit;
        I_ArgPos++;
      }

      if (I_KeywordSet_Reject >= 0){
        Args_Fit[I_ArgPos] = &P_D_Reject;
        I_ArgPos++;
      }

      B_DoFit = true;
      if (I_KeywordSet_Mask >= 0){
        *P_I_A1_Mask = P_I_A2_Mask->row(i);
        Args_Fit[I_ArgPos] = &P_I_A1_Mask;
        I_ArgPos++;
        if (P_I_A1_Mask->sum() == 0)
          B_DoFit = false;
      }

      #ifdef __DEBUG_FITARR__
        cout << "CFits::LinFitBevington: Starting Fit1D: D_A2_CCD_In(i=" << i << ", *) = " << D_A2_CCD_In.row(i) << endl;
      #endif
      if (B_DoFit){
        #ifdef __DEBUG_FITARR__
          cout << "CFits::LinFitBevington: D_A2_SF_In(i=" << i << ", *) = " << D_A2_SF_In.row(i) << endl;
        #endif
        Eigen::Array<ImageT, Eigen::Dynamic, 1> ccdRow(D_A2_CCD_In.row(i));
        Eigen::Array<SlitFuncT, Eigen::Dynamic, 1> slitFuncRow(D_A2_SF_In.row(i));
        int status = pfs::drp::stella::math::LinFitBevingtonEigen(ccdRow,
                                                                  slitFuncRow,
                                                                  D_A1_SP_Out(i),
                                                                  D_A1_Sky_Out(i),
                                                                  B_WithSky,
                                                                  S_A1_Args_Fit,
                                                                  Args_Fit);
        if (status != 1){
          #ifdef __WARNINGS_ON__
            string message("CFits::LinFitBevington: WARNING: LinFitBevington(D_A2_CCD_In(i,blitz::Range::all()),D_A2_SF_In(i,blitz::Range::all()),D_A1_SP_Out(i),D_A1_Sky_Out(i),D_A1_STDDEV_Out(i),D_A1_Covariance_Out(i)) returned status = ");
            message += to_string(status);
            cout << message << endl;
            cout << "CFits::LinFitBevington: D_A2_SF_In(0, *) = " << D_A2_SF_In.row(0) << ": LinFitBevingtonEigen returned status = " << status << endl;
          #endif
//            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
      }
      #ifdef __DEBUG_FITARR__
        cout << "CFits::LinFitBevington(Array, Array, Array, Array): D_A1_SP_Out(i=" << i << ") set to " << D_A1_SP_Out(i) << endl;
        cout << "CFits::LinFitBevington(Array, Array, Array, Array): D_A1_Sky_Out(i=" << i << ") set to " << D_A1_Sky_Out(i) << endl;
      #endif

      I_ArgPos = 0;
      if (I_KeywordSet_MeasureErrors >= 0){
        P_D_A2_Sigma->row(i) = (*P_D_A1_Sigma);
        #ifdef __DEBUG_FITARR__
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A2_Sigma(i=" << i << ",*) set to " << P_D_A2_Sigma->row(i) << endl;
        #endif
        I_ArgPos++;
      }

      if (I_KeywordSet_Sigma >= 0){
        P_D_A2_Sigma_Out->row(i) = (*P_D_A1_Sigma_Out);
        #ifdef __DEBUG_FITARR__
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A2_Sigma_Out(i=" << i << ",*) set to " << P_D_A2_Sigma_Out->row(i) << endl;
        #endif
        I_ArgPos++;
      }

      if (I_KeywordSet_YFit >= 0){
        P_D_A2_YFit->row(i) = (*P_D_A1_YFit);
        #ifdef __DEBUG_FITARR__
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A2_YFit(i=" << i << ",*) set to " << P_D_A2_YFit->row(i) << endl;
        #endif
        I_ArgPos++;
      }

      if (I_KeywordSet_Mask >= 0){
        P_I_A2_Mask->row(i) = (*P_I_A1_Mask);
        #ifdef __DEBUG_FITARR__
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_I_A1_Mask = " << (*P_I_A1_Mask) << endl;
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_I_A2_Mask(i=" << i << ",*) set to " << P_I_A2_Mask->row(i) << endl;
        #endif
        I_ArgPos++;
      }
    }
    #ifdef __DEBUG_FITARR__
      cout << "CFits::LinFitBevington(Array, Array, Array, Array): D_A1_SP_Out = " << D_A1_SP_Out << endl;
//        cout << "CFits::LinFitBevington(Array, Array, Array, Array): D_A1_Sky_Out set to " << D_A1_Sky_Out << endl;
    #endif
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::LinFitBevingtonEigen(D_A2_CCD, D_A2_SF, SP, Sky, withSky, Args, ArgV) finished" << endl;
    #endif
    return true;
  }

  template< typename ImageT, typename SlitFuncT>
  bool LinFitBevingtonNdArray(ndarray::Array<ImageT, 2, 1> const& D_A2_CCD_In,
                              ndarray::Array<SlitFuncT, 2, 1> const& D_A2_SF_In,
                              ndarray::Array<ImageT, 1, 1> & D_A1_SP_Out,
                              ndarray::Array<ImageT, 1, 1> & D_A1_Sky_Out,
                              bool B_WithSky,
                              std::vector<string> const& S_A1_Args_In,
                              std::vector<void *> & ArgV_In)
  /// MEASURE_ERRORS_IN = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
  /// REJECT_IN         = double
  /// MASK_INOUT        = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
  /// CHISQ_OUT         = blitz::Array<double, 1>(D_A2_CCD_In.rows())
  /// Q_OUT             = blitz::Array<double, 1>(D_A2_CCD_In.rows())
  /// SIGMA_OUT         = blitz::Array<double, 2>(D_A2_CCD_In.rows(),2)
  /// YFIT_OUT          = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
  {
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::LinFitBevingtonNdArray(D_A2_CCD, D_A2_SF, SP, Sky, withSky, Args, ArgV) started" << endl;
    #endif
    #ifdef __DEBUG_FITARR__
      cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr) started" << endl;
      cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr): D_A2_CCD_In = " << D_A2_CCD_In << endl;
      cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr): D_A2_SF_In = " << D_A2_SF_In << endl;
    #endif
    if (D_A2_CCD_In.getShape()[0] != D_A2_SF_In.getShape()[0]){
      string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
      message += to_string(D_A2_CCD_In.getShape()[0]) + ") != D_A2_SF_In.getShape()[0](=" + to_string(D_A2_SF_In.getShape()[0]) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (D_A2_CCD_In.getShape()[1] != D_A2_SF_In.getShape()[1]){
      string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[1](=");
      message += to_string(D_A2_CCD_In.getShape()[1]) + ") != D_A2_SF_In.getShape()[1](=" + to_string(D_A2_SF_In.getShape()[1]) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (D_A1_SP_Out.getShape()[0] != D_A2_CCD_In.getShape()[0]){
      string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
      message += to_string(D_A2_CCD_In.getShape()[0]) + ") != D_A1_SP_Out.getShape()[0](=" + to_string(D_A1_SP_Out.getShape()[0]) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (D_A1_Sky_Out.getShape()[0] != D_A2_CCD_In.getShape()[0]){
      string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
      message += to_string(D_A2_CCD_In.getShape()[0]) + ") != D_A1_Sky_Out.getShape()[0](=" + to_string(D_A1_Sky_Out.getShape()[0]) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    int i, I_ArgPos = 0;
    int I_KeywordSet_MeasureErrors, I_KeywordSet_Reject, I_KeywordSet_Mask, I_KeywordSet_ChiSq, I_KeywordSet_Q, I_KeywordSet_Sigma, I_KeywordSet_YFit;
    D_A1_SP_Out.deep() = 0;
    D_A1_Sky_Out.deep() = 0;

    std::vector<string> S_A1_Args_Fit(10);
    for (auto it = S_A1_Args_Fit.begin(); it != S_A1_Args_Fit.end(); ++it)
      *it = " ";
    std::vector<void *> Args_Fit(10);

    ndarray::Array<ImageT, 1, 1> D_A1_Sigma = ndarray::allocate(D_A2_CCD_In.getShape()[1]);
    PTR(ndarray::Array<ImageT, 1, 1>) P_D_A1_Sigma(new ndarray::Array<ImageT, 1, 1>(D_A1_Sigma));

    ndarray::Array<ImageT, 2, 2> D_A2_Sigma = ndarray::allocate(D_A2_CCD_In.getShape()[0], D_A2_CCD_In.getShape()[1]);
    PTR(ndarray::Array<ImageT, 2, 2>) P_D_A2_Sigma(new ndarray::Array<ImageT, 2, 2>(D_A2_Sigma));
    I_KeywordSet_MeasureErrors = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MEASURE_ERRORS_IN");
    if (I_KeywordSet_MeasureErrors >= 0)
    {
      P_D_A2_Sigma.reset();
      P_D_A2_Sigma = *((PTR(ndarray::Array<ImageT, 2, 2>)*)ArgV_In[I_KeywordSet_MeasureErrors]);
      if (P_D_A2_Sigma->getShape()[0] != D_A2_CCD_In.getShape()[0]){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
        message += to_string(D_A2_CCD_In.getShape()[0]) + ") != P_D_A2_Sigma->getShape()[0](=" + to_string(P_D_A2_Sigma->getShape()[0]) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (P_D_A2_Sigma->getShape()[1] != D_A2_CCD_In.getShape()[1]){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[1](=");
        message += to_string(D_A2_CCD_In.getShape()[1]) + ") != P_D_A2_Sigma->getShape()[1](=" + to_string(P_D_A2_Sigma->getShape()[1]) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_FITARR__
        cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A2_Sigma = " << *P_D_A2_Sigma << endl;
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
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
        message += to_string(D_A2_CCD_In.getShape()[0]) + ") != P_D_A1_ChiSq->getShape()[0](=" + to_string(P_D_A1_ChiSq->getShape()[0]) + ")";
        cout << message << endl;
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
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
        message += to_string(D_A2_CCD_In.getShape()[0]) + ") != P_D_A1_Q->getShape()[0](=" + to_string(P_D_A1_Q->getShape()[0]) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      S_A1_Args_Fit[I_ArgPos] = "Q_OUT";
      I_ArgPos++;
    }

    ndarray::Array<ImageT, 1, 1> D_A1_Sigma_Out = ndarray::allocate(2);
    PTR(ndarray::Array<ImageT, 1, 1>) P_D_A1_Sigma_Out(new ndarray::Array<ImageT, 1, 1>(D_A1_Sigma_Out));

    ndarray::Array<ImageT, 2, 2> D_A2_Sigma_Out = ndarray::allocate(D_A2_CCD_In.getShape()[0], 2);
    PTR(ndarray::Array<ImageT, 2, 2>) P_D_A2_Sigma_Out(new ndarray::Array<ImageT, 2, 2>(D_A2_Sigma_Out));
    I_KeywordSet_Sigma = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "SIGMA_OUT");
    if (I_KeywordSet_Sigma >= 0)
    {
      P_D_A2_Sigma_Out.reset();
      P_D_A2_Sigma_Out = *((PTR(ndarray::Array<ImageT, 2, 2>)*)ArgV_In[I_KeywordSet_Sigma]);
      if (P_D_A2_Sigma_Out->getShape()[0] != D_A2_CCD_In.getShape()[0]){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
        message += to_string(D_A2_CCD_In.getShape()[0]) + ") != P_D_A2_Sigma_Out->getShape()[0](=" + to_string(P_D_A2_Sigma_Out->getShape()[0]) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (P_D_A2_Sigma_Out->getShape()[1] != 2){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: P_D_A2_Sigma->getShape()[1](=");
        message += to_string(P_D_A2_Sigma->getShape()[0]) + ") != 2";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      S_A1_Args_Fit[I_ArgPos] = "SIGMA_OUT";
      I_ArgPos++;
    }


    ndarray::Array<ImageT, 1, 1> D_A1_YFit = ndarray::allocate(D_A2_CCD_In.getShape()[1]);
    PTR(ndarray::Array<ImageT, 1, 1>) P_D_A1_YFit(new ndarray::Array<ImageT, 1, 1>(D_A1_YFit));

    ndarray::Array<ImageT, 2, 2> D_A2_YFit = ndarray::allocate(D_A2_CCD_In.getShape()[0], D_A2_CCD_In.getShape()[1]);
    PTR(ndarray::Array<ImageT, 2, 2>) P_D_A2_YFit(new ndarray::Array<ImageT, 2, 2>(D_A2_YFit));
    I_KeywordSet_YFit = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "YFIT_OUT");
    if (I_KeywordSet_YFit >= 0)
    {
      P_D_A2_YFit.reset();
      P_D_A2_YFit = *((PTR(ndarray::Array<ImageT, 2, 2>)*)ArgV_In[I_KeywordSet_YFit]);
      if (P_D_A2_YFit->getShape()[0] != D_A2_CCD_In.getShape()[0]){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
        message += to_string(D_A2_CCD_In.getShape()[0]) + ") != P_D_A2_YFit->getShape()[0](=" + to_string(P_D_A2_YFit->getShape()[0]) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (P_D_A2_YFit->getShape()[1] != D_A2_CCD_In.getShape()[1]){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[1](=");
        message += to_string(D_A2_CCD_In.getShape()[1]) + ") != P_D_A2_YFit->getShape()[1](=" + to_string(P_D_A2_YFit->getShape()[1]) + ")";
        cout << message << endl;
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
        cout << "CFits::LinFitBevington2D: P_D_Reject = " << *P_D_Reject << endl;
      #endif
      S_A1_Args_Fit[I_ArgPos] = "REJECT_IN";
      I_ArgPos++;
    }

    ndarray::Array<unsigned short, 1, 1> I_A1_Mask = ndarray::allocate(D_A2_CCD_In.getShape()[1]);
    PTR(ndarray::Array<unsigned short, 1, 1>) P_I_A1_Mask(new ndarray::Array<unsigned short, 1, 1>(I_A1_Mask));

    ndarray::Array<unsigned short, 2, 2> I_A2_Mask = ndarray::allocate(D_A2_CCD_In.getShape()[0], D_A2_CCD_In.getShape()[1]);
    I_A2_Mask.deep() = 1;
    PTR(ndarray::Array<unsigned short, 2, 2>) P_I_A2_Mask(new ndarray::Array<unsigned short, 2, 2>(I_A2_Mask));
    I_KeywordSet_Mask = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MASK_INOUT");
    if (I_KeywordSet_Mask >= 0)
    {
      P_I_A2_Mask.reset();
      P_I_A2_Mask = *((PTR(ndarray::Array<unsigned short, 2, 2>)*)ArgV_In[I_KeywordSet_Mask]);
      if (P_I_A2_Mask->getShape()[0] != D_A2_CCD_In.getShape()[0]){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[0](=");
        message += to_string(D_A2_CCD_In.getShape()[0]) + ") != P_I_A2_Mask->getShape()[0](=" + to_string(P_I_A2_Mask->getShape()[0]) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (P_I_A2_Mask->getShape()[1] != D_A2_CCD_In.getShape()[1]){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: D_A2_CCD_In.getShape()[1](=");
        message += to_string(D_A2_CCD_In.getShape()[1]) + ") != P_I_A2_Mask->getShape()[1](=" + to_string(P_I_A2_Mask->getShape()[1]) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_FITARR__
        cout << "CFits::LinFitBevington2D: P_I_A2_Mask = " << *P_I_A2_Mask << endl;
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
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A1_Sigma set to " << *P_D_A1_Sigma << endl;
        #endif
        Args_Fit[I_ArgPos] = &P_D_A1_Sigma;
        #ifdef __DEBUG_FITARR__
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): PP_Args_Fit[I_ArgPos=" << I_ArgPos << "] = " << *((PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>)*)Args_Fit[I_ArgPos]) << endl;
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
        cout << "CFits::LinFitBevington: Starting Fit1D: D_A2_CCD_In(i=" << i << ", *) = " << D_A2_CCD_In[ndarray::view(i)()] << endl;
      #endif
      if (B_DoFit){
        #ifdef __DEBUG_FITARR__
          cout << "CFits::LinFitBevington: D_A2_SF_In(i=" << i << ", *) = " << D_A2_SF_In[ndarray::view(i)()] << endl;
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
            string message("CFits::LinFitBevington: WARNING: LinFitBevington(D_A2_CCD_In(i,blitz::Range::all()),D_A2_SF_In(i,blitz::Range::all()),D_A1_SP_Out(i),D_A1_Sky_Out(i),D_A1_STDDEV_Out(i),D_A1_Covariance_Out(i)) returned status = ");
            message += to_string(status);
            cout << message << endl;
            cout << "CFits::LinFitBevington: D_A2_SF_In(0, *) = " << D_A2_SF_In[ndarray::view(0)()] << ": LinFitBevingtonNdArray returned status = " << status << endl;
          #endif
//            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
      }
      #ifdef __DEBUG_FITARR__
        cout << "CFits::LinFitBevington(Array, Array, Array, Array): D_A1_SP_Out(i=" << i << ") set to " << D_A1_SP_Out[i] << endl;
        cout << "CFits::LinFitBevington(Array, Array, Array, Array): D_A1_Sky_Out(i=" << i << ") set to " << D_A1_Sky_Out[i] << endl;
      #endif

      if (I_KeywordSet_Sigma >= 0){
        #ifdef __DEBUG_FITARR__
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A1_Sigma_Out = " << (*P_D_A1_Sigma_Out) << endl;
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A2_Sigma_Out(i=" << i << ",*) = " << (*P_D_A2_Sigma_Out)[ndarray::view(i)()] << endl;
        #endif
        (*P_D_A2_Sigma_Out)[ndarray::view(i)()] = (*P_D_A1_Sigma_Out);
      }

      if (I_KeywordSet_YFit >= 0){
        (*P_D_A2_YFit)[ndarray::view(i)()] = (*P_D_A1_YFit);
        #ifdef __DEBUG_FITARR__
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_D_A2_YFit(i=" << i << ",*) set to " << (*P_D_A2_YFit)[ndarray::view(i)()] << endl;
        #endif
      }

      if (I_KeywordSet_Mask >= 0){
        (*P_I_A2_Mask)[ndarray::view(i)()] = (*P_I_A1_Mask);
        #ifdef __DEBUG_FITARR__
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_I_A1_Mask = " << (*P_I_A1_Mask) << endl;
          cout << "CFits::LinFitBevington(Array, Array, Array, Array): P_I_A2_Mask(i=" << i << ",*) set to " << (*P_I_A2_Mask)[ndarray::view(i)()] << endl;
        #endif
      }
    }
    #ifdef __DEBUG_FITARR__
      cout << "CFits::LinFitBevington(Array, Array, Array, Array): D_A1_SP_Out = " << D_A1_SP_Out << endl;
//        cout << "CFits::LinFitBevington(Array, Array, Array, Array): D_A1_Sky_Out set to " << D_A1_Sky_Out << endl;
    #endif
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::LinFitBevingtonNdArray(D_A2_CCD, D_A2_SF, SP, Sky, withSky, Args, ArgV) finished" << endl;
    #endif
    return true;
  }

  template<typename ImageT, typename SlitFuncT>
  int LinFitBevingtonEigen(Eigen::Array<ImageT, Eigen::Dynamic, 1> const& D_A1_CCD_In,
                            Eigen::Array<SlitFuncT, Eigen::Dynamic, 1> const& D_A1_SF_In,
                            ImageT &D_SP_Out,
                            ImageT &D_Sky_Out,
                            bool B_WithSky,
                            std::vector<string> const& S_A1_Args_In,
                            std::vector<void *> & ArgV_In)
  /// MEASURE_ERRORS_IN = blitz::Array<double,1>(D_A1_CCD_In.size)
  /// REJECT_IN = double
  /// MASK_INOUT = blitz::Array<double,1>(D_A1_CCD_In.size)
  /// CHISQ_OUT = double
  /// Q_OUT = double
  /// SIGMA_OUT = blitz::Array<double,1>(2): [0]: sigma_sp, [1]: sigma_sky
  /// YFIT_OUT = blitz::Array<double, 1>(D_A1_CCD_In.size)
  /// ALLOW_SKY_LT_ZERO = 1
  /// ALLOW_SPEC_LT_ZERO = 1
  {
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::LinFitBevingtonEigen(D_A1_CCD, D_A1_SF, SP, Sky, withSky, Args, ArgV) started" << endl;
    #endif
    int status = 1;
    #ifdef __DEBUG_FIT__
      cout << "CFits::LinFitBevington(Array, Array, double, double, bool, CSArr, PPArr) started" << endl;
      cout << "CFits::LinFitBevington: D_A1_CCD_In = " << D_A1_CCD_In << endl;
      cout << "CFits::LinFitBevington: D_A1_SF_In = " << D_A1_SF_In << endl;
    #endif

    if (D_A1_CCD_In.size() != D_A1_SF_In.size()){
      string message("CFits::LinFitBevington: ERROR: D_A1_CCD_In.size(=");
      message += to_string(D_A1_CCD_In.size()) + ") != D_A1_SF_In.size(=" + to_string(D_A1_SF_In.size()) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    //  /// Set D_A1_SF_In to zero where D_A1_CCD_In == zero
    Eigen::Array<SlitFuncT, Eigen::Dynamic, 1> D_A1_SF(D_A1_SF_In.size());
    D_A1_SF = D_A1_SF_In;
    Eigen::Array<ImageT, Eigen::Dynamic, 1> D_A1_CCD(D_A1_CCD_In.size());
    D_A1_CCD = D_A1_CCD_In;

    if ((D_A1_CCD_In.sum() == 0.) || (D_A1_SF.sum() == 0.)){
      #ifdef __WARNINGS_ON__
        cout << "CFits::LinFitBevington: Warning: (D_A1_CCD_In.sum(=" << D_A1_CCD_In.sum() << " == 0.) || (D_A1_SF.sum(=" << D_A1_SF.sum() << ") == 0.) => returning false" << endl;
      #endif
      D_SP_Out = 0.;
      D_Sky_Out = 0.;
      status = 0;
      return status;
    }
    int i, I_Pos;
    int I_KeywordSet_Reject, I_KeywordSet_Mask, I_KeywordSet_MeasureErrors, I_KeywordSet_SigmaOut, I_KeywordSet_ChiSqOut, I_KeywordSet_QOut, I_KeywordSet_YFitOut, I_KeywordSet_AllowSkyLTZero, I_KeywordSet_AllowSpecLTZero;
    double sigdat;
    int ndata = D_A1_CCD_In.size();
    PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>) P_D_A1_Sig(new Eigen::Array<ImageT, Eigen::Dynamic, 1>(D_A1_CCD_In.size()));
    P_D_A1_Sig->setConstant(0.);
    Eigen::Array<ImageT, Eigen::Dynamic, 1> D_A1_Sig(D_A1_CCD_In.size());
    Eigen::Array<ImageT, Eigen::Dynamic, 1> D_A1_WT(ndata);

    /// a: D_Sky_Out
    /// b: D_SP_Out
    /// x: D_A1_SF_In
    /// y: D_A1_CCD_In
    bool B_AllowSkyLTZero = false;
    I_KeywordSet_AllowSkyLTZero = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "ALLOW_SKY_LT_ZERO");
    if (I_KeywordSet_AllowSkyLTZero >= 0){
      if(*((int*)ArgV_In[I_KeywordSet_AllowSkyLTZero]) > 0){
        B_AllowSkyLTZero = true;
        cout << "CFits::LinFitBevington: KeyWord_Set(ALLOW_SKY_LT_ZERO)" << endl;
      }
    }

    bool B_AllowSpecLTZero = false;
    I_KeywordSet_AllowSpecLTZero = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "ALLOW_SPEC_LT_ZERO");
    if (I_KeywordSet_AllowSpecLTZero >= 0){
      if (I_KeywordSet_AllowSkyLTZero < 0){
        if (*((int*)ArgV_In[I_KeywordSet_AllowSkyLTZero]) > 0){
          B_AllowSpecLTZero = true;
          cout << "CFits::LinFitBevington: KeyWord_Set(ALLOW_SPEC_LT_ZERO)" << endl;
        }
      }
    }

    float D_Reject(-1.);
    I_KeywordSet_Reject = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "REJECT_IN");
    if (I_KeywordSet_Reject >= 0)
    {
      D_Reject = *(float*)ArgV_In[I_KeywordSet_Reject];
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: KeyWord_Set(REJECT_IN): D_Reject = " << D_Reject << endl;
      #endif
    }
    bool B_Reject = false;
    if (D_Reject > 0.)
      B_Reject = true;

    PTR(Eigen::Array<unsigned short, Eigen::Dynamic, 1>) P_I_A1_Mask(new Eigen::Array<unsigned short, Eigen::Dynamic, 1>(D_A1_CCD_In.size()));
    Eigen::Array<unsigned short, Eigen::Dynamic, 1> I_A1_Mask_Orig(D_A1_CCD_In.size());
    P_I_A1_Mask->setConstant(1);
    I_KeywordSet_Mask = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MASK_INOUT");
    if (I_KeywordSet_Mask >= 0)
    {
      P_I_A1_Mask.reset();
      P_I_A1_Mask = *((PTR(Eigen::Array<unsigned short, Eigen::Dynamic, 1>)*)ArgV_In[I_KeywordSet_Mask]);
      if (P_I_A1_Mask->size() != D_A1_CCD_In.size()){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: P_I_A1_Mask->size()(=");
        message += to_string(P_I_A1_Mask->size()) + ") != D_A1_CCD_In.size()(=" + to_string(D_A1_CCD_In.size()) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: KeyWord_Set(MASK_INOUT): *P_I_A1_Mask = " << *P_I_A1_Mask << endl;
      #endif
    }
    I_A1_Mask_Orig = *P_I_A1_Mask;
    #ifdef __DEBUG_FIT__
      cout << "CFits::LinFitBevington: *P_I_A1_Mask set to " << *P_I_A1_Mask << endl;
      cout << "CFits::LinFitBevington: I_A1_Mask_Orig set to " << I_A1_Mask_Orig << endl;
    #endif

    PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>) P_D_A1_Sigma_Out(new Eigen::Array<ImageT, Eigen::Dynamic, 1>(2));
    Eigen::Array<ImageT, Eigen::Dynamic, 1> D_A1_Sigma_Out(2);
    I_KeywordSet_SigmaOut = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "SIGMA_OUT");
    if (I_KeywordSet_SigmaOut >= 0)
    {
      P_D_A1_Sigma_Out.reset();
      P_D_A1_Sigma_Out = *(PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>)*)ArgV_In[I_KeywordSet_SigmaOut];
      if (P_D_A1_Sigma_Out->size() != 2){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: P_D_A1_Sigma_Out->size()(=");
        message += to_string(P_D_A1_Sigma_Out->size()) + ") != 2";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: KeyWord_Set(SIGMA_OUT)" << endl;
      #endif
    }
    P_D_A1_Sigma_Out->setConstant(0.);
    D_A1_Sigma_Out = *P_D_A1_Sigma_Out;

    PTR(ImageT) P_D_ChiSqr_Out(new ImageT(0.));
    I_KeywordSet_ChiSqOut = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "CHISQ_OUT");
    if (I_KeywordSet_ChiSqOut >= 0)
    {
      P_D_ChiSqr_Out.reset();
      P_D_ChiSqr_Out = *(PTR(ImageT)*)ArgV_In[I_KeywordSet_ChiSqOut];
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: KeyWord_Set(CHISQ_OUT)" << endl;
      #endif
    }
    *P_D_ChiSqr_Out = 0.;

    PTR(ImageT) P_D_Q_Out(new ImageT(0.));
    I_KeywordSet_QOut = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "Q_OUT");
    if (I_KeywordSet_QOut >= 0)
    {
      P_D_Q_Out.reset();
      P_D_Q_Out = *(PTR(ImageT)*)ArgV_In[I_KeywordSet_QOut];
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: KeyWord_Set(Q_OUT)" << endl;
      #endif
    }
    *P_D_Q_Out = 1.;

    D_SP_Out = 0.0;
    I_KeywordSet_MeasureErrors = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MEASURE_ERRORS_IN");
    if (I_KeywordSet_MeasureErrors >= 0)
    {
      /// Accumulate sums...
      P_D_A1_Sig.reset();
      P_D_A1_Sig = *(PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>)*)ArgV_In[I_KeywordSet_MeasureErrors];
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: *P_D_A1_Sig = " << *P_D_A1_Sig << endl;
      #endif
      if (D_A1_CCD_In.size() != P_D_A1_Sig->size()){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: P_D_A1_Sig->size()(=");
        message += to_string(P_D_A1_Sig->size()) + ") != D_A1_CCD_In.size()(=" + to_string(D_A1_CCD_In.size()) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: KeyWord_Set(MEASURE_ERRORS_IN): *P_D_A1_Sig = " << *P_D_A1_Sig << endl;
      #endif
    }

    Eigen::Array<ImageT, Eigen::Dynamic, 1> D_A1_YFit(D_A1_CCD_In.size());
    PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>) P_D_A1_YFit(new Eigen::Array<ImageT, Eigen::Dynamic, 1>(D_A1_CCD_In.size()));
    I_KeywordSet_YFitOut = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "YFIT_OUT");
    if (I_KeywordSet_YFitOut >= 0)
    {
      P_D_A1_YFit.reset();
      P_D_A1_YFit = *(PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>)*)ArgV_In[I_KeywordSet_YFitOut];
      if (P_D_A1_YFit->size() != D_A1_CCD_In.size()){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonEigen: ERROR: P_D_A1_YFit->size()(=");
        message += to_string(P_D_A1_YFit->size()) + ") != D_A1_CCD_In.size()(=" + to_string(D_A1_CCD_In.size()) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      P_D_A1_YFit->setConstant(0.);
    }
    if (P_I_A1_Mask->sum() == 0){
      #ifdef __WARNINGS_ON__
        cout << "CFits::LinFitBevington: WARNING: P_I_A1_Mask->sum() == 0" << endl;
      #endif
      D_SP_Out = 0.;
      D_Sky_Out = 0.;
      status = 0;
      return status;
    }

    int I_SumMaskLast;
    ImageT D_SDevReject;
    Eigen::Array<ImageT, Eigen::Dynamic, 1> D_A1_Check(D_A1_CCD_In.size());
    Eigen::Array<unsigned short, Eigen::Dynamic, 1> I_A1_LastMask(P_I_A1_Mask->size());
    Eigen::Array<ImageT, Eigen::Dynamic, 1> D_A1_Diff(D_A1_CCD_In.size());
    D_A1_Diff.setConstant(0.);
    ImageT D_Sum_Weights = 0.;
    ImageT D_Sum_XSquareTimesWeight = 0;
    ImageT D_Sum_XTimesWeight = 0.;
    ImageT D_Sum_YTimesWeight = 0.;
    ImageT D_Sum_XYTimesWeight = 0.;
    ImageT D_Delta = 0.;

    bool B_Run = true;
    int I_Run = -1;
    int I_MaskSum;
    while (B_Run){
      D_SP_Out = 0.0;

      I_Run++;
      /// remove bad pixels marked by mask
      I_MaskSum = P_I_A1_Mask->sum();
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": I_MaskSum = " << I_MaskSum << endl;
      #endif
      if (I_MaskSum == 0){
        string message("LinFitBevington: ERROR: I_MaskSum == 0");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      D_A1_Sig.resize(I_MaskSum);
      D_A1_CCD.resize(I_MaskSum);
      D_A1_SF.resize(I_MaskSum);
      D_A1_WT.resize(I_MaskSum);
      D_A1_YFit.resize(I_MaskSum);
      D_A1_Sig.setConstant(0.);
      D_A1_CCD.setConstant(0.);
      D_A1_SF.setConstant(0.);
      D_A1_WT.setConstant(0.);
      D_A1_YFit.setConstant(0.);

      I_Pos = 0;
      for (size_t ii = 0; ii < P_I_A1_Mask->size(); ii++){
        if ((*P_I_A1_Mask)(ii) == 1){
          D_A1_CCD(I_Pos) = D_A1_CCD_In(ii);
          D_A1_SF(I_Pos) = D_A1_SF_In(ii);
          D_A1_Sig(I_Pos) = (*P_D_A1_Sig)(ii);
          I_Pos++;
        }
      }
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_CCD set to " << D_A1_CCD << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_SF set to " << D_A1_SF << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_Sig set to " << D_A1_Sig << endl;
      #endif

      D_Sum_Weights = 0.;
      D_Sum_XSquareTimesWeight = 0.;
      D_Sum_XTimesWeight = 0.;
      D_Sum_XYTimesWeight = 0.;
      D_Sum_YTimesWeight = 0.;
      if (I_KeywordSet_MeasureErrors >= 0)
      {
        ///    D_A1_WT = D_A1_SF;
        for (i=0; i < I_MaskSum; i++)
        {
          /// ... with weights
          if (std::fabs(D_A1_Sig(i)) < 0.00000000000000001){
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": ERROR: D_A1_Sig = " << D_A1_Sig << endl;
            string message("CFits::LinFitBevington: I_Run=");
            message += to_string(I_Run) + ": i = " + to_string(i) + ": ERROR: D_A1_Sig(" + to_string(i) + ") == 0.";
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          D_A1_WT(i) = 1. / pow(D_A1_Sig(i), 2);
        }
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ":: D_A1_WT set to " << D_A1_WT << endl;
        #endif
        for (i=0; i < I_MaskSum; i++)
        {
          D_Sum_Weights += D_A1_WT(i);
          D_Sum_XTimesWeight += D_A1_SF(i) * D_A1_WT(i);
          D_Sum_YTimesWeight += D_A1_CCD(i) * D_A1_WT(i);
          D_Sum_XYTimesWeight += D_A1_SF(i) * D_A1_CCD(i) * D_A1_WT(i);
          D_Sum_XSquareTimesWeight += D_A1_SF(i) * D_A1_SF(i) * D_A1_WT(i);
        }
      }
      else
      {
        for (i = 0; i < I_MaskSum; i++)
        {
          /// ... or without weights
          D_Sum_XTimesWeight += D_A1_SF(i);
          D_Sum_YTimesWeight += D_A1_CCD(i);
          D_Sum_XYTimesWeight += D_A1_SF(i) * D_A1_CCD(i);
          D_Sum_XSquareTimesWeight += D_A1_SF(i) * D_A1_SF(i);
        }
        D_Sum_Weights = I_MaskSum;
      }
      D_Delta = D_Sum_Weights * D_Sum_XSquareTimesWeight - pow(D_Sum_XTimesWeight, 2);

      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_Weights set to " << D_Sum_Weights << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XTimesWeight set to " << D_Sum_XTimesWeight << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_YTimesWeight set to " << D_Sum_YTimesWeight << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XYTimesWeight set to " << D_Sum_XYTimesWeight << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XSquareTimesWeight set to " << D_Sum_XSquareTimesWeight << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Delta set to " << D_Delta << endl;
      #endif


      if (!B_WithSky)
      {
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out < 0. = setting D_Sky_Out to 0 " << endl;
        #endif
        D_SP_Out = D_Sum_XYTimesWeight / D_Sum_XSquareTimesWeight;
        D_Sky_Out = 0.0;
      }
      else
      {
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out >= 0." << D_Sky_Out << endl;
        #endif
        D_Sky_Out = ((D_Sum_XSquareTimesWeight * D_Sum_YTimesWeight) - (D_Sum_XTimesWeight * D_Sum_XYTimesWeight)) / D_Delta;

        D_SP_Out = ((D_Sum_Weights * D_Sum_XYTimesWeight) - (D_Sum_XTimesWeight * D_Sum_YTimesWeight)) / D_Delta;
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_SP_Out set to " << D_SP_Out << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out set to " << D_Sky_Out << endl;
        #endif
      }
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_Weights >= " << D_Sum_Weights << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XSquareTimesWeight >= " << D_Sum_XSquareTimesWeight << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Delta >= " << D_Delta << endl;
      #endif
      (*P_D_A1_Sigma_Out)(0) = sqrt(D_Sum_Weights / D_Delta);
      (*P_D_A1_Sigma_Out)(1) = sqrt(D_Sum_XSquareTimesWeight / D_Delta);
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": P_D_A1_Sigma_Out(0) set to " << (*P_D_A1_Sigma_Out)(0) << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": P_D_A1_Sigma_Out(1) set to " << (*P_D_A1_Sigma_Out)(1) << endl;
      #endif
      if ((!B_AllowSpecLTZero) && (D_SP_Out < 0.))
        D_SP_Out = 0.;

      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out set to " << D_Sky_Out << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_SP_Out set to " << D_SP_Out << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": std::fabs(D_SP_Out) = " << std::fabs(D_SP_Out) << endl;
      #endif

      *P_D_A1_YFit = D_Sky_Out + D_SP_Out * D_A1_SF_In.template cast<ImageT>();
      D_A1_YFit = D_Sky_Out + D_SP_Out * D_A1_SF.template cast<ImageT>();
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": *P_D_A1_YFit set to " << *P_D_A1_YFit << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_YFit set to " << D_A1_YFit << endl;
      #endif
      *P_D_ChiSqr_Out = 0.;
      if (I_KeywordSet_MeasureErrors < 0)
      {
        for (i = 0; i < I_MaskSum; i++)
        {
          *P_D_ChiSqr_Out += pow(D_A1_CCD(i) - D_A1_YFit(i), 2);
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": P_D_ChiSqr_Out set to " << *P_D_ChiSqr_Out << endl;
          #endif
        }

        /// for unweighted data evaluate typical sig using chi2, and adjust the standard deviations
        if (I_MaskSum == 2){
          string message("CFits::LinFitBevington: I_Run=");
          message += to_string(I_Run) + ": ERROR: Sum of Mask (=" + to_string(I_MaskSum) + ") must be greater than 2";
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        sigdat = sqrt((*P_D_ChiSqr_Out) / (I_MaskSum - 2));
        (*P_D_A1_Sigma_Out)(0) *= sigdat;
        (*P_D_A1_Sigma_Out)(1) *= sigdat;
      }
      else
      {
        for (i = 0; i < I_MaskSum; i++)
        {
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_CCD(" << i << ") = " << D_A1_CCD(i) << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_SF(" << i << ") = " << D_A1_SF(i) << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_Sig(" << i << ") = " << D_A1_Sig(i) << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_YFit(" << i << ") = " << D_A1_YFit(i) << endl;
          #endif
          if (abs(D_A1_Sig(i)) < 0.00000000000000001){
            string message("CFits::LinFitBevington: I_Run=");
            message += to_string(I_Run) + ": i = " + to_string(i) + ": ERROR: D_A1_Sig(" + to_string(i) + ") == 0.";
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          *P_D_ChiSqr_Out += pow((D_A1_CCD(i) - D_A1_YFit(i)) / D_A1_Sig(i), 2);
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": P_D_ChiSqr_Out set to " << *P_D_ChiSqr_Out << endl;
          #endif
        }
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": P_D_ChiSqr_Out set to " << *P_D_ChiSqr_Out << endl;
        #endif
        if (I_MaskSum > 2)
          *P_D_Q_Out = pfs::drp::stella::math::GammQ(0.5 * (I_MaskSum - 2), 0.5 * (*P_D_ChiSqr_Out));
      }
      if (std::fabs(D_SP_Out) < 0.000001)
        B_Reject = false;
      if (!B_Reject)
        B_Run = false;
      else{

        I_SumMaskLast = P_I_A1_Mask->sum();
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: I_SumMaskLast = " << I_SumMaskLast << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_CCD = " << D_A1_CCD << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_YFit = " << D_A1_YFit << endl;
        #endif
        Eigen::Array<ImageT, Eigen::Dynamic, 1> tempArr(D_A1_CCD.size());
        tempArr = Eigen::pow(D_A1_CCD - D_A1_YFit, 2);
        D_SDevReject = sqrt(tempArr.sum() / ImageT(I_SumMaskLast));//(blitz::sum(pow(D_A1_CCD - (D_A1_YFit),2)) / I_SumMaskLast);

        /// NOTE: Should be square! Test!!!
        D_A1_Diff = D_A1_CCD_In - (*P_D_A1_YFit);
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_SDevReject = " << D_SDevReject << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_CCD_In = " << D_A1_CCD_In << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: *P_D_A1_YFit = " << *P_D_A1_YFit << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_CCD_In - (*P_D_A1_YFit) = " << D_A1_Diff << endl;
        #endif
        D_A1_Check = abs(D_A1_Diff);
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_Check = " << D_A1_Check << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": before Reject: *P_I_A1_Mask = " << *P_I_A1_Mask << endl;
        #endif
        I_A1_LastMask = *P_I_A1_Mask;
        for (size_t pos = 0; pos < D_A1_Check.size(); ++pos){
          (*P_I_A1_Mask)(pos) = (D_A1_Check(pos) > (D_Reject * D_SDevReject)) ? 0 : 1;
          if (I_A1_Mask_Orig(pos) < 1)
            (*P_I_A1_Mask)(pos) = 0;
        }
        if (P_I_A1_Mask->sum() == I_A1_Mask_Orig.sum())
          B_Reject = false;
        else{
          for (size_t pos = 0; pos < P_I_A1_Mask->size(); ++pos)
            if (I_A1_LastMask(pos) < 1)
              (*P_I_A1_Mask)(pos) = 0;
        }
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: *P_I_A1_Mask = " << *P_I_A1_Mask << endl;
        #endif
        if (I_SumMaskLast == P_I_A1_Mask->sum()){
          B_Run = false;
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": leaving while loop" << endl;
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
      cout << "CFits::LinFitBevington: *P_D_A1_YFit set to " << *P_D_A1_YFit << endl;
      cout << "CFits::LinFitBevington: *P_I_A1_Mask set to " << *P_I_A1_Mask << endl;
    #endif
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::LinFitBevingtonEigen(D_A1_CCD, D_A1_SF, SP, Sky, withSky, Args, ArgV) finished" << endl;
    #endif
    return status;
  }

  template<typename ImageT, typename SlitFuncT>
  SpectrumBackground< ImageT > LinFitBevingtonNdArray( ndarray::Array<ImageT, 1, 1> const& D_A1_CCD_In,
                                                       ndarray::Array<SlitFuncT, 1, 1> const& D_A1_SF_In,
                                                       bool B_WithSky )
  {
    SpectrumBackground< ImageT > out;
    std::vector< string > args(1, " ");
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
                             std::vector<string> const& S_A1_Args_In,
                             std::vector<void *> & ArgV_In)
  /// MEASURE_ERRORS_IN = blitz::Array<double,1>(D_A1_CCD_In.size)
  /// REJECT_IN = double
  /// MASK_INOUT = blitz::Array<double,1>(D_A1_CCD_In.size)
  /// CHISQ_OUT = double
  /// Q_OUT = double
  /// SIGMA_OUT = blitz::Array<double,1>(2): [0]: sigma_sp, [1]: sigma_sky
  /// YFIT_OUT = blitz::Array<double, 1>(D_A1_CCD_In.size)
  /// ALLOW_SKY_LT_ZERO = 1
  /// ALLOW_SPEC_LT_ZERO = 1
  {
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::LinFitBevingtonNdArray(D_A1_CCD, D_A1_SF, SP, Sky, withSky, Args, ArgV) started" << endl;
    #endif
    int status = 1;
    #ifdef __DEBUG_FIT__
      cout << "CFits::LinFitBevington(Array, Array, double, double, bool, CSArr, PPArr) started" << endl;
      cout << "CFits::LinFitBevington: D_A1_CCD_In = " << D_A1_CCD_In << endl;
      cout << "CFits::LinFitBevington: D_A1_SF_In = " << D_A1_SF_In << endl;
    #endif

    if (D_A1_CCD_In.size() != D_A1_SF_In.size()){
      string message("CFits::LinFitBevington: ERROR: D_A1_CCD_In.size(=");
      message += to_string(D_A1_CCD_In.size()) + ") != D_A1_SF_In.size(=" + to_string(D_A1_SF_In.size()) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    //  /// Set D_A1_SF_In to zero where D_A1_CCD_In == zero
    ndarray::Array<SlitFuncT, 1, 1> D_A1_SF = ndarray::allocate(D_A1_SF_In.getShape()[0]);
    D_A1_SF.deep() = D_A1_SF_In;
    ndarray::Array<ImageT, 1, 1> D_A1_CCD = ndarray::allocate(D_A1_CCD_In.getShape()[0]);
    D_A1_CCD.deep() = D_A1_CCD_In;

    if ((D_A1_CCD_In.asEigen().sum() == 0.) || (D_A1_SF.asEigen().sum() == 0.)){
      #ifdef __WARNINGS_ON__
        cout << "CFits::LinFitBevington: Warning: (D_A1_CCD_In.sum(=" << D_A1_CCD_In.asEigen().sum() << " == 0.) || (D_A1_SF.sum(=" << D_A1_SF.asEigen().sum() << ") == 0.) => returning false" << endl;
      #endif
      D_SP_Out = 0.;
      D_Sky_Out = 0.;
      status = 0;
      return status;
    }
    int i, I_Pos;
    int I_KeywordSet_Reject, I_KeywordSet_Mask, I_KeywordSet_MeasureErrors, I_KeywordSet_SigmaOut, I_KeywordSet_ChiSqOut, I_KeywordSet_QOut, I_KeywordSet_YFitOut, I_KeywordSet_AllowSkyLTZero, I_KeywordSet_AllowSpecLTZero;
    double sigdat;
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
        cout << "CFits::LinFitBevington: KeyWord_Set(ALLOW_SKY_LT_ZERO)" << endl;
      }
    }

    bool B_AllowSpecLTZero = false;
    I_KeywordSet_AllowSpecLTZero = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "ALLOW_SPEC_LT_ZERO");
    if (I_KeywordSet_AllowSpecLTZero >= 0){
      if (I_KeywordSet_AllowSkyLTZero < 0){
        if (*((int*)ArgV_In[I_KeywordSet_AllowSkyLTZero]) > 0){
          B_AllowSpecLTZero = true;
          cout << "CFits::LinFitBevington: KeyWord_Set(ALLOW_SPEC_LT_ZERO)" << endl;
        }
      }
    }

    float D_Reject(-1.);
    I_KeywordSet_Reject = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "REJECT_IN");
    if (I_KeywordSet_Reject >= 0)
    {
      D_Reject = *(float*)ArgV_In[I_KeywordSet_Reject];
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: KeyWord_Set(REJECT_IN): D_Reject = " << D_Reject << endl;
      #endif
    }
    bool B_Reject = false;
    if (D_Reject > 0.)
      B_Reject = true;

    ndarray::Array<unsigned short, 1, 1> I_A1_Mask_Orig = ndarray::allocate(ndata);
    ndarray::Array<unsigned short, 1, 1> I_A1_Mask = ndarray::allocate(ndata);
    I_A1_Mask.deep() = 1;
    PTR(ndarray::Array<unsigned short, 1, 1>) P_I_A1_Mask(new ndarray::Array<unsigned short, 1, 1>(I_A1_Mask));
    I_KeywordSet_Mask = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MASK_INOUT");
    if (I_KeywordSet_Mask >= 0)
    {
      P_I_A1_Mask.reset();
      P_I_A1_Mask = *((PTR(ndarray::Array<unsigned short, 1, 1>)*)ArgV_In[I_KeywordSet_Mask]);
      if (P_I_A1_Mask->getShape()[0] != ndata){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: P_I_A1_Mask->getShape()[0](=");
        message += to_string(P_I_A1_Mask->getShape()[0]) + ") != ndata(=" + to_string(ndata) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: KeyWord_Set(MASK_INOUT): *P_I_A1_Mask = " << *P_I_A1_Mask << endl;
      #endif
    }
    I_A1_Mask_Orig.deep() = *P_I_A1_Mask;
    #ifdef __DEBUG_FIT__
      cout << "CFits::LinFitBevington: *P_I_A1_Mask set to " << *P_I_A1_Mask << endl;
      cout << "CFits::LinFitBevington: I_A1_Mask_Orig set to " << I_A1_Mask_Orig << endl;
    #endif

    ndarray::Array<ImageT, 1, 1> D_A1_Sigma_Out = ndarray::allocate(2);
    PTR(ndarray::Array<ImageT, 1, 1>) P_D_A1_Sigma_Out(new ndarray::Array<ImageT, 1, 1>(D_A1_Sigma_Out));
    I_KeywordSet_SigmaOut = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "SIGMA_OUT");
    if (I_KeywordSet_SigmaOut >= 0)
    {
      P_D_A1_Sigma_Out.reset();
      P_D_A1_Sigma_Out = *(PTR(ndarray::Array<ImageT, 1, 1>)*)ArgV_In[I_KeywordSet_SigmaOut];
      if (P_D_A1_Sigma_Out->getShape()[0] != 2){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: P_D_A1_Sigma_Out->getShape()[0](=");
        message += to_string(P_D_A1_Sigma_Out->getShape()[0]) + ") != 2";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: KeyWord_Set(SIGMA_OUT)" << endl;
      #endif
    }
    P_D_A1_Sigma_Out->deep() = 0.;
    #ifdef __DEBUG_FIT__
      cout << "CFits::LinFitBevington: *P_D_A1_Sigma_Out set to " << *P_D_A1_Sigma_Out << endl;
    #endif

    PTR(ImageT) P_D_ChiSqr_Out(new ImageT(0.));
    I_KeywordSet_ChiSqOut = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "CHISQ_OUT");
    if (I_KeywordSet_ChiSqOut >= 0)
    {
      P_D_ChiSqr_Out.reset();
      P_D_ChiSqr_Out = *(PTR(ImageT)*)ArgV_In[I_KeywordSet_ChiSqOut];
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: KeyWord_Set(CHISQ_OUT)" << endl;
      #endif
    }
    *P_D_ChiSqr_Out = 0.;
    #ifdef __DEBUG_FIT__
      cout << "CFits::LinFitBevington: *P_D_ChiSqr_Out set to " << *P_D_ChiSqr_Out << endl;
    #endif

    PTR(ImageT) P_D_Q_Out(new ImageT(0.));
    I_KeywordSet_QOut = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "Q_OUT");
    if (I_KeywordSet_QOut >= 0)
    {
      P_D_Q_Out.reset();
      P_D_Q_Out = *(PTR(ImageT)*)ArgV_In[I_KeywordSet_QOut];
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: KeyWord_Set(Q_OUT)" << endl;
      #endif
    }
    *P_D_Q_Out = 1.;
    #ifdef __DEBUG_FIT__
      cout << "CFits::LinFitBevington: *P_D_Q_Out set to " << *P_D_Q_Out << endl;
    #endif

    D_SP_Out = 0.0;
    I_KeywordSet_MeasureErrors = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "MEASURE_ERRORS_IN");
    if (I_KeywordSet_MeasureErrors >= 0)
    {
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: keyword MEASURE_ERRORS_IN set" << endl;
      #endif
      P_D_A1_Sig.reset();
      P_D_A1_Sig = *(PTR(ndarray::Array<ImageT, 1, 1>)*)ArgV_In[I_KeywordSet_MeasureErrors];
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: *P_D_A1_Sig = " << *P_D_A1_Sig << endl;
      #endif
      if (P_D_A1_Sig->getShape()[0] != ndata){
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: P_D_A1_Sig->size()(=");
        message += to_string(P_D_A1_Sig->size()) + ") != ndata(=" + to_string(ndata) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      D_A1_Sig.deep() = *P_D_A1_Sig;
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: KeyWord_Set(MEASURE_ERRORS_IN): *P_D_A1_Sig = " << *P_D_A1_Sig << endl;
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
        string message("pfs::drp::stella::math::CurveFitting::LinFitBevingtonNdArray: ERROR: P_D_A1_YFit->size()(=");
        message += to_string(P_D_A1_YFit->size()) + ") != ndata(=" + to_string(ndata) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
    }
    P_D_A1_YFit->deep() = 0.;
    #ifdef __DEBUG_FIT__
      cout << "CFits::LinFitBevington: *P_D_A1_YFit set to " << *P_D_A1_YFit << endl;
    #endif

    if (P_I_A1_Mask->asEigen().sum() == 0){
      #ifdef __WARNINGS_ON__
        cout << "CFits::LinFitBevington: WARNING: P_I_A1_Mask->sum() == 0" << endl;
      #endif
      D_SP_Out = 0.;
      D_Sky_Out = 0.;
      status = 0;
      #ifdef __DEBUG_CURVEFIT__
        cout << "CurveFitting::LinFitBevingtonNdArray(D_A1_CCD, D_A1_SF, SP, Sky, withSky, Args, ArgV) finished, status = " << status << endl;
      #endif
      return status;
    }

    int I_SumMaskLast;
    ImageT D_SDevReject;
    ndarray::Array<ImageT, 1, 1> D_A1_Check = ndarray::allocate(ndata);
    ndarray::Array<unsigned short, 1, 1> I_A1_LastMask = ndarray::allocate(P_I_A1_Mask->getShape()[0]);
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
      cout << "CFits::LinFitBevington: starting while loop" << endl;
    #endif
    while (B_Run){
      D_SP_Out = 0.0;

      I_Run++;
      /// remove bad pixels marked by mask
      I_MaskSum = P_I_A1_Mask->asEigen().sum();
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": I_MaskSum = " << I_MaskSum << endl;
      #endif
      if (I_MaskSum == 0){
        string message("LinFitBevington: WARNING: I_MaskSum == 0");
        cout << message << endl;
        status = 0;
        return status;
//          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
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
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_CCD set to " << D_A1_CCD << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_SF set to " << D_A1_SF << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_Sig set to " << D_A1_Sig << endl;
      #endif

      D_Sum_Weights = 0.;
      D_Sum_XSquareTimesWeight = 0.;
      D_Sum_XTimesWeight = 0.;
      D_Sum_XYTimesWeight = 0.;
      D_Sum_YTimesWeight = 0.;
      if (I_KeywordSet_MeasureErrors >= 0)
      {
        ///    D_A1_WT = D_A1_SF;
        for (i=0; i < I_MaskSum; i++)
        {
          /// ... with weights
          if (std::fabs(D_A1_Sig[i]) < 0.00000000000000001){
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": ERROR: D_A1_Sig = " << D_A1_Sig << endl;
            string message("CFits::LinFitBevington: I_Run=");
            message += to_string(I_Run) + ": i = " + to_string(i) + ": ERROR: D_A1_Sig(" + to_string(i) + ") == 0.";
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          D_A1_WT[i] = 1. / pow(D_A1_Sig[i], 2);
        }
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ":: D_A1_WT set to " << D_A1_WT << endl;
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
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_Weights set to " << D_Sum_Weights << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XTimesWeight set to " << D_Sum_XTimesWeight << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_YTimesWeight set to " << D_Sum_YTimesWeight << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XYTimesWeight set to " << D_Sum_XYTimesWeight << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XSquareTimesWeight set to " << D_Sum_XSquareTimesWeight << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Delta set to " << D_Delta << endl;
      #endif


      if (!B_WithSky)
      {
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out < 0. = setting D_Sky_Out to 0 " << endl;
        #endif
        D_SP_Out = D_Sum_XYTimesWeight / D_Sum_XSquareTimesWeight;
        D_Sky_Out = 0.0;
      }
      else
      {
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out >= 0." << D_Sky_Out << endl;
        #endif
        D_Sky_Out = ((D_Sum_XSquareTimesWeight * D_Sum_YTimesWeight) - (D_Sum_XTimesWeight * D_Sum_XYTimesWeight)) / D_Delta;

        D_SP_Out = ((D_Sum_Weights * D_Sum_XYTimesWeight) - (D_Sum_XTimesWeight * D_Sum_YTimesWeight)) / D_Delta;
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_SP_Out set to " << D_SP_Out << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out set to " << D_Sky_Out << endl;
        #endif
      }
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_Weights >= " << D_Sum_Weights << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sum_XSquareTimesWeight >= " << D_Sum_XSquareTimesWeight << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Delta >= " << D_Delta << endl;
      #endif
      (*P_D_A1_Sigma_Out)[0] = sqrt(D_Sum_Weights / D_Delta);
      (*P_D_A1_Sigma_Out)[1] = sqrt(D_Sum_XSquareTimesWeight / D_Delta);
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": P_D_A1_Sigma_Out(0) set to " << (*P_D_A1_Sigma_Out)[0] << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": P_D_A1_Sigma_Out(1) set to " << (*P_D_A1_Sigma_Out)[1] << endl;
      #endif
      if ((!B_AllowSpecLTZero) && (D_SP_Out < 0.))
        D_SP_Out = 0.;

      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_Sky_Out set to " << D_Sky_Out << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_SP_Out set to " << D_SP_Out << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": std::fabs(D_SP_Out) = " << std::fabs(D_SP_Out) << endl;
      #endif

      P_D_A1_YFit->deep() = D_Sky_Out + D_SP_Out * D_A1_SF_In;//.template cast<ImageT>();
      D_A1_YFit.deep() = D_Sky_Out + D_SP_Out * D_A1_SF;//.template cast<ImageT>();
      #ifdef __DEBUG_FIT__
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": *P_D_A1_YFit set to " << *P_D_A1_YFit << endl;
        cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": D_A1_YFit set to " << D_A1_YFit << endl;
      #endif
      *P_D_ChiSqr_Out = 0.;
      if (I_KeywordSet_MeasureErrors < 0)
      {
        for (i = 0; i < I_MaskSum; i++)
        {
          *P_D_ChiSqr_Out += pow(D_A1_CCD[i] - D_A1_YFit[i], 2);
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": P_D_ChiSqr_Out set to " << *P_D_ChiSqr_Out << endl;
          #endif
        }

        /// for unweighted data evaluate typical sig using chi2, and adjust the standard deviations
        if (I_MaskSum == 2){
          string message("CFits::LinFitBevington: I_Run=");
          message += to_string(I_Run) + ": ERROR: Sum of Mask (=" + to_string(I_MaskSum) + ") must be greater than 2";
          cout << message << endl;
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
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_CCD(" << i << ") = " << D_A1_CCD[i] << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_SF(" << i << ") = " << D_A1_SF[i] << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_Sig(" << i << ") = " << D_A1_Sig[i] << endl;
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": D_A1_YFit(" << i << ") = " << D_A1_YFit[i] << endl;
          #endif
          if (abs(D_A1_Sig[i]) < 0.00000000000000001){
            string message("CFits::LinFitBevington: I_Run=");
            message += to_string(I_Run) + ": i = " + to_string(i) + ": ERROR: D_A1_Sig(" + to_string(i) + ") == 0.";
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
          *P_D_ChiSqr_Out += pow((D_A1_CCD[i] - D_A1_YFit[i]) / D_A1_Sig[i], 2);
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": i = " << i << ": P_D_ChiSqr_Out set to " << *P_D_ChiSqr_Out << endl;
          #endif
        }
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": P_D_ChiSqr_Out set to " << *P_D_ChiSqr_Out << endl;
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
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: I_SumMaskLast = " << I_SumMaskLast << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_CCD = " << D_A1_CCD << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_YFit = " << D_A1_YFit << endl;
        #endif
        ndarray::Array<ImageT, 1, 1> tempArr = ndarray::allocate(D_A1_CCD.getShape()[0]);
        tempArr.deep() = D_A1_CCD - D_A1_YFit;
        Eigen::Array<ImageT, Eigen::Dynamic, 1> tempEArr = tempArr.asEigen();
        tempArr.asEigen() = tempEArr.pow(2);
        D_SDevReject = sqrt(tempArr.asEigen().sum() / ImageT(I_SumMaskLast));

        D_A1_Diff.deep() = D_A1_CCD_In - (*P_D_A1_YFit);
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_SDevReject = " << D_SDevReject << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_CCD_In = " << D_A1_CCD_In << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: *P_D_A1_YFit = " << *P_D_A1_YFit << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_CCD_In - (*P_D_A1_YFit) = " << D_A1_Diff << endl;
        #endif
        tempEArr = D_A1_Diff.asEigen();
        D_A1_Check.asEigen() = tempEArr.abs();
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: D_A1_Check = " << D_A1_Check << endl;
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": before Reject: *P_I_A1_Mask = " << *P_I_A1_Mask << endl;
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
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": Reject: *P_I_A1_Mask = " << *P_I_A1_Mask << endl;
        #endif
        if (I_SumMaskLast == P_I_A1_Mask->asEigen().sum()){
          B_Run = false;
          #ifdef __DEBUG_FIT__
            cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": leaving while loop" << endl;
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
      cout << "CFits::LinFitBevington: *P_D_A1_YFit set to " << *P_D_A1_YFit << endl;
      cout << "CFits::LinFitBevington: *P_I_A1_Mask set to " << *P_I_A1_Mask << endl;
    #endif
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::LinFitBevingtonNdArray(D_A1_CCD, D_A1_SF, SP, Sky, withSky, Args, ArgV) finished" << endl;
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
      cout << "CurveFitting::GSER(Gamser, a, x) started" << endl;
    #endif
    T D_GLn_Out = 0;
    int n;
    int ITMax = 100;
    double d_sum, del, ap;

    #ifdef __DEBUG_LINFIT__
      cout << "CFits::GSER: D_Gamser_Out = " << D_Gamser_Out << endl;
      cout << "CFits::GSER: a = " << a << endl;
      cout << "CFits::GSER: x = " << x << endl;
    #endif

    D_GLn_Out = GammLn(a);
    #ifdef __DEBUG_LINFIT__
      cout << "CFits::GSER: D_GLn_Out = " << D_GLn_Out << endl;
    #endif
    if (x <= 0.){
      if (x < 0.){
        string message("CFits::GSER: ERROR: x less than 0!");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      D_Gamser_Out = 0.;
      #ifdef __DEBUG_LINFIT__
        cout << "CFits::GSER: x<=0: D_Gamser_Out = " << D_Gamser_Out << endl;
        cout << "CFits::GSER: x<=0: D_GLn_Out = " << D_GLn_Out << endl;
      #endif
      #ifdef __DEBUG_CURVEFIT__
        cout << "CurveFitting::GSER(Gamser, a, x) finished" << endl;
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
            cout << "CFits::GSER: x>0: D_Gamser_Out = " << D_Gamser_Out << endl;
            cout << "CFits::GSER: x>0: D_GLn_Out = " << D_GLn_Out << endl;
          #endif
          #ifdef __DEBUG_CURVEFIT__
            cout << "CurveFitting::GSER(Gamser, a, x) finished" << endl;
          #endif
          return D_GLn_Out;
        }
      }
      string message("CFits::GSER: ERROR: a too large, ITMax too small in routine GSER");
      cout << message << endl;
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
      cout << "CurveFitting::GammLn(xx) started" << endl;
    #endif
    double x,y,tmp,ser;
    static double cof[6]={76.18009172947146, -86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5};

    #ifdef __DEBUG_LINFIT__
      cout << "CFits::GammLn: xx = " << xx << endl;
    #endif

    y = x = xx;
    tmp = x + 5.5;
    tmp -= (x+0.5) * log(tmp);
    #ifdef __DEBUG_LINFIT__
      cout << "CFits::GammLn: tmp = " << tmp << endl;
    #endif
    ser = 1.000000000190015;
    for (int o = 0; o <= 5; o++){
      ser += cof[o] / ++y;
    }
    T D_Result = T(-tmp + log(2.5066282746310005 * ser / x));
    #ifdef __DEBUG_LINFIT__
      cout << "CFits::GammLn: ser = " << ser << endl;
      cout << "CFits::GammLn: returning (-tmp + log(2.5066282746310005 * ser / xx)) = " << D_Result << endl;
    #endif
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::GammLn(xx) finished" << endl;
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
      cout << "CurveFitting::GCF(GammCF, a, x) Started" << endl;
    #endif
    T D_GLn_Out = 0;
    int n;
    int ITMAX = 100;             /// Maximum allowed number of iterations
    T an, b, c, d, del, h;
    double FPMIN = 1.0e-30;      /// Number near the smallest representable floating-point number
    double EPS = 1.0e-7;         /// Relative accuracy

    D_GLn_Out = GammLn(a);
    #ifdef __DEBUG_FIT__
      cout << "CFits::GCF: D_GLn_Out set to " << D_GLn_Out << endl;
    #endif

    b = x + 1. - a;
    #ifdef __DEBUG_FIT__
      cout << "CFits::GCF: x=" << x << ", a=" << a << ": b set to " << b << endl;
    #endif
    c = 1. / FPMIN;
    #ifdef __DEBUG_FIT__
      cout << "CFits::GCF: c set to " << c << endl;
    #endif
    d = 1. / b;
    #ifdef __DEBUG_FIT__
      cout << "CFits::GCF: d set to " << d << endl;
    #endif
    h = d;
    for (n=1; n <= ITMAX; n++){
      an = -n * (n - a);
      #ifdef __DEBUG_FIT__
        cout << "CFits::GCF: n = " << n << ": an set to " << an << endl;
      #endif
      b += 2.;
      #ifdef __DEBUG_FIT__
        cout << "CFits::GCF: n = " << n << ": b set to " << b << endl;
      #endif
      d = an * d + b;
      #ifdef __DEBUG_FIT__
        cout << "CFits::GCF: n = " << n << ": d set to " << d << endl;
      #endif
      if (std::fabs(d) < FPMIN)
        d = FPMIN;
      c = b + an / c;
      #ifdef __DEBUG_FIT__
        cout << "CFits::GCF: n = " << n << ": c set to " << c << endl;
      #endif
      if (std::fabs(c) < FPMIN)
        c = FPMIN;
      d = 1. / d;
      #ifdef __DEBUG_FIT__
        cout << "CFits::GCF: n = " << n << ": d set to " << d << endl;
      #endif
      del = d * c;
      #ifdef __DEBUG_FIT__
        cout << "CFits::GCF: n = " << n << ": del set to " << del << endl;
      #endif

      h *= del;
      if (std::fabs(del-1.) < EPS)
        break;
    }
    if (n > ITMAX){
      string message("CFits::GCF: ERROR: a too large, ITMAX too small in GCF");
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    D_GammCF_Out = exp(-x+a*log(x) - D_GLn_Out) * h;
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::GCF(GammCF, a, x) finished" << endl;
    #endif
    return D_GLn_Out;
  }

  /**
   * Function to calculate incomplete Gamma Function P(a,x)
   **/
  template<typename T>
  T GammP(T const a, T const x){
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::GammP(a, x) Started" << endl;
    #endif
    T D_Out = 0;
    #ifdef __DEBUG_FIT__
      cout << "CFits::GammP started: a = " << a << ", x = " << x << endl;
    #endif
    T gamser, gammcf;
    if (x < 0.){
      string message("pfs::drp::stella::math::CurveFitting::GammP: ERROR: x(=");
      message += to_string(x) + ") < 0.";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (a <= 0.){
      string message("pfs::drp::stella::math::CurveFitting::GammP: ERROR: a(=");
      message += to_string(a) + ") <= 0.";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (x < (a+1.)){
      GSER(gamser, a, x);
      D_Out = gamser;
      #ifdef __DEBUG_CURVEFIT__
        cout << "CurveFitting::GammP(a, x) finished" << endl;
      #endif
      return D_Out;
    }
    else{
      GCF(gammcf, a, x);
      D_Out = 1. - gammcf;
      #ifdef __DEBUG_CURVEFIT__
        cout << "CurveFitting::GammP(a, x) finished" << endl;
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
      cout << "CurveFitting::GammQ(a, x) started" << endl;
    #endif
    T D_Out = 0;
    #ifdef __DEBUG_FIT__
      cout << "CFits::GammQ started: a = " << a << ", x = " << x << endl;
    #endif
    T gamser = 0.;
    T gammcf = 0.;
    if (x < 0.){
      string message("pfs::drp::stella::math::CurveFitting::GammQ: ERROR: x(=");
      message += to_string(x) + ") < 0.";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if(a <= 0.){
      string message("pfs::drp::stella::math::CurveFitting::GammQ: ERROR: a(=");
      message += to_string(a) + ") < 0.";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (x < (a+1.)){
      GSER(gamser, a, x);
      #ifdef __DEBUG_FIT__
        cout << "CFits::GammQ: x < (a+1.): gamser = " << gamser << endl;
      #endif
      D_Out = 1. - gamser;
      #ifdef __DEBUG_CURVEFIT__
        cout << "CurveFitting::GammQ(a, x) finished" << endl;
      #endif
      return D_Out;
    }
    else{
      GCF(gammcf, a, x);
      #ifdef __DEBUG_FIT__
        cout << "CFits::GammQ: x < (a+1.): gammcf = " << gammcf << endl;
      #endif
      D_Out = gammcf;
      #ifdef __DEBUG_CURVEFIT__
        cout << "CurveFitting::GammQ(a, x) finished" << endl;
      #endif
      return D_Out;
    }
  }

  template< typename T, typename U >
  ndarray::Array<T, 1, 1> chebyshev(ndarray::Array<T, 1, 1> const& x_In, ndarray::Array<U, 1, 1> const& coeffs_In){
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::chebyshev(a, coeffs) started" << endl;
    #endif
    int nCoeffs = coeffs_In.getShape()[0];
    cout << "pfs::drp::stella::math::CurveFitting::chebyshev: coeffs_In = " << nCoeffs << ": ";
    for (int i = 0; i < nCoeffs; ++i)
      printf("%.12f ", coeffs_In[i]);
    printf("\n");

    ndarray::Array<double, 1, 1> range = ndarray::allocate(2);
    range[0] = min(x_In);
    range[1] = max(x_In);
    cout << "chebyshev: range = " << range << endl;
    ndarray::Array<T, 1, 1> xScaled = pfs::drp::stella::math::convertRangeToUnity(x_In, range);
    cout << "chebyshev: xScaled = " << xScaled << endl;

    ndarray::Array<T, 1, 1> tmpArr = ndarray::allocate(x_In.getShape()[0]);
    ndarray::Array<T, 1, 1> c0Arr = ndarray::allocate(x_In.getShape()[0]);
    ndarray::Array<T, 1, 1> c1Arr = ndarray::allocate(x_In.getShape()[0]);
    ndarray::Array<T, 1, 1> yCalc = ndarray::allocate(x_In.getShape()[0]);
    double c0, c1, tmp;
    if (coeffs_In.getShape()[0] == 1){
        c0 = coeffs_In[0];
        c1 = 0;
    }
    else if (coeffs_In.getShape()[0] == 2){
        c0 = coeffs_In[0];
        c1 = coeffs_In[1];
    }
    else{
      ndarray::Array<double, 1, 1> x2 = ndarray::allocate(xScaled.getShape()[0]);
      x2.deep() = 2. * xScaled;
      c0 = coeffs_In[coeffs_In.getShape()[0] - 2];
      c1 = coeffs_In[coeffs_In.getShape()[0] - 1];
      cout << "chebyshev: c0 = " << c0 << ", c1 = " << c1 << endl;
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
        cout << "chebyshev: i = " << i << ": c0 = " << c0 << ", c0Arr = " << c0Arr << ", c1Arr = " << c1Arr << endl;
      }
    }
    if (coeffs_In.getShape()[0] < 3)
      yCalc.deep() = c0 + c1 * xScaled;
    else if (coeffs_In.getShape()[0] == 3)
      yCalc.deep() = c0 + c1Arr * xScaled;
    else
      yCalc.deep() = c0Arr + c1Arr * xScaled;
    cout << "chebyshev: yCalc = " << yCalc << endl;
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::chebyshev(a, coeffs) finished" << endl;
    #endif
    return yCalc;
    /*
    ndarray::Array<double, 1, 1> Tn = ndarray::allocate(coeffs_In.getShape()[0]);
    Tn[0] = 1;
    for (int i = 0; i < x_In.getShape()[0]; ++i){
      if (nCoeffs > 1)
        Tn[1] = x_In[i];
      for (int j = 2; j < nCoeffs; ++j){
        Tn[j] = 2. * x_In[i] * Tn[j-1] - Tn[j-2];
        cout << "pfs::drp::stella::math::CurveFitting::chebyshev: x_In[" << i << "] = " << x_In[i] << ": Tn[" << j-2 << "] = " << Tn[j-2] << ", Tn[" << j-1 << "] = " << Tn[j-1] << ": Tn[" << j << "] = " << Tn[j] << endl;
      }
      yCalc[i] = coeffs_In[0];
      cout << "pfs::drp::stella::math::CurveFitting::chebyshev: yCalc[" << i << "] = " << yCalc[i] << endl;
      for (int j = 1; j < nCoeffs; ++j){
        yCalc[i] += coeffs_In[j] * Tn[j];
        printf("pfs::drp::stella::math::CurveFitting::chebyshev: Tn[%i] = %.12f, coeffs_In[j] * Tn[j] = %.12f: yCalc[%i] = %.12f\n", j, Tn[j], coeffs_In[j] * Tn[j], i, yCalc[i]);
      }
    }
    return yCalc;
    */
  }

  ndarray::Array<double, 1, 1> gaussFit(ndarray::Array<double, 2, 1> const& xy_In,
                                        ndarray::Array<double, 1, 1> const& guess_In){
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::gaussFit(xy, guess) started" << endl;
    #endif
    gaussian_functor gf(xy_In.asEigen());
//    gaussian_functor<Eigen::Derived> gf(xy_In.asEigen());
    Eigen::VectorXd guess(3);
    guess[0] = guess_In[0];
    guess[1] = guess_In[1];
    guess[2] = guess_In[2];
    Eigen::LevenbergMarquardt<gaussian_functor> solver(gf);
    solver.setXtol(1.0e-6);
    solver.setFtol(1.0e-6);
    solver.minimize(guess);
    ndarray::Array<double, 1, 1> result = ndarray::allocate(guess_In.getShape()[0]);
    result[0] = guess[0];
    result[1] = guess[1];
    result[2] = guess[2];
    #ifdef __DEBUG_CURVEFIT__
      cout << "CurveFitting::gaussFit(xy, guess) finished" << endl;
    #endif
    return result;
  }

  template ndarray::Array<float, 1, 1> chebyshev(ndarray::Array<float, 1, 1> const& x_In, ndarray::Array<float, 1, 1> const& coeffs_In);
  template ndarray::Array<double, 1, 1> chebyshev(ndarray::Array<double, 1, 1> const& x_In, ndarray::Array<float, 1, 1> const& coeffs_In);
  template ndarray::Array<float, 1, 1> chebyshev(ndarray::Array<float, 1, 1> const& x_In, ndarray::Array<double, 1, 1> const& coeffs_In);
  template ndarray::Array<double, 1, 1> chebyshev(ndarray::Array<double, 1, 1> const& x_In, ndarray::Array<double, 1, 1> const& coeffs_In);

  template ndarray::Array<float, 1, 1> Poly(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, float, float);
  template ndarray::Array<float, 1, 1> Poly(ndarray::Array<float, 1, 1> const&, ndarray::Array<float const, 1, 1> const&, float, float);
  template ndarray::Array<float, 1, 1> Poly(ndarray::Array<float, 1, 1> const&, ndarray::Array<double, 1, 1> const&, float, float);
  template ndarray::Array<float, 1, 1> Poly(ndarray::Array<float, 1, 1> const&, ndarray::Array<double const, 1, 1> const&, float, float);
  template ndarray::Array<double, 1, 1> Poly(ndarray::Array<double, 1, 1> const&, ndarray::Array<float, 1, 1> const&, double, double);
  template ndarray::Array<double, 1, 1> Poly(ndarray::Array<double, 1, 1> const&, ndarray::Array<float const, 1, 1> const&, double, double);
  template ndarray::Array<double, 1, 1> Poly(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&, double, double);
  template ndarray::Array<double, 1, 1> Poly(ndarray::Array<double, 1, 1> const&, ndarray::Array<double const, 1, 1> const&, double, double);

//  template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const, float const, std::vector<string> const&, std::vector<void *> &);
  template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&, size_t const, double const, std::vector<string> const&, std::vector<void *> &);
//  template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const, float const, float const, size_t const, std::vector<string> const&, std::vector<void *> &);
  template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&, size_t const, double const, double const, size_t const, std::vector<string> const&, std::vector<void *> &);
//  template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const, std::vector<string> const&, std::vector<void *> &);
  template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&, size_t const, std::vector<string> const&, std::vector<void *> &);
//  template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const, float, float);
  template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&, size_t const, double, double);
//  template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const, float const, float, float);
  template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&, size_t const, double const, double, double);
//  template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const, float const, float const, size_t const, float, float);
  template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&, size_t const, double const, double const, size_t const, double, double);

  template int LinFitBevingtonEigen(Eigen::Array<float, Eigen::Dynamic, 1> const&,
                                          Eigen::Array<float, Eigen::Dynamic, 1> const&,
                                          float &,
                                          float &,
                                          bool,
                                          std::vector<string> const&,
                                          std::vector<void *> &);
  template int LinFitBevingtonEigen(Eigen::Array<double, Eigen::Dynamic, 1> const&,
                                          Eigen::Array<float, Eigen::Dynamic, 1> const&,
                                          double &,
                                          double &,
                                          bool,
                                          std::vector<string> const&,
                                          std::vector<void *> &);
  template int LinFitBevingtonEigen(Eigen::Array<float, Eigen::Dynamic, 1> const&,
                                          Eigen::Array<double, Eigen::Dynamic, 1> const&,
                                          float &,
                                          float &,
                                          bool,
                                          std::vector<string> const&,
                                          std::vector<void *> &);
  template int LinFitBevingtonEigen(Eigen::Array<double, Eigen::Dynamic, 1> const&,
                                          Eigen::Array<double, Eigen::Dynamic, 1> const&,
                                          double &,
                                          double &,
                                          bool,
                                          std::vector<string> const&,
                                          std::vector<void *> &);

  template SpectrumBackground< float > LinFitBevingtonNdArray( ndarray::Array<float, 1, 1> const&,
                                                               ndarray::Array<float, 1, 1> const&,
                                                               bool );
  template SpectrumBackground< double > LinFitBevingtonNdArray( ndarray::Array<double, 1, 1> const&,
                                                                ndarray::Array<float, 1, 1> const&,
                                                                bool );
  template SpectrumBackground< float > LinFitBevingtonNdArray( ndarray::Array<float, 1, 1> const&,
                                                               ndarray::Array<double, 1, 1> const&,
                                                               bool );
  template SpectrumBackground< double > LinFitBevingtonNdArray( ndarray::Array<double, 1, 1> const&,
                                                                ndarray::Array<double, 1, 1> const&,
                                                                bool );

  template int LinFitBevingtonNdArray(ndarray::Array<float, 1, 1> const&,
                                            ndarray::Array<float, 1, 1> const&,
                                            float &,
                                            float &,
                                            bool,
                                            std::vector<string> const&,
                                            std::vector<void *> &);
  template int LinFitBevingtonNdArray(ndarray::Array<double, 1, 1> const&,
                                            ndarray::Array<float, 1, 1> const&,
                                            double &,
                                            double &,
                                            bool,
                                            std::vector<string> const&,
                                            std::vector<void *> &);
  template int LinFitBevingtonNdArray(ndarray::Array<float, 1, 1> const&,
                                            ndarray::Array<double, 1, 1> const&,
                                            float &,
                                            float &,
                                            bool,
                                            std::vector<string> const&,
                                            std::vector<void *> &);
  template int LinFitBevingtonNdArray(ndarray::Array<double, 1, 1> const&,
                                            ndarray::Array<double, 1, 1> const&,
                                            double &,
                                            double &,
                                            bool,
                                            std::vector<string> const&,
                                            std::vector<void *> &);

  template bool LinFitBevingtonEigen(Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> const&,
                                           Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> const&,
                                           Eigen::Array<float, Eigen::Dynamic, 1> &,
                                           Eigen::Array<float, Eigen::Dynamic, 1> &,
                                           bool,
                                           std::vector<string> const&,
                                           std::vector<void *> &);
  template bool LinFitBevingtonEigen(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> &,
                                           const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> &,
                                           Eigen::Array<double, Eigen::Dynamic, 1> &,
                                           Eigen::Array<double, Eigen::Dynamic, 1> &,
                                           bool,
                                           const std::vector<string> &,
                                           std::vector<void *> &);
  template bool LinFitBevingtonEigen(const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> &,
                                           const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> &,
                                           Eigen::Array<float, Eigen::Dynamic, 1> &,
                                           Eigen::Array<float, Eigen::Dynamic, 1> &,
                                           bool,
                                           const std::vector<string> &,
                                           std::vector<void *> &);
  template bool LinFitBevingtonEigen(Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> const&,
                                           Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> const&,
                                           Eigen::Array<double, Eigen::Dynamic, 1> &,
                                           Eigen::Array<double, Eigen::Dynamic, 1> &,
                                           bool,
                                           std::vector<string> const&,
                                           std::vector<void *> &);

  template bool LinFitBevingtonNdArray(ndarray::Array<float, 2, 1> const&,
                                             ndarray::Array<float, 2, 1> const&,
                                             ndarray::Array<float, 1, 1> &,
                                             ndarray::Array<float, 1, 1> &,
                                             bool,
                                             std::vector<string> const&,
                                             std::vector<void *> &);
  template bool LinFitBevingtonNdArray(ndarray::Array<double, 2, 1> const&,
                                             ndarray::Array<float, 2, 1> const&,
                                             ndarray::Array<double, 1, 1> &,
                                             ndarray::Array<double, 1, 1> &,
                                             bool,
                                             const std::vector<string> &,
                                             std::vector<void *> &);
  template bool LinFitBevingtonNdArray(ndarray::Array<float, 2, 1> const&,
                                             ndarray::Array<double, 2, 1> const&,
                                             ndarray::Array<float, 1, 1> &,
                                             ndarray::Array<float, 1, 1> &,
                                             bool,
                                             const std::vector<string> &,
                                             std::vector<void *> &);
  template bool LinFitBevingtonNdArray(ndarray::Array<double, 2, 1> const&,
                                             ndarray::Array<double, 2, 1> const&,
                                             ndarray::Array<double, 1, 1> &,
                                             ndarray::Array<double, 1, 1> &,
                                             bool,
                                             std::vector<string> const&,
                                             std::vector<void *> &);

  template float GammLn(float const D_X_In);
  template double GammLn(double const D_X_In);

  template float GCF(float & D_Gamser_In, float const a, float const x);
  template double GCF(double & D_Gamser_In, double const a, double const x);

  template float GammP(float const a, float const x);
  template double GammP(double const a, double const x);

  template float GammQ(float const a, float const x);
  template double GammQ(double const a, double const x);

  template float GSER(float & D_Gamser_In, float const a, float const x);
  template double GSER(double & D_Gamser_In, double const a, double const x);

}}}}
