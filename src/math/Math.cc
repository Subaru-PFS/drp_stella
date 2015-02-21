#include "pfs/drp/stella/math/Math.h"
  namespace pfs{ namespace drp{ namespace stella{ namespace math{

    template<typename T, typename U>
    ndarray::Array<T, 1, 1> Poly(ndarray::Array<T, 1, 1> const& x_In,
                                 ndarray::Array<U, 1, 1> const& coeffs_In){
      int ii = 0;
      ndarray::Array<T, 1, 1> arr_Out = ndarray::allocate(int(x_In.size()));
      #ifdef __DEBUG_POLY__
        cout << "Poly: x_In = " << x_In << endl;
        cout << "Poly: coeffs_In = " << coeffs_In << endl;
        cout << "Poly: arr_Out set to " << arr_Out << endl;
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
      auto x_In_begin = x_In.begin();
      auto coeffs_In_begin = coeffs_In.begin();
      for (ii = I_PolynomialOrder-1; ii >= 0; ii--){
        for (int i = 0; i < arr_Out.getShape()[0]; ++i)
          *(arr_Out_begin + i) = (*(arr_Out_begin + i)) * (*(x_In_begin + i)) + (*(coeffs_In_begin + ii));
        #ifdef __DEBUG_POLY__
          cout << "Poly: I_PolynomialOrder != 0: for (ii = " << ii << "; ii >= 0; ii--) arr_Out set to " << arr_Out << endl;
        #endif
      }
      return arr_Out;
    }
    
    /**
     * Calculates aperture minimum pixel, central position, and maximum pixel for the trace,
     * and writes result to minCenMax_Out and returns it
     **/
    ndarray::Array<size_t, 2, 2> calcMinCenMax(ndarray::Array<float const, 1, 1> const& xCenters_In,
                                               float const xHigh_In,
                                               float const xLow_In,
                                               int const nPixCutLeft_In,
                                               int const nPixCutRight_In){
      ndarray::Array<size_t, 1, 1> floor = pfs::drp::stella::math::floor(xCenters_In, size_t(0));
      ndarray::Array<size_t, 2, 2> minCenMax_Out = ndarray::allocate(xCenters_In.getShape()[0], 3);
      minCenMax_Out[ndarray::view()()] = 0;

      minCenMax_Out[ndarray::view()(1)] = floor;

      #ifdef __DEBUG_MINCENMAX__
        cout << "CFits::calcMinCenMax: minCenMax_Out(*,1) = " << minCenMax_Out[ndarray::view()(1)] << endl;
      #endif
      ndarray::Array<const float, 1, 1> F_A1_Temp = ndarray::copy(xCenters_In + xLow_In);

      minCenMax_Out[ndarray::view()(0)] = pfs::drp::stella::math::floor(F_A1_Temp, size_t(0));

      #ifdef __DEBUG_MINCENMAX__
        cout << "CFits::calcMinCenMax: minCenMax_Out(*,0) = " << minCenMax_Out[ndarray::view()(0)] << endl;
      #endif
      F_A1_Temp = ndarray::copy(xCenters_In + xHigh_In);

      minCenMax_Out[ndarray::view()(2)] = pfs::drp::stella::math::floor(F_A1_Temp, size_t(0));

      #ifdef __DEBUG_MINCENMAX__
        cout << "CFits::calcMinCenMax: minCenMax_Out(*,2) = " << minCenMax_Out[ndarray::view()(2)] << endl;
      #endif

      ndarray::Array<size_t, 1, 1> I_A1_NPixLeft = ndarray::copy(minCenMax_Out[ndarray::view()(1)] - minCenMax_Out[ndarray::view()(0)]);
      ndarray::Array<size_t, 1, 1> I_A1_NPixRight = ndarray::copy(minCenMax_Out[ndarray::view()(2)] - minCenMax_Out[ndarray::view()(1)]);
      ndarray::Array<size_t, 1, 1> I_A1_I_NPixX = ndarray::copy(minCenMax_Out[ndarray::view()(2)] - minCenMax_Out[ndarray::view()(0)] + 1);

      #ifdef __DEBUG_MINCENMAX__
        cout << "CFits::calcMinCenMax: I_A1_NPixLeft(=" << I_A1_NPixLeft << endl;
        cout << "CFits::calcMinCenMax: I_A1_NPixRight(=" << I_A1_NPixRight << endl;
        cout << "CFits::calcMinCenMax: I_A1_I_NPixX = " << I_A1_I_NPixX << endl;
      #endif

      size_t I_MaxPixLeft = pfs::drp::stella::math::max(I_A1_NPixLeft);
      size_t I_MaxPixRight = pfs::drp::stella::math::max(I_A1_NPixRight);
      size_t I_MinPixLeft = pfs::drp::stella::math::min(I_A1_NPixLeft);
      size_t I_MinPixRight = pfs::drp::stella::math::min(I_A1_NPixRight);

      if (I_MaxPixLeft > I_MinPixLeft)
        minCenMax_Out[ndarray::view()(0)] = minCenMax_Out[ndarray::view()(1)] - I_MaxPixLeft + nPixCutLeft_In;

      if (I_MaxPixRight > I_MinPixRight)
        minCenMax_Out[ndarray::view()(2)] = minCenMax_Out[ndarray::view()(1)] + I_MaxPixRight - nPixCutRight_In;

      #ifdef __DEBUG_MINCENMAX__
        cout << "CFits::calcMinCenMax: minCenMax_Out = " << minCenMax_Out << endl;
      #endif

      return minCenMax_Out;
    }

    /**
     * Fix(double)
     * Returns integer value cut at decimal point. If D_In is negative the integer value greater or equal than D_In is returned.
     **/
    template <typename T>
    int Fix(T D_In){
      return (((D_In < T(0.)) && (T(static_cast<int>(D_In)) < D_In)) ? static_cast<int>(D_In) + 1 : static_cast<int>(D_In));
    }
    
    /**
     * Fix(double)
     * Returns long integer value cut at decimal point (See int Fix(double)).
     **/
    template <typename T>
    long FixL(T D_In){
      return ((D_In < 0.) && (T(static_cast<long>(D_In)) < D_In)) ? static_cast<long>(D_In) + 1 : static_cast<long>(D_In);
    }

    template <typename T>
    int Int(T D_In){
      return static_cast<int>(D_In);
    }

    template <typename T>
    long Long(T D_In){
      return static_cast<long>(D_In);
    }

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

      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: Starting " << endl;
      #endif
      assert(D_A1_X_In.getShape()[0] == D_A1_Y_In.getShape()[0]);

      int I_NReject = 0;
      std::vector<T> D_A1_X(D_A1_X_In.begin(), D_A1_X_In.end());
      std::vector<T> D_A1_Y(D_A1_Y_In.begin(), D_A1_Y_In.end());
      std::vector<T> D_A1_X_New(D_A1_X.size());
      std::vector<T> D_A1_Y_New(D_A1_Y.size());
      std::vector<T> D_A1_MeasureErrors(D_A1_X.size());
      std::vector<T> D_A1_MeasureErrors_New(D_A1_X.size());
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
        assert(P_D_A1_MeasureErrors->getShape()[0] == D_A1_X_In.getShape()[0]);
      }
      else{
        for (auto it = measureErrors.begin(); it != measureErrors.end(); ++it)
          *it = (*it > 0) ? sqrt(*it) : 1;
      }
      ndarray::Array<T, 1, 1> D_A1_MeasureErrorsBak = copy(measureErrors);

      PTR(std::vector<size_t>) P_I_A1_NotRejected(new std::vector<size_t>());
      bool B_KeyWordSet_NotRejected = false;
      I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "NOT_REJECTED");
      if (I_Pos >= 0){
        P_I_A1_NotRejected.reset();
        P_I_A1_NotRejected = *((PTR(std::vector<size_t>)*)(ArgV[I_Pos]));
        #ifdef __DEBUG_POLYFIT__
          cout << "PolyFit: P_I_A1_NotRejected = ";
          for (auto it = P_I_A1_NotRejected->begin(); it != P_I_A1_NotRejected->end(); ++it)
            cout << *it << " ";
          cout << endl;
        #endif
        B_KeyWordSet_NotRejected = true;
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: KeyWord NOT_REJECTED read" << endl;
        #endif
      }

      PTR(std::vector<size_t>) P_I_A1_Rejected(new std::vector<size_t>());
      bool B_KeyWordSet_Rejected = false;
      I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "REJECTED");
      if (I_Pos >= 0){
        P_I_A1_Rejected.reset();
        P_I_A1_Rejected = *((PTR(std::vector<size_t>)*)(ArgV[I_Pos]));
        #ifdef __DEBUG_POLYFIT__
          cout << "PolyFit: P_I_A1_NotRejected = ";
          for (auto it = P_I_A1_Rejected->begin(); it != P_I_A1_Rejected->end(); ++it)
            cout << *it << " ";
          cout << endl;
        #endif
        B_KeyWordSet_Rejected = true;
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: KeyWord REJECTED read" << endl;
        #endif
      }

      PTR(int) P_I_NRejected(new int(0));
      bool B_KeyWordSet_NRejected = false;
      I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "N_REJECTED");
      if (I_Pos >= 0){
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: Reading KeyWord N_REJECTED" << endl;
          cout << "CFits::PolyFit: I_Pos = " << I_Pos << endl;
        #endif
        P_I_NRejected.reset();
        P_I_NRejected = *((PTR(int)*)(ArgV[I_Pos]));
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: P_I_NRejected = " << *P_I_NRejected << endl;
        #endif
        B_KeyWordSet_NRejected = true;
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: KeyWord N_REJECTED read" << endl;
        #endif
      }

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
          cout << "CFits::PolyFit: PolyFit(D_A1_X, D_A1_Y, I_Degree_In, S_A1_Args_In, ArgV) returned D_A1_Coeffs_Out = " << D_A1_Coeffs_Out << endl;
        #endif
        ndarray::Array<T, 1, 1> D_A1_YFit = pfs::drp::stella::math::Poly(D_A1_XArr, 
                                                                      D_A1_Coeffs_Out);

        ndarray::Array<T, 1, 1> D_A1_Temp = ndarray::allocate(D_A1_Y.size());
        for (int pos = 0; pos < D_A1_Y.size(); ++pos)
          D_A1_Temp[pos] = D_A1_Y[pos] - D_A1_YFit[pos];
        Eigen::Array<T, Eigen::Dynamic, 1> tempEArr = D_A1_Temp.asEigen();
        D_A1_Temp.asEigen() = tempEArr.pow(2) / T(D_A1_Y.size());
//        D_A1_Temp.deep() = D_A1_Temp / D_A1_Y.size();
        double D_SDev = double(sqrt(D_A1_Temp.asEigen().sum()));

        D_A1_PolyRes.deep() = pfs::drp::stella::math::Poly(D_A1_X_In, 
                                                           D_A1_Coeffs_Out);
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
            D_A1_X.push_back(D_A1_X_In[i_pos]);
            D_A1_Y.push_back(D_A1_Y_In[i_pos]);
            if (B_HaveMeasureErrors)
              D_A1_MeasureErrors.push_back(D_A1_MeasureErrorsBak[i_pos]);
            I_A1_OrigPos.push_back(D_A1_Y_In[i_pos]);

            I_DataValues_New++;
          }
          else{
            P_I_A1_Rejected->push_back(i_pos);
            #ifdef __DEBUG_POLYFIT__
              cout << "CFits::PolyFit: Rejecting D_A1_X_In(" << i_pos << ") = " << D_A1_X_In[i_pos] << endl;
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
            if (fabs((*P_I_A1_Rejected)[i_pos] - I_A1_Rejected_Old[i_pos]) > 0.0001)
              B_Run = true;
          }
        }
        i_iter++;
        if ((I_NIter >= 0) && (i_iter >= I_NIter))
          B_Run = false;
      }
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: *P_I_NRejected = " << *P_I_NRejected << endl;
        cout << "CFits::PolyFit: I_DataValues_New = " << I_DataValues_New << endl;
      #endif
      *P_I_A1_NotRejected = I_A1_OrigPos;
      I_A1_OrigPos.resize(D_A1_X_In.getShape()[0]);
      for (size_t i_pos = 0; i_pos < I_A1_OrigPos.size(); ++i_pos)
        I_A1_OrigPos[i_pos] = i_pos;
      std::vector<size_t> V_OrigPos(I_A1_OrigPos.begin(), I_A1_OrigPos.end());
      std::vector<size_t> V_NotRejected(P_I_A1_NotRejected->begin(), P_I_A1_NotRejected->end());
      *P_I_A1_Rejected = pfs::drp::stella::math::removeSubArrayFromArray(V_OrigPos, V_NotRejected);

      return D_A1_Coeffs_Out;
    }

    /** **********************************************************************/

    template< typename T >
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In){
      assert(D_A1_X_In.getShape()[0] == D_A1_Y_In.getShape()[0]);
      std::vector<string> S_A1_Args(1);
      S_A1_Args[0] = " ";
      std::vector<void *> PP_Args(1);
      return pfs::drp::stella::math::PolyFit(D_A1_X_In, 
                                             D_A1_Y_In, 
                                             I_Degree_In, 
                                             S_A1_Args, 
                                             PP_Args);
    }

/**
    CHISQ=double(chisq): out
    COVAR=covar: out
    MEASURE_ERRORS=measure_errors: in
    SIGMA=sigma: out
    STATUS=status: out
    YERROR=yerror
    YFIT=yfit: out
    LSIGMA=lsigma: lower sigma rejection threshold
    USIGMA=usigma:
    ;**/

    template< typename T >
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In,
                                         std::vector<string> const& S_A1_Args_In,
                                         std::vector<void *> & ArgV){

      assert(D_A1_X_In.getShape()[0] == D_A1_Y_In.getShape()[0]);
      
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: Starting " << endl;
      #endif
      size_t const nCoeffs(I_Degree_In + 1);
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: nCoeffs set to " << nCoeffs << endl;
      #endif
      ndarray::Array<double, 1, 1> D_A1_Out = ndarray::allocate(nCoeffs);
      D_A1_Out.deep() = 0.;
      int i, j, I_Pos;

      const int nDataPoints(D_A1_X_In.getShape()[0]);
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: nDataPoints set to " << nDataPoints << endl;
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
        assert(P_D_A1_MeasureErrors->getShape()[0] == nDataPoints);
        D_A1_MeasureErrors.deep() = *P_D_A1_MeasureErrors;
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: B_HaveMeasureError set to TRUE" << endl;
          cout << "CFits::PolyFit: *P_D_A1_MeasureErrors set to " << *P_D_A1_MeasureErrors << endl;
        #endif
      }
      else{
        D_A1_MeasureErrors.deep() = 1.;
      }

      D_A1_SDevSquare.deep() = D_A1_MeasureErrors * D_A1_MeasureErrors;
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: D_A1_SDevSquare set to " << D_A1_SDevSquare << endl;
      #endif
      ndarray::Array<T, 1, 1> D_A1_YFit = ndarray::allocate(nDataPoints);
      PTR(ndarray::Array<T, 1, 1>) P_D_A1_YFit(new ndarray::Array<T, 1, 1>(D_A1_YFit));
      sTemp = "YFIT";
      if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
      {
        P_D_A1_YFit.reset();
        P_D_A1_YFit = *((PTR(ndarray::Array<T, 1, 1>)*)ArgV[I_Pos]);
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: KeyWord_Set(YFIT)" << endl;
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
        assert(P_D_A1_Sigma->getShape()[0] == nCoeffs);
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: KeyWord_Set(SIGMA): *P_D_A1_Sigma set to " << (*P_D_A1_Sigma) << endl;
        #endif
      }

      ndarray::Array<T, 2, 2> D_A2_Covar = ndarray::allocate(nCoeffs, nCoeffs);
      PTR(ndarray::Array<T, 2, 2>) P_D_A2_Covar(new ndarray::Array<T, 2, 2>(D_A2_Covar));
      sTemp = "COVAR";
      if ((I_Pos = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, sTemp)) >= 0)
      {
        P_D_A2_Covar.reset();
        P_D_A2_Covar = *((PTR(ndarray::Array<T, 2, 2>)*)ArgV[I_Pos]);
        assert(P_D_A2_Covar->getShape()[0] == nCoeffs);
        assert(P_D_A2_Covar->getShape()[1] == nCoeffs);
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: KeyWord_Set(COVAR): *P_D_A2_Covar set to " << (*P_D_A2_Covar) << endl;
        #endif
      }

      ndarray::Array<T, 1, 1> D_A1_B = ndarray::allocate(nCoeffs);
      ndarray::Array<T, 1, 1> D_A1_Z = ndarray::allocate(nDataPoints);
      D_A1_Z.deep() = 1;
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: D_A1_Z set to " << D_A1_Z << endl;
      #endif

      ndarray::Array<T, 1, 1> D_A1_WY = ndarray::allocate(nDataPoints);
      D_A1_WY.deep() = D_A1_Y_In;
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: D_A1_WY set to " << D_A1_WY << endl;
      #endif

      if (B_HaveMeasureError){
        D_A1_WY.deep() = D_A1_WY / D_A1_SDevSquare;
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: B_HaveMeasureError: D_A1_WY set to " << D_A1_WY << endl;
        #endif
      }

      if (B_HaveMeasureError){
        (*P_D_A2_Covar)[0][0] = sum(1./D_A1_SDevSquare);
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: B_HaveMeasureError: (*P_D_A2_Covar)(0,0) set to " << (*P_D_A2_Covar)[0][0] << endl;
        #endif
      }
      else{
        (*P_D_A2_Covar)[0][0] = nDataPoints;
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: !B_HaveMeasureError: (*P_D_A2_Covar)(0,0) set to " << (*P_D_A2_Covar)[0][0] << endl;
        #endif
      }

      D_A1_B[0] = sum(D_A1_WY);
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: D_A1_B(0) set to " << D_A1_B[0] << endl;
      #endif

      T D_Sum;
      for (int p = 1; p <= 2 * I_Degree_In; p++){
        D_A1_Z.deep() = D_A1_Z * D_A1_X_In;
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: for(p(=" << p << ")...): D_A1_Z set to " << D_A1_Z << endl;
        #endif
        if (p < nCoeffs){
          D_A1_B[p] = sum(D_A1_WY * D_A1_Z);
          #ifdef __DEBUG_POLYFIT__
            cout << "CFits::PolyFit: for(p(=" << p << ")...): p < nCoeffs(=" << nCoeffs << "): D_A1_B(p) set to " << D_A1_B[p] << endl;
          #endif
        }
        if (B_HaveMeasureError){
          D_Sum = sum(D_A1_Z / D_A1_SDevSquare);
          #ifdef __DEBUG_POLYFIT__
            cout << "CFits::PolyFit: for(p(=" << p << ")...): B_HaveMeasureError: D_Sum set to " << D_Sum << endl;
          #endif
        }
        else{
          D_Sum = sum(D_A1_Z);
          #ifdef __DEBUG_POLYFIT__
            cout << "CFits::PolyFit: for(p(=" << p << ")...): !B_HaveMeasureError: D_Sum set to " << D_Sum << endl;
          #endif
        }
        if (p - int(I_Degree_In) > 0){
          i = p - int(I_Degree_In);
        }
        else{
          i = 0;
        }
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: for(p(=" << p << ")...): I_Degree_In = " << I_Degree_In << ": i set to " << i << endl;
        #endif
        for (j = i; j <= I_Degree_In; j++){
          (*P_D_A2_Covar)[j][p-j] = D_Sum;
          #ifdef __DEBUG_POLYFIT__
            cout << "CFits::PolyFit: for(p(=" << p << ")...): for(j(=" << j << ")...): (*P_D_A2_Covar)(j,p-j=" << p-j << ") set to " << (*P_D_A2_Covar)[j][p-j] << endl;
          #endif
        }
      }

      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: before InvertGaussJ: (*P_D_A2_Covar) = " << (*P_D_A2_Covar) << endl;
      #endif
      P_D_A2_Covar->asEigen() = P_D_A2_Covar->asEigen().inverse();
//      if (!pfs::drp::stella::math::InvertGaussJ(*P_D_A2_Covar)){
//        cout << "CFits::PolyFit: ERROR! InvertGaussJ(*P_D_A2_Covar=" << *P_D_A2_Covar << ") returned false!" << endl;
//        string message("CFits::PolyFit: ERROR! InvertGaussJ(*P_D_A2_Covar) returned false!");
//        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
//      }
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: InvertGaussJ: (*P_D_A2_Covar) set to " << (*P_D_A2_Covar) << endl;
        cout << "CFits::PolyFit: MatrixTimesVecArr: P_D_A2_Covar->rows() = " << P_D_A2_Covar->getShape()[0] << endl;
        cout << "CFits::PolyFit: MatrixTimesVecArr: P_D_A2_Covar->cols() = " << P_D_A2_Covar->getShape()[1] << endl;
        cout << "CFits::PolyFit: MatrixTimesVecArr: (*P_D_A2_Covar) = " << (*P_D_A2_Covar) << endl;
        cout << "CFits::PolyFit: MatrixTimesVecArr: D_A1_B = " << D_A1_B.getShape()[0] << ": " << D_A1_B << endl;
      #endif
      ndarray::Array<T, 1, 1> T_A1_Out = ndarray::allocate(D_A1_Out.getShape()[0]);
      T_A1_Out.asEigen() = P_D_A2_Covar->asEigen() * D_A1_B.asEigen();
      for (int pos = 0; pos < D_A1_Out.getShape()[0]; ++pos)
        D_A1_Out[pos] = T_A1_Out[pos];
//      D_A1_Out.deep() = MatrixTimesVecArr(*P_D_A2_Covar, D_A1_B);
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: MatrixTimesVecArr: P_D_A1_YFit->size() = " << P_D_A1_YFit->getShape()[0] << ": D_A1_Out set to " << D_A1_Out << endl;
      #endif

      P_D_A1_YFit->deep() = D_A1_Out[I_Degree_In];
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: InvertGaussJ: (*P_D_A1_YFit) set to " << (*P_D_A1_YFit) << endl;
      #endif

      for (int k=I_Degree_In-1; k >= 0; k--){
        P_D_A1_YFit->deep() = D_A1_Out[k] + (*P_D_A1_YFit) * D_A1_X_In;
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: for(k(=" << k << ")...): (*P_D_A1_YFit) set to " << (*P_D_A1_YFit) << endl;
        #endif
      }

      for (int k = 0; k < nCoeffs; k++){
        (*P_D_A1_Sigma)[k] = (*P_D_A2_Covar)[k][k];
      }
      for (auto it = P_D_A1_Sigma->begin(); it != P_D_A1_Sigma->end(); ++it){
        *it = (*it > 0) ? sqrt(*it) : 1.;
      }
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: (*P_D_A1_Sigma) set to " << (*P_D_A1_Sigma) << endl;
      #endif

      double D_ChiSq = 0.;
      ndarray::Array<T, 1, 1> D_A1_Diff = ndarray::allocate(nDataPoints);
      Eigen::Array<T, Eigen::Dynamic, 1> EArr_Temp = D_A1_Y_In.asEigen() - P_D_A1_YFit->asEigen();
      D_A1_Diff.asEigen() = EArr_Temp.pow(2);
      if (B_HaveMeasureError){
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: B_HaveMeasureError: D_A1_Diff set to " << D_A1_Diff << endl;
        #endif
        D_ChiSq = sum(D_A1_Diff / D_A1_SDevSquare);
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: B_HaveMeasureError: D_ChiSq set to " << D_ChiSq << endl;
        #endif

      }
      else{
        D_ChiSq = sum(D_A1_Diff);
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: !B_HaveMeasureError: D_ChiSq set to " << D_ChiSq << endl;
        #endif

        double dTemp = sqrt(D_ChiSq / (nDataPoints - nCoeffs));
        P_D_A1_Sigma->deep() = (*P_D_A1_Sigma) * dTemp;
        #ifdef __DEBUG_POLYFIT__
          cout << "CFits::PolyFit: !B_HaveMeasureError: (*P_D_A1_Sigma) set to " << (*P_D_A1_Sigma) << endl;
        #endif
      }
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: returning D_A1_Out = " << D_A1_Out << endl;
      #endif

      return D_A1_Out;
    }

    /** **********************************************************************/

    template< typename T >
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In,
                                         T const D_Reject_In){
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: Starting " << endl;
      #endif
      std::vector<string> S_A1_Args(1);
      S_A1_Args[0] = " ";
      std::vector<void *> PP_Args;
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
                                    size_t const I_NIter){
      #ifdef __DEBUG_POLYFIT__
        cout << "CFits::PolyFit: Starting " << endl;
      #endif
      std::vector<string> S_A1_Args(1);
      S_A1_Args[0] = " ";
      std::vector<void *> PP_Args(1);
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
        cout << "CFits::PolyFit: PolyFit returned D_A1_Out = " << D_A1_Out << endl;
      #endif
      return D_A1_Out;
    }
    
    template<typename T>
    std::vector<T> indGen(T len){
      std::vector<T> vecOut;
      for (T i = 0; i < len; ++i)
        vecOut.push_back(i);
      return vecOut;
    }

    template< typename T >
    std::vector<T> removeSubArrayFromArray(std::vector<T> const& A1_Array_InOut,
                                           std::vector<T> const& A1_SubArray){
      std::vector<T> A1_Array_Out;
      int I_NElements = 0;
      bool B_InSubArray = false;
      for (auto i_orig = A1_Array_InOut.begin(); i_orig != A1_Array_InOut.end(); ++i_orig){
        B_InSubArray = false;
        for (auto i_sub = A1_SubArray.begin(); i_sub != A1_SubArray.end(); ++i_sub){
          if (*i_orig == *i_sub)
            B_InSubArray = true;
        }
        if (!B_InSubArray){
          A1_Array_Out.push_back(*i_orig);
          I_NElements++;
        }
      }
      return A1_Array_Out;
    }

    
/*    template< typename T >
    bool InvertGaussJ(ndarray::Array<T, 2, 1> & AArray){
      int N = AArray.getShape()[1];
      assert(N == AArray.getShape()[0]);
      ndarray::Array<T, 2, 1> Unity(N, N);
      Unity.deep() = 0.;
      for (int m = 0; m < N; m ++){
        Unity[m][m] = 1.;
      }
      if (!pfs::drp::stella::math::InvertGaussJ(AArray, Unity)){
        cout << "CFits::InvertGaussJ: ERROR: InvertGaussJ(AArray=" << AArray << ", Unity=" << Unity << ") retuned FALSE" << endl;
        return false;
      }
      return true;
    }
*/
    
    template<typename T>
    bool countPixGTZero(ndarray::Array<T, 1, 1> &vec_InOut){
      int count = 0;
      if (vec_InOut.getShape()[0] < 1)
        return false;
      int pos = 0;
      for (auto i = vec_InOut.begin(); i != vec_InOut.end(); ++i){
        #ifdef __DEBUG_COUNTPIXGTZERO__
          cout << "CFits::countPixGTZero: pos = " << pos << ": vec_InOut(pos) = " << *i << endl;
        #endif
        if (*i <= T(0))
          count = 0;
        else
          count++;
        #ifdef __DEBUG_COUNTPIXGTZERO__
          cout << "CFits::countPixGTZero: pos = " << pos << ": count set to " << count << endl;
        #endif
        *i = T(count);
        #ifdef __DEBUG_COUNTPIXGTZERO__
          cout << "CFits::countPixGTZero: pos = " << pos << ": vec_InOut(i) set to " << *i << endl;
        #endif
      }
      return true;
    }

    template<typename T>
    int firstIndexWithValueGEFrom(ndarray::Array<T, 1, 1> const& vec_In, const T minValue_In, const int fromIndex_In){
      if ((vec_In.getShape()[0] < 1) || (fromIndex_In >= int(vec_In.getShape()[0]))){
        cout << "pfs::drp::stella::math::firstIndexWithValueGEFrom: Error: vec_In.getShape()[0] =" << vec_In.getShape()[0] << " < 1 or fromIndex_In >= vec_In.getShape()[0] => Returning -1" << endl;
        return -1;
      }
      int pos = fromIndex_In;
      for (auto i = vec_In.begin()+pos; i != vec_In.end(); ++i){
        if (*i >= minValue_In)
          return pos;
        ++pos;
      }
      #ifdef __DEBUG_INDEX__
        cout << "pfs::drp::stella::math::firstIndexWithValueGEFrom: not found => Returning -1" << endl;
      #endif
      return -1;
    }

    template<typename T>
    int lastIndexWithZeroValueBefore(ndarray::Array<T, 1, 1> const& vec_In, const int startPos_In){
      if ( ( startPos_In < 0 ) || ( startPos_In >= static_cast<int>(vec_In.size()) ) )
        return -1;
      int pos = startPos_In;
      for (auto i = vec_In.begin() + startPos_In; i != vec_In.begin(); --i){
        if (fabs(double(*i)) < 0.00000000000000001)
          return pos;
        --pos;
      }
      return -1;
    }

    template<typename T>
    int firstIndexWithZeroValueFrom(ndarray::Array<T, 1, 1> const& vec_In, const int startPos_In){
      if (startPos_In < 0 || startPos_In >= vec_In.getShape()[0])
        return -1;
      int pos = startPos_In;
      for (auto i = vec_In.begin() + pos; i != vec_In.end(); ++i){
        #ifdef __DEBUG_FINDANDTRACE__
          cout << "CFits::FirstIndexWithZeroValueFrom: pos = " << pos << endl;
          cout << "CFits::FirstIndexWithZeroValueFrom: I_A1_VecIn(pos) = " << *i << endl;
        #endif
        if (fabs(*i) < 0.00000000000000001)
          return pos;
        ++pos;
      }
      return -1;
    }
    
    template< typename ImageT, typename SlitFuncT>
    bool LinFitBevingtonEigen(Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic> const& D_A2_CCD_In,      /// yvec: in
                              Eigen::Array<SlitFuncT, Eigen::Dynamic, Eigen::Dynamic> const& D_A2_SF_In,       /// xvec: in
                              Eigen::Array<ImageT, Eigen::Dynamic, 1> & D_A1_SP_Out,                         /// a1: out
                              Eigen::Array<ImageT, Eigen::Dynamic, 1> & D_A1_Sky_Out,                        /// a0: out
                              bool B_WithSky,                           /// with sky: in
                              vector<string> const& S_A1_Args_In,   ///: in
                              vector<void *> & ArgV_In)                    ///: in
    /// MEASURE_ERRORS_IN = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
    /// REJECT_IN         = double
    /// MASK_INOUT        = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
    /// CHISQ_OUT         = blitz::Array<double, 1>(D_A2_CCD_In.rows())
    /// Q_OUT             = blitz::Array<double, 1>(D_A2_CCD_In.rows())
    /// SIGMA_OUT         = blitz::Array<double, 2>(D_A2_CCD_In.rows(),2)
    /// YFIT_OUT          = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
    {
      #ifdef __DEBUG_FITARR__
        cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr) started" << endl;
        cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr): D_A2_CCD_In = " << D_A2_CCD_In << endl;
        cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr): D_A2_SF_In = " << D_A2_SF_In << endl;
      #endif
      assert(D_A2_CCD_In.size() == D_A2_SF_In.size());
      assert (D_A1_SP_Out.size() == D_A2_CCD_In.rows());
      assert (D_A1_Sky_Out.size() == D_A2_CCD_In.rows());
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
        assert(P_D_A2_Sigma->rows() == D_A2_CCD_In.rows());
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
        assert(P_D_A1_ChiSq->rows() == D_A2_CCD_In.rows());
        S_A1_Args_Fit[I_ArgPos] = "CHISQ_OUT";
        I_ArgPos++;
      }

      PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>) P_D_A1_Q(new Eigen::Array<ImageT, Eigen::Dynamic, 1>(D_A2_CCD_In.rows()));
      I_KeywordSet_Q = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "Q_OUT");
      if (I_KeywordSet_Q >= 0)
      {
        P_D_A1_Q.reset();
        P_D_A1_Q = *((PTR(Eigen::Array<ImageT, Eigen::Dynamic, 1>)*)ArgV_In[I_KeywordSet_Q]);
        assert(P_D_A1_Q->rows() == D_A2_CCD_In.rows());
        S_A1_Args_Fit[I_ArgPos] = "Q_OUT";
        I_ArgPos++;
      }

      PTR(Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic>) P_D_A2_Sigma_Out(new Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic>(D_A2_CCD_In.rows(), 2));
      I_KeywordSet_Sigma = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "SIGMA_OUT");
      if (I_KeywordSet_Sigma >= 0)
      {
        P_D_A2_Sigma_Out.reset();
        P_D_A2_Sigma_Out = *((PTR(Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic>)*)ArgV_In[I_KeywordSet_Sigma]);
        assert(P_D_A2_Sigma_Out->rows() == D_A2_CCD_In.rows());
        assert(P_D_A2_Sigma_Out->cols() == 2);
        S_A1_Args_Fit[I_ArgPos] = "SIGMA_OUT";
        I_ArgPos++;
      }

      I_KeywordSet_YFit = pfs::drp::stella::utils::KeyWord_Set(S_A1_Args_In, "YFIT_OUT");
      if (I_KeywordSet_YFit >= 0)
      {
        P_D_A2_YFit.reset();
        P_D_A2_YFit = *((PTR(Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic>)*)ArgV_In[I_KeywordSet_YFit]);
        assert(P_D_A2_YFit->rows() == D_A2_CCD_In.rows());
        assert(P_D_A2_YFit->cols() == D_A2_CCD_In.cols());
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
        assert(P_I_A2_Mask->rows() == D_A2_CCD_In.rows());
        assert(P_I_A2_Mask->cols() == D_A2_CCD_In.cols());
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
            string message("CFits::LinFitBevington: WARNING: LinFitBevington(D_A2_CCD_In(i,blitz::Range::all()),D_A2_SF_In(i,blitz::Range::all()),D_A1_SP_Out(i),D_A1_Sky_Out(i),D_A1_STDDEV_Out(i),D_A1_Covariance_Out(i)) returned status = ");
            message += to_string(status);
            cout << message << endl;
            cout << "CFits::LinFitBevington: D_A2_SF_In(0, *) = " << D_A2_SF_In.row(0) << endl;
            
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
      return true;
    }

    template< typename ImageT, typename SlitFuncT>
    bool LinFitBevingtonNdArray(ndarray::Array<ImageT, 2, 1> const& D_A2_CCD_In,      /// yvec: in
                                ndarray::Array<SlitFuncT, 2, 1> const& D_A2_SF_In,       /// xvec: in
                                ndarray::Array<ImageT, 1, 1> & D_A1_SP_Out,                         /// a1: out
                                ndarray::Array<ImageT, 1, 1> & D_A1_Sky_Out,                        /// a0: out
                                bool B_WithSky,                           /// with sky: in
                                vector<string> const& S_A1_Args_In,   ///: in
                                vector<void *> & ArgV_In)                    ///: in
    /// MEASURE_ERRORS_IN = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
    /// REJECT_IN         = double
    /// MASK_INOUT        = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
    /// CHISQ_OUT         = blitz::Array<double, 1>(D_A2_CCD_In.rows())
    /// Q_OUT             = blitz::Array<double, 1>(D_A2_CCD_In.rows())
    /// SIGMA_OUT         = blitz::Array<double, 2>(D_A2_CCD_In.rows(),2)
    /// YFIT_OUT          = blitz::Array<double, 2>(D_A2_CCD_In.rows(), D_A2_CCD_In.cols())
    {
      #ifdef __DEBUG_FITARR__
        cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr) started" << endl;
        cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr): D_A2_CCD_In = " << D_A2_CCD_In << endl;
        cout << "CFits::LinFitBevington(Array, Array, Array, Array, bool, CSArr, PPArr): D_A2_SF_In = " << D_A2_SF_In << endl;
      #endif
      assert(D_A2_CCD_In.getShape()[0] == D_A2_SF_In.getShape()[0]);
      assert(D_A2_CCD_In.getShape()[1] == D_A2_SF_In.getShape()[1]);
      assert (D_A1_SP_Out.getShape()[0] == D_A2_CCD_In.getShape()[0]);
      assert (D_A1_Sky_Out.getShape()[0] == D_A2_CCD_In.getShape()[0]);
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
        assert(P_D_A2_Sigma->getShape()[0] == D_A2_CCD_In.getShape()[0]);
        assert(P_D_A2_Sigma->getShape()[1] == D_A2_CCD_In.getShape()[1]);
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
        assert(P_D_A1_ChiSq->getShape()[0] == D_A2_CCD_In.getShape()[0]);
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
        assert(P_D_A1_Q->getShape()[0] == D_A2_CCD_In.getShape()[0]);
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
        assert(P_D_A2_Sigma_Out->getShape()[0] == D_A2_CCD_In.getShape()[0]);
        assert(P_D_A2_Sigma_Out->getShape()[1] == 2);
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
        assert(P_D_A2_YFit->getShape()[0] == D_A2_CCD_In.getShape()[0]);
        assert(P_D_A2_YFit->getShape()[1] == D_A2_CCD_In.getShape()[1]);
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
        assert(P_I_A2_Mask->getShape()[0] == D_A2_CCD_In.getShape()[0]);
        assert(P_I_A2_Mask->getShape()[1] == D_A2_CCD_In.getShape()[1]);
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
            string message("CFits::LinFitBevington: WARNING: LinFitBevington(D_A2_CCD_In(i,blitz::Range::all()),D_A2_SF_In(i,blitz::Range::all()),D_A1_SP_Out(i),D_A1_Sky_Out(i),D_A1_STDDEV_Out(i),D_A1_Covariance_Out(i)) returned status = ");
            message += to_string(status);
            cout << message << endl;
            cout << "CFits::LinFitBevington: D_A2_SF_In(0, *) = " << D_A2_SF_In[ndarray::view(0)()] << endl;
            
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
      return true;
    }

    /**
      Fit(blitz::Array<double, 1> y, const blitz::Array<double, 1> x, a1, a0);
      calculates a0 and a1 for the system of equations yvec = a0 + a1 * xvec
    **/
    template<typename ImageT, typename SlitFuncT>
    int LinFitBevingtonEigen(Eigen::Array<ImageT, Eigen::Dynamic, 1> const& D_A1_CCD_In,      /// yvec: in
                              Eigen::Array<SlitFuncT, Eigen::Dynamic, 1> const& D_A1_SF_In,       /// xvec: in
                              ImageT &D_SP_Out,                         /// a1: out
                              ImageT &D_Sky_Out,                        /// a0: in/out
                              bool B_WithSky,                        /// with sky: in
                              std::vector<string> const& S_A1_Args_In,   ///: in
                              std::vector<void *> & ArgV_In)                    ///: in
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
        cout << "CFits::LinFitBevington: Warning: (D_A1_CCD_In.sum(=" << D_A1_CCD_In.sum() << " == 0.) || (D_A1_SF.sum(=" << D_A1_SF.sum() << ") == 0.) => returning false" << endl;
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
        assert(P_I_A1_Mask->size() == D_A1_CCD_In.size());
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
        assert(P_D_A1_Sigma_Out->size() == 2);
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
        assert(D_A1_CCD_In.size() == P_D_A1_Sig->size());
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
        assert(P_D_A1_YFit->size() == D_A1_CCD_In.size());
        P_D_A1_YFit->setConstant(0.);
      }
      if (P_I_A1_Mask->sum() == 0){
        cout << "CFits::LinFitBevington: WARNING: P_I_A1_Mask->sum() == 0" << endl;
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
            if (fabs(D_A1_Sig(i)) < 0.00000000000000001){
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
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": fabs(D_SP_Out) = " << fabs(D_SP_Out) << endl;
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
        if (fabs(D_SP_Out) < 0.000001)
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

      return status;
    }

    template<typename ImageT, typename SlitFuncT>
    int LinFitBevingtonNdArray(ndarray::Array<ImageT, 1, 1> const& D_A1_CCD_In,      /// yvec: in
                               ndarray::Array<SlitFuncT, 1, 1> const& D_A1_SF_In,       /// xvec: in
                               ImageT &D_SP_Out,                         /// a1: out
                               ImageT &D_Sky_Out,                        /// a0: in/out
                               bool B_WithSky,                        /// with sky: in
                               std::vector<string> const& S_A1_Args_In,   ///: in
                               std::vector<void *> & ArgV_In)                    ///: in
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
        cout << "CFits::LinFitBevington: Warning: (D_A1_CCD_In.sum(=" << D_A1_CCD_In.asEigen().sum() << " == 0.) || (D_A1_SF.sum(=" << D_A1_SF.asEigen().sum() << ") == 0.) => returning false" << endl;
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
        assert(P_I_A1_Mask->getShape()[0] == ndata);
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
        assert(P_D_A1_Sigma_Out->getShape()[0] == 2);
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: KeyWord_Set(SIGMA_OUT)" << endl;
        #endif
      }
      P_D_A1_Sigma_Out->deep() = 0.;

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
        P_D_A1_Sig.reset();
        P_D_A1_Sig = *(PTR(ndarray::Array<ImageT, 1, 1>)*)ArgV_In[I_KeywordSet_MeasureErrors];
        #ifdef __DEBUG_FIT__
          cout << "CFits::LinFitBevington: *P_D_A1_Sig = " << *P_D_A1_Sig << endl;
        #endif
        assert(P_D_A1_Sig->getShape()[0] == ndata);
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
        assert(P_D_A1_YFit->getShape()[0] == ndata);
      }
      P_D_A1_YFit->deep() = 0.;
      if (P_I_A1_Mask->asEigen().sum() == 0){
        cout << "CFits::LinFitBevington: WARNING: P_I_A1_Mask->sum() == 0" << endl;
        D_SP_Out = 0.;
        D_Sky_Out = 0.;
        status = 0;
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
        D_A1_Sig = math::resize(D_A1_Sig, I_MaskSum);
        D_A1_CCD = math::resize(D_A1_CCD, I_MaskSum);
        D_A1_SF = math::resize(D_A1_SF, I_MaskSum);
        D_A1_WT = math::resize(D_A1_WT, I_MaskSum);
        D_A1_YFit = math::resize(D_A1_YFit, I_MaskSum);

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
            if (fabs(D_A1_Sig[i]) < 0.00000000000000001){
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
          cout << "CFits::LinFitBevington: I_Run=" << I_Run << ": fabs(D_SP_Out) = " << fabs(D_SP_Out) << endl;
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
        if (fabs(D_SP_Out) < 0.000001)
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

      return status;
    }
    
    /**
     * Helper function to calculate incomplete Gamma Function
     **/
    template< typename T>
    T GSER(T & D_Gamser_Out, T const a, T const x)
    {
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
        return D_GLn_Out;
      }
      else{
        ap = a;
        del = d_sum = 1. / a;
        for (n=1; n <= ITMax; n++){
          ++ap;
          del *= x/ap;
          d_sum += del;
          if (fabs(del) < fabs(d_sum) * 3.e-7){
            D_Gamser_Out = d_sum * exp(-x+a*log(x) - D_GLn_Out);
            #ifdef __DEBUG_LINFIT__
              cout << "CFits::GSER: x>0: D_Gamser_Out = " << D_Gamser_Out << endl;
              cout << "CFits::GSER: x>0: D_GLn_Out = " << D_GLn_Out << endl;
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
      return D_Result;
    }

    /**
     * Helper function to calculate incomplete Gamma Function
     **/
    template<typename T>
    T GCF(T & D_GammCF_Out, T const a, T const x)
    {
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
        if (fabs(d) < FPMIN)
          d = FPMIN;
        c = b + an / c;
        #ifdef __DEBUG_FIT__
          cout << "CFits::GCF: n = " << n << ": c set to " << c << endl;
        #endif
        if (fabs(c) < FPMIN)
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
        if (fabs(del-1.) < EPS)
          break;
      }
      if (n > ITMAX){
        string message("CFits::GCF: ERROR: a too large, ITMAX too small in GCF");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      D_GammCF_Out = exp(-x+a*log(x) - D_GLn_Out) * h;
      return D_GLn_Out;
    }

    /**
     * Function to calculate incomplete Gamma Function P(a,x)
     **/
    template<typename T>
    T GammP(T const a, T const x){
      T D_Out = 0;
      #ifdef __DEBUG_FIT__
        cout << "CFits::GammP started: a = " << a << ", x = " << x << endl;
      #endif
      T gamser, gammcf, gln;
      assert(x >= 0.);
      assert(a > 0.);
      if (x < (a+1.)){
        gln = GSER(gamser, a, x);
        D_Out = gamser;
        return D_Out;
      }
      else{
        gln = GCF(gammcf, a, x);
        D_Out = 1. - gammcf;
        return D_Out;
      }
    }

    /**
     * Function to calculate incomplete Gamma Function Q(a,x) = 1. - P(a,x)
     **/
    template<typename T>
    T GammQ(T const a, T const x){
      T D_Out = 0;
      #ifdef __DEBUG_FIT__
        cout << "CFits::GammQ started: a = " << a << ", x = " << x << endl;
      #endif
      T gamser = 0.;
      T gammcf = 0.;
      T gln = 0.;
      assert(x >= 0.);
      assert(a > 0.);
      if (x < (a+1.)){
        gln = GSER(gamser, a, x);
        #ifdef __DEBUG_FIT__
          cout << "CFits::GammQ: x < (a+1.): gamser = " << gamser << endl;
        #endif
        D_Out = 1. - gamser;
        return D_Out;
      }
      else{
        gln = GCF(gammcf, a, x);
        #ifdef __DEBUG_FIT__
          cout << "CFits::GammQ: x < (a+1.): gammcf = " << gammcf << endl;
        #endif
        D_Out = gammcf;
        return D_Out;
      }
    }

    /**
     *  bool IsOddNumber(long No) const
     *  Returns TRUE, if <No> is an Odd Number, FALSE if <No> is an Even Number.
     **/
    bool IsOddNumber(long No)
    {
      return (fabs((double)(((double)No) / 2.) - (double)(int)(((double)No) / 2.)) > 0.3);
    }

    /**
     * function GetRowFromIndex(int I_Index_In, int I_NRows_In) const
     * task: Returns Row specified by I_Index_In from the formula
     *       Col = (int)(I_Index_In / I_NRows_In)
     *       Row = fiberTraceNumber - Col * I_NRows_In
     **/
    int GetRowFromIndex(int I_Index_In, int I_NRows_In)
    {
      return (I_Index_In - (I_NRows_In * pfs::drp::stella::math::GetColFromIndex(I_Index_In, I_NRows_In)));
    }

    /**
     * function GetColFromIndex(int I_Index_In, int I_NRows_In) const
     * task: Returns Col specified by I_Index_In from the formula
     *       Col = (int)(I_Index_In / I_NRows_In)
     *       Row = fiberTraceNumber - Col * I_NRows_In
     **/
    int GetColFromIndex(int I_Index_In, int I_NRows_In)
    {
      return ((int)(I_Index_In / I_NRows_In));
    }

    template<typename T>
    T Round(const T ToRound, int DigitsBehindDot){
      long TempLong;
      int TempInt;
      T TempDbl;

      bool B_IsNegative = ToRound < 0.;
      TempLong = long(ToRound * pow(10., DigitsBehindDot));
      TempDbl = (ToRound - T(TempLong)) * pow(10., DigitsBehindDot);
      TempInt = int(abs(TempDbl * 10.));
      if (TempInt > 4){
        if (B_IsNegative)
          TempLong--;
        else
          TempLong++;
      }
      return (T(TempLong) / pow(10., DigitsBehindDot));
    }

    /************************************************************/

    template<typename T>
    long RoundL(const T ToRound){
      return long(Round(ToRound, 0));
    }

    /************************************************************/

    template<typename T>
    int Round(const T ToRound){
      return (int)Round(ToRound, 0);
    }
    
//    template<typename T>
//    void resize(blitz::Array<T, 1> &arr_InOut, unsigned int newSize){
//      blitz::Array<T, 1> *newArr = new blitz::Array<T, 1>(newSize);
//      *newArr = 0;
//      arr_InOut.resize(0);
//      &arr_InOut = newArr;
//      return;
//    }
    
    template< typename T, typename U >
    U floor1(T const& rhs, U const& outType){
      U outVal = U(std::llround(std::floor(rhs)));
      return outVal;
    }
    
    template< typename T, typename U >
    ndarray::Array<U, 1, 1> floor(const ndarray::Array<const T, 1, 1> &rhs, const U outType){
      ndarray::Array<U, 1, 1> outVal = allocate(rhs.getShape());
      typename ndarray::Array<U, 1, 1>::Iterator iOut = outVal.begin();
      for (auto iIn = rhs.begin(); iIn != rhs.end(); ++iIn){
        *iOut = floor1(*iIn, outType);
        ++iOut;
      }
      return outVal;
    }
    
    template< typename T, typename U >
    ndarray::Array<U, 2, 2> floor(const ndarray::Array<const T, 2, 2> &rhs, const U outType){
      ndarray::Array<U, 2, 2> outVal = allocate(rhs.getShape());
      typename ndarray::Array<U, 2, 2>::Iterator iOut = outVal.begin();
      typename ndarray::Array<U, 2, 2>::Reference::Iterator jOut = iOut->begin();
      for (auto iIn = rhs.begin(); iIn != rhs.end(); ++iIn) {
        for (auto jIn = iIn->begin(); jIn != iIn->end(); ++jIn) {
          *jOut = floor1(*jIn, outType);
          ++jOut;
        }
        ++iOut;
      }
      return outVal;
    }

    template<typename T>
    T max(ndarray::Array<T, 1, 1> const& in){
      T max = in[0];
      for (auto it = in.begin(); it != in.end(); ++it){
        if (*it > max)
          max = *it;
      }
      return max;
    }

    template<typename T>
    T min(ndarray::Array<T, 1, 1> const& in){
      T min = in[0];
      for (auto it = in.begin(); it != in.end(); ++it){
        if (*it < min)
          min = *it;
      }
      return min;
    }
    
    template <typename T>
    ndarray::Array<double, 1, 1> Double(ndarray::Array<T, 1, 1> const& arr_In){
      ndarray::Array<double, 1, 1> arr_Out = ndarray::allocate(arr_In.getShape()[0]);
      auto it_arr_Out = arr_Out.begin();
      auto it_arr_In = arr_In.begin();
      for (int i = 0; i < arr_In.getShape()[0]; ++i)
        (*(it_arr_Out + i)) = double((*(it_arr_In + i)));
      return arr_Out;
    }
    
    template <typename T>
    ndarray::Array<double, 2, 2> Double(ndarray::Array<T, 2, 2> const& arr_In){
      ndarray::Array<double, 2, 2> arr_Out = ndarray::allocate(arr_In.getShape()[0], arr_In.getShape()[1]);
      auto it_arr_Out = arr_Out.begin();
      auto it_arr_In = arr_In.begin();
      for (int i = 0; i < arr_In.getShape()[0]; ++i){
        auto itJ_Out = (it_arr_Out + i)->begin();
        auto itJ_In = (it_arr_In + i)->begin();
        for (int j = 0; j < arr_In.getShape()[1]; ++j){
          (*(itJ_Out + j)) = double((*(itJ_In + j)));
        }
      }
      return arr_Out;
    }
    
    template <typename T>
    ndarray::Array<float, 1, 1> Float(ndarray::Array<T, 1, 1> const& arr_In){
      ndarray::Array<float, 1, 1> arr_Out = ndarray::allocate(arr_In.getShape()[0]);
      auto it_arr_Out = arr_Out.begin();
      auto it_arr_In = arr_In.begin();
      for (int i = 0; i < arr_In.getShape()[0]; ++i)
        (*(it_arr_Out + i)) = float((*(it_arr_In + i)));
      return arr_Out;
    }
    
    template <typename T>
    ndarray::Array<float, 2, 2> Float(ndarray::Array<T, 2, 2> const& arr_In){
      ndarray::Array<float, 2, 2> arr_Out = ndarray::allocate(arr_In.getShape()[0], arr_In.getShape()[1]);
      auto it_arr_Out = arr_Out.begin();
      auto it_arr_In = arr_In.begin();
      for (int i = 0; i < arr_In.getShape()[0]; ++i){
        auto itJ_Out = (it_arr_Out + i)->begin();
        auto itJ_In = (it_arr_In + i)->begin();
        for (int j = 0; j < arr_In.getShape()[1]; ++j){
          (*(itJ_Out + j)) = float((*(itJ_In + j)));
        }
      }
      return arr_Out;
    }
    
  //  template <typename T>
  //  ndarray::Array<int, 1, 1> Int(ndarray::Array<T, 1, 1> const& arr_In){
  //    ndarray::Array<int, 1, 1> arr_Out = ndarray::allocate(arr_In.getShape()[0]);
  //    auto it_arr_Out = arr_Out.begin();
  //    auto it_arr_In = arr_In.begin();
  //    for (int i = 0; i < arr_In.getShape()[0]; ++i)
  //      (*(it_arr_Out + i)) = int((*(it_arr_In + i)));
  //    return arr_Out;
  //  }
    
    template<typename T>
    ndarray::Array<T, 1, 1> indGenNdArr(T const size){
      ndarray::Array<T, 1, 1> outArr = ndarray::allocate(int(size));
      T ind = 0;
      for (auto it = outArr.begin(); it != outArr.end(); ++it){
        *it = ind;
        ++ind;
      }
      #ifdef __DEBUG_INDGEN__
        cout << "indGen: outArr = " << outArr.getShape() << ": " << outArr << endl;
      #endif
      return outArr;
    }

    template<typename T>
    ndarray::Array<T, 1, 1> replicate(T const val, int const size){
      ndarray::Array<T, 1, 1> out = ndarray::allocate(size);
      for (auto it = out.begin(); it != out.end(); ++it)
        *it = val;
      return out;
    }
        
    template<typename T>
    ndarray::Array<T, 2, 2> calcPosRelativeToCenter(ndarray::Array<T, 2, 2> const& swath_In, ndarray::Array<T, 1, 1> const& xCenters_In){
      ndarray::Array<T, 1, 1> indPos = pfs::drp::stella::math::indGenNdArr(swath_In.getShape()[1]);
      ndarray::Array<T, 1, 1> ones = replicate(float(1.), swath_In.getShape()[0]);
      #ifdef __DEBUG_CALCPOSRELATIVETOCENTER__
        cout << "calcPosRelativeToCenter: indPos = " << indPos << endl;
        cout << "calcPosRelativeToCenter: ones = " << ones << endl;
      #endif
      ndarray::EigenView<T, 1, 1> indPosEigen = indPos.asEigen();
      ndarray::EigenView<T, 1, 1> onesEigen = ones.asEigen();
      #ifdef __DEBUG_CALCPOSRELATIVETOCENTER__
        cout << "calcPosRelativeToCenter: indPosEigen = " << indPosEigen << endl;
        cout << "calcPosRelativeToCenter: onesEigen = " << onesEigen << endl;
      #endif
      Eigen::Matrix<T, swath_In.getShape()[0], swath_In.getShape()[1]> indMat = indPosEigen * onesEigen;
      #ifdef __DEBUG_CALCPOSRELATIVETOCENTER__
        cout << "calcPosRelativeToCenter: indMat = " << indMat << endl;
      #endif
      
      ndarray::Array<T, 2, 2> outArr = ndarray::copy(indMat);
      #ifdef __DEBUG_CALCPOSRELATIVETOCENTER__
        cout << "calcPosRelativeToCenter: outArr = " << outArr << endl;
      #endif

      return outArr;
    }
    
    template<typename T>
    ndarray::Array<size_t, 1, 1> getIndicesInValueRange(ndarray::Array<T, 1, 1> const& arr_In, T const lowRange_In, T const highRange_In){
      std::vector<size_t> indices;
      size_t pos = 0;
      for (auto it = arr_In.begin(); it != arr_In.end(); ++it){
        if ((lowRange_In <= *it) && (*it < highRange_In)){
          indices.push_back(pos);
        }
        ++pos;
      }
      ndarray::Array<size_t, 1, 1> arr_Out = ndarray::allocate(indices.size());
      auto itVec = indices.begin();
      for (auto itArr = arr_Out.begin(); itArr != arr_Out.end(); ++itArr, ++itVec){
        *itArr = *itVec;
      }
      #ifdef __DEBUG_GETINDICESINVALUERANGE__
        cout << "arr_Out = " << arr_Out << endl;
      #endif
      return arr_Out;
    }
    
    template<typename T>
    ndarray::Array<size_t, 2, 2> getIndicesInValueRange(ndarray::Array<T, 2, 2> const& arr_In, T const lowRange_In, T const highRange_In){
      std::vector<size_t> indicesRow;
      std::vector<size_t> indicesCol;
      #ifdef __DEBUG_GETINDICESINVALUERANGE__
        cout << "getIndicesInValueRange: arr_In.getShape() = " << arr_In.getShape() << endl;
      #endif
      for (size_t iRow = 0; iRow < arr_In.getShape()[0]; ++iRow){
        for (size_t iCol = 0; iCol < arr_In.getShape()[1]; ++iCol){
          if ((lowRange_In <= arr_In[iRow][iCol]) && (arr_In[iRow][iCol] < highRange_In)){
            indicesRow.push_back(iRow);
            indicesCol.push_back(iCol);
            #ifdef __DEBUG_GETINDICESINVALUERANGE__
              cout << "getIndicesInValueRange: lowRange_In = " << lowRange_In << ", highRange_In = " << highRange_In << ": arr_In[" << iRow << ", " << iCol << "] = " << arr_In[iRow][iCol] << endl;
            #endif
          }
        }
      }
      ndarray::Array<size_t, 2, 2> arr_Out = ndarray::allocate(indicesRow.size(), 2);
      for (size_t iRow = 0; iRow < arr_Out.getShape()[0]; ++iRow){
        arr_Out[iRow][0] = indicesRow[iRow];
        arr_Out[iRow][1] = indicesCol[iRow];
      }
      #ifdef __DEBUG_GETINDICESINVALUERANGE__
        cout << "getIndicesInValueRange: lowRange_In = " << lowRange_In << ", highRange_In = " << highRange_In << ": arr_Out = [" << arr_Out.getShape() << "] = " << arr_Out << endl;
      #endif
      return arr_Out;
    }
    
    template<typename T>
    ndarray::Array<T, 1, 1> moment(ndarray::Array<T, 1, 1> const& arr_In, int maxMoment_In){
      ndarray::Array<T, 1, 1> D_A1_Out = ndarray::allocate(maxMoment_In);
      D_A1_Out.deep() = 0.;
      if ((maxMoment_In < 1) && (arr_In.getShape()[0] < 2)){
        cout << "CFits::Moment: ERROR: arr_In must contain 2 OR more elements." << endl;
        return D_A1_Out;
      }
      int I_NElements = arr_In.getShape()[0];
      T D_Mean = arr_In.asEigen().mean();
      T D_Kurt = 0.;
      T D_Var = 0.;
      T D_Skew = 0.;
      D_A1_Out[0] = D_Mean;
      if (maxMoment_In == 1)
        return D_A1_Out;

      ndarray::Array<T, 1, 1> D_A1_Resid = ndarray::allocate(I_NElements);
      D_A1_Resid.deep() = arr_In;
      D_A1_Resid.deep() -= D_Mean;

      Eigen::Array<T, Eigen::Dynamic, 1> E_A1_Resid = D_A1_Resid.asEigen();
//      T sum = E_A1_Resid.sum();
      D_Var = (E_A1_Resid.pow(2).sum() - pow(E_A1_Resid.sum(), 2)/T(I_NElements)) / (T(I_NElements)-1.);
      D_A1_Out[1] = D_Var;
      if (maxMoment_In <= 2)
        return D_A1_Out;
      T D_SDev = 0.;
      D_SDev = sqrt(D_Var);

      if (D_SDev != 0.){
        D_Skew = E_A1_Resid.pow(3).sum() / (I_NElements * pow(D_SDev,3));
        D_A1_Out[2] = D_Skew;

        if (maxMoment_In <= 3)
          return D_A1_Out;
        D_Kurt = E_A1_Resid.pow(4).sum() / (I_NElements * pow(D_SDev,4)) - 3.;
        D_A1_Out[3] = D_Kurt;
      }
      return D_A1_Out;
    }
    
    template<typename T>
    ndarray::Array<T, 1, 1> getSubArray(ndarray::Array<T, 1, 1> const& arr_In, 
                                        ndarray::Array<size_t, 1, 1> const& indices_In){
      ndarray::Array<T, 1, 1> arr_Out = ndarray::allocate(indices_In.getShape()[0]);
      for (int ind = 0; ind < indices_In.getShape()[0]; ++ind){
        arr_Out[ind] = arr_In[indices_In[ind]];
      }
      return arr_Out;
    }

    template<typename T>
    ndarray::Array<T, 1, 1> getSubArray(ndarray::Array<T, 2, 2> const& arr_In, 
                                        ndarray::Array<size_t, 2, 2> const& indices_In){
      ndarray::Array<T, 1, 1> arr_Out = ndarray::allocate(indices_In.getShape()[0]);
      for (size_t iRow = 0; iRow < indices_In.getShape()[0]; ++iRow){
        arr_Out[iRow] = arr_In[indices_In[iRow][0]][indices_In[iRow][1]];
        #ifdef __DEBUG_GETSUBARRAY__
          cout << "getSubArray: arr_Out[" << iRow << "] = " << arr_Out[iRow] << endl;
        #endif
      }
      return arr_Out;
    }

    template<typename T>
    ndarray::Array<T, 1, 1> getSubArray(ndarray::Array<T, 2, 2> const& arr_In, 
                                        std::vector< std::pair<size_t, size_t> > const& indices_In){
//      cout << "getSubArray: arr_In = " << arr_In << endl;
      ndarray::Array<T, 1, 1> arr_Out = ndarray::allocate(indices_In.size());
      for (size_t iRow = 0; iRow < indices_In.size(); ++iRow){
        arr_Out[iRow] = arr_In[indices_In[iRow].first][indices_In[iRow].second];
        #ifdef __DEBUG_GETSUBARRAY__
          cout << "getSubArray: arr_Out[" << iRow << "] = " << arr_Out[iRow] << endl;
        #endif
      }
      return arr_Out;
    }
    
    template< typename T >
    ndarray::Array< T, 1, 1 > resize(ndarray::Array< T, 1, 1 > const& arr_In, size_t newSize){
      ndarray::Array< T, 1, 1 > arrOut = ndarray::allocate(newSize);
      arrOut.deep() = 0;
      return arrOut;
    }

    template<typename T>
    std::vector<int> sortIndices(const std::vector<T> &vec_In){
      #ifdef __DEBUG_FITS_SORT__
        cout << "CFits::SortIndices(vec_In = " << vec_In << ") started" << endl;
      #endif
      std::vector<T> vec(vec_In.size());
      vec = vec_In;
      blitz::Array<T, 1> D_A1_In(vec.data(), blitz::shape(vec.size()), blitz::neverDeleteData);
      int I_M = 7;
      int I_NStack = 50;
      blitz::firstIndex i;

      int I_I, I_Indxt, I_Ir, I_J, I_K, I_L, I_SizeIn;
      int I_JStack = 0;
      blitz::Array<int, 1> I_A1_IStack(I_NStack);
      blitz::Array<int, 1> I_A1_Indx;
      T D_A;

      I_SizeIn = D_A1_In.size();
      I_Ir = I_SizeIn - 1;
      I_L = 0;

      I_A1_IStack = 0;
      I_A1_Indx.resize(I_SizeIn);
      I_A1_Indx = i;

      #ifdef __DEBUG_FITS_SORT__
        cout << "CFits::SortIndices() starting for(;;)" << endl;
      #endif
      for(;;)
      {
        if (I_Ir - I_L < I_M)
        {
          for (I_J = I_L + 1; I_J <= I_Ir; I_J++)
          {
            I_Indxt = I_A1_Indx(I_J);
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(): I_Indxt set to " << I_Indxt << endl;
            #endif
            D_A = D_A1_In(I_Indxt);
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(): D_A set to " << D_A << endl;
            #endif
            for (I_I = I_J - 1; I_I >= I_L; I_I--)
            {
              if (D_A1_In(I_A1_Indx(I_I)) <= D_A)
              {
                #ifdef __DEBUG_FITS_SORT__
                  cout << "CFits::SortIndices(): D_A1_In(P_I_A1_Indx(I_I = " << I_I << ") = " << I_A1_Indx(I_I) << " <= D_A = " << D_A << " =>  BREAK" << endl;
                #endif
                break;
              }
              I_A1_Indx(I_I + 1) = I_A1_Indx(I_I);
              #ifdef __DEBUG_FITS_SORT__
                cout << "CFits::SortIndices(): 1. P_I_A1_Indx(I_I+1 = " << I_I + 1 << ") set to " << I_A1_Indx(I_I+1) << " => P_I_A1_Indx = " << I_A1_Indx << endl;
              #endif

            }
            I_A1_Indx(I_I + 1) = I_Indxt;
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(): 2. P_I_A1_Indx(I_I+1 = " << I_I + 1 << ") set to " << I_A1_Indx(I_I+1) << " => P_I_A1_Indx = " << I_A1_Indx << endl;
            #endif

          }
          if (I_JStack == 0)
          {
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(): I_JStack <= 0 =>  BREAK" << endl;
            #endif
            break;
          }
          I_Ir = I_A1_IStack(I_JStack--);
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(): I_Ir(=" << I_Ir << ") set to I_A1_IStack(I_JStack--=" << I_JStack << ") = " << I_A1_IStack(I_JStack) << endl;
          #endif
          I_L  = I_A1_IStack(I_JStack--);
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(): I_L(=" << I_L << ") set to I_A1_IStack(I_JStack--=" << I_JStack << ") = " << I_A1_IStack(I_JStack) << endl;
          #endif

        }
        else
        {
          I_K = (I_L + I_Ir) >> 1;
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(): I_K(=" << I_K << ") set to (I_L[=" << I_L << "] + I_Ir[=" << I_Ir << "] >> 1)  = " << ((I_L + I_Ir) >> 1) << endl;
          #endif
          std::swap(I_A1_Indx(I_K),
               I_A1_Indx(I_L + 1));
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): P_I_A1_Indx(I_K=" << I_K << ")=" << I_A1_Indx(I_K) << " and P_I_A1_Indx(I_L(=" << I_L << ")+1)=" << I_A1_Indx(I_L+1) << " std::swapped" << " => P_I_A1_Indx = " << I_A1_Indx << endl;
          #endif
          if (D_A1_In(I_A1_Indx(I_L))
            > D_A1_In(I_A1_Indx(I_Ir)))
          {
            std::swap(I_A1_Indx(I_L),
                 I_A1_Indx(I_Ir));
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): P_I_A1_Indx(I_L=" << I_L << ")=" << I_A1_Indx(I_L) << " and P_I_A1_Indx(I_Ir(=" << I_Ir << "))=" << I_A1_Indx(I_Ir) << " std::swapped" << " => P_I_A1_Indx = " << I_A1_Indx << endl;
            #endif

          }
          if (D_A1_In(I_A1_Indx(I_L + 1))
            > D_A1_In(I_A1_Indx(I_Ir)))
          {
            std::swap(I_A1_Indx(I_L + 1),
                 I_A1_Indx(I_Ir));
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): P_I_A1_Indx(I_L=" << I_L << "+1)=" << I_A1_Indx(I_L + 1) << " and P_I_A1_Indx(I_Ir(=" << I_Ir << "))=" << I_A1_Indx(I_L+1) << " std::swapped" << " => P_I_A1_Indx = " << I_A1_Indx << endl;
            #endif

          }
          if (D_A1_In(I_A1_Indx(I_L))
            > D_A1_In(I_A1_Indx(I_L + 1)))
          {
            std::swap(I_A1_Indx(I_L),
                 I_A1_Indx(I_L + 1));
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): P_I_A1_Indx(I_L=" << I_L << ")=" << I_A1_Indx(I_L) << " and P_I_A1_Indx(I_L(=" << I_L << ")+1)=" << I_A1_Indx(I_L+1) << " std::swapped" << " => P_I_A1_Indx = " << I_A1_Indx << endl;
            #endif

          }
          I_I = I_L + 1;
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): I_I(=" << I_I << ") set to (I_L[=" << I_L << "] + 1)  = " << I_L + 1 << endl;
          #endif
          I_J = I_Ir;
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): I_J(=" << I_J << ") set to I_Ir[=" << I_Ir << "]" << endl;
          #endif
          I_Indxt = I_A1_Indx(I_L + 1);
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): I_Indxt(=" << I_Indxt << ") set to P_I_A1_Indx(I_L = " << I_L << "+1)" << endl;
          #endif
          D_A = D_A1_In(I_Indxt);
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): D_A(=" << D_A << ") set to D_A1_In(I_Indxt = " << I_Indxt << ")" << endl;
          #endif
          for (;;)
          {
            do
            {
              I_I++;
              #ifdef __DEBUG_FITS_SORT__
                cout << "CFits::SortIndices(D_A1_In): I_I set to " << I_I << " => D_A1_In(P_I_A1_Indx(I_I)) = " << D_A1_In(I_A1_Indx(I_I)) << endl;
              #endif

            }
            while(D_A1_In(I_A1_Indx(I_I)) < D_A && I_I < I_SizeIn - 2);
            do
            {
              I_J--;
              #ifdef __DEBUG_FITS_SORT__
                cout << "CFits::SortIndices(D_A1_In): I_J set to " << I_J << " => D_A1_In(P_I_A1_Indx(I_J)) = " << D_A1_In(I_A1_Indx(I_J)) << endl;
              #endif

            }
            while(D_A1_In(I_A1_Indx(I_J)) > D_A && I_J > 0);
            if (I_J < I_I)
            {
              #ifdef __DEBUG_FITS_SORT__
                cout << "CFits::SortIndices(D_A1_In): I_J(=" << I_J << ") < I_I(=" << I_I << ") => BREAK" << endl;
              #endif
              break;
            }
            std::swap(I_A1_Indx(I_I),
                 I_A1_Indx(I_J));
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): P_I_A1_Indx(I_I=" << I_I << ")=" << I_A1_Indx(I_I) << " and P_I_A1_Indx(I_J(=" << I_J << "))=" << I_A1_Indx(I_J) << " std::swapped" << " => P_I_A1_Indx = " << I_A1_Indx << endl;
            #endif

          }
          I_A1_Indx(I_L + 1) = I_A1_Indx(I_J);
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): P_I_A1_Indx(I_L=" << I_L << "+1) set to P_I_A1_Indx(I_J=" << I_J << ") = " << I_A1_Indx(I_L+1) << ") => P_I_A1_Indx = " << I_A1_Indx << endl;
          #endif
          I_A1_Indx(I_J) = I_Indxt;
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): P_I_A1_Indx(I_J=" << I_J << ") set to I_Indxt(=" << I_Indxt << ") => P_I_A1_Indx = " << I_A1_Indx << endl;
          #endif
          I_JStack += 2;
          #ifdef __DEBUG_FITS_SORT__
            cout << "CFits::SortIndices(D_A1_In): I_JStack = " << I_JStack << endl;
          #endif
          if (I_JStack > I_NStack)
          {
            cout << "CFits::SortIndices: ERROR: I_NStack ( = " << I_NStack << ") too small!!!";
            exit(EXIT_FAILURE);
          }
          if (I_Ir - I_I + 1 >= I_J - I_L)
          {
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): I_Ir(= " << I_Ir << ") - I_I(=" << I_I << ") + 1 = " << I_Ir - I_I + 1 << " >= I_J(="<< I_J << ") + I_L(=" << I_L << ") = " << I_J - I_L << endl;
            #endif
            I_A1_IStack(I_JStack) = I_Ir;
            I_A1_IStack(I_JStack - 1) = I_I;
            I_Ir = I_J - 1;
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): I_I set to I_J(=" << I_J << ") - 1" << endl;
            #endif

          }
          else
          {
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): I_Ir(= " << I_Ir << ") - I_I(=" << I_I << ") + 1 = " << I_Ir - I_I + 1 << " < I_J(="<< I_J << ") + I_L(=" << I_L << ") = " << I_J - I_L << endl;
            #endif
            I_A1_IStack(I_JStack) = I_J - 1;
            I_A1_IStack(I_JStack - 1) = I_L;
            I_L = I_I;
            #ifdef __DEBUG_FITS_SORT__
              cout << "CFits::SortIndices(D_A1_In): I_L set to I_I(=" << I_I << endl;
            #endif

          }
        }
      }
      I_A1_IStack.resize(0);
      std::vector<int> vecIndx(I_A1_Indx.size());
      for (int i = 0; i < static_cast<int>(I_A1_Indx.size()); ++i){
        vecIndx[i] = I_A1_Indx(i);
      }
      return (vecIndx);
    }
    
    template ndarray::Array< size_t, 1, 1 > resize( ndarray::Array< size_t, 1, 1 > const& arr_In, size_t newSize);
    template ndarray::Array< short, 1, 1 > resize( ndarray::Array< short, 1, 1 > const& arr_In, size_t newSize);
    template ndarray::Array< int, 1, 1 > resize( ndarray::Array< int, 1, 1 > const& arr_In, size_t newSize);
    template ndarray::Array< long, 1, 1 > resize( ndarray::Array< long, 1, 1 > const& arr_In, size_t newSize);
    template ndarray::Array< float, 1, 1 > resize( ndarray::Array< float, 1, 1 > const& arr_In, size_t newSize);
    template ndarray::Array< double, 1, 1 > resize( ndarray::Array< double, 1, 1 > const& arr_In, size_t newSize);

    template ndarray::Array<size_t, 1, 1> getSubArray(ndarray::Array<size_t, 1, 1> const&, ndarray::Array<size_t, 1, 1> const&);
    template ndarray::Array<int, 1, 1> getSubArray(ndarray::Array<int, 1, 1> const&, ndarray::Array<size_t, 1, 1> const&);
    template ndarray::Array<long, 1, 1> getSubArray(ndarray::Array<long, 1, 1> const&, ndarray::Array<size_t, 1, 1> const&);
    template ndarray::Array<float, 1, 1> getSubArray(ndarray::Array<float, 1, 1> const&, ndarray::Array<size_t, 1, 1> const&);
    template ndarray::Array<double, 1, 1> getSubArray(ndarray::Array<double, 1, 1> const&, ndarray::Array<size_t, 1, 1> const&);

    template ndarray::Array<size_t, 1, 1> getSubArray(ndarray::Array<size_t, 2, 2> const&, ndarray::Array<size_t, 2, 2> const&);
    template ndarray::Array<int, 1, 1> getSubArray(ndarray::Array<int, 2, 2> const&, ndarray::Array<size_t, 2, 2> const&);
    template ndarray::Array<long, 1, 1> getSubArray(ndarray::Array<long, 2, 2> const&, ndarray::Array<size_t, 2, 2> const&);
    template ndarray::Array<float, 1, 1> getSubArray(ndarray::Array<float, 2, 2> const&, ndarray::Array<size_t, 2, 2> const&);
    template ndarray::Array<double, 1, 1> getSubArray(ndarray::Array<double, 2, 2> const&, ndarray::Array<size_t, 2, 2> const&);

    template ndarray::Array<size_t, 1, 1> getSubArray(ndarray::Array<size_t, 2, 2> const&, std::vector< std::pair<size_t, size_t> > const&);
    template ndarray::Array<int, 1, 1> getSubArray(ndarray::Array<int, 2, 2> const&, std::vector< std::pair<size_t, size_t> > const&);
    template ndarray::Array<long, 1, 1> getSubArray(ndarray::Array<long, 2, 2> const&, std::vector< std::pair<size_t, size_t> > const&);
    template ndarray::Array<float, 1, 1> getSubArray(ndarray::Array<float, 2, 2> const&, std::vector< std::pair<size_t, size_t> > const&);
    template ndarray::Array<double, 1, 1> getSubArray(ndarray::Array<double, 2, 2> const&, std::vector< std::pair<size_t, size_t> > const&);

    template std::vector<size_t> removeSubArrayFromArray(std::vector<size_t> const&, std::vector<size_t> const&);
    template std::vector<int> removeSubArrayFromArray(std::vector<int> const&, std::vector<int> const&);
    template std::vector<long> removeSubArrayFromArray(std::vector<long> const&, std::vector<long> const&);
    template std::vector<float> removeSubArrayFromArray(std::vector<float> const&, std::vector<float> const&);
    template std::vector<double> removeSubArrayFromArray(std::vector<double> const&, std::vector<double> const&);

    template ndarray::Array<size_t, 1, 1> getIndicesInValueRange(ndarray::Array<size_t, 1, 1> const&, size_t const, size_t const);
    template ndarray::Array<size_t, 1, 1> getIndicesInValueRange(ndarray::Array<int, 1, 1> const&, int const, int const);
    template ndarray::Array<size_t, 1, 1> getIndicesInValueRange(ndarray::Array<long, 1, 1> const&, long const, long const);
    template ndarray::Array<size_t, 1, 1> getIndicesInValueRange(ndarray::Array<float, 1, 1> const&, float const, float const);
    template ndarray::Array<size_t, 1, 1> getIndicesInValueRange(ndarray::Array<double, 1, 1> const&, double const, double const);

    template ndarray::Array<size_t, 2, 2> getIndicesInValueRange(ndarray::Array<size_t, 2, 2> const&, size_t const, size_t const);
    template ndarray::Array<size_t, 2, 2> getIndicesInValueRange(ndarray::Array<int, 2, 2> const&, int const, int const);
    template ndarray::Array<size_t, 2, 2> getIndicesInValueRange(ndarray::Array<long, 2, 2> const&, long const, long const);
    template ndarray::Array<size_t, 2, 2> getIndicesInValueRange(ndarray::Array<float, 2, 2> const&, float const, float const);
    template ndarray::Array<size_t, 2, 2> getIndicesInValueRange(ndarray::Array<double, 2, 2> const&, double const, double const);
    
    template ndarray::Array<size_t, 1, 1> replicate(size_t const val, int const size);
    template ndarray::Array<unsigned short, 1, 1> replicate(unsigned short const val, int const size);
    template ndarray::Array<int, 1, 1> replicate(int const val, int const size);
    template ndarray::Array<long, 1, 1> replicate(long const val, int const size);
    template ndarray::Array<float, 1, 1> replicate(float const val, int const size);
    template ndarray::Array<double, 1, 1> replicate(double const val, int const size);

    template ndarray::Array<size_t, 1, 1> indGenNdArr(size_t const);
    template ndarray::Array<unsigned short, 1, 1> indGenNdArr(unsigned short const);
    template ndarray::Array<int, 1, 1> indGenNdArr(int const);
    template ndarray::Array<long, 1, 1> indGenNdArr(long const);
    template ndarray::Array<float, 1, 1> indGenNdArr(float const);
    template ndarray::Array<double, 1, 1> indGenNdArr(double const);

    template ndarray::Array<double, 1, 1> Double(ndarray::Array<size_t, 1, 1> const&);
    template ndarray::Array<double, 1, 1> Double(ndarray::Array<unsigned short, 1, 1> const&);
    template ndarray::Array<double, 1, 1> Double(ndarray::Array<int, 1, 1> const&);
    template ndarray::Array<double, 1, 1> Double(ndarray::Array<long, 1, 1> const&);
    template ndarray::Array<double, 1, 1> Double(ndarray::Array<float, 1, 1> const&);
    template ndarray::Array<double, 1, 1> Double(ndarray::Array<float const, 1, 1> const&);
    template ndarray::Array<double, 1, 1> Double(ndarray::Array<double, 1, 1> const&);

    template ndarray::Array<float, 1, 1> Float(ndarray::Array<size_t, 1, 1> const&);
    template ndarray::Array<float, 1, 1> Float(ndarray::Array<unsigned short, 1, 1> const&);
    template ndarray::Array<float, 1, 1> Float(ndarray::Array<int, 1, 1> const&);
    template ndarray::Array<float, 1, 1> Float(ndarray::Array<long, 1, 1> const&);
    template ndarray::Array<float, 1, 1> Float(ndarray::Array<float, 1, 1> const&);
    template ndarray::Array<float, 1, 1> Float(ndarray::Array<double, 1, 1> const&);

    template ndarray::Array<double, 2, 2> Double(ndarray::Array<size_t, 2, 2> const&);
    template ndarray::Array<double, 2, 2> Double(ndarray::Array<unsigned short, 2, 2> const&);
    template ndarray::Array<double, 2, 2> Double(ndarray::Array<int, 2, 2> const&);
    template ndarray::Array<double, 2, 2> Double(ndarray::Array<long, 2, 2> const&);
    template ndarray::Array<double, 2, 2> Double(ndarray::Array<float, 2, 2> const&);
    template ndarray::Array<double, 2, 2> Double(ndarray::Array<float const, 2, 2> const&);
    template ndarray::Array<double, 2, 2> Double(ndarray::Array<double, 2, 2> const&);

    template ndarray::Array<float, 2, 2> Float(ndarray::Array<size_t, 2, 2> const&);
    template ndarray::Array<float, 2, 2> Float(ndarray::Array<unsigned short, 2, 2> const&);
    template ndarray::Array<float, 2, 2> Float(ndarray::Array<int, 2, 2> const&);
    template ndarray::Array<float, 2, 2> Float(ndarray::Array<long, 2, 2> const&);
    template ndarray::Array<float, 2, 2> Float(ndarray::Array<float, 2, 2> const&);
    template ndarray::Array<float, 2, 2> Float(ndarray::Array<double, 2, 2> const&);

//    template ndarray::Array<int, 1, 1> Int(ndarray::Array<size_t, 1, 1> const&);
//    template ndarray::Array<int, 1, 1> Int(ndarray::Array<unsigned short, 1, 1> const&);
//    template ndarray::Array<int, 1, 1> Int(ndarray::Array<long, 1, 1> const&);
//    template ndarray::Array<int, 1, 1> Int(ndarray::Array<float, 1, 1> const&);
//    template ndarray::Array<int, 1, 1> Int(ndarray::Array<double, 1, 1> const&);
    
    template size_t min(ndarray::Array<size_t, 1, 1> const&);
    template unsigned short min(ndarray::Array<unsigned short, 1, 1> const&);
    template int min(ndarray::Array<int, 1, 1> const&);
    template long min(ndarray::Array<long, 1, 1> const&);
    template float min(ndarray::Array<float, 1, 1> const&);
    template double min(ndarray::Array<double, 1, 1> const&);
    
    template size_t max(ndarray::Array<size_t, 1, 1> const&);
    template unsigned short max(ndarray::Array<unsigned short, 1, 1> const&);
    template int max(ndarray::Array<int, 1, 1> const&);
    template long max(ndarray::Array<long, 1, 1> const&);
    template float max(ndarray::Array<float, 1, 1> const&);
    template double max(ndarray::Array<double, 1, 1> const&);

    template size_t floor1(float const&, size_t const&);
    template size_t floor1(double const&, size_t const&);
    template unsigned int floor1(float const&, unsigned int const&);
    template unsigned int floor1(double const&, unsigned int const&);
//    template unsigned long math::floor(float, unsigned long);
//    template unsigned long math::floor(double, unsigned long);
    template float floor1(float const&, float const&);
    template float floor1(double const&, float const&);
    template double floor1(float const&, double const&);
    template double floor1(double const&, double const&);

    template ndarray::Array<size_t, 1, 1> floor(const ndarray::Array<const float, 1, 1>&, const size_t);
    template ndarray::Array<size_t, 1, 1> floor(const ndarray::Array<const double, 1, 1>&, const size_t);
    template ndarray::Array<unsigned int, 1, 1> floor(const ndarray::Array<const float, 1, 1>&, const unsigned int);
    template ndarray::Array<unsigned int, 1, 1> floor(const ndarray::Array<const double, 1, 1>&, const unsigned int);
  //  template ndarray::Array<unsigned long, 1, 1> math::floor(const ndarray::Array<const float, 1, 1>&, const unsigned long);
  //  template ndarray::Array<unsigned long, 1, 1> math::floor(const ndarray::Array<const double, 1, 1>&, const unsigned long);
    template ndarray::Array<float, 1, 1> floor(const ndarray::Array<const float, 1, 1>&, const float);
    template ndarray::Array<float, 1, 1> floor(const ndarray::Array<const double, 1, 1>&, const float);
    template ndarray::Array<double, 1, 1> floor(const ndarray::Array<const float, 1, 1>&, const double);
    template ndarray::Array<double, 1, 1> floor(const ndarray::Array<const double, 1, 1>&, const double);

    template ndarray::Array<size_t, 2, 2> floor(const ndarray::Array<const float, 2, 2>&, const size_t);
    template ndarray::Array<size_t, 2, 2> floor(const ndarray::Array<const double, 2, 2>&, const size_t);
    template ndarray::Array<unsigned int, 2, 2> floor(const ndarray::Array<const float, 2, 2>&, const unsigned int);
    template ndarray::Array<unsigned int, 2, 2> floor(const ndarray::Array<const double, 2, 2>&, const unsigned int);
  //  template ndarray::Array<unsigned long math::floor(ndarray::Array<float, 2, 2>, unsigned long);
  //  template ndarray::Array<unsigned long math::floor(ndarray::Array<double, 2, 2>, unsigned double);
    template ndarray::Array<float, 2, 2> floor(const ndarray::Array<const float, 2, 2>&, const float);
    template ndarray::Array<float, 2, 2> floor(const ndarray::Array<const double, 2, 2>&, const float);
    template ndarray::Array<double, 2, 2> floor(const ndarray::Array<const float, 2, 2>&, const double);
    template ndarray::Array<double, 2, 2> floor(const ndarray::Array<const double, 2, 2>&, const double);

    template ndarray::Array<float, 1, 1> Poly(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&);
    template ndarray::Array<float, 1, 1> Poly(ndarray::Array<float, 1, 1> const&, ndarray::Array<double, 1, 1> const&);
    template ndarray::Array<double, 1, 1> Poly(ndarray::Array<double, 1, 1> const&, ndarray::Array<float, 1, 1> const&);
    template ndarray::Array<double, 1, 1> Poly(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&);
  
    template int firstIndexWithZeroValueFrom(ndarray::Array<unsigned short, 1, 1> const& vec_In,
                                             const int startPos_In);
    template int firstIndexWithZeroValueFrom(ndarray::Array<unsigned int, 1, 1> const& vec_In,
                                             const int startPos_In);
    template int firstIndexWithZeroValueFrom(ndarray::Array<int, 1, 1> const& vec_In,
                                             const int startPos_In);
    template int firstIndexWithZeroValueFrom(ndarray::Array<long, 1, 1> const& vec_In,
                                             const int startPos_In);
    template int firstIndexWithZeroValueFrom(ndarray::Array<float, 1, 1> const& vec_In,
                                             const int startPos_In);
    template int firstIndexWithZeroValueFrom(ndarray::Array<double, 1, 1> const& vec_In,
                                             const int startPos_In);
    
    template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const, float const, std::vector<string> const&, std::vector<void *> &);
    template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&, size_t const, double const, std::vector<string> const&, std::vector<void *> &);
    template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const, float const, float const, size_t const, std::vector<string> const&, std::vector<void *> &);
    template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&, size_t const, double const, double const, size_t const, std::vector<string> const&, std::vector<void *> &);
    template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const, std::vector<string> const&, std::vector<void *> &);
    template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&, size_t const, std::vector<string> const&, std::vector<void *> &);
    template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const);
    template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&, size_t const);
    template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const, float const);
    template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&, size_t const, double const);
    template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&, size_t const, float const, float const, size_t const);
    template ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&, size_t const, double const, double const, size_t const);
  
    template int Fix(unsigned short);
    template int Fix(unsigned int);
    template int Fix(int);
    template int Fix(long);
    template int Fix(float);
    template int Fix(double);

    template long FixL(unsigned short);
    template long FixL(unsigned int);
    template long FixL(int);
    template long FixL(long);
    template long FixL(float);
    template long FixL(double);

    template int Int(unsigned short);
    template int Int(unsigned int);
    template int Int(int);
    template int Int(long);
    template int Int(float);
    template int Int(double);

    template long Long(unsigned short);
    template long Long(unsigned int);
    template long Long(int);
    template long Long(long);
    template long Long(float);
    template long Long(double);

    template int Round(const unsigned short ToRound);
    template int Round(const unsigned int ToRound);
    template int Round(const int ToRound);
    template int Round(const long ToRound);
    template int Round(const float ToRound);
    template int Round(const double ToRound);

    template unsigned short Round(const unsigned short ToRound, int DigitsBehindDot);
    template unsigned int Round(const unsigned int ToRound, int DigitsBehindDot);
    template int Round(const int ToRound, int DigitsBehindDot);
    template long Round(const long ToRound, int DigitsBehindDot);
    template float Round(const float ToRound, int DigitsBehindDot);
    template double Round(const double ToRound, int DigitsBehindDot);

    template long RoundL(const unsigned short ToRound);
    template long RoundL(const unsigned int ToRound);
    template long RoundL(const int ToRound);
    template long RoundL(const long ToRound);
    template long RoundL(const float ToRound);
    template long RoundL(const double ToRound);

    template bool countPixGTZero(ndarray::Array<unsigned short, 1, 1> &vec_InOut);
    template bool countPixGTZero(ndarray::Array<unsigned int, 1, 1> &vec_InOut);
    template bool countPixGTZero(ndarray::Array<int, 1, 1> &vec_InOut);
    template bool countPixGTZero(ndarray::Array<long, 1, 1> &vec_InOut);
    template bool countPixGTZero(ndarray::Array<float, 1, 1> &vec_InOut);
    template bool countPixGTZero(ndarray::Array<double, 1, 1> &vec_InOut);

    template int firstIndexWithValueGEFrom(ndarray::Array<unsigned short, 1, 1> const& vecIn,
                                                 const unsigned short minValue,
                                                 const int fromIndex);
    template int firstIndexWithValueGEFrom(ndarray::Array<unsigned int, 1, 1> const& vecIn,
                                                 const unsigned int minValue,
                                                 const int fromIndex);
    template int firstIndexWithValueGEFrom(ndarray::Array<int, 1, 1> const& vecIn,
                                                 const int minValue,
                                                 const int fromIndex);
    template int firstIndexWithValueGEFrom(ndarray::Array<long, 1, 1> const& vecIn,
                                                 const long minValue,
                                                 const int fromIndex);
    template int firstIndexWithValueGEFrom(ndarray::Array<float, 1, 1> const& vecIn,
                                                 const float minValue,
                                                 const int fromIndex);
    template int firstIndexWithValueGEFrom(ndarray::Array<double, 1, 1> const& vecIn,
                                                 const double minValue,
                                                 const int fromIndex);

    template int lastIndexWithZeroValueBefore(ndarray::Array<unsigned short, 1, 1> const& vec_In,
                                                    const int startPos_In);
    template int lastIndexWithZeroValueBefore(ndarray::Array<unsigned int, 1, 1> const& vec_In,
                                                    const int startPos_In);
    template int lastIndexWithZeroValueBefore(ndarray::Array<int, 1, 1> const& vec_In,
                                                    const int startPos_In);
    template int lastIndexWithZeroValueBefore(ndarray::Array<long, 1, 1> const& vec_In,
                                                    const int startPos_In);
    template int lastIndexWithZeroValueBefore(ndarray::Array<float, 1, 1> const& vec_In,
                                                    const int startPos_In);
    template int lastIndexWithZeroValueBefore(ndarray::Array<double, 1, 1> const& vec_In,
                                                    const int startPos_In);

    template ndarray::Array<size_t, 1, 1> moment(const ndarray::Array<size_t, 1, 1> &D_A1_Arr_In, int I_MaxMoment_In);
    template ndarray::Array<int, 1, 1> moment(const ndarray::Array<int, 1, 1> &D_A1_Arr_In, int I_MaxMoment_In);
    template ndarray::Array<long, 1, 1> moment(const ndarray::Array<long, 1, 1> &D_A1_Arr_In, int I_MaxMoment_In);
    template ndarray::Array<float, 1, 1> moment(const ndarray::Array<float, 1, 1> &D_A1_Arr_In, int I_MaxMoment_In);
    template ndarray::Array<double, 1, 1> moment(const ndarray::Array<double, 1, 1> &D_A1_Arr_In, int I_MaxMoment_In);

    template std::vector<int> sortIndices(const std::vector<unsigned short> &vec_In);
    template std::vector<int> sortIndices(const std::vector<unsigned int> &vec_In);
    template std::vector<int> sortIndices(const std::vector<int> &vec_In);
    template std::vector<int> sortIndices(const std::vector<long> &vec_In);
    template std::vector<int> sortIndices(const std::vector<float> &vec_In);
    template std::vector<int> sortIndices(const std::vector<double> &vec_In);

    template std::vector<unsigned short> indGen(unsigned short);
    template std::vector<unsigned int> indGen(unsigned int);
    template std::vector<int> indGen(int);
    template std::vector<float> indGen(float);
    template std::vector<double> indGen(double);

    template int LinFitBevingtonEigen(Eigen::Array<float, Eigen::Dynamic, 1> const& D_A1_CCD_In,
                                            Eigen::Array<float, Eigen::Dynamic, 1> const& D_A1_SF_In,       /// xvec: in
                                            float &D_SP_Out,                         /// a1: out
                                            float &D_Sky_Out,                        /// a0: in/out
                                            bool B_WithSky,                        /// with sky: in
                                            std::vector<string> const& S_A1_Args_In,   ///: in
                                            std::vector<void *> &ArgV_In);                    ///: in
    template int LinFitBevingtonEigen(Eigen::Array<double, Eigen::Dynamic, 1> const& D_A1_CCD_In,
                                            Eigen::Array<float, Eigen::Dynamic, 1> const& D_A1_SF_In,       /// xvec: in
                                            double &D_SP_Out,                         /// a1: out
                                            double &D_Sky_Out,                        /// a0: in/out
                                            bool B_WithSky,                        /// with sky: in
                                            std::vector<string> const& S_A1_Args_In,   ///: in
                                            std::vector<void *> &ArgV_In);                    ///: in
    template int LinFitBevingtonEigen(Eigen::Array<float, Eigen::Dynamic, 1> const& D_A1_CCD_In,
                                            Eigen::Array<double, Eigen::Dynamic, 1> const& D_A1_SF_In,       /// xvec: in
                                            float &D_SP_Out,                         /// a1: out
                                            float &D_Sky_Out,                        /// a0: in/out
                                            bool B_WithSky,                        /// with sky: in
                                            std::vector<string> const& S_A1_Args_In,   ///: in
                                            std::vector<void *> &ArgV_In);                    ///: in
    template int LinFitBevingtonEigen(Eigen::Array<double, Eigen::Dynamic, 1> const& D_A1_CCD_In,
                                            Eigen::Array<double, Eigen::Dynamic, 1> const& D_A1_SF_In,       /// xvec: in
                                            double &D_SP_Out,                         /// a1: out
                                            double &D_Sky_Out,                        /// a0: in/out
                                            bool B_WithSky,                        /// with sky: in
                                            std::vector<string> const& S_A1_Args_In,   ///: in
                                            std::vector<void *> &ArgV_In);                    ///: in

    template int LinFitBevingtonNdArray(ndarray::Array<float, 1, 1> const& D_A1_CCD_In,
                                              ndarray::Array<float, 1, 1> const& D_A1_SF_In,       /// xvec: in
                                              float &D_SP_Out,                         /// a1: out
                                              float &D_Sky_Out,                        /// a0: in/out
                                              bool B_WithSky,                        /// with sky: in
                                              std::vector<string> const& S_A1_Args_In,   ///: in
                                              std::vector<void *> &ArgV_In);                    ///: in
    template int LinFitBevingtonNdArray(ndarray::Array<double, 1, 1> const& D_A1_CCD_In,
                                              ndarray::Array<float, 1, 1> const& D_A1_SF_In,       /// xvec: in
                                              double &D_SP_Out,                         /// a1: out
                                              double &D_Sky_Out,                        /// a0: in/out
                                              bool B_WithSky,                        /// with sky: in
                                              std::vector<string> const& S_A1_Args_In,   ///: in
                                              std::vector<void *> &ArgV_In);                    ///: in
    template int LinFitBevingtonNdArray(ndarray::Array<float, 1, 1> const& D_A1_CCD_In,
                                              ndarray::Array<double, 1, 1> const& D_A1_SF_In,       /// xvec: in
                                              float &D_SP_Out,                         /// a1: out
                                              float &D_Sky_Out,                        /// a0: in/out
                                              bool B_WithSky,                        /// with sky: in
                                              std::vector<string> const& S_A1_Args_In,   ///: in
                                              std::vector<void *> &ArgV_In);                    ///: in
    template int LinFitBevingtonNdArray(ndarray::Array<double, 1, 1> const& D_A1_CCD_In,
                                              ndarray::Array<double, 1, 1> const& D_A1_SF_In,       /// xvec: in
                                              double &D_SP_Out,                         /// a1: out
                                              double &D_Sky_Out,                        /// a0: in/out
                                              bool B_WithSky,                        /// with sky: in
                                              std::vector<string> const& S_A1_Args_In,   ///: in
                                              std::vector<void *> &ArgV_In);                    ///: in

    template bool LinFitBevingtonEigen(Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> const& D_A2_CCD_In,      /// yvec: in
                                             Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> const& D_A2_SF_In,       /// xvec: in
                                             Eigen::Array<float, Eigen::Dynamic, 1> & D_A1_SP_Out,                         /// a1: out
                                             Eigen::Array<float, Eigen::Dynamic, 1> & D_A1_Sky_Out,                        /// a0: out
                                             bool B_WithSky,                           /// with sky: in
                                             vector<string> const& S_A1_Args_In,   ///: in
                                             vector<void *> &ArgV_In);                    ///: in
    template bool LinFitBevingtonEigen(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> &D_A2_CCD_In,      /// yvec: in
                                             const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> &D_A2_SF_In,       /// xvec: in
                                             Eigen::Array<double, Eigen::Dynamic, 1> &D_A1_SP_Out,                         /// a1: out
                                             Eigen::Array<double, Eigen::Dynamic, 1> &D_A1_Sky_Out,                        /// a0: out
                                             bool B_WithSky,                           /// with sky: in
                                             const vector<string> &S_A1_Args_In,   ///: in
                                             vector<void *> &ArgV_In);                    ///: in
    template bool LinFitBevingtonEigen(const Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> &D_A2_CCD_In,      /// yvec: in
                                             const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> &D_A2_SF_In,       /// xvec: in
                                             Eigen::Array<float, Eigen::Dynamic, 1> &D_A1_SP_Out,                         /// a1: out
                                             Eigen::Array<float, Eigen::Dynamic, 1> &D_A1_Sky_Out,                        /// a0: out
                                             bool B_WithSky,                           /// with sky: in
                                             const vector<string> &S_A1_Args_In,   ///: in
                                             vector<void *> &ArgV_In);                    ///: in
    template bool LinFitBevingtonEigen(Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> const& D_A2_CCD_In,      /// yvec: in
                                             Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> const& D_A2_SF_In,       /// xvec: in
                                             Eigen::Array<double, Eigen::Dynamic, 1> & D_A1_SP_Out,                         /// a1: out
                                             Eigen::Array<double, Eigen::Dynamic, 1> & D_A1_Sky_Out,                        /// a0: out
                                             bool B_WithSky,                           /// with sky: in
                                             vector<string> const& S_A1_Args_In,   ///: in
                                             vector<void *> & ArgV_In);                    ///: in

    template bool LinFitBevingtonNdArray(ndarray::Array<float, 2, 1> const& D_A2_CCD_In,      /// yvec: in
                                               ndarray::Array<float, 2, 1> const& D_A2_SF_In,       /// xvec: in
                                               ndarray::Array<float, 1, 1> & D_A1_SP_Out,                         /// a1: out
                                               ndarray::Array<float, 1, 1> & D_A1_Sky_Out,                        /// a0: out
                                               bool B_WithSky,                           /// with sky: in
                                               vector<string> const& S_A1_Args_In,   ///: in
                                               vector<void *> &ArgV_In);                    ///: in
    template bool LinFitBevingtonNdArray(ndarray::Array<double, 2, 1> const& D_A2_CCD_In,      /// yvec: in
                                               ndarray::Array<float, 2, 1> const& D_A2_SF_In,       /// xvec: in
                                               ndarray::Array<double, 1, 1> & D_A1_SP_Out,                         /// a1: out
                                               ndarray::Array<double, 1, 1> & D_A1_Sky_Out,                        /// a0: out
                                               bool B_WithSky,                           /// with sky: in
                                               const vector<string> &S_A1_Args_In,   ///: in
                                               vector<void *> &ArgV_In);                    ///: in
    template bool LinFitBevingtonNdArray(ndarray::Array<float, 2, 1> const& D_A2_CCD_In,      /// yvec: in
                                               ndarray::Array<double, 2, 1> const& D_A2_SF_In,       /// xvec: in
                                               ndarray::Array<float, 1, 1> & D_A1_SP_Out,                         /// a1: out
                                               ndarray::Array<float, 1, 1> & D_A1_Sky_Out,                        /// a0: out
                                               bool B_WithSky,                           /// with sky: in
                                               const vector<string> &S_A1_Args_In,   ///: in
                                               vector<void *> &ArgV_In);                    ///: in
    template bool LinFitBevingtonNdArray(ndarray::Array<double, 2, 1> const& D_A2_CCD_In,      /// yvec: in
                                               ndarray::Array<double, 2, 1> const& D_A2_SF_In,       /// xvec: in
                                               ndarray::Array<double, 1, 1> & D_A1_SP_Out,                         /// a1: out
                                               ndarray::Array<double, 1, 1> & D_A1_Sky_Out,                        /// a0: out
                                               bool B_WithSky,                           /// with sky: in
                                               vector<string> const& S_A1_Args_In,   ///: in
                                               vector<void *> & ArgV_In);                    ///: in

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


  }/// end namespace math


}}}
