#include "pfs/drp/stella/math/Math.h"
  namespace pfs{ namespace drp{ namespace stella{ namespace math{
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
        cout << "calcMinCenMax: minCenMax_Out(*,1) = " << minCenMax_Out[ndarray::view()(1)] << endl;
      #endif
      ndarray::Array<const float, 1, 1> F_A1_Temp = ndarray::copy(xCenters_In + xLow_In);

      minCenMax_Out[ndarray::view()(0)] = pfs::drp::stella::math::floor(F_A1_Temp, size_t(0));

      #ifdef __DEBUG_MINCENMAX__
        cout << "calcMinCenMax: minCenMax_Out(*,0) = " << minCenMax_Out[ndarray::view()(0)] << endl;
      #endif
      F_A1_Temp = ndarray::copy(xCenters_In + xHigh_In);

      minCenMax_Out[ndarray::view()(2)] = pfs::drp::stella::math::floor(F_A1_Temp, size_t(0));

      #ifdef __DEBUG_MINCENMAX__
        cout << "calcMinCenMax: minCenMax_Out(*,2) = " << minCenMax_Out[ndarray::view()(2)] << endl;
      #endif

      ndarray::Array<size_t, 1, 1> I_A1_NPixLeft = ndarray::copy(minCenMax_Out[ndarray::view()(1)] - minCenMax_Out[ndarray::view()(0)]);
      ndarray::Array<size_t, 1, 1> I_A1_NPixRight = ndarray::copy(minCenMax_Out[ndarray::view()(2)] - minCenMax_Out[ndarray::view()(1)]);
      ndarray::Array<size_t, 1, 1> I_A1_I_NPixX = ndarray::copy(minCenMax_Out[ndarray::view()(2)] - minCenMax_Out[ndarray::view()(0)] + 1);

      #ifdef __DEBUG_MINCENMAX__
        cout << "calcMinCenMax: I_A1_NPixLeft(=" << I_A1_NPixLeft << endl;
        cout << "calcMinCenMax: I_A1_NPixRight(=" << I_A1_NPixRight << endl;
        cout << "calcMinCenMax: I_A1_I_NPixX = " << I_A1_I_NPixX << endl;
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
        cout << "calcMinCenMax: minCenMax_Out = " << minCenMax_Out << endl;
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
        cout << "InvertGaussJ: ERROR: InvertGaussJ(AArray=" << AArray << ", Unity=" << Unity << ") retuned FALSE" << endl;
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
          cout << "countPixGTZero: pos = " << pos << ": vec_InOut(pos) = " << *i << endl;
        #endif
        if (*i <= T(0))
          count = 0;
        else
          count++;
        #ifdef __DEBUG_COUNTPIXGTZERO__
          cout << "countPixGTZero: pos = " << pos << ": count set to " << count << endl;
        #endif
        *i = T(count);
        #ifdef __DEBUG_COUNTPIXGTZERO__
          cout << "countPixGTZero: pos = " << pos << ": vec_InOut(i) set to " << *i << endl;
        #endif
      }
      return true;
    }

    template<typename T>
    int firstIndexWithValueGEFrom(ndarray::Array<T, 1, 1> const& vec_In, const T minValue_In, const int fromIndex_In){
      if ((vec_In.getShape()[0] < 1) || (fromIndex_In >= int(vec_In.getShape()[0]))){
        cout << "pfs::drp::stella::math::firstIndexWithValueGEFrom: Error: vec_In.getShape()[0] =" << vec_In.getShape()[0] << " < 1 or fromIndex_In(=" << fromIndex_In << ") >= vec_In.getShape()[0] => Returning -1" << endl;
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
      for (auto i = vec_In.begin() + pos; i != vec_In.end(); ++i, ++pos){
        #ifdef __DEBUG_FINDANDTRACE__
          cout << "FirstIndexWithZeroValueFrom: pos = " << pos << endl;
          cout << "FirstIndexWithZeroValueFrom: I_A1_VecIn(pos) = " << *i << endl;
        #endif
        if (fabs(*i) < 0.00000000000000001)
          return pos;
      }
      return -1;
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
    size_t maxIndex(ndarray::Array<T, 1, 1> const& in){
      T max = in[0];
      size_t maxIndex = 0;
      size_t ind = 0;
      for (auto it = in.begin(); it != in.end(); ++it, ++ind){
        if (*it > max){
          max = *it;
          maxIndex = ind;
        }
      }
      return maxIndex;
    }

    template<typename T>
    T min(ndarray::Array<T, 1, 1> const& in){
      T min = in[0];
      for (auto it = in.begin(); it != in.end(); ++it){
        if (*it < min){
          min = *it;
        }
      }
      return min;
    }

    template<typename T>
    size_t minIndex(ndarray::Array<T, 1, 1> const& in){
      T min = in[0];
      size_t minIndex = 0;
      size_t ind = 0;
      for (auto it = in.begin(); it != in.end(); ++it, ++ind){
        if (*it < min){
          min = *it;
          minIndex = ind;
        }
      }
      return minIndex;
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
    ndarray::Array<double, 2, 1> Double(ndarray::Array<T, 2, 1> const& arr_In){
      ndarray::Array<double, 2, 1> arr_Out = ndarray::allocate(arr_In.getShape()[0], arr_In.getShape()[1]);
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
        cout << "Moment: ERROR: arr_In must contain 2 OR more elements." << endl;
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

    template<typename T>
    std::vector<int> sortIndices(const std::vector<T> &vec_In){
      int I_M = 7;
      int I_NStack = 50;

      int I_SizeIn = vec_In.size();
      int I_Ir = I_SizeIn - 1;
      int I_L = 0;

      int I_I, I_Indxt, I_J, I_K;
      int I_JStack = 0;
      std::vector<int> I_A1_IStack(I_NStack);
      std::vector<int> I_A1_Indx(I_SizeIn);
      for (int i = 0; i < I_SizeIn; ++i)
        I_A1_Indx[i] = i;
      for (auto it = I_A1_IStack.begin(); it != I_A1_IStack.end(); ++it)
        *it = 0;
      T D_A;

      #ifdef __DEBUG_SORT__
        cout << "SortIndices() starting for(;;)" << endl;
      #endif
      for(;;)
      {
        if (I_Ir - I_L < I_M)
        {
          for (I_J = I_L + 1; I_J <= I_Ir; I_J++)
          {
            I_Indxt = I_A1_Indx[I_J];
            #ifdef __DEBUG_SORT__
              cout << "SortIndices(): I_Indxt set to " << I_Indxt << endl;
            #endif
            D_A = vec_In[I_Indxt];
            #ifdef __DEBUG_SORT__
              cout << "SortIndices(): D_A set to " << D_A << endl;
            #endif
            for (I_I = I_J - 1; I_I >= I_L; I_I--)
            {
              if (vec_In[I_A1_Indx[I_I]] <= D_A)
              {
                #ifdef __DEBUG_SORT__
                  cout << "SortIndices(): vec_In[P_I_A1_Indx(I_I = " << I_I << ") = " << I_A1_Indx[I_I] << "] <= D_A = " << D_A << " =>  BREAK" << endl;
                #endif
                break;
              }
              I_A1_Indx[I_I + 1] = I_A1_Indx[I_I];
            }
            I_A1_Indx[I_I + 1] = I_Indxt;
          }
          if (I_JStack == 0)
          {
            #ifdef __DEBUG_SORT__
              cout << "SortIndices(): I_JStack <= 0 =>  BREAK" << endl;
            #endif
            break;
          }
          I_Ir = I_A1_IStack[I_JStack--];
          #ifdef __DEBUG_SORT__
            cout << "SortIndices(): I_Ir(=" << I_Ir << ") set to I_A1_IStack(I_JStack--=" << I_JStack << ") = " << I_A1_IStack[I_JStack] << endl;
          #endif
          I_L  = I_A1_IStack[I_JStack--];
          #ifdef __DEBUG_SORT__
            cout << "SortIndices(): I_L(=" << I_L << ") set to I_A1_IStack(I_JStack--=" << I_JStack << ") = " << I_A1_IStack[I_JStack] << endl;
          #endif

        }
        else
        {
          I_K = (I_L + I_Ir) >> 1;
          #ifdef __DEBUG_SORT__
            cout << "SortIndices(): I_K(=" << I_K << ") set to (I_L[=" << I_L << "] + I_Ir[=" << I_Ir << "] >> 1)  = " << ((I_L + I_Ir) >> 1) << endl;
          #endif
          std::swap(I_A1_Indx[I_K],
               I_A1_Indx[I_L + 1]);
          #ifdef __DEBUG_SORT__
            cout << "SortIndices(vec_In): P_I_A1_Indx(I_K=" << I_K << ")=" << I_A1_Indx[I_K] << " and P_I_A1_Indx(I_L(=" << I_L << ")+1)=" << I_A1_Indx[I_L+1] << " std::swapped" << endl;
          #endif
          if (vec_In[I_A1_Indx[I_L]]
            > vec_In[I_A1_Indx[I_Ir]])
          {
            std::swap(I_A1_Indx[I_L],
                 I_A1_Indx[I_Ir]);
            #ifdef __DEBUG_SORT__
              cout << "SortIndices(vec_In): P_I_A1_Indx(I_L=" << I_L << ")=" << I_A1_Indx[I_L] << " and P_I_A1_Indx(I_Ir(=" << I_Ir << "))=" << I_A1_Indx[I_Ir] << " std::swapped" << endl;
            #endif

          }
          if (vec_In[I_A1_Indx[I_L + 1]]
            > vec_In[I_A1_Indx[I_Ir]])
          {
            std::swap(I_A1_Indx[I_L + 1],
                 I_A1_Indx[I_Ir]);
            #ifdef __DEBUG_SORT__
              cout << "SortIndices(vec_In): P_I_A1_Indx(I_L=" << I_L << "+1)=" << I_A1_Indx[I_L + 1] << " and P_I_A1_Indx(I_Ir(=" << I_Ir << "))=" << I_A1_Indx[I_L+1] << " std::swapped" << endl;
            #endif

          }
          if (vec_In[I_A1_Indx[I_L]]
            > vec_In[I_A1_Indx[I_L + 1]])
          {
            std::swap(I_A1_Indx[I_L],
                 I_A1_Indx[I_L + 1]);
            #ifdef __DEBUG_SORT__
              cout << "SortIndices(vec_In): P_I_A1_Indx(I_L=" << I_L << ")=" << I_A1_Indx[I_L] << " and P_I_A1_Indx(I_L(=" << I_L << ")+1)=" << I_A1_Indx[I_L+1] << " std::swapped" << endl;
            #endif

          }
          I_I = I_L + 1;
          #ifdef __DEBUG_SORT__
            cout << "SortIndices(vec_In): I_I(=" << I_I << ") set to (I_L[=" << I_L << "] + 1)  = " << I_L + 1 << endl;
          #endif
          I_J = I_Ir;
          #ifdef __DEBUG_SORT__
            cout << "SortIndices(vec_In): I_J(=" << I_J << ") set to I_Ir[=" << I_Ir << "]" << endl;
          #endif
          I_Indxt = I_A1_Indx[I_L + 1];
          #ifdef __DEBUG_SORT__
            cout << "SortIndices(vec_In): I_Indxt(=" << I_Indxt << ") set to P_I_A1_Indx(I_L = " << I_L << "+1)" << endl;
          #endif
          D_A = vec_In[I_Indxt];
          #ifdef __DEBUG_SORT__
            cout << "SortIndices(vec_In): D_A(=" << D_A << ") set to vec_In[I_Indxt = " << I_Indxt << "]" << endl;
          #endif
          for (;;)
          {
            do
            {
              I_I++;
              #ifdef __DEBUG_SORT__
                cout << "SortIndices(vec_In): I_I set to " << I_I << " => vec_In[P_I_A1_Indx(I_I)] = " << vec_In[I_A1_Indx[I_I]] << endl;
              #endif
            }
            while(vec_In[I_A1_Indx[I_I]] < D_A && I_I < I_SizeIn - 2);
            do
            {
              I_J--;
              #ifdef __DEBUG_SORT__
                cout << "SortIndices(vec_In): I_J set to " << I_J << " => vec_In(P_I_A1_Indx(I_J)) = " << vec_In[I_A1_Indx[I_J]] << endl;
              #endif
            }
            while(vec_In[I_A1_Indx[I_J]] > D_A && I_J > 0);
            if (I_J < I_I)
            {
              #ifdef __DEBUG_SORT__
                cout << "SortIndices(vec_In): I_J(=" << I_J << ") < I_I(=" << I_I << ") => BREAK" << endl;
              #endif
              break;
            }
            std::swap(I_A1_Indx[I_I],
                 I_A1_Indx[I_J]);
            #ifdef __DEBUG_SORT__
              cout << "SortIndices(vec_In): P_I_A1_Indx(I_I=" << I_I << ")=" << I_A1_Indx[I_I] << " and P_I_A1_Indx(I_J(=" << I_J << "))=" << I_A1_Indx[I_J] << " std::swapped" << endl;
            #endif
          }
          I_A1_Indx[I_L + 1] = I_A1_Indx[I_J];
          #ifdef __DEBUG_SORT__
            cout << "SortIndices(vec_In): P_I_A1_Indx(I_L=" << I_L << "+1) set to P_I_A1_Indx(I_J=" << I_J << ") = " << I_A1_Indx[I_L+1] << endl;
          #endif
          I_A1_Indx[I_J] = I_Indxt;
          #ifdef __DEBUG_SORT__
            cout << "SortIndices(vec_In): P_I_A1_Indx(I_J=" << I_J << ") set to I_Indxt(=" << I_Indxt << ")" << endl;
          #endif
          I_JStack += 2;
          #ifdef __DEBUG_SORT__
            cout << "SortIndices(vec_In): I_JStack = " << I_JStack << endl;
          #endif
          if (I_JStack > I_NStack)
          {
            cout << "SortIndices: ERROR: I_NStack ( = " << I_NStack << ") too small!!!";
            exit(EXIT_FAILURE);
          }
          if (I_Ir - I_I + 1 >= I_J - I_L)
          {
            #ifdef __DEBUG_SORT__
              cout << "SortIndices(vec_In): I_Ir(= " << I_Ir << ") - I_I(=" << I_I << ") + 1 = " << I_Ir - I_I + 1 << " >= I_J(="<< I_J << ") + I_L(=" << I_L << ") = " << I_J - I_L << endl;
            #endif
            I_A1_IStack[I_JStack] = I_Ir;
            I_A1_IStack[I_JStack - 1] = I_I;
            I_Ir = I_J - 1;
            #ifdef __DEBUG_SORT__
              cout << "SortIndices(vec_In): I_I set to I_J(=" << I_J << ") - 1" << endl;
            #endif

          }
          else
          {
            #ifdef __DEBUG_SORT__
              cout << "SortIndices(vec_In): I_Ir(= " << I_Ir << ") - I_I(=" << I_I << ") + 1 = " << I_Ir - I_I + 1 << " < I_J(="<< I_J << ") + I_L(=" << I_L << ") = " << I_J - I_L << endl;
            #endif
            I_A1_IStack[I_JStack] = I_J - 1;
            I_A1_IStack[I_JStack - 1] = I_L;
            I_L = I_I;
            #ifdef __DEBUG_SORT__
              cout << "SortIndices(vec_In): I_L set to I_I(=" << I_I << endl;
            #endif

          }
        }
      }
      return (I_A1_Indx);
    }
        
    template<typename T>
    ndarray::Array<T, 1, 1> resize(ndarray::Array<T, 1, 1> const& arr, size_t const newSize){
      ndarray::Array<T, 1, 1> arrOut = ndarray::allocate(newSize);
      arrOut.deep() = 0;
      for (auto itArrIn = arr.begin(), itArrOut = arrOut.begin(); (itArrIn != arr.end()) && (itArrOut != arrOut.end()); ++itArrIn, ++itArrOut){
        *itArrOut = *itArrIn;
      }
      return arrOut;
    }
    
/*    template< typename T >
    ndarray::Array< T, 2, 1 > get2DArray(ndarray::Array< T, 1, 1 > const& xIn, ndarray::Array< T, 1, 1 > const& yIn){
      ndarray::Array< T, 2, 1 > arrOut = ndarray::allocate(yIn.getShape()[0], yIn.getShape()[0]);
      for (size_t iRow = 0; iRow < yIn.getShape()[0]; ++iRow){
        for (size_t iCol = 0; iCol < xIn.getShape()[0]; ++iCol){
          arrOut[iRow][iCol]
        }
      }
    }
*/
    double uvalue(double x, double low, double high)
    {
      return (x - low)/(high-low);
    }

    Eigen::VectorXd uvalues(Eigen::VectorXd const& xVals)
    {
      Eigen::VectorXd xvals(xVals.size());
      const double low = xVals.minCoeff();
      const double high = xVals.maxCoeff();
      #ifdef __DEBUG_XCOR__
        cout << "math::uvalues: xVals = " << xVals << endl;
        cout << "math::uvalues: low = " << low << ", high = " << high << endl;
      #endif
      for (int i = 0; i < xvals.size(); ++i){
        xvals(i) = uvalue(xVals[i], low, high);
      }
      return xvals;
    }
    
    template< typename T >
    T xCor(ndarray::Array< T, 2, 1 > const& arrA_In,
           ndarray::Array< T, 2, 1 > const& arrB_In,
           ndarray::Array< T, 1, 1 > const& range_In,
           float const& stepSize_In){
      
      ndarray::Array< double, 1, 1 > xValuesA = ndarray::allocate(arrA_In.getShape()[0]);
      xValuesA.deep() = arrA_In[ndarray::view()(0)];
      #ifdef __DEBUG_XCOR__
        cout << "math::xCor: arrA_In = " << arrA_In << endl;
        cout << "math::xCor: arrB_In = " << arrB_In << endl;
        cout << "math::xCor: range_In = " << range_In << ", stepSize_In = " << stepSize_In << endl;
        cout << "math::xCor: xValuesA = " << xValuesA << endl;
      #endif
      ndarray::Array< double, 1, 1 > xValuesBShifted = ndarray::allocate(arrB_In.getShape()[0]);
      ndarray::Array< double, 1, 1 > yValuesB = ndarray::allocate(arrB_In.getShape()[0]);
      ndarray::Array< double, 1, 1 > yValuesInterpolated = ndarray::allocate(arrA_In.getShape()[0]);
      xValuesBShifted.deep() = arrB_In[ndarray::view()(0)];
      yValuesB.deep() = arrB_In[ndarray::view()(1)];
      
      std::vector<double> xShift(0);
      xShift.reserve((range_In[1] - range_In[0]) / stepSize_In + 10);
      std::vector<double> chiSquare(0);
      chiSquare.reserve((range_In[1] - range_In[0]) / stepSize_In + 10);
      
      ndarray::Array<double, 1, 1> diff = ndarray::allocate(arrA_In.getShape()[0]);
      typedef Eigen::Spline<double,1> Spline2d;

      const size_t interpolOrder = 3;
      for (double shift = range_In[0]; shift <= range_In[1]; shift += stepSize_In){
        xValuesBShifted.deep() = arrB_In[ndarray::view()(0)] + shift;
        #ifdef __DEBUG_XCOR__
          cout << "math::xCor: shift = " << shift << ": xValuesBShifted = " << xValuesBShifted << endl;
        #endif
        Eigen::VectorXd uxValues = uvalues(xValuesBShifted.asEigen());
        #ifdef __DEBUG_XCOR__
          cout << "math::xCor: shift = " << shift << ": uxValues = " << uxValues << endl;
        #endif
        const Spline2d spline = Eigen::SplineFitting<Spline2d>::Interpolate(yValuesB.asEigen().transpose(), interpolOrder, uxValues.transpose());
        for (int i = 0; i < arrA_In.getShape()[0]; ++i){
          const double uv = uvalue(arrA_In[i][0], xValuesBShifted.asEigen().minCoeff(), xValuesBShifted.asEigen().maxCoeff());
          #ifdef __DEBUG_XCOR__
            cout << "math::xCor: shift = " << shift << ": i = " << i << ": arrA_In[" << i << "][0] = " << arrA_In[i][0] << ": uv = " << uv << ": spline(uv) = " << spline(uv) << endl;
          #endif
          yValuesInterpolated[i] = double(spline(uv)(0));
        }
        #ifdef __DEBUG_XCOR__
          cout << "math::xCor: shift = " << shift << ": yValuesInterpolated = " << yValuesInterpolated << endl;
        #endif
        xShift.push_back(shift);
        diff.deep() = arrA_In[ndarray::view()(1)] - yValuesInterpolated;
        #ifdef __DEBUG_XCOR__
          cout << "math::xCor: shift = " << shift << ": diff = " << diff << endl;
        #endif
        chiSquare.push_back(diff.asEigen().array().pow(2).sum());
        #ifdef __DEBUG_XCOR__
          cout << "math::xCor: shift = " << shift << ": chiSquare[" << chiSquare.size()-1 << "] = " << chiSquare[chiSquare.size()-1] << endl;
        #endif
      }
      T minShift = T(xShift[std::min_element(chiSquare.begin(), chiSquare.end()) - chiSquare.begin()]);
      #ifdef __DEBUG_XCOR__
        cout << "math::xCor: chiSquare = ";
        for (auto it=chiSquare.begin(); it!=chiSquare.end(); ++it)
          cout << *it << " ";
        cout << endl;
      #endif
      cout << "math::xCor: minShift = " << minShift << endl;
      return minShift;
    }
    
    template< typename T >
    ndarray::Array< T const, 1, 1 > vecToNdArray(std::vector<T> const& vec_In){
      ndarray::Array< T const, 1, 1 > arr_Out = ndarray::external(vec_In.data(), ndarray::makeVector(int(vec_In.size())), ndarray::makeVector(1));
      return arr_Out;
    }
    
    template ndarray::Array< size_t const, 1, 1 > vecToNdArray(std::vector<size_t> const&);
    template ndarray::Array< unsigned short const, 1, 1 > vecToNdArray(std::vector<unsigned short> const&);
    template ndarray::Array< unsigned int const, 1, 1 > vecToNdArray(std::vector<unsigned int> const&);
    template ndarray::Array< int const, 1, 1 > vecToNdArray(std::vector<int> const&);
    template ndarray::Array< long const, 1, 1 > vecToNdArray(std::vector<long> const&);
    template ndarray::Array< float const, 1, 1 > vecToNdArray(std::vector<float> const&);
    template ndarray::Array< double const, 1, 1 > vecToNdArray(std::vector<double> const&);
    
    template float xCor(ndarray::Array< float, 2, 1 > const&, ndarray::Array< float, 2, 1 > const&, ndarray::Array< float, 1, 1 > const&, float const&);
    template double xCor(ndarray::Array< double, 2, 1 > const&, ndarray::Array< double, 2, 1 > const&, ndarray::Array< double, 1, 1 > const&, float const&);
    
    template ndarray::Array< size_t, 1, 1 > resize( ndarray::Array< size_t, 1, 1 > const& arr_In, size_t newSize);
    template ndarray::Array< unsigned short, 1, 1 > resize( ndarray::Array< unsigned short, 1, 1 > const& arr_In, size_t newSize);
    template ndarray::Array< short, 1, 1 > resize( ndarray::Array< short, 1, 1 > const& arr_In, size_t newSize);
    template ndarray::Array< unsigned int, 1, 1 > resize( ndarray::Array< unsigned int, 1, 1 > const& arr_In, size_t newSize);
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
    template ndarray::Array<unsigned int, 1, 1> indGenNdArr(unsigned int const);
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

    template ndarray::Array<double, 2, 1> Double(ndarray::Array<size_t, 2, 1> const&);
    template ndarray::Array<double, 2, 1> Double(ndarray::Array<unsigned short, 2, 1> const&);
    template ndarray::Array<double, 2, 1> Double(ndarray::Array<int, 2, 1> const&);
    template ndarray::Array<double, 2, 1> Double(ndarray::Array<long, 2, 1> const&);
    template ndarray::Array<double, 2, 1> Double(ndarray::Array<float, 2, 1> const&);
    template ndarray::Array<double, 2, 1> Double(ndarray::Array<float const, 2, 1> const&);
    template ndarray::Array<double, 2, 1> Double(ndarray::Array<double, 2, 1> const&);

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
    
    template size_t minIndex(ndarray::Array<size_t, 1, 1> const&);
    template size_t minIndex(ndarray::Array<unsigned short, 1, 1> const&);
    template size_t minIndex(ndarray::Array<int, 1, 1> const&);
    template size_t minIndex(ndarray::Array<long, 1, 1> const&);
    template size_t minIndex(ndarray::Array<float, 1, 1> const&);
    template size_t minIndex(ndarray::Array<double, 1, 1> const&);
    
    template size_t max(ndarray::Array<size_t, 1, 1> const&);
    template unsigned short max(ndarray::Array<unsigned short, 1, 1> const&);
    template int max(ndarray::Array<int, 1, 1> const&);
    template long max(ndarray::Array<long, 1, 1> const&);
    template float max(ndarray::Array<float, 1, 1> const&);
    template double max(ndarray::Array<double, 1, 1> const&);

    template size_t maxIndex(ndarray::Array<size_t, 1, 1> const&);
    template size_t maxIndex(ndarray::Array<unsigned short, 1, 1> const&);
    template size_t maxIndex(ndarray::Array<int, 1, 1> const&);
    template size_t maxIndex(ndarray::Array<long, 1, 1> const&);
    template size_t maxIndex(ndarray::Array<float, 1, 1> const&);
    template size_t maxIndex(ndarray::Array<double, 1, 1> const&);

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
  }/// end namespace math


}}}
      
template< typename T >
std::ostream& operator<<(std::ostream& os, std::vector<T> const& obj)
{
  for (int i = 0 ; i < obj.size(); ++i){
    os << obj[i] << " ";
  }
  os << endl;
  return os;
}

template std::ostream& operator<<(std::ostream&, std::vector<int> const&);
template std::ostream& operator<<(std::ostream&, std::vector<float> const&);
template std::ostream& operator<<(std::ostream&, std::vector<double> const&);
