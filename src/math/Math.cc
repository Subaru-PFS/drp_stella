#include "pfs/drp/stella/math/Math.h"
  namespace pfs{ namespace drp{ namespace stella{ namespace math{
    /**
     * Calculates aperture minimum pixel, central position, and maximum pixel for the trace,
     * and writes result to minCenMax_Out and returns it
     **/
    template< typename T, typename U >
    ndarray::Array<size_t, 2, 1> calcMinCenMax(ndarray::Array<T, 1, 1> const& xCenters_In,
                                               U const xHigh_In,
                                               U const xLow_In,
                                               int const nPixCutLeft_In,
                                               int const nPixCutRight_In){
      ndarray::Array<size_t, 1, 1> floor = pfs::drp::stella::math::floor(xCenters_In, size_t(0));
      ndarray::Array<size_t, 2, 2> minCenMax_Out = ndarray::allocate(xCenters_In.getShape()[0], 3);
      minCenMax_Out[ndarray::view()()] = 0;

      minCenMax_Out[ndarray::view()(1)] = floor;

      #ifdef __DEBUG_MINCENMAX__
        cout << "calcMinCenMax: minCenMax_Out(*,1) = " << minCenMax_Out[ndarray::view()(1)] << endl;
      #endif
      ndarray::Array<double, 1, 1> D_A1_Temp = ndarray::allocate(xCenters_In.getShape()[0]);
      D_A1_Temp.deep() = xCenters_In + xLow_In;

      minCenMax_Out[ndarray::view()(0)] = pfs::drp::stella::math::floor(ndarray::Array<double const, 1, 1>(D_A1_Temp), size_t(0));

      #ifdef __DEBUG_MINCENMAX__
        cout << "calcMinCenMax: minCenMax_Out(*,0) = " << minCenMax_Out[ndarray::view()(0)] << endl;
      #endif
      D_A1_Temp.deep() = xCenters_In + xHigh_In;

      minCenMax_Out[ndarray::view()(2)] = pfs::drp::stella::math::floor(ndarray::Array<double const, 1, 1>(D_A1_Temp), size_t(0));

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
    ndarray::Array<U, 1, 1> floor(const ndarray::Array<T, 1, 1> &rhs, const U outType){
      ndarray::Array<U, 1, 1> outVal = allocate(rhs.getShape());
      typename ndarray::Array<U, 1, 1>::Iterator iOut = outVal.begin();
      for (auto iIn = rhs.begin(); iIn != rhs.end(); ++iIn){
        *iOut = floor1(*iIn, outType);
        ++iOut;
      }
      return outVal;
    }
    
    template< typename T, typename U >
    ndarray::Array<U, 2, 2> floor(const ndarray::Array<T, 2, 2> &rhs, const U outType){
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
    ndarray::Array<float, 2, 1> Float(ndarray::Array<T, 2, 1> const& arr_In){
      ndarray::Array<float, 2, 1> arr_Out = ndarray::allocate(arr_In.getShape()[0], arr_In.getShape()[1]);
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
    ndarray::Array<T, 2, 1> calcPosRelativeToCenter(ndarray::Array<T, 2, 1> const& swath_In, ndarray::Array<T, 1, 1> const& xCenters_In){
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
      
      ndarray::Array<T, 2, 1> outArr = ndarray::copy(indMat);
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
    ndarray::Array<size_t, 2, 1> getIndicesInValueRange(ndarray::Array<T, 2, 1> const& arr_In, T const lowRange_In, T const highRange_In){
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
      ndarray::Array<size_t, 2, 1> arr_Out = ndarray::allocate(indicesRow.size(), 2);
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
    std::vector< size_t > getIndices( std::vector< T > const& vec_In ){
      std::vector< size_t > vecOut( 0 );
      for (size_t pos = 0; pos < vec_In.size(); ++pos ){
        if ( int( vec_In[ pos ] ) == 1 )
          vecOut.push_back( pos );
      }
      return vecOut;
    }
    
    template<typename T>
    ndarray::Array< size_t, 1, 1 > getIndices( ndarray::Array< T, 1, 1 > const& arr_In ){
      ndarray::Array< size_t, 1, 1 > arrOut = ndarray::allocate( std::accumulate( arr_In.begin(), arr_In.end(), 0 ) );
      auto itOut = arrOut.begin();
      int pos = 0;
      for (auto itIn = arr_In.begin(); itIn != arr_In.end(); ++itIn ){
        if ( int( *itIn ) == 1 ){
          *itOut = pos;
          ++pos;
          ++itOut;
        }
      }
      return arrOut;
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
    
    template<typename T, typename U>
    ndarray::Array<T, 1, 1> getSubArray(ndarray::Array<T, 1, 1> const& arr_In, 
                                        ndarray::Array<U, 1, 1> const& indices_In){
      ndarray::Array<T, 1, 1> arr_Out = ndarray::allocate(indices_In.getShape()[0]);
      for (int ind = 0; ind < indices_In.getShape()[0]; ++ind){
        arr_Out[ind] = arr_In[indices_In[ind]];
      }
      return arr_Out;
    }

/*    template<typename T>
    ndarray::Array<T, 1, 1> getSubArray(ndarray::Array<T, 2, 1> const& arr_In, 
                                        ndarray::Array<size_t, 2, 1> const& indices_In){
      ndarray::Array<T, 1, 1> arr_Out = ndarray::allocate(indices_In.getShape()[0]);
      for (size_t iRow = 0; iRow < indices_In.getShape()[0]; ++iRow){
        arr_Out[iRow] = arr_In[indices_In[iRow][0]][indices_In[iRow][1]];
        #ifdef __DEBUG_GETSUBARRAY__
          cout << "getSubArray: arr_Out[" << iRow << "] = " << arr_Out[iRow] << endl;
        #endif
      }
      return arr_Out;
    }*/

    template< typename T, typename U >
    ndarray::Array<T, 1, 1> getSubArray(ndarray::Array<T, 2, 1> const& arr_In, 
                                        ndarray::Array<U, 2, 1> const& indices_In){
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
    ndarray::Array<T, 1, 1> getSubArray(ndarray::Array<T, 2, 1> const& arr_In, 
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
    void insertSorted(std::vector< dataXY< T > > & dataXYVec_In,
                      dataXY< T > & toInsert_In){
      if (dataXYVec_In.size() == 0){
        dataXYVec_In.push_back(toInsert_In);
        return;
      }
      for (auto it = dataXYVec_In.begin(); it != dataXYVec_In.end(); ++it){
        if (it->x > toInsert_In.x){
          dataXYVec_In.insert(it, toInsert_In);
          return;
        }
      }
      dataXYVec_In.push_back(toInsert_In);
      return;
    }

    template<typename T>
    std::vector<int> sortIndices(const std::vector<T> &vec_In){
      int I_SizeIn = vec_In.size();
      std::vector<int> I_A1_Indx(1);
      I_A1_Indx[0] = 0;
      bool isInserted = false;
      for (int i = 1; i < I_SizeIn; ++i){
        isInserted = false;
        auto it = I_A1_Indx.begin();
        while((!isInserted) && (it != I_A1_Indx.end())){
          if (vec_In[*it] >= vec_In[i]){
            I_A1_Indx.insert(it, i);
            isInserted = true;
          }
          ++it;
        }
        if (!isInserted)
          I_A1_Indx.push_back(i);
      }
/*      int I_M = 7;
      int I_NStack = 50;

      int I_Ir = I_SizeIn - 1;
      int I_L = 0;

      int I_I, I_Indxt, I_J, I_K;
      int I_JStack = 0;
      std::vector<int> I_A1_IStack(I_NStack);
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
      }*/
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
    
    template< typename PsfT, typename CoordT >
    CoordT xCor(ndarray::Array< CoordT, 2, 1 > const& arrA_In,
                ndarray::Array< PsfT, 2, 1 > const& arrB_In,
                ndarray::Array< CoordT, 1, 1 > const& range_In,
                CoordT const& stepSize_In){
      
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
      /// Fit Gaussian to chiSquare
      ndarray::Array<double, 1, 1> xShiftArr = ndarray::external(xShift.data(), ndarray::makeVector(int(xShift.size())), ndarray::makeVector(1));
      ndarray::Array<double, 1, 1> chiSquareArr = ndarray::external(chiSquare.data(), ndarray::makeVector(int(chiSquare.size())), ndarray::makeVector(1));
      ndarray::Array<double, 1, 1> eChiSquareArr = ndarray::allocate(chiSquareArr.getShape()[0]);
      ndarray::Array<double, 1, 1> D_A1_Guess = ndarray::allocate(4);
      #ifdef __DEBUG_XCOR__
        cout << "pfs::drp::stella::math::xCor: chiSquareArr = " << chiSquareArr << endl;
      #endif
      D_A1_Guess[3] = max(chiSquareArr);///Peak
      D_A1_Guess[0] = 0. - D_A1_Guess[3] + min(chiSquareArr);///Centroid
      D_A1_Guess[1] = 0.;///Sigma
      D_A1_Guess[2] = (range_In[1] - range_In[0])/2.;///constant offset
      ndarray::Array<int, 2, 1> I_A2_Limited = ndarray::allocate(4, 2);
      I_A2_Limited.deep() = 1;
      #ifdef __DEBUG_XCOR__
        cout << "pfs::drp::stella::math::xCor: D_A1_Guess = " << D_A1_Guess << endl;
        cout << "pfs::drp::stella::math::xCor: I_A2_Limited = " << I_A2_Limited << endl;
      #endif
      ndarray::Array<double, 2, 1> D_A2_Limits = ndarray::allocate(4, 2);
      D_A2_Limits[3][0] = 0.;
      D_A2_Limits[3][1] = 1.5 * max(chiSquareArr);
      D_A2_Limits[0][0] = 0. - D_A1_Guess[3];
      D_A2_Limits[0][1] = 0.;
      D_A2_Limits[1][0] = range_In[0];
      D_A2_Limits[1][1] = range_In[1];
      D_A2_Limits[2][0] = 0.;
      D_A2_Limits[2][1] = 2.;
      #ifdef __DEBUG_XCOR__
        cout << "pfs::drp::stella::math::xCor: D_A2_Limits = " << D_A2_Limits << endl;
      #endif
      eChiSquareArr.asEigen() = chiSquareArr.asEigen().array().sqrt();
      ndarray::Array<double, 1, 1> D_A1_GaussFitCoeffs = ndarray::allocate(4);
      ndarray::Array<double, 1, 1> D_A1_GaussFitECoeffs = ndarray::allocate(4);
      if (!MPFitGaussLim(xShiftArr,
                         chiSquareArr,
                         eChiSquareArr,
                         D_A1_Guess,
                         I_A2_Limited,
                         D_A2_Limits,
                         1,
                         false,
                         D_A1_GaussFitCoeffs,
                         D_A1_GaussFitECoeffs,
                         true)){
        string message("pfs::drp::stella::math::xCor: ERROR: MPFitGaussLim returned FALSE");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      CoordT minShift = CoordT(D_A1_GaussFitCoeffs[1]);//T(xShift[std::min_element(chiSquare.begin(), chiSquare.end()) - chiSquare.begin()]);
      #ifdef __DEBUG_XCOR__
        cout << "math::xCor: xShiftArr = " << xShiftArr << endl;
        cout << "math::xCor: chiSquareArr = " << chiSquareArr << endl;
        cout << "math::xCor: D_A1_Guess = " << D_A1_Guess << endl;
        cout << "math::xCor: D_A1_GaussFitCoeffs = " << D_A1_GaussFitCoeffs << endl;
      #endif
      cout << "math::xCor: minShift = " << minShift << endl;
      return minShift;
    }
    
    template< typename T, typename U >
    T convertRangeToUnity(T number,
                          ndarray::Array<U, 1, 1> const& range){
      if (range.getShape()[0] != 2){
        string message("pfs::drp::stella::math::convertRangeToUnity: ERROR: range.getShape()[0](=");
        message += to_string(range.getShape()[0]) + ") != 2";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_CONVERTRANGETOUNITY__
        cout << "pfs::drp::stella::math::convertRangeToUnity: range = " << range << ", number = " << number << endl;
        cout << "pfs::drp::stella::math::convertRangeToUnity: number - range[0] = " << number - range[0] << endl;
        cout << "pfs::drp::stella::math::convertRangeToUnity: range[1] - range[0] = " << range[1] - range[0] << endl;
      #endif
      T out = ((number - range[0]) * 2. / (range[1] - range[0])) - 1.;
      #ifdef __DEBUG_CONVERTRANGETOUNITY__
        cout << "pfs::drp::stella::math::convertRangeToUnity: out = " << out << endl;
      #endif
      return out;
    }
    
    template< typename T, typename U >
    ndarray::Array<T, 1, 1> convertRangeToUnity(ndarray::Array<T, 1, 1> const& numbers,
                                                ndarray::Array<U, 1, 1> const& range){
      ndarray::Array<T, 1, 1> out = ndarray::allocate(numbers.getShape()[0]);
      auto itIn = numbers.begin();
      for (auto itOut = out.begin(); itOut != out.end(); ++itOut, ++itIn){
        *itOut = convertRangeToUnity(*itIn, range);
      }
      return out;
    }

    template< typename T, typename U >
    bool checkIfValuesAreInRange(ndarray::Array<T, 1, 1> const& numbers,
                                 ndarray::Array<U, 1, 1> const& range){
      for (auto it = numbers.begin(); it != numbers.end(); ++it){
        if ((*it < range[0]) || (*it > range[1]))
          return false;
      }
      return true;
    }
    
    template< typename T >
    ndarray::Array<T, 1, 1> vectorToNdArray(std::vector<T> & vector){
      ndarray::Array<T, 1, 1> ndArray = ndarray::external(vector.data(), ndarray::makeVector(int(vector.size())), ndarray::makeVector(1));
      return ndArray;
    }
    
    template< typename T >
    ndarray::Array< T const, 1, 1 > vectorToNdArray(std::vector<T> const& vec_In){
      ndarray::Array< T const, 1, 1 > arr_Out = ndarray::external(vec_In.data(), ndarray::makeVector(int(vec_In.size())), ndarray::makeVector(1));
      return arr_Out;
    }

    template< typename T >
    std::vector< T > ndArrayToVector( ndarray::Array< T, 1, 1 > const& ndArray_In){
      std::vector< T > vec;
      vec.reserve(ndArray_In.getShape()[ 0 ]);
      for (auto it = ndArray_In.begin(); it != ndArray_In.end(); ++it)
        vec.push_back( *it );
      return vec;
    }
    
    template< typename T >
    ndarray::Array<T, 2, 1> ndArray21(T nRows, T nCols){
      ndarray::Array<T, 2, 1> out = ndarray::allocate(int(nRows), int(nCols));
      return out;
    }
    
    template< typename T >
    ndarray::Array<T, 2, 2> ndArray22(T nRows, T nCols){
      ndarray::Array<T, 2, 2> out = ndarray::allocate(int(nRows), int(nCols));
      return out;
    }

    template< typename T >
    ndarray::Array< T, 1, 1 > getZMinMaxInRange( ndarray::Array< T, 1, 1 > const& x_In,
                                                 ndarray::Array< T, 1, 1 > const& y_In,
                                                 ndarray::Array< T, 1, 1 > const& z_In,
                                                 ndarray::Array< T, 1, 1 > const& xRange_In,
                                                 ndarray::Array< T, 1, 1 > const& yRange_In ){
      size_t sizeX = x_In.getShape()[ 0 ];
      size_t sizeY = y_In.getShape()[ 0 ];
      size_t sizeZ = z_In.getShape()[ 0 ];
      size_t sizeXRange = xRange_In.getShape()[ 0 ];
      size_t sizeYRange = yRange_In.getShape()[ 0 ];
      if ( sizeX != sizeY ){
        string message("pfs::drp::stella::math::getZMinMaxInRange: ERROR: x_In.getShape()[ 0 ](=");
        message += to_string( sizeX ) + ") != y_In.getShape()[ 0 ](=" + to_string( sizeY ) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if ( sizeX != sizeZ ){
        string message("pfs::drp::stella::math::getZMinMaxInRange: ERROR: x_In.getShape()[ 0 ](=");
        message += to_string( sizeX ) + ") != z_In.getShape()[ 0 ](=" + to_string( sizeZ ) + ")";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if ( sizeXRange != 2 ){
        string message("pfs::drp::stella::math::getZMinMaxInRange: ERROR: xRange_In.getShape()[ 0 ](=");
        message += to_string( sizeXRange ) + ") != 2";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if ( sizeYRange != 2 ){
        string message("pfs::drp::stella::math::getZMinMaxInRange: ERROR: yRange_In.getShape()[ 0 ](=");
        message += to_string( sizeYRange ) + ") != 2";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      T minValue = 0.;
      T maxValue = 0.;
      bool minAndMaxInitialized = false;
      for (auto itX = x_In.begin(), itY = y_In.begin(), itZ = z_In.begin(); itX != x_In.end(); ++itX, ++itY, ++itZ){
        if ( ( *itX >= xRange_In[ 0 ] ) && ( *itX <= xRange_In[ 1 ] ) ){
          if ( ( *itY >= yRange_In[ 0 ] ) && ( *itY <= yRange_In[ 1 ] ) ){
            if ( !minAndMaxInitialized ){
              minValue = *itZ;
              maxValue = *itZ;
              minAndMaxInitialized = true;
            }
            else{
              if ( *itZ < minValue )
                minValue = *itZ;
              if ( *itZ > maxValue )
                maxValue = *itZ;
            }
          }
        }
      }
      ndarray::Array< T, 1, 1 > outVals = ndarray::allocate( 2 );
      outVals[ 0 ] = minValue;
      outVals[ 1 ] = maxValue;
      return outVals;
    }
    
    template < typename T >
    ndarray::Array< T, 2, 1 > createRectangularGrid( ndarray::Array< T, 1, 1 > const& xRange,
                                                     ndarray::Array< T, 1, 1 > const& yRange,
                                                     T xStep,
                                                     T yStep ){
      size_t nX = (xRange[1] - xRange[0]) / xStep + 1;
      size_t nY = (yRange[1] - yRange[0]) / yStep + 1;
      ndarray::Array< T, 2, 1 > outArr = ndarray::allocate( nX * nY, 2 );
      T x = xRange[0];
      T y = yRange[0];
      size_t pos = 0;
      for (size_t iX = 0; iX < nX; ++iX){
        y = yRange[0];
        for (size_t iY = 0; iY < nY; ++ iY){
          outArr[ pos ][ 0 ] = x;
          outArr[ pos ][ 1 ] = y;
          y += yStep;
          ++pos;
        }
        x += xStep;
      }
      return outArr;
    }
    
    template< typename T >
    ndarray::Array< T, 2, 1 > createPolarGrid( T rMax,
                                               T rStep,
                                               T phiStep ){
      size_t nStepsR = (rMax / rStep);
      size_t nStepsPhi = (360. / phiStep);
//      cout << "nStepsR = " << nStepsR << ", nStepsPhi = " << nStepsPhi << endl;
      ndarray::Array< T, 2, 1 > arrOut = ndarray::allocate( nStepsR * nStepsPhi + 1, 2 );
      T r = 0.;
      T phi = 0.;
      arrOut[ 0 ][ 0 ] = 0.;
      arrOut[ 0 ][ 1 ] = 0.;
      size_t pos = 1;
      for (size_t iR = 1; iR <= nStepsR; ++iR ){
        r += rStep;
//        cout << "r = " << r << endl;
        phi = 0.;
        for (size_t iPhi = 1; iPhi <= nStepsPhi; ++iPhi){
          phi += phiStep;
//          cout << "phi = " << phi << endl;
          arrOut[ pos ][ 0 ] = r * cos( phi * 2. * D_PI / 360.);
          arrOut[ pos ][ 1 ] = r * sin( phi * 2. * D_PI / 360.);
//          cout << "arrOut[" << pos << "][0] = " << arrOut[pos][0] << ", arrOut[" << pos << "][1] = " << arrOut[pos][1] << endl;
          ++pos;
        }
      }
      return arrOut;
    }
    
    template< typename T >
    T calculateChiSquare( ndarray::Array< T, 1, 1 > const& expected,
                          ndarray::Array< T, 1, 1 > const& observed ){
      if ( expected.getShape()[0] != observed.getShape()[0] ){
        string message("pfs::drp::stella::math::calculateChiSquare: ERROR: expected.getShape()[ 0 ](=");
        message += to_string( expected.getShape()[ 0 ] ) + ") != observed.getShape()[ 0 ] (= ";
        message += to_string( observed.getShape()[ 0 ] ) + "0";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      T chiSquare = 0.;
      size_t pos = 0;
      for (auto itExp = expected.begin(), itObs = observed.begin(); itExp != expected.end(); ++itExp, ++itObs, ++pos){
        chiSquare += pow(*itExp - *itObs, T(2.)) / *itExp;
        cout << "calculateChiSquare: pos = " << pos << ": *itExp = " << *itExp << ", *itObs = " << *itObs << ": chiSquare = " << chiSquare << endl;
      }
      return chiSquare;
    }
    
    template< typename T > 
    ndarray::Array< T, 1, 1 > getDataInRange( ndarray::Array< T, 1, 1 > const& xArr,
                                              ndarray::Array< T, 1, 1 > const& yArr,
                                              ndarray::Array< T, 1, 1 > const& zArr,
                                              ndarray::Array< T, 1, 1 > const& xRange,
                                              ndarray::Array< T, 1, 1 > const& yRange ){
      if ( xArr.getShape()[ 0 ] != yArr.getShape()[ 0 ] ){
        string message("pfs::drp::stella::math::getDataInRange: ERROR: xArr.getShape()[ 0 ](=");
        message += to_string( xArr.getShape()[ 0 ] ) + ") != yArr.getShape()[ 0 ] (= ";
        message += to_string( yArr.getShape()[ 0 ] ) + "0";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if ( xArr.getShape()[ 0 ] != zArr.getShape()[ 0 ] ){
        string message("pfs::drp::stella::math::getDataInRange: ERROR: xArr.getShape()[ 0 ](=");
        message += to_string( xArr.getShape()[ 0 ] ) + ") != zArr.getShape()[ 0 ] (= ";
        message += to_string( zArr.getShape()[ 0 ] ) + "0";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if ( xRange.getShape()[ 0 ] != 2 ){
        string message("pfs::drp::stella::math::getDataInRange: ERROR: xRange.getShape()[ 0 ](=");
        message += to_string( xRange.getShape()[ 0 ] ) + ") != 2";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if ( yRange.getShape()[ 0 ] != 2 ){
        string message("pfs::drp::stella::math::getDataInRange: ERROR: yRange.getShape()[ 0 ](=");
        message += to_string( yRange.getShape()[ 0 ] ) + ") != 2";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      std::vector< size_t > indices;
      size_t ind = 0;
      for ( auto itX = xArr.begin(), itY = yArr.begin(); itX != xArr.end(); ++itX, ++itY ){
        if ( *itX >= xRange[ 0 ] && *itX <= xRange[ 1 ] && *itY >= yRange[ 0 ] && *itY <= yRange[ 1 ] )
          indices.push_back( ind );
        ++ind;
      }
      ndarray::Array< T, 1, 1 > zOut = ndarray::allocate( indices.size() );
      for ( ind = 0; ind < indices.size(); ++ind )
        zOut[ ind ] = zArr[ indices[ ind ] ];
      return zOut;
    }
    
    template< typename T > 
    ndarray::Array< T, 1, 1 > getDataInRange( ndarray::Array< T, 1, 1 > const& xArr,
                                              ndarray::Array< T, 1, 1 > const& yArr,
                                              ndarray::Array< T, 1, 1 > const& zArr,
                                              ndarray::Array< T, 1, 1 > const& rRange ){
      if ( xArr.getShape()[ 0 ] != yArr.getShape()[ 0 ] ){
        string message("pfs::drp::stella::math::getDataInRange: ERROR: xArr.getShape()[ 0 ](=");
        message += to_string( xArr.getShape()[ 0 ] ) + ") != yArr.getShape()[ 0 ] (= ";
        message += to_string( yArr.getShape()[ 0 ] ) + "0";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if ( xArr.getShape()[ 0 ] != zArr.getShape()[ 0 ] ){
        string message("pfs::drp::stella::math::getDataInRange: ERROR: xArr.getShape()[ 0 ](=");
        message += to_string( xArr.getShape()[ 0 ] ) + ") != zArr.getShape()[ 0 ] (= ";
        message += to_string( zArr.getShape()[ 0 ] ) + "0";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if ( rRange.getShape()[ 0 ] != 2 ){
        string message("pfs::drp::stella::math::getDataInRange: ERROR: rRange.getShape()[ 0 ](=");
        message += to_string( rRange.getShape()[ 0 ] ) + ") != 2";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      std::vector< size_t > indices;
      size_t ind = 0;
      double len;
      for ( auto itX = xArr.begin(), itY = yArr.begin(); itX != xArr.end(); ++itX, ++itY ){
        len = sqrt( ( *itX * *itX ) + ( *itY * *itY ) );
        if ( len >= rRange[ 0 ] && len <= rRange[ 1 ] )
          indices.push_back( ind );
        ++ind;
      }
      ndarray::Array< T, 1, 1 > zOut = ndarray::allocate( indices.size() );
      for ( ind = 0; ind < indices.size(); ++ind )
        zOut[ ind ] = zArr[ indices[ ind ] ];
      return zOut;
    }
    
    template< typename T >
    int isMonotonic(ndarray::Array< T, 1, 1 > const& arrIn){
      int I_M = 0;
      if ( arrIn.getShape()[ 0 ] < 2 )
        return I_M;
      double D_DA = arrIn[ 1 ] - arrIn[ 0 ];
      if ( D_DA < 0 )
        I_M = -1;
      else if ( D_DA > 0 )
        I_M = 1;
      if ( arrIn.getShape()[ 0 ] < 3 )
        return I_M;
      for ( int i_pos = 2; i_pos < arrIn.getShape()[ 0 ]; i_pos++ ){
        D_DA = arrIn[ i_pos ] - arrIn[ i_pos - 1 ];
        if ( ( D_DA < 0 ) && ( I_M == 1 ) )
          return 0;
        if ( ( D_DA > 0 ) && ( I_M == -1 ) )
          return 0;
        if ( I_M == 0 ){
          if ( D_DA < 0 )
            I_M = -1;
          else if ( D_DA > 0 )
            I_M = 1;
        }
      }
      return I_M;
    }
    
    template< typename T >
    T calcRMS( ndarray::Array< T, 1, 1 > const& arrIn ){
      T rms = 0;
      for ( auto itArr = arrIn.begin(); itArr != arrIn.end(); ++itArr )
        rms += T( pow( double( *itArr ), 2. ) );
      rms  = rms / arrIn.getShape()[ 0 ];
      return sqrt( rms );
    }
    
    /**
      CrossCorrelate with Gauss fit to ChiSquare to find subpixel of minimum
     **/
    template< typename T >
    bool crossCorrelate(ndarray::Array<T, 1, 1> const& DA1_Static,
                        ndarray::Array<T, 1, 1> const& DA1_Moving,
                        int const I_NPixMaxLeft,
                        int const I_NPixMaxRight,
                        double &D_Out,
                        double &D_ChiSquare_Out){
      /// Check that both arrays have the same size
      if ( DA1_Moving.getShape()[ 0 ] != DA1_Static.getShape()[ 0 ] ){
        cout << "CFits::CrossCorrelate: ERROR: DA1_Moving.size() = " << DA1_Moving.getShape()[ 0 ] << " != DA1_Static.size() = " << DA1_Static.getShape()[ 0 ] << endl;
        return false;
      }

      int I_Size = DA1_Static.getShape()[ 0 ];

      /// Check I_NPixMaxLeft and I_NPixMaxRight
      int nPixMaxLeft = I_NPixMaxLeft;
      if ( nPixMaxLeft >= I_Size ){
        nPixMaxLeft = I_Size - 1;
        cout << "CFits::CrossCorrelate: Warning: nPixMaxLeft too large, set to " << nPixMaxLeft << endl;
      }
      int nPixMaxRight = I_NPixMaxRight;
      if ( nPixMaxRight >= I_Size ){
        nPixMaxRight = I_Size - 1;
        cout << "CFits::CrossCorrelate: Warning: nPixMaxRight too large, set to " << nPixMaxRight << endl;
      }

      int I_Pix = 0. - nPixMaxLeft;
      int I_NPixMove = nPixMaxLeft + nPixMaxRight + 1;
      int run = 0;
      #ifdef __DEBUG_FITS_CROSSCORRELATE__
        cout << "CFits::CrossCorrelate: I_Pix = " << I_Pix << endl;
        cout << "CFits::CrossCorrelate: I_NPixMove = " << I_NPixMove << endl;
      #endif

      ndarray::Array< double, 1, 1 > DA1_StaticTemp;
      ndarray::Array< double, 1, 1 > DA1_MovingTemp;
      ndarray::Array< double, 1, 1 > DA1_Diff;
      ndarray::Array< double, 1, 1 > DA1_ChiSquare = ndarray::allocate( I_NPixMove );
      ndarray::Array< int, 1, 1 > IA1_NPix = ndarray::allocate( I_NPixMove );

      for (int i = I_Pix; i <= nPixMaxRight; i++){
        if (i < 0){
          #ifdef __DEBUG_FITS_CROSSCORRELATE__
            cout << "CFits::CrossCorrelate: i=" << i << " < 0: new array size = " << DA1_Static.getShape()[ 0 ] + i << endl;
            cout << "CFits::CrossCorrelate: i=" << i << " < 0: DA1_StaticTemp.size() = " << DA1_StaticTemp.getShape()[ 0 ] << endl;
          #endif
          DA1_StaticTemp = ndarray::allocate( DA1_Static.getShape()[ 0 ] + i );
          #ifdef __DEBUG_FITS_CROSSCORRELATE__
            cout << "CFits::CrossCorrelate: i=" << i << " < 0: DA1_MovingTemp.getShape()[ 0  = " << DA1_MovingTemp.getShape()[ 0 ] << endl;
          #endif
          DA1_MovingTemp = ndarray::allocate( DA1_Static.getShape()[ 0 ] + i );
          DA1_Diff = ndarray::allocate( DA1_Static.getShape()[ 0 ] + i );
          #ifdef __DEBUG_FITS_CROSSCORRELATE__
            cout << "CFits::CrossCorrelate: i," << i << " < 0: DA1_Static = " << DA1_Static << endl;
            cout << "CFits::CrossCorrelate: i," << i << " < 0: Setting DA1_StaticTemp to DA1_Static(Range(0," << DA1_Static.getShape()[ 0 ] + i - 1 << "))" << endl;
          #endif
          DA1_StaticTemp[ ndarray::view() ] = DA1_Static[ ndarray::view( 0, DA1_Static.getShape()[ 0 ] + i ) ];
          #ifdef __DEBUG_FITS_CROSSCORRELATE__
            cout << "CFits::CrossCorrelate: i," << i << " < 0: DA1_moving = " << DA1_Moving << endl;
            cout << "CFits::CrossCorrelate: i," << i << " < 0: Setting DA1_MovingTemp to DA1_Moving(Range(" << 0-i << "," << DA1_Moving.getShape()[ 0 ] - 1 << "))" << endl;
          #endif
          DA1_MovingTemp[ ndarray::view() ] = DA1_Moving[ ndarray::view( 0 - i, DA1_Moving.getShape()[ 0 ] )];
        } 
        else{
          #ifdef __DEBUG_FITS_CROSSCORRELATE__
            cout << "CFits::CrossCorrelate: i=" << i << " >= 0: new array size = " << DA1_Static.getShape()[ 0 ] + i << endl;
          #endif
          DA1_StaticTemp = ndarray::allocate( DA1_Static.getShape()[ 0 ] - i );
          DA1_MovingTemp = ndarray::allocate( DA1_Static.getShape()[ 0 ] - i );
          DA1_Diff = ndarray::allocate( DA1_Static.getShape()[ 0 ] - i );
          DA1_StaticTemp[ ndarray::view() ] = DA1_Static[ ndarray::view( i, DA1_Static.getShape()[ 0 ] ) ];
          DA1_MovingTemp[ ndarray::view() ] = DA1_Moving[ ndarray::view( 0, DA1_Moving.getShape()[ 0 ] - i ) ];
        }
        #ifdef __DEBUG_FITS_CROSSCORRELATE__
          cout << "CFits::CrossCorrelate: DA1_StaticTemp = " << DA1_StaticTemp << endl;
          cout << "CFits::CrossCorrelate: DA1_MovingTemp = " << DA1_MovingTemp << endl;
        #endif

        /// Calculate difference of both arrays and square
        DA1_Diff.deep() = DA1_StaticTemp - DA1_MovingTemp;
        DA1_Diff.deep() = DA1_Diff * DA1_Diff;
        #ifdef __DEBUG_FITS_CROSSCORRELATE__
          cout << "CFits::CrossCorrelate: DA1_Diff = " << DA1_Diff << endl;
        #endif

        /// Calculate sum of squares of differences
        DA1_ChiSquare[ run ] = std::accumulate( DA1_Diff.begin(), DA1_Diff.end(), 0. ) / double( DA1_Diff.getShape()[ 0 ] );
        #ifdef __DEBUG_FITS_CROSSCORRELATE__
          cout << "CFits::CrossCorrelate: DA1_ChiSquare(run = " << run << ") = " << DA1_ChiSquare[run] << endl;
        #endif

        /// Save number of pixels used for calculations
    //    IA1_NPix(run) = DA1_StaticTemp.size();
    //    #ifdef __DEBUG_FITS_CROSSCORRELATE__
    //      cout << "CFits::CrossCorrelate: IA1_NPix(run = " << run << ") = " << IA1_NPix(run) << endl;
    //    #endif

        run++;
      }
      /// Normalise DA1_ChiSquare to number of pixels
    //  Array<double, 1> DA1_NPix(IA1_NPix.size());
    //  if (!this->CastIntArrToDblArr(IA1_NPix, DA1_NPix)){
    //    cout << "CFits::CrossCorrelate: ERROR: CastIntArrToDblArr(IA1_NPix) returned FALSE" << endl;
    //    returm false;
    //  }
    //  DA1_ChiSquare = DA1_ChiSquare / DA1_NPix;
      #ifdef __DEBUG_FITS_CROSSCORRELATE__
        cout << "CFits::CrossCorrelate: DA1_ChiSquare = " << DA1_ChiSquare << endl;
      #endif


      size_t minInd = minIndex( DA1_ChiSquare );
      #ifdef __DEBUG_CFITS_CROSSCORRELATE__
        cout << "CFits::CrossCorrelate: minInd = " << minInd << endl;
        cout << "CFits::CrossCorrelate: DA1_ChiSquare(minInd) = " << DA1_ChiSquare[ minInd ] << endl;
      #endif
      if ( ( minInd == 0 ) || ( minInd == ( DA1_ChiSquare.getShape()[ 0 ] - 1 ) ) ){
        D_Out = double( minInd - nPixMaxLeft );
        D_ChiSquare_Out = DA1_ChiSquare[ minInd ];
      }
      int I_Start = minInd - 2;
      if (I_Start < 0)
        I_Start = 0;
      int I_End = minInd + 2;
      if ( I_End >= DA1_ChiSquare.getShape()[ 0 ] )
        I_End = DA1_ChiSquare.getShape()[ 0 ];
      ndarray::Array< double, 1, 1 > P_D_A1_X = indGenNdArr( double( DA1_ChiSquare.getShape()[ 0 ] ) );
      ndarray::Array< double, 1, 1 > D_A1_X = ndarray::allocate( I_End - I_Start);
      D_A1_X = P_D_A1_X[ ndarray::view( I_Start, I_End ) ];
      ndarray::Array< double, 1, 1 > D_A1_ChiSqu = ndarray::allocate( I_End - I_Start );
      D_A1_ChiSqu = DA1_ChiSquare[ ndarray::view( I_Start, I_End ) ];
      ndarray::Array< double, 1, 1 > P_D_A1_MeasureErrors = replicate(1., D_A1_ChiSqu.size());
      ndarray::Array< double, 1, 1 > D_A1_Guess = ndarray::allocate( 4 );
      D_A1_Guess[ 0 ] = max( DA1_ChiSquare );
      D_A1_Guess[ 1 ] = 0. - ( max( DA1_ChiSquare ) - DA1_ChiSquare[ minInd ] );
      D_A1_Guess[ 2 ] = double( minInd );
      D_A1_Guess[ 3 ] = 2.;
      ndarray::Array< int, 2, 1 > I_A2_Limited = ndarray::allocate( ndarray::makeVector( 4, 2 ) );
      I_A2_Limited[ ndarray::makeVector( 0, 0 ) ] = 1;
      I_A2_Limited[ ndarray::makeVector( 0, 1 ) ] = 1;
      I_A2_Limited[ ndarray::makeVector( 1, 0 ) ] = 1;
      I_A2_Limited[ ndarray::makeVector( 1, 1 ) ] = 1;
      I_A2_Limited[ ndarray::makeVector( 2, 0 ) ] = 1;
      I_A2_Limited[ ndarray::makeVector( 2, 1 ) ] = 1;
      I_A2_Limited[ ndarray::makeVector( 3, 0 ) ] = 1;
      I_A2_Limited[ ndarray::makeVector( 3, 1 ) ] = 1;
      ndarray::Array< double, 2, 1 > D_A2_Limits = ndarray::allocate( ndarray::makeVector( 4, 2 ) );
      D_A2_Limits[ ndarray::makeVector( 0, 0 ) ] = D_A1_Guess[ 0 ] / 2.;
      D_A2_Limits[ ndarray::makeVector( 0, 1 ) ] = D_A1_Guess[ 0 ] * 1.1;
      if ( D_A2_Limits[ ndarray::makeVector( 0, 1 ) ] < D_A2_Limits[ ndarray::makeVector( 0, 0 ) ] ){
        cout << "CFits::CrossCorrelate: ERROR: D_A2_Limits(0,1) < D_A2_Limits(0,0)" << endl;
        return false;
      }
      D_A2_Limits[ ndarray::makeVector( 1, 0 ) ] = 1.5 * D_A1_Guess[ 1 ];
      D_A2_Limits[ ndarray::makeVector( 1, 1 ) ] = 0.;
      if ( D_A2_Limits[ ndarray::makeVector( 1, 1 ) ] < D_A2_Limits[ ndarray::makeVector( 1, 0 ) ] ){
        cout << "CFits::CrossCorrelate: ERROR: D_A2_Limits(1,1) < D_A2_Limits(1,0)" << endl;
        return false;
      }
      D_A2_Limits[ ndarray::makeVector( 2, 0 ) ] = double( minInd - 1 );
      D_A2_Limits[ ndarray::makeVector( 2, 1 ) ] = double( minInd + 1 );
      if ( D_A2_Limits[ ndarray::makeVector( 2, 1 ) ] < D_A2_Limits[ ndarray::makeVector( 2, 0 ) ] ){
        cout << "CFits::CrossCorrelate: ERROR: D_A2_Limits(2,1) < D_A2_Limits(2,0)" << endl;
        return false;
      }
      D_A2_Limits[ ndarray::makeVector( 3, 0 ) ] = 0.;
      D_A2_Limits[ ndarray::makeVector( 3, 1 ) ] = DA1_ChiSquare.getShape()[ 0 ];
      if ( D_A2_Limits[ ndarray::makeVector( 3, 1 ) ] < D_A2_Limits[ ndarray::makeVector( 3, 0 ) ] ){
        cout << "CFits::CrossCorrelate: ERROR: D_A2_Limits(3,1) < D_A2_Limits(3,0)" << endl;
        return false;
      }
      ndarray::Array< double, 1, 1 > D_A1_GaussCoeffs = ndarray::allocate( 4 );
      D_A1_GaussCoeffs.deep() = 0.;
      ndarray::Array< double, 1, 1 > D_A1_EGaussCoeffs = ndarray::allocate( 4 );
      D_A1_EGaussCoeffs.deep() = 0.;
      if ( !MPFitGaussLim( D_A1_X,
                           D_A1_ChiSqu,
                           P_D_A1_MeasureErrors,
                           D_A1_Guess,
                           I_A2_Limited,
                           D_A2_Limits,
                           true,
                           false,
                           D_A1_GaussCoeffs,
                           D_A1_EGaussCoeffs ) ){
        cout << "CFits::CrossCorrelate: WARNING: GaussFit returned FALSE" << endl;
        //        return false;
      }
      #ifdef __DEBUG_CFITS_CROSSCORRELATE__
        cout << "CFits::CrossCorrelate: D_A1_X = " << D_A1_X << endl;
        cout << "CFits::CrossCorrelate: D_A1_ChiSqu = " << D_A1_ChiSqu << endl;
        cout << "CFits::CrossCorrelate: D_A1_Guess = " << D_A1_Guess << endl;
        cout << "CFits::CrossCorrelate: D_A2_Limits = " << D_A2_Limits << endl;
        cout << "CFits::CrossCorrelate: D_A1_GaussCoeffs = " << D_A1_GaussCoeffs << endl;
      #endif
      D_Out = D_A1_GaussCoeffs[ 2 ] - double( nPixMaxLeft );
      D_ChiSquare_Out = D_A1_GaussCoeffs[ 1 ] + D_A1_GaussCoeffs[ 0 ];
      #ifdef __DEBUG_FITS_CROSSCORRELATE__
        cout << "CFits::CrossCorrelate: D_Out = " << D_Out << endl;
        cout << "CFits::CrossCorrelate: D_ChiSquare_Out = " << D_ChiSquare_Out << endl;
      #endif
      return true;
    }

    /**
      CrossCorrelate
     **/
    template< typename T >
    bool crossCorrelate( ndarray::Array< T, 1, 1 > const& DA1_Static,
                         ndarray::Array< T, 1> const& DA1_Moving,
                         int const I_NPixMaxLeft,
                         int const I_NPixMaxRight,
                         int &I_Out,
                         double &D_ChiSquare_Out){
      /// Check that both arrays have the same size
      if ( DA1_Moving.getShape()[ 0 ] != DA1_Static.getShape()[ 0 ] ){
        cout << "CFits::CrossCorrelate: ERROR: DA1_Moving.size() = " << DA1_Moving.getShape()[ 0 ] << " != DA1_Static.size() = " << DA1_Static.getShape()[ 0 ] << endl;
        return false;
      }

      int I_Size = DA1_Static.getShape()[ 0 ];

      /// Check I_NPixMaxLeft and I_NPixMaxRight
      int nPixMaxLeft = I_NPixMaxLeft;
      if ( nPixMaxLeft >= I_Size ){
        nPixMaxLeft = I_Size - 1;
        cout << "CFits::CrossCorrelate: Warning: nPixMaxLeft too large, set to " << nPixMaxLeft << endl;
      }
      int nPixMaxRight = I_NPixMaxRight;
      if ( nPixMaxRight >= I_Size){
        nPixMaxRight = I_Size - 1;
        cout << "CFits::CrossCorrelate: Warning: nPixMaxRight too large, set to " << nPixMaxRight << endl;
      }

      int I_Pix = 0. - nPixMaxLeft;
      int I_NPixMove = nPixMaxLeft + nPixMaxRight + 1;
      int run = 0;
      #ifdef __DEBUG_FITS_CROSSCORRELATE__
        cout << "CFits::CrossCorrelate: I_Pix = " << I_Pix << endl;
        cout << "CFits::CrossCorrelate: I_NPixMove = " << I_NPixMove << endl;
      #endif

      ndarray::Array< double, 1, 1 > DA1_StaticTemp;
      ndarray::Array< double, 1, 1 > DA1_MovingTemp;
      ndarray::Array< double, 1, 1 > DA1_Diff;
      ndarray::Array< double, 1, 1 > DA1_ChiSquare = ndarray::allocate( I_NPixMove );
      ndarray::Array< int, 1, 1 > IA1_NPix = ndarray::allocate( I_NPixMove );

      for ( int i = I_Pix; i <= nPixMaxRight; i++ ){
        if ( i < 0 ){
          #ifdef __DEBUG_FITS_CROSSCORRELATE__
            cout << "CFits::CrossCorrelate: i=" << i << " < 0: new array size = " << DA1_Static.size()+i << endl;
            cout << "CFits::CrossCorrelate: i=" << i << " < 0: DA1_StaticTemp.size() = " << DA1_StaticTemp.size() << endl;
          #endif
          DA1_StaticTemp = ndarray::allocate( DA1_Static.getShape()[ 0 ] + i );
          #ifdef __DEBUG_FITS_CROSSCORRELATE__
            cout << "CFits::CrossCorrelate: i=" << i << " < 0: DA1_MovingTemp.size() = " << DA1_MovingTemp.getShape()[ 0 ] << endl;
          #endif
          DA1_MovingTemp = ndarray::allocate (DA1_Static.getShape()[ 0 ] + i );
          DA1_Diff = ndarray::allocate( DA1_Static.getShape()[ 0 ] + i );
          #ifdef __DEBUG_FITS_CROSSCORRELATE__
            cout << "CFits::CrossCorrelate: i," << i << " < 0: DA1_Static = " << DA1_Static << endl;
            cout << "CFits::CrossCorrelate: i," << i << " < 0: Setting DA1_StaticTemp to DA1_Static(Range(0," << DA1_Static.size() + i - 1 << "))" << endl;
          #endif
          DA1_StaticTemp.deep() = DA1_Static[ ndarray::view( 0, DA1_Static.getShape()[ 0 ] + i ) ];
          #ifdef __DEBUG_FITS_CROSSCORRELATE__
            cout << "CFits::CrossCorrelate: i," << i << " < 0: DA1_moving = " << DA1_Moving << endl;
            cout << "CFits::CrossCorrelate: i," << i << " < 0: Setting DA1_MovingTemp to DA1_Moving(Range(" << 0-i << "," << DA1_Moving.getShape() << "))" << endl;
          #endif
          DA1_MovingTemp.deep() = DA1_Moving[ ndarray::view( 0 - i, DA1_Moving.getShape()[ 0 ] ) ];
        } 
        else{
          #ifdef __DEBUG_FITS_CROSSCORRELATE__
            cout << "CFits::CrossCorrelate: i=" << i << " >= 0: new array size = " << DA1_Static.getShape()[ 0 ]+i << endl;
          #endif
          DA1_StaticTemp = ndarray::allocate( DA1_Static.getShape()[ 0 ] - i );
          DA1_MovingTemp = ndarray::allocate( DA1_Static.getShape()[ 0 ] - i );
          DA1_Diff = ndarray::allocate( DA1_Static.getShape()[ 0 ] - i );
          DA1_StaticTemp.deep() = DA1_Static[ ndarray::view( i, DA1_Static.getShape()[ 0 ] ) ];
          DA1_MovingTemp.deep() = DA1_Moving[ ndarray::view( 0, DA1_Moving.getShape()[ 0 ] - i ) ];
        }
        #ifdef __DEBUG_FITS_CROSSCORRELATE__
          cout << "CFits::CrossCorrelate: DA1_StaticTemp = " << DA1_StaticTemp << endl;
          cout << "CFits::CrossCorrelate: DA1_MovingTemp = " << DA1_MovingTemp << endl;
        #endif

        /// Calculate difference of both arrays and square
        DA1_Diff.deep() = DA1_StaticTemp - DA1_MovingTemp;
        DA1_Diff.deep() = DA1_Diff * DA1_Diff;
        #ifdef __DEBUG_FITS_CROSSCORRELATE__
          cout << "CFits::CrossCorrelate: DA1_Diff = " << DA1_Diff << endl;
        #endif

        /// Calculate sum of squares of differences
        DA1_ChiSquare[ run ] = std::accumulate( DA1_Diff.begin(), DA1_Diff.end(), 0. ) / DA1_Diff.getShape()[ 0 ];
        #ifdef __DEBUG_FITS_CROSSCORRELATE__
          cout << "CFits::CrossCorrelate: DA1_ChiSquare(run = " << run << ") = " << DA1_ChiSquare[ run ] << endl;
        #endif

        /// Save number of pixels used for calculations
        IA1_NPix[ run ] = DA1_StaticTemp.getShape()[ 0 ];
        #ifdef __DEBUG_FITS_CROSSCORRELATE__
          cout << "CFits::CrossCorrelate: IA1_NPix(run = " << run << ") = " << IA1_NPix[ run ] << endl;
        #endif

        run++;
      }
      /// Normalise DA1_ChiSquare to number of pixels
      DA1_ChiSquare.deep() = DA1_ChiSquare / IA1_NPix;
      #ifdef __DEBUG_FITS_CROSSCORRELATE__
        cout << "CFits::CrossCorrelate: DA1_ChiSquare = " << DA1_ChiSquare << endl;
      #endif

      size_t minInd = minIndex( DA1_ChiSquare );
      D_ChiSquare_Out = DA1_ChiSquare( minInd );
      I_Out = minInd - nPixMaxLeft;
      #ifdef __DEBUG_FITS_CROSSCORRELATE__
        cout << "CFits::CrossCorrelate: minInd " << minInd << endl;
        cout << "CFits::CrossCorrelate: I_Out = " << I_Out << endl;
      #endif
      return true;
    }
    
    /**
      LsToFit
     **/
    template< typename T >
    bool lsToFit( ndarray::Array< T, 1, 1 > const& XXVecArr, 
                  ndarray::Array< T, 1, 1 > const& YVecArr, 
                  T const& XM, 
                  T & D_Out){
      ///Normalize to preserve significance.
      ndarray::Array< T, 1, 1 > XVecArr = ndarray::allocate( XXVecArr.getShape()[ 0 ] );
      XVecArr.deep() = XXVecArr - XXVecArr[0];

      int NDegree = 2;
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: NDegree set to " << NDegree << endl;
      #endif

      long N = XXVecArr.size();
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: N set to " << N << endl;
      #endif

      ///Correlation matrix
      ndarray::Array< T, 2, 1 > CorrMArr = ndarray::allocate( ndarray::makeVector( NDegree + 1, NDegree + 1 ) );

      ndarray::Array< T, 1, 1 > BVecArr = ndarray::allocate( NDegree + 1 );

      ///0 - Form the normal equations
      CorrMArr[ ndarray::makeVector( 0, 0 ) ] = N;
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: CorrMArr(0,0) set to " << CorrMArr[ndarray::makeVector(0,0)] << endl;
      #endif

      BVecArr[0] = std::accumulate( YVecArr.begin(), YVecArr.end(), 0. );
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: BVecArr(0) set to " << BVecArr[0] << endl;
      #endif

      ndarray::Array< T, 1, 1 > ZVecArr = ndarray::allocate( XXVecArr.getShape()[ 0 ] );
      ZVecArr.deep() = XVecArr;
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: ZVecArr set to " << ZVecArr << endl;
      #endif

      ndarray::Array< T, 1, 1 > TempVecArr = ndarray::allocate( YVecArr.getShape()[ 0 ] );
      TempVecArr.deep() = YVecArr;
      TempVecArr.deep() = TempVecArr * ZVecArr;
      BVecArr[ 1 ] = std::accumulate( TempVecArr.begin(), TempVecArr.end(), 0. );
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: BVecArr(1) set to " << BVecArr[1] << endl;
      #endif

      CorrMArr[ ndarray::makeVector( 0, 1 ) ] = std::accumulate( ZVecArr.begin(), ZVecArr.end(), 0. );
      CorrMArr[ ndarray::makeVector( 1, 0 ) ] = CorrMArr[ ndarray::makeVector( 0, 1 ) ];
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: CorrMArr(0,1) set to " << CorrMArr[ndarray::makeVector(0,1)] << endl;
        cout << "CFits::LsToFit: CorrMArr(1,0) set to " << CorrMArr[ndarray::makeVector(1,0)] << endl;
      #endif

      ZVecArr.deep() = ZVecArr * XVecArr;
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: ZVecArr set to " << ZVecArr << endl;
      #endif

      TempVecArr = ndarray::allocate( YVecArr.getShape()[ 0 ] );
      TempVecArr.deep() = YVecArr;
      TempVecArr.deep() = TempVecArr * ZVecArr;
      BVecArr[ 2 ] = std::accumulate( TempVecArr.begin(), TempVecArr.end(), 0. );
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: BVecArr(2) set to " << BVecArr[2] << endl;
      #endif

      CorrMArr[ ndarray::makeVector( 0, 2 ) ] = CorrMArr[ ndarray::makeVector( 1, 1 ) ] = CorrMArr[ ndarray::makeVector( 2, 0 ) ] = std::accumulate( ZVecArr.begin(), ZVecArr.end(), 0. );
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: CorrMArr(0,2) set to " << CorrMArr[ndarray::makeVector(0,2)] << endl;
        cout << "CFits::LsToFit: CorrMArr(1,1) set to " << CorrMArr[ndarray::makeVector(1,1)] << endl;
        cout << "CFits::LsToFit: CorrMArr(2,0) set to " << CorrMArr[ndarray::makeVector(2,0)] << endl;
      #endif

      ZVecArr.deep() = ZVecArr * XVecArr;
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: ZVecArr set to " << ZVecArr << endl;
      #endif

      CorrMArr[ ndarray::makeVector( 1, 2 ) ] = CorrMArr[ ndarray::makeVector( 2, 1 ) ] = sum(ZVecArr);
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: CorrMArr(1,2) set to " << CorrMArr[ndarray::makeVector(1,2)] << endl;
        cout << "CFits::LsToFit: CorrMArr(2,1) set to " << CorrMArr[ndarray::makeVector(2,1)] << endl;
      #endif

      TempVecArr = ndarray::allocate( ZVecArr.getShape()[ 0 ] );
      TempVecArr.deep() = ZVecArr;
      TempVecArr.deep() = TempVecArr * XVecArr;
      CorrMArr[ ndarray::makeVector( 2, 2 ) ] = std::accumulate( TempVecArr.begin(), TempVecArr.end(), 0. );
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: CorrMArr(2,2) set to " << CorrMArr[ndarray::makeVector(2,2)] << endl;
      #endif

      ndarray::Array< T, 2, 1 > CorrInvMArr = ndarray::allocate( ndarray::makeVector( CorrMArr.getShape()[ 0 ], CorrMArr.getShape()[ 1 ] ) );
      CorrInvMArr.asEigen() = CorrMArr.asEigen().inverse();
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: CorrInvMArr set to " << CorrInvMArr << endl;
      #endif
      ndarray::Array< T, 1, 1 > p_CVecArr = ndarray::allocate( BVecArr.getShape()[ 1 ] );
      p_CVecArr.asEigen() = BVecArr.asEigen().transpose() * CorrInvMArr.asEigen();
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: p_CVecArr set to " << p_CVecArr << endl;
      #endif

      double XM0 = XM - XXVecArr[0];
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: XM0 set to " << XM0 << endl;
      #endif

      D_Out = p_CVecArr[ 0 ] + ( p_CVecArr[ 1 ] * XM0 ) + ( p_CVecArr[ 2 ] * pow( XM0, 2 ) );
      #ifdef __DEBUG_FITS_LSTOFIT__
        cout << "CFits::LsToFit: D_Out set to " << D_Out << endl;
      #endif
      return true;
    }

    template< typename T >    
    bool hInterPol( ndarray::Array< T, 1, 1 > const& VVecArr,
                    ndarray::Array< T, 1, 1 > const& XVecArr,
                    ndarray::Array< int, 1, 1 > & SVecArr,
                    ndarray::Array< T, 1, 1 > const& UVecArr,
                    std::vector< string > const& CS_A1_In,
                    ndarray::Array< T, 1, 1 > & D1_Out){
      #ifdef __DEBUG_FITS_INTERPOL__
        cout << "CFits::HInterPol: VVecArr.size() = " << VVecArr.getShape()[ 0 ] << endl;
        cout << "CFits::HInterPol: XVecArr.size() = " << XVecArr.getShape()[ 0 ] << endl;
        cout << "CFits::HInterPol: SVecArr.size() = " << SVecArr.getShape()[ 0 ] << endl;
        cout << "CFits::HInterPol: UVecArr.size() = " << UVecArr.getShape()[ 0 ] << endl;
        cout << "CFits::HInterPol: CS_A1_In.size() = " << CS_A1_In.size() << endl;
      #endif

      int M = VVecArr.getShape()[ 0 ];
//      firstIndex i;

      ndarray::Array< int, 1, 1 > IA1_Temp = ndarray::allocate( SVecArr.getShape()[ 0 ] );
      IA1_Temp.deep() = 0;

      ndarray::Array< T, 1, 1 > DA1_Temp = ndarray::allocate( SVecArr.getShape()[ 0 ] );
      DA1_Temp.deep() = 0.;

      ndarray::Array< T, 1, 1 > DA1_TempA = ndarray::allocate( SVecArr.getShape()[ 0 ] );
      DA1_TempA.deep() = 0.;

      ndarray::Array< T, 1, 1 > DA1_VTempP1 = ndarray::allocate( SVecArr.getShape()[ 0 ] );
      DA1_VTempP1.deep() = 0.;

      ndarray::Array< T, 1, 1 > DA1_VTemp = ndarray::allocate( SVecArr.getShape()[ 0 ] );
      DA1_VTemp.deep() = 0.;

      ndarray::Array< T, 1, 1 > DA1_XTempP1 = ndarray::allocate( SVecArr.getShape()[ 0 ] );
      DA1_XTempP1.deep() = 0.;

      ndarray::Array< T, 1, 1 > DA1_XTemp = ndarray::allocate( SVecArr.getShape()[ 0 ] );
      DA1_XTemp.deep() = 0.;

      ndarray::Array< int, 1, 1 > IA1_STemp = ndarray::allocate( SVecArr.getShape()[ 0 ] );
      IA1_STemp.deep() = 0;

      ndarray::Array< T, 1, 1 > PVecArr = ndarray::allocate( SVecArr.getShape()[ 0 ] );
      PVecArr.deep() = 0.;

      ndarray::Array< T, 1, 1 > TmpVecArr = indGenNdArr( T( 4 ) );

      ndarray::Array< T, 1, 1 > T1VecArr = ndarray::allocate( 4 );
      T1VecArr.deep() = 0.;

      ndarray::Array< T, 1, 1 > T2VecArr = ndarray::allocate( 4 );
      T2VecArr.deep() = 0.;

      ndarray::Array< T, 1, 1 > X1VecArr = ndarray::allocate( SVecArr.getShape()[ 0 ] );
      X1VecArr.deep() = 0.;

      ndarray::Array< T, 1, 1 > X0VecArr = ndarray::allocate( SVecArr.getShape()[ 0 ] );
      X0VecArr.deep() = 0.;

      ndarray::Array< T, 1, 1 > X2VecArr = ndarray::allocate( SVecArr.getShape()[ 0 ] );
      X2VecArr.deep() = 0.;

      ndarray::Array< T, 1, 1 > X0Arr = ndarray::allocate( 4 );
      X0Arr.deep() = 0.;

      ndarray::Array< T, 1, 1 > V0Arr = ndarray::allocate( 4 );
      V0Arr.deep() = 0.;

      ndarray::Array< T, 1, 1 > QArr = ndarray::allocate( SVecArr.getShape()[ 0 ] );
      QArr.deep() = 0.;

      /**
      Clip interval, which forces extrapolation.
      u[i] is between x[s[i]] and x[s[i]+1].
       **/
      int s0int;
      double s0;
      /// Least square fit quadratic, 4 points
      if ( pfs::drp::stella::utils::KeyWord_Set( CS_A1_In, std::string("LSQUADRATIC") ) >= 0 ){
        # ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::HInterPol: KeywordSet(LSQUADRATIC)" << endl;
        #endif
        SVecArr.deep() = whereLt( SVecArr, 1, 1, SVecArr );
        SVecArr.deep() = whereGt( SVecArr, M - 3, M - 3, SVecArr );
        # ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::HInterPol: LSQUADRATIC: SVecArr.size() set to " << SVecArr.getShape()[ 0 ] << endl;
        #endif
        PVecArr.deep() = VVecArr[ 0 ];   /// Result
        for (int m = 0; m < SVecArr.getShape()[ 0 ]; m++)
        {
          s0 = double( SVecArr[ m ] ) - 1.;
          s0int = (int)s0;
          TmpVecArr.deep() = TmpVecArr + s0;
          T1VecArr.deep() = XVecArr[ ndarray::view( s0int, s0int + 4 ) ];
          T2VecArr.deep() = VVecArr[ ndarray::view( s0int, s0int + 4 ) ];
          #ifdef __DEBUG_FITS_INTERPOL__
                cout << "CFits::HInterPol: Starting LsToFit(T1VecArr, T2VecArr, UVecArr(m)" << endl;
          #endif
          if ( !lsToFit( T1VecArr, T2VecArr, UVecArr[ m ], PVecArr[ m ] ) )
            return false;
        }
      }
      else if ( pfs::drp::stella::utils::KeyWord_Set( CS_A1_In, std::string( "QUADRATIC" ) ) >= 0 ){
        # ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::HInterPol: KeywordSet(QUADRATIC)" << endl;
        #endif
        SVecArr.deep() = whereLt( SVecArr, 1, 1, SVecArr );
        SVecArr.deep() = whereGt( SVecArr, M - 2, M - 2, SVecArr );
        # ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::HInterPol: QUADRATIC: SVecArr.size() set to " << SVecArr.getShape()[ 0 ] << endl;
        #endif

        X1VecArr.deep() = getSubArray( XVecArr,
                                       SVecArr );

        IA1_Temp.deep() = SVecArr - 1;

        X0VecArr.deep() = getSubArray( XVecArr,
                                       IA1_Temp);

        IA1_Temp.deep() = SVecArr + 1;
        X2VecArr.deep() = getSubArray( XVecArr,
                                       IA1_Temp);

        IA1_Temp.deep() = SVecArr - 1;
        DA1_Temp.deep() = getSubArray( VVecArr,
                                       IA1_Temp);

        IA1_Temp.deep() = SVecArr + 1;
        DA1_TempA.deep() = getSubArray( VVecArr,
                                        IA1_Temp);

        PVecArr.deep() = DA1_Temp
                         * ( UVecArr - X1VecArr ) * ( UVecArr - X2VecArr )
                         / ( ( X0VecArr - X1VecArr ) * ( X0VecArr - X2VecArr ) )
                         + DA1_TempA
                         * ( UVecArr - X0VecArr ) * ( UVecArr - X1VecArr )
                         / ( ( X2VecArr - X0VecArr ) * ( X2VecArr - X1VecArr ) );
      }
      else if ( pfs::drp::stella::utils::KeyWord_Set( CS_A1_In, std::string( "SPLINE" ) ) >= 0 ){
        # ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::HInterPol: KeywordSet(SPLINE)" << endl;
        #endif
        SVecArr.deep() = whereLt( SVecArr, 1, 1, SVecArr );
        SVecArr.deep() = whereGt( SVecArr, M-3, M-3, SVecArr );
        # ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::HInterPol: SPLINE: SVecArr.size() set to " << SVecArr.getShape()[ 0 ] << endl;
        #endif
        PVecArr = ndarray::allocate( SVecArr.getShape()[ 0 ] );
        PVecArr.deep() = VVecArr( 0 );
        int SOld = -1;
        for (int m = 0; m < SVecArr.getShape()[ 0 ]; m++){
          s0 = SVecArr[ m ] - 1.;
          s0int = (int)s0;
          if (abs(SOld - s0int) > 0){
            X0Arr = ndarray::allocate( 4 );
            X0Arr.deep() = XVecArr[ ndarray::view( s0int, s0int + 4 ) ];
            V0Arr = ndarray::allocate( 4 );
            V0Arr.deep() = XVecArr[ ndarray::view( s0int, s0int + 4 ) ];
            QArr = splineI( X0Arr, V0Arr );
            SOld = s0int;
          }
          if ( !splInt( X0Arr, 
                        V0Arr, 
                        QArr, 
                        UVecArr[ m ], 
                        PVecArr[ m ] ) ){
            cout << "CFits::HInterPol: ERROR: SplInt(X0Arr, V0Arr, QArr, UVecArr(m), PVecArr(m)) returned FALSE" << endl;
            return false;
          }
        }
      }
      else  /// Linear, not regular
      {
        DA1_XTemp.deep() = getSubArray( XVecArr,
                                        SVecArr);
        # ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::HInterPol: DA1_XTemp set to " << DA1_XTemp << endl;
        #endif
        DA1_VTemp.deep() = getSubArray( VVecArr,
                                        SVecArr);
        # ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::HInterPol: DA1_VTemp set to " << DA1_VTemp << endl;
        #endif

        IA1_STemp.deep() = SVecArr + 1;
        # ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::HInterPol: IA1_STemp set to " << IA1_STemp << endl;
        #endif

        DA1_XTempP1 = getSubArray( XVecArr,
                                   IA1_STemp);
        # ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::HInterPol: DA1_XTempP1 set to " << DA1_XTempP1 << endl;
        #endif

        DA1_VTempP1 = getSubArray( VVecArr,
                                   IA1_STemp);
        # ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::HInterPol: DA1_VTempP1 set to " << DA1_VTempP1 << endl;
        #endif

        PVecArr.deep() = (UVecArr - DA1_XTemp)
                         * (DA1_VTempP1 - DA1_VTemp)
                         / (DA1_XTempP1 - DA1_XTemp)
                         + DA1_VTemp;
      }
      #ifdef __DEBUG_FITS_INTERPOL__
        cout << "CFits::HInterPol: Ready: Returning PVecArr = " << PVecArr << endl;
      #endif

      D1_Out = ndarray::allocate( PVecArr.getShape()[ 0 ]);
      D1_Out.deep() = PVecArr;

      return true;
    }
    
    template< typename T, typename U >
    ndarray::Array< U, 1, 1 > whereLt( ndarray::Array< T, 1, 1 > const& arrayToCompareTo,
                                       T const valueToCompareTo, 
                                       U const valueIfArrayValueLowerThanValueToCompareTo,
                                       U const valueIfArrayValueGreaterOrEqualToValueToCopareTo ){
      ndarray::Array< U, 1, 1 > arrOut = ndarray::allocate( arrayToCompareTo.getShape()[ 0 ] );
      auto itOut = arrOut.begin();
      for ( auto itComp = arrayToCompareTo.begin(); itComp != arrayToCompareTo.end(); ++itComp, ++itOut ){
        if ( *itComp < valueToCompareTo )
          *itOut = valueIfArrayValueLowerThanValueToCompareTo;
        else
          *itOut = valueIfArrayValueGreaterOrEqualToValueToCopareTo;
      }
      return arrOut;
    }
    
    template< typename T, typename U >
    ndarray::Array< U, 1, 1 > whereLt( ndarray::Array< T, 1, 1 > const& arrayToCompareTo,
                                       T const valueToCompareTo, 
                                       U const valueIfArrayValueLowerThanValueToCompareTo,
                                       ndarray::Array< U, 1, 1 > const& valuesIfArrayValueGreaterOrEqualToValueToCopareTo ){
      ndarray::Array< U, 1, 1 > arrOut = ndarray::allocate( arrayToCompareTo.getShape()[ 0 ] );
      auto itOut = arrOut.begin();
      auto itElse = valuesIfArrayValueGreaterOrEqualToValueToCopareTo.begin();
      for ( auto itComp = arrayToCompareTo.begin(); itComp != arrayToCompareTo.end(); ++itComp, ++itOut, ++itElse ){
        if ( *itComp < valueToCompareTo )
          *itOut = valueIfArrayValueLowerThanValueToCompareTo;
        else
          *itOut = *itElse;
      }
      return arrOut;
    }
    
    template< typename T, typename U >
    ndarray::Array< U, 1, 1 > whereLe( ndarray::Array< T, 1, 1 > const& arrayToCompareTo,
                                       T const valueToCompareTo, 
                                       U const valueIfArrayValueLowerOrEqualToValueToCompareTo,
                                       U const valueIfArrayValueGreaterThanValueToCopareTo ){
      ndarray::Array< U, 1, 1 > arrOut = ndarray::allocate( arrayToCompareTo.getShape()[ 0 ] );
      auto itOut = arrOut.begin();
      for ( auto itComp = arrayToCompareTo.begin(); itComp != arrayToCompareTo.end(); ++itComp, ++itOut ){
        if ( *itComp <= valueToCompareTo )
          *itOut = valueIfArrayValueLowerOrEqualToValueToCompareTo;
        else
          *itOut = valueIfArrayValueGreaterThanValueToCopareTo;
      }
      return arrOut;
    }
    
    template< typename T, typename U >
    ndarray::Array< U, 1, 1 > whereLe( ndarray::Array< T, 1, 1 > const& arrayToCompareTo,
                                       T const valueToCompareTo, 
                                       U const valueIfArrayValueLowerOrEqualToValueToCompareTo,
                                       ndarray::Array< U, 1, 1 > const& valuesIfArrayValueGreaterThanValueToCopareTo ){
      ndarray::Array< U, 1, 1 > arrOut = ndarray::allocate( arrayToCompareTo.getShape()[ 0 ] );
      auto itOut = arrOut.begin();
      auto itElse = valuesIfArrayValueGreaterThanValueToCopareTo.begin();
      for ( auto itComp = arrayToCompareTo.begin(); itComp != arrayToCompareTo.end(); ++itComp, ++itOut, ++itElse ){
        if ( *itComp <= valueToCompareTo )
          *itOut = valueIfArrayValueLowerOrEqualToValueToCompareTo;
        else
          *itOut = *itElse;
      }
      return arrOut;
    }
    
    template< typename T, typename U >
    ndarray::Array< U, 1, 1 > whereGt( ndarray::Array< T, 1, 1 > const& arrayToCompareTo,
                                       T const valueToCompareTo, 
                                       U const valueIfArrayValueGreaterThanValueToCompareTo,
                                       U const valueIfArrayValueLessOrEqualToValueToCopareTo ){
      ndarray::Array< U, 1, 1 > arrOut = ndarray::allocate( arrayToCompareTo.getShape()[ 0 ] );
      auto itOut = arrOut.begin();
      for ( auto itComp = arrayToCompareTo.begin(); itComp != arrayToCompareTo.end(); ++itComp, ++itOut ){
        if ( *itComp > valueToCompareTo )
          *itOut = valueIfArrayValueGreaterThanValueToCompareTo;
        else
          *itOut = valueIfArrayValueLessOrEqualToValueToCopareTo;
      }
      return arrOut;
    }
    
    template< typename T, typename U >
    ndarray::Array< U, 1, 1 > whereGt( ndarray::Array< T, 1, 1 > const& arrayToCompareTo,
                                       T const valueToCompareTo, 
                                       U const valueIfArrayValueGreaterThanValueToCompareTo,
                                       ndarray::Array< U, 1, 1 > const& valuesIfArrayValueLessOrEqualToValueToCopareTo ){
      ndarray::Array< U, 1, 1 > arrOut = ndarray::allocate( arrayToCompareTo.getShape()[ 0 ] );
      auto itOut = arrOut.begin();
      auto itElse = valuesIfArrayValueLessOrEqualToValueToCopareTo.begin();
      for ( auto itComp = arrayToCompareTo.begin(); itComp != arrayToCompareTo.end(); ++itComp, ++itOut, ++itElse ){
        if ( *itComp > valueToCompareTo )
          *itOut = valueIfArrayValueGreaterThanValueToCompareTo;
        else
          *itOut = *itElse;
      }
      return arrOut;
    }
    
    template< typename T, typename U >
    ndarray::Array< U, 1, 1 > whereGe( ndarray::Array< T, 1, 1 > const& arrayToCompareTo,
                                       T const valueToCompareTo, 
                                       U const valueIfArrayValueGreaterOrEqualToValueToCompareTo,
                                       U const valueIfArrayValueLessThanValueToCopareTo ){
      ndarray::Array< U, 1, 1 > arrOut = ndarray::allocate( arrayToCompareTo.getShape()[ 0 ] );
      auto itOut = arrOut.begin();
      for ( auto itComp = arrayToCompareTo.begin(); itComp != arrayToCompareTo.end(); ++itComp, ++itOut ){
        if ( *itComp >= valueToCompareTo )
          *itOut = valueIfArrayValueGreaterOrEqualToValueToCompareTo;
        else
          *itOut = valueIfArrayValueLessThanValueToCopareTo;
      }
      return arrOut;
    }
    
    template< typename T, typename U >
    ndarray::Array< U, 1, 1 > whereGe( ndarray::Array< T, 1, 1 > const& arrayToCompareTo,
                                       T const valueToCompareTo, 
                                       U const valueIfArrayValueGreaterOrEqualToValueToCompareTo,
                                       ndarray::Array< U, 1, 1 > const& valuesIfArrayValueLessThanValueToCopareTo ){
      ndarray::Array< U, 1, 1 > arrOut = ndarray::allocate( arrayToCompareTo.getShape()[ 0 ] );
      auto itOut = arrOut.begin();
      auto itElse = valuesIfArrayValueLessThanValueToCopareTo.begin();
      for ( auto itComp = arrayToCompareTo.begin(); itComp != arrayToCompareTo.end(); ++itComp, ++itOut, ++itElse ){
        if ( *itComp >= valueToCompareTo )
          *itOut = valueIfArrayValueGreaterOrEqualToValueToCompareTo;
        else
          *itOut = *itElse;
      }
      return arrOut;
    }

    template< typename T >
    ndarray::Array< T, 1, 1 > splineI( ndarray::Array< T, 1, 1 > const& XVecArr, 
                                       ndarray::Array< T, 1, 1 > const& YVecArr, 
                                       T const YP1, 
                                       T const YPN){
      int m, o, N = XVecArr.getShape()[ 0 ];
      double p, qn, sig, un;
      ndarray::Array< T, 1, 1 > UVecArr = ndarray::allocate( N - 1 );
      ndarray::Array< T, 1, 1 > Y2VecArr = ndarray::allocate( N );

      if ( YP1 > 0.99e30 )  /// The lower boundary condition is set either to be "natural"
      {
        Y2VecArr[ 0 ] = UVecArr[ 0 ] = 0.0;
      }
      else                /// or else to have a specified first derivative
      {
        Y2VecArr[ 0 ] = -0.5;
        UVecArr[ 0 ]  = (3.0 / ( XVecArr[ 1 ] - XVecArr[ 0 ] ) ) * ( ( YVecArr[ 1 ] - YVecArr[ 0 ]) / ( XVecArr[ 1 ] - XVecArr[ 0 ] ) - YP1 );
      }

      /**
      This is the decomposition loop of the tridiagonal algorithm. Y2VecArr and UVecArr are used for temporary storage of the decomposed factors.
      **/
      for ( m = 1; m < N - 1; ++m )
      {
        sig = ( XVecArr[ m ] - XVecArr[ m - 1 ] ) / ( XVecArr[ m + 1 ] - XVecArr[ m - 1 ] );
        p = sig * Y2VecArr[ m - 1 ] + 2.0;
        Y2VecArr[ m ] = ( sig - 1.0 ) / p;
        UVecArr[ m ]  = ( YVecArr[ m + 1 ] - YVecArr[ m ] ) / (XVecArr[ m + 1 ] - XVecArr[ m ] ) - ( YVecArr[ m ] - YVecArr[ m - 1 ] ) / ( XVecArr[ m ] - XVecArr[ m - 1 ] );
        UVecArr[ m ]  = ( 6.0 * UVecArr[ m ] / ( XVecArr[ m + 1 ] - XVecArr[ m - 1 ] ) - sig * UVecArr[ m-1 ] ) / p;
      }
      if ( YPN > 0.99e30 )  /// The upper boundary condition is set either to be "natural"
        qn = un = 0.0;
      else                /// or else to have a specified first derivative
      {
        qn = 0.5;
        un = (3.0 / ( XVecArr[ N - 1 ] - XVecArr[ N - 2 ] ) ) * ( YPN - ( YVecArr[ N - 1 ] - YVecArr[ N - 2 ] ) / ( XVecArr[ N - 1 ] - XVecArr[ N - 2 ] ) );
      }
      Y2VecArr[ N - 1 ] = ( un - qn * UVecArr[ N - 2 ] ) / ( qn * Y2VecArr[ N - 2 ] + 1.0 );

      /// This is the backsubstitution loop of the tridiagonal algorithm
      for (o = N - 2; o >= 0; o--)
      {
        Y2VecArr[ o ] = Y2VecArr[ o ] * Y2VecArr[ o + 1 ] + UVecArr[ o ];
      }
      return Y2VecArr;
    }

    template< typename T >
    ndarray::Array< T, 1, 1 > splineI( ndarray::Array< T, 1, 1 > const& XVecArr, 
                                       ndarray::Array< T, 1, 1 > const& YVecArr){
      return splineI( XVecArr, 
                      YVecArr, 
                      T( 1.0e30 ), 
                      T( 1.0e30) );
    }

    /**
      SplInt
      Given the Arrays XAVecArr(0:N-1) and YAVecArr(0:N-1), which tabulate a function (whith the XAVecArr(i)'s in order), and given the array Y2AVecArr(0:N-1), which is the output from Spline above, and given a value of X, this routine returns a cubic-spline interpolated value Y;
     **/
    template< typename T >
    bool splInt( ndarray::Array< T, 1, 1 > const& XAVecArr, 
                 ndarray::Array< T, 1, 1 > const& YAVecArr, 
                 ndarray::Array< T, 1, 1> const& Y2AVecArr, 
                 T X,
                 T & Y){
      int klo, khi, o, N;
      double h, b, a;

      N = XAVecArr.getShape()[ 0 ];
      /**
      We will find the right place in the table by means of bisection. This is optimal if sequential calls to this routine are at random values of X. If sequential calls are in order, and closely spaced, one would do better to store previous values of klo and khi and test if they remain appropriate on the next call.
      **/
      klo = 1;
      khi = N;
      while (khi - klo > 1)
      {
        o = (khi + klo) >> 1;
        if ( XAVecArr[ o ] > X )
          khi = o;
        else
          klo = o;
      } /// klo and khi now bracket the input value of X
      h = XAVecArr[ khi ] - XAVecArr[ klo ];
      if ( h == 0.0 ){  /// The XAVecArr(i)'s must be distinct
        cout << "CFits::SplInt: ERROR: Bad XAVecArr input to routine SplInt" << endl;
        return false;
      }
      a = ( XAVecArr[ khi ] - X ) / h;
      b = (X - XAVecArr[ klo ] ) / h; /// Cubic Spline polynomial is now evaluated.
      Y = a * YAVecArr[ klo ] + b * YAVecArr[ khi ] + ((a * a * a - a) * Y2AVecArr[ khi ] ) * (h * h) / 6.0;
      return true;
    }

    /**
      InterPol linear, not regular
     **/
    template< typename T >
    bool interPol( ndarray::Array< T, 1, 1 > const& VVecArr,
                   ndarray::Array< T, 1, 1 > const& XVecArr,
                   ndarray::Array< T, 1, 1 > const& UVecArr,
                   ndarray::Array< T, 1, 1> & D_A1_Out){
      return interPol( VVecArr, 
                       XVecArr, 
                       UVecArr, 
                       D_A1_Out, 
                       false);
    }

    template< typename T >
    bool interPol( ndarray::Array< T, 1, 1 > const& VVecArr,
                   ndarray::Array< T, 1, 1 > const& XVecArr,
                   ndarray::Array< T, 1, 1 > const& UVecArr,
                   ndarray::Array< T, 1, 1 > & D_A1_Out,
                   bool B_PreserveFlux){
      std::vector< std::string > cs_a1(1);
      cs_a1[ 0 ] = std::string(" ");
      D_A1_Out = ndarray::allocate( UVecArr.getShape()[ 0 ] );
      if ( B_PreserveFlux ){
        ndarray::Array< T, 1, 1 > D_A1_U = ndarray::allocate( 2 );
        ndarray::Array< T, 1, 1 > D_A1_X = ndarray::allocate( XVecArr.getShape()[ 0 ] + 1 );
        D_A1_X[ 0 ] = XVecArr[ 0 ] - ( ( XVecArr[ 1 ] - XVecArr[ 0 ] ) / 2. );
        D_A1_X[ D_A1_X.getShape()[ 0 ] - 1 ] = XVecArr[ XVecArr.getShape()[ 0 ] - 1 ] + ( ( XVecArr[ XVecArr.getShape()[ 0 ] - 1 ] - XVecArr[ XVecArr.getShape()[ 0 ] - 2] ) / 2. );
        for (int i_pix = 1; i_pix < XVecArr.getShape()[ 0 ]; ++i_pix ){
          D_A1_X[ i_pix ] = XVecArr[ i_pix - 1 ] + ( ( XVecArr[ i_pix ] - XVecArr[ i_pix - 1 ] ) / 2. );
        }
        #ifdef __DEBUG_FITS_INTERPOL__
          cout << "CFits::InterPol: XVecArr = " << XVecArr << endl;
          cout << "CFits::InterPol: D_A1_X = " << D_A1_X << endl;
        #endif

        ndarray::Array< int, 1, 1 > I_A1_Ind = ndarray::allocate( D_A1_X.getShape()[ 0 ] );
        ndarray::Array< size_t, 1, 1 > P_I_A1_Ind;
        int I_Start = 0;
        int I_End = 0;
        int I_NInd = 0;
        double D_Start, D_End;
        for (int i_pix = 0; i_pix < UVecArr.getShape()[ 0 ]; ++i_pix ){
          if ( i_pix == 0 ){
            D_A1_U[ 0 ] = UVecArr[ 0 ] - ( ( UVecArr[ 1 ] - UVecArr[ 0 ] ) / 2. );
            D_A1_U[ 1 ] = UVecArr[ 0 ] + ( ( UVecArr[ 1 ] - UVecArr[ 0 ] ) / 2. );
    //        if (!this->IntegralUnderCurve(XVecArr, VVecArr, D_A1_X, (*P_A1_Out)(i_pix))){
    //          cout << "CFits::InterPol: ERROR: IntegralUnderCurve(XVecArr = " << XVecArr << ", VVecArr = " << VVecArr << ", D_A1_X = " << D_A1_X << ") returned FALSE" << endl;
    //          return false;
    //        }
          }
          else if ( i_pix == UVecArr.getShape()[ 0 ] - 1 ){
            D_A1_U[ 0 ] = UVecArr[ UVecArr.getShape()[ 0 ] - 1 ] - ( ( UVecArr[ UVecArr.getShape()[ 0 ] - 1 ] - UVecArr[ UVecArr.getShape()[ 0 ] - 2 ] ) / 2. );
            D_A1_U[ 1 ] = UVecArr[ UVecArr.getShape()[ 0 ] - 1 ] + ( ( UVecArr[ UVecArr.getShape()[ 0 ] - 1 ] - UVecArr[ UVecArr.getShape()[ 0 ] - 2 ] ) / 2. );
    //        if (!this->IntegralUnderCurve(XVecArr, VVecArr, D_A1_X, (*P_A1_Out)(i_pix))){
    //          cout << "CFits::InterPol: ERROR: IntegralUnderCurve(XVecArr = " << XVecArr << ", VVecArr = " << VVecArr << ", D_A1_X = " << D_A1_X << ") returned FALSE" << endl;
    //          return false;
    //        }

          }
          else{
            D_A1_U[ 0 ] = UVecArr[ i_pix ] - ( ( UVecArr[ i_pix ] - UVecArr[ i_pix - 1 ] ) / 2. );
            D_A1_U[ 1 ] = UVecArr[ i_pix ] + ( ( UVecArr[ i_pix + 1 ] - UVecArr[ i_pix ] ) / 2. );
    //        if (!this->IntegralUnderCurve(XVecArr, VVecArr, D_A1_X, (*P_A1_Out)(i_pix))){
    //          cout << "CFits::InterPol: ERROR: IntegralUnderCurve(XVecArr = " << XVecArr << ", VVecArr = " << VVecArr << ", D_A1_X = " << D_A1_X << ") returned FALSE" << endl;
    //          return false;
    //        }
          }
          I_A1_Ind = whereLt( D_A1_X, D_A1_U[ 0 ], 1, 0 );
          P_I_A1_Ind = getIndices( I_A1_Ind );
          int I_NInd = P_I_A1_Ind.getShape()[ 0 ];
          if ( I_NInd < 1 ){
            #ifdef __DEBUG_FITS_INTERPOL__
              cout << "CFits::InterPol: WARNING: 1. I_A1_Ind = " << I_A1_Ind << ": I_NInd < 1" << endl;
            #endif
            I_Start = 0;
          }
          else{
            I_Start = P_I_A1_Ind[ I_NInd - 1 ];
          }
          #ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::InterPol: i_pix = " << i_pix << ": D_A1_U = " << D_A1_U << endl;
          #endif
          I_A1_Ind = whereGt( D_A1_X, D_A1_U(1), 1, 0);
          P_I_A1_Ind = getIndices( I_A1_Ind );
          I_NInd = P_I_A1_Ind.getShape()[ 0 ];
          if ( I_NInd < 1 ){
            #ifdef __DEBUG_FITS_INTERPOL__
              cout << "CFits::InterPol: WARNING: 2. I_A1_Ind = " << I_A1_Ind << ": I_NInd < 1" << endl;
            #endif
            I_End = D_A1_X.getShape()[ 0 ] - 1;
          }
          else{
            I_End = P_I_A1_Ind[ 0 ];
          }
          #ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::InterPol: i_pix = " << i_pix << ": D_A1_X(" << I_Start << ":" << I_End << ") = " << D_A1_X[ndarray::view(I_Start, I_End)] << endl;
          #endif

          D_Start = D_A1_U[ 0 ];
          if ( D_A1_X[ I_Start ] > D_A1_U[ 0 ] )
            D_Start = D_A1_X[ I_Start ];
          D_A1_Out[ i_pix ] = 0.;
          if ( ( D_A1_U[ 1 ] > D_A1_X[ 0 ] ) && ( D_A1_U[ 0 ] < D_A1_X[ D_A1_X.getShape()[ 0 ] - 1 ]) ){
            do {
              if ( D_A1_U[ 1 ] < D_A1_X[ I_Start + 1 ] ){
                D_End = D_A1_U[ 1 ];
              }
              else{
                D_End = D_A1_X[ I_Start + 1 ];
              }
              #ifdef __DEBUG_FITS_INTERPOL__
                cout << "CFits::InterPol: i_pix = " << i_pix << ": I_Start = " << I_Start << ", I_End = " << I_End << endl;
                cout << "CFits::InterPol: i_pix = " << i_pix << ": D_Start = " << D_Start << ", D_End = " << D_End << endl;
              #endif
              D_A1_Out[ i_pix ] += VVecArr[ I_Start ] * ( D_End - D_Start ) / ( D_A1_X[ I_Start + 1 ] - D_A1_X[ I_Start ] );
              D_Start = D_End;
              if ( D_A1_U[ 1 ] >= D_A1_X[ I_Start + 1 ] )
                I_Start++;
              #ifdef __DEBUG_FITS_INTERPOL__
                cout << "CFits::InterPol: i_pix = " << i_pix << ": D_A1_Out(" << i_pix << ") = " << D_A1_Out[i_pix] << endl;
              #endif
              if ( I_Start + 1 >= D_A1_X.getShape()[ 0 ] )
                break;
            } while ( D_End < D_A1_U[ 1 ]-( ( D_A1_U[ 1 ] - D_A1_U[ 0 ] ) / 100000000.) );
          }
    //      for (int i_p=I_Start; i_p<I_End; i_p++){
    //      }
    //      return false;
        }
        return true;
      }

      if ( !interPol( VVecArr, 
                      XVecArr, 
                      UVecArr, 
                      cs_a1, 
                      D_A1_Out ) ){
        cout << "CFits::InterPol: ERROR: InterPol returned FALSE" << endl;
        return false;
      }
      #ifdef __DEBUG_FITS_INTERPOL__
        cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, D_A1_Out): Ready " << endl;
      #endif

      return true;
    }

    template< typename T >
    bool interPol( ndarray::Array< T, 1, 1 > const& VVecArr,
                   ndarray::Array< T, 1, 1 > const& XVecArr,
                   ndarray::Array< T, 1, 1 > const& UVecArr,
                   std::vector< std::string > const& CS_A1_In,
                   ndarray::Array< T, 1, 1 > & D_A1_Out ){
      #ifdef __DEBUG_FITS_INTERPOL__
        cout << "CFits::InterPol: VVecArr.size() = " << VVecArr.getShape()[ 0 ] << endl;
        cout << "CFits::InterPol: XVecArr.size() = " << XVecArr.getShape()[ 0 ] << endl;
        cout << "CFits::InterPol: UVecArr.size() = " << UVecArr.getShape()[ 0 ] << endl;
        cout << "CFits::InterPol: CS_A1_In.size() = " << CS_A1_In.size() << endl;
        cout << "CFits::InterPol(D_A1_V = " << VVecArr << ", D_A1_X = " << XVecArr << ", D_A1_U = " << UVecArr << ", CS_A1_In) Started" << endl;
      #endif

      int M = VVecArr.getShape()[ 0 ];
      #ifdef __DEBUG_FITS_INTERPOL__
        cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, CS_A1_In): M set to " << M << endl;
      #endif
//      firstIndex i;

      if ( XVecArr.getShape()[ 0 ] != M ){
        cout << "CFits::InterPol: ERROR: XVecArr and VVecArr must have same # of elements!" << endl;
        return false;
      }
      ndarray::Array< int, 1, 1 > SVecArr = valueLocate( XVecArr, UVecArr );
      #ifdef __DEBUG_FITS_INTERPOL__
        cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, CS_A1_In): SVecArr set to " << SVecArr << endl;
      #endif
      SVecArr.deep() = whereLt( SVecArr, 0, 0, SVecArr );
      #ifdef __DEBUG_FITS_INTERPOL__
        cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, CS_A1_In): SVecArr set to " << SVecArr << endl;
      #endif

      SVecArr.deep() = whereGt( SVecArr, M-2, M-2, SVecArr );
      #ifdef __DEBUG_FITS_INTERPOL__
        cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, CS_A1_In): SVecArr set to " << SVecArr << endl;
      #endif

      #ifdef __DEBUG_FITS_INTERPOL__
        cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, CS_A1_In): Starting HInterPol " << endl;
      #endif
      if ( hInterPol( VVecArr, 
                      XVecArr, 
                      SVecArr, 
                      UVecArr, 
                      CS_A1_In, 
                      D_A1_Out ) ){
        cout << "CFits::InterPol: ERROR: HInterPol returned FALSE" << endl;
        return false;
      }

      #ifdef __DEBUG_FITS_INTERPOL__
        cout << "CFits::InterPol(D_A1_V, D_A1_X, D_A1_U, CS_A1_In): Ready " << endl;
      #endif

      return true;
    }
    
    template< typename T >
    ndarray::Array< int, 1, 1 > valueLocate( ndarray::Array< T, 1, 1 > const& VecArr, 
                                             ndarray::Array< T, 1, 1 > const& ValVecArr){
      #ifdef __DEBUG_FITS_INTERPOL__
        cout << "CFits::ValueLocate: VecArr = " << VecArr << endl;
        cout << "CFits::ValueLocate: ValVecArr = " << ValVecArr << endl;
      #endif
      if ( VecArr.getShape()[ 0 ] < 1 ){
        cout << "CFits::ValueLocate: ERROR: VecArr.size() < 1 => Returning FALSE" << endl;
        exit(EXIT_FAILURE);
      }
      if ( ValVecArr.getShape()[ 0 ] < 1 ){
        cout << "CFits::ValueLocate: ERROR: ValVecArr.size() < 1 => Returning FALSE" << endl;
        exit(EXIT_FAILURE);
      }
      ndarray::Array< int, 1, 1 > IntVecArr = ndarray::allocate( ValVecArr.getShape()[ 0 ] );

      int n;
      int N = VecArr.getShape()[ 0 ];
      int M = ValVecArr.getShape()[ 0 ];

      bool Increasing = false;
      int ii=0;
      while( VecArr[ ii ] == VecArr[ ii + 1 ] ){
        ii++;
      }
      if ( VecArr[ ii + 1 ] > VecArr[ ii ] )
        Increasing = true;

      #ifdef __DEBUG_FITS_INTERPOL__
        if (Increasing)
          cout << "CFits::ValueLocate: Increasing = TRUE" << endl;
        else
          cout << "CFits::ValueLocate: Increasing = FALSE" << endl;
      #endif

      /// For every element in ValVecArr
      for ( int m = 0; m < M; m++ ){
        #ifdef __DEBUG_FITS_INTERPOL__
          cout << "CFits::ValueLocate: ValVecArr(m) = " << ValVecArr[m] << endl;
        #endif
        if ( Increasing ){
          if ( ValVecArr[ m ] < VecArr[ 0 ] ){
            IntVecArr[ m ] = 0 - 1;
          }
          else if ( VecArr[ N - 1 ] <= ValVecArr[ m ] ){
            IntVecArr[ m ] = N - 1;
          }
          else{
            n = -1;
            while (n < N-1){
              n++;
              if ( ( VecArr[ n ] <= ValVecArr[ m ] ) && ( ValVecArr[ m ] < VecArr[ n + 1 ] ) ){
                IntVecArr[ m ] = n;
                break;
              }
            }
          }
          #ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::ValueLocate: Increasing = TRUE: IntVecArr(m) = " << IntVecArr[m] << endl;
          #endif
        }
        else{/// if (Decreasing)
          if ( VecArr[ 0 ] <= ValVecArr[ m ] )
            IntVecArr[ m ] = 0 - 1;
          else if ( ValVecArr[ m ] < VecArr[ N - 1 ] )
            IntVecArr[ m ] = N - 1;
          else{
            n = -1;
            while ( n < N - 1 ){
              n++;
              if ( ( VecArr[ n + 1 ] <= ValVecArr[ m ] ) && ( ValVecArr[ m ] < VecArr[ n ] ) ){
                IntVecArr[ m ] = n;
                break;
              }
            }
          }
          #ifdef __DEBUG_FITS_INTERPOL__
            cout << "CFits::ValueLocate: Increasing = FALSE: IntVecArr(m) = " << IntVecArr[m] << endl;
          #endif
        }
      }
      #ifdef __DEBUG_FITS_INTERPOL__
        cout << "CFits::ValueLocate: IntVecArr = " << IntVecArr << endl;
      #endif
      return IntVecArr;
    }
    
    template< typename T >
    StretchAndCrossCorrelateResult< T > stretchAndCrossCorrelate( ndarray::Array< T, 1, 1 > const& spec,
                                                                  ndarray::Array< T, 1, 1 > const& specRef,
                                                                  int const radiusXCor,
                                                                  int const stretchMinLength,
                                                                  int const stretchMaxLength,
                                                                  int const nStretches ){
      /// Stretch Reference Spectrum
      ndarray::Array< T, 1, 1 > refX = indGenNdArr( T(specRef.getShape()[ 0 ] ) );
      #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATE__
        cout << "CFits::StretchAndCrossCorrelate: refX = " << refX << endl;
        cout << "CFits::StretchAndCrossCorrelate: specRef = " << specRef << endl;
      #endif

      ndarray::Array< T, 1, 1 > specTemp;
      ndarray::Array< T, 1, 1 > specRefTemp;
      ndarray::Array< T, 2, 1 > specCalib_Out;
      double rmsOut = 0.;
    //  int I_PixShift = 0;
      int linePos = 0;
      int lineTemp = 0;
      int nLines = 0;
      ndarray::Array< T, 1, 1 > refY;
      ndarray::Array< T, 1, 1 > refXStretched;
      ndarray::Array< double, 1, 1 > xCorChiSq = ndarray::allocate( nStretches );
      ndarray::Array< int, 1, 1 > refStretchLength = ndarray::allocate( nStretches );
      ndarray::Array< double, 1, 1 > pixShift = ndarray::allocate( nStretches );
      refStretchLength.deep() = 0;
      xCorChiSq.deep() = 0.;
      refStretchLength[ 0 ] = stretchMinLength;
      double D_Temp = 0.;
      for (int i_stretch = 0; i_stretch < nStretches; i_stretch++){
        refXStretched = ndarray::allocate( refStretchLength[ i_stretch ] );
        refXStretched[ 0 ] = refX[ 0 ];
        for (int i_x_stretch=1; i_x_stretch < refStretchLength[ i_stretch ]; i_x_stretch++)
          refXStretched[ i_x_stretch ] = refXStretched[ i_x_stretch - 1 ] + ( ( refX[ refX.getShape()[ 0 ] - 1 ] - refX[ 0 ] ) / refStretchLength[ i_stretch ] );
        #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATE__
          cout << "CFits::StretchAndCrossCorrelate: i_stretch = " << i_stretch << ": refStretchLength = " << refStretchLength[i_stretch] << ": refXStretched = " << refXStretched << endl;
        #endif
        refY = ndarray::allocate( refXStretched.getShape()[ 0 ] );
        if ( interPol( specRef,
                       refX,
                       refXStretched,
                       refY ) ){
          cout << "CFits::StretchAndCrossCorrelate: i_stretch = " << i_stretch << ": refStretchLength = " << refStretchLength[ i_stretch ] << ": ERROR: InterPol(specRef=" << specRef << ", refX=" << refX << ", refXStretched=" << refXStretched << ", refY) returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }

        /// Cross-correlate D_A1_Spec to reference spectrum
        specTemp = ndarray::allocate( spec.getShape()[ 0 ] );
        specTemp.deep() = spec;
        specRefTemp = ndarray::allocate( refY.getShape()[ 0 ] );
        specRefTemp.deep() = refY;
        if ( specTemp.getShape()[ 0 ] < specRefTemp.getShape()[ 0 ] ){
          ndarray::Array< T, 1, 1 > tempArr = ndarray::allocate( specRefTemp.getShape()[ 0 ] );
          tempArr.deep() = specRefTemp;
          specRefTemp = ndarray::allocate( specTemp.getShape()[ 0 ] );
          specRefTemp.deep() = 0.;
          specRefTemp[ ndarray::view( 0, tempArr.getShape()[ 0 ] ) ] = tempArr[ ndarray::view() ];
        }
        if ( specRefTemp.getShape()[ 0 ] < specTemp.getShape()[ 0 ] ){
          ndarray::Array< T, 1, 1 > tempArr = ndarray::allocate( specTemp.getShape()[ 0 ] );
          tempArr.deep() = specTemp;
          specTemp = ndarray::allocate( specRefTemp.getShape()[ 0 ] );
          specTemp.deep() = 0.;
          specTemp[ ndarray::view( 0, tempArr.getShape()[ 0 ] ) ] = tempArr[ ndarray::view() ];
        }

        if ( !crossCorrelate( specRefTemp,
                              specTemp,
                              radiusXCor,
                              radiusXCor,
                              pixShift[ i_stretch ],
                              xCorChiSq[ i_stretch ] ) ){
          cout << "CFits::StretchAndCrossCorrelate: i_stretch = " << i_stretch << ": refStretchLength = " << refStretchLength[i_stretch] << ": ERROR: CrossCorrelate returned FALSE" << endl;
          exit(EXIT_FAILURE);
        }
        #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATE__
          cout << "CFits::StretchAndCrossCorrelate: i_stretch = " << i_stretch << ": refStretchLength = " << refStretchLength[i_stretch] << ": pixShift[i_stretch] = " << pixShift[i_stretch] << endl;
          cout << "CFits::StretchAndCrossCorrelate: i_stretch = " << i_stretch << ": refStretchLength = " << refStretchLength[i_stretch] << ": xCorChiSq[i_stretch] = " << xCorChiSq[i_stretch] << endl;
        #endif
        if (i_stretch < nStretches - 1){
          D_Temp = double( stretchMaxLength - stretchMinLength ) / double( nStretches );
          refStretchLength[ i_stretch + 1 ] = refStretchLength[ i_stretch ] + int(D_Temp);
          #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATE__
            cout << "CFits::StretchAndCrossCorrelate: i_stretch = " << i_stretch << ": stretchMaxLength(=" << stretchMaxLength << ") - stretchMinLength(=" << stretchMinLength << ") / nStretches(=" << nStretches << ") = " << D_Temp << ", int(D_Temp) = " << int(D_Temp) << endl;
            cout << "CFits::StretchAndCrossCorrelate: i_stretch = " << i_stretch << ": refStretchLength(i+1) = " << refStretchLength[i_stretch+1] << endl;
          #endif
        }
      }/// end for (int i_stretch=0; i_stretch<I_N_Stretches_In; i_stretch++){

      #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATE__
        cout << "CFits::StretchAndCrossCorrelate: refStretchLength = " << refStretchLength << endl;
        cout << "CFits::StretchAndCrossCorrelate: xCorChiSq = " << xCorChiSq << endl;
        cout << "CFits::StretchAndCrossCorrelate: pixShift = " << pixShift << endl;
      #endif

//      double minXCorChiSq = max(D_A1_XCorChiSq);
      double minXCorChiSq = min( xCorChiSq );
      int minXCorChiSqPos = 0;
      for ( int i_stretch = 0; i_stretch < nStretches; i_stretch++ ){
        if ( xCorChiSq[ i_stretch ] < minXCorChiSq ){
          minXCorChiSq = xCorChiSq[ i_stretch ];
          minXCorChiSqPos = i_stretch;
        }
      }
      #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATE__
        cout << "CFits::StretchAndCrossCorrelate: I_MinXCorChiSqPos = " << I_MinXCorChiSqPos << endl;
      #endif
      double shift_Out = pixShift[ minXCorChiSqPos ];
      double stretch_Out = double( refStretchLength[ minXCorChiSqPos ] );

      refXStretched = ndarray::allocate( refStretchLength[ minXCorChiSqPos ] );
      refXStretched[ 0 ] = refX[ 0 ];
      for (int i_x_stretch = 1; i_x_stretch < refStretchLength[ minXCorChiSqPos ]; i_x_stretch++)
        refXStretched[ i_x_stretch ] = refXStretched[ i_x_stretch - 1 ] + ( ( refX[ refX.getShape()[ 0 ] - 1 ] - refX[ 0 ] ) / stretch_Out );
      #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATE__
        cout << "CFits::StretchAndCrossCorrelate: minXCorChiSqPos = " << minXCorChiSqPos << ": refStretchLength = " << refStretchLength[minXCorChiSqPos] << ": refXStretched = " << refXStretched << endl;
      #endif
      if ( interPol( specRef,
                     refX,
                     refXStretched,
                     refY ) ){
        cout << "CFits::StretchAndCrossCorrelate: I_MinXCorChiSqPos = " << minXCorChiSqPos << ": refStretchLength = " << refStretchLength[ minXCorChiSqPos ] << ": ERROR: InterPol(specRef=" << specRef << ", refX=" << refX << ", refXStretched=" << refXStretched << ", refY) returned FALSE" << endl;
        exit(EXIT_FAILURE);
      }
      ndarray::Array< T, 1, 1 > refXTemp = indGenNdArr( T( refY.getShape()[ 0 ] ) );
      refXStretched = ndarray::allocate( refY.getShape()[ 0 ] );
      refXStretched.deep() = refXTemp - shift_Out;

      StretchAndCrossCorrelateResult< T > result;
      result.stretch = stretch_Out;
      result.shift = shift_Out;
      result.specStretchedMinChiSq = ndarray::allocate( refY.getShape()[ 0 ], 2 );
      result.specStretchedMinChiSq[ ndarray::view()( 0 ) ] = refXStretched;//[ ndarray::view() ];
      result.specStretchedMinChiSq[ ndarray::view()( 1 ) ] = refY;

      return result;
    }
/*
    template< typename T >
    bool CFits::StretchAndCrossCorrelateSpec( ndarray::Array< T, 1, 1 > const& spec,
                                              ndarray::Array< T, 1, 1 > const& specRef,
                                              ndarray::Array< T, 2, 1 > const& lineList_WLenPix,
                                              int const radiusXCor,
                                              int const stretchMinLength,
                                              int const stretchMaxLength,
                                              int const nStretches,
                                              int const lengthPieces,
                                              int const nCalcs,
                                              int const polyFitOrder_Stretch,
                                              int const polyFitOrder_Shift,
                                              std::string const& fName,
                                              ndarray::Array< T, 2, 1 > & lineList_WLenPix_Out){
      if ( spec.getShape()[ 0 ] != specRef.getShape()[ 0 ] ){
        cout << "CFits::StretchAndCrossCorrelate: ERROR: spec.getShape()[0](=" << spec.getShape()[ 0 ] << " != specRef.getShape()[0](=" << specRef.getShape() << ") => Returning FALSE" << endl;
        exit( EXIT_FAILURE );
      }
      ndarray::Array< double, 1, 1 > chiSqMin_Stretch = ndarray::allocate( nCalcs );
      chiSqMin_Stretch.deep() = 0.;
      ndarray::Array< double, 1, 1 > chiSqMin_Shift = ndarray::allocate( nCalcs );
      chiSqMin_Shift.deep() = 0.;
      ndarray::Array< double, 1, 1 > xCenter = ndarray::allocate( nCalcs );
      xCenter.deep() = 0.;
      ndarray::Array< double, 1, 1 > specPiece = ndarray::allocate( lengthPieces );
      ndarray::Array< double, 1, 1 > specRefPiece = ndarray::allocate( lengthPieces );
      int start = 0;
      int end = 0;
      ndarray::Array< double, 2, 1 > specPieceStretched_MinChiSq;
      ndarray::Array< double, 2, 1 > lineList_Pixels_AllPieces = ndarray::allocate( lineList_WLenPix.getShape()[ 0 ], nCalcs );
      lineList_Pixels_AllPieces.deep() = 0.;
      ndarray::Array< double, 1, 1 > x = indGenNdArr( double( specRef.getShape()[ 0 ] ) );
      ndarray::Array< double, 1, 1 > xPiece;
      ndarray::Array< double, 1, 1 > xPieceStretched;

      for ( int i_run = 0; i_run < nCalcs; i_run++ ){
        end = start + lengthPieces;
        if ( end >= spec.getShape()[ 0 ] )
          end = spec.getShape()[ 0 ] - 1;
        #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
          cout << "CFits::StretchAndCrossCorrelateSpec: i_run = " << i_run << ": start = " << start << ", end = " << end << endl;
        #endif
        if ( end <= start ){
          cout << "CFits::StretchAndCrossCorrelateSpec: i_run = " << i_run << ": ERROR: end <= start" << endl;
          exit( EXIT_FAILURE );
        }
        xCenter[ i_run ] = double( start ) + ( double( end - start ) / 2. );

        specPiece = ndarray::allocate( end - start + 1);
        specPiece.deep() = spec[ndarray::view( start, end + 1 ) ];
        #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
          cout << "CFits::StretchAndCrossCorrelateSpec: i_run = " << i_run << ": specPiece = " << specPiece << endl;
        #endif
        specRefPiece = ndarray::allocate( end - start + 1 );
        specRefPiece.deep() = specRef[ ndarray::view( start, end + 1 ) ];
        #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
          cout << "CFits::StretchAndCrossCorrelateSpec: i_run = " << i_run << ": specRefPiece = " << specRefPiece << endl;
        #endif
        /// stretch and crosscorrelate pieces
        StretchAndCrossCorrelateResult< double > stretchAndCrossCorrelateResult = stretchAndCrossCorrelate( specPiece,
                                                                                                            specRefPiece,
                                                                                                            radiusXCor,
                                                                                                            stretchMinLength,
                                                                                                            stretchMaxLengthIn,
                                                                                                            nStretches );
        chiSqMin_Stretch[ i_run ] = stretchAndCrossCorrelateResult.stretch;
        chiSqMin_Shift[ i_run ] = stretchAndCrossCorrelateResult.shift;
        specPieceStretched_MinChiSq.deep() = stretchAndCrossCorrelateResult.specStretchedMinChiSq;
        #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
          cout << "CFits::StretchAndCrossCorrelateSpec: i_run=" << i_run << ": chiSqMin_Stretch[i_run] = " << chiSqMin_Stretch[i_run] << endl;
          cout << "CFits::StretchAndCrossCorrelateSpec: i_run=" << i_run << ": chiSqMin_Shift[i_run] = " << chiSqMin_Shift[i_run] << endl;
        #endif

        xPiece = ndarray::allocate( end - start + 1 );
        xPiece.deep() = x[ndarray::view( start, end + 1 ) ];

        xPieceStretched = ndarray::allocate( chiSqMin_Stretch[ i_run ] );
        xPieceStretched[ 0 ] = start;
        for ( int i_pix=1; i_pix < xPieceStretched.getShape()[ 0 ]; i_pix++ ){
          xPieceStretched[ i_pix ] = xPieceStretched[ i_pix - 1 ] + (xPiece.getShape()[ 0 ] / chiSqMin_Stretch[ i_run ] );
        }
        #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
          cout << "CFits::StretchAndCrossCorrelateSpec: i_run=" << i_run << ": xPieceStretched = " << xPieceStretched << endl;
        #endif

        double weightLeft = 0.;
        double weightRight = 0.;
        ndarray::Array< int, 1, 1 > valueLocate = valueLocate( xPieceStretched, 
                                                               lineList_WLenPix[ ndarray::view()( 1 ) ] );
        for ( int i_line = 0; i_line < lineList_Pixels_AllPieces.getShape()[ 0 ]; i_line++ ){//i_line < lineList_Pixels_AllPieces.rows()
          if ( ( valueLocate[ i_line ] >= 0 ) && ( valueLocate[ i_line ] < xPieceStretched.getShape()[ 0 ] - 1 ) ){
            weightRight = ( xPieceStretched[ valueLocate[ i_line ] + 1 ] - xPieceStretched[ valueLocate[ i_line ] ] ) * ( lineList_WLenPix[ i_line ][ 1 ] - xPieceStretched[ valueLocate[ i_line ] ] );
            #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
              cout << "CFits::StretchAndCrossCorrelateSpec: i_line = " << i_line << ": xPieceStretched[ valueLocate[ i_line ]=" << valueLocate[i_line] << ") = " << xPieceStretched[valueLocate[i_line]] << ", xPieceStretched[valueLocate[i_line]+1=" << valueLocate[i_line]+1 << ") = " << xPieceStretched[valueLocate[i_line]+1] << ", weightRight = " << weightRight << endl;
            #endif
            weightLeft = 1. - weightRight;

            #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
              cout << "CFits::StretchAndCrossCorrelateSpec: i_line = " << i_line << ": weightLeft = " << weightLeft << endl;
            #endif
            lineList_Pixels_AllPieces[i_line][i_run] = start + ( valueLocate[ i_line ] * weightLeft ) + ( ( valueLocate[ i_line ] + 1 ) * weightRight ) - chiSqMin_Shift[ i_run ];
            #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
              cout << "CFits::StretchAndCrossCorrelateSpec: i_line = " << i_line << ": lineList_Pixels_AllPieces[i_line][i_run] = " << lineList_Pixels_AllPieces[i_line][i_run] << endl;
            #endif
          }
        }

        // for next run
        start += ( spec.getShape()[ 0 ] - lengthPieces ) / ( nCalcs - 1 );
      }/// end for (int i_run = 0; i_run < I_NStretches_In; i_run++){

      #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
        cout << "CFits::StretchAndCrossCorrelateSpec: chiSqMin_Shift = " << chiSqMin_Shift << endl;
        cout << "CFits::StretchAndCrossCorrelateSpec: chiSqMin_Stretch = " << chiSqMin_Stretch << endl;
        cout << "CFits::StretchAndCrossCorrelateSpec: lineList_Pixels_AllPieces = " << lineList_Pixels_AllPieces << endl;
      #endif

      Array<int, 1> I_A1_Where(D_A2_LineList_Pixels_AllPieces.cols());
      Array<int, 1> *P_I_A1_IndWhere;
      int I_NInd = 0;
      D_A2_LineList_WLenPix_Out.resize(D_A2_LineList_WLenPix_In.rows(), D_A2_LineList_WLenPix_In.cols());
      D_A2_LineList_WLenPix_Out = 0.;
      D_A2_LineList_WLenPix_Out(Range::all(), 0) = D_A2_LineList_WLenPix_In(Range::all(), 0);
      for (int i_line=0; i_line<D_A2_LineList_WLenPix_In.rows(); i_line++){
        I_A1_Where = where(fabs(D_A2_LineList_Pixels_AllPieces(i_line, Range::all())) > 0.001, 1, 0);
        P_I_A1_IndWhere = this->GetIndex(I_A1_Where, I_NInd);
        if (I_NInd == 1)
          D_A2_LineList_WLenPix_Out(i_line, 1) = D_A2_LineList_Pixels_AllPieces(i_line, (*P_I_A1_IndWhere)(0));
        else{
          for (int i_ind=0; i_ind<I_NInd; i_ind++){
            D_A2_LineList_WLenPix_Out(i_line, 1) += D_A2_LineList_Pixels_AllPieces(i_line, (*P_I_A1_IndWhere)(i_ind));
          }
          D_A2_LineList_WLenPix_Out(i_line, 1) = D_A2_LineList_WLenPix_Out(i_line, 1) / I_NInd;
        }
        if (D_A2_LineList_WLenPix_In.cols() == 3){
          D_A2_LineList_WLenPix_Out(i_line, 2) = D_A2_LineList_WLenPix_In(i_line, 2);
        }
        delete(P_I_A1_IndWhere);
      }
      cout << "CFits::StretchAndCrossCorrelateSpec: D_A2_LineList_WLenPix_Out = " << D_A2_LineList_WLenPix_Out << endl;

      /// Check positions
      Array<double, 2> D_A2_Dist(D_A2_LineList_Pixels_AllPieces.rows(), D_A2_LineList_Pixels_AllPieces.cols());
      D_A2_Dist = 0.;
      for (int i_row=0; i_row<D_A2_LineList_Pixels_AllPieces.rows(); i_row++){
        for (int i_col = 0; i_col < D_A2_LineList_Pixels_AllPieces.cols(); i_col++){
          if (fabs(D_A2_LineList_Pixels_AllPieces(i_row, i_col)) > 0.00000000000001)
            D_A2_Dist(i_row, i_col) = D_A2_LineList_Pixels_AllPieces(i_row, i_col) - D_A2_LineList_WLenPix_In(i_row, 1);
        }
      }
      cout << "CFits::StretchAndCrossCorrelateSpec: D_A2_Dist = " << D_A2_Dist << endl;
      Array<int, 2> I_A2_Where(D_A2_LineList_Pixels_AllPieces.rows(), D_A2_LineList_Pixels_AllPieces.cols());
      I_A2_Where = where(fabs(D_A2_LineList_Pixels_AllPieces) > 0.000001, 1, 0);
      #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
        cout << "CFits::StretchAndCrossCorrelateSpec: I_A2_Where = " << I_A2_Where << endl;
      #endif
      Array<int, 2> I_A2_IndWhere(2,2);
      if (!this->GetIndex(I_A2_Where, I_NInd, I_A2_IndWhere)){
        cout << "CFits::StretchAndCrossCorrelateSpec: ERROR: GetIndex returned FALSE" << endl;
        return false;
      }
      #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
        cout << "CFits::StretchAndCrossCorrelateSpec: I_A2_IndWhere = " << I_A2_IndWhere << endl;
      #endif
      Array<double, 1> D_A1_Dist_SubArr(2);
      if (!this->GetSubArrCopy(D_A2_Dist, I_A2_IndWhere, D_A1_Dist_SubArr)){
        cout << "CFits::StretchAndCrossCorrelateSpec: ERROR: GetSubArrCopy returned FALSE" << endl;
        return false;
      }
      #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
        cout << "CFits::StretchAndCrossCorrelateSpec: D_A1_Dist_SubArr = " << D_A1_Dist_SubArr << endl;
      #endif
      double D_MedianDiff = this->Median(D_A1_Dist_SubArr);
      Array<double, 1> *P_D_A1_Sort = this->BubbleSort(D_A1_Dist_SubArr);
      #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
        cout << "CFits::StretchAndCrossCorrelateSpec: D_MedianDiff = " << D_MedianDiff << endl;
        cout << "CFits::StretchAndCrossCorrelateSpec: *P_D_A1_Sort = " << *P_D_A1_Sort << endl;
      #endif
      Array<double, 1> D_A1_Dist_Temp(D_A1_Dist_SubArr.size()-4);
      D_A1_Dist_Temp = (*P_D_A1_Sort)(Range(2, D_A1_Dist_SubArr.size()-3));
      #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
        cout << "CFits::StretchAndCrossCorrelateSpec: D_A1_Dist_Temp = " << D_A1_Dist_Temp << endl;
      #endif
      double D_StdDev_Diff = this->StdDev(D_A1_Dist_Temp);
      #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
        cout << "CFits::StretchAndCrossCorrelateSpec: D_StdDev_Diff = " << D_StdDev_Diff << endl;
      #endif
      I_A1_Where.resize(D_A1_Dist_SubArr.size());
      I_A1_Where = where(fabs(D_A1_Dist_SubArr - D_MedianDiff) > (3. * D_StdDev_Diff), 1, 0);
      #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
        cout << "CFits::StretchAndCrossCorrelateSpec: I_A1_Where = " << I_A1_Where << endl;
      #endif
      Array<int, 1> I_A1_IndWhere(2);
      int I_NBad = 0;
      if (max(I_A1_Where) > 0){
        if (!this->GetIndex(I_A1_Where, I_NBad, I_A1_IndWhere)){
          cout << "CFits::StretchAndCrossCorrelateSpec: ERROR: 2. GetIndex returned FALSE" << endl;
          return false;
        }
        #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
          cout << "CFits::StretchAndCrossCorrelateSpec: I_NBad = " << I_NBad << ": I_A1_IndWhere = " << I_A1_IndWhere << endl;
        #endif
        for (int i_bad=0; i_bad<I_NBad; i_bad++){
          D_A2_LineList_Pixels_AllPieces(I_A2_IndWhere(I_A1_IndWhere(i_bad), 0), I_A2_IndWhere(I_A1_IndWhere(i_bad), 1)) = 0.;
          #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
            cout << "CFits::StretchAndCrossCorrelateSpec: i_bad = " << i_bad << ": D_A2_LineList_Pixels_AllPieces(" << I_A2_IndWhere(I_A1_IndWhere(i_bad), 0) << ", " << I_A2_IndWhere(I_A1_IndWhere(i_bad), 1) << ") set to " << D_A2_LineList_Pixels_AllPieces(I_A2_IndWhere(I_A1_IndWhere(i_bad), 0), I_A2_IndWhere(I_A1_IndWhere(i_bad), 1)) << endl;
          #endif
        }
        #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
          cout << "CFits::StretchAndCrossCorrelateSpec: D_A2_LineList_Pixels_AllPieces = " << D_A2_LineList_Pixels_AllPieces << endl;
        #endif

        I_A1_Where.resize(D_A2_LineList_Pixels_AllPieces.cols());
        D_A2_LineList_WLenPix_Out(Range::all(), 1) = 0.;
        for (int i_line=0; i_line<D_A2_LineList_WLenPix_In.rows(); i_line++){
          I_A1_Where = where(fabs(D_A2_LineList_Pixels_AllPieces(i_line, Range::all())) > 0.001, 1, 0);
          P_I_A1_IndWhere = this->GetIndex(I_A1_Where, I_NInd);
          #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
            cout << "CFits::StretchAndCrossCorrelateSpec: i_line = " << i_line << ": I_NInd = " << I_NInd << endl;
          #endif
          if (I_NInd == 0)
            D_A2_LineList_WLenPix_Out(i_line, 1) = D_A2_LineList_WLenPix_In(i_line, 1) + D_MedianDiff;
          else if (I_NInd == 1)
            D_A2_LineList_WLenPix_Out(i_line, 1) = D_A2_LineList_Pixels_AllPieces(i_line, (*P_I_A1_IndWhere)(0));
          else{
            for (int i_ind=0; i_ind<I_NInd; i_ind++){
              D_A2_LineList_WLenPix_Out(i_line, 1) += D_A2_LineList_Pixels_AllPieces(i_line, (*P_I_A1_IndWhere)(i_ind));
              #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
                cout << "CFits::StretchAndCrossCorrelateSpec: i_line = " << i_line << ": i_ind = " << i_ind << ": (*P_I_A1_IndWhere)(" << i_ind << ") = " << (*P_I_A1_IndWhere)(i_ind) << endl;
                cout << "CFits::StretchAndCrossCorrelateSpec: i_line = " << i_line << ": i_ind = " << i_ind << ": D_A2_LineList_WLenPix_Out(" << i_line << ", 1) set to " << D_A2_LineList_WLenPix_Out(i_line, 1) << endl;
              #endif
            }
            D_A2_LineList_WLenPix_Out(i_line, 1) = D_A2_LineList_WLenPix_Out(i_line, 1) / I_NInd;
          }
          #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
            cout << "CFits::StretchAndCrossCorrelateSpec: D_A2_LineList_WLenPix_Out(" << i_line << ", 1) set to " << D_A2_LineList_WLenPix_Out(i_line, 1) << endl;
          #endif
          delete(P_I_A1_IndWhere);
        }
      }
    //  if (I_NBad > 0)
    //    return false;


      #ifdef __DEBUG_FITS_STRETCHANDCROSSCORRELATESPEC__
        cout << "CFits::StretchAndCrossCorrelateSpec: D_A2_LineList_Pixels_AllPieces = " << D_A2_LineList_Pixels_AllPieces << endl;
        cout << "CFits::StretchAndCrossCorrelateSpec: D_A2_LineList_WLenPix_In = " << D_A2_LineList_WLenPix_In << endl;
        cout << "CFits::StretchAndCrossCorrelateSpec: D_A2_LineList_WLenPix_Out = " << D_A2_LineList_WLenPix_Out << endl;
      #endif

      return true;
    }
*/
    template StretchAndCrossCorrelateResult< float > stretchAndCrossCorrelate( ndarray::Array< float, 1, 1 > const&,
                                                                               ndarray::Array< float, 1, 1 > const&,
                                                                               int const,
                                                                               int const,
                                                                               int const,
                                                                               int const );
    template StretchAndCrossCorrelateResult< double > stretchAndCrossCorrelate( ndarray::Array< double, 1, 1 > const&,
                                                                                ndarray::Array< double, 1, 1 > const&,
                                                                                int const,
                                                                                int const,
                                                                                int const,
                                                                                int const );

    
    template bool interPol( ndarray::Array< float, 1, 1 > const&,
                            ndarray::Array< float, 1, 1 > const&,
                            ndarray::Array< float, 1, 1 > const&,
                            ndarray::Array< float, 1, 1> &);
    template bool interPol( ndarray::Array< double, 1, 1 > const&,
                            ndarray::Array< double, 1, 1 > const&,
                            ndarray::Array< double, 1, 1 > const&,
                            ndarray::Array< double, 1, 1> &);
    
    template bool interPol( ndarray::Array< float, 1, 1 > const&,
                            ndarray::Array< float, 1, 1 > const&,
                            ndarray::Array< float, 1, 1 > const&,
                            ndarray::Array< float, 1, 1 > &,
                            bool );
    template bool interPol( ndarray::Array< double, 1, 1 > const&,
                            ndarray::Array< double, 1, 1 > const&,
                            ndarray::Array< double, 1, 1 > const&,
                            ndarray::Array< double, 1, 1 > &,
                            bool );
    
    template bool interPol( ndarray::Array< float, 1, 1 > const&,
                            ndarray::Array< float, 1, 1 > const&,
                            ndarray::Array< float, 1, 1 > const&,
                            std::vector< std::string > const&,
                            ndarray::Array< float, 1, 1 > &);
    template bool interPol( ndarray::Array< double, 1, 1 > const&,
                            ndarray::Array< double, 1, 1 > const&,
                            ndarray::Array< double, 1, 1 > const&,
                            std::vector< std::string > const&,
                            ndarray::Array< double, 1, 1 > &);
    
    template ndarray::Array< int, 1, 1 > valueLocate( ndarray::Array< double, 1, 1 > const&, 
                                                      ndarray::Array< double, 1, 1 > const&);
    template ndarray::Array< int, 1, 1 > valueLocate( ndarray::Array< int, 1, 1 > const&, 
                                                      ndarray::Array< int, 1, 1 > const&);
    
    template bool hInterPol( ndarray::Array< float, 1, 1 > const&,
                             ndarray::Array< float, 1, 1 > const&,
                             ndarray::Array< int, 1, 1 > &,
                             ndarray::Array< float, 1, 1 > const&,
                             std::vector< std::string > const&,
                             ndarray::Array< float, 1, 1 > &);
    template bool hInterPol( ndarray::Array< double, 1, 1 > const&,
                             ndarray::Array< double, 1, 1 > const&,
                             ndarray::Array< int, 1, 1 > &,
                             ndarray::Array< double, 1, 1 > const&,
                             std::vector< std::string > const&,
                             ndarray::Array< double, 1, 1 > &);

    template bool splInt( ndarray::Array< double, 1, 1 > const&, 
                          ndarray::Array< double, 1, 1 > const&, 
                          ndarray::Array< double, 1, 1> const&, 
                          double X,
                          double & Y);

    template ndarray::Array< float, 1, 1 > splineI( ndarray::Array< float, 1, 1 > const&, 
                                                     ndarray::Array< float, 1, 1 > const&, 
                                                     float const, 
                                                     float const);
    template ndarray::Array< double, 1, 1 > splineI( ndarray::Array< double, 1, 1 > const&, 
                                                     ndarray::Array< double, 1, 1 > const&, 
                                                     double const, 
                                                     double const);

    template ndarray::Array< float, 1, 1 > splineI( ndarray::Array< float, 1, 1 > const&, 
                                                     ndarray::Array< float, 1, 1 > const& );
    template ndarray::Array< double, 1, 1 > splineI( ndarray::Array< double, 1, 1 > const&, 
                                                     ndarray::Array< double, 1, 1 > const& );
    
    template ndarray::Array< int, 1, 1 > whereGe( ndarray::Array< int, 1, 1 > const&,
                                                  int const, 
                                                  int const,
                                                  int const );
    
    template ndarray::Array< int, 1, 1 > whereGe( ndarray::Array< double, 1, 1 > const&,
                                                  double const, 
                                                  int const,
                                                  int const );

    template ndarray::Array< double, 1, 1 > whereGe( ndarray::Array< double, 1, 1 > const&,
                                                     double const, 
                                                     double const,
                                                     double const );

    template ndarray::Array< int, 1, 1 > whereGe( ndarray::Array< int, 1, 1 > const&,
                                                  int const, 
                                                  int const,
                                                  ndarray::Array< int, 1, 1 > const& );

    template ndarray::Array< int, 1, 1 > whereGe( ndarray::Array< double, 1, 1 > const&,
                                                  double const, 
                                                  int const,
                                                  ndarray::Array< int, 1, 1 > const& );

    template ndarray::Array< double, 1, 1 > whereGe( ndarray::Array< double, 1, 1 > const&,
                                                     double const, 
                                                     double const,
                                                     ndarray::Array< double, 1, 1 > const& );

    template ndarray::Array< int, 1, 1 > whereGt( ndarray::Array< int, 1, 1 > const&,
                                                  int const, 
                                                  int const,
                                                  int const );

    template ndarray::Array< int, 1, 1 > whereGt( ndarray::Array< double, 1, 1 > const&,
                                                  double const, 
                                                  int const,
                                                  int const );

    template ndarray::Array< double, 1, 1 > whereGt( ndarray::Array< double, 1, 1 > const&,
                                                     double const, 
                                                     double const,
                                                     double const );

    template ndarray::Array< int, 1, 1 > whereGt( ndarray::Array< int, 1, 1 > const&,
                                                  int const, 
                                                  int const,
                                                  ndarray::Array< int, 1 , 1 > const& );

    template ndarray::Array< int, 1, 1 > whereGt( ndarray::Array< double, 1, 1 > const&,
                                                  double const, 
                                                  int const,
                                                  ndarray::Array< int, 1 , 1 > const& );

    template ndarray::Array< double, 1, 1 > whereGt( ndarray::Array< double, 1, 1 > const&,
                                                     double const, 
                                                     double const,
                                                     ndarray::Array< double, 1, 1 > const& );

    template ndarray::Array< int, 1, 1 > whereLt( ndarray::Array< int, 1, 1 > const&,
                                                  int const, 
                                                  int const,
                                                  int const );

    template ndarray::Array< int, 1, 1 > whereLt( ndarray::Array< double, 1, 1 > const&,
                                                  double const, 
                                                  int const,
                                                  int const );
    
    template ndarray::Array< double, 1, 1 > whereLt( ndarray::Array< double, 1, 1 > const&,
                                                     double const, 
                                                     double const,
                                                     double const );

    template ndarray::Array< int, 1, 1 > whereLt( ndarray::Array< int, 1, 1 > const&,
                                                  int const, 
                                                  int const,
                                                  ndarray::Array< int, 1, 1 > const& );

    template ndarray::Array< int, 1, 1 > whereLt( ndarray::Array< double, 1, 1 > const&,
                                                  double const, 
                                                  int const,
                                                  ndarray::Array< int, 1, 1 > const& );

    template ndarray::Array< double, 1, 1 > whereLt( ndarray::Array< double, 1, 1 > const&,
                                                     double const, 
                                                     double const,
                                                     ndarray::Array< double, 1, 1 > const& );

    template ndarray::Array< int, 1, 1 > whereLe( ndarray::Array< int, 1, 1 > const&,
                                                  int const, 
                                                  int const,
                                                  int const );

    template ndarray::Array< int, 1, 1 > whereLe( ndarray::Array< double, 1, 1 > const&,
                                                  double const, 
                                                  int const,
                                                  int const );

    template ndarray::Array< double, 1, 1 > whereLe( ndarray::Array< double, 1, 1 > const&,
                                                     double const, 
                                                     double const,
                                                     double const );

    template ndarray::Array< int, 1, 1 > whereLe( ndarray::Array< int, 1, 1 > const&,
                                                  int const, 
                                                  int const,
                                                  ndarray::Array< int, 1, 1 > const& );

    template ndarray::Array< int, 1, 1 > whereLe( ndarray::Array< double, 1, 1 > const&,
                                                  double const, 
                                                  int const,
                                                  ndarray::Array< int, 1, 1 > const& );

    template ndarray::Array< double, 1, 1 > whereLe( ndarray::Array< double, 1, 1 > const&,
                                                     double const, 
                                                     double const,
                                                     ndarray::Array< double, 1, 1 > const& );
    
    template bool lsToFit( ndarray::Array< double, 1, 1 > const&, 
                           ndarray::Array< double, 1, 1 > const&, 
                           double const&, 
                           double &);
    
    template bool crossCorrelate( ndarray::Array< float, 1, 1 > const&,
                                  ndarray::Array< float, 1, 1 > const&,
                                  int const,
                                  int const,
                                  double &,
                                  double &);

    template bool crossCorrelate( ndarray::Array< double, 1, 1 > const&,
                                  ndarray::Array< double, 1, 1 > const&,
                                  int const,
                                  int const,
                                  double &,
                                  double &);
    
    template bool crossCorrelate( ndarray::Array< double, 1, 1 > const&,
                                  ndarray::Array< double, 1> const&,
                                  int const,
                                  int const,
                                  int &,
                                  double & );
    
    template bool crossCorrelate( ndarray::Array< float, 1, 1 > const&,
                                  ndarray::Array< float, 1> const&,
                                  int const,
                                  int const,
                                  int &,
                                  double & );
    
    template float calcRMS( ndarray::Array< float, 1, 1 > const& );
    template double calcRMS( ndarray::Array< double, 1, 1 > const& );
    
    
    template int isMonotonic( ndarray::Array< size_t, 1, 1 > const& );
    template int isMonotonic( ndarray::Array< int, 1, 1 > const& );
    template int isMonotonic( ndarray::Array< float, 1, 1 > const& );
    template int isMonotonic( ndarray::Array< double, 1, 1 > const& );
    
    template ndarray::Array< float, 1, 1 > getDataInRange( ndarray::Array< float, 1, 1 > const&,
                                                           ndarray::Array< float, 1, 1 > const&,
                                                           ndarray::Array< float, 1, 1 > const&,
                                                           ndarray::Array< float, 1, 1 > const&,
                                                           ndarray::Array< float, 1, 1 > const& );
    template ndarray::Array< double, 1, 1 > getDataInRange( ndarray::Array< double, 1, 1 > const&,
                                                            ndarray::Array< double, 1, 1 > const&,
                                                            ndarray::Array< double, 1, 1 > const&,
                                                            ndarray::Array< double, 1, 1 > const&,
                                                            ndarray::Array< double, 1, 1 > const& );

    template ndarray::Array< float, 1, 1 > getDataInRange( ndarray::Array< float, 1, 1 > const&,
                                                           ndarray::Array< float, 1, 1 > const&,
                                                           ndarray::Array< float, 1, 1 > const&,
                                                           ndarray::Array< float, 1, 1 > const& );
    template ndarray::Array< double, 1, 1 > getDataInRange( ndarray::Array< double, 1, 1 > const&,
                                                            ndarray::Array< double, 1, 1 > const&,
                                                            ndarray::Array< double, 1, 1 > const&,
                                                            ndarray::Array< double, 1, 1 > const& );
    
    template float calculateChiSquare( ndarray::Array< float, 1, 1 > const&, ndarray::Array< float, 1, 1 > const&);
    template double calculateChiSquare( ndarray::Array< double, 1, 1 > const&, ndarray::Array< double, 1, 1 > const&);
    
    template ndarray::Array< double, 2, 1 > createRectangularGrid( ndarray::Array< double, 1, 1 > const&, ndarray::Array< double, 1, 1 > const&, double, double);    
    
    template ndarray::Array< double, 2, 1 > createPolarGrid( double, double, double );
    
    template ndarray::Array< float, 1, 1 > getZMinMaxInRange( ndarray::Array< float, 1, 1 > const&,
                                                              ndarray::Array< float, 1, 1 > const&,
                                                              ndarray::Array< float, 1, 1 > const&,
                                                              ndarray::Array< float, 1, 1 > const&,
                                                              ndarray::Array< float, 1, 1 > const& );
    template ndarray::Array< double, 1, 1 > getZMinMaxInRange( ndarray::Array< double, 1, 1 > const&,
                                                               ndarray::Array< double, 1, 1 > const&,
                                                               ndarray::Array< double, 1, 1 > const&,
                                                               ndarray::Array< double, 1, 1 > const&,
                                                               ndarray::Array< double, 1, 1 > const& );
    
    template void insertSorted(std::vector< dataXY< int > > &, dataXY< int > &);
    template void insertSorted(std::vector< dataXY< float > > &, dataXY< float > &);
    template void insertSorted(std::vector< dataXY< double > > &, dataXY< double > &);

    template ndarray::Array<int, 2, 1> ndArray21(int, int);
    template ndarray::Array<float, 2, 1> ndArray21(float, float);
    template ndarray::Array<double, 2, 1> ndArray21(double, double);
    template ndarray::Array<int, 2, 2> ndArray22(int, int);
    template ndarray::Array<float, 2, 2> ndArray22(float, float);
    template ndarray::Array<double, 2, 2> ndArray22(double, double);
    
    template ndarray::Array< unsigned short const, 1, 1 > vectorToNdArray(std::vector<unsigned short> const&);
    template ndarray::Array< unsigned int const, 1, 1 > vectorToNdArray(std::vector<unsigned int> const&);
    template ndarray::Array< unsigned long const, 1, 1> vectorToNdArray(std::vector<unsigned long> const&);
    template ndarray::Array< int const, 1, 1 > vectorToNdArray(std::vector<int> const&);
    template ndarray::Array< long const, 1, 1 > vectorToNdArray(std::vector<long> const&);
    template ndarray::Array< float const, 1, 1 > vectorToNdArray(std::vector<float> const&);
    template ndarray::Array< double const, 1, 1 > vectorToNdArray(std::vector<double> const&);
    
    template ndarray::Array<unsigned short, 1, 1> vectorToNdArray(std::vector<unsigned short> &);
    template ndarray::Array<unsigned int, 1, 1> vectorToNdArray(std::vector<unsigned int> &);
    template ndarray::Array<unsigned long, 1, 1> vectorToNdArray(std::vector<unsigned long> &);
    template ndarray::Array<int, 1, 1> vectorToNdArray(std::vector<int> &);
    template ndarray::Array<long, 1, 1> vectorToNdArray(std::vector<long> &);
    template ndarray::Array<float, 1, 1> vectorToNdArray(std::vector<float> &);
    template ndarray::Array<double, 1, 1> vectorToNdArray(std::vector<double> &);
    
    template std::vector< unsigned short > ndArrayToVector( ndarray::Array< unsigned short, 1, 1 > const&);
    template std::vector< unsigned int > ndArrayToVector( ndarray::Array< unsigned int, 1, 1 > const&);
    template std::vector< unsigned long > ndArrayToVector( ndarray::Array< unsigned long, 1, 1 > const&);
    template std::vector< int > ndArrayToVector( ndarray::Array< int, 1, 1 > const&);
    template std::vector< long > ndArrayToVector( ndarray::Array< long, 1, 1 > const&);
    template std::vector< float > ndArrayToVector( ndarray::Array< float, 1, 1 > const&);
    template std::vector< double > ndArrayToVector( ndarray::Array< double, 1, 1 > const&);
    
    template bool checkIfValuesAreInRange(ndarray::Array<int, 1, 1> const&, ndarray::Array<int, 1, 1> const& range);
    template bool checkIfValuesAreInRange(ndarray::Array<int, 1, 1> const&, ndarray::Array<float, 1, 1> const& range);
    template bool checkIfValuesAreInRange(ndarray::Array<int, 1, 1> const&, ndarray::Array<double, 1, 1> const& range);
    template bool checkIfValuesAreInRange(ndarray::Array<float, 1, 1> const&, ndarray::Array<int, 1, 1> const& range);
    template bool checkIfValuesAreInRange(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const& range);
    template bool checkIfValuesAreInRange(ndarray::Array<float, 1, 1> const&, ndarray::Array<double, 1, 1> const& range);
    template bool checkIfValuesAreInRange(ndarray::Array<double, 1, 1> const&, ndarray::Array<int, 1, 1> const& range);
    template bool checkIfValuesAreInRange(ndarray::Array<double, 1, 1> const&, ndarray::Array<float, 1, 1> const& range);
    template bool checkIfValuesAreInRange(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const& range);
    
    template float convertRangeToUnity(float, ndarray::Array<float, 1, 1> const&);
    template float convertRangeToUnity(float, ndarray::Array<double, 1, 1> const&);
    template double convertRangeToUnity(double, ndarray::Array<float, 1, 1> const&);
    template double convertRangeToUnity(double, ndarray::Array<double, 1, 1> const&);
    template ndarray::Array<float, 1, 1> convertRangeToUnity(ndarray::Array<float, 1, 1> const&, ndarray::Array<float, 1, 1> const&);
    template ndarray::Array<float, 1, 1> convertRangeToUnity(ndarray::Array<float, 1, 1> const&, ndarray::Array<double, 1, 1> const&);
    template ndarray::Array<double, 1, 1> convertRangeToUnity(ndarray::Array<double, 1, 1> const&, ndarray::Array<float, 1, 1> const&);
    template ndarray::Array<double, 1, 1> convertRangeToUnity(ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&);
    
    template float xCor(ndarray::Array< float, 2, 1 > const&, ndarray::Array< float, 2, 1 > const&, ndarray::Array< float, 1, 1 > const&, float const&);
    template float xCor(ndarray::Array< float, 2, 1 > const&, ndarray::Array< double, 2, 1 > const&, ndarray::Array< float, 1, 1 > const&, float const&);
    template double xCor(ndarray::Array< double, 2, 1 > const&, ndarray::Array< float, 2, 1 > const&, ndarray::Array< double, 1, 1 > const&, double const&);
    template double xCor(ndarray::Array< double, 2, 1 > const&, ndarray::Array< double, 2, 1 > const&, ndarray::Array< double, 1, 1 > const&, double const&);
    
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
    template ndarray::Array<size_t, 1, 1> getSubArray(ndarray::Array<size_t, 1, 1> const&, ndarray::Array<int, 1, 1> const&);
    template ndarray::Array<int, 1, 1> getSubArray(ndarray::Array<int, 1, 1> const&, ndarray::Array<int, 1, 1> const&);
    template ndarray::Array<long, 1, 1> getSubArray(ndarray::Array<long, 1, 1> const&, ndarray::Array<int, 1, 1> const&);
    template ndarray::Array<float, 1, 1> getSubArray(ndarray::Array<float, 1, 1> const&, ndarray::Array<int, 1, 1> const&);
    template ndarray::Array<double, 1, 1> getSubArray(ndarray::Array<double, 1, 1> const&, ndarray::Array<int, 1, 1> const&);

    template ndarray::Array<size_t, 1, 1> getSubArray(ndarray::Array<size_t, 2, 1> const&, ndarray::Array<size_t, 2, 1> const&);
    template ndarray::Array<int, 1, 1> getSubArray(ndarray::Array<int, 2, 1> const&, ndarray::Array<size_t, 2, 1> const&);
    template ndarray::Array<long, 1, 1> getSubArray(ndarray::Array<long, 2, 1> const&, ndarray::Array<size_t, 2, 1> const&);
    template ndarray::Array<float, 1, 1> getSubArray(ndarray::Array<float, 2, 1> const&, ndarray::Array<size_t, 2, 1> const&);
    template ndarray::Array<double, 1, 1> getSubArray(ndarray::Array<double, 2, 1> const&, ndarray::Array<size_t, 2, 1> const&);
    template ndarray::Array<size_t, 1, 1> getSubArray(ndarray::Array<size_t, 2, 1> const&, ndarray::Array<int, 2, 1> const&);
    template ndarray::Array<int, 1, 1> getSubArray(ndarray::Array<int, 2, 1> const&, ndarray::Array<int, 2, 1> const&);
    template ndarray::Array<long, 1, 1> getSubArray(ndarray::Array<long, 2, 1> const&, ndarray::Array<int, 2, 1> const&);
    template ndarray::Array<float, 1, 1> getSubArray(ndarray::Array<float, 2, 1> const&, ndarray::Array<int, 2, 1> const&);
    template ndarray::Array<double, 1, 1> getSubArray(ndarray::Array<double, 2, 1> const&, ndarray::Array<int, 2, 1> const&);
    template ndarray::Array<size_t, 1, 1> getSubArray(ndarray::Array<size_t, 2, 1> const&, ndarray::Array<long, 2, 1> const&);
    template ndarray::Array<int, 1, 1> getSubArray(ndarray::Array<int, 2, 1> const&, ndarray::Array<long, 2, 1> const&);
    template ndarray::Array<long, 1, 1> getSubArray(ndarray::Array<long, 2, 1> const&, ndarray::Array<long, 2, 1> const&);
    template ndarray::Array<float, 1, 1> getSubArray(ndarray::Array<float, 2, 1> const&, ndarray::Array<long, 2, 1> const&);
    template ndarray::Array<double, 1, 1> getSubArray(ndarray::Array<double, 2, 1> const&, ndarray::Array<long, 2, 1> const&);
    template ndarray::Array<int, 1, 1> getSubArray(ndarray::Array<int, 2, 1> const&, ndarray::Array<unsigned int, 2, 1> const&);
    template ndarray::Array<long, 1, 1> getSubArray(ndarray::Array<long, 2, 1> const&, ndarray::Array<unsigned int, 2, 1> const&);
    template ndarray::Array<float, 1, 1> getSubArray(ndarray::Array<float, 2, 1> const&, ndarray::Array<unsigned int, 2, 1> const&);
    template ndarray::Array<double, 1, 1> getSubArray(ndarray::Array<double, 2, 1> const&, ndarray::Array<unsigned int, 2, 1> const&);

    template ndarray::Array<size_t, 1, 1> getSubArray(ndarray::Array<size_t, 2, 1> const&, std::vector< std::pair<size_t, size_t> > const&);
    template ndarray::Array<int, 1, 1> getSubArray(ndarray::Array<int, 2, 1> const&, std::vector< std::pair<size_t, size_t> > const&);
    template ndarray::Array<long, 1, 1> getSubArray(ndarray::Array<long, 2, 1> const&, std::vector< std::pair<size_t, size_t> > const&);
    template ndarray::Array<float, 1, 1> getSubArray(ndarray::Array<float, 2, 1> const&, std::vector< std::pair<size_t, size_t> > const&);
    template ndarray::Array<double, 1, 1> getSubArray(ndarray::Array<double, 2, 1> const&, std::vector< std::pair<size_t, size_t> > const&);

    template std::vector<size_t> removeSubArrayFromArray(std::vector<size_t> const&, std::vector<size_t> const&);
    template std::vector<int> removeSubArrayFromArray(std::vector<int> const&, std::vector<int> const&);
    template std::vector<long> removeSubArrayFromArray(std::vector<long> const&, std::vector<long> const&);
    template std::vector<float> removeSubArrayFromArray(std::vector<float> const&, std::vector<float> const&);
    template std::vector<double> removeSubArrayFromArray(std::vector<double> const&, std::vector<double> const&);

    template std::vector<size_t> getIndices(std::vector<size_t> const&);
    template std::vector<size_t> getIndices(std::vector<int> const&);
    template std::vector<size_t> getIndices(std::vector<long> const&);
    template std::vector<size_t> getIndices(std::vector<float> const&);
    template std::vector<size_t> getIndices(std::vector<double> const&);

    template ndarray::Array< size_t, 1, 1 > getIndices(ndarray::Array< size_t, 1, 1 > const&);
    template ndarray::Array< size_t, 1, 1 > getIndices(ndarray::Array< int, 1, 1 > const&);
    template ndarray::Array< size_t, 1, 1 > getIndices(ndarray::Array< long, 1, 1 > const&);
    template ndarray::Array< size_t, 1, 1 > getIndices(ndarray::Array< float, 1, 1 > const&);
    template ndarray::Array< size_t, 1, 1 > getIndices(ndarray::Array< double, 1, 1 > const&);

    template ndarray::Array<size_t, 1, 1> getIndicesInValueRange(ndarray::Array<size_t, 1, 1> const&, size_t const, size_t const);
    template ndarray::Array<size_t, 1, 1> getIndicesInValueRange(ndarray::Array<int, 1, 1> const&, int const, int const);
    template ndarray::Array<size_t, 1, 1> getIndicesInValueRange(ndarray::Array<long, 1, 1> const&, long const, long const);
    template ndarray::Array<size_t, 1, 1> getIndicesInValueRange(ndarray::Array<float, 1, 1> const&, float const, float const);
    template ndarray::Array<size_t, 1, 1> getIndicesInValueRange(ndarray::Array<double, 1, 1> const&, double const, double const);

    template ndarray::Array<size_t, 2, 1> getIndicesInValueRange(ndarray::Array<size_t, 2, 1> const&, size_t const, size_t const);
    template ndarray::Array<size_t, 2, 1> getIndicesInValueRange(ndarray::Array<int, 2, 1> const&, int const, int const);
    template ndarray::Array<size_t, 2, 1> getIndicesInValueRange(ndarray::Array<long, 2, 1> const&, long const, long const);
    template ndarray::Array<size_t, 2, 1> getIndicesInValueRange(ndarray::Array<float, 2, 1> const&, float const, float const);
    template ndarray::Array<size_t, 2, 1> getIndicesInValueRange(ndarray::Array<double, 2, 1> const&, double const, double const);
    
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
    template ndarray::Array<double, 1, 1> Double(ndarray::Array<double const, 1, 1> const&);

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

    template ndarray::Array<float, 2, 1> Float(ndarray::Array<size_t, 2, 1> const&);
    template ndarray::Array<float, 2, 1> Float(ndarray::Array<unsigned short, 2, 1> const&);
    template ndarray::Array<float, 2, 1> Float(ndarray::Array<int, 2, 1> const&);
    template ndarray::Array<float, 2, 1> Float(ndarray::Array<long, 2, 1> const&);
    template ndarray::Array<float, 2, 1> Float(ndarray::Array<float, 2, 1> const&);
    template ndarray::Array<float, 2, 1> Float(ndarray::Array<double, 2, 1> const&);

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
    template ndarray::Array<size_t, 1, 1> floor(const ndarray::Array<float, 1, 1>&, const size_t);
    template ndarray::Array<size_t, 1, 1> floor(const ndarray::Array<double, 1, 1>&, const size_t);
    template ndarray::Array<unsigned int, 1, 1> floor(const ndarray::Array<const float, 1, 1>&, const unsigned int);
    template ndarray::Array<unsigned int, 1, 1> floor(const ndarray::Array<const double, 1, 1>&, const unsigned int);
    template ndarray::Array<unsigned int, 1, 1> floor(const ndarray::Array<float, 1, 1>&, const unsigned int);
    template ndarray::Array<unsigned int, 1, 1> floor(const ndarray::Array<double, 1, 1>&, const unsigned int);
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
    
    template ndarray::Array<size_t, 2, 1> calcMinCenMax(ndarray::Array<float const, 1, 1> const&, float const, float const, int const, int const);
    template ndarray::Array<size_t, 2, 1> calcMinCenMax(ndarray::Array<double const, 1, 1> const&, float const, float const, int const, int const);
    template ndarray::Array<size_t, 2, 1> calcMinCenMax(ndarray::Array<float, 1, 1> const&, float const, float const, int const, int const);
    template ndarray::Array<size_t, 2, 1> calcMinCenMax(ndarray::Array<double, 1, 1> const&, float const, float const, int const, int const);

    template ndarray::Array<size_t, 2, 1> calcMinCenMax(ndarray::Array<float const, 1, 1> const&, double const, double const, int const, int const);
    template ndarray::Array<size_t, 2, 1> calcMinCenMax(ndarray::Array<double const, 1, 1> const&, double const, double const, int const, int const);
    template ndarray::Array<size_t, 2, 1> calcMinCenMax(ndarray::Array<float, 1, 1> const&, double const, double const, int const, int const);
    template ndarray::Array<size_t, 2, 1> calcMinCenMax(ndarray::Array<double, 1, 1> const&, double const, double const, int const, int const);
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
