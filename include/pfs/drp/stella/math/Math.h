///TODO: Replace all pointers with sharedPointers!

#ifndef __PFS_DRP_STELLA_MATH_H__
#define __PFS_DRP_STELLA_MATH_H__

#include <vector>
#include <iostream>
#include "lsst/base.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/config.h"
//#include "../blitz.h"
#include "../utils/Utils.h"
#include "ndarray.h"
#include "ndarray/eigen.h"

//#define __DEBUG_FIT__
//#define __DEBUG_FITARR__
//#define __DEBUG_POLY__
//#define __DEBUG_POLYFIT__
//#define __DEBUG_MINCENMAX__
//#define __DEBUG_INDGEN__
#define __DEBUG_SORT__

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
using namespace std;

namespace pfs { namespace drp { namespace stella {
  namespace math{

    /**
     * Calculates aperture minimum pixel, central position, and maximum pixel for the trace,
     * and writes result to I_A2_MinCenMax_Out
     * Note that if the width of the trace varies depending on the position of the aperture center,
     * 1 pixel left and/or right of the maximum aperture width will get cut off to reduce possible
     * cross-talk between adjacent apertures
     **/
    ndarray::Array<size_t, 2, 2> calcMinCenMax(ndarray::Array<float const, 1, 1> const& xCenters_In,
                                               float const xHigh_In,/// >= 0
                                               float const xLow_In,/// <= 0
                                               int const nPixCutLeft_In,
                                               int const nPixCutRight_In);

    /**
     * Fix(double)
     * Returns integer value cut at decimal point. If D_In is negative the integer value greater than or equal to D_In is returned,
     * e.g. D_In = -99.8 => returns -99.
     **/
    template <typename T>
    int Fix(T D_In);
    //%template(fixd) Fix(double);
    /**
     * FixL(double)
     * Returns integer value cut at decimal point (See int Fix(double)).
     **/
    template <typename T>
    long FixL(T D_In);

    /**
     * Rounds x downward, returning the largest integral value that is not greater than x.

     * @param rhs: value to be rounded down
     * @param outType: type of this parameter determines the type of the return value. The value of this parameter has no meaning.
     * @return rounded down value of rhs, type of outType
     */
    template <typename T, typename U>
    U floor1(T const& rhs, U const& outType);
    
    template <typename T, typename U>
    ndarray::Array<U, 1, 1> floor(ndarray::Array<const T, 1, 1> const& rhs, U const outType);
    
    template <typename T, typename U>
    ndarray::Array<U, 2, 2> floor(ndarray::Array<const T, 2, 2> const& rhs, U const outType);

    /**
     * Int(double)
     * Returns integer portion of D_In. If D_In is negative returns the first negative integer less than or equal to Number,
     * e.g. D_In = -99.8 => returns -100.
     **/
    template <typename T>
    int Int(T D_In);

    /**
     * Returns integer value cut at decimal point (See int Int(double)).
     **/
    template <typename T>
    long Long(T D_In);
    template <typename T>
    ndarray::Array<double, 1, 1> Double(ndarray::Array<T, 1, 1> const& arr_In);
    
    template <typename T>
    ndarray::Array<double, 2, 1> Double(ndarray::Array<T, 2, 1> const& arr_In);
    
    template <typename T>
    ndarray::Array<double, 2, 2> Double(ndarray::Array<T, 2, 2> const& arr_In);
    
    template <typename T>
    ndarray::Array<float, 1, 1> Float(ndarray::Array<T, 1, 1> const& arr_In);
    
    template <typename T>
    ndarray::Array<float, 2, 2> Float(ndarray::Array<T, 2, 2> const& arr_In);
    
    template <typename T>
    int Round(const T ToRound);

    template <typename T>
    T Round(const T ToRound, int DigitsBehindDot);

    template <typename T>
    long RoundL(const T ToRound);
    
    /**
     * @brief Creates standard vector of length len containing the index numbers as values
     * @param len: length of output vector
     * @return 
     */
    template<typename T>
    std::vector<T> indGen(T len);
    
    template< typename T >
    std::vector<T> removeSubArrayFromArray(std::vector<T> const& A1_Array_InOut,
                                           std::vector<T> const& A1_SubArray);

    /**
     *        InvertGaussJ(AArray)
     *        Linear equation solution by Gauss-Jordan elimination with B == Unity
     *        AArray(0:N-1, 0:N-1) is the input matrix.
     *        On output, AArray is replaced by its matrix inverse.
     **
    template< typename T >
    bool InvertGaussJ(ndarray::Array<T, 2, 1> &AArray);
*/
    template<typename T>
    bool countPixGTZero(ndarray::Array<T, 1, 1> &vec_InOut);

    /**
     *        function FirstIndexWithValueGEFrom
     *        returns first index of integer input vector where value is greater than or equal to I_MinValue, starting at index I_FromIndex
     *        returns -1 if values are always smaller than I_MinValue
     **/
    template<typename T>
    int firstIndexWithValueGEFrom(ndarray::Array<T, 1, 1> const& vecIn, const T minValue, const int fromIndex);

    /**
     *        function LastIndexWithZeroValueBefore
     *        returns last index of integer input vector where value is equal to zero, starting at index I_StartPos
     *        returns -1 if values are always greater than 0 before I_StartPos
     **/
    template<typename T>
    int lastIndexWithZeroValueBefore(ndarray::Array<T, 1, 1> const& vec_In, const int startPos_In);

    /**
     *        function FirstIndexWithZeroValueFrom
     *        returns first index of integer input vector where value is equal to zero, starting at index I_StartPos
     *        returns -1 if values are always greater than 0 past I_StartPos
     **/
    template<typename T>
    int firstIndexWithZeroValueFrom(ndarray::Array<T, 1, 1> const& vec_In, const int startPos_In);

    bool IsOddNumber(long No);

    /**
     *      SortIndices(blitz::Array<double, 1> D_A1_In)
     *      Returns an integer array of the same size like <D_A1_In>,
     *      containing the indixes of <D_A1_In> in sorted order.
     **/
    template<typename T>
    std::vector<int> sortIndices(const std::vector<T> &vec_In);

    /**
     *       function GetRowFromIndex(int I_Index_In, int I_NRows_In) const
     *       task: Returns Row specified by I_Index_In from the formula
     *             Col = (int)(I_Index_In / I_NRows_In)
     *             Row = I_Index_In - Col * I_NRows_In
     **/
    int GetRowFromIndex(int I_Index_In, int I_NRows_In);

    /**
     *       function GetColFromIndex(int I_Index_In, int I_NRows_In) const
     *       task: Returns Col specified by I_Index_In from the formula
     *             Col = (int)(I_Index_In / I_NRows_In)
     *             Row = I_Index_In - Col * I_NRows_In
     **/
    int GetColFromIndex(int I_Index_In, int I_NRows_In);

    template<typename T>
    ndarray::Array<T, 1, 1> moment(ndarray::Array<T, 1, 1> const& arr_In, int maxMoment_In);

    template<typename T>
    T max(ndarray::Array<T, 1, 1> const& in);

    template<typename T>
    size_t maxIndex(ndarray::Array<T, 1, 1> const& in);
 
    template<typename T>
    T min(ndarray::Array<T, 1, 1> const& in);
 
    template<typename T>
    size_t minIndex(ndarray::Array<T, 1, 1> const& in);

    template<typename T>
    ndarray::Array<T, 1, 1> indGenNdArr(T const size);

    template<typename T>
    ndarray::Array<T, 1, 1> replicate(T const val, int const size);
    
    template<typename T>
    ndarray::Array<T, 2, 2> calcPosRelativeToCenter(ndarray::Array<T, 2, 2> const& swath_In, ndarray::Array<T, 1, 1> const& xCenters_In);
    
    /*
     * @brief: Return vector of indices where lowRange_In <= arr_In < highRange_In
     */
    template<typename T>
    ndarray::Array<size_t, 1, 1> getIndicesInValueRange(ndarray::Array<T, 1, 1> const& arr_In, T const lowRange_In, T const highRange_In);
    template<typename T>
    ndarray::Array<size_t, 2, 2> getIndicesInValueRange(ndarray::Array<T, 2, 2> const& arr_In, T const lowRange_In, T const highRange_In);

    /*
     * @brief: Returns array to copies of specified elements of arr_In
     */
    template<typename T>
    ndarray::Array<T, 1, 1> getSubArray(ndarray::Array<T, 1, 1> const& arr_In, ndarray::Array<size_t, 1, 1> const& indices_In);

    template<typename T>
    ndarray::Array<T, 1, 1> getSubArray(ndarray::Array<T, 2, 2> const& arr_In, ndarray::Array<size_t, 2, 2> const& indices_In);

    template<typename T>
    ndarray::Array<T, 1, 1> getSubArray(ndarray::Array<T, 2, 2> const& arr_In, std::vector< std::pair<size_t, size_t> > const& indices_In);
    
    template< typename T >
    ndarray::Array< T, 1, 1 > resize(ndarray::Array< T, 1, 1 > const& arr_In, size_t newSize); 
    
//    template< typename T >
//    ndarray::Array< T, 2, 1 > get2DArray(ndarray::Array< T, 1, 1 > const& xIn, ndarray::Array< T, 1, 1 > const& yIn);
  }/// end namespace math
  
}}}

template< typename T >
std::ostream& operator<<(std::ostream& os, std::vector<T> const& obj);

#endif