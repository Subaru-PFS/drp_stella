///TODO: Replace all pointers with sharedPointers!

#ifndef __PFS_DRP_STELLA_MATH_H__
#define __PFS_DRP_STELLA_MATH_H__

#include <vector>
#include <iostream>
#include "lsst/base.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/config.h"
#include "chebyshev.h"
#include "../utils/Utils.h"
#include "ndarray.h"
#include "ndarray/eigen.h"
#include "../cmpfit-1.2/MPFitting_ndarray.h"
#include <unsupported/Eigen/Splines>

//#define __DEBUG_FIT__
//#define __DEBUG_FITARR__
//#define __DEBUG_POLY__
//#define __DEBUG_POLYFIT__
//#define __DEBUG_MINCENMAX__
//#define __DEBUG_INDGEN__
//#define __DEBUG_SORT__
//#define __DEBUG_XCOR__

/// constants
#define CONST_PI 3.141592653589793238462643383280    /* pi */

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
using namespace std;

namespace pfs { namespace drp { namespace stella {
  namespace math{
    
    template< typename T >
    struct dataIndex { 
        T number;
        size_t index;
    };

    template< typename T >
    struct by_number { 
        bool operator()(dataIndex<T> const &left, dataIndex<T> const &right) { 
            return left.number < right.number;
        }
    };
    
    template< typename T >
    struct dataXY { 
        T x;
        T y;
    };

    template< typename T >
    struct byX { 
        bool operator()(dataXY<T> const &left, dataXY<T> const &right) { 
            return left.x < right.x;
        }
    };

    template< typename T >
    struct byY { 
        bool operator()(dataXY<T> const &left, dataXY<T> const &right) { 
            return left.y < right.y;
        }
    };

    /*
     * @brief insert <toInsert_In> into <dataXYVec_In> in sorted order in x
     * 
     * @param dataXYVec_In vector to insert <toInsert_In> in sorted order
     * @param toInsert_In  data pair to insert
     */
    template< typename T >
    void insertSorted(std::vector< dataXY< T > > & dataXYVec_In,
                      dataXY< T > & toInsert_In);
    
    /**
     * Calculates aperture minimum pixel, central position, and maximum pixel for the trace,
     * and writes result to I_A2_MinCenMax_Out
     * Note that if the width of the trace varies depending on the position of the aperture center,
     * 1 pixel left and/or right of the maximum aperture width will get cut off to reduce possible
     * cross-talk between adjacent apertures
     **/
    template< typename T, typename U >
    ndarray::Array<size_t, 2, 1> calcMinCenMax(ndarray::Array<T const, 1, 1> const& xCenters_In,
                                               U const xHigh_In,/// >= 0
                                               U const xLow_In,/// <= 0
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
    ndarray::Array<float, 2, 1> Float(ndarray::Array<T, 2, 1> const& arr_In);
    
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
    ndarray::Array<T, 2, 1> calcPosRelativeToCenter(ndarray::Array<T, 2, 1> const& swath_In, ndarray::Array<T, 1, 1> const& xCenters_In);
    
    /*
     * @brief: Return vector of indices where lowRange_In <= arr_In < highRange_In
     */
    template<typename T>
    ndarray::Array<size_t, 1, 1> getIndicesInValueRange(ndarray::Array<T, 1, 1> const& arr_In, T const lowRange_In, T const highRange_In);
    template<typename T>
    ndarray::Array<size_t, 2, 1> getIndicesInValueRange(ndarray::Array<T, 2, 1> const& arr_In, T const lowRange_In, T const highRange_In);

    /*
     * @brief: Returns array to copies of specified elements of arr_In
     */
    template<typename T>
    ndarray::Array<T, 1, 1> getSubArray(ndarray::Array<T, 1, 1> const& arr_In, ndarray::Array<size_t, 1, 1> const& indices_In);

    template<typename T, typename U>
    ndarray::Array<T, 1, 1> getSubArray(ndarray::Array<T, 2, 1> const& arr_In, ndarray::Array<U, 2, 1> const& indices_In);

    template<typename T>
    ndarray::Array<T, 1, 1> getSubArray(ndarray::Array<T, 2, 1> const& arr_In, std::vector< std::pair<size_t, size_t> > const& indices_In);
    
    template< typename T >
    ndarray::Array< T, 1, 1 > resize(ndarray::Array< T, 1, 1 > const& arr_In, size_t newSize); 

    /*
     * @brief: cross-correlates arrA_In and arrB_In within the range range_In (e.g. [-1.,1.]) in steps stepSize_In
     * 
     * @param arrA_In: vector to be kept constant arrA_In[:][0]: x values, arrA_In[:][1]: y values
     * @param arrA_In: vector to be shifted from range_In[0] to range_In[1] with steps of size stepSize_In. This vector will get interpolated to the grid points of vecA_In.
     * @param range_In: 2-element vector containing the lowest and highest shifts, e.g. [-1., 1.]
     * @param stepSize_In: step size for the shifts between range_In[0] and range_In[1]
     */    
    template< typename PsfT, typename CoordT >
    CoordT xCor(ndarray::Array< CoordT, 2, 1 > const& arrA_In,
                ndarray::Array< PsfT, 2, 1 > const& arrB_In,
                ndarray::Array< CoordT, 1, 1 > const& range_In,
                CoordT const& stepSize_In);

    /*
     * Helper methods for xCor
     */    
    double uvalue(double x, double low, double high);
    Eigen::VectorXd uvalues(Eigen::VectorXd const& xvals);
    
    /*
     * @brief convert given number in given range to a number in range [-1,1]
     * @param number: number to be converted
     * @param range: range number is from
     */
    template< typename T, typename U >
    T convertRangeToUnity(T number,
                          ndarray::Array<U, 1, 1> const& range);
    
    /*
     * @brief convert given numbers in given range to a number in range [-1,1]
     * @param numbers: numbers to be converted
     * @param range: range numbers are from
     */
    template< typename T, typename U >
    ndarray::Array<T, 1, 1> convertRangeToUnity(ndarray::Array<T, 1, 1> const& numbers,
                                                ndarray::Array<U, 1, 1> const& range);
    
    /**
     * @brief check if the values in numbers are within the given range
     * @param numbers numbers to be check if they all fall into the given range
     * @param range expected data range of numbers
     * @return true if all values in numbers fall into the given range, otherwise return false
     */
    template< typename T, typename U >
    bool checkIfValuesAreInRange(ndarray::Array<T, 1, 1> const& numbers,
                                 ndarray::Array<U, 1, 1> const& range);
    
    /**
     * @brief return ndarray pointing to data of vector
     * @param vector : vector to be converted to ndarray
     */
    template< typename T >
    ndarray::Array<T, 1, 1> vectorToNdArray(std::vector<T> & vector);
    
    template< typename T >
    ndarray::Array< T const, 1, 1 > vectorToNdArray(std::vector<T> const& vec_In);
    
    /**
     * @brief return vector containing copy of data in ndarray
     * @param ndArray_In : ndarray to be converted to vector
     */
    template< typename T >
    std::vector< T > ndArrayToVector( ndarray::Array< T, 1, 1 > const& ndArray_In);
    
    /**
     * @brief return ndarray of specified size and type
     */
    template< typename T >
    ndarray::Array<T, 2, 1> ndArray21(T nRows, T nCols);
    template< typename T >
    ndarray::Array<T, 2, 2> ndArray22(T nRows, T nCols);
    
    /**
     * @brief return minimum and maximum z value in given x and y range. return value: (zMin, zMax)
     * @param x_In: x values, size N
     * @param y_In: y values, size N
     * @param z_In: z values, size N
     * @param xRange_In: (xMin, xMax)
     * @param yRange_In: (yMin, yMax)
     */
    template< typename T >
    ndarray::Array< T, 1, 1 > getZMinMaxInRange( ndarray::Array< T, 1, 1 > const& x_In,
                                                 ndarray::Array< T, 1, 1 > const& y_In,
                                                 ndarray::Array< T, 1, 1 > const& z_In,
                                                 ndarray::Array< T, 1, 1 > const& xRange_In,
                                                 ndarray::Array< T, 1, 1 > const& yRange_In );
    
//    template< typename T >
//    ndarray::Array< T, 2, 1 > get2DArray(ndarray::Array< T, 1, 1 > const& xIn, ndarray::Array< T, 1, 1 > const& yIn);
  }/// end namespace math
  
}}}

template< typename T >
std::ostream& operator<<(std::ostream& os, std::vector<T> const& obj);

#endif