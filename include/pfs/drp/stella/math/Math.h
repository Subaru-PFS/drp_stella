///TODO: Replace all pointers with sharedPointers!

#ifndef __PFS_DRP_STELLA_MATH_H__
#define __PFS_DRP_STELLA_MATH_H__

#include <vector>
#include <iostream>
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
//#define __DEBUG_INTERPOL__
//#define __DEBUG_RESIZE__
//#define __DEBUG_STRETCH__
//#define __DEBUG_STRETCHANDCROSSCORRELATE__
//#define __DEBUG_CROSSCORRELATE__
//#define __DEBUG_WHERE__

/// constants
#define CONST_PI 3.141592653589793238462643383280    /* pi */

using namespace std;

namespace pfs { namespace drp { namespace stella {
  /// Center of Pixel Zero (0.0 means that [0,0] is the center of the pixel, 0.5 means that [0,0] is the lower left corner)
  const float PIXEL_CENTER = 0.0;

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
     * @param[in,out] dataXYVec_In :: vector to insert <toInsert_In> in sorted order
     * @param[in] toInsert_In      :: data pair to insert
     */
    template< typename T >
    void insertSorted( std::vector< dataXY< T > > & dataXYVec_In,
                       dataXY< T > & toInsert_In );
    
    /**
     * Calculates aperture minimum pixel, central position, and maximum pixel for the trace,
     * and writes result to I_A2_MinCenMax_Out
     * Note that if the width of the trace varies depending on the position of the aperture center,
     * 1 pixel left and/or right of the maximum aperture width will get cut off to reduce possible
     * cross-talk between adjacent apertures
     * @param[in] xCenters_In       :: x center positions of trace
     * @param[in] xHigh_In          :: width of trace right of trace (>=0)
     * @param[in] xLow_In           :: width of trace left of trace (<=0)
     * @param[in] nPixCutLeft_In    :: number of pixels to cut off left of trace
     * @param[in] nPixCutRight_In   :: number of pixels to cut off right of trace
     **/
    template< typename T, typename U >
    ndarray::Array<size_t, 2, 1> calcMinCenMax(ndarray::Array<T, 1, 1> const& xCenters_In,
                                               U const xHigh_In,
                                               U const xLow_In,
                                               int const nPixCutLeft_In = 0,
                                               int const nPixCutRight_In = 0);

    /**
     * Returns integer value cut at decimal point. If D_In is negative the integer value greater than or equal to D_In is returned,
     * e.g. D_In = -99.8 => returns -99.
     * @param[in] D_In :: number to fix
     **/
    template <typename T>
    int Fix(T D_In);

    /**
     * Returns integer value cut at decimal point (See int Fix(double)).
     * @param[in] D_In :: number to fix
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
    ndarray::Array<U, 1, 1> floor(ndarray::Array<T, 1, 1> const& rhs, U const outType);
    
    template <typename T, typename U>
    ndarray::Array<U, 2, 2> floor(ndarray::Array<T, 2, 2> const& rhs, U const outType);

    /**
     * Int(double)
     * Returns integer portion of D_In. If D_In is negative returns the first negative integer less than or equal to Number,
     * e.g. D_In = -99.8 => returns -100.
     * @param[in] D_In :: number to convert to int
     **/
    template <typename T>
    int Int(T D_In);

    /**
     * Returns integer value cut at decimal point (See int Int(double)).
     * @param[in] D_In :: number to convert to long
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
     *        returns first index of integer input vector where value is greater than or equal to I_MinValue, starting at index I_FromIndex
     *        returns -1 if values are always smaller than I_MinValue
     * @param[in] vecIn     :: 1D array to search for number >= minValue
     * @param[in] minValue  :: minimum value to search for
     * @param[in] fromIndex :: index position to start search 
     **/
    template<typename T>
    int firstIndexWithValueGEFrom( ndarray::Array< T, 1, 1 > const& vecIn, 
                                   const T minValue, 
                                   const int fromIndex);

    /**
     *        returns last index of integer input vector where value is equal to zero, starting at index I_StartPos
     *        returns -1 if values are always greater than 0 before I_StartPos
     * @param[in] vec_In       :: 1D array to search
     * @param[in] startPos_In :: index position to start search
     **/
    template<typename T>
    int lastIndexWithZeroValueBefore( ndarray::Array< T, 1, 1 > const& vec_In, 
                                      const int startPos_In );

    /**
     *        returns first index of integer input vector where value is equal to zero, starting at index I_StartPos
     *        returns -1 if values are always greater than 0 past I_StartPos
     * @param[in] vec_In       :: 1D array to search
     * @param[in] startPos_In :: index position to start search
     **/
    template<typename T>
    int firstIndexWithZeroValueFrom( ndarray::Array< T, 1, 1 > const& vec_In, 
                                     const int startPos_In );

    bool IsOddNumber(long No);

    /**
     *      Returns an integer array of the same size like <vec_In>,
     *      containing the indixes of <vec_In> in sorted order.
     * @param[in] vec_In       :: vector to sort
     **/
    template<typename T>
    std::vector<int> sortIndices(const std::vector<T> &vec_In);
    template<typename T>
    ndarray::Array<size_t, 1, 1> sortIndices(ndarray::Array<T, 1, 1> const& arr_In);

    /**
     *       task: Returns Row specified by I_Index_In from the formula
     *             Col = (int)(I_Index_In / I_NRows_In)
     *             Row = I_Index_In - Col * I_NRows_In
     * @param[in] I_Index_In   :: index number to convert to row number
     * @param[in] I_NRows_In   :: number of rows of 2D array
     **/
    int GetRowFromIndex(int I_Index_In, int I_NRows_In);

    /**
     *       task: Returns Col specified by I_Index_In from the formula
     *             Col = (int)(I_Index_In / I_NRows_In)
     *             Row = I_Index_In - Col * I_NRows_In
     * @param[in] I_Index_In   :: index number to convert to column number
     * @param[in] I_NRows_In   :: number of rows of 2D array
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
     * @param[in] arr_In         :: 1D array to search for values in given range
     * @param[in] lowRange_In    :: lower range
     * @param[in] lowRange_In    :: upper range
     */
    template< typename T, int I >
    ndarray::Array< size_t, 1, 1 > getIndicesInValueRange( ndarray::Array< T, 1, I > const& arr_In, 
                                                           T const lowRange_In, 
                                                           T const highRange_In );
    /*
     * @brief: Return vector of indices where lowRange_In <= arr_In < highRange_In
     * @param[in] arr_In         :: 2D array to search for values in given range
     * @param[in] lowRange_In    :: lower range
     * @param[in] lowRange_In    :: upper range
     */
    template< typename T >
    ndarray::Array< size_t, 2, 1 > getIndicesInValueRange( ndarray::Array< T, 2, 1 > const& arr_In, 
                                                           T const lowRange_In, 
                                                           T const highRange_In );

    /*
     * @brief: Return vector of indices where vec_In == 1
     * @param[in] vec_In  :: vector to search for ones
     */
    template<typename T>
    std::vector< size_t > getIndices( std::vector< T > const& vec_In );

    /*
     * @brief: Return vector of indices where arr_In == 1
     * @param[in] arr_In  :: 1D array to search for ones
     */
    template<typename T>
    ndarray::Array< size_t, 1, 1 > getIndices( ndarray::Array< T, 1, 1 > const& arr_In );
    
    /*
     * @brief: Return vector of indices where arr_In == 1
     * @param[in] arr_In  :: 2D array to search for ones
     */
    template<typename T>
    ndarray::Array< size_t, 2, 1 > getIndices( ndarray::Array< T, 2, 1 > const& arr_In );

    /*
     * @brief: Returns array to copies of specified elements of arr_In
     * @param[in] arr_In     :: 1D array to create subarray from
     * @param[in] indices_In :: indices of arr_In which shall be copied to output subarray
     */
    template< typename T, typename U, int I >
    ndarray::Array< T, 1, 1 > getSubArray( ndarray::Array< T, 1, I > const& arr_In, 
                                           ndarray::Array< U, 1, 1 > const& indices_In );

    /*
     * @brief: Returns array to copies of specified elements of arr_In
     * @param[in] arr_In     :: 2D array to create subarray from
     * @param[in] indices_In :: indices of arr_In which shall be copied to output subarray
     */
    template< typename T, typename U >
    ndarray::Array< T, 1, 1 > getSubArray( ndarray::Array< T, 2, 1 > const& arr_In, 
                                           ndarray::Array< U, 2, 1 > const& indices_In );

    template< typename T >
    ndarray::Array< T, 1, 1 > getSubArray( ndarray::Array< T, 2, 1 > const& arr_In, 
                                           std::vector< std::pair< size_t, size_t > > const& indices_In );
    
    /*
     * @brief: resize arr_In to size newSize. Will result in cutting off a longer array at newSize or adding zeros to a shorter array.
     * 
     * @param arr_In: array to be resized
     * @param newSize: new size for arr_In
     */
    template< typename T >
    bool resize( ndarray::Array< T, 1, 1 > & arr_In, 
                 size_t newSize ); 

    template< typename T >
    bool resize( ndarray::Array< T, 2, 1 > & arr_In, 
                 size_t newSizeRows, 
                 size_t newSizeCols ); 

    /*
     * @brief: cross-correlates arrA_In and arrB_In within the range range_In (e.g. [-1.,1.]) in steps stepSize_In
     * 
     * @param arrA_In: vector to be kept constant arrA_In[:][0]: x values, arrA_In[:][1]: y values
     * @param arrB_In: vector to be shifted from range_In[0] to range_In[1] with steps of size stepSize_In. This vector will get interpolated to the grid points of vecA_In.
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
     * @param[in] vector :: vector to be converted to ndarray
     * @param[in] deep   :: create a deep copy of vec_In?
     */
    template< typename T >
    ndarray::Array< T, 1, 1 > vectorToNdArray( std::vector< T > & vector, bool deep = true );
    
    template< typename T >
    ndarray::Array< T const, 1, 1 > vectorToNdArray( std::vector< T > const& vec_In, bool deep = true );
    
    /**
     * @brief return vector containing copy of data in ndarray
     * @param ndArray_In : ndarray to be converted to vector
     */
    template< typename T >
    std::vector< T > ndArrayToVector( ndarray::Array< T, 1, 1 > const& ndArray_In);
    
    /**
     * @brief return ndarray of specified size and type
     * @param[in] nRows  :: number of rows of output array
     * @param[out] nCols :: number of columns of output array
     */
    template< typename T >
    ndarray::Array<T, 2, 1> ndArray21(T nRows, T nCols);
    
    /**
     * @brief return ndarray of specified size and type
     * @param[in] nRows  :: number of rows of output array
     * @param[out] nCols :: number of columns of output array
     */
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

    /**
     * @brief return array( 2, nGridPoints ) with x = array(0,*) and y = array(1,*)
     * @param xRange: [xMin, xMax]
     * @param yRange: [yMin, yMax]
     * @param xStep: dX
     * @param yStep: dY
     */    
    template< typename T >
    ndarray::Array< T, 2, 1 > createRectangularGrid( ndarray::Array< T, 1, 1 > const& xRange,
                                                     ndarray::Array< T, 1, 1 > const& yRange,
                                                     T xStep,
                                                     T yStep );
    
    /**
     * @brief return array( 2, nGridPoints ) with x = array(0,*) and y = array(1,*)
     * @param rMax: 0 <= r <= rMax
     * @param rStep: dR
     * @param phiStep: dPhi (degrees)
     */    
    template< typename T >
    ndarray::Array< T, 2, 1 > createPolarGrid( T rMax,
                                               T rStep,
                                               T phiStep );
    
    /**
     * @brief: Calculate the Chi-square for the expected and observed data values as Xhi^2 = sum_i (expected_i - observed_i )^2 / expected_i
     * @param expected: 1d array of length n of expected data values
     * @param observed: 1d array of length n of observed data values
     */
    template< typename T >
    T calculateChiSquare( ndarray::Array< T, 1, 1 > const& expected,
                          ndarray::Array< T, 1, 1 > const& observed );
    
    template< typename T > 
    ndarray::Array< T, 1, 1 > getDataInRange( ndarray::Array< T, 1, 1 > const& xArr,
                                              ndarray::Array< T, 1, 1 > const& yArr,
                                              ndarray::Array< T, 1, 1 > const& zArr,
                                              ndarray::Array< T, 1, 1 > const& xRange,
                                              ndarray::Array< T, 1, 1 > const& yRange );
    
    template< typename T > 
    ndarray::Array< T, 1, 1 > getDataInRange( ndarray::Array< T, 1, 1 > const& xArr,
                                              ndarray::Array< T, 1, 1 > const& yArr,
                                              ndarray::Array< T, 1, 1 > const& zArr,
                                              ndarray::Array< T, 1, 1 > const& rRange );
    
    /**
     * Returns -1 if D_A1_Array_In is monotonically decreasing
     *         0  if D_A1_Array_In is non monotonic
     *         +1 if D_A1_Array_In is monotonically increasing
     */
    template< typename T >
    int isMonotonic(ndarray::Array< T, 1, 1 > const& D_A1_Array_In);

    /**
     * @brief Calculate Root Mean Squared of arrIn
     */
    template< typename T >
    T calcRMS( ndarray::Array< T, 1, 1 > const& arrIn );
    
    struct CrossCorrelateResult{
        double pixShift;
        double chiSquare;
    };
    
    /**
      CrossCorrelate with Gauss fit to ChiSquare to find subpixel of minimum
     **/
    template< typename T >
    CrossCorrelateResult crossCorrelate( ndarray::Array<T, 1, 1> const& DA1_Static,
                                         ndarray::Array<T, 1, 1> const& DA1_Moving,
                                         int const I_NPixMaxLeft,
                                         int const I_NPixMaxRight );
    
    /**
      CrossCorrelate
     **/
    template< typename T >
    CrossCorrelateResult crossCorrelateI( ndarray::Array< T, 1, 1 > const& DA1_Static,
                                          ndarray::Array< T, 1, 1 > const& DA1_Moving,
                                          int const I_NPixMaxLeft,
                                          int const I_NPixMaxRight );
    template< typename T >
    T lsToFit( ndarray::Array< T, 1, 1 > const& XXVecArr, 
                  ndarray::Array< T, 1, 1 > const& YVecArr, 
                  T const& XM );
    
    template< typename T, typename U, int I >
    ndarray::Array< U, 1, 1 > where( ndarray::Array< T, 1, I > const& arrayToCompareTo,
                                     std::string const& op,
                                     T const valueToCompareTo, 
                                     U const valueIfTrue,
                                     U const valueIfFalse );
    
    template< typename T, typename U, int I >
    ndarray::Array< U, 2, 1 > where( ndarray::Array< T, 2, I > const& arrayToCompareTo,
                                     std::string const& op,
                                     T const valueToCompareTo, 
                                     U const valueIfTrue,
                                     U const valueIfFalse );
    
    template< typename T, typename U >
    ndarray::Array< U, 1, 1 > where( ndarray::Array< T, 1, 1 > const& arrayToCompareTo,
                                     std::string const& op,
                                     T const valueToCompareTo, 
                                     U const valueIfTrue,
                                     ndarray::Array< U, 1, 1 > const& valuesIfFalse );
    
    template< typename T, typename U >
    ndarray::Array< U, 1, 1 > where( ndarray::Array< T, 1, 1 > const& arrayToCompareTo,
                                     std::string const& op,
                                     T const valueToCompareTo, 
                                     ndarray::Array< U, 1, 1 > const& valuesIfTrue,
                                     ndarray::Array< U, 1, 1 > const& valuesIfFalse );
    
    template< typename T, typename U, int I, int J >
    ndarray::Array< U, 2, 1 > where( ndarray::Array< T, 2, I > const& arrayToCompareTo,
                                     std::string const& op,
                                     T const valueToCompareTo, 
                                     U const valueIfTrue,
                                     ndarray::Array< U, 2, J > const& valuesIfFalse );
    
    template< typename T, typename U >
    ndarray::Array< U, 2, 1 > where( ndarray::Array< T, 2, 1 > const& arrayToCompareTo,
                                     std::string const& op,
                                     T const valueToCompareTo, 
                                     ndarray::Array< U, 2, 1 > const& valuesIfTrue,
                                     ndarray::Array< U, 2, 1 > const& valuesIfFalse );
    
    /**
      Spline
      Given Arrays XVecArr(0:N-1) and YVecArr(0:N-1) containing a tabulated function, i.e., y_i = f(x_i), with x_1 < x_2 < ... < x_N, 
      and given values YP1 and YPN for the first derivative of the interpolating function at points 1 and N, respectively, this routine 
      returns an Array y2(0:N-1) that contains the second derivatives of the interpolating function at the tabulated points x_i. If YP1 
      and/or YPN are equal to 1x10^30 or larger, the routine is signaled to set the corresponding boundary condition for a natural spline, 
      with zero second derivative on that boundary.
     **/
    template< typename T >
    ndarray::Array< T, 1, 1 > splineI( ndarray::Array< T, 1, 1 > const& XVecArr, 
                                       ndarray::Array< T, 1, 1 > const& YVecArr, 
                                       T const YP1, 
                                       T const YPN);
    
    /**
      Spline
      Given Arrays XVecArr(0:N-1) and YVecArr(0:N-1) containing a tabulated function, i.e., y_i = f(x_i), with x_1 < x_2 < ... < x_N, 
      this routine returns an Array y2(0:N-1) that contains the second derivatives of the interpolating function at the tabulated points x_i. 
      The routine is signaled to set the corresponding boundary condition for a natural spline, with zero second derivative on that boundary.
     **/
    template< typename T >
    ndarray::Array< T, 1, 1 > splineI( ndarray::Array< T, 1, 1 > const& XVecArr, 
                                       ndarray::Array< T, 1, 1 > const& YVecArr);

    template< typename T >
    T splInt( ndarray::Array< T, 1, 1 > const& XAVecArr, 
                 ndarray::Array< T, 1, 1 > const& YAVecArr, 
                 ndarray::Array< T, 1, 1> const& Y2AVecArr, 
                 T X );

    template< typename T, int I >    
    ndarray::Array< T, 1, 1 > hInterPol( ndarray::Array< T, 1, 1 > const& VVecArr,
                                         ndarray::Array< T, 1, 1 > const& XVecArr,
                                         ndarray::Array< int, 1, 1 > & SVecArr,
                                         ndarray::Array< T, 1, I > const& UVecArr,
                                         std::vector< string > const& CS_A1_In);
//                    ndarray::Array< T, 1, 1 > & D1_Out);

    /**
     ValueLocate
     Returns the Start Index of the Range of the two indices of the monotonically increasing or decreasing Vector VecArr, in which Val falls.
     If Vector is monotonically increasing, the result is
       if j = -1       Value(i) < VecArr(0)
       if 0 <= j < N-1 VecArr(j) <= Value(i) < VecArr(j+1)
       if j = N-1      VecArr(N-1) <= Value(i)

     If Vector is monotonically decreasing, the result is
       if j = -1       VecArr(0) <= ValVecArr(i)
       if 0 <= j < N-1 VecArr(j+1) <= ValVecArr(i) < VecArr(j)
       if j = N-1      ValVecArr(i) < VecArr(N-1)
    **/
    template< typename T, int I >
    ndarray::Array< int, 1, 1 > valueLocate( ndarray::Array< T, 1, 1 > const& VecArr, 
                                             ndarray::Array< T, 1, I > const& ValVecArr );

    /**
      InterPol linear, not regular
     **/
    template< typename T >
    ndarray::Array< T, 1, 1 > interPol( ndarray::Array< T, 1, 1 > const& VVecArr,
                                        ndarray::Array< T, 1, 1 > const& XVecArr,
                                        ndarray::Array< T, 1, 1 > const& UVecArr );
  
    template< typename T, int I >
    ndarray::Array< T, 1, 1 > interPol( ndarray::Array< T, 1, 1 > const& VVecArr,
                                        ndarray::Array< T, 1, 1 > const& XVecArr,
                                        ndarray::Array< T, 1, I > const& UVecArr,
                                        bool B_PreserveFlux );
  
    /**
      InterPol irregular
     **/
      /**
      FUNCTION INTERPOL, V, X, U, SPLINE=spline, LSQUADRATIC=ls2, QUADRATIC=quad
      ;+
      ; NAME:
      ;       INTERPOL
      ;
      ; PURPOSE:
      ;       Linearly interpolate vectors with a regular or irregular
      ;       grid.
      ;       Quadratic or a 4 point least-square fit to a quadratic
      ;       interpolation may be used as an option.
      ;
      ; CATEGORY:
      ;       E1 - Interpolation
      ;
      ; CALLING SEQUENCE:
      ;       Result = INTERPOL(V, N)         ;For regular grids.
      ;
      ;       Result = INTERPOL(V, X, U)      ;For irregular grids.
      ;
      ; INPUTS:
      ;       V:      The input vector can be any type except
      ;
      ;
      ;       For regular grids:
      ;       N:      The number of points in the result when both
      ;               input and output grids are regular.
      ;
      ;       Irregular grids:
      ;       X:      The absicissae values for V.  This vector must
      ;               have same # of elements as V.  The values MUST be
      ;               monotonically ascending or descending.
      ;
      ;       U:      The absicissae values for the result.  The result
      ;               will have the same number of elements as U.  U
      ;               does not need to be monotonic.  If U is outside
      ;               the range of X, then the closest two endpoints of
      ;               (X,V) are linearly extrapolated.
      ;
      ; Keyword Input Parameters:
      ;       LSQUADRATIC = if set, interpolate using a least squares
      ;         quadratic fit to the equation y = a + bx + cx^2, for
      ;         each 4 point neighborhood (x[i-1], x[i], x[i+1], x[i+2])
      ;         surrounding the interval, x[i] <= u < x[i+1].
      ;
      ;       QUADRATIC = if set, interpolate by fitting a quadratic
      ;         y = a + bx + cx^2, to the three point neighborhood
      ;         (x[i-1], x[i], x[i+1]) surrounding the interval
      ;         x[i] <= u < x[i+1].
      ;
      ;       SPLINE = if set, interpolate by fitting a cubic spline to
      ;         the 4 point neighborhood (x[i-1], x[i], x[i+1], x[i+2])
      ;         surrounding the interval, x[i] <= u < x[i+1].
      ;
      ;       Note: if LSQUADRATIC or QUADRATIC or SPLINE is not set,
      ;       the default linear interpolation is used.
      ;
      ; OUTPUTS:
      ;       INTERPOL returns a floating-point vector of N points
      ;       determined by interpolating the input vector by the
      ;       specified method.
      ;
      ;       If the input vector is double or complex, the result is
      ;       double or complex.
      ;
      ; COMMON BLOCKS:
      ;       None.
      ;
      ; SIDE EFFECTS:
      ;       None.
      ;
      ; RESTRICTIONS:
      ;       None.
      ;
      ; PROCEDURE:
      ;       For linear interpolation,
      ;       Result(i) = V(x) + (x - FIX(x)) * (V(x+1) - V(x))
      ;
      ;       where   x = i*(m-1)/(N-1) for regular grids.
      ;               m = # of elements in V, i=0 to N-1.
      ;
      ;       For irregular grids, x = U(i).
      ;               m = number of points of input vector.
      ;
      ;         For QUADRATIC interpolation, the equation y=a+bx+cx^2 is
      ;       solved explicitly for each three point interval, and is
      ;       then evaluated at the interpolate.
      ;         For LSQUADRATIC interpolation, the coefficients a, b,
      ;       and c, from the above equation are found, for the four
      ;       point interval surrounding the interpolate using a least
      ;       square fit.  Then the equation is evaluated at the
      ;       interpolate.
      ;         For SPLINE interpolation, a cubic spline is fit over the
      ;       4 point interval surrounding each interpolate, using the
      ;       routine SPL_INTERP().
      ;
      ; MODIFICATION HISTORY:
      ;       Written, DMS, October, 1982.
      ;       Modified, Rob at NCAR, February, 1991.  Made larger arrays possible
      ;               and correct by using long indexes into the array instead of
      ;               integers.
      ;       Modified, DMS, August, 1998.  Now use binary intervals which
      ;               speed things up considerably when U is random.
      ;       DMS, May, 1999.  Use new VALUE_LOCATE function to find intervals,
      ;               which speeds things up by a factor of around 100, when
      ;               interpolating from large arrays.  Also added SPLINE,
      ;               QUADRATIC, and LSQUADRATIC keywords.
      ;-
      ;**/
    template< typename T, int I >
    ndarray::Array< T, 1, 1 > interPol( ndarray::Array< T, 1, 1 > const& VVecArr,
                                        ndarray::Array< T, 1, 1 > const& XVecArr,
                                        ndarray::Array< T, 1, I > const& UVecArr,
                                        std::vector< std::string > const& CS_A1_In );

    template< typename T >
    struct StretchAndCrossCorrelateResult{
        double stretch;
        double shift;
        ndarray::Array< T, 2, 1 > specStretchedMinChiSq;
    };
    
    template< typename T >
    StretchAndCrossCorrelateResult< T > stretchAndCrossCorrelate( ndarray::Array< T, 1, 1 > const& spec,
                                                                  ndarray::Array< T, 1, 1 > const& specRef,
                                                                  int const radiusXCor,
                                                                  int const stretchMinLength,
                                                                  int const stretchMaxLength,
                                                                  int const nStretches );
    
    template< typename T > 
    double median( ndarray::Array< T, 1, 1 > const& vec );
    
    template< typename T >
    ndarray::Array< T, 1, 1 > fabs( ndarray::Array< T, 1, 1 > const& arr );

    template< typename T >
    ndarray::Array< T, 2, 1 > fabs( ndarray::Array< T, 2, 1 > const& arr );
    
    template< typename T >
    ndarray::Array< T, 1, 1 > stretch( ndarray::Array< T, 1, 1 > const& spec,
                                       int newLength );
    
    template< typename T, int I >
    ndarray::Array< T, 1, 1 > unique( ndarray::Array< T, 1, I > const& data );
    
    template< typename T > 
    int find( ndarray::Array< T, 1, 1 > const& arrToSearch,
              T val );
    
    template< typename T > 
    int find( std::vector< T > const& vecToSearch,
              T val );

    //    template< typename T >
//    ndarray::Array< T, 2, 1 > get2DArray(ndarray::Array< T, 1, 1 > const& xIn, ndarray::Array< T, 1, 1 > const& yIn);
  }/// end namespace math
  
}}}

template< typename T >
std::ostream& operator<<(std::ostream& os, std::vector<T> const& obj);

#endif
