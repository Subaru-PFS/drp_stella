///TODO: Replace all pointers with sharedPointers!

#ifndef __PFS_DRP_STELLA_MATH_H__
#define __PFS_DRP_STELLA_MATH_H__

#include <numeric>      // std::accumulate
#include <vector>

#include "lsst/log/Log.h"
#include "ndarray.h"

#include "pfs/drp/stella/utils/Utils.h"

/// constants
#define CONST_PI 3.141592653589793238462643383280    /* pi */

namespace pfs { namespace drp { namespace stella {
  namespace math{
    
    template< typename T >
    struct dataXY { 
        T x;
        T y;
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
     * Rounds x downward, returning the largest integral value that is not greater than x.

     * @param rhs: value to be rounded down
     * @param outType: type of this parameter determines the type of the return value. The value of this parameter has no meaning.
     * @return rounded down value of rhs, type of outType
     */
    template <typename T, typename U>
    ndarray::Array<U, 1, 1> floor(ndarray::Array<T, 1, 1> const& rhs, U const outType);

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
                                   const int fromIndex = 0);

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

    /**
     *      Returns an integer array of the same size like <vec_In>,
     *      containing the indixes of <vec_In> in sorted order.
     * @param[in] vec_In       :: vector to sort
     **/
    template<typename T>
    std::vector<int> sortIndices(const std::vector<T> &vec_In);

    template<typename T>
    ndarray::Array<T, 1, 1> moment(ndarray::Array<T, 1, 1> const& arr_In, int maxMoment_In);

    template<typename T>
    T max(ndarray::Array<T, 1, 1> const& in);
 
    template<typename T>
    T min(ndarray::Array<T, 1, 1> const& in);

    template<typename T>
    ndarray::Array<T, 1, 1> indGenNdArr(T const size);

    template<typename T>
    ndarray::Array<T, 1, 1> replicate(T const val, int const size);
    
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
     * @param[in] arr_In     :: 1D vector to create subarray from
     * @param[in] indices_In :: indices of arr_In which shall be copied to output subarray
     */
    template< typename T, typename U >
    std::vector< T > getSubVector( std::vector< T > const& arr_In,
                                   std::vector< U > const& indices_In );

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

    template< typename T, int I >
    bool resize( ndarray::Array< T, 2, I > & arr_In,
                 size_t newSizeRows, 
                 size_t newSizeCols ); 

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
        float pixShift;
        float chiSquare;
    };
    
    /**
      CrossCorrelate with Gauss fit to ChiSquare to find subpixel of minimum
     **/
    template< typename T >
    CrossCorrelateResult crossCorrelate( ndarray::Array<T, 1, 1> const& DA1_Static,
                                         ndarray::Array<T, 1, 1> const& DA1_Moving,
                                         int const I_NPixMaxLeft,
                                         int const I_NPixMaxRight );
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

    /**
      SplInt
      Given the Arrays XAVecArr(0:N-1) and YAVecArr(0:N-1), which tabulate a function (whith the XAVecArr(i)'s in order), and given the array Y2AVecArr(0:N-1), which is the output from Spline above, and given a value of X, this routine returns a cubic-spline interpolated value Y;
     **/
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
                                         std::vector<std::string> const& CS_A1_In);

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
      ;       If the input vector is float or complex, the result is
      ;       float or complex.
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
        float stretch;
        float shift;
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
    float median( ndarray::Array< T, 1, 1 > const& vec );
    
    template< typename T >
    ndarray::Array< T, 1, 1 > stretch( ndarray::Array< T, 1, 1 > const& spec,
                                       int newLength );
    
    template< typename T > 
    int find( ndarray::Array< T, 1, 1 > const& arrToSearch,
              T val );
  }/// end namespace math

}}}

template< typename T >
std::ostream& operator<<(std::ostream& os, std::vector<T> const& obj);

#endif
