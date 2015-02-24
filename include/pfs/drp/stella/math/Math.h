///TODO: Replace all pointers with sharedPointers!

#ifndef __PFS_DRP_STELLA_MATH_H__
#define __PFS_DRP_STELLA_MATH_H__

#include <vector>
#include <iostream>
#include "lsst/base.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/config.h"
#include "../blitz.h"
#include "../utils/Utils.h"
#include "ndarray.h"
#include "ndarray/eigen.h"

//#define __DEBUG_FIT__
//#define __DEBUG_FITARR__
//#define __DEBUG_POLY__
//#define __DEBUG_POLYFIT__
//#define __DEBUG_MINCENMAX__
//#define __DEBUG_INDGEN__

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
using namespace std;

namespace pfs { namespace drp { namespace stella {
  namespace math{

    /*************************************************************
     * Poly
     *
     * INPUTS:
     *       D_A1_X_In:      The variable.  1D array.
     *
     *       D_A1_Coeffs_In: The 1D array of polynomial coefficients.  The degree of
     *                       of the polynomial is N_ELEMENTS(VecCoeffs) - 1.
     *
     * OUTPUTS:
     *       POLY returns a result equal to:
     *                C[0] + c[1] * X + c[2]*X^2 + ...
     *
    **/
    template<typename T, typename U>
    ndarray::Array<T, 1, 1> Poly(ndarray::Array<T, 1, 1> const& x_In,
                                 ndarray::Array<U, 1, 1> const& coeffs_In);

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
     * PURPOSE:
     *   Perform a least-square polynomial fit with optional error estimates.
     *
     *   This routine uses matrix inversion.  A newer version of this routine,
     *   SVDFIT, uses Singular Value Decomposition.  The SVD technique is more
     *   flexible, but slower.
     *
     * INPUTS:
     *   X:  The independent variable vector.
     *
     *   Y:  The dependent variable vector, should be same length as x.
     *
     *   Degree: The degree of the polynomial to fit.
     *
     * OUTPUTS:
     *   POLY_FIT returns a vector of coefficients with a length of NDegree+1.
     *
     * KEYWORDS:
     *   CHISQ=chisq: double: out:
     *     Sum of squared errors divided by MEASURE_ERRORS if specified.
     *
     *   COVAR=covar: blitz::Array<double, 2>(I_Degree+1, I_Degree+1): out:
     *     Covariance matrix of the coefficients.
     *
     *   MEASURE_ERRORS=measure_errors: blitz::Array<double, 1>(D_A1_X_In.size()): in:
     *     Set this keyword to a vector containing standard
     *     measurement errors for each point Y[i].  This vector must be the same
     *     length as X and Y.
     *
     *     Note - For Gaussian errors (e.g. instrumental uncertainties),
     *       MEASURE_ERRORS should be set to the standard
     *       deviations of each point in Y. For Poisson or statistical weighting
     *       MEASURE_ERRORS should be set to sqrt(Y).
     *
     *   SIGMA=sigma: blitz::Array<double, 1>(I_Degree+1): out:
     *     The 1-sigma error estimates of the returned parameters.
     *
     *     Note: if MEASURE_ERRORS is omitted, then you are assuming that
     *       your model is correct. In this case, SIGMA is multiplied
     *       by SQRT(CHISQ/(N-M)), where N is the number of points
     *       in X and M is the number of terms in the fitting function.
     *       See section 15.2 of Numerical Recipes in C (2nd ed) for details.
     *
     *   STATUS=status: int: out:
     *     Set this keyword to a named variable to receive the status
     *     of the operation. Possible status values are:
     *     0 for successful completion, 1 for a singular array (which
     *     indicates that the inversion is invalid), and 2 which is a
     *     warning that a small pivot element was used and that significant
     *     accuracy was probably lost.
     *
     *   YFIT:   blitz::Vector of calculated Y's. These values have an error
     *           of + or - YBAND.
     *
    CHISQ=chisq: double: out
    COVAR=covar: blitz::Array<double, 2>(I_Degree+1, I_Degree+1): out
    MEASURE_ERRORS=measure_errors: blitz::Array<double, 1>(D_A1_X_In.size()): in
    SIGMA=sigma: blitz::Array<double, 1>(I_Degree+1): out
    STATUS=status: int: out
    YFIT=yfit: blitz::Array<double, 1>(D_A1_X_In.size()): out
    **/

    template< typename T >
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In,
                                         std::vector<string> const& S_A1_Args_In,
                                         std::vector<void *> & ArgV);

    template< typename T >
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In);

/** Additional Keywords:
    REJECTED=blitz::Array<int, 1>
    NOT_REJECTED=blitz::Array<int, 1>
    N_REJECTED=int
    **/
    template<typename T>
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In,
                                         T const D_Reject_In,
                                         std::vector<string> const& S_A1_Args_In,
                                         std::vector<void *> & ArgV);

    template< typename T >
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                    ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                    size_t const I_Degree_In,
                                    T const D_LReject_In,
                                    T const D_UReject_In,
                                    size_t const I_NIter,
                                    std::vector<string> const& S_A1_Args_In,
                                    std::vector<void *> & ArgV);

    template< typename T >
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In,
                                         T const D_Reject_In);

    template< typename T>
    ndarray::Array<double, 1, 1> PolyFit(ndarray::Array<T, 1, 1> const& D_A1_X_In,
                                         ndarray::Array<T, 1, 1> const& D_A1_Y_In,
                                         size_t const I_Degree_In,
                                         T const D_LReject_In,
                                         T const D_HReject_In,
                                         size_t const I_NIter);


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

    /**
       CHANGES to original function:
         * D_Sky_Out must be >= 0. unless stated otherwise by the ALLOW_SKY_LT_ZERO parameter
         * D_SP_Out must be >= 0. unless stated otherwise by the ALLOW_SPEC_LT_ZERO parameter
         * added REJECT_IN as optinal parameter to reject cosmic rays from fit (times sigma)
         * added MASK_INOUT as optional parameter
     **/
      
      template< typename ImageT, typename SlitFuncT >
      int LinFitBevingtonEigen(Eigen::Array<ImageT, Eigen::Dynamic, 1> const& D_A1_CCD_In,      /// yvec: in
                               Eigen::Array<SlitFuncT, Eigen::Dynamic, 1> const& D_A1_SF_In,       /// xvec: in
                               ImageT &D_SP_Out,                         /// a1: out
                               ImageT &D_Sky_Out,                        /// a0: in/out
                               bool B_WithSky,                        /// with sky: in
                               std::vector<string> const& S_A1_Args_In,   ///: in
                               std::vector<void *> & ArgV_In);                   ///: in
    /// MEASURE_ERRORS_IN = Eigen::Array<ImageT, Eigen::Dynamic, 1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = float                                                : in
    /// MASK_INOUT = Eigen::Array<unsigned short, Eigen::Dynamic, 1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = ImageT                                                : out
    /// Q_OUT = float                                                    : out
    /// SIGMA_OUT = Eigen::Array<ImageT, Eigen::Dynamic, 1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = ndarray::Array<ImageT, 1>(D_A1_CCD_In.size)                     : out
    /// ALLOW_SKY_LT_ZERO = int[0,1]
    /// ALLOW_SPEC_LT_ZERO = int[0,1]
      
      template< typename ImageT, typename SlitFuncT >
      int LinFitBevingtonNdArray(ndarray::Array<ImageT, 1, 1> const& D_A1_CCD_In,      /// yvec: in
                                 ndarray::Array<SlitFuncT, 1, 1> const& D_A1_SF_In,       /// xvec: in
                                 ImageT &D_SP_Out,                         /// a1: out
                                 ImageT &D_Sky_Out,                        /// a0: in/out
                                 bool B_WithSky,                        /// with sky: in
                                 std::vector<string> const& S_A1_Args_In,   ///: in
                                 std::vector<void *> & ArgV_In);                   ///: in
    /// MEASURE_ERRORS_IN = blitz::Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = float                                                : in
    /// MASK_INOUT = blitz::Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = blitz::Array<double,1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = blitz::Array<double, 1>(D_A1_CCD_In.size)                     : out
    /// ALLOW_SKY_LT_ZERO = 1
    /// ALLOW_SPEC_LT_ZERO = 1

     /**
            CHANGES to original function:
              * D_Sky_Out must be >= 0.
              * D_SP_Out must be >= 0.
              * if D_Sky_Out is set to be < -1.e-10 in input it is set to 0. and D_SP_Out is calculated as if there was no sky at all
              * added REJECT_IN as optinal parameter to reject cosmic rays from fit (times sigma)
              * added MASK_INOUT as optional parameter
      **/
  /// MEASURE_ERRORS_IN = blitz::Array<double,2>(D_A2_CCD_In.rows, D_A2_CCD_In.cols) : in
  /// REJECT_IN = double                                                      : in
  /// MASK_INOUT = blitz::Array<double,2>(D_A1_CCD_In.rows,D_A1_CCD_In.cols)         : in/out
  /// CHISQ_OUT = blitz::Array<double,1>(D_A2_CCD_In.rows)                           : out
  /// Q_OUT = blitz::Array<double,1>(D_A2_CCD_In.rows)                               : out
  /// SIGMA_OUT = blitz::Array<double,2>(D_A2_CCD_In.rows, 2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
      template< typename ImageT, typename SlitFuncT>
      bool LinFitBevingtonEigen(Eigen::Array<ImageT, Eigen::Dynamic, Eigen::Dynamic> const& D_A2_CCD_In,      ///: in
                                Eigen::Array<SlitFuncT, Eigen::Dynamic, Eigen::Dynamic> const& D_A2_SF_In,       ///: in
                                Eigen::Array<ImageT, Eigen::Dynamic, 1> & D_A1_SP_Out,             ///: out
                                Eigen::Array<ImageT, Eigen::Dynamic, 1> & D_A1_Sky_Out,            ///: in/out
                                bool B_WithSky,                           ///: with sky: in
                                vector<string> const& S_A1_Args_In,   ///: in
                                vector<void *> &ArgV_In);                   ///: in/out
      template< typename ImageT, typename SlitFuncT>
      bool LinFitBevingtonNdArray(ndarray::Array<ImageT, 2, 1> const& D_A2_CCD_In,      ///: in
                                  ndarray::Array<SlitFuncT, 2, 1> const& D_A2_SF_In,       ///: in
                                  ndarray::Array<ImageT, 1, 1> & D_A1_SP_Out,             ///: out
                                  ndarray::Array<ImageT, 1, 1> & D_A1_Sky_Out,            ///: in/out
                                  bool B_WithSky,                           ///: with sky: in
                                  vector<string> const& S_A1_Args_In,   ///: in
                                  vector<void *> &ArgV_In);                   ///: in/out
    /// MEASURE_ERRORS_IN = blitz::Array<double,1>(D_A1_CCD_In.size)             : in
    /// REJECT_IN = double                                                : in
    /// MASK_INOUT = blitz::Array<int,1>(D_A1_CCD_In.size)                    : in/out
    /// CHISQ_OUT = double                                                : out
    /// Q_OUT = double                                                    : out
    /// SIGMA_OUT = blitz::Array<double,1>(2): [*,0]: sigma_sp, [*,1]: sigma_sky : out
    /// YFIT_OUT = blitz::Array<double, 1>(D_A1_CCD_In.size)                     : out

    /**
     *       Helper function to calculate incomplete Gamma Function
     **/
    template< typename T >
    T GammLn(T const D_X_In);

    /**
     *      Helper function to calculate incomplete Gamma Function
     **/
    template< typename T >
    T GCF(T & D_Gamser_In, T const a, T const x);

    /**
     *      Function to calculate incomplete Gamma Function P
     **/
    template< typename T>
    T GammP(T const a, T const x);

    /**
     *      Function to calculate incomplete Gamma Function Q = 1 - P
     **/
    template<typename T>
    T GammQ(T const a, T const x);

    /**
     *      Helper function to calculate incomplete Gamma Function
     **/
    template< typename T >
    T GSER(T & D_Gamser_In, T const a, T const x);

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
  }/// end namespace math
  
}}}
#endif