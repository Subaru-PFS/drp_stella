#ifndef __PFS_DRP_STELLA_MATH_SURFACEFITTING_H__
#define __PFS_DRP_STELLA_MATH_SURFACEFITTING_H__

#include <vector>
#include <iostream>
#include <cmath>
#include <cstdio>
//#include <cstring>
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "lsst/base.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/config.h"
#include "../utils/Utils.h"
#include "Math.h"
#include "ndarray.h"
#include "ndarray/eigen.h"

#include "LinearAlgebra3D.h"

//#define __DEBUG_CALC_TPS__

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
using namespace std;

namespace pfs { namespace drp { namespace stella {
  namespace math{
    namespace tps{
      /*
       * Thin Plate Spline, or TPS for short, is an interpolation method that finds a "minimally bended" smooth surface that passes through all given 
       * points. TPS of 3 control points is a plane, more than 3 is generally a curved surface and less than 3 is undefined.
       * 
       * The name "Thin Plate" comes from the fact that a TPS more or less simulates how a thin metal plate would behave if it was forced through the 
       * same control points.
       * 
       * Thin plate splines are particularily popular in representing shape transformations, for example, image morphing or shape detection. Consider 
       * two equally sized sets of 2D-points, A being the vertices of the original shape and B of the target shape. Let zi=Bix - Aix. Then fit a TPS 
       * over points (aix, aiy, zi) to get interpolation function for translation of points in x direction. Repeat the same for y.
       * 
       * In some cases, e.g. when the control point coordinates are noisy, you may want to relax the interpolation requirements slightly so that the 
       * resulting surface doesn't have to go exactly exactly through the control points. This is called regularization and is controlled by 
       * regularization parameter λ. If λ is zero, interpolation is exact and if it's very large, the resulting TPS surface is reduced to a least 
       * squares fitted plane ("bending energy" of a plane is 0). In our example, the regularization parameter is also made scale invariant with an 
       * extra parameter α.
      */

      /* Solve a linear equation system a*x=b using inplace LU decomposition.
       * 
       * Stores x in 'b' and overwrites 'a' (with a pivotted LUD).
       *
       * Matrix 'b' may have any (>0) number of columns but
       * must contain as many rows as 'a'.
       *
       * Possible return values:
       *  0=success
       *  1=singular matrix
       *  2=a.rows != b.rows
       */
      template <typename T> 
      int LU_Solve( boost::numeric::ublas::matrix<T> & a,
                    boost::numeric::ublas::matrix<T> & b );
      
      /*
       *  Solves a linear system A x X = B using Gauss elimination,
       *  given by Boost uBlas matrices 'a' and 'b'.
       *
       *  If the elimination succeeds, the function returns true,
       *  A is inverted and B contains values for X.
       *
       *  In case A is singular (linear system is unsolvable), the
       *  function returns false and leaves A and B in scrambled state.
       *
       *  TODO: make further use of uBlas views instead of direct
       *  element access (for better optimizations)
       */
      template <class T> 
      bool gauss_solve(boost::numeric::ublas::matrix<T> & a,
                       boost::numeric::ublas::matrix<T> & b );
      
      template < typename T >
      T tps_base_func(T r);
      
      /*
       *  Calculate Thin Plate Spline (TPS) weights from
       *  control points and build a new height grid by
       *  interpolating with them.
       *
      template< typename T >
      ndarray::Array<T, 2, 1> calc_tps(std::vector<T> const& xVec_In,
                                              std::vector<T> const& yVec_In,
                                              std::vector<T> const& zVec_In,
                                              int nRows,
                                              int nCols,
                                              double regularization = 0.0);
      */
      
    }
    
    template < typename T >
    double fitPointTPS(std::vector< Vec > const& controlPoints, 
                       boost::numeric::ublas::matrix<double> const& mtxV, 
                       T const xPositionFit, 
                       T const yPositionFit);
    
    template < typename T >
    double fitPointTPSEigen(ndarray::Array< const float, 1, 1 > const& controlPointsX,
                            ndarray::Array< const float, 1, 1 > const& controlPointsY,
                            ndarray::Array< const T, 1, 1 > const& controlPointsZ,
                            ndarray::Array< double, 1, 1 > const& mtxV, 
                            float const xPositionFit, 
                            float const yPositionFit);
    
    template< typename T >
    ndarray::Array< T, 2, 1 > interpolateThinPlateSpline( ndarray::Array< const float, 1, 1 > const& xArr,
                                                          ndarray::Array< const float, 1, 1 > const& yArr,
                                                          ndarray::Array< const T, 1, 1 > const& zArr,
                                                          ndarray::Array< const float, 1, 1 > const& xPositionsFit,
                                                          ndarray::Array< const float, 1, 1 > const& yPositionsFit,
                                                          bool const isXYPositionsGridPoints,
                                                          double const regularization = 0. );
    
    ndarray::Array< float, 2, 1 > interpolateThinPlateSpline( std::vector< float > const& xVec,
                                                              std::vector< float > const& yVec,
                                                              std::vector< float > const& zVec,
                                                              std::vector< float > const& xPositionsFitVec,
                                                              std::vector< float > const& yPositionsFitVec,
                                                              bool const isXYPositionsGridPoints,
                                                              double const regularization = 0. );
    
    template< typename T >
    ndarray::Array< T, 2, 1 > interpolateThinPlateSplineEigen( ndarray::Array< const float, 1, 1 > const& xArr,
                                                               ndarray::Array< const float, 1, 1 > const& yArr,
                                                               ndarray::Array< const T, 1, 1 > const& zArr,
                                                               ndarray::Array< const float, 1, 1 > const& xPositionsFit,
                                                               ndarray::Array< const float, 1, 1 > const& yPositionsFit,
                                                               bool const isXYPositionsGridPoints,
                                                               double const regularization );
    
  }
}}}

template< typename T >
std::ostream& operator<<(std::ostream& os, boost::numeric::ublas::matrix<T> const& obj);

#endif
