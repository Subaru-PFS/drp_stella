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
namespace pfsDRPStella = pfs::drp::stella;
using namespace std;

namespace pfs { namespace drp { namespace stella {
  namespace math{
    template < typename ValueT, typename CoordsT >
    class ThinPlateSpline{
        public: 
            
            /**
             * @brief Create (regularized) ThinPlateSpline object and calculate coefficients of fit
             * @param controlPointsX : x positions of input data to fit
             * @param controlPointsY : y positions of input data to fit
             * @param controlPointsZ : z values of input data to fit
             * @param regularization : >= 0. If 0 the fit is forced through the input data
             * @return ThinPlateSpline object
             */
            explicit ThinPlateSpline( ndarray::Array< const CoordsT, 1, 1 > const& controlPointsX,
                                      ndarray::Array< const CoordsT, 1, 1 > const& controlPointsY,
                                      ndarray::Array< const ValueT, 1, 1 > const& controlPointsZ,
                                      double const regularization = 0.);
            
            /**
             * @brief Create weighted ThinPlateSpline object and calculate coefficients of fit
             * @param controlPointsX : x positions of input data to fit
             * @param controlPointsY : y positions of input data to fit
             * @param controlPointsZ : z values of input data to fit
             * @param controlPointsWeights : >= 0. If 0 the fit is forced through the input data
             * @return ThinPlateSpline object
             */
            explicit ThinPlateSpline( ndarray::Array< const CoordsT, 1, 1 > const& controlPointsX,
                                      ndarray::Array< const CoordsT, 1, 1 > const& controlPointsY,
                                      ndarray::Array< const ValueT, 1, 1 > const& controlPointsZ,
                                      ndarray::Array< const ValueT, 1, 1 > const& controlPointsWeights);
            
            virtual ~ThinPlateSpline(){}
            
            /** @brief Calculate coefficients for regularized fit without weights
             * 
             */
            bool calculateCoefficients();
            
            /** @brief Fit one point at [xPositionFit, yPositionFit] and return result
             *  @param xPositionFit : x position of data point to be fit
             *  @param yPositionFit : y position of data point to be fit
             */
            ValueT fitPoint(CoordsT const xPositionFit, 
                            CoordsT const yPositionFit);

            /* @brief Fit array for given [x, y] positions or grid
             *  @param xPositionFit : x position of data point to be fit
             *  @param yPositionFit : y position of data point to be fit
             *  @param isXYPositionsGridPoints :  if isXYPositionsGridPoints: returned array will have shape(yPositionsFit.getShape()[0], xPositionsFit.getShape()[0])
             *                                                          else: shape(xPositionsFit.getShape()[0], 1)
             */
            ndarray::Array< ValueT, 2, 1 > fitArray( ndarray::Array< const CoordsT, 1, 1 > const& xPositionsFit,
                                                     ndarray::Array< const CoordsT, 1, 1 > const& yPositionsFit,
                                                     bool const isXYPositionsGridPoints); /// fit positions
            
        protected:
                
        private:
            /** @brief fill lhs matrix for (regularized) fit
             */
            ndarray::Array< double, 2, 1 > fillRegularizedMatrix();
            
            /** @brief fill lhs matrix for weighted fit
             */
            ndarray::Array< double, 2, 1 > fillWeightedMatrix();
            
            /** @brief base function for thin-plate-spline fitting Phi = r^2 * log r
             */
            ValueT tps_base_func(ValueT r);
            
            ndarray::Array< const CoordsT, 1, 1 > _controlPointsX;
            ndarray::Array< const CoordsT, 1, 1 > _controlPointsY;
            ndarray::Array< const ValueT, 1, 1 > _controlPointsZ;
            ndarray::Array< const ValueT, 1, 1 > _controlPointsWeight;
            ndarray::Array< double, 1, 1 > _coefficients;
            const double _regularization;
            const bool _isWeightsSet;
    };  
    
  }
}}}

#endif
