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

//#define __TPS_BASE_FUNC_MULTIQUADRIC__
//#define __DEBUG_CALC_TPS__
//#define __DEBUG_TPS__
//#define __DEBUG_TPS_FITPOINT__

namespace afwGeom = lsst::afw::geom;
namespace afwImage = lsst::afw::image;
namespace pfsDRPStella = pfs::drp::stella;
using namespace std;

namespace pfs { namespace drp { namespace stella {
  namespace math{
    template < typename ValueT, typename CoordsT >
    class ThinPlateSpline{
        public: 
            explicit ThinPlateSpline();
                        
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
                                      double const regularization = 0.,
                                      ValueT const radiusNormalizationFactor = 1. );
            
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
                                      ndarray::Array< const ValueT, 1, 1 > const& controlPointsWeights,
                                      ValueT const radiusNormalizationFactor = 1. );

            ThinPlateSpline( ThinPlateSpline const& tps);
            
            virtual ~ThinPlateSpline(){}
            
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
            
            ValueT getRadiusNormalizationFactor() const {
                return _radiusNormalizationFactor;
            }
            
            ndarray::Array< CoordsT, 1, 1 > getControlPointsX () const {
                return _controlPointsX;
            }

            ndarray::Array< CoordsT, 1, 1 > getControlPointsY() const {
                return _controlPointsY;
            }

            ndarray::Array< ValueT, 1, 1 > getControlPointsZ() const {
                return _controlPointsZ;
            }

            ndarray::Array< ValueT, 1, 1 > getControlPointsWeight() const {
                return _controlPointsWeight;
            }

            ndarray::Array< double, 1, 1 > getCoefficients() const {
                return _coefficients;
            }
            
            ValueT getRegularization() const {
                return _regularization;
            }
            
            ValueT getBendingEnergy() const{
                return _bendingEnergy;
            }
            
            bool isWeightsSet() const{
                return _isWeightsSet;
            }
            
            ThinPlateSpline< ValueT, CoordsT >& operator=(ThinPlateSpline< ValueT, CoordsT > const& tps);

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
            
            /** @brief Calculate coefficients for regularized fit without weights
             * 
             */
            bool calculateCoefficients();
            
            ndarray::Array< CoordsT, 1, 1 > _controlPointsX;
            ndarray::Array< CoordsT, 1, 1 > _controlPointsY;
            ndarray::Array< ValueT, 1, 1 > _controlPointsZ;
            ndarray::Array< ValueT, 1, 1 > _controlPointsWeight;
            ndarray::Array< double, 1, 1 > _coefficients;
            double _regularization;
            ValueT _radiusNormalizationFactor;
            ValueT _bendingEnergy;
            bool _isWeightsSet;
    };  

    template < typename ValueT, typename CoordsT >
    class ThinPlateSplineChiSquare{
        public: 
            explicit ThinPlateSplineChiSquare();
            
            /**
             * @brief Create ThinPlateSpline object and calculate coefficients of fit by minimizing Chi square
             * @param controlPointsX : x positions of input data to fit
             * @param controlPointsY : y positions of input data to fit
             * @param controlPointsZ : z values of input data to fit
             * @param gridPointsX : x values of xy-grid on which to fit the thin-plate spline
             * @param gridPointsY : y values of xy-grid on which to fit the thin-plate spline
             * @return ThinPlateSpline object
             */
            explicit ThinPlateSplineChiSquare( ndarray::Array< const CoordsT, 1, 1 > const& controlPointsX,
                                               ndarray::Array< const CoordsT, 1, 1 > const& controlPointsY,
                                               ndarray::Array< const ValueT, 1, 1 > const& controlPointsZ,
                                               ndarray::Array< const CoordsT, 1, 1 > const& fitPointsX,
                                               ndarray::Array< const CoordsT, 1, 1 > const& fitPointsY,
                                               bool const isXYPositionsGridPoints,
                                               ValueT const regularization = 0.,
                                               ValueT const radiusNormalizationFactor = 1. );
            
            ThinPlateSplineChiSquare( ThinPlateSplineChiSquare const& tps);
            
            virtual ~ThinPlateSplineChiSquare(){}
            
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
            
            ndarray::Array< const CoordsT, 1, 1 > getControlPointsX () const {
                return _controlPointsX;
            }

            ndarray::Array< const CoordsT, 1, 1 > getControlPointsY() const {
                return _controlPointsY;
            }

            ndarray::Array< const ValueT, 1, 1 > getControlPointsZ() const {
                return _controlPointsZ;
            }

            ndarray::Array< CoordsT, 2, 1 > getFitPointsXY() const {
                return _fitPointsXY;
            }

            ndarray::Array< double, 1, 1 > getCoefficients() const {
                return _coefficients;
            }
            
            ValueT getRegularization() const {
                return _regularization;
            }
            
            ValueT getRadiusNormalizationFactor() const {
                return _radiusNormalizationFactor;
            }
            
            ValueT getBendingEnergy() const{
                return _bendingEnergy;
            }
            
        protected:
                
        private:
            /** @brief create vector of gridPointsXY of size _gridPointsX.getShape()[0] * _gridPointsY.getShape()[0] X 2
             */
            void createGridPointsXY( ndarray::Array< const CoordsT, 1, 1 > const& gridPointsX,
                                     ndarray::Array< const CoordsT, 1, 1 > const& gridPointsY );
            
            /** @brief fill lhs matrix for (regularized) fit
             */
            ndarray::Array< double, 2, 1 > fillMatrix();
            
            /** @brief base function for thin-plate-spline fitting Phi = r^2 * log (r / normFactor)
             */
            ValueT tps_base_func(ValueT const r );
            
            /** @brief Calculate coefficients of thin-plate spline
             * 
             */
            bool calculateCoefficients();
            
            ndarray::Array< const CoordsT, 1, 1 > _controlPointsX;
            ndarray::Array< const CoordsT, 1, 1 > _controlPointsY;
            ndarray::Array< const ValueT, 1, 1 > _controlPointsZ;
            ndarray::Array< CoordsT, 2, 1 > _fitPointsXY;
            ndarray::Array< double, 1, 1 > _coefficients;
            ValueT _regularization;
            ValueT _radiusNormalizationFactor;
            ValueT _bendingEnergy;
    };  
    
  }
}}}

#endif
