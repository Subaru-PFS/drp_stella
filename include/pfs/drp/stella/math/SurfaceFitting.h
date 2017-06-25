#ifndef __PFS_DRP_STELLA_MATH_SURFACEFITTING_H__
#define __PFS_DRP_STELLA_MATH_SURFACEFITTING_H__

#include <iostream>
#include <vector>

#include "boost/numeric/ublas/matrix.hpp"
#include <Eigen/LU>
#include <Eigen/QR>

#include "../Controls.h"
#include "lsst/pex/exceptions/Exception.h"
#include "ndarray.h"
#include "ndarray/eigen.h"
#include "ndarray/eigen_fwd.h"
#include "../utils/Utils.h"

/// constants
#define CONST_PI 3.141592653589793238462643383280    /* pi */

namespace pexExcept = lsst::pex::exceptions;
namespace pfsDRPStella = pfs::drp::stella;

using namespace std;

namespace pfs { namespace drp { namespace stella {
    namespace math{
      
        template< typename ValueT, typename CoordsT >
        class ThinPlateSplineBase{
            public: 
                ThinPlateSplineBase();
                ThinPlateSplineBase( ThinPlateSplineBase const& tpsb );
                virtual ~ThinPlateSplineBase(){}
              
                ValueT getRBFShapeParameter() const {
                    return _tpsControl.shapeParameter;
                }

                ndarray::Array< CoordsT, 1, 1 > getDataPointsX () const {
                    return _dataPointsX;
                }

                ndarray::Array< CoordsT, 1, 1 > getDataPointsY() const {
                    return _dataPointsY;
                }

                ndarray::Array< ValueT, 1, 1 > getDataPointsZ() const {
                    return _dataPointsZ;
                }

                ndarray::Array< ValueT, 1, 1 > getDataPointsWeight() const {
                    return _dataPointsWeight;
                }

                ndarray::Array< double, 1, 1 > getCoefficients() const {
                    return _coefficients;
                }

                ndarray::Array< CoordsT, 2, 1 > getKnots() const {
                    return _knots;
                }

                ValueT getRegularization() const {
                    return _tpsControl.regularization;
                }

                ValueT getBendingEnergy() const{
                    return _bendingEnergy;
                }

                ValueT getChiSquare() const{
                    return _chiSquare;
                }

                ValueT getRegularizationBase() const{
                    return _regularizationBase;
                }

                bool isWeightsSet() const{
                    return _isWeightsSet;
                }

                ndarray::Array< ValueT, 1, 1 > getZFit( bool deep = false) const;

                ThinPlateSplineBase< ValueT, CoordsT >& operator=( ThinPlateSplineBase< ValueT, CoordsT > const& tps );

                /* @brief Fit array for given [x, y] positions or grid
                 *  @param xPositionFit : x position of data point to be fit
                 *  @param yPositionFit : y position of data point to be fit
                 *  @param isXYPositionsGridPoints :  if isXYPositionsGridPoints: returned array will have shape(yPositionsFit.getShape()[0], xPositionsFit.getShape()[0])
                 *                                                          else: shape(xPositionsFit.getShape()[0], 1)
                 */
                ndarray::Array< ValueT, 2, 1 > fitArray( ndarray::Array< const CoordsT, 1, 1 > const& xPositionsFit,
                                                         ndarray::Array< const CoordsT, 1, 1 > const& yPositionsFit,
                                                         bool const isXYPositionsGridPoints); /// fit positions
                
                ndarray::Array< double, 2, 1 > getMatrix() const{
                    return _matrix; 
                }
                
                ndarray::Array< double, 1, 1 > getRHS() const{
                    return _rhs;
                }
                
                bool setCoefficients( ndarray::Array< const double, 1, 1 > const& coefficients );
                
                TPSControl getTPSControl(){
                    return _tpsControl;
                }
            
                TPSControl getTPSControl() const{
                    return _tpsControl;
                }
                
            protected:
                /** @brief fill lhs matrix for fit
                 */
                virtual void fillMatrix() = 0;

                /** @brief fill rhs vector for fit
                 */
                virtual void fillRHS() = 0;
                
                /**
                 * @brief Add regularization (_regularization * _regularizationBase^2) to diagonal elements of matrix
                 */                
                void addRegularizationToMatrix( );
                
                /**
                 * @brief Add weights (8 * Pi / _dataPointsWeight) to diagonal elements of matrix
                 */                
                void addWeightsToMatrix( );

                /** @brief Calculate coefficients for fit
                 * 
                 */
                virtual bool calculateCoefficients();
            
                /** @brief Fit one point at [xPositionFit, yPositionFit] and return result
                 *  @param xPositionFit : x position of data point to be fit
                 *  @param yPositionFit : y position of data point to be fit
                 */
                ValueT fitPoint(CoordsT const xPositionFit, 
                                CoordsT const yPositionFit);
                
                /** @brief base function for thin-plate-spline fitting Phi = r^2 * log ( r / _shapeParameter )
                 */
                ValueT tps_base_func( ValueT r );

                ndarray::Array< CoordsT, 1, 1 > _dataPointsX;
                ndarray::Array< CoordsT, 1, 1 > _dataPointsY;
                ndarray::Array< ValueT, 1, 1 > _dataPointsZ;
                ndarray::Array< ValueT, 1, 1 > _dataPointsWeight;
                ndarray::Array< CoordsT, 2, 1 > _knots;
                ndarray::Array< double, 1, 1 > _coefficients;
                ndarray::Array< ValueT, 1, 1 > _zFit;
                ndarray::Array< double, 2, 1 > _matrix;
                ndarray::Array< double, 1, 1 > _rhs;
                TPSControl _tpsControl;
                ValueT _bendingEnergy;
                ValueT _chiSquare;
                ValueT _regularizationBase;
                bool _isWeightsSet;
        };
      
        template < typename ValueT, typename CoordsT >
        class ThinPlateSpline: public ThinPlateSplineBase< ValueT, CoordsT >{
            public: 
                explicit ThinPlateSpline();

                /**
                 * @brief Create (regularized) ThinPlateSpline object and calculate coefficients of fit
                 * @param dataPointsX : x positions of input data to fit
                 * @param dataPointsY : y positions of input data to fit
                 * @param dataPointsZ : z values of input data to fit
                 * @param tpsControl : ThinPlateSplineControl, controls parameters like regularization
                 * @return ThinPlateSpline object
                 */
                explicit ThinPlateSpline( ndarray::Array< const CoordsT, 1, 1 > const& dataPointsX,
                                          ndarray::Array< const CoordsT, 1, 1 > const& dataPointsY,
                                          ndarray::Array< const ValueT, 1, 1 > const& dataPointsZ,
                                          TPSControl const& tpsControl );

                /**
                 * @brief Create weighted ThinPlateSpline object and calculate coefficients of fit
                 * @param dataPointsX : x positions of input data to fit
                 * @param dataPointsY : y positions of input data to fit
                 * @param dataPointsZ : z values of input data to fit
                 * @param dataPointsWeights : >= 0. If 0 the fit is forced through the input data
                 * @param tpsControl : ThinPlateSplineControl, controls parameters like regularization
                 * @return ThinPlateSpline object
                 */
                explicit ThinPlateSpline( ndarray::Array< const CoordsT, 1, 1 > const& dataPointsX,
                                          ndarray::Array< const CoordsT, 1, 1 > const& dataPointsY,
                                          ndarray::Array< const ValueT, 1, 1 > const& dataPointsZ,
                                          ndarray::Array< const ValueT, 1, 1 > const& dataPointsWeights,
                                          TPSControl const& tpsControl );

                ThinPlateSpline( ThinPlateSpline const& tps);

                virtual ~ThinPlateSpline(){}

            protected:
                /** @brief fill lhs matrix for fit
                 */
                virtual void fillMatrix();

                /** @brief fill rhs vector for fit
                 */
                virtual void fillRHS();

            private:
        };

        template < typename ValueT, typename CoordsT >
        class ThinPlateSplineChiSquare: public ThinPlateSplineBase< ValueT, CoordsT >{
            public: 
                explicit ThinPlateSplineChiSquare();

                /**
                 * @brief Create ThinPlateSpline object and calculate coefficients of fit by minimizing Chi square
                 * @param dataPointsX : x positions of input data to fit
                 * @param dataPointsY : y positions of input data to fit
                 * @param dataPointsZ : z values of input data to fit
                 * @param knotsX : x values on which to fit the thin-plate spline
                 * @param knotsY : y values on which to fit the thin-plate spline
                 * @param isXYPositionsGridPoints : are knotsX and knotsY grid points? 
                 *                                  If Yes, there will be knotsX.getShape()[0] X knotsY.getShape()[0] knots, 
                 *                                  otherwise knotsX.getShape()[0] = knotsY.getShape()[0] knots
                 * @param tpsControl : ThinPlateSplineControl, controls parameters like regularization
                 * @return ThinPlateSpline object
                 */
                explicit ThinPlateSplineChiSquare( ndarray::Array< const CoordsT, 1, 1 > const& dataPointsX,
                                                   ndarray::Array< const CoordsT, 1, 1 > const& dataPointsY,
                                                   ndarray::Array< const ValueT, 1, 1 > const& dataPointsZ,
                                                   ndarray::Array< const CoordsT, 1, 1 > const& knotsX,
                                                   ndarray::Array< const CoordsT, 1, 1 > const& knotsY,
                                                   bool const isXYPositionsGridPoints,
                                                   TPSControl const& tpsControl );

                ThinPlateSplineChiSquare( ThinPlateSplineChiSquare const& tps);

                virtual ~ThinPlateSplineChiSquare(){}
            
            protected:

            private:
                /** @brief create vector of gridPointsXY of size _gridPointsX.getShape()[0] * _gridPointsY.getShape()[0] X 2
                 */
                void createGridPointsXY( ndarray::Array< const CoordsT, 1, 1 > const& gridPointsX,
                                         ndarray::Array< const CoordsT, 1, 1 > const& gridPointsY );

                /** @brief fill lhs matrix for (non-regularized and non-weighted) fit
                 */
                virtual void fillMatrix();

                /** @brief fill rhs vector for fit
                 */
                virtual void fillRHS();

                /** @brief Calculate coefficients of thin-plate spline
                 * 
                 */
                virtual bool calculateCoefficients();
                
        };  
    }
}}}

#endif
