#include "pfs/drp/stella/math/SurfaceFitting.h"

using namespace boost::numeric::ublas;

namespace pfs{ namespace drp{ namespace stella{ 
  namespace math{

    template< typename ValueT, typename CoordsT >
    ThinPlateSpline< ValueT, CoordsT >::ThinPlateSpline( ):
      _controlPointsX(),
      _controlPointsY(),
      _controlPointsZ(),
      _controlPointsWeight(),
      _coefficients(),
      _regularization(0.),
      _radiusNormalizationFactor(1.),
      _bendingEnergy(0.),
      _isWeightsSet(false)
    {
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline() finished" << endl;
      #endif
    }
          
    template< typename ValueT, typename CoordsT >
    ThinPlateSpline< ValueT, CoordsT >::ThinPlateSpline( ndarray::Array< const CoordsT, 1, 1 > const& controlPointsX,
                                                         ndarray::Array< const CoordsT, 1, 1 > const& controlPointsY,
                                                         ndarray::Array< const ValueT, 1, 1 > const& controlPointsZ,
                                                         double const regularization,
                                                         ValueT const radiusNormalizationFactor ):
      _controlPointsX(ndarray::copy(controlPointsX)),
      _controlPointsY(ndarray::copy(controlPointsY)),
      _controlPointsZ(ndarray::copy(controlPointsZ)),
      _controlPointsWeight(pfsDRPStella::utils::get1DndArray(ValueT(controlPointsZ.getShape()[0]))),
      _coefficients(),
      _regularization(regularization),
      _radiusNormalizationFactor(radiusNormalizationFactor),
      _bendingEnergy(0.),
      _isWeightsSet(false)
    {
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline(controlPointsX, controlPointsY, controlPointsZ, regularization, radiusNormalizationFactor) started" << endl;
      #endif
      /// Check input data:
      if (controlPointsX.getShape()[0] != controlPointsY.getShape()[0]){
        std::string message("ThinPlateSpline::ThinPlateSpline: ERROR: controlPointsX.getShape()[0] = ");
        message += std::to_string(controlPointsX.getShape()[0]) + " != controlPointsY.getShape()[0] = ";
        message += std::to_string(controlPointsY.getShape()[0]);
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      if (controlPointsX.getShape()[0] != controlPointsZ.getShape()[0]){
        std::string message("ThinPlateSpline::ThinPlateSpline: ERROR: controlPointsX.getShape()[0] = ");
        message += std::to_string(controlPointsX.getShape()[0]) + " != controlPointsZ.getShape()[0] = ";
        message += std::to_string(controlPointsZ.getShape()[0]);
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      
      _coefficients = ndarray::allocate(controlPointsZ.getShape()[0] + 3);
      if (!calculateCoefficients()){
        std::string message("ThinPlateSpline::ThinPlateSpline: ERROR: calculateCoefficients returned FALSE");
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline(controlPointsX, controlPointsY, controlPointsZ, regularization, radiusNormalizationFactor) finished" << endl;
      #endif
    }
            
    template< typename ValueT, typename CoordsT >
    ThinPlateSpline< ValueT, CoordsT >::ThinPlateSpline( ndarray::Array< const CoordsT, 1, 1 > const& controlPointsX,
                                                         ndarray::Array< const CoordsT, 1, 1 > const& controlPointsY,
                                                         ndarray::Array< const ValueT, 1, 1 > const& controlPointsZ,
                                                         ndarray::Array< const ValueT, 1, 1 > const& controlPointsWeights,
                                                         ValueT const radiusNormalizationFactor):
      _controlPointsX(ndarray::copy(controlPointsX)),
      _controlPointsY(ndarray::copy(controlPointsY)),
      _controlPointsZ(ndarray::copy(controlPointsZ)),
      _controlPointsWeight(ndarray::copy(controlPointsWeights)),
      _regularization(0.),
      _radiusNormalizationFactor(radiusNormalizationFactor),
      _bendingEnergy(0.),
      _isWeightsSet(true)
    {
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline(controlPointsX, controlPointsY, controlPointsZ, controlPointsWeights, radiusNormalizationFactor) started" << endl;
      #endif
      if (controlPointsX.getShape()[0] != controlPointsY.getShape()[0]){
        std::string message("ThinPlateSpline::ThinPlateSpline: ERROR: controlPointsX.getShape()[0] = ");
        message += std::to_string(controlPointsX.getShape()[0]) + " != controlPointsY.getShape()[0] = ";
        message += std::to_string(controlPointsY.getShape()[0]);
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      if (controlPointsX.getShape()[0] != controlPointsZ.getShape()[0]){
        std::string message("ThinPlateSpline::ThinPlateSpline: ERROR: controlPointsX.getShape()[0] = ");
        message += std::to_string(controlPointsX.getShape()[0]) + " != controlPointsZ.getShape()[0] = ";
        message += std::to_string(controlPointsZ.getShape()[0]);
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      if (controlPointsX.getShape()[0] != controlPointsWeights.getShape()[0]){
        std::string message("ThinPlateSpline::ThinPlateSpline: ERROR: controlPointsX.getShape()[0] = ");
        message += std::to_string(controlPointsX.getShape()[0]) + " != controlPointsWeights.getShape()[0] = ";
        message += std::to_string(controlPointsWeights.getShape()[0]);
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
      
      _coefficients = ndarray::allocate( controlPointsZ.getShape()[ 0 ] + 3 );
      if ( !calculateCoefficients() ){
        std::string message( "ThinPlateSpline::ThinPlateSpline: ERROR: calculateCoefficients returned FALSE" );
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline(controlPointsX, controlPointsY, controlPointsZ, controlPointsWeights, radiusNormalizationFactor) finished" << endl;
      #endif
    }
    
    template< typename ValueT, typename CoordsT >
    ThinPlateSpline< ValueT, CoordsT >::ThinPlateSpline( ThinPlateSpline< ValueT, CoordsT > const& tps ):
            _controlPointsX( tps.getControlPointsX() ),
            _controlPointsY( tps.getControlPointsY() ),
            _controlPointsZ( tps.getControlPointsZ() ),
            _controlPointsWeight( tps.getControlPointsWeight() ),
            _regularization( tps.getRegularization() ),
            _radiusNormalizationFactor( tps.getRadiusNormalizationFactor() ),
            _bendingEnergy( tps.getBendingEnergy() ),
            _isWeightsSet( tps.isWeightsSet() )
    { 
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline( tps ) started" << endl;
      #endif
      if ( tps.getCoefficients().getShape()[ 0 ] == 0 ){
        std::string message( "ThinPlateSpline::ThinPlateSpline( tps ): ERROR: tps.getCoefficients().getShape()[ 0 ] == 0" );
        std::cout << message << std::endl;
        cout << "ThinPlateSpline::ThinPlateSpine( tps ): _controlPointsX.getShape()[ 0 ] = " << _controlPointsX.getShape()[ 0 ] << endl;
        cout << "ThinPlateSpline::ThinPlateSpine( tps ): _controlPointsY.getShape()[ 0 ] = " << _controlPointsY.getShape()[ 0 ] << endl;
        cout << "ThinPlateSpline::ThinPlateSpine( tps ): _controlPointsZ.getShape()[ 0 ] = " << _controlPointsZ.getShape()[ 0 ] << endl;
        cout << "ThinPlateSpline::ThinPlateSpine( tps ): _controlPointsWeight.getShape()[ 0 ] = " << _controlPointsWeight.getShape()[ 0 ] << endl;
        cout << "ThinPlateSpline::ThinPlateSpine( tps ): _regularization = " << _regularization << endl;
        cout << "ThinPlateSpline::ThinPlateSpine( tps ): _radiusNormalizationFactor = " << _radiusNormalizationFactor << endl;
        cout << "ThinPlateSpline::ThinPlateSpine( tps ): _bendingEnergy = " << _bendingEnergy << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      _coefficients = ndarray::allocate( tps.getCoefficients().getShape() );
      _coefficients.deep() = tps.getCoefficients();
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline(tps): _coefficients = " << _coefficients << endl;
        cout << "ThinPlateSpline(tps) finished" << endl;
      #endif
    }

    template< typename ValueT, typename CoordsT >
    ValueT ThinPlateSpline< ValueT, CoordsT >::tps_base_func(ValueT r){
      if ( r == 0.0 )
        return 0.0;
      else
        return r * r * log( r / _radiusNormalizationFactor );
    }
            
    template< typename ValueT, typename CoordsT >
    bool ThinPlateSpline< ValueT, CoordsT >::calculateCoefficients(){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::calculateCoefficients() started" << endl;
      #endif
      #ifdef __DEBUG_CALC_TPS__
        cout << "interpolateThinPlateSpline: _controlPointsX = (" << _controlPointsX.getShape()[0] << ") = " << _controlPointsX << endl;
        cout << "interpolateThinPlateSpline: _controlPointsY = (" << _controlPointsY.getShape()[0] << ") = " << _controlPointsY << endl;
        cout << "interpolateThinPlateSpline: _controlPointsZ = (" << _controlPointsZ.getShape()[0] << ") = " << _controlPointsZ << endl;
      #endif

      // You We need at least 3 points to define a plane
      if ( _controlPointsX.getShape()[0] < 3 ){
        string message("interpolateThinPlateSpline: ERROR: _controlPointsX.getShape()[0] = ");
        message += to_string(_controlPointsX.getShape()[0]) + " < 3";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }

      unsigned p = _controlPointsX.getShape()[0];

      // Allocate the matrix and vector
      ndarray::Array< double, 2, 1 > mtx_l = ndarray::allocate( p + 3, p + 3 );
      ndarray::Array< double, 1, 1 > mtx_v = ndarray::allocate( p + 3 );
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "memory for mtx_l, mtx_v allocated" << std::endl;
      #endif

      if (_isWeightsSet)
        mtx_l.deep() = fillWeightedMatrix();
      else
        mtx_l.deep() = fillRegularizedMatrix();

      // Fill the right hand vector V
      for ( unsigned i = 0; i < p; ++i )
        mtx_v[ i ] = _controlPointsZ[ i ];
      mtx_v[ p ] = mtx_v[p+1] = mtx_v[ p + 2 ] = 0.0;
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "interpolateThinPlateSpline: mtx_v = " << mtx_v << std::endl;
      #endif

      // Solve the linear system "inplace"
      _coefficients.asEigen() = mtx_l.asEigen().colPivHouseholderQr().solve(mtx_v.asEigen());
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "interpolateThinPlateSpline: after colPivHouseholderQr: _coefficients = " << _coefficients << std::endl;
      #endif

      // Calculate bending energy
      ndarray::Array< double, 1, 1 > mtxv_dot_mtxl = ndarray::allocate(mtx_l.getShape()[1]);
      for (int i = 0; i < mtx_l.getShape()[ 1 ]; ++i){
        mtxv_dot_mtxl[ i ] = 0.;
        for (int j = 0; j < mtx_l.getShape()[ 0 ]; ++j)
          mtxv_dot_mtxl[ i ] += mtx_v[ j ] * mtx_l[ j ][ i ];
      }  
      double bendingEnergy = 0.;
      for (int i = 0; i < mtx_l.getShape()[ 1 ]; ++i)
        bendingEnergy += mtx_v[ i ] * mtxv_dot_mtxl[ i ];

      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::calculateCoefficients() finished" << endl;
      #endif
      return true;
    }
    
    template< typename ValueT, typename CoordsT >
    ndarray::Array< double, 2, 1 > ThinPlateSpline< ValueT, CoordsT >::fillRegularizedMatrix(){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::fillRegularizedMatrix() started" << endl;
      #endif
      unsigned p = _controlPointsX.getShape()[0];

      // Allocate the matrix and vector
      ndarray::Array< double, 2, 1 > mtx_l = ndarray::allocate(p+3, p+3);
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "memory for mtx_l, mtx_v allocated" << std::endl;
      #endif

      // Fill K (p x p, upper left of L) and calculate
      // mean edge length from control points
      //
      // K is symmetrical so we really have to
      // calculate only about half of the coefficients.
      double a = 0.0;
      ndarray::Array< double, 1, 1 > pt_i = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_j = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_diff = ndarray::allocate(2);
      double elen;
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "memory for pt_i, pt_j, pt_diff allocated" << std::endl;
      #endif
      for ( unsigned i=0; i<p; ++i ){
        for ( unsigned j=i+1; j<p; ++j ){
          pt_i[0] = _controlPointsX[i];
          pt_i[1] = _controlPointsY[i];
          pt_j[0] = _controlPointsX[j];
          pt_j[1] = _controlPointsY[j];
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "i = " << i << ", j = " << j << ": pt_i set to " << pt_i << ", pt_j = " << pt_j << std::endl;
          #endif
          pt_diff.deep() = pt_i - pt_j;
          pt_diff.asEigen() = pt_diff.asEigen().array() * pt_diff.asEigen().array();
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "i = " << i << ", j = " << j << ": pt_diff set to " << pt_diff << std::endl;
          #endif
          elen = sqrt(pt_diff.asEigen().sum());
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "i = " << i << ", j = " << j << ": elen set to " << elen << std::endl;
          #endif
          mtx_l[i][j] = mtx_l[j][i] = ThinPlateSpline::tps_base_func(elen);
          a += elen * 2; // same for upper & lower tri
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "i = " << i << ", j = " << j << ": mtx_l[i][j] set to " << mtx_l[i][j] << ", a set to " << a << std::endl;
          #endif
        }
      }
      a /= (double)(p*p);
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "interpolateThinPlateSpline: a = " << a << std::endl;
      #endif

      // Fill the rest of L
      for ( unsigned i = 0; i < p; ++i ){
        // diagonal: reqularization parameters (lambda * a^2)
        mtx_l[i][i] = _regularization * (a*a);

        // P (p x 3, upper right)
        // P transposed (3 x p, bottom left)
        mtx_l[i][p+0] = mtx_l[p+0][i] = 1.0;
        mtx_l[i][p+1] = mtx_l[p+1][i] = _controlPointsX[i];
        mtx_l[i][p+2] = mtx_l[p+2][i] = _controlPointsY[i];
      }
      // O (3 x 3, lower right)
      for ( unsigned i=p; i < p+3; ++i ){
        for ( unsigned j=p; j < p+3; ++j ){
          mtx_l[i][j] = 0.0;
        }
      }
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "interpolateThinPlateSpline: mtx_l = " << mtx_l << std::endl;
      #endif
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::fillRegularizedMatrix() finished" << endl;
      #endif
      return mtx_l;
    }
    
    template< typename ValueT, typename CoordsT >
    ndarray::Array< double, 2, 1 > ThinPlateSpline< ValueT, CoordsT >::fillWeightedMatrix(){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::fillWeightedMatrix() started" << endl;
      #endif
      unsigned p = _controlPointsX.getShape()[0];
      ndarray::Array<double, 2, 1> cArr = ndarray::allocate(p, p);
      ndarray::Array<double, 2, 1> fArr = ndarray::allocate(p, 3);
      ndarray::Array<double, 2, 1> fTArr = ndarray::allocate(3, p);
      double r_i;
      double phi;
      cout << "8. * CONST_PI = " << 8. * CONST_PI << endl;
      for (int i = 0; i < p; ++i){
        for (int j = 0; j < i; ++j){
          r_i = sqrt(pow(_controlPointsX[j] - _controlPointsX[i], 2) + pow(_controlPointsY[j] - _controlPointsY[i], 2));
          phi = tps_base_func(r_i);
          cArr[i][j] = cArr[j][i] = phi;
        }
        if ( _controlPointsWeight[i] < 0.000000000000001)
          cArr[i][i] = 0.;
        else
          cArr[i][i] = 8. * CONST_PI / _controlPointsWeight[i];
        fArr[i][0] = fTArr[0][i] = 1;
        fArr[i][1] = fTArr[1][i] = _controlPointsX[i];
        fArr[i][2] = fTArr[2][i] = _controlPointsY[i];
      }
      #ifdef __DEBUG_CALC_TPS__
        cout << "interpolateThinPlateSpline: cArr = " << cArr << endl;
        cout << "interpolateThinPlateSpline: cArr.getShape() = " << cArr.getShape() << endl;
        cout << "interpolateThinPlateSpline: p = " << p << endl;
      #endif
      ndarray::Array<double, 2, 1> mtx_l = ndarray::allocate(p + 3, p + 3);
      #ifdef __DEBUG_CALC_TPS__
        cout << "interpolateThinPlateSpline: mtx_l[ndarray::view(0, p)(0, p)].getShape() = " << mtx_l[ndarray::view(0, p)(0, p)].getShape() << endl;
      #endif
      mtx_l.deep() = 0.;
      mtx_l[ndarray::view(0, p)(0, p)].deep() = cArr;
      #ifdef __DEBUG_CALC_TPS__
        cout << "interpolateThinPlateSpline: mtx_l[ndarray::view(0, p)(0, p)] = " << mtx_l[ndarray::view(0, p)(0, p)] << endl;
      #endif
      mtx_l[ndarray::view(p, p + 3)(0, p)] = fTArr;
      #ifdef __DEBUG_CALC_TPS__
        cout << "interpolateThinPlateSpline: mtx_l[ndarray::view(p, p+3)(0, p)] = " << mtx_l[ndarray::view(p, p+3)(0, p)] << endl;
      #endif
      mtx_l[ndarray::view(0, p)(p, p + 3)] = fArr;
      #ifdef __DEBUG_CALC_TPS__
        cout << "interpolateThinPlateSpline: mtx_l[ndarray::view(0, p)(p, p+3)] = " << mtx_l[ndarray::view(0, p)(p, p+3)] << endl;
      #endif
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::fillWeightedMatrix() finished" << endl;
      #endif
      return mtx_l;
    }    
            
    template< typename ValueT, typename CoordsT >
    ValueT ThinPlateSpline< ValueT, CoordsT >::fitPoint(CoordsT const xPositionFit, 
                                                        CoordsT const yPositionFit){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::fitPoint(xPositionsFit, yPositionsFit) started" << endl;
      #endif
      #ifdef __DEBUG_TPS_FITPOINT__
        std::cout << "ThinPlateSpline::fitPoint: x = " << xPositionFit << ", y = " << yPositionFit << std::endl;
        std::cout << "ThinPlateSpline::fitPoint: _coefficients = " << _coefficients << std::endl;
      #endif
      unsigned p = _controlPointsX.getShape()[0];
      double h = _coefficients[p] + (_coefficients[p+1] * xPositionFit) + (_coefficients[p+2] * yPositionFit);
      ndarray::Array<CoordsT, 1, 1> pt_i = ndarray::allocate(2);
      ndarray::Array<CoordsT, 1, 1> pt_cur = ndarray::allocate(2);
      pt_cur[0] = xPositionFit;
      pt_cur[1] = yPositionFit;
      #ifdef __DEBUG_TPS_FITPOINT__
        cout << "ThinPlateSpline::fitPoint: pt_cur = [ " << pt_cur[0] << ", " << pt_cur[1] << "]" << endl;
      #endif
      ndarray::Array<CoordsT, 1, 1> pt_diff = ndarray::allocate(2);
      ValueT len;
      for ( unsigned i = 0; i < p; ++i ){
        pt_i[0] = _controlPointsX[i];
        pt_i[1] = _controlPointsY[i];
        #ifdef __DEBUG_TPS_FITPOINT__
          cout << "ThinPlateSpline::fitPoint: i=" << i << ": pt_i = [ " << pt_i[0] << ", " << pt_i[1] << "]" << endl;
        #endif
        pt_diff.deep() = pt_i - pt_cur;
        pt_diff.deep() = pt_diff * pt_diff;
        #ifdef __DEBUG_TPS_FITPOINT__
          cout << "ThinPlateSpline::fitPoint: i=" << i << ": pt_diff = [ " << pt_diff[0] << ", " << pt_diff[1] << "]" << endl;
        #endif
        len = sqrt(pt_diff.asEigen().sum());
        #ifdef __DEBUG_TPS_FITPOINT__
          cout << "ThinPlateSpline::fitPoint: i=" << i << ": len = " << len << endl;
        #endif
        h += _coefficients[i] * tps_base_func( len );
        #ifdef __DEBUG_TPS_FITPOINT__
          cout << "ThinPlateSpline::fitPoint: i=" << i << ": h = " << h << endl;
        #endif
      }
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::fitPoint(xPositionsFit, yPositionsFit) finished" << endl;
      #endif
      return ValueT(h);
    }

    template< typename ValueT, typename CoordsT >
    ndarray::Array< ValueT, 2, 1 > ThinPlateSpline< ValueT, CoordsT >::fitArray( ndarray::Array< const CoordsT, 1, 1 > const& xPositionsFit,
                                                                                 ndarray::Array< const CoordsT, 1, 1 > const& yPositionsFit,
                                                                                 bool const isXYPositionsGridPoints){ /// fit positions
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::fitArray(xPositionsFit, yPositionsFit, isXYPositionsGridPoints) started" << endl;
      #endif
      ndarray::Array< ValueT, 2, 1 > arrOut;
      if (isXYPositionsGridPoints){
        arrOut = ndarray::allocate(yPositionsFit.getShape()[0], xPositionsFit.getShape()[0]);
        for ( int yPos = 0; yPos < yPositionsFit.size(); ++yPos ){
          for ( int xPos = 0; xPos < xPositionsFit.size(); ++xPos ){
            arrOut[yPos][xPos] = fitPoint(xPositionsFit[xPos], 
                                          yPositionsFit[yPos]);
          }
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "ThinPlateSpline::fitArray: y = " << yPositionsFit[yPos] << ": arrOut[" << yPos << "][*] = " << arrOut[ndarray::view(yPos)()] << std::endl;
          #endif
        }
      }
      else{/// arrOut will be a vector
        arrOut = ndarray::allocate(yPositionsFit.getShape()[0], 1);
        for ( int iPos = 0; iPos < xPositionsFit.size(); ++iPos ){
          arrOut[iPos][0] = fitPoint(xPositionsFit[iPos], 
                                     yPositionsFit[iPos]);
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "ThinPlateSpline::fitArray: x = " << xPositionsFit[iPos] << ", y = " << yPositionsFit[iPos] << ": arrOut[" << iPos << "][0] = " << arrOut[iPos][0] << std::endl;
          #endif
        }
      }

      // Calc bending energy
    /*  matrix<double> w( p, 1 );
      for ( int i=0; i<p; ++i )
        w(i,0) = mtx_v(i,0);
      matrix<double> be = prod( prod<matrix<double> >( trans(w), mtx_orig_k ), w );
      bending_energy = be(0,0);
      std::cout << "ThinPlateSpline::fitArray: bending_energy = " << bending_energy << std::endl;
    */  
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSpline::fitArray: arrOut = " << arrOut << std::endl;
      #endif

      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::fitArray(xPositionsFit, yPositionsFit, isXYPositionsGridPoints) finished" << endl;
      #endif
      return arrOut;
    }
            
    template< typename ValueT, typename CoordsT >
    ThinPlateSpline< ValueT, CoordsT >& ThinPlateSpline< ValueT, CoordsT >::operator=(ThinPlateSpline< ValueT, CoordsT > const& tps){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::operator=(tps) started" << endl;
        cout << " tps.getControlPointsX.getShape()[ 0 ] = " << tps.getControlPointsX().getShape()[ 0 ] << endl;
      #endif
      _controlPointsX = copy( tps.getControlPointsX() );
      _controlPointsY = copy( tps.getControlPointsY() );
      _controlPointsZ = copy( tps.getControlPointsZ() );
      _controlPointsWeight = copy( tps.getControlPointsWeight() );
      _coefficients = copy( tps.getCoefficients() );
      _regularization = tps.getRegularization();
      _radiusNormalizationFactor = tps.getRadiusNormalizationFactor();
      _bendingEnergy = tps.getBendingEnergy();
      _isWeightsSet = tps.isWeightsSet();
      #ifdef __DEBUG_TPS__
        cout << " this->getControlPointsX.getShape()[ 0 ] = " << this->getControlPointsX().getShape()[ 0 ] << endl;
        cout << "ThinPlateSpline::operator=(tps) finished" << endl;
      #endif
      return *this;
    }            
    
    template class ThinPlateSpline< float, float >;
    template class ThinPlateSpline< float, double >;
    template class ThinPlateSpline< double, float >;
    template class ThinPlateSpline< double, double >;

    template< typename ValueT, typename CoordsT >
    ThinPlateSplineChiSquare< ValueT, CoordsT >::ThinPlateSplineChiSquare( ):
      _controlPointsX(),
      _controlPointsY(),
      _controlPointsZ(),
      _fitPointsXY(),
      _coefficients(),
      _regularization(0.),
      _bendingEnergy(0.)
    {
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare() finished" << endl;
      #endif
    }

    template< typename ValueT, typename CoordsT >
    ThinPlateSplineChiSquare< ValueT, CoordsT >::ThinPlateSplineChiSquare( ndarray::Array< const CoordsT, 1, 1 > const& controlPointsX,
                                                                           ndarray::Array< const CoordsT, 1, 1 > const& controlPointsY,
                                                                           ndarray::Array< const ValueT, 1, 1 > const& controlPointsZ,
                                                                           ndarray::Array< const CoordsT, 1, 1 > const& fitPointsX,
                                                                           ndarray::Array< const CoordsT, 1, 1 > const& fitPointsY,
                                                                           bool const isXYPositionsGridPoints,
                                                                           ValueT const regularization,
                                                                           ValueT const radiusNormalizationFactor ):
      _controlPointsX( ndarray::copy( controlPointsX ) ),
      _controlPointsY( ndarray::copy( controlPointsY ) ),
      _controlPointsZ( ndarray::copy( controlPointsZ ) ),
      _regularization( regularization ),
      _radiusNormalizationFactor( radiusNormalizationFactor ),
      _bendingEnergy(0.)
    {
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare(controlPointsX, controlPointsY, controlPointsZ, fitPointsX, fitPointsY, isXYPositionsGridPoints, regularization, radiusNormalizationFactor) started" << endl;
      #endif
      /// Check input data:
      if ( controlPointsX.getShape()[ 0 ] != controlPointsY.getShape()[ 0 ] ){
        std::string message( "ThinPlateSpline::ThinPlateSplineChiSquare: ERROR: controlPointsX.getShape()[0] = " );
        message += std::to_string( controlPointsX.getShape()[ 0 ] ) + " != controlPointsY.getShape()[0] = ";
        message += std::to_string( controlPointsY.getShape()[ 0 ] );
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      if ( controlPointsX.getShape()[ 0 ] != controlPointsZ.getShape()[ 0 ] ){
        std::string message( "ThinPlateSpline::ThinPlateSplineChiSquare: ERROR: controlPointsX.getShape()[0] = " );
        message += std::to_string( controlPointsX.getShape()[ 0 ] ) + " != controlPointsZ.getShape()[0] = ";
        message += std::to_string( controlPointsZ.getShape()[ 0 ] );
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      
      if (isXYPositionsGridPoints){
        _fitPointsXY = ndarray::allocate( fitPointsX.getShape()[ 0 ] * fitPointsY.getShape()[ 0 ], 2 );
        createGridPointsXY( fitPointsX, fitPointsY);
        _coefficients = ndarray::allocate( fitPointsX.getShape()[ 0 ] * fitPointsY.getShape()[ 0 ] + 3);
      }
      else{
        if ( fitPointsX.getShape()[ 0 ] != fitPointsY.getShape()[ 0 ]){
          std::string message( "ThinPlateSpline::ThinPlateSplineChiSquare: ERROR: fitPointsX.getShape()[0] = " );
          message += std::to_string( fitPointsX.getShape()[ 0 ] ) + " != fitPointsY.getShape()[0] = ";
          message += std::to_string( fitPointsY.getShape()[ 0 ] );
          std::cout << message << std::endl;
          throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
        }
        _fitPointsXY = ndarray::allocate( fitPointsX.getShape()[ 0 ], 2 );
        for (size_t iPix = 0; iPix < fitPointsX.getShape()[ 0 ]; ++iPix ){
          _fitPointsXY[ iPix ][ 0 ] = fitPointsX[ iPix ];
          _fitPointsXY[ iPix ][ 1 ] = fitPointsY[ iPix ];
        }
        _coefficients = ndarray::allocate( fitPointsX.getShape()[ 0 ] + 3);
      }
      
      if ( !calculateCoefficients() ){
        std::string message( "ThinPlateSpline::ThinPlateSplineChiSquare: ERROR: calculateCoefficients returned FALSE" );
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare(controlPointsX, controlPointsY, controlPointsZ, fitPointsX, fitPointsY, isXYPositionsGridPoints, regularization, radiusNormalizationFactor) finished" << endl;
      #endif
    }
    
    template< typename ValueT, typename CoordsT >
    ThinPlateSplineChiSquare< ValueT, CoordsT >::ThinPlateSplineChiSquare( ThinPlateSplineChiSquare< ValueT, CoordsT > const& tps ):
            _controlPointsX( tps.getControlPointsX() ),
            _controlPointsY( tps.getControlPointsY() ),
            _controlPointsZ( tps.getControlPointsZ() ),
            _regularization( tps.getRegularization() ),
            _radiusNormalizationFactor( tps.getRadiusNormalizationFactor() ),
            _bendingEnergy( tps.getBendingEnergy() )
    { 
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare(tps) started" << endl;
      #endif
      _fitPointsXY = ndarray::allocate( tps.getFitPointsXY().getShape() );
      _fitPointsXY.deep() = tps.getFitPointsXY();
      
      _coefficients = ndarray::allocate( tps.getCoefficients().getShape() );
      _coefficients.deep() = tps.getCoefficients();
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare(tps) finished" << endl;
      #endif
    }
            
    template< typename ValueT, typename CoordsT >
    ValueT ThinPlateSplineChiSquare< ValueT, CoordsT >::tps_base_func( ValueT const r ){
      if ( r == 0.0 )
        return 0.0;
      else{
        #ifdef __TPS_BASE_FUNC_MULTIQUADRIC__
          return pow( ( r * r ) + ( _radiusNormalizationFactor * _radiusNormalizationFactor ), -0.5 );
        #else 
          return r * r * log( r / _radiusNormalizationFactor );
        #endif
      }
    }
            
    template< typename ValueT, typename CoordsT >
    bool ThinPlateSplineChiSquare< ValueT, CoordsT >::calculateCoefficients(){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::calculateCoefficients() started" << endl;
      #endif
      #ifdef __DEBUG_CALC_TPS__
        cout << "ThinPlateSplineChiSquare::calculateCoefficients: _controlPointsX = (" << _controlPointsX.getShape()[0] << ") = " << _controlPointsX << endl;
        cout << "ThinPlateSplineChiSquare::calculateCoefficients: _controlPointsY = (" << _controlPointsY.getShape()[0] << ") = " << _controlPointsY << endl;
        cout << "ThinPlateSplineChiSquare::calculateCoefficients: _controlPointsZ = (" << _controlPointsZ.getShape()[0] << ") = " << _controlPointsZ << endl;
      #endif

      // You We need at least 3 points to define a plane
      if ( _controlPointsX.getShape()[ 0 ] < 3 ){
        string message( "ThinPlateSplineChiSquare::calculateCoefficients: ERROR: _controlPointsX.getShape()[0] = " );
        message += to_string( _controlPointsX.getShape()[ 0 ] ) + " < 3";
        cout << message << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }

      unsigned nFitPoints = _fitPointsXY.getShape()[ 0 ];
      unsigned nDataPoints = _controlPointsX.getShape()[ 0 ];

      // Allocate the matrix and vector
      ndarray::Array< double, 2, 1 > mtx_l = ndarray::allocate( nFitPoints + 3, nFitPoints + 3);
      ndarray::Array< double, 1, 1 > mtx_v = ndarray::allocate( nFitPoints + 3);
      mtx_v.deep() = 0.;
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSplineChiSquare::calculateCoefficients: memory for mtx_l, mtx_v allocated" << std::endl;
      #endif

      mtx_l.deep() = fillMatrix();

      // Fill the right hand vector V
      ndarray::Array< double, 1, 1 > pt_i = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_k = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_diff = ndarray::allocate(2);
      double elen_ik;
      double r_ik;
      for ( unsigned i = 0; i < nFitPoints; ++i ){
        pt_i[ 0 ] = _fitPointsXY[ i ][ 0 ];
        pt_i[ 1 ] = _fitPointsXY[ i ][ 1 ];
        for ( unsigned k = 0; k < nDataPoints; ++k ){
          pt_k[ 0 ] = _controlPointsX[ k ];
          pt_k[ 1 ] = _controlPointsY[ k ];
          pt_diff.deep() = pt_i - pt_k;
          pt_diff.asEigen() = pt_diff.asEigen().array() * pt_diff.asEigen().array();
          elen_ik = sqrt( pt_diff.asEigen().sum() );
          mtx_v[ i ] += 2. * tps_base_func( elen_ik );// - _controlPointsZ[ k ];
        }
      }
      mtx_v[ nFitPoints ] = 2. * double( nDataPoints );
      for (unsigned k = 0; k < nDataPoints; ++k ){
        mtx_v[ nFitPoints + 1 ] += 2. * _controlPointsX[ k ];
        mtx_v[ nFitPoints + 2 ] += 2. * _controlPointsY[ k ];
      }
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSplineChiSquare::calculateCoefficients: mtx_v = " << mtx_v << std::endl;
      #endif

      // Solve the linear system "inplace"
      _coefficients.asEigen() = mtx_l.asEigen().colPivHouseholderQr().solve(mtx_v.asEigen());
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSplineChiSquare::calculateCoefficients: after colPivHouseholderQr: _coefficients = " << _coefficients << std::endl;
      #endif

      // Calculate bending energy
      ndarray::Array< double, 1, 1 > mtxv_dot_mtxl = ndarray::allocate(mtx_l.getShape()[1]);
      for (int i = 0; i < mtx_l.getShape()[ 1 ]; ++i){
        mtxv_dot_mtxl[ i ] = 0.;
        for (int j = 0; j < mtx_l.getShape()[ 0 ]; ++j)
          mtxv_dot_mtxl[ i ] += mtx_v[ j ] * mtx_l[ j ][ i ];
      }  
      double bendingEnergy = 0.;
      for (int i = 0; i < mtx_l.getShape()[ 1 ]; ++i)
        bendingEnergy += mtx_v[ i ] * mtxv_dot_mtxl[ i ];
        
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::calculateCoefficients() finished" << endl;
      #endif
      return true;
    }
    
    template< typename ValueT, typename CoordsT >
    void ThinPlateSplineChiSquare< ValueT, CoordsT >::createGridPointsXY( ndarray::Array< const CoordsT, 1, 1 > const& gridPointsX,
                                                                          ndarray::Array< const CoordsT, 1, 1 > const& gridPointsY ){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::createGridPointsXY(gridPointsX, gridPointsY) started" << endl;
      #endif
      unsigned row = 0;
      for ( unsigned x = 0; x < gridPointsX.getShape()[ 0 ]; ++x ){
        for ( unsigned y = 0; y < gridPointsY.getShape()[ 0 ]; ++y ){
          _fitPointsXY[ row ][ 0 ] = gridPointsX[ x ];
          _fitPointsXY[ row ][ 1 ] = gridPointsY[ y ];
          ++row;
        }
      }
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSplineChiSquare::createGridPointsXY: _fitPointsXY = " << _fitPointsXY << std::endl;
      #endif
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::createGridPointsXY(gridPointsX, gridPointsY) finished" << endl;
      #endif
    }
    
    template< typename ValueT, typename CoordsT >
    ndarray::Array< double, 2, 1 > ThinPlateSplineChiSquare< ValueT, CoordsT >::fillMatrix(){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::fillMatrix() started" << endl;
      #endif
      unsigned nFitPoints = _fitPointsXY.getShape()[ 0 ];
      unsigned nControlPoints = _controlPointsX.getShape()[ 0 ];

      // Allocate the matrix and vector
      ndarray::Array< double, 2, 1 > mtx_l = ndarray::allocate( nFitPoints + 3, nFitPoints + 3);
//      ndarray::Array< double, 2, 1 > fArr = ndarray::allocate( nGridPoints, 3 );
//      ndarray::Array< double, 2, 1 > fTArr = ndarray::allocate( 3, nGridPoints );
      mtx_l.deep() = 0.;
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "memory for mtx_l, mtx_v allocated" << std::endl;
      #endif

//      createGridPointsXY();

      ndarray::Array< double, 1, 1 > pt_i = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_j = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_k = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_diff = ndarray::allocate(2);
      double elen, elen_ik, elen_jk, tpsBaseValue, tpsBaseValueSquared;
      // Fill K (p x p)
      //
      // K is symmetrical so we really have to
      // calculate only about half of the coefficients.
      for ( unsigned i = 0; i < nFitPoints; ++i ){
        pt_i[ 0 ] = _fitPointsXY[ i ][ 0 ];
        pt_i[ 1 ] = _fitPointsXY[ i ][ 1 ];
        #ifdef __DEBUG_CALC_TPS__
          std::cout << "i = " << i << ": pt_i set to " << pt_i << std::endl;
        #endif
        for (unsigned k = 0; k < nControlPoints; ++k){
          pt_k[ 0 ] = _controlPointsX[ k ];
          pt_k[ 1 ] = _controlPointsY[ k ];
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "i = " << i << ", k = " << k << ": pt_k set to " << pt_k << std::endl;
          #endif

          pt_diff.deep() = pt_i - pt_k;
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "i = " << i << ", k = " << k << ": pt_diff set to " << pt_diff << std::endl;
          #endif
          pt_diff.asEigen() = pt_diff.asEigen().array() * pt_diff.asEigen().array();
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "i = " << i << ", k = " << k << ": pt_diff set to " << pt_diff << std::endl;
          #endif

          elen = sqrt( pt_diff.asEigen().sum() );
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "i = " << i << ", k = " << k << ": elen set to " << elen << std::endl;
          #endif

          tpsBaseValue = tps_base_func(elen);
          tpsBaseValueSquared = tpsBaseValue * tpsBaseValue;
          mtx_l[ i ][ i ] += tpsBaseValueSquared / _controlPointsZ[ k ];
        }
        mtx_l [ i ][ i ] = 2. * mtx_l[ i ][ i ] + _regularization;
        std::cout << "mtx_l[ " << i << " ][ " << i << " ] set to " << mtx_l[ i ][ i ] << std::endl;
        
        for ( unsigned j = i + 1; j < nFitPoints; ++j ){
          pt_j[ 0 ] = _fitPointsXY[ j ][ 0 ];
          pt_j[ 1 ] = _fitPointsXY[ j ][ 1 ];
          for ( unsigned k = 0; k < nControlPoints; ++k ){
            pt_k[0] = _controlPointsX[k];
            pt_k[1] = _controlPointsY[k];
            #ifdef __DEBUG_CALC_TPS__
              std::cout << "i = " << i << ", j = " << j << ": pt_i set to " << pt_i << ", pt_j = " << pt_j << ", pt_k = " << pt_k << std::endl;
            #endif
            pt_diff.deep() = pt_i - pt_k;
            pt_diff.asEigen() = pt_diff.asEigen().array() * pt_diff.asEigen().array();
            #ifdef __DEBUG_CALC_TPS__
              std::cout << "i = " << i << ", j = " << j << ", k = " << k << ": pt_diff set to " << pt_diff << std::endl;
            #endif
            elen_ik = sqrt( pt_diff.asEigen().sum() );
            #ifdef __DEBUG_CALC_TPS__
              std::cout << "i = " << i << ", j = " << j << ", k = " << k << ": elen_ik set to " << elen_ik << std::endl;
            #endif

            pt_diff.deep() = pt_j - pt_k;
            pt_diff.asEigen() = pt_diff.asEigen().array() * pt_diff.asEigen().array();
            #ifdef __DEBUG_CALC_TPS__
              std::cout << "i = " << i << ", j = " << j << ", k = " << k << ": pt_diff set to " << pt_diff << std::endl;
            #endif
            elen_jk = sqrt( pt_diff.asEigen().sum() );
            #ifdef __DEBUG_CALC_TPS__
              std::cout << "i = " << i << ", j = " << j << ", k = " << k << ": elen_jk set to " << elen_jk << std::endl;
            #endif
            mtx_l[ i ][ j ] += 2. * tps_base_func( elen_ik ) * tps_base_func( elen_jk ) / _controlPointsZ[ k ];
//            mtx_l[ i ][ j ] += elen_ik * elen_ik * elen_jk * elen_jk * tps_base_func( elen_ik ) * tps_base_func( elen_jk ) / _controlPointsZ[ k ];
          }
          mtx_l[ j ][ i ] = mtx_l[ i ][ j ];
//          #ifdef __DEBUG_CALC_TPS__
            std::cout << "i = " << i << ", j = " << j << ": mtx_l[i][j] set to " << mtx_l[i][j] << std::endl;
//          #endif
        }
        
        // P (p x 3, upper right)
        // P transposed (3 x p, bottom left)
        //*
        for ( unsigned k = 0; k < nControlPoints; ++k ){
          pt_k[0] = _controlPointsX[k];
          pt_k[1] = _controlPointsY[k];
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "i = " << i << ", k = " << k << ": pt_i set to " << pt_i << ", pt_j = " << pt_j << ", pt_k = " << pt_k << std::endl;
          #endif
          pt_diff.deep() = pt_i - pt_k;
          pt_diff.asEigen() = pt_diff.asEigen().array() * pt_diff.asEigen().array();
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "i = " << i << ", k = " << k << ": pt_diff set to " << pt_diff << std::endl;
          #endif
          elen_ik = sqrt( pt_diff.asEigen().sum() );
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "i = " << i << ", k = " << k << ": elen_ik set to " << elen_ik << std::endl;
          #endif
          mtx_l[ i ][ nFitPoints + 0 ] += 2. * tps_base_func( elen_ik)  / _controlPointsZ[ k ];
          mtx_l[ i ][ nFitPoints + 1 ] += 2. * _controlPointsX[ k ] * tps_base_func( elen_ik ) / _controlPointsZ[ k ];
          mtx_l[ i ][ nFitPoints + 2 ] += 2. * _controlPointsY[ k ] * tps_base_func( elen_ik ) / _controlPointsZ[ k ];
        }
        mtx_l[ nFitPoints + 0 ][ i ] = mtx_l[ i ][ nFitPoints + 0 ];
        mtx_l[ nFitPoints + 1 ][ i ] = mtx_l[ i ][ nFitPoints + 1 ];
        mtx_l[ nFitPoints + 2 ][ i ] = mtx_l[ i ][ nFitPoints + 2 ];
        //*/
      }
      //*
      // O (3 x 3, lower right)
      for ( unsigned k = 0; k < nControlPoints; ++k ){
        mtx_l[ nFitPoints ][ nFitPoints ] += 2. / _controlPointsZ[ k ];
        mtx_l[ nFitPoints + 1 ][ nFitPoints + 1 ] += 2. * _controlPointsX[ k ] * _controlPointsX[ k ] / _controlPointsZ[ k ];
        mtx_l[ nFitPoints + 2 ][ nFitPoints + 2 ] += 2. * _controlPointsY[ k ] * _controlPointsY[ k ] / _controlPointsZ[ k ];
        mtx_l[ nFitPoints ][ nFitPoints + 1 ] += 2. * _controlPointsX[ k ] / _controlPointsZ[ k ];
        mtx_l[ nFitPoints ][ nFitPoints + 2 ] += 2. * _controlPointsY[ k ] / _controlPointsZ[ k ];
        mtx_l[ nFitPoints + 1 ][ nFitPoints + 2 ] += 2. * _controlPointsX[ k ] * _controlPointsY[ k ] / _controlPointsZ[ k ];
      }
      mtx_l[ nFitPoints + 1 ][ nFitPoints ] = mtx_l[ nFitPoints ][ nFitPoints + 1 ];
      mtx_l[ nFitPoints + 2 ][ nFitPoints ] = mtx_l[ nFitPoints ][ nFitPoints + 2 ];
      mtx_l[ nFitPoints + 2 ][ nFitPoints + 1 ] = mtx_l[ nFitPoints + 1 ][ nFitPoints + 2 ];
      //*/

        
/*        
        
        // Fill the rest of L

        // P (p x 3, upper right)
        // P transposed (3 x p, bottom left)
      for ( unsigned i = 0; i < nFitPoints; ++i ){
        mtx_l[ i ][ nFitPoints + 0 ] = mtx_l[ nFitPoints + 0 ][ i ] = 1.0;
        mtx_l[ i ][ nFitPoints + 1] = mtx_l[ nFitPoints + 1 ][ i ] = _fitPointsXY[ i ][ 0 ];
        mtx_l[ i ][ nFitPoints + 2] = mtx_l[ nFitPoints + 2][ i ] = _fitPointsXY[ i ][ 1 ];
      
        // O (3 x 3, lower right)
        for ( unsigned i = nFitPoints; i < nFitPoints + 3; ++i ){
          for ( unsigned j = nFitPoints; j < nFitPoints + 3; ++j ){
            mtx_l[i][j] = 0.0;
          }
        }
      }        
      
*/      
      
      
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "mtx_l set to " << mtx_l << std::endl;
      #endif
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::fillMatrix() finished" << endl;
      #endif
      return mtx_l;
    }
            
    template< typename ValueT, typename CoordsT >
    ValueT ThinPlateSplineChiSquare< ValueT, CoordsT >::fitPoint(CoordsT const xPositionFit, 
                                                                 CoordsT const yPositionFit){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::fitPoint(xPositionFit, yPositionFit) started" << endl;
      #endif
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSpline::fitPoint: x = " << xPositionFit << ", y = " << yPositionFit << std::endl;
        //std::cout << "ThinPlateSpline::fitPoint: _coefficients = " << _coefficients << std::endl;
      #endif
      unsigned p = _fitPointsXY.getShape()[0];
      double h = _coefficients[ p ] + ( _coefficients[ p + 1 ] * xPositionFit ) + ( _coefficients[ p + 2 ] * yPositionFit );
      ndarray::Array<CoordsT, 1, 1> pt_i = ndarray::allocate(2);
      ndarray::Array<CoordsT, 1, 1> pt_cur = ndarray::allocate(2);
      pt_cur[0] = xPositionFit;
      pt_cur[1] = yPositionFit;
      ndarray::Array<CoordsT, 1, 1> pt_diff = ndarray::allocate(2);
      ValueT len;
      //double h = 0.;
      for ( unsigned i = 0; i < p; ++i ){
        pt_i[0] = _fitPointsXY[ i ][ 0 ];
        pt_i[1] = _fitPointsXY[ i ][ 1 ];
        pt_diff.deep() = pt_i - pt_cur;
        pt_diff.asEigen() = pt_diff.asEigen().array() * pt_diff.asEigen().array();
        len = sqrt(pt_diff.asEigen().sum());
        h += _coefficients[i] * tps_base_func( len );
      }
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::fitPoint(xPositionFit, yPositionFit) finished" << endl;
      #endif
      return ValueT(h);
    }

    template< typename ValueT, typename CoordsT >
    ndarray::Array< ValueT, 2, 1 > ThinPlateSplineChiSquare< ValueT, CoordsT >::fitArray( ndarray::Array< const CoordsT, 1, 1 > const& xPositionsFit,
                                                                                          ndarray::Array< const CoordsT, 1, 1 > const& yPositionsFit,
                                                                                          bool const isXYPositionsGridPoints){ /// fit positions
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::fitArray(xPositionsFit, yPositionsFit, isXYPositionsGridPoints) started" << endl;
      #endif
      ndarray::Array< ValueT, 2, 1 > arrOut;
      if (isXYPositionsGridPoints){
        arrOut = ndarray::allocate(yPositionsFit.getShape()[0], xPositionsFit.getShape()[0]);
        for ( int yPos = 0; yPos < yPositionsFit.size(); ++yPos ){
          for ( int xPos = 0; xPos < xPositionsFit.size(); ++xPos ){
            arrOut[yPos][xPos] = fitPoint(xPositionsFit[xPos], 
                                          yPositionsFit[yPos]);
          }
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "ThinPlateSpline::fitArray: y = " << yPositionsFit[yPos] << ": arrOut[" << yPos << "][*] = " << arrOut[ndarray::view(yPos)()] << std::endl;
          #endif
        }
      }
      else{/// arrOut will be a vector
        arrOut = ndarray::allocate(yPositionsFit.getShape()[0], 1);
        for ( int iPos = 0; iPos < xPositionsFit.size(); ++iPos ){
          arrOut[iPos][0] = fitPoint(xPositionsFit[iPos], 
                                     yPositionsFit[iPos]);
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "ThinPlateSpline::fitArray: x = " << xPositionsFit[iPos] << ", y = " << yPositionsFit[iPos] << ": arrOut[" << iPos << "][0] = " << arrOut[iPos][0] << std::endl;
          #endif
        }
      }

      // Calc bending energy
    /*  matrix<double> w( p, 1 );
      for ( int i=0; i<p; ++i )
        w(i,0) = mtx_v(i,0);
      matrix<double> be = prod( prod<matrix<double> >( trans(w), mtx_orig_k ), w );
      bending_energy = be(0,0);
      std::cout << "ThinPlateSpline::fitArray: bending_energy = " << bending_energy << std::endl;
    */  
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSpline::fitArray: arrOut = " << arrOut << std::endl;
      #endif

      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::fitArray(xPositionsFit, yPositionsFit, isXYPositionsGridPoints) finished" << endl;
      #endif
      return arrOut;
    }
    
    template class ThinPlateSplineChiSquare< float, float >;
    template class ThinPlateSplineChiSquare< float, double >;
    template class ThinPlateSplineChiSquare< double, float >;
    template class ThinPlateSplineChiSquare< double, double >;

  }
}}}
      
template< typename T >
std::ostream& operator<<(std::ostream& os, matrix<T> const& obj)
{
  for (int i = 0 ; i < obj.size1(); ++i){
    for (int j = 0; j < obj.size2(); ++j){
      os << obj(i, j) << " ";
    }
    os << endl;
  }
  return os;
}

template std::ostream& operator<<(std::ostream&, matrix<int> const&);
template std::ostream& operator<<(std::ostream&, matrix<float> const&);
template std::ostream& operator<<(std::ostream&, matrix<double> const&);
