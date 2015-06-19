#include "pfs/drp/stella/math/SurfaceFitting.h"

using namespace boost::numeric::ublas;

namespace pfs{ namespace drp{ namespace stella{ 
  namespace math{
          
    template< typename ValueT, typename CoordsT >
    ThinPlateSpline< ValueT, CoordsT >::ThinPlateSpline( ndarray::Array< const CoordsT, 1, 1 > const& controlPointsX,
                                                         ndarray::Array< const CoordsT, 1, 1 > const& controlPointsY,
                                                         ndarray::Array< const ValueT, 1, 1 > const& controlPointsZ,
                                                         double const regularization ):
      _controlPointsX(ndarray::copy(controlPointsX)),
      _controlPointsY(ndarray::copy(controlPointsY)),
      _controlPointsZ(ndarray::copy(controlPointsZ)),
      _controlPointsWeight(pfsDRPStella::utils::get1DndArray(ValueT(controlPointsZ.getShape()[0]))),
      _regularization(regularization),
      _isWeightsSet(false)
    {
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
    }
            
    template< typename ValueT, typename CoordsT >
    ThinPlateSpline< ValueT, CoordsT >::ThinPlateSpline( ndarray::Array< const CoordsT, 1, 1 > const& controlPointsX,
                                                         ndarray::Array< const CoordsT, 1, 1 > const& controlPointsY,
                                                         ndarray::Array< const ValueT, 1, 1 > const& controlPointsZ,
                                                         ndarray::Array< const ValueT, 1, 1 > const& controlPointsWeights):
      _controlPointsX(ndarray::copy(controlPointsX)),
      _controlPointsY(ndarray::copy(controlPointsY)),
      _controlPointsZ(ndarray::copy(controlPointsZ)),
      _controlPointsWeight(ndarray::copy(controlPointsWeights)),
      _regularization(0.),
      _isWeightsSet(true)
    {
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
      
      _coefficients = ndarray::allocate(controlPointsZ.getShape()[0] + 3);
      if (!calculateCoefficients()){
        std::string message("ThinPlateSpline::ThinPlateSpline: ERROR: calculateCoefficients returned FALSE");
        std::cout << message << std::endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }
    }

    template< typename ValueT, typename CoordsT >
    ValueT ThinPlateSpline< ValueT, CoordsT >::tps_base_func(ValueT r){
      if ( r == 0.0 )
        return 0.0;
      else
        return r*r * log(r);
    }
            
    template< typename ValueT, typename CoordsT >
    bool ThinPlateSpline< ValueT, CoordsT >::calculateCoefficients(){
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
      ndarray::Array< double, 2, 1 > mtx_l = ndarray::allocate(p+3, p+3);
      ndarray::Array< double, 1, 1 > mtx_v = ndarray::allocate(p+3);
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "memory for mtx_l, mtx_v allocated" << std::endl;
      #endif

      if (_isWeightsSet)
        mtx_l.deep() = fillWeightedMatrix();
      else
        mtx_l.deep() = fillRegularizedMatrix();

      // Fill the right hand vector V
      for ( unsigned i=0; i<p; ++i )
        mtx_v[i] = _controlPointsZ[i];
      mtx_v[p] = mtx_v[p+1] = mtx_v[p+2] = 0.0;
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "interpolateThinPlateSpline: mtx_v = " << mtx_v << std::endl;
      #endif

      // Solve the linear system "inplace"
      _coefficients.asEigen() = mtx_l.asEigen().colPivHouseholderQr().solve(mtx_v.asEigen());
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "interpolateThinPlateSpline: after colPivHouseholderQr: _coefficients = " << _coefficients << std::endl;
      #endif

      return true;
    }
    
    template< typename ValueT, typename CoordsT >
    ndarray::Array< double, 2, 1 > ThinPlateSpline< ValueT, CoordsT >::fillRegularizedMatrix(){
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
      return mtx_l;
    }
    
    template< typename ValueT, typename CoordsT >
    ndarray::Array< double, 2, 1 > ThinPlateSpline< ValueT, CoordsT >::fillWeightedMatrix(){
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
      return mtx_l;
    }    
            
    template< typename ValueT, typename CoordsT >
    ValueT ThinPlateSpline< ValueT, CoordsT >::fitPoint(CoordsT const xPositionFit, 
                                                        CoordsT const yPositionFit){
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSpline::fitPoint: x = " << xPositionFit << ", y = " << yPositionFit << std::endl;
        //std::cout << "ThinPlateSpline::fitPoint: _coefficients = " << _coefficients << std::endl;
      #endif
      unsigned p = _controlPointsX.getShape()[0];
      double h = _coefficients[p] + (_coefficients[p+1] * xPositionFit) + (_coefficients[p+2] * yPositionFit);
      ndarray::Array<CoordsT, 1, 1> pt_i = ndarray::allocate(2);
      ndarray::Array<CoordsT, 1, 1> pt_cur = ndarray::allocate(2);
      pt_cur[0] = xPositionFit;
      pt_cur[1] = yPositionFit;
      ndarray::Array<CoordsT, 1, 1> pt_diff = ndarray::allocate(2);
      ValueT len;
      for ( unsigned i = 0; i < p; ++i ){
        pt_i[0] = _controlPointsX[i];
        pt_i[1] = _controlPointsY[i];
        pt_diff.deep() = pt_i - pt_cur;
        pt_diff.deep() = pt_diff * pt_diff;
        len = sqrt(pt_diff.asEigen().sum());
        h += _coefficients[i] * tps_base_func( len );
      }
      return ValueT(h);
    }

    template< typename ValueT, typename CoordsT >
    ndarray::Array< ValueT, 2, 1 > ThinPlateSpline< ValueT, CoordsT >::fitArray( ndarray::Array< const CoordsT, 1, 1 > const& xPositionsFit,
                                                                                 ndarray::Array< const CoordsT, 1, 1 > const& yPositionsFit,
                                                                                 bool const isXYPositionsGridPoints){ /// fit positions
      ndarray::Array< ValueT, 2, 1 > arrOut;
      if (isXYPositionsGridPoints){
        arrOut = ndarray::allocate(yPositionsFit.getShape()[0], xPositionsFit.getShape()[0]);
        for ( int xPos = 0; xPos < xPositionsFit.size(); ++xPos ){
          for ( int yPos = 0; yPos < yPositionsFit.size(); ++yPos ){
            arrOut[yPos][xPos] = fitPoint(xPositionsFit[xPos], 
                                          yPositionsFit[yPos]);
            #ifdef __DEBUG_CALC_TPS__
              std::cout << "ThinPlateSpline::fitArray: x = " << xPositionsFit[xPos] << ", y = " << yPositionsFit[yPos] << ": arrOut[" << yPos << "][" << xPos << "] = " << arrOut[yPos][xPos] << std::endl;
            #endif
          }
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

      return arrOut;
    }
    
    template class ThinPlateSpline<float, float>;
    template class ThinPlateSpline<float, double>;
    template class ThinPlateSpline<double, float>;
    template class ThinPlateSpline<double, double>;

    template< typename ValueT, typename CoordsT >
    ThinPlateSplineChiSquare< ValueT, CoordsT >::ThinPlateSplineChiSquare( ndarray::Array< const CoordsT, 1, 1 > const& controlPointsX,
                                                                           ndarray::Array< const CoordsT, 1, 1 > const& controlPointsY,
                                                                           ndarray::Array< const ValueT, 1, 1 > const& controlPointsZ,
                                                                           ndarray::Array< const CoordsT, 1, 1 > const& gridPointsX,
                                                                           ndarray::Array< const CoordsT, 1, 1 > const& gridPointsY ):
      _controlPointsX(ndarray::copy(controlPointsX)),
      _controlPointsY(ndarray::copy(controlPointsY)),
      _controlPointsZ(ndarray::copy(controlPointsZ)),
      _gridPointsX(ndarray::copy(gridPointsX)),
      _gridPointsY(ndarray::copy(gridPointsY))
    {
      /// Check input data:
      if ( controlPointsX.getShape()[ 0 ] != controlPointsY.getShape()[ 0 ] ){
        std::string message( "ThinPlateSpline::ThinPlateSpline: ERROR: controlPointsX.getShape()[0] = " );
        message += std::to_string( controlPointsX.getShape()[ 0 ] ) + " != controlPointsY.getShape()[0] = ";
        message += std::to_string( controlPointsY.getShape()[ 0 ] );
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      if ( controlPointsX.getShape()[ 0 ] != controlPointsZ.getShape()[ 0 ] ){
        std::string message( "ThinPlateSpline::ThinPlateSpline: ERROR: controlPointsX.getShape()[0] = " );
        message += std::to_string( controlPointsX.getShape()[ 0 ] ) + " != controlPointsZ.getShape()[0] = ";
        message += std::to_string( controlPointsZ.getShape()[ 0 ] );
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      
      _gridPointsXY = ndarray::allocate( gridPointsX.getShape()[ 0 ] * gridPointsY.getShape()[ 0 ], 2 );
      createGridPointsXY();
      
      _coefficients = ndarray::allocate( gridPointsX.getShape()[ 0 ] * gridPointsY.getShape()[ 0 ] );
      if ( !calculateCoefficients() ){
        std::string message( "ThinPlateSpline::ThinPlateSpline: ERROR: calculateCoefficients returned FALSE" );
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
    }
            
    template< typename ValueT, typename CoordsT >
    ValueT ThinPlateSplineChiSquare< ValueT, CoordsT >::tps_base_func(ValueT r){
      if ( r == 0.0 )
        return 0.0;
      else
        return r*r * log(r);
    }
            
    template< typename ValueT, typename CoordsT >
    bool ThinPlateSplineChiSquare< ValueT, CoordsT >::calculateCoefficients(){
      #ifdef __DEBUG_CALC_TPS__
        cout << "interpolateThinPlateSpline: _controlPointsX = (" << _controlPointsX.getShape()[0] << ") = " << _controlPointsX << endl;
        cout << "interpolateThinPlateSpline: _controlPointsY = (" << _controlPointsY.getShape()[0] << ") = " << _controlPointsY << endl;
        cout << "interpolateThinPlateSpline: _controlPointsZ = (" << _controlPointsZ.getShape()[0] << ") = " << _controlPointsZ << endl;
      #endif

      // You We need at least 3 points to define a plane
      if ( _controlPointsX.getShape()[ 0 ] < 3 ){
        string message( "interpolateThinPlateSpline: ERROR: _controlPointsX.getShape()[0] = " );
        message += to_string( _controlPointsX.getShape()[ 0 ] ) + " < 3";
        cout << message << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }

      unsigned nGridPoints = _gridPointsX.getShape()[ 0 ] * _gridPointsY.getShape()[ 0 ];
      unsigned nDataPoints = _controlPointsX.getShape()[ 0 ];

      // Allocate the matrix and vector
      ndarray::Array< double, 2, 1 > mtx_l = ndarray::allocate( nGridPoints, nGridPoints );
      ndarray::Array< double, 1, 1 > mtx_v = ndarray::allocate( nGridPoints );
      mtx_v.deep() = 0.;
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "memory for mtx_l, mtx_v allocated" << std::endl;
      #endif

      mtx_l.deep() = fillMatrix();

      // Fill the right hand vector V
      double r_ik;
      unsigned i = 0;
      for ( unsigned x = 0; x < _gridPointsX.getShape()[ 0 ]; ++x ){
        for ( unsigned y = 0; y < _gridPointsY.getShape()[ 0 ]; ++y ){
          for ( unsigned k = 0; k < nDataPoints; ++k ){
            r_ik = sqrt( pow( _controlPointsX[ k ] - _gridPointsX[ x ], 2 ) + pow( _controlPointsY[ k ] - _gridPointsY[ y ], 2) );
            mtx_v[ i ] += 2. * pow( r_ik, 2 ) * log10( r_ik ) - _controlPointsZ[ i ];
          }
          ++i;
        }
      }
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "interpolateThinPlateSpline: mtx_v = " << mtx_v << std::endl;
      #endif

      // Solve the linear system "inplace"
      _coefficients.asEigen() = mtx_l.asEigen().colPivHouseholderQr().solve(mtx_v.asEigen());
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "interpolateThinPlateSpline: after colPivHouseholderQr: _coefficients = " << _coefficients << std::endl;
      #endif

      return true;
    }
    
    template< typename ValueT, typename CoordsT >
    void ThinPlateSplineChiSquare< ValueT, CoordsT >::createGridPointsXY(){
      unsigned nGridPointsX = _gridPointsX.getShape()[ 0 ];
      unsigned nGridPointsY = _gridPointsY.getShape()[ 0 ];
      unsigned row = 0;
      for ( unsigned x = 0; x < nGridPointsX; ++x ){
        for ( unsigned y = 0; y < nGridPointsY; ++y ){
          _gridPointsXY[ row ][ 0 ] = _gridPointsX[ x ];
          _gridPointsXY[ row ][ 1 ] = _gridPointsY[ y ];
        }
      }
    }
    
    template< typename ValueT, typename CoordsT >
    ndarray::Array< double, 2, 1 > ThinPlateSplineChiSquare< ValueT, CoordsT >::fillMatrix(){
      unsigned nGridPoints = _gridPointsX.getShape()[ 0 ] * _gridPointsY.getShape()[ 0 ];
      unsigned nControlPoints = _controlPointsX.getShape()[ 0 ];

      // Allocate the matrix and vector
      ndarray::Array< double, 2, 1 > mtx_l = ndarray::allocate( nGridPoints, nGridPoints );
      mtx_l.deep() = 0.;
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "memory for mtx_l, mtx_v allocated" << std::endl;
      #endif

      createGridPointsXY();

      ndarray::Array< double, 1, 1 > pt_i = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_j = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_k = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_diff = ndarray::allocate(2);
      double elen, elen_ik, elen_jk, tpsBaseValue, tpsBaseValueSquared;
      // Fill K (p x p)
      //
      // K is symmetrical so we really have to
      // calculate only about half of the coefficients.
      for ( unsigned i = 0; i < nGridPoints; ++i ){
        for (unsigned k = 0; k < nControlPoints; ++k){
          pt_i[ 0 ] = _gridPointsXY[ i ][ 0 ];
          pt_i[ 1 ] = _gridPointsXY[ i ][ 1 ];
          pt_k[ 0 ] = _controlPointsX[ k ];
          pt_k[ 1 ] = _controlPointsY[ k ];

          pt_diff.deep() = pt_i - pt_k;
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
        mtx_l [ i ][ i ] = 2. * mtx_l[ i ][ i ];
        
        pt_i[ 0 ] = _gridPointsXY[ i ][ 0 ];
        pt_i[ 1 ] = _gridPointsXY[ i ][ 1 ];
        for ( unsigned j = i + 1; j < nGridPoints; ++j ){
          pt_j[ 0 ] = _gridPointsXY[ j ][ 0 ];
          pt_j[ 1 ] = _gridPointsXY[ j ][ 1 ];
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
            mtx_l[ i ][ j ] += elen_ik * elen_ik * elen_jk * elen_jk * tps_base_func( elen_ik ) * tps_base_func( elen_jk ) / _controlPointsZ[ k ];
          }
          mtx_l[ j ][ i ] = mtx_l[ i ][ j ];
          #ifdef __DEBUG_CALC_TPS__
            std::cout << "i = " << i << ", j = " << j << ", k = " << k << ": mtx_l[i][j] set to " << mtx_l[i][j] << std::endl;
          #endif
        }
      }
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "mtx_l[i][j] set to " << mtx_l[i][j] << std::endl;
      #endif
      return mtx_l;
    }
            
    template< typename ValueT, typename CoordsT >
    ValueT ThinPlateSplineChiSquare< ValueT, CoordsT >::fitPoint(CoordsT const xPositionFit, 
                                                        CoordsT const yPositionFit){
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSpline::fitPoint: x = " << xPositionFit << ", y = " << yPositionFit << std::endl;
        //std::cout << "ThinPlateSpline::fitPoint: _coefficients = " << _coefficients << std::endl;
      #endif
      unsigned p = _gridPointsXY.getShape()[0];
      //double h = _coefficients[p] + (_coefficients[p+1] * xPositionFit) + (_coefficients[p+2] * yPositionFit);
      ndarray::Array<CoordsT, 1, 1> pt_i = ndarray::allocate(2);
      ndarray::Array<CoordsT, 1, 1> pt_cur = ndarray::allocate(2);
      pt_cur[0] = xPositionFit;
      pt_cur[1] = yPositionFit;
      ndarray::Array<CoordsT, 1, 1> pt_diff = ndarray::allocate(2);
      ValueT len;
      double h = 0.;
      for ( unsigned i = 0; i < p; ++i ){
        pt_i[0] = _gridPointsXY[ i ][ 0 ];
        pt_i[1] = _gridPointsXY[ i ][ 1 ];
        pt_diff.deep() = pt_i - pt_cur;
        pt_diff.deep() = pt_diff * pt_diff;
        len = sqrt(pt_diff.asEigen().sum());
        h += _coefficients[i] * tps_base_func( len );
      }
      return ValueT(h);
    }

    template< typename ValueT, typename CoordsT >
    ndarray::Array< ValueT, 2, 1 > ThinPlateSplineChiSquare< ValueT, CoordsT >::fitArray( ndarray::Array< const CoordsT, 1, 1 > const& xPositionsFit,
                                                                                          ndarray::Array< const CoordsT, 1, 1 > const& yPositionsFit,
                                                                                          bool const isXYPositionsGridPoints){ /// fit positions
      ndarray::Array< ValueT, 2, 1 > arrOut;
      if (isXYPositionsGridPoints){
        arrOut = ndarray::allocate(yPositionsFit.getShape()[0], xPositionsFit.getShape()[0]);
        for ( int xPos = 0; xPos < xPositionsFit.size(); ++xPos ){
          for ( int yPos = 0; yPos < yPositionsFit.size(); ++yPos ){
            arrOut[yPos][xPos] = fitPoint(xPositionsFit[xPos], 
                                          yPositionsFit[yPos]);
            #ifdef __DEBUG_CALC_TPS__
              std::cout << "ThinPlateSpline::fitArray: x = " << xPositionsFit[xPos] << ", y = " << yPositionsFit[yPos] << ": arrOut[" << yPos << "][" << xPos << "] = " << arrOut[yPos][xPos] << std::endl;
            #endif
          }
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
