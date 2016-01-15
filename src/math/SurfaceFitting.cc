#include "pfs/drp/stella/math/SurfaceFitting.h"

using namespace boost::numeric::ublas;

namespace pfs{ namespace drp{ namespace stella{ 
  namespace math{
    
    template< typename ValueT, typename CoordsT >
    ThinPlateSplineBase< ValueT, CoordsT >::ThinPlateSplineBase( ):
      _dataPointsX(),
      _dataPointsY(),
      _dataPointsZ(),
      _dataPointsWeight(),
      _knots(),
      _coefficients(),
      _zFit(),
      _matrix(),
      _rhs(),
//      _regularization(0.),
//      _shapeParameter(1.),
      _tpsControl(),
      _bendingEnergy(0.),
      _chiSquare(0.),
      _regularizationBase(0.),
      _isWeightsSet(false)
    {
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineBase() finished: _zFit.shape = " << _zFit.getShape() << endl;
      #endif
    }
    
    template< typename ValueT, typename CoordsT >
    ThinPlateSplineBase< ValueT, CoordsT >::ThinPlateSplineBase( ThinPlateSplineBase const& tpsb ):
      _dataPointsX( tpsb.getDataPointsX() ),
      _dataPointsY( tpsb.getDataPointsY() ),
      _dataPointsZ( tpsb.getDataPointsZ() ),
      _dataPointsWeight( tpsb.getDataPointsWeight() ),
      _knots( tpsb.getKnots() ),
      _matrix( tpsb.getMatrix() ),
      _rhs( tpsb.getRHS() ),
//      _regularization( tpsb.getRegularization() ),
//      _shapeParameter( tpsb.getRadiusNormalizationFactor() ),
      _tpsControl(),
      _bendingEnergy( tpsb.getBendingEnergy() ),
      _chiSquare( tpsb.getChiSquare() ),
      _regularizationBase( tpsb.getRegularizationBase() ),
      _isWeightsSet( tpsb.isWeightsSet() )
    {
      if ( tpsb.getCoefficients().getShape()[ 0 ] == 0 ){
        std::string message( "ThinPlateSplineBase::ThinPlateSplineBase( tps ): ERROR: tps.getCoefficients().getShape()[ 0 ] == 0" );
        std::cout << message << std::endl;
        cout << "ThinPlateSplineBase::ThinPlateSpine( tps ): _dataPointsX.getShape()[ 0 ] = " << this->_dataPointsX.getShape()[ 0 ] << endl;
        cout << "ThinPlateSplineBase::ThinPlateSpineBase( tps ): _dataPointsY.getShape()[ 0 ] = " << this->_dataPointsY.getShape()[ 0 ] << endl;
        cout << "ThinPlateSplineBase::ThinPlateSpineBase( tps ): _dataPointsZ.getShape()[ 0 ] = " << this->_dataPointsZ.getShape()[ 0 ] << endl;
        cout << "ThinPlateSplineBase::ThinPlateSpineBase( tps ): _dataPointsWeight.getShape()[ 0 ] = " << this->_dataPointsWeight.getShape()[ 0 ] << endl;
        cout << "ThinPlateSplineBase::ThinPlateSpineBase( tps ): _tpsControl.regularization = " << this->_tpsControl.regularization << endl;
        cout << "ThinPlateSplineBase::ThinPlateSpineBase( tps ): _tpsControl.shapeParameter = " << this->_tpsControl.shapeParameter << endl;
        cout << "ThinPlateSplineBase::ThinPlateSpineBase( tps ): _bendingEnergy = " << this->_bendingEnergy << endl;
        cout << "ThinPlateSplineBase::ThinPlateSpineBase( tps ): _chiSquare = " << this->_chiSquare << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      this->_coefficients = ndarray::allocate( tpsb.getCoefficients().getShape() );
      this->_coefficients.deep() = tpsb.getCoefficients();
      this->_zFit = ndarray::allocate( tpsb.getZFit().getShape() );
      this->_zFit.deep() = tpsb.getZFit();
      if ( this->_zFit.getShape()[0] == 0 ){
        string message( "ThinPlateSplineBase::ThinPlateSpineBase( tps ): ERROR: _zFit.getShape()[0] == 0" );
        cout << message << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
      }
      _tpsControl = tpsb.getTPSControl();
//      cout << "tpsb(tpsb): tpsb.getTPSControl().shapeParameter = " << tpsb.getTPSControl().shapeParameter << ", _tpsControl.shapeParameter = " << _tpsControl.shapeParameter << endl;
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineBase( ThinPlateSplineBase ) finished: _zFit.shape = " << _zFit.getShape() << endl;
      #endif
    }

    template< typename ValueT, typename CoordsT >
    ndarray::Array< ValueT, 1, 1 > ThinPlateSplineBase< ValueT, CoordsT >::getZFit( bool deep ) const {
      if ( deep ){
        ndarray::Array< ValueT, 1, 1 > zFit;
        zFit = ndarray::allocate( _zFit.getShape() );
        zFit.deep() = _zFit;
        return zFit;
      }
      return _zFit;
    }            
            
    template< typename ValueT, typename CoordsT >
    bool ThinPlateSplineBase< ValueT, CoordsT >::calculateCoefficients(){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineBase::calculateCoefficients() started" << endl;
      #endif
      #ifdef __DEBUG_CALCULATE_COEFFICIENTS__
        cout << "ThinPlateSplineBase::calculateCoefficients: _dataPointsX = (" << this->_dataPointsX.getShape()[0] << ") = " << this->_dataPointsX << endl;
        cout << "ThinPlateSplineBase::calculateCoefficients: _dataPointsY = (" << this->_dataPointsY.getShape()[0] << ") = " << this->_dataPointsY << endl;
        cout << "ThinPlateSplineBase::calculateCoefficients: _dataPointsZ = (" << this->_dataPointsZ.getShape()[0] << ") = " << this->_dataPointsZ << endl;
      #endif
      if ( this->_zFit.getShape()[0] == 0 ){
        string message("ThinPlateSplineBase::calculateCoefficients: ERROR: _zFit.getShape()[0] == 0");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }

      // We need at least 3 points to define a plane
      if ( this->_dataPointsX.getShape()[0] < 3 ){
        string message("ThinPlateSplineBase::calculateCoefficients: ERROR: _dataPointsX.getShape()[0] = ");
        message += to_string(this->_dataPointsX.getShape()[0]) + " < 3";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());    
      }

      unsigned p = this->_dataPointsX.getShape()[0];

      // Allocate the matrix and vector
      this->fillMatrix();//ndarray::allocate( p + 3, p + 3 );
      this->fillRHS();//ndarray::allocate( p + 3 );
//      #ifdef __DEBUG_CALCULATE_COEFFICIENTS__
//        std::cout << "ThinPlateSplineBase::calcualteCoefficients: memory for mtx_l, mtx_v allocated" << std::endl;
//      #endif

//      mtx_l.deep() = fillMatrix();
      #ifdef __DEBUG_CALCULATE_COEFFICIENTS__
        Eigen::EigenSolver<Eigen::MatrixXd> es( this->_matrix.asEigen() );
        cout << "ThinPlateSpline::calculateCoefficients: EigenValues of mtx_l = " << es.eigenvalues() << endl;
        cout << "ThinPlateSpline::calculateCoefficients: EigenVectors of mtx_l = " << es.eigenvectors() << endl;
      #endif        
      if (_isWeightsSet)
        addWeightsToMatrix( );
      else
        addRegularizationToMatrix( );

      #ifdef __DEBUG_CALCULATE_COEFFICIENTS__
        Eigen::EigenSolver<Eigen::MatrixXd> esa( this->_matrix.asEigen());
        cout << "ThinPlateSpline::calculateCoefficients: EigenValues a of mtx_l = " << es.eigenvalues() << endl;
        cout << "ThinPlateSpline::calculateCoefficients: EigenVectors a of mtx_l = " << es.eigenvectors() << endl;
      #endif  
      // Fill the right hand vector V
//      mtx_v.deep() = this->fillRHSVector();

      // Solve the linear system "inplace"
      this->_coefficients.asEigen() = this->_matrix.asEigen().fullPivLu().solve( this->_rhs.asEigen() );
//      this->_coefficients.asEigen() = mtx_l.asEigen().colPivHouseholderQr().solve(mtx_v.asEigen());
      #ifdef __DEBUG_CALCULATE_COEFFICIENTS__
        std::cout << "ThinPlateSplineBase::calculateCoefficients: after colPivHouseholderQr: _coefficients = " << this->_coefficients << std::endl;
      #endif

      // Calculate bending energy
      ndarray::Array< double, 1, 1 > coeffs_dot_mtxl = ndarray::allocate( this->_knots.getShape()[ 0 ] );
      coeffs_dot_mtxl.deep() = 0.;
      for (int i = 0; i < this->_knots.getShape()[ 0 ]; ++i){
        for (int j = 0; j < this->_knots.getShape()[ 0 ]; ++j)
          coeffs_dot_mtxl[ i ] += this->_coefficients[ j ] * this->_matrix[ j ][ i ];
      }  
      this->_bendingEnergy = 0.;
      for (int i = 0; i < this->_knots.getShape()[ 0 ]; ++i)
        this->_bendingEnergy += this->_coefficients[ i ] * coeffs_dot_mtxl[ i ];
      #ifdef __DEBUG_CALCULATE_COEFFICIENTS__
        std::cout << "ThinPlateSplineBase::calculateCoefficients: _bendingEnergy = " << _bendingEnergy << std::endl;
      #endif
      
      // fit input data
      ndarray::Array< ValueT, 2, 1 > zFit = fitArray( this->_dataPointsX, 
                                                      this->_dataPointsY, 
                                                      false );
      #ifdef __DEBUG_CALCULATE_COEFFICIENTS__
        std::cout << "ThinPlateSplineBase::calculateCoefficients: zFit calculated: zFit.getShape() = " << zFit.getShape() << ", this->_zFit.getShape() = " << this->_zFit.getShape() << std::endl;
      #endif
      this->_zFit.deep() = zFit[ ndarray::view()( 0 ) ];
      #ifdef __DEBUG_CALCULATE_COEFFICIENTS__
        std::cout << "ThinPlateSplineBase::calculateCoefficients: _zFit assigned: _zFit.getShape() = " << this->_zFit.getShape() << std::endl;
      #endif

      // Calculate ChiSquare
      this->_chiSquare = 0.;
      auto itZFit = this->_zFit.begin();
      int iPos = 0;
      for ( auto itZIn = this->_dataPointsZ.begin(); itZIn != this->_dataPointsZ.end(); ++itZIn, ++itZFit ){
        this->_chiSquare += (*itZIn - *itZFit) * ( *itZIn - *itZFit ) / *itZIn;
        #ifdef __DEBUG_CALCULATE_COEFFICIENTS__
          std::cout << "ThinPlateSplineBase::calculateCoefficients: iPos = " << iPos << ": *itZIn = " << *itZIn << ", *itZFit = " << *itZFit << ", _chiSquare = " << _chiSquare << std::endl;
        #endif
        ++iPos;
      }
      #ifdef __DEBUG_CALCULATE_COEFFICIENTS__
        std::cout << "ThinPlateSplineBase::calculateCoefficients: _chiSquare = " << this->_chiSquare << std::endl;
      #endif
      
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineBase::calculateCoefficients() finished" << endl;
      #endif
      return true;
    }
    
    template< typename ValueT, typename CoordsT >
    void ThinPlateSplineBase< ValueT, CoordsT >::addRegularizationToMatrix( ){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineBase::addRegularizationToMatrix() started" << endl;
      #endif
      for ( unsigned i = 0; i < this->_knots.getShape()[0]; ++i ){
        // diagonal: reqularization parameters (lambda * a^2)
        this->_matrix[ i ][ i ] = this->_tpsControl.regularization * ( this->_regularizationBase * this->_regularizationBase );
      }
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::addRegularizationToMatrix() finished" << endl;
      #endif
      return;
    }
    
    template< typename ValueT, typename CoordsT >
    void ThinPlateSplineBase< ValueT, CoordsT >::addWeightsToMatrix( ){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineBase::addWeightsToMatrix() started" << endl;
      #endif
      
//      cout << "8. * CONST_PI = " << 8. * CONST_PI << endl;
      for ( int i = 0; i < this->_knots.getShape()[0]; ++i ){
        if ( this->_dataPointsWeight[ i ] < 0.000000000000001 )
          this->_matrix[ i ][ i ] = 0.;
        else{
//          mtx_l[i][i] = _regularizationBase / _dataPointsWeight[i];
          this->_matrix[ i ][ i ] = 8. * CONST_PI / this->_dataPointsWeight[i];
        }
      }
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineBase::addWeightsToMatrix() finished" << endl;
      #endif
      return;
    }    
            
    template< typename ValueT, typename CoordsT >
    ValueT ThinPlateSplineBase< ValueT, CoordsT >::fitPoint( CoordsT const xPositionFit, 
                                                             CoordsT const yPositionFit ){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineBase::fitPoint(xPositionsFit, yPositionsFit) started" << endl;
      #endif
      #ifdef __DEBUG_TPS_FITPOINT__
        cout << "ThinPlateSplineBase::fitPoint: x = " << xPositionFit << ", y = " << yPositionFit << endl;
        cout << "ThinPlateSplineBase::fitPoint: _coefficients = " << this->_coefficients << endl;
      #endif
      unsigned p = this->_knots.getShape()[ 0 ];
      double h = this->_coefficients[ p ] + ( this->_coefficients[ p + 1 ] * xPositionFit ) + ( this->_coefficients[ p + 2 ] * yPositionFit );
      ndarray::Array< CoordsT, 1, 1 > pt_i = ndarray::allocate( 2 );
      ndarray::Array< CoordsT, 1, 1 > pt_cur = ndarray::allocate( 2 );
      pt_cur[ 0 ] = xPositionFit;
      pt_cur[ 1 ] = yPositionFit;
      #ifdef __DEBUG_TPS_FITPOINT__
        cout << "ThinPlateSplineBase::fitPoint: pt_cur = [ " << pt_cur[0] << ", " << pt_cur[1] << "]" << endl;
      #endif
      ndarray::Array<CoordsT, 1, 1> pt_diff = ndarray::allocate( 2 );
      ValueT len;
      for ( unsigned i = 0; i < p; ++i ){
        pt_i[ 0 ] = this->_knots[ i ][ 0 ];
        pt_i[ 1 ] = this->_knots[ i ][ 1 ];
        #ifdef __DEBUG_TPS_FITPOINT__
          cout << "ThinPlateSplineBase::fitPoint: i=" << i << ": pt_i = [ " << pt_i[0] << ", " << pt_i[1] << "]" << endl;
        #endif
        pt_diff.deep() = pt_i - pt_cur;
        pt_diff.deep() = pt_diff * pt_diff;
        #ifdef __DEBUG_TPS_FITPOINT__
          cout << "ThinPlateSplineBase::fitPoint: i=" << i << ": pt_diff = [ " << pt_diff[0] << ", " << pt_diff[1] << "]" << endl;
        #endif
        len = sqrt( pt_diff.asEigen().sum() );
        #ifdef __DEBUG_TPS_FITPOINT__
          cout << "ThinPlateSplineBase::fitPoint: i=" << i << ": len = " << len << endl;
        #endif
        h += this->_coefficients[i] * this->tps_base_func( len );
        #ifdef __DEBUG_TPS_FITPOINT__
          cout << "ThinPlateSplineBase::fitPoint: i=" << i << ": h = " << h << endl;
        #endif
      }
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineBase::fitPoint(xPositionsFit, yPositionsFit) finished" << endl;
      #endif
      return ValueT(h);
    }

    template< typename ValueT, typename CoordsT >
    ndarray::Array< ValueT, 2, 1 > ThinPlateSplineBase< ValueT, CoordsT >::fitArray( ndarray::Array< const CoordsT, 1, 1 > const& xPositionsFit,
                                                                                     ndarray::Array< const CoordsT, 1, 1 > const& yPositionsFit,
                                                                                     bool const isXYPositionsGridPoints ){ /// fit positions
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineBase::fitArray(xPositionsFit, yPositionsFit, isXYPositionsGridPoints) started" << endl;
      #endif
      ndarray::Array< ValueT, 2, 1 > arrOut;
      if ( isXYPositionsGridPoints ){
        arrOut = ndarray::allocate(yPositionsFit.getShape()[0], xPositionsFit.getShape()[0]);
        for ( int yPos = 0; yPos < yPositionsFit.size(); ++yPos ){
          for ( int xPos = 0; xPos < xPositionsFit.size(); ++xPos ){
            arrOut[yPos][xPos] = fitPoint( xPositionsFit[ xPos ], 
                                           yPositionsFit[ yPos ] );
          }
          #ifdef __DEBUG_CALC_TPS__
            cout << "ThinPlateSplineBase::fitArray: y = " << yPositionsFit[yPos] << ": arrOut[" << yPos << "][*] = " << arrOut[ndarray::view(yPos)()] << endl;
          #endif
        }
      }
      else{/// arrOut will be a vector
        arrOut = ndarray::allocate( yPositionsFit.getShape()[ 0 ], 1 );
        for ( int iPos = 0; iPos < xPositionsFit.size(); ++iPos ){
          arrOut[iPos][0] = fitPoint( xPositionsFit[ iPos ], 
                                      yPositionsFit[ iPos ] );
          #ifdef __DEBUG_CALC_TPS__
            cout << "ThinPlateSplineBase::fitArray: x = " << xPositionsFit[iPos] << ", y = " << yPositionsFit[iPos] << ": arrOut[" << iPos << "][0] = " << arrOut[iPos][0] << std::endl;
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
        std::cout << "ThinPlateSplineBase::fitArray: arrOut = " << arrOut << std::endl;
      #endif

      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineBase::fitArray(xPositionsFit, yPositionsFit, isXYPositionsGridPoints) finished" << endl;
      #endif
      return arrOut;
    }

    template< typename ValueT, typename CoordsT >
    ValueT ThinPlateSplineBase< ValueT, CoordsT >::tps_base_func( ValueT r ){
//      cout << "tps_base_func: _tpsControl.shapeParameter = " << this->_tpsControl.shapeParameter << endl;
//      cout << "tps_base_func: _tpsControl.shapeParameter = " << _tpsControl.shapeParameter << endl;
      //    enum BASE_FUNC {  TPS=0, MULTIQUADRIC, MULTIQUADRIC_A, R, R_CUBE, INVERSE_MULTIQUADRIC, GENERALIZED_INVERSE_MULTIQUADRIC, WENDLAND_CSRBF, GAUSSIAN, MATERN_C0, MATERN_C2, MATERN_C4 };

      if ( _tpsControl.baseFunc.compare( "MULTIQUADRIC" ) == 0 ){
        return ( sqrt( ( r * r ) + _tpsControl.shapeParameter ) );
      }
      else if ( _tpsControl.baseFunc.compare( "MULTIQUADRIC_A" ) == 0 ){
        return ( sqrt( 1. + ( _tpsControl.shapeParameter * _tpsControl.shapeParameter * r * r ) ) );
      }
      else if ( _tpsControl.baseFunc.compare( "INVERSE_MULTIQUADRIC" ) == 0 ){
        return ( 1. / sqrt( 1. + ( _tpsControl.shapeParameter * _tpsControl.shapeParameter * r * r ) ) );
      }
      else if ( _tpsControl.baseFunc.compare( "GENERALIZED_INVERSE_MULTIQUADRIC" ) == 0 ){
        return ( pow( 1. + ( _tpsControl.shapeParameter * _tpsControl.shapeParameter * r * r ), _tpsControl.gMQParameter ) );
      }
      else if ( _tpsControl.baseFunc.compare( "TPS" ) == 0 ){
        #ifdef __DEBUG_FILL_MATRIX__
          cout << "tps_base_func: r = " << r << ": r * r * log( r / _tpsControl.shapeParameter(=" << _tpsControl.shapeParameter << ") ) = " << r * r * log( r / _tpsControl.shapeParameter ) << endl;
        #endif
        if ( r == 0.0 )
          return 0.0;
        else
          return ( r * r * _tpsControl.shapeParameter * _tpsControl.shapeParameter * log( r * _tpsControl.shapeParameter ) );
      }
      else if ( _tpsControl.baseFunc.compare( "WENDLAND_CSRBF" ) == 0 ){
        return ( pow( 1. - r, 4 ) + ( ( 4. * r ) + 1) );
      }
      else if ( _tpsControl.baseFunc.compare( "R" ) == 0 ){
        return ( r / _tpsControl.shapeParameter );
      }
      else if ( _tpsControl.baseFunc.compare( "R_CUBE" ) == 0 ){
        return ( r * r * r / _tpsControl.shapeParameter );
      }
      else if ( _tpsControl.baseFunc.compare( "GAUSSIAN" ) == 0 ){
        return ( exp( 0. - ( r * r * _tpsControl.shapeParameter * _tpsControl.shapeParameter ) ) );
      }
      else if ( _tpsControl.baseFunc.compare( "MATERN_C0" ) == 0 ){
        return ( exp( 0. - ( r * _tpsControl.shapeParameter ) ) );
      }
      else if ( _tpsControl.baseFunc.compare( "MATERN_C2" ) == 0 ){
        return ( exp( 0. - ( r * _tpsControl.shapeParameter ) ) * ( 1. + ( r * _tpsControl.shapeParameter ) ) );
      }
      else if ( _tpsControl.baseFunc.compare( "MATERN_C4" ) == 0 ){
        return ( exp( 0. - ( r * _tpsControl.shapeParameter ) ) * ( 3. + ( 3. * r * _tpsControl.shapeParameter ) + ( r * _tpsControl.shapeParameter * r * _tpsControl.shapeParameter ) ) );
      }
      else{
        std::string message("tps_base_func: ERROR: Could not identify baseFunc = ");
        message += _tpsControl.baseFunc;
        cout << message << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
      }
    }
            
    template< typename ValueT, typename CoordsT >
    ThinPlateSplineBase< ValueT, CoordsT >& ThinPlateSplineBase< ValueT, CoordsT >::operator=(ThinPlateSplineBase< ValueT, CoordsT > const& tps){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineBase::operator=(tps) started" << endl;
        cout << " tps.getDataPointsX.getShape()[ 0 ] = " << tps.getDataPointsX().getShape()[ 0 ] << endl;
      #endif
      _dataPointsX = copy( tps.getDataPointsX() );
      _dataPointsY = copy( tps.getDataPointsY() );
      _dataPointsZ = copy( tps.getDataPointsZ() );
      _dataPointsWeight = copy( tps.getDataPointsWeight() );
      _knots = copy( tps.getKnots() );
      _coefficients = copy( tps.getCoefficients() );
      _zFit = copy( tps.getZFit() );
      _matrix = copy( tps.getMatrix() );
      _rhs = copy( tps.getRHS() );
      _tpsControl = tps.getTPSControl();
//      _regularization = tps.getRegularization();
//      _shapeParameter = tps.getRadiusNormalizationFactor();
      _bendingEnergy = tps.getBendingEnergy();
      _chiSquare = tps.getChiSquare();
      _regularizationBase = tps.getRegularizationBase();
      _isWeightsSet = tps.isWeightsSet();
//      cout << "operator=: tps.getTPSControl().shapeParameter = " << tps.getTPSControl().shapeParameter << ", _tpsControl.shapeParameter = " << _tpsControl.shapeParameter << endl;
      #ifdef __DEBUG_TPS__
        cout << " this->getDataPointsX.getShape()[ 0 ] = " << this->getDataPointsX().getShape()[ 0 ] << endl;
        cout << "ThinPlateSplineBase::operator=(tps) finished" << endl;
      #endif
      return *this;
    }
    
    template< typename ValueT, typename CoordsT >
    bool ThinPlateSplineBase< ValueT, CoordsT >::setCoefficients( ndarray::Array< const double, 1, 1 > const& coefficients ){
      if ( coefficients.getShape()[ 0 ] != this->_coefficients.getShape()[ 0 ] ){
        string message( "ThinPlateSplineBase::setCoefficients: ERROR: coefficients.getShape()[ 0 ](=" );
        message += to_string( coefficients.getShape()[ 0 ] ) + "!= _coefficients.getShape()[0](=" + to_string( this->_coefficients.getShape()[ 0 ] ) + ")";
        cout << message << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
      }
      this->_coefficients.deep() = coefficients;
      return true;
    }
        
    template class ThinPlateSplineBase< float, float >;
    template class ThinPlateSplineBase< float, double >;
    template class ThinPlateSplineBase< double, float >;
    template class ThinPlateSplineBase< double, double >;

    template< typename ValueT, typename CoordsT >
    ThinPlateSpline< ValueT, CoordsT >::ThinPlateSpline( ):
        ThinPlateSplineBase< ValueT, CoordsT >()
    {
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline() finished: _zFit.shape = " << this->_zFit.getShape() << endl;
      #endif
    }
          
    template< typename ValueT, typename CoordsT >
    ThinPlateSpline< ValueT, CoordsT >::ThinPlateSpline( ndarray::Array< const CoordsT, 1, 1 > const& dataPointsX,
                                                         ndarray::Array< const CoordsT, 1, 1 > const& dataPointsY,
                                                         ndarray::Array< const ValueT, 1, 1 > const& dataPointsZ,
                                                         TPSControl const& tpsControl ):
          ThinPlateSplineBase< ValueT, CoordsT >()
    {
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline(dataPointsX, dataPointsY, dataPointsZ, regularization, shapeParameter) started" << endl;
      #endif
      this->_dataPointsX = ndarray::copy( dataPointsX );
      this->_dataPointsY = ndarray::copy( dataPointsY );
      this->_dataPointsZ = ndarray::copy( dataPointsZ );
      this->_dataPointsWeight = pfsDRPStella::utils::get1DndArray( ValueT( dataPointsZ.getShape()[ 0 ] ) );
//      this->_regularization = regularization;
//      this->_shapeParameter = shapeParameter;
      this->_tpsControl = tpsControl;
//      cout << "tps(ndarray, ndarray, ndarray, TPSControl): tpsControl.shapeParameter = " << tpsControl.shapeParameter << ", _tpsControl.shapeParameter = " << this->_tpsControl.shapeParameter << endl;
      
      /// Check input data:
      if (dataPointsX.getShape()[ 0 ] != dataPointsY.getShape()[ 0 ]){
        std::string message( "ThinPlateSpline::ThinPlateSpline: ERROR: dataPointsX.getShape()[0] = " );
        message += std::to_string(dataPointsX.getShape()[ 0 ]) + " != dataPointsY.getShape()[0] = ";
        message += std::to_string(dataPointsY.getShape()[ 0 ]);
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      if (dataPointsX.getShape()[ 0 ] != dataPointsZ.getShape()[ 0 ]){
        std::string message( "ThinPlateSpline::ThinPlateSpline: ERROR: dataPointsX.getShape()[0] = " );
        message += std::to_string(dataPointsX.getShape()[ 0 ]) + " != dataPointsZ.getShape()[0] = ";
        message += std::to_string(dataPointsZ.getShape()[ 0 ]);
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      
      this->_knots = ndarray::allocate( this->_dataPointsX.getShape()[ 0 ], 2 );
      this->_knots[ ndarray::view()( 0 ) ] = this->_dataPointsX;
      this->_knots[ ndarray::view()( 1 ) ] = this->_dataPointsY;
      
      this->_zFit = ndarray::allocate( dataPointsZ.getShape()[ 0 ] );
      if ( this->_zFit.getShape()[0] == 0 ){
        string message( "ThinPlateSpline::ThinPlateSpline: ERROR: _zFit.getShape()[0] == 0" );
        cout << message << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      
      this->_matrix = ndarray::allocate( this->_dataPointsX.getShape()[ 0 ] + 3, this->_dataPointsX.getShape()[ 0 ] + 3 );
      this->_rhs = ndarray::allocate( this->_dataPointsX.getShape()[ 0 ] + 3 );
      
      this->_coefficients = ndarray::allocate( dataPointsZ.getShape()[ 0 ] + 3 );
      if ( !this->calculateCoefficients() ){
        std::string message( "ThinPlateSpline::ThinPlateSpline: ERROR: calculateCoefficients returned FALSE" );
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline(dataPointsX, dataPointsY, dataPointsZ, regularization, shapeParameter) finished" << endl;
      #endif
    }
            
    template< typename ValueT, typename CoordsT >
    ThinPlateSpline< ValueT, CoordsT >::ThinPlateSpline( ndarray::Array< const CoordsT, 1, 1 > const& dataPointsX,
                                                         ndarray::Array< const CoordsT, 1, 1 > const& dataPointsY,
                                                         ndarray::Array< const ValueT, 1, 1 > const& dataPointsZ,
                                                         ndarray::Array< const ValueT, 1, 1 > const& dataPointsWeights,
                                                         TPSControl const& tpsControl ):
            ThinPlateSplineBase< ValueT, CoordsT >()
    {
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline(dataPointsX, dataPointsY, dataPointsZ, dataPointsWeights, shapeParameter) started" << endl;
      #endif
      if ( dataPointsX.getShape()[ 0 ] != dataPointsY.getShape()[ 0 ] ){
        std::string message( "ThinPlateSpline::ThinPlateSpline: ERROR: dataPointsX.getShape()[0] = " );
        message += std::to_string( dataPointsX.getShape()[ 0 ] ) + " != dataPointsY.getShape()[0] = ";
        message += std::to_string( dataPointsY.getShape()[ 0 ] );
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      if ( dataPointsX.getShape()[ 0 ] != dataPointsZ.getShape()[ 0 ] ){
        std::string message( "ThinPlateSpline::ThinPlateSpline: ERROR: dataPointsX.getShape()[0] = " );
        message += std::to_string( dataPointsX.getShape()[ 0 ]) + " != dataPointsZ.getShape()[0] = ";
        message += std::to_string( dataPointsZ.getShape()[ 0 ]);
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      if ( dataPointsX.getShape()[ 0 ] != dataPointsWeights.getShape()[ 0 ] ){
        std::string message("ThinPlateSpline::ThinPlateSpline: ERROR: dataPointsX.getShape()[0] = ");
        message += std::to_string( dataPointsX.getShape()[ 0 ] ) + " != dataPointsWeights.getShape()[0] = ";
        message += std::to_string( dataPointsWeights.getShape()[ 0 ] );
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      
      this->_dataPointsX = ndarray::copy( dataPointsX );
      this->_dataPointsY = ndarray::copy( dataPointsY );
      this->_dataPointsZ = ndarray::copy( dataPointsZ );
      this->_dataPointsWeight = ndarray::copy( dataPointsWeights );
      this->_tpsControl = tpsControl;
//      cout << "tps(ndarray, ndarray, ndarray, ndarray, TPSControl): tpsControl.shapeParameter = " << tpsControl.shapeParameter << ", _tpsControl.shapeParameter = " << this->_tpsControl.shapeParameter << endl;
      this->_zFit = ndarray::allocate( dataPointsZ.getShape()[ 0 ] );
      if ( this->_zFit.getShape()[ 0 ] == 0 ){
        string message( "calculateCoefficients: ERROR: _zFit.getShape()[0] == 0" );
        cout << message << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      
      this->_knots = ndarray::allocate( this->_dataPointsX.getShape()[ 0 ], 2 );
      this->_knots[ ndarray::view()( 0 ) ] = this->_dataPointsX;
      this->_knots[ ndarray::view()( 1 ) ] = this->_dataPointsY;
      
      this->_matrix = ndarray::allocate( this->_dataPointsX.getShape()[ 0 ] + 3, this->_dataPointsX.getShape()[ 0 ] + 3 );
      this->_rhs = ndarray::allocate( this->_dataPointsX.getShape()[ 0 ] + 3 );
      
      this->_coefficients = ndarray::allocate( dataPointsZ.getShape()[ 0 ] + 3 );
      if ( !this->calculateCoefficients() ){
        std::string message( "ThinPlateSpline::ThinPlateSpline: ERROR: calculateCoefficients returned FALSE" );
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      this->_isWeightsSet = true;
      
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline(dataPointsX, dataPointsY, dataPointsZ, dataPointsWeights, shapeParameter) finished" << endl;
      #endif
    }
    
    template< typename ValueT, typename CoordsT >
    ThinPlateSpline< ValueT, CoordsT >::ThinPlateSpline( ThinPlateSpline< ValueT, CoordsT > const& tps ):
        ThinPlateSplineBase< ValueT, CoordsT >( tps )
    { 
      //cout << "tps(tps): tps.getTPSControl().shapeParameter = " << tps.getTPSControl().shapeParameter << ", _tpsControl.shapeParameter = " << this->_tpsControl.shapeParameter << endl;
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline(tps): _coefficients = " << this->_coefficients << endl;
        cout << "ThinPlateSpline(tps) finished" << endl;
      #endif
    }
    
    template< typename ValueT, typename CoordsT >
    void ThinPlateSpline< ValueT, CoordsT >::fillMatrix(){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::fillMatrix() started" << endl;
      #endif
      unsigned p = this->_dataPointsX.getShape()[ 0 ];

      // Fill K (p x p, upper left of L) and calculate
      // mean edge length from control points
      //
      // K is symmetrical so we really have to
      // calculate only about half of the coefficients.
      this->_regularizationBase = 0.0;
      ndarray::Array< double, 1, 1 > pt_i = ndarray::allocate( 2 );
      ndarray::Array< double, 1, 1 > pt_j = ndarray::allocate( 2 );
      ndarray::Array< double, 1, 1 > pt_diff = ndarray::allocate( 2 );
      double elen;
      #ifdef __DEBUG_FILL_MATRIX__
        std::cout << "ThinPlateSpline::fillMatrix: memory for pt_i, pt_j, pt_diff allocated" << std::endl;
      #endif
      for ( unsigned i = 0; i < p; ++i ){
        for ( unsigned j = i + 1; j < p; ++j ){
          pt_i[ 0 ] = this->_dataPointsX[ i ];
          pt_i[ 1 ] = this->_dataPointsY[ i ];
          pt_j[ 0 ] = this->_dataPointsX[ j ];
          pt_j[ 1 ] = this->_dataPointsY[ j ];
          #ifdef __DEBUG_FILL_MATRIX__
            std::cout << "ThinPlateSpline::fillMatrix: i = " << i << ", j = " << j << ": pt_i set to " << pt_i << ", pt_j = " << pt_j << std::endl;
          #endif
          pt_diff.deep() = pt_i - pt_j;
          pt_diff.asEigen() = pt_diff.asEigen().array() * pt_diff.asEigen().array();
          #ifdef __DEBUG_FILL_MATRIX__
            std::cout << "ThinPlateSpline::fillMatrix: i = " << i << ", j = " << j << ": pt_diff set to " << pt_diff << std::endl;
          #endif
          elen = sqrt( pt_diff.asEigen().sum() );
          #ifdef __DEBUG_FILL_MATRIX__
            std::cout << "ThinPlateSpline::fillMatrix: i = " << i << ", j = " << j << ": elen set to " << elen << std::endl;
          #endif
          this->_matrix[ i ][ j ] = this->_matrix[ j ][ i ] = this->tps_base_func( elen );
          this->_regularizationBase += elen * 2; // same for upper & lower tri
          #ifdef __DEBUG_FILL_MATRIX__
            std::cout << "ThinPlateSpline::fillMatrix: i = " << i << ", j = " << j << ": this->_matrix[i][j] set to " << this->_matrix[ i ][ j ] << ", _regularizationBase set to " << this->_regularizationBase << std::endl;
          #endif
        }
      }
      this->_regularizationBase /= ( double )( p * p );
      #ifdef __DEBUG_FILL_MATRIX__
        std::cout << "ThinPlateSpline::fillMatrix: _regularizationBase = " << this->_regularizationBase << std::endl;
      #endif

      // Fill the rest of L
      for ( unsigned i = 0; i < p; ++i ){
        // P (p x 3, upper right)
        // P transposed (3 x p, bottom left)
        this->_matrix[ i ][ p + 0 ] = this->_matrix[ p + 0 ][ i ] = 1.0;
        this->_matrix[ i ][ p + 1 ] = this->_matrix[ p + 1 ][ i ] = this->_dataPointsX[ i ];
        this->_matrix[ i ][ p + 2 ] = this->_matrix[ p + 2 ][ i ] = this->_dataPointsY[ i ];
      }
      // O (3 x 3, lower right)
      for ( unsigned i = p; i < p + 3; ++i ){
        for ( unsigned j = p; j < p + 3; ++j ){
          this->_matrix[ i ][ j ] = 0.0;
        }
      }
      #ifdef __DEBUG_FILL_MATRIX__
        std::cout << "ThinPlateSpline::fillMatrix: this->_matrix = " << this->_matrix << std::endl;

        Eigen::EigenSolver<Eigen::MatrixXd> es( this->_matrix.asEigen() );
        cout << "ThinPlateSpline::fillMatrix: EigenValues of mtx_l = " << es.eigenvalues() << endl;
        cout << "ThinPlateSpline::fillMatrix: EigenVectors of mtx_l = " << es.eigenvectors() << endl;
      #endif
        
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::fillMatrix() finished" << endl;
      #endif
      return;
    }
    
    template< typename ValueT, typename CoordsT >
    void ThinPlateSpline< ValueT, CoordsT >::fillRHS(){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::fillRHS() started" << endl;
      #endif
      unsigned p = this->_dataPointsX.getShape()[ 0 ];

      // Allocate the matrix and vector
      #ifdef __DEBUG_CALCULATE_COEFFICIENTS__
        std::cout << "ThinPlateSpline::fillRHSVector: memory for rhs allocated" << std::endl;
      #endif
      for ( unsigned i = 0; i < p; ++i )
        this->_rhs[ i ] = this->_dataPointsZ[ i ];
      this->_rhs[ p ] = this->_rhs[ p + 1 ] = this->_rhs[ p + 2 ] = 0.0;
      #ifdef __DEBUG_CALCULATE_COEFFICIENTS__
        std::cout << "ThinPlateSpline::fillRHSVector: this->_rhs = " << this->_rhs << std::endl;
      #endif
      return;
    }
    
/*    template< typename ValueT, typename CoordsT >
    ThinPlateSpline< ValueT, CoordsT >& ThinPlateSpline< ValueT, CoordsT >::operator=(ThinPlateSpline< ValueT, CoordsT > const& tps){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::operator=(tps) started" << endl;
        cout << " tps.getControlPointsX.getShape()[ 0 ] = " << tps.getControlPointsX().getShape()[ 0 ] << endl;
      #endif
      ThinPlateSplineBase::operator=( tps );
      #ifdef __DEBUG_TPS__
        cout << " this->getControlPointsX.getShape()[ 0 ] = " << this->getControlPointsX().getShape()[ 0 ] << endl;
        cout << "ThinPlateSplineBase::operator=(tps) finished" << endl;
      #endif
      return *this;
    }*/
    
    template class ThinPlateSpline< float, float >;
    template class ThinPlateSpline< float, double >;
    template class ThinPlateSpline< double, float >;
    template class ThinPlateSpline< double, double >;

    template< typename ValueT, typename CoordsT >
    ThinPlateSplineChiSquare< ValueT, CoordsT >::ThinPlateSplineChiSquare( ):
            ThinPlateSplineBase< ValueT, CoordsT >( )
    {
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare() finished" << endl;
      #endif
    }

    template< typename ValueT, typename CoordsT >
    ThinPlateSplineChiSquare< ValueT, CoordsT >::ThinPlateSplineChiSquare( ndarray::Array< const CoordsT, 1, 1 > const& dataPointsX,
                                                                           ndarray::Array< const CoordsT, 1, 1 > const& dataPointsY,
                                                                           ndarray::Array< const ValueT, 1, 1 > const& dataPointsZ,
                                                                           ndarray::Array< const CoordsT, 1, 1 > const& knotsX,
                                                                           ndarray::Array< const CoordsT, 1, 1 > const& knotsY,
                                                                           bool const isXYPositionsGridPoints,
                                                                           TPSControl const& tpsControl ):
            ThinPlateSplineBase< ValueT, CoordsT >( )
    {
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare(dataPointsX, dataPointsY, dataPointsZ, knotsX, knotsY, isXYPositionsGridPoints, regularization, shapeParameter) started" << endl;
      #endif
      /// Check input data:
      if ( dataPointsX.getShape()[ 0 ] != dataPointsY.getShape()[ 0 ] ){
        std::string message( "ThinPlateSpline::ThinPlateSplineChiSquare: ERROR: dataPointsX.getShape()[0] = " );
        message += std::to_string( dataPointsX.getShape()[ 0 ] ) + " != dataPointsY.getShape()[0] = ";
        message += std::to_string( dataPointsY.getShape()[ 0 ] );
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      if ( dataPointsX.getShape()[ 0 ] != dataPointsZ.getShape()[ 0 ] ){
        std::string message( "ThinPlateSpline::ThinPlateSplineChiSquare: ERROR: dataPointsX.getShape()[0] = " );
        message += std::to_string( dataPointsX.getShape()[ 0 ] ) + " != dataPointsZ.getShape()[0] = ";
        message += std::to_string( dataPointsZ.getShape()[ 0 ] );
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      this->_dataPointsX = ndarray::copy( dataPointsX );
      this->_dataPointsY = ndarray::copy( dataPointsY );
      this->_dataPointsZ = ndarray::copy( dataPointsZ );
      this->_tpsControl = tpsControl;
      //cout << "tpscs(ndarray, ndarray, ndarray, ndarray, ndarray, bool, TPSControl): tpsControl.shapeParameter = " << tpsControl.shapeParameter << ", _tpsControl.shapeParameter = " << this->_tpsControl.shapeParameter << endl;
      
      if (isXYPositionsGridPoints){
        this->_knots = ndarray::allocate( knotsX.getShape()[ 0 ] * knotsY.getShape()[ 0 ], 2 );
        createGridPointsXY( knotsX, knotsY);
        this->_coefficients = ndarray::allocate( knotsX.getShape()[ 0 ] * knotsY.getShape()[ 0 ] + 6);
        this->_matrix = ndarray::allocate( knotsX.getShape()[ 0 ] * knotsY.getShape()[ 0 ] + 6, knotsX.getShape()[ 0 ] * knotsY.getShape()[ 0 ] + 6 );
        this->_rhs = ndarray::allocate( knotsX.getShape()[ 0 ] * knotsY.getShape()[ 0 ] + 6 );
      }
      else{
        if ( knotsX.getShape()[ 0 ] != knotsY.getShape()[ 0 ]){
          std::string message( "ThinPlateSpline::ThinPlateSplineChiSquare: ERROR: knotsX.getShape()[0] = " );
          message += std::to_string( knotsX.getShape()[ 0 ] ) + " != knotsY.getShape()[0] = ";
          message += std::to_string( knotsY.getShape()[ 0 ] );
          std::cout << message << std::endl;
          throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
        }
        this->_knots = ndarray::allocate( knotsX.getShape()[ 0 ], 2 );
        for (size_t iPix = 0; iPix < knotsX.getShape()[ 0 ]; ++iPix ){
          this->_knots[ iPix ][ 0 ] = knotsX[ iPix ];
          this->_knots[ iPix ][ 1 ] = knotsY[ iPix ];
        }
        this->_coefficients = ndarray::allocate( knotsX.getShape()[ 0 ] + 6);
        this->_matrix = ndarray::allocate( knotsX.getShape()[ 0 ] + 6, knotsX.getShape()[ 0 ] + 6 );
        this->_rhs = ndarray::allocate( knotsX.getShape()[ 0 ] + 6 );
      }
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSpline::ThinPlateSplineChiSquare: _knots = " << this->_knots << endl;
      #endif
      this->_zFit = ndarray::allocate( dataPointsZ.getShape()[ 0 ] );
      
      if ( !this->calculateCoefficients() ){
        std::string message( "ThinPlateSpline::ThinPlateSplineChiSquare: ERROR: calculateCoefficients returned FALSE" );
        std::cout << message << std::endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare(dataPointsX, dataPointsY, dataPointsZ, knotsX, knotsY, isXYPositionsGridPoints, regularization, shapeParameter) finished" << endl;
      #endif
    }
    
    template< typename ValueT, typename CoordsT >
    ThinPlateSplineChiSquare< ValueT, CoordsT >::ThinPlateSplineChiSquare( ThinPlateSplineChiSquare< ValueT, CoordsT > const& tps ):
            ThinPlateSplineBase< ValueT, CoordsT >( tps )
    { 
      //cout << "tpscs(tpscs): tps.getTPSControl().shapeParameter = " << tps.getTPSControl().shapeParameter << ", _tpsControl.shapeParameter = " << this->_tpsControl.shapeParameter << endl;
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare(tps) finished" << endl;
      #endif
    }
            
    template< typename ValueT, typename CoordsT >
    bool ThinPlateSplineChiSquare< ValueT, CoordsT >::calculateCoefficients(){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::calculateCoefficients() started" << endl;
      #endif
      #ifdef __DEBUG_CALC_TPS__
        cout << "ThinPlateSplineChiSquare::calculateCoefficients: _dataPointsX = (" << this->_dataPointsX.getShape()[0] << ") = " << this->_dataPointsX << endl;
        cout << "ThinPlateSplineChiSquare::calculateCoefficients: _dataPointsY = (" << this->_dataPointsY.getShape()[0] << ") = " << this->_dataPointsY << endl;
        cout << "ThinPlateSplineChiSquare::calculateCoefficients: _dataPointsZ = (" << this->_dataPointsZ.getShape()[0] << ") = " << this->_dataPointsZ << endl;
      #endif

      // You We need at least 3 points to define a plane
      if ( this->_dataPointsX.getShape()[ 0 ] < 3 ){
        string message( "ThinPlateSplineChiSquare::calculateCoefficients: ERROR: _dataPointsX.getShape()[0] = " );
        message += to_string( this->_dataPointsX.getShape()[ 0 ] ) + " < 3";
        cout << message << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );    
      }

      unsigned nKnots = this->_knots.getShape()[ 0 ];
      //unsigned nDataPoints = this->_dataPointsX.getShape()[ 0 ];

      // Allocate the matrix and vector
      this->fillMatrix();//ndarray::allocate( nKnots + 6, nKnots + 6);
      this->fillRHS();//ndarray::allocate( nKnots + 6);
//      mtx_v.deep() = 0.;
//      #ifdef __DEBUG_CALC_TPS__
//        std::cout << "ThinPlateSplineChiSquare::calculateCoefficients: memory for mtx_l, mtx_v allocated" << std::endl;
//      #endif
//
//      mtx_l.deep() = fillMatrix();

      // Fill the right hand vector V
/*      ndarray::Array< double, 1, 1 > pt_i = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_k = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_diff = ndarray::allocate(2);
      double elen_ik;
//      double r_ik;
      for ( unsigned i = 0; i < nKnots; ++i ){
        pt_i[ 0 ] = this->_knots[ i ][ 0 ];
        pt_i[ 1 ] = this->_knots[ i ][ 1 ];
        for ( unsigned k = 0; k < nDataPoints; ++k ){
          pt_k[ 0 ] = this->_dataPointsX[ k ];
          pt_k[ 1 ] = this->_dataPointsY[ k ];
          pt_diff.deep() = pt_i - pt_k;
          pt_diff.asEigen() = pt_diff.asEigen().array() * pt_diff.asEigen().array();
          elen_ik = sqrt( pt_diff.asEigen().sum() );
          mtx_v[ i ] += 2. * this->tps_base_func( elen_ik );// - _dataPointsZ[ k ];
        }
      }
      mtx_v[ nKnots ] = 2. * double( nDataPoints );
      for (unsigned k = 0; k < nDataPoints; ++k ){
        mtx_v[ nKnots + 1 ] += 2. * this->_dataPointsX[ k ];
        mtx_v[ nKnots + 2 ] += 2. * this->_dataPointsY[ k ];
      }
      //mtx_v[ nKnots + 3 ] = mtx_v[ nKnots + 4 ] = mtx_v[ nKnots + 5 ] = 0.;
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSplineChiSquare::calculateCoefficients: mtx_v = " << mtx_v << std::endl;
      #endif
*/
      #ifdef __DEBUG_CALCUATE_COEFFICIENTS__
        Eigen::EigenSolver<Eigen::MatrixXd> es( this->_matrix.asEigen() );
//      ndarray::Array< double, 2, 1 > eigenValues = ndarray::allocate( nKnots, 2 );
//      eigenValues.asEigen() = es.eigenvalues();
        cout << "ThinPlateSpline::calculateCoefficients: EigenValues of mtx_l = " << es.eigenvalues() << endl;
        cout << "ThinPlateSpline::calculateCoefficients: EigenVectors of mtx_l = " << es.eigenvectors() << endl;
      #endif
      // Solve the linear system "inplace"
//      this->_coefficients.asEigen() = this->_matrix.asEigen().fullPivLu().solve( this->_rhs.asEigen() );
      this->_coefficients.asEigen() = this->_matrix.asEigen().colPivHouseholderQr().solve( this->_rhs.asEigen() );
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSplineChiSquare::calculateCoefficients: after colPivHouseholderQr: _coefficients = " << this->_coefficients << std::endl;
      #endif

      // Calculate bending energy
      ndarray::Array< double, 1, 1 > coeffs_dot_mtxl = ndarray::allocate( nKnots );
      coeffs_dot_mtxl.deep() = 0.;
      for (int i = 0; i < nKnots; ++i){
        for (int j = 0; j < nKnots; ++j)
          coeffs_dot_mtxl[ i ] += this->_coefficients[ j ] * this->_matrix[ j ][ i ];
      }  
      this->_bendingEnergy = 0.;
      for (int i = 0; i < nKnots; ++i)
        this->_bendingEnergy += this->_coefficients[ i ] * coeffs_dot_mtxl[ i ];
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSplineChiSquare::calculateCoefficients: _bendingEnergy = " << _bendingEnergy << std::endl;
      #endif
        
      // fit input data
      ndarray::Array< ValueT, 2, 1 > zFit = this->fitArray( this->_dataPointsX, this->_dataPointsY, false );
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSplineChiSquare::calculateCoefficients: zFit calculated: zFit.getShape() = " << zFit.getShape() << ", _zFit.getShape() = " << _zFit.getShape() << std::endl;
      #endif
      this->_zFit.deep() = zFit[ ndarray::view()( 0 ) ];
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSplineChiSquare::calculateCoefficients: _zFit assigned: _zFit.getShape() = " << this->_zFit.getShape() << std::endl;
      #endif

      // Calculate ChiSquare
      this->_chiSquare = 0.;
      auto itZFit = this->_zFit.begin();
      int iControlPoint = 0;
      for ( auto itZIn = this->_dataPointsZ.begin(); itZIn != this->_dataPointsZ.end(); ++itZIn, ++itZFit ){
        this->_chiSquare += ( *itZIn - *itZFit ) * ( *itZIn - *itZFit ) / std::fabs(*itZIn);
        #ifdef __DEBUG_CALC_TPS__
          std::cout << "cThinPlateSplineChiSquare::alculateCoefficients: iControlPoint = " << iControlPoint << ": _chiSquare = " << this->_chiSquare << std::endl;
        #endif
        ++iControlPoint;
      }
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "cThinPlateSplineChiSquare::alculateCoefficients: _chiSquare = " << this->_chiSquare << std::endl;
      #endif
      
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
          this->_knots[ row ][ 0 ] = gridPointsX[ x ];
          this->_knots[ row ][ 1 ] = gridPointsY[ y ];
          ++row;
        }
      }
      #ifdef __DEBUG_CALC_TPS__
        std::cout << "ThinPlateSplineChiSquare::createGridPointsXY: _knots = " << this->_knots << std::endl;
      #endif
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::createGridPointsXY(gridPointsX, gridPointsY) finished" << endl;
      #endif
    }
    
    template< typename ValueT, typename CoordsT >
    void ThinPlateSplineChiSquare< ValueT, CoordsT >::fillMatrix(){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::fillMatrix() started" << endl;
      #endif
      unsigned nKnots = this->_knots.getShape()[ 0 ];
      unsigned nDataPoints = this->_dataPointsX.getShape()[ 0 ];
      this->_regularizationBase = 0.0;
      
      double sumXi = 0.;
      double sumYi = 0.;
      for (unsigned i = 0; i < nKnots; ++i){
        sumXi += this->_knots[ i ][ 0 ];
        sumYi += this->_knots[ i ][ 1 ];
      }

      this->_matrix.deep() = 0.;

      ndarray::Array< double, 1, 1 > pt_i = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_j = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_k = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_diff = ndarray::allocate(2);
      double elen, elen_ik, elen_jk, tpsBaseValue, tpsBaseValueSquared;
      // Fill K (p x p)
      //
      // K is symmetrical so we really have to
      // calculate only about half of the coefficients.
      for ( unsigned i = 0; i < nKnots; ++i ){
        pt_i[ 0 ] = this->_knots[ i ][ 0 ];
        pt_i[ 1 ] = this->_knots[ i ][ 1 ];
        #ifdef __DEBUG_FILL_MATRIX__
          std::cout << "ThinPlateSplineChiSquare::fillMatrix: i = " << i << ": pt_i set to " << pt_i << std::endl;
        #endif
        for (unsigned k = 0; k < nDataPoints; ++k){
          pt_k[ 0 ] = this->_dataPointsX[ k ];
          pt_k[ 1 ] = this->_dataPointsY[ k ];
          #ifdef __DEBUG_FILL_MATRIX__
            std::cout << "ThinPlateSplineChiSquare::fillMatrix: i = " << i << ", k = " << k << ": pt_k set to " << pt_k << std::endl;
          #endif

          pt_diff.deep() = pt_i - pt_k;
          #ifdef __DEBUG_FILL_MATRIX__
            std::cout << "ThinPlateSplineChiSquare::fillMatrix: i = " << i << ", k = " << k << ": pt_diff set to " << pt_diff << std::endl;
          #endif
          pt_diff.asEigen() = pt_diff.asEigen().array() * pt_diff.asEigen().array();
          #ifdef __DEBUG_FILL_MATRIX__
            std::cout << "ThinPlateSplineChiSquare::fillMatrix: i = " << i << ", k = " << k << ": pt_diff set to " << pt_diff << std::endl;
          #endif

          elen = sqrt( pt_diff.asEigen().sum() );
          #ifdef __DEBUG_FILL_MATRIX__
            std::cout << "ThinPlateSplineChiSquare::fillMatrix: i = " << i << ", k = " << k << ": elen set to " << elen << std::endl;
          #endif

          tpsBaseValue = this->tps_base_func( elen );
          tpsBaseValueSquared = tpsBaseValue * tpsBaseValue;
          this->_matrix[ i ][ i ] += tpsBaseValueSquared / this->_dataPointsZ[ k ];
        }
        this->_matrix[ i ][ i ] = 2. * this->_matrix[ i ][ i ] + this->_tpsControl.regularization;
        #ifdef __DEBUG_FILL_MATRIX__
          std::cout << "ThinPlateSplineChiSquare::fillMatrix: this->_matrix[ " << i << " ][ " << i << " ] set to " << this->_matrix[ i ][ i ] << std::endl;
        #endif
        
        for ( unsigned j = i + 1; j < nKnots; ++j ){
          pt_j[ 0 ] = this->_knots[ j ][ 0 ];
          pt_j[ 1 ] = this->_knots[ j ][ 1 ];
          for ( unsigned k = 0; k < nDataPoints; ++k ){
            pt_k[0] = this->_dataPointsX[ k ];
            pt_k[1] = this->_dataPointsY[ k ];
            #ifdef __DEBUG_FILL_MATRIX__
              std::cout << "ThinPlateSplineChiSquare::fillMatrix: i = " << i << ", j = " << j << ": pt_i set to " << pt_i << ", pt_j = " << pt_j << ", pt_k = " << pt_k << std::endl;
            #endif
            pt_diff.deep() = pt_i - pt_k;
            pt_diff.asEigen() = pt_diff.asEigen().array() * pt_diff.asEigen().array();
            #ifdef __DEBUG_FILL_MATRIX__
              std::cout << "ThinPlateSplineChiSquare::fillMatrix: i = " << i << ", j = " << j << ", k = " << k << ": pt_diff set to " << pt_diff << std::endl;
            #endif
            elen_ik = sqrt( pt_diff.asEigen().sum() );
            #ifdef __DEBUG_FILL_MATRIX__
              std::cout << "ThinPlateSplineChiSquare::fillMatrix: i = " << i << ", j = " << j << ", k = " << k << ": elen_ik set to " << elen_ik << std::endl;
            #endif

            pt_diff.deep() = pt_j - pt_k;
            pt_diff.asEigen() = pt_diff.asEigen().array() * pt_diff.asEigen().array();
            #ifdef __DEBUG_FILL_MATRIX__
              std::cout << "ThinPlateSplineChiSquare::fillMatrix: i = " << i << ", j = " << j << ", k = " << k << ": pt_diff set to " << pt_diff << std::endl;
            #endif
            elen_jk = sqrt( pt_diff.asEigen().sum() );
            #ifdef __DEBUG_FILL_MATRIX__
              std::cout << "ThinPlateSplineChiSquare::fillMatrix: i = " << i << ", j = " << j << ", k = " << k << ": elen_jk set to " << elen_jk << std::endl;
            #endif
            this->_matrix[ i ][ j ] += this->tps_base_func( elen_ik ) * this->tps_base_func( elen_jk ) / this->_dataPointsZ[ k ];
          }
          this->_matrix[ i ][ j ] = 2. * this->_matrix[ i ][ j ];
          this->_matrix[ j ][ i ] = this->_matrix[ i ][ j ];
          #ifdef __DEBUG_FILL_MATRIX__
            std::cout << "ThinPlateSplineChiSquare::fillMatrix: i = " << i << ", j = " << j << ": this->_matrix[i][j] set to " << this->_matrix[i][j] << std::endl;
          #endif
        }
        
        // P (p x 3, upper right)
        // P transposed (3 x p, bottom left)
        //*
        for ( unsigned k = 0; k < nDataPoints; ++k ){
          pt_k[0] = this->_dataPointsX[k];
          pt_k[1] = this->_dataPointsY[k];
          #ifdef __DEBUG_FILL_MATRIX__
            std::cout << "ThinPlateSplineChiSquare::fillMatrix: i = " << i << ", k = " << k << ": pt_i set to " << pt_i << ", pt_j = " << pt_j << ", pt_k = " << pt_k << std::endl;
          #endif
          pt_diff.deep() = pt_i - pt_k;
          pt_diff.asEigen() = pt_diff.asEigen().array() * pt_diff.asEigen().array();
          #ifdef __DEBUG_FILL_MATRIX__
            std::cout << "ThinPlateSplineChiSquare::fillMatrix: i = " << i << ", k = " << k << ": pt_diff set to " << pt_diff << std::endl;
          #endif
          elen_ik = sqrt( pt_diff.asEigen().sum() );
          #ifdef __DEBUG_FILL_MATRIX__
            std::cout << "ThinPlateSplineChiSquare::fillMatrix: i = " << i << ", k = " << k << ": elen_ik set to " << elen_ik << std::endl;
          #endif
          this->_matrix[ i ][ nKnots + 0 ] += this->tps_base_func( elen_ik)  / this->_dataPointsZ[ k ];
          this->_matrix[ i ][ nKnots + 1 ] += this->_dataPointsX[ k ] * this->tps_base_func( elen_ik ) / this->_dataPointsZ[ k ];
          this->_matrix[ i ][ nKnots + 2 ] += this->_dataPointsY[ k ] * this->tps_base_func( elen_ik ) / this->_dataPointsZ[ k ];
        }
        this->_matrix[ i ][ nKnots + 0 ] = 2. * this->_matrix[ i ][ nKnots + 0 ];
        this->_matrix[ i ][ nKnots + 1 ] = 2. * this->_matrix[ i ][ nKnots + 1 ];
        this->_matrix[ i ][ nKnots + 2 ] = 2. * this->_matrix[ i ][ nKnots + 2 ];
        this->_matrix[ nKnots + 0 ][ i ] = this->_matrix[ i ][ nKnots + 0 ];
        this->_matrix[ nKnots + 1 ][ i ] = this->_matrix[ i ][ nKnots + 1 ];
        this->_matrix[ nKnots + 2 ][ i ] = this->_matrix[ i ][ nKnots + 2 ];
        
        this->_matrix[ i ][ nKnots + 3 ] = this->_matrix[ nKnots + 3 ][ i ] = 1.;
        this->_matrix[ i ][ nKnots + 4 ] = this->_matrix[ nKnots + 4 ][ i ] = this->_knots[ i ][ 0 ];
        this->_matrix[ i ][ nKnots + 5 ] = this->_matrix[ nKnots + 5 ][ i ] = this->_knots[ i ][ 1 ];
      }
      //*
      // (3 x 3, lower right)
      for ( unsigned k = 0; k < nDataPoints; ++k ){
        this->_matrix[ nKnots + 0 ][ nKnots + 0 ] += 2. / this->_dataPointsZ[ k ];
        this->_matrix[ nKnots + 1 ][ nKnots + 1 ] += this->_dataPointsX[ k ] * this->_dataPointsX[ k ] / this->_dataPointsZ[ k ];
        this->_matrix[ nKnots + 2 ][ nKnots + 2 ] += this->_dataPointsY[ k ] * this->_dataPointsY[ k ] / this->_dataPointsZ[ k ];
        this->_matrix[ nKnots + 0 ][ nKnots + 1 ] += this->_dataPointsX[ k ] / this->_dataPointsZ[ k ];
        this->_matrix[ nKnots + 0 ][ nKnots + 2 ] += this->_dataPointsY[ k ] / this->_dataPointsZ[ k ];
        this->_matrix[ nKnots + 1 ][ nKnots + 2 ] += this->_dataPointsX[ k ] * this->_dataPointsY[ k ] / this->_dataPointsZ[ k ];
      }
      this->_matrix[ nKnots + 1 ][ nKnots + 1 ] = 2. * this->_matrix[ nKnots + 1 ][ nKnots + 1 ];
      this->_matrix[ nKnots + 2 ][ nKnots + 2 ] = 2. * this->_matrix[ nKnots + 2 ][ nKnots + 2 ];
      this->_matrix[ nKnots + 0 ][ nKnots + 1 ] = 2. * this->_matrix[ nKnots + 0 ][ nKnots + 1 ];
      this->_matrix[ nKnots + 0 ][ nKnots + 2 ] = 2. * this->_matrix[ nKnots + 0 ][ nKnots + 2 ];
      this->_matrix[ nKnots + 1 ][ nKnots + 2 ] = 2. * this->_matrix[ nKnots + 1 ][ nKnots + 2 ];
      
      this->_matrix[ nKnots + 1 ][ nKnots + 0 ] = this->_matrix[ nKnots + 0 ][ nKnots + 1 ];
      this->_matrix[ nKnots + 2 ][ nKnots + 0 ] = this->_matrix[ nKnots + 0 ][ nKnots + 2 ];
      this->_matrix[ nKnots + 2 ][ nKnots + 1 ] = this->_matrix[ nKnots + 1 ][ nKnots + 2 ];
      //*/

        
/*        
        
        // Fill the rest of L

        // P (p x 3, upper right)
        // P transposed (3 x p, bottom left)
      for ( unsigned i = 0; i < nKnots; ++i ){
        mtx_l[ i ][ nKnots + 0 ] = mtx_l[ nKnots + 0 ][ i ] = 1.0;
        mtx_l[ i ][ nKnots + 1] = mtx_l[ nKnots + 1 ][ i ] = _knots[ i ][ 0 ];
        mtx_l[ i ][ nKnots + 2] = mtx_l[ nKnots + 2][ i ] = _knots[ i ][ 1 ];
      
        // O (3 x 3, lower right)
        for ( unsigned i = nKnots; i < nKnots + 3; ++i ){
          for ( unsigned j = nKnots; j < nKnots + 3; ++j ){
            mtx_l[i][j] = 0.0;
          }
        }
      }        
      
*/      
      
      
      #ifdef __DEBUG_FILL_MATRIX__
        std::cout << "ThinPlateSplineChiSquare::fillMatrix: this->_matrix set to " << this->_matrix << std::endl;

        Eigen::EigenSolver<Eigen::MatrixXd> es( this->_matrix.asEigen() );
        cout << "ThinPlateSplineChiSquare::fillMatrix: EigenValues of mtx_l = " << es.eigenvalues() << endl;
        cout << "ThinPlateSplineChiSquare::fillMatrix: EigenVectors of mtx_l = " << es.eigenvectors() << endl;
      #endif
        
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::fillMatrix() finished" << endl;
      #endif
      return;
    }
    
    template< typename ValueT, typename CoordsT >
    void ThinPlateSplineChiSquare< ValueT, CoordsT >::fillRHS(){
      #ifdef __DEBUG_TPS__
        cout << "ThinPlateSplineChiSquare::fillRHS() started" << endl;
      #endif
      unsigned nKnots = this->_knots.getShape()[ 0 ];
      unsigned nDataPoints = this->_dataPointsX.getShape()[ 0 ];

      this->_rhs.deep() = 0.;
      #ifdef __DEBUG_FILL_RHS__
        std::cout << "ThinPlateSplineChiSquare::fillRHSVector: memory for rhs allocated" << std::endl;
      #endif

      ndarray::Array< double, 1, 1 > pt_i = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_k = ndarray::allocate(2);
      ndarray::Array< double, 1, 1 > pt_diff = ndarray::allocate(2);
      double elen_ik;
      for ( unsigned i = 0; i < nKnots; ++i ){
        pt_i[ 0 ] = this->_knots[ i ][ 0 ];
        pt_i[ 1 ] = this->_knots[ i ][ 1 ];
        for ( unsigned k = 0; k < nDataPoints; ++k ){
          pt_k[ 0 ] = this->_dataPointsX[ k ];
          pt_k[ 1 ] = this->_dataPointsY[ k ];
          pt_diff.deep() = pt_i - pt_k;
          pt_diff.asEigen() = pt_diff.asEigen().array() * pt_diff.asEigen().array();
          elen_ik = sqrt( pt_diff.asEigen().sum() );
          this->_rhs[ i ] += this->tps_base_func( elen_ik );// - _dataPointsZ[ k ];
        }
        this->_rhs[ i ] = 2. * this->_rhs[ i ];
      }
      this->_rhs[ nKnots + 0 ] = 2. * double( nDataPoints );
      for (unsigned k = 0; k < nDataPoints; ++k ){
        this->_rhs[ nKnots + 1 ] += this->_dataPointsX[ k ];
        this->_rhs[ nKnots + 2 ] += this->_dataPointsY[ k ];
      }
      this->_rhs[ nKnots + 1 ] = 2. * this->_rhs[ nKnots + 1 ];
      this->_rhs[ nKnots + 2 ] = 2. * this->_rhs[ nKnots + 2 ];
      #ifdef __DEBUG_FILL_RHS__
        std::cout << "ThinPlateSplineChiSquare::fillRHSVector: this->_rhs = " << this->_rhs << std::endl;
      #endif
      return;
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
