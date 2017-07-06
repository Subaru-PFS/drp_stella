#include "pfs/drp/stella/PSF.h"

namespace pfs{ namespace drp{ namespace stella{

  /// Set the _twoDPSFControl
  template<typename T>
  void PSF<T>::setTwoDPSFControl(PTR(TwoDPSFControl) &twoDPSFControl){
    #ifdef __DEBUG_PSF__
      cout << "PSF::Constructor(TwoDPSFControl) started" << endl;
    #endif
    assert(twoDPSFControl->signalThreshold >= 0.);
    assert(twoDPSFControl->signalThreshold < twoDPSFControl->saturationLevel / 2.);
    assert(twoDPSFControl->swathWidth > twoDPSFControl->yFWHM * 10);
    assert(twoDPSFControl->xFWHM * 4. > twoDPSFControl->nTermsGaussFit);
    assert(twoDPSFControl->yFWHM * 4. > twoDPSFControl->nTermsGaussFit);
    assert(twoDPSFControl->nTermsGaussFit > 2);
    assert(twoDPSFControl->nTermsGaussFit < 6);
    assert(twoDPSFControl->saturationLevel > 0.);
    assert(twoDPSFControl->nKnotsX > 10);
    assert(twoDPSFControl->nKnotsY > 10);
    _twoDPSFControl->signalThreshold = twoDPSFControl->signalThreshold;
    _twoDPSFControl->swathWidth = twoDPSFControl->swathWidth;
    _twoDPSFControl->xFWHM = twoDPSFControl->xFWHM;
    _twoDPSFControl->yFWHM = twoDPSFControl->yFWHM;
    _twoDPSFControl->nTermsGaussFit = twoDPSFControl->nTermsGaussFit;
    _twoDPSFControl->saturationLevel = twoDPSFControl->saturationLevel;
    _twoDPSFControl->nKnotsX = twoDPSFControl->nKnotsX;
    _twoDPSFControl->nKnotsY = twoDPSFControl->nKnotsY;
    _isTwoDPSFControlSet = true;
    #ifdef __DEBUG_PSF__
      cout << "PSF::Constructor(TwoDPSFControl) finished" << endl;
    #endif
  }

  template< typename T >
  typename PSF<T>::Vector PSF< T >::reconstructFromThinPlateSplineFit(double const regularization) const {
    #ifdef __DEBUG_PSF__
      cout << "PSF::reconstructFromThinPlateSplineFit(regularization) started" << endl;
    #endif
    Vector const& xArr = _imagePSF_XRelativeToCenter;
    Vector const& yArr = _imagePSF_YRelativeToCenter;
    Vector const& zArr = _imagePSF_ZNormalized;
    TPSControl tpsControl;
    tpsControl.regularization = regularization;
    Spline tps = Spline(xArr, yArr, zArr, tpsControl);
    Image zFitArr = tps.fitArray( xArr, yArr, false);
    Vector zVec( _imagePSF_XRelativeToCenter.size() );
    for (int i = 0; i < _imagePSF_XRelativeToCenter.size(); ++i)
      zVec[ i ] = T( zFitArr[ ndarray::makeVector( i, 0 ) ] );
    #ifdef __DEBUG_PSF__
      cout << "PSF::reconstructFromThinPlateSplineFit(regularization) finished" << endl;
    #endif
    return zVec;
  }

  template< typename T >
  void PSF< T >::setImagePSF_ZFit(Vector const& zFit){
    #ifdef __DEBUG_PSF__
      cout << "PSF::setImagePSF_ZFit(zFit) started" << endl;
    #endif
    if (zFit.getShape()[0] != _imagePSF_XRelativeToCenter.size()){
      cout << "PSF::setImagePSF_ZFit: zFit.getShape()[0]=" << zFit.getShape()[0] << " != _imagePSF_XRelativeToCenter.size()=" << _imagePSF_XRelativeToCenter.size() << " => returning FALSE" << endl;
    }
    _imagePSF_ZFit = ndarray::copy(zFit);
    #ifdef __DEBUG_PSF__
      cout << "PSF::setImagePSF_ZFit(zFit) finished" << endl;
    #endif
  }

  template< typename T >
  void PSF< T >::setImagePSF_ZTrace(Vector const& zTrace){
    #ifdef __DEBUG_PSF__
      cout << "PSF::setImagePSF_ZTrace(zTrace) started" << endl;
    #endif
    if (zTrace.getShape()[0] != _imagePSF_XRelativeToCenter.size()){
      cout << "PSF::setImagePSF_ZTrace: zTrace.getShape()[0]=" << zTrace.getShape()[0] << " != _imagePSF_XRelativeToCenter.size()=" << _imagePSF_XRelativeToCenter.size() << " => returning FALSE" << endl;
      throw LSST_EXCEPT(pexExcept::LengthError, "Size mismatch");
    }
    _imagePSF_ZTrace = ndarray::copy(zTrace);
    #ifdef __DEBUG_PSF__
      cout << "PSF::setImagePSF_ZTrace(zTrace) finished" << endl;
    #endif
  }

  template< typename T >
  void PSF< T >::setImagePSF_ZNormalized(Vector const& zNormalized){
    #ifdef __DEBUG_PSF__
      cout << "PSF::setImagePSF_ZNormalized(zNormalized) started" << endl;
    #endif
    if (zNormalized.getShape()[0] != _imagePSF_XRelativeToCenter.size()){
      cout << "PSF::setImagePSF_ZNormalized: zNormalized.getShape()[0]=" << zNormalized.getShape()[0] << " != _imagePSF_XRelativeToCenter.size()=" << _imagePSF_XRelativeToCenter.size() << " => returning FALSE" << endl;
      throw LSST_EXCEPT(pexExcept::LengthError, "Size mismatch");
    }
    _imagePSF_ZNormalized = ndarray::copy(zNormalized);
    #ifdef __DEBUG_PSF__
      cout << "PSF::setImagePSF_ZNormalized(zNormalized) finished" << endl;
    #endif
  }

  template< typename T >
  void PSF< T >::setXCentersPSFCCD(Vector const& xCentersPSFCCD_In){
    #ifdef __DEBUG_PSF__
      cout << "PSF::setXCentersPSFCCD(xCentersPSFCCD) started" << endl;
    #endif
    _xCentersPSFCCD = xCentersPSFCCD_In;
    #ifdef __DEBUG_PSF__
      cout << "PSF::setXCentersPSFCCD(xCentersPSFCCD) finished" << endl;
    #endif
  }

  template< typename T >
  void PSF< T >::setYCentersPSFCCD(Vector const& yCentersPSFCCD_In){
    #ifdef __DEBUG_PSF__
      cout << "PSF::setYCentersPSFCCD(yCentersPSFCCD) started" << endl;
    #endif
    _yCentersPSFCCD = yCentersPSFCCD_In;
    #ifdef __DEBUG_PSF__
      cout << "PSF::setYCentersPSFCCD(yCentersPSFCCD) started" << endl;
    #endif
  }

  template< typename T >
  typename PSF<T>::Vector PSF<T>::reconstructPSFFromFit(
      Vector const& xGridRelativeToCenterFit_In,
      Vector const& yGridRelativeToCenterFit_In,
      Image const& zFit_In,
      double regularization
      ) const
  {
    #ifdef __DEBUG_PSF__
      cout << "PSF::reconstructPSFFromFit(xGridRelativeToCenterFit, yGridRelativeToCenterFit, zFit, regularization) started" << endl;
    #endif
    int nX = xGridRelativeToCenterFit_In.getShape()[ 0 ];
    int nY = yGridRelativeToCenterFit_In.getShape()[ 0 ];
    if ( zFit_In.getShape()[ 0 ] * zFit_In.getShape()[ 1 ] != nX * nY ){
      string message("PSF::reconstructPSFFromFit: ERROR: zFit_In.getShape()[ 0 ](=");
      message += to_string( zFit_In.getShape()[ 0 ] ) + ") * zFit_In.getShape()[ 1 ](=" + to_string( zFit_In.getShape()[ 1 ] );
      message += ") (=" + to_string( zFit_In.getShape()[ 0 ] * zFit_In.getShape()[ 1 ] ) + ") != xGridRelativeToCenterFit_In.getShape()[ 0 ](=";
      message += to_string( nX ) + ") * yGridRelativeToCenterFit_In.getShape()[ 0 ](=" + to_string( nY ) + ") (=" + to_string( nX * nY ) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    Vector xRelativeToCenter = ndarray::allocate(nX * nY);
    Vector yRelativeToCenter = ndarray::allocate(nX * nY);
    Vector zRelativeToCenter = ndarray::allocate(nX * nY);

    auto itXRel = xRelativeToCenter.begin();
    auto itYRel = yRelativeToCenter.begin();
    auto itZRel = zRelativeToCenter.begin();
    auto itZGridRow = zFit_In.begin();
    for ( auto itY = yGridRelativeToCenterFit_In.begin(); itY != yGridRelativeToCenterFit_In.end(); ++itY, ++itZGridRow ){
      auto itZGridCol = itZGridRow->begin();
      for ( auto itX = xGridRelativeToCenterFit_In.begin(); itX != xGridRelativeToCenterFit_In.end(); ++itX, ++itZGridCol ){
        *itXRel = *itX;
        *itYRel = *itY;
        *itZRel = *itZGridCol;
        ++itXRel;
        ++itYRel;
        ++itZRel;
      }
    }
    #ifdef __DEBUG_PFS_RECONSTRUCTPFSFROMFIT__
      cout << "PFS::reconstructPFSFromFit: xRelativeToCenter set to " << xRelativeToCenter << endl;
      cout << "PFS::reconstructPFSFromFit: yRelativeToCenter set to " << yRelativeToCenter << endl;
      cout << "PFS::reconstructPFSFromFit: zRelativeToCenter set to " << zRelativeToCenter << endl;
      cout << "PFS::reconstructPFSFromFit: starting construct ThinPlateSpline" << endl;
    #endif
    TPSControl tpsControl;
    tpsControl.regularization = regularization;
    Spline tps = Spline(xRelativeToCenter, yRelativeToCenter, zRelativeToCenter, tpsControl);
    Vector const& xArr = _imagePSF_XRelativeToCenter;
    Vector const& yArr = _imagePSF_YRelativeToCenter;
    #ifdef __DEBUG_PFS_RECONSTRUCTPFSFROMFIT__
      cout << "PFS::reconstructPFSFromFit: starting tps.fitArray" << endl;
    #endif
    Image zFitArr = tps.fitArray(xArr, yArr, false);
    Vector zVec(_imagePSF_XRelativeToCenter.size());
    for ( int i = 0; i < _imagePSF_XRelativeToCenter.size(); ++i )
      zVec[ i ] = zFitArr[ ndarray::makeVector( i, 0 ) ];
    #ifdef __DEBUG_PFS_RECONSTRUCTPFSFROMFIT__
      cout << "PFS::reconstructPFSFromFit: zVec set to " << zVec << endl;
    #endif
    #ifdef __DEBUG_PSF__
      cout << "PSF::reconstructPSFFromFit(xGridRelativeToCenterFit, yGridRelativeToCenterFit, zFit, regularization) finished" << endl;
    #endif
    return zVec;
  }

  template< typename T >
  typename PSF<T>::Vector PSF<T>::reconstructPSFFromFit(
      Vector const& xGridRelativeToCenterFit_In,
      Vector const& yGridRelativeToCenterFit_In,
      Image const& zFit_In,
      Image const& weights_In
      ) const
  {
    #ifdef __DEBUG_PSF__
      cout << "PSF::reconstructPSFFromFit(xGridRelativeToCenterFit, yGridRelativeToCenterFit, zFit, weights) started" << endl;
    #endif
    int nX = xGridRelativeToCenterFit_In.getShape()[ 0 ];
    int nY = yGridRelativeToCenterFit_In.getShape()[ 0 ];
    if ( zFit_In.getShape()[ 0 ] * zFit_In.getShape()[ 1 ] != nX * nY ){
      string message("PSF::reconstructPSFFromFit: ERROR: zFit_In.getShape()[ 0 ](=");
      message += to_string( zFit_In.getShape()[ 0 ] ) + ") * zFit_In.getShape()[ 1 ](=" + to_string( zFit_In.getShape()[ 1 ] );
      message += ") (=" + to_string( zFit_In.getShape()[ 0 ] * zFit_In.getShape()[ 1 ] ) + ") != xGridRelativeToCenterFit_In.getShape()[ 0 ](=";
      message += to_string( nX ) + ") * yGridRelativeToCenterFit_In.getShape()[ 0 ](=" + to_string( nY ) + ") (=" + to_string( nX * nY ) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if ( weights_In.getShape()[ 0 ] * weights_In.getShape()[ 1 ] != nX * nY ){
      string message("PSF::reconstructPSFFromFit: ERROR: weights_In.getShape()[ 0 ](=");
      message += to_string( weights_In.getShape()[ 0 ] ) + ") * weights_In.getShape()[ 1 ](=" + to_string( weights_In.getShape()[ 1 ] );
      message += ") (=" + to_string( zFit_In.getShape()[ 0 ] * zFit_In.getShape()[ 1 ] ) + ") != xGridRelativeToCenterFit_In.getShape()[ 0 ](=";
      message += to_string( nX ) + ") * yGridRelativeToCenterFit_In.getShape()[ 0 ](=" + to_string( nY ) + ") (=" + to_string( nX * nY ) + ")";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    Vector xRelativeToCenter = ndarray::allocate(nX * nY);
    Vector yRelativeToCenter = ndarray::allocate(nX * nY);
    Vector zRelativeToCenter = ndarray::allocate(nX * nY);
    Vector weights = ndarray::allocate(nX * nY);

    auto itXRel = xRelativeToCenter.begin();
    auto itYRel = yRelativeToCenter.begin();
    auto itZRel = zRelativeToCenter.begin();
    auto itWeights = weights.begin();
    auto itZGridRow = zFit_In.begin();
    auto itWeightsRow = weights_In.begin();
    for ( auto itY = yGridRelativeToCenterFit_In.begin(); itY != yGridRelativeToCenterFit_In.end(); ++itY, ++itZGridRow, ++itWeightsRow ){
      auto itZGridCol = itZGridRow->begin();
      auto itWeightsCol = itWeightsRow->begin();
      for ( auto itX = xGridRelativeToCenterFit_In.begin(); itX != xGridRelativeToCenterFit_In.end(); ++itX, ++itZGridCol, ++itWeightsCol, ++itXRel, ++itYRel, ++itZRel, ++itWeights ){
        *itXRel = *itX;
        *itYRel = *itY;
        *itZRel = *itZGridCol;
        *itWeights = *itWeightsCol;
      }
    }
    #ifdef __DEBUG_PFS_RECONSTRUCTPFSFROMFIT__
      cout << "PFS::reconstructPFSFromFit: xRelativeToCenter set to " << xRelativeToCenter << endl;
      cout << "PFS::reconstructPFSFromFit: yRelativeToCenter set to " << yRelativeToCenter << endl;
      cout << "PFS::reconstructPFSFromFit: zRelativeToCenter set to " << zRelativeToCenter << endl;
      cout << "PFS::reconstructPFSFromFit: weights set to " << weights << endl;
      cout << "PFS::reconstructPFSFromFit: starting construct ThinPlateSpline" << endl;
    #endif
    TPSControl tpsControl;
    Spline tps = Spline( xRelativeToCenter, yRelativeToCenter, zRelativeToCenter, weights, tpsControl);
    Vector const& xArr = _imagePSF_XRelativeToCenter;
    Vector const& yArr = _imagePSF_YRelativeToCenter;
    #ifdef __DEBUG_PFS_RECONSTRUCTPFSFROMFIT__
      cout << "PFS::reconstructPFSFromFit: starting tps.fitArray" << endl;
    #endif
    Image zFitArr = tps.fitArray(xArr, yArr, false);
    Vector zVec(_imagePSF_XRelativeToCenter.size());
    for ( int i = 0; i < _imagePSF_XRelativeToCenter.size(); ++i )
      zVec[ i ] = zFitArr[ ndarray::makeVector( i, 0 ) ];
    #ifdef __DEBUG_PFS_RECONSTRUCTPFSFROMFIT__
      cout << "PFS::reconstructPFSFromFit: zVec set to " << zVec << endl;
    #endif
    #ifdef __DEBUG_PSF__
      cout << "PSF::reconstructPSFFromFit(xGridRelativeToCenterFit, yGridRelativeToCenterFit, zFit, weights) finished" << endl;
    #endif
    return zVec;
  }

  template< typename T >
  double PSF< T >::fitFittedPSFToZTrace(Vector const& zFit_In ){
    #ifdef __DEBUG_PSF__
      cout << "PSF::fitFittedPSFToZTrace(zFitVec) started" << endl;
    #endif
    Vector measureErrors = ndarray::allocate(zFit_In.getShape());
    for (auto && mm : measureErrors) {
        mm = 0.0;
    }
    double result = fitFittedPSFToZTrace( zFit_In,
                                          measureErrors );
    #ifdef __DEBUG_PSF__
      cout << "PSF::fitFittedPSFToZTrace(zFit) finished" << endl;
    #endif
    return result;
  }

  template< typename T >
  double PSF< T >::fitFittedPSFToZTrace(Vector const& zFit_In,
                                        Vector const& measureErrors_In ){
    #ifdef __DEBUG_PSF__
      cout << "PSF::fitFittedPSFToZTrace(zFitArr, measureErrors) started" << endl;
    #endif
    std::vector< std::string > args(1);
    args[0] = "MEASURE_ERRORS_IN";
    std::vector< void* > argV(1);
    PTR(Vector) p_measureErrors(new Vector(measureErrors_In));
    argV[0] = &p_measureErrors;
    T fittedValue, fittedConstant;
    int result;
    result = pfs::drp::stella::math::LinFitBevingtonNdArray( _imagePSF_ZTrace,
                                                             zFit_In,
                                                             fittedValue,
                                                             fittedConstant,
                                                             false,
                                                             args,
                                                             argV );
    #ifdef __DEBUG_PSF__
      cout << "PSF::fitFittedPSFToZTrace(zFitArr, measureErrors) finished" << endl;
    #endif
    return fittedValue;
  }

  template< typename T >
  void PSFSet< T >::setPSF(const std::size_t i,
                           const PTR(PSF< T >) & psf)
  {
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::setPSF(i, psf) started" << endl;
    #endif
    if (i > _psfs->size()){
      string message("PSFSet::setPSF: ERROR: i=");
      message += to_string(i) + " > _psfs->size()=" + to_string(_psfs->size()) + " => Returning FALSE";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (i == _psfs->size()){
      _psfs->push_back(psf);
    }
    else{
      (*_psfs)[i] = psf;
    }
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::setPSF(i, psf) finished" << endl;
    #endif
  }

  /// add one PSF to the set
  template< typename T >
  void PSFSet< T >::addPSF(const PTR(PSF< T >) & psf )/// the Spectrum to add
  {
    _psfs->push_back(psf);
  }

  template< typename T >
  PTR(PSF< T >)& PSFSet< T >::getPSF(const size_t i){
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::getPSF(i) started" << endl;
    #endif
    if (i >= _psfs->size()){
      string message("PSFSet::getPSF(i=");
      message += to_string(i) + string("): ERROR: i > _psfs->size()=") + to_string(_psfs->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::getPSF(i) finished" << endl;
    #endif
    return _psfs->at(i);
  }

  template< typename T >
  const PTR(const PSF< T >) PSFSet< T >::getPSF(const size_t i) const {
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::getPSF(i) const started" << endl;
    #endif
    if (i >= _psfs->size()){
      string message("PSFSet::getPSF(i=");
      message += to_string(i) + string("): ERROR: i > _psfs->size()=") + to_string(_psfs->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::getPSF(i) const started" << endl;
    #endif
    return _psfs->at(i);
  }

  template< typename T >
  void PSFSet< T >::erase(const size_t iStart, const size_t iEnd){
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::erase(iStart, iEnd) started" << endl;
    #endif
    if (iStart >= _psfs->size()){
      string message("PSFSet::erase(iStart=");
      message += to_string(iStart) + string("): ERROR: iStart >= _psfs->size()=") + to_string(_psfs->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (iEnd >= _psfs->size()){
      string message("PSFSet::erase(iEnd=");
      message += to_string(iEnd) + string("): ERROR: iEnd >= _psfs->size()=") + to_string(_psfs->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if ((iEnd > 0) && (iStart > iEnd)){
      string message("PSFSet::erase(iStart=");
      message += to_string(iStart) + string("): ERROR: iStart > iEnd=") + to_string(iEnd);
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (iStart == (_psfs->size()-1)){
      _psfs->pop_back();
    }
    else{
      if (iEnd == 0)
        _psfs->erase(_psfs->begin() + iStart);
      else
        _psfs->erase(_psfs->begin() + iStart, _psfs->begin() + iEnd);
    }
    #ifdef __DEBUG_PSFSet__
      cout << "PSFSet::erase(iStart, iEnd) started" << endl;
    #endif
  }

  namespace math{

    template< typename PsfT, typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
    PTR(PSFSet< PsfT >) calculate2dPSFPerBin(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace,
                                          Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrum,
                                          TwoDPSFControl const& twoDPSFControl){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::calculate2dPSFPerBin(fiberTrace, spectrum, twoDPSFControl) started" << endl;
      #endif
      ndarray::Array< PsfT, 2, 1 > collapsedPSF = ndarray::allocate(1,1);
      return calculate2dPSFPerBin(fiberTrace, spectrum, twoDPSFControl, collapsedPSF);
    }

    template< typename PsfT, typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
    PTR(PSFSet< PsfT >) calculate2dPSFPerBin(FiberTrace< ImageT, MaskT, VarianceT > const& fiberTrace,
                                             Spectrum< ImageT, MaskT, VarianceT, WavelengthT > const& spectrum,
                                             TwoDPSFControl const& twoDPSFControl,
                                             ndarray::Array< PsfT, 2, 1 > const& collapsedPSF){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::calculate2dPSFPerBin(fiberTrace, spectrum, twoDPSFControl, collapsedPSF) started" << endl;
      #endif
      int swathWidth = twoDPSFControl.swathWidth;
      ndarray::Array<size_t, 2, 1> binBoundY = fiberTrace.calcSwathBoundY(swathWidth);
      if ( binBoundY[ ndarray::makeVector<size_t>( binBoundY.getShape()[0]-1, 1 ) ] != fiberTrace.getHeight()-1){
        string message("calculate2dPSFPerBin: FiberTrace");
        message += to_string(fiberTrace.getITrace()) + ": ERROR: binBoundY[binBoundY.getShape()[0]-1=";
        message += to_string(binBoundY.getShape()[0]-1) + "][1] = " + to_string( binBoundY[ ndarray::makeVector<size_t>( binBoundY.getShape()[0]-1, 1 ) ] ) + "!= fiberTrace.getHeight()-1 = ";
        message += to_string(fiberTrace.getHeight()-1);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_CALC2DPSF__
        cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": fiberTrace.getHeight() = " << fiberTrace.getHeight() << ": binBoundY = " << binBoundY << endl;
      #endif

      PTR(PSFSet< PsfT >) psfSet(new PSFSet< PsfT >());
      for (int iBin = 0; iBin < binBoundY.getShape()[0]; ++iBin){
        #ifdef __DEBUG_CALC2DPSF__
          cout << "binBoundY = " << binBoundY << endl;
        #endif
        /// start calculate2dPSF for bin iBin
        if ( binBoundY[ ndarray::makeVector( iBin, 1 ) ] >= fiberTrace.getHeight()){
          string message("calculate2dPSFPerBin: FiberTrace");
          message += to_string(fiberTrace.getITrace()) + ": iBin " + to_string(iBin) + ": ERROR: binBoundY[" + to_string(iBin) + "][1]=";
          message += to_string( binBoundY[ ndarray::makeVector( iBin, 1 ) ] ) + " >= fiberTrace.getHeight()=" + to_string(fiberTrace.getHeight());
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        #ifdef __DEBUG_CALC2DPSF__
          cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": calculating PSF for iBin " << iBin << ": binBoundY[" << iBin << "][0] = " << binBoundY[ ndarray::makeVector( iBin, 0 ) ] << ", binBoundY[" << iBin << "][1] = " << binBoundY[ndarray::makeVector( iBin, 1 ) ] << endl;
        #endif
        PTR(TwoDPSFControl) pTwoDPSFControl(new TwoDPSFControl(twoDPSFControl));
        PTR(PSF< PsfT >) psf(new PSF< PsfT >((unsigned int)(binBoundY[ ndarray::makeVector( iBin, 0 ) ] ),
                                             (unsigned int)(binBoundY[ ndarray::makeVector( iBin, 1 ) ] ),
                                             pTwoDPSFControl,
                                             fiberTrace.getITrace(),
                                             iBin));
        if (psf->getYHigh() != binBoundY[ ndarray::makeVector( iBin, 1 ) ] ){
          string message("calculate2dPSFPerBin: FiberTrace");
          message += to_string(fiberTrace.getITrace()) + ": iBin " + to_string(iBin) + ": ERROR: psf->getYHigh(=";
          message += to_string(psf->getYHigh()) + ") != binBoundY[" + to_string(iBin) + "][1]=" + to_string(binBoundY[ ndarray::makeVector( iBin, 1 ) ] );
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        #ifdef __DEBUG_CALC2DPSF__
          cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": iBin " << iBin << ": starting extractPSFs()" << endl;
        #endif
        psf->extractPSFs(fiberTrace, spectrum, collapsedPSF);
        if (psf->getITrace() != fiberTrace.getITrace()){
          string message("calculate2dPSFPerBin: FiberTrace");
          message += to_string(fiberTrace.getITrace()) + string(": iBin ") + to_string(iBin) + string(": ERROR: psf->getITrace(=");
          message += to_string(psf->getITrace()) + ") != fiberTrace.getITrace(=" + to_string(fiberTrace.getITrace());
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        #ifdef __DEBUG_CALC2DPSF__
          cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": iBin " << iBin << ": extractPSFs() finished" << endl;
          cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": iBin = " << iBin << ": psf->getImagePSF_XTrace().size() = " << psf->getImagePSF_XTrace().size() << endl;
        #endif
        if (psf->getImagePSF_XRelativeToCenter().size() > 3)
          psfSet->addPSF(psf);
      }
      #ifdef __DEBUG_PSF__
        cout << "psfMath::calculate2dPSFPerBin(fiberTrace, spectrum, twoDPSFControl, collapsedPSF) finished" << endl;
      #endif
      return psfSet;
    }

    template< typename PsfT, typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT >
    std::vector< PTR( PSFSet< PsfT > ) > calculate2dPSFPerBin( FiberTraceSet< ImageT, MaskT, VarianceT > const& fiberTraceSet,
                                                               SpectrumSet< ImageT, MaskT, VarianceT, WavelengthT > const& spectrumSet,
                                                               TwoDPSFControl const& twoDPSFControl ){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::calculate2dPSFPerBin(fiberTraceSet, spectrumSet, twoDPSFControl) started" << endl;
      #endif
      std::vector< PTR(PSFSet< PsfT >)> vecOut(0);
      for (size_t i = 0; i < fiberTraceSet.size(); ++i){
        PTR(PSFSet< PsfT >) psfSet = calculate2dPSFPerBin< PsfT, ImageT, MaskT, VarianceT, WavelengthT >( *( fiberTraceSet.getFiberTrace( i ) ),
                                                                                                          *( spectrumSet.getSpectrum( i ) ),
                                                                                                          twoDPSFControl );
        vecOut.push_back(psfSet);
      }
      #ifdef __DEBUG_PSF__
        cout << "psfMath::calculate2dPSFPerBin(fiberTraceSet, spectrumSet, twoDPSFControl) finished" << endl;
      #endif
      return vecOut;
    }

    template< typename PsfT, typename CoordsT >
    ndarray::Array< PsfT, 2, 1 > interpolatePSFThinPlateSpline(PSF< PsfT > & psf,
                                                               ndarray::Array< CoordsT, 1, 1 > const& xPositions,
                                                               ndarray::Array< CoordsT, 1, 1 > const& yPositions,
                                                               bool const isXYPositionsGridPoints,
                                                               double const regularization,
                                                               PsfT const shapeParameter,
                                                               unsigned short const mode ){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFThinPlateSpline(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization) started" << endl;
      #endif
      if (mode > 1){
        string message("PSF::InterPolateThinPlateSplineChiSquare: ERROR: mode must be 0 or 1");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      typedef typename PSF<PsfT>::Vector Vector;
      Vector xRelativeToCenter = psf.getImagePSF_XRelativeToCenter();
      Vector yRelativeToCenter = psf.getImagePSF_YRelativeToCenter();
      if (xRelativeToCenter.size() < 3){
        string message("PSF::InterPolateThinPlateSpline: ERROR: xRelativeToCenter.size()(=");
        message += to_string( xRelativeToCenter.size() ) + ") < 3";
        cout << message << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
      }

      Vector xArr = xRelativeToCenter;
      Vector yArr = yRelativeToCenter;
      Vector zArr = (mode == 0) ? psf.getImagePSF_ZNormalized() : psf.getImagePSF_ZTrace();
      cout << "PSF::interpolatePSFThinPlateSpline: starting interpolateThinPlateSplineEigen" << endl;
      TPSControl tpsControl;
      tpsControl.regularization = regularization;
      tpsControl.shapeParameter = shapeParameter;
      math::ThinPlateSpline<PsfT, PsfT> tps = math::ThinPlateSpline<PsfT, PsfT>( xArr,
                                                                                 yArr,
                                                                                 zArr,
                                                                                 tpsControl );
      ndarray::Array< PsfT, 1, 1 > xPos = ndarray::allocate( xPositions.getShape() );
      ndarray::Array< PsfT, 1, 1 > yPos = ndarray::allocate( yPositions.getShape() );
      auto itXPos = xPos.begin();
      for (auto itX = xPositions.begin(); itX != xPositions.end(); ++itX, ++itXPos ){
        *itXPos = PsfT(*itX);
      }
      auto itYPos = yPos.begin();
      for (auto itY = yPositions.begin(); itY != yPositions.end(); ++itY, ++itYPos ){
        *itYPos = PsfT(*itY);
      }
      cout << "PSF::interpolatePSFThinPlateSpline: xPos = " << xPos << endl;
      cout << "PSF::interpolatePSFThinPlateSpline: yPos = " << yPos << endl;
      ndarray::Array< PsfT, 2, 1 > arr_Out = ndarray::copy( tps.fitArray( xPos,
                                                                          yPos,
                                                                          isXYPositionsGridPoints ) );
      cout << "PSF::interpolatePSFThinPlateSpline: arr_Out = " << arr_Out << endl;
      ndarray::Array< PsfT, 1, 1 > zRec = ndarray::copy( tps.getZFit() );
      psf.setImagePSF_ZFit(zRec);
      psf.setThinPlateSpline( tps );
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFThinPlateSpline(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization, shapeParameter, mode): tps.getDataPointsX().getShape()[ 0 ] = " << tps.getDataPointsX().getShape()[ 0 ] << endl;
        cout << "psfMath::interpolatePSFThinPlateSpline(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization, shapeParameter, mode): psf.getThinPlateSpline().getDataPointsX().getShape()[ 0 ] = " << psf.getThinPlateSpline().getDataPointsX().getShape()[ 0 ] << endl;
        cout << "psfMath::interpolatePSFThinPlateSpline(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization, shapeParameter, mode) finished" << endl;
      #endif
      return arr_Out;
    }

    template< typename PsfT, typename WeightT, typename CoordsT >
    ndarray::Array< PsfT, 2, 1> interpolatePSFThinPlateSpline( PSF< PsfT > & psf,
                                                               ndarray::Array<WeightT, 1, 1> const& weights,
                                                               ndarray::Array<CoordsT, 1, 1> const& xPositions,
                                                               ndarray::Array<CoordsT, 1, 1> const& yPositions,
                                                               bool const isXYPositionsGridPoints,
                                                               PsfT const shapeParameter,
                                                               unsigned short const mode ){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFThinPlateSpline(psf, weights, xPositions, yPositions, isXYPositionsGridPoints) started" << endl;
      #endif
      if (mode > 1){
        string message("PSF::InterPolateThinPlateSplineChiSquare: ERROR: mode must be 0 or 1");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      typedef typename PSF<PsfT>::Vector Vector;
      Vector xRelativeToCenter = psf.getImagePSF_XRelativeToCenter();
      Vector yRelativeToCenter = psf.getImagePSF_YRelativeToCenter();
      if (xRelativeToCenter.size() < 3){
        string message("PSF::InterPolateThinPlateSpline: ERROR: xRelativeToCenter.size()(=");
        message += to_string( xRelativeToCenter.size() ) + ") < 3";
        cout << message << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
      }

      Vector xArr = xRelativeToCenter;
      Vector yArr = yRelativeToCenter;
      Vector zArr = (mode == 0) ? psf.getImagePSF_ZNormalized() : psf.getImagePSF_ZTrace();
      #ifdef __DEBUG_CALC_TPS__
        cout << "PSF::interpolatePSFThinPlateSpline: xArr = " << xArr << endl;
        cout << "PSF::interpolatePSFThinPlateSpline: yArr = " << yArr << endl;
        cout << "PSF::interpolatePSFThinPlateSpline: zArr = " << zArr << endl;
        cout << "PSF::interpolatePSFThinPlateSpline: starting interpolateThinPlateSplineEigen" << endl;
      #endif
      ndarray::Array<PsfT, 1, 1> weightsT = ndarray::allocate(weights.getShape()[0]);
      auto itWeightsT = weightsT.begin();
      for (auto itWeights = weights.begin(); itWeights != weights.end(); ++itWeights, ++itWeightsT)
        *itWeightsT = PsfT(*itWeights);
      TPSControl tpsControl;
      math::ThinPlateSpline< PsfT, PsfT > tps = math::ThinPlateSpline< PsfT, PsfT >( xArr,
                                                                                     yArr,
                                                                                     zArr,
                                                                                     weightsT,
                                                                                     tpsControl );

      ndarray::Array< PsfT, 1, 1 > xPos = ndarray::allocate( xPositions.getShape() );
      ndarray::Array< PsfT, 1, 1 > yPos = ndarray::allocate( yPositions.getShape() );
      auto itXPos = xPos.begin();
      for (auto itX = xPositions.begin(); itX != xPositions.end(); ++itX, ++itXPos ){
        *itXPos = PsfT(*itX);
      }
      auto itYPos = yPos.begin();
      for (auto itY = yPositions.begin(); itY != yPositions.end(); ++itY, ++itYPos ){
        *itYPos = PsfT(*itY);
      }
      ndarray::Array< PsfT, 2, 1 > zFit = ndarray::copy( tps.fitArray( xPos,
                                                                       yPos,
                                                                       isXYPositionsGridPoints ) );
      ndarray::Array< PsfT, 1, 1 > zRec = ndarray::copy( tps.getZFit() );
      psf.setImagePSF_ZFit(zRec);
      psf.setThinPlateSpline( tps );
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFThinPlateSpline(): tps.getDataPointsX().getShape()[ 0 ] = " << tps.getDataPointsX().getShape()[ 0 ] << endl;
        cout << "psfMath::interpolatePSFThinPlateSpline(): psf.getThinPlateSpline().getDataPointsX().getShape()[ 0 ] = " << psf.getThinPlateSpline().getDataPointsX().getShape()[ 0 ] << endl;
        cout << "psfMath::interpolatePSFThinPlateSpline(psf, weights, xPositions, yPositions, isXYPositionsGridPoints) finished" << endl;
      #endif
      return zFit;
    }

    template< typename PsfT, typename CoordsT >
    ndarray::Array< PsfT, 2, 1 > interpolatePSFThinPlateSplineChiSquare( PSF< PsfT > & psf,
                                                                         ndarray::Array< CoordsT, 1, 1 > const& xPositions,
                                                                         ndarray::Array< CoordsT, 1, 1 > const& yPositions,
                                                                         bool const isXYPositionsGridPoints,
                                                                         PsfT const regularization,
                                                                         PsfT const shapeParameter,
                                                                         unsigned short const mode ){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFThinPlateSplineChiSquare(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization) started" << endl;
      #endif
      if (mode > 1){
        string message("PSF::InterPolateThinPlateSplineChiSquare: ERROR: mode must be 0 or 1");
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      typedef typename PSF<PsfT>::Vector Vector;
      Vector xRelativeToCenter = psf.getImagePSF_XRelativeToCenter();
      Vector yRelativeToCenter = psf.getImagePSF_YRelativeToCenter();
      if (xRelativeToCenter.size() < 3){
        string message("PSF::InterPolateThinPlateSplineChiSquare: ERROR: xRelativeToCenter.size()(=");
        message += to_string( xRelativeToCenter.size() ) + ") < 3";
        cout << message << endl;
        throw LSST_EXCEPT( pexExcept::Exception, message.c_str() );
      }

      Vector xArr = xRelativeToCenter;
      Vector yArr = yRelativeToCenter;
      Vector zArr = (mode == 0) ? psf.getImagePSF_ZNormalized() : psf.getImagePSF_ZTrace();
      #ifdef __DEBUG_CALC_TPS__
        cout << "PSF::interpolatePSFThinPlateSplineChiSquare: xArr = " << xArr << endl;
        cout << "PSF::interpolatePSFThinPlateSplineChiSquare: yArr = " << yArr << endl;
        cout << "PSF::interpolatePSFThinPlateSplineChiSquare: psf.getImagePSF_ZNormalized() = " << psf.getImagePSF_ZNormalized() << endl;
        cout << "PSF::interpolatePSFThinPlateSplineChiSquare: zArr = " << zArr << endl;
      #endif
      ndarray::Array< PsfT, 1, 1 > xPos = ndarray::allocate( xPositions.getShape() );
      ndarray::Array< PsfT, 1, 1 > yPos = ndarray::allocate( yPositions.getShape() );
      auto itXPos = xPos.begin();
      for (auto itX = xPositions.begin(); itX != xPositions.end(); ++itX, ++itXPos ){
        *itXPos = PsfT(*itX);
      }
      auto itYPos = yPos.begin();
      for (auto itY = yPositions.begin(); itY != yPositions.end(); ++itY, ++itYPos ){
        *itYPos = PsfT(*itY);
      }
      cout << "PSF::interpolatePSFThinPlateSplineChiSquare: xArr = " << xArr << endl;
      cout << "PSF::interpolatePSFThinPlateSplineChiSquare: yArr = " << yArr << endl;
      cout << "PSF::interpolatePSFThinPlateSplineChiSquare: zArr = " << zArr << endl;
      cout << "PSF::interpolatePSFThinPlateSplineChiSquare: xPos = " << xPos << endl;
      cout << "PSF::interpolatePSFThinPlateSplineChiSquare: yPos = " << yPos << endl;
      cout << "PSF::interpolatePSFThinPlateSplineChiSquare: starting interpolateThinPlateSplineChiSquare" << endl;
      TPSControl tpsControl;
      tpsControl.regularization = regularization;
      tpsControl.shapeParameter = shapeParameter;
      math::ThinPlateSplineChiSquare< PsfT, PsfT > tps = math::ThinPlateSplineChiSquare< PsfT, PsfT >( xArr,
                                                                                                       yArr,
                                                                                                       zArr,
                                                                                                       xPos,
                                                                                                       yPos,
                                                                                                       isXYPositionsGridPoints,
                                                                                                       tpsControl );
      ndarray::Array< PsfT, 2, 1 > arr_Out = ndarray::copy( tps.fitArray( xPos,
                                                                          yPos,
                                                                          isXYPositionsGridPoints ) );
      ndarray::Array< PsfT, 1, 1 > zRec = ndarray::copy( tps.getZFit() );
      psf.setImagePSF_ZFit(zRec);
      psf.setThinPlateSplineChiSquare( tps );
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFThinPlateSplineChiSquare(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization) finished" << endl;
      #endif
      return arr_Out;
    }

    template < typename PsfT, typename CoordsT >
    ndarray::Array< PsfT, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< PsfT > & psfSet,
                                                                  ndarray::Array< CoordsT, 1, 1 > const& xPositions,
                                                                  ndarray::Array< CoordsT, 1, 1 > const& yPositions,
                                                                  bool const isXYPositionsGridPoints,
                                                                  double const regularization){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFSetThinPlateSpline(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization) started" << endl;
      #endif
      ndarray::Array< PsfT, 3, 1 > arrOut = ndarray::allocate(yPositions.getShape()[0], xPositions.getShape()[0], psfSet.size());
      for (size_t i = 0; i < psfSet.size(); ++i){
        #ifdef __DEBUG_CALC_TPS__
          if (psfSet.getPSF(i)->getImagePSF_XRelativeToCenter().size() < 3){
            string message("PSF::InterPolatePSFSetThinPlateSpline: ERROR: i=");
            message += to_string(i) + ": imagePSF_XRelativeToCenter().size()=" + to_string(psfSet.getPSF(i)->getImagePSF_XRelativeToCenter().size()) + " < 3";
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
        ndarray::Array< PsfT, 2, 1 > arr = ndarray::allocate(yPositions.getShape()[0], xPositions.getShape()[0]);
        arr.deep() = interpolatePSFThinPlateSpline( *( psfSet.getPSF( i ) ),
                                                    xPositions,
                                                    yPositions,
                                                    isXYPositionsGridPoints,
                                                    regularization );
        #ifdef __DEBUG_CALC_TPS__
          cout << "interpolatePSFSetThinPlateSpline: arr.getShape() = " << arr.getShape() << ", arrOut.getShape() = " << arrOut.getShape() << endl;
          cout << "interpolatePSFSetThinPlateSpline: arr = " << arr << endl;
        #endif
        arrOut[ndarray::view()()(i)].deep() = arr;
      }
      #ifdef __DEBUG_CALC_TPS__
        cout << "interpolatePSFSetThinPlateSpline: arrOut[:][:][0] = " << arrOut[ndarray::view()()(0)] << endl;
      #endif
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFSetThinPlateSpline(psf, xPositions, yPositions, isXYPositionsGridPoints, regularization) finished" << endl;
      #endif
      return arrOut;
    }

    template < typename PsfT, typename WeightT, typename CoordsT >
    ndarray::Array< PsfT, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< PsfT > & psfSet,
                                                                  ndarray::Array< WeightT, 2, 1 > const& weightArr,
                                                                  ndarray::Array< CoordsT, 1, 1 > const& xPositions,
                                                                  ndarray::Array< CoordsT, 1, 1 > const& yPositions,
                                                                  bool const isXYPositionsGridPoints){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFSetThinPlateSpline(psf, weights, xPositions, yPositions, isXYPositionsGridPoints) started" << endl;
      #endif
      ndarray::Array< PsfT, 3, 1 > arrOut = ndarray::allocate(yPositions.getShape()[0], xPositions.getShape()[0], psfSet.size());
      for (size_t i = 0; i < psfSet.size(); ++i){
        #ifdef __DEBUG_CALC_TPS__
        if (psfSet.getPSF(i)->getImagePSF_XRelativeToCenter().size() < 3){
          string message("PSF::InterPolatePSFSetThinPlateSpline: ERROR: i=");
          message += to_string(i) + ": imagePSF_XRelativeToCenter().size()=" + to_string(psfSet.getPSF(i)->getImagePSF_XRelativeToCenter().size()) + " < 3";
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        #endif
        size_t size = psfSet.getPSF(i)->getImagePSF_ZNormalized().size();
        ndarray::Array< WeightT, 1, 1 > weights = ndarray::allocate(size);
        weights.deep() = weightArr[ndarray::view(0, size)(i)];
        ndarray::Array< PsfT, 2, 1 > arr = ndarray::copy(interpolatePSFThinPlateSpline(*(psfSet.getPSF(i)),
                                                                                       weights,
                                                                                       xPositions,
                                                                                       yPositions,
                                                                                       isXYPositionsGridPoints ));
        #ifdef __DEBUG_CALC_TPS__
          cout << "interpolatePSFSetThinPlateSpline: arr.getShape() = " << arr.getShape() << ", arrOut.getShape() = " << arrOut.getShape() << endl;
          cout << "interpolatePSFSetThinPlateSpline: arr = " << arr << endl;
        #endif
        arrOut[ndarray::view()()(i)].deep() = arr;
      }
      #ifdef __DEBUG_CALC_TPS__
        cout << "interpolatePSFSetThinPlateSpline: arrOut[:][:][0] = " << arrOut[ndarray::view()()(0)] << endl;
      #endif
      #ifdef __DEBUG_PSF__
        cout << "psfMath::interpolatePSFSetThinPlateSpline(psf, weights, xPositions, yPositions, isXYPositionsGridPoints) started" << endl;
      #endif
      return arrOut;
    }

    template< typename PsfT, typename CoordT >
    ndarray::Array< PsfT, 2, 1 > collapseFittedPSF( ndarray::Array< CoordT, 1, 1 > const& xGridVec_In,
                                                    ndarray::Array< CoordT, 1, 1 > const& yGridVec_In,
                                                    ndarray::Array< PsfT, 2, 1 > const& zArr_In,
                                                    int const direction){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::collapseFittedPSF(xGridVec, yGridVec, zArr, direction) started" << endl;
      #endif
      if (xGridVec_In.getShape()[0] != zArr_In.getShape()[1]){
        string message("PSF::math::collapseFittedPSF: ERROR: xGridVec_In.getShape()[0]=");
        message += to_string(xGridVec_In.getShape()[0]) + " != zArr_In.getShape()[1]=" + to_string(zArr_In.getShape()[1]);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      if (yGridVec_In.getShape()[0] != zArr_In.getShape()[0]){
        string message("PSF::math::collapseFittedPSF: ERROR: yGridVec_In.getShape()[0]=");
        message += to_string(yGridVec_In.getShape()[0]) + " != zArr_In.getShape()[0]=" + to_string(zArr_In.getShape()[0]);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      size_t collapsedPSFLength;
      ndarray::Array< PsfT, 1, 1 > tempVec;
      ndarray::Array< PsfT, 2, 1 > collapsedPSF;
      if (direction == 0){
        collapsedPSFLength = yGridVec_In.getShape()[0];
        tempVec = ndarray::allocate(xGridVec_In.getShape()[0]);
        collapsedPSF = ndarray::allocate(collapsedPSFLength, 2);
        collapsedPSF[ndarray::view()(0)].deep() = yGridVec_In;
      }
      else{
        collapsedPSFLength = xGridVec_In.getShape()[0];
        tempVec = ndarray::allocate(yGridVec_In.getShape()[0]);
        collapsedPSF = ndarray::allocate(collapsedPSFLength, 2);
        collapsedPSF[ndarray::view()(0)].deep() = xGridVec_In;
      }
      for (size_t iPos = 0; iPos < collapsedPSFLength; ++iPos){
        assert(iPos < std::numeric_limits<int>::max());
        if (direction == 0){
          tempVec.deep() = zArr_In[ndarray::view(static_cast<int>(iPos))()];
        }
        else{
          tempVec.deep() = zArr_In[ndarray::view()(static_cast<int>(iPos))];
        }
        collapsedPSF[ ndarray::makeVector( static_cast<int>(iPos), 1 ) ] = tempVec.asEigen().array().sum();
      }
      #ifdef __DEBUG_PSF__
        cout << "psfMath::collapseFittedPSF(xGridVec, yGridVec, zArr, direction) finished" << endl;
      #endif
      return collapsedPSF;
    }

    template< typename PsfT, typename CoordT >
    ndarray::Array< PsfT, 2, 1 > collapsePSF( PSF< PsfT > const& psf_In,
                                               ndarray::Array< CoordT, 1, 1 > const& coordinatesX_In,
                                               ndarray::Array< CoordT, 1, 1 > const& coordinatesY_In,
                                               int const direction,
                                               double const regularization){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::collapsePSF(, psf, coordinatesX, coordinatesY, direction, regularization) started" << endl;
      #endif
      ndarray::Array< PsfT, 1, 1 > xRelativeToCenter = psf_In.getImagePSF_XRelativeToCenter().deep();
      ndarray::Array< PsfT, 1, 1 > yRelativeToCenter = psf_In.getImagePSF_YRelativeToCenter().deep();
      TPSControl tpsControl;
      tpsControl.regularization = regularization;
      ndarray::Array< const PsfT, 1, 1 > zNormalized = psf_In.getImagePSF_ZNormalized();
      math::ThinPlateSpline< PsfT, CoordT > tpsA = math::ThinPlateSpline< PsfT, CoordT >( xRelativeToCenter,
                                                                                          yRelativeToCenter,
                                                                                          zNormalized,
                                                                                          tpsControl );
      ndarray::Array< PsfT, 2, 1 > interpolatedPSF = tpsA.fitArray( coordinatesX_In,
                                                                    coordinatesY_In,
                                                                    true);
      #ifdef __DEBUG_PSF__
        cout << "psfMath::collapsePSF(psf, coordinatesX, coordinatesY, direction, regularization) finishing" << endl;
      #endif
      return math::collapseFittedPSF(coordinatesX_In,
                                     coordinatesY_In,
                                     interpolatedPSF,
                                     direction);
    }

    template< typename T >
    ndarray::Array< T, 2, 1> calcPositionsRelativeToCenter(T const centerPos_In,
                                                           T const width_In){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::calcPositionsRelativeToCenter(centerPos, width) started" << endl;
      #endif
      #ifdef __DEBUG_CPRTC__
        cout << "PSF::math::calcPositionsRelativeToCenter: centerPos_In = " << centerPos_In << ", width_In = " << width_In << endl;
      #endif
      size_t low = std::floor(centerPos_In - (width_In / 2.));
      size_t high = std::floor(centerPos_In + (width_In / 2.));
      #ifdef __DEBUG_CPRTC__
        cout << "PSF::math::calcPositionsRelativeToCenter: low = " << low << ", high = " << high << endl;
      #endif
      ndarray::Array< T, 2, 1 > arr_Out = ndarray::allocate(high - low + 1, 2);
      double dCenter = PIXEL_CENTER - centerPos_In + std::floor(centerPos_In);
      #ifdef __DEBUG_CPRTC__
        cout << "PSF::math::calcPositionsRelativeToCenter: dCenter = " << dCenter << endl;
      #endif
      for (size_t iPos = 0; iPos <= high - low; ++iPos){
        assert(iPos < std::numeric_limits<int>::max()); // safe to cast
        arr_Out[ ndarray::makeVector(static_cast<int>(iPos), 0)] = T(low + iPos);
        arr_Out[ ndarray::makeVector(static_cast<int>(iPos), 1)] =
            arr_Out[ ndarray::makeVector(static_cast<int>(iPos), 0 ) ] -
            T(std::floor(centerPos_In)) + dCenter;
        #ifdef __DEBUG_CPRTC__
          cout << "PSF::math::calcPositionsRelativeToCenter: arr_Out[" << iPos << "][0] = "
               << arr_Out[ndarray::makeVector(static_cast<int>(iPos), 0 ) ] << ", arr_Out[" << iPos
               << "][1] = " << arr_Out[ndarray::makeVector(static_cast<int>(iPos), 1 ) ] << endl;
        #endif
      }
      #ifdef __DEBUG_CPRTC__
        cout << "PSF::math::calcPositionsRelativeToCenter: arr_Out = " << arr_Out << endl;
      #endif
      #ifdef __DEBUG_PSF__
        cout << "psfMath::calcPositionsRelativeToCenter(centerPos, width) finished" << endl;
      #endif
      return arr_Out;
    }


    template< typename PsfT, typename CoordsT >
    ndarray::Array< CoordsT, 2, 1 > compareCenterPositions(PSF< PsfT > & psf,
                                                           ndarray::Array< const CoordsT, 1, 1 > const& xPositions,
                                                           ndarray::Array< const CoordsT, 1, 1 > const& yPositions,
                                                           float dXMax,
                                                           float dYMax,
                                                           bool setPsfXY){
      #ifdef __DEBUG_PSF__
        cout << "psfMath::compareCenterPositions(psf, xPositions, yPositions, dXMax, dYMax, setPsfXY) started" << endl;
      #endif
      if (xPositions.getShape()[0] != yPositions.getShape()[0]){
        string message("pfs::drp::stella::math::compareCenterPositions: ERROR: xPositions.getShape()[0]=");
        message += to_string(xPositions.getShape()[0]) + " != yPositions.getShape()[1]=" + to_string(yPositions.getShape()[1]);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_COMPARECENTERPOSITIONS__
        cout << "pfs::drp::stella::math::compareCenterPositions: xPosititions = " << xPositions.getShape()[0] << ": " << xPositions << endl;
        cout << "pfs::drp::stella::math::compareCenterPositions: yPosititions = " << yPositions.getShape()[0] << ": " << yPositions << endl;
      #endif
      size_t nPSFs = psf.getXCentersPSFCCD().size();
      #ifdef __DEBUG_COMPARECENTERPOSITIONS__
        cout << "pfs::drp::stella::math::compareCenterPositions: nPSFs = " << nPSFs << endl;
      #endif
      ndarray::Array< CoordsT, 2, 1 > dXdYdR_Out = ndarray::allocate(nPSFs, 3);
      dXdYdR_Out.deep() = -1.;
      CoordsT xPSF, yPSF, dX, dY, dR;
      size_t iList;
      bool found;
      std::vector< PsfT > xCenters_Out;
      std::vector< PsfT > yCenters_Out;
      for (size_t iPSF = 0; iPSF < nPSFs; ++iPSF){
        xPSF = psf.getXCentersPSFCCD()[iPSF];
        yPSF = psf.getYCentersPSFCCD()[iPSF];
        found = false;
        iList = 0;
        while ((!found) && (iList < static_cast<size_t>(xPositions.getShape()[0]))){
          dX = xPositions[iList] - xPSF;
          #ifdef __DEBUG_COMPARECENTERPOSITIONS__
            cout << "pfs::drp::stella::math::compareCenterPositions: spot number " << iPSF << ": xPSF = " << xPSF << ", xPosititions[" << iList << "] = " << xPositions[iList] << ": dX = " << dX << endl;
          #endif
          if (std::fabs(dX) < dXMax){
            dY = yPositions[iList] - yPSF;
            #ifdef __DEBUG_COMPARECENTERPOSITIONS__
              cout << "pfs::drp::stella::math::compareCenterPositions: spot number " << iPSF << " yPosititions[" << iList << "] = " << yPositions[iList] << ": dY = " << dY << endl;
            #endif
            if (std::fabs(dY) < dYMax){
              dR = sqrt(pow(dX, 2) + pow(dY, 2));
              dXdYdR_Out[ ndarray::makeVector( int( iPSF ), 0 ) ] = dX;
              dXdYdR_Out[ ndarray::makeVector( int( iPSF ), 1 ) ] = dY;
              dXdYdR_Out[ ndarray::makeVector( int( iPSF ), 2 ) ] = dR;
              found = true;
              if (setPsfXY){
                xCenters_Out.push_back(xPositions[iList]);
                yCenters_Out.push_back(yPositions[iList]);
              }
              #ifdef __DEBUG_COMPARECENTERPOSITIONS__
                cout << "pfs::drp::stella::math::compareCenterPositions: spot number " << iPSF << " found in input lists at position " << iList << endl;
              #endif
            }
          }
          ++iList;
          #ifdef __DEBUG_COMPARECENTERPOSITIONS__
            cout << "pfs::drp::stella::math::compareCenterPositions: spot number " << iPSF << " iList = " << iList << endl;
          #endif
        }
        if (!found){
          cout << "pfs::drp::stella::math::compareCenterPositions: WARNING: spot number " << iPSF << " not found in input lists" << endl;
          xCenters_Out.push_back(xPSF);
          yCenters_Out.push_back(yPSF);
        }
      }
      if (setPsfXY){
        psf.setXCentersPSFCCD(toArray(xCenters_Out).deep());
        psf.setYCentersPSFCCD(toArray(yCenters_Out).deep());
      }
      #ifdef __DEBUG_PSF__
        cout << "psfMath::compareCenterPositions(psf, xPositions, yPositions, dXMax, dYMax, setPsfXY) finished" << endl;
      #endif
      return dXdYdR_Out;
    }

    template ndarray::Array< float, 2, 1 > compareCenterPositions(PSF< float > &,
                                                                  ndarray::Array< const float, 1, 1 > const&,
                                                                  ndarray::Array< const float, 1, 1 > const&,
                                                                  float,
                                                                  float,
                                                                  bool);
    template ndarray::Array< float, 2, 1 > compareCenterPositions(PSF< double > &,
                                                                  ndarray::Array< const float, 1, 1 > const&,
                                                                  ndarray::Array< const float, 1, 1 > const&,
                                                                  float,
                                                                  float,
                                                                  bool);
    template ndarray::Array< double, 2, 1 > compareCenterPositions(PSF< float > &,
                                                                   ndarray::Array< const double, 1, 1 > const&,
                                                                   ndarray::Array< const double, 1, 1 > const&,
                                                                   float,
                                                                   float,
                                                                   bool);
    template ndarray::Array< double, 2, 1 > compareCenterPositions(PSF< double > &,
                                                                   ndarray::Array< const double, 1, 1 > const&,
                                                                   ndarray::Array< const double, 1, 1 > const&,
                                                                   float,
                                                                   float,
                                                                   bool);

    template ndarray::Array< float, 2, 1 > calcPositionsRelativeToCenter(float const, float const);
    template ndarray::Array< double, 2, 1 > calcPositionsRelativeToCenter(double const, double const);

    template ndarray::Array< float, 2, 1 > collapseFittedPSF( ndarray::Array< float, 1, 1 > const&,
                                                              ndarray::Array< float, 1, 1 > const&,
                                                              ndarray::Array< float, 2, 1 > const&,
                                                              int const);
    template ndarray::Array< double, 2, 1 > collapseFittedPSF( ndarray::Array< float, 1, 1 > const&,
                                                               ndarray::Array< float, 1, 1 > const&,
                                                               ndarray::Array< double, 2, 1 > const&,
                                                               int const);
    template ndarray::Array< float, 2, 1 > collapseFittedPSF( ndarray::Array< double, 1, 1 > const&,
                                                              ndarray::Array< double, 1, 1 > const&,
                                                              ndarray::Array< float, 2, 1 > const&,
                                                              int const);
    template ndarray::Array< double, 2, 1 > collapseFittedPSF( ndarray::Array< double, 1, 1 > const&,
                                                               ndarray::Array< double, 1, 1 > const&,
                                                               ndarray::Array< double, 2, 1 > const&,
                                                               int const);

    template ndarray::Array< double, 2, 1 > collapsePSF( PSF< double > const&,
                                                        ndarray::Array< double, 1, 1 > const&,
                                                        ndarray::Array< double, 1, 1 > const&,
                                                        int const,
                                                        double const);
    template ndarray::Array< float, 2, 1 > collapsePSF( PSF< float > const&,
                                                        ndarray::Array< float, 1, 1 > const&,
                                                        ndarray::Array< float, 1, 1 > const&,
                                                        int const,
                                                        double const);

    template ndarray::Array< double, 2, 1 > interpolatePSFThinPlateSpline(PSF< double > &,
                                                                          ndarray::Array< double, 1, 1 > const&,
                                                                          ndarray::Array< double, 1, 1 > const&,
                                                                          bool const,
                                                                          double const,
                                                                          double const,
                                                                          unsigned short const);

    template ndarray::Array<double, 2, 1> interpolatePSFThinPlateSpline(PSF< double > &,
                                                                        ndarray::Array< double, 1, 1 > const&,
                                                                        ndarray::Array< double, 1, 1 > const&,
                                                                        ndarray::Array< double, 1, 1 > const&,
                                                                        bool const,
                                                                        double const,
                                                                        unsigned short const);

    template ndarray::Array< double, 2, 1 > interpolatePSFThinPlateSplineChiSquare( PSF< double > &,
                                                                                    ndarray::Array< double, 1, 1 > const&,
                                                                                    ndarray::Array< double, 1, 1 > const&,
                                                                                    bool const,
                                                                                    double const,
                                                                                    double const,
                                                                                    unsigned short const );

    template ndarray::Array< float, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< float > &,
                                                                            ndarray::Array< float, 1, 1 > const&,
                                                                            ndarray::Array< float, 1, 1 > const&,
                                                                            bool const,
                                                                            double const);
    template ndarray::Array< double, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< double > &,
                                                                             ndarray::Array< float, 1, 1 > const&,
                                                                             ndarray::Array< float, 1, 1 > const&,
                                                                             bool const,
                                                                             double const);
    template ndarray::Array< float, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< float > &,
                                                                            ndarray::Array< double, 1, 1 > const&,
                                                                            ndarray::Array< double, 1, 1 > const&,
                                                                            bool const,
                                                                            double const);
    template ndarray::Array< double, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< double > &,
                                                                             ndarray::Array< double, 1, 1 > const&,
                                                                             ndarray::Array< double, 1, 1 > const&,
                                                                             bool const,
                                                                             double const);

    template ndarray::Array< float, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< float > &,
                                                                            ndarray::Array< float, 2, 1 > const&,
                                                                            ndarray::Array< float, 1, 1 > const&,
                                                                            ndarray::Array< float, 1, 1 > const&,
                                                                            bool const);
    template ndarray::Array< double, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< double > &,
                                                                             ndarray::Array< float, 2, 1 > const&,
                                                                             ndarray::Array< float, 1, 1 > const&,
                                                                             ndarray::Array< float, 1, 1 > const&,
                                                                             bool const);
    template ndarray::Array< float, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< float > &,
                                                                            ndarray::Array< double, 2, 1 > const&,
                                                                            ndarray::Array< float, 1, 1 > const&,
                                                                            ndarray::Array< float, 1, 1 > const&,
                                                                            bool const);
    template ndarray::Array< double, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< double > &,
                                                                             ndarray::Array< double, 2, 1 > const&,
                                                                             ndarray::Array< float, 1, 1 > const&,
                                                                             ndarray::Array< float, 1, 1 > const&,
                                                                             bool const);
    template ndarray::Array< float, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< float > &,
                                                                            ndarray::Array< float, 2, 1 > const&,
                                                                            ndarray::Array< double, 1, 1 > const&,
                                                                            ndarray::Array< double, 1, 1 > const&,
                                                                            bool const);
    template ndarray::Array< double, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< double > &,
                                                                             ndarray::Array< float, 2, 1 > const&,
                                                                             ndarray::Array< double, 1, 1 > const&,
                                                                             ndarray::Array< double, 1, 1 > const&,
                                                                             bool const);
    template ndarray::Array< float, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< float > &,
                                                                            ndarray::Array< double, 2, 1 > const&,
                                                                            ndarray::Array< double, 1, 1 > const&,
                                                                            ndarray::Array< double, 1, 1 > const&,
                                                                            bool const);
    template ndarray::Array< double, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< double > &,
                                                                             ndarray::Array< double, 2, 1 > const&,
                                                                             ndarray::Array< double, 1, 1 > const&,
                                                                             ndarray::Array< double, 1, 1 > const&,
                                                                             bool const);

    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&,
                                                       Spectrum< float, unsigned short, float, float > const&,
                                                       TwoDPSFControl const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&,
                                                        Spectrum< float, unsigned short, float, float > const&,
                                                        TwoDPSFControl const&);
    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&,
                                                       Spectrum< double, unsigned short, float, float > const&,
                                                       TwoDPSFControl const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&,
                                                        Spectrum< double, unsigned short, float, float > const&,
                                                        TwoDPSFControl const&);
    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&,
                                                       Spectrum< float, unsigned short, float, double > const&,
                                                       TwoDPSFControl const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&,
                                                        Spectrum< float, unsigned short, float, double > const&,
                                                        TwoDPSFControl const&);
    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&,
                                                       Spectrum< double, unsigned short, float, double > const&,
                                                       TwoDPSFControl const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&,
                                                        Spectrum< double, unsigned short, float, double > const&,
                                                        TwoDPSFControl const&);

    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&,
                                                       Spectrum< float, unsigned short, float, float > const&,
                                                       TwoDPSFControl const&,
                                                        ndarray::Array< float, 2, 1 > const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&,
                                                        Spectrum< float, unsigned short, float, float > const&,
                                                        TwoDPSFControl const&,
                                                        ndarray::Array< double, 2, 1 > const&);
    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&,
                                                       Spectrum< double, unsigned short, float, float > const&,
                                                       TwoDPSFControl const&,
                                                        ndarray::Array< float, 2, 1 > const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&,
                                                        Spectrum< double, unsigned short, float, float > const&,
                                                        TwoDPSFControl const&,
                                                        ndarray::Array< double, 2, 1 > const&);
    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&,
                                                       Spectrum< float, unsigned short, float, double > const&,
                                                       TwoDPSFControl const&,
                                                        ndarray::Array< float, 2, 1 > const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< float, unsigned short, float > const&,
                                                        Spectrum< float, unsigned short, float, double > const&,
                                                        TwoDPSFControl const&,
                                                        ndarray::Array< double, 2, 1 > const&);
    template PTR(PSFSet< float >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&,
                                                       Spectrum< double, unsigned short, float, double > const&,
                                                       TwoDPSFControl const&,
                                                        ndarray::Array< float, 2, 1 > const&);
    template PTR(PSFSet< double >) calculate2dPSFPerBin(FiberTrace< double, unsigned short, float > const&,
                                                        Spectrum< double, unsigned short, float, double > const&,
                                                        TwoDPSFControl const&,
                                                        ndarray::Array< double, 2, 1 > const&);

    template std::vector<PTR(PSFSet< float >)> calculate2dPSFPerBin( FiberTraceSet< float, unsigned short, float > const&,
                                                                     SpectrumSet< float, unsigned short, float, float > const&,
                                                                     TwoDPSFControl const&);
    template std::vector<PTR(PSFSet< double >)> calculate2dPSFPerBin( FiberTraceSet< float, unsigned short, float > const&,
                                                                      SpectrumSet< float, unsigned short, float, float > const&,
                                                                      TwoDPSFControl const&);
    template std::vector<PTR(PSFSet< float >)> calculate2dPSFPerBin( FiberTraceSet< double, unsigned short, float > const&,
                                                                     SpectrumSet< double, unsigned short, float, float > const&,
                                                                     TwoDPSFControl const&);
    template std::vector<PTR(PSFSet< double >)> calculate2dPSFPerBin( FiberTraceSet< double, unsigned short, float > const&,
                                                                      SpectrumSet< double, unsigned short, float, float > const&,
                                                                      TwoDPSFControl const&);
    template std::vector<PTR(PSFSet< float >)> calculate2dPSFPerBin( FiberTraceSet< float, unsigned short, float > const&,
                                                                     SpectrumSet< float, unsigned short, float, double > const&,
                                                                     TwoDPSFControl const&);
    template std::vector<PTR(PSFSet< double >)> calculate2dPSFPerBin( FiberTraceSet< float, unsigned short, float > const&,
                                                                      SpectrumSet< float, unsigned short, float, double > const&,
                                                                      TwoDPSFControl const&);
    template std::vector<PTR(PSFSet< float >)> calculate2dPSFPerBin( FiberTraceSet< double, unsigned short, float > const&,
                                                                     SpectrumSet< double, unsigned short, float, double > const&,
                                                                     TwoDPSFControl const&);
    template std::vector<PTR(PSFSet< double >)> calculate2dPSFPerBin( FiberTraceSet< double, unsigned short, float > const&,
                                                                      SpectrumSet< double, unsigned short, float, double > const&,
                                                                      TwoDPSFControl const&);
  }

  template class PSF< float >;
  template class PSFSet< float >;

}}}
