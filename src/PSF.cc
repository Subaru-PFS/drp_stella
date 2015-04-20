#include "pfs/drp/stella/PSF.h"

namespace pfs{ namespace drp{ namespace stella{

  template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
  PSF<ImageT, MaskT, VarianceT, WavelengthT>::PSF(PSF &psf, const bool deep)
    : _twoDPSFControl(psf.getTwoDPSFControl()),
      _iTrace(psf.getITrace()),
      _iBin(psf.getIBin()),
      _yMin(psf.getYLow()),
      _yMax(psf.getYHigh()),
      _imagePSF_XTrace(psf.getImagePSF_XTrace()),
      _imagePSF_YTrace(psf.getImagePSF_YTrace()),
      _imagePSF_ZTrace(psf.getImagePSF_ZTrace()),
      _imagePSF_XRelativeToCenter(psf.getImagePSF_XRelativeToCenter()),
      _imagePSF_YRelativeToCenter(psf.getImagePSF_YRelativeToCenter()),
      _imagePSF_ZNormalized(psf.getImagePSF_ZNormalized()),
      _imagePSF_Weight(psf.getImagePSF_Weight()),
      _pixelsFit(psf.getPixelsFit()),
      _isTwoDPSFControlSet(psf.isTwoDPSFControlSet()),
      _isPSFsExtracted(psf.isPSFsExtracted()),
      _surfaceFit(psf.getSurfaceFit()) 
  {
    if (deep){
      PTR(TwoDPSFControl) ptr(new TwoDPSFControl(*(psf.getTwoDPSFControl())));
      _twoDPSFControl.reset();
      _twoDPSFControl = ptr;
    }
  //  else{
  //    _twoDPSFControl(psf.getTwoDPSFControl());
  //  }
  }

  /// Set the _twoDPSFControl
  template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
  bool PSF<ImageT, MaskT, VarianceT, WavelengthT>::setTwoDPSFControl(PTR(TwoDPSFControl) &twoDPSFControl){
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
    return true;
  }

  template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
  bool PSF<ImageT, MaskT, VarianceT, WavelengthT>::extractPSFs(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTraceIn,
                                                               Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrumIn){
    ndarray::Array<ImageT, 2, 1> collapsedPSF = ndarray::allocate(1, 1);
    return extractPSFs(fiberTraceIn, spectrumIn, collapsedPSF);
  }

  template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
  bool PSF<ImageT, MaskT, VarianceT, WavelengthT>::extractPSFs(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTraceIn,
                                                               Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrumIn,
                                                               ndarray::Array<ImageT, 2, 1> const& collapsedPSF)
  {
    if (!_isTwoDPSFControlSet){
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: ERROR: _twoDPSFControl is not set";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (fiberTraceIn.getHeight() < _yMax){
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: ERROR: fiberTraceIn.getHeight(=" + to_string(fiberTraceIn.getHeight());
      message += " < _yMax = " + to_string(_yMax);
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    ndarray::Array<float, 1, 1> xCenters = copy(fiberTraceIn.getXCenters());

    ndarray::Array<double, 2, 1> trace_In = ndarray::allocate(_yMax - _yMin + 1, fiberTraceIn.getImage()->getWidth());
    ndarray::Array<double, 2, 1> mask_In = ndarray::allocate(_yMax - _yMin + 1, fiberTraceIn.getImage()->getWidth());
    ndarray::Array<double, 2, 1> stddev_In = ndarray::allocate(_yMax - _yMin + 1, fiberTraceIn.getImage()->getWidth());
    ndarray::Array<double, 1, 1> spectrum_In = ndarray::allocate(_yMax - _yMin + 1);
    ndarray::Array<ImageT, 1, 1> spectrum_T = ndarray::allocate(_yMax - _yMin + 1);
    spectrum_T.deep() = spectrumIn.getSpectrum()[ndarray::view(_yMin, _yMax+1)];
    ndarray::Array<double, 1, 1> spectrumVariance_In = ndarray::allocate(_yMax - _yMin + 1);

    ndarray::Array<double, 2, 1> D_A2_PixArray = math::Double(fiberTraceIn.getImage()->getArray());
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: D_A2_PixArray.getShape() = " << D_A2_PixArray.getShape() << ", _yMin = " << _yMin << ", _yMax = " << _yMax << endl;
    #endif
    trace_In = D_A2_PixArray[ndarray::view(_yMin, _yMax + 1)()];
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: traceIn = " << trace_In << endl;
    #endif

    ndarray::Array<double, 2, 1> D_A2_StdDevArray = math::Double(fiberTraceIn.getVariance()->getArray());
    for (auto itRowSDA = D_A2_StdDevArray.begin() + _yMin, itRowSD = stddev_In.begin(); itRowSDA != D_A2_StdDevArray.begin() + _yMax + 1; ++itRowSDA, ++itRowSD){
      for (auto itColSDA = itRowSDA->begin(), itColSD = itRowSD->begin(); itColSDA != itRowSDA->end(); ++itColSDA, ++itColSD){
        *itColSD = sqrt(*itColSDA > 0. ? *itColSDA : 1.);
      }
    }
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: D_A2_StdDevArray = " << D_A2_StdDevArray << endl;
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: stddev_In = " << stddev_In << endl;
    #endif

    int i, j;
    if (fabs(stddev_In.asEigen().maxCoeff(&i, &j)) < 0.000001){
      double D_MaxStdDev = fabs(stddev_In[i][j]);
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: ERROR: fabs(max(stddev_In))=" + to_string(D_MaxStdDev) + " < 0.000001";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

    auto itMRow = mask_In.begin();
    i = 0;
    for (auto itRow = fiberTraceIn.getMask()->getArray().begin() + _yMin; itRow <= fiberTraceIn.getMask()->getArray().begin() + _yMax; ++itRow, ++itMRow){
      auto itMCol = itMRow->begin();
      j = 0;
      for (auto itCol = itRow->begin(); itCol != itRow->end(); ++itCol, ++itMCol){
        *itMCol = *itCol == 0 ? 1 : 0;
        #ifdef __DEBUG_CALC2DPSF__
          cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: row " << i << ", col " << j << ": *itCol = " << *itCol << " => *itMCol set to " << *itMCol << endl;
          ++j;
        #endif
      }
      #ifdef __DEBUG_CALC2DPSF__
        ++i;
      #endif
    }
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: mask_In = " << mask_In << endl;
    #endif

    spectrum_In = math::Double(ndarray::Array<ImageT, 1, 1>(spectrumIn.getSpectrum()[ndarray::view(_yMin, _yMax + 1)]));
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: spectrum_In = " << spectrum_In << endl;
    #endif

    spectrumVariance_In = math::Double(ndarray::Array<VarianceT, 1, 1>(spectrumIn.getVariance()[ndarray::view(_yMin, _yMax + 1)]));
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: spectrumVariance_In = " << spectrumVariance_In << endl;
    #endif

    ndarray::Array<double, 1, 1> spectrumSigma = ndarray::allocate(spectrumVariance_In.size());
    spectrumSigma.asEigen() = Eigen::Array<double, Eigen::Dynamic, 1>(spectrumVariance_In.asEigen()).sqrt();
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: spectrumSigma = " << spectrumSigma << endl;
    #endif

    ndarray::Array<float, 1, 1> xCentersSwathF = ndarray::copy(xCenters[ndarray::view(_yMin, _yMax + 1)]);
    ndarray::Array<float, 1, 1> xCentersFloor = math::floor(ndarray::Array<const float, 1, 1>(xCentersSwathF), float(0));
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: xCentersSwathF = " << xCentersSwathF << endl;
    #endif

    ndarray::Array<size_t, 2, 2> minCenMax = math::calcMinCenMax(xCentersSwathF,
                                                                 fiberTraceIn.getFiberTraceFunction()->fiberTraceFunctionControl.xHigh,
                                                                 fiberTraceIn.getFiberTraceFunction()->fiberTraceFunctionControl.xLow,
                                                                 1,
                                                                 1);
    ndarray::Array<float, 1, 1> xCentersOffset = ndarray::copy(xCentersSwathF);
    xCentersOffset.deep() -= xCentersFloor;
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: minCenMax = " << minCenMax << endl;
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: xCentersOffset = " << xCentersOffset << endl;
    #endif

    _imagePSF_XRelativeToCenter.resize(0);
    _imagePSF_YRelativeToCenter.resize(0);
    _imagePSF_XTrace.resize(0);
    _imagePSF_YTrace.resize(0);
    _imagePSF_ZNormalized.resize(0);
    _imagePSF_ZTrace.resize(0);
    _imagePSF_Weight.resize(0);

    _imagePSF_XRelativeToCenter.reserve(1000);
    _imagePSF_YRelativeToCenter.reserve(1000);
    _imagePSF_XTrace.reserve(1000);
    _imagePSF_YTrace.reserve(1000);
    _imagePSF_ZNormalized.reserve(1000);
    _imagePSF_ZTrace.reserve(1000);
    _imagePSF_Weight.reserve(1000);
    double dX, dY, pixOffsetY, xStart, yStart;
    int i_xCenter, i_yCenter, i_Left, i_Right, i_Down, i_Up;
    int nPix = 0;

    /// Find emission lines
    /// set everything below signal threshold to zero
    ndarray::Array<int, 1, 1> traceCenterSignal = ndarray::allocate(spectrum_In.size());
    auto itSpec = spectrum_In.begin();
    for (auto itSig = traceCenterSignal.begin(); itSig != traceCenterSignal.end(); ++itSig, ++itSpec){
      *itSig = *itSpec < _twoDPSFControl->signalThreshold ? 0. : *itSpec;
    }
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: 1. traceCenterSignal = " << traceCenterSignal << endl;
    #endif

    /// look for signal wider than 2 FWHM
    if (!math::countPixGTZero(traceCenterSignal)){
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: ERROR: CountPixGTZero(traceCenterSignal=" << traceCenterSignal << ") returned FALSE" << endl;
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: ERROR: CountPixGTZero(traceCenterSignal) returned FALSE";
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: 2. traceCenterSignal = " << traceCenterSignal << endl;
    #endif

    /// identify emission lines
    int I_FirstWideSignal, I_FirstWideSignalStart, I_FirstWideSignalEnd;
    int I_MinWidth = int(_twoDPSFControl->yFWHM)+1;
    int I_StartIndex = 0;
    double D_MaxTimesApertureWidth = 3.5;
    ndarray::Array<int, 2, 2> I_A2_Limited = ndarray::allocate(_twoDPSFControl->nTermsGaussFit, 2);
    I_A2_Limited[ndarray::view()()] = 1;
    if (_twoDPSFControl->nTermsGaussFit == 5)
      I_A2_Limited[ndarray::view(4)()] = 0;
    ndarray::Array<ImageT, 2, 2> D_A2_Limits = ndarray::allocate(_twoDPSFControl->nTermsGaussFit, 2);
    /// 0: peak value
    /// 1: center position
    /// 2: sigma
    /// 3: constant background
    /// 4: linear background
    D_A2_Limits[0][0] = _twoDPSFControl->signalThreshold;
    D_A2_Limits[2][0] = _twoDPSFControl->yFWHM / (2. * 2.3548);
    D_A2_Limits[2][1] = _twoDPSFControl->yFWHM;
    if (_twoDPSFControl->nTermsGaussFit > 3)
      D_A2_Limits[3][0] = 0.;
    if (_twoDPSFControl->nTermsGaussFit > 4){
      D_A2_Limits[4][0] = 0.;
      D_A2_Limits[4][1] = 1000.;
    }
    ndarray::Array<ImageT, 1, 1> D_A1_GaussFit_Coeffs = ndarray::allocate(_twoDPSFControl->nTermsGaussFit);
    ndarray::Array<ImageT, 1, 1> D_A1_GaussFit_ECoeffs = ndarray::allocate(_twoDPSFControl->nTermsGaussFit);

    ndarray::Array<ImageT, 1, 1> D_A1_Guess = ndarray::allocate(_twoDPSFControl->nTermsGaussFit);
    float gaussCenterX;
    float gaussCenterY;
    int emissionLineNumber = 0;
    do{
      I_FirstWideSignal = math::firstIndexWithValueGEFrom(traceCenterSignal, I_MinWidth, I_StartIndex);
      #ifdef __DEBUG_CALC2DPSF__
        cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: I_FirstWideSignal found at index " << I_FirstWideSignal << ", I_StartIndex = " << I_StartIndex << endl;
      #endif
      if (I_FirstWideSignal < 0){
        cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: No emission line found" << endl;
        break;
      }
      I_FirstWideSignalStart = math::lastIndexWithZeroValueBefore(traceCenterSignal, I_FirstWideSignal);
      if (I_FirstWideSignalStart < 0){
        I_FirstWideSignalStart = 0;
      }
      #ifdef __DEBUG_CALC2DPSF__
        cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: I_FirstWideSignalStart = " << I_FirstWideSignalStart << endl;
      #endif

      I_FirstWideSignalEnd = math::firstIndexWithZeroValueFrom(traceCenterSignal, I_FirstWideSignal);
      #ifdef __DEBUG_CALC2DPSF__
        cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: I_FirstWideSignalEnd = " << I_FirstWideSignalEnd << endl;
      #endif

      if (I_FirstWideSignalEnd < 0){
        I_FirstWideSignalEnd = traceCenterSignal.size()-1;
      }
      #ifdef __DEBUG_CALC2DPSF__
        cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: End of first wide signal found at index " << I_FirstWideSignalEnd << endl;
      #endif

      if ((I_FirstWideSignalEnd - I_FirstWideSignalStart + 1) > (_twoDPSFControl->yFWHM * D_MaxTimesApertureWidth)){
        I_FirstWideSignalEnd = I_FirstWideSignalStart + int(D_MaxTimesApertureWidth * _twoDPSFControl->yFWHM);
      }

      /// Set start index for next run
      I_StartIndex = I_FirstWideSignalEnd+1;

      /// Fit Gaussian and Trace Aperture
      #ifdef __DEBUG_CALC2DPSF__
        cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: End of first wide signal found at index " << I_FirstWideSignalEnd << endl;
      #endif

      if (math::max(ndarray::Array<double, 1, 1>(spectrum_In[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)])) < _twoDPSFControl->saturationLevel){
        D_A2_Limits[0][1] = 1.5 * math::max(ndarray::Array<double, 1, 1>(spectrum_In[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)]));
        D_A2_Limits[1][0] = I_FirstWideSignalStart;
        D_A2_Limits[1][1] = I_FirstWideSignalEnd;
        if (_twoDPSFControl->nTermsGaussFit > 3)
          D_A2_Limits[3][1] = math::min(ndarray::Array<double, 1, 1>(spectrum_In[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)]));

        D_A1_Guess[0] = math::max(ndarray::Array<double, 1, 1>(spectrum_In[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)]));
        D_A1_Guess[1] = I_FirstWideSignalStart + (math::maxIndex(ndarray::Array<double, 1, 1>(spectrum_In[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)])));
        D_A1_Guess[2] = _twoDPSFControl->yFWHM / 2.3548;
        if (_twoDPSFControl->nTermsGaussFit > 3)
          D_A1_Guess[3] = math::min(ndarray::Array<double, 1, 1>(spectrum_In[ndarray::view(I_FirstWideSignalStart, I_FirstWideSignalEnd + 1)]));
        if (_twoDPSFControl->nTermsGaussFit > 4)
          D_A1_Guess[4] = (spectrum_In(I_FirstWideSignalEnd) - spectrum_In(I_FirstWideSignalStart)) / (I_FirstWideSignalEnd - I_FirstWideSignalStart);
        ndarray::Array<ImageT, 1, 1> D_A1_X = math::indGenNdArr(ImageT(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1));
        D_A1_X[ndarray::view()] += I_FirstWideSignalStart;
        ndarray::Array<ImageT, 1, 1> D_A1_Y = ndarray::allocate(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
        for (int ooo = 0; ooo < I_FirstWideSignalEnd - I_FirstWideSignalStart + 1; ++ooo){
          D_A1_Y[ooo] = spectrum_In(I_FirstWideSignalStart + ooo);
          if (D_A1_Y[ooo] < 0.01)
            D_A1_Y[ooo] = 0.01;
        }
        ndarray::Array<ImageT, 1, 1> D_A1_StdDev = ndarray::allocate(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
        for (int ooo = 0; ooo < I_FirstWideSignalEnd - I_FirstWideSignalStart + 1; ++ooo){
          if (fabs(spectrumSigma(I_FirstWideSignalStart + ooo)) < 0.1)
            D_A1_StdDev[ooo] = 0.1;
          else
            D_A1_StdDev[ooo] = spectrumSigma(I_FirstWideSignalStart + ooo);
        }
        #ifdef __DEBUG_CALC2DPSF__
          cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: emissionLineNumber = " << emissionLineNumber << ": Starting GaussFit: D_A1_X=" << D_A1_X << ", D_A1_Y = " << D_A1_Y << ", D_A1_StdDev = " << D_A1_StdDev << ", D_A1_Guess = " << D_A1_Guess << ", I_A2_Limited = " << I_A2_Limited << ", D_A2_Limits = " << D_A2_Limits << endl;
        #endif
        int iBackground = 0;
        if (_twoDPSFControl->nTermsGaussFit == 4)
          iBackground = 1;
        else if (_twoDPSFControl->nTermsGaussFit == 5)
          iBackground = 2;
        float yCenterOffset;
        bool success;
        success = MPFitGaussLim(D_A1_X,
                                D_A1_Y,
                                D_A1_StdDev,
                                D_A1_Guess,
                                I_A2_Limited,
                                D_A2_Limits,
                                iBackground,
                                false,
                                D_A1_GaussFit_Coeffs,
                                D_A1_GaussFit_ECoeffs);
        if (!success){
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: WARNING: Fit FAILED" << endl;
          #endif
        }
        else{
          success = false;
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
          #endif
          if ((D_A1_GaussFit_Coeffs[2] < (_twoDPSFControl->yFWHM / 1.5)) &&
              ((_twoDPSFControl->nTermsGaussFit < 5) ||
               ((_twoDPSFControl->nTermsGaussFit > 4) && (D_A1_GaussFit_Coeffs[4] < 1000.)))){
            ++emissionLineNumber;
            gaussCenterY = D_A1_GaussFit_Coeffs[1] + 0.5;
            success = true;
          }
          else{
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: D_A1_GaussFit_Coeffs(2)(=" << D_A1_GaussFit_Coeffs[2] << ") >= (_twoDPSFControl->yFWHM / 1.5)(=" << (_twoDPSFControl->yFWHM / 1.5) << ") || ((_twoDPSFControl->nTermsGaussFit(=" << _twoDPSFControl->nTermsGaussFit << ") < 5) || ((_twoDPSFControl->nTermsGaussFit > 4) && (D_A1_GaussFit_Coeffs(4)(=" << D_A1_GaussFit_Coeffs[4] << ") >= 1000.)) => Skipping emission line" << endl;
          }
        }
        if (success && (collapsedPSF.getShape()[0] > 2)){
          ndarray::Array<ImageT, 2, 1> indexRelToCenter = math::calcPositionsRelativeToCenter(ImageT(gaussCenterY), ImageT(4. * _twoDPSFControl->yFWHM));
          ndarray::Array<ImageT, 2, 1> arrA = ndarray::allocate(indexRelToCenter.getShape()[0], 2);
          arrA[ndarray::view()(0)].deep() = indexRelToCenter[ndarray::view()(1)];
          ndarray::Array<size_t, 1, 1> indVec = ndarray::allocate(indexRelToCenter.getShape()[0]);
          for (int iInd = 0; iInd < indexRelToCenter.getShape()[0]; ++iInd)
            indVec[iInd] = size_t(indexRelToCenter[iInd][0]);
          ndarray::Array<ImageT, 1, 1> yDataVec = math::getSubArray(spectrum_T, indVec);
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: D_A1_Y = " << D_A1_Y << ": indVec = " << indVec << endl;
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: yDataVec = " << yDataVec << endl;
          #endif
          arrA[ndarray::view()(1)].deep() = yDataVec;//yDataVec;
          
          ndarray::Array< ImageT, 1, 1 > range = ndarray::allocate(2);
          range[0] = _twoDPSFControl->xCorRangeLowLimit;
          range[1] = _twoDPSFControl->xCorRangeHighLimit;
          const float stepSize = _twoDPSFControl->xCorStepSize; 
          
          double maxA = yDataVec.asEigen().maxCoeff();
          ndarray::Array< ImageT, 1, 1 > yValCollapsedPSF = ndarray::allocate(collapsedPSF.getShape()[0]);
          yValCollapsedPSF.deep() = collapsedPSF[ndarray::view()(1)];
          double maxCollapsedPSF = yValCollapsedPSF.asEigen().maxCoeff();
          collapsedPSF[ndarray::view()(1)].deep() = collapsedPSF[ndarray::view()(1)] * maxA / maxCollapsedPSF;
          
          ImageT xCorMinPos = math::xCor(arrA,/// x must be 'y' relative to center
                                         collapsedPSF,/// x is 'y' relative to center
                                         range,
                                         stepSize);
//          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": gaussCenterY = " << gaussCenterY << endl;
//          #endif
          gaussCenterY += xCorMinPos;
//          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": xCorMinPos = " << xCorMinPos << ": gaussCenterY = " << gaussCenterY << endl;
//          #endif
        }
        if (!success){
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: WARNING: Fit FAILED" << endl;
          #endif
        }
        else{
//          #ifdef __DEBUG_CALC2DPSF__
//            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
//          #endif
//          if ((D_A1_GaussFit_Coeffs[2] < (_twoDPSFControl->yFWHM / 1.5)) &&
//              ((_twoDPSFControl->nTermsGaussFit < 5) ||
//               ((_twoDPSFControl->nTermsGaussFit > 4) && (D_A1_GaussFit_Coeffs[4] < 1000.)))){
//            ++emissionLineNumber;
//            gaussCenterY = D_A1_GaussFit_Coeffs[1] + 0.5;
//            float yCenterOffset = gaussCenterY - floor(gaussCenterY);
            yCenterOffset = gaussCenterY - std::floor(gaussCenterY);
            dY = 0.5 - yCenterOffset;
            #ifdef __DEBUG_CALC2DPSF__
              cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": dY = " << dY << endl;
            #endif
            i_Down = int(gaussCenterY - (2. * _twoDPSFControl->yFWHM));
            #ifdef __DEBUG_CALC2DPSF__
              cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": i_Down = " << i_Down << endl;
            #endif
            if (i_Down >= 0){
              i_Up = int(gaussCenterY + (2. * _twoDPSFControl->yFWHM));
              #ifdef __DEBUG_CALC2DPSF__
                cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": fiberTraceIn.getShape()[0] = " << trace_In.getShape()[0] << ", i_Up = " << i_Up << endl;
              #endif
              if (i_Up < trace_In.getShape()[0]){
                /// x-Centers from Gaussian center
                ndarray::Array<float, 1, 1> yCentersFromGaussCenter = math::indGenNdArr(float(i_Up - i_Down + 1));
                yCentersFromGaussCenter.deep() += fiberTraceIn.getFiberTraceFunction()->yCenter + fiberTraceIn.getFiberTraceFunction()->yLow + _yMin + i_Down + yCenterOffset;
                ndarray::Array<float, 1, 1> xCentersFromGaussCenter = math::calculateXCenters(fiberTraceIn.getFiberTraceFunction(), yCentersFromGaussCenter);
                
                /// original x-Centers
                ndarray::Array<float, 1, 1> yCentersTemp = math::indGenNdArr(float(i_Up - i_Down + 1));
                yCentersTemp.deep() += fiberTraceIn.getFiberTraceFunction()->yCenter + fiberTraceIn.getFiberTraceFunction()->yLow + _yMin + i_Down + 0.5;
                ndarray::Array<float, 1, 1> xCentersTemp = math::calculateXCenters(fiberTraceIn.getFiberTraceFunction(), yCentersTemp);
                #ifdef __DEBUG_CALC2DPSF__
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": _yMin = " << _yMin << ", i_Down = " << i_Down << ", dY = " << dY << endl;
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": yCentersFromGaussCenter = " << yCentersFromGaussCenter << endl;
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": xCentersFromGaussCenter = " << xCentersFromGaussCenter << endl;
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": xCentersSwathF.getShape() = " << xCentersSwathF.getShape() << endl;
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": xCentersSwathF[i_Down:i_Up+1] = " << xCentersSwathF[ndarray::view(i_Down, i_Up+1)] << endl;
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": xCentersTemp = " << xCentersTemp << endl;
                  //exit(EXIT_FAILURE);
                #endif
                
                int nPixPSF = 0;
                double sumPSF = 0.;
                int yMinRel = i_Down - std::floor(gaussCenterY);
                #ifdef __DEBUG_CALC2DPSF__
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": yMinRel = " << yMinRel << endl;
                #endif
                for (int iY = 0; iY <= i_Up - i_Down; ++iY){
/*                  /// difference in trace center between rows iY and iY(GaussCenterY)
                  #ifdef __DEBUG_CALC2DPSF__
                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": xCentersOffset_In.getShape()[0] = " << xCentersOffset_In.getShape()[0] << endl;
                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": gaussCenterY = " << gaussCenterY << ", xCentersOffset_In[i_Down + iY=" << i_Down + iY << "] = " << xCentersOffset_In[i_Down + iY] << endl;
                  #endif
 */
                  float xCenterOffset;
                  if (floor(xCentersFromGaussCenter[iY]) == floor(xCentersSwathF[i_Down + iY])){
                    xCenterOffset = xCentersFromGaussCenter[iY] - floor(xCentersFromGaussCenter[iY]);
                    dX = 0.5 - xCenterOffset;
                  }
                  else if (floor(xCentersFromGaussCenter[iY]) < floor(xCentersSwathF[i_Down + iY])){
                    xCenterOffset = xCentersFromGaussCenter[iY] - floor(xCentersFromGaussCenter[iY]) - 1.;
                    dX = 0.5 - xCenterOffset;
                  }
                  else{
                    xCenterOffset = xCentersFromGaussCenter[iY] - floor(xCentersFromGaussCenter[iY]) + 1;
                    dX = 0.5 - xCenterOffset;
                  }

                  #ifdef __DEBUG_CALC2DPSF__
                    cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": iY = " << iY << ": xCentersFromGaussCenter[iY]=" << xCentersFromGaussCenter[iY] << ", xCentersTemp[iY]=" << xCentersTemp[iY] << ", xCentersSwathF[iY]=" << xCentersSwathF[iY] << ": xCenterOffset = " << xCenterOffset << ": dX = " << dX << endl;
                  #endif

                  /// most left pixel of FiberTrace affected by PSF of (emissionLineNumber-1) in this row
                  i_Left = int(minCenMax[i_Down + iY][1] - minCenMax[i_Down + iY][0] + xCenterOffset - (2. * _twoDPSFControl->xFWHM));

                  /// most right pixel affected by PSF of (emissionLineNumber-1) in this row
                  i_Right = int(minCenMax[i_Down + iY][1] - minCenMax[i_Down + iY][0] + xCenterOffset + (2. * _twoDPSFControl->xFWHM));
                  #ifdef __DEBUG_CALC2DPSF__
                    cout << "PSF trace" << _iTrace << " bin " << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": i_Left = " << i_Left << ", i_Right = " << i_Right << endl;
                  #endif
                  if (i_Left < 0)
                    i_Left = 0;
                  if (i_Right >= fiberTraceIn.getImage()->getWidth())
                    i_Right = fiberTraceIn.getImage()->getWidth() - 1;
                  /// HERE!!!
//                  xStart = int(xCentersOffset[i_Down + iY]) + 0.5 - xCentersOffset[i_Down + iY] + dTraceGaussCenterX - i_xCenter + i_Left;
//                  #ifdef __DEBUG_CALC2DPSF__
//                    cout << "int(xCentersOffset[i_Down=" << i_Down << "]=" << xCentersOffset[i_Down] << "), int(xCentersOffset[i_Up=" << i_Up << "]=" <<xCentersOffset[i_Up] << ")" << endl;
//                  #endif
                    
//                  yStart = i_Down - i_yCenter - pixOffsetY;
//                  #ifdef __DEBUG_CALC2DPSF__
//                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": dTrace = " << dTrace << ": i_Left = " << i_Left << ", i_Right = " << i_Right << endl;
//                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": xStart = " << xStart << endl;
//                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": yStart = " << yStart << endl;
//                  #endif*/
                  int rowTrace = i_Down + iY;
                  int xMinRel = minCenMax[rowTrace][0] - minCenMax[rowTrace][1];
                  #ifdef __DEBUG_CALC2DPSF__
                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": xMinRel = " << xMinRel << endl;
                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": i_Right = " << i_Right << ", i_Left = " << i_Left << endl;
                  #endif
                  for (int iX = 0; iX <= i_Right - i_Left; ++iX){
                    int colTrace = i_Left + iX;
                    #ifdef __DEBUG_CALC2DPSF__
                      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": xCentersFromGaussCenter[" << iY << "] = " << xCentersFromGaussCenter[iY] << ", xCentersSwathF[" << iY << "] = " << xCentersSwathF[iY] << endl;
                      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": iX = " << iX << ": iY = " << iY << endl;
                      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": trace_In[i_Down + iY = " << i_Down + iY << "][i_Left + iX = " << i_Left + iX << "] = " << trace_In[i_Down + iY][i_Left + iX] << endl;
                    #endif
                    _imagePSF_XRelativeToCenter.push_back(float(dX + float(xMinRel + iX)));
                    _imagePSF_YRelativeToCenter.push_back(float(dY + float(yMinRel + iY)));
                    _imagePSF_ZNormalized.push_back(float(trace_In[rowTrace][colTrace]));
                    _imagePSF_ZTrace.push_back(float(trace_In[rowTrace][colTrace]));
                    _imagePSF_Weight.push_back(fabs(trace_In[rowTrace][colTrace]) > 0.000001 ? float(1. / sqrt(fabs(trace_In[rowTrace][colTrace]))) : 0.1);//trace_In(i_Down+iY, i_Left+iX) > 0 ? sqrt(trace_In(i_Down+iY, i_Left+iX)) : 0.0000000001);//stddev_In(i_Down+iY, i_Left+iX) > 0. ? 1./pow(stddev_In(i_Down+iY, i_Left+iX),2) : 1.);
                    _imagePSF_XTrace.push_back(float(colTrace));
                    _imagePSF_YTrace.push_back(float(rowTrace + _yMin));
                    #ifdef __DEBUG_CALC2DPSF__
                      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": x = " << _imagePSF_XRelativeToCenter[nPix] << ", y = " << _imagePSF_YRelativeToCenter[nPix] << ": val = " << trace_In[i_Down + iY][i_Left + iX] << " = " << _imagePSF_ZNormalized[nPix] << "; XOrig = " << _imagePSF_XTrace[nPix] << ", YOrig = " << _imagePSF_YTrace[nPix] << endl;
                    #endif
                    ++nPix;
                    ++nPixPSF;
                    sumPSF += trace_In[rowTrace][colTrace];
                    #ifdef __DEBUG_CALC2DPSF__
                      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1 = " << emissionLineNumber-1 << ": nPixPSF = " << nPixPSF << ", sumPSF = " << sumPSF << endl;
                    #endif
//                    string message("debug exit");
//                    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
                  }/// end for (int iX = 0; iX <= i_Right - i_Left; ++iX){
                }/// end for (int iY = 0; iY <= i_Up - i_Down; ++iY){
                int pixelNo = _imagePSF_ZNormalized.size() - nPixPSF;
                #ifdef __DEBUG_CALC2DPSF__
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1 = " << emissionLineNumber-1 << ": *_imagePSF_ZNormalized.begin()=" << *(_imagePSF_ZNormalized.begin()) << " *(_imagePSF_ZNormalized.end()-1)=" << *(_imagePSF_ZNormalized.end()-1) << endl;
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1 = " << emissionLineNumber-1 << ": _imagePSF_ZNormalized[nPix=" << nPix << " - nPixPSF=" << nPixPSF << " = " << nPix - nPixPSF << " = " << _imagePSF_ZNormalized[nPix-nPixPSF] << endl;
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1 = " << emissionLineNumber-1 << ": _imagePSF_ZNormalized[nPix-1=" << nPix-1 << "] = " << _imagePSF_ZNormalized[nPix-1] << endl;
                #endif
                if (fabs(sumPSF) < 0.00000001){
                  string message("PSF trace");
                  message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: emissionLineNumber-1 = " + to_string(emissionLineNumber-1);
                  message += ": ERROR: sumPSF == 0";
                  cout << message << endl;
                  throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
                }
                for (std::vector<float>::iterator iter = _imagePSF_ZNormalized.end() - nPixPSF; iter < _imagePSF_ZNormalized.end(); ++iter){
                  #ifdef __DEBUG_CALC2DPSF__
                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1 = " << emissionLineNumber-1 << ": _imagePSF_ZNormalized[pixelNo=" << pixelNo << "] = " << _imagePSF_ZNormalized[pixelNo] << ", sumPSF = " << sumPSF << endl;
                  #endif
                  (*iter) = (*iter) / sumPSF;
                  #ifdef __DEBUG_CALC2DPSF__
                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1 = " << emissionLineNumber-1 << ": _imagePSF_ZNormalized[pixelNo=" << pixelNo << "] = " << _imagePSF_ZNormalized[pixelNo] << endl;
                  #endif
                  ++pixelNo;
                }
              }/// end if (i_Up < trace_In.getShape()[0]){
              else{
              #ifdef __DEBUG_CALC2DPSF__
                cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1 = " << emissionLineNumber-1 << ": WARNING: i_Up(=" << i_Up << ") >= trace_In.getShape()[0](=" << trace_In.getShape()[0] << endl;
              #endif
              }
            }
            else{
              #ifdef __DEBUG_CALC2DPSF__
                cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1 = " << emissionLineNumber-1 << ": WARNING: i_Down = " << i_Down << " < 0" << endl;
              #endif
            }
          //}
//          else{
//            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: D_A1_GaussFit_Coeffs(2)(=" << D_A1_GaussFit_Coeffs[2] << ") >= (_twoDPSFControl->yFWHM / 1.5)(=" << (_twoDPSFControl->yFWHM / 1.5) << ") || ((_twoDPSFControl->nTermsGaussFit(=" << _twoDPSFControl->nTermsGaussFit << ") < 5) || ((_twoDPSFControl->nTermsGaussFit > 4) && (D_A1_GaussFit_Coeffs(4)(=" << D_A1_GaussFit_Coeffs[4] << ") >= 1000.)) => Skipping emission line" << endl;
//          }
        }/// end if MPFitGaussLim
      }/// end if (max(spectrum_In(Range(I_FirstWideSignalStart, I_FirstWideSignalEnd))) < _twoDPSFControl->saturationLevel){
    } while(true);
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: nPix = " << nPix << ", _imagePSF_XRelativeToCenter.size() = " << _imagePSF_XRelativeToCenter.size() << endl;
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: nPix = " << nPix << ", _imagePSF_XTrace.size() = " << _imagePSF_XTrace.size() << endl;
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: nPix = " << nPix << ", _imagePSF_YRelativeToCenter.size() = " << _imagePSF_YRelativeToCenter.size() << endl;
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: nPix = " << nPix << ", _imagePSF_YTrace.size() = " << _imagePSF_YTrace.size() << endl;
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: nPix = " << nPix << ", _imagePSF_ZTrace.size() = " << _imagePSF_ZTrace.size() << endl;
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: nPix = " << nPix << ", _imagePSF_ZNormalized.size() = " << _imagePSF_ZNormalized.size() << endl;
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: nPix = " << nPix << ", _imagePSF_Weight.size() = " << _imagePSF_Weight.size() << endl;
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: xFWHM = " << _twoDPSFControl->xFWHM << ", yFWHM = " << _twoDPSFControl->yFWHM << endl;
      std::ofstream of;
      std::string ofname = __DEBUGDIR__ + std::string("pix_x_xo_y_yo_val_norm_weight_iBin");
      if (_iBin < 10)
        ofname += std::string("0");
      ofname += to_string(_iBin)+std::string(".dat");
      of.open(ofname);
      if (!of){
        string message("PSF::extractPSFs: ERROR: Could not open file <");
        message += ofname + ">";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      for (int iPix=0; iPix<nPix; ++iPix){
        of << _imagePSF_XRelativeToCenter[iPix];
        of << " " << _imagePSF_XTrace[iPix];
        of << " " << _imagePSF_YRelativeToCenter[iPix];
        of << " " << _imagePSF_YTrace[iPix];
        of << " " << _imagePSF_ZTrace[iPix];
        of << " " << _imagePSF_ZNormalized[iPix];
        of << " " << _imagePSF_Weight[iPix] << endl;
      }
      of.close();
      cout << "ofname = <" << ofname << "> written" << endl;
    #endif
    if (nPix != _imagePSF_XRelativeToCenter.size()){
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: ERROR: nPix != _imagePSF_XRelativeToCenter.size()";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (nPix != _imagePSF_XTrace.size()){
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: ERROR: nPix != _imagePSF_XTrace.size()";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (nPix != _imagePSF_YRelativeToCenter.size()){
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: ERROR: nPix != _imagePSF_YRelativeToCenter.size()";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (nPix != _imagePSF_YTrace.size()){
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: ERROR: nPix != _imagePSF_YTrace.size()";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (nPix != _imagePSF_ZTrace.size()){
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: ERROR: nPix != _imagePSF_ZTrace.size()";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (nPix != _imagePSF_ZNormalized.size()){
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: ERROR: nPix != _imagePSF_ZNormalized.size()";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (nPix != _imagePSF_Weight.size()){
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: ERROR: nPix != _imagePSF_Weight.size()";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    _isPSFsExtracted = true;
    return true;
  }

  /*template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
  bool PSF<ImageT, MaskT, VarianceT, WavelengthT>::fitPSFKernel()
  {
    if (!_isPSFsExtracted){
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string(_iBin) + ":fitPSFKernel: ERROR: _isPSFsExtracted == false";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    /// fit bispline
    if (!_surfaceFit.doFit(_imagePSF_XRelativeToCenter, _imagePSF_YRelativeToCenter, _imagePSF_ZNormalized, _imagePSF_Weight, _twoDPSFControl->nKnotsX, _twoDPSFControl->nKnotsY, _twoDPSFControl->smooth)){
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::fitPSFKernel: ERROR: doFit returned FALSE";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }

  /*    /// prepare input for kriging
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: _imagePSF_XRelativeToCenter.size() = " << _imagePSF_XRelativeToCenter.size() << ", _imagePSF_YRelativeToCenter.size() = " << _imagePSF_YRelativeToCenter.size() << ", _imagePSF_ZNormalized.size() = " << _imagePSF_ZNormalized.size() << ", nPix = " << nPix << endl;
      for (int i=0; i<nPix; i++)
        cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: i=" << i << ": (*_imagePSF_XRelativeToCenter)[i] = " << (*_imagePSF_XRelativeToCenter)[i] << ", (*_imagePSF_YRelativeToCenter)[i] = " << (*_imagePSF_YRelativeToCenter)[i] << ", (*_imagePSF_ZNormalized)[i] = " << (*_imagePSF_ZNormalized)[i] << endl;
    #endif
    std::vector<double> krigingInput_X;
    krigingInput_X.reserve(_twoDPSFControl->nKrigingPointsX * _twoDPSFControl->nKrigingPointsY);
    std::vector<double> krigingInput_Y;
    krigingInput_Y.reserve(_twoDPSFControl->nKrigingPointsX * _twoDPSFControl->nKrigingPointsY);
    std::vector<double> krigingInput_Val;
    krigingInput_Val.reserve(_twoDPSFControl->nKrigingPointsX * _twoDPSFControl->nKrigingPointsY);
    double xRangeMin = (*(std::min_element(_imagePSF_XRelativeToCenter.begin(), _imagePSF_XRelativeToCenter.end())));
    double xRangeMax = (*(std::max_element(_imagePSF_XRelativeToCenter.begin(), _imagePSF_XRelativeToCenter.end()))) + 0.000001;
    double yRangeMin = (*(std::min_element(_imagePSF_YRelativeToCenter.begin(), _imagePSF_YRelativeToCenter.end())));
    double yRangeMax = (*(std::max_element(_imagePSF_YRelativeToCenter.begin(), _imagePSF_YRelativeToCenter.end()))) + 0.000001;
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: xRangeMin = " << xRangeMin << ", xRangeMax = " << xRangeMax << endl;
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: yRangeMin = " << yRangeMin << ", yRangeMax = " << yRangeMax << endl;
    #endif
    double xStep = (xRangeMax - xRangeMin) / _twoDPSFControl->nKrigingPointsX;
    double yStep = (yRangeMax - yRangeMin) / _twoDPSFControl->nKrigingPointsY;
    double xCenterOrig = xRangeMin - (xStep / 2.);
    double yCenterOrig = yRangeMin - (yStep / 2.);
    double xCenter, yCenter;
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: xStep = " << xStep << ", yStep = " << yStep << endl;
    #endif
    double value, xEnd, yEnd;
    int nPixelsInRange;
    blitz::Array<double, 1> moments(4);
    xCenter = xCenterOrig;
    for (int ix = 0; ix < _twoDPSFControl->nKrigingPointsX; ++ix){
      xCenter += xStep;
      yCenter = yCenterOrig;
      for (int iy = 0; iy < _twoDPSFControl->nKrigingPointsY; ++iy){
        yCenter += yStep;
        #ifdef __DEBUG_CALC2DPSF__
          cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: ix=" << ix << ", iy=" << iy << ": xCenter = " << xCenter << ", yCenter = " << yCenter << endl;
        #endif
        xStart = xCenter - (xStep/2.);
        xEnd = xCenter + (xStep/2.);
        yStart = yCenter - (yStep/2.);
        yEnd = yCenter + (yStep/2.);
        #ifdef __DEBUG_CALC2DPSF__
          cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: ix=" << ix << ", iy=" << iy << ": xStart = " << xStart << ", xEnd = " << xEnd << endl;
          cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: ix=" << ix << ", iy=" << iy << ": yStart = " << yStart << ", yEnd = " << yEnd << endl;
        #endif
        nPixelsInRange = 0;
        std::vector<double> valuesInRange;
        std::vector<double> valuesInRange_XOrig;
        std::vector<double> valuesInRange_YOrig;
        for (int ipix = 0; ipix < _imagePSF_XRelativeToCenter.size(); ++ipix){
          if (((*_imagePSF_XRelativeToCenter)[ipix] >= xStart) && ((*_imagePSF_XRelativeToCenter)[ipix] < xEnd) && ((*_imagePSF_YRelativeToCenter)[ipix] >= yStart) && ((*_imagePSF_YRelativeToCenter)[ipix] < yEnd)){
            #ifdef __DEBUG_CALC2DPSF__
              cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: ix=" << ix << ", iy=" << iy << ": pixel ipix=" << ipix << " in range: (*_imagePSF_XRelativeToCenter)[ipix] = " << (*_imagePSF_XRelativeToCenter)[ipix] << ", (*_imagePSF_YRelativeToCenter)[ipix] = " << (*_imagePSF_YRelativeToCenter)[ipix] << ", (*_imagePSF_ZNormalized)[ipix] = " << (*_imagePSF_ZNormalized)[ipix] << endl;
            #endif
            valuesInRange.push_back((*_imagePSF_ZNormalized)[ipix]);
            valuesInRange_XOrig.push_back((*_imagePSF_XTrace)[ipix]);
            valuesInRange_YOrig.push_back((*_imagePSF_YTrace)[ipix]);
            ++nPixelsInRange;
          }
        }
        #ifdef __DEBUG_CALC2DPSF__
          cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: ix=" << ix << ", iy=" << iy << ": nPixelsInRange: " << nPixelsInRange << endl;
        #endif
        if (nPixelsInRange > 1){
          blitz::Array<double, 1> tempArr(valuesInRange.data(), blitz::shape(valuesInRange.size()), blitz::neverDeleteData);
          moments = math::Moment(tempArr, 2);
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: ix=" << ix << ", iy=" << iy << ": 1. moments = " << moments << endl;
          #endif
          for (int ipix=nPixelsInRange-1; ipix >= 0; --ipix){
            if (std::pow(valuesInRange[ipix] - moments(0), 2) > (3. * moments(1))){
              #ifdef __DEBUG_CALC2DPSF__
                cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: ix=" << ix << ", iy=" << iy << ": rejecting pixel ipix = " << ipix << ": valuesInRange_XOrig[ipix] = " << valuesInRange_XOrig[ipix] << ", valuesInRange_YOrig[ipix] = " << valuesInRange_YOrig[ipix] << ", valuesInRange[ipix] = " << valuesInRange[ipix] << endl;
              #endif
              valuesInRange.erase(valuesInRange.begin() + ipix);
              valuesInRange_XOrig.erase(valuesInRange_XOrig.begin() + ipix);
              valuesInRange_YOrig.erase(valuesInRange_YOrig.begin() + ipix);
            }
          }
          blitz::Array<double, 1> tempArrNew(valuesInRange.data(), blitz::shape(valuesInRange.size()), blitz::neverDeleteData);
          moments = math::Moment(tempArrNew, 1);
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: ix=" << ix << ", iy=" << iy << ": 2. moments = " << moments << endl;
          #endif
          krigingInput_X.push_back(xCenter);
          krigingInput_Y.push_back(yCenter);
          krigingInput_Val.push_back(moments(0));
        }
      }
    }
    #ifdef __DEBUG_CALC2DPSF__
      std::ofstream ofkrig_in;
      std::string ofname_in = __DEBUGDIR__ + std::string("kriging_in_pix_x_y_val");
      if (_iBin < 10)
        ofname += std::string("0");
      ofname += to_string(_iBin)+std::string(".dat");
      ofkrig_in.open(ofname_in);
      for (int i=0; i<krigingInput_X.size(); i++)
        ofkrig_in << krigingInput_X[i] << " " << krigingInput_Y[i] << " " << krigingInput_Val[i] << endl;
      ofkrig_in.close();
    #endif

    CGeostat krig;
    const size_t dim_cspace = 2;
    krig.initialize(krigingInput_Val.size(), dim_cspace);
    gsl_vector *lower = gsl_vector_alloc(dim_cspace);
    gsl_vector *upper = gsl_vector_alloc(dim_cspace);

    blitz::Array<double, 1> D_A1_KrigingInput_X(krigingInput_X.data(), blitz::shape(krigingInput_X.size()), blitz::neverDeleteData);
    blitz::Array<double, 1> D_A1_KrigingInput_Y(krigingInput_Y.data(), blitz::shape(krigingInput_Y.size()), blitz::neverDeleteData);
    gsl_vector_set(lower, 0, blitz::min(D_A1_KrigingInput_X));
    gsl_vector_set(lower, 1, blitz::min(D_A1_KrigingInput_Y));
    gsl_vector_set(upper, 0, blitz::max(D_A1_KrigingInput_X));
    gsl_vector_set(upper, 1, blitz::max(D_A1_KrigingInput_Y));

    krig.setDomain(lower, upper);
    gsl_vector_free(lower);
    gsl_vector_free(upper);

    gsl_vector *pixPos = gsl_vector_alloc(dim_cspace);
    for (int iPix=0; iPix<krigingInput_Val.size(); ++iPix){
      gsl_vector_set(pixPos, 0, krigingInput_X[iPix]);
      gsl_vector_set(pixPos, 1, krigingInput_Y[iPix]);
      krig.setCoordinate(iPix, pixPos);
      krig.setData(iPix, krigingInput_Val[iPix]);
    }

    krig.estimate(CVariogram::VARIO_SPH, 0, 1.);
    double pred, var;
    std::vector<double> pixelsFit(_imagePSF_ZNormalized.size());
  */
/*
    return true;
  }*/

  template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
  bool PSF<ImageT, MaskT, VarianceT, WavelengthT>::calculatePSF()
  {
    _pixelsFit.resize(_imagePSF_XRelativeToCenter.size());
    if (!_surfaceFit.estimate(_imagePSF_XRelativeToCenter, _imagePSF_YRelativeToCenter, _pixelsFit)){
      string message("PSF trace");
      message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::calculatePSFs: ERROR: surfaceFit.estimate returned FALSE";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    #ifdef __DEBUG_CALC2DPSF__
      std::ofstream ofkrig;
      std::string fname = __DEBUGDIR__ + std::string("psf_x_y_in_fit_iBin");
      if (_iBin < 10)
        fname += std::string("0");
      fname += to_string(_iBin)+std::string(".dat");
      ofkrig.open(fname);
      if (!ofkrig){
        string message("PSF::calculatePSF: ERROR: Could not open file <");
        message += fname + ">";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      //    #endif

      for (int iPix = 0; iPix < _imagePSF_XTrace.size(); ++iPix){
        //      gsl_vector_set(pixPos, 0, (*_imagePSF_XRelativeToCenter)[iPix]);
        //      gsl_vector_set(pixPos, 1, (*_imagePSF_YRelativeToCenter)[iPix]);
        //      krig.getPredictData(pred, var, pixPos);
        //      (*_pixelsFit)[iPix] = pred;
        //      #ifdef __DEBUG_CALC2DPSF__
        cout << "PSF trace" << _iTrace << " bin" << _iBin << "::calculatePSF: iPix=" << iPix << ": x=" << _imagePSF_XRelativeToCenter[iPix] << ", y=" << _imagePSF_YRelativeToCenter[iPix] << ": original pixel value = " << _imagePSF_ZNormalized[iPix] << ", predicted pixel value = " << _pixelsFit[iPix] << ", difference = " << _imagePSF_ZNormalized[iPix] - _pixelsFit[iPix] << endl;
        ofkrig << _imagePSF_XRelativeToCenter[iPix] << " " << _imagePSF_YRelativeToCenter[iPix] << " " << _imagePSF_ZNormalized[iPix] << " " << _pixelsFit[iPix] << endl;
        //      #endif
        //      _pixelsFit.push_back(pred);
      }
      //    gsl_vector_free(pixPos);
      //    #ifdef __DEBUG_CALC2DPSF__
      ofkrig.close();
    #endif
    return true;
  }
  
  template< typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT >    
  std::vector< ImageT > PSF<ImageT, MaskT, VarianceT, WavelengthT>::reconstructFromThinPlateSplineFit(double const regularization){
    ndarray::Array<float, 1, 1> xArr = ndarray::external(_imagePSF_XRelativeToCenter.data(), ndarray::makeVector(int(_imagePSF_XRelativeToCenter.size())), ndarray::makeVector(1));
    ndarray::Array<float, 1, 1> yArr = ndarray::external(_imagePSF_YRelativeToCenter.data(), ndarray::makeVector(int(_imagePSF_YRelativeToCenter.size())), ndarray::makeVector(1));
    ndarray::Array<float, 1, 1> zArr = ndarray::external(_imagePSF_ZNormalized.data(), ndarray::makeVector(int(_imagePSF_ZNormalized.size())), ndarray::makeVector(1));
    math::ThinPlateSpline<float, float> tps = math::ThinPlateSpline<float, float>(xArr,
                                                                                  yArr,
                                                                                  zArr,
                                                                                  regularization);
    ndarray::Array< float, 2, 1 > zFitArr = tps.fitArray(xArr,
                                                         yArr,
                                                         false);
    std::vector< ImageT > zVec(_imagePSF_XRelativeToCenter.size());
    for (int i = 0; i < _imagePSF_XRelativeToCenter.size(); ++i)
      zVec[i] = ImageT(zFitArr[i][0]);
    return zVec;
  }
  
  template< typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT >    
  bool PSF<ImageT, MaskT, VarianceT, WavelengthT>::setImagePSF_ZFit(ndarray::Array<ImageT, 1, 1> const& zFit){
    if (zFit.getShape()[0] != _imagePSF_XRelativeToCenter.size()){
      cout << "PSF::setImagePSF_ZFit: zFit.getShape()[0]=" << zFit.getShape()[0] << " != _imagePSF_XRelativeToCenter.size()=" << _imagePSF_XRelativeToCenter.size() << " => returning FALSE" << endl;
    }
    _imagePSF_ZFit.resize(zFit.getShape()[0]);
    for (int i = 0; i < zFit.getShape()[0]; ++i)
      _imagePSF_ZFit[i] = float(zFit[i]);
    return true;
  }

  template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>    
  bool PSFSet<ImageT, MaskT, VarianceT, WavelengthT>::setPSF(const size_t i,     /// which position?
                                                             const PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>) & psf)
  {
    if (i > _psfs->size()){
      string message("PSFSet::setPSF: ERROR: i=");
      message += to_string(i) + " > _psfs->size()=" + to_string(_psfs->size()) + " => Returning FALSE";
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    if (i == static_cast<int>(_psfs->size())){
        _psfs->push_back(psf);
    }
    else{
      (*_psfs)[i] = psf;
    }
    return true;
  }

  /// add one PSF to the set
  template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>    
  void PSFSet<ImageT, MaskT, VarianceT, WavelengthT>::addPSF(const PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>) & psf /// the Spectrum to add
  )
  {
    _psfs->push_back(psf);
  }

  template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
  PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>)& PSFSet<ImageT, MaskT, VarianceT, WavelengthT>::getPSF(const size_t i){
    if (i >= _psfs->size()){
      string message("PSFSet::getPSF(i=");
      message += to_string(i) + string("): ERROR: i > _psfs->size()=") + to_string(_psfs->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return _psfs->at(i); 
  }

  template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
  const PTR(const PSF<ImageT, MaskT, VarianceT, WavelengthT>) PSFSet<ImageT, MaskT, VarianceT, WavelengthT>::getPSF(const size_t i) const { 
    if (i >= _psfs->size()){
      string message("PSFSet::getPSF(i=");
      message += to_string(i) + string("): ERROR: i > _psfs->size()=") + to_string(_psfs->size());
      cout << message << endl;
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    return _psfs->at(i); 
  }

  template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
  bool PSFSet<ImageT, MaskT, VarianceT, WavelengthT>::erase(const size_t iStart, const size_t iEnd){
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
    return true;
  }

  namespace math{

    template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
    PTR(PSFSet<ImageT, MaskT, VarianceT, WavelengthT>) calculate2dPSFPerBin(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace,
                                                                            Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrum,
                                                                            TwoDPSFControl const& twoDPSFControl){
      ndarray::Array<ImageT, 2, 1> collapsedPSF = ndarray::allocate(1,1);
      return calculate2dPSFPerBin(fiberTrace, spectrum, twoDPSFControl, collapsedPSF);
    }
    
    template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
    PTR(PSFSet<ImageT, MaskT, VarianceT, WavelengthT>) calculate2dPSFPerBin(FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace,
                                                                            Spectrum<ImageT, MaskT, VarianceT, WavelengthT> const& spectrum,
                                                                            TwoDPSFControl const& twoDPSFControl,
                                                                            ndarray::Array<ImageT, 2, 1> const& collapsedPSF){
      int swathWidth = twoDPSFControl.swathWidth;
      ndarray::Array<size_t, 2, 1> binBoundY = fiberTrace.calcSwathBoundY(swathWidth);
      if (binBoundY[binBoundY.getShape()[0]-1][1] != fiberTrace.getHeight()-1){
        string message("calculate2dPSFPerBin: FiberTrace");
        message += to_string(fiberTrace.getITrace()) + ": ERROR: binBoundY[binBoundY.getShape()[0]-1=";
        message += to_string(binBoundY.getShape()[0]-1) + "][1] = " + to_string(binBoundY[binBoundY.getShape()[0]-1][1]) + "!= fiberTrace.getHeight()-1 = ";
        message += to_string(fiberTrace.getHeight()-1);
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      #ifdef __DEBUG_CALC2DPSF__
        cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": fiberTrace.getHeight() = " << fiberTrace.getHeight() << ": binBoundY = " << binBoundY << endl;
//        exit(EXIT_FAILURE);
      #endif

      PTR(PSFSet<ImageT, MaskT, VarianceT, WavelengthT>) psfSet(new PSFSet<ImageT, MaskT, VarianceT, WavelengthT>());
      for (int iBin = 0; iBin < binBoundY.getShape()[0]; ++iBin){
        cout << "binBoundY = " << binBoundY << endl;
        /// start calculate2dPSF for bin iBin
        if (binBoundY[iBin][1] >= fiberTrace.getHeight()){
          string message("calculate2dPSFPerBin: FiberTrace");
          message += to_string(fiberTrace.getITrace()) + ": iBin " + to_string(iBin) + ": ERROR: binBoundY[" + to_string(iBin) + "][1]=";
          message += to_string(binBoundY[iBin][1]) + " >= fiberTrace.getHeight()=" + to_string(fiberTrace.getHeight());
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        #ifdef __DEBUG_CALC2DPSF__
          cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": calculating PSF for iBin " << iBin << ": binBoundY[" << iBin << "][0] = " << binBoundY[iBin][0] << ", binBoundY[" << iBin << "][1] = " << binBoundY[iBin][1] << endl;
        #endif
        PTR(TwoDPSFControl) pTwoDPSFControl(new TwoDPSFControl(twoDPSFControl));
        PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>) psf(new PSF< ImageT, MaskT, VarianceT, WavelengthT>((unsigned int)(binBoundY[iBin][0]),
                                                                                                            (unsigned int)(binBoundY[iBin][1]),
                                                                                                            pTwoDPSFControl,
                                                                                                            fiberTrace.getITrace(),
                                                                                                            iBin));
        if (psf->getYHigh() != binBoundY[iBin][1]){
          string message("calculate2dPSFPerBin: FiberTrace");
          message += to_string(fiberTrace.getITrace()) + ": iBin " + to_string(iBin) + ": ERROR: psf->getYHigh(=";
          message += to_string(psf->getYHigh()) + ") != binBoundY[" + to_string(iBin) + "][1]=" + to_string(binBoundY[iBin][1]);
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        #ifdef __DEBUG_CALC2DPSF__
          cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": iBin " << iBin << ": starting extractPSFs()" << endl;
        #endif
        if (!psf->extractPSFs(fiberTrace, 
                              spectrum,
                              collapsedPSF)){
          string message("calculate2dPSFPerBin: FiberTrace");
          message += to_string(fiberTrace.getITrace()) + string(": iBin ") + to_string(iBin) + string(": ERROR: psf->extractPSFs() returned FALSE");
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
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
//        #ifdef __DEBUG_CALC2DPSF__
//          if (iBin == 21)
//            exit(EXIT_FAILURE);
//        #endif
      }

      return psfSet;
    }
    
    template< typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT >
    std::vector< PTR( PSFSet< ImageT, MaskT, VarianceT, WavelengthT > ) > calculate2dPSFPerBin( FiberTraceSet< ImageT, MaskT, VarianceT > const& fiberTraceSet,
                                                                                                SpectrumSet< ImageT, MaskT, VarianceT, WavelengthT > const& spectrumSet,
                                                                                                TwoDPSFControl const& twoDPSFControl ){
      std::vector< PTR(PSFSet<ImageT, MaskT, VarianceT, WavelengthT>)> vecOut(0);
      for (int i = 0; i < fiberTraceSet.size(); ++i){
        PTR(PSFSet<ImageT, MaskT, VarianceT, WavelengthT>) psfSet = calculate2dPSFPerBin(*(fiberTraceSet.getFiberTrace(i)), 
                                                                                         *(spectrumSet.getSpectrum(i)), twoDPSFControl);
        vecOut.push_back(psfSet);
      }
      return vecOut;
    }
    
    template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
    ndarray::Array<ImageT, 2, 1> interpolatePSFThinPlateSpline(PSF<ImageT, MaskT, VarianceT, WavelengthT> & psf,
                                                               ndarray::Array<float, 1, 1> const& xPositions,
                                                               ndarray::Array<float, 1, 1> const& yPositions,
                                                               bool const isXYPositionsGridPoints,
                                                               double const regularization){

      ndarray::Array<float, 1, 1> xArr = ndarray::allocate(psf.getImagePSF_XRelativeToCenter().size());
      if (xArr.size() < 3){
        string message("PSF::InterPolateThinPlateSpline: ERROR: xArr.size(=");
        message += to_string(xArr.size()) + ") < 3";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      ndarray::Array<float, 1, 1> yArr = ndarray::allocate(psf.getImagePSF_YRelativeToCenter().size());
      ndarray::Array<float, 1, 1> zArr = ndarray::allocate(psf.getImagePSF_ZNormalized().size());
      for (int i = 0; i < xArr.getShape()[0]; ++i){
        xArr[i] = psf.getImagePSF_XRelativeToCenter()[i];
        yArr[i] = psf.getImagePSF_YRelativeToCenter()[i];
        zArr[i] = psf.getImagePSF_ZNormalized()[i];
      }
      #ifdef __DEBUG_CALC_TPS__
        cout << "PSF::interpolatePSFThinPlateSpline: xArr = " << xArr << endl;
        cout << "PSF::interpolatePSFThinPlateSpline: yArr = " << yArr << endl;
        cout << "PSF::interpolatePSFThinPlateSpline: psf.getImagePSF_ZNormalized() = " << psf.getImagePSF_ZNormalized() << endl;
        cout << "PSF::interpolatePSFThinPlateSpline: zArr = " << zArr << endl;
      #endif
      ndarray::Array<ImageT, 1, 1> zT = ndarray::allocate(zArr.size());
      auto it = zArr.begin();
      for (auto itT = zT.begin(); itT != zT.end(); ++itT, ++it)
        *itT = ImageT(*it);
      #ifdef __DEBUG_CALC_TPS__
        cout << "PSF::interpolatePSFThinPlateSpline: zT = " << zT << endl;
      #endif
      cout << "PSF::interpolatePSFThinPlateSpline: starting interpolateThinPlateSplineEigen" << endl;
      math::ThinPlateSpline<float, ImageT> tps = math::ThinPlateSpline<float, ImageT>( xArr, 
                                                                                       yArr, 
                                                                                       ndarray::Array<ImageT const, 1, 1>(zT), 
                                                                                       regularization );
      ndarray::Array<ImageT, 2, 1> arr_Out = ndarray::copy(tps.fitArray(xPositions, 
                                                                        yPositions, 
                                                                        isXYPositionsGridPoints));
      ndarray::Array< ImageT, 2, 1 > zRec = ndarray::copy(tps.fitArray(xArr,
                                                          yArr,
                                                          false));
      zT.deep() = zRec[ndarray::view()(0)];
      if (!psf.setImagePSF_ZFit(zT)){
        cout << "PSF::interpolatePSFThinPlateSpline: WARNING: psf.setImagePSF_ZFit(zT) returned FALSE" << endl;
      }
      return arr_Out;
    }
    
    template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
    ndarray::Array<ImageT, 2, 1> interpolatePSFThinPlateSpline(PSF<ImageT, MaskT, VarianceT, WavelengthT> & psf,
                                                               ndarray::Array<ImageT, 1, 1> const& weights,
                                                               ndarray::Array<float, 1, 1> const& xPositions,
                                                               ndarray::Array<float, 1, 1> const& yPositions,
                                                               bool const isXYPositionsGridPoints){

      ndarray::Array<float, 1, 1> xArr = ndarray::allocate(psf.getImagePSF_XRelativeToCenter().size());
      if (xArr.getShape()[0] < 3){
        string message("PSF::InterPolateThinPlateSpline: ERROR: xArr.getShape()[0](=");
        message += to_string(xArr.getShape()[0]) + ") < 3";
        cout << message << endl;
        throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
      }
      ndarray::Array<float, 1, 1> yArr = ndarray::allocate(psf.getImagePSF_YRelativeToCenter().size());
      ndarray::Array<float, 1, 1> zArr = ndarray::allocate(psf.getImagePSF_ZNormalized().size());
      for (int i = 0; i < xArr.getShape()[0]; ++i){
        xArr[i] = psf.getImagePSF_XRelativeToCenter()[i];
        yArr[i] = psf.getImagePSF_YRelativeToCenter()[i];
        zArr[i] = psf.getImagePSF_ZNormalized()[i];
      }
      #ifdef __DEBUG_CALC_TPS__
        cout << "PSF::interpolatePSFThinPlateSpline: psf.getImagePSF_ZNormalized() = " << psf.getImagePSF_ZNormalized() << endl;
        cout << "PSF::interpolatePSFThinPlateSpline: xArr = " << xArr << endl;
        cout << "PSF::interpolatePSFThinPlateSpline: yArr = " << yArr << endl;
        cout << "PSF::interpolatePSFThinPlateSpline: zArr = " << zArr << endl;
      #endif
      ndarray::Array<ImageT, 1, 1> zT = ndarray::allocate(zArr.size());
      auto it = zArr.begin();
      for (auto itT = zT.begin(); itT != zT.end(); ++itT, ++it)
        *itT = ImageT(*it);
      #ifdef __DEBUG_CALC_TPS__
        cout << "PSF::interpolatePSFThinPlateSpline: zT = " << zT << endl;
      #endif
      cout << "PSF::interpolatePSFThinPlateSpline: starting interpolateThinPlateSplineEigen" << endl;
      math::ThinPlateSpline<float, ImageT> tps = math::ThinPlateSpline<float, ImageT>( xArr, 
                                                                                       yArr, 
                                                                                       zT, 
                                                                                       weights);
      ndarray::Array< ImageT, 2, 1 > zFit = ndarray::copy(tps.fitArray(xPositions, 
                                                                       yPositions, 
                                                                       isXYPositionsGridPoints));
      ndarray::Array< ImageT, 2, 1 > zRec = ndarray::copy(tps.fitArray(xArr,
                                                          yArr,
                                                          false));
      zT.deep() = zRec[ndarray::view()(0)];
      if (!psf.setImagePSF_ZFit(zT)){
        cout << "PSF::interpolatePSFThinPlateSpline: WARNING: psf.setImagePSF_ZFit(zT) returned FALSE" << endl;
      }
      return zFit;
    }
    
    template < typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT >
    ndarray::Array< ImageT, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< ImageT, MaskT, VarianceT, WavelengthT > & psfSet,
                                                                    ndarray::Array< float, 1, 1 > const& xPositions,
                                                                    ndarray::Array< float, 1, 1 > const& yPositions,
                                                                    bool const isXYPositionsGridPoints,
                                                                    double const regularization){
      ndarray::Array<ImageT, 3, 1> arrOut = ndarray::allocate(yPositions.getShape()[0], xPositions.getShape()[0], psfSet.size());
      for (int i = 0; i < psfSet.size(); ++i){
        #ifdef __DEBUG_CALC_TPS__
          if (psfSet.getPSF(i)->getImagePSF_XRelativeToCenter().size() < 3){
            string message("PSF::InterPolatePSFSetThinPlateSpline: ERROR: i=");
            message += to_string(i) + ": imagePSF_XRelativeToCenter().size()=" + to_string(psfSet.getPSF(i)->getImagePSF_XRelativeToCenter().size()) + " < 3";
            cout << message << endl;
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
          }
        #endif
        ndarray::Array<ImageT, 2, 1> arr = ndarray::allocate(yPositions.getShape()[0], xPositions.getShape()[0]);
        arr.deep() = interpolatePSFThinPlateSpline(*(psfSet.getPSF(i)), xPositions, yPositions, isXYPositionsGridPoints, regularization);
        #ifdef __DEBUG_CALC_TPS__
          cout << "interpolatePSFSetThinPlateSpline: arr.getShape() = " << arr.getShape() << ", arrOut.getShape() = " << arrOut.getShape() << endl;
          cout << "interpolatePSFSetThinPlateSpline: arr = " << arr << endl;
        #endif
        arrOut[ndarray::view()()(i)].deep() = arr;
//        PTR(ndarray::Array<ImageT, 2, 1>) pArr(new ndarray::Array<ImageT, 2, 1>(arr));
//        vecOut.push_back(arr);
      }
      #ifdef __DEBUG_CALC_TPS__
        cout << "interpolatePSFSetThinPlateSpline: arrOut[:][:][0] = " << arrOut[ndarray::view()()(0)] << endl;
      #endif
      return arrOut;
    }
    
    template < typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT >
    ndarray::Array< ImageT, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< ImageT, MaskT, VarianceT, WavelengthT > & psfSet,
                                                                    ndarray::Array< ImageT, 2, 1 > const& weightArr,
                                                                    ndarray::Array< float, 1, 1 > const& xPositions,
                                                                    ndarray::Array< float, 1, 1 > const& yPositions,
                                                                    bool const isXYPositionsGridPoints){
      ndarray::Array<ImageT, 3, 1> arrOut = ndarray::allocate(yPositions.getShape()[0], xPositions.getShape()[0], psfSet.size());
      for (int i = 0; i < psfSet.size(); ++i){
        #ifdef __DEBUG_CALC_TPS__
        if (psfSet.getPSF(i)->getImagePSF_XRelativeToCenter().size() < 3){
          string message("PSF::InterPolatePSFSetThinPlateSpline: ERROR: i=");
          message += to_string(i) + ": imagePSF_XRelativeToCenter().size()=" + to_string(psfSet.getPSF(i)->getImagePSF_XRelativeToCenter().size()) + " < 3";
          cout << message << endl;
          throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        #endif
        size_t size = psfSet.getPSF(i)->getImagePSF_ZNormalized().size();
        ndarray::Array<ImageT, 1, 1> weights = ndarray::allocate(size);
        weights.deep() = weightArr[ndarray::view(0, size)(i)];
        ndarray::Array<ImageT, 2, 1> arr = ndarray::copy(interpolatePSFThinPlateSpline(*(psfSet.getPSF(i)), 
                                                                                       weights,
                                                                                       xPositions, 
                                                                                       yPositions,
                                                                                       isXYPositionsGridPoints));
        #ifdef __DEBUG_CALC_TPS__
          cout << "interpolatePSFSetThinPlateSpline: arr.getShape() = " << arr.getShape() << ", arrOut.getShape() = " << arrOut.getShape() << endl;
          cout << "interpolatePSFSetThinPlateSpline: arr = " << arr << endl;
        #endif
        arrOut[ndarray::view()()(i)].deep() = arr;
//        PTR(ndarray::Array<ImageT, 2, 1>) pArr(new ndarray::Array<ImageT, 2, 1>(arr));
//        vecOut.push_back(arr);
      }
      #ifdef __DEBUG_CALC_TPS__
        cout << "interpolatePSFSetThinPlateSpline: arrOut[:][:][0] = " << arrOut[ndarray::view()()(0)] << endl;
      #endif
      return arrOut;
    }
    
    template< typename ImageT, typename CoordT >
    ndarray::Array< ImageT, 2, 1 > collapseFittedPSF( ndarray::Array< CoordT, 1, 1 > const& xGridVec_In,
                                                      ndarray::Array< CoordT, 1, 1 > const& yGridVec_In,
                                                      ndarray::Array< ImageT, 2, 1 > const& zArr_In,
                                                      int const direction){
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
      ndarray::Array< ImageT, 1, 1 > tempVec;
      ndarray::Array< ImageT, 2, 1 > collapsedPSF;
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
      for (int iPos = 0; iPos < collapsedPSFLength; ++iPos){
        if (direction == 0){
          tempVec.deep() = zArr_In[ndarray::view(iPos)()];
        }
        else{
          tempVec.deep() = zArr_In[ndarray::view()(iPos)];
        }
        collapsedPSF[iPos][1] = tempVec.asEigen().array().sum();
      }
      return collapsedPSF;
    }
    
    template< typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT >
    ndarray::Array< float, 2, 1 > collapsePSF( PSF< ImageT, MaskT, VarianceT, WavelengthT > const& psf_In,
                                               ndarray::Array< float, 1, 1 > const& coordinatesX_In,
                                               ndarray::Array< float, 1, 1 > const& coordinatesY_In,
                                               int const direction,
                                               double const regularization){
      ndarray::Array< const float, 1, 1 > xRelativeToCenter = ndarray::external(psf_In.getImagePSF_XRelativeToCenter().data(), ndarray::makeVector(int(psf_In.getImagePSF_XRelativeToCenter().size())), ndarray::makeVector(1));
      ndarray::Array< const float, 1, 1 > yRelativeToCenter = ndarray::external(psf_In.getImagePSF_YRelativeToCenter().data(), ndarray::makeVector(int(psf_In.getImagePSF_YRelativeToCenter().size())), ndarray::makeVector(1));
      ndarray::Array< const float, 1, 1 > zNormalized = ndarray::external(psf_In.getImagePSF_ZNormalized().data(), ndarray::makeVector(int(psf_In.getImagePSF_ZNormalized().size())), ndarray::makeVector(1));
      math::ThinPlateSpline<float, float> tpsA = math::ThinPlateSpline<float, float>( xRelativeToCenter,
                                                                                       yRelativeToCenter,
                                                                                       zNormalized,
                                                                                       regularization );
      ndarray::Array< float, 2, 1 > interpolatedPSF = tpsA.fitArray( coordinatesX_In,
                                                                     coordinatesY_In,
                                                                     true);
      return math::collapseFittedPSF(coordinatesX_In,
                                     coordinatesY_In,
                                     interpolatedPSF,
                                     direction);
    }
    
    template< typename T >
    ndarray::Array< T, 2, 1> calcPositionsRelativeToCenter(T const centerPos_In,
                                                           T const width_In){
      #ifdef __DEBUG_CPRTC__
        cout << "PSF::math::calcPositionsRelativeToCenter: centerPos_In = " << centerPos_In << ", width_In = " << width_In << endl;
      #endif
      size_t low = std::floor(centerPos_In - (width_In / 2.));
      size_t high = std::floor(centerPos_In + (width_In / 2.));
      #ifdef __DEBUG_CPRTC__
        cout << "PSF::math::calcPositionsRelativeToCenter: low = " << low << ", high = " << high << endl;
      #endif
      ndarray::Array< T, 2, 1 > arr_Out = ndarray::allocate(high - low + 1, 2);
      double dCenter = 0.5 - centerPos_In + std::floor(centerPos_In);
      #ifdef __DEBUG_CPRTC__
        cout << "PSF::math::calcPositionsRelativeToCenter: dCenter = " << dCenter << endl;
      #endif
      for (int iPos = 0; iPos <= high - low; ++iPos){
        arr_Out[iPos][0] = T(low + iPos);
        arr_Out[iPos][1] = arr_Out[iPos][0] - T(std::floor(centerPos_In)) + dCenter;
        #ifdef __DEBUG_CPRTC__
          cout << "PSF::math::calcPositionsRelativeToCenter: arr_Out[" << iPos << "][0] = " << arr_Out[iPos][0] << ", arr_Out[" << iPos << "][1] = " << arr_Out[iPos][1] << endl;
        #endif
      }
      #ifdef __DEBUG_CPRTC__
        cout << "PSF::math::calcPositionsRelativeToCenter: arr_Out = " << arr_Out << endl;
      #endif
      return arr_Out;
    }
    
    template ndarray::Array< float, 2, 1> calcPositionsRelativeToCenter(float const, float const);
    template ndarray::Array< double, 2, 1> calcPositionsRelativeToCenter(double const, double const);

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

    template ndarray::Array< float, 2, 1 > collapsePSF( PSF< float, unsigned short, float, float > const&,
                                                        ndarray::Array< float, 1, 1 > const&,
                                                        ndarray::Array< float, 1, 1 > const&,
                                                        int const,
                                                        double const);
    template ndarray::Array< float, 2, 1 > collapsePSF( PSF< double, unsigned short, float, float > const&,
                                                        ndarray::Array< float, 1, 1 > const&,
                                                        ndarray::Array< float, 1, 1 > const&,
                                                        int const,
                                                        double const);

    template ndarray::Array<float, 2, 1> interpolatePSFThinPlateSpline(PSF<float, unsigned short, float, float> &, 
                                                                       ndarray::Array<float, 1, 1> const&, 
                                                                       ndarray::Array<float, 1, 1> const&,
                                                                       bool const,
                                                                       double const);
    template ndarray::Array<double, 2, 1> interpolatePSFThinPlateSpline(PSF<double, unsigned short, float, float> &, 
                                                                        ndarray::Array<float, 1, 1> const&, 
                                                                        ndarray::Array<float, 1, 1> const&,
                                                                        bool const,
                                                                        double const);

    template ndarray::Array<float, 2, 1> interpolatePSFThinPlateSpline(PSF<float, unsigned short, float, float> &, 
                                                                       ndarray::Array<float, 1, 1> const&, 
                                                                       ndarray::Array<float, 1, 1> const&, 
                                                                       ndarray::Array<float, 1, 1> const&,
                                                                       bool const);
    template ndarray::Array<double, 2, 1> interpolatePSFThinPlateSpline(PSF<double, unsigned short, float, float> &, 
                                                                        ndarray::Array<double, 1, 1> const&, 
                                                                        ndarray::Array<float, 1, 1> const&, 
                                                                        ndarray::Array<float, 1, 1> const&,
                                                                        bool const);
    
    template ndarray::Array< float, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< float, unsigned short, float, float > &, 
                                                                            ndarray::Array< float, 1, 1 > const&, 
                                                                            ndarray::Array< float, 1, 1 > const&,
                                                                            bool const,
                                                                            double const);
    template ndarray::Array< double, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< double, unsigned short, float, float > &, 
                                                                             ndarray::Array< float, 1, 1 > const&, 
                                                                             ndarray::Array< float, 1, 1 > const&,
                                                                             bool const,
                                                                             double const);
    
    template ndarray::Array< float, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< float, unsigned short, float, float > &, 
                                                                            ndarray::Array< float, 2, 1 > const&, 
                                                                            ndarray::Array< float, 1, 1 > const&, 
                                                                            ndarray::Array< float, 1, 1 > const&,
                                                                            bool const);
    template ndarray::Array< double, 3, 1 > interpolatePSFSetThinPlateSpline(PSFSet< double, unsigned short, float, float > &, 
                                                                             ndarray::Array< double, 2, 1 > const&, 
                                                                             ndarray::Array< float, 1, 1 > const&, 
                                                                             ndarray::Array< float, 1, 1 > const&,
                                                                             bool const);
    
    template PTR(PSFSet<float, unsigned short, float, float>) calculate2dPSFPerBin(FiberTrace<float, unsigned short, float> const&, 
                                                                                   Spectrum<float, unsigned short, float, float> const&,
                                                                                   TwoDPSFControl const&);
    template PTR(PSFSet<double, unsigned short, float, float>) calculate2dPSFPerBin(FiberTrace<double, unsigned short, float> const&, 
                                                                                    Spectrum<double, unsigned short, float, float> const&,
                                                                                    TwoDPSFControl const&);
    
    template PTR(PSFSet<float, unsigned short, float, float>) calculate2dPSFPerBin(FiberTrace<float, unsigned short, float> const&, 
                                                                                   Spectrum<float, unsigned short, float, float> const&,
                                                                                   TwoDPSFControl const&,
                                                                                   ndarray::Array<float, 2, 1> const&);
    template PTR(PSFSet<double, unsigned short, float, float>) calculate2dPSFPerBin(FiberTrace<double, unsigned short, float> const&, 
                                                                                    Spectrum<double, unsigned short, float, float> const&,
                                                                                    TwoDPSFControl const&,
                                                                                    ndarray::Array<double, 2, 1> const&);
    
    template std::vector<PTR(PSFSet<float, unsigned short, float, float>)> calculate2dPSFPerBin(FiberTraceSet<float, unsigned short, float> const&, 
                                                                                                SpectrumSet<float, unsigned short, float, float> const&,
                                                                                                TwoDPSFControl const&);
    template std::vector<PTR(PSFSet<double, unsigned short, float, float>)> calculate2dPSFPerBin(FiberTraceSet<double, unsigned short, float> const&, 
                                                                                                 SpectrumSet<double, unsigned short, float, float> const&,
                                                                                                 TwoDPSFControl const&);
  }

  template class PSF<float, unsigned short, float, float>;
  template class PSF<double, unsigned short, float, float>;

  template class PSFSet<float, unsigned short, float, float>;
  template class PSFSet<double, unsigned short, float, float>;

}}}
