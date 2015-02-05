#include "pfs/drp/stella/PSF.h"

namespace drpStella = pfs::drp::stella;

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
drpStella::PSF<ImageT, MaskT, VarianceT, WavelengthT>::PSF(PSF &psf, const bool deep)
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
    PTR(drpStella::TwoDPSFControl) ptr(new drpStella::TwoDPSFControl(*(psf.getTwoDPSFControl())));
    _twoDPSFControl.reset();
    _twoDPSFControl = ptr;
  }
//  else{
//    _twoDPSFControl(psf.getTwoDPSFControl());
//  }
}

/// Set the _twoDPSFControl
template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
bool drpStella::PSF<ImageT, MaskT, VarianceT, WavelengthT>::setTwoDPSFControl(PTR(TwoDPSFControl) &twoDPSFControl){
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
bool drpStella::PSF<ImageT, MaskT, VarianceT, WavelengthT>::extractPSFs(const FiberTrace<ImageT, MaskT, VarianceT> &fiberTraceIn,
                                                                        const Spectrum<ImageT, MaskT, VarianceT, WavelengthT> &spectrumIn)
{
  if (!_isTwoDPSFControlSet){
    string message("PSF trace");
    message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: ERROR: _twoDPSFControl is not set";
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  
  ndarray::Array<float, 1, 1> xCenters = copy(fiberTraceIn.getXCenters());

  blitz::Array<double, 2> trace_In(_yMax - _yMin + 1, fiberTraceIn.getImage()->getWidth());
  blitz::Array<double, 2> mask_In(_yMax - _yMin + 1, fiberTraceIn.getImage()->getWidth());
  blitz::Array<double, 2> stddev_In(_yMax - _yMin + 1, fiberTraceIn.getImage()->getWidth());
  blitz::Array<double, 1> spectrum_In(_yMax - _yMin + 1);
  blitz::Array<double, 1> spectrumVariance_In(_yMax - _yMin + 1);

  blitz::Array<ImageT, 2> T_A2_PixArray = utils::ndarrayToBlitz(fiberTraceIn.getImage()->getArray());
  blitz::Array<double, 2> D_A2_PixArray = math::Double(T_A2_PixArray);
  trace_In = D_A2_PixArray(blitz::Range(_yMin, _yMax), blitz::Range::all());

  blitz::Array<VarianceT, 2> T_A2_VarArray = utils::ndarrayToBlitz(fiberTraceIn.getVariance()->getArray());
  blitz::Array<double, 2> D_A2_StdDevArray = math::Double(T_A2_VarArray);
  stddev_In = blitz::sqrt(blitz::where(D_A2_StdDevArray(blitz::Range(_yMin, _yMax), blitz::Range::all()) > 0., D_A2_StdDevArray(blitz::Range(_yMin, _yMax), blitz::Range::all()), 1.));
  if (fabs(blitz::max(stddev_In)) < 0.000001){
    double D_MaxStdDev = fabs(blitz::max(stddev_In));
    string message("PSF trace");
    message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: ERROR: fabs(blitz::max(stddev_In))=" + to_string(D_MaxStdDev) + " < 0.000001";
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }

  blitz::Array<MaskT, 2> T_A2_MaskArray = utils::ndarrayToBlitz(fiberTraceIn.getMask()->getArray());
//    blitz::Array<int, 2> I_A2_MaskArray(D_A2_PixArray.rows(), D_A2_PixArray.cols());
  mask_In = blitz::where(T_A2_MaskArray(blitz::Range(_yMin, _yMax), blitz::Range::all()) == 0, 1, 0);

  blitz::Array<ImageT, 1> F_A1_Spectrum(const_cast<ImageT*>(spectrumIn.getSpectrum()->data()), blitz::shape(spectrumIn.getLength()), blitz::neverDeleteData);
  spectrum_In = math::Double(F_A1_Spectrum(blitz::Range(_yMin, _yMax)));

  blitz::Array<VarianceT, 1> F_A1_SpectrumVariance(const_cast<VarianceT*>(spectrumIn.getVariance()->data()), blitz::shape(spectrumIn.getLength()), blitz::neverDeleteData);
  spectrumVariance_In = math::Double(F_A1_SpectrumVariance(blitz::Range(_yMin, _yMax)));
  ///TODO: replace with variance from MaskedImage
  spectrumVariance_In = spectrum_In;
  blitz::Array<double, 1> spectrumSigma(spectrumVariance_In.size());
  spectrumSigma = blitz::sqrt(spectrumVariance_In);

  ndarray::Array<float, 1, 1> xCentersSwathF = ndarray::copy(xCenters[ndarray::view(_yMin, _yMax + 1)]);
  ndarray::Array<const float, 1, 1> xCentersSwathFConst = ndarray::copy(xCenters[ndarray::view(_yMin, _yMax + 1)]);
  ndarray::Array<float, 1, 1> xCentersFloor = math::floor(xCentersSwathFConst, float(0));
  ndarray::Array<float, 1, 1> xCentersOffset_In = copy(xCentersSwathF);
  xCentersOffset_In.deep() -= xCentersFloor;

  ndarray::Array<size_t, 2, 2> minCenMax = drpStella::math::calcMinCenMax(xCentersSwathF,
                                                                          fiberTraceIn.getFiberTraceFunction()->fiberTraceFunctionControl.xHigh,
                                                                          fiberTraceIn.getFiberTraceFunction()->fiberTraceFunctionControl.xLow,
                                                                          1,
                                                                          1);
  xCentersSwathF[ndarray::view()] -= minCenMax[ndarray::view()(0)];
  #ifdef __DEBUG_CALC2DPSF__
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: minCenMax = " << minCenMax << endl;
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: xCentersSwathF = " << xCentersSwathF << endl;
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
  double dTrace, dTraceGaussCenterX, pixOffsetY, pixPosX, pixPosY, xStart, yStart;
  int i_xCenter, i_yCenter, i_Left, i_Right, i_Down, i_Up;
  int nPix = 0;

  ///FIND EMISSION LINES
  /// create center column
//    blitz::Array<double, 1> traceCenterCol(_spectrum->size(), blitz::shape(_spectrum->size()), blitz::neverDeleteData);//(trace_In.rows());
//    blitz::Array<double, 1> traceCenterStdDevCol(_spectrumVariance->data(), blitz::shape(_spectrumVariance->size()), blitz::neverDeleteData);//(trace_In.rows());
//    for (int iRow=0; iRow<trace_In.rows(); ++iRow){
//      traceCenterCol(iRow) = trace_In(iRow, int(xCentersSwathF[iRow])) * mask_In(iRow, int(xCentersSwathF[iRow]));
//      traceCenterStdDevCol(iRow) = stddev_In(iRow, int(xCentersSwathF[iRow])) * mask_In(iRow, int(xCentersSwathF[iRow]));
//    }
  #ifdef __DEBUG_CALC2DPSF__
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: spectrum_In = " << spectrum_In << endl;
    std::string colname = __DEBUGDIR__ + std::string("traceSpectrum_iBin");
    if (_iBin < 10)
      colname += std::string("0");
    colname += to_string(_iBin)+std::string(".fits");
    utils::WriteFits(&spectrum_In, colname);
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: spectrumSigma = " << spectrumSigma << endl;
    std::string fname_trace = __DEBUGDIR__ + std::string("trace_iBin");
    if (_iBin < 10)
      fname_trace += std::string("0");
    fname_trace += to_string(_iBin)+std::string(".fits");
    utils::WriteFits(&trace_In, fname_trace);
  #endif

  /// set everything below signal threshold to zero
  blitz::Array<int, 1> traceCenterSignal(spectrum_In.size());
  traceCenterSignal = blitz::where(spectrum_In < _twoDPSFControl->signalThreshold, 0., spectrum_In);
  #ifdef __DEBUG_CALC2DPSF__
    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: 1. traceCenterSignal = " << traceCenterSignal << endl;
  #endif

  /// look for signal wider than 2 FWHM
  if (!math::CountPixGTZero(traceCenterSignal)){
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
  blitz::Array<int, 2> I_A2_Limited(_twoDPSFControl->nTermsGaussFit, 2);
  I_A2_Limited = 1;
  if (_twoDPSFControl->nTermsGaussFit == 5)
    I_A2_Limited(4,blitz::Range::all()) = 0;
  blitz::Array<double, 2> D_A2_Limits(_twoDPSFControl->nTermsGaussFit,2);
  /// 0: peak value
  /// 1: center position
  /// 2: sigma
  /// 3: constant background
  /// 4: linear background
  D_A2_Limits(0,0) = _twoDPSFControl->signalThreshold;
  D_A2_Limits(2,0) = _twoDPSFControl->yFWHM / (2. * 2.3548);
  D_A2_Limits(2,1) = 2. * _twoDPSFControl->yFWHM;
  if (_twoDPSFControl->nTermsGaussFit > 3)
    D_A2_Limits(3,0) = 0.;
  if (_twoDPSFControl->nTermsGaussFit > 4){
    D_A2_Limits(4,0) = 0.;
    D_A2_Limits(4,1) = 1000.;
  }
  blitz::Array<double, 1> D_A1_GaussFit_Coeffs(_twoDPSFControl->nTermsGaussFit);
  blitz::Array<double, 1> D_A1_GaussFit_ECoeffs(_twoDPSFControl->nTermsGaussFit);

  blitz::firstIndex i;

  blitz::Array<double, 1> D_A1_Guess(_twoDPSFControl->nTermsGaussFit);
  blitz::Array<double, 2> D_A2_GaussCenters(1,2);
  int emissionLineNumber = 0;
  do{
    I_FirstWideSignal = math::FirstIndexWithValueGEFrom(traceCenterSignal, I_MinWidth, I_StartIndex);
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: I_FirstWideSignal found at index " << I_FirstWideSignal << ", I_StartIndex = " << I_StartIndex << endl;
    #endif
    if (I_FirstWideSignal < 0){
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: No emission line found" << endl;
      break;
    }
    I_FirstWideSignalStart = math::LastIndexWithZeroValueBefore(traceCenterSignal, I_FirstWideSignal);
    if (I_FirstWideSignalStart < 0){
      I_FirstWideSignalStart = 0;
    }
    #ifdef __DEBUG_CALC2DPSF__
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: I_FirstWideSignalStart = " << I_FirstWideSignalStart << endl;
    #endif

    I_FirstWideSignalEnd = math::FirstIndexWithZeroValueFrom(traceCenterSignal, I_FirstWideSignal);
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

    if (blitz::max(spectrum_In(blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd))) < _twoDPSFControl->saturationLevel){
      D_A2_Limits(0,1) = 1.5 * blitz::max(spectrum_In(blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)));
      D_A2_Limits(1,0) = I_FirstWideSignalStart;
      D_A2_Limits(1,1) = I_FirstWideSignalEnd;
      if (_twoDPSFControl->nTermsGaussFit > 3)
        D_A2_Limits(3,1) = blitz::min(spectrum_In(blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)));

      D_A1_Guess(0) = blitz::max(spectrum_In(blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)));
      D_A1_Guess(1) = I_FirstWideSignalStart + (blitz::maxIndex(spectrum_In(blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd))))(0);
//        if (D_A1_Guess(1) > (1.5 * _twoDPSFControl->yFWHM)){
        D_A1_Guess(2) = _twoDPSFControl->yFWHM / 2.3548;
        if (_twoDPSFControl->nTermsGaussFit > 3)
          D_A1_Guess(3) = blitz::min(spectrum_In(blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)));
        if (_twoDPSFControl->nTermsGaussFit > 4)
          D_A1_Guess(4) = (spectrum_In(I_FirstWideSignalEnd) - spectrum_In(I_FirstWideSignalStart)) / (I_FirstWideSignalEnd - I_FirstWideSignalStart);
        blitz::Array<double, 1> D_A1_X(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
        D_A1_X = i;
        D_A1_X += I_FirstWideSignalStart;
        blitz::Array<double, 1> D_A1_Y(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
        D_A1_Y = blitz::where(fabs(spectrum_In(blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd))) < 0.01, 0.01, spectrum_In(blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)));
        blitz::Array<double, 1> D_A1_StdDev(I_FirstWideSignalEnd - I_FirstWideSignalStart + 1);
        D_A1_StdDev = blitz::where(fabs(spectrumSigma(blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd))) < 0.1, 0.1, spectrumSigma(blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd)));
        #ifdef __DEBUG_CALC2DPSF__
          cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: emissionLineNumber = " << emissionLineNumber << ": Starting GaussFit: D_A1_X=" << D_A1_X << ", D_A1_Y = " << D_A1_Y << ", D_A1_StdDev = " << D_A1_StdDev << ", D_A1_Guess = " << D_A1_Guess << ", I_A2_Limited = " << I_A2_Limited << ", D_A2_Limits = " << D_A2_Limits << endl;
        #endif
        int iBackground = 0;
        if (_twoDPSFControl->nTermsGaussFit == 4)
          iBackground = 1;
        else if (_twoDPSFControl->nTermsGaussFit == 5)
          iBackground = 2;
        if (!MPFitGaussLim(D_A1_X,
                          D_A1_Y,
                          D_A1_StdDev,
                          D_A1_Guess,
                          I_A2_Limited,
                          D_A2_Limits,
                          iBackground,
                          false,
                          D_A1_GaussFit_Coeffs,
                          D_A1_GaussFit_ECoeffs)){
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: WARNING: GaussFit FAILED" << endl;
          #endif
        }
        else{
          #ifdef __DEBUG_CALC2DPSF__
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: D_A1_GaussFit_Coeffs = " << D_A1_GaussFit_Coeffs << endl;
          #endif
          if ((D_A1_GaussFit_Coeffs(2) < (_twoDPSFControl->yFWHM / 1.5)) &&
              ((_twoDPSFControl->nTermsGaussFit < 5) ||
              ((_twoDPSFControl->nTermsGaussFit > 4) && (D_A1_GaussFit_Coeffs(4) < 1000.)))){
            ++emissionLineNumber;
            if (D_A2_GaussCenters.rows() < emissionLineNumber)
              D_A2_GaussCenters.resizeAndPreserve(emissionLineNumber, 2);
            D_A2_GaussCenters(emissionLineNumber-1,1) = D_A1_GaussFit_Coeffs(1)+0.5;
            D_A2_GaussCenters(emissionLineNumber-1,0) = xCentersSwathF[int(D_A1_GaussFit_Coeffs(1))] + ((D_A1_GaussFit_Coeffs(1) -  int(D_A1_GaussFit_Coeffs(1))) * (xCentersSwathF[int(D_A1_GaussFit_Coeffs(1))+1] - xCentersSwathF[int(D_A1_GaussFit_Coeffs(1))]));
            if (int(D_A2_GaussCenters(emissionLineNumber-1,0)) > int(xCentersSwathF[int(D_A1_GaussFit_Coeffs(1))])){
              D_A2_GaussCenters(emissionLineNumber-1,0) = int(D_A2_GaussCenters(emissionLineNumber-1,0)) - 0.0000001;
            }
            if (int(D_A2_GaussCenters(emissionLineNumber-1,0)) < int(xCentersSwathF[int(D_A1_GaussFit_Coeffs(1))])){
              D_A2_GaussCenters(emissionLineNumber-1,0) = int(D_A2_GaussCenters(emissionLineNumber-1,0)) + 1.0000001;
            }
            #ifdef __DEBUG_CALC2DPSF__
              cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: xCentersSwathF[int(D_A1_GaussFit_Coeffs(1))=" << int(D_A1_GaussFit_Coeffs(1)) << "] = " << xCentersSwathF[int(D_A1_GaussFit_Coeffs(1))] << endl;
              cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: xCentersSwathF[int(D_A1_GaussFit_Coeffs(1))+1=" << int(D_A1_GaussFit_Coeffs(1))+1 << "] = " << xCentersSwathF[int(D_A1_GaussFit_Coeffs(1))+1] << endl;
              cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: D_A1_GaussFit_Coeffs(1) - int(D_A1_GaussFit_Coeffs(1)) = " << D_A1_GaussFit_Coeffs(1) - int(D_A1_GaussFit_Coeffs(1)) << endl;
              cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: D_A2_GaussCenters(emissionLineNumber=" << emissionLineNumber << "-1,*) = " << D_A2_GaussCenters(emissionLineNumber-1, blitz::Range::all()) << endl;
            #endif

            /// add emission line to PSFs

            /// difference between xCentersOffset_In(iY) and 2D GaussCenter of emissionLine
            dTraceGaussCenterX = xCentersSwathF[int(D_A2_GaussCenters(emissionLineNumber-1, 1))] - D_A2_GaussCenters(emissionLineNumber-1, 0);

            i_yCenter = int(D_A2_GaussCenters(emissionLineNumber-1, 1));
            i_xCenter = int(D_A2_GaussCenters(emissionLineNumber-1, 0));

            i_Down = int(D_A2_GaussCenters(emissionLineNumber-1, 1) - (2.*_twoDPSFControl->yFWHM));
            if (i_Down >= 0){
//                i_Down = 0;
              i_Up = int(D_A2_GaussCenters(emissionLineNumber-1, 1) + (2.*_twoDPSFControl->yFWHM));
              #ifdef __DEBUG_CALC2DPSF__
                cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": fiberTraceIn.rows() = " << trace_In.rows() << ", i_Up = " << i_Up << endl;
              #endif
              if (i_Up < trace_In.rows()){
//                i_Up = trace_In.rows() - 1;
                pixOffsetY = D_A2_GaussCenters(emissionLineNumber-1,1) - int(D_A2_GaussCenters(emissionLineNumber-1,1)) - 0.5;
                #ifdef __DEBUG_CALC2DPSF__
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": D_A2_GaussCenters(" << emissionLineNumber-1 << ",0) = " << D_A2_GaussCenters(emissionLineNumber-1,0) << ": D_A2_GaussCenters(" << emissionLineNumber-1 << ",1) = " << D_A2_GaussCenters(emissionLineNumber-1,1) << ":  i_yCenter = " << i_yCenter << ", dTraceGaussCenterX = " << dTraceGaussCenterX << endl;
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": i_Down = " << i_Down << ", i_Up = " << i_Up << endl;
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ":pixOffsetY = " << pixOffsetY << endl;
                #endif

                int nPixPSF = 0;
                double sumPSF = 0.;
                for (int iY = 0; iY <= i_Up - i_Down; ++iY){
                  /// difference in trace center between rows iY and iY(GaussCenterY)
                  #ifdef __DEBUG_CALC2DPSF__
                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": D_A2_GaussCenters.rows() = " << D_A2_GaussCenters.rows() << "xCentersOffset_In.size() = " << xCentersOffset_In.size() << endl;
                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": D_A2_GaussCenters(emissionLineNumber-1, 1) = " << D_A2_GaussCenters(emissionLineNumber-1, 1) << ", xCentersOffset_In(i_Down + iY=" << i_Down + iY << ") = " << xCentersOffset_In(i_Down + iY) << endl;
                  #endif
                  dTrace = xCentersOffset_In(int(D_A2_GaussCenters(emissionLineNumber-1, 1))) - xCentersOffset_In(i_Down + iY);

                  /// most left pixel affected by PSF of (emissionLineNumber-1) in this row
                  i_Left = int(xCentersSwathF[i_Down + iY] - dTraceGaussCenterX - (2. * _twoDPSFControl->xFWHM));
                  //i_Left = int(D_A2_GaussCenters(emissionLineNumber-1, 0) - (1.5*_twoDPSFControl->xFWHM)) + int(xCentersSwathF[i_Down+iY]) - int(xCentersSwathF[i_yCenter]);

                  /// most right pixel affected by PSF of (emissionLineNumber-1) in this row
                  i_Right = int(xCentersSwathF[i_Down + iY] - dTraceGaussCenterX + (2. * _twoDPSFControl->xFWHM));
                  //i_Right = int(D_A2_GaussCenters(emissionLineNumber-1, 0) + (1.5*_twoDPSFControl->xFWHM)) + int(xCentersSwathF[i_Down+iY]) - int(xCentersSwathF[i_yCenter]);
                  if (i_Left < 0)
                    i_Left = 0;
                  if (i_Right >= fiberTraceIn.getWidth())
                    i_Right = fiberTraceIn.getWidth() - 1;
                  xStart = int(xCentersSwathF[i_Down + iY]) + 0.5 - xCentersSwathF[i_Down + iY] + dTraceGaussCenterX - i_xCenter + i_Left;
                  yStart = i_Down - i_yCenter - pixOffsetY;
                  //xStart = i_Left - i_xCenter - pixOffsetX;
                  //yStart = i_Down - i_yCenter - pixOffsetY;
                  #ifdef __DEBUG_CALC2DPSF__
                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": dTrace = " << dTrace << ": i_Left = " << i_Left << ", i_Right = " << i_Right << endl;
                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": xStart = " << xStart << endl;
                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": yStart = " << yStart << endl;
                  #endif
                  for (int iX = 0; iX <= i_Right - i_Left; ++iX){
                    #ifdef __DEBUG_CALC2DPSF__
                      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": iX = " << iX << ": iY = " << iY << endl;
                    #endif
                    _imagePSF_XRelativeToCenter.push_back(float(xStart + iX));
                    _imagePSF_YRelativeToCenter.push_back(float(yStart + iY));
                    _imagePSF_ZNormalized.push_back(float(trace_In(i_Down+iY, i_Left+iX)));
                    _imagePSF_ZTrace.push_back(float(trace_In(i_Down+iY, i_Left+iX)));
                    _imagePSF_Weight.push_back(fabs(trace_In(i_Down+iY, i_Left+iX)) > 0.000001 ? float(1. / sqrt(fabs(trace_In(i_Down+iY, i_Left+iX)))) : 0.1);//trace_In(i_Down+iY, i_Left+iX) > 0 ? sqrt(trace_In(i_Down+iY, i_Left+iX)) : 0.0000000001);//stddev_In(i_Down+iY, i_Left+iX) > 0. ? 1./pow(stddev_In(i_Down+iY, i_Left+iX),2) : 1.);
                    _imagePSF_XTrace.push_back(float(i_Left + iX));
                    _imagePSF_YTrace.push_back(float(i_Down + iY));
                    #ifdef __DEBUG_CALC2DPSF__
                      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": xCentersSwathF[i_Down + iY=" << i_Down + iY << "] = " << xCentersSwathF[i_Down + iY] << ", xCentersOffset_In(" << i_Down + iY << ") = " << xCentersOffset_In(i_Down + iY) << endl;
                      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1=" << emissionLineNumber-1 << ": x = " << (*_imagePSF_XRelativeToCenter)[nPix] << ", y = " << (*_imagePSF_YRelativeToCenter)[nPix] << ": val = " << trace_In(i_Down + iY, i_Left + iX) << " ^= " << (*_imagePSF_ZNormalized)[nPix] << "; XOrig = " << (*_imagePSF_XTrace)[nPix] << ", YOrig = " << (*_imagePSF_YTrace)[nPix] << endl;
                    #endif
                    ++nPix;
                    ++nPixPSF;
                    sumPSF += trace_In(i_Down+iY, i_Left+iX);
                    #ifdef __DEBUG_CALC2DPSF__
                      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1 = " << emissionLineNumber-1 << ": nPixPSF = " << nPixPSF << ", sumPSF = " << sumPSF << endl;
                    #endif
                  }
                }
                int pixelNo = _imagePSF_ZNormalized.size()-nPixPSF;
                #ifdef __DEBUG_CALC2DPSF__
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1 = " << emissionLineNumber-1 << ": *_imagePSF_ZNormalized.begin()=" << *(_imagePSF_ZNormalized.begin()) << " *(_imagePSF_ZNormalized.end()-1)=" << *(_imagePSF_ZNormalized.end()-1) << endl;
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1 = " << emissionLineNumber-1 << ": (*_imagePSF_ZNormalized)[nPix=" << nPix << " - nPixPSF=" << nPixPSF << " = " << nPix - nPixPSF << " = " << (*_imagePSF_ZNormalized)[nPix-nPixPSF] << endl;
                  cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1 = " << emissionLineNumber-1 << ": (*_imagePSF_ZNormalized)[nPix-1=" << nPix-1 << "] = " << (*_imagePSF_ZNormalized)[nPix-1] << endl;
                #endif
                for (std::vector<float>::iterator iter=_imagePSF_ZNormalized.end()-nPixPSF; iter<_imagePSF_ZNormalized.end(); ++iter){
                  #ifdef __DEBUG_CALC2DPSF__
                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1 = " << emissionLineNumber-1 << ": (*_imagePSF_ZNormalized)[pixelNo=" << pixelNo << "] = " << (*_imagePSF_ZNormalized)[pixelNo] << ", sumPSF = " << sumPSF << endl;
                  #endif
                  if (fabs(sumPSF) < 0.00000001){
                    string message("PSF trace");
                    message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: emissionLineNumber-1 = " + to_string(emissionLineNumber-1);
                    message += ": ERROR: sumPSF == 0";
                    cout << message << endl;
                    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
                  }
                  (*iter) = (*iter) / sumPSF;
                  #ifdef __DEBUG_CALC2DPSF__
                    cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: emissionLineNumber-1 = " << emissionLineNumber-1 << ": (*_imagePSF_ZNormalized)[pixelNo=" << pixelNo << "] = " << (*_imagePSF_ZNormalized)[pixelNo] << endl;
                  #endif
                  ++pixelNo;
                }
              }
            }
          }
          else{
            cout << "PSF trace" << _iTrace << " bin" << _iBin << "::extractPSFs: while: D_A1_GaussFit_Coeffs(2)(=" << D_A1_GaussFit_Coeffs(2) << ") >= (_twoDPSFControl->yFWHM / 1.5)(=" << (_twoDPSFControl->yFWHM / 1.5) << ") || ((_twoDPSFControl->nTermsGaussFit(=" << _twoDPSFControl->nTermsGaussFit << ") < 5) || ((_twoDPSFControl->nTermsGaussFit > 4) && (D_A1_GaussFit_Coeffs(4)(=" << D_A1_GaussFit_Coeffs(4) << ") >= 1000.)) => Skipping emission line" << endl;
          }
        }/// end if MPFitGaussLim
      //}/// end if (D_A1_Guess(1) > (1.5 * _twoDPSFControl->yFWHM)){
    }/// end if (blitz::max(spectrum_In(blitz::Range(I_FirstWideSignalStart, I_FirstWideSignalEnd))) < _twoDPSFControl->saturationLevel){
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
      of << (*_imagePSF_XRelativeToCenter)[iPix];
      of << " " << (*_imagePSF_XTrace)[iPix];
      of << " " << (*_imagePSF_YRelativeToCenter)[iPix];
      of << " " << (*_imagePSF_YTrace)[iPix];
      of << " " << (*_imagePSF_ZTrace)[iPix];
      of << " " << (*_imagePSF_ZNormalized)[iPix];
      of << " " << (*_imagePSF_Weight)[iPix] << endl;
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

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
bool drpStella::PSF<ImageT, MaskT, VarianceT, WavelengthT>::fitPSFKernel()
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

  return true;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
bool drpStella::PSF<ImageT, MaskT, VarianceT, WavelengthT>::calculatePSF()
{
  _pixelsFit.resize(_imagePSF_XRelativeToCenter.size());
  if (!_surfaceFit.estimate(_imagePSF_XRelativeToCenter, _imagePSF_YRelativeToCenter, _pixelsFit)){
    string message("PSF trace");
    message += to_string(_iTrace) + " bin" + to_string(_iBin) + "::extractPSFs: ERROR: surfaceFit.estimate returned FALSE";
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
      cout << "PSF trace" << _iTrace << " bin" << _iBin << "::calculatePSF: iPix=" << iPix << ": x=" << (*_imagePSF_XRelativeToCenter)[iPix] << ", y=" << (*_imagePSF_YRelativeToCenter)[iPix] << ": original pixel value = " << (*_imagePSF_ZNormalized)[iPix] << ", predicted pixel value = " << (*_pixelsFit)[iPix] << ", difference = " << (*_imagePSF_ZNormalized)[iPix]-(*_pixelsFit)[iPix] << endl;
      ofkrig << (*_imagePSF_XRelativeToCenter)[iPix] << " " << (*_imagePSF_YRelativeToCenter)[iPix] << " " << (*_imagePSF_ZNormalized)[iPix] << " " << (*_pixelsFit)[iPix] << endl;
      //      #endif
      //      _pixelsFit.push_back(pred);
    }
    //    gsl_vector_free(pixPos);
    //    #ifdef __DEBUG_CALC2DPSF__
    ofkrig.close();
  #endif
  return true;
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>    
bool drpStella::PSFSet<ImageT, MaskT, VarianceT, WavelengthT>::setPSF(const size_t i,     /// which position?
                                                                      const PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>) & psf)
{
  if (i > _psfs->size()){
    string message("drpStella::PSFSet::setPSF: ERROR: i=");
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
void drpStella::PSFSet<ImageT, MaskT, VarianceT, WavelengthT>::addPSF(const PTR(PSF<ImageT, MaskT, VarianceT, WavelengthT>) & psf /// the Spectrum to add
)
{
  _psfs->push_back(psf);
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR(drpStella::PSFSet<ImageT, MaskT, VarianceT, WavelengthT>) drpStella::math::calculate2dPSFPerBin(const drpStella::FiberTrace<ImageT, MaskT, VarianceT> & fiberTrace,
                                                                                                          const drpStella::Spectrum<ImageT, MaskT, VarianceT, WavelengthT> & spectrum,
                                                                                                          const PTR(drpStella::TwoDPSFControl) & twoDPSFControl){
  int swathWidth = twoDPSFControl->swathWidth;
  ndarray::Array<int, 2, 1> ndArr = fiberTrace.calculateBinBoundY(swathWidth);
  blitz::Array<int, 2> binBoundY = utils::ndarrayToBlitz(ndArr);

  blitz::Array<double, 2> D_A2_2dPSF(2,2);

  PTR(drpStella::PSFSet<ImageT, MaskT, VarianceT, WavelengthT>) psfSet(new drpStella::PSFSet<ImageT, MaskT, VarianceT, WavelengthT>());
  for (int iBin = 0; iBin < binBoundY.rows(); ++iBin){
    /// start calculate2dPSF for bin iBin
    #ifdef __DEBUG_CALC2DPSF__
      cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": calculating PSF for iBin " << iBin << endl;
    #endif
    PTR(drpStella::PSF<ImageT, MaskT, VarianceT, WavelengthT>) psf(new drpStella::PSF< ImageT, MaskT, VarianceT, WavelengthT>((unsigned int)(binBoundY(iBin,0)),
                                                                                                                              (unsigned int)(binBoundY(iBin,1)),
                                                                                                                              twoDPSFControl,
                                                                                                                              fiberTrace.getITrace(),
                                                                                                                              iBin));
    #ifdef __DEBUG_CALC2DPSF__
      cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": iBin " << iBin << ": starting extractPSFs()" << endl;
    #endif
    if (!psf->extractPSFs(fiberTrace, 
                          spectrum)){
      string message("calculate2dPSFPerBin: FiberTrace");
      message += to_string(fiberTrace.getITrace()) + string(": iBin ") + to_string(iBin) + string(": ERROR: psf->extractPSFs() returned FALSE");
      throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
    }
    #ifdef __DEBUG_CALC2DPSF__
      cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": iBin " << iBin << ": extractPSFs() finished" << endl;
      std::vector<float> xTrace((*(psf->getImagePSF_XTrace())));
      cout << "calculate2dPSFPerBin: FiberTrace" << fiberTrace.getITrace() << ": iBin = " << iBin << ": xTrace.size() = " << xTrace.size() << endl;
    #endif
    psfSet->addPSF(psf);
  }
    
  return psfSet;
}
  
template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
PTR(drpStella::PSF<ImageT, MaskT, VarianceT, WavelengthT>)& drpStella::PSFSet<ImageT, MaskT, VarianceT, WavelengthT>::getPSF(const size_t i){
  if (i >= _psfs->size()){
    string message("PSFSet::getPSF(i=");
    message += to_string(i) + string("): ERROR: i > _psfs->size()=") + to_string(_psfs->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  return _psfs->at(i); 
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
const PTR(const drpStella::PSF<ImageT, MaskT, VarianceT, WavelengthT>) drpStella::PSFSet<ImageT, MaskT, VarianceT, WavelengthT>::getPSF(const size_t i) const { 
  if (i >= _psfs->size()){
    string message("PSFSet::getPSF(i=");
    message += to_string(i) + string("): ERROR: i > _psfs->size()=") + to_string(_psfs->size());
    cout << message << endl;
    throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
  }
  return _psfs->at(i); 
}

template<typename ImageT, typename MaskT, typename VarianceT, typename WavelengthT>
bool drpStella::PSFSet<ImageT, MaskT, VarianceT, WavelengthT>::erase(const size_t iStart, const size_t iEnd){
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


template class drpStella::PSF<float, unsigned short, float, float>;
template class drpStella::PSF<double, unsigned short, float, float>;

template class drpStella::PSFSet<float, unsigned short, float, float>;
template class drpStella::PSFSet<double, unsigned short, float, float>;

template PTR(drpStella::PSFSet<float, unsigned short, float, float>) drpStella::math::calculate2dPSFPerBin(drpStella::FiberTrace<float, unsigned short, float> const&, 
                                                                                                                 drpStella::Spectrum<float, unsigned short, float, float> const&,
                                                                                                                 PTR(drpStella::TwoDPSFControl) const&);
template PTR(drpStella::PSFSet<double, unsigned short, float, float>) drpStella::math::calculate2dPSFPerBin(drpStella::FiberTrace<double, unsigned short, float> const&, 
                                                                                                                  drpStella::Spectrum<double, unsigned short, float, float> const&,
                                                                                                                  PTR(drpStella::TwoDPSFControl) const&);

