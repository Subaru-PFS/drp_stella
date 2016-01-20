#ifndef __PFS_DRP_STELLA_CONTROLS_H__
#define __PFS_DRP_STELLA_CONTROLS_H__

#include <vector>
#include "lsst/base.h"

#define stringify( name ) # name
using namespace std;
namespace pfs { namespace drp { namespace stella {

/**
 * Description of fiber trace function
 */
struct FiberTraceFunctionControl {
  /// enum corresponding to legal values of interpolation string
//  enum INTERPOLATION {  CHEBYSHEV=0, LEGENDRE, CUBIC, LINEAR, POLYNOMIAL, NVALUES };
  enum INTERPOLATION {  POLYNOMIAL=0, CHEBYSHEV, NVALUES };
  std::vector<std::string> INTERPOLATION_NAMES = { stringify( CHEBYSHEV ),
//                                                   stringify( LEGENDRE ),
//                                                   stringify( CUBIC ),
//                                                   stringify( LINEAR ),
                                                   stringify( POLYNOMIAL) };
//  std::vector<std::string> INTERPOLATION_NAMES = { stringify( POLYNOMIAL ) };
  LSST_CONTROL_FIELD(interpolation, std::string, "Interpolation schemes, NOTE that only POLYNOMIAL fitting is implement yet!");
  LSST_CONTROL_FIELD(order, unsigned int, "Polynomial order");
  LSST_CONTROL_FIELD(xLow, float, "Lower (left) limit of aperture relative to center position of trace in x (< 0.)");
  LSST_CONTROL_FIELD(xHigh, float, "Upper (right) limit of aperture relative to center position of trace in x");
  LSST_CONTROL_FIELD(nPixCutLeft, int, "Number of pixels to cut off from the width left of center");
  LSST_CONTROL_FIELD(nPixCutRight, int, "Number of pixels to cut off from the width right of from center");
  FiberTraceFunctionControl() :
      interpolation("POLYNOMIAL"),
      order(5),
      xLow(-4.),
      xHigh(4.),
      nPixCutLeft(1),
      nPixCutRight(1) {}
      
  FiberTraceFunctionControl(const FiberTraceFunctionControl& ftfc) : 
      interpolation(ftfc.interpolation),
      order(ftfc.order),
      xLow(ftfc.xLow),
      xHigh(ftfc.xHigh),
      nPixCutLeft(ftfc.nPixCutLeft),
      nPixCutRight(ftfc.nPixCutRight) {}
      
  ~FiberTraceFunctionControl() {}
  
  bool isClassInvariant() const{
    bool isFunctionValid = false;
    #ifdef __DEBUG_FIBERTRACEFUNCTIONCONTROL__
      cout << "FiberTraceFunctionControl::isClassInvariant: fiberTraceFunction->fiberTraceFunctionControl.interpolation = <" << fiberTraceFunction->fiberTraceFunctionControl.interpolation << ">" << endl;
      cout << "FiberTraceFunctionControl::isClassInvariant: fiberTraceFunction->fiberTraceFunctionControl.order = <" << fiberTraceFunction->fiberTraceFunctionControl.order << ">" << endl;
      cout << "FiberTraceFunctionControl::isClassInvariant: fiberTraceFunction->fiberTraceFunctionControl.xLow = <" << fiberTraceFunction->fiberTraceFunctionControl.xLow << ">" << endl;
      cout << "FiberTraceFunctionControl::isClassInvariant: fiberTraceFunction->fiberTraceFunctionControl.xHigh = <" << fiberTraceFunction->fiberTraceFunctionControl.xHigh << ">" << endl;
      cout << "FiberTraceFunctionControl::isClassInvariant: fiberTraceFunction->xCenter = <" << fiberTraceFunction->xCenter << ">" << endl;
      cout << "FiberTraceFunctionControl::isClassInvariant: fiberTraceFunction->yCenter = <" << fiberTraceFunction->yCenter << ">" << endl;
      cout << "FiberTraceFunctionControl::isClassInvariant: fiberTraceFunction->yLow = <" << fiberTraceFunction->yLow << ">" << endl;
      cout << "FiberTraceFunctionControl::isClassInvariant: fiberTraceFunction->yHigh = <" << fiberTraceFunction->yHigh << ">" << endl;
      cout << "FiberTraceFunctionControl::isClassInvariant: fiberTraceFunction->nPixCutLeft = <" << fiberTraceFunction->nPixCutLeft << ">" << endl;
      cout << "FiberTraceFunctionControl::isClassInvariant: fiberTraceFunction->nPixCutRight = <" << fiberTraceFunction->nPixCutRight << ">" << endl;
      cout << "FiberTraceFunctionControl::isClassInvariant: fiberTraceFunction->coefficients = <";
      for (int i = 0; i < static_cast<int>(fiberTraceFunction->coefficients.size()); i++)
        cout << fiberTraceFunction->coefficients[i] << " ";
      cout << ">" << endl;
    #endif

    for ( int fooInt = POLYNOMIAL; fooInt != NVALUES; fooInt++ ){
      #ifdef __DEBUG_FIBERTRACEFUNCTIONCONTROL__
        cout << "FiberTraceFunctionControl::isClassInvariant: INTERPOLATION_NAMES[fooInt] = <" << INTERPOLATION_NAMES[fooInt] << ">" << endl;
      #endif
      if (interpolation.compare(INTERPOLATION_NAMES[fooInt]) == 0){
        isFunctionValid = true;
        #ifdef __DEBUG_FIBERTRACEFUNCTIONCONTROL__
          cout << "FiberTraceFunctionControl::isClassInvariant: " << interpolation << " is valid" << endl;
        #endif
      }
    }
    if (!isFunctionValid){
        #ifdef __DEBUG_FIBERTRACEFUNCTIONCONTROL__
          cout << "FiberTraceFunctionControl::isClassInvariant: " << interpolation << " is NOT valid" << endl;
        #endif
        return false;
    }
    return true;
  }
      
  PTR(FiberTraceFunctionControl) getPointer() const{
    PTR(FiberTraceFunctionControl) ptr(new FiberTraceFunctionControl(*this));
    return ptr;
  }
};

struct FiberTraceFunction {
  FiberTraceFunctionControl fiberTraceFunctionControl; /// User defined Polynomial interpolation and order, xLow, xHigh (width of fiber trace)
  double xCenter; /// Central position of fiber trace in x
  unsigned int yCenter; /// Central position of fiber trace in y
  int yLow; /// lower limit of fiber trace relative to center (< 0)
  unsigned int yHigh; /// lower limit of fiber trace relative to center (>= 0)
  std::vector<double> coefficients; /// polynomial coefficients of fiber trace function

  FiberTraceFunction() :
  fiberTraceFunctionControl(),
  xCenter(0.),
  yCenter(0),
  yLow(0),
  yHigh(0),
  coefficients(4) {}
  
  FiberTraceFunction(const FiberTraceFunction &ftf) :
  fiberTraceFunctionControl(ftf.fiberTraceFunctionControl),
  xCenter(ftf.xCenter),
  yCenter(ftf.yCenter),
  yLow(ftf.yLow),
  yHigh(ftf.yHigh),
  coefficients(ftf.coefficients) {}
  
  ~FiberTraceFunction() {}
  
  bool isClassInvariant() const{
    if (!fiberTraceFunctionControl.isClassInvariant()){
      cout << "FiberTraceFunction::isClassInvariant: ERROR: fiberTraceFunctionControl is not valid! => Returning FALSE" << endl;
      return false;
    }

    if (coefficients.size() <= fiberTraceFunctionControl.order){
      cout << "FiberTraceFunction::isClassInvariant: ERROR: coefficients.size(=" << coefficients.size() << ") < fiberTraceFunctionControl.order(=" << fiberTraceFunctionControl.order << ") => Returning FALSE" << endl;
      return false;
    }

    if (xCenter < 0.){
      cout << "FiberTraceFunction::isClassInvariant: ERROR: xCenter(=" << xCenter << ") < 0 => Returning FALSE" << endl;
      return false;
    }

    if (yCenter < 0.){
      cout << "FiberTraceFunction::isClassInvariant: ERROR: yCenter(=" << yCenter << ") < 0 => Returning FALSE" << endl;
      return false;
    }

    if ((fiberTraceFunctionControl.xLow + xCenter) < 0.){
      cout << "FiberTraceFunction::isClassInvariant: ERROR: (fiberTraceFunctionControl.xLow(=" << fiberTraceFunctionControl.xLow << ") + xCenter(=" << xCenter << ") = " << fiberTraceFunctionControl.xLow + xCenter << " < 0 => Returning FALSE" << endl;
      return false;
    }

    if (yLow > 0.){
      cout << "FiberTraceFunction::isClassInvariant: ERROR: (yLow(=" << yLow << ") > 0 => Returning FALSE" << endl;
      return false;
    }

    if (yLow + yCenter < 0.){
      cout << "FiberTraceFunction::isClassInvariant: ERROR: (yLow(=" << yLow << ") + yCenter(=" << yCenter << ") = " << yLow + yCenter << " < 0 => Returning FALSE" << endl;
      return false;
    }

    if (yHigh < 0.){
      cout << "FiberTraceFunction::isClassInvariant: ERROR: (yHigh(=" << yHigh << ") < 0 => Returning FALSE" << endl;
      return false;
    }
    
    if ( double( fiberTraceFunctionControl.nPixCutLeft + fiberTraceFunctionControl.nPixCutRight )  - ( fiberTraceFunctionControl.xHigh - fiberTraceFunctionControl.xLow ) < 3. ){
      cout << "FiberTraceFunction::isClassInvariant: ERROR: (fiberTraceFunctionControl.nPixCutLeft + fiberTraceFunctionControl.nPixCutRight  - ( fiberTraceFunctionControl.xHigh - fiberTraceFunctionControl.xLow )(=" << (fiberTraceFunctionControl.nPixCutLeft + fiberTraceFunctionControl.nPixCutRight ) - ( fiberTraceFunctionControl.xHigh - fiberTraceFunctionControl.xLow ) << ") < 3 => Returning FALSE" << endl;
      return false;
    }

    return true;
  }
  
  bool setCoefficients(ndarray::Array<double, 1, 1> const& coeffs_In){
      if (coeffs_In.getShape()[0] != (fiberTraceFunctionControl.order + 1)){
          cout << "FiberTraceFunction::setCoefficients: ERROR: size of coeffs_In must be order + 1" << endl;
          return false;
      }
      coefficients.resize(fiberTraceFunctionControl.order + 1);
      for (int i = 0; i < (fiberTraceFunctionControl.order + 1); ++i)
          coefficients[i] = coeffs_In[i];
      return true;
  }
  
  PTR(FiberTraceFunction) getPointer(){
    PTR(FiberTraceFunction) ptr(new FiberTraceFunction(*this));
    return ptr;
  }
};

struct FiberTraceFunctionFindingControl {
  /// enum corresponding to legal values of interpolation string
  LSST_CONTROL_FIELD(fiberTraceFunctionControl, FiberTraceFunctionControl, "Interpolation function and order");
  LSST_CONTROL_FIELD(apertureFWHM, float, "FWHM of an assumed Gaussian spatial profile for tracing the spectra");
  LSST_CONTROL_FIELD(signalThreshold, float, "Signal below this threshold is assumed zero for tracing the spectra");   // Should we use lsst::afw::detection::Threshold?
  LSST_CONTROL_FIELD(nTermsGaussFit, unsigned short, "1 to look for maximum only without GaussFit; 3 to fit Gaussian; 4 to fit Gaussian plus constant (sky), Spatial profile must be at least 5 pixels wide; 5 to fit Gaussian plus linear term (sloped sky), Spatial profile must be at least 6 pixels wide");
  LSST_CONTROL_FIELD(saturationLevel, float, "CCD saturation level");
  LSST_CONTROL_FIELD(minLength, unsigned int, "Minimum aperture length to count as found FiberTrace");
  LSST_CONTROL_FIELD(maxLength, unsigned int, "Maximum aperture length to count as found FiberTrace");
  LSST_CONTROL_FIELD(nLost, unsigned int, "Number of consecutive times the trace is lost before aborting the tracing");

  FiberTraceFunctionFindingControl() :
  fiberTraceFunctionControl(),
  apertureFWHM(2.5),
  signalThreshold(10.),
  nTermsGaussFit(3),
  saturationLevel(65000.),
  minLength(1000),
  maxLength(4096),
  nLost(10)
  {}
  
  FiberTraceFunctionFindingControl(const FiberTraceFunctionFindingControl &ftffc) :
      fiberTraceFunctionControl(ftffc.fiberTraceFunctionControl),
      apertureFWHM(ftffc.apertureFWHM),
      signalThreshold(ftffc.signalThreshold),
      nTermsGaussFit(ftffc.nTermsGaussFit),
      saturationLevel(ftffc.saturationLevel),
      minLength(ftffc.minLength),
      maxLength(ftffc.maxLength),
      nLost(ftffc.nLost)
      {}
      
  ~FiberTraceFunctionFindingControl() {}
  
  bool isClassInvariant() const{
      return false;
  }
      
  PTR(FiberTraceFunctionFindingControl) getPointer(){
    PTR(FiberTraceFunctionFindingControl) ptr(new FiberTraceFunctionFindingControl(*this));
    return ptr;
  }
};

/**
 * Control Fiber trace extraction
 */
struct FiberTraceProfileFittingControl {
    enum {  PISKUNOV=0, SPLINE3, NVALUES_P } PROFILE_INTERPOLATION;/// Profile interpolation method
    std::vector<std::string> PROFILE_INTERPOLATION_NAMES = { stringify( PISKUNOV ),
                                                             stringify( SPLINE3 ) };
    enum {  NONE=0, BEFORE_EXTRACTION, DURING_EXTRACTION, NVALUES } TELLURIC;/// Determine background/sky not at all or before or during profile determination/extraction
    std::vector<std::string> TELLURIC_NAMES = { stringify( NONE ),
                                                stringify( BEFORE_EXTRACTION ),
                                                stringify( DURING_EXTRACTION ) };
    LSST_CONTROL_FIELD(profileInterpolation, std::string, "Method for determining the spatial profile, [PISKUNOV, SPLINE3], default: SPLINE3");
    LSST_CONTROL_FIELD(ccdReadOutNoise, float, "CCD readout noise");
    LSST_CONTROL_FIELD(swathWidth, size_t, "Size of individual extraction swaths, set to 0 to calculate automatically");
    LSST_CONTROL_FIELD(telluric, std::string, "profileInterpolation==PISKUNOV: Method for determining the background (+sky in case of slit spectra, default: NONE)");
    LSST_CONTROL_FIELD(overSample, unsigned int, "Oversampling factor for the determination of the spatial profile (default: 10)");
    LSST_CONTROL_FIELD(maxIterSF, unsigned int, "profileInterpolation==PISKUNOV: Maximum number of iterations for the determination of the spatial profile (default: 8)");
    LSST_CONTROL_FIELD(maxIterSky, unsigned int, "profileInterpolation==PISKUNOV: Maximum number of iterations for the determination of the (constant) background/sky (default: 10)");
    LSST_CONTROL_FIELD(maxIterSig, unsigned int, "Maximum number of iterations for masking bad pixels and CCD defects (default: 2)");
    LSST_CONTROL_FIELD(lambdaSF, float, "profileInterpolation==PISKUNOV: Lambda smoothing factor for spatial profile (default: 1. / overSample)");
    LSST_CONTROL_FIELD(lambdaSP, float, "profileInterpolation==PISKUNOV: Lambda smoothing factor for spectrum (default: 0)");
    LSST_CONTROL_FIELD(wingSmoothFactor, float, "profileInterpolation==PISKUNOV: Lambda smoothing factor to remove possible oscillation of the wings of the spatial profile (default: 0.)");
    LSST_CONTROL_FIELD(lowerSigma, float, "lower sigma rejection threshold if maxIterSig > 0 (default: 3.)");
    LSST_CONTROL_FIELD(upperSigma, float, "upper sigma rejection threshold if maxIterSig > 0 (default: 2.5)");
//    LSST_CONTROL_FIELD(xCorProf, unsigned short, "Number of Cross-correlations of profile and spectrum from one pixel to the left to one pixel to the right");

    FiberTraceProfileFittingControl() :
        profileInterpolation("SPLINE3"),
        ccdReadOutNoise(1.),
        swathWidth(500),
        telluric("NONE"),
        overSample(15),
        maxIterSF(8),
        maxIterSky(0),
        maxIterSig(1),
        lambdaSF(1./static_cast<float>(overSample)),
        lambdaSP(0.),
        wingSmoothFactor(2.),
        lowerSigma(3.),
        upperSigma(3.)//,
        //xCorProf(0)
        {}

    FiberTraceProfileFittingControl(const FiberTraceProfileFittingControl &fiberTraceProfileFittingControl) :
        profileInterpolation(fiberTraceProfileFittingControl.profileInterpolation),
        ccdReadOutNoise(fiberTraceProfileFittingControl.ccdReadOutNoise),
        swathWidth(fiberTraceProfileFittingControl.swathWidth),
        telluric(fiberTraceProfileFittingControl.telluric),
        overSample(fiberTraceProfileFittingControl.overSample),
        maxIterSF(fiberTraceProfileFittingControl.maxIterSF),
        maxIterSky(fiberTraceProfileFittingControl.maxIterSky),
        maxIterSig(fiberTraceProfileFittingControl.maxIterSig),
        lambdaSF(fiberTraceProfileFittingControl.lambdaSF),
        lambdaSP(fiberTraceProfileFittingControl.lambdaSP),
        wingSmoothFactor(fiberTraceProfileFittingControl.wingSmoothFactor),
        lowerSigma(fiberTraceProfileFittingControl.lowerSigma),
        upperSigma(fiberTraceProfileFittingControl.upperSigma)//,
        //xCorProf(fiberTraceProfileFittingControl.xCorProf)
        {}
        
    ~FiberTraceProfileFittingControl() {}
  
  bool isClassInvariant() const{
    bool isProfileInterpolationValid = false;
    bool isTelluricValid = false;
    for ( int fooInt = PISKUNOV; fooInt != NVALUES_P; fooInt++ ){
      #ifdef __DEBUG_FIBERTRACEPROFILEFITTINGCONTROL__
        cout << "FiberTraceProfileFittingControl::isClassInvariant: PROFILE_INTERPOLATION_NAMES[fooInt] = <" << PROFILE_INTERPOLATION_NAMES[fooInt] << ">" << endl;
      #endif
      if (profileInterpolation.compare(PROFILE_INTERPOLATION_NAMES[fooInt]) == 0){
        isProfileInterpolationValid = true;
        #ifdef __DEBUG_FIBERTRACEPROFILEFITTINGCONTROL__
          cout << "FiberTraceProfileFittingControl::isClassInvariant: " << profileInterpolation << " is valid" << endl;
        #endif
      }
    }

    if (!isProfileInterpolationValid){
      cout << "FiberTraceProfileFittingControl::isClassInvariant: ERROR: fiberTraceProfileFittingControl.profileInterpolation is not valid! => Returning FALSE" << endl;
      return false;
    }

    for ( int fooInt = NONE; fooInt != NVALUES; fooInt++ ){
      #ifdef __DEBUG_FIBERTRACEPROFILEFITTINGCONTROL__
        cout << "FiberTraceProfileFittingControl::isClassInvariant: TELLURIC_NAMES[fooInt] = <" << TELLURIC_NAMES[fooInt] << ">" << endl;
      #endif
      if (telluric.compare(TELLURIC_NAMES[fooInt]) == 0){
        isTelluricValid = true;
        #ifdef __DEBUG_FIBERTRACEPROFILEFITTINGCONTROL__
          cout << "FiberTraceProfileFittingControl::isClassInvariant: " << telluric << " is valid" << endl;
        #endif
      }
    }
    if (!isTelluricValid){
      cout << "FiberTraceProfileFittingControl::isClassInvariant: ERROR: telluric(=" << telluric << ") is not valid! => Returning FALSE" << endl;
      return false;
    }

    if (ccdReadOutNoise < 0.){
      cout << "FiberTraceProfileFittingControl::isClassInvariant: ERROR: ccdReadOutNoise(=" << ccdReadOutNoise << ") < 0 => Returning FALSE" << endl;
      return false;
    }

    if (overSample == 0){
      cout << "FiberTraceProfileFittingControl::isClassInvariant: ERROR: overSample(=" << overSample << ") == 0 => Returning FALSE" << endl;
      return false;
    }

    if (maxIterSF == 0){
      cout << "FiberTraceProfileFittingControl::isClassInvariant: ERROR: maxIterSF(=" << maxIterSF << ") == 0 => Returning FALSE" << endl;
      return false;
    }

    if ((telluric.compare(TELLURIC_NAMES[0]) != 0) && (maxIterSky == 0)){
      cout << "FiberTraceProfileFittingControl::isClassInvariant: ERROR: telluric set to not NONE and (maxIterSky(=" << maxIterSky << ") == 0 => Returning FALSE" << endl;
      return false;
    }

    if (lambdaSF < 0.){
      cout << "FiberTraceProfileFittingControl::isClassInvariant: ERROR: lambdaSF(=" << lambdaSF << ") < 0. => Returning FALSE" << endl;
      return false;
    }

    if (lambdaSP < 0.){
      cout << "FiberTraceProfileFittingControl::isClassInvariant: ERROR: lambdaSP(=" << lambdaSP << ") < 0. => Returning FALSE" << endl;
      return false;
    }

    if (wingSmoothFactor < 0.){
      cout << "FiberTraceProfileFittingControl::isClassInvariant: ERROR: wingSmoothFactor(=" << wingSmoothFactor << ") < 0. => Returning FALSE" << endl;
      return false;
    }
    
    if (lowerSigma <= 0.){
      cout << "FiberTraceProfileFittingControl::isClassInvariant: ERROR: lowerSigma(=" << lowerSigma << ") <= 0. => Returning FALSE" << endl;
      return false;
    }
    
    if (lowerSigma <= 0.){
      cout << "FiberTraceProfileFittingControl::isClassInvariant: ERROR: lowerSigma(=" << lowerSigma << ") <= 0. => Returning FALSE" << endl;
      return false;
    }
    
    if (upperSigma <= 0.){
      cout << "FiberTraceProfileFittingControl::isClassInvariant: ERROR: upperSigma(=" << upperSigma << ") <= 0. => Returning FALSE" << endl;
      return false;
    }
    
    return true;
  }
        
  PTR(FiberTraceProfileFittingControl) getPointer() const{
    PTR(FiberTraceProfileFittingControl) ptr(new FiberTraceProfileFittingControl(*this));
    return ptr;
  }
};


/**
 * Description of 2D PSF
 */
struct TwoDPSFControl {
    LSST_CONTROL_FIELD(signalThreshold, float, "Minimum signal above continuum to count as emission line");
    LSST_CONTROL_FIELD(swathWidth, unsigned int, "Size of individual extraction swaths");
    LSST_CONTROL_FIELD(xFWHM, float, "FWHM of an assumed Gaussian PSF perpendicular to the dispersion direction");
    LSST_CONTROL_FIELD(yFWHM, float, "FWHM of an assumed Gaussian PSF in the dispersion direction");
    LSST_CONTROL_FIELD(nTermsGaussFit, unsigned short, "3 to fit Gaussian; 4 to fit Gaussian plus constant (sky), profile must be at least 5 pixels wide; 5 to fit Gaussian plus linear term (sloped sky), profile must be at least 6 pixels wide");
    LSST_CONTROL_FIELD(xCorRangeLowLimit, float, "Lower limit for cross-corellation to given collapsed PSF relative to estimated center position");
    LSST_CONTROL_FIELD(xCorRangeHighLimit, float, "Upper limit for cross-corellation to given collapsed PSF relative to estimated center position");
    LSST_CONTROL_FIELD(xCorStepSize, float, "Step size for cross-corellation to given collapsed PSF");
    LSST_CONTROL_FIELD(saturationLevel, float, "CCD saturation level");
    LSST_CONTROL_FIELD(nKnotsX, unsigned int, "Number of interpolation knots in X direction");
    LSST_CONTROL_FIELD(nKnotsY, unsigned int, "Number of interpolation knots in Y direction");
    LSST_CONTROL_FIELD(regularization, float, "Smoothing factor for thin-plate-spline interpolation");
    LSST_CONTROL_FIELD(weightBase, float, "Weights of thin-plate-spline interpolation will be calculated as weightBase / pixelValue");

    TwoDPSFControl() :
    signalThreshold(1000.),
    swathWidth(500),
    xFWHM(2.5),
    yFWHM(2.5),
    nTermsGaussFit(3),
    xCorRangeLowLimit(-0.2),
    xCorRangeHighLimit(0.2),
    xCorStepSize(0.01),
    saturationLevel(65000.),
    nKnotsX(75),
    nKnotsY(75),
    regularization(0.00005),
    weightBase(10000.){}

    TwoDPSFControl(const TwoDPSFControl &twoDPSFControl) :
    signalThreshold(twoDPSFControl.signalThreshold),
    swathWidth(twoDPSFControl.swathWidth),
    xFWHM(twoDPSFControl.xFWHM),
    yFWHM(twoDPSFControl.yFWHM),
    nTermsGaussFit(twoDPSFControl.nTermsGaussFit),
    xCorRangeLowLimit(twoDPSFControl.xCorRangeLowLimit),
    xCorRangeHighLimit(twoDPSFControl.xCorRangeHighLimit),
    xCorStepSize(twoDPSFControl.xCorStepSize),
    saturationLevel(twoDPSFControl.saturationLevel),
    nKnotsX(twoDPSFControl.nKnotsX),
    nKnotsY(twoDPSFControl.nKnotsY),
    regularization(twoDPSFControl.regularization),
    weightBase(twoDPSFControl.weightBase){}

    ~TwoDPSFControl() {}
    
    PTR(TwoDPSFControl) getPointer(){
      PTR(TwoDPSFControl) ptr(new TwoDPSFControl(*this));
      return ptr;
    }
};

/**
 * Description of Thin-Plate Spline
 */
struct TPSControl {
    enum BASE_FUNC {  TPS=0, MULTIQUADRIC, MULTIQUADRIC_A, R, R_CUBE, INVERSE_MULTIQUADRIC, GENERALIZED_INVERSE_MULTIQUADRIC, WENDLAND_CSRBF, GAUSSIAN, MATERN_C0, MATERN_C2, MATERN_C4 };
    std::vector<std::string> BASE_FUNC_NAMES = { stringify( BASE_FUNC ) };
    LSST_CONTROL_FIELD( baseFunc, std::string, "Name of base function" );
    LSST_CONTROL_FIELD( gMQParameter, double, "Exponential parameter for generalized multiquadric RBF" );
    LSST_CONTROL_FIELD( regularization, double, "Smoothing factor for thin-plate-spline interpolation" );
    LSST_CONTROL_FIELD( shapeParameter, double, "Additional parameter determining the shape of the Radial Basis Function" );
    LSST_CONTROL_FIELD( minimumSeparationDistance, double, "Minimum separation between adjacent knots" );

    TPSControl() :
    baseFunc("TPS"),
    gMQParameter(-2.),
    regularization(0.00001),
    shapeParameter(1.),
    minimumSeparationDistance(0.001)
    {}

    TPSControl(const TPSControl &tpsControl) :
    baseFunc( tpsControl.baseFunc ),
    gMQParameter( tpsControl.gMQParameter ),
    regularization( tpsControl.regularization ),
    shapeParameter( tpsControl.shapeParameter ),
    minimumSeparationDistance( tpsControl.minimumSeparationDistance )
    {}

    ~TPSControl() {}
    
    PTR(TPSControl) getPointer(){
      PTR(TPSControl) ptr(new TPSControl(*this));
      return ptr;
    }
};

struct DispCorControl {
    enum {  POLYNOMIAL=0, CHEBYSHEV, NVALUES_P } FITTING_FUNCTION;/// Profile interpolation method
    std::vector<std::string> PROFILE_INTERPOLATION_NAMES = { stringify( POLYNOMIAL ),
                                                             stringify( CHEBYSHEV ) };
    LSST_CONTROL_FIELD( fittingFunction, std::string, "Function for fitting the dispersion" );
    LSST_CONTROL_FIELD( order, int, "Fitting function order" );
    LSST_CONTROL_FIELD( searchRadius, int, "Radius in pixels relative to line list to search for emission line peak" );
    LSST_CONTROL_FIELD( fwhm, float, "FWHM of emission lines" );
    LSST_CONTROL_FIELD( radiusXCor, int, "Radius in pixels in which to cross correlate a spectrum relative to the reference spectrum" );
    LSST_CONTROL_FIELD( lengthPieces, int, "Length of pieces of spectrum to match to reference spectrum by stretching and shifting" );
    LSST_CONTROL_FIELD( nCalcs, int, "Number of iterations > spectrumLength / lengthPieces, e.g. spectrum length is 3800 pixels, <lengthPieces> = 500, <nCalcs> = 15: run 1: pixels 0-499, run 2: 249-749,...");
    LSST_CONTROL_FIELD( stretchMinLength, int, "Minimum length to stretched pieces to (< lengthPieces)" );
    LSST_CONTROL_FIELD( stretchMaxLength, int, "Maximum length to stretched pieces to (> lengthPieces)" );
    LSST_CONTROL_FIELD( nStretches, int, "Number of stretches between <stretchMinLength> and <stretchMaxLength>");

    DispCorControl() :
        fittingFunction( "POLYNOMIAL" ),
        order( 5 ),
        searchRadius( 2 ),
        fwhm( 2.6 ),
        radiusXCor( 35 ),
        lengthPieces( 500 ),
        nCalcs( 15 ),
        stretchMinLength( 450 ),
        stretchMaxLength( 550 ),
        nStretches( 100 )
        {}

    DispCorControl( const DispCorControl &dispCorControl ) :
        fittingFunction( dispCorControl.fittingFunction ),
        order( dispCorControl.order ),
        searchRadius( dispCorControl.searchRadius ),
        fwhm( dispCorControl.fwhm ),
        radiusXCor( dispCorControl.radiusXCor ),
        lengthPieces( dispCorControl.lengthPieces ),
        nCalcs( dispCorControl.nCalcs ),
        stretchMinLength( dispCorControl.stretchMinLength ),
        stretchMaxLength( dispCorControl.stretchMaxLength ),
        nStretches( dispCorControl.nStretches )
        {}
        
    ~DispCorControl() {}
    
    PTR( DispCorControl ) getPointer(){
      PTR( DispCorControl ) ptr( new DispCorControl( *this ) );
      return ptr;
    }
};
}}}
#endif