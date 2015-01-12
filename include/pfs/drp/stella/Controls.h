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
  enum INTERPOLATION {  CHEBYSHEV=0, LEGENDRE, CUBIC, LINEAR, POLYNOMIAL, NVALUES };
  std::vector<std::string> INTERPOLATION_NAMES = { stringify( CHEBYSHEV ),
                                                   stringify( LEGENDRE ),
                                                   stringify( CUBIC ),
                                                   stringify( LINEAR ),
                                                   stringify( POLYNOMIAL) };
  LSST_CONTROL_FIELD(interpolation, std::string, "Interpolation schemes");
  LSST_CONTROL_FIELD(order, unsigned int, "Polynomial order");
  LSST_CONTROL_FIELD(xLow, float, "Lower (left) limit of aperture relative to center position of trace in x (< 0.)");
  LSST_CONTROL_FIELD(xHigh, float, "Upper (right) limit of aperture relative to center position of trace in x");

  FiberTraceFunctionControl() :
      interpolation("POLYNOMIAL"),
      order(3),
      xLow(-4.),
      xHigh(4.) {}
      
  FiberTraceFunctionControl(const FiberTraceFunctionControl& ftfc) : 
      interpolation(ftfc.interpolation),
      order(ftfc.order),
      xLow(ftfc.xLow),
      xHigh(ftfc.xHigh) {}
      
  ~FiberTraceFunctionControl() {}
      
  PTR(FiberTraceFunctionControl) getPointer() const{
    PTR(FiberTraceFunctionControl) ptr(new FiberTraceFunctionControl(*this));
    return ptr;
  }
};

struct FiberTraceFunction {
  FiberTraceFunctionControl fiberTraceFunctionControl; /// User defined Polynomial interpolation and order, xLow, xHigh (width of fiber trace)
  float xCenter; /// Central position of fiber trace in x
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
  signalThreshold(0.),
  nTermsGaussFit(3),
  saturationLevel(65000.),
  minLength(10),
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
    LSST_CONTROL_FIELD(swathWidth, unsigned int, "Size of individual extraction swaths");
    LSST_CONTROL_FIELD(telluric, std::string, "profileInterpolation==PISKUNOV: Method for determining the background (+sky in case of slit spectra, default: NONE)");
    LSST_CONTROL_FIELD(overSample, unsigned int, "Oversampling factor for the determination of the spatial profile (default: 10)");
    LSST_CONTROL_FIELD(maxIterSF, unsigned int, "profileInterpolation==PISKUNOV: Maximum number of iterations for the determination of the spatial profile (default: 8)");
    LSST_CONTROL_FIELD(maxIterSky, unsigned int, "profileInterpolation==PISKUNOV: Maximum number of iterations for the determination of the (constant) background/sky (default: 10)");
    LSST_CONTROL_FIELD(maxIterSig, unsigned int, "Maximum number of iterations for masking bad pixels and CCD defects (default: 2)");
    LSST_CONTROL_FIELD(lambdaSF, float, "profileInterpolation==PISKUNOV: Lambda smoothing factor for spatial profile (default: 1. / overSample)");
    LSST_CONTROL_FIELD(lambdaSP, float, "profileInterpolation==PISKUNOV: Lambda smoothing factor for spectrum (default: 0)");
    LSST_CONTROL_FIELD(wingSmoothFactor, float, "profileInterpolation==PISKUNOV: Lambda smoothing factor to remove possible oscillation of the wings of the spatial profile (default: 0.)");
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
        wingSmoothFactor(2.)//,
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
        wingSmoothFactor(fiberTraceProfileFittingControl.wingSmoothFactor)//,
        //xCorProf(fiberTraceProfileFittingControl.xCorProf)
        {}
        
    ~FiberTraceProfileFittingControl() {}
        
    PTR(FiberTraceProfileFittingControl) getPointer(){
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
    LSST_CONTROL_FIELD(saturationLevel, float, "CCD saturation level");
    LSST_CONTROL_FIELD(nKnotsX, unsigned int, "Number of interpolation knots in X direction");
    LSST_CONTROL_FIELD(nKnotsY, unsigned int, "Number of interpolation knots in Y direction");
    LSST_CONTROL_FIELD(smooth, float, "Smoothing factor for bidirectional spline interpolation");

    TwoDPSFControl() :
    signalThreshold(1000.),
    swathWidth(500),
    xFWHM(2.5),
    yFWHM(2.5),
    nTermsGaussFit(5),
    saturationLevel(65000.),
    nKnotsX(75),
    nKnotsY(75),
    smooth(35000.){}

    TwoDPSFControl(const TwoDPSFControl &twoDPSFControl) :
    signalThreshold(twoDPSFControl.signalThreshold),
    swathWidth(twoDPSFControl.swathWidth),
    xFWHM(twoDPSFControl.xFWHM),
    yFWHM(twoDPSFControl.yFWHM),
    nTermsGaussFit(twoDPSFControl.nTermsGaussFit),
    saturationLevel(twoDPSFControl.saturationLevel),
    nKnotsX(twoDPSFControl.nKnotsX),
    nKnotsY(twoDPSFControl.nKnotsY),
    smooth(twoDPSFControl.smooth){}

    ~TwoDPSFControl() {}
    
    PTR(TwoDPSFControl) getPointer(){
      PTR(TwoDPSFControl) ptr(new TwoDPSFControl(*this));
      return ptr;
    }
};
}}}
#endif