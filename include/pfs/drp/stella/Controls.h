#ifndef __PFS_DRP_STELLA_CONTROLS_H__
#define __PFS_DRP_STELLA_CONTROLS_H__

#include <vector>

#include "lsst/base.h"
#include "lsst/pex/config.h"
#include "ndarray.h"
#include "pfs/drp/stella/utils/Utils.h"

#define stringify( name ) # name

using namespace std;

namespace pfs { namespace drp { namespace stella {

/**
 * Description of fiber trace function
 */
struct FiberTraceFunctionControl {
  /// enum corresponding to legal values of interpolation string
  enum INTERPOLATION { POLYNOMIAL=0, CHEBYSHEV, NVALUES };
  LSST_CONTROL_FIELD(interpolation, std::string, "Interpolation schemes, NOTE that only POLYNOMIAL fitting is implement yet!");
  LSST_CONTROL_FIELD(order, unsigned int, "Polynomial order");
  LSST_CONTROL_FIELD(xLow, float, "Lower (left) limit of aperture relative to center position of trace in x (< 0.)");
  LSST_CONTROL_FIELD(xHigh, float, "Upper (right) limit of aperture relative to center position of trace in x");
  LSST_CONTROL_FIELD(nPixCutLeft, int, "Number of pixels to cut off from the width left of center");
  LSST_CONTROL_FIELD(nPixCutRight, int, "Number of pixels to cut off from the width right of from center");
  LSST_CONTROL_FIELD(nRows, int, "Number of CCD rows");
  FiberTraceFunctionControl() :
      interpolation("POLYNOMIAL"),
      order(5),
      xLow(-5.),
      xHigh(5.),
      nPixCutLeft(1),
      nPixCutRight(1),
      nRows(1){}
};

struct FiberTraceFunction{
  PTR(FiberTraceFunctionControl) fiberTraceFunctionControl; /// User defined Polynomial interpolation and order, xLow, xHigh (width of fiber trace)
  float xCenter; /// Central position of fiber trace in x
  unsigned int yCenter; /// Central position of fiber trace in y
  int yLow; /// lower limit of fiber trace relative to center (< 0)
  unsigned int yHigh; /// lower limit of fiber trace relative to center (>= 0)
  ndarray::Array<float, 1, 1> coefficients; /// polynomial coefficients of fiber trace function

  FiberTraceFunction() :
  fiberTraceFunctionControl(new FiberTraceFunctionControl()),
  xCenter(0.),
  yCenter(0),
  yLow(0),
  yHigh(0),
  coefficients(utils::get1DndArray(float(1))){
  };
  
  bool setCoefficients(ndarray::Array<float, 1, 1> const& coeffs_In){
      assert(coeffs_In.getShape()[0] > 0); // safe to cast
      if (static_cast<size_t>(coeffs_In.getShape()[0]) != (fiberTraceFunctionControl->order + 1)) {
          cout << "FiberTraceFunction::setCoefficients: ERROR: size of coeffs_In must be order + 1" << endl;
          return false;
      }
      coefficients = ndarray::allocate(fiberTraceFunctionControl->order + 1);
      coefficients.deep() = coeffs_In;
      return true;
  };
};

struct FiberTraceFunctionFindingControl {
  /// enum corresponding to legal values of interpolation string
  LSST_CONTROL_FIELD(fiberTraceFunctionControl, PTR(FiberTraceFunctionControl), "Interpolation function and order");
  LSST_CONTROL_FIELD(apertureFWHM, float, "FWHM of an assumed Gaussian spatial profile for tracing the spectra");
  LSST_CONTROL_FIELD(signalThreshold, float, "Signal below this threshold is assumed zero for tracing the spectra");   // Should we use lsst::afw::detection::Threshold?
  LSST_CONTROL_FIELD(nTermsGaussFit, unsigned int, "1 to look for maximum only without GaussFit; 3 to fit Gaussian; 4 to fit Gaussian plus constant (sky), Spatial profile must be at least 5 pixels wide; 5 to fit Gaussian plus linear term (sloped sky), Spatial profile must be at least 6 pixels wide");
  LSST_CONTROL_FIELD(saturationLevel, float, "CCD saturation level");
  LSST_CONTROL_FIELD(minLength, unsigned int, "Minimum aperture length to count as found FiberTrace");
  LSST_CONTROL_FIELD(maxLength, unsigned int, "Maximum aperture length to count as found FiberTrace");
  LSST_CONTROL_FIELD(nLost, unsigned int, "Number of consecutive times the trace is lost before aborting the tracing");

  FiberTraceFunctionFindingControl() :
  fiberTraceFunctionControl(new FiberTraceFunctionControl()),
  apertureFWHM(2.5),
  signalThreshold(10.),
  nTermsGaussFit(3),
  saturationLevel(65000.),
  minLength(1000),
  maxLength(4096),
  nLost(10)
  {}
};

/**
 * Control Fiber trace extraction
 */
struct FiberTraceProfileFittingControl {
    LSST_CONTROL_FIELD(swathWidth, int, "Size of individual extraction swaths, set to 0 to calculate automatically");
    LSST_CONTROL_FIELD(overSample, unsigned int, "Oversampling factor for the determination of the spatial profile (default: 10)");
    LSST_CONTROL_FIELD(maxIterSig, unsigned int, "Maximum number of iterations for masking bad pixels and CCD defects (default: 2)");
    LSST_CONTROL_FIELD(lowerSigma, float, "lower sigma rejection threshold if maxIterSig > 0 (default: 3.)");
    LSST_CONTROL_FIELD(upperSigma, float, "upper sigma rejection threshold if maxIterSig > 0 (default: 3.)");

    FiberTraceProfileFittingControl() :
        swathWidth(500),
        overSample(10),
        maxIterSig(1),
        lowerSigma(3.),
        upperSigma(3.)
        {}
};

struct DispCorControl {
    LSST_CONTROL_FIELD(order, int, "Fitting function order");
    LSST_CONTROL_FIELD(searchRadius, int,
                       "Radius in pixels relative to line list to search for emission line peak");
    LSST_CONTROL_FIELD(fwhm, float, "FWHM of emission lines");
    LSST_CONTROL_FIELD(maxDistance, float,
                       "Reject emission lines with center more than this far from the predicted position\n"
                       "\n"
                       "<maxDistance> should be large enough to allow small differences between the\n"
                       "predicted and the measured emission line, but small enough to make sure that\n"
                       "mis-identified lines are identified as such. As a rule of thumb about 1 half\n"
                       "of the FWHM should be a good start");

    DispCorControl() :
        order( 5 ),
        searchRadius( 2 ),
        fwhm( 2.6 ),
        maxDistance(1.5)
        {}
};
}}}
#endif
