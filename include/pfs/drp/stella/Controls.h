#ifndef __PFS_DRP_STELLA_CONTROLS_H__
#define __PFS_DRP_STELLA_CONTROLS_H__

#include <vector>

#include "ndarray.h"
#include "lsst/pex/config.h"
#include "pfs/drp/stella/utils/checkSize.h"


namespace pfs {
namespace drp {
namespace stella {

/**
 * Parameters controlling fiber trace function
 */
struct FiberTraceFunctionControl {
    LSST_CONTROL_FIELD(order, int, "Polynomial order");
    LSST_CONTROL_FIELD(xLow, float,
                       "Lower (left) limit of aperture relative to center position of trace in x (< 0.)");
    LSST_CONTROL_FIELD(xHigh, float,
                       "Upper (right) limit of aperture relative to center position of trace in x");
    LSST_CONTROL_FIELD(nPixCutLeft, int, "Number of pixels to cut off from the width left of center");
    LSST_CONTROL_FIELD(nPixCutRight, int, "Number of pixels to cut off from the width right of from center");
    LSST_CONTROL_FIELD(nRows, int, "Number of CCD rows");

    FiberTraceFunctionControl() :
        order(5),
        xLow(-4.5),
        xHigh(4.5),
        nPixCutLeft(1),
        nPixCutRight(1),
        nRows(1)
        {}
};


/// Function for tracing fiber positions
struct FiberTraceFunction {
    FiberTraceFunctionControl ctrl;  ///< Parameters for function
    float xCenter;  ///< Central position of fiber trace in x
    std::size_t yCenter;  ///< Central position of fiber trace in y
    std::ptrdiff_t yLow;  ///< lower limit of fiber trace relative to center (< 0)
    std::ptrdiff_t yHigh;  ///< lower limit of fiber trace relative to center (>= 0)
    ndarray::Array<float, 1, 1> coefficients;  ///< polynomial coefficients of fiber trace function

    FiberTraceFunction(FiberTraceFunctionControl const& ctrl_=FiberTraceFunctionControl()) :
        ctrl(ctrl_),
        xCenter(0.),
        yCenter(0),
        yLow(-5),
        yHigh(5),
        coefficients(ndarray::allocate(ctrl.order + 1)) {
        coefficients.deep() = 0;
    }

    void setCoefficients(ndarray::Array<float, 1, 1> const& coeffs) {
        utils::checkSize(int(coeffs.getNumElements()), ctrl.order + 1, "FiberTraceFunction::setCoefficients");
        coefficients.deep() = coeffs;
    }

    /**
     * Calculates aperture minimum pixel, central position, and maximum pixel for the trace,
     *
     * Note that if the width of the trace varies depending on the position of the aperture center,
     * 1 pixel left and/or right of the maximum aperture width will get cut off to reduce possible
     * cross-talk between adjacent apertures
     * @param[in] xCenters_In       :: x center positions of trace
     * @param[in] xHigh_In          :: width of trace right of trace (>=0)
     * @param[in] xLow_In           :: width of trace left of trace (<=0)
     * @param[in] nPixCutLeft_In    :: number of pixels to cut off left of trace
     * @param[in] nPixCutRight_In   :: number of pixels to cut off right of trace
     **/
    ndarray::Array<std::size_t, 2, -2> calcMinCenMax(ndarray::Array<float const, 1, 1> const& xCenters);

    /**
     * @brief calculate the xCenters of a FiberTrace from 0 to FiberTrace.getTrace().getHeight()-1
     *
     * NOTE that the WCS starts at [0., 0.], so an xCenter of 1.1 refers to position 0.1 of the second pixel
     *
     * @param function  FiberTraceFunction to use when calculating the xCenters
     */
    ndarray::Array<float, 1, 1> calculateXCenters() const;

    /**
     * @brief calculate the xCenters of a FiberTrace from 0 to FiberTrace.getTrace().getHeight()-1
     *
     * NOTE that the WCS starts at [0., 0.], so an xCenter of 1.1 refers to position 0.1 of the second pixel
     *
     * @param function  FiberTraceFunction to use when calculating the xCenters
     * @param yIn  This range in y will be converted to [-1.0,1.0] when calculating the xCenters
     */
    ndarray::Array<float, 1, 1> calculateXCenters(ndarray::Array<float, 1, 1> const& yIn) const;
    
};


/// Parameters controlling the finding of fiber traces
struct FiberTraceFindingControl {
    LSST_CONTROL_FIELD(apertureFwhm, float,
                       "FWHM of an assumed Gaussian spatial profile for tracing the spectra");
    LSST_CONTROL_FIELD(signalThreshold, float,
                       "Signal below this threshold is assumed zero for tracing the spectra");
    LSST_CONTROL_FIELD(nTermsGaussFit, int,
                       "1 to look for maximum only without GaussFit; 3 to fit Gaussian;\n"
                       "4 to fit Gaussian plus constant (sky), Spatial profile must be at\n"
                       "least 5 pixels wide; 5 to fit Gaussian plus linear term (sloped sky),\n"
                       "Spatial profile must be at least 6 pixels wide");
    LSST_CONTROL_FIELD(saturationLevel, float, "CCD saturation level");
    LSST_CONTROL_FIELD(minLength, int, "Minimum aperture length to count as found FiberTrace");
    LSST_CONTROL_FIELD(maxLength, int, "Maximum aperture length to count as found FiberTrace");
    LSST_CONTROL_FIELD(nLost, int,
                       "Number of consecutive times the trace is lost before aborting the tracing");

    FiberTraceFindingControl() :
        apertureFwhm(2.5),
        signalThreshold(120.),
        nTermsGaussFit(3),
        saturationLevel(65000.),
        minLength(3000),
        maxLength(4096),
        nLost(10)
        {}
};


/**
 * Control Fiber trace extraction
 */
struct FiberTraceProfileFittingControl {
    LSST_CONTROL_FIELD(swathWidth, int,
                       "Size of individual extraction swaths, set to 0 to calculate automatically");
    LSST_CONTROL_FIELD(overSample, int,
                       "Oversampling factor for the determination of the spatial profile (default: 10)");
    LSST_CONTROL_FIELD(maxIterSig, int,
                       "Maximum number of iterations for masking bad pixels and CCD defects (default: 2)");
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

/// Parameters controlling the dispersion correction (wavelength solution)
struct DispersionCorrectionControl {
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

    DispersionCorrectionControl() :
        order(5),
        searchRadius(2),
        fwhm(2.6),
        maxDistance(1.5)
        {}
};

}}} //namespace pfs::drp::stella

#endif // include guard
