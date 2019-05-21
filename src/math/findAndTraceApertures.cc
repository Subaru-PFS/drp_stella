#include "ndarray.h"
#include "ndarray/eigen.h"

#include "lsst/log/Log.h"

#include "pfs/drp/stella/math/Math.h"
#include "pfs/drp/stella/math/CurveFitting.h"
#include "pfs/drp/stella/cmpfit-1.2/MPFitting_ndarray.h"
#include "pfs/drp/stella/math/findAndTraceApertures.h"

//#define __DEBUG_FINDANDTRACE__ 1
//#define __DEBUG_XCENTERS__ 1

namespace afwImage = lsst::afw::image;

namespace pfs {
namespace drp {
namespace stella {
namespace math {

namespace {


/**
 * @brief Calculate boundaries for the swaths used for profile calculation
 *
 * @param swathWidth_In : Approximate width for the swaths, will be adjusted
 * to fill the length of the FiberTrace with equally sized swaths
 * @return 2D array containing the pixel numbers for the start and the end
 * of each swath
 */
ndarray::Array<std::size_t, 2, 1>
calcSwathBoundY(
    std::size_t height,  // Number of rows in image
    std::size_t swathWidth
) {
    std::size_t nSwaths = std::max(round(double(height)/swathWidth), 1.0);
    std::size_t const binHeight = height/nSwaths;
    if (nSwaths > 1) {
        nSwaths = (2*nSwaths) - 1;
    }

    // Calculate boundaries of bins
    // Do a test run first to address any rounding errors
    ndarray::Array<std::size_t, 2, 1> boundaries = ndarray::allocate(nSwaths, 2);
    boundaries[0][0] = 0;
    std::size_t index = binHeight;
    boundaries[0][1] = index;
    for (std::size_t iSwath = 1; iSwath < nSwaths; iSwath++) {
        index = binHeight;
        if (iSwath == 1) {
            boundaries[iSwath][0] = boundaries[iSwath - 1][0] + std::size_t(0.5*binHeight);
        } else {
            boundaries[iSwath][0] = boundaries[iSwath - 2][1] + 1;
        }
        boundaries[iSwath][1] = boundaries[iSwath][0] + index;
        if (boundaries[iSwath][1] >= height - 1) {
            nSwaths = iSwath + 1;
        }
        if (iSwath == nSwaths - 1) {
            boundaries[iSwath][1] = height - 1;
        }
    }

    // Repeat, for real this time
    for (size_t iSwath = 1; iSwath < nSwaths; iSwath++){
        index = binHeight;
        if (iSwath == 1) {
            boundaries[iSwath][0] = boundaries[iSwath - 1][0] + std::size_t(0.5*binHeight);
        } else {
            boundaries[iSwath][0] = boundaries[iSwath - 2][1] + 1;
        }
        boundaries[iSwath][1] = boundaries[iSwath][0] + index;
        if (iSwath == nSwaths - 1) {
            boundaries[iSwath][1] = height - 1;
        }
    }
    boundaries[nSwaths - 1][ 1 ] = height - 1;
    return boundaries;
}


/**
 * @brief Helper function for calcProfile, calculates profile for a swath
 *
 * A swath is approximately FiberTraceProfileFittingControl.swathWidth long
 * Each swath is overlapping the previous swath for half of the swath width
 * spectrum:
 * |-----------------------------------------------------------------
 * swaths:
 * |---------------|--------------|--------------|--------------|----
 *         |---------------|--------------|--------------|-----------
 * @param swath : CCD image of the FiberTrace swath
 * @param xCentersSwath : 1D array containing the x center positions for the swath
 * @param iSwath : number of swath
 */
template<typename ImageT, typename MaskT, typename VarianceT>
ndarray::Array<float, 2, 1>
calcProfileSwath(
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> const& swath,
    ndarray::Array<float const, 1, 1> const& xCentersSwath,
    FiberTraceProfileFittingControl const& fitting,
    std::size_t iSwath,
    std::size_t fiberId
) {
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.FiberTrace._calcProfileSwath");

    utils::checkSize(std::size_t(swath.getHeight()), xCentersSwath.getNumElements(),
                     "FiberTrace::_calcProfileSwath image vs centers");

    auto const& imageSwath = swath.getImage()->getArray();
    auto const& maskSwath = swath.getMask()->getArray();
    auto const& varianceSwath = swath.getVariance()->getArray();
    int const width = imageSwath.getShape()[1], height = imageSwath.getShape()[0];

    // Normalize rows in imageSwath
    ndarray::Array<float, 2, 1> imageSwathNormalized = ndarray::allocate(imageSwath.getShape());
    ndarray::Array<float, 1, 1> sumArr = ndarray::allocate(width);
    for (int iRow = 0; iRow < height; ++iRow) {
        sumArr.deep() = ndarray::Array<ImageT const, 1, 1>(imageSwath[ndarray::view(iRow)()]);
        imageSwathNormalized[ndarray::view(iRow)()] = imageSwath[ndarray::view(iRow)()]/ndarray::sum(sumArr);
    }
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath <<
        ": imageSwath = " << imageSwath);
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath <<
        ": imageSwathNormalized = " << imageSwathNormalized);

    // Calculate pixel offset to xCenter
    ndarray::Array<float, 1, 1> xCentersTemp = ndarray::allocate(height);
    xCentersTemp.deep() = xCentersSwath + 0.5;
    ndarray::Array<std::size_t const, 1, 1> const xCentersInt = math::floor<std::size_t>(xCentersTemp);
    ndarray::Array<float, 1, 1> pixelOffset = ndarray::allocate(height);
    pixelOffset.deep() = xCentersInt - xCentersSwath;
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath <<
        ": pixelOffset = " << pixelOffset);
    ndarray::Array<float, 1, 1> const xCenterArrayIndexX = ndarray::allocate(width);
    xCenterArrayIndexX.deep() = ndarray::arange(0, width);
    ndarray::Array<float, 1, 1> const xCenterArrayIndexY = ndarray::allocate(height);
    xCenterArrayIndexY.deep() = 1.0;
    ndarray::Array<float, 2, 1> xArray = ndarray::allocate(height, width);
    xArray.asEigen() = xCenterArrayIndexY.asEigen()*xCenterArrayIndexX.asEigen().transpose();
    double xMin = 1.;
    double xMax = -1.;
    auto itOffset = pixelOffset.begin();
    for (auto itX = xArray.begin(); itX != xArray.end(); ++itX) {
        for (auto itXY = itX->begin(); itXY != itX->end(); ++itXY) {
            *itXY += *itOffset;
            if (*itXY < xMin) xMin = *itXY;
            if (*itXY > xMax) xMax = *itXY;
        }
        ++itOffset;
    }
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath << ": xArray = " << xArray);
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath << ": xMin = " << xMin
                      << ", xMax = " << xMax);
    double xOverSampleStep = 1./fitting.overSample;
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath << ": initial xOverSampleStep = "
                      << xOverSampleStep);

    // adjust xOverSampleStep to cover x from xMin + xOverSampleStep/2 to xMax - xOverSampleStep/2
    int const nSteps = (xMax - xMin)/xOverSampleStep + 1;
    xOverSampleStep = (xMax - xMin)/nSteps;
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath << ": final xOverSampleStep = "
                      << xOverSampleStep);
    ndarray::Array<float, 1, 1> xOverSampled = ndarray::allocate(nSteps);
    double const xStart = xMin + 0.5*xOverSampleStep;
    {
        int iStep = 0;
        for (auto it = xOverSampled.begin(); it != xOverSampled.end(); ++it, ++iStep) {
            *it = xStart + (iStep * xOverSampleStep);
        }
    }
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath << ": xOverSampled = "
                      << xOverSampled);
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath
                      << ": _fiberTraceProfileFittingControl->maxIterSig = "
                      << fitting.maxIterSig);

    /// calculate oversampled profile values
    int iStep = 0;
    std::vector<std::pair<float, float>> valOverSampled;
    bool stepHasValues = false;
    ImageT mean = 0.;
    double rangeStart = xOverSampled[0] - 0.5*xOverSampleStep;
    double rangeEnd = rangeStart + xOverSampleStep;
    for (auto it = xOverSampled.begin(); it != xOverSampled.end(); ++it, ++iStep) {
        stepHasValues = false;
        if (iStep == nSteps - 1) {
            rangeEnd += xOverSampleStep/100.;
        }
        auto indices = math::getIndicesInValueRange<float>(xArray, rangeStart, rangeEnd);
        LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath
                          << ": iStep" << iStep << ": indicesInValueRange = " << indices);
        for (int iterSig = 0; iterSig <= std::max(fitting.maxIterSig, 1); ++iterSig) {
            ndarray::Array<float const, 1, 1> const subArr = math::getSubArray(imageSwathNormalized, indices);
            ndarray::Array<float const, 1, 1> const xSubArr = math::getSubArray(xArray, indices);
            std::size_t const nValues = subArr.getShape()[0];
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath
                              << ": iStep" << iStep << ": iterSig = " << iterSig << ": nValues = " << nValues);
            if (nValues > 1) {
                LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath
                                  << ": iStep = " << iStep << ": iterSig = " << iterSig << ": xSubArr = ["
                                  << xSubArr.getShape()[0] << "]: " << xSubArr);
                LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath
                                  << ": iStep = " << iStep << ": iterSig = " << iterSig << ": subArr = ["
                                  << subArr.getShape()[0] << "]: " << subArr);
                if (fitting.maxIterSig > iterSig) {
                    ndarray::Array<float const, 1, 1> const moments = math::moment(subArr, 2);
                    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath
                                      << ": iStep = " << iStep << ": iterSig = " << iterSig << ": moments = "
                                      << moments);
                    float const mom = std::sqrt(moments[1]);
                    for (int i = nValues - 1; i >= 0; --i) {
                        LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath <<
                            ": iStep" << iStep << ": iterSig = " << iterSig << ": moments[0](=" <<
                            moments[0] << ") - subArr[" << i << "](=" << subArr[i] << ") = " <<
                            moments[0] - subArr[i] <<
                            ", 0. - (_fiberTraceProfileFittingControl->upperSigma(=" <<
                            fitting.upperSigma << ") * sqrt(moments[1](=" <<
                            moments[1] << "))(= " << mom << ") = " <<
                            0. - (fitting.upperSigma*mom));
                        LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath
                                          << ": iStep" << iStep << ": iterSig = " << iterSig
                                          << ": _fiberTraceProfileFittingControl->lowerSigma(="
                                          << fitting.lowerSigma << ") * sqrt(moments[1](="
                                          << moments[1] << "))(= " << mom << ") = "
                                          << fitting.lowerSigma*mom);
                        if ((moments[0] - subArr[i] < 0. - (fitting.upperSigma*mom)) ||
                            (moments[0] - subArr[i] > (fitting.lowerSigma*mom))) {
                          LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath
                                            << ": iStep = " << iStep << ": iterSig = " << iterSig
                                            << ": rejecting element " << i << "from subArr");
                          indices.erase(indices.begin() + i);
                        }
                    }
                }
                ndarray::Array<float const, 1, 1> const moments = math::moment(subArr, 1);
                mean = moments[0];
                LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath <<
                    ": iStep = " << iStep << ": iterSig = " << iterSig << ": mean = " << mean);
                stepHasValues = true;
            }
        }
        if (stepHasValues) {
            valOverSampled.push_back(std::pair<float, float>(*it, mean));
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath
                              << ": valOverSampledVec[" << iStep << "] = (" << valOverSampled[iStep].first
                              << ", " << valOverSampled[iStep].second << ")");
        }
        rangeStart = rangeEnd;
        rangeEnd += xOverSampleStep;
    }

    auto xVecMean = std::make_shared<std::vector<float>>(valOverSampled.size());
    std::transform(valOverSampled.begin(), valOverSampled.end(), xVecMean->begin(),
                   [](std::pair<float, float> const& values) { return values.first; });
    auto yVecMean = std::make_shared<std::vector<float>>(valOverSampled.size());
    std::transform(valOverSampled.begin(), valOverSampled.end(), yVecMean->begin(),
                   [](std::pair<float, float> const& values) { return values.second; });
    math::Spline<float> spline(*xVecMean, *yVecMean, math::Spline<float>::CUBIC_NATURAL); // X must be sorted

    // calculate profile for each row in imageSwath
    ndarray::Array<float, 2, 1> profArraySwath = ndarray::allocate(height, width);
    for (int yy = 0; yy < height; ++yy) {
        LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath <<
                    ": xArray[" << yy << "][*] = " << xArray[ndarray::view(yy)()]);
        for (int xx = 0; xx < width; ++xx) {
            /// The spline's knots are calculated from bins in x centered at the
            /// oversampled positions in x.
            /// Outside the range in x on which the spline is defined, which is
            /// [min(xRange) + overSample/2., max(xRange) - overSample/2.], so to
            /// say in the outer (half overSample), the spline is extrapolated from
            /// the 1st derivative at the end points.
            double const value = spline(xArray[yy][xx]);

            /// Set possible negative profile values to Zero as they are not physical
            profArraySwath[yy][xx] = std::max(value, 0.0);
        }
        LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath
                          << ": profArraySwath[" << yy << "][*] = " << profArraySwath[ndarray::view(yy)()]);
        profArraySwath[ndarray::view(yy)()] =
            profArraySwath[ndarray::view(yy)()]/ndarray::sum(profArraySwath[ndarray::view(yy)()]);
        LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath
                          << ": normalized profArraySwath[" << yy << "][*] = "
                          << profArraySwath[ndarray::view(yy)()]);
    }
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath
                      << ": profArraySwath = " << profArraySwath);
    return profArraySwath;
}


/**
 * @brief Calculate the spatial profile for the FiberTrace
 *
 * Normally this would be a Flat FiberTrace, but in principle, if the spectrum
 * shows some kind of continuum, the spatial profile can still be calculated
 */
template<typename ImageT, typename MaskT, typename VarianceT>
void calcProfile(
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> & trace, // Image of the trace; modified
    ndarray::Array<float const, 1, 1> const& xCenters,
    ndarray::Array<std::size_t const, 2, -2> const& minCenMax,
    FiberTraceProfileFittingControl const& fitting,
    std::size_t fiberId
) {
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.FiberTrace._calcProfile");

    unsigned int const nCols = minCenMax[0][2] - minCenMax[0][0] + 1;
    std::size_t const height = trace.getImage()->getHeight();
    auto traceArray = trace.getImage()->getArray();

    // Calculate boundaries for swaths
    ndarray::Array<std::size_t, 2, 1> swathBoundsY = calcSwathBoundY(height, fitting.swathWidth);
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": swathBoundsY = " << swathBoundsY);
    std::size_t const nSwaths = swathBoundsY.getShape()[0];

    ndarray::Array<std::size_t, 1, 1> nPixArr = ndarray::allocate(nSwaths);
    nPixArr.deep() = swathBoundsY[ndarray::view()(1)] - swathBoundsY[ndarray::view()(0)] + 1;
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": nPixArr = " << nPixArr);
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": nSwaths = " << nSwaths);
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": height = " << height);

    // for each swath
    ndarray::Array<float, 3, 2> slitFuncsSwaths = ndarray::allocate(nPixArr[0], nCols, nSwaths - 1);
    ndarray::Array<float, 2, 1> lastSlitFuncSwath = ndarray::allocate(nPixArr[nSwaths - 1], nCols);
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": slitFuncsSwaths.getShape() = "
                      << slitFuncsSwaths.getShape());
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": slitFuncsSwaths.getShape()[0] = "
                      << slitFuncsSwaths.getShape()[0] << ", slitFuncsSwaths.getShape()[1] = "
                      << slitFuncsSwaths.getShape()[1] << ", slitFuncsSwaths.getShape()[2] = "
                      << slitFuncsSwaths.getShape()[2]);
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": slitFuncsSwaths[view(0)()(0) = "
                      << slitFuncsSwaths[ndarray::view(0)()(0)]);
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": slitFuncsSwaths[view(0)(0,9)(0) = "
                      << slitFuncsSwaths[ndarray::view(0)(0,slitFuncsSwaths.getShape()[1])(0)]);

    lsst::afw::image::Image<ImageT> profile(nCols, height);
    profile = 0.0;

    for (unsigned int iSwath = 0; iSwath < nSwaths; ++iSwath){
        std::size_t yMin = swathBoundsY[iSwath][0];
        std::size_t yMax = swathBoundsY[iSwath][1] + 1;
        LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iSwath = " << iSwath << ": yMin = "
                          << yMin << ", yMax = " << yMax);

        unsigned int nRows = yMax - yMin;
        lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> swath{nCols, nRows};
        auto imageSwath = swath.getImage()->getArray();
        auto maskSwath = swath.getMask()->getArray();
        auto varianceSwath = swath.getVariance()->getArray();
        ndarray::Array<float, 1, 1> xCentersSwath = ndarray::copy(xCenters[ndarray::view(yMin, yMax)]);
        for (unsigned int iRow = 0; iRow < nRows; ++iRow) {
            std::size_t const xLow = minCenMax[iRow + yMin][0];
            std::size_t const xHigh = minCenMax[iRow + yMin][2] + 1;
            imageSwath[ndarray::view(iRow)()] =
                trace.getImage()->getArray()[ndarray::view(yMin + iRow)(xLow, xHigh)];
            maskSwath[ndarray::view(iRow)()] =
                trace.getMask()->getArray()[ndarray::view(yMin + iRow)(xLow, xHigh)];
            varianceSwath[ndarray::view(iRow)()] =
                trace.getVariance()->getArray()[ndarray::view(yMin + iRow)(xLow, xHigh)];
        }
        LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": swath " << iSwath << ": imageSwath = " << imageSwath);

        if (iSwath < nSwaths - 1) {
            slitFuncsSwaths[ndarray::view()()(iSwath)] = calcProfileSwath(swath, xCentersSwath, fitting,
                                                                          iSwath, fiberId);
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": slitFuncsSwaths.getShape() = "
                              << slitFuncsSwaths.getShape());
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": imageSwath.getShape() = " << imageSwath.getShape());
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": xCentersSwath.getShape() = "
                              << xCentersSwath.getShape());
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": swathBoundsY = " << swathBoundsY);
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": nPixArr = " << nPixArr);
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": swath " << iSwath
                              << ": slitFuncsSwaths[ndarray::view()()(iSwath)] = "
                              << slitFuncsSwaths[ndarray::view()()(iSwath)]);
        } else {
            lastSlitFuncSwath.deep() = calcProfileSwath(swath, xCentersSwath, fitting, iSwath, fiberId);
        }
    }

    if (nSwaths == 1) {
        profile.getArray() = lastSlitFuncSwath;
        return;
    }

    std::size_t bin = 0;
    double weightBin0 = 0.;
    double weightBin1 = 0.;
    double rowSum;
    auto profileArray = profile.getArray();
    for (std::size_t iSwath = 0; iSwath < nSwaths - 1; ++iSwath) {
        for (std::size_t iRow = 0; iRow < nPixArr[iSwath]; ++iRow) {
            rowSum = ndarray::sum(slitFuncsSwaths[ndarray::view(iRow)()(iSwath)]);
            if (std::fabs(rowSum) > 0.00000000000000001 && std::isfinite(rowSum)) {
                for (std::size_t iPix = 0; iPix < slitFuncsSwaths.getShape()[1]; iPix++) {
                    slitFuncsSwaths[iRow][iPix][iSwath] = slitFuncsSwaths[iRow][iPix][iSwath]/rowSum;
                }
                LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": slitFuncsSwaths(" << iRow << ", *, "
                                  << iSwath << ") = "
                                  << slitFuncsSwaths[ndarray::view(iRow)()(iSwath)]);
                  rowSum = ndarray::sum(slitFuncsSwaths[ndarray::view(iRow)()(iSwath)]);
                  LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow << ": iSwath = "
                                    << iSwath << ": rowSum = " << rowSum);
            } else {
                slitFuncsSwaths[iRow][ndarray::view()][iSwath] = 0.0;
            }
        }
    }
    for (std::size_t iRow = 0; iRow < nPixArr[nSwaths - 1]; ++iRow) {
        rowSum = ndarray::sum(lastSlitFuncSwath[ndarray::view(iRow)()]);
        if (std::fabs(rowSum) > 0.00000000000000001 && std::isfinite(rowSum)) {
            lastSlitFuncSwath[ndarray::view(iRow)()] = lastSlitFuncSwath[ndarray::view(iRow)()]/rowSum;
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": lastSlitFuncSwath(" << iRow << ", *) = "
                              << lastSlitFuncSwath[ndarray::view(iRow)()]);
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow << ": rowSum = " << rowSum);
        } else {
            lastSlitFuncSwath[ndarray::view(iRow)()] = 0.0;
        }
    }
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": swathBoundsY.getShape() = "
                      << swathBoundsY.getShape() << ", nSwaths = " << nSwaths);
    int iRowSwath = 0;
    for (std::size_t iRow = 0; iRow < height; ++iRow) {
        iRowSwath = iRow - swathBoundsY[bin][0];
        LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow << ", bin = "
                          << bin << ", iRowSwath = " << iRowSwath);
        if ((bin == 0) && (iRow < swathBoundsY[1][0])) {
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": bin=" << bin << " == 0 && iRow="
                              << iRow << " < swathBoundsY[1][0]=" << swathBoundsY[1][0]);
            profileArray[ndarray::view(iRow)()] = slitFuncsSwaths[ndarray::view(iRowSwath)()(0)];
        } else if ((bin == nSwaths - 1) && (iRow >= swathBoundsY[bin-1][1])) {
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": bin=" << bin << " == nSwaths-1="
                              << nSwaths-1 << " && iRow=" << iRow << " >= swathBoundsY[bin-1="
                              << bin - 1 << "][0]=" << swathBoundsY[bin - 1][0]);
            profileArray[ndarray::view(iRow)()] = lastSlitFuncSwath[ndarray::view(iRowSwath)()];
        } else {
            weightBin1 = float(iRow - swathBoundsY[bin + 1][0])/
                (swathBoundsY[bin][1] - swathBoundsY[bin + 1][0]);
            weightBin0 = 1. - weightBin1;
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow << ": nSwaths = " << nSwaths);
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow << ": bin = " << bin);
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow <<
                ": swathBoundsY(bin, *) = " << swathBoundsY[ndarray::view(bin)()]);
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow <<
                ": swathBoundsY(bin+1, *) = " << swathBoundsY[ndarray::view(bin + 1)()]);
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow << ": weightBin0 = "
                              << weightBin0);
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow << ": weightBin1 = "
                              << weightBin1);
            if (bin == nSwaths - 2){
                LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": slitFuncsSwaths(iRowSwath, *, bin) = " <<
                    slitFuncsSwaths[ndarray::view(iRowSwath)()(bin)] <<
                    ", lastSlitFuncSwath[ndarray::view(int(iRow - swathBoundsY[bin+1][0])=" <<
                    int(iRow - swathBoundsY[bin+1][0]) << ")] = " <<
                    lastSlitFuncSwath[ndarray::view(int(iRow - swathBoundsY[bin + 1][0]))()]);
                LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": profile.getArray().getShape() = "
                                  << profileArray.getShape());
                LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": slitFuncsSwath.getShape() = "
                                  << slitFuncsSwaths.getShape());
                LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": lastSlitFuncSwath.getShape() = "
                                  << lastSlitFuncSwath.getShape());
                LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow << ", iRowSwath = "
                                  << iRowSwath << ", bin = " << bin << ", swathBoundsY[bin+1][0] = "
                                  << swathBoundsY[bin+1][0] << ", iRow - swathBoundsY[bin+1][0] = "
                                  << iRow - swathBoundsY[bin + 1][0]);
                LOGLS_TRACE(_log, "_fiberId = " << fiberId <<
                    ": profile.getArray()[ndarray::view(iRow)()] = " << profileArray[ndarray::view(iRow)()]);
                LOGLS_TRACE(_log, "_fiberId = " << fiberId
                                  << ": slitFuncsSwaths[ndarray::view(iRowSwath)()(bin)] = "
                                  << slitFuncsSwaths[ndarray::view(iRowSwath)()(bin)]);
                LOGLS_TRACE(_log, "_fiberId = " << fiberId <<
                    ": ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()" <<
                    "(bin)]).getShape() = " << slitFuncsSwaths[ndarray::view(iRowSwath)()(bin)].getShape());
                LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": weightBin0 = " << weightBin0
                                  << ", weightBin1 = " << weightBin1);
                LOGLS_TRACE(_log, "_fiberId = " << fiberId <<
                    ": lastSlitFuncSwath[ndarray::view(int(iRow - swathBoundsY[bin+1][0]))()] = " <<
                    lastSlitFuncSwath[ndarray::view(int(iRow - swathBoundsY[bin + 1][0]))()]);
                profileArray[ndarray::view(iRow)()] =
                    (slitFuncsSwaths[ndarray::view(iRowSwath)()(bin)]*weightBin0) +
                    (lastSlitFuncSwath[ndarray::view(int(iRow - swathBoundsY[bin + 1][0]))()]*weightBin1);
                LOGLS_TRACE(_log, "_fiberId = " << fiberId
                                  << ": profile.getArray()[ndarray::view(iRow)()] set to "
                                  << profileArray[ndarray::view(iRow)()]);
            } else {
                LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": slitFuncsSwaths(iRowSwath, *, bin) = "
                                  << slitFuncsSwaths[ndarray::view(iRowSwath)()(bin)]
                                  << ", slitFuncsSwaths(int(iRow - swathBoundsY[bin+1][0])="
                                  << int(iRow - swathBoundsY[bin + 1][0]) << ", *, bin+1) = "
                                  << slitFuncsSwaths[ndarray::view(
                                     int(iRow - swathBoundsY[bin + 1][0]))()(bin + 1)]);
                profileArray[ndarray::view(iRow)()] =
                    (slitFuncsSwaths[ndarray::view(iRowSwath)()(bin)]*weightBin0) +
                    (slitFuncsSwaths[ndarray::view(int(iRow - swathBoundsY[bin + 1][0]))()(bin + 1)]*weightBin1);
            }
            double const dSumSFRow = ndarray::sum(profileArray[ndarray::view(iRow)()]);
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow << ": bin = "
                              << bin << ": dSumSFRow = " << dSumSFRow);
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow << ": bin = "
                              << bin << ": profile.getArray().getShape() = "
                              << profileArray.getShape());
            if (std::fabs(dSumSFRow) >= 0.000001 && std::isfinite(dSumSFRow)){
                LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow << ": bin = "
                                  << bin << ": normalizing profile.getArray()[iRow = "
                                  << iRow << ", *]");
                profileArray[ndarray::view(iRow)()] /= dSumSFRow;
            } else {
                profileArray[ndarray::view(iRow)()] = 0.0;
            }
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow << ": bin = "
                              << bin << ": profile.getArray()(" << iRow << ", *) set to "
                              << profileArray[ndarray::view(iRow)()]);
        }

        if (iRow == swathBoundsY[bin][1]){
            bin++;
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iRow = " << iRow << ": bin set to " << bin);
        }
    } // end for (int i_row = 0; i_row < slitFuncsSwaths.rows(); i_row++) {
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": profile.getArray() set to ["
                      << profile.getHeight() << ", " << profile.getWidth() << "]: "
                      << profileArray);

    std::size_t const width = profile.getWidth();
    traceArray.deep() = 0.0;
    for (std::size_t i = 0; i < minCenMax.getShape()[0]; ++i) {
        std::size_t const xMax = minCenMax[i][2] + 1;
        std::size_t const xMin = minCenMax[i][0];
        if (xMax - xMin != width){
            std::string message("_minCenMax[");
            message += std::to_string(i) + "][2](=" + std::to_string(minCenMax[i][2]);
            message += ") - _minCenMax[" + std::to_string(i) + "][0](=";
            message += std::to_string(minCenMax[i][0]) + ") + 1 = ";
            message += std::to_string(minCenMax[i][2] - minCenMax[i][0] + 1) +" != width(=";
            message += std::to_string(width);
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        traceArray[ndarray::view(i)(xMin, xMax)] = profileArray[ndarray::view(i)()];
    }
}


/**
 * @brief Create _trace from maskedImage and _fiberTraceFunction
 *
 * @param maskedImage : MaskedImage from which to extract the FiberTrace from
 * Pre: _xCenters set/calculated
 */
template<typename ImageT, typename MaskT, typename VarianceT>
lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT>
createTrace(
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> const& maskedImage,
    ndarray::Array<float const, 1, 1> const& xCenters,
    ndarray::Array<std::size_t, 2, -2> const& minCenMax,
    std::string const& maskPlane,
    FiberTraceFunction const& function
) {
    ndarray::Array<std::size_t, 1, 1> xMinPix(minCenMax[ndarray::view()(0)]);
    int const xMin = *std::min_element(xMinPix.begin(), xMinPix.end());
    ndarray::Array<std::size_t, 1, 1> xMaxPix(minCenMax[ndarray::view()(2)]);
    int const xMax = *std::max_element(xMaxPix.begin(), xMaxPix.end());

    lsst::geom::Point<int> lowerLeft(xMin, function.yCenter + function.yLow);
    lsst::geom::Extent<int, 2> extent(xMax - xMin + 1, function.yHigh - function.yLow + 1);
    lsst::geom::Box2I box(lowerLeft, extent);
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> trace(maskedImage, box, lsst::afw::image::PARENT,
                                                                  true);
    // mark FiberTrace in Mask
    minCenMax.deep() -= std::size_t(xMin);
    auto mask = *trace.getMask();
    mask.addMaskPlane(maskPlane);
    const auto maskVal = mask.getPlaneBitMask(maskPlane);
    for (std::size_t y = 0; y < minCenMax.getShape()[0]; ++y) {
        for (std::size_t x = minCenMax[y][0]; x <= minCenMax[y][2]; ++x) {
            mask(x, y) |= maskVal;
        }
    }

    return trace;
}


template <typename MaskT>
ndarray::Array<std::size_t, 2, 2>
calculateMinCenMax(
    lsst::afw::image::Mask<MaskT> const& mask,
    std::string const& maskPlane,
    std::size_t fiberId
) {
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.FiberTrace._reconstructMinCenMax");
    ndarray::Array<std::size_t, 2, 2> minCenMax = ndarray::allocate(mask.getHeight(), 3);
    auto xMin = minCenMax[ndarray::view()(0)];
    auto xCen = minCenMax[ndarray::view()(1)];
    auto xMax = minCenMax[ndarray::view()(2)];

    auto const ftMask = mask->getPlaneBitMask(maskPlane);
    int yy = 0;
    auto itMaskRow = mask->getArray().begin();
    for (auto itMin = xMin.begin(), itMax = xMax.begin(), itCen = xCen.begin(); itMin != xMin.end();
         ++itMin, ++itMax, ++itCen, ++itMaskRow, ++yy) {
        LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": *itMaskRow = " << *itMaskRow);
        bool xMinFound = false;
        bool xMaxFound = false;
        int xx = 0;
        for (auto itMaskCol = itMaskRow->begin(); itMaskCol != itMaskRow->end(); ++itMaskCol, ++xx) {
            if (*itMaskCol & ftMask) {
                if (!xMinFound) {
                    *itMin = xx;
                    xMinFound = true;
                    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iY = " << yy << ", iX = " << xx << ": xMinFound");
                }
                *itMax = xx;
            }
        }
        if (*itMax > 0) {
            xMaxFound = true;
            LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": iY = " << yy << ", iX = " << xx << ": xMaxFound");
        }
        if (!xMinFound || !xMaxFound) {
            std::string message("_fiberId = ");
            message += to_string(fiberId) + ": xMin or xMax not found";
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        *itCen = std::size_t((*itMin + *itMax)/2);
    }
    LOGLS_TRACE(_log, "_fiberId = " << fiberId << ": _minCenMax = " << minCenMax);
    return minCenMax;
}

} // anonymous namespace


template<typename ImageT, typename MaskT, typename VarianceT>
FiberTraceSet<ImageT, MaskT, VarianceT> findAndTraceApertures(
    afwImage::MaskedImage<ImageT, MaskT, VarianceT> const& maskedImage,
    DetectorMap const& detectorMap,
    FiberTraceFindingControl const& finding,
    FiberTraceFunctionControl const& function,
    FiberTraceProfileFittingControl const& fitting
) {
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.FiberTrace.findAndTraceApertures");
    LOGLS_TRACE(_log, "::pfs::drp::stella::math::findAndTraceApertures started");

    if (int(finding.apertureFwhm * 2.) + 1 <= int(finding.nTermsGaussFit)) {
        std::string message("finding.apertureFwhm too small for GaussFit ");
        message += "-> Try lower finding.nTermsGaussFit!";
        throw LSST_EXCEPT(pexExcept::LogicError, message.c_str());
    }

    afwImage::MaskedImage<ImageT, MaskT, VarianceT> maskedImageCopy(maskedImage, true);
    afwImage::Image<ImageT> & image = *maskedImageCopy.getImage();
    afwImage::Image<VarianceT> const& variance = *maskedImageCopy.getVariance();

    std::size_t const expectTraces = maskedImage.getWidth()/(function.xLow + function.xHigh);
    FiberTraceSet<ImageT, MaskT, VarianceT> traces(expectTraces + 1);
    lsst::geom::Point2I nextSearchStart(0, 0);
    for (;;) {
        FiberTraceFunction func{function};
        LOGLS_TRACE(_log, "fiberTraceFunction.fiberTraceFunctionControl set");
        auto result = findCenterPositionsOneTrace(image, variance, finding, nextSearchStart);
        std::size_t const length = result.index.size();
        LOGLS_TRACE(_log, "findCenterPositionsOneTrace trace length " <<
                    length << " vs " << finding.minLength);
        if (length < std::size_t(finding.minLength)) {
            break;
        }

        nextSearchStart = result.nextSearchStart;

        LOGLS_TRACE(_log, "D_A1_ApertureCenterIndex = " << result.index);
        LOGLS_TRACE(_log, "D_A1_ApertureCenterPos = " << result.position);

        std::size_t middle = 0.5*length;
        func.xCenter = result.position[middle];
        func.yCenter = result.index[middle];
        func.yHigh = std::size_t(result.index[length - 1]) - func.yCenter;
        func.yLow = std::ptrdiff_t(result.index[0]) - func.yCenter;
        LOGLS_TRACE(_log, "fiberTraceFunction->xCenter = " << func.xCenter);
        LOGLS_TRACE(_log, "fiberTraceFunction->yCenter = " << func.yCenter);
        LOGLS_TRACE(_log, "fiberTraceFunction->yHigh = " << func.yHigh);
        LOGLS_TRACE(_log, "fiberTraceFunction->yLow = " << func.yLow);
        LOGLS_TRACE(_log, "fiberTraceFunction bbox low = " << func.yCenter + func.yLow);

        /// Fit Polynomial
        float xRangeMin = result.index[0];
        float xRangeMax = result.index[length - 1];
        auto const fit = fitPolynomial(vectorToArray(result.index), vectorToArray(result.position),
                                       function.order, xRangeMin, xRangeMax);
        func.coefficients.deep() = fit.coeffs;
        LOGLS_TRACE(_log, "after PolyFit: fiberTraceFunction->coefficients = " << func.coefficients);

        // Find the fiberId
        lsst::geom::Point2D const center(func.xCenter, func.yCenter);
        std::size_t const fiberId = detectorMap.findFiberId(center);

        FiberTraceFunction useFunction{func};
        useFunction.ctrl.nRows = maskedImage.getHeight();
        ndarray::Array<float const, 1, 1> xCenters = useFunction.calculateXCenters();
        ndarray::Array<std::size_t, 2, -2> minCenMax = useFunction.calcMinCenMax(xCenters);

        lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> trace = createTrace(
            maskedImage, xCenters, minCenMax, fiberMaskPlane, useFunction);
        calcProfile(trace, xCenters, minCenMax, fitting, fiberId);
        traces.add(trace, fiberId);
    }

    traces.sortTracesByXCenter();

    return traces;
}


namespace {

struct PointCompare {
    bool operator()(lsst::geom::Point2D const& left, lsst::geom::Point2D const& right) {
        return std::tie(left.getX(), left.getY()) < std::tie(right.getX(), right.getY());
    }
};

} // anonymous namespace


template<typename ImageT, typename VarianceT>
FindCenterPositionsOneTraceResult findCenterPositionsOneTrace(
    afwImage::Image<ImageT> & image,
    afwImage::Image<VarianceT> const& variance,
    FiberTraceFindingControl const& finding,
    lsst::geom::Point<int> const& start
) {
    using FloatArray = ndarray::Array<float, 1, 1>;
    using IntArray = ndarray::Array<std::size_t, 1, 1>;

    std::size_t const width = image.getWidth();
    std::size_t const height = image.getHeight();

    lsst::geom::Point<int> nextSearchStart(start);
    ndarray::Array<ImageT, 2, 1> const imageArray = image.getArray();
    ndarray::Array<VarianceT const, 2, 1> const varianceArray = variance.getArray();
    std::size_t minWidth = std::max(int(1.5*finding.apertureFwhm), int(finding.nTermsGaussFit));
    float maxTimesApertureWidth = 4.0; // XXX make configurable
    std::vector<float> gaussFitVariances;
    std::vector<float> gaussFitMean;
    int const nInd = 100; // XXX make configurable
    ndarray::Array<float, 2, 1> indGaussArr = ndarray::allocate(nInd, 2);

    std::ptrdiff_t maxApertureWidth = maxTimesApertureWidth*finding.apertureFwhm;

    std::ptrdiff_t firstWideSignal;
    std::ptrdiff_t firstWideSignalEnd;
    std::ptrdiff_t firstWideSignalStart;
    std::size_t rowBak;
    bool apertureFound;
    ndarray::Array<float, 1, 1> indexCol = ndarray::allocate(width);
    indexCol.deep() = ndarray::arange(0, width);
    #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: D_A1_IndexCol = " << indexCol << endl;
    #endif
    FloatArray guess = ndarray::allocate(finding.nTermsGaussFit);
    FloatArray gaussFitCoeffs = ndarray::allocate(finding.nTermsGaussFit);
    FloatArray gaussFitCoeffsBak = ndarray::allocate(finding.nTermsGaussFit);
    ndarray::Array<std::size_t, 1, 1> signal = ndarray::allocate(width);
    signal.deep() = 0;

    FloatArray apertureCenter = ndarray::allocate(height);
    FloatArray apertureCenterErr = ndarray::allocate(height);
    IntArray apertureCenterIndex = ndarray::allocate(height);
    apertureCenter.deep() = 0.;
    apertureCenterErr.deep() = 0.;
    apertureCenterIndex.deep() = 0;

    IntArray indSignal;
    #if defined(__DEBUG_FINDANDTRACE__)
    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: started" << endl;
    #endif
    IntArray ind;
    IntArray where;
    #if defined(__DEBUG_FINDANDTRACE__)
    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: fiberTraceFunctionFindingControl->signalThreshold = " <<
        finding.signalThreshold << endl;
    #endif

    FindCenterPositionsOneTraceResult result;
    result.index.reserve(height);
    result.position.reserve(height);
    result.error.reserve(height);

    // Search for Apertures

    int startIndex = start.getX();
    for (std::size_t row = nextSearchStart.getY(); row < height; ++row) {
        #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
        cout << "i_Row = " << row << ": ccdArray[i_Row][*] = " << imageArray[ndarray::view(row)()] << endl;
        #elif defined(__DEBUG_FINDANDTRACE__)
        cout << "i_Row = " << row << " starting" << endl;
        #endif

        nextSearchStart.setY(row);
        nextSearchStart.setX(startIndex);
        rowBak = row;
        for (std::size_t col = 0; col < width; ++col) {
            if (col == 0) {
                signal[col] = (imageArray[row][col] > finding.signalThreshold) ? 1 : 0;
            } else if (imageArray[row][col] > finding.signalThreshold) {
                signal[col] = signal[col - 1] + 1;
            }
        }

        apertureFound = false;
        while (!apertureFound) {
            gaussFitVariances.resize(0);
            gaussFitMean.resize(0);
            #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
            cout << "pfs::drp::stella::math::findAndTraceApertures: while: I_A1_Signal = " << signal << endl;
            cout << "pfs::drp::stella::math::findAndTraceApertures: while: I_MinWidth = " << minWidth << endl;
            cout << "pfs::drp::stella::math::findAndTraceApertures: while: I_StartIndex = " << startIndex << endl;
            #endif
            firstWideSignal = firstIndexWithValueGEFrom(signal, minWidth, startIndex);
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": I_FirstWideSignal found at index " << firstWideSignal << ", I_StartIndex = " <<
                startIndex << endl;
            #endif
            if (firstWideSignal < 0) {
                startIndex = 0;
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                    ": No Aperture found in row " << row << ", trying next row" << endl;
                #endif
                break;
            }

            firstWideSignalStart = lastIndexWithZeroValueBefore(signal, firstWideSignal) + 1;
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": 1. I_FirstWideSignalStart = " << firstWideSignalStart << endl;
            #endif

            firstWideSignalEnd = firstIndexWithZeroValueFrom(signal, firstWideSignal) - 1;
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": I_FirstWideSignalEnd = " << firstWideSignalEnd << endl;
            #endif

            if (firstWideSignalStart < 0) {
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                    ": WARNING: No start of aperture found -> Going to next Aperture." << endl;
                #endif

                if (firstWideSignalEnd < 0) {
                    #if defined(__DEBUG_FINDANDTRACE__)
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                        ": 1. WARNING: No end of aperture found -> Going to next row." << endl;
                    #endif
                    break;
                }
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                    ": End of first wide signal found at index " << firstWideSignalEnd << endl;
                #endif
                // Set start index for next run
                startIndex = firstWideSignalEnd + 1;
                continue;
            }
            // Fit Gaussian and Trace Aperture
            if (firstWideSignalEnd < 0) {
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                    ": 2. WARNING: No end of aperture found -> Going to next row." << endl;
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                    ": B_ApertureFound = " << apertureFound << endl;
                #endif
                break;
            }
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": End of first wide signal found at index " << firstWideSignalEnd << endl;
            #endif

            if (firstWideSignalEnd - firstWideSignalStart + 1 > maxApertureWidth) {
                firstWideSignalEnd = firstWideSignalStart + maxApertureWidth;
            }
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": I_FirstWideSignalStart = " << firstWideSignalStart << ", I_FirstWideSignalEnd = " <<
                firstWideSignalEnd << endl;
            #endif
            signal[ndarray::view(firstWideSignalStart, firstWideSignalEnd + 1)] = 0;

            // Set start index for next run
            startIndex = firstWideSignalEnd + 1;
            std::size_t const length = firstWideSignalEnd - firstWideSignalStart + 1;

            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": I_Length = " << length << endl;
            #endif
            if (finding.nTermsGaussFit == 0) { // look for maximum only
                apertureCenter.deep() = 0.;
                apertureCenterErr.deep() = 0.;
                apertureFound = true;
                std::size_t maxPos = 0;
                ImageT tMax = 0.;
                for (std::ptrdiff_t i = firstWideSignalStart; i <= firstWideSignalEnd; ++i) {
                    if (imageArray[row][i] > tMax){
                        tMax = imageArray[row][i];
                        maxPos = i;
                    }
                }
                apertureCenter[row] = maxPos;
                apertureCenterErr[row] = 0.5;
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                    ": Aperture found at " << apertureCenter[row] << endl;
                #endif

                // Set signal to zero
                if (firstWideSignalEnd - 1 >= firstWideSignalStart + 1) {
                    imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                }
                continue;
            }
            // Fitting multiple Gaussian terms
            if (length <= std::size_t(finding.nTermsGaussFit)) {
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                    ": WARNING: Width of aperture <= " << finding.nTermsGaussFit <<
                    "-> abandoning aperture" << endl;
                #endif

                // Set signal to zero
                if (firstWideSignalEnd - 1 >= firstWideSignalStart + 1) {
                  imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                }
                continue;
            }
            // populate Arrays for GaussFit
            FloatArray x = copy(indexCol[ndarray::view(firstWideSignalStart, firstWideSignalEnd + 1)]);
            FloatArray y = copy(imageArray[ndarray::view(row)(firstWideSignalStart, firstWideSignalEnd + 1)]);
            for (auto it = y.begin(); it != y.end(); ++it) {
                if (*it < 0.000001) {
                    *it = 1.;
                }
            }
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: 1. D_A1_Y set to " <<
                y << endl;
            #endif
            ndarray::Array<VarianceT const, 1, 1> const varianceSlice =
                varianceArray[ndarray::view(row)(firstWideSignalStart, firstWideSignalEnd + 1)];
            FloatArray yErr = ndarray::allocate(length);
            std::transform(varianceSlice.begin(), varianceSlice.end(), yErr.begin(),
                           [](ImageT value) { return value > 0 ? std::sqrt(value) : 1.0; });

            // Guess values for GaussFit
            if (finding.nTermsGaussFit == 3) {
                guess[0] = *std::max_element(y.begin(), y.end());
                guess[1] = firstWideSignalStart + 0.5*(firstWideSignalEnd - firstWideSignalStart);
                guess[2] = finding.apertureFwhm/2.35;
            } else if (finding.nTermsGaussFit > 3) {
                guess[3] = std::max(std::min(y[0], y[length - 1]), float(0.1));
                if (finding.nTermsGaussFit > 4) {
                    guess[4] = (y[length - 1] - y[0])/(length - 1);
                }
            }

            gaussFitCoeffs.deep() = 0.;
            FloatArray gaussFitCoeffsErr = ndarray::allocate(gaussFitCoeffs.size());
            gaussFitCoeffsErr.deep() = 0.;

            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": D_A1_X = " << x << ", D_A1_Y = " << y << endl;
            #endif

            ndarray::Array<int, 2, 1> limited = ndarray::allocate(finding.nTermsGaussFit, 2);
            limited.deep() = 1;
            ndarray::Array<float, 2, 1> limits = ndarray::allocate(finding.nTermsGaussFit, 2);
            limits[0][0] = 0.; // Peak lower limit
            limits[0][1] = 2.*guess[0]; // Peak upper limit
            limits[1][0] = firstWideSignalStart; // Centroid lower limit
            limits[1][1] = firstWideSignalEnd; // Centroid upper limit
            limits[2][0] = finding.minSigma; // Sigma lower limit
            limits[2][1] = finding.maxSigma; // Sigma upper limit
            if (finding.nTermsGaussFit > 3) {
                limits[3][0] = 0.;
                limits[3][1] = 2.*guess[3];
                if (finding.nTermsGaussFit > 4) {
                    limits[4][0] = guess[4]/10.;
                    limits[4][1] = guess[4]*10.;
                }
            }
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": 1. starting MPFitGaussLim: D_A1_Guess = " << guess << ", I_A2_Limited = " <<
                limited << ", D_A2_Limits = " << limits << endl;
            #endif

            if (!MPFitGaussLim(x, y, yErr, guess, limited, limits, 0, false,
                               gaussFitCoeffs, gaussFitCoeffsErr)) {
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": while: WARNING: GaussFit FAILED -> abandoning aperture" << endl;
                #endif

                // Set start index for next run
                startIndex = firstWideSignalEnd + 1;
                imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                continue;
            }
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                ": while: D_A1_GaussFit_Coeffs = " << gaussFitCoeffs << endl;
            if (gaussFitCoeffs[0] > finding.saturationLevel){
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": while: WARNING: Signal appears to be saturated" << endl;
            }
            if (gaussFitCoeffs[1] < firstWideSignalStart + 0.25*length ||
                gaussFitCoeffs[1] > firstWideSignalStart + 0.75*length) {
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": while: Warning: Center of Gaussian far away from middle of signal" << endl;
            }
            #endif
            if (gaussFitCoeffs[1] < firstWideSignalStart ||
                gaussFitCoeffs[1] > firstWideSignalEnd) {
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": while: Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                #endif

                // Set signal to zero
                imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;

                /// Set start index for next run
                startIndex = firstWideSignalEnd + 1;
                continue;
            }
            if (gaussFitCoeffs[2] < finding.minSigma || gaussFitCoeffs[2] > finding.maxSigma) {
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": while: WARNING: sigma = " << gaussFitCoeffs[2] <<
                    " outside range -> abandoning aperture" << endl;
                #endif

                // Set signal to zero
                imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": while: B_ApertureFound = " << apertureFound <<
                    ": 1. Signal set to zero from I_FirstWideSignalStart = " <<
                    firstWideSignalStart << " to I_FirstWideSignalEnd = " <<
                    firstWideSignalEnd << endl;
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": while: 1. ccdArray(i_Row = " << row << ", Range(I_FirstWideSignalStart = " <<
                    firstWideSignalStart << ", I_FirstWideSignalEnd = " << firstWideSignalEnd <<
                    ")) set to " <<
                    imageArray[ndarray::view(row)(firstWideSignalStart, firstWideSignalEnd)] << endl;
                #endif
                // Set start index for next run
                startIndex = firstWideSignalEnd + 1;
                continue;
            }
            apertureCenter.deep() = 0.;
            apertureCenterErr.deep() = 0.;
            apertureFound = true;
            apertureCenter[row] = gaussFitCoeffs[1];
            apertureCenterErr[row] = gaussFitCoeffsErr[1];
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: while: i_Row = " << row <<
                ": Aperture found at " << apertureCenter[row] << endl;
            #endif
            imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
        } // while no aperture found

        if (apertureFound) {
            /// Trace Aperture
            std::set<lsst::geom::Point2D, PointCompare> xySorted;
            int apertureLength = 1;
            int length = 1;
            int apertureLost = 0;
            #if defined(__DEBUG_FINDANDTRACE__)
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                ": Starting to trace aperture " << endl;
            #endif
            gaussFitCoeffsBak.deep() = gaussFitCoeffs;
            while (apertureFound && (apertureLost < finding.nLost) && (row < height - 1) &&
                   length < finding.maxLength) {
                ++row;
                ++apertureLength;
                ++length;
                if (finding.nTermsGaussFit == 0) {
                    // look for maximum only
                    apertureFound = true;
                    std::size_t maxPos = 0;
                    ImageT tMax = 0.;
                    for (std::ptrdiff_t i = firstWideSignalStart; i <= firstWideSignalEnd; ++i) {
                        if (imageArray[row][i] > tMax) {
                            tMax = imageArray[row][i];
                            maxPos = i;
                        }
                    }
                    if (tMax < finding.signalThreshold) {
                        apertureLost++;
                    } else {
                        apertureCenter[row] = maxPos;
                        apertureCenterErr[row] = 0.5; // Half a pixel uncertainty
                        #if defined(__DEBUG_FINDANDTRACE__)
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                            ": Aperture found at " << apertureCenter[row] << endl;
                        #endif
                        if (apertureCenter[row] < apertureCenter[row - 1]) {
                            --firstWideSignalStart;
                            --firstWideSignalEnd;
                        } else if (apertureCenter[row] > apertureCenter[row - 1]) {
                            ++firstWideSignalStart;
                            ++firstWideSignalEnd;
                        }
                        imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                    }
                } else {
                    firstWideSignalStart = gaussFitCoeffsBak[1] - 1.6*gaussFitCoeffsBak[2];
                    firstWideSignalEnd = gaussFitCoeffsBak[1] + 1.6 * gaussFitCoeffsBak[2] + 1;
                    if (firstWideSignalStart < 0. || firstWideSignalEnd >= std::ptrdiff_t(width)) {
                        #if defined(__DEBUG_FINDANDTRACE__)
                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                            ": start or end of aperture outside CCD -> Aperture lost" << endl;
                        #endif
                        // Set signal to zero
                        if (firstWideSignalStart < 0) {
                            firstWideSignalStart = 0;
                        }
                        if (firstWideSignalEnd >= std::ptrdiff_t(width)) {
                            firstWideSignalEnd = width - 1;
                        }
                        imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                        ++apertureLost;
                    } else {
                        length = firstWideSignalEnd - firstWideSignalStart + 1;
                        std::size_t const fitSize = firstWideSignalEnd - firstWideSignalStart + 1;

                        if (length <= finding.nTermsGaussFit) {
                            #if defined(__DEBUG_FINDANDTRACE__)
                            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                ": Warning: Width of Aperture <= " << finding.nTermsGaussFit <<
                                " -> Lost Aperture" << endl;
                            #endif
                            // Set signal to zero
                            imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                            ++apertureLost;
                        } else {
                            FloatArray x = copy(indexCol[ndarray::view(firstWideSignalStart, firstWideSignalEnd + 1)]);
                            FloatArray y = copy(imageArray[ndarray::view(row)(firstWideSignalStart, firstWideSignalEnd + 1)]);
                            FloatArray yErr = ndarray::allocate(fitSize);
                            auto const var = varianceArray[ndarray::view(row)(firstWideSignalStart, firstWideSignalEnd + 1)];
                            yErr.deep() = 1.0;

                            std::size_t numGood = 0;
                            for (auto it = y.begin(); it != y.end(); ++it) {
                                if (*it < 0.000001) {
                                    *it = 1.;
                                }
                                if (*it >= finding.signalThreshold) {
                                    ++numGood;
                                }
                            }
                            if (numGood < minWidth) {
                                // Set signal to zero
                                imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                                ++apertureLost;
                                #if defined(__DEBUG_FINDANDTRACE__)
                                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                    ": Signal not wide enough => Aperture lost" << endl;
                                #endif
                            } else {
                                #if defined(__DEBUG_FINDANDTRACE__)
                                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                    ": 2. D_A1_Y set to " << y << endl;
                                #endif
                                guess.deep() = gaussFitCoeffsBak;
                                gaussFitCoeffs.deep() = 0.;
                                ndarray::Array<float, 1, 1> gaussFitCoeffsErr = ndarray::allocate(gaussFitCoeffs.size());
                                gaussFitCoeffsErr.deep() = 0.;

                                #if defined(__DEBUG_FINDANDTRACE__)
                                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                    ": while: D_A1_X = " << x << ", D_A1_Y = " << y << endl;
                                #endif

                                ndarray::Array<int, 2, 1> limited = ndarray::allocate(finding.nTermsGaussFit, 2);
                                limited.deep() = 1;
                                ndarray::Array<float, 2, 1> limits = ndarray::allocate(finding.nTermsGaussFit, 2);
                                limits[0][0] = 0.; // Peak lower limit
                                limits[0][1] = 2.*guess[0]; // Peak upper limit
                                limits[1][0] = firstWideSignalStart; // Centroid lower limit
                                limits[1][1] = firstWideSignalEnd; // Centroid upper limit
                                limits[2][0] = finding.minSigma; // Sigma lower limit
                                limits[2][1] = finding.maxSigma; // Sigma upper limit

                                // Restrict values from wandering once we're established
                                // XXX hard-coded values here should be made configurable
                                if (gaussFitVariances.size() > 15) {
                                    double const sum = std::accumulate(gaussFitMean.end() - 10, gaussFitMean.end(), 0.0);
                                    double const mean = sum/10.; // we're using the last 10 points; see previous line
                                    #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
                                    for (int iMean = 0; iMean < gaussFitMean.size(); ++iMean){
                                        cout << "gaussFitMean[" << iMean << "] = " << gaussFitMean[iMean] << endl;
                                        cout << "gaussFitVariances[" << iMean << ") = " << gaussFitVariances[iMean] << endl;
                                    }
                                    cout << "sum = " << sum << ", mean = " << mean << endl;
                                    #endif
                                    double const sqSum = std::inner_product(gaussFitMean.end() - 10, gaussFitMean.end(),
                                                                            gaussFitMean.end() - 10, 0.0);
                                    double const stdev = std::sqrt(sqSum/10 - mean*mean);
                                    #if defined(__DEBUG_FINDANDTRACE__)
                                    cout << "GaussFitMean: sq_sum = " << sqSum << ", stdev = " << stdev << endl;
                                    #endif
                                    guess[1] = mean;
                                    limits[1][0] = mean - (3. * stdev) - 0.1;
                                    limits[1][1] = mean + (3. * stdev) + 0.1;
                                    #if defined(__DEBUG_FINDANDTRACE__)
                                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                        ": while: D_A1_Guess[1] = " << guess[1] << ", Limits = " <<
                                        limits[ndarray::view(1)()] << endl;
                                    #endif
                                    // for (int iMean = 0; iMean < gaussFitMean.size(); ++iMean) // Inserted in error?
                                    double const sumVar = std::accumulate(gaussFitVariances.end() - 10, gaussFitVariances.end(), 0.0);
                                    double const meanVar = sumVar/10.;
                                    #if defined(__DEBUG_FINDANDTRACE__)
                                    cout << "GaussFitVariance: sum = " << sumVar << ", mean = " << meanVar << endl;
                                    #endif
                                    double const sqSumVar = std::inner_product(gaussFitVariances.end() - 10, gaussFitVariances.end(),
                                                                               gaussFitVariances.end() - 10, 0.0);
                                    double const stdevVar = std::sqrt(sqSumVar/10 - meanVar*meanVar);
                                    #if defined(__DEBUG_FINDANDTRACE__)
                                    cout << "sq_sum = " << sqSumVar << ", stdev = " << stdevVar << endl;
                                    #endif
                                    guess[2] = meanVar;
                                    limits[2][0] = meanVar - (3.*stdevVar);
                                    limits[2][1] = meanVar + (3.*stdevVar);
                                    #if defined(__DEBUG_FINDANDTRACE__)
                                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                        ": while: D_A1_Guess[2] = " << guess[2] << ", Limits = " <<
                                        limits[ndarray::view(2)()] << endl;
                                    #endif
                                }
                                if (finding.nTermsGaussFit > 3) {
                                    limits[3][0] = 0.;
                                    limits[3][1] = 2.*guess[3];
                                    if (finding.nTermsGaussFit > 4) {
                                        limits[4][0] = guess[4]/10.;
                                        limits[4][1] = guess[4]*10.;
                                    }
                                }
                                #if defined(__DEBUG_FINDANDTRACE__)
                                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                    ": while: 2. starting MPFitGaussLim: D_A2_Limits = " << limits << endl;
                                #endif
                                if (limits[1][0] > *std::max_element(x.begin(), x.end()) ||
                                    limits[1][1] < *std::min_element(x.begin(), x.end())) {
                                    string message("pfs::drp::stella::math::findCenterPositionsOneTrace: ERROR: (limits[1][0](=");
                                    message += to_string(limits[1][0]) + ") > max(x)(=";
                                    message += to_string(*std::max_element(x.begin(), x.end()));
                                    message += ")) || (limits[1][1](=";
                                    message += to_string(limits[1][1]) + ") < min(x)(=";
                                    message += to_string(*std::min_element(x.begin(), x.end())) + "))";
                                    throw LSST_EXCEPT(pexExcept::RuntimeError, message.c_str());
                                }
                                if (!MPFitGaussLim(x, y, yErr, guess, limited, limits, 0, false, gaussFitCoeffs,
                                                   gaussFitCoeffsErr)) {
                                    #if defined(__DEBUG_FINDANDTRACE__)
                                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                        ": Warning: GaussFit FAILED" << endl;
                                    #endif
                                    // Set signal to zero
                                    imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                                    ++apertureLost;
                                } else {
                                    gaussFitMean.push_back(gaussFitCoeffs[1]);
                                    gaussFitVariances.push_back(gaussFitCoeffs[2]);

                                    #if defined(__DEBUG_FINDANDTRACE__)
                                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                        ": D_A1_GaussFit_Coeffs = " << gaussFitCoeffs << endl;
                                    if (gaussFitCoeffs[0] < finding.saturationLevel/5.){
                                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                            ": WARNING: Signal less than 20% of saturation level" << endl;
                                    }
                                    if (gaussFitCoeffs[0] > finding.saturationLevel){
                                      cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                          ": WARNING: Signal appears to be saturated" << endl;
                                    }
                                    #endif
                                    if (gaussFitCoeffs[0] < finding.signalThreshold){
                                        #if defined(__DEBUG_FINDANDTRACE__)
                                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                            ": WARNING: peak = " << gaussFitCoeffs[1] << " lower than signalThreshold -> abandoning aperture" << endl;
                                        #endif
                                        // Set signal to zero
                                        imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                                        #if defined(__DEBUG_FINDANDTRACE__)
                                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                            ": 2. Signal set to zero from I_FirstWideSignalStart = " << firstWideSignalStart <<
                                            " to I_FirstWideSignalEnd = " << firstWideSignalEnd << endl;
                                        #endif
                                        ++apertureLost;
                                    } else if ((gaussFitCoeffs[1] < gaussFitCoeffsBak[1] - finding.maxOffset) ||
                                        (gaussFitCoeffs[1] > gaussFitCoeffsBak[1] + finding.maxOffset)) {
                                        #if defined(__DEBUG_FINDANDTRACE__)
                                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                            ": Warning: Center of Gaussian too far away from middle of signal -> abandoning aperture" << endl;
                                        #endif
                                        /// Set signal to zero
                                        imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                                        ++apertureLost;
                                    } else if ((gaussFitCoeffs[2] < finding.minSigma) ||
                                        (gaussFitCoeffs[2] > finding.maxSigma)) {
                                        #if defined(__DEBUG_FINDANDTRACE__)
                                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                            ": WARNING: Sigma = " << gaussFitCoeffs[2] << " outside range -> abandoning aperture" << endl;
                                        #endif
                                        /// Set signal to zero
                                        imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                                        #if defined(__DEBUG_FINDANDTRACE__)
                                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                            ": 2. Signal set to zero from I_FirstWideSignalStart = " << firstWideSignalStart <<
                                            " to I_FirstWideSignalEnd = " << firstWideSignalEnd << endl;
                                        #endif
                                        ++apertureLost;
                                    } else {
                                        apertureLost = 0;
                                        apertureFound = true;
                                        apertureCenter[row] = gaussFitCoeffs[1];
                                        apertureCenterErr[row] = gaussFitCoeffsErr[1];
                                        #if defined(__DEBUG_FINDANDTRACE__)
                                        cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                            ": Aperture found at " << apertureCenter[row] << endl;
                                        #endif
                                        // XXX Most of the rest of this block appears vestigial:
                                        // populating data structures that aren't used.
                                        gaussFitCoeffsBak.deep() = gaussFitCoeffs;
                                        std::size_t const xCorMinPos = 0.;
                                        std::size_t const xySize = fitSize + 2;
                                        ndarray::Array<float, 2, 1> xyRelativeToCenter = ndarray::allocate(xySize, 2);
                                        xyRelativeToCenter[0][0] = x[0] - gaussFitCoeffs[1] - 1.;
                                        xyRelativeToCenter[0][1] = 0.;
                                        xyRelativeToCenter[xySize - 1][0] = x[fitSize - 1] - gaussFitCoeffs[1] + 1.;
                                        xyRelativeToCenter[xySize - 1][1] = 0.;
                                        xyRelativeToCenter[ndarray::view(1, xySize - 1)(0)] = x - gaussFitCoeffs[1];
                                        xyRelativeToCenter[ndarray::view(1, xySize - 1)(1)] = y;

                                        // Oversampled Gaussian
                                        indGaussArr[ndarray::view()(0)] = xyRelativeToCenter[0][0];
                                        std::size_t ind = 0;
                                        float const factor = (xyRelativeToCenter[xySize - 1][0] - xyRelativeToCenter[0][0])/nInd;
                                        for (auto iter = indGaussArr.begin(); iter != indGaussArr.end(); ++iter, ++ind) {
                                            float const value = (*iter)[0] + ind*factor;
                                            (*iter)[0] = value;
                                            (*iter)[1] = gaussFitCoeffs[0] * exp(-(value*value)/(2.*guess[2]*guess[2]));
                                        }
                                        if (gaussFitVariances.size() > 20) {
                                            ndarray::Array<float, 2, 1> xysRelativeToCenter = ndarray::allocate(xySorted.size(), 2);
                                            auto itSorted = xySorted.begin();
                                            for (auto iter = xysRelativeToCenter.begin(); iter != xysRelativeToCenter.end();
                                                ++iter, ++itSorted) {
                                                (*iter)[0] = itSorted->getX();
                                                (*iter)[1] = itSorted->getY();
                                            }
                                            #if defined(__DEBUG_FINDANDTRACE__)
                                            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                                ": xCorMinPos = " << xCorMinPos << endl;
                                            #endif
                                        }
                                        if (gaussFitVariances.size() > 10) {
                                            for (auto iter = xyRelativeToCenter.begin(); iter != xyRelativeToCenter.end(); ++iter) {
                                                xySorted.insert(lsst::geom::Point2D((*iter)[0] + xCorMinPos, (*iter)[1]));
                                            }
                                            apertureCenter[row] = apertureCenter[row] + xCorMinPos;
                                            #if defined(__DEBUG_FINDANDTRACE__)
                                            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                                                ": Aperture position corrected to " << apertureCenter[row] << endl;
                                            #endif
                                        }
                                    } // end else if (D_A1_GaussFit_Coeffs(0) >= signalThreshold
                                } // end else if (GaussFit(D_A1_X, D_A1_Y, D_A1_GaussFit_Coeffs, S_A1_KeyWords_GaussFit, PP_Args_GaussFit))
                                imageArray[ndarray::view(row)(firstWideSignalStart + 1, firstWideSignalEnd)] = 0.;
                            } // end else if (sum(I_A1_Signal) >= I_MinWidth){
                        } // end if (I_Length > 3)
                    } // end else if (I_ApertureStart >= 0. && I_ApertureEnd < ccdArray.getShape()[1])
                } // end else if GaussFit
                if (apertureLost == finding.nLost && apertureLength < finding.minLength) {
                    row = rowBak;
                    #if defined(__DEBUG_FINDANDTRACE__)
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row set to " << row << endl;
                    #endif
                }
            } //end while(B_ApertureFound && (I_ApertureLost < 3) && i_Row < ccdArray.getShape()[0] - 2))

            /// Fit Polynomial to traced aperture positions
            #if defined(__DEBUG_FINDANDTRACE__) && __DEBUG_FINDANDTRACE__ > 2
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                ": D_A1_ApertureCenter = " << apertureCenter << endl;
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                ": I_A1_ApertureCenterIndex.getShape() = " << apertureCenterIndex.getShape() << endl;
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                ": D_A1_ApertureCenter.getShape() = " << apertureCenter.getShape() << endl;
            cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                ": D_A1_EApertureCenter.getShape() = " << apertureCenterErr.getShape() << endl;
            #endif

            auto itInd = apertureCenterIndex.begin();
            auto itCen = apertureCenter.begin();
            auto itECen = apertureCenterErr.begin();
            apertureLength = 0;
            result.index.resize(0);
            result.position.resize(0);
            result.error.resize(0);
            result.nextSearchStart = nextSearchStart;
            for (std::size_t iInd = 0; iInd < height; ++iInd) {
                if (*(itCen + iInd) > 0.) {
                    (*(itInd + iInd)) = 0;
                    ++apertureLength;
                    result.index.push_back(iInd);
                    result.position.push_back((*(itCen + iInd)));
                    result.error.push_back((*(itECen + iInd)));
                    #if defined(__DEBUG_FINDANDTRACE__)
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                        ": result.apertureCenterIndex[" << result.index.size() - 1 <<
                        "] set to " << result.index[result.index.size() - 1] << endl;
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                        ": result.apertureCenterPos[" << result.position.size() - 1 <<
                        "] set to " << result.position[result.position.size() - 1] << endl;
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                        ": result.eApertureCenterPos[" << result.error.size() - 1 <<
                        "] set to " << result.error[result.error.size() - 1] << endl;
                    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                        ": result.nextSearchStart set to " << result.nextSearchStart << endl;
                    #endif
                }
            }
            if (apertureLength > finding.minLength){
                #if defined(__DEBUG_FINDANDTRACE__)
                cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: i_Row = " << row <<
                    ": result.apertureCenterIndex.size() = " << result.index.size() << endl;
                #endif
                return result;
            }
        } // if aperture found
    } // for each row
    #if defined(__DEBUG_FINDANDTRACE__)
    cout << "pfs::drp::stella::math::findCenterPositionsOneTrace: result.apertureCenterIndex.size() = " <<
        result.index.size() << endl;
    #endif
    return result;
}

// Explicit instantiations
template FiberTraceSet<float, lsst::afw::image::MaskPixel, float> findAndTraceApertures(
    afwImage::MaskedImage<float, lsst::afw::image::MaskPixel, float> const&,
    DetectorMap const&,
    FiberTraceFindingControl const&,
    FiberTraceFunctionControl const&,
    FiberTraceProfileFittingControl const&
);

template FindCenterPositionsOneTraceResult findCenterPositionsOneTrace(
    afwImage::Image<float> &,
    afwImage::Image<float> const&,
    FiberTraceFindingControl const&,
    lsst::geom::Point<int> const&
);

}}}} // namespace pfs::drp::stella::math
