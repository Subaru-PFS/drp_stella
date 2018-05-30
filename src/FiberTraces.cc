#include <algorithm>
#include "ndarray/eigen.h"

#include "lsst/log/Log.h"
#include "lsst/pex/exceptions/Exception.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Box.h"

#include "pfs/drp/stella/FiberTraces.h"
#include "pfs/drp/stella/math/findAndTraceApertures.h"
#include "pfs/drp/stella/cmpfit-1.2/MPFitting_ndarray.h"
#include "pfs/drp/stella/math/Chebyshev.h"
#include "pfs/drp/stella/math/CurveFitting.h"
#include "pfs/drp/stella/spline.h"
#include "pfs/drp/stella/utils/Utils.h"
#include "pfs/drp/stella/utils/checkSize.h"

//#define __DEBUG_FINDANDTRACE__ 1
namespace afwImage = lsst::afw::image;

namespace pfs { namespace drp { namespace stella {

template<typename ImageT, typename MaskT, typename VarianceT>
FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    MaskedImageT const& maskedImage,
    std::size_t fiberTraceId
) : _trace(maskedImage),
    _xCenters(ndarray::allocate(0)),
    _fiberId(fiberTraceId)
    {}


template<typename ImageT, typename MaskT, typename VarianceT>
FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    afwImage::MaskedImage<ImageT, MaskT, VarianceT> const & maskedImage,
    FiberTraceFunction const& function,
    FiberTraceProfileFittingControl const& fitting,
    size_t fiberId
) :
    // XXX _trace is set in _calcProfile
    //_trace(function.yHigh - function.yLow + 1,
    //       static_cast<int>(function.ctrl.xHigh - function.ctrl.xLow + 1)),
    _fiberId(fiberId),
    _function(function),
    _fitting(fitting)
{
    _function.ctrl.nRows = maskedImage.getHeight();
    _xCenters = math::calculateXCenters(function);
    _createTrace(maskedImage);
    _calcProfile();
}


template<typename ImageT, typename MaskT, typename VarianceT>
FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace,
    bool deep
) : _trace(fiberTrace.getTrace(), deep),
    _xCenters(ndarray::copy(fiberTrace.getXCenters())),
    _fiberId(fiberTrace.getFiberId()),
    _function(fiberTrace.getFunction()),
    _fitting(fiberTrace.getFitting())
    {}


template<typename ImageT, typename MaskT, typename VarianceT>
PTR(Spectrum)
FiberTrace<ImageT, MaskT, VarianceT>::extractSpectrum(
    MaskedImageT const& spectrumImage,
    bool fitBackground,
    float clipNSigma,
    bool useProfile
) {
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.FiberTrace.extractFromProfile");
    auto const bbox = _trace.getBBox();
    MaskedImageT traceIm(spectrumImage, bbox);
    std::size_t const height = bbox.getHeight();
    std::size_t const width = bbox.getWidth();

    auto spectrum = std::make_shared<Spectrum>(spectrumImage.getHeight(), _fiberId);
    auto slice = ndarray::view(bbox.getMinY(), bbox.getMaxY());  // slice of trace on spectrum

    MaskT const ftMask = _trace.getMask()->getPlaneBitMask(maskPlane);
    MaskT const noData = _trace.getMask()->getPlaneBitMask("NO_DATA");

    // Select pixels for extraction
    ndarray::Array<bool, 2, 1> select{height, width};  // select this pixel for extraction?
    {
        auto maskRow = _trace.getMask()->getArray().begin();
        for (auto selectRow = select.begin(); selectRow != select.end(); ++selectRow, ++maskRow) {
            auto maskIter = maskRow->begin();
            for (auto selectIter = selectRow->begin(); selectIter != selectRow->end();
                 ++selectIter, ++maskIter) {
                *selectIter = (*maskIter & ftMask) > 0;
            }
        }
    }

    // Extract
    if (useProfile) {
        auto const result = math::fitProfile2d(traceIm, select, _trace.getImage()->getArray(), fitBackground,
                                               clipNSigma);
        spectrum->getSpectrum()[slice] = ndarray::copy(std::get<0>(result));
        spectrum->getBackground()[slice] = ndarray::copy(std::get<1>(result));
        spectrum->getVariance()[slice] = ndarray::copy(std::get<2>(result));
    } else {                            // simple profile fit
        auto specIt = spectrum->getSpectrum().begin();
        auto varIt = spectrum->getVariance().begin();
        auto itSelectRow = select.begin();
        auto itTraceRow = traceIm.getImage()->getArray().begin();
        auto itMaskRow = traceIm.getMask()->getArray().begin();
        auto itVarRow = traceIm.getVariance()->getArray().begin();
        for (std::size_t y = 0; y < height; ++y, ++specIt, ++varIt, ++itSelectRow, ++itTraceRow, ++itVarRow) {
            *specIt = 0.0;
            *varIt = 0.0;
            auto itTraceCol = itTraceRow->begin();
            auto itMaskCol = itMaskRow->begin();
            auto itVarCol = itVarRow->begin();
            for (auto itSelectCol = itSelectRow->begin(); itSelectCol != itSelectRow->end();
                 ++itMaskCol, ++itTraceCol, ++itVarCol) {
                if (*itSelectCol) {
                    *specIt += *itTraceCol;
                    *varIt += *itVarCol;
                }
            }
        }
    }

    // Fix the ends of the spectrum
    auto spec = spectrum->getSpectrum();
    auto bg = spectrum->getBackground();
    auto var = spectrum->getVariance();
    auto & mask = spectrum->getMask();

    int const yMin = bbox.getMinY();
    int const yMax = bbox.getMaxY();
    float const inf = std::numeric_limits<float>::infinity();

    std::fill(spec.begin(), spec.begin() + yMin, 0.0);
    std::fill(bg.begin(), bg.begin() + yMin, 0.0);
    std::fill(var.begin(), var.begin() + yMin, inf);
    std::fill(mask.begin(true), mask.begin(true) + yMin, noData);

    std::fill(spec.begin() + yMax, spec.end(), 0.0);
    std::fill(bg.begin() + yMax, bg.end(), 0.0);
    std::fill(var.begin() + yMax, var.end(), inf);
    std::fill(mask.begin(true) + yMax, mask.end(true), noData);

    // Accumulate the mask
    auto const& traceMask = *_trace.getMask();
    auto const& spectrumMask = *spectrumImage.getMask();
    for (int yImage = 0, ySpec = bbox.getMinY(); yImage < bbox.getHeight(); ++yImage, ++ySpec) {
        MaskT value = 0;
        for (int xImage = 0, xSpec = bbox.getMinX(); xImage < bbox.getWidth(); ++xImage, ++xSpec) {
            if (traceMask(xImage, yImage) & ftMask) {
                value |= spectrumMask(xSpec, ySpec);
            }
        }
        mask(ySpec, 0) |= value;  // not a typo: it's 1D
    }

    return spectrum;
}


template<typename ImageT, typename MaskT, typename VarianceT>
void FiberTrace<ImageT, MaskT, VarianceT>::_createTrace(MaskedImageT const& maskedImage) {
    _minCenMax = _function.calcMinCenMax(_xCenters);

    ndarray::Array<std::size_t, 1, 1> xMinPix(_minCenMax[ndarray::view()(0)]);
    int xMin = *std::min_element(xMinPix.begin(), xMinPix.end());

    ndarray::Array<std::size_t, 1, 1> xMaxPix(_minCenMax[ndarray::view()(2)]);
    int xMax = *std::max_element(xMaxPix.begin(), xMaxPix.end());

    lsst::afw::geom::Point<int> lowerLeft(xMin, _function.yCenter + _function.yLow);
    lsst::afw::geom::Extent<int, 2> extent(xMax - xMin + 1, _function.yHigh - _function.yLow + 1);
    lsst::afw::geom::Box2I box(lowerLeft, extent);
    _trace = MaskedImageT(maskedImage, box, lsst::afw::image::PARENT, true);

    /// mark FiberTrace in Mask
    _minCenMax.deep() -= static_cast<size_t>(xMin);
    _trace.getMask()->addMaskPlane(maskPlane);
    const auto ftMask = _trace.getMask()->getPlaneBitMask(maskPlane);
    _markFiberTraceInMask(ftMask);
}


template<typename ImageT, typename MaskT, typename VarianceT>
PTR(afwImage::Image<ImageT>)
FiberTrace<ImageT, MaskT, VarianceT>::constructImage(const Spectrum & spectrum) const
{
    auto out = std::make_shared<afwImage::Image<ImageT>>(_trace.getBBox());
    auto const maskVal = _trace.getMask()->getPlaneBitMask(maskPlane);

    int const height = _trace.getHeight();
    int const width  = _trace.getImage()->getWidth();

    auto spec = spectrum.getSpectrum().begin();
    auto bg = spectrum.getBackground().begin();
    for (int y = 0; y < height; ++y, ++spec, ++bg) {
        auto profileIter = _trace.getImage()->row_begin(y);
        auto maskIter = _trace.getMask()->row_begin(y);
        auto outIter = out->row_begin(y);
        float const bgValue = *bg;
        float const specValue = *spec;
        for (int x = 0; x < width; ++x, ++profileIter, ++maskIter, ++outIter) {
            if (*maskIter & maskVal) {
                *outIter = bgValue + specValue*(*profileIter);
            }
        }
    }

    return out;
}


template<typename ImageT, typename MaskT, typename VarianceT>
ndarray::Array<std::size_t, 2, 1>
FiberTrace<ImageT, MaskT, VarianceT>::_calcSwathBoundY(std::size_t swathWidth) const {
    std::size_t const height = _trace.getImage()->getHeight();

    std::size_t nSwaths = std::max(round(double(height)/swathWidth), 1.0);
    std::size_t binHeight = height/nSwaths;
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


template<typename ImageT, typename MaskT, typename VarianceT>
void FiberTrace<ImageT, MaskT, VarianceT>::_calcProfile() {
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.FiberTrace._calcProfile");

    unsigned int const nCols = _minCenMax[0][2] - _minCenMax[0][0] + 1;
    std::size_t const height = _trace.getImage()->getHeight();
    auto traceArray = _trace.getImage()->getArray();

    /// Calculate boundaries for swaths
    ndarray::Array<std::size_t, 2, 1> swathBoundsY = _calcSwathBoundY(_fitting.swathWidth);
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": swathBoundsY = " << swathBoundsY);
    std::size_t const nSwaths = swathBoundsY.getShape()[0];

    ndarray::Array<std::size_t, 1, 1> nPixArr = ndarray::allocate(nSwaths);
    nPixArr.deep() = swathBoundsY[ndarray::view()(1)] - swathBoundsY[ndarray::view()(0)] + 1;
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": nPixArr = " << nPixArr);
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": nSwaths = " << nSwaths);
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": height = " << height);

    /// for each swath
    ndarray::Array<float, 3, 2> slitFuncsSwaths = ndarray::allocate(nPixArr[0], nCols, nSwaths - 1);
    ndarray::Array<float, 2, 1> lastSlitFuncSwath = ndarray::allocate(nPixArr[nSwaths - 1], nCols);
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths.getShape() = "
                      << slitFuncsSwaths.getShape());
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths.getShape()[0] = "
                      << slitFuncsSwaths.getShape()[0] << ", slitFuncsSwaths.getShape()[1] = "
                      << slitFuncsSwaths.getShape()[1] << ", slitFuncsSwaths.getShape()[2] = "
                      << slitFuncsSwaths.getShape()[2]);
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths[view(0)()(0) = "
                      << slitFuncsSwaths[ndarray::view(0)()(0)]);
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths[view(0)(0,9)(0) = "
                      << slitFuncsSwaths[ndarray::view(0)(0,slitFuncsSwaths.getShape()[1])(0)]);

    _overSampledProfileFitXPerSwath.resize(0);
    _overSampledProfileFitYPerSwath.resize(0);
    _profileFittingInputXPerSwath.resize(0);
    _profileFittingInputYPerSwath.resize(0);
    _profileFittingInputXMeanPerSwath.resize(0);
    _profileFittingInputYMeanPerSwath.resize(0);

    lsst::afw::image::Image<ImageT> profile(nCols, height);
    profile = 0.0;

    for (unsigned int iSwath = 0; iSwath < nSwaths; ++iSwath){
        std::size_t yMin = swathBoundsY[iSwath][0];
        std::size_t yMax = swathBoundsY[iSwath][1] + 1;
        LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": yMin = "
                          << yMin << ", yMax = " << yMax);

        unsigned int nRows = yMax - yMin;
        lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> swath{nCols, nRows};
        auto imageSwath = swath.getImage()->getArray();
        auto maskSwath = swath.getMask()->getArray();
        auto varianceSwath = swath.getVariance()->getArray();
        ndarray::Array<float, 1, 1> xCentersSwath = ndarray::copy(_xCenters[ndarray::view(yMin, yMax)]);
        for (unsigned int iRow = 0; iRow < nRows; ++iRow) {
            auto const fromSlice = ndarray::view(yMin + iRow)(_minCenMax[iRow + yMin][0],
                                                              _minCenMax[iRow + yMin][2] + 1);
            auto const toSlice = ndarray::view(iRow)();
            imageSwath[toSlice] = _trace.getImage()->getArray()[fromSlice];
            maskSwath[toSlice] = _trace.getMask()->getArray()[fromSlice];
            varianceSwath[toSlice] = _trace.getVariance()->getArray()[fromSlice];
        }
        LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": swath " << iSwath << ": imageSwath = " << imageSwath);

        if (iSwath < nSwaths - 1) {
            auto const slice = ndarray::view()()(iSwath);
            slitFuncsSwaths[slice] = _calcProfileSwath(swath, xCentersSwath, iSwath);
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths.getShape() = "
                              << slitFuncsSwaths.getShape());
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": imageSwath.getShape() = " << imageSwath.getShape());
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": xCentersSwath.getShape() = "
                              << xCentersSwath.getShape());
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": swathBoundsY = " << swathBoundsY);
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": nPixArr = " << nPixArr);
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": swath " << iSwath
                              << ": slitFuncsSwaths[ndarray::view()()(iSwath)] = "
                              << slitFuncsSwaths[slice]);
        } else {
            lastSlitFuncSwath.deep() = _calcProfileSwath(swath, xCentersSwath, iSwath);
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
            auto const slice = ndarray::view(iRow)()(iSwath);
            rowSum = ndarray::sum(slitFuncsSwaths[slice]);
            if (std::fabs(rowSum) > 0.00000000000000001) {
                for (std::size_t iPix = 0; iPix < slitFuncsSwaths.getShape()[1]; iPix++) {
                    slitFuncsSwaths[iRow][iPix][iSwath] = slitFuncsSwaths[iRow][iPix][iSwath]/rowSum;
                }
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths(" << iRow << ", *, "
                                  << iSwath << ") = "
                                  << slitFuncsSwaths[slice]);
                  rowSum = ndarray::sum(slitFuncsSwaths[slice]);
                  LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow << ": iSwath = "
                                    << iSwath << ": rowSum = " << rowSum);
            }
        }
    }
    for (std::size_t iRow = 0; iRow < nPixArr[nSwaths - 1]; ++iRow) {
        auto const slice = ndarray::view(iRow)();
        rowSum = ndarray::sum(lastSlitFuncSwath[slice]);
        if (std::fabs(rowSum) > 0.00000000000000001) {
            lastSlitFuncSwath[ndarray::view(iRow)()] = lastSlitFuncSwath[slice]/rowSum;
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": lastSlitFuncSwath(" << iRow << ", *) = "
                              << lastSlitFuncSwath[slice]);
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow << ": rowSum = " << rowSum);
        }
    }
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": swathBoundsY.getShape() = "
                      << swathBoundsY.getShape() << ", nSwaths = " << nSwaths);
    int iRowSwath = 0;
    for (std::size_t iRow = 0; iRow < height; ++iRow) {
        iRowSwath = iRow - swathBoundsY[bin][0];
        auto const rowSlice = ndarray::view(iRow)();
        auto const swathSlice = ndarray::view(iRowSwath)();
        LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow << ", bin = "
                          << bin << ", iRowSwath = " << iRowSwath);
        if ((bin == 0) && (iRow < swathBoundsY[1][0])) {
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": bin=" << bin << " == 0 && iRow="
                              << iRow << " < swathBoundsY[1][0]=" << swathBoundsY[1][0]);
            profileArray[rowSlice] = slitFuncsSwaths[swathSlice(0)];
        } else if ((bin == nSwaths - 1) && (iRow >= swathBoundsY[bin-1][1])) {
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": bin=" << bin << " == nSwaths-1="
                              << nSwaths-1 << " && iRow=" << iRow << " >= swathBoundsY[bin-1="
                              << bin - 1 << "][0]=" << swathBoundsY[bin - 1][0]);
            profileArray[rowSlice] = lastSlitFuncSwath[swathSlice];
        } else {
            weightBin1 = float(iRow - swathBoundsY[bin + 1][0])/
                (swathBoundsY[bin][1] - swathBoundsY[bin + 1][0]);
            weightBin0 = 1. - weightBin1;
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow << ": nSwaths = " << nSwaths);
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow << ": bin = " << bin);
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow <<
                ": swathBoundsY(bin, *) = " << swathBoundsY[ndarray::view(bin)()]);
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow <<
                ": swathBoundsY(bin+1, *) = " << swathBoundsY[ndarray::view(bin + 1)()]);
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow << ": weightBin0 = "
                              << weightBin0);
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow << ": weightBin1 = "
                              << weightBin1);
            if (bin == nSwaths - 2){
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths(iRowSwath, *, bin) = " <<
                    slitFuncsSwaths[swathSlice(bin)] <<
                    ", lastSlitFuncSwath[ndarray::view(int(iRow - swathBoundsY[bin+1][0])=" <<
                    int(iRow - swathBoundsY[bin+1][0]) << ")] = " <<
                    lastSlitFuncSwath[ndarray::view(int(iRow - swathBoundsY[bin + 1][0]))()]);
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": profile.getArray().getShape() = "
                                  << profileArray.getShape());
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": slitFuncsSwath.getShape() = "
                                  << slitFuncsSwaths.getShape());
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": lastSlitFuncSwath.getShape() = "
                                  << lastSlitFuncSwath.getShape());
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow << ", iRowSwath = "
                                  << iRowSwath << ", bin = " << bin << ", swathBoundsY[bin+1][0] = "
                                  << swathBoundsY[bin+1][0] << ", iRow - swathBoundsY[bin+1][0] = "
                                  << iRow - swathBoundsY[bin + 1][0]);
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId <<
                    ": profile.getArray()[ndarray::view(iRow)()] = " << profileArray[rowSlice]);
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId
                                  << ": slitFuncsSwaths[ndarray::view(iRowSwath)()(bin)] = "
                                  << slitFuncsSwaths[ndarray::view(iRowSwath)()(bin)]);
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId <<
                    ": ndarray::Array<float, 1, 0>(slitFuncsSwaths[ndarray::view(iRowSwath)()" <<
                    "(bin)]).getShape() = " << slitFuncsSwaths[swathSlice(bin)].getShape());
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": weightBin0 = " << weightBin0
                                  << ", weightBin1 = " << weightBin1);
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId <<
                    ": lastSlitFuncSwath[ndarray::view(int(iRow - swathBoundsY[bin+1][0]))()] = " <<
                    lastSlitFuncSwath[ndarray::view(int(iRow - swathBoundsY[bin + 1][0]))()]);
                profileArray[rowSlice] = (slitFuncsSwaths[swathSlice(bin)]*weightBin0) +
                    (lastSlitFuncSwath[ndarray::view(int(iRow - swathBoundsY[bin + 1][0]))()]*weightBin1);
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId
                                  << ": profile.getArray()[ndarray::view(iRow)()] set to "
                                  << profileArray[rowSlice]);
            } else {
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": slitFuncsSwaths(iRowSwath, *, bin) = "
                                  << slitFuncsSwaths[swathSlice(bin)]
                                  << ", slitFuncsSwaths(int(iRow - swathBoundsY[bin+1][0])="
                                  << int(iRow - swathBoundsY[bin + 1][0]) << ", *, bin+1) = "
                                  << slitFuncsSwaths[ndarray::view(
                                     int(iRow - swathBoundsY[bin + 1][0]))()(bin + 1)]);
                profileArray[rowSlice] = (slitFuncsSwaths[swathSlice(bin)]*weightBin0) +
                    (slitFuncsSwaths[ndarray::view(int(iRow - swathBoundsY[bin + 1][0]))()(bin + 1)]*weightBin1);
            }
            double const dSumSFRow = ndarray::sum(profileArray[rowSlice]);
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow << ": bin = "
                              << bin << ": dSumSFRow = " << dSumSFRow);
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow << ": bin = "
                              << bin << ": profile.getArray().getShape() = "
                              << profileArray.getShape());
            if (std::fabs(dSumSFRow) >= 0.000001){
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow << ": bin = "
                                  << bin << ": normalizing profile.getArray()[iRow = "
                                  << iRow << ", *]");
                profileArray[rowSlice] /= dSumSFRow;
            }
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow << ": bin = "
                              << bin << ": profile.getArray()(" << iRow << ", *) set to "
                              << profileArray[rowSlice]);
        }

        if (iRow == swathBoundsY[bin][1]){
            bin++;
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iRow = " << iRow << ": bin set to " << bin);
        }
    } // end for (int i_row = 0; i_row < slitFuncsSwaths.rows(); i_row++) {
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": profile.getArray() set to ["
                      << profile.getHeight() << ", " << profile.getWidth() << "]: "
                      << profileArray);

    std::size_t const width = profile.getWidth();
    traceArray.deep() = 0.0;
    for (std::size_t i = 0; i < _minCenMax.getShape()[0]; ++i) {
        std::size_t const xMax = _minCenMax[i][2] + 1;
        std::size_t const xMin = _minCenMax[i][0];
        if (xMax - xMin != width){
            std::string message("_minCenMax[");
            message += std::to_string(i) + "][2](=" + std::to_string(_minCenMax[i][2]);
            message += ") - _minCenMax[" + std::to_string(i) + "][0](=";
            message += std::to_string(_minCenMax[i][0]) + ") + 1 = ";
            message += std::to_string(_minCenMax[i][2] - _minCenMax[i][0] + 1) +" != width(=";
            message += std::to_string(width);
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        auto const slice = ndarray::view(i);
        traceArray[slice(xMin, xMax)] = profileArray[slice()];
    }
}


template<typename ImageT, typename MaskT, typename VarianceT>
ndarray::Array<float, 2, 1> FiberTrace<ImageT, MaskT, VarianceT>::_calcProfileSwath(
    lsst::afw::image::MaskedImage<ImageT, MaskT, VarianceT> const& swath,
    ndarray::Array<float const, 1, 1> const& xCentersSwath,
    size_t iSwath
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
        auto const slice = ndarray::view(iRow)();
        imageSwathNormalized[slice] = imageSwath[slice]/ndarray::sum(sumArr);
    }
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath <<
        ": imageSwath = " << imageSwath);
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath <<
        ": imageSwathNormalized = " << imageSwathNormalized);

    // Calculate pixel offset to xCenter
    ndarray::Array<float, 1, 1> xCentersTemp = ndarray::allocate(height);
    xCentersTemp.deep() = xCentersSwath + 0.5;
    ndarray::Array<std::size_t const, 1, 1> const xCentersInt = math::floor<std::size_t>(xCentersTemp);
    ndarray::Array<float, 1, 1> pixelOffset = ndarray::allocate(height);
    pixelOffset.deep() = xCentersInt - xCentersSwath;
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath <<
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
    auto xVec = std::make_shared<std::vector<float>>();
    xVec->reserve(xArray.getNumElements());
    auto yVec = std::make_shared<std::vector<float>>();
    yVec->reserve(xArray.getNumElements());
    auto itRowIm = imageSwathNormalized.begin();
    for (auto itRow = xArray.begin(); itRow != xArray.end(); ++itRow, ++itRowIm) {
        auto itColIm = itRowIm->begin();
        for (auto itCol = itRow->begin(); itCol != itRow->end(); ++itCol, ++itColIm) {
            xVec->push_back(*itCol);
            yVec->push_back(*itColIm);
        }
    }
    _profileFittingInputXPerSwath.push_back(xVec);
    _profileFittingInputYPerSwath.push_back(yVec);
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": xArray = " << xArray);
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": xMin = " << xMin
                      << ", xMax = " << xMax);
    double xOverSampleStep = 1./_fitting.overSample;
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": initial xOverSampleStep = "
                      << xOverSampleStep);

    // adjust xOverSampleStep to cover x from xMin + xOverSampleStep/2 to xMax - xOverSampleStep/2
    int const nSteps = (xMax - xMin)/xOverSampleStep + 1;
    xOverSampleStep = (xMax - xMin)/nSteps;
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": final xOverSampleStep = "
                      << xOverSampleStep);
    ndarray::Array<float, 1, 1> xOverSampled = ndarray::allocate(nSteps);
    double const xStart = xMin + 0.5*xOverSampleStep;
    {
        int iStep = 0;
        for (auto it = xOverSampled.begin(); it != xOverSampled.end(); ++it, ++iStep) {
            *it = xStart + (iStep * xOverSampleStep);
        }
    }
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath << ": xOverSampled = "
                      << xOverSampled);
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                      << ": _fiberTraceProfileFittingControl->maxIterSig = "
                      << _fitting.maxIterSig);
    auto xOverSampledFit = std::make_shared<std::vector<float>>(xOverSampled.begin(), xOverSampled.end());
    _overSampledProfileFitXPerSwath.push_back(xOverSampledFit);

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
        LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                          << ": iStep" << iStep << ": indicesInValueRange = " << indices);
        for (unsigned int iterSig = 0; iterSig <= std::max(_fitting.maxIterSig, 1U); ++iterSig) {
            ndarray::Array<float, 1, 1> subArr = math::getSubArray(imageSwathNormalized, indices);
            ndarray::Array<float, 1, 1> xSubArr = math::getSubArray(xArray, indices);
            std::size_t const nValues = subArr.getShape()[0];
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                              << ": iStep" << iStep << ": iterSig = " << iterSig << ": nValues = " << nValues);
            if (nValues > 1) {
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                                  << ": iStep = " << iStep << ": iterSig = " << iterSig << ": xSubArr = ["
                                  << xSubArr.getShape()[0] << "]: " << xSubArr);
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                                  << ": iStep = " << iStep << ": iterSig = " << iterSig << ": subArr = ["
                                  << subArr.getShape()[0] << "]: " << subArr);
                if (_fitting.maxIterSig > iterSig) {
                    ndarray::Array<float, 1, 1> moments = math::moment(subArr, 2);
                    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                                      << ": iStep = " << iStep << ": iterSig = " << iterSig << ": moments = "
                                      << moments);
                    float mom = std::sqrt(moments[1]);
                    for (int i = nValues - 1; i >= 0; --i) {
                        LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath <<
                            ": iStep" << iStep << ": iterSig = " << iterSig << ": moments[0](=" <<
                            moments[0] << ") - subArr[" << i << "](=" << subArr[i] << ") = " <<
                            moments[0] - subArr[i] <<
                            ", 0. - (_fiberTraceProfileFittingControl->upperSigma(=" <<
                            _fitting.upperSigma << ") * sqrt(moments[1](=" <<
                            moments[1] << "))(= " << mom << ") = " <<
                            0. - (_fitting.upperSigma*mom));
                        LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                                          << ": iStep" << iStep << ": iterSig = " << iterSig
                                          << ": _fiberTraceProfileFittingControl->lowerSigma(="
                                          << _fitting.lowerSigma << ") * sqrt(moments[1](="
                                          << moments[1] << "))(= " << mom << ") = "
                                          << _fitting.lowerSigma*mom);
                        if ((moments[0] - subArr[i] < 0. - (_fitting.upperSigma*mom)) ||
                            (moments[0] - subArr[i] > (_fitting.lowerSigma*mom))) {
                          LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                                            << ": iStep = " << iStep << ": iterSig = " << iterSig
                                            << ": rejecting element " << i << "from subArr");
                          indices.erase(indices.begin() + i);
                        }
                    }
                }
                ndarray::Array<float, 1, 1> moments = math::moment(subArr, 1);
                mean = moments[0];
                LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath <<
                    ": iStep = " << iStep << ": iterSig = " << iterSig << ": mean = " << mean);
                stepHasValues = true;
            }
        }
        if (stepHasValues) {
            valOverSampled.push_back(std::pair<float, float>(*it, mean));
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                              << ": valOverSampledVec[" << iStep << "] = (" << valOverSampled[iStep].first
                              << ", " << valOverSampled[iStep].second << ")");
        }
        rangeStart = rangeEnd;
        rangeEnd += xOverSampleStep;
    }

    auto xVecMean = std::make_shared<std::vector<float>>();
    std::transform(valOverSampled.begin(), valOverSampled.end(), xVecMean->begin(),
                   [](std::pair<float, float> const& values) { return values.first; });
    auto yVecMean = std::make_shared<std::vector<float>>();
    std::transform(valOverSampled.begin(), valOverSampled.end(), yVecMean->begin(),
                   [](std::pair<float, float> const& values) { return values.second; });
    _profileFittingInputXMeanPerSwath.push_back(xVecMean);
    _profileFittingInputYMeanPerSwath.push_back(yVecMean);

    math::Spline<float> spline(*xVecMean, *yVecMean, math::Spline<float>::CUBIC_NATURAL); // X must be sorted

    auto yOverSampledFit = std::make_shared<std::vector<float>>(nSteps);
    std::transform(xOverSampledFit->begin(), xOverSampledFit->end(), yOverSampledFit->begin(),
                   [&spline](float x) { return spline(x); });
    _overSampledProfileFitYPerSwath.push_back(yOverSampledFit);

    // calculate profile for each row in imageSwath
    ndarray::Array<float, 2, 1> profArraySwath = ndarray::allocate(height, width);
    for (int yy = 0; yy < height; ++yy) {
        auto const slice = ndarray::view(yy)();
        LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath <<
                    ": xArray[" << yy << "][*] = " << xArray[slice]);
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
        LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                          << ": profArraySwath[" << yy << "][*] = " << profArraySwath[slice]);
        profArraySwath[slice] = profArraySwath[slice]/ndarray::sum(profArraySwath[slice]);
        LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                          << ": normalized profArraySwath[" << yy << "][*] = "
                          << profArraySwath[slice]);
    }
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iSwath = " << iSwath
                      << ": profArraySwath = " << profArraySwath);
    return profArraySwath;
}


template<typename ImageT, typename MaskT, typename VarianceT>
void FiberTrace<ImageT, MaskT, VarianceT>::_markFiberTraceInMask(MaskT value) {
    /// Call getMinCenMax which will reconstruct _minCenMax in case it is empty
    ndarray::Array<size_t, 2, -2> minCenMax = _getMinCenMax();

    for (std::size_t y = 0; y < minCenMax.getShape()[0]; ++y) {
        for (std::size_t x = minCenMax[y][0]; x <= minCenMax[y][2]; ++x) {
            (*_trace.getMask())(x, y) |= value;
        }
    }
}


template<typename ImageT, typename MaskT, typename VarianceT>
ndarray::Array<size_t, 2, -2> FiberTrace<ImageT, MaskT, VarianceT>::_getMinCenMax() {
    if (_minCenMax.asEigen().maxCoeff() == 0) {
        _reconstructMinCenMax();
    }
    return _minCenMax;
}


template<typename ImageT, typename MaskT, typename VarianceT>
void FiberTrace<ImageT, MaskT, VarianceT>::_reconstructMinCenMax() {
    LOG_LOGGER _log = LOG_GET("pfs.drp.stella.FiberTrace._reconstructMinCenMax");
    _minCenMax = ndarray::allocate(_trace.getImage()->getHeight(), 3);
    auto xMin = _minCenMax[ndarray::view()(0)];
    auto xCen = _minCenMax[ndarray::view()(1)];
    auto xMax = _minCenMax[ndarray::view()(2)];

    auto const ftMask = _trace.getMask()->getPlaneBitMask(maskPlane);
    int yy = 0;
    auto itMaskRow = _trace.getMask()->getArray().begin();
    for (auto itMin = xMin.begin(), itMax = xMax.begin(), itCen = xCen.begin(); itMin != xMin.end();
         ++itMin, ++itMax, ++itCen, ++itMaskRow, ++yy) {
        LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": *itMaskRow = " << *itMaskRow);
        bool xMinFound = false;
        bool xMaxFound = false;
        int xx = 0;
        for (auto itMaskCol = itMaskRow->begin(); itMaskCol != itMaskRow->end(); ++itMaskCol, ++xx) {
            if (*itMaskCol & ftMask) {
                if (!xMinFound) {
                    *itMin = xx;
                    xMinFound = true;
                    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iY = " << yy << ", iX = " << xx << ": xMinFound");
                }
                *itMax = xx;
            }
        }
        if (*itMax > 0) {
            xMaxFound = true;
            LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": iY = " << yy << ", iX = " << xx << ": xMaxFound");
        }
        if (!xMinFound || !xMaxFound) {
            std::string message("_fiberId = ");
            message += to_string(_fiberId) + ": xMin or xMax not found";
            throw LSST_EXCEPT(pexExcept::Exception, message.c_str());
        }
        *itCen = std::size_t((*itMin + *itMax)/2);
    }
    LOGLS_TRACE(_log, "_fiberId = " << _fiberId << ": _minCenMax = " << _minCenMax);
}


template<typename ImageT, typename MaskT, typename VarianceT>
FiberTraceSet<ImageT, MaskT, VarianceT>::FiberTraceSet(
    FiberTraceSet<ImageT, MaskT, VarianceT> const &other,
    bool const deep
) : _traces(other.getInternal()) {
    if (deep) {
        // Replace entries in the collection with copies
        for (auto tt : _traces) {
            tt = std::make_shared<FiberTraceT>(*tt, true);
        }
    }
}


template<typename ImageT, typename MaskT, typename VarianceT>
void FiberTraceSet<ImageT, MaskT, VarianceT >::sortTracesByXCenter()
{
    std::size_t num = _traces.size();
    std::vector<float> xCenters;
    xCenters.reserve(num);
    std::transform(_traces.begin(), _traces.end(), xCenters.begin(),
                   [](std::shared_ptr<FiberTraceT> ft) {
                       auto const& box = ft->getTrace().getBBox();
                       return 0.5*(box.getMinX() + box.getMaxX()); });
    std::vector<std::size_t> indices = math::sortIndices(xCenters);

    Collection sorted;
    sorted.reserve(num);
    std::transform(indices.begin(), indices.end(), sorted.begin(),
                   [this](std::size_t ii) { return _traces[ii]; });
    _traces = std::move(sorted);
}


// Explicit instantiation
template class FiberTrace<float, lsst::afw::image::MaskPixel, float>;
template class FiberTraceSet<float, lsst::afw::image::MaskPixel, float>;

}}}
