#include <algorithm>

#include "lsst/log/Log.h"

#include "pfs/drp/stella/FiberTraces.h"
#include "pfs/drp/stella/math/CurveFitting.h"

//#define __DEBUG_FINDANDTRACE__ 1

namespace afwImage = lsst::afw::image;

namespace pfs { namespace drp { namespace stella {

template<typename ImageT, typename MaskT, typename VarianceT>
FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    MaskedImageT const& trace,
    std::size_t fiberId
) : _trace(trace),
    _fiberId(fiberId)
    {}


template<typename ImageT, typename MaskT, typename VarianceT>
FiberTrace<ImageT, MaskT, VarianceT>::FiberTrace(
    FiberTrace<ImageT, MaskT, VarianceT> const& fiberTrace,
    bool deep
) : _trace(fiberTrace.getTrace(), deep),
    _fiberId(fiberTrace.getFiberId())
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

    MaskT const ftMask = _trace.getMask()->getPlaneBitMask(fiberMaskPlane);
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
        spectrum->getSpectrum()[ndarray::view(bbox.getMinY(), bbox.getMaxY() + 1)] = std::get<0>(result);
        // Non-finite values can result from attempting to extract a row which is mostly bad.
        auto const badSpectrum = spectrum->getMask().getPlaneBitMask("BAD");
        auto const extracted = spectrum->getSpectrum();
        for (std::size_t y = bbox.getMinY(); y <= std::size_t(bbox.getMaxY()); ++y) {
            if (!std::isfinite(extracted[y])) {
                spectrum->getMask().getArray()[0][y] = badSpectrum;
            }
        }
        spectrum->getBackground()[ndarray::view(bbox.getMinY(), bbox.getMaxY() + 1)] = std::get<1>(result);
        spectrum->getVariance()[ndarray::view(bbox.getMinY(), bbox.getMaxY() + 1)] = std::get<2>(result);
    } else {                            // simple profile fit
        auto specIt = spectrum->getSpectrum().begin();
        auto varIt = spectrum->getVariance().begin();
        auto itSelectRow = select.begin();
        auto itTraceRow = traceIm.getImage()->getArray().begin();
        auto itVarRow = traceIm.getVariance()->getArray().begin();
        for (std::size_t y = 0; y < height; ++y, ++specIt, ++varIt, ++itSelectRow, ++itTraceRow, ++itVarRow) {
            *specIt = 0.0;
            *varIt = 0.0;
            auto itTraceCol = itTraceRow->begin();
            auto itVarCol = itVarRow->begin();
            for (auto itSelectCol = itSelectRow->begin(); itSelectCol != itSelectRow->end();
                 ++itTraceCol, ++itVarCol, ++itSelectCol) {
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

    spec[ndarray::view(0, yMin)] = 0.0;
    bg[ndarray::view(0, yMin)] = 0.0;
    var[ndarray::view(0, yMin)] = 0.0;
    std::fill(mask.begin(true), mask.begin(true) + yMin, noData);

    spec[ndarray::view(yMax, spectrumImage.getHeight())] = 0.0;
    bg[ndarray::view(yMax, spectrumImage.getHeight())] = 0.0;
    var[ndarray::view(yMax, spectrumImage.getHeight())] = 0.0;
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
PTR(afwImage::Image<ImageT>)
FiberTrace<ImageT, MaskT, VarianceT>::constructImage(const Spectrum & spectrum) const {
    auto out = std::make_shared<afwImage::Image<ImageT>>(_trace.getBBox());
    auto const maskVal = _trace.getMask()->getPlaneBitMask(fiberMaskPlane);

    std::size_t const height = _trace.getHeight();
    std::size_t const width  = _trace.getImage()->getWidth();
    std::size_t const y0 = _trace.getBBox().getMinY();

    auto spec = spectrum.getSpectrum().begin() + y0;
    auto bg = spectrum.getBackground().begin() + y0;
    for (std::size_t y = 0; y < height; ++y, ++spec, ++bg) {
        auto profileIter = _trace.getImage()->row_begin(y);
        auto maskIter = _trace.getMask()->row_begin(y);
        auto outIter = out->row_begin(y);
        float const bgValue = *bg;
        float const specValue = *spec;
        for (std::size_t x = 0; x < width; ++x, ++profileIter, ++maskIter, ++outIter) {
            if (*maskIter & maskVal) {
                *outIter = bgValue + specValue*(*profileIter);
            }
        }
    }
    return out;
}


template<typename ImageT, typename MaskT, typename VarianceT>
FiberTraceSet<ImageT, MaskT, VarianceT>::FiberTraceSet(
    FiberTraceSet<ImageT, MaskT, VarianceT> const &other,
    bool const deep
) : _traces(other.getInternal()),
    _metadata(deep ? other._metadata : std::make_shared<lsst::daf::base::PropertyList>()) {
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
    std::size_t const num = _traces.size();
    std::vector<float> xCenters(num);
    std::transform(_traces.begin(), _traces.end(), xCenters.begin(),
                   [](std::shared_ptr<FiberTraceT> ft) {
                       auto const& box = ft->getTrace().getBBox();
                       return 0.5*(box.getMinX() + box.getMaxX()); });
    std::vector<std::size_t> indices = math::sortIndices(xCenters);

    Collection sorted(num);
    std::transform(indices.begin(), indices.end(), sorted.begin(),
                   [this](std::size_t ii) { return _traces[ii]; });
    _traces = std::move(sorted);
}


// Explicit instantiation
template class FiberTrace<float, lsst::afw::image::MaskPixel, float>;
template class FiberTraceSet<float, lsst::afw::image::MaskPixel, float>;

}}}
