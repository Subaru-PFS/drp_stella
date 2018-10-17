#include <algorithm>

#include "lsst/log/Log.h"

#include "pfs/drp/stella/FiberTrace.h"
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
    MaskT const badData = traceIm.getMask()->getPlaneBitMask({"BAD", "SAT", "CR"});
    MaskT const badSpectrum = spectrum->getMask().getPlaneBitMask("BAD");

    // Select pixels for extraction
    ndarray::Array<bool, 2, 1> select{height, width};  // select this pixel for extraction?
    {
        auto traceRow = _trace.getMask()->getArray().begin();
        auto imRow = traceIm.getMask()->getArray().begin();
        auto specRow = spectrum->getMask().begin(true);
        for (auto selectRow = select.begin(); selectRow != select.end();
             ++selectRow, ++traceRow, ++imRow, ++specRow) {
            auto traceIter = traceRow->begin();
            auto imIter = imRow->begin();
            MaskT value = 0;
            for (auto selectIter = selectRow->begin(); selectIter != selectRow->end();
                 ++selectIter, ++traceIter, ++imIter) {
                *selectIter = ((*traceIter & ftMask) > 0) && ((*imIter & badData) == 0);
                if (*selectIter) {
                    value |= *imIter;
                }
            }
            *specRow = value;
        }
    }

    // Extract
    if (useProfile) {
        auto const result = math::fitProfile2d(traceIm, select, _trace.getImage()->getArray(), fitBackground,
                                               clipNSigma);
        spectrum->getSpectrum()[ndarray::view(bbox.getMinY(), bbox.getMaxY() + 1)] = std::get<0>(result);
        spectrum->getMask().getArray()[0][ndarray::view(bbox.getMinY(),
                                                        bbox.getMaxY() + 1)] = std::get<1>(result);
        // Non-finite values can result from attempting to extract a row which is mostly bad.
        auto const extracted = spectrum->getSpectrum();
        for (std::size_t y = bbox.getMinY(); y <= std::size_t(bbox.getMaxY()); ++y) {
            if (!std::isfinite(extracted[y])) {
                spectrum->getMask().getArray()[0][y] |= badSpectrum;
            }
        }
        spectrum->getVariance()[ndarray::view(bbox.getMinY(), bbox.getMaxY() + 1)] = std::get<2>(result);
        spectrum->getBackground()[ndarray::view(bbox.getMinY(), bbox.getMaxY() + 1)] = std::get<3>(result);
    } else {                            // simple profile fit
        auto specIt = spectrum->getSpectrum().begin() + bbox.getMinY();
        auto maskIt = spectrum->getMask().begin() + bbox.getMinY();
        auto varIt = spectrum->getVariance().begin() + bbox.getMinY();
        auto itSelectRow = select.begin();
        auto itTraceRow = traceIm.getImage()->getArray().begin();
        auto itVarRow = traceIm.getVariance()->getArray().begin();
        for (std::size_t y = 0; y < height;
             ++y, ++specIt, ++maskIt, ++varIt, ++itSelectRow, ++itTraceRow, ++itVarRow) {
            *specIt = 0.0;
            *varIt = 0.0;
            auto itTraceCol = itTraceRow->begin();
            auto itVarCol = itVarRow->begin();
            std::size_t num = 0;
            for (auto itSelectCol = itSelectRow->begin(); itSelectCol != itSelectRow->end();
                 ++itTraceCol, ++itVarCol, ++itSelectCol) {
                if (*itSelectCol) {
                    *specIt += *itTraceCol;
                    *varIt += *itVarCol;
                    ++num;
                }
            }
            if (num == 0) {
                *maskIt |= badSpectrum;
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

    if (yMin > 0) {
        spec[ndarray::view(0, yMin)] = 0.0;
        bg[ndarray::view(0, yMin)] = 0.0;
        var[ndarray::view(0, yMin)] = 0.0;
        std::fill(mask.begin(true), mask.begin(true) + yMin, noData);
    }

    if (yMax < spectrumImage.getHeight()) {
        spec[ndarray::view(yMax + 1, spectrumImage.getHeight())] = 0.0;
        bg[ndarray::view(yMax + 1, spectrumImage.getHeight())] = 0.0;
        var[ndarray::view(yMax + 1, spectrumImage.getHeight())] = 0.0;
        std::fill(mask.begin(true) + yMax + 1, mask.end(true), noData);
    }

    return spectrum;
}


template<typename ImageT, typename MaskT, typename VarianceT>
PTR(afwImage::Image<ImageT>)
FiberTrace<ImageT, MaskT, VarianceT>::constructImage(
    Spectrum const& spectrum,
    lsst::geom::Box2I const& bbox
) const {
    auto out = std::make_shared<afwImage::Image<ImageT>>(bbox);
    *out = 0.0;
    constructImage(*out, spectrum);
    return out;
}


template<typename ImageT, typename MaskT, typename VarianceT>
void FiberTrace<ImageT, MaskT, VarianceT>::constructImage(
    afwImage::Image<ImageT> & image,
    Spectrum const& spectrum
) const {
    auto box = image.getBBox(lsst::afw::image::PARENT);
    box.clip(_trace.getBBox(lsst::afw::image::PARENT));

    // std::size_t const height = box.getHeight();
    // std::size_t const width  = box.getWidth();
    // std::size_t const x0 = max(image.getBBox().getMinX(), _trace.getBBox().getMinX());
    // std::size_t const y0 = max(image.getBBox().getMinY(), _trace.getBBox().getMinY());

    auto const maskVal = _trace.getMask()->getPlaneBitMask(fiberMaskPlane);
    auto spec = spectrum.getSpectrum().begin() + box.getMinY();
    auto bg = spectrum.getBackground().begin() + box.getMinY();
    for (std::ptrdiff_t y = box.getMinY(); y <= box.getMaxY(); ++y, ++spec, ++bg) {
        auto profileIter = _trace.getImage()->row_begin(y - _trace.getY0()) + box.getMinX() - _trace.getX0();
        auto maskIter = _trace.getMask()->row_begin(y - _trace.getY0()) + box.getMinX() - _trace.getX0();;
        auto imageIter = image.row_begin(y - image.getY0()) + box.getMinX() - image.getX0();;
        float const bgValue = *bg;
        float const specValue = *spec;
        for (std::ptrdiff_t x = box.getMinX(); x <= box.getMaxX();
             ++x, ++profileIter, ++maskIter, ++imageIter) {
            if (*maskIter & maskVal) {
                *imageIter += bgValue + specValue*(*profileIter);
            }
        }
    }
}


// Explicit instantiation
template class FiberTrace<float, lsst::afw::image::MaskPixel, float>;

}}} // namespace pfs::drp::stella
