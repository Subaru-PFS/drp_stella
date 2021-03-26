#include <algorithm>

#include "lsst/log/Log.h"

#include "pfs/drp/stella/FiberTrace.h"

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
