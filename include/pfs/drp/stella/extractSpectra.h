#ifndef PFS_DRP_STELLA_EXTRACTSPECTRA_H
#define PFS_DRP_STELLA_EXTRACTSPECTRA_H

#include <utility>  // for std::pair

#include "lsst/afw/image/MaskedImage.h"

#include "pfs/drp/stella/FiberTraceSet.h"
#include "pfs/drp/stella/SpectrumSet.h"


namespace pfs {
namespace drp {
namespace stella {


std::pair<SpectrumSet, lsst::afw::image::Image<double>> extractSpectra(
    lsst::afw::image::MaskedImage<float> const& image,
    FiberTraceSet<float> const& fiberTraces,
    lsst::afw::image::MaskPixel badBitMask,
    int bgBlockSize,
    float minFracMask,
    float minFracImage
);


}}}  // namespace pfs::drp::stella

#endif  // include guard
