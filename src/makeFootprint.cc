#include <numeric>

#include "ndarray.h"

#include "pfs/drp/stella/makeFootprint.h"

namespace pfs {
namespace drp {
namespace stella {


lsst::afw::detection::Footprint makeFootprint(
    lsst::afw::image::Image<float> const& image,
    lsst::geom::Point2I const& peak,
    int height,
    float width
) {
    int const left = std::max(int(peak.getX() - width), 0);
    int const right = std::min(int(peak.getX() + width + 0.5), image.getBBox().getMaxX());
    int const bottom = std::max(peak.getY() - height, 0);
    int const top = std::min(peak.getY() + height, image.getBBox().getMaxY());
    int const num = top - bottom + 1;  // top..bottom range is inclusive
    
    ndarray::Array<float, 1, 1> collapsed = ndarray::allocate(num);
    for (int ii = 0, yy = bottom; yy <= top; ++yy, ++ii) {
        collapsed[ii] = std::accumulate(image.row_begin(yy) + left, image.row_begin(yy) + right, 0.0);
    }

    int const middle = std::min(std::max(int(peak.getY() + 0.5) - bottom, 1), num - 2);
    auto const maxElem = std::max_element(collapsed.begin() + middle - 1, collapsed.begin() + middle + 2);
    int const center = maxElem - collapsed.begin();

    int low = center;
    for (; low > 0 && collapsed[low - 1] < collapsed[low]; --low);  // no block; test/increment only

    int high = center;
    for (; high < num - 1 && collapsed[high + 1] < collapsed[high]; ++high);  // no block; test/increment only

    // low,high went one past where they should have stopped
    lsst::geom::Box2I const box{lsst::geom::Point2I(left, low + bottom),
                                lsst::geom::Point2I(right, high + bottom)};
    auto const spans = std::make_shared<lsst::afw::geom::SpanSet>(box);
    lsst::afw::detection::Footprint footprint{spans, image.getBBox()};
    auto fpPeak = footprint.getPeaks().addNew();
    fpPeak->setFx(peak.getX());
    fpPeak->setFy(peak.getY());

    return footprint;
}


}}}  // namespace pfs::drp::stella
