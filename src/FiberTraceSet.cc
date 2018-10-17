#include <algorithm>

#include "pfs/drp/stella/FiberTraceSet.h"
#include "pfs/drp/stella/math/Math.h"

//#define __DEBUG_FINDANDTRACE__ 1

namespace afwImage = lsst::afw::image;

namespace pfs { namespace drp { namespace stella {


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
template class FiberTraceSet<float, lsst::afw::image::MaskPixel, float>;

}}} // namespace pfs::drp::stella
