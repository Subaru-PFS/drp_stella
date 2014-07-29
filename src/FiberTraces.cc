#include "pfs/drp/stella/FiberTraces.h"

namespace pfs { namespace drp { namespace stella {
/**
 * ctor
 */
FiberTrace::FiberTrace(lsst::afw::geom::Box2I const& bbox) :
    _bbox(bbox), _xcenters(bbox.getHeight())
{
    ;
}

//std::vector<float> const & getProfile(int y) const = 0;

/************************************************************************************************************/

void
FiberTraceSet::setFiberTrace(int const i, PTR(FiberTrace) trace)
{
    if (i >= _traces.size()) {
        _traces.resize(i + 1);
    }
    _traces[i] = trace;
}
        
}}}
