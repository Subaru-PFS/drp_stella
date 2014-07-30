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

/************************************************************************************************************/
/**
 * ctor
 */
ImageFiberTrace::ImageFiberTrace(lsst::afw::geom::Box2I const& bbox) :
    FiberTrace(bbox)
{
    ;
}

ndarray::Array<float, 1, 1>
ImageFiberTrace::getProfile(int y)
{
    ndarray::Array<float, 1, 1>::Index shape = ndarray::makeVector(getBBox().getWidth());
    ndarray::Array<float, 1, 1>::Index strides = ndarray::makeVector(1);

    return ndarray::external(&(*_profArray)(0, y), shape, strides);
}

ndarray::Array<float const, 1, 1>
ImageFiberTrace::getProfile(int y) const
{
    ndarray::Array<float const, 1, 1>::Index shape = ndarray::makeVector(getBBox().getWidth());
    ndarray::Array<float const, 1, 1>::Index strides = ndarray::makeVector(1);

    return ndarray::external(&(*_profArray)(getBBox().getMinX(), y), shape, strides);
}

/************************************************************************************************************/

void
FiberTraceSet::setFiberTrace(int const i, PTR(FiberTrace) trace)
{
    PTR(ImageFiberTrace) itrace = boost::dynamic_pointer_cast<ImageFiberTrace>(trace);

    if (itrace) {
        itrace->_profArray = _profArray;
        trace = itrace;
    }
    
    if (static_cast<unsigned int>(i) >= _traces.size()) { // cast makes clang happy
        _traces.resize(i + 1);
    }
    _traces[i] = itrace ? itrace : trace;
}
        
}}}
