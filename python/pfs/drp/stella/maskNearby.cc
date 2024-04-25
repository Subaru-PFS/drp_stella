#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/base.h"
#include "lsst/utils/python.h"
#include "pfs/drp/stella/maskNearby.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

PYBIND11_PLUGIN(maskNearby) {
    py::module mod("maskNearby");
    mod.def("maskNearby", &maskNearby, "values"_a, "distance"_a);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
