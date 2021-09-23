#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/base.h"
#include "pfs/drp/stella/makeFootprint.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

PYBIND11_PLUGIN(makeFootprint) {
    py::module mod("makeFootprint");
    mod.def("makeFootprint", &makeFootprint, "image"_a, "peak"_a, "height"_a, "width"_a);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
