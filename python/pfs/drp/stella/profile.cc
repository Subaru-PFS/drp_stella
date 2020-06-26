#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/profile.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


PYBIND11_PLUGIN(profile) {
    py::module mod("profile");
    mod.def("calculateSwathProfile", &calculateSwathProfile, "values"_a, "mask"_a,
            "rejIter"_a=1, "rejThresh"_a=3.0);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
