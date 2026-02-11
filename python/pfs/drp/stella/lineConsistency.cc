#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/base.h"
#include "pfs/drp/stella/lineConsistency.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

PYBIND11_MODULE(lineConsistency, mod) {
    mod.def("checkLineConsistency", &checkLineConsistency, "fiberId"_a, "wavelength"_a, "xx"_a, "yy"_a, "xErr"_a, "yErr"_a, "threshold"_a=3.0);
    mod.def("checkTraceConsistency", &checkTraceConsistency, "fiberId"_a, "xx"_a, "yy"_a, "xErr"_a, "threshold"_a=3.0);
}

} // anonymous namespace

}}} // pfs::drp::stella
