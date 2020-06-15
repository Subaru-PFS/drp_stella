#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/math/quartiles.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella { namespace math {

namespace {

PYBIND11_PLUGIN(math) {
    py::module mod("math");
    mod.def("calculateQuartiles", &calculateQuartiles<double, 1>, "values"_a, "mask"_a);
    return mod.ptr();
}

} // anonymous namespace

}}}} // pfs::drp::stella::math
