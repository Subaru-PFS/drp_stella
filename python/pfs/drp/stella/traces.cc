#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/traces.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

void declareTracePeak(py::module &mod) {
    py::class_<TracePeak, std::shared_ptr<TracePeak>> cls(mod, "TracePeak");
    cls.def(py::init<int, int, float, int>(), "row"_a, "low"_a, "peak"_a, "high"_a);
    cls.def_readonly("span", &TracePeak::span);
    cls.def_readonly("peak", &TracePeak::peak);
    cls.def_property_readonly("row", [](TracePeak const& self) { return self.span.getY(); });
    cls.def_property_readonly("low", [](TracePeak const& self) { return self.span.getX0(); });
    cls.def_property_readonly("high", [](TracePeak const& self) { return self.span.getX1(); });
}


PYBIND11_PLUGIN(traces) {
    py::module mod("traces");
    declareTracePeak(mod);
    mod.def("findTracePeaks", &findTracePeaks, "image"_a, "threshold"_a, "badBitMask"_a=0);
    mod.def("centroidTrace", &centroidTrace, "peaks"_a, "image"_a, "radius"_a, "badBitMask"_a=0);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella