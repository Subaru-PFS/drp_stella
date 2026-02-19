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
    {
        py::class_<ConsistencyResult> cls(mod, "ConsistencyResult");
        cls.def_readonly("fiberId", &ConsistencyResult::fiberId);
        cls.def_readonly("wavelength", &ConsistencyResult::wavelength);
        cls.def_readonly("x", &ConsistencyResult::x);
        cls.def_readonly("y", &ConsistencyResult::y);
        cls.def_readonly("xErr", &ConsistencyResult::xErr);
        cls.def_readonly("yErr", &ConsistencyResult::yErr);
        cls.def_readonly("flux", &ConsistencyResult::flux);
        cls.def_readonly("fluxErr", &ConsistencyResult::fluxErr);
    }
    mod.def(
        "checkLineConsistency", &checkLineConsistency,
        "fiberId"_a, "wavelength"_a, "xx"_a, "yy"_a, "xErr"_a, "yErr"_a, "flux"_a, "fluxErr"_a,
        "control"_a=lsst::afw::math::StatisticsControl()
    );

    mod.def(
        "checkTraceConsistency", &checkTraceConsistency,
        "fiberId"_a, "wavelength"_a, "xx"_a, "yy"_a, "xErr"_a, "flux"_a, "fluxErr"_a,
        "control"_a=lsst::afw::math::StatisticsControl()
    );
}

} // anonymous namespace

}}} // pfs::drp::stella
