#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/base.h"
#include "pfs/drp/stella/fitLine.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

PYBIND11_PLUGIN(fitLine) {
    py::module mod("fitLine");

    py::class_<FitLineResult> cls(mod, "FitLineResult");
    cls.def_readonly("rms", &FitLineResult::rms);
    cls.def_readonly("isValid", &FitLineResult::isValid);
    cls.def_readonly("amplitude", &FitLineResult::amplitude);
    cls.def_readonly("center", &FitLineResult::center);
    cls.def_readonly("rmsSize", &FitLineResult::rmsSize);
    cls.def_readonly("bg0", &FitLineResult::bg0);
    cls.def_readonly("bg1", &FitLineResult::bg1);
    cls.def_readonly("num", &FitLineResult::num);

    mod.def("fitLine", &fitLine, "spectrum"_a, "peakPosition"_a, "rmsSize"_a, "badBitMask"_a,
            "fittingHalfSize"_a);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
