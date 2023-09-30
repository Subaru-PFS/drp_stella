#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/utils/python.h"
#include "pfs/drp/stella/RotScaleDistortion.h"
#include "pfs/drp/stella/python/Distortion.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


template <typename Class>
auto declareRotScaleDistortion(py::module & mod, char const* name) {
    using Array1D = typename Class::Array1D;
    auto cls = python::wrapDistortion<Class>(mod, name);
    cls.def(py::init<lsst::geom::Box2D const&, Array1D const&>(), "range"_a, "parameters"_a);
    cls.def("getRange", &Class::getRange);
    cls.def("getParameters", &Class::getParameters);
    cls.def_static("fit", &Class::fit, "range"_a, "x"_a, "y"_a, "xMeas"_a, "yMeas"_a,
                   "xErr"_a, "yErr"_a, "isLine"_a, "slope"_a, "threshold"_a=1.0e-6,
                   "forced"_a=nullptr, "params"_a=nullptr);
    lsst::utils::python::addOutputOp(cls, "__str__");
    lsst::utils::python::addOutputOp(cls, "__repr__");
    return cls;
}


PYBIND11_PLUGIN(RotScaleDistortion) {
    py::module mod("RotScaleDistortion");
    declareRotScaleDistortion<RotScaleDistortion>(mod, "RotScaleDistortion");
    declareRotScaleDistortion<DoubleRotScaleDistortion>(mod, "DoubleRotScaleDistortion");  // same API
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
