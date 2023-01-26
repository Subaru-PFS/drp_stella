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


void declareRotTiltDistortion(py::module & mod) {
    using Class = RotScaleDistortion;
    using Array1D = RotScaleDistortion::Array1D;
    auto cls = python::wrapDistortion<Class>(mod, "RotScaleDistortion");
    cls.def(py::init<lsst::geom::Box2D const&, Array1D const&>(), "range"_a, "parameters"_a);
    cls.def("getOnRightCcd", py::overload_cast<double>(&Class::getOnRightCcd, py::const_));
    cls.def("getOnRightCcd", py::overload_cast<Array1D const&>(&Class::getOnRightCcd, py::const_));
    cls.def("getRange", &Class::getRange);
    cls.def("getParameters", &Class::getParameters);
    cls.def_static("fit", &Class::fit, "range"_a, "x"_a, "y"_a, "xMeas"_a, "yMeas"_a,
                   "xErr"_a, "yErr"_a, "useForWavelength"_a, "maxFuncCalls"_a=100000);
    lsst::utils::python::addOutputOp(cls, "__str__");
    lsst::utils::python::addOutputOp(cls, "__repr__");
}


PYBIND11_PLUGIN(RotScaleDistortion) {
    py::module mod("RotScaleDistortion");
    declareRotTiltDistortion(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
