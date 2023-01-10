#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/utils/python.h"
#include "pfs/drp/stella/Distortion.h"
#include "pfs/drp/stella/python/Distortion.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


auto declareDistortion(py::module & mod) {
    using Class = Distortion;
    py::class_<Class, std::shared_ptr<Class>> cls(mod, "Distortion");
    cls.def("clone", &Class::clone);
    cls.def("__call__", py::overload_cast<lsst::geom::Point2D const&>(&Class::operator(), py::const_),
            "xy"_a);
    cls.def("__call__", py::overload_cast<double, double>(&Class::operator(), py::const_),
            "x"_a, "y"_a);
    cls.def("__call__",
            py::overload_cast<ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&>(&Class::operator(), py::const_),
            "x"_a, "y"_a);
    cls.def("__call__",
            py::overload_cast<ndarray::Array<double, 2, 1> const&>(&Class::operator(), py::const_),
            "xy"_a);
    cls.def("calculateChi2",
            py::overload_cast<ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<bool, 1, 1> const&,
                              float>(&Class::calculateChi2, py::const_),
            "x"_a, "y"_a, "xx"_a, "yy"_a, "xErr"_a, "yErr"_a, "good"_a=nullptr, "sysErr"_a=0.0);
    return cls;
}


PYBIND11_PLUGIN(Distortion) {
    py::module mod("Distortion");
    declareDistortion(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
