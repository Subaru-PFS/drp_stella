#ifndef PFS_DRP_STELLA_PYTHON_DISTORTION_H
#define PFS_DRP_STELLA_PYTHON_DISTORTION_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "pfs/drp/stella/BaseDistortion.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs {
namespace drp {
namespace stella {
namespace python {

/// Use pybind11 to wrap a subclass of Distortion
///
/// Sub-classes of Distortion can use this to define the virtual overrides.
/// They need to add any custom functions defined in the subclass, including
/// the constructor.
template <typename Class>
auto wrapDistortion(py::module & mod, char const* name) {
    py::class_<Class> cls(mod, name);

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
    cls.def_static("fit", &Class::fit, "order"_a, "range"_a, "x"_a, "y"_a, "xMeas"_a, "yMeas"_a,
                   "xErr"_a, "yErr"_a, "fitStatic"_a, "threshold"_a=1.0e-6);
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
    cls.def_static("getNumParametersForOrder", &Class::getNumParametersForOrder, "order"_a);
    cls.def("getNumParameters", &Class::getNumParameters);
    cls.def("getOrder", &Class::getOrder);
    cls.def("getRange", &Class::getRange);
    cls.def("getCoefficients", &Class::getCoefficients);
    cls.def("removeLowOrder", &Class::removeLowOrder, "order"_a);
    cls.def("merge", &Class::merge, "other"_a);
    return cls;
}


}}}}  // namespace pfs::drp::stella::python

#endif  // include guard
