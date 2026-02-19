#ifndef PFS_DRP_STELLA_PYTHON_DISTORTION_H
#define PFS_DRP_STELLA_PYTHON_DISTORTION_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "pfs/drp/stella/Distortion.h"

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
    pybind11::module::import("pfs.drp.stella.Distortion");
    py::classh<Class, Distortion> cls(mod, name);
    cls.def("clone", &Class::clone);
    cls.def("getNumParameters", &Class::getNumParameters);
    cls.def("evaluate", py::overload_cast<lsst::geom::Point2D const&>(&Class::evaluate, py::const_),
            "xy"_a);
    cls.def("__call__", py::overload_cast<lsst::geom::Point2D const&>(&Class::operator(), py::const_),
            "xy"_a);
    cls.def("__call__", py::overload_cast<double, double>(&Class::operator(), py::const_),
            "x"_a, "y"_a);
    cls.def("__call__", py::overload_cast<Distortion::Array1D const&, Distortion::Array1D const&>(
            &Class::operator(), py::const_), "x"_a, "y"_a);
    cls.def("__call__", py::overload_cast<Distortion::Array2D const&>(&Class::operator(), py::const_),
            "xy"_a);
    return cls;
}


/// Use pybind11 to wrap a subclass of AnalyticDistortion
///
/// Sub-classes of AnalyticDistortion can use this to define the virtual overrides.
/// They need to add any custom functions defined in the subclass, including
/// the constructor.
template <typename Class>
auto wrapAnalyticDistortion(py::module & mod, char const* name) {
    pybind11::module::import("pfs.drp.stella.Distortion");
    auto cls = wrapDistortion<Class>(mod, name);
    cls.def_static(
        "fit",
        py::overload_cast<
            int, int, lsst::geom::Box2D const&,
            ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&,
            ndarray::Array<double, 1, 1> const&,  ndarray::Array<double, 1, 1> const&,
            ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&,
            ndarray::Array<bool, 1, 1> const&, ndarray::Array<double, 1, 1> const&,
            double,
            ndarray::Array<bool, 1, 1> const&, ndarray::Array<double, 1, 1> const&
        >(&Class::fit),
        "xOrder"_a, "yOrder"_a, "range"_a, "x"_a, "y"_a, "xMeas"_a, "yMeas"_a,
        "xErr"_a, "yErr"_a, "isLine"_a, "slope"_a, "threshold"_a=1.0e-6, "forced"_a=nullptr,
        "params"_a=nullptr
    );
    cls.def_static(
        "fit",
        py::overload_cast<
            int, lsst::geom::Box2D const&,
            ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&,
            ndarray::Array<double, 1, 1> const&,  ndarray::Array<double, 1, 1> const&,
            ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&,
            ndarray::Array<bool, 1, 1> const&, ndarray::Array<double, 1, 1> const&,
            double,
            ndarray::Array<bool, 1, 1> const&, ndarray::Array<double, 1, 1> const&
        >(&Class::fit),
        "order"_a, "range"_a, "x"_a, "y"_a, "xMeas"_a, "yMeas"_a,
        "xErr"_a, "yErr"_a, "isLine"_a, "slope"_a, "threshold"_a=1.0e-6, "forced"_a=nullptr,
        "params"_a=nullptr
    );
    cls.def_static("getNumParametersForOrder", &Class::getNumParametersForOrder, "order"_a);
    cls.def("getOrder", &Class::getOrder);
    cls.def("getXOrder", &Class::getXOrder);
    cls.def("getYOrder", &Class::getYOrder);
    cls.def("getRange", &Class::getRange);
    cls.def("getCoefficients", &Class::getCoefficients);
    return cls;
}


}}}}  // namespace pfs::drp::stella::python

#endif  // include guard
