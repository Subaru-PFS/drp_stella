#ifndef PFS_DRP_STELLA_PYTHON_BASEDETECTORMAP_H
#define PFS_DRP_STELLA_PYTHON_BASEDETECTORMAP_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "pfs/drp/stella/BaseDetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs {
namespace drp {
namespace stella {
namespace python {

/// Use pybind11 to wrap a subclass of BaseDetectorMap
///
/// Sub-classes of BaseDetectorMap can use this to define the virtual overrides.
template <typename Class>
auto wrapDetectorMap(py::module & mod, char const* name) {
    pybind11::module::import("pfs.drp.stella.BaseDetectorMap");
    py::class_<Class, std::shared_ptr<Class>, BaseDetectorMap> cls(mod, name);
    cls.def("getXCenter", py::overload_cast<int>(&Class::getXCenter, py::const_), "fiberId"_a);
    cls.def("getXCenter", py::overload_cast<int, float>(&Class::getXCenter, py::const_),
            "fiberId"_a, "row"_a);
    cls.def("getWavelength", py::overload_cast<int>(&Class::getWavelength, py::const_), "fiberId"_a);
    cls.def("getWavelength", py::overload_cast<int, float>(&Class::getWavelength, py::const_),
            "fiberId"_a, "row"_a);
    cls.def("findFiberId", py::overload_cast<lsst::geom::PointD const&>(&Class::findFiberId, py::const_),
            "point"_a);
    cls.def("findPoint", py::overload_cast<int, float>(&Class::findPoint, py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("findWavelength", py::overload_cast<int, float>(&Class::findWavelength, py::const_),
            "fiberId"_a, "row"_a);
    return cls;
}


}}}}  // namespace pfs::drp::stella::python

#endif  // include guard
