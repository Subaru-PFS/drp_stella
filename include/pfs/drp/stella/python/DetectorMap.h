#ifndef PFS_DRP_STELLA_PYTHON_DETECTORMAP_H
#define PFS_DRP_STELLA_PYTHON_DETECTORMAP_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "pfs/drp/stella/DetectorMap.h"
#include "pfs/drp/stella/utils/checkSize.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs {
namespace drp {
namespace stella {
namespace python {

/// Use pybind11 to wrap a subclass of DetectorMap
///
/// Sub-classes of DetectorMap can use this to define the virtual overrides.
template <typename Class>
auto wrapDetectorMap(py::module & mod, char const* name) {
    pybind11::module::import("pfs.drp.stella.DetectorMap");
    py::class_<Class, std::shared_ptr<Class>, DetectorMap> cls(mod, name);
    cls.def("clone", py::overload_cast<>(&Class::clone, py::const_));
    cls.def("findFiberId", py::overload_cast<lsst::geom::PointD const&>(&Class::findFiberId, py::const_),
            "point"_a);
    cls.def("measureSlitOffsets",
            py::overload_cast<ndarray::Array<int, 1, 1> const&, ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&
                              >(&Class::measureSlitOffsets),
            "fiberId"_a, "wavelength"_a, "x"_a, "y"_a, "xErr"_a, "yErr"_a);
    return cls;
}


}}}}  // namespace pfs::drp::stella::python

#endif  // include guard
