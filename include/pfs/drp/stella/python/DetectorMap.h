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
    cls.def("findPoint",
            [](Class const& self, ndarray::Array<int, 1, 1> const& fiberId,
               ndarray::Array<float, 1, 1> const& wavelength) {
                   std::size_t const num = fiberId.size();
                   utils::checkSize(wavelength.size(), num, "fiberId/wavelength");
                   ndarray::Array<float, 2, 1> out = ndarray::allocate(num, 2);
                   for (std::size_t ii = 0; ii < num; ++ii) {
                       auto const point = self.findPoint(fiberId[ii], wavelength[ii]);
                       out[ii][0] = point.getX();
                       out[ii][1] = point.getY();
                   }
                   return out;
               }, "fiberId"_a, "wavelength"_a);
    cls.def("findWavelength", py::overload_cast<int, float>(&Class::findWavelength, py::const_),
            "fiberId"_a, "row"_a);
    return cls;
}


}}}}  // namespace pfs::drp::stella::python

#endif  // include guard
