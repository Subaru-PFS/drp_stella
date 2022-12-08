#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "pfs/drp/stella/DetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

void declareDetectorMap(py::module & mod) {
    using Class = DetectorMap;
    py::class_<Class, std::shared_ptr<Class>> cls(mod, "DetectorMap");
    // getXCenter(int), getWavelength(int), findFiberId, findPoint, and findWavelength are pure virtual.
    // Sub-classes should call pfs::drp::stella::python::wrapDetectorMap to get these methods.
    cls.def("getBBox", &Class::getBBox);
    cls.def_property_readonly("bbox", &Class::getBBox);
    cls.def("getFiberId", &Class::getFiberId);
    cls.def_property_readonly("fiberId", &Class::getFiberId);
    cls.def("getNumFibers", &Class::getNumFibers);
    cls.def("__len__", &Class::getNumFibers);
    cls.def("__contains__", &Class::contains);
    cls.def("applySlitOffset", &Class::applySlitOffset, "spatial"_a, "spectral"_a);
    cls.def("getSpatialOffsets", py::overload_cast<>(&Class::getSpatialOffsets, py::const_),
            py::return_value_policy::reference);
    cls.def("getSpatialOffset", &Class::getSpatialOffset, "fiberId"_a);
    cls.def("getSpectralOffsets", py::overload_cast<>(&Class::getSpectralOffsets, py::const_),
            py::return_value_policy::reference);
    cls.def("getSpectralOffset", &Class::getSpectralOffset, "fiberId"_a);
    cls.def("setSlitOffsets",
            py::overload_cast<Class::Array1D const&, Class::Array1D const&>(&Class::setSlitOffsets),
            "spatial"_a, "spectral"_a);
    cls.def("setSlitOffsets", py::overload_cast<int, double, double>(&Class::setSlitOffsets),
            "fiberId"_a, "spatial"_a, "spectral"_a);
    cls.def("getXCenter", py::overload_cast<>(&Class::getXCenter, py::const_));
    cls.def("getXCenter", py::overload_cast<int>(&Class::getXCenter, py::const_));
    cls.def("getXCenter", py::overload_cast<int, double>(&Class::getXCenter, py::const_));
    cls.def("getXCenter",
            py::overload_cast<ndarray::Array<int, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&>(&Class::getXCenter, py::const_));
    cls.def("getXCenter",
            py::overload_cast<int, ndarray::Array<double, 1, 1> const&>(&Class::getXCenter, py::const_));
    cls.def("getWavelength", py::overload_cast<>(&Class::getWavelength, py::const_));
    cls.def("getWavelength", py::overload_cast<int>(&Class::getWavelength, py::const_), "fiberId"_a);
    cls.def("getWavelength", py::overload_cast<int, double>(&Class::getWavelength, py::const_),
            "fiberId"_a, "row"_a);
    cls.def("getDispersionAtCenter", py::overload_cast<int>(&Class::getDispersionAtCenter, py::const_),
            "fiberId"_a);
    cls.def("getDispersionAtCenter", py::overload_cast<>(&Class::getDispersionAtCenter, py::const_));
    cls.def("getDispersion", py::overload_cast<int, double, double>(&Class::getDispersion, py::const_),
            "fiberId"_a, "wavelength"_a, "dWavelength"_a=0.1);
    cls.def("getDispersion", py::overload_cast<ndarray::Array<int, 1, 1> const&,
                                               ndarray::Array<double, 1, 1> const&,
                                               double>(&Class::getDispersion, py::const_),
            "fiberId"_a, "wavelength"_a, "dWavelength"_a=0.1);
    cls.def("findFiberId", py::overload_cast<lsst::geom::PointD const&>(&Class::findFiberId, py::const_),
            "point"_a);
    cls.def("findPoint", py::overload_cast<int, double, bool>(&Class::findPoint, py::const_),
            "fiberId"_a, "wavelength"_a, "throwOnError"_a=false);
    cls.def("findPoint", py::overload_cast<int, Class::Array1D const&>(&Class::findPoint, py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("findPoint",
            py::overload_cast<Class::FiberIds const&,
                              Class::Array1D const&>(&Class::findPoint, py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("findWavelength", py::overload_cast<int, double, bool>(&Class::findWavelength, py::const_),
            "fiberId"_a, "row"_a, "throwOnError"_a=false);
    cls.def("findWavelength",
            py::overload_cast<int, Class::Array1D const&>(&Class::findWavelength, py::const_),
            "fiberId"_a, "row"_a);
    cls.def("findWavelength",
            py::overload_cast<Class::FiberIds const&,
                              Class::Array1D const&>(&Class::findWavelength, py::const_),
            "fiberId"_a, "row"_a);
    cls.def("getVisitInfo", &Class::getVisitInfo);
    cls.def("setVisitInfo", &Class::setVisitInfo, "visitInfo"_a);
    cls.def_property("visitInfo", &Class::getVisitInfo, &Class::setVisitInfo);
    cls.def("getMetadata", py::overload_cast<>(&Class::getMetadata));
    cls.def_property_readonly("metadata", py::overload_cast<>(&Class::getMetadata));
}


PYBIND11_PLUGIN(DetectorMap) {
    py::module mod("DetectorMap");
    declareDetectorMap(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
