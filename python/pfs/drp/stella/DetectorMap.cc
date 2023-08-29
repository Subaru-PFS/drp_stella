#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "pfs/drp/stella/DetectorMap.h"
#include "pfs/drp/stella/utils/checkSize.h"

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
    // For convenience, add overloads that support a single-precision numpy array
    cls.def("getXCenter",
            [](Class const& self, int fiberId, ndarray::Array<double, 1, 1> const& row) {
                ndarray::Array<double, 1, 1> result(row.getShape());
                for (std::size_t ii = 0; ii < row.getShape()[0]; ++ii) {
                    result[ii] = self.getXCenter(fiberId, row[ii]);
                }
                return result;
            });
    cls.def("getXCenter",
            [](
                Class const& self,
                ndarray::Array<int, 1, 1> const& fiberId,
                ndarray::Array<float, 1, 1> const& row
            ) {
                utils::checkSize(fiberId.getShape(), row.getShape(), "fiberId vs row");
                ndarray::Array<double, 1, 1> result(fiberId.getShape());
                for (std::size_t ii = 0; ii < row.getShape()[0]; ++ii) {
                    result[ii] = self.getXCenter(fiberId[ii], row[ii]);
                }
                return result;
            });
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
    cls.def("getSlope", py::overload_cast<int, double>(&Class::getSlope, py::const_),
            "fiberId"_a, "row"_a);
    cls.def("getSlope", py::overload_cast<ndarray::Array<int, 1, 1> const&,
                                          ndarray::Array<double, 1, 1> const&,
                                          ndarray::Array<bool, 1, 1> const&>(&Class::getSlope, py::const_),
            "fiberId"_a, "row"_a, "calculate"_a=nullptr);
    cls.def("getSlope",
            [](
                Class const& self,
                ndarray::Array<int, 1, 1> const& fiberId,
                ndarray::Array<float, 1, 1> const& row,
                ndarray::Array<bool, 1, 1> const& calculate
            ) {
                utils::checkSize(fiberId.getShape(), row.getShape(), "fiberId vs row");
                std::size_t const num = fiberId.size();
                bool const haveCalculate = !calculate.isEmpty();
                if (haveCalculate) {
                    utils::checkSize(calculate.size(), num, "fiberId vs calculate");
                }
                ndarray::Array<double, 1, 1> slope = ndarray::allocate(num);
                for (std::size_t ii = 0; ii < num; ++ii) {
                    if (haveCalculate && !calculate[ii]) {
                        slope[ii] = std::numeric_limits<double>::quiet_NaN();
                        continue;
                    }
                    slope[ii] = self.getSlope(fiberId[ii], row[ii]);
                }
                return slope;
            });
    cls.def("findFiberId", py::overload_cast<double, double>(&Class::findFiberId, py::const_), "x"_a, "y"_a);
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
    // For convenience, add overloads that accept a single-precision numpy array
    cls.def("findWavelength",
            [](Class const& self, int fiberId, ndarray::Array<float, 1, 1> const& row) {
                ndarray::Array<double, 1, 1> result(row.getShape());
                for (std::size_t ii = 0; ii < row.getShape()[0]; ++ii) {
                    result[ii] = self.findWavelength(fiberId, row[ii]);
                }
                return result;
            });
    cls.def("findWavelength",
            [](
                Class const& self,
                ndarray::Array<int, 1, 1> const& fiberId,
                ndarray::Array<float, 1, 1> const& row
            ) {
                utils::checkSize(fiberId.getShape(), row.getShape(), "fiberId vs row");
                ndarray::Array<double, 1, 1> result(fiberId.getShape());
                for (std::size_t ii = 0; ii < row.getShape()[0]; ++ii) {
                    result[ii] = self.findWavelength(fiberId[ii], row[ii]);
                }
                return result;
            });
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
