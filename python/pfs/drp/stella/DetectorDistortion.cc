#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/utils/python.h"
#include "pfs/drp/stella/DetectorDistortion.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declareDetectorDistortion(py::module & mod) {
    using Class = DetectorDistortion;
    py::class_<Class> cls(mod, "DetectorDistortion");
    cls.def(py::init<int, lsst::geom::Box2D const&, ndarray::Array<double, 1, 1> const&,
                     ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&>(),
            "distortionOrder"_a, "range"_a, "xDistortion"_a, "yDistortion"_a, "rightCcd"_a);
    cls.def("__call__", py::overload_cast<double, double>(&Class::operator(), py::const_),
            "x"_a, "y"_a);
    cls.def("__call__",
            py::overload_cast<ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&>(&Class::operator(), py::const_),
            "x"_a, "y"_a);
    cls.def("__call__",
            py::overload_cast<lsst::geom::Point2D const&, bool>(&Class::operator(), py::const_),
            "point"_a, "onRightCcd"_a);
    cls.def("__call__",
            py::overload_cast<ndarray::Array<double, 2, 1> const&,
                              ndarray::Array<bool, 1, 1> const&>(&Class::operator(), py::const_),
            "xiEta"_a, "onRightCcd"_a);
    cls.def_static("calculateDesignMatrix", &Class::calculateDesignMatrix,
                   "distortionOrder"_a, "range"_a, "x"_a, "y"_a);
    cls.def("calculateChi2",
            py::overload_cast<ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<bool, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<bool, 1, 1> const&,
                              float>(&Class::calculateChi2, py::const_),
            "x"_a, "y"_a, "onRightCcd"_a,
            "xx"_a, "yy"_a, "xErr"_a, "yErr"_a, "good"_a=nullptr, "sysErr"_a=0.0);
    cls.def_static("getNumParameters", py::overload_cast<int>(&Class::getNumParameters), "distortionOrder"_a);
    cls.def_static("makeRightCcdCoefficients", &Class::makeRightCcdCoefficients);
    cls.def("getDistortionOrder", &Class::getDistortionOrder);
    cls.def("getRange", &Class::getRange);
    cls.def_static("getNumDistortion", py::overload_cast<int>(&Class::getNumDistortion), "order"_a);
    cls.def("getOnRightCcd", py::overload_cast<double>(&Class::getOnRightCcd, py::const_));
    cls.def("getOnRightCcd",
            py::overload_cast<ndarray::Array<double, 1, 1> const&>(&Class::getOnRightCcd, py::const_));
    cls.def("getRightCcd", &Class::getRightCcd);
    cls.def("getXCoefficients", &Class::getXCoefficients);
    cls.def("getYCoefficients", &Class::getYCoefficients);
    cls.def("getXDistortion", &Class::getXDistortion);
    cls.def("getYDistortion", &Class::getYDistortion);
    cls.def("getRightCcdCoefficients", &Class::getRightCcdCoefficients);
    lsst::utils::python::addOutputOp(cls, "__str__");
    lsst::utils::python::addOutputOp(cls, "__repr__");
}


PYBIND11_PLUGIN(DetectorDistortion) {
    py::module mod("DetectorDistortion");
    declareDetectorDistortion(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
