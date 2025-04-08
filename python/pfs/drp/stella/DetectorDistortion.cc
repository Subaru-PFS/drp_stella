#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/DetectorDistortion.h"
#include "pfs/drp/stella/python/Distortion.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declareDetectorDistortion(py::module & mod) {
    using Class = DetectorDistortion;
    auto cls = python::wrapAnalyticDistortion<Class>(mod, "DetectorDistortion");
    cls.def(py::init<int, lsst::geom::Box2D const&, ndarray::Array<double, 1, 1> const&>(),
            "distortionOrder"_a, "range"_a, "coeff"_a);
    cls.def(py::init<int, lsst::geom::Box2D const&, ndarray::Array<double, 1, 1> const&,
                     ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&>(),
            "distortionOrder"_a, "range"_a, "xDistortion"_a, "yDistortion"_a, "rightCcd"_a);
    cls.def_static("makeRightCcdCoefficients", &Class::makeRightCcdCoefficients);
    cls.def_static("getNumDistortionForOrder", &Class::getNumDistortionForOrder, "order"_a);
    cls.def("getNumDistortion", &Class::getNumDistortion);
    cls.def("getOnRightCcd", py::overload_cast<double>(&Class::getOnRightCcd, py::const_));
    cls.def("getOnRightCcd",
            py::overload_cast<ndarray::Array<double, 1, 1> const&>(&Class::getOnRightCcd, py::const_));
    cls.def("getRightCcd", &Class::getRightCcd);
    cls.def("getXCoefficients", &Class::getXCoefficients);
    cls.def("getYCoefficients", &Class::getYCoefficients);
    cls.def("getXDistortion", &Class::getXDistortion);
    cls.def("getYDistortion", &Class::getYDistortion);
    cls.def("getRightCcdCoefficients", &Class::getRightCcdCoefficients);
    lsst::cpputils::python::addOutputOp(cls, "__str__");
    lsst::cpputils::python::addOutputOp(cls, "__repr__");
}


PYBIND11_PLUGIN(DetectorDistortion) {
    py::module mod("DetectorDistortion");
    declareDetectorDistortion(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
