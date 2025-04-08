#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/GlobalDetectorModel.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declareGlobalDetectorModelScaling(py::module & mod) {
    using Class = GlobalDetectorModelScaling;
    py::class_<Class> cls(mod, "GlobalDetectorModelScaling");
    cls.def(py::init<double, double, double, int, int, std::size_t, float>(),
            "fiberPitch"_a, "dispersion"_a, "wavelengthCenter"_a, "minFiberId"_a, "maxFiberId"_a,
            "height"_a, "buffer"_a=0.05);
    cls.def_readwrite("fiberPitch", &Class::fiberPitch);
    cls.def_readwrite("dispersion", &Class::dispersion);
    cls.def_readwrite("wavelengthCenter", &Class::wavelengthCenter);
    cls.def_readwrite("minFiberId", &Class::minFiberId);
    cls.def_readwrite("maxFiberId", &Class::maxFiberId);
    cls.def_readwrite("height", &Class::height);
    cls.def_readwrite("buffer", &Class::buffer);
    cls.def("__call__", py::overload_cast<int, double>(&Class::operator(), py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("__call__",
            py::overload_cast<ndarray::Array<int, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&>(&Class::operator(), py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("getRange", &Class::getRange);
    lsst::cpputils::python::addOutputOp(cls, "__str__");
    lsst::cpputils::python::addOutputOp(cls, "__repr__");
}


void declareGlobalDetectorModel(py::module & mod) {
    using Class = GlobalDetectorModel;
    py::class_<Class> cls(mod, "GlobalDetectorModel");
    cls.def(py::init<int, GlobalDetectorModelScaling const&, float,
                     ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&,
                     ndarray::Array<double, 1, 1> const&>(),
            "distortionOrder"_a, "scaling"_a, "fiberCenter"_a,
            "xDistortion"_a, "yDistortion"_a, "highCcd"_a);
    cls.def("__call__", py::overload_cast<int, double>(&Class::operator(), py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("__call__",
            py::overload_cast<ndarray::Array<int, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&>(&Class::operator(), py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("__call__",
            py::overload_cast<lsst::geom::Point2D const&, bool>(&Class::operator(), py::const_),
            "xiEta"_a, "onHighCcd"_a);
    cls.def("__call__",
            py::overload_cast<ndarray::Array<double, 2, 1> const&,
                              ndarray::Array<bool, 1, 1> const&>(&Class::operator(), py::const_),
            "xiEta"_a, "onHighCcd"_a);
    cls.def_static("calculateDesignMatrix", &Class::calculateDesignMatrix,
                   "distortionOrder"_a, "xiEtaRange"_a, "xiEta"_a);
    cls.def("calculateChi2",
            py::overload_cast<ndarray::Array<double, 2, 1> const&,
                              ndarray::Array<bool, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<bool, 1, 1> const&,
                              float>(&Class::calculateChi2, py::const_),
            "xiEta"_a, "onHighCcd"_a,
            "xx"_a, "yy"_a, "xErr"_a, "yErr"_a, "good"_a=nullptr, "sysErr"_a=0.0);
    cls.def("calculateChi2",
            py::overload_cast<ndarray::Array<int, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<bool, 1, 1> const&,
                              float>(&Class::calculateChi2, py::const_),
            "fiberId"_a, "wavelength"_a, "xx"_a, "yy"_a, "xErr"_a, "yErr"_a, "good"_a=nullptr,
            "sysErr"_a=0.0);
    cls.def("getScaling", &Class::getScaling);
    cls.def_static("getNumParameters", py::overload_cast<int>(&Class::getNumParameters), "distortionOrder"_a);
    cls.def_static("makeHighCcdCoefficients", &Class::makeHighCcdCoefficients);
    cls.def("getDistortionOrder", &Class::getDistortionOrder);
    cls.def("getFiberCenter", &Class::getFiberCenter);
    cls.def_static("getNumDistortion", py::overload_cast<int>(&Class::getNumDistortion), "order"_a);
    cls.def("getOnHighCcd", py::overload_cast<int>(&Class::getOnHighCcd, py::const_));
    cls.def("getOnHighCcd",
            py::overload_cast<ndarray::Array<int, 1, 1> const&>(&Class::getOnHighCcd, py::const_));
    cls.def("getHighCcd", &Class::getHighCcd);
    cls.def("getXCoefficients", &Class::getXCoefficients);
    cls.def("getYCoefficients", &Class::getYCoefficients);
    cls.def("getXDistortion", &Class::getXDistortion);
    cls.def("getYDistortion", &Class::getYDistortion);
    cls.def("getHighCcdCoefficients", &Class::getHighCcdCoefficients);
    lsst::cpputils::python::addOutputOp(cls, "__str__");
    lsst::cpputils::python::addOutputOp(cls, "__repr__");
}


PYBIND11_PLUGIN(GlobalDetectorModel) {
    py::module mod("GlobalDetectorModel");
    declareGlobalDetectorModelScaling(mod);
    declareGlobalDetectorModel(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
