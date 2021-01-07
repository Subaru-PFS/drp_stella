#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/utils/python.h"
#include "pfs/drp/stella/GlobalDetectorMap.h"
#include "pfs/drp/stella/python/DetectorMap.h"

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
    lsst::utils::python::addOutputOp(cls, "__str__");
    lsst::utils::python::addOutputOp(cls, "__repr__");
}


void declareFiberMap(py::module & mod) {
    using Class = FiberMap;
    py::class_<Class> cls(mod, "FiberMap");
    cls.def(py::init<ndarray::Array<int, 1, 1>>(), "fiberId"_a);
    cls.def("__call__", py::overload_cast<int>(&Class::operator(), py::const_), "fiberId"_a);
    cls.def("__call__", py::overload_cast<ndarray::Array<int, 1, 1> const&>(&Class::operator(), py::const_),
            "fiberId"_a);
    cls.def("__len__", &Class::size);
    cls.def("size", &Class::size);
    lsst::utils::python::addOutputOp(cls, "__str__");
    lsst::utils::python::addOutputOp(cls, "__repr__");
}


void declareGlobalDetectorModel(py::module & mod) {
    using Class = GlobalDetectorModel;
    py::class_<Class> cls(mod, "GlobalDetectorModel");
    cls.def(py::init<lsst::geom::Box2I const&, int, ndarray::Array<int, 1, 1> const&,
                     GlobalDetectorModelScaling const&, float,
                     ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&,
                     ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&,
                     ndarray::Array<double, 1, 1> const&>(),
            "bbox"_a, "distortionOrder"_a, "fiberId"_a, "scaling"_a, "fiberCenter"_a,
            "xDistortion"_a, "yDistortion"_a, "highCcd"_a,
            "spatialOffsets"_a=nullptr, "spectralOffsets"_a=nullptr);
    cls.def("__call__", py::overload_cast<int, double>(&Class::operator(), py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("__call__", py::overload_cast<int, double, std::size_t>(&Class::operator(), py::const_),
            "fiberId"_a, "wavelength"_a, "fiberIndex"_a);
    cls.def("__call__",
            py::overload_cast<ndarray::Array<int, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&>(&Class::operator(), py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("__call__",
            py::overload_cast<ndarray::Array<int, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<std::size_t, 1, 1> const&>(&Class::operator(), py::const_),
            "fiberId"_a, "wavelength"_a, "fiberIndex"_a);
    cls.def("__call__",
            py::overload_cast<lsst::geom::Point2D const&, std::size_t, bool>(&Class::operator(), py::const_),
            "xiEta"_a, "fiberIndex"_a, "onHighCcd"_a);
    cls.def("__call__",
            py::overload_cast<ndarray::Array<double, 2, 1> const&,
                              ndarray::Array<std::size_t, 1, 1> const&,
                              ndarray::Array<bool, 1, 1> const&>(&Class::operator(), py::const_),
            "xiEta"_a, "fiberIndex"_a, "onHighCcd"_a);
    cls.def_static("calculateDesignMatrix", &Class::calculateDesignMatrix,
                   "distortionOrder"_a, "xiEtaRange"_a, "xiEta"_a);
    cls.def("calculateChi2",
            py::overload_cast<ndarray::Array<double, 2, 1> const&,
                              ndarray::Array<std::size_t, 1, 1> const&,
                              ndarray::Array<bool, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<bool, 1, 1> const&,
                              float>(&Class::calculateChi2, py::const_),
            "xiEta"_a, "fiberIndex"_a, "onHighCcd"_a,
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
    cls.def("measureSlitOffsets",
            py::overload_cast<ndarray::Array<double, 2, 1> const&,
                              ndarray::Array<std::size_t, 1, 1> const&,
                              ndarray::Array<bool, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<bool, 1, 1> const&>(&Class::measureSlitOffsets),
            "xiEta"_a, "fiberIndex"_a, "onHighCcd"_a, "xx"_a, "yy"_a, "xErr"_a, "yErr"_a, "good"_a=nullptr);
    cls.def("measureSlitOffsets",
            py::overload_cast<ndarray::Array<int, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<bool, 1, 1> const&>(&Class::measureSlitOffsets),
            "fiberId"_a, "wavelength"_a, "xx"_a, "yy"_a, "xErr"_a, "yErr"_a, "good"_a=nullptr);
    cls.def("getFiberId", &Class::getFiberId);
    cls.def("getScaling", &Class::getScaling);
    cls.def_static("getNumParameters", py::overload_cast<int, std::size_t>(&Class::getNumParameters),
                   "distortionOrder"_a, "numFibers"_a);
    cls.def_static("makeHighCcdCoefficients", &Class::makeHighCcdCoefficients);
    cls.def("getDistortionOrder", &Class::getDistortionOrder);
    cls.def("getFiberCenter", &Class::getFiberCenter);
    cls.def("getNumFibers", &Class::getNumFibers);
    cls.def_static("getNumDistortion", py::overload_cast<int>(&Class::getNumDistortion), "order"_a);
    cls.def("getFiberIndex", py::overload_cast<int>(&Class::getFiberIndex, py::const_), "fiberId"_a);
    cls.def("getFiberIndex",
            py::overload_cast<ndarray::Array<int, 1, 1> const&>(&Class::getFiberIndex, py::const_),
            "fiberId"_a);
    cls.def("getOnHighCcd", py::overload_cast<int>(&Class::getOnHighCcd, py::const_));
    cls.def("getOnHighCcd",
            py::overload_cast<ndarray::Array<int, 1, 1> const&>(&Class::getOnHighCcd, py::const_));
    cls.def("getHighCcd", &Class::getHighCcd);
    cls.def("getXCoefficients", &Class::getXCoefficients);
    cls.def("getYCoefficients", &Class::getYCoefficients);
    cls.def("getXDistortion", &Class::getXDistortion);
    cls.def("getYDistortion", &Class::getYDistortion);
    cls.def("getHighCcdCoefficients", &Class::getHighCcdCoefficients);
    lsst::utils::python::addOutputOp(cls, "__str__");
    lsst::utils::python::addOutputOp(cls, "__repr__");
}


void declareGlobalDetectorMap(py::module & mod) {
    using Class = GlobalDetectorMap;
    auto cls = python::wrapDetectorMap<Class>(mod, "GlobalDetectorMap");
    cls.def(py::init<lsst::geom::Box2I, GlobalDetectorModel const&, DetectorMap::VisitInfo const&,
                     std::shared_ptr<lsst::daf::base::PropertySet>>(),
                     "bbox"_a, "model"_a,
                     "visitInfo"_a=DetectorMap::VisitInfo(lsst::daf::base::PropertyList()),
                     "metadata"_a=nullptr);
    cls.def("getModel", &Class::getModel);
    cls.def_property_readonly("model", &Class::getModel);
    cls.def("getFiberId", &Class::getFiberId);
    cls.def("getDistortionOrder", &Class::getDistortionOrder);
    cls.def("measureSlitOffsets", &Class::measureSlitOffsets,
            "fiberId"_a, "wavelength"_a, "x"_a, "y"_a, "xErr"_a, "yErr"_a);

    cls.def(py::pickle(
        [](Class const& self) {
            return py::make_tuple(
                self.getBBox(), self.getDistortionOrder(), self.getFiberId(), self.getModel().getFiberPitch(),
                self.getModel().getDispersion(), self.getModel().getWavelengthCenter(),
                self.getModel().getBuffer(), self.getModel().getFiberCenter(),
                self.getModel().getXCoefficients(), self.getModel().getYCoefficients(),
                self.getModel().getHighCcdCoefficients(),
                self.getModel().getSpatialOffsets(), self.getModel().getSpectralOffsets(),
                self.getVisitInfo(), self.getMetadata());
        },
        [](py::tuple const& t){
            return GlobalDetectorMap(
                t[0].cast<lsst::geom::Box2I>(),  // bbox
                t[1].cast<int>(),  // distortionOrder
                t[2].cast<ndarray::Array<int, 1, 1>>(),  // fiberId
                t[3].cast<double>(),  // fiberPitch
                t[4].cast<double>(),  // dispersion
                t[5].cast<double>(),  // wavelengthCenter
                t[6].cast<float>(),  // buffer
                t[7].cast<float>(),  // fiberCenter
                t[8].cast<ndarray::Array<double, 1, 1>>(),  // xCoeff
                t[9].cast<ndarray::Array<double, 1, 1>>(),  // yCoeff
                t[10].cast<ndarray::Array<double, 1, 1>>(),  // highCcd
                t[11].cast<ndarray::Array<double, 1, 1>>(),  // spatialOffsets
                t[12].cast<ndarray::Array<double, 1, 1>>(),  // spectralOffsets
                t[13].cast<lsst::afw::image::VisitInfo>(),  // visitInfo
                t[14].cast<std::shared_ptr<lsst::daf::base::PropertySet>>()  // metadata
            );
        }
    ));
}


PYBIND11_PLUGIN(GlobalDetectorMap) {
    py::module mod("GlobalDetectorMap");
    pybind11::module::import("pfs.drp.stella.DetectorMap");
    declareGlobalDetectorModelScaling(mod);
    declareFiberMap(mod);
    declareGlobalDetectorModel(mod);
    declareGlobalDetectorMap(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
