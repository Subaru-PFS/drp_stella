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


void declareGlobalDetectorModel(py::module & mod) {
    using Class = GlobalDetectorModel;
    py::class_<Class> cls(mod, "GlobalDetectorModel");
    cls.def(py::init<lsst::geom::Box2I const&, int, ndarray::Array<int, 1, 1> const&, bool,
                     ndarray::Array<double const, 1, 1> const&>(),
            "bbox"_a, "distortionOrder"_a, "fiberId"_a, "dualDetector"_a, "parameters"_a);
    cls.def("__call__", py::overload_cast<int, double>(&Class::operator(), py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("__call__",
            py::overload_cast<ndarray::Array<int, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&>(&Class::operator(), py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def_static("fit", &Class::fit, "bbox"_a, "distortionOrder"_a, "dualDetector"_a,
                  "fiberId"_a, "wavelength"_a, "x"_a, "y"_a, "xErr"_a, "yErr"_a, "good"_a=nullptr,
                  "parameters"_a=nullptr);
    cls.def("getBBox", &Class::getBBox);
    cls.def("getFiberId", &Class::getFiberId);
    cls.def("getParameters", &Class::getParameters);
    cls.def_static("getNumParameters", py::overload_cast<int, std::size_t>(&Class::getNumParameters),
                   "distortionOrder"_a, "numFibers"_a);
    cls.def("getDistortionOrder", &Class::getDistortionOrder);
    cls.def("getNumFibers", &Class::getNumFibers);
    cls.def("getDualDetector", &Class::getDualDetector);
    cls.def("getDispersion", &Class::getDispersion);
    cls.def("getWavelengthCenter", &Class::getWavelengthCenter);
    lsst::utils::python::addOutputOp(cls, "__str__");
    lsst::utils::python::addOutputOp(cls, "__repr__");
}


void declareGlobalDetectorMap(py::module & mod) {
    using Class = GlobalDetectorMap;
    auto cls = python::wrapDetectorMap<Class>(mod, "GlobalDetectorMap");
    cls.def(py::init<lsst::geom::Box2I, DetectorMap::FiberIds const&, int,
                     bool, ndarray::Array<double, 1, 1> const&, DetectorMap::Array1D const&,
                     DetectorMap::Array1D const&, DetectorMap::VisitInfo const&,
                     std::shared_ptr<lsst::daf::base::PropertySet>>(),
                     "bbox"_a, "fiberId"_a, "distortionOrder"_a, "dualDetector"_a, "parameters"_a,
                     "spatialOffsets"_a=nullptr, "spectralOffsets"_a=nullptr,
                     "visitInfo"_a=DetectorMap::VisitInfo(lsst::daf::base::PropertyList()),
                     "metadata"_a=nullptr);
    cls.def(py::init<GlobalDetectorModel const&,
                     lsst::afw::image::VisitInfo const&,
                     std::shared_ptr<lsst::daf::base::PropertySet>>(),
            "model"_a, "visitInfo"_a=Class::VisitInfo(lsst::daf::base::PropertySet()), "metadata"_a=nullptr
    );
    cls.def("getModel", &Class::getModel);
    cls.def_property_readonly("model", &Class::getModel);
    cls.def("getFiberId", &Class::getFiberId);
    cls.def("getDistortionOrder", &Class::getDistortionOrder);
    cls.def("getDualDetector", &Class::getDualDetector);
    cls.def("getParameters", &Class::getParameters);

    cls.def(py::pickle(
        [](Class const& self) {
            return py::make_tuple(self.getBBox(), self.getFiberId(), self.getDistortionOrder(),
                                  self.getDualDetector(), self.getParameters(),
                                  self.getSpatialOffsets(), self.getSpectralOffsets(),
                                  self.getVisitInfo(), self.getMetadata());
        },
        [](py::tuple const& t){
            return GlobalDetectorMap(
                t[0].cast<lsst::geom::Box2I>(),
                t[1].cast<DetectorMap::FiberIds>(),
                t[2].cast<int>(),
                t[3].cast<bool>(),
                t[4].cast<ndarray::Array<double, 1, 1>>(),
                t[5].cast<DetectorMap::Array1D>(),
                t[6].cast<DetectorMap::Array1D>(),
                t[7].cast<Class::VisitInfo>(),
                t[8].cast<std::shared_ptr<lsst::daf::base::PropertySet>>()
            );
        }
    ));
}


PYBIND11_PLUGIN(GlobalDetectorMap) {
    py::module mod("GlobalDetectorMap");
    pybind11::module::import("pfs.drp.stella.DetectorMap");
    declareGlobalDetectorModel(mod);
    declareGlobalDetectorMap(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
