#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/utils/python.h"
#include "pfs/drp/stella/DistortedDetectorMap.h"
#include "pfs/drp/stella/python/DetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declareDistortedDetectorMap(py::module & mod) {
    using Class = DistortedDetectorMap;
    auto cls = python::wrapDetectorMap<Class>(mod, "DistortedDetectorMap");
    cls.def(py::init<SplinedDetectorMap const&, DetectorDistortion const&,
                     DetectorMap::VisitInfo const&, std::shared_ptr<lsst::daf::base::PropertySet>>(),
                     "base"_a, "model"_a,
                     "visitInfo"_a=DetectorMap::VisitInfo(lsst::daf::base::PropertyList()),
                     "metadata"_a=nullptr);
    cls.def("getBase", &Class::getBase);
    cls.def("getDistortion", &Class::getDistortion);
    cls.def_property_readonly("base", &Class::getBase);
    cls.def_property_readonly("distortion", &Class::getDistortion);

    cls.def(py::pickle(
        [](Class const& self) {
            return py::make_tuple(self.getBase(), self.getDistortion(),
                                  self.getVisitInfo(), self.getMetadata());
        },
        [](py::tuple const& t){
            return DistortedDetectorMap(
                t[0].cast<SplinedDetectorMap>(),  // base
                t[1].cast<DetectorDistortion>(),  // distortion
                t[2].cast<lsst::afw::image::VisitInfo>(),  // visitInfo
                t[3].cast<std::shared_ptr<lsst::daf::base::PropertySet>>()  // metadata
            );
        }
    ));
}


PYBIND11_PLUGIN(DistortedDetectorMap) {
    py::module mod("DistortedDetectorMap");
    pybind11::module::import("pfs.drp.stella.DetectorMap");
    declareDistortedDetectorMap(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
