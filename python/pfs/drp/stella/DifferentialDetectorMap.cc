#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/utils/python.h"
#include "pfs/drp/stella/DifferentialDetectorMap.h"
#include "pfs/drp/stella/python/DetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declareDifferentialDetectorMap(py::module & mod) {
    using Class = DifferentialDetectorMap;
    auto cls = python::wrapDetectorMap<Class>(mod, "DifferentialDetectorMap");
    cls.def(py::init<std::shared_ptr<SplinedDetectorMap>, GlobalDetectorModel const&,
                     DetectorMap::VisitInfo const&, std::shared_ptr<lsst::daf::base::PropertySet>>(),
                     "base"_a, "model"_a,
                     "visitInfo"_a=DetectorMap::VisitInfo(lsst::daf::base::PropertyList()),
                     "metadata"_a=nullptr);
    cls.def("getBase", &Class::getBase);
    cls.def("getModel", &Class::getModel);
    cls.def_property_readonly("base", &Class::getBase);
    cls.def_property_readonly("model", &Class::getModel);

    cls.def(py::pickle(
        [](Class const& self) {
            return py::make_tuple(self.getBase(), self.getModel(), self.getVisitInfo(), self.getMetadata());
        },
        [](py::tuple const& t){
            return DifferentialDetectorMap(
                t[0].cast<std::shared_ptr<SplinedDetectorMap>>(),  // base
                t[1].cast<GlobalDetectorModel>(),  // model
                t[2].cast<lsst::afw::image::VisitInfo>(),  // visitInfo
                t[3].cast<std::shared_ptr<lsst::daf::base::PropertySet>>()  // metadata
            );
        }
    ));
}


PYBIND11_PLUGIN(DifferentialDetectorMap) {
    py::module mod("DifferentialDetectorMap");
    pybind11::module::import("pfs.drp.stella.DetectorMap");
    declareDifferentialDetectorMap(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
