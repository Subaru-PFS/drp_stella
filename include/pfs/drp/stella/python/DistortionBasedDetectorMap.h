#ifndef PFS_DRP_STELLA_PYTHON_DISTORTIONBASEDDETECTORMAP_H
#define PFS_DRP_STELLA_PYTHON_DISTORTIONBASEDDETECTORMAP_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/utils/python.h"
#include "pfs/drp/stella/python/DetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella { namespace python {


template <typename Class>
void wrapDistortionBasedDetectorMap(py::module & mod, const char* name) {
    auto cls = wrapDetectorMap<Class>(mod, name);
    cls.def(py::init<SplinedDetectorMap const&, typename Class::Distortion const&,
                     DetectorMap::VisitInfo const&, std::shared_ptr<lsst::daf::base::PropertySet>>(),
                     "base"_a, "distortion"_a,
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
            return Class(
                t[0].cast<SplinedDetectorMap>(),  // base
                t[1].cast<typename Class::Distortion>(),  // distortion
                t[2].cast<lsst::afw::image::VisitInfo>(),  // visitInfo
                t[3].cast<std::shared_ptr<lsst::daf::base::PropertySet>>()  // metadata
            );
        }
    ));
}


}}}} // pfs::drp::stella::python

#endif  // include guard
