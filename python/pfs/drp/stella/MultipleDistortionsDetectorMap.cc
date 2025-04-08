#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/MultipleDistortionsDetectorMap.h"
#include "pfs/drp/stella/python/DistortionBasedDetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

PYBIND11_PLUGIN(MultipleDistortionsDetectorMap) {
    py::module mod("MultipleDistortionsDetectorMap");
    pybind11::module::import("pfs.drp.stella.DetectorMap");
    auto cls = python::wrapDetectorMap<MultipleDistortionsDetectorMap>(
        mod, "MultipleDistortionsDetectorMap"
    );

    cls.def(
        py::init<
            SplinedDetectorMap const&,
            MultipleDistortionsDetectorMap::DistortionList const&,
            lsst::afw::image::VisitInfo const&,
            std::shared_ptr<lsst::daf::base::PropertySet>,
            float
        >(),
        "base"_a,
        "distortions"_a,
        "visitInfo"_a=lsst::afw::image::VisitInfo(lsst::daf::base::PropertySet()), "metadata"_a=std::shared_ptr<lsst::daf::base::PropertySet>(),
        "samplingFactor"_a=50.0
    );

    cls.def("getBase", &MultipleDistortionsDetectorMap::getBase);
    cls.def("getDistortions", &MultipleDistortionsDetectorMap::getDistortions);
    cls.def_property_readonly("base", &MultipleDistortionsDetectorMap::getBase);
    cls.def_property_readonly("distortions", &MultipleDistortionsDetectorMap::getDistortions);

    cls.def(py::pickle(
        [](MultipleDistortionsDetectorMap const& self) {
            return py::make_tuple(
                self.getBase(), self.getDistortions(), self.getVisitInfo(), self.getMetadata()
            );
        },
        [](py::tuple const& t){
            return MultipleDistortionsDetectorMap(
                t[0].cast<SplinedDetectorMap>(),  // base
                t[1].cast<MultipleDistortionsDetectorMap::DistortionList>(),  // distortions
                t[2].cast<lsst::afw::image::VisitInfo>(),  // visitInfo
                t[3].cast<std::shared_ptr<lsst::daf::base::PropertySet>>()  // metadata
            );
        }
    ));

    return mod.ptr();
}

}}} // pfs::drp::stella
