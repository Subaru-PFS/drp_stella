#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/python/DetectorMap.h"
#include "pfs/drp/stella/OpticalModelDetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

PYBIND11_MODULE(OpticalModelDetectorMap, mod) {
    pybind11::module::import("pfs.drp.stella.DetectorMap");
    auto cls = python::wrapDetectorMap<OpticalModelDetectorMap>(
        mod, "OpticalModelDetectorMap"
    );

    cls.def(
        py::init<
            lsst::geom::Box2I const&,
            SlitModel const&,
            OpticalModel const&,
            DetectorModel const&,
            lsst::afw::image::VisitInfo const&,
            std::shared_ptr<lsst::daf::base::PropertySet>
        >(),
        "bbox"_a,
        "slitModel"_a,
        "opticalModel"_a,
        "detectorModel"_a,
        "visitInfo"_a=lsst::afw::image::VisitInfo(lsst::daf::base::PropertySet()), "metadata"_a=std::shared_ptr<lsst::daf::base::PropertySet>()
    );
    cls.def("getSlitModel", &OpticalModelDetectorMap::getSlitModel);
    cls.def("getOpticalModel", &OpticalModelDetectorMap::getOpticalModel);
    cls.def("getDetectorModel", &OpticalModelDetectorMap::getDetectorModel);
    cls.def_property_readonly("slitModel", &OpticalModelDetectorMap::getSlitModel);
    cls.def_property_readonly("opticalModel", &OpticalModelDetectorMap::getOpticalModel);
    cls.def_property_readonly("detectorModel", &OpticalModelDetectorMap::getDetectorModel);
    cls.def("getXDetectorSpline", &OpticalModelDetectorMap::getXDetectorSpline, "fiberId"_a);
    cls.def("getYDetectorSpline", &OpticalModelDetectorMap::getYDetectorSpline, "fiberId"_a);
    cls.def("getWavelengthSpline", &OpticalModelDetectorMap::getWavelengthSpline, "fiberId"_a);
    cls.def("getRowSpline", &OpticalModelDetectorMap::getRowSpline, "fiberId"_a);
    cls.def("findPointFull", &OpticalModelDetectorMap::findPointFull, "fiberId"_a, "wavelength"_a);

    cls.def(py::pickle(
        [](OpticalModelDetectorMap const& self) {
            return py::make_tuple(
                self.getBBox(),
                self.getSlitModel(),
                self.getOpticalModel(),
                self.getDetectorModel(),
                self.getVisitInfo(),
                self.getMetadata()
            );
        },
        [](py::tuple const& t){
            return OpticalModelDetectorMap(
                t[0].cast<lsst::geom::Box2I>(),  // bbox
                t[1].cast<SlitModel>(),  // slitModel
                t[2].cast<OpticalModel>(),  // opticalModel
                t[3].cast<DetectorModel>(),  // detectorModel
                t[4].cast<lsst::afw::image::VisitInfo>(),  // visitInfo
                t[5].cast<std::shared_ptr<lsst::daf::base::PropertySet>>()  // metadata
            );
        }
    ));
}

}}} // pfs::drp::stella
