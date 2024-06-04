#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/utils/python.h"
#include "pfs/drp/stella/LayeredDetectorMap.h"
#include "pfs/drp/stella/python/DistortionBasedDetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

PYBIND11_PLUGIN(LayeredDetectorMap) {
    py::module mod("LayeredDetectorMap");
    pybind11::module::import("pfs.drp.stella.DetectorMap");
    auto cls = python::wrapDetectorMap<LayeredDetectorMap>(
        mod, "LayeredDetectorMap"
    );

    cls.def(
        py::init<
            lsst::geom::Box2I const&,
            ndarray::Array<double, 1, 1> const&,
            ndarray::Array<double, 1, 1> const&,
            SplinedDetectorMap const&,
            LayeredDetectorMap::DistortionList const&,
            bool,
            lsst::geom::AffineTransform const&,
            lsst::afw::image::VisitInfo const&,
            std::shared_ptr<lsst::daf::base::PropertySet>,
            float
        >(),
        "bbox"_a,
        "spatialOffsets"_a,
        "spectralOffsets"_a,
        "base"_a,
        "distortions"_a,
        "dividedDetector"_a,
        "rightCcd"_a=lsst::geom::AffineTransform(),
        "visitInfo"_a=lsst::afw::image::VisitInfo(lsst::daf::base::PropertySet()), "metadata"_a=std::shared_ptr<lsst::daf::base::PropertySet>(),
        "samplingFactor"_a=10.0
    );
    cls.def(
        py::init<
            lsst::geom::Box2I const&,
            ndarray::Array<double, 1, 1> const&,
            ndarray::Array<double, 1, 1> const&,
            SplinedDetectorMap const&,
            LayeredDetectorMap::DistortionList const&,
            bool,
            ndarray::Array<double, 1, 1> const&,
            lsst::afw::image::VisitInfo const&,
            std::shared_ptr<lsst::daf::base::PropertySet>,
            float
        >(),
        "bbox"_a,
        "spatialOffsets"_a,
        "spectralOffsets"_a,
        "base"_a,
        "distortions"_a,
        "dividedDetector"_a,
        "rightCcd"_a=ndarray::Array<double, 1, 1>(),
        "visitInfo"_a=lsst::afw::image::VisitInfo(lsst::daf::base::PropertySet()), "metadata"_a=std::shared_ptr<lsst::daf::base::PropertySet>(),
        "samplingFactor"_a=10.0
    );

    cls.def("getBase", &LayeredDetectorMap::getBase);
    cls.def("getDistortions", &LayeredDetectorMap::getDistortions);
    cls.def("getDividedDetector", &LayeredDetectorMap::getDividedDetector);
    cls.def("getRightCcd", &LayeredDetectorMap::getRightCcd);
    cls.def("getRightCcdParameters", &LayeredDetectorMap::getRightCcdParameters);
    cls.def_property_readonly("base", &LayeredDetectorMap::getBase);
    cls.def_property_readonly("distortions", &LayeredDetectorMap::getDistortions);
    cls.def_property_readonly("dividedDetector", &LayeredDetectorMap::getDividedDetector);
    cls.def_property_readonly("rightCcd", &LayeredDetectorMap::getRightCcd);

    cls.def(py::pickle(
        [](LayeredDetectorMap const& self) {
            return py::make_tuple(
                self.getBBox(),
                self.getSpatialOffsets(),
                self.getSpectralOffsets(),
                self.getBase(),
                self.getDistortions(),
                self.getDividedDetector(),
                self.getRightCcdParameters(),
                self.getVisitInfo(),
                self.getMetadata()
            );
        },
        [](py::tuple const& t){
            return LayeredDetectorMap(
                t[0].cast<lsst::geom::Box2I>(),  // bbox
                t[1].cast<ndarray::Array<double, 1, 1>>(),  // spatialOffsets
                t[2].cast<ndarray::Array<double, 1, 1>>(),  // spectralOffsets
                t[3].cast<SplinedDetectorMap>(),  // base
                t[4].cast<LayeredDetectorMap::DistortionList>(),  // distortions
                t[5].cast<bool>(),  // dividedDetector
                t[6].cast<ndarray::Array<double, 1, 1>>(),  // rightCcd
                t[7].cast<lsst::afw::image::VisitInfo>(),  // visitInfo
                t[8].cast<std::shared_ptr<lsst::daf::base::PropertySet>>()  // metadata
            );
        }
    ));

    return mod.ptr();
}

}}} // pfs::drp::stella
