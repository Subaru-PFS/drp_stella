#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/python/DetectorMap.h"
#include "pfs/drp/stella/OpticalModelDetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {


void declareOpticalModelData(py::module_ & mod) {
    py::class_<OpticalModelData> cls(mod, "OpticalModelData");
    cls.def(
        py::init<
            OpticalModelData::Array1D const&,
            OpticalModelData::Array2D const&,
            OpticalModelData::Array2D const&,
            OpticalModelData::Array2D const&
        >(),
        "wavelength"_a, "slit"_a, "detector"_a, "pixels"_a
    );

    cls.def_readonly("wavelength", &OpticalModelData::wavelength);
    cls.def_readonly("slit", &OpticalModelData::slit);
    cls.def_readonly("detector", &OpticalModelData::detector);
    cls.def_readonly("pixels", &OpticalModelData::pixels);

    cls.def("getArray", &OpticalModelData::getArray, "system"_a);
    cls.def("getSpline", &OpticalModelData::getSpline, "x"_a, "y"_a);

    py::enum_<OpticalModelData::Coordinate> coord(cls, "Coordinate");
    coord.value("WAVELENGTH", OpticalModelData::Coordinate::WAVELENGTH);
    coord.value("SLIT_SPATIAL", OpticalModelData::Coordinate::SLIT_SPATIAL);
    coord.value("SLIT_SPECTRAL", OpticalModelData::Coordinate::SLIT_SPECTRAL);
    coord.value("DETECTOR_X", OpticalModelData::Coordinate::DETECTOR_X);
    coord.value("DETECTOR_Y", OpticalModelData::Coordinate::DETECTOR_Y);
    coord.value("PIXELS_P", OpticalModelData::Coordinate::PIXELS_P);
    coord.value("PIXELS_Q", OpticalModelData::Coordinate::PIXELS_Q);
    coord.value("ROW", OpticalModelData::Coordinate::ROW);
    coord.value("COL", OpticalModelData::Coordinate::COL);
}


void declareOpticalModelDetectorMap(py::module_ & mod) {
    py::module::import("pfs.drp.stella.DetectorMap");
    auto cls = python::wrapDetectorMap<OpticalModelDetectorMap>(
        mod, "OpticalModelDetectorMap"
    );

    cls.def(
        py::init<
            lsst::geom::Box2I const&,
            SlitModel const&,
            OpticsModel const&,
            DetectorModel const&,
            lsst::afw::image::VisitInfo const&,
            std::shared_ptr<lsst::daf::base::PropertySet>
        >(),
        "bbox"_a,
        "slitModel"_a,
        "opticsModel"_a,
        "detectorModel"_a,
        "visitInfo"_a=lsst::afw::image::VisitInfo(lsst::daf::base::PropertySet()), "metadata"_a=std::shared_ptr<lsst::daf::base::PropertySet>()
    );
    cls.def("getSlitModel", &OpticalModelDetectorMap::getSlitModel);
    cls.def("getOpticsModel", &OpticalModelDetectorMap::getOpticsModel);
    cls.def("getDetectorModel", &OpticalModelDetectorMap::getDetectorModel);
    cls.def_property_readonly("slitModel", &OpticalModelDetectorMap::getSlitModel);
    cls.def_property_readonly("opticsModel", &OpticalModelDetectorMap::getOpticsModel);
    cls.def_property_readonly("detectorModel", &OpticalModelDetectorMap::getDetectorModel);
    cls.def("getData", &OpticalModelDetectorMap::getData, "fiberId"_a);
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
                self.getOpticsModel(),
                self.getDetectorModel(),
                self.getVisitInfo(),
                self.getMetadata()
            );
        },
        [](py::tuple const& t){
            return OpticalModelDetectorMap(
                t[0].cast<lsst::geom::Box2I>(),  // bbox
                t[1].cast<SlitModel>(),  // slitModel
                t[2].cast<OpticsModel>(),  // opticsModel
                t[3].cast<DetectorModel>(),  // detectorModel
                t[4].cast<lsst::afw::image::VisitInfo>(),  // visitInfo
                t[5].cast<std::shared_ptr<lsst::daf::base::PropertySet>>()  // metadata
            );
        }
    ));
}


PYBIND11_MODULE(OpticalModelDetectorMap, mod) {
    declareOpticalModelData(mod);
    declareOpticalModelDetectorMap(mod);
}


}}} // pfs::drp::stella
