#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/python/DetectorMap.h"
#include "pfs/drp/stella/OpticalModelDetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {


void declareOpticalModelDetectorMapData(py::module_ & mod) {
    py::class_<OpticalModelDetectorMap::Data> cls(mod, "OpticalModelDetectorMap::Data");
    cls.def(
        py::init<
            OpticalModelDetectorMap::Array1D const&,
            ndarray::Array<double, 2, 2> const&,
            ndarray::Array<double, 2, 2> const&,
            ndarray::Array<double, 2, 2> const&
        >(),
        "wavelength"_a, "slit"_a, "detector"_a, "pixels"_a
    );

    cls.def_readonly("wavelength", &OpticalModelDetectorMap::Data::wavelength);
    cls.def_readonly("slit", &OpticalModelDetectorMap::Data::slit);
    cls.def_readonly("detector", &OpticalModelDetectorMap::Data::detector);
    cls.def_readonly("pixels", &OpticalModelDetectorMap::Data::pixels);

    cls.def("getArray", &OpticalModelDetectorMap::Data::getArray, "system"_a);
    cls.def("getSpline", &OpticalModelDetectorMap::Data::getSpline, "x"_a, "y"_a);
}


void declareOpticalModelDetectorMap(py::module_ & mod) {
    py::module::import("pfs.drp.stella.DetectorMap");
    auto cls = python::wrapDetectorMap<OpticalModelDetectorMap>(
        mod, "OpticalModelDetectorMap"
    );

    py::enum_<OpticalModelDetectorMap::Coordinate> coord(cls, "Coordinate");
    coord.value("WAVELENGTH", OpticalModelDetectorMap::Coordinate::WAVELENGTH);
    coord.value("SLIT_SPATIAL", OpticalModelDetectorMap::Coordinate::SLIT_SPATIAL);
    coord.value("SLIT_SPECTRAL", OpticalModelDetectorMap::Coordinate::SLIT_SPECTRAL);
    coord.value("DETECTOR_X", OpticalModelDetectorMap::Coordinate::DETECTOR_X);
    coord.value("DETECTOR_Y", OpticalModelDetectorMap::Coordinate::DETECTOR_Y);
    coord.value("PIXELS_P", OpticalModelDetectorMap::Coordinate::PIXELS_P);
    coord.value("PIXELS_Q", OpticalModelDetectorMap::Coordinate::PIXELS_Q);
    coord.value("ROW", OpticalModelDetectorMap::Coordinate::ROW);
    coord.value("COL", OpticalModelDetectorMap::Coordinate::COL);

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
    cls.def(
        "getSpline",
        py::overload_cast<
            int, OpticalModelDetectorMap::Coordinate, OpticalModelDetectorMap::Coordinate
        >(&OpticalModelDetectorMap::getSpline, py::const_),
        "fiberId"_a, "coordFrom"_a, "coordTo"_a
    );
    cls.def(
        "calculate",
        py::overload_cast<
            int, OpticalModelDetectorMap::Coordinate, OpticalModelDetectorMap::Coordinate, double
        >(&OpticalModelDetectorMap::calculate, py::const_),
        "fiberId"_a, "coordFrom"_a, "coordTo"_a, "value"_a
    );
    cls.def(
        "calculate",
        py::overload_cast<
            OpticalModelDetectorMap::FiberIds const&,
            OpticalModelDetectorMap::Coordinate,
            OpticalModelDetectorMap::Coordinate,
            OpticalModelDetectorMap::Array1D const&
        >(
            &OpticalModelDetectorMap::calculate, py::const_
        ),
        "fiberId"_a, "coordFrom"_a, "coordTo"_a, "value"_a
    );
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
    declareOpticalModelDetectorMapData(mod);
    declareOpticalModelDetectorMap(mod);
}


}}} // pfs::drp::stella
