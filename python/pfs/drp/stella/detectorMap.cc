#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/base.h"
#include "pfs/drp/stella/DetectorMap.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

void declareDetectorMap(py::module &mod)
{
    using Class = DetectorMap;
    py::class_<Class, PTR(Class)> cls(mod, "DetectorMap");

    cls.def(py::init<lsst::geom::Box2I,
                     DetectorMap::FiberMap const&,
                     std::vector<ndarray::Array<float, 1, 1>> const&,
                     std::vector<ndarray::Array<float, 1, 1>> const&,
                     std::vector<ndarray::Array<float, 1, 1>> const&,
                     std::vector<ndarray::Array<float, 1, 1>> const&,
                     Class::Array2D const&,
                     lsst::afw::image::VisitInfo const&,
                     std::shared_ptr<lsst::daf::base::PropertySet>
                     >(),
            "bbox"_a, "fiberIds"_a, "centerKnots"_a, "centerValues"_a, "wavelengthKnots"_a,
            "wavelengthValues"_a, "slitOffsets"_a=nullptr,
            "visitInfo"_a=Class::VisitInfo(lsst::daf::base::PropertySet()), "metadata"_a=nullptr);

    // These enums are better represented as attributes in python, so they can
    // be used as constants, without the need to cast to integers.
    cls.def_property_readonly_static("DX", [](py::object const&) { return int(Class::ArrayRow::DX); });
    cls.def_property_readonly_static("DY", [](py::object const&) { return int(Class::ArrayRow::DY); });
    cls.def_property_readonly_static("DFOCUS",
                                     [](py::object const&) { return int(Class::ArrayRow::DFOCUS); });

    cls.def("findFiberId", &Class::findFiberId, "pixelPos"_a);
    cls.def("findPoint", &Class::findPoint, "fiberId"_a, "wavelength"_a);
    cls.def("findWavelength", &Class::findWavelength, "fiberId"_a, "row"_a);

    cls.def("getBBox", &Class::getBBox);
    cls.def_property_readonly("bbox", &Class::getBBox);

    cls.def("getNumFibers", &Class::getNumFibers);
    cls.def("__len__", &Class::getNumFibers);

    cls.def("getFiberIds", [](Class & self) { return self.getFiberIds(); },
            py::return_value_policy::reference_internal);
    cls.def_property_readonly("fiberIds", py::overload_cast<>(&Class::getFiberIds));

    cls.def("getWavelength",
            [](Class const& self, std::size_t fiberId) { return self.getWavelength(fiberId); },
            "fiberId"_a);
    cls.def("getWavelength", [](Class & self) { return self.getWavelength(); });
    cls.def("getWavelength", py::overload_cast<std::size_t, float>(&Class::getWavelength, py::const_));
    cls.def("setWavelength", py::overload_cast<std::size_t, Class::Array1D const&>(&Class::setWavelength),
            "fiberId"_a, "wavelength"_a);
    cls.def("setWavelength", py::overload_cast<std::size_t, Class::Array1D const&,
                                               Class::Array1D const&>(&Class::setWavelength),
            "fiberId"_a, "knots"_a, "wavelength"_a);
    cls.def_property_readonly("wavelength", py::overload_cast<>(&Class::getWavelength, py::const_));

    cls.def("getXCenter", (ndarray::Array<float, 1, 1> (Class::*)(std::size_t) const)&Class::getXCenter,
            "fiberId"_a);
    cls.def("getXCenter", [](Class & self) { return self.getXCenter(); });
    cls.def("getXCenter", py::overload_cast<std::size_t, float>(&Class::getXCenter, py::const_));
    cls.def("setXCenter", py::overload_cast<std::size_t, Class::Array1D const&>(&Class::setXCenter),
            "fiberId"_a, "xCenters"_a);
    cls.def("setXCenter", py::overload_cast<std::size_t, Class::Array1D const&,
                                            Class::Array1D const&>(&Class::setXCenter),
            "fiberId"_a, "knots"_a, "xCenters"_a);
    cls.def_property_readonly("xCenter", py::overload_cast<>(&Class::getXCenter, py::const_));

    cls.def("getSlitOffsets", [](Class & self) { return self.getSlitOffsets(); },
            py::return_value_policy::reference_internal);
    cls.def("getSlitOffsets",
            (ndarray::Array<float const, 1, 0> const (Class::*)(std::size_t) const)&Class::getSlitOffsets,
            "fiberId"_a);
    cls.def("setSlitOffsets", (void (Class::*)(ndarray::Array<float, 2, 1> const&))&Class::setSlitOffsets,
            "offsets"_a);
    cls.def("setSlitOffsets", (void (Class::*)(std::size_t,
                                               ndarray::Array<float, 1, 0> const&))&Class::setSlitOffsets,
            "fiberId"_a, "offsets"_a);
    cls.def_property("slitOffsets", py::overload_cast<>(&Class::getSlitOffsets, py::const_),
                     py::overload_cast<ndarray::Array<float, 2, 1> const&>(&Class::setSlitOffsets));

    cls.def("getFiberIndex", &Class::getFiberIndex);

    cls.def("getVisitInfo", &Class::getVisitInfo);
    cls.def("setVisitInfo", &Class::setVisitInfo);
    cls.def_property("visitInfo", &Class::getVisitInfo, &Class::setVisitInfo);

    cls.def("getMetadata", py::overload_cast<>(&Class::getMetadata));
    cls.def_property_readonly("metadata", py::overload_cast<>(&Class::getMetadata));

    cls.def("getCenterSpline", &Class::getCenterSpline);
    cls.def("getWavelengthSpline", &Class::getWavelengthSpline);

    cls.def("__getstate__",
        [](DetectorMap const& self) {
            std::size_t const numFibers = self.getFiberIds().getNumElements();
            std::vector<Class::Array1D> centerKnots;
            std::vector<Class::Array1D> centerValues;
            std::vector<Class::Array1D> wavelengthKnots;
            std::vector<Class::Array1D> wavelengthValues;
            centerKnots.reserve(numFibers);
            centerValues.reserve(numFibers);
            wavelengthKnots.reserve(numFibers);
            wavelengthValues.reserve(numFibers);
            for (std::size_t ii = 0; ii < numFibers; ++ii) {
                centerKnots.emplace_back(ndarray::copy(self.getCenterSpline(ii).getX()));
                centerValues.emplace_back(ndarray::copy(self.getCenterSpline(ii).getY()));
                wavelengthKnots.emplace_back(ndarray::copy(self.getWavelengthSpline(ii).getX()));
                wavelengthValues.emplace_back(ndarray::copy(self.getWavelengthSpline(ii).getY()));
            }
            return py::make_tuple(self.getBBox(), self.getFiberIds(), centerKnots, centerValues,
                                  wavelengthKnots, wavelengthValues, self.getSlitOffsets(),
                                  self.getVisitInfo(), self.getMetadata());
        });
    cls.def("__setstate__",
        [](DetectorMap & self, py::tuple const& t) {
            new (&self) DetectorMap(
                t[0].cast<lsst::geom::Box2I>(),
                t[1].cast<DetectorMap::FiberMap>(),
                t[2].cast<std::vector<ndarray::Array<float, 1, 1>>>(),
                t[3].cast<std::vector<ndarray::Array<float, 1, 1>>>(),
                t[4].cast<std::vector<ndarray::Array<float, 1, 1>>>(),
                t[5].cast<std::vector<ndarray::Array<float, 1, 1>>>(),
                t[6].cast<DetectorMap::Array2D>(),
                t[7].cast<Class::VisitInfo>(),
                t[8].cast<std::shared_ptr<lsst::daf::base::PropertySet>>()
                );
        });
}


PYBIND11_PLUGIN(detectorMap) {
    py::module mod("detectorMap");

    declareDetectorMap(mod);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
