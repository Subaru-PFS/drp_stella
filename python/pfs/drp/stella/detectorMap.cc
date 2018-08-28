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

    cls.def(py::init<lsst::afw::geom::Box2I,
                     DetectorMap::FiberMap const&,
                     DetectorMap::Array2D const&,
                     DetectorMap::Array2D const&,
                     std::size_t,
                     DetectorMap::Array2D const&,
                     DetectorMap::Array1D const&
                     >(),
            "bbox"_a, "fiberIds"_a, "xCenters"_a, "wavelengths"_a, "nKnot"_a=25,
            "slitOffsets"_a=py::none(), "throughput"_a=py::none());
    cls.def(py::init<lsst::afw::geom::Box2I,
                     DetectorMap::FiberMap const&,
                     ndarray::Array<float const, 2, 1> const&,
                     ndarray::Array<float const, 2, 1> const&,
                     ndarray::Array<float const, 2, 1> const&,
                     ndarray::Array<float const, 2, 1> const&,
                     Class::Array2D const&,
                     Class::Array1D const&,
                     lsst::afw::image::VisitInfo const&,
                     std::shared_ptr<lsst::daf::base::PropertySet>
                     >(),
            "bbox"_a, "fiberIds"_a, "centerKnots"_a, "centerValues"_a, "wavelengthKnots"_a,
            "wavelengthValues"_a, "slitOffsets"_a, "throughput"_a,
            "visitInfo"_a=Class::VisitInfo(lsst::daf::base::PropertySet()), "metadata"_a=nullptr);

    py::enum_<Class::ArrayRow>(cls, "ArrayRow")
        .value("DX", Class::ArrayRow::DX)
        .value("DY", Class::ArrayRow::DY)
        .value("DFOCUS", Class::ArrayRow::DFOCUS)
        .export_values();

    cls.def("findFiberId", &Class::findFiberId, "pixelPos"_a);
    cls.def("findPoint", &Class::findPoint, "fiberId"_a, "wavelength"_a);
    cls.def("findWavelength", &Class::findWavelength, "fiberId"_a, "row"_a);

    cls.def("getBBox", &Class::getBBox);
    cls.def_property_readonly("bbox", &Class::getBBox);

    cls.def("getNKnot", &Class::getNKnot);
    cls.def_property_readonly("nKnot", &Class::getNKnot);

    cls.def("getFiberIds", [](Class & self) { return self.getFiberIds(); },
            py::return_value_policy::reference_internal);
    cls.def_property_readonly("fiberIds", py::overload_cast<>(&Class::getFiberIds));

    cls.def("getWavelength",
            [](Class const& self, std::size_t fiberId) { return self.getWavelength(fiberId); },
            "fiberId"_a);
    cls.def("getWavelength", [](Class & self) { return self.getWavelength(); });
    cls.def("setWavelength", &Class::setWavelength, "fiberId"_a, "wavelength"_a);
    cls.def_property_readonly("wavelength", py::overload_cast<>(&Class::getWavelength, py::const_));

    cls.def("getXCenter", (ndarray::Array<float, 1, 1> (Class::*)(std::size_t) const)&Class::getXCenter,
            "fiberId"_a);
    cls.def("getXCenter", [](Class & self) { return self.getXCenter(); });
    cls.def("setXCenter", &Class::setXCenter, "fiberId"_a, "xCenters"_a);
    cls.def_property_readonly("xCenter", py::overload_cast<>(&Class::getXCenter, py::const_));

    cls.def("getThroughput",
            [](Class const& self, std::size_t fiberId) { return self.getThroughput(fiberId); },
            "fiberId"_a);
    cls.def("getThroughput", [](Class & self) { return self.getThroughput(); },
            py::return_value_policy::reference_internal);
    cls.def("setThroughput",
            [](Class & self, std::size_t fiberId, float throughput) {
                self.setThroughput(fiberId, throughput);
            },
            "fiberId"_a, "throughput"_a);
    cls.def_property_readonly("throughput", py::overload_cast<>(&Class::getThroughput));

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
            std::size_t const numKnots = self.getCenterSpline(0).getX().getNumElements();
            Class::Array2D centerKnots = ndarray::allocate(numFibers, numKnots);
            Class::Array2D centerValues = ndarray::allocate(numFibers, numKnots);
            Class::Array2D wavelengthKnots = ndarray::allocate(numFibers, numKnots);
            Class::Array2D wavelengthValues = ndarray::allocate(numFibers, numKnots);
            for (std::size_t ii = 0; ii < numFibers; ++ii) {
                centerKnots[ii] = self.getCenterSpline(ii).getX();
                centerValues[ii] = self.getCenterSpline(ii).getY();
                wavelengthKnots[ii] = self.getWavelengthSpline(ii).getX();
                wavelengthValues[ii] = self.getWavelengthSpline(ii).getY();
            }
            return py::make_tuple(self.getBBox(), self.getFiberIds(), centerKnots, centerValues,
                                  wavelengthKnots, wavelengthValues, self.getSlitOffsets(),
                                  self.getThroughput(), self.getVisitInfo(), self.getMetadata());
        });
    cls.def("__setstate__",
        [](DetectorMap & self, py::tuple const& t) {
            new (&self) DetectorMap(
                t[0].cast<lsst::afw::geom::Box2I>(), t[1].cast<DetectorMap::FiberMap>(),
                t[2].cast<DetectorMap::Array2D>(), t[3].cast<DetectorMap::Array2D>(),
                t[4].cast<DetectorMap::Array2D>(), t[5].cast<DetectorMap::Array2D>(),
                t[6].cast<DetectorMap::Array2D>(), t[7].cast<DetectorMap::Array1D>(),
                t[8].cast<Class::VisitInfo>(),
                t[9].cast<std::shared_ptr<lsst::daf::base::PropertySet>>()
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
