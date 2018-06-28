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
                     Class::Array1D const&
                     >(),
            "bbox"_a, "fiberIds"_a, "centerKnots"_a, "centerValues"_a, "wavelengthKnots"_a,
            "wavelengthValues"_a, "slitOffsets"_a, "throughput"_a);

    py::enum_<Class::ArrayRow>(cls, "ArrayRow")
        .value("DX", Class::ArrayRow::DX)
        .value("DY", Class::ArrayRow::DY)
        .value("DFOCUS", Class::ArrayRow::DFOCUS)
        .export_values();
    
    cls.def("findFiberId", &Class::findFiberId, "pixelPos"_a);
    cls.def("findPoint", &Class::findPoint, "fiberId"_a, "wavelength"_a);
    cls.def("findWavelength", &Class::findWavelength, "fiberId"_a, "row"_a);

    cls.def("getBBox", &Class::getBBox);
    cls.def("getNKnot", &Class::getNKnot);
    cls.def("getFiberIds", [](Class & self) { return self.getFiberIds(); },
            py::return_value_policy::reference_internal);
    cls.def("getWavelength",
            [](Class const& self, std::size_t fiberId) { return self.getWavelength(fiberId); },
            "fiberId"_a);
    cls.def("getWavelength", [](Class & self) { return self.getWavelength(); });
    cls.def("setWavelength", &Class::setWavelength, "fiberId"_a, "wavelength"_a);
    cls.def("getXCenter", (ndarray::Array<float, 1, 1> (Class::*)(std::size_t) const)&Class::getXCenter,
            "fiberId"_a);
    cls.def("getXCenter", [](Class & self) { return self.getXCenter(); });
    cls.def("setXCenter", &Class::setXCenter, "fiberId"_a, "xCenters"_a);
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
    cls.def("getFiberIndex", &Class::getFiberIndex);

    cls.def("getVisitInfo", &Class::getVisitInfo);
    cls.def("setVisitInfo", &Class::setVisitInfo);

    cls.def("getCenterSpline", &Class::getCenterSpline);
    cls.def("getWavelengthSpline", &Class::getWavelengthSpline);

    cls.def("__getstate__",
        [](DetectorMap const& self) {
            return py::make_tuple(self.getBBox(), self.getFiberIds(), self.getXCenter(), self.getWavelength(),
                                  self.getNKnot(), self.getSlitOffsets(), self.getThroughput());
        });
    cls.def("__setstate__",
        [](DetectorMap & self, py::tuple const& t) {
            new (&self) DetectorMap(t[0].cast<lsst::afw::geom::Box2I>(), t[1].cast<DetectorMap::FiberMap>(),
                                    t[2].cast<DetectorMap::Array2D>(), t[3].cast<DetectorMap::Array2D>(),
                                    t[4].cast<std::size_t>(), t[5].cast<DetectorMap::Array2D>(),
                                    t[6].cast<DetectorMap::Array1D>());
        });
}

void declareFunctions(py::module &mod)
{
}

PYBIND11_PLUGIN(detectorMap) {
    py::module mod("detectorMap");

    declareDetectorMap(mod);
    declareFunctions(mod);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
