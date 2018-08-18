#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/math/findAndTraceApertures.h"
#include "pfs/drp/stella/FiberTraces.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

template <typename ImageT,
          typename MaskT=lsst::afw::image::MaskPixel,
          typename VarianceT=lsst::afw::image::VariancePixel>
void declareFiberTrace(py::module &mod)
{
    using Class = FiberTrace<ImageT, MaskT, VarianceT>;
    py::class_<Class, std::shared_ptr<Class>> cls(mod, "FiberTrace");

    cls.def(py::init<typename Class::MaskedImageT const&, std::size_t>(),
            "maskedImage"_a, "fiberTraceId"_a=0);
    cls.def(py::init<Class&, bool>(), "fiberTrace"_a, "deep"_a=false);

    cls.def("getTrace", [](Class const& self) { return self.getTrace(); });
    cls.def("getFiberId", &Class::getFiberId);
    cls.def_property_readonly("trace", [](Class const& self) { return self.getTrace(); });
    cls.def_property("fiberId", &Class::getFiberId, &Class::setFiberId);

    cls.def("extractSpectrum", &Class::extractSpectrum, "image"_a,
            "fitBackground"_a=false, "clipNSigma"_a=0.0, "useProfile"_a=true);

    cls.def("constructImage",
            (std::shared_ptr<typename Class::Image>(Class::*)(Spectrum const&) const)
                &Class::constructImage, "spectrum"_a);
    cls.def("constructImage",
            (std::shared_ptr<typename Class::Image>(Class::*)(
                Spectrum const&, lsst::afw::geom::Box2I const&) const)
                &Class::constructImage, "spectrum"_a, "bbox"_a);
    cls.def("constructImage",
            (void(Class::*)(typename Class::Image &, Spectrum const&) const)
                &Class::constructImage, "image"_a, "spectrum"_a);

    cls.def("__getstate__",
            [](Class const& self) { return py::make_tuple(self.getTrace(), self.getFiberId()); });
    cls.def("__setstate__",
            [](Class & self, py::tuple const& t) {
                new (&self) Class(t[0].cast<typename Class::MaskedImageT>(),
                                  t[1].cast<std::size_t>()); });
}


template <typename ImageT,
          typename MaskT=lsst::afw::image::MaskPixel,
          typename VarianceT=lsst::afw::image::VariancePixel>
void declareFiberTraceSet(py::module &mod)
{
    using Class = FiberTraceSet<ImageT, MaskT, VarianceT>;
    py::class_<Class, std::shared_ptr<Class>> cls(mod, "FiberTraceSet");

    cls.def(py::init<std::size_t, std::shared_ptr<lsst::daf::base::PropertySet>>(),
            "reservation"_a, "metadata"_a=nullptr);
    cls.def(py::init<Class const&, bool>(), "fiberTraceSet"_a, "deep"_a=false);
    cls.def("size", &Class::size);
    cls.def("get", [](Class const& self, std::size_t index) { return self.get(index); });
    cls.def("set", &Class::set, "index"_a, "trace"_a);
    cls.def("add",
            [](Class & self, std::shared_ptr<typename Class::FiberTraceT> ft) { return self.add(ft); });
    cls.def("getMetadata", py::overload_cast<>(&Class::getMetadata));
    cls.def_property_readonly("metadata", py::overload_cast<>(&Class::getMetadata));
    cls.def("getInternal", &Class::getInternal);
    // Pythonic APIs
    cls.def("__len__", &Class::size);
    cls.def("__getitem__", [](Class const& self, std::size_t index) { return self.get(index); });
    cls.def("__setitem__", &Class::set);
    cls.def("__getstate__",
            [](Class const& self) { return py::make_tuple(self.getInternal(), self.getMetadata()); });
    cls.def("__setstate__",
            [](Class & self, py::tuple const& t) {
                new (&self) Class(t[0].cast<typename Class::Collection>(),
                                  t[1].cast<std::shared_ptr<lsst::daf::base::PropertySet>>()); });
}


template <typename ImageT>
void declareFunctions(py::module &mod)
{
    mod.def("findAndTraceApertures", math::findAndTraceApertures<ImageT>,
            "maskedImage"_a, "detectorMap"_a, "finding"_a, "function"_a, "fitting"_a);
    mod.def("findCenterPositionsOneTrace", math::findCenterPositionsOneTrace<ImageT>,
            "image"_a, "variance"_a, "control"_a, "nextSearchStart"_a);
}


PYBIND11_PLUGIN(fiberTraces) {
    py::module mod("fiberTraces");

    declareFiberTrace<float>(mod);

    declareFiberTraceSet<float>(mod);

    declareFunctions<float>(mod);

    mod.attr("fiberMaskPlane") = fiberMaskPlane;

    py::class_<math::FindCenterPositionsOneTraceResult, PTR(math::FindCenterPositionsOneTraceResult)>
        findResult(mod, "FindCenterPositionsOneTraceResult");
    findResult.def_readwrite("index",
                             &math::FindCenterPositionsOneTraceResult::index);
    findResult.def_readwrite("position",
                             &math::FindCenterPositionsOneTraceResult::position);
    findResult.def_readwrite("error",
                             &math::FindCenterPositionsOneTraceResult::error);
    findResult.def_readwrite("nextSearchStart",
                             &math::FindCenterPositionsOneTraceResult::nextSearchStart);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
