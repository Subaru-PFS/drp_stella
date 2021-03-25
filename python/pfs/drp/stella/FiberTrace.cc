#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/FiberTrace.h"

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

    cls.def("constructImage", py::overload_cast<Spectrum const&>(&Class::constructImage, py::const_),
            "spectrum"_a);
    cls.def("constructImage",
            py::overload_cast<Spectrum const&, lsst::geom::Box2I const&>(&Class::constructImage, py::const_),
            "spectrum"_a, "bbox"_a);
    cls.def("constructImage",
            py::overload_cast<typename Class::Image &, Spectrum const&>(&Class::constructImage, py::const_),
            "image"_a, "spectrum"_a);

    cls.def(py::pickle(
        [](Class const& self) { return py::make_tuple(self.getTrace(), self.getFiberId()); },
        [](py::tuple const& t) {
            return Class(t[0].cast<typename Class::MaskedImageT>(), t[1].cast<std::size_t>());
        }
    ));
}


PYBIND11_PLUGIN(FiberTrace) {
    py::module mod("FiberTrace");
    declareFiberTrace<float>(mod);
    mod.attr("fiberMaskPlane") = fiberMaskPlane;
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
