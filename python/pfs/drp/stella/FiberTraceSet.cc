#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/FiberTraceSet.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

template <typename ImageT,
          typename MaskT=lsst::afw::image::MaskPixel,
          typename VarianceT=lsst::afw::image::VariancePixel>
void declareFiberTraceSet(py::module &mod)
{
    using Class = FiberTraceSet<ImageT, MaskT, VarianceT>;
    py::classh<Class> cls(mod, "FiberTraceSet");

    cls.def(py::init<std::size_t, std::shared_ptr<lsst::daf::base::PropertySet>>(),
            "reservation"_a, "metadata"_a=nullptr);
    cls.def(py::init<Class const&, bool>(), "fiberTraceSet"_a, "deep"_a=false);
    cls.def("size", &Class::size);
    cls.def("get", [](Class const& self, std::ptrdiff_t index) { return self.get(index); });
    cls.def("set", &Class::set, "index"_a, "trace"_a);
    cls.def("add",
            [](Class & self, std::shared_ptr<typename Class::FiberTraceT> ft) { return self.add(ft); });
    cls.def("getMetadata", py::overload_cast<>(&Class::getMetadata));
    cls.def_property_readonly("metadata", py::overload_cast<>(&Class::getMetadata));
    cls.def("getInternal", &Class::getInternal);
    cls.def("sortTracesByXCenter", &Class::sortTracesByXCenter);
    cls.def(
        "extractSpectra",
        &Class::extractSpectra,
        "image"_a,
        "badBitMask"_a=0,
        "minFracMask"_a=0.3,
        "minFracImage"_a=0.4
    );
    // Pythonic APIs
    cls.def("__len__", &Class::size);
    cls.def("__getitem__", [](Class const& self, std::ptrdiff_t index) { return self.get(index); });
    cls.def("__setitem__", &Class::set);
    cls.def(py::pickle(
        [](Class const& self) { return py::make_tuple(self.getInternal(), self.getMetadata()); },
        [](py::tuple const& t) {
            return Class(t[0].cast<typename Class::Collection>(),
                         t[1].cast<std::shared_ptr<lsst::daf::base::PropertySet>>());
        }
    ));
}


PYBIND11_MODULE(FiberTraceSet, mod) {
    declareFiberTraceSet<float>(mod);
}

} // anonymous namespace

}}} // pfs::drp::stella
