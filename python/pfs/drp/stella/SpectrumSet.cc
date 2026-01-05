#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/SpectrumSet.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

void declareSpectrumSet(py::module &mod) {
    using Class = SpectrumSet;
    py::classh<Class> cls(mod, "SpectrumSet");

    cls.def(py::init<std::size_t>(), "length"_a);
    cls.def(py::init<std::size_t, std::size_t>(), "nSpectra"_a, "length"_a);

    cls.def("size", &Class::size);
    cls.def("reserve", &Class::reserve);
    cls.def("add", py::overload_cast<std::shared_ptr<Spectrum>>(&Class::add), "spectrum"_a);
    cls.def("getLength", &Class::getLength);

    cls.def_property_readonly("fiberId", &Class::getAllFiberIds);
    cls.def("getAllFiberIds", &Class::getAllFiberIds);
    cls.def("getAllFluxes", &Class::getAllFluxes);
    cls.def("getAllWavelengths", &Class::getAllWavelengths);
    cls.def("getAllMasks", &Class::getAllMasks);
    cls.def("getAllVariances", &Class::getAllVariances);
    cls.def("getAllNormalizations", &Class::getAllNormalizations);
    cls.def("getAllNotes", &Class::getAllNotes);

    // Pythonic APIs
    cls.def("__len__", &Class::size);
    cls.def("__getitem__", [](Class const& self, std::size_t i) {
            if (i >= self.size()) throw py::index_error();
            return self.get(i);
        });
    cls.def("__setitem__",
            [](Class& self, std::size_t i, std::shared_ptr<Spectrum> spectrum) { self.set(i, spectrum); });

    cls.def(py::pickle(
        [](Class const& self) { return py::make_tuple(self.getInternal()); },
        [](py::tuple const& t) { return Class(t[0].cast<SpectrumSet::Collection>()); }
    ));
}


PYBIND11_MODULE(SpectrumSet, mod) {
    declareSpectrumSet(mod);
}

} // anonymous namespace

}}} // pfs::drp::stella
