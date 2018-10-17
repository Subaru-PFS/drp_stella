#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pfs/drp/stella/ReferenceLine.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

void declareReferenceLine(py::module &mod) {
    using Class = ReferenceLine;

    py::class_<Class, std::shared_ptr<Class>> cls(mod, "ReferenceLine");

    /* Member types and enums */
    py::enum_<ReferenceLine::Status>(cls, "Status", py::arithmetic())
        .value("NOWT", ReferenceLine::Status::NOWT)
        .value("FIT", ReferenceLine::Status::FIT)
        .value("RESERVED", ReferenceLine::Status::RESERVED)
        .value("MISIDENTIFIED", ReferenceLine::Status::MISIDENTIFIED)
        .value("CLIPPED", ReferenceLine::Status::CLIPPED)
        .value("SATURATED", ReferenceLine::Status::SATURATED)
        .value("INTERPOLATED", ReferenceLine::Status::INTERPOLATED)
        .value("CR", ReferenceLine::Status::CR)
        .export_values();

    cls.def(py::init<std::string, ReferenceLine::Status, double, double>(),
            "description"_a, "status"_a=ReferenceLine::Status::NOWT,
            "wavelength"_a=0, "guessedIntensity"_a=0);
    cls.def_readwrite("description", &ReferenceLine::description);
    cls.def_readwrite("status", &ReferenceLine::status);
    cls.def_readwrite("wavelength", &ReferenceLine::wavelength);
    cls.def_readwrite("guessedIntensity", &ReferenceLine::guessedIntensity);
    cls.def_readwrite("guessedPosition", &ReferenceLine::guessedPosition);
    cls.def_readwrite("fitIntensity", &ReferenceLine::fitIntensity);
    cls.def_readwrite("fitPosition", &ReferenceLine::fitPosition);
    cls.def_readwrite("fitPositionErr", &ReferenceLine::fitPositionErr);

    cls.def("__getstate__",
        [](ReferenceLine const& self) {
            return py::make_tuple(self.description, self.status, self.wavelength, self.guessedIntensity,
                                  self.guessedPosition, self.fitIntensity, self.fitPosition,
                                  self.fitPositionErr);
        });
    cls.def("__setstate__",
        [](ReferenceLine & self, py::tuple const& t) {
            new (&self) ReferenceLine(t[0].cast<std::string>(), ReferenceLine::Status(t[1].cast<int>()),
                                      t[2].cast<double>(), t[3].cast<double>(), t[4].cast<double>(),
                                      t[5].cast<double>(), t[6].cast<double>(), t[7].cast<double>());
        });
}


PYBIND11_PLUGIN(ReferenceLine) {
    py::module mod("ReferenceLine");
    declareReferenceLine(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
