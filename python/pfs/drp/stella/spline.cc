#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/base.h"
#include "pfs/drp/stella/spline.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella { namespace math {

namespace {

template <typename T>
void declareSpline(py::module &mod, std::string const& suffix) {
    using Class = Spline<T>;
    py::class_<Class, std::shared_ptr<Class>> cls(mod, ("Spline" + suffix).c_str());

    py::enum_<typename Class::InterpolationTypes> interpolation(cls, "InterpolationTypes");
    interpolation.value("NOTAKNOT", Class::InterpolationTypes::CUBIC_NOTAKNOT);
    interpolation.value("NATURAL", Class::InterpolationTypes::CUBIC_NATURAL);
    interpolation.export_values();

    py::enum_<typename Class::ExtrapolationTypes> extrapolation(cls, "ExtrapolationTypes");
    extrapolation.value("ALL", Class::ExtrapolationTypes::EXTRAPOLATE_ALL);
    extrapolation.value("SINGLE", Class::ExtrapolationTypes::EXTRAPOLATE_SINGLE);
    extrapolation.value("NONE", Class::ExtrapolationTypes::EXTRAPOLATE_NONE);
    extrapolation.export_values();

    cls.def(py::init<typename Class::Array const&, typename Class::Array const&,
                     typename Class::InterpolationTypes, typename Class::ExtrapolationTypes>(),
                     "x"_a, "y"_a, "interpolationType"_a=Class::InterpolationTypes::CUBIC_NOTAKNOT,
                     "extrapolationType"_a=Class::ExtrapolationTypes::EXTRAPOLATE_ALL);
    cls.def("__call__", py::overload_cast<T const>(&Class::operator(), py::const_));
    cls.def("__call__", py::overload_cast<ndarray::Array<T, 1, 1> const&>(&Class::operator(), py::const_));
    // Copy arrays so that they are writable, and can be used freely elsewhere
    cls.def("getX", [](Class const& self) { return typename Class::Array(ndarray::copy(self.getX())); });
    cls.def("getY", [](Class const& self) { return typename Class::Array(ndarray::copy(self.getY())); });
    cls.def("getInterpolationType", &Class::getInterpolationType);
    cls.def_property_readonly("interpolationType", &Class::getInterpolationType);
    cls.def("getExtrapolationType", &Class::getExtrapolationType);
    cls.def_property_readonly("extrapolationType", &Class::getExtrapolationType);
}

PYBIND11_PLUGIN(spline) {
    py::module mod("spline");

    declareSpline<double>(mod, "D");

    return mod.ptr();
}

} // anonymous namespace

}}}} // pfs::drp::stella
