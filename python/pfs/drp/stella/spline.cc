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
    py::class_<Class, PTR(Class)> cls(mod, ("Spline" + suffix).c_str());

    py::enum_<typename Class::InterpolationTypes> type(cls, "InterpolationTypes");
    type.value("NOTAKNOT", Class::InterpolationTypes::CUBIC_NOTAKNOT);
    type.value("NATURAL", Class::InterpolationTypes::CUBIC_NATURAL);
    type.export_values();

    cls.def(py::init<typename Class::ConstArray const&, typename Class::ConstArray const&,
                     typename Class::InterpolationTypes>(),
                     "x"_a, "y"_a, "type"_a=Class::InterpolationTypes::CUBIC_NOTAKNOT);
    cls.def("__call__", py::overload_cast<T const>(&Class::operator(), py::const_));
    cls.def("__call__", py::overload_cast<ndarray::Array<T, 1, 1> const&>(&Class::operator(), py::const_));
    // Copy arrays so that they are writable, and can be used freely elsewhere
    cls.def("getX", [](Class const& self) { return typename Class::Array(ndarray::copy(self.getX())); });
    cls.def("getY", [](Class const& self) { return typename Class::Array(ndarray::copy(self.getY())); });
    cls.def("getInterpolationType", &Class::getInterpolationType);
    cls.def_property_readonly("interpolationType", &Class::getInterpolationType);
}

PYBIND11_PLUGIN(spline) {
    py::module mod("spline");

    declareSpline<float>(mod, "F");

    return mod.ptr();
}

} // anonymous namespace

}}}} // pfs::drp::stella
