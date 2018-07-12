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

    cls.def(py::init<typename Class::ConstArray const&, typename Class::ConstArray const&>(), "x"_a, "y"_a);
    cls.def("__call__", &Class::operator());
    cls.def("getX", &Class::getX);
    cls.def("getY", &Class::getY);
}

PYBIND11_PLUGIN(spline) {
    py::module mod("spline");

    declareSpline<float>(mod, "F");

    return mod.ptr();
}

} // anonymous namespace

}}}} // pfs::drp::stella