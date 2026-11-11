#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/BSplineDistortion.h"
#include "pfs/drp/stella/python/Distortion.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declareBSpline2D(py::module & mod) {
    using Class = BSpline2D;
    py::classh<Class> cls(mod, "BSpline2D");
    cls.def(py::init<int>(), "order"_a=3);
    cls.def(py::init<int, ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&,
                     ndarray::Array<double, 2, 1> const&>(),
            "order"_a, "xKnots"_a, "yKnots"_a, "coeff"_a);
    cls.def("copy", &Class::copy);
    cls.def("getOrder", &Class::getOrder);
    cls.def("getXKnots", &Class::getXKnots);
    cls.def("getYKnots", &Class::getYKnots);
    cls.def("getCoefficients", &Class::getCoefficients);
    cls.def("isEmpty", &Class::isEmpty);
    cls.def("getNumParameters", &Class::getNumParameters);
    cls.def("getRange", &Class::getRange);
    cls.def("__call__", py::overload_cast<double, double>(&Class::operator(), py::const_), "x"_a, "y"_a);
    cls.def("__call__", py::overload_cast<lsst::geom::Point2D const&>(&Class::operator(), py::const_),
            "xy"_a);
    cls.def("__call__", py::overload_cast<ndarray::Array<double, 1, 1> const&,
                                          ndarray::Array<double, 1, 1> const&>(&Class::operator(), py::const_),
            "x"_a, "y"_a);
    lsst::cpputils::python::addOutputOp(cls, "__str__");
    lsst::cpputils::python::addOutputOp(cls, "__repr__");
}


void declareBSplineDistortion(py::module & mod) {
    using Class = BSplineDistortion;
    auto cls = python::wrapDistortion<Class>(mod, "BSplineDistortion");
    cls.def(py::init<ndarray::Array<double, 1, 1> const&, Class::SplineList const&, Class::SplineList const&>(),
            "xBoundaries"_a, "xSplines"_a, "ySplines"_a);
    cls.def("getXBoundaries", &Class::getXBoundaries);
    cls.def("getXSplines", &Class::getXSplines);
    cls.def("getYSplines", &Class::getYSplines);
    cls.def("getNumRegions", &Class::getNumRegions);
    cls.def("getRegion", &Class::getRegion, "x"_a);
    lsst::cpputils::python::addOutputOp(cls, "__str__");
    lsst::cpputils::python::addOutputOp(cls, "__repr__");
}


PYBIND11_MODULE(BSplineDistortion, mod) {
    declareBSpline2D(mod);
    declareBSplineDistortion(mod);
}

} // anonymous namespace

}}} // pfs::drp::stella
