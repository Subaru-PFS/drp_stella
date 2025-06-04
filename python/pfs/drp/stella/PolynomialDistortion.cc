#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/PolynomialDistortion.h"
#include "pfs/drp/stella/python/Distortion.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declarePolynomialDistortion(py::module & mod) {
    using Class = PolynomialDistortion;
    auto cls = python::wrapAnalyticDistortion<Class>(mod, "PolynomialDistortion");
    cls.def(py::init<int, lsst::geom::Box2D const&, ndarray::Array<double, 1, 1> const&>(),
            "distortionOrder"_a, "range"_a, "coeff"_a);
    cls.def(py::init<int, lsst::geom::Box2D const&, ndarray::Array<double, 1, 1> const&,
                     ndarray::Array<double, 1, 1> const&>(),
            "distortionOrder"_a, "range"_a, "xCoeff"_a, "yCoeff"_a);
    cls.def_static("getNumDistortionForOrder", &Class::getNumDistortionForOrder, "order"_a);
    cls.def("getNumDistortion", &Class::getNumDistortion);
    cls.def("getXCoefficients", &Class::getXCoefficients);
    cls.def("getYCoefficients", &Class::getYCoefficients);
    cls.def("getXPoly", &Class::getXPoly);
    cls.def("getYPoly", &Class::getYPoly);
    lsst::cpputils::python::addOutputOp(cls, "__str__");
    lsst::cpputils::python::addOutputOp(cls, "__repr__");
}


PYBIND11_PLUGIN(PolynomialDistortion) {
    py::module mod("PolynomialDistortion");
    declarePolynomialDistortion(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
