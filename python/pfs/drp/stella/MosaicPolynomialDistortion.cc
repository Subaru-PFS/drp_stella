#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/MosaicPolynomialDistortion.h"
#include "pfs/drp/stella/python/Distortion.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declareMosaicPolynomialDistortion(py::module & mod) {
    using Class = MosaicPolynomialDistortion;
    using Array1D = ndarray::Array<double, 1, 1>;
    auto cls = python::wrapAnalyticDistortion<Class>(mod, "MosaicPolynomialDistortion");
    cls.def(py::init<int, lsst::geom::Box2D const&, ndarray::Array<double, 1, 1> const&>(),
            "order"_a, "range"_a, "coeff"_a);
    cls.def(py::init<int, lsst::geom::Box2D const&, ndarray::Array<double, 1, 1> const&,
                     ndarray::Array<double, 1, 1> const&, ndarray::Array<double, 1, 1> const&>(),
            "order"_a, "range"_a, "affineCoeff"_a, "xCoeff"_a, "yCoeff"_a);
    cls.def_static("getNumDistortionForOrder", &Class::getNumDistortionForOrder, "order"_a);
    cls.def("getNumDistortion", &Class::getNumDistortion);
    cls.def("getAffineCoefficients", &Class::getAffineCoefficients);
    cls.def("getXCoefficients", &Class::getXCoefficients);
    cls.def("getYCoefficients", &Class::getYCoefficients);
    cls.def("getAffine", &Class::getAffine);
    cls.def("getXPoly", &Class::getXPoly);
    cls.def("getYPoly", &Class::getYPoly);
    cls.def("getOnRightCcd", py::overload_cast<double>(&Class::getOnRightCcd, py::const_));
    cls.def("getOnRightCcd", py::overload_cast<Array1D const&>(&Class::getOnRightCcd, py::const_));
    lsst::cpputils::python::addOutputOp(cls, "__str__");
    lsst::cpputils::python::addOutputOp(cls, "__repr__");
}


PYBIND11_PLUGIN(MosaicPolynomialDistortion) {
    py::module mod("MosaicPolynomialDistortion");
    declareMosaicPolynomialDistortion(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
