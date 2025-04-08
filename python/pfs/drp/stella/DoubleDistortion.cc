#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/DoubleDistortion.h"
#include "pfs/drp/stella/python/Distortion.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declareDoubleDistortion(py::module & mod) {
    using Class = DoubleDistortion;
    using Array1D = ndarray::Array<double, 1, 1>;
    auto cls = python::wrapAnalyticDistortion<Class>(mod, "DoubleDistortion");
    cls.def(py::init<int, lsst::geom::Box2D const&, Array1D const&>(),
            "distortionOrder"_a, "range"_a, "coeff"_a);
    cls.def(py::init<int, lsst::geom::Box2D const&, Array1D const&, Array1D const&, Array1D const&,
                     Array1D const&>(),
            "distortionOrder"_a, "range"_a, "xLeft"_a, "yLeft"_a, "xRight"_a, "yRight"_a);
    cls.def_static("getNumDistortionForOrder", &Class::getNumDistortionForOrder, "order"_a);
    cls.def("getNumDistortion", &Class::getNumDistortion);
    cls.def("getOnRightCcd", py::overload_cast<double>(&Class::getOnRightCcd, py::const_));
    cls.def("getOnRightCcd", py::overload_cast<Array1D const&>(&Class::getOnRightCcd, py::const_));
    cls.def("getXLeftCoefficients", &Class::getXLeftCoefficients);
    cls.def("getYLeftCoefficients", &Class::getYLeftCoefficients);
    cls.def("getXRightCoefficients", &Class::getXRightCoefficients);
    cls.def("getYRightCoefficients", &Class::getYRightCoefficients);
    cls.def("getXLeft", &Class::getXLeft);
    cls.def("getYLeft", &Class::getYLeft);
    cls.def("getXRight", &Class::getXRight);
    cls.def("getYRight", &Class::getYRight);
    cls.def("getLeft", &Class::getLeft);
    cls.def("getRight", &Class::getRight);
    lsst::cpputils::python::addOutputOp(cls, "__str__");
    lsst::cpputils::python::addOutputOp(cls, "__repr__");
}


PYBIND11_PLUGIN(DoubleDistortion) {
    py::module mod("DoubleDistortion");
    mod.import("pfs.drp.stella.math");  // for NormalizedPolynomial2D
    declareDoubleDistortion(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
