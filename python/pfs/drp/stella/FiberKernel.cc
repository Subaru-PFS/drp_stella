#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/FiberKernel.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declarePolynomialKernel(py::module & mod) {
    py::class_<PolynomialKernel> cls(mod, "PolynomialKernel");
    cls.def_property_readonly("halfWidth", &PolynomialKernel::getHalfWidth);
    cls.def_property_readonly("order", &PolynomialKernel::getOrder);
    cls.def_property_readonly("numPoly", &PolynomialKernel::getNumPoly);
    cls.def_property_readonly("numParams", &PolynomialKernel::getNumParams);
    cls.def_property_readonly("coefficients", &PolynomialKernel::getCoefficients);
    cls.def_property_readonly("polynomials", &PolynomialKernel::getPolynomials);
    cls.def("getHalfWidth", &PolynomialKernel::getHalfWidth);
    cls.def("getOrder", &PolynomialKernel::getOrder);
    cls.def("getNumPoly", &PolynomialKernel::getNumPoly);
    cls.def("getNumParams", &PolynomialKernel::getNumParams);
    cls.def("getCoefficients", &PolynomialKernel::getCoefficients);
    cls.def("getPolynomials", &PolynomialKernel::getPolynomials);

    cls.def(
        "convolve",
        py::overload_cast<FiberTrace<float> const&, lsst::geom::Box2I const&>(
            &PolynomialKernel::convolve, py::const_
        ),
        "fiberTrace"_a, "bbox"_a
    );
    cls.def(
        "convolve",
        py::overload_cast<FiberTraceSet<float> const&, lsst::geom::Box2I const&>(
            &PolynomialKernel::convolve, py::const_
        ),
        "fiberTraceSet"_a, "bbox"_a
    );
    cls.def(
        "convolve",
        py::overload_cast<lsst::afw::image::Image<float> const&>(&PolynomialKernel::convolve, py::const_),
        "image"_a
    );
    cls.def(
        "makeOffsetImages",
        py::overload_cast<lsst::geom::Extent2I const&>(&PolynomialKernel::makeOffsetImages, py::const_),
        "dims"_a
    );
    cls.def(
        "makeOffsetImages",
        py::overload_cast<int, int>(&PolynomialKernel::makeOffsetImages, py::const_),
        "width"_a,
        "height"_a
    );
}


void declareFiberKernel(py::module & mod) {
    py::class_<FiberKernel, PolynomialKernel> cls(mod, "FiberKernel");
    cls.def(
        py::init<
            lsst::geom::Box2D const&,
            int,
            int,
            ndarray::ArrayRef<double const, 1, 1> const&
        >(),
        "range"_a,
        "halfWidth"_a,
        "order"_a,
        "coefficients"_a
    );
    cls.def("evaluate", py::overload_cast<double, double>(&FiberKernel::evaluate, py::const_), "x"_a, "y"_a);
    cls.def(
        "evaluate",
        py::overload_cast<lsst::geom::Point2D const&>(&FiberKernel::evaluate, py::const_),
        "xy"_a
    );
}


void declareImageKernel(py::module & mod) {
    py::class_<ImageKernel, PolynomialKernel> cls(mod, "ImageKernel");
    cls.def(
        py::init<
            lsst::geom::Box2D const&,
            int,
            int,
            ndarray::ArrayRef<double const, 1, 1> const&
        >(),
        "range"_a,
        "halfWidth"_a,
        "order"_a,
        "coefficients"_a
    );
}


PYBIND11_MODULE(FiberKernel, mod) {
    declarePolynomialKernel(mod);
    declareFiberKernel(mod);
    declareImageKernel(mod);
    mod.def(
        "fitFiberKernel",
        &fitFiberKernel,
        "image"_a,
        "fiberTraces"_a,
        "badBitMask"_a=0,
        "kernelHalfWidth"_a=2,
        "kernelOrder"_a=3,
        "xBackgroundSize"_a=500,
        "yBackgroundSize"_a=500,
        "rows"_a=nullptr,
        "maxIter"_a=20,
        "andersonDepth"_a=5,
        "fluxTol"_a=1.0e-3,
        "lsqThreshold"_a=1.0e-16
    );
    mod.def(
        "fitImageKernel",
        &fitImageKernel,
        "source"_a,
        "target"_a,
        "badBitMask"_a=0,
        "kernelHalfWidth"_a=2,
        "kernelOrder"_a=3,
        "xBackgroundSize"_a=500,
        "yBackgroundSize"_a=500,
        "rows"_a=nullptr,
        "lsqThreshold"_a=1.0e-16
    );
}


} // anonymous namespace

}}} // pfs::drp::stella
