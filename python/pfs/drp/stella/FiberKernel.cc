#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/FiberKernel.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declareFiberKernel(py::module & mod) {
    py::class_<FiberKernel> cls(mod, "FiberKernel");
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
    cls.def_property_readonly("halfWidth", &FiberKernel::getHalfWidth);
    cls.def_property_readonly("order", &FiberKernel::getOrder);
    cls.def_property_readonly("numPoly", &FiberKernel::getNumPoly);
    cls.def_property_readonly("numParams", &FiberKernel::getNumParams);
    cls.def_property_readonly("coefficients", &FiberKernel::getCoefficients);
    cls.def_property_readonly("polynomials", &FiberKernel::getPolynomials);
    cls.def("getHalfWidth", &FiberKernel::getHalfWidth);
    cls.def("getOrder", &FiberKernel::getOrder);
    cls.def("getNumPoly", &FiberKernel::getNumPoly);
    cls.def("getNumParams", &FiberKernel::getNumParams);
    cls.def("getCoefficients", &FiberKernel::getCoefficients);
    cls.def("getPolynomials", &FiberKernel::getPolynomials);

    cls.def(
        "__call__",
        py::overload_cast<FiberTrace<float> const&, lsst::geom::Box2I const&>(
            &FiberKernel::operator(), py::const_
        ),
        "fiberTrace"_a, "bbox"_a
    );
    cls.def(
        "__call__",
        py::overload_cast<FiberTraceSet<float> const&, lsst::geom::Box2I const&>(
            &FiberKernel::operator(), py::const_
        ),
        "fiberTraceSet"_a, "bbox"_a
    );
    cls.def("evaluate", py::overload_cast<double, double>(&FiberKernel::evaluate, py::const_), "x"_a, "y"_a);
    cls.def(
        "evaluate",
        py::overload_cast<lsst::geom::Point2D const&>(&FiberKernel::evaluate, py::const_),
        "xy"_a
    );
    cls.def(
        "makeOffsetImages",
        py::overload_cast<lsst::geom::Extent2I const&>(&FiberKernel::makeOffsetImages, py::const_),
        "dims"_a
    );
    cls.def(
        "makeOffsetImages",
        py::overload_cast<int, int>(&FiberKernel::makeOffsetImages, py::const_),
        "width"_a,
        "height"_a
    );
}


void declareImageKernel(py::module & mod) {
    py::class_<ImageKernel> cls(mod, "ImageKernel");
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
    cls.def_property_readonly("halfWidth", &ImageKernel::getHalfWidth);
    cls.def_property_readonly("order", &ImageKernel::getOrder);
    cls.def_property_readonly("numPoly", &ImageKernel::getNumPoly);
    cls.def_property_readonly("numParams", &ImageKernel::getNumParams);
    cls.def_property_readonly("coefficients", &ImageKernel::getCoefficients);
    cls.def_property_readonly("polynomials", &ImageKernel::getPolynomials);
    cls.def("getHalfWidth", &ImageKernel::getHalfWidth);
    cls.def("getOrder", &ImageKernel::getOrder);
    cls.def("getNumPoly", &ImageKernel::getNumPoly);
    cls.def("getNumParams", &ImageKernel::getNumParams);
    cls.def("getCoefficients", &ImageKernel::getCoefficients);
    cls.def("getPolynomials", &ImageKernel::getPolynomials);

    cls.def("__call__", &ImageKernel::operator(), "image"_a);
    cls.def(
        "makeOffsetImages",
        py::overload_cast<lsst::geom::Extent2I const&>(&ImageKernel::makeOffsetImages, py::const_),
        "dims"_a
    );
    cls.def(
        "makeOffsetImages",
        py::overload_cast<int, int>(&ImageKernel::makeOffsetImages, py::const_),
        "width"_a,
        "height"_a
    );
}


PYBIND11_MODULE(FiberKernel, mod) {
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
