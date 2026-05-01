#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/FiberKernel.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declareBaseKernel(py::module & mod) {
    py::class_<BaseKernel> cls(mod, "BaseKernel");
    cls.def_property_readonly("halfWidth", &BaseKernel::getHalfWidth);
    cls.def_property_readonly("numParams", &BaseKernel::getNumParams);
    cls.def_property_readonly("coefficients", &BaseKernel::getCoefficients);
    cls.def("getHalfWidth", &BaseKernel::getHalfWidth);
    cls.def("getNumParams", &BaseKernel::getNumParams);
    cls.def("getCoefficients", &BaseKernel::getCoefficients);

    cls.def(
        "convolve",
        py::overload_cast<FiberTrace<float> const&, lsst::geom::Box2I const&>(
            &BaseKernel::convolve, py::const_
        ),
        "fiberTrace"_a, "bbox"_a
    );
    cls.def(
        "convolve",
        py::overload_cast<FiberTraceSet<float> const&, lsst::geom::Box2I const&>(
            &BaseKernel::convolve, py::const_
        ),
        "fiberTraceSet"_a, "bbox"_a
    );
    cls.def(
        "convolve",
        py::overload_cast<lsst::afw::image::Image<float> const&>(&BaseKernel::convolve, py::const_),
        "image"_a
    );
    cls.def(
        "makeOffsetImages",
        py::overload_cast<lsst::geom::Extent2I const&>(&BaseKernel::makeOffsetImages, py::const_),
        "dims"_a
    );
    cls.def(
        "makeOffsetImages",
        py::overload_cast<int, int>(&BaseKernel::makeOffsetImages, py::const_),
        "width"_a,
        "height"_a
    );
}


void declareFiberKernel(py::module & mod) {
    py::class_<FiberKernel, BaseKernel> cls(mod, "FiberKernel");
    cls.def(
        py::init<
            lsst::geom::Extent2I const&,
            int,
            int,
            int,
            ndarray::ArrayRef<double const, 1, 1> const&
        >(),
        "range"_a,
        "halfWidth"_a,
        "xKernelNum"_a,
        "yKernelNum"_a,
        "coefficients"_a
    );
    cls.def("evaluate", py::overload_cast<double, double>(&FiberKernel::evaluate, py::const_), "x"_a, "y"_a);
    cls.def(
        "evaluate",
        py::overload_cast<lsst::geom::Point2D const&>(&FiberKernel::evaluate, py::const_),
        "xy"_a
    );
}


PYBIND11_MODULE(FiberKernel, mod) {
    declareBaseKernel(mod);
    declareFiberKernel(mod);
    mod.def(
        "fitFiberKernel",
        py::overload_cast<
            lsst::afw::image::MaskedImage<float> const&,
            FiberTraceSet<float> const&,
            lsst::afw::image::MaskPixel,
            int,
            int,
            int,
            ndarray::Array<int, 1, 1> const&,
            int,
            int,
            double,
            double
        >(&fitFiberKernel),
        "image"_a,
        "fiberTraces"_a,
        "badBitMask"_a=0,
        "kernelHalfWidth"_a=2,
        "xKernelNum"_a=7,
        "yKernelNum"_a=7,
        "rows"_a=nullptr,
        "maxIter"_a=20,
        "andersonDepth"_a=5,
        "fluxTol"_a=1.0e-3,
        "lsqThreshold"_a=1.0e-16
    );
    mod.def(
        "fitFiberKernel",
        py::overload_cast<
            lsst::afw::image::MaskedImage<float> const&,
            lsst::afw::image::MaskedImage<float> const&,
            lsst::afw::image::MaskPixel,
            int,
            int,
            int,
            ndarray::Array<int, 1, 1> const&,
            double
        >(&fitFiberKernel),
        "source"_a,
        "target"_a,
        "badBitMask"_a=0,
        "kernelHalfWidth"_a=2,
        "xKernelNum"_a=7,
        "yKernelNum"_a=7,
        "rows"_a=nullptr,
        "lsqThreshold"_a=1.0e-16
    );
}


} // anonymous namespace

}}} // pfs::drp::stella
