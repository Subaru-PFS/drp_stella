#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/FiberKernel.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


template <typename T>
void declareLinearInterpolationHelper(py::module & mod) {
    py::class_<detail::LinearInterpolationHelper<T>> cls(mod, "LinearInterpolationHelper");
    cls.def(py::init<ndarray::Array<T, 1, 1> const&, std::size_t>(), "x"_a, "length"_a);
    cls.def("getX", &detail::LinearInterpolationHelper<T>::getX);
    cls.def_property_readonly("x", &detail::LinearInterpolationHelper<T>::getX);
    cls.def("getLength", &detail::LinearInterpolationHelper<T>::getLength);
    cls.def_property_readonly("length", &detail::LinearInterpolationHelper<T>::getLength);
    cls.def("__call__", &detail::LinearInterpolationHelper<T>::operator(), "x"_a);
}


void declareBaseKernel(py::module & mod) {
    py::class_<BaseKernel> cls(mod, "BaseKernel");
    cls.def_property_readonly("dims", &BaseKernel::getDims);
    cls.def_property_readonly("halfWidth", &BaseKernel::getHalfWidth);
    cls.def_property_readonly("numParams", &BaseKernel::getNumParams);
    cls.def_property_readonly("values", &BaseKernel::getValues);
    cls.def("getDims", &BaseKernel::getDims);
    cls.def("getHalfWidth", &BaseKernel::getHalfWidth);
    cls.def("getNumParams", &BaseKernel::getNumParams);
    cls.def("getValues", &BaseKernel::getValues);

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
            ndarray::Array<double const, 1, 1> const&
        >(),
        "dims"_a,
        "halfWidth"_a,
        "xNumBlocks"_a,
        "yNumBlocks"_a,
        "values"_a
    );
    cls.def_property_readonly("xNumBlocks", &FiberKernel::getXNumBlocks);
    cls.def_property_readonly("yNumBlocks", &FiberKernel::getYNumBlocks);
    cls.def("getXNumBlocks", &FiberKernel::getXNumBlocks);
    cls.def("getYNumBlocks", &FiberKernel::getYNumBlocks);
    cls.def("evaluate", py::overload_cast<double, double>(&FiberKernel::evaluate, py::const_), "x"_a, "y"_a);
    cls.def(
        "evaluate",
        py::overload_cast<lsst::geom::Point2D const&>(&FiberKernel::evaluate, py::const_),
        "xy"_a
    );
}


PYBIND11_MODULE(FiberKernel, mod) {
    declareLinearInterpolationHelper<double>(mod);
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
        "andersonDamping"_a=0.25,
        "fluxTol"_a=1.0e-2,
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
