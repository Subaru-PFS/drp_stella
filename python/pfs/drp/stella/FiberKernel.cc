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
    cls.def(
        "__call__",
        py::overload_cast<FiberTrace<float> const&>(&FiberKernel::operator(), py::const_),
        "fiberTrace"_a
    );
    cls.def(
        "__call__",
        py::overload_cast<FiberTraceSet<float> const&>(&FiberKernel::operator(), py::const_),
        "fiberTraceSet"_a
    );
}


PYBIND11_MODULE(FiberKernel, mod) {
    declareFiberKernel(mod);
    mod.def(
        "fitFiberKernel",
        &fitFiberKernel,
        "image"_a,
        "fiberTraces"_a,
        "spectra"_a,
        "badBitMask"_a=0,
        "kernelHalfWidth"_a=2,
        "kernelOrder"_a=3,
        "xBackgroundSize"_a=500,
        "yBackgroundSize"_a=500
    );
}


} // anonymous namespace

}}} // pfs::drp::stella
