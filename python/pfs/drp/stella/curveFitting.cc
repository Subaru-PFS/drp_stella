#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "lsst/pex/config/python.h"  // for LSST_DECLARE_CONTROL_FIELD

#include "pfs/drp/stella/math/CurveFitting.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella { namespace math {

namespace {

void declareGaussCoeffs(py::module &mod)
{
    py::class_<GaussCoeffs, std::shared_ptr<GaussCoeffs>> cls(mod, "GaussCoeffs");
    cls.def(py::init<>());
    cls.def(py::init<GaussCoeffs const&>(), "gaussCoeffs"_a);
    cls.def_readwrite("constantBackground", &GaussCoeffs::constantBackground);
    cls.def_readwrite("linearBackground", &GaussCoeffs::linearBackground);
    cls.def_readwrite("mu", &GaussCoeffs::mu);
    cls.def_readwrite("sigma", &GaussCoeffs::sigma);
    cls.def_readwrite("strength", &GaussCoeffs::strength);
    cls.def("toNdArray", &GaussCoeffs::toNdArray);
    cls.def("set", &GaussCoeffs::set<float>, "coeffs"_a);
}

PYBIND11_PLUGIN(curveFitting) {
    py::module mod("curveFitting");

    declareGaussCoeffs(mod);

    return mod.ptr();
}

} // anonymous namespace

}}}} // pfs::drp::stella
