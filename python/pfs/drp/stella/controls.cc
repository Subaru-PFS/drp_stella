#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "lsst/pex/config/python.h"  // for LSST_DECLARE_CONTROL_FIELD

#include "pfs/drp/stella/Controls.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

void declareFiberTraceFunction(py::module &mod)
{
    py::class_<FiberTraceFunction, std::shared_ptr<FiberTraceFunction>> cls(mod, "FiberTraceFunction");
    cls.def(py::init<>());
    cls.def(py::init<FiberTraceFunction const&>(), "ftf"_a);
    cls.def_readwrite("fiberTraceFunctionControl", &FiberTraceFunction::fiberTraceFunctionControl);
    cls.def_readwrite("xCenter", &FiberTraceFunction::xCenter);
    cls.def_readwrite("yCenter", &FiberTraceFunction::yCenter);
    cls.def_readwrite("yLow", &FiberTraceFunction::yLow);
    cls.def_readwrite("yHigh", &FiberTraceFunction::yHigh);
    cls.def_readwrite("coefficients", &FiberTraceFunction::coefficients);
    cls.def("setCoefficients", &FiberTraceFunction::setCoefficients, "coeffs"_a);
}

void declareFiberTraceFunctionControl(py::module &mod)
{
    py::class_<FiberTraceFunctionControl, std::shared_ptr<FiberTraceFunctionControl>> cls(
        mod, "FiberTraceFunctionControl");
    cls.def(py::init<>());
    cls.def(py::init<FiberTraceFunctionControl const&>(), "ftfc"_a);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionControl, interpolation);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionControl, order);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionControl, xLow);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionControl, xHigh);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionControl, nPixCutLeft);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionControl, nPixCutRight);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionControl, nRows);
}


void declareFiberTraceFunctionFindingControl(py::module &mod)
{
    py::class_<FiberTraceFunctionFindingControl,
               std::shared_ptr<FiberTraceFunctionFindingControl>> cls(
        mod, "FiberTraceFunctionFindingControl");
    cls.def(py::init<>());
    cls.def(py::init<FiberTraceFunctionFindingControl const&>(), "ftffc"_a);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionFindingControl, fiberTraceFunctionControl);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionFindingControl, apertureFWHM);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionFindingControl, signalThreshold);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionFindingControl, nTermsGaussFit);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionFindingControl, saturationLevel);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionFindingControl, minLength);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionFindingControl, maxLength);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionFindingControl, nLost);
}

void declareFiberTraceProfileFittingControl(py::module &mod)
{
    py::class_<FiberTraceProfileFittingControl,
               std::shared_ptr<FiberTraceProfileFittingControl>> cls(
        mod, "FiberTraceProfileFittingControl");
    cls.def(py::init<>());
    cls.def(py::init<FiberTraceProfileFittingControl const&>(), "ftpfc"_a);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceProfileFittingControl, swathWidth);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceProfileFittingControl, overSample);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceProfileFittingControl, maxIterSig);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceProfileFittingControl, lowerSigma);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceProfileFittingControl, upperSigma);
}

void declareDispCorControl(py::module &mod)
{
    py::class_<DispCorControl, std::shared_ptr<DispCorControl>> cls(mod, "DispCorControl");
    cls.def(py::init<>());
    cls.def(py::init<DispCorControl const&>(), "control"_a);
    LSST_DECLARE_CONTROL_FIELD(cls, DispCorControl, order);
    LSST_DECLARE_CONTROL_FIELD(cls, DispCorControl, searchRadius);
    LSST_DECLARE_CONTROL_FIELD(cls, DispCorControl, fwhm);
    LSST_DECLARE_CONTROL_FIELD(cls, DispCorControl, maxDistance);
}


PYBIND11_PLUGIN(controls) {
    py::module mod("controls");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    declareFiberTraceFunction(mod);
    declareFiberTraceFunctionControl(mod);
    declareFiberTraceFunctionFindingControl(mod);
    declareFiberTraceProfileFittingControl(mod);
    declareDispCorControl(mod);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
