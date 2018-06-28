#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
    cls.def_readwrite("ctrl", &FiberTraceFunction::ctrl);
    cls.def_readwrite("xCenter", &FiberTraceFunction::xCenter);
    cls.def_readwrite("yCenter", &FiberTraceFunction::yCenter);
    cls.def_readwrite("yLow", &FiberTraceFunction::yLow);
    cls.def_readwrite("yHigh", &FiberTraceFunction::yHigh);
    cls.def_readwrite("coefficients", &FiberTraceFunction::coefficients);
    cls.def("setCoefficients", &FiberTraceFunction::setCoefficients, "coeffs"_a);
    cls.def("calcMinCenMax", &FiberTraceFunction::calcMinCenMax);

    cls.def("__getstate__",
            [](FiberTraceFunction const& self) {
                return py::make_tuple(self.ctrl, self.xCenter, self.yCenter,
                                      self.yLow, self.yHigh, self.coefficients);
            });
    cls.def("__setstate__",
            [](FiberTraceFunction & self, py::tuple const& t) {
                new (&self) FiberTraceFunction();
                self.ctrl = t[0].cast<FiberTraceFunctionControl>();
                self.xCenter = t[1].cast<float>();
                self.yCenter = t[2].cast<std::size_t>();
                self.yLow = t[3].cast<std::ptrdiff_t>();
                self.yHigh = t[4].cast<std::ptrdiff_t>();
                self.coefficients = t[5].cast<ndarray::Array<float, 1, 1>>();
            });
}


void declareFiberTraceFunctionControl(py::module &mod)
{
    py::class_<FiberTraceFunctionControl, std::shared_ptr<FiberTraceFunctionControl>> cls(
        mod, "FiberTraceFunctionControl");
    cls.def(py::init<>());
    cls.def(py::init<FiberTraceFunctionControl const&>(), "ctrl"_a);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionControl, order);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionControl, xLow);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionControl, xHigh);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionControl, nPixCutLeft);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionControl, nPixCutRight);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFunctionControl, nRows);

    cls.def("__getstate__",
            [](FiberTraceFunctionControl const& self) {
                return py::make_tuple(self.order, self.xLow, self.xHigh, self.nPixCutLeft,
                                      self.nPixCutRight, self.nRows);
            });
    cls.def("__setstate__",
            [](FiberTraceFunctionControl & self, py::tuple const& t) {
                new (&self) FiberTraceFunctionControl();
                self.order = t[0].cast<int>();
                self.xLow = t[1].cast<float>();
                self.xHigh = t[2].cast<float>();
                self.nPixCutLeft = t[3].cast<int>();
                self.nPixCutRight = t[4].cast<int>();
                self.nRows = t[5].cast<int>();
            });
}


void declareFiberTraceFindingControl(py::module &mod)
{
    py::class_<FiberTraceFindingControl, std::shared_ptr<FiberTraceFindingControl>> cls(
        mod, "FiberTraceFindingControl");
    cls.def(py::init<>());
    cls.def(py::init<FiberTraceFindingControl const&>(), "ctrl"_a);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFindingControl, apertureFwhm);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFindingControl, signalThreshold);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFindingControl, nTermsGaussFit);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFindingControl, saturationLevel);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFindingControl, minLength);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFindingControl, maxLength);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceFindingControl, nLost);

    cls.def("__getstate__",
            [](FiberTraceFindingControl const& self) {
                return py::make_tuple(self.apertureFwhm, self.signalThreshold,
                                      self.nTermsGaussFit, self.saturationLevel,
                                      self.minLength, self.maxLength, self.nLost);
            });
    cls.def("__setstate__",
            [](FiberTraceFindingControl & self, py::tuple const& t) {
                new (&self) FiberTraceFindingControl();
                self.apertureFwhm = t[0].cast<float>();
                self.signalThreshold = t[1].cast<float>();
                self.nTermsGaussFit = t[2].cast<int>();
                self.saturationLevel = t[3].cast<float>();
                self.minLength = t[4].cast<int>();
                self.maxLength = t[5].cast<int>();
                self.nLost = t[6].cast<int>();
            });
}


void declareFiberTraceProfileFittingControl(py::module &mod)
{
    py::class_<FiberTraceProfileFittingControl,
               std::shared_ptr<FiberTraceProfileFittingControl>> cls(
        mod, "FiberTraceProfileFittingControl");
    cls.def(py::init<>());
    cls.def(py::init<FiberTraceProfileFittingControl const&>(), "ctrl"_a);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceProfileFittingControl, swathWidth);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceProfileFittingControl, overSample);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceProfileFittingControl, maxIterSig);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceProfileFittingControl, lowerSigma);
    LSST_DECLARE_CONTROL_FIELD(cls, FiberTraceProfileFittingControl, upperSigma);

    cls.def("__getstate__",
            [](FiberTraceProfileFittingControl const& self) {
                return py::make_tuple(self.swathWidth, self.overSample, self.maxIterSig,
                                      self.lowerSigma, self.upperSigma);
            });
    cls.def("__setstate__",
            [](FiberTraceProfileFittingControl & self, py::tuple const& t) {
                new (&self) FiberTraceProfileFittingControl();
                self.swathWidth = t[0].cast<int>();
                self.overSample = t[1].cast<int>();
                self.maxIterSig = t[2].cast<int>();
                self.lowerSigma = t[3].cast<float>();
                self.upperSigma = t[4].cast<float>();
            });
}


void declareDispCorControl(py::module &mod)
{
    using Class = DispersionCorrectionControl;
    py::class_<DispersionCorrectionControl,
               std::shared_ptr<DispersionCorrectionControl>> cls(
        mod, "DispersionCorrectionControl");
    cls.def(py::init<>());
    cls.def(py::init<DispersionCorrectionControl const&>(), "ctrl"_a);
    LSST_DECLARE_CONTROL_FIELD(cls, DispersionCorrectionControl, order);
    LSST_DECLARE_CONTROL_FIELD(cls, DispersionCorrectionControl, searchRadius);
    LSST_DECLARE_CONTROL_FIELD(cls, DispersionCorrectionControl, fwhm);
    LSST_DECLARE_CONTROL_FIELD(cls, DispersionCorrectionControl, maxDistance);
}


PYBIND11_PLUGIN(controls) {
    py::module mod("controls");

    declareFiberTraceFunction(mod);
    declareFiberTraceFunctionControl(mod);
    declareFiberTraceFindingControl(mod);
    declareFiberTraceProfileFittingControl(mod);
    declareDispCorControl(mod);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
