#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/utils/python.h"

#include "pfs/drp/stella/traces.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

void declareTracePeak(py::module &mod) {
    py::class_<TracePeak, std::shared_ptr<TracePeak>> cls(mod, "TracePeak");
    cls.def(py::init<int, int, double, int, double, float, float>(),
            "row"_a, "low"_a, "peak"_a, "high"_a, "peakErr"_a=NaN, "flux"_a=NaN, "fluxErr"_a=NaN);
    cls.def_readonly("span", &TracePeak::span);
    cls.def_readonly("peak", &TracePeak::peak);
    cls.def_readonly("peakErr", &TracePeak::peakErr);
    cls.def_readonly("flux", &TracePeak::flux);
    cls.def_readonly("fluxErr", &TracePeak::fluxErr);
    cls.def_property_readonly("row", [](TracePeak const& self) { return self.span.getY(); });
    cls.def_property_readonly("low", [](TracePeak const& self) { return self.span.getX0(); });
    cls.def_property_readonly("high", [](TracePeak const& self) { return self.span.getX1(); });
    lsst::utils::python::addOutputOp(cls, "__str__");
    lsst::utils::python::addOutputOp(cls, "__repr__");
}


PYBIND11_PLUGIN(traces) {
    py::module mod("traces");
    declareTracePeak(mod);
    mod.def("findTracePeaks",
            py::overload_cast<lsst::afw::image::MaskedImage<float> const&, float,
                              lsst::afw::image::MaskPixel>(&findTracePeaks),
            "image"_a, "threshold"_a, "badBitMask"_a=0);
    mod.def("findTracePeaks",
            py::overload_cast<lsst::afw::image::MaskedImage<float> const&, DetectorMap const&,
                              float, float, lsst::afw::image::MaskPixel,
                              ndarray::Array<int, 1, 1> const&>(&findTracePeaks),
            "image"_a, "detectorMap"_a, "threshold"_a, "radius"_a, "badBitMask"_a=0, "fiberId"_a=nullptr);
    mod.def("centroidPeak", &centroidPeak, "peak"_a, "image"_a, "psfSigma"_a,
            "badBitMask"_a=0, "extent"_a=3.0, "ampAst4"_a=1.33);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
