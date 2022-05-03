#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/base.h"
#include "lsst/utils/python.h"
#include "pfs/drp/stella/fitLine.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

PYBIND11_PLUGIN(fitLine) {
    py::module mod("fitLine");

    py::class_<FitLineResult> cls(mod, "FitLineResult");
    cls.def_readonly("rms", &FitLineResult::rms);
    cls.def_readonly("isValid", &FitLineResult::isValid);
    cls.def_readonly("num", &FitLineResult::num);
    cls.def_readonly("amplitude", &FitLineResult::amplitude);
    cls.def_readonly("center", &FitLineResult::center);
    cls.def_readonly("rmsSize", &FitLineResult::rmsSize);
    cls.def_readonly("bg0", &FitLineResult::bg0);
    cls.def_readonly("bg1", &FitLineResult::bg1);
    cls.def_readonly("amplitudeErr", &FitLineResult::amplitudeErr);
    cls.def_readonly("centerErr", &FitLineResult::centerErr);
    cls.def_readonly("rmsSizeErr", &FitLineResult::rmsSizeErr);
    cls.def_readonly("bg0Err", &FitLineResult::bg0Err);
    cls.def_readonly("bg1Err", &FitLineResult::bg1Err);
    lsst::utils::python::addOutputOp(cls, "__str__");
    lsst::utils::python::addOutputOp(cls, "__repr__");

    mod.def("fitLine",
            py::overload_cast<ndarray::Array<Spectrum::ImageT const, 1, 1> const&,
                              ndarray::Array<lsst::afw::image::MaskPixel const, 1, 1> const&,
                              float, float, lsst::afw::image::MaskPixel,
                              std::size_t>(&fitLine<Spectrum::ImageT>),
            "flux"_a, "mask"_a, "peakPosition"_a, "rmsSize"_a, "badBitMask"_a,
            "fittingHalfSize"_a=0);

    // pybind11 has trouble recognising the vanilla function (possibly because of the templated function?)
    mod.def("fitLine", [](Spectrum const& spectrum, float peakPosition, float rmsSize,
                          lsst::afw::image::MaskPixel badBitMask, std::size_t fittingHalfSize) {
                              return fitLine(spectrum, peakPosition, rmsSize, badBitMask, fittingHalfSize); },
            "spectrum"_a, "peakPosition"_a, "rmsSize"_a, "badBitMask"_a, "fittingHalfSize"_a=0);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
