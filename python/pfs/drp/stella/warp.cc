#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/cpputils/python.h"

#include "pfs/drp/stella/warp.h"


namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


PYBIND11_MODULE(warp, mod) {
    mod.def(
        "warpFiber",
        py::overload_cast<
            lsst::afw::image::MaskedImage<float> const&, DetectorMap const&, int, int, std::string const&
        >(&warpFiber),
        "image"_a, "detectorMap"_a, "fiberId"_a, "halfWidth"_a, "warpingKernelName"_a="lanczos3"
    );
    mod.def(
        "warpFiber",
        py::overload_cast<
            lsst::afw::image::MaskedImage<float> const&,
            DetectorMap const&,
            int,
            int,
            lsst::afw::math::WarpingControl const&,
            lsst::afw::image::MaskedImage<float>::SinglePixel const&
        >(&warpFiber),
        "image"_a, "detectorMap"_a, "fiberId"_a, "halfWidth"_a, "ctrl"_a, "pad"_a
    );
    mod.def(
        "warpImage",
        py::overload_cast<
            lsst::afw::image::MaskedImage<float> const&,
            DetectorMap const&,
            DetectorMap const&,
            std::string const&,
            int
        >(&warpImage),
        "fromImage"_a,
        "fromDetectorMap"_a,
        "toDetectorMap"_a,
        "warpingKernelName"_a="lanczos3",
        "numWavelengthKnots"_a=75
    );
    mod.def(
        "warpImage",
        py::overload_cast<
            lsst::afw::image::MaskedImage<float> const&,
            DetectorMap const&,
            DetectorMap const&,
            lsst::afw::math::WarpingControl const&,
            lsst::afw::image::MaskedImage<float>::SinglePixel const&,
            int
        >(&warpImage),
        "fromImage"_a, "fromDetectorMap"_a, "toDetectorMap"_a, "ctrl"_a, "pad"_a, "numWavelengthKnots"_a=75
    );
}


} // anonymous namespace

}}} // pfs::drp::stella
