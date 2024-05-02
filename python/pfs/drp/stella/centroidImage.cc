#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/utils/python.h"
#include "pfs/drp/stella/centroidImage.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


template <typename T>
void defineCentroidImage(py::module & mod) {
    mod.def("findPeak", &findPeak, "image"_a, "center"_a, "halfWidth"_a, "badBitMask"_a=0);
    mod.def("centroidImage",
            py::overload_cast<lsst::afw::image::Image<T> const&,
                              std::shared_ptr<lsst::afw::detection::Psf>>(&centroidImage<T>),
            "image"_a, "psf"_a);
    mod.def("centroidImage",
            py::overload_cast<lsst::afw::image::Image<T> const&, float, float>(&centroidImage<T>),
            "image"_a, "sigma"_a, "nSigma"_a=3.0);
    mod.def("centroidImage",
            py::overload_cast<lsst::afw::image::Image<T> const&, float>(&centroidImage<T>),
            "image"_a, "nSigma"_a=3.0);
}


PYBIND11_PLUGIN(centroidImage) {
    py::module mod("centroidImage");
    mod.def("centroidExposure",
            py::overload_cast<lsst::afw::image::Exposure<float> const&,
                              lsst::geom::Point2D const&>(&centroidExposure),
            "exposure"_a, "point"_a);
    defineCentroidImage<double>(mod);
    return mod.ptr();
}


} // anonymous namespace

}}} // pfs::drp::stella
