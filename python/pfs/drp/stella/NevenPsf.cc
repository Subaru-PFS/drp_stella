#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "pfs/drp/stella/NevenPsf.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


PYBIND11_PLUGIN(NevenPsf) {
    py::module mod("NevenPsf");
    pybind11::module::import("pfs.drp.stella.SpectralPsf");

    py::class_<NevenPsf, std::shared_ptr<NevenPsf>, SpectralPsf, OversampledPsf> cls(mod, "NevenPsf");
    cls.def(py::init<std::shared_ptr<DetectorMap>,
                     ndarray::Array<float const, 1, 1> const&,
                     ndarray::Array<float const, 1, 1> const&,
                     std::vector<ndarray::Array<double const, 2, 1>> const&,
                     int,
                     lsst::geom::Extent2I const&,
                     float>(),
            "detectorMap"_a, "x"_a, "y"_a, "images"_a, "oversampleFactor"_a, "targetSize"_a,
            "xMaxDistance"_a=20.0);

    cls.def("getX", &NevenPsf::getX);
    cls.def("getY", &NevenPsf::getY);
    cls.def("getImages", &NevenPsf::getImages);
    cls.def("getXMaxDistance", &NevenPsf::getXMaxDistance);

    cls.def_property_readonly("x", &NevenPsf::getX);
    cls.def_property_readonly("y", &NevenPsf::getY);
    cls.def_property_readonly("images", &NevenPsf::getImages);
    cls.def_property_readonly("xMaxDistance", &NevenPsf::getXMaxDistance);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
