#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "pfs/drp/stella/NevenPsf.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


PYBIND11_MODULE(NevenPsf, mod) {
    pybind11::module::import("pfs.drp.stella.SpectralPsf");

    py::classh<NevenPsf, SpectralPsf, OversampledPsf> cls(mod, "NevenPsf");
    cls.def(py::init<std::shared_ptr<DetectorMap>,
                     ndarray::Array<int, 1, 1> const&,
                     ndarray::Array<double, 1, 1> const&,
                     std::vector<ndarray::Array<float, 2, 1>> const&,
                     int,
                     lsst::geom::Extent2I const&>(),
            "detectorMap"_a, "fiberId"_a, "wavelength"_a, "images"_a, "oversampleFactor"_a, "targetSize"_a);

    cls.def("__len__", &NevenPsf::size);
    cls.def("size", &NevenPsf::size);
    cls.def("getFiberId", &NevenPsf::getFiberId);
    cls.def("getWavelength", &NevenPsf::getWavelength);
    cls.def("getImages", &NevenPsf::getImages);
}

} // anonymous namespace

}}} // pfs::drp::stella
