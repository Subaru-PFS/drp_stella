#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "pfs/drp/stella/NevenPsf.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs {
namespace drp {
namespace stella {
namespace tests {

class NevenPsf : public pfs::drp::stella::NevenPsf {
  public:
    // Convert from regular NevenPsf
    NevenPsf(pfs::drp::stella::NevenPsf const& psf) : pfs::drp::stella::NevenPsf(psf) {}

    // Expose protected method
    std::shared_ptr<OversampledPsf::Image> computeOversampledKernelImage(
        int fiberId,
        double wavelength
    ) const {
        return computeOversampledKernelImage(getDetectorMap()->findPoint(fiberId, wavelength));
    }
    std::shared_ptr<OversampledPsf::Image> computeOversampledKernelImage(
        lsst::geom::Point2D const& position
    ) const {
        return doComputeOversampledKernelImage(position);
    }
};


namespace {

PYBIND11_PLUGIN(nevenPsf) {
    py::module mod("nevenPsf");
    pybind11::module::import("pfs.drp.stella.NevenPsf");

    py::class_<NevenPsf, std::shared_ptr<NevenPsf>,
               pfs::drp::stella::NevenPsf, SpectralPsf, OversampledPsf> cls(mod, "NevenPsf");
    cls.def(py::init<pfs::drp::stella::NevenPsf const&>());

    cls.def("computeOversampledKernelImage",
            py::overload_cast<int, double>(&NevenPsf::computeOversampledKernelImage, py::const_));
    cls.def("computeOversampledKernelImage",
            py::overload_cast<lsst::geom::Point2D const&>(
                &NevenPsf::computeOversampledKernelImage, py::const_));

    return mod.ptr();
}

} // anonymous namespace


}}}}
