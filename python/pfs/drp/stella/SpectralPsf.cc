#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/geom/Point.h"
#include "pfs/drp/stella/SpectralPsf.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declareSpectralPsf(py::module & mod) {
    using Class = SpectralPsf;
    py::class_<Class, std::shared_ptr<Class>, lsst::afw::detection::Psf> cls(mod, "SpectralPsf");
    cls.def("getDetectorMap", &Class::getDetectorMap);
    cls.def_property_readonly("detectorMap", &Class::getDetectorMap);
    cls.def("computeImage", py::overload_cast<int, float>(&Class::computeImage, py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("computeImage", py::overload_cast<lsst::geom::Point2D const&>(&Class::computeImage, py::const_),
            "position"_a);
    cls.def("computeKernelImage", py::overload_cast<int, float>(&Class::computeKernelImage, py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("computeKernelImage",
            py::overload_cast<lsst::geom::Point2D const&>(&Class::computeKernelImage, py::const_),
            "position"_a);
    cls.def("computePeak", py::overload_cast<int, float>(&Class::computePeak, py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("computePeak", py::overload_cast<lsst::geom::Point2D const&>(&Class::computePeak, py::const_),
            "position"_a);
    cls.def("computeApertureFlux",
            py::overload_cast<double, int, float>(&Class::computeApertureFlux, py::const_),
            "radius"_a, "fiberId"_a, "wavelength"_a);
    cls.def("computeApertureFlux",
             py::overload_cast<double, lsst::geom::Point2D const&>(&Class::computeApertureFlux, py::const_),
            "radius"_a, "position"_a);
    cls.def("computeShape", py::overload_cast<int, float>(&Class::computeShape, py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("computeShape", py::overload_cast<lsst::geom::Point2D const&>(&Class::computeShape, py::const_),
            "position"_a);
    cls.def("getLocalKernel", py::overload_cast<int, float>(&Class::getLocalKernel, py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("getLocalKernel",
            py::overload_cast<lsst::geom::Point2D const&>(&Class::getLocalKernel, py::const_),
            "position"_a);
    cls.def("computeBBox", py::overload_cast<int, float>(&Class::computeBBox, py::const_),
            "fiberId"_a, "wavelength"_a);
    cls.def("computeBBox", py::overload_cast<lsst::geom::Point2D const&>(&Class::computeBBox, py::const_),
            "position"_a);
}


void declareOversampledPsf(py::module & mod) {
    using Class = OversampledPsf;
    py::class_<Class, std::shared_ptr<Class>, lsst::afw::detection::Psf> cls(mod, "OversampledPsf");
    cls.def("getOversampleFactor", &Class::getOversampleFactor);
    cls.def_property_readonly("oversampleFactor", &Class::getOversampleFactor);
    cls.def("getTargetSize", &Class::getTargetSize);
    cls.def_property_readonly("targetSize", &Class::getTargetSize);
}


void declareImagingSpectralPsf(py::module & mod) {
    using Class = ImagingSpectralPsf;
    py::class_<Class, std::shared_ptr<Class>, SpectralPsf> cls(mod, "ImagingSpectralPsf");
    cls.def(py::init<std::shared_ptr<lsst::afw::detection::Psf> const, DetectorMap const&>(),
            "psf"_a, "detectorMap"_a);
    cls.def("getBase", &Class::getBase);
    cls.def_property_readonly("imagePsf", &Class::getBase);
}


PYBIND11_PLUGIN(SpectralPsf) {
    py::module mod("SpectralPsf");
    pybind11::module::import("lsst.afw.detection");
    pybind11::module::import("pfs.drp.stella.detectorMapContinued");
    declareSpectralPsf(mod);
    declareOversampledPsf(mod);
    declareImagingSpectralPsf(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella