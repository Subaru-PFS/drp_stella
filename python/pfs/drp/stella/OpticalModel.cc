#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "pfs/drp/stella/OpticalModel.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declareSlitModel(py::module_ & mod) {
    py::class_<SlitModel> cls(mod, "SlitModel");
    cls.def(
        py::init<
            OpticalModel::Array1I const&,
            double, double,
            OpticalModel::Array1D const&, OpticalModel::Array1D const&,
            OpticalModel::DistortionList const&
        >(),
        "fiberId"_a,
        "fiberPitch"_a, "wavelengthDispersion"_a,
        "spatialOffsets"_a, "spectralOffsets"_a,
        "distortions"_a=OpticalModel::DistortionList()
    );
    cls.def(
        py::init<SplinedDetectorMap const&, OpticalModel::DistortionList const&>(),
        "source"_a, "distortions"_a=OpticalModel::DistortionList()
    );

    cls.def("getFiberId", &SlitModel::getFiberId);
    cls.def("getFiberPitch", &SlitModel::getFiberPitch);
    cls.def("getWavelengthDispersion", &SlitModel::getWavelengthDispersion);
    cls.def("getSpatialOffsets", &SlitModel::getSpatialOffsets);
    cls.def("getSpectralOffsets", &SlitModel::getSpectralOffsets);
    cls.def("getDistortions", &SlitModel::getDistortions);

    cls.def("getSpatialOffset", &SlitModel::getSpatialOffset, "fiberId"_a);
    cls.def("getSpectralOffset", &SlitModel::getSpectralOffset, "fiberId"_a);

    cls.def(
        "spectrographToSlit",
        py::overload_cast<int, double>(&SlitModel::spectrographToSlit, py::const_),
        "fiberId"_a, "wavelength"_a
    );
    cls.def(
        "spectrographToSlit",
        py::overload_cast<OpticalModel::Array1I const&, OpticalModel::Array1D const&>(&SlitModel::spectrographToSlit, py::const_),
        "fiberId"_a, "wavelength"_a
    );
}


void declareOpticalModel(py::module_ & mod) {
    py::class_<OpticalModel> cls(mod, "OpticalModel");
    cls.def(
        py::init<
            OpticalModel::Array2D const&,
            OpticalModel::Array2D const&,
            OpticalModel::Array2D const&,
            OpticalModel::Array2D const&,
            OpticalModel::DistortionList const&
        >(),
        "spatial"_a, "spectral"_a, "x"_a, "wavelength"_a, "distortions"_a
    );
    cls.def(
        py::init<SplinedDetectorMap const&, OpticalModel::DistortionList const&>(),
        "source"_a, "distortions"_a=OpticalModel::DistortionList()
    );

    cls.def("getSpatial", &OpticalModel::getSpatial);
    cls.def("getSpectral", &OpticalModel::getSpectral);
    cls.def("getX", &OpticalModel::getX);
    cls.def("getY", &OpticalModel::getY);
    cls.def("getSlitToDetector", &OpticalModel::getSlitToDetector);
    cls.def("getDetectorToSlit", &OpticalModel::getDetectorToSlit);
    cls.def("getDistortions", &OpticalModel::getDistortions);

    cls.def(
        "slitToDetector",
        py::overload_cast<double, double>(&OpticalModel::slitToDetector, py::const_),
        "spatial"_a, "spectral"_a
    );
    cls.def(
        "slitToDetector",
        py::overload_cast<
            OpticalModel::Array1D const&, OpticalModel::Array1D const&
        >(&OpticalModel::slitToDetector, py::const_),
        "spatial"_a, "spectral"_a
    );
    cls.def(
        "slitToDetector",
        py::overload_cast<lsst::geom::Point2D const&>(&OpticalModel::slitToDetector, py::const_),
        "xy"_a
    );

    cls.def(
        "detectorToSlit",
        py::overload_cast<double, double>(&OpticalModel::detectorToSlit, py::const_),
        "x"_a, "y"_a
    );
    cls.def(
        "detectorToSlit",
        py::overload_cast<
            OpticalModel::Array1D const&, OpticalModel::Array1D const&
        >(&OpticalModel::detectorToSlit, py::const_),
        "x"_a, "y"_a
    );
    cls.def(
        "detectorToSlit",
        py::overload_cast<lsst::geom::Point2D const&>(&OpticalModel::detectorToSlit, py::const_),
        "xy"_a
    );
}


void declareDetectorModel(py::module_ & mod) {
    py::class_<DetectorModel> cls(mod, "DetectorModel");
    cls.def(
        py::init<
            lsst::geom::Box2I const&,
            bool,
            lsst::geom::AffineTransform const&,
            OpticalModel::DistortionList const&
        >(),
        "bbox"_a,
        "isDivided"_a,
        "rightCcd"_a=lsst::geom::AffineTransform(),
        "distortions"_a=OpticalModel::DistortionList()
    );

    cls.def("getBBox", &DetectorModel::getBBox);
    cls.def("getIsDivided", &DetectorModel::getIsDivided);
    cls.def("getRightCcd", &DetectorModel::getRightCcd);
    cls.def("getDistortions", &DetectorModel::getDistortions);

    cls.def(
        "detectorToPixels",
        py::overload_cast<double, double>(&DetectorModel::detectorToPixels, py::const_),
        "x"_a, "y"_a
    );
    cls.def(
        "detectorToPixels",
        py::overload_cast<
            OpticalModel::Array1D const&, OpticalModel::Array1D const&
        >(&DetectorModel::detectorToPixels, py::const_),
        "x"_a, "y"_a
    );
    cls.def(
        "detectorToPixels",
        py::overload_cast<lsst::geom::Point2D const&>(&DetectorModel::detectorToPixels, py::const_),
        "xy"_a
    );

    cls.def(
        "pixelsToDetector",
        py::overload_cast<double, double>(&DetectorModel::pixelsToDetector, py::const_),
        "p"_a, "q"_a
    );
    cls.def(
        "pixelsToDetector",
        py::overload_cast<
            OpticalModel::Array1D const&, OpticalModel::Array1D const&
        >(&DetectorModel::pixelsToDetector, py::const_),
        "p"_a, "q"_a
    );
    cls.def(
        "pixelsToDetector",
        py::overload_cast<lsst::geom::Point2D const&>(&DetectorModel::pixelsToDetector, py::const_),
        "pq"_a
    );
}


PYBIND11_MODULE(OpticalModel, mod) {
    py::module::import("pfs.drp.stella.GridTransform");  // for OpticalModel::slitToDetector etc.
    declareSlitModel(mod);
    declareOpticalModel(mod);
    declareDetectorModel(mod);
}


} // anonymous namespace

}}} // pfs::drp::stella
