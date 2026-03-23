#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "pfs/drp/stella/OpticalModel.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

//@{
/// Declare a function that requires a specific array type
template <typename T, int C, typename Cls>
void defineArrayFunction(
    py::class_<Cls> & cls, char const* funcName,
    typename Cls::Array2D (Cls::*function)(typename Cls::Array1I const&, typename Cls::Array1D const&) const,
    char const* name1, char const* name2
) {
    cls.def(
        funcName,
        [function](
            Cls const& self, ndarray::Array<int, 1, C> const& one, ndarray::Array<T, 1, C> const& two
        ) {
            return (self.*function)(utils::convertArray<int>(one), utils::convertArray<double>(two));
        },
        (name1 + std::string("_a")).c_str(),
        (name2 + std::string("_a")).c_str()
    );
}
template <typename T, int C, typename Cls>
void defineArrayFunction(
    py::class_<Cls> & cls, char const* funcName,
    typename Cls::Array2D (Cls::*function)(typename Cls::Array1D const&, typename Cls::Array1D const&) const,
    char const* name1, char const* name2
) {
    cls.def(
        funcName,
        [function](Cls const& self, ndarray::Array<T, 1, C> const& one, ndarray::Array<T, 1, C> const& two) {
            return (self.*function)(
                utils::convertArray<double>(one), utils::convertArray<double>(two)
            );
        },
        (name1 + std::string("_a")).c_str(),
        (name2 + std::string("_a")).c_str()
    );
}
//@}


void declareSlitModel(py::module_ & mod) {
    py::class_<SlitModel> cls(mod, "SlitModel");
    cls.def(
        py::init<
            OpticsModel::Array1I const&,
            double, double, double, double,
            OpticsModel::Array1D const&, OpticsModel::Array1D const&,
            OpticsModel::DistortionList const&
        >(),
        "fiberId"_a,
        "fiberPitch"_a, "fiberCenter"_a, "wavelengthDispersion"_a, "wavelengthCenter"_a,
        "spatialOffsets"_a, "spectralOffsets"_a,
        "distortions"_a=OpticsModel::DistortionList()
    );
    cls.def(
        py::init<SplinedDetectorMap const&, OpticsModel::DistortionList const&>(),
        "source"_a, "distortions"_a=OpticsModel::DistortionList()
    );
    cls.def("copy", &SlitModel::copy);
    cls.def("withDistortion", &SlitModel::withDistortion, "distortion"_a);
    cls.def("withoutDistortion", &SlitModel::withoutDistortion);

    cls.def("getFiberId", &SlitModel::getFiberId);
    cls.def("getFiberPitch", &SlitModel::getFiberPitch);
    cls.def("getFiberMin", &SlitModel::getFiberMin);
    cls.def("getWavelengthDispersion", &SlitModel::getWavelengthDispersion);
    cls.def("getWavelengthMin", &SlitModel::getWavelengthMin);
    cls.def("getSpatialOffsets", &SlitModel::getSpatialOffsets);
    cls.def("getSpectralOffsets", &SlitModel::getSpectralOffsets);
    cls.def("getDistortions", &SlitModel::getDistortions);
    cls.def("getSpatialOffset", &SlitModel::getSpatialOffset, "fiberId"_a);
    cls.def("getSpectralOffset", &SlitModel::getSpectralOffset, "fiberId"_a);
    cls.def_property_readonly("fiberId", &SlitModel::getFiberId);
    cls.def_property_readonly("fiberPitch", &SlitModel::getFiberPitch);
    cls.def_property_readonly("fiberMin", &SlitModel::getFiberMin);
    cls.def_property_readonly("wavelengthDispersion", &SlitModel::getWavelengthDispersion);
    cls.def_property_readonly("wavelengthMin", &SlitModel::getWavelengthMin);
    cls.def_property_readonly("spatialOffsets", &SlitModel::getSpatialOffsets);
    cls.def_property_readonly("spectralOffsets", &SlitModel::getSpectralOffsets);
    cls.def_property_readonly("distortions", &SlitModel::getDistortions);

    cls.def(
        "spectrographToSlit",
        py::overload_cast<int, double>(&SlitModel::spectrographToSlit, py::const_),
        "fiberId"_a, "wavelength"_a
    );
    cls.def(
        "spectrographToSlit",
        py::overload_cast<OpticsModel::Array1I const&, OpticsModel::Array1D const&>(&SlitModel::spectrographToSlit, py::const_),
        "fiberId"_a, "wavelength"_a
    );
    defineArrayFunction<float, 1>(
        cls, "spectrographToSlit",
        py::overload_cast<OpticsModel::Array1I const&, OpticsModel::Array1D const&>(
            &SlitModel::spectrographToSlit, py::const_
        ),
        "fiberId", "wavelength"
    );
    defineArrayFunction<float, 0>(
        cls, "spectrographToSlit",
        py::overload_cast<OpticsModel::Array1I const&, OpticsModel::Array1D const&>(
            &SlitModel::spectrographToSlit, py::const_
        ),
        "fiberId", "wavelength"
    );
    defineArrayFunction<double, 0>(
        cls, "spectrographToSlit",
        py::overload_cast<OpticsModel::Array1I const&, OpticsModel::Array1D const&>(
            &SlitModel::spectrographToSlit, py::const_
        ),
        "fiberId", "wavelength"
    );
}


void declareOpticsModel(py::module_ & mod) {
    py::module::import("pfs.drp.stella.GridTransform");  // for slitToDetector, detectorToSlit
    py::class_<OpticsModel> cls(mod, "OpticsModel");
    cls.def(
        py::init<
            OpticsModel::Array2D const&,
            OpticsModel::Array2D const&,
            OpticsModel::Array2D const&,
            OpticsModel::Array2D const&,
            OpticsModel::DistortionList const&
        >(),
        "spatial"_a, "spectral"_a, "x"_a, "wavelength"_a, "distortions"_a
    );
    cls.def(
        py::init<SplinedDetectorMap const&, OpticsModel::DistortionList const&>(),
        "source"_a, "distortions"_a=OpticsModel::DistortionList()
    );
    cls.def("copy", &OpticsModel::copy);

    cls.def("getSpatial", &OpticsModel::getSpatial);
    cls.def("getSpectral", &OpticsModel::getSpectral);
    cls.def("getX", &OpticsModel::getX);
    cls.def("getY", &OpticsModel::getY);
    cls.def("getSlitToDetector", &OpticsModel::getSlitToDetector);
    cls.def("getDetectorToSlit", &OpticsModel::getDetectorToSlit);
    cls.def("getDistortions", &OpticsModel::getDistortions);
    cls.def_property_readonly("spatial", &OpticsModel::getSpatial);
    cls.def_property_readonly("spectral", &OpticsModel::getSpectral);
    cls.def_property_readonly("x", &OpticsModel::getX);
    cls.def_property_readonly("y", &OpticsModel::getY);
    cls.def_property_readonly("distortions", &OpticsModel::getDistortions);

    cls.def(
        "slitToDetector",
        py::overload_cast<double, double>(&OpticsModel::slitToDetector, py::const_),
        "spatial"_a, "spectral"_a
    );
    cls.def(
        "slitToDetector",
        py::overload_cast<
            OpticsModel::Array1D const&, OpticsModel::Array1D const&
        >(&OpticsModel::slitToDetector, py::const_),
        "spatial"_a, "spectral"_a
    );
    cls.def(
        "slitToDetector",
        py::overload_cast<lsst::geom::Point2D const&>(&OpticsModel::slitToDetector, py::const_),
        "xy"_a
    );
    cls.def(
        "slitToDetector",
        py::overload_cast<OpticsModel::Array2D const&>(&OpticsModel::slitToDetector, py::const_),
        "spatialSpectral"_a
    );
    defineArrayFunction<float, 1>(
        cls, "slitToDetector",
        py::overload_cast<OpticsModel::Array1D const&, OpticsModel::Array1D const&>(
            &OpticsModel::slitToDetector, py::const_
        ),
        "spatial", "spectral"
    );
    defineArrayFunction<float, 0>(
        cls, "slitToDetector",
        py::overload_cast<OpticsModel::Array1D const&, OpticsModel::Array1D const&>(
            &OpticsModel::slitToDetector, py::const_
        ),
        "spatial", "spectral"
    );
    defineArrayFunction<double, 0>(
        cls, "slitToDetector",
        py::overload_cast<OpticsModel::Array1D const&, OpticsModel::Array1D const&>(
            &OpticsModel::slitToDetector, py::const_
        ),
        "spatial", "spectral"
    );

    cls.def(
        "detectorToSlit",
        py::overload_cast<double, double>(&OpticsModel::detectorToSlit, py::const_),
        "x"_a, "y"_a
    );
    cls.def(
        "detectorToSlit",
        py::overload_cast<
            OpticsModel::Array1D const&, OpticsModel::Array1D const&
        >(&OpticsModel::detectorToSlit, py::const_),
        "x"_a, "y"_a
    );
    cls.def(
        "detectorToSlit",
        py::overload_cast<lsst::geom::Point2D const&>(&OpticsModel::detectorToSlit, py::const_),
        "xy"_a
    );
    cls.def(
        "detectorToSlit",
        py::overload_cast<OpticsModel::Array2D const&>(&OpticsModel::detectorToSlit, py::const_),
        "xy"_a
    );
    defineArrayFunction<float, 1>(
        cls, "detectorToSlit",
        py::overload_cast<OpticsModel::Array1D const&, OpticsModel::Array1D const&>(
            &OpticsModel::detectorToSlit, py::const_
        ),
        "x", "y"
    );
    defineArrayFunction<float, 0>(
        cls, "detectorToSlit",
        py::overload_cast<OpticsModel::Array1D const&, OpticsModel::Array1D const&>(
            &OpticsModel::detectorToSlit, py::const_
        ),
        "x", "y"
    );
    defineArrayFunction<double, 0>(
        cls, "detectorToSlit",
        py::overload_cast<OpticsModel::Array1D const&, OpticsModel::Array1D const&>(
            &OpticsModel::detectorToSlit, py::const_
        ),
        "x", "y"
    );
}


void declareDetectorModel(py::module_ & mod) {
    py::class_<DetectorModel> cls(mod, "DetectorModel");

    // DetectorModel(Box2I, AffineTransform, DistortionList) needs to be defined first, otherwise:
    // "TypeError: object of type 'lsst.geom._geom.AffineTransform' has no len()"
    cls.def(
        py::init<
            lsst::geom::Box2I const&,
            lsst::geom::AffineTransform const&,
            OpticsModel::DistortionList const&
        >(),
        "bbox"_a,
        "rightCcd"_a,
        "distortions"_a=OpticsModel::DistortionList()
    );
    cls.def(
        py::init<
            lsst::geom::Box2I const&,
            OpticsModel::DistortionList const&
        >(),
        "bbox"_a,
        "distortions"_a=OpticsModel::DistortionList()
    );
    cls.def("copy", &DetectorModel::copy);

    cls.def("getBBox", &DetectorModel::getBBox);
    cls.def("getIsDivided", &DetectorModel::getIsDivided);
    cls.def("getRightCcd", &DetectorModel::getRightCcd);
    cls.def("getDistortions", &DetectorModel::getDistortions);
    cls.def_property_readonly("bbox", &DetectorModel::getBBox);
    cls.def_property_readonly("isDivided", &DetectorModel::getIsDivided);
    cls.def_property_readonly("rightCcd", &DetectorModel::getRightCcd);
    cls.def_property_readonly("distortions", &DetectorModel::getDistortions);

    cls.def(
        "detectorToPixels",
        py::overload_cast<double, double>(&DetectorModel::detectorToPixels, py::const_),
        "x"_a, "y"_a
    );
    cls.def(
        "detectorToPixels",
        py::overload_cast<
            OpticsModel::Array1D const&, OpticsModel::Array1D const&
        >(&DetectorModel::detectorToPixels, py::const_),
        "x"_a, "y"_a
    );
    cls.def(
        "detectorToPixels",
        py::overload_cast<lsst::geom::Point2D const&>(&DetectorModel::detectorToPixels, py::const_),
        "xy"_a
    );
    cls.def(
        "detectorToPixels",
        py::overload_cast<OpticsModel::Array2D const&>(&DetectorModel::detectorToPixels, py::const_),
        "xy"_a
    );
    defineArrayFunction<float, 1>(
        cls, "detectorToPixels",
        py::overload_cast<OpticsModel::Array1D const&, OpticsModel::Array1D const&>(
            &DetectorModel::detectorToPixels, py::const_
        ),
        "x", "y"
    );
    defineArrayFunction<float, 0>(
        cls, "detectorToPixels",
        py::overload_cast<OpticsModel::Array1D const&, OpticsModel::Array1D const&>(
            &DetectorModel::detectorToPixels, py::const_
        ),
        "x", "y"
    );
    defineArrayFunction<double, 0>(
        cls, "detectorToPixels",
        py::overload_cast<OpticsModel::Array1D const&, OpticsModel::Array1D const&>(
            &DetectorModel::detectorToPixels, py::const_
        ),
        "x", "y"
    );
    cls.def(
        "pixelsToDetector",
        py::overload_cast<double, double>(&DetectorModel::pixelsToDetector, py::const_),
        "p"_a, "q"_a
    );
    cls.def(
        "pixelsToDetector",
        py::overload_cast<
            OpticsModel::Array1D const&, OpticsModel::Array1D const&
        >(&DetectorModel::pixelsToDetector, py::const_),
        "p"_a, "q"_a
    );
    cls.def(
        "pixelsToDetector",
        py::overload_cast<lsst::geom::Point2D const&>(&DetectorModel::pixelsToDetector, py::const_),
        "pq"_a
    );
    cls.def(
        "pixelsToDetector",
        py::overload_cast<OpticsModel::Array2D const&>(&DetectorModel::pixelsToDetector, py::const_),
        "pq"_a
    );
    defineArrayFunction<float, 1>(
        cls, "pixelsToDetector",
        py::overload_cast<OpticsModel::Array1D const&, OpticsModel::Array1D const&>(
            &DetectorModel::pixelsToDetector, py::const_
        ),
        "p", "q"
    );
    defineArrayFunction<float, 0>(
        cls, "pixelsToDetector",
        py::overload_cast<OpticsModel::Array1D const&, OpticsModel::Array1D const&>(
            &DetectorModel::pixelsToDetector, py::const_
        ),
        "p", "q"
    );
    defineArrayFunction<double, 0>(
        cls, "pixelsToDetector",
        py::overload_cast<OpticsModel::Array1D const&, OpticsModel::Array1D const&>(
            &DetectorModel::pixelsToDetector, py::const_
        ),
        "p", "q"
    );
}


PYBIND11_MODULE(OpticalModel, mod) {
    declareSlitModel(mod);
    declareOpticsModel(mod);
    declareDetectorModel(mod);
}


} // anonymous namespace

}}} // pfs::drp::stella
