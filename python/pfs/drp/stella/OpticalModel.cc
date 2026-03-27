#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "pfs/drp/stella/OpticalModel.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


/// Declare a function that requires specific array types
///
/// Converts the input arrays to the required types.
template <typename Input1, typename Input2, int C, typename Cls, typename T1, typename T2>
void defineArrayFunction(
    py::class_<Cls> & cls, char const* funcName,
    typename ndarray::Array<double, 2, 1> (Cls::*function)(
        typename ndarray::Array<T1, 1, 1> const&, typename ndarray::Array<T2, 1, 1> const&
    ) const,
    char const* name1, char const* name2
) {
    cls.def(
        funcName,
        [function](
            Cls const& self, ndarray::Array<Input1, 1, C> const& one, ndarray::Array<Input2, 1, C> const& two
        ) {
            return (self.*function)(utils::convertArray<T1>(one), utils::convertArray<T2>(two));
        },
        py::arg(name1),
        py::arg(name2)
    );
}


/// Define overloads for methods that map between coordinate systems
///
/// This defines overloads for each of the methods, and adds array conversions.
template <typename Cls, typename Output, typename Input1, typename Input2>
void defineOverloads(
    py::class_<Cls> & cls,
    char const* funcName,
    lsst::geom::Point2D (Cls::*scalars)(Input1, Input2) const,
    lsst::geom::Point2D (Cls::*point)(lsst::geom::Point2D const&) const,
    ndarray::Array<Output, 2, 1> (Cls::*arrays)(
        ndarray::Array<Input1, 1, 1> const&, ndarray::Array<Input2, 1, 1> const&
    ) const,
    ndarray::Array<Output, 2, 1> (Cls::*array2d)(ndarray::Array<Input1, 2, 1> const&) const,
     char const* name1, char const* name2, char const* names
) {
    cls.def(funcName, scalars, py::arg(name1), py::arg(name2));
    cls.def(funcName, arrays, py::arg(name1), py::arg(name2));
    defineArrayFunction<float, float, 1>(cls, funcName, arrays, name1, name2);
    defineArrayFunction<float, float, 0>(cls, funcName, arrays, name1, name2);
    defineArrayFunction<double, double, 0>(cls, funcName, arrays, name1, name2);
    cls.def(funcName, point, py::arg(names));
    cls.def(funcName, array2d, py::arg(names));
}


/// Specialization for functions that take int,double instead of double,double
///
/// In that case, we don't have the Point2D or Array2D inputs.
template <typename Cls, typename Output>
void defineOverloads(
    py::class_<Cls> & cls,
    char const* funcName,
    lsst::geom::Point2D (Cls::*scalars)(int, double) const,
    ndarray::Array<Output, 2, 1> (Cls::*arrays)(
        ndarray::Array<int, 1, 1> const&, ndarray::Array<double, 1, 1> const&
    ) const,
     char const* name1, char const* name2
) {
    cls.def(funcName, scalars, py::arg(name1), py::arg(name2));
    cls.def(funcName, arrays, py::arg(name1), py::arg(name2));
    defineArrayFunction<int, float, 1>(cls, funcName, arrays, name1, name2);
    defineArrayFunction<int, float, 0>(cls, funcName, arrays, name1, name2);
    defineArrayFunction<int, double, 0>(cls, funcName, arrays, name1, name2);
}


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

    defineOverloads(
        cls,
        "spectrographToSlit",
        &SlitModel::spectrographToSlit,
        &SlitModel::spectrographToSlit,
        "fiberId", "wavelength"
    );
    defineOverloads(
        cls,
        "spectrographToPreSlit",
        &SlitModel::spectrographToPreSlit,
        &SlitModel::spectrographToPreSlit,
        "fiberId", "wavelength"
    );
    defineOverloads(
        cls,
        "preSlitToSlit",
        &SlitModel::preSlitToSlit,
        &SlitModel::preSlitToSlit,
        &SlitModel::preSlitToSlit,
        &SlitModel::preSlitToSlit,
        "spatial", "spectral", "spatialSpectral"
    );
    defineOverloads(
        cls,
        "slitToPreSlit",
        &SlitModel::slitToPreSlit,
        &SlitModel::slitToPreSlit,
        &SlitModel::slitToPreSlit,
        &SlitModel::slitToPreSlit,
        "spatial", "spectral", "spatialSpectral"
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

    defineOverloads(
        cls,
        "slitToDetector",
        &OpticsModel::slitToDetector,
        &OpticsModel::slitToDetector,
        &OpticsModel::slitToDetector,
        &OpticsModel::slitToDetector,
        "spatial", "spectral", "spatialSpectral"
    );
    defineOverloads(
        cls,
        "detectorToSlit",
        &OpticsModel::detectorToSlit,
        &OpticsModel::detectorToSlit,
        &OpticsModel::detectorToSlit,
        &OpticsModel::detectorToSlit,
        "x", "y", "xy"
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

    defineOverloads(
        cls,
        "detectorToPixels",
        &DetectorModel::detectorToPixels,
        &DetectorModel::detectorToPixels,
        &DetectorModel::detectorToPixels,
        &DetectorModel::detectorToPixels,
        "x", "y", "xy"
    );
    defineOverloads(
        cls,
        "pixelsToDetector",
        &DetectorModel::pixelsToDetector,
        &DetectorModel::pixelsToDetector,
        &DetectorModel::pixelsToDetector,
        &DetectorModel::pixelsToDetector,
        "p", "q", "pq"
    );
}


PYBIND11_MODULE(OpticalModel, mod) {
    declareSlitModel(mod);
    declareOpticsModel(mod);
    declareDetectorModel(mod);
}


} // anonymous namespace

}}} // pfs::drp::stella
