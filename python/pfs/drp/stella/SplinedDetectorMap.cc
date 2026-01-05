#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "pfs/drp/stella/SplinedDetectorMap.h"
#include "pfs/drp/stella/python/DetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

void declareSplinedDetectorMap(py::module & mod) {
    using Class = SplinedDetectorMap;
    auto cls = python::wrapDetectorMap<Class>(mod, "SplinedDetectorMap");
    cls.def(py::init<lsst::geom::Box2I,
                     DetectorMap::FiberIds const&,
                     std::vector<ndarray::Array<double, 1, 1>> const&,
                     std::vector<ndarray::Array<double, 1, 1>> const&,
                     std::vector<ndarray::Array<double, 1, 1>> const&,
                     std::vector<ndarray::Array<double, 1, 1>> const&,
                     DetectorMap::Array1D const&,
                     DetectorMap::Array1D const&,
                     lsst::afw::image::VisitInfo const&,
                     std::shared_ptr<lsst::daf::base::PropertySet>
                     >(),
            "bbox"_a, "fiberId"_a, "centerKnots"_a, "centerValues"_a, "wavelengthKnots"_a,
            "wavelengthValues"_a, "spatialOffsets"_a=nullptr, "spectralOffsets"_a=nullptr,
            "visitInfo"_a=Class::VisitInfo(lsst::daf::base::PropertySet()), "metadata"_a=nullptr);

    cls.def("getXCenterSpline", &Class::getXCenterSpline, "fiberId"_a);
    cls.def("getWavelengthSpline", &Class::getWavelengthSpline, "fiberId"_a);
    cls.def("setXCenter", &Class::setXCenter, "fiberId"_a, "knots"_a, "xCenter"_a);
    cls.def("setWavelength", &Class::setWavelength, "fiberId"_a, "knots"_a, "wavelength"_a);

    cls.def(py::pickle(
        [](Class const& self) {
            DetectorMap::FiberIds const& fiberId = self.getFiberId();
            std::size_t const numFibers = fiberId.getNumElements();
            std::vector<ndarray::Array<double, 1, 1>> xCenterKnots;
            std::vector<ndarray::Array<double, 1, 1>> xCenterValues;
            std::vector<ndarray::Array<double, 1, 1>> wavelengthKnots;
            std::vector<ndarray::Array<double, 1, 1>> wavelengthValues;
            xCenterKnots.reserve(numFibers);
            xCenterValues.reserve(numFibers);
            wavelengthKnots.reserve(numFibers);
            wavelengthValues.reserve(numFibers);
            for (std::size_t ii = 0; ii < numFibers; ++ii) {
                xCenterKnots.emplace_back(ndarray::copy(self.getXCenterSpline(fiberId[ii]).getX()));
                xCenterValues.emplace_back(ndarray::copy(self.getXCenterSpline(fiberId[ii]).getY()));
                wavelengthKnots.emplace_back(ndarray::copy(self.getWavelengthSpline(fiberId[ii]).getX()));
                wavelengthValues.emplace_back(ndarray::copy(self.getWavelengthSpline(fiberId[ii]).getY()));
            }
            return py::make_tuple(self.getBBox(), self.getFiberId(), xCenterKnots, xCenterValues,
                                  wavelengthKnots, wavelengthValues, self.getSpatialOffsets(),
                                  self.getSpectralOffsets(), self.getVisitInfo(), self.getMetadata());
        },
        [](py::tuple const& t){
            return SplinedDetectorMap(
                t[0].cast<lsst::geom::Box2I>(),
                t[1].cast<DetectorMap::FiberIds>(),
                t[2].cast<std::vector<ndarray::Array<double, 1, 1>>>(),
                t[3].cast<std::vector<ndarray::Array<double, 1, 1>>>(),
                t[4].cast<std::vector<ndarray::Array<double, 1, 1>>>(),
                t[5].cast<std::vector<ndarray::Array<double, 1, 1>>>(),
                t[6].cast<ndarray::Array<double, 1, 1>>(),
                t[7].cast<ndarray::Array<double, 1, 1>>(),
                t[8].cast<Class::VisitInfo>(),
                t[9].cast<std::shared_ptr<lsst::daf::base::PropertySet>>()
            );
        }
    ));
}


PYBIND11_MODULE(SplinedDetectorMap, mod) {
    declareSplinedDetectorMap(mod);
}

} // anonymous namespace

}}} // pfs::drp::stella
