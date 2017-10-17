#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "lsst/base.h"
#include "pfs/drp/stella/DetectorMap.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

void declareDetectorMap(py::module &mod)
{
    using Class = DetectorMap;
    py::class_<Class, PTR(Class)> cls(mod, "DetectorMap");

    cls.def(py::init<lsst::afw::geom::Box2I,
                     ndarray::Array<int,   1, 1> const,
                     ndarray::Array<float, 2, 1> const,
                     ndarray::Array<float, 2, 1> const,
                     ndarray::Array<float, 2, 1> const*,
                     std::size_t
            >(),
            "bbox"_a, "fiberIds"_a, "xCenters"_a, "wavelengths"_a, "slitOffsets"_a=nullptr, "nKnot"_a=25);

    cls.attr("FIBER_DX") =     &Class::FIBER_DX;
    cls.attr("FIBER_DY") =     &Class::FIBER_DY;
    cls.attr("FIBER_DFOCUS") = &Class::FIBER_DFOCUS;
    
    cls.def("findFiberId", &Class::findFiberId, "pixelPos"_a);
    cls.def("findPoint", &Class::findPoint, "fiberId"_a, "wavelength"_a);
    cls.def("getFiberIds", &Class::getFiberIds);
    cls.def("getWavelength", &Class::getWavelength, "fiberId"_a);
    cls.def("setWavelength", &Class::setWavelength, "fiberId"_a, "wavelength"_a);
    cls.def("getXCenter", &Class::getXCenter, "fiberId"_a);
    cls.def("setXCenter", &Class::setXCenter, "fiberId"_a, "xCenters"_a);
    cls.def("getSlitOffsets",
            (ndarray::Array<float, 2, 1> const& (Class::*)() const)&Class::getSlitOffsets);
    cls.def("getSlitOffsets",
            (ndarray::Array<float, 1, 0> const (Class::*)(std::size_t) const)&Class::getSlitOffsets,
            "fiberId"_a);
    cls.def("setSlitOffsets", &Class::setSlitOffsets, "slitOffsets"_a);
    cls.def("getFiberIdx", &Class::getFiberIdx);
}

void declareFunctions(py::module &mod)
{
}

PYBIND11_PLUGIN(detectorMap) {
    py::module mod("detectorMap");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    declareDetectorMap(mod);
    declareFunctions(mod);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
