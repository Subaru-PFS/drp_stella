#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "pfs/drp/stella/FiberTraces.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

template <typename ImageT,
          typename MaskT=lsst::afw::image::MaskPixel,
          typename VarianceT=lsst::afw::image::VariancePixel>
void declareFiberTrace(py::module &mod)
{
    using Class = FiberTrace<ImageT, MaskT, VarianceT>;
    py::class_<Class, PTR(Class)> cls(mod, "FiberTrace");

    cls.def(py::init<PTR(typename Class::MaskedImageT const) const&,
                     std::size_t>(),
            "maskedImage"_a, "fiberTraceId"_a=0);
    cls.def(py::init<PTR(typename Class::MaskedImageT const) const&,
                     PTR(FiberTraceFunction const) const&,
                     PTR(FiberTraceProfileFittingControl) const&,
                     std::size_t>(),
            "maskedImage"_a, "fiberTraceFunction"_a, "fiberTraceProfileFittingControl"_a, "fiberId"_a=0);
    cls.def(py::init<Class&, bool>(), "fiberTrace"_a, "deep"_a=false);

    cls.def("getTrace", (PTR(typename Class::MaskedImageT)(Class::*)())&Class::getTrace);
    cls.def("getXCenters", &Class::getXCenters);
    cls.def("getFiberId", &Class::getFiberId);

    cls.def("extractSpectrum", &Class::extractSpectrum, "image"_a,
            "fitBackground"_a=false, "clipNSigma"_a=0.0, "useProfile"_a=true);

    cls.def("getReconstructed2DSpectrum",
            (PTR(typename Class::Image)(Class::*)(Spectrum const&) const)
                &Class::getReconstructed2DSpectrum, "spectrum"_a);
}


template <typename ImageT,
          typename MaskT=lsst::afw::image::MaskPixel,
          typename VarianceT=lsst::afw::image::VariancePixel>
void declareFiberTraceSet(py::module &mod)
{
    using Class = FiberTraceSet<ImageT, MaskT, VarianceT>;
    py::class_<Class, PTR(Class)> cls(mod, "FiberTraceSet");

    cls.def(py::init<>());
    cls.def(py::init<Class const&, bool>(), "fiberTraceSet"_a, "deep"_a=false);
    cls.def("getNtrace", &Class::getNtrace);
    cls.def("getFiberTrace",
            (PTR(typename Class::FiberTraceT)(Class::*)(std::size_t const))&Class::getFiberTrace,
            "index"_a);
    cls.def("setFiberTrace", &Class::setFiberTrace, "index"_a, "trace"_a);
    cls.def("addFiberTrace", &Class::addFiberTrace, "trace"_a, "index"_a=0);
    cls.def("getTraces", [](Class const& self) { return *self.getTraces(); });
    // Pythonic APIs
    cls.def("__len__", &Class::getNtrace);
    cls.def("__getitem__", [](Class const& self, std::size_t i) {
            if (i >= self.getNtrace()) throw py::index_error();
            
            return self.getFiberTrace(i);
        });
}


template <typename ImageT>
void declareFunctions(py::module &mod)
{
    mod.def("findAndTraceApertures", math::findAndTraceApertures<ImageT>,
            "maskedImage"_a, "detectorMap"_a, "fiberTraceFunctionFindingControl"_a,
            " fiberTraceProfileFittingControl"_a);
    mod.def("findCenterPositionsOneTrace", math::findCenterPositionsOneTrace<ImageT>,
            "ccdImage"_a, "ccdImageVariance"_a, "control"_a, "nextSearchStart"_a);
}


PYBIND11_PLUGIN(fiberTraces) {
    py::module mod("fiberTraces");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    declareFiberTrace<float>(mod);

    declareFiberTraceSet<float>(mod);

    declareFunctions<float>(mod);

    py::class_<math::FindCenterPositionsOneTraceResult, PTR(math::FindCenterPositionsOneTraceResult)>
        findResult(mod, "FindCenterPositionsOneTraceResult");
    findResult.def_readwrite("apertureCenterIndex",
                             &math::FindCenterPositionsOneTraceResult::apertureCenterIndex);
    findResult.def_readwrite("apertureCenterPos",
                             &math::FindCenterPositionsOneTraceResult::apertureCenterPos);
    findResult.def_readwrite("eApertureCenterPos",
                             &math::FindCenterPositionsOneTraceResult::eApertureCenterPos);
    findResult.def_readwrite("nextSearchStart",
                             &math::FindCenterPositionsOneTraceResult::nextSearchStart);

    py::class_<math::dataXY<float>, PTR(math::dataXY<float>)> coord(mod, "Coordinates");
    coord.def(py::init<>());
    coord.def("__init__",
              [](math::dataXY<float>& self, float x_, float y_) {
                  new (&self) math::dataXY<float>();
                  self.x = x_;
                  self.y = y_;
              });
    coord.def_readwrite("x", &math::dataXY<float>::x);
    coord.def_readwrite("y", &math::dataXY<float>::y);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
