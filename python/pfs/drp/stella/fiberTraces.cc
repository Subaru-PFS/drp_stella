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
void declareFiberTrace(py::module &mod, std::string const& suffix)
{
    using Class = FiberTrace<ImageT, MaskT, VarianceT>;
    py::class_<Class, PTR(Class)> cls(mod, ("FiberTrace" + suffix).c_str());

    cls.def(py::init<std::size_t, std::size_t, std::size_t>(), "width"_a=0, "height"_a=0, "iTrace"_a=0);
    cls.def(py::init<PTR(typename Class::MaskedImageT const) const&,
                     PTR(FiberTraceFunction const) const&,
                     std::size_t>(),
            "maskedImage"_a, "fiberTraceFunction"_a, "iTrace"_a=0);
    cls.def(py::init<Class const&>(), "fiberTrace"_a);
    cls.def(py::init<Class&, bool>(), "fiberTrace"_a, "deep"_a);

    cls.def("getTrace", (PTR(typename Class::MaskedImageT)(Class::*)())&Class::getTrace);
    cls.def("setTrace", &Class::setTrace, "trace"_a);
    cls.def("getImage", &Class::getImage);
    cls.def("setImage", &Class::setImage, "image"_a);
    cls.def("getMask", &Class::getMask);
    cls.def("setMask", &Class::setMask, "mask"_a);
    cls.def("getVariance", &Class::getVariance);
    cls.def("setVariance", &Class::setVariance, "variance"_a);
    cls.def("getProfile", &Class::getProfile);
    cls.def("setProfile", &Class::setProfile, "profile"_a);
    cls.def("getFiberTraceFunction", &Class::getFiberTraceFunction);
    cls.def("setFiberTraceFunction", &Class::setFiberTraceFunction, "function"_a);
    cls.def("getFiberTraceProfileFittingControl", &Class::getFiberTraceProfileFittingControl);
    cls.def("setFiberTraceProfileFittingControl",
            (void (Class::*)(PTR(FiberTraceProfileFittingControl const) const&))
                &Class::setFiberTraceProfileFittingControl,
            "control"_a);
    cls.def("getXCenters", &Class::getXCenters);
    cls.def("setXCenters", &Class::setXCenters, "centers"_a);
    cls.def("getXCentersMeas", &Class::getXCentersMeas);
    cls.def("setXCentersMeas", &Class::setXCentersMeas, "centers"_a);
    cls.def("getITrace", &Class::getITrace);
    cls.def("setITrace", &Class::setITrace, "iTrace"_a);
    cls.def("getWidth", &Class::getWidth);
    cls.def("getHeight", &Class::getHeight);
    cls.def("getTraceCoefficients", &Class::getTraceCoefficients);

    cls.def("extractFromProfile", &Class::extractFromProfile);
    cls.def("extractSum", &Class::extractSum);
    cls.def("createTrace", &Class::createTrace, "maskedImage"_a);
    cls.def("calcProfile", &Class::calcProfile);

    cls.def("getReconstructed2DSpectrum",
            (PTR(typename Class::Image)(Class::*)(typename Class::SpectrumT const&) const)
                &Class::getReconstructed2DSpectrum, "spectrum"_a);
    cls.def("getReconstructed2DSpectrum",
            (PTR(typename Class::Image)(Class::*)(typename Class::SpectrumT const&,
                                                  typename Class::SpectrumT const&) const)
                &Class::getReconstructed2DSpectrum,
            "spectrum"_a, "background"_a);
    cls.def("getReconstructedBackground", &Class::getReconstructedBackground, "background"_a);

    cls.def("calcProfileSwath", &Class::calcProfileSwath, "image"_a, "mask"_a, "variance"_a, "xCenters"_a,
            "index"_a);
    cls.def("calcSwathBoundY", &Class::calcSwathBoundY, "width"_a);

    cls.def("isTraceSet", &Class::isTraceSet);
    cls.def("isProfileSet", &Class::isProfileSet);
    cls.def("isFiberTraceProfileFittingControlSet", &Class::isFiberTraceProfileFittingControlSet);
}


template <typename ImageT,
          typename MaskT=lsst::afw::image::MaskPixel,
          typename VarianceT=lsst::afw::image::VariancePixel>
void declareFiberTraceSet(py::module &mod, std::string const& suffix)
{
    using Class = FiberTraceSet<ImageT, MaskT, VarianceT>;
    py::class_<Class, PTR(Class)> cls(mod, ("FiberTraceSet" + suffix).c_str());

    cls.def(py::init<std::size_t>(), "nTraces"_a=0);
    cls.def(py::init<Class const&, bool>(), "fiberTraceSet"_a, "deep"_a=false);
    cls.def("size", &Class::size);
    cls.def("__len__", &Class::size);
    cls.def("createTraces", &Class::createTraces, "maskedImage"_a);
    cls.def("getFiberTrace",
            (PTR(typename Class::FiberTraceT)(Class::*)(std::size_t const))&Class::getFiberTrace,
            "index"_a);
    cls.def("__getitem__", [](Class const& self, std::size_t index) { return self.getFiberTrace(index); });
    cls.def("erase", &Class::erase, "iStart"_a, "iEnd"_a=0);
    cls.def("setFiberTrace", &Class::setFiberTrace, "index"_a, "trace"_a);
    cls.def("addFiberTrace", &Class::addFiberTrace, "trace"_a, "index"_a=0);
    cls.def("getTraces", [](Class const& self) { return *self.getTraces(); });
    cls.def("setFiberTraceProfileFittingControl", &Class::setFiberTraceProfileFittingControl, "control"_a);
    cls.def("setAllProfiles", &Class::setAllProfiles, "fiberTraceSet"_a);
    cls.def("sortTracesByXCenter", &Class::sortTracesByXCenter);
    cls.def("calcProfileAllTraces", &Class::calcProfileAllTraces);
    cls.def("extractTraceNumberFromProfile", &Class::extractTraceNumberFromProfile, "index"_a);
    cls.def("extractAllTracesFromProfile", &Class::extractAllTracesFromProfile);
}


template <typename ImageT>
void declareFunctions(py::module &mod, std::string const& suffix)
{
    mod.def(("findAndTraceApertures" + suffix).c_str(), math::findAndTraceApertures<ImageT>,
            "maskedImage"_a, "control"_a);
    mod.def(("findCenterPositionsOneTrace" + suffix).c_str(), math::findCenterPositionsOneTrace<ImageT>,
            "ccdImage"_a, "ccdImageVariance"_a, "control"_a);
    mod.def(("makeNormFlatFiberTrace" + suffix).c_str(), math::makeNormFlatFiberTrace<ImageT>,
            "maskedImage"_a, "fiberTraceFunctionWide"_a, "fiberTraceFunctionControlNarrow"_a,
            "fiberTraceProfileFittingControl"_a, "minSNR"_a=100.0, "iTrace"_a=0);
    mod.def("assignITrace",
            math::assignITrace<ImageT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel,
                               int, float, 1>,
            "fiberTraceSet"_a, "traceIds"_a, "xCenters"_a);
    mod.def("findITrace", math::findITrace<ImageT, lsst::afw::image::MaskPixel,
                                           lsst::afw::image::VariancePixel, float, 0>,
            "fiberTrace"_a, "xCenters"_a, "nTraces"_a, "nRows"_a, "startPos"_a=0);
    mod.def("addFiberTraceToCcdImage",
            math::addFiberTraceToCcdImage<ImageT, lsst::afw::image::MaskPixel,
                                          lsst::afw::image::VariancePixel, float, float>,
            "fiberTrace"_a, "fiberTraceRepresentation"_a, "ccdImage"_a);

    mod.def("ccdToFiberTraceCoordinates",
            math::ccdToFiberTraceCoordinates<float, ImageT, lsst::afw::image::MaskPixel,
                                             lsst::afw::image::VariancePixel>,
            "ccdCoordinates"_a, "fiberTrace"_a);

    mod.def("fiberTraceCoordinatesRelativeTo",
            math::fiberTraceCoordinatesRelativeTo<float, ImageT, lsst::afw::image::MaskPixel,
                                                  lsst::afw::image::VariancePixel>,
            "fiberTraceCoordinates"_a, "ccdCoordinatesCenter"_a, "fiberTrace"_a);
    mod.def("markFiberTraceInMask", utils::markFiberTraceInMask<ImageT, lsst::afw::image::MaskPixel,
                                                               lsst::afw::image::VariancePixel>,
            "fiberTrace"_a, "mask"_a, "value"_a=1);
}


PYBIND11_PLUGIN(fiberTraces) {
    py::module mod("fiberTraces");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    declareFiberTrace<float>(mod, "F");
    declareFiberTrace<double>(mod, "D");

    declareFiberTraceSet<float>(mod, "F");
    declareFiberTraceSet<double>(mod, "D");

    declareFunctions<float>(mod, "F");

    py::class_<math::FindCenterPositionsOneTraceResult, PTR(math::FindCenterPositionsOneTraceResult)>
        findResult(mod, "FindCenterPositionsOneTraceResult");
    findResult.def_readwrite("apertureCenterIndex",
                             &math::FindCenterPositionsOneTraceResult::apertureCenterIndex);
    findResult.def_readwrite("apertureCenterPos",
                             &math::FindCenterPositionsOneTraceResult::apertureCenterPos);
    findResult.def_readwrite("eApertureCenterPos",
                             &math::FindCenterPositionsOneTraceResult::eApertureCenterPos);

    py::class_<math::dataXY<float>, PTR(math::dataXY<float>)> coord(mod, "CoordinatesF");
    coord.def(py::init<>());
    coord.def("__init__",
              [](math::dataXY<float>& self, float x_, float y_) {
                  new (&self) math::dataXY<float>();
                  self.x = x_;
                  self.y = y_;
              });
    coord.def_readwrite("x", &math::dataXY<float>::x);
    coord.def_readwrite("y", &math::dataXY<float>::y);

    mod.def("calculateXCenters",
            (ndarray::Array<float, 1, 1>(*)(PTR(FiberTraceFunction const) const&,
                                            std::size_t const&, std::size_t const&))&math::calculateXCenters,
            "fiberTraceFunction"_a, "height"_a=0, "width"_a=0);
    mod.def("calculateXCenters",
            (ndarray::Array<float, 1, 1>(*)(PTR(FiberTraceFunction const) const&,
                                            ndarray::Array<float, 1, 1> const&,
                                            std::size_t const&, std::size_t const&))&math::calculateXCenters,
            "fiberTraceFunction"_a, "yIn"_a, "height"_a=0, "width"_a=0);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
