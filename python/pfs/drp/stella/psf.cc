#include <pybind11/pybind11.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "pfs/drp/stella/PSF.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

template <typename T>
void declareExtractPSFResult(py::module &mod, std::string const& suffix) {
    using Class = PSF<T>::ExtractPSFResult;
    py::class_<Class, std::shared_ptr<Class>> cls(mod, ("ExtractPSFResult" + suffix).c_str());
    cls.def(py::init(std::size_t)(), "length"_a);
    cls.def_readonly("xRelativeToCenter", &Class::xRelativeToCenter);
    cls.def_readonly("yRelativeToCenter", &Class::yRelativeToCenter);
    cls.def_readonly("zNormalized", &Class::zNormalized);
    cls.def_readonly("zTrace", &Class::zTrace);
    cls.def_readonly("weight", &Class::weight);
    cls.def_readonly("xTrace", &Class::xTrace);
    cls.def_readonly("yTrace", &Class::yTrace);
    cls.def_readonly("xCenterPSFCCD", &Class::xCenterPSFCCD);
    cls.def_readonly("yCenterPSFCCD", &Class::yCenterPSFCCD);
}

#if 0
template <typename ImageT,
          typename MaskT=lsst::afw::image::MaskPixel,
          typename VarianceT=lsst::afw::image::VariancePixel,
          typename WavelengthT=lsst::afw::image::VariancePixel>
void declarePSFOperations(py::class_ &cls, std::string const& suffix)
{
}
#endif

template <typename T>
void declarePSF(py::module &mod, std::string const& suffix) {
    using Class = PSF<T>;
    py::class_<Class, std::shared_ptr<Class>> cls(mod, ("PSF" + suffix).c_str());

    cls.def(py::init(std::size_t, std::size_t)(), "iTrace"_a=0, "iBin"_a=0);
    cls.def(py::init(Class const&)(), "psf"_a);
    cls.def(py::init(std::size_t, std::size_t, PTR(TwoDPSFControl) const, std::size_t, std::size_t)(),
            "yLow"_a, "yHigh"_a, "twoDPSFControl"_a, "iTrace"_a=0, "iBin"_a=0);

    typedef Class::Vector(Class::*)() VectorGetter;

    cls.def("getIBin", &Class::getIBin);
    cls.def("getITrace", &Class::getITrace);
    cls.def("getYLow", &Class::getYLow);
    cls.def("getYHigh", &Class::getYHigh);
    cls.def("getImagePSF_XTrace", (VectorGetter)&Class::getImagePSF_XTrace);
    cls.def("getImagePSF_YTrace", (VectorGetter)&Class::getImagePSF_YTrace);
    cls.def("getImagePSF_ZTrace", (VectorGetter)&Class::getImagePSF_ZTrace);
    cls.def("getImagePSF_XRelativeToCenter", (VectorGetter)&Class::getImagePSF_XRelativeToCenter);
    cls.def("getImagePSF_YRelativeToCenter", (VectorGetter)&Class::getImagePSF_YRelativeToCenter);
    cls.def("getImagePSF_ZNormalized", (VectorGetter)&Class::getImagePSF_ZNormalized);
    cls.def("getImagePSF_ZFit", (VectorGetter)&Class::getImagePSF_ZFit);
    cls.def("getImagePSF_Weight", (VectorGetter)&Class::getImagePSF_Weight);
    cls.def("getXCentersPSFCCD", (VectorGetter)&Class::getXCentersPSFCCD);
    cls.def("getYCentersPSFCCD", (VectorGetter)&Class::getYCentersPSFCCD);
    cls.def("getNPixPerPSF", (Class::VectorInt(Class::*)())&Class::getNPixPerPSF);
    cls.def("getXRangePolynomial", (VectorGetter)&Class::getXRangePolynomial);

    cls.def("setImagePSF_ZTrace", &Class::setImagePSF_ZTrace, "zTrace"_a);
    cls.def("setImagePSF_ZNormalized", &Class::setImagePSF_ZNormalized, "zNormalized"_a);
    cls.def("setImagePSF_ZFit", &Class::setImagePSF_ZFit, "zFit"_a);
    cls.def("setXCentersPSFCCD", &Class::setXCentersPSFCCD, "xCentersPSFCCD_In"_a);
    cls.def("setYCentersPSFCCD", &Class::setYCentersPSFCCD, "yCentersPSFCCD_In"_a);

    cls.def("isTwoDPSFControlSet", &Class::isTwoDPSFControlSet);
    cls.def("isPSFsExtracted", &Class::isPSFsExtracted);

    cls.def("getTwoDPSFControl", &Class::getTwoDPSFControl);
    cls.def("setTwoDPSFControl", &Class::setTwoDPSFControl, "twoDPSFControl"_a);

    // Doesn't look like any of the more complicated methods are used in python...
}

template <typename T>
void declarePSFSet(py::module &mod, std::string const& suffix) {
    using Class = PSFSet<T>;
    py::class_<Class, std::shared_ptr<Class>> cls(mod, ("PSFSet" + suffix).c_str());

    // Doesn't look like this is used from python...
}

PYBIND11_PLUGIN(psf) {
    py::module mod("psf");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    declareExtractPSFResult<float>(mod, "F");
    declareExtractPSFResult<double>(mod, "D");

    declarePSF<float>(mod, "F");
    declarePSF<double>(mod, "D");

    declarePSFSet<float>(mod, "F");
    declarePSFSet<double>(mod, "D");

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
