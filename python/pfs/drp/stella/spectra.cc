#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "pfs/drp/stella/Spectra.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

template <typename T>
void declareSpectrum(py::module &mod) {
    using Class = Spectrum<T>;
    py::class_<Class, std::shared_ptr<Class>> cls(mod, "Spectrum");

    cls.def(py::init<std::size_t, std::size_t>(), "length"_a=0, "iTrace"_a=0);
    cls.def(py::init<Class&, std::size_t, bool>(), "spectrum"_a, "iTrace"_a=0, "deep"_a=false);
    cls.def(py::init<Class const&>(), "spectrum"_a);

    cls.def("getSpectrum", (typename Class::SpectrumVector (Class::*)()) &Class::getSpectrum);
    cls.def("setSpectrum", &Class::setSpectrum, "spectrum"_a);
    cls.def_property("spectrum", (typename Class::SpectrumVector (Class::*)()) &Class::getSpectrum,
                             &Class::setSpectrum);

    cls.def("getSky", (typename Class::SpectrumVector (Class::*)()) &Class::getSky);
    cls.def("setSky", &Class::setSpectrum, "sky"_a);
    cls.def_property("sky", (typename Class::SpectrumVector (Class::*)()) &Class::getSky, &Class::setSky);

    cls.def("getVariance", (typename Class::VarianceVector (Class::*)()) &Class::getVariance);
    cls.def("setVariance", &Class::setVariance, "variance"_a);
    cls.def_property("variance", (typename Class::VarianceVector (Class::*)()) &Class::getVariance,
                     &Class::setVariance);

    cls.def("getCovar", (typename Class::CovarianceMatrix (Class::*)()) &Class::getCovar);
    cls.def("setCovar", &Class::setCovar, "covar"_a);
    cls.def_property("covariance", (typename Class::CovarianceMatrix (Class::*)()) &Class::getCovar,
                     &Class::setCovar);

    cls.def("getWavelength", (typename Class::WavelengthVector (Class::*)()) &Class::getWavelength);
    cls.def("setWavelength", &Class::setWavelength, "wavelength"_a);
    cls.def_property("wavelength", (typename Class::WavelengthVector (Class::*)()) &Class::getWavelength,
                     &Class::setWavelength);

    cls.def("getDispersion", (typename Class::WavelengthVector (Class::*)()) &Class::getDispersion);
    cls.def("setDispersion", &Class::setDispersion, "dispersion"_a);
    cls.def_property("dispersion", (typename Class::WavelengthVector (Class::*)()) &Class::getDispersion,
                     &Class::setDispersion);

    cls.def("getMask", (typename Class::Mask (Class::*)()) &Class::getMask);
    cls.def("setMask", &Class::setMask, "mask"_a);
    cls.def_property("mask", (typename Class::Mask (Class::*)()) &Class::getMask, &Class::setMask);

    cls.def("getLength", &Class::getLength);
    cls.def("setLength", &Class::setLength, "length"_a);
    cls.def_property("length", &Class::getLength, &Class::setLength);

    cls.def("getITrace", &Class::getITrace);
    cls.def("setITrace", &Class::setITrace, "iTrace"_a);
    cls.def_property("iTrace", &Class::getITrace, &Class::setITrace);

    cls.def("getDispCoeffs", &Class::getDispCoeffs);
    cls.def("setDispCoeffs", &Class::setDispCoeffs, "coeffs"_a);
    cls.def_property("dispCoeffs", &Class::getDispCoeffs, &Class::setDispCoeffs);

    cls.def("getDispRms", &Class::getDispRms);
    cls.def("getDispRmsCheck", &Class::getDispRmsCheck);

    cls.def("getNGoodLines", &Class::getNGoodLines);

    cls.def("getDispCorControl", &Class::getDispCorControl);

    cls.def("identify", &Class::template identify<float>, "lineList"_a, "dispCorControl"_a,
            "nLinesCheck"_a=0);

    cls.def("isWavelengthSet", &Class::isWavelengthSet);

    cls.def("getYLow", &Class::getYLow);
    cls.def("setYLow", &Class::setYLow, "yLow"_a);
    cls.def_property("yLow", &Class::getYLow, &Class::setYLow);

    cls.def("getYHigh", &Class::getYHigh);
    cls.def("setYHigh", &Class::setYHigh, "yHigh"_a);
    cls.def_property("yHigh", &Class::getYHigh, &Class::setYHigh);

    cls.def("getNCCDRows", &Class::getNCCDRows);
    cls.def("setNCCDRows", &Class::setNCCDRows, "nCCDRows"_a);
    cls.def_property("nCCDRows", &Class::getNCCDRows, &Class::setNCCDRows);
}

template <typename T>
void declareSpectrumSet(py::module &mod) {
    using Class = SpectrumSet<T>;
    py::class_<Class, PTR(Class)> cls(mod, "SpectrumSet");

    cls.def(py::init<std::size_t, std::size_t>(), "nSpectra"_a=0, "length"_a=0);
    cls.def(py::init<Class const&>(), "spectrumSet"_a);
    cls.def(py::init<typename Class::Spectra const&>(), "spectra"_a);

    cls.def("size", &Class::size);

    cls.def("getSpectrum", (PTR(typename Class::SpectrumT) (Class::*)(std::size_t)) &Class::getSpectrum,
            "i"_a);
    cls.def("setSpectrum",
            (void (Class::*)(std::size_t, PTR(typename Class::SpectrumT) const&)) &Class::setSpectrum,
            "i"_a, "spectrum"_a);
    cls.def("addSpectrum", (void (Class::*)(PTR(typename Class::SpectrumT) const&)) &Class::addSpectrum,
            "spectrum"_a);

    cls.def("getSpectra", [](Class const& self) { return *self.getSpectra(); });

    cls.def("erase", &Class::erase, "iStart"_a, "iEnd"_a=0);

    cls.def("getAllFluxes", &Class::getAllFluxes);
    cls.def("getAllWavelengths", &Class::getAllWavelengths);
    cls.def("getAllDispersions", &Class::getAllDispersions);
    cls.def("getAllMasks", &Class::getAllMasks);
    cls.def("getAllSkies", &Class::getAllSkies);
    cls.def("getAllVariances", &Class::getAllVariances);
    cls.def("getAllCovars", &Class::getAllCovars);

    // Pythonic APIs
    cls.def("__len__", &Class::size);
    cls.def("__getitem__", [](Class const& self, std::size_t i) { return self.getSpectrum(i); });
}

template <typename T, typename U>
void declareStretchAndCrossCorrelate(py::module &mod) {
    using Class = math::StretchAndCrossCorrelateSpecResult<T, U>;
    py::class_<Class, std::shared_ptr<Class>> cls(mod, "StretchAndCrossCorrelateSpecResult");
    cls.def_readonly("lineList", &Class::lineList);
    cls.def_readonly("specPieces", &Class::specPieces);

    mod.def("stretchAndCrossCorrelateSpec", math::stretchAndCrossCorrelateSpec<T, U>,
            "spec"_a, "specRef"_a, "lineList_WLenPix"_a, "dispCorControl"_a);
}

PYBIND11_PLUGIN(spectra) {
    py::module mod("spectra");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    declareSpectrum<float>(mod);

    declareSpectrumSet<float>(mod);

    declareStretchAndCrossCorrelate<float, float>(mod);

    mod.def("createLineList", math::createLineList<float, 1>, "wLen"_a, "lines"_a);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
