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

    cls.def(py::init<std::size_t, std::size_t>(), "length"_a, "iTrace"_a=0);

    cls.def("getSpectrum", (typename Class::SpectrumVector (Class::*)()) &Class::getSpectrum);
    cls.def("setSpectrum", &Class::setSpectrum, "spectrum"_a);
    cls.def_property("spectrum", (typename Class::SpectrumVector (Class::*)()) &Class::getSpectrum,
                             &Class::setSpectrum);

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

    cls.def("getMask", (typename Class::Mask (Class::*)()) &Class::getMask);
    cls.def("setMask", &Class::setMask, "mask"_a);
    cls.def_property("mask", (typename Class::Mask (Class::*)()) &Class::getMask, &Class::setMask);

    cls.def("getITrace", &Class::getITrace);
    cls.def("setITrace", &Class::setITrace, "iTrace"_a);
    cls.def_property("iTrace", &Class::getITrace, &Class::setITrace);

    cls.def("getDispRms", &Class::getDispRms);
    cls.def("getDispRmsCheck", &Class::getDispRmsCheck);

    cls.def("getNGoodLines", &Class::getNGoodLines);

    cls.def("identify", &Class::identify, "lineList"_a, "dispCorControl"_a, "nLinesCheck"_a=0);

    cls.def("isWavelengthSet", &Class::isWavelengthSet);
}

template <typename T>
void declareSpectrumSet(py::module &mod) {
    using Class = SpectrumSet<T>;
    py::class_<Class, PTR(Class)> cls(mod, "SpectrumSet");

    cls.def(py::init<std::size_t, std::size_t>(), "nSpectra"_a=0, "length"_a=0);

    cls.def("getNtrace", &Class::getNtrace);

    cls.def("getSpectrum", (PTR(typename Class::SpectrumT) (Class::*)(std::size_t)) &Class::getSpectrum,
            "i"_a);
    cls.def("setSpectrum",
            (void (Class::*)(std::size_t, PTR(typename Class::SpectrumT))) &Class::setSpectrum,
            "i"_a, "spectrum"_a);
    cls.def("addSpectrum", (void (Class::*)(PTR(typename Class::SpectrumT))) &Class::addSpectrum,
            "spectrum"_a);

    cls.def("getAllFluxes", &Class::getAllFluxes);
    cls.def("getAllWavelengths", &Class::getAllWavelengths);
    cls.def("getAllMasks", &Class::getAllMasks);
    cls.def("getAllCovars", &Class::getAllCovars);

    // Pythonic APIs
    cls.def("__len__", &Class::getNtrace);
    cls.def("__getitem__", [](Class const& self, std::size_t i) { return self.getSpectrum(i); });
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

    mod.def("createLineList", math::createLineList<float, 1>, "wLen"_a, "lines"_a);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
