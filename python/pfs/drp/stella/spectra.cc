#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "pfs/drp/stella/Spectra.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

/************************************************************************************************************/

void declareReferenceLine(py::module &mod) {
    using Class = ReferenceLine;

    py::class_<Class, std::shared_ptr<Class>> cls(mod, "ReferenceLine");

    /* Member types and enums */
    py::enum_<ReferenceLine::Status>(cls, "Status", py::arithmetic())
        .value("NOWT", ReferenceLine::Status::NOWT)
        .value("FIT", ReferenceLine::Status::FIT)
        .value("RESERVED", ReferenceLine::Status::RESERVED)
        .value("MISIDENTIFIED", ReferenceLine::Status::MISIDENTIFIED)
        .value("CLIPPED", ReferenceLine::Status::CLIPPED)
        .value("SATURATED", ReferenceLine::Status::SATURATED)
        .value("INTERPOLATED", ReferenceLine::Status::INTERPOLATED)
        .value("CR", ReferenceLine::Status::CR)
        .export_values();

    cls.def(py::init<std::string, ReferenceLine::Status, float, float>(),
            "description"_a, "status"_a=ReferenceLine::Status::NOWT,
            "wavelength"_a=0, "guessedIntensity"_a=0);
    cls.def_readwrite("description", &ReferenceLine::description);
    cls.def_readwrite("status", &ReferenceLine::status);
    cls.def_readwrite("wavelength", &ReferenceLine::wavelength);
    cls.def_readwrite("guessedIntensity", &ReferenceLine::guessedIntensity);
    cls.def_readwrite("guessedPixelPos", &ReferenceLine::guessedPixelPos);
    cls.def_readwrite("fitIntensity", &ReferenceLine::fitIntensity);
    cls.def_readwrite("fitPixelPos", &ReferenceLine::fitPixelPos);
    cls.def_readwrite("fitPixelPosErr", &ReferenceLine::fitPixelPosErr);
}

/************************************************************************************************************/
void declareSpectrum(py::module &mod) {
    using Class = Spectrum;
    py::class_<Class, std::shared_ptr<Class>> cls(mod, "Spectrum");

    cls.def(py::init<std::size_t, std::size_t>(), "length"_a, "fiberId"_a=0);

    cls.def("getSpectrum", (ndarray::Array<Spectrum::ImageT, 1, 1> (Class::*)()) &Class::getSpectrum);
    cls.def("setSpectrum", &Class::setSpectrum, "spectrum"_a);
    cls.def_property("spectrum", (ndarray::Array<Spectrum::ImageT, 1, 1> (Class::*)()) &Class::getSpectrum,
                             &Class::setSpectrum);

    cls.def("getVariance", (ndarray::Array<Spectrum::VarianceT, 1, 1> (Class::*)()) &Class::getVariance);
    cls.def("setVariance", &Class::setVariance, "variance"_a);
    cls.def_property("variance", (ndarray::Array<Spectrum::VarianceT, 1, 1> (Class::*)()) &Class::getVariance,
                     &Class::setVariance);

    cls.def("getCovar", (ndarray::Array<Spectrum::VarianceT, 2, 1> (Class::*)()) &Class::getCovar);
    cls.def("setCovar", &Class::setCovar, "covar"_a);
    cls.def_property("covariance", (ndarray::Array<Spectrum::VarianceT, 2, 1> (Class::*)()) &Class::getCovar,
                     &Class::setCovar);

    cls.def("getWavelength", (ndarray::Array<Spectrum::ImageT, 1, 1> (Class::*)()) &Class::getWavelength);
    cls.def("setWavelength", &Class::setWavelength, "wavelength"_a);
    cls.def_property("wavelength", (ndarray::Array<Spectrum::ImageT, 1, 1> (Class::*)()) &Class::getWavelength,
                     &Class::setWavelength);

    cls.def("getMask", (typename Class::Mask (Class::*)()) &Class::getMask);
    cls.def("setMask", &Class::setMask, "mask"_a);
    cls.def_property("mask", (typename Class::Mask (Class::*)()) &Class::getMask, &Class::setMask);

    cls.def("getFiberId", &Class::getFiberId);
    cls.def("setFiberId", &Class::setFiberId, "fiberId"_a);
    cls.def_property("fiberId", &Class::getFiberId, &Class::setFiberId);

    cls.def("identify", &Class::identify, "lineList"_a, "dispCorControl"_a, "nLinesCheck"_a=0);

    cls.def("getReferenceLines", &Class::getReferenceLines);    

    cls.def("isWavelengthSet", &Class::isWavelengthSet);
}

void declareSpectrumSet(py::module &mod) {
    using Class = SpectrumSet;
    py::class_<Class, PTR(Class)> cls(mod, "SpectrumSet");

    cls.def(py::init<std::size_t, std::size_t>(), "nSpectra"_a=0, "length"_a=0);

    cls.def("getNtrace", &Class::getNtrace);

    cls.def("getSpectrum", (PTR(Spectrum) (Class::*)(std::size_t)) &Class::getSpectrum,
            "i"_a);
    cls.def("setSpectrum",
            (void (Class::*)(std::size_t, PTR(Spectrum))) &Class::setSpectrum, "i"_a, "spectrum"_a);
    cls.def("addSpectrum", (void (Class::*)(PTR(Spectrum))) &Class::addSpectrum, "spectrum"_a);

    cls.def("getAllFluxes", &Class::getAllFluxes);
    cls.def("getAllWavelengths", &Class::getAllWavelengths);
    cls.def("getAllMasks", &Class::getAllMasks);
    cls.def("getAllCovars", &Class::getAllCovars);

    // Pythonic APIs
    cls.def("__len__", &Class::getNtrace);
    cls.def("__getitem__", [](Class const& self, std::size_t i) {
            if (i >= self.getNtrace()) throw py::index_error();
            
            return self.getSpectrum(i);
        });
}

PYBIND11_PLUGIN(spectra) {
    py::module mod("spectra");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    declareReferenceLine(mod);
    declareSpectrum(mod);
    declareSpectrumSet(mod);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
