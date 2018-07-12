#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
    cls.def_readwrite("guessedPosition", &ReferenceLine::guessedPosition);
    cls.def_readwrite("fitIntensity", &ReferenceLine::fitIntensity);
    cls.def_readwrite("fitPosition", &ReferenceLine::fitPosition);
    cls.def_readwrite("fitPositionErr", &ReferenceLine::fitPositionErr);

    cls.def("__getstate__",
        [](ReferenceLine const& self) {
            return py::make_tuple(self.description, self.status, self.wavelength, self.guessedIntensity,
                                  self.guessedPosition, self.fitIntensity, self.fitPosition,
                                  self.fitPositionErr);
        });
    cls.def("__setstate__",
        [](ReferenceLine & self, py::tuple const& t) {
            new (&self) ReferenceLine(t[0].cast<std::string>(), ReferenceLine::Status(t[1].cast<int>()),
                                      t[2].cast<float>(), t[3].cast<float>(), t[4].cast<float>(),
                                      t[5].cast<float>(), t[6].cast<float>(), t[7].cast<float>());
        });
}

/************************************************************************************************************/
void declareSpectrum(py::module &mod) {
    using Class = Spectrum;
    py::class_<Class, std::shared_ptr<Class>> cls(mod, "Spectrum");

    cls.def(py::init<std::size_t, std::size_t>(), "length"_a, "fiberId"_a=0);
    cls.def(py::init<Spectrum::ImageArray const&, Spectrum::Mask const&, Spectrum::ImageArray const&,
                     Spectrum::CovarianceMatrix const&, Spectrum::ImageArray const&,
                     Spectrum::ReferenceLineList const&, std::size_t>(),
            "spectrum"_a, "mask"_a, "background"_a, "covariance"_a, "wavelength"_a,
            "lines"_a=Spectrum::ReferenceLineList(), "fiberId"_a=0);

    cls.def("getSpectrum", (ndarray::Array<Spectrum::ImageT, 1, 1> (Class::*)()) &Class::getSpectrum);
    cls.def("setSpectrum", &Class::setSpectrum, "spectrum"_a);
    cls.def_property("spectrum", (ndarray::Array<Spectrum::ImageT, 1, 1> (Class::*)()) &Class::getSpectrum,
                             &Class::setSpectrum);

    cls.def("getBackground", (ndarray::Array<Spectrum::ImageT, 1, 1> (Class::*)()) &Class::getBackground);
    cls.def("setBackground", &Class::setBackground, "background"_a);
    cls.def_property("background",
                     (ndarray::Array<Spectrum::ImageT, 1, 1> (Class::*)()) &Class::getBackground,
                     &Class::setBackground);

    cls.def("getVariance", (ndarray::Array<Spectrum::VarianceT, 1, 1> (Class::*)()) &Class::getVariance);
    cls.def("setVariance", &Class::setVariance, "variance"_a);
    cls.def_property("variance", (ndarray::Array<Spectrum::VarianceT, 1, 1> (Class::*)()) &Class::getVariance,
                     &Class::setVariance);

    cls.def("getCovariance", (ndarray::Array<Spectrum::VarianceT, 2, 1> (Class::*)()) &Class::getCovariance);
    cls.def("setCovariance", &Class::setCovariance, "covariance"_a);
    cls.def_property("covariance",
                     (ndarray::Array<Spectrum::VarianceT, 2, 1> (Class::*)()) &Class::getCovariance,
                     &Class::setCovariance);

    cls.def("getWavelength", (ndarray::Array<Spectrum::ImageT, 1, 1> (Class::*)()) &Class::getWavelength);
    cls.def("setWavelength", &Class::setWavelength, "wavelength"_a);
    cls.def_property("wavelength", (ndarray::Array<Spectrum::ImageT, 1, 1> (Class::*)()) &Class::getWavelength,
                     &Class::setWavelength);

    cls.def("getMask", [](Class & self) { return self.getMask(); },
            py::return_value_policy::reference_internal);
    cls.def("setMask", &Class::setMask, "mask"_a);
    cls.def_property("mask", [](Class & self) { return self.getMask(); }, &Class::setMask,
                     py::return_value_policy::reference_internal);

    cls.def("getFiberId", &Class::getFiberId);
    cls.def("setFiberId", &Class::setFiberId, "fiberId"_a);
    cls.def_property("fiberId", &Class::getFiberId, &Class::setFiberId);

    cls.def("identify", &Class::identify, "lineList"_a, "dispCorControl"_a, "nLinesCheck"_a=0);

    cls.def("getReferenceLines", [](Class & self) { return self.getReferenceLines(); },
            py::return_value_policy::reference_internal);
    cls.def("setReferenceLines", &Class::setReferenceLines);
    cls.def_property("referenceLines", [](Class & self) { return self.getReferenceLines(); },
                     &Class::setReferenceLines, py::return_value_policy::reference_internal);

    cls.def("getNumPixels", &Class::getNumPixels);
    cls.def("__len__", &Class::getNumPixels);

    cls.def("isWavelengthSet", &Class::isWavelengthSet);

    cls.def("__getstate__",
        [](Spectrum const& self) {
            return py::make_tuple(self.getSpectrum(), self.getMask(), self.getBackground(),
                                  self.getCovariance(), self.getWavelength(), self.getReferenceLines(),
                                  self.getFiberId());
        });
    cls.def("__setstate__",
        [](Spectrum & self, py::tuple const& t) {
            new (&self) Spectrum(t[0].cast<Spectrum::ImageArray>(), t[1].cast<Spectrum::Mask>(),
                                 t[2].cast<Spectrum::ImageArray>(), t[3].cast<Spectrum::CovarianceMatrix>(),
                                 t[4].cast<Spectrum::ImageArray>(), t[5].cast<Spectrum::ReferenceLineList>(),
                                 t[6].cast<std::size_t>());
        });
}

void declareSpectrumSet(py::module &mod) {
    using Class = SpectrumSet;
    py::class_<Class, PTR(Class)> cls(mod, "SpectrumSet");

    cls.def(py::init<std::size_t>(), "length"_a);
    cls.def(py::init<std::size_t, std::size_t>(), "nSpectra"_a, "length"_a);

    cls.def("size", &Class::size);
    cls.def("reserve", &Class::reserve);
    cls.def("add", (void (Class::*)(PTR(Spectrum))) &Class::add, "spectrum"_a);
    cls.def("getLength", &Class::getLength);

    cls.def("getAllFluxes", &Class::getAllFluxes);
    cls.def("getAllWavelengths", &Class::getAllWavelengths);
    cls.def("getAllMasks", &Class::getAllMasks);
    cls.def("getAllCovariances", &Class::getAllCovariances);
    cls.def("getAllBackgrounds", &Class::getAllBackgrounds);

    // Pythonic APIs
    cls.def("__len__", &Class::size);
    cls.def("__getitem__", [](Class const& self, std::size_t i) {
            if (i >= self.size()) throw py::index_error();
            return self.get(i);
        });
    cls.def("__setitem__",
            [](Class& self, std::size_t i, PTR(Spectrum) spectrum) { self.set(i, spectrum); });

    cls.def("__getstate__", [](SpectrumSet const& self) { return py::make_tuple(self.getInternal()); });
    cls.def("__setstate__",
            [](SpectrumSet & self, py::tuple const& t) {
            new (&self) SpectrumSet(t[0].cast<SpectrumSet::Collection>());
        });
}

PYBIND11_PLUGIN(spectra) {
    py::module mod("spectra");

    declareReferenceLine(mod);
    declareSpectrum(mod);
    declareSpectrumSet(mod);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
