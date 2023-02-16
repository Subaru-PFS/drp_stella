#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/Spectrum.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

void declareSpectrum(py::module &mod) {
    using Class = Spectrum;
    py::class_<Class, std::shared_ptr<Class>> cls(mod, "Spectrum");

    cls.def(py::init<std::size_t, int>(), "length"_a, "fiberId"_a=0);
    cls.def(py::init<Spectrum::ImageArray const&, Spectrum::Mask const&, Spectrum::ImageArray const&,
                     Spectrum::ImageArray const&, Spectrum::CovarianceMatrix const&,
                     Spectrum::WavelengthArray const&, int, std::shared_ptr<lsst::daf::base::PropertySet>>(),
            "flux"_a, "mask"_a, "background"_a, "norm"_a, "covariance"_a, "wavelength"_a, "fiberId"_a=0,
            "notes"_a=nullptr);

    cls.def("getFlux", py::overload_cast<>(&Class::getFlux));
    cls.def("setFlux", &Class::setFlux, "flux"_a);
    cls.def_property("flux", py::overload_cast<>(&Class::getFlux), &Class::setFlux);

    // "spectrum" is a synonym for "flux"
    cls.def("getSpectrum", py::overload_cast<>(&Class::getSpectrum));
    cls.def("setSpectrum", &Class::setSpectrum, "spectrum"_a);
    cls.def_property("spectrum", py::overload_cast<>(&Class::getSpectrum), &Class::setSpectrum);

    cls.def("getBackground", py::overload_cast<>(&Class::getBackground));
    cls.def("setBackground", &Class::setBackground, "background"_a);
    cls.def_property("background", py::overload_cast<>(&Class::getBackground), &Class::setBackground);

    cls.def("getNorm", py::overload_cast<>(&Class::getNorm));
    cls.def("setNorm", &Class::setNorm, "norm"_a);
    cls.def_property("norm", py::overload_cast<>(&Class::getNorm), &Class::setNorm);

    cls.def("getVariance", py::overload_cast<>(&Class::getVariance));
    cls.def("setVariance", &Class::setVariance, "variance"_a);
    cls.def_property("variance", py::overload_cast<>(&Class::getVariance), &Class::setVariance);

    cls.def("getCovariance", py::overload_cast<>(&Class::getCovariance));
    cls.def("setCovariance", &Class::setCovariance, "covariance"_a);
    cls.def_property("covariance", py::overload_cast<>(&Class::getCovariance), &Class::setCovariance);

    cls.def("getWavelength", py::overload_cast<>(&Class::getWavelength));
    cls.def("setWavelength", &Class::setWavelength, "wavelength"_a);
    cls.def_property("wavelength", py::overload_cast<>(&Class::getWavelength), &Class::setWavelength);

    cls.def("getMask", [](Class & self) { return self.getMask(); },
            py::return_value_policy::reference_internal);
    cls.def("setMask", &Class::setMask, "mask"_a);
    cls.def_property("mask", [](Class & self) { return self.getMask(); }, &Class::setMask,
                     py::return_value_policy::reference_internal);

    cls.def("getFiberId", &Class::getFiberId);
    cls.def("setFiberId", &Class::setFiberId, "fiberId"_a);
    cls.def_property("fiberId", &Class::getFiberId, &Class::setFiberId);

    cls.def("getNotes", py::overload_cast<>(&Class::getNotes), py::return_value_policy::reference_internal);
    cls.def_property_readonly("notes", py::overload_cast<>(&Class::getNotes));

    cls.def("getNumPixels", &Class::getNumPixels);
    cls.def("__len__", &Class::getNumPixels);

    cls.def("isWavelengthSet", &Class::isWavelengthSet);

    cls.def("getNormFlux", &Class::getNormFlux);
    cls.def_property_readonly("normFlux", &Class::getNormFlux);

    cls.def(py::pickle(
        [](Class const& self) {
            return py::make_tuple(self.getSpectrum(), self.getMask(), self.getBackground(), self.getNorm(),
                                  self.getCovariance(), self.getWavelength(), self.getFiberId(),
                                  self.getNotes().deepCopy());
        },
        [](py::tuple const& t) {
            return Spectrum(t[0].cast<Spectrum::ImageArray>(), t[1].cast<Spectrum::Mask>(),
                            t[2].cast<Spectrum::ImageArray>(), t[3].cast<Spectrum::ImageArray>(),
                            t[4].cast<Spectrum::CovarianceMatrix>(), t[5].cast<Spectrum::WavelengthArray>(),
                            t[6].cast<std::size_t>(),
                            t[7].cast<std::shared_ptr<lsst::daf::base::PropertySet>>());
        }
    ));
}


PYBIND11_PLUGIN(Spectrum) {
    py::module mod("Spectrum");
    declareSpectrum(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
