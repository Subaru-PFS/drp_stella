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


PYBIND11_PLUGIN(Spectrum) {
    py::module mod("Spectrum");
    declareSpectrum(mod);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
