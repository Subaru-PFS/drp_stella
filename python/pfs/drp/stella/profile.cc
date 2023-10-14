#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/profile.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


PYBIND11_PLUGIN(profile) {
    py::module mod("profile");
    py::module::import("lsst.afw.image");

    py::class_<FitProfilesResults> cls(mod, "FitProfilesResults");
    cls.def_readonly("profiles", &FitProfilesResults::profiles);
    cls.def_readonly("masks", &FitProfilesResults::masks);
    cls.def_readonly("backgrounds", &FitProfilesResults::backgrounds);

    mod.def("fitProfiles", &fitProfiles, "images"_a, "centers"_a, "spectra"_a, "fiberId"_a,
            "ySwaths"_a, "badBitMask"_a, "oversample"_a, "radius"_a, "bgSize"_a,
            "rejIter"_a=1, "rejThresh"_a=4.0, "matrixTol"_a=1e-4);
    mod.def("fitAmplitudes", &fitAmplitudes, "image"_a, "centers"_a, "sigma"_a,
            "badBitMask"_a=0, "maxSigma"_a=4.0);
    mod.def("calculateSwathProfile", &calculateSwathProfile, "values"_a, "mask"_a,
            "rejIter"_a=1, "rejThresh"_a=3.0);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
