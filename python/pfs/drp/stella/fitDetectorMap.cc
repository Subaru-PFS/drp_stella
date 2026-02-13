#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/base.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/fitDetectorMap.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


PYBIND11_MODULE(fitDetectorMap, mod) {
    mod.def(
        "fitDetectorMap",
        fitDetectorMap<float>,
        "order"_a,
        "base"_a,
        "fiberIdLine"_a,
        "wavelength"_a,
        "xLine"_a,
        "yLine"_a,
        "xErrLine"_a,
        "yErrLine"_a,
        "fiberIdTrace"_a,
        "xTrace"_a,
        "yTrace"_a,
        "xErrTrace"_a,
        "start"_a = nullptr
     );
}


} // anonymous namespace

}}} // pfs::drp::stella
