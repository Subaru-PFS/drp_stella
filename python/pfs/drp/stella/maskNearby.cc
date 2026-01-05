#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/base.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/maskNearby.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

PYBIND11_MODULE(maskNearby, mod) {
    mod.def("maskNearby", &maskNearby, "values"_a, "distance"_a);
}

} // anonymous namespace

}}} // pfs::drp::stella
