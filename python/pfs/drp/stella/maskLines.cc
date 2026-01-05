#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/base.h"
#include "pfs/drp/stella/maskLines.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

PYBIND11_MODULE(maskLines, mod) {
    mod.def("maskLines", &maskLines, "wavelength"_a, "lines"_a, "maskRadius"_a, "sortedLines"_a=false);
}

} // anonymous namespace

}}} // pfs::drp::stella
