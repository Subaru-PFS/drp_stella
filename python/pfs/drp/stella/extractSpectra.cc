#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/extractSpectra.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


PYBIND11_MODULE(extractSpectra, mod) {
    mod.def(
        "extractSpectra",
        &extractSpectra,
        "image"_a,
        "fiberTraces"_a,
        "badBitMask"_a=0,
        "xBlockSize"_a=100,
        "yBlockSize"_a=100,
        "minFracMask"_a=0.3,
        "minFracImage"_a=0.4
    );
}


} // anonymous namespace

}}} // pfs::drp::stella
