#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/utils/python.h"
#include "lsst/afw/table.h"
#include "pfs/drp/stella/photometry.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


PYBIND11_PLUGIN(photometry) {
    py::module mod("photometry");
    mod.def("photometer", &photometer, "image"_a, "fiberId"_a, "wavelength"_a, "psf"_a, "badBitMask"_a=0,
            "positions"_a=nullptr);
    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
