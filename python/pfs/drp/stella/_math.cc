#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "pfs/drp/stella/PSF.h"
#include "pfs/drp/stella/utils/Utils.h" // for pfs::drp::stella::utils::testPolyFit

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

PYBIND11_PLUGIN(_math) {
    py::module mod("_math");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    mod.def("calcMinCenMax", &math::calcMinCenMax<float, float>);

    // Doesn't really belong here, but putting it in its own file would be overkill
    mod.def("testPolyFit", &utils::testPolyFit);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella