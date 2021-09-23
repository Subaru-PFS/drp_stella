#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/utils/python.h"
#include "pfs/drp/stella/DoubleDetectorMap.h"
#include "pfs/drp/stella/python/DistortionBasedDetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

PYBIND11_PLUGIN(DoubleDetectorMap) {
    py::module mod("DoubleDetectorMap");
    pybind11::module::import("pfs.drp.stella.DetectorMap");
    python::wrapDistortionBasedDetectorMap<DoubleDetectorMap>(mod, "DoubleDetectorMap");
    return mod.ptr();
}

}}} // pfs::drp::stella
