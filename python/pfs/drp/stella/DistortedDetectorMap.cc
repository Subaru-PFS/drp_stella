#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/utils/python.h"
#include "pfs/drp/stella/DistortedDetectorMap.h"
#include "pfs/drp/stella/python/DistortionBasedDetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

PYBIND11_PLUGIN(DistortedDetectorMap) {
    py::module mod("DistortedDetectorMap");
    pybind11::module::import("pfs.drp.stella.DetectorMap");
    python::wrapDistortionBasedDetectorMap<DistortedDetectorMap>(mod, "DistortedDetectorMap");
    return mod.ptr();
}

}}} // pfs::drp::stella
