#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/DistortedDetectorMap.h"
#include "pfs/drp/stella/python/DistortionBasedDetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

PYBIND11_MODULE(DistortedDetectorMap, mod) {
    pybind11::module::import("pfs.drp.stella.DetectorMap");
    python::wrapDistortionBasedDetectorMap<DistortedDetectorMap>(mod, "DistortedDetectorMap");
}

}}} // pfs::drp::stella
