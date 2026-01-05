#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/PolynomialDetectorMap.h"
#include "pfs/drp/stella/python/DistortionBasedDetectorMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

PYBIND11_MODULE(PolynomialDetectorMap, mod) {
    pybind11::module::import("pfs.drp.stella.DetectorMap");
    python::wrapDistortionBasedDetectorMap<PolynomialDetectorMap>(mod, "PolynomialDetectorMap");
}

}}} // pfs::drp::stella
