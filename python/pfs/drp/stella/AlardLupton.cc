#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "pfs/drp/stella/AlardLupton.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declareKernelSolution(py::module & mod) {
    py::class_<KernelSolution> cls(mod, "KernelSolution");
    cls.def_readonly("bbox", &KernelSolution::bbox);
    cls.def_readonly("kernelHalfWidth", &KernelSolution::kernelHalfWidth);
    cls.def_readonly("backgroundOrder", &KernelSolution::backgroundOrder);
    cls.def_readonly("kernel", &KernelSolution::kernel);
    cls.def_readonly("background", &KernelSolution::background);
    cls.def_readonly("rejected", &KernelSolution::rejected);
    cls.def_readonly("success", &KernelSolution::success);
    cls.def_readonly("numPixels", &KernelSolution::numPixels);
    cls.def_readonly("numFit", &KernelSolution::numFit);
    cls.def_readonly("numRejected", &KernelSolution::numRejected);
    cls.def_readonly("numIter", &KernelSolution::numIter);
    cls.def_readonly("chi2", &KernelSolution::chi2);
    cls.def_readonly("rms", &KernelSolution::rms);
    cls.def("getNumParams", &KernelSolution::getNumParams);
    cls.def_property_readonly("numParams", &KernelSolution::getNumParams);
    cls.def("getKernelSum", &KernelSolution::getKernelSum);
    cls.def_property_readonly("kernelSum", &KernelSolution::getKernelSum);
    cls.def("getDegreesOfFreedom", &KernelSolution::getDegreesOfFreedom);
    cls.def_property_readonly("degreesOfFreedom", &KernelSolution::getDegreesOfFreedom);
    cls.def("getReducedChi2", &KernelSolution::getReducedChi2);
    cls.def_property_readonly("reducedChi2", &KernelSolution::getReducedChi2);
    cls.def("__repr__", [](KernelSolution const& self) {
        std::ostringstream os;
        os << self;
        return os.str();
    });
}


void declareAlardLuptonResult(py::module & mod) {
    py::class_<AlardLuptonResult> cls(mod, "AlardLuptonResult");
    cls.def_readonly("difference", &AlardLuptonResult::difference);
    cls.def_readonly("solutions", &AlardLuptonResult::solutions);
    cls.def_readonly("numRegionsX", &AlardLuptonResult::numRegionsX);
    cls.def_readonly("numRegionsY", &AlardLuptonResult::numRegionsY);
    cls.def_readonly("kernelHalfWidth", &AlardLuptonResult::kernelHalfWidth);
    cls.def("getSolutionAt", &AlardLuptonResult::getSolutionAt, "x"_a, "y"_a);
    cls.def("getChi2", &AlardLuptonResult::getChi2);
    cls.def_property_readonly("chi2", &AlardLuptonResult::getChi2);
    cls.def("getNumFit", &AlardLuptonResult::getNumFit);
    cls.def_property_readonly("numFit", &AlardLuptonResult::getNumFit);
    cls.def("getNumRejected", &AlardLuptonResult::getNumRejected);
    cls.def_property_readonly("numRejected", &AlardLuptonResult::getNumRejected);
}


PYBIND11_MODULE(AlardLupton, mod) {
    declareKernelSolution(mod);
    declareAlardLuptonResult(mod);
    mod.def(
        "fitAlardLuptonKernel",
        &fitAlardLuptonKernel,
        "source"_a,
        "target"_a,
        "kernelHalfWidth"_a=10,
        "numRegionsX"_a=1,
        "numRegionsY"_a=1,
        "backgroundOrder"_a=1,
        "badBitMask"_a=0,
        "rejIter"_a=2,
        "rejThresh"_a=3.0,
        "lsqThreshold"_a=1.0e-6
    );
}


} // anonymous namespace

}}} // pfs::drp::stella
