#include <sstream>
#include <stdexcept>

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
    cls.def("getFirstMoment", &KernelSolution::getFirstMoment);
    cls.def_property_readonly("firstMoment", &KernelSolution::getFirstMoment);
    cls.def("getSecondMoment", &KernelSolution::getSecondMoment);
    cls.def_property_readonly("secondMoment", &KernelSolution::getSecondMoment);
    cls.def("__repr__", [](KernelSolution const& self) {
        std::ostringstream os;
        os << self;
        return os.str();
    });
    cls.def(py::pickle(
        [](KernelSolution const& self) {
            return py::make_tuple(
                self.bbox, self.kernelHalfWidth, self.backgroundOrder, self.kernel, self.background,
                self.rejected, self.success, self.numPixels, self.numFit, self.numRejected,
                self.numIter, self.chi2, self.rms
            );
        },
        [](py::tuple const& state) {
            if (state.size() != 13) {
                throw std::runtime_error("Invalid state for KernelSolution");
            }
            return KernelSolution(
                state[0].cast<lsst::geom::Box2I>(),
                state[1].cast<int>(),
                state[2].cast<int>(),
                state[3].cast<ndarray::Array<double, 2, 2>>(),
                state[4].cast<ndarray::Array<double, 1, 1>>(),
                state[5].cast<ndarray::Array<bool, 2, 2>>(),
                state[6].cast<bool>(),
                state[7].cast<std::size_t>(),
                state[8].cast<std::size_t>(),
                state[9].cast<std::size_t>(),
                state[10].cast<int>(),
                state[11].cast<double>(),
                state[12].cast<double>()
            );
        }
    ));
}


void declareAlardLuptonResult(py::module & mod) {
    py::class_<AlardLuptonResult> cls(mod, "AlardLuptonResult");
    cls.def_readonly("convolved", &AlardLuptonResult::convolved);
    cls.def_readonly("difference", &AlardLuptonResult::difference);
    cls.def_readonly("solutions", &AlardLuptonResult::solutions);
    cls.def_readonly("numRegionsX", &AlardLuptonResult::numRegionsX);
    cls.def_readonly("numRegionsY", &AlardLuptonResult::numRegionsY);
    cls.def_readonly("kernelHalfWidth", &AlardLuptonResult::kernelHalfWidth);
    cls.def_readonly("commonKernelSum", &AlardLuptonResult::commonKernelSum);
    cls.def_readonly("commonKernelSumScatter", &AlardLuptonResult::commonKernelSumScatter);
    cls.def("getSolutionAt", &AlardLuptonResult::getSolutionAt, "x"_a, "y"_a);
    cls.def("getChi2", &AlardLuptonResult::getChi2);
    cls.def_property_readonly("chi2", &AlardLuptonResult::getChi2);
    cls.def("getNumFit", &AlardLuptonResult::getNumFit);
    cls.def_property_readonly("numFit", &AlardLuptonResult::getNumFit);
    cls.def("getNumRejected", &AlardLuptonResult::getNumRejected);
    cls.def_property_readonly("numRejected", &AlardLuptonResult::getNumRejected);
    cls.def("getKernelFirstMoments", &AlardLuptonResult::getKernelFirstMoments);
    cls.def_property_readonly("kernelFirstMoments", &AlardLuptonResult::getKernelFirstMoments);
    cls.def("getKernelSecondMoments", &AlardLuptonResult::getKernelSecondMoments);
    cls.def_property_readonly("kernelSecondMoments", &AlardLuptonResult::getKernelSecondMoments);
    cls.def(py::pickle(
        [](AlardLuptonResult const& self) {
            return py::make_tuple(
                self.convolved, self.difference, self.solutions, self.numRegionsX, self.numRegionsY,
                self.kernelHalfWidth, self.commonKernelSum, self.commonKernelSumScatter
            );
        },
        [](py::tuple const& state) {
            if (state.size() != 8) {
                throw std::runtime_error("Invalid state for AlardLuptonResult");
            }
            return AlardLuptonResult(
                state[0].cast<lsst::afw::image::MaskedImage<float>>(),
                state[1].cast<lsst::afw::image::MaskedImage<float>>(),
                state[2].cast<std::vector<KernelSolution>>(),
                state[3].cast<int>(),
                state[4].cast<int>(),
                state[5].cast<int>(),
                state[6].cast<double>(),
                state[7].cast<double>()
            );
        }
    ));
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
        "lsqThreshold"_a=1.0e-6,
        "commonKernelSum"_a=false,
        "minSignalToNoise"_a=0.0
    );
}


} // anonymous namespace

}}} // pfs::drp::stella
