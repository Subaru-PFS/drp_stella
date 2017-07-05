#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>

#include "pfs/drp/stella/Example.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

template <typename T>
void declareExample(py::module &mod) {
    mod.def("addImagesWithEigen", addImagesWithEigen<T>, "im1"_a, "im2"_a);
    mod.def("addImagesWithBlitz", addImagesWithBlitz<T>, "im1"_a, "im2"_a);
}

PYBIND11_PLUGIN(_example) {
    py::module mod("_example");

    declareExample<double>(mod);
    declareExample<float>(mod);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
