#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/base.h"
#include "pfs/drp/stella/math/symmetricTridiagonal.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella { namespace math {

namespace {

template <typename T>
void declareSymmetricTridiagonal(py::module &mod, std::string const& suffix) {
    py::class_<SymmetricTridiagonalWorkspace<T>> cls(
        mod, ("SymmetricTridiagonalWorkspace" + suffix).c_str());
    cls.def(py::init<>());
    cls.def("reset", &SymmetricTridiagonalWorkspace<T>::reset, "num"_a);
    cls.def_readonly("longArray1", &SymmetricTridiagonalWorkspace<T>::longArray1);
    cls.def_readonly("longArray2", &SymmetricTridiagonalWorkspace<T>::longArray2);
    cls.def_readonly("shortArray", &SymmetricTridiagonalWorkspace<T>::shortArray);

    mod.def(("solveSymmetricTridiagonal" + suffix).c_str(), &solveSymmetricTridiagonal<T>,
            "diagonal"_a, "offDiag"_a, "answer"_a, "workspace"_a=SymmetricTridiagonalWorkspace<T>());
    mod.def(("invertSymmetricTridiagonal" + suffix).c_str(), &invertSymmetricTridiagonal<T>,
             "diagonal"_a, "offDiag"_a, "workspace"_a=SymmetricTridiagonalWorkspace<T>());
}

PYBIND11_PLUGIN(symmetricTridiagonal) {
    py::module mod("symmetricTridiagonal");
    declareSymmetricTridiagonal<double>(mod, "");

    return mod.ptr();
}

} // anonymous namespace

}}}} // pfs::drp::stella::math
