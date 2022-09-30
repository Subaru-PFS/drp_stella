#ifndef PFS_DRP_STELLA_PYTHON_MATH_H
#define PFS_DRP_STELLA_PYTHON_MATH_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"
#include "pfs/drp/stella/math/quartiles.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs {
namespace drp {
namespace stella {
namespace python {

//@{
/// Add pybind11 bindings for calculateMedian and calculateQuartiles
template <typename T, int C1, int C2>
auto wrapQuartiles(py::module & mod) {
    mod.def(
        "calculateMedian",
        py::overload_cast<
            ndarray::Array<T, 1, C1> const&, ndarray::Array<bool, 1, C2> const&
        >(math::calculateMedian<T, C1, C2>),
         "values"_a,
         "mask"_a
    );
    mod.def(
        "calculateQuartiles",
        py::overload_cast<
            ndarray::Array<T, 1, C1> const&, ndarray::Array<bool, 1, C2> const&
        >(math::calculateQuartiles<T, C1, C2>),
         "values"_a,
         "mask"_a
    );
}

template <typename T>
auto wrapQuartiles(py::module & mod) {
    wrapQuartiles<T, 0, 0>(mod);
    wrapQuartiles<T, 1, 0>(mod);
    wrapQuartiles<T, 0, 1>(mod);
    wrapQuartiles<T, 1, 1>(mod);
}
//@}


}}}}  // namespace pfs::drp::stella::python

#endif