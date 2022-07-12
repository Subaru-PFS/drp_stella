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


/// Add pybind11 bindings for calculateMedian and calculateQuartiles
template <typename T, int C>
auto wrapQuartiles(py::module & mod) {
    mod.def(
        "calculateMedian",
        py::overload_cast<
            ndarray::Array<T, 1, C> const&, ndarray::Array<bool, 1, C> const&
        >(math::calculateMedian<T, C>),
         "values"_a,
         "mask"_a
    );
    mod.def(
        "calculateQuartiles",
        py::overload_cast<
            ndarray::Array<T, 1, C> const&, ndarray::Array<bool, 1, C> const&
        >(math::calculateQuartiles<T, C>),
         "values"_a,
         "mask"_a
    );
}

}}}}  // namespace pfs::drp::stella::python

#endif