#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "pfs/drp/stella/math/Math.h"   // for pfs::drp::stella::math::calcMinCenMax
#include "pfs/drp/stella/utils/Utils.h" // for pfs::drp::stella::utils::testPolyFit

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {

PYBIND11_PLUGIN(_math) {
    py::module mod("_math");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    py::class_<math::CrossCorrelateResult, std::shared_ptr<math::CrossCorrelateResult>>
            cls(mod, "CrossCorrelateResult");
    cls.def_readwrite("pixShift", &math::CrossCorrelateResult::pixShift);
    cls.def_readwrite("chiSquare", &math::CrossCorrelateResult::chiSquare);

    mod.def("crossCorrelate", &math::crossCorrelate<float>,
            "DA1_Static"_a, "DA1_Moving"_a, "I_NPixMaxLeft"_a, "I_NPixMaxRight"_a);
    mod.def("calcMinCenMax", &math::calcMinCenMax<float, float>);
    mod.def("firstIndexWithValueGEFrom", &math::firstIndexWithValueGEFrom<long>,
            "vecIn"_a, "minValue"_a, "fromIndex"_a=0);
    mod.def("firstIndexWithValueGEFrom", &math::firstIndexWithValueGEFrom<int>,
            "vecIn"_a, "minValue"_a, "fromIndex"_a=0);

    // Doesn't really belong here, but putting it in its own file would be overkill
    mod.def("testPolyFit", &utils::testPolyFit);

    return mod.ptr();
}

} // anonymous namespace

}}} // pfs::drp::stella
