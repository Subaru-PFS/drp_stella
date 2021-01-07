#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/afw/math/FunctionLibrary.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/math/quartiles.h"
#include "pfs/drp/stella/math/NormalizedPolynomial.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella { namespace math {

namespace {


template <typename T>
void declareNormalizedPolynomial(py::module & mod, std::string const& suffix) {
    using Class = NormalizedPolynomial2<T>;
    py::class_<Class, std::shared_ptr<Class>, lsst::afw::math::BasePolynomialFunction2<T>>
            cls(mod, ("NormalizedPolynomial2" + suffix).c_str());

    cls.def(py::init<unsigned int>(), "order"_a);
    cls.def(py::init<std::vector<double> const &, lsst::geom::Box2D const&>(),
            "params"_a, "range"_a=lsst::geom::Box2D(lsst::geom::Point2D(-1, 1), lsst::geom::Point2D(-1, 1)));

    cls.def("__call__", &Class::operator(), "x"_a, "y"_a);
    cls.def("clone", &Class::clone);
    cls.def("getOrder", &Class::getOrder);
    cls.def("getDFuncDParameters", &Class::getDFuncDParameters);
    cls.def("getXYRange", &Class::getXYRange);
}


// Evaluate a 2D function
template <typename FuncT, typename T, int N, int C>
ndarray::Array<T, N, C> evaluateFunction2(
    FuncT const& func,  // functor
    ndarray::Array<T, N, C> const& xx,  // x coordinate
    ndarray::Array<T, N, C> const& yy  // y coordinate
) {
    utils::checkSize(xx.getShape(), yy.getShape(), "x vs y");
    ndarray::Array<T, N, C> out = ndarray::allocate(xx.getShape());
    auto xIter = xx.begin();
    auto yIter = yy.begin();
    for (auto outIter = out.begin(); outIter != out.end(); ++outIter, ++xIter, ++yIter) {
        *outIter = func(*xIter, *yIter);
    }
    return out;
}


PYBIND11_PLUGIN(math) {
    py::module mod("math");
    declareNormalizedPolynomial<double>(mod, "D");
    mod.def("calculateQuartiles", &calculateQuartiles<double, 1>, "values"_a, "mask"_a);
    mod.def("evaluatePolynomial",
            &evaluateFunction2<lsst::afw::math::Chebyshev1Function2<double>, double, 1, 1>,
            "poly"_a, "x"_a, "y"_a);
    mod.def("evaluatePolynomial",
            &evaluateFunction2<lsst::afw::math::PolynomialFunction2<double>, double, 1, 1>,
            "poly"_a, "x"_a, "y"_a);
    mod.def("evaluatePolynomial",
            &evaluateFunction2<NormalizedPolynomial2<double>, double, 1, 1>,
            "poly"_a, "x"_a, "y"_a);
    return mod.ptr();
}

} // anonymous namespace

}}}} // pfs::drp::stella::math
