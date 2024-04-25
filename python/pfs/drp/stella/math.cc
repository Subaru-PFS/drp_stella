#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/afw/math/FunctionLibrary.h"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/math/quartiles.h"
#include "pfs/drp/stella/math/NormalizedPolynomial.h"
#include "pfs/drp/stella/math/solveLeastSquares.h"
#include "pfs/drp/stella/math/SparseSquareMatrix.h"
#include "pfs/drp/stella/python/math.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella { namespace math {

namespace {


template <typename T>
void declareNormalizedPolynomial1(py::module & mod, std::string const& suffix) {
    using Class = NormalizedPolynomial1<T>;
    py::class_<Class, std::shared_ptr<Class>, lsst::afw::math::PolynomialFunction1<T>>
            cls(mod, ("NormalizedPolynomial1" + suffix).c_str());

    cls.def(py::init<ndarray::Array<double, 1, 1> const &, double, double>(),
            "params"_a, "min"_a=-1.0, "max"_a=1.0);
    cls.def(py::init<unsigned int, double, double>(), "order"_a, "min"_a=-1.0, "max"_a=1.0);

    cls.def("__call__", py::overload_cast<double>(&Class::operator(), py::const_), "x"_a);
    cls.def("__call__",
            py::overload_cast<ndarray::Array<double, 1, 1> const&>(&Class::operator(), py::const_), "x"_a);
    cls.def("clone", &Class::clone);
    cls.def("getOrder", &Class::getOrder);
    cls.def("getDFuncDParameters", &Class::getDFuncDParameters);
    cls.def("getMin", &Class::getMin);
    cls.def("getMax", &Class::getMax);
}


template <typename T>
void declareNormalizedPolynomial2(py::module & mod, std::string const& suffix) {
    using Class = NormalizedPolynomial2<T>;
    py::class_<Class, std::shared_ptr<Class>, lsst::afw::math::BasePolynomialFunction2<T>>
            cls(mod, ("NormalizedPolynomial2" + suffix).c_str());

    cls.def(py::init<unsigned int, lsst::geom::Box2D const&>(),
            "order"_a, "range"_a=lsst::geom::Box2D(lsst::geom::Point2D(-1, 1), lsst::geom::Point2D(-1, 1)));
    cls.def(py::init<ndarray::Array<double, 1, 1> const &, lsst::geom::Box2D const&>(),
            "params"_a, "range"_a=lsst::geom::Box2D(lsst::geom::Point2D(-1, 1), lsst::geom::Point2D(-1, 1)));

    cls.def("__call__", py::overload_cast<double, double>(&Class::operator(), py::const_), "x"_a, "y"_a);
    cls.def("__call__",
            py::overload_cast<ndarray::Array<double, 1, 1> const&,
                              ndarray::Array<double, 1, 1> const&>(&Class::operator(), py::const_),
            "x"_a, "y"_a);
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


template <typename T, int N, int C>
std::pair<ndarray::Array<T, N, C>, ndarray::Array<T, N, N>> evaluateAffineTransform(
    lsst::geom::AffineTransform const& transform,
    ndarray::Array<T, N, C> const& xIn,
    ndarray::Array<T, N, C> const& yIn
) {
    utils::checkSize(xIn.getShape(), yIn.getShape(), "x vs y");
    ndarray::Array<T, N, C> xOut = ndarray::allocate(xIn.getShape());
    ndarray::Array<T, N, C> yOut = ndarray::allocate(yIn.getShape());
    auto xIter = xIn.begin();
    auto yIter = yIn.begin();
    for (auto xx = xOut.begin(), yy = yOut.begin(); xx != xOut.end(); ++xx, ++yy, ++xIter, ++yIter) {
        auto const result = transform(lsst::geom::Point2D(*xIter, *yIter));
        *xx = result.getX();
        *yy = result.getY();
    }
    return std::make_pair(xOut, yOut);
}


void declareMatrixTriplets(py::module & mod) {
    using ElemT = double;
    using IndexT = std::ptrdiff_t;
    using Class = MatrixTriplets<ElemT, IndexT>;
    py::class_<Class> cls(mod, "MatrixTriplets");
    cls.def(py::init<IndexT, IndexT, float>(), "numRows"_a, "numCols"_a, "nonZeroPerRow"_a=2.0);
    cls.def("size", &Class::size);
    cls.def("__len__", &Class::size);
    cls.def("add", &Class::add, "row"_a, "col"_a, "value"_a);
    cls.def("clear", &Class::clear);
    cls.def("getTriplets", [](Class & self) {
        ndarray::Array<IndexT, 1, 1> rows = ndarray::allocate(self.size());
        ndarray::Array<IndexT, 1, 1> cols = ndarray::allocate(self.size());
        ndarray::Array<ElemT, 1, 1> values = ndarray::allocate(self.size());
        auto iter = self.begin();
        for (std::size_t ii = 0; iter != self.end(); ++ii, ++iter) {
            rows[ii] = iter->row();
            cols[ii] = iter->col();
            values[ii] = iter->value();
        }
        return std::make_tuple(rows, cols, values);
    });
}


template <bool symmetric>
void declareSparseSquareMatrix(py::module & mod, char const* name) {
    using Class = SparseSquareMatrix<symmetric>;
    py::class_<Class> cls(mod, name);
    cls.def(py::init<std::size_t, float>(), "size"_a, "nonZeroPerRow"_a=2.0);
    cls.def("size", &Class::size);
    cls.def("__len__", &Class::size);
    cls.def("add", &Class::add, "row"_a, "col"_a, "value"_a);
    cls.def(
        "solve",
        py::overload_cast<ndarray::Array<double, 1, 1> const&, bool>(&Class::template solve<>, py::const_),
        "rhs"_a,
        "debug"_a=false
    );
    cls.def(
        "solve",
        py::overload_cast<ndarray::Array<double, 1, 1> &, ndarray::Array<double, 1, 1> const&, bool>(
            &Class::template solve<>, py::const_
        ),
        "solution"_a,
        "rhs"_a,
        "debug"_a=false
    );
    cls.def("reset", &Class::reset);
    cls.def("getTriplets", &Class::getTriplets);
}


PYBIND11_PLUGIN(math) {
    py::module mod("math");
    py::module::import("lsst.afw.math");
    declareNormalizedPolynomial1<double>(mod, "D");
    declareNormalizedPolynomial2<double>(mod, "D");
    python::wrapQuartiles<float>(mod);
    python::wrapQuartiles<double>(mod);
    mod.def("evaluatePolynomial",
            &evaluateFunction2<lsst::afw::math::Chebyshev1Function2<double>, double, 1, 1>,
            "poly"_a, "x"_a, "y"_a);
    mod.def("evaluatePolynomial",
            &evaluateFunction2<lsst::afw::math::PolynomialFunction2<double>, double, 1, 1>,
            "poly"_a, "x"_a, "y"_a);
    mod.def("evaluatePolynomial",
            &evaluateFunction2<NormalizedPolynomial2<double>, double, 1, 1>,
            "poly"_a, "x"_a, "y"_a);
    mod.def("evaluateAffineTransform", &evaluateAffineTransform<double, 1, 1>, "transform"_a, "x"_a, "y"_a);
    mod.def("solveLeastSquaresDesign", &solveLeastSquaresDesign, "design"_a, "meas"_a,
            "err"_a, "threshold"_a=1.0e-6, "forced"_a=nullptr, "params"_a=nullptr);
    declareMatrixTriplets(mod);
    declareSparseSquareMatrix<false>(mod, "NonsymmetricSparseSquareMatrix");
    declareSparseSquareMatrix<true>(mod, "SymmetricSparseSquareMatrix");
    return mod.ptr();
}

} // anonymous namespace

}}}} // pfs::drp::stella::math
