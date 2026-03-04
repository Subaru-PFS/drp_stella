#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/base.h"
#include "lsst/cpputils/python.h"
#include "pfs/drp/stella/GridTransform.h"

namespace py = pybind11;

using namespace pybind11::literals;

namespace pfs { namespace drp { namespace stella {

namespace {


void declareTriangle(py::module_ & mod) {
    py::class_<Triangle> cls(mod, "Triangle");
    cls.def(
        py::init<lsst::geom::Point2D const&, lsst::geom::Point2D const&, lsst::geom::Point2D const&>(),
        "a"_a, "b"_a, "c"_a
    );
    cls.def(py::init<>());
    cls.def("isEmpty", &Triangle::isEmpty);
    cls.def("getArea", &Triangle::getArea);
    cls.def("getVertex1", &Triangle::getVertex1);
    cls.def("getVertex2", &Triangle::getVertex2);
    cls.def("getVertex3", &Triangle::getVertex3);
    cls.def("getVertices", &Triangle::getVertices);
    cls.def(
        "contains",
        py::overload_cast<double, double>(&Triangle::contains, py::const_),
        "x"_a, "y"_a
    );
    cls.def(
        "contains",
        py::overload_cast<lsst::geom::Point2D const&>(&Triangle::contains, py::const_),
        "point"_a
    );
    cls.def(
        "contains",
        py::overload_cast<lsst::geom::Point3D const&>(&Triangle::contains, py::const_),
        "barycentric"_a
    );
    cls.def("getCentroid", &Triangle::getCentroid);
    cls.def("getBarycentricCoordinates", &Triangle::getBarycentricCoordinates, "point"_a);
    cls.def("interpolate", &Triangle::interpolate, "barycentric"_a);
}


/// Wrapper for GridTree to hold copies of the input arrays
///
/// GridTree holds references to the input arrays, which is fine in C++ when
/// you're holding onto the arrays for the lifetime of the GridTree. But when
/// we create a GridTree in Python, we need to make sure the arrays don't get
/// garbage collected while the GridTree is still around.
///
/// Uses composition instead of inheritance, because I want the wrapper to copy
/// the input arrays first (with inheritance, the parent constructor gets called
/// before the child constructor).
class GridTreeWrapper {
  public:
    using Array2D = ndarray::Array<double, 2, 1>;

    GridTreeWrapper(
        Array2D const& x, Array2D const& y, int maxPointsPerLeaf=4
    ) : _xCopy(ndarray::copy(x)),
        _yCopy(ndarray::copy(y)),
        _tree(_xCopy.deep(), _yCopy.deep(), maxPointsPerLeaf) {}

    GridTreeWrapper(GridTree const& tree)
      : _xCopy(tree.getX().deep()), _yCopy(tree.getY().deep()), _tree(tree) {}

    lsst::geom::Point2I find(
        double x, double y,
        double distance=std::numeric_limits<double>::infinity()
    ) const {
        return _tree.find(x, y, distance);
    }

    lsst::geom::Point2I find(
        lsst::geom::Point2D const& point,
        double distance=std::numeric_limits<double>::infinity()
    ) const {
        return _tree.find(point, distance);
    }

    // Add bounds checking to getValue, since python users expect it
    lsst::geom::Point2D getValue(lsst::geom::Point2I const& point) const {
        if (point.getX() < 0 || point.getX() >= int(_xCopy.getShape()[0]) ||
            point.getY() < 0 || point.getY() >= int(_xCopy.getShape()[1])) {
            throw std::out_of_range("Point is out of bounds");
        }
        return _tree.getValue(point);
    }
    lsst::geom::Point2D getValue(int ii, int jj) const {
        return getValue(lsst::geom::Point2I(ii, jj));
    }

    std::pair<
        std::tuple<lsst::geom::Point2I, lsst::geom::Point2I, lsst::geom::Point2I>,
        Triangle
    > findTriangle(
        double x, double y
    ) const {
        return _tree.findTriangle(x, y);
    }
    std::pair<
        std::tuple<lsst::geom::Point2I, lsst::geom::Point2I, lsst::geom::Point2I>,
        Triangle
    > findTriangle(
        lsst::geom::Point2D const& point
    ) const {
        return _tree.findTriangle(point);
    }

  private:
    Array2D _xCopy;
    Array2D _yCopy;
    GridTree _tree;
};


void declareGridTree(py::module_ & mod) {
    py::class_<GridTreeWrapper> cls(mod, "GridTree");
    cls.def(
        py::init<GridTreeWrapper::Array2D const&, GridTreeWrapper::Array2D const&, int>(),
        "x"_a, "y"_a, "maxPointsPerLeaf"_a=4
    );
    double const infinity = std::numeric_limits<double>::infinity();
    cls.def(
        "find",
        py::overload_cast<double, double, double>(&GridTreeWrapper::find, py::const_),
        "x"_a, "y"_a, "distance"_a=infinity
    );
    cls.def(
        "find",
        py::overload_cast<lsst::geom::Point2D const&, double>(
            &GridTreeWrapper::find, py::const_
        ),
        "point"_a, "distance"_a=infinity
    );
    cls.def(
        "getValue",
        py::overload_cast<lsst::geom::Point2I const&>(&GridTreeWrapper::getValue, py::const_),
        "point"_a
    );
    cls.def("getValue", py::overload_cast<int, int>(&GridTreeWrapper::getValue, py::const_), "i"_a, "j"_a);
    cls.def(
        "findTriangle",
        py::overload_cast<double, double>(&GridTreeWrapper::findTriangle, py::const_),
        "x"_a, "y"_a
    );
    cls.def(
        "findTriangle",
        py::overload_cast<lsst::geom::Point2D const&>(&GridTreeWrapper::findTriangle, py::const_),
        "point"_a
    );
}


void declareGridTransform(py::module_ & mod) {
    py::class_<GridTransform> cls(mod, "GridTransform");
    cls.def(
        py::init<
            GridTransform::Array2D const&, GridTransform::Array2D const&,
            GridTransform::Array2D const&, GridTransform::Array2D const&
        >(),
        "u"_a, "v"_a, "x"_a, "y"_a
    );
    cls.def(
        py::init<
            GridTransform::Array2D const&, GridTransform::Array2D const&,
            GridTransform::Array2D const&, GridTransform::Array2D const&,
            GridTransform::DistortionList const&
        >(),
        "u"_a, "v"_a, "x"_a, "y"_a, "distortions"_a
    );
     cls.def(
        py::init<
            GridTransform::Array2D const&, GridTransform::Array2D const&,
            GridTransform::Array2D const&, GridTransform::Array2D const&,
            std::shared_ptr<Distortion> const&
        >(),
        "u"_a, "v"_a, "x"_a, "y"_a, "distortion"_a
    );
    cls.def("getU", &GridTransform::getU);
    cls.def("getV", &GridTransform::getV);
    cls.def("getX", &GridTransform::getX);
    cls.def("getY", &GridTransform::getY);
    cls.def("getTree", [](GridTransform const& self) { return GridTreeWrapper(self.getTree()); });
    cls.def("inverse", &GridTransform::inverse);
    cls.def(
        "__call__",
        py::overload_cast<double, double>(&GridTransform::calculateXY, py::const_),
        "u"_a, "v"_a
    );
    cls.def(
        "__call__",
        py::overload_cast<GridTransform::Array1D const&, GridTransform::Array1D const&>(
            &GridTransform::calculateXY, py::const_
        ),
        "u"_a, "v"_a
    );
    cls.def(
        "__call__",
        py::overload_cast<lsst::geom::Point2D const&>(&GridTransform::calculateXY, py::const_),
        "uv"_a
    );
    cls.def(
        "calculateXY",
        py::overload_cast<double, double>(&GridTransform::calculateXY, py::const_),
        "u"_a, "v"_a
    );
    cls.def(
        "calculateXY",
        py::overload_cast<GridTransform::Array1D const&, GridTransform::Array1D const&>(
            &GridTransform::calculateXY, py::const_
        ),
        "u"_a, "v"_a
    );
    cls.def(
        "calculateXY",
        py::overload_cast<lsst::geom::Point2D const&>(&GridTransform::calculateXY, py::const_),
        "uv"_a
    );
    cls.def(
        "calculateX",
        py::overload_cast<double, double>(&GridTransform::calculateX, py::const_),
        "u"_a, "v"_a
    );
    cls.def(
        "calculateX",
        py::overload_cast<GridTransform::Array1D const&, GridTransform::Array1D const&>(
            &GridTransform::calculateX, py::const_
        ),
        "u"_a, "v"_a
    );
    cls.def(
        "calculateX",
        py::overload_cast<lsst::geom::Point2D const&>(&GridTransform::calculateX, py::const_),
        "uv"_a
    );
    cls.def(
        "calculateY",
        py::overload_cast<double, double>(&GridTransform::calculateY, py::const_),
        "u"_a, "v"_a
    );
    cls.def(
        "calculateY",
        py::overload_cast<GridTransform::Array1D const&, GridTransform::Array1D const&>(
            &GridTransform::calculateY, py::const_
        ),
        "u"_a, "v"_a
    );
    cls.def(
        "calculateY",
        py::overload_cast<lsst::geom::Point2D const&>(&GridTransform::calculateY, py::const_),
        "uv"_a
    );
}


PYBIND11_MODULE(GridTransform, mod) {
    declareTriangle(mod);
    declareGridTree(mod);
    declareGridTransform(mod);
}


} // anonymous namespace

}}} // pfs::drp::stella
