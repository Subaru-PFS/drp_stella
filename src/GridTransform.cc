#include <list>

#include "ndarray.h"

#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"

#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/GridTransform.h"

namespace pfs {
namespace drp {
namespace stella {


namespace {


double arrayLookup(ndarray::Array<double, 2, 1> const& array, lsst::geom::Point2I const& point) {
    assert(point.getX() >= 0 && point.getX() < int(array.getShape()[0]));
    assert(point.getY() >= 0 && point.getY() < int(array.getShape()[1]));
    return array[point.getX()][point.getY()];
}


}  // anonymous namespace


////////////////////////////////////////////////////////////////////////////////
// Triangle
////////////////////////////////////////////////////////////////////////////////


bool Triangle::isEmpty() const {
    if (_v1 == _v2 || _v2 == _v3) {
        return true;
    }
    double const term1 = (_v2.getX() - _v1.getX()) * (_v3.getY() - _v1.getY());
    double const term2 = (_v3.getX() - _v1.getX()) * (_v2.getY() - _v1.getY());
    return term1 - term2 < std::numeric_limits<double>::epsilon() * (std::abs(term1) + std::abs(term2));
}


double Triangle::getArea() const {
    double const x1 = _v1.getX();
    double const y1 = _v1.getY();
    double const x2 = _v2.getX();
    double const y2 = _v2.getY();
    double const x3 = _v3.getX();
    double const y3 = _v3.getY();
    return 0.5*std::abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2));
}


lsst::geom::Point2D Triangle::getCentroid() const {
    double const x = (_v1.getX() + _v2.getX() + _v3.getX()) / 3.0;
    double const y = (_v1.getY() + _v2.getY() + _v3.getY()) / 3.0;
    return lsst::geom::Point2D(x, y);
}


lsst::geom::Point3D Triangle::getBarycentricCoordinates(lsst::geom::Point2D const& point) const {
    lsst::geom::AffineTransform const& toBarycentric = getBarycentricTransform();
    lsst::geom::Point2D transformed = toBarycentric(point);
    double const lambda1 = transformed.getX();
    double const lambda2 = transformed.getY();
    double const lambda3 = 1.0 - lambda1 - lambda2;
    return lsst::geom::Point3D(lambda1, lambda2, lambda3);
}


lsst::geom::Point2D Triangle::interpolate(lsst::geom::Point3D const& barycentric) const {
    double const lambda1 = barycentric.getX();
    double const lambda2 = barycentric.getY();
    double const lambda3 = barycentric.getZ();
    double const x = lambda1 * _v1.getX() + lambda2 * _v2.getX() + lambda3 * _v3.getX();
    double const y = lambda1 * _v1.getY() + lambda2 * _v2.getY() + lambda3 * _v3.getY();
    return lsst::geom::Point2D(x, y);
}


lsst::geom::AffineTransform & Triangle::getBarycentricTransform() const {
    if (!_haveBarycentric) {
        double const x1 = _v1.getX();
        double const y1 = _v1.getY();
        double const x2 = _v2.getX();
        double const y2 = _v2.getY();
        double const x3 = _v3.getX();
        double const y3 = _v3.getY();
        double const denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
        double const invDenom = 1.0 / denom;

        double const y2_minus_y3 = y2 - y3;
        double const x3_minus_x2 = x3 - x2;
        double const y3_minus_y1 = y3 - y1;
        double const x1_minus_x3 = x1 - x3;

        lsst::geom::Extent2D translation{
            -y2_minus_y3*x3*invDenom - x3_minus_x2*y3*invDenom,
            -y3_minus_y1*x3*invDenom - x1_minus_x3*y3*invDenom
        };
        lsst::geom::LinearTransform::Matrix matrix;
        matrix << y2_minus_y3*invDenom, x3_minus_x2*invDenom,
                y3_minus_y1*invDenom, x1_minus_x3*invDenom;

        _toBarycentric = lsst::geom::AffineTransform(lsst::geom::LinearTransform(matrix), translation);
        _haveBarycentric = true;
    }
    return _toBarycentric;
}


////////////////////////////////////////////////////////////////////////////////
// GridTree
////////////////////////////////////////////////////////////////////////////////


GridTree::GridTree(
    ArrayRef2D const& x,
    ArrayRef2D const& y,
    int maxPointsPerLeaf
) : _x(x), _y(y), _numCols(x.getShape()[0]), _numRows(x.getShape()[1]), _maxPointsPerLeaf(maxPointsPerLeaf) {
    utils::checkSize(x.getShape(), y.getShape(), "x vs y");
    auto const num = x.getNumElements();

    std::list<lsst::geom::Point2I> points;
    for (std::ptrdiff_t ii = 0; ii < _numCols; ++ii) {
        for (std::ptrdiff_t jj = 0; jj < _numRows; ++jj) {
            points.emplace_back(ii, jj);
        }
    }

    _tree.reserve(2*std::size_t(num/maxPointsPerLeaf + 1));
    Node::build(_tree, points, x, y, maxPointsPerLeaf);
}


lsst::geom::Point2I GridTree::find(double x, double y, double distance) const {
    if (!std::isfinite(x) || !std::isfinite(y)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError, "Cannot find point for non-finite point or distance"
        );
    }
    return getRoot().find(_x, _y, x, y, distance).first;
}


std::pair<
    std::tuple<lsst::geom::Point2I, lsst::geom::Point2I, lsst::geom::Point2I>,
    Triangle
> GridTree::findTriangle(
    double x, double y
) const {
    if (!std::isfinite(x) || !std::isfinite(y)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError, "Cannot find triangle for non-finite point"
        );
    }
    lsst::geom::Point2I const result = find(x, y);
    if (result.getX() < 0 || result.getY() < 0) {
        // No point found
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "No point found in grid");
    }
    lsst::geom::Point2D const start = getValue(result);

    // The grid could be oriented in any direction, so we need to check the neighbours to find
    // the correct triangle. We'll save the options, in case we discover that the point is outside
    // the grid, and we need to find the closest triangle instead.
    int const ii = result.getX();
    int const jj = result.getY();

    std::vector<std::tuple<lsst::geom::Point2I, lsst::geom::Point2I, Triangle>> candidates;
    candidates.reserve(4);
    if (ii + 1 < _numCols) {
        lsst::geom::Point2I right{ii + 1, jj};
        if (jj + 1 < _numRows) {
            lsst::geom::Point2I up{ii, jj + 1};
            Triangle cand{start, getValue(right), getValue(up)};
            if (cand.contains(lsst::geom::Point2D(x, y))) {
                return std::make_pair(std::make_tuple(result, right, up), cand);
            }
            candidates.emplace_back(right, up, cand);
        }
        if (jj - 1 >= 0) {
            lsst::geom::Point2I down{ii, jj - 1};
            Triangle cand{start, getValue(right), getValue(down)};
            if (cand.contains(lsst::geom::Point2D(x, y))) {
                return std::make_pair(std::make_tuple(result, right, down), cand);
            }
            candidates.emplace_back(right, down, cand);
        }
    }
    if (ii - 1 >= 0) {
        lsst::geom::Point2I left{ii - 1, jj};
        if (jj + 1 < _numRows) {
            lsst::geom::Point2I up{ii, jj + 1};
            Triangle cand{start, getValue(left), getValue(up)};
            if (cand.contains(lsst::geom::Point2D(x, y))) {
                return std::make_pair(std::make_tuple(result, left, up), cand);
            }
            candidates.emplace_back(left, up, cand);
        }
        if (jj - 1 >= 0) {
            lsst::geom::Point2I down{ii, jj - 1};
            Triangle cand{start, getValue(left), getValue(down)};
            if (cand.contains(lsst::geom::Point2D(x, y))) {
                return std::make_pair(std::make_tuple(result, left, down), cand);
            }
            candidates.emplace_back(left, down, cand);
        }
    }

    std::pair<lsst::geom::Point2I, lsst::geom::Point2I> vertices;
    Triangle const* triangle = nullptr;
    double distance = std::numeric_limits<double>::infinity();
    for (auto const& cand : candidates) {
        Triangle const& tri = std::get<2>(cand);
        double const dd = (tri.getCentroid() - lsst::geom::Point2D(x, y)).computeSquaredNorm();
        if (dd < distance) {
            distance = dd;
            triangle = &tri;
            vertices = std::make_pair(std::get<0>(cand), std::get<1>(cand));
        }
    }
    if (!std::isfinite(distance) && triangle == nullptr) {
        std::cerr << "No triangle found in grid for point (" << x << ", " << y << ")" << std::endl;
        std::cerr << "Closest point is (" << start.getX() << ", " << start.getY() << ")" << std::endl;
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "No triangle found in grid");
    }
    return std::make_pair(std::make_tuple(result, vertices.first, vertices.second), *triangle);
}


lsst::geom::Point2D GridTree::getValue(lsst::geom::Point2I const& point) const {
    return lsst::geom::Point2D(arrayLookup(_x, point), arrayLookup(_y, point));
}


////////////////////////////////////////////////////////////////////////////////
// GridTree::Node
////////////////////////////////////////////////////////////////////////////////


GridTree::Node::Node(
    bool dividesX,
    double minValue,
    double maxValue,
    int level
) : dividesX(dividesX),
    minValue(minValue),
    maxValue(maxValue),
    divideValue(std::numeric_limits<double>::quiet_NaN()),
    left(),
    right(),
    leaves(),
    level(level)
{}


std::shared_ptr<GridTree::Node> GridTree::Node::build(
    std::vector<std::shared_ptr<Node>> & tree,
    std::list<lsst::geom::Point2I> & points,
    Array2D const& xArray,
    Array2D const& yArray,
    unsigned int maxPointsPerLeaf,
    bool dividesX,
    int level
) {
    double minValue = std::numeric_limits<double>::infinity();
    double maxValue = -std::numeric_limits<double>::infinity();
    for (auto const& pp : points) {
        double value = dividesX ? arrayLookup(xArray, pp) : arrayLookup(yArray, pp);
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }
    std::shared_ptr<Node> node = std::make_shared<Node>(dividesX, minValue, maxValue, level);
    tree.push_back(node);

    if (points.size() <= maxPointsPerLeaf) {
        // Leaf node
        node->leaves.reserve(points.size());
        node->leaves.assign(points.begin(), points.end());
        return node;
    }

    // Find the median point and split the points into two groups
    std::vector<double> coordinates;
    coordinates.reserve(points.size());
    for (auto const& point : points) {
        coordinates.push_back(dividesX ? arrayLookup(xArray, point) : arrayLookup(yArray, point));
    }
    std::nth_element(
        coordinates.begin(), coordinates.begin() + coordinates.size() / 2, coordinates.end()
    );
    double const median = coordinates[coordinates.size() / 2];
    node->divideValue = median;

    // Assign points to left and right groups
    // Using a list here so we can pull points off the list; if we used a vector,
    // we'd make multiple copies as we descend the tree.
    std::list<lsst::geom::Point2I> leftPoints;
    std::list<lsst::geom::Point2I> rightPoints;
    while (points.size() > 0) {
        lsst::geom::Point2I pp = points.front();
        points.pop_front();
        if ((dividesX ? arrayLookup(xArray, pp) : arrayLookup(yArray, pp)) < median) {
            leftPoints.push_back(pp);
        } else {
            rightPoints.push_back(pp);
        }
    }

    node->left = Node::build(tree, leftPoints, xArray, yArray, maxPointsPerLeaf, !dividesX, level + 1);
    node->right = Node::build(tree, rightPoints, xArray, yArray, maxPointsPerLeaf, !dividesX, level + 1);

    return node;
}


std::pair<lsst::geom::Point2I, double> GridTree::Node::find(
    Array2D const& xArray, Array2D const& yArray,
    double x, double y,
    double distance
) const {
    if (leaves.size() > 0) {
        // Leaf node: find the closest point
        lsst::geom::Point2I best;
        double bestDistance2 = std::pow(distance, 2);
        for (auto const& point : leaves) {
            double const dx = arrayLookup(xArray, point) - x;
            double const dy = arrayLookup(yArray, point) - y;
            double const distance2 = dx*dx + dy*dy;
            if (distance2 < bestDistance2) {
                best = point;
                bestDistance2 = distance2;
            }
        }
        return std::make_pair(best, std::sqrt(bestDistance2));
    }

    // Branching node: check children
    double const value = dividesX ? x : y;
    lsst::geom::Point2I best;
    double bestDistance = std::numeric_limits<double>::infinity();
    if (left && value < divideValue + distance) {
        auto const leftResult = left->find(xArray, yArray, x, y, distance);
        if (leftResult.second < distance) {
            best = leftResult.first;
            distance = leftResult.second;
            bestDistance = distance;
        }
    }
    if (right && value > divideValue - distance) {
        auto const rightResult = right->find(xArray, yArray, x, y, distance);
        if (rightResult.second < distance) {
            best = rightResult.first;
            distance = rightResult.second;
            bestDistance = distance;
        }
    }
    return std::make_pair(best, bestDistance);
}


////////////////////////////////////////////////////////////////////////////////
// GridTransform
////////////////////////////////////////////////////////////////////////////////


GridTransform::GridTransform(
    Array2D const& u,
    Array2D const& v,
    Array2D const& x,
    Array2D const& y
) : _u(ndarray::copy(u)),
    _v(ndarray::copy(v)),
    _x(ndarray::copy(x)),
    _y(ndarray::copy(y)),
    _tree(_u.deep(), _v.deep()) {
    utils::checkSize(_v.getShape(), _u.getShape(), "v");
    utils::checkSize(_x.getShape(), _u.getShape(), "x");
    utils::checkSize(_y.getShape(), _u.getShape(), "y");
}


namespace {


/// Apply a single distortion to the x and y arrays in place.
void applyDistortion(
    GridTransform::Array2D & x,
    GridTransform::Array2D & y,
    std::shared_ptr<Distortion> const& distortion
) {
    if (!distortion) {
        return;
    }
    utils::checkSize(x.getShape(), y.getShape(), "x vs y");
    auto const shape = x.getShape();

    for (std::size_t ii = 0; ii < shape[0]; ++ii) {
        for (std::size_t jj = 0; jj < shape[1]; ++jj) {
            auto const dist = (*distortion)(x[ii][jj], y[ii][jj]);
            x[ii][jj] += dist.getX();
            y[ii][jj] += dist.getY();
        }
    }
}


/// Apply a single distortion to the x and y arrays, returning the distorted arrays.
std::pair<GridTransform::Array2D, GridTransform::Array2D> applyDistortion(
    GridTransform::Array2D const& x,
    GridTransform::Array2D const& y,
    std::shared_ptr<Distortion> const& distortion
) {
    if (!distortion) {
        return std::make_pair(x, y);
    }
    utils::checkSize(x.getShape(), y.getShape(), "x vs y");
    GridTransform::Array2D xDist = ndarray::copy(x);
    GridTransform::Array2D yDist = ndarray::copy(y);
    applyDistortion(xDist, yDist, distortion);
    return std::make_pair(xDist, yDist);
}


/// Apply a list of distortions to the x and y arrays, returning the distorted arrays.
std::pair<GridTransform::Array2D, GridTransform::Array2D> applyDistortion(
    GridTransform::Array2D const& x,
    GridTransform::Array2D const& y,
    GridTransform::DistortionList const& distortions
) {
    if (distortions.size() == 0) {
        return std::make_pair(x, y);
    }

    utils::checkSize(x.getShape(), y.getShape(), "x vs y");
    GridTransform::Array2D xDist = ndarray::copy(x);
    GridTransform::Array2D yDist = ndarray::copy(y);

    for (auto const& dd : distortions) {
        applyDistortion(xDist, yDist, dd);
    }
    return std::make_pair(xDist, yDist);
}


}  // anonymous namespace


GridTransform::GridTransform(
    Array2D const& u,
    Array2D const& v,
    std::pair<Array2D, Array2D> const& xy
) : GridTransform(u, v, xy.first, xy.second) {}


GridTransform::GridTransform(
    Array2D const& u,
    Array2D const& v,
    Array2D const& x,
    Array2D const& y,
    std::shared_ptr<Distortion> const& distortion
) : GridTransform(u, v, applyDistortion(x, y, distortion)) {}


GridTransform::GridTransform(
    Array2D const& u,
    Array2D const& v,
    Array2D const& x,
    Array2D const& y,
    DistortionList const& distortions
) : GridTransform(u, v, applyDistortion(x, y, distortions)) {}


lsst::geom::Point2D GridTransform::calculateXY(double u, double v) const {
    auto const interp = getInterpolation(u, v);
    double x = 0;
    double y = 0;
    for (auto const& pair : interp) {
        auto const& point = pair.first;
        auto const& weight = pair.second;
        x += weight*arrayLookup(_x, point);
        y += weight*arrayLookup(_y, point);
    }
    return lsst::geom::Point2D(x, y);
}


ndarray::Array<double, 2, 1> GridTransform::calculateXY(Array1D const& u, Array1D const& v) const {
    utils::checkSize(u.size(), v.size(), "u vs v");
    std::size_t const length = u.size();
    ndarray::Array<double, 2, 1> result = ndarray::allocate(length, 2);
    for (std::size_t ii = 0; ii < length; ++ii) {
        lsst::geom::Point2D const xy = calculateXY(u[ii], v[ii]);
        result[ii][0] = xy.getX();
        result[ii][1] = xy.getY();
    }
    return result;
}


double GridTransform::calculateX(double u, double v) const {
    auto const interp = getInterpolation(u, v);
    double x = 0;
    for (auto const& pair : interp) {
        auto const& point = pair.first;
        auto const& weight = pair.second;
        x += weight*arrayLookup(_x, point);
    }
    return x;
}


ndarray::Array<double, 1, 1> GridTransform::calculateX(Array1D const& u, Array1D const& v) const {
    utils::checkSize(u.size(), v.size(), "u vs v");
    std::size_t const length = u.size();
    ndarray::Array<double, 1, 1> result = ndarray::allocate(length);
    for (std::size_t ii = 0; ii < length; ++ii) {
        result[ii] = calculateX(u[ii], v[ii]);
    }
    return result;
}


double GridTransform::calculateY(double u, double v) const {
    auto const interp = getInterpolation(u, v);
    double y = 0;
    for (auto const& pair : interp) {
        auto const& point = pair.first;
        auto const& weight = pair.second;
        y += weight*arrayLookup(_y, point);
    }
    return y;
}


ndarray::Array<double, 1, 1> GridTransform::calculateY(Array1D const& u, Array1D const& v) const {
    utils::checkSize(u.size(), v.size(), "u vs v");
    std::size_t const length = u.size();
    ndarray::Array<double, 1, 1> result = ndarray::allocate(length);
    for (std::size_t ii = 0; ii < length; ++ii) {
        result[ii] = calculateY(u[ii], v[ii]);
    }
    return result;
}


GridTransform::InterpolationInputs GridTransform::getInterpolation(double u, double v) const {
    auto const result = _tree.findTriangle(u, v);
    lsst::geom::Point2I const& p1 = std::get<0>(result.first);
    lsst::geom::Point2I const& p2 = std::get<1>(result.first);
    lsst::geom::Point2I const& p3 = std::get<2>(result.first);
    Triangle const& triangle = result.second;
    auto const barycentric = triangle.getBarycentricCoordinates(lsst::geom::Point2D(u, v));
    InterpolationInputs inputs;
    inputs.reserve(3);
    inputs.emplace_back(p1, barycentric.getX());
    inputs.emplace_back(p2, barycentric.getY());
    inputs.emplace_back(p3, barycentric.getZ());
    return inputs;
}


////////////////////////////////////////////////////////////////////////////////
// GridDistortion
////////////////////////////////////////////////////////////////////////////////


namespace {


// Singleton class that manages the persistence catalog's schema and keys
class GridDistortionSchema {
    using DoubleArray = lsst::afw::table::Array<double>;
  public:
    lsst::afw::table::Schema schema;
    lsst::afw::table::Key<int> numCols;
    lsst::afw::table::Key<int> numRows;
    lsst::afw::table::Key<DoubleArray> u;
    lsst::afw::table::Key<DoubleArray> v;
    lsst::afw::table::Key<DoubleArray> x;
    lsst::afw::table::Key<DoubleArray> y;

    static GridDistortionSchema const &get() {
        static GridDistortionSchema const instance;
        return instance;
    }

  private:
    GridDistortionSchema()
      : schema(),
        numCols(schema.addField<int>("numCols", "number of columns in grid", "")),
        numRows(schema.addField<int>("numRows", "number of rows in grid", "")),
        u(schema.addField<DoubleArray>("u", "u values", "")),
        v(schema.addField<DoubleArray>("v", "v values", "")),
        x(schema.addField<DoubleArray>("x", "x values", "")),
        y(schema.addField<DoubleArray>("y", "y values", ""))
        {}
};


}  // anonymous namespace


void GridDistortion::write(lsst::afw::table::io::OutputArchiveHandle & handle) const {
    GridDistortionSchema const &schema = GridDistortionSchema::get();
    lsst::afw::table::BaseCatalog cat = handle.makeCatalog(schema.schema);
    std::shared_ptr<lsst::afw::table::BaseRecord> record = cat.addNew();
    record->set(schema.numCols, _transform.getU().getShape()[0]);
    record->set(schema.numRows, _transform.getU().getShape()[1]);
    record->set(schema.u, utils::flattenArray(_transform.getU()));
    record->set(schema.v, utils::flattenArray(_transform.getV()));
    record->set(schema.x, utils::flattenArray(_transform.getX()));
    record->set(schema.y, utils::flattenArray(_transform.getY()));
    handle.saveCatalog(cat);
}


class GridDistortion::Factory : public lsst::afw::table::io::PersistableFactory {
  public:
    std::shared_ptr<lsst::afw::table::io::Persistable> read(
        lsst::afw::table::io::InputArchive const& archive,
        lsst::afw::table::io::CatalogVector const& catalogs
    ) const override {
        static auto const& schema = GridDistortionSchema::get();
        LSST_ARCHIVE_ASSERT(catalogs.front().size() == 1u);
        lsst::afw::table::BaseRecord const& record = catalogs.front().front();
        LSST_ARCHIVE_ASSERT(record.getSchema() == schema.schema);

        int const numCols = record.get(schema.numCols);
        int const numRows = record.get(schema.numRows);
        ndarray::Array<double, 2, 1> u = utils::unflattenArray(record.get(schema.u), numCols, numRows);
        ndarray::Array<double, 2, 1> v = utils::unflattenArray(record.get(schema.v), numCols, numRows);
        ndarray::Array<double, 2, 1> x = utils::unflattenArray(record.get(schema.x), numCols, numRows);
        ndarray::Array<double, 2, 1> y = utils::unflattenArray(record.get(schema.y), numCols, numRows);

        return std::make_shared<GridDistortion>(u, v, x, y);
    }

    Factory(std::string const& name) : lsst::afw::table::io::PersistableFactory(name) {}
};


namespace {

GridDistortion::Factory registration("GridDistortion");

}  // anonymous namespace


}}}  // namespace pfs::drp::stella
