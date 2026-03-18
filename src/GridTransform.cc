#include <queue>

#include "ndarray.h"

#include "lsst/afw/table.h"
#include "lsst/afw/table/io/OutputArchive.h"
#include "lsst/afw/table/io/InputArchive.h"
#include "lsst/afw/table/io/CatalogVector.h"
#include "lsst/afw/table/io/Persistable.cc"

#include "pfs/drp/stella/math/quartiles.h"
#include "pfs/drp/stella/utils/checkSize.h"
#include "pfs/drp/stella/utils/math.h"
#include "pfs/drp/stella/GridTransform.h"

namespace pfs {
namespace drp {
namespace stella {


namespace {


double arrayLookup(ndarray::Array<double, 2, 1> const& array, lsst::geom::Point2I const& point) {
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
) : _x(x), _y(y),
    _numCols(x.getShape()[0]), _numRows(x.getShape()[1]),
    _maxPointsPerLeaf(maxPointsPerLeaf),
    _lastFound(-1, -1)
{
    utils::checkSize(x.getShape(), y.getShape(), "x vs y");
    lsst::geom::Box2I box{lsst::geom::Point2I(0, 0), lsst::geom::Extent2I(_numCols, _numRows)};
    _root = Node::build(box, x, y, maxPointsPerLeaf);
}


lsst::geom::Point2I GridTree::find(double x, double y, double distance) const {
    if (!std::isfinite(x) || !std::isfinite(y)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError, "Cannot find point for non-finite point or distance"
        );
    }
    double distance2 = distance*distance;
    lsst::geom::Point2I closest{-1, -1};

    // Requests for nearby points are common, so we check the last-found point first,
    // which can greatly reduce the search radius.
    if (_lastFound.getX() >= 0 && _lastFound.getY() >= 0) {
        double const dx = arrayLookup(_x, _lastFound) - x;
        double const dy = arrayLookup(_y, _lastFound) - y;
        double const lastDistance2 = dx*dx + dy*dy;
        if (lastDistance2 < distance2) {
            distance2 = lastDistance2;
            closest = _lastFound;
        }
    }

    using QueueElement = std::pair<std::shared_ptr<Node>, double>;  // point and distance-squared
    auto compare = [](QueueElement const& a, QueueElement const& b) { return a.second > b.second; };
    std::priority_queue<QueueElement, std::vector<QueueElement>, decltype(compare)> queue(compare);
    queue.emplace(_root, 0.0);

    while (!queue.empty()) {
        auto const element = queue.top();
        queue.pop();
        Node const& node = *element.first;
        double const nodeDistance2 = element.second;
        if (nodeDistance2 > distance2) {
            // All remaining nodes are too far away
            break;
        }
        auto const pointDistance = node.find(x, y, distance2);
        if (pointDistance.second < distance2) {
            closest = pointDistance.first;
            distance2 = pointDistance.second;
        }
        if (node.left) {
            queue.emplace(node.left, node.left->getDistance2(x, y));
        }
        if (node.right) {
            queue.emplace(node.right, node.right->getDistance2(x, y));
        }
    }

    if (closest.getX() >= 0 && closest.getY() >= 0) {
        _lastFound = closest;
    }
    return closest;
}


GridTree::FindTriangleResult GridTree::findTriangle(
    double x, double y
) const {
    if (!std::isfinite(x) || !std::isfinite(y)) {
        throw LSST_EXCEPT(
            lsst::pex::exceptions::InvalidParameterError, "Cannot find triangle for non-finite point"
        );
    }
    lsst::geom::Point2I const closest = find(x, y);
    if (closest.getX() < 0 || closest.getY() < 0) {
        // No point found
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "No point found in grid");
    }

    // The grid could be oriented in any direction, so we need to check the neighbours to find
    // the correct triangle. We'll save the options, in case we discover that the point is outside
    // the grid, and we need to find the closest triangle instead.
    int const ii = closest.getX();
    int const jj = closest.getY();

    std::vector<std::tuple<lsst::geom::Point2I, lsst::geom::Point2I, Triangle>> candidates;
    candidates.reserve(4);
    if (ii + 1 < _numCols) {
        lsst::geom::Point2I right{ii + 1, jj};
        if (jj + 1 < _numRows) {
            lsst::geom::Point2I up{ii, jj + 1};
            Triangle const& cand = getTriangle(closest, right, up);
            if (cand.contains(lsst::geom::Point2D(x, y))) {
                return std::make_pair(std::make_tuple(closest, right, up), cand);
            }
            candidates.emplace_back(right, up, cand);
        }
        if (jj - 1 >= 0) {
            lsst::geom::Point2I down{ii, jj - 1};
            Triangle const& cand = getTriangle(closest, right, down);
            if (cand.contains(lsst::geom::Point2D(x, y))) {
                return std::make_pair(std::make_tuple(closest, right, down), cand);
            }
            candidates.emplace_back(right, down, cand);
        }
    }
    if (ii - 1 >= 0) {
        lsst::geom::Point2I left{ii - 1, jj};
        if (jj + 1 < _numRows) {
            lsst::geom::Point2I up{ii, jj + 1};
            Triangle const& cand = getTriangle(closest, left, up);
            if (cand.contains(lsst::geom::Point2D(x, y))) {
                return std::make_pair(std::make_tuple(closest, left, up), cand);
            }
            candidates.emplace_back(left, up, cand);
        }
        if (jj - 1 >= 0) {
            lsst::geom::Point2I down{ii, jj - 1};
            Triangle const& cand = getTriangle(closest, left, down);
            if (cand.contains(lsst::geom::Point2D(x, y))) {
                return std::make_pair(std::make_tuple(closest, left, down), cand);
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
        throw LSST_EXCEPT(lsst::pex::exceptions::RuntimeError, "No triangle found in grid");
    }
    return std::make_pair(std::make_tuple(closest, vertices.first, vertices.second), *triangle);
}


lsst::geom::Point2D GridTree::getValue(lsst::geom::Point2I const& point) const {
    return lsst::geom::Point2D(arrayLookup(_x, point), arrayLookup(_y, point));
}


Triangle const& GridTree::getTriangle(TriangleKey const& key) const {
    return *_triangles(key, [this](TriangleKey const& key ) {
        return std::make_shared<Triangle>(
            getValue(std::get<0>(key)), getValue(std::get<1>(key)), getValue(std::get<2>(key))
        );
    });
}


////////////////////////////////////////////////////////////////////////////////
// GridTree::Node
////////////////////////////////////////////////////////////////////////////////


GridTree::Node::Node(
    double xMin, double xMax, double yMin, double yMax, lsst::geom::Box2I const& box, int level
) : xMin(xMin),
    xMax(xMax),
    yMin(yMin),
    yMax(yMax),
    box(box),
    left(),
    right(),
    xValues(),
    yValues(),
    level(level)
{}


double GridTree::Node::getDistance2(double x, double y) const {
    double dx = 0.0;
    if (x < xMin) {
        dx = xMin - x;
    } else if (x > xMax) {
        dx = x - xMax;
    }
    double dy = 0.0;
    if (y < yMin) {
        dy = yMin - y;
    } else if (y > yMax) {
        dy = y - yMax;
    }
    return dx*dx + dy*dy;
}


std::shared_ptr<GridTree::Node> GridTree::Node::build(
    lsst::geom::Box2I const& box,
    Array2D const& xArray,
    Array2D const& yArray,
    unsigned int maxPointsPerLeaf,
    int level
) {
    double const infinity = std::numeric_limits<double>::infinity();
    double xMin = infinity;
    double xMax = -infinity;
    double yMin = infinity;
    double yMax = -infinity;
    for (int ii = box.getMinX(); ii <= box.getMaxX(); ++ii) {
        for (int jj = box.getMinY(); jj <= box.getMaxY(); ++jj) {
            double const x = xArray[ii][jj];
            double const y = yArray[ii][jj];
            xMin = std::min(xMin, x);
            xMax = std::max(xMax, x);
            yMin = std::min(yMin, y);
            yMax = std::max(yMax, y);
        }
    }

    std::shared_ptr<Node> node = std::make_shared<Node>(xMin, xMax, yMin, yMax, box, level);

    int const width = box.getWidth();
    int const height = box.getHeight();
    unsigned int const area = box.getArea();
    if (area <= maxPointsPerLeaf) {
        // Leaf node
        node->box = box;
        node->xValues = ndarray::allocate(width, height);
        node->yValues = ndarray::allocate(width, height);
        for (int ii = box.getMinX(); ii <= box.getMaxX(); ++ii) {
            for (int jj = box.getMinY(); jj <= box.getMaxY(); ++jj) {
                lsst::geom::Point2I const point{ii, jj};
                node->xValues[ii - box.getMinX()][jj - box.getMinY()] = arrayLookup(xArray, point);
                node->yValues[ii - box.getMinX()][jj - box.getMinY()] = arrayLookup(yArray, point);
            }
        }
        return node;
    }

    // Branching node
    lsst::geom::Box2I leftBox, rightBox;
    if (width >= height) {
        int const mid = box.getCenterX();
        leftBox = lsst::geom::Box2I(box.getMin(), lsst::geom::Point2I(mid, box.getMaxY()));
        rightBox = lsst::geom::Box2I(lsst::geom::Point2I(mid + 1, box.getMinY()), box.getMax());
    } else {
        int const mid = box.getCenterY();
        leftBox = lsst::geom::Box2I(box.getMin(), lsst::geom::Point2I(box.getMaxX(), mid));
        rightBox = lsst::geom::Box2I(lsst::geom::Point2I(box.getMinX(), mid + 1), box.getMax());
    }

    node->left = Node::build(leftBox, xArray, yArray, maxPointsPerLeaf, level + 1);
    node->right = Node::build(rightBox, xArray, yArray, maxPointsPerLeaf, level + 1);

    return node;
}


std::pair<lsst::geom::Point2I, double> GridTree::Node::find(
    double x, double y,
    double distance2
) const {
    if (xValues.empty()) {
        return {lsst::geom::Point2I(-1, -1), std::numeric_limits<double>::infinity()};
    }
    lsst::geom::Point2I best;
    for (int ii = 0; ii < box.getWidth(); ++ii) {
        for (int jj = 0; jj < box.getHeight(); ++jj) {
            double const dx = xValues[ii][jj] - x;
            double const dx2 = dx*dx;
            if (dx2 > distance2) {
                continue;
            }
            double const dy = yValues[ii][jj] - y;
            double const dy2 = dy*dy;
            double const r2 = dx2 + dy2;
            if (r2 > distance2) {
                continue;
            }
            best = lsst::geom::Point2I(box.getMinX() + ii, box.getMinY() + jj);
            distance2 = r2;
        }
    }

    return {best, distance2};
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
