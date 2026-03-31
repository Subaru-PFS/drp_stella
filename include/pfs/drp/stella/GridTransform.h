#ifndef PFS_DRP_STELLA_GRIDTRANSFORM_H
#define PFS_DRP_STELLA_GRIDTRANSFORM_H

#include <tuple>

#include "ndarray_fwd.h"

#include "lsst/geom/Point.h"
#include "lsst/geom/AffineTransform.h"
#include "lsst/cpputils/Cache.h"
#include "lsst/cpputils/hashCombine.h"

#include "pfs/drp/stella/Distortion.h"

namespace pfs {
namespace drp {
namespace stella {


/// A triangle in 2D space, defined by three vertices
///
/// This is used for interpolation. The barycentric coordinates of a point with
/// respect to the triangle in the original space serves as the interpolation
/// weights for the corresponding triangle in the target space.
class Triangle {
  public:
    /// Ctor
    ///
    /// @param v1 First vertex of the triangle
    /// @param v2 Second vertex of the triangle
    /// @param v3 Third vertex of the triangle
    Triangle(
        lsst::geom::Point2D const& v1, lsst::geom::Point2D const& v2, lsst::geom::Point2D const& v3
    ) : _v1(v1), _v2(v2), _v3(v3), _haveBarycentric(false), _toBarycentric() {}

    /// Default constructor creates an empty triangle at the origin
    explicit Triangle() : Triangle(lsst::geom::Point2D(), lsst::geom::Point2D(), lsst::geom::Point2D()) {}

    Triangle(Triangle const&) = default;
    Triangle(Triangle &&) = default;
    Triangle & operator=(Triangle const&) = default;
    Triangle & operator=(Triangle &&) = default;
    ~Triangle() = default;

    //@{
    /// Access the vertices of the triangle
    lsst::geom::Point2D const& getVertex1() const { return _v1; }
    lsst::geom::Point2D const& getVertex2() const { return _v2; }
    lsst::geom::Point2D const& getVertex3() const { return _v3; }
    std::tuple<lsst::geom::Point2D, lsst::geom::Point2D, lsst::geom::Point2D> getVertices() const {
        return std::make_tuple(_v1, _v2, _v3);
    }
    //@}

    /// Is the triangle empty?
    bool isEmpty() const;

    /// Calculate the area of the triangle
    double getArea() const;

    //@{
    /// Triangle contains the point, or it's on the edge
    bool contains(double x, double y) const {
        return contains(lsst::geom::Point2D(x, y));
    }
    bool contains(lsst::geom::Point2D const& point) const {
        return contains(getBarycentricCoordinates(point));
    }
    bool contains(lsst::geom::Point3D const& barycentric) const {
        return barycentric.getX() >= 0 && barycentric.getY() >= 0 && barycentric.getZ() >= 0;
    }
    //@}

    /// Get the centroid of the triangle
    lsst::geom::Point2D getCentroid() const;

    /// Get the barycentric coordinates of a point with respect to the triangle
    lsst::geom::Point3D getBarycentricCoordinates(lsst::geom::Point2D const& point) const;

    /// Interpolate a point given its barycentric coordinates
    ///
    /// The barycentric coordinates can be calculated for a triangle in one
    /// space, and then applied to the corresponding triangle in another space
    /// to find the interpolated point.
    lsst::geom::Point2D interpolate(lsst::geom::Point3D const& barycentric) const;

  private:
    /// Get the affine transform that maps the triangle to barycentric coordinates.
    lsst::geom::AffineTransform & getBarycentricTransform() const;

    lsst::geom::Point2D _v1;  ///< First vertex of the triangle
    lsst::geom::Point2D _v2;  ///< Second vertex of the triangle
    lsst::geom::Point2D _v3;  ///< Third vertex of the triangle

    mutable bool _haveBarycentric;  ///< Has the barycentric transform been calculated?
    mutable lsst::geom::AffineTransform _toBarycentric;  ///< Transform to barycentric coordinates
};


/// Default number of points in a leaf node of the tree; can be overridden by the caller
/// This has been roughly tuned to give good performance for the optical models we use for PFS.
const int GRID_TREE_MAX_POINTS_PER_LEAF = 32;


/// A spatial index for a grid of points
///
/// This is used to find the closest point to a given coordinate, for use in
/// interpolation. It is implemented as a binary tree, and provides methods to
/// find the closest point, and to find the enclosing triangle for a given
/// coordinate.
class GridTree {
  public:
    using Array2D = ndarray::Array<double, 2, 1>;
    using ArrayRef2D = ndarray::ArrayRef<double, 2, 1>;
    using Shape = ndarray::Vector<int, 2>;

    /// Ctor
    ///
    /// @param x 2D array of x coordinates of the grid points
    /// @param y 2D array of y coordinates of the grid points
    /// @param maxPointsPerLeaf Maximum number of points in a leaf node of the tree
    GridTree(
        ArrayRef2D const& x,
        ArrayRef2D const& y,
        int maxPointsPerLeaf=GRID_TREE_MAX_POINTS_PER_LEAF
    );

    GridTree(GridTree const&) = default;
    GridTree(GridTree &&) = default;
    GridTree & operator=(GridTree const&) = default;
    GridTree & operator=(GridTree &&) = default;
    ~GridTree() = default;

    //@{
    /// Find the closest point to the given coordinates
    ///
    /// The distance parameter can be used to limit the search to points within
    /// a certain distance of the given coordinates.
    ///
    /// Returns {-1,-1} if no point is found within the given distance.
    lsst::geom::Point2I find(
        double x, double y,
        double distance=std::numeric_limits<double>::infinity()
    ) const;
    lsst::geom::Point2I find(
        lsst::geom::Point2D const& point,
        double distance=std::numeric_limits<double>::infinity()
    ) const {
        return find(point.getX(), point.getY(), distance*distance);
    }
    //@}

    //@{
    /// Get the grid value at the given index
    ///
    /// To increase performance, no bounds checking is performed here; it is the
    /// caller's responsibility to ensure that the index is valid.
    lsst::geom::Point2D getValue(lsst::geom::Point2I const& point) const;
    lsst::geom::Point2D getValue(int ii, int jj) const {
        return getValue(lsst::geom::Point2I(ii, jj));
    }
    //@}

    using FindTriangleResult = std::pair<
        std::tuple<lsst::geom::Point2I, lsst::geom::Point2I, lsst::geom::Point2I>,
        Triangle
    >;

    //@{
    /// Find the triangle that contains the given coordinates
    ///
    /// We find the closest point to the given coordinates, and then check the
    /// four triangles that have that point as a vertex.
    ///
    /// If the coordinates are outside the grid, then we return the closest
    /// triangle.
    FindTriangleResult findTriangle(
        double x, double y
    ) const;
    FindTriangleResult findTriangle(
        lsst::geom::Point2D const& point
    ) const {
        return findTriangle(point.getX(), point.getY());
    }
    //@}

    //@{
    /// Accessors
    ArrayRef2D const& getX() const { return _x; }
    ArrayRef2D const& getY() const { return _y; }
    //@}

  private:

    /// A node in the tree
    struct Node {
        double xMin, xMax;  ///< node bounds in x
        double yMin, yMax;  ///< node bounds in y
        lsst::geom::Box2I box;  ///< indices of points in this node
        std::shared_ptr<Node> left;  ///< left child node, or nullptr if this is a leaf node
        std::shared_ptr<Node> right;  ///< right child node, or nullptr if this is a leaf node
        ndarray::Array<double, 2, 1> xValues;  ///< x values of the points in this node, if it's a leaf node
        ndarray::Array<double, 2, 1> yValues;  ///< y values of the points in this node, if it's a leaf node
        int level;  ///< level of the node in the tree, for debugging purposes

        /// Ctor
        Node(
            double xMin, double xMax, double yMin, double yMax,
            lsst::geom::Box2I const& box,
            int level
        );

        /// Get the minimum distance-squared from the node to the given coordinates
        double getDistance2(double x, double y) const;

        /// Build a node in the tree recursively
        ///
        /// @param box Indices of points in this node
        /// @param xArray 2D array of x coordinates of the grid points
        /// @param yArray 2D array of y coordinates of the grid points
        /// @param maxPointsPerLeaf Maximum number of points in a leaf node of the tree
        /// @param level Level of the node in the tree, for debugging purposes
        /// @return The newly created node
        static std::shared_ptr<Node> build(
            lsst::geom::Box2I const& box,
            Array2D const& xArray,
            Array2D const& yArray,
            unsigned int maxPointsPerLeaf,
            int level=0
        );

        /// Find the closest point to the given coordinates in this leaf node
        ///
        /// For a leaf node, scans all points. Has no effect for a branch node.
        ///
        /// @param x x coordinate of the point to find
        /// @param y y coordinate of the point to find
        /// @param distance2 Maximum distance-squared to search for points
        /// @return The closest point and its distance from the given coordinates.
        ///       Returns {-1,-1} if no point is found within the given distance.
        std::pair<lsst::geom::Point2I, double> find(
            double x, double y,
            double distance2=std::numeric_limits<double>::infinity()
        ) const;
    };

    ArrayRef2D _x;  ///< x coordinates of the grid points; shape is (numCols, numRows)
    ArrayRef2D _y;  ///< y coordinates of the grid points; shape is (numCols, numRows)
    int _numCols, _numRows;  ///< number of columns and rows in the grid
    unsigned int _maxPointsPerLeaf;  ///< maximum number of points in a leaf node of the tree
    std::shared_ptr<Node> _root;  ///< root node of the tree

    using TrianglePtr = std::shared_ptr<Triangle>;
    using TriangleKey = std::tuple<lsst::geom::Point2I, lsst::geom::Point2I, lsst::geom::Point2I>;
    /// Hash operation on Triangle, for caching
    struct TriangleHash {
        std::size_t operator()(TriangleKey const& key) const {
            std::size_t seed = 0;
            return lsst::cpputils::hashCombine(
                seed,
                std::get<0>(key).getX(), std::get<0>(key).getY(),
                std::get<1>(key).getX(), std::get<1>(key).getY(),
                std::get<2>(key).getX(), std::get<2>(key).getY()
            );
        }
    };

    //@{
    /// Get the triangle corresponding to the given key, calculating it if necessary
    Triangle const& getTriangle(TriangleKey const& key) const;
    Triangle const& getTriangle(
        lsst::geom::Point2I const& p1, lsst::geom::Point2I const& p2, lsst::geom::Point2I const& p3
    ) const {
        return getTriangle(std::make_tuple(p1, p2, p3));
    }
    //@}

    /// Cache of triangles for interpolation; maps from the indices of the triangle vertices to the triangle
    mutable lsst::cpputils::Cache<TriangleKey, TrianglePtr, TriangleHash> _triangles;
    mutable lsst::geom::Point2I _lastFound;  ///< Cache the last-found point to speed up nearby searches
};


/// A grid of values that serve as a transformation, u,v --> x,y
///
/// Given a grid of points in u,v space and the corresponding points in x,y
/// space, we can interpolate to find the x,y coordinates corresponding to any
/// u,v coordinates.
class GridTransform {
  public:
    using Array1D = ndarray::Array<double, 1, 1>;
    using Array2D = ndarray::Array<double, 2, 1>;
    using InterpolationInputs = std::vector<std::pair<lsst::geom::Point2I, double>>;  // index and weight
    using DistortionList = std::vector<std::shared_ptr<Distortion>>;

    //@{
    /// Ctor
    GridTransform(
        Array2D const& u,
        Array2D const& v,
        Array2D const& x,
        Array2D const& y
    );
    GridTransform(
        Array2D const& u,
        Array2D const& v,
        Array2D const& x,
        Array2D const& y,
        DistortionList const& distortionList
    );
    GridTransform(
        Array2D const& u,
        Array2D const& v,
        Array2D const& x,
        Array2D const& y,
        std::shared_ptr<Distortion> const& distortion
    );
    //@}

    GridTransform(GridTransform const&) = default;
    GridTransform(GridTransform &&) = default;
    GridTransform & operator=(GridTransform const&) = default;
    GridTransform & operator=(GridTransform &&) = default;
    ~GridTransform() = default;

    //@{
    /// Accessors
    Array2D const& getU() const { return _u; }
    Array2D const& getV() const { return _v; }
    Array2D const& getX() const { return _x; }
    Array2D const& getY() const { return _y; }
    GridTree const& getTree() const { return _tree; }
    //@}

    /// Get the inverse of this transform, which maps x,y --> u,v
    GridTransform inverse() const {
        return GridTransform(_x, _y, _u, _v);
    }

    //@{
    /// Interpolate to find the x,y coordinates corresponding to the given u,v coordinates
    lsst::geom::Point2D calculateXY(double u, double v) const;
    ndarray::Array<double, 2, 1> calculateXY(Array1D const& u, Array1D const& v) const;
    lsst::geom::Point2D calculateXY(lsst::geom::Point2D const& uv) const {
        return calculateXY(uv.getX(), uv.getY());
    }
    lsst::geom::Point2D operator()(double u, double v) const { return calculateXY(u, v); }
    ndarray::Array<double, 2, 1> operator()(Array1D const& u, Array1D const& v) const {
        return calculateXY(u, v);
    }
    lsst::geom::Point2D operator()(lsst::geom::Point2D const& uv) const { return calculateXY(uv); }
    //@}

    //@{
    /// Interpolate to find the x coordinate corresponding to the given u,v coordinates
    double calculateX(double u, double v) const;
    ndarray::Array<double, 1, 1> calculateX(Array1D const& u, Array1D const& v) const;
    double calculateX(lsst::geom::Point2D const& uv) const {
        return calculateX(uv.getX(), uv.getY());
    }
    //@}

    //@{
    /// Interpolate to find the y coordinate corresponding to the given u,v coordinates
    double calculateY(double u, double v) const;
    ndarray::Array<double, 1, 1> calculateY(Array1D const& u, Array1D const& v) const;
    double calculateY(lsst::geom::Point2D const& uv) const {
        return calculateY(uv.getX(), uv.getY());
    }
    //@}

    /// Return the inputs needed to perform interpolation for the given u,v coordinates
    ///
    /// @return the indices of the grid points to use for interpolation, and their corresponding weights
    InterpolationInputs getInterpolation(double u, double v) const;

    //@{
    /// Calculate the relative areas of the triangles used for interpolation
    ///
    /// The relative area is the area of x,y triangle divided by the area of the
    /// corresponding u,v triangle.
    double calculateRelativeArea(double u, double v) const;
    double calculateRelativeArea(lsst::geom::Point2D const& uv) const {
        return calculateRelativeArea(uv.getX(), uv.getY());
    }
    //@}

  private:

    /// Convenience constructor for distorted grids
    GridTransform(Array2D const& u, Array2D const& v, std::pair<Array2D, Array2D> const& xy);

    //@{
    /// Grid of values for interpolation
    Array2D _u;
    Array2D _v;
    Array2D _x;
    Array2D _y;
    //@}

    GridTree _tree;  ///< tree of u,v values
};


/// Distortion interface implemented by a GridTransform
class GridDistortion : public Distortion {
  public:
    using Array2D = ndarray::Array<double, 2, 1>;

    //@{
    /// Ctor
    GridDistortion(GridTransform const& transform) : _transform(transform) {}
    GridDistortion(
        Array2D const& u,
        Array2D const& v,
        Array2D const& x,
        Array2D const& y
    ) : _transform(u, v, x, y) {}
    //@}

    GridDistortion(GridDistortion const&) = default;
    GridDistortion(GridDistortion &&) = default;
    GridDistortion & operator=(GridDistortion const&) = default;
    GridDistortion & operator=(GridDistortion &&) = default;
    virtual ~GridDistortion() = default;

    virtual std::shared_ptr<Distortion> clone() const override {
        return std::make_shared<GridDistortion>(
            ndarray::copy(_transform.getU()), ndarray::copy(_transform.getV()),
            ndarray::copy(_transform.getX()), ndarray::copy(_transform.getY())
        );
    }

    virtual std::size_t getNumParameters() const override { return 4 * _transform.getU().getNumElements(); }

    virtual lsst::geom::Point2D evaluate(lsst::geom::Point2D const& xy) const override {
        return _transform.calculateXY(xy);
    }

    bool isPersistable() const noexcept { return true; }

    class Factory;

  protected:
    std::string getPersistenceName() const { return "GridDistortion"; }
    std::string getPythonModule() const { return "pfs.drp.stella"; }
    void write(lsst::afw::table::io::OutputArchiveHandle & handle) const;

  private:
    GridTransform _transform;
};


}}}  // namespace pfs::drp::stella

#endif  // include guard
