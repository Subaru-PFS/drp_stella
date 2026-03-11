import numpy as np

import lsst.utils.tests
from lsst.geom import Point2D, Point2I, Point3D

from pfs.drp.stella.GridTransform import Triangle, GridTree, GridTransform
from pfs.drp.stella.tests import runTests

display = None


class TriangleTestCase(lsst.utils.tests.TestCase):
    """Tests for Triangle"""
    def setUp(self):
        self.p1 = Point2D(0, 0)
        self.p2 = Point2D(1, 0)
        self.p3 = Point2D(0, 1)

    def testBasic(self):
        """Test basic operations on a Triangle"""
        triangle = Triangle(self.p1, self.p2, self.p3)
        self.assertFalse(triangle.isEmpty())
        self.assertFloatsAlmostEqual(triangle.getArea(), 0.5)

        # Accessors
        self.assertEqual(triangle.getVertex1(), self.p1)
        self.assertEqual(triangle.getVertex2(), self.p2)
        self.assertEqual(triangle.getVertex3(), self.p3)
        vertices = triangle.getVertices()
        self.assertEqual(len(vertices), 3)
        self.assertEqual(vertices[0], self.p1)
        self.assertEqual(vertices[1], self.p2)
        self.assertEqual(vertices[2], self.p3)

        # contains, getBarycentricCoordinates
        point = Point2D(0.25, 0.25)
        self.assertTrue(triangle.contains(point))
        self.assertEqual(triangle.getBarycentricCoordinates(point), Point3D(0.5, 0.25, 0.25))

        point = Point2D(1, 1)
        self.assertFalse(triangle.contains(point))
        self.assertEqual(triangle.getBarycentricCoordinates(point), Point3D(-1, 1, 1))

        # getCentroid
        self.assertEqual(triangle.getCentroid(), Point2D(1/3, 1/3))

    def testEmpty(self):
        """Test the behavior of an empty triangle"""
        triangle = Triangle()
        self.assertTrue(triangle.isEmpty())
        self.assertFloatsAlmostEqual(triangle.getArea(), 0.0)
        self.assertFalse(triangle.contains(Point2D(0, 0)))
        barycentric = triangle.getBarycentricCoordinates(Point2D(0, 0))
        self.assertTrue(np.all(np.isnan(barycentric)))


class GridTreeTestCase(lsst.utils.tests.TestCase):
    """Tests for GridTree"""
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def makeData(
        self,
        shape: tuple[int, int] = (10, 10),
        xScale: float = 10.0,
        yScale: float = 3.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create the data for the test

        We create a regular grid of points, with the specified shape and scale.

        We set the ``shape`` attribute to the shape of the grid, and the ``i``
        and ``j`` attributes to the indices of the grid points. We also set the
        ``xScale`` and ``yScale`` attributes to the specified scales.

        Parameters
        ----------
        shape : `tuple` of `int`
            The shape of the grid to create.
        xScale, yScale : `float`
            The scale of the grid in the x and y directions.

        Returns
        -------
        x, y : `numpy.ndarray`
            The x and y coordinates of the grid points.
        """
        self.shape = shape
        self.i, self.j = np.indices(shape)
        self.xScale = xScale
        self.yScale = yScale
        xx = (xScale*self.i).astype(float)
        yy = (yScale*self.j).astype(float)
        return xx, yy

    def testBasic(self):
        """Test basic operations on a GridTree"""
        xx, yy = self.makeData()

        xMax = self.xScale*(self.shape[0] - 1)
        yMax = self.yScale*(self.shape[1] - 1)

        tree = GridTree(xx, yy)

        self.assertEqual(tree.getValue(Point2I(0, 0)), Point2D(0, 0))
        self.assertEqual(tree.getValue(Point2I(5, 5)), Point2D(50, 15))
        self.assertEqual(tree.getValue(Point2I(7, 3)), Point2D(70, 9))

        for _ in range(123):
            # Find the closest point to a random point inside the grid
            point = Point2D(self.rng.uniform(0, xMax), self.rng.uniform(0, yMax))
            index = tree.find(point)
            expect = np.argmin(np.hypot(xx - point.getX(), yy - point.getY()))
            if False:
                expectIndex = Point2I(self.i.flat[expect], self.j.flat[expect])
                print(
                    point,
                    tree.getValue(index),
                    tree.getValue(expectIndex),
                    np.hypot(
                        tree.getValue(index).getX() - point.getX(), tree.getValue(index).getY() - point.getY()
                    ),
                    np.hypot(
                        tree.getValue(expectIndex).getX() - point.getX(), tree.getValue(expectIndex).getY() - point.getY()
                    ),
                )
            self.assertEqual(index, Point2I(self.i.flat[expect], self.j.flat[expect]))
            self.assertEqual(
                tree.getValue(index),
                Point2D(self.i.flat[expect]*self.xScale, self.j.flat[expect]*self.yScale)
            )

            # Find the triangle containing the point
            vertices, triangle = tree.findTriangle(point)
            self.assertEqual(len(vertices), 3)
            for vv, pp in zip(vertices, triangle.getVertices()):
                self.assertEqual(tree.getValue(vv), pp)
            self.assertTrue(triangle.contains(point))
            barycentric = triangle.getBarycentricCoordinates(point)
            interp = triangle.interpolate(barycentric)
            self.assertFloatsAlmostEqual(interp.getX(), point.getX(), atol=1.0e-10)
            self.assertFloatsAlmostEqual(interp.getY(), point.getY(), atol=1.0e-10)

    def testFindOutside(self):
        """Find the closest point to a point outside the grid"""
        xx, yy = self.makeData()

        xMax = self.xScale*(self.shape[0] - 1)
        yMax = self.yScale*(self.shape[1] - 1)

        tree = GridTree(xx, yy)

        for _ in range(23):
            point = Point2D(self.rng.uniform(-xMax, 0), self.rng.uniform(yMax, 2*yMax))  # Outside the grid
            index = tree.find(point)
            expect = np.argmin(np.hypot(xx - point.getX(), yy - point.getY()))
            self.assertEqual(index, Point2I(self.i.flat[expect], self.j.flat[expect]))
            self.assertEqual(
                tree.getValue(index),
                Point2D(self.i.flat[expect]*self.xScale, self.j.flat[expect]*self.yScale)
            )


class GridTransformTestCase(lsst.utils.tests.TestCase):
    """Tests for GridTransform"""
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def makeData(
        self,
        shape: tuple[int, int] = (10, 10),
        uScale: float = 10.0,
        vScale: float = 3.0,
        rotation: float = 67.8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create the data for the test

        We create a regular grid of points, with the specified shape and scale,
        and rotate it by the specified angle.

        We set the ``shape`` attribute to the shape of the grid, and the ``i``
        and ``j`` attributes to the indices of the grid points. We also set the
        ``uScale`` and ``vScale`` attributes to the specified scales, and the
        ``theta`` attribute to the specified rotation.

        Parameters
        ----------
        shape : `tuple` of `int`
            The shape of the grid to create.
        uScale, vScale : `float`
            The scale of the grid in the u and v directions.
        rotation : `float`
            The rotation of the grid in degrees.

        Returns
        -------
        u, v : `numpy.ndarray`
            The u and v coordinates of the grid points.
        x, y : `numpy.ndarray`
            The x and y coordinates of the grid points, after rotation.
        """
        self.shape = shape
        self.i, self.j = np.indices(shape)
        self.uScale = uScale
        self.vScale = vScale
        self.theta = np.radians(rotation)

        uu = (uScale*self.i).astype(float)
        vv = (vScale*self.j).astype(float)

        # Rotate the u,v grid by the specified angle
        xx, yy = self.calculateXY(uu, vv)

        return uu, vv, xx, yy

    def calculateXY(self, u, v):
        """Calculate the x,y coordinates corresponding to the given u,v coordinates

        Parameters
        ----------
        u, v : array-like
            The u and v coordinates to transform.

        Returns
        -------
        x, y : as for ``u`` and ``v``
            The x and y coordinates corresponding to the given u and v
            coordinates.
        """
        xx = u*np.cos(self.theta) - v*np.sin(self.theta)
        yy = u*np.sin(self.theta) + v*np.cos(self.theta)
        return xx, yy

    def testBasic(self):
        """Test basic operations on a GridTransform"""
        uu, vv, xx, yy = self.makeData()
        transform = GridTransform(uu, vv, xx, yy)

        # Accessors
        self.assertFloatsEqual(transform.getU(), uu)
        self.assertFloatsEqual(transform.getV(), vv)
        self.assertFloatsEqual(transform.getX(), xx)
        self.assertFloatsEqual(transform.getY(), yy)

        # Inverse: arrays should be swapped
        inverse = transform.inverse()
        self.assertFloatsEqual(inverse.getU(), xx)
        self.assertFloatsEqual(inverse.getV(), yy)
        self.assertFloatsEqual(inverse.getX(), uu)
        self.assertFloatsEqual(inverse.getY(), vv)

        # Basic interpolation: a random point of the grid should map to the corresponding point
        for _ in range(23):
            selection = self.rng.uniform(0, self.shape[0]), self.rng.uniform(0, self.shape[1])
            index = int(selection[0]), int(selection[1])
            uv = Point2D(uu[index], vv[index])
            expect = Point2D(xx[index], yy[index])

            interp = transform.calculateXY(uv)
            self.assertFloatsAlmostEqual(interp.getX(), expect.getX(), atol=1.0e-10)
            self.assertFloatsAlmostEqual(interp.getY(), expect.getY(), atol=1.0e-10)

        # Basic interpolation: a random point inside the grid
        uMin, uMax = np.min(uu), np.max(uu)
        vMin, vMax = np.min(vv), np.max(vv)
        for _ in range(23):
            uv = Point2D(self.rng.uniform(uMin, uMax), self.rng.uniform(vMin, vMax))
            interp = transform.calculateXY(uv)

            expect = self.calculateXY(uv.getX(), uv.getY())
            self.assertFloatsAlmostEqual(interp.getX(), expect[0], atol=1.0e-10)
            self.assertFloatsAlmostEqual(interp.getY(), expect[1], atol=1.0e-10)

            invInterp = inverse.calculateXY(interp)
            self.assertFloatsAlmostEqual(invInterp.getX(), uv.getX(), atol=1.0e-10)
            self.assertFloatsAlmostEqual(invInterp.getY(), uv.getY(), atol=1.0e-10)

    def testExtrapolation(self):
        """Test extrapolation outside the grid

        Our GridTransform is a linear transformation, and we're doing linear
        interpolation, so extrapolation should be very good.
        """
        uu, vv, xx, yy = self.makeData()
        transform = GridTransform(uu, vv, xx, yy)

        # It's a linear transformation, and we're doing linear interpolation, so should be safe.
        uMax = np.max(uu)
        vMax = np.max(vv)
        for _ in range(23):
            uv = Point2D(self.rng.uniform(uMax, 2*uMax), self.rng.uniform(-vMax, 0))  # Outside the grid
            interp = transform.calculateXY(uv)
            expect = self.calculateXY(uv.getX(), uv.getY())
            self.assertFloatsAlmostEqual(interp[0], expect[0], atol=1.0e-10)
            self.assertFloatsAlmostEqual(interp[1], expect[1], atol=1.0e-10)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    runTests()
