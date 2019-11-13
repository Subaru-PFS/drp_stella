from types import SimpleNamespace
import numpy as np

__all__ = ["getIndices", "calculateCentroid", "calculateSecondMoments"]


def getIndices(bbox):
    """Return x and y indices for an image, given the bounding box

    Parameters
    ----------
    bbox : `lsst.geom.Box2I`
        Bounding box for image.

    Returns
    -------
    xx, yy : `numpy.ndarray` of `int`
        Indices for the pixels in the image in x and y.
    """
    return (np.arange(bbox.getMinX(), bbox.getMaxX() + 1, dtype=float)[np.newaxis, :],
            np.arange(bbox.getMinY(), bbox.getMaxY() + 1, dtype=float)[:, np.newaxis])


def calculateCentroid(image):
    """Calculate centroid for an image

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image on which to calculate centroid.

    Returns
    -------
    centroid : `types.SimpleNamespace`
        Centroid coordinates in the ``x`` and ``y`` attributes.
    """
    xx, yy = getIndices(image.getBBox())
    norm = np.sum(image.array.astype(float))
    xCen = np.sum(np.sum(image.array.astype(float), axis=0)*xx)/norm
    yCen = np.sum(np.sum(image.array.astype(float), axis=1)*yy.T)/norm
    return SimpleNamespace(x=xCen, y=yCen)


def calculateSecondMoments(image, centroid=None):
    """Calculate second moments for an image

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image on which to calculate centroid.
    centroid : `types.SimpleNamespace`, optional
        Object with ``x`` and ``y`` attributes with the centroids.

    Returns
    -------
    moments : `types.SimpleNamespace`
        Second moments in the ``xx``, ``yy`` and ``xy`` attributes.
    """
    if centroid is None:
        centroid = calculateCentroid(image)
    norm = np.sum(image.array.astype(float))
    xx, yy = getIndices(image.getBBox())
    xWidth = np.sum(np.sum(image.array.astype(float), axis=0)*(xx - centroid.x)**2)/norm
    yWidth = np.sum(np.sum(image.array.astype(float), axis=1)*(yy.T - centroid.y)**2)/norm
    xyWidth = np.sum(image.array.astype(float)*(xx - centroid.x)*(yy - centroid.y))/norm
    return SimpleNamespace(xx=xWidth, yy=yWidth, xy=xyWidth)
